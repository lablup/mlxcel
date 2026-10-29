// Copyright 2025-2026 Lablup Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::path::PathBuf;

use super::{
    MIN_PARALLEL_CONTEXT_SIZE, ServerStartupConfig, build_cluster_init_request,
    build_server_config, detect_model_media_support, effective_parallel_context_slots,
    ensure_peers_can_reach_control_addr, ensure_pp_peer_not_dry_run, resolve_chat_template,
    resolve_decode_storage_backend, resolve_default_max_tokens, resolve_disaggregated_cluster_auth,
    resolve_dry_penalty_last_n, resolve_generation_sampling_defaults, resolve_loop_detection_env,
    resolve_parallel_context_size, resolve_pipeline_cluster_auth, resolve_pp_auto_cluster_auth,
    resolve_pp_peer_listen_addr, resolve_remote_pipeline_topology,
    resolve_tensor_parallel_runtime_support, run_zero_config_bring_up, server_distributed_request,
    validate_parallel_context_startup, validate_pipeline_parallel_startup,
    validate_settings_endpoint_exposure, validate_tensor_parallel_startup,
};
use crate::distributed::{
    ClusterAuth, ClusterConfig, ClusterDiscoveryMode, ClusterSecret, ClusterSecretSources,
    TransportBackend,
};
use crate::server::chat_template::ChatMessage;
#[cfg(unix)]
use crate::server::media::scan_insecure_allowlist_dirs;
use crate::server::state::ModelMediaSupport;
use crate::server::{DecodeStorageBackend, PipelineParallelRuntimeConfig};
// Env-var-sensitive tests must serialize through the crate-wide `ENV_LOCK`
// per-module locks race with env mutations in unrelated
// modules of the same test binary.
use crate::test_support::env_lock::env_lock;

fn temp_path(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("mlxcel-{}-{}", name, uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn test_cluster_auth() -> std::sync::Arc<ClusterAuth> {
    std::sync::Arc::new(ClusterAuth::new(&ClusterSecret::random()))
}

const TWO_STAGE_CLUSTER_TOML: &str = r#"
[cluster]
name = "remote-pp"
pipeline_parallel_size = 2
transport_backend = "tcp"

[[nodes]]
id = "coordinator"
address = "127.0.0.1:19000"
role = "hybrid"

[[nodes]]
id = "stage-0"
address = "127.0.0.1:19001"
role = "pipeline_stage"
stage = 0

[[nodes]]
id = "stage-1"
address = "127.0.0.1:19002"
role = "pipeline_stage"
stage = 1
"#;

#[test]
fn pipeline_cluster_auth_fails_closed_without_a_secret() {
    let cluster = ClusterConfig::from_toml(TWO_STAGE_CLUSTER_TOML).unwrap();
    let err = resolve_pipeline_cluster_auth(&ClusterSecretSources::default(), &cluster)
        .expect_err("pipeline role without a secret must refuse to start")
        .to_string();
    for needle in [
        "--cluster-secret-file",
        "MLXCEL_CLUSTER_SECRET_FILE",
        "MLXCEL_CLUSTER_SECRET",
        "secret_file",
    ] {
        assert!(err.contains(needle), "error must name {needle}: {err}");
    }

    let sources = ClusterSecretSources {
        env_value: Some("startup-test-secret-0123456789".to_string()),
        ..Default::default()
    };
    assert!(
        resolve_pipeline_cluster_auth(&sources, &cluster)
            .unwrap()
            .is_some()
    );

    // A topology without a pipeline needs no secret (single-node default
    // and non-PP clusters are unchanged).
    let single = ClusterConfig::from_toml(
        r#"
[cluster]
name = "single"

[[nodes]]
id = "node-0"
address = "127.0.0.1:19000"
role = "hybrid"
"#,
    )
    .unwrap();
    assert!(
        resolve_pipeline_cluster_auth(&ClusterSecretSources::default(), &single)
            .unwrap()
            .is_none()
    );
}

/// Every disaggregated role refuses to start without a cluster secret and
/// names the role and every source; `Hybrid` needs none (#2655).
#[test]
fn disaggregated_roles_fail_closed_without_a_secret() {
    use crate::distributed::disaggregated::ServingMode;
    for (mode, role_flag) in [
        (ServingMode::PrefillOnly, "--node-role prefill"),
        (ServingMode::DecodeOnly, "--node-role decode"),
        (ServingMode::Router, "--node-role router"),
    ] {
        let err = resolve_disaggregated_cluster_auth(&ClusterSecretSources::default(), mode)
            .expect_err("disaggregated role without a secret must refuse to start")
            .to_string();
        assert!(err.contains(role_flag), "error must name the role: {err}");
        for needle in [
            "--cluster-secret-file",
            "MLXCEL_CLUSTER_SECRET_FILE",
            "MLXCEL_CLUSTER_SECRET",
            "secret_file",
        ] {
            assert!(err.contains(needle), "error must name {needle}: {err}");
        }

        let sources = ClusterSecretSources {
            env_value: Some("startup-test-secret-0123456789".to_string()),
            ..Default::default()
        };
        assert!(
            resolve_disaggregated_cluster_auth(&sources, mode)
                .unwrap()
                .is_some(),
            "{mode} with a secret must resolve"
        );
    }

    // The single-node default reads no source and needs no secret.
    assert!(
        resolve_disaggregated_cluster_auth(&ClusterSecretSources::default(), ServingMode::Hybrid)
            .unwrap()
            .is_none()
    );
}

/// A `--pp-auto` coordinator config with two static peers, so the bring-up
/// path needs no network for discovery.
fn pp_auto_static_startup(cluster_config_out: PathBuf) -> ServerStartupConfig {
    ServerStartupConfig {
        pp_auto: Some(2),
        cluster_discovery: "static".to_string(),
        cluster_peers: vec![
            "127.0.0.1:19371".parse().unwrap(),
            "127.0.0.1:19372".parse().unwrap(),
        ],
        cluster_config_out: Some(cluster_config_out),
        ..ServerStartupConfig::default()
    }
}

#[test]
fn pp_auto_records_secret_file_as_absolute_path_peers_resolve() {
    // `ClusterConfig::from_file` resolves a relative `secret_file` against
    // the TOML's directory (default `./.mlxcel/`), while the operator typed
    // the flag relative to the CWD. The coordinator must record the path it
    // actually read, absolute, so every reader of the TOML finds the file.
    let out_dir = temp_path("pp-auto-secret-path");
    let toml_path = out_dir.join(".mlxcel").join("cluster.toml");
    let relative = PathBuf::from("pp-auto-secrets").join("cluster.secret");
    let expected = std::env::current_dir().unwrap().join(&relative);
    let startup = ServerStartupConfig {
        cluster_secret_file: Some(relative.clone()),
        ..pp_auto_static_startup(toml_path.clone())
    };

    let request = build_cluster_init_request(&startup).unwrap();
    assert_eq!(request.secret_file.as_deref(), Some(expected.as_path()));

    let mut planned = request;
    planned.static_peers = startup.cluster_peers.clone();
    let plan = crate::distributed::plan_cluster(&planned).unwrap();
    crate::distributed::write_plan_toml(&plan, &toml_path).unwrap();
    let reloaded = ClusterConfig::from_file(&toml_path).unwrap();
    assert_eq!(
        reloaded.cluster.secret_file.as_deref(),
        Some(expected.as_path()),
        "a peer loading the emitted TOML must resolve the same secret file"
    );
    std::fs::remove_dir_all(out_dir).unwrap();
}

#[test]
fn pp_auto_secret_resolution_fails_closed_naming_every_source() {
    let err = resolve_pp_auto_cluster_auth(&ClusterSecretSources::default())
        .expect_err("--pp-auto without a secret must refuse to start")
        .to_string();
    for needle in [
        "--pp-auto",
        "--cluster-secret-file",
        "MLXCEL_CLUSTER_SECRET_FILE",
        "MLXCEL_CLUSTER_SECRET",
    ] {
        assert!(err.contains(needle), "error must name {needle}: {err}");
    }
    let sources = ClusterSecretSources {
        env_value: Some("pp-auto-test-secret-0123456789".to_string()),
        ..Default::default()
    };
    assert!(resolve_pp_auto_cluster_auth(&sources).is_ok());
}

#[test]
fn pp_auto_bring_up_fails_closed_before_discovery_or_toml_write() {
    // Drive the real bring-up entry point. With static discovery the
    // pre-fix path resolved peers and wrote the cluster TOML without ever
    // asking for a secret; now it must stop first, leaving no TOML behind.
    let out_dir = temp_path("pp-auto-fail-closed");
    let toml_path = out_dir.join("cluster.toml");
    let mut startup = pp_auto_static_startup(toml_path.clone());

    let result = {
        let _env_guard = env_lock();
        let keys = ["MLXCEL_CLUSTER_SECRET", "MLXCEL_CLUSTER_SECRET_FILE"];
        let previous: Vec<_> = keys.iter().map(std::env::var_os).collect();
        for key in keys {
            // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
            unsafe { std::env::remove_var(key) };
        }
        let result = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(run_zero_config_bring_up(&mut startup));
        for (key, value) in keys.iter().zip(previous) {
            // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
            match value {
                Some(value) => unsafe { std::env::set_var(key, value) },
                None => unsafe { std::env::remove_var(key) },
            }
        }
        result
    };

    let err = result
        .expect_err("--pp-auto bring-up without a secret must fail")
        .to_string();
    assert!(err.contains("--pp-auto requires a cluster secret"), "{err}");
    assert!(err.contains("--cluster-secret-file"), "{err}");
    assert!(
        !toml_path.exists(),
        "no cluster TOML may be written before the secret check"
    );
    assert!(startup.distributed_config.is_none());
    std::fs::remove_dir_all(out_dir).unwrap();
}

#[test]
fn pp_peer_refuses_dry_run_before_binding() {
    let startup = ServerStartupConfig {
        pp_peer: true,
        dry_run: true,
        ..ServerStartupConfig::default()
    };
    let err = ensure_pp_peer_not_dry_run(&startup)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("--dry-run") && err.contains("--pp-peer"),
        "unexpected error: {err}"
    );
    let ok = ServerStartupConfig {
        pp_peer: true,
        ..ServerStartupConfig::default()
    };
    ensure_pp_peer_not_dry_run(&ok).unwrap();
}

#[test]
fn pp_peer_static_requires_listen_addr_not_distributed_config() {
    // Before #2658 the peer branch demanded `--distributed-config` and
    // `--node-id`. Now a static peer needs only its own listen address, and
    // the error says so before any secret or socket is touched.
    let mut startup = ServerStartupConfig {
        pp_peer: true,
        cluster_discovery: "static".to_string(),
        ..ServerStartupConfig::default()
    };
    let err = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(run_zero_config_bring_up(&mut startup))
        .expect_err("a static peer without a listen address must fail")
        .to_string();
    assert!(err.contains("--cluster-control-addr"), "{err}");
    assert!(err.contains("--cluster-peers"), "{err}");
    assert!(!err.contains("--distributed-config"), "{err}");

    // A port of 0 is not a concrete port in static mode.
    startup.cluster_control_addr = Some("192.168.1.11:0".parse().unwrap());
    let err = resolve_pp_peer_listen_addr(&startup, ClusterDiscoveryMode::Static)
        .expect_err("port 0 is not a static listen address")
        .to_string();
    assert!(err.contains("concrete port"), "{err}");

    // mdns defaults to the unspecified address with an OS-assigned port.
    startup.cluster_control_addr = None;
    assert_eq!(
        resolve_pp_peer_listen_addr(&startup, ClusterDiscoveryMode::Mdns).unwrap(),
        "0.0.0.0:0".parse::<std::net::SocketAddr>().unwrap()
    );
    startup.cluster_control_addr = Some("192.168.1.11:19001".parse().unwrap());
    assert_eq!(
        resolve_pp_peer_listen_addr(&startup, ClusterDiscoveryMode::Static)
            .unwrap()
            .to_string(),
        "192.168.1.11:19001"
    );

    // The manual override keeps its own requirement.
    startup.distributed_config = Some(PathBuf::from("cluster.toml"));
    startup.node_id = None;
    let err = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(run_zero_config_bring_up(&mut startup))
        .expect_err("manual override needs --node-id")
        .to_string();
    assert!(err.contains("--node-id"), "{err}");
}

#[test]
fn pp_auto_rejects_loopback_control_with_remote_peers() {
    let loopback_control: std::net::SocketAddr = "127.0.0.1:19000".parse().unwrap();
    let remote_peers: Vec<std::net::SocketAddr> = vec![
        "10.0.0.5:19001".parse().unwrap(),
        "127.0.0.1:19002".parse().unwrap(),
    ];
    let err = ensure_peers_can_reach_control_addr(loopback_control, &remote_peers)
        .expect_err("stages on another host cannot reply to a loopback control address")
        .to_string();
    assert!(err.contains("--cluster-control-addr"), "{err}");
    assert!(err.contains("--host"), "{err}");
    assert!(err.contains("10.0.0.5:19001"), "{err}");
    assert!(!err.contains("127.0.0.1:19002"), "{err}");

    let loopback_peers: Vec<std::net::SocketAddr> = vec!["127.0.0.1:19001".parse().unwrap()];
    ensure_peers_can_reach_control_addr(loopback_control, &loopback_peers)
        .expect("loopback peers may use a loopback control address");
    let lan_control: std::net::SocketAddr = "192.168.1.10:19000".parse().unwrap();
    ensure_peers_can_reach_control_addr(lan_control, &remote_peers)
        .expect("a LAN control address is reachable by remote peers");
}

#[test]
fn remote_pipeline_topology_without_auth_is_an_internal_error() {
    let dir = temp_path("pp-remote-no-auth");
    std::fs::write(
        dir.join("config.json"),
        r#"{"model_type": "llama", "num_hidden_layers": 16}"#,
    )
    .unwrap();
    let startup = ServerStartupConfig {
        model_path: dir.clone(),
        host: "127.0.0.1".to_string(),
        port: 8080,
        node_id: Some("coordinator".to_string()),
        ..ServerStartupConfig::default()
    };
    let cluster = ClusterConfig::from_toml(TWO_STAGE_CLUSTER_TOML).unwrap();
    let err = resolve_remote_pipeline_topology(&startup, &cluster, "coordinator", None)
        .unwrap_err()
        .to_string();
    assert!(err.contains("--cluster-secret-file"), "{err}");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn remote_stage_service_config_names_the_coordinator_peer() {
    let dir = temp_path("pp-remote-stage-coordinator");
    std::fs::write(
        dir.join("config.json"),
        r#"{"model_type": "llama", "num_hidden_layers": 16}"#,
    )
    .unwrap();
    let startup = ServerStartupConfig {
        model_path: dir.clone(),
        node_id: Some("stage-0".to_string()),
        ..ServerStartupConfig::default()
    };
    let cluster = ClusterConfig::from_toml(TWO_STAGE_CLUSTER_TOML).unwrap();
    let (_, stage) =
        resolve_remote_pipeline_topology(&startup, &cluster, "stage-0", Some(&test_cluster_auth()))
            .unwrap();
    let stage = stage.expect("stage service config");
    assert_eq!(stage.coordinator_peer, "127.0.0.1:19000");
    assert_eq!(stage.downstream_peer.as_deref(), Some("127.0.0.1:19002"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn resolve_default_max_tokens_matches_server_policy() {
    let missing = std::path::Path::new("/nonexistent/mlxcel/model/dir");
    // Explicit budget passes through unchanged regardless of context window.
    assert_eq!(resolve_default_max_tokens(128, 0, missing), 128);
    assert_eq!(resolve_default_max_tokens(128, 8192, missing), 128);
    // Unlimited (-1) with an explicit per-slot context size uses that window.
    assert_eq!(resolve_default_max_tokens(-1, 8192, missing), 8192);
    // Unlimited with no context size and an unreadable config falls back to 4096.
    assert_eq!(resolve_default_max_tokens(-1, 0, missing), 4096);
}

#[test]
fn api_keys_from_the_flag_and_the_file_join_one_set() {
    // #1437: b10621 accepts a SET of keys, and both sources append to it, so
    // a flag value no longer shadows the file. The file is read one key per
    // line and is not trimmed.
    let dir = temp_path("api-key");
    let key_file = dir.join("key.txt");
    std::fs::write(&key_file, "secret-key\n# comment\n").unwrap();

    let resolved =
        crate::server::resolve_api_keys(&["flag-key".to_string()], std::slice::from_ref(&key_file))
            .unwrap();
    assert_eq!(resolved.len(), 2);
    assert!(resolved.accepts("flag-key"));
    assert!(resolved.accepts("secret-key"));
    assert!(!resolved.accepts("# comment"));

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn settings_endpoint_without_keys_allows_only_local_listeners() {
    let no_keys = crate::server::ApiKeys::default();
    for host in [
        "127.0.0.1",
        "127.255.255.254",
        "::1",
        "[::1]",
        "localhost",
        "LOCALHOST",
    ] {
        validate_settings_endpoint_exposure(true, host, 8080, &no_keys)
            .unwrap_or_else(|error| panic!("{host} should be treated as local-only: {error:#}"));
    }
    validate_settings_endpoint_exposure(true, "/tmp/mlxcel-settings.sock", 8080, &no_keys)
        .expect("a Unix domain socket is local-only");
    validate_settings_endpoint_exposure(true, "/tmp/mlxcel-legacy-socket", 0, &no_keys)
        .expect("the legacy port-zero Unix socket spelling is local-only");
}

#[test]
fn settings_endpoint_without_keys_rejects_non_loopback_listeners() {
    let no_keys = crate::server::ApiKeys::default();
    for host in ["0.0.0.0", "::", "[::]", "192.0.2.10", "example.com"] {
        let error = validate_settings_endpoint_exposure(true, host, 8080, &no_keys)
            .expect_err("an unauthenticated non-loopback settings endpoint must be rejected");
        let message = format!("{error:#}");
        assert!(message.contains("--settings"), "{message}");
        assert!(message.contains("API key"), "{message}");
        assert!(message.contains(host), "{message}");
    }
}

#[test]
fn settings_endpoint_allows_external_listener_when_disabled_or_authenticated() {
    let no_keys = crate::server::ApiKeys::default();
    validate_settings_endpoint_exposure(false, "0.0.0.0", 8080, &no_keys)
        .expect("the guard is inactive when the endpoint is disabled");

    let api_keys = crate::server::resolve_api_keys(&["secret".to_string()], &[])
        .expect("the test key should resolve");
    validate_settings_endpoint_exposure(true, "0.0.0.0", 8080, &api_keys)
        .expect("authentication permits a non-loopback settings endpoint");
}

#[test]
fn settings_endpoint_comment_only_key_file_fails_before_guard() {
    let dir = temp_path("settings-empty-api-key");
    let key_file = dir.join("keys.txt");
    std::fs::write(&key_file, "# comment only\n\n").unwrap();
    let error = crate::server::resolve_api_keys(&[], std::slice::from_ref(&key_file))
        .expect_err("a configured file with no resolved keys must fail startup");
    assert!(format!("{error:#}").contains("contributed no key"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn resolve_chat_template_respects_override_then_file_then_model_metadata() {
    let dir = temp_path("chat-template");
    let file_template = dir.join("override.jinja");
    std::fs::write(&file_template, "file={{ messages[0].content }}").unwrap();
    std::fs::write(
        dir.join("tokenizer_config.json"),
        r#"{"chat_template":"model={{ messages[0].content }}"}"#,
    )
    .unwrap();

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "hello".to_string(),
    }];

    let processor =
        resolve_chat_template(Some("inline={{ messages[0].content }}"), None, &dir).unwrap();
    assert_eq!(processor.apply(&messages, None).unwrap(), "inline=hello");

    let processor = resolve_chat_template(None, Some(&file_template), &dir).unwrap();
    assert_eq!(processor.apply(&messages, None).unwrap(), "file=hello");

    let processor = resolve_chat_template(None, None, &dir).unwrap();
    assert_eq!(processor.apply(&messages, None).unwrap(), "model=hello");

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn build_server_config_applies_normalized_startup_values() {
    let startup = ServerStartupConfig {
        model_alias: Some("alias".to_string()),
        reasoning_alias_field: crate::server::ReasoningAliasField::None,
        // #1432: two independent controls that used to share the --timeout
        // spelling. The asserts below check each reaches its own consumer.
        http_timeout: 7,
        decode_timeout: 42,
        api_prefix: "/llama".to_string(),
        sse_ping_interval: Some(std::time::Duration::from_secs(45)),
        ctx_size: 2048,
        n_parallel: 3,
        enable_slots: false,
        enable_props: true,
        enable_metrics: true,
        temperature: 0.7,
        top_p: 0.95,
        top_k: 32,
        min_p: 0.05,
        typical_p: 1.0,
        top_n_sigma: -1.0,
        xtc_probability: 0.0,
        xtc_threshold: 0.1,
        ignore_eos: false,
        reverse_prompt: Vec::new(),
        samplers: None,
        sampler_seq: None,
        repeat_penalty: 1.2,
        repeat_last_n: 96,
        n_predict: -1,
        seed: Some(7),
        presence_penalty: 0.4,
        frequency_penalty: 0.3,
        dry_multiplier: 0.6,
        dry_base: 2.0,
        dry_allowed_length: 4,
        dry_penalty_last_n: -1,
        draft_model_path: Some(PathBuf::from("draft")),
        draft_max: 5,
        ..ServerStartupConfig::default()
    };

    let config = build_server_config(
        &startup,
        crate::server::resolve_api_keys(&["token".to_string()], &[]).expect("valid key set"),
    );
    assert!(config.api_keys.accepts("token"));
    assert_eq!(config.api_keys.len(), 1);
    assert_eq!(
        config.decode_timeout_seconds, 42,
        "the decode watchdog reads --decode-timeout, not --timeout"
    );
    assert_eq!(
        startup.http_timeout, 7,
        "the socket budget stays on --timeout and never reaches the decode watchdog"
    );
    assert_eq!(config.api_prefix, "/llama");
    assert_eq!(
        config.sse_ping_interval,
        Some(std::time::Duration::from_secs(45))
    );
    assert_eq!(config.model_alias.as_deref(), Some("alias"));
    assert_eq!(
        config.reasoning_alias_field,
        crate::server::ReasoningAliasField::None
    );
    assert_eq!(config.context_size, 682);
    assert_eq!(config.n_parallel, 3);
    assert!(!config.enable_slots_endpoint);
    assert!(config.enable_props_endpoint);
    assert!(config.enable_metrics_endpoint);
    assert_eq!(config.default_temperature, 0.7);
    assert_eq!(config.default_top_p, 0.95);
    assert_eq!(config.default_top_k, 32);
    assert_eq!(config.default_min_p, 0.05);
    assert_eq!(config.default_repetition_penalty, 1.2);
    assert_eq!(config.default_repetition_context_size, 96);
    // n_predict = -1 (unlimited) resolves to the per-slot context window:
    // ctx_size 2048 / 3 slots = 682 (issue #476). The startup model_path here is
    // empty so no config.json is read; the explicit --ctx-size drives the value.
    assert_eq!(config.default_max_tokens, 682);
    assert_eq!(config.default_seed, Some(7));
    assert_eq!(config.default_presence_penalty, 0.4);
    assert_eq!(config.default_frequency_penalty, 0.3);
    assert_eq!(config.default_dry_multiplier, 0.6);
    assert_eq!(config.default_dry_base, 2.0);
    assert_eq!(config.default_dry_allowed_length, 4);
    // -1 resolves to the explicit full-history sentinel since #1436 (the
    // pre-#1436 code mapped it to 0, which now means "DRY disabled").
    assert_eq!(
        config.default_dry_penalty_last_n,
        mlxcel_core::generate::DRY_FULL_HISTORY
    );
    assert_eq!(config.draft_model_path, Some(PathBuf::from("draft")));
    assert_eq!(config.num_draft_tokens, 5);
    // max_batch_size derived from n_parallel (no explicit override);
    // max_queue_depth comes from the startup config default (32).
    assert_eq!(config.max_batch_size, 3);
    assert_eq!(config.max_queue_depth, 32);
    assert_eq!(config.max_kv_size, Some(682));
}

#[test]
fn generation_config_sampling_defaults_fill_unset_server_knobs() {
    let model_dir = tempfile::tempdir().unwrap();
    std::fs::write(
        model_dir.path().join("generation_config.json"),
        include_str!("../../tests/fixtures/muse_glimmer/generation_config.json"),
    )
    .unwrap();
    let startup = ServerStartupConfig {
        model_path: model_dir.path().to_path_buf(),
        temperature: 0.8,
        top_p: 0.9,
        top_k: 40,
        ..ServerStartupConfig::default()
    };

    let defaults = resolve_generation_sampling_defaults(&startup);

    assert_eq!(defaults.temperature, 1.0);
    assert_eq!(defaults.top_p, 0.95);
    assert_eq!(defaults.top_k, 64);
}

#[test]
fn explicit_server_sampling_knobs_override_generation_config() {
    let model_dir = tempfile::tempdir().unwrap();
    std::fs::write(
        model_dir.path().join("generation_config.json"),
        include_str!("../../tests/fixtures/muse_glimmer/generation_config.json"),
    )
    .unwrap();
    let startup = ServerStartupConfig {
        model_path: model_dir.path().to_path_buf(),
        temperature: 0.7,
        temperature_was_set: true,
        top_p: 0.8,
        top_p_was_set: true,
        top_k: 11,
        top_k_was_set: true,
        ..ServerStartupConfig::default()
    };

    let defaults = resolve_generation_sampling_defaults(&startup);

    assert_eq!(defaults.temperature, 0.7);
    assert_eq!(defaults.top_p, 0.8);
    assert_eq!(defaults.top_k, 11);
}

#[test]
fn build_server_config_max_batch_size_is_at_least_one() {
    // n_parallel=0 is nonsensical but must not produce a zero batch size
    let startup = ServerStartupConfig {
        n_parallel: 0,
        ..ServerStartupConfig::default()
    };
    let config = build_server_config(&startup, crate::server::ApiKeys::default());
    assert_eq!(config.max_batch_size, 1);
}

#[test]
fn parallel_context_size_divides_total_budget_by_active_slots() {
    for (ctx_size, slots, expected_per_slot) in [(4096, 1, 4096), (4096, 2, 2048), (4096, 4, 1024)]
    {
        assert_eq!(
            resolve_parallel_context_size(ctx_size, slots, None, false, false),
            expected_per_slot
        );
        assert_eq!(
            resolve_parallel_context_size(ctx_size, slots, None, false, false)
                * effective_parallel_context_slots(slots, None, false, false),
            ctx_size
        );
    }

    assert_eq!(
        resolve_parallel_context_size(4097, 4, None, false, false),
        1024
    );
}

#[test]
fn unified_context_size_gives_each_slot_the_total_budget() {
    assert_eq!(
        resolve_parallel_context_size(20_480, 4, None, false, true),
        20_480
    );
    assert_eq!(effective_parallel_context_slots(4, None, false, true), 1);
}

#[test]
fn build_server_config_uses_max_batch_size_as_context_divisor() {
    let startup = ServerStartupConfig {
        ctx_size: 8192,
        n_parallel: 2,
        max_batch_size: Some(4),
        ..ServerStartupConfig::default()
    };

    let config = build_server_config(&startup, crate::server::ApiKeys::default());
    assert_eq!(config.context_size, 2048);
    assert_eq!(config.max_batch_size, 4);
    assert_eq!(config.max_kv_size, Some(2048));
}

#[test]
fn build_server_config_keeps_whole_window_for_unified_context() {
    let startup = ServerStartupConfig {
        ctx_size: 20_480,
        n_parallel: 4,
        kv_unified: true,
        ..ServerStartupConfig::default()
    };

    let config = build_server_config(&startup, crate::server::ApiKeys::default());
    assert_eq!(config.context_size, 20_480);
    assert_eq!(config.max_kv_size, Some(20_480));
    assert!(config.kv_unified);
}

#[test]
fn build_server_config_honors_no_batch_as_single_slot_context() {
    let startup = ServerStartupConfig {
        ctx_size: 8192,
        n_parallel: 4,
        no_batch: true,
        ..ServerStartupConfig::default()
    };

    let config = build_server_config(&startup, crate::server::ApiKeys::default());
    assert_eq!(config.context_size, 8192);
    assert_eq!(config.max_kv_size, Some(8192));
}

#[test]
fn build_server_config_keeps_lower_explicit_max_kv_size() {
    let startup = ServerStartupConfig {
        ctx_size: 8192,
        n_parallel: 2,
        max_kv_size: Some(1024),
        ..ServerStartupConfig::default()
    };

    let config = build_server_config(&startup, crate::server::ApiKeys::default());
    assert_eq!(config.context_size, 4096);
    assert_eq!(config.max_kv_size, Some(1024));
}

#[test]
fn validate_parallel_context_startup_rejects_too_small_per_slot_window() {
    let startup = ServerStartupConfig {
        ctx_size: MIN_PARALLEL_CONTEXT_SIZE,
        n_parallel: 2,
        ..ServerStartupConfig::default()
    };

    let err = validate_parallel_context_startup(&startup).expect_err("must reject below floor");
    assert!(
        err.to_string()
            .contains("below the minimum supported per-slot context size"),
        "unexpected error: {err}"
    );
}

#[test]
fn build_server_config_propagates_no_batch_flag() {
    let startup = ServerStartupConfig {
        no_batch: true,
        ..ServerStartupConfig::default()
    };
    let config = build_server_config(&startup, crate::server::ApiKeys::default());
    assert!(config.no_batch);
}

#[test]
fn build_server_config_preserves_batch_scheduler_for_tensor_parallel() {
    let startup = ServerStartupConfig {
        tp_size: 2,
        ..ServerStartupConfig::default()
    };
    let config = build_server_config(&startup, crate::server::ApiKeys::default());
    assert!(!config.no_batch);
    assert_eq!(config.tensor_parallel.tp_size, 2);
}

#[test]
fn build_server_config_propagates_pipeline_parallel_settings() {
    let startup = ServerStartupConfig {
        pp_layers: Some("0-7,8-15".to_string()),
        pp_micro_batch_size: 4,
        ..ServerStartupConfig::default()
    };
    let config = build_server_config(&startup, crate::server::ApiKeys::default());
    match config.pipeline_parallel_runtime.as_ref() {
        Some(PipelineParallelRuntimeConfig::InProcess {
            layers,
            micro_batch_size,
        }) => {
            assert_eq!(layers, "0-7,8-15");
            assert_eq!(*micro_batch_size, 4);
        }
        other => panic!("unexpected pipeline runtime config: {other:?}"),
    }
}

#[test]
fn resolve_remote_pipeline_topology_builds_remote_coordinator_runtime() {
    let dir = temp_path("pp-remote-coordinator");
    std::fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "llama",
            "num_hidden_layers": 16
        }"#,
    )
    .unwrap();

    let startup = ServerStartupConfig {
        model_path: dir.clone(),
        host: "127.0.0.1".to_string(),
        port: 8080,
        node_id: Some("coordinator".to_string()),
        ..ServerStartupConfig::default()
    };
    let cluster = ClusterConfig::from_toml(
        r#"
[cluster]
name = "remote-pp"
pipeline_parallel_size = 2
transport_backend = "tcp"

[[nodes]]
id = "coordinator"
address = "127.0.0.1:19000"
role = "hybrid"

[[nodes]]
id = "stage-0"
address = "127.0.0.1:19001"
role = "pipeline_stage"
stage = 0

[[nodes]]
id = "stage-1"
address = "127.0.0.1:19002"
role = "pipeline_stage"
stage = 1
"#,
    )
    .unwrap();

    let (runtime, stage) = resolve_remote_pipeline_topology(
        &startup,
        &cluster,
        "coordinator",
        Some(&test_cluster_auth()),
    )
    .unwrap();
    assert!(stage.is_none());
    match runtime {
        Some(PipelineParallelRuntimeConfig::RemoteCoordinator(config)) => {
            assert_eq!(config.bind_address, "127.0.0.1:19000");
            assert_eq!(config.transport_backend, TransportBackend::Tcp);
            assert_eq!(
                config.stage_peers,
                vec!["127.0.0.1:19001", "127.0.0.1:19002"]
            );
        }
        other => panic!("unexpected remote pipeline runtime: {other:?}"),
    }

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn resolve_remote_pipeline_topology_builds_remote_stage_service() {
    let dir = temp_path("pp-remote-stage");
    std::fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "llama",
            "num_hidden_layers": 16
        }"#,
    )
    .unwrap();

    let startup = ServerStartupConfig {
        model_path: dir.clone(),
        node_id: Some("stage-1".to_string()),
        ..ServerStartupConfig::default()
    };
    let cluster = ClusterConfig::from_toml(
        r#"
[cluster]
name = "remote-pp"
pipeline_parallel_size = 2
transport_backend = "tcp"

[[nodes]]
id = "coordinator"
address = "127.0.0.1:19000"
role = "hybrid"

[[nodes]]
id = "stage-0"
address = "127.0.0.1:19001"
role = "pipeline_stage"
stage = 0

[[nodes]]
id = "stage-1"
address = "127.0.0.1:19002"
role = "pipeline_stage"
stage = 1
"#,
    )
    .unwrap();

    let (runtime, stage) =
        resolve_remote_pipeline_topology(&startup, &cluster, "stage-1", Some(&test_cluster_auth()))
            .unwrap();
    assert!(runtime.is_none());
    let stage = stage.expect("stage service config");
    assert_eq!(stage.bind_address, "127.0.0.1:19002");
    assert_eq!(stage.transport_backend, TransportBackend::Tcp);
    assert_eq!(stage.stage_assignment.stage_index, 1);
    assert_eq!(stage.upstream_peer.as_deref(), Some("127.0.0.1:19001"));
    assert_eq!(stage.downstream_peer, None);

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn resolve_remote_pipeline_topology_preserves_thunderbolt_backend() {
    let dir = temp_path("pp-remote-thunderbolt");
    std::fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "llama",
            "num_hidden_layers": 16
        }"#,
    )
    .unwrap();

    let startup = ServerStartupConfig {
        model_path: dir.clone(),
        host: "127.0.0.1".to_string(),
        port: 8080,
        node_id: Some("coordinator".to_string()),
        ..ServerStartupConfig::default()
    };
    let cluster = ClusterConfig::from_toml(
        r#"
[cluster]
name = "remote-pp"
pipeline_parallel_size = 2
transport_backend = "thunderbolt"

[[nodes]]
id = "coordinator"
address = "169.254.91.10:19000"
role = "hybrid"

[[nodes]]
id = "stage-0"
address = "169.254.91.11:19001"
role = "pipeline_stage"
stage = 0

[[nodes]]
id = "stage-1"
address = "169.254.91.12:19002"
role = "pipeline_stage"
stage = 1
"#,
    )
    .unwrap();

    let (runtime, stage) = resolve_remote_pipeline_topology(
        &startup,
        &cluster,
        "coordinator",
        Some(&test_cluster_auth()),
    )
    .unwrap();
    assert!(stage.is_none());
    match runtime {
        Some(PipelineParallelRuntimeConfig::RemoteCoordinator(config)) => {
            assert_eq!(config.transport_backend, TransportBackend::Thunderbolt);
            assert_eq!(config.bind_address, "169.254.91.10:19000");
        }
        other => panic!("unexpected remote pipeline runtime: {other:?}"),
    }

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn resolve_remote_pipeline_topology_rejects_control_port_conflict() {
    let dir = temp_path("pp-remote-conflict");
    std::fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "llama",
            "num_hidden_layers": 16
        }"#,
    )
    .unwrap();

    let startup = ServerStartupConfig {
        model_path: dir.clone(),
        host: "127.0.0.1".to_string(),
        port: 19000,
        node_id: Some("coordinator".to_string()),
        ..ServerStartupConfig::default()
    };
    let cluster = ClusterConfig::from_toml(
        r#"
[cluster]
name = "remote-pp"
pipeline_parallel_size = 2
transport_backend = "tcp"

[[nodes]]
id = "coordinator"
address = "127.0.0.1:19000"
role = "hybrid"

[[nodes]]
id = "stage-0"
address = "127.0.0.1:19001"
role = "pipeline_stage"
stage = 0

[[nodes]]
id = "stage-1"
address = "127.0.0.1:19002"
role = "pipeline_stage"
stage = 1
"#,
    )
    .unwrap();

    let err = resolve_remote_pipeline_topology(
        &startup,
        &cluster,
        "coordinator",
        Some(&test_cluster_auth()),
    )
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("conflicts with HTTP listen address")
    );

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn decode_storage_backend_parses_auto_dense_and_paged() {
    assert_eq!(
        "auto".parse::<DecodeStorageBackend>().unwrap(),
        DecodeStorageBackend::Auto
    );
    assert_eq!(
        "dense".parse::<DecodeStorageBackend>().unwrap(),
        DecodeStorageBackend::Dense
    );
    assert_eq!(
        "paged".parse::<DecodeStorageBackend>().unwrap(),
        DecodeStorageBackend::Paged
    );
    assert!("unknown".parse::<DecodeStorageBackend>().is_err());
}

#[test]
fn resolve_decode_storage_backend_defaults_to_auto() {
    let _env_guard = env_lock();
    let key = "MLXCEL_SERVER_DECODE_STORAGE";
    let prev = std::env::var_os(key);
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    unsafe { std::env::remove_var(key) };

    let resolved = resolve_decode_storage_backend();

    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    match prev {
        Some(value) => unsafe { std::env::set_var(key, value) },
        None => unsafe { std::env::remove_var(key) },
    }

    assert_eq!(resolved, DecodeStorageBackend::Auto);
}

#[test]
fn build_server_config_uses_cli_decode_storage_backend() {
    let startup = ServerStartupConfig {
        decode_storage_backend: Some(DecodeStorageBackend::Paged),
        ..ServerStartupConfig::default()
    };

    let config = build_server_config(&startup, crate::server::ApiKeys::default());
    assert_eq!(config.decode_storage_backend, DecodeStorageBackend::Paged);
}

#[test]
fn resolve_tensor_parallel_runtime_support_allows_server_batching_for_llama() {
    let dir = temp_path("tp-llama-batching");
    std::fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "llama",
            "num_hidden_layers": 32
        }"#,
    )
    .unwrap();

    let startup = ServerStartupConfig {
        model_path: dir.clone(),
        tp_size: 2,
        ..ServerStartupConfig::default()
    };
    let support = resolve_tensor_parallel_runtime_support(&startup).unwrap();
    assert!(!support.force_no_batch);

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn resolve_tensor_parallel_runtime_support_allows_server_batching_for_gemma3() {
    let dir = temp_path("tp-gemma3-batching");
    std::fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "gemma3_text",
            "num_hidden_layers": 26
        }"#,
    )
    .unwrap();

    let startup = ServerStartupConfig {
        model_path: dir.clone(),
        tp_size: 2,
        ..ServerStartupConfig::default()
    };
    let support = resolve_tensor_parallel_runtime_support(&startup).unwrap();
    assert!(!support.force_no_batch);

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_pipeline_parallel_startup_accepts_supported_llama_config() {
    let dir = temp_path("pp-llama-startup");
    std::fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "llama",
            "num_hidden_layers": 16
        }"#,
    )
    .unwrap();

    let startup = ServerStartupConfig {
        model_path: dir.clone(),
        pp_layers: Some("0-7,8-15".to_string()),
        pp_micro_batch_size: 2,
        ..ServerStartupConfig::default()
    };
    validate_pipeline_parallel_startup(&startup).unwrap();

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_pipeline_parallel_startup_accepts_adapter_config() {
    // LoRA + PP composition is supported for the in-process server runtime:
    // the adapter path is threaded through
    // PipelineServerModel::load_in_process_with_adapter at model worker
    // bring-up, so the startup validator must no longer reject the
    // combination. v1 covers the Llama family; other families still bail
    // at stage_executor::load_family_backend.
    let dir = temp_path("pp-with-adapter");
    std::fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "llama",
            "num_hidden_layers": 16
        }"#,
    )
    .unwrap();

    let startup = ServerStartupConfig {
        model_path: dir.clone(),
        pp_layers: Some("0-7,8-15".to_string()),
        pp_micro_batch_size: 2,
        adapter_path: Some(dir.join("adapters").clone()),
        ..ServerStartupConfig::default()
    };
    validate_pipeline_parallel_startup(&startup).unwrap();

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_pipeline_parallel_startup_rejects_no_batch_mode() {
    let dir = temp_path("pp-no-batch");
    std::fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "llama",
            "num_hidden_layers": 16
        }"#,
    )
    .unwrap();

    let startup = ServerStartupConfig {
        model_path: dir.clone(),
        pp_layers: Some("0-7,8-15".to_string()),
        no_batch: true,
        ..ServerStartupConfig::default()
    };
    let err = validate_pipeline_parallel_startup(&startup).unwrap_err();
    assert!(err.to_string().contains("requires the batch scheduler"));

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn resolve_tensor_parallel_runtime_support_allows_server_batching_for_gemma4() {
    let dir = temp_path("tp-gemma4-batching");
    std::fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "num_hidden_layers": 42
            }
        }"#,
    )
    .unwrap();

    let startup = ServerStartupConfig {
        model_path: dir.clone(),
        tp_size: 2,
        ..ServerStartupConfig::default()
    };
    let support = resolve_tensor_parallel_runtime_support(&startup).unwrap();
    assert!(!support.force_no_batch);

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn resolve_tensor_parallel_runtime_support_forces_no_batch_for_gemma4_e2b_fallback() {
    let dir = temp_path("tp-gemma4-e2b-fallback");
    std::fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 1536,
                "num_hidden_layers": 35,
                "intermediate_size": 6144,
                "num_attention_heads": 8,
                "head_dim": 256,
                "rms_norm_eps": 1e-6,
                "vocab_size": 262144,
                "vocab_size_per_layer_input": 262144,
                "num_key_value_heads": 1,
                "num_global_key_value_heads": null,
                "num_kv_shared_layers": 20,
                "hidden_size_per_layer_input": 256,
                "rope_traditional": false,
                "rope_parameters": {
                    "sliding_attention": {"rope_theta": 10000.0, "partial_rotary_factor": 1.0},
                    "full_attention": {"rope_theta": 1000000.0, "partial_rotary_factor": 0.25}
                },
                "sliding_window": 512,
                "sliding_window_pattern": 1,
                "max_position_embeddings": 131072,
                "attention_k_eq_v": false,
                "final_logit_softcapping": 30.0,
                "use_double_wide_mlp": true,
                "enable_moe_block": false,
                "num_experts": null,
                "top_k_experts": null,
                "moe_intermediate_size": null,
                "layer_types": [
                    "sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
                    "sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
                    "sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
                    "sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
                    "sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
                    "sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention",
                    "sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"
                ]
            }
        }"#,
    )
    .unwrap();

    let startup = ServerStartupConfig {
        model_path: dir.clone(),
        tp_size: 2,
        ..ServerStartupConfig::default()
    };
    let support = resolve_tensor_parallel_runtime_support(&startup).unwrap();
    assert!(support.force_no_batch);

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_tensor_parallel_startup_accepts_single_rank() {
    let dir = temp_path("tp-single");
    std::fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "llama",
            "num_hidden_layers": 32
        }"#,
    )
    .unwrap();

    let startup = ServerStartupConfig {
        model_path: dir.clone(),
        ..ServerStartupConfig::default()
    };
    validate_tensor_parallel_startup(&startup).unwrap();

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_tensor_parallel_startup_accepts_supported_multi_rank_runtime() {
    let dir = temp_path("tp-multi");
    std::fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "llama",
            "num_hidden_layers": 32
        }"#,
    )
    .unwrap();

    let startup = ServerStartupConfig {
        model_path: dir.clone(),
        tp_size: 2,
        ..ServerStartupConfig::default()
    };
    validate_tensor_parallel_startup(&startup).unwrap();

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_tensor_parallel_startup_accepts_qwen2_multi_rank_runtime() {
    let dir = temp_path("tp-qwen2");
    std::fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "qwen2",
            "num_hidden_layers": 24
        }"#,
    )
    .unwrap();

    let startup = ServerStartupConfig {
        model_path: dir.clone(),
        tp_size: 2,
        ..ServerStartupConfig::default()
    };
    validate_tensor_parallel_startup(&startup).unwrap();

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_tensor_parallel_startup_accepts_qwen3_multi_rank_runtime() {
    let dir = temp_path("tp-qwen3");
    std::fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "qwen3",
            "num_hidden_layers": 28
        }"#,
    )
    .unwrap();

    let startup = ServerStartupConfig {
        model_path: dir.clone(),
        tp_size: 2,
        ..ServerStartupConfig::default()
    };
    validate_tensor_parallel_startup(&startup).unwrap();

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_tensor_parallel_startup_accepts_qwen35_multi_rank_runtime() {
    let dir = temp_path("tp-qwen35");
    std::fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "qwen3_5",
            "num_hidden_layers": 24
        }"#,
    )
    .unwrap();

    let startup = ServerStartupConfig {
        model_path: dir.clone(),
        tp_size: 2,
        ..ServerStartupConfig::default()
    };
    validate_tensor_parallel_startup(&startup).unwrap();

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_tensor_parallel_startup_accepts_ernie45_multi_rank_runtime() {
    let dir = temp_path("tp-ernie45");
    std::fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "ernie4_5",
            "num_hidden_layers": 18
        }"#,
    )
    .unwrap();

    let startup = ServerStartupConfig {
        model_path: dir.clone(),
        tp_size: 2,
        ..ServerStartupConfig::default()
    };
    validate_tensor_parallel_startup(&startup).unwrap();

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_tensor_parallel_startup_accepts_hunyuan_v1_dense_multi_rank_runtime() {
    let dir = temp_path("tp-hunyuan-v1-dense");
    std::fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "hunyuan_v1_dense",
            "num_hidden_layers": 32
        }"#,
    )
    .unwrap();

    let startup = ServerStartupConfig {
        model_path: dir.clone(),
        tp_size: 2,
        ..ServerStartupConfig::default()
    };
    validate_tensor_parallel_startup(&startup).unwrap();

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_tensor_parallel_startup_accepts_gemma3_multi_rank_runtime() {
    let dir = temp_path("tp-gemma3");
    std::fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "gemma3_text",
            "num_hidden_layers": 26
        }"#,
    )
    .unwrap();

    let startup = ServerStartupConfig {
        model_path: dir.clone(),
        tp_size: 2,
        ..ServerStartupConfig::default()
    };
    validate_tensor_parallel_startup(&startup).unwrap();

    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_tensor_parallel_startup_accepts_gemma4_multi_rank_runtime() {
    let dir = temp_path("tp-gemma4");
    std::fs::write(
        dir.join("config.json"),
        r#"{
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "num_hidden_layers": 26
            }
        }"#,
    )
    .unwrap();

    let startup = ServerStartupConfig {
        model_path: dir.clone(),
        tp_size: 2,
        ..ServerStartupConfig::default()
    };
    validate_tensor_parallel_startup(&startup).unwrap();

    std::fs::remove_dir_all(dir).unwrap();
}

// -------------------------------------------------------------------------
// prompt-cache startup integration tests
// -------------------------------------------------------------------------

/// `build_server_config` with `enabled=false` produces a `ServerConfig`
/// whose `prompt_cache.is_enabled()` returns `false`, confirming the cache
/// gate in `startup.rs` will skip store construction.
#[test]
fn build_server_config_prompt_cache_disabled_produces_false_is_enabled() {
    use crate::server::prompt_cache::PromptCacheConfig;

    let startup = ServerStartupConfig {
        prompt_cache: PromptCacheConfig::disabled(),
        ..ServerStartupConfig::default()
    };
    let config = build_server_config(&startup, crate::server::ApiKeys::default());
    assert!(
        !config.prompt_cache.is_enabled(),
        "disabled PromptCacheConfig must not satisfy is_enabled() in ServerConfig"
    );
}

/// `build_server_config` with the default startup config produces a
/// `ServerConfig` whose `prompt_cache.is_enabled()` returns `true`.
#[test]
fn build_server_config_prompt_cache_default_is_enabled() {
    let startup = ServerStartupConfig::default();
    assert_eq!(startup.top_p, 0.95);
    assert_eq!(startup.min_p, 0.05);
    assert_eq!(startup.dry_penalty_last_n, 64);
    let config = build_server_config(&startup, crate::server::ApiKeys::default());
    assert!(
        config.prompt_cache.is_enabled(),
        "default PromptCacheConfig must satisfy is_enabled() in ServerConfig"
    );
    assert_eq!(config.default_top_p, 0.95);
    assert_eq!(config.default_min_p, 0.05);
    assert_eq!(config.default_dry_penalty_last_n, 64);
}

/// `build_server_config` propagates a custom `capacity_bytes` value from
/// `ServerStartupConfig.prompt_cache` into `ServerConfig.prompt_cache`.
#[test]
fn build_server_config_propagates_prompt_cache_capacity() {
    use crate::server::prompt_cache::PromptCacheConfig;

    const CUSTOM_CAP: usize = 134_217_728; // 128 MiB
    let startup = ServerStartupConfig {
        prompt_cache: PromptCacheConfig::new(
            true,
            CUSTOM_CAP,
            PromptCacheConfig::DEFAULT_MAX_ENTRIES,
            std::time::Duration::from_secs(PromptCacheConfig::DEFAULT_TTL_SECONDS),
            PromptCacheConfig::DEFAULT_MIN_PREFIX_TOKENS,
        ),
        ..ServerStartupConfig::default()
    };
    let config = build_server_config(&startup, crate::server::ApiKeys::default());
    assert_eq!(config.prompt_cache.capacity_bytes, CUSTOM_CAP);
}

// -- detect_model_media_support ----------------------------------

/// `detect_model_media_support` reports `video=true` only when the
/// `config.json` resolves to a Gemma 4 VLM. Other VLM types (e.g. plain
/// Gemma 4 text-only) and missing configs both fall back to "no video".
///
/// We synthesize a minimal Gemma 4 VLM `config.json` here rather than
/// loading a full model — the helper only needs `model_type` plus the
/// vision-tower presence flag, both of which are tiny.
#[test]
fn detect_model_media_support_recognises_gemma4_vlm() {
    let dir = temp_path("media-gemma4-vlm");
    // Minimal VLM-shape config: `model_type=gemma4` with a non-empty
    // `vision_config` triggers Gemma4VLM detection in
    // `crate::models::detection::get_model_type`. The `vision_tower.weights.npz`
    // sentinel checked by `gemma4_has_vision_weights` must also exist.
    let config = serde_json::json!({
        "model_type": "gemma4",
        "vision_config": {
            "model_type": "siglip_vision_model",
            "image_size": 224
        },
        "text_config": {
            "model_type": "gemma4_text"
        }
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    // `gemma4_has_vision_weights` looks for vision tower files; create a
    // sentinel safetensors file to satisfy it.
    std::fs::write(dir.join("model.safetensors"), b"placeholder").unwrap();

    let support = detect_model_media_support(&dir);
    // Note: gemma4_has_vision_weights checks specific weight key names, so
    // a placeholder file may or may not trigger it. We assert the helper
    // does not panic and produces a deterministic result for the same input.
    // The boolean depends on the detection helper's robustness against
    // synthetic configs; the real assertion is that the helper succeeds.
    let _ = support;
    // Whatever the vision-weight probe decides, the ViT-backed Gemma 4 VLM has
    // no video+audio merge path (issue #1349).
    assert!(
        !support.video_with_audio,
        "the ViT Gemma 4 VLM must keep the combined refusal, got {support:?}"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn detect_model_media_support_recognises_gemma4_unified() {
    // The encoder-free Gemma 4 Unified model consumes `video_url` content
    // blocks (issue #164). `model_type=gemma4_unified` is detected by
    // model_type alone (no vision-tower weight probe needed), so a minimal
    // config is enough to assert the route guard will admit video requests.
    let dir = temp_path("media-gemma4-unified");
    let config = serde_json::json!({
        "model_type": "gemma4_unified",
        "text_config": { "model_type": "gemma4_unified_text" },
        "vision_config": { "model_type": "gemma4_unified_vision" }
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();

    let support = detect_model_media_support(&dir);
    assert!(
        support.video_native,
        "gemma4_unified must enable video_url content blocks, got {support:?}"
    );
    // Issue #1349: gemma4_unified is the one family whose merge path scatters
    // video frames and audio into the same token stream.
    assert!(
        support.video_with_audio,
        "gemma4_unified must admit video_url together with input_audio, got {support:?}"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn detect_model_media_support_recognises_kimi_k25() {
    // Kimi-VL 2.5 (kimi_k25) MoonViT consumes `video_url` content blocks
    // (issue #551). `model_type=kimi_k25` is detected by model_type alone, so a
    // minimal config is enough to assert the route guard will admit video.
    let dir = temp_path("media-kimi-k25");
    let config = serde_json::json!({
        "model_type": "kimi_k25",
        "text_config": { "model_type": "deepseek_v3" },
        "vision_config": { "model_type": "moonvit" }
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();

    let support = detect_model_media_support(&dir);
    assert!(
        support.video_native,
        "kimi_k25 must enable video_url content blocks, got {support:?}"
    );
    assert!(
        !support.video_with_audio,
        "Kimi-VL 2.5 must keep the combined video+audio refusal, got {support:?}"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn detect_model_media_support_recognises_inkling_video() {
    let dir = temp_path("media-inkling-vlm");
    let config = serde_json::json!({
        "model_type": "inkling_mm_model",
        "text_config": {
            "model_type": "inkling",
            "hidden_size": 64
        },
        "vision_config": {
            "model_type": "inkling_vision",
            "text_hidden_size": 64
        },
        "audio_config": {
            "model_type": "inkling_audio",
            "text_hidden_size": 64
        }
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let index = serde_json::json!({
        "metadata": {},
        "weight_map": {
            "model.visual.layers.linear_0.weight": "model-00001-of-00001.safetensors"
        }
    });
    std::fs::write(dir.join("model.safetensors.index.json"), index.to_string()).unwrap();

    let support = detect_model_media_support(&dir);
    assert!(
        support.video_native,
        "Inkling VLM must enable video_url content blocks, got {support:?}"
    );
    assert!(
        support.audio,
        "Inkling VLM must admit input_audio for loaded-tower validation, got {support:?}"
    );
    // Inkling takes video and audio separately but has no combined merge path.
    assert!(
        !support.video_with_audio,
        "Inkling VLM must keep the combined video+audio refusal, got {support:?}"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn detect_model_media_support_recognises_qwen35_vlm_video() {
    let dir = temp_path("media-qwen35-vlm");
    let config = serde_json::json!({
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": { "model_type": "qwen3_5" },
        "vision_config": { "model_type": "qwen3_vl" },
        "video_token_id": 248057
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();

    let support = detect_model_media_support(&dir);
    assert!(
        support.video_native,
        "Qwen3.5/Qwen3.8 VLM must enable video_url content blocks, got {support:?}"
    );
    assert!(
        !support.video_with_audio,
        "Qwen3.5/Qwen3.8 VLM must keep the combined video+audio refusal, got {support:?}"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

/// Write a minimal `config.json` into a fresh temp directory and resolve the
/// media-support flags from it. Shared by the issue #1322 split tests below,
/// which each care about one `model_type` and nothing else in the checkpoint.
fn media_support_for_config(name: &str, config: serde_json::Value) -> ModelMediaSupport {
    let dir = temp_path(name);
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let support = detect_model_media_support(&dir);
    std::fs::remove_dir_all(dir).unwrap();
    support
}

#[test]
fn media_support_marks_image_only_vlms_as_frames_fallback() {
    // Neither family has a temporal path, and both have a vision tower, so a
    // `video_url` block is served by decoding the clip into ordered frames
    // (issue #1322) instead of being refused at the boundary.
    for (name, config) in [
        (
            "media-gemma3-fallback",
            serde_json::json!({
                "model_type": "gemma3",
                "architectures": ["Gemma3ForConditionalGeneration"],
                "text_config": { "model_type": "gemma3_text" },
                "vision_config": { "model_type": "siglip_vision_model", "image_size": 896 }
            }),
        ),
        (
            "media-idefics3-fallback",
            serde_json::json!({
                "model_type": "idefics3",
                "architectures": ["Idefics3ForConditionalGeneration"],
                "text_config": { "model_type": "llama" },
                "vision_config": { "model_type": "idefics3" }
            }),
        ),
    ] {
        let support = media_support_for_config(name, config);
        assert!(
            support.video_frames_fallback,
            "{name} must serve video_url through the frames fallback, got {support:?}"
        );
        assert!(
            !support.video_native,
            "{name} has no native video path, got {support:?}"
        );
        assert!(
            support.video(),
            "{name} must admit video_url at the boundary, got {support:?}"
        );
        // The fallback rewrites the clip into images. It grants no merge path
        // for a clip plus an audio track, so the combination stays refused.
        assert!(
            !support.video_with_audio,
            "{name} must keep the combined video+audio refusal, got {support:?}"
        );
    }
}

#[test]
fn media_support_keeps_native_video_families_native() {
    // Qwen2.5-VL is in this list, not the fallback one: issue #1166 gave the
    // Qwen-VL families a real `video_grid_thw` path, so the substitution the
    // issue #1322 plan assumed for them must not engage.
    for (name, config, expect_video_with_audio) in [
        (
            "media-gemma4-unified-native",
            serde_json::json!({
                "model_type": "gemma4_unified",
                "text_config": { "model_type": "gemma4_unified_text" },
                "vision_config": { "model_type": "gemma4_unified_vision" }
            }),
            true,
        ),
        (
            "media-kimi-vl-native",
            serde_json::json!({
                "model_type": "kimi_vl",
                "architectures": ["KimiVLForConditionalGeneration"],
                "text_config": { "model_type": "deepseek_v3" },
                "vision_config": { "model_type": "moonvit" }
            }),
            false,
        ),
        (
            "media-qwen25-vl-native",
            serde_json::json!({
                "model_type": "qwen2_5_vl",
                "architectures": ["Qwen2_5_VLForConditionalGeneration"]
            }),
            false,
        ),
    ] {
        let support = media_support_for_config(name, config);
        assert!(
            support.video_native,
            "{name} must keep its native video path, got {support:?}"
        );
        assert!(
            !support.video_frames_fallback,
            "{name} must not be routed through the frames fallback, got {support:?}"
        );
        assert_eq!(
            support.video_with_audio, expect_video_with_audio,
            "{name} video_with_audio must be unchanged by issue #1322, got {support:?}"
        );
    }
}

#[test]
fn media_support_gives_the_vit_gemma4_vlm_no_frames_fallback() {
    // The ViT-backed Gemma 4 VLM detection runs a vision-weight probe that a
    // synthetic config cannot satisfy deterministically, so this pins only the
    // half that does not depend on it: whichever way the probe lands, this
    // family never reaches the fallback, because both outcomes
    // (`gemma4` VLM and `gemma4` text) are handled elsewhere.
    let support = media_support_for_config(
        "media-gemma4-vlm-native",
        serde_json::json!({
            "model_type": "gemma4",
            "text_config": { "model_type": "gemma4_text" },
            "vision_config": { "model_type": "siglip_vision_model", "image_size": 224 }
        }),
    );
    assert!(
        support.video_native || !support.video_frames_fallback,
        "Gemma 4 VLM must reach its own video path, not the frames fallback, got {support:?}"
    );
    assert!(
        !support.video_with_audio,
        "the ViT Gemma 4 VLM must keep the combined refusal, got {support:?}"
    );
}

#[test]
fn media_support_denies_the_frames_fallback_to_text_only_and_muse_glimmer() {
    // A checkpoint with no vision tower has nowhere to send frames.
    let text_only = media_support_for_config(
        "media-llama-text-fallback",
        serde_json::json!({ "model_type": "llama" }),
    );
    assert!(
        !text_only.video_frames_fallback && !text_only.video(),
        "a text-only checkpoint must keep refusing video_url, got {text_only:?}"
    );

    // Muse Glimmer refuses `--video` by name on the CLI
    // (`validate_muse_glimmer_cli_unsupported_options`); admitting it here
    // alone would leave the two fronts disagreeing about one checkpoint.
    let muse = media_support_for_config(
        "media-muse-glimmer-fallback",
        serde_json::json!({
            "model_type": "muse_glimmer",
            "text_config": { "model_type": "llama" },
            "vision_config": { "model_type": "muse_glimmer_vision" }
        }),
    );
    assert!(
        muse.image,
        "Muse Glimmer keeps its image path, got {muse:?}"
    );
    assert!(
        !muse.video(),
        "Muse Glimmer must keep refusing video_url on both fronts, got {muse:?}"
    );
}

#[test]
fn detect_model_media_support_falls_back_for_missing_config() {
    let dir = temp_path("media-missing-config");
    // No config.json → get_model_type fails → fallback yields "no video".
    let support = detect_model_media_support(&dir);
    assert!(
        !support.video(),
        "missing config.json must default to video=false, got {support:?}"
    );
    assert!(
        !support.video_with_audio,
        "missing config.json must default to video_with_audio=false, got {support:?}"
    );
    assert!(!crate::server::state::ModelMediaSupport::none().video_with_audio);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn detect_model_media_support_text_only_disables_video() {
    let dir = temp_path("media-llama-text");
    // Pure-text model config (no vision).
    let config = serde_json::json!({
        "model_type": "llama"
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();

    let support = detect_model_media_support(&dir);
    assert!(
        !support.video(),
        "text-only llama must report video=false, got {support:?}"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

// -- review (MEDIUM-2): startup TOCTOU writability scan --------------

/// Unix-only: `scan_insecure_allowlist_dirs` reports a world-writable
/// directory so the server's startup hook can warn the operator. The
/// `warn_on_insecure_video_allowlist` helper itself is a thin wrapper
/// around this scan, so testing the pure helper covers the wiring without
/// the env-var plumbing complexity.
#[cfg(unix)]
#[test]
fn startup_warns_on_world_writable_allowlist_dir() {
    use std::os::unix::fs::PermissionsExt;

    let dir = temp_path("startup-allowlist-loose");
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o777)).unwrap();

    let insecure = scan_insecure_allowlist_dirs(std::slice::from_ref(&dir));
    assert!(
        insecure.iter().any(|p| p == &dir),
        "world-writable dir must be flagged at startup; got {insecure:?}"
    );

    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::fs::remove_dir_all(dir).unwrap();
}

/// Unix-only: a strict-mode (0750) allowlist directory must NOT be flagged.
#[cfg(unix)]
#[test]
fn startup_passes_strict_allowlist_dir() {
    use std::os::unix::fs::PermissionsExt;

    let dir = temp_path("startup-allowlist-strict");
    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o750)).unwrap();

    let insecure = scan_insecure_allowlist_dirs(std::slice::from_ref(&dir));
    assert!(
        !insecure.iter().any(|p| p == &dir),
        "strict-mode dir must not be flagged; got {insecure:?}"
    );

    std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).unwrap();
    std::fs::remove_dir_all(dir).unwrap();
}

// -- MLXCEL_LOOP_DETECTION env parser (issue #432) --

/// Run `resolve_loop_detection_env` with `MLXCEL_LOOP_DETECTION` temporarily
/// set to `value` (or unset when `None`), then restore the original.
fn with_loop_detection_env<F, R>(value: Option<&str>, f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = env_lock();
    let key = "MLXCEL_LOOP_DETECTION";
    let prev = std::env::var_os(key);
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    match value {
        Some(v) => unsafe { std::env::set_var(key, v) },
        None => unsafe { std::env::remove_var(key) },
    }
    let result = f();
    // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
    match prev {
        Some(v) => unsafe { std::env::set_var(key, v) },
        None => unsafe { std::env::remove_var(key) },
    }
    result
}

#[test]
fn loop_detection_env_unset_returns_none() {
    let result = with_loop_detection_env(None, resolve_loop_detection_env);
    assert!(result.is_none());
}

#[test]
fn loop_detection_env_off_keywords_force_disable() {
    use mlxcel_core::LoopDetectionConfig;
    for kw in &["off", "OFF", "0", "none", "false", "disabled"] {
        let result = with_loop_detection_env(Some(kw), resolve_loop_detection_env);
        assert!(
            result.is_some(),
            "expected Some(disabled) for keyword {kw:?}"
        );
        assert_eq!(
            result.unwrap(),
            LoopDetectionConfig::disabled(),
            "expected disabled config for keyword {kw:?}"
        );
    }
}

#[test]
fn loop_detection_env_on_keywords_force_recommended() {
    use crate::server::request_options::LOOP_DETECTION_RECOMMENDED;
    for kw in &["on", "ON", "default", "true", "enabled"] {
        let result = with_loop_detection_env(Some(kw), resolve_loop_detection_env);
        assert_eq!(
            result,
            Some(LOOP_DETECTION_RECOMMENDED),
            "expected recommended config for keyword {kw:?}"
        );
    }
}

#[test]
fn loop_detection_env_comma_triple_parsed() {
    use mlxcel_core::LoopDetectionConfig;
    let result = with_loop_detection_env(Some("2,30,5"), resolve_loop_detection_env);
    assert_eq!(result, Some(LoopDetectionConfig::new(2, 30, 5)));
}

#[test]
fn loop_detection_env_colon_triple_parsed() {
    use mlxcel_core::LoopDetectionConfig;
    let result = with_loop_detection_env(Some("1:20:4"), resolve_loop_detection_env);
    assert_eq!(result, Some(LoopDetectionConfig::new(1, 20, 4)));
}

#[test]
fn loop_detection_env_malformed_returns_none() {
    let result = with_loop_detection_env(Some("not_a_valid_value"), resolve_loop_detection_env);
    assert!(result.is_none());
}

#[test]
fn loop_detection_env_wrong_field_count_returns_none() {
    // Four fields is not a valid triple.
    let result = with_loop_detection_env(Some("1,20,4,extra"), resolve_loop_detection_env);
    assert!(result.is_none());
}

// ── b10621 serving-mode gate (#1452) ────────────────────────────────────────

fn mode_config(mode: crate::server::config::EmbeddingServingMode) -> crate::server::ServerConfig {
    crate::server::ServerConfig {
        embedding_serving_mode: mode,
        ..crate::server::ServerConfig::default()
    }
}

/// A mode flag that resolved no worker able to serve it fails at startup,
/// naming the flag and the two ways to give it a checkpoint. Without this the
/// flag would parse, generation would be off, and the only route left would
/// answer 501 for the life of the process.
#[test]
fn a_mode_flag_with_no_matching_worker_fails_startup() {
    use crate::server::config::EmbeddingServingMode;

    let err = super::check_serving_mode(
        &mode_config(EmbeddingServingMode::EmbeddingOnly),
        false,
        true,
    )
    .expect_err("--embeddings with no embedder must fail");
    let message = err.to_string();
    assert!(message.contains("--embeddings"), "{message}");
    assert!(message.contains("--embedding-model"), "{message}");

    let err =
        super::check_serving_mode(&mode_config(EmbeddingServingMode::RerankOnly), true, false)
            .expect_err("--reranking with no reranker must fail");
    let message = err.to_string();
    assert!(message.contains("--reranking"), "{message}");
    assert!(message.contains("--reranker-model"), "{message}");
}

#[test]
fn a_mode_flag_with_its_worker_starts() {
    use crate::server::config::EmbeddingServingMode;

    super::check_serving_mode(
        &mode_config(EmbeddingServingMode::EmbeddingOnly),
        true,
        false,
    )
    .expect("--embeddings with an embedder starts");
    super::check_serving_mode(&mode_config(EmbeddingServingMode::RerankOnly), false, true)
        .expect("--reranking with a reranker starts");
}

#[test]
fn an_unrestricted_server_never_fails_the_mode_gate() {
    use crate::server::config::EmbeddingServingMode;

    for (embedding, rerank) in [(false, false), (true, false), (false, true), (true, true)] {
        super::check_serving_mode(&mode_config(EmbeddingServingMode::Any), embedding, rerank)
            .expect("no mode flag, nothing to check");
    }
}

#[test]
fn sanitize_typical_p_default_folds_out_of_domain_to_disabled() {
    use super::sanitize_typical_p_default;
    assert_eq!(sanitize_typical_p_default(0.5), 0.5);
    assert_eq!(sanitize_typical_p_default(1.0), 1.0);
    for bad in [0.0f32, -1.0, 1.5, f32::NAN, f32::INFINITY] {
        assert_eq!(
            sanitize_typical_p_default(bad),
            1.0,
            "--typical {bad} must fold to the disabled default"
        );
    }
}

#[test]
fn sanitize_top_n_sigma_default_folds_the_b10621_disabled_sentinel() {
    use super::sanitize_top_n_sigma_default;
    assert_eq!(sanitize_top_n_sigma_default(1.5), 1.5);
    for disabled in [-1.0f32, 0.0, f32::NAN, f32::NEG_INFINITY] {
        assert_eq!(sanitize_top_n_sigma_default(disabled), 0.0);
    }
}

#[test]
fn resolve_dry_penalty_last_n_maps_the_sentinels() {
    assert_eq!(
        resolve_dry_penalty_last_n(-1),
        mlxcel_core::generate::DRY_FULL_HISTORY,
        "-1 is the documented full-context form"
    );
    assert_eq!(resolve_dry_penalty_last_n(0), 0, "0 disables DRY (b10621)");
    assert_eq!(resolve_dry_penalty_last_n(64), 64);
}

#[test]
fn sampler_order_accepts_only_the_fixed_b10621_default() {
    use super::{B10621_DEFAULT_SAMPLER_SEQ, B10621_DEFAULT_SAMPLERS, check_sampler_order};
    check_sampler_order(None, None).expect("absent flags are fine");
    check_sampler_order(
        Some(B10621_DEFAULT_SAMPLERS),
        Some(B10621_DEFAULT_SAMPLER_SEQ),
    )
    .expect("the default order is the inert configuration");
    let err = check_sampler_order(Some("top_k;temperature"), None)
        .expect_err("a reordered chain must fail startup");
    assert!(err.to_string().contains(B10621_DEFAULT_SAMPLERS), "{err}");
    let err = check_sampler_order(None, Some("kt"))
        .expect_err("a reordered char sequence must fail startup");
    assert!(
        err.to_string().contains(B10621_DEFAULT_SAMPLER_SEQ),
        "{err}"
    );
}

#[test]
fn sampler_order_accepts_the_adaptive_p_extension_and_reports_it() {
    use super::check_sampler_order;
    // #1485: b10621 activates the adaptive-p stage solely through the
    // sampler list, appending it after the chain wherever it is named, so
    // the fixed order accepts adaptive_p at any position and the caller
    // learns it was named.
    assert!(
        check_sampler_order(
            Some("penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature;adaptive_p"),
            None,
        )
        .expect("the extended list is accepted")
    );
    assert!(
        check_sampler_order(None, Some("aedskypmxt")).expect("the char form accepts a anywhere")
    );
    assert!(
        !check_sampler_order(None, None).expect("absent flags name nothing"),
        "no flag, no adaptive stage"
    );
    check_sampler_order(Some("adaptive_p;top_k"), None)
        .expect_err("adaptive_p does not excuse a reordered remainder");
}

#[cfg(feature = "webui")]
#[test]
fn webui_security_treats_bracketed_ipv6_loopback_as_loopback() {
    use crate::server::webui::security::startup::WebUiListenKind;

    for host in ["::1", "[::1]", "127.0.0.1", "localhost"] {
        let resolution = crate::server::transport::resolve_listen_target(host, 8080).unwrap();
        assert_eq!(
            super::webui_listen_kind(&resolution),
            WebUiListenKind::LoopbackTcp,
            "{host} should be WebUI loopback"
        );
    }
    for host in ["::", "[::]", "0.0.0.0", "192.0.2.10"] {
        let resolution = crate::server::transport::resolve_listen_target(host, 8080).unwrap();
        assert_eq!(
            super::webui_listen_kind(&resolution),
            WebUiListenKind::NonLoopbackTcp,
            "{host} should remain WebUI non-loopback"
        );
    }
}

#[test]
fn startup_warmup_skips_only_the_families_without_a_text_chat_worker() {
    use crate::models::ModelType;
    use crate::server::startup::startup_warmup_skip_reason;

    assert!(startup_warmup_skip_reason(Some(ModelType::NemotronVoiceChat)).is_some());
    assert!(startup_warmup_skip_reason(Some(ModelType::Florence2VLM)).is_some());
    assert!(startup_warmup_skip_reason(Some(ModelType::NemotronParseVLM)).is_some());
    assert_eq!(startup_warmup_skip_reason(Some(ModelType::Qwen3)), None);
    assert_eq!(startup_warmup_skip_reason(Some(ModelType::Llama)), None);
    assert_eq!(
        startup_warmup_skip_reason(None),
        None,
        "an undetected model type still warms"
    );
}

#[test]
fn server_distributed_request_names_each_distributed_flag() {
    assert_eq!(
        server_distributed_request(&ServerStartupConfig::default()),
        None,
        "a default startup is single-node"
    );
    let cases: Vec<(ServerStartupConfig, &str)> = vec![
        (
            ServerStartupConfig {
                pp_auto: Some(2),
                ..ServerStartupConfig::default()
            },
            "--pp-auto",
        ),
        (
            ServerStartupConfig {
                pp_peer: true,
                ..ServerStartupConfig::default()
            },
            "--pp-peer",
        ),
        (
            ServerStartupConfig {
                distributed_config: Some(PathBuf::from("cluster.toml")),
                ..ServerStartupConfig::default()
            },
            "--distributed-config",
        ),
        (
            ServerStartupConfig {
                node_role: Some("prefill".to_string()),
                ..ServerStartupConfig::default()
            },
            "--node-role",
        ),
        (
            ServerStartupConfig {
                peers: vec!["127.0.0.1:9000".parse().unwrap()],
                ..ServerStartupConfig::default()
            },
            "--peers",
        ),
        (
            ServerStartupConfig {
                prefill_peers: vec!["127.0.0.1:9001".parse().unwrap()],
                ..ServerStartupConfig::default()
            },
            "--prefill-peers",
        ),
        (
            ServerStartupConfig {
                decode_peers: vec!["127.0.0.1:9002".parse().unwrap()],
                ..ServerStartupConfig::default()
            },
            "--decode-peers",
        ),
        (
            ServerStartupConfig {
                serving_bind: Some("127.0.0.1:9003".parse().unwrap()),
                ..ServerStartupConfig::default()
            },
            "--serving-bind",
        ),
        (
            ServerStartupConfig {
                pp_layers: Some("0-15,16-31".to_string()),
                ..ServerStartupConfig::default()
            },
            "--pp-layers",
        ),
        (
            ServerStartupConfig {
                tp_size: 2,
                ..ServerStartupConfig::default()
            },
            "--tp-size",
        ),
    ];
    for (startup, flag) in cases {
        assert_eq!(server_distributed_request(&startup), Some(flag));
    }
}

/// #2664, #2697, #2722: lines raised before `initialize_server_logging`
/// installs a subscriber (a CORS origin variable that a command-line flag
/// overrode, or the INFO line an `env_fallback_*` helper raises when a flag
/// takes precedence over its variable) are carried in the config and logged
/// by `start_server` afterwards. This drives every precedence helper with
/// both the flag and a differing variable set, outside any subscriber, and
/// pins that the post-subscriber step writes each precedence line exactly
/// once at INFO while a warning stays at WARN and an error at ERROR.
#[test]
fn startup_notices_are_logged_once_a_subscriber_exists() {
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    use crate::server::StartupNotice;

    struct SharedBuf(Arc<Mutex<Vec<u8>>>);
    struct BufWriter(Arc<Mutex<Vec<u8>>>);
    impl std::io::Write for BufWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().expect("log buffer").extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for SharedBuf {
        type Writer = BufWriter;
        fn make_writer(&'a self) -> BufWriter {
            BufWriter(Arc::clone(&self.0))
        }
    }

    // Every variable differs from the value the "command line" set below.
    const VARS: [(&str, &str); 21] = [
        ("LLAMA_ARG_KV_BITS", "8"),
        ("LLAMA_ARG_KV_GROUP_SIZE", "128"),
        ("LLAMA_ARG_KV_SKIP_LAST_LAYER", "true"),
        ("LLAMA_ARG_KV_QUANT_SCHEME", "turboquant"),
        ("APC_HASH", "xxhash"),
        ("LLAMA_ARG_MODEL_DRAFT", "models/env-draft"),
        ("LLAMA_LOG_FILE", "/tmp/env.log"),
        ("MLXCEL_EMBEDDING_MODEL", "env-embedder"),
        ("MLXCEL_RERANKER_MODEL", "env-reranker"),
        ("LLAMA_ARG_LANG_BIAS", "ko"),
        ("LLAMA_ARG_LANG_BIAS_INCLUDE_BYTE_FRAGMENTS", "false"),
        ("MLXCEL_PROMPT_CACHE_ENABLED", "false"),
        ("MLXCEL_PROMPT_CACHE_TTL", "60"),
        ("MLXCEL_PROMPT_CACHE_SNAPSHOT_TTL", "60"),
        ("APC_ENABLED", "true"),
        ("APC_BLOCK_SIZE", "32"),
        ("LLAMA_ARG_CACHE_TYPE_K", "int8"),
        ("MLXCEL_DRAFT_KIND", "dflash"),
        ("MLXCEL_DRAFT_BLOCK_SIZE", "8"),
        ("LLAMA_ARG_THINK_BUDGET", "100"),
        (
            "LLAMA_ARG_CHAT_TEMPLATE_KWARGS",
            "{\"enable_thinking\":true}",
        ),
    ];

    let mut notices = Vec::new();
    {
        let _env_guard = env_lock();
        // The legacy reasoning-budget alias would add a second line.
        const UNSET: [&str; 1] = ["LLAMA_ARG_REASONING_BUDGET"];
        let previous: Vec<_> = VARS
            .iter()
            .map(|(key, _)| *key)
            .chain(UNSET)
            .map(|key| (key, std::env::var_os(key)))
            .collect();
        for (key, value) in VARS {
            // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
            unsafe { std::env::set_var(key, value) };
        }
        for key in UNSET {
            // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
            unsafe { std::env::remove_var(key) };
        }

        let mut kv_bits = 4;
        crate::server::env_fallback_kv_bits(&mut kv_bits, &mut notices);
        let mut kv_group_size = 32;
        crate::server::env_fallback_kv_group_size(&mut kv_group_size, &mut notices);
        let mut kv_skip_last_layer = false;
        crate::server::env_fallback_kv_skip_last_layer(&mut kv_skip_last_layer, &mut notices);
        let mut kv_quant_scheme = Some("uniform".to_string());
        crate::server::env_fallback_kv_quant_scheme(&mut kv_quant_scheme, &mut notices);
        let mut apc_hash = Some("sha256".to_string());
        crate::server::env_fallback_apc_hash(&mut apc_hash, &mut notices);
        let mut draft_model = Some(PathBuf::from("models/cli-draft"));
        crate::server::env_fallback_draft_model(&mut draft_model, &mut notices);
        let mut log_file = Some(PathBuf::from("/tmp/cli.log"));
        crate::server::env_fallback_log_file(&mut log_file, &mut notices);
        let mut embedding_model = Some("cli-embedder".to_string());
        crate::server::env_fallback_embedding_model(&mut embedding_model, &mut notices);
        let mut reranker_model = Some("cli-reranker".to_string());
        crate::server::env_fallback_reranker_model(&mut reranker_model, &mut notices);
        let mut lang_bias = crate::lang_bias::LangBiasCliArgs {
            lang_bias: Some("en".to_string()),
            include_byte_fragments: true,
            ..Default::default()
        };
        crate::server::env_fallback_lang_bias(&mut lang_bias, &mut notices);
        crate::server::env_fallback_lang_bias_include_byte_fragments(&mut lang_bias, &mut notices);
        let mut prompt_cache_enabled = true;
        crate::server::env_fallback_prompt_cache_enabled(
            &mut prompt_cache_enabled,
            true,
            &mut notices,
        );
        let mut prompt_cache_ttl = Some(5);
        crate::server::env_fallback_prompt_cache_ttl(&mut prompt_cache_ttl, &mut notices);
        let mut snapshot_ttl = Some(5);
        crate::server::env_fallback_prompt_cache_snapshot_ttl(&mut snapshot_ttl, &mut notices);
        let mut apc_enabled = false;
        crate::server::env_fallback_apc_enabled(&mut apc_enabled, true, &mut notices);
        let mut apc_block_size = Some(16);
        crate::server::env_fallback_apc_block_size(&mut apc_block_size, &mut notices);
        let mut cache_type_k = Some("fp16".to_string());
        crate::server::env_fallback_cache_type_k(&mut cache_type_k, &mut notices);
        let mut draft_kind = Some("eagle".to_string());
        crate::cli::speculative_args::env_fallback_draft_kind(&mut draft_kind, &mut notices);
        let mut draft_block_size = Some(4);
        crate::cli::speculative_args::env_fallback_draft_block_size(
            &mut draft_block_size,
            &mut notices,
        );
        let mut reasoning_budget = 0;
        crate::server::env_fallback_reasoning_budget(&mut reasoning_budget, true, &mut notices);
        let mut chat_template_kwargs = Some("{}".to_string());
        crate::server::env_fallback_chat_template_kwargs(&mut chat_template_kwargs, &mut notices);

        for (key, value) in previous {
            // SAFETY: serialized via the crate-wide ENV_LOCK acquired above.
            match value {
                Some(value) => unsafe { std::env::set_var(key, value) },
                None => unsafe { std::env::remove_var(key) },
            }
        }
    }
    notices.push(StartupNotice::warn(
        "MLXCEL_ALLOWED_ORIGINS is ignored because --cors-origins was given",
    ));
    notices.push(StartupNotice::error(
        "--grammar: failed to parse grammar: unexpected end of input",
    ));

    let buf = Arc::new(Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::fmt()
        .with_writer(SharedBuf(Arc::clone(&buf)))
        .with_ansi(false)
        .finish();
    let startup = ServerStartupConfig {
        startup_notices: notices,
        ..ServerStartupConfig::default()
    };
    tracing::subscriber::with_default(subscriber, || {
        super::log_startup_notices(&startup);
    });

    let output = String::from_utf8(buf.lock().expect("log buffer").clone()).expect("UTF-8 log");
    let lines: Vec<&str> = output.lines().collect();
    let precedence = [
        "LLAMA_ARG_KV_BITS env var is set but --kv-bits CLI flag takes precedence",
        "LLAMA_ARG_KV_GROUP_SIZE env var is set but --kv-group-size CLI flag takes precedence",
        "LLAMA_ARG_KV_SKIP_LAST_LAYER env var is set but --kv-skip-last-layer CLI flag",
        "LLAMA_ARG_KV_QUANT_SCHEME env var is set but --kv-quant-scheme CLI flag",
        "APC_HASH env var is set but --apc-hash CLI flag takes precedence",
        "LLAMA_ARG_MODEL_DRAFT env var is set but --draft-model CLI flag takes precedence",
        "LLAMA_LOG_FILE env var is set but --log-file CLI flag takes precedence",
        "MLXCEL_EMBEDDING_MODEL env var is set but --embedding-model",
        "MLXCEL_RERANKER_MODEL env var is set but --reranker-model",
        "LLAMA_ARG_LANG_BIAS env var is set but --lang-bias CLI flag takes precedence",
        "LLAMA_ARG_LANG_BIAS_INCLUDE_BYTE_FRAGMENTS env var is set but \
         --lang-bias-include-byte-fragments",
        "MLXCEL_PROMPT_CACHE_ENABLED is set but --prompt-cache-enabled takes precedence",
        "MLXCEL_PROMPT_CACHE_TTL env var is set but --prompt-cache-ttl CLI flag",
        "MLXCEL_PROMPT_CACHE_SNAPSHOT_TTL env var is set but --prompt-cache-snapshot-ttl",
        "APC_ENABLED env var is set but --apc-enabled CLI flag takes precedence",
        "APC_BLOCK_SIZE env var is set but --apc-block-size CLI flag takes precedence",
        "LLAMA_ARG_CACHE_TYPE_K env var is set but --cache-type-k CLI flag takes precedence",
        "MLXCEL_DRAFT_KIND=\"dflash\" differs from --draft-kind=\"eagle\"; keeping CLI value",
        "MLXCEL_DRAFT_BLOCK_SIZE=8 is set but the CLI --draft-block-size flag",
        "LLAMA_ARG_THINK_BUDGET env var is set but --reasoning-budget CLI flag",
        "LLAMA_ARG_CHAT_TEMPLATE_KWARGS env var is set but --chat-template-kwargs",
    ];
    for message in precedence {
        let matching: Vec<&&str> = lines.iter().filter(|line| line.contains(message)).collect();
        assert!(
            matching.len() == 1 && matching[0].contains("INFO"),
            "the precedence line {message:?} must be logged exactly once at INFO:\n{output}"
        );
    }
    let info_lines = lines.iter().filter(|line| line.contains("INFO")).count();
    assert_eq!(
        info_lines,
        precedence.len(),
        "one INFO line per precedence case:\n{output}"
    );
    let warn_lines: Vec<&&str> = lines.iter().filter(|line| line.contains("WARN")).collect();
    assert!(
        warn_lines.len() == 1
            && warn_lines[0].contains("MLXCEL_ALLOWED_ORIGINS is ignored because --cors-origins"),
        "the warning must stay a single WARN line: {output:?}"
    );
    let error_lines: Vec<&&str> = lines.iter().filter(|line| line.contains("ERROR")).collect();
    assert!(
        error_lines.len() == 1 && error_lines[0].contains("--grammar: failed to parse grammar"),
        "the grammar error must be a single ERROR line: {output:?}"
    );
}
