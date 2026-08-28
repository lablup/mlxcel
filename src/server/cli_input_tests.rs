// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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
    DecodeStorageBackend, MAX_KV_SIZE_MIN, ServerStartupInput, env_fallback_cache_type_k,
    env_fallback_cache_type_v, env_fallback_kv_bits, env_fallback_kv_group_size,
    env_fallback_kv_quant_scheme, env_fallback_kv_skip_last_layer, resolve_compat_toggle,
    resolve_kv_cache_mode, resolve_max_kv_size, resolve_prefill_chunk_size, resolve_seed,
};
use crate::lang_bias::LangBiasCliArgs;
// Tests that mutate env vars (via `EnvGuard` or directly) must acquire the
// crate-wide `ENV_LOCK` *before* the guard so the lock outlives the
// guard's `Drop` (which calls `remove_var`). — a per-module
// lock would race with env mutations in unrelated modules of the same
// test binary.
use crate::test_support::env_lock::env_lock;

fn sample_input() -> ServerStartupInput {
    ServerStartupInput {
        chat_compat: Default::default(),
        model_path: PathBuf::from("models/foo"),
        adapter_path: Some(PathBuf::from("adapters/bar")),
        model_alias: Some("alias".to_string()),
        host: "127.0.0.1".to_string(),
        port: 8080,
        api_keys: vec!["secret".to_string()],
        api_key_files: vec![PathBuf::from("api.key")],
        n_parallel: 2,
        ctx_size: 4096,
        n_predict: 256,
        timeout: 3600,
        timeout_was_set: false,
        decode_timeout: 600,
        decode_timeout_was_set: false,
        api_prefix: String::new(),
        sse_ping_interval: 30,
        threads_http: -1,
        reuse_port: false,
        ssl_cert_file: None,
        ssl_key_file: None,
        cors_origins: "*".to_string(),
        cors_methods: "GET, POST, DELETE, OPTIONS".to_string(),
        cors_headers: "*".to_string(),
        cors_credentials: true,
        no_cors_credentials: false,
        draft_model_path: Some(PathBuf::from("models/draft")),
        draft_max: 8,
        // speculative-decoding selector flags default off
        // (None = auto-detect at dispatch time when a drafter is set).
        draft_kind: None,
        draft_block_size: None,
        max_batch_size: Some(4),
        max_queue_depth: 32,
        audio_queue_depth: 8,
        audio_request_timeout_secs: 120,
        embedding_model_path: None,
        embedding_batch_size: 16,
        embedding_max_length: None,
        embedding_queue_depth: 8,
        embedding_request_timeout_secs: 120,
        reranker_model_path: None,
        rerank_batch_size: 0,
        prefill_chunk_size: 512,
        prefill_grant_interval: None,
        batch_size: None,
        ubatch_size: None,
        enable_preemption: false,
        preemption_policy: "longest-first".to_string(),
        no_batch: false,
        max_batch_prefill: 1,
        max_batch_prefill_tokens: None,
        decode_storage_backend: None,
        chat_template: Some("{{ prompt }}".to_string()),
        chat_template_file: Some(PathBuf::from("chat.jinja")),
        slots: true,
        no_slots: false,
        props: true,
        metrics: true,
        slot_save_path: None,
        router_models_dir: None,
        models_max: 4,
        models_autoload: true,
        models_preset: None,
        tags: None,
        warmup: true,
        no_warmup: false,
        temperature: 0.8,
        temperature_was_set: false,
        top_k: 40,
        top_k_was_set: false,
        top_p: 0.9,
        top_p_was_set: false,
        min_p: 0.1,
        typical_p: 1.0,
        top_n_sigma: -1.0,
        xtc_probability: 0.0,
        xtc_threshold: 0.1,
        ignore_eos: false,
        reverse_prompt: Vec::new(),
        samplers: None,
        sampler_seq: None,
        seed: 42,
        repeat_last_n: 64,
        repeat_penalty: 1.1,
        presence_penalty: 0.2,
        frequency_penalty: 0.3,
        dry_multiplier: 0.4,
        dry_base: 1.75,
        dry_allowed_length: 2,
        dry_penalty_last_n: -1,
        dry_sequence_breakers: vec!["\n".to_string()],
        verbose: true,
        log_disable: false,
        log_file: Some(PathBuf::from("server.log")),
        log_format: crate::server::logging::LogFormatOptions::default(),
        distributed_config: None,
        node_role: None,
        node_id: None,
        peers: Vec::new(),
        prefill_peers: Vec::new(),
        decode_peers: Vec::new(),
        serving_bind: None,
        pp_layers: None,
        pp_micro_batch_size: 1,
        pp_auto: None,
        pp_peer: false,
        cluster_discovery: "static".to_string(),
        cluster_name: None,
        cluster_peers: Vec::new(),
        cluster_discovery_port: None,
        cluster_control_addr: None,
        cluster_config_out: None,
        dry_run: false,
        tp_size: 1,
        tp_moe_mode: "expert_parallel".to_string(),
        tp_embedding_mode: "replicated".to_string(),
        tp_lm_head_mode: "replicated".to_string(),
        vision_cache_size: 20,
        max_image_payload_size: crate::server::DEFAULT_MAX_IMAGE_PAYLOAD_SIZE,
        max_images_per_request: crate::server::DEFAULT_MAX_IMAGES_PER_REQUEST,
        max_image_width: crate::server::DEFAULT_MAX_IMAGE_WIDTH,
        max_image_height: crate::server::DEFAULT_MAX_IMAGE_HEIGHT,
        max_image_decode_alloc_bytes: crate::server::DEFAULT_MAX_IMAGE_DECODE_ALLOC_BYTES,
        enable_elastic_pp: false,
        elastic_pp_drain_timeout: 120,
        elastic_pp_pressure_fraction: 0.92,
        elastic_pp_cool_down: 30,
        metrics_port: None,
        debug_pp_trace: None,
        lang_bias_config: None,
        reasoning_budget: -1,
        chat_template_kwargs: None,
        // prompt-cache knobs — use defaults for the helper.
        prompt_cache_enabled: true,
        prompt_cache_capacity_bytes: None,
        prompt_cache_max_entries: None,
        prompt_cache_ttl_seconds: None,
        prompt_cache_min_prefix: None,
        prompt_cache_snapshot_capacity_bytes: None,
        prompt_cache_snapshot_max_entries: None,
        prompt_cache_snapshot_ttl_seconds: None,
        // APC knobs — the fixture keeps APC off so existing whole-prefix
        // expectations stay exact (the serve binaries default it ON).
        apc_enabled: false,
        apc_block_size: None,
        apc_num_blocks: None,
        apc_hash: None,
        // (B11): KV cache type split flags — default to None (FP16).
        cache_type_k: None,
        cache_type_v: None,
        kv_cache_mode_legacy: None,
        // continuous-batching KV quantization knobs (off by default).
        kv_bits: 0,
        kv_group_size: mlxcel_core::cache::DEFAULT_KV_GROUP_SIZE,
        kv_quant_scheme: None,
        kv_skip_last_layer: true,
        // max KV cache size (0 = unbounded, the default).
        max_kv_size: 0,
        // paged KV pool block-budget directive (None = unbounded, the default).
        kv_cache_budget: None,
        // experimental VLM prefix-cache toggle off in tests (#124 step c).
        enable_vlm_prefix_cache: false,
        // CORS allow-list unset in tests (#244): permissive default.
        allowed_origins: Vec::new(),
        // Responses API store defaults.
        responses_store_max_entries: 1024,
        responses_store_ttl_secs: 3600,
        conversation_store_max_entries: 256,
        conversation_store_ttl_secs: 3600,
        // (A4): default to None for baseline-path tests.
        #[cfg(feature = "surgery")]
        surgery_config_path: None,
        // serve-level diffusion knobs (#217 phase 3): engine defaults in tests.
        max_denoising_steps: None,
        diffusion_sampler: "entropy-bound".to_string(),
        diffusion_threshold: 0.9,
        rope: crate::cli::rope_args::RopeOverrideArgs::default(),
        cache_compat: crate::cli::cache_args::CacheCompatArgs::default(),
        infill: crate::cli::infill_args::InfillArgs::default(),
        embedding_compat: crate::cli::embedding_compat_args::EmbeddingCompatArgs::default(),
    }
}

#[test]
fn resolve_compat_toggle_honors_disable_override() {
    assert!(resolve_compat_toggle(true, false));
    assert!(!resolve_compat_toggle(true, true));
    assert!(!resolve_compat_toggle(false, false));
}

#[test]
fn resolve_seed_maps_negative_values_to_random_mode() {
    assert_eq!(resolve_seed(-1), None);
    assert_eq!(resolve_seed(7), Some(7));
}

#[test]
fn into_startup_config_normalizes_edge_only_flags() {
    let mut input = sample_input();
    input.no_slots = true;
    input.no_warmup = true;
    input.seed = -1;

    let startup = input.into_startup_config().expect("valid startup input");

    assert!(!startup.enable_slots);
    assert!(!startup.warmup);
    assert_eq!(startup.seed, None);
    assert_eq!(startup.adapter_path, Some(PathBuf::from("adapters/bar")));
    assert_eq!(
        startup.draft_model_path,
        Some(PathBuf::from("models/draft"))
    );
    assert_eq!(startup.log_file, Some(PathBuf::from("server.log")));
}

#[test]
fn into_startup_config_propagates_image_limits() {
    let mut input = sample_input();
    input.max_image_payload_size = 8192;
    input.max_images_per_request = 4;
    input.max_image_width = 4096;
    input.max_image_height = 2048;
    input.max_image_decode_alloc_bytes = 64 * 1024 * 1024;

    let startup = input.into_startup_config().expect("valid startup input");
    assert_eq!(startup.max_image_payload_size, 8192);
    assert_eq!(startup.max_images_per_request, 4);
    assert_eq!(startup.max_image_width, 4096);
    assert_eq!(startup.max_image_height, 2048);
    assert_eq!(startup.max_image_decode_alloc_bytes, 64 * 1024 * 1024);
}

#[test]
fn resolve_prefill_chunk_size_batch_size_alias_takes_effect() {
    let r = resolve_prefill_chunk_size(512, Some(1024), None);
    assert_eq!(r.prefill_chunk_size, 1024);
    assert!(!r.batch_size_conflict);
    assert!(!r.ubatch_size_provided);
}

#[test]
fn resolve_prefill_chunk_size_explicit_prefill_wins_with_conflict() {
    let r = resolve_prefill_chunk_size(256, Some(1024), None);
    assert_eq!(r.prefill_chunk_size, 256);
    assert!(r.batch_size_conflict);
}

#[test]
fn resolve_prefill_chunk_size_no_batch_size_returns_prefill() {
    let r = resolve_prefill_chunk_size(768, None, None);
    assert_eq!(r.prefill_chunk_size, 768);
    assert!(!r.batch_size_conflict);
}

#[test]
fn resolve_prefill_chunk_size_ubatch_sets_provided_flag() {
    let r = resolve_prefill_chunk_size(512, None, Some(256));
    assert!(r.ubatch_size_provided);
    assert_eq!(r.prefill_chunk_size, 512);
}

#[test]
fn resolve_prefill_chunk_size_both_same_value_no_conflict() {
    let r = resolve_prefill_chunk_size(1024, Some(1024), None);
    assert_eq!(r.prefill_chunk_size, 1024);
    assert!(!r.batch_size_conflict);
}

#[test]
fn into_startup_config_propagates_no_batch_flag() {
    let mut input = sample_input();
    input.no_batch = true;

    let startup = input.into_startup_config().expect("valid startup input");
    assert!(startup.no_batch);

    let mut input2 = sample_input();
    input2.no_batch = false;
    let startup2 = input2.into_startup_config().expect("valid startup input");
    assert!(!startup2.no_batch);
}

#[test]
fn into_startup_config_resolves_batch_size_alias() {
    let mut input = sample_input();
    input.batch_size = Some(1024);
    let startup = input.into_startup_config().expect("valid startup input");
    assert_eq!(startup.prefill_chunk_size, 1024);
    assert!(!startup.batch_size_conflict);
    assert!(!startup.ubatch_size_provided);
}

#[test]
fn into_startup_config_detects_batch_size_conflict() {
    let mut input = sample_input();
    input.prefill_chunk_size = 256;
    input.batch_size = Some(1024);
    input.ubatch_size = Some(64);
    let startup = input.into_startup_config().expect("valid startup input");
    assert_eq!(startup.prefill_chunk_size, 256);
    assert!(startup.batch_size_conflict);
    assert!(startup.ubatch_size_provided);
}

#[test]
fn into_startup_config_propagates_pp_layers() {
    let mut input = sample_input();
    input.pp_layers = Some("0-15,16-31".to_string());
    let startup = input.into_startup_config().expect("valid startup input");
    assert_eq!(startup.pp_layers, Some("0-15,16-31".to_string()));
}

#[test]
fn into_startup_config_pp_layers_none_by_default() {
    let startup = sample_input()
        .into_startup_config()
        .expect("valid startup input");
    assert_eq!(startup.pp_layers, None);
}

#[test]
fn into_startup_config_propagates_pp_micro_batch_size() {
    let mut input = sample_input();
    input.pp_micro_batch_size = 4;
    let startup = input.into_startup_config().expect("valid startup input");
    assert_eq!(startup.pp_micro_batch_size, 4);
}

// -------------------------------------------------------------------------
// chat_template_kwargs normalization
// -------------------------------------------------------------------------

#[test]
fn into_startup_config_accepts_valid_chat_template_kwargs_json() {
    let mut input = sample_input();
    input.chat_template_kwargs = Some(r#"{"preserve_thinking": true}"#.to_string());
    let startup = input
        .into_startup_config()
        .expect("valid JSON object should succeed");
    let kwargs = startup
        .chat_template_kwargs
        .expect("non-empty kwargs should materialize");
    assert!(kwargs.preserve_thinking());
}

#[test]
fn into_startup_config_rejects_malformed_chat_template_kwargs_json() {
    let mut input = sample_input();
    input.chat_template_kwargs = Some("{not-json".to_string());
    let err = input
        .into_startup_config()
        .expect_err("malformed JSON must error at startup");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("chat-template-kwargs"),
        "error should reference the flag, got: {msg}"
    );
}

#[test]
fn into_startup_config_rejects_non_object_chat_template_kwargs_json() {
    let mut input = sample_input();
    input.chat_template_kwargs = Some("[true]".to_string());
    let err = input
        .into_startup_config()
        .expect_err("arrays must be rejected at startup");
    let msg = format!("{err:#}");
    assert!(msg.contains("chat-template-kwargs"));
    assert!(
        msg.contains("object"),
        "error should mention object, got: {msg}"
    );
}

#[test]
fn into_startup_config_empty_chat_template_kwargs_collapses_to_none() {
    let mut input = sample_input();
    input.chat_template_kwargs = Some("".to_string());
    let startup = input
        .into_startup_config()
        .expect("empty string is a valid no-op");
    assert!(startup.chat_template_kwargs.is_none());

    let mut input = sample_input();
    input.chat_template_kwargs = Some("{}".to_string());
    let startup = input
        .into_startup_config()
        .expect("empty JSON object is a valid no-op");
    assert!(startup.chat_template_kwargs.is_none());
}

// -------------------------------------------------------------------------
// B7 — LLAMA_ARG_LANG_BIAS env-var fallback tests (plan §6.4)
//
// Each test manages the env var explicitly (set + cleanup) and delegates to a
// helper that calls `env_fallback_lang_bias` directly, keeping the env
// mutation inside each test's stack frame for clarity.
//
// NOTE: Rust test threads share a process, so env-var mutations must be
// cleaned up regardless of test outcome. These tests run in-process and are
// intentionally structured to be self-contained.
// -------------------------------------------------------------------------

/// Helper that cleans up `LLAMA_ARG_LANG_BIAS` at drop time.
struct EnvGuard(&'static str);

impl EnvGuard {
    fn set(key: &'static str, value: &str) -> Self {
        // SAFETY: callers acquire `env_lock()` before constructing this guard
        // (see the `let _env_guard = env_lock();` lines in each test), so only
        // one thread mutates the process environment at a time.
        #[allow(unsafe_code)]
        unsafe {
            std::env::set_var(key, value);
        }
        EnvGuard(key)
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        // SAFETY: same env_lock serialization as in `set`; the lock guard is
        // dropped after this guard, so the lock still covers `remove_var`.
        #[allow(unsafe_code)]
        unsafe {
            std::env::remove_var(self.0);
        }
    }
}

/// B7 acceptance test: only `LLAMA_ARG_LANG_BIAS` is set (no CLI flag) →
/// the env value flows into `LangBiasCliArgs.lang_bias` and resolves to the
/// expected `LangBiasConfig`.
#[test]
fn env_var_feeds_parser() {
    use super::env_fallback_lang_bias;
    use crate::lang_bias::parse_lang_bias_entries;

    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_LANG_BIAS", "ja=-inf,zh=-5");

    let mut args = LangBiasCliArgs::default(); // lang_bias = None (no CLI flag)
    env_fallback_lang_bias(&mut args);

    // The env var value should now be in args.lang_bias.
    assert_eq!(
        args.lang_bias.as_deref(),
        Some("ja=-inf,zh=-5"),
        "env var value should be copied into lang_bias when CLI flag is absent"
    );

    // Resolve to a LangBiasConfig and verify the bias_set matches the expected pairs.
    let config = args.resolve().unwrap().unwrap();
    let expected = parse_lang_bias_entries("ja=-inf,zh=-5").unwrap();
    assert_eq!(
        config.bias_set.ordered.len(),
        expected.ordered.len(),
        "resolved bias_set should have the same number of entries as the env var"
    );
    for (i, ((got_lang, got_bias), (exp_lang, exp_bias))) in config
        .bias_set
        .ordered
        .iter()
        .zip(expected.ordered.iter())
        .enumerate()
    {
        assert_eq!(
            got_lang, exp_lang,
            "entry {i}: language code mismatch (got {got_lang:?}, expected {exp_lang:?})"
        );
        // Use exact equality for f32 since both come from the same parse path.
        assert_eq!(
            got_bias, exp_bias,
            "entry {i}: bias mismatch (got {got_bias}, expected {exp_bias})"
        );
    }
}

/// B7 acceptance test: `LLAMA_ARG_LANG_BIAS` is set without any CLI flag →
/// resolves to the env-var config.
#[test]
fn env_without_cli_parses() {
    use super::env_fallback_lang_bias;

    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_LANG_BIAS", "ko=+5");

    let mut args = LangBiasCliArgs::default();
    env_fallback_lang_bias(&mut args);

    assert_eq!(
        args.lang_bias.as_deref(),
        Some("ko=+5"),
        "env var value should be present when CLI flag is absent"
    );

    let config = args.resolve().unwrap().unwrap();
    assert_eq!(config.bias_set.ordered.len(), 1);
    use mlxcel_core::lang_analyzer::LanguageCode;
    assert_eq!(config.bias_set.ordered[0].0, LanguageCode::Ko);
    assert_eq!(config.bias_set.ordered[0].1, 5.0_f32);
}

/// B7 acceptance test: both CLI `--lang-bias ja=-inf` and env
/// `LLAMA_ARG_LANG_BIAS=ko=+5` are set → CLI wins (env is ignored) and an
/// INFO-level log message is emitted.
///
/// The log message is emitted via `tracing::info!` which writes to the
/// subscriber registered for the current thread; we verify that the CLI value
/// is kept without attempting to capture the log output (subscriber setup is
/// out of scope for a unit test).
#[test]
fn cli_overrides_env() {
    use super::env_fallback_lang_bias;

    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_LANG_BIAS", "ko=+5");

    // Simulate CLI providing `--lang-bias ja=-inf`.
    let mut args = LangBiasCliArgs {
        lang_bias: Some("ja=-inf".to_owned()),
        ..Default::default()
    };

    env_fallback_lang_bias(&mut args);

    // CLI value must be preserved; env var must NOT overwrite it.
    assert_eq!(
        args.lang_bias.as_deref(),
        Some("ja=-inf"),
        "CLI --lang-bias should take precedence over LLAMA_ARG_LANG_BIAS env var"
    );

    let config = args.resolve().unwrap().unwrap();
    assert_eq!(config.bias_set.ordered.len(), 1);
    use mlxcel_core::lang_analyzer::LanguageCode;
    assert_eq!(config.bias_set.ordered[0].0, LanguageCode::Ja);
    assert_eq!(config.bias_set.ordered[0].1, f32::NEG_INFINITY);
}

// -------------------------------------------------------------------------
// LLAMA_ARG_LANG_BIAS_INCLUDE_BYTE_FRAGMENTS env-var fallback
//
// Mirrors the B7 tests above. The env-var fallback for the byte-fragment
// opt-in is permissive about truthiness (accepts `true`/`false`/`1`/`0`) and
// respects CLI precedence.
// -------------------------------------------------------------------------

/// Env var set without CLI flag → `include_byte_fragments` is flipped to `true`.
#[test]
fn byte_fragments_env_var_feeds_flag_true() {
    use super::env_fallback_lang_bias_include_byte_fragments;

    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_LANG_BIAS_INCLUDE_BYTE_FRAGMENTS", "true");

    let mut args = LangBiasCliArgs::default();
    assert!(
        !args.include_byte_fragments,
        "default must be false before fallback runs"
    );
    env_fallback_lang_bias_include_byte_fragments(&mut args);
    assert!(
        args.include_byte_fragments,
        "truthy env var must flip include_byte_fragments to true"
    );
}

/// Env var supports `1` / `0` forms too.
#[test]
fn byte_fragments_env_var_accepts_numeric_forms() {
    use super::env_fallback_lang_bias_include_byte_fragments;

    let _env_guard = env_lock();
    // `1` → true
    {
        let _guard = EnvGuard::set("LLAMA_ARG_LANG_BIAS_INCLUDE_BYTE_FRAGMENTS", "1");
        let mut args = LangBiasCliArgs::default();
        env_fallback_lang_bias_include_byte_fragments(&mut args);
        assert!(args.include_byte_fragments, "`1` must parse as true");
    }
    // `0` → false → no flip when CLI was already false.
    {
        let _guard = EnvGuard::set("LLAMA_ARG_LANG_BIAS_INCLUDE_BYTE_FRAGMENTS", "0");
        let mut args = LangBiasCliArgs::default();
        env_fallback_lang_bias_include_byte_fragments(&mut args);
        assert!(
            !args.include_byte_fragments,
            "`0` must keep include_byte_fragments=false"
        );
    }
}

/// CLI `--lang-bias-include-byte-fragments` beats the env var.
#[test]
fn byte_fragments_cli_overrides_env() {
    use super::env_fallback_lang_bias_include_byte_fragments;

    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_LANG_BIAS_INCLUDE_BYTE_FRAGMENTS", "false");

    // CLI already set include_byte_fragments=true.
    let mut args = LangBiasCliArgs {
        include_byte_fragments: true,
        ..Default::default()
    };
    env_fallback_lang_bias_include_byte_fragments(&mut args);
    assert!(
        args.include_byte_fragments,
        "CLI --lang-bias-include-byte-fragments must win against env 'false'"
    );
}

/// Unparseable env var is ignored (warn-and-drop), leaving CLI default.
#[test]
fn byte_fragments_env_var_unparseable_is_ignored() {
    use super::env_fallback_lang_bias_include_byte_fragments;

    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_LANG_BIAS_INCLUDE_BYTE_FRAGMENTS", "maybe");

    let mut args = LangBiasCliArgs::default();
    env_fallback_lang_bias_include_byte_fragments(&mut args);
    assert!(
        !args.include_byte_fragments,
        "unparseable env var must leave the CLI default (false) in place"
    );
}

// -------------------------------------------------------------------------
// prompt-cache CLI/env config tests
// -------------------------------------------------------------------------

/// Default construction of `ServerStartupInput` with prompt-cache defaults
/// produces `PromptCacheConfig::default()` after normalization.
#[test]
fn prompt_cache_defaults_round_trip_through_into_startup_config() {
    use crate::server::prompt_cache::PromptCacheConfig;

    let input = sample_input();
    let startup = input.into_startup_config().expect("valid input");

    let expected = PromptCacheConfig::default();
    assert_eq!(startup.prompt_cache.enabled, expected.enabled);
    assert_eq!(startup.prompt_cache.capacity_bytes, expected.capacity_bytes);
    assert_eq!(startup.prompt_cache.max_entries, expected.max_entries);
    assert_eq!(startup.prompt_cache.ttl, expected.ttl);
    assert_eq!(
        startup.prompt_cache.min_prefix_tokens,
        expected.min_prefix_tokens
    );
}

/// CLI-supplied capacity is propagated.
#[test]
fn prompt_cache_capacity_cli_overrides_default() {
    let mut input = sample_input();
    input.prompt_cache_capacity_bytes = Some(1024 * 1024 * 512); // 512 MiB

    let startup = input.into_startup_config().expect("valid input");
    assert_eq!(startup.prompt_cache.capacity_bytes, 1024 * 1024 * 512);
}

/// CLI-supplied max_entries is propagated.
#[test]
fn prompt_cache_max_entries_cli_overrides_default() {
    let mut input = sample_input();
    input.prompt_cache_max_entries = Some(256);

    let startup = input.into_startup_config().expect("valid input");
    assert_eq!(startup.prompt_cache.max_entries, 256);
}

/// CLI-supplied TTL is propagated.
#[test]
fn prompt_cache_ttl_cli_overrides_default() {
    let mut input = sample_input();
    input.prompt_cache_ttl_seconds = Some(600);

    let startup = input.into_startup_config().expect("valid input");
    assert_eq!(startup.prompt_cache.ttl.as_secs(), 600);
}

/// CLI-supplied min_prefix is propagated.
#[test]
fn prompt_cache_min_prefix_cli_overrides_default() {
    let mut input = sample_input();
    input.prompt_cache_min_prefix = Some(64);

    let startup = input.into_startup_config().expect("valid input");
    assert_eq!(startup.prompt_cache.min_prefix_tokens, 64);
}

/// The three snapshot-store knobs reach `PromptCacheConfig` from the CLI
/// fields, and each one is independent of the others (issue #1146).
#[test]
fn prompt_cache_snapshot_limits_cli_override_defaults() {
    let mut input = sample_input();
    input.prompt_cache_snapshot_capacity_bytes = Some(3 * 1024 * 1024 * 1024);
    input.prompt_cache_snapshot_max_entries = Some(64);
    input.prompt_cache_snapshot_ttl_seconds = Some(900);

    let startup = input.into_startup_config().expect("valid input");
    assert_eq!(
        startup.prompt_cache.snapshot_capacity_bytes,
        3 * 1024 * 1024 * 1024
    );
    assert_eq!(startup.prompt_cache.snapshot_max_entries, 64);
    assert_eq!(startup.prompt_cache.snapshot_ttl.as_secs(), 900);
}

/// Leaving the snapshot knobs unset keeps the compiled-in defaults, and does
/// not disturb the sibling KV-cache budget.
#[test]
fn prompt_cache_snapshot_limits_default_when_unset() {
    use crate::server::prompt_cache::PromptCacheConfig;

    let startup = sample_input().into_startup_config().expect("valid input");
    assert_eq!(
        startup.prompt_cache.snapshot_capacity_bytes,
        PromptCacheConfig::DEFAULT_SNAPSHOT_CAPACITY_BYTES
    );
    assert_eq!(
        startup.prompt_cache.snapshot_max_entries,
        PromptCacheConfig::DEFAULT_SNAPSHOT_MAX_ENTRIES
    );
    assert_eq!(
        startup.prompt_cache.snapshot_ttl.as_secs(),
        PromptCacheConfig::DEFAULT_SNAPSHOT_TTL_SECONDS
    );
    assert_eq!(
        startup.prompt_cache.capacity_bytes,
        PromptCacheConfig::DEFAULT_CAPACITY_BYTES,
        "the snapshot budget is separate from the KV budget"
    );
}

/// Disabling via CLI produces `enabled = false` in the config.
#[test]
fn prompt_cache_disabled_cli_propagates_through() {
    let mut input = sample_input();
    input.prompt_cache_enabled = false;

    let startup = input.into_startup_config().expect("valid input");
    assert!(!startup.prompt_cache.enabled);
    assert!(!startup.prompt_cache.is_enabled());
}

/// `MLXCEL_PROMPT_CACHE_ENABLED=false` disables the cache.
#[test]
fn prompt_cache_enabled_env_var_sets_false() {
    use super::env_fallback_prompt_cache_enabled;

    let _env_guard = env_lock();
    let _guard = EnvGuard::set("MLXCEL_PROMPT_CACHE_ENABLED", "false");

    let mut enabled = true; // default
    env_fallback_prompt_cache_enabled(&mut enabled, false);
    assert!(
        !enabled,
        "MLXCEL_PROMPT_CACHE_ENABLED=false must set enabled=false"
    );
}

/// `MLXCEL_PROMPT_CACHE_ENABLED=1` enables the cache.
#[test]
fn prompt_cache_enabled_env_var_accepts_numeric_one() {
    use super::env_fallback_prompt_cache_enabled;

    let _env_guard = env_lock();
    let _guard = EnvGuard::set("MLXCEL_PROMPT_CACHE_ENABLED", "1");

    let mut enabled = false;
    env_fallback_prompt_cache_enabled(&mut enabled, false);
    assert!(
        enabled,
        "MLXCEL_PROMPT_CACHE_ENABLED=1 must set enabled=true"
    );
}

/// `LLAMA_ARG_CACHE_REUSE` is an integer, not a boolean cache-enable alias.
///
/// Since #1453 the variable is bound to the real `--cache-reuse` option rather
/// than validated inside the prompt-cache enable fallback, so the coverage
/// moved with it; what is asserted has not changed.
#[test]
fn prompt_cache_llama_arg_cache_reuse_boolean_is_rejected() {
    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_CACHE_REUSE", "on");

    let error = crate::cli::cache_args::from_env().expect_err("boolean cache-reuse must fail");
    assert!(error.contains("cache-reuse"), "{error}");
}

/// Positive minimum reuse chunk sizes are not implemented and fail clearly.
#[test]
fn prompt_cache_llama_arg_cache_reuse_positive_is_rejected() {
    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_CACHE_REUSE", "256");

    let error =
        crate::cli::cache_args::from_env().expect_err("positive cache-reuse must fail until #1453");
    assert!(error.contains("--cache-reuse 256"), "{error}");
    assert!(error.contains("--cache-reuse 0"), "{error}");
}

/// Cache enablement and cache-reuse tuning stay independent settings: an
/// invalid reuse value must not be masked by a valid enable value, and it must
/// not disable the cache either.
#[test]
fn prompt_cache_mlxcel_env_does_not_hide_invalid_cache_reuse() {
    let _env_guard = env_lock();
    let _mlxcel = EnvGuard::set("MLXCEL_PROMPT_CACHE_ENABLED", "true");
    let _llama = EnvGuard::set("LLAMA_ARG_CACHE_REUSE", "on");

    let mut enabled = false;
    super::env_fallback_prompt_cache_enabled(&mut enabled, false);
    assert!(enabled, "the actual enable setting must still be applied");
    assert!(
        crate::cli::cache_args::from_env().is_err(),
        "and the invalid reuse value must still be reported"
    );
}

/// `LLAMA_ARG_CACHE_PROMPT` is the b10621 spelling of the same setting, and a
/// falsy value has to reach the config as a disable rather than as an absence.
#[test]
fn llama_arg_cache_prompt_disables_the_prompt_cache() {
    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_CACHE_PROMPT", "0");

    let resolved = crate::cli::cache_args::from_env().expect("0 is a valid boolean");
    assert_eq!(resolved.prompt_cache_enabled, Some(false));
}

/// `LLAMA_ARG_CONT_BATCHING=0` pins the decode width to one sequence.
#[test]
fn llama_arg_cont_batching_zero_pins_the_decode_width() {
    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_CONT_BATCHING", "0");

    let resolved = crate::cli::cache_args::from_env().expect("0 is a valid boolean");
    assert!(resolved.single_sequence_decode);
}

/// `LLAMA_ARG_CACHE_RAM` is stated in MiB, not bytes.
#[test]
fn llama_arg_cache_ram_is_read_in_mebibytes() {
    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_CACHE_RAM", "64");

    let resolved = crate::cli::cache_args::from_env().expect("64 MiB is valid");
    assert_eq!(resolved.capacity_bytes, Some(64 * 1024 * 1024));
}

/// CLI-set `enabled=false` wins over any env var.
#[test]
fn prompt_cache_cli_wins_over_env_for_enabled() {
    use super::env_fallback_prompt_cache_enabled;

    let _env_guard = env_lock();
    let _guard = EnvGuard::set("MLXCEL_PROMPT_CACHE_ENABLED", "true");

    let mut enabled = false; // CLI said false
    env_fallback_prompt_cache_enabled(&mut enabled, true /* cli_was_set */);
    assert!(!enabled, "CLI value must win when cli_was_set=true");
}

/// `MLXCEL_PROMPT_CACHE_CAPACITY_BYTES` is applied when CLI flag is absent.
#[test]
fn prompt_cache_capacity_env_var_applied_when_cli_absent() {
    use super::env_fallback_prompt_cache_capacity_bytes;

    let _env_guard = env_lock();
    let _guard = EnvGuard::set("MLXCEL_PROMPT_CACHE_CAPACITY_BYTES", "1073741824"); // 1 GiB

    let mut value: Option<usize> = None;
    env_fallback_prompt_cache_capacity_bytes(&mut value);
    assert_eq!(value, Some(1_073_741_824));
}

/// CLI-set `capacity_bytes` wins over env var.
#[test]
fn prompt_cache_capacity_cli_wins_over_env() {
    use super::env_fallback_prompt_cache_capacity_bytes;

    let _env_guard = env_lock();
    let _guard = EnvGuard::set("MLXCEL_PROMPT_CACHE_CAPACITY_BYTES", "1073741824");

    let mut value: Option<usize> = Some(536_870_912); // CLI set 512 MiB
    env_fallback_prompt_cache_capacity_bytes(&mut value);
    assert_eq!(value, Some(536_870_912), "CLI value must be preserved");
}

/// `MLXCEL_PROMPT_CACHE_MAX_ENTRIES` is applied when CLI flag is absent.
#[test]
fn prompt_cache_max_entries_env_var_applied() {
    use super::env_fallback_prompt_cache_max_entries;

    let _env_guard = env_lock();
    let _guard = EnvGuard::set("MLXCEL_PROMPT_CACHE_MAX_ENTRIES", "512");

    let mut value: Option<usize> = None;
    env_fallback_prompt_cache_max_entries(&mut value);
    assert_eq!(value, Some(512));
}

/// `MLXCEL_PROMPT_CACHE_TTL` is applied when CLI flag is absent.
#[test]
fn prompt_cache_ttl_env_var_applied() {
    use super::env_fallback_prompt_cache_ttl;

    let _env_guard = env_lock();
    let _guard = EnvGuard::set("MLXCEL_PROMPT_CACHE_TTL", "1800");

    let mut value: Option<u64> = None;
    env_fallback_prompt_cache_ttl(&mut value);
    assert_eq!(value, Some(1800));
}

/// `MLXCEL_PROMPT_CACHE_MIN_PREFIX` is applied when CLI flag is absent.
#[test]
fn prompt_cache_min_prefix_env_var_applied() {
    use super::env_fallback_prompt_cache_min_prefix;

    let _env_guard = env_lock();
    let _guard = EnvGuard::set("MLXCEL_PROMPT_CACHE_MIN_PREFIX", "16");

    let mut value: Option<usize> = None;
    env_fallback_prompt_cache_min_prefix(&mut value);
    assert_eq!(value, Some(16));
}

/// `MLXCEL_PROMPT_CACHE_SNAPSHOT_CAPACITY_BYTES` is applied when the CLI flag
/// is absent, and the CLI flag wins when both are present.
#[test]
fn prompt_cache_snapshot_capacity_bytes_env_var_applied() {
    use super::env_fallback_prompt_cache_snapshot_capacity_bytes;

    let _env_guard = env_lock();
    let _guard = EnvGuard::set("MLXCEL_PROMPT_CACHE_SNAPSHOT_CAPACITY_BYTES", "1048576");

    let mut value: Option<usize> = None;
    env_fallback_prompt_cache_snapshot_capacity_bytes(&mut value);
    assert_eq!(value, Some(1_048_576));

    let mut from_cli = Some(4096);
    env_fallback_prompt_cache_snapshot_capacity_bytes(&mut from_cli);
    assert_eq!(from_cli, Some(4096), "CLI flag beats env var");
}

/// `MLXCEL_PROMPT_CACHE_SNAPSHOT_MAX_ENTRIES` is applied when the CLI flag is
/// absent.
#[test]
fn prompt_cache_snapshot_max_entries_env_var_applied() {
    use super::env_fallback_prompt_cache_snapshot_max_entries;

    let _env_guard = env_lock();
    let _guard = EnvGuard::set("MLXCEL_PROMPT_CACHE_SNAPSHOT_MAX_ENTRIES", "12");

    let mut value: Option<usize> = None;
    env_fallback_prompt_cache_snapshot_max_entries(&mut value);
    assert_eq!(value, Some(12));
}

/// `MLXCEL_PROMPT_CACHE_SNAPSHOT_TTL` is applied when the CLI flag is absent,
/// and an unparseable value is ignored rather than failing startup.
#[test]
fn prompt_cache_snapshot_ttl_env_var_applied() {
    use super::env_fallback_prompt_cache_snapshot_ttl;

    let _env_guard = env_lock();
    {
        let _guard = EnvGuard::set("MLXCEL_PROMPT_CACHE_SNAPSHOT_TTL", "5400");
        let mut value: Option<u64> = None;
        env_fallback_prompt_cache_snapshot_ttl(&mut value);
        assert_eq!(value, Some(5400));
    }

    let _guard = EnvGuard::set("MLXCEL_PROMPT_CACHE_SNAPSHOT_TTL", "forever");
    let mut value: Option<u64> = None;
    env_fallback_prompt_cache_snapshot_ttl(&mut value);
    assert_eq!(value, None, "unparseable TTL is ignored");
}

/// Unparseable `MLXCEL_PROMPT_CACHE_ENABLED` is ignored and the original value
/// (default `true`) is preserved.
#[test]
fn prompt_cache_enabled_unparseable_env_var_ignored() {
    use super::env_fallback_prompt_cache_enabled;

    let _env_guard = env_lock();
    let _guard = EnvGuard::set("MLXCEL_PROMPT_CACHE_ENABLED", "maybe-yes");

    let mut enabled = true;
    env_fallback_prompt_cache_enabled(&mut enabled, false);
    assert!(
        enabled,
        "unparseable MLXCEL_PROMPT_CACHE_ENABLED must leave original value in place"
    );
}

/// `LLAMA_ARG_CACHE_REUSE=0` is a no-op and leaves prompt caching enabled.
#[test]
fn prompt_cache_llama_arg_cache_reuse_zero_leaves_cache_enabled() {
    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_CACHE_REUSE", "0");

    let resolved = crate::cli::cache_args::from_env().expect("zero is supported");
    assert_eq!(
        resolved.prompt_cache_enabled, None,
        "LLAMA_ARG_CACHE_REUSE=0 must not say anything about whether the cache is enabled"
    );
}

/// Integration: `into_startup_config` with `enabled=false` produces a
/// `ServerStartupConfig` whose `prompt_cache.is_enabled()` returns `false`.
#[test]
fn b10621_no_cache_prompt_disables_the_store() {
    let mut input = sample_input();
    input.cache_compat.no_cache_prompt = true;

    let startup = input.into_startup_config().expect("valid input");
    assert!(
        !startup.prompt_cache.is_enabled(),
        "--no-cache-prompt must reach PromptCacheConfig, not just parse"
    );
}

/// `--cache-prompt` asserts the default and cannot outvote mlxcel's own
/// disable. Between two explicit operator statements the safe reading is the
/// one that caches nothing.
#[test]
fn b10621_cache_prompt_cannot_reenable_over_the_native_disable() {
    let mut input = sample_input();
    input.prompt_cache_enabled = false;
    input.cache_compat.cache_prompt = Some(true);

    let startup = input.into_startup_config().expect("valid input");
    assert!(!startup.prompt_cache.is_enabled());
}

/// `--cache-ram` is the same budget as `--prompt-cache-capacity-bytes`, stated
/// in MiB, and it has to reach the config in bytes.
#[test]
fn b10621_cache_ram_sets_the_prompt_cache_budget_in_bytes() {
    let mut input = sample_input();
    input.prompt_cache_capacity_bytes = None;
    input.cache_compat.cache_ram = Some(64);

    let startup = input.into_startup_config().expect("valid input");
    assert_eq!(startup.prompt_cache.capacity_bytes, 64 * 1024 * 1024);
}

/// The native spelling is the more specific request and wins, the way
/// `--prefill-chunk-size` wins over `--batch-size`.
#[test]
fn the_native_capacity_flag_wins_over_cache_ram() {
    let mut input = sample_input();
    input.prompt_cache_capacity_bytes = Some(7 * 1024 * 1024);
    input.cache_compat.cache_ram = Some(64);

    let startup = input.into_startup_config().expect("valid input");
    assert_eq!(startup.prompt_cache.capacity_bytes, 7 * 1024 * 1024);
}

/// `--cache-ram 0` is upstream's disable sentinel, and it has to disable
/// rather than fall through to the compiled-in default: `is_enabled()` reads
/// `capacity_bytes > 0`, so a zero that arrived as "unset" would silently
/// leave a 2 GiB cache running.
#[test]
fn b10621_cache_ram_zero_disables_the_store() {
    let mut input = sample_input();
    input.prompt_cache_capacity_bytes = None;
    input.cache_compat.cache_ram = Some(0);

    let startup = input.into_startup_config().expect("valid input");
    assert_eq!(startup.prompt_cache.capacity_bytes, 0);
    assert!(!startup.prompt_cache.is_enabled());
}

/// `--no-cont-batching` pins the decode width to one sequence. It is not
/// mlxcel's `--no-batch`, which would also take the scheduler out.
#[test]
fn b10621_no_cont_batching_pins_the_decode_width_without_removing_the_scheduler() {
    let mut input = sample_input();
    input.max_batch_size = None;
    input.cache_compat.no_cont_batching = true;

    let startup = input.into_startup_config().expect("valid input");
    assert_eq!(startup.max_batch_size, Some(1));
    assert!(
        !startup.no_batch,
        "--no-cont-batching must not fall through to the legacy sequential worker"
    );
}

/// A positive `--cache-reuse` fails the command line rather than the first
/// request that would have reused a chunk.
#[test]
fn b10621_positive_cache_reuse_fails_startup_configuration() {
    let mut input = sample_input();
    input.cache_compat.cache_reuse = Some(256);

    let error = input
        .into_startup_config()
        .expect_err("KV-shift chunk reuse is not implemented");
    assert!(error.to_string().contains("--cache-reuse 256"), "{error}");
}

/// Integration: `into_startup_config` with `enabled=false` produces a
/// `ServerStartupConfig` whose `prompt_cache.is_enabled()` returns `false`.
#[test]
fn startup_config_prompt_cache_disabled_produces_false_is_enabled() {
    let mut input = sample_input();
    input.prompt_cache_enabled = false;

    let startup = input.into_startup_config().expect("valid input");
    assert!(
        !startup.prompt_cache.is_enabled(),
        "disabled prompt cache must not satisfy is_enabled()"
    );
}

/// Integration: `into_startup_config` with the default produces a
/// `ServerStartupConfig` whose `prompt_cache.is_enabled()` returns `true`.
#[test]
fn startup_config_prompt_cache_default_is_enabled() {
    let startup = sample_input().into_startup_config().expect("valid input");
    assert!(
        startup.prompt_cache.is_enabled(),
        "default prompt cache must satisfy is_enabled()"
    );
}

/// Integration: `into_startup_config` with `capacity_bytes` overridden
/// propagates the correct value to `ServerStartupConfig.prompt_cache`.
#[test]
fn prompt_cache_e2e_cli_capacity_bytes_flows_to_startup_config() {
    let mut input = sample_input();
    input.prompt_cache_capacity_bytes = Some(134_217_728); // 128 MiB

    let startup = input.into_startup_config().expect("valid input");
    assert_eq!(startup.prompt_cache.capacity_bytes, 134_217_728);
}

// ─────────────────────────────────────────────────────────────────────────────
// (B11) — resolve_kv_cache_mode tests
//
// These tests cover the K/V-to-KVCacheMode mapping logic: all supported pairs,
// unsupported combinations, and legacy/split flag interaction.
// ─────────────────────────────────────────────────────────────────────────────

use mlxcel_core::cache::KVCacheMode;

/// Default (no flags) → FP16.
#[test]
fn kv_cache_mode_default_is_fp16() {
    let mode = resolve_kv_cache_mode(None, None, None).expect("default must succeed");
    assert_eq!(mode, KVCacheMode::Fp16);
}

/// Both sides fp16 → Fp16.
#[test]
fn kv_cache_mode_fp16_fp16_maps_to_fp16() {
    let mode = resolve_kv_cache_mode(Some("fp16"), Some("fp16"), None).unwrap();
    assert_eq!(mode, KVCacheMode::Fp16);
}

/// Both sides int8 → Int8.
#[test]
fn kv_cache_mode_int8_int8_maps_to_int8() {
    let mode = resolve_kv_cache_mode(Some("int8"), Some("int8"), None).unwrap();
    assert_eq!(mode, KVCacheMode::Int8);
}

/// K=fp16, V=turbo4 → Turbo4Asym.
#[test]
fn kv_cache_mode_fp16_turbo4_maps_to_turbo4asym() {
    let mode = resolve_kv_cache_mode(Some("fp16"), Some("turbo4"), None).unwrap();
    assert_eq!(mode, KVCacheMode::Turbo4Asym);
}

/// K=fp16, V=turbo4-asym (explicit alias) → Turbo4Asym.
#[test]
fn kv_cache_mode_fp16_turbo4_asym_maps_to_turbo4asym() {
    let mode = resolve_kv_cache_mode(Some("fp16"), Some("turbo4-asym"), None).unwrap();
    assert_eq!(mode, KVCacheMode::Turbo4Asym);
}

/// K=turbo4, V=turbo4 → Turbo4 (symmetric, allowlist-gated at runtime).
#[test]
fn kv_cache_mode_turbo4_turbo4_maps_to_turbo4() {
    let mode = resolve_kv_cache_mode(Some("turbo4"), Some("turbo4"), None).unwrap();
    assert_eq!(mode, KVCacheMode::Turbo4);
}

/// K=fp16, V=turbo4-delegated → Turbo4Delegated.
#[test]
fn kv_cache_mode_fp16_turbo4_delegated_maps_to_delegated() {
    let mode = resolve_kv_cache_mode(Some("fp16"), Some("turbo4-delegated"), None).unwrap();
    assert_eq!(mode, KVCacheMode::Turbo4Delegated);
}

/// Unspecified K defaults to fp16 — K=None + V=turbo4 → Turbo4Asym.
#[test]
fn kv_cache_mode_k_defaults_to_fp16_when_unset() {
    let mode = resolve_kv_cache_mode(None, Some("turbo4"), None).unwrap();
    assert_eq!(mode, KVCacheMode::Turbo4Asym);
}

/// Unspecified V defaults to fp16 — K=None + V=None → Fp16.
#[test]
fn kv_cache_mode_v_defaults_to_fp16_when_unset() {
    let mode = resolve_kv_cache_mode(Some("fp16"), None, None).unwrap();
    assert_eq!(mode, KVCacheMode::Fp16);
}

/// Unsupported combination — K=int8, V=turbo4 → error.
#[test]
fn kv_cache_mode_int8_turbo4_is_unsupported() {
    let err = resolve_kv_cache_mode(Some("int8"), Some("turbo4"), None)
        .expect_err("int8/turbo4 must be rejected");
    assert!(
        err.contains("unsupported"),
        "error must mention 'unsupported', got: {err}"
    );
    assert!(
        err.contains("supported pairs"),
        "error must list supported pairs, got: {err}"
    );
}

/// Unsupported combination — K=turbo4, V=fp16 → error.
#[test]
fn kv_cache_mode_turbo4_fp16_is_unsupported() {
    let err = resolve_kv_cache_mode(Some("turbo4"), Some("fp16"), None)
        .expect_err("turbo4/fp16 must be rejected");
    assert!(err.contains("unsupported"));
}

/// Unknown K string → error mentioning the bad value.
#[test]
fn kv_cache_mode_unknown_k_string_errors() {
    let err = resolve_kv_cache_mode(Some("bfloat16"), Some("fp16"), None)
        .expect_err("unrecognised K must be rejected");
    assert!(
        err.contains("bfloat16"),
        "error must name the bad value, got: {err}"
    );
}

/// Unknown V string → error mentioning the bad value.
#[test]
fn kv_cache_mode_unknown_v_string_errors() {
    let err = resolve_kv_cache_mode(Some("fp16"), Some("bf16"), None)
        .expect_err("unrecognised V must be rejected");
    assert!(
        err.contains("bf16"),
        "error must name the bad value, got: {err}"
    );
}

/// Legacy --kv-cache-mode shorthand sets the mode when split flags are absent.
#[test]
fn kv_cache_mode_legacy_flag_sets_mode() {
    let mode =
        resolve_kv_cache_mode(None, None, Some("fp16+turbo4")).expect("legacy flag must work");
    assert_eq!(mode, KVCacheMode::Turbo4Asym);
}

/// Legacy --kv-cache-mode=int8 shorthand sets Int8.
#[test]
fn kv_cache_mode_legacy_int8_sets_int8() {
    let mode = resolve_kv_cache_mode(None, None, Some("int8")).expect("legacy int8 must work");
    assert_eq!(mode, KVCacheMode::Int8);
}

/// Split flags win over legacy when both are provided.
#[test]
fn kv_cache_mode_split_flags_take_precedence_over_legacy() {
    // split says fp16/fp16 (Fp16), legacy says int8 — split wins
    let mode = resolve_kv_cache_mode(Some("fp16"), Some("fp16"), Some("int8")).unwrap();
    assert_eq!(
        mode,
        KVCacheMode::Fp16,
        "split flags must win over legacy --kv-cache-mode"
    );
}

/// Unknown legacy value → clear error.
#[test]
fn kv_cache_mode_legacy_unknown_value_errors() {
    let err = resolve_kv_cache_mode(None, None, Some("unknown-mode"))
        .expect_err("unknown legacy mode must fail");
    assert!(
        err.contains("unknown-mode"),
        "error must name the bad value, got: {err}"
    );
}

/// Integration: split flags flow through `into_startup_config` to
/// `ServerStartupConfig.kv_cache_mode`.
#[test]
fn into_startup_config_kv_cache_mode_from_split_flags() {
    let mut input = sample_input();
    input.cache_type_k = Some("fp16".to_string());
    input.cache_type_v = Some("turbo4".to_string());

    let startup = input
        .into_startup_config()
        .expect("fp16+turbo4 is a supported pair");
    assert_eq!(startup.kv_cache_mode, KVCacheMode::Turbo4Asym);
}

/// Integration: legacy flag flows through `into_startup_config`.
#[test]
fn into_startup_config_kv_cache_mode_from_legacy_flag() {
    let mut input = sample_input();
    input.kv_cache_mode_legacy = Some("int8".to_string());

    let startup = input.into_startup_config().expect("int8 legacy mode valid");
    assert_eq!(startup.kv_cache_mode, KVCacheMode::Int8);
}

/// Integration: default (no flags) → FP16 in the startup config.
#[test]
fn into_startup_config_kv_cache_mode_default_is_fp16() {
    let startup = sample_input()
        .into_startup_config()
        .expect("default input is valid");
    assert_eq!(startup.kv_cache_mode, KVCacheMode::Fp16);
}

#[test]
fn into_startup_config_propagates_decode_storage_backend() {
    let mut input = sample_input();
    input.decode_storage_backend = Some(DecodeStorageBackend::Paged);

    let startup = input.into_startup_config().expect("paged backend is valid");
    assert_eq!(
        startup.decode_storage_backend,
        Some(DecodeStorageBackend::Paged)
    );
}

/// Integration: unsupported pair propagated as an error.
#[test]
fn into_startup_config_kv_cache_mode_unsupported_pair_errors() {
    let mut input = sample_input();
    input.cache_type_k = Some("int8".to_string());
    input.cache_type_v = Some("turbo4".to_string());

    let err = input
        .into_startup_config()
        .expect_err("int8/turbo4 must fail");
    let msg = format!("{err:#}");
    assert!(msg.contains("KV cache mode error") || msg.contains("unsupported"));
}

// ─────────────────────────────────────────────────────────────────────────────
// (B11) — env-var fallback tests for LLAMA_ARG_CACHE_TYPE_K/V
// ─────────────────────────────────────────────────────────────────────────────

/// `LLAMA_ARG_CACHE_TYPE_K` is applied when the CLI flag is absent.
#[test]
fn cache_type_k_env_var_applied_when_cli_absent() {
    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_CACHE_TYPE_K", "int8");

    let mut value: Option<String> = None;
    env_fallback_cache_type_k(&mut value);
    assert_eq!(value.as_deref(), Some("int8"));
}

/// CLI `--cache-type-k` wins over `LLAMA_ARG_CACHE_TYPE_K`.
#[test]
fn cache_type_k_cli_wins_over_env() {
    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_CACHE_TYPE_K", "int8");

    let mut value: Option<String> = Some("fp16".to_string());
    env_fallback_cache_type_k(&mut value);
    assert_eq!(value.as_deref(), Some("fp16"), "CLI must win");
}

/// `LLAMA_ARG_CACHE_TYPE_V` is applied when the CLI flag is absent.
#[test]
fn cache_type_v_env_var_applied_when_cli_absent() {
    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_CACHE_TYPE_V", "turbo4");

    let mut value: Option<String> = None;
    env_fallback_cache_type_v(&mut value);
    assert_eq!(value.as_deref(), Some("turbo4"));
}

/// CLI `--cache-type-v` wins over `LLAMA_ARG_CACHE_TYPE_V`.
#[test]
fn cache_type_v_cli_wins_over_env() {
    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_CACHE_TYPE_V", "turbo4");

    let mut value: Option<String> = Some("fp16".to_string());
    env_fallback_cache_type_v(&mut value);
    assert_eq!(value.as_deref(), Some("fp16"), "CLI must win");
}

/// Empty env var string is not applied (treated as absent).
#[test]
fn cache_type_k_empty_env_var_is_ignored() {
    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_CACHE_TYPE_K", "   ");

    let mut value: Option<String> = None;
    env_fallback_cache_type_k(&mut value);
    assert!(
        value.is_none(),
        "whitespace-only env var must not be applied"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// env-var fallback tests for kv-bits / kv-group-size /
// kv-quant-scheme / kv-skip-last-layer (Fix 1, Fix 2)
// ─────────────────────────────────────────────────────────────────────────────

/// Fix 1: when LLAMA_ARG_KV_BITS and --kv-bits agree (clap injected the env
/// value as the CLI value), the helper must NOT emit the misleading "env
/// ignored" conflict. We test indirectly: the function must not panic and the
/// value must be preserved.
#[test]
fn kv_bits_no_conflict_when_env_and_cli_agree() {
    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_KV_BITS", "8");

    // Simulate clap having already injected LLAMA_ARG_KV_BITS=8 into the
    // CLI value.
    let mut value: i32 = 8;
    env_fallback_kv_bits(&mut value);
    assert_eq!(value, 8, "value must be preserved when env and CLI agree");
}

/// Fix 1: when LLAMA_ARG_KV_BITS differs from --kv-bits, the CLI value must
/// still win (backward compat).
#[test]
fn kv_bits_cli_wins_when_env_differs() {
    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_KV_BITS", "4");

    let mut value: i32 = 8; // explicit --kv-bits 8
    env_fallback_kv_bits(&mut value);
    assert_eq!(value, 8, "CLI --kv-bits must win over differing env var");
}

/// Fix 1: LLAMA_ARG_KV_BITS is applied when no CLI flag was given (value == 0
/// is the sentinel meaning "not set").
#[test]
fn kv_bits_env_applied_when_cli_absent() {
    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_KV_BITS", "8");

    let mut value: i32 = 0;
    env_fallback_kv_bits(&mut value);
    assert_eq!(value, 8, "env var must apply when CLI flag was not given");
}

/// Fix 1: kv-group-size — env and CLI agree (clap injected), no spurious conflict.
#[test]
fn kv_group_size_no_conflict_when_env_and_cli_agree() {
    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_KV_GROUP_SIZE", "32");

    // Simulate clap injecting the env value as the CLI value.
    let mut value: i32 = 32;
    env_fallback_kv_group_size(&mut value);
    assert_eq!(value, 32, "value must be preserved when env and CLI agree");
}

/// Fix 1: kv-quant-scheme — env and CLI agree (clap injected), no conflict.
#[test]
fn kv_quant_scheme_no_conflict_when_env_and_cli_agree() {
    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_KV_QUANT_SCHEME", "turboquant");

    let mut value: Option<String> = Some("turboquant".to_string());
    env_fallback_kv_quant_scheme(&mut value);
    assert_eq!(
        value.as_deref(),
        Some("turboquant"),
        "value must be preserved when env and CLI agree"
    );
}

/// Fix 1: kv-skip-last-layer — env and CLI agree (clap injected `false`),
/// no spurious conflict log and the value is preserved.
#[test]
fn kv_skip_last_layer_no_conflict_when_env_and_cli_agree() {
    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_KV_SKIP_LAST_LAYER", "false");

    // Simulate clap injecting the env value `false` as the CLI value.
    let mut value: bool = false;
    env_fallback_kv_skip_last_layer(&mut value);
    assert!(!value, "value must remain false when env and CLI agree");
}

/// Fix 2: when LLAMA_ARG_KV_SKIP_LAST_LAYER is unparseable, the function must
/// fall through to MLXCEL_KV_SKIP_LAST_LAYER and apply the valid fallback.
#[test]
fn kv_skip_last_layer_unparseable_llama_falls_through_to_mlxcel() {
    let _env_guard = env_lock();
    let _guard1 = EnvGuard::set("LLAMA_ARG_KV_SKIP_LAST_LAYER", "garbage");
    let _guard2 = EnvGuard::set("MLXCEL_KV_SKIP_LAST_LAYER", "false");

    // CLI default is `true` (the sentinel meaning "not overridden").
    let mut value: bool = true;
    env_fallback_kv_skip_last_layer(&mut value);
    assert!(
        !value,
        "MLXCEL_KV_SKIP_LAST_LAYER=false must apply when LLAMA_ARG_KV_SKIP_LAST_LAYER is unparseable"
    );
}

/// Fix 2: a parseable LLAMA_ARG_KV_SKIP_LAST_LAYER must still take effect
/// (regression guard — the fall-through fix must not break the happy path).
#[test]
fn kv_skip_last_layer_parseable_llama_applies() {
    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_KV_SKIP_LAST_LAYER", "false");

    let mut value: bool = true;
    env_fallback_kv_skip_last_layer(&mut value);
    assert!(!value, "LLAMA_ARG_KV_SKIP_LAST_LAYER=false must be applied");
}

/// Fix 2: when both LLAMA and MLXCEL env vars are unparseable, value stays at
/// the CLI default (true).
#[test]
fn kv_skip_last_layer_both_unparseable_keeps_cli_default() {
    let _env_guard = env_lock();
    let _guard1 = EnvGuard::set("LLAMA_ARG_KV_SKIP_LAST_LAYER", "garbage");
    let _guard2 = EnvGuard::set("MLXCEL_KV_SKIP_LAST_LAYER", "also-garbage");

    let mut value: bool = true;
    env_fallback_kv_skip_last_layer(&mut value);
    assert!(
        value,
        "value must remain at CLI default when all env vars are unparseable"
    );
}

// ── H1: `--max-kv-size` validation ───────────────────────────────

#[test]
fn resolve_max_kv_size_zero_is_disabled() {
    assert_eq!(resolve_max_kv_size(0), Ok(None));
}

#[test]
fn resolve_max_kv_size_accepts_default_min() {
    assert_eq!(
        resolve_max_kv_size(MAX_KV_SIZE_MIN),
        Ok(Some(MAX_KV_SIZE_MIN))
    );
}

#[test]
fn resolve_max_kv_size_accepts_typical_value() {
    assert_eq!(resolve_max_kv_size(4096), Ok(Some(4096)));
}

#[test]
fn resolve_max_kv_size_accepts_i32_max() {
    let max = i32::MAX as usize;
    assert_eq!(resolve_max_kv_size(max), Ok(Some(max)));
}

#[test]
fn resolve_max_kv_size_rejects_below_min() {
    let err = resolve_max_kv_size(MAX_KV_SIZE_MIN - 1).expect_err("must reject");
    assert!(
        err.contains("below the minimum"),
        "error must mention the minimum bound: got {err:?}"
    );
    // The smallest valid non-zero value is exactly `MAX_KV_SIZE_MIN`.
    let err = resolve_max_kv_size(1).expect_err("must reject");
    assert!(err.contains("below the minimum"));
}

#[test]
fn resolve_max_kv_size_rejects_above_i32_max() {
    let too_big = (i32::MAX as usize) + 1;
    let err = resolve_max_kv_size(too_big).expect_err("must reject");
    assert!(
        err.contains("i32::MAX"),
        "error must mention the i32 overflow: got {err:?}"
    );
}

#[test]
fn into_startup_config_rejects_overflowing_max_kv_size() {
    let mut input = sample_input();
    input.max_kv_size = (i32::MAX as usize) + 1;
    let err = input
        .into_startup_config()
        .expect_err("overflowing --max-kv-size must be rejected at startup");
    let msg = format!("{err}");
    assert!(
        msg.contains("--max-kv-size"),
        "error must mention the flag name: got {msg:?}"
    );
}

#[test]
fn into_startup_config_rejects_below_min_max_kv_size() {
    let mut input = sample_input();
    // A non-zero value below the minimum is rejected (`0` is the documented
    // disabled sentinel and must keep being accepted).
    input.max_kv_size = 32;
    let err = input
        .into_startup_config()
        .expect_err("--max-kv-size below the minimum must be rejected at startup");
    let msg = format!("{err}");
    assert!(
        msg.contains("--max-kv-size") && msg.contains("minimum"),
        "error must explain the floor: got {msg:?}"
    );
}

#[test]
fn into_startup_config_accepts_zero_and_typical_max_kv_size() {
    // `0` lowers to `None` (disabled) and must not produce an error.
    let mut input = sample_input();
    input.max_kv_size = 0;
    let cfg = input.into_startup_config().expect("zero must be accepted");
    assert!(cfg.max_kv_size.is_none());

    // A typical non-zero value round-trips through `Option<usize>`.
    let mut input = sample_input();
    input.max_kv_size = 4096;
    let cfg = input
        .into_startup_config()
        .expect("typical value must be accepted");
    assert_eq!(cfg.max_kv_size, Some(4096));
}

// ---------------------------------------------------------------------------
// b10621 HTTP transport resolution (#1432)
// ---------------------------------------------------------------------------

#[test]
fn cors_credentials_env_can_disable_the_enabled_by_default_flag() {
    // clap resolves a flag's env binding by treating only a truthy value as an
    // occurrence, so a falsey `LLAMA_ARG_CORS_CREDENTIALS` would silently lose
    // to the `true` default. The runtime fallback is what makes it work.
    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_CORS_CREDENTIALS", "0");

    let mut value = true;
    super::env_fallback_cors_credentials(&mut value, false, false);
    assert!(
        !value,
        "LLAMA_ARG_CORS_CREDENTIALS=0 must disable credentials"
    );
}

#[test]
fn cors_credentials_env_is_overridden_by_an_explicit_flag() {
    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_CORS_CREDENTIALS", "0");

    let mut explicit = true;
    super::env_fallback_cors_credentials(&mut explicit, true, false);
    assert!(explicit, "--cors-credentials outranks the environment");

    let mut explicit_off = false;
    super::env_fallback_cors_credentials(&mut explicit_off, false, true);
    assert!(
        !explicit_off,
        "--no-cors-credentials outranks the environment"
    );
}

#[test]
fn cors_credentials_env_can_enable_the_flag() {
    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_CORS_CREDENTIALS", "true");

    let mut value = false;
    super::env_fallback_cors_credentials(&mut value, false, false);
    assert!(value);
}

#[test]
fn into_startup_config_splits_the_socket_and_decode_timeouts() {
    let mut input = sample_input();
    input.timeout = 7;
    input.decode_timeout = 900;
    let cfg = input.into_startup_config().expect("valid");
    assert_eq!(cfg.http_timeout, 7, "--timeout is the socket budget");
    assert_eq!(
        cfg.decode_timeout, 900,
        "--decode-timeout is the decode watchdog"
    );
}

#[test]
fn into_startup_config_rejects_an_unusable_api_prefix() {
    let mut input = sample_input();
    input.api_prefix = "llama".to_string();
    let err = input
        .into_startup_config()
        .expect_err("a prefix without a leading slash must fail startup");
    assert!(format!("{err}").contains("--api-prefix"), "{err}");
}

#[test]
fn into_startup_config_rejects_half_a_tls_configuration() {
    let mut input = sample_input();
    input.ssl_cert_file = Some(PathBuf::from("cert.pem"));
    let err = input
        .into_startup_config()
        .expect_err("a certificate without a key must fail startup");
    assert!(format!("{err}").contains("--ssl-key-file"), "{err}");
}

#[test]
fn into_startup_config_builds_the_b10621_cors_policy() {
    let mut input = sample_input();
    input.cors_origins = "localhost".to_string();
    let cfg = input.into_startup_config().expect("valid");
    assert_eq!(
        cfg.cors_policy.origins,
        crate::server::OriginPolicy::Localhost
    );
}

#[test]
fn into_startup_config_prefers_the_mlxcel_allow_list_when_it_is_set() {
    let mut input = sample_input();
    input.allowed_origins = vec!["https://app.example.com".to_string()];
    let cfg = input.into_startup_config().expect("valid");
    assert!(matches!(
        cfg.cors_policy.origins,
        crate::server::OriginPolicy::AllowList(_)
    ));
}

// ---------------------------------------------------------------------------
// b10621 API-key environment bindings (#1437)
// ---------------------------------------------------------------------------

#[test]
fn llama_api_key_env_adds_to_the_cli_keys_rather_than_shadowing_them() {
    // b10621 applies every environment variable before the command line and
    // both call the same appending handler, so a server started with
    // `LLAMA_API_KEY=env --api-key cli` accepts BOTH. clap's own `env`
    // attribute cannot express that, which is why the fallback exists.
    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_API_KEY", "env-key");

    let mut values = vec!["cli-key".to_string()];
    super::env_fallback_api_keys(&mut values);
    assert_eq!(values, vec!["env-key".to_string(), "cli-key".to_string()]);

    let resolved = crate::server::resolve_api_keys(&values, &[]).expect("valid");
    assert!(resolved.accepts("env-key"));
    assert!(resolved.accepts("cli-key"));
}

#[test]
fn llama_api_key_env_alone_configures_a_key() {
    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_API_KEY", "solo-key,second");

    let mut values = Vec::new();
    super::env_fallback_api_keys(&mut values);
    let resolved = crate::server::resolve_api_keys(&values, &[]).expect("valid");
    assert_eq!(resolved.len(), 2);
    assert!(resolved.accepts("solo-key") && resolved.accepts("second"));
}

#[test]
fn llama_arg_api_key_file_env_adds_to_the_cli_files() {
    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_API_KEY_FILE", "/etc/mlxcel/env-keys");

    let mut values = vec![PathBuf::from("/etc/mlxcel/cli-keys")];
    super::env_fallback_api_key_files(&mut values);
    assert_eq!(
        values,
        vec![
            PathBuf::from("/etc/mlxcel/env-keys"),
            PathBuf::from("/etc/mlxcel/cli-keys"),
        ]
    );
}

#[test]
fn into_startup_config_carries_every_api_key_source() {
    let mut input = sample_input();
    input.api_keys = vec!["a,b".to_string()];
    input.api_key_files = Vec::new();
    let cfg = input.into_startup_config().expect("valid");
    assert_eq!(cfg.api_keys, vec!["a,b".to_string()]);
    assert!(cfg.api_key_files.is_empty());
}

// ── b10621 reasoning template kwargs (issue #1447) ──────────────────────────

mod reasoning_template_kwargs {
    use super::super::apply_reasoning_template_kwargs;
    use crate::server::chat_template_kwargs::ChatTemplateKwargs;
    use serde_json::{Value, json};

    fn base(pairs: &[(&str, Value)]) -> Option<ChatTemplateKwargs> {
        if pairs.is_empty() {
            return None;
        }
        let mut kwargs = ChatTemplateKwargs::new();
        for (key, value) in pairs {
            kwargs.set(key, value.clone());
        }
        Some(kwargs)
    }

    fn apply(
        start: &[(&str, Value)],
        flags: &[(&str, Option<Value>)],
    ) -> Option<ChatTemplateKwargs> {
        let flags: Vec<(String, Option<Value>)> = flags
            .iter()
            .map(|(k, v)| ((*k).to_owned(), v.clone()))
            .collect();
        apply_reasoning_template_kwargs(base(start), &flags)
    }

    #[test]
    fn no_flags_leaves_the_base_untouched() {
        assert!(apply(&[], &[]).is_none());
        let kept = apply(&[("a", json!(1))], &[]).expect("base survives");
        assert_eq!(kept.get("a"), Some(&json!(1)));
    }

    #[test]
    fn a_flag_writes_its_key_onto_an_absent_base() {
        let merged = apply(&[], &[("enable_thinking", Some(json!(true)))]).expect("some");
        assert_eq!(merged.get("enable_thinking"), Some(&json!(true)));
    }

    #[test]
    fn a_flag_wins_over_the_same_key_from_chat_template_kwargs() {
        // b10621 applies whichever handler ran last and deprecates setting
        // `enable_thinking` through the kwargs, so the dedicated flag wins.
        let merged = apply(
            &[("enable_thinking", json!(true)), ("other", json!("keep"))],
            &[("enable_thinking", Some(json!(false)))],
        )
        .expect("some");
        assert_eq!(merged.get("enable_thinking"), Some(&json!(false)));
        assert_eq!(
            merged.get("other"),
            Some(&json!("keep")),
            "unrelated kwargs must survive"
        );
    }

    #[test]
    fn an_erase_removes_the_key_rather_than_setting_a_sentinel() {
        // `--reasoning-effort default` calls
        // `default_template_kwargs.erase("reasoning_effort")` upstream. A
        // template testing `reasoning_effort is defined` must see it undefined,
        // so a null sentinel would not do.
        let merged = apply(
            &[("reasoning_effort", json!("high")), ("other", json!(1))],
            &[("reasoning_effort", None)],
        )
        .expect("some");
        assert_eq!(merged.get("reasoning_effort"), None);
        assert_eq!(merged.get("other"), Some(&json!(1)));
    }

    #[test]
    fn erasing_a_key_that_was_never_set_is_a_no_op() {
        let merged = apply(&[("other", json!(1))], &[("reasoning_effort", None)]).expect("some");
        assert_eq!(merged.get("reasoning_effort"), None);
        assert_eq!(merged.get("other"), Some(&json!(1)));
    }

    #[test]
    fn a_per_request_kwarg_still_wins_over_both() {
        // The layering here is the SERVER default; the request map is applied
        // over it by `merge_server_and_request`, which is the precedence the
        // whole chain depends on.
        let server = apply(&[], &[("enable_thinking", Some(json!(false)))]);
        let mut per_request = ChatTemplateKwargs::new();
        per_request.set("enable_thinking", json!(true));
        let merged = crate::server::chat_template_kwargs::merge_server_and_request(
            server.as_ref(),
            &per_request,
        );
        assert_eq!(
            merged.get("enable_thinking"),
            Some(&json!(true)),
            "a per-request kwarg outranks both the flag and --chat-template-kwargs"
        );
    }
}
