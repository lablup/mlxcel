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
        model_path: PathBuf::from("models/foo"),
        adapter_path: Some(PathBuf::from("adapters/bar")),
        model_alias: Some("alias".to_string()),
        host: "127.0.0.1".to_string(),
        port: 8080,
        api_key: Some("secret".to_string()),
        api_key_file: Some(PathBuf::from("api.key")),
        n_parallel: 2,
        ctx_size: 4096,
        n_predict: 256,
        timeout: 600,
        draft_model_path: Some(PathBuf::from("models/draft")),
        draft_max: 8,
        // speculative-decoding selector flags default off
        // (None = auto-detect at dispatch time when a drafter is set).
        draft_kind: None,
        draft_block_size: None,
        max_batch_size: Some(4),
        max_queue_depth: 32,
        prefill_chunk_size: 512,
        batch_size: None,
        ubatch_size: None,
        enable_preemption: false,
        preemption_policy: "longest-first".to_string(),
        no_batch: false,
        max_batch_prefill: 1,
        decode_storage_backend: None,
        chat_template: Some("{{ prompt }}".to_string()),
        chat_template_file: Some(PathBuf::from("chat.jinja")),
        slots: true,
        no_slots: false,
        props: true,
        metrics: true,
        warmup: true,
        no_warmup: false,
        temperature: 0.8,
        top_k: 40,
        top_p: 0.9,
        min_p: 0.1,
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
        distributed_config: None,
        node_role: None,
        node_id: None,
        peers: Vec::new(),
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
        // APC knobs — disabled by default.
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
        // Responses API store defaults.
        responses_store_max_entries: 1024,
        responses_store_ttl_secs: 3600,
        conversation_store_max_entries: 256,
        conversation_store_ttl_secs: 3600,
        // (A4): default to None for baseline-path tests.
        #[cfg(feature = "surgery")]
        surgery_config_path: None,
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

/// `LLAMA_ARG_CACHE_REUSE=on` enables the cache (llama.cpp compat).
#[test]
fn prompt_cache_llama_arg_cache_reuse_on_sets_true() {
    use super::env_fallback_prompt_cache_enabled;

    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_CACHE_REUSE", "on");

    let mut enabled = false;
    env_fallback_prompt_cache_enabled(&mut enabled, false);
    assert!(enabled, "LLAMA_ARG_CACHE_REUSE=on must set enabled=true");
}

/// `LLAMA_ARG_CACHE_REUSE=off` disables the cache (llama.cpp compat).
#[test]
fn prompt_cache_llama_arg_cache_reuse_off_sets_false() {
    use super::env_fallback_prompt_cache_enabled;

    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_CACHE_REUSE", "off");

    let mut enabled = true;
    env_fallback_prompt_cache_enabled(&mut enabled, false);
    assert!(!enabled, "LLAMA_ARG_CACHE_REUSE=off must set enabled=false");
}

/// When both `MLXCEL_PROMPT_CACHE_ENABLED` and `LLAMA_ARG_CACHE_REUSE` are
/// set, the `MLXCEL_` key takes precedence.
#[test]
fn prompt_cache_mlxcel_env_wins_over_llama_arg_cache_reuse() {
    use super::env_fallback_prompt_cache_enabled;

    let _env_guard = env_lock();
    let _mlxcel = EnvGuard::set("MLXCEL_PROMPT_CACHE_ENABLED", "true");
    let _llama = EnvGuard::set("LLAMA_ARG_CACHE_REUSE", "off");

    let mut enabled = false;
    env_fallback_prompt_cache_enabled(&mut enabled, false);
    assert!(
        enabled,
        "MLXCEL_PROMPT_CACHE_ENABLED must win over LLAMA_ARG_CACHE_REUSE"
    );
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

/// `LLAMA_ARG_CACHE_REUSE=0` disables the cache (numeric form).
#[test]
fn prompt_cache_llama_arg_cache_reuse_zero_sets_false() {
    use super::env_fallback_prompt_cache_enabled;

    let _env_guard = env_lock();
    let _guard = EnvGuard::set("LLAMA_ARG_CACHE_REUSE", "0");

    let mut enabled = true;
    env_fallback_prompt_cache_enabled(&mut enabled, false);
    assert!(!enabled, "LLAMA_ARG_CACHE_REUSE=0 must set enabled=false");
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
