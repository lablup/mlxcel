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

use super::{build_startup_input, serve_preflight_batch, serve_preflight_ctx_len};

fn sample_args() -> crate::ServeArgs {
    crate::ServeArgs {
        model: PathBuf::from("models/foo"),
        models_dir: None,
        adapter: Some(PathBuf::from("adapters/bar")),
        alias: Some("alias".to_string()),
        host: "127.0.0.1".to_string(),
        port: 9000,
        api_key: Some("secret".to_string()),
        api_key_file: Some(PathBuf::from("api.key")),
        n_parallel: 3,
        ctx_size: 8192,
        max_kv_size: 0,
        kv_cache_budget: None,
        enable_vlm_prefix_cache: false,
        allowed_origins: Vec::new(),
        n_predict: 128,
        draft_model: Some(PathBuf::from("models/draft")),
        draft_max: 4,
        max_batch_size: Some(4),
        no_batch: false,
        max_queue_depth: 32,
        audio_queue_depth: 8,
        audio_request_timeout_secs: 120,
        prefill_chunk_size: 512,
        batch_size: None,
        ubatch_size: None,
        enable_preemption: false,
        preemption_policy: "longest-first".to_string(),
        max_batch_prefill: 1,
        timeout: 30,
        chat_template: Some("{{ prompt }}".to_string()),
        chat_template_file: Some(PathBuf::from("chat.jinja")),
        slots: true,
        _no_slots: true,
        props: true,
        metrics: false,
        warmup: true,
        _no_warmup: true,
        temp: 0.7,
        top_k: 50,
        top_p: 0.95,
        min_p: 0.2,
        seed: -1,
        repeat_last_n: 32,
        repeat_penalty: 1.2,
        presence_penalty: 0.3,
        frequency_penalty: 0.4,
        dry_multiplier: 0.5,
        dry_base: 1.9,
        dry_allowed_length: 3,
        dry_penalty_last_n: -1,
        dry_sequence_breakers: vec!["\n".to_string(), "\t".to_string()],
        verbose: true,
        log_disable: false,
        log_file: Some(PathBuf::from("server.log")),
        distributed_config: None,
        node_role: None,
        node_id: None,
        peers: vec![],
        prefill_peers: vec![],
        decode_peers: vec![],
        serving_bind: None,
        pp_layers: None,
        pp_micro_batch_size: 1,
        pp_auto: None,
        pp_peer: false,
        cluster_discovery: "static".to_string(),
        cluster_name: None,
        cluster_peers: vec![],
        cluster_discovery_port: None,
        cluster_control_addr: None,
        cluster_config_out: None,
        dry_run: false,
        tp_size: 1,
        tp_moe_mode: "expert_parallel".to_string(),
        tp_embedding_mode: "replicated".to_string(),
        tp_lm_head_mode: "replicated".to_string(),
        _no_webui: false,
        _jinja: false,
        _n_gpu_layers: None,
        _mmproj: None,
        _flash_attn: false,
        _mlock: false,
        _no_mmap: false,
        _cont_batching: false,
        estimate_memory: false,
        force_memory: false,
        turbo: mlxcel::cli::turbo_args::TurboKvCacheArgs::default(),
        batch_quant: mlxcel::cli::batch_quant_args::BatchKvQuantArgs::default(),
        speculative: mlxcel::cli::speculative_args::SpeculativeArgs::default(),
        decode_storage_backend: None,
        vision_cache_size: 20,
        max_image_payload_size: mlxcel::server::DEFAULT_MAX_IMAGE_PAYLOAD_SIZE,
        max_images_per_request: mlxcel::server::DEFAULT_MAX_IMAGES_PER_REQUEST,
        max_image_width: mlxcel::server::DEFAULT_MAX_IMAGE_WIDTH,
        max_image_height: mlxcel::server::DEFAULT_MAX_IMAGE_HEIGHT,
        max_image_decode_alloc_bytes: mlxcel::server::DEFAULT_MAX_IMAGE_DECODE_ALLOC_BYTES,
        enable_elastic_pp: false,
        elastic_pp_drain_timeout: 120,
        elastic_pp_pressure_fraction: 0.92,
        elastic_pp_cool_down: 30,
        metrics_port: None,
        debug_pp_trace: None,
        lang_bias: mlxcel::lang_bias::LangBiasCliArgs::default(),
        reasoning_budget: -1,
        chat_template_kwargs: None,
        prompt_cache_enabled: true,
        no_prompt_cache: false,
        prompt_cache_capacity_bytes: None,
        prompt_cache_max_entries: None,
        prompt_cache_ttl: None,
        prompt_cache_min_prefix: None,
        apc_enabled: false,
        apc_block_size: None,
        apc_num_blocks: None,
        apc_hash: None,
        responses_store_max_entries: 1024,
        responses_store_ttl_secs: 3600,
        conversation_store_max_entries: 256,
        conversation_store_ttl_secs: 3600,
        // (A4): keep tests on the baseline path by default.
        #[cfg(feature = "surgery")]
        surgery: None,
        // serve-level diffusion knobs (#217 phase 3): engine defaults in tests.
        diffusion: crate::DiffusionServeOptions::default(),
    }
}

#[test]
fn build_startup_input_preserves_edge_flags_for_normalization() {
    let input = build_startup_input(sample_args()).expect("resolve");

    assert_eq!(input.model_path, PathBuf::from("models/foo"));
    assert_eq!(input.adapter_path, Some(PathBuf::from("adapters/bar")));
    assert_eq!(input.draft_model_path, Some(PathBuf::from("models/draft")));
    assert!(input.slots);
    assert!(input.no_slots);
    assert!(input.warmup);
    assert!(input.no_warmup);
    assert_eq!(input.seed, -1);
    assert_eq!(
        input.dry_sequence_breakers,
        vec!["\n".to_string(), "\t".to_string()]
    );
    assert_eq!(input.decode_storage_backend, None);
}

#[test]
fn serve_preflight_batch_uses_max_batch_size_when_batching_enabled() {
    let args = sample_args();
    assert_eq!(serve_preflight_batch(&args), 4);
}

#[test]
fn serve_preflight_batch_falls_back_to_parallelism_and_honors_no_batch() {
    let mut args = sample_args();
    args.max_batch_size = None;
    assert_eq!(serve_preflight_batch(&args), 3);

    args.no_batch = true;
    assert_eq!(serve_preflight_batch(&args), 1);
}

#[test]
fn serve_preflight_ctx_len_uses_default_and_max_kv_cap() {
    let mut args = sample_args();
    args.ctx_size = 0;
    assert_eq!(
        serve_preflight_ctx_len(&args),
        mlxcel::memory_estimate::DEFAULT_CTX_LEN
    );

    args.ctx_size = 8192;
    assert_eq!(serve_preflight_ctx_len(&args), 2048);

    args.ctx_size = 8192;
    args.max_kv_size = 2048;
    assert_eq!(serve_preflight_ctx_len(&args), 2048);
}

#[test]
fn serve_preflight_ctx_len_uses_parallel_context_semantics() {
    let mut args = sample_args();
    args.ctx_size = 4096;
    args.max_batch_size = None;
    args.n_parallel = 4;
    assert_eq!(serve_preflight_ctx_len(&args), 1024);

    args.max_batch_size = Some(2);
    assert_eq!(serve_preflight_ctx_len(&args), 2048);

    args.no_batch = true;
    assert_eq!(serve_preflight_ctx_len(&args), 4096);
}

#[test]
fn build_startup_input_propagates_decode_storage_backend() {
    let mut args = sample_args();
    args.decode_storage_backend = Some(mlxcel::server::DecodeStorageBackend::Paged);

    let input = build_startup_input(args).expect("resolve");
    assert_eq!(
        input.decode_storage_backend,
        Some(mlxcel::server::DecodeStorageBackend::Paged)
    );
}

#[test]
fn build_startup_input_propagates_image_limits() {
    let mut args = sample_args();
    args.max_image_payload_size = 1234;
    args.max_images_per_request = 3;
    args.max_image_width = 2048;
    args.max_image_height = 1024;
    args.max_image_decode_alloc_bytes = 16 * 1024 * 1024;

    let input = build_startup_input(args).expect("resolve");
    assert_eq!(input.max_image_payload_size, 1234);
    assert_eq!(input.max_images_per_request, 3);
    assert_eq!(input.max_image_width, 2048);
    assert_eq!(input.max_image_height, 1024);
    assert_eq!(input.max_image_decode_alloc_bytes, 16 * 1024 * 1024);
}

#[test]
fn build_startup_input_propagates_pp_layers() {
    let mut args = sample_args();
    args.pp_layers = Some("0-15,16-31".to_string());
    let input = build_startup_input(args).expect("resolve");
    assert_eq!(input.pp_layers, Some("0-15,16-31".to_string()));
}

#[test]
fn build_startup_input_pp_layers_none_by_default() {
    let input = build_startup_input(sample_args()).expect("resolve");
    assert_eq!(input.pp_layers, None);
}

// Axis B B8: default lang_bias args produce None config (baseline no-op).
#[test]
fn build_startup_input_lang_bias_defaults_to_none() {
    let input = build_startup_input(sample_args()).expect("resolve");
    assert!(
        input.lang_bias_config.is_none(),
        "no --lang-bias flag should yield None (baseline bit-exact)"
    );
}

// Axis B B8: a populated --lang-bias is resolved and threaded through.
#[test]
fn build_startup_input_lang_bias_propagates_when_active() {
    let mut args = sample_args();
    args.lang_bias = mlxcel::lang_bias::LangBiasCliArgs {
        lang_bias: Some("ja=-inf".to_string()),
        ..Default::default()
    };
    let input = build_startup_input(args).expect("resolve");
    let cfg = input.lang_bias_config.expect("resolved config");
    assert_eq!(cfg.bias_set.ordered.len(), 1);
}
