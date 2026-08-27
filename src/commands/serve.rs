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

//! CLI `serve` command wiring.
//!
//! This module owns the binary-only translation from clap-facing arguments to
//! the normalized server startup input. The actual llama-server compatibility
//! rules live in `mlxcel::server::ServerStartupInput` so `main.rs` stays focused
//! on schema and routing.

use anyhow::Context;
use mlxcel::cli::speculative_args::{env_fallback_draft_block_size, env_fallback_draft_kind};
use mlxcel::cli::turbo_args::resolve_kv_cache_mode;
use mlxcel::downloader::resolve_model_source_with_override;
use mlxcel::memory_estimate::{QuantHint, estimate_total_memory, format_bytes, format_estimate};
use mlxcel::server::{
    ServerStartupInput, env_fallback_apc_block_size, env_fallback_apc_enabled,
    env_fallback_apc_hash, env_fallback_apc_num_blocks, env_fallback_api_key_files,
    env_fallback_api_keys, env_fallback_batch_size, env_fallback_cache_type_k,
    env_fallback_cache_type_v, env_fallback_chat_template_kwargs, env_fallback_cors_credentials,
    env_fallback_draft_model, env_fallback_embedding_model, env_fallback_endpoint_slots,
    env_fallback_kv_bits, env_fallback_kv_group_size, env_fallback_kv_quant_scheme,
    env_fallback_kv_skip_last_layer, env_fallback_lang_bias,
    env_fallback_lang_bias_include_byte_fragments, env_fallback_log_file,
    env_fallback_prompt_cache_capacity_bytes, env_fallback_prompt_cache_enabled,
    env_fallback_prompt_cache_max_entries, env_fallback_prompt_cache_min_prefix,
    env_fallback_prompt_cache_snapshot_capacity_bytes,
    env_fallback_prompt_cache_snapshot_max_entries, env_fallback_prompt_cache_snapshot_ttl,
    env_fallback_prompt_cache_ttl, env_fallback_reasoning_budget, env_fallback_reranker_model,
    env_fallback_ubatch_size, long_cli_flag_was_set, resolve_parallel_context_size, start_server,
};
use mlxcel_core::cache::KVCacheMode;

/// Run the `mlxcel serve` subcommand.
///
/// The Tokio runtime is built here rather than by `#[tokio::main]` so
/// `--threads-http` can size its worker pool (#1432); llama-server b10621
/// sizes its own HTTP thread pool from the same flag. Everything below runs
/// inside that runtime, so the process has exactly one.
pub(crate) fn run_serve(args: crate::ServeArgs) -> anyhow::Result<()> {
    let workers =
        mlxcel::server::transport::resolve_http_threads(args.threads_http, args.n_parallel);
    let runtime = mlxcel::server::transport::build_http_runtime(workers)
        .context("failed to build the HTTP runtime; check --threads-http")?;
    runtime.block_on(run_serve_async(args))
}

async fn run_serve_async(mut args: crate::ServeArgs) -> anyhow::Result<()> {
    // Resolve `-m` into a concrete model directory (epic #92, issue #94)
    // before the memory preflight or the server reads it. An existing path is
    // used verbatim (byte-identical to the pre-#94 local-path behavior); an
    // `owner/name` HuggingFace repo-id is reused from the legacy CWD / HF cache
    // / mlxcel store, or auto-downloaded into the mlxcel store on a miss. Done
    // here (not in `build_startup_input`) so the preflight estimate also sees
    // the resolved path.
    args.model = resolve_model_source_with_override(
        &args.model,
        args.models_dir.as_deref(),
        args.revision.as_deref(),
    )?;

    // Issue #56: preflight memory check before the server begins
    // accepting connections. Refuses to start when total > available
    // unless --force was passed. Skipped when --estimate-memory is
    // off.
    run_serve_memory_preflight(&args)?;

    start_server(build_startup_input(args)?.into_startup_config()?).await
}

/// Run the `--estimate-memory` preflight for `mlxcel serve`.
///
/// Mirrors `commands::generate::run_memory_preflight` but routed off
/// `ServeArgs`. Uses the configured `ctx_size` when nonzero, falling
/// back to the estimator's default 8192 sizing otherwise (matching
/// what `--recommend-quant` historically used). When `total >
/// available` and `--force` was not set, returns `Err` so the server
/// aborts before any worker thread is spawned.
fn run_serve_memory_preflight(args: &crate::ServeArgs) -> anyhow::Result<()> {
    if !args.estimate_memory {
        return Ok(());
    }

    let requested = resolve_kv_cache_mode(
        args.turbo.cache_type_k.as_deref(),
        args.turbo.cache_type_v.as_deref(),
        args.turbo.kv_cache_mode.as_deref(),
    )
    .map_err(|e| anyhow::anyhow!("{}", e))?;
    // Estimate against the mode the caches will really be built with (issue
    // #1350), not the one requested. `--estimate-memory --kv-cache-mode int8`
    // on an MLA latent family resolves back to fp16 at startup, so sizing an
    // int8 cache here under-counts the KV budget, and this preflight can abort
    // the server outright when the total exceeds the available memory.
    // `run_serve` resolved `-m` to a concrete directory before calling us, so
    // the `model_type` lookup is a plain file read. Resolving quietly:
    // `into_startup_config` performs the same substitution and reports it once
    // logging exists.
    let (kv_cache_mode, _) =
        mlxcel::cli::turbo_args::resolve_effective_kv_cache_mode(requested, &args.model);
    let kv_int8 = matches!(kv_cache_mode, KVCacheMode::Int8);

    let ctx_len = serve_preflight_ctx_len(args);
    let batch = serve_preflight_batch(args);

    let estimate = estimate_total_memory(&args.model, ctx_len, batch, QuantHint::Default, kv_int8);

    let banner = format_estimate(&args.model, &estimate);
    println!("{banner}");

    if !estimate.fits {
        if args.force_memory {
            eprintln!(
                "WARNING: --estimate-memory preflight says this load is over budget by {}. \
                 Continuing because --force was set.",
                format_bytes(estimate.overflow_bytes()),
            );
        } else {
            return Err(anyhow::anyhow!(
                "--estimate-memory: total {} exceeds available {} by {}. \
                 Pass --force (or --no-memory-check) to override, or rerun with \
                 a smaller --ctx-size, smaller --max-batch-size, or a smaller model.",
                format_bytes(estimate.total_bytes),
                format_bytes(estimate.available_bytes),
                format_bytes(estimate.overflow_bytes()),
            ));
        }
    }

    Ok(())
}

fn serve_preflight_ctx_len(args: &crate::ServeArgs) -> u64 {
    // `--ctx-size 0` is the "use model default" sentinel; in that case we
    // fall back to 8192 to match the historical sizing used by
    // `--recommend-quant`. Explicit `--ctx-size` is a total budget shared by
    // active slots, matching llama.cpp server semantics. `--max-kv-size`
    // caps the plain KV cache length after the per-slot window is resolved.
    let mut ctx_len = if args.ctx_size > 0 {
        resolve_parallel_context_size(
            args.ctx_size,
            args.n_parallel,
            args.max_batch_size,
            args.no_batch,
        ) as u64
    } else {
        mlxcel::memory_estimate::DEFAULT_CTX_LEN
    };
    if args.max_kv_size > 0 {
        ctx_len = ctx_len.min(args.max_kv_size as u64);
    }
    ctx_len.max(1)
}

fn serve_preflight_batch(args: &crate::ServeArgs) -> u64 {
    if args.no_batch {
        return 1;
    }
    let active_sequences = args.max_batch_size.unwrap_or(args.n_parallel).max(1);
    u64::try_from(active_sequences).unwrap_or(u64::MAX)
}

fn build_startup_input(mut args: crate::ServeArgs) -> anyhow::Result<ServerStartupInput> {
    // Translate `--turbo-boundary-v` into the `MLXCEL_KV_BOUNDARY_V_LAYERS`
    // env var before any caller of `mlxcel-core` constructs a cache.
    // mlxcel-core reads this env var on first cache instantiation, and the
    // write site must be upstream of any code that spawns tasks reading the
    // process environment. The tokio worker threads spawned by
    // `#[tokio::main]` are still parked at this point (no task has been
    // scheduled yet), so the only env reader is this thread. See the
    // function-level SAFETY note on `TurboKvCacheArgs::apply_to_environment`
    // for the full precondition.
    args.turbo.apply_to_environment();

    // Axis B (B7): apply `LLAMA_ARG_LANG_BIAS` env-var fallback
    // before resolving, so env-supplied values flow through the same
    // validation path as CLI flags. CLI flag wins on conflict.
    env_fallback_lang_bias(&mut args.lang_bias);
    // env-var fallback for the byte-fragment opt-in flag.
    env_fallback_lang_bias_include_byte_fragments(&mut args.lang_bias);
    // env-var fallback for the thinking-budget default.
    env_fallback_reasoning_budget(
        &mut args.reasoning_budget,
        long_cli_flag_was_set("reasoning-budget"),
    );
    // env-var fallback for the chat-template kwargs default.
    env_fallback_chat_template_kwargs(&mut args.chat_template_kwargs);
    // env-var fallbacks for prompt-cache knobs.
    env_fallback_prompt_cache_enabled(
        &mut args.prompt_cache_enabled,
        long_cli_flag_was_set("prompt-cache-enabled"),
    )
    .map_err(anyhow::Error::msg)?;
    env_fallback_prompt_cache_capacity_bytes(&mut args.prompt_cache_capacity_bytes);
    env_fallback_prompt_cache_max_entries(&mut args.prompt_cache_max_entries);
    env_fallback_prompt_cache_ttl(&mut args.prompt_cache_ttl);
    env_fallback_prompt_cache_min_prefix(&mut args.prompt_cache_min_prefix);
    env_fallback_prompt_cache_snapshot_capacity_bytes(
        &mut args.prompt_cache_snapshot_capacity_bytes,
    );
    env_fallback_prompt_cache_snapshot_max_entries(&mut args.prompt_cache_snapshot_max_entries);
    env_fallback_prompt_cache_snapshot_ttl(&mut args.prompt_cache_snapshot_ttl);
    // env-var fallbacks for the APC knobs.
    env_fallback_apc_enabled(&mut args.apc_enabled, long_cli_flag_was_set("apc-enabled"));
    env_fallback_apc_block_size(&mut args.apc_block_size);
    env_fallback_apc_num_blocks(&mut args.apc_num_blocks);
    env_fallback_apc_hash(&mut args.apc_hash);
    // (B11): env-var fallbacks for KV cache type split flags.
    // The clap `env = "..."` attribute already reads these env vars; the
    // explicit calls below maintain the warn-on-conflict pattern used by
    // other LLAMA_ARG_* pairs.
    env_fallback_cache_type_k(&mut args.turbo.cache_type_k);
    env_fallback_cache_type_v(&mut args.turbo.cache_type_v);
    // env-var fallbacks for the continuous-batching KV
    // quantization knobs. The flags themselves live in
    // `mlxcel::cli::batch_quant_args::BatchKvQuantArgs` (flattened on
    // `ServeArgs`); these helpers honor the warn-on-CLI-conflict pattern
    // shared with the other LLAMA_ARG_* env vars.
    env_fallback_kv_bits(&mut args.batch_quant.kv_bits);
    env_fallback_kv_group_size(&mut args.batch_quant.kv_group_size);
    env_fallback_kv_quant_scheme(&mut args.batch_quant.kv_quant_scheme);
    env_fallback_kv_skip_last_layer(&mut args.batch_quant.kv_skip_last_layer);

    // env-var fallbacks for the speculative-decoding selector
    // flags. `clap` already reads `LLAMA_ARG_DRAFT_KIND` /
    // `LLAMA_ARG_DRAFT_BLOCK_SIZE` via the `env = "..."` attr on each flag;
    // the helpers below layer the mlxcel-native `MLXCEL_DRAFT_KIND` /
    // `MLXCEL_DRAFT_BLOCK_SIZE` aliases on top with the same warn-on-conflict
    // pattern shared with the other `MLXCEL_*` / `LLAMA_ARG_*` pairs.
    env_fallback_draft_kind(&mut args.speculative.draft_kind);
    env_fallback_draft_block_size(&mut args.speculative.draft_block_size);
    env_fallback_draft_model(&mut args.draft_model);
    env_fallback_batch_size(&mut args.batch_size);
    env_fallback_ubatch_size(&mut args.ubatch_size);
    env_fallback_log_file(&mut args.log_file);
    env_fallback_endpoint_slots(
        &mut args.slots,
        long_cli_flag_was_set("slots"),
        long_cli_flag_was_set("no-slots"),
    );
    env_fallback_api_keys(&mut args.api_key);
    env_fallback_api_key_files(&mut args.api_key_file);
    env_fallback_cors_credentials(
        &mut args.cors_credentials,
        long_cli_flag_was_set("cors-credentials"),
        long_cli_flag_was_set("no-cors-credentials"),
    );

    // Axis B (B8): resolve --lang-bias / --lang-bias-config early so
    // errors surface before the server starts. Empty resolution = None =
    // baseline bit-exact path.
    let lang_bias_config = args
        .lang_bias
        .resolve()
        .map_err(|e| anyhow::anyhow!("--lang-bias: {e}"))?;

    // `--embedding-model` accepts the same path-or-repo-id shapes as `-m`
    // and resolves through the same store lookup / auto-download.
    env_fallback_embedding_model(&mut args.embedding_model);
    let embedding_model_path = args
        .embedding_model
        .as_deref()
        .map(|value| {
            resolve_model_source_with_override(
                std::path::Path::new(value),
                args.models_dir.as_deref(),
                args.revision.as_deref(),
            )
        })
        .transpose()?;

    // `--reranker-model` accepts the same path-or-repo-id shapes as `-m`.
    env_fallback_reranker_model(&mut args.reranker_model);
    let reranker_model_path = args
        .reranker_model
        .as_deref()
        .map(|value| {
            resolve_model_source_with_override(
                std::path::Path::new(value),
                args.models_dir.as_deref(),
                args.revision.as_deref(),
            )
        })
        .transpose()?;

    Ok(ServerStartupInput {
        model_path: args.model,
        adapter_path: args.adapter,
        model_alias: args.alias,
        host: args.host,
        port: args.port,
        api_keys: args.api_key,
        api_key_files: args.api_key_file,
        n_parallel: args.n_parallel,
        ctx_size: args.ctx_size,
        n_predict: args.n_predict,
        // HTTP transport (#1432). `timeout` is now the socket read/write
        // budget; the decode watchdog moved to `--decode-timeout`. Both
        // "was set" flags include the environment binding, because a
        // deployment that only ever set `LLAMA_ARG_TIMEOUT` is exactly the one
        // whose meaning changed.
        timeout: args.timeout,
        timeout_was_set: long_cli_flag_was_set("timeout")
            || std::env::var_os("LLAMA_ARG_TIMEOUT").is_some(),
        decode_timeout: args.decode_timeout,
        decode_timeout_was_set: long_cli_flag_was_set("decode-timeout")
            || std::env::var_os("MLXCEL_DECODE_TIMEOUT").is_some(),
        api_prefix: args.api_prefix,
        sse_ping_interval: args.sse_ping_interval,
        threads_http: args.threads_http,
        reuse_port: args.reuse_port,
        ssl_cert_file: args.ssl_cert_file,
        ssl_key_file: args.ssl_key_file,
        cors_origins: args.cors_origins,
        cors_methods: args.cors_methods,
        cors_headers: args.cors_headers,
        cors_credentials: args.cors_credentials,
        no_cors_credentials: args._no_cors_credentials,
        draft_model_path: args.draft_model,
        draft_max: args.draft_max,
        // forward the speculative-decoding selector flags
        // resolved above via env-var fallbacks. Reconciliation into a
        // typed `DrafterKind` happens later, at the dispatch site.
        draft_kind: args.speculative.draft_kind,
        draft_block_size: args.speculative.draft_block_size,
        max_batch_size: args.max_batch_size,
        no_batch: args.no_batch,
        max_queue_depth: args.max_queue_depth,
        audio_queue_depth: args.audio_queue_depth,
        audio_request_timeout_secs: args.audio_request_timeout_secs,
        embedding_model_path,
        embedding_batch_size: args.embedding_batch_size,
        embedding_max_length: args.embedding_max_length,
        embedding_queue_depth: args.embedding_queue_depth,
        embedding_request_timeout_secs: args.embedding_request_timeout_secs,
        reranker_model_path,
        rerank_batch_size: args.rerank_batch_size,
        prefill_chunk_size: args.prefill_chunk_size,
        prefill_grant_interval: args.prefill_grant_interval,
        batch_size: args.batch_size,
        ubatch_size: args.ubatch_size,
        enable_preemption: args.enable_preemption,
        preemption_policy: args.preemption_policy,
        max_batch_prefill: args.max_batch_prefill,
        max_batch_prefill_tokens: args.max_batch_prefill_tokens,
        decode_storage_backend: args.decode_storage_backend,
        chat_template: args.chat_template,
        chat_template_file: args.chat_template_file,
        slots: args.slots,
        no_slots: args._no_slots,
        props: args.props,
        metrics: args.metrics,
        warmup: args.warmup,
        no_warmup: args._no_warmup,
        temperature: args.temp,
        temperature_was_set: long_cli_flag_was_set("temp") || long_cli_flag_was_set("temperature"),
        top_k: args.top_k,
        top_k_was_set: long_cli_flag_was_set("top-k")
            || std::env::var_os("LLAMA_ARG_TOP_K").is_some(),
        top_p: args.top_p,
        top_p_was_set: long_cli_flag_was_set("top-p"),
        min_p: args.min_p,
        seed: args.seed,
        repeat_last_n: args.repeat_last_n,
        repeat_penalty: args.repeat_penalty,
        presence_penalty: args.presence_penalty,
        frequency_penalty: args.frequency_penalty,
        dry_multiplier: args.dry_multiplier,
        dry_base: args.dry_base,
        dry_allowed_length: args.dry_allowed_length,
        dry_penalty_last_n: args.dry_penalty_last_n,
        dry_sequence_breakers: args.dry_sequence_breakers,
        verbose: args.verbose,
        log_disable: args.log_disable,
        log_file: args.log_file,
        distributed_config: args.distributed_config,
        node_role: args.node_role,
        node_id: args.node_id,
        peers: args.peers,
        prefill_peers: args.prefill_peers,
        decode_peers: args.decode_peers,
        serving_bind: args.serving_bind,
        pp_layers: args.pp_layers,
        pp_micro_batch_size: args.pp_micro_batch_size,
        pp_auto: args.pp_auto,
        pp_peer: args.pp_peer,
        cluster_discovery: args.cluster_discovery,
        cluster_name: args.cluster_name,
        cluster_peers: args.cluster_peers,
        cluster_discovery_port: args.cluster_discovery_port,
        cluster_control_addr: args.cluster_control_addr,
        cluster_config_out: args.cluster_config_out,
        dry_run: args.dry_run,
        tp_size: args.tp_size,
        tp_moe_mode: args.tp_moe_mode,
        tp_embedding_mode: args.tp_embedding_mode,
        tp_lm_head_mode: args.tp_lm_head_mode,
        vision_cache_size: args.vision_cache_size,
        max_image_payload_size: args.max_image_payload_size,
        max_images_per_request: args.max_images_per_request,
        max_image_width: args.max_image_width,
        max_image_height: args.max_image_height,
        max_image_decode_alloc_bytes: args.max_image_decode_alloc_bytes,
        enable_elastic_pp: args.enable_elastic_pp,
        elastic_pp_drain_timeout: args.elastic_pp_drain_timeout,
        elastic_pp_pressure_fraction: args.elastic_pp_pressure_fraction,
        elastic_pp_cool_down: args.elastic_pp_cool_down,
        metrics_port: args.metrics_port,
        debug_pp_trace: args.debug_pp_trace,
        lang_bias_config,
        reasoning_budget: args.reasoning_budget,
        chat_template_kwargs: args.chat_template_kwargs,
        // prompt-cache knobs already resolved via env-var fallbacks above.
        // `--no-prompt-cache` is the highest-precedence opt-out: it wins over
        // both `--prompt-cache-enabled` and the env-var fallbacks.
        prompt_cache_enabled: args.prompt_cache_enabled && !args.no_prompt_cache,
        prompt_cache_capacity_bytes: args.prompt_cache_capacity_bytes,
        prompt_cache_max_entries: args.prompt_cache_max_entries,
        prompt_cache_ttl_seconds: args.prompt_cache_ttl,
        prompt_cache_min_prefix: args.prompt_cache_min_prefix,
        prompt_cache_snapshot_capacity_bytes: args.prompt_cache_snapshot_capacity_bytes,
        prompt_cache_snapshot_max_entries: args.prompt_cache_snapshot_max_entries,
        prompt_cache_snapshot_ttl_seconds: args.prompt_cache_snapshot_ttl,
        // APC knobs already resolved via env-var fallbacks above.
        apc_enabled: args.apc_enabled,
        apc_block_size: args.apc_block_size,
        apc_num_blocks: args.apc_num_blocks,
        apc_hash: args.apc_hash,
        // (B11): KV cache type split flags already resolved via
        // env-var fallbacks (and clap `env = "..."`) above.
        cache_type_k: args.turbo.cache_type_k,
        cache_type_v: args.turbo.cache_type_v,
        kv_cache_mode_legacy: args.turbo.kv_cache_mode,
        // continuous-batching KV quantization knobs (flattened
        // from `BatchKvQuantArgs`).
        kv_bits: args.batch_quant.kv_bits,
        kv_group_size: args.batch_quant.kv_group_size,
        kv_quant_scheme: args.batch_quant.kv_quant_scheme,
        kv_skip_last_layer: args.batch_quant.kv_skip_last_layer,
        // maximum KV cache size for plain (non-sliding) caches.
        // clap reads `LLAMA_ARG_MAX_KV_SIZE` directly via the `env = ...`
        // attribute on the flag, so no separate env-fallback helper is needed.
        max_kv_size: args.max_kv_size,
        // paged KV pool block-budget directive (#122 b3). Already parsed by
        // clap into a `PagedBudgetDirective` (`Bytes`/`Auto`); resolved to a
        // block count on the worker thread.
        kv_cache_budget: args.kv_cache_budget,
        // experimental VLM prompt-prefix cache toggle (#124 step c).
        enable_vlm_prefix_cache: args.enable_vlm_prefix_cache,
        // CORS allow-list origins (#244); validated in into_startup_config.
        allowed_origins: args.allowed_origins,
        // Responses API in-memory store limits. clap reads the
        // matching `LLAMA_ARG_*` env vars directly via the `env = ...`
        // attributes on the flags.
        responses_store_max_entries: args.responses_store_max_entries,
        responses_store_ttl_secs: args.responses_store_ttl_secs,
        conversation_store_max_entries: args.conversation_store_max_entries,
        conversation_store_ttl_secs: args.conversation_store_ttl_secs,
        // (A4): forward the surgery YAML path. clap reads
        // `MLXCEL_SURGERY` directly via the `env = ...` attribute on
        // the flag, so no separate env-fallback helper is needed.
        #[cfg(feature = "surgery")]
        surgery_config_path: args.surgery,
        // serve-level block-diffusion knobs (#217 phase 3). Only diffusion
        // models read them; autoregressive models ignore them.
        max_denoising_steps: args.diffusion.max_denoising_steps,
        diffusion_sampler: args.diffusion.diffusion_sampler,
        diffusion_threshold: args.diffusion.diffusion_threshold,
        rope: args.rope.clone(),
    })
}

#[cfg(test)]
#[path = "serve_tests.rs"]
mod tests;
