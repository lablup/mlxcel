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

use mlxcel::cli::speculative_args::{env_fallback_draft_block_size, env_fallback_draft_kind};
use mlxcel::cli::turbo_args::resolve_kv_cache_mode;
use mlxcel::downloader::resolve_model_source_with_override;
use mlxcel::memory_estimate::{QuantHint, estimate_total_memory, format_bytes, format_estimate};
use mlxcel::server::{
    ServerStartupInput, env_fallback_apc_block_size, env_fallback_apc_enabled,
    env_fallback_apc_hash, env_fallback_apc_num_blocks, env_fallback_cache_type_k,
    env_fallback_cache_type_v, env_fallback_chat_template_kwargs, env_fallback_kv_bits,
    env_fallback_kv_group_size, env_fallback_kv_quant_scheme, env_fallback_kv_skip_last_layer,
    env_fallback_lang_bias, env_fallback_lang_bias_include_byte_fragments,
    env_fallback_prompt_cache_capacity_bytes, env_fallback_prompt_cache_enabled,
    env_fallback_prompt_cache_max_entries, env_fallback_prompt_cache_min_prefix,
    env_fallback_prompt_cache_ttl, env_fallback_reasoning_budget, long_cli_flag_was_set,
    resolve_parallel_context_size, start_server,
};
use mlxcel_core::cache::KVCacheMode;

/// Run the `mlxcel serve` subcommand.
#[tokio::main]
pub(crate) async fn run_serve(mut args: crate::ServeArgs) -> anyhow::Result<()> {
    // Resolve `-m` into a concrete model directory (epic #92, issue #94)
    // before the memory preflight or the server reads it. An existing path is
    // used verbatim (byte-identical to the pre-#94 local-path behavior); an
    // `owner/name` HuggingFace repo-id is reused from the legacy CWD / HF cache
    // / mlxcel store, or auto-downloaded into the mlxcel store on a miss. Done
    // here (not in `build_startup_input`) so the preflight estimate also sees
    // the resolved path.
    args.model = resolve_model_source_with_override(&args.model, args.models_dir.as_deref())?;

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

    let kv_cache_mode = resolve_kv_cache_mode(
        args.turbo.cache_type_k.as_deref(),
        args.turbo.cache_type_v.as_deref(),
        args.turbo.kv_cache_mode.as_deref(),
    )
    .map_err(|e| anyhow::anyhow!("{}", e))?;
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
    env_fallback_reasoning_budget(&mut args.reasoning_budget);
    // env-var fallback for the chat-template kwargs default.
    env_fallback_chat_template_kwargs(&mut args.chat_template_kwargs);
    // env-var fallbacks for prompt-cache knobs.
    env_fallback_prompt_cache_enabled(
        &mut args.prompt_cache_enabled,
        long_cli_flag_was_set("prompt-cache-enabled"),
    );
    env_fallback_prompt_cache_capacity_bytes(&mut args.prompt_cache_capacity_bytes);
    env_fallback_prompt_cache_max_entries(&mut args.prompt_cache_max_entries);
    env_fallback_prompt_cache_ttl(&mut args.prompt_cache_ttl);
    env_fallback_prompt_cache_min_prefix(&mut args.prompt_cache_min_prefix);
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

    // Axis B (B8): resolve --lang-bias / --lang-bias-config early so
    // errors surface before the server starts. Empty resolution = None =
    // baseline bit-exact path.
    let lang_bias_config = args
        .lang_bias
        .resolve()
        .map_err(|e| anyhow::anyhow!("--lang-bias: {e}"))?;

    Ok(ServerStartupInput {
        model_path: args.model,
        adapter_path: args.adapter,
        model_alias: args.alias,
        host: args.host,
        port: args.port,
        api_key: args.api_key,
        api_key_file: args.api_key_file,
        n_parallel: args.n_parallel,
        ctx_size: args.ctx_size,
        n_predict: args.n_predict,
        timeout: args.timeout,
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
        prefill_chunk_size: args.prefill_chunk_size,
        batch_size: args.batch_size,
        ubatch_size: args.ubatch_size,
        enable_preemption: args.enable_preemption,
        preemption_policy: args.preemption_policy,
        max_batch_prefill: args.max_batch_prefill,
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
        top_k: args.top_k,
        top_p: args.top_p,
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
        prompt_cache_enabled: args.prompt_cache_enabled,
        prompt_cache_capacity_bytes: args.prompt_cache_capacity_bytes,
        prompt_cache_max_entries: args.prompt_cache_max_entries,
        prompt_cache_ttl_seconds: args.prompt_cache_ttl,
        prompt_cache_min_prefix: args.prompt_cache_min_prefix,
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
    })
}

#[cfg(test)]
#[path = "serve_tests.rs"]
mod tests;
