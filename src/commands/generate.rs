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

//! CLI text-generation command handler.
//!
//! This module keeps the user-facing `generate` flow readable by separating
//! prompt preparation, generation-mode selection, and terminal output helpers.

use anyhow::{Result, anyhow, ensure};
use std::io::{self, Write as IoWrite};
use std::path::Path;
use std::time::{Duration, Instant};

use mlxcel::{
    CxxGenerator, GenerationStats, LanguageModel, RuntimeSetup, SamplingConfig,
    SpeculativeGenerator,
    distributed::{
        PipelineWorkerInput, RequestId,
        pipeline::{
            load_in_process_stage_worker_with_adapter, resolve_in_process_pipeline_num_layers,
        },
        resolve_model_shard_plan, shard_config_from_cli, validate_supported_runtime,
    },
    downloader::resolve_model_source_with_override,
    initialize_runtime, load_model, load_model_with_adapter, load_model_with_tensor_parallel,
    memory_estimate::{
        MemoryEstimate, QuantHint, estimate_total_memory, format_bytes, format_estimate,
    },
    quant_advisor::{advise_quantization, print_quant_advice},
    sampling::{ResolvedSamplingParams, build_sampling_config},
    server::chat_template::{ChatMessage, ChatTemplateProcessor},
    tokenizer::load_tokenizer,
    vision::merge::InputEmbeddings,
    vlm_runtime::prepared_embedding_refs,
};
use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::generation_policy::{
    initial_token_history, merged_eos_token_ids, seed_rng_if_needed,
};
use mlxcel_core::lang_analyzer::LangBiasConfig;
use mlxcel_core::sampling::{TokenBiasMap, sample_token_optimized};

use mlxcel::cli::speculative_args::resolve_draft_block_size;
use mlxcel::cli::turbo_args::resolve_kv_cache_mode;
use mlxcel_core::drafter::{DrafterKind, resolve_drafter_kind};

use super::generate_vlm;
use crate::GenerateArgs;

fn generation_stats_from_duration(
    prompt_tokens: usize,
    generated_tokens: usize,
    total_time: Duration,
) -> GenerationStats {
    let decode_time_ms = total_time.as_secs_f64() * 1000.0;
    let decode_tok_per_sec = if total_time.as_secs_f64() > 0.0 {
        generated_tokens as f64 / total_time.as_secs_f64()
    } else {
        0.0
    };

    GenerationStats {
        prompt_tokens,
        generated_tokens,
        prefill_time_ms: 0.0,
        decode_time_ms,
        prefill_tok_per_sec: 0.0,
        decode_tok_per_sec,
    }
}

fn print_runtime_setup(runtime: &RuntimeSetup) {
    if let Some(invalid) = runtime.invalid_device_override.as_deref() {
        eprintln!(
            "Ignoring invalid MLXCEL_DEVICE value {:?}; using gpu.",
            invalid
        );
    }
    println!("Runtime device: {}", runtime.device);
    if let Some(max_memory) = runtime.wired_limit_bytes {
        println!(
            "Wired memory limit: {:.1} GB",
            max_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    } else if runtime.device == mlxcel::RuntimeDevice::Gpu {
        let max_memory = mlxcel_core::gpu_max_memory_size();
        println!(
            "GPU memory: {:.1} GB (no wired limit)",
            max_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }
    // Issue #55: surface the soft allocator cap when the operator set one
    // via MLXCEL_MEMORY_LIMIT, so the preflight intent is visible at boot.
    if let Some(memory_limit) = runtime.memory_limit_bytes {
        println!(
            "MLX allocator memory limit: {:.1} GB (MLXCEL_MEMORY_LIMIT)",
            memory_limit as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }
}

fn load_generation_model(
    args: &GenerateArgs,
    preflight: Option<&MemoryEstimate>,
) -> Result<(mlxcel::LoadedModel, mlxcel::tokenizer::MlxcelTokenizer)> {
    println!("Loading model from {:?}...", args.model.model);
    let load_start = Instant::now();
    let shard_config = shard_config_from_cli(
        args.tensor_parallel.tp_size,
        &args.tensor_parallel.tp_moe_mode,
        &args.tensor_parallel.tp_embedding_mode,
        &args.tensor_parallel.tp_lm_head_mode,
    )?;
    let result = if shard_config.tp_size > 1 {
        load_model_with_tensor_parallel(
            &args.model.model,
            args.model.adapter.as_deref(),
            &shard_config,
        )
    } else if let Some(ref adapter_path) = args.model.adapter {
        println!("Loading LoRA adapter from {:?}...", adapter_path);
        load_model_with_adapter(&args.model.model, adapter_path)
    } else {
        load_model(&args.model.model)
    }?;
    let load_elapsed = load_start.elapsed();
    // Issue #55: surface "resident after load" so operators (and the
    // capstone preflight #56) can see how much MLX-allocator memory the
    // model actually consumed once weight realisation finished. On
    // Apple Silicon (Metal) this reads from the Metal allocator; on
    // Linux/CUDA from the CUDA allocator; on CPU-only it reads from the
    // no-gpu common allocator. Each backend may use a different
    // definition of "active", but the number is always whatever MLX
    // itself will compare against `memory_limit()` next.
    let snap = mlxcel_core::memory::snapshot();
    println!(
        "Model loaded in {:.3}s (resident: {:.2} GB, peak: {:.2} GB).",
        load_elapsed.as_secs_f64(),
        snap.active_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        snap.peak_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    tracing::info!(
        active_bytes = snap.active_bytes,
        peak_bytes = snap.peak_bytes,
        cache_bytes = snap.cache_bytes,
        limit_bytes = snap.limit_bytes,
        load_seconds = load_elapsed.as_secs_f64(),
        "Model resident after load",
    );

    // Issue #56: compare the pre-load estimate against MLX's
    // observed active memory once loading is complete. The delta
    // feeds future headroom-factor calibration (see the recipe on
    // `memory_estimate::DEFAULT_HEADROOM_FACTOR`).
    //
    // On Linux/CPU MLX returns zero for most memory metrics, so we
    // skip the delta when `snap.active_bytes == 0`, it would just
    // print misleading "100% under-estimate" lines. The structural
    // wiring is verified by the call site and the unit tests; the
    // numerical delta is meaningful only on Apple Silicon (Metal) /
    // CUDA backends that populate the active counter.
    if let Some(est) = preflight {
        log_estimate_vs_actual_delta(est, &snap);
    }
    Ok(result)
}

/// Log the delta between a pre-load `MemoryEstimate` and the
/// post-load MLX allocator snapshot.
///
/// Skips when MLX reports zero active bytes (Linux/CPU has no
/// per-process allocator counter on the no-gpu backend). When active
/// bytes are nonzero, prints a `delta` line and emits a tracing
/// event so an off-line collector can chart preflight accuracy
/// across loads, feeding the manual recalibration recipe on
/// `DEFAULT_HEADROOM_FACTOR`.
fn log_estimate_vs_actual_delta(est: &MemoryEstimate, snap: &mlxcel_core::memory::MemorySnapshot) {
    if snap.active_bytes == 0 {
        // No allocator counter to compare against (no-gpu CPU
        // backend). Surface the no-op so operators reading the log
        // know the preflight estimate is structurally wired but
        // can't be validated numerically on this host.
        println!(
            "Memory estimate vs actual: skipped (MLX active_memory() is 0, \
             non-Metal/CUDA backend; estimate was {} and is structurally valid \
             but cannot be verified without a populated allocator counter)",
            format_bytes(est.total_bytes),
        );
        tracing::info!(
            estimate_total = est.total_bytes,
            actual_active = snap.active_bytes,
            skipped = true,
            reason = "active_memory zero on this backend",
            "Memory estimate vs actual delta",
        );
        return;
    }

    let est_bytes = est.total_bytes;
    let actual = snap.active_bytes;
    let (delta_label, delta_bytes) = estimate_delta_label_and_bytes(est_bytes, actual);
    let ratio = if est_bytes > 0 {
        actual as f64 / est_bytes as f64
    } else {
        0.0
    };
    println!(
        "Memory estimate vs actual: estimate {} | actual {} | {} {} (ratio {:.3})",
        format_bytes(est_bytes),
        format_bytes(actual),
        delta_label,
        format_bytes(delta_bytes),
        ratio,
    );
    tracing::info!(
        estimate_total = est_bytes,
        actual_active = actual,
        delta_bytes,
        ratio,
        headroom_factor = est.headroom_factor,
        weights_bytes = est.weights_bytes,
        kv_cache_bytes = est.kv_cache_bytes,
        runtime_headroom_bytes = est.runtime_headroom_bytes,
        "Memory estimate vs actual delta",
    );
}

fn estimate_delta_label_and_bytes(estimate: u64, actual: u64) -> (&'static str, u64) {
    if actual >= estimate {
        ("under-estimated by", actual.saturating_sub(estimate))
    } else {
        ("over-estimated by", estimate.saturating_sub(actual))
    }
}

fn memory_preflight_ctx_len(prompt_tokens: usize, max_tokens: usize) -> u64 {
    let total = prompt_tokens.saturating_add(max_tokens).max(1);
    u64::try_from(total).unwrap_or(u64::MAX)
}

/// Run the `--estimate-memory` preflight for `mlxcel generate`.
///
/// Returns `Some(estimate)` when the user passed `--estimate-memory`
/// (so the caller can later log the estimate-vs-actual delta), and
/// `None` when the preflight was not requested. The function never
/// allocates on MLX and never touches the model.
///
/// When `total > available` and `--force` was not set, returns
/// `Err(...)` with an actionable message that names the over-budget
/// figure and the override flags. Always prints the formatted
/// breakdown before aborting so operators can see the same byte
/// table `mlxcel inspect` would have shown.
fn run_memory_preflight(
    args: &GenerateArgs,
    prompt_token_count: usize,
) -> Result<Option<MemoryEstimate>> {
    if !args.generation.estimate_memory {
        return Ok(None);
    }

    let kv_cache_mode = resolve_kv_cache_mode(
        args.generation.turbo.cache_type_k.as_deref(),
        args.generation.turbo.cache_type_v.as_deref(),
        args.generation.turbo.kv_cache_mode.as_deref(),
    )
    .map_err(|e| anyhow::anyhow!("{}", e))?;
    let kv_int8 = matches!(kv_cache_mode, KVCacheMode::Int8);

    // Size the KV cache for the tokens that can actually enter the cache:
    // rendered prompt tokens plus the requested decode budget. This still runs
    // before model load, but after tokenizer/template processing has made the
    // prompt length knowable.
    let ctx_len = memory_preflight_ctx_len(prompt_token_count, args.generation.max_tokens);

    let estimate =
        estimate_total_memory(&args.model.model, ctx_len, 1, QuantHint::Default, kv_int8);

    let banner = format_estimate(&args.model.model, &estimate);
    println!("{banner}");

    if !estimate.fits {
        if args.generation.force_memory {
            eprintln!(
                "WARNING: --estimate-memory preflight says this load is over budget by {}. \
                 Continuing because --force was set.",
                format_bytes(estimate.overflow_bytes()),
            );
        } else {
            return Err(anyhow::anyhow!(
                "--estimate-memory: total {} exceeds available {} by {}. \
                 Pass --force (or --no-memory-check) to override, or rerun with \
                 a smaller --max-tokens / a smaller model.",
                format_bytes(estimate.total_bytes),
                format_bytes(estimate.available_bytes),
                format_bytes(estimate.overflow_bytes()),
            ));
        }
    }

    Ok(Some(estimate))
}

fn cli_pipeline_requested(args: &GenerateArgs) -> bool {
    args.pipeline_parallel.pp_size > 1 || args.pipeline_parallel.pp_layers.is_some()
}

fn validate_pipeline_parallel_args(args: &GenerateArgs) -> Result<()> {
    let pp = &args.pipeline_parallel;
    ensure!(
        pp.pp_micro_batch_size > 0,
        "--pp-micro-batch-size must be greater than 0"
    );
    if pp.pp_layers.is_none() && pp.pp_size <= 1 {
        return Ok(());
    }

    // 2D (PP x TP) composition is now supported. See
    // `docs/en/distributed/pipeline-parallelism.md` and
    // `docs/en/distributed/tensor-parallelism.md` for the operator guide.
    let tp_size = args.tensor_parallel.tp_size;
    if tp_size > 1 {
        ensure!(
            pp.pp_size >= 2 || pp.pp_layers.is_some(),
            "2D parallelism requires --pp-size >= 2 (or an explicit --pp-layers spec) \
             alongside --tensor-parallel-size > 1"
        );
        // Soft guard against obvious topology mistakes. A negative-like sanity
        // check here surfaces a clear error instead of a cryptic routing or
        // sharding failure later on. The full `pp_size * tp_size == nodes`
        // check is performed at the cluster-TOML validator layer for remote
        // topologies; here we only guard the local single-process 2D case.
        let total_ranks = (pp.pp_size as u64).saturating_mul(tp_size as u64);
        ensure!(
            total_ranks > 0,
            "inconsistent 2D topology: pp_size={} tp_size={}",
            pp.pp_size,
            tp_size
        );
    }
    // LoRA adapter composition with PP is supported, adapters are loaded at
    // stage initialization via `load_in_process_stage_worker_with_adapter`.
    // Single-adapter only; multi-adapter stacking and runtime hot-swap
    // remain out of scope for v1.
    ensure!(
        args.model.draft_model.is_none(),
        "CLI pipeline parallelism does not support speculative decoding yet"
    );
    ensure!(
        args.generation.image.is_empty()
            && args.generation.audio.is_none()
            && args.generation.video.is_empty(),
        "CLI pipeline parallelism currently supports text-only generation"
    );
    if let Some(spec) = pp.pp_layers.as_deref() {
        ensure!(
            !spec.trim().is_empty(),
            "--pp-layers must not be empty when provided"
        );
    } else {
        ensure!(
            pp.pp_size >= 2,
            "--pp-size must be at least 2 to enable pipeline parallelism"
        );
    }
    Ok(())
}

fn resolve_cli_pipeline_assignments(
    model_dir: &Path,
    num_layers: usize,
    args: &GenerateArgs,
) -> Result<Vec<mlxcel::distributed::StageAssignment>> {
    // Use the model-aware profile builder so MoE expert variation and
    // Gemma 4 KV-shared adjacency are honoured by default. This drops the
    // earlier requirement for manual `--pp-layers` on those models.
    let (assignments, report) =
        mlxcel::distributed::pipeline::resolve_in_process_stage_assignments_for_model(
            model_dir,
            num_layers,
            Some(args.pipeline_parallel.pp_size),
            args.pipeline_parallel.pp_layers.as_deref(),
        )?;
    mlxcel::distributed::pipeline::log_partition_quality(&report);
    Ok(assignments)
}

fn resolve_cli_pipeline_num_layers(model_dir: &Path) -> Result<usize> {
    resolve_in_process_pipeline_num_layers(model_dir).map_err(|err| anyhow!("{err}"))
}

fn generate_pipeline_text(
    model_dir: &Path,
    num_layers: usize,
    prompt_tokens: &[i32],
    max_tokens: usize,
    sampling_config: &SamplingConfig,
    args: &GenerateArgs,
) -> Result<(Vec<i32>, GenerationStats)> {
    let assignments = resolve_cli_pipeline_assignments(model_dir, num_layers, args)?;
    ensure!(
        assignments.len() >= 2,
        "pipeline execution requires at least 2 stages"
    );

    if let Some(ref adapter_path) = args.model.adapter {
        println!(
            "Loading LoRA adapter from {:?} across {} pipeline stages...",
            adapter_path,
            assignments.len(),
        );
    }
    let mut worker_loop = load_in_process_stage_worker_with_adapter(
        model_dir,
        &assignments,
        args.pipeline_parallel.pp_micro_batch_size,
        args.model.adapter.as_deref(),
    )?;

    let request_id = RequestId::new();
    let prompt_ids = mlxcel_core::from_slice_i32(prompt_tokens, &[1, prompt_tokens.len() as i32]);

    let prefill_start = Instant::now();
    let mut current_logits = worker_loop
        .run_to_completion(vec![PipelineWorkerInput::new(
            request_id.clone(),
            prompt_ids,
        )])?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("pipeline worker loop did not return a prefill output"))?
        .logits;
    let prefill_elapsed = prefill_start.elapsed();

    seed_rng_if_needed(sampling_config);
    let eos_token_ids = merged_eos_token_ids(
        mlxcel::read_eos_token_ids(model_dir),
        &sampling_config.stop_token_ids,
    );
    let mut token_history =
        initial_token_history(prompt_tokens, sampling_config.needs_token_history());
    let mut generated_tokens = Vec::with_capacity(max_tokens);
    let decode_start = Instant::now();

    for _ in 0..max_tokens {
        let (token_arr, _processed_logits) = sample_token_optimized(
            current_logits.as_ref().unwrap(),
            sampling_config,
            &token_history,
        );
        mlxcel_core::eval(&token_arr);
        let token_id = mlxcel_core::item_i32(&token_arr);
        generated_tokens.push(token_id);
        if sampling_config.needs_token_history() {
            token_history.push(token_id);
        }
        if eos_token_ids.contains(&token_id) {
            break;
        }

        let next_input = mlxcel_core::from_slice_i32(&[token_id], &[1, 1]);
        current_logits = worker_loop
            .run_to_completion(vec![PipelineWorkerInput::new(
                request_id.clone(),
                next_input,
            )])?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("pipeline worker loop did not return a decode output"))?
            .logits;
    }

    let decode_elapsed = decode_start.elapsed();
    let stats = GenerationStats {
        prompt_tokens: prompt_tokens.len(),
        generated_tokens: generated_tokens.len(),
        prefill_time_ms: prefill_elapsed.as_secs_f64() * 1000.0,
        decode_time_ms: decode_elapsed.as_secs_f64() * 1000.0,
        prefill_tok_per_sec: if prefill_elapsed.as_secs_f64() > 0.0 {
            prompt_tokens.len() as f64 / prefill_elapsed.as_secs_f64()
        } else {
            0.0
        },
        decode_tok_per_sec: if decode_elapsed.as_secs_f64() > 0.0 {
            generated_tokens.len() as f64 / decode_elapsed.as_secs_f64()
        } else {
            0.0
        },
    };

    Ok((generated_tokens, stats))
}

fn validate_tensor_parallel_args(args: &GenerateArgs) -> Result<()> {
    let shard_config = shard_config_from_cli(
        args.tensor_parallel.tp_size,
        &args.tensor_parallel.tp_moe_mode,
        &args.tensor_parallel.tp_embedding_mode,
        &args.tensor_parallel.tp_lm_head_mode,
    )?;
    let summary = resolve_model_shard_plan(&args.model.model, shard_config)?;
    if summary.shard_config.tp_size > 1 {
        println!("Tensor parallel request: {}", summary.summary_line());
    }
    validate_supported_runtime(
        &args.model.model,
        summary.shard_config.clone(),
        args.model.adapter.as_deref(),
    )
    .map(|_| ())
}

fn apply_user_chat_template(processor: &ChatTemplateProcessor, user_prompt: &str) -> String {
    let messages = [ChatMessage {
        role: "user".to_string(),
        content: user_prompt.to_string(),
    }];

    processor
        .apply(&messages, None)
        .unwrap_or_else(|_| user_prompt.to_string())
}

/// Apply chat template with image placeholders for VLM models.
///
/// Creates multimodal content entries that Gemma3-style templates can
/// render into `<start_of_image>` tokens (which are later expanded into
/// full image-token blocks by `apply_image_token_blocks`).
///
/// Only used when the template explicitly handles `type == 'image'`
/// content items. Templates without image support fall back to text-only.
fn apply_vlm_chat_template(
    processor: &ChatTemplateProcessor,
    user_prompt: &str,
    num_images: usize,
) -> String {
    // Only attempt multimodal rendering when the template handles image
    // content items.  Templates that don't (e.g. Vicuna, ChatML) would
    // render the raw JSON list as text, producing garbled output.
    if !processor.supports_image_content() {
        return apply_user_chat_template(processor, user_prompt);
    }

    // Build a multimodal content list: [{type: image}, ..., {type: text, text: prompt}]
    let mut content_parts: Vec<serde_json::Value> = Vec::new();
    for _ in 0..num_images {
        content_parts.push(serde_json::json!({"type": "image"}));
    }
    content_parts.push(serde_json::json!({"type": "text", "text": user_prompt}));

    let messages = serde_json::json!([{
        "role": "user",
        "content": content_parts,
    }]);

    processor.apply_raw(&messages, None).unwrap_or_else(|_| {
        // Fallback: text-only template
        apply_user_chat_template(processor, user_prompt)
    })
}

fn resolve_cli_prompt(
    user_prompt: &str,
    no_chat_template: bool,
    processor: Option<&ChatTemplateProcessor>,
    num_images: usize,
) -> String {
    if no_chat_template {
        return user_prompt.to_string();
    }

    processor.map_or_else(
        || user_prompt.to_string(),
        |processor| {
            if num_images > 0 {
                apply_vlm_chat_template(processor, user_prompt, num_images)
            } else {
                apply_user_chat_template(processor, user_prompt)
            }
        },
    )
}

fn load_cli_prompt(
    model_path: &Path,
    user_prompt: &str,
    no_chat_template: bool,
    num_images: usize,
) -> String {
    let processor = if no_chat_template {
        None
    } else {
        ChatTemplateProcessor::from_model_path(model_path)
            .ok()
            .flatten()
    };

    resolve_cli_prompt(
        user_prompt,
        no_chat_template,
        processor.as_ref(),
        num_images,
    )
}

fn tokenize_prompt(
    tokenizer: &mlxcel::tokenizer::MlxcelTokenizer,
    prompt: &str,
) -> Result<Vec<i32>> {
    // If the prompt already starts with a BOS token string (e.g. from a chat
    // template that embeds <bos>), skip add_special_tokens to avoid double-BOS.
    // Matches mlx-lm generate.py behaviour.
    let add_special = !prompt.starts_with("<bos>") && !prompt.starts_with("<s>");
    let prompt_token_ids = tokenizer
        .encode(prompt, add_special)
        .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
    Ok(prompt_token_ids.iter().map(|&x| x as i32).collect())
}

/// Resolve the parsed `LangBiasConfig` into a concrete [`TokenBiasMap`] for
/// the loaded tokenizer, or return an empty map when no language bias is
/// active.
///
/// The empty-map path is the **baseline bit-exact** contract
/// Axis B: no disk I/O, no vocab scan, no sampling-path changes.
///
/// # Errors
/// Returns an error when the tokenizer is not HuggingFace-compatible but the
/// user explicitly requested language bias. SentencePiece/Tiktoken tokenizers
/// are not supported by the lang_analyzer vocabulary scanner in Phase 1.
fn resolve_cli_token_bias(
    lang_bias_config: Option<&LangBiasConfig>,
    tokenizer: &mlxcel::tokenizer::MlxcelTokenizer,
    model_path: &Path,
) -> Result<TokenBiasMap> {
    let Some(cfg) = lang_bias_config else {
        return Ok(TokenBiasMap::default());
    };
    // Empty bias set is also a no-op: `resolve_token_bias` short-circuits,
    // but we short-circuit earlier here too to avoid any tokenizer I/O.
    if cfg.bias_set.ordered.is_empty() {
        return Ok(TokenBiasMap::default());
    }

    let hf = tokenizer.hf_tokenizer().ok_or_else(|| {
        anyhow::anyhow!(
            "--lang-bias requires a HuggingFace tokenizer.json; this model uses \
             a SentencePiece/Tiktoken tokenizer which is not supported by the \
             Axis B Phase 1 language analyzer"
        )
    })?;

    let json_path = model_path.join("tokenizer.json");
    let json_bytes = std::fs::read(&json_path).map_err(|e| {
        anyhow::anyhow!(
            "--lang-bias: failed to read tokenizer.json at {:?} for vocab-hash \
             cache key: {e}",
            json_path
        )
    })?;

    cfg.resolve_token_bias(hf, &json_bytes)
        .map_err(|e| anyhow::anyhow!("--lang-bias: resolve failed: {e}"))
}

fn build_cli_sampling_config(args: &GenerateArgs, stop_token_ids: Vec<i32>) -> SamplingConfig {
    build_sampling_config(ResolvedSamplingParams {
        temperature: args.sampling.temp,
        top_k: args.sampling.top_k,
        top_p: args.sampling.top_p,
        min_p: args.sampling.min_p,
        seed: args.sampling.seed,
        repetition_penalty: args.sampling.repetition_penalty,
        dry_multiplier: args.sampling.dry_multiplier,
        dry_base: args.sampling.dry_base,
        dry_allowed_length: args.sampling.dry_allowed_length,
        dry_penalty_last_n: args.sampling.dry_penalty_last_n,
        dry_sequence_breakers: Vec::new(),
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        stop_token_ids,
    })
}

pub(super) fn print_generation_preamble(user_prompt: &str) -> Result<()> {
    println!("Generating...");
    print!("{}", user_prompt);
    io::stdout().flush()?;
    Ok(())
}

fn generated_suffix<'a>(full_text: &'a str, prompt_text: &str) -> &'a str {
    full_text.strip_prefix(prompt_text).unwrap_or(full_text)
}

fn decode_generated_text(
    tokenizer: &mlxcel::tokenizer::MlxcelTokenizer,
    prompt_tokens: &[i32],
    generated_tokens: &[i32],
) -> String {
    let all_tokens: Vec<u32> = prompt_tokens
        .iter()
        .map(|&x| x as u32)
        .chain(generated_tokens.iter().map(|&x| x as u32))
        .collect();
    let full_text = tokenizer.decode(&all_tokens, false).unwrap_or_default();
    let prompt_decoded = tokenizer
        .decode(
            &prompt_tokens.iter().map(|&x| x as u32).collect::<Vec<_>>(),
            false,
        )
        .unwrap_or_default();

    generated_suffix(&full_text, &prompt_decoded).to_string()
}

fn print_generation_result(
    generated_text: &str,
    stats: &GenerationStats,
    profile: bool,
) -> Result<()> {
    print!("{}", generated_text);
    io::stdout().flush()?;

    println!();
    println!();

    if profile {
        println!("[Profile Results]");
        stats.print();
    } else {
        let total_time_sec = stats.decode_time_ms / 1000.0;
        println!(
            "[Generated {} tokens in {:.2}s = {:.2} tok/s]",
            stats.generated_tokens, total_time_sec, stats.decode_tok_per_sec
        );
    }

    Ok(())
}

fn generate_standard<M: LanguageModel>(
    model: &M,
    prompt_tokens: &[i32],
    max_tokens: usize,
    sampling_config: &SamplingConfig,
    profile: bool,
    kv_cache_mode: KVCacheMode,
    token_bias: TokenBiasMap,
) -> (Vec<i32>, GenerationStats) {
    // Axis B (B8): thread the resolved token-bias into the CxxGenerator. Empty
    // map preserves bit-exact baseline via `CxxGenerator::compose_sampling`.
    let mut generator = CxxGenerator::new_with_kv_mode(model.num_layers(), kv_cache_mode)
        .with_token_bias(token_bias);

    if profile {
        return generator.generate_with_stats(model, prompt_tokens, max_tokens, sampling_config);
    }

    let _ = generator.generate(model, prompt_tokens, 1, sampling_config);
    generator.reset_with_model(model);

    let capture_path = std::env::var("MLXCEL_METAL_CAPTURE_PATH").ok();
    if let Some(ref path) = capture_path {
        // Requires the mlxcel binary to be launched with
        // `MTL_CAPTURE_ENABLED=1`; otherwise Metal drops the capture.
        // Warmup above already primed MLX compile caches so the capture
        // covers steady-state decode work only.
        mlxcel_core::metal_start_capture(path);
    }

    let start_time = Instant::now();
    let tokens = generator.generate(model, prompt_tokens, max_tokens, sampling_config);
    let total_time = start_time.elapsed();
    let generated_len = tokens.len();

    if capture_path.is_some() {
        mlxcel_core::metal_stop_capture();
    }

    (
        tokens,
        generation_stats_from_duration(prompt_tokens.len(), generated_len, total_time),
    )
}

fn generate_with_embeddings<M: LanguageModel>(
    model: &M,
    prompt_tokens: &[i32],
    embeddings: &InputEmbeddings,
    max_tokens: usize,
    sampling_config: &SamplingConfig,
    profile: bool,
    kv_cache_mode: KVCacheMode,
    token_bias: TokenBiasMap,
) -> Result<(Vec<i32>, GenerationStats)> {
    // Axis B (B8): same wiring as the text-only CxxGenerator path above.
    let mut generator = CxxGenerator::new_with_kv_mode(model.num_layers(), kv_cache_mode)
        .with_token_bias(token_bias);
    let (input_embeds, mask_ref) = prepared_embedding_refs(embeddings)?;

    if profile {
        return Ok(generator.generate_with_stats_and_embeddings(
            model,
            prompt_tokens,
            Some(input_embeds),
            mask_ref,
            max_tokens,
            sampling_config,
        ));
    }

    let start_time = Instant::now();
    let tokens = generator.generate_streaming_with_embeddings(
        model,
        prompt_tokens,
        Some(input_embeds),
        mask_ref,
        max_tokens,
        sampling_config,
        |_| true,
    );
    let total_time = start_time.elapsed();
    let generated_len = tokens.len();

    Ok((
        tokens,
        generation_stats_from_duration(prompt_tokens.len(), generated_len, total_time),
    ))
}

fn run_generation_mode<M: LanguageModel>(
    model: &M,
    args: &GenerateArgs,
    prompt_tokens: &[i32],
    sampling_config: &SamplingConfig,
    vlm_embeddings: Option<&InputEmbeddings>,
    kv_cache_mode: KVCacheMode,
    mut token_bias: TokenBiasMap,
) -> Result<(Vec<i32>, GenerationStats)> {
    // issue #350: mask this model's reserved multimodal placeholder tokens
    // (audio / image / video span markers) to -inf in the output logits so
    // they can never leak into generated text. Merged into the token-bias map
    // here, before any generator is built, so it reaches every sub-path below
    // (speculative, VLM-embedding, and standard). No-op for non-multimodal
    // models whose suppressed set is empty.
    token_bias.suppress_tokens(&model.output_suppressed_token_ids());

    let output = if let Some(ref draft_model_path) = args.model.draft_model {
        // resolve the effective DrafterKind from
        // (a) the explicit `--draft-kind` CLI flag, OR
        // (b) the drafter's `config.json::model_type` auto-detection.
        //
        // When `--draft-kind` is unset AND the auto-detect maps to the
        // default DFlash kind (no `model_type` or unknown `model_type`),
        // we keep the classic `SpeculativeGenerator` path so all the
        // existing offline speculative-decoding workflows continue to
        // function bit-exactly. An explicit `--draft-kind` (or an
        // auto-detected MTP shape) routes through the kind-specific
        // generator path.
        let explicit_kind = args
            .speculative
            .parse_kind()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let resolved_kind = resolve_drafter_kind(draft_model_path, explicit_kind)
            .map_err(|e| anyhow::anyhow!("--draft-kind / drafter config: {e}"))?;
        let block_size = resolve_draft_block_size(args.speculative.draft_block_size, resolved_kind);
        let user_requested_explicit_kind = explicit_kind.is_some();

        println!("Loading draft model from {:?}...", draft_model_path);
        let (draft_model, _draft_tokenizer) = load_model(draft_model_path)?;
        println!("Draft model loaded.");
        println!(
            "Resolved drafter kind: {} (block_size = {block_size}{})",
            resolved_kind,
            if args.speculative.draft_block_size.is_some() {
                ", explicit"
            } else {
                ", default"
            },
        );

        // when the operator explicitly opted into MTP /
        // DFlash via `--draft-kind`, we MUST dispatch to the kind-specific
        // generator. The concrete `MtpGenerator<T>` / `DFlashGenerator`
        // round loops are wired in sub-6 and sub-12,
        // respectively, but those round loops require model-specific
        // `MtpTarget` / `SpeculativeTarget` impls that this offline CLI
        // path does not yet provide. Surface a clear, actionable error
        // that names the responsible sub-issue so an operator who passed
        // `--draft-kind mtp` doesn't silently fall back to the classic
        // path and miss the perf they were trying to validate. When
        // `--draft-kind` was unset (auto-detect resolved to a kind),
        // we instead log an info line and keep the classic path so the
        // default `--draft-model some/dflash-drafter` workflow remains
        // backward-compatible.
        if user_requested_explicit_kind {
            return Err(anyhow!(
                "--draft-kind {kind} is plumbed end-to-end but \
                 the offline `mlxcel generate` path does not yet construct the \
                 kind-specific `{generator}` round loop on this target model. \
                 The runtime wiring lands in {tracker}. For now, omit \
                 `--draft-kind` to use the classic SpeculativeGenerator with \
                 your `--draft-model` drafter.",
                kind = resolved_kind,
                generator = match resolved_kind {
                    DrafterKind::Mtp => "MtpGenerator",
                    DrafterKind::Dflash => "DFlashGenerator",
                    DrafterKind::InternalMtp => "InternalMtpGenerator",
                    // `DrafterKind` is `#[non_exhaustive]`; future variants
                    // land in follow-up epics and surface a generic name
                    // until they get their own tracker hint.
                    _ => "speculative round loop",
                },
                tracker = match resolved_kind {
                    DrafterKind::Mtp =>
                        "the MtpGenerator round loop and the per-target MtpTarget impls",
                    DrafterKind::Dflash =>
                        "the DFlashGenerator round loop and the per-target SpeculativeTarget impls",
                    DrafterKind::InternalMtp => "follow-up sub-issues",
                    _ => "follow-up speculative-decoding sub-issues",
                },
            ));
        }
        // Auto-detect resolved a kind but the operator didn't explicitly
        // request it. Log the resolution for diagnostic purposes and
        // fall through to the classic SpeculativeGenerator so the
        // historical `--draft-model <path>` workflow remains
        // bit-exactly the same as before this change.
        tracing::info!(
            drafter = %draft_model_path.display(),
            resolved_kind = %resolved_kind,
            block_size = block_size,
            "Auto-detected drafter kind; using classic SpeculativeGenerator path \
             (pass --draft-kind explicitly once the {} round loop is wired for this target)",
            resolved_kind,
        );

        let draft_num_layers = draft_model.num_layers();
        let main_num_layers = model.num_layers();
        // Axis B (B8): speculative decoding must apply the bias on the target
        // (main) model only: see `SpeculativeGenerator::with_token_bias` and
        // `draft_sampling` for the acceptance-rate rationale.
        let mut spec_generator = SpeculativeGenerator::new(main_num_layers, draft_num_layers)
            .with_token_bias(token_bias);

        spec_generator.generate(
            model,
            &draft_model,
            prompt_tokens,
            args.generation.max_tokens,
            args.model.num_draft_tokens,
            sampling_config,
        )
    } else if let Some(embeddings) = vlm_embeddings {
        generate_with_embeddings(
            model,
            prompt_tokens,
            embeddings,
            args.generation.max_tokens,
            sampling_config,
            args.generation.profile,
            kv_cache_mode,
            token_bias,
        )?
    } else {
        generate_standard(
            model,
            prompt_tokens,
            args.generation.max_tokens,
            sampling_config,
            args.generation.profile,
            kv_cache_mode,
            token_bias,
        )
    };

    Ok(output)
}

/// Parse the `--surgery <FILE>` YAML configuration when supplied and
/// install the resulting [`crate::surgery::SurgeryPipeline`] as the
/// process-wide active pipeline.
///
/// Returns early with a friendly `anyhow::Error` if the YAML cannot be
/// parsed, the file is missing, or any referenced donor checkpoint
/// (`source*` field) cannot be located. Surfacing these errors *before*
/// any model weights are touched mirrors the contract called out in
/// acceptance criterion (a).
///
/// When the flag is absent, this is a no-op: the active-pipeline slot
/// stays at `None` and the load path follows the bit-exact baseline.
///
/// Used by: `run_generate`
#[cfg(feature = "surgery")]
fn install_surgery_pipeline_from_cli(args: &GenerateArgs) -> Result<()> {
    let Some(ref path) = args.surgery else {
        return Ok(());
    };
    if !path.exists() {
        return Err(anyhow::anyhow!(
            "--surgery: config file does not exist: {}",
            path.display()
        ));
    }
    let pipeline = mlxcel::surgery::load_pipeline_from_file(path)
        .map_err(|e| anyhow::anyhow!("--surgery: {e}"))?;
    println!(
        "Surgery: loaded {} operation(s) from {}",
        pipeline.len(),
        path.display()
    );
    mlxcel::surgery::set_active_pipeline(Some(std::sync::Arc::new(pipeline)));
    Ok(())
}

/// Build [`ChatOptions`] for the interactive REPL from the parsed generate
/// args. Reuses the same sampling-knob mapping `build_cli_sampling_config`
/// uses (so the REPL and one-shot `generate` sample identically) and resolves
/// the KV-cache mode through the shared `resolve_kv_cache_mode` helper.
///
/// `stop_token_ids` is left empty here and filled in by `run_chat` from the
/// model's config once the model directory is resolved, mirroring the one-shot
/// path's `read_eos_token_ids(&args.model.model)`.
fn chat_options_from_args(args: &GenerateArgs) -> Result<crate::commands::ChatOptions> {
    let kv_cache_mode = resolve_kv_cache_mode(
        args.generation.turbo.cache_type_k.as_deref(),
        args.generation.turbo.cache_type_v.as_deref(),
        args.generation.turbo.kv_cache_mode.as_deref(),
    )
    .map_err(|e| anyhow::anyhow!("{}", e))?;

    let sampling = ResolvedSamplingParams {
        temperature: args.sampling.temp,
        top_k: args.sampling.top_k,
        top_p: args.sampling.top_p,
        min_p: args.sampling.min_p,
        seed: args.sampling.seed,
        repetition_penalty: args.sampling.repetition_penalty,
        dry_multiplier: args.sampling.dry_multiplier,
        dry_base: args.sampling.dry_base,
        dry_allowed_length: args.sampling.dry_allowed_length,
        dry_penalty_last_n: args.sampling.dry_penalty_last_n,
        dry_sequence_breakers: Vec::new(),
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        stop_token_ids: Vec::new(),
    };

    let mut opts = crate::commands::ChatOptions::new(
        args.model.model.clone(),
        args.generation.max_tokens,
        sampling,
    );
    opts.models_dir = args.model.models_dir.clone();
    opts.kv_cache_mode = kv_cache_mode;
    opts.no_chat_template = args.generation.no_chat_template;
    Ok(opts)
}

pub(crate) fn run_generate(args: GenerateArgs) -> Result<()> {
    // Epic #92 / issue #96: no `-p/--prompt` means "interactive chat". Route to
    // the reusable REPL entry point before any one-shot-only setup. The REPL
    // initializes its own runtime, resolves `-m` (repo-id auto-download), loads
    // the model + tokenizer, and reuses the same chat-template / SamplingConfig
    // / streaming-generation path as the one-shot flow below. Advanced
    // parallelism / speculative / surgery flags are not applied in the
    // interactive scope (per #96: "scoped to the CLI run/generate path only").
    if args.generation.prompt.is_none() {
        let opts = chat_options_from_args(&args)?;
        return crate::commands::run_chat(opts);
    }

    run_generate_once(args)
}

/// One-shot (`-p`-supplied) text generation: the historical `generate` flow.
fn run_generate_once(mut args: GenerateArgs) -> Result<()> {
    // Safe: the only caller (`run_generate`) guarantees `prompt` is `Some`.
    let user_prompt = args
        .generation
        .prompt
        .clone()
        .expect("run_generate_once requires a prompt");

    let runtime = initialize_runtime();
    print_runtime_setup(&runtime);

    // Axis A weight-load surgery. Parse the
    // YAML and install the pipeline *before* any heavier validation
    // so a malformed / missing surgery config fails fast with a clear
    // error rather than being masked by an unrelated tensor-parallel
    // or pipeline-parallel diagnostic. When `--surgery` is absent this
    // is a no-op and the load path remains bit-exact identical to the
    // earlier baseline. This reads only the `--surgery` YAML path, never
    // the model directory, so it must stay ahead of the `-m` resolver below,
    // a malformed surgery config never triggers an auto-download.
    #[cfg(feature = "surgery")]
    install_surgery_pipeline_from_cli(&args)?;

    // Resolve `-m` into a concrete model directory (epic #92, issue #94)
    // before any consumer reads it. An existing path is used verbatim
    // (byte-identical to the pre-#94 local-path behavior); an `owner/name`
    // HuggingFace repo-id is reused from the legacy CWD / HF cache / mlxcel
    // store, or auto-downloaded into the mlxcel store on a miss. Placed after
    // the (model-independent) surgery YAML validation but before the
    // tensor/pipeline-parallel validators and the quantization-advice,
    // tokenizer, memory-preflight, and model-load steps, all of which read
    // the model directory and therefore need the resolved path.
    args.model.model =
        resolve_model_source_with_override(&args.model.model, args.model.models_dir.as_deref())?;

    validate_tensor_parallel_args(&args)?;
    validate_pipeline_parallel_args(&args)?;

    // Parse and validate language bias arguments early (before model load).
    // Empty/absent CLI flags resolve to `None`, which keeps the generation
    // path bit-exact identical to the pre-B8 baseline (acceptance).
    let lang_bias_config: Option<LangBiasConfig> = args
        .lang_bias
        .resolve()
        .map_err(|e| anyhow::anyhow!("--lang-bias: {e}"))?;

    // Quantization recommendation and BF16 warning (before loading the model).
    let hw = mlxcel_core::hardware::get_hardware();
    if args.generation.recommend_quant {
        let advice = advise_quantization(&args.model.model, hw, None);
        print_quant_advice(&advice, hw);
        return Ok(());
    }

    // BF16 warning on M5 hardware (even without --recommend-quant).
    if hw.has_neural_accelerator {
        let advice = advise_quantization(&args.model.model, hw, None);
        if advice.model_uses_bfloat16 {
            eprintln!(
                "WARNING: This model uses BFloat16 weights, which are not supported by \
                 the M5 Neural Accelerator. For best performance, use an INT8 or FP16 \
                 quantized variant of this model (--recommend-quant for guidance)."
            );
        }
    }

    let pipeline_requested = cli_pipeline_requested(&args);
    let tokenizer = load_tokenizer(&args.model.model)?;
    let prompt = load_cli_prompt(
        &args.model.model,
        &user_prompt,
        args.generation.no_chat_template,
        args.generation.image.len(),
    );
    let mut prompt_tokens = tokenize_prompt(&tokenizer, &prompt)?;

    // Memory preflight (issue #56). Runs after prompt rendering/tokenization so
    // long prompts are included in the KV-cache budget, but still before the
    // model weights are loaded.
    let preflight_estimate = run_memory_preflight(&args, prompt_tokens.len())?;

    let sampling_config =
        build_cli_sampling_config(&args, mlxcel::read_eos_token_ids(&args.model.model));

    // Axis B (B8): resolve the parsed LangBiasConfig into a concrete
    // TokenBiasMap once per command invocation. Empty map = baseline bit-exact
    // path (no tokenizer.json read, no vocab scan, no sampling-path changes).
    let token_bias =
        resolve_cli_token_bias(lang_bias_config.as_ref(), &tokenizer, &args.model.model)?;
    if !token_bias.is_empty() {
        println!(
            "Language bias active: {} token entr{} biased",
            token_bias.len(),
            if token_bias.len() == 1 { "y" } else { "ies" }
        );
        // B9: emit structured debug trace once per generator construction.
        let (languages_str, policy_str) = if let Some(cfg) = &lang_bias_config {
            let langs: Vec<&str> = cfg
                .bias_set
                .ordered
                .iter()
                .map(|(code, _)| code.as_str())
                .collect();
            let langs_joined = langs.join(",");
            let policy = match cfg.policy {
                mlxcel_core::InclusionPolicy::Conservative => "conservative",
                mlxcel_core::InclusionPolicy::Strict => "strict",
            };
            (langs_joined, policy)
        } else {
            (String::new(), "conservative")
        };
        // emit byte_fragment_entries only when non-zero so the
        // existing B9 field shape is preserved for Phase 1 configs.
        let byte_fragment_entries = token_bias.byte_fragment_len();
        if byte_fragment_entries > 0 {
            tracing::debug!(
                entries = token_bias.len(),
                byte_fragment_entries,
                languages = %languages_str,
                policy = %policy_str,
                "lang_bias resolved"
            );
        } else {
            tracing::debug!(
                entries = token_bias.len(),
                languages = %languages_str,
                policy = %policy_str,
                "lang_bias resolved"
            );
        }
    }

    // Resolve the effective KV cache mode from the shared TurboQuant flag
    // group. The helper accepts the same precedence rules as `mlxcel serve`
    // and `mlxcel-server` (split flags > legacy shorthand > FP16 default),
    // so all three binaries route through one resolution path.
    let kv_cache_mode = resolve_kv_cache_mode(
        args.generation.turbo.cache_type_k.as_deref(),
        args.generation.turbo.cache_type_v.as_deref(),
        args.generation.turbo.kv_cache_mode.as_deref(),
    )
    .map_err(|e| anyhow::anyhow!("{}", e))?;

    // SAFETY: translate `--turbo-boundary-v` into the `MLXCEL_KV_BOUNDARY_V_LAYERS`
    // env var BEFORE any generator or worker thread is spawned. mlxcel-core
    // reads the env var on first cache instantiation (see
    // `cache::turbo::boundary::boundary_v_layers_from_env`), so the write
    // must happen on the single-threaded CLI startup path. When the flag
    // is absent, this is a no-op and any caller-set
    // `MLXCEL_KV_BOUNDARY_V_LAYERS` survives untouched.
    args.generation.turbo.apply_to_environment();
    if let Some(boundary) = args.generation.turbo.turbo_boundary_v
        && matches!(
            kv_cache_mode,
            KVCacheMode::Turbo4Asym
                | KVCacheMode::Turbo4
                | KVCacheMode::Turbo4Delegated
                | KVCacheMode::Turbo3Asym
        )
    {
        println!("Boundary-V: protecting {boundary} layer(s) on each end at Fp16");
    }

    match kv_cache_mode {
        KVCacheMode::Int8 => {
            println!("KV cache mode: int8 (per-token absmax quantization)");
        }
        KVCacheMode::Turbo4Asym => {
            println!("KV cache mode: fp16+turbo4 (asymmetric Fp16-K + Turbo4-V, ~26% KV savings)");
        }
        KVCacheMode::Turbo4 => {
            println!(
                "KV cache mode: turbo4 (symmetric Turbo4-K + Turbo4-V, ~73% KV savings; \
                 allowlisted families only; non-allowlisted models fall back to Turbo4Asym)"
            );
        }
        KVCacheMode::Turbo4Delegated => {
            println!(
                "KV cache mode: turbo4-delegated (Fp16-K + Turbo4-V with hot/cold split, \
                 ~26% KV savings + 97-100% FP16 decode speed at long context)"
            );
        }
        KVCacheMode::Turbo3Asym => {
            println!(
                "KV cache mode: fp16+turbo3 (asymmetric Fp16-K + Turbo3-V, \
                 ~5.1x total KV savings)"
            );
        }
        KVCacheMode::Fp16 => {}
    }

    let (generated_tokens, stats) = if pipeline_requested {
        // Axis B (B8): pipeline-parallel text generation samples via
        // `sample_token_optimized` directly and does not go through the
        // CxxGenerator/SpeculativeGenerator wrappers. We inject the token-bias
        // on the composed `SamplingConfig` before the pipeline is started.
        let mut pipeline_sampling = sampling_config.clone();
        if !token_bias.is_empty() && pipeline_sampling.token_bias.is_empty() {
            pipeline_sampling.token_bias = token_bias.clone();
        }
        let num_layers = resolve_cli_pipeline_num_layers(&args.model.model)?;
        print_generation_preamble(&user_prompt)?;
        generate_pipeline_text(
            &args.model.model,
            num_layers,
            &prompt_tokens,
            args.generation.max_tokens,
            &pipeline_sampling,
            &args,
        )?
    } else {
        let (model, _loaded_tokenizer) = load_generation_model(&args, preflight_estimate.as_ref())?;
        // Block-diffusion models generate by canvas denoising, not
        // autoregressive decoding: route them to the diffusion engine BEFORE
        // the standard CxxGenerator loop (issue #217, phase 1).
        if let mlxcel::LoadedModel::DiffusionGemma(diffusion_model) = &model {
            return super::generate_diffusion::run_diffusion_generation(
                diffusion_model,
                &args,
                &tokenizer,
                &prompt_tokens,
                &user_prompt,
            );
        }
        let vlm_embeddings = generate_vlm::compute_vlm_embeddings(
            &model,
            &mut prompt_tokens,
            &prompt,
            &args.generation.image,
            args.generation.audio.as_deref(),
            &args.generation.video,
            args.generation.fps,
            &tokenizer,
        )?;
        print_generation_preamble(&user_prompt)?;
        run_generation_mode(
            &model,
            &args,
            &prompt_tokens,
            &sampling_config,
            vlm_embeddings.as_ref(),
            kv_cache_mode,
            token_bias,
        )?
    };
    let generated_text = decode_generated_text(&tokenizer, &prompt_tokens, &generated_tokens);
    print_generation_result(&generated_text, &stats, args.generation.profile)?;

    // Cleanup
    mlxcel_core::clear_memory_cache();

    Ok(())
}

#[cfg(test)]
#[path = "generate_tests.rs"]
mod tests;
