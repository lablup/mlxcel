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
use std::io::{self, IsTerminal, Write as IoWrite};
use std::path::Path;
use std::time::{Duration, Instant};

use mlxcel::{
    GenerationStats, LanguageModel, RuntimeSetup, SamplingConfig, SpeculativeGenerator,
    distributed::{
        PipelineWorkerInput, RequestId,
        pipeline::{
            load_in_process_stage_worker_with_adapter, resolve_in_process_pipeline_num_layers,
        },
        resolve_model_shard_plan, shard_config_from_cli, validate_supported_runtime,
    },
    downloader::resolve_model_source_with_override,
    initialize_runtime,
    memory_estimate::{
        MemoryEstimate, QuantHint, estimate_total_memory, format_bytes, format_estimate,
    },
    quant_advisor::{advise_quantization, print_quant_advice},
    sampling::{ResolvedSamplingParams, build_sampling_config},
    select_backend,
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
use mlxcel_core::drafter::{DrafterKind, load_drafter, resolve_drafter_kind};

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
    // Surface the backend's GPU count (epic #486, sub-issue #487). Always 1 on
    // Metal; reports the real adapter count on a CUDA multi-GPU host, which is
    // what `--tp-size` shards across.
    println!("Detected {} GPU(s).", mlxcel_core::gpu_device_count());
    let load_start = Instant::now();
    let shard_config = shard_config_from_cli(
        args.tensor_parallel.tp_size,
        &args.tensor_parallel.tp_moe_mode,
        &args.tensor_parallel.tp_embedding_mode,
        &args.tensor_parallel.tp_lm_head_mode,
    )?;
    // Route model loading through the compute-backend seam (issue #338). Under
    // default features `select_backend()` folds to the MLX backend with no
    // runtime dispatch.
    let backend = select_backend();
    let result = if shard_config.tp_size > 1 {
        backend.load_model_with_tensor_parallel(
            &args.model.model,
            args.model.adapter.as_deref(),
            &shard_config,
        )
    } else if let Some(ref adapter_path) = args.model.adapter {
        println!("Loading LoRA adapter from {:?}...", adapter_path);
        backend.load_model_with_adapter(&args.model.model, adapter_path)
    } else {
        backend.load_model(&args.model.model)
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

/// Resolve a possibly-unlimited `-n/--max-tokens` value against the model's
/// context window (issue #476, llama.cpp parity).
///
/// The unlimited sentinel (`-n -1`) becomes `context_window - prompt_len` (read
/// from the checkpoint `config.json`, falling back to the shared default when
/// the model exposes no context length). An explicit `-n N` is returned
/// unchanged. A one-line note is printed when the unlimited default resolves so
/// the effective cap stays visible to the operator.
fn resolve_cli_max_tokens(requested: usize, model_dir: &Path, prompt_len: usize) -> usize {
    use mlxcel::cli::max_tokens::{
        DEFAULT_CONTEXT_WINDOW_FALLBACK, UNLIMITED_MAX_TOKENS, resolve_unlimited_max_tokens,
    };
    if requested != UNLIMITED_MAX_TOKENS {
        return requested;
    }
    let window =
        mlxcel::read_model_context_window(model_dir).unwrap_or(DEFAULT_CONTEXT_WINDOW_FALLBACK);
    let resolved = resolve_unlimited_max_tokens(requested, window, prompt_len);
    println!(
        "Max tokens: unlimited (-1) -> {resolved} (model context window {window}, prompt {prompt_len})"
    );
    resolved
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

/// Apply chat template with image / video / audio placeholders for VLM models.
///
/// Creates multimodal content entries that Gemma3-style templates can
/// render into `<start_of_image>` / `<|video|>` / `<|audio|>` markers (which
/// are later expanded into full soft-token blocks by the per-family expansion
/// helpers).
///
/// Only used when the template explicitly handles `type == 'image'`
/// content items. Video content parts are only emitted when the template
/// also handles `type == 'video'`, and audio content parts only when it
/// handles `type == 'audio'`. Templates without image support fall back to
/// text-only.
fn apply_vlm_chat_template(
    processor: &ChatTemplateProcessor,
    user_prompt: &str,
    num_images: usize,
    num_videos: usize,
    num_audios: usize,
) -> String {
    // Only attempt multimodal rendering when the template handles image
    // content items.  Templates that don't (e.g. Vicuna, ChatML) would
    // render the raw JSON list as text, producing garbled output.
    if !processor.supports_image_content() {
        return apply_user_chat_template(processor, user_prompt);
    }

    // Build a multimodal content list:
    // [{type: image}, ..., {type: video}, ..., {type: text, text: prompt},
    //  {type: audio}, ...].
    // Video and audio items are only included when the template renders them
    // (so the marker lands inside the user turn, alongside the question,
    // instead of before it). Placing the marker in the user turn matters: a
    // video spliced before the user turn yields no grounded answer (issue
    // #164), and an audio block placed in the model turn forces an immediate
    // EOS (issue #436).
    //
    // Ordering (issue #797): images and video precede the prompt text, but the
    // audio placeholder must follow it. Upstream mlx-vlm's Gemma 4 formatter
    // (`_format_list_with_image_type`, selected by `LIST_WITH_IMAGE_TYPE_TEXT`
    // for `gemma4` / `gemma3n`) builds the user content as
    // `[image]*n + [text] + [audio]*n`, i.e. audio AFTER the text, and the
    // server audio path lands the block in the same place (it splices the
    // BOA / AUDIO / EOA run right before the user turn's closing
    // `<end_of_turn>`, see `expand_gemma4_audio_tokens_for_server`). Rendering
    // audio BEFORE the text fed the 12B Unified checkpoint an out-of-distribution
    // frame that deterministically flipped it from transcription into answering
    // the perceived content on acoustically hard clips, and diverged from the
    // server. Keep audio last so the CLI and server render the identical audio
    // user turn.
    // https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/prompt_utils.py
    let emit_video = num_videos > 0 && processor.supports_video_content();
    let emit_audio = num_audios > 0 && processor.supports_audio_content();
    let mut content_parts: Vec<serde_json::Value> = Vec::new();
    for _ in 0..num_images {
        content_parts.push(serde_json::json!({"type": "image"}));
    }
    if emit_video {
        for _ in 0..num_videos {
            content_parts.push(serde_json::json!({"type": "video"}));
        }
    }
    content_parts.push(serde_json::json!({"type": "text", "text": user_prompt}));
    if emit_audio {
        for _ in 0..num_audios {
            content_parts.push(serde_json::json!({"type": "audio"}));
        }
    }

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
    num_videos: usize,
    num_audios: usize,
) -> String {
    if no_chat_template {
        return user_prompt.to_string();
    }

    processor.map_or_else(
        || user_prompt.to_string(),
        |processor| {
            // Route an audio-bearing request through the VLM template only when
            // the template actually renders audio content items. This keeps the
            // prompt byte-identical to the text-only path for every model whose
            // template does not handle `type == 'audio'` (no regression), while
            // letting Gemma 4 emit a `<|audio|>` marker in the user turn so the
            // per-family token expansion finds and expands it in place
            // (issue #436).
            let emit_audio = num_audios > 0 && processor.supports_audio_content();
            if num_images > 0 || num_videos > 0 || emit_audio {
                apply_vlm_chat_template(processor, user_prompt, num_images, num_videos, num_audios)
            } else {
                apply_user_chat_template(processor, user_prompt)
            }
        },
    )
}

fn load_cli_prompt(
    model_path: &Path,
    tokenizer: &crate::MlxcelTokenizer,
    user_prompt: &str,
    no_chat_template: bool,
    num_images: usize,
    num_videos: usize,
    num_audios: usize,
) -> String {
    let processor = if no_chat_template {
        None
    } else {
        let mut processor = ChatTemplateProcessor::from_model_path(model_path)
            .ok()
            .flatten();
        // CLI/server parity (upstream mlx-lm PR #1114): a tokenizer with a
        // recognized think-marker pair defaults `enable_thinking=true`.
        // Without this, templates that branch on `enable_thinking is defined
        // and enable_thinking is false` (Qwen3 family) render an empty
        // `<think>\n\n</think>` block; models not trained with that block
        // (e.g. the Qwen3-Omni Instruct thinker) emit an immediate
        // end-of-text after it.
        //
        // Exception (issue #686): the Gemma-4 thinking-channel template's
        // thinking-OFF branch already renders a well-formed CLOSED priming
        // scaffold that makes the model answer directly, matching
        // transformers' no-`enable_thinking` default. Forcing thinking on there
        // instead yields a bare `<|turn>model\n` that greedy-collapses to
        // `<pad>`, so keep its default at false.
        if let Some(p) = processor.as_mut()
            && tokenizer.infer_thinking_markers().has_thinking()
            && !p.wants_thinking_default_off()
        {
            p.set_default_enable_thinking(true);
        }
        processor
    };

    resolve_cli_prompt(
        user_prompt,
        no_chat_template,
        processor.as_ref(),
        num_images,
        num_videos,
        num_audios,
    )
}

/// Number of `<|video|>` content parts to render into the CLI chat prompt.
///
/// Only the encoder-free `gemma4_unified` model expands a real `video_token_id`
/// placeholder inside the user turn (issue #164); every other family (including
/// the ViT-backed `gemma4` VLM, which splices video frames after BOS via a
/// sentinel) keeps `0` so its prompt rendering is byte-for-byte unchanged. On
/// any detection failure we conservatively return `0`.
fn cli_video_content_part_count(model_path: &Path, num_videos: usize) -> usize {
    if num_videos == 0 {
        return 0;
    }
    match mlxcel::models::get_model_type(model_path) {
        Ok(mlxcel::models::ModelType::Gemma4Unified) => num_videos,
        _ => 0,
    }
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
        // XTC is not yet exposed as a CLI flag; the CLI generation path
        // keeps it disabled.
        xtc_probability: 0.0,
        xtc_threshold: 0.1,
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

/// Split the reasoning channel out of a one-shot generation before display.
///
/// Reasoning-capable checkpoints (Gemma 4's `<|channel>thought` / `<channel|>`
/// pair, Qwen-style `<think>` / `</think>`) emit their chain-of-thought inline
/// with the answer, and `decode_generated_text` renders with special tokens so
/// those raw markers reach the terminal (issue #884). Route the whole decoded
/// reply through the shared `mlxcel::reasoning_stream` splitter so the channel
/// is suppressed by default (only the final answer prints, no raw markers) and
/// surfaced dimmed when `--show-reasoning` is set. A non-thinking model has no
/// markers, so the filter is an inert passthrough and the returned string is
/// byte-identical to `generated_text`. When the rendered prompt primed an open
/// thinking marker the filter starts inside the channel so the primed reasoning
/// is suppressed too.
fn filter_reasoning_for_display(
    tokenizer: &mlxcel::tokenizer::MlxcelTokenizer,
    prompt: &str,
    generated_text: &str,
    show_reasoning: bool,
) -> String {
    let markers = tokenizer.infer_thinking_markers();
    // When the rendered prompt primed an open thinking marker (`<think>\n` for
    // Qwen-style, `<|channel>thought\n` for a thinking-on Gemma-4 channel) the
    // generated text starts already inside the channel with no open marker, so
    // start the filter in the reasoning state to keep the primed thought body
    // and its raw close marker off the terminal.
    let primed = mlxcel::reasoning_stream::prompt_primed_open_thinking(&markers, prompt);
    let dim = io::stdout().is_terminal();
    mlxcel::reasoning_stream::render_full(&markers, generated_text, primed, show_reasoning, dim)
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
    model_path: &Path,
    prompt_tokens: &[i32],
    max_tokens: usize,
    sampling_config: &SamplingConfig,
    profile: bool,
    kv_cache_mode: KVCacheMode,
    token_bias: TokenBiasMap,
) -> Result<(Vec<i32>, GenerationStats)> {
    // Route generation through the inference-session seam (issue #448, ADR 0004).
    // Under default features `select_backend()` folds to MLX and the session
    // wraps the same `CxxGenerator`, so the delegated generation methods run the
    // identical decode loop and CLI output is byte-identical. Axis B (B8): the
    // resolved token-bias is threaded into the session; an empty map preserves
    // bit-exact baseline via the generator's `compose_sampling`. `model_path` is
    // threaded for a session-driven backend (issue #449 OpenXLA) that loads its
    // own weights/config; MLX ignores it.
    let mut session = select_backend().create_session(
        model_path,
        model.num_layers(),
        kv_cache_mode,
        token_bias,
    )?;

    if profile {
        return Ok(session.generate_with_stats(model, prompt_tokens, max_tokens, sampling_config));
    }

    let _ = session.generate(model, prompt_tokens, 1, sampling_config);
    session.reset_with_model(model);

    let capture_path = std::env::var("MLXCEL_METAL_CAPTURE_PATH").ok();
    if let Some(ref path) = capture_path {
        // Requires the mlxcel binary to be launched with
        // `MTL_CAPTURE_ENABLED=1`; otherwise Metal drops the capture.
        // Warmup above already primed MLX compile caches so the capture
        // covers steady-state decode work only.
        mlxcel_core::metal_start_capture(path);
    }

    let start_time = Instant::now();
    let tokens = session.generate(model, prompt_tokens, max_tokens, sampling_config);
    let total_time = start_time.elapsed();
    let generated_len = tokens.len();

    if capture_path.is_some() {
        mlxcel_core::metal_stop_capture();
    }

    Ok((
        tokens,
        generation_stats_from_duration(prompt_tokens.len(), generated_len, total_time),
    ))
}

fn generate_with_embeddings<M: LanguageModel>(
    model: &M,
    model_path: &Path,
    prompt_tokens: &[i32],
    embeddings: &InputEmbeddings,
    max_tokens: usize,
    sampling_config: &SamplingConfig,
    profile: bool,
    kv_cache_mode: KVCacheMode,
    token_bias: TokenBiasMap,
) -> Result<(Vec<i32>, GenerationStats)> {
    // Axis B (B8): same session wiring as the text-only path above (issue #448).
    // `model_path` is threaded for the session-driven OpenXLA backend (#449).
    let mut session = select_backend().create_session(
        model_path,
        model.num_layers(),
        kv_cache_mode,
        token_bias,
    )?;
    let (input_embeds, mask_ref) = prepared_embedding_refs(embeddings)?;

    if profile {
        return Ok(session.generate_with_stats_and_embeddings(
            model,
            prompt_tokens,
            Some(input_embeds),
            mask_ref,
            max_tokens,
            sampling_config,
        ));
    }

    let start_time = Instant::now();
    let tokens = session.generate_streaming_with_embeddings(
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

/// Read `num_hidden_layers` from a model directory's `config.json` (0 if absent
/// or unparsable). The OpenXLA session stores it as metadata; the bundled graph
/// fixes the architecture, so a best-effort value is sufficient.
#[cfg(feature = "xla-backend")]
fn xla_num_layers(model_dir: &Path) -> usize {
    std::fs::read_to_string(model_dir.join("config.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| {
            v.get("num_hidden_layers")
                .and_then(serde_json::Value::as_u64)
        })
        .map_or(0, |n| n as usize)
}

/// Self-contained text generation for the OpenXLA backend (issue #449 Phase 3).
///
/// The OpenXLA engine drives generation from its own session and has no MLX
/// `LoadedModel`, so this path skips `load_model` (which the XLA backend rejects)
/// and the generic model-threaded loop: it creates the session straight from the
/// model directory and runs the session's own greedy loop, which seeds KV and
/// samples on-device. Text-only and greedy by design (no VLM / draft / sampling
/// knobs), so the caller routes here only when `MLXCEL_BACKEND=xla` is selected.
#[cfg(feature = "xla-backend")]
fn generate_xla_text(
    model_path: &Path,
    num_layers: usize,
    prompt_tokens: &[i32],
    max_tokens: usize,
    kv_cache_mode: KVCacheMode,
    token_bias: TokenBiasMap,
) -> Result<(Vec<i32>, GenerationStats)> {
    let mut session =
        select_backend().create_session(model_path, num_layers, kv_cache_mode, token_bias)?;
    let start_time = Instant::now();
    let tokens = match &mut session {
        mlxcel::Session::Xla(s) => {
            let eos = s.eos_token_ids().to_vec();
            s.generate_greedy(prompt_tokens, max_tokens, &eos)
                .map_err(|e| anyhow!("OpenXLA generation failed: {e}"))?
        }
        // `select_backend()` returned the XLA backend, so its `create_session`
        // yields an XLA session; any other variant would be a wiring bug.
        _ => anyhow::bail!("the OpenXLA backend did not produce an XLA session"),
    };
    let total_time = start_time.elapsed();
    let generated_len = tokens.len();
    Ok((
        tokens,
        generation_stats_from_duration(prompt_tokens.len(), generated_len, total_time),
    ))
}

// Takes a concrete `&LoadedModel` (rather than a generic `M: LanguageModel`)
// so the `--draft-kind mtp` branch can match the target's family and select the
// matching per-target `MtpTarget` adapter (issue #166). Every inner call
// (`generate_standard`, `generate_with_embeddings`, `SpeculativeGenerator::generate`)
// stays generic over `LanguageModel`; `LoadedModel` implements that trait, so
// the monomorphized code for the non-MTP paths is identical to the prior generic
// form. The sole caller already passes `&LoadedModel`.
fn run_generation_mode(
    model: &mlxcel::LoadedModel,
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

        // issue #166: when the operator explicitly passes `--draft-kind mtp`,
        // drive the kind-specific `MtpGenerator` round loop here, reusing the
        // SAME per-target `MtpTarget` adapters the server burst path uses
        // (`src/models/gemma4_mtp_target.rs`). This runs BEFORE
        // `load_model(draft_model_path)` because an MTP assistant is loaded as a
        // `Drafter` (via `load_drafter`), not as a full `LoadedModel`. DFlash /
        // InternalMtp explicit kinds and the auto-detect classic path are
        // untouched and fall through below.
        if should_route_offline_mtp(user_requested_explicit_kind, resolved_kind) {
            if vlm_embeddings.is_some() {
                return Err(anyhow!(
                    "--draft-kind mtp does not support multimodal (image / audio / \
                     video) input in the offline `mlxcel generate` path; rerun with \
                     a text-only prompt, or omit --draft-model for multimodal \
                     generation"
                ));
            }
            return run_offline_mtp(
                model,
                draft_model_path,
                prompt_tokens,
                args.generation.max_tokens,
                sampling_config,
                block_size as usize,
                token_bias,
            );
        }

        println!("Loading draft model from {:?}...", draft_model_path);
        let (draft_model, _draft_tokenizer) = select_backend().load_model(draft_model_path)?;
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

        // MTP is handled above (issue #166). The remaining explicit kinds
        // (DFlash, InternalMtp) still need their kind-specific round loops and
        // per-target `SpeculativeTarget` impls wired into this offline path, so
        // surface a clear, actionable error that names the responsible follow-up
        // rather than silently falling back to the classic path (which would
        // miss the perf the operator asked for). When `--draft-kind` was unset
        // (auto-detect resolved to a kind), we instead log an info line and keep
        // the classic path so the default `--draft-model some/dflash-drafter`
        // workflow remains backward-compatible.
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
            &args.model.model,
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
            &args.model.model,
            prompt_tokens,
            args.generation.max_tokens,
            sampling_config,
            args.generation.profile,
            kv_cache_mode,
            token_bias,
        )?
    };

    Ok(output)
}

/// Routing gate for the offline MTP speculative path (issue #166).
///
/// Returns `true` only when the operator explicitly passed `--draft-kind mtp`
/// (an auto-detected MTP shape with no explicit flag keeps the classic
/// `SpeculativeGenerator` path for backward compatibility, matching the prior
/// behavior). DFlash / InternalMtp explicit kinds return `false` and fall
/// through to the deferred-error branch. Extracted as a pure function so the
/// loop-construction decision is unit-testable without loading a model.
fn should_route_offline_mtp(
    user_requested_explicit_kind: bool,
    resolved_kind: DrafterKind,
) -> bool {
    user_requested_explicit_kind && resolved_kind == DrafterKind::Mtp
}

/// Drive a constructed [`mlxcel_core::speculative::mtp::target::MtpTarget`]
/// adapter through the [`mlxcel_core::speculative::mtp::MtpGenerator`] round
/// loop and return the emitted tokens plus timing stats.
///
/// Mirrors the server burst driver
/// (`src/server/batch/speculative_burst.rs::drive_mtp_generator`) minus the
/// drafter-recovery / adaptive-policy bookkeeping the offline single-shot path
/// does not need. The cooperative-cancel flag is always clear offline (there is
/// no client to disconnect mid-generation) and logprob capture is disabled (the
/// CLI prints decoded text, not per-token logprobs).
fn drive_offline_mtp<T>(
    adapter: T,
    drafter: Box<dyn mlxcel_core::drafter::Drafter>,
    prompt_tokens: &[i32],
    max_tokens: usize,
    sampling: &SamplingConfig,
    token_history: &[i32],
    block_size: usize,
) -> (Vec<i32>, GenerationStats)
where
    T: mlxcel_core::speculative::mtp::target::MtpTarget,
{
    use std::sync::atomic::AtomicBool;

    use mlxcel_core::sampling::LogprobsConfig;
    use mlxcel_core::speculative::mtp::MtpGenerator;

    let logprobs = LogprobsConfig::default();
    let cancel = AtomicBool::new(false);
    let mut generator = MtpGenerator::new(adapter, drafter, block_size);
    let (tokens, _logprobs, stats) = generator.generate(
        prompt_tokens,
        max_tokens,
        sampling,
        token_history,
        &cancel,
        &logprobs,
    );
    (tokens, stats)
}

/// Construct and drive the MTP speculative round loop for the offline
/// `mlxcel generate` path (issue #166).
///
/// Reuses the exact per-target adapters the server burst path selects
/// (`src/models/gemma4_mtp_target.rs`) and the same `MtpGenerator` round-loop
/// driver, so the offline path is byte-identical to the server's speculative
/// output at temperature 0 and identical to the non-speculative offline path
/// for the same target / prompt / `-n` (the MTP greedy-parity invariant: the
/// loop accepts exactly the tokens the target would have produced greedily).
///
/// The drafter is loaded through [`load_drafter`] (an MTP assistant is a
/// `Drafter`, not a full `LoadedModel`), compatibility-checked, then bound to
/// the SAME concrete target the adapter wraps BEFORE the generator runs.
/// [`MtpGenerator::generate`] does not bind internally, so the bind here is
/// load-bearing: without it the first `draft_block` returns
/// `DrafterError::BindNotCalled` and the loop emits only the seed bonus.
///
/// A target that is not MTP-capable returns a clear error instead of silently
/// falling back, matching the issue's contract.
fn run_offline_mtp(
    model: &mlxcel::LoadedModel,
    draft_model_path: &Path,
    prompt_tokens: &[i32],
    max_tokens: usize,
    sampling_config: &SamplingConfig,
    block_size: usize,
    token_bias: TokenBiasMap,
) -> Result<(Vec<i32>, GenerationStats)> {
    use mlxcel::LoadedModel;
    use mlxcel::models::gemma4_mtp_target::{
        Gemma4MtpTargetAdapter, Gemma4UnifiedMtpTargetAdapter, Gemma4VLMtpTargetAdapter,
    };

    if block_size < 2 {
        return Err(anyhow!(
            "--draft-kind mtp with block_size={block_size} produces no draft \
             proposals (need >= 2); pass --draft-block-size with a value >= 2"
        ));
    }

    // Resolve the concrete target reference the drafter binds to, and reject any
    // non-MTP-capable target. Mirrors the server burst dispatch
    // (`run_mtp_burst`): bind to the same concrete Gemma 4 model the adapter
    // wraps below. VLM / Unified wrappers expose their text backbone through the
    // `LanguageModel` impl, so the compat check / bind see the text hidden size
    // and vocab the assistant was trained against.
    let target_lm: &dyn LanguageModel = match model {
        LoadedModel::Gemma4(wrapper) => wrapper as &dyn LanguageModel,
        LoadedModel::Gemma4VLM(vlm) => vlm as &dyn LanguageModel,
        LoadedModel::Gemma4Unified(unified) => unified as &dyn LanguageModel,
        _ => {
            return Err(anyhow!(
                "--draft-kind mtp is only supported for Gemma 4 (text, VLM, or \
                 Unified) targets; the loaded target is not MTP-capable. Omit \
                 --draft-kind to use the classic SpeculativeGenerator with your \
                 --draft-model drafter."
            ));
        }
    };

    println!("Loading MTP drafter from {:?}...", draft_model_path);
    let (mut drafter, kind) = load_drafter(draft_model_path, Some(DrafterKind::Mtp))
        .map_err(|e| anyhow!("MTP drafter load failed: {e}"))?;
    if kind != DrafterKind::Mtp {
        return Err(anyhow!(
            "drafter at {draft_model_path:?} did not resolve to an MTP drafter \
             (got {kind})"
        ));
    }

    // Compatibility gate BEFORE binding (rejects a mismatched
    // backbone-hidden-size / vocab pairing), then bind.
    drafter
        .validate_target_compat(target_lm)
        .map_err(|e| anyhow!("MTP drafter incompatible with target: {e}"))?;
    drafter
        .bind(target_lm)
        .map_err(|e| anyhow!("MTP drafter bind failed: {e}"))?;
    println!("MTP drafter loaded and bound (block_size = {block_size}).");

    // Inject the resolved token bias (CLI `--lang-bias` plus the model's
    // reserved multimodal placeholder suppression from issue #350) into the
    // sampling config so the adapter applies the SAME bias the non-speculative
    // `CxxGenerator` path applies via `with_token_bias`. This is what keeps the
    // temp-0 output byte-identical to the non-speculative path: the adapter's
    // `prefill_and_seed` / `verify_forward` read `sampler.token_bias`.
    let mut sampling = sampling_config.clone();
    sampling.token_bias = token_bias;

    // History-dependent-penalty context for the first-bonus sample (mirrors the
    // server burst path and the classic decode path's first-token seed). Empty
    // when no repetition / frequency / presence / DRY penalty is configured.
    let token_history = initial_token_history(prompt_tokens, sampling.needs_token_history());

    // Select the per-target adapter exactly as the server does, then drive the
    // round loop. `seq_id = None` selects the wrapper's internal single-sequence
    // fallback slot, the documented offline / single-row CLI usage.
    let (mut tokens, mut stats) = match model {
        LoadedModel::Gemma4(wrapper) => drive_offline_mtp(
            Gemma4MtpTargetAdapter::new_with_block_size(wrapper, None, block_size),
            drafter,
            prompt_tokens,
            max_tokens,
            &sampling,
            &token_history,
            block_size,
        ),
        LoadedModel::Gemma4VLM(vlm) => drive_offline_mtp(
            Gemma4VLMtpTargetAdapter::new_with_block_size(vlm, None, block_size),
            drafter,
            prompt_tokens,
            max_tokens,
            &sampling,
            &token_history,
            block_size,
        ),
        LoadedModel::Gemma4Unified(unified) => drive_offline_mtp(
            Gemma4UnifiedMtpTargetAdapter::new_with_block_size(unified, None, block_size),
            drafter,
            prompt_tokens,
            max_tokens,
            &sampling,
            &token_history,
            block_size,
        ),
        // Unreachable: the variant gate above already returned an error for any
        // non-MTP-capable target. Kept for match exhaustiveness.
        _ => unreachable!("non-MTP-capable target rejected by the variant gate above"),
    };

    // issue #166: strip the terminal EOS / stop token so the offline MTP output
    // is byte-identical to the non-speculative `mlxcel generate` path. The
    // `MtpGenerator` pushes a token onto its `emitted` vec and THEN checks EOS,
    // so its returned vector includes the terminal stop token. Both reference
    // paths exclude it: `CxxGenerator::generate` breaks on EOS BEFORE pushing,
    // and the server burst `finalize_burst_success` does the same. Without this,
    // `decode_generated_text` (which decodes with skip_special_tokens = false)
    // would render the leaked stop token (e.g. `<end_of_turn>`) and inflate the
    // printed generated-token count by one. Use the SAME merged EOS set the
    // generator used: the target's eos ids plus the sampling `stop_token_ids`.
    let eos_tokens = merged_eos_token_ids(target_lm.eos_token_ids(), &sampling.stop_token_ids);
    tokens = strip_trailing_eos(tokens, &eos_tokens);

    // Realign the stats with the stripped output so the printed
    // "[Generated N tokens ...]" line and tok/s match the non-speculative path,
    // which counts EOS-excluded tokens.
    stats.generated_tokens = tokens.len();
    stats.decode_tok_per_sec = if stats.decode_time_ms > 0.0 {
        tokens.len() as f64 / (stats.decode_time_ms / 1000.0)
    } else {
        0.0
    };

    Ok((tokens, stats))
}

/// Truncate `tokens` at the first EOS / stop token so the returned vector
/// excludes the terminal stop token, matching `CxxGenerator::generate` and the
/// server burst `finalize_burst_success` (issue #166). The `MtpGenerator` never
/// emits tokens after an EOS, so truncating at the first occurrence is
/// equivalent to (and more robust than) dropping only a trailing one. An empty
/// `eos_tokens` set is a no-op.
fn strip_trailing_eos(mut tokens: Vec<i32>, eos_tokens: &[i32]) -> Vec<i32> {
    if let Some(pos) = tokens.iter().position(|t| eos_tokens.contains(t)) {
        tokens.truncate(pos);
    }
    tokens
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
        // XTC is not yet exposed as a CLI flag; the REPL keeps it disabled.
        xtc_probability: 0.0,
        xtc_threshold: 0.1,
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
    opts.show_reasoning = args.generation.show_reasoning;
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
        ensure!(
            args.generation.output_audio.is_none(),
            "--output-audio requires a one-shot -p/--prompt run (not interactive chat)"
        );
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

    // --output-audio (issue #665): validate the speech-output request before
    // any heavy work. The Qwen3-Omni talker conditions on the chat-templated
    // <|im_start|> role segments, and multimodal-conditioned speech (which
    // needs the thinker hidden-state tap) is not wired yet.
    if args.generation.output_audio.is_some() {
        ensure!(
            !pipeline_requested,
            "--output-audio is not supported with pipeline parallelism"
        );
        ensure!(
            !args.generation.no_chat_template,
            "--output-audio requires the chat template; do not combine it with --no-chat-template"
        );
        ensure!(
            args.generation.image.is_empty()
                && args.generation.audio.is_none()
                && args.generation.video.is_empty(),
            "--output-audio currently supports text-only prompts (no --image/--audio/--video)"
        );
    }

    let tokenizer = load_tokenizer(&args.model.model)?;
    let prompt = load_cli_prompt(
        &args.model.model,
        &tokenizer,
        &user_prompt,
        args.generation.no_chat_template,
        args.generation.image.len(),
        cli_video_content_part_count(&args.model.model, args.generation.video.len()),
        usize::from(args.generation.audio.is_some()),
    );
    let mut prompt_tokens = tokenize_prompt(&tokenizer, &prompt)?;

    // llama.cpp parity (issue #476): resolve an unlimited `-n -1` into a
    // concrete budget (model context window minus the rendered prompt) now, so
    // every downstream consumer (the memory preflight below, plus the standard /
    // VLM / speculative / XLA / pipeline generators) reads a plain usize. An
    // explicit `-n N` passes through unchanged.
    args.generation.max_tokens = resolve_cli_max_tokens(
        args.generation.max_tokens,
        &args.model.model,
        prompt_tokens.len(),
    );

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

    // OpenXLA backend (issue #449): the engine drives generation from its own
    // session and has no MLX `LoadedModel`, so route around `load_model` (which
    // the XLA backend rejects) and the model-threaded loop entirely, then fall
    // back into the shared decode / print path. Compiled only under
    // `xla-backend`; taken at runtime only when `MLXCEL_BACKEND=xla` is selected,
    // so the default flow below is unchanged. (The conditional move of
    // `token_bias` is sound because this branch diverges with `return`.)
    #[cfg(feature = "xla-backend")]
    if select_backend().name() == "xla" {
        let num_layers = xla_num_layers(&args.model.model);
        print_generation_preamble(&user_prompt)?;
        let (generated_tokens, stats) = generate_xla_text(
            &args.model.model,
            num_layers,
            &prompt_tokens,
            args.generation.max_tokens,
            kv_cache_mode,
            token_bias,
        )?;
        let generated_text = decode_generated_text(&tokenizer, &prompt_tokens, &generated_tokens);
        let visible = filter_reasoning_for_display(
            &tokenizer,
            &prompt,
            &generated_text,
            args.generation.show_reasoning,
        );
        print_generation_result(&visible, &stats, args.generation.profile)?;
        mlxcel_core::clear_memory_cache();
        return Ok(());
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
        // --output-audio (issue #665): fail before generation when the loaded
        // model carries no talker/code2wav speech stack.
        if args.generation.output_audio.is_some()
            && !matches!(model, mlxcel::LoadedModel::Qwen3OmniMoe(_))
        {
            anyhow::bail!(
                "--output-audio is only supported for Qwen3-Omni models (this model has no \
                 talker/code2wav speech stack)"
            );
        }
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
        // LLaDA-2 MoE generates by block-wise unmasking, not autoregressive
        // decode: route it to its own driver before the CxxGenerator loop.
        if let mlxcel::LoadedModel::Llada2Moe(llada2_model) = &model {
            return super::generate_llada2::run_llada2_generation(
                llada2_model,
                &args,
                &tokenizer,
                &prompt_tokens,
                &user_prompt,
            );
        }
        // Reject an off-ladder `--image-soft-tokens` before loading any image:
        // the budget drives the resize target, so an unsupported value is a
        // user error, not something to clamp silently.
        let image_soft_tokens = args
            .generation
            .image_soft_tokens
            .map(mlxcel::vision::processors::gemma4::validate_image_soft_tokens)
            .transpose()
            .map_err(|err| anyhow::anyhow!("--image-soft-tokens: {err}"))?;
        let vlm_embeddings = generate_vlm::compute_vlm_embeddings(
            &model,
            &mut prompt_tokens,
            &prompt,
            &args.generation.image,
            args.generation.audio.as_deref(),
            &args.generation.video,
            args.generation.fps,
            &tokenizer,
            image_soft_tokens,
        )?;
        print_generation_preamble(&user_prompt)?;
        let generation = run_generation_mode(
            &model,
            &args,
            &prompt_tokens,
            &sampling_config,
            vlm_embeddings.as_ref(),
            kv_cache_mode,
            token_bias,
        )?;
        // --output-audio (issue #665): the speech pass runs AFTER text
        // generation completes, re-conditioning the lazily-loaded talker on
        // [prompt + generated] and vocoding through code2wav.
        if let Some(wav_path) = args.generation.output_audio.clone() {
            generate_vlm::run_speech_synthesis(
                &model,
                &args,
                &wav_path,
                &prompt_tokens,
                &generation.0,
            )?;
        }
        generation
    };
    let generated_text = decode_generated_text(&tokenizer, &prompt_tokens, &generated_tokens);
    let visible = filter_reasoning_for_display(
        &tokenizer,
        &prompt,
        &generated_text,
        args.generation.show_reasoning,
    );
    print_generation_result(&visible, &stats, args.generation.profile)?;

    // Cleanup
    mlxcel_core::clear_memory_cache();

    Ok(())
}

#[cfg(test)]
#[path = "generate_tests.rs"]
mod tests;
