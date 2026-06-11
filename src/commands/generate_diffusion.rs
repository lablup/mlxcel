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

//! CLI driver for block-diffusion text generation (issue #217, phases 1-2).
//!
//! Thin bridge between the parsed [`crate::GenerateArgs`] flag surface and
//! [`DiffusionGemmaModel::generate_diffusion_streaming`]: resolves the
//! diffusion options, seeds the RNG, optionally preprocesses image input and
//! expands the prompt with image tokens, streams decoded text through the
//! shared incremental detokenizer, and prints the generation stats.

use anyhow::{Result, anyhow};
use std::io::{self, Write as IoWrite};
use std::path::Path;

use mlxcel::models::DiffusionGemmaModel;
use mlxcel::models::diffusion_gemma::{
    DiffusionGenerateOptions, DiffusionGenerationStats, DiffusionSamplerKind,
    DiffusionVisionPrefill,
};
use mlxcel::server::model_provider::model_worker::StreamingDecodeState;
use mlxcel::tokenizer::MlxcelTokenizer;
use mlxcel::vision::processors::gemma4::{Gemma4ImageInput, Gemma4Processor};

use super::generate::print_generation_preamble;
use crate::GenerateArgs;

/// Build the Gemma 4 image processor for the DiffusionGemma checkpoint.
///
/// The diffusion checkpoint's `image_processor` is identical to the gemma-4
/// VLM family (size 224x224, patch 16, pooling_kernel_size 3, 280 soft
/// tokens). Values are read from `processor_config.json` when present and
/// fall back to those defaults otherwise.
fn build_image_processor(model_dir: &Path, default_soft_tokens: usize) -> Gemma4Processor {
    let image_cfg = std::fs::read_to_string(model_dir.join("processor_config.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|cfg| cfg.get("image_processor").cloned());
    let get = |key: &str, default: usize| -> usize {
        image_cfg
            .as_ref()
            .and_then(|cfg| cfg.get(key))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(default)
    };
    Gemma4Processor::new(
        get("patch_size", 16),
        get("max_soft_tokens", default_soft_tokens),
        get("pooling_kernel_size", 3),
    )
}

/// Result of preparing image input: the expanded prompt ids (with each image
/// placeholder rewritten to `boi + image_token * N + eoi`) and the prefill.
struct PreparedDiffusionVision {
    expanded_ids: Vec<i32>,
    prefill: DiffusionVisionPrefill,
}

/// Preprocess `--image` paths, expand the prompt, and build the vision
/// prefill (inputs_embeds + overlay block ids).
fn prepare_diffusion_vision(
    model: &DiffusionGemmaModel,
    model_dir: &Path,
    image_paths: &[std::path::PathBuf],
    prompt_tokens: &[i32],
) -> Result<PreparedDiffusionVision> {
    let vision = model.vision().ok_or_else(|| {
        anyhow!(
            "This DiffusionGemma checkpoint does not include a vision tower; \
             run with a text-only prompt"
        )
    })?;

    let images: Vec<image::DynamicImage> = image_paths
        .iter()
        .map(|path| {
            image::open(path).map_err(|e| anyhow!("Failed to load image {:?}: {}", path, e))
        })
        .collect::<Result<Vec<_>>>()?;
    println!("Loaded {} image(s).", images.len());

    let processor = build_image_processor(model_dir, vision.soft_tokens_per_image);
    let processed: Vec<Gemma4ImageInput> = processor.preprocess(&images);
    let num_soft_tokens: Vec<usize> = processed.iter().map(|img| img.num_soft_tokens).collect();

    let mut expanded_ids = prompt_tokens.to_vec();
    mlxcel::vlm_runtime::expand_gemma4_image_tokens_pub(
        &mut expanded_ids,
        vision.image_token_id,
        vision.boi_token_id,
        vision.eoi_token_id,
        &num_soft_tokens,
    )?;

    let prefill = model
        .prepare_vision_prefill(&expanded_ids, &processed)
        .map_err(|e| anyhow!("{e}"))?;

    println!(
        "DiffusionGemma: expanded {} image(s) into {} soft token(s) ({} total prompt tokens)",
        images.len(),
        num_soft_tokens.iter().sum::<usize>(),
        expanded_ids.len()
    );

    Ok(PreparedDiffusionVision {
        expanded_ids,
        prefill,
    })
}

fn parse_sampler_kind(name: &str) -> Result<DiffusionSamplerKind> {
    match name {
        "entropy-bound" => Ok(DiffusionSamplerKind::EntropyBound),
        "confidence-threshold" => Ok(DiffusionSamplerKind::ConfidenceThreshold),
        other => Err(anyhow!("Unsupported diffusion sampler: {other:?}")),
    }
}

fn diffusion_options_from_args(args: &GenerateArgs) -> Result<DiffusionGenerateOptions> {
    let diffusion = &args.generation.diffusion;
    Ok(DiffusionGenerateOptions {
        max_new_tokens: args.generation.max_tokens,
        temperature: args.sampling.temp,
        sampler: parse_sampler_kind(&diffusion.diffusion_sampler)?,
        confidence_threshold: diffusion.diffusion_threshold,
        max_denoising_steps: diffusion.max_denoising_steps,
        min_canvas_length: diffusion.diffusion_min_canvas_length,
        max_canvas_length: diffusion.diffusion_max_canvas_length,
        full_canvas: diffusion.diffusion_full_canvas,
        // Stop ids from generation_config.json (when present) join the
        // checkpoint's embedded EOS set inside the engine.
        extra_eos_token_ids: mlxcel::read_eos_token_ids(&args.model.model),
        ..DiffusionGenerateOptions::default()
    })
}

fn print_diffusion_stats(stats: &DiffusionGenerationStats, profile: bool) {
    println!();
    println!();
    println!(
        "[Generated {} tokens in {:.2}s = {:.2} tok/s]",
        stats.generated_tokens, stats.generation_time_s, stats.generation_tps
    );
    if profile {
        println!("[Diffusion Profile]");
        println!(
            "  Prompt: {} tokens in {:.3}s = {:.2} tok/s",
            stats.prompt_tokens, stats.prompt_time_s, stats.prompt_tps
        );
        println!(
            "  Canvas: {} tokens across {} blocks = {:.2} tok/s",
            stats.canvas_tokens, stats.blocks, stats.canvas_tps
        );
        println!(
            "  Denoising: {} steps, work {} tokens = {:.2} work tok/s",
            stats.denoising_steps, stats.work_tokens, stats.work_tps
        );
        println!("  Finish reason: {:?}", stats.finish_reason);
    }
}

/// Run one block-diffusion generation for the CLI `generate` flow.
///
/// Text-only (phase 1) and single/multi image input (phase 2) are supported;
/// audio and video inputs are rejected with a clear error.
pub(crate) fn run_diffusion_generation(
    model: &DiffusionGemmaModel,
    args: &GenerateArgs,
    tokenizer: &MlxcelTokenizer,
    prompt_tokens: &[i32],
    user_prompt: &str,
) -> Result<()> {
    if args.generation.audio.is_some() {
        return Err(anyhow!(
            "DiffusionGemma audio input is not supported; run with text or --image input"
        ));
    }
    if !args.generation.video.is_empty() {
        return Err(anyhow!(
            "DiffusionGemma video input is not supported; run with text or --image input"
        ));
    }
    if !args.generation.image.is_empty() && !model.supports_images() {
        return Err(anyhow!(
            "This DiffusionGemma checkpoint does not include a vision tower; \
             run with a text-only prompt"
        ));
    }
    if args.model.draft_model.is_some() {
        return Err(anyhow!(
            "DiffusionGemma does not support speculative decoding (--draft-model); block \
             diffusion already denoises whole canvases per step"
        ));
    }

    let options = diffusion_options_from_args(args)?;
    if let Some(seed) = args.sampling.seed {
        mlxcel_core::random_seed(seed);
    }

    // Image input: preprocess + expand the prompt before seeding the decode
    // state, so the generated-token detokenizer sees the expanded prefix.
    let prepared_vision = if args.generation.image.is_empty() {
        None
    } else {
        Some(prepare_diffusion_vision(
            model,
            Path::new(&args.model.model),
            &args.generation.image,
            prompt_tokens,
        )?)
    };
    let (engine_prompt_tokens, vision_prefill): (&[i32], Option<&DiffusionVisionPrefill>) =
        match &prepared_vision {
            Some(prepared) => (&prepared.expanded_ids, Some(&prepared.prefill)),
            None => (prompt_tokens, None),
        };

    print_generation_preamble(user_prompt)?;

    // Stream through the shared incremental detokenizer (byte-fallback
    // safe); raw ids are collected in parallel so any held-back tail bytes
    // can be flushed from a byte-exact full decode afterward.
    let mut decode_state = StreamingDecodeState::new(tokenizer, engine_prompt_tokens);
    let mut generated_ids: Vec<u32> = Vec::with_capacity(options.max_new_tokens);
    let mut streamed = String::new();
    let mut stdout = io::stdout();

    let stats = model
        .generate_diffusion_streaming(engine_prompt_tokens, &options, vision_prefill, |token_id| {
            generated_ids.push(token_id as u32);
            if let Some(text) = decode_state.on_token(token_id, tokenizer) {
                print!("{text}");
                let _ = stdout.flush();
                streamed.push_str(&text);
            }
            true
        })
        .map_err(|e| anyhow!("{e}"))?;

    // Flush any tail the streaming view held back (e.g. a multi-byte char
    // split across the final tokens).
    let full_text = tokenizer.decode(&generated_ids, true).unwrap_or_default();
    if let Some(tail) = full_text.strip_prefix(&streamed)
        && !tail.is_empty()
    {
        print!("{tail}");
        let _ = stdout.flush();
    }

    print_diffusion_stats(&stats, args.generation.profile);

    mlxcel_core::clear_memory_cache();
    Ok(())
}
