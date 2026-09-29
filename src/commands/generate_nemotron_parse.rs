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

//! CLI driver for Nemotron-Parse page parsing (issue #1369).
//!
//! Nemotron-Parse is encoder-decoder: the page is encoded once and the
//! decoder cross-attends to it, so like Florence-2 it is routed here before
//! the autoregressive loop. `-p` is the task prompt that seeds the decoder
//! (for example the default
//! `</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>`);
//! an empty prompt makes the model loop, so it is replaced by the default
//! with a warning. `--repetition-penalty` applies (the model card's script
//! uses 1.1); the answer is printed with its special tokens so the
//! `<x_..><y_..>` boxes and `<class_..>` tags survive.

use std::time::Instant;

use anyhow::{Context, Result, anyhow, ensure};

use mlxcel::models::NemotronParseVlmModel;
use mlxcel::models::nemotron_parse::DEFAULT_TASK_PROMPT;

use super::generate::print_generation_preamble;
use crate::GenerateArgs;

/// Resolve the `-p` value into the task prompt the decoder is seeded with.
/// Returns the prompt and whether the default was substituted.
pub(crate) fn resolve_task_prompt(user_prompt: &str) -> (&str, bool) {
    if user_prompt.trim().is_empty() {
        (DEFAULT_TASK_PROMPT, true)
    } else {
        (user_prompt, false)
    }
}

/// Parse one page from the CLI flag surface and print the markdown answer
/// plus a generation-stats line.
pub(crate) fn run_nemotron_parse_generation(
    model: &NemotronParseVlmModel,
    args: &GenerateArgs,
    user_prompt: &str,
) -> Result<()> {
    ensure!(
        args.generation.audio.is_none(),
        "Nemotron-Parse does not take --audio input"
    );
    ensure!(
        args.generation.video.is_empty(),
        "Nemotron-Parse does not take --video input"
    );
    ensure!(
        args.generation.image.len() == 1,
        "Nemotron-Parse parses exactly one page image per run: pass --image <path> (got {})",
        args.generation.image.len()
    );

    let (prompt, substituted) = resolve_task_prompt(user_prompt);
    if substituted {
        eprintln!(
            "Warning: Nemotron-Parse needs a task prompt (an empty one makes the decoder loop); \
             using the default {DEFAULT_TASK_PROMPT}"
        );
    }

    // Decode through the shared admission limits so an oversized or
    // decompression-bomb payload is rejected before any pixel work.
    let image_path = &args.generation.image[0];
    let bytes = std::fs::read(image_path)
        .with_context(|| format!("Failed to read image {image_path:?}"))?;
    let mut images =
        mlxcel::decode_image_payloads_with_limits(&[bytes], mlxcel::current_image_input_limits())
            .with_context(|| format!("Failed to decode image {image_path:?}"))?;
    let image = images
        .pop()
        .ok_or_else(|| anyhow!("Image decoding returned no image for {image_path:?}"))?;

    print_generation_preamble(prompt)?;
    println!();

    let started = Instant::now();
    let run = model.run(
        &image,
        prompt,
        args.generation.max_tokens,
        args.sampling.repetition_penalty,
        None,
    )?;
    let elapsed = started.elapsed().as_secs_f64();

    println!("{}", run.text);
    println!();
    let tps = if elapsed > 0.0 {
        run.generated_tokens as f64 / elapsed
    } else {
        0.0
    };
    println!(
        "[Prompt: {} tokens, Generated {} tokens in {:.2}s = {:.2} tok/s]",
        run.prompt_tokens, run.generated_tokens, elapsed, tps
    );

    mlxcel_core::clear_memory_cache();
    Ok(())
}
