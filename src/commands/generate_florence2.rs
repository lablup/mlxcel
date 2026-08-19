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

//! CLI driver for Florence-2 task generation (issue #856).
//!
//! Florence-2 is an encoder-decoder (seq2seq) VLM: the encoder consumes the
//! fused image-plus-prompt sequence and the decoder cross-attends to it, so
//! it cannot run on the autoregressive loop `run_generation_mode` drives.
//! `run_generate_once` routes `LoadedModel::Florence2VLM` here before that
//! loop, mirroring the DiffusionGemma / LLaDA-2 early exits. The server side
//! of the same split is `server/florence2_worker.rs` (issue #1073), which
//! prints the identical `render_task_result` text into `message.content` so
//! HTTP answers are comparable against CLI answers byte for byte.
//!
//! The `-p/--prompt` string selects one of the fifteen task modes
//! (`<CAPTION>`, `<OCR>`, `<OD>`, ...), optionally followed by the input
//! text the task interpolates. Image decoding routes through the shared
//! [`mlxcel::ImageInputLimits`] admission bounds so an oversized or
//! decompression-bomb payload is rejected before any pixel work.

use std::time::Instant;

use anyhow::{Context, Result, anyhow, ensure};

use mlxcel::models::florence2::{Florence2VlmModel, parse_task_prompt, render_task_result};

use super::generate::print_generation_preamble;
use crate::GenerateArgs;

/// Run one Florence-2 task from the CLI flag surface and print the parsed
/// answer plus a generation-stats line.
pub(crate) fn run_florence2_generation(
    model: &Florence2VlmModel,
    args: &GenerateArgs,
    user_prompt: &str,
) -> Result<()> {
    ensure!(
        args.generation.audio.is_none(),
        "Florence-2 does not take --audio input"
    );
    ensure!(
        args.generation.video.is_empty(),
        "Florence-2 does not take --video input"
    );
    ensure!(
        !args.generation.image.is_empty(),
        "Florence-2 is an image-task model: pass --image <path> together with a task prompt \
         such as -p '<CAPTION>', -p '<OCR>', or -p '<OD>'"
    );
    ensure!(
        args.generation.image.len() == 1,
        "Florence-2 processes one image per request; got {} --image paths",
        args.generation.image.len()
    );

    let (task, input) = parse_task_prompt(user_prompt).map_err(|e| anyhow!("-p/--prompt: {e}"))?;

    // Decode through the shared admission limits (decompression-bomb
    // defense, issue #855 handoff): `preprocess_with_sizes` takes an
    // already-decoded image, so the bound has to hold at this boundary.
    let image_path = &args.generation.image[0];
    let bytes = std::fs::read(image_path)
        .with_context(|| format!("Failed to read image {image_path:?}"))?;
    let mut images =
        mlxcel::decode_image_payloads_with_limits(&[bytes], mlxcel::current_image_input_limits())
            .with_context(|| format!("Failed to decode image {image_path:?}"))?;
    let image = images
        .pop()
        .ok_or_else(|| anyhow!("Image decoding returned no image for {image_path:?}"))?;

    print_generation_preamble(user_prompt)?;
    println!();

    let started = Instant::now();
    let run = model.run_task(task, input.as_deref(), &image, args.generation.max_tokens)?;
    let elapsed = started.elapsed().as_secs_f64();

    println!(
        "{}",
        render_task_result(&run.output.result, &run.output.raw_text)
    );
    println!();
    let tps = if elapsed > 0.0 {
        run.generated_tokens as f64 / elapsed
    } else {
        0.0
    };
    println!(
        "[Generated {} tokens in {:.2}s = {:.2} tok/s]",
        run.generated_tokens, elapsed, tps
    );
    if args.generation.profile {
        println!("[Raw answer] {}", run.output.raw_text);
    }

    mlxcel_core::clear_memory_cache();
    Ok(())
}
