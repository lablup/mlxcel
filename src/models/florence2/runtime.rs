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

//! Florence-2 runtime registration: the loadable model unit and the CLI
//! task-prompt surface.
//!
//! [`Florence2VlmModel`] bundles the fused [`Florence2Model`] with its
//! [`Florence2Processor`] so the runtime holds one unit that can take a task
//! prompt and an image and return a parsed answer. It is what
//! `LoadedModel::Florence2VLM` stores.
//!
//! Florence-2 is an encoder-decoder (seq2seq) model: the answer is decoded
//! against cached encoder output through cross-attention, with its own
//! [`super::Florence2SeqCache`] rather than the decoder-only `KVCache` list.
//! It therefore cannot run on the autoregressive loop every other
//! `LoadedModel` family uses. The CLI routes this family to
//! [`Florence2VlmModel::run_task`] before the standard generation loop
//! (mirroring the DiffusionGemma early exit in `commands/generate.rs`), and
//! `mlxcel-server` refuses the checkpoint at startup until a seq2seq worker
//! path exists. The [`LanguageModel`] impl below exists for trait
//! completeness (warmup, tooling), not as a generation path.

use std::path::Path;

use anyhow::{Result, anyhow};
use image::DynamicImage;

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use super::coords::Florence2ImageSize;
use super::model::Florence2Model;
use super::processor::{Florence2Output, Florence2Processor};
use super::tasks::Florence2Task;

/// The loadable Florence-2 runtime unit: fused model plus processor.
///
/// Holds MLX weight handles, so the owning provider serializes access.
pub struct Florence2VlmModel {
    model: Florence2Model,
    processor: Florence2Processor,
}

/// One [`Florence2VlmModel::run_task`] call: the parsed answer plus the
/// generated-token count the CLI needs for its throughput line.
pub struct Florence2RunOutput {
    /// Raw decoded answer and its parsed form.
    pub output: Florence2Output,
    /// Number of decoder tokens generated (EOS excluded).
    pub generated_tokens: usize,
}

impl Florence2VlmModel {
    /// Load model and processor from a checkpoint directory. Quantized
    /// checkpoints are rejected by [`Florence2Model::load`] with a named
    /// error; use a bf16 or f16 export.
    pub fn load(model_path: &Path) -> Result<Self> {
        let model = Florence2Model::load(model_path)?;
        let processor = Florence2Processor::from_pretrained(model_path)?;
        Ok(Self { model, processor })
    }

    /// The fused vision-language model.
    pub fn model(&self) -> &Florence2Model {
        &self.model
    }

    /// The task-prompt and image processor.
    pub fn processor(&self) -> &Florence2Processor {
        &self.processor
    }

    /// Run one task end to end: expand and tokenize the task prompt,
    /// preprocess the image, decode greedily, and parse the answer against
    /// the original image size.
    ///
    /// Same pipeline as [`Florence2Processor::run`], kept separate so the
    /// caller also gets the generated-token count for stats.
    pub fn run_task(
        &self,
        task: Florence2Task,
        input: Option<&str>,
        image: &DynamicImage,
        max_new_tokens: usize,
    ) -> Result<Florence2RunOutput> {
        let prompt_ids = self.processor.encode_prompt(task, input)?;
        let processed = self
            .processor
            .image_processor()
            .preprocess_with_sizes(std::slice::from_ref(image));
        let (width, height) = *processed
            .original_sizes
            .first()
            .ok_or_else(|| anyhow!("Florence-2: image preprocessing returned no images"))?;

        let generated =
            self.model
                .generate_greedy(&processed.pixel_values, &prompt_ids, max_new_tokens)?;
        let generated_tokens = generated.len();
        let raw_text = self.processor.decode_answer(&generated)?;
        let result =
            self.processor
                .post_process(&raw_text, task, Florence2ImageSize::new(width, height));
        Ok(Florence2RunOutput {
            output: Florence2Output { raw_text, result },
            generated_tokens,
        })
    }
}

/// Parse a CLI `-p/--prompt` string into a Florence-2 task and its optional
/// input text.
///
/// Accepted forms:
/// - `"<OD>"` or `"od"`: a bare task marker, angle brackets optional,
///   case-insensitive (per [`Florence2Task::from_str`]).
/// - `"<CAPTION_TO_PHRASE_GROUNDING> a green car"` or
///   `"CAPTION_TO_PHRASE_GROUNDING a green car"`: marker followed by the
///   input text the task interpolates.
/// - `"<REGION_TO_CATEGORY><loc_52><loc_332><loc_932><loc_774>"`: a region
///   input needs no separating space, everything after the first closing
///   `>` is the input.
///
/// Whether the task actually takes input text is validated later by
/// [`Florence2Task::expand`], which rejects a missing or superfluous input;
/// this function only splits the syntax. Anything that does not start with a
/// recognized marker is an error listing the valid markers, matching the
/// deliberate strictness of `expand` (upstream silently misparses sloppy
/// prompts into nonsense questions).
pub fn parse_task_prompt(prompt: &str) -> Result<(Florence2Task, Option<String>), String> {
    let trimmed = prompt.trim();
    if trimmed.is_empty() {
        return Err(format!(
            "empty Florence-2 prompt; pass a task marker such as {}",
            valid_task_markers()
        ));
    }

    if let Some(rest) = trimmed.strip_prefix('<') {
        let Some(end) = rest.find('>') else {
            return Err(format!(
                "Florence-2 prompt {trimmed:?} opens a task marker without a closing '>'"
            ));
        };
        let task: Florence2Task = rest[..end].parse().map_err(|_| {
            format!(
                "unknown Florence-2 task marker <{}>; valid markers: {}",
                &rest[..end],
                valid_task_markers()
            )
        })?;
        let input = rest[end + 1..].trim();
        let input = (!input.is_empty()).then(|| input.to_string());
        return Ok((task, input));
    }

    // Bare form: the whole prompt is a task name, or a task name followed by
    // input text.
    if let Ok(task) = trimmed.parse::<Florence2Task>() {
        return Ok((task, None));
    }
    if let Some((head, rest)) = trimmed.split_once(char::is_whitespace)
        && let Ok(task) = head.parse::<Florence2Task>()
    {
        let input = rest.trim();
        let input = (!input.is_empty()).then(|| input.to_string());
        return Ok((task, input));
    }

    Err(format!(
        "prompt {trimmed:?} does not start with a Florence-2 task; valid markers: {}",
        valid_task_markers()
    ))
}

fn valid_task_markers() -> String {
    Florence2Task::ALL
        .iter()
        .map(|task| task.token())
        .collect::<Vec<_>>()
        .join(", ")
}

impl LanguageModel for Florence2VlmModel {
    /// Honest minimal trait forward: a text-only teacher-forced BART pass.
    /// The prompt is encoded by the bidirectional encoder, and the decoder
    /// consumes the same ids shifted right behind `decoder_start_token_id`
    /// (HuggingFace `shift_tokens_right`), returning per-position logits.
    ///
    /// A fresh seq2seq cache is built per call and the passed decoder-only
    /// `caches` are ignored: cross-attention K/V cannot live in `KVCache`,
    /// so an incremental decode driven through this trait would silently
    /// re-encode each step. The CLI routes Florence-2 to
    /// [`Florence2VlmModel::run_task`] before the autoregressive loop and the
    /// server refuses the checkpoint at startup, so this exists for trait
    /// completeness (warmup, tooling) rather than as a generation path.
    fn forward(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let text = self.model.text_model();
        let encoder_hidden = text.encode_tokens(input_ids);

        let shape = mlxcel_core::array_shape(input_ids);
        let (batch, seq) = (shape[0], shape[1]);
        let start = text.config().decoder_start_token_id;
        let start_col =
            mlxcel_core::from_slice_i32(&vec![start; batch.max(1) as usize], &[batch, 1]);
        let decoder_input = if seq > 1 {
            let prefix = mlxcel_core::slice(input_ids, &[0, 0], &[batch, seq - 1]);
            mlxcel_core::concatenate(&start_col, &prefix, 1)
        } else {
            start_col
        };

        let mut cache = self.model.make_cache();
        text.decode(&decoder_input, &encoder_hidden, &mut cache)
    }

    /// Florence-2 keeps its own dual (self + cross attention) cache, built by
    /// [`Florence2Model::make_cache`]; there is nothing to store in the
    /// decoder-only `KVCache` list.
    fn make_caches(&self) -> Vec<KVCache> {
        Vec::new()
    }

    fn num_layers(&self) -> usize {
        self.model.config().text.decoder_layers as usize
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![self.model.config().text.eos_token_id]
    }

    /// Seq2seq generation is a model-owned loop over a single sequence; the
    /// batched/paged scheduler must never pick this model up.
    fn supports_batching(&self) -> bool {
        false
    }

    fn supports_padded_prefill(&self) -> bool {
        false
    }
}

#[cfg(test)]
#[path = "florence2_runtime_tests.rs"]
mod florence2_runtime_tests;
