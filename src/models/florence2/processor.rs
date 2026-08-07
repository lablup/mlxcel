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

//! The Florence-2 processor: task prompt and image in, structured result out.
//!
//! [`Florence2Processor::run`] is the whole request path in one call:
//!
//! 1. Expand the task marker into its English prompt ([`super::tasks`]).
//! 2. Tokenize it with the checkpoint's BART tokenizer, adding `<s>` / `</s>`.
//! 3. Preprocess the image to 768x768 NCHW, keeping its original size.
//! 4. Drive [`Florence2Model::generate_greedy`], which fuses image features
//!    in front of the prompt embeddings and decodes greedily.
//! 5. Decode the generated ids **keeping special tokens**, because the answer
//!    to a spatial task is mostly `<loc_*>` tokens and dropping them would
//!    leave only the labels.
//! 6. Parse the answer against the *original* image size ([`super::parse`]).
//!
//! Reference:
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/processing_florence2.py

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use image::DynamicImage;

use crate::tokenizer::{MlxcelTokenizer, load_tokenizer};
use crate::vision::processors::florence2::Florence2ImageProcessor;

use super::coords::Florence2ImageSize;
use super::model::Florence2Model;
use super::postprocess::{self, Florence2TaskResult};
use super::tasks::Florence2Task;

/// One task run: the answer as the model wrote it, and its parsed form.
///
/// The raw text is kept alongside the parsed result because it is the only
/// way to tell "the model found nothing" from "the parser rejected what the
/// model wrote", and the two need different fixes.
#[derive(Debug, Clone, PartialEq)]
pub struct Florence2Output {
    /// Decoded answer with special tokens intact, for example
    /// `"<s>car<loc_52><loc_332><loc_932><loc_774>"`.
    pub raw_text: String,
    /// The answer parsed for the requested task.
    pub result: Florence2TaskResult,
}

/// Tokenizer and image preprocessing for a Florence-2 checkpoint.
pub struct Florence2Processor {
    tokenizer: MlxcelTokenizer,
    image_processor: Florence2ImageProcessor,
}

impl Florence2Processor {
    /// Load both halves from a checkpoint directory (`tokenizer.json` +
    /// `preprocessor_config.json`).
    pub fn from_pretrained(model_path: &Path) -> Result<Self> {
        let tokenizer = load_tokenizer(model_path)
            .with_context(|| format!("Florence-2: failed to load tokenizer from {model_path:?}"))?;
        let image_processor =
            Florence2ImageProcessor::from_pretrained(model_path).map_err(|e| anyhow!("{e}"))?;
        Ok(Self {
            tokenizer,
            image_processor,
        })
    }

    /// The tokenizer, for callers that need to decode partial output while
    /// streaming.
    pub fn tokenizer(&self) -> &MlxcelTokenizer {
        &self.tokenizer
    }

    /// The image preprocessing configuration.
    pub fn image_processor(&self) -> &Florence2ImageProcessor {
        &self.image_processor
    }

    /// Expand a task into its prompt sentence and tokenize it.
    ///
    /// `add_special_tokens` is on, so the checkpoint's `RobertaProcessing`
    /// post-processor wraps the ids in `<s>` ... `</s>`; that is the form the
    /// encoder was trained on.
    pub fn encode_prompt(&self, task: Florence2Task, input: Option<&str>) -> Result<Vec<i32>> {
        let prompt = task.expand(input).map_err(|e| anyhow!("{e}"))?;
        let ids = self
            .tokenizer
            .encode(&prompt, true)
            .with_context(|| format!("Florence-2: failed to tokenize prompt {prompt:?}"))?;
        Ok(ids.into_iter().map(|id| id as i32).collect())
    }

    /// Decode generated ids back to text, keeping special tokens.
    ///
    /// Location tokens are marked special in `tokenizer.json`, so skipping
    /// special tokens here would silently delete every coordinate in the
    /// answer and leave the spatial tasks returning empty results.
    pub fn decode_answer(&self, ids: &[i32]) -> Result<String> {
        let ids: Vec<u32> = ids.iter().map(|id| *id as u32).collect();
        self.tokenizer
            .decode(&ids, false)
            .context("Florence-2: failed to decode generated ids")
    }

    /// Parse an answer for `task` against the original image size.
    pub fn post_process(
        &self,
        text: &str,
        task: Florence2Task,
        size: Florence2ImageSize,
    ) -> Florence2TaskResult {
        postprocess::post_process(text, task, size)
    }

    /// Run one task end to end against a loaded model.
    pub fn run(
        &self,
        model: &Florence2Model,
        task: Florence2Task,
        input: Option<&str>,
        image: &DynamicImage,
        max_new_tokens: usize,
    ) -> Result<Florence2Output> {
        let prompt_ids = self.encode_prompt(task, input)?;
        let processed = self
            .image_processor
            .preprocess_with_sizes(std::slice::from_ref(image));
        let (width, height) = *processed
            .original_sizes
            .first()
            .ok_or_else(|| anyhow!("Florence-2: image preprocessing returned no images"))?;

        let generated =
            model.generate_greedy(&processed.pixel_values, &prompt_ids, max_new_tokens)?;
        let raw_text = self.decode_answer(&generated)?;
        let result = self.post_process(&raw_text, task, Florence2ImageSize::new(width, height));
        Ok(Florence2Output { raw_text, result })
    }
}

#[cfg(test)]
#[path = "florence2_processor_tests.rs"]
mod florence2_processor_tests;
