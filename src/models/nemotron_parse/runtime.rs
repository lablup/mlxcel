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

//! Nemotron-Parse runtime unit: model, page processor, and tokenizer.
//!
//! [`NemotronParseVlmModel`] is what `LoadedModel::NemotronParseVLM` stores.
//! Like Florence-2 it is an encoder-decoder model with its own seq2seq decode
//! cache, so the CLI routes it to its own driver before the autoregressive
//! loop and `mlxcel-server` serves it on a dedicated single-stream worker;
//! neither calls the [`LanguageModel`] impl below to generate.

use std::path::Path;
use std::sync::atomic::AtomicBool;

use anyhow::{Context, Result, anyhow};
use image::DynamicImage;

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::tokenizer::{MlxcelTokenizer, load_tokenizer};

use super::model::NemotronParseModel;
use super::processor::{NemotronParseImageProcessor, seed_ids};

/// One page run.
#[derive(Debug, Clone, PartialEq)]
pub struct NemotronParseRunOutput {
    /// Decoded answer with special tokens kept, so the `<x_..><y_..>`
    /// coordinates and `<class_..>` tags survive.
    pub text: String,
    /// Decoder seed length (the tokenized task prompt).
    pub prompt_tokens: usize,
    /// Generated tokens, EOS excluded.
    pub generated_tokens: usize,
    /// Whether decoding ended on `</s>`.
    pub hit_eos: bool,
}

/// The loadable Nemotron-Parse unit.
pub struct NemotronParseVlmModel {
    model: NemotronParseModel,
    image_processor: NemotronParseImageProcessor,
    tokenizer: MlxcelTokenizer,
}

impl NemotronParseVlmModel {
    /// Load model, processor, and tokenizer from one checkpoint directory.
    pub fn load(model_path: &Path) -> Result<Self> {
        let model = NemotronParseModel::load(model_path)?;
        let (h, w) = model.config().vision.image_size;
        let image_processor =
            NemotronParseImageProcessor::from_pretrained(model_path, (h as u32, w as u32))?;
        let patch = model.config().vision.patch_size as u32;
        let kw = model.config().vision.neck_kernel_w as u32;
        let (fh, fw) = image_processor.final_size;
        if fh % patch != 0 || fw % (patch * kw) != 0 {
            return Err(anyhow!(
                "Nemotron-Parse page size {fh}x{fw} is not a multiple of the patch grid \
                 ({patch} rows, {} columns)",
                patch * kw
            ));
        }
        let tokenizer = load_tokenizer(model_path).with_context(|| {
            format!("Nemotron-Parse: failed to load tokenizer from {model_path:?}")
        })?;
        Ok(Self {
            model,
            image_processor,
            tokenizer,
        })
    }

    pub fn model(&self) -> &NemotronParseModel {
        &self.model
    }

    pub fn image_processor(&self) -> &NemotronParseImageProcessor {
        &self.image_processor
    }

    pub fn tokenizer(&self) -> &MlxcelTokenizer {
        &self.tokenizer
    }

    /// Decoder seed for `prompt` (see [`seed_ids`]).
    pub fn seed_ids(&self, prompt: &str) -> Result<Vec<i32>> {
        seed_ids(
            &self.tokenizer,
            prompt,
            self.model.config().decoder_start_token_id,
        )
    }

    /// Parse one page: preprocess, encode, seed with the tokenized `prompt`,
    /// and decode greedily. All state (pixels, encoder output, decode cache)
    /// is local to the call, so consecutive requests share nothing.
    pub fn run(
        &self,
        image: &DynamicImage,
        prompt: &str,
        max_new_tokens: usize,
        repetition_penalty: f32,
        cancel: Option<&AtomicBool>,
    ) -> Result<NemotronParseRunOutput> {
        let seed = self.seed_ids(prompt)?;
        let pixels = self.image_processor.preprocess(image);
        let generation =
            self.model
                .generate(&pixels, &seed, max_new_tokens, repetition_penalty, cancel)?;
        let ids: Vec<u32> = generation.tokens.iter().map(|id| *id as u32).collect();
        let text = self
            .tokenizer
            .decode(&ids, false)
            .context("Nemotron-Parse: failed to decode generated ids")?;
        Ok(NemotronParseRunOutput {
            text,
            prompt_tokens: seed.len(),
            generated_tokens: generation.tokens.len(),
            hit_eos: generation.hit_eos,
        })
    }
}

impl LanguageModel for NemotronParseVlmModel {
    /// Trait-completeness forward only. Nemotron-Parse has no text encoder:
    /// its decoder cross-attends to page features, and there is no page
    /// here. The ids are run through the decoder against a single all-zero
    /// memory row so the call returns correctly shaped `[B, T, vocab]`
    /// logits without the ~13k-token vision pass. Generation goes through
    /// [`NemotronParseVlmModel::run`] (CLI) and the dedicated server worker;
    /// neither reaches this.
    fn forward(
        &self,
        input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(input_ids);
        let (batch, seq) = (shape[0], shape[1]);
        let bound = self.model.config().max_sequence_length;
        let truncated;
        let ids: &MlxArray = if seq > bound {
            tracing::warn!(
                "Nemotron-Parse trait forward: truncating a {seq}-token input to {bound}; use \
                 NemotronParseVlmModel::run to generate"
            );
            truncated = mlxcel_core::slice(input_ids, &[0, 0], &[batch, bound]);
            &truncated
        } else {
            input_ids
        };
        let d_model = self.model.config().text.d_model;
        let memory = mlxcel_core::zeros(&[batch, 1, d_model], self.model.text_dtype());
        let mut cache = self.model.make_cache();
        self.model.decode(ids, &memory, &mut cache)
    }

    /// The seq2seq cache lives in `NemotronParseSeqCache`; nothing goes into
    /// the decoder-only `KVCache` list.
    fn make_caches(&self) -> Vec<KVCache> {
        Vec::new()
    }

    fn num_layers(&self) -> usize {
        self.model.config().text.decoder_layers
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![self.model.config().eos_token_id]
    }

    fn supports_batching(&self) -> bool {
        false
    }

    fn supports_padded_prefill(&self) -> bool {
        false
    }
}
