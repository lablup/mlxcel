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

//! Qwen3-VL-Embedding: the generative Qwen3-VL stack pooled at its last token.
//!
//! The checkpoint is an ordinary `Qwen3VLForConditionalGeneration` export
//! (`model_type: qwen3_vl`) with a sentence-transformers `1_Pooling` module
//! declaring `pooling_mode: lasttoken`, so every weight, the vision tower,
//! the DeepStack mergers and the interleaved M-RoPE text decoder included,
//! loads through the existing VLM loader
//! ([`crate::loading::load_qwen3_vl`]). The only differences from generation
//! are that the tied `lm_head` is never applied (the backbone stops at the
//! final norm through [`crate::models::qwen3_vl::Qwen3VLModel::forward_hidden`])
//! and that the input is wrapped in the checkpoint's own chat template.
//!
//! Text formatting renders
//!
//! ```text
//! <|im_start|>system
//! {instruction}<|im_end|>
//! <|im_start|>user
//! [<|vision_start|><|image_pad|><|vision_end|>]{text}<|im_end|>
//! <|im_start|>assistant
//! ```
//!
//! with `add_generation_prompt = true`, and the pooled position is the final
//! `\n` of the assistant header. The instruction defaults to the checkpoint's
//! `config_sentence_transformers.json` prompt (`Represent the user's input.`)
//! and a caller-supplied instruction that does not end in punctuation gets a
//! trailing `.`, matching the reference wrapper.
//!
//! Image inputs take one extra step. The engine tokenizes before the family
//! sees the image, so `format_text` emits exactly one `<|image_pad|>`; the
//! forward pass expands that single placeholder into the patch count the
//! processor computed for this image before merging the vision features.
//! Image batches are always one row: Qwen3-VL's DeepStack injection and the
//! M-RoPE deltas are per sequence.

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlxcel_core::utils::create_causal_padding_mask;
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::{Value, json};

use crate::embeddings::limits::{config_normalize_flag, read_json};
use crate::embeddings::model::{EmbeddingBatch, EmbeddingModel, EmbeddingOutput, ImageInput};
use crate::embeddings::pooling::{PoolingMode, pool, resolve_pooling_mode};
use crate::server::chat_template::ChatTemplateProcessor;

/// Instruction used when the request carries none, and the fallback when the
/// checkpoint ships no `config_sentence_transformers.json` prompt.
pub const DEFAULT_INSTRUCTION: &str = "Represent the user's input.";

/// Qwen3-VL-Embedding: causal Qwen3-VL stack, last-token pooling.
pub struct Qwen3VLEmbeddingModel {
    vlm: crate::vision::Qwen3VLModel,
    chat_template: ChatTemplateProcessor,
    default_instruction: String,
    pooling: PoolingMode,
    normalize: bool,
    embedding_dim: usize,
}

/// Append a `.` when `instruction` ends on an alphanumeric character.
///
/// The reference wrapper only guards against an instruction that reads as an
/// unfinished sentence, so anything already ending in punctuation (ASCII or
/// not) is left alone.
pub(crate) fn with_trailing_punctuation(instruction: &str) -> String {
    match instruction.chars().next_back() {
        Some(last) if last.is_alphanumeric() => format!("{instruction}."),
        _ => instruction.to_string(),
    }
}

/// Read the checkpoint's default prompt from
/// `config_sentence_transformers.json` (`prompts[default_prompt_name]`).
fn checkpoint_default_instruction(model_dir: &Path) -> Option<String> {
    let config = read_json(&model_dir.join("config_sentence_transformers.json"))?;
    let name = config.get("default_prompt_name")?.as_str()?;
    let prompt = config.get("prompts")?.get(name)?.as_str()?;
    (!prompt.trim().is_empty()).then(|| prompt.to_string())
}

/// Apply the checkpoint's `preprocessor_config.json` pixel bounds to the
/// shared Qwen2-VL processor.
///
/// The generative loader constructs the processor with the Qwen2-VL defaults
/// (a 12.8M pixel ceiling, so up to 12544 visual tokens for a 16px patch and
/// a merge size of 2). This checkpoint declares 1310720, which keeps one
/// image at 1280 tokens and therefore well inside the embedder's 8192-token
/// budget. Reading the file rather than hard-coding it keeps a re-exported
/// checkpoint with different bounds correct.
fn apply_pixel_bounds(
    processor: &mut crate::vision::processors::qwen2_vl::Qwen2VLProcessor,
    model_dir: &Path,
) {
    let Some(config) = read_json(&model_dir.join("preprocessor_config.json")) else {
        return;
    };
    let read = |key: &str| config.get(key).and_then(Value::as_u64).filter(|&v| v > 0);
    if let Some(min_pixels) = read("min_pixels") {
        processor.min_pixels = min_pixels as usize;
    }
    if let Some(max_pixels) = read("max_pixels") {
        processor.max_pixels = max_pixels as usize;
    }
}

/// Replace every `image_token_id` in a single-row `[1, L]` id array with
/// `counts` consecutive copies, expanding the mask alongside it.
///
/// Returns `(input_ids, attention_mask)` of the expanded width. `counts` is
/// consumed in placeholder order, so a request carrying several images maps
/// each placeholder to its own patch count.
pub(crate) fn expand_image_placeholders(
    ids: &[i32],
    mask: &[i32],
    image_token_id: i32,
    counts: &[usize],
) -> Result<(Vec<i32>, Vec<i32>)> {
    let placeholders = ids.iter().filter(|&&id| id == image_token_id).count();
    if placeholders != counts.len() {
        bail!(
            "Qwen3-VL-Embedding: the prompt carries {placeholders} image placeholder(s) but \
             {} image(s) were preprocessed",
            counts.len()
        );
    }
    let extra: usize = counts.iter().map(|&c| c.saturating_sub(1)).sum();
    let mut out_ids = Vec::with_capacity(ids.len() + extra);
    let mut out_mask = Vec::with_capacity(ids.len() + extra);
    let mut next = 0usize;
    for (&id, &flag) in ids.iter().zip(mask) {
        if id == image_token_id {
            let count = counts[next];
            next += 1;
            if count == 0 {
                bail!("Qwen3-VL-Embedding: an image expanded to zero visual tokens");
            }
            out_ids.extend(std::iter::repeat_n(id, count));
            out_mask.extend(std::iter::repeat_n(flag, count));
        } else {
            out_ids.push(id);
            out_mask.push(flag);
        }
    }
    Ok((out_ids, out_mask))
}

/// Render the checkpoint's chat template around one input.
///
/// `with_image` inserts the `{"type": "image"}` content item the template
/// expands into `<|vision_start|><|image_pad|><|vision_end|>`; an empty
/// `text` contributes no content item at all.
///
/// Free rather than a method so the rendering can be exercised against a
/// checkpoint's `chat_template.jinja` without loading its weights.
pub(crate) fn render_prompt(
    chat_template: &ChatTemplateProcessor,
    instruction: &str,
    text: &str,
    with_image: bool,
) -> Result<String> {
    let mut user: Vec<Value> = Vec::with_capacity(2);
    if with_image {
        user.push(json!({"type": "image"}));
    }
    if !text.is_empty() {
        user.push(json!({"type": "text", "text": text}));
    }
    let messages = json!([
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": user},
    ]);
    chat_template.apply_raw(&messages, None)
}

impl Qwen3VLEmbeddingModel {
    /// Load a Qwen3-VL-Embedding checkpoint from `model_dir`.
    pub fn load(model_dir: &Path, config: &Value) -> Result<Self> {
        let loaded = crate::loading::load_qwen3_vl(model_dir)
            .with_context(|| format!("failed to load {} as Qwen3-VL", model_dir.display()))?;
        let crate::LoadedModel::Qwen3VL(mut vlm) = loaded else {
            bail!(
                "{} did not load as a Qwen3-VL stack; Qwen3-VL-Embedding needs the dense \
                 Qwen3-VL layout",
                model_dir.display()
            );
        };
        apply_pixel_bounds(&mut vlm.processor, model_dir);

        let chat_template = ChatTemplateProcessor::from_model_path(model_dir)
            .with_context(|| {
                format!(
                    "failed to read the chat template in {}",
                    model_dir.display()
                )
            })?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "{} ships no chat template; Qwen3-VL-Embedding needs one to wrap its \
                     instruction and input",
                    model_dir.display()
                )
            })?;

        let embedding_dim = config
            .get("text_config")
            .and_then(|text| text.get("hidden_size"))
            .and_then(Value::as_u64)
            .filter(|&v| v > 0)
            .map(|v| v as usize)
            .ok_or_else(|| {
                anyhow::anyhow!("Qwen3-VL-Embedding: config.json has no text_config.hidden_size")
            })?;

        Ok(Self {
            vlm,
            chat_template,
            default_instruction: checkpoint_default_instruction(model_dir)
                .unwrap_or_else(|| DEFAULT_INSTRUCTION.to_string()),
            pooling: resolve_pooling_mode(model_dir, PoolingMode::LastToken)?,
            normalize: config_normalize_flag(config),
            embedding_dim,
        })
    }

    /// The instruction actually rendered for a request.
    pub(crate) fn resolve_instruction(&self, instruction: Option<&str>) -> String {
        let task = instruction
            .map(str::trim)
            .filter(|task| !task.is_empty())
            .unwrap_or(&self.default_instruction);
        with_trailing_punctuation(task)
    }

    /// Render this checkpoint's chat template around one input.
    pub(crate) fn prompt_for(
        &self,
        text: &str,
        instruction: Option<&str>,
        with_image: bool,
    ) -> Result<String> {
        render_prompt(
            &self.chat_template,
            &self.resolve_instruction(instruction),
            text,
            with_image,
        )
    }

    /// Text-only forward: no vision tower, no DeepStack, causal mask over the
    /// right-padded batch.
    fn embed_text_batch(&self, batch: &EmbeddingBatch) -> UniquePtr<MlxArray> {
        let text_model = &self.vlm.text_model;
        text_model.clear_mrope_state();
        text_model.clear_deepstack_state();
        let mask = create_causal_padding_mask(batch.attention_mask, 0);
        let mut caches = text_model.make_caches();
        text_model.forward_hidden(batch.input_ids, None, &mut caches, Some(&mask))
    }

    /// Visual tokens each image contributes, in input order.
    ///
    /// This is the grid the processor will actually feed the vision tower,
    /// so the count the engine expands the prompt with and the count
    /// [`Self::embed_image_row`] merges are derived from the same call.
    fn visual_token_counts(&self, images: &[ImageInput]) -> Vec<usize> {
        let decoded: Vec<image::DynamicImage> =
            images.iter().map(|input| input.image.clone()).collect();
        let merge = self.vlm.spatial_merge_size.max(1) as i32;
        self.vlm
            .processor
            .compute_grid_thw(&decoded)
            .into_iter()
            .map(|(t, h, w)| (t * (h / merge) * (w / merge)).max(0) as usize)
            .collect()
    }

    /// Image forward for a single row. The engine has already expanded the
    /// placeholder through [`EmbeddingModel::expand_image_tokens`], so the
    /// batch's ids and mask are the ones the forward pass consumes.
    fn embed_image_row(
        &self,
        batch: &EmbeddingBatch,
        images: &[ImageInput],
    ) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)> {
        let shape = mlxcel_core::array_shape(batch.input_ids);
        if shape[0] != 1 {
            bail!(
                "Qwen3-VL-Embedding embeds images one at a time (DeepStack injection and the \
                 M-RoPE delta are per sequence), got a batch of {}",
                shape[0]
            );
        }
        let decoded: Vec<image::DynamicImage> =
            images.iter().map(|input| input.image.clone()).collect();
        let (pixel_values, grid_thw) = self.vlm.processor.preprocess_with_grid(&decoded);

        let text_model = &self.vlm.text_model;
        text_model.clear_mrope_state();
        text_model.clear_deepstack_state();
        // Populates the fallback M-RoPE slot and the DeepStack state that
        // `forward_hidden` reads back for this single sequence.
        let merged = self
            .vlm
            .get_input_embeddings(batch.input_ids, &pixel_values, &grid_thw);
        let causal = create_causal_padding_mask(batch.attention_mask, 0);
        let mut caches = text_model.make_caches();
        let hidden = text_model.forward_hidden(
            batch.input_ids,
            Some(&merged.inputs_embeds),
            &mut caches,
            Some(&causal),
        );
        text_model.clear_mrope_state();
        text_model.clear_deepstack_state();
        Ok((hidden, mlxcel_core::copy(batch.attention_mask)))
    }
}

impl EmbeddingModel for Qwen3VLEmbeddingModel {
    fn embed(&self, batch: &EmbeddingBatch) -> Result<EmbeddingOutput> {
        let images = batch.images.unwrap_or(&[]);
        let (hidden, mask) = if images.is_empty() {
            (
                self.embed_text_batch(batch),
                mlxcel_core::copy(batch.attention_mask),
            )
        } else {
            self.embed_image_row(batch, images)?
        };
        let pooled = pool(&hidden, &mask, self.pooling);
        Ok(EmbeddingOutput {
            embeddings: pooled,
            last_hidden_state: None,
        })
    }

    fn default_pooling(&self) -> PoolingMode {
        PoolingMode::LastToken
    }

    fn normalize(&self) -> bool {
        self.normalize
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    fn supports_images(&self) -> bool {
        true
    }

    /// Wrap the input in the checkpoint's chat template.
    ///
    /// An empty `text` is the engine's image call ([`crate::embeddings::EmbeddingEngine::embed_image`]);
    /// every text path rejects an empty string before it reaches here, so the
    /// empty string is an unambiguous "this row carries an image" signal. The
    /// single `<|image_pad|>` the template emits is expanded to the patch
    /// count by [`EmbeddingModel::expand_image_tokens`] below.
    fn format_text(&self, text: &str, instruction: Option<&str>) -> String {
        match self.prompt_for(text, instruction, text.is_empty()) {
            Ok(prompt) => prompt,
            Err(err) => {
                // `format_text` cannot fail in the trait, and a template that
                // does not render is a load-time defect rather than a per-row
                // one. Falling back to the raw text keeps the request alive
                // with a loud log instead of silently embedding nothing.
                tracing::error!(
                    target: "mlxcel::embeddings",
                    "Qwen3-VL-Embedding chat template failed to render: {err:#}"
                );
                text.to_string()
            }
        }
    }

    /// Expand the one `<|image_pad|>` the chat template emits into the patch
    /// count the Qwen2-VL processor computes for this image.
    ///
    /// Running before padding is what keeps `usage.prompt_tokens` describing
    /// the sequence the forward pass actually sees.
    fn expand_image_tokens(&self, ids: &[u32], images: &[ImageInput]) -> Result<Vec<u32>> {
        if images.is_empty() {
            return Ok(ids.to_vec());
        }
        let counts = self.visual_token_counts(images);
        let signed: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
        let mask = vec![1i32; signed.len()];
        let (expanded, _) =
            expand_image_placeholders(&signed, &mask, self.vlm.image_token_id, &counts)?;
        Ok(expanded.into_iter().map(|id| id as u32).collect())
    }
}

#[cfg(test)]
#[path = "qwen3_vl_embedding_tests.rs"]
mod qwen3_vl_embedding_tests;
