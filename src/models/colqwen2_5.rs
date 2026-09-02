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

//! ColQwen2.5 (`model_type: qwen2_5_vl` with `architectures:
//! ["ColQwen2_5"]`, or the native `model_type: colqwen2` with
//! `["ColQwen2ForRetrieval"]`): the Qwen2.5-VL stack turned into a
//! late-interaction visual document retriever.
//!
//! The backbone is the one `mlxcel generate` already runs: the windowed
//! Qwen2.5-VL vision tower, the patch merger, and the Qwen2 decoder with
//! M-RoPE. Retrieval changes only the ends:
//!
//! - the decoder stops at its final norm through
//!   [`crate::models::Qwen2VLModel::forward_hidden`], which is the exact
//!   prefix of the generation forward pass, so no `[B, L, vocab_size]`
//!   logit tensor is ever built;
//! - `custom_text_proj` (`[128, 2048]` plus bias) projects every token's
//!   hidden state to 128 dimensions, each token vector is L2-normalized and
//!   padding rows are zeroed;
//! - similarity is MaxSim over the token sets
//!   ([`crate::embeddings::maxsim`]), so [`EmbeddingModel::multi_vector`] is
//!   `true` and the engine returns `[num_real_tokens, 128]` per input.
//!
//! Position ids follow the input: a text-only micro-batch uses the
//! sequential `[3, B, L]` positions the backbone builds for a text prefill,
//! and an image input keeps the real M-RoPE grid `compute_rope_index`
//! derives. The per-request M-RoPE slot is cleared before a text batch so a
//! previous image request cannot leak its stored positions into it.
//!
//! Prompt formats follow the reference `ColQwen2_5_Processor`: an image
//! document is
//! `<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|endoftext|>`
//! and a query is `Query: {text}` followed by ten `<|endoftext|>`
//! augmentation tokens.
//!
//! The raw HuggingFace export stores the vision tower under `visual.*` and
//! keeps `Conv3d`'s native `[out, in, kT, kH, kW]` patch-embedding layout,
//! while the encoder in this tree expects the mlx-converted names and the
//! channels-last filter; [`rewrite_colqwen25_key`] and the encoder module's
//! [`normalize_patch_embed_layout`] bridge both. The layout normalizer is
//! shared with the Qwen2.5-VL generation loader, which needs it for the same
//! reason.

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::utils::create_causal_padding_mask;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::Value;

use crate::embeddings::loader::{load_embedding_weights, quantization_params};
use crate::embeddings::model::{EmbeddingBatch, EmbeddingModel, EmbeddingOutput, ImageInput};
use crate::embeddings::pooling::PoolingMode;
use crate::models::col_late_interaction::{
    apply_dense_projection_override, embedding_dim, format_query, project_and_normalize,
    reject_lora_only_checkpoint,
};
use crate::models::qwen2_vl::{Qwen2VLConfig, Qwen2VLModel};
use crate::multimodal::qwen_vl::insert_qwen_vl_image_tokens;
use crate::vision::Qwen25VLModel;
use crate::vision::encoders::qwen2_5_vl::{
    Qwen25VLVisionConfig, Qwen25VLVisionEncoder, normalize_patch_embed_layout,
};
use crate::vision::processors::qwen2_vl::Qwen2VLProcessor;

/// Checkpoint key of the 128-dimension projection after sanitization.
const PROJECTION_PREFIX: &str = "custom_text_proj";

/// Weight prefix the Qwen2.5-VL vision encoder is loaded from. The raw
/// HuggingFace export stores the tower as `visual.*`; mlx conversions
/// already store `vision_tower.*`.
const VISION_PREFIX: &str = "vision_tower";

/// Default vision token ids, matching the generation loader.
const DEFAULT_IMAGE_TOKEN_ID: i64 = 151_655;
const DEFAULT_VIDEO_TOKEN_ID: i64 = 151_656;
const DEFAULT_VISION_START_TOKEN_ID: i64 = 151_652;

/// Query augmentation token (the checkpoint's padding token).
const ENDOFTEXT: &str = "<|endoftext|>";

/// The document prompt the reference processor renders for an image item
/// (`ColQwen2_5_Processor.visual_prompt_prefix`). The turn is closed with
/// `<|endoftext|>` rather than an assistant header: this is a retriever, so
/// nothing is ever generated after the page.
const IMAGE_DOCUMENT_PROMPT: &str = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|endoftext|>";

/// ColQwen2.5: the Qwen2.5-VL stack plus a 128-dimension projection.
pub struct ColQwen25Model {
    vlm: Qwen25VLModel,
    custom_text_proj: UnifiedLinear,
    embedding_dim: usize,
}

impl ColQwen25Model {
    /// Load a ColQwen2.5 checkpoint from `model_dir`.
    pub fn load(model_dir: &Path, config: &Value) -> Result<Self> {
        reject_lora_only_checkpoint(model_dir)?;

        let mut text_config: Qwen2VLConfig = serde_json::from_value(config.clone())
            .context("failed to parse the ColQwen2.5 config.json as a Qwen2-VL text config")?;
        // The embedder stops at the final norm, so no head is ever applied.
        // Forcing the tied flag keeps an untied `lm_head` out of memory and
        // keeps the constructor from failing on a head this path never reads.
        text_config.tie_word_embeddings = true;

        let mut vision_config: Qwen25VLVisionConfig = serde_json::from_value(
            config
                .get("vision_config")
                .cloned()
                .ok_or_else(|| anyhow!("ColQwen2.5 config.json has no `vision_config`"))?,
        )
        .context("failed to parse the ColQwen2.5 `vision_config` block")?;
        if let Some(quant) = quantization_params(config) {
            if vision_config.quant_group_size == 0 {
                vision_config.quant_group_size = quant.group_size;
            }
            if vision_config.quant_bits == 0 {
                vision_config.quant_bits = quant.bits;
            }
        }

        let mut weights = sanitize_colqwen25_weights(load_embedding_weights(model_dir, config)?);
        normalize_patch_embed_layout(&mut weights, VISION_PREFIX, vision_config.in_channels);
        apply_dense_projection_override(&mut weights, PROJECTION_PREFIX);
        weights.retain(|key, _| !key.starts_with("lm_head."));

        let text_model = Qwen2VLModel::from_weights(&weights, &text_config)
            .map_err(|e| anyhow!("ColQwen2.5 text backbone: {e}"))?;
        let vision_encoder =
            Qwen25VLVisionEncoder::from_weights(&weights, &vision_config, VISION_PREFIX)
                .map_err(|e| anyhow!("ColQwen2.5 vision encoder: {e}"))?;

        let (group_size, bits) =
            quantization_params(config).map_or((0, 0), |quant| (quant.group_size, quant.bits));
        let custom_text_proj =
            UnifiedLinear::from_weights(&weights, PROJECTION_PREFIX, group_size, bits).map_err(
                |e| {
                    anyhow!(
                        "ColQwen2.5 projection `{PROJECTION_PREFIX}` is missing or malformed: {e}"
                    )
                },
            )?;

        let processor = image_processor(model_dir, &vision_config);
        let token_id = |key: &str, fallback: i64| -> i32 {
            config.get(key).and_then(Value::as_i64).unwrap_or(fallback) as i32
        };

        let vlm = Qwen25VLModel {
            text_model,
            vision_encoder,
            processor,
            image_token_id: token_id("image_token_id", DEFAULT_IMAGE_TOKEN_ID),
            video_token_id: token_id("video_token_id", DEFAULT_VIDEO_TOKEN_ID),
            vision_start_token_id: token_id("vision_start_token_id", DEFAULT_VISION_START_TOKEN_ID),
            spatial_merge_size: vision_config.spatial_merge_size,
        };

        Ok(Self {
            vlm,
            custom_text_proj,
            embedding_dim: embedding_dim(config),
        })
    }

    /// `[B, L, 128]` token vectors for one padded micro-batch.
    fn forward(&self, batch: &EmbeddingBatch) -> Result<UniquePtr<MlxArray>> {
        let images = batch.images.unwrap_or(&[]);
        let embeddings = if images.is_empty() {
            text_input_embeddings(&self.vlm.text_model, batch.input_ids)
        } else {
            // `compute_rope_index` reads row 0 of `input_ids` and derives one
            // M-RoPE grid from it, so an image batch must be a single row.
            // The engine embeds images one at a time, which is what makes
            // this an invariant rather than a limitation; failing loudly
            // beats silently giving rows 1.. the first row's positions.
            let batch_size = mlxcel_core::array_shape(batch.input_ids)[0];
            if batch_size != 1 {
                bail!(
                    "ColQwen2.5 embeds images one input at a time; got {batch_size} rows with \
                     {} image(s)",
                    images.len()
                );
            }
            let decoded: Vec<image::DynamicImage> =
                images.iter().map(|input| input.image.clone()).collect();
            let (pixel_values, grid_thw) = self.vlm.processor.preprocess_with_grid(&decoded);
            // Also sets this request's M-RoPE positions on the backbone.
            self.vlm
                .get_input_embeddings(batch.input_ids, &pixel_values, &grid_thw)
                .inputs_embeds
        };
        Ok(token_vectors(
            &self.vlm.text_model,
            &self.custom_text_proj,
            batch.input_ids,
            &embeddings,
            batch.attention_mask,
        ))
    }
}

/// Token embeddings for a text-only micro-batch.
///
/// Clearing the per-request M-RoPE slot first is load-bearing: an earlier
/// image request leaves its `[3, 1, L]` grid there, and
/// [`Qwen2VLModel::forward_hidden`] reuses a stored grid whenever it covers
/// the requested range. Without the clear, a text batch of the same width
/// would silently be given an image's spatial positions.
pub(crate) fn text_input_embeddings(
    text_model: &Qwen2VLModel,
    input_ids: &MlxArray,
) -> UniquePtr<MlxArray> {
    text_model.clear_mrope_state();
    text_model.get_embed_tokens(input_ids)
}

/// Run the causal stack over one padded micro-batch and project every token
/// to the retrieval width.
///
/// Shared with the tests so the guarded ordering above (mask, fresh caches,
/// head-free forward, per-token normalization) is exercised as product code
/// rather than restated.
pub(crate) fn token_vectors(
    text_model: &Qwen2VLModel,
    projection: &UnifiedLinear,
    input_ids: &MlxArray,
    input_embeddings: &MlxArray,
    attention_mask: &MlxArray,
) -> UniquePtr<MlxArray> {
    let mask = create_causal_padding_mask(attention_mask, 0);
    let mut caches = text_model.make_caches();
    let hidden = text_model.forward_hidden(
        input_ids,
        Some(input_embeddings),
        &mut caches,
        Some(&mask),
        None,
    );
    project_and_normalize(&hidden, projection, attention_mask)
}

impl EmbeddingModel for ColQwen25Model {
    fn embed(&self, batch: &EmbeddingBatch) -> Result<EmbeddingOutput> {
        Ok(EmbeddingOutput {
            embeddings: self.forward(batch)?,
            last_hidden_state: None,
        })
    }

    /// Reported for the startup log only: a late-interaction family keeps
    /// every token vector and never pools.
    fn default_pooling(&self) -> PoolingMode {
        PoolingMode::LastToken
    }

    fn multi_vector(&self) -> bool {
        true
    }

    fn supports_images(&self) -> bool {
        true
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// An empty text is the engine's image path (`embed_image` formats the
    /// empty string), and a non-empty one is a query. Text items can never
    /// be empty: the route and the engine both reject an empty string
    /// before it reaches here.
    fn format_text(&self, text: &str, _instruction: Option<&str>) -> String {
        if text.is_empty() {
            IMAGE_DOCUMENT_PROMPT.to_string()
        } else {
            format_query(text, ENDOFTEXT)
        }
    }

    fn expand_image_tokens(&self, ids: &[u32], images: &[ImageInput]) -> Result<Vec<u32>> {
        if images.is_empty() {
            return Ok(ids.to_vec());
        }
        let decoded: Vec<image::DynamicImage> =
            images.iter().map(|input| input.image.clone()).collect();
        let grid_thw = self.vlm.processor.compute_grid_thw(&decoded);
        let mut tokens: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
        insert_qwen_vl_image_tokens(
            &mut tokens,
            &grid_thw,
            self.vlm.spatial_merge_size,
            self.vlm.vision_start_token_id,
            self.vlm.image_token_id,
        )
        .ok_or_else(|| {
            anyhow!(
                "ColQwen2.5 could not expand the `<|image_pad|>` placeholder for {} image(s); the \
                 rendered prompt carried a placeholder count that does not match",
                images.len()
            )
        })?;
        Ok(tokens.into_iter().map(|token| token as u32).collect())
    }
}

/// Rewrite one checkpoint key into the layout the Qwen2.5-VL constructors
/// expect.
///
/// Three layouts reach this function:
///
/// - `vidore/colqwen2.5-base` and `-v0.2` merged exports: `model.*`,
///   `visual.*`, `custom_text_proj.*`;
/// - the native `transformers` retrieval layout
///   (`ColQwen2ForRetrieval`): everything nested under `vlm.`, with the
///   projection named `embedding_proj_layer.*`;
/// - mlx conversions of Qwen2.5-VL: already `model.*` and `vision_tower.*`.
///
/// The `vlm.` wrapper is stripped first, so the remaining rules do not have
/// to be written twice.
#[must_use]
pub(crate) fn rewrite_colqwen25_key(key: &str) -> String {
    let key = key.strip_prefix("vlm.").unwrap_or(key);
    if let Some(rest) = key.strip_prefix("embedding_proj_layer.") {
        format!("{PROJECTION_PREFIX}.{rest}")
    } else if let Some(rest) = key.strip_prefix("model.language_model.") {
        format!("model.{rest}")
    } else if let Some(rest) = key.strip_prefix("model.visual.") {
        format!("{VISION_PREFIX}.{rest}")
    } else if let Some(rest) = key.strip_prefix("visual.") {
        format!("{VISION_PREFIX}.{rest}")
    } else if let Some(rest) = key.strip_prefix("language_model.") {
        rest.to_string()
    } else {
        key.to_string()
    }
}

/// Apply [`rewrite_colqwen25_key`] to a whole weight map.
#[must_use]
pub(crate) fn sanitize_colqwen25_weights(weights: WeightMap) -> WeightMap {
    weights
        .into_iter()
        .map(|(key, value)| (rewrite_colqwen25_key(&key), value))
        .collect()
}

/// Build the dynamic-resolution processor, honoring the checkpoint's pixel
/// budget.
///
/// `preprocessor_config.json` caps ColQwen2.5 at `768 * 28 * 28` pixels,
/// which is what bounds an image at 768 visual tokens; the shared
/// `Qwen2VLProcessor` default is 64 times larger, so reading the budget is
/// what keeps a page from expanding into thousands of token vectors.
fn image_processor(model_dir: &Path, vision_config: &Qwen25VLVisionConfig) -> Qwen2VLProcessor {
    let mut processor = Qwen2VLProcessor::new(
        vision_config.patch_size,
        vision_config.temporal_patch_size,
        vision_config.spatial_merge_size,
    );
    let Some(config) = std::fs::read_to_string(model_dir.join("preprocessor_config.json"))
        .ok()
        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
    else {
        return processor;
    };
    let pixels = |key: &str| -> Option<usize> {
        config
            .get(key)
            .or_else(|| config.get("size")?.get(key))
            .and_then(Value::as_u64)
            .filter(|&v| v > 0)
            .map(|v| v as usize)
    };
    if let Some(max_pixels) = pixels("max_pixels") {
        processor.max_pixels = max_pixels;
    }
    if let Some(min_pixels) = pixels("min_pixels") {
        processor.min_pixels = min_pixels.min(processor.max_pixels);
    }
    processor
}

#[cfg(test)]
#[path = "colqwen2_5_tests.rs"]
mod colqwen2_5_tests;
