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

//! Llama-Nemotron-VL-Embed: a SigLIP-400M tower, InternVL's pixel-shuffle
//! `mlp1` connector and a bidirectional Llama 3.2 1B, mean pooled.
//!
//! `nvidia/llama-nemotron-embed-vl-1b-v2` declares `model_type:
//! llama_nemotron_vl` with `architectures: ["LlamaNemotronVLModel"]`, and its
//! `llm_config` declares `LlamaBidirectionalModel` (`model_type:
//! llama_bidirec`), the Llama decoder with `is_causal = False` on every
//! attention module. mlxcel composes the three pieces from parts it already
//! runs:
//!
//! - [`crate::vision::encoders::siglip::SigLipVisionModel`] at
//!   `vision_model.vision_model`, whose `last_hidden_state` (the
//!   `post_layernorm` output, `select_layer: -1`) is what the reference reads.
//!   The attention-pooling `head.*` is dropped at load: an embedder never
//!   touches it.
//! - [`crate::vision::internvl::InternVLConnector`] at `mlp1`, which is
//!   `pixel_shuffle(0.5)` (`ps_version: v2`, both permutes) followed by
//!   `LayerNorm(4608) -> Linear(4608 -> 2048) -> GELU -> Linear(2048 ->
//!   2048)`. 1024 SigLIP patches per `512x512` tile become 256 language
//!   tokens.
//! - [`crate::models::llama3::Llama3Model`] for the text side, driven layer by
//!   layer with a bidirectional padding mask so no position attends causally,
//!   and stopped at the final norm (`lm_head` is tied and never applied).
//!
//! Prompt construction follows `processing_llama_nemotron_vl.py`. The task
//! prefix is caller-side for text (`query: ` before a question, `passage: `
//! before a document), which matches the text-only sibling. An image item
//! carries no caller text, so the family emits the document form itself:
//! `passage: <img>{<IMG_CONTEXT> * 256 * tiles}</img> `. The engine tokenizes
//! before the family sees the image, so `format_text` emits one
//! `<IMG_CONTEXT>` and [`LlamaNemotronVLEmbeddingModel::embed`] expands it to
//! the real tile count.

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlxcel_core::utils::create_bidirectional_padding_mask;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::Value;

use crate::embeddings::limits::{config_normalize_flag, read_json};
use crate::embeddings::loader::{load_embedding_weights, quantization_params};
use crate::embeddings::model::{EmbeddingBatch, EmbeddingModel, EmbeddingOutput, ImageInput};
use crate::embeddings::pooling::{PoolingMode, pool, resolve_pooling_mode};
use crate::models::embedding_sanitize::sanitize_decoder_embedding_weights;
use crate::models::llama_nemotron_vl_tiling::NemotronTiling;
use crate::models::llama3::{Llama3Model, ModelArgs};
use crate::vision::config::VisionConfig;
use crate::vision::encoders::VisionEncoder;
use crate::vision::encoders::siglip::SigLipVisionModel;
use crate::vision::internvl::InternVLConnector;
use crate::vision::merge;

/// Opening marker of the image block (`<img>`, id 128256).
pub const IMG_START_TOKEN: &str = "<img>";
/// Closing marker of the image block (`</img>`, id 128257).
pub const IMG_END_TOKEN: &str = "</img>";
/// Placeholder each projected visual token occupies (`<IMG_CONTEXT>`).
pub const IMG_CONTEXT_TOKEN: &str = "<IMG_CONTEXT>";
/// `processor_config.json` `passage_prefix` fallback.
pub const DEFAULT_PASSAGE_PREFIX: &str = "passage:";
/// Visual tokens one `512x512` tile contributes after the pixel shuffle.
pub const DEFAULT_NUM_IMAGE_TOKEN: usize = 256;
/// `nn.LayerNorm` default epsilon, which is what `mlp1.0` was trained with.
const MLP1_LAYER_NORM_EPS: f32 = 1e-5;

/// Llama-Nemotron-VL-Embed: SigLIP + `mlp1` + bidirectional Llama, mean pooled.
pub struct LlamaNemotronVLEmbeddingModel {
    vision: SigLipVisionModel,
    connector: InternVLConnector,
    text: Llama3Model,
    tiling: NemotronTiling,
    img_context_token_id: i32,
    num_image_token: usize,
    passage_prefix: String,
    pooling: PoolingMode,
    normalize: bool,
    embedding_dim: usize,
}

/// Normalize the checkpoint's weight keys onto the layouts the reused
/// constructors expect.
///
/// Three edits, in order:
///
/// 1. `language_model.*` loses its prefix so
///    [`sanitize_decoder_embedding_weights`] can re-add the `model.` one
///    [`Llama3Model::from_weights`] reads. Going through the shared helper
///    also drops a generation head and folds any `Dense` module folder, which
///    this checkpoint has none of but a re-export might.
/// 2. `vision_model.vision_model.head.*` (the SigLIP attention-pooling head)
///    is dropped: `extract_feature` reads `last_hidden_state`, never the head.
/// 3. Buffers that are not parameters (`rotary_emb.inv_freq`, `position_ids`)
///    are dropped so they cannot be mistaken for tensors by a later pass.
pub(crate) fn sanitize_nemotron_vl_weights(weights: &mut WeightMap) {
    let renames: Vec<(String, String)> = weights
        .keys()
        .filter_map(|key| {
            key.strip_prefix("language_model.")
                .map(|rest| (key.clone(), rest.to_string()))
        })
        .collect();
    for (from, to) in renames {
        if let Some(tensor) = weights.remove(&from) {
            weights.insert(to, tensor);
        }
    }

    weights.retain(|key, _| {
        !key.starts_with("vision_model.vision_model.head.")
            && !key.contains("rotary_emb.inv_freq")
            && !key.ends_with("position_ids")
    });

    sanitize_decoder_embedding_weights(weights);
}

/// Read the tiling and prompt settings from `processor_config.json`, falling
/// back to the published values when a key or the whole file is absent.
fn read_processor_config(model_dir: &Path) -> (NemotronTiling, usize, String) {
    let mut tiling = NemotronTiling::default();
    let mut num_image_token = DEFAULT_NUM_IMAGE_TOKEN;
    let mut passage_prefix = DEFAULT_PASSAGE_PREFIX.to_string();

    let Some(config) = read_json(&model_dir.join("processor_config.json")) else {
        return (tiling, num_image_token, passage_prefix);
    };
    let positive = |key: &str| {
        config
            .get(key)
            .and_then(Value::as_u64)
            .filter(|&v| v > 0)
            .map(|v| v as usize)
    };
    if let Some(size) = positive("image_size") {
        tiling.image_size = size;
    }
    if let Some(tiles) = positive("max_input_tiles") {
        tiling.max_tiles = tiles.max(tiling.min_tiles);
    }
    if let Some(use_thumbnail) = config.get("use_thumbnail").and_then(Value::as_bool) {
        tiling.use_thumbnail = use_thumbnail;
    }
    if let Some(tokens) = positive("num_image_token") {
        num_image_token = tokens;
    }
    if let Some(prefix) = config.get("passage_prefix").and_then(Value::as_str) {
        passage_prefix = prefix.to_string();
    }
    (tiling, num_image_token, passage_prefix)
}

impl LlamaNemotronVLEmbeddingModel {
    /// Load a Llama-Nemotron-VL-Embed checkpoint from `model_dir`.
    pub fn load(model_dir: &Path, config: &Value) -> Result<Self> {
        let llm_config = config
            .get("llm_config")
            .or_else(|| config.get("text_config"));
        let llm_config = llm_config.ok_or_else(|| {
            anyhow::anyhow!("Llama-Nemotron-VL-Embed: config.json has no llm_config block")
        })?;
        let mut args: ModelArgs = serde_json::from_value(llm_config.clone())
            .context("failed to parse llm_config as a Llama config")?;
        args.set_checkpoint_label(model_dir);
        // The embedder stops at the final norm, so no head is ever applied.
        // Forcing the tied flag keeps the constructor from looking for an
        // `lm_head` this checkpoint does not ship and this path never reads.
        args.tie_word_embeddings = true;

        let vision_config: VisionConfig = config
            .get("vision_config")
            .ok_or_else(|| {
                anyhow::anyhow!("Llama-Nemotron-VL-Embed: config.json has no vision_config block")
            })
            .and_then(|value| {
                serde_json::from_value(value.clone())
                    .context("failed to parse vision_config as a SigLIP vision config")
            })?;

        let downsample_ratio = config
            .get("downsample_ratio")
            .and_then(Value::as_f64)
            .filter(|&r| r > 0.0)
            .unwrap_or(0.5) as f32;
        let img_context_token_id = config
            .get("img_context_token_id")
            .and_then(Value::as_i64)
            .and_then(|id| i32::try_from(id).ok())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Llama-Nemotron-VL-Embed: config.json has no usable img_context_token_id"
                )
            })?;

        let mut weights = load_embedding_weights(model_dir, config)?;
        sanitize_nemotron_vl_weights(&mut weights);

        let (group_size, bits) = quantization_params(config)
            .map(|q| (q.group_size, q.bits))
            .unwrap_or((0, 0));

        let vision =
            SigLipVisionModel::from_weights(&weights, &vision_config, "vision_model.vision_model")
                .map_err(|e| anyhow::anyhow!("Llama-Nemotron-VL-Embed vision tower: {e}"))?;
        let connector = InternVLConnector::from_weights(
            &weights,
            "mlp1",
            MLP1_LAYER_NORM_EPS,
            downsample_ratio,
            group_size,
            bits,
        )
        .map_err(|e| anyhow::anyhow!("Llama-Nemotron-VL-Embed mlp1 connector: {e}"))?;
        let text = Llama3Model::from_weights(&weights, &args)
            .map_err(|e| anyhow::anyhow!("Llama-Nemotron-VL-Embed text backbone: {e}"))?;

        let (tiling, num_image_token, passage_prefix) = read_processor_config(model_dir);

        Ok(Self {
            vision,
            connector,
            text,
            tiling,
            img_context_token_id,
            num_image_token,
            passage_prefix,
            pooling: resolve_pooling_mode(model_dir, PoolingMode::Mean)?,
            normalize: config_normalize_flag(config),
            embedding_dim: args.hidden_size,
        })
    }

    /// Project one batch of tiles into `[tiles, num_image_token, hidden]`
    /// language-space vectors.
    fn extract_features(&self, pixel_values: &MlxArray, dtype: i32) -> UniquePtr<MlxArray> {
        let pixels = mlxcel_core::astype(pixel_values, dtype);
        let hidden = self.vision.forward(&pixels).hidden_states;
        self.connector.forward(&hidden)
    }

    /// Bidirectional Llama over `[B, L, hidden]` embeddings, stopped at the
    /// final norm.
    fn forward_text(
        &self,
        embeddings: &MlxArray,
        attention_mask: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let mask = create_bidirectional_padding_mask(attention_mask);
        let mut caches = self.text.make_caches();
        let mut hidden = mlxcel_core::copy(embeddings);
        for (index, layer) in self.text.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &mut caches[index], Some(&mask));
        }
        self.text.norm.forward(&hidden)
    }

    /// Visual tokens each image contributes: `num_image_token` per tile,
    /// counting the thumbnail tile a split image gains.
    fn visual_token_counts(&self, images: &[ImageInput]) -> Vec<usize> {
        images
            .iter()
            .map(|input| self.tiling.tiles(&input.image).len() * self.num_image_token)
            .collect()
    }

    /// Run one image row. The engine has already expanded the placeholder
    /// through [`EmbeddingModel::expand_image_tokens`], so the batch's ids and
    /// mask are the ones the forward pass consumes.
    fn embed_image_row(
        &self,
        batch: &EmbeddingBatch,
        images: &[ImageInput],
    ) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)> {
        let shape = mlxcel_core::array_shape(batch.input_ids);
        if shape[0] != 1 {
            bail!(
                "Llama-Nemotron-VL-Embed embeds images one at a time, got a batch of {}",
                shape[0]
            );
        }
        let decoded: Vec<image::DynamicImage> =
            images.iter().map(|input| input.image.clone()).collect();
        let (pixel_values, _tiles_per_image) = self.tiling.preprocess(&decoded);

        let embeddings = self.text.embed_tokens.forward(batch.input_ids);
        let features = self.extract_features(&pixel_values, mlxcel_core::array_dtype(&embeddings));
        let merged = merge::merge_llava(
            self.img_context_token_id,
            &features,
            &embeddings,
            batch.input_ids,
        );
        let hidden = self.forward_text(&merged.inputs_embeds, batch.attention_mask);
        Ok((hidden, mlxcel_core::copy(batch.attention_mask)))
    }
}

impl EmbeddingModel for LlamaNemotronVLEmbeddingModel {
    fn embed(&self, batch: &EmbeddingBatch) -> Result<EmbeddingOutput> {
        let images = batch.images.unwrap_or(&[]);
        let (hidden, mask) = if images.is_empty() {
            let embeddings = self.text.embed_tokens.forward(batch.input_ids);
            (
                self.forward_text(&embeddings, batch.attention_mask),
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
        PoolingMode::Mean
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

    /// Text inputs pass through unchanged (the `query: ` / `passage: ` prefix
    /// is the caller's, as for the text-only sibling).
    ///
    /// An empty `text` is the engine's image call
    /// ([`crate::embeddings::EmbeddingEngine::embed_image`]); every text path
    /// rejects an empty string before it reaches here, so the empty string is
    /// an unambiguous "this row carries an image" signal. The one
    /// `<IMG_CONTEXT>` emitted here is expanded to `num_image_token * tiles`
    /// by [`EmbeddingModel::expand_image_tokens`] below, which is where the
    /// tile count is known.
    fn format_text(&self, text: &str, _instruction: Option<&str>) -> String {
        if text.is_empty() {
            format!(
                "{} {IMG_START_TOKEN}{IMG_CONTEXT_TOKEN}{IMG_END_TOKEN} ",
                self.passage_prefix
            )
        } else {
            text.to_string()
        }
    }

    /// Expand the one `<IMG_CONTEXT>` the document prompt carries into
    /// `num_image_token * tiles` copies.
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
        let (expanded, _) = crate::models::qwen3_vl_embedding::expand_image_placeholders(
            &signed,
            &mask,
            self.img_context_token_id,
            &counts,
        )
        .context("Llama-Nemotron-VL-Embed image block expansion")?;
        Ok(expanded.into_iter().map(|id| id as u32).collect())
    }
}

#[cfg(test)]
#[path = "llama_nemotron_vl_embedding_tests.rs"]
mod llama_nemotron_vl_embedding_tests;
