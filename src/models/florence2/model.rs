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

//! Florence-2 vision-language fusion: the assembled model that joins the
//! DaViT tower to the BART seq2seq engine.
//!
//! Florence-2 does *not* scatter image features into placeholder slots the
//! way Llava-style models do. It pools the vision tower output into a short
//! feature sequence, projects it into the text embedding space, and
//! **concatenates** it in front of the task-prompt embeddings. The result is
//! the encoder input; the decoder then cross-attends to it. So the encoder
//! sees `[image features | prompt embeddings]` and there is no image token in
//! the prompt at all.
//!
//! Pipeline, in order:
//!
//! 1. [`Florence2Model::encode_image`] runs the tower, adds the learned 2-D
//!    grid position embedding and the temporal embedding, pools according to
//!    `image_feature_source`, then applies `image_projection` (1024 -> 768)
//!    and `image_proj_norm`.
//! 2. [`Florence2Model::merge_input_ids_with_image_features`] concatenates
//!    those features with the embedded prompt and builds the joint attention
//!    mask.
//! 3. [`Florence2Model::encode`] feeds the fused sequence to the BART
//!    encoder, and [`Florence2Model::generate_greedy`] runs the decode loop
//!    against it.
//!
//! Reference:
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/florence2.py

use std::path::Path;

use anyhow::{Result, anyhow};
use serde_json::Value;

use mlxcel_core::layers::LayerNorm;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::checkpoint::{FLORENCE2_TEXT_PREFIX, Florence2Config, sanitize};
use super::fusion::{
    LearnedPositionEmbedding2D, PositionalEmbeddingCosine1D, additive_attention_mask,
};
use super::layers::layer_norm;
use super::{
    FLORENCE2_VISION_PREFIX, Florence2DaViT, Florence2SeqCache, Florence2TextModel,
    argmax_last_position,
};

/// Florence-2 processes one still frame per request. Video support upstream
/// stacks frames on this axis; the pooling below is written in terms of it so
/// the shapes stay readable, and the temporal embedding contributes row 0
/// rather than nothing.
const NUM_FRAMES: i32 = 1;

/// The assembled Florence-2 vision-language model.
///
/// Holds MLX weight handles, so the owning provider serializes access.
pub struct Florence2Model {
    config: Florence2Config,
    vision_tower: Florence2DaViT,
    text: Florence2TextModel,
    /// `[dim_embed[-1], d_model]`, applied as `features @ image_projection`.
    image_projection: UniquePtr<MlxArray>,
    image_proj_norm: LayerNorm,
    image_pos_embed: LearnedPositionEmbedding2D,
    visual_temporal_embed: PositionalEmbeddingCosine1D,
    /// Activation dtype of the text stack. Image features are cast to it
    /// before the concatenation so a mixed-precision checkpoint (bf16 vision
    /// tower alongside a quantized text stack) does not silently promote the
    /// fused sequence to f32.
    dtype: i32,
    /// Activation dtype of the vision tower, taken from its first conv
    /// weight. Pixel input is cast to it for the same reason.
    vision_dtype: i32,
}

impl Florence2Model {
    /// Load the whole model from a checkpoint directory (`config.json` +
    /// safetensors). The checkpoint is read once and both halves are built
    /// from the same weight map.
    pub fn load(model_path: &Path) -> Result<Self> {
        let config_path = model_path.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| anyhow!("Failed to read {config_path:?}: {e}"))?;
        let config_str = crate::models::sanitize_config_json(&config_str);
        let config: Value = serde_json::from_str(&config_str)
            .map_err(|e| anyhow!("Failed to parse Florence-2 config: {e}"))?;
        let parsed = Florence2Config::from_model_config(&config)?;

        let weights = mlxcel_core::weights::load_weights_from_dir(model_path)
            .map_err(|e| anyhow!("Failed to load Florence-2 weights: {e}"))?;
        let mut weights = sanitize(weights);

        // Precision policy, matching `load_vlm_weights`: promote bf16 to f16
        // on Apple Silicon for the non-quantized case, and leave a quantized
        // checkpoint alone so its bf16 scales/biases stay dtype-consistent
        // with the bf16 activation path the quantized kernels expect. A
        // Florence-2 quant is mixed by construction (the DaViT tower stays
        // dense), which is exactly why the fused path pins its activation
        // dtype from the text stack rather than assuming one.
        if !crate::models::sanitize::config_has_quantization_metadata(&config) {
            let _ = crate::models::convert_bf16_weights(&mut weights);
        }

        Self::from_weights(&weights, parsed)
            .map_err(|e| anyhow!("Failed to build Florence-2 model: {e}"))
    }

    /// Build from an already-loaded and already-sanitized [`WeightMap`].
    pub fn from_weights(weights: &WeightMap, config: Florence2Config) -> Result<Self, String> {
        let vision_tower =
            Florence2DaViT::from_weights(weights, &config.vision, FLORENCE2_VISION_PREFIX)?;
        let text =
            Florence2TextModel::from_weights(weights, config.text.clone(), FLORENCE2_TEXT_PREFIX)?;

        let image_dim = config.vision.output_dim();
        let d_model = config.text.d_model;

        // `image_projection` is a bare tensor rather than a `Linear`: upstream
        // registers it as a raw parameter and applies it as a right-hand
        // matmul, so it is stored `[in, out]` with no bias and no transpose.
        let image_projection = weights
            .get("image_projection")
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| "Florence-2 weight not found: image_projection".to_string())?;
        let projection_shape = mlxcel_core::array_shape(&image_projection);
        if projection_shape != vec![image_dim, d_model] {
            return Err(format!(
                "Florence-2 image_projection: expected shape [{image_dim}, {d_model}], got {projection_shape:?}"
            ));
        }
        let image_proj_norm = layer_norm(weights, "image_proj_norm")?;

        let pos_spec = config
            .vision
            .image_pos_embed
            .as_ref()
            .ok_or_else(|| "Florence-2 config missing image_pos_embed".to_string())?;
        let pos_kind = pos_spec.get("type").and_then(Value::as_str).unwrap_or("");
        if pos_kind != "learned_abs_2d" {
            return Err(format!(
                "Florence-2 image_pos_embed type {pos_kind:?} not supported (expected learned_abs_2d)"
            ));
        }
        let image_pos_embed =
            LearnedPositionEmbedding2D::from_weights(weights, "image_pos_embed", image_dim)?;

        let temporal_spec = config
            .vision
            .visual_temporal_embedding
            .as_ref()
            .ok_or_else(|| "Florence-2 config missing visual_temporal_embedding".to_string())?;
        let temporal_kind = temporal_spec
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("");
        if temporal_kind != "COSINE" {
            return Err(format!(
                "Florence-2 visual_temporal_embedding type {temporal_kind:?} not supported (expected COSINE)"
            ));
        }
        let max_temporal = temporal_spec
            .get("max_temporal_embeddings")
            .and_then(Value::as_i64)
            .unwrap_or(100) as i32;
        let visual_temporal_embed = PositionalEmbeddingCosine1D::from_weights(
            weights,
            "visual_temporal_embed",
            image_dim,
            max_temporal,
        )?;

        // The pooling recipe is order-sensitive (it decides the layout of the
        // concatenated feature sequence) and upstream's dataclass default is
        // in the opposite order from every real checkpoint, so refuse to
        // guess it.
        if config.vision.image_feature_source.is_empty() {
            return Err("Florence-2 config missing image_feature_source".to_string());
        }

        let dtype = text.dtype();
        let vision_dtype = weights
            .get(&format!("{FLORENCE2_VISION_PREFIX}convs.0.proj.weight"))
            .map(|w| mlxcel_core::array_dtype(w))
            .unwrap_or(dtype);

        Ok(Self {
            config,
            vision_tower,
            text,
            image_projection,
            image_proj_norm,
            image_pos_embed,
            visual_temporal_embed,
            dtype,
            vision_dtype,
        })
    }

    /// Parsed configuration for both halves.
    pub fn config(&self) -> &Florence2Config {
        &self.config
    }

    /// The BART text core, for callers that need the decode primitives
    /// directly.
    pub fn text_model(&self) -> &Florence2TextModel {
        &self.text
    }

    /// The DaViT tower.
    pub fn vision_tower(&self) -> &Florence2DaViT {
        &self.vision_tower
    }

    /// Activation dtype of the fused path (that of the text stack).
    pub fn dtype(&self) -> i32 {
        self.dtype
    }

    /// Encode NCHW `pixel_values` into projected image features
    /// `[batch, tokens, d_model]` ready to concatenate with prompt
    /// embeddings.
    pub fn encode_image(&self, pixel_values: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let pixel_values = if mlxcel_core::array_dtype(pixel_values) == self.vision_dtype {
            mlxcel_core::copy(pixel_values)
        } else {
            mlxcel_core::astype(pixel_values, self.vision_dtype)
        };
        let features = self.vision_tower.forward(&pixel_values);
        self.encode_image_features(&features)
    }

    /// The fusion half of [`Self::encode_image`], starting from raw backbone
    /// tokens `[batch, H*W, dim_embed[-1]]`. Split out so a caller holding
    /// cached tower output (or a test pinning the tower separately) can drive
    /// the projection without re-running the tower.
    pub fn encode_image_features(
        &self,
        features: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let shape = mlxcel_core::array_shape(features);
        if shape.len() != 3 {
            return Err(format!(
                "Florence-2 image features must be [batch, tokens, dim], got {shape:?}"
            ));
        }
        let (batch, num_tokens, dim) = (shape[0], shape[1], shape[2]);
        let image_dim = self.config.vision.output_dim();
        if dim != image_dim {
            return Err(format!(
                "Florence-2 image features have width {dim}, expected dim_embed[-1] {image_dim}"
            ));
        }
        let side = (num_tokens as f64).sqrt().round() as i32;
        if side * side != num_tokens {
            return Err(format!(
                "Florence-2 image features have {num_tokens} tokens, which is not a square feature map"
            ));
        }

        // Learned 2-D grid position embedding over [B*T, H, W, D].
        let x = mlxcel_core::reshape(features, &[batch * NUM_FRAMES, side, side, dim]);
        let pos = self.image_pos_embed.forward(side, side)?;
        let x = mlxcel_core::add(&x, &pos);

        // Temporal embedding over the frame axis: [B, T, H*W, D] + [1, T, 1, D].
        let x = mlxcel_core::reshape(&x, &[batch, NUM_FRAMES, num_tokens, dim]);
        let temporal = self.visual_temporal_embed.forward(NUM_FRAMES)?;
        let temporal = mlxcel_core::reshape(&temporal, &[1, NUM_FRAMES, 1, dim]);
        let x = mlxcel_core::add(&x, &temporal);

        // Pool per `image_feature_source` and concatenate along the token
        // axis. For the base-ft recipe this is one spatially averaged token
        // followed by the 576 temporally averaged grid tokens: 577 in total.
        let mut fused: Option<UniquePtr<MlxArray>> = None;
        for source in &self.config.vision.image_feature_source {
            let pooled = match source.as_str() {
                "spatial_avg_pool" => mlxcel_core::mean_axis(&x, 2, false),
                "temporal_avg_pool" => mlxcel_core::mean_axis(&x, 1, false),
                "last_frame" => {
                    let last = mlxcel_core::slice(
                        &x,
                        &[0, NUM_FRAMES - 1, 0, 0],
                        &[batch, NUM_FRAMES, num_tokens, dim],
                    );
                    mlxcel_core::reshape(&last, &[batch, num_tokens, dim])
                }
                other => {
                    return Err(format!(
                        "Florence-2 image_feature_source {other:?} is not supported"
                    ));
                }
            };
            fused = Some(match fused {
                Some(previous) => mlxcel_core::concatenate(&previous, &pooled, 1),
                None => pooled,
            });
        }
        // `image_feature_source` was checked non-empty at load time.
        let fused = fused.ok_or_else(|| "Florence-2 image_feature_source is empty".to_string())?;

        let projected = mlxcel_core::matmul(&fused, &self.image_projection);
        let projected = self.image_proj_norm.forward(&projected);
        Ok(if mlxcel_core::array_dtype(&projected) == self.dtype {
            projected
        } else {
            mlxcel_core::astype(&projected, self.dtype)
        })
    }

    /// Embed the task prompt, dropping any image placeholder id.
    ///
    /// Returns `None` for an empty (or entirely placeholder) prompt, which is
    /// the image-only case: the encoder input is then the image features
    /// alone.
    pub fn embed_prompt(&self, prompt_ids: &[i32]) -> Option<UniquePtr<MlxArray>> {
        let filtered: Vec<i32> = prompt_ids
            .iter()
            .copied()
            .filter(|id| *id != self.config.image_token_id)
            .collect();
        if filtered.is_empty() {
            return None;
        }
        let ids = mlxcel_core::from_slice_i32(&filtered, &[1, filtered.len() as i32]);
        Some(self.text.embed_tokens(&ids))
    }

    /// Concatenate image features in front of the prompt embeddings and build
    /// the joint attention mask.
    ///
    /// Returns `(inputs_embeds [batch, image + prompt, d_model],
    /// attention_mask [batch, image + prompt])`. The mask is all ones here
    /// because both segments are real content; it exists so a padded batch
    /// can carry zeros through the same path, and
    /// [`Self::encode_with_image_features`] converts it to the additive form
    /// the encoder's attention consumes.
    pub fn merge_input_ids_with_image_features(
        &self,
        image_features: &MlxArray,
        inputs_embeds: Option<&MlxArray>,
    ) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>), String> {
        let image_shape = mlxcel_core::array_shape(image_features);
        if image_shape.len() != 3 {
            return Err(format!(
                "Florence-2 image features must be [batch, tokens, dim], got {image_shape:?}"
            ));
        }
        let (batch, image_len, dim) = (image_shape[0], image_shape[1], image_shape[2]);

        let Some(prompt_embeds) = inputs_embeds else {
            return Ok((
                mlxcel_core::copy(image_features),
                mlxcel_core::ones(&[batch, image_len], self.dtype),
            ));
        };

        let prompt_shape = mlxcel_core::array_shape(prompt_embeds);
        if prompt_shape.len() != 3 || prompt_shape[0] != batch || prompt_shape[2] != dim {
            return Err(format!(
                "Florence-2 prompt embeddings {prompt_shape:?} are not compatible with image features {image_shape:?}"
            ));
        }

        let merged = mlxcel_core::concatenate(image_features, prompt_embeds, 1);
        let mask = mlxcel_core::ones(&[batch, image_len + prompt_shape[1]], self.dtype);
        Ok((merged, mask))
    }

    /// Full encoder pass: tower -> fusion -> BART encoder. Returns encoder
    /// hidden states `[batch, image + prompt, d_model]`.
    pub fn encode(
        &self,
        pixel_values: &MlxArray,
        prompt_ids: &[i32],
    ) -> Result<UniquePtr<MlxArray>, String> {
        let image_features = self.encode_image(pixel_values)?;
        self.encode_with_image_features(&image_features, prompt_ids)
    }

    /// [`Self::encode`] starting from already-projected image features.
    pub fn encode_with_image_features(
        &self,
        image_features: &MlxArray,
        prompt_ids: &[i32],
    ) -> Result<UniquePtr<MlxArray>, String> {
        let prompt_embeds = self.embed_prompt(prompt_ids);
        let (inputs_embeds, attention_mask) =
            self.merge_input_ids_with_image_features(image_features, prompt_embeds.as_deref())?;

        let seq = mlxcel_core::array_shape(&inputs_embeds)[1];
        let max_positions = self.config.text.max_position_embeddings;
        if seq > max_positions {
            return Err(format!(
                "Florence-2 fused encoder input length {seq} exceeds max_position_embeddings {max_positions}"
            ));
        }

        let additive = additive_attention_mask(&attention_mask, self.dtype);
        Ok(self
            .text
            .encode_embeds_with_mask(&inputs_embeds, Some(&additive)))
    }

    /// Fresh decode-loop cache sized to the decoder depth.
    pub fn make_cache(&self) -> Florence2SeqCache {
        self.text.make_cache()
    }

    /// Run the decoder one step (or several) against encoder hidden states.
    /// Returns logits `[batch, seq, vocab_size]`.
    pub fn decode(
        &self,
        decoder_input_ids: &MlxArray,
        encoder_hidden_states: &MlxArray,
        cache: &mut Florence2SeqCache,
    ) -> UniquePtr<MlxArray> {
        self.text
            .decode(decoder_input_ids, encoder_hidden_states, cache)
    }

    /// Greedy caption / task generation: fuse image and prompt, then decode
    /// from `decoder_start_token_id` until EOS or `max_new_tokens`. Returns
    /// the generated ids (EOS excluded).
    pub fn generate_greedy(
        &self,
        pixel_values: &MlxArray,
        prompt_ids: &[i32],
        max_new_tokens: usize,
    ) -> Result<Vec<i32>> {
        let encoder_hidden = self
            .encode(pixel_values, prompt_ids)
            .map_err(|e| anyhow!("{e}"))?;

        let text_config = &self.config.text;
        let mut cache = self.make_cache();
        let mut generated = Vec::new();
        let mut next = text_config.decoder_start_token_id;
        for _ in 0..max_new_tokens {
            if cache.offset() >= text_config.max_position_embeddings {
                break;
            }
            let token = mlxcel_core::from_slice_i32(&[next], &[1, 1]);
            let logits = self.decode(&token, &encoder_hidden, &mut cache);
            next = argmax_last_position(&logits)?;
            if next == text_config.eos_token_id {
                break;
            }
            generated.push(next);
        }
        Ok(generated)
    }
}

#[cfg(test)]
#[path = "florence2_fusion_tests.rs"]
mod florence2_fusion_tests;
