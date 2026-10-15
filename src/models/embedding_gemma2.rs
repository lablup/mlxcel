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

//! EmbeddingGemma 2 (`google/embeddinggemma-2`, `mlx-community/embeddinggemma-2-*`)
//! multimodal bidirectional encoder built on the Gemma 4 architecture.
//!
//! Architecture summary:
//! - 24-layer bidirectional Gemma 4 text encoder (`hidden_size: 512`,
//!   `intermediate_size: 2048`, `vocab_size: 262144`)
//! - Hybrid local/global bidirectional attention: sliding-window layers (`|q - k| <= 512`,
//!   `head_dim: 256`, `num_key_value_heads: 2`, `rope_theta: 10000`) and full-attention
//!   layers `[5, 11, 17, 23]` (`head_dim: 512`, `num_key_value_heads: 1`, `rope_theta: 1000000`)
//! - `q_norm` (`RMSNorm`), `k_norm` (`RMSNorm`), `v_norm` (`RMSNormNoScale`), SDPA `scale = 1.0`
//! - Projection-only Per-Layer Embeddings (`language_model.ple`) + per-layer `ple_block` and `layer_scalar`
//! - Final `RMSNorm(512)` followed by in-model `language_model.embedding_projection` (`512 -> 768`, no bias)
//! - Masked mean pooling (`PoolingMode::Mean`) and L2 normalization (Matryoshka `768 / 512 / 256 / 128`)

#[path = "embedding_gemma2_config.rs"]
mod config;
#[path = "embedding_gemma2_layers.rs"]
mod layers;

pub use config::{
    EmbeddingGemma2Config, EmbeddingGemma2LayerConfig, quant_params_for_prefix,
    sanitize_embedding_gemma2_weights,
};
pub use layers::{
    EmbeddingGemma2Attention, EmbeddingGemma2Layer, EmbeddingGemma2Mlp, EmbeddingGemma2PleBlock,
};

use std::path::Path;

use anyhow::{Context, Result};
use layers::rms_norm_from_weights;
use mlxcel_core::layers::{RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::utils::{create_bidirectional_padding_mask, create_bidirectional_window_mask};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::Value;

use crate::audio::{AudioConfig, AudioEncoder};
use crate::embeddings::limits::config_normalize_flag;
use crate::embeddings::loader::load_embedding_weights;
use crate::embeddings::model::{EmbeddingBatch, EmbeddingModel, EmbeddingOutput, ImageInput};
use crate::embeddings::pooling::{PoolingMode, pool, resolve_pooling_mode};
use crate::vision::encoders::gemma4::{Gemma4VisionConfig, Gemma4VisionModel};
use crate::vision::gemma4_multimodal_embedder::{Gemma4MultimodalEmbedder, masked_scatter};
use crate::vision::merge;
use crate::vision::processors::gemma4::Gemma4Processor;

/// Optional vision tower + projector bundled with an EmbeddingGemma 2 checkpoint.
pub struct EmbeddingGemma2Vision {
    pub tower: Gemma4VisionModel,
    pub embedder: Gemma4MultimodalEmbedder,
    pub processor: Gemma4Processor,
}

/// Optional audio tower + projector bundled with an EmbeddingGemma 2 checkpoint.
pub struct EmbeddingGemma2Audio {
    pub tower: AudioEncoder,
    pub embedder: Gemma4MultimodalEmbedder,
}

/// EmbeddingGemma 2 multimodal embedding model.
pub struct EmbeddingGemma2Model {
    pub config: EmbeddingGemma2Config,
    pub embed_tokens: UnifiedEmbedding,
    pub per_layer_model_projection: Option<UnifiedLinear>,
    pub per_layer_projection_norm: Option<RMSNorm>,
    pub layers: Vec<EmbeddingGemma2Layer>,
    pub norm: RMSNorm,
    pub embedding_projection: UnifiedLinear,
    pub vision: Option<EmbeddingGemma2Vision>,
    pub audio: Option<EmbeddingGemma2Audio>,
    pooling: PoolingMode,
    normalize: bool,
}

impl EmbeddingGemma2Model {
    /// Load an `EmbeddingGemma2Model` from disk.
    pub fn load(model_dir: &Path, config: &Value) -> Result<Self> {
        let pooling = resolve_pooling_mode(model_dir, PoolingMode::Mean)?;
        let normalize = config_normalize_flag(config);
        let mut weights = load_embedding_weights(model_dir, config)?;
        Self::from_weights(&mut weights, config, pooling, normalize)
    }

    /// Construct an `EmbeddingGemma2Model` from an already-loaded `WeightMap`.
    pub fn from_weights(
        weights: &mut WeightMap,
        raw_config: &Value,
        pooling: PoolingMode,
        normalize: bool,
    ) -> Result<Self> {
        sanitize_embedding_gemma2_weights(weights);
        let cfg = EmbeddingGemma2Config::from_json(raw_config, Some(weights));

        let (emb_gs, emb_bits) = quant_params_for_prefix(raw_config, "language_model.embed_tokens");
        let embed_tokens = UnifiedEmbedding::from_weights(
            weights,
            "language_model.embed_tokens",
            emb_gs,
            emb_bits,
        )
        .map_err(|e| anyhow::anyhow!("Failed to load language_model.embed_tokens: {e}"))?;

        let ple_proj_prefix = "language_model.ple.per_layer_model_projection";
        let (ple_gs, ple_bits) = quant_params_for_prefix(raw_config, ple_proj_prefix);
        let per_layer_model_projection =
            if weights.contains_key(&format!("{ple_proj_prefix}.weight")) {
                Some(
                    UnifiedLinear::from_weights(weights, ple_proj_prefix, ple_gs, ple_bits)
                        .map_err(|e| anyhow::anyhow!("Failed to load {ple_proj_prefix}: {e}"))?,
                )
            } else {
                None
            };

        let ple_norm_prefix = "language_model.ple.per_layer_projection_norm";
        let per_layer_projection_norm =
            if weights.contains_key(&format!("{ple_norm_prefix}.weight")) {
                Some(
                    rms_norm_from_weights(weights, ple_norm_prefix, cfg.rms_norm_eps)
                        .map_err(|e| anyhow::anyhow!("Failed to load {ple_norm_prefix}: {e}"))?,
                )
            } else {
                None
            };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_cfg in &cfg.layers {
            let prefix = format!("language_model.layers.{}", layer_cfg.layer_idx);
            let layer = EmbeddingGemma2Layer::from_weights(
                weights,
                &prefix,
                layer_cfg,
                cfg.rms_norm_eps,
                raw_config,
            )
            .map_err(|e| anyhow::anyhow!("Failed to load {prefix}: {e}"))?;
            layers.push(layer);
        }

        let norm = rms_norm_from_weights(weights, "language_model.norm", cfg.rms_norm_eps)
            .map_err(|e| anyhow::anyhow!("Failed to load language_model.norm: {e}"))?;

        let proj_prefix = "language_model.embedding_projection";
        let (proj_gs, proj_bits) = quant_params_for_prefix(raw_config, proj_prefix);
        let embedding_projection =
            UnifiedLinear::from_weights(weights, proj_prefix, proj_gs, proj_bits)
                .map_err(|e| anyhow::anyhow!("Failed to load {proj_prefix}: {e}"))?;

        let vision = Self::load_optional_vision(weights, raw_config)?;
        let audio = Self::load_optional_audio(weights, raw_config)?;

        Ok(Self {
            config: cfg,
            embed_tokens,
            per_layer_model_projection,
            per_layer_projection_norm,
            layers,
            norm,
            embedding_projection,
            vision,
            audio,
            pooling,
            normalize,
        })
    }

    fn load_optional_vision(
        weights: &WeightMap,
        raw_config: &Value,
    ) -> Result<Option<EmbeddingGemma2Vision>> {
        let Some(vc_val) = raw_config.get("vision_config").filter(|v| !v.is_null()) else {
            return Ok(None);
        };
        if !weights.keys().any(|k| k.starts_with("vision_tower.")) {
            return Ok(None);
        }
        let vc: Gemma4VisionConfig =
            serde_json::from_value(vc_val.clone()).context("EmbeddingGemma2 vision_config")?;
        let (v_gs, v_bits) = quant_params_for_prefix(raw_config, "vision_tower");
        let tower = Gemma4VisionModel::from_weights(weights, "vision_tower", &vc, v_gs, v_bits)
            .map_err(|e| anyhow::anyhow!("Failed to load EmbeddingGemma2 vision_tower: {e}"))?;
        let (ev_gs, ev_bits) = quant_params_for_prefix(raw_config, "embed_vision");
        let embedder = Gemma4MultimodalEmbedder::from_weights(
            weights,
            "embed_vision",
            vc.hidden_size,
            vc.rms_norm_eps(),
            ev_gs,
            ev_bits,
        )
        .map_err(|e| anyhow::anyhow!("Failed to load EmbeddingGemma2 embed_vision: {e}"))?;
        let processor = Gemma4Processor::new(
            vc.patch_size,
            vc.default_output_length,
            vc.pooling_kernel_size,
        );
        Ok(Some(EmbeddingGemma2Vision {
            tower,
            embedder,
            processor,
        }))
    }

    fn load_optional_audio(
        weights: &WeightMap,
        raw_config: &Value,
    ) -> Result<Option<EmbeddingGemma2Audio>> {
        let Some(ac_val) = raw_config.get("audio_config").filter(|v| !v.is_null()) else {
            return Ok(None);
        };
        if !weights.contains_key("audio_tower.output_proj.weight") {
            return Ok(None);
        }
        let ac: AudioConfig =
            serde_json::from_value(ac_val.clone()).context("EmbeddingGemma2 audio_config")?;
        let (a_gs, a_bits) = quant_params_for_prefix(raw_config, "audio_tower");
        let tower = AudioEncoder::from_weights(weights, "audio_tower", &ac, a_gs, a_bits)
            .map_err(|e| anyhow::anyhow!("Failed to load EmbeddingGemma2 audio_tower: {e}"))?;
        let (ea_gs, ea_bits) = quant_params_for_prefix(raw_config, "embed_audio");
        let audio_proj_dim = ac.output_proj_dims.unwrap_or(ac.hidden_size);
        let embedder = Gemma4MultimodalEmbedder::from_weights(
            weights,
            "embed_audio",
            audio_proj_dim,
            ac.rms_norm_eps,
            ea_gs,
            ea_bits,
        )
        .map_err(|e| anyhow::anyhow!("Failed to load EmbeddingGemma2 embed_audio: {e}"))?;
        Ok(Some(EmbeddingGemma2Audio { tower, embedder }))
    }

    /// Run the bidirectional encoder + PLE + final `norm` + `embedding_projection`
    /// starting from `inputs_embeds` (`[B, L, hidden_size]`).
    pub fn forward_from_embeds(
        &self,
        inputs_embeds: &MlxArray,
        attention_mask: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(inputs_embeds);
        if shape.len() < 2 || shape[0] == 0 || shape[1] == 0 {
            let b = if shape.is_empty() { 0 } else { shape[0] };
            let l = if shape.len() < 2 { 0 } else { shape[1] };
            return mlxcel_core::zeros(
                &[b, l, self.config.embedding_dim as i32],
                mlxcel_core::array_dtype(inputs_embeds),
            );
        }
        let (b, l) = (shape[0], shape[1]);

        let ple = if let (Some(proj), Some(norm)) = (
            &self.per_layer_model_projection,
            &self.per_layer_projection_norm,
        ) {
            let raw = proj.forward(inputs_embeds);
            let scaled = mlxcel_core::multiply_scalar(
                &raw,
                (self.config.hidden_size as f32).powf(-0.5),
            );
            let reshaped = mlxcel_core::reshape(
                &scaled,
                &[
                    b,
                    l,
                    self.config.num_hidden_layers as i32,
                    self.config.ple_dim,
                ],
            );
            Some(norm.forward(&reshaped))
        } else {
            None
        };

        let full_mask = create_bidirectional_padding_mask(attention_mask);
        // Only materialize the O(L^2) window mask when at least one layer uses sliding attention
        // and the sequence length exceeds sliding_window + 1. For shorter sequences, all token
        // distances are within the window so full_mask is mathematically identical.
        let has_sliding = self.layers.iter().any(|layer| layer.self_attn.is_sliding);
        let sliding_mask = if has_sliding && l > self.config.sliding_window + 1 {
            Some(create_bidirectional_window_mask(
                attention_mask,
                self.config.sliding_window + 1,
            ))
        } else {
            None
        };

        let mut h: Option<UniquePtr<MlxArray>> = None;
        for (idx, layer) in self.layers.iter().enumerate() {
            let ple_i = ple.as_ref().and_then(|p| {
                if idx >= self.config.num_hidden_layers {
                    None
                } else {
                    let idx_i32 = idx as i32;
                    let sliced = mlxcel_core::slice(
                        p,
                        &[0, 0, idx_i32, 0],
                        &[b, l, idx_i32 + 1, self.config.ple_dim],
                    );
                    Some(mlxcel_core::squeeze_axis(&sliced, 2))
                }
            });
            let mask = if layer.self_attn.is_sliding && let Some(sm) = &sliding_mask {
                sm
            } else {
                &full_mask
            };
            let current_x = match &h {
                Some(prev) => prev.as_ref().unwrap(),
                None => inputs_embeds,
            };
            h = Some(layer.forward(current_x, mask, ple_i.as_deref()));
        }

        let h_final = match h {
            Some(out) => out,
            None => mlxcel_core::copy(inputs_embeds),
        };
        let normed = self.norm.forward(&h_final);
        self.embedding_projection.forward(&normed)
    }

    /// Compute scaled token embeddings (`embed_tokens(input_ids) * sqrt(hidden_size)`),
    /// optionally scattering image or audio embeddings into `<|image|>` / `<|audio|>`
    /// placeholder positions before computing PLE and running the encoder.
    pub fn forward_multimodal(
        &self,
        input_ids: &MlxArray,
        attention_mask: &MlxArray,
        images: Option<&[ImageInput]>,
        audio_mel_and_mask: Option<(&MlxArray, &MlxArray)>,
    ) -> Result<UniquePtr<MlxArray>> {
        let raw_embeds = self.embed_tokens.forward(input_ids);
        let mut inputs_embeds = mlxcel_core::multiply_scalar(
            &raw_embeds,
            (self.config.hidden_size as f32).sqrt(),
        );

        if let (Some(vision), Some(imgs)) = (&self.vision, images)
            && !imgs.is_empty()
        {
            let decoded: Vec<_> = imgs.iter().map(|img| img.image.clone()).collect();
            let processed = vision.processor.preprocess(&decoded);
            let mut feat_list = Vec::with_capacity(processed.len());
            for item in &processed {
                if let Some(pixel_values) = item.pixel_values.as_ref() {
                    let v_out = vision.tower.forward(pixel_values, item.patch_grid);
                    let projected = vision.embedder.forward(&v_out);
                    feat_list.push(projected);
                }
            }
            if !feat_list.is_empty() {
                let merged_feats = if feat_list.len() == 1 {
                    mlxcel_core::astype(
                        feat_list[0].as_ref().unwrap(),
                        mlxcel_core::array_dtype(&inputs_embeds),
                    )
                } else {
                    let cat = crate::vision::encoders::qwen2_vl::concat_many(&feat_list, 1);
                    mlxcel_core::astype(&cat, mlxcel_core::array_dtype(&inputs_embeds))
                };
                inputs_embeds = merge::merge_llava(
                    self.config.image_token_id,
                    &merged_feats,
                    &inputs_embeds,
                    input_ids,
                )
                .inputs_embeds;
            }
        }

        if let (Some(audio), Some((audio_mel, audio_mask))) = (&self.audio, audio_mel_and_mask) {
            let (audio_enc, _) = audio
                .tower
                .forward(audio_mel, audio_mask)
                .map_err(|e| anyhow::anyhow!("EmbeddingGemma2 audio tower failed: {e}"))?;
            let audio_proj = audio.embedder.forward(&audio_enc);
            let audio_size: i32 = mlxcel_core::array_shape(&audio_proj).iter().product();
            if audio_size > 0 {
                let audio_proj =
                    mlxcel_core::astype(&audio_proj, mlxcel_core::array_dtype(&inputs_embeds));
                let tok_arr = mlxcel_core::from_slice_i32(&[self.config.audio_token_id], &[1]);
                let tok_mask = mlxcel_core::equal(input_ids, &tok_arr);
                let tok_mask_exp = mlxcel_core::expand_dims(&tok_mask, -1);
                let tok_mask_exp = mlxcel_core::broadcast_to(
                    &tok_mask_exp,
                    &mlxcel_core::array_shape(&inputs_embeds),
                );
                inputs_embeds = masked_scatter(&inputs_embeds, &tok_mask_exp, &audio_proj);
            }
        }

        Ok(self.forward_from_embeds(&inputs_embeds, attention_mask))
    }
}

impl EmbeddingModel for EmbeddingGemma2Model {
    fn embed(&self, batch: &EmbeddingBatch) -> Result<EmbeddingOutput> {
        let hidden =
            self.forward_multimodal(batch.input_ids, batch.attention_mask, batch.images, None)?;
        let embeddings = pool(&hidden, batch.attention_mask, self.pooling);
        Ok(EmbeddingOutput {
            embeddings,
            last_hidden_state: None,
        })
    }

    fn default_pooling(&self) -> PoolingMode {
        PoolingMode::Mean
    }

    fn pooling(&self) -> PoolingMode {
        self.pooling
    }

    fn normalize(&self) -> bool {
        self.normalize
    }

    fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    fn supports_images(&self) -> bool {
        self.vision.is_some()
    }

    fn format_text(&self, text: &str, _instruction: Option<&str>) -> String {
        text.to_string()
    }

    fn expand_image_tokens(&self, ids: &[u32], images: &[ImageInput]) -> Result<Vec<u32>> {
        let Some(vision) = &self.vision else {
            return Ok(ids.to_vec());
        };
        if images.is_empty() {
            return Ok(ids.to_vec());
        }
        let decoded: Vec<_> = images.iter().map(|img| img.image.clone()).collect();
        let processed = vision.processor.preprocess(&decoded);
        let counts: Vec<usize> = processed.iter().map(|p| p.num_soft_tokens).collect();
        if counts.is_empty() {
            return Ok(ids.to_vec());
        }
        let img_tok_u32 = self.config.image_token_id as u32;
        if ids.contains(&img_tok_u32) {
            let signed: Vec<i32> = ids.iter().map(|&id| id as i32).collect();
            let mask = vec![1i32; signed.len()];
            let (expanded, _) = crate::models::qwen3_vl_embedding::expand_image_placeholders(
                &signed,
                &mask,
                self.config.image_token_id,
                &counts,
            )
            .context("EmbeddingGemma 2 image placeholder expansion")?;
            Ok(expanded.into_iter().map(|id| id as u32).collect())
        } else {
            let mut out = ids.to_vec();
            for count in counts {
                out.push(self.config.boi_token_id as u32);
                out.extend(std::iter::repeat_n(img_tok_u32, count));
                out.push(self.config.eoi_token_id as u32);
            }
            Ok(out)
        }
    }
}

#[cfg(test)]
#[path = "embedding_gemma2_tests.rs"]
mod embedding_gemma2_tests;
