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

//! Top-level RT-DETRv2 model: vision tower (backbone + hybrid encoder) ->
//! per-level decoder input projection -> encoder query selection -> deformable
//! decoder.
//!
//! Port of https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/rt_detr_v2/rt_detr_v2.py.

use std::path::Path;

use mlxcel_core::layers::{LayerNorm, Linear};
use mlxcel_core::weights::{WeightMap, load_weights_from_dir};
use mlxcel_core::{MlxArray, UniquePtr};

use super::backbone::Backbone;
use super::common::to_f32;
use super::config::RtDetrV2Config;
use super::hybrid_encoder::{HybridEncoder, load_layer_norm};
use super::layers::ConvBn;
use super::sanitize;
use super::transformer::{Decoder, Mlp, SpatialShape, generate_anchors};

/// The vision tower: backbone -> per-level 1x1 conv+BN input projection ->
/// hybrid encoder.
struct VisionTower {
    backbone: Backbone,
    encoder_input_proj: Vec<ConvBn>,
    hybrid_encoder: HybridEncoder,
}

impl VisionTower {
    fn from_weights(weights: &WeightMap, cfg: &RtDetrV2Config) -> Result<Self, String> {
        let eps = cfg.batch_norm_eps;
        let backbone = Backbone::from_weights(weights, "vision.backbone", cfg.backbone(), eps)?;
        let mut encoder_input_proj = Vec::with_capacity(cfg.num_levels());
        for i in 0..cfg.num_levels() {
            encoder_input_proj.push(ConvBn::from_weights(
                weights,
                &format!("vision.encoder_input_proj.{i}"),
                1,
                eps,
            )?);
        }
        let hybrid_encoder = HybridEncoder::from_weights(weights, "vision.hybrid_encoder", cfg)?;
        Ok(Self {
            backbone,
            encoder_input_proj,
            hybrid_encoder,
        })
    }

    fn forward(&self, pixel_values: &MlxArray) -> Vec<UniquePtr<MlxArray>> {
        let c_features = self.backbone.forward(pixel_values);
        let proj: Vec<UniquePtr<MlxArray>> = self
            .encoder_input_proj
            .iter()
            .zip(c_features.iter())
            .map(|(p, c)| p.forward(c))
            .collect();
        self.hybrid_encoder.forward(proj)
    }
}

/// Detection forward output: pre-decode logits and normalized boxes.
pub struct DetectionOutput {
    /// (B, num_queries, num_labels).
    pub pred_logits: UniquePtr<MlxArray>,
    /// (B, num_queries, 4) normalized (cx, cy, w, h) in [0, 1].
    pub pred_boxes: UniquePtr<MlxArray>,
}

/// The full RT-DETRv2 detection model.
pub struct RtDetrV2Model {
    config: RtDetrV2Config,
    vision: VisionTower,
    decoder_input_proj: Vec<ConvBn>,
    enc_output_fc: Linear,
    enc_output_ln: LayerNorm,
    enc_score_head: Linear,
    enc_bbox_head: Mlp,
    decoder: Decoder,
}

impl RtDetrV2Model {
    /// Load a model from a directory containing `config.json` and one or more
    /// `*.safetensors`. Accepts both pre-converted MLX checkpoints and raw HF
    /// `RTDetrV2ForObjectDetection` checkpoints (the rename pipeline runs only
    /// when [`sanitize::needs_sanitize`] detects HF layout).
    pub fn load<P: AsRef<Path>>(dir: P) -> Result<Self, String> {
        let dir = dir.as_ref();
        let config_path = dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| format!("failed to read {}: {e}", config_path.display()))?;
        let config = RtDetrV2Config::from_json_str(&config_str)?;
        config.validate()?;

        let raw = load_weights_from_dir(dir)?;
        let weights = if sanitize::needs_sanitize(&raw) {
            sanitize::sanitize(raw)
        } else {
            raw
        };

        Self::from_weights(weights, config)
    }

    /// Build the model from an already-sanitized (MLX-layout) weight map.
    pub fn from_weights(weights: WeightMap, config: RtDetrV2Config) -> Result<Self, String> {
        let eps = config.batch_norm_eps;
        let ln_eps = config.layer_norm_eps;
        let d = config.d_model;

        let vision = VisionTower::from_weights(&weights, &config)?;

        let mut decoder_input_proj = Vec::with_capacity(config.decoder_in_channels.len());
        for i in 0..config.decoder_in_channels.len() {
            decoder_input_proj.push(ConvBn::from_weights(
                &weights,
                &format!("decoder_input_proj.{i}"),
                1,
                eps,
            )?);
        }

        let enc_output_fc = Linear::from_weights(&weights, "enc_output.fc")?;
        let enc_output_ln = load_layer_norm(&weights, "enc_output.ln", ln_eps)?;
        let enc_score_head = Linear::from_weights(&weights, "enc_score_head")?;
        let enc_bbox_head = Mlp::from_weights(&weights, "enc_bbox_head", 3)?;
        let decoder = Decoder::from_weights(&weights, "decoder", &config)?;

        // `denoising_class_embed` is training-only; intentionally not loaded.
        let _ = d;

        Ok(Self {
            config,
            vision,
            decoder_input_proj,
            enc_output_fc,
            enc_output_ln,
            enc_score_head,
            enc_bbox_head,
            decoder,
        })
    }

    pub fn config(&self) -> &RtDetrV2Config {
        &self.config
    }

    /// Forward pass.
    ///
    /// `pixel_values`: (B, image_size, image_size, 3) NHWC in [0, 1]. The whole
    /// graph runs in f32 for box-coordinate precision regardless of the stored
    /// checkpoint dtype.
    pub fn forward(&self, pixel_values: &MlxArray) -> DetectionOutput {
        let pixel_values = to_f32(pixel_values);
        let enc_features = self.vision.forward(&pixel_values);

        // Per-level decoder input projection (1x1 conv + BN).
        let proj: Vec<UniquePtr<MlxArray>> = self
            .decoder_input_proj
            .iter()
            .zip(enc_features.iter())
            .map(|(p, f)| p.forward(f))
            .collect();

        let spatial_shapes: Vec<SpatialShape> = proj
            .iter()
            .map(|f| {
                let s = mlxcel_core::array_shape(f);
                (s[1], s[2])
            })
            .collect();

        // Flatten each level (B, H*W, C) and concatenate along the token axis.
        let mut flat = {
            let s = mlxcel_core::array_shape(&proj[0]);
            mlxcel_core::reshape(&proj[0], &[s[0], s[1] * s[2], s[3]])
        };
        for f in proj.iter().skip(1) {
            let s = mlxcel_core::array_shape(f);
            let r = mlxcel_core::reshape(f, &[s[0], s[1] * s[2], s[3]]);
            flat = mlxcel_core::concatenate(&flat, &r, 1);
        }

        // Encoder query selection: score every position, take top-K.
        let (anchors, valid_mask) = generate_anchors(&spatial_shapes);
        // memory = flat * valid_mask (broadcast over channels).
        let memory = mlxcel_core::multiply(&flat, &valid_mask);
        let output_memory = self
            .enc_output_ln
            .forward(&self.enc_output_fc.forward(&memory));
        let enc_scores = self.enc_score_head.forward(&output_memory); // (B, S, num_labels)
        let enc_coord_logits =
            mlxcel_core::add(&self.enc_bbox_head.forward(&output_memory), &anchors); // (B, S, 4)

        let k = self.config.num_queries as i32;
        // scores_max over labels (axis -1) -> (B, S).
        let scores_max = mlxcel_core::max_axis(&enc_scores, -1, false);
        // top-K indices: argpartition on -scores then take first K, then sort.
        let neg_scores = mlxcel_core::multiply_scalar(&scores_max, -1.0);
        let part = mlxcel_core::argpartition(&neg_scores, k - 1, 1); // (B, S)
        let topk_idx = slice_first_k(&part, k); // (B, K)
        let topk_scores = mlxcel_core::take_along_axis(&scores_max, &topk_idx, 1); // (B, K)
        let order = mlxcel_core::argsort(&mlxcel_core::multiply_scalar(&topk_scores, -1.0), 1);
        let topk_idx = mlxcel_core::take_along_axis(&topk_idx, &order, 1); // (B, K) sorted

        let s0 = mlxcel_core::array_shape(&topk_idx)[0];
        // Gather reference points: idx broadcast to (B, K, 4).
        let gather_idx_b =
            mlxcel_core::broadcast_to(&mlxcel_core::expand_dims(&topk_idx, 2), &[s0, k, 4]);
        let ref_points_unact = mlxcel_core::take_along_axis(&enc_coord_logits, &gather_idx_b, 1); // (B, K, 4)

        // Gather target content: idx broadcast to (B, K, D).
        let d_model = mlxcel_core::array_shape(&output_memory)[2];
        let gather_idx_d =
            mlxcel_core::broadcast_to(&mlxcel_core::expand_dims(&topk_idx, 2), &[s0, k, d_model]);
        let target = mlxcel_core::take_along_axis(&output_memory, &gather_idx_d, 1); // (B, K, D)

        let dec = self
            .decoder
            .forward(&target, &ref_points_unact, &flat, &spatial_shapes);

        DetectionOutput {
            pred_logits: dec.pred_logits,
            pred_boxes: dec.pred_boxes,
        }
    }
}

/// Slice the first `k` entries along axis 1 of a 2D index array `(B, S)`.
fn slice_first_k(idx: &MlxArray, k: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(idx);
    mlxcel_core::slice(idx, &[0, 0], &[shape[0], k])
}
