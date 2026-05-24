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

//! RT-DETRv2 transformer: deformable-attention decoder, MLP heads, and the
//! anchor priors used by encoder query selection.
//!
//! Port of `references/mlx-vlm/mlx_vlm/models/rt_detr_v2/transformer.py`.
//! Multi-scale deformable attention samples each feature level via the shared
//! bilinear `grid_sample` (see [`super::layers::grid_sample`]) and weighted-sums
//! across levels with the softmaxed attention weights.

use mlxcel_core::layers::{LayerNorm, Linear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::RtDetrV2Config;
use super::hybrid_encoder::{SelfAttention, load_layer_norm};
use super::layers::{Activation, grid_sample};

/// Per-level spatial shape `(H, W)`.
pub type SpatialShape = (i32, i32);

/// Numerically-stable inverse sigmoid: `log(clip(x) / (1 - clip(x)))`.
pub fn inverse_sigmoid(x: &MlxArray, eps: f32) -> UniquePtr<MlxArray> {
    let dt = mlxcel_core::array_dtype(x);
    let zero = mlxcel_core::full_f32(&[1], 0.0, dt);
    let one = mlxcel_core::full_f32(&[1], 1.0, dt);
    let eps_a = mlxcel_core::full_f32(&[1], eps, dt);
    let x = mlxcel_core::clip(x, &zero, &one);
    let x1 = mlxcel_core::clip(&x, &eps_a, &one);
    let one_minus = mlxcel_core::subtract(&one, &x);
    let x2 = mlxcel_core::clip(&one_minus, &eps_a, &one);
    mlxcel_core::log(&mlxcel_core::divide(&x1, &x2))
}

/// Multi-layer perceptron with ReLU between layers. Field `layers` is a list of
/// `nn.Linear` matching `RTDetrV2MLPPredictionHead` saved keys
/// `.layers.{i}.{weight,bias}`.
pub struct Mlp {
    layers: Vec<Linear>,
}

impl Mlp {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        num_layers: usize,
    ) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(Linear::from_weights(
                weights,
                &format!("{prefix}.layers.{i}"),
            )?);
        }
        Ok(Self { layers })
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let n = self.layers.len();
        let mut h = mlxcel_core::copy(x);
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h);
            if i < n - 1 {
                h = mlxcel_core::relu(&h);
            }
        }
        h
    }
}

/// Multi-scale deformable attention.
///
/// Reference points are 4D `(cx, cy, w, h)` normalized to `[0, 1]`. Sampling
/// offsets are predicted per `(n_heads, n_levels, n_points)` and scaled by
/// `1/n_points * ref_wh * offset_scale` before being added to the reference
/// center. Sampling itself is bilinear `grid_sample` per level; outputs are
/// concatenated and weighted-summed by the softmaxed attention weights.
struct MsDeformableAttention {
    sampling_offsets: Linear,
    attention_weights: Linear,
    value_proj: Linear,
    output_proj: Linear,
    n_heads: i32,
    head_dim: i32,
    n_levels: i32,
    n_points: i32,
    offset_scale: f32,
    /// `decoder_method == "default"` maps sampling locations from `[0,1]` to
    /// `[-1,1]` for grid_sample; `"discrete"` uses them as-is.
    method_default: bool,
}

impl MsDeformableAttention {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &RtDetrV2Config,
    ) -> Result<Self, String> {
        let d = cfg.d_model;
        let n_heads = cfg.decoder_attention_heads;
        Ok(Self {
            sampling_offsets: Linear::from_weights(weights, &format!("{prefix}.sampling_offsets"))?,
            attention_weights: Linear::from_weights(
                weights,
                &format!("{prefix}.attention_weights"),
            )?,
            value_proj: Linear::from_weights(weights, &format!("{prefix}.value_proj"))?,
            output_proj: Linear::from_weights(weights, &format!("{prefix}.output_proj"))?,
            n_heads: n_heads as i32,
            head_dim: (d / n_heads) as i32,
            n_levels: cfg.decoder_n_levels as i32,
            n_points: cfg.decoder_n_points as i32,
            offset_scale: cfg.decoder_offset_scale,
            method_default: cfg.decoder_method == "default",
        })
    }

    /// Args:
    /// - `query`: (B, Q, D).
    /// - `reference_points`: (B, Q, 1, 4) center+wh, broadcast across levels.
    /// - `value`: (B, sum_HW, D) flattened multi-scale encoder features.
    /// - `spatial_shapes`: per-level (H, W).
    /// - `pos`: optional (B, Q, D) added to query.
    fn forward(
        &self,
        query: &MlxArray,
        reference_points: &MlxArray,
        value: &MlxArray,
        spatial_shapes: &[SpatialShape],
        pos: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let query = match pos {
            Some(p) => mlxcel_core::add(query, p),
            None => mlxcel_core::copy(query),
        };
        let qshape = mlxcel_core::array_shape(&query);
        let (b, q, d) = (qshape[0], qshape[1], qshape[2]);
        let n_heads = self.n_heads;
        let head_dim = self.head_dim;
        let total_points = self.n_levels * self.n_points;

        let v_len = mlxcel_core::array_shape(value)[1];
        let v = self.value_proj.forward(value);
        let v = mlxcel_core::reshape(&v, &[b, v_len, n_heads, head_dim]);

        let offsets = self.sampling_offsets.forward(&query);
        let offsets = mlxcel_core::reshape(&offsets, &[b, q, n_heads, total_points, 2]);

        let attn = self.attention_weights.forward(&query);
        let attn = mlxcel_core::reshape(&attn, &[b, q, n_heads, total_points]);
        let attn = mlxcel_core::softmax(&attn, -1);

        // n_points_scale = 1/n_points broadcast to (1,1,1,total_points,1).
        let n_pts_scale = 1.0f32 / self.n_points as f32;

        // ref_xy: (B, Q, 1, 1, 2), ref_wh: (B, Q, 1, 1, 2) by slicing the 4D
        // reference and inserting the head axis. reference_points is
        // (B, Q, 1, 4): [:, :, None(level)->kept-as-1, :2] / [..., 2:].
        let ref_xy = slice_last2(reference_points, 0, 2); // (B,Q,1,2)
        let ref_wh = slice_last2(reference_points, 2, 4); // (B,Q,1,2)
        // Insert the head axis at position 2 -> (B,Q,1,1,2).
        let ref_xy = mlxcel_core::expand_dims(&ref_xy, 2);
        let ref_wh = mlxcel_core::expand_dims(&ref_wh, 2);

        // loc = ref_xy + offsets * n_pts_scale * ref_wh * offset_scale.
        let scaled = mlxcel_core::multiply_scalar(&offsets, n_pts_scale * self.offset_scale);
        let scaled = mlxcel_core::multiply(&scaled, &ref_wh);
        let loc = mlxcel_core::add(&ref_xy, &scaled); // (B,Q,n_heads,total_points,2)

        // Split loc per level along the points axis (axis 3).
        // Each level slice: (B,Q,n_heads,n_points,2).
        let mut sampled_per_level: Vec<UniquePtr<MlxArray>> =
            Vec::with_capacity(spatial_shapes.len());

        // Build the per-level value chunks (split along axis 1).
        let mut v_offset = 0i32;
        for (lvl, &(h, w)) in spatial_shapes.iter().enumerate() {
            let level_size = h * w;
            // value chunk: (B, level_size, n_heads, head_dim).
            let v_chunk = mlxcel_core::slice(
                &v,
                &[0, v_offset, 0, 0],
                &[b, v_offset + level_size, n_heads, head_dim],
            );
            v_offset += level_size;
            // reshape to (B, H, W, n_heads, head_dim), then
            // transpose(0,3,1,2,4) -> (B,n_heads,H,W,head_dim), then
            // reshape to (B*n_heads, H, W, head_dim).
            let v_l = mlxcel_core::reshape(&v_chunk, &[b, h, w, n_heads, head_dim]);
            let v_l = mlxcel_core::transpose_axes(&v_l, &[0, 3, 1, 2, 4]);
            let v_l = mlxcel_core::reshape(&v_l, &[b * n_heads, h, w, head_dim]);

            // sampling locations for this level: slice loc points
            // [lvl*n_points, (lvl+1)*n_points) along axis 3.
            let p0 = (lvl as i32) * self.n_points;
            let p1 = p0 + self.n_points;
            let samp = mlxcel_core::slice(&loc, &[0, 0, 0, p0, 0], &[b, q, n_heads, p1, 2]); // (B,Q,n_heads,n_points,2)
            // transpose(0,2,1,3,4) -> (B,n_heads,Q,n_points,2), reshape to
            // (B*n_heads, Q, n_points, 2).
            let samp = mlxcel_core::transpose_axes(&samp, &[0, 2, 1, 3, 4]);
            let samp = mlxcel_core::reshape(&samp, &[b * n_heads, q, self.n_points, 2]);
            let samp = if self.method_default {
                // 2*samp - 1.
                let t = mlxcel_core::multiply_scalar(&samp, 2.0);
                let one = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::array_dtype(&t));
                mlxcel_core::subtract(&t, &one)
            } else {
                samp
            };
            // grid_sample -> (B*n_heads, Q, n_points, head_dim).
            sampled_per_level.push(grid_sample(&v_l, &samp));
        }

        // Concatenate along the points axis -> (B*n_heads, Q, total_points, head_dim).
        let mut sampled = mlxcel_core::copy(&sampled_per_level[0]);
        for s in sampled_per_level.iter().skip(1) {
            sampled = mlxcel_core::concatenate(&sampled, s, 2);
        }

        // attn weights: (B,Q,n_heads,total_points) -> transpose(0,2,1,3) ->
        // (B,n_heads,Q,total_points) -> reshape (B*n_heads, Q, total_points).
        let w = mlxcel_core::transpose_axes(&attn, &[0, 2, 1, 3]);
        let w = mlxcel_core::reshape(&w, &[b * n_heads, q, total_points]);
        let w = mlxcel_core::expand_dims(&w, 3); // (B*n_heads, Q, total_points, 1)

        // out = (sampled * w).sum(axis=2) -> (B*n_heads, Q, head_dim).
        let weighted = mlxcel_core::multiply(&sampled, &w);
        let out = mlxcel_core::sum_axis(&weighted, 2, false);
        // reshape (B,n_heads,Q,head_dim) -> transpose(0,2,1,3) -> (B,Q,n_heads,head_dim)
        // -> reshape (B,Q,D).
        let out = mlxcel_core::reshape(&out, &[b, n_heads, q, head_dim]);
        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);
        let out = mlxcel_core::reshape(&out, &[b, q, d]);
        self.output_proj.forward(&out)
    }
}

/// One decoder block: self-attn -> norm -> deformable cross-attn -> norm ->
/// FFN -> norm.
struct DecoderLayer {
    self_attn: SelfAttention,
    self_attn_ln: LayerNorm,
    encoder_attn: MsDeformableAttention,
    encoder_attn_ln: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_ln: LayerNorm,
    act: Activation,
}

impl DecoderLayer {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &RtDetrV2Config,
    ) -> Result<Self, String> {
        let d = cfg.d_model;
        Ok(Self {
            self_attn: SelfAttention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                d,
                cfg.decoder_attention_heads,
            )?,
            self_attn_ln: load_layer_norm(
                weights,
                &format!("{prefix}.self_attn_layer_norm"),
                cfg.layer_norm_eps,
            )?,
            encoder_attn: MsDeformableAttention::from_weights(
                weights,
                &format!("{prefix}.encoder_attn"),
                cfg,
            )?,
            encoder_attn_ln: load_layer_norm(
                weights,
                &format!("{prefix}.encoder_attn_layer_norm"),
                cfg.layer_norm_eps,
            )?,
            fc1: Linear::from_weights(weights, &format!("{prefix}.fc1"))?,
            fc2: Linear::from_weights(weights, &format!("{prefix}.fc2"))?,
            final_ln: load_layer_norm(
                weights,
                &format!("{prefix}.final_layer_norm"),
                cfg.layer_norm_eps,
            )?,
            act: Activation::parse(&cfg.decoder_activation_function)?,
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        pos: &MlxArray,
        encoder_hidden_states: &MlxArray,
        reference_points: &MlxArray,
        spatial_shapes: &[SpatialShape],
    ) -> UniquePtr<MlxArray> {
        // Self-attention.
        let residual = mlxcel_core::copy(x);
        let h = self.self_attn.forward(x, Some(pos));
        let h = mlxcel_core::add(&residual, &h);
        let h = self.self_attn_ln.forward(&h);

        // Deformable cross-attention.
        let residual = mlxcel_core::copy(&h);
        let c = self.encoder_attn.forward(
            &h,
            reference_points,
            encoder_hidden_states,
            spatial_shapes,
            Some(pos),
        );
        let h = mlxcel_core::add(&residual, &c);
        let h = self.encoder_attn_ln.forward(&h);

        // FFN.
        let residual = mlxcel_core::copy(&h);
        let f = self.fc2.forward(&self.act.apply(&self.fc1.forward(&h)));
        let h = mlxcel_core::add(&residual, &f);
        self.final_ln.forward(&h)
    }
}

/// Output of the decoder forward (only the inference-relevant trajectories).
pub struct DecoderOutput {
    /// (B, num_queries, num_labels) — last layer logits.
    pub pred_logits: UniquePtr<MlxArray>,
    /// (B, num_queries, 4) — last layer reference points (cx,cy,w,h) in [0,1].
    pub pred_boxes: UniquePtr<MlxArray>,
}

/// Decoder stack with iterative bbox refinement. Per-layer `bbox_embed`
/// (3-layer MLP) and `class_embed` (Linear) heads are attached here.
pub struct Decoder {
    layers: Vec<DecoderLayer>,
    query_pos_head: Mlp,
    bbox_embed: Vec<Mlp>,
    class_embed: Vec<Linear>,
    layer_norm_eps: f32,
}

impl Decoder {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &RtDetrV2Config,
    ) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(cfg.decoder_layers);
        for i in 0..cfg.decoder_layers {
            layers.push(DecoderLayer::from_weights(
                weights,
                &format!("{prefix}.layers.{i}"),
                cfg,
            )?);
        }
        // query_pos_head is a 2-layer MLP (4 -> 2d -> d).
        let query_pos_head = Mlp::from_weights(weights, &format!("{prefix}.query_pos_head"), 2)?;

        let mut bbox_embed = Vec::with_capacity(cfg.decoder_layers);
        let mut class_embed = Vec::with_capacity(cfg.decoder_layers);
        for i in 0..cfg.decoder_layers {
            bbox_embed.push(Mlp::from_weights(
                weights,
                &format!("{prefix}.bbox_embed.{i}"),
                3,
            )?);
            class_embed.push(Linear::from_weights(
                weights,
                &format!("{prefix}.class_embed.{i}"),
            )?);
        }

        Ok(Self {
            layers,
            query_pos_head,
            bbox_embed,
            class_embed,
            layer_norm_eps: cfg.layer_norm_eps,
        })
    }

    /// Args:
    /// - `target`: (B, Q, D) initial query content.
    /// - `reference_points_unact`: (B, Q, 4) in logit space (pre-sigmoid).
    /// - `encoder_hidden_states`: (B, sum_HW, D).
    /// - `spatial_shapes`: per-level (H, W).
    pub fn forward(
        &self,
        target: &MlxArray,
        reference_points_unact: &MlxArray,
        encoder_hidden_states: &MlxArray,
        spatial_shapes: &[SpatialShape],
    ) -> DecoderOutput {
        let mut hidden = mlxcel_core::copy(target);
        let mut ref_points = mlxcel_core::sigmoid(reference_points_unact); // (B,Q,4)

        let mut last_logits: Option<UniquePtr<MlxArray>> = None;
        let mut last_refs: Option<UniquePtr<MlxArray>> = None;

        for idx in 0..self.layers.len() {
            // (B, Q, 1, 4) broadcasts across feature levels in cross-attn.
            let ref_input = mlxcel_core::expand_dims(&ref_points, 2);
            let pos_embed = self.query_pos_head.forward(&ref_points);
            hidden = self.layers[idx].forward(
                &hidden,
                &pos_embed,
                encoder_hidden_states,
                &ref_input,
                spatial_shapes,
            );

            let predicted_corners = self.bbox_embed[idx].forward(&hidden);
            let inv = inverse_sigmoid(&ref_points, 1e-5);
            let new_refs = mlxcel_core::sigmoid(&mlxcel_core::add(&predicted_corners, &inv));
            ref_points = mlxcel_core::copy(&new_refs);

            last_refs = Some(new_refs);
            last_logits = Some(self.class_embed[idx].forward(&hidden));
        }

        DecoderOutput {
            pred_logits: last_logits.expect("decoder must have >= 1 layer"),
            pred_boxes: last_refs.expect("decoder must have >= 1 layer"),
        }
    }

    pub fn layer_norm_eps(&self) -> f32 {
        self.layer_norm_eps
    }
}

/// Multi-scale anchor priors for encoder query selection.
///
/// Returns `(anchors_logit, valid_mask)` where anchors are `(1, sum_HW, 4)` in
/// logit space and the mask (`(1, sum_HW, 1)`, 0/1 float) marks positions whose
/// sigmoid falls in `[eps, 1-eps]`. Built in f32 on the host to mirror
/// `generate_anchors` exactly (`grid_size=0.05`, `eps=1e-2`).
pub fn generate_anchors(
    spatial_shapes: &[SpatialShape],
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    const GRID_SIZE: f32 = 0.05;
    const EPS: f32 = 1e-2;
    let big = f32::MAX;

    let total: usize = spatial_shapes.iter().map(|&(h, w)| (h * w) as usize).sum();
    let mut anchors_logit = vec![0f32; total * 4];
    let mut valid_mask = vec![0f32; total];

    let mut pos = 0usize;
    for (level, &(h, w)) in spatial_shapes.iter().enumerate() {
        let wh = GRID_SIZE * 2f32.powi(level as i32);
        for r in 0..h {
            for col in 0..w {
                // grid_xy = (col+0.5)/w, (r+0.5)/h ; anchors = [gx, gy, wh, wh].
                let gx = (col as f32 + 0.5) / w as f32;
                let gy = (r as f32 + 0.5) / h as f32;
                let anchor = [gx, gy, wh, wh];
                // valid if all components in (eps, 1-eps).
                let valid = anchor.iter().all(|&v| v > EPS && v < 1.0 - EPS);
                valid_mask[pos] = if valid { 1.0 } else { 0.0 };
                let base = pos * 4;
                for (j, &a) in anchor.iter().enumerate() {
                    // logit = log(a / (1-a)); masked-out positions -> +big.
                    anchors_logit[base + j] = if valid { (a / (1.0 - a)).ln() } else { big };
                }
                pos += 1;
            }
        }
    }

    let anchors = mlxcel_core::from_slice_f32(&anchors_logit, &[1, total as i32, 4]);
    let mask = mlxcel_core::from_slice_f32(&valid_mask, &[1, total as i32, 1]);
    (anchors, mask)
}

/// Slice `[start, stop)` along the last axis of a rank-N array, preserving rank.
fn slice_last2(a: &MlxArray, start: i32, stop: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(a);
    let last = shape.len() - 1;
    let mut starts: Vec<i32> = vec![0; shape.len()];
    let mut stops: Vec<i32> = shape.clone();
    starts[last] = start;
    stops[last] = stop;
    mlxcel_core::slice(a, &starts, &stops)
}
