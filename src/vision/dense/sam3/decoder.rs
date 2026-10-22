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
//! SAM3 post-norm DETR decoder with iterative box refinement and BoxRPB.

use std::f32::consts::PI;

use mlxcel_core::layers::LayerNorm;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::DetrDecoderConfig;
use super::linear::UnifiedLinear;

fn weight(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(key)
        .map(|value| mlxcel_core::copy(value))
        .ok_or_else(|| format!("SAM3 weight not found: {key}"))
}

pub(crate) fn layer_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<LayerNorm, String> {
    Ok(LayerNorm::new(
        weight(weights, &format!("{prefix}.weight"))?,
        weights
            .get(&format!("{prefix}.bias"))
            .map(|value| mlxcel_core::copy(value)),
        eps,
    ))
}

#[rustfmt::skip]
pub(crate) struct MultiheadAttention { q: UnifiedLinear, k: UnifiedLinear, v: UnifiedLinear, out: UnifiedLinear, heads: i32, head_dim: i32, scale: f32 }

impl MultiheadAttention {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        hidden: usize,
        heads: usize,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let linear = |name: &str| {
            UnifiedLinear::from_weights(weights, &format!("{prefix}.{name}"), group_size, bits)
        };
        let head_dim = (hidden / heads) as i32;
        Ok(Self {
            q: linear("q_proj")?,
            k: linear("k_proj")?,
            v: linear("v_proj")?,
            out: linear("o_proj")?,
            heads: heads as i32,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    pub(crate) fn forward(
        &self,
        query: &MlxArray,
        key: &MlxArray,
        value: &MlxArray,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let qs = mlxcel_core::array_shape(query);
        let ks = mlxcel_core::array_shape(key);
        let (b, nq, nk, d) = (qs[0], qs[1], ks[1], qs[2]);
        let split = |linear: &UnifiedLinear, x: &MlxArray, n: i32| {
            let x = linear.forward(x);
            let x = mlxcel_core::reshape(&x, &[b, n, self.heads, self.head_dim]);
            mlxcel_core::transpose_axes(&x, &[0, 2, 1, 3])
        };
        let q = split(&self.q, query, nq);
        let k = split(&self.k, key, nk);
        let v = split(&self.v, value, nk);
        let mask_ptr = mask.map_or(std::ptr::null(), |m| m as *const MlxArray);
        // SAFETY: mask_ptr is null or borrowed for the duration of this call.
        let x = unsafe {
            mlxcel_core::fast_scaled_dot_product_attention(&q, &k, &v, self.scale, mask_ptr)
        };
        let x = mlxcel_core::transpose_axes(&x, &[0, 2, 1, 3]);
        self.out.forward(&mlxcel_core::reshape(&x, &[b, nq, d]))
    }
}

struct DecoderLayer {
    self_attn: MultiheadAttention,
    self_norm: LayerNorm,
    text_attn: MultiheadAttention,
    text_norm: LayerNorm,
    vision_attn: MultiheadAttention,
    vision_norm: LayerNorm,
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
    mlp_norm: LayerNorm,
}

impl DecoderLayer {
    fn from_weights(
        weights: &WeightMap,
        cfg: &DetrDecoderConfig,
        prefix: &str,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let (d, heads, eps) = (cfg.hidden_size, cfg.num_attention_heads, cfg.layer_norm_eps);
        Ok(Self {
            self_attn: MultiheadAttention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                d,
                heads,
                gs,
                bits,
            )?,
            self_norm: layer_norm(weights, &format!("{prefix}.self_attn_layer_norm"), eps)?,
            text_attn: MultiheadAttention::from_weights(
                weights,
                &format!("{prefix}.text_cross_attn"),
                d,
                heads,
                gs,
                bits,
            )?,
            text_norm: layer_norm(
                weights,
                &format!("{prefix}.text_cross_attn_layer_norm"),
                eps,
            )?,
            vision_attn: MultiheadAttention::from_weights(
                weights,
                &format!("{prefix}.vision_cross_attn"),
                d,
                heads,
                gs,
                bits,
            )?,
            vision_norm: layer_norm(
                weights,
                &format!("{prefix}.vision_cross_attn_layer_norm"),
                eps,
            )?,
            fc1: UnifiedLinear::from_weights(weights, &format!("{prefix}.mlp.fc1"), gs, bits)?,
            fc2: UnifiedLinear::from_weights(weights, &format!("{prefix}.mlp.fc2"), gs, bits)?,
            mlp_norm: layer_norm(weights, &format!("{prefix}.mlp_layer_norm"), eps)?,
        })
    }

    fn forward(
        &self,
        hidden: &MlxArray,
        query_pos: &MlxArray,
        text: &MlxArray,
        vision: &MlxArray,
        vision_pos: &MlxArray,
        text_mask: Option<&MlxArray>,
        vision_mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let qk = mlxcel_core::add(hidden, query_pos);
        let x = self.self_attn.forward(&qk, &qk, hidden, None);
        let x = self.self_norm.forward(&mlxcel_core::add(hidden, &x));
        let q = mlxcel_core::add(&x, query_pos);
        let y = self.text_attn.forward(&q, text, text, text_mask);
        let x = self.text_norm.forward(&mlxcel_core::add(&x, &y));
        let q = mlxcel_core::add(&x, query_pos);
        let k = mlxcel_core::add(vision, vision_pos);
        let y = self.vision_attn.forward(&q, &k, vision, vision_mask);
        let x = self.vision_norm.forward(&mlxcel_core::add(&x, &y));
        let y = self.fc2.forward(&mlxcel_core::relu(&self.fc1.forward(&x)));
        self.mlp_norm.forward(&mlxcel_core::add(&x, &y))
    }
}

struct MlpHead {
    layers: Vec<UnifiedLinear>,
}

impl MlpHead {
    fn from_names(
        weights: &WeightMap,
        prefix: &str,
        names: &[&str],
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(names.len());
        for name in names {
            layers.push(UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.{name}"),
                gs,
                bits,
            )?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let mut y = mlxcel_core::copy(x);
        for (index, layer) in self.layers.iter().enumerate() {
            y = layer.forward(&y);
            if index + 1 != self.layers.len() {
                y = mlxcel_core::relu(&y);
            }
        }
        y
    }
}

fn inverse_sigmoid(x: &MlxArray) -> UniquePtr<MlxArray> {
    let lo = mlxcel_core::full_like(x, 1e-5);
    let hi = mlxcel_core::full_like(x, 1.0 - 1e-5);
    let clipped = mlxcel_core::clip(x, &lo, &hi);
    let one = mlxcel_core::full_like(&clipped, 1.0);
    mlxcel_core::log(&mlxcel_core::divide(
        &clipped,
        &mlxcel_core::subtract(&one, &clipped),
    ))
}

fn sine_box_embedding(boxes: &MlxArray, hidden: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(boxes);
    let (b, q, feats) = (shape[0], shape[1], hidden / 2);
    let dim: Vec<f32> = (0..feats)
        .map(|i| 10_000.0f32.powf((2 * (i / 2)) as f32 / feats as f32))
        .collect();
    let dim = mlxcel_core::reshape(&mlxcel_core::from_slice_f32(&dim, &[feats]), &[1, 1, feats]);
    let even: Vec<i32> = (0..feats).step_by(2).collect();
    let odd: Vec<i32> = (1..feats).step_by(2).collect();
    let even = mlxcel_core::from_slice_i32(&even, &[feats / 2]);
    let odd = mlxcel_core::from_slice_i32(&odd, &[feats / 2]);
    let encode = |coord: i32| {
        let x = mlxcel_core::slice(boxes, &[0, 0, coord], &[b, q, coord + 1]);
        let x = mlxcel_core::multiply_scalar(&x, 2.0 * PI);
        let x = mlxcel_core::divide(&x, &dim);
        let sin = mlxcel_core::sin(&mlxcel_core::take(&x, &even, 2));
        let cos = mlxcel_core::cos(&mlxcel_core::take(&x, &odd, 2));
        mlxcel_core::reshape(&mlxcel_core::stack_owned(&[sin, cos], -1), &[b, q, feats])
    };
    // HF order is y, x, width, height.
    let y = encode(1);
    let x = encode(0);
    let w = encode(2);
    let h = encode(3);
    mlxcel_core::concatenate_many(&[&y, &x, &w, &h], -1)
}

fn additive_text_mask(mask: &MlxArray) -> UniquePtr<MlxArray> {
    let valid = mlxcel_core::astype(mask, mlxcel_core::dtype::FLOAT32);
    let invalid = mlxcel_core::subtract(&mlxcel_core::full_like(&valid, 1.0), &valid);
    let invalid = mlxcel_core::multiply_scalar(&invalid, -1e9);
    mlxcel_core::expand_dims_multi(&invalid, &[1, 2])
}

pub struct DetrDecoderOutput {
    pub hidden_states: UniquePtr<MlxArray>,
    pub boxes: UniquePtr<MlxArray>,
    pub presence_logits: UniquePtr<MlxArray>,
}

pub struct DetrDecoder {
    layers: Vec<DecoderLayer>,
    output_norm: LayerNorm,
    query_embed: UniquePtr<MlxArray>,
    reference_points: UniquePtr<MlxArray>,
    presence_token: UniquePtr<MlxArray>,
    presence_head: MlpHead,
    presence_norm: LayerNorm,
    box_head: MlpHead,
    ref_point_head: MlpHead,
    rpb_x: MlpHead,
    rpb_y: MlpHead,
    hidden: i32,
    heads: i32,
    queries: i32,
}

impl DetrDecoder {
    pub fn from_weights(
        weights: &WeightMap,
        cfg: &DetrDecoderConfig,
        prefix: &str,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            layers.push(DecoderLayer::from_weights(
                weights,
                cfg,
                &format!("{prefix}.layers.{i}"),
                gs,
                bits,
            )?);
        }
        let head3 = ["layer1", "layer2", "layer3"];
        let head2 = ["layer1", "layer2"];
        Ok(Self {
            layers,
            output_norm: layer_norm(
                weights,
                &format!("{prefix}.output_layer_norm"),
                cfg.layer_norm_eps,
            )?,
            query_embed: weight(weights, &format!("{prefix}.query_embed.weight"))?,
            reference_points: weight(weights, &format!("{prefix}.reference_points.weight"))?,
            presence_token: weight(weights, &format!("{prefix}.presence_token.weight"))?,
            presence_head: MlpHead::from_names(
                weights,
                &format!("{prefix}.presence_head"),
                &head3,
                gs,
                bits,
            )?,
            presence_norm: layer_norm(
                weights,
                &format!("{prefix}.presence_layer_norm"),
                cfg.layer_norm_eps,
            )?,
            box_head: MlpHead::from_names(
                weights,
                &format!("{prefix}.box_head"),
                &head3,
                gs,
                bits,
            )?,
            ref_point_head: MlpHead::from_names(
                weights,
                &format!("{prefix}.ref_point_head"),
                &head2,
                gs,
                bits,
            )?,
            rpb_x: MlpHead::from_names(
                weights,
                &format!("{prefix}.box_rpb_embed_x"),
                &head2,
                gs,
                bits,
            )?,
            rpb_y: MlpHead::from_names(
                weights,
                &format!("{prefix}.box_rpb_embed_y"),
                &head2,
                gs,
                bits,
            )?,
            hidden: cfg.hidden_size as i32,
            heads: cfg.num_attention_heads as i32,
            queries: cfg.num_queries as i32,
        })
    }

    pub fn forward(
        &self,
        vision: &MlxArray,
        text: &MlxArray,
        vision_pos: &MlxArray,
        text_mask: Option<&MlxArray>,
        spatial_shape: Option<(i32, i32)>,
    ) -> DetrDecoderOutput {
        let b = mlxcel_core::array_shape(vision)[0];
        let query = mlxcel_core::broadcast_to(
            &mlxcel_core::expand_dims(&self.query_embed, 0),
            &[b, self.queries, self.hidden],
        );
        let refs = mlxcel_core::sigmoid(&mlxcel_core::broadcast_to(
            &mlxcel_core::expand_dims(&self.reference_points, 0),
            &[b, self.queries, 4],
        ));
        let presence = mlxcel_core::broadcast_to(
            &mlxcel_core::expand_dims(&self.presence_token, 0),
            &[b, 1, self.hidden],
        );
        let mut hidden = mlxcel_core::concatenate(&presence, &query, 1);
        let mut reference_boxes = refs;
        let text_cross_mask = text_mask.map(additive_text_mask);
        let mut all_hidden = Vec::with_capacity(self.layers.len());
        let mut all_boxes = Vec::with_capacity(self.layers.len());
        let mut all_presence = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let query_pos = self
                .ref_point_head
                .forward(&sine_box_embedding(&reference_boxes, self.hidden));
            let zero =
                mlxcel_core::zeros(&[b, 1, self.hidden], mlxcel_core::array_dtype(&query_pos));
            let query_pos = mlxcel_core::concatenate(&zero, &query_pos, 1);
            let rpb = spatial_shape.map(|shape| self.compute_rpb(&reference_boxes, shape));
            let vision_mask = rpb.as_ref().map(|bias| {
                let s = mlxcel_core::array_shape(bias);
                let zero = mlxcel_core::zeros(&[b, s[1], 1, s[3]], mlxcel_core::array_dtype(bias));
                mlxcel_core::concatenate(&zero, bias, 2)
            });
            hidden = layer.forward(
                &hidden,
                &query_pos,
                text,
                vision,
                vision_pos,
                text_cross_mask.as_deref(),
                vision_mask.as_deref(),
            );
            let query_hs =
                mlxcel_core::slice(&hidden, &[0, 1, 0], &[b, self.queries + 1, self.hidden]);
            let query_hs = self.output_norm.forward(&query_hs);
            let delta = self.box_head.forward(&query_hs);
            let new_ref = mlxcel_core::sigmoid(&mlxcel_core::add(
                &inverse_sigmoid(&reference_boxes),
                &delta,
            ));
            reference_boxes = mlxcel_core::stop_gradient(&new_ref);
            let pres = mlxcel_core::slice(&hidden, &[0, 0, 0], &[b, 1, self.hidden]);
            let pres = self
                .presence_head
                .forward(&self.presence_norm.forward(&pres));
            let pres = mlxcel_core::squeeze_axis(&pres, -1);
            let pres = mlxcel_core::clip(
                &pres,
                &mlxcel_core::full_like(&pres, -10.0),
                &mlxcel_core::full_like(&pres, 10.0),
            );
            all_hidden.push(query_hs);
            all_boxes.push(new_ref);
            all_presence.push(pres);
        }
        DetrDecoderOutput {
            hidden_states: mlxcel_core::stack_owned(&all_hidden, 0),
            boxes: mlxcel_core::stack_owned(&all_boxes, 0),
            presence_logits: mlxcel_core::stack_owned(&all_presence, 0),
        }
    }

    fn compute_rpb(&self, boxes: &MlxArray, shape: (i32, i32)) -> UniquePtr<MlxArray> {
        let (height, width) = shape;
        let s = mlxcel_core::array_shape(boxes);
        let (b, q) = (s[0], s[1]);
        let coord = |i| {
            mlxcel_core::squeeze_axis(&mlxcel_core::slice(boxes, &[0, 0, i], &[b, q, i + 1]), 2)
        };
        let (cx, cy, w, h) = (coord(0), coord(1), coord(2), coord(3));
        let half_w = mlxcel_core::multiply_scalar(&w, 0.5);
        let half_h = mlxcel_core::multiply_scalar(&h, 0.5);
        let x_bounds = mlxcel_core::stack_owned(
            &[
                mlxcel_core::subtract(&cx, &half_w),
                mlxcel_core::add(&cx, &half_w),
            ],
            -1,
        );
        let y_bounds = mlxcel_core::stack_owned(
            &[
                mlxcel_core::subtract(&cy, &half_h),
                mlxcel_core::add(&cy, &half_h),
            ],
            -1,
        );
        let xs: Vec<f32> = (0..width).map(|i| i as f32 / width as f32).collect();
        let ys: Vec<f32> = (0..height).map(|i| i as f32 / height as f32).collect();
        let xs = mlxcel_core::reshape(
            &mlxcel_core::from_slice_f32(&xs, &[width]),
            &[1, 1, width, 1],
        );
        let ys = mlxcel_core::reshape(
            &mlxcel_core::from_slice_f32(&ys, &[height]),
            &[1, 1, height, 1],
        );
        let dx = mlxcel_core::subtract(&xs, &mlxcel_core::expand_dims(&x_bounds, 2));
        let dy = mlxcel_core::subtract(&ys, &mlxcel_core::expand_dims(&y_bounds, 2));
        let encode = |delta: &MlxArray| {
            let x = mlxcel_core::multiply_scalar(delta, 8.0);
            let log = mlxcel_core::log2(&mlxcel_core::add(
                &mlxcel_core::abs(&x),
                &mlxcel_core::full_like(&x, 1.0),
            ));
            mlxcel_core::multiply_scalar(
                &mlxcel_core::multiply(&mlxcel_core::sign(&x), &log),
                1.0 / 8.0f32.log2(),
            )
        };
        let rx = self.rpb_x.forward(&encode(&dx));
        let ry = self.rpb_y.forward(&encode(&dy));
        let bias = mlxcel_core::add(
            &mlxcel_core::expand_dims(&ry, 3),
            &mlxcel_core::expand_dims(&rx, 2),
        );
        let bias = mlxcel_core::reshape(&bias, &[b, q, height * width, self.heads]);
        mlxcel_core::transpose_axes(&bias, &[0, 3, 1, 2])
    }
}
