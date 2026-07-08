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

//! PaddleOCR-VL NaViT dynamic-resolution vision encoder.
//!
//! A SigLIP-style ViT adapted for native dynamic resolution (NaViT):
//! - Conv2d patch embedding (kernel == stride == patch_size), evaluated as a
//!   Linear over flattened `[C, patch, patch]` patches.
//! - Learned absolute position embeddings, bilinearly interpolated to each
//!   image's `(h, w)` patch grid.
//! - 2D vision rotary position embeddings in raw row-major order (NO spatial
//!   merge reordering, unlike Qwen2-VL).
//! - Fused QKV attention over variable-length packed sequences (`cu_seqlens`).
//! - `post_layernorm` after the transformer stack.
//!
//! The spatial-merge projector lives in the connector
//! (`vision::connectors::paddleocr_vl`), matching the reference module split.
//!
//! Used by: PaddleOCR-VL
//! Reference: https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/paddleocr_vl/vision.py.

use super::VisionEncoderOutput;
use super::qwen2_vl::{apply_rotary_pos_emb_vision, concat_many};
use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use std::collections::BTreeMap;

/// PaddleOCR-VL vision encoder configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct PaddleOcrVisionConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_num_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_channels")]
    pub num_channels: usize,
    #[serde(default = "default_image_size")]
    pub image_size: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f32,
    #[serde(default = "default_spatial_merge_size")]
    pub spatial_merge_size: usize,
    /// Quantization group_size (inherited from top-level config).
    #[serde(default)]
    pub quant_group_size: i32,
    /// Quantization bits (inherited from top-level config).
    #[serde(default)]
    pub quant_bits: i32,
}

fn default_hidden_size() -> usize {
    1152
}
fn default_intermediate_size() -> usize {
    4304
}
fn default_num_layers() -> usize {
    27
}
fn default_num_heads() -> usize {
    16
}
fn default_num_channels() -> usize {
    3
}
fn default_image_size() -> usize {
    384
}
fn default_patch_size() -> usize {
    14
}
fn default_layer_norm_eps() -> f32 {
    1e-6
}
fn default_spatial_merge_size() -> usize {
    2
}

fn load_layer_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<LayerNorm, String> {
    let weight = weights
        .get(&format!("{prefix}.weight"))
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {prefix}.weight"))?;
    let bias = weights
        .get(&format!("{prefix}.bias"))
        .map(|b| mlxcel_core::copy(b));
    Ok(LayerNorm::new(weight, bias, eps))
}

// Patch embedding: Conv2d(kernel == stride == patch) evaluated as a Linear.
struct PatchEmbed {
    weight: UniquePtr<MlxArray>, // [embed, C * patch * patch]
    bias: Option<UniquePtr<MlxArray>>,
}

impl PatchEmbed {
    fn from_weights(
        weights: &WeightMap,
        config: &PaddleOcrVisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let key = format!("{prefix}.weight");
        let w = weights.get(&key).ok_or_else(|| format!("Missing {key}"))?;
        let shape = mlxcel_core::array_shape(w);
        let embed = config.hidden_size as i32;
        let in_features = (config.num_channels * config.patch_size * config.patch_size) as i32;

        // PyTorch Conv2d weight is [out, in_channels, kH, kW]; flattening the
        // trailing three axes yields [embed, C * patch * patch] in (C, h, w)
        // order, matching the processor's per-patch pixel flattening. Some
        // exports carry the MLX channels-last layout [out, kH, kW, in]; detect
        // that and reorder back to (C, h, w) before flattening.
        let weight = if shape.len() == 4 {
            let last_is_channels = shape[3] == config.num_channels as i32
                && shape[1] == config.patch_size as i32
                && shape[2] == config.patch_size as i32;
            if last_is_channels {
                let reordered = mlxcel_core::transpose_axes(w, &[0, 3, 1, 2]);
                mlxcel_core::reshape(&reordered, &[embed, in_features])
            } else {
                mlxcel_core::reshape(w, &[embed, in_features])
            }
        } else if shape.len() == 2 {
            mlxcel_core::copy(w)
        } else {
            return Err(format!(
                "Unexpected patch_embedding weight shape: {shape:?}"
            ));
        };

        let bias = weights
            .get(&format!("{prefix}.bias"))
            .map(|b| mlxcel_core::copy(b));

        Ok(Self { weight, bias })
    }

    /// hidden_states: `[total_patches, C * patch * patch]` -> `[total_patches, embed]`.
    fn forward(&self, hidden_states: &MlxArray) -> UniquePtr<MlxArray> {
        let wt = mlxcel_core::transpose(&self.weight);
        let out = mlxcel_core::matmul(hidden_states, &wt);
        match &self.bias {
            Some(b) => mlxcel_core::add(&out, b),
            None => out,
        }
    }
}

// Learned absolute position embedding with bilinear interpolation.
struct PositionEmbedding {
    weight: UniquePtr<MlxArray>, // [num_positions, embed]
    num_positions: i32,
    embed: i32,
}

impl PositionEmbedding {
    fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        let key = format!("{prefix}.weight");
        let w = weights.get(&key).ok_or_else(|| format!("Missing {key}"))?;
        let shape = mlxcel_core::array_shape(w);
        Ok(Self {
            weight: mlxcel_core::copy(w),
            num_positions: shape[0],
            embed: shape[1],
        })
    }

    /// Bilinearly interpolate the `sqrt(N) x sqrt(N)` learned grid to `[h*w, embed]`.
    ///
    /// Mirrors https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/paddleocr_vl/vision.py
    /// (`interpolate_pos_encoding`) and the shared `bilinear_interpolate`
    /// helper (align_corners=False). Integer indices and weights are derived on
    /// the host and gathered with `take`, so no device floor/clip is needed.
    fn interpolate(&self, h: i32, w: i32) -> UniquePtr<MlxArray> {
        let side = (self.num_positions as f64).sqrt().round() as i32;
        let h_in = side;
        let w_in = side;

        let row_pos: Vec<f64> = (0..h)
            .map(|i| (i as f64 + 0.5) * h_in as f64 / h as f64 - 0.5)
            .collect();
        let col_pos: Vec<f64> = (0..w)
            .map(|j| (j as f64 + 0.5) * w_in as f64 / w as f64 - 0.5)
            .collect();

        let clamp = |v: i32, hi: i32| v.max(0).min(hi - 1);
        // Clipped floor / ceil indices and the reference's clipped-floor weights.
        let row_floor: Vec<i32> = row_pos
            .iter()
            .map(|&p| clamp(p.floor() as i32, h_in))
            .collect();
        let row_ceil: Vec<i32> = row_pos
            .iter()
            .map(|&p| clamp(p.floor() as i32 + 1, h_in))
            .collect();
        let col_floor: Vec<i32> = col_pos
            .iter()
            .map(|&p| clamp(p.floor() as i32, w_in))
            .collect();
        let col_ceil: Vec<i32> = col_pos
            .iter()
            .map(|&p| clamp(p.floor() as i32 + 1, w_in))
            .collect();
        let row_w: Vec<f32> = row_pos
            .iter()
            .zip(&row_floor)
            .map(|(&p, &f)| (p - f as f64) as f32)
            .collect();
        let col_w: Vec<f32> = col_pos
            .iter()
            .zip(&col_floor)
            .map(|(&p, &f)| (p - f as f64) as f32)
            .collect();

        let n = (h * w) as usize;
        let mut idx_tl = Vec::with_capacity(n);
        let mut idx_tr = Vec::with_capacity(n);
        let mut idx_bl = Vec::with_capacity(n);
        let mut idx_br = Vec::with_capacity(n);
        let mut w_tl = Vec::with_capacity(n);
        let mut w_tr = Vec::with_capacity(n);
        let mut w_bl = Vec::with_capacity(n);
        let mut w_br = Vec::with_capacity(n);
        for i in 0..h as usize {
            for j in 0..w as usize {
                let rf = row_floor[i];
                let rc = row_ceil[i];
                let cf = col_floor[j];
                let cc = col_ceil[j];
                idx_tl.push(rf * w_in + cf);
                idx_tr.push(rf * w_in + cc);
                idx_bl.push(rc * w_in + cf);
                idx_br.push(rc * w_in + cc);
                let rw = row_w[i];
                let cw = col_w[j];
                w_tl.push((1.0 - rw) * (1.0 - cw));
                w_tr.push((1.0 - rw) * cw);
                w_bl.push(rw * (1.0 - cw));
                w_br.push(rw * cw);
            }
        }

        let flat = mlxcel_core::reshape(&self.weight, &[h_in * w_in, self.embed]);
        let gather = |idx: &[i32], wts: &[f32]| -> UniquePtr<MlxArray> {
            let idx_arr = mlxcel_core::from_slice_i32(idx, &[n as i32]);
            let g = mlxcel_core::take(&flat, &idx_arr, 0);
            let wa = mlxcel_core::from_slice_f32(wts, &[n as i32, 1]);
            mlxcel_core::multiply(&g, &wa)
        };

        let tl = gather(&idx_tl, &w_tl);
        let tr = gather(&idx_tr, &w_tr);
        let bl = gather(&idx_bl, &w_bl);
        let br = gather(&idx_br, &w_br);

        let a = mlxcel_core::add(&tl, &tr);
        let b = mlxcel_core::add(&bl, &br);
        mlxcel_core::add(&a, &b)
    }
}

// VisionRotaryEmbedding: `1 / theta^(2i/dim)` frequency table.
struct VisionRotaryEmbedding {
    dim: usize,
    theta: f32,
}

impl VisionRotaryEmbedding {
    fn new(dim: usize) -> Self {
        Self {
            dim,
            theta: 10000.0,
        }
    }

    fn forward(&self, seqlen: i32) -> UniquePtr<MlxArray> {
        let half_dim = self.dim / 2;
        let dim_f = self.dim as f32;
        let mut inv = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            inv.push(1.0 / self.theta.powf((2 * i) as f32 / dim_f));
        }
        let inv_freq = mlxcel_core::from_slice_f32(&inv, &[half_dim as i32]);
        let seq = mlxcel_core::arange_i32(0, seqlen, 1);
        let seq = mlxcel_core::astype(&seq, mlxcel_core::dtype::FLOAT32);
        mlxcel_core::outer(&seq, &inv_freq)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttentionDispatch {
    Single,
    Uniform { segment_len: i32 },
    Bucketed,
    Sequential,
}

// Fused-QKV attention over packed variable-length sequences.
struct Attention {
    qkv: UnifiedLinear,
    out_proj: UnifiedLinear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl Attention {
    fn from_weights(
        weights: &WeightMap,
        config: &PaddleOcrVisionConfig,
        prefix: &str,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let qkv =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.self_attn.qkv"), gs, bits)?;
        let out_proj = UnifiedLinear::from_weights(
            weights,
            &format!("{prefix}.self_attn.out_proj"),
            gs,
            bits,
        )?;
        let head_dim = (config.hidden_size / config.num_attention_heads) as i32;
        Ok(Self {
            qkv,
            out_proj,
            num_heads: config.num_attention_heads as i32,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn dispatch_for(cu_seqlens: &[i32]) -> AttentionDispatch {
        let num_segments = cu_seqlens.len().saturating_sub(1);
        if num_segments <= 1 {
            return AttentionDispatch::Single;
        }

        let mut counts = BTreeMap::<i32, usize>::new();
        let mut first_len = None;
        let mut uniform = true;
        for window in cu_seqlens.windows(2) {
            let len = window[1] - window[0];
            if first_len.is_none() {
                first_len = Some(len);
            } else if Some(len) != first_len {
                uniform = false;
            }
            *counts.entry(len).or_insert(0) += 1;
        }

        if uniform {
            AttentionDispatch::Uniform {
                segment_len: first_len.unwrap_or(0),
            }
        } else if counts.len() < num_segments {
            AttentionDispatch::Bucketed
        } else {
            AttentionDispatch::Sequential
        }
    }

    fn attend(&self, q: &MlxArray, k: &MlxArray, v: &MlxArray) -> UniquePtr<MlxArray> {
        unsafe {
            mlxcel_core::layers::attention_from_ptr(q, k, v, self.scale, std::ptr::null(), 0.0, 0)
        }
    }

    fn slice_attn_segment(&self, x: &MlxArray, start: i32, end: i32) -> UniquePtr<MlxArray> {
        mlxcel_core::slice(
            x,
            &[0, 0, start, 0],
            &[1, self.num_heads, end, self.head_dim],
        )
    }

    fn attend_sequential(
        &self,
        q: &MlxArray,
        k: &MlxArray,
        v: &MlxArray,
        cu_seqlens: &[i32],
    ) -> UniquePtr<MlxArray> {
        let num_segments = cu_seqlens.len() - 1;
        let mut outs = Vec::with_capacity(num_segments);
        for seg in 0..num_segments {
            let start = cu_seqlens[seg];
            let end = cu_seqlens[seg + 1];
            let q_seg = self.slice_attn_segment(q, start, end);
            let k_seg = self.slice_attn_segment(k, start, end);
            let v_seg = self.slice_attn_segment(v, start, end);
            outs.push(self.attend(&q_seg, &k_seg, &v_seg));
        }

        if outs.len() == 1 {
            outs.into_iter().next().unwrap()
        } else {
            concat_many(&outs, 2)
        }
    }

    fn attend_uniform(
        &self,
        q: &MlxArray,
        k: &MlxArray,
        v: &MlxArray,
        seq_len: i32,
        num_segments: i32,
        segment_len: i32,
    ) -> UniquePtr<MlxArray> {
        let pack = |x: &MlxArray| {
            let x = mlxcel_core::squeeze_axis(x, 0);
            let x = mlxcel_core::reshape(
                &x,
                &[self.num_heads, num_segments, segment_len, self.head_dim],
            );
            mlxcel_core::transpose_axes(&x, &[1, 0, 2, 3])
        };

        let q_batched = pack(q);
        let k_batched = pack(k);
        let v_batched = pack(v);
        let out = self.attend(&q_batched, &k_batched, &v_batched);

        // [segments, heads, segment_len, head_dim] -> [1, heads, seq, head_dim].
        let out = mlxcel_core::transpose_axes(&out, &[1, 0, 2, 3]);
        let out = mlxcel_core::reshape(&out, &[self.num_heads, seq_len, self.head_dim]);
        mlxcel_core::expand_dims(&out, 0)
    }

    fn stack_bucket_segments(
        &self,
        x: &MlxArray,
        entries: &[(usize, i32, i32)],
    ) -> UniquePtr<MlxArray> {
        if entries.len() == 1 {
            let (_, start, end) = entries[0];
            return self.slice_attn_segment(x, start, end);
        }

        let parts: Vec<UniquePtr<MlxArray>> = entries
            .iter()
            .map(|&(_, start, end)| {
                let segment = self.slice_attn_segment(x, start, end);
                mlxcel_core::squeeze_axis(&segment, 0)
            })
            .collect();
        mlxcel_core::stack_owned(&parts, 0)
    }

    fn attend_bucketed(
        &self,
        q: &MlxArray,
        k: &MlxArray,
        v: &MlxArray,
        cu_seqlens: &[i32],
    ) -> UniquePtr<MlxArray> {
        let num_segments = cu_seqlens.len() - 1;
        let mut buckets = BTreeMap::<i32, Vec<(usize, i32, i32)>>::new();
        for seg in 0..num_segments {
            let start = cu_seqlens[seg];
            let end = cu_seqlens[seg + 1];
            buckets
                .entry(end - start)
                .or_default()
                .push((seg, start, end));
        }

        let mut per_segment: Vec<Option<UniquePtr<MlxArray>>> =
            std::iter::repeat_with(|| None).take(num_segments).collect();

        for (segment_len, entries) in buckets {
            let q_batched = self.stack_bucket_segments(q, &entries);
            let k_batched = self.stack_bucket_segments(k, &entries);
            let v_batched = self.stack_bucket_segments(v, &entries);
            let bucket_out = self.attend(&q_batched, &k_batched, &v_batched);

            for (batch_idx, &(seg, _, _)) in entries.iter().enumerate() {
                let batch_idx = batch_idx as i32;
                per_segment[seg] = Some(mlxcel_core::slice(
                    &bucket_out,
                    &[batch_idx, 0, 0, 0],
                    &[batch_idx + 1, self.num_heads, segment_len, self.head_dim],
                ));
            }
        }

        let ordered: Vec<UniquePtr<MlxArray>> = per_segment
            .into_iter()
            .map(|segment| segment.expect("attention bucket produced every segment"))
            .collect();
        concat_many(&ordered, 2)
    }

    fn forward(
        &self,
        x: &MlxArray,
        cu_seqlens: &[i32],
        rotary_pos_emb: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let seq_len = shape[0];

        let qkv = self.qkv.forward(x);
        let qkv = mlxcel_core::reshape(&qkv, &[seq_len, 3, self.num_heads, self.head_dim]);
        let qkv = mlxcel_core::transpose_axes(&qkv, &[1, 0, 2, 3]);

        let q = mlxcel_core::slice(
            &qkv,
            &[0, 0, 0, 0],
            &[1, seq_len, self.num_heads, self.head_dim],
        );
        let k = mlxcel_core::slice(
            &qkv,
            &[1, 0, 0, 0],
            &[2, seq_len, self.num_heads, self.head_dim],
        );
        let v = mlxcel_core::slice(
            &qkv,
            &[2, 0, 0, 0],
            &[3, seq_len, self.num_heads, self.head_dim],
        );
        let q = mlxcel_core::squeeze_axis(&q, 0);
        let k = mlxcel_core::squeeze_axis(&k, 0);
        let v = mlxcel_core::squeeze_axis(&v, 0);

        let q = apply_rotary_pos_emb_vision(&q, rotary_pos_emb);
        let k = apply_rotary_pos_emb_vision(&k, rotary_pos_emb);

        let q = mlxcel_core::expand_dims(&mlxcel_core::transpose_axes(&q, &[1, 0, 2]), 0);
        let k = mlxcel_core::expand_dims(&mlxcel_core::transpose_axes(&k, &[1, 0, 2]), 0);
        let v = mlxcel_core::expand_dims(&mlxcel_core::transpose_axes(&v, &[1, 0, 2]), 0);

        let output = match Self::dispatch_for(cu_seqlens) {
            AttentionDispatch::Single | AttentionDispatch::Sequential => {
                self.attend_sequential(&q, &k, &v, cu_seqlens)
            }
            AttentionDispatch::Uniform { segment_len } => self.attend_uniform(
                &q,
                &k,
                &v,
                seq_len,
                (cu_seqlens.len() - 1) as i32,
                segment_len,
            ),
            AttentionDispatch::Bucketed => self.attend_bucketed(&q, &k, &v, cu_seqlens),
        };

        let output = mlxcel_core::squeeze_axis(&output, 0);
        let output = mlxcel_core::transpose_axes(&output, &[1, 0, 2]);
        let output = mlxcel_core::reshape(&output, &[seq_len, -1]);
        self.out_proj.forward(&output)
    }
}

// MLP with exact (erf) GELU.
struct MLP {
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
}

impl MLP {
    fn from_weights(weights: &WeightMap, prefix: &str, gs: i32, bits: i32) -> Result<Self, String> {
        Ok(Self {
            fc1: UnifiedLinear::from_weights(weights, &format!("{prefix}.mlp.fc1"), gs, bits)?,
            fc2: UnifiedLinear::from_weights(weights, &format!("{prefix}.mlp.fc2"), gs, bits)?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.fc1.forward(x);
        let h = mlxcel_core::gelu(&h);
        self.fc2.forward(&h)
    }
}

struct EncoderLayer {
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
    attn: Attention,
    mlp: MLP,
}

impl EncoderLayer {
    fn from_weights(
        weights: &WeightMap,
        config: &PaddleOcrVisionConfig,
        prefix: &str,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            layer_norm1: load_layer_norm(weights, &format!("{prefix}.layer_norm1"), 1e-6)?,
            layer_norm2: load_layer_norm(weights, &format!("{prefix}.layer_norm2"), 1e-6)?,
            attn: Attention::from_weights(weights, config, prefix, gs, bits)?,
            mlp: MLP::from_weights(weights, prefix, gs, bits)?,
        })
    }

    fn forward(&self, x: &MlxArray, cu_seqlens: &[i32], rotary: &MlxArray) -> UniquePtr<MlxArray> {
        let attn_out = self
            .attn
            .forward(&self.layer_norm1.forward(x), cu_seqlens, rotary);
        let h = mlxcel_core::add(x, &attn_out);
        let mlp_out = self.mlp.forward(&self.layer_norm2.forward(&h));
        mlxcel_core::add(&h, &mlp_out)
    }
}

/// PaddleOCR-VL vision encoder (patch embed + transformer + post-layernorm).
pub struct PaddleOcrVisionEncoder {
    patch_embed: PatchEmbed,
    position_embedding: PositionEmbedding,
    rotary_pos_emb: VisionRotaryEmbedding,
    layers: Vec<EncoderLayer>,
    post_layernorm: LayerNorm,
}

impl PaddleOcrVisionEncoder {
    pub fn from_weights(
        weights: &WeightMap,
        config: &PaddleOcrVisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = config.quant_group_size;
        let bits = config.quant_bits;

        let patch_embed = PatchEmbed::from_weights(
            weights,
            config,
            &format!("{prefix}.embeddings.patch_embedding"),
        )?;
        let position_embedding = PositionEmbedding::from_weights(
            weights,
            &format!("{prefix}.embeddings.position_embedding"),
        )?;

        let head_dim = config.hidden_size / config.num_attention_heads;
        let rotary_pos_emb = VisionRotaryEmbedding::new(head_dim / 2);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(EncoderLayer::from_weights(
                weights,
                config,
                &format!("{prefix}.layers.{i}"),
                gs,
                bits,
            )?);
        }

        let post_layernorm = load_layer_norm(
            weights,
            &format!("{prefix}.post_layernorm"),
            config.layer_norm_eps,
        )?;

        Ok(Self {
            patch_embed,
            position_embedding,
            rotary_pos_emb,
            layers,
            post_layernorm,
        })
    }

    /// Row-major 2D position ids per image (NO spatial-merge reordering):
    /// `hids = (arange(t*h*w) % (h*w)) // w`, `wids = ... % w`.
    fn rot_pos_emb(&self, grid_thw: &[(i32, i32, i32)]) -> UniquePtr<MlxArray> {
        let mut h_ids: Vec<i32> = Vec::new();
        let mut w_ids: Vec<i32> = Vec::new();
        let mut max_grid = 0i32;
        for &(t, h, w) in grid_thw {
            max_grid = max_grid.max(h).max(w);
            for p in 0..(t * h * w) {
                let pid = p % (h * w);
                h_ids.push(pid / w);
                w_ids.push(pid % w);
            }
        }
        let total = h_ids.len() as i32;

        // Interleave [h_id, w_id] per token -> flat lookup indices [total*2].
        let mut pos_flat = Vec::with_capacity((total * 2) as usize);
        for k in 0..total as usize {
            pos_flat.push(h_ids[k]);
            pos_flat.push(w_ids[k]);
        }
        let pos_arr = mlxcel_core::from_slice_i32(&pos_flat, &[total * 2]);

        let table = self.rotary_pos_emb.forward(max_grid);
        let all_freqs = mlxcel_core::take(&table, &pos_arr, 0);
        let freq_shape = mlxcel_core::array_shape(&all_freqs);
        let half = freq_shape[1];
        let reshaped = mlxcel_core::reshape(&all_freqs, &[total, 2, half]);
        mlxcel_core::reshape(&reshaped, &[total, 2 * half])
    }

    fn compute_cu_seqlens(grid_thw: &[(i32, i32, i32)]) -> Vec<i32> {
        let mut cu = vec![0i32];
        let mut cumulative = 0i32;
        for &(t, h, w) in grid_thw {
            let per_frame = h * w;
            for _ in 0..t {
                cumulative += per_frame;
                cu.push(cumulative);
            }
        }
        cu
    }

    /// pixel_values: `[total_patches, C * patch * patch]`.
    /// Returns per-token hidden states `[total_patches, embed]` (pre-projector).
    pub fn forward_with_grid(
        &self,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> VisionEncoderOutput {
        let mut h = self.patch_embed.forward(pixel_values);

        // Add per-image interpolated learned position embeddings. Cache repeated
        // dynamic-resolution grids in multi-image OCR batches so identical page
        // sizes reuse the same interpolation graph instead of rebuilding it.
        let mut pos_cache = BTreeMap::<(i32, i32), UniquePtr<MlxArray>>::new();
        let mut pos_parts: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(grid_thw.len());
        for &(t, gh, gw) in grid_thw {
            let per = pos_cache
                .entry((gh, gw))
                .or_insert_with(|| self.position_embedding.interpolate(gh, gw));
            for _ in 0..t {
                pos_parts.push(mlxcel_core::copy(per.as_ref().unwrap()));
            }
        }
        let pos = if pos_parts.len() == 1 {
            pos_parts.into_iter().next().unwrap()
        } else {
            concat_many(&pos_parts, 0)
        };
        h = mlxcel_core::add(&h, &pos);

        let rotary = self.rot_pos_emb(grid_thw);
        let cu_seqlens = Self::compute_cu_seqlens(grid_thw);

        for layer in &self.layers {
            h = layer.forward(&h, &cu_seqlens, &rotary);
        }

        let h = self.post_layernorm.forward(&h);
        VisionEncoderOutput { hidden_states: h }
    }
}

/// VisionEncoder trait: PaddleOCR-VL requires grid_thw, so the plain entry
/// point is unsupported (use `forward_with_grid`).
impl super::VisionEncoder for PaddleOcrVisionEncoder {
    fn forward(&self, _pixel_values: &MlxArray) -> VisionEncoderOutput {
        panic!("PaddleOCR-VL vision encoder requires grid_thw; use forward_with_grid() instead");
    }
}

#[cfg(test)]
mod tests {
    use super::{Attention, AttentionDispatch};
    use mlxcel_core::layers::{Linear, UnifiedLinear};
    use mlxcel_core::{MlxArray, UniquePtr};

    fn dummy_linear() -> UnifiedLinear {
        UnifiedLinear::Regular(Linear::new(
            mlxcel_core::from_slice_f32(&[0.0], &[1, 1]),
            None,
        ))
    }

    fn tiny_attention() -> Attention {
        Attention {
            qkv: dummy_linear(),
            out_proj: dummy_linear(),
            num_heads: 2,
            head_dim: 4,
            scale: 0.5,
        }
    }

    fn varied(shape: &[i32], offset: i32) -> UniquePtr<MlxArray> {
        let n: i32 = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| ((((i + offset) % 19) as f32) - 9.0) * 0.025)
            .collect();
        mlxcel_core::from_slice_f32(&data, shape)
    }

    fn assert_allclose(actual: &MlxArray, expected: &MlxArray) {
        let close = mlxcel_core::allclose(actual, expected, 1e-4, 1e-4);
        assert!(
            mlxcel_core::item_bool(&close),
            "fast attention path diverged from sequential packed reference"
        );
    }

    #[test]
    fn attention_dispatch_uses_single_for_one_segment() {
        assert_eq!(Attention::dispatch_for(&[0, 4]), AttentionDispatch::Single);
    }

    #[test]
    fn attention_dispatch_batches_uniform_segments() {
        assert_eq!(
            Attention::dispatch_for(&[0, 4, 8, 12]),
            AttentionDispatch::Uniform { segment_len: 4 }
        );
    }

    #[test]
    fn attention_dispatch_buckets_repeated_variable_segments() {
        assert_eq!(
            Attention::dispatch_for(&[0, 4, 10, 14]),
            AttentionDispatch::Bucketed
        );
    }

    #[test]
    fn attention_dispatch_keeps_unique_variable_segments_sequential() {
        assert_eq!(
            Attention::dispatch_for(&[0, 4, 10, 18]),
            AttentionDispatch::Sequential
        );
    }

    #[test]
    fn uniform_attention_fast_path_matches_sequential_segments() {
        let attention = tiny_attention();
        let q = varied(&[1, 2, 8, 4], 0);
        let k = varied(&[1, 2, 8, 4], 7);
        let v = varied(&[1, 2, 8, 4], 13);
        let cu = [0, 4, 8];

        let expected = attention.attend_sequential(&q, &k, &v, &cu);
        let actual = attention.attend_uniform(&q, &k, &v, 8, 2, 4);
        assert_allclose(&actual, &expected);
    }

    #[test]
    fn bucketed_attention_fast_path_matches_sequential_segments() {
        let attention = tiny_attention();
        let q = varied(&[1, 2, 14, 4], 0);
        let k = varied(&[1, 2, 14, 4], 5);
        let v = varied(&[1, 2, 14, 4], 11);
        let cu = [0, 4, 10, 14];

        let expected = attention.attend_sequential(&q, &k, &v, &cu);
        let actual = attention.attend_bucketed(&q, &k, &v, &cu);
        assert_allclose(&actual, &expected);
    }
}
