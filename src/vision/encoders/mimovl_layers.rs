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

//! MiMo-VL tower layers: patch embedding, windowed/sink attention, SiLU MLP,
//! residual blocks and the bias-free LayerNorm patch merger.

use super::MimovlVisionConfig;
use crate::vision::encoders::qwen2_vl::{apply_rotary_pos_emb_vision, concat_many};
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

struct VisionRMSNorm {
    weight: UniquePtr<MlxArray>,
    eps: f32,
}

impl VisionRMSNorm {
    fn from_weights(weights: &WeightMap, prefix: &str, eps: f32) -> Result<Self, String> {
        let key = format!("{prefix}.weight");
        let weight = weights
            .get(&key)
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {key}"))?;
        Ok(Self { weight, eps })
    }

    /// Reduce in float32, as Qwen2.5-VL's tower does: f16 sums over 1280
    /// channels saturate at 65504 and erase whole token rows (#1596).
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let x_dtype = mlxcel_core::array_dtype(x);
        if mlxcel_core::array_dtype(&self.weight) == x_dtype {
            mlxcel_core::fast_rms_norm(x, &self.weight, self.eps)
        } else {
            let w = mlxcel_core::astype(&self.weight, x_dtype);
            mlxcel_core::fast_rms_norm(x, &w, self.eps)
        }
    }
}

pub(super) struct PatchEmbed {
    /// `[out, T * C * P * P]`, temporal-major to match the processor rows.
    weight: UniquePtr<MlxArray>,
    in_features: usize,
    temporal_patch_size: usize,
}

impl PatchEmbed {
    pub(super) fn from_weights(
        weights: &WeightMap,
        config: &MimovlVisionConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let key = format!("{prefix}.patch_embed.proj.weight");
        let weight = weights
            .get(&key)
            .ok_or_else(|| format!("Weight not found: {key}"))?;
        let shape = mlxcel_core::array_shape(weight);
        let c = config.in_channels as i32;
        let t = config.temporal_patch_size as i32;
        let p = config.patch_size as i32;
        let out = config.hidden_size as i32;
        let in_features =
            config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size;

        // Two source layouts reach here:
        //   * 2-D `[out, C * T * P * P]` or 5-D `[out, C, T, P, P]`: the
        //     PyTorch / mlx-vlm sanitized order (channel-major per patch),
        //   * 5-D `[out, T, P, P, C]`: MLX's Conv3d channel-last order.
        // The processor emits `T, C, P, P` per patch, so either source is
        // reordered to temporal-major once, here.
        let reordered = match shape.len() {
            2 => {
                if shape != [out, (c * t * p * p)] {
                    return Err(format!(
                        "MiMo-VL {key} has shape {shape:?}; expected \
                         [{out}, {}]",
                        c * t * p * p
                    ));
                }
                let shaped = mlxcel_core::reshape(weight, &[out, c, t, p, p]);
                let transposed = mlxcel_core::transpose_axes(&shaped, &[0, 2, 1, 3, 4]);
                mlxcel_core::reshape(&transposed, &[out, (c * t * p * p)])
            }
            5 if shape[1] == c => {
                let transposed = mlxcel_core::transpose_axes(weight, &[0, 2, 1, 3, 4]);
                mlxcel_core::reshape(&transposed, &[out, (c * t * p * p)])
            }
            5 if shape[4] == c => {
                let transposed = mlxcel_core::transpose_axes(weight, &[0, 1, 4, 2, 3]);
                mlxcel_core::reshape(&transposed, &[out, (c * t * p * p)])
            }
            _ => {
                return Err(format!(
                    "MiMo-VL {key} has unsupported shape {shape:?}; expected 2-D \
                     [{out}, {}] or a 5-D Conv3d weight",
                    c * t * p * p
                ));
            }
        };

        Ok(Self {
            weight: reordered,
            in_features,
            temporal_patch_size: config.temporal_patch_size,
        })
    }

    pub(super) fn forward(&self, pixel_values: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let shape = mlxcel_core::array_shape(pixel_values);
        if shape.len() != 2 || shape[1] <= 0 {
            return Err(format!(
                "MiMo-VL pixel_values must be [rows, C*P*P], got {shape:?}"
            ));
        }
        let per_slice = shape[1] as usize;
        let expected_per_slice = self.in_features / self.temporal_patch_size;
        if per_slice != expected_per_slice {
            return Err(format!(
                "MiMo-VL pixel_values row width {per_slice} does not match the patch embedding's \
                 {expected_per_slice}"
            ));
        }
        let rows = shape[0] as usize;
        if !rows.is_multiple_of(self.temporal_patch_size) {
            return Err(format!(
                "MiMo-VL pixel_values has {rows} rows, which is not a multiple of \
                 temporal_patch_size {}",
                self.temporal_patch_size
            ));
        }
        let patches = (rows / self.temporal_patch_size) as i32;
        let flat = mlxcel_core::reshape(pixel_values, &[patches, self.in_features as i32]);
        let weight_t = mlxcel_core::transpose(&self.weight);
        Ok(mlxcel_core::matmul(&flat, &weight_t))
    }
}

/// LayerNorm with weight only (`bias=False`), as the reference merger uses.
struct LayerNormNoBias {
    weight: UniquePtr<MlxArray>,
    bias: UniquePtr<MlxArray>,
    eps: f32,
}

impl LayerNormNoBias {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        dim: usize,
        eps: f32,
    ) -> Result<Self, String> {
        let key = format!("{prefix}.weight");
        let weight = weights
            .get(&key)
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {key}"))?;
        let bias = mlxcel_core::zeros(&[dim as i32], mlxcel_core::array_dtype(&weight));
        Ok(Self { weight, bias, eps })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let dtype = mlxcel_core::array_dtype(x);
        let weight = mlxcel_core::astype(&self.weight, dtype);
        let bias = mlxcel_core::astype(&self.bias, dtype);
        mlxcel_core::layer_norm(x, &weight, &bias, self.eps)
    }
}

struct VisionAttention {
    qkv: UnifiedLinear,
    proj: UnifiedLinear,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    window_size: i32,
    sinks: Option<UniquePtr<MlxArray>>,
}

impl VisionAttention {
    pub(super) fn from_weights(
        weights: &WeightMap,
        config: &MimovlVisionConfig,
        prefix: &str,
        use_sinks: bool,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let head_dim = i32::try_from(config.qk_channels)
            .map_err(|_| "MiMo-VL qk_channels does not fit in i32".to_string())?;
        let sinks = if use_sinks {
            let key = format!("{prefix}.attn.sinks");
            Some(
                weights
                    .get(&key)
                    .map(|w| mlxcel_core::copy(w))
                    .ok_or_else(|| {
                        format!(
                            "MiMo-VL {key} is required: this block runs windowed attention with \
                             per-head sinks"
                        )
                    })?,
            )
        } else {
            None
        };
        Ok(Self {
            qkv: UnifiedLinear::from_weights(weights, &format!("{prefix}.attn.qkv"), gs, bits)?,
            proj: UnifiedLinear::from_weights(weights, &format!("{prefix}.attn.proj"), gs, bits)?,
            num_heads: config.num_heads as i32,
            num_kv_heads: config.num_key_value_heads as i32,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            window_size: i32::try_from(config.visual_token_window_size).unwrap_or(0),
            sinks,
        })
    }

    /// Build the additive attention mask for a segment of `n` tokens.
    ///
    /// The window is `|i - j| > window_size`; a sink is added to the logit of
    /// key 0 for every query, exactly as upstream does.
    fn mask(&self, n: i32, full_attn: bool, dtype: i32) -> Option<UniquePtr<MlxArray>> {
        let mut mask: Option<UniquePtr<MlxArray>> = None;
        if !full_attn && self.window_size > 0 {
            let idx = mlxcel_core::astype(
                &mlxcel_core::arange_i32(0, n, 1),
                mlxcel_core::dtype::FLOAT32,
            );
            let row = mlxcel_core::expand_dims(&idx, 1);
            let col = mlxcel_core::expand_dims(&idx, 0);
            let diff = mlxcel_core::subtract(&row, &col);
            let outside = mlxcel_core::greater(
                &mlxcel_core::abs(&diff),
                &mlxcel_core::full_f32(&[1], self.window_size as f32, mlxcel_core::dtype::FLOAT32),
            );
            let neg_inf =
                mlxcel_core::full_f32(&[n, n], f32::NEG_INFINITY, mlxcel_core::dtype::FLOAT32);
            let zero = mlxcel_core::full_f32(&[n, n], 0.0, mlxcel_core::dtype::FLOAT32);
            let banded = mlxcel_core::where_cond(&outside, &neg_inf, &zero);
            mask = Some(mlxcel_core::expand_dims(
                &mlxcel_core::expand_dims(&banded, 0),
                0,
            ));
        }
        if let Some(sinks) = &self.sinks {
            let sinks = mlxcel_core::astype(sinks, mlxcel_core::dtype::FLOAT32);
            let sink_col = mlxcel_core::reshape(&sinks, &[self.num_heads, 1]);
            let combined = if n > 1 {
                let rest = mlxcel_core::full_f32(
                    &[self.num_heads, n - 1],
                    0.0,
                    mlxcel_core::dtype::FLOAT32,
                );
                mlxcel_core::concatenate(&sink_col, &rest, 1)
            } else {
                sink_col
            };
            let sink_mask = mlxcel_core::reshape(&combined, &[1, self.num_heads, 1, n]);
            mask = Some(match mask {
                Some(banded) => mlxcel_core::add(&banded, &sink_mask),
                None => sink_mask,
            });
        }
        mask.map(|mask| mlxcel_core::astype(&mask, dtype))
    }

    fn attend_segment(
        &self,
        q: &MlxArray,
        k: &MlxArray,
        v: &MlxArray,
        start: i32,
        end: i32,
        full_attn: bool,
    ) -> UniquePtr<MlxArray> {
        // [1, H, n, D] per segment.
        let q_seg = mlxcel_core::slice(
            q,
            &[0, 0, start, 0],
            &[1, self.num_heads, end, self.head_dim],
        );
        let k_seg = mlxcel_core::slice(
            k,
            &[0, 0, start, 0],
            &[1, self.num_heads, end, self.head_dim],
        );
        let v_seg = mlxcel_core::slice(
            v,
            &[0, 0, start, 0],
            &[1, self.num_heads, end, self.head_dim],
        );
        let mask = self.mask(end - start, full_attn, mlxcel_core::array_dtype(&q_seg));
        let mask_ptr = mask
            .as_ref()
            .map(|m| m.as_ref().expect("mask array") as *const MlxArray)
            .unwrap_or(std::ptr::null());
        // SAFETY: the mask pointer is either null or borrowed from `mask`
        // for the duration of this call.
        unsafe {
            mlxcel_core::layers::attention_from_ptr(
                &q_seg, &k_seg, &v_seg, self.scale, mask_ptr, 0.0, 0,
            )
        }
    }

    pub(super) fn forward(
        &self,
        x: &MlxArray,
        cu_seqlens: &[i32],
        freqs: &MlxArray,
        full_attn: bool,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let n = shape[0];
        let qkv = self.qkv.forward(x);
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;

        let q = mlxcel_core::slice(&qkv, &[0, 0], &[n, q_dim]);
        let k = mlxcel_core::slice(&qkv, &[0, q_dim], &[n, q_dim + kv_dim]);
        let v = mlxcel_core::slice(&qkv, &[0, q_dim + kv_dim], &[n, q_dim + 2 * kv_dim]);
        let q = mlxcel_core::reshape(&q, &[n, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[n, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[n, self.num_kv_heads, self.head_dim]);
        let q = apply_rotary_pos_emb_vision(&q, freqs);
        let k = apply_rotary_pos_emb_vision(&k, freqs);

        let repeats = self.num_heads / self.num_kv_heads;
        let (k, v) = if repeats > 1 {
            (
                mlxcel_core::repeat(&k, repeats, 1),
                mlxcel_core::repeat(&v, repeats, 1),
            )
        } else {
            (k, v)
        };

        let q = mlxcel_core::expand_dims(&mlxcel_core::transpose_axes(&q, &[1, 0, 2]), 0);
        let k = mlxcel_core::expand_dims(&mlxcel_core::transpose_axes(&k, &[1, 0, 2]), 0);
        let v = mlxcel_core::expand_dims(&mlxcel_core::transpose_axes(&v, &[1, 0, 2]), 0);

        let mut segments = Vec::with_capacity(cu_seqlens.len().saturating_sub(1));
        for pair in cu_seqlens.windows(2) {
            segments.push(self.attend_segment(&q, &k, &v, pair[0], pair[1], full_attn));
        }
        let output = if segments.len() == 1 {
            segments.pop().expect("one segment")
        } else {
            concat_many(&segments, 2)
        };
        let output = mlxcel_core::squeeze_axis(&output, 0);
        let output = mlxcel_core::transpose_axes(&output, &[1, 0, 2]);
        let output = mlxcel_core::reshape(&output, &[n, -1]);
        self.proj.forward(&output)
    }
}

struct VisionMlp {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
}

impl VisionMlp {
    fn from_weights(weights: &WeightMap, prefix: &str, gs: i32, bits: i32) -> Result<Self, String> {
        Ok(Self {
            gate_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.mlp.gate_proj"),
                gs,
                bits,
            )?,
            up_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.mlp.up_proj"),
                gs,
                bits,
            )?,
            down_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.mlp.down_proj"),
                gs,
                bits,
            )?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = mlxcel_core::silu(&self.gate_proj.forward(x));
        let up = self.up_proj.forward(x);
        self.down_proj.forward(&mlxcel_core::multiply(&gate, &up))
    }
}

pub(super) struct VisionBlock {
    norm1: VisionRMSNorm,
    norm2: VisionRMSNorm,
    attn: VisionAttention,
    mlp: VisionMlp,
}

impl VisionBlock {
    pub(super) fn from_weights(
        weights: &WeightMap,
        config: &MimovlVisionConfig,
        prefix: &str,
        use_sinks: bool,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            norm1: VisionRMSNorm::from_weights(
                weights,
                &format!("{prefix}.norm1"),
                config.rms_norm_eps,
            )?,
            norm2: VisionRMSNorm::from_weights(
                weights,
                &format!("{prefix}.norm2"),
                config.rms_norm_eps,
            )?,
            attn: VisionAttention::from_weights(weights, config, prefix, use_sinks, gs, bits)?,
            mlp: VisionMlp::from_weights(weights, prefix, gs, bits)?,
        })
    }

    pub(super) fn forward(
        &self,
        x: &MlxArray,
        cu_seqlens: &[i32],
        freqs: &MlxArray,
        full_attn: bool,
    ) -> UniquePtr<MlxArray> {
        let normed = self.norm1.forward(x);
        let attn = self.attn.forward(&normed, cu_seqlens, freqs, full_attn);
        let h = mlxcel_core::add(x, &attn);
        let normed = self.norm2.forward(&h);
        mlxcel_core::add(&h, &self.mlp.forward(&normed))
    }
}

pub(super) struct PatchMerger {
    ln_q: LayerNormNoBias,
    mlp_0: UnifiedLinear,
    mlp_2: UnifiedLinear,
    merged_width: usize,
}

impl PatchMerger {
    pub(super) fn from_weights(
        weights: &WeightMap,
        config: &MimovlVisionConfig,
        prefix: &str,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let merged_width =
            config.hidden_size * config.spatial_merge_size * config.spatial_merge_size;
        Ok(Self {
            ln_q: LayerNormNoBias::from_weights(
                weights,
                &format!("{prefix}.ln_q"),
                config.hidden_size,
                1e-6,
            )?,
            mlp_0: UnifiedLinear::from_weights(weights, &format!("{prefix}.mlp.0"), gs, bits)?,
            mlp_2: UnifiedLinear::from_weights(weights, &format!("{prefix}.mlp.2"), gs, bits)?,
            merged_width,
        })
    }

    pub(super) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.ln_q.forward(x);
        let h = mlxcel_core::reshape(&h, &[-1, self.merged_width as i32]);
        let h = self.mlp_0.forward(&h);
        let h = mlxcel_core::gelu(&h);
        self.mlp_2.forward(&h)
    }
}
