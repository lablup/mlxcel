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

//! PLaMo 3 attention, MLP and decoder block.

use mlxcel_core::layers::{RMSNorm, UnifiedLinear};
use mlxcel_core::utils::{PrefillAttention, slice_axis};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::afmoe::{validate_norm, validate_projection};
use crate::models::gemma3::{Cache, CacheInterface};
use crate::models::plamo2::{
    PRE_MIXER_NORM_OFFSET, PRE_MLP_NORM_OFFSET, offset_rms_norm, post_mixer_norm_offset,
    post_mlp_norm_offset,
};
use crate::models::plamo3_config::ModelArgs;
use crate::models::rope_utils::RopeScalingKind;

/// `q_norm` / `k_norm` offset (`PLAMO3_Q_NORM_OFFSET`, `PLAMO3_K_NORM_OFFSET`).
pub(crate) const QK_NORM_OFFSET: f32 = 1.0;

/// The six PLaMo 3 norm offsets in `(name, offset)` form, pinned by
/// `plamo3_tests::every_norm_offset_is_folded_into_its_weight`.
#[cfg(test)]
pub(crate) fn norm_offsets() -> [(&'static str, f32); 6] {
    [
        ("pre_mixer_norm", PRE_MIXER_NORM_OFFSET),
        ("post_mixer_norm", post_mixer_norm_offset()),
        ("pre_mlp_norm", PRE_MLP_NORM_OFFSET),
        ("post_mlp_norm", post_mlp_norm_offset()),
        ("qk_norm", QK_NORM_OFFSET),
        ("norm", crate::models::plamo2::FINAL_NORM_OFFSET),
    ]
}

/// One layer's rotation: a plain base, or a frequency table (YaRN on the
/// full-attention layers) with its magnitude multiplier.
pub(crate) struct LayerRope {
    pub base: f32,
    pub scale: f32,
    pub freqs: Option<UniquePtr<MlxArray>>,
    /// YaRN `concentration` (`0.1 ln(factor) + 1`). The reference multiplies
    /// both cos and sin by it, which is Q and K each scaled before rotation.
    pub mscale: f32,
}

impl LayerRope {
    pub(crate) fn plain(base: f32) -> Self {
        Self {
            base,
            scale: 1.0,
            freqs: None,
            mscale: 1.0,
        }
    }

    /// Build from a resolved [`RopeScalingKind`] (already overlaid with any
    /// `--rope-scaling` request), checking the table width so a mismatch is a
    /// load error rather than an uncatchable MLX throw at the first token.
    pub(crate) fn from_kind(
        kind: RopeScalingKind,
        base: f32,
        head_dim: usize,
    ) -> Result<Self, String> {
        let scale = kind.scale();
        let mscale = kind.attn_scale();
        let freqs = match kind {
            RopeScalingKind::Yarn { freqs, .. } | RopeScalingKind::Llama3 { freqs } => Some(freqs),
            _ => None,
        };
        if let Some(table) = freqs.as_ref() {
            let shape = mlxcel_core::array_shape(table);
            if shape != [(head_dim / 2) as i32] {
                return Err(format!(
                    "PLaMo 3 rope frequency table has shape {shape:?}; head_dim {head_dim} \
                     needs [{}]",
                    head_dim / 2
                ));
            }
        }
        Ok(Self {
            base,
            scale,
            freqs,
            mscale,
        })
    }

    fn magnitude(&self, x: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        (self.mscale != 1.0).then(|| mlxcel_core::multiply_scalar(x, self.mscale))
    }

    pub(crate) fn rotate(&self, x: &MlxArray, dims: i32, offset: i32) -> UniquePtr<MlxArray> {
        let scaled = self.magnitude(x);
        let x = scaled.as_deref().unwrap_or(x);
        match self.freqs.as_ref() {
            Some(freqs) => {
                mlxcel_core::fast_rope_with_freqs(x, dims, false, self.scale, offset, freqs)
            }
            None => mlxcel_core::fast_rope(x, dims, false, self.base, self.scale, offset),
        }
    }

    fn rotate_batched(&self, x: &MlxArray, dims: i32, offsets: &[i32]) -> UniquePtr<MlxArray> {
        let scaled = self.magnitude(x);
        let x = scaled.as_deref().unwrap_or(x);
        match self.freqs.as_ref() {
            Some(freqs) => mlxcel_core::fast_rope_batched_with_freqs(
                x, dims, false, self.scale, offsets, freqs,
            ),
            None => mlxcel_core::fast_rope_batched(x, dims, false, self.base, self.scale, offsets),
        }
    }
}

/// PLaMo 3 attention: packed `qkv_proj`, per-head offset RMSNorm on Q and K
/// **before** RoPE, non-traditional RoPE over the full head, no biases.
pub struct Attention {
    pub qkv_proj: UnifiedLinear,
    pub o_proj: UnifiedLinear,
    pub q_norm: RMSNorm,
    pub k_norm: RMSNorm,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub(crate) rope: LayerRope,
}

impl Attention {
    pub(crate) fn prefill_attention(&self) -> PrefillAttention {
        PrefillAttention::new(
            self.num_heads as usize,
            self.num_kv_heads as usize,
            self.head_dim as usize,
        )
    }

    /// `[B, heads, L, head_dim]` Q, K, V with the Q/K norms applied (no RoPE).
    fn heads(&self, x: &MlxArray) -> [UniquePtr<MlxArray>; 3] {
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
        let qkv = self.qkv_proj.forward(x);
        let split = |start: i32, width: i32, heads: i32| {
            let part = slice_axis(&qkv, -1, start, start + width);
            let part = mlxcel_core::reshape(&part, &[b, l, heads, self.head_dim]);
            mlxcel_core::transpose_axes(&part, &[0, 2, 1, 3])
        };
        let q = split(0, q_dim, self.num_heads);
        let k = split(q_dim, kv_dim, self.num_kv_heads);
        let v = split(q_dim + kv_dim, kv_dim, self.num_kv_heads);
        [self.q_norm.forward(&q), self.k_norm.forward(&k), v]
    }

    /// Q and K after the norm and the rotation at `offset`, the order the
    /// reference applies them in (norm first, then RoPE).
    pub(crate) fn rotated_qk(
        &self,
        x: &MlxArray,
        offset: i32,
    ) -> (
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
        UniquePtr<MlxArray>,
    ) {
        let [q, k, v] = self.heads(x);
        (
            self.rope.rotate(&q, self.head_dim, offset),
            self.rope.rotate(&k, self.head_dim, offset),
            v,
        )
    }

    fn merge_heads(&self, out: &MlxArray, b: i32, l: i32) -> UniquePtr<MlxArray> {
        let out = mlxcel_core::transpose_axes(out, &[0, 2, 1, 3]);
        let out = mlxcel_core::reshape(&out, &[b, l, self.num_heads * self.head_dim]);
        self.o_proj.forward(&out)
    }

    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        cache: &mut dyn CacheInterface,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);
        let (q, k, v) = self.rotated_qk(x, cache.offset());
        let (keys, values) = cache.update_and_fetch(k, v);
        // Sliding layers always receive an explicit mask for a multi-token
        // forward, so a mask-less one is a full layer left to MLX's causal
        // mode (#3072).
        let out = if l > 1 && mask.is_none() {
            mlxcel_core::maskless_prefill_attention(&q, &keys, &values, self.scale, 0)
        } else {
            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            // SAFETY: `mask_ptr` is null or borrows `mask`, which outlives the call.
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    &q, &keys, &values, self.scale, mask_ptr, 0.0, 0,
                )
            }
        };
        self.merge_heads(&out, b, l)
    }

    pub(crate) fn forward_batched_decode(
        &self,
        x: &MlxArray,
        caches: &mut [&mut Cache],
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);
        let [q, k, v] = self.heads(x);
        let offsets: Vec<i32> = caches
            .iter_mut()
            .map(|cache| cache.as_interface().offset())
            .collect();
        let q = self.rope.rotate_batched(&q, self.head_dim, &offsets);
        let k = self.rope.rotate_batched(&k, self.head_dim, &offsets);
        let out = mlxcel_core::cache::attend_batched_rows(&q, &k, &v, caches, self.scale, None);
        self.merge_heads(&out, b, l)
    }

    pub(crate) fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        prefix: &str,
        rope: LayerRope,
    ) -> Result<Self, String> {
        let (gs, bits) = (args.group_size(), args.bits());
        let d = args.head_dim;
        let qkv_out = (args.num_attention_heads + 2 * args.num_key_value_heads) * d;
        let qkv = format!("{prefix}.qkv_proj");
        let o = format!("{prefix}.o_proj");
        validate_projection(weights, &qkv, qkv_out, args.hidden_size, gs, bits)?;
        validate_projection(
            weights,
            &o,
            args.hidden_size,
            args.num_attention_heads * d,
            gs,
            bits,
        )?;
        let norm = |leaf: &str| {
            let key = format!("{prefix}.{leaf}.weight");
            validate_norm(weights, &key, d)?;
            offset_rms_norm(weights, &key, args.rms_norm_eps, QK_NORM_OFFSET)
        };
        Ok(Self {
            qkv_proj: UnifiedLinear::from_weights(weights, &qkv, gs, bits)?,
            o_proj: UnifiedLinear::from_weights(weights, &o, gs, bits)?,
            q_norm: norm("q_norm")?,
            k_norm: norm("k_norm")?,
            num_heads: args.num_attention_heads as i32,
            num_kv_heads: args.num_key_value_heads as i32,
            head_dim: d as i32,
            scale: (d as f32).powf(-0.5),
            rope,
        })
    }
}

/// SwiGLU MLP over a packed `gate_up_proj` (`gate` first, then `up`).
pub struct Mlp {
    pub gate_up_proj: UnifiedLinear,
    pub down_proj: UnifiedLinear,
    intermediate_size: i32,
}

impl Mlp {
    pub(crate) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate_up = self.gate_up_proj.forward(x);
        let i = self.intermediate_size;
        let gate = slice_axis(&gate_up, -1, 0, i);
        let up = slice_axis(&gate_up, -1, i, 2 * i);
        self.down_proj
            .forward(&mlxcel_core::compiled_swiglu_activation(&gate, &up))
    }

    fn from_weights(weights: &WeightMap, args: &ModelArgs, prefix: &str) -> Result<Self, String> {
        let (gs, bits) = (args.group_size(), args.bits());
        let gate_up = format!("{prefix}.gate_up_proj");
        let down = format!("{prefix}.down_proj");
        let i = args.intermediate_size;
        validate_projection(weights, &gate_up, 2 * i, args.hidden_size, gs, bits)?;
        validate_projection(weights, &down, args.hidden_size, i, gs, bits)?;
        Ok(Self {
            gate_up_proj: UnifiedLinear::from_weights(weights, &gate_up, gs, bits)?,
            down_proj: UnifiedLinear::from_weights(weights, &down, gs, bits)?,
            intermediate_size: i as i32,
        })
    }
}

/// One PLaMo 3 block:
/// `h = x + post_mixer_norm(mixer(pre_mixer_norm(x)))`,
/// `out = h + post_mlp_norm(mlp(pre_mlp_norm(h)))`.
pub struct DecoderLayer {
    pub mixer: Attention,
    pub mlp: Mlp,
    pub pre_mixer_norm: RMSNorm,
    pub post_mixer_norm: RMSNorm,
    pub pre_mlp_norm: RMSNorm,
    pub post_mlp_norm: RMSNorm,
    pub is_full: bool,
}

impl DecoderLayer {
    fn mlp_half(&self, h: UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
        let mlp = self.mlp.forward(&self.pre_mlp_norm.forward(&h));
        mlxcel_core::add(&h, &self.post_mlp_norm.forward(&mlp))
    }

    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        cache: &mut dyn CacheInterface,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let attn = self
            .mixer
            .forward(&self.pre_mixer_norm.forward(x), cache, mask);
        self.mlp_half(mlxcel_core::add(x, &self.post_mixer_norm.forward(&attn)))
    }

    pub(crate) fn forward_batched_decode(
        &self,
        x: &MlxArray,
        rows: &mut [&mut Cache],
    ) -> UniquePtr<MlxArray> {
        let attn = self
            .mixer
            .forward_batched_decode(&self.pre_mixer_norm.forward(x), rows);
        self.mlp_half(mlxcel_core::add(x, &self.post_mixer_norm.forward(&attn)))
    }

    pub(crate) fn from_weights(
        weights: &WeightMap,
        args: &ModelArgs,
        layer_idx: usize,
        rope: LayerRope,
    ) -> Result<Self, String> {
        // The decoder is `model.layers` and its list `model.layers.layers`.
        let prefix = format!("model.layers.layers.{layer_idx}");
        let norm = |leaf: &str, offset: f32| {
            let key = format!("{prefix}.{leaf}.weight");
            validate_norm(weights, &key, args.hidden_size)?;
            offset_rms_norm(weights, &key, args.rms_norm_eps, offset)
        };
        Ok(Self {
            mixer: Attention::from_weights(weights, args, &format!("{prefix}.mixer"), rope)?,
            mlp: Mlp::from_weights(weights, args, &format!("{prefix}.mlp"))?,
            pre_mixer_norm: norm("pre_mixer_norm", PRE_MIXER_NORM_OFFSET)?,
            post_mixer_norm: norm("post_mixer_norm", post_mixer_norm_offset())?,
            pre_mlp_norm: norm("pre_mlp_norm", PRE_MLP_NORM_OFFSET)?,
            post_mlp_norm: norm("post_mlp_norm", post_mlp_norm_offset())?,
            is_full: args.is_full_layer(layer_idx),
        })
    }
}
