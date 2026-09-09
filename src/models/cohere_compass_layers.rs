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

//! Cohere Compass decoder internals: norm dispatch, per-layer-type RoPE,
//! attention with the sliding/full split, SwiGLU MLP, and the parallel block.
//!
//! Split out of `cohere_compass.rs` to keep each file inside the project's
//! per-file size budget; see that module's header for the architecture.

use crate::models::cohere_compass_config::{CompassTextConfig, LayerRopeSpec};
use crate::models::cohere_compass_rope::CompassMRoPE;
use crate::models::qwen3_vl::apply_multimodal_rotary_pos_emb;
use mlxcel_core::layers::{FusedQKVLinear, KVCache, LayerNorm, RMSNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

/// `norm_type` dispatch. The published checkpoints are LayerNorm; the config
/// key exists because the family also ships an RMSNorm variant.
pub(crate) enum CompassNorm {
    Layer(LayerNorm),
    Rms(RMSNorm),
}

impl CompassNorm {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &CompassTextConfig,
    ) -> Result<Self, String> {
        let weight = weights
            .get(&format!("{prefix}.weight"))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {prefix}.weight"))?;
        match config.norm_type.as_str() {
            "layer_norm" => {
                let bias = weights
                    .get(&format!("{prefix}.bias"))
                    .map(|w| mlxcel_core::copy(w));
                Ok(Self::Layer(LayerNorm::new(
                    weight,
                    bias,
                    config.layer_norm_eps,
                )))
            }
            "rms_norm" => Ok(Self::Rms(RMSNorm::new(weight, config.rms_norm_eps))),
            other => Err(format!(
                "cohere_compass: unsupported norm_type {other:?} (expected \"layer_norm\" or \
                 \"rms_norm\")"
            )),
        }
    }

    pub(crate) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Layer(n) => n.forward(x),
            Self::Rms(n) => n.forward(x),
        }
    }
}

pub(crate) struct CompassAttention {
    qkv_proj: FusedQKVLinear,
    o_proj: UnifiedLinear,
    /// `None` on a layer type whose `rope_parameters` entry is JSON `null`:
    /// those layers carry no positional encoding at all.
    rope: Option<CompassMRoPE>,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    /// `0` for a full-attention layer.
    window_size: i32,
}

impl CompassAttention {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &CompassTextConfig,
        prefix: &str,
        rope_spec: Option<LayerRopeSpec>,
        window_size: i32,
    ) -> Result<Self, String> {
        let head_dim = config.head_dim();
        let num_heads = config.num_attention_heads as i32;
        let num_kv_heads = config.num_kv_heads() as i32;
        let gs = config.group_size();
        let bits = config.bits();

        let qkv_proj = FusedQKVLinear::from_weights_separate(
            weights,
            &format!("{prefix}.self_attn"),
            gs,
            bits,
            num_heads,
            num_kv_heads,
            head_dim as i32,
        )?;
        let o_proj =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.self_attn.o_proj"), gs, bits)?;

        let rope = rope_spec
            .map(|spec| CompassMRoPE::new(head_dim, spec.theta, spec.mrope_section))
            .transpose()?;

        Ok(Self {
            qkv_proj,
            o_proj,
            rope,
            num_heads,
            num_kv_heads,
            head_dim: head_dim as i32,
            scale: (head_dim as f32).powf(-0.5),
            window_size,
        })
    }

    /// Project and reshape into `[B, heads, L, head_dim]`.
    fn project(&self, x: &MlxArray, b: i32, l: i32) -> [UniquePtr<MlxArray>; 3] {
        let (q, k, v) = self.qkv_proj.forward(x);
        let q = mlxcel_core::reshape(&q, &[b, l, self.num_heads, self.head_dim]);
        let k = mlxcel_core::reshape(&k, &[b, l, self.num_kv_heads, self.head_dim]);
        let v = mlxcel_core::reshape(&v, &[b, l, self.num_kv_heads, self.head_dim]);
        [
            mlxcel_core::transpose_axes(&q, &[0, 2, 1, 3]),
            mlxcel_core::transpose_axes(&k, &[0, 2, 1, 3]),
            mlxcel_core::transpose_axes(&v, &[0, 2, 1, 3]),
        ]
    }

    /// Shared tail: cache update, windowed SDPA, output projection.
    fn attend(
        &self,
        q: &MlxArray,
        k: UniquePtr<MlxArray>,
        v: UniquePtr<MlxArray>,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
        b: i32,
        l: i32,
    ) -> UniquePtr<MlxArray> {
        let (cache_k, cache_v) = cache.update_and_fetch(k, v);

        let attn_out = if l > 1 {
            // A sliding layer's prefill mask is clamped to `window_size` keys
            // once the prefix outgrows the window, while the dense `KVCache`
            // still hands back every live key. Slice K/V down to the mask's key
            // axis so the two agree; a full-width mask keeps every key. Same
            // handling as `cohere2.rs` (issues #408 / #419).
            let k_shape = mlxcel_core::array_shape(&cache_k);
            let k_len = k_shape[2];
            let mask_klen = mask
                .map(|m| *mlxcel_core::array_shape(m).last().unwrap_or(&k_len))
                .unwrap_or(k_len);
            let (k_used, v_used) = if self.window_size > 0 && k_len > mask_klen {
                let v_shape = mlxcel_core::array_shape(&cache_v);
                let start = k_len - mask_klen;
                (
                    Some(mlxcel_core::slice(
                        &cache_k,
                        &[0, 0, start, 0],
                        &[k_shape[0], k_shape[1], k_len, k_shape[3]],
                    )),
                    Some(mlxcel_core::slice(
                        &cache_v,
                        &[0, 0, start, 0],
                        &[v_shape[0], v_shape[1], k_len, v_shape[3]],
                    )),
                )
            } else {
                (None, None)
            };
            let k_ref: &MlxArray = k_used
                .as_ref()
                .map(|p| p.as_ref().unwrap())
                .unwrap_or_else(|| cache_k.as_ref().unwrap());
            let v_ref: &MlxArray = v_used
                .as_ref()
                .map(|p| p.as_ref().unwrap())
                .unwrap_or_else(|| cache_v.as_ref().unwrap());
            let mask_ptr = mask.map(|m| m as *const _).unwrap_or(std::ptr::null());
            unsafe {
                mlxcel_core::layers::attention_from_ptr(
                    q,
                    k_ref,
                    v_ref,
                    self.scale,
                    mask_ptr,
                    0.0,
                    self.window_size,
                )
            }
        } else {
            mlxcel_core::causal_attention(q, &cache_k, &cache_v, self.scale, 0.0, self.window_size)
        };

        let attn_out = mlxcel_core::transpose_axes(&attn_out, &[0, 2, 1, 3]);
        let attn_out = mlxcel_core::reshape(&attn_out, &[b, l, self.num_heads * self.head_dim]);
        self.o_proj.forward(&attn_out)
    }

    /// Rotate with the `[3, B, L]` (T, H, W) MRoPE position ids and attend.
    ///
    /// There is deliberately no `fast_rope` text-only shortcut here, unlike
    /// `qwen3_vl.rs`. Compass pre-permutes `inv_freq` before assigning axes, so
    /// even when all three axes carry the same position the table is not the
    /// natural-order 1-D RoPE that `fast_rope` builds; taking that shortcut
    /// would rotate every sliding layer against a permuted frequency list. A
    /// text-only sequence simply arrives here with all three rows equal.
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
        position_ids: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, l) = (shape[0], shape[1]);
        let [q, k, v] = self.project(x, b, l);

        let (q, k) = match &self.rope {
            None => (q, k),
            Some(table) => {
                let (cos, sin) = table.forward(position_ids);
                apply_multimodal_rotary_pos_emb(&q, &k, &cos, &sin)
            }
        };

        self.attend(&q, k, v, cache, mask, b, l)
    }
}

pub(crate) struct CompassMLP {
    gate_proj: UnifiedLinear,
    up_proj: UnifiedLinear,
    down_proj: UnifiedLinear,
}

impl CompassMLP {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &CompassTextConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let gs = config.group_size();
        let bits = config.bits();
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
        let gate = self.gate_proj.forward(x);
        let up = self.up_proj.forward(x);
        let activated = mlxcel_core::compiled_swiglu_activation(&gate, &up);
        self.down_proj.forward(&activated)
    }
}

/// Parallel block: attention and MLP both read the same normed input and both
/// land on the residual. There is no `post_attention_layernorm` in this family.
pub(crate) struct CompassBlock {
    pub(crate) self_attn: CompassAttention,
    pub(crate) mlp: CompassMLP,
    pub(crate) input_layernorm: CompassNorm,
}

impl CompassBlock {
    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        cache: &mut KVCache,
        mask: Option<&MlxArray>,
        position_ids: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let n = self.input_layernorm.forward(x);
        let attn = self.self_attn.forward(&n, cache, mask, position_ids);
        let ff = self.mlp.forward(&n);
        let sum = mlxcel_core::add(&attn, &ff);
        mlxcel_core::add(&sum, x)
    }
}
