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

//! Qwen2-0.5B-shaped query resampler for DeepSeek-OCR 2
//! (`vision_model.qwen2_encoder.*`).
//!
//! Replaces DeepSeek-OCR 1's CLIP stage. The SAM compressor grid is flattened
//! to `(B, S, 896)` image tokens; a learnable query bank of the same length is
//! concatenated after them, and the joint sequence runs through 24
//! Qwen2-style transformer layers (GQA + rotary + SwiGLU, pre-RMSNorm) under a
//! mixed mask: image tokens attend bidirectionally among themselves and never
//! see the queries, while queries attend to all image tokens and causally to
//! earlier queries. Only the query outputs are returned, `(B, Q, 896)`.
//!
//! Reference: mlx-vlm `mlx_vlm/models/deepseekocr_2/` (deepencoderv2 resampler).
//! Layout convention: activations are channels-last `(B, tokens, C)`; linear
//! weights are `(out_features, in_features)`.

use mlxcel_core::layers::{Linear, RMSNorm};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

/// Large negative additive-mask value; `softmax` drives these positions to 0.
const MASK_NEG: f32 = -1e9;

/// Static configuration of the query resampler. Only `dim` is spelled out in
/// `vision_config.width["qwen2-0-5b"]`; the rest are fixed checkpoint properties.
#[derive(Clone)]
pub struct Qwen2ResamplerConfig {
    pub dim: i32,
    pub num_layers: usize,
    pub num_heads: i32,
    pub num_kv_heads: i32,
    pub head_dim: i32,
    pub intermediate: i32,
    pub rms_eps: f32,
    pub rope_theta: f32,
    /// Query-bank length for the 1024 (global) view: `(1024/16/4)^2 = 256`.
    pub query_len_global: i32,
    /// Query-bank length for the 768 tile view: `(768/16/4)^2 = 144`.
    pub query_len_tile: i32,
}

impl Default for Qwen2ResamplerConfig {
    fn default() -> Self {
        Self {
            dim: 896,
            num_layers: 24,
            num_heads: 14,
            num_kv_heads: 2,
            head_dim: 64,
            intermediate: 4864,
            rms_eps: 1e-6,
            rope_theta: 1_000_000.0,
            query_len_global: 256,
            query_len_tile: 144,
        }
    }
}

/// Additive attention mask for the `[image (S) | queries (Q)]` sequence, flat
/// row-major `(S+Q) * (S+Q)`. Row `i` is the query position, column `j` the key.
/// Blocked cells are `MASK_NEG`, allowed cells `0`:
/// - image rows (`i < S`): allowed over image keys, blocked over query keys.
/// - query rows (`i >= S`): allowed over all image keys, causal within the
///   query block (`allowed iff (j - S) <= (i - S)`).
pub fn mixed_attn_mask(s: i32, q: i32) -> Vec<f32> {
    let n = (s + q) as usize;
    let s = s as usize;
    let mut m = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let blocked = if i < s {
                j >= s
            } else if j < s {
                false
            } else {
                (j - s) > (i - s)
            };
            if blocked {
                m[i * n + j] = MASK_NEG;
            }
        }
    }
    m
}

struct Qwen2Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    rope_theta: f32,
}

impl Qwen2Attention {
    /// `x`: `(B, L, D)`, `mask`: additive `(1, 1, L, L)` broadcastable. Position
    /// ids are `0..L-1` (offset 0), rotary over the full head_dim.
    fn forward(&self, x: &MlxArray, mask: &MlxArray) -> UniquePtr<MlxArray> {
        let sh = mlxcel_core::array_shape(x);
        let (b, l) = (sh[0], sh[1]);
        let (heads, kv, hd) = (self.num_heads, self.num_kv_heads, self.head_dim);

        let to_bhld = |proj: &Linear, n_heads: i32| {
            let p = proj.forward(x); // (B, L, n_heads*hd)
            let p = mlxcel_core::reshape(&p, &[b, l, n_heads, hd]);
            mlxcel_core::transpose_axes(&p, &[0, 2, 1, 3]) // (B, n_heads, L, hd)
        };
        let q = to_bhld(&self.q_proj, heads);
        let k = to_bhld(&self.k_proj, kv);
        let v = to_bhld(&self.v_proj, kv);

        // Rotary (NeoX half-split, offset 0) over the full head_dim.
        let q = mlxcel_core::fast_rope(&q, hd, false, self.rope_theta, 1.0, 0);
        let k = mlxcel_core::fast_rope(&k, hd, false, self.rope_theta, 1.0, 0);

        // GQA: repeat each KV head to match the query-head count.
        let k = mlxcel_core::utils::repeat_kv(&k, heads / kv);
        let v = mlxcel_core::utils::repeat_kv(&v, heads / kv);

        // SAFETY: q/k/v/mask are valid arrays live for the call.
        let out = unsafe {
            mlxcel_core::scaled_dot_product_attention(&q, &k, &v, self.scale, mask as *const _)
        };
        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]); // (B, L, heads, hd)
        let out = mlxcel_core::reshape(&out, &[b, l, heads * hd]);
        self.o_proj.forward(&out)
    }
}

struct Qwen2Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Qwen2Mlp {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let g = mlxcel_core::silu(&self.gate.forward(x));
        let u = self.up.forward(x);
        self.down.forward(&mlxcel_core::multiply(&g, &u))
    }
}

struct Qwen2Layer {
    input_layernorm: RMSNorm,
    attn: Qwen2Attention,
    post_attention_layernorm: RMSNorm,
    mlp: Qwen2Mlp,
}

impl Qwen2Layer {
    fn forward(&self, x: &MlxArray, mask: &MlxArray) -> UniquePtr<MlxArray> {
        let y = self.attn.forward(&self.input_layernorm.forward(x), mask);
        let x = mlxcel_core::add(x, &y);
        let y = self.mlp.forward(&self.post_attention_layernorm.forward(&x));
        mlxcel_core::add(&x, &y)
    }
}

/// The query resampler: query banks, 24 transformer layers, final RMSNorm.
pub struct Qwen2Resampler {
    config: Qwen2ResamplerConfig,
    query_global: UniquePtr<MlxArray>, // (query_len_global, dim)
    query_tile: UniquePtr<MlxArray>,   // (query_len_tile, dim)
    layers: Vec<Qwen2Layer>,
    norm: RMSNorm,
}

impl Qwen2Resampler {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: Qwen2ResamplerConfig,
    ) -> Result<Self, String> {
        let get = |name: &str| -> Result<UniquePtr<MlxArray>, String> {
            weights
                .get(name)
                .map(|w| mlxcel_core::copy(w))
                .ok_or_else(|| format!("Qwen2 resampler weight missing: {name}"))
        };
        let lin = |p: &str, bias: bool| -> Result<Linear, String> {
            Ok(Linear::new(
                get(&format!("{p}.weight"))?,
                if bias {
                    Some(get(&format!("{p}.bias"))?)
                } else {
                    None
                },
            ))
        };
        let rms = |p: &str| -> Result<RMSNorm, String> {
            Ok(RMSNorm::new(get(&format!("{p}.weight"))?, config.rms_eps))
        };

        let scale = (config.head_dim as f32).powf(-0.5);
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let lp = format!("{prefix}.layers.{i}");
            layers.push(Qwen2Layer {
                input_layernorm: rms(&format!("{lp}.input_layernorm"))?,
                attn: Qwen2Attention {
                    q_proj: lin(&format!("{lp}.self_attn.q_proj"), true)?,
                    k_proj: lin(&format!("{lp}.self_attn.k_proj"), true)?,
                    v_proj: lin(&format!("{lp}.self_attn.v_proj"), true)?,
                    o_proj: lin(&format!("{lp}.self_attn.o_proj"), false)?,
                    num_heads: config.num_heads,
                    num_kv_heads: config.num_kv_heads,
                    head_dim: config.head_dim,
                    scale,
                    rope_theta: config.rope_theta,
                },
                post_attention_layernorm: rms(&format!("{lp}.post_attention_layernorm"))?,
                mlp: Qwen2Mlp {
                    gate: lin(&format!("{lp}.mlp.gate_proj"), false)?,
                    up: lin(&format!("{lp}.mlp.up_proj"), false)?,
                    down: lin(&format!("{lp}.mlp.down_proj"), false)?,
                },
            });
        }

        Ok(Self {
            query_global: get(&format!("{prefix}.query_1024"))?,
            query_tile: get(&format!("{prefix}.query_768"))?,
            norm: rms(&format!("{prefix}.norm"))?,
            config,
            layers,
        })
    }

    /// `image_tokens`: `(B, S, dim)` flattened SAM grid. Returns `(B, Q, dim)`
    /// with `Q == S` (the query outputs). Selects `query_1024` for the global
    /// view (`S == query_len_global`) and `query_768` for a tile.
    pub fn forward(&self, image_tokens: &MlxArray) -> UniquePtr<MlxArray> {
        let sh = mlxcel_core::array_shape(image_tokens);
        let (b, s, dim) = (sh[0], sh[1], sh[2]);
        let bank = if s == self.config.query_len_global {
            &self.query_global
        } else {
            &self.query_tile
        };
        let q = mlxcel_core::array_shape(bank)[0];

        // Broadcast the query bank to (B, Q, dim) in the image dtype.
        let queries = mlxcel_core::reshape(bank, &[1, q, dim]);
        let queries = mlxcel_core::broadcast_to(&queries, &[b, q, dim]);
        let queries = mlxcel_core::astype(&queries, mlxcel_core::array_dtype(image_tokens));
        let mut x = mlxcel_core::concatenate(image_tokens, &queries, 1); // (B, S+Q, dim)

        let mask = mlxcel_core::from_slice_f32(&mixed_attn_mask(s, q), &[1, 1, s + q, s + q]);
        for layer in &self.layers {
            x = layer.forward(&x, &mask);
        }
        x = self.norm.forward(&x);

        // Keep only the query outputs: positions S .. S+Q.
        mlxcel_core::slice(&x, &[0, s, 0], &[b, s + q, dim])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_mask_quadrants_are_exact() {
        // S = 2, Q = 2, N = 4. Blocked = MASK_NEG, allowed = 0.
        let m = mixed_attn_mask(2, 2);
        let at = |i: usize, j: usize| m[i * 4 + j];
        // Image rows (0,1): image keys (0,1) allowed, query keys (2,3) blocked.
        for i in 0..2 {
            assert_eq!(at(i, 0), 0.0);
            assert_eq!(at(i, 1), 0.0);
            assert_eq!(at(i, 2), MASK_NEG);
            assert_eq!(at(i, 3), MASK_NEG);
        }
        // Query row 2 (first query): all image keys allowed; query key 2 allowed
        // (self), query key 3 blocked (future).
        assert_eq!(at(2, 0), 0.0);
        assert_eq!(at(2, 1), 0.0);
        assert_eq!(at(2, 2), 0.0);
        assert_eq!(at(2, 3), MASK_NEG);
        // Query row 3 (second query): all image keys + both query keys allowed.
        assert_eq!(at(3, 0), 0.0);
        assert_eq!(at(3, 1), 0.0);
        assert_eq!(at(3, 2), 0.0);
        assert_eq!(at(3, 3), 0.0);
    }
}
