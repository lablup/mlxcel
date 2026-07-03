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

//! LFM2-VL packed-patch vision tower (SigLIP2-style, native variable resolution).
//!
//! Port of the LiquidAI LFM2-VL vision encoder
//! (https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/lfm2_vl). Unlike
//! the fixed-grid [`super::siglip::SigLipVisionModel`], this tower consumes
//! pre-packed patch vectors `(1, num_patches, P*P*C)` at each image's native
//! patch count and resizes its learned position grid per image, so no
//! convolution, padding, or pixel attention mask is involved.
//!
//! Structure (`vision_tower.*` in the checkpoint; all weights plain, not
//! quantized): a `patch_embedding` Linear (`P*P*C -> hidden`), a learned
//! `position_embedding` table `(num_patches, hidden)` bicubically resampled to
//! each image's `(h, w)` grid (reusing `kimi_vl::pos_emb`), N pre-norm
//! encoder layers (LayerNorm -> MHA -> residual, LayerNorm -> gelu-tanh MLP ->
//! residual), and a final `post_layernorm`. `vision_use_head` is false, so no
//! pooled head weights ship.
//!
//! Used by: `vision::lfm2_vl::Lfm2VlModel`, `loading::load_lfm2_vl`.

use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::kimi_vl::pos_emb::Learnable2DInterpPosEmb;

/// LFM2-VL vision-tower config (parsed from the `vision_config` sub-object).
#[derive(Debug, Clone)]
pub struct Lfm2VlVisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub patch_size: usize,
    /// Position-table length; the init grid side is `sqrt(num_patches)`.
    pub num_patches: usize,
    pub layer_norm_eps: f32,
    /// Which encoder layer's output feeds the projector (`-1` = full stack).
    pub vision_feature_layer: i32,
}

impl Default for Lfm2VlVisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            patch_size: 16,
            num_patches: 256,
            layer_norm_eps: 1e-6,
            vision_feature_layer: -1,
        }
    }
}

fn load_layer_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<LayerNorm, String> {
    let weight = weights
        .get(&format!("{prefix}.weight"))
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {prefix}.weight"))?;
    let bias = weights
        .get(&format!("{prefix}.bias"))
        .map(|w| mlxcel_core::copy(w));
    Ok(LayerNorm::new(weight, bias, eps))
}

struct Attention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    out_proj: UnifiedLinear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl Attention {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        hidden: usize,
        num_heads: usize,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let head_dim = (hidden / num_heads) as i32;
        Ok(Self {
            q_proj: UnifiedLinear::from_weights(weights, &format!("{prefix}.q_proj"), gs, bits)?,
            k_proj: UnifiedLinear::from_weights(weights, &format!("{prefix}.k_proj"), gs, bits)?,
            v_proj: UnifiedLinear::from_weights(weights, &format!("{prefix}.v_proj"), gs, bits)?,
            out_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.out_proj"),
                gs,
                bits,
            )?,
            num_heads: num_heads as i32,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    /// `x`: `[1, N, hidden]`. Bidirectional (no mask), softmax over the key axis.
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (b, n) = (shape[0], shape[1]);
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        let reshape_heads = |t: &MlxArray| -> UniquePtr<MlxArray> {
            let t = mlxcel_core::reshape(t, &[b, n, self.num_heads, self.head_dim]);
            mlxcel_core::transpose_axes(&t, &[0, 2, 1, 3])
        };
        let q = reshape_heads(&q);
        let k = reshape_heads(&k);
        let v = reshape_heads(&v);

        let out = unsafe {
            mlxcel_core::layers::attention_from_ptr(
                &q,
                &k,
                &v,
                self.scale,
                std::ptr::null(),
                0.0,
                0,
            )
        };
        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);
        let out = mlxcel_core::reshape(&out, &[b, n, self.num_heads * self.head_dim]);
        self.out_proj.forward(&out)
    }
}

struct Mlp {
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
}

impl Mlp {
    fn from_weights(weights: &WeightMap, prefix: &str, gs: i32, bits: i32) -> Result<Self, String> {
        Ok(Self {
            fc1: UnifiedLinear::from_weights(weights, &format!("{prefix}.fc1"), gs, bits)?,
            fc2: UnifiedLinear::from_weights(weights, &format!("{prefix}.fc2"), gs, bits)?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // hidden_act = "gelu_pytorch_tanh" (tanh-approximate GELU).
        let x = self.fc1.forward(x);
        let x = mlxcel_core::gelu_approx(&x);
        self.fc2.forward(&x)
    }
}

struct EncoderLayer {
    layer_norm1: LayerNorm,
    self_attn: Attention,
    layer_norm2: LayerNorm,
    mlp: Mlp,
}

impl EncoderLayer {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &Lfm2VlVisionConfig,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            layer_norm1: load_layer_norm(
                weights,
                &format!("{prefix}.layer_norm1"),
                cfg.layer_norm_eps,
            )?,
            self_attn: Attention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                cfg.hidden_size,
                cfg.num_attention_heads,
                gs,
                bits,
            )?,
            layer_norm2: load_layer_norm(
                weights,
                &format!("{prefix}.layer_norm2"),
                cfg.layer_norm_eps,
            )?,
            mlp: Mlp::from_weights(weights, &format!("{prefix}.mlp"), gs, bits)?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let r = self.self_attn.forward(&self.layer_norm1.forward(x));
        let h = mlxcel_core::add(x, &r);
        let r = self.mlp.forward(&self.layer_norm2.forward(&h));
        mlxcel_core::add(&h, &r)
    }
}

/// LFM2-VL packed-patch vision tower.
pub struct Lfm2VlVisionTower {
    patch_embedding: UnifiedLinear,
    position_embedding: Learnable2DInterpPosEmb,
    layers: Vec<EncoderLayer>,
    post_layernorm: LayerNorm,
    vision_feature_layer: i32,
}

impl Lfm2VlVisionTower {
    /// Load from the `vision_tower.*` weights. `gs`/`bits` are threaded through
    /// to `UnifiedLinear`, which loads plain (non-quantized) tensors as a regular
    /// `Linear` when no `.scales` companion is present (the released checkpoint).
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        cfg: &Lfm2VlVisionConfig,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let patch_embedding = UnifiedLinear::from_weights(
            weights,
            &format!("{prefix}.embeddings.patch_embedding"),
            gs,
            bits,
        )?;

        let side = (cfg.num_patches as f64).sqrt().round() as i32;
        let position_embedding = Learnable2DInterpPosEmb::from_weights(
            weights,
            &format!("{prefix}.embeddings.position_embedding"),
            side,
            side,
            cfg.hidden_size as i32,
        )?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(EncoderLayer::from_weights(
                weights,
                &format!("{prefix}.encoder.layers.{i}"),
                cfg,
                gs,
                bits,
            )?);
        }

        let post_layernorm = load_layer_norm(
            weights,
            &format!("{prefix}.post_layernorm"),
            cfg.layer_norm_eps,
        )?;

        Ok(Self {
            patch_embedding,
            position_embedding,
            layers,
            post_layernorm,
            vision_feature_layer: cfg.vision_feature_layer,
        })
    }

    /// `packed_patches`: `[1, h*w, P*P*C]`; `grid`: `(h, w)` patch grid.
    /// Returns `[1, h*w, hidden]`.
    pub fn forward(&self, packed_patches: &MlxArray, grid: (i32, i32)) -> UniquePtr<MlxArray> {
        // Patch embedding, then add the per-image resampled position grid.
        let mut h = self.patch_embedding.forward(packed_patches);
        h = self.position_embedding.add_to(&h, &[grid]);

        // `vision_feature_layer`: -1 (default) runs the full stack then
        // post_layernorm; any other value truncates to layers 0..=idx (negative
        // indexes from the end) and skips post_layernorm.
        let num = self.layers.len() as i32;
        let truncate = if self.vision_feature_layer == -1 {
            None
        } else {
            let idx = if self.vision_feature_layer < 0 {
                num + self.vision_feature_layer
            } else {
                self.vision_feature_layer
            };
            Some(idx.clamp(0, num - 1) as usize)
        };

        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h);
            if truncate == Some(i) {
                return h;
            }
        }
        self.post_layernorm.forward(&h)
    }
}
