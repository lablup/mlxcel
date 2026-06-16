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

//! Molmo-Point Vision Encoder components
//!
//! Reuses Molmo2 ViT blocks (they are architecturally identical).
//! Adds the Molmo-Point-specific connector and point predictor building blocks:
//!   - MolmoPointConnector: attention pooling + SwiGLU projection (no wo layer)
//!   - PointPredictor: patch/subpatch queries and keys, location head
//!   - MolmoPointPatchRope: 1D rotary for patch embeddings
//!   - PadWithLearnedVector: "no more points" class embedding
//!
//! Reference: https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/molmo_point/molmo_point.py

use mlxcel_core::layers::Linear;
use mlxcel_core::utils::slice_axis;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::molmo2::ImageProjectorMLP;

// ---------- ViTAttentionNoOutput ----------

/// ViT attention without output projection (out_layer=False in Python).
///
/// Used by the MolmoPointConnector for attention pooling where the output
/// is projected through a separate SwiGLU MLP rather than a linear wo layer.
pub struct ViTAttentionNoOutput {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl ViTAttentionNoOutput {
    pub fn forward(
        &self,
        inputs_q: &MlxArray,
        inputs_kv: Option<&MlxArray>,
        attn_mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let inputs_k = inputs_kv.unwrap_or(inputs_q);
        let inputs_v = inputs_kv.unwrap_or(inputs_q);

        let xq = self.wq.forward(inputs_q);
        let xk = self.wk.forward(inputs_k);
        let xv = self.wv.forward(inputs_v);

        let q_shape = mlxcel_core::array_shape(&xq);
        let bsz = q_shape[0];
        let q_len = q_shape[1];
        let k_shape = mlxcel_core::array_shape(&xk);
        let kv_len = k_shape[1];

        let xq = mlxcel_core::reshape(&xq, &[bsz, q_len, self.num_heads, self.head_dim]);
        let mut xk = mlxcel_core::reshape(&xk, &[bsz, kv_len, self.num_kv_heads, self.head_dim]);
        let mut xv = mlxcel_core::reshape(&xv, &[bsz, kv_len, self.num_kv_heads, self.head_dim]);

        if self.num_heads != self.num_kv_heads {
            let n_rep = self.num_heads / self.num_kv_heads;
            xk = mlxcel_core::repeat(&xk, n_rep, 2);
            xv = mlxcel_core::repeat(&xv, n_rep, 2);
        }

        let q = mlxcel_core::transpose_axes(&xq, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&xk, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&xv, &[0, 2, 1, 3]);

        // Float32 attention for numerical stability
        let q = mlxcel_core::astype(&q, mlxcel_core::dtype::FLOAT32);
        let k = mlxcel_core::astype(&k, mlxcel_core::dtype::FLOAT32);
        let v = mlxcel_core::astype(&v, mlxcel_core::dtype::FLOAT32);

        let k_t = mlxcel_core::transpose_axes(&k, &[0, 1, 3, 2]);
        let mut scores = mlxcel_core::matmul(&q, &k_t);
        scores = mlxcel_core::multiply_scalar(&scores, self.scale);

        if let Some(mask) = attn_mask {
            let neg_inf = mlxcel_core::full_like(&scores, -1e9);
            scores = mlxcel_core::where_cond(mask, &scores, &neg_inf);
        }

        let weights = mlxcel_core::softmax(&scores, -1);
        let out = mlxcel_core::matmul(&weights, &v);

        let dtype = mlxcel_core::array_dtype(inputs_q);
        let out = mlxcel_core::astype(&out, dtype);

        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);
        // Reshape back: [B, L, num_heads * head_dim] -- no wo projection
        mlxcel_core::reshape(&out, &[bsz, q_len, self.num_heads * self.head_dim])
    }

    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
    ) -> Result<Self, String> {
        let wq = Linear::from_weights(weights, &format!("{prefix}.wq"))?;
        let wk = Linear::from_weights(weights, &format!("{prefix}.wk"))?;
        let wv = Linear::from_weights(weights, &format!("{prefix}.wv"))?;

        Ok(Self {
            wq,
            wk,
            wv,
            num_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }
}

// ---------- MolmoPointConnector ----------

/// Attention-based 2D pooling connector for Molmo-Point.
///
/// Unlike Molmo2, the attention pooling omits the wo output projection
/// (out_layer=False in Python). The pooled features are then projected
/// through the SwiGLU ImageProjectorMLP.
pub struct MolmoPointConnector {
    image_projector: ImageProjectorMLP,
    image_pooling_2d: ViTAttentionNoOutput,
    positional_embeddings: Option<UniquePtr<MlxArray>>, // [n_pos, pool_dim]
    pooling_attention_mask: bool,
}

impl MolmoPointConnector {
    pub fn forward(&self, to_pool: &MlxArray, to_pool_mask: &MlxArray) -> UniquePtr<MlxArray> {
        // Apply positional embeddings if configured
        let mut pool_input = if let Some(pos_emb) = &self.positional_embeddings {
            let x_shape = mlxcel_core::array_shape(to_pool);
            let l = x_shape[1];
            let dim = x_shape[2];
            // pos_emb: [n_pos, dim], slice to [l, dim], reshape to [1, l, dim]
            let pos_slice = slice_axis(pos_emb, 0, 0, l);
            let pos_3d = mlxcel_core::reshape(&pos_slice, &[1, l, dim]);
            let pos_cast = mlxcel_core::astype(&pos_3d, mlxcel_core::array_dtype(to_pool));
            mlxcel_core::add(to_pool, &pos_cast)
        } else {
            mlxcel_core::copy(to_pool)
        };

        let attn_mask = if self.pooling_attention_mask {
            let mask_shape = mlxcel_core::array_shape(to_pool_mask);
            Some(mlxcel_core::reshape(
                to_pool_mask,
                &[mask_shape[0], 1, 1, mask_shape[1]],
            ))
        } else {
            // Mask by multiplying: to_pool * mask[:, :, None]
            let mask_f = mlxcel_core::astype(to_pool_mask, mlxcel_core::array_dtype(&pool_input));
            let mask_shape = mlxcel_core::array_shape(&mask_f);
            let mask_3d = mlxcel_core::reshape(&mask_f, &[mask_shape[0], mask_shape[1], 1]);
            pool_input = mlxcel_core::multiply(&pool_input, &mask_3d);
            None
        };

        // Compute denominator for weighted mean query
        let pool_shape = mlxcel_core::array_shape(&pool_input);
        let mask_flat = mlxcel_core::reshape(to_pool_mask, &[-1, pool_shape[1]]);
        let denom_f32 = mlxcel_core::astype(&mask_flat, mlxcel_core::dtype::FLOAT32);
        let denom = mlxcel_core::sum_axis(&denom_f32, -1, true);
        let ones = mlxcel_core::ones(&[1, 1], mlxcel_core::dtype::FLOAT32);
        let denom = mlxcel_core::maximum(&denom, &ones);

        // query = sum(pool_input, axis=-2) / denom -> [B, 1, dim]
        let sum_pool = mlxcel_core::sum_axis(&pool_input, -2, true);
        let sum_f32 = mlxcel_core::astype(&sum_pool, mlxcel_core::dtype::FLOAT32);
        let denom_3d = mlxcel_core::reshape(&denom, &[-1, 1, 1]);
        let query = mlxcel_core::divide(&sum_f32, &denom_3d);
        let query = mlxcel_core::astype(&query, mlxcel_core::array_dtype(&pool_input));

        // Cross-attention pooling (no output projection)
        let pooled = self
            .image_pooling_2d
            .forward(&query, Some(&pool_input), attn_mask.as_deref());

        // Project through SwiGLU MLP
        self.image_projector.forward(&pooled)
    }

    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        positional_embeddings_size: Option<usize>,
        pooling_attention_mask: bool,
    ) -> Result<Self, String> {
        let image_projector =
            ImageProjectorMLP::from_weights(weights, &format!("{prefix}.image_projector"))?;
        let image_pooling_2d = ViTAttentionNoOutput::from_weights(
            weights,
            &format!("{prefix}.image_pooling_2d"),
            num_heads,
            num_kv_heads,
            head_dim,
        )?;

        let positional_embeddings = if positional_embeddings_size.is_some() {
            weights
                .get(&format!("{prefix}.positional_embeddings.bias"))
                .map(|w| mlxcel_core::copy(w))
        } else {
            None
        };

        Ok(Self {
            image_projector,
            image_pooling_2d,
            positional_embeddings,
            pooling_attention_mask,
        })
    }
}

// ---------- PadWithLearnedVector ----------

/// Pads a tensor along dim=1 with a learned vector (for "no more points" class).
pub struct PadWithLearnedVector {
    pub vector: UniquePtr<MlxArray>, // [dim]
}

impl PadWithLearnedVector {
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let b = shape[0];
        let dim = mlxcel_core::array_shape(&self.vector)[0];
        let vec_3d = mlxcel_core::reshape(&self.vector, &[1, 1, dim]);
        let vec_broadcast = mlxcel_core::broadcast_to(&vec_3d, &[b, 1, dim]);
        let vec_cast = mlxcel_core::astype(&vec_broadcast, mlxcel_core::array_dtype(x));
        // Concatenate along axis 1
        mlxcel_core::concatenate(x, &vec_cast, 1)
    }

    pub fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        let vector = weights
            .get(&format!("{prefix}.vector"))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {prefix}.vector"))?;
        Ok(Self { vector })
    }
}

// ---------- MolmoPointPatchRope ----------

/// 1D rotary position embedding for patch tokens.
pub struct MolmoPointPatchRope {
    inv_freq: UniquePtr<MlxArray>, // [dim/2]
}

impl MolmoPointPatchRope {
    pub fn forward(&self, x: &MlxArray, position_ids: &MlxArray) -> UniquePtr<MlxArray> {
        // x: [N, dim], position_ids: [N]
        let x_f32 = mlxcel_core::astype(x, mlxcel_core::dtype::FLOAT32);
        let pos_f32 = mlxcel_core::astype(position_ids, mlxcel_core::dtype::FLOAT32);
        let inv_freq = mlxcel_core::astype(&self.inv_freq, mlxcel_core::dtype::FLOAT32);

        // freqs = position_ids[:, None] * inv_freq[None, :]
        let n = mlxcel_core::array_shape(&pos_f32)[0];
        let half_dim = mlxcel_core::array_shape(&inv_freq)[0];
        let pos_2d = mlxcel_core::reshape(&pos_f32, &[n, 1]);
        let inv_2d = mlxcel_core::reshape(&inv_freq, &[1, half_dim]);
        let freqs = mlxcel_core::multiply(&pos_2d, &inv_2d);

        // emb = [freqs, freqs]
        let emb = mlxcel_core::concatenate(&freqs, &freqs, -1);
        let cos_emb = mlxcel_core::cos(&emb);
        let sin_emb = mlxcel_core::sin(&emb);

        // rotate_half: reshape x to [N, 2, hs//2], split, then [-x2, x1]
        let hs = mlxcel_core::array_shape(&x_f32)[1];
        let x_reshaped = mlxcel_core::reshape(&x_f32, &[n, 2, hs / 2]);
        let x1 = slice_axis(&x_reshaped, 1, 0, 1);
        let x2 = slice_axis(&x_reshaped, 1, 1, 2);
        let x1_flat = mlxcel_core::reshape(&x1, &[n, hs / 2]);
        let x2_flat = mlxcel_core::reshape(&x2, &[n, hs / 2]);
        let neg_x2 = mlxcel_core::negative(&x2_flat);
        let rotated = mlxcel_core::concatenate(&neg_x2, &x1_flat, -1);

        // out = x * cos + rotated * sin
        let out = mlxcel_core::add(
            &mlxcel_core::multiply(&x_f32, &cos_emb),
            &mlxcel_core::multiply(&rotated, &sin_emb),
        );

        mlxcel_core::astype(&out, mlxcel_core::array_dtype(x))
    }

    pub fn new(theta: f32, dim: i32) -> Self {
        let half_dim = dim / 2;
        let mut inv_data = vec![0.0f32; half_dim as usize];
        for i in 0..half_dim {
            let exponent = (2 * i) as f32 / dim as f32;
            inv_data[i as usize] = 1.0 / theta.powf(exponent);
        }
        let inv_freq = mlxcel_core::from_slice_f32(&inv_data, &[half_dim]);
        Self { inv_freq }
    }
}

// ---------- PointPredictor ----------

/// Point predictor for Molmo-Point: produces patch, subpatch, and location logits.
pub struct PointPredictor {
    pub x_norm: Option<mlxcel_core::layers::RMSNorm>,
    pub patch_rotary: Option<MolmoPointPatchRope>,
    pub patch_q: Linear,
    pub patch_k: Linear,
    pub subpatch_q: Linear,
    pub subpatch_k: Linear,
    pub add_no_point_class_embed: PadWithLearnedVector,
    pub subpatch_loc_k: Option<Linear>, // 3x3 location head
}

impl PointPredictor {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        layer_norm_eps: f32,
        has_layer_norm_x: bool,
        token_prediction_rotary: &str,
        rotary_theta: f32,
        patch_embed_dim: i32,
        has_patch_location: bool,
    ) -> Result<Self, String> {
        let x_norm = if has_layer_norm_x {
            let w = get_weight_copy(weights, &format!("{prefix}.x_norm.weight"))?;
            Some(mlxcel_core::layers::RMSNorm::new(w, layer_norm_eps))
        } else {
            None
        };

        let patch_rotary = if token_prediction_rotary == "one_d" {
            Some(MolmoPointPatchRope::new(rotary_theta, patch_embed_dim))
        } else {
            None
        };

        let patch_q = Linear::from_weights(weights, &format!("{prefix}.patch_q"))?;
        let patch_k = Linear::from_weights(weights, &format!("{prefix}.patch_k"))?;
        let subpatch_q = Linear::from_weights(weights, &format!("{prefix}.subpatch_q"))?;
        let subpatch_k = Linear::from_weights(weights, &format!("{prefix}.subpatch_k"))?;
        let add_no_point_class_embed = PadWithLearnedVector::from_weights(
            weights,
            &format!("{prefix}.add_no_point_class_embed"),
        )?;

        let subpatch_loc_k = if has_patch_location {
            Some(Linear::from_weights(
                weights,
                &format!("{prefix}.subpatch_loc_k"),
            )?)
        } else {
            None
        };

        Ok(Self {
            x_norm,
            patch_rotary,
            patch_q,
            patch_k,
            subpatch_q,
            subpatch_k,
            add_no_point_class_embed,
            subpatch_loc_k,
        })
    }
}

fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {name}"))
}
