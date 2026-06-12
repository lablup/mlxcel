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

//! Molmo2 Vision Encoder + Adapter
//!
//! Architecture:
//! - ViT: 25 transformer blocks, Linear patch embedding (not Conv2d),
//!   positional embedding with bicubic interpolation
//! - Adapter: Attention pooling 2D + SwiGLU image projector
//! - Layer selection: [-3, -9] = [22, 16] → concatenate → pool_dim = 2*1152
//!
//! Reference: https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/molmo2/vision.py

use mlxcel_core::layers::{LayerNorm, Linear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

// ViT MLP.
pub(crate) struct ViTMLP {
    w1: Linear,
    w2: Linear,
}

impl ViTMLP {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.w1.forward(x);
        let h = mlxcel_core::gelu_approx(&h);
        self.w2.forward(&h)
    }

    fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        let w1 = Linear::from_weights(weights, &format!("{}.w1", prefix))?;
        let w2 = Linear::from_weights(weights, &format!("{}.w2", prefix))?;
        Ok(Self { w1, w2 })
    }
}

// ViT Multi-Head Dot Product Attention (supports cross-attention).
// Used by: Molmo2, MolmoPoint
pub(crate) struct ViTAttention {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
    float32_attention: bool,
}

impl ViTAttention {
    fn forward(
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

        // Repeat KV heads if GQA
        if self.num_heads != self.num_kv_heads {
            let n_rep = self.num_heads / self.num_kv_heads;
            xk = mlxcel_core::repeat(&xk, n_rep, 2);
            xv = mlxcel_core::repeat(&xv, n_rep, 2);
        }

        // Transpose to [B, heads, L, head_dim]
        let q = mlxcel_core::transpose_axes(&xq, &[0, 2, 1, 3]);
        let k = mlxcel_core::transpose_axes(&xk, &[0, 2, 1, 3]);
        let v = mlxcel_core::transpose_axes(&xv, &[0, 2, 1, 3]);

        // Float32 attention for stability (only convert when needed)
        let dtype = mlxcel_core::array_dtype(inputs_q);
        let (q, k, v) = if self.float32_attention {
            (
                mlxcel_core::astype(&q, mlxcel_core::dtype::FLOAT32),
                mlxcel_core::astype(&k, mlxcel_core::dtype::FLOAT32),
                mlxcel_core::astype(&v, mlxcel_core::dtype::FLOAT32),
            )
        } else {
            (q, k, v)
        };

        // Use fast SDPA kernel instead of manual matmul+softmax
        let mask_ptr = attn_mask
            .map(|m| m as *const MlxArray)
            .unwrap_or(std::ptr::null());
        let out = unsafe {
            mlxcel_core::layers::attention_from_ptr(&q, &k, &v, self.scale, mask_ptr, 0.0, 0)
        };

        // Cast back to input dtype if needed
        let out = if self.float32_attention {
            mlxcel_core::astype(&out, dtype)
        } else {
            out
        };

        // Transpose and reshape back
        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);
        let out = mlxcel_core::reshape(&out, &[bsz, q_len, self.num_heads * self.head_dim]);
        self.wo.forward(&out)
    }

    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        _hidden_size: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        float32_attention: bool,
    ) -> Result<Self, String> {
        let wq = Linear::from_weights(weights, &format!("{}.wq", prefix))?;
        let wk = Linear::from_weights(weights, &format!("{}.wk", prefix))?;
        let wv = Linear::from_weights(weights, &format!("{}.wv", prefix))?;
        let wo = Linear::from_weights(weights, &format!("{}.wo", prefix))?;

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            num_heads,
            num_kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            float32_attention,
        })
    }
}

// ViT Block.
// Used by: Molmo2, MolmoPoint
pub(crate) struct Molmo2VisionBlock {
    attention: ViTAttention,
    feed_forward: ViTMLP,
    attention_norm: LayerNorm,
    ffn_norm: LayerNorm,
}

impl Molmo2VisionBlock {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // Pre-norm attention + residual
        let normed = self.attention_norm.forward(x);
        let attn_out = self.attention.forward(&normed, None, None);
        let h = mlxcel_core::add(x, &attn_out);

        // Pre-norm MLP + residual
        let normed = self.ffn_norm.forward(&h);
        let mlp_out = self.feed_forward.forward(&normed);
        mlxcel_core::add(&h, &mlp_out)
    }

    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        _hidden_size: i32,
        _intermediate_size: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        layer_norm_eps: f32,
        float32_attention: bool,
    ) -> Result<Self, String> {
        let attention = ViTAttention::from_weights(
            weights,
            &format!("{}.attention", prefix),
            _hidden_size,
            num_heads,
            num_kv_heads,
            head_dim,
            float32_attention,
        )?;
        let feed_forward = ViTMLP::from_weights(weights, &format!("{}.feed_forward", prefix))?;

        let attn_norm_w = get_weight_copy(weights, &format!("{}.attention_norm.weight", prefix))?;
        let attn_norm_b = weights
            .get(&format!("{}.attention_norm.bias", prefix))
            .map(|w| mlxcel_core::copy(w));
        let ffn_norm_w = get_weight_copy(weights, &format!("{}.ffn_norm.weight", prefix))?;
        let ffn_norm_b = weights
            .get(&format!("{}.ffn_norm.bias", prefix))
            .map(|w| mlxcel_core::copy(w));

        let attention_norm = LayerNorm::new(attn_norm_w, attn_norm_b, layer_norm_eps);
        let ffn_norm = LayerNorm::new(ffn_norm_w, ffn_norm_b, layer_norm_eps);

        Ok(Self {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
        })
    }
}

// Vision Transformer (returns all hidden states).
// Used by: Molmo2, MolmoPoint
pub(crate) struct Molmo2VisionTransformer {
    patch_embedding: Linear, // Linear, not Conv2d (patches already flattened)
    positional_embedding: UniquePtr<MlxArray>, // [image_num_pos, hidden_size]
    blocks: Vec<Molmo2VisionBlock>,
    image_num_pos: usize,
}

impl Molmo2VisionTransformer {
    pub(crate) fn add_pos_emb(
        &self,
        x: &MlxArray,
        patch_h: i32,
        patch_w: i32,
    ) -> UniquePtr<MlxArray> {
        let num_pos = self.image_num_pos as i32;
        let hidden_size = mlxcel_core::array_shape(&self.positional_embedding)[1];

        // For default size, use positional embedding directly
        // For non-default sizes, truncate/extend (bicubic interpolation would
        // require additional FFI but default 378x378 crops are always used)
        let num_patches = patch_h * patch_w;
        let pos_emb = if num_patches == num_pos {
            mlxcel_core::copy(&self.positional_embedding)
        } else if num_patches < num_pos {
            // Truncate
            let indices: Vec<i32> = (0..num_patches).collect();
            let idx = mlxcel_core::from_slice_i32(&indices, &[num_patches]);
            mlxcel_core::take(&self.positional_embedding, &idx, 0)
        } else {
            // For larger sizes, repeat last position (rare case)
            mlxcel_core::copy(&self.positional_embedding)
        };

        // x + pos_emb[None, :, :]
        let pos_emb = mlxcel_core::reshape(&pos_emb, &[1, num_patches.min(num_pos), hidden_size]);
        let pos_emb = mlxcel_core::astype(&pos_emb, mlxcel_core::array_dtype(x));
        mlxcel_core::add(x, &pos_emb)
    }

    pub(crate) fn forward(
        &self,
        x: &MlxArray,
        patch_num: Option<(i32, i32)>,
    ) -> Vec<UniquePtr<MlxArray>> {
        let default_patch_size = (self.image_num_pos as f64).sqrt() as i32;
        let (patch_h, patch_w) = patch_num.unwrap_or((default_patch_size, default_patch_size));

        let x = self.patch_embedding.forward(x);
        let mut x = self.add_pos_emb(&x, patch_h, patch_w);

        let mut hidden_states = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            x = block.forward(&x);
            hidden_states.push(mlxcel_core::copy(&x));
        }
        hidden_states
    }

    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        num_layers: usize,
        hidden_size: i32,
        intermediate_size: i32,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        image_num_pos: usize,
        layer_norm_eps: f32,
        float32_attention: bool,
    ) -> Result<Self, String> {
        let patch_embedding =
            Linear::from_weights(weights, &format!("{}.patch_embedding", prefix))?;
        let positional_embedding =
            get_weight_copy(weights, &format!("{}.positional_embedding", prefix))?;

        let mut blocks = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let block = Molmo2VisionBlock::from_weights(
                weights,
                &format!("{}.transformer.{}", prefix, i),
                hidden_size,
                intermediate_size,
                num_heads,
                num_kv_heads,
                head_dim,
                layer_norm_eps,
                float32_attention,
            )?;
            blocks.push(block);
        }

        Ok(Self {
            patch_embedding,
            positional_embedding,
            blocks,
            image_num_pos,
        })
    }
}

// Image Projector MLP (SwiGLU).
// Used by: Molmo2, MolmoPoint
pub(crate) struct ImageProjectorMLP {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl ImageProjectorMLP {
    pub(crate) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // silu(w1(x)) * w3(x) → w2(...)
        let gate = self.w1.forward(x);
        let gate = mlxcel_core::silu(&gate);
        let up = self.w3.forward(x);
        let h = mlxcel_core::multiply(&gate, &up);
        self.w2.forward(&h)
    }

    pub(crate) fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        let w1 = Linear::from_weights(weights, &format!("{}.w1", prefix))?;
        let w2 = Linear::from_weights(weights, &format!("{}.w2", prefix))?;
        let w3 = Linear::from_weights(weights, &format!("{}.w3", prefix))?;
        Ok(Self { w1, w2, w3 })
    }
}

// Molmo2 Vision Model (ViT + Adapter).
pub struct Molmo2VisionModel {
    image_vit: Molmo2VisionTransformer,
    image_pooling_2d: ViTAttention,
    image_projector: ImageProjectorMLP,
    vit_layers: Vec<usize>, // Which ViT layers to extract features from
    pooling_attention_mask: bool,
}

impl Molmo2VisionModel {
    /// Encode images through the ViT, extracting features from selected layers
    fn encode_image(&self, images: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(images);
        let batch_size = shape[0];
        let num_crops = shape[1];
        let num_patch = shape[2];
        let patch_dim = shape[3];

        // Reshape to [B*num_crops, num_patch, patch_dim]
        let flat = mlxcel_core::reshape(images, &[batch_size * num_crops, num_patch, patch_dim]);
        let hidden_states = self.image_vit.forward(&flat, None);

        // Select and concatenate features from specified layers
        let features: Vec<&MlxArray> = self
            .vit_layers
            .iter()
            .map(|&layer| hidden_states[layer].as_ref().unwrap())
            .collect();

        let image_features = if features.len() == 1 {
            mlxcel_core::copy(features[0])
        } else {
            // Concatenate along last dimension
            let mut result = mlxcel_core::copy(features[0]);
            for &feat in &features[1..] {
                result = mlxcel_core::concatenate(&result, feat, -1);
            }
            result
        };

        // Reshape back to [B, num_crops, num_patch, features_dim]
        let feat_dim = mlxcel_core::array_shape(&image_features);
        let last_dim = feat_dim[feat_dim.len() - 1];
        mlxcel_core::reshape(
            &image_features,
            &[batch_size, num_crops, num_patch, last_dim],
        )
    }

    /// Full forward: encode → pool → project
    pub fn forward(&self, images: &MlxArray, pooled_patches_idx: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(images);
        let batch_size = shape[0];

        let image_features = self.encode_image(images);
        let feat_shape = mlxcel_core::array_shape(&image_features);
        let dim = feat_shape[feat_shape.len() - 1];

        // Flatten features: [B, num_crops * num_patch, dim]
        let flat_features = mlxcel_core::reshape(&image_features, &[batch_size, -1, dim]);

        // Build valid mask from pooling indices
        let pool_shape = mlxcel_core::array_shape(pooled_patches_idx);
        // pooled_patches_idx shape: [batch, num_pooled, pool_size]

        // valid = pooled_patches_idx >= 0
        let zeros = mlxcel_core::zeros_like(pooled_patches_idx);
        let valid = mlxcel_core::greater_equal(pooled_patches_idx, &zeros);
        // valid_token = any(valid, axis=-1)
        let valid_i32 = mlxcel_core::astype(&valid, mlxcel_core::dtype::INT32);

        // Clip indices to >= 0
        let idx = mlxcel_core::maximum(pooled_patches_idx, &zeros);

        // Gather features at pooling indices
        // idx shape: [B, num_pooled, pool_size]
        // flat_features shape: [B, total_patches, dim]
        // We need to do batched gather: for each batch, gather from flat_features using idx
        let to_pool = self.batched_gather(&flat_features, &idx, batch_size);
        // to_pool shape: [B, num_pooled, pool_size, dim]

        // Mask invalid positions
        let valid_4d =
            mlxcel_core::reshape(&valid, &[pool_shape[0], pool_shape[1], pool_shape[2], 1]);
        let valid_f = mlxcel_core::astype(&valid_4d, mlxcel_core::array_dtype(&to_pool));
        let to_pool = mlxcel_core::multiply(&to_pool, &valid_f);

        // Reshape for attention: [B * num_pooled, pool_size, dim]
        let to_pool = mlxcel_core::reshape(&to_pool, &[-1, pool_shape[2], dim]);

        // Build query: mean of valid patches per pooled position
        let (query, attn_mask) = if self.pooling_attention_mask {
            let valid_flat = mlxcel_core::reshape(&valid, &[-1, 1, 1, pool_shape[2]]);
            let valid_for_sum = mlxcel_core::reshape(&valid, &[-1, pool_shape[2]]);
            let valid_f32 = mlxcel_core::astype(&valid_for_sum, mlxcel_core::dtype::FLOAT32);
            let denom = mlxcel_core::sum_axis(&valid_f32, -1, true);
            // Clamp denom to at least 1
            let ones = mlxcel_core::ones(&[1, 1], mlxcel_core::dtype::FLOAT32);
            let denom = mlxcel_core::maximum(&denom, &ones);
            let denom = mlxcel_core::astype(&denom, mlxcel_core::array_dtype(&to_pool));
            let denom = mlxcel_core::reshape(&denom, &[-1, 1, 1]);

            // sum along pool_size axis (axis=-2 = axis=1 in 3D)
            let sum_pool = mlxcel_core::sum_axis(&to_pool, -2, true);
            let query = mlxcel_core::divide(&sum_pool, &denom);
            (query, Some(valid_flat))
        } else {
            let query = mlxcel_core::mean_axis(&to_pool, -2, true);
            (query, None)
        };

        // Cross-attention pooling
        let pooled = self.image_pooling_2d.forward(
            &query,
            Some(&to_pool),
            attn_mask.as_ref().map(|m| m.as_ref().unwrap()),
        );

        // Reshape: [B, num_pooled, hidden_size]
        let pooled_shape = mlxcel_core::array_shape(&pooled);
        let pooled_dim = pooled_shape[pooled_shape.len() - 1];
        let pooled = mlxcel_core::reshape(&pooled, &[batch_size, -1, pooled_dim]);

        // Project through SwiGLU MLP
        let projected = self.image_projector.forward(&pooled);

        // Flatten to [total_valid_tokens, output_dim]
        let proj_shape = mlxcel_core::array_shape(&projected);
        let out_dim = proj_shape[proj_shape.len() - 1];
        let projected = mlxcel_core::reshape(&projected, &[-1, out_dim]);

        // Filter valid tokens: valid_token = any(valid, axis=-1)
        // sum valid along pool_size axis, then check > 0
        let valid_sum = mlxcel_core::sum_axis(&valid_i32, -1, false);
        let zero_scalar = mlxcel_core::from_slice_i32(&[0], &[1]);
        let valid_token = mlxcel_core::greater(&valid_sum, &zero_scalar);
        let valid_flat = mlxcel_core::reshape(&valid_token, &[-1]);

        // Eval and extract valid indices on host
        mlxcel_core::eval(&valid_flat);
        let total_pooled = mlxcel_core::array_shape(&valid_flat)[0];
        let mut valid_indices: Vec<i32> = Vec::new();
        for i in 0..total_pooled {
            let idx_arr = mlxcel_core::from_slice_i32(&[i], &[1]);
            let val = mlxcel_core::take(&valid_flat, &idx_arr, 0);
            mlxcel_core::eval(&val);
            if mlxcel_core::item_bool(&val) {
                valid_indices.push(i);
            }
        }

        if valid_indices.is_empty() {
            return mlxcel_core::zeros(&[0, out_dim], mlxcel_core::array_dtype(&projected));
        }

        let indices = mlxcel_core::from_slice_i32(&valid_indices, &[valid_indices.len() as i32]);
        mlxcel_core::take(&projected, &indices, 0)
    }

    /// Batched gather: for each batch, gather from features using indices
    fn batched_gather(
        &self,
        features: &MlxArray,
        indices: &MlxArray,
        batch_size: i32,
    ) -> UniquePtr<MlxArray> {
        let idx_shape = mlxcel_core::array_shape(indices);
        let num_pooled = idx_shape[1];
        let pool_size = idx_shape[2];
        let feat_shape = mlxcel_core::array_shape(features);
        let dim = feat_shape[2];

        // Flatten indices: [B, num_pooled * pool_size]
        let flat_idx = mlxcel_core::reshape(indices, &[batch_size, num_pooled * pool_size]);

        // Build batch indices
        let mut batch_idx_data = Vec::with_capacity((batch_size * num_pooled * pool_size) as usize);
        for b in 0..batch_size {
            for _ in 0..(num_pooled * pool_size) {
                batch_idx_data.push(b);
            }
        }
        let batch_idx =
            mlxcel_core::from_slice_i32(&batch_idx_data, &[batch_size, num_pooled * pool_size]);

        // Flatten for gather
        let batch_idx_flat = mlxcel_core::reshape(&batch_idx, &[-1]);
        let flat_idx_flat = mlxcel_core::reshape(&flat_idx, &[-1]);

        // Gather: features[batch_idx, flat_idx]
        // Use advanced indexing via take
        let features_2d = mlxcel_core::reshape(features, &[batch_size * feat_shape[1], dim]);

        // Compute linear indices: batch_idx * num_patches + flat_idx
        let num_patches = mlxcel_core::from_slice_i32(&[feat_shape[1]], &[1]);
        let offset = mlxcel_core::multiply(&batch_idx_flat, &num_patches);
        let linear_idx = mlxcel_core::add(&offset, &flat_idx_flat);

        let gathered = mlxcel_core::take(&features_2d, &linear_idx, 0);

        // Reshape to [B, num_pooled, pool_size, dim]
        mlxcel_core::reshape(&gathered, &[batch_size, num_pooled, pool_size, dim])
    }

    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        vit_num_layers: usize,
        vit_hidden_size: i32,
        vit_intermediate_size: i32,
        vit_num_heads: i32,
        vit_num_kv_heads: i32,
        vit_head_dim: i32,
        vit_image_num_pos: usize,
        vit_layer_norm_eps: f32,
        vit_float32_attention: bool,
        adapter_hidden_size: i32,
        _adapter_intermediate_size: i32,
        _adapter_text_hidden_size: i32,
        adapter_num_heads: i32,
        adapter_num_kv_heads: i32,
        adapter_head_dim: i32,
        adapter_float32_attention: bool,
        vit_layers: &[i32],
        pooling_attention_mask: bool,
    ) -> Result<Self, String> {
        let image_vit = Molmo2VisionTransformer::from_weights(
            weights,
            &format!("{}.image_vit", prefix),
            vit_num_layers,
            vit_hidden_size,
            vit_intermediate_size,
            vit_num_heads,
            vit_num_kv_heads,
            vit_head_dim,
            vit_image_num_pos,
            vit_layer_norm_eps,
            vit_float32_attention,
        )?;

        // Pool dim = hidden_size * len(vit_layers)
        let _pool_dim = vit_hidden_size * vit_layers.len() as i32;

        let image_pooling_2d = ViTAttention::from_weights(
            weights,
            &format!("{}.image_pooling_2d", prefix),
            adapter_hidden_size,
            adapter_num_heads,
            adapter_num_kv_heads,
            adapter_head_dim,
            adapter_float32_attention,
        )?;

        let image_projector =
            ImageProjectorMLP::from_weights(weights, &format!("{}.image_projector", prefix))?;

        // Convert negative layer indices to positive
        let resolved_layers: Vec<usize> = vit_layers
            .iter()
            .map(|&layer| {
                if layer < 0 {
                    (layer + vit_num_layers as i32) as usize
                } else {
                    layer as usize
                }
            })
            .collect();

        Ok(Self {
            image_vit,
            image_pooling_2d,
            image_projector,
            vit_layers: resolved_layers,
            pooling_attention_mask,
        })
    }
}

// Helper Functions.
fn get_weight_copy(weights: &WeightMap, name: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(name)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Weight not found: {}", name))
}
