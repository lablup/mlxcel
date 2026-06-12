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

//! Molmo v1 Vision Encoder + Adapter (CLIP-style ViT).
//!
//! Differs from the Molmo2 vision tower (`encoders::molmo2`) in several ways
//! that justify a dedicated implementation rather than reuse:
//! - The ViT linear layers (wq/wk/wv/wo, feed_forward w1/w2) are **4-bit
//!   quantized** in Molmo-7B, so they use `UnifiedLinear` (Molmo2's tower is
//!   unquantized `Linear`).
//! - A learned `class_embedding` is prepended and a `pre_ln` LayerNorm runs
//!   before the transformer stack; the cls token is stripped afterwards.
//! - Multi-layer feature extraction (`vit_layers = [-2, -9]`) concatenates two
//!   hidden states along the feature dim.
//! - Pooling is **direct spatial 2x2** (reference `attention-meanq`): features
//!   are reshaped into 2x2 patch windows and pooled with a mean query, not the
//!   processor-supplied pooling-index gather Molmo2 uses.
//! - `image_padding_embed = "pad_and_partial_pad"` adds two learned pad
//!   embeddings based on the per-patch `image_masks`.
//!
//! This implementation targets the single-image / single-crop-batch inference
//! path used by the CLI and server (`batch=1`).
//!
//! Reference: https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/molmo/vision.py

use mlxcel_core::layers::{LayerNorm, Linear, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

/// Static configuration for the Molmo v1 vision tower.
#[derive(Debug, Clone)]
pub struct MolmoVisionConfig {
    pub image_num_layers: usize,
    pub image_emb_dim: i32,
    pub image_num_heads: i32,
    pub image_num_kv_heads: i32,
    pub image_head_dim: i32,
    pub image_num_pos: usize,
    pub image_norm_eps: f32,
    /// Number of patches per side of one crop (e.g. 24 for 336/14).
    pub image_num_patch: (i32, i32),
    pub image_pooling_h: i32,
    pub image_pooling_w: i32,
    /// Negative-or-positive ViT layer indices to concatenate (e.g. [-2, -9]).
    pub vit_layers: Vec<i32>,
    pub group_size: i32,
    pub bits: i32,
}

impl Default for MolmoVisionConfig {
    fn default() -> Self {
        Self {
            image_num_layers: 23,
            image_emb_dim: 1024,
            image_num_heads: 16,
            image_num_kv_heads: 16,
            image_head_dim: 64,
            image_num_pos: 577,
            image_norm_eps: 1e-5,
            image_num_patch: (24, 24),
            image_pooling_h: 2,
            image_pooling_w: 2,
            vit_layers: vec![-2, -9],
            group_size: 64,
            bits: 4,
        }
    }
}

impl MolmoVisionConfig {
    /// Pooled patches-per-crop along each axis (round-up division).
    fn llm_patches_per_crop(&self) -> (i32, i32) {
        let (h, w) = self.image_num_patch;
        let ph = (h + self.image_pooling_h - 1) / self.image_pooling_h;
        let pw = (w + self.image_pooling_w - 1) / self.image_pooling_w;
        (ph, pw)
    }
}

// ViT MLP (quantized w1/w2 + fast GELU).
struct MolmoViTMLP {
    w1: UnifiedLinear,
    w2: UnifiedLinear,
}

impl MolmoViTMLP {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.w1.forward(x);
        let h = mlxcel_core::gelu_approx(&h);
        self.w2.forward(&h)
    }

    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let w1 = UnifiedLinear::from_weights(weights, &format!("{}.w1", prefix), group_size, bits)?;
        let w2 = UnifiedLinear::from_weights(weights, &format!("{}.w2", prefix), group_size, bits)?;
        Ok(Self { w1, w2 })
    }
}

// ViT Multi-Head Dot Product Attention (quantized; supports cross-attention).
struct MolmoViTAttention {
    wq: UnifiedLinear,
    wk: UnifiedLinear,
    wv: UnifiedLinear,
    wo: UnifiedLinear,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    embed_dim: i32,
    scale: f32,
}

impl MolmoViTAttention {
    /// Self-attention when `inputs_kv` is None, cross-attention otherwise.
    /// Float32 attention for numerical stability (matches reference SDPA).
    fn forward(&self, inputs_q: &MlxArray, inputs_kv: Option<&MlxArray>) -> UniquePtr<MlxArray> {
        let inputs_k = inputs_kv.unwrap_or(inputs_q);
        let inputs_v = inputs_kv.unwrap_or(inputs_q);

        let xq = self.wq.forward(inputs_q);
        let xk = self.wk.forward(inputs_k);
        let xv = self.wv.forward(inputs_v);

        let q_shape = mlxcel_core::array_shape(&xq);
        let bsz = q_shape[0];
        let q_len = q_shape[1];
        let kv_len = mlxcel_core::array_shape(&xk)[1];

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

        let dtype = mlxcel_core::array_dtype(inputs_q);
        let q = mlxcel_core::astype(&q, mlxcel_core::dtype::FLOAT32);
        let k = mlxcel_core::astype(&k, mlxcel_core::dtype::FLOAT32);
        let v = mlxcel_core::astype(&v, mlxcel_core::dtype::FLOAT32);

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
        let out = mlxcel_core::astype(&out, dtype);

        let out = mlxcel_core::transpose_axes(&out, &[0, 2, 1, 3]);
        let out = mlxcel_core::reshape(&out, &[bsz, q_len, self.embed_dim]);
        self.wo.forward(&out)
    }

    #[allow(clippy::too_many_arguments)]
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        embed_dim: i32,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let wq = UnifiedLinear::from_weights(weights, &format!("{}.wq", prefix), group_size, bits)?;
        let wk = UnifiedLinear::from_weights(weights, &format!("{}.wk", prefix), group_size, bits)?;
        let wv = UnifiedLinear::from_weights(weights, &format!("{}.wv", prefix), group_size, bits)?;
        let wo = UnifiedLinear::from_weights(weights, &format!("{}.wo", prefix), group_size, bits)?;
        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            num_heads,
            num_kv_heads,
            head_dim,
            embed_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }
}

// ViT residual block (pre-norm; LayerNorm with bias).
struct MolmoVisionBlock {
    attention: MolmoViTAttention,
    feed_forward: MolmoViTMLP,
    attention_norm: LayerNorm,
    ffn_norm: LayerNorm,
}

impl MolmoVisionBlock {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let normed = self.attention_norm.forward(x);
        let attn_out = self.attention.forward(&normed, None);
        let h = mlxcel_core::add(x, &attn_out);

        let normed = self.ffn_norm.forward(&h);
        let mlp_out = self.feed_forward.forward(&normed);
        mlxcel_core::add(&h, &mlp_out)
    }

    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &MolmoVisionConfig,
    ) -> Result<Self, String> {
        let attention = MolmoViTAttention::from_weights(
            weights,
            &format!("{}.attention", prefix),
            config.image_num_heads,
            config.image_num_kv_heads,
            config.image_head_dim,
            config.image_emb_dim,
            config.group_size,
            config.bits,
        )?;
        let feed_forward = MolmoViTMLP::from_weights(
            weights,
            &format!("{}.feed_forward", prefix),
            config.group_size,
            config.bits,
        )?;

        let attn_norm_w = get_weight_copy(weights, &format!("{}.attention_norm.weight", prefix))?;
        let attn_norm_b = weights
            .get(&format!("{}.attention_norm.bias", prefix))
            .map(|w| mlxcel_core::copy(w));
        let ffn_norm_w = get_weight_copy(weights, &format!("{}.ffn_norm.weight", prefix))?;
        let ffn_norm_b = weights
            .get(&format!("{}.ffn_norm.bias", prefix))
            .map(|w| mlxcel_core::copy(w));

        Ok(Self {
            attention,
            feed_forward,
            attention_norm: LayerNorm::new(attn_norm_w, attn_norm_b, config.image_norm_eps),
            ffn_norm: LayerNorm::new(ffn_norm_w, ffn_norm_b, config.image_norm_eps),
        })
    }
}

// Vision Transformer: class token + pos emb + pre_ln + resblocks.
struct MolmoVisionTransformer {
    class_embedding: UniquePtr<MlxArray>,      // [image_emb_dim]
    positional_embedding: UniquePtr<MlxArray>, // [image_num_pos, image_emb_dim]
    patch_embedding: Linear,                   // non-quantized Linear
    pre_ln: LayerNorm,
    blocks: Vec<MolmoVisionBlock>,
    intermediate_size: i32, // patch flatten dim (patch*patch*3)
}

impl MolmoVisionTransformer {
    /// Add positional embedding (cls slot + grid slots), matching the reference
    /// `add_pos_emb`. For the default crop size the grid matches `image_num_pos
    /// - 1` exactly, so no interpolation is needed.
    fn add_pos_emb(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // positional_embedding[None, :, :] broadcast over batch.
        let pe_shape = mlxcel_core::array_shape(&self.positional_embedding);
        let num_pos = pe_shape[0];
        let emb_dim = pe_shape[1];
        let pe = mlxcel_core::reshape(&self.positional_embedding, &[1, num_pos, emb_dim]);
        let pe = mlxcel_core::astype(&pe, mlxcel_core::array_dtype(x));
        mlxcel_core::add(x, &pe)
    }

    fn forward(&self, x: &MlxArray) -> Vec<UniquePtr<MlxArray>> {
        // x: [B, num_patch, n_pixels]. Pad last dim up to intermediate_size if
        // the processor produced a narrower patch (quantization padding).
        let shape = mlxcel_core::array_shape(x);
        let cur = shape[shape.len() - 1];
        let x = if cur < self.intermediate_size {
            let pad_width = [0, 0, 0, 0, 0, self.intermediate_size - cur];
            mlxcel_core::pad(x, &pad_width, 0.0)
        } else {
            mlxcel_core::copy(x)
        };

        // Cast pixels to the patch-embedding weight dtype (f16 on Apple Silicon)
        // before the matmul, mirroring the reference `pixel_values.astype(dtype)`.
        let weight_dtype = mlxcel_core::array_dtype(&self.class_embedding);
        let x = if mlxcel_core::array_dtype(&x) != weight_dtype {
            mlxcel_core::astype(&x, weight_dtype)
        } else {
            x
        };

        let x = self.patch_embedding.forward(&x);
        let bsz = mlxcel_core::array_shape(&x)[0];

        // Prepend class embedding: broadcast [emb] -> [B, 1, emb].
        let emb_dim = mlxcel_core::array_shape(&self.class_embedding)[0];
        let cls = mlxcel_core::reshape(&self.class_embedding, &[1, 1, emb_dim]);
        let cls = mlxcel_core::broadcast_to(&cls, &[bsz, 1, emb_dim]);
        let cls = mlxcel_core::astype(&cls, mlxcel_core::array_dtype(&x));
        let x = mlxcel_core::concatenate(&cls, &x, 1);

        let x = self.add_pos_emb(&x);
        let mut x = self.pre_ln.forward(&x);

        let mut hidden_states = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            x = block.forward(&x);
            hidden_states.push(mlxcel_core::copy(&x));
        }
        hidden_states
    }

    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &MolmoVisionConfig,
        num_layers: usize,
    ) -> Result<Self, String> {
        let class_embedding = get_weight_copy(weights, &format!("{}.class_embedding", prefix))?;
        let positional_embedding =
            get_weight_copy(weights, &format!("{}.positional_embedding", prefix))?;
        let patch_embedding =
            Linear::from_weights(weights, &format!("{}.patch_embedding", prefix))?;

        let pre_ln_w = get_weight_copy(weights, &format!("{}.pre_ln.weight", prefix))?;
        let pre_ln_b = weights
            .get(&format!("{}.pre_ln.bias", prefix))
            .map(|w| mlxcel_core::copy(w));
        let pre_ln = LayerNorm::new(pre_ln_w, pre_ln_b, config.image_norm_eps);

        let mut blocks = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            blocks.push(MolmoVisionBlock::from_weights(
                weights,
                &format!("{}.transformer.resblocks.{}", prefix, i),
                config,
            )?);
        }

        Ok(Self {
            class_embedding,
            positional_embedding,
            patch_embedding,
            pre_ln,
            blocks,
            // patch_size^2 * 3 = 14*14*3 = 588 for the default Molmo-7B patch.
            intermediate_size: 14 * 14 * 3,
        })
    }
}

// Image Projector MLP (quantized SwiGLU: silu(w1(x)) * w3(x) -> w2).
struct MolmoImageProjector {
    w1: UnifiedLinear,
    w2: UnifiedLinear,
    w3: UnifiedLinear,
}

impl MolmoImageProjector {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let gate = self.w1.forward(x);
        let gate = mlxcel_core::silu(&gate);
        let up = self.w3.forward(x);
        let h = mlxcel_core::multiply(&gate, &up);
        self.w2.forward(&h)
    }

    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let w1 = UnifiedLinear::from_weights(weights, &format!("{}.w1", prefix), group_size, bits)?;
        let w2 = UnifiedLinear::from_weights(weights, &format!("{}.w2", prefix), group_size, bits)?;
        let w3 = UnifiedLinear::from_weights(weights, &format!("{}.w3", prefix), group_size, bits)?;
        Ok(Self { w1, w2, w3 })
    }
}

/// Molmo v1 Vision Model (CLIP ViT + attention pooling + SwiGLU projector).
pub struct MolmoVisionModel {
    image_vit: MolmoVisionTransformer,
    image_pooling_2d: MolmoViTAttention,
    image_projector: MolmoImageProjector,
    config: MolmoVisionConfig,
    /// Resolved (non-negative) ViT layer indices to extract.
    vit_layers: Vec<usize>,
    pad_embed: Option<UniquePtr<MlxArray>>, // [2, image_emb_dim * len(vit_layers)]
    num_prefix_tokens: usize,
}

impl MolmoVisionModel {
    /// Run the ViT over `[B, T, N, D]` crops, select+concat `vit_layers`, strip
    /// the cls token, and mask all-padding crops. Returns `[B, T, N_patch,
    /// feat_dim]` where `feat_dim = image_emb_dim * len(vit_layers)`.
    fn encode_image(&self, images: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(images);
        let b = shape[0];
        let t = shape[1];
        let n = shape[2];
        let d = shape[3];

        let flat = mlxcel_core::reshape(images, &[b * t, n, d]);

        // mask = ~all(crop == -1): an all -1 crop is padding.
        let neg_one = mlxcel_core::from_slice_f32(&[-1.0], &[1]);
        let is_neg = mlxcel_core::equal(&flat, &neg_one);
        let is_neg_i = mlxcel_core::astype(&is_neg, mlxcel_core::dtype::INT32);
        // sum over (patch, pixel); if equal to n*d the crop is all padding.
        let sum_nd = mlxcel_core::sum_axis(&is_neg_i, 2, false); // [B*T, N]
        let sum_n = mlxcel_core::sum_axis(&sum_nd, 1, false); // [B*T]
        let total = mlxcel_core::from_slice_i32(&[n * d], &[1]);
        let all_pad = mlxcel_core::greater_equal(&sum_n, &total); // [B*T] bool
        let not_pad = {
            let one = mlxcel_core::from_slice_i32(&[1], &[1]);
            let all_pad_i = mlxcel_core::astype(&all_pad, mlxcel_core::dtype::INT32);
            mlxcel_core::subtract(&one, &all_pad_i) // 1 if valid, 0 if padding
        };

        let hidden_states = self.image_vit.forward(&flat);

        // Select and concat the requested ViT layers along the feature dim.
        let mut image_features =
            mlxcel_core::copy(hidden_states[self.vit_layers[0]].as_ref().unwrap());
        for &layer in &self.vit_layers[1..] {
            image_features = mlxcel_core::concatenate(
                &image_features,
                hidden_states[layer].as_ref().unwrap(),
                -1,
            );
        }

        // Strip cls token (num_prefix_tokens == 1).
        let feat_shape = mlxcel_core::array_shape(&image_features);
        let seq = feat_shape[1];
        let feat_dim = feat_shape[2];
        let image_features = if self.num_prefix_tokens > 0 {
            mlxcel_core::slice(
                &image_features,
                &[0, self.num_prefix_tokens as i32, 0],
                &[b * t, seq, feat_dim],
            )
        } else {
            image_features
        };

        // Zero out features of all-padding crops: features * not_pad[:, None, None].
        let not_pad_f = mlxcel_core::astype(&not_pad, mlxcel_core::array_dtype(&image_features));
        let not_pad_f = mlxcel_core::reshape(&not_pad_f, &[b * t, 1, 1]);
        let image_features = mlxcel_core::multiply(&image_features, &not_pad_f);

        mlxcel_core::reshape(&image_features, &[b, t, n, feat_dim])
    }

    /// Apply `pad_and_partial_pad` learned pad embeddings using `image_masks`.
    /// `image_features`: [B, T, N_patch, feat_dim]; `image_masks`: [B, T, N_patch].
    fn apply_pad_embed(
        &self,
        image_features: UniquePtr<MlxArray>,
        image_masks: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let Some(pad_embed) = self.pad_embed.as_ref() else {
            return image_features;
        };
        let dtype = mlxcel_core::array_dtype(&image_features);

        // pad_embed: [2, feat_dim] -> two [1,1,1,feat_dim] vectors.
        let pe_shape = mlxcel_core::array_shape(pad_embed);
        let feat_dim = pe_shape[1];
        let pad0 = mlxcel_core::slice(pad_embed, &[0, 0], &[1, feat_dim]);
        let pad1 = mlxcel_core::slice(pad_embed, &[1, 0], &[2, feat_dim]);
        let pad0 = mlxcel_core::reshape(&pad0, &[1, 1, 1, feat_dim]);
        let pad1 = mlxcel_core::reshape(&pad1, &[1, 1, 1, feat_dim]);
        let pad0 = mlxcel_core::astype(&pad0, dtype);
        let pad1 = mlxcel_core::astype(&pad1, dtype);

        // all_pad = (mask == 0); partial_pad = (mask < 1) & !all_pad.
        let zero = mlxcel_core::from_slice_f32(&[0.0], &[1]);
        let one = mlxcel_core::from_slice_f32(&[1.0], &[1]);
        let all_pad = mlxcel_core::equal(image_masks, &zero);
        let lt_one = mlxcel_core::less(image_masks, &one);
        let not_all_pad = mlxcel_core::logical_not(&all_pad);
        let partial_pad = mlxcel_core::logical_and(&lt_one, &not_all_pad);

        let all_pad_f = mlxcel_core::astype(&all_pad, dtype);
        let partial_pad_f = mlxcel_core::astype(&partial_pad, dtype);
        let all_pad_f = mlxcel_core::expand_dims(&all_pad_f, -1); // [B,T,N,1]
        let partial_pad_f = mlxcel_core::expand_dims(&partial_pad_f, -1);

        let add0 = mlxcel_core::multiply(&pad0, &all_pad_f);
        let add1 = mlxcel_core::multiply(&pad1, &partial_pad_f);
        let h = mlxcel_core::add(&image_features, &add0);
        mlxcel_core::add(&h, &add1)
    }

    /// Full forward: encode -> pad_embed -> spatial 2x2 pool (attention-meanq)
    /// -> projector. Returns `[B, num_image, h*w, text_hidden]`.
    pub fn forward(&self, images: &MlxArray, image_masks: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(images);
        let batch_size = shape[0];
        let num_image = shape[1];

        let cfg = &self.config;
        let mut image_features = self.encode_image(images);
        image_features = self.apply_pad_embed(image_features, image_masks);

        let feat_dim = mlxcel_core::array_shape(&image_features);
        let feat_dim = feat_dim[feat_dim.len() - 1];

        let (patch_h, patch_w) = cfg.image_num_patch;
        // [B, num_image, patch_h, patch_w, feat_dim]
        let mut image_features = mlxcel_core::reshape(
            &image_features,
            &[batch_size, num_image, patch_h, patch_w, feat_dim],
        );

        // Pad an odd patch grid so 2x2 pooling tiles evenly.
        let (mut grid_h, mut grid_w) = (patch_h, patch_w);
        if patch_h % cfg.image_pooling_h == 1 {
            let pad_width = [0, 0, 0, 0, 0, 1, 0, 1, 0, 0];
            image_features = mlxcel_core::pad(&image_features, &pad_width, 0.0);
            grid_h += 1;
            grid_w += 1;
        }

        let h_blocks = grid_h / cfg.image_pooling_h;
        let w_blocks = grid_w / cfg.image_pooling_w;

        // Rearrange to [B*num_image*h_blocks*w_blocks, pool_h*pool_w, feat_dim].
        let image_features = mlxcel_core::reshape(
            &image_features,
            &[
                batch_size,
                num_image,
                h_blocks,
                cfg.image_pooling_h,
                w_blocks,
                cfg.image_pooling_w,
                feat_dim,
            ],
        );
        let image_features = mlxcel_core::transpose_axes(&image_features, &[0, 1, 2, 4, 3, 5, 6]);
        let image_features = mlxcel_core::reshape(
            &image_features,
            &[
                batch_size * num_image * h_blocks * w_blocks,
                cfg.image_pooling_h * cfg.image_pooling_w,
                feat_dim,
            ],
        );

        // attention-meanq: query = mean over the pooled patches.
        let query = mlxcel_core::mean_axis(&image_features, -2, true);
        let pooled = self.image_pooling_2d.forward(&query, Some(&image_features));

        let (lh, lw) = cfg.llm_patches_per_crop();
        let pooled_dim = {
            let s = mlxcel_core::array_shape(&pooled);
            s[s.len() - 1]
        };
        let pooled = mlxcel_core::reshape(&pooled, &[batch_size, num_image, lh * lw, pooled_dim]);

        // Project to the text hidden dim.
        self.image_projector.forward(&pooled)
    }

    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: MolmoVisionConfig,
    ) -> Result<Self, String> {
        // Resolve negative ViT layer indices; cap the loaded ViT depth to the
        // deepest layer actually consumed (mirrors molmo_point's truncation).
        let resolved: Vec<usize> = config
            .vit_layers
            .iter()
            .map(|&l| {
                if l < 0 {
                    (l + config.image_num_layers as i32) as usize
                } else {
                    l as usize
                }
            })
            .collect();
        let last_needed = *resolved.iter().max().unwrap_or(&0) + 1;
        let num_layers = last_needed.min(config.image_num_layers);

        let image_vit = MolmoVisionTransformer::from_weights(
            weights,
            &format!("{}.image_vit", prefix),
            &config,
            num_layers,
        )?;

        // Pooling attention input is len(vit_layers) * image_emb_dim wide.
        let image_pooling_2d = MolmoViTAttention::from_weights(
            weights,
            &format!("{}.image_pooling_2d", prefix),
            config.image_num_heads,
            config.image_num_kv_heads,
            config.image_head_dim,
            config.image_emb_dim,
            config.group_size,
            config.bits,
        )?;

        let image_projector = MolmoImageProjector::from_weights(
            weights,
            &format!("{}.image_projector", prefix),
            config.group_size,
            config.bits,
        )?;

        let pad_embed = weights
            .get(&format!("{}.pad_embed", prefix))
            .map(|w| mlxcel_core::copy(w));

        Ok(Self {
            image_vit,
            image_pooling_2d,
            image_projector,
            config,
            vit_layers: resolved,
            pad_embed,
            num_prefix_tokens: 1,
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
