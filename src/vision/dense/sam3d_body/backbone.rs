// Copyright 2025-2026 Lablup Inc.
// SPDX-License-Identifier: Apache-2.0
//! Pinned SAM 3D Body DINOv3-H+ backbone, including storage tokens and axial RoPE.
use super::config::Config;
use super::layers::{
    Array, LayerScale, Linear, Norm, SwiGlu, as_dtype, floating_array as array, linear, shape,
    upstream_dtype,
};
use super::rope::{Rope, apply_rope};
use anyhow::{Result, ensure};
use mlxcel_core::{self as mx, MlxArray, weights::WeightMap};

struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    heads: i32,
    head_dim: i32,
}
impl Attention {
    fn from_weights(w: &WeightMap, prefix: &str, c: &Config) -> Result<Self> {
        Ok(Self {
            query: linear(
                w,
                &format!("{prefix}.q_proj"),
                c.embed_dim,
                c.embed_dim,
                true,
            )?,
            key: linear(
                w,
                &format!("{prefix}.k_proj"),
                c.embed_dim,
                c.embed_dim,
                false,
            )?,
            value: linear(
                w,
                &format!("{prefix}.v_proj"),
                c.embed_dim,
                c.embed_dim,
                true,
            )?,
            output: linear(
                w,
                &format!("{prefix}.o_proj"),
                c.embed_dim,
                c.embed_dim,
                true,
            )?,
            heads: c.num_heads as i32,
            head_dim: c.head_dim as i32,
        })
    }
    fn forward(&self, x: &MlxArray, sin: &MlxArray, cos: &MlxArray) -> Result<Array> {
        let dims = mx::array_shape(x);
        let split = |x: &MlxArray| {
            mx::transpose_axes(
                &mx::reshape(x, &[dims[0], dims[1], self.heads, self.head_dim]),
                &[0, 2, 1, 3],
            )
        };
        let q = split(&self.query.forward(x));
        let k = split(&self.key.forward(x));
        let v = split(&self.value.forward(x));
        let patches = mx::array_shape(sin)[0];
        ensure!(dims[1] > patches, "Body attention requires CLS prefix");
        let (q, k) = apply_rope(&q, &k, sin, cos, (dims[1] - patches) as usize)?;
        let scale = (self.head_dim as f64).powf(-0.5) as f32;
        // SAFETY: projections and RoPE produce compatible BHND arrays;
        // null denotes the upstream unmasked fast SDPA call.
        let out =
            unsafe { mx::fast_scaled_dot_product_attention(&q, &k, &v, scale, std::ptr::null()) };
        let out = mx::reshape(&mx::transpose_axes(&out, &[0, 2, 1, 3]), &dims);
        Ok(self.output.forward(&out))
    }
}

struct Block {
    norm1: Norm,
    attention: Attention,
    ls1: LayerScale,
    norm2: Norm,
    mlp: SwiGlu,
    ls2: LayerScale,
}
impl Block {
    fn from_weights(w: &WeightMap, prefix: &str, c: &Config, hidden: usize) -> Result<Self> {
        let norm1 = Norm::from_weights(w, &format!("{prefix}.norm1"), 1e-5, false)?;
        let norm2 = Norm::from_weights(w, &format!("{prefix}.norm2"), 1e-5, false)?;
        shape(
            &norm1.weight,
            &[c.embed_dim as i32],
            &format!("{prefix}.norm1"),
        )?;
        shape(
            &norm2.weight,
            &[c.embed_dim as i32],
            &format!("{prefix}.norm2"),
        )?;
        Ok(Self {
            norm1,
            norm2,
            attention: Attention::from_weights(w, &format!("{prefix}.attention"), c)?,
            ls1: LayerScale::from_weights(w, &format!("{prefix}.ls1"), c.embed_dim)?,
            ls2: LayerScale::from_weights(w, &format!("{prefix}.ls2"), c.embed_dim)?,
            mlp: SwiGlu::from_weights(w, &format!("{prefix}.mlp"), c.embed_dim, hidden)?,
        })
    }
    fn forward(&self, x: &MlxArray, sin: &MlxArray, cos: &MlxArray) -> Result<Array> {
        let h = self.attention.forward(&self.norm1.forward(x), sin, cos)?;
        let x = mx::add(x, &self.ls1.forward(&h));
        Ok(mx::add(
            &x,
            &self.ls2.forward(&self.mlp.forward(&self.norm2.forward(&x))),
        ))
    }
}

pub struct Backbone {
    projection: Array,
    projection_bias: Array,
    cls: Array,
    storage: Array,
    rope: Rope,
    blocks: Vec<Block>,
    norm: Norm,
    patch_size: i32,
    channels: i32,
    storage_count: i32,
    image_size: [usize; 2],
}
impl Backbone {
    pub fn from_weights(w: &WeightMap, c: &Config) -> Result<Self> {
        ensure!(
            c.embed_dim > 0 && c.embed_dim <= i32::MAX as usize && c.depth > 0,
            "invalid Body backbone width/depth"
        );
        ensure!(
            c.num_heads > 0
                && c.head_dim > 0
                && c.num_heads.checked_mul(c.head_dim) == Some(c.embed_dim),
            "Body heads do not match embedding width"
        );
        ensure!(
            c.patch_size > 0
                && c.patch_size <= i32::MAX as usize
                && c.image_size
                    .iter()
                    .all(|&v| v > 0 && v <= i32::MAX as usize && v % c.patch_size == 0),
            "invalid Body backbone image/patch dimensions"
        );
        ensure!(
            c.num_storage_tokens < i32::MAX as usize,
            "Body storage token count overflows"
        );
        let hidden = c.embed_dim as f64 * c.ffn_ratio;
        ensure!(
            hidden.is_finite() && hidden >= 1. && hidden < i32::MAX as f64,
            "invalid Body SwiGLU hidden width"
        );
        ensure!(
            c.rope_periods == c.head_dim / 4,
            "Body RoPE period count must equal head_dim / 4"
        );
        let channels = c.embed_dim as i32;
        let patch_size = c.patch_size as i32;
        let projection = array(w, "backbone.patch_embed.projection.weight")?;
        shape(
            &projection,
            &[channels, patch_size, patch_size, 3],
            "backbone.patch_embed.projection.weight",
        )?;
        let projection_bias = array(w, "backbone.patch_embed.projection.bias")?;
        shape(
            &projection_bias,
            &[channels],
            "backbone.patch_embed.projection.bias",
        )?;
        let cls = array(w, "backbone.cls_token")?;
        shape(&cls, &[1, 1, channels], "backbone.cls_token")?;
        let storage = array(w, "backbone.storage_tokens")?;
        shape(
            &storage,
            &[1, c.num_storage_tokens as i32, channels],
            "backbone.storage_tokens",
        )?;
        let rope = Rope::from_weights(w, "backbone.rope_embed", c.head_dim)?;
        let mut blocks = Vec::new();
        for index in 0..c.depth {
            blocks.push(Block::from_weights(
                w,
                &format!("backbone.blocks.{index}"),
                c,
                hidden as usize,
            )?);
        }
        let norm = Norm::from_weights(w, "backbone.norm", 1e-5, false)?;
        shape(&norm.weight, &[channels], "backbone.norm")?;
        Ok(Self {
            projection,
            projection_bias,
            cls,
            storage,
            rope,
            blocks,
            norm,
            patch_size,
            channels,
            storage_count: c.num_storage_tokens as i32,
            image_size: c.image_size,
        })
    }
    pub fn forward(&self, image: &MlxArray) -> Result<Array> {
        let dims = mx::array_shape(image);
        ensure!(
            dims.len() == 4
                && dims[0] > 0
                && dims[1] as usize == self.image_size[0]
                && dims[2] as usize == self.image_size[1]
                && dims[3] == 3,
            "Body backbone expects configured NHWC RGB image, got {dims:?}"
        );
        ensure!(
            mx::array_dtype(image) == mx::dtype::FLOAT32,
            "Body backbone input must be F32"
        );
        let (batch, height, width) = (
            dims[0],
            dims[1] / self.patch_size,
            dims[2] / self.patch_size,
        );
        let patches = height
            .checked_mul(width)
            .ok_or_else(|| anyhow::anyhow!("Body patch count overflows"))?;
        let tokens = patches
            .checked_add(1)
            .and_then(|n| n.checked_add(self.storage_count))
            .ok_or_else(|| anyhow::anyhow!("Body token count overflows"))?;
        ensure!(
            batch
                .checked_mul(tokens)
                .and_then(|n| n.checked_mul(self.channels))
                .is_some(),
            "Body token tensor exceeds MLX dimensions"
        );
        // Do not downcast input to checkpoint dtype. Upstream passes F32 image
        // into the BF16 convolution; MLX promotion determines activation dtype.
        // Cast to upstream's promoted dtype explicitly, since the CUDA build's
        // promotion table would resolve BF16 with F32 to BF16 (issue #2739).
        let dtype = upstream_dtype(&[
            image,
            &self.projection,
            &self.projection_bias,
            &self.cls,
            &self.storage,
        ]);
        let x = mx::try_conv2d(
            image,
            &as_dtype(&self.projection, dtype),
            self.patch_size,
            self.patch_size,
            0,
            0,
            1,
            1,
            1,
        )?;
        let x = mx::reshape(
            &mx::add(&x, &as_dtype(&self.projection_bias, dtype)),
            &[batch, patches, self.channels],
        );
        let cls = mx::broadcast_to(&as_dtype(&self.cls, dtype), &[batch, 1, self.channels]);
        let storage = mx::broadcast_to(
            &as_dtype(&self.storage, dtype),
            &[batch, self.storage_count, self.channels],
        );
        let mut x = mx::concatenate(&mx::concatenate(&cls, &storage, 1), &x, 1);
        let (sin, cos) = self.rope.forward(height as usize, width as usize)?;
        for block in &self.blocks {
            x = block.forward(&x, &sin, &cos)?;
        }
        let x = self.norm.forward(&x);
        let x = mx::slice(
            &x,
            &[0, 1 + self.storage_count, 0],
            &[batch, tokens, self.channels],
        );
        Ok(mx::reshape(&x, &[batch, height, width, self.channels]))
    }
}

#[cfg(test)]
#[path = "backbone_tests.rs"]
mod tests;
