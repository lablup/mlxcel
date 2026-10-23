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
//! SAM3 ViT-L image encoder and feature-pyramid neck (NHWC).
use mlxcel_core::layers::LayerNorm;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::linear::UnifiedLinear;

use super::config::{VisionEncoderConfig, VitConfig};

mod fpn;
pub use fpn::FpnNeck;

fn weight(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(key)
        .map(|value| mlxcel_core::copy(value))
        .ok_or_else(|| format!("SAM3 weight not found: {key}"))
}

fn layer_norm(weights: &WeightMap, prefix: &str, eps: f32) -> Result<LayerNorm, String> {
    Ok(LayerNorm::new(
        weight(weights, &format!("{prefix}.weight"))?,
        weights
            .get(&format!("{prefix}.bias"))
            .map(|value| mlxcel_core::copy(value)),
        eps,
    ))
}

struct Conv2d {
    weight: UniquePtr<MlxArray>,
    bias: Option<UniquePtr<MlxArray>>,
    stride: i32,
    padding: i32,
}

impl Conv2d {
    fn load(weights: &WeightMap, prefix: &str, stride: i32, padding: i32) -> Result<Self, String> {
        Ok(Self {
            weight: weight(weights, &format!("{prefix}.weight"))?,
            bias: weights
                .get(&format!("{prefix}.bias"))
                .map(|value| mlxcel_core::copy(value)),
            stride,
            padding,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let y = mlxcel_core::conv2d(
            x,
            &self.weight,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            1,
            1,
            1,
        );
        match &self.bias {
            Some(bias) => mlxcel_core::add(&y, bias),
            None => y,
        }
    }
}

struct ConvTranspose2d {
    weight: UniquePtr<MlxArray>,
    bias: Option<UniquePtr<MlxArray>>,
    stride: i32,
}

impl ConvTranspose2d {
    fn load(weights: &WeightMap, prefix: &str, stride: i32) -> Result<Self, String> {
        Ok(Self {
            weight: weight(weights, &format!("{prefix}.weight"))?,
            bias: weights
                .get(&format!("{prefix}.bias"))
                .map(|value| mlxcel_core::copy(value)),
            stride,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let y = mlxcel_core::conv_transpose2d(
            x,
            &self.weight,
            self.stride,
            self.stride,
            0,
            0,
            1,
            1,
            0,
            0,
            1,
        );
        match &self.bias {
            Some(bias) => mlxcel_core::add(&y, bias),
            None => y,
        }
    }
}

fn axial_rope(
    dim: i32,
    width: i32,
    height: i32,
    theta: f32,
    scale: f32,
) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let quarter = dim / 4;
    let mut cos = Vec::with_capacity((width * height * dim) as usize);
    let mut sin = Vec::with_capacity(cos.capacity());
    for index in 0..width * height {
        let x = (index % width) as f32 * scale;
        let y = (index / width) as f32 * scale;
        for coord in [x, y] {
            for i in 0..quarter {
                let angle = coord / theta.powf((4 * i) as f32 / dim as f32);
                cos.extend_from_slice(&[angle.cos(), angle.cos()]);
                sin.extend_from_slice(&[angle.sin(), angle.sin()]);
            }
        }
    }
    (
        mlxcel_core::from_slice_f32(&cos, &[width * height, dim]),
        mlxcel_core::from_slice_f32(&sin, &[width * height, dim]),
    )
}

fn rotate_pairwise(x: &MlxArray) -> UniquePtr<MlxArray> {
    let s = mlxcel_core::array_shape(x);
    let (b, heads, n, dim) = (s[0], s[1], s[2], s[3]);
    let pairs = mlxcel_core::reshape(x, &[b, heads, n, dim / 2, 2]);
    let even = mlxcel_core::slice(&pairs, &[0, 0, 0, 0, 0], &[b, heads, n, dim / 2, 1]);
    let odd = mlxcel_core::slice(&pairs, &[0, 0, 0, 0, 1], &[b, heads, n, dim / 2, 2]);
    let even = mlxcel_core::squeeze_axis(&even, 4);
    let odd = mlxcel_core::multiply_scalar(&mlxcel_core::squeeze_axis(&odd, 4), -1.0);
    mlxcel_core::reshape(
        &mlxcel_core::stack_owned(&[odd, even], -1),
        &[b, heads, n, dim],
    )
}

fn apply_rope(x: &MlxArray, cos: &MlxArray, sin: &MlxArray) -> UniquePtr<MlxArray> {
    mlxcel_core::add(
        &mlxcel_core::multiply(x, cos),
        &mlxcel_core::multiply(&rotate_pairwise(x), sin),
    )
}

struct Attention {
    q: UnifiedLinear,
    k: UnifiedLinear,
    v: UnifiedLinear,
    out: UnifiedLinear,
    heads: i32,
    head_dim: i32,
    scale: f32,
}

impl Attention {
    fn load(
        weights: &WeightMap,
        prefix: &str,
        cfg: &VitConfig,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let linear = |name: &str| {
            UnifiedLinear::from_weights(weights, &format!("{prefix}.{name}"), gs, bits)
        };
        let head_dim = (cfg.hidden_size / cfg.num_attention_heads) as i32;
        Ok(Self {
            q: linear("q_proj")?,
            k: linear("k_proj")?,
            v: linear("v_proj")?,
            out: linear("o_proj")?,
            heads: cfg.num_attention_heads as i32,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn forward(&self, x: &MlxArray, cos: &MlxArray, sin: &MlxArray) -> UniquePtr<MlxArray> {
        let s = mlxcel_core::array_shape(x);
        let (b, h, w, d) = (s[0], s[1], s[2], s[3]);
        let n = h * w;
        let flat = mlxcel_core::reshape(x, &[b, n, d]);
        let split = |linear: &UnifiedLinear| {
            let y = linear.forward(&flat);
            let y = mlxcel_core::reshape(&y, &[b, n, self.heads, self.head_dim]);
            mlxcel_core::transpose_axes(&y, &[0, 2, 1, 3])
        };
        let q = apply_rope(&split(&self.q), cos, sin);
        let k = apply_rope(&split(&self.k), cos, sin);
        let v = split(&self.v);
        // SAFETY: null selects unmasked attention and all arrays remain live.
        let y = unsafe {
            mlxcel_core::fast_scaled_dot_product_attention(&q, &k, &v, self.scale, std::ptr::null())
        };
        let y = mlxcel_core::transpose_axes(&y, &[0, 2, 1, 3]);
        let y = mlxcel_core::reshape(&y, &[b, n, d]);
        mlxcel_core::reshape(&self.out.forward(&y), &[b, h, w, d])
    }
}

fn window_partition(x: &MlxArray, size: i32) -> (UniquePtr<MlxArray>, (i32, i32)) {
    let s = mlxcel_core::array_shape(x);
    let (b, h, w, c) = (s[0], s[1], s[2], s[3]);
    let (ph, pw) = ((size - h % size) % size, (size - w % size) % size);
    let x = if ph > 0 || pw > 0 {
        mlxcel_core::pad(x, &[0, 0, 0, ph, 0, pw, 0, 0], 0.0)
    } else {
        mlxcel_core::copy(x)
    };
    let (hp, wp) = (h + ph, w + pw);
    let x = mlxcel_core::reshape(&x, &[b, hp / size, size, wp / size, size, c]);
    let x = mlxcel_core::transpose_axes(&x, &[0, 1, 3, 2, 4, 5]);
    (mlxcel_core::reshape(&x, &[-1, size, size, c]), (hp, wp))
}

fn window_unpartition(
    x: &MlxArray,
    size: i32,
    padded: (i32, i32),
    original: (i32, i32),
) -> UniquePtr<MlxArray> {
    let (hp, wp) = padded;
    let (h, w) = original;
    let s = mlxcel_core::array_shape(x);
    let (b, c) = (s[0] / (hp / size * wp / size), s[3]);
    let x = mlxcel_core::reshape(x, &[b, hp / size, wp / size, size, size, c]);
    let x = mlxcel_core::transpose_axes(&x, &[0, 1, 3, 2, 4, 5]);
    let x = mlxcel_core::reshape(&x, &[b, hp, wp, c]);
    if hp == h && wp == w {
        x
    } else {
        mlxcel_core::slice(&x, &[0, 0, 0, 0], &[b, h, w, c])
    }
}

struct Block {
    norm1: LayerNorm,
    attention: Attention,
    norm2: LayerNorm,
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
    window: i32,
}

impl Block {
    fn load(
        weights: &WeightMap,
        prefix: &str,
        cfg: &VitConfig,
        global: bool,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            norm1: layer_norm(
                weights,
                &format!("{prefix}.layer_norm1"),
                cfg.layer_norm_eps,
            )?,
            attention: Attention::load(weights, &format!("{prefix}.attention"), cfg, gs, bits)?,
            norm2: layer_norm(
                weights,
                &format!("{prefix}.layer_norm2"),
                cfg.layer_norm_eps,
            )?,
            fc1: UnifiedLinear::from_weights(weights, &format!("{prefix}.mlp.fc1"), gs, bits)?,
            fc2: UnifiedLinear::from_weights(weights, &format!("{prefix}.mlp.fc2"), gs, bits)?,
            window: if global { 0 } else { cfg.window_size as i32 },
        })
    }

    fn forward(&self, x: &MlxArray, cos: &MlxArray, sin: &MlxArray) -> UniquePtr<MlxArray> {
        let residual = mlxcel_core::copy(x);
        let normed = self.norm1.forward(x);
        let attended = if self.window > 0 {
            let s = mlxcel_core::array_shape(&normed);
            let (windows, padded) = window_partition(&normed, self.window);
            let y = self.attention.forward(&windows, cos, sin);
            window_unpartition(&y, self.window, padded, (s[1], s[2]))
        } else {
            self.attention.forward(&normed, cos, sin)
        };
        let x = mlxcel_core::add(&residual, &attended);
        let mlp = self.fc2.forward(&super::activation::gelu(
            &self.fc1.forward(&self.norm2.forward(&x)),
        ));
        mlxcel_core::add(&x, &mlp)
    }
}

pub struct ViTBackbone {
    patch: Conv2d,
    position_embeddings: UniquePtr<MlxArray>,
    norm: LayerNorm,
    blocks: Vec<Block>,
    window_size: i32,
    hidden_size: i32,
    heads: i32,
    theta: f32,
}

impl ViTBackbone {
    pub fn from_weights(
        weights: &WeightMap,
        cfg: &VitConfig,
        prefix: &str,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            blocks.push(Block::load(
                weights,
                &format!("{prefix}.layers.{i}"),
                cfg,
                cfg.global_attn_indexes.contains(&i),
                gs,
                bits,
            )?);
        }
        Ok(Self {
            patch: Conv2d::load(
                weights,
                &format!("{prefix}.embeddings.patch_embeddings.projection"),
                cfg.patch_size as i32,
                0,
            )?,
            position_embeddings: weight(
                weights,
                &format!("{prefix}.embeddings.position_embeddings"),
            )?,
            norm: layer_norm(weights, &format!("{prefix}.layer_norm"), cfg.layer_norm_eps)?,
            blocks,
            window_size: cfg.window_size as i32,
            hidden_size: cfg.hidden_size as i32,
            heads: cfg.num_attention_heads as i32,
            theta: cfg.rope_theta,
        })
    }

    pub fn forward(&self, image: &MlxArray) -> UniquePtr<MlxArray> {
        let mut x = self.patch.forward(image);
        let s = mlxcel_core::array_shape(&x);
        let (b, h, w, d) = (s[0], s[1], s[2], s[3]);
        let ps = (mlxcel_core::array_shape(&self.position_embeddings)[1] as f32).sqrt() as i32;
        let pos = mlxcel_core::reshape(&self.position_embeddings, &[1, ps, ps, d]);
        let tiled = mlxcel_core::tile(&pos, &[1, h / ps + 1, w / ps + 1, 1]);
        let pos = mlxcel_core::slice(&tiled, &[0, 0, 0, 0], &[1, h, w, d]);
        x = self.norm.forward(&mlxcel_core::add(&x, &pos));
        let head_dim = self.hidden_size / self.heads;
        let (window_cos, window_sin) = axial_rope(
            head_dim,
            self.window_size,
            self.window_size,
            self.theta,
            1.0,
        );
        let (global_cos, global_sin) = axial_rope(
            head_dim,
            h,
            w,
            self.theta,
            self.window_size as f32 / h as f32,
        );
        for block in &self.blocks {
            x = if block.window == 0 {
                block.forward(&x, &global_cos, &global_sin)
            } else {
                block.forward(&x, &window_cos, &window_sin)
            };
        }
        mlxcel_core::reshape(&x, &[b, h, w, d])
    }
}

#[rustfmt::skip]
pub struct VisionEncoder { pub backbone: ViTBackbone, pub neck: FpnNeck }

impl VisionEncoder {
    pub fn from_weights(
        weights: &WeightMap,
        cfg: &VisionEncoderConfig,
        prefix: &str,
        gs: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            backbone: ViTBackbone::from_weights(
                weights,
                &cfg.backbone_config,
                &format!("{prefix}.backbone"),
                gs,
                bits,
            )?,
            neck: FpnNeck::from_weights(weights, cfg, &format!("{prefix}.neck"))?,
        })
    }

    pub fn forward_backbone(&self, image: &MlxArray) -> UniquePtr<MlxArray> {
        self.backbone.forward(image)
    }
    pub fn forward_neck(&self, backbone: &MlxArray) -> Vec<UniquePtr<MlxArray>> {
        self.neck.forward(backbone)
    }
    pub fn forward(&self, image: &MlxArray) -> Vec<UniquePtr<MlxArray>> {
        self.forward_neck(&self.forward_backbone(image))
    }
}
