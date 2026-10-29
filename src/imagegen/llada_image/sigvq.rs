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

//! SigVQ: the image-to-codebook encoder and the discrete-token semantic
//! projector.
//!
//! A port of `sigvq.py` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/llada_image/sigvq.py>.
//! Editing encodes the half-resolution source through a 40-block ViT, a 1x1
//! `quant_conv` and a nearest-codeword search on the unit sphere; VQ
//! generation skips the encoder and embeds the prior's token ids directly.
//! Both end in `prior_projector[1](silu(prior_projector[0](embed(ids))))`.
//! The encoder is only loaded for editing.

use anyhow::{Result, bail};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::conditioning::{AffineLayerNorm, Dense, attention, gelu, shaped};
use super::config::SigVqConfig;

fn f32_scalar(value: f32) -> UniquePtr<MlxArray> {
    mlxcel_core::full_f32(&[], value, dtype::FLOAT32)
}

/// `resize_positions`: bilinear `grid_sample` of the square position grid to
/// `height x width` with half-pixel centres and border clamping, in f32.
/// Returns `[1, height * width, D]` f32.
#[must_use]
pub fn resize_positions(
    embedding: &MlxArray,
    size: i32,
    height: i32,
    width: i32,
) -> UniquePtr<MlxArray> {
    let d = mlxcel_core::array_shape(embedding)[1];
    let grid = mlxcel_core::astype(embedding, dtype::FLOAT32);
    let coords = |n: i32| {
        let a = mlxcel_core::astype(&mlxcel_core::arange_i32(0, n, 1), dtype::FLOAT32);
        let scaled = mlxcel_core::divide(
            &mlxcel_core::multiply(
                &mlxcel_core::add(&a, &f32_scalar(0.5)),
                &f32_scalar(size as f32),
            ),
            &f32_scalar(n as f32),
        );
        let shifted = mlxcel_core::subtract(&scaled, &f32_scalar(0.5));
        mlxcel_core::clip(&shifted, &f32_scalar(0.0), &f32_scalar((size - 1) as f32))
    };
    let (y, x) = (coords(height), coords(width));
    let (y0, x0) = (
        mlxcel_core::astype(&y, dtype::INT32),
        mlxcel_core::astype(&x, dtype::INT32),
    );
    let last = mlxcel_core::from_slice_i32(&[size - 1], &[1]);
    let one = mlxcel_core::from_slice_i32(&[1], &[1]);
    let y1 = mlxcel_core::minimum(&mlxcel_core::add(&y0, &one), &last);
    let x1 = mlxcel_core::minimum(&mlxcel_core::add(&x0, &one), &last);
    let dy = mlxcel_core::reshape(
        &mlxcel_core::subtract(&y, &mlxcel_core::astype(&y0, dtype::FLOAT32)),
        &[height, 1, 1],
    );
    let dx = mlxcel_core::reshape(
        &mlxcel_core::subtract(&x, &mlxcel_core::astype(&x0, dtype::FLOAT32)),
        &[1, width, 1],
    );
    let size_arr = mlxcel_core::from_slice_i32(&[size], &[1]);
    // `grid[yi[:, None], xi[None, :]]` on the flattened `[size * size, D]` grid.
    let gather = |yi: &MlxArray, xi: &MlxArray| {
        let rows = mlxcel_core::multiply(&mlxcel_core::reshape(yi, &[height, 1]), &size_arr);
        let index = mlxcel_core::add(&rows, &mlxcel_core::reshape(xi, &[1, width]));
        let flat = mlxcel_core::reshape(&index, &[height * width]);
        mlxcel_core::reshape(&mlxcel_core::take(&grid, &flat, 0), &[height, width, d])
    };
    let one_f = f32_scalar(1.0);
    let blend = |a: &MlxArray, b: &MlxArray, t: &MlxArray| {
        mlxcel_core::add(
            &mlxcel_core::multiply(a, &mlxcel_core::subtract(&one_f, t)),
            &mlxcel_core::multiply(b, t),
        )
    };
    let top = blend(&gather(&y0, &x0), &gather(&y0, &x1), &dx);
    let bottom = blend(&gather(&y1, &x0), &gather(&y1, &x1), &dx);
    let mixed = mlxcel_core::add(
        &mlxcel_core::multiply(&mlxcel_core::subtract(&one_f, &dy), &top),
        &mlxcel_core::multiply(&dy, &bottom),
    );
    mlxcel_core::reshape(&mixed, &[1, height * width, d])
}

/// `nn.Conv2d` (NHWC input, `[O, kh, kw, C]` kernel, stride = kernel) plus bias.
struct PatchConv {
    weight: UniquePtr<MlxArray>,
    bias: UniquePtr<MlxArray>,
    stride: i32,
}

impl PatchConv {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let y = mlxcel_core::conv2d(x, &self.weight, self.stride, self.stride, 0, 0, 1, 1, 1);
        mlxcel_core::add(&y, &self.bias)
    }
}

struct SigVqBlock {
    norm1: AffineLayerNorm,
    norm2: AffineLayerNorm,
    qkv: Dense,
    proj: Dense,
    fc1: Dense,
    fc2: Dense,
    heads: i32,
}

impl SigVqBlock {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let qkv = self.qkv.forward(&self.norm1.forward(x));
        let shape = mlxcel_core::array_shape(&qkv);
        let third = shape[2] / 3;
        let part = |i: i32| {
            mlxcel_core::slice(
                &qkv,
                &[0, 0, i * third],
                &[shape[0], shape[1], (i + 1) * third],
            )
        };
        let attended =
            self.proj
                .forward(&attention(&part(0), &part(1), &part(2), self.heads, None));
        let x = mlxcel_core::add(x, &attended);
        let mlp = self
            .fc2
            .forward(&gelu(&self.fc1.forward(&self.norm2.forward(&x))));
        mlxcel_core::add(&x, &mlp)
    }
}

/// The encoder half (`visual`, `vqmodel`), loaded only for editing.
struct Encoder {
    patch_embed: PatchConv,
    position_embedding: UniquePtr<MlxArray>,
    blocks: Vec<SigVqBlock>,
    quant_conv: PatchConv,
    codebook: UniquePtr<MlxArray>,
}

/// `SigVQ`.
pub struct SigVq {
    config: SigVqConfig,
    encoder: Option<Encoder>,
    prior_token_embedding: UniquePtr<MlxArray>,
    prior_projector: [Dense; 2],
}

/// Unit-normalize the last axis: the norm is computed in f32 and rounded to
/// `x`'s dtype, then floored at 1e-12.
fn normalize(x: &MlxArray) -> UniquePtr<MlxArray> {
    let dt = mlxcel_core::array_dtype(x);
    let squares = mlxcel_core::square(&mlxcel_core::astype(x, dtype::FLOAT32));
    let norm = mlxcel_core::sqrt(&mlxcel_core::sum_axis_native_dtype(&squares, -1, true));
    let norm = mlxcel_core::astype(&norm, dt);
    let floor = mlxcel_core::full_f32(&[], 1e-12, dt);
    mlxcel_core::divide(x, &mlxcel_core::maximum(&norm, &floor))
}

impl SigVq {
    /// Build from sanitized weights (`prior_projector.0/1`, NHWC conv
    /// kernels). The encoder weights are read when `include_encoder`.
    pub fn from_weights(
        config: SigVqConfig,
        weights: &WeightMap,
        include_encoder: bool,
    ) -> Result<Self> {
        let semantic = config.semantic_embed_dim;
        let encoder = if include_encoder {
            let dim = config.hidden_size;
            let p = config.patch_size;
            let blocks = (0..config.num_hidden_layers)
                .map(|i| {
                    let b = format!("visual.blocks.{i}");
                    let eps = config.norm_eps;
                    let bias = config.attention_bias;
                    Ok(SigVqBlock {
                        norm1: AffineLayerNorm::load(weights, &format!("{b}.norm1"), dim, eps)?,
                        norm2: AffineLayerNorm::load(weights, &format!("{b}.norm2"), dim, eps)?,
                        qkv: Dense::load(weights, &format!("{b}.attn.qkv"), dim, 3 * dim, bias)?,
                        proj: Dense::load(weights, &format!("{b}.attn.proj"), dim, dim, bias)?,
                        fc1: Dense::load(
                            weights,
                            &format!("{b}.mlp.fc1"),
                            dim,
                            config.intermediate_size,
                            true,
                        )?,
                        fc2: Dense::load(
                            weights,
                            &format!("{b}.mlp.fc2"),
                            config.intermediate_size,
                            dim,
                            true,
                        )?,
                        heads: config.num_attention_heads,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let grid = config.position_grid();
            Some(Encoder {
                patch_embed: PatchConv {
                    weight: shaped(
                        weights,
                        "visual.patch_embed.proj.weight",
                        &[dim, p, p, config.in_channels],
                    )?,
                    bias: shaped(weights, "visual.patch_embed.proj.bias", &[dim])?,
                    stride: p,
                },
                position_embedding: shaped(
                    weights,
                    "visual.embeddings.position_embedding.weight",
                    &[grid * grid, dim],
                )?,
                blocks,
                quant_conv: PatchConv {
                    weight: shaped(
                        weights,
                        "vqmodel.quant_conv.weight",
                        &[config.codebook_embed_dim, 1, 1, dim],
                    )?,
                    bias: shaped(
                        weights,
                        "vqmodel.quant_conv.bias",
                        &[config.codebook_embed_dim],
                    )?,
                    stride: 1,
                },
                codebook: shaped(
                    weights,
                    "vqmodel.quantize.embedding.weight",
                    &[config.codebook_size, config.codebook_embed_dim],
                )?,
            })
        } else {
            None
        };
        Ok(Self {
            prior_token_embedding: shaped(
                weights,
                "prior_token_embedding.weight",
                &[config.codebook_size, semantic],
            )?,
            prior_projector: [
                Dense::load(weights, "prior_projector.0", semantic, semantic, true)?,
                Dense::load(weights, "prior_projector.1", semantic, semantic, true)?,
            ],
            encoder,
            config,
        })
    }

    #[must_use]
    pub fn config(&self) -> &SigVqConfig {
        &self.config
    }

    #[must_use]
    pub fn has_encoder(&self) -> bool {
        self.encoder.is_some()
    }

    /// `encode`: NHWC RGB pixels in `[-1, 1]` (`[B, H, W, 3]`) to codebook
    /// ids `[B, (H / p) * (W / p)]`.
    pub fn encode(&self, pixels: &MlxArray) -> Result<UniquePtr<MlxArray>> {
        let Some(encoder) = &self.encoder else {
            bail!("SigVQ was loaded without its image encoder");
        };
        let shape = mlxcel_core::array_shape(pixels);
        if shape.len() != 4 || shape[3] != 3 {
            bail!("SigVQ expects an NHWC RGB image batch, got {shape:?}");
        }
        let p = self.config.patch_size;
        if shape[1] % p != 0 || shape[2] % p != 0 {
            bail!("SigVQ dimensions must be divisible by its patch size {p}");
        }
        let hidden = encoder.patch_embed.forward(pixels);
        let hs = mlxcel_core::array_shape(&hidden);
        let (b, h, w, d) = (hs[0], hs[1], hs[2], hs[3]);
        let mut hidden = mlxcel_core::reshape(&hidden, &[b, h * w, d]);
        let positions = resize_positions(
            &encoder.position_embedding,
            self.config.position_grid(),
            h,
            w,
        );
        hidden = mlxcel_core::add(
            &hidden,
            &mlxcel_core::astype(&positions, mlxcel_core::array_dtype(&hidden)),
        );
        for block in &encoder.blocks {
            hidden = block.forward(&hidden);
        }
        let hidden = encoder
            .quant_conv
            .forward(&mlxcel_core::reshape(&hidden, &[b, h, w, d]));
        let hidden = mlxcel_core::reshape(&hidden, &[b, h * w, -1]);
        let hidden = normalize(&hidden);
        let codebook = normalize(&encoder.codebook);
        let dt = mlxcel_core::array_dtype(&hidden);
        let two = mlxcel_core::full_f32(&[], 2.0, dt);
        let sq = |t: &MlxArray| mlxcel_core::power(t, &two);
        let distances = mlxcel_core::subtract(
            &mlxcel_core::add(
                &mlxcel_core::sum_axis_native_dtype(&sq(&hidden), -1, true),
                &mlxcel_core::sum_axis_native_dtype(&sq(&codebook), -1, false),
            ),
            &mlxcel_core::multiply(
                &two,
                &mlxcel_core::matmul(&hidden, &mlxcel_core::transpose(&codebook)),
            ),
        );
        Ok(mlxcel_core::argmin(&distances, -1, false))
    }

    /// Semantic features `[B, N, semantic_embed_dim]` of codebook `token_ids`.
    #[must_use]
    pub fn embed_tokens(&self, token_ids: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(token_ids);
        let flat = mlxcel_core::reshape(token_ids, &[-1]);
        let features = mlxcel_core::take(&self.prior_token_embedding, &flat, 0);
        let mut out_shape = shape;
        out_shape.push(self.config.semantic_embed_dim);
        let features = mlxcel_core::reshape(&features, &out_shape);
        let hidden = mlxcel_core::silu(&self.prior_projector[0].forward(&features));
        self.prior_projector[1].forward(&hidden)
    }

    /// `__call__(pixels=...)`: encode, then embed.
    pub fn features_from_pixels(&self, pixels: &MlxArray) -> Result<UniquePtr<MlxArray>> {
        let ids = self.encode(pixels)?;
        Ok(self.embed_tokens(&ids))
    }
}

#[cfg(test)]
#[path = "sigvq_tests.rs"]
mod tests;
