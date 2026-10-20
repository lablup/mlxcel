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

//! `Flux2Vae`: decoder, optional encoder, and the packed-latent decode path.

use std::path::Path;

use anyhow::{Context, Result, bail};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::config::Flux2VaeConfig;
use super::layers::{
    Conv2d, DownEncoderBlock, GroupNorm, MidBlock, UpDecoderBlock, nchw_to_nhwc, nhwc_to_nchw,
};
use crate::imagegen::latent::unpatchify_latents;

/// `Flux2Decoder`: conv-in, mid block, up blocks, norm, silu, conv-out.
struct Decoder {
    conv_in: Conv2d,
    mid_block: MidBlock,
    up_blocks: Vec<UpDecoderBlock>,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl Decoder {
    fn load(weights: &WeightMap, config: &Flux2VaeConfig) -> Result<Self> {
        let eps = Flux2VaeConfig::NORM_EPS;
        let groups = config.norm_num_groups;
        let channels = &config.block_out_channels;
        let top = channels[channels.len() - 1];
        let reversed: Vec<i32> = channels.iter().rev().copied().collect();
        let up_blocks = reversed
            .iter()
            .enumerate()
            .map(|(i, &out)| {
                let prev = if i == 0 { out } else { reversed[i - 1] };
                UpDecoderBlock::load(
                    weights,
                    &format!("decoder.up_blocks.{i}"),
                    prev,
                    out,
                    config.layers_per_block + 1,
                    groups,
                    eps,
                    i != reversed.len() - 1,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv_in: Conv2d::load(weights, "decoder.conv_in", 1, 1)?,
            mid_block: MidBlock::load(
                weights,
                "decoder.mid_block",
                top,
                groups,
                eps,
                config.mid_block_add_attention,
            )?,
            up_blocks,
            conv_norm_out: GroupNorm::load(weights, "decoder.conv_norm_out", groups, eps)?,
            conv_out: Conv2d::load(weights, "decoder.conv_out", 1, 1)?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let mut h = self.conv_in.forward(x);
        h = self.mid_block.forward(&h);
        for block in &self.up_blocks {
            h = block.forward(&h);
        }
        let h = mlxcel_core::silu(&self.conv_norm_out.forward(&h));
        self.conv_out.forward(&h)
    }
}

/// `Flux2Encoder`: conv-in, down blocks, mid block, norm, silu, conv-out
/// producing mean and log-variance channels.
struct Encoder {
    conv_in: Conv2d,
    down_blocks: Vec<DownEncoderBlock>,
    mid_block: MidBlock,
    conv_norm_out: GroupNorm,
    conv_out: Conv2d,
}

impl Encoder {
    fn load(weights: &WeightMap, config: &Flux2VaeConfig) -> Result<Self> {
        let eps = Flux2VaeConfig::NORM_EPS;
        let groups = config.norm_num_groups;
        let channels = &config.block_out_channels;
        let top = channels[channels.len() - 1];
        let down_blocks = channels
            .iter()
            .enumerate()
            .map(|(i, &out)| {
                let input = if i > 0 { channels[i - 1] } else { channels[0] };
                DownEncoderBlock::load(
                    weights,
                    &format!("encoder.down_blocks.{i}"),
                    input,
                    out,
                    config.layers_per_block,
                    groups,
                    eps,
                    i != channels.len() - 1,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv_in: Conv2d::load(weights, "encoder.conv_in", 1, 1)?,
            down_blocks,
            mid_block: MidBlock::load(
                weights,
                "encoder.mid_block",
                top,
                groups,
                eps,
                config.mid_block_add_attention,
            )?,
            conv_norm_out: GroupNorm::load(weights, "encoder.conv_norm_out", groups, eps)?,
            conv_out: Conv2d::load(weights, "encoder.conv_out", 1, 1)?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let mut h = self.conv_in.forward(x);
        for block in &self.down_blocks {
            h = block.forward(&h);
        }
        h = self.mid_block.forward(&h);
        let h = mlxcel_core::silu(&self.conv_norm_out.forward(&h));
        self.conv_out.forward(&h)
    }
}

/// Batch-norm running statistics applied to packed latents before decoding.
struct BatchNormStats {
    running_mean: UniquePtr<MlxArray>,
    running_var: UniquePtr<MlxArray>,
    eps: f32,
}

/// Layout of 4-D convolution weights in a checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ConvLayout {
    /// PyTorch / Diffusers: `(out, in, kH, kW)`.
    Torch,
    /// MLX: `(out, kH, kW, in)`.
    Mlx,
}

/// Decide the convolution layout from weights whose two layouts have different
/// shapes.
///
/// A square-shape comparison per tensor is ambiguous whenever a channel count
/// equals the kernel size (the encoder's RGB `conv_in` is `(C, 3, 3, 3)` in both
/// layouts). The layout is a property of the whole checkpoint, so it is decided
/// once from tensors that are unambiguous, and then applied to every conv.
pub(super) fn detect_conv_layout(
    weights: &WeightMap,
    config: &Flux2VaeConfig,
) -> Result<ConvLayout> {
    let top = *config
        .block_out_channels
        .last()
        .expect("validated non-empty");
    let first = config.block_out_channels[0];
    // (key, out, in, kernel) of convs whose in-channels differ from the kernel.
    let probes = [
        ("decoder.conv_in.weight", top, config.latent_channels, 3),
        ("decoder.conv_out.weight", config.out_channels, first, 3),
    ];
    for (key, out, inn, k) in probes {
        if inn == k {
            continue;
        }
        let Some(w) = weights.get(key) else {
            continue;
        };
        let shape = mlxcel_core::array_shape(w);
        if shape == [out, inn, k, k] {
            return Ok(ConvLayout::Torch);
        }
        if shape == [out, k, k, inn] {
            return Ok(ConvLayout::Mlx);
        }
        bail!("Unsupported convolution weight shape for {key}: {shape:?}");
    }
    // Every probe was ambiguous or absent: Diffusers checkpoints are torch layout.
    Ok(ConvLayout::Torch)
}

/// Keep only the tensors the VAE reads, cast floats to `compute_dtype`, and put
/// conv weights in MLX layout.
pub(super) fn prepare_weights(
    raw: WeightMap,
    config: &Flux2VaeConfig,
    include_encoder: bool,
    compute_dtype: i32,
) -> Result<WeightMap> {
    let mut prefixes = vec!["decoder.", "post_quant_conv.", "bn."];
    if include_encoder {
        prefixes.extend(["encoder.", "quant_conv."]);
    }
    let layout = detect_conv_layout(&raw, config)?;
    let mut weights = WeightMap::new();
    for (key, value) in raw {
        if key.ends_with(".num_batches_tracked") || !prefixes.iter().any(|p| key.starts_with(p)) {
            continue;
        }
        let mut tensor = if is_floating(&value) {
            mlxcel_core::astype(&value, compute_dtype)
        } else {
            value
        };
        if mlxcel_core::array_shape(&tensor).len() == 4 && layout == ConvLayout::Torch {
            tensor = mlxcel_core::transpose_axes(&tensor, &[0, 2, 3, 1]);
        }
        weights.insert(key, tensor);
    }
    Ok(weights)
}

fn is_floating(a: &MlxArray) -> bool {
    matches!(
        mlxcel_core::array_dtype(a),
        dtype::FLOAT16 | dtype::FLOAT32 | dtype::BFLOAT16
    )
}

/// The FLUX.2 VAE: decode (always) and encode (when loaded with the encoder).
pub struct Flux2Vae {
    config: Flux2VaeConfig,
    decoder: Decoder,
    post_quant_conv: Option<Conv2d>,
    bn: Option<BatchNormStats>,
    encoder: Option<Encoder>,
    quant_conv: Option<Conv2d>,
}

impl Flux2Vae {
    /// Default compute dtype for a new loading boundary.
    ///
    /// Follows the project rule (bf16 checkpoints run as f16 unless the policy
    /// says otherwise, see `bf16_to_f16_at_load`). Upstream runs bf16.
    #[must_use]
    pub fn default_dtype() -> i32 {
        if crate::models::bf16_to_f16_at_load(false, None) {
            dtype::FLOAT16
        } else {
            dtype::BFLOAT16
        }
    }

    /// Load `<model_dir>/vae/` with the default compute dtype.
    pub fn load(model_dir: &Path, include_encoder: bool) -> Result<Self> {
        Self::load_with_dtype(model_dir, include_encoder, Self::default_dtype())
    }

    /// Load `<model_dir>/vae/`, casting every floating weight to `compute_dtype`.
    pub fn load_with_dtype(
        model_dir: &Path,
        include_encoder: bool,
        compute_dtype: i32,
    ) -> Result<Self> {
        let vae_dir = model_dir.join("vae");
        let config = Flux2VaeConfig::from_dir(&vae_dir)?;
        let raw = mlxcel_core::weights::load_weights_from_dir(&vae_dir)
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("cannot load VAE weights from {}", vae_dir.display()))?;
        Self::from_raw_weights(config, raw, include_encoder, compute_dtype)
    }

    /// Build from already-loaded checkpoint tensors (raw Diffusers or MLX names).
    pub fn from_raw_weights(
        config: Flux2VaeConfig,
        raw: WeightMap,
        include_encoder: bool,
        compute_dtype: i32,
    ) -> Result<Self> {
        config.validate()?;
        let weights = prepare_weights(raw, &config, include_encoder, compute_dtype)?;
        let decoder = Decoder::load(&weights, &config)?;
        let post_quant_conv = weights
            .contains_key("post_quant_conv.weight")
            .then(|| Conv2d::load(&weights, "post_quant_conv", 1, 0))
            .transpose()?;
        let bn = match (
            weights.get("bn.running_mean"),
            weights.get("bn.running_var"),
        ) {
            (Some(mean), Some(var)) => Some(BatchNormStats {
                running_mean: mlxcel_core::copy(mean),
                running_var: mlxcel_core::copy(var),
                eps: config.batch_norm_eps,
            }),
            _ => None,
        };
        let (encoder, quant_conv) = if include_encoder {
            (
                Some(Encoder::load(&weights, &config)?),
                Some(Conv2d::load(&weights, "quant_conv", 1, 0)?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            config,
            decoder,
            post_quant_conv,
            bn,
            encoder,
            quant_conv,
        })
    }

    #[must_use]
    pub fn config(&self) -> &Flux2VaeConfig {
        &self.config
    }

    /// Whether the encoder was loaded.
    #[must_use]
    pub fn has_encoder(&self) -> bool {
        self.encoder.is_some()
    }

    /// Decode `[B, latent_channels, H, W]` latents to `[B, out_channels, 8H, 8W]`
    /// pixels (about `[-1, 1]`).
    pub fn decode(&self, latents: &MlxArray) -> Result<UniquePtr<MlxArray>> {
        let latents = squeeze_frames(latents)?;
        let scale = mlxcel_core::full_f32(
            &[1],
            self.config.scaling_factor,
            mlxcel_core::array_dtype(&latents),
        );
        let shift = mlxcel_core::full_f32(
            &[1],
            self.config.shift_factor,
            mlxcel_core::array_dtype(&latents),
        );
        let x = mlxcel_core::add(&mlxcel_core::divide(&latents, &scale), &shift);
        let mut x = nchw_to_nhwc(&x);
        if let Some(conv) = &self.post_quant_conv {
            x = conv.forward(&x);
        }
        Ok(nhwc_to_nchw(&self.decoder.forward(&x)))
    }

    /// Decode patchified latents `[B, 4C, H, W]` as the pipeline produces them:
    /// undo the batch-norm normalization, unpatchify, then [`Self::decode`].
    pub fn decode_packed_latents(&self, packed: &MlxArray) -> Result<UniquePtr<MlxArray>> {
        let packed = squeeze_frames(packed)?;
        let Some(bn) = &self.bn else {
            bail!("this VAE has no bn.* statistics; use decode() on unpatchified latents");
        };
        let channels = mlxcel_core::array_shape(&bn.running_mean)[0];
        let mean = mlxcel_core::reshape(&bn.running_mean, &[1, channels, 1, 1]);
        let var = mlxcel_core::reshape(&bn.running_var, &[1, channels, 1, 1]);
        let eps = mlxcel_core::full_f32(&[1], bn.eps, mlxcel_core::array_dtype(&var));
        let std = mlxcel_core::sqrt(&mlxcel_core::add(&var, &eps));
        let latents = mlxcel_core::add(&mlxcel_core::multiply(&packed, &std), &mean);
        let unpatched = unpatchify_latents(&latents)?;
        self.decode(&unpatched)
    }

    /// Encode `[B, in_channels, H, W]` pixels to `[B, latent_channels, H/8, W/8]`
    /// latents (the posterior mean).
    pub fn encode(&self, image: &MlxArray) -> Result<UniquePtr<MlxArray>> {
        let (Some(encoder), Some(quant_conv)) = (&self.encoder, &self.quant_conv) else {
            bail!("Flux2Vae was loaded without encoder weights");
        };
        let image = squeeze_frames(image)?;
        let moments = quant_conv.forward(&encoder.forward(&nchw_to_nhwc(&image)));
        // NHWC channel split: first half is the mean, second the log-variance.
        let shape = mlxcel_core::array_shape(&moments);
        let half = shape[3] / 2;
        let mean = mlxcel_core::slice(
            &moments,
            &[0, 0, 0, 0],
            &[shape[0], shape[1], shape[2], half],
        );
        let dt = mlxcel_core::array_dtype(&mean);
        let shift = mlxcel_core::full_f32(&[1], self.config.shift_factor, dt);
        let scale = mlxcel_core::full_f32(&[1], self.config.scaling_factor, dt);
        let latents = mlxcel_core::multiply(&mlxcel_core::subtract(&mean, &shift), &scale);
        Ok(nhwc_to_nchw(&latents))
    }
}

/// Accept `[B, C, 1, H, W]` as well as `[B, C, H, W]`.
fn squeeze_frames(x: &MlxArray) -> Result<UniquePtr<MlxArray>> {
    let shape = mlxcel_core::array_shape(x);
    match shape.len() {
        4 => Ok(mlxcel_core::copy(x)),
        5 if shape[2] == 1 => Ok(mlxcel_core::reshape(
            x,
            &[shape[0], shape[1], shape[3], shape[4]],
        )),
        _ => bail!("Expected a [B, C, H, W] tensor, got shape={shape:?}"),
    }
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
