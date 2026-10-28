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

//! The Qwen-Image VAE (`AutoencoderKLQwenImage21`) on its single-frame image
//! path, a port of `vae.py` (mlx-vlm `73d366ec`).
//!
//! The model is a 3-D causal VAE; for one frame every causal 3-D conv is a
//! padded 2-D conv, the temporal `time_conv`s are never called, and the
//! temporal factor of the shortcut blocks only adds an all-zero (encoder) or
//! dropped (decoder) frame. Both the residual layout (Qwen-Image-2.1, with
//! pixel-unshuffle / pixel-shuffle shortcuts and a wider decoder) and the
//! plain layout (Ming-Image) load from the same config.
//!
//! Tensors are NHWC and f32 throughout, the dtype the checkpoint stores.
//! Large stride-1 convolutions run in row bands ([`CONV_BAND_ELEMENTS`]),
//! because MLX's CUDA convolution fails to allocate a 3x3 288-to-288 conv at
//! 1024x1024 even with 40 GB free; each band convolves its rows of the padded
//! input plus a `k - 1` row halo, so every output pixel sums the same
//! products. The oracle fixture generator applies the identical rule to
//! upstream (`tests/fixtures/generate_qwen_image_fixtures.py`).
//!
//! Used by: QwenImage, MingImage.

use std::path::Path;

use anyhow::{Context, Result, bail, ensure};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::config::QwenImageVaeConfig;
use crate::imagegen::mage_flow::layers::{silu, split_last};
use crate::imagegen::vae::{ConvLayout, probe_conv_layout};

/// A stride-1 conv whose `H * W * max(C_in, C_out)` exceeds this runs in
/// row bands of `CONV_BAND_ELEMENTS / (W * max(C_in, C_out))` output rows.
pub const CONV_BAND_ELEMENTS: i64 = 1 << 27;

fn weight(w: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>> {
    w.get(key)
        .map(|a| mlxcel_core::copy(a))
        .with_context(|| format!("Qwen-Image VAE weight not found: {key}"))
}

fn f32_scalar(v: f32) -> UniquePtr<MlxArray> {
    mlxcel_core::full_f32(&[1], v, dtype::FLOAT32)
}

/// `nn.Conv2d` over NHWC (MLX weight layout), banded per [`CONV_BAND_ELEMENTS`].
pub struct Conv {
    weight: UniquePtr<MlxArray>,
    bias: Option<UniquePtr<MlxArray>>,
    stride: i32,
    padding: i32,
}

impl Conv {
    pub fn load(w: &WeightMap, prefix: &str, stride: i32, padding: i32) -> Result<Self> {
        Ok(Self {
            weight: weight(w, &format!("{prefix}.weight"))?,
            bias: w
                .get(&format!("{prefix}.bias"))
                .map(|b| mlxcel_core::copy(b)),
            stride,
            padding,
        })
    }

    fn add_bias(&self, y: UniquePtr<MlxArray>) -> UniquePtr<MlxArray> {
        match &self.bias {
            Some(b) => mlxcel_core::add(&y, b),
            None => y,
        }
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        self.forward_banded(x, CONV_BAND_ELEMENTS)
    }

    /// [`Self::forward`] with band limit `limit` in place of
    /// [`CONV_BAND_ELEMENTS`].
    pub fn forward_banded(&self, x: &MlxArray, limit: i64) -> UniquePtr<MlxArray> {
        let s = mlxcel_core::array_shape(x);
        let ws = mlxcel_core::array_shape(&self.weight);
        let (h, w, c_in) = (s[1], s[2], s[3]);
        let (c_out, kh) = (ws[0], ws[1]);
        let width = i64::from(w) * i64::from(c_in.max(c_out));
        if self.stride != 1 || i64::from(h) * width <= limit {
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
            return self.add_bias(y);
        }
        let rows = (limit / width).max(1) as i32;
        let p = self.padding;
        let xp = mlxcel_core::pad(x, &[0, 0, p, p, p, p, 0, 0], 0.0);
        let ps = mlxcel_core::array_shape(&xp);
        let out_h = h + 2 * p - kh + 1;
        let mut parts = Vec::new();
        let mut r0 = 0;
        while r0 < out_h {
            let r1 = out_h.min(r0 + rows);
            let band = mlxcel_core::slice(&xp, &[0, r0, 0, 0], &[ps[0], r1 + kh - 1, ps[2], ps[3]]);
            parts.push(self.add_bias(mlxcel_core::conv2d(
                &band,
                &self.weight,
                1,
                1,
                0,
                0,
                1,
                1,
                1,
            )));
            r0 = r1;
        }
        let refs: Vec<&MlxArray> = parts.iter().map(|a| &**a).collect();
        mlxcel_core::concatenate_many(&refs, 1)
    }
}

/// `QwenImageRMSNorm`: `x / ||x||_channels * sqrt(C) * gamma`, the norm in f32.
struct ChannelRmsNorm {
    gamma: UniquePtr<MlxArray>,
    scale: f32,
}

impl ChannelRmsNorm {
    fn load(w: &WeightMap, prefix: &str) -> Result<Self> {
        let gamma = weight(w, &format!("{prefix}.gamma"))?;
        let gamma = mlxcel_core::reshape(&gamma, &[-1]);
        let dim = mlxcel_core::array_shape(&gamma)[0];
        Ok(Self {
            gamma,
            scale: (dim as f32).sqrt(),
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let xf = mlxcel_core::astype(x, dtype::FLOAT32);
        let sum = mlxcel_core::sum_axis(&mlxcel_core::square(&xf), -1, true);
        let norm = mlxcel_core::rsqrt(&mlxcel_core::add(&sum, &f32_scalar(1e-12)));
        let normalized = mlxcel_core::astype(
            &mlxcel_core::multiply(&xf, &norm),
            mlxcel_core::array_dtype(x),
        );
        let scale = mlxcel_core::full_f32(&[1], self.scale, mlxcel_core::array_dtype(x));
        mlxcel_core::multiply(&mlxcel_core::multiply(&normalized, &scale), &self.gamma)
    }
}

/// `QwenImageResidualBlock`.
struct ResidualBlock {
    norm1: ChannelRmsNorm,
    conv1: Conv,
    norm2: ChannelRmsNorm,
    conv2: Conv,
    shortcut: Option<Conv>,
}

impl ResidualBlock {
    fn load(w: &WeightMap, p: &str, in_dim: i32, out_dim: i32) -> Result<Self> {
        Ok(Self {
            norm1: ChannelRmsNorm::load(w, &format!("{p}.norm1"))?,
            conv1: Conv::load(w, &format!("{p}.conv1"), 1, 1)?,
            norm2: ChannelRmsNorm::load(w, &format!("{p}.norm2"))?,
            conv2: Conv::load(w, &format!("{p}.conv2"), 1, 1)?,
            shortcut: (in_dim != out_dim)
                .then(|| Conv::load(w, &format!("{p}.conv_shortcut"), 1, 0))
                .transpose()?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = match &self.shortcut {
            Some(conv) => conv.forward(x),
            None => mlxcel_core::copy(x),
        };
        let y = self.conv1.forward(&silu(&self.norm1.forward(x)));
        let y = self.conv2.forward(&silu(&self.norm2.forward(&y)));
        mlxcel_core::add(&y, &h)
    }
}

/// `QwenImageAttentionBlock`: single-head self-attention over `H * W`.
struct AttentionBlock {
    norm: ChannelRmsNorm,
    to_qkv: Conv,
    proj: Conv,
}

impl AttentionBlock {
    fn load(w: &WeightMap, p: &str) -> Result<Self> {
        Ok(Self {
            norm: ChannelRmsNorm::load(w, &format!("{p}.norm"))?,
            to_qkv: Conv::load(w, &format!("{p}.to_qkv"), 1, 0)?,
            proj: Conv::load(w, &format!("{p}.proj"), 1, 0)?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let s = mlxcel_core::array_shape(x);
        let (b, h, w, c) = (s[0], s[1], s[2], s[3]);
        let qkv = self.to_qkv.forward(&self.norm.forward(x));
        let qkv = mlxcel_core::reshape(&qkv, &[b, 1, h * w, 3 * c]);
        let [q, k, v]: [UniquePtr<MlxArray>; 3] = split_last(&qkv, 3)
            .try_into()
            .unwrap_or_else(|_| unreachable!("split into 3"));
        let out = mlxcel_core::layers::attention(&q, &k, &v, 1.0 / (c as f32).sqrt(), None, 0.0, 0);
        let out = self
            .proj
            .forward(&mlxcel_core::reshape(&out, &[b, h, w, c]));
        mlxcel_core::add(&out, x)
    }
}

/// `QwenImageMidBlock` with one attention layer.
struct MidBlock {
    first: ResidualBlock,
    attention: AttentionBlock,
    second: ResidualBlock,
}

impl MidBlock {
    fn load(w: &WeightMap, p: &str, dim: i32) -> Result<Self> {
        Ok(Self {
            first: ResidualBlock::load(w, &format!("{p}.resnets.0"), dim, dim)?,
            attention: AttentionBlock::load(w, &format!("{p}.attentions.0"))?,
            second: ResidualBlock::load(w, &format!("{p}.resnets.1"), dim, dim)?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        self.second
            .forward(&self.attention.forward(&self.first.forward(x)))
    }
}

/// `QwenImageResample` on one frame: nearest 2x + 3x3 conv, or a bottom /
/// right pad by one + 3x3 stride-2 conv. `time_conv` is video-only.
enum Resample {
    Up(Conv),
    Down(Conv),
}

impl Resample {
    fn load_up(w: &WeightMap, p: &str) -> Result<Self> {
        Ok(Self::Up(Conv::load(w, &format!("{p}.resample.1"), 1, 1)?))
    }

    fn load_down(w: &WeightMap, p: &str) -> Result<Self> {
        Ok(Self::Down(Conv::load(w, &format!("{p}.resample.1"), 2, 0)?))
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        match self {
            Self::Up(conv) => {
                let s = mlxcel_core::array_shape(x);
                let (b, h, w, c) = (s[0], s[1], s[2], s[3]);
                let x = mlxcel_core::reshape(x, &[b, h, 1, w, 1, c]);
                let x = mlxcel_core::broadcast_to(&x, &[b, h, 2, w, 2, c]);
                conv.forward(&mlxcel_core::reshape(&x, &[b, h * 2, w * 2, c]))
            }
            Self::Down(conv) => conv.forward(&mlxcel_core::pad(x, &[0, 0, 0, 1, 0, 1, 0, 0], 0.0)),
        }
    }
}

/// `QwenImageAvgDown3D` on one frame (the temporal pad adds a zero frame in
/// front): channels in `(c, ft, fs, fs)` order, averaged in groups.
struct AvgDown {
    out_dim: i32,
    factor_t: i32,
    factor_s: i32,
}

impl AvgDown {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let (ft, fs) = (self.factor_t, self.factor_s);
        let s = mlxcel_core::array_shape(x);
        let (b, h, w, c) = (s[0], s[1], s[2], s[3]);
        let x = mlxcel_core::reshape(x, &[b, h / fs, fs, w / fs, fs, c]);
        let x = mlxcel_core::transpose_axes(&x, &[0, 1, 3, 5, 2, 4]);
        let x = mlxcel_core::reshape(&x, &[b, h / fs, w / fs, c, 1, fs, fs]);
        let x = if ft == 1 {
            x
        } else {
            let mut frames: Vec<UniquePtr<MlxArray>> =
                (0..ft - 1).map(|_| mlxcel_core::zeros_like(&x)).collect();
            frames.push(x);
            let refs: Vec<&MlxArray> = frames.iter().map(|a| &**a).collect();
            mlxcel_core::concatenate_many(&refs, 4)
        };
        let group = c * ft * fs * fs / self.out_dim;
        let x = mlxcel_core::reshape(&x, &[b, h / fs, w / fs, self.out_dim, group]);
        mlxcel_core::mean_axis(&x, -1, false)
    }
}

/// `QwenImageDupUp3D` on one frame with `first_chunk`: channels repeated,
/// pixel-shuffled, and only the last temporal sub-frame kept.
struct DupUp {
    out_dim: i32,
    factor_t: i32,
    factor_s: i32,
}

impl DupUp {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let (ft, fs) = (self.factor_t, self.factor_s);
        let s = mlxcel_core::array_shape(x);
        let (b, h, w, c) = (s[0], s[1], s[2], s[3]);
        let repeats = self.out_dim * ft * fs * fs / c;
        let x = mlxcel_core::broadcast_to(
            &mlxcel_core::reshape(x, &[b, h, w, c, 1]),
            &[b, h, w, c, repeats],
        );
        let x = mlxcel_core::reshape(&x, &[b, h, w, self.out_dim, ft, fs, fs]);
        let x = mlxcel_core::slice(
            &x,
            &[0, 0, 0, 0, ft - 1, 0, 0],
            &[b, h, w, self.out_dim, ft, fs, fs],
        );
        let x = mlxcel_core::reshape(&x, &[b, h, w, self.out_dim, fs, fs]);
        let x = mlxcel_core::transpose_axes(&x, &[0, 1, 4, 2, 5, 3]);
        mlxcel_core::reshape(&x, &[b, h * fs, w * fs, self.out_dim])
    }
}

/// One stage of the encoder or decoder.
enum Stage {
    /// `QwenImageResidualDownBlock`: resnets, optional downsampler, plus
    /// the average shortcut of the stage input.
    ResidualDown {
        resnets: Vec<ResidualBlock>,
        down: Option<Resample>,
        shortcut: AvgDown,
    },
    /// `QwenImageResidualUpBlock`: resnets, optional upsampler, plus the
    /// duplicate shortcut of the stage input.
    ResidualUp {
        resnets: Vec<ResidualBlock>,
        up: Option<Resample>,
        shortcut: Option<DupUp>,
    },
    /// A plain sequence (the non-residual encoder's flat list and
    /// `QwenImageUpBlock`).
    Plain {
        resnets: Vec<ResidualBlock>,
        resample: Option<Resample>,
    },
}

impl Stage {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let run = |resnets: &[ResidualBlock], resample: Option<&Resample>| {
            let mut y = mlxcel_core::copy(x);
            for r in resnets {
                y = r.forward(&y);
            }
            match resample {
                Some(r) => r.forward(&y),
                None => y,
            }
        };
        match self {
            Self::ResidualDown {
                resnets,
                down,
                shortcut,
            } => mlxcel_core::add(&run(resnets, down.as_ref()), &shortcut.forward(x)),
            Self::ResidualUp {
                resnets,
                up,
                shortcut,
            } => {
                let y = run(resnets, up.as_ref());
                match shortcut {
                    Some(s) => mlxcel_core::add(&y, &s.forward(x)),
                    None => y,
                }
            }
            Self::Plain { resnets, resample } => run(resnets, resample.as_ref()),
        }
    }
}

fn resnet_chain(
    w: &WeightMap,
    p: &str,
    in_dim: i32,
    out_dim: i32,
    count: i32,
) -> Result<Vec<ResidualBlock>> {
    let mut dim = in_dim;
    (0..count)
        .map(|j| {
            let block = ResidualBlock::load(w, &format!("{p}.resnets.{j}"), dim, out_dim);
            dim = out_dim;
            block
        })
        .collect()
}

struct Encoder {
    conv_in: Conv,
    stages: Vec<Stage>,
    mid: MidBlock,
    norm_out: ChannelRmsNorm,
    conv_out: Conv,
}

impl Encoder {
    fn load(w: &WeightMap, cfg: &QwenImageVaeConfig) -> Result<Self> {
        let dims: Vec<i32> = std::iter::once(1)
            .chain(cfg.dim_mult.iter().copied())
            .map(|u| cfg.base_dim * u)
            .collect();
        let n = cfg.dim_mult.len();
        let mut stages = Vec::new();
        let mut flat = 0;
        for i in 0..n {
            let (in_dim, out_dim) = (dims[i], dims[i + 1]);
            let last = i == n - 1;
            if cfg.is_residual {
                let p = format!("encoder.down_blocks.{i}");
                let temporal = !last && cfg.temperal_downsample[i];
                stages.push(Stage::ResidualDown {
                    resnets: resnet_chain(w, &p, in_dim, out_dim, cfg.num_res_blocks)?,
                    down: (!last)
                        .then(|| Resample::load_down(w, &format!("{p}.downsampler")))
                        .transpose()?,
                    shortcut: AvgDown {
                        out_dim,
                        factor_t: if temporal { 2 } else { 1 },
                        factor_s: if last { 1 } else { 2 },
                    },
                });
            } else {
                let mut cur = in_dim;
                let mut resnets = Vec::new();
                for _ in 0..cfg.num_res_blocks {
                    resnets.push(ResidualBlock::load(
                        w,
                        &format!("encoder.down_blocks.{flat}"),
                        cur,
                        out_dim,
                    )?);
                    flat += 1;
                    cur = out_dim;
                }
                let resample = if last {
                    None
                } else {
                    let r = Resample::load_down(w, &format!("encoder.down_blocks.{flat}"))?;
                    flat += 1;
                    Some(r)
                };
                stages.push(Stage::Plain { resnets, resample });
            }
        }
        let top = dims[n];
        Ok(Self {
            conv_in: Conv::load(w, "encoder.conv_in", 1, 1)?,
            stages,
            mid: MidBlock::load(w, "encoder.mid_block", top)?,
            norm_out: ChannelRmsNorm::load(w, "encoder.norm_out")?,
            conv_out: Conv::load(w, "encoder.conv_out", 1, 1)?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let mut x = self.conv_in.forward(x);
        for stage in &self.stages {
            x = stage.forward(&x);
        }
        let x = self.mid.forward(&x);
        self.conv_out.forward(&silu(&self.norm_out.forward(&x)))
    }
}

struct Decoder {
    conv_in: Conv,
    mid: MidBlock,
    stages: Vec<Stage>,
    norm_out: ChannelRmsNorm,
    conv_out: Conv,
}

impl Decoder {
    fn load(w: &WeightMap, cfg: &QwenImageVaeConfig) -> Result<Self> {
        let mult = &cfg.dim_mult;
        let n = mult.len();
        let dims: Vec<i32> = std::iter::once(mult[n - 1])
            .chain(mult.iter().rev().copied())
            .map(|u| cfg.decoder_base_dim * u)
            .collect();
        let temporal_up: Vec<bool> = cfg.temperal_downsample.iter().rev().copied().collect();
        let mut stages = Vec::new();
        for i in 0..n {
            let (mut in_dim, out_dim) = (dims[i], dims[i + 1]);
            if i > 0 && !cfg.is_residual {
                in_dim /= 2;
            }
            let up_flag = i != n - 1;
            let p = format!("decoder.up_blocks.{i}");
            let resnets = resnet_chain(w, &p, in_dim, out_dim, cfg.num_res_blocks + 1)?;
            if cfg.is_residual {
                let temporal = up_flag && temporal_up[i];
                stages.push(Stage::ResidualUp {
                    resnets,
                    up: up_flag
                        .then(|| Resample::load_up(w, &format!("{p}.upsampler")))
                        .transpose()?,
                    shortcut: up_flag.then_some(DupUp {
                        out_dim,
                        factor_t: if temporal { 2 } else { 1 },
                        factor_s: 2,
                    }),
                });
            } else {
                stages.push(Stage::Plain {
                    resnets,
                    resample: up_flag
                        .then(|| Resample::load_up(w, &format!("{p}.upsamplers.0")))
                        .transpose()?,
                });
            }
        }
        Ok(Self {
            conv_in: Conv::load(w, "decoder.conv_in", 1, 1)?,
            mid: MidBlock::load(w, "decoder.mid_block", dims[0])?,
            stages,
            norm_out: ChannelRmsNorm::load(w, "decoder.norm_out")?,
            conv_out: Conv::load(w, "decoder.conv_out", 1, 1)?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let x = self.conv_in.forward(x);
        let mut x = self.mid.forward(&x);
        for stage in &self.stages {
            x = stage.forward(&x);
        }
        self.conv_out.forward(&silu(&self.norm_out.forward(&x)))
    }
}

/// `QwenImageVAE` restricted to one frame.
pub struct QwenImageVae {
    config: QwenImageVaeConfig,
    encoder: Option<(Encoder, Conv)>,
    post_quant_conv: Conv,
    decoder: Decoder,
}

/// Put every conv kernel in MLX `(out, kH, kW, in)` layout, decided once for
/// the checkpoint so a converted tree is never permuted twice, and drop the
/// video-only `time_conv`s.
fn prepare_weights(raw: WeightMap, cfg: &QwenImageVaeConfig) -> Result<WeightMap> {
    let top = *cfg.dim_mult.last().context("dim_mult must not be empty")?;
    let probes = [
        (
            "decoder.conv_in.weight",
            cfg.decoder_base_dim * top,
            cfg.z_dim,
            3,
        ),
        (
            "encoder.conv_out.weight",
            2 * cfg.z_dim,
            cfg.base_dim * top,
            3,
        ),
    ];
    let layout = probe_conv_layout(&raw, &probes)?;
    let mut out = WeightMap::new();
    for (key, value) in raw {
        if key.contains(".time_conv.") {
            continue;
        }
        let shape = mlxcel_core::array_shape(&value);
        let value = match shape.len() {
            4 if layout == ConvLayout::Torch && !key.ends_with(".gamma") => {
                mlxcel_core::transpose_axes(&value, &[0, 2, 3, 1])
            }
            5 if !key.ends_with(".gamma") => {
                bail!(
                    "Qwen-Image VAE weight {key} is a 3-D kernel {shape:?}; only the image layout is supported"
                )
            }
            _ => value,
        };
        out.insert(key, mlxcel_core::astype(&value, dtype::FLOAT32));
    }
    Ok(out)
}

impl QwenImageVae {
    /// Build from raw (`vae/*.safetensors`) weights. `include_encoder` loads
    /// the encoder and `quant_conv` an edit needs.
    pub fn new(config: QwenImageVaeConfig, raw: WeightMap, include_encoder: bool) -> Result<Self> {
        config.validate()?;
        let w = prepare_weights(raw, &config)?;
        let encoder = include_encoder
            .then(|| -> Result<_> {
                Ok((
                    Encoder::load(&w, &config)?,
                    Conv::load(&w, "quant_conv", 1, 0)?,
                ))
            })
            .transpose()?;
        Ok(Self {
            post_quant_conv: Conv::load(&w, "post_quant_conv", 1, 0)?,
            decoder: Decoder::load(&w, &config)?,
            encoder,
            config,
        })
    }

    /// Read the weights of `vae_dir`.
    pub fn load(vae_dir: &Path, config: QwenImageVaeConfig, include_encoder: bool) -> Result<Self> {
        let raw = mlxcel_core::weights::load_weights_from_dir_filtered(vae_dir, |k| {
            include_encoder || !(k.starts_with("encoder.") || k.starts_with("quant_conv."))
        })
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("cannot load the Qwen-Image VAE from {}", vae_dir.display()))?;
        Self::new(config, raw, include_encoder)
    }

    #[must_use]
    pub fn config(&self) -> &QwenImageVaeConfig {
        &self.config
    }

    /// Normalized latents `[1, z_dim, h, w]` f32 to pixels `[1, H, W, C]` f32,
    /// clipped to `[-1, 1]`; the latents are first de-normalized with
    /// `latents_std` / `latents_mean` as upstream's pipeline does.
    pub fn decode_normalized(&self, latents: &MlxArray) -> Result<UniquePtr<MlxArray>> {
        let z = self.config.z_dim;
        let shape = mlxcel_core::array_shape(latents);
        ensure!(
            shape.len() == 4 && shape[1] == z,
            "Qwen-Image VAE latents must be [B, {z}, h, w], got {shape:?}"
        );
        let x = mlxcel_core::astype(latents, dtype::FLOAT32);
        let (mean, std) = self.latent_stats(4);
        let x = mlxcel_core::add(&mlxcel_core::multiply(&x, &std), &mean);
        let x = mlxcel_core::transpose_axes(&x, &[0, 2, 3, 1]);
        let out = self.decoder.forward(&self.post_quant_conv.forward(&x));
        Ok(mlxcel_core::clip(&out, &f32_scalar(-1.0), &f32_scalar(1.0)))
    }

    /// Pixels `[1, H, W, C]` in `[-1, 1]` to normalized latents
    /// `[1, z_dim, h, w]` f32: the posterior mean, then
    /// `(mean - latents_mean) / latents_std`.
    pub fn encode_normalized(&self, pixels: &MlxArray) -> Result<UniquePtr<MlxArray>> {
        let (encoder, quant_conv) = self
            .encoder
            .as_ref()
            .context("the Qwen-Image VAE was loaded without its encoder")?;
        let x = mlxcel_core::astype(pixels, dtype::FLOAT32);
        let h = quant_conv.forward(&encoder.forward(&x));
        let mean = split_last(&h, 2).swap_remove(0);
        let mean = mlxcel_core::transpose_axes(&mean, &[0, 3, 1, 2]);
        let (lm, ls) = self.latent_stats(4);
        Ok(mlxcel_core::divide(&mlxcel_core::subtract(&mean, &lm), &ls))
    }

    /// `latents_mean` / `latents_std` shaped to broadcast over `[B, z, ...]`;
    /// a config without them (Ming-Image, which scales with
    /// `scaling_factor` / `shift_factor` itself) normalizes with 0 and 1.
    fn latent_stats(&self, ndim: usize) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let z = self.config.z_dim as usize;
        let mut shape = vec![1, self.config.z_dim];
        shape.resize(ndim, 1);
        let or = |v: &[f32], fill: f32| {
            if v.is_empty() {
                vec![fill; z]
            } else {
                v.to_vec()
            }
        };
        (
            mlxcel_core::from_slice_f32(&or(&self.config.latents_mean, 0.0), &shape),
            mlxcel_core::from_slice_f32(&or(&self.config.latents_std, 1.0), &shape),
        )
    }
}

#[cfg(test)]
#[path = "vae_tests.rs"]
mod tests;
