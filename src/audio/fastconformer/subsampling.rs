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

//! Depthwise-striding conv subsampling with causal (asymmetric) padding.
//!
//! Port of `CausalDwStridingSubsampling` in
//! `mlx_audio/stt/models/nemotron_asr/conformer.py`. The conv stack keeps
//! NeMo's module indices (`conv.0`, `conv.2`/`conv.3`, `conv.5`/`conv.6`, ...;
//! the ReLU slots carry no weights). Every 3x3 stride-2 conv is preceded by
//! `(left, right)` padding on both the time and the frequency axis: `(2, 1)`
//! when `causal_downsampling` is set, `(1, 1)` otherwise. Each stage maps a
//! length `n` to `floor((n + left + right - 3) / 2) + 1`, so 128 mel bins
//! become 65, 33, then 17 frequency rows and the flattened feature width is
//! `channels * 17`.

use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::ConformerArgs;
use super::layers::{conv2d_weight, maybe_copy_weight};

struct SubsamplingConv {
    weight: UniquePtr<MlxArray>,
    bias: Option<UniquePtr<MlxArray>>,
    stride: i32,
    groups: i32,
    /// Pad `(left, right)` on time and frequency before the conv.
    pad: Option<(i32, i32)>,
    relu: bool,
}

impl SubsamplingConv {
    fn forward(&self, x: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let padded;
        let input = match self.pad {
            Some((left, right)) => {
                padded = mlxcel_core::pad(x, &[0, 0, left, right, left, right, 0, 0], 0.0);
                &*padded
            }
            None => x,
        };
        let y = mlxcel_core::try_conv2d(
            input,
            &self.weight,
            self.stride,
            self.stride,
            0,
            0,
            1,
            1,
            self.groups,
        )
        .map_err(|e| format!("FastConformer subsampling conv2d failed: {e}"))?;
        let y = match &self.bias {
            Some(bias) => mlxcel_core::add(&y, bias),
            None => y,
        };
        Ok(if self.relu { mlxcel_core::relu(&y) } else { y })
    }
}

/// `pre_encode`: `[B, T, feat_in]` mel frames to `[B, T', d_model]`.
pub struct CausalDwStridingSubsampling {
    convs: Vec<SubsamplingConv>,
    out: UnifiedLinear,
    args: ConformerArgs,
}

impl CausalDwStridingSubsampling {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        args: &ConformerArgs,
        quantization: (i32, i32),
    ) -> Result<Self, String> {
        let (group_size, bits) = quantization;
        let channels = args.subsampling_conv_channels as i32;
        let (left, right) = args.subsampling_padding();
        let pad = Some((left as i32, right as i32));
        let conv = |idx: usize, in_per_group: i32, stride, groups, pad, relu| {
            Ok::<_, String>(SubsamplingConv {
                weight: conv2d_weight(
                    weights,
                    &format!("{prefix}.conv.{idx}.weight"),
                    in_per_group,
                )?,
                bias: maybe_copy_weight(weights, &format!("{prefix}.conv.{idx}.bias")),
                stride,
                groups,
                pad,
                relu,
            })
        };

        let mut convs = vec![conv(0, 1, 2, 1, pad, true)?];
        for stage in 0..args.sampling_num().saturating_sub(1) {
            let base = 2 + 3 * stage;
            convs.push(conv(base, 1, 2, channels, pad, false)?);
            convs.push(conv(base + 1, channels, 1, 1, None, true)?);
        }

        let out = UnifiedLinear::from_weights(weights, &format!("{prefix}.out"), group_size, bits)?;
        if let Some(w) = weights.get(&format!("{prefix}.out.weight"))
            && weights.get(&format!("{prefix}.out.scales")).is_none()
        {
            let expected = channels * args.subsampled_length(args.feat_in) as i32;
            let shape = mlxcel_core::array_shape(w);
            if shape.len() != 2 || shape[1] != expected || shape[0] != args.d_model as i32 {
                return Err(format!(
                    "{prefix}.out.weight has shape {shape:?}, expected [{}, {expected}]",
                    args.d_model
                ));
            }
        }
        Ok(Self {
            convs,
            out,
            args: args.clone(),
        })
    }

    /// Output frame count for `length` input frames.
    pub fn output_length(&self, length: usize) -> usize {
        self.args.subsampled_length(length)
    }

    /// `x: [B, T, feat_in]` to `[B, T', d_model]`.
    pub fn forward(&self, x: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let mut h = mlxcel_core::expand_dims(x, -1);
        for conv in &self.convs {
            h = conv.forward(&h)?;
        }
        // [B, T', F', C] -> [B, T', C * F'] with channel-major ordering.
        let shape = mlxcel_core::array_shape(&h);
        let h = mlxcel_core::transpose_axes(&h, &[0, 1, 3, 2]);
        let h = mlxcel_core::reshape(&h, &[shape[0], shape[1], shape[2] * shape[3]]);
        Ok(self.out.forward(&h))
    }
}
