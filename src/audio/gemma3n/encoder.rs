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

use super::attention::Gemma3nAudioAttention;
use super::checked_unified_linear;
use super::config::Gemma3nAudioConfig;
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

fn copy_weight(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(key)
        .map(|weight| mlxcel_core::copy(weight))
        .ok_or_else(|| format!("Gemma3n audio weight not found: {key}"))
}

fn checked_weight(
    weights: &WeightMap,
    key: &str,
    expected: &[i32],
) -> Result<UniquePtr<MlxArray>, String> {
    let weight = copy_weight(weights, key)?;
    let actual = mlxcel_core::array_shape(&weight);
    if actual != expected {
        return Err(format!(
            "Gemma3n audio weight {key} has shape {actual:?}; expected {expected:?}"
        ));
    }
    Ok(weight)
}

struct AudioRmsNorm {
    weight: UniquePtr<MlxArray>,
    eps: f32,
}

impl AudioRmsNorm {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        size: usize,
        eps: f32,
    ) -> Result<Self, String> {
        Ok(Self {
            weight: checked_weight(weights, &format!("{prefix}.weight"), &[size as i32])?,
            eps,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        mlxcel_core::rms_norm(x, &self.weight, self.eps)
    }
}

struct CumulativeGroupNorm {
    weight: UniquePtr<MlxArray>,
    eps: f32,
}

impl CumulativeGroupNorm {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        channels: usize,
        eps: f32,
    ) -> Result<Self, String> {
        Ok(Self {
            weight: checked_weight(weights, &format!("{prefix}.weight"), &[channels as i32])?,
            eps,
        })
    }

    /// Input and output are `[B, T, F, C]`. Statistics are float32 and are
    /// cumulative over time after reducing both frequency and channels.
    fn forward(&self, hidden_states: &MlxArray) -> UniquePtr<MlxArray> {
        let input_dtype = mlxcel_core::array_dtype(hidden_states);
        let x = mlxcel_core::astype(hidden_states, mlxcel_core::dtype::FLOAT32);
        let sum_at_time = mlxcel_core::sum_axis(&mlxcel_core::sum_axis(&x, 3, true), 2, true);
        let cumulative_sum = mlxcel_core::cumsum(&sum_at_time, 1, false, true);
        let count_at_time = mlxcel_core::sum_axis(
            &mlxcel_core::sum_axis(&mlxcel_core::ones_like(&x), 3, true),
            2,
            true,
        );
        let cumulative_count = mlxcel_core::cumsum(&count_at_time, 1, false, true);
        let one = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::dtype::FLOAT32);
        let safe_count = mlxcel_core::maximum(&cumulative_count, &one);
        let cumulative_mean = mlxcel_core::divide(&cumulative_sum, &safe_count);

        // Preserve the pinned reference's cumulative squared-difference
        // algorithm; replacing it with E[x^2]-E[x]^2 is not equivalent.
        let centered = mlxcel_core::subtract(&x, &cumulative_mean);
        let squared = mlxcel_core::multiply(&centered, &centered);
        let squared_at_time =
            mlxcel_core::sum_axis(&mlxcel_core::sum_axis(&squared, 3, true), 2, true);
        let cumulative_squared = mlxcel_core::cumsum(&squared_at_time, 1, false, true);
        let variance = mlxcel_core::divide(&cumulative_squared, &safe_count);
        let eps = mlxcel_core::full_f32(&[1], self.eps, mlxcel_core::dtype::FLOAT32);
        let inverse_stddev = mlxcel_core::rsqrt(&mlxcel_core::add(&variance, &eps));
        let normalized = mlxcel_core::multiply(&centered, &inverse_stddev);
        let normalized = mlxcel_core::multiply(&normalized, &self.weight);
        mlxcel_core::astype(&normalized, input_dtype)
    }
}

struct SscpConvBlock {
    conv_weight: UniquePtr<MlxArray>,
    norm: CumulativeGroupNorm,
    time_padding_after: i32,
    stride_time: i32,
    stride_frequency: i32,
}

impl SscpConvBlock {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        kernel: [usize; 2],
        stride: [usize; 2],
        eps: f32,
    ) -> Result<Self, String> {
        let conv_weight = checked_weight(
            weights,
            &format!("{prefix}.conv.weight"),
            &[
                output_channels as i32,
                kernel[0] as i32,
                kernel[1] as i32,
                input_channels as i32,
            ],
        )?;
        Ok(Self {
            conv_weight,
            norm: CumulativeGroupNorm::from_weights(
                weights,
                &format!("{prefix}.norm"),
                output_channels,
                eps,
            )?,
            time_padding_after: kernel[0] as i32 - 1,
            stride_time: stride[0] as i32,
            stride_frequency: stride[1] as i32,
        })
    }

    fn forward(&self, x: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        // Match the maintained reference's explicit
        // `audio_encodings_padded.to(self.conv.weight.dtype)` boundary. The
        // processor emits float32 mel features, while released checkpoints use
        // BF16 convolution weights; leaving the input in float32 promotes the
        // entire SSCP path and changes later greedy logits.
        let x = mlxcel_core::astype(x, mlxcel_core::array_dtype(&self.conv_weight));
        // Reverse-causal time padding `(0, kernel-1)` and SAME-like frequency
        // padding `(1,1)`, in MLX's NHWC layout.
        let x = mlxcel_core::pad(&x, &[0, 0, 0, self.time_padding_after, 1, 1, 0, 0], 0.0);
        let x = mlxcel_core::try_conv2d(
            &x,
            &self.conv_weight,
            self.stride_time,
            self.stride_frequency,
            0,
            0,
            1,
            1,
            1,
        )
        .map_err(|error| format!("Gemma3n SSCP conv2d failed: {error}"))?;
        Ok(mlxcel_core::relu(&self.norm.forward(&x)))
    }
}

struct SubSampleConvProjection {
    conv0: SscpConvBlock,
    conv1: SscpConvBlock,
    input_projection: UnifiedLinear,
}

fn final_frequency_size(config: &Gemma3nAudioConfig) -> usize {
    config
        .sscp_conv_kernel_size
        .iter()
        .zip(&config.sscp_conv_stride_size)
        .fold(config.input_feat_size, |frequency, (kernel, stride)| {
            (frequency + 2 - kernel[1]) / stride[1] + 1
        })
}

impl SubSampleConvProjection {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Gemma3nAudioConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let channels = &config.sscp_conv_channel_size;
        Ok(Self {
            conv0: SscpConvBlock::from_weights(
                weights,
                &format!("{prefix}.conv_0"),
                1,
                channels[0],
                config.sscp_conv_kernel_size[0],
                config.sscp_conv_stride_size[0],
                config.sscp_conv_group_norm_eps,
            )?,
            conv1: SscpConvBlock::from_weights(
                weights,
                &format!("{prefix}.conv_1"),
                channels[0],
                channels[1],
                config.sscp_conv_kernel_size[1],
                config.sscp_conv_stride_size[1],
                config.sscp_conv_group_norm_eps,
            )?,
            input_projection: checked_unified_linear(
                weights,
                &format!("{prefix}.input_proj_linear"),
                final_frequency_size(config) * channels[1],
                config.hidden_size,
                group_size,
                bits,
            )?,
        })
    }

    fn forward(&self, audio_mel: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let x = mlxcel_core::expand_dims(audio_mel, -1);
        let x = self.conv0.forward(&x)?;
        let x = self.conv1.forward(&x)?;
        let shape = mlxcel_core::array_shape(&x);
        let x = mlxcel_core::reshape(&x, &[shape[0], shape[1], shape[2] * shape[3]]);
        Ok(self.input_projection.forward(&x))
    }
}

fn clip_gradient(x: &MlxArray, limit: f32) -> UniquePtr<MlxArray> {
    let minimum = mlxcel_core::full_f32(&[1], -limit, mlxcel_core::array_dtype(x));
    let maximum = mlxcel_core::full_f32(&[1], limit, mlxcel_core::array_dtype(x));
    mlxcel_core::clip(x, &minimum, &maximum)
}

struct FeedForward {
    clipping: f32,
    residual_weight: f32,
    pre_norm: AudioRmsNorm,
    layer1: UnifiedLinear,
    layer2: UnifiedLinear,
    post_norm: AudioRmsNorm,
}

impl FeedForward {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Gemma3nAudioConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            clipping: config.gradient_clipping,
            residual_weight: config.conf_residual_weight,
            pre_norm: AudioRmsNorm::from_weights(
                weights,
                &format!("{prefix}.pre_layer_norm"),
                config.hidden_size,
                config.rms_norm_eps,
            )?,
            layer1: checked_unified_linear(
                weights,
                &format!("{prefix}.ffw_layer_1"),
                config.hidden_size,
                config.hidden_size * 4,
                group_size,
                bits,
            )?,
            layer2: checked_unified_linear(
                weights,
                &format!("{prefix}.ffw_layer_2"),
                config.hidden_size * 4,
                config.hidden_size,
                group_size,
                bits,
            )?,
            post_norm: AudioRmsNorm::from_weights(
                weights,
                &format!("{prefix}.post_layer_norm"),
                config.hidden_size,
                config.rms_norm_eps,
            )?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let residual = mlxcel_core::copy(x);
        let x = self.pre_norm.forward(&clip_gradient(x, self.clipping));
        let x = mlxcel_core::silu(&self.layer1.forward(&x));
        let x = self.layer2.forward(&x);
        let x = self.post_norm.forward(&clip_gradient(&x, self.clipping));
        mlxcel_core::add(
            &residual,
            &mlxcel_core::multiply_scalar(&x, self.residual_weight),
        )
    }
}

struct ConformerAttention {
    clipping: f32,
    pre_norm: AudioRmsNorm,
    attention: Gemma3nAudioAttention,
    post: UnifiedLinear,
    post_norm: AudioRmsNorm,
}

impl ConformerAttention {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Gemma3nAudioConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            clipping: config.gradient_clipping,
            pre_norm: AudioRmsNorm::from_weights(
                weights,
                &format!("{prefix}.pre_attn_norm"),
                config.hidden_size,
                config.rms_norm_eps,
            )?,
            attention: Gemma3nAudioAttention::from_weights(
                weights,
                &format!("{prefix}.attn"),
                config,
                group_size,
                bits,
            )?,
            post: checked_unified_linear(
                weights,
                &format!("{prefix}.post"),
                config.hidden_size,
                config.hidden_size,
                group_size,
                bits,
            )?,
            post_norm: AudioRmsNorm::from_weights(
                weights,
                &format!("{prefix}.post_norm"),
                config.hidden_size,
                config.rms_norm_eps,
            )?,
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        invalid_mask: &MlxArray,
        causal_valid_mask: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let residual = mlxcel_core::copy(x);
        let normalized = self.pre_norm.forward(&clip_gradient(x, self.clipping));
        let attention = self
            .attention
            .forward(&normalized, invalid_mask, causal_valid_mask);
        let shape = mlxcel_core::array_shape(&attention);
        let attention =
            mlxcel_core::reshape(&attention, &[shape[0], shape[1], shape[2] * shape[3]]);
        let attention = self.post.forward(&attention);
        let attention = self
            .post_norm
            .forward(&clip_gradient(&attention, self.clipping));
        mlxcel_core::add(&residual, &attention)
    }
}

struct LightConv1d {
    clipping: f32,
    causal_padding: i32,
    pre_norm: AudioRmsNorm,
    linear_start: UnifiedLinear,
    depthwise_weight: UniquePtr<MlxArray>,
    conv_norm: AudioRmsNorm,
    linear_end: UnifiedLinear,
}

impl LightConv1d {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Gemma3nAudioConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            clipping: config.gradient_clipping,
            causal_padding: config.conf_conv_kernel_size as i32 - 1,
            pre_norm: AudioRmsNorm::from_weights(
                weights,
                &format!("{prefix}.pre_layer_norm"),
                config.hidden_size,
                config.rms_norm_eps,
            )?,
            linear_start: checked_unified_linear(
                weights,
                &format!("{prefix}.linear_start"),
                config.hidden_size,
                config.hidden_size * 2,
                group_size,
                bits,
            )?,
            depthwise_weight: checked_weight(
                weights,
                &format!("{prefix}.depthwise_conv1d.weight"),
                &[
                    config.hidden_size as i32,
                    config.conf_conv_kernel_size as i32,
                    1,
                ],
            )?,
            conv_norm: AudioRmsNorm::from_weights(
                weights,
                &format!("{prefix}.conv_norm"),
                config.hidden_size,
                config.rms_norm_eps,
            )?,
            linear_end: checked_unified_linear(
                weights,
                &format!("{prefix}.linear_end"),
                config.hidden_size,
                config.hidden_size,
                group_size,
                bits,
            )?,
        })
    }

    fn forward(&self, x: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let residual = mlxcel_core::copy(x);
        let x = self.linear_start.forward(&self.pre_norm.forward(x));
        let shape = mlxcel_core::array_shape(&x);
        let half = shape[2] / 2;
        let left = mlxcel_core::slice(&x, &[0, 0, 0], &[shape[0], shape[1], half]);
        let right = mlxcel_core::slice(&x, &[0, 0, half], &[shape[0], shape[1], shape[2]]);
        let x = mlxcel_core::multiply(&left, &mlxcel_core::sigmoid(&right));
        let x = mlxcel_core::pad(&x, &[0, 0, self.causal_padding, 0, 0, 0], 0.0);
        let x = mlxcel_core::try_conv1d(
            &x,
            &self.depthwise_weight,
            1,
            0,
            1,
            mlxcel_core::array_shape(&x)[2],
        )
        .map_err(|error| format!("Gemma3n audio depthwise conv1d failed: {error}"))?;
        let x = self.conv_norm.forward(&clip_gradient(&x, self.clipping));
        let x = self.linear_end.forward(&mlxcel_core::silu(&x));
        Ok(mlxcel_core::add(&residual, &x))
    }
}

struct ConformerBlock {
    clipping: f32,
    feed_forward_start: FeedForward,
    attention: ConformerAttention,
    light_conv: LightConv1d,
    feed_forward_end: FeedForward,
    norm: AudioRmsNorm,
}

impl ConformerBlock {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Gemma3nAudioConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            clipping: config.gradient_clipping,
            feed_forward_start: FeedForward::from_weights(
                weights,
                &format!("{prefix}.ffw_layer_start"),
                config,
                group_size,
                bits,
            )?,
            attention: ConformerAttention::from_weights(
                weights,
                &format!("{prefix}.attention"),
                config,
                group_size,
                bits,
            )?,
            light_conv: LightConv1d::from_weights(
                weights,
                &format!("{prefix}.lconv1d"),
                config,
                group_size,
                bits,
            )?,
            feed_forward_end: FeedForward::from_weights(
                weights,
                &format!("{prefix}.ffw_layer_end"),
                config,
                group_size,
                bits,
            )?,
            norm: AudioRmsNorm::from_weights(
                weights,
                &format!("{prefix}.norm"),
                config.hidden_size,
                config.rms_norm_eps,
            )?,
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        invalid_mask: &MlxArray,
        causal_valid_mask: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let x = self.feed_forward_start.forward(x);
        let x = self.attention.forward(&x, invalid_mask, causal_valid_mask);
        let valid = mlxcel_core::astype(
            &mlxcel_core::reshape(
                &mlxcel_core::logical_not(invalid_mask),
                &[
                    mlxcel_core::array_shape(&x)[0],
                    mlxcel_core::array_shape(&x)[1],
                    1,
                ],
            ),
            mlxcel_core::array_dtype(&x),
        );
        let x = self
            .light_conv
            .forward(&mlxcel_core::multiply(&x, &valid))?;
        let x = self.feed_forward_end.forward(&x);
        Ok(self.norm.forward(&clip_gradient(&x, self.clipping)))
    }
}

pub struct Gemma3nAudioEncoder {
    config: Gemma3nAudioConfig,
    subsample: SubSampleConvProjection,
    conformer: Vec<ConformerBlock>,
}

impl Gemma3nAudioEncoder {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Gemma3nAudioConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        config.validate()?;
        let mut conformer = Vec::with_capacity(config.conf_num_hidden_layers);
        for index in 0..config.conf_num_hidden_layers {
            conformer.push(ConformerBlock::from_weights(
                weights,
                &format!("{prefix}.conformer.{index}"),
                config,
                group_size,
                bits,
            )?);
        }
        Ok(Self {
            config: config.clone(),
            subsample: SubSampleConvProjection::from_weights(
                weights,
                &format!("{prefix}.subsample_conv_projection"),
                config,
                group_size,
                bits,
            )?,
            conformer,
        })
    }

    fn causal_valid_mask(&self) -> UniquePtr<MlxArray> {
        let chunk = self.config.conf_attention_chunk_size as i32;
        let context = self.config.context_size() as i32;
        let diagonal =
            (self.config.max_past_horizon() + self.config.conf_attention_context_right) as i32;
        let lower = mlxcel_core::transpose_axes(
            &mlxcel_core::tril(
                &mlxcel_core::ones(&[context, chunk], mlxcel_core::dtype::FLOAT32),
                0,
            ),
            &[1, 0],
        );
        let upper = mlxcel_core::tril(
            &mlxcel_core::ones(&[chunk, context], mlxcel_core::dtype::FLOAT32),
            diagonal,
        );
        mlxcel_core::astype(
            &mlxcel_core::multiply(&lower, &upper),
            mlxcel_core::dtype::BOOL,
        )
    }

    fn stride_mask(mask: &MlxArray, stride: usize, length: i32) -> UniquePtr<MlxArray> {
        let source_len = mlxcel_core::array_shape(mask)[1];
        let indices: Vec<i32> = (0..length)
            .map(|index| (index * stride as i32).min(source_len - 1))
            .collect();
        mlxcel_core::take(mask, &mlxcel_core::from_slice_i32(&indices, &[length]), 1)
    }

    pub fn forward(
        &self,
        audio_mel: &MlxArray,
        invalid_mel_mask: &MlxArray,
    ) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>), String> {
        let mel_shape = mlxcel_core::array_shape(audio_mel);
        let mask_shape = mlxcel_core::array_shape(invalid_mel_mask);
        if mel_shape.len() != 3
            || mel_shape[2] != self.config.input_feat_size as i32
            || mask_shape != mel_shape[..2]
        {
            return Err(format!(
                "Gemma3n audio input shapes must be [B,T,{}] and [B,T], got {mel_shape:?} and {mask_shape:?}",
                self.config.input_feat_size
            ));
        }

        let mut encodings = self.subsample.forward(audio_mel)?;
        let mut invalid_mask = Self::stride_mask(
            invalid_mel_mask,
            self.config.time_stride_product(),
            mlxcel_core::array_shape(&encodings)[1],
        );
        let causal_valid_mask = self.causal_valid_mask();
        for block in &self.conformer {
            encodings = block.forward(&encodings, &invalid_mask, &causal_valid_mask)?;
        }

        if self.config.conf_reduction_factor > 1 {
            let reduced_len = (mlxcel_core::array_shape(&encodings)[1]
                + self.config.conf_reduction_factor as i32
                - 1)
                / self.config.conf_reduction_factor as i32;
            encodings =
                Self::stride_mask(&encodings, self.config.conf_reduction_factor, reduced_len);
            invalid_mask = Self::stride_mask(
                &invalid_mask,
                self.config.conf_reduction_factor,
                reduced_len,
            );
        }
        let shape = mlxcel_core::array_shape(&encodings);
        let expanded_mask = mlxcel_core::reshape(&invalid_mask, &[shape[0], shape[1], 1]);
        encodings = mlxcel_core::where_cond(
            &expanded_mask,
            &mlxcel_core::zeros_like(&encodings),
            &encodings,
        );
        Ok((encodings, invalid_mask))
    }
}

#[cfg(test)]
#[path = "encoder_tests.rs"]
mod tests;
