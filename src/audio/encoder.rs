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

//! Gemma4 Conformer audio encoder.
//!
//! Architecture: SSCP -> 12x ConformerBlock -> output projection.
//! Ported from: https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/gemma4/audio.py
//!
//! Used by: Gemma4 VLM (audio modality)

use super::attention::AudioAttention;
use super::config::AudioConfig;
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

fn copy_weight(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(key)
        .map(|weight| mlxcel_core::copy(weight))
        .ok_or_else(|| format!("Audio weight not found: {key}"))
}

/// Debug probe: dump an intermediate activation as raw f32 + shape when
/// `MLXCEL_AUDIO_PROBE_DIR` is set. Used for parity diffing against the
/// reference implementation; no-op in normal operation.
pub(crate) fn audio_probe_dump(name: &str, arr: &MlxArray) {
    if let Ok(dir) = std::env::var("MLXCEL_AUDIO_PROBE_DIR") {
        let arr_f32 = mlxcel_core::astype(arr, mlxcel_core::dtype::FLOAT32);
        mlxcel_core::eval(&arr_f32);
        let bytes = mlxcel_core::array_to_raw_bytes(&arr_f32);
        let shape = mlxcel_core::array_shape(&arr_f32);
        let _ = std::fs::write(format!("{dir}/{name}.f32"), &bytes);
        let _ = std::fs::write(format!("{dir}/{name}.shape"), format!("{shape:?}"));
    }
}

// ---------------------------------------------------------------------------
// AudioRMSNorm: weight applied directly (no +1 offset like Gemma text RMSNorm)
// ---------------------------------------------------------------------------

struct AudioRMSNorm {
    weight: UniquePtr<MlxArray>,
    eps: f32,
}

impl AudioRMSNorm {
    fn from_weights(weights: &WeightMap, prefix: &str, eps: f32) -> Result<Self, String> {
        Ok(Self {
            weight: copy_weight(weights, &format!("{prefix}.weight"))?,
            eps,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        mlxcel_core::rms_norm(x, &self.weight, self.eps)
    }
}

// ---------------------------------------------------------------------------
// ClippableLinear: `linear.weight` plus the checkpoint's input/output clamp
// bounds. The Gemma 4 audio checkpoints ship FINITE per-layer bounds (for
// example lconv1d.linear_end input is clamped to roughly +-5.8 while block
// activations reach far larger tails), and the reference implementation
// clamps input and output at inference, so the clamps are part of the
// trained function. Skipping them decorrelates the Conformer stack: the
// per-block error compounds from ~8% relative RMS after block 0 to ~95%
// after block 11 (issue #782).
// ---------------------------------------------------------------------------

pub(crate) struct AudioLinear {
    linear: UnifiedLinear,
    input_bounds: Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>,
    output_bounds: Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)>,
}

impl AudioLinear {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let bounds =
            |min_key: &str, max_key: &str| match (weights.get(min_key), weights.get(max_key)) {
                (Some(min), Some(max)) => Some((mlxcel_core::copy(min), mlxcel_core::copy(max))),
                _ => None,
            };
        Ok(Self {
            linear: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.linear"),
                group_size,
                bits,
            )?,
            input_bounds: bounds(
                &format!("{prefix}.input_min"),
                &format!("{prefix}.input_max"),
            ),
            output_bounds: bounds(
                &format!("{prefix}.output_min"),
                &format!("{prefix}.output_max"),
            ),
        })
    }

    pub(crate) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let clipped_input;
        let x = if let Some((min, max)) = &self.input_bounds {
            clipped_input = mlxcel_core::clip(x, min, max);
            clipped_input.as_ref().unwrap()
        } else {
            x
        };
        let y = self.linear.forward(x);
        if let Some((min, max)) = &self.output_bounds {
            mlxcel_core::clip(&y, min, max)
        } else {
            y
        }
    }
}

// ---------------------------------------------------------------------------
// SSCPConvBlock: Conv2d + LayerNorm(channels) + ReLU with symmetric padding
// ---------------------------------------------------------------------------

struct SSCPConvBlock {
    conv_weight: UniquePtr<MlxArray>,
    norm_weight: UniquePtr<MlxArray>,
    eps: f32,
}

impl SSCPConvBlock {
    fn from_weights(weights: &WeightMap, prefix: &str, eps: f32) -> Result<Self, String> {
        Ok(Self {
            conv_weight: copy_weight(weights, &format!("{prefix}.conv.weight"))?,
            norm_weight: copy_weight(weights, &format!("{prefix}.norm.weight"))?,
            eps,
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        mask: &MlxArray,
    ) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>), String> {
        // x: [B, T, F, C] (MLX channel-last)
        // mask: [B, T] (True = invalid/padding)

        // Zero out invalid positions
        let shape = mlxcel_core::array_shape(x);
        let batch = shape[0];
        let t_in = shape[1];

        let mask_expanded = mlxcel_core::reshape(mask, &[batch, t_in, 1, 1]);
        let zeros = mlxcel_core::zeros_like(x);
        let x = mlxcel_core::where_cond(&mask_expanded, &zeros, x);

        // Symmetric padding: (1,1,1,1) on T and F dims
        // pad_width is flat: [B_before, B_after, T_before, T_after, F_before, F_after, C_before, C_after]
        let x = mlxcel_core::pad(&x, &[0, 0, 1, 1, 1, 1, 0, 0], 0.0);

        // Conv2d with stride=2, padding=0. Routed through the fallible FFI
        // variant: the input shape is data-dependent (derived from the runtime
        // audio length), so a conv shape fault must degrade to a recoverable
        // per-request error rather than abort the process via std::terminate.
        let x = mlxcel_core::try_conv2d(&x, &self.conv_weight, 2, 2, 0, 0, 1, 1, 1)
            .map_err(|e| format!("audio SSCP conv2d failed: {e}"))?;
        let out_shape = mlxcel_core::array_shape(&x);
        let t_out = out_shape[1];

        // Downsample mask by stride 2
        let mask_sliced = slice_dim1_with_stride(mask, 2, t_out);

        // LayerNorm over channels (last dim), bias=None
        // SAFETY: norm_weight is a valid MlxArray loaded from weights, and
        // bias is intentionally null (this LayerNorm has no bias parameter).
        let x = unsafe {
            mlxcel_core::fast_layer_norm(
                &x,
                self.norm_weight.as_ref().unwrap() as *const MlxArray,
                std::ptr::null(),
                self.eps,
            )
        };

        // ReLU
        let x = mlxcel_core::relu(&x);

        Ok((x, mask_sliced))
    }
}

/// Slice dimension 1 with stride, keeping at most `max_len` elements.
/// Equivalent to `arr[:, ::stride][:, :max_len]`.
fn slice_dim1_with_stride(arr: &MlxArray, stride: i32, max_len: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(arr);
    let t = shape[1];
    let num_elements = (t + stride - 1) / stride;
    let actual_len = num_elements.min(max_len);
    let indices: Vec<i32> = (0..actual_len).map(|i| i * stride).collect();
    let indices_arr = mlxcel_core::from_slice_i32(&indices, &[actual_len]);
    mlxcel_core::take(arr, &indices_arr, 1)
}

// ---------------------------------------------------------------------------
// SubSampleConvProjection: 2x SSCPConvBlock -> flatten -> Linear
// ---------------------------------------------------------------------------

struct SubSampleConvProjection {
    layer0: SSCPConvBlock,
    layer1: SSCPConvBlock,
    input_proj_linear: UnifiedLinear,
}

impl SubSampleConvProjection {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &AudioConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            layer0: SSCPConvBlock::from_weights(
                weights,
                &format!("{prefix}.layer0"),
                config.rms_norm_eps,
            )?,
            layer1: SSCPConvBlock::from_weights(
                weights,
                &format!("{prefix}.layer1"),
                config.rms_norm_eps,
            )?,
            input_proj_linear: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.input_proj_linear"),
                group_size,
                bits,
            )?,
        })
    }

    fn forward(
        &self,
        audio_mel: &MlxArray,
        mask: &MlxArray,
    ) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>), String> {
        // audio_mel: [B, T, F_in=128]
        // Add channel dim: [B, T, F, 1]
        let x = mlxcel_core::expand_dims(audio_mel, -1);

        let (x, mask) = self.layer0.forward(&x, mask)?;
        let (x, mask) = self.layer1.forward(&x, &mask)?;

        // Flatten F*C -> [B, T, F*C]
        let shape = mlxcel_core::array_shape(&x);
        let batch = shape[0];
        let t = shape[1];
        let fc = shape[2] * shape[3];
        let x = mlxcel_core::reshape(&x, &[batch, t, fc]);

        // Project to hidden_size
        let x = self.input_proj_linear.forward(&x);
        Ok((x, mask))
    }
}

// ---------------------------------------------------------------------------
// ConformerFeedForward: pre-norm -> linear -> SiLU -> linear -> post-norm
// with residual * 0.5 and gradient clipping
// ---------------------------------------------------------------------------

struct ConformerFeedForward {
    gradient_clipping: f32,
    residual_weight: f32,
    pre_layer_norm: AudioRMSNorm,
    ffw_layer_1: AudioLinear,
    ffw_layer_2: AudioLinear,
    post_layer_norm: AudioRMSNorm,
}

impl ConformerFeedForward {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &AudioConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            gradient_clipping: config.gradient_clipping,
            residual_weight: config.residual_weight,
            pre_layer_norm: AudioRMSNorm::from_weights(
                weights,
                &format!("{prefix}.pre_layer_norm"),
                config.rms_norm_eps,
            )?,
            ffw_layer_1: AudioLinear::from_weights(
                weights,
                &format!("{prefix}.ffw_layer_1"),
                group_size,
                bits,
            )?,
            ffw_layer_2: AudioLinear::from_weights(
                weights,
                &format!("{prefix}.ffw_layer_2"),
                group_size,
                bits,
            )?,
            post_layer_norm: AudioRMSNorm::from_weights(
                weights,
                &format!("{prefix}.post_layer_norm"),
                config.rms_norm_eps,
            )?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let residual = mlxcel_core::copy(x);

        let x = clip_gradient(x, self.gradient_clipping);
        let x = self.pre_layer_norm.forward(&x);
        let x = self.ffw_layer_1.forward(&x);
        let x = mlxcel_core::silu(&x);
        let x = self.ffw_layer_2.forward(&x);
        let x = clip_gradient(&x, self.gradient_clipping);
        let x = self.post_layer_norm.forward(&x);

        // residual + x * residual_weight
        let scaled = mlxcel_core::multiply_scalar(&x, self.residual_weight);
        mlxcel_core::add(&residual, &scaled)
    }
}

fn clip_gradient(x: &MlxArray, clipping: f32) -> UniquePtr<MlxArray> {
    let min_val = mlxcel_core::full_f32(&[1], -clipping, mlxcel_core::array_dtype(x));
    let max_val = mlxcel_core::full_f32(&[1], clipping, mlxcel_core::array_dtype(x));
    mlxcel_core::clip(x, &min_val, &max_val)
}

// ---------------------------------------------------------------------------
// ConformerLightConv1d: norm -> linear(2x) -> GLU -> causal depthwise Conv1d
// -> norm -> SiLU -> linear + residual
// ---------------------------------------------------------------------------

struct ConformerLightConv1d {
    gradient_clipping: f32,
    causal_padding: usize,
    pre_layer_norm: AudioRMSNorm,
    linear_start: AudioLinear,
    depthwise_conv1d_weight: UniquePtr<MlxArray>,
    conv_norm: AudioRMSNorm,
    linear_end: AudioLinear,
}

impl ConformerLightConv1d {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &AudioConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            gradient_clipping: config.gradient_clipping,
            causal_padding: config.conv_kernel_size - 1,
            pre_layer_norm: AudioRMSNorm::from_weights(
                weights,
                &format!("{prefix}.pre_layer_norm"),
                config.rms_norm_eps,
            )?,
            linear_start: AudioLinear::from_weights(
                weights,
                &format!("{prefix}.linear_start"),
                group_size,
                bits,
            )?,
            depthwise_conv1d_weight: copy_weight(
                weights,
                &format!("{prefix}.depthwise_conv1d.weight"),
            )?,
            conv_norm: AudioRMSNorm::from_weights(
                weights,
                &format!("{prefix}.conv_norm"),
                config.rms_norm_eps,
            )?,
            linear_end: AudioLinear::from_weights(
                weights,
                &format!("{prefix}.linear_end"),
                group_size,
                bits,
            )?,
        })
    }

    fn forward(&self, x: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let residual = mlxcel_core::copy(x);

        let x = self.pre_layer_norm.forward(x);
        let x = self.linear_start.forward(&x);

        // GLU: split in half along last dim and gate
        let shape = mlxcel_core::array_shape(&x);
        let ndim = shape.len();
        let half = shape[ndim - 1] / 2;
        let x1 = slice_last_dim(&x, 0, half);
        let x2 = slice_last_dim(&x, half, half * 2);
        let gate = mlxcel_core::sigmoid(&x2);
        let x = mlxcel_core::multiply(&x1, &gate);

        // Causal padding for Conv1d: pad T dimension
        let x = mlxcel_core::pad(&x, &[0, 0, self.causal_padding as i32, 0, 0, 0], 0.0);

        // Depthwise conv1d: groups = hidden_size (each channel independently).
        // Routed through the fallible FFI variant: the input shape is
        // data-dependent (derived from the runtime audio length), so a conv
        // shape fault must degrade to a recoverable per-request error rather
        // than abort the process via std::terminate.
        let channels = mlxcel_core::array_shape(&x)[2];
        let x = mlxcel_core::try_conv1d(&x, &self.depthwise_conv1d_weight, 1, 0, 1, channels)
            .map_err(|e| format!("audio depthwise conv1d failed: {e}"))?;

        let x = clip_gradient(&x, self.gradient_clipping);
        let x = self.conv_norm.forward(&x);
        let x = mlxcel_core::silu(&x);
        let x = self.linear_end.forward(&x);

        Ok(mlxcel_core::add(&x, &residual))
    }
}

fn slice_last_dim(x: &MlxArray, start: i32, end: i32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let ndim = shape.len();
    let mut starts = vec![0; ndim];
    let mut ends = shape;
    starts[ndim - 1] = start;
    ends[ndim - 1] = end;
    mlxcel_core::slice(x, &starts, &ends)
}

// ---------------------------------------------------------------------------
// ConformerBlock: FFW1 -> Attention -> LightConv1d -> FFW2 -> clamp -> RMSNorm
// ---------------------------------------------------------------------------

struct ConformerBlock {
    gradient_clipping: f32,
    feed_forward1: ConformerFeedForward,
    self_attn: AudioAttention,
    lconv1d: ConformerLightConv1d,
    feed_forward2: ConformerFeedForward,
    norm_pre_attn: AudioRMSNorm,
    norm_post_attn: AudioRMSNorm,
    norm_out: AudioRMSNorm,
}

impl ConformerBlock {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &AudioConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            gradient_clipping: config.gradient_clipping,
            feed_forward1: ConformerFeedForward::from_weights(
                weights,
                &format!("{prefix}.feed_forward1"),
                config,
                group_size,
                bits,
            )?,
            self_attn: AudioAttention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                config,
                group_size,
                bits,
            )?,
            lconv1d: ConformerLightConv1d::from_weights(
                weights,
                &format!("{prefix}.lconv1d"),
                config,
                group_size,
                bits,
            )?,
            feed_forward2: ConformerFeedForward::from_weights(
                weights,
                &format!("{prefix}.feed_forward2"),
                config,
                group_size,
                bits,
            )?,
            norm_pre_attn: AudioRMSNorm::from_weights(
                weights,
                &format!("{prefix}.norm_pre_attn"),
                config.rms_norm_eps,
            )?,
            norm_post_attn: AudioRMSNorm::from_weights(
                weights,
                &format!("{prefix}.norm_post_attn"),
                config.rms_norm_eps,
            )?,
            norm_out: AudioRMSNorm::from_weights(
                weights,
                &format!("{prefix}.norm_out"),
                config.rms_norm_eps,
            )?,
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        mask: &MlxArray,
        causal_valid_mask: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let x = self.feed_forward1.forward(x);

        // Attention with pre/post norm and residual
        let residual = mlxcel_core::copy(&x);
        let attn_in = clip_gradient(&x, self.gradient_clipping);
        let attn_in = self.norm_pre_attn.forward(&attn_in);
        let attn_out = self.self_attn.forward(&attn_in, mask, causal_valid_mask);
        let attn_out = clip_gradient(&attn_out, self.gradient_clipping);
        let attn_out = self.norm_post_attn.forward(&attn_out);
        let x = mlxcel_core::add(&residual, &attn_out);

        // Zero out invalid positions before lconv1d
        let validity = mlxcel_core::logical_not(mask);
        let shape = mlxcel_core::array_shape(&x);
        let validity_expanded = mlxcel_core::reshape(&validity, &[shape[0], shape[1], 1]);
        let validity_f = mlxcel_core::astype(&validity_expanded, mlxcel_core::array_dtype(&x));
        let x = mlxcel_core::multiply(&x, &validity_f);

        let x = self.lconv1d.forward(&x)?;
        let x = self.feed_forward2.forward(&x);
        let x = clip_gradient(&x, self.gradient_clipping);
        Ok(self.norm_out.forward(&x))
    }
}

// ---------------------------------------------------------------------------
// AudioEncoder: SSCP + N ConformerBlocks + output projection
// ---------------------------------------------------------------------------

pub struct AudioEncoder {
    config: AudioConfig,
    subsample_conv_projection: SubSampleConvProjection,
    layers: Vec<ConformerBlock>,
    output_proj_weight: Option<UniquePtr<MlxArray>>,
    output_proj_bias: Option<UniquePtr<MlxArray>>,
}

impl AudioEncoder {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &AudioConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(ConformerBlock::from_weights(
                weights,
                &format!("{prefix}.layers.{i}"),
                config,
                group_size,
                bits,
            )?);
        }

        let (output_proj_weight, output_proj_bias) = if config.output_proj_dims.is_some() {
            (
                Some(copy_weight(
                    weights,
                    &format!("{prefix}.output_proj.weight"),
                )?),
                Some(copy_weight(weights, &format!("{prefix}.output_proj.bias"))?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            config: config.clone(),
            subsample_conv_projection: SubSampleConvProjection::from_weights(
                weights,
                &format!("{prefix}.subsample_conv_projection"),
                config,
                group_size,
                bits,
            )?,
            layers,
            output_proj_weight,
            output_proj_bias,
        })
    }

    fn build_causal_valid_mask(&self) -> UniquePtr<MlxArray> {
        let chunk_size = self.config.attention_chunk_size as i32;
        let max_future = self.config.attention_context_right as i32;
        let max_past = self.config.max_past_horizon() as i32;
        let upper_diagonal = max_past + max_future;
        let ctx = self.config.context_size() as i32;

        // lower_causal = tril(ones(ctx, chunk_size))^T
        let ones_lower = mlxcel_core::ones(&[ctx, chunk_size], mlxcel_core::dtype::FLOAT32);
        let lower_causal = mlxcel_core::tril(&ones_lower, 0);
        let lower_causal_t = mlxcel_core::transpose_axes(&lower_causal, &[1, 0]);

        // upper_causal = tril(ones(chunk_size, ctx), k=upper_diagonal)
        let ones_upper = mlxcel_core::ones(&[chunk_size, ctx], mlxcel_core::dtype::FLOAT32);
        let upper_causal = mlxcel_core::tril(&ones_upper, upper_diagonal);

        // mask = (lower * upper) as bool
        let mask = mlxcel_core::multiply(&lower_causal_t, &upper_causal);
        mlxcel_core::astype(&mask, mlxcel_core::dtype::BOOL)
    }

    pub fn forward(
        &self,
        audio_mel: &MlxArray,
        audio_mel_mask: &MlxArray,
    ) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>), String> {
        audio_probe_dump("in_mel", audio_mel);
        audio_probe_dump(
            "in_mask",
            &mlxcel_core::astype(audio_mel_mask, mlxcel_core::dtype::FLOAT32),
        );
        let (mut encodings, mut current_mask) = self
            .subsample_conv_projection
            .forward(audio_mel, audio_mel_mask)?;
        audio_probe_dump("sscp_out", &encodings);

        let causal_valid_mask = self.build_causal_valid_mask();

        for (idx, block) in self.layers.iter().enumerate() {
            encodings = block.forward(&encodings, &current_mask, &causal_valid_mask)?;
            audio_probe_dump(&format!("block_{idx:02}"), &encodings);
        }

        // Output projection (with bias)
        if let (Some(weight), Some(bias)) = (&self.output_proj_weight, &self.output_proj_bias) {
            // Manual linear + bias: x @ weight^T + bias
            let w_t = mlxcel_core::transpose_axes(weight, &[1, 0]);
            encodings = mlxcel_core::add(&mlxcel_core::matmul(&encodings, &w_t), bias);
        }

        // Truncate mask if needed
        let enc_t = mlxcel_core::array_shape(&encodings)[1];
        let mask_t = mlxcel_core::array_shape(&current_mask)[1];
        if mask_t != enc_t {
            let mask_shape = mlxcel_core::array_shape(&current_mask);
            current_mask = mlxcel_core::slice(&current_mask, &[0, 0], &[mask_shape[0], enc_t]);
        }

        // Zero out invalid (masked) positions
        let batch = mlxcel_core::array_shape(&encodings)[0];
        let mask_expanded = mlxcel_core::reshape(&current_mask, &[batch, enc_t, 1]);
        let zeros = mlxcel_core::zeros_like(&encodings);
        encodings = mlxcel_core::where_cond(&mask_expanded, &zeros, &encodings);
        audio_probe_dump("tower_out", &encodings);

        Ok((encodings, current_mask))
    }
}

#[cfg(test)]
#[path = "encoder_tests.rs"]
mod tests;
