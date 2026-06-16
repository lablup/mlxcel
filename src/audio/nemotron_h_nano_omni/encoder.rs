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

//! Parakeet/Conformer encoder used by Nemotron H Nano Omni audio.
//!
//! Faithful Rust port of upstream
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/nemotron_h_nano_omni/audio.py:
//! - `ParakeetEncoderSubsamplingConv2D` — depthwise/pointwise stack with
//!   stride-2 reduction along the time axis.
//! - `ParakeetEncoderRelPositionalEncoding` — Transformer-XL-style
//!   sinusoidal relative positional encoding (length `2T-1`).
//! - `ParakeetEncoderAttention` — multi-head self-attention with
//!   relative-position mixing (Transformer-XL biases `u, v`).
//! - `ParakeetEncoderConvolutionModule` — pointwise → GLU → depthwise
//!   Conv1d → BatchNorm → SiLU → pointwise.
//! - `ParakeetEncoderFeedForward` — Linear → SiLU → Linear.
//! - `ParakeetEncoderBlock` — FFN(0.5) → MHSA → Conv → FFN(0.5) → out-norm.
//! - `SoundEncoder` wrapper (subsampling + N blocks).
//!
//! Some primitives are not yet exposed by `mlxcel-core` (BatchNorm, GLU,
//! relative-position rel_shift). They are implemented inline here as
//! straight tensor math so we do not block on a core change.
//!
//! Used by: Nemotron H Nano Omni VLM (audio modality)

use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::NemotronOmniAudioConfig;

fn copy_weight(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(key)
        .map(|w| mlxcel_core::copy(w))
        .ok_or_else(|| format!("Audio weight not found: {key}"))
}

fn maybe_copy_weight(weights: &WeightMap, key: &str) -> Option<UniquePtr<MlxArray>> {
    weights.get(key).map(|w| mlxcel_core::copy(w))
}

/// Apply one stage of upstream `_get_output_length` in MLX-tensor-land:
/// `lengths = floor((lengths + 2*((K-1)/2) - K) / S + 1)` then back to int32.
fn next_layer_output_lengths(
    lengths: &MlxArray,
    kernel_size: i32,
    stride: i32,
) -> UniquePtr<MlxArray> {
    let dtype = mlxcel_core::array_dtype(lengths);
    let f32_lengths = mlxcel_core::astype(lengths, mlxcel_core::dtype::FLOAT32);
    let padding = (kernel_size - 1) / 2;
    let add_pad = (2 * padding - kernel_size) as f32;
    let pad_arr = mlxcel_core::full_f32(&[1], add_pad, mlxcel_core::dtype::FLOAT32);
    let added = mlxcel_core::add(&f32_lengths, &pad_arr);
    let stride_arr = mlxcel_core::full_f32(&[1], stride as f32, mlxcel_core::dtype::FLOAT32);
    let divided = mlxcel_core::divide(&added, &stride_arr);
    let one_arr = mlxcel_core::full_f32(&[1], 1.0, mlxcel_core::dtype::FLOAT32);
    let plus_one = mlxcel_core::add(&divided, &one_arr);
    let floored = mlxcel_core::floor(&plus_one);
    // Preserve int32 lengths going into the next layer.
    let _ = dtype;
    mlxcel_core::astype(&floored, mlxcel_core::dtype::INT32)
}

/// Apply N stages of `_get_output_length` (the same kernel/stride for
/// every stage) — equivalent to upstream
/// `_get_subsampling_output_length`.
fn subsampling_output_lengths_arr(
    lengths: &MlxArray,
    kernel_size: i32,
    stride: i32,
    num_stages: usize,
) -> UniquePtr<MlxArray> {
    let mut current = mlxcel_core::astype(lengths, mlxcel_core::dtype::INT32);
    for _ in 0..num_stages {
        current = next_layer_output_lengths(&current, kernel_size, stride);
    }
    current
}

/// LayerNorm with optional bias (Parakeet uses bias=True). Mirrors
/// `nn.LayerNorm(hidden_size)` from upstream.
struct ParakeetLayerNorm {
    weight: UniquePtr<MlxArray>,
    bias: Option<UniquePtr<MlxArray>>,
    eps: f32,
}

impl ParakeetLayerNorm {
    fn from_weights(weights: &WeightMap, prefix: &str, eps: f32) -> Result<Self, String> {
        Ok(Self {
            weight: copy_weight(weights, &format!("{prefix}.weight"))?,
            bias: maybe_copy_weight(weights, &format!("{prefix}.bias")),
            eps,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // SAFETY: weight is guaranteed valid (loaded from weights map).
        // bias may be null (LayerNorm with bias=False is valid in MLX).
        let weight_ptr: *const MlxArray = self
            .weight
            .as_ref()
            .expect("LayerNorm weight UniquePtr is non-null after load");
        let bias_ptr: *const MlxArray = match &self.bias {
            Some(b) => b
                .as_ref()
                .expect("LayerNorm bias UniquePtr is non-null after load"),
            None => std::ptr::null(),
        };
        unsafe { mlxcel_core::fast_layer_norm(x, weight_ptr, bias_ptr, self.eps) }
    }
}

/// Inference-time BatchNorm: `y = (x - running_mean) / sqrt(running_var + eps) * weight + bias`.
///
/// The released checkpoint stores `weight`, `bias`, `running_mean`,
/// `running_var`, and (filtered out by the loader) `num_batches_tracked`.
/// MLX `nn.BatchNorm` defaults to `eps=1e-5`.
struct ParakeetBatchNorm {
    weight: UniquePtr<MlxArray>,
    bias: UniquePtr<MlxArray>,
    running_mean: UniquePtr<MlxArray>,
    running_var: UniquePtr<MlxArray>,
    eps: f32,
}

impl ParakeetBatchNorm {
    fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            weight: copy_weight(weights, &format!("{prefix}.weight"))?,
            bias: copy_weight(weights, &format!("{prefix}.bias"))?,
            running_mean: copy_weight(weights, &format!("{prefix}.running_mean"))?,
            running_var: copy_weight(weights, &format!("{prefix}.running_var"))?,
            eps: 1e-5,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        // x shape: [B, T, C]. mean/var/weight/bias are [C].
        let centered = mlxcel_core::subtract(x, &self.running_mean);
        let eps_arr = mlxcel_core::full_f32(&[1], self.eps, mlxcel_core::array_dtype(x));
        let var_eps = mlxcel_core::add(&self.running_var, &eps_arr);
        let inv_std = mlxcel_core::rsqrt(&var_eps);
        let scaled = mlxcel_core::multiply(&centered, &inv_std);
        let with_weight = mlxcel_core::multiply(&scaled, &self.weight);
        mlxcel_core::add(&with_weight, &self.bias)
    }
}

/// `nn.glu(x, axis=-1)` → split last dim in half and `a * sigmoid(b)`.
fn glu_last_axis(x: &MlxArray) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let last = *shape.last().expect("glu: input must have rank >= 1");
    let half = last / 2;
    let starts_a: Vec<i32> = shape.iter().map(|_| 0).collect();
    let mut ends_a: Vec<i32> = shape.clone();
    let mut starts_b: Vec<i32> = shape.iter().map(|_| 0).collect();
    let mut ends_b: Vec<i32> = shape.clone();
    let last_idx = shape.len() - 1;
    ends_a[last_idx] = half;
    starts_b[last_idx] = half;
    ends_b[last_idx] = last;
    let a = mlxcel_core::slice(x, &starts_a, &ends_a);
    let b = mlxcel_core::slice(x, &starts_b, &ends_b);
    let gate = mlxcel_core::sigmoid(&b);
    mlxcel_core::multiply(&a, &gate)
}

// ---------------------------------------------------------------------------
// SubsamplingConv2D
// ---------------------------------------------------------------------------

/// Each Conv2d weight is stored in MLX layout `[out, kh, kw, in/groups]`
/// after the loader applies the upstream `transpose(0,2,3,1)` tweak.
struct SubsamplingConv2DLayer {
    weight: UniquePtr<MlxArray>,
    stride: i32,
    padding: i32,
    groups: i32,
}

impl SubsamplingConv2DLayer {
    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        mlxcel_core::conv2d(
            x,
            &self.weight,
            self.stride,
            self.stride,
            self.padding,
            self.padding,
            1,
            1,
            self.groups,
        )
    }
}

struct ParakeetSubsamplingConv2D {
    layers: Vec<SubsamplingConv2DLayer>,
    /// Index sequence describing whether each conv is followed by a ReLU
    /// before the next conv. Mirrors the upstream `[Conv2d, ReLU,
    /// (Conv2d, Conv2d, ReLU)*]` layout.
    relu_after: Vec<bool>,
    linear: UnifiedLinear,
    // Time-domain output length per conv layer (using upstream's
    // `(L + 2P - K)/S + 1` per-stride-2 stage, and identity for stride=1).
    layer_strides: Vec<usize>,
    config: NemotronOmniAudioConfig,
    // Output frequency length after all subsampling stages, used for the
    // flattening reshape into `[B, T, F*C]`.
    out_freq_length: i32,
}

impl ParakeetSubsamplingConv2D {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &NemotronOmniAudioConfig,
    ) -> Result<Self, String> {
        let stride = config.subsampling_conv_stride as i32;
        let kernel = config.subsampling_conv_kernel_size as i32;
        let padding = (kernel - 1) / 2;
        let channels = config.subsampling_conv_channels as i32;
        let num_layers = config.num_subsampling_layers();

        // Mirror the upstream `for _ in range(num_layers - 1): [Conv2d
        // (depthwise stride=2), Conv2d 1x1 pointwise, ReLU]` after the
        // initial `[Conv2d (stride=2), ReLU]`. Layer indices in the
        // checkpoint follow the upstream `nn.Module` `layers` list.
        let mut layers = Vec::new();
        let mut relu_after = Vec::new();
        let mut layer_strides = Vec::new();

        // Layer 0: Conv2d(1 -> channels, stride=2, padding, groups=1).
        layers.push(SubsamplingConv2DLayer {
            weight: copy_weight(weights, &format!("{prefix}.layers.0.weight"))?,
            stride,
            padding,
            groups: 1,
        });
        relu_after.push(true);
        layer_strides.push(stride as usize);
        let _ = channels; // used later for sanity checks if needed.

        // Subsequent layers: pairs of (depthwise stride=2, pointwise
        // stride=1) followed by ReLU. The MLX implementation uses
        // index-based naming `layers.1`, `layers.2`, ... in the
        // checkpoint where the `nn.ReLU` modules consume even slots.
        // The upstream code uses a flat `self.layers` list:
        //   layer 0 -> Conv2d
        //   layer 1 -> ReLU
        //   layer 2 -> Conv2d (depthwise)
        //   layer 3 -> Conv2d (1x1 pointwise)
        //   layer 4 -> ReLU
        //   layer 5 -> Conv2d (depthwise)
        //   layer 6 -> Conv2d (1x1 pointwise)
        //   layer 7 -> ReLU
        //   ...
        // The ReLU layers carry no weights, so the checkpoint only has
        // `layers.{0,2,3,5,6,...}.weight`.
        let mut layer_idx = 2usize;
        for _ in 0..(num_layers - 1) {
            // Depthwise conv (stride=2).
            layers.push(SubsamplingConv2DLayer {
                weight: copy_weight(weights, &format!("{prefix}.layers.{layer_idx}.weight"))?,
                stride,
                padding,
                groups: config.subsampling_conv_channels as i32,
            });
            relu_after.push(false);
            layer_strides.push(stride as usize);
            layer_idx += 1;
            // Pointwise conv (1x1, stride=1).
            layers.push(SubsamplingConv2DLayer {
                weight: copy_weight(weights, &format!("{prefix}.layers.{layer_idx}.weight"))?,
                stride: 1,
                padding: 0,
                groups: 1,
            });
            relu_after.push(true);
            layer_strides.push(1);
            layer_idx += 2; // skip the implicit ReLU module.
        }

        // Output frequency length per upstream:
        //   out_length = num_mel_bins // (stride ** num_layers)
        let stride_pow = (stride as usize).pow(num_layers as u32);
        let out_freq_length = (config.num_mel_bins / stride_pow) as i32;
        let in_features = (config.subsampling_conv_channels as i32) * out_freq_length;

        let linear = UnifiedLinear::from_weights(
            weights,
            &format!("{prefix}.linear"),
            // The released checkpoint quantizes audio linears with the
            // model defaults; group_size/bits=0 disables quantization
            // detection but `from_weights` falls back to the
            // non-quantized branch when scales are absent.
            64,
            4,
        )?;
        let _ = in_features; // documented by the linear's input weight shape.

        Ok(Self {
            layers,
            relu_after,
            linear,
            layer_strides,
            config: config.clone(),
            out_freq_length,
        })
    }

    fn forward(
        &self,
        input_features: &MlxArray,
        attention_mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // input_features: [B, T, num_mel_bins]
        // Add channel dim: [B, T, F, 1]
        let mut x = mlxcel_core::expand_dims(input_features, -1);

        // Track per-clip time lengths for masking, mirroring upstream
        // `current_lengths` semantics. The lengths live on-device as an
        // int32 [B] array so we don't need a CPU sync per layer.
        let mut current_lengths: Option<UniquePtr<MlxArray>> = attention_mask.map(|mask| {
            let summed = mlxcel_core::sum_axis(mask, -1, false);
            mlxcel_core::astype(&summed, mlxcel_core::dtype::INT32)
        });

        for (idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x);
            // Mask after each conv layer (matches upstream ordering).
            if let Some(lengths) = current_lengths.as_mut() {
                let stride = self.layer_strides[idx];
                if stride != 1 {
                    *lengths = next_layer_output_lengths(
                        lengths,
                        self.config.subsampling_conv_kernel_size as i32,
                        stride as i32,
                    );
                }
                let shape = mlxcel_core::array_shape(&x);
                let batch = shape[0];
                let t = shape[1];
                let arange = mlxcel_core::arange_i32(0, t, 1);
                let arange = mlxcel_core::reshape(&arange, &[1, t]);
                let lengths_2d = mlxcel_core::reshape(lengths, &[batch, 1]);
                let mask = mlxcel_core::less(&arange, &lengths_2d);
                let mask = mlxcel_core::astype(&mask, mlxcel_core::array_dtype(&x));
                let mask = mlxcel_core::reshape(&mask, &[batch, t, 1, 1]);
                x = mlxcel_core::multiply(&x, &mask);
            }
            if self.relu_after[idx] {
                x = mlxcel_core::relu(&x);
            }
        }

        // Flatten F*C → [B, T, F*C] (transpose first to match upstream
        // `transpose(0,1,3,2)`):
        let pre_shape = mlxcel_core::array_shape(&x);
        let batch = pre_shape[0];
        let t = pre_shape[1];
        let fdim = pre_shape[2];
        let cdim = pre_shape[3];
        let _ = (fdim, cdim, self.out_freq_length); // shape sanity left implicit.
        let transposed = mlxcel_core::transpose_axes(&x, &[0, 1, 3, 2]);
        let flattened = mlxcel_core::reshape(&transposed, &[batch, t, -1]);

        self.linear.forward(&flattened)
    }
}

// ---------------------------------------------------------------------------
// Relative positional encoding
// ---------------------------------------------------------------------------

/// Build the descending position vector used by the Parakeet relative
/// positional encoding. Mirrors upstream
/// `mx.arange(seq_length - 1, -seq_length, -1)` exactly:
/// for `seq_length = T` returns `[T-1, T-2, ..., 0, -1, ..., -(T-1)]`
/// (length `2T - 1`).
///
/// Extracted as a free helper so unit tests can pin the position vector
/// without standing up a full `MlxArray`.
pub(crate) fn parakeet_rel_position_vector(seq_length: usize) -> Vec<f32> {
    let len = 2 * seq_length - 1;
    let start = seq_length as i64 - 1;
    (0..len).map(|i| (start - i as i64) as f32).collect()
}

/// Sinusoidal relative positional encoding (length `2T - 1`, hidden_size).
///
/// Computed at runtime per call (depends on the post-subsampling time
/// length). Cheap given small `T`.
fn parakeet_rel_position_embedding(
    seq_length: usize,
    hidden_size: usize,
    dtype: i32,
) -> UniquePtr<MlxArray> {
    let positions = parakeet_rel_position_vector(seq_length);
    let positions_arr = mlxcel_core::from_slice_f32(&positions, &[(2 * seq_length - 1) as i32]);

    // inv_freq = 1 / (10000 ** (arange(0, hidden, 2) / hidden))
    let half = hidden_size / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|k| {
            let denom = 10_000f32.powf((2 * k) as f32 / hidden_size as f32);
            1.0 / denom
        })
        .collect();
    let inv_freq_arr = mlxcel_core::from_slice_f32(&inv_freq, &[half as i32]);

    // freqs = positions[:, None] * inv_freq[None, :]
    let positions_2d = mlxcel_core::reshape(&positions_arr, &[(2 * seq_length - 1) as i32, 1]);
    let inv_freq_2d = mlxcel_core::reshape(&inv_freq_arr, &[1, half as i32]);
    let freqs = mlxcel_core::multiply(&positions_2d, &inv_freq_2d);

    // sin / cos stack to produce hidden_size channels.
    let sin_t = mlxcel_core::sin(&freqs);
    let cos_t = mlxcel_core::cos(&freqs);
    // stack along last axis -> shape [2T-1, half, 2]
    let stacked = mlxcel_core::stack_owned(&[sin_t, cos_t], -1);
    let pos_embed =
        mlxcel_core::reshape(&stacked, &[(2 * seq_length - 1) as i32, hidden_size as i32]);

    mlxcel_core::astype(&pos_embed, dtype)
}

// ---------------------------------------------------------------------------
// Feed-forward
// ---------------------------------------------------------------------------

struct ParakeetFeedForward {
    linear1: UnifiedLinear,
    linear2: UnifiedLinear,
}

impl ParakeetFeedForward {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            linear1: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.linear1"),
                group_size,
                bits,
            )?,
            linear2: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.linear2"),
                group_size,
                bits,
            )?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.linear1.forward(x);
        let h = mlxcel_core::silu(&h);
        self.linear2.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// Convolution module
// ---------------------------------------------------------------------------

struct ParakeetConvModule {
    pointwise1: UniquePtr<MlxArray>,
    depthwise: UniquePtr<MlxArray>,
    norm: ParakeetBatchNorm,
    pointwise2: UniquePtr<MlxArray>,
    pointwise1_bias: Option<UniquePtr<MlxArray>>,
    depthwise_bias: Option<UniquePtr<MlxArray>>,
    pointwise2_bias: Option<UniquePtr<MlxArray>>,
    kernel: usize,
    channels: i32,
}

impl ParakeetConvModule {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &NemotronOmniAudioConfig,
    ) -> Result<Self, String> {
        let pointwise1 = copy_weight(weights, &format!("{prefix}.pointwise_conv1.weight"))?;
        let depthwise = copy_weight(weights, &format!("{prefix}.depthwise_conv.weight"))?;
        let pointwise2 = copy_weight(weights, &format!("{prefix}.pointwise_conv2.weight"))?;
        let pointwise1_bias = if config.convolution_bias {
            Some(copy_weight(
                weights,
                &format!("{prefix}.pointwise_conv1.bias"),
            )?)
        } else {
            maybe_copy_weight(weights, &format!("{prefix}.pointwise_conv1.bias"))
        };
        let depthwise_bias = if config.convolution_bias {
            Some(copy_weight(
                weights,
                &format!("{prefix}.depthwise_conv.bias"),
            )?)
        } else {
            maybe_copy_weight(weights, &format!("{prefix}.depthwise_conv.bias"))
        };
        let pointwise2_bias = if config.convolution_bias {
            Some(copy_weight(
                weights,
                &format!("{prefix}.pointwise_conv2.bias"),
            )?)
        } else {
            maybe_copy_weight(weights, &format!("{prefix}.pointwise_conv2.bias"))
        };
        let norm = ParakeetBatchNorm::from_weights(weights, &format!("{prefix}.norm"))?;
        Ok(Self {
            pointwise1,
            depthwise,
            norm,
            pointwise2,
            pointwise1_bias,
            depthwise_bias,
            pointwise2_bias,
            kernel: config.conv_kernel_size,
            channels: config.hidden_size as i32,
        })
    }

    /// `x: [B, T, C]`, `validity_mask: [B, T]` (true = valid frame, the
    /// caller-supplied mask is `output_mask` from the encoder so we
    /// match the upstream "all_masked_rows" guard).
    fn forward(&self, x: &MlxArray, all_masked_rows: Option<&MlxArray>) -> UniquePtr<MlxArray> {
        // Pointwise conv 1: out = 2 * channels (for GLU split).
        let h = mlxcel_core::conv1d(x, &self.pointwise1, 1, 0, 1, 1);
        let h = if let Some(bias) = &self.pointwise1_bias {
            mlxcel_core::add(&h, bias)
        } else {
            h
        };
        // GLU on last axis: a * sigmoid(b).
        let h = glu_last_axis(&h);

        // Optional zero-out of fully-masked rows (preserves upstream
        // numerical equivalence for fully-padded batches).
        let h = if let Some(rows) = all_masked_rows {
            let zeros = mlxcel_core::zeros_like(&h);
            // rows is [B, T, 1] when caller supplies it (already broadcast-ready).
            mlxcel_core::where_cond(rows, &zeros, &h)
        } else {
            h
        };

        // Depthwise Conv1d, padding = (kernel-1)/2, groups = channels.
        let pad = ((self.kernel - 1) / 2) as i32;
        let h = mlxcel_core::conv1d(&h, &self.depthwise, 1, pad, 1, self.channels);
        let h = if let Some(bias) = &self.depthwise_bias {
            mlxcel_core::add(&h, bias)
        } else {
            h
        };

        // BatchNorm over channels (last axis on [B, T, C]).
        let h = self.norm.forward(&h);
        // SiLU.
        let h = mlxcel_core::silu(&h);

        // Pointwise conv 2: channels -> channels.
        let h = mlxcel_core::conv1d(&h, &self.pointwise2, 1, 0, 1, 1);
        if let Some(bias) = &self.pointwise2_bias {
            mlxcel_core::add(&h, bias)
        } else {
            h
        }
    }
}

// ---------------------------------------------------------------------------
// Self-attention with relative positional bias.
// ---------------------------------------------------------------------------

struct ParakeetAttention {
    q_proj: UnifiedLinear,
    k_proj: UnifiedLinear,
    v_proj: UnifiedLinear,
    o_proj: UnifiedLinear,
    relative_k_proj: UnifiedLinear,
    bias_u: UniquePtr<MlxArray>,
    bias_v: UniquePtr<MlxArray>,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl ParakeetAttention {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &NemotronOmniAudioConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let head_dim = config.head_dim() as i32;
        Ok(Self {
            q_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.q_proj"),
                group_size,
                bits,
            )?,
            k_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.k_proj"),
                group_size,
                bits,
            )?,
            v_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.v_proj"),
                group_size,
                bits,
            )?,
            o_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.o_proj"),
                group_size,
                bits,
            )?,
            relative_k_proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.relative_k_proj"),
                group_size,
                bits,
            )?,
            bias_u: copy_weight(weights, &format!("{prefix}.bias_u"))?,
            bias_v: copy_weight(weights, &format!("{prefix}.bias_v"))?,
            num_heads: config.num_attention_heads as i32,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
        })
    }

    /// Transformer-XL `_rel_shift`: shifts the BD matrix so the relative
    /// indices line up with absolute positions.
    fn rel_shift(&self, scores: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(scores);
        let b = shape[0];
        let h = shape[1];
        let q = shape[2];
        let p = shape[3];
        // pad on last axis with width (1, 0).
        let padded = mlxcel_core::pad(scores, &[0, 0, 0, 0, 0, 0, 1, 0], 0.0);
        // reshape to [B, H, P+1, Q].
        let reshaped = mlxcel_core::reshape(&padded, &[b, h, p + 1, q]);
        // drop the first row -> [B, H, P, Q].
        let starts = vec![0, 0, 1, 0];
        let ends = vec![b, h, p + 1, q];
        let cropped = mlxcel_core::slice(&reshaped, &starts, &ends);
        // reshape back to [B, H, Q, P].
        mlxcel_core::reshape(&cropped, &[b, h, q, p])
    }

    fn forward(
        &self,
        hidden_states: &MlxArray,
        position_embeddings: &MlxArray,
        attention_mask: Option<&MlxArray>,
        valid_queries_f: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(hidden_states);
        let batch = shape[0];
        let seq = shape[1];

        let q = self.q_proj.forward(hidden_states);
        let k = self.k_proj.forward(hidden_states);
        let v = self.v_proj.forward(hidden_states);

        // [B, T, H, D] -> [B, H, T, D]
        let reshape_qkv = |t: &MlxArray| -> UniquePtr<MlxArray> {
            let r = mlxcel_core::reshape(t, &[batch, seq, self.num_heads, self.head_dim]);
            mlxcel_core::transpose_axes(&r, &[0, 2, 1, 3])
        };
        let q = reshape_qkv(&q);
        let k = reshape_qkv(&k);
        let v = reshape_qkv(&v);

        // Add bias_u for the AC term (used inside SDPA).
        let bias_u_view =
            mlxcel_core::reshape(&self.bias_u, &[1, self.num_heads, 1, self.head_dim]);
        let q_with_u = mlxcel_core::add(&q, &bias_u_view);

        // Add bias_v for the BD term.
        let bias_v_view =
            mlxcel_core::reshape(&self.bias_v, &[1, self.num_heads, 1, self.head_dim]);
        let q_with_v = mlxcel_core::add(&q, &bias_v_view);

        // Relative key projection over `2T-1` positions: shape
        // [B, 2T-1, H*D] -> [B, 2T-1, H, D] -> [B, H, 2T-1, D].
        let rel_k = self.relative_k_proj.forward(position_embeddings);
        let rel_k = mlxcel_core::reshape(&rel_k, &[batch, -1, self.num_heads, self.head_dim]);
        let rel_k_t = mlxcel_core::transpose_axes(&rel_k, &[0, 2, 3, 1]);
        // matrix_bd = q_with_v @ rel_k_t  -> [B, H, T, 2T-1]
        let matrix_bd = mlxcel_core::matmul(&q_with_v, &rel_k_t);
        // rel_shift -> [B, H, T, 2T-1] (still 2T-1 columns; upstream
        // slices `[..., :T]` afterwards). Apply the shift, then trim.
        let matrix_bd = self.rel_shift(&matrix_bd);
        let bd_shape = mlxcel_core::array_shape(&matrix_bd);
        let bd_starts = vec![0, 0, 0, 0];
        let bd_ends = vec![bd_shape[0], bd_shape[1], bd_shape[2], seq];
        let matrix_bd = mlxcel_core::slice(&matrix_bd, &bd_starts, &bd_ends);
        let matrix_bd = mlxcel_core::multiply_scalar(&matrix_bd, self.scale);

        // Apply attention mask additive bias to BD.
        let matrix_bd = if let Some(mask) = attention_mask {
            // Where mask is False, replace with -infinity. mask shape is
            // [B, 1, T, T]. Use `mx.finfo(dtype).min` proxy via large
            // negative float consistent with upstream.
            let dtype = mlxcel_core::array_dtype(&matrix_bd);
            let neg_inf = mlxcel_core::full_f32(&[1], dtype_min(dtype), dtype);
            let neg_inf =
                mlxcel_core::broadcast_to(&neg_inf, &mlxcel_core::array_shape(&matrix_bd));
            mlxcel_core::where_cond(mask, &matrix_bd, &neg_inf)
        } else {
            matrix_bd
        };

        // SAFETY: q_with_u/k/v all come from contiguous matmul/reshape.
        // matrix_bd is a valid MlxArray reference.
        let attn = unsafe {
            let mask_ptr: *const MlxArray = matrix_bd
                .as_ref()
                .expect("matrix_bd UniquePtr is non-null after matmul");
            mlxcel_core::scaled_dot_product_attention(&q_with_u, &k, &v, self.scale, mask_ptr)
        };

        // Upstream zeros out queries that have no valid keys (when
        // `attention_mask` is provided).
        let attn = if let Some(valid) = valid_queries_f {
            mlxcel_core::multiply(&attn, valid)
        } else {
            attn
        };

        // [B, H, T, D] -> [B, T, H, D] -> [B, T, H*D]
        let attn = mlxcel_core::transpose_axes(&attn, &[0, 2, 1, 3]);
        let attn_shape = mlxcel_core::array_shape(&attn);
        let attn = mlxcel_core::reshape(&attn, &[attn_shape[0], attn_shape[1], -1]);

        self.o_proj.forward(&attn)
    }
}

fn dtype_min(dtype: i32) -> f32 {
    // Approximate dtype-min. The exact MLX semantics call
    // `mx.finfo(dtype).min`; for f16 and bf16 we use values within
    // their representable range (-65504 / -3.39e38 are the actual
    // limits but using a safely-large-but-finite negative is what
    // upstream uses for `where(mask, x, finfo.min)`).
    match dtype {
        // FLOAT16 -> -65504.0 (max finite negative for f16).
        d if d == mlxcel_core::dtype::FLOAT16 => -65504.0,
        // BFLOAT16 -> ~-3.39e38; clamp to a representable value.
        d if d == mlxcel_core::dtype::BFLOAT16 => -3.38e38,
        _ => f32::MIN,
    }
}

// ---------------------------------------------------------------------------
// Conformer block
// ---------------------------------------------------------------------------

struct ParakeetEncoderBlock {
    feed_forward1: ParakeetFeedForward,
    self_attn: ParakeetAttention,
    conv: ParakeetConvModule,
    feed_forward2: ParakeetFeedForward,
    norm_feed_forward1: ParakeetLayerNorm,
    norm_self_att: ParakeetLayerNorm,
    norm_conv: ParakeetLayerNorm,
    norm_feed_forward2: ParakeetLayerNorm,
    norm_out: ParakeetLayerNorm,
}

impl ParakeetEncoderBlock {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &NemotronOmniAudioConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        // LayerNorm eps is the MLX default 1e-5.
        let ln_eps = 1e-5f32;
        Ok(Self {
            feed_forward1: ParakeetFeedForward::from_weights(
                weights,
                &format!("{prefix}.feed_forward1"),
                group_size,
                bits,
            )?,
            self_attn: ParakeetAttention::from_weights(
                weights,
                &format!("{prefix}.self_attn"),
                config,
                group_size,
                bits,
            )?,
            conv: ParakeetConvModule::from_weights(weights, &format!("{prefix}.conv"), config)?,
            feed_forward2: ParakeetFeedForward::from_weights(
                weights,
                &format!("{prefix}.feed_forward2"),
                group_size,
                bits,
            )?,
            norm_feed_forward1: ParakeetLayerNorm::from_weights(
                weights,
                &format!("{prefix}.norm_feed_forward1"),
                ln_eps,
            )?,
            norm_self_att: ParakeetLayerNorm::from_weights(
                weights,
                &format!("{prefix}.norm_self_att"),
                ln_eps,
            )?,
            norm_conv: ParakeetLayerNorm::from_weights(
                weights,
                &format!("{prefix}.norm_conv"),
                ln_eps,
            )?,
            norm_feed_forward2: ParakeetLayerNorm::from_weights(
                weights,
                &format!("{prefix}.norm_feed_forward2"),
                ln_eps,
            )?,
            norm_out: ParakeetLayerNorm::from_weights(
                weights,
                &format!("{prefix}.norm_out"),
                ln_eps,
            )?,
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        position_embeddings: &MlxArray,
        attention_mask: Option<&MlxArray>,
        valid_queries_f: Option<&MlxArray>,
        all_masked_rows: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Half-FFN.
        let residual = mlxcel_core::copy(x);
        let n = self.norm_feed_forward1.forward(x);
        let f1 = self.feed_forward1.forward(&n);
        let half = mlxcel_core::multiply_scalar(&f1, 0.5);
        let h = mlxcel_core::add(&residual, &half);

        // Self-attention.
        let normed = self.norm_self_att.forward(&h);
        let attn = self.self_attn.forward(
            &normed,
            position_embeddings,
            attention_mask,
            valid_queries_f,
        );
        let h = mlxcel_core::add(&h, &attn);

        // Conv module.
        let normed = self.norm_conv.forward(&h);
        let conv_out = self.conv.forward(&normed, all_masked_rows);
        let h = mlxcel_core::add(&h, &conv_out);

        // Half-FFN.
        let normed = self.norm_feed_forward2.forward(&h);
        let f2 = self.feed_forward2.forward(&normed);
        let half = mlxcel_core::multiply_scalar(&f2, 0.5);
        let h = mlxcel_core::add(&h, &half);

        // Out norm.
        self.norm_out.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// Top-level encoder wrapper
// ---------------------------------------------------------------------------

/// Top-level Parakeet/Conformer encoder used by Nemotron H Nano Omni.
pub struct NemotronOmniSoundEncoder {
    config: NemotronOmniAudioConfig,
    subsampling: ParakeetSubsamplingConv2D,
    layers: Vec<ParakeetEncoderBlock>,
    input_scale: f32,
}

impl NemotronOmniSoundEncoder {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &NemotronOmniAudioConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let inner = format!("{prefix}.encoder");
        let subsampling = ParakeetSubsamplingConv2D::from_weights(
            weights,
            &format!("{inner}.subsampling"),
            config,
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            layers.push(ParakeetEncoderBlock::from_weights(
                weights,
                &format!("{inner}.layers.{layer_idx}"),
                config,
                group_size,
                bits,
            )?);
        }
        let input_scale = if config.scale_input {
            (config.hidden_size as f32).sqrt()
        } else {
            1.0
        };
        Ok(Self {
            config: config.clone(),
            subsampling,
            layers,
            input_scale,
        })
    }

    pub fn config(&self) -> &NemotronOmniAudioConfig {
        &self.config
    }

    /// `input_features: [B, T, num_mel_bins]`; `attention_mask: [B, T]`.
    /// Returns `[B, T_out, hidden_size]`.
    pub fn forward(
        &self,
        input_features: &MlxArray,
        attention_mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mut hidden = self.subsampling.forward(input_features, attention_mask);
        if self.input_scale != 1.0 {
            hidden = mlxcel_core::multiply_scalar(&hidden, self.input_scale);
        }

        let shape = mlxcel_core::array_shape(&hidden);
        let batch = shape[0];
        let seq = shape[1] as usize;

        let dtype = mlxcel_core::array_dtype(&hidden);
        let pos_embed = parakeet_rel_position_embedding(seq, self.config.hidden_size, dtype);
        // Broadcast to [B, 2T-1, hidden_size].
        let pos_embed = mlxcel_core::reshape(
            &pos_embed,
            &[1, (2 * seq - 1) as i32, self.config.hidden_size as i32],
        );
        let pos_embed = mlxcel_core::broadcast_to(
            &pos_embed,
            &[batch, (2 * seq - 1) as i32, self.config.hidden_size as i32],
        );

        // Build the [B, 1, T, T] attention mask + ancillary masks if a
        // caller-supplied mask is present.
        let (full_mask, valid_queries_f, all_masked_rows) = if let Some(mask) = attention_mask {
            // output_lengths = subsampling_output_length(sum(mask)).
            // We compute on-device to avoid CPU sync.
            let summed = mlxcel_core::sum_axis(mask, -1, false);
            let kernel = self.config.subsampling_conv_kernel_size as i32;
            let stride = self.config.subsampling_conv_stride as i32;
            let stages = self.config.num_subsampling_layers();
            let output_lengths = subsampling_output_lengths_arr(&summed, kernel, stride, stages);
            let output_lengths_arr = mlxcel_core::reshape(&output_lengths, &[batch, 1]);
            let arange = mlxcel_core::arange_i32(0, seq as i32, 1);
            let arange = mlxcel_core::reshape(&arange, &[1, seq as i32]);
            let output_mask = mlxcel_core::less(&arange, &output_lengths_arr);
            // attention_mask = output_mask[:, None, :, None] & output_mask[:, None, None, :]
            let oma = mlxcel_core::reshape(&output_mask, &[batch, 1, seq as i32, 1]);
            let omb = mlxcel_core::reshape(&output_mask, &[batch, 1, 1, seq as i32]);
            let full = mlxcel_core::logical_and(&oma, &omb);
            // valid_queries = any(full_mask, axis=-1) -> [B, 1, T]
            let valid_q = mlxcel_core::any_axis(&full, -1, false);
            let valid_q = mlxcel_core::astype(&valid_q, dtype);
            let valid_q = mlxcel_core::reshape(&valid_q, &[batch, 1, seq as i32, 1]);

            // Conv all_masked_rows guard: rows where every element of
            // the [B, 1, T, T] mask is False -> [B, T, 1] for broadcast
            // against the conv hidden state.
            let row_any = mlxcel_core::any_axis(&full, -1, false);
            let row_any = mlxcel_core::reshape(&row_any, &[batch, seq as i32, 1]);
            let row_none = mlxcel_core::logical_not(&row_any);
            (Some(full), Some(valid_q), Some(row_none))
        } else {
            (None, None, None)
        };

        for block in &self.layers {
            hidden = block.forward(
                &hidden,
                &pos_embed,
                full_mask.as_deref(),
                valid_queries_f.as_deref(),
                all_masked_rows.as_deref(),
            );
        }

        hidden
    }
}
