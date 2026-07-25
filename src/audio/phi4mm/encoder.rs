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

use mlxcel_core::layers::{LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::Phi4MMAudioConfig;

fn copy_weight(
    weights: &WeightMap,
    name: &str,
    shape: &[i32],
) -> Result<UniquePtr<MlxArray>, String> {
    let weight = weights
        .get(name)
        .ok_or_else(|| format!("Phi4MM audio weight missing: {name}"))?;
    let actual = mlxcel_core::array_shape(weight);
    if actual != shape {
        return Err(format!(
            "Phi4MM audio weight {name} has shape {actual:?}, expected {shape:?}"
        ));
    }
    Ok(mlxcel_core::copy(weight))
}

fn layer_norm(weights: &WeightMap, prefix: &str, dim: i32) -> Result<LayerNorm, String> {
    Ok(LayerNorm::new(
        copy_weight(weights, &format!("{prefix}.weight"), &[dim])?,
        Some(copy_weight(weights, &format!("{prefix}.bias"), &[dim])?),
        1e-5,
    ))
}

struct Conv1d {
    weight: UniquePtr<MlxArray>,
    bias: UniquePtr<MlxArray>,
    groups: i32,
    causal_padding: i32,
}

impl Conv1d {
    fn load(
        weights: &WeightMap,
        prefix: &str,
        out: i32,
        kernel: i32,
        input_per_group: i32,
        groups: i32,
        causal_padding: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            weight: copy_weight(
                weights,
                &format!("{prefix}.weight"),
                &[out, kernel, input_per_group],
            )?,
            bias: copy_weight(weights, &format!("{prefix}.bias"), &[out])?,
            groups,
            causal_padding,
        })
    }

    fn forward(&self, x: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let x = if self.causal_padding > 0 {
            mlxcel_core::pad(x, &[0, 0, self.causal_padding, 0, 0, 0], 0.0)
        } else {
            mlxcel_core::copy(x)
        };
        let y = mlxcel_core::try_conv1d(&x, &self.weight, 1, 0, 1, self.groups)
            .map_err(|error| format!("Phi4MM Conformer conv1d failed: {error}"))?;
        Ok(mlxcel_core::add(&y, &self.bias))
    }
}

struct Conv2d {
    weight: UniquePtr<MlxArray>,
    bias: UniquePtr<MlxArray>,
    kernel: i32,
    stride: i32,
    padding: i32,
    groups: i32,
}

impl Conv2d {
    fn load(
        weights: &WeightMap,
        prefix: &str,
        out: i32,
        input_per_group: i32,
        kernel: i32,
        stride: i32,
        padding: i32,
        groups: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            weight: copy_weight(
                weights,
                &format!("{prefix}.weight"),
                &[out, kernel, kernel, input_per_group],
            )?,
            bias: copy_weight(weights, &format!("{prefix}.bias"), &[out])?,
            kernel,
            stride,
            padding,
            groups,
        })
    }

    fn forward(&self, x: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        // The released NeMo subsampler leaves `is_causal` at its false
        // default. Symmetric padding produces ceil(T/2) at each stage; the
        // top-level `causal=true` applies to the Conformer ConvModules below.
        let x = if self.padding == 0 {
            mlxcel_core::copy(x)
        } else {
            mlxcel_core::pad(
                x,
                &[
                    0,
                    0,
                    self.padding,
                    self.padding,
                    self.padding,
                    self.padding,
                    0,
                    0,
                ],
                0.0,
            )
        };
        let y = mlxcel_core::try_conv2d(
            &x,
            &self.weight,
            self.stride,
            self.stride,
            0,
            0,
            1,
            1,
            self.groups,
        )
        .map_err(|error| {
            format!(
                "Phi4MM NeMo subsampling {}x{} stride-{} conv2d failed: {error}",
                self.kernel, self.kernel, self.stride
            )
        })?;
        Ok(mlxcel_core::add(&y, &self.bias))
    }
}

struct Subsampler {
    conv0: Conv2d,
    conv1_depthwise: Conv2d,
    conv1_pointwise: Conv2d,
    conv2_depthwise: Conv2d,
    conv2_pointwise: Conv2d,
    out: UnifiedLinear,
}

#[doc(hidden)]
pub struct Phi4MMAudioSubsampleDiagnostics {
    pub conv0: UniquePtr<MlxArray>,
    pub conv1_depthwise: UniquePtr<MlxArray>,
    pub conv1_pointwise: UniquePtr<MlxArray>,
    pub conv1: UniquePtr<MlxArray>,
    pub conv2_depthwise: UniquePtr<MlxArray>,
    pub conv2_pointwise: UniquePtr<MlxArray>,
    pub conv2: UniquePtr<MlxArray>,
    pub projected: UniquePtr<MlxArray>,
}

impl Subsampler {
    fn load(weights: &WeightMap, prefix: &str, channels: i32) -> Result<Self, String> {
        Ok(Self {
            conv0: Conv2d::load(
                weights,
                &format!("{prefix}.conv.0"),
                channels,
                1,
                3,
                2,
                1,
                1,
            )?,
            conv1_depthwise: Conv2d::load(
                weights,
                &format!("{prefix}.conv.2"),
                channels,
                1,
                3,
                2,
                1,
                channels,
            )?,
            conv1_pointwise: Conv2d::load(
                weights,
                &format!("{prefix}.conv.3"),
                channels,
                channels,
                1,
                1,
                0,
                1,
            )?,
            conv2_depthwise: Conv2d::load(
                weights,
                &format!("{prefix}.conv.5"),
                channels,
                1,
                3,
                2,
                1,
                channels,
            )?,
            conv2_pointwise: Conv2d::load(
                weights,
                &format!("{prefix}.conv.6"),
                channels,
                channels,
                1,
                1,
                0,
                1,
            )?,
            out: UnifiedLinear::from_weights(weights, &format!("{prefix}.out"), 0, 0)?,
        })
    }

    fn forward(&self, x: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        Ok(self.forward_diagnostics(x)?.projected)
    }

    fn forward_diagnostics(&self, x: &MlxArray) -> Result<Phi4MMAudioSubsampleDiagnostics, String> {
        let shape = mlxcel_core::array_shape(x);
        let x = mlxcel_core::reshape(x, &[shape[0], shape[1], shape[2], 1]);
        let x = self.conv0.forward(&x)?;
        let conv0 = relu(&x);
        let conv1_depthwise = self.conv1_depthwise.forward(&conv0)?;
        let conv1_pointwise = self.conv1_pointwise.forward(&conv1_depthwise)?;
        let conv1 = relu(&conv1_pointwise);
        let conv2_depthwise = self.conv2_depthwise.forward(&conv1)?;
        let conv2_pointwise = self.conv2_pointwise.forward(&conv2_depthwise)?;
        let conv2 = relu(&conv2_pointwise);
        // PyTorch keeps convolution output as [B,C,T,F] and flattens after
        // transpose(1,2), so channel must precede frequency for embed.out.
        let x = mlxcel_core::transpose_axes(&conv2, &[0, 1, 3, 2]);
        let shape = mlxcel_core::array_shape(&x);
        let x = mlxcel_core::reshape(&x, &[shape[0], shape[1], shape[2] * shape[3]]);
        let projected = self.out.forward(&x);
        Ok(Phi4MMAudioSubsampleDiagnostics {
            conv0,
            conv1_depthwise,
            conv1_pointwise,
            conv1,
            conv2_depthwise,
            conv2_pointwise,
            conv2,
            projected,
        })
    }
}

fn relu(x: &MlxArray) -> UniquePtr<MlxArray> {
    let zero = mlxcel_core::full_f32(&[1], 0.0, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::maximum(x, &zero)
}

struct GluFeedForward {
    norm: LayerNorm,
    gate: UnifiedLinear,
    out: UnifiedLinear,
    inner: i32,
}

impl GluFeedForward {
    fn load(weights: &WeightMap, prefix: &str, dim: i32, inner: i32) -> Result<Self, String> {
        Ok(Self {
            norm: layer_norm(weights, &format!("{prefix}.layer_norm"), dim)?,
            gate: UnifiedLinear::from_weights(weights, &format!("{prefix}.net.0.linear"), 0, 0)?,
            out: UnifiedLinear::from_weights(weights, &format!("{prefix}.net.2"), 0, 0)?,
            inner,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let normalized = self.norm.forward(x);
        let gated = self.gate.forward(&normalized);
        let shape = mlxcel_core::array_shape(&gated);
        let left = mlxcel_core::slice(&gated, &[0, 0, 0], &[shape[0], shape[1], self.inner]);
        let gate = mlxcel_core::slice(
            &gated,
            &[0, 0, self.inner],
            &[shape[0], shape[1], self.inner * 2],
        );
        self.out
            .forward(&mlxcel_core::multiply(&left, &mlxcel_core::silu(&gate)))
    }
}

struct SelfAttention {
    q: UnifiedLinear,
    k: UnifiedLinear,
    v: UnifiedLinear,
    out: UnifiedLinear,
    heads: i32,
    head_dim: i32,
}

impl SelfAttention {
    fn load(weights: &WeightMap, prefix: &str, dim: i32, heads: i32) -> Result<Self, String> {
        Ok(Self {
            q: UnifiedLinear::from_weights(weights, &format!("{prefix}.linear_q"), 0, 0)?,
            k: UnifiedLinear::from_weights(weights, &format!("{prefix}.linear_k"), 0, 0)?,
            v: UnifiedLinear::from_weights(weights, &format!("{prefix}.linear_v"), 0, 0)?,
            out: UnifiedLinear::from_weights(weights, &format!("{prefix}.linear_out"), 0, 0)?,
            heads,
            head_dim: dim / heads,
        })
    }

    fn forward(&self, x: &MlxArray, relative_bias: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let (batch, length) = (shape[0], shape[1]);
        let split = |value: UniquePtr<MlxArray>| {
            let value = mlxcel_core::reshape(&value, &[batch, length, self.heads, self.head_dim]);
            mlxcel_core::transpose_axes(&value, &[0, 2, 1, 3])
        };
        let q = mlxcel_core::multiply_scalar(
            &split(self.q.forward(x)),
            (self.head_dim as f32).powf(-0.5),
        );
        let k = split(self.k.forward(x));
        let v = split(self.v.forward(x));
        let kt = mlxcel_core::transpose_axes(&k, &[0, 1, 3, 2]);
        let scores = mlxcel_core::add(&mlxcel_core::matmul(&q, &kt), relative_bias);
        let probs = mlxcel_core::softmax_precise(&scores, -1);
        let context = mlxcel_core::matmul(&probs, &v);
        let context = mlxcel_core::transpose_axes(&context, &[0, 2, 1, 3]);
        self.out
            .forward(&mlxcel_core::reshape(&context, &[batch, length, -1]))
    }
}

struct ConvModule {
    norm: LayerNorm,
    glu: Conv1d,
    glu_b1: UniquePtr<MlxArray>,
    glu_b2: UniquePtr<MlxArray>,
    depthwise: Conv1d,
    pointwise: Conv1d,
    ext_pointwise: Conv1d,
    dim: i32,
}

impl ConvModule {
    fn load(weights: &WeightMap, prefix: &str, dim: i32) -> Result<Self, String> {
        Ok(Self {
            norm: layer_norm(weights, &format!("{prefix}.layer_norm"), dim)?,
            glu: Conv1d::load(
                weights,
                &format!("{prefix}.glu.ext_pw_conv_1d"),
                dim * 2,
                1,
                dim,
                1,
                0,
            )?,
            glu_b1: copy_weight(weights, &format!("{prefix}.glu.b1"), &[1, dim, 1])?,
            glu_b2: copy_weight(weights, &format!("{prefix}.glu.b2"), &[1, dim, 1])?,
            depthwise: Conv1d::load(
                weights,
                &format!("{prefix}.dw_sep_conv_1d.dw_conv"),
                dim,
                3,
                1,
                dim,
                2,
            )?,
            pointwise: Conv1d::load(
                weights,
                &format!("{prefix}.dw_sep_conv_1d.pw_conv"),
                dim,
                1,
                dim,
                1,
                0,
            )?,
            ext_pointwise: Conv1d::load(
                weights,
                &format!("{prefix}.ext_pw_conv_1d"),
                dim,
                1,
                dim,
                1,
                0,
            )?,
            dim,
        })
    }

    fn forward(&self, x: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let normalized = self.norm.forward(x);
        let h = self.glu.forward(&normalized)?;
        let shape = mlxcel_core::array_shape(&h);
        let left = mlxcel_core::slice(&h, &[0, 0, 0], &[shape[0], shape[1], self.dim]);
        let gate = mlxcel_core::slice(&h, &[0, 0, self.dim], &[shape[0], shape[1], self.dim * 2]);
        let b1 = mlxcel_core::reshape(&self.glu_b1, &[1, 1, self.dim]);
        let b2 = mlxcel_core::reshape(&self.glu_b2, &[1, 1, self.dim]);
        let h = mlxcel_core::multiply(
            &mlxcel_core::add(&left, &b1),
            &mlxcel_core::silu(&mlxcel_core::add(&gate, &b2)),
        );
        let h = self.depthwise.forward(&h)?;
        let h = self.pointwise.forward(&h)?;
        let h = mlxcel_core::silu(&h);
        let h = self.ext_pointwise.forward(&h)?;
        Ok(h)
    }
}

struct ConformerBlock {
    ff_in: GluFeedForward,
    attention_norm: LayerNorm,
    attention: SelfAttention,
    conv: ConvModule,
    ff_out: GluFeedForward,
    output_norm: LayerNorm,
}

#[doc(hidden)]
pub struct Phi4MMAudioBlockDiagnostics {
    pub after_ff_in: UniquePtr<MlxArray>,
    pub attention: UniquePtr<MlxArray>,
    pub after_attention: UniquePtr<MlxArray>,
    pub convolution: UniquePtr<MlxArray>,
    pub after_convolution: UniquePtr<MlxArray>,
    pub ff_out: UniquePtr<MlxArray>,
    pub output: UniquePtr<MlxArray>,
}

impl ConformerBlock {
    fn load(
        weights: &WeightMap,
        prefix: &str,
        dim: i32,
        inner: i32,
        heads: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            ff_in: GluFeedForward::load(weights, &format!("{prefix}.feed_forward_in"), dim, inner)?,
            attention_norm: layer_norm(weights, &format!("{prefix}.layer_norm_att"), dim)?,
            attention: SelfAttention::load(weights, &format!("{prefix}.self_attn"), dim, heads)?,
            conv: ConvModule::load(weights, &format!("{prefix}.conv"), dim)?,
            ff_out: GluFeedForward::load(
                weights,
                &format!("{prefix}.feed_forward_out"),
                dim,
                inner,
            )?,
            output_norm: layer_norm(weights, &format!("{prefix}.layer_norm"), dim)?,
        })
    }

    fn forward(
        &self,
        x: &MlxArray,
        relative_bias: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>, String> {
        Ok(self.forward_diagnostics(x, relative_bias)?.output)
    }

    fn forward_diagnostics(
        &self,
        x: &MlxArray,
        relative_bias: &MlxArray,
    ) -> Result<Phi4MMAudioBlockDiagnostics, String> {
        let after_ff_in = mlxcel_core::add(
            x,
            &mlxcel_core::multiply_scalar(&self.ff_in.forward(x), 0.5),
        );
        let attention_input = self.attention_norm.forward(&after_ff_in);
        let attention = self.attention.forward(&attention_input, relative_bias);
        let after_attention = mlxcel_core::add(&after_ff_in, &attention);
        let convolution = self.conv.forward(&after_attention)?;
        let after_convolution = mlxcel_core::add(&after_attention, &convolution);
        let ff_out = mlxcel_core::multiply_scalar(&self.ff_out.forward(&after_convolution), 0.5);
        let pre_norm = mlxcel_core::add(&after_convolution, &ff_out);
        let output = self.output_norm.forward(&pre_norm);
        Ok(Phi4MMAudioBlockDiagnostics {
            after_ff_in,
            attention,
            after_attention,
            convolution,
            after_convolution,
            ff_out,
            output,
        })
    }
}

#[doc(hidden)]
pub struct Phi4MMAudioEncoderDiagnostics {
    pub subsample: Phi4MMAudioSubsampleDiagnostics,
    pub block0: Phi4MMAudioBlockDiagnostics,
    pub selected_blocks: Vec<(usize, UniquePtr<MlxArray>)>,
    pub encoded: UniquePtr<MlxArray>,
}

pub struct Phi4MMAudioEncoder {
    mean: UniquePtr<MlxArray>,
    invstd: UniquePtr<MlxArray>,
    subsampler: Subsampler,
    blocks: Vec<ConformerBlock>,
    relative_bias: UniquePtr<MlxArray>,
    heads: i32,
    max_distance: i32,
}

impl Phi4MMAudioEncoder {
    pub fn from_weights(
        weights: &WeightMap,
        config: &Phi4MMAudioConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let dim = config.attention_dim as i32;
        let mut blocks = Vec::with_capacity(config.num_blocks);
        for index in 0..config.num_blocks {
            blocks.push(ConformerBlock::load(
                weights,
                &format!("{prefix}.encoders.{index}"),
                dim,
                config.linear_units as i32,
                config.attention_heads as i32,
            )?);
        }
        Ok(Self {
            mean: copy_weight(
                weights,
                &format!("{prefix}.encoder_embedding.global_mean"),
                &[80],
            )?,
            invstd: copy_weight(
                weights,
                &format!("{prefix}.encoder_embedding.global_invstd"),
                &[80],
            )?,
            subsampler: Subsampler::load(
                weights,
                &format!("{prefix}.embed"),
                config.conv_channels as i32,
            )?,
            blocks,
            relative_bias: copy_weight(
                weights,
                &format!("{prefix}.relative_attention_bias_layer.bias_values.weight"),
                &[
                    (config.relative_bias_max_distance * 2) as i32,
                    config.attention_heads as i32,
                ],
            )?,
            heads: config.attention_heads as i32,
            max_distance: config.relative_bias_max_distance as i32,
        })
    }

    pub fn forward(&self, features: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let features = mlxcel_core::astype(features, mlxcel_core::array_dtype(&self.mean));
        let normalized =
            mlxcel_core::multiply(&mlxcel_core::subtract(&features, &self.mean), &self.invstd);
        let subsampled = self.subsampler.forward(&normalized)?;
        let subsampled_shape = mlxcel_core::array_shape(&subsampled);
        let length = subsampled_shape[1];
        let hidden = subsampled_shape[2];
        let mut chunks = Vec::new();
        let mut start = 0;
        while start < length {
            let end = (start + 500).min(length);
            let mut h = mlxcel_core::slice(&subsampled, &[0, start, 0], &[1, end, hidden]);
            let bias = self.relative_bias(end - start);
            for block in &self.blocks {
                h = block.forward(&h, &bias)?;
            }
            chunks.push(h);
            start = end;
        }
        Ok(crate::vision::encoders::qwen2_vl::concat_many(&chunks, 1))
    }

    /// Real-checkpoint parity checkpoints used by the independent IREE audio
    /// diagnostic. This is deliberately not part of the production inference
    /// path and only accepts the single <=500-row Conformer chunk exercised by
    /// the pinned #874 fixture.
    #[doc(hidden)]
    pub fn forward_diagnostics(
        &self,
        features: &MlxArray,
    ) -> Result<Phi4MMAudioEncoderDiagnostics, String> {
        let features = mlxcel_core::astype(features, mlxcel_core::array_dtype(&self.mean));
        let normalized =
            mlxcel_core::multiply(&mlxcel_core::subtract(&features, &self.mean), &self.invstd);
        let subsample = self.subsampler.forward_diagnostics(&normalized)?;
        let shape = mlxcel_core::array_shape(&subsample.projected);
        if shape[1] > 500 {
            return Err(format!(
                "Phi4MM audio diagnostic requires one <=500-row chunk, got {}",
                shape[1]
            ));
        }
        let bias = self.relative_bias(shape[1]);
        let block0 = self.blocks[0].forward_diagnostics(&subsample.projected, &bias)?;
        let mut h = mlxcel_core::copy(&block0.output);
        let mut selected_blocks = Vec::new();
        for (index, block) in self.blocks.iter().enumerate().skip(1) {
            h = block.forward(&h, &bias)?;
            if matches!(index, 1 | 5 | 11 | 17 | 23) {
                selected_blocks.push((index, mlxcel_core::copy(&h)));
            }
        }
        Ok(Phi4MMAudioEncoderDiagnostics {
            subsample,
            block0,
            selected_blocks,
            encoded: h,
        })
    }

    fn relative_bias(&self, length: i32) -> UniquePtr<MlxArray> {
        let mut indices = Vec::with_capacity((length * length) as usize);
        for query in 0..length {
            for key in 0..length {
                indices.push(
                    (key - query).clamp(-self.max_distance, self.max_distance - 1)
                        + self.max_distance,
                );
            }
        }
        let indices = mlxcel_core::from_slice_i32(&indices, &[length, length]);
        let bias = mlxcel_core::take(&self.relative_bias, &indices, 0);
        let bias = mlxcel_core::transpose_axes(&bias, &[2, 0, 1]);
        mlxcel_core::reshape(&bias, &[1, self.heads, length, length])
    }
}

pub struct Phi4MMAudioProjection {
    speech_1: UnifiedLinear,
    speech_2: UnifiedLinear,
    vision_1: UnifiedLinear,
    vision_2: UnifiedLinear,
}

#[doc(hidden)]
pub struct Phi4MMAudioProjectionDiagnostics {
    pub first: UniquePtr<MlxArray>,
    pub output: UniquePtr<MlxArray>,
}

impl Phi4MMAudioProjection {
    pub fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            speech_1: UnifiedLinear::from_weights(weights, &format!("{prefix}.speech.0"), 0, 0)?,
            speech_2: UnifiedLinear::from_weights(weights, &format!("{prefix}.speech.2"), 0, 0)?,
            vision_1: UnifiedLinear::from_weights(weights, &format!("{prefix}.vision.0"), 0, 0)?,
            vision_2: UnifiedLinear::from_weights(weights, &format!("{prefix}.vision.2"), 0, 0)?,
        })
    }

    pub fn forward(&self, encoded: &MlxArray, vision_mode: bool) -> UniquePtr<MlxArray> {
        self.forward_diagnostics(encoded, vision_mode).output
    }

    #[doc(hidden)]
    pub fn forward_diagnostics(
        &self,
        encoded: &MlxArray,
        vision_mode: bool,
    ) -> Phi4MMAudioProjectionDiagnostics {
        let (first, second) = if vision_mode {
            (&self.vision_1, &self.vision_2)
        } else {
            (&self.speech_1, &self.speech_2)
        };
        let first = first.forward(encoded);
        let output = second.forward(&mlxcel_core::gelu(&first));
        Phi4MMAudioProjectionDiagnostics { first, output }
    }
}
