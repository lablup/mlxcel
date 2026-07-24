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

//! Phi4MM's pinned SpeechLib/Conformer audio artifact contract.
//!
//! The audio encoder is compiled as a module separate from the decoder bundle:
//! it owns only audio encoder/projection weights and returns projected decoder
//! embeddings plus their exact valid length. Keeping this schema separate stops
//! an audio-capable artifact from silently loading a decoder-only or partial
//! checkpoint.

use std::collections::{HashMap, HashSet};

use serde_json::Value;

use super::builder::{Builder, Precision, Ty, Val};

const RAW_PREFIX: &str = "model.embed_tokens_extend.audio_embed";

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Phi4AudioConfig {
    pub input_size: usize,
    pub attention_dim: usize,
    pub attention_heads: usize,
    pub num_blocks: usize,
    pub linear_units: usize,
    pub time_reduction: usize,
    pub conv_channels: usize,
    pub kernel_size: usize,
    pub relative_bias_max_distance: usize,
    pub projection_hidden: usize,
}

impl Phi4AudioConfig {
    pub(crate) fn from_json_str(text: &str) -> Result<Self, String> {
        let root: Value =
            serde_json::from_str(text).map_err(|error| format!("parse config.json: {error}"))?;
        if root.get("model_type").and_then(Value::as_str) != Some("phi4mm") {
            return Err("Phi4MM audio requires config.json model_type = \"phi4mm\"".to_string());
        }
        let config = root
            .pointer("/audio_processor/config")
            .ok_or("Phi4MM config is missing audio_processor.config")?;
        let integer = |name: &str| {
            config
                .get(name)
                .and_then(Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or_else(|| format!("Phi4MM audio config requires positive integer `{name}`"))
        };
        let nested_integer = |path: &str, label: &str| {
            config
                .pointer(path)
                .and_then(Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or_else(|| format!("Phi4MM audio config requires positive integer `{label}`"))
        };
        let parsed = Self {
            input_size: integer("input_size")?,
            attention_dim: integer("attention_dim")?,
            attention_heads: integer("attention_heads")?,
            num_blocks: integer("num_blocks")?,
            linear_units: integer("linear_units")?,
            time_reduction: integer("time_reduction")?,
            conv_channels: nested_integer(
                "/nemo_conv_settings/conv_channels",
                "nemo_conv_settings.conv_channels",
            )?,
            kernel_size: integer("kernel_size")?,
            relative_bias_max_distance: nested_integer(
                "/relative_attention_bias_args/t5_bias_max_distance",
                "relative_attention_bias_args.t5_bias_max_distance",
            )?,
            projection_hidden: root
                .get("hidden_size")
                .and_then(Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or("Phi4MM config requires positive integer hidden_size")?,
        };
        parsed.validate_published(config, &root)?;
        Ok(parsed)
    }

    fn validate_published(&self, config: &Value, root: &Value) -> Result<(), String> {
        let string_is =
            |name: &str, expected: &str| config.get(name).and_then(Value::as_str) == Some(expected);
        let bool_is = |name: &str, expected: bool| {
            config.get(name).and_then(Value::as_bool) == Some(expected)
        };
        let published = self.input_size == 80
            && self.attention_dim == 1024
            && self.attention_heads == 16
            && self.num_blocks == 24
            && self.linear_units == 1536
            && self.time_reduction == 8
            && self.conv_channels == 1024
            && self.kernel_size == 3
            && self.relative_bias_max_distance == 500
            && self.projection_hidden == 3072
            && string_is("input_layer", "nemo_conv")
            && string_is("activation", "swish")
            && string_is("conv_activation", "swish")
            && string_is("conv_glu_type", "swish")
            && bool_is("causal", true)
            && bool_is("batch_norm", false)
            && bool_is("bias_in_glu", true)
            && config.get("depthwise_multiplier").and_then(Value::as_u64) == Some(1)
            && config
                .get("depthwise_seperable_out_channel")
                .and_then(Value::as_u64)
                == Some(1024)
            && config
                .pointer("/encoder_embedding_config/input_size")
                .and_then(Value::as_u64)
                == Some(80)
            && config
                .pointer("/relative_attention_bias_args/type")
                .and_then(Value::as_str)
                == Some("t5")
            && !config
                .pointer("/relative_attention_bias_args/t5_bias_symmetric")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            && !config
                .pointer("/nemo_conv_settings/is_causal")
                .and_then(Value::as_bool)
                .unwrap_or(false);
        let projection = root.pointer("/embd_layer/audio_embd_layer");
        let published_projection = projection.is_some_and(|value| {
            value.get("compression_rate").and_then(Value::as_u64) == Some(8)
                && value.get("downsample_rate").and_then(Value::as_u64) == Some(1)
                && value.get("projection_cls").and_then(Value::as_str) == Some("mlp")
                && value.get("use_conv_downsample").and_then(Value::as_bool) == Some(false)
                && value.get("use_qformer").and_then(Value::as_bool) == Some(false)
        });
        if !published || !published_projection {
            return Err(
                "unsupported Phi4MM audio architecture: expected the pinned published Cascades/MLP contract"
                    .to_string(),
            );
        }
        Ok(())
    }

    pub(crate) fn encoded_bucket_len(&self, frame_bucket: usize) -> Result<usize, String> {
        if frame_bucket == 0 {
            return Err("Phi4MM audio frame bucket must be positive".to_string());
        }
        if self.time_reduction != 8 {
            return Err(format!(
                "Phi4MM audio graph implements the pinned 3x stride-2 reduction, not time_reduction={}",
                self.time_reduction
            ));
        }
        // The pinned NeMo subsampler is three symmetric-pad Conv2D stages
        // (kernel=3, pad=1, stride=2). Derive the emitted row count from that
        // exact operator chain instead of assuming the configured aggregate
        // reduction remains equivalent for every boundary length.
        Ok((0..3).fold(frame_bucket, |length, _| (length - 1) / 2 + 1))
    }

    pub(crate) fn encoded_valid_len(&self, frame_len: usize) -> Result<usize, String> {
        self.encoded_bucket_len(frame_len)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Phi4AudioWeightSpec {
    pub name: String,
    pub shape: Vec<usize>,
}

fn push(specs: &mut Vec<Phi4AudioWeightSpec>, name: String, shape: &[usize]) {
    specs.push(Phi4AudioWeightSpec {
        name,
        shape: shape.to_vec(),
    });
}

fn push_linear(specs: &mut Vec<Phi4AudioWeightSpec>, stem: &str, output: usize, input: usize) {
    push(specs, format!("{stem}.weight"), &[output, input]);
    push(specs, format!("{stem}.bias"), &[output]);
}

fn push_layer_norm(specs: &mut Vec<Phi4AudioWeightSpec>, stem: &str, dim: usize) {
    push(specs, format!("{stem}.weight"), &[dim]);
    push(specs, format!("{stem}.bias"), &[dim]);
}

fn push_ffn(specs: &mut Vec<Phi4AudioWeightSpec>, stem: &str, dim: usize, inner: usize) {
    push_layer_norm(specs, &format!("{stem}.layer_norm"), dim);
    push_linear(specs, &format!("{stem}.net.0.linear"), inner * 2, dim);
    push_linear(specs, &format!("{stem}.net.2"), dim, inner);
}

pub(crate) fn phi4_audio_weight_specs(config: &Phi4AudioConfig) -> Vec<Phi4AudioWeightSpec> {
    let dim = config.attention_dim;
    let channels = config.conv_channels;
    let mut specs = Vec::with_capacity(887);
    let encoder = format!("{RAW_PREFIX}.encoder");

    push(
        &mut specs,
        format!("{encoder}.encoder_embedding.global_mean"),
        &[config.input_size],
    );
    push(
        &mut specs,
        format!("{encoder}.encoder_embedding.global_invstd"),
        &[config.input_size],
    );
    let embed = format!("{encoder}.embed");
    push(
        &mut specs,
        format!("{embed}.conv.0.weight"),
        &[channels, 1, 3, 3],
    );
    push(&mut specs, format!("{embed}.conv.0.bias"), &[channels]);
    for (depthwise, pointwise) in [(2, 3), (5, 6)] {
        push(
            &mut specs,
            format!("{embed}.conv.{depthwise}.weight"),
            &[channels, 1, 3, 3],
        );
        push(
            &mut specs,
            format!("{embed}.conv.{depthwise}.bias"),
            &[channels],
        );
        push(
            &mut specs,
            format!("{embed}.conv.{pointwise}.weight"),
            &[channels, channels, 1, 1],
        );
        push(
            &mut specs,
            format!("{embed}.conv.{pointwise}.bias"),
            &[channels],
        );
    }
    let reduced_frequency = config.input_size.div_ceil(8);
    push_linear(
        &mut specs,
        &format!("{embed}.out"),
        dim,
        channels * reduced_frequency,
    );

    for layer in 0..config.num_blocks {
        let block = format!("{encoder}.encoders.{layer}");
        push_ffn(
            &mut specs,
            &format!("{block}.feed_forward_in"),
            dim,
            config.linear_units,
        );
        push_layer_norm(&mut specs, &format!("{block}.layer_norm_att"), dim);
        for projection in ["linear_q", "linear_k", "linear_v", "linear_out"] {
            push_linear(
                &mut specs,
                &format!("{block}.self_attn.{projection}"),
                dim,
                dim,
            );
        }

        let conv = format!("{block}.conv");
        push_layer_norm(&mut specs, &format!("{conv}.layer_norm"), dim);
        push(&mut specs, format!("{conv}.glu.b1"), &[1, dim, 1]);
        push(&mut specs, format!("{conv}.glu.b2"), &[1, dim, 1]);
        push(
            &mut specs,
            format!("{conv}.glu.ext_pw_conv_1d.weight"),
            &[dim * 2, dim, 1],
        );
        push(
            &mut specs,
            format!("{conv}.glu.ext_pw_conv_1d.bias"),
            &[dim * 2],
        );
        push(
            &mut specs,
            format!("{conv}.dw_sep_conv_1d.dw_conv.weight"),
            &[dim, 1, config.kernel_size],
        );
        push(
            &mut specs,
            format!("{conv}.dw_sep_conv_1d.dw_conv.bias"),
            &[dim],
        );
        for projection in ["dw_sep_conv_1d.pw_conv", "ext_pw_conv_1d"] {
            push(
                &mut specs,
                format!("{conv}.{projection}.weight"),
                &[dim, dim, 1],
            );
            push(&mut specs, format!("{conv}.{projection}.bias"), &[dim]);
        }
        push_ffn(
            &mut specs,
            &format!("{block}.feed_forward_out"),
            dim,
            config.linear_units,
        );
        push_layer_norm(&mut specs, &format!("{block}.layer_norm"), dim);
    }
    push(
        &mut specs,
        format!("{encoder}.relative_attention_bias_layer.bias_values.weight"),
        &[
            config.relative_bias_max_distance * 2,
            config.attention_heads,
        ],
    );

    for mode in ["speech", "vision"] {
        let projection = format!("{RAW_PREFIX}.audio_projection.{mode}");
        push_linear(
            &mut specs,
            &format!("{projection}.0"),
            config.projection_hidden,
            dim,
        );
        push_linear(
            &mut specs,
            &format!("{projection}.2"),
            config.projection_hidden,
            config.projection_hidden,
        );
    }
    specs
}

pub(crate) fn validate_phi4_audio_weight_shapes<F>(
    config: &Phi4AudioConfig,
    mut actual_shape: F,
) -> Result<(), String>
where
    F: FnMut(&str) -> Option<Vec<usize>>,
{
    let specs = phi4_audio_weight_specs(config);
    let unique = specs
        .iter()
        .map(|spec| spec.name.as_str())
        .collect::<HashSet<_>>();
    if unique.len() != specs.len() {
        return Err("Phi4MM audio weight schema contains duplicate tensor names".to_string());
    }
    for spec in specs {
        let actual = actual_shape(&spec.name)
            .ok_or_else(|| format!("Phi4MM audio checkpoint is missing `{}`", spec.name))?;
        if actual != spec.shape {
            return Err(format!(
                "Phi4MM audio weight `{}` has shape {actual:?}, expected {:?}",
                spec.name, spec.shape
            ));
        }
    }
    Ok(())
}

fn lookup<'a>(weights: &'a HashMap<String, Val>, name: &str) -> &'a Val {
    weights
        .get(name)
        .unwrap_or_else(|| panic!("Phi4MM audio emitter schema is missing {name}"))
}

fn add_last_bias(b: &mut Builder, value: &Val, bias: &Val) -> Val {
    let axis = value.ty.shape.len() - 1;
    let broadcast = b.broadcast(bias, &[axis], value.ty.shape.clone());
    b.add(value, &broadcast)
}

fn linear_bias(b: &mut Builder, weights: &HashMap<String, Val>, input: &Val, stem: &str) -> Val {
    let projected = b.linear_seq(input, lookup(weights, &format!("{stem}.weight")));
    add_last_bias(b, &projected, lookup(weights, &format!("{stem}.bias")))
}

fn layer_norm(b: &mut Builder, weights: &HashMap<String, Val>, input: &Val, stem: &str) -> Val {
    let axis = input.ty.shape.len() - 1;
    let dim = input.ty.shape[axis];
    let zero = b.const_f32(0.0);
    let sum = b.reduce_add(input, axis, &zero);
    let count = b.const_f32(dim as f32);
    let count = b.broadcast(&count, &[], sum.ty.shape.clone());
    let mean = b.divide(&sum, &count);
    let leading = (0..axis).collect::<Vec<_>>();
    let mean = b.broadcast(&mean, &leading, input.ty.shape.clone());
    let centered = b.subtract(input, &mean);
    let squared = b.multiply(&centered, &centered);
    let variance = b.reduce_add(&squared, axis, &zero);
    let variance = b.divide(&variance, &count);
    let epsilon = b.const_f32(1e-5);
    let epsilon = b.broadcast(&epsilon, &[], variance.ty.shape.clone());
    let variance = b.add(&variance, &epsilon);
    let inverse_std = b.rsqrt(&variance);
    let inverse_std = b.broadcast(&inverse_std, &leading, input.ty.shape.clone());
    let normalized = b.multiply(&centered, &inverse_std);
    let weight = b.broadcast(
        lookup(weights, &format!("{stem}.weight")),
        &[axis],
        input.ty.shape.clone(),
    );
    let normalized = b.multiply(&normalized, &weight);
    let bias = b.broadcast(
        lookup(weights, &format!("{stem}.bias")),
        &[axis],
        input.ty.shape.clone(),
    );
    b.add(&normalized, &bias)
}

fn silu(b: &mut Builder, input: &Val) -> Val {
    let one = b.const_f32(1.0);
    let one = b.broadcast(&one, &[], input.ty.shape.clone());
    let negative = b.negate(input);
    let exponential = b.exponential(&negative);
    let denominator = b.add(&one, &exponential);
    b.divide(input, &denominator)
}

fn gelu(b: &mut Builder, input: &Val) -> Val {
    let shape = input.ty.shape.clone();
    let inverse_sqrt_two = b.const_f32(std::f32::consts::FRAC_1_SQRT_2);
    let inverse_sqrt_two = b.broadcast(&inverse_sqrt_two, &[], shape.clone());
    let scaled = b.multiply(input, &inverse_sqrt_two);
    let erf = b.erf(&scaled);
    let one = b.const_f32(1.0);
    let one = b.broadcast(&one, &[], shape.clone());
    let half = b.const_f32(0.5);
    let half = b.broadcast(&half, &[], shape);
    let cdf = b.add(&one, &erf);
    let cdf = b.multiply(&half, &cdf);
    b.multiply(input, &cdf)
}

fn softmax_last(b: &mut Builder, input: &Val) -> Val {
    let axis = input.ty.shape.len() - 1;
    let negative_infinity = b.const_f32(f32::NEG_INFINITY);
    let maximum = b.reduce_max(input, axis, &negative_infinity);
    let leading = (0..axis).collect::<Vec<_>>();
    let maximum = b.broadcast(&maximum, &leading, input.ty.shape.clone());
    let shifted = b.subtract(input, &maximum);
    let exponent = b.exponential(&shifted);
    let zero = b.const_f32(0.0);
    let sum = b.reduce_add(&exponent, axis, &zero);
    let sum = b.broadcast(&sum, &leading, input.ty.shape.clone());
    b.divide(&exponent, &sum)
}

fn mask_rows(b: &mut Builder, input: &Val, active: &Val) -> Val {
    let active = b.broadcast(active, &[0], input.ty.shape.clone());
    let zero = b.const_f32(0.0);
    let zero = b.broadcast(&zero, &[], input.ty.shape.clone());
    b.select(&active, input, &zero)
}

fn round_bf16(b: &mut Builder, input: &Val) -> Val {
    let narrow = b.convert(input, "bf16");
    b.convert(&narrow, "f32")
}

fn conv2d(
    b: &mut Builder,
    weights: &HashMap<String, Val>,
    input: &Val,
    stem: &str,
    stride: usize,
    padding: usize,
    groups: usize,
) -> Val {
    let kernel = b.transpose(lookup(weights, &format!("{stem}.weight")), &[2, 3, 1, 0]);
    let convolved = b.convolution(
        input,
        &kernel,
        &[stride, stride],
        &[(padding, padding), (padding, padding)],
        groups,
    );
    add_last_bias(b, &convolved, lookup(weights, &format!("{stem}.bias")))
}

fn conv1d(
    b: &mut Builder,
    weights: &HashMap<String, Val>,
    input: &Val,
    stem: &str,
    causal_padding: usize,
    groups: usize,
) -> Val {
    let zero = b.const_f32(0.0);
    let input = if causal_padding == 0 {
        input.clone()
    } else {
        b.pad(input, &zero, &[0, causal_padding, 0], &[0, 0, 0])
    };
    let kernel = b.transpose(lookup(weights, &format!("{stem}.weight")), &[2, 1, 0]);
    let convolved = b.convolution(&input, &kernel, &[1], &[(0, 0)], groups);
    add_last_bias(b, &convolved, lookup(weights, &format!("{stem}.bias")))
}

fn feed_forward(
    b: &mut Builder,
    weights: &HashMap<String, Val>,
    input: &Val,
    stem: &str,
    inner: usize,
) -> Val {
    let normalized = layer_norm(b, weights, input, &format!("{stem}.layer_norm"));
    let gated = linear_bias(b, weights, &normalized, &format!("{stem}.net.0.linear"));
    let rows = gated.ty.shape[0];
    let left = b.slice(&gated, &[(0, rows), (0, inner)]);
    let gate = b.slice(&gated, &[(0, rows), (inner, inner * 2)]);
    let activated = silu(b, &gate);
    let combined = b.multiply(&left, &activated);
    linear_bias(b, weights, &combined, &format!("{stem}.net.2"))
}

fn relative_attention_bias(
    b: &mut Builder,
    weights: &HashMap<String, Val>,
    config: &Phi4AudioConfig,
    length: usize,
) -> Val {
    let distance = config.relative_bias_max_distance as isize;
    let mut indices = Vec::with_capacity(length * length);
    for query in 0..length {
        for key in 0..length {
            indices.push(
                ((key as isize - query as isize).clamp(-distance, distance - 1) + distance) as i32,
            );
        }
    }
    let indices = b.const_tensor_i32(&indices, vec![length, length, 1]);
    let gathered = b.gather_rows_nd(
        lookup(
            weights,
            &format!("{RAW_PREFIX}.encoder.relative_attention_bias_layer.bias_values.weight"),
        ),
        &indices,
    );
    b.transpose(&gathered, &[2, 0, 1])
}

fn self_attention(
    b: &mut Builder,
    weights: &HashMap<String, Val>,
    config: &Phi4AudioConfig,
    input: &Val,
    relative_bias: &Val,
    active_rows: &Val,
    stem: &str,
) -> Val {
    let length = input.ty.shape[0];
    let heads = config.attention_heads;
    let head_dim = config.attention_dim / heads;
    let project = |b: &mut Builder, suffix: &str| {
        let value = linear_bias(b, weights, input, &format!("{stem}.self_attn.{suffix}"));
        let value = b.reshape(&value, vec![length, heads, head_dim]);
        b.transpose(&value, &[1, 0, 2])
    };
    let query = project(b, "linear_q");
    let scale = b.const_f32((head_dim as f32).powf(-0.5));
    let scale = b.broadcast(&scale, &[], query.ty.shape.clone());
    let query = b.multiply(&query, &scale);
    let key = project(b, "linear_k");
    let value = project(b, "linear_v");
    let key = b.transpose(&key, &[0, 2, 1]);
    let scores = b.dot_general(
        &query,
        &key,
        &[0],
        &[0],
        &[2],
        &[1],
        vec![heads, length, length],
    );
    let scores = b.add(&scores, relative_bias);
    let active_keys = b.broadcast(active_rows, &[2], vec![heads, length, length]);
    let zero = b.const_f32(0.0);
    let zero = b.broadcast(&zero, &[], vec![heads, length, length]);
    let negative = b.const_f32(-1e30);
    let negative = b.broadcast(&negative, &[], vec![heads, length, length]);
    let key_mask = b.select(&active_keys, &zero, &negative);
    let scores = b.add(&scores, &key_mask);
    let probabilities = softmax_last(b, &scores);
    let context = b.dot_general(
        &probabilities,
        &value,
        &[0],
        &[0],
        &[2],
        &[1],
        vec![heads, length, head_dim],
    );
    let context = b.transpose(&context, &[1, 0, 2]);
    let context = b.reshape(&context, vec![length, config.attention_dim]);
    linear_bias(
        b,
        weights,
        &context,
        &format!("{stem}.self_attn.linear_out"),
    )
}

fn convolution_module(
    b: &mut Builder,
    weights: &HashMap<String, Val>,
    config: &Phi4AudioConfig,
    input: &Val,
    stem: &str,
) -> Val {
    let length = input.ty.shape[0];
    let dim = config.attention_dim;
    let normalized = layer_norm(b, weights, input, &format!("{stem}.conv.layer_norm"));
    let normalized = b.reshape(&normalized, vec![1, length, dim]);
    let glu_stem = format!("{stem}.conv.glu.ext_pw_conv_1d");
    let gated = conv1d(b, weights, &normalized, &glu_stem, 0, 1);
    let left = b.slice(&gated, &[(0, 1), (0, length), (0, dim)]);
    let gate = b.slice(&gated, &[(0, 1), (0, length), (dim, dim * 2)]);
    let bias = |b: &mut Builder, suffix: &str| {
        let value = b.transpose(
            lookup(weights, &format!("{stem}.conv.glu.{suffix}")),
            &[0, 2, 1],
        );
        b.broadcast(&value, &[0, 1, 2], vec![1, length, dim])
    };
    let left_bias = bias(b, "b1");
    let gate_bias = bias(b, "b2");
    let left = b.add(&left, &left_bias);
    let gate = b.add(&gate, &gate_bias);
    let activated_gate = silu(b, &gate);
    let gated = b.multiply(&left, &activated_gate);
    let depthwise = conv1d(
        b,
        weights,
        &gated,
        &format!("{stem}.conv.dw_sep_conv_1d.dw_conv"),
        config.kernel_size - 1,
        dim,
    );
    let pointwise = conv1d(
        b,
        weights,
        &depthwise,
        &format!("{stem}.conv.dw_sep_conv_1d.pw_conv"),
        0,
        1,
    );
    let activated = silu(b, &pointwise);
    let output = conv1d(
        b,
        weights,
        &activated,
        &format!("{stem}.conv.ext_pw_conv_1d"),
        0,
        1,
    );
    b.reshape(&output, vec![length, dim])
}

fn conformer_block(
    b: &mut Builder,
    weights: &HashMap<String, Val>,
    config: &Phi4AudioConfig,
    input: &Val,
    relative_bias: &Val,
    active_rows: &Val,
    layer: usize,
) -> ConformerBlockValues {
    let stem = format!("{RAW_PREFIX}.encoder.encoders.{layer}");
    let scale = b.const_f32(0.5);
    let scale = b.broadcast(&scale, &[], input.ty.shape.clone());
    let ff = feed_forward(
        b,
        weights,
        input,
        &format!("{stem}.feed_forward_in"),
        config.linear_units,
    );
    let ff = b.multiply(&ff, &scale);
    let after_ff_in = b.add(input, &ff);
    let attention_input = layer_norm(b, weights, &after_ff_in, &format!("{stem}.layer_norm_att"));
    let attention = self_attention(
        b,
        weights,
        config,
        &attention_input,
        relative_bias,
        active_rows,
        &stem,
    );
    let after_attention = b.add(&after_ff_in, &attention);
    let convolution = convolution_module(b, weights, config, &after_attention, &stem);
    let after_convolution = b.add(&after_attention, &convolution);
    let ff = feed_forward(
        b,
        weights,
        &after_convolution,
        &format!("{stem}.feed_forward_out"),
        config.linear_units,
    );
    let ff = b.multiply(&ff, &scale);
    let hidden = b.add(&after_convolution, &ff);
    let output = layer_norm(b, weights, &hidden, &format!("{stem}.layer_norm"));
    let output = mask_rows(b, &output, active_rows);
    ConformerBlockValues {
        after_ff_in,
        attention,
        after_attention,
        convolution,
        after_convolution,
        ff_out: ff,
        output,
    }
}

struct ConformerBlockValues {
    after_ff_in: Val,
    attention: Val,
    after_attention: Val,
    convolution: Val,
    after_convolution: Val,
    ff_out: Val,
    output: Val,
}

struct SubsampleValues {
    conv0: Val,
    conv1_depthwise: Val,
    conv1_pointwise: Val,
    conv1: Val,
    conv2_depthwise: Val,
    conv2_pointwise: Val,
    conv2: Val,
    projected: Val,
}

fn subsample(
    b: &mut Builder,
    weights: &HashMap<String, Val>,
    config: &Phi4AudioConfig,
    input: &Val,
) -> SubsampleValues {
    let bucket = input.ty.shape[0];
    let mut hidden = b.reshape(input, vec![1, bucket, config.input_size, 1]);
    let embed = format!("{RAW_PREFIX}.encoder.embed");
    hidden = conv2d(b, weights, &hidden, &format!("{embed}.conv.0"), 2, 1, 1);
    // MLX evaluates the first convolution with BF16 features/weights/bias and
    // materializes its BF16 result. `relu()`'s F32 zero then promotes the
    // remaining subsampler and all later captured stages to F32.
    hidden = round_bf16(b, &hidden);
    let relu = |b: &mut Builder, value: &Val| {
        let zero = b.const_f32(0.0);
        let zero = b.broadcast(&zero, &[], value.ty.shape.clone());
        b.maximum(value, &zero)
    };
    let conv0 = relu(b, &hidden);
    let conv1_depthwise = conv2d(
        b,
        weights,
        &conv0,
        &format!("{embed}.conv.2"),
        2,
        1,
        config.conv_channels,
    );
    let conv1_pointwise = conv2d(
        b,
        weights,
        &conv1_depthwise,
        &format!("{embed}.conv.3"),
        1,
        0,
        1,
    );
    let conv1 = relu(b, &conv1_pointwise);
    let conv2_depthwise = conv2d(
        b,
        weights,
        &conv1,
        &format!("{embed}.conv.5"),
        2,
        1,
        config.conv_channels,
    );
    let conv2_pointwise = conv2d(
        b,
        weights,
        &conv2_depthwise,
        &format!("{embed}.conv.6"),
        1,
        0,
        1,
    );
    let conv2 = relu(b, &conv2_pointwise);
    let hidden = b.transpose(&conv2, &[0, 1, 3, 2]);
    let shape = hidden.ty.shape.clone();
    let hidden = b.reshape(&hidden, vec![shape[1], shape[2] * shape[3]]);
    let projected = linear_bias(b, weights, &hidden, &format!("{embed}.out"));
    SubsampleValues {
        conv0,
        conv1_depthwise,
        conv1_pointwise,
        conv1,
        conv2_depthwise,
        conv2_pointwise,
        conv2,
        projected,
    }
}

struct ProjectionValues {
    speech_first: Val,
    speech: Val,
    vision_first: Val,
    vision: Val,
    selected: Val,
}

fn project_audio(
    b: &mut Builder,
    weights: &HashMap<String, Val>,
    config: &Phi4AudioConfig,
    encoded: &Val,
    projection_mode: &Val,
    active_rows: &Val,
) -> ProjectionValues {
    let branch = |b: &mut Builder, mode: &str| {
        let first = linear_bias(
            b,
            weights,
            encoded,
            &format!("{RAW_PREFIX}.audio_projection.{mode}.0"),
        );
        let activated = gelu(b, &first);
        let projected = linear_bias(
            b,
            weights,
            &activated,
            &format!("{RAW_PREFIX}.audio_projection.{mode}.2"),
        );
        (first, projected)
    };
    let (speech_first, speech) = branch(b, "speech");
    let (vision_first, vision) = branch(b, "vision");
    let shape = vec![encoded.ty.shape[0], config.projection_hidden];
    let zero = b.const_f32(0.0);
    let zero = b.broadcast(&zero, &[], shape.clone());
    let code = |b: &mut Builder, value: i32| {
        let value = b.const_i32(value);
        b.compare("EQ", projection_mode, &value, "SIGNED")
    };
    let speech_active = code(b, 1);
    let vision_active = code(b, 2);
    let speech_active = b.broadcast(&speech_active, &[], shape.clone());
    let vision_active = b.broadcast(&vision_active, &[], shape);
    let selected = b.select(&speech_active, &speech, &zero);
    let selected = b.select(&vision_active, &vision, &selected);
    let selected = mask_rows(b, &selected, active_rows);
    ProjectionValues {
        speech_first,
        speech,
        vision_first,
        vision,
        selected,
    }
}

/// Emit `audio.main` for a static frame bucket.
///
/// Inputs after the 887 resident weights are the exact #874 host-oracle shapes
/// `features[1,frame_bucket,80]`, `frame_mask[1,frame_bucket]`, the unpadded
/// frame length, and projection mode (1=speech, 2=vision/mixed). Outputs are
/// projected decoder embeddings `[1,subsampled_bucket,3072]` and the exact
/// valid subsampled length.
pub(crate) fn emit_phi4_audio_with(
    config: &Phi4AudioConfig,
    frame_bucket: usize,
    precision: Precision,
) -> Result<String, String> {
    emit_phi4_audio_entry(config, frame_bucket, precision, false)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Phi4AudioDiagnosticSpec {
    pub name: &'static str,
    pub shape: Vec<usize>,
}

pub(crate) fn phi4_audio_diagnostic_specs(
    config: &Phi4AudioConfig,
    frame_bucket: usize,
) -> Result<Vec<Phi4AudioDiagnosticSpec>, String> {
    let reduce = |length: usize| (length - 1) / 2 + 1;
    let time0 = reduce(frame_bucket);
    let time1 = reduce(time0);
    let time2 = reduce(time1);
    let frequency0 = reduce(config.input_size);
    let frequency1 = reduce(frequency0);
    let frequency2 = reduce(frequency1);
    let encoder = vec![1, time2, config.attention_dim];
    let projection = vec![1, time2, config.projection_hidden];
    let mut specs = vec![
        Phi4AudioDiagnosticSpec {
            name: "subsample.conv0",
            shape: vec![1, time0, frequency0, config.conv_channels],
        },
        Phi4AudioDiagnosticSpec {
            name: "subsample.conv1.depthwise",
            shape: vec![1, time1, frequency1, config.conv_channels],
        },
        Phi4AudioDiagnosticSpec {
            name: "subsample.conv1.pointwise",
            shape: vec![1, time1, frequency1, config.conv_channels],
        },
        Phi4AudioDiagnosticSpec {
            name: "subsample.conv1",
            shape: vec![1, time1, frequency1, config.conv_channels],
        },
        Phi4AudioDiagnosticSpec {
            name: "subsample.conv2.depthwise",
            shape: vec![1, time2, frequency2, config.conv_channels],
        },
        Phi4AudioDiagnosticSpec {
            name: "subsample.conv2.pointwise",
            shape: vec![1, time2, frequency2, config.conv_channels],
        },
        Phi4AudioDiagnosticSpec {
            name: "subsample.conv2",
            shape: vec![1, time2, frequency2, config.conv_channels],
        },
        Phi4AudioDiagnosticSpec {
            name: "subsample.projected",
            shape: encoder.clone(),
        },
    ];
    for name in [
        "block0.after_ff_in",
        "block0.attention",
        "block0.after_attention",
        "block0.convolution",
        "block0.after_convolution",
        "block0.ff_out",
        "block0.output",
        "block1.output",
        "block5.output",
        "block11.output",
        "block17.output",
        "block23.output",
        "encoder.output",
    ] {
        specs.push(Phi4AudioDiagnosticSpec {
            name,
            shape: encoder.clone(),
        });
    }
    for name in [
        "projection.speech.first",
        "projection.speech.output",
        "projection.vision.first",
        "projection.vision.output",
    ] {
        specs.push(Phi4AudioDiagnosticSpec {
            name,
            shape: projection.clone(),
        });
    }
    if time2 != config.encoded_bucket_len(frame_bucket)? {
        return Err("Phi4MM diagnostic subsampling shape drifted from audio contract".to_string());
    }
    Ok(specs)
}

pub(crate) fn emit_phi4_audio_diagnostic_with(
    config: &Phi4AudioConfig,
    frame_bucket: usize,
    precision: Precision,
) -> Result<String, String> {
    emit_phi4_audio_entry(config, frame_bucket, precision, true)
}

fn emit_phi4_audio_entry(
    config: &Phi4AudioConfig,
    frame_bucket: usize,
    precision: Precision,
    diagnostic: bool,
) -> Result<String, String> {
    if config.attention_dim % config.attention_heads != 0 {
        return Err("Phi4MM audio attention_dim must be divisible by heads".to_string());
    }
    let encoded_len = config.encoded_bucket_len(frame_bucket)?;
    if encoded_len > config.relative_bias_max_distance {
        return Err(format!(
            "Phi4MM audio bucket encodes to {encoded_len} rows, exceeding the pinned 500-row Conformer chunk"
        ));
    }
    let specs = phi4_audio_weight_specs(config);
    let mut declarations = Vec::with_capacity(specs.len() + 4);
    let mut weights = HashMap::with_capacity(specs.len());
    for (index, spec) in specs.iter().enumerate() {
        let ty = Ty::f32(spec.shape.clone());
        declarations.push((ty.clone(), spec.name.clone()));
        weights.insert(spec.name.clone(), Builder::arg(index, ty));
    }
    let features_index = declarations.len();
    declarations.push((
        Ty::f32(vec![1, frame_bucket, config.input_size]),
        "features".to_string(),
    ));
    let mask_index = declarations.len();
    declarations.push((
        Ty::new(vec![1, frame_bucket], "i32"),
        "frame_mask".to_string(),
    ));
    let length_index = declarations.len();
    declarations.push((Ty::scalar("i32"), "frame_length".to_string()));
    let mode_index = declarations.len();
    declarations.push((Ty::scalar("i32"), "projection_mode".to_string()));

    let mut b = Builder::new().with_precision(precision);
    let features = Builder::arg(
        features_index,
        Ty::f32(vec![1, frame_bucket, config.input_size]),
    );
    let frame_mask = Builder::arg(mask_index, Ty::new(vec![1, frame_bucket], "i32"));
    let frame_length = Builder::arg(length_index, Ty::scalar("i32"));
    let projection_mode = Builder::arg(mode_index, Ty::scalar("i32"));

    let features = b.reshape(&features, vec![frame_bucket, config.input_size]);
    let frame_mask = b.reshape(&frame_mask, vec![frame_bucket]);
    let zero_i32 = b.const_i32(0);
    let zero_i32 = b.broadcast(&zero_i32, &[], vec![frame_bucket]);
    let frame_active = b.compare("GT", &frame_mask, &zero_i32, "SIGNED");
    let mean = b.broadcast(
        lookup(
            &weights,
            &format!("{RAW_PREFIX}.encoder.encoder_embedding.global_mean"),
        ),
        &[1],
        vec![frame_bucket, config.input_size],
    );
    let inverse_std = b.broadcast(
        lookup(
            &weights,
            &format!("{RAW_PREFIX}.encoder.encoder_embedding.global_invstd"),
        ),
        &[1],
        vec![frame_bucket, config.input_size],
    );
    // The MLX oracle casts features to the BF16 mean tensor. Its subtract and
    // multiply therefore each materialize a BF16 boundary; the first ReLU then
    // promotes the Conv2D stream and every captured Conformer stage to F32.
    // Mirror those actual boundaries instead of demoting every contraction.
    let centered = b.subtract(&features, &mean);
    let centered = round_bf16(&mut b, &centered);
    let normalized = b.multiply(&centered, &inverse_std);
    let normalized = round_bf16(&mut b, &normalized);
    // Host bucket padding is zero in transport space. Re-apply the frame mask
    // after normalization so padded rows become the encoder's mathematical
    // zero padding rather than `(0 - global_mean) * global_invstd`.
    let normalized = mask_rows(&mut b, &normalized, &frame_active);
    let subsampled = subsample(&mut b, &weights, config, &normalized);

    let reduction_minus_one = b.const_i32((config.time_reduction - 1) as i32);
    let reduction = b.const_i32(config.time_reduction as i32);
    let valid_length = b.add(&frame_length, &reduction_minus_one);
    let valid_length = b.divide(&valid_length, &reduction);
    let positions = b.iota(encoded_len);
    let valid_length_vector = b.broadcast(&valid_length, &[], vec![encoded_len]);
    let active_rows = b.compare("LT", &positions, &valid_length_vector, "SIGNED");
    let mut encoded = mask_rows(&mut b, &subsampled.projected, &active_rows);
    let relative_bias = relative_attention_bias(&mut b, &weights, config, encoded_len);
    let mut block0 = None;
    let mut selected_blocks = Vec::new();
    for layer in 0..config.num_blocks {
        let values = conformer_block(
            &mut b,
            &weights,
            config,
            &encoded,
            &relative_bias,
            &active_rows,
            layer,
        );
        encoded = values.output.clone();
        if layer == 0 {
            block0 = Some(values);
        } else if matches!(layer, 1 | 5 | 11 | 17 | 23) {
            selected_blocks.push((layer, encoded.clone()));
        }
    }
    let projection = project_audio(
        &mut b,
        &weights,
        config,
        &encoded,
        &projection_mode,
        &active_rows,
    );
    let signature = declarations
        .iter()
        .enumerate()
        .map(|(index, (ty, location))| {
            format!("%arg{index}: {} loc(\"{}\")", ty.render(), location)
        })
        .collect::<Vec<_>>()
        .join(", ");
    if diagnostic {
        let block0 = block0.ok_or("Phi4MM diagnostic requires Conformer block 0")?;
        let selected = |layer: usize| {
            selected_blocks
                .iter()
                .find_map(|(candidate, value)| (*candidate == layer).then_some(value.clone()))
                .ok_or_else(|| format!("Phi4MM diagnostic requires Conformer block {layer}"))
        };
        let reshape_encoder = |b: &mut Builder, value: &Val| {
            b.reshape(value, vec![1, encoded_len, config.attention_dim])
        };
        let reshape_projection = |b: &mut Builder, value: &Val| {
            b.reshape(value, vec![1, encoded_len, config.projection_hidden])
        };
        let mut checkpoints = vec![
            subsampled.conv0,
            subsampled.conv1_depthwise,
            subsampled.conv1_pointwise,
            subsampled.conv1,
            subsampled.conv2_depthwise,
            subsampled.conv2_pointwise,
            subsampled.conv2,
            reshape_encoder(&mut b, &subsampled.projected),
            reshape_encoder(&mut b, &block0.after_ff_in),
            reshape_encoder(&mut b, &block0.attention),
            reshape_encoder(&mut b, &block0.after_attention),
            reshape_encoder(&mut b, &block0.convolution),
            reshape_encoder(&mut b, &block0.after_convolution),
            reshape_encoder(&mut b, &block0.ff_out),
            reshape_encoder(&mut b, &block0.output),
        ];
        for layer in [1, 5, 11, 17, 23] {
            checkpoints.push(reshape_encoder(&mut b, &selected(layer)?));
        }
        checkpoints.extend([
            reshape_encoder(&mut b, &encoded),
            reshape_projection(&mut b, &projection.speech_first),
            reshape_projection(&mut b, &projection.speech),
            reshape_projection(&mut b, &projection.vision_first),
            reshape_projection(&mut b, &projection.vision),
        ]);
        let specs = phi4_audio_diagnostic_specs(config, frame_bucket)?;
        if checkpoints.len() != specs.len() {
            return Err("Phi4MM diagnostic checkpoint schema length drifted".to_string());
        }
        for (checkpoint, spec) in checkpoints.iter().zip(&specs) {
            if checkpoint.ty.shape != spec.shape {
                return Err(format!(
                    "Phi4MM diagnostic checkpoint {} has shape {:?}, expected {:?}",
                    spec.name, checkpoint.ty.shape, spec.shape
                ));
            }
        }
        let return_values = checkpoints
            .iter()
            .map(|value| value.name.as_str())
            .chain(std::iter::once(valid_length.name.as_str()))
            .collect::<Vec<_>>()
            .join(", ");
        let return_types = checkpoints
            .iter()
            .map(|value| value.ty.render())
            .chain(std::iter::once("tensor<i32>".to_string()))
            .collect::<Vec<_>>()
            .join(", ");
        Ok(format!(
            "module @audio {{\n  func.func public @diagnostic({signature}) -> ({return_types}) {{\n{body}    return {return_values} : {return_types} loc(\"audio diagnostic checkpoints\")\n  }}\n}}\n",
            body = b.body(),
        ))
    } else {
        let projected = b.reshape(
            &projection.selected,
            vec![1, encoded_len, config.projection_hidden],
        );
        Ok(format!(
            "module @audio {{\n  // #874 oracle stages: features tensor<1x{frame_bucket}x{input_size}xf32>, encoder tensor<1x{encoded_len}x{attention_dim}xf32>, projection {projected_ty}\n  func.func public @main({signature}) -> ({projected_ty}, tensor<i32>) {{\n{body}    return {projected}, {length} : {projected_ty}, tensor<i32> loc(\"projection\")\n  }}\n}}\n",
            projected_ty = projected.ty.render(),
            body = b.body(),
            projected = projected.name,
            length = valid_length.name,
            input_size = config.input_size,
            attention_dim = config.attention_dim,
        ))
    }
}

pub(crate) fn emit_phi4_audio(
    config: &Phi4AudioConfig,
    frame_bucket: usize,
) -> Result<String, String> {
    emit_phi4_audio_with(config, frame_bucket, Precision::F32)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::Read;
    use std::path::PathBuf;
    use std::process::Command;

    use super::*;

    fn pinned_config() -> Phi4AudioConfig {
        Phi4AudioConfig {
            input_size: 80,
            attention_dim: 1024,
            attention_heads: 16,
            num_blocks: 24,
            linear_units: 1536,
            time_reduction: 8,
            conv_channels: 1024,
            kernel_size: 3,
            relative_bias_max_distance: 500,
            projection_hidden: 3072,
        }
    }

    fn minimum_compile_config() -> Phi4AudioConfig {
        Phi4AudioConfig {
            // Keep the released 80-bin input so all grouped Conv2D stages
            // exercise production-shaped frequency reductions.
            input_size: 80,
            attention_dim: 4,
            attention_heads: 2,
            num_blocks: 1,
            linear_units: 3,
            time_reduction: 8,
            conv_channels: 2,
            kernel_size: 3,
            relative_bias_max_distance: 4,
            projection_hidden: 6,
        }
    }

    fn safetensors_header(path: &std::path::Path) -> Value {
        let mut file = File::open(path).expect("open safetensors shard");
        let mut length = [0_u8; 8];
        file.read_exact(&mut length).expect("read header length");
        let length = usize::try_from(u64::from_le_bytes(length)).expect("header length fits usize");
        let mut bytes = vec![0_u8; length];
        file.read_exact(&mut bytes)
            .expect("read safetensors header");
        serde_json::from_slice(&bytes).expect("parse safetensors header")
    }

    fn compile_audio_graph_cpu(
        config: &Phi4AudioConfig,
        frame_bucket: usize,
        precision: Precision,
        label: &str,
    ) {
        let compiler =
            PathBuf::from(std::env::var("MLXCEL_XLA_IREE_COMPILE").expect("set compiler path"));
        let graph =
            emit_phi4_audio_with(config, frame_bucket, precision).expect("emit audio graph");
        let stem = format!("mlxcel-phi4mm-audio-{label}-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stem}.mlir"));
        let output = std::env::temp_dir().join(format!("{stem}.vmfb"));
        std::fs::write(&input, graph).expect("write temporary StableHLO");
        let result = Command::new(compiler)
            .arg("--iree-input-type=stablehlo")
            .arg("--iree-hal-target-device=local")
            .arg("--iree-hal-local-target-device-backends=llvm-cpu")
            .arg(&input)
            .arg("-o")
            .arg(&output)
            .output()
            .expect("run iree-compile");
        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
        assert!(
            result.status.success(),
            "iree-compile failed:\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
    }

    #[test]
    fn pinned_schema_is_complete_and_propagates_exact_lengths() {
        let config = pinned_config();
        let specs = phi4_audio_weight_specs(&config);
        assert_eq!(specs.len(), 887);
        // #874's official host oracle uses the released non-causal NeMo
        // subsampler: three kernel-3, symmetric-pad-1, stride-2 stages.
        for (frames, projected_rows) in [
            (1, 1),
            (7, 1),
            (8, 1),
            (9, 2),
            (148, 19),
            (351, 44),
            (500, 63),
        ] {
            assert_eq!(
                config.encoded_bucket_len(frames).unwrap(),
                projected_rows,
                "{frames} host feature frames"
            );
            assert_eq!(
                config.encoded_valid_len(frames).unwrap(),
                projected_rows,
                "{frames} valid host feature frames"
            );
        }
        let oracle: Value = serde_json::from_str(include_str!(
            "../../../../../tests/fixtures/phi4mm_audio_parity.json"
        ))
        .expect("parse #874 pinned official oracle");
        assert_eq!(oracle["feature_shape"], serde_json::json!([1, 351, 80]));
        assert_eq!(oracle["audio_embed_size"], 44);
        assert_eq!(
            config
                .encoded_valid_len(oracle["feature_shape"][1].as_u64().unwrap() as usize)
                .unwrap(),
            oracle["audio_embed_size"].as_u64().unwrap() as usize
        );
    }

    #[test]
    fn shape_validation_rejects_missing_and_drifted_weights() {
        let config = pinned_config();
        let specs = phi4_audio_weight_specs(&config);
        let mut shapes = specs
            .iter()
            .map(|spec| (spec.name.clone(), spec.shape.clone()))
            .collect::<HashMap<_, _>>();
        validate_phi4_audio_weight_shapes(&config, |name| shapes.get(name).cloned()).unwrap();
        let first = specs[0].name.clone();
        shapes.remove(&first);
        assert!(
            validate_phi4_audio_weight_shapes(&config, |name| shapes.get(name).cloned())
                .unwrap_err()
                .contains("missing")
        );
        shapes.insert(first.clone(), vec![79]);
        let error = validate_phi4_audio_weight_shapes(&config, |name| shapes.get(name).cloned())
            .unwrap_err();
        assert!(error.contains(&first));
        assert!(error.contains("expected [80]"));
    }

    #[test]
    fn emitted_graph_tracks_oracle_stages_projection_mode_and_length() {
        let graph = emit_phi4_audio(&minimum_compile_config(), 32).expect("emit minimum graph");
        assert!(graph.contains("module @audio"));
        assert!(graph.contains("func.func public @main"));
        assert!(graph.contains("loc(\"features\")"));
        assert!(graph.contains("loc(\"frame_mask\")"));
        assert!(graph.contains("loc(\"frame_length\")"));
        assert!(graph.contains("loc(\"projection_mode\")"));
        assert!(graph.contains(
            "#874 oracle stages: features tensor<1x32x80xf32>, encoder tensor<1x4x4xf32>, projection tensor<1x4x6xf32>"
        ));
        assert!(graph.contains("stablehlo.convolution"));
        assert!(graph.contains("stablehlo.gather"));
        assert!(graph.contains("stablehlo.dot_general"));
        assert!(graph.contains("chlo.erf"));
        assert!(graph.contains("audio_projection.speech.0.weight"));
        assert!(graph.contains("audio_projection.vision.0.weight"));
        assert!(graph.contains("stablehlo.divide"));
        assert!(graph.contains("-> (tensor<1x4x6xf32>, tensor<i32>)"));
        assert!(!graph.contains("model.layers.0.self_attn"));
    }

    #[test]
    fn emitted_graph_rejects_invalid_attention_and_oversized_chunk() {
        let mut config = minimum_compile_config();
        config.attention_heads = 3;
        assert!(
            emit_phi4_audio(&config, 8)
                .unwrap_err()
                .contains("divisible")
        );

        let config = minimum_compile_config();
        assert!(
            emit_phi4_audio(&config, 33)
                .unwrap_err()
                .contains("exceeding the pinned")
        );
    }

    #[test]
    #[ignore = "requires MLXCEL_XLA_IREE_COMPILE pointing at iree-compile"]
    fn minimum_audio_graph_compiles_with_iree_cpu() {
        compile_audio_graph_cpu(&minimum_compile_config(), 32, Precision::F32, "minimum");
    }

    #[test]
    #[ignore = "requires MLXCEL_XLA_IREE_COMPILE pointing at iree-compile"]
    fn pinned_full_audio_graph_compiles_with_iree_cpu() {
        // The 512-frame production bucket contains #874's real 351-frame
        // oracle sample and emits all 24 Conformer blocks and 887 weights.
        compile_audio_graph_cpu(&pinned_config(), 512, Precision::F32, "pinned-full");
    }

    #[test]
    #[ignore = "requires MLXCEL_XLA_IREE_COMPILE pointing at iree-compile"]
    fn pinned_full_mixed_bf16_audio_graph_compiles_with_iree_cpu() {
        compile_audio_graph_cpu(
            &pinned_config(),
            512,
            Precision::Bf16,
            "pinned-full-mixed-bf16",
        );
    }

    #[test]
    #[ignore = "requires PHI4MM_MODEL_DIR pointing at the pinned official checkpoint"]
    fn real_checkpoint_has_exact_audio_schema_and_shapes() {
        let model_dir =
            PathBuf::from(std::env::var("PHI4MM_MODEL_DIR").expect("set PHI4MM_MODEL_DIR"));
        let config_text =
            std::fs::read_to_string(model_dir.join("config.json")).expect("read config");
        let config = Phi4AudioConfig::from_json_str(&config_text).expect("pinned audio config");
        let index: Value = serde_json::from_str(
            &std::fs::read_to_string(model_dir.join("model.safetensors.index.json"))
                .expect("read weight index"),
        )
        .expect("parse weight index");
        let weight_map = index["weight_map"].as_object().expect("weight_map object");
        let mut headers = HashMap::new();
        for file_name in weight_map.values().filter_map(Value::as_str) {
            headers
                .entry(file_name.to_string())
                .or_insert_with(|| safetensors_header(&model_dir.join(file_name)));
        }
        validate_phi4_audio_weight_shapes(&config, |name| {
            let file_name = weight_map.get(name)?.as_str()?;
            let metadata = headers.get(file_name)?.get(name)?;
            assert_eq!(
                metadata.get("dtype").and_then(Value::as_str),
                Some("BF16"),
                "{name} dtype"
            );
            metadata
                .get("shape")?
                .as_array()?
                .iter()
                .map(|dim| dim.as_u64().and_then(|value| usize::try_from(value).ok()))
                .collect()
        })
        .expect("complete exact Phi4MM audio schema");
    }
}
