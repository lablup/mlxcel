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

//! Qwen2.5-VL StableHLO vision tower.
//!
//! Host preprocessing supplies the exact window permutation plus separate
//! full-media and window-isolated masks. The graph keeps the permutation
//! static for one bucket while selecting the checkpoint-declared full
//! attention layers during emission.

use super::builder::{Builder, Ty, Val};
use super::qwen2_vl::{
    QWEN2_VL_PATCH_BUCKETS, Qwen2VlConfig, Qwen2VlWeightSpec, QwenVlVisionVariant,
};

#[cfg(feature = "diagnostics")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Qwen25VlVisionDiagnosticLayout {
    pub(crate) window_layer_index: usize,
    pub(crate) full_layer_indices: Vec<usize>,
}

struct Args {
    values: Vec<Val>,
    declarations: Vec<String>,
    cursor: usize,
}

impl Args {
    fn new(specs: &[Qwen2VlWeightSpec]) -> Self {
        let mut values = Vec::with_capacity(specs.len() + 4);
        let mut declarations = Vec::with_capacity(specs.len() + 4);
        for (index, spec) in specs.iter().enumerate() {
            let ty = Ty::f32(spec.shape.clone());
            declarations.push(format!(
                "%arg{index}: {} loc(\"{}\")",
                ty.render(),
                spec.name
            ));
            values.push(Builder::arg(index, ty));
        }
        Self {
            values,
            declarations,
            cursor: 0,
        }
    }

    fn take(&mut self) -> Val {
        let value = self.values[self.cursor].clone();
        self.cursor += 1;
        value
    }

    fn push_input(&mut self, ty: Ty, name: &str) -> Val {
        let index = self.values.len();
        self.declarations
            .push(format!("%arg{index}: {} loc(\"{name}\")", ty.render()));
        let value = Builder::arg(index, ty);
        self.values.push(value.clone());
        value
    }
}

fn bias_2d(builder: &mut Builder, value: &Val, bias: &Val) -> Val {
    let rows = value.ty.shape[0];
    let width = value.ty.shape[1];
    let bias = builder.broadcast(bias, &[1], vec![rows, width]);
    builder.add(value, &bias)
}

fn linear_2d(builder: &mut Builder, value: &Val, weight: &Val, bias: &Val) -> Val {
    let value = builder.linear_seq(value, weight);
    bias_2d(builder, &value, bias)
}

fn rms_norm(builder: &mut Builder, value: &Val, weight: &Val, epsilon: f32) -> Val {
    let rows = value.ty.shape[0];
    let width = value.ty.shape[1];
    let zero = builder.const_f32(0.0);
    let squared = builder.multiply(value, value);
    let sum = builder.reduce_add(&squared, 1, &zero);
    let width_scalar = builder.const_f32(width as f32);
    let width_rows = builder.broadcast(&width_scalar, &[], vec![rows]);
    let mean = builder.divide(&sum, &width_rows);
    let epsilon = builder.const_f32(epsilon);
    let epsilon = builder.broadcast(&epsilon, &[], vec![rows]);
    let mean = builder.add(&mean, &epsilon);
    let inverse = builder.rsqrt(&mean);
    let inverse = builder.broadcast(&inverse, &[0], vec![rows, width]);
    let normalized = builder.multiply(value, &inverse);
    let weight = builder.broadcast(weight, &[1], vec![rows, width]);
    builder.multiply(&normalized, &weight)
}

fn silu(builder: &mut Builder, value: &Val) -> Val {
    let one = builder.const_f32(1.0);
    let one = builder.broadcast(&one, &[], value.ty.shape.clone());
    let negative = builder.negate(value);
    let exponential = builder.exponential(&negative);
    let denominator = builder.add(&one, &exponential);
    builder.divide(value, &denominator)
}

fn exact_gelu(builder: &mut Builder, value: &Val) -> Val {
    let shape = value.ty.shape.clone();
    let half = builder.const_f32(0.5);
    let half = builder.broadcast(&half, &[], shape.clone());
    let one = builder.const_f32(1.0);
    let one = builder.broadcast(&one, &[], shape.clone());
    let inv_sqrt_two = builder.const_f32(std::f32::consts::FRAC_1_SQRT_2);
    let inv_sqrt_two = builder.broadcast(&inv_sqrt_two, &[], shape);
    let scaled = builder.multiply(value, &inv_sqrt_two);
    let erf = builder.erf(&scaled);
    let cdf = builder.add(&one, &erf);
    let half_value = builder.multiply(value, &half);
    builder.multiply(&half_value, &cdf)
}

fn rotate_half(builder: &mut Builder, value: &Val) -> Val {
    let tokens = value.ty.shape[0];
    let heads = value.ty.shape[1];
    let width = value.ty.shape[2];
    let half = width / 2;
    let first = builder.slice(value, &[(0, tokens), (0, heads), (0, half)]);
    let second = builder.slice(value, &[(0, tokens), (0, heads), (half, width)]);
    let second = builder.negate(&second);
    builder.concatenate(&second, &first, 2)
}

fn apply_vision_rope(builder: &mut Builder, value: &Val, freqs: &Val) -> Val {
    let tokens = value.ty.shape[0];
    let heads = value.ty.shape[1];
    let half = freqs.ty.shape[1];
    let cos = builder.cosine(freqs);
    let sin = builder.sine(freqs);
    let cos = builder.concatenate(&cos, &cos, 1);
    let sin = builder.concatenate(&sin, &sin, 1);
    let cos = builder.broadcast(&cos, &[0, 2], vec![tokens, heads, half * 2]);
    let sin = builder.broadcast(&sin, &[0, 2], vec![tokens, heads, half * 2]);
    let rotated = rotate_half(builder, value);
    let direct = builder.multiply(value, &cos);
    let rotated = builder.multiply(&rotated, &sin);
    builder.add(&direct, &rotated)
}

fn attention(
    builder: &mut Builder,
    hidden: &Val,
    args: &mut Args,
    config: &Qwen2VlConfig,
    freqs: &Val,
    attention_bias: &Val,
) -> Val {
    let tokens = hidden.ty.shape[0];
    let head_dim = config.hidden / config.heads;
    let qkv = linear_2d(builder, hidden, &args.take(), &args.take());
    let qkv = builder.reshape(&qkv, vec![tokens, 3, config.heads, head_dim]);
    let q = builder.slice(
        &qkv,
        &[(0, tokens), (0, 1), (0, config.heads), (0, head_dim)],
    );
    let k = builder.slice(
        &qkv,
        &[(0, tokens), (1, 2), (0, config.heads), (0, head_dim)],
    );
    let v = builder.slice(
        &qkv,
        &[(0, tokens), (2, 3), (0, config.heads), (0, head_dim)],
    );
    let q = builder.reshape(&q, vec![tokens, config.heads, head_dim]);
    let k = builder.reshape(&k, vec![tokens, config.heads, head_dim]);
    let v = builder.reshape(&v, vec![tokens, config.heads, head_dim]);
    let q = apply_vision_rope(builder, &q, freqs);
    let k = apply_vision_rope(builder, &k, freqs);
    let q = builder.transpose(&q, &[1, 0, 2]);
    let k = builder.transpose(&k, &[1, 0, 2]);
    let v = builder.transpose(&v, &[1, 0, 2]);
    let scores = builder.dot_general(
        &q,
        &k,
        &[0],
        &[0],
        &[2],
        &[2],
        vec![config.heads, tokens, tokens],
    );
    let scale = builder.const_f32((head_dim as f32).powf(-0.5));
    let scale = builder.broadcast(&scale, &[], vec![config.heads, tokens, tokens]);
    let scores = builder.multiply(&scores, &scale);
    let bias = builder.broadcast(attention_bias, &[1, 2], vec![config.heads, tokens, tokens]);
    let scores = builder.add(&scores, &bias);
    let negative_infinity = builder.const_f32(f32::NEG_INFINITY);
    let maximum = builder.reduce_max(&scores, 2, &negative_infinity);
    let maximum = builder.broadcast(&maximum, &[0, 1], vec![config.heads, tokens, tokens]);
    let shifted = builder.subtract(&scores, &maximum);
    let exponentials = builder.exponential(&shifted);
    let zero = builder.const_f32(0.0);
    let denominator = builder.reduce_add(&exponentials, 2, &zero);
    let denominator = builder.broadcast(&denominator, &[0, 1], vec![config.heads, tokens, tokens]);
    let probabilities = builder.divide(&exponentials, &denominator);
    let context = builder.dot_general(
        &probabilities,
        &v,
        &[0],
        &[0],
        &[2],
        &[1],
        vec![config.heads, tokens, head_dim],
    );
    let context = builder.transpose(&context, &[1, 0, 2]);
    let context = builder.reshape(&context, vec![tokens, config.hidden]);
    linear_2d(builder, &context, &args.take(), &args.take())
}

fn encoder_layer(
    builder: &mut Builder,
    hidden: &Val,
    args: &mut Args,
    config: &Qwen2VlConfig,
    freqs: &Val,
    attention_bias: &Val,
) -> Val {
    let norm1 = rms_norm(builder, hidden, &args.take(), config.layer_norm_eps);
    let attention = attention(builder, &norm1, args, config, freqs, attention_bias);
    let residual = builder.add(hidden, &attention);
    let norm2 = rms_norm(builder, &residual, &args.take(), config.layer_norm_eps);
    let gate = linear_2d(builder, &norm2, &args.take(), &args.take());
    let gate = silu(builder, &gate);
    let up = linear_2d(builder, &norm2, &args.take(), &args.take());
    let activated = builder.multiply(&gate, &up);
    let down = linear_2d(builder, &activated, &args.take(), &args.take());
    builder.add(&residual, &down)
}

fn emit_qwen2_5_vl_inner(
    config: &Qwen2VlConfig,
    patch_bucket: usize,
    #[cfg(feature = "diagnostics")] diagnostic_layout: Option<&Qwen25VlVisionDiagnosticLayout>,
) -> String {
    assert!(
        QWEN2_VL_PATCH_BUCKETS.contains(&patch_bucket),
        "unqualified Qwen2.5-VL patch bucket"
    );
    let QwenVlVisionVariant::Qwen25 {
        full_attention_blocks,
        ..
    } = &config.variant
    else {
        panic!("Qwen2.5-VL emitter requires the Qwen2.5 variant");
    };
    let specs = config.weight_specs();
    let mut args = Args::new(&specs);
    let mut builder = Builder::new();
    let patch_weight = args.take();
    let patch_width =
        config.channels * config.temporal_patch_size * config.patch_size * config.patch_size;
    let head_dim = config.hidden / config.heads;
    let patches = args.push_input(Ty::f32(vec![patch_bucket, patch_width]), "patches.windowed");
    let freqs = args.push_input(
        Ty::f32(vec![patch_bucket, head_dim / 2]),
        "vision_rope.windowed_freqs",
    );
    let full_bias = args.push_input(
        Ty::f32(vec![patch_bucket, patch_bucket]),
        "full_attention.bias",
    );
    let window_bias = args.push_input(
        Ty::f32(vec![patch_bucket, patch_bucket]),
        "window_attention.bias",
    );
    let mut hidden = builder.linear_seq(&patches, &patch_weight);
    #[cfg(feature = "diagnostics")]
    let reordered_patch_embedding = diagnostic_layout.map(|_| hidden.clone());
    #[cfg(feature = "diagnostics")]
    let mut window_layer_state = None;
    #[cfg(feature = "diagnostics")]
    let mut full_layer_states = Vec::new();
    for layer in 0..config.depth {
        let bias = if full_attention_blocks.contains(&layer) {
            &full_bias
        } else {
            &window_bias
        };
        hidden = encoder_layer(&mut builder, &hidden, &mut args, config, &freqs, bias);
        #[cfg(feature = "diagnostics")]
        if let Some(layout) = diagnostic_layout {
            if layer == layout.window_layer_index {
                window_layer_state = Some(hidden.clone());
            }
            if layout.full_layer_indices.contains(&layer) {
                full_layer_states.push(hidden.clone());
            }
        }
    }
    hidden = rms_norm(&mut builder, &hidden, &args.take(), config.layer_norm_eps);
    let merge_width = config.hidden * config.spatial_merge_size * config.spatial_merge_size;
    let merged = builder.reshape(
        &hidden,
        vec![
            patch_bucket / (config.spatial_merge_size * config.spatial_merge_size),
            merge_width,
        ],
    );
    let merged = linear_2d(&mut builder, &merged, &args.take(), &args.take());
    let merged = exact_gelu(&mut builder, &merged);
    let projected = linear_2d(&mut builder, &merged, &args.take(), &args.take());
    assert_eq!(args.cursor, specs.len(), "Qwen2.5-VL weight schema drifted");
    #[cfg(feature = "diagnostics")]
    if let Some(layout) = diagnostic_layout {
        let mut outputs = vec![
            reordered_patch_embedding.expect("diagnostics capture patch embedding"),
            window_layer_state.expect("diagnostics capture window layer"),
        ];
        assert_eq!(
            full_layer_states.len(),
            layout.full_layer_indices.len(),
            "diagnostics capture every configured full-attention layer"
        );
        outputs.extend(full_layer_states);
        outputs.push(projected);
        let result_types = outputs
            .iter()
            .map(|value| value.ty.render())
            .collect::<Vec<_>>();
        return format!(
            "module @qwen2_vl_vision {{\n  func.func public @main({signature}) -> ({result_signature}) {{\n{body}    return {results} : {result_signature}\n  }}\n}}\n",
            signature = args.declarations.join(", "),
            result_signature = result_types.join(", "),
            body = builder.body(),
            results = outputs
                .iter()
                .map(|value| value.name.as_str())
                .collect::<Vec<_>>()
                .join(", "),
        );
    }
    format!(
        "module @qwen2_vl_vision {{\n  func.func public @main({signature}) -> {result_type} {{\n{body}    return {result} : {result_type}\n  }}\n}}\n",
        signature = args.declarations.join(", "),
        result_type = projected.ty.render(),
        body = builder.body(),
        result = projected.name,
    )
}

pub(super) fn emit_qwen2_5_vl(config: &Qwen2VlConfig, patch_bucket: usize) -> String {
    emit_qwen2_5_vl_inner(
        config,
        patch_bucket,
        #[cfg(feature = "diagnostics")]
        None,
    )
}

#[cfg(feature = "diagnostics")]
pub(crate) fn emit_qwen2_5_vl_diagnostics(
    config: &Qwen2VlConfig,
    patch_bucket: usize,
) -> Result<(String, Qwen25VlVisionDiagnosticLayout), String> {
    let QwenVlVisionVariant::Qwen25 {
        full_attention_blocks,
        ..
    } = &config.variant
    else {
        return Err("Qwen2.5-VL diagnostics require the Qwen2.5 vision variant".to_string());
    };
    let window_layer_index = (0..config.depth)
        .find(|layer| !full_attention_blocks.contains(layer))
        .ok_or_else(|| {
            "Qwen2.5-VL diagnostics require at least one window-attention layer".to_string()
        })?;
    if full_attention_blocks.is_empty() {
        return Err(
            "Qwen2.5-VL diagnostics require at least one configured full-attention layer"
                .to_string(),
        );
    }
    let layout = Qwen25VlVisionDiagnosticLayout {
        window_layer_index,
        full_layer_indices: (0..config.depth)
            .filter(|layer| full_attention_blocks.contains(layer))
            .collect(),
    };
    Ok((
        emit_qwen2_5_vl_inner(config, patch_bucket, Some(&layout)),
        layout,
    ))
}
