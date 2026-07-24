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

//! StableHLO emitter for Molmo v1's ViT, attention pool, and SwiGLU projector.

use super::builder::{Builder, Ty, Val};
use super::molmo_vision_config::{
    MolmoVisionConfig, MolmoVisionWeightDType, MolmoVisionWeightSpec,
};

struct Args {
    values: Vec<Val>,
    declarations: Vec<String>,
    cursor: usize,
}

impl Args {
    fn new(specs: &[MolmoVisionWeightSpec]) -> Self {
        let mut values = Vec::with_capacity(specs.len() + 2);
        let mut declarations = Vec::with_capacity(specs.len() + 2);
        for (index, spec) in specs.iter().enumerate() {
            let element = match spec.dtype {
                MolmoVisionWeightDType::Float32 => "f32",
                MolmoVisionWeightDType::Float16 => "f16",
                MolmoVisionWeightDType::Uint32 => "ui32",
            };
            let ty = Ty::new(spec.shape.clone(), element);
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

    fn take_quant(&mut self, builder: &mut Builder, config: &MolmoVisionConfig) -> Val {
        let packed = self.take();
        let scales = self.take();
        let biases = self.take();
        builder.dequant_affine(&packed, &scales, &biases, config.bits, config.group_size)
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

fn bias_rows(builder: &mut Builder, value: &Val, bias: &Val) -> Val {
    let bias = builder.broadcast(bias, &[1], value.ty.shape.clone());
    builder.add(value, &bias)
}

fn quant_linear(
    builder: &mut Builder,
    args: &mut Args,
    config: &MolmoVisionConfig,
    value: &Val,
    bias: bool,
) -> Val {
    let weight = args.take_quant(builder, config);
    let projected = builder.linear_seq(value, &weight);
    if bias {
        bias_rows(builder, &projected, &args.take())
    } else {
        projected
    }
}

fn layer_norm(builder: &mut Builder, value: &Val, weight: &Val, bias: &Val, epsilon: f32) -> Val {
    let rows = value.ty.shape[0];
    let width = value.ty.shape[1];
    let zero = builder.const_f32(0.0);
    let width_scalar = builder.const_f32(width as f32);
    let width_rows = builder.broadcast(&width_scalar, &[], vec![rows]);
    let sum = builder.reduce_add(value, 1, &zero);
    let mean = builder.divide(&sum, &width_rows);
    let mean = builder.broadcast(&mean, &[0], vec![rows, width]);
    let centered = builder.subtract(value, &mean);
    let squared = builder.multiply(&centered, &centered);
    let squared_sum = builder.reduce_add(&squared, 1, &zero);
    let variance = builder.divide(&squared_sum, &width_rows);
    let epsilon = builder.const_f32(epsilon);
    let epsilon = builder.broadcast(&epsilon, &[], vec![rows]);
    let variance = builder.add(&variance, &epsilon);
    let inv_std = builder.rsqrt(&variance);
    let inv_std = builder.broadcast(&inv_std, &[0], vec![rows, width]);
    let normalized = builder.multiply(&centered, &inv_std);
    let weight = builder.broadcast(weight, &[1], vec![rows, width]);
    let bias = builder.broadcast(bias, &[1], vec![rows, width]);
    let normalized = builder.multiply(&normalized, &weight);
    builder.add(&normalized, &bias)
}

fn scalar_broadcast(builder: &mut Builder, value: f32, shape: Vec<usize>) -> Val {
    let scalar = builder.const_f32(value);
    builder.broadcast(&scalar, &[], shape)
}

fn tanh_gelu(builder: &mut Builder, value: &Val) -> Val {
    let shape = value.ty.shape.clone();
    let half = scalar_broadcast(builder, 0.5, shape.clone());
    let one = scalar_broadcast(builder, 1.0, shape.clone());
    let coefficient = scalar_broadcast(builder, 0.044_715, shape.clone());
    let scale = scalar_broadcast(builder, 0.797_884_6, shape);
    let squared = builder.multiply(value, value);
    let cubed = builder.multiply(&squared, value);
    let nonlinear = builder.multiply(&coefficient, &cubed);
    let inner = builder.add(value, &nonlinear);
    let scaled = builder.multiply(&scale, &inner);
    let tanh = builder.tanh(&scaled);
    let cdf = builder.add(&one, &tanh);
    let half_value = builder.multiply(value, &half);
    builder.multiply(&half_value, &cdf)
}

fn softmax_last(builder: &mut Builder, scores: &Val) -> Val {
    let last = scores.ty.shape.len() - 1;
    let negative_infinity = builder.const_f32(f32::NEG_INFINITY);
    let maximum = builder.reduce_max(scores, last, &negative_infinity);
    let mut broadcast_shape = maximum.ty.shape.clone();
    broadcast_shape.push(scores.ty.shape[last]);
    let dimensions = (0..maximum.ty.shape.len()).collect::<Vec<_>>();
    let maximum = builder.broadcast(&maximum, &dimensions, broadcast_shape);
    let shifted = builder.subtract(scores, &maximum);
    let exponentials = builder.exponential(&shifted);
    let zero = builder.const_f32(0.0);
    let denominator = builder.reduce_add(&exponentials, last, &zero);
    let mut broadcast_shape = denominator.ty.shape.clone();
    broadcast_shape.push(scores.ty.shape[last]);
    let dimensions = (0..denominator.ty.shape.len()).collect::<Vec<_>>();
    let denominator = builder.broadcast(&denominator, &dimensions, broadcast_shape);
    builder.divide(&exponentials, &denominator)
}

fn self_attention(
    builder: &mut Builder,
    args: &mut Args,
    config: &MolmoVisionConfig,
    hidden: &Val,
) -> Val {
    let crops = config.max_crops;
    let tokens = config.positions;
    let rows = crops * tokens;
    let q = quant_linear(builder, args, config, hidden, true);
    let k = quant_linear(builder, args, config, hidden, true);
    let v = quant_linear(builder, args, config, hidden, true);
    let q = builder.reshape(&q, vec![crops, tokens, config.heads, config.head_dim]);
    let q = builder.transpose(&q, &[0, 2, 1, 3]);
    let k = builder.reshape(&k, vec![crops, tokens, config.heads, config.head_dim]);
    let k = builder.transpose(&k, &[0, 2, 1, 3]);
    let v = builder.reshape(&v, vec![crops, tokens, config.heads, config.head_dim]);
    let v = builder.transpose(&v, &[0, 2, 1, 3]);
    let scores = builder.dot_general(
        &q,
        &k,
        &[0, 1],
        &[0, 1],
        &[3],
        &[3],
        vec![crops, config.heads, tokens, tokens],
    );
    let scale = scalar_broadcast(
        builder,
        (config.head_dim as f32).powf(-0.5),
        scores.ty.shape.clone(),
    );
    let scores = builder.multiply(&scores, &scale);
    let probabilities = softmax_last(builder, &scores);
    let context = builder.dot_general(
        &probabilities,
        &v,
        &[0, 1],
        &[0, 1],
        &[3],
        &[2],
        vec![crops, config.heads, tokens, config.head_dim],
    );
    let context = builder.transpose(&context, &[0, 2, 1, 3]);
    let context = builder.reshape(&context, vec![rows, config.hidden]);
    quant_linear(builder, args, config, &context, true)
}

fn encoder_layer(
    builder: &mut Builder,
    args: &mut Args,
    config: &MolmoVisionConfig,
    hidden: &Val,
) -> Val {
    let epsilon = f32::from_bits(config.layer_norm_eps_bits);
    let norm = layer_norm(builder, hidden, &args.take(), &args.take(), epsilon);
    let attention = self_attention(builder, args, config, &norm);
    let residual = builder.add(hidden, &attention);
    let norm = layer_norm(builder, &residual, &args.take(), &args.take(), epsilon);
    let expanded = quant_linear(builder, args, config, &norm, true);
    let activated = tanh_gelu(builder, &expanded);
    let contracted = quant_linear(builder, args, config, &activated, true);
    builder.add(&residual, &contracted)
}

fn mask_padding_crops(
    builder: &mut Builder,
    pixels: &Val,
    features: &Val,
    config: &MolmoVisionConfig,
) -> Val {
    let negative_one = scalar_broadcast(builder, -1.0, pixels.ty.shape.clone());
    let is_padding = builder.compare("EQ", pixels, &negative_one, "FLOAT");
    let is_padding = builder.convert(&is_padding, "f32");
    let zero = builder.const_f32(0.0);
    let patch_sums = builder.reduce_add(&is_padding, 2, &zero);
    let crop_sums = builder.reduce_add(&patch_sums, 1, &zero);
    let expected = scalar_broadcast(
        builder,
        (config.patches_per_crop * config.patch_width) as f32,
        vec![config.max_crops],
    );
    let all_padding = builder.compare("EQ", &crop_sums, &expected, "FLOAT");
    let all_padding = builder.broadcast(
        &all_padding,
        &[0],
        vec![
            config.max_crops,
            config.patches_per_crop,
            features.ty.shape[2],
        ],
    );
    let zero = scalar_broadcast(builder, 0.0, features.ty.shape.clone());
    builder.select(&all_padding, &zero, features)
}

fn apply_pad_embed(builder: &mut Builder, features: &Val, masks: &Val, pad_embed: &Val) -> Val {
    let width = features.ty.shape[2];
    let zero_mask = scalar_broadcast(builder, 0.0, masks.ty.shape.clone());
    let one_mask = scalar_broadcast(builder, 1.0, masks.ty.shape.clone());
    let all_padding = builder.compare("EQ", masks, &zero_mask, "FLOAT");
    let below_one = builder.compare("LT", masks, &one_mask, "FLOAT");
    let not_all_padding = builder.compare("NE", masks, &zero_mask, "FLOAT");
    let partial_padding = builder.and(&below_one, &not_all_padding);
    let all_padding = builder.convert(&all_padding, "f32");
    let partial_padding = builder.convert(&partial_padding, "f32");
    let all_padding = builder.reshape(
        &all_padding,
        vec![features.ty.shape[0], features.ty.shape[1], 1],
    );
    let partial_padding = builder.reshape(
        &partial_padding,
        vec![features.ty.shape[0], features.ty.shape[1], 1],
    );
    let all_padding = builder.broadcast(&all_padding, &[0, 1, 2], features.ty.shape.clone());
    let partial_padding =
        builder.broadcast(&partial_padding, &[0, 1, 2], features.ty.shape.clone());
    let complete = builder.slice(pad_embed, &[(0, 1), (0, width)]);
    let complete = builder.reshape(&complete, vec![width]);
    let complete = builder.broadcast(&complete, &[2], features.ty.shape.clone());
    let partial = builder.slice(pad_embed, &[(1, 2), (0, width)]);
    let partial = builder.reshape(&partial, vec![width]);
    let partial = builder.broadcast(&partial, &[2], features.ty.shape.clone());
    let complete = builder.multiply(&complete, &all_padding);
    let partial = builder.multiply(&partial, &partial_padding);
    let features = builder.add(features, &complete);
    builder.add(&features, &partial)
}

fn attention_pool(
    builder: &mut Builder,
    args: &mut Args,
    config: &MolmoVisionConfig,
    features: &Val,
) -> Val {
    let patch_side = usize::try_from((config.patches_per_crop as f64).sqrt() as u64)
        .expect("patch side fits usize");
    let block_h = patch_side / config.pool_h;
    let block_w = patch_side / config.pool_w;
    let width = features.ty.shape[2];
    let blocks = builder.reshape(
        features,
        vec![
            config.max_crops,
            block_h,
            config.pool_h,
            block_w,
            config.pool_w,
            width,
        ],
    );
    let blocks = builder.transpose(&blocks, &[0, 1, 3, 2, 4, 5]);
    let batch = config.max_crops * block_h * block_w;
    let pool_size = config.pool_h * config.pool_w;
    let blocks = builder.reshape(&blocks, vec![batch, pool_size, width]);
    let flattened = builder.reshape(&blocks, vec![batch * pool_size, width]);
    let zero = builder.const_f32(0.0);
    let sum = builder.reduce_add(&blocks, 1, &zero);
    let divisor = scalar_broadcast(builder, pool_size as f32, vec![batch]);
    let divisor = builder.broadcast(&divisor, &[0], vec![batch, width]);
    let query = builder.divide(&sum, &divisor);
    let q = quant_linear(builder, args, config, &query, true);
    let k = quant_linear(builder, args, config, &flattened, true);
    let v = quant_linear(builder, args, config, &flattened, true);
    let q = builder.reshape(&q, vec![batch, config.heads, 1, config.head_dim]);
    let k = builder.reshape(&k, vec![batch, pool_size, config.heads, config.head_dim]);
    let k = builder.transpose(&k, &[0, 2, 1, 3]);
    let v = builder.reshape(&v, vec![batch, pool_size, config.heads, config.head_dim]);
    let v = builder.transpose(&v, &[0, 2, 1, 3]);
    let scores = builder.dot_general(
        &q,
        &k,
        &[0, 1],
        &[0, 1],
        &[3],
        &[3],
        vec![batch, config.heads, 1, pool_size],
    );
    let scale = scalar_broadcast(
        builder,
        (config.head_dim as f32).powf(-0.5),
        scores.ty.shape.clone(),
    );
    let scores = builder.multiply(&scores, &scale);
    let probabilities = softmax_last(builder, &scores);
    let context = builder.dot_general(
        &probabilities,
        &v,
        &[0, 1],
        &[0, 1],
        &[3],
        &[2],
        vec![batch, config.heads, 1, config.head_dim],
    );
    let context = builder.reshape(&context, vec![batch, config.hidden]);
    let output = quant_linear(builder, args, config, &context, true);
    builder.reshape(
        &output,
        vec![
            config.max_crops,
            config.projected_rows_per_crop(),
            config.hidden,
        ],
    )
}

fn silu(builder: &mut Builder, value: &Val) -> Val {
    let negated = builder.negate(value);
    let exponential = builder.exponential(&negated);
    let one = scalar_broadcast(builder, 1.0, value.ty.shape.clone());
    let denominator = builder.add(&one, &exponential);
    builder.divide(value, &denominator)
}

pub(crate) fn emit_molmo_vision(config: &MolmoVisionConfig) -> String {
    let specs = config.weight_specs();
    let mut args = Args::new(&specs);
    let mut builder = Builder::new();
    let class_embedding = args.take();
    let patch_embedding = args.take();
    let position_embedding = args.take();
    let pre_ln_weight = args.take();
    let pre_ln_bias = args.take();
    let pixels = args.push_input(
        Ty::f32(vec![
            config.max_crops,
            config.patches_per_crop,
            config.patch_width,
        ]),
        "molmo.pixel_values",
    );
    let masks = args.push_input(
        Ty::f32(vec![config.max_crops, config.patches_per_crop]),
        "molmo.image_masks",
    );
    let rows = config.max_crops * config.patches_per_crop;
    let pixels_2d = builder.reshape(&pixels, vec![rows, config.patch_width]);
    let patches = builder.linear_seq(&pixels_2d, &patch_embedding);
    let patches = builder.reshape(
        &patches,
        vec![config.max_crops, config.patches_per_crop, config.hidden],
    );
    let class_embedding = builder.reshape(&class_embedding, vec![1, 1, config.hidden]);
    let class_embedding = builder.broadcast(
        &class_embedding,
        &[0, 1, 2],
        vec![config.max_crops, 1, config.hidden],
    );
    let hidden = builder.concatenate(&class_embedding, &patches, 1);
    let position_embedding = builder.broadcast(
        &position_embedding,
        &[1, 2],
        vec![config.max_crops, config.positions, config.hidden],
    );
    let hidden = builder.add(&hidden, &position_embedding);
    let hidden = builder.reshape(
        &hidden,
        vec![config.max_crops * config.positions, config.hidden],
    );
    let mut hidden = layer_norm(
        &mut builder,
        &hidden,
        &pre_ln_weight,
        &pre_ln_bias,
        f32::from_bits(config.layer_norm_eps_bits),
    );
    let mut selected = vec![None; config.selected_layers.len()];
    for layer in 0..config.emitted_layers() {
        hidden = encoder_layer(&mut builder, &mut args, config, &hidden);
        for (slot, &selected_layer) in config.selected_layers.iter().enumerate() {
            if layer == selected_layer {
                selected[slot] = Some(hidden.clone());
            }
        }
    }
    let mut selected_values = Vec::with_capacity(selected.len());
    for value in selected {
        let value = value.expect("selected layer was emitted");
        let value = builder.reshape(
            &value,
            vec![config.max_crops, config.positions, config.hidden],
        );
        selected_values.push(builder.slice(
            &value,
            &[
                (0, config.max_crops),
                (1, config.positions),
                (0, config.hidden),
            ],
        ));
    }
    let mut selected = selected_values.remove(0);
    for value in selected_values {
        selected = builder.concatenate(&selected, &value, 2);
    }
    let selected = mask_padding_crops(&mut builder, &pixels, &selected, config);
    let pad_embed = args.take();
    let selected = apply_pad_embed(&mut builder, &selected, &masks, &pad_embed);
    let pooled = attention_pool(&mut builder, &mut args, config, &selected);
    let pooled_rows = config.max_crops * config.projected_rows_per_crop();
    let pooled = builder.reshape(&pooled, vec![pooled_rows, config.hidden]);
    let gate = quant_linear(&mut builder, &mut args, config, &pooled, false);
    let gate = silu(&mut builder, &gate);
    let up = quant_linear(&mut builder, &mut args, config, &pooled, false);
    let activated = builder.multiply(&gate, &up);
    let projected = quant_linear(&mut builder, &mut args, config, &activated, false);
    let projected = builder.reshape(
        &projected,
        vec![
            config.max_crops,
            config.projected_rows_per_crop(),
            config.text_hidden,
        ],
    );
    assert_eq!(args.cursor, specs.len(), "Molmo vision schema drifted");
    format!(
        "module @molmo_vision {{\n  func.func public @main({signature}) -> {result} {{\n{body}    \
         return {value} : {result}\n  }}\n}}\n",
        signature = args.declarations.join(", "),
        result = projected.ty.render(),
        body = builder.body(),
        value = projected.name,
    )
}

#[cfg(test)]
#[path = "molmo_vision_tests.rs"]
mod tests;
