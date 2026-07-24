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

//! StableHLO emitter for Molmo2's flattened-patch ViT and indexed 2D pooler.

use super::builder::{Builder, Ty, Val};
use super::molmo2_config::{Molmo2VisionConfig, Molmo2VisionWeightSpec};

struct Args {
    values: Vec<Val>,
    declarations: Vec<String>,
    cursor: usize,
}

impl Args {
    fn new(specs: &[Molmo2VisionWeightSpec]) -> Self {
        let mut values = Vec::with_capacity(specs.len());
        let mut declarations = Vec::with_capacity(specs.len() + 2);
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

    fn input(&mut self, ty: Ty, name: &str) -> Val {
        let index = self.values.len();
        self.declarations
            .push(format!("%arg{index}: {} loc(\"{name}\")", ty.render()));
        let value = Builder::arg(index, ty);
        self.values.push(value.clone());
        value
    }
}

fn broadcast_bias(builder: &mut Builder, value: &Val, bias: &Val) -> Val {
    let rank = value.ty.shape.len();
    let bias = builder.broadcast(bias, &[rank - 1], value.ty.shape.clone());
    builder.add(value, &bias)
}

fn linear(builder: &mut Builder, value: &Val, weight: &Val, bias: Option<&Val>) -> Val {
    let rank = value.ty.shape.len();
    let mut output_shape = value.ty.shape[..rank - 1].to_vec();
    output_shape.push(weight.ty.shape[0]);
    let output = builder.dot_general(value, weight, &[], &[], &[rank - 1], &[1], output_shape);
    bias.map_or(output.clone(), |bias| {
        broadcast_bias(builder, &output, bias)
    })
}

fn layer_norm(builder: &mut Builder, value: &Val, weight: &Val, bias: &Val, epsilon: f32) -> Val {
    let rank = value.ty.shape.len();
    let axis = rank - 1;
    let width = value.ty.shape[axis];
    let leading = value.ty.shape[..axis].to_vec();
    let zero = builder.const_f32(0.0);
    let width_value = builder.const_f32(width as f32);
    let width_value = builder.broadcast(&width_value, &[], leading.clone());
    let sum = builder.reduce_add(value, axis, &zero);
    let mean = builder.divide(&sum, &width_value);
    let mean = builder.broadcast(
        &mean,
        &(0..axis).collect::<Vec<_>>(),
        value.ty.shape.clone(),
    );
    let centered = builder.subtract(value, &mean);
    let squared = builder.multiply(&centered, &centered);
    let variance = builder.reduce_add(&squared, axis, &zero);
    let variance = builder.divide(&variance, &width_value);
    let epsilon = builder.const_f32(epsilon);
    let epsilon = builder.broadcast(&epsilon, &[], leading.clone());
    let variance = builder.add(&variance, &epsilon);
    let inv_std = builder.rsqrt(&variance);
    let inv_std = builder.broadcast(
        &inv_std,
        &(0..axis).collect::<Vec<_>>(),
        value.ty.shape.clone(),
    );
    let normalized = builder.multiply(&centered, &inv_std);
    let weight = builder.broadcast(weight, &[axis], value.ty.shape.clone());
    let bias = builder.broadcast(bias, &[axis], value.ty.shape.clone());
    let normalized = builder.multiply(&normalized, &weight);
    builder.add(&normalized, &bias)
}

fn tanh_gelu(builder: &mut Builder, value: &Val) -> Val {
    let shape = value.ty.shape.clone();
    let half = builder.const_f32(0.5);
    let half = builder.broadcast(&half, &[], shape.clone());
    let one = builder.const_f32(1.0);
    let one = builder.broadcast(&one, &[], shape.clone());
    let coefficient = builder.const_f32(0.044_715);
    let coefficient = builder.broadcast(&coefficient, &[], shape.clone());
    let scale = builder.const_f32(0.797_884_6);
    let scale = builder.broadcast(&scale, &[], shape);
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

fn silu(builder: &mut Builder, value: &Val) -> Val {
    let shape = value.ty.shape.clone();
    let one = builder.const_f32(1.0);
    let one = builder.broadcast(&one, &[], shape);
    let negative = builder.negate(value);
    let exponential = builder.exponential(&negative);
    let denominator = builder.add(&one, &exponential);
    builder.divide(value, &denominator)
}

fn softmax_last(builder: &mut Builder, scores: &Val) -> Val {
    let axis = scores.ty.shape.len() - 1;
    let leading = scores.ty.shape[..axis].to_vec();
    let negative_infinity = builder.const_f32(f32::NEG_INFINITY);
    let maximum = builder.reduce_max(scores, axis, &negative_infinity);
    let maximum = builder.broadcast(
        &maximum,
        &(0..axis).collect::<Vec<_>>(),
        scores.ty.shape.clone(),
    );
    let shifted = builder.subtract(scores, &maximum);
    let exponentials = builder.exponential(&shifted);
    let zero = builder.const_f32(0.0);
    let denominator = builder.reduce_add(&exponentials, axis, &zero);
    let denominator = builder.broadcast(
        &denominator,
        &(0..axis).collect::<Vec<_>>(),
        scores.ty.shape.clone(),
    );
    debug_assert_eq!(denominator.ty.shape[..axis], leading);
    builder.divide(&exponentials, &denominator)
}

fn selected_slot(selected_layers: &[usize], layer: usize) -> Option<usize> {
    selected_layers
        .iter()
        .position(|&selected| selected == layer)
}

fn self_attention(
    builder: &mut Builder,
    hidden: &Val,
    args: &mut Args,
    config: &Molmo2VisionConfig,
) -> Val {
    let q_weight = args.take();
    let q_bias = args.take();
    let k_weight = args.take();
    let k_bias = args.take();
    let v_weight = args.take();
    let v_bias = args.take();
    let o_weight = args.take();
    let o_bias = args.take();
    let q = linear(builder, hidden, &q_weight, Some(&q_bias));
    let k = linear(builder, hidden, &k_weight, Some(&k_bias));
    let v = linear(builder, hidden, &v_weight, Some(&v_bias));
    let crops = config.static_crops;
    let tokens = config.patches_per_crop;
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
    let scale = builder.const_f32((config.head_dim as f32).powf(-0.5));
    let scale = builder.broadcast(&scale, &[], scores.ty.shape.clone());
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
    let context = builder.reshape(&context, vec![crops, tokens, config.hidden]);
    linear(builder, &context, &o_weight, Some(&o_bias))
}

fn indexed_pool(
    builder: &mut Builder,
    features: &Val,
    signed_indices: &Val,
    args: &mut Args,
    config: &Molmo2VisionConfig,
) -> (Val, Val) {
    let groups = config.static_pool_groups;
    let group_size = config.pool_size;
    let zero_i32 = builder.const_i32(0);
    let zero_indices = builder.broadcast(&zero_i32, &[], vec![groups, group_size]);
    let valid = builder.compare("GE", signed_indices, &zero_indices, "SIGNED");
    let safe = builder.maximum(signed_indices, &zero_indices);
    let safe = builder.reshape(&safe, vec![groups, group_size, 1]);
    let gathered = builder.gather_rows_nd(features, &safe);
    let valid_f32 = builder.convert(&valid, "f32");
    let valid_features = builder.broadcast(&valid_f32, &[0, 1], gathered.ty.shape.clone());
    let gathered = builder.multiply(&gathered, &valid_features);
    let zero_f32 = builder.const_f32(0.0);
    let sums = builder.reduce_add(&gathered, 1, &zero_f32);
    let counts = builder.reduce_add(&valid_f32, 1, &zero_f32);
    let denominator = if config.pooling_attention_mask {
        let one = builder.const_f32(1.0);
        let ones = builder.broadcast(&one, &[], vec![groups]);
        builder.maximum(&counts, &ones)
    } else {
        // Match the MLX reference: invalid entries are zeroed above, but an
        // unmasked pooling query is the mean over the full fixed-size window.
        let group_size = builder.const_f32(group_size as f32);
        builder.broadcast(&group_size, &[], vec![groups])
    };
    let denominator = builder.broadcast(&denominator, &[0], vec![groups, config.selected_width()]);
    let query = builder.divide(&sums, &denominator);

    let q_weight = args.take();
    let q_bias = args.take();
    let k_weight = args.take();
    let k_bias = args.take();
    let v_weight = args.take();
    let v_bias = args.take();
    let o_weight = args.take();
    let o_bias = args.take();
    let q = linear(builder, &query, &q_weight, Some(&q_bias));
    let k = linear(builder, &gathered, &k_weight, Some(&k_bias));
    let v = linear(builder, &gathered, &v_weight, Some(&v_bias));
    let q = builder.reshape(&q, vec![groups, 1, config.pool_heads, config.pool_head_dim]);
    let q = builder.transpose(&q, &[0, 2, 1, 3]);
    let k = builder.reshape(
        &k,
        vec![groups, group_size, config.pool_heads, config.pool_head_dim],
    );
    let k = builder.transpose(&k, &[0, 2, 1, 3]);
    let v = builder.reshape(
        &v,
        vec![groups, group_size, config.pool_heads, config.pool_head_dim],
    );
    let v = builder.transpose(&v, &[0, 2, 1, 3]);
    let scores = builder.dot_general(
        &q,
        &k,
        &[0, 1],
        &[0, 1],
        &[3],
        &[3],
        vec![groups, config.pool_heads, 1, group_size],
    );
    let scale = builder.const_f32((config.pool_head_dim as f32).powf(-0.5));
    let scale = builder.broadcast(&scale, &[], scores.ty.shape.clone());
    let mut scores = builder.multiply(&scores, &scale);
    if config.pooling_attention_mask {
        let mask = builder.broadcast(&valid, &[0, 3], scores.ty.shape.clone());
        let masked = builder.const_f32(-1.0e30);
        let masked = builder.broadcast(&masked, &[], scores.ty.shape.clone());
        scores = builder.select(&mask, &scores, &masked);
    }
    let probabilities = softmax_last(builder, &scores);
    let context = builder.dot_general(
        &probabilities,
        &v,
        &[0, 1],
        &[0, 1],
        &[3],
        &[2],
        vec![groups, config.pool_heads, 1, config.pool_head_dim],
    );
    let context = builder.transpose(&context, &[0, 2, 1, 3]);
    let context = builder.reshape(&context, vec![groups, config.pool_hidden]);
    let pooled = linear(builder, &context, &o_weight, Some(&o_bias));
    (pooled, counts)
}

pub(crate) fn emit_molmo2_vision(config: &Molmo2VisionConfig) -> String {
    let specs = config.weight_specs();
    let mut args = Args::new(&specs);
    let patch_weight = args.take();
    let patch_bias = args.take();
    let position_embedding = args.take();
    let patches = args.input(
        Ty::f32(vec![
            config.static_crops,
            config.patches_per_crop,
            config.patch_dim,
        ]),
        "molmo2.patches",
    );
    let signed_indices = args.input(
        Ty::new(vec![config.static_pool_groups, config.pool_size], "i32"),
        "molmo2.image_token_pooling.signed",
    );
    let mut builder = Builder::new();
    let mut hidden = linear(&mut builder, &patches, &patch_weight, Some(&patch_bias));
    let position = builder.broadcast(
        &position_embedding,
        &[1, 2],
        vec![config.static_crops, config.patches_per_crop, config.hidden],
    );
    hidden = builder.add(&hidden, &position);
    let mut selected = vec![None::<Val>; config.selected_layers.len()];
    for layer in 0..config.emitted_layers {
        // Norm weights follow attention projection weights in the persisted
        // schema. Pull them before emitting the attention and pass normalized
        // hidden to the projection sequence.
        let block_start = args.cursor;
        let norm_weight = args.values[block_start + 8].clone();
        let norm_bias = args.values[block_start + 9].clone();
        let normalized = layer_norm(
            &mut builder,
            &hidden,
            &norm_weight,
            &norm_bias,
            config.layer_norm_eps,
        );
        let attention = self_attention(&mut builder, &normalized, &mut args, config);
        // encoder_block consumes the already-taken attention schema, so finish
        // this block explicitly.
        let _attention_norm_weight = args.take();
        let _attention_norm_bias = args.take();
        let ffn_norm_weight = args.take();
        let ffn_norm_bias = args.take();
        let residual = builder.add(&hidden, &attention);
        let normalized = layer_norm(
            &mut builder,
            &residual,
            &ffn_norm_weight,
            &ffn_norm_bias,
            config.layer_norm_eps,
        );
        let w1 = args.take();
        let b1 = args.take();
        let w2 = args.take();
        let b2 = args.take();
        let mlp = linear(&mut builder, &normalized, &w1, Some(&b1));
        let mlp = tanh_gelu(&mut builder, &mlp);
        let mlp = linear(&mut builder, &mlp, &w2, Some(&b2));
        hidden = builder.add(&residual, &mlp);
        if let Some(slot) = selected_slot(&config.selected_layers, layer) {
            selected[slot] = Some(hidden.clone());
        }
    }
    let mut selected = selected.into_iter();
    let mut selected_features = match selected.next() {
        Some(Some(feature)) => feature,
        _ => unreachable!("validated Molmo2 selected layers must be emitted"),
    };
    for feature in selected {
        let feature = match feature {
            Some(feature) => feature,
            None => unreachable!("validated Molmo2 selected layer must be emitted"),
        };
        selected_features = builder.concatenate(&selected_features, &feature, 2);
    }
    let selected_features = builder.reshape(
        &selected_features,
        vec![
            config.static_crops * config.patches_per_crop,
            config.selected_width(),
        ],
    );
    let (pooled, _counts) = indexed_pool(
        &mut builder,
        &selected_features,
        &signed_indices,
        &mut args,
        config,
    );
    let w1 = args.take();
    let w2 = args.take();
    let w3 = args.take();
    let gate = linear(&mut builder, &pooled, &w1, None);
    let gate = silu(&mut builder, &gate);
    let up = linear(&mut builder, &pooled, &w3, None);
    let projected = builder.multiply(&gate, &up);
    let projected = linear(&mut builder, &projected, &w2, None);
    assert_eq!(
        args.cursor,
        specs.len(),
        "Molmo2 vision weight schema drifted"
    );
    format!(
        "module @molmo2_vision {{\n  func.func public @main({signature}) -> {output} {{\n{body}    return {value} : {output}\n  }}\n}}\n",
        signature = args.declarations.join(", "),
        output = projected.ty.render(),
        body = builder.body(),
        value = projected.name,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(pooling_attention_mask: bool) -> Molmo2VisionConfig {
        Molmo2VisionConfig::from_json_strs(
            &serde_json::json!({
                "model_type":"molmo2","image_patch_id":151938,
                "vit_config":{"hidden_size":8,"intermediate_size":16,"num_attention_heads":2,
                    "head_dim":4,"num_hidden_layers":2,"image_default_input_size":[28,28],
                    "image_patch_size":14,"image_num_pos":4,"layer_norm_eps":1e-6},
                "adapter_config":{"hidden_size":8,"intermediate_size":12,"text_hidden_size":10,
                    "num_attention_heads":2,"head_dim":4,"vit_layers":[0,1],
                    "pooling_attention_mask":pooling_attention_mask}
            })
            .to_string(),
            &serde_json::json!({"patch_size":14,"max_crops":1,"overlap_margins":[0,0],
                "pooling_size":[2,2],"size":{"height":28,"width":28}})
            .to_string(),
        )
        .unwrap()
    }

    #[test]
    fn emitted_pooling_clamps_then_masks_and_keeps_additive_merge_outside_graph() {
        let config = test_config(true);
        let mlir = emit_molmo2_vision(&config);
        assert!(mlir.contains("molmo2.image_token_pooling.signed"));
        assert!(mlir.contains("stablehlo.compare GE"));
        assert!(mlir.contains("stablehlo.maximum"));
        assert!(mlir.contains("\"stablehlo.gather\""));
        assert!(mlir.contains("stablehlo.select"));
        assert!(mlir.contains("vision_tower.image_vit.patch_embedding.weight"));
        assert!(mlir.contains("vision_tower.image_projector.w3.weight"));
        assert!(!mlir.contains("image_input_idx"));
    }

    #[test]
    fn unmasked_pooling_uses_full_window_denominator() {
        let mlir = emit_molmo2_vision(&test_config(false));
        assert!(mlir.contains("stablehlo.constant dense<0x40800000> : tensor<f32>"));
        assert!(!mlir.contains("stablehlo.select"));
    }

    #[test]
    fn selected_layers_keep_adapter_order_instead_of_encoder_order() {
        assert_eq!(selected_slot(&[22, 16], 22), Some(0));
        assert_eq!(selected_slot(&[22, 16], 16), Some(1));
        assert_eq!(selected_slot(&[22, 16], 21), None);
    }
}
