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

//! StableHLO emitter for the pinned LLaVA SigLIP/CLIP tower and projector.
//!
//! `vision.main` accepts one contiguous normalized image in canonical
//! `[1, channels, image_size, image_size]` F32 layout and returns projected
//! image tokens in `[image_tokens, text_hidden]` F32. Image resize and
//! normalization remain outside the graph. Every weight stays in canonical
//! Hugging Face/PyTorch layout; patch convolution is the only OIHW consumer and
//! linear weights remain `[out, in]`.

use super::builder::{Builder, Ty, Val};
use super::numeric_ops::{exact_gelu, layer_norm_2d as layer_norm, stable_softmax, tanh_gelu};
use super::vision_config::{
    LlavaVisionConfig, VisionActivation, VisionProjector, VisionWeightSpec,
};

struct Args {
    values: Vec<Val>,
    declarations: Vec<String>,
    cursor: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pinned() -> Option<LlavaVisionConfig> {
        let path = std::path::Path::new("/tmp/mlxcel-llava-hf-1090956d");
        path.join("config.json")
            .is_file()
            .then(|| LlavaVisionConfig::from_model_dir(path).unwrap())
    }

    #[test]
    fn pinned_siglip_graph_has_exact_unbatched_output_contract() {
        let Some(config) = pinned() else {
            return;
        };
        let mlir = emit_vision(&config);
        assert!(mlir.contains("stablehlo.convolution"));
        assert!(mlir.contains("stablehlo.reduce"));
        assert!(mlir.contains("stablehlo.exponential"));
        assert!(mlir.contains("stablehlo.tanh"));
        assert!(mlir.contains("chlo.erf"));
        assert!(mlir.contains("-> tensor<729x1024xf32>"));
        assert!(!mlir.contains("-> tensor<1x729x1024xf32>"));
        assert!(mlir.contains("return "));
    }

    #[test]
    fn gemma3_graph_avg_pools_norms_and_projects_without_llava_mlp() {
        let config = LlavaVisionConfig::from_json_str(
            &serde_json::json!({
                "model_type": "gemma3",
                "image_token_index": 99,
                "mm_tokens_per_image": 1,
                "vision_config": {
                    "model_type": "siglip_vision_model",
                    "image_size": 28,
                    "patch_size": 14,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2
                },
                "text_config": {"hidden_size": 12}
            })
            .to_string(),
        )
        .unwrap();
        let mlir = emit_vision(&config);
        assert!(mlir.contains("vision_tower.vision_model.post_layernorm.weight"));
        assert!(mlir.contains("vision_tower.vision_model.post_layernorm.bias"));
        assert!(mlir.contains("multi_modal_projector.mm_soft_emb_norm.weight"));
        assert!(mlir.contains("multi_modal_projector.mm_input_projection_weight"));
        assert!(!mlir.contains("multi_modal_projector.linear_1"));
        assert!(mlir.contains("-> tensor<1x12xf32>"));
    }

    #[cfg(feature = "iree")]
    #[test]
    fn pinned_siglip_graph_compiles_for_cpu() {
        let Some(config) = pinned() else {
            return;
        };
        let mlir = emit_vision(&config);
        let compiler = crate::iree::iree_compile_bin().unwrap();
        let cache = std::env::temp_dir().join("mlxcel-xla-vision-emitter-test");
        std::fs::create_dir_all(&cache).unwrap();
        let vmfb = crate::iree::compile_one(
            &compiler,
            &mlir,
            crate::iree::target_flags("local-task").unwrap(),
            &cache,
            "pinned-llava-vision",
            0,
        )
        .unwrap();
        assert!(vmfb.metadata().unwrap().len() > 0);
    }
}

impl Args {
    fn new(specs: &[VisionWeightSpec]) -> Self {
        let mut values = Vec::with_capacity(specs.len() + 1);
        let mut declarations = Vec::with_capacity(specs.len() + 1);
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

fn gemma_rms_norm(builder: &mut Builder, value: &Val, weight: &Val, epsilon: f32) -> Val {
    let rows = value.ty.shape[0];
    let width = value.ty.shape[1];
    let zero = builder.const_f32(0.0);
    let squared = builder.multiply(value, value);
    let squared_sum = builder.reduce_add(&squared, 1, &zero);
    let width_scalar = builder.const_f32(width as f32);
    let width_rows = builder.broadcast(&width_scalar, &[], vec![rows]);
    let mean_square = builder.divide(&squared_sum, &width_rows);
    let epsilon = builder.const_f32(epsilon);
    let epsilon = builder.broadcast(&epsilon, &[], vec![rows]);
    let mean_square = builder.add(&mean_square, &epsilon);
    let inv_rms = builder.rsqrt(&mean_square);
    let inv_rms = builder.broadcast(&inv_rms, &[0], vec![rows, width]);
    let normalized = builder.multiply(value, &inv_rms);
    let one = builder.const_f32(1.0);
    let one = builder.broadcast(&one, &[], vec![width]);
    let weight = builder.add(weight, &one);
    let weight = builder.broadcast(&weight, &[1], vec![rows, width]);
    builder.multiply(&normalized, &weight)
}

fn gemma3_project(
    builder: &mut Builder,
    hidden: &Val,
    args: &mut Args,
    config: &LlavaVisionConfig,
    tokens_per_side: usize,
    kernel_size: usize,
) -> Val {
    let pooled = builder.reshape(
        hidden,
        vec![
            tokens_per_side,
            kernel_size,
            tokens_per_side,
            kernel_size,
            config.hidden,
        ],
    );
    let zero = builder.const_f32(0.0);
    let pooled = builder.reduce_add(&pooled, 3, &zero);
    let pooled = builder.reduce_add(&pooled, 1, &zero);
    let divisor = builder.const_f32((kernel_size * kernel_size) as f32);
    let divisor = builder.broadcast(
        &divisor,
        &[],
        vec![tokens_per_side, tokens_per_side, config.hidden],
    );
    let pooled = builder.divide(&pooled, &divisor);
    let pooled = builder.reshape(
        &pooled,
        vec![tokens_per_side * tokens_per_side, config.hidden],
    );
    let normalized = gemma_rms_norm(builder, &pooled, &args.take(), config.layer_norm_eps);
    let projection = args.take();
    builder.dot_general(
        &normalized,
        &projection,
        &[],
        &[],
        &[1],
        &[0],
        vec![tokens_per_side * tokens_per_side, config.text_hidden],
    )
}

fn activate(builder: &mut Builder, value: &Val, activation: VisionActivation) -> Val {
    match activation {
        VisionActivation::ExactGelu => exact_gelu(builder, value),
        VisionActivation::GeluPytorchTanh => tanh_gelu(builder, value),
    }
}

fn attention(
    builder: &mut Builder,
    hidden: &Val,
    args: &mut Args,
    config: &LlavaVisionConfig,
) -> AttentionValues {
    let q = linear_2d(builder, hidden, &args.take(), &args.take());
    let k = linear_2d(builder, hidden, &args.take(), &args.take());
    let v = linear_2d(builder, hidden, &args.take(), &args.take());
    let projected_q = q.clone();
    let projected_k = k.clone();
    let projected_v = v.clone();
    let tokens = hidden.ty.shape[0];
    let head_dim = config.hidden / config.heads;
    let q = builder.reshape(&q, vec![tokens, config.heads, head_dim]);
    let q = builder.transpose(&q, &[1, 0, 2]);
    let k = builder.reshape(&k, vec![tokens, config.heads, head_dim]);
    let k = builder.transpose(&k, &[1, 0, 2]);
    let v = builder.reshape(&v, vec![tokens, config.heads, head_dim]);
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
    let probabilities = stable_softmax(builder, &scores, 2);
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
    let output = linear_2d(builder, &context, &args.take(), &args.take());
    AttentionValues {
        q: projected_q,
        k: projected_k,
        v: projected_v,
        context,
        output,
    }
}

struct AttentionValues {
    q: Val,
    k: Val,
    v: Val,
    context: Val,
    output: Val,
}

struct EncoderLayerValues {
    norm1: Val,
    attention: AttentionValues,
    attention_residual: Val,
    norm2: Val,
    mlp_fc1: Val,
    mlp_activation: Val,
    mlp_fc2: Val,
    output: Val,
}

fn encoder_layer(
    builder: &mut Builder,
    hidden: &Val,
    args: &mut Args,
    config: &LlavaVisionConfig,
) -> EncoderLayerValues {
    let norm1 = layer_norm(
        builder,
        hidden,
        &args.take(),
        &args.take(),
        config.layer_norm_eps,
    );
    let attention = attention(builder, &norm1, args, config);
    let attention_residual = builder.add(hidden, &attention.output);
    let norm2 = layer_norm(
        builder,
        &attention_residual,
        &args.take(),
        &args.take(),
        config.layer_norm_eps,
    );
    let mlp_fc1 = linear_2d(builder, &norm2, &args.take(), &args.take());
    let mlp_activation = activate(builder, &mlp_fc1, config.activation);
    let mlp_fc2 = linear_2d(builder, &mlp_activation, &args.take(), &args.take());
    let output = builder.add(&attention_residual, &mlp_fc2);
    EncoderLayerValues {
        norm1,
        attention,
        attention_residual,
        norm2,
        mlp_fc1,
        mlp_activation,
        mlp_fc2,
        output,
    }
}

fn emit_vision_impl(config: &LlavaVisionConfig, diagnostics: bool) -> String {
    let specs = config.weight_specs();
    let mut args = Args::new(&specs);
    let mut builder = Builder::new();
    let patch_weight = args.take();
    let patch_bias = args.take();
    let class_embedding = config.class_token.then(|| args.take());
    let position_embedding = args.take();
    let pre_layer_norm = config.class_token.then(|| (args.take(), args.take()));
    let pixels = args.push_input(
        Ty::f32(vec![
            1,
            config.channels,
            config.image_size,
            config.image_size,
        ]),
        "pixels.nchw",
    );
    let patches = builder.convolution_nchw(
        &pixels,
        &patch_weight,
        [config.patch_size, config.patch_size],
    );
    let patch_bias = builder.broadcast(
        &patch_bias,
        &[1],
        vec![1, config.hidden, config.patch_grid(), config.patch_grid()],
    );
    let patches = builder.add(&patches, &patch_bias);
    let patches = builder.transpose(&patches, &[0, 2, 3, 1]);
    let patches = builder.reshape(&patches, vec![config.patch_grid().pow(2), config.hidden]);
    let mut hidden = if let Some(class_embedding) = class_embedding {
        let class_embedding = builder.reshape(&class_embedding, vec![1, config.hidden]);
        builder.concatenate(&class_embedding, &patches, 0)
    } else {
        patches
    };
    hidden = builder.add(&hidden, &position_embedding);
    if let Some((weight, bias)) = pre_layer_norm {
        hidden = layer_norm(&mut builder, &hidden, &weight, &bias, config.layer_norm_eps);
    }
    let mut diagnostic_values = diagnostics.then(|| vec![hidden.clone()]);
    for layer in 0..=config.feature_layer {
        let values = encoder_layer(&mut builder, &hidden, &mut args, config);
        if layer == 0
            && let Some(outputs) = &mut diagnostic_values
        {
            outputs.extend([
                values.norm1.clone(),
                values.attention.q.clone(),
                values.attention.k.clone(),
                values.attention.v.clone(),
                values.attention.context.clone(),
                values.attention.output.clone(),
                values.attention_residual.clone(),
                values.norm2.clone(),
                values.mlp_fc1.clone(),
                values.mlp_activation.clone(),
                values.mlp_fc2.clone(),
                values.output.clone(),
            ]);
        }
        hidden = values.output;
        if let Some(outputs) = &mut diagnostic_values {
            outputs.push(hidden.clone());
        }
    }
    if matches!(config.projector, VisionProjector::Gemma3AvgPool { .. }) {
        hidden = layer_norm(
            &mut builder,
            &hidden,
            &args.take(),
            &args.take(),
            config.layer_norm_eps,
        );
    }
    if config.drop_first_token {
        hidden = builder.slice(&hidden, &[(1, config.position_count()), (0, config.hidden)]);
    }
    let projected = match config.projector {
        VisionProjector::LlavaMlp => {
            let projected = linear_2d(&mut builder, &hidden, &args.take(), &args.take());
            let projected = exact_gelu(&mut builder, &projected);
            linear_2d(&mut builder, &projected, &args.take(), &args.take())
        }
        VisionProjector::Gemma3AvgPool {
            tokens_per_side,
            kernel_size,
            ..
        } => gemma3_project(
            &mut builder,
            &hidden,
            &mut args,
            config,
            tokens_per_side,
            kernel_size,
        ),
    };
    assert_eq!(args.cursor, specs.len(), "vision weight schema drifted");
    let outputs = if let Some(mut values) = diagnostic_values {
        values.push(hidden);
        values.push(projected);
        values
    } else {
        vec![projected]
    };
    let output_types = outputs
        .iter()
        .map(|value| value.ty.render())
        .collect::<Vec<_>>();
    let result_type = if output_types.len() == 1 {
        output_types[0].clone()
    } else {
        format!("({})", output_types.join(", "))
    };
    let result_values = outputs
        .iter()
        .map(|value| value.name.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "module @vision {{\n  func.func public @main({signature}) -> {result_type} {{\n{body}    \
         return {result_values} : {return_types}\n  }}\n}}\n",
        signature = args.declarations.join(", "),
        body = builder.body(),
        return_types = output_types.join(", "),
    )
}

pub(crate) fn emit_vision(config: &LlavaVisionConfig) -> String {
    emit_vision_impl(config, false)
}

#[cfg(any(test, feature = "diagnostics"))]
pub(crate) fn emit_vision_diagnostics(config: &LlavaVisionConfig) -> String {
    emit_vision_impl(config, true)
}
