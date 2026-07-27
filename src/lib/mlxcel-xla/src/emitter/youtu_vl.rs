// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

//! StableHLO Youtu-VL windowed vision tower and built-in patch merger.

use super::builder::{Builder, Ty, Val};
use super::numeric_ops::{exact_gelu, layer_norm_2d, rms_norm_2d, stable_softmax, tanh_gelu};
#[cfg(any(test, feature = "diagnostics"))]
use super::youtu_vl_plan::YoutuVlDiagnosticStage;
use super::youtu_vl_plan::{YOUTU_VL_PATCH_BUCKETS, YoutuVlVisionConfig, YoutuVlWeightSpec};

struct Args {
    values: Vec<Val>,
    declarations: Vec<String>,
    cursor: usize,
}

impl Args {
    fn new(specs: &[YoutuVlWeightSpec]) -> Self {
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

    fn input(&mut self, ty: Ty, name: &str) -> Val {
        let index = self.values.len();
        self.declarations
            .push(format!("%arg{index}: {} loc(\"{name}\")", ty.render()));
        let value = Builder::arg(index, ty);
        self.values.push(value.clone());
        value
    }
}

fn add_bias(builder: &mut Builder, value: &Val, bias: &Val) -> Val {
    let shape = value.ty.shape.clone();
    let bias = builder.broadcast(bias, &[1], shape);
    builder.add(value, &bias)
}

fn round_bf16(builder: &mut Builder, value: &Val) -> Val {
    let rounded = builder.convert(value, "bf16");
    builder.convert(&rounded, "f32")
}

fn linear(builder: &mut Builder, value: &Val, weight: &Val, bias: &Val) -> Val {
    let value = builder.linear_seq(value, weight);
    let value = add_bias(builder, &value, bias);
    round_bf16(builder, &value)
}

fn layer_norm(builder: &mut Builder, value: &Val, weight: &Val, bias: &Val, epsilon: f32) -> Val {
    let normalized = layer_norm_2d(builder, value, weight, bias, epsilon);
    round_bf16(builder, &normalized)
}

fn rms_norm(builder: &mut Builder, value: &Val, weight: &Val, epsilon: f32) -> Val {
    let value = rms_norm_2d(builder, value, weight, epsilon);
    round_bf16(builder, &value)
}

fn gelu_tanh(builder: &mut Builder, value: &Val) -> Val {
    let value = tanh_gelu(builder, value);
    round_bf16(builder, &value)
}

fn gelu_exact(builder: &mut Builder, value: &Val) -> Val {
    let value = exact_gelu(builder, value);
    round_bf16(builder, &value)
}

fn rotate_half(builder: &mut Builder, value: &Val) -> Val {
    let (tokens, heads, width) = (value.ty.shape[0], value.ty.shape[1], value.ty.shape[2]);
    let half = width / 2;
    let first = builder.slice(value, &[(0, tokens), (0, heads), (0, half)]);
    let second = builder.slice(value, &[(0, tokens), (0, heads), (half, width)]);
    let second = builder.negate(&second);
    builder.concatenate(&second, &first, 2)
}

fn apply_rope(builder: &mut Builder, value: &Val, freqs: &Val) -> Val {
    let tokens = value.ty.shape[0];
    let heads = value.ty.shape[1];
    let half = freqs.ty.shape[1];
    let cos = builder.cosine(freqs);
    let sin = builder.sine(freqs);
    let cos = builder.concatenate(&cos, &cos, 1);
    let sin = builder.concatenate(&sin, &sin, 1);
    let cos = builder.broadcast(&cos, &[0, 2], vec![tokens, heads, half * 2]);
    let sin = builder.broadcast(&sin, &[0, 2], vec![tokens, heads, half * 2]);
    let direct = builder.multiply(value, &cos);
    let rotated = rotate_half(builder, value);
    let rotated = builder.multiply(&rotated, &sin);
    let value = builder.add(&direct, &rotated);
    round_bf16(builder, &value)
}

fn softmax(builder: &mut Builder, value: &Val) -> Val {
    let probabilities = stable_softmax(builder, value, 2);
    round_bf16(builder, &probabilities)
}

fn attention(
    builder: &mut Builder,
    hidden: &Val,
    args: &mut Args,
    config: &YoutuVlVisionConfig,
    freqs: &Val,
    bias: &Val,
) -> Val {
    let tokens = hidden.ty.shape[0];
    let head_dim = config.hidden / config.heads;
    let q = linear(builder, hidden, &args.take(), &args.take());
    let k = linear(builder, hidden, &args.take(), &args.take());
    let v = linear(builder, hidden, &args.take(), &args.take());
    let q = builder.reshape(&q, vec![tokens, config.heads, head_dim]);
    let k = builder.reshape(&k, vec![tokens, config.heads, head_dim]);
    let v = builder.reshape(&v, vec![tokens, config.heads, head_dim]);
    let q = apply_rope(builder, &q, freqs);
    let k = apply_rope(builder, &k, freqs);
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
    let scores = round_bf16(builder, &scores);
    let scale = builder.const_f32((head_dim as f32).powf(-0.5));
    let scale = builder.broadcast(&scale, &[], vec![config.heads, tokens, tokens]);
    let scores = builder.multiply(&scores, &scale);
    let scores = round_bf16(builder, &scores);
    let bias = builder.broadcast(bias, &[1, 2], vec![config.heads, tokens, tokens]);
    let scores = builder.add(&scores, &bias);
    let scores = round_bf16(builder, &scores);
    let probabilities = softmax(builder, &scores);
    let context = builder.dot_general(
        &probabilities,
        &v,
        &[0],
        &[0],
        &[2],
        &[1],
        vec![config.heads, tokens, head_dim],
    );
    let context = round_bf16(builder, &context);
    let context = builder.transpose(&context, &[1, 0, 2]);
    let context = builder.reshape(&context, vec![tokens, config.hidden]);
    linear(builder, &context, &args.take(), &args.take())
}

fn layer(
    builder: &mut Builder,
    hidden: &Val,
    args: &mut Args,
    config: &YoutuVlVisionConfig,
    freqs: &Val,
    bias: &Val,
) -> Val {
    let normalized = layer_norm(
        builder,
        hidden,
        &args.take(),
        &args.take(),
        config.layer_norm_eps,
    );
    let attention = attention(builder, &normalized, args, config, freqs, bias);
    let residual = builder.add(hidden, &attention);
    let residual = round_bf16(builder, &residual);
    let normalized = layer_norm(
        builder,
        &residual,
        &args.take(),
        &args.take(),
        config.layer_norm_eps,
    );
    let up = linear(builder, &normalized, &args.take(), &args.take());
    let up = gelu_tanh(builder, &up);
    let down = linear(builder, &up, &args.take(), &args.take());
    let residual = builder.add(&residual, &down);
    round_bf16(builder, &residual)
}

struct YoutuVlGraph {
    args: Args,
    builder: Builder,
    patch_projection: Val,
    layer_outputs: Vec<Val>,
    post_layernorm: Val,
    projected: Val,
}

fn build_youtu_vl(config: &YoutuVlVisionConfig, patch_bucket: usize) -> YoutuVlGraph {
    assert!(
        YOUTU_VL_PATCH_BUCKETS.contains(&patch_bucket),
        "unqualified Youtu-VL patch bucket"
    );
    let specs = config.weight_specs();
    let mut args = Args::new(&specs);
    let mut builder = Builder::new();
    let patch_width = config.channels * config.patch_size * config.patch_size;
    let head_dim = config.hidden / config.heads;
    let patches = args.input(
        Ty::f32(vec![patch_bucket, patch_width]),
        "patches.window_order",
    );
    let freqs = args.input(
        Ty::f32(vec![patch_bucket, head_dim / 2]),
        "vision_rope.freqs",
    );
    let window_bias = args.input(
        Ty::f32(vec![patch_bucket, patch_bucket]),
        "window_attention.bias",
    );
    let full_bias = args.input(
        Ty::f32(vec![patch_bucket, patch_bucket]),
        "full_attention.bias",
    );
    let patches = round_bf16(&mut builder, &patches);
    let mut hidden = linear(&mut builder, &patches, &args.take(), &args.take());
    let patch_projection = hidden.clone();
    let mut layer_outputs = Vec::with_capacity(config.depth);
    for layer_index in 0..config.depth {
        let bias = if config.full_attention_layers.contains(&layer_index) {
            &full_bias
        } else {
            &window_bias
        };
        hidden = layer(&mut builder, &hidden, &mut args, config, &freqs, bias);
        layer_outputs.push(hidden.clone());
    }
    hidden = layer_norm(
        &mut builder,
        &hidden,
        &args.take(),
        &args.take(),
        config.layer_norm_eps,
    );
    let post_layernorm = hidden.clone();
    hidden = rms_norm(&mut builder, &hidden, &args.take(), 1e-6);
    let merged_width = config.hidden * config.spatial_merge_size * config.spatial_merge_size;
    let merged = builder.reshape(&hidden, vec![patch_bucket / 4, merged_width]);
    let merged = linear(&mut builder, &merged, &args.take(), &args.take());
    let merged = gelu_exact(&mut builder, &merged);
    let projected = linear(&mut builder, &merged, &args.take(), &args.take());
    assert_eq!(args.cursor, specs.len(), "Youtu-VL weight schema drifted");
    YoutuVlGraph {
        args,
        builder,
        patch_projection,
        layer_outputs,
        post_layernorm,
        projected,
    }
}

pub(crate) fn emit_youtu_vl(config: &YoutuVlVisionConfig, patch_bucket: usize) -> String {
    let graph = build_youtu_vl(config, patch_bucket);
    format!(
        "module @youtu_vl_vision {{\n  func.func public @main({signature}) -> {result_type} {{\n{body}    return {result} : {result_type}\n  }}\n}}\n",
        signature = graph.args.declarations.join(", "),
        result_type = graph.projected.ty.render(),
        body = graph.builder.body(),
        result = graph.projected.name,
    )
}

#[cfg(any(test, feature = "diagnostics"))]
pub(crate) fn emit_youtu_vl_diagnostics(
    config: &YoutuVlVisionConfig,
    patch_bucket: usize,
) -> Result<String, String> {
    let graph = build_youtu_vl(config, patch_bucket);
    let stages = config.diagnostic_stages();
    let outputs = stages
        .iter()
        .map(|stage| match stage {
            YoutuVlDiagnosticStage::PatchProjection => &graph.patch_projection,
            YoutuVlDiagnosticStage::Layer { index, .. } => &graph.layer_outputs[*index],
            YoutuVlDiagnosticStage::PostLayerNorm => &graph.post_layernorm,
            YoutuVlDiagnosticStage::MergerWindowOrder => &graph.projected,
        })
        .collect::<Vec<_>>();
    let return_values = outputs
        .iter()
        .map(|value| value.name.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let return_types = outputs
        .iter()
        .map(|value| value.ty.render())
        .collect::<Vec<_>>()
        .join(", ");
    Ok(format!(
        "module @youtu_vl_vision_diagnostics {{\n  func.func public @main({signature}) -> ({return_types}) {{\n{body}    return {return_values} : {return_types} loc(\"Youtu-VL diagnostic checkpoints\")\n  }}\n}}\n",
        signature = graph.args.declarations.join(", "),
        body = graph.builder.body(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn actual_schedule_config() -> YoutuVlVisionConfig {
        YoutuVlVisionConfig::from_json_str(
            r#"{"model_type":"youtu_vl","hidden_size":2560,"vision_config":{"num_hidden_layers":27,"hidden_size":1152,"intermediate_size":4304,"num_attention_heads":16,"num_channels":3,"patch_size":16,"spatial_merge_size":2,"window_size":256,"fullatt_block_indexes":[7,15,23,26],"out_hidden_size":2560,"num_patches":256}}"#,
        )
        .unwrap()
    }

    #[test]
    fn emits_window_and_full_attention_inputs() {
        let config = YoutuVlVisionConfig::from_json_str(
            r#"{"model_type":"youtu_vl","hidden_size":12,"vision_config":{"num_hidden_layers":2,"hidden_size":8,"intermediate_size":16,"num_attention_heads":2,"num_channels":3,"patch_size":2,"spatial_merge_size":2,"window_size":8,"fullatt_block_indexes":[1],"out_hidden_size":12,"num_patches":256}}"#,
        )
        .unwrap();
        let ir = emit_youtu_vl(&config, 16);
        assert!(ir.contains("window_attention.bias"));
        assert!(ir.contains("full_attention.bias"));
        assert!(ir.contains("merger.mlp.2.weight"));
        assert!(ir.contains("tensor<4x12xf32>"));
        assert!(
            ir.contains("stablehlo.convert"),
            "the BF16 checkpoint carrier tape must remain explicit"
        );
        assert!(
            ir.contains("chlo.erf"),
            "the merger uses PyTorch's exact GELU, unlike the vision MLP"
        );
    }

    #[test]
    fn diagnostic_emitter_exposes_two_layer_oracle_stages() {
        let config = YoutuVlVisionConfig::from_json_str(
            r#"{"model_type":"youtu_vl","hidden_size":12,"vision_config":{"num_hidden_layers":2,"hidden_size":16,"intermediate_size":32,"num_attention_heads":4,"num_channels":3,"patch_size":2,"spatial_merge_size":2,"window_size":12,"fullatt_block_indexes":[1],"out_hidden_size":12,"num_patches":256}}"#,
        )
        .unwrap();
        let ir = emit_youtu_vl_diagnostics(&config, 64).unwrap();
        assert!(ir.contains("module @youtu_vl_vision_diagnostics"));
        assert!(ir.contains(
            "tensor<64x16xf32>, tensor<64x16xf32>, tensor<64x16xf32>, tensor<64x16xf32>, tensor<16x12xf32>"
        ));
        assert!(ir.contains("Youtu-VL diagnostic checkpoints"));
        assert_eq!(
            config.diagnostic_contract_identity(),
            "youtu-vl-vision-diagnostics-v3:module=[patch_projection,layer.0.window,layer.1.full,post_layernorm,merger.window_order]:host=[merger.restored_order]"
        );
    }

    #[test]
    fn diagnostic_emitter_accepts_actual_youtu_vl_schedule() {
        let config = actual_schedule_config();
        let stages = config
            .diagnostic_stages()
            .iter()
            .map(YoutuVlDiagnosticStage::name)
            .collect::<Vec<_>>();
        assert_eq!(
            stages,
            [
                "patch_projection",
                "layer.0.window",
                "layer.7.full",
                "layer.15.full",
                "layer.23.full",
                "layer.26.full",
                "post_layernorm",
                "merger.window_order",
            ]
        );
        let ir = emit_youtu_vl_diagnostics(&config, 16).unwrap();
        let result_signature = stages
            .iter()
            .take(stages.len() - 1)
            .map(|_| "tensor<16x1152xf32>")
            .chain(std::iter::once("tensor<4x2560xf32>"))
            .collect::<Vec<_>>()
            .join(", ");
        assert!(ir.contains(&result_signature));
        assert!(
            config
                .diagnostic_contract_identity()
                .contains("layer.26.full")
        );
    }

    #[test]
    #[ignore = "requires MLXCEL_XLA_IREE_COMPILE pointing at the pinned iree-compile"]
    fn actual_schedule_diagnostic_graph_compiles_with_iree_cpu() {
        let compiler =
            std::env::var("MLXCEL_XLA_IREE_COMPILE").expect("set MLXCEL_XLA_IREE_COMPILE");
        let graph = emit_youtu_vl_diagnostics(&actual_schedule_config(), 16).unwrap();
        let stem = format!("mlxcel-youtu-vl-diagnostics-v3-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stem}.mlir"));
        let output = std::env::temp_dir().join(format!("{stem}.vmfb"));
        std::fs::write(&input, graph).expect("write temporary StableHLO");
        let result = std::process::Command::new(compiler)
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
}
