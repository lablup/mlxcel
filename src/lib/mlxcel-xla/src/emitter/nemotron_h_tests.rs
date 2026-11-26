// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
// SPDX-License-Identifier: Apache-2.0

use super::config::{Config, DenseMlpActivation, ExpertAct, MixerKind, MoeScoring};
use super::{emit_decode, emit_decode_ragged, emit_prefill};

const TINY: &str = include_str!("../../assets/nemotron_h/config.json");

fn tiny_moe_json() -> serde_json::Value {
    let mut v: serde_json::Value = serde_json::from_str(TINY).unwrap();
    v["hybrid_override_pattern"] = serde_json::json!("M*EM-");
    v["n_routed_experts"] = serde_json::json!(8);
    v["num_experts_per_tok"] = serde_json::json!(2);
    v["moe_intermediate_size"] = serde_json::json!(12);
    v["n_group"] = serde_json::json!(2);
    v["topk_group"] = serde_json::json!(1);
    v["norm_topk_prob"] = serde_json::json!(true);
    v["routed_scaling_factor"] = serde_json::json!(1.25);
    v["n_shared_experts"] = serde_json::json!(1);
    v["moe_shared_expert_intermediate_size"] = serde_json::json!(20);
    v
}

fn write_shape_checkpoint(path: &std::path::Path, shapes: &[(String, Vec<usize>)]) {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for (name, shape) in shapes {
        let bytes = 4 * shape.iter().product::<usize>();
        header.insert(
            name.clone(),
            serde_json::json!({
                "dtype": "F32", "shape": shape, "data_offsets": [offset, offset + bytes]
            }),
        );
        offset += bytes;
    }
    let text = serde_json::Value::Object(header).to_string();
    let mut bytes = (text.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(text.as_bytes());
    bytes.resize(bytes.len() + offset, 0);
    std::fs::write(path, bytes).unwrap();
}

fn tiny_moe_checkpoint_shapes() -> Vec<(String, Vec<usize>)> {
    let mut shapes = vec![
        ("backbone.embeddings.weight".into(), vec![32, 16]),
        ("backbone.norm_f.weight".into(), vec![16]),
    ];
    for li in 0..5 {
        let p = format!("backbone.layers.{li}.");
        let m = format!("{p}mixer.");
        shapes.push((format!("{p}norm.weight"), vec![16]));
        match li {
            0 | 3 => {
                shapes.extend([
                    (format!("{m}in_proj.weight"), vec![100, 16]),
                    (format!("{m}out_proj.weight"), vec![16, 32]),
                    (format!("{m}conv1d.weight"), vec![64, 1, 4]),
                    (format!("{m}conv1d.bias"), vec![64]),
                    (format!("{m}dt_bias"), vec![4]),
                    (format!("{m}A_log"), vec![4]),
                    (format!("{m}D"), vec![4]),
                    (format!("{m}norm.weight"), vec![32]),
                ]);
            }
            1 => {
                shapes.extend([
                    (format!("{m}q_proj.weight"), vec![32, 16]),
                    (format!("{m}k_proj.weight"), vec![16, 16]),
                    (format!("{m}v_proj.weight"), vec![16, 16]),
                    (format!("{m}o_proj.weight"), vec![16, 32]),
                ]);
            }
            2 => {
                shapes.extend([
                    (format!("{m}gate.weight"), vec![8, 16]),
                    (format!("{m}gate.e_score_correction_bias"), vec![8]),
                    (format!("{m}switch_mlp.fc1.weight"), vec![8, 12, 16]),
                    (format!("{m}switch_mlp.fc2.weight"), vec![8, 16, 12]),
                    (format!("{m}shared_experts.up_proj.weight"), vec![20, 16]),
                    (format!("{m}shared_experts.down_proj.weight"), vec![16, 20]),
                ]);
            }
            4 => {
                shapes.extend([
                    (format!("{m}up_proj.weight"), vec![32, 16]),
                    (format!("{m}down_proj.weight"), vec![16, 32]),
                ]);
            }
            _ => unreachable!(),
        }
    }
    shapes
}

#[test]
fn nemotron_h_schedule_and_independent_widths() {
    let c = Config::from_json_str(TINY).unwrap();
    assert_eq!(
        c.layer_mixers,
        [
            MixerKind::Mamba2,
            MixerKind::Attention,
            MixerKind::None,
            MixerKind::Mamba2,
            MixerKind::None
        ]
    );
    assert_eq!(c.attn_layer_count(), 1);
    assert_eq!(
        (0..5).map(|li| c.attn_layer_index(li)).collect::<Vec<_>>(),
        [None, Some(0), None, None, None]
    );
    assert_eq!(
        (0..5).map(|li| c.state_layer_index(li)).collect::<Vec<_>>(),
        [Some(0), None, None, Some(1), None]
    );
    assert_eq!(
        (0..5).map(|li| c.layer_has_ffn(li)).collect::<Vec<_>>(),
        [false, false, true, false, true]
    );
    assert!((0..5).all(|li| !c.layer_uses_rope(li)));
    assert_eq!(c.head_dim, 8); // 4 * 8 != hidden 16
    assert_eq!(c.dense_mlp_activation, DenseMlpActivation::Relu2);
    assert_eq!(c.ssm.as_ref().unwrap().norm_group_size, 16);
    assert!(c.check_ssm_norm_grouping().is_ok());
    assert!(c.pins_f32_precision());
    assert_eq!(crate::config_json_has_recurrent_state(TINY), Ok(true));
    assert_eq!(c.state_tensors()[0].dims, [2, 4, 8, 8]);
    assert_eq!(c.state_tensors()[1].dims, [2, 3, 64]);
}

#[test]
fn nemotron_h_preserves_checkpoint_chunk_size() {
    let mut v: serde_json::Value = serde_json::from_str(TINY).unwrap();
    v["chunk_size"] = serde_json::json!(256);
    assert_eq!(
        Config::from_json_str(&v.to_string())
            .unwrap()
            .ssm
            .unwrap()
            .chunk_size,
        256
    );
    v["chunk_size"] = serde_json::json!(0);
    assert!(
        Config::from_json_str(&v.to_string())
            .unwrap_err()
            .contains("chunk_size")
    );
    v["chunk_size"] = serde_json::json!(-1);
    assert!(
        Config::from_json_str(&v.to_string())
            .unwrap_err()
            .contains("chunk_size")
    );
}

#[test]
fn nemotron_h_block_aliases_and_rejections() {
    let original: serde_json::Value = serde_json::from_str(TINY).unwrap();
    for key in ["layers_block_type", "layer_types"] {
        let mut v = original.clone();
        v.as_object_mut().unwrap().remove("hybrid_override_pattern");
        v.as_object_mut().unwrap().remove("num_hidden_layers");
        v[key] = serde_json::json!(["conv", "full_attention", "mlp", "linear_attention", "mlp"]);
        assert_eq!(
            Config::from_json_str(&v.to_string()).unwrap().layer_mixers,
            Config::from_json_str(TINY).unwrap().layer_mixers
        );
    }
    let mut v = original.clone();
    v["layer_types"] = serde_json::json!(["moe"]);
    assert_eq!(
        Config::from_json_str(&v.to_string()).unwrap().n_layers,
        5,
        "override wins"
    );
    v["hybrid_override_pattern"] = serde_json::json!(["M", "*", "-", "M", "-"]);
    assert_eq!(Config::from_json_str(&v.to_string()).unwrap().n_layers, 5);
    for (key, value, message) in [
        (
            "hybrid_override_pattern",
            serde_json::json!("M?"),
            "block code",
        ),
        (
            "hybrid_override_pattern",
            serde_json::json!(""),
            "at least one",
        ),
        ("n_groups", serde_json::json!(0), "positive"),
        ("mlp_hidden_act", serde_json::json!("silu"), "relu2"),
        (
            "num_key_value_heads",
            serde_json::json!(3),
            "attention dimensions",
        ),
    ] {
        let mut v = original.clone();
        v[key] = value;
        assert!(
            Config::from_json_str(&v.to_string())
                .unwrap_err()
                .contains(message)
        );
    }
}

#[test]
fn nemotron_h_moe_schedule_router_and_weight_schema() {
    let v = tiny_moe_json();
    let c = Config::from_json_str(&v.to_string()).unwrap();
    assert_eq!(
        c.layer_mixers,
        [
            MixerKind::Mamba2,
            MixerKind::Attention,
            MixerKind::None,
            MixerKind::Mamba2,
            MixerKind::None,
        ]
    );
    assert_eq!(
        (0..c.n_layers)
            .map(|li| c.is_moe_layer(li))
            .collect::<Vec<_>>(),
        [false, false, true, false, false]
    );
    assert_eq!(
        (0..c.n_layers)
            .map(|li| c.layer_has_ffn(li))
            .collect::<Vec<_>>(),
        [false, false, true, false, true]
    );
    let moe = c.moe.as_ref().unwrap();
    assert_eq!(moe.n_experts, 8);
    assert_eq!(moe.top_k, 2);
    assert_eq!(moe.intermediate, 12);
    assert_eq!(moe.expert_act, ExpertAct::Relu2);
    assert_eq!(
        moe.scoring,
        MoeScoring::Sigmoid {
            correction_bias: true,
            n_group: 2,
            topk_group: 1,
            routed_scaling_factor: 1.25,
            norm_topk_prob: true,
        }
    );
    assert_eq!(moe.shared.unwrap().intermediate, 20);

    let names = crate::weight_names::weight_names(&c);
    for expected in [
        "backbone.layers.2.mixer.gate.weight",
        "backbone.layers.2.mixer.gate.e_score_correction_bias",
        "backbone.layers.2.mixer.switch_mlp.fc1.weight",
        "backbone.layers.2.mixer.switch_mlp.fc2.weight",
        "backbone.layers.2.mixer.shared_experts.up_proj.weight",
        "backbone.layers.2.mixer.shared_experts.down_proj.weight",
    ] {
        assert!(
            names.iter().any(|name| name == expected),
            "missing {expected}"
        );
    }
    assert!(
        !names.iter().any(|name| {
            name.contains("layers.2.mixer.switch_mlp.gate_proj")
                || name.contains("layers.2.mixer.shared_experts.gate_proj")
        }),
        "relu2 experts have no gate projection"
    );

    let graph = emit_prefill(&c, false);
    for location in [
        "params['layers'][2]['moe_router']",
        "params['layers'][2]['moe_correction_bias']",
        "params['layers'][2]['moe_up']",
        "params['layers'][2]['moe_down']",
        "params['layers'][2]['moe_shared_up']",
        "params['layers'][2]['moe_shared_down']",
    ] {
        assert!(
            graph.contains(location),
            "missing graph argument {location}"
        );
    }
    assert!(!graph.contains("params['layers'][2]['moe_gate']"));
    assert!(graph.contains("stablehlo.logistic") || graph.contains("stablehlo.exponential"));
    assert!(graph.contains("stablehlo.maximum"), "relu2 maximum");
    assert!(
        graph.contains("applies stablehlo.minimum across dimensions = [1]"),
        "sigmoid routing uses compiler-compatible first-index reductions"
    );
    assert!(
        !graph.contains(":2 = stablehlo.reduce("),
        "sigmoid routing must not emit tuple reductions"
    );
}

#[test]
fn nemotron_h_moe_rejects_invalid_group_and_shared_shapes() {
    for (key, value, message) in [
        ("n_routed_experts", serde_json::json!(7), "divisible"),
        ("n_group", serde_json::json!(8), "at least two"),
        ("topk_group", serde_json::json!(3), "between 1"),
        (
            "num_experts_per_tok",
            serde_json::json!(5),
            "grouped candidate capacity 4",
        ),
        ("moe_latent_size", serde_json::json!(8), "unsupported"),
        (
            "moe_shared_expert_intermediate_size",
            serde_json::json!(0),
            "positive",
        ),
    ] {
        let mut v = tiny_moe_json();
        v[key] = value;
        assert!(
            Config::from_json_str(&v.to_string())
                .unwrap_err()
                .contains(message),
            "{key} should report {message}"
        );
    }
}

#[test]
fn nemotron_h_moe_checkpoint_load_validates_every_expert_tensor() {
    let dir = tempfile::tempdir().unwrap();
    let config = tiny_moe_json();
    std::fs::write(
        dir.path().join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let checkpoint = dir.path().join("model.safetensors");
    let shapes = tiny_moe_checkpoint_shapes();
    write_shape_checkpoint(&checkpoint, &shapes);
    let loaded = Config::from_json(dir.path()).expect("load a complete E-block checkpoint");
    assert!(loaded.is_moe_layer(2));

    for name in [
        "backbone.layers.2.mixer.gate.weight",
        "backbone.layers.2.mixer.gate.e_score_correction_bias",
        "backbone.layers.2.mixer.switch_mlp.fc1.weight",
        "backbone.layers.2.mixer.switch_mlp.fc2.weight",
        "backbone.layers.2.mixer.shared_experts.up_proj.weight",
        "backbone.layers.2.mixer.shared_experts.down_proj.weight",
    ] {
        let mut malformed = shapes.clone();
        let shape = &mut malformed
            .iter_mut()
            .find(|(tensor, _)| tensor == name)
            .unwrap()
            .1;
        shape[0] += 1;
        write_shape_checkpoint(&checkpoint, &malformed);
        let error = Config::from_json(dir.path()).expect_err(name);
        assert!(error.contains(name) && error.contains("shape"), "{error}");
    }
}

#[test]
fn nemotron_h_compact_kv_then_state_in_all_supported_graphs() {
    let mut c = Config::from_json_str(TINY).unwrap();
    c.context_capacity = 32;
    let decode = emit_decode(&c, false);
    assert!(decode.contains("tensor<1x32x2x8xf32>"));
    assert!(decode.contains("@kv_import") && decode.contains("@state_import"));
    let prefill = emit_prefill(&c, false);
    let ragged = emit_decode_ragged(&c, 4, false).unwrap();
    assert!(ragged.contains("tensor<4x1x32x2x8xf32>"));
    let sig = ragged
        .lines()
        .find(|line| line.contains("func.func public @main"))
        .unwrap();
    assert!(sig.find("loc(\"kcache\")").unwrap() < sig.find("loc(\"ssm_h\")").unwrap());
    for graph in [&decode, &prefill, &ragged] {
        assert!(graph.contains("stablehlo.maximum"), "relu2");
        assert!(
            !graph.contains("['layers'][1]['up']"),
            "no FFN on attention block"
        );
        assert!(
            !graph.contains("['layers'][2]['wq']"),
            "no attention on FFN block"
        );
        assert!(!graph.contains("['post_ln']"), "one norm per block");
    }
    let names = crate::weight_names::weight_names(&c);
    assert_eq!(names.len(), 2 + 2 * 9 + 5 + 2 * 3);
    assert!(names.contains(&"backbone.layers.1.mixer.q_proj.weight".into()));
    assert!(names.contains(&"backbone.layers.2.mixer.up_proj.weight".into()));
    assert!(
        !names
            .iter()
            .any(|name| name.contains("unused") || name.contains("mlp."))
    );
}

#[test]
fn freeze_nemotron_h_goldens() {
    if std::env::var("MLXCEL_FREEZE_NEMOTRON_H").as_deref() != Ok("1") {
        return;
    }
    use crate::validation::{GraphKind, emit_graphs};
    for (kind, name) in [
        (GraphKind::Decode { sample: true }, "decode.mlir"),
        (GraphKind::Prefill { sample: true }, "prefill.mlir"),
        (GraphKind::Prefill { sample: false }, "prefill_logits.mlir"),
        (
            GraphKind::DecodeRagged {
                b_max: 4,
                sample: false,
            },
            "decode_ragged_logits_b4.mlir",
        ),
    ] {
        let graph = emit_graphs(TINY, &[kind]).unwrap();
        std::fs::write(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("assets/nemotron_h")
                .join(name),
            &graph[0].1,
        )
        .unwrap();
    }
}
