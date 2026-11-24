// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
// SPDX-License-Identifier: Apache-2.0

use super::config::{Config, DenseMlpActivation, MixerKind};
use super::{emit_decode, emit_decode_ragged, emit_prefill};

const TINY: &str = include_str!("../../assets/nemotron_h/config.json");

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
        ("hybrid_override_pattern", serde_json::json!("ME"), "#3345"),
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
