// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
// SPDX-License-Identifier: Apache-2.0
use super::config::{Config, MixerKind};
use super::{emit_decode, emit_decode_ragged, emit_prefill};
const TINY: &str = include_str!("../../assets/falcon_h1/config.json");

#[test]
fn falcon_h1_dual_state_and_explicit_independent_widths() {
    let c = Config::from_json_str(TINY).unwrap();
    assert_eq!(c.layer_mixers, [MixerKind::ParallelMamba2Attention; 2]);
    assert_eq!(c.attn_layer_count(), 2);
    for li in 0..2 {
        assert_eq!(c.attn_layer_index(li), Some(li));
        assert_eq!(c.state_layer_index(li), Some(li));
        assert!(c.layer_has_ffn(li));
    }
    assert_eq!(c.head_dim, 4);
    assert_eq!(c.ssm.as_ref().unwrap().head_dim, 6);
    assert_eq!(c.ssm.as_ref().unwrap().d_inner(), 24);
    assert_eq!(c.inter, 24);
    assert_eq!(c.state_tensors()[0].dims, [2, 4, 6, 4]);
    assert_eq!(c.state_tensors()[1].dims, [2, 3, 40]);
    assert!(c.pins_f32_precision());
    assert!(c.check_ssm_norm_grouping().is_ok());
    assert_eq!(crate::config_json_has_recurrent_state(TINY), Ok(true));
    assert!(
        !c.parallel_block,
        "Cohere's attention+MLP topology is distinct"
    );
    assert_eq!(c.embedding_multiplier, Some(0.75));
    assert_eq!(
        c.logit_mul,
        Some(0.625),
        "raw tied head must not divide by embedding scale"
    );
}

#[test]
fn falcon_h1_all_graphs_take_both_mixers_and_second_norm() {
    let mut c = Config::from_json_str(TINY).unwrap();
    c.context_capacity = 32;
    let names = crate::weight_names::weight_names(&c);
    assert_eq!(names.len(), 2 + 16 * 2);
    assert!(names.contains(&"model.final_layernorm.weight".into()));
    assert!(!names.iter().any(|x| x.contains("mamba.norm")));
    for li in 0..2 {
        for suffix in [
            "pre_ff_layernorm.weight",
            "feed_forward.gate_proj.weight",
            "self_attn.k_proj.weight",
            "mamba.in_proj.weight",
        ] {
            assert!(names.contains(&format!("model.layers.{li}.{suffix}")));
        }
    }
    for graph in [
        emit_decode(&c, false),
        emit_prefill(&c, false),
        emit_decode_ragged(&c, 4, false).unwrap(),
    ] {
        for li in 0..2 {
            for name in ["ssm.in_proj", "wq", "post_ln", "gate"] {
                assert!(graph.contains(&format!("['layers'][{li}]['{name}']")));
            }
        }
        assert!(graph.contains("ssm_h") && graph.contains("ssm_conv"));
        assert!(
            graph.contains("2x32x2x4xf32>"),
            "KV rows exist in the graph signature"
        );
    }
}

#[test]
fn falcon_h1_optional_norm_bias_and_auto_head_dimension() {
    let mut v: serde_json::Value = serde_json::from_str(TINY).unwrap();
    v["mamba_d_head"] = serde_json::json!("auto");
    v["mamba_rms_norm"] = serde_json::json!(true);
    v["mamba_proj_bias"] = serde_json::json!(true);
    let c = Config::from_json_str(&v.to_string()).unwrap();
    assert_eq!(c.ssm.as_ref().unwrap().head_dim, 6);
    let names = crate::weight_names::weight_names(&c);
    assert!(names.contains(&"model.layers.0.mamba.norm.weight".into()));
    assert!(names.contains(&"model.layers.0.mamba.in_proj.bias".into()));
    assert!(!names.contains(&"model.layers.0.mamba.out_proj.bias".into()));
    let after = emit_decode(&c, false);
    v["mamba_norm_before_gate"] = serde_json::json!(true);
    let before = emit_decode(&Config::from_json_str(&v.to_string()).unwrap(), false);
    assert_ne!(
        before, after,
        "gating order changes the emitted computation"
    );
    v["mamba_rms_norm"] = serde_json::json!(false);
    let before = emit_decode(&Config::from_json_str(&v.to_string()).unwrap(), false);
    v["mamba_norm_before_gate"] = serde_json::json!(false);
    let after = emit_decode(&Config::from_json_str(&v.to_string()).unwrap(), false);
    assert_eq!(before, after, "without a norm, gating always occurs once");
}

#[test]
fn falcon_h1_rejects_invalid_dimensions_and_factors() {
    for (key, value) in [
        ("mamba_d_head", serde_json::json!(4)),
        ("mamba_n_groups", serde_json::json!(0)),
        ("mamba_n_heads", serde_json::json!(0)),
        ("ssm_multipliers", serde_json::json!([1, 2])),
        ("mlp_multipliers", serde_json::json!([1, 2, 3])),
        ("key_multiplier", serde_json::json!("Infinity")),
        ("rope_theta", serde_json::json!(-1)),
    ] {
        let mut v: serde_json::Value = serde_json::from_str(TINY).unwrap();
        v[key] = value;
        assert!(Config::from_json_str(&v.to_string()).is_err(), "{key}");
    }
}

#[test]
#[ignore = "requires the downloaded raw-HF Falcon-H1 checkpoint"]
fn falcon_h1_original_checkpoint_passes_shape_validation() {
    let path = std::env::var("MLXCEL_FALCON_H1_MODEL").expect("MLXCEL_FALCON_H1_MODEL");
    let c = Config::from_json(std::path::Path::new(&path)).unwrap();
    assert!(c.has_kv_cache() && c.has_recurrent_state());
}
#[test]
fn freeze_falcon_h1_goldens() {
    if std::env::var("MLXCEL_FREEZE_FALCON_H1").as_deref() != Ok("1") {
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
                .join("assets/falcon_h1")
                .join(name),
            &graph[0].1,
        )
        .unwrap();
    }
}
