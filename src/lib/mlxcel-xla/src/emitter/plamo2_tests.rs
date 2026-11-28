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

//! PLaMo 2 config, weight-order, and graph-structure tests (issue #3347).

use std::collections::HashMap;

use super::config::{Config, MixerKind, SsmKind};
use super::plamo2_config::{post_mixer_norm_offset, post_mlp_norm_offset, validate_checkpoint};
use super::{emit_decode, emit_decode_ragged, emit_prefill};

const TINY: &str = include_str!("../../assets/plamo2/config.json");

fn tiny() -> Config {
    Config::from_json_str(TINY).expect("tiny plamo2 parses")
}

fn with(key: &str, value: serde_json::Value) -> Result<Config, String> {
    let mut v: serde_json::Value = serde_json::from_str(TINY).unwrap();
    v[key] = value;
    Config::from_json_str(&v.to_string())
}

#[test]
fn plamo2_schedule_state_and_window() {
    let c = tiny();
    use MixerKind::{Attention, Plamo2Mamba};
    assert_eq!(
        c.layer_mixers,
        [Plamo2Mamba, Attention, Plamo2Mamba, Attention]
    );
    assert_eq!(c.attn_layer_count(), 2);
    assert_eq!(c.attn_layer_index(1), Some(0));
    assert_eq!(c.attn_layer_index(3), Some(1));
    assert_eq!(c.attn_layer_index(0), None);
    assert_eq!(c.state_layer_index(0), Some(0));
    assert_eq!(c.state_layer_index(2), Some(1));
    assert_eq!(c.state_layer_index(1), None);
    assert!(
        (0..4).all(|li| c.layer_has_ffn(li)),
        "every block has an MLP"
    );
    let ssm = c.ssm.as_ref().unwrap();
    assert_eq!(ssm.kind, SsmKind::Plamo2 { dt_rank: 64 });
    assert_eq!(
        (ssm.d_inner(), ssm.conv_dim, ssm.in_proj_dim()),
        (16, 16, 32)
    );
    assert_eq!(ssm.time_step_limit, (0.001, 100.0));
    assert_eq!(c.state_tensors()[0].dims, [2, 4, 4, 4]);
    assert_eq!(c.state_tensors()[1].dims, [2, 3, 16]);
    assert!(c.has_kv_cache() && c.has_recurrent_state() && c.pins_f32_precision());
    assert!(c.tie_word_embeddings, "plamo-2-1b has no lm_head tensor");
    // HF keeps keys of age <= attention_window_size; the emitter's band keeps
    // age < sliding_window, so the band is one wider.
    assert_eq!(c.sliding_window, Some(9));
    assert!(c.is_sliding_layer(1), "layer 1 is local");
    assert!(!c.is_sliding_layer(3), "full_attention_idx = [3] is global");
    assert_eq!(c.rope_local_base, Some(5000.0));
    assert_eq!(
        c.rope,
        super::config::RopeScaling::Plain,
        "HF PLaMo 2 RoPE is unscaled"
    );
    assert!(c.local_rope_layer(1) && !c.local_rope_layer(3));
    assert_eq!(crate::config_json_has_recurrent_state(TINY), Ok(true));
}

#[test]
fn plamo2_offsets_match_the_reference_values() {
    assert_eq!(post_mixer_norm_offset(), 0.2);
    // f32(1 / 5**1.5) computed in double and rounded once, as HF does.
    assert_eq!(post_mlp_norm_offset(), (1.0f64 / 5f64.powf(1.5)) as f32);
}

#[test]
fn plamo2_weight_order_is_lock_step_with_the_checkpoint_names() {
    let c = tiny();
    let names = crate::weight_names::weight_names(&c);
    let p = |li: usize, s: &str| format!("model.layers.layers.{li}.{s}");
    let mlp_and_norms = |li| {
        vec![
            p(li, "mlp.down_proj.weight"),
            p(li, "mlp.gate_up_proj.weight"),
            p(li, "pre_mixer_norm.weight"),
            p(li, "post_mixer_norm.weight"),
            p(li, "mlp.gate_up_proj.weight"),
        ]
    };
    let mut want = vec![
        "model.embed_tokens.weight".to_string(),
        "model.norm.weight".to_string(),
    ];
    for li in 0..4 {
        want.extend(mlp_and_norms(li));
        if li % 2 == 1 {
            for s in [
                "mixer.qkv_proj.weight",
                "mixer.o_proj.weight",
                "mixer.qkv_proj.weight",
                "mixer.qkv_proj.weight",
                "mixer.q_weight",
                "mixer.k_weight",
            ] {
                want.push(p(li, s));
            }
        }
        want.push(p(li, "pre_mlp_norm.weight"));
        want.push(p(li, "post_mlp_norm.weight"));
        if li % 2 == 0 {
            for s in [
                "in_proj.weight",
                "conv1d.weight",
                "bcdt_proj.weight",
                "dt_proj.weight",
                "dt_norm_weight",
                "B_norm_weight",
                "C_norm_weight",
                "A_log",
                "D",
                "dt_bias",
                "out_proj.weight",
            ] {
                want.push(p(li, &format!("mixer.{s}")));
            }
        }
    }
    assert_eq!(names, want);
    // The fused projections are row-sliced: gate before up, then Q, K, V.
    let specs = crate::weights::weight_specs(&c);
    let rows: Vec<(usize, usize)> = specs
        .iter()
        .filter_map(|s| match s {
            crate::weights::WeightSpec::Rows { start, end, .. } => Some((*start, *end)),
            _ => None,
        })
        .take(5)
        .collect();
    // Layer 0 gate, up; layer 1 gate, up, then wk.
    assert_eq!(rows, [(0, 24), (24, 48), (0, 24), (24, 48), (16, 24)]);
}

#[test]
fn plamo2_checkpoint_check_accepts_exact_shapes_and_names_a_mismatch() {
    let c = tiny();
    let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();
    let p = |li: usize, s: &str| format!("model.layers.layers.{li}.{s}");
    shapes.insert("model.embed_tokens.weight".into(), vec![32, 16]);
    shapes.insert("model.norm.weight".into(), vec![16]);
    for li in 0..4 {
        for norm in [
            "pre_mixer_norm",
            "post_mixer_norm",
            "pre_mlp_norm",
            "post_mlp_norm",
        ] {
            shapes.insert(p(li, &format!("{norm}.weight")), vec![16]);
        }
        shapes.insert(p(li, "mlp.gate_up_proj.weight"), vec![48, 16]);
        shapes.insert(p(li, "mlp.down_proj.weight"), vec![16, 24]);
        if li % 2 == 1 {
            shapes.insert(p(li, "mixer.qkv_proj.weight"), vec![32, 16]);
            shapes.insert(p(li, "mixer.o_proj.weight"), vec![16, 16]);
            shapes.insert(p(li, "mixer.q_weight"), vec![4, 4]);
            shapes.insert(p(li, "mixer.k_weight"), vec![2, 4]);
            continue;
        }
        for (s, shape) in [
            ("in_proj.weight", vec![32, 16]),
            ("conv1d.weight", vec![16, 1, 4]),
            ("bcdt_proj.weight", vec![72, 16]),
            ("dt_proj.weight", vec![4, 64]),
            ("dt_norm_weight", vec![64]),
            ("B_norm_weight", vec![4]),
            ("C_norm_weight", vec![4]),
            ("A_log", vec![4]),
            ("D", vec![4]),
            ("dt_bias", vec![4]),
            ("out_proj.weight", vec![16, 16]),
        ] {
            shapes.insert(p(li, &format!("mixer.{s}")), shape);
        }
    }
    validate_checkpoint(&c, &shapes).expect("exact shapes pass");
    let mut bad = shapes.clone();
    bad.insert(p(2, "mixer.conv1d.weight"), vec![48, 1, 4]);
    let error = validate_checkpoint(&c, &bad).unwrap_err();
    assert!(error.contains("conv1d"), "{error}");
    let mut bad = shapes.clone();
    bad.insert(p(1, "mixer.q_weight"), vec![4]);
    assert!(
        validate_checkpoint(&c, &bad)
            .unwrap_err()
            .contains("q_weight")
    );
    let mut missing = shapes;
    missing.remove(&p(0, "mixer.dt_norm_weight"));
    assert!(validate_checkpoint(&c, &missing).is_err());
}

#[test]
fn plamo2_graphs_carry_both_state_kinds_and_head_scaled_qk() {
    let mut c = tiny();
    c.context_capacity = 32;
    for graph in [
        emit_decode(&c, false),
        emit_prefill(&c, false),
        emit_decode_ragged(&c, 4, false).unwrap(),
    ] {
        for (li, name) in [
            (0, "ssm.bcdt_proj"),
            (2, "ssm.dt_norm"),
            (1, "q_norm"),
            (3, "wq"),
        ] {
            assert!(
                graph.contains(&format!("['layers'][{li}]['{name}']")),
                "{li} {name}"
            );
        }
        assert!(
            !graph.contains("['layers'][0]['wq']"),
            "no attention on a Mamba block"
        );
        assert!(!graph.contains("['layers'][0]['q_norm']"));
        assert!(
            graph.contains("['layers'][0]['post_ln']"),
            "post-mixer norm"
        );
        assert!(graph.contains("ssm_h") && graph.contains("ssm_conv"));
        assert!(graph.contains("2x32x2x4xf32>"), "two compact KV rows");
        // The local band constant is W + 1 = 9.
        assert!(graph.contains("constant dense<9> : tensor<i32>"));
    }
}

#[test]
fn plamo2_attention_only_config_has_no_recurrent_state() {
    let c = with("mamba_enabled", serde_json::json!(false)).unwrap();
    assert!(c.layer_mixers.iter().all(|m| *m == MixerKind::Attention));
    assert!(c.ssm.is_none() && !c.has_recurrent_state());
    assert!(
        c.pins_f32_precision(),
        "PLaMo 2 Q/K norms are authored in f32"
    );
    let names = crate::weight_names::weight_names(&c);
    assert!(!names.iter().any(|n| n.contains("in_proj")));
}

#[test]
fn plamo2_rejects_unsupported_variants() {
    for (key, value) in [
        ("mamba_step", serde_json::json!(1)),
        ("n_expert", serde_json::json!(8)),
        ("full_attention_idx", serde_json::json!([4])),
        ("full_attention_idx", serde_json::json!(3)),
        (
            "quantization",
            serde_json::json!({"bits": 4, "group_size": 64}),
        ),
        ("rope_theta", serde_json::json!(-1)),
        ("attention_window_size", serde_json::json!(0)),
        ("use_predefined_initial_state", serde_json::json!(true)),
    ] {
        assert!(with(key, value).is_err(), "{key}");
    }
}

#[test]
#[ignore = "requires the downloaded pfnet/plamo-2-1b checkpoint"]
fn plamo2_original_checkpoint_passes_shape_validation() {
    let path = std::env::var("MLXCEL_PLAMO2_MODEL").expect("MLXCEL_PLAMO2_MODEL");
    let c = Config::from_json(std::path::Path::new(&path)).unwrap();
    assert_eq!(c.attn_layer_count(), 8);
    assert_eq!(c.recurrent_layer_count(), 8);
    assert_eq!(c.sliding_window, Some(2049));
    assert!((0..16).all(|li| c.layer_mixer(li) != MixerKind::Attention || c.is_sliding_layer(li)));
}

#[test]
fn freeze_plamo2_goldens() {
    if std::env::var("MLXCEL_FREEZE_PLAMO2").as_deref() != Ok("1") {
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
                .join("assets/plamo2")
                .join(name),
            &graph[0].1,
        )
        .unwrap();
    }
}
