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

//! Mamba2 config, weight order, and state-threaded graph checks (issue #3338).
//! The numeric gate against HF is `spike/openxla/ssm_model_check.py`.

use super::builder::Precision;
use super::config::{Config, MixerKind, SsmConfig, StateElem, StateTensorSpec, WeightScheme};
use super::model::{
    emit_decode, emit_decode_batched, emit_decode_ragged, emit_decode_with, emit_prefill,
    emit_prefill_embeddings, emit_prefill_embeddings_deepstack_with, emit_prefill_with,
};
use super::recurrent_config::quote_non_finite_json;
use crate::weights::{WeightSpec, weight_specs_q};

/// A two-layer Mamba2 with `n_groups = 2 < num_heads`, written the way HF
/// serializes it (a bare `Infinity` in `time_step_limit`).
const TINY_MAMBA2: &str = r#"{
  "model_type": "mamba2", "hidden_size": 16, "num_hidden_layers": 2,
  "vocab_size": 40, "num_heads": 4, "head_dim": 8, "state_size": 16,
  "n_groups": 2, "conv_kernel": 4, "expand": 2, "use_conv_bias": true,
  "use_bias": false, "time_step_limit": [0.0, Infinity],
  "layer_norm_epsilon": 1e-05, "chunk_size": 256, "rms_norm": true,
  "hidden_act": "silu", "tie_word_embeddings": true
}"#;

fn tiny() -> Config {
    Config::from_json_str(TINY_MAMBA2).expect("tiny mamba2 parses")
}

fn main_signature(graph: &str) -> &str {
    graph
        .lines()
        .find(|line| line.contains("func.func public @main"))
        .expect("graph has a main")
}

fn assert_lowerable(graph: &str) {
    for banned in ["scatter", "stablehlo.while", "stablehlo.custom_call"] {
        assert!(!graph.contains(banned), "graph must not emit {banned}");
    }
}

#[test]
fn mamba2_config_parses_ssm_schedule_and_state_abi() {
    let c = tiny();
    let expected = SsmConfig {
        n_heads: 4,
        head_dim: 8,
        state_size: 16,
        n_groups: 2,
        conv_kernel: 4,
        conv_dim: 32 + 2 * 2 * 16,
        use_conv_bias: true,
        use_bias: false,
        time_step_limit: (0.0, f32::INFINITY),
        norm_eps: 1e-5,
        norm_group_size: 32,
        // The spike's chunk, not the checkpoint's training chunk of 256.
        chunk_size: 128,
    };
    assert_eq!(c.ssm.as_ref(), Some(&expected));
    assert_eq!(c.layer_mixers, vec![MixerKind::Mamba2; 2]);
    assert_eq!(c.weight_scheme, WeightScheme::Mamba2);
    assert_eq!((c.hidden, c.n_layers, c.vocab), (16, 2, 40));
    assert_eq!((c.n_q, c.n_kv, c.head_dim, c.group()), (0, 0, 0, 0));
    assert!(!c.has_kv_cache());
    assert!(c.has_recurrent_state());
    assert!(!c.layer_has_ffn(0) && !c.layer_has_ffn(1));
    assert_eq!(c.state_layer_index(0), Some(0));
    assert_eq!(c.state_layer_index(1), Some(1));
    assert_eq!(c.state_layer_index(2), None);
    assert_eq!(
        c.state_tensors(),
        vec![
            StateTensorSpec {
                name: "ssm_h",
                dims: vec![2, 4, 8, 16],
                elem: StateElem::F32,
            },
            StateTensorSpec {
                name: "ssm_conv",
                dims: vec![2, 3, 96],
                elem: StateElem::F32,
            },
        ]
    );
}

#[test]
fn attention_configs_keep_the_kv_cache_and_no_state() {
    let c = Config::llama_3_2_1b();
    assert!(c.has_kv_cache());
    assert!(c.state_tensors().is_empty());
    assert!(!c.has_recurrent_state());
    assert!((0..c.n_layers).all(|li| c.layer_has_ffn(li)));
    assert_eq!(c.state_layer_index(0), None);
    assert!(c.reject_recurrent("ragged decode").is_ok());
    // A literal with more layers than its schedule still reads as attention.
    let deeper = Config {
        n_layers: 40,
        ..Config::llama_3_2_1b()
    };
    assert_eq!(deeper.layer_mixer(39), MixerKind::Attention);
    assert!(deeper.has_kv_cache());
}

#[test]
fn mamba2_config_rejects_inconsistent_dims() {
    let bad = |from: &str, to: &str| {
        let text = TINY_MAMBA2.replace(from, to);
        assert_ne!(text, TINY_MAMBA2, "{from} must occur in the fixture");
        Config::from_json_str(&text).expect_err(to)
    };
    assert!(bad(r#""expand": 2"#, r#""expand": 3"#).contains("expand * hidden_size"));
    assert!(bad(r#""n_groups": 2"#, r#""n_groups": 3"#).contains("n_groups"));
    assert!(bad(r#""rms_norm": true"#, r#""rms_norm": false"#).contains("rms_norm"));
    assert!(bad(r#""hidden_act": "silu""#, r#""hidden_act": "gelu""#).contains("hidden_act"));
    assert!(bad(r#""hidden_size": 16"#, r#""hidden_size": 0"#).contains("must be positive"));
    assert!(bad(r#""num_heads": 4"#, r#""num_heads": 0"#).contains("expand * hidden_size"));
    assert!(
        bad("[0.0, Infinity]", "[0.5, 0.1]").contains("time_step_limit"),
        "an inverted clamp is rejected"
    );
}

#[test]
fn mamba2_config_reads_every_infinity_spelling() {
    for spelling in [
        "Infinity",
        "null",
        r#""Infinity""#,
        r#"{"__float__": "Infinity"}"#,
    ] {
        let text = TINY_MAMBA2.replace("[0.0, Infinity]", &format!("[0.0, {spelling}]"));
        let c = Config::from_json_str(&text).expect(spelling);
        let limit = c.ssm.expect("ssm").time_step_limit;
        assert_eq!(limit, (0.0, f32::INFINITY), "{spelling}");
    }
}

#[test]
fn quote_non_finite_json_leaves_strings_and_finite_text_alone() {
    assert_eq!(quote_non_finite_json(r#"{"a": [1, 2]}"#), None);
    assert_eq!(
        quote_non_finite_json(r#"{"a": "Infinity", "b": [-Infinity, NaN]}"#).as_deref(),
        Some(r#"{"a": "Infinity", "b": ["-Infinity", "NaN"]}"#)
    );
    assert_eq!(
        quote_non_finite_json(r#"{"k\"Infinity": Infinity}"#).as_deref(),
        Some(r#"{"k\"Infinity": "Infinity"}"#)
    );
}

/// The checkpoint tensor each Mamba2 graph argument reads, by its loc suffix.
fn tensor_for_loc(li: usize, key: &str) -> String {
    let p = format!("backbone.layers.{li}.");
    let m = format!("{p}mixer.");
    match key {
        "in_ln" => format!("{p}norm.weight"),
        "ssm.in_proj" => format!("{m}in_proj.weight"),
        "ssm.in_proj_b" => format!("{m}in_proj.bias"),
        "ssm.conv_w" => format!("{m}conv1d.weight"),
        "ssm.conv_b" => format!("{m}conv1d.bias"),
        "ssm.dt_bias" => format!("{m}dt_bias"),
        "ssm.a_log" => format!("{m}A_log"),
        "ssm.d" => format!("{m}D"),
        "ssm.norm_w" => format!("{m}norm.weight"),
        "ssm.out_proj" => format!("{m}out_proj.weight"),
        "ssm.out_proj_b" => format!("{m}out_proj.bias"),
        other => panic!("unexpected layer arg {other}"),
    }
}

/// The weight args of `graph`'s main, mapped to checkpoint tensor names.
fn graph_weight_tensors(graph: &str) -> Vec<String> {
    main_signature(graph)
        .split("loc(\"")
        .skip(1)
        .filter_map(|part| part.split("\")").next())
        .filter_map(|loc| match loc {
            "params['embed']" => Some("backbone.embeddings.weight".to_string()),
            "params['final_norm']" => Some("backbone.norm_f.weight".to_string()),
            "params['lm_head']" => Some("lm_head.weight".to_string()),
            _ => {
                let rest = loc.strip_prefix("params['layers'][")?;
                let (li, key) = rest.split_once("]['")?;
                let key = key.strip_suffix("']")?;
                Some(tensor_for_loc(li.parse().ok()?, key))
            }
        })
        .collect()
}

#[test]
fn weight_specs_follow_take_ssm_layer_weights_order() {
    for (use_bias, tie) in [(false, true), (true, false)] {
        let mut c = tiny();
        c.tie_word_embeddings = tie;
        c.ssm.as_mut().expect("ssm").use_bias = use_bias;
        let specs: Vec<String> = weight_specs_q(&c, false)
            .iter()
            .map(|spec| spec.tensor_name().to_string())
            .collect();
        assert_eq!(specs, graph_weight_tensors(&emit_prefill(&c, false)));
        assert_eq!(specs, graph_weight_tensors(&emit_decode(&c, true)));
        // Projections are `Proj` (f16-resident capable); the rest stay whole.
        let projections = weight_specs_q(&c, false)
            .into_iter()
            .filter(|spec| matches!(spec, WeightSpec::Proj(_)))
            .count();
        assert_eq!(projections, 2 * c.n_layers);
    }
}

#[test]
fn mamba2_prefill_threads_state_after_real_len_and_drops_the_kv_cache() {
    let c = tiny();
    let graph = emit_prefill(&c, true);
    assert_lowerable(&graph);
    let sig = main_signature(&graph);
    let real_len = sig.find("loc(\"real_len\")").expect("real_len arg");
    let h = sig
        .find("tensor<2x4x8x16xf32> loc(\"ssm_h\")")
        .expect("ssm_h arg");
    let conv = sig
        .find("tensor<2x3x96xf32> loc(\"ssm_conv\")")
        .expect("ssm_conv arg");
    assert!(real_len < h && h < conv, "{sig}");
    assert!(
        sig.ends_with("-> (tensor<i32>, tensor<2x4x8x16xf32>, tensor<2x3x96xf32>) {"),
        "{sig}"
    );
    assert!(
        !graph.contains("kcache"),
        "no KV cache for an attention-free model"
    );
    // No RoPE table and no attention mask for a model without attention.
    assert_eq!(
        graph.matches("\"stablehlo.gather\"(").count(),
        1,
        "only the token embedding gathers"
    );
    assert!(graph.contains("stablehlo.dynamic_update_slice"));
}

#[test]
fn mamba2_decode_keeps_state_in_globals_seeded_by_state_import() {
    let c = tiny();
    for precision in [Precision::F32, Precision::F16] {
        let graph = emit_decode_with(&c, true, precision);
        assert_lowerable(&graph);
        assert!(graph.contains("util.global private mutable @ssm_h : tensor<2x4x8x16xf32>"));
        assert!(graph.contains("util.global private mutable @ssm_conv : tensor<2x3x96xf32>"));
        assert!(graph.contains(
            "func.func public @state_import(%s0: tensor<2x4x8x16xf32>, %s1: tensor<2x3x96xf32>)"
        ));
        assert!(graph.contains("util.global.load @ssm_h"));
        assert!(graph.contains("util.global.store %s1, @ssm_conv"));
        for absent in ["@kcache", "@vcache", "@kv_import"] {
            assert!(
                !graph.contains(absent),
                "{absent} in an attention-free decode"
            );
        }
        let sig = main_signature(&graph);
        assert!(!sig.contains("ssm_"), "state never appears in @main: {sig}");
        assert!(sig.ends_with("-> tensor<i32> {"), "{sig}");
        // The recurrence stays f32 even when the projections demote.
        assert!(graph.contains("@ssm_h : tensor<2x4x8x16xf32>"));
    }
}

#[test]
fn graphs_without_a_state_contract_reject_mamba2() {
    let c = tiny();
    let errors = [
        emit_decode_batched(&c, 4, false).expect_err("batched"),
        emit_decode_ragged(&c, 4, false).expect_err("ragged"),
        emit_prefill_embeddings(&c, false).expect_err("embeddings"),
        emit_prefill_embeddings_deepstack_with(&c, false, Precision::F32).expect_err("deepstack"),
    ];
    for error in errors {
        assert!(error.contains("mamba2"), "{error}");
    }
}

#[test]
fn mamba2_prefill_and_decode_emit_at_every_precision() {
    let c = tiny();
    for precision in [Precision::F32, Precision::F16, Precision::Bf16] {
        assert!(emit_prefill_with(&c, false, precision).contains("module @prefill"));
        assert!(emit_decode_with(&c, false, precision).contains("module @decode_step"));
    }
}

#[test]
fn checkpoint_cross_check_bounds_every_ssm_dim() {
    let dir = tempfile::tempdir().expect("tempdir");
    let c = tiny();
    let write = |shapes: &[(String, Vec<usize>)]| {
        let mut header = serde_json::Map::new();
        for (name, shape) in shapes {
            header.insert(
                name.clone(),
                serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [0, 0]}),
            );
        }
        let text = serde_json::Value::Object(header).to_string();
        let mut bytes = (text.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(text.as_bytes());
        std::fs::write(dir.path().join("model.safetensors"), bytes).expect("write");
    };
    let mut shapes: Vec<(String, Vec<usize>)> = vec![
        ("backbone.embeddings.weight".into(), vec![40, 16]),
        ("backbone.norm_f.weight".into(), vec![16]),
    ];
    for li in 0..2 {
        let m = format!("backbone.layers.{li}.mixer.");
        shapes.push((format!("backbone.layers.{li}.norm.weight"), vec![16]));
        shapes.push((format!("{m}in_proj.weight"), vec![32 + 96 + 4, 16]));
        shapes.push((format!("{m}out_proj.weight"), vec![16, 32]));
        shapes.push((format!("{m}conv1d.weight"), vec![96, 1, 4]));
        shapes.push((format!("{m}conv1d.bias"), vec![96]));
        for name in ["dt_bias", "A_log", "D"] {
            shapes.push((format!("{m}{name}"), vec![4]));
        }
        shapes.push((format!("{m}norm.weight"), vec![32]));
    }
    write(&shapes);
    c.validate_recurrent_checkpoint(dir.path())
        .expect("matching checkpoint");

    // The MLX conversion's [C, K, 1] conv layout carries the same bytes.
    let mut mlx = shapes.clone();
    for (name, shape) in &mut mlx {
        if name.ends_with("conv1d.weight") {
            *shape = vec![96, 4, 1];
        }
    }
    write(&mlx);
    c.validate_recurrent_checkpoint(dir.path())
        .expect("MLX conv layout");

    // A config asking for a larger state than the checkpoint holds is refused.
    let mut wider = c.clone();
    let ssm = wider.ssm.as_mut().expect("ssm");
    ssm.state_size = 32;
    ssm.conv_dim = 32 + 2 * 2 * 32;
    write(&shapes);
    let error = wider
        .validate_recurrent_checkpoint(dir.path())
        .expect_err("state_size beyond the checkpoint");
    assert!(error.contains("in_proj.weight"), "{error}");

    // A missing layer is refused, not read as zeros.
    let deeper = Config {
        n_layers: 3,
        layer_mixers: vec![MixerKind::Mamba2; 3],
        ..c.clone()
    };
    let error = deeper
        .validate_recurrent_checkpoint(dir.path())
        .expect_err("missing layer");
    assert!(error.contains("backbone.layers.2"), "{error}");

    // Attention configs never read the checkpoint here.
    Config::llama_3_2_1b()
        .validate_recurrent_checkpoint(dir.path())
        .expect("no-op for attention-only configs");
}
