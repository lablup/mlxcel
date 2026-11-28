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

//! Mamba2/Mamba1 config, weight order, and state-threaded graph checks (issues
//! #3338 and #3348). The numeric gates against HF are the OpenXLA SSM oracles.

use super::builder::Precision;
use super::config::{
    BcdtNorm, Config, MixerKind, QuantConfig, SsmConfig, SsmKind, StateElem, StateTensorSpec,
    WeightScheme,
};
use super::model::{
    emit_decode, emit_decode_batched, emit_decode_with, emit_prefill, emit_prefill_embeddings,
    emit_prefill_embeddings_deepstack_with, emit_prefill_with,
};
use crate::hf_json::quote_non_finite_json;
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

const TINY_MAMBA1: &str = r#"{
  "model_type": "mamba", "hidden_size": 8, "intermediate_size": 16,
  "num_hidden_layers": 2, "vocab_size": 40, "state_size": 4,
  "time_step_rank": 2, "conv_kernel": 4, "use_conv_bias": true,
  "use_bias": false, "tie_word_embeddings": true, "layer_norm_epsilon": 1e-5
}"#;

fn tiny() -> Config {
    Config::from_json_str(TINY_MAMBA2).expect("tiny mamba2 parses")
}

fn tiny_mamba1() -> Config {
    Config::from_json_str(TINY_MAMBA1)
        .expect("tiny mamba1 parses")
        .with_context_capacity(9)
        .expect("small test capacity")
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
        kind: SsmKind::Mamba2,
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
fn mamba1_and_falcon_mamba_parse_diagonal_state_abi() {
    for (model_type, bcdt_norm, norm_eps) in [
        ("mamba", BcdtNorm::None, 1e-5),
        ("falcon_mamba", BcdtNorm::Weightless, 1e-6),
    ] {
        let text = TINY_MAMBA1
            .replace(
                r#""model_type": "mamba""#,
                &format!(r#""model_type": "{model_type}""#),
            )
            .replace(
                r#""layer_norm_epsilon": 1e-5"#,
                r#""layer_norm_epsilon": 1e-5, "mixer_rms_eps": 1e-6"#,
            );
        let c = Config::from_json_str(&text).expect("Mamba1 config parses");
        let ssm = c.ssm.as_ref().expect("Mamba1 carries SSM config");
        assert_eq!(
            ssm.kind,
            SsmKind::Mamba1 {
                dt_rank: 2,
                bcdt_norm,
            }
        );
        assert_eq!(ssm.norm_eps, norm_eps);
        assert_eq!((ssm.d_inner(), ssm.conv_dim), (16, 16));
        assert_eq!(c.layer_mixers, vec![MixerKind::Mamba1; 2]);
        assert_eq!(c.weight_scheme, WeightScheme::Mamba1);
        assert!(!c.has_kv_cache());
        assert_eq!(
            c.state_tensors(),
            vec![
                StateTensorSpec {
                    name: "ssm_h",
                    dims: vec![2, 16, 4],
                    elem: StateElem::F32,
                },
                StateTensorSpec {
                    name: "ssm_conv",
                    dims: vec![2, 3, 16],
                    elem: StateElem::F32,
                },
            ]
        );
    }
}

#[test]
fn mamba1_legacy_bias_aliases_and_canonical_precedence() {
    let aliases = TINY_MAMBA1
        .replace(r#""use_conv_bias": true"#, r#""conv_bias": false"#)
        .replace(r#""use_bias": false"#, r#""bias": true"#);
    let c = Config::from_json_str(&aliases).expect("legacy Mamba1 aliases parse");
    let ssm = c.ssm.as_ref().expect("Mamba1 carries SSM config");
    assert!(!ssm.use_conv_bias);
    assert!(ssm.use_bias);
    let names = weight_specs_q(&c, false)
        .into_iter()
        .map(|spec| spec.tensor_name().to_string())
        .collect::<Vec<_>>();
    assert!(!names.iter().any(|name| name.ends_with("conv1d.bias")));
    assert!(names.iter().any(|name| name.ends_with("in_proj.bias")));
    assert!(names.iter().any(|name| name.ends_with("out_proj.bias")));

    let both = TINY_MAMBA1
        .replace(
            r#""use_conv_bias": true"#,
            r#""conv_bias": false, "use_conv_bias": true"#,
        )
        .replace(r#""use_bias": false"#, r#""bias": true, "use_bias": false"#);
    let c = Config::from_json_str(&both).expect("canonical Mamba1 keys take precedence");
    let ssm = c.ssm.as_ref().expect("Mamba1 carries SSM config");
    assert!(ssm.use_conv_bias);
    assert!(!ssm.use_bias);
}

#[test]
fn mamba1_rejects_unsupported_norms_and_nonpositive_epsilons() {
    for replacement in [
        r#""layer_norm_epsilon": 1e-5, "rms_norm": false"#,
        r#""layer_norm_epsilon": 0"#,
        r#""layer_norm_epsilon": -1"#,
    ] {
        let text = TINY_MAMBA1.replace(r#""layer_norm_epsilon": 1e-5"#, replacement);
        let error = Config::from_json_str(&text).expect_err("invalid Mamba1 norm must fail");
        assert!(
            error.contains("not supported") || error.contains("finite and positive"),
            "{error}"
        );
    }

    let falcon = TINY_MAMBA1
        .replace(
            r#""model_type": "mamba""#,
            r#""model_type": "falcon_mamba""#,
        )
        .replace(
            r#""layer_norm_epsilon": 1e-5"#,
            r#""layer_norm_epsilon": 1e-5, "mixer_rms_eps": 0"#,
        );
    let error = Config::from_json_str(&falcon).expect_err("zero Falcon mixer epsilon must fail");
    assert!(error.contains("mixer_rms_eps") && error.contains("finite and positive"));
}

#[test]
fn mamba1_weight_specs_match_the_hf_mixer_order() {
    let c = tiny_mamba1();
    let names = weight_specs_q(&c, false)
        .into_iter()
        .map(|spec| spec.tensor_name().to_string())
        .collect::<Vec<_>>();
    assert_eq!(
        &names[..11],
        &[
            "backbone.embeddings.weight",
            "backbone.norm_f.weight",
            "backbone.layers.0.norm.weight",
            "backbone.layers.0.mixer.in_proj.weight",
            "backbone.layers.0.mixer.conv1d.weight",
            "backbone.layers.0.mixer.conv1d.bias",
            "backbone.layers.0.mixer.x_proj.weight",
            "backbone.layers.0.mixer.dt_proj.weight",
            "backbone.layers.0.mixer.dt_proj.bias",
            "backbone.layers.0.mixer.A_log",
            "backbone.layers.0.mixer.D",
        ]
    );
    assert_eq!(names[11], "backbone.layers.0.mixer.out_proj.weight");
    assert_eq!(names, graph_weight_tensors(&emit_prefill(&c, false)));
    assert_eq!(names, graph_weight_tensors(&emit_decode(&c, false)));
}

#[test]
fn mamba1_whole_model_graphs_thread_diagonal_state() {
    let c = tiny_mamba1();
    let prefill = emit_prefill(&c, false);
    assert_lowerable(&prefill);
    let sig = main_signature(&prefill);
    assert!(sig.contains("tensor<2x16x4xf32> loc(\"ssm_h\")"), "{sig}");
    assert!(
        sig.contains("tensor<2x3x16xf32> loc(\"ssm_conv\")"),
        "{sig}"
    );
    assert!(
        sig.ends_with("-> (tensor<40xf32>, tensor<2x16x4xf32>, tensor<2x3x16xf32>) {"),
        "{sig}"
    );
    let decode = emit_decode(&c, false);
    assert_lowerable(&decode);
    assert!(decode.contains("util.global private mutable @ssm_h : tensor<2x16x4xf32>"));
    assert!(decode.contains(
        "func.func public @state_import(%s0: tensor<2x16x4xf32>, %s1: tensor<2x3x16xf32>)"
    ));
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

/// The quoted non-finite spelling is a Mamba2-only tolerance. An attention
/// config reads optional floats as `Option`, so a bare `Infinity` or `NaN`
/// quoted into a string would silently take the field's default (here: no
/// QKV clipping) instead of failing like the strict parse did (issue #3339).
#[test]
fn attention_config_refuses_a_quoted_non_finite_number() {
    let dense = |extra: &str| {
        format!(
            r#"{{"model_type":"llama","hidden_size":64,"intermediate_size":128,
            "num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,
            "rms_norm_eps":1e-5,"rope_theta":10000.0,"vocab_size":32{extra}}}"#
        )
    };
    Config::from_json_str(&dense("")).expect("the strict document parses");
    for token in ["Infinity", "-Infinity", "NaN"] {
        let error = Config::from_json_str(&dense(&format!(r#","partial_rotary_factor":{token}"#)))
            .expect_err(token);
        assert!(error.contains("Infinity or NaN"), "{token}: {error}");
    }
    // A string value that merely spells the word is ordinary JSON.
    Config::from_json_str(&dense(r#","note":"Infinity""#)).expect("a string is not a number");
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
        "ssm.x_proj" => format!("{m}x_proj.weight"),
        "ssm.dt_proj" => format!("{m}dt_proj.weight"),
        "ssm.dt_proj_b" => format!("{m}dt_proj.bias"),
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

/// Write a zero-filled F32 safetensors file whose header declares `shapes`
/// with consistent data offsets. A shape too large to back with data (the
/// hostile-dim cases) is declared with empty offsets instead.
pub(super) fn write_safetensors(path: &std::path::Path, shapes: &[(String, Vec<usize>)]) {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for (name, shape) in shapes {
        let bytes = shape
            .iter()
            .try_fold(4usize, |acc, &dim| acc.checked_mul(dim))
            .filter(|&bytes| bytes <= 1 << 20)
            .unwrap_or(0);
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
    std::fs::write(path, bytes).expect("write");
}

/// The tiny config's tensors, two layers, tied head.
fn tiny_checkpoint_shapes() -> Vec<(String, Vec<usize>)> {
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
    shapes
}

/// `num_hidden_layers` sizes `layer_mixers` before any checkpoint is read, so
/// an absurd count is refused at parse time, for Mamba2 and dense configs.
#[test]
fn config_parse_refuses_an_unbounded_layer_count() {
    let max = super::config::MAX_CONFIG_LAYERS;
    let at_cap = TINY_MAMBA2.replace(
        r#""num_hidden_layers": 2"#,
        &format!(r#""num_hidden_layers": {max}"#),
    );
    assert_eq!(
        Config::from_json_str(&at_cap).expect("at the cap").n_layers,
        max
    );
    let over = TINY_MAMBA2.replace(
        r#""num_hidden_layers": 2"#,
        &format!(r#""num_hidden_layers": {}"#, max + 1),
    );
    let error = Config::from_json_str(&over).expect_err("mamba2 over the cap");
    assert!(error.contains("layers"), "{error}");
    let dense = format!(
        r#"{{"model_type":"llama","hidden_size":64,"intermediate_size":128,
            "num_hidden_layers":{},"num_attention_heads":4,"num_key_value_heads":2,
            "rms_norm_eps":1e-5,"rope_theta":10000.0,"vocab_size":32}}"#,
        max + 1
    );
    let error = Config::from_json_str(&dense).expect_err("dense over the cap");
    assert!(error.contains("layers"), "{error}");
}

/// A header alone must not vouch for dims: every declared tensor has to be
/// backed by `product(shape) * dtype` bytes of the file, or a few-KB
/// checkpoint could pass the cross-check with any dims its config asks for.
#[test]
fn checkpoint_cross_check_refuses_headers_larger_than_their_data() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("model.safetensors");
    let c = tiny();
    write_safetensors(&path, &tiny_checkpoint_shapes());
    c.validate_recurrent_checkpoint(dir.path())
        .expect("a backed checkpoint");
    // Truncate the data section: the header now claims more than the file has.
    let full = std::fs::read(&path).expect("read");
    std::fs::write(&path, &full[..full.len() - 4]).expect("truncate");
    let error = c
        .validate_recurrent_checkpoint(dir.path())
        .expect_err("a header larger than its data");
    assert!(error.contains("data_offsets"), "{error}");
    // Empty offsets for a declared shape are refused too.
    let text = String::from_utf8_lossy(&full[8..]).into_owned();
    let start = text.find("\"data_offsets\":[").expect("offsets") + 16;
    let end = start + text[start..].find(']').expect("close");
    let mut bytes = full.clone();
    let zero = format!("{:<width$}", "0,0", width = end - start);
    bytes[8 + start..8 + end].copy_from_slice(zero.as_bytes());
    std::fs::write(&path, &bytes).expect("rewrite");
    let error = c
        .validate_recurrent_checkpoint(dir.path())
        .expect_err("empty offsets");
    assert!(error.contains("data_offsets"), "{error}");
}

/// Shard names from the index go through the loader's plain-file-name rule:
/// `..`, `/` and nested paths are refused before anything is opened.
#[test]
fn checkpoint_cross_check_refuses_index_shards_outside_the_model_dir() {
    let dir = tempfile::tempdir().expect("tempdir");
    let c = tiny();
    for hostile in [
        "..",
        "/",
        "/etc/passwd",
        "../model.safetensors",
        "sub/x.safetensors",
    ] {
        let index = serde_json::json!({
            "weight_map": {"backbone.embeddings.weight": hostile}
        });
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            index.to_string(),
        )
        .expect("write index");
        let error = c
            .validate_recurrent_checkpoint(dir.path())
            .expect_err("hostile shard name");
        assert!(error.contains("plain file names"), "{hostile}: {error}");
    }
}

/// `H*P` and `G*N` are each bounded by the checkpoint, but their product (the
/// per-layer `ssm_h` state) is bounded only by this check: a narrow-hidden
/// checkpoint of a few MB could otherwise ask for gigabytes of state.
#[test]
fn checkpoint_cross_check_bounds_the_state_by_the_in_proj() {
    let dir = tempfile::tempdir().expect("tempdir");
    // hidden 1, d_inner = expand = 64, H = 1, N = 64: in_proj [64+64+128+1, 1]
    // holds 257 elements, the state 64 * 64 = 4096 per layer.
    let text = r#"{
      "model_type": "mamba2", "hidden_size": 1, "num_hidden_layers": 1,
      "vocab_size": 4, "num_heads": 1, "head_dim": 64, "state_size": 64,
      "n_groups": 1, "conv_kernel": 2, "expand": 64, "tie_word_embeddings": true
    }"#;
    let c = Config::from_json_str(text).expect("parses");
    let m = "backbone.layers.0.mixer.";
    let shapes: Vec<(String, Vec<usize>)> = vec![
        ("backbone.embeddings.weight".into(), vec![4, 1]),
        ("backbone.norm_f.weight".into(), vec![1]),
        ("backbone.layers.0.norm.weight".into(), vec![1]),
        (format!("{m}in_proj.weight"), vec![64 + 192 + 1, 1]),
        (format!("{m}out_proj.weight"), vec![1, 64]),
        (format!("{m}conv1d.weight"), vec![192, 1, 2]),
        (format!("{m}conv1d.bias"), vec![192]),
        (format!("{m}dt_bias"), vec![1]),
        (format!("{m}A_log"), vec![1]),
        (format!("{m}D"), vec![1]),
        (format!("{m}norm.weight"), vec![64]),
    ];
    write_safetensors(&dir.path().join("model.safetensors"), &shapes);
    let error = c
        .validate_recurrent_checkpoint(dir.path())
        .expect_err("state wider than the in_proj");
    assert!(error.contains("state"), "{error}");
}

#[test]
fn checkpoint_cross_check_bounds_every_ssm_dim() {
    let dir = tempfile::tempdir().expect("tempdir");
    let c = tiny();
    let write = |shapes: &[(String, Vec<usize>)]| {
        write_safetensors(&dir.path().join("model.safetensors"), shapes);
    };
    let shapes = tiny_checkpoint_shapes();
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

    // An affine-quantized config compares the packed width without
    // overflowing on a hostile header dim: `(2^59 + 2) * 32` wraps to the
    // 4-bit width of hidden 16 (64), which a wrapping multiply accepts.
    let mut huge = shapes.clone();
    for (name, shape) in &mut huge {
        if name.ends_with("in_proj.weight") {
            *shape = vec![32 + 96 + 4, (1usize << 59) + 2];
        }
    }
    write(&huge);
    let quantized = Config {
        quantization: Some(QuantConfig {
            bits: 4,
            group_size: 16,
        }),
        ..c.clone()
    };
    let error = quantized
        .validate_recurrent_checkpoint(dir.path())
        .expect_err("an absurd packed width is refused");
    assert!(error.contains("in_proj.weight"), "{error}");

    // Attention configs never read the checkpoint here.
    Config::llama_3_2_1b()
        .validate_recurrent_checkpoint(dir.path())
        .expect("no-op for attention-only configs");
}

/// HF `Mamba2Config` defaults `tie_word_embeddings` to false, so a config that
/// omits it must read `lm_head.weight` (and refuse a checkpoint without one)
/// rather than silently reuse the embeddings as the head.
#[test]
fn mamba2_omitted_tie_word_embeddings_follows_the_hf_default() {
    let text = TINY_MAMBA2.replace(r#", "tie_word_embeddings": true"#, "");
    assert_ne!(text, TINY_MAMBA2, "the fixture states the key");
    let untied = Config::from_json_str(&text).expect("parses without the key");
    assert!(!untied.tie_word_embeddings);
    assert!(
        weight_specs_q(&untied, false)
            .iter()
            .any(|spec| matches!(spec, WeightSpec::Whole(name) if name == "lm_head.weight")),
        "an untied Mamba2 uploads its own head"
    );
    assert!(tiny().tie_word_embeddings, "an explicit true is kept");
}

/// Issue #3340: an f16 recurrent state drifts from the HF fp32 reference on
/// hip, so a recurrent config pins F32 on every device like an MLA config, and
/// only an explicit `MLXCEL_XLA_PRECISION` replaces the pin.
#[test]
fn recurrent_config_pins_f32_unless_precision_is_explicit() {
    use super::builder::resolve_pinned_precision_with_override as resolve;

    let mamba2 = tiny();
    assert!(mamba2.pins_f32_precision());
    for device in ["hip", "cuda", "metal", "local-task"] {
        for raw in [None, Some(""), Some("fp8")] {
            assert_eq!(
                resolve(device, mamba2.pins_f32_precision(), raw).unwrap(),
                Precision::F32,
                "{device} {raw:?}"
            );
        }
    }
    assert_eq!(resolve("hip", true, Some("f16")).unwrap(), Precision::F16);
    assert_eq!(
        resolve("cuda", true, Some(" bf16 ")).unwrap(),
        Precision::Bf16
    );
    assert!(resolve("metal", true, Some("bf16")).is_err());

    // An attention-only config keeps the per-device default (f16 on hip).
    let dense = Config::from_json_str(
        r#"{"model_type": "llama", "hidden_size": 16, "intermediate_size": 32,
            "num_hidden_layers": 2, "num_attention_heads": 2,
            "num_key_value_heads": 2, "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0, "vocab_size": 32}"#,
    )
    .unwrap();
    assert!(!dense.pins_f32_precision());
    assert_eq!(
        resolve("hip", dense.pins_f32_precision(), None).unwrap(),
        Precision::F16
    );
    assert!(resolve("hip", false, Some("fp8")).is_err());
}

/// The server's prompt-cache guard asks the crate root whether a checkpoint
/// threads recurrent state (issue #3341); it must agree with the config.
#[test]
fn config_json_has_recurrent_state_matches_the_config() {
    assert_eq!(
        crate::config_json_has_recurrent_state(TINY_MAMBA2),
        Ok(true)
    );
    let dense = r#"{"model_type": "llama", "hidden_size": 16, "intermediate_size": 32,
        "num_hidden_layers": 2, "num_attention_heads": 2,
        "num_key_value_heads": 2, "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0, "vocab_size": 32}"#;
    assert_eq!(crate::config_json_has_recurrent_state(dense), Ok(false));
    // An unsupported family is an error, so the caller's list decides.
    assert!(crate::config_json_has_recurrent_state(r#"{"model_type": "jamba"}"#).is_err());
}
