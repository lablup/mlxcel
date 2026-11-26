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

//! Jamba config, weight schema, checkpoint, and graph checks (issue #3349).

use super::config::{
    BcdtNorm, Config, ExpertAct, MixerKind, MoeScoring, SsmConfig, SsmKind, WeightScheme,
};
use super::model::{ProbeMode, emit_ssm_probe};
use super::recurrent_tests::write_safetensors;
use super::{emit_decode, emit_decode_ragged, emit_prefill};

/// The seeded four-block MoE fixture: Mamba1+MoE, attention+MLP, Mamba1+MLP,
/// attention+MoE (`spike/openxla/jamba_moe_fixture.py`).
const TINY: &str = include_str!("../../assets/jamba/config.json");

/// `ai21labs/AI21-Jamba-Reasoning-3B` `config.json` (revision 75243704).
const REASONING_3B: &str = r#"{
  "architectures": ["JambaForCausalLM"], "attention_dropout": 0.0,
  "attn_layer_offset": 7, "attn_layer_period": 14, "bos_token_id": 1,
  "dtype": "bfloat16", "eos_token_id": [2, 519], "expert_layer_offset": 1,
  "expert_layer_period": 2, "hidden_act": "silu", "hidden_size": 2560,
  "initializer_range": 0.02, "intermediate_size": 8192, "mamba_conv_bias": true,
  "mamba_d_conv": 4, "mamba_d_state": 16, "mamba_dt_rank": 160, "mamba_expand": 2,
  "mamba_proj_bias": false, "max_position_embeddings": 262144, "model_type": "jamba",
  "num_attention_heads": 20, "num_experts": 1, "num_experts_per_tok": 1,
  "num_hidden_layers": 28, "num_key_value_heads": 1, "num_logits_to_keep": 1,
  "output_router_logits": false, "pad_token_id": 0, "rms_norm_eps": 1e-06,
  "router_aux_loss_coef": 0.001, "sliding_window": null, "tie_word_embeddings": true,
  "use_cache": true, "use_mamba_kernels": true, "vocab_size": 65536
}"#;

fn with(key: &str, value: serde_json::Value) -> String {
    let mut v: serde_json::Value = serde_json::from_str(TINY).unwrap();
    v[key] = value;
    v.to_string()
}

#[test]
fn jamba_3b_schedule_is_dense_with_two_attention_blocks() {
    let c = Config::from_json_str(REASONING_3B).expect("Jamba 3B parses");
    assert_eq!(c.weight_scheme, WeightScheme::Jamba);
    assert_eq!(
        (c.hidden, c.inter, c.n_q, c.n_kv, c.head_dim),
        (2560, 8192, 20, 1, 128)
    );
    assert!(c.tie_word_embeddings);
    assert!(c.moe.is_none(), "num_experts = 1 keeps every FFN dense");
    let attention: Vec<usize> = (0..28)
        .filter(|&li| c.layer_mixer(li) == MixerKind::Attention)
        .collect();
    assert_eq!(attention, vec![7, 21]);
    assert_eq!(c.attn_layer_count(), 2);
    assert_eq!(c.recurrent_layer_count(), 26);
    assert_eq!(c.attn_layer_index(21), Some(1));
    assert_eq!(c.state_layer_index(8), Some(7));
    assert!((0..28).all(|li| c.layer_has_ffn(li) && !c.is_ffn_free_ssm_layer(li)));
    assert!((0..28).all(|li| c.layer_has_post_mixer_norm(li)));
    assert!(
        (0..28).all(|li| !c.layer_uses_rope(li)),
        "Jamba attention has no RoPE"
    );
    assert!(c.has_recurrent_state() && c.has_kv_cache());
    let ssm = c.ssm.as_ref().unwrap();
    assert_eq!(
        ssm.kind,
        SsmKind::Mamba1 {
            dt_rank: 160,
            bcdt_norm: BcdtNorm::Weighted
        }
    );
    assert_eq!(
        (ssm.d_inner(), ssm.state_size, ssm.conv_kernel),
        (5120, 16, 4)
    );
    assert!(ssm.use_conv_bias && !ssm.use_bias);
    assert_eq!(ssm.norm_eps, 1e-6);
    let state: Vec<(&str, Vec<usize>)> = c
        .state_tensors()
        .into_iter()
        .map(|spec| (spec.name, spec.dims))
        .collect();
    assert_eq!(
        state,
        vec![
            ("ssm_h", vec![26, 5120, 16]),
            ("ssm_conv", vec![26, 3, 5120])
        ]
    );
}

#[test]
fn jamba_tiny_moe_schedule_and_hf_router() {
    let c = Config::from_json_str(TINY).expect("tiny Jamba parses");
    assert_eq!(
        c.layer_mixers,
        vec![
            MixerKind::Mamba1,
            MixerKind::Attention,
            MixerKind::Mamba1,
            MixerKind::Attention
        ]
    );
    let moe = c.moe.as_ref().expect("four experts make the MoE blocks");
    assert_eq!(moe.layer_mask, Some(vec![true, false, false, true]));
    assert_eq!((moe.n_experts, moe.top_k, moe.intermediate), (4, 2, 24));
    // HF JambaSparseMoeBlock: softmax over every expert, top-k, no renorm.
    assert_eq!(moe.scoring, MoeScoring::Softmax);
    assert!(!moe.norm_topk_prob);
    assert_eq!(moe.routed_scaling_factor, 1.0);
    assert_eq!(moe.expert_act, ExpertAct::SwiGlu);
    assert!(moe.shared.is_none());
    assert_eq!(moe.weight_prefix, "feed_forward");
    assert_eq!(
        (0..4).map(|li| c.is_moe_layer(li)).collect::<Vec<_>>(),
        vec![true, false, false, true]
    );
}

/// The IREE decode compile drops aggressive fusion for exactly these configs
/// (`iree::decode_compile_flags`, #3349): a softmax router, not any MoE.
#[test]
fn softmax_moe_selects_the_unfused_decode_compile() {
    for (what, json, softmax) in [
        ("jamba MoE", TINY.to_string(), true),
        ("jamba 3B dense", REASONING_3B.to_string(), false),
        ("jamba one expert", with("num_experts", 1.into()), false),
        (
            "qwen2_moe",
            include_str!("../../assets/qwen2-moe-tiny/config.json").to_string(),
            true,
        ),
        (
            "olmoe",
            include_str!("../../assets/olmoe-tiny/config.json").to_string(),
            true,
        ),
        (
            "llama",
            include_str!("../../assets/llama-3.2-1b/config.json").to_string(),
            false,
        ),
    ] {
        let c = Config::from_json_str(&json).unwrap();
        assert_eq!(c.has_softmax_moe(), softmax, "{what}");
    }
}

#[test]
fn jamba_config_rejections() {
    for (key, value, needle) in [
        (
            "attn_layer_offset",
            serde_json::json!(2),
            "attn_layer_offset",
        ),
        (
            "expert_layer_period",
            serde_json::json!(0),
            "expert_layer_period",
        ),
        (
            "num_experts_per_tok",
            serde_json::json!(5),
            "num_experts_per_tok",
        ),
        ("hidden_act", serde_json::json!("gelu"), "hidden_act"),
        ("sliding_window", serde_json::json!(4), "sliding_window"),
        ("mamba_dt_rank", serde_json::json!("half"), "mamba_dt_rank"),
        ("rms_norm_eps", serde_json::json!(0.0), "rms_norm_eps"),
        (
            "num_key_value_heads",
            serde_json::json!(3),
            "attention dimensions",
        ),
    ] {
        let error = Config::from_json_str(&with(key, value)).expect_err(key);
        assert!(error.contains(needle), "{key}: {error}");
    }
    let auto = Config::from_json_str(&with("mamba_dt_rank", serde_json::json!("auto"))).unwrap();
    assert!(matches!(
        auto.ssm.unwrap().kind,
        SsmKind::Mamba1 { dt_rank: 1, .. }
    ));
}

#[test]
fn jamba_weight_schema_names_every_block_shape() {
    let c = Config::from_json_str(TINY).unwrap();
    let names = crate::weight_names::weight_names(&c);
    let layer = |li: usize| -> Vec<String> {
        let p = format!("model.layers.{li}.");
        names
            .iter()
            .filter_map(|n| n.strip_prefix(&p).map(str::to_string))
            .collect()
    };
    let mamba = [
        "mamba.in_proj.weight",
        "mamba.conv1d.weight",
        "mamba.conv1d.bias",
        "mamba.x_proj.weight",
        "mamba.dt_proj.weight",
        "mamba.dt_proj.bias",
        "mamba.A_log",
        "mamba.D",
        "mamba.dt_layernorm.weight",
        "mamba.b_layernorm.weight",
        "mamba.c_layernorm.weight",
        "mamba.out_proj.weight",
    ];
    let moe = [
        "feed_forward.router.weight",
        "feed_forward.switch_mlp.gate_proj.weight",
        "feed_forward.switch_mlp.up_proj.weight",
        "feed_forward.switch_mlp.down_proj.weight",
    ];
    let norms = ["input_layernorm.weight", "pre_ff_layernorm.weight"];
    let mut want0: Vec<&str> = norms.to_vec();
    want0.extend(moe);
    want0.extend(mamba);
    assert_eq!(layer(0), want0, "Mamba1 + MoE");
    let mut want2 = vec![
        "feed_forward.down_proj.weight",
        "feed_forward.gate_proj.weight",
    ];
    want2.extend(norms);
    want2.push("feed_forward.up_proj.weight");
    want2.extend(mamba);
    assert_eq!(layer(2), want2, "Mamba1 + dense MLP");
    let attention = [
        "self_attn.k_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_proj.weight",
        "self_attn.v_proj.weight",
    ];
    let mut want1 = vec![
        "feed_forward.down_proj.weight",
        "feed_forward.gate_proj.weight",
    ];
    want1.extend(norms);
    want1.push("feed_forward.up_proj.weight");
    want1.extend(attention);
    assert_eq!(layer(1), want1, "attention + dense MLP");
    let mut want3: Vec<&str> = norms.to_vec();
    want3.extend(attention);
    want3.extend(moe);
    assert_eq!(layer(3), want3, "attention + MoE");
    assert_eq!(
        &names[..2],
        ["model.embed_tokens.weight", "model.final_layernorm.weight"]
    );
    assert!(
        !names
            .iter()
            .any(|n| n.contains("unused") || n == "lm_head.weight")
    );
}

#[test]
fn weighted_bcdt_norm_scales_the_weightless_norm() {
    let probe = |mode| emit_ssm_probe(&SsmConfig::tiny_mamba1(mode), 8, 3, ProbeMode::Decode);
    let (weightless, weighted) = (probe(BcdtNorm::Weightless), probe(BcdtNorm::Weighted));
    assert_eq!(weighted.matches("stablehlo.rsqrt").count(), 3);
    for name in ["ssm.dt_norm_w", "ssm.b_norm_w", "ssm.c_norm_w"] {
        assert!(
            !weightless.contains(name) && weighted.contains(name),
            "{name}"
        );
    }
    assert_eq!(
        weighted.matches("stablehlo.multiply").count(),
        weightless.matches("stablehlo.multiply").count() + 3,
        "one per-channel scale for each of dt, B, and C"
    );
}

#[test]
fn jamba_graphs_thread_compact_kv_then_mamba1_state() {
    let mut c = Config::from_json_str(TINY).unwrap();
    c.context_capacity = 32;
    let decode = emit_decode(&c, false);
    assert!(
        decode.contains("tensor<2x32x2x4xf32>"),
        "two attention KV rows"
    );
    assert!(decode.contains("@kv_import") && decode.contains("@state_import"));
    assert!(
        decode.contains("tensor<2x32x4xf32>"),
        "two Mamba1 [D,N] rows"
    );
    let prefill = emit_prefill(&c, false);
    let ragged = emit_decode_ragged(&c, 4, false).unwrap();
    let sig = ragged
        .lines()
        .find(|line| line.contains("func.func public @main"))
        .unwrap();
    assert!(sig.find("loc(\"kcache\")").unwrap() < sig.find("loc(\"ssm_h\")").unwrap());
    for graph in [&decode, &prefill, &ragged] {
        for li in [0, 2] {
            assert!(graph.contains(&format!("['layers'][{li}]['ssm.dt_norm_w']")));
            assert!(graph.contains(&format!("['layers'][{li}]['post_ln']")));
            assert!(!graph.contains(&format!("['layers'][{li}]['wq']")));
        }
        for li in [1, 3] {
            assert!(!graph.contains(&format!("['layers'][{li}]['ssm.in_proj']")));
        }
        assert!(graph.contains("['layers'][0]['moe_router']"));
        assert!(graph.contains("['layers'][3]['moe_router']"));
        assert!(graph.contains("['layers'][2]['down']"));
        assert!(!graph.contains("['layers'][0]['down']"));
    }
}

/// The tiny fixture's tensors in the stacked layout the loader reads.
fn tiny_checkpoint_shapes() -> Vec<(String, Vec<usize>)> {
    let (h, i, e, d, n, r) = (16, 24, 4, 32, 4, 2);
    let mut shapes: Vec<(String, Vec<usize>)> = vec![
        ("model.embed_tokens.weight".into(), vec![32, h]),
        ("model.final_layernorm.weight".into(), vec![h]),
    ];
    for li in 0..4 {
        let p = format!("model.layers.{li}.");
        let mut push = |name: &str, shape: Vec<usize>| shapes.push((format!("{p}{name}"), shape));
        push("input_layernorm.weight", vec![h]);
        push("pre_ff_layernorm.weight", vec![h]);
        if li % 2 == 1 {
            push("self_attn.q_proj.weight", vec![16, h]);
            push("self_attn.k_proj.weight", vec![8, h]);
            push("self_attn.v_proj.weight", vec![8, h]);
            push("self_attn.o_proj.weight", vec![h, 16]);
        } else {
            push("mamba.in_proj.weight", vec![2 * d, h]);
            push("mamba.conv1d.weight", vec![d, 1, 4]);
            push("mamba.conv1d.bias", vec![d]);
            push("mamba.x_proj.weight", vec![r + 2 * n, d]);
            push("mamba.dt_proj.weight", vec![d, r]);
            push("mamba.dt_proj.bias", vec![d]);
            push("mamba.A_log", vec![d, n]);
            push("mamba.D", vec![d]);
            push("mamba.dt_layernorm.weight", vec![r]);
            push("mamba.b_layernorm.weight", vec![n]);
            push("mamba.c_layernorm.weight", vec![n]);
            push("mamba.out_proj.weight", vec![h, d]);
        }
        if li % 3 == 0 {
            push("feed_forward.router.weight", vec![e, h]);
            push("feed_forward.switch_mlp.gate_proj.weight", vec![e, i, h]);
            push("feed_forward.switch_mlp.up_proj.weight", vec![e, i, h]);
            push("feed_forward.switch_mlp.down_proj.weight", vec![e, h, i]);
        } else {
            push("feed_forward.gate_proj.weight", vec![i, h]);
            push("feed_forward.up_proj.weight", vec![i, h]);
            push("feed_forward.down_proj.weight", vec![h, i]);
        }
    }
    shapes
}

#[test]
fn jamba_checkpoint_validation_covers_mixer_norms_and_experts() {
    let c = Config::from_json_str(TINY).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    write_safetensors(&path, &tiny_checkpoint_shapes());
    c.validate_recurrent_checkpoint(dir.path())
        .expect("the fixture layout validates");
    for missing in [
        "model.layers.0.mamba.dt_layernorm.weight",
        "model.layers.2.mamba.c_layernorm.weight",
        "model.layers.3.feed_forward.router.weight",
        "model.layers.1.feed_forward.up_proj.weight",
        "model.layers.2.pre_ff_layernorm.weight",
    ] {
        let shapes: Vec<_> = tiny_checkpoint_shapes()
            .into_iter()
            .filter(|(name, _)| name != missing)
            .collect();
        write_safetensors(&path, &shapes);
        let error = c
            .validate_recurrent_checkpoint(dir.path())
            .expect_err(missing);
        assert!(error.contains(missing), "{missing}: {error}");
    }
    let mut wrong = tiny_checkpoint_shapes();
    for (name, shape) in &mut wrong {
        if name == "model.layers.0.mamba.b_layernorm.weight" {
            *shape = vec![5];
        }
    }
    write_safetensors(&path, &wrong);
    let error = c.validate_recurrent_checkpoint(dir.path()).unwrap_err();
    assert!(error.contains("b_layernorm"), "{error}");
}

/// Regenerate the Jamba goldens with `MLXCEL_FREEZE_JAMBA=1`.
#[test]
fn freeze_jamba_goldens() {
    if std::env::var("MLXCEL_FREEZE_JAMBA").as_deref() != Ok("1") {
        return;
    }
    use crate::validation::{GraphKind, emit_graphs};
    for (kind, name) in [
        (GraphKind::Decode { sample: true }, "decode.mlir"),
        (GraphKind::Prefill { sample: true }, "prefill.mlir"),
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
                .join("assets/jamba")
                .join(name),
            &graph[0].1,
        )
        .unwrap();
    }
}
