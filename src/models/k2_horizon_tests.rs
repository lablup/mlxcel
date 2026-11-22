// Copyright 2025-2026 Lablup Inc.
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

//! K2-Horizon config translation and refusals. Checkpoint-free: the fixtures
//! are the published `IFM/K2-Horizon-0.9B` and `IFM/K2-Horizon-7B` configs with
//! the fields the dense path neither reads nor screens dropped.

use serde_json::{Value, json};

use super::{
    K2HorizonConfig, TURN_TERMINATOR, added_token_id, decoder_token_id, eos_with_turn_terminator,
    parse_config,
};
use crate::models::llama3::EosTokenId;

/// `IFM/K2-Horizon-0.9B` `config.json` (flat `rope_theta`, YaRN under
/// `rope_parameters`, no `rope_scaling` key).
fn config_0_9b() -> Value {
    json!({
        "architectures": ["K2HorizonForCausalLM"],
        "attention_bias": false,
        "eos_token_id": 1,
        "head_dim": 64,
        "hidden_act": "silu",
        "hidden_size": 1536,
        "intermediate_size": 5120,
        "layernorm_num_groups": 1,
        "max_position_embeddings": 131072,
        "model_type": "k2_horizon",
        "num_attention_heads": 32,
        "num_experts": 0,
        "num_experts_per_tok": 0,
        "num_hidden_layers": 28,
        "num_key_value_heads": 8,
        "original_max_position_embeddings": 8192,
        "query_key_norm": false,
        "rms_norm_eps": 1e-06,
        "rope_head_dim": 64,
        "rope_parameters": {
            "attention_factor": 1.2772588722239782,
            "beta_fast": 128.0,
            "beta_slow": 4.0,
            "factor": 16.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "yarn",
            "truncate": true
        },
        "rope_theta": 1000000.0,
        "sliding_window": null,
        "tie_word_embeddings": false,
        "use_sliding_window": false,
        "vocab_size": 64256
    })
}

/// `IFM/K2-Horizon-7B` `config.json` (no flat `rope_theta`, the base only under
/// `rope_parameters`, grouped RMSNorm).
fn config_7b() -> Value {
    json!({
        "architectures": ["K2HorizonForCausalLM"],
        "attention_bias": false,
        "attention_gate_func": null,
        "eos_token_id": 1,
        "head_dim": 128,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "layernorm_num_groups": 4,
        "max_position_embeddings": 524288,
        "model_type": "k2_horizon",
        "mova_num_experts": 0,
        "num_attention_heads": 32,
        "num_experts": 0,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "query_key_norm": false,
        "rms_norm_eps": 1e-06,
        "rope_head_dim": 128,
        "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"},
        "sliding_window": null,
        "tie_word_embeddings": false,
        "use_sliding_window": false,
        "vocab_size": 250624
    })
}

fn parse(value: &Value) -> K2HorizonConfig {
    serde_json::from_value(value.clone()).expect("config parses")
}

fn refusal(value: &Value) -> String {
    parse_config(&value.to_string()).expect_err("config must be refused")
}

#[test]
fn config_0_9b_converts_to_yarn_at_base_1e6() {
    let config = parse_config(&config_0_9b().to_string()).expect("0.9B passes validate");
    let args = config.to_llama3_args();

    assert_eq!(args.rope_theta, 1e6);
    // Without the lift there is no `rope_scaling` key at all and this is None.
    let scaling = args
        .rope_scaling
        .as_ref()
        .expect("YaRN lifted into rope_scaling");
    assert_eq!(scaling.rope_type(), "yarn");
    assert_eq!(scaling.factor, Some(16.0));
    assert_eq!(scaling.beta_fast, Some(128.0));
    assert_eq!(scaling.beta_slow, Some(4.0));
    assert_eq!(scaling.original_max_position_embeddings, Some(8192.0));

    assert_eq!(args.head_dim, Some(64));
    assert_eq!(args.num_key_value_heads, Some(8));
    assert_eq!(args.max_position_embeddings, Some(131_072));
    assert!(!args.tie_word_embeddings);
    assert!(!args.rope_traditional);
    assert_eq!(args.eos_token_ids(), vec![1]);
}

#[test]
fn config_7b_converts_to_base_1e7_without_scaling() {
    let config = parse(&config_7b());
    // The field itself, not the resolved base: the resolver's fallback is also
    // 1e7, so only the field shows whether the lift ran.
    assert_eq!(config.rope_theta, Some(1e7));
    let args = config.to_llama3_args();
    assert_eq!(args.rope_theta, 1e7);
    assert!(args.rope_scaling.is_none());
}

#[test]
fn nested_base_is_lifted_when_flat_key_is_null() {
    let mut value = config_7b();
    value["rope_theta"] = Value::Null;
    value["rope_parameters"] = json!({"rope_theta": 500000.0, "rope_type": "default"});
    let config = parse(&value);
    assert_eq!(config.rope_theta, Some(5e5));
    assert_eq!(config.to_llama3_args().rope_theta, 5e5);
}

#[test]
fn grouped_rmsnorm_is_refused() {
    let err = refusal(&config_7b());
    assert!(err.contains("layernorm_num_groups (4)"), "{err}");

    let mut value = config_0_9b();
    value["layernorm_num_groups"] = json!(2);
    assert!(refusal(&value).contains("layernorm_num_groups (2)"));
}

#[test]
fn unsupported_features_are_refused_by_name() {
    let cases: [(&str, Value, &str); 6] = [
        ("query_key_norm", json!(true), "query_key_norm"),
        ("num_experts", json!(64), "num_experts (64)"),
        ("mova_num_experts", json!(8), "mova_num_experts (8)"),
        (
            "attention_gate_func",
            json!("sigmoid"),
            "attention_gate_func",
        ),
        ("use_sliding_window", json!(true), "use_sliding_window"),
        ("rope_head_dim", json!(32), "rope_head_dim (32)"),
    ];
    for (field, bad, needle) in cases {
        let mut value = config_0_9b();
        value[field] = bad;
        let err = refusal(&value);
        assert!(err.contains(needle), "{field}: {err}");
    }
}

#[test]
fn omitted_screened_fields_take_the_hf_defaults() {
    // HF's `K2HorizonConfig` defaults `query_key_norm` to true and
    // `num_experts` to 128, so a config that omits either describes a QK-norm
    // or MoE model and must not load as the dense decoder.
    for (field, needle) in [
        ("query_key_norm", "query_key_norm"),
        ("num_experts", "num_experts (128)"),
    ] {
        let mut value = config_0_9b();
        value.as_object_mut().unwrap().remove(field);
        let err = refusal(&value);
        assert!(err.contains(needle), "{field}: {err}");
    }
}

#[test]
fn null_screened_fields_are_accepted() {
    let mut value = config_0_9b();
    value["attention_gate_func"] = Value::Null;
    value["use_sliding_window"] = Value::Null;
    value["rope_head_dim"] = Value::Null;
    parse_config(&value.to_string()).expect("null screened fields load");
}

#[test]
fn bad_geometry_is_refused() {
    let cases: [(&str, Value, &str); 4] = [
        (
            "num_attention_heads",
            json!(0),
            "num_attention_heads must be at least 1",
        ),
        ("num_key_value_heads", json!(5), "num_key_value_heads (5)"),
        ("head_dim", json!(63), "head_dim (63)"),
        ("rms_norm_eps", json!(0.0), "rms_norm_eps"),
    ];
    for (field, bad, needle) in cases {
        let mut value = config_0_9b();
        value[field] = bad;
        let err = refusal(&value);
        assert!(err.contains(needle), "{field}: {err}");
    }
}

#[test]
fn turn_terminator_resolves_by_name_and_joins_the_eos_ids() {
    let tokenizer_json = json!({"added_tokens": [
        {"id": 1, "content": "<|endoftext|>", "special": true},
        {"id": 64019, "content": TURN_TERMINATOR, "special": true}
    ]});
    assert_eq!(
        added_token_id(&tokenizer_json, TURN_TERMINATOR),
        Some(64019)
    );
    assert_eq!(
        added_token_id(&json!({"added_tokens": []}), TURN_TERMINATOR),
        None
    );

    let tokenizer_config = json!({"added_tokens_decoder": {
        "1": {"content": "<|endoftext|>"},
        "64019": {"content": TURN_TERMINATOR}
    }});
    assert_eq!(
        decoder_token_id(&tokenizer_config, TURN_TERMINATOR),
        Some(64019)
    );

    let mut args = parse(&config_0_9b()).to_llama3_args();
    args.eos_token_id = eos_with_turn_terminator(args.eos_token_id.as_ref(), Some(64019));
    assert_eq!(args.eos_token_ids(), vec![1, 64019]);

    // Already declared, or unresolved: the declared set is unchanged.
    let declared = EosTokenId::Multiple(vec![1, 64019]);
    let kept = eos_with_turn_terminator(Some(&declared), Some(64019));
    assert!(matches!(kept, Some(EosTokenId::Multiple(ref ids)) if ids == &vec![1, 64019]));
    let unresolved = eos_with_turn_terminator(Some(&EosTokenId::Single(1)), None);
    assert!(matches!(unresolved, Some(EosTokenId::Multiple(ref ids)) if ids == &vec![1]));
    assert!(eos_with_turn_terminator(None, None).is_none());
}

#[test]
fn turn_terminator_resolves_from_the_real_checkpoint_tokenizer() {
    // Skips when the checkpoint is not present on this host.
    let dir = std::path::Path::new("models/k2-horizon-0.9b");
    if !dir.join("tokenizer.json").exists() {
        return;
    }
    assert_eq!(super::resolve_turn_terminator(dir), Some(64019));
}

#[test]
fn oversized_geometry_is_refused() {
    // 2147483648 wraps to a negative i32 in an `as i32` cast.
    for field in [
        "num_attention_heads",
        "num_key_value_heads",
        "hidden_size",
        "intermediate_size",
        "vocab_size",
        "max_position_embeddings",
    ] {
        let mut value = config_0_9b();
        value[field] = json!(2_147_483_648u64);
        let err = refusal(&value);
        assert!(
            err.contains(field) && err.contains("exceeds"),
            "{field}: {err}"
        );
    }
}

#[test]
fn non_silu_hidden_act_is_refused() {
    let mut value = config_0_9b();
    value["hidden_act"] = json!("gelu");
    let err = refusal(&value);
    assert!(err.contains("hidden_act (gelu)"), "{err}");

    // Omitted takes the HF default, silu.
    value.as_object_mut().unwrap().remove("hidden_act");
    parse_config(&value.to_string()).expect("omitted hidden_act is silu");
}

#[test]
fn omitted_head_dim_defaults_to_128_like_hf() {
    // 1536 / 32 = 48 would be the old fallback; HF and mlx-vlm use 128.
    let mut value = config_0_9b();
    value.as_object_mut().unwrap().remove("head_dim");
    value.as_object_mut().unwrap().remove("rope_head_dim");
    let config = parse_config(&value.to_string()).expect("loads with the default head_dim");
    assert_eq!(config.resolved_head_dim(), 128);

    // The real 0.9B declares 64 and keeps it.
    let declared = parse_config(&config_0_9b().to_string()).expect("0.9B loads");
    assert_eq!(declared.resolved_head_dim(), 64);
}

#[test]
fn attention_factor_must_match_the_shared_yarn_derivation() {
    // 0.9B's value is `0.1 * ln(16) + 1`.
    parse_config(&config_0_9b().to_string()).expect("0.9B attention_factor matches");

    let mut value = config_0_9b();
    value["rope_parameters"]["attention_factor"] = json!(1.0);
    let err = refusal(&value);
    assert!(err.contains("attention_factor (1)"), "{err}");

    // Absent is fine: the shared path derives it.
    value["rope_parameters"]
        .as_object_mut()
        .unwrap()
        .remove("attention_factor");
    parse_config(&value.to_string()).expect("absent attention_factor loads");
}
