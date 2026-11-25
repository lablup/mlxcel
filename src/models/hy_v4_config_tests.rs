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

//! Hy4 configuration tests (issue #2354).

use super::super::hy_v4_tests::{Tiny, tiny_config, tiny_config_json};
use super::{HyV4Config, IndexerType, ParityArms};

/// The keys `mlx-community/Hy4-preview-4bit` ships in `config.json`, with its
/// two 78-entry lists rebuilt from their pattern: `full` on layers 0, 1 and
/// every `i % 4 == 1`, and a dense MLP on layer 0 only.
fn shipped_config_json() -> serde_json::Value {
    let indexer_types: Vec<&str> = (0..78)
        .map(|i| {
            if i < 2 || i % 4 == 1 {
                "full"
            } else {
                "shared"
            }
        })
        .collect();
    let mlp_layer_types: Vec<&str> = (0..78)
        .map(|i| if i == 0 { "dense" } else { "sparse" })
        .collect();
    serde_json::json!({
        "architectures": ["HYV4ForCausalLM"],
        "attention_bias": false,
        "bos_token_id": 120000,
        "enable_lm_head_fp32": true,
        "eos_token_id": 120025,
        "hc_eps": 1e-06,
        "hc_magnitude": 2.0,
        "hc_mult": 4,
        "head_dim": 64,
        "hidden_size": 6144,
        "index_head_dim": 128,
        "index_n_heads": 32,
        "index_topk": 2048,
        "indexer_types": indexer_types,
        "intermediate_size": 18432,
        "kv_lora_rank": 512,
        "max_position_embeddings": 1048576,
        "mlp_layer_types": mlp_layer_types,
        "model_type": "hy_v4",
        "moe_intermediate_size": 2048,
        "n_group": 1,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "norm_topk_prob": true,
        "num_attention_heads": 64,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 78,
        "num_key_value_heads": 8,
        "pad_token_id": 120002,
        "q_lora_rank": 2048,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
        "rms_norm_eps": 1e-05,
        "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
        "routed_scaling_factor": 2.827,
        "swiglu_limit": 10.0,
        "tie_word_embeddings": false,
        "topk_group": 1,
        "v_head_dim": 256,
        "vocab_size": 120832,
    })
}

#[test]
fn shipped_checkpoint_config_parses_to_the_expected_geometry() {
    let config: HyV4Config = serde_json::from_value(shipped_config_json()).expect("parses");
    let base = &config.base;
    assert_eq!(base.num_hidden_layers, 78);
    assert_eq!(base.rope_theta, 1e7);
    assert_eq!(base.rms_norm_eps, 1e-5);
    assert_eq!((base.q_lora_rank, base.kv_lora_rank), (2048, 512));
    assert_eq!(
        (
            base.qk_nope_head_dim,
            base.qk_rope_head_dim,
            base.v_head_dim
        ),
        (192, 64, 256)
    );
    assert_eq!(
        (base.index_n_heads, base.index_head_dim, base.index_topk),
        (32, 128, 2048)
    );
    assert_eq!((base.group_size(), base.bits()), (64, 4));
    assert_eq!(base.n_routed_experts, Some(256));
    assert_eq!(base.num_experts_per_tok, 8);
    assert!(!base.tie_word_embeddings);
    assert_eq!(base.mscale_all_dim(), 0.0);

    let full = config
        .indexer_types
        .iter()
        .filter(|kind| **kind == IndexerType::Full)
        .count();
    assert_eq!(full, 21);
    assert_eq!(config.mlp_is_moe.iter().filter(|moe| !**moe).count(), 1);
    assert!(!config.mlp_is_moe[0]);
    assert_eq!(
        (config.hc_mult, config.hc_eps, config.hc_magnitude),
        (4, 1e-6, 2.0)
    );
    assert_eq!(config.swiglu_limit, Some(10.0));
    assert_eq!(config.eos_token_ids, [120025]);
}

#[test]
fn config_reads_the_hy4_keys_and_lifts_rope_theta() {
    let config = tiny_config(Tiny::new(2));
    assert_eq!(config.base.rope_theta, 1e7);
    assert_eq!(
        config.indexer_types,
        [IndexerType::Full, IndexerType::Shared]
    );
    assert_eq!(config.mlp_is_moe, [false, true]);
    assert_eq!(config.hc_mult, 2);
    assert_eq!(config.swiglu_limit, Some(10.0));
    assert_eq!(config.eos_token_ids, [7]);
    assert!(config.base.indexer_rope_interleave);
}

/// Without the lists the upstream defaults apply: `full` on layers 0, 1 and
/// every `i % 4 == 1`, and a dense MLP on layer 0 only.
#[test]
fn config_defaults_follow_upstream_when_the_lists_are_absent() {
    let mut json = tiny_config_json(Tiny::new(2));
    let object = json.as_object_mut().expect("object");
    object.remove("indexer_types");
    object.remove("mlp_layer_types");
    object.remove("eos_token_id");
    object.insert("num_hidden_layers".into(), 7.into());
    let config: HyV4Config = serde_json::from_value(json).expect("parses");
    use IndexerType::{Full, Shared};
    assert_eq!(
        config.indexer_types,
        [Full, Full, Shared, Shared, Shared, Full, Shared]
    );
    assert_eq!(
        config.mlp_is_moe,
        [false, true, true, true, true, true, true]
    );
    assert_eq!(config.eos_token_ids, [120025]);
}

#[test]
fn config_rejects_bad_indexer_and_mlp_lists() {
    let parse = |edit: &dyn Fn(&mut serde_json::Value)| {
        let mut json = tiny_config_json(Tiny::new(2));
        edit(&mut json);
        serde_json::from_value::<HyV4Config>(json)
            .expect_err("config must be refused")
            .to_string()
    };

    let err = parse(&|json| json["indexer_types"] = serde_json::json!(["full"]));
    assert!(
        err.contains("`indexer_types` must have one entry per hidden layer, got 1 for 2 layers"),
        "{err}"
    );
    let err = parse(&|json| json["indexer_types"] = serde_json::json!(["full", "sparse"]));
    assert!(
        err.contains("unsupported indexer type `sparse` at indexer_types[1]"),
        "{err}"
    );
    let err = parse(&|json| json["indexer_types"] = serde_json::json!(["shared", "full"]));
    assert!(err.contains("indexer_types[0] is `shared`"), "{err}");
    let err = parse(&|json| json["mlp_layer_types"] = serde_json::json!(["dense"]));
    assert!(
        err.contains("`mlp_layer_types` must have one entry per hidden layer"),
        "{err}"
    );
    let err = parse(&|json| json["mlp_layer_types"] = serde_json::json!(["dense", "moe"]));
    assert!(err.contains("unsupported MLP type `moe`"), "{err}");
}

#[test]
fn parity_arms_parse_and_reject_unknown_names() {
    assert_eq!(ParityArms::parse("").unwrap(), ParityArms::default());
    let arms = ParityArms::parse("no_sinks, rope_half_split").unwrap();
    assert!(arms.no_sinks && arms.rope_half_split && !arms.no_gate && !arms.no_swiglu_limit);
    let err = ParityArms::parse("no_sink").unwrap_err();
    assert!(err.contains("unknown arm `no_sink`"), "{err}");
}

#[test]
fn hc_mult_and_eos_ids_are_bounded() {
    let parse = |edit: &dyn Fn(&mut serde_json::Value)| {
        let mut json = tiny_config_json(Tiny::new(2));
        edit(&mut json);
        serde_json::from_value::<HyV4Config>(json)
            .expect_err("config must be refused")
            .to_string()
    };
    let err = parse(&|json| json["hc_mult"] = serde_json::json!(65));
    assert!(err.contains("hc_mult must be between 1 and 64"), "{err}");
    let err = parse(&|json| json["hc_mult"] = serde_json::json!(0));
    assert!(err.contains("hc_mult must be between 1 and 64"), "{err}");
    let err = parse(&|json| {
        json["hc_mult"] = serde_json::json!(64);
        json["hidden_size"] = serde_json::json!(20_000_000);
    });
    assert!(err.contains("does not fit in i32"), "{err}");
    let err = parse(&|json| json["eos_token_id"] = serde_json::json!(4_294_967_297_i64));
    assert!(err.contains("invalid eos_token_id"), "{err}");
    let err = parse(&|json| json["eos_token_id"] = serde_json::json!([1, -2]));
    assert!(err.contains("invalid eos_token_id"), "{err}");
}
