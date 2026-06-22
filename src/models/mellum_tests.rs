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

//! Unit tests for the Mellum (Mellum 2) loader.
//!
//! These cover the checkpoint-free surface: config deserialization (the real
//! `JetBrains/Mellum2-12B-A2.5B-Base` field set, including the per-layer-type
//! `rope_parameters` dict and the `layer_types` schedule), the YaRN attention
//! factor derivation, the per-expert -> `switch_mlp` stacking in `sanitize`,
//! the per-layer cache selection (full -> `KVCache`, sliding ->
//! `RotatingKVCache`), and tied/untied LM-head handling. End-to-end generation
//! is validated separately against a real MLX Mellum 2 checkpoint, which needs a
//! Metal device.

use super::mellum::{Cache, MellumModel, ModelArgs, compute_yarn_rope};
use mlxcel_core::weights::WeightMap;

/// Trimmed `JetBrains/Mellum2-12B-A2.5B-Base` config with the fields the loader
/// reads. `layer_types` makes every 4th layer (indices 3, 7, ... 27) full
/// attention and the rest sliding; `rope_parameters` carries YaRN for the full
/// layers and default RoPE for the sliding layers.
const MELLUM_12B_CONFIG: &str = r#"{
    "model_type": "mellum",
    "architectures": ["MellumForCausalLM"],
    "hidden_size": 2304,
    "head_dim": 128,
    "num_hidden_layers": 28,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-06,
    "vocab_size": 98304,
    "intermediate_size": 7168,
    "moe_intermediate_size": 896,
    "num_experts": 64,
    "num_experts_per_tok": 8,
    "norm_topk_prob": true,
    "tie_word_embeddings": false,
    "max_position_embeddings": 131072,
    "sliding_window": 1024,
    "layer_types": [
        "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention", "full_attention"
    ],
    "mlp_layer_types": [
        "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse",
        "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse",
        "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse",
        "sparse", "sparse", "sparse", "sparse", "sparse", "sparse", "sparse"
    ],
    "rope_parameters": {
        "full_attention": {
            "rope_type": "yarn",
            "rope_theta": 500000.0,
            "factor": 16.0,
            "original_max_position_embeddings": 8192,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "attention_factor": 1.2772588722239782
        },
        "sliding_attention": {
            "rope_type": "default",
            "rope_theta": 500000.0
        }
    }
}"#;

fn parse_config() -> ModelArgs {
    serde_json::from_str(MELLUM_12B_CONFIG).expect("parse mellum config")
}

#[test]
fn mellum_config_parses_core_fields() {
    let args = parse_config();

    assert_eq!(args.model_type, "mellum");
    assert_eq!(args.hidden_size, 2304);
    assert_eq!(args.head_dim, 128);
    assert_eq!(args.num_hidden_layers, 28);
    assert_eq!(args.num_attention_heads, 32);
    assert_eq!(args.num_key_value_heads, 4);
    assert_eq!(args.vocab_size, 98304);
    assert_eq!(args.num_experts, 64);
    assert_eq!(args.num_experts_per_tok, 8);
    assert_eq!(args.moe_intermediate_size, 896);
    assert!(args.norm_topk_prob);
    assert!(!args.tie_word_embeddings);
    assert_eq!(args.sliding_window, 1024);
    assert_eq!(args.max_position_embeddings, 131072);
    assert_eq!(args.layer_types.len(), 28);
}

#[test]
fn mellum_layer_types_select_full_and_sliding() {
    let args = parse_config();

    // Every 4th layer (indices 3, 7, 11, 15, 19, 23, 27) is full attention.
    for (idx, expected_sliding) in (0..28).map(|i| (i, (i + 1) % 4 != 0)) {
        assert_eq!(
            args.is_sliding(idx),
            expected_sliding,
            "layer {idx} sliding flag",
        );
    }

    // All 28 layers are sparse MoE in this checkpoint.
    assert!((0..28).all(|i| args.is_moe_layer(i)));

    // Both layer types share base 500000, but only full uses YaRN.
    assert_eq!(args.rope_theta_for("full_attention"), 500_000.0);
    assert_eq!(args.rope_theta_for("sliding_attention"), 500_000.0);
}

#[test]
fn mellum_full_layer_yarn_attention_factor_matches_config() {
    let args = parse_config();
    let params = args
        .rope_parameters
        .get("full_attention")
        .expect("full_attention rope params");

    let yarn = compute_yarn_rope(args.head_dim, params).expect("full layers use YaRN");
    // mlx-lm derives the attention factor from mscale/mscale_all_dim; the value
    // must equal the config's explicit `attention_factor`.
    assert!(
        (yarn.mscale - 1.277_258_9).abs() < 1e-5,
        "yarn mscale = {}",
        yarn.mscale
    );

    // Sliding layers use default RoPE, so no YaRN frequencies.
    let sliding = args
        .rope_parameters
        .get("sliding_attention")
        .expect("sliding_attention rope params");
    assert!(compute_yarn_rope(args.head_dim, sliding).is_none());
}

#[test]
fn mellum_sanitize_stacks_experts_and_pops_tied_head() {
    // One MoE layer with 2 experts; out=4, in=3.
    let num_experts = 2usize;
    let out = 4i32;
    let in_dim = 3i32;
    let mut weights = WeightMap::new();
    weights.insert(
        "lm_head.weight".to_string(),
        mlxcel_core::from_slice_f32(&[0.0; 8], &[2, 4]),
    );
    for proj in ["gate_proj", "up_proj", "down_proj"] {
        for e in 0..num_experts {
            weights.insert(
                format!("model.layers.0.mlp.experts.{e}.{proj}.weight"),
                mlxcel_core::from_slice_f32(&vec![0.0; (out * in_dim) as usize], &[out, in_dim]),
            );
        }
    }

    let mut args: ModelArgs = serde_json::from_str(MELLUM_12B_CONFIG).unwrap();
    args.num_hidden_layers = 1;
    args.num_experts = num_experts;
    args.tie_word_embeddings = true;

    MellumModel::sanitize(&mut weights, &args);

    // Tied head popped.
    assert!(!weights.contains_key("lm_head.weight"));

    for proj in ["gate_proj", "up_proj", "down_proj"] {
        let stacked = format!("model.layers.0.mlp.switch_mlp.{proj}.weight");
        let arr = weights
            .get(&stacked)
            .unwrap_or_else(|| panic!("missing stacked {stacked}"));
        assert_eq!(
            mlxcel_core::array_shape(arr),
            vec![num_experts as i32, out, in_dim],
            "stacked {proj} shape",
        );
        // Per-expert tensors removed after stacking.
        for e in 0..num_experts {
            assert!(
                !weights.contains_key(&format!("model.layers.0.mlp.experts.{e}.{proj}.weight")),
                "per-expert {proj} for e{e} should be consumed",
            );
        }
    }
}

/// Minimal valid Mellum config for building a tiny model end to end.
fn tiny_config(tie: bool) -> ModelArgs {
    let json = format!(
        r#"{{
            "model_type": "mellum",
            "hidden_size": 8,
            "head_dim": 4,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "rms_norm_eps": 1e-06,
            "vocab_size": 8,
            "intermediate_size": 8,
            "moe_intermediate_size": 4,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "norm_topk_prob": true,
            "tie_word_embeddings": {tie},
            "max_position_embeddings": 131072,
            "sliding_window": 16,
            "layer_types": [
                "sliding_attention", "sliding_attention",
                "sliding_attention", "full_attention"
            ],
            "rope_parameters": {{
                "full_attention": {{
                    "rope_type": "yarn",
                    "rope_theta": 500000.0,
                    "factor": 16.0,
                    "original_max_position_embeddings": 8192,
                    "beta_fast": 32.0,
                    "beta_slow": 1.0
                }},
                "sliding_attention": {{
                    "rope_type": "default",
                    "rope_theta": 500000.0
                }}
            }}
        }}"#
    );
    serde_json::from_str(&json).expect("parse tiny mellum config")
}

fn zeros(shape: &[i32]) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let n: i32 = shape.iter().product();
    mlxcel_core::from_slice_f32(&vec![0.0; n as usize], shape)
}

fn tiny_weights(args: &ModelArgs) -> WeightMap {
    let h = args.hidden_size as i32;
    let hd = args.head_dim as i32;
    let q_out = args.num_attention_heads as i32 * hd;
    let kv_out = args.num_key_value_heads as i32 * hd;
    let moe = args.moe_intermediate_size as i32;
    let vocab = args.vocab_size as i32;

    let mut w = WeightMap::new();
    w.insert("model.embed_tokens.weight".into(), zeros(&[vocab, h]));
    w.insert("model.norm.weight".into(), zeros(&[h]));
    if !args.tie_word_embeddings {
        w.insert("lm_head.weight".into(), zeros(&[vocab, h]));
    }

    for l in 0..args.num_hidden_layers {
        let p = format!("model.layers.{l}");
        w.insert(format!("{p}.self_attn.q_proj.weight"), zeros(&[q_out, h]));
        w.insert(format!("{p}.self_attn.k_proj.weight"), zeros(&[kv_out, h]));
        w.insert(format!("{p}.self_attn.v_proj.weight"), zeros(&[kv_out, h]));
        w.insert(format!("{p}.self_attn.o_proj.weight"), zeros(&[h, q_out]));
        w.insert(format!("{p}.self_attn.q_norm.weight"), zeros(&[hd]));
        w.insert(format!("{p}.self_attn.k_norm.weight"), zeros(&[hd]));
        w.insert(format!("{p}.input_layernorm.weight"), zeros(&[h]));
        w.insert(format!("{p}.post_attention_layernorm.weight"), zeros(&[h]));
        w.insert(
            format!("{p}.mlp.gate.weight"),
            zeros(&[args.num_experts as i32, h]),
        );
        for e in 0..args.num_experts {
            w.insert(
                format!("{p}.mlp.experts.{e}.gate_proj.weight"),
                zeros(&[moe, h]),
            );
            w.insert(
                format!("{p}.mlp.experts.{e}.up_proj.weight"),
                zeros(&[moe, h]),
            );
            w.insert(
                format!("{p}.mlp.experts.{e}.down_proj.weight"),
                zeros(&[h, moe]),
            );
        }
    }
    w
}

#[test]
fn mellum_make_caches_selects_rotating_for_sliding_and_standard_for_full() {
    let args = tiny_config(false);
    let weights = tiny_weights(&args);
    let model = MellumModel::from_weights(&weights, &args).expect("build tiny mellum");

    assert!(model.lm_head.is_some(), "untied model has a real lm_head");

    let caches = model.make_caches();
    assert_eq!(caches.len(), 4);
    // layer_types: sliding, sliding, sliding, full.
    assert!(matches!(caches[0], Cache::Rotating(_)));
    assert!(matches!(caches[1], Cache::Rotating(_)));
    assert!(matches!(caches[2], Cache::Rotating(_)));
    assert!(matches!(caches[3], Cache::Standard(_)));
}

#[test]
fn mellum_tied_model_has_no_lm_head() {
    let args = tiny_config(true);
    let weights = tiny_weights(&args);
    let model = MellumModel::from_weights(&weights, &args).expect("build tied tiny mellum");
    assert!(
        model.lm_head.is_none(),
        "tied model projects through embed_tokens.as_linear"
    );
}
