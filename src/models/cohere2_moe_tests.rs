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

//! Unit tests for Command MoE (Cohere2 MoE).
//!
//! The checkpoint-free surface (config defaults + alias resolution, layer
//! classification, RoPE gating, gate/combine resolution, EOS parsing) runs in
//! the default `cargo test` pass. The MLX-op surface (router scoring, the
//! shared-expert combine, and a tiny end-to-end forward) is `#[ignore]`d and
//! serialized through a shared guard, matching the `olmoe_tests.rs` /
//! `gemma3n_helpers_tests.rs` convention; run those with `--ignored`.
//!
//! Real-checkpoint load + generate (CLI and the OpenAI-compatible server) is the
//! completion bar and is validated separately once a `cohere2_moe` checkpoint is
//! available.

use super::{
    Cohere2MoeConfig, Cohere2MoeModel, EosTokenId, combine_shared_expert, router_topk_scores,
};
use mlxcel_core::weights::WeightMap;
use std::sync::{Mutex, OnceLock};

/// Serialize the MLX-touching tests: MLX evaluation is not safe to run
/// concurrently from multiple test threads.
fn test_guard() -> &'static Mutex<()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD.get_or_init(|| Mutex::new(()))
}

// Config parsing, defaults, and alias resolution (pure).

/// Only `model_type` is present; every other key must fall back to its default.
const MINIMAL_CONFIG: &str = r#"{ "model_type": "cohere2_moe" }"#;

#[test]
fn config_defaults_apply_for_absent_keys() {
    let cfg: Cohere2MoeConfig = serde_json::from_str(MINIMAL_CONFIG).expect("parse minimal config");

    assert_eq!(cfg.hidden_size, 1024);
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.num_hidden_layers, 36);
    assert_eq!(cfg.intermediate_size, 1024);
    assert_eq!(cfg.num_attention_heads, 64);
    assert_eq!(cfg.num_key_value_heads, 8);
    assert_eq!(cfg.rope_theta, 50000.0);
    assert_eq!(cfg.vocab_size, 256000);
    assert_eq!(cfg.layer_norm_eps, 1e-5);
    assert!(cfg.rms_norm_eps.is_none());
    assert_eq!(cfg.logit_scale, 0.0625);
    assert!(!cfg.attention_bias);
    assert!(!cfg.layer_norm_bias);
    assert_eq!(cfg.sliding_window, 4096);
    assert_eq!(cfg.sliding_window_pattern, 4);
    assert!(cfg.layer_types.is_none());
    assert_eq!(cfg.num_experts, 128);
    assert_eq!(cfg.num_experts_per_tok, 8);
    assert!(cfg.norm_topk_prob);
    assert_eq!(cfg.moe_gate_act, "sigmoid");
    assert_eq!(cfg.moe_num_shared_experts, 4);
    assert_eq!(cfg.shared_expert_combination_strategy, "average");
    assert_eq!(cfg.first_k_dense_replace, 0);
    assert_eq!(cfg.prefix_dense_sliding_window_pattern, 1);

    // Resolved derivations.
    assert_eq!(cfg.shared_expert_count(), 4);
    assert_eq!(cfg.prefix_dense_intermediate(), 1024);
    assert!(cfg.gate_is_sigmoid());
    assert!(cfg.shared_combine_average());
    assert_eq!(cfg.group_size(), 64);
    assert_eq!(cfg.bits(), 4);
}

#[test]
fn alias_keys_override_canonical_keys() {
    let json = r#"{
        "model_type": "cohere2_moe",
        "moe_num_shared_experts": 4,
        "num_shared_experts": 1,
        "moe_gate_act": "sigmoid",
        "expert_selection_fn": "softmax"
    }"#;
    let cfg: Cohere2MoeConfig = serde_json::from_str(json).expect("parse alias config");

    // num_shared_experts overrides moe_num_shared_experts.
    assert_eq!(cfg.shared_expert_count(), 1);
    // expert_selection_fn overrides moe_gate_act.
    assert!(!cfg.gate_is_sigmoid());
}

#[test]
fn prefix_dense_intermediate_falls_back_to_intermediate_size() {
    // Absent: falls back to intermediate_size.
    let json = r#"{ "model_type": "cohere2_moe", "intermediate_size": 777 }"#;
    let cfg: Cohere2MoeConfig = serde_json::from_str(json).unwrap();
    assert_eq!(cfg.prefix_dense_intermediate(), 777);

    // Present: used as-is.
    let json = r#"{ "model_type": "cohere2_moe", "intermediate_size": 777, "prefix_dense_intermediate_size": 42 }"#;
    let cfg: Cohere2MoeConfig = serde_json::from_str(json).unwrap();
    assert_eq!(cfg.prefix_dense_intermediate(), 42);
}

#[test]
fn quantization_block_parses_and_resolves() {
    let json = r#"{
        "model_type": "cohere2_moe",
        "quantization": { "group_size": 32, "bits": 8 }
    }"#;
    let cfg: Cohere2MoeConfig = serde_json::from_str(json).expect("parse quantized config");
    assert_eq!(cfg.group_size(), 32);
    assert_eq!(cfg.bits(), 8);
}

#[test]
fn shared_combine_strategy_resolves() {
    let base = |strategy: &str| {
        let json = format!(
            r#"{{ "model_type": "cohere2_moe", "shared_expert_combination_strategy": "{strategy}" }}"#
        );
        serde_json::from_str::<Cohere2MoeConfig>(&json).unwrap()
    };
    assert!(base("average").shared_combine_average());
    assert!(!base("sum").shared_combine_average());
}

#[test]
fn eos_token_parses_single_list_and_falls_back() {
    // Single int.
    let cfg: Cohere2MoeConfig =
        serde_json::from_str(r#"{ "model_type": "cohere2_moe", "eos_token_id": 7 }"#).unwrap();
    assert!(matches!(cfg.eos_token_id, Some(EosTokenId::Single(7))));

    // List.
    let cfg: Cohere2MoeConfig =
        serde_json::from_str(r#"{ "model_type": "cohere2_moe", "eos_token_id": [1, 2, 3] }"#)
            .unwrap();
    assert!(matches!(cfg.eos_token_id, Some(EosTokenId::Multiple(_))));

    // Absent -> the model reports the 255001 fallback (checked in the forward
    // test through eos_token_ids()).
    let cfg: Cohere2MoeConfig = serde_json::from_str(MINIMAL_CONFIG).unwrap();
    assert!(cfg.eos_token_id.is_none());
}

// Layer classification and RoPE gating (pure).

fn config_from(json: &str) -> Cohere2MoeConfig {
    serde_json::from_str(json).expect("parse config")
}

#[test]
fn layer_classification_default_pattern() {
    // Default pattern 4, no dense prefix, no layer_types: sliding iff
    // (i + 1) % 4 != 0, so layers 0,1,2 sliding, 3 global, 4,5,6 sliding, 7
    // global, ...
    let cfg = config_from(
        r#"{ "model_type": "cohere2_moe", "num_hidden_layers": 8, "sliding_window_pattern": 4 }"#,
    );
    let expected_sliding = [true, true, true, false, true, true, true, false];
    for (i, &want) in expected_sliding.iter().enumerate() {
        assert_eq!(
            cfg.is_sliding_window_layer(i),
            want,
            "layer {i} sliding flag"
        );
    }
}

#[test]
fn layer_classification_layer_types_override_pattern() {
    // layer_types present: it drives sliding vs global, ignoring the pattern.
    let cfg = config_from(
        r#"{
            "model_type": "cohere2_moe",
            "num_hidden_layers": 4,
            "sliding_window_pattern": 4,
            "layer_types": ["full_attention", "sliding_attention", "full_attention", "sliding_attention"]
        }"#,
    );
    assert!(!cfg.is_sliding_window_layer(0));
    assert!(cfg.is_sliding_window_layer(1));
    assert!(!cfg.is_sliding_window_layer(2));
    assert!(cfg.is_sliding_window_layer(3));
}

#[test]
fn first_k_dense_replace_prefix_is_always_global() {
    // The first `first_k_dense_replace` layers are global regardless of the
    // pattern that would otherwise make them sliding.
    let cfg = config_from(
        r#"{
            "model_type": "cohere2_moe",
            "num_hidden_layers": 6,
            "sliding_window_pattern": 4,
            "first_k_dense_replace": 2
        }"#,
    );
    // Layers 0,1 would be sliding under the pattern, but the dense prefix forces
    // global.
    assert!(!cfg.is_sliding_window_layer(0));
    assert!(!cfg.is_sliding_window_layer(1));
    assert!(cfg.is_dense_layer(0));
    assert!(cfg.is_dense_layer(1));
    assert!(!cfg.is_dense_layer(2));
    // Layer 2 onwards resumes the pattern: (2+1)%4 != 0 -> sliding.
    assert!(cfg.is_sliding_window_layer(2));
}

#[test]
fn rope_gating_forces_rope_on_prefix_dense_and_follows_sliding() {
    let cfg = config_from(
        r#"{
            "model_type": "cohere2_moe",
            "num_hidden_layers": 6,
            "sliding_window_pattern": 4,
            "first_k_dense_replace": 2,
            "prefix_dense_sliding_window_pattern": 1
        }"#,
    );
    // Prefix dense layers are global but still get RoPE (forced by
    // prefix_dense_sliding_window_pattern == 1).
    assert!(cfg.layer_uses_rope(0));
    assert!(cfg.layer_uses_rope(1));
    // Layer 2: sliding -> RoPE.
    assert!(cfg.is_sliding_window_layer(2));
    assert!(cfg.layer_uses_rope(2));
    // Layer 3: (3+1)%4 == 0 -> global, non-prefix -> NO RoPE.
    assert!(!cfg.is_sliding_window_layer(3));
    assert!(!cfg.layer_uses_rope(3));
}

#[test]
fn prefix_dense_without_forced_rope_uses_no_positional_encoding() {
    // prefix_dense_sliding_window_pattern != 1 -> prefix dense global layers get
    // no RoPE.
    let cfg = config_from(
        r#"{
            "model_type": "cohere2_moe",
            "num_hidden_layers": 4,
            "first_k_dense_replace": 1,
            "prefix_dense_sliding_window_pattern": 4
        }"#,
    );
    assert!(cfg.is_dense_layer(0));
    assert!(!cfg.is_sliding_window_layer(0));
    assert!(!cfg.layer_uses_rope(0));
}

// Router scoring (MLX; ignored + serialized).

// Four-expert, top-2 router logits. logits = [1, 3, 2, 0].
const LOGITS: [f32; 4] = [1.0, 3.0, 2.0, 0.0];
// Elementwise sigmoid of the logits.
const SIGMOID: [f32; 4] = [0.7310586, 0.9525741, 0.8807971, 0.5];
// Full softmax over all four experts.
const FULL_SOFTMAX: [f32; 4] = [0.0871443, 0.6439142, 0.2368828, 0.0320586];

/// Read the selected expert indices back as a sorted Vec so assertions are
/// independent of argpartition's internal ordering within the top-k.
fn selected_index_set(topk_indices: &mlxcel_core::MlxArray) -> Vec<i32> {
    let idx = mlxcel_core::astype(topk_indices, mlxcel_core::dtype::INT32);
    let i0 = mlxcel_core::item_i32(&mlxcel_core::slice(&idx, &[0, 0], &[1, 1]));
    let i1 = mlxcel_core::item_i32(&mlxcel_core::slice(&idx, &[0, 1], &[1, 2]));
    let mut got = vec![i0, i1];
    got.sort();
    got
}

#[test]
#[ignore = "requires serial MLX execution"]
fn sigmoid_gate_gathers_activated_scores_not_a_fresh_softmax() {
    let _guard = test_guard().lock().unwrap();
    let logits = mlxcel_core::from_slice_f32(&LOGITS, &[1, 4]);
    // sigmoid gate, no renorm.
    let (topk_indices, scores) = router_topk_scores(&logits, 2, true, false);

    // Top-2 by logit are experts 1 and 2.
    assert_eq!(selected_index_set(&topk_indices), vec![1, 2]);

    // Scores must be the sigmoid values gathered at the selected experts, NOT a
    // fresh softmax over the two selected logits. Gather the hand-computed
    // sigmoid with the same indices so the comparison is order-agnostic.
    let sig = mlxcel_core::from_slice_f32(&SIGMOID, &[1, 4]);
    let expected = mlxcel_core::take_along_axis(&sig, &topk_indices, -1);
    let close = mlxcel_core::allclose(&scores, &expected, 1e-4, 1e-5);
    mlxcel_core::eval(&close);
    assert!(
        mlxcel_core::item_bool(&close),
        "sigmoid scores must equal the elementwise sigmoid gathered at the top-k"
    );

    // The two sigmoid values sum to 1.8333712 (> 1); a fresh top-k softmax would
    // have summed to exactly 1.
    let sum = mlxcel_core::sum_axis(&scores, -1, true);
    mlxcel_core::eval(&sum);
    let sum = mlxcel_core::item_f32(&sum);
    assert!(
        (sum - 1.8333712).abs() < 1e-3,
        "unexpected sigmoid score sum: {sum}"
    );
    assert!(
        sum > 1.0,
        "sigmoid top-k scores must not sum to 1, got {sum}"
    );
}

#[test]
#[ignore = "requires serial MLX execution"]
fn softmax_gate_gathers_full_softmax_probs() {
    let _guard = test_guard().lock().unwrap();
    let logits = mlxcel_core::from_slice_f32(&LOGITS, &[1, 4]);
    // softmax gate, no renorm.
    let (topk_indices, scores) = router_topk_scores(&logits, 2, false, false);

    assert_eq!(selected_index_set(&topk_indices), vec![1, 2]);

    let full = mlxcel_core::from_slice_f32(&FULL_SOFTMAX, &[1, 4]);
    let expected = mlxcel_core::take_along_axis(&full, &topk_indices, -1);
    let close = mlxcel_core::allclose(&scores, &expected, 1e-4, 1e-5);
    mlxcel_core::eval(&close);
    assert!(
        mlxcel_core::item_bool(&close),
        "softmax scores must equal the full softmax gathered at the top-k"
    );

    // Full-softmax probs at the top-2 sum to 0.880797 (< 1), NOT to 1.
    let sum = mlxcel_core::sum_axis(&scores, -1, true);
    mlxcel_core::eval(&sum);
    let sum = mlxcel_core::item_f32(&sum);
    assert!(
        (sum - 0.880797).abs() < 1e-3,
        "unexpected softmax score sum: {sum}"
    );
    assert!(
        sum < 0.99,
        "un-normalized softmax scores must sum to < 1, got {sum}"
    );
}

#[test]
#[ignore = "requires serial MLX execution"]
fn norm_topk_prob_renormalizes_gathered_scores() {
    let _guard = test_guard().lock().unwrap();
    let logits = mlxcel_core::from_slice_f32(&LOGITS, &[1, 4]);
    // sigmoid gate WITH renorm.
    let (topk_indices, scores) = router_topk_scores(&logits, 2, true, true);

    // Expected = sigmoid gathered at the top-k, divided by its own sum.
    let sig = mlxcel_core::from_slice_f32(&SIGMOID, &[1, 4]);
    let gathered = mlxcel_core::take_along_axis(&sig, &topk_indices, -1);
    let gathered_sum = mlxcel_core::sum_axis(&gathered, -1, true);
    let expected = mlxcel_core::divide(&gathered, &gathered_sum);
    let close = mlxcel_core::allclose(&scores, &expected, 1e-4, 1e-5);
    mlxcel_core::eval(&close);
    assert!(
        mlxcel_core::item_bool(&close),
        "renormalized scores must equal gathered sigmoid / its sum"
    );

    // Renormalized scores sum to 1.
    let sum = mlxcel_core::sum_axis(&scores, -1, true);
    mlxcel_core::eval(&sum);
    let sum = mlxcel_core::item_f32(&sum);
    assert!(
        (sum - 1.0).abs() < 1e-4,
        "renormalized scores must sum to 1, got {sum}"
    );
}

#[test]
#[ignore = "requires serial MLX execution"]
fn norm_topk_prob_clamp_avoids_nan_when_all_scores_underflow() {
    let _guard = test_guard().lock().unwrap();
    // Logits so negative that sigmoid underflows to exactly 0 in f32 (e^100
    // overflows). The two largest (experts 0, 1) are both -100, so the gathered
    // sigmoid sum is 0. Without the 1e-12 denominator clamp this is 0/0 = NaN;
    // with the clamp the scores stay finite (all 0).
    let logits = mlxcel_core::from_slice_f32(&[-100.0, -100.0, -300.0, -300.0], &[1, 4]);
    let (_topk_indices, scores) = router_topk_scores(&logits, 2, true, true);

    let fin = mlxcel_core::isfinite(&scores);
    let fin_f = mlxcel_core::astype(&fin, mlxcel_core::dtype::FLOAT32);
    let min_fin = mlxcel_core::min_all(&fin_f);
    mlxcel_core::eval(&min_fin);
    assert!(
        mlxcel_core::item_f32(&min_fin) >= 0.5,
        "the 1e-12 clamp must keep all renormalized scores finite"
    );
}

// Shared-expert combine (MLX; ignored + serialized).

#[test]
#[ignore = "requires serial MLX execution"]
fn shared_expert_average_halves_the_sum_but_sum_does_not() {
    let _guard = test_guard().lock().unwrap();
    let routed = mlxcel_core::from_slice_f32(&[2.0, 4.0], &[1, 2]);
    let shared = mlxcel_core::from_slice_f32(&[4.0, 6.0], &[1, 2]);

    // "average": (y + y_s) / 2 = [3, 5].
    let avg = combine_shared_expert(&routed, &shared, true);
    let want_avg = mlxcel_core::from_slice_f32(&[3.0, 5.0], &[1, 2]);
    let close = mlxcel_core::allclose(&avg, &want_avg, 1e-5, 1e-6);
    mlxcel_core::eval(&close);
    assert!(mlxcel_core::item_bool(&close), "average must halve the sum");

    // "sum": y + y_s = [6, 10].
    let sum = combine_shared_expert(&routed, &shared, false);
    let want_sum = mlxcel_core::from_slice_f32(&[6.0, 10.0], &[1, 2]);
    let close = mlxcel_core::allclose(&sum, &want_sum, 1e-5, 1e-6);
    mlxcel_core::eval(&close);
    assert!(mlxcel_core::item_bool(&close), "sum must not halve");
}

// Synthetic end-to-end forward (MLX; ignored + serialized).

/// A tiny config: 4 layers, first_k_dense_replace = 1, 4 experts top-2, shared
/// experts enabled, sliding_window_pattern = 2. `gate_act` in
/// {"sigmoid","softmax"}, `strategy` in {"average","sum"}.
fn tiny_config(gate_act: &str, strategy: &str) -> Cohere2MoeConfig {
    let json = format!(
        r#"{{
            "model_type": "cohere2_moe",
            "hidden_size": 8,
            "head_dim": 4,
            "num_hidden_layers": 4,
            "intermediate_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "rope_theta": 10000.0,
            "vocab_size": 10,
            "layer_norm_eps": 1e-5,
            "logit_scale": 0.0625,
            "sliding_window": 16,
            "sliding_window_pattern": 2,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "norm_topk_prob": true,
            "moe_gate_act": "{gate_act}",
            "moe_num_shared_experts": 2,
            "shared_expert_combination_strategy": "{strategy}",
            "first_k_dense_replace": 1
        }}"#
    );
    serde_json::from_str(&json).expect("parse tiny cohere2_moe config")
}

/// Deterministic small non-zero weights so activations stay bounded and finite.
fn small(shape: &[i32], phase: usize) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let n: i32 = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|i| ((i as usize + phase) as f32).sin() * 0.05)
        .collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

fn ones(shape: &[i32]) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let n: i32 = shape.iter().product();
    mlxcel_core::from_slice_f32(&vec![1.0; n as usize], shape)
}

fn tiny_weights(cfg: &Cohere2MoeConfig) -> WeightMap {
    let h = cfg.hidden_size as i32;
    let hd = cfg.head_dim as i32;
    let q_out = cfg.num_attention_heads as i32 * hd;
    let kv_out = cfg.num_key_value_heads as i32 * hd;
    let inter = cfg.intermediate_size as i32;
    let prefix_inter = cfg.prefix_dense_intermediate() as i32;
    let shared_inter = inter * cfg.shared_expert_count() as i32;
    let vocab = cfg.vocab_size as i32;

    let mut w = WeightMap::new();
    let mut phase = 1usize;
    let mut next = || {
        phase += 7;
        phase
    };

    w.insert(
        "model.embed_tokens.weight".into(),
        small(&[vocab, h], next()),
    );
    w.insert("model.norm.weight".into(), ones(&[h]));

    for l in 0..cfg.num_hidden_layers {
        let p = format!("model.layers.{l}");
        w.insert(
            format!("{p}.self_attn.q_proj.weight"),
            small(&[q_out, h], next()),
        );
        w.insert(
            format!("{p}.self_attn.k_proj.weight"),
            small(&[kv_out, h], next()),
        );
        w.insert(
            format!("{p}.self_attn.v_proj.weight"),
            small(&[kv_out, h], next()),
        );
        w.insert(
            format!("{p}.self_attn.o_proj.weight"),
            small(&[h, q_out], next()),
        );
        w.insert(format!("{p}.input_layernorm.weight"), ones(&[h]));

        if cfg.is_dense_layer(l) {
            w.insert(
                format!("{p}.mlp.gate_proj.weight"),
                small(&[prefix_inter, h], next()),
            );
            w.insert(
                format!("{p}.mlp.up_proj.weight"),
                small(&[prefix_inter, h], next()),
            );
            w.insert(
                format!("{p}.mlp.down_proj.weight"),
                small(&[h, prefix_inter], next()),
            );
        } else {
            w.insert(
                format!("{p}.mlp.gate.weight"),
                small(&[cfg.num_experts as i32, h], next()),
            );
            for e in 0..cfg.num_experts {
                w.insert(
                    format!("{p}.mlp.experts.{e}.gate_proj.weight"),
                    small(&[inter, h], next()),
                );
                w.insert(
                    format!("{p}.mlp.experts.{e}.up_proj.weight"),
                    small(&[inter, h], next()),
                );
                w.insert(
                    format!("{p}.mlp.experts.{e}.down_proj.weight"),
                    small(&[h, inter], next()),
                );
            }
            if cfg.shared_expert_count() > 0 {
                w.insert(
                    format!("{p}.mlp.shared_experts.gate_proj.weight"),
                    small(&[shared_inter, h], next()),
                );
                w.insert(
                    format!("{p}.mlp.shared_experts.up_proj.weight"),
                    small(&[shared_inter, h], next()),
                );
                w.insert(
                    format!("{p}.mlp.shared_experts.down_proj.weight"),
                    small(&[h, shared_inter], next()),
                );
            }
        }
    }
    w
}

fn all_finite(x: &mlxcel_core::MlxArray) -> bool {
    let fin = mlxcel_core::isfinite(x);
    let fin_f = mlxcel_core::astype(&fin, mlxcel_core::dtype::FLOAT32);
    let m = mlxcel_core::min_all(&fin_f);
    mlxcel_core::eval(&m);
    mlxcel_core::item_f32(&m) >= 0.5
}

#[test]
#[ignore = "requires serial MLX execution"]
fn tiny_model_builds_with_expected_layer_topology() {
    let _guard = test_guard().lock().unwrap();
    let cfg = tiny_config("sigmoid", "average");
    let weights = tiny_weights(&cfg);
    let model = Cohere2MoeModel::from_weights(&weights, &cfg).expect("build tiny model");

    use mlxcel_core::generate::LanguageModel;
    assert_eq!(LanguageModel::num_layers(&model), 4);
    // No eos_token_id in the tiny config -> the 255001 Cohere fallback.
    assert_eq!(LanguageModel::eos_token_ids(&model), vec![255001]);
    assert_eq!(model.make_caches().len(), 4);
}

#[test]
#[ignore = "requires serial MLX execution"]
fn tiny_model_forward_is_finite_and_prefill_matches_decode() {
    let _guard = test_guard().lock().unwrap();
    use mlxcel_core::generate::LanguageModel;

    for gate_act in ["sigmoid", "softmax"] {
        for strategy in ["average", "sum"] {
            let cfg = tiny_config(gate_act, strategy);
            let vocab = cfg.vocab_size as i32;
            let weights = tiny_weights(&cfg);
            let model = Cohere2MoeModel::from_weights(&weights, &cfg)
                .unwrap_or_else(|e| panic!("build tiny model ({gate_act}/{strategy}): {e}"));

            let seq = [1i32, 3, 5];

            // Full prefill of the 3-token sequence.
            let mut caches = model.make_caches();
            let ids = mlxcel_core::from_slice_i32(&seq, &[1, 3]);
            let prefill = model.forward_impl(&ids, &mut caches, None);
            mlxcel_core::eval(&prefill);
            assert_eq!(
                mlxcel_core::array_shape(&prefill),
                vec![1, 3, vocab],
                "prefill logits shape ({gate_act}/{strategy})"
            );
            assert!(
                all_finite(&prefill),
                "prefill logits must be finite ({gate_act}/{strategy})"
            );
            let last_prefill = mlxcel_core::slice(&prefill, &[0, 2, 0], &[1, 3, vocab]);

            // Token-by-token decode through a fresh cache.
            let mut caches2 = model.make_caches();
            let mut last_decode = None;
            for &t in &seq {
                let id = mlxcel_core::from_slice_i32(&[t], &[1, 1]);
                let out = model.forward_impl(&id, &mut caches2, None);
                mlxcel_core::eval(&out);
                assert_eq!(
                    mlxcel_core::array_shape(&out),
                    vec![1, 1, vocab],
                    "decode logits shape ({gate_act}/{strategy})"
                );
                last_decode = Some(out);
            }
            let last_decode = last_decode.unwrap();
            assert!(
                all_finite(&last_decode),
                "decode logits must be finite ({gate_act}/{strategy})"
            );

            // KV-cache consistency: the last prefill position matches the final
            // decode step.
            let close = mlxcel_core::allclose(&last_prefill, &last_decode, 2e-3, 2e-3);
            mlxcel_core::eval(&close);
            assert!(
                mlxcel_core::item_bool(&close),
                "prefill last position must match decode ({gate_act}/{strategy})"
            );
        }
    }
}
