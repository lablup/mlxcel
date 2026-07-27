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

//! Unit tests for the Bailing MoE loader, its router, its two checkpoint
//! spellings and its config guards.
//!
//! Everything here is checkpoint-free. The config tests parse the verbatim
//! `inclusionAI/Ling-lite-1.5` `config.json`; the shape tests build synthetic
//! weight maps whose tensor names and shapes mirror the real export (a
//! full-size one for the positive control, built from lazy MLX arrays that are
//! never evaluated, so no weight data is read or allocated); the router tests
//! drive `BailingMoeGate` directly with an identity `gate_proj`, so the input
//! row *is* the router logit row.
//!
//! Four groups carry more weight than the rest, because the real-checkpoint gate
//! cannot see what they cover:
//!
//! 1. **Grouped routing** (`n_group > 1`) and **expert bias**
//!    (`moe_router_enable_expert_bias`) are exercised by no published Bailing
//!    checkpoint: `Ling-lite-1.5` declares neither field, and both default to
//!    off. A token-exact real-model run says nothing about either branch, so
//!    they are pinned here against hand-computed values instead.
//! 2. **Selection uses the biased scores while the returned weights come from
//!    the unbiased ones.** Gathering from the biased copy leaves the output
//!    finite and plausible while misweighting every routed contribution, so
//!    `selection_uses_the_biased_scores_while_the_weights_come_from_the_unbiased_ones`
//!    constructs a bias large enough to change the *selected set* and then
//!    asserts the returned weights are the unbiased scores at those indices, and
//!    asserts what the wrong gather would have produced.
//! 3. **The router rename collides with the expert projection names.** After
//!    upstream's `sanitize` the router lives at a key ending in `gate_proj`,
//!    which every routed expert also has.
//!    `router_lookup_is_anchored_against_the_expert_gate_proj` puts both tensors
//!    in one map and pins which one is read.
//! 4. **The config guards** exist because an MLX C++ exception crossing the cxx
//!    bridge is an uncatchable `std::terminate` at the first forward pass rather
//!    than a load error, so every rejection has a matching acceptance and the
//!    real config is asserted to survive all of them.

use super::{
    BailingMoeGate, BailingMoeMLP, BailingMoeModel, BailingMoeSparseBlock, FeedForward, ModelArgs,
    Quantization, ScoreFunction, TokenIdField, load_lm_head, normalize_lm_head_weight,
    router_prefix, validate_weights,
};
use crate::models::switch_layers::group_mask_scores;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

// The real checkpoint's config.

/// `inclusionAI/Ling-lite-1.5`'s `config.json`, field for field.
///
/// Reproduced verbatim, including the keys this loader ignores, because the
/// point of the parse test is that serde accepts the file exactly as shipped.
const LING_LITE_CONFIG: &str = r#"{
    "architectures": [
        "BailingMoeForCausalLM"
    ],
    "attention_dropout": 0.0,
    "auto_map": {
        "AutoConfig": "configuration_bailing_moe.BailingMoeConfig",
        "AutoModel": "modeling_bailing_moe.BailingMoeModel",
        "AutoModelForCausalLM": "modeling_bailing_moe.BailingMoeForCausalLM"
    },
    "eos_token_id": 126081,
    "pad_token_id": 126081,
    "first_k_dense_replace": 0,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.006,
    "intermediate_size": 1408,
    "max_position_embeddings": 32768,
    "model_type": "bailing_moe",
    "moe_intermediate_size": 1408,
    "num_experts": 64,
    "num_shared_experts": 2,
    "norm_topk_prob": true,
    "num_attention_heads": 16,
    "num_experts_per_tok": 6,
    "num_hidden_layers": 28,
    "num_key_value_heads": 4,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-06,
    "rope_scaling": null,
    "rope_theta": 600000,
    "tie_word_embeddings": false,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.40.0",
    "use_cache": true,
    "use_bias": false,
    "use_qkv_bias": false,
    "vocab_size": 126464,
    "output_router_logits": false,
    "embedding_dropout": 0.0,
    "norm_head": false,
    "norm_softmax": false,
    "output_dropout": 0.0
}"#;

fn ling_lite() -> ModelArgs {
    serde_json::from_str(LING_LITE_CONFIG).expect("the Ling-lite-1.5 config parses")
}

#[test]
fn parses_the_real_ling_lite_config() {
    let args = ling_lite();

    assert_eq!(args.model_type, "bailing_moe");
    assert_eq!(args.hidden_size, 2048);
    assert_eq!(args.num_hidden_layers, 28);
    assert_eq!(args.num_attention_heads, 16);
    assert_eq!(args.num_key_value_heads, Some(4));
    assert_eq!(args.num_kv_heads(), 4);
    assert_eq!(args.intermediate_size, 1408);
    assert_eq!(args.moe_intermediate_size, Some(1408));
    assert_eq!(args.moe_intermediate_size(), 1408);
    assert_eq!(args.num_experts, Some(64));
    assert_eq!(args.num_experts(), 64);
    assert_eq!(args.num_shared_experts, 2);
    assert_eq!(args.num_experts_per_tok, 6);
    assert!(args.norm_topk_prob);
    assert_eq!(args.first_k_dense_replace, 0);
    assert_eq!(args.max_position_embeddings, 32_768);
    assert_eq!(args.vocab_size, 126_464);
    assert!(!args.use_bias);
    assert!(!args.use_qkv_bias);
    assert!(!args.norm_head);
    assert!(!args.norm_softmax);
    assert!(!args.use_qk_norm);
    assert!(!args.tie_word_embeddings);
    assert!((args.rms_norm_eps - 1e-6).abs() < 1e-12);
    assert!((args.rope_theta - 600_000.0).abs() < 1e-3);
    // `"rope_scaling": null` is an explicit null, not an absent key, and must
    // parse rather than fail the whole file.
    assert!(args.rope_scaling.is_none());
    // A raw `inclusionAI` export carries no `quantization` block.
    assert!(args.quantization.is_none());
    assert_eq!(args.eos_token_ids(), vec![126_081]);

    // Derived geometry.
    assert_eq!(args.head_dim(), 128, "2048 / 16");
    assert!(args.has_routed_experts());
    assert!(args.has_shared_expert());
    // `first_k_dense_replace` is 0, so every one of the 28 layers is sparse and
    // the dense-prefix branch is unreachable on this checkpoint.
    assert!(args.is_moe_layer(0));
    assert!(args.is_moe_layer(27));

    args.validate().expect("the real config must validate");
}

#[test]
fn config_defaults_reproduce_the_upstream_routing_defaults() {
    // `Ling-lite-1.5` declares none of these eight fields. Declaring any of them
    // required would fail the parse of the real file outright, and defaulting
    // any of them differently from upstream changes the routing distribution on
    // every token while leaving the output finite.
    let args = ling_lite();

    assert!(
        !args.moe_router_enable_expert_bias,
        "upstream default False"
    );
    assert!(
        args.moe_router_enable_routed_scaling,
        "upstream default True"
    );
    assert!(
        (args.routed_scaling_factor - 1.0).abs() < 1e-12,
        "default 1.0"
    );
    assert_eq!(
        args.score_function, "softmax",
        "upstream default \"softmax\""
    );
    assert_eq!(args.n_group, 1, "upstream default 1");
    assert_eq!(args.topk_group, 4, "upstream default 4");
    assert_eq!(args.moe_shared_expert_intermediate_size, None);
    assert!(
        args.moe_router_enable_shared_expert,
        "upstream default True"
    );

    // The one that costs the most to get wrong. DeepSeek-V3's router is
    // sigmoid, and this port reuses the DeepSeek MoE machinery, so inheriting
    // its score function would change every routing weight while producing
    // perfectly plausible text.
    assert_eq!(args.score_function(), ScoreFunction::Softmax);
    assert_ne!(args.score_function(), ScoreFunction::Sigmoid);

    // The same defaults hold for a config carrying nothing but the five fields
    // serde genuinely requires, so none of them is being supplied by the real
    // file behind the assertions above.
    let bare: ModelArgs = serde_json::from_str(
        r#"{"hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 4,
            "intermediate_size": 128, "vocab_size": 32}"#,
    )
    .expect("a minimal config parses");
    assert_eq!(bare.model_type, "bailing_moe");
    assert!(!bare.moe_router_enable_expert_bias);
    assert!(bare.moe_router_enable_routed_scaling);
    assert!((bare.routed_scaling_factor - 1.0).abs() < 1e-12);
    assert_eq!(bare.score_function(), ScoreFunction::Softmax);
    assert_eq!(bare.n_group, 1);
    assert_eq!(bare.topk_group, 4);
    assert_eq!(bare.moe_shared_expert_intermediate_size, None);
    assert!(bare.moe_router_enable_shared_expert);
    // No routed experts at all: upstream gates its MoE block on
    // `args.num_experts is not None`, so an absent field is a dense model.
    assert_eq!(bare.num_experts, None);
    assert!(!bare.has_routed_experts());
    assert!(!bare.is_moe_layer(0));
    // `num_shared_experts` defaults to zero, so `moe_router_enable_shared_expert`
    // being true on its own does not build a shared MLP.
    assert!(!bare.has_shared_expert());
    // Absent `partial_rotary_factor` and `rotary_dim` rotate the whole head.
    assert_eq!(bare.rope_dims(), bare.head_dim() as i32);
    bare.validate().expect("a minimal config validates");
}

#[test]
fn token_id_fields_accept_a_scalar_or_a_list() {
    // `Ling-lite-1.5` writes a single int, but serde fails the whole config when
    // one field does not match its declared type, so the list form has to be
    // accepted too.
    let args = ling_lite();
    assert!(matches!(args.eos_token_id, Some(TokenIdField::Single(_))));
    assert_eq!(args.eos_token_ids(), vec![126_081]);

    let listed: ModelArgs = serde_json::from_str(
        r#"{"hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 4,
            "intermediate_size": 128, "vocab_size": 32, "eos_token_id": [126081, 126080]}"#,
    )
    .expect("parses");
    assert!(matches!(
        listed.eos_token_id,
        Some(TokenIdField::Multiple(_))
    ));
    assert_eq!(listed.eos_token_ids(), vec![126_081, 126_080]);

    // No stop token declared is not an error: guessing one could truncate at an
    // ordinary token.
    let bare: ModelArgs = serde_json::from_str(
        r#"{"hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 4,
            "intermediate_size": 128, "vocab_size": 32}"#,
    )
    .expect("parses");
    assert!(bare.eos_token_ids().is_empty());
}

#[test]
fn an_unknown_score_function_is_rejected_rather_than_silently_defaulted() {
    let mut args = ling_lite();

    args.score_function = "sigmoid".to_string();
    assert_eq!(args.score_function(), ScoreFunction::Sigmoid);
    args.validate().expect("sigmoid is a real Bailing option");

    args.score_function = "softmax".to_string();
    assert_eq!(args.score_function(), ScoreFunction::Softmax);
    args.validate().expect("softmax is the default");

    for spelling in ["relu", "Softmax", "", "top_k"] {
        args.score_function = spelling.to_string();
        let err = args
            .validate()
            .unwrap_err_or_panic(&format!("score_function {spelling:?} must be rejected"));
        assert!(err.contains("score_function"), "unhelpful error: {err}");
    }
}

// Derived geometry that no shape assertion downstream can catch.

#[test]
fn the_fused_qkv_split_is_uneven_under_grouped_query_attention() {
    let args = ling_lite();

    // `(16 + 2 * 4) * 128`, which is the real `[3072, 2048]` weight's row count.
    assert_eq!(args.qkv_out_features(), 3072);

    // 16 query heads of width 128 give a 2048-wide Q block; 4 KV heads give a
    // 512-wide K and a 512-wide V block. The offsets are therefore
    // `(q_size, q_size + kv_size)`, not thirds.
    assert_eq!(args.qkv_split_offsets(), (2048, 2560));
    let (k_start, v_start) = args.qkv_split_offsets();
    assert_eq!(k_start, 16 * 128);
    assert_eq!(v_start - k_start, 4 * 128);
    assert_eq!(args.qkv_out_features() - v_start, 4 * 128);

    // An even three-way split would put the boundaries at 1024 and 2048, taking
    // K and V out of the query channels. MLX's `slice` clamps an out-of-range
    // stop rather than throwing, so that mistake produces tensors of plausible
    // shape assembled from the wrong channels.
    let even = args.qkv_out_features() / 3;
    assert_ne!(k_start, even);
    assert_ne!(v_start, 2 * even);

    // Multi-head attention degenerates to the even split, which is why a
    // checkpoint without GQA cannot expose the mistake.
    let mut mha = ling_lite();
    mha.num_key_value_heads = Some(16);
    assert_eq!(mha.qkv_out_features(), 3 * 2048);
    assert_eq!(mha.qkv_split_offsets(), (2048, 4096));
}

#[test]
fn the_shared_expert_is_one_wide_mlp_of_num_shared_experts_times_the_expert_width() {
    // `shared_dim = moe_shared_expert_intermediate_size or moe_intermediate_size`,
    // and the MLP is built at `shared_dim * num_shared_experts`. On
    // `Ling-lite-1.5` that is 1408 * 2 = 2816, which the checkpoint confirms:
    // `mlp.shared_experts.gate_proj.weight` is `[2816, 2048]` and
    // `mlp.shared_experts.down_proj.weight` is `[2048, 2816]`. There is no
    // per-shared-expert axis anywhere.
    let args = ling_lite();
    assert_eq!(args.shared_expert_intermediate_size(), 2816);
    assert_ne!(
        args.shared_expert_intermediate_size(),
        args.moe_intermediate_size(),
        "a single expert-width MLP would be the `num_shared_experts == 1` case"
    );

    // The explicit per-expert width wins over `moe_intermediate_size`.
    let mut explicit = ling_lite();
    explicit.moe_shared_expert_intermediate_size = Some(512);
    assert_eq!(explicit.shared_expert_intermediate_size(), 1024);

    // Built only when there is at least one shared expert AND the flag is set.
    let mut disabled = ling_lite();
    disabled.moe_router_enable_shared_expert = false;
    assert!(!disabled.has_shared_expert());
    let mut none = ling_lite();
    none.num_shared_experts = 0;
    assert!(!none.has_shared_expert());
    assert!(args.has_shared_expert());
}

#[test]
fn the_rotary_width_is_the_whole_head_unless_the_config_narrows_it() {
    let mut args = ling_lite();
    assert_eq!(args.rope_dims(), 128, "the full head width");

    args.partial_rotary_factor = 0.5;
    assert_eq!(args.rope_dims(), 64);
    args.validate().expect("half a 128-wide head is even");

    // An explicit `rotary_dim` wins over the factor entirely.
    args.rotary_dim = Some(32);
    assert_eq!(args.rope_dims(), 32);
    args.validate()
        .expect("32 is even and fits a 128-wide head");
}

#[test]
fn the_dense_prefix_is_the_layers_below_first_k_dense_replace() {
    let mut args = ling_lite();
    args.first_k_dense_replace = 3;
    assert!(!args.is_moe_layer(0));
    assert!(!args.is_moe_layer(2));
    assert!(args.is_moe_layer(3));
    assert!(args.is_moe_layer(27));
    args.validate().expect("3 of 28 layers dense is ordinary");

    // A prefix longer than the stack means config and checkpoint disagree about
    // which layers carry experts at all.
    args.first_k_dense_replace = 29;
    let err = args.validate().unwrap_err_or_panic("prefix past the stack");
    assert!(err.contains("first_k_dense_replace"), "{err}");
}

// The router rename collides with the expert projection names.

#[test]
fn router_prefix_resolves_both_checkpoint_spellings_and_names_both_when_neither_exists() {
    // A raw `inclusionAI` export stores the router at `mlp.gate.weight`.
    let mut raw = WeightMap::new();
    raw.insert("model.layers.0.mlp.gate.weight".into(), ones(&[4, 8]));
    assert_eq!(
        router_prefix(&raw, "model.layers.0.mlp").unwrap(),
        "model.layers.0.mlp.gate"
    );

    // Upstream's `sanitize` renames it into the `gate_proj` spelling.
    let mut converted = WeightMap::new();
    converted.insert(
        "model.layers.0.mlp.gate.gate_proj.weight".into(),
        ones(&[4, 8]),
    );
    assert_eq!(
        router_prefix(&converted, "model.layers.0.mlp").unwrap(),
        "model.layers.0.mlp.gate.gate_proj"
    );

    // Neither: the message has to name both, because which one a reader should
    // expect depends on whether their checkpoint went through mlx-lm.
    let err = router_prefix(&WeightMap::new(), "model.layers.0.mlp")
        .unwrap_err_or_panic("no router weight anywhere");
    assert!(err.contains("model.layers.0.mlp.gate.weight"), "{err}");
    assert!(
        err.contains("model.layers.0.mlp.gate.gate_proj.weight"),
        "{err}"
    );
}

#[test]
fn router_lookup_is_anchored_against_the_expert_gate_proj() {
    // THE collision. After the rename the router lives at a key ending in
    // `gate_proj`, and every routed expert has a `gate_proj` too. A rule that
    // matched on the suffix would read an expert's SwiGLU gate as the router:
    // the model still loads, still generates, and routes on the wrong matrix.
    //
    // The two tensors are deliberately different shapes here so the assertion
    // is about which tensor was read, not about whether a lookup succeeded.
    let num_experts = 4;
    let moe_intermediate = 16;
    let hidden = 8;
    assert_ne!(num_experts, moe_intermediate);

    for router_key in [
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlp.gate.gate_proj.weight",
    ] {
        let mut w = WeightMap::new();
        w.insert(router_key.into(), ones(&[num_experts, hidden]));
        for expert in 0..num_experts {
            w.insert(
                format!("model.layers.0.mlp.experts.{expert}.gate_proj.weight"),
                ones(&[moe_intermediate, hidden]),
            );
            w.insert(
                format!("model.layers.0.mlp.experts.{expert}.up_proj.weight"),
                ones(&[moe_intermediate, hidden]),
            );
            w.insert(
                format!("model.layers.0.mlp.experts.{expert}.down_proj.weight"),
                ones(&[hidden, moe_intermediate]),
            );
        }

        let prefix = router_prefix(&w, "model.layers.0.mlp").unwrap();
        assert_eq!(prefix, router_key.strip_suffix(".weight").unwrap());
        assert!(
            !prefix.contains(".experts."),
            "the router prefix must never land inside the expert key space: {prefix}"
        );

        // Read it the way the loader does and check the projection width: the
        // router emits one score per expert, an expert's gate emits
        // `moe_intermediate_size` channels.
        let router = UnifiedLinear::from_weights(&w, &prefix, 64, 4).unwrap();
        let out = router.forward(&ones(&[1, hidden]));
        assert_eq!(
            mlxcel_core::array_shape(&out),
            vec![1, num_experts],
            "read an expert projection instead of the router"
        );
    }
}

// Router arithmetic.

#[test]
fn selection_uses_the_biased_scores_while_the_weights_come_from_the_unbiased_ones() {
    // `group_expert_select` captures `orig_scores` BEFORE adding the correction
    // bias, selects the top-k on the biased copy, and gathers the returned
    // weights from `orig_scores`. Gathering from the biased copy instead leaves
    // the output finite and plausible while misweighting every routed
    // contribution, which no shape or NaN check can see.
    //
    // The bias below is large enough to change the SELECTED SET, so the two
    // readings differ in both the indices and the weights.
    let logits = [3.0f32, 2.0, 1.0, 0.0];
    let unbiased = softmax_ref(&logits);

    let mut gate = identity_gate(4);
    gate.top_k = 2;

    // Without the bias the top two are experts 0 and 1.
    let (indices, weights) = gate.forward(&row(&logits));
    assert_eq!(sorted_indices(&indices), vec![0, 1]);
    assert_close_slice(&gathered(&indices, &weights), &[unbiased[0], unbiased[1]]);

    // The bias promotes expert 3 from last to first, evicting expert 1.
    gate.expert_bias = Some(row(&[0.0, 0.0, 0.0, 10.0]));
    let (indices, weights) = gate.forward(&row(&logits));
    assert_eq!(
        sorted_indices(&indices),
        vec![0, 3],
        "the bias must move the selected set"
    );

    let picked = gathered(&indices, &weights);
    assert_close_slice(&picked, &[unbiased[0], unbiased[3]]);

    // And the value that would have come out of the wrong gather. Expert 3's
    // biased score is ~10.03; its unbiased score is ~0.032. Both are finite,
    // both look like a routing weight, and only this assertion separates them.
    assert!(
        (picked[1] - (unbiased[3] + 10.0)).abs() > 1.0,
        "the weights must not come from the biased copy (got {})",
        picked[1]
    );
    assert!(picked[1] < 0.1, "expert 3's unbiased score is ~0.032");
}

#[test]
fn norm_topk_prob_is_skipped_when_top_k_is_one() {
    // Upstream normalizes only when `top_k > 1`. With a single expert the
    // normalization would divide the one weight by itself and hand back exactly
    // 1.0, discarding the router's confidence entirely.
    let logits = [3.0f32, 2.0, 1.0, 0.0];
    let expected = softmax_ref(&logits)[0];

    let mut gate = identity_gate(4);
    gate.top_k = 1;
    gate.norm_topk_prob = true;

    let (indices, weights) = gate.forward(&row(&logits));
    assert_eq!(sorted_indices(&indices), vec![0]);
    let picked = gathered(&indices, &weights);
    assert_close(picked[0], expected);
    assert!(
        (picked[0] - 1.0).abs() > 0.1,
        "a top_k == 1 weight must not be renormalized to 1.0 (got {})",
        picked[0]
    );

    // At top_k == 2 the same flag does normalize, so the guard is conditional
    // rather than dead.
    gate.top_k = 2;
    let (_, weights) = gate.forward(&row(&logits));
    let sum: f32 = row_f32(&weights).iter().sum();
    assert_close(sum, 1.0);
}

#[test]
fn norm_topk_prob_divides_by_the_sum_plus_a_1e_20_epsilon() {
    // The denominator is `scores.sum(-1, keepdims=True) + 1e-20`, not a bare
    // sum. The epsilon is only observable on an all-zero score row, which the
    // softmax path cannot produce (it always sums to 1) but the sigmoid path
    // can: `sigmoid(-200)` underflows to exactly zero in float32.
    let logits = [-200.0f32; 4];
    let raw = row_f32(&mlxcel_core::sigmoid(&row(&logits)));
    assert_eq!(
        raw,
        vec![0.0; 4],
        "precondition: sigmoid must underflow to exactly zero here"
    );

    let mut gate = identity_gate(4);
    gate.top_k = 2;
    gate.norm_topk_prob = true;
    gate.score_function = ScoreFunction::Sigmoid;

    let (_, weights) = gate.forward(&row(&logits));
    let picked = row_f32(&weights);
    assert!(
        picked.iter().all(|v| v.is_finite()),
        "a bare `sum` denominator gives 0/0 = NaN here, and that NaN reaches the \
         logits without anything throwing: {picked:?}"
    );
    assert_eq!(picked, vec![0.0; 2]);
}

#[test]
fn routed_scaling_is_applied_even_when_the_flag_is_false() {
    // `BailingMoeGate` upstream stores `moe_router_enable_routed_scaling` and
    // never reads it again: `group_expert_select` ends with an unconditional
    // `scores * routed_scaling_factor`. This port mirrors that, so a checkpoint
    // decoded here matches the reference token for token. The behavior is
    // pinned rather than left to a reader's guess.
    let logits = [3.0f32, 2.0, 1.0, 0.0];

    let mut gate = identity_gate(4);
    gate.top_k = 2;
    gate.norm_topk_prob = true;
    let (_, unscaled) = gate.forward(&row(&logits));
    let unscaled = row_f32(&unscaled);

    gate.routed_scaling_factor = 2.5;
    let (_, scaled) = gate.forward(&row(&logits));
    let scaled = row_f32(&scaled);

    for (a, b) in unscaled.iter().zip(scaled.iter()) {
        assert_close(*b, a * 2.5);
    }

    // The config flag never reaches the gate at all, so the loader says out loud
    // when the two readings would diverge instead of leaving the choice
    // invisible. They coincide at the default factor of 1.0, which is the only
    // value any published checkpoint uses.
    let mut args = ling_lite();
    assert!(!args.routed_scaling_flag_is_ignored_observably());
    args.moe_router_enable_routed_scaling = false;
    assert!(
        !args.routed_scaling_flag_is_ignored_observably(),
        "the factor is still 1.0, so honoring the flag would change nothing"
    );
    args.routed_scaling_factor = 2.5;
    assert!(args.routed_scaling_flag_is_ignored_observably());
    args.moe_router_enable_routed_scaling = true;
    assert!(!args.routed_scaling_flag_is_ignored_observably());
}

#[test]
fn the_sigmoid_score_function_is_not_the_softmax_one() {
    // Reusing DeepSeek-V3's gate as-is would land here: the scores stop being a
    // distribution over the experts and every routed weight changes, with
    // nothing about the shapes or the finiteness to show for it.
    let logits = [3.0f32, 2.0, 1.0, 0.0];

    let mut gate = identity_gate(4);
    gate.top_k = 2;

    let (_, softmax_weights) = gate.forward(&row(&logits));
    gate.score_function = ScoreFunction::Sigmoid;
    let (indices, sigmoid_weights) = gate.forward(&row(&logits));

    // Both rank the experts the same way, so the selected set is identical and
    // only the weights separate the two.
    assert_eq!(sorted_indices(&indices), vec![0, 1]);
    let sigmoid_picked = gathered(&indices, &sigmoid_weights);
    assert_close(sigmoid_picked[0], 1.0 / (1.0 + (-3.0f32).exp()));
    assert_close(sigmoid_picked[1], 1.0 / (1.0 + (-2.0f32).exp()));

    let softmax_picked = row_f32(&softmax_weights);
    assert!(
        softmax_picked
            .iter()
            .zip(sigmoid_picked.iter())
            .any(|(a, b)| (a - b).abs() > 0.1),
        "softmax and sigmoid routing weights must differ"
    );
}

#[test]
fn the_gate_casts_its_weights_back_to_the_router_logit_dtype() {
    // The score function runs in float32 and the result is cast back to the
    // router logits' own dtype only at the very end, after normalization and
    // scaling. Returning float32 would silently promote the whole MoE mixture.
    let logits = [3.0f32, 2.0, 1.0, 0.0];
    let x = mlxcel_core::astype(&row(&logits), mlxcel_core::dtype::FLOAT16);

    let mut w = WeightMap::new();
    w.insert(
        "gate.weight".into(),
        mlxcel_core::astype(
            &mlxcel_core::eye(4, 4, 0, mlxcel_core::dtype::FLOAT32),
            mlxcel_core::dtype::FLOAT16,
        ),
    );
    let mut gate = identity_gate(4);
    gate.gate_proj = UnifiedLinear::from_weights(&w, "gate", 64, 4).unwrap();
    gate.top_k = 2;

    let logit_dtype = mlxcel_core::array_dtype(&gate.gate_proj.forward(&x));
    let (_, weights) = gate.forward(&x);
    assert_eq!(mlxcel_core::array_dtype(&weights), logit_dtype);
    assert_eq!(
        mlxcel_core::array_dtype(&weights),
        mlxcel_core::dtype::FLOAT16
    );
}

// Grouped routing. No published Bailing checkpoint reaches this branch:
// `Ling-lite-1.5` does not declare `n_group`, and the default of 1 gates the
// whole thing off. It is therefore pinned against hand-computed values.

#[test]
fn group_mask_scores_keeps_the_groups_with_the_largest_top_two_sum() {
    // 16 experts in 4 groups of 4, keeping the best 2 groups. The values are
    // chosen so that the three plausible group scores disagree about which two
    // groups survive:
    //
    //   group 0 = [5, 5, 0, 0]   top-2 = 10   total = 10   max = 5
    //   group 1 = [9, 0, 0, 0]   top-2 =  9   total =  9   max = 9
    //   group 2 = [20, 0, 0, 0]  top-2 = 20   total = 20   max = 20
    //   group 3 = [3, 3, 3, 3]   top-2 =  6   total = 12   max = 3
    //
    // Sum of the top two keeps {2, 0}; the group total would keep {2, 3}; the
    // group max would keep {2, 1}. Only one of the three is upstream's.
    let scores = mlxcel_core::from_slice_f32(
        &[
            5.0, 5.0, 0.0, 0.0, //
            9.0, 0.0, 0.0, 0.0, //
            20.0, 0.0, 0.0, 0.0, //
            3.0, 3.0, 3.0, 3.0,
        ],
        &[1, 16],
    );

    let masked = row_f32(&group_mask_scores(&scores, 4, 2));
    #[rustfmt::skip]
    let expected = vec![
        5.0, 5.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        20.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ];
    assert_eq!(masked, expected);

    // What the two wrong group scores would have produced, so neither can be
    // introduced silently.
    let by_total = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 3.0,
    ];
    let by_max = [
        0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    assert_ne!(masked.as_slice(), by_total.as_slice());
    assert_ne!(masked.as_slice(), by_max.as_slice());

    // Keeping three of the four groups zeroes only the weakest, which is the
    // boundary the guard `topk_group < n_group` stops one short of.
    #[rustfmt::skip]
    let keep_three = vec![
        5.0, 5.0, 0.0, 0.0,
        9.0, 0.0, 0.0, 0.0,
        20.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ];
    assert_eq!(row_f32(&group_mask_scores(&scores, 4, 3)), keep_three);
}

#[test]
fn grouped_routing_restricts_the_selection_to_the_surviving_groups() {
    // 8 experts in 2 groups of 4, keeping 1 group. The single best expert is 0,
    // and the ungrouped top two are experts 0 and 4, which live in different
    // groups. Grouping scores each group by the sum of its top two softmax
    // probabilities: group 0 gets `p(1.0) + p(0.0)` and group 1 gets
    // `p(0.9) + p(0.85)`, and the second is larger, so group 1 wins outright and
    // the selection must come entirely from it.
    //
    // The gaps are small on purpose. Softmax is steep enough that a wider spread
    // would let the single largest logit carry its group no matter what the
    // second member contributes, and the test would then pass under a
    // group-max rule too.
    let logits = [1.0f32, 0.0, 0.0, 0.0, 0.9, 0.85, 0.0, 0.0];
    let unbiased = softmax_ref(&logits);
    assert!(
        unbiased[4] + unbiased[5] > unbiased[0] + unbiased[1],
        "precondition: group 1 must win on the sum of its top two"
    );
    assert!(
        unbiased[0] > unbiased[4],
        "precondition: the single best expert must be in the losing group"
    );

    let mut ungrouped = identity_gate(8);
    ungrouped.top_k = 2;
    let (indices, _) = ungrouped.forward(&row(&logits));
    assert_eq!(sorted_indices(&indices), vec![0, 4]);

    let mut grouped = identity_gate(8);
    grouped.top_k = 2;
    grouped.n_group = 2;
    grouped.topk_group = 1;
    let (indices, weights) = grouped.forward(&row(&logits));
    assert_eq!(
        sorted_indices(&indices),
        vec![4, 5],
        "the highest-scoring expert overall is in the zeroed group"
    );

    // The mask acts on the selection copy only: the returned weights are still
    // the unmasked softmax values at the selected indices.
    assert_close_slice(&gathered(&indices, &weights), &[unbiased[4], unbiased[5]]);
}

#[test]
fn config_validation_rejects_grouped_routing_that_would_index_out_of_range() {
    // `group_expert_select` computes `k = n_group - topk_group` and calls
    // `argpartition(kth = k - 1)`, which is out of range the moment
    // `topk_group >= n_group`, and it reshapes the score row into `n_group`
    // groups without checking that `num_experts` divides evenly. Upstream checks
    // neither. MLX signals an out-of-range `argpartition` by throwing, and an
    // MLX C++ exception crossing the cxx bridge is an uncatchable
    // `std::terminate` at the first forward pass rather than a load error.
    let base = ling_lite();

    let cases: [(fn(&mut ModelArgs), &str); 7] = [
        (|a: &mut ModelArgs| a.n_group = 0, "n_group"),
        (
            |a: &mut ModelArgs| {
                a.n_group = 4;
                a.topk_group = 4;
            },
            "topk_group",
        ),
        (
            |a: &mut ModelArgs| {
                a.n_group = 4;
                a.topk_group = 5;
            },
            "topk_group",
        ),
        (
            |a: &mut ModelArgs| {
                a.n_group = 4;
                a.topk_group = 0;
            },
            "topk_group",
        ),
        // 64 experts do not divide into 5 equal groups.
        (
            |a: &mut ModelArgs| {
                a.n_group = 5;
                a.topk_group = 2;
            },
            "divisible",
        ),
        // One expert per group cannot be scored by its top two.
        (
            |a: &mut ModelArgs| {
                a.n_group = 64;
                a.topk_group = 2;
            },
            "at least 2",
        ),
        // The router selects more indices than the row has.
        (
            |a: &mut ModelArgs| a.num_experts_per_tok = 65,
            "num_experts_per_tok",
        ),
    ];
    for (mutate, expected) in cases {
        let mut args = base.clone();
        mutate(&mut args);
        let err = args
            .validate()
            .unwrap_err_or_panic("a hostile routing parameter must be rejected");
        assert!(err.contains(expected), "unhelpful error: {err}");
    }

    // Zero experts per token is rejected for its own reason: the slice that
    // takes the top-k would be empty.
    let mut zero_k = base.clone();
    zero_k.num_experts_per_tok = 0;
    assert!(
        zero_k
            .validate()
            .unwrap_err_or_panic("zero experts per token")
            .contains("num_experts_per_tok")
    );

    // A non-finite scaling factor multiplies every routed weight into NaN
    // without anything throwing.
    for factor in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut args = base.clone();
        args.routed_scaling_factor = factor;
        assert!(
            args.validate()
                .unwrap_err_or_panic("a non-finite routed_scaling_factor")
                .contains("routed_scaling_factor")
        );
    }

    // The positive controls: real grouped configurations stay accepted.
    for (n_group, topk_group) in [(1usize, 4usize), (8, 4), (4, 1), (2, 1), (16, 15)] {
        let mut args = base.clone();
        args.n_group = n_group;
        args.topk_group = topk_group;
        args.validate().unwrap_or_else(|e| {
            panic!("n_group {n_group} / topk_group {topk_group} must be accepted: {e}")
        });
    }
}

// Router expert bias. Also unreachable from any published checkpoint:
// `Ling-lite-1.5` does not declare `moe_router_enable_expert_bias`, and there is
// no `expert_bias` tensor anywhere in its 5603 keys.

#[test]
fn an_absurd_topk_group_saturates_rather_than_truncating_when_grouping_is_off() {
    // `validate_routing` bounds `topk_group` to `1..=n_group - 1` only on the
    // grouped branch. `n_group` defaults to 1 (and `Ling-lite-1.5` declares
    // neither field), so on every real config `topk_group` is carried into the
    // gate unbounded, and a plain `as i32` would truncate a value above
    // `i32::MAX` to a negative number. The field is unreachable at `n_group == 1`
    // (`forward` reads it only under `n_group > 1`), so this is defense in depth
    // rather than a live bug, but a stored negative would become
    // `slice_axis(..., 0, k)` with a negative `k` the moment a later change did
    // reach it, and `slice_axis` reads `-1` as "to the end" rather than as an
    // error.
    let mut args = router_args(4, 2);
    args.n_group = 1;
    args.topk_group = usize::MAX;
    args.validate()
        .expect("topk_group is inert while n_group is 1");

    let weights = router_weights(4, None);
    let gate = BailingMoeGate::from_weights(&weights, &args, "mlp").unwrap();
    assert_eq!(gate.n_group, 1);
    assert_eq!(gate.topk_group, i32::MAX);

    // The grouped branch stays unreached, so routing is the plain top-k.
    let (indices, _) = gate.forward(&row(&[3.0, 2.0, 1.0, 0.0]));
    assert_eq!(sorted_indices(&indices), vec![0, 1]);
}

#[test]
fn the_expert_bias_is_zero_filled_when_the_flag_is_set_but_no_tensor_ships() {
    // Upstream initializes `expert_bias` to zeros when the flag is set, so a
    // checkpoint that carries no tensor still routes with a zero bias rather
    // than failing to load.
    let mut args = router_args(4, 2);
    args.moe_router_enable_expert_bias = true;

    let weights = router_weights(4, None);
    let gate = BailingMoeGate::from_weights(&weights, &args, "mlp").unwrap();
    let bias = gate
        .expert_bias
        .as_ref()
        .expect("a zero bias is still a bias");
    assert_eq!(mlxcel_core::array_shape(bias), vec![4]);
    assert_eq!(row_f32(&mlxcel_core::reshape(bias, &[1, 4])), vec![0.0; 4]);

    // A zero bias must route exactly as no bias at all.
    let logits = [3.0f32, 2.0, 1.0, 0.0];
    let (biased_idx, biased_w) = gate.forward(&row(&logits));

    args.moe_router_enable_expert_bias = false;
    let plain = BailingMoeGate::from_weights(&weights, &args, "mlp").unwrap();
    assert!(plain.expert_bias.is_none());
    let (plain_idx, plain_w) = plain.forward(&row(&logits));

    assert_eq!(sorted_indices(&biased_idx), sorted_indices(&plain_idx));
    assert_close_slice(&row_f32(&biased_w), &row_f32(&plain_w));
}

#[test]
fn the_expert_bias_is_loaded_when_the_checkpoint_ships_one_and_ignored_when_the_flag_is_clear() {
    let logits = [3.0f32, 2.0, 1.0, 0.0];
    let bias = [0.0f32, 0.0, 0.0, 10.0];
    let weights = router_weights(4, Some(&bias));

    let mut args = router_args(4, 2);
    args.moe_router_enable_expert_bias = true;
    let gate = BailingMoeGate::from_weights(&weights, &args, "mlp").unwrap();
    assert!(gate.expert_bias.is_some());
    let (indices, _) = gate.forward(&row(&logits));
    assert_eq!(sorted_indices(&indices), vec![0, 3]);

    // The same tensor with the flag clear must not be read. The `expert_bias`
    // key survives upstream's `sanitize` untouched (the rename only moves
    // `.weight` and `.bias`), so its mere presence cannot be the trigger.
    args.moe_router_enable_expert_bias = false;
    let ignored = BailingMoeGate::from_weights(&weights, &args, "mlp").unwrap();
    assert!(ignored.expert_bias.is_none());
    let (indices, _) = ignored.forward(&row(&logits));
    assert_eq!(sorted_indices(&indices), vec![0, 1]);
}

#[test]
fn validate_weights_rejects_an_expert_bias_of_the_wrong_width() {
    let mut args = tiny_args();
    args.moe_router_enable_expert_bias = true;
    let mut weights = tiny_weights(&args);
    weights.insert(
        "model.layers.0.mlp.gate.expert_bias".into(),
        ones(&[args.num_experts() as i32 + 1]),
    );
    let err = validate_weights(&weights, &args)
        .unwrap_err_or_panic("a bias that is not one value per expert");
    assert!(err.contains("expert_bias"), "{err}");

    // The right width is accepted, so the guard is not rejecting the field
    // outright.
    weights.insert(
        "model.layers.0.mlp.gate.expert_bias".into(),
        ones(&[args.num_experts() as i32]),
    );
    validate_weights(&weights, &args).unwrap();
}

// `norm_head` is live upstream; `norm_softmax` is not.

#[test]
fn normalize_lm_head_weight_divides_each_column_by_its_l2_norm_plus_1e_7() {
    // Upstream's `sanitize` computes
    // `w / (linalg.norm(w.astype(float32), axis=0, keepdims=True) + 1e-7)` and
    // casts back. Axis 0 of an `[vocab_size, hidden_size]` head is the
    // vocabulary axis, so each hidden channel is divided by its norm across the
    // whole vocabulary.
    //
    // Three columns, each pinning a different part of the formula:
    //
    //   column 0 = [3, 4]      norm 5      -> [0.6, 0.8]   (the normalization)
    //   column 1 = [0, 0]      norm 0      -> [0, 0]       (no 0/0 NaN)
    //   column 2 = [1e-7, 0]   norm 1e-7   -> [0.5, 0]     (the epsilon's value)
    //
    // The last one is the sharp one: with a `1e-6` epsilon it would be 0.0909,
    // with no epsilon it would be 1.0.
    let w = mlxcel_core::from_slice_f32(&[3.0, 0.0, 1e-7, 4.0, 0.0, 0.0], &[2, 3]);
    let normalized = normalize_lm_head_weight(&w);

    assert_eq!(mlxcel_core::array_shape(&normalized), vec![2, 3]);
    let got = rows_f32(&normalized);
    assert_close_slice(&got, &[0.6, 0.0, 0.5, 0.8, 0.0, 0.0]);

    // The dtype comes back as it went in, not as the float32 the arithmetic ran
    // in, so a bf16 head stays bf16.
    let bf16 = mlxcel_core::astype(&w, mlxcel_core::dtype::BFLOAT16);
    let normalized = normalize_lm_head_weight(&bf16);
    assert_eq!(
        mlxcel_core::array_dtype(&normalized),
        mlxcel_core::dtype::BFLOAT16
    );
}

#[test]
fn norm_head_is_applied_only_when_the_flag_is_set() {
    let mut args = tiny_args();
    let weights = tiny_weights(&args);
    let x = filled(&[1, args.hidden_size as i32]);

    assert!(!args.norm_head, "the real checkpoint clears it");
    let plain = load_lm_head(&weights, &args).unwrap();
    let reference = UnifiedLinear::from_weights(&weights, "lm_head", 64, 4).unwrap();
    assert_close(
        max_abs_diff(&plain.forward(&x), &reference.forward(&x)),
        0.0,
    );

    args.norm_head = true;
    let normed = load_lm_head(&weights, &args).unwrap();
    let mut expected_weights = WeightMap::new();
    expected_weights.insert(
        "lm_head.weight".into(),
        normalize_lm_head_weight(weights.get("lm_head.weight").unwrap()),
    );
    let expected = UnifiedLinear::from_weights(&expected_weights, "lm_head", 64, 4).unwrap();
    assert_close(
        max_abs_diff(&normed.forward(&x), &expected.forward(&x)),
        0.0,
    );

    // And the two really do differ, so the no-op comparison above is not
    // passing because the normalization is a no-op on this weight.
    assert!(
        max_abs_diff(&plain.forward(&x), &normed.forward(&x)) > 1e-4,
        "norm_head must change the logits"
    );
}

#[test]
fn norm_head_refuses_a_quantized_output_head() {
    // The stored `.weight` of a quantized head is a packed bit field, not the
    // weight, so dividing it by a column norm is not the normalization upstream
    // performs and would corrupt every logit while leaving the checkpoint
    // apparently loadable.
    let mut args = tiny_args();
    args.norm_head = true;
    args.quantization = Some(Quantization {
        group_size: 32,
        bits: 4,
    });
    let mut weights = tiny_weights(&args);
    let vocab = args.vocab_size as i32;
    let cols = args.hidden_size as i32 / args.group_size();
    let packed_in = args.hidden_size as i32 * args.bits() / 32;
    weights.insert("lm_head.weight".into(), ones(&[vocab, packed_in]));
    weights.insert("lm_head.scales".into(), filled(&[vocab, cols]));
    weights.insert("lm_head.biases".into(), filled(&[vocab, cols]));

    let err = load_lm_head(&weights, &args).unwrap_err_or_panic("norm_head on a quantized head");
    assert!(err.contains("norm_head"), "{err}");
    assert!(err.contains("quantized"), "{err}");

    // Clearing the flag loads the same quantized head fine, so the refusal is
    // about the combination and not about quantization.
    args.norm_head = false;
    load_lm_head(&weights, &args).unwrap();
}

#[test]
fn norm_softmax_true_is_rejected_rather_than_silently_ignored() {
    // mlx-lm declares the field and never reads it, the vendored
    // `modeling_bailing_moe.py` does not mention it, and
    // `configuration_bailing_moe.py` does not even name it as a parameter, so it
    // survives only through `**kwargs`. A checkpoint that sets it true is asking
    // for behavior no released implementation defines.
    let mut args = ling_lite();
    assert!(!args.norm_softmax, "the real checkpoint clears it");
    args.validate().unwrap();

    args.norm_softmax = true;
    let err = args
        .validate()
        .unwrap_err_or_panic("norm_softmax: true must be rejected");
    assert!(err.contains("norm_softmax"), "{err}");
}

// Config validation. `config.json` arrives from a third-party HuggingFace repo
// in the `mlxcel generate -m <org>/<repo>` flow, and the download layer never
// parses it.

#[test]
fn config_validation_rejects_impossible_architecture_scalars() {
    let base = ling_lite();
    let cases: [(fn(&mut ModelArgs), &str); 12] = [
        (
            |a: &mut ModelArgs| a.num_attention_heads = 0,
            "num_attention_heads",
        ),
        (
            |a: &mut ModelArgs| a.num_attention_heads = 1 << 20,
            "num_attention_heads",
        ),
        (|a: &mut ModelArgs| a.hidden_size = 0, "hidden_size"),
        (|a: &mut ModelArgs| a.hidden_size = 1 << 20, "hidden_size"),
        (|a: &mut ModelArgs| a.hidden_size = 2050, "divisible"),
        (
            |a: &mut ModelArgs| a.num_key_value_heads = Some(0),
            "num_key_value_heads",
        ),
        (
            |a: &mut ModelArgs| a.num_key_value_heads = Some(17),
            "num_key_value_heads",
        ),
        (
            |a: &mut ModelArgs| a.num_key_value_heads = Some(5),
            "divisible",
        ),
        (
            |a: &mut ModelArgs| a.num_hidden_layers = 0,
            "num_hidden_layers",
        ),
        (
            |a: &mut ModelArgs| a.num_hidden_layers = usize::MAX,
            "num_hidden_layers",
        ),
        (|a: &mut ModelArgs| a.vocab_size = 0, "vocab_size"),
        (
            |a: &mut ModelArgs| a.max_position_embeddings = 1 << 30,
            "max_position_embeddings",
        ),
    ];
    for (mutate, expected) in cases {
        let mut args = base.clone();
        mutate(&mut args);
        let err = args
            .validate()
            .unwrap_err_or_panic("an impossible scalar must be rejected");
        assert!(err.contains(expected), "unhelpful error: {err}");
    }

    // The three widths that size an `as i32` cast on the way to a projection.
    for mutate in [
        (|a: &mut ModelArgs| a.intermediate_size = usize::MAX) as fn(&mut ModelArgs),
        |a: &mut ModelArgs| a.intermediate_size = 0,
        |a: &mut ModelArgs| a.moe_intermediate_size = Some(usize::MAX),
        |a: &mut ModelArgs| a.moe_intermediate_size = Some(0),
    ] {
        let mut args = base.clone();
        mutate(&mut args);
        assert!(
            args.validate()
                .unwrap_err_or_panic("an impossible FFN width")
                .contains("intermediate_size")
        );
    }

    // The shared-expert width is a product, so it has to be checked after the
    // multiply rather than on either factor alone.
    let mut overflow = base.clone();
    overflow.moe_shared_expert_intermediate_size = Some(usize::MAX);
    assert!(
        overflow
            .validate()
            .unwrap_err_or_panic("an overflowing shared width")
            .contains("shared")
    );
    let mut wide = base.clone();
    wide.moe_shared_expert_intermediate_size = Some(1 << 21);
    wide.num_shared_experts = 4;
    assert!(
        wide.validate()
            .unwrap_err_or_panic("a shared width past the ceiling")
            .contains("shared")
    );
    let mut many = base.clone();
    many.num_shared_experts = 1 << 20;
    assert!(
        many.validate()
            .unwrap_err_or_panic("too many shared experts")
            .contains("num_shared_experts")
    );

    // Too many routed experts, which bounds the per-expert weight-probe loop.
    let mut experts = base.clone();
    experts.num_experts = Some(1 << 20);
    assert!(
        experts
            .validate()
            .unwrap_err_or_panic("too many experts")
            .contains("num_experts")
    );
}

#[test]
fn the_zero_checks_run_before_the_divisibility_checks() {
    // `0.is_multiple_of(0)` is true in Rust, so a divisibility check on its own
    // lets a zero divisor through and `head_dim()` then divides by zero. Both
    // zero pairs have to be caught by the magnitude check that precedes it.
    let mut args = ling_lite();
    args.hidden_size = 0;
    args.num_attention_heads = 0;
    assert!(0usize.is_multiple_of(0), "the trap this test exists for");
    let err = args
        .validate()
        .unwrap_err_or_panic("zero heads and zero width");
    assert!(err.contains("num_attention_heads"), "{err}");

    let mut args = ling_lite();
    args.num_key_value_heads = Some(0);
    let err = args.validate().unwrap_err_or_panic("zero KV heads");
    assert!(err.contains("num_key_value_heads"), "{err}");

    // `num_experts_per_tok` is checked against a zero expert count by the
    // `has_routed_experts` gate rather than by a division, but the same shape of
    // mistake is pinned here.
    let mut args = ling_lite();
    args.num_experts = Some(0);
    assert!(!args.has_routed_experts());
    args.validate()
        .expect("no routed experts at all is a dense model, not an error");
}

#[test]
fn config_validation_rejects_rope_parameters_mlx_would_throw_on() {
    // `mlx::core::fast::rope` requires `dims` positive, even, and no larger than
    // the last axis, and enforces that by throwing. `fast_rope` crosses the cxx
    // bridge as `UniquePtr<MlxArray>` rather than a `Result`, so that throw is
    // an uncatchable `std::terminate` at the FIRST FORWARD PASS, long after the
    // checkpoint appeared to load cleanly.
    let base = ling_lite();

    // `partial_rotary_factor` is a float, which widens what a config can express
    // beyond the integer fields. The `as i32` cast saturates, so NaN becomes 0
    // and an infinity becomes `i32::MAX`; both are out of range, but the message
    // has to name the field the reader will find in their config.
    for factor in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut args = base.clone();
        args.partial_rotary_factor = factor;
        assert!(
            args.validate()
                .unwrap_err_or_panic("a non-finite partial_rotary_factor")
                .contains("partial_rotary_factor")
        );
    }
    for factor in [0.0f32, -0.5, 1.5, 0.001] {
        let mut args = base.clone();
        args.partial_rotary_factor = factor;
        args.validate()
            .unwrap_err_or_panic(&format!("partial_rotary_factor {factor} must be rejected"));
    }
    // int(128 * 0.0234375) == 3, which is odd: RoPE rotates channel pairs.
    let mut odd = base.clone();
    odd.partial_rotary_factor = 3.0 / 128.0;
    assert_eq!(odd.rope_dims(), 3);
    assert!(
        odd.validate()
            .unwrap_err_or_panic("an odd rotary width")
            .contains("even")
    );

    // An explicit `rotary_dim` bypasses the factor and needs the same contract.
    for dims in [0usize, 3, 129, usize::MAX] {
        let mut args = base.clone();
        args.rotary_dim = Some(dims);
        args.validate()
            .unwrap_err_or_panic(&format!("rotary_dim {dims} must be rejected"));
    }

    // RoPE exponentiates the base per channel, so a zero, negative or non-finite
    // one makes every rotated channel NaN with nothing throwing.
    for theta in [0.0f32, -600_000.0, f32::NAN, f32::INFINITY] {
        let mut args = base.clone();
        args.rope_theta = theta;
        assert!(
            args.validate()
                .unwrap_err_or_panic("a bad rope_theta")
                .contains("rope_theta")
        );
    }

    // The boundaries stay accepted: the whole head, and the narrowest legal
    // width of two channels.
    let mut full = base.clone();
    full.partial_rotary_factor = 1.0;
    assert_eq!(full.rope_dims(), 128);
    full.validate().unwrap();
    let mut narrow = base.clone();
    narrow.rotary_dim = Some(2);
    narrow.validate().unwrap();
}

#[test]
fn config_validation_rejects_an_rms_norm_eps_that_would_nan_every_hidden_state() {
    // `fast::rms_norm` never inspects `eps`; it computes
    // `x * weight * rsqrt(mean(x^2) + eps)` and hands back NaN, which reaches
    // the logits and then the sampler. The checkpoint loads, generation runs,
    // and the output is uniform garbage, so the rejection has to happen at load
    // or not at all.
    for eps in [0.0f32, -1e-6, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut args = ling_lite();
        args.rms_norm_eps = eps;
        assert!(
            args.validate()
                .unwrap_err_or_panic("a bad rms_norm_eps")
                .contains("rms_norm_eps")
        );
    }
    for eps in [1e-6f32, 1e-12, 1e-5, 1.0] {
        let mut args = ling_lite();
        args.rms_norm_eps = eps;
        args.validate()
            .unwrap_or_else(|e| panic!("rms_norm_eps {eps} is ordinary and must be accepted: {e}"));
    }
}

#[test]
fn config_validation_rejects_a_quantization_block_that_would_abort_an_mlx_kernel() {
    // MLX derives the unpacked width as `packed_in * 32 / bits`, which divides
    // by zero at 0 and collapses to zero above 32, and the throw that follows
    // crosses the cxx bridge as an uncatchable abort.
    for (group_size, bits, expected) in [
        (64, 0, "bits"),
        (64, -4, "bits"),
        (64, 33, "bits"),
        (0, 4, "group_size"),
        (-64, 4, "group_size"),
    ] {
        let mut args = ling_lite();
        args.quantization = Some(Quantization { group_size, bits });
        assert!(
            args.validate()
                .unwrap_err_or_panic("a hostile quantization block")
                .contains(expected)
        );
    }

    // Every pair a real export declares stays accepted. This is a range check
    // rather than an allowlist because mlxcel re-derives an effective bit width
    // from the tensor shapes, which mixed-precision exports rely on.
    for (group_size, bits) in [(32, 4), (64, 4), (128, 4), (64, 8), (64, 6), (16, 4)] {
        let mut args = ling_lite();
        args.quantization = Some(Quantization { group_size, bits });
        args.validate().unwrap();
        assert_eq!(args.group_size(), group_size);
        assert_eq!(args.bits(), bits);
    }
}

#[test]
fn config_validation_rejects_a_rope_scaling_block_this_loader_does_not_implement() {
    // Upstream threads the block into `initialize_rope`; this loader always
    // builds the plain rotation, so accepting a scaled block would place every
    // token at the wrong position while the model still loaded and still
    // generated fluent text.
    let scaled: ModelArgs = serde_json::from_str(
        r#"{"hidden_size": 2048, "num_hidden_layers": 28, "num_attention_heads": 16,
            "intermediate_size": 1408, "vocab_size": 126464,
            "rope_scaling": {"rope_type": "yarn", "factor": 4.0}}"#,
    )
    .expect("parses");
    let err = scaled
        .validate()
        .unwrap_err_or_panic("a yarn rope_scaling block must be rejected");
    assert!(err.contains("rope_scaling"), "{err}");

    // The three no-op spellings stay accepted, including the `"default"` block
    // several vendors emit instead of a null.
    for block in [
        "null",
        "{}",
        r#"{"rope_type": "default"}"#,
        r#"{"type": "default"}"#,
    ] {
        let args: ModelArgs = serde_json::from_str(&format!(
            r#"{{"hidden_size": 2048, "num_hidden_layers": 28, "num_attention_heads": 16,
                 "intermediate_size": 1408, "vocab_size": 126464, "rope_scaling": {block}}}"#
        ))
        .expect("parses");
        args.validate()
            .unwrap_or_else(|e| panic!("rope_scaling {block} must be accepted: {e}"));
    }
}

// Weight-shape validation.

#[test]
fn validate_weights_accepts_the_real_ling_lite_shape_signature() {
    // The positive control for every rejection below, and the guard against the
    // guards being tightened into refusing the genuine checkpoint. The map holds
    // the real key names at the real shapes, built from lazy MLX arrays that are
    // never evaluated, so nothing here reads or allocates weight data.
    let args = ling_lite();
    args.validate().unwrap();

    let weights = ling_lite_shape_signature(&args);
    // 28 layers * (2 attention projections + 2 norms + 1 router + 64 * 3 expert
    // projections + 3 shared projections) + word_embeddings + norm + lm_head.
    assert_eq!(weights.len(), 28 * 200 + 3, "5603 tensors, as shipped");
    validate_weights(&weights, &args)
        .expect("the real Ling-lite-1.5 shape signature must pass every guard");
}

#[test]
fn validate_weights_rejects_projections_and_norms_that_disagree_with_the_config() {
    let args = tiny_args();
    let hidden = args.hidden_size as i32;
    let qkv = args.qkv_out_features() as i32;

    let cases: [(&str, UniquePtr<MlxArray>, &str); 6] = [
        // A fused projection one KV head short still slices without throwing,
        // because MLX's `slice` clamps an out-of-range stop; the reshape that
        // follows aborts the process instead of returning an error.
        (
            "model.layers.0.attention.query_key_value.weight",
            ones(&[qkv - 8, hidden]),
            "query_key_value",
        ),
        (
            "model.layers.0.attention.query_key_value.weight",
            ones(&[hidden, qkv]),
            "query_key_value",
        ),
        (
            "model.layers.0.attention.dense.weight",
            ones(&[hidden, hidden * 2]),
            "dense",
        ),
        (
            "model.layers.0.input_layernorm.weight",
            ones(&[hidden + 1]),
            "input_layernorm",
        ),
        ("model.norm.weight", ones(&[hidden, hidden]), "model.norm"),
        (
            "model.layers.0.mlp.shared_experts.down_proj.weight",
            ones(&[hidden, hidden]),
            "shared_experts.down_proj",
        ),
    ];
    // The shared MLP is 48 wide, not `hidden`, so the replacement above really
    // is the wrong shape.
    assert_ne!(args.shared_expert_intermediate_size(), args.hidden_size);
    for (key, replacement, expected) in cases {
        let mut weights = tiny_weights(&args);
        weights.insert(key.into(), replacement);
        let err = validate_weights(&weights, &args)
            .unwrap_err_or_panic(&format!("a wrong shape at {key} must be rejected"));
        assert!(err.contains(expected), "unhelpful error for {key}: {err}");
    }

    // A missing tensor is named rather than defaulted.
    let mut weights = tiny_weights(&args);
    weights.remove("model.layers.1.attention.dense.weight");
    let err = validate_weights(&weights, &args).unwrap_err_or_panic("a missing projection");
    assert!(
        err.contains("model.layers.1.attention.dense.weight"),
        "{err}"
    );
}

#[test]
fn validate_weights_checks_every_expert_index_not_only_the_first() {
    // `stack_individual_experts` gathers contiguously from index 0 until the
    // first gap, so a checkpoint missing a late expert registers a SHORT stack
    // that the router can index past. MLX's gather adds the axis size to a
    // negative index but does not range-check a positive one, so the missing
    // planes would be read out of bounds and the result would reach the logits.
    let args = tiny_args();
    let last = args.num_experts() - 1;

    let mut weights = tiny_weights(&args);
    weights.remove(&format!("model.layers.0.mlp.experts.{last}.up_proj.weight"));
    let err = validate_weights(&weights, &args).unwrap_err_or_panic("a missing last expert");
    assert!(err.contains(&format!("experts.{last}.up_proj")), "{err}");

    // A late expert of the wrong width is caught on the same path.
    let mut weights = tiny_weights(&args);
    weights.insert(
        format!("model.layers.0.mlp.experts.{last}.down_proj.weight"),
        ones(&[
            args.hidden_size as i32,
            args.moe_intermediate_size() as i32 + 1,
        ]),
    );
    let err = validate_weights(&weights, &args).unwrap_err_or_panic("a mis-shaped last expert");
    assert!(err.contains(&format!("experts.{last}.down_proj")), "{err}");
}

#[test]
fn validate_weights_accepts_the_stacked_layout_and_rejects_a_short_one() {
    // An mlx-lm conversion has already stacked the per-expert tensors into
    // `mlp.switch_mlp.{proj}`. Both layouts have to load, and the stacked one
    // has to be checked on its gather axis.
    let args = tiny_args();
    let experts = args.num_experts() as i32;
    let hidden = args.hidden_size as i32;
    let moe = args.moe_intermediate_size() as i32;

    let stacked = |planes: i32| {
        let mut w = tiny_weights(&args);
        for layer in 0..args.num_hidden_layers {
            let mlp = format!("model.layers.{layer}.mlp");
            for expert in 0..args.num_experts() {
                for leaf in ["gate_proj", "up_proj", "down_proj"] {
                    w.remove(&format!("{mlp}.experts.{expert}.{leaf}.weight"));
                }
            }
            w.insert(
                format!("{mlp}.switch_mlp.gate_proj.weight"),
                ones(&[planes, moe, hidden]),
            );
            w.insert(
                format!("{mlp}.switch_mlp.up_proj.weight"),
                ones(&[planes, moe, hidden]),
            );
            w.insert(
                format!("{mlp}.switch_mlp.down_proj.weight"),
                ones(&[planes, hidden, moe]),
            );
        }
        w
    };

    validate_weights(&stacked(experts), &args).expect("the converted layout must load");
    // More planes than the config claims is fine: the router can never reach
    // them.
    validate_weights(&stacked(experts + 1), &args).unwrap();

    let err = validate_weights(&stacked(experts - 1), &args)
        .unwrap_err_or_panic("a stack shorter than num_experts");
    assert!(err.contains("expert planes"), "{err}");
}

#[test]
fn validate_weights_rejects_a_quantized_projection_packed_for_a_different_input_width() {
    // Quantization packs the INPUT axis only, so a projection honestly packed
    // for a different `hidden_size` keeps exactly the right row count and
    // survives every other check. MLX reconstructs the input width as
    // `scales.shape(-1) * group_size` and throws when it disagrees with the
    // activation, which crosses the cxx bridge as an uncatchable abort at the
    // first forward pass.
    let mut args = tiny_args();
    args.quantization = Some(Quantization {
        group_size: 32,
        bits: 4,
    });
    // One group of 32 describes the whole 32-wide input, so `cols == 1` is the
    // honest packing and `cols == 2` is a tensor built for a 64-wide input.
    assert_eq!(args.hidden_size as i32 / args.group_size(), 1);

    // The positive control first, so the width check cannot be passing by
    // rejecting everything quantized.
    let mut weights = tiny_weights(&args);
    quantize_attention_block(&args, &mut weights, 0, 1);
    validate_weights(&weights, &args).expect("a consistently packed block must load");

    let mut weights = tiny_weights(&args);
    quantize_attention_block(&args, &mut weights, 0, 2);
    let err = validate_weights(&weights, &args).unwrap_err_or_panic("a mis-packed input width");
    assert!(err.contains("input width"), "{err}");

    // The affine zero points must have the same shape as the scales they belong
    // to; MLX rejects a mismatch by throwing.
    let mut weights = tiny_weights(&args);
    quantize_attention_block(&args, &mut weights, 0, 1);
    weights.insert(
        "model.layers.0.attention.dense.biases".into(),
        filled(&[args.hidden_size as i32, 2]),
    );
    let err = validate_weights(&weights, &args).unwrap_err_or_panic("mis-shaped zero points");
    assert!(err.contains("same shape"), "{err}");
}

#[test]
fn validate_weights_rejects_an_output_head_that_disagrees_with_the_config() {
    // The output head is the one projection nothing else in the loader checks,
    // and the axis that aborts the process is not the row count. Rows only
    // bound an argmax over the logits. The input width is the inner dimension
    // of the matmul that produces them, and MLX throws on a mismatch rather
    // than broadcasting, which crosses the cxx bridge as an uncatchable abort
    // at the first forward pass.
    let args = tiny_args();
    let hidden = args.hidden_size as i32;
    let vocab = args.vocab_size as i32;

    // The positive control first, so the guard cannot be passing by rejecting
    // every head.
    validate_weights(&tiny_weights(&args), &args).expect("the tiny checkpoint must pass");

    // A head built for a different model width. The row count is still exactly
    // `vocab_size`, so nothing but the input axis can see this.
    let mut narrow = tiny_weights(&args);
    narrow.insert("lm_head.weight".into(), filled(&[vocab, hidden - 8]));
    let err =
        validate_weights(&narrow, &args).unwrap_err_or_panic("an output head of the wrong width");
    assert!(err.contains("lm_head.weight"), "{err}");

    // A head that does not cover the vocabulary the config declares.
    let mut short = tiny_weights(&args);
    short.insert("lm_head.weight".into(), filled(&[vocab - 4, hidden]));
    assert!(
        validate_weights(&short, &args)
            .unwrap_err_or_panic("an output head short of vocab_size")
            .contains("lm_head.weight")
    );

    // A missing head is named rather than surfacing later.
    let mut absent = tiny_weights(&args);
    absent.remove("lm_head.weight");
    assert!(
        validate_weights(&absent, &args)
            .unwrap_err_or_panic("a missing output head")
            .contains("lm_head.weight")
    );

    // Tied embeddings ship no head at all, so the guard must not demand one.
    let mut tied_args = args.clone();
    tied_args.tie_word_embeddings = true;
    let mut tied = tiny_weights(&tied_args);
    tied.remove("lm_head.weight");
    validate_weights(&tied, &tied_args).expect("a tied checkpoint ships no lm_head");

    // The quantized head, which is the form that actually reaches
    // `quantized_matmul`: packing compresses the input axis only, so the row
    // count is still exactly `vocab_size` and only `scales.shape(-1) *
    // group_size` says which width the tensor was built for.
    let mut quant_args = tiny_args();
    quant_args.quantization = Some(Quantization {
        group_size: 32,
        bits: 4,
    });
    let packed_in = hidden * quant_args.bits() / 32;
    assert_eq!(hidden / quant_args.group_size(), 1);

    let mut honest = tiny_weights(&quant_args);
    honest.insert("lm_head.weight".into(), ones(&[vocab, packed_in]));
    honest.insert("lm_head.scales".into(), filled(&[vocab, 1]));
    honest.insert("lm_head.biases".into(), filled(&[vocab, 1]));
    validate_weights(&honest, &quant_args).expect("a consistently packed head must load");

    let mut mispacked = tiny_weights(&quant_args);
    mispacked.insert("lm_head.weight".into(), ones(&[vocab, packed_in]));
    mispacked.insert("lm_head.scales".into(), filled(&[vocab, 2]));
    mispacked.insert("lm_head.biases".into(), filled(&[vocab, 2]));
    let err = validate_weights(&mispacked, &quant_args)
        .unwrap_err_or_panic("a head packed for a different input width");
    assert!(err.contains("input width"), "{err}");

    // The same reconstruction applied to the token table, whose width the
    // shared `validate_embedding_table` deliberately leaves to the scales.
    let mut table = tiny_weights(&quant_args);
    table.insert(
        "model.word_embeddings.weight".into(),
        ones(&[vocab, packed_in]),
    );
    table.insert("model.word_embeddings.scales".into(), filled(&[vocab, 2]));
    table.insert("model.word_embeddings.biases".into(), filled(&[vocab, 2]));
    let err = validate_weights(&table, &quant_args)
        .unwrap_err_or_panic("a token table packed for a different input width");
    assert!(err.contains("input width"), "{err}");
}

// Construction and forward.

#[test]
fn a_tiny_model_prefills_then_decodes() {
    let args = tiny_args();
    let weights = tiny_weights(&args);
    let model = BailingMoeModel::from_weights(&weights, &args).unwrap();

    assert_eq!(LanguageModel::num_layers(&model), args.num_hidden_layers);
    assert_eq!(LanguageModel::eos_token_ids(&model), vec![19]);

    let mut caches = model.make_caches();
    let prompt = mlxcel_core::from_slice_i32(&[1, 2, 3, 4, 5], &[1, 5]);
    let logits = model.forward(&prompt, &mut caches, None);
    assert_eq!(
        mlxcel_core::array_shape(&logits),
        vec![1, 5, args.vocab_size as i32]
    );
    assert!(max_abs(&logits).is_finite());
    assert_eq!(caches[0].offset, 5);

    let next = mlxcel_core::from_slice_i32(&[6], &[1, 1]);
    let step = model.forward(&next, &mut caches, None);
    assert_eq!(
        mlxcel_core::array_shape(&step),
        vec![1, 1, args.vocab_size as i32]
    );
    assert!(max_abs(&step).is_finite());
    assert_eq!(caches[0].offset, 6);
}

#[test]
fn a_dense_prefix_layer_is_a_plain_mlp_and_the_rest_are_sparse() {
    // `first_k_dense_replace` is 0 on `Ling-lite-1.5`, so this branch is
    // unreachable from the real checkpoint and needs a synthetic config.
    let mut args = tiny_args();
    args.first_k_dense_replace = 1;
    let weights = tiny_weights(&args);
    let model = BailingMoeModel::from_weights(&weights, &args).unwrap();

    assert!(matches!(model.layers[0].mlp, FeedForward::Dense(_)));
    assert!(matches!(model.layers[1].mlp, FeedForward::Sparse(_)));

    let mut caches = model.make_caches();
    let prompt = mlxcel_core::from_slice_i32(&[1, 2, 3, 4], &[1, 4]);
    let logits = model.forward(&prompt, &mut caches, None);
    assert_eq!(
        mlxcel_core::array_shape(&logits),
        vec![1, 4, args.vocab_size as i32]
    );
    assert!(max_abs(&logits).is_finite());
}

#[test]
fn the_shared_expert_is_added_at_a_fixed_weight_of_one() {
    // Upstream ends the block with `out = out + self.shared_experts(x)`: no
    // routing weight, no averaging, and never packed into the switch tensors.
    // Zeroing every routed expert leaves the block output equal to the shared
    // MLP's own output, which any other combining rule would scale away from.
    let args = tiny_args();
    let mut weights = tiny_weights(&args);
    let hidden = args.hidden_size as i32;
    let moe = args.moe_intermediate_size() as i32;
    for expert in 0..args.num_experts() {
        let p = format!("model.layers.0.mlp.experts.{expert}");
        weights.insert(format!("{p}.gate_proj.weight"), zeros(&[moe, hidden]));
        weights.insert(format!("{p}.up_proj.weight"), zeros(&[moe, hidden]));
        weights.insert(format!("{p}.down_proj.weight"), zeros(&[hidden, moe]));
    }

    let block = BailingMoeSparseBlock::from_weights(&weights, &args, "model.layers.0.mlp").unwrap();
    assert!(block.shared_experts.is_some());
    let shared =
        BailingMoeMLP::from_weights(&weights, &args, "model.layers.0.mlp.shared_experts").unwrap();

    let x = filled(&[1, 3, hidden]);
    let combined = block.forward(&x);
    let shared_only = shared.forward(&x);
    assert_eq!(
        mlxcel_core::array_shape(&combined),
        mlxcel_core::array_shape(&shared_only)
    );
    assert!(
        max_abs_diff(&combined, &shared_only) < 1e-5,
        "the shared expert is added at exactly 1.0"
    );
    assert!(
        max_abs(&shared_only) > 1e-3,
        "the comparison above must be against a non-trivial output"
    );

    // Turning the flag off drops the shared MLP entirely, so a checkpoint that
    // ships the tensors but clears the flag routes without them.
    let mut disabled = args.clone();
    disabled.moe_router_enable_shared_expert = false;
    let block =
        BailingMoeSparseBlock::from_weights(&weights, &disabled, "model.layers.0.mlp").unwrap();
    assert!(block.shared_experts.is_none());
    assert!(
        max_abs(&block.forward(&x)) < 1e-6,
        "every routed expert is zero"
    );
}

// Synthetic checkpoints and helpers.

/// A Bailing MoE small enough to run in a unit test, keeping the real
/// checkpoint's shape: grouped-query attention (4 query heads over 2 KV heads),
/// four routed experts with two active per token, and two shared experts.
///
/// `hidden_size` 32 over 4 heads gives an 8-wide head, so the fused projection
/// is `(4 + 2 * 2) * 8 = 64` and splits at `(32, 48)`. The four FFN widths are
/// deliberately all different (`hidden` 32, `moe_intermediate_size` 24, the
/// shared MLP 24 * 2 = 48, the dense-prefix MLP 40) so a shape assertion cannot
/// pass by comparing against the wrong one.
fn tiny_args() -> ModelArgs {
    serde_json::from_str(
        r#"{
            "model_type": "bailing_moe",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 40,
            "moe_intermediate_size": 24,
            "num_experts": 4,
            "num_shared_experts": 2,
            "num_experts_per_tok": 2,
            "norm_topk_prob": true,
            "first_k_dense_replace": 0,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-06,
            "rope_theta": 600000,
            "tie_word_embeddings": false,
            "vocab_size": 20,
            "eos_token_id": 19
        }"#,
    )
    .expect("the tiny config parses")
}

/// A raw `inclusionAI`-layout checkpoint for [`tiny_args`]: `model.word_embeddings`,
/// `model.layers.N.attention.{query_key_value,dense}`, per-expert
/// `mlp.experts.{e}.{gate,up,down}_proj`, the router at `mlp.gate`, a single wide
/// `mlp.shared_experts`, the two norms and an untied `lm_head`.
///
/// Layers below `first_k_dense_replace` get a plain `mlp.{gate,up,down}_proj` at
/// `intermediate_size` instead.
fn tiny_weights(args: &ModelArgs) -> WeightMap {
    let hidden = args.hidden_size as i32;
    let vocab = args.vocab_size as i32;
    let qkv = args.qkv_out_features() as i32;
    let q_size = (args.num_attention_heads * args.head_dim()) as i32;
    let dense_ff = args.intermediate_size as i32;
    let moe_ff = args.moe_intermediate_size() as i32;
    let shared_ff = args.shared_expert_intermediate_size() as i32;

    let mut w = WeightMap::new();
    w.insert(
        "model.word_embeddings.weight".into(),
        filled(&[vocab, hidden]),
    );
    w.insert("model.norm.weight".into(), ones(&[hidden]));
    w.insert("lm_head.weight".into(), filled(&[vocab, hidden]));

    for layer in 0..args.num_hidden_layers {
        let p = format!("model.layers.{layer}");
        w.insert(format!("{p}.input_layernorm.weight"), ones(&[hidden]));
        w.insert(
            format!("{p}.post_attention_layernorm.weight"),
            ones(&[hidden]),
        );
        w.insert(
            format!("{p}.attention.query_key_value.weight"),
            filled(&[qkv, hidden]),
        );
        w.insert(
            format!("{p}.attention.dense.weight"),
            filled(&[hidden, q_size]),
        );

        let mlp = format!("{p}.mlp");
        if args.is_moe_layer(layer) {
            w.insert(
                format!("{mlp}.gate.weight"),
                filled(&[args.num_experts() as i32, hidden]),
            );
            for expert in 0..args.num_experts() {
                let e = format!("{mlp}.experts.{expert}");
                w.insert(format!("{e}.gate_proj.weight"), filled(&[moe_ff, hidden]));
                w.insert(format!("{e}.up_proj.weight"), filled(&[moe_ff, hidden]));
                w.insert(format!("{e}.down_proj.weight"), filled(&[hidden, moe_ff]));
            }
            if args.has_shared_expert() {
                let s = format!("{mlp}.shared_experts");
                w.insert(
                    format!("{s}.gate_proj.weight"),
                    filled(&[shared_ff, hidden]),
                );
                w.insert(format!("{s}.up_proj.weight"), filled(&[shared_ff, hidden]));
                w.insert(
                    format!("{s}.down_proj.weight"),
                    filled(&[hidden, shared_ff]),
                );
            }
        } else {
            w.insert(
                format!("{mlp}.gate_proj.weight"),
                filled(&[dense_ff, hidden]),
            );
            w.insert(format!("{mlp}.up_proj.weight"), filled(&[dense_ff, hidden]));
            w.insert(
                format!("{mlp}.down_proj.weight"),
                filled(&[hidden, dense_ff]),
            );
        }
    }
    w
}

/// Turn layer `layer`'s two attention projections into a quantized block:
/// a `.weight` packed along the input axis, plus `.scales` and `.biases` with
/// `cols` groups each. Only the shapes are ever read.
///
/// The packed input axis is `in_features * bits / 32`, which is what makes the
/// mis-packing this exercises invisible to a row check: the row count is
/// untouched by quantization, and only `scales.shape(-1) * group_size` says
/// which input width the tensor was built for.
fn quantize_attention_block(args: &ModelArgs, weights: &mut WeightMap, layer: usize, cols: i32) {
    let hidden = args.hidden_size as i32;
    let packed_in = hidden * args.bits() / 32;
    for (name, rows) in [
        ("query_key_value", args.qkv_out_features() as i32),
        ("dense", hidden),
    ] {
        let prefix = format!("model.layers.{layer}.attention.{name}");
        weights.insert(format!("{prefix}.weight"), ones(&[rows, packed_in]));
        weights.insert(format!("{prefix}.scales"), filled(&[rows, cols]));
        weights.insert(format!("{prefix}.biases"), filled(&[rows, cols]));
    }
}

/// The full `inclusionAI/Ling-lite-1.5` key set at its real shapes.
///
/// Every tensor is an unevaluated `mx::full`, so this costs graph nodes rather
/// than the checkpoint's 33.6 GB: `validate_weights` reads shape metadata only
/// and never forces an evaluation.
fn ling_lite_shape_signature(args: &ModelArgs) -> WeightMap {
    let hidden = args.hidden_size as i32;
    let vocab = args.vocab_size as i32;
    let qkv = args.qkv_out_features() as i32;
    let q_size = (args.num_attention_heads * args.head_dim()) as i32;
    let moe = args.moe_intermediate_size() as i32;
    let shared = args.shared_expert_intermediate_size() as i32;

    // The shapes read straight out of the shard headers.
    assert_eq!((qkv, hidden), (3072, 2048));
    assert_eq!((moe, shared), (1408, 2816));

    let mut w = WeightMap::new();
    w.insert(
        "model.word_embeddings.weight".into(),
        lazy(&[vocab, hidden]),
    );
    w.insert("model.norm.weight".into(), lazy(&[hidden]));
    w.insert("lm_head.weight".into(), lazy(&[vocab, hidden]));

    for layer in 0..args.num_hidden_layers {
        let p = format!("model.layers.{layer}");
        w.insert(format!("{p}.input_layernorm.weight"), lazy(&[hidden]));
        w.insert(
            format!("{p}.post_attention_layernorm.weight"),
            lazy(&[hidden]),
        );
        w.insert(
            format!("{p}.attention.query_key_value.weight"),
            lazy(&[qkv, hidden]),
        );
        w.insert(
            format!("{p}.attention.dense.weight"),
            lazy(&[hidden, q_size]),
        );

        let mlp = format!("{p}.mlp");
        w.insert(
            format!("{mlp}.gate.weight"),
            lazy(&[args.num_experts() as i32, hidden]),
        );
        for expert in 0..args.num_experts() {
            let e = format!("{mlp}.experts.{expert}");
            w.insert(format!("{e}.gate_proj.weight"), lazy(&[moe, hidden]));
            w.insert(format!("{e}.up_proj.weight"), lazy(&[moe, hidden]));
            w.insert(format!("{e}.down_proj.weight"), lazy(&[hidden, moe]));
        }
        let s = format!("{mlp}.shared_experts");
        w.insert(format!("{s}.gate_proj.weight"), lazy(&[shared, hidden]));
        w.insert(format!("{s}.up_proj.weight"), lazy(&[shared, hidden]));
        w.insert(format!("{s}.down_proj.weight"), lazy(&[hidden, shared]));
    }
    w
}

/// A config sized so the router weight can be the identity: `hidden_size` equals
/// `num_experts`, so the input row *is* the router logit row.
fn router_args(num_experts: usize, top_k: usize) -> ModelArgs {
    let mut args = tiny_args();
    args.hidden_size = num_experts;
    args.num_attention_heads = 1;
    args.num_key_value_heads = Some(1);
    args.num_experts = Some(num_experts);
    args.num_experts_per_tok = top_k;
    args.norm_topk_prob = false;
    args
}

/// A router-only weight map under the `mlp` prefix, in the raw spelling, with an
/// optional `expert_bias` sidecar.
fn router_weights(num_experts: i32, expert_bias: Option<&[f32]>) -> WeightMap {
    let mut w = WeightMap::new();
    w.insert(
        "mlp.gate.weight".into(),
        mlxcel_core::eye(num_experts, num_experts, 0, mlxcel_core::dtype::FLOAT32),
    );
    if let Some(bias) = expert_bias {
        w.insert(
            "mlp.gate.expert_bias".into(),
            mlxcel_core::from_slice_f32(bias, &[num_experts]),
        );
    }
    w
}

/// A [`BailingMoeGate`] whose `gate_proj` is the identity, so `forward`'s input
/// row is the router logit row and every assertion is about the routing
/// arithmetic rather than about a matmul.
fn identity_gate(num_experts: i32) -> BailingMoeGate {
    let mut w = WeightMap::new();
    w.insert(
        "gate.weight".into(),
        mlxcel_core::eye(num_experts, num_experts, 0, mlxcel_core::dtype::FLOAT32),
    );
    BailingMoeGate {
        gate_proj: UnifiedLinear::from_weights(&w, "gate", 64, 4).unwrap(),
        expert_bias: None,
        top_k: 1,
        n_group: 1,
        topk_group: 1,
        routed_scaling_factor: 1.0,
        norm_topk_prob: false,
        score_function: ScoreFunction::Softmax,
    }
}

/// The reference softmax, in Rust, so the expected routing weights are computed
/// independently of MLX.
fn softmax_ref(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|v| v / sum).collect()
}

fn row(values: &[f32]) -> UniquePtr<MlxArray> {
    mlxcel_core::from_slice_f32(values, &[1, values.len() as i32])
}

fn lazy(shape: &[i32]) -> UniquePtr<MlxArray> {
    mlxcel_core::full_f32(shape, 0.0, mlxcel_core::dtype::FLOAT32)
}

fn ones(shape: &[i32]) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    mlxcel_core::from_slice_f32(&vec![1.0; n as usize], shape)
}

fn zeros(shape: &[i32]) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    mlxcel_core::from_slice_f32(&vec![0.0; n as usize], shape)
}

/// Deterministic pseudo-random filler in `[-0.5, 0.5)`. A short repeating
/// pattern would make every head and every expert nearly collinear, flattening
/// the routing scores and the attention logits to the point where a forward test
/// is technically green and practically blind.
fn filled(shape: &[i32]) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    let mut state: u32 = 0x9E37_79B9;
    let data: Vec<f32> = (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect();
    mlxcel_core::from_slice_f32(&data, shape)
}

/// Every element of a rank-2 `[1, n]` row.
fn row_f32(x: &MlxArray) -> Vec<f32> {
    let shape = mlxcel_core::array_shape(x);
    assert_eq!(shape.len(), 2, "expected a rank-2 array, got {shape:?}");
    assert_eq!(shape[0], 1, "expected a single row, got {shape:?}");
    (0..shape[1])
        .map(|i| mlxcel_core::item_f32(&mlxcel_core::slice(x, &[0, i], &[1, i + 1])))
        .collect()
}

/// Every element of a rank-2 array in row-major order.
fn rows_f32(x: &MlxArray) -> Vec<f32> {
    let shape = mlxcel_core::array_shape(x);
    assert_eq!(shape.len(), 2, "expected a rank-2 array, got {shape:?}");
    let mut out = Vec::new();
    for r in 0..shape[0] {
        for c in 0..shape[1] {
            out.push(mlxcel_core::item_f32(&mlxcel_core::slice(
                x,
                &[r, c],
                &[r + 1, c + 1],
            )));
        }
    }
    out
}

/// The selected expert indices of a `[1, top_k]` index row, ascending.
///
/// `argpartition` fixes which indices land in the top-k slice but not their
/// order within it, so every assertion about a selected *set* sorts first.
fn sorted_indices(indices: &MlxArray) -> Vec<i32> {
    let as_i32 = mlxcel_core::astype(indices, mlxcel_core::dtype::INT32);
    let shape = mlxcel_core::array_shape(&as_i32);
    assert_eq!(shape[0], 1, "expected a single row, got {shape:?}");
    let mut out: Vec<i32> = (0..shape[1])
        .map(|i| mlxcel_core::item_i32(&mlxcel_core::slice(&as_i32, &[0, i], &[1, i + 1])))
        .collect();
    out.sort_unstable();
    out
}

/// The routing weights reordered to match [`sorted_indices`], so a weight can be
/// compared against the expert it belongs to rather than against a slot.
fn gathered(indices: &MlxArray, weights: &MlxArray) -> Vec<f32> {
    let as_i32 = mlxcel_core::astype(indices, mlxcel_core::dtype::INT32);
    let shape = mlxcel_core::array_shape(&as_i32);
    let mut pairs: Vec<(i32, f32)> = (0..shape[1])
        .map(|i| {
            let index = mlxcel_core::item_i32(&mlxcel_core::slice(&as_i32, &[0, i], &[1, i + 1]));
            let weight = mlxcel_core::item_f32(&mlxcel_core::slice(weights, &[0, i], &[1, i + 1]));
            (index, weight)
        })
        .collect();
    pairs.sort_by_key(|(index, _)| *index);
    pairs.into_iter().map(|(_, weight)| weight).collect()
}

fn max_abs(x: &MlxArray) -> f32 {
    mlxcel_core::item_f32(&mlxcel_core::max_all(&mlxcel_core::abs(x)))
}

fn max_abs_diff(a: &MlxArray, b: &MlxArray) -> f32 {
    max_abs(&mlxcel_core::subtract(a, b))
}

fn assert_close(got: f32, expected: f32) {
    assert!(
        (got - expected).abs() <= 2e-4,
        "expected {expected}, got {got}"
    );
}

fn assert_close_slice(got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len(), "{got:?} vs {expected:?}");
    for (a, b) in got.iter().zip(expected.iter()) {
        assert!((a - b).abs() <= 2e-4, "expected {expected:?}, got {got:?}");
    }
}

/// `Result::expect_err` without requiring the `Ok` type to be `Debug`.
///
/// `ModelArgs::validate` returns `Result<(), String>`, which `expect_err` can
/// print, but `load_lm_head` and `router_prefix` return values that cannot, and
/// spelling the same assertion two ways across one file invites one of them
/// drifting into `unwrap()`.
trait UnwrapErrOrPanic<E> {
    fn unwrap_err_or_panic(self, message: &str) -> E;
}

impl<T, E> UnwrapErrOrPanic<E> for Result<T, E> {
    fn unwrap_err_or_panic(self, message: &str) -> E {
        match self {
            Ok(_) => panic!("{message}"),
            Err(e) => e,
        }
    }
}
