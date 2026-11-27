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

//! DSpark drafter config tests (issue #1339).
//!
//! The published `LiquidAI/LFM2.5-2.6B-DSpark` `config.json` differs from the
//! Qwen 3.5 DFlash shape in four fields the plain DFlash loader used to drop
//! (`markov_rank`, `rope_is_neox_style`, `runtime_block_size`,
//! `enable_confidence_head`), in nesting `num_target_layers` under
//! `dflash_config`, and in what `block_size` counts (proposals, not rows).

use super::*;
use serde_json::json;

/// The published `LiquidAI/LFM2.5-2.6B-DSpark` config, verbatim except for
/// the `layer_types` list and `dtype`, which the loader does not read.
fn published_dspark_config() -> serde_json::Value {
    json!({
        "architectures": ["Lfm2DSparkDraftModel"],
        "model_type": "qwen3",
        "hidden_size": 2048,
        "num_hidden_layers": 5,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 64,
        "intermediate_size": 6144,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-05,
        "vocab_size": 128000,
        "rope_theta": 10000000.0,
        "max_position_embeddings": 128000,
        "block_size": 9,
        "dflash_config": {
            "mask_token_id": 125017,
            "target_layer_ids": [2, 9, 17, 21, 27],
            "num_target_layers": 30
        },
        "markov_rank": 256,
        "rope_is_neox_style": false,
        "enable_confidence_head": true,
        "markov_head_type": "vanilla"
    })
}

#[test]
fn dspark_config_lifts_dflash_config_and_markov_rank() {
    let cfg = DFlashConfig::from_json(&published_dspark_config()).expect("parse DSpark config");
    assert_eq!(cfg.markov_rank, 256);
    assert!(cfg.is_dspark());
    assert!(!cfg.rope_is_neox_style);
    assert!(cfg.enable_confidence_head);
    assert_eq!(cfg.runtime_block_size, None);
    // The nested block is lifted in full, `num_target_layers` included: the
    // LFM2.5-2.6B target has 30 layers, not the Qwen default of 32.
    assert_eq!(cfg.mask_token_id, 125017);
    assert_eq!(cfg.target_layer_ids, vec![2, 9, 17, 21, 27]);
    assert_eq!(cfg.num_target_layers, Some(30));
    // Backbone geometry parses as an ordinary DFlash drafter.
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.head_dim, 64);
    assert_eq!(cfg.vocab_size, 128000);
    assert!((cfg.rope_theta - 1e7).abs() < 1.0);
    assert!(cfg.tie_word_embeddings, "LFM2 ties its embeddings");
}

#[test]
fn dspark_verify_width_is_block_size_plus_one() {
    let cfg = DFlashConfig::from_json(&published_dspark_config()).expect("parse DSpark config");
    // `block_size` is gamma, the proposal count; the verify block adds the
    // anchor row.
    assert_eq!(cfg.block_size, 9);
    assert_eq!(cfg.verify_width(), 10);
    // With no `runtime_block_size` the runtime width is the eight-row default.
    assert_eq!(cfg.runtime_verify_width(), DSPARK_DEFAULT_VERIFY_WIDTH);
    assert_eq!(cfg.runtime_verify_width(), 8);

    // An explicit `runtime_block_size` below the trained width wins; one above
    // it is capped at the trained width.
    let mut explicit = published_dspark_config();
    explicit["runtime_block_size"] = json!(6);
    let cfg = DFlashConfig::from_json(&explicit).expect("parse");
    assert_eq!(cfg.runtime_verify_width(), 6);
    explicit["runtime_block_size"] = json!(12);
    let cfg = DFlashConfig::from_json(&explicit).expect("parse");
    assert_eq!(cfg.runtime_verify_width(), 10);

    // A DFlash checkpoint counts the bonus row in `block_size` already.
    let dflash = DFlashConfig::default();
    assert!(!dflash.is_dspark());
    assert_eq!(dflash.verify_width(), dflash.block_size);
    assert_eq!(dflash.runtime_verify_width(), dflash.block_size);
}

#[test]
fn dflash_defaults_keep_neox_rope_and_no_markov_head() {
    // The four new fields must default to the Qwen 3.5 DFlash behaviour so an
    // existing DFlash checkpoint loads exactly as before.
    let cfg = DFlashConfig::from_json(&json!({})).expect("parse");
    assert_eq!(cfg.markov_rank, 0);
    assert!(!cfg.is_dspark());
    assert!(cfg.rope_is_neox_style);
    assert_eq!(cfg.runtime_block_size, None);
    assert!(!cfg.enable_confidence_head);
}

#[test]
fn probe_accepts_the_dspark_architecture_marker() {
    // Both markers are present on the published checkpoint.
    assert!(is_dflash_drafter_config(&published_dspark_config()));
    // The architecture alone qualifies, as it does for `DFlashDraftModel`.
    assert!(is_dflash_drafter_config(&json!({
        "architectures": ["Lfm2DSparkDraftModel"],
        "model_type": "qwen3",
    })));
    // An ordinary LFM2 target is not a drafter.
    assert!(!is_dflash_drafter_config(&json!({
        "architectures": ["Lfm2ForCausalLM"],
        "model_type": "lfm2",
    })));
}

// =========================================================================
// Speculators and OpenBMB DSpark layouts (issue #3261)
// =========================================================================

/// `RedHatAI/Qwen3-4B-speculator.dspark` `config.json` at revision
/// `bdffaa5f`, verbatim except for the `auto_map`, `dtype`,
/// `speculators_version` and `transformers_version` keys and the dropout,
/// token-id and initializer keys of the backbone, none of which the loader
/// reads.
fn speculators_dspark_config() -> serde_json::Value {
    json!({
        "architectures": ["DSparkDraftModel"],
        "aux_hidden_state_layer_ids": [1, 9, 17, 25, 33],
        "block_size": 7,
        "confidence_head_with_markov": true,
        "draft_vocab_size": 151936,
        "enable_confidence_head": true,
        "markov_head_type": "vanilla",
        "markov_rank": 256,
        "mask_token_id": 151669,
        "sample_from_anchor": true,
        "sliding_window_non_causal": false,
        "speculators_config": {
            "algorithm": "dspark",
            "default_proposal_method": "greedy",
            "proposal_methods": [{
                "accept_tolerance": 0.0,
                "proposal_type": "greedy",
                "speculative_tokens": 7,
                "verifier_accept_k": 1
            }],
            "verifier": {
                "architectures": ["Qwen3ForCausalLM"],
                "name_or_path": "Qwen/Qwen3-4B"
            }
        },
        "speculators_model_type": "dspark",
        "target_hidden_size": null,
        "tie_word_embeddings": false,
        "transformer_layer_config": {
            "attention_bias": false,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 2560,
            "intermediate_size": 9728,
            "layer_types": [
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention"
            ],
            "max_position_embeddings": 40960,
            "max_window_layers": 28,
            "model_type": "qwen3",
            "num_attention_heads": 32,
            "num_hidden_layers": 5,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
            "sliding_window": 2048,
            "tie_word_embeddings": false,
            "use_cache": true,
            "use_sliding_window": true,
            "vocab_size": 151936
        }
    })
}

/// `openbmb/MiniCPM5-2B-DSpark` `config.json` at revision `a63eb96b`,
/// verbatim except for the `auto_map`, `dtype`, `transformers_version`,
/// dropout, token-id and initializer keys, none of which the loader reads.
fn openbmb_dspark_config() -> serde_json::Value {
    json!({
        "architectures": ["Qwen3DSparkModel"],
        "attention_bias": false,
        "block_size": 7,
        "draft_vocab_size": 130560,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "layer_types": [
            "full_attention", "full_attention", "full_attention", "full_attention",
            "full_attention"
        ],
        "max_position_embeddings": 131072,
        "max_window_layers": 5,
        "mlp_bias": false,
        "model_type": "qwen3",
        "num_attention_heads": 16,
        "num_hidden_layers": 5,
        "num_key_value_heads": 2,
        "num_target_layers": 42,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {"rope_theta": 5000000, "rope_type": "default"},
        "sliding_window": null,
        "tie_word_embeddings": false,
        "use_cache": true,
        "use_sliding_window": false,
        "vocab_size": 130560,
        "attention_mode": "gqa",
        "confidence_head_alpha": 1.0,
        "confidence_head_with_markov": true,
        "enable_confidence_head": true,
        "markov_head_type": "vanilla",
        "markov_rank": 256,
        "mask_token_id": 75982,
        "projector_type": "dspark",
        "target_layer_ids": [1, 10, 20, 30, 39]
    })
}

#[test]
fn speculators_layout_parses() {
    let cfg = DFlashConfig::from_json(&speculators_dspark_config()).expect("Speculators config");
    assert!(cfg.is_dspark());
    // The taps come from `aux_hidden_state_layer_ids`, not the Qwen 3.5
    // default of `[1, 8, 15, 22, 29]`.
    assert_eq!(cfg.target_layer_ids, vec![1, 9, 17, 25, 33]);
    // `block_size` 7 counts proposals; the anchor row makes the width 8.
    assert_eq!(cfg.block_size, 7);
    assert!(cfg.sample_from_anchor);
    assert!(!cfg.dspark_bonus_anchor);
    assert_eq!(cfg.verify_width(), 8);
    assert_eq!(cfg.runtime_verify_width(), 8);
    assert_eq!(cfg.mask_token_id, 151669);
    // The backbone is read from `transformer_layer_config`.
    assert_eq!(cfg.sliding_window, Some(2048));
    assert_eq!(cfg.layer_types, vec!["sliding_attention"; 5]);
    assert_eq!(cfg.hidden_size, 2560);
    assert_eq!(cfg.intermediate_size, 9728);
    assert_eq!(cfg.num_hidden_layers, 5);
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.num_key_value_heads, 8);
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.vocab_size, 151936);
    assert_eq!(cfg.max_position_embeddings, 40960);
    assert!((cfg.rms_norm_eps - 1e-6).abs() < 1e-12);
    // `rope_theta` is under the backbone's `rope_parameters`.
    assert!((cfg.rope_theta - 1e6).abs() < 1.0);
    assert!(
        cfg.rope_is_neox_style,
        "NeoX pairing unless the config says otherwise"
    );
    assert!(!cfg.tie_word_embeddings);
    assert_eq!(cfg.markov_rank, 256);
    assert!(cfg.enable_confidence_head);
    // Full vocabulary on the published checkpoint: no reduced head.
    assert_eq!(cfg.draft_vocab_size, Some(151936));
    assert_eq!(cfg.draft_vocab(), 151936);
    assert!(!cfg.has_reduced_draft_vocab());
    // Causality is unset, so it follows the layer type: every layer slides
    // and `sliding_window_non_causal` is false, so every layer is causal.
    assert_eq!(cfg.is_causal, None);
    for layer in 0..5 {
        assert_eq!(cfg.layer_sliding_window(layer), Some(2048));
        assert!(cfg.layer_is_causal(layer));
    }
    // The depth is not declared and is not invented.
    assert_eq!(cfg.num_target_layers, None);
    assert!(is_dflash_drafter_config(&speculators_dspark_config()));
}

#[test]
fn openbmb_layout_parses() {
    let cfg = DFlashConfig::from_json(&openbmb_dspark_config()).expect("OpenBMB config");
    assert!(cfg.is_dspark());
    assert_eq!(cfg.target_layer_ids, vec![1, 10, 20, 30, 39]);
    assert_eq!(cfg.num_target_layers, Some(42));
    assert_eq!(cfg.block_size, 7);
    assert_eq!(cfg.verify_width(), 8);
    assert_eq!(cfg.runtime_verify_width(), 8);
    assert_eq!(cfg.mask_token_id, 75982);
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.intermediate_size, 6144);
    assert_eq!(cfg.num_attention_heads, 16);
    assert_eq!(cfg.num_key_value_heads, 2);
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.vocab_size, 130560);
    assert_eq!(cfg.draft_vocab(), 130560);
    assert!(!cfg.has_reduced_draft_vocab());
    assert!((cfg.rope_theta - 5e6).abs() < 1.0);
    assert!(!cfg.tie_word_embeddings);
    assert_eq!(cfg.markov_rank, 256);
    // Full, non-causal attention: the flat layout keeps the DFlash default.
    assert_eq!(cfg.sliding_window, None);
    assert_eq!(cfg.is_causal, Some(false));
    for layer in 0..5 {
        assert_eq!(cfg.layer_sliding_window(layer), None);
        assert!(!cfg.layer_is_causal(layer));
    }
    assert!(is_dflash_drafter_config(&openbmb_dspark_config()));
}

#[test]
fn draft_vocab_out_of_range_is_rejected() {
    for layout in [speculators_dspark_config(), openbmb_dspark_config()] {
        let vocab = DFlashConfig::from_json(&layout)
            .expect("published")
            .vocab_size;
        // Zero, and one past the target vocabulary, on both sides of the
        // valid range `1..=vocab_size`.
        for bad in [0, vocab + 1, usize::MAX >> 12] {
            let mut config = layout.clone();
            config["draft_vocab_size"] = json!(bad);
            let err = DFlashConfig::from_json(&config).expect_err("out-of-range draft vocab");
            assert!(err.contains(&format!("draft_vocab_size {bad}")), "{err}");
            assert!(err.contains(&format!("1..={vocab}")), "{err}");
        }
        // The two ends of the range load.
        for good in [1, vocab] {
            let mut config = layout.clone();
            config["draft_vocab_size"] = json!(good);
            let cfg = DFlashConfig::from_json(&config).expect("in-range draft vocab");
            assert_eq!(cfg.draft_vocab(), good);
            assert_eq!(cfg.has_reduced_draft_vocab(), good != vocab);
        }
        // Absent means the target vocabulary.
        let mut config = layout.clone();
        config.as_object_mut().unwrap().remove("draft_vocab_size");
        let cfg = DFlashConfig::from_json(&config).expect("no draft vocab key");
        assert_eq!(cfg.draft_vocab_size, None);
        assert_eq!(cfg.draft_vocab(), vocab);
    }
    // The range is checked on every layout, the LFM2 one included.
    let mut lfm2 = published_dspark_config();
    lfm2["draft_vocab_size"] = json!(128001);
    assert!(DFlashConfig::from_json(&lfm2).is_err());
}

#[test]
fn missing_num_target_layers_is_not_derived_from_taps() {
    // The Speculators layout declares no depth. Upstream `DFlashConfig`
    // would derive `max(taps) + 1 = 34`; the Qwen3-4B target has 36 layers,
    // and the DSpark loader leaves the depth to the bound target.
    let cfg = DFlashConfig::from_json(&speculators_dspark_config()).expect("Speculators config");
    assert_eq!(cfg.num_target_layers, None);
    assert_ne!(cfg.num_target_layers, Some(34), "not max(taps) + 1");
    assert_ne!(cfg.num_target_layers, Some(32), "not the Qwen 3.5 default");

    // The OpenBMB layout without its `num_target_layers` reads the same way.
    let mut openbmb = openbmb_dspark_config();
    openbmb.as_object_mut().unwrap().remove("num_target_layers");
    let cfg = DFlashConfig::from_json(&openbmb).expect("OpenBMB without a depth");
    assert_eq!(cfg.num_target_layers, None);

    // A declared depth is kept, on either layout.
    let mut speculators = speculators_dspark_config();
    speculators["num_target_layers"] = json!(36);
    let cfg = DFlashConfig::from_json(&speculators).expect("Speculators with a depth");
    assert_eq!(cfg.num_target_layers, Some(36));

    // Layouts that predate the change keep their defaults: a plain DFlash
    // config without the key still reads the Qwen 3.5 depth.
    assert_eq!(
        DFlashConfig::from_json(&json!({}))
            .unwrap()
            .num_target_layers,
        Some(32)
    );
}

/// What `block_size` counts, at the boundaries: a Speculators checkpoint
/// with `sample_from_anchor: false`, or any DSpark config with
/// `dspark_bonus_anchor`, already counts the anchor row.
#[test]
fn dspark_block_size_counts_the_anchor_only_when_the_checkpoint_says_so() {
    let mut no_anchor_sample = speculators_dspark_config();
    no_anchor_sample["sample_from_anchor"] = json!(false);
    let cfg = DFlashConfig::from_json(&no_anchor_sample).expect("sample_from_anchor false");
    assert!(!cfg.sample_from_anchor);
    assert!(cfg.dspark_bonus_anchor);
    assert_eq!(cfg.verify_width(), 7);

    // The OpenBMB layout is not Speculators: `sample_from_anchor: false`
    // alone does not change what its block counts.
    let mut openbmb = openbmb_dspark_config();
    openbmb["sample_from_anchor"] = json!(false);
    let cfg = DFlashConfig::from_json(&openbmb).expect("OpenBMB sample_from_anchor false");
    assert!(!cfg.dspark_bonus_anchor);
    assert_eq!(cfg.verify_width(), 8);
    openbmb["dspark_bonus_anchor"] = json!(true);
    let cfg = DFlashConfig::from_json(&openbmb).expect("OpenBMB bonus anchor");
    assert_eq!(cfg.verify_width(), 7);

    // Block size 1: one proposal, a two-row verify block, the narrowest the
    // round loop drafts at. Block size 0 leaves a one-row width, which the
    // pairing gate refuses (`dspark_config_pairing_error`).
    let mut one = openbmb_dspark_config();
    one["block_size"] = json!(1);
    let cfg = DFlashConfig::from_json(&one).expect("block size 1");
    assert_eq!(cfg.verify_width(), 2);
    assert_eq!(cfg.runtime_verify_width(), 2);
    one["block_size"] = json!(0);
    let cfg = DFlashConfig::from_json(&one).expect("block size 0 parses");
    assert_eq!(cfg.runtime_verify_width(), 1);
    // A block that already includes the anchor and has one row drafts
    // nothing either.
    one["block_size"] = json!(1);
    one["dspark_bonus_anchor"] = json!(true);
    assert_eq!(
        DFlashConfig::from_json(&one)
            .unwrap()
            .runtime_verify_width(),
        1
    );
}

/// Causality follows upstream `DFlashAttention`: an explicit `is_causal`
/// wins, and when it is unset a layer is causal exactly when it slides and
/// `sliding_window_non_causal` is false.
#[test]
fn dspark_layer_causality_follows_the_layer_type_unless_set() {
    let mut mixed = speculators_dspark_config();
    mixed["transformer_layer_config"]["layer_types"] = json!([
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention"
    ]);
    let cfg = DFlashConfig::from_json(&mixed).expect("mixed layer types");
    assert_eq!(
        (0..5).map(|l| cfg.layer_is_causal(l)).collect::<Vec<_>>(),
        vec![true, false, true, false, true]
    );
    assert_eq!(
        (0..5)
            .map(|l| cfg.layer_sliding_window(l))
            .collect::<Vec<_>>(),
        vec![Some(2048), None, Some(2048), None, Some(2048)]
    );
    // A layer index past the stack is not sliding and not causal.
    assert_eq!(cfg.layer_sliding_window(5), None);
    assert!(!cfg.layer_is_causal(5));

    let mut non_causal = speculators_dspark_config();
    non_causal["sliding_window_non_causal"] = json!(true);
    let cfg = DFlashConfig::from_json(&non_causal).expect("non-causal sliding");
    assert!((0..5).all(|l| !cfg.layer_is_causal(l)));
    assert!((0..5).all(|l| cfg.layer_sliding_window(l) == Some(2048)));

    let mut forced = openbmb_dspark_config();
    forced["is_causal"] = json!(true);
    let cfg = DFlashConfig::from_json(&forced).expect("explicit causal");
    assert!((0..5).all(|l| cfg.layer_is_causal(l)));
    let mut forced_off = speculators_dspark_config();
    forced_off["is_causal"] = json!(false);
    let cfg = DFlashConfig::from_json(&forced_off).expect("explicit non-causal");
    assert!((0..5).all(|l| !cfg.layer_is_causal(l)));
}

/// The sliding fields are bounded at load: the window sizes a cache and a
/// mask, and a list that does not match the stack would index past it.
#[test]
fn dspark_sliding_fields_are_bounded_at_load() {
    for bad in [json!(null), json!(0), json!(1), json!(u64::MAX >> 8)] {
        let mut config = speculators_dspark_config();
        config["transformer_layer_config"]["sliding_window"] = bad.clone();
        let err = DFlashConfig::from_json(&config).expect_err("bad sliding window");
        assert!(err.contains("sliding_window"), "{bad}: {err}");
    }
    // The smallest window: one context row plus the proposal row itself.
    let mut two = speculators_dspark_config();
    two["transformer_layer_config"]["sliding_window"] = json!(2);
    let cfg = DFlashConfig::from_json(&two).expect("window 2");
    assert_eq!(cfg.layer_sliding_window(0), Some(2));

    let mut short = speculators_dspark_config();
    short["transformer_layer_config"]["layer_types"] = json!(["sliding_attention"]);
    let err = DFlashConfig::from_json(&short).expect_err("layer_types shorter than the stack");
    assert!(err.contains("layer_types has 1 entries"), "{err}");

    let mut unknown = openbmb_dspark_config();
    unknown["layer_types"][2] = json!("local_attention");
    let err = DFlashConfig::from_json(&unknown).expect_err("unknown layer type");
    assert!(err.contains("local_attention"), "{err}");
}

/// A field upstream implements and this port does not is refused by name
/// rather than dropped.
#[test]
fn unsupported_dspark_fields_are_refused_by_name() {
    let cases: [(&str, serde_json::Value); 7] = [
        ("hidden_act", json!("gelu")),
        ("markov_head_type", json!("gated")),
        ("projector_type", json!("dflash")),
        ("attention_sink_bias", json!(true)),
        ("final_logit_softcapping", json!(30.0)),
        ("draft_window_size", json!(512)),
        (
            "rope_parameters",
            json!({"rope_theta": 5000000, "rope_type": "yarn", "factor": 4.0}),
        ),
    ];
    for (key, value) in cases {
        let mut config = openbmb_dspark_config();
        config[key] = value;
        let err = DFlashConfig::from_json(&config).expect_err(key);
        let named = if key == "rope_parameters" {
            "rope_scaling"
        } else {
            key
        };
        assert!(err.contains(named), "{key}: {err}");
    }
    // The values the published checkpoints carry are accepted.
    let mut explicit = openbmb_dspark_config();
    explicit["attention_sink_bias"] = json!(false);
    explicit["final_logit_softcapping"] = json!(null);
    explicit["draft_window_size"] = json!(0);
    assert!(DFlashConfig::from_json(&explicit).is_ok());
}

/// A plain DFlash checkpoint is untouched by the DSpark-only fields: the
/// published `z-lab/Qwen3.5-4B-DFlash` config declares sliding layers, and
/// its attention must stay what it was before issue #3261.
#[test]
fn plain_dflash_configs_ignore_the_dspark_only_fields() {
    let cfg = DFlashConfig::from_json(&json!({
        "architectures": ["DFlashDraftModel"],
        "model_type": "qwen3",
        "num_hidden_layers": 6,
        "layer_types": [
            "sliding_attention", "sliding_attention", "sliding_attention",
            "sliding_attention", "sliding_attention", "full_attention"
        ],
        "sliding_window": 4096,
        "num_target_layers": 32,
        "dflash_config": {"mask_token_id": 248077, "target_layer_ids": [1, 5, 9, 13]},
    }))
    .expect("plain DFlash config with sliding layer types");
    assert!(!cfg.is_dspark());
    assert!(!cfg.has_sliding_layers());
    assert!((0..6).all(|l| cfg.layer_sliding_window(l).is_none()));
    assert!((0..6).all(|l| !cfg.layer_is_causal(l)));
    assert_eq!(cfg.num_target_layers, Some(32));
    assert_eq!(cfg.verify_width(), cfg.block_size);

    // Nor is it held to the DSpark rules for them: a list that does not
    // match the stack, or sliding layers with no window, still load.
    assert!(
        DFlashConfig::from_json(&json!({
            "layer_types": ["sliding_attention"],
            "sliding_window": null,
        }))
        .is_ok()
    );
}

#[test]
fn probe_accepts_the_speculators_and_openbmb_markers() {
    // Each marker on its own.
    assert!(is_dflash_drafter_config(&json!({
        "speculators_model_type": "dspark",
    })));
    assert!(is_dflash_drafter_config(&json!({
        "architectures": ["Qwen3DSparkModel"],
        "model_type": "qwen3",
    })));
    // Another Speculators algorithm is not a DSpark drafter, and the
    // Speculators `architectures` entry alone is not keyed on.
    assert!(!is_dflash_drafter_config(&json!({
        "speculators_model_type": "eagle3",
    })));
    assert!(!is_dflash_drafter_config(&json!({
        "architectures": ["DSparkDraftModel"],
    })));
    // The two targets are full models.
    assert!(!is_dflash_drafter_config(&json!({
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
    })));
    assert!(!is_dflash_drafter_config(&json!({
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
    })));
}
