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
    assert_eq!(cfg.num_target_layers, 30);
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
