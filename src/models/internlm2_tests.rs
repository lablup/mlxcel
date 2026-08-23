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

//! Config-level tests for InternLM2's rotary schedule (#1324).
//!
//! The defect here was one stage earlier than `internlm3`'s: `ModelArgs` did
//! not declare `rope_scaling`, so the block never became a value at all. A
//! parse test is the only thing that catches that, because a dropped block
//! produces the correct schedule for every sequence inside
//! `max_position_embeddings` and is invisible below it.

use super::ModelArgs;
use crate::models::dynamic_ntk_rope::DynamicNtkRopeMode;

/// `models/internlm2-7b-4bit`'s config, trimmed to the fields `ModelArgs`
/// reads. Note the `type` spelling, which is what this family uses.
const CHECKPOINT_CONFIG: &str = r#"{
    "model_type": "internlm2",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "intermediate_size": 14336,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-05,
    "vocab_size": 92544,
    "bias": false,
    "max_position_embeddings": 32768,
    "rope_theta": 1000000,
    "rope_scaling": {"type": "dynamic", "factor": 2.0},
    "tie_word_embeddings": false,
    "quantization": {"group_size": 64, "bits": 4}
}"#;

fn args(json: &str) -> ModelArgs {
    serde_json::from_str(json).unwrap_or_else(|err| panic!("config must parse: {err}"))
}

#[test]
fn the_checkpoints_dynamic_block_is_no_longer_dropped() {
    let args = args(CHECKPOINT_CONFIG);
    assert!(
        args.rope_scaling.is_some(),
        "the block must survive deserialization"
    );

    let rope = args.rope().expect("the checkpoint's block must resolve");
    assert_eq!(rope.mode(), DynamicNtkRopeMode::Dynamic { factor: 2.0 });
    // Positions were already unscaled here, and stay that way.
    assert_eq!(rope.scale(), 1.0);
    // Inside `max_position_embeddings` the schedule is bit-identical to the one
    // that shipped, which is why no ordinary prompt can observe this change.
    assert_eq!(rope.base_for(56), 1_000_000.0);
    assert_eq!(rope.base_for(32768), 1_000_000.0);
    // Past it, the base finally moves.
    assert!(rope.base_for(65536) > 1_000_000.0);
}

#[test]
fn a_config_without_the_block_keeps_the_plain_schedule() {
    let stripped = CHECKPOINT_CONFIG.replace(
        "\"rope_scaling\": {\"type\": \"dynamic\", \"factor\": 2.0},",
        "",
    );
    let rope = args(&stripped)
        .rope()
        .expect("an absent block must resolve");
    assert_eq!(rope.mode(), DynamicNtkRopeMode::Default);
    assert_eq!(rope.scale(), 1.0);
    assert_eq!(rope.base_for(1_000_000), 1_000_000.0);
}

#[test]
fn an_unimplemented_scheme_fails_the_load() {
    let json = CHECKPOINT_CONFIG.replace(
        "{\"type\": \"dynamic\", \"factor\": 2.0}",
        "{\"type\": \"yarn\", \"factor\": 4.0}",
    );
    let err = args(&json).rope().expect_err("yarn must be rejected");
    assert!(err.contains("yarn"), "error must name the scheme: {err}");
}
