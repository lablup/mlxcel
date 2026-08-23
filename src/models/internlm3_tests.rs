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

//! Config-level tests for InternLM3's rotary schedule (#1324).
//!
//! These parse the real `config.json` shape rather than constructing
//! `ModelArgs` by hand, because the defect lived in the step between the config
//! and the schedule: the block parsed, and the schedule ignored it.

use super::ModelArgs;
use crate::models::dynamic_ntk_rope::DynamicNtkRopeMode;

/// `mlx-community/internlm3-8b-instruct-4bit`'s config, trimmed to the fields
/// `ModelArgs` reads. The `rope_scaling` block, `rope_theta` and
/// `max_position_embeddings` are verbatim from the checkpoint.
const VALIDATION_CONFIG: &str = r#"{
    "model_type": "internlm3",
    "hidden_size": 4096,
    "num_hidden_layers": 48,
    "intermediate_size": 10240,
    "num_attention_heads": 32,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-05,
    "vocab_size": 128512,
    "bias": false,
    "qkv_bias": false,
    "max_position_embeddings": 32768,
    "rope_theta": 50000000,
    "rope_scaling": {"factor": 6.0, "rope_type": "dynamic"},
    "tie_word_embeddings": false,
    "quantization": {"group_size": 64, "bits": 4}
}"#;

fn args(json: &str) -> ModelArgs {
    serde_json::from_str(json).unwrap_or_else(|err| panic!("config must parse: {err}"))
}

#[test]
fn the_validation_checkpoint_resolves_to_an_unscaled_dynamic_schedule() {
    let args = args(VALIDATION_CONFIG);
    let rope = args.rope().expect("the checkpoint's block must resolve");

    assert_eq!(rope.mode(), DynamicNtkRopeMode::Dynamic { factor: 6.0 });
    // The defect: this returned 2.0, so every position was doubled.
    assert_eq!(rope.scale(), 1.0);
    // 56 is the prompt length the greedy-parity gate uses; well inside 32768,
    // so the base is exactly `rope_theta`.
    assert_eq!(rope.base_for(56), 50_000_000.0);
    assert_eq!(rope.base_for(32768), 50_000_000.0);
    assert!(rope.base_for(65536) > 50_000_000.0);
}

#[test]
fn an_absent_block_also_leaves_positions_unscaled() {
    // The other half of the same `unwrap_or(2.0)`: a config with no
    // `rope_scaling` at all got the doubled schedule too, which is what any
    // future InternLM3 export without the block would have decoded with.
    let stripped = VALIDATION_CONFIG.replace(
        "\"rope_scaling\": {\"factor\": 6.0, \"rope_type\": \"dynamic\"},",
        "",
    );
    let args = args(&stripped);
    assert!(args.rope_scaling.is_none());
    let rope = args.rope().expect("an absent block must resolve");
    assert_eq!(rope.mode(), DynamicNtkRopeMode::Default);
    assert_eq!(rope.scale(), 1.0);
    assert_eq!(rope.base_for(1_000_000), 50_000_000.0);
}

#[test]
fn a_linear_block_still_divides_positions_by_the_factor() {
    let json = VALIDATION_CONFIG.replace(
        "{\"factor\": 6.0, \"rope_type\": \"dynamic\"}",
        "{\"factor\": 4.0, \"rope_type\": \"linear\"}",
    );
    let rope = args(&json).rope().expect("linear must resolve");
    assert_eq!(rope.mode(), DynamicNtkRopeMode::Linear { factor: 4.0 });
    assert_eq!(rope.scale(), 0.25);
}

#[test]
fn an_unimplemented_scheme_fails_the_load_rather_than_decoding_wrongly() {
    let json = VALIDATION_CONFIG.replace(
        "{\"factor\": 6.0, \"rope_type\": \"dynamic\"}",
        "{\"factor\": 8.0, \"rope_type\": \"yarn\"}",
    );
    let err = args(&json).rope().expect_err("yarn must be rejected");
    assert!(err.contains("yarn"), "error must name the scheme: {err}");
}

#[test]
fn a_config_spelling_the_scheme_key_as_type_still_resolves() {
    // No published InternLM3 export spells it this way, but the shared reader
    // accepts it and the family should not care which spelling arrives.
    let json = VALIDATION_CONFIG.replace(
        "{\"factor\": 6.0, \"rope_type\": \"dynamic\"}",
        "{\"factor\": 6.0, \"type\": \"dynamic\"}",
    );
    let rope = args(&json)
        .rope()
        .expect("the `type` spelling must resolve");
    assert_eq!(rope.mode(), DynamicNtkRopeMode::Dynamic { factor: 6.0 });
}
