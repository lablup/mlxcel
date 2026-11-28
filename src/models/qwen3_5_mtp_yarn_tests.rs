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

//! YaRN in the Qwen 3.5 MTP drafter (issue #3479): the drafter's table and
//! attention factor come from the same builder as the target's, and the
//! binary-side `drafter_loader::load_drafter` installs them.

use std::path::Path;

use mlxcel_core::drafter::DrafterError;
use mlxcel_core::drafter::qwen3_5_mtp::Qwen35MtpTextConfig;

use super::drafter_loader::{load_drafter, load_qwen35_mtp, qwen35_mtp_yarn};
use super::embedding_test_support::write_f32_safetensors;
use super::qwen3_5::{qwen35_yarn, yarn_inv_freq};
use super::qwen3_5_tests::{
    assert_matches_hf_inv_freq, qwen3_8_27b_text_config, sensenova_yarn_parameters, yarn_config,
    yarn_parameters_with,
};

/// The Qwen3.8-27B text config with `rope_parameters` replaced, parsed as
/// the drafter's `text_config` (which mirrors the target's).
fn drafter_text_config(rope_parameters: serde_json::Value) -> Qwen35MtpTextConfig {
    let mut value = qwen3_8_27b_text_config();
    value["rope_parameters"] = rope_parameters;
    serde_json::from_value(value).expect("the target text_config parses as the drafter's")
}

#[test]
fn drafter_yarn_table_matches_the_target() {
    for parameters in [
        sensenova_yarn_parameters(),
        // The original context then comes from `max_position_embeddings`,
        // which the drafter must read as the target does.
        yarn_parameters_with(serde_json::json!({"original_max_position_embeddings": null})),
    ] {
        let target = yarn_config(parameters.clone())
            .yarn()
            .expect("valid")
            .expect("target yarn");
        let drafter = qwen35_mtp_yarn(&drafter_text_config(parameters))
            .expect("valid")
            .expect("drafter yarn");
        let target_inv_freq = yarn_inv_freq(&target.freqs);
        assert_eq!(yarn_inv_freq(&drafter.freqs), target_inv_freq);
        assert_matches_hf_inv_freq(&target_inv_freq);
        assert_eq!(drafter.attention_factor, target.attention_factor);
    }
}

#[test]
fn drafter_without_yarn_has_no_table() {
    let plain = drafter_text_config(serde_json::json!({
        "partial_rotary_factor": 0.25,
        "rope_theta": 10000000,
        "rope_type": "default"
    }));
    assert!(qwen35_mtp_yarn(&plain).expect("valid").is_none());

    let mut value = qwen3_8_27b_text_config();
    value
        .as_object_mut()
        .expect("object")
        .remove("rope_parameters");
    let absent: Qwen35MtpTextConfig = serde_json::from_value(value).expect("parses");
    assert!(qwen35_mtp_yarn(&absent).expect("valid").is_none());
}

#[test]
fn invalid_drafter_yarn_block_fails_with_the_target_message() {
    let parameters = yarn_parameters_with(serde_json::json!({"factor": null}));
    let Err(target) = yarn_config(parameters.clone()).yarn() else {
        panic!("the target refuses a YaRN block without a factor");
    };
    let Err(DrafterError::Config(drafter)) = qwen35_mtp_yarn(&drafter_text_config(parameters))
    else {
        panic!("the drafter refuses it as a config error");
    };
    assert_eq!(drafter, target);
}

// ── Through the loader ─────────────────────────────────────────────────────

/// Tiny YaRN block over the fixture geometry below (head_dim 4, partial
/// rotary 0.5, so 2 rotary dims).
fn tiny_yarn_parameters() -> serde_json::Value {
    serde_json::json!({
        "type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 4096,
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.5
    })
}

/// Write a tiny `qwen3_5_mtp` checkpoint (the 15-tensor inventory) with
/// `rope_parameters` in its `text_config`.
fn write_tiny_drafter(dir: &Path, rope_parameters: serde_json::Value) {
    let config = serde_json::json!({
        "model_type": "qwen3_5_mtp",
        "block_size": 3,
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "intermediate_size": 16,
            "vocab_size": 16,
            "rms_norm_eps": 1e-6,
            "rope_parameters": rope_parameters,
            "mtp_num_hidden_layers": 1,
            "tie_word_embeddings": false
        }
    });
    std::fs::write(dir.join("config.json"), config.to_string()).expect("write config");
    let tensor = |name: &str, shape: &[i32], value: f32| {
        let n: i32 = shape.iter().product();
        (name.to_string(), shape.to_vec(), vec![value; n as usize])
    };
    let tensors = [
        tensor("fc.weight", &[8, 16], 0.05),
        tensor("pre_fc_norm_embedding.weight", &[8], 1.0),
        tensor("pre_fc_norm_hidden.weight", &[8], 1.0),
        tensor("norm.weight", &[8], 1.0),
        tensor("layers.0.input_layernorm.weight", &[8], 1.0),
        tensor("layers.0.post_attention_layernorm.weight", &[8], 1.0),
        tensor("layers.0.self_attn.q_proj.weight", &[16, 8], 0.05),
        tensor("layers.0.self_attn.k_proj.weight", &[4, 8], 0.05),
        tensor("layers.0.self_attn.v_proj.weight", &[4, 8], 0.05),
        tensor("layers.0.self_attn.o_proj.weight", &[8, 8], 0.05),
        tensor("layers.0.self_attn.q_norm.weight", &[4], 1.0),
        tensor("layers.0.self_attn.k_norm.weight", &[4], 1.0),
        tensor("layers.0.mlp.gate_proj.weight", &[16, 8], 0.05),
        tensor("layers.0.mlp.up_proj.weight", &[16, 8], 0.05),
        tensor("layers.0.mlp.down_proj.weight", &[8, 16], 0.05),
    ];
    write_f32_safetensors(&dir.join("model.safetensors"), &tensors);
}

#[test]
fn loader_installs_the_yarn_table_on_the_drafter() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_tiny_drafter(dir.path(), tiny_yarn_parameters());
    let model = load_qwen35_mtp(dir.path()).expect("tiny YaRN drafter loads");
    let expected = qwen35_yarn(Some(&tiny_yarn_parameters()), 2, 10000.0, None)
        .expect("valid")
        .expect("yarn");
    let (freqs, factor) = model.yarn_rope().expect("the loader installs the table");
    assert_eq!(yarn_inv_freq(freqs), yarn_inv_freq(&expected.freqs));
    assert_eq!(factor, expected.attention_factor);
    assert_ne!(factor, 1.0, "factor 4 has a non-unit attention factor");

    // The public entry point (CLI, server drafter slot, speculative bench)
    // goes through the same arm.
    load_drafter(dir.path(), None).expect("loads through load_drafter");
}

#[test]
fn loader_leaves_a_plain_drafter_on_rope_theta() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_tiny_drafter(
        dir.path(),
        serde_json::json!({"rope_theta": 10000.0, "partial_rotary_factor": 0.5}),
    );
    let model = load_qwen35_mtp(dir.path()).expect("tiny drafter loads");
    assert!(model.yarn_rope().is_none());
}

#[test]
fn load_drafter_refuses_an_unbuildable_yarn_block() {
    // Core alone would load this drafter on plain `rope_theta`; the binary
    // entry point must refuse it as the target does.
    let dir = tempfile::tempdir().expect("tempdir");
    let mut parameters = tiny_yarn_parameters();
    parameters.as_object_mut().expect("object").remove("factor");
    write_tiny_drafter(dir.path(), parameters);
    let Err(err) = load_drafter(dir.path(), None) else {
        panic!("a YaRN block without a factor must fail the drafter load");
    };
    assert!(
        matches!(&err, DrafterError::Config(m) if m.contains("`factor` is missing")),
        "{err}"
    );
}
