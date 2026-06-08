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

//! Tests for the APC hybrid-SSM opt-out detector.

use super::*;
use serde_json::json;

#[test]
fn known_hybrid_model_types_match() {
    for mt in [
        "jamba",
        "mamba",
        "mamba2",
        "nemotron_h",
        "gated_delta",
        "kimi_linear",
        "qwen3_next",
    ] {
        assert!(
            is_hybrid_ssm_model_type(mt),
            "{mt} should be detected as hybrid"
        );
    }
}

/// Drift guard (#124): every entry in the authoritative
/// [`HYBRID_SSM_MODEL_TYPES`] constant must be detected by both the bare
/// `model_type` predicate and the full `config.json` detector. Iterating the
/// constant itself means a future addition (or the alias rows in the module
/// doc table) cannot fall out of sync with the detection logic. This is the
/// unified-cache carve-out's first line of defence: if a recurrent family is
/// in the list but not detected, APC would not be disabled for it.
#[test]
fn all_hybrid_ssm_model_types_round_trip() {
    assert!(
        !HYBRID_SSM_MODEL_TYPES.is_empty(),
        "the hybrid-SSM carve-out list must not be empty"
    );
    for &mt in HYBRID_SSM_MODEL_TYPES {
        assert!(
            is_hybrid_ssm_model_type(mt),
            "{mt} is in HYBRID_SSM_MODEL_TYPES but is_hybrid_ssm_model_type rejected it"
        );
        let cfg = json!({ "model_type": mt });
        assert_eq!(
            detect_hybrid_ssm(&cfg).as_deref(),
            Some(mt),
            "detect_hybrid_ssm must flag the top-level model_type {mt}"
        );
    }
}

#[test]
fn case_and_whitespace_insensitive() {
    assert!(is_hybrid_ssm_model_type("JAMBA"));
    assert!(is_hybrid_ssm_model_type("  qwen3_next  "));
    assert!(is_hybrid_ssm_model_type("Mamba2"));
}

#[test]
fn non_hybrid_model_types_do_not_match() {
    for mt in ["llama", "qwen2", "qwen3", "gemma3_text", "phi3", ""] {
        assert!(
            !is_hybrid_ssm_model_type(mt),
            "{mt} must not be detected as hybrid"
        );
    }
}

#[test]
fn detect_top_level_model_type() {
    let cfg = json!({"model_type": "jamba", "hidden_size": 4096});
    assert_eq!(detect_hybrid_ssm(&cfg).as_deref(), Some("jamba"));
}

#[test]
fn detect_nested_text_config_model_type() {
    // Common VLM shape: outer config.json has text_config.model_type.
    let cfg = json!({
        "model_type": "vlm-wrapper",
        "text_config": {"model_type": "nemotron_h"},
    });
    assert_eq!(detect_hybrid_ssm(&cfg).as_deref(), Some("nemotron_h"));
}

#[test]
fn detect_architectures_array_fallback() {
    let cfg = json!({"architectures": ["JambaForCausalLM"]});
    let detected = detect_hybrid_ssm(&cfg);
    assert!(detected.is_some());
    assert_eq!(detected.unwrap(), "JambaForCausalLM");
}

#[test]
fn detect_architectures_nemotronh_normalisation() {
    // Verify our underscore-stripping/lowercase normalisation handles
    // "NemotronHForCausalLM" as a hybrid match.
    let cfg = json!({"architectures": ["NemotronHForCausalLM"]});
    let detected = detect_hybrid_ssm(&cfg);
    assert_eq!(detected.as_deref(), Some("NemotronHForCausalLM"));
}

#[test]
fn detect_returns_none_for_non_hybrid() {
    let cfg = json!({
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
    });
    assert!(detect_hybrid_ssm(&cfg).is_none());
}

#[test]
fn detect_returns_none_when_config_lacks_model_type() {
    let cfg = json!({"hidden_size": 4096});
    assert!(detect_hybrid_ssm(&cfg).is_none());
}

#[test]
fn detect_top_level_takes_precedence_over_text_config() {
    // If both fields are present and one of them is hybrid, we still flag
    // as hybrid. The top-level wins on order of evaluation.
    let cfg = json!({
        "model_type": "qwen3_next",
        "text_config": {"model_type": "llama"},
    });
    assert_eq!(detect_hybrid_ssm(&cfg).as_deref(), Some("qwen3_next"));
}

#[test]
fn detect_returns_first_arch_match() {
    // First entry in the architectures array is the canonical one. We
    // return its raw spelling (preserved camelCase) for diagnostic logs.
    let cfg = json!({
        "architectures": ["FalconMambaForCausalLM"],
        "model_type": "unknown_thing",
    });
    let detected = detect_hybrid_ssm(&cfg);
    assert_eq!(detected.as_deref(), Some("FalconMambaForCausalLM"));
}
