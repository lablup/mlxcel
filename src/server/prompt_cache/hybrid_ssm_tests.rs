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

/// Z1T (#2364) keeps AFT-conv pool sums and a conv window per layer, so APC
/// must be disabled for it whether it is found by `model_type` or by the
/// architecture name.
#[test]
fn z1t_is_detected_as_recurrent() {
    assert!(is_hybrid_ssm_model_type("z1t"));
    assert_eq!(
        detect_hybrid_ssm(&json!({ "architectures": ["Z1TForCausalLM"] })).as_deref(),
        Some("Z1TForCausalLM")
    );
}

/// The original LongCat-Flash-Lite-Sparse config has no `model_type`; its
/// `LongcatCausalLM` architecture alone must turn APC off (#2362).
#[test]
fn longcat_sparse_is_detected_without_model_type() {
    assert!(is_hybrid_ssm_model_type("longcat_flash_sparse"));
    assert_eq!(
        detect_hybrid_ssm(&json!({ "architectures": ["LongcatCausalLM"] })).as_deref(),
        Some("LongcatCausalLM")
    );
}

/// Baichuan-M1 keeps a per-sequence K/V convolution state next to its dense
/// KV cache (#2567). A restored KV prefix would arrive without the conv state
/// of its last token, so APC must treat the family as recurrent.
#[test]
fn baichuan_m1_is_detected_as_recurrent() {
    assert!(is_hybrid_ssm_model_type("baichuan_m1"));
    assert_eq!(
        detect_hybrid_ssm(&json!({ "model_type": "baichuan_m1" })).as_deref(),
        Some("baichuan_m1")
    );
    assert_eq!(
        detect_hybrid_ssm(&json!({ "architectures": ["BaichuanM1ForCausalLM"] })).as_deref(),
        Some("BaichuanM1ForCausalLM")
    );
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

/// Every recurrent `model_type` epic #564 covers (#3341). The XLA guard calls
/// [`recurrent_prefix_sharing_family`], which consults this list, so a family
/// the OpenXLA emitter cannot load yet is still kept out of prefix sharing.
const EPIC_RECURRENT_TYPES: &[&str] = &[
    "mamba",
    "mamba2",
    "falcon_mamba",
    "jamba",
    "nemotron_h",
    "falcon_h1",
    "plamo2",
    "qwen3_next",
    "qwen3_5",
    "kimi_linear",
    "rwkv7",
    "recurrent_gemma",
    "granitemoehybrid",
];

#[test]
fn every_epic_recurrent_type_is_refused_prefix_sharing_on_any_backend() {
    for &mt in EPIC_RECURRENT_TYPES {
        assert!(is_hybrid_ssm_model_type(mt), "{mt} must be listed");
        let cfg = json!({ "model_type": mt });
        for backend_says_recurrent in [false, true] {
            assert_eq!(
                recurrent_prefix_sharing_family(&cfg, backend_says_recurrent).as_deref(),
                Some(mt),
                "{mt} (backend recurrent = {backend_says_recurrent})"
            );
        }
    }
}

#[test]
fn backend_recurrent_state_refuses_an_unlisted_family() {
    let cfg = json!({ "model_type": "some_future_ssm" });
    assert!(recurrent_prefix_sharing_family(&cfg, false).is_none());
    assert_eq!(
        recurrent_prefix_sharing_family(&cfg, true).as_deref(),
        Some("some_future_ssm")
    );
    assert_eq!(
        recurrent_prefix_sharing_family(&json!({}), true).as_deref(),
        Some("recurrent")
    );
    assert!(recurrent_prefix_sharing_family(&json!({ "model_type": "llama" }), false).is_none());
}

/// The four families #3341 adds, by `model_type`, by the VLM `text_config`
/// spelling, and by architecture name alone.
#[test]
fn newly_listed_hybrids_are_detected_every_way() {
    for (mt, arch) in [
        ("falcon_h1", "FalconH1ForCausalLM"),
        ("plamo2", "Plamo2ForCausalLM"),
        ("qwen3_5", "Qwen3_5ForConditionalGeneration"),
        ("qwen3_5_moe", "Qwen3_5MoeForConditionalGeneration"),
        ("granitemoehybrid", "GraniteMoeHybridForCausalLM"),
    ] {
        assert_eq!(
            detect_hybrid_ssm(&json!({ "model_type": mt })).as_deref(),
            Some(mt)
        );
        assert_eq!(
            detect_hybrid_ssm(&json!({ "architectures": [arch] })).as_deref(),
            Some(arch),
            "{arch}"
        );
    }
    let qwen35_vlm = json!({
        "model_type": "some_wrapper",
        "text_config": { "model_type": "qwen3_5_text" },
    });
    assert_eq!(
        detect_hybrid_ssm(&qwen35_vlm).as_deref(),
        Some("qwen3_5_text")
    );
    assert!(is_hybrid_ssm_model_type("plamo2vl"));
}

/// The new architecture fragments must not catch the attention-only siblings.
#[test]
fn new_fragments_do_not_match_attention_siblings() {
    for arch in [
        "Qwen3ForCausalLM",
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen3MoeForCausalLM",
        "GraniteMoeForCausalLM",
        "GraniteForCausalLM",
        "PlamoForCausalLM",
        "FalconForCausalLM",
    ] {
        assert!(
            detect_hybrid_ssm(&json!({ "architectures": [arch] })).is_none(),
            "{arch} must not be detected as hybrid"
        );
    }
}

/// HF Mamba2 and Nemotron-H configs carry a bare `Infinity`. A strict parse
/// used to fail there and leave APC on for exactly those families (#3341).
#[test]
fn detect_from_path_tolerates_bare_infinity() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type": "mamba2", "time_step_limit": [0.0, Infinity]}"#,
    )
    .unwrap();
    assert_eq!(
        detect_hybrid_ssm_from_path(dir.path()).unwrap().as_deref(),
        Some("mamba2")
    );
}

fn apc_requested() -> PromptCacheConfig {
    let mut config = PromptCacheConfig::default();
    config.apc.enabled = true;
    assert!(config.apc_enabled());
    config
}

#[test]
fn disable_apc_turns_apc_off_only_for_listed_families() {
    for (config_json, hybrid) in [
        (r#"{"model_type": "granitemoehybrid"}"#, true),
        (r#"{"model_type": "falcon_h1"}"#, true),
        (r#"{"model_type": "plamo2"}"#, true),
        (
            r#"{"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text"}}"#,
            true,
        ),
        (r#"{"model_type": "llama"}"#, false),
    ] {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), config_json).unwrap();
        let mut config = apc_requested();
        let snapshot_bytes = config.snapshot_capacity_bytes;
        let checkpoint_bytes = config.state_checkpoint_capacity_bytes;
        assert_eq!(
            disable_apc_for_hybrid_ssm(&mut config, dir.path()).is_some(),
            hybrid,
            "{config_json}"
        );
        assert_eq!(config.apc_enabled(), !hybrid, "{config_json}");
        // Only APC changes: the store, its snapshots and the #3047
        // turn-boundary checkpoints keep their budgets.
        assert!(config.is_enabled());
        assert_eq!(config.snapshot_capacity_bytes, snapshot_bytes);
        assert_eq!(config.state_checkpoint_capacity_bytes, checkpoint_bytes);
    }
}

/// The local checkpoints of two newly listed families (skipped when absent;
/// `MLXCEL_REQUIRE_MODELS=1` makes a missing one fail).
#[test]
fn local_qwen35_and_granite_h_checkpoints_turn_apc_off() {
    for (name, family) in [
        ("models/qwen3.5-0.8b-4bit", "qwen3_5"),
        ("models/mlx/granite-4.0-h-tiny-4bit", "granitemoehybrid"),
    ] {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join(name);
        if !dir.join("config.json").is_file() {
            assert!(
                std::env::var_os("MLXCEL_REQUIRE_MODELS").is_none(),
                "{name} is required but missing"
            );
            eprintln!("skipping: {name} not found");
            continue;
        }
        let mut config = apc_requested();
        assert_eq!(
            disable_apc_for_hybrid_ssm(&mut config, &dir).as_deref(),
            Some(family),
            "{name}"
        );
        assert!(!config.apc_enabled(), "{name}");
    }
}
