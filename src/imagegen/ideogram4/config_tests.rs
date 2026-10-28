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

use serde_json::json;

use super::*;

fn extra(v: Value) -> Map<String, Value> {
    v.as_object().unwrap().clone()
}

#[test]
fn the_checkpoint_transformer_config_parses() {
    let config = Ideogram4TransformerConfig::from_json(&json!({
        "_class_name": "Ideogram4Transformer2DModel",
        "adaln_dim": 512, "attention_head_dim": 256, "in_channels": 128,
        "intermediate_size": 12288, "llm_features_dim": 53248,
        "mrope_section": [24, 20, 20], "norm_eps": 1e-05,
        "num_attention_heads": 18, "num_layers": 34, "rope_theta": 5000000
    }))
    .unwrap();
    assert_eq!(config, Ideogram4TransformerConfig::default());
    assert_eq!(config.head_dim(), 256);
}

#[test]
fn malformed_transformer_configs_are_rejected() {
    assert!(Ideogram4TransformerConfig::from_json(&json!({"num_attention_heads": 7})).is_err());
    assert!(Ideogram4TransformerConfig::from_json(&json!({"mrope_section": [1, 2]})).is_err());
    // 3 * 50 - 2 = 148 is past head_dim / 2 = 128.
    assert!(
        Ideogram4TransformerConfig::from_json(&json!({"mrope_section": [24, 50, 20]})).is_err()
    );
    assert!(Ideogram4TransformerConfig::from_json(&json!({"num_layers": 0})).is_err());
}

#[test]
fn dimensions_follow_upstream() {
    validate_dimensions(512, 512).unwrap();
    validate_dimensions(2048, 352).unwrap();
    for (w, h, want) in [
        (240, 512, "width must be in [256, 2048], got 240"),
        (512, 2064, "height must be in [256, 2048], got 2064"),
        (520, 512, "width must be divisible by 16, got 520"),
        (2048, 336, "aspect ratio must be at most 6:1, got 2048x336"),
    ] {
        assert_eq!(validate_dimensions(w, h).unwrap_err().to_string(), want);
    }
}

#[test]
fn defaults_resolve_to_the_default_preset_schedule() {
    let options = Ideogram4Options::from_extra(&Map::new()).unwrap();
    assert!(options.auto_json_caption);
    let plan = options.plan(None, None).unwrap();
    assert_eq!(plan.steps, 20);
    assert_eq!(plan.guidance, 7.0);
    assert_eq!((plan.mu, plan.std), (0.0, 1.75));
    let schedule = plan.guidance_schedule.as_ref().unwrap();
    assert_eq!(schedule.len(), 20);
    // The loop runs from index 19 down, so 7.0 comes first and 3.0 last.
    assert_eq!(
        (
            plan.guidance_at(19),
            plan.guidance_at(1),
            plan.guidance_at(0)
        ),
        (7.0, 3.0, 3.0)
    );
}

#[test]
fn explicit_guidance_or_steps_disable_the_schedule() {
    let turbo =
        Ideogram4Options::from_extra(&extra(json!({"sampler_preset": "V4_TURBO_12"}))).unwrap();
    // `--steps` equal to the preset's keeps the schedule.
    let plan = turbo.plan(Some(12), None).unwrap();
    assert_eq!(plan.guidance_schedule.as_ref().map(Vec::len), Some(12));
    assert_eq!((plan.mu, plan.std), (0.5, 1.75));
    // Another step count runs the preset's final guidance throughout.
    let plan = turbo.plan(Some(8), None).unwrap();
    assert_eq!(
        (plan.steps, plan.guidance, plan.guidance_schedule),
        (8, 7.0, None)
    );
    // An explicit guidance wins over the schedule.
    let plan = turbo.plan(None, Some(4.5)).unwrap();
    assert_eq!(
        (plan.steps, plan.guidance, plan.guidance_schedule),
        (12, 4.5, None)
    );
}

#[test]
fn an_explicit_guidance_schedule_must_match_the_steps() {
    let options = Ideogram4Options::from_extra(&extra(json!({
        "guidance_schedule": [1, 2, 3], "mu": 0.25, "std": 1.0
    })))
    .unwrap();
    let plan = options.plan(Some(3), Some(9.0)).unwrap();
    assert_eq!(plan.guidance_schedule, Some(vec![1.0, 2.0, 3.0]));
    assert_eq!((plan.mu, plan.std), (0.25, 1.0));
    assert_eq!(
        options.plan(None, None).unwrap_err().to_string(),
        "guidance_schedule must have 20 items, got 3"
    );
}

#[test]
fn bad_options_are_rejected() {
    for (bad, needle) in [
        (
            json!({"sampler_preset": "V5"}),
            "expected one of ['V4_DEFAULT_20', 'V4_QUALITY_48', 'V4_TURBO_12']",
        ),
        (
            json!({"sampler_preset": 3}),
            "sampler_preset must be a string",
        ),
        (
            json!({"auto_json_caption": "no"}),
            "auto_json_caption must be a boolean",
        ),
        (
            json!({"guidance_schedule": "7"}),
            "guidance_schedule must be a list",
        ),
        (
            json!({"guidance_schedule": [1, "x"]}),
            "guidance_schedule entries",
        ),
        (json!({"mu": "a"}), "mu must be a finite number"),
        (json!({"std": 0}), "std must be positive"),
        (
            json!({"prompt_expansion_model": "mlx-community/Qwen3-4B"}),
            "prompt_expansion_model is not supported",
        ),
    ] {
        let err = Ideogram4Options::from_extra(&extra(bad))
            .unwrap_err()
            .to_string();
        assert!(err.contains(needle), "{err}");
    }
    // An explicit null is the same as absent.
    assert!(Ideogram4Options::from_extra(&extra(json!({"prompt_expansion_model": null}))).is_ok());
}
