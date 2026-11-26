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

use super::{NO_THINKING_ALPHA, ReasoningEffortSteering};
use serde_json::json;

/// The release's block (`KRAFTON/Raon-1-mini` at `6691a076`).
fn published() -> ReasoningEffortSteering {
    ReasoningEffortSteering::from_config_value(&json!({
        "reasoning_effort_steering": {
            "alpha": {"none": 0, "low": 0.3, "medium": 0.1, "high": 0.05, "xhigh": 0.02, "max": 0},
            "default": "high",
            "layers": [12, 14, 16, 18, 20]
        }
    }))
    .unwrap()
    .unwrap()
}

#[test]
fn every_published_level_resolves_to_its_alpha() {
    let s = published();
    s.validate(25).unwrap();
    for (level, alpha) in [
        ("low", 0.3),
        ("medium", 0.1),
        ("high", 0.05),
        ("xhigh", 0.02),
        ("max", 0.0),
        ("none", 0.1),
    ] {
        assert_eq!(s.alpha_for(Some(level)).unwrap(), alpha as f32, "{level}");
    }
    // No effort is the default level, `high`.
    assert_eq!(s.alpha_for(None).unwrap(), 0.05f32);
    // Spelling is normalized, not translated.
    assert_eq!(s.alpha_for(Some(" LOW ")).unwrap(), 0.3f32);
    assert_eq!(
        s.accepted_levels(),
        ["none", "low", "medium", "high", "xhigh", "max"]
    );
}

#[test]
fn none_uses_a_positive_config_alpha_and_otherwise_the_no_thinking_alpha() {
    let mut s = published();
    assert_eq!(s.alpha_for(Some("none")).unwrap(), NO_THINKING_ALPHA);
    s.alpha.insert("none".into(), 0.25);
    assert_eq!(s.alpha_for(Some("none")).unwrap(), 0.25f32);
    // A config without a `none` key still accepts `none`.
    s.alpha.remove("none");
    assert_eq!(s.alpha_for(Some("none")).unwrap(), NO_THINKING_ALPHA);
    assert!(s.accepted_levels().contains(&"none".to_string()));
}

#[test]
fn an_unknown_level_errors_with_the_accepted_set() {
    let s = published();
    for bad in ["minimal", "off", "disabled", "0", "extreme", ""] {
        let err = s.alpha_for(Some(bad)).unwrap_err();
        assert!(
            err.contains("none, low, medium, high, xhigh, max"),
            "{bad}: {err}"
        );
    }
}

#[test]
fn config_rules_are_enforced() {
    let mut s = published();
    s.default = "none".into();
    assert!(
        s.validate(25)
            .unwrap_err()
            .contains("default cannot be none")
    );

    let mut s = published();
    s.default = "ultra".into();
    assert!(
        s.validate(25)
            .unwrap_err()
            .contains("not one of the alpha keys")
    );

    let mut s = published();
    s.alpha.insert("low".into(), -0.1);
    assert!(s.validate(25).unwrap_err().contains("non-negative"));

    let mut s = published();
    s.layers.clear();
    assert!(s.validate(25).unwrap_err().contains("must not be empty"));

    let s = published();
    assert!(s.validate(20).unwrap_err().contains("out of range"));

    let mut s = published();
    s.layers.push(12);
    assert!(s.validate(25).unwrap_err().contains("listed twice"));

    // Layers must be non-negative integers.
    let err = ReasoningEffortSteering::from_config_value(&json!({
        "reasoning_effort_steering": {"alpha": {"high": 0.05}, "default": "high", "layers": [-1]}
    }))
    .unwrap_err();
    assert!(err.contains("reasoning_effort_steering"), "{err}");
}

#[test]
fn a_missing_block_is_none() {
    assert_eq!(
        ReasoningEffortSteering::from_config_value(&json!({"hidden_size": 8})).unwrap(),
        None
    );
    assert_eq!(
        ReasoningEffortSteering::from_config_value(&json!({"reasoning_effort_steering": null}))
            .unwrap(),
        None
    );
}
