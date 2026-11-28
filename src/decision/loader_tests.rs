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

//! Family detection from `config.json` alone.

use serde_json::{Value, json};

use super::{DecisionKind, detect_decision_kind};
use crate::decision::decider2::DecisionSettings;

/// The published `nativ-community/decider-2b` `config.json`.
fn decider2_config() -> Value {
    serde_json::from_str(include_str!(
        "../../tests/fixtures/decision/decider2_config.json"
    ))
    .expect("fixture parses")
}

#[test]
fn published_decider2_config_is_detected() {
    let config = decider2_config();
    assert_eq!(
        detect_decision_kind(&config).expect("decider2"),
        DecisionKind::Decider2
    );
    assert_eq!(DecisionKind::Decider2.as_str(), "decider2");
    let settings = DecisionSettings::from_config(&config).expect("decision_config");
    assert_eq!(settings.temperature, 1.145);
    assert_eq!(settings.temperature_by_type["choice"], 1.164);
    assert_eq!(settings.temperature_by_type["noul"], 1.624);
    assert_eq!(settings.temperature_by_type["score"], 1.124);
    assert!(settings.isolated_levels);
    assert_eq!(settings.max_state_tokens, 32768);
}

#[test]
fn plain_qwen3_5_text_config_is_not_a_decision_model() {
    let mut config = decider2_config();
    config["model_type"] = json!("qwen3_5_text");
    let error = detect_decision_kind(&config).expect_err("plain Qwen3.5");
    assert!(error.to_string().contains("qwen3_5_text"), "{error}");
}

#[test]
fn decision_config_defaults_and_absence() {
    let settings = DecisionSettings::from_config(&json!({"decision_config": {"temperature": 1.0}}))
        .expect("minimal");
    assert!(!settings.isolated_levels);
    assert_eq!(settings.max_state_tokens, 32768);
    assert!(settings.temperature_by_type.is_empty());
    assert!(DecisionSettings::from_config(&json!({"model_type": "decider2"})).is_err());
}
