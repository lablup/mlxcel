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

//! Privacy Filter detection, calibration loading and `--operating-point`
//! handling on configs alone (no weights). The real-checkpoint gates are in
//! `privacy_filter_checkpoint_tests.rs`.

use serde_json::{Value, json};

use super::*;
use crate::token_classification::TokenModelKind;
use crate::token_classification::loader::{
    TokenClassifierLoadOptions, detect_token_model_kind, load_token_classifier,
    load_token_classifier_with,
};

fn checkpoint_config() -> Value {
    serde_json::from_str(include_str!(
        "../../tests/fixtures/token_classification/openai_privacy_filter_config.json"
    ))
    .expect("fixture parses")
}

/// Write `config` (and optionally a calibration file) into a fresh directory.
fn checkpoint_dir(config: &Value, calibration: Option<&Value>) -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::write(dir.path().join("config.json"), config.to_string()).expect("config written");
    if let Some(calibration) = calibration {
        std::fs::write(dir.path().join(CALIBRATION_FILE), calibration.to_string())
            .expect("calibration written");
    }
    dir
}

fn calibration(points: &[(&str, [f64; 6])]) -> Value {
    let mut operating_points = serde_json::Map::new();
    for (name, values) in points {
        let biases: serde_json::Map<String, Value> = super::super::viterbi::VITERBI_BIAS_KEYS
            .iter()
            .zip(values)
            .map(|(key, value)| ((*key).to_string(), json!(value)))
            .collect();
        operating_points.insert((*name).to_string(), json!({"biases": biases}));
    }
    json!({"operating_points": operating_points})
}

#[test]
fn checkpoint_config_detects_as_privacy_filter() {
    let config = checkpoint_config();
    assert_eq!(
        detect_token_model_kind(&config).unwrap(),
        TokenModelKind::PrivacyFilter
    );
    let labels = id2label_labels(&config).unwrap();
    assert_eq!(labels.len(), 33);
    assert_eq!(labels[0], "O");
    assert_eq!(labels[1], "B-account_number");
    assert_eq!(labels[32], "S-secret");
    assert_eq!(
        TokenModelKind::PrivacyFilter.as_str(),
        "openai_privacy_filter"
    );
    assert_eq!(
        TokenModelKind::PrivacyFilter.output_shape(),
        crate::token_classification::TokenOutputShape::TokenClassification
    );
    assert!(TokenModelKind::ALL.contains(&TokenModelKind::PrivacyFilter));
    // The Viterbi decoder accepts the checkpoint's full BIOES set.
    ViterbiDecoder::new(&labels, TransitionBiases::default()).expect("every entity has B/I/E/S");

    let mut without_labels = config.clone();
    without_labels.as_object_mut().unwrap().remove("id2label");
    assert!(detect_token_model_kind(&without_labels).is_err());
}

#[test]
fn calibration_file_is_optional_but_strict() {
    let config = checkpoint_config();
    // No file: all zeros.
    let bare = checkpoint_dir(&config, None);
    assert_eq!(
        load_transition_biases(bare.path(), DEFAULT_OPERATING_POINT).unwrap(),
        TransitionBiases::default()
    );

    let file = calibration(&[
        ("default", [0.0; 6]),
        ("strict", [0.5, -0.25, 1.0, 0.0, 2.5, -1.5]),
    ]);
    let dir = checkpoint_dir(&config, Some(&file));
    assert_eq!(
        load_transition_biases(dir.path(), "default").unwrap(),
        TransitionBiases::default()
    );
    assert_eq!(
        load_transition_biases(dir.path(), "strict").unwrap(),
        TransitionBiases {
            background_stay: 0.5,
            background_to_start: -0.25,
            inside_to_continue: 1.0,
            inside_to_end: 0.0,
            end_to_background: 2.5,
            end_to_start: -1.5,
        }
    );
    let error = load_transition_biases(dir.path(), "aggressive")
        .unwrap_err()
        .to_string();
    assert!(
        error.starts_with("Unknown Privacy Filter operating point 'aggressive'"),
        "{error}"
    );
    assert!(error.contains("'default', 'strict'"), "{error}");

    // A missing key fails naming the key.
    let mut broken = file.clone();
    broken["operating_points"]["default"]["biases"]
        .as_object_mut()
        .unwrap()
        .remove("transition_bias_end_to_start");
    let dir = checkpoint_dir(&config, Some(&broken));
    let error = format!(
        "{:#}",
        load_transition_biases(dir.path(), "default").unwrap_err()
    );
    assert!(
        error.contains("missing biases: [\"transition_bias_end_to_start\"]"),
        "{error}"
    );
}

/// Load-time failures happen before any weight is read: these directories
/// hold a `config.json` and nothing else.
#[test]
fn operating_point_problems_fail_before_weights() {
    let config = checkpoint_config();
    let dir = checkpoint_dir(&config, Some(&calibration(&[("default", [0.0; 6])])));
    let error = format!(
        "{:#}",
        load_token_classifier_with(
            dir.path(),
            &TokenClassifierLoadOptions {
                operating_point: Some("nope".into()),
            },
        )
        .err()
        .expect("unknown operating point")
    );
    assert!(
        error.contains("Unknown Privacy Filter operating point 'nope'"),
        "{error}"
    );
    assert!(!error.contains("safetensors"), "{error}");

    // The default point of a checkpoint without the file is fine; the load
    // then fails on the missing weights, which proves the order.
    let bare = checkpoint_dir(&config, None);
    let error = format!("{:#}", load_token_classifier(bare.path()).err().unwrap());
    assert!(
        error.contains("failed to load openai_privacy_filter"),
        "{error}"
    );

    // Rampart (argmax only) refuses the flag by kind, before its weights.
    let rampart: Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/token_classification/rampart_config.json"
    ))
    .unwrap();
    let dir = checkpoint_dir(&rampart, None);
    let error = format!(
        "{:#}",
        load_token_classifier_with(
            dir.path(),
            &TokenClassifierLoadOptions {
                operating_point: Some("default".into()),
            },
        )
        .err()
        .expect("argmax-only family refuses the flag")
    );
    assert!(
        error.contains("--operating-point 'default' applies only to Viterbi-decoded"),
        "{error}"
    );
    assert!(error.contains("bert_token_classifier"), "{error}");
    assert!(!error.contains("safetensors"), "{error}");
}

#[test]
fn a_privacy_filter_passed_to_extract_names_mlxcel_tag() {
    let dir = checkpoint_dir(&checkpoint_config(), None);
    let error = crate::token_classification::load_span_extractor(dir.path())
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("use `mlxcel tag`"), "{error}");
    assert!(error.contains("openai_privacy_filter"), "{error}");
}
