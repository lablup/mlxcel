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

//! One test per validation rule (1 to 8), each asserting upstream's exact
//! message, plus answer shapes, the loader and the stub through [`predict`].

use serde_json::{Value, json};

use super::stub::{STUB_MODEL_TYPE, StubDecisionModel};
use super::validate::python_repr;
use super::*;

const ALL_TYPES: &[QuestionKind] = &QuestionKind::ALL;
const ALL_MEDIA: &[MediaKind] = &[MediaKind::Image, MediaKind::Video, MediaKind::Audio];

fn state() -> Value {
    json!("the billing page charged me twice")
}

fn check(questions: Value) -> Result<Vec<(String, Question)>, DecisionError> {
    validate_questions(
        ALL_TYPES,
        ALL_MEDIA,
        Some(&state()),
        &questions,
        MediaPresence::default(),
    )
}

fn message(result: Result<Vec<(String, Question)>, DecisionError>) -> String {
    match result {
        Err(DecisionError::InvalidInput(message)) => message,
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

fn choice() -> Value {
    json!({"topic": {"type": "choice", "criteria": ["billing", "technical"]}})
}

// Rule 1.
#[test]
fn rule1_state_is_required_without_media() {
    let none = validate_questions(
        ALL_TYPES,
        ALL_MEDIA,
        None,
        &choice(),
        MediaPresence::default(),
    );
    assert_eq!(
        message(none),
        "state is required unless images or audio are given"
    );
    let null = validate_questions(
        ALL_TYPES,
        ALL_MEDIA,
        Some(&Value::Null),
        &choice(),
        MediaPresence::default(),
    );
    assert_eq!(
        message(null),
        "state is required unless images or audio are given"
    );
    for media in [
        MediaPresence {
            images: true,
            ..MediaPresence::default()
        },
        MediaPresence {
            videos: true,
            ..MediaPresence::default()
        },
        MediaPresence {
            audio: true,
            ..MediaPresence::default()
        },
    ] {
        validate_questions(ALL_TYPES, ALL_MEDIA, None, &choice(), media)
            .expect("media stands in for state");
    }
}

// Rule 2.
#[test]
fn rule2_a_model_without_decision_types_is_refused() {
    let result = validate_questions(
        &[],
        ALL_MEDIA,
        Some(&state()),
        &choice(),
        MediaPresence::default(),
    );
    assert_eq!(
        message(result),
        "This model does not support decision prediction"
    );
}

// Rule 3.
#[test]
fn rule3_undeclared_media_is_refused_by_kind() {
    let cases = [
        (
            MediaPresence {
                images: true,
                ..MediaPresence::default()
            },
            "This model does not support image input",
        ),
        (
            MediaPresence {
                videos: true,
                ..MediaPresence::default()
            },
            "This model does not support video input",
        ),
        (
            MediaPresence {
                audio: true,
                ..MediaPresence::default()
            },
            "This model does not support audio input",
        ),
    ];
    for (media, expected) in cases {
        let result = validate_questions(ALL_TYPES, &[], Some(&state()), &choice(), media);
        assert_eq!(message(result), expected);
    }
    // A declared medium passes.
    validate_questions(
        ALL_TYPES,
        &[MediaKind::Image],
        Some(&state()),
        &choice(),
        MediaPresence {
            images: true,
            ..MediaPresence::default()
        },
    )
    .expect("declared images pass");
}

// Rule 4.
#[test]
fn rule4_questions_must_be_a_nonempty_map_of_objects() {
    for empty in [json!({}), json!([]), Value::Null, json!("topic")] {
        assert_eq!(
            message(check(empty)),
            "At least one named question is required"
        );
    }
    assert_eq!(
        message(check(json!({"topic": "billing"}))),
        "Questions must map names to question dictionaries"
    );
}

// Rule 5.
#[test]
fn rule5_type_defaults_to_choice_and_must_be_supported() {
    let parsed = check(json!({"topic": {"criteria": ["a", "b"]}})).expect("default type");
    assert_eq!(parsed[0].1.kind, QuestionKind::Choice);

    assert_eq!(
        message(check(
            json!({"q": {"type": "ranking", "criteria": ["a", "b"]}})
        )),
        "This model does not support 'ranking' decisions"
    );
    let only_choice = validate_questions(
        &[QuestionKind::Choice],
        ALL_MEDIA,
        Some(&state()),
        &json!({"q": {"type": "noul"}}),
        MediaPresence::default(),
    );
    assert_eq!(
        message(only_choice),
        "This model does not support 'noul' decisions"
    );
    let no_choice = validate_questions(
        &[QuestionKind::Bool],
        ALL_MEDIA,
        Some(&state()),
        &json!({"q": {"criteria": ["a", "b"]}}),
        MediaPresence::default(),
    );
    assert_eq!(
        message(no_choice),
        "This model does not support 'choice' decisions"
    );
    // `noul` folds into `bool`.
    let noul = check(json!({"q": {"type": "noul"}})).expect("noul is bool");
    assert_eq!(noul[0].1.kind, QuestionKind::Bool);
}

// Rule 6.
#[test]
fn rule6_criteria_counts_labels_and_score_order() {
    assert_eq!(
        message(check(
            json!({"q": {"type": "choice", "criteria": ["only"]}})
        )),
        "choice needs at least 2 criteria"
    );
    assert_eq!(
        message(check(json!({"q": {"type": "score", "criteria": "low"}}))),
        "score needs at least 2 criteria"
    );
    assert_eq!(
        message(check(json!({"q": {"type": "multi_label", "criteria": []}}))),
        "multi_label needs at least 1 criteria"
    );
    assert_eq!(
        message(check(
            json!({"q": {"type": "choice", "criteria": ["a", ""]}})
        )),
        "Decision labels must be nonempty strings"
    );
    assert_eq!(
        message(check(
            json!({"q": {"type": "choice", "criteria": ["a", 3]}})
        )),
        "Decision labels must be nonempty strings"
    );
    assert_eq!(
        message(check(
            json!({"q": {"type": "choice", "criteria": {"": "x", "b": null}}})
        )),
        "Decision labels must be nonempty strings"
    );
    assert_eq!(
        message(check(
            json!({"q": {"type": "multi_label", "criteria": ["a", "a"]}})
        )),
        "Decision labels must be unique"
    );
    assert_eq!(
        message(check(
            json!({"q": {"type": "score", "criteria": {"low": 0, "high": 1}}})
        )),
        "score criteria must be an ordered list"
    );
    // A map means label -> description; the description may be null.
    let parsed = check(json!({"q": {"criteria": {"billing": "money", "sales": null}}}))
        .expect("map criteria");
    assert_eq!(
        parsed[0].1.labels(),
        [
            Label {
                name: "billing".into(),
                description: Some("money".into())
            },
            Label {
                name: "sales".into(),
                description: None
            },
        ]
    );
}

// Rule 7.
#[test]
fn rule7_threshold_is_a_number_in_the_unit_interval() {
    let parsed = check(json!({"q": {"type": "multi_label", "criteria": ["a"]}})).expect("default");
    assert_eq!(parsed[0].1.threshold, 0.5);
    let parsed = check(json!({"q": {"type": "multi_label", "criteria": ["a"], "threshold": 1}}))
        .expect("integer one");
    assert_eq!(parsed[0].1.threshold, 1.0);
    for bad in [
        json!(true),
        json!(1.5),
        json!(-0.1),
        json!("0.5"),
        Value::Null,
    ] {
        assert_eq!(
            message(check(
                json!({"q": {"type": "multi_label", "criteria": ["a"], "threshold": bad}})
            )),
            "threshold must be a number between zero and one"
        );
    }
}

// Rule 8.
#[test]
fn rule8_bool_criteria_map_false_and_true() {
    assert_eq!(
        message(check(
            json!({"q": {"type": "bool", "criteria": ["no", "yes"]}})
        )),
        "bool criteria must map false and true to descriptions"
    );
    assert_eq!(
        message(check(json!({"q": {"type": "noul", "criteria": "yes"}}))),
        "bool criteria must map false and true to descriptions"
    );
    let parsed = check(json!({"q": {"type": "bool", "criteria": {"true": "refund it"}}}))
        .expect("map criteria");
    assert_eq!(
        parsed[0].1.criteria,
        Criteria::Bool {
            when_false: None,
            when_true: Some("refund it".into())
        }
    );
    check(json!({"q": {"type": "bool"}})).expect("criteria may be absent");
}

#[test]
fn rules_run_in_contract_order() {
    // No state and an undeclared medium together: rule 1 is not hit because
    // media is present, so rule 3 answers.
    let result = validate_questions(
        ALL_TYPES,
        &[],
        None,
        &json!({}),
        MediaPresence {
            audio: true,
            ..MediaPresence::default()
        },
    );
    assert_eq!(message(result), "This model does not support audio input");
    // Empty types win over an invalid question map.
    let result = validate_questions(
        &[],
        ALL_MEDIA,
        Some(&state()),
        &json!({}),
        MediaPresence::default(),
    );
    assert_eq!(
        message(result),
        "This model does not support decision prediction"
    );
}

#[test]
fn python_repr_matches_upstream_formatting() {
    assert_eq!(python_repr(&json!("ranking")), "'ranking'");
    assert_eq!(python_repr(&json!("it's")), "\"it's\"");
    assert_eq!(python_repr(&json!(5)), "5");
    assert_eq!(python_repr(&Value::Null), "None");
    assert_eq!(python_repr(&json!(true)), "True");
}

#[test]
fn stub_answers_keep_label_order_and_fold_noul_into_bool() {
    let mut model = StubDecisionModel;
    let questions = json!({
        "topic": {"type": "choice", "criteria": ["technical", "billing", "sales"]},
        "urgent": {"type": "noul"},
        "severity": {"type": "score", "criteria": ["low", "medium", "high"]},
    });
    let result = predict(
        &mut model,
        Some(&state()),
        &questions,
        DecisionMedia::default(),
    )
    .expect("stub predicts");
    let body = result.to_response_json("stub");
    let topic = &body["answers"]["topic"];
    assert_eq!(topic["type"], "choice");
    assert_eq!(topic["value"], "billing");
    let order: Vec<&String> = topic["probabilities"].as_object().unwrap().keys().collect();
    assert_eq!(order, ["technical", "billing", "sales"]);
    assert_eq!(topic["metadata"]["confidence"], 0.6);
    assert_eq!(body["answers"]["urgent"]["type"], "bool");
    assert_eq!(body["answers"]["urgent"]["value"], false);
    let severity = &body["answers"]["severity"];
    assert!((severity["value"].as_f64().unwrap() - 1.0).abs() < 1e-9);
    assert_eq!(severity["metadata"]["legend"]["2"], "high");
    assert_eq!(body["usage"]["output_tokens"], 0);
    assert_eq!(body["usage"]["input_tokens"], 6);
    let names: Vec<&String> = body["answers"].as_object().unwrap().keys().collect();
    assert_eq!(names, ["topic", "urgent", "severity"]);
}

#[test]
fn multi_label_value_uses_the_threshold() {
    let parsed =
        check(json!({"q": {"type": "multi_label", "criteria": ["a", "b", "c"], "threshold": 0.4}}))
            .expect("valid");
    let answer = Answer::multi_label(&parsed[0].1, vec![0.9, 0.1, 0.4], Default::default())
        .expect("lengths match");
    let json = answer.to_json();
    assert_eq!(json["value"], json!(["a", "c"]));
    assert_eq!(json["scores"]["b"], 0.1);
    assert!(json.get("metadata").is_none());
}

#[test]
fn a_model_answering_the_wrong_questions_is_internal() {
    let parsed = check(choice()).expect("valid");
    let result = DecisionResult {
        answers: vec![("other".into(), Answer::bool(0.5, Default::default()))],
        usage: DecisionUsage::default(),
    };
    assert!(matches!(
        check_result(&parsed, &result),
        Err(DecisionError::Internal(_))
    ));
    let result = DecisionResult {
        answers: vec![("topic".into(), Answer::bool(f64::NAN, Default::default()))],
        usage: DecisionUsage::default(),
    };
    assert!(matches!(
        check_result(&parsed, &result),
        Err(DecisionError::Internal(_))
    ));
}

#[test]
fn loader_detects_the_stub_and_refuses_other_checkpoints() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("config.json"),
        json!({"model_type": STUB_MODEL_TYPE}).to_string(),
    )
    .expect("write config");
    let loaded = load_decision_model(dir.path()).expect("stub loads");
    assert_eq!(loaded.kind, DecisionKind::Stub);
    assert_eq!(loaded.model.decision_media(), [MediaKind::Image]);

    let other = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        other.path().join("config.json"),
        json!({"model_type": "qwen3"}).to_string(),
    )
    .expect("write config");
    let err = format!(
        "{:#}",
        load_decision_model(other.path()).expect_err("not a decision model")
    );
    assert!(
        err.starts_with(&format!(
            "{} is not a decision checkpoint",
            other.path().display()
        )),
        "{err}"
    );
    let missing = format!(
        "{:#}",
        load_decision_model(&other.path().join("absent")).expect_err("no config")
    );
    assert!(
        missing.contains("is not a decision checkpoint"),
        "{missing}"
    );
}
