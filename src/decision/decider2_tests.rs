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

//! Decider-2b rendering, prompt ids, rows and answers, without weights.
//!
//! The prompt-id tests read the checkpoint's `tokenizer.json` and skip when
//! `models/decider-2b` is absent; everything else runs anywhere. Expected ids
//! come from `tests/fixtures/decision/decider2_parity.json`, generated with
//! upstream's `Decider2._prompt` by `tests/fixtures/generate_decider2_fixture.py`.

use std::collections::HashMap;
use std::path::PathBuf;

use serde_json::{Value, json};

use super::decider2::{DecisionSettings, build_prompts};
use super::decider2_prompt::{
    Decider2Prompter, python_float, python_json, render_question, render_state,
};
use super::decider2_rows::{
    Row, Slot, Temperatures, answer, group_rows, input_tokens, padded_length, plan_rows,
};
use super::{Answer, DecisionError, MediaPresence, Question, QuestionKind, validate_questions};

const TYPES: &[QuestionKind] = &[
    QuestionKind::Choice,
    QuestionKind::Score,
    QuestionKind::Bool,
];

pub(super) fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../tests/fixtures/decision/decider2_parity.json"
    ))
    .expect("fixture parses")
}

/// `models/decider-2b` when it holds a complete checkpoint; `None` skips.
pub(super) fn checkpoint(need_weights: bool) -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/decider-2b");
    let ready = dir.join("tokenizer.json").is_file()
        && dir.join("config.json").is_file()
        && (!need_weights || dir.join("model.safetensors").is_file());
    if !ready {
        eprintln!(
            "skipping: models/decider-2b not present (mlxcel download nativ-community/decider-2b)"
        );
    }
    ready.then_some(dir)
}

fn question(spec: Value) -> Question {
    let questions = json!({ "q": spec });
    let mut validated = validate_questions(
        TYPES,
        &[],
        Some(&json!("state")),
        &questions,
        MediaPresence::default(),
    )
    .expect("valid question");
    validated.remove(0).1
}

fn invalid(spec: Value) -> String {
    match render_question(&question(spec)) {
        Err(DecisionError::InvalidInput(message)) => message,
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn python_float_matches_repr() {
    for (value, repr) in [
        (1e-05, "1e-05"),
        (4.5, "4.5"),
        (123.0, "123.0"),
        (0.1, "0.1"),
        (0.0001, "0.0001"),
        (-2.5e-7, "-2.5e-07"),
        (1e16, "1e+16"),
        (1.2345678901234568e17, "1.2345678901234568e+17"),
        (999999999999999.9, "999999999999999.9"),
        (1e300, "1e+300"),
    ] {
        assert_eq!(python_float(value), repr, "{value}");
    }
}

#[test]
fn state_rendering_matches_upstream_contexts() {
    for case in fixture()["prompts"].as_array().expect("prompts") {
        assert_eq!(
            render_state(Some(&case["state"])),
            case["context"].as_str().expect("context"),
            "{}",
            case["name"]
        );
    }
    // Seven items stay unannotated; separators and non-ASCII follow Python.
    let short = json!([1, 2, 3, 4, 5, 6, {"k": "é", "n": null}]);
    assert_eq!(
        render_state(Some(&short)),
        r#"[1, 2, 3, 4, 5, 6, {"k": "é", "n": null}]"#
    );
    assert_eq!(render_state(Some(&json!("as is"))), "as is");
    assert_eq!(
        python_json(&json!({"a": "line\nbreak\u{1}"})),
        r#"{"a": "line\nbreak\u0001"}"#
    );
}

#[test]
fn choice_options_render_descriptions_like_upstream() {
    let rendered =
        render_question(&question(fixture()["prompts"][1]["question"].clone())).expect("renders");
    assert_eq!(rendered.kind, "choice");
    assert_eq!(
        rendered.options[0],
        "accounts: Sign-in, passwords and profiles"
    );
    assert_eq!(rendered.options[1], "billing");
    assert_eq!(rendered.options[2], "cards");
    assert_eq!(
        rendered.options[3],
        r#"deposits: {"limit": 5000, "currency": "EUR"}"#
    );
    assert_eq!(rendered.options[4], r#"exports: ["csv", "pdf"]"#);
    assert_eq!(rendered.options[6], "loans: 3");
    assert_eq!(rendered.options[8], "notifications: true");
}

#[test]
fn score_and_bool_render_like_upstream() {
    let score = render_question(&question(json!({
        "type": "score", "instructions": "Rate it", "criteria": ["low", {"x": 1.0}, "high"]
    })))
    .expect("renders");
    assert_eq!(score.kind, "score");
    assert_eq!(score.options, ["0: low", r#"1: {"x": 1.0}"#, "2: high"]);
    assert_eq!(score.legend, ["low", r#"{"x": 1.0}"#, "high"]);

    let plain = render_question(&question(json!({"type": "noul", "instructions": "Ok?"})))
        .expect("renders");
    assert_eq!(plain.kind, "noul");
    assert_eq!(plain.options, ["no", "yes"]);
    let described = render_question(&question(json!({
        "type": "bool", "instructions": "Ok?", "criteria": {"false": "", "true": 2}
    })))
    .expect("renders");
    assert_eq!(described.options, ["no", "yes: 2"]);

    let encoded = render_question(&question(
        json!({"instructions": ["a", 1], "criteria": ["x", "y"]}),
    ))
    .expect("renders");
    assert_eq!(encoded.instruction, r#"["a", 1]"#);
    let fallback = render_question(&question(
        json!({"question": "Which?", "criteria": ["x", "y"]}),
    ))
    .expect("renders");
    assert_eq!(fallback.instruction, "Which?");
}

#[test]
fn model_limits_are_invalid_input() {
    let many: Vec<String> = (0..256).map(|i| format!("o{i}")).collect();
    assert_eq!(
        invalid(json!({"instructions": "Pick", "criteria": many})),
        "choice needs 2 to 255 options"
    );
    let most: Vec<String> = (0..255).map(|i| format!("o{i}")).collect();
    assert!(render_question(&question(json!({"instructions": "Pick", "criteria": most}))).is_ok());
    let levels: Vec<String> = (0..11).map(|i| format!("level {i}")).collect();
    assert_eq!(
        invalid(json!({"type": "score", "instructions": "Rate", "criteria": levels})),
        "score needs 2 to 10 levels"
    );
    assert_eq!(
        invalid(json!({"instructions": "", "criteria": ["a", "b"]})),
        "Question needs instructions"
    );
    assert_eq!(
        invalid(json!({"criteria": ["a", "b"]})),
        "Question needs instructions"
    );
}

fn temperatures() -> Temperatures {
    Temperatures {
        default: 1.145,
        by_type: HashMap::from([
            ("choice".to_string(), 1.164),
            ("noul".to_string(), 1.624),
            ("score".to_string(), 1.124),
        ]),
    }
}

#[test]
fn isolated_scores_become_one_row_per_level() {
    let rendered = [
        render_question(&question(
            json!({"instructions": "Team?", "criteria": ["a", "b"]}),
        ))
        .expect("choice"),
        render_question(&question(json!({
            "type": "score", "instructions": "Urgency?", "criteria": ["0: calm", "busy", "on fire"]
        })))
        .expect("score"),
        render_question(&question(
            json!({"type": "bool", "instructions": "Refund?"}),
        ))
        .expect("bool"),
    ];
    let (rows, slots) = plan_rows(&rendered, &temperatures(), true);
    assert_eq!(rows.len(), 5);
    assert_eq!(
        slots,
        [
            Slot {
                start: 0,
                count: 1,
                isolated: false
            },
            Slot {
                start: 1,
                count: 3,
                isolated: true
            },
            Slot {
                start: 4,
                count: 1,
                isolated: false
            },
        ]
    );
    assert_eq!(
        rows[1].instruction,
        "Urgency?\nProposed answer: calm\nDoes the proposed answer fit?"
    );
    assert_eq!(rows[2].options, ["no", "yes"]);
    let temps: Vec<f32> = rows.iter().map(|row| row.temperature).collect();
    assert_eq!(temps, [1.164, 1.124, 1.124, 1.124, 1.624]);

    let (packed, _) = plan_rows(&rendered, &temperatures(), false);
    assert_eq!(packed.len(), 3);
    assert_eq!(packed[1].options, ["0: 0: calm", "1: busy", "2: on fire"]);
}

#[test]
fn groups_respect_the_token_budget() {
    assert_eq!(group_rows(&[100, 100, 100]), vec![0..3]);
    assert_eq!(group_rows(&[4000, 4000, 10]), [0..2, 2..3]);
    assert_eq!(group_rows(&[9000, 10]), [0..1, 1..2]);
    assert_eq!(group_rows(&[10, 3000, 3000]), [0..2, 2..3]);
    assert_eq!(padded_length(1), 64);
    assert_eq!(padded_length(64), 64);
    assert_eq!(padded_length(65), 128);
    assert_eq!(
        input_tokens(&[vec![1, 2, 3, 4], vec![1, 2, 5], vec![1, 2]]),
        2 + 2 + 1
    );
    assert_eq!(input_tokens(&[vec![7, 8, 9]]), 3);
}

#[test]
fn answers_round_and_carry_upstream_metadata() {
    let choice = question(json!({"instructions": "Team?", "criteria": ["a", "b", "c"]}));
    let score =
        question(json!({"type": "score", "instructions": "Rate", "criteria": ["lo", "hi"]}));
    let boolean = question(json!({"type": "bool", "instructions": "Ok?"}));
    let probabilities = vec![
        vec![0.123_456, 0.654_321, 0.222_223],
        vec![0.8, 0.2],
        vec![0.4, 0.6],
        vec![0.49996, 0.50004],
    ];
    let choice_answer = answer(
        &choice,
        &render_question(&choice).expect("choice"),
        Slot {
            start: 0,
            count: 1,
            isolated: false,
        },
        &probabilities,
    );
    let json = choice_answer.to_json();
    assert_eq!(json["value"], "b");
    assert_eq!(
        json["probabilities"],
        json!({"a": 0.1235, "b": 0.6543, "c": 0.2222})
    );
    let keys: Vec<&str> = json["metadata"]
        .as_object()
        .expect("metadata")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(keys, ["confidence", "certainty"]);

    let score_answer = answer(
        &score,
        &render_question(&score).expect("score"),
        Slot {
            start: 1,
            count: 2,
            isolated: true,
        },
        &probabilities,
    );
    let json = score_answer.to_json();
    // Fits 0.2 and 0.6 over a mass of 0.8: levels 0.25 / 0.75.
    assert_eq!(json["value"], 0.75);
    assert_eq!(json["probabilities"], json!({"0": 0.25, "1": 0.75}));
    let metadata = &json["metadata"];
    assert_eq!(metadata["legend"], json!({"0": "lo", "1": "hi"}));
    assert_eq!(metadata["level_fit"], json!({"0": 0.2, "1": 0.6}));
    assert_eq!(metadata["fit_mass"], 0.8);
    assert_eq!(metadata["confidence"], 0.75);
    let keys: Vec<&str> = metadata
        .as_object()
        .expect("metadata")
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        keys,
        ["confidence", "certainty", "legend", "level_fit", "fit_mass"]
    );

    let bool_answer = answer(
        &boolean,
        &render_question(&boolean).expect("bool"),
        Slot {
            start: 3,
            count: 1,
            isolated: false,
        },
        &probabilities,
    );
    assert_eq!(bool_answer, Answer::bool(0.5, serde_json::Map::new()));
    assert_eq!(
        bool_answer.to_json(),
        json!({"type": "bool", "value": true, "probability": 0.5})
    );
}

/// The rendered prompt ids equal upstream's for the 3-option and 12-option
/// choices, the score options, an isolated score row and the annotated
/// 8-item list state.
#[test]
fn prompt_ids_match_upstream() {
    let Some(dir) = checkpoint(false) else {
        return;
    };
    let prompter = Decider2Prompter::load(&dir, 32768).expect("tokenizer");
    let fixture = fixture();
    let expected_labels: Vec<u32> =
        serde_json::from_value(fixture["label_token_ids"].clone()).expect("label ids");
    assert_eq!(prompter.label_ids, expected_labels);
    assert_eq!(prompter.labels[25], "Z");
    assert_eq!(prompter.labels[26], "AA");
    for case in fixture["prompts"].as_array().expect("prompts") {
        let rendered = render_question(&question(case["question"].clone())).expect("renders");
        let context = render_state(Some(&case["state"]));
        let (rows, _) = plan_rows(&[rendered], &temperatures(), case["level"].is_number());
        let row = case["level"].as_u64().map_or(0, |level| level as usize);
        let context_ids = prompter.context_ids(&context).expect("context");
        let ids = prompter
            .prompt(&context_ids, &rows[row].instruction, &rows[row].options)
            .expect("prompt");
        let expected: Vec<u32> = serde_json::from_value(case["ids"].clone()).expect("ids");
        assert_eq!(ids, expected, "{}", case["name"]);
    }
}

#[test]
fn long_state_is_truncated_not_refused() {
    let Some(dir) = checkpoint(false) else {
        return;
    };
    let prompter = Decider2Prompter::load(&dir, 16).expect("tokenizer");
    let state = "word ".repeat(500);
    let options = ["a".to_string(), "b".to_string()];
    let context_ids = prompter.context_ids(&state).expect("context");
    let ids = prompter
        .prompt(&context_ids, "Pick", &options)
        .expect("prompt");
    let tail = prompter
        .encode("\n\nQuestion: Pick\nOptions:\n(A) a\n(B) b\nAnswer: (")
        .expect("tail");
    assert_eq!(ids.len(), 16 + tail.len());
    assert_eq!(&ids[16..], tail.as_slice());
}

fn two_option_row(instruction: &str) -> Row {
    Row {
        instruction: instruction.to_string(),
        options: vec!["a".to_string(), "b".to_string()],
        temperature: 1.0,
    }
}

/// The shared state is encoded once and every row matches a one-row prompt.
#[test]
fn build_prompts_shares_the_state_encoding() {
    let Some(dir) = checkpoint(false) else {
        return;
    };
    let prompter = Decider2Prompter::load(&dir, 32768).expect("tokenizer");
    let rows = [two_option_row("First?"), two_option_row("Second?")];
    let prompts = build_prompts(&prompter, "some state", &rows, 1 << 16, 1 << 16).expect("fits");
    let context = prompter.context_ids("some state").expect("context");
    for (row, ids) in rows.iter().zip(&prompts) {
        let alone = prompter
            .prompt(&context, &row.instruction, &row.options)
            .expect("prompt");
        assert_eq!(ids, &alone);
    }
}

/// Many questions over one state, or one oversized question, are refused as
/// invalid input instead of being queued for prefill.
#[test]
fn build_prompts_bounds_request_and_row_tokens() {
    let Some(dir) = checkpoint(false) else {
        return;
    };
    let prompter = Decider2Prompter::load(&dir, 32768).expect("tokenizer");
    let state = "word ".repeat(200);
    let one = build_prompts(&prompter, &state, &[two_option_row("Q?")], 1 << 16, 1 << 16)
        .expect("one row fits");
    let per_row = one[0].len();
    let rows: Vec<Row> = (0..10).map(|_| two_option_row("Q?")).collect();
    assert_eq!(
        build_prompts(&prompter, &state, &rows, 1 << 16, per_row * 10)
            .expect("exactly at the budget")
            .len(),
        10
    );
    match build_prompts(&prompter, &state, &rows, 1 << 16, per_row * 10 - 1) {
        Err(DecisionError::InvalidInput(message)) => {
            assert!(message.contains("prompt tokens in total"), "{message}");
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
    let long = two_option_row(&"why ".repeat(5000));
    match build_prompts(&prompter, "s", &[long], 1024, 1 << 20) {
        Err(DecisionError::InvalidInput(message)) => {
            assert!(message.contains("at most 1024"), "{message}");
        }
        other => panic!("expected InvalidInput, got {other:?}"),
    }
}

#[test]
fn decision_config_rejects_unusable_temperatures() {
    let config = |temperature: Value, by_type: Value| json!({"decision_config": {"temperature": temperature, "temperature_by_type": by_type}});
    assert!(DecisionSettings::from_config(&config(json!(1.1), json!({"choice": 1.2}))).is_ok());
    assert!(DecisionSettings::from_config(&config(json!(0.0), json!({}))).is_err());
    assert!(DecisionSettings::from_config(&config(json!(-1.0), json!({}))).is_err());
    assert!(DecisionSettings::from_config(&config(json!(1.0), json!({"score": 0.0}))).is_err());
}
