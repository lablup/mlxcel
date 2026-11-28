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

//! `mlxcel decide` input preparation, which runs before any model loads.

use serde_json::json;

use super::*;

fn with_questions(questions: &str) -> DecideArgs {
    DecideArgs {
        state: Some("hello".into()),
        questions: Some(questions.into()),
        ..DecideArgs::default()
    }
}

fn error(args: &DecideArgs) -> String {
    format!("{:#}", prepare_decide(args).expect_err("must fail"))
}

#[test]
fn malformed_questions_are_rejected() {
    assert!(error(&with_questions("{oops")).starts_with("questions are not valid JSON"));
    assert_eq!(
        error(&with_questions("{}")),
        "At least one named question is required"
    );
    assert_eq!(
        error(&with_questions("[1, 2]")),
        "At least one named question is required"
    );
    assert_eq!(
        error(&with_questions(r#"{"q": ["a", "b"]}"#)),
        "Questions must map names to question dictionaries"
    );
}

#[test]
fn neither_state_nor_media_is_rejected() {
    let args = DecideArgs {
        questions: Some(r#"{"q": {}}"#.into()),
        ..DecideArgs::default()
    };
    assert_eq!(
        error(&args),
        "one of the arguments --state --state-file --image --video --audio is required"
    );
    let with_image = DecideArgs {
        images: vec!["a.png".into()],
        ..args
    };
    let prepared = prepare_decide(&with_image).expect("media stands in for state");
    assert_eq!(prepared.state, None);
}

#[test]
fn sources_are_exclusive_and_questions_required() {
    let both_states = DecideArgs {
        state_file: Some("state.txt".into()),
        ..with_questions(r#"{"q": {}}"#)
    };
    assert_eq!(
        error(&both_states),
        "--state and --state-file are mutually exclusive"
    );
    let no_questions = DecideArgs {
        state: Some("x".into()),
        ..DecideArgs::default()
    };
    assert_eq!(
        error(&no_questions),
        "one of the arguments --questions --questions-file is required"
    );
}

#[test]
fn files_are_read_for_state_and_questions() {
    let dir = tempfile::tempdir().expect("tempdir");
    let state = dir.path().join("state.txt");
    let questions = dir.path().join("questions.json");
    std::fs::write(&state, "Ünïcode state").expect("write");
    std::fs::write(&questions, r#"{"q": {"type": "bool"}}"#).expect("write");
    let prepared = prepare_decide(&DecideArgs {
        state_file: Some(state),
        questions_file: Some(questions),
        ..DecideArgs::default()
    })
    .expect("files read");
    assert_eq!(prepared.state, Some(json!("Ünïcode state")));
    assert_eq!(prepared.questions, json!({"q": {"type": "bool"}}));
}
