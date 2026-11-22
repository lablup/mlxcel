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

//! `mlxcel decide` argument parsing and the checks that run before the model
//! is resolved. The model path below does not exist, so any error that names
//! the questions or the state proves the check ran before loading.

use clap::Parser;

use super::*;
use crate::{Cli, Commands};

const MISSING_MODEL: &str = "/nonexistent/decision-model-for-tests";

fn parse(args: &[&str]) -> Result<DecideArgs, clap::Error> {
    let mut argv = vec!["mlxcel", "decide", "-m", MISSING_MODEL];
    argv.extend_from_slice(args);
    match Cli::try_parse_from(argv)?.command {
        Commands::Decide(args) => Ok(args),
        other => panic!("parsed to another command: {other:?}"),
    }
}

fn run(args: &[&str]) -> String {
    let parsed = parse(args).expect("arguments parse");
    format!("{:#}", run_decide(parsed).expect_err("decide must fail"))
}

#[test]
fn malformed_questions_json_fails_before_the_model_loads() {
    let message = run(&["--state", "hi", "--questions", "{not json"]);
    assert!(
        message.contains("questions are not valid JSON"),
        "{message}"
    );
}

#[test]
fn empty_questions_fail_before_the_model_loads() {
    let message = run(&["--state", "hi", "--questions", "{}"]);
    assert_eq!(message, "At least one named question is required");
}

#[test]
fn non_object_question_fails_before_the_model_loads() {
    let message = run(&["--state", "hi", "--questions", r#"{"topic": "billing"}"#]);
    assert_eq!(message, "Questions must map names to question dictionaries");
}

#[test]
fn neither_state_nor_media_is_refused_before_the_model_loads() {
    let message = run(&["--questions", r#"{"topic": {"criteria": ["a", "b"]}}"#]);
    assert_eq!(
        message,
        "one of the arguments --state --state-file --image --video --audio is required"
    );
}

#[test]
fn state_and_state_file_are_mutually_exclusive() {
    let err = parse(&[
        "--state",
        "hi",
        "--state-file",
        "state.txt",
        "--questions",
        "{}",
    ])
    .expect_err("conflict");
    assert_eq!(err.kind(), clap::error::ErrorKind::ArgumentConflict);
}

#[test]
fn questions_and_questions_file_are_mutually_exclusive_and_required() {
    let err = parse(&[
        "--state",
        "hi",
        "--questions",
        "{}",
        "--questions-file",
        "q.json",
    ])
    .expect_err("conflict");
    assert_eq!(err.kind(), clap::error::ErrorKind::ArgumentConflict);
    let err = parse(&["--state", "hi"]).expect_err("missing questions");
    assert_eq!(err.kind(), clap::error::ErrorKind::MissingRequiredArgument);
}

#[test]
fn media_flags_repeat() {
    let args = parse(&[
        "--image",
        "a.png",
        "--image",
        "https://example.com/b.png",
        "--video",
        "c.mp4",
        "--audio",
        "d.wav",
        "--questions",
        "{}",
    ])
    .expect("parses");
    assert_eq!(args.images, ["a.png", "https://example.com/b.png"]);
    assert_eq!(args.videos.len(), 1);
    assert_eq!(args.audio.as_deref(), Some("d.wav"));
}
