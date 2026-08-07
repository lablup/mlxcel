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

//! Unit tests for the Florence-2 runtime surface: CLI task-prompt parsing.
//! Model-backed behavior (loading, LanguageModel forward) needs the real
//! checkpoint and is covered by `tests/florence2_fusion_parity.rs` and the
//! CLI validation in the integration issue.

use super::parse_task_prompt;
use crate::models::florence2::Florence2Task;

#[test]
fn parses_bare_marker_forms() {
    for prompt in ["<OD>", "OD", "od", " <od> "] {
        let (task, input) = parse_task_prompt(prompt).unwrap();
        assert_eq!(task, Florence2Task::ObjectDetection, "prompt {prompt:?}");
        assert_eq!(input, None, "prompt {prompt:?}");
    }
}

#[test]
fn parses_every_task_marker_round_trip() {
    for task in Florence2Task::ALL {
        let (parsed, input) = parse_task_prompt(task.token()).unwrap();
        assert_eq!(parsed, task);
        assert_eq!(input, None);
    }
}

#[test]
fn parses_marker_with_input_text() {
    let (task, input) = parse_task_prompt("<CAPTION_TO_PHRASE_GROUNDING> a green car").unwrap();
    assert_eq!(task, Florence2Task::CaptionToPhraseGrounding);
    assert_eq!(input.as_deref(), Some("a green car"));

    // Bare marker followed by input text.
    let (task, input) = parse_task_prompt("caption_to_phrase_grounding a green car").unwrap();
    assert_eq!(task, Florence2Task::CaptionToPhraseGrounding);
    assert_eq!(input.as_deref(), Some("a green car"));
}

#[test]
fn parses_region_input_without_separating_space() {
    let (task, input) =
        parse_task_prompt("<REGION_TO_CATEGORY><loc_52><loc_332><loc_932><loc_774>").unwrap();
    assert_eq!(task, Florence2Task::RegionToCategory);
    assert_eq!(
        input.as_deref(),
        Some("<loc_52><loc_332><loc_932><loc_774>")
    );
}

#[test]
fn rejects_unknown_and_malformed_prompts() {
    // Unknown marker.
    let err = parse_task_prompt("<NOT_A_TASK>").unwrap_err();
    assert!(
        err.contains("<OD>"),
        "error should list valid markers: {err}"
    );

    // Free-form text with no marker.
    let err = parse_task_prompt("describe this image").unwrap_err();
    assert!(
        err.contains("<CAPTION>"),
        "error should list valid markers: {err}"
    );

    // Unclosed marker.
    let err = parse_task_prompt("<OD").unwrap_err();
    assert!(err.contains("closing"), "unexpected error: {err}");

    // Empty prompt.
    let err = parse_task_prompt("   ").unwrap_err();
    assert!(err.contains("empty"), "unexpected error: {err}");
}

#[test]
fn input_split_defers_validation_to_expand() {
    // `<OD>` takes no input; the parser still splits the syntax and `expand`
    // rejects it, keeping the strict boundary in one place.
    let (task, input) = parse_task_prompt("<OD> spurious text").unwrap();
    assert_eq!(task, Florence2Task::ObjectDetection);
    assert_eq!(input.as_deref(), Some("spurious text"));
    assert!(task.expand(input.as_deref()).is_err());

    // A task that requires input still errors in `expand` when it is absent.
    let (task, input) = parse_task_prompt("<OPEN_VOCABULARY_DETECTION>").unwrap();
    assert_eq!(task, Florence2Task::OpenVocabularyDetection);
    assert_eq!(input, None);
    assert!(task.expand(None).is_err());
}
