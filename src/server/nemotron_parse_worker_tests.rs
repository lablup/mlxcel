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

//! Request-boundary tests for the Nemotron-Parse seq2seq worker
//! (issue #1369). The generation half runs against a real checkpoint.

use super::*;

#[test]
fn empty_or_blank_text_selects_the_default_task_prompt() {
    assert_eq!(resolve_task_prompt(""), Ok(DEFAULT_TASK_PROMPT));
    assert_eq!(resolve_task_prompt("  \n "), Ok(DEFAULT_TASK_PROMPT));
}

#[test]
fn a_task_prompt_is_trimmed_and_kept() {
    let p = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>";
    assert_eq!(resolve_task_prompt(&format!(" {p} ")), Ok(p));
}

#[test]
fn an_oversized_task_prompt_is_rejected() {
    let long = "a".repeat(MAX_TASK_PROMPT_BYTES + 1);
    let err = resolve_task_prompt(&long).unwrap_err();
    assert!(err.contains("at most"), "{err}");
    assert!(resolve_task_prompt(&"a".repeat(MAX_TASK_PROMPT_BYTES)).is_ok());
}

#[test]
fn control_characters_in_the_task_prompt_are_rejected() {
    assert!(resolve_task_prompt("<predict_bbox>\u{0}x").is_err());
    assert!(resolve_task_prompt("<predict_bbox>\u{1b}[2J").is_err());
}

#[test]
fn audio_or_video_is_refused() {
    assert_eq!(reject_media(false, false), None);
    assert_eq!(
        reject_media(true, false),
        Some(NEMOTRON_PARSE_MEDIA_UNSUPPORTED_MSG)
    );
    assert_eq!(
        reject_media(false, true),
        Some(NEMOTRON_PARSE_MEDIA_UNSUPPORTED_MSG)
    );
}

#[test]
fn exactly_one_page_image_is_required() {
    assert!(reject_image_count(1).is_none());
    assert!(reject_image_count(0).unwrap().contains("got 0 images"));
    assert!(reject_image_count(2).unwrap().contains("got 2 images"));
}

#[test]
fn finish_reason_is_length_only_when_the_budget_ran_out_without_eos() {
    assert_eq!(nemotron_parse_finish_reason(true, 10, 256), "stop");
    assert_eq!(nemotron_parse_finish_reason(false, 256, 256), "length");
    // A cancelled run stops early without EOS and reports "stop".
    assert_eq!(nemotron_parse_finish_reason(false, 5, 256), "stop");
}
