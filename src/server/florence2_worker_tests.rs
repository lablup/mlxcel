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

//! Request-boundary tests for the Florence-2 seq2seq worker (issue #1073):
//! task-input validation for the 7 input-taking modes and the strict
//! `<loc_*>` region-quadruple parser. The end-to-end loop (encoder pass,
//! decode, response mapping) is validated against a real checkpoint through
//! the server; the per-request encoder-state isolation property is covered
//! at the model level in
//! `models/florence2/florence2_tests.rs::sequential_requests_are_isolated`.

use super::*;
use crate::models::florence2::Florence2Task;

// ---------------------------------------------------------------------------
// parse_region_bins
// ---------------------------------------------------------------------------

#[test]
fn region_bins_accepts_exactly_four_loc_tokens() {
    assert_eq!(
        parse_region_bins("<loc_52><loc_332><loc_932><loc_774>"),
        Some([52, 332, 932, 774])
    );
    assert_eq!(
        parse_region_bins("<loc_0><loc_0><loc_999><loc_999>"),
        Some([0, 0, 999, 999])
    );
}

#[test]
fn region_bins_rejects_malformed_forms() {
    // too few / too many tokens
    assert_eq!(parse_region_bins("<loc_1><loc_2><loc_3>"), None);
    assert_eq!(
        parse_region_bins("<loc_1><loc_2><loc_3><loc_4><loc_5>"),
        None
    );
    // junk between, before, or after tokens
    assert_eq!(parse_region_bins("<loc_1> <loc_2><loc_3><loc_4>"), None);
    assert_eq!(parse_region_bins("x<loc_1><loc_2><loc_3><loc_4>"), None);
    assert_eq!(parse_region_bins("<loc_1><loc_2><loc_3><loc_4>x"), None);
    // non-numeric, empty, negative, out-of-range, over-long digits
    assert_eq!(parse_region_bins("<loc_a><loc_2><loc_3><loc_4>"), None);
    assert_eq!(parse_region_bins("<loc_><loc_2><loc_3><loc_4>"), None);
    assert_eq!(parse_region_bins("<loc_-1><loc_2><loc_3><loc_4>"), None);
    assert_eq!(parse_region_bins("<loc_1000><loc_2><loc_3><loc_4>"), None);
    assert_eq!(parse_region_bins("<loc_0001><loc_2><loc_3><loc_4>"), None);
    // empty input
    assert_eq!(parse_region_bins(""), None);
}

// ---------------------------------------------------------------------------
// validate_task_input
// ---------------------------------------------------------------------------

/// The 7 input-taking task modes named by the issue: 3 free-text, 4 region.
const TEXT_INPUT_TASKS: [Florence2Task; 3] = [
    Florence2Task::CaptionToPhraseGrounding,
    Florence2Task::ReferringExpressionSegmentation,
    Florence2Task::OpenVocabularyDetection,
];
const REGION_INPUT_TASKS: [Florence2Task; 4] = [
    Florence2Task::RegionToSegmentation,
    Florence2Task::RegionToCategory,
    Florence2Task::RegionToDescription,
    Florence2Task::RegionToOcr,
];

/// Drift guard: the validator's task partition must match `takes_input()`.
#[test]
fn input_taking_task_set_matches_takes_input() {
    let validated: Vec<Florence2Task> = TEXT_INPUT_TASKS
        .into_iter()
        .chain(REGION_INPUT_TASKS)
        .collect();
    for task in Florence2Task::ALL {
        assert_eq!(
            task.takes_input(),
            validated.contains(&task),
            "takes_input()/validator drift for {}",
            task.token()
        );
    }
}

#[test]
fn absent_input_is_left_to_expand() {
    // Presence/absence is Florence2Task::expand's call; the boundary
    // validator only constrains input that IS present.
    for task in Florence2Task::ALL {
        assert!(validate_task_input(task, None).is_ok(), "{}", task.token());
    }
}

#[test]
fn input_on_inputless_task_is_rejected() {
    let err = validate_task_input(Florence2Task::Caption, Some("extra")).unwrap_err();
    assert!(err.contains("takes no input text"), "got: {err}");
}

#[test]
fn oversized_input_is_rejected_for_every_input_taking_task() {
    let oversized = "a".repeat(MAX_TASK_INPUT_BYTES + 1);
    for task in TEXT_INPUT_TASKS.into_iter().chain(REGION_INPUT_TASKS) {
        let err = validate_task_input(task, Some(&oversized)).unwrap_err();
        assert!(err.contains("at most"), "{}: {err}", task.token());
    }
}

#[test]
fn control_characters_are_rejected() {
    for input in ["a\u{0}b", "a\nb", "a\u{1b}[31mb"] {
        let err =
            validate_task_input(Florence2Task::CaptionToPhraseGrounding, Some(input)).unwrap_err();
        assert!(err.contains("control characters"), "got: {err}");
    }
}

#[test]
fn free_text_tasks_accept_plain_text() {
    for task in TEXT_INPUT_TASKS {
        assert!(
            validate_task_input(task, Some("a green car parked by the road")).is_ok(),
            "{}",
            task.token()
        );
    }
}

#[test]
fn free_text_tasks_reject_angle_brackets() {
    for task in TEXT_INPUT_TASKS {
        for input in ["<loc_1>", "a <CAPTION> b", "1 < 2", "</s>"] {
            let err = validate_task_input(task, Some(input)).unwrap_err();
            assert!(
                err.contains("without '<' or '>'"),
                "{} {input:?}: {err}",
                task.token()
            );
        }
    }
}

#[test]
fn region_tasks_accept_well_formed_region() {
    for task in REGION_INPUT_TASKS {
        assert!(
            validate_task_input(task, Some("<loc_52><loc_332><loc_932><loc_774>")).is_ok(),
            "{}",
            task.token()
        );
    }
}

#[test]
fn region_tasks_reject_free_text_and_malformed_regions() {
    for task in REGION_INPUT_TASKS {
        for input in [
            "the red car",
            "<loc_1><loc_2><loc_3>",
            "<loc_1><loc_2><loc_3><loc_4000>",
            "<loc_1><loc_2><loc_3><loc_4> trailing",
        ] {
            let err = validate_task_input(task, Some(input)).unwrap_err();
            assert!(
                err.contains("exactly four location tokens"),
                "{} {input:?}: {err}",
                task.token()
            );
        }
    }
}

#[test]
fn max_input_bound_admits_a_long_grounding_caption() {
    // A whole <MORE_DETAILED_CAPTION> paragraph fed back for grounding is
    // the longest legitimate input; the bound must not reject it.
    let paragraph = "The image shows two cats sleeping on a pink couch. ".repeat(20);
    assert!(paragraph.len() <= MAX_TASK_INPUT_BYTES);
    assert!(
        validate_task_input(
            Florence2Task::CaptionToPhraseGrounding,
            Some(paragraph.trim())
        )
        .is_ok()
    );
}
