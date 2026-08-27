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

//! Unit tests for b10621 reasoning-format placement (issue #1447).

use super::*;

const ANSWER: &str = "The answer is 42.";
const WITH_THOUGHTS: &str = "<think>Let me count.</think>The answer is 42.";
const THOUGHTS: &str = "Let me count.";

fn shape(format: ReasoningFormat) -> ShapedResponse {
    shape_response(
        format,
        ANSWER.to_owned(),
        || WITH_THOUGHTS.to_owned(),
        Some(THOUGHTS.to_owned()),
    )
}

// ── parsing ─────────────────────────────────────────────────────────────────

#[test]
fn every_b10621_name_parses_to_its_own_format() {
    for (name, expected) in [
        ("auto", ReasoningFormat::Auto),
        ("none", ReasoningFormat::None),
        ("deepseek", ReasoningFormat::DeepSeek),
        ("deepseek-legacy", ReasoningFormat::DeepSeekLegacy),
    ] {
        assert_eq!(ReasoningFormat::parse(name), Ok(expected), "{name}");
        assert_eq!(expected.as_str(), name, "round trip for {name}");
    }
}

#[test]
fn an_unknown_name_is_reported_with_the_upstream_vocabulary() {
    // Case-sensitive, exactly as `common_reasoning_format_from_name` is.
    for name in ["DeepSeek", "legacy", "", "deepseek_legacy", "Auto"] {
        let err = ReasoningFormat::parse(name).expect_err("must be rejected");
        assert_eq!(err.0, name);
        let text = err.to_string();
        assert!(text.contains(name), "must quote the value: {text}");
        assert!(
            text.contains("none, deepseek, deepseek-legacy, auto"),
            "must list the accepted set: {text}"
        );
    }
}

#[test]
fn the_default_is_auto_as_it_is_upstream() {
    assert_eq!(ReasoningFormat::default(), ReasoningFormat::Auto);
}

// ── placement ───────────────────────────────────────────────────────────────

#[test]
fn none_leaves_the_thoughts_unparsed_in_content_and_emits_no_reasoning() {
    let shaped = shape(ReasoningFormat::None);
    assert_eq!(shaped.content, WITH_THOUGHTS);
    assert_eq!(shaped.reasoning_content, None);
}

#[test]
fn deepseek_puts_the_thoughts_in_reasoning_content_only() {
    for format in [ReasoningFormat::DeepSeek, ReasoningFormat::Auto] {
        let shaped = shape(format);
        assert_eq!(shaped.content, ANSWER, "{format}");
        assert_eq!(
            shaped.reasoning_content.as_deref(),
            Some(THOUGHTS),
            "{format}"
        );
    }
}

#[test]
fn deepseek_legacy_puts_the_thoughts_in_both() {
    let shaped = shape(ReasoningFormat::DeepSeekLegacy);
    assert_eq!(shaped.content, WITH_THOUGHTS);
    assert_eq!(shaped.reasoning_content.as_deref(), Some(THOUGHTS));
}

#[test]
fn auto_is_deepseek_here_because_there_is_nothing_else_to_detect() {
    // Upstream's `auto` inspects the template's declared format; mlxcel's
    // reasoning split uses one marker set for every family it supports, so the
    // two resolve to the same placement. Pinned so a future third behavior
    // cannot appear silently.
    assert_eq!(
        shape(ReasoningFormat::Auto),
        shape(ReasoningFormat::DeepSeek)
    );
}

#[test]
fn a_response_with_no_thoughts_never_gets_a_reasoning_field() {
    for format in [
        ReasoningFormat::Auto,
        ReasoningFormat::None,
        ReasoningFormat::DeepSeek,
        ReasoningFormat::DeepSeekLegacy,
    ] {
        let shaped = shape_response(format, ANSWER.to_owned(), || ANSWER.to_owned(), None);
        assert_eq!(shaped.content, ANSWER, "{format}");
        assert_eq!(shaped.reasoning_content, None, "{format}");
    }
}

// ── the two predicates the response paths branch on ─────────────────────────

#[test]
fn the_placement_predicates_partition_the_four_values() {
    for (format, keeps, emits) in [
        (ReasoningFormat::Auto, false, true),
        (ReasoningFormat::None, true, false),
        (ReasoningFormat::DeepSeek, false, true),
        (ReasoningFormat::DeepSeekLegacy, true, true),
    ] {
        assert_eq!(format.keeps_thoughts_in_content(), keeps, "{format}");
        assert_eq!(format.emits_reasoning_content(), emits, "{format}");
    }
}

#[test]
fn the_thoughts_form_is_only_built_when_a_format_needs_it() {
    // Two of the four formats never look at it, and building it is a second
    // pass over the generated text on every request.
    for (format, expected_calls) in [
        (ReasoningFormat::Auto, 0),
        (ReasoningFormat::DeepSeek, 0),
        (ReasoningFormat::None, 1),
        (ReasoningFormat::DeepSeekLegacy, 1),
    ] {
        let mut calls = 0;
        let _ = shape_response(
            format,
            ANSWER.to_owned(),
            || {
                calls += 1;
                WITH_THOUGHTS.to_owned()
            },
            Some(THOUGHTS.to_owned()),
        );
        assert_eq!(calls, expected_calls, "{format}");
    }
}
