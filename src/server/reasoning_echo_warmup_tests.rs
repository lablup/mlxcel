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

//! The next-turn warm-up (#1144) against reasoning re-injection (#2118): the
//! warmed target must be a prefix of what a content-only follow-up renders
//! after its own fill.

use super::{JAMBA_TEMPLATE, render, request, scope, store, turn1, turn2_content_only};
use crate::server::chat_request::render_next_turn_history;
use crate::server::chat_template::ChatTemplateProcessor;
use crate::server::types::ChatCompletionRequest;

/// The head both probe renders agree on: the string form of the clip the
/// route applies to the tokenized probes (`clip_warmup_target`).
fn warm_target(
    processor: &ChatTemplateProcessor,
    rendered: &ChatCompletionRequest,
    reply: &str,
    reply_reasoning: Option<&str>,
) -> Option<String> {
    let history = render_next_turn_history(processor, rendered, None, reply, reply_reasoning)?;
    Some(
        history
            .probe_a
            .chars()
            .zip(history.probe_b.chars())
            .take_while(|(a, b)| a == b)
            .map(|(a, _)| a)
            .collect(),
    )
}

/// Turn 1 to turn 2: rendering the reply with the stored reasoning makes the
/// probes extend this turn's history and the target prefix turn 2's filled
/// render. Without the reasoning (the old warm-up) the Jamba probe renders the
/// earlier user turn without its instruction and the warm-up bails.
#[tokio::test]
async fn warmup_target_prefixes_the_filled_follow_up() {
    let processor = ChatTemplateProcessor::with_template(JAMBA_TEMPLATE.to_string());
    let store = store();
    let turn1 = turn1();
    assert!(store.record(&scope(), &turn1.messages, "Paris.", "trace"));

    assert!(
        warm_target(&processor, &turn1, "Paris.", None).is_none(),
        "a reply rendered without its reasoning cannot extend this turn"
    );
    let target = warm_target(&processor, &turn1, "Paris.", Some("trace"))
        .expect("the reply with its stored reasoning extends this turn");
    assert!(
        target.contains("<|im_start|>assistant\nParis.<|im_end|>\n<|im_start|>user\n"),
        "the target covers the reply: {target:?}"
    );

    let turn2 = turn2_content_only();
    let filled = store.fill(&scope(), &turn2);
    let next = render(&processor, &filled).await;
    assert!(
        next.starts_with(&target),
        "the warm-up target must prefix the follow-up's filled render;\n\
         target: {target:?}\nnext: {next:?}"
    );
}

/// Turn 2 to turn 3: the warm-up must render the request as it was rendered
/// (filled). The unfilled turn-2 request re-renders turn 1 content-only, so
/// its target diverges from turn 3's filled render at the first user turn.
#[tokio::test]
async fn warmup_renders_the_filled_request() {
    let processor = ChatTemplateProcessor::with_template(JAMBA_TEMPLATE.to_string());
    let store = store();
    assert!(store.record(&scope(), &turn1().messages, "Paris.", "trace"));
    let turn2 = turn2_content_only();
    assert!(store.record(&scope(), &turn2.messages, "Lyon.", "trace two"));
    let turn2_filled = store.fill(&scope(), &turn2).into_owned();

    let turn3 = request(serde_json::json!([
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "Paris."},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "Lyon."},
        {"role": "user", "content": "q3"},
    ]));
    let filled3 = store.fill(&scope(), &turn3);
    assert_eq!(filled3.messages[3].reasoning.as_deref(), Some("trace two"));
    let next = render(&processor, &filled3).await;

    let target = warm_target(&processor, &turn2_filled, "Lyon.", Some("trace two"))
        .expect("the filled request extends");
    assert!(target.contains("Lyon."), "{target:?}");
    assert!(
        next.starts_with(&target),
        "target: {target:?}\nnext: {next:?}"
    );

    let unfilled = warm_target(&processor, &turn2, "Lyon.", Some("trace two"))
        .expect("the unfilled request still extends its own history");
    assert!(
        !next.starts_with(&unfilled),
        "an unfilled warm-up keys a vector the follow-up never renders"
    );
}

/// `record` reports whether the follow-up's fill will find the trace, which is
/// what gates the warm-up's reply reasoning.
#[test]
fn record_reports_whether_the_trace_was_stored() {
    let store = store();
    assert!(store.record(&scope(), &turn1().messages, "Paris.", "trace"));
    assert!(!store.record(&scope(), &turn1().messages, "Paris.", "  "));
    assert!(!store.record(&scope(), &turn2_content_only().messages[..2], "x", "trace"));
    let small = super::ReasoningEchoStore::new(8 * 1024, 64);
    assert!(!small.record(&scope(), &turn1().messages, "Paris.", &"x".repeat(2048)));
    let disabled = super::ReasoningEchoStore::new(0, 64);
    assert!(!disabled.record(&scope(), &turn1().messages, "Paris.", "trace"));
}
