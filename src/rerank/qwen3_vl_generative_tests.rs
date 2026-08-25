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

//! Qwen3-VL reranker prompt gates.
//!
//! The forward pass needs the 2B checkpoint and is covered by
//! `real_checkpoint_tests.rs`. What is asserted here is everything the module
//! decides before the weights matter: the message list handed to the
//! checkpoint's own `reranker.jinja`, the rendering that template produces for
//! text and image sides, the truncation split, and the yes/no ids read from
//! `1_LogitScore/config.json`.
//!
//! The rendering tests read the real template out of the checkpoint directory
//! and soft-skip when it is not downloaded, rather than pinning a copy of
//! someone else's template into this repository.

use serde_json::json;

use super::*;
use crate::models::embedding_test_support::local_checkpoint;

const RERANKER: &str = "Qwen/Qwen3-VL-Reranker-2B";

/// Load the checkpoint's reranker template, or `None` to soft-skip.
fn reranker_template() -> Option<ChatTemplateProcessor> {
    let dir = local_checkpoint(RERANKER)?;
    let source = std::fs::read_to_string(dir.join(RERANKER_TEMPLATE_PATH)).ok()?;
    Some(ChatTemplateProcessor::with_template(source))
}

#[test]
fn messages_omit_the_system_turn_when_no_instruction_is_given() {
    let messages = rerank_messages(None, "a query", false, "a document", false);
    let array = messages.as_array().expect("array");
    assert_eq!(array.len(), 2, "no system turn: {messages}");
    assert_eq!(array[0]["role"], "query");
    assert_eq!(array[1]["role"], "document");
    assert_eq!(array[0]["content"][0]["text"], "a query");

    // A blank instruction is not an instruction either.
    assert_eq!(
        rerank_messages(Some("  "), "q", false, "d", false)
            .as_array()
            .expect("array")
            .len(),
        2
    );

    let with_instruction = rerank_messages(Some("Find charts"), "q", true, "d", true);
    let array = with_instruction.as_array().expect("array");
    assert_eq!(array.len(), 3);
    assert_eq!(array[0]["role"], "system");
    assert_eq!(array[0]["content"][0]["text"], "Find charts");
    assert_eq!(
        array[1]["content"][0],
        json!({"type": "image"}),
        "the image content item comes before the text one"
    );
    assert_eq!(array[1]["content"][1]["text"], "q");
    assert_eq!(array[2]["content"][0], json!({"type": "image"}));
}

#[test]
fn image_only_side_carries_no_text_item() {
    let messages = rerank_messages(None, "", true, "", true);
    let array = messages.as_array().expect("array");
    assert_eq!(array[0]["content"].as_array().expect("array").len(), 1);
    assert_eq!(array[0]["content"][0], json!({"type": "image"}));
    assert_eq!(array[1]["content"].as_array().expect("array").len(), 1);
}

#[test]
fn checkpoint_template_renders_the_reference_prompt() {
    let Some(template) = reranker_template() else {
        eprintln!("skipping: {RERANKER} is not downloaded");
        return;
    };
    let rendered = template
        .apply_raw(
            &rerank_messages(None, "what is panda?", false, "a bear species", false),
            None,
        )
        .expect("template renders");
    assert_eq!(
        rendered,
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the \
         Query and the Instruct provided. Note that the answer can only be \"yes\" or \
         \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: Given a search query, retrieve \
         relevant candidates that answer the query.<Query>:what is panda?\n<Document>:a bear \
         species<|im_end|>\n<|im_start|>assistant\n",
        "the template supplies its own default instruction and the generation prompt"
    );

    // A supplied instruction replaces the template default.
    let custom = template
        .apply_raw(
            &rerank_messages(Some("Find charts."), "q", false, "d", false),
            None,
        )
        .expect("template renders");
    assert!(
        custom.contains("<Instruct>: Find charts.<Query>:q"),
        "{custom}"
    );
    assert!(!custom.contains("Given a search query"), "{custom}");
}

#[test]
fn checkpoint_template_renders_image_placeholders() {
    let Some(template) = reranker_template() else {
        eprintln!("skipping: {RERANKER} is not downloaded");
        return;
    };
    let rendered = template
        .apply_raw(&rerank_messages(None, "", true, "", true), None)
        .expect("template renders");
    assert_eq!(
        rendered
            .matches("<|vision_start|><|image_pad|><|vision_end|>")
            .count(),
        2,
        "one placeholder per image side: {rendered}"
    );
    assert!(rendered.ends_with("<|im_start|>assistant\n"), "{rendered}");
}

#[test]
fn logit_score_ids_come_from_the_checkpoint_module() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_eq!(
        logit_score_ids(dir.path()),
        None,
        "a checkpoint without the module has no yes/no ids to read"
    );
    std::fs::create_dir_all(dir.path().join("1_LogitScore")).expect("module dir");
    std::fs::write(
        dir.path().join(LOGIT_SCORE_CONFIG_PATH),
        r#"{"true_token_id": 9693, "false_token_id": 2152}"#,
    )
    .expect("module config");
    assert_eq!(logit_score_ids(dir.path()), Some((9693, 2152)));

    // The real checkpoint publishes the same pair.
    if let Some(real) = local_checkpoint(RERANKER) {
        assert_eq!(
            logit_score_ids(&real),
            Some((9693, 2152)),
            "{RERANKER} publishes yes=9693 / no=2152"
        );
    }
}

#[test]
fn truncation_splits_the_pair_longest_first() {
    // The scaffold and the image tokens are reserved first, then the leftover
    // budget is split between the two sides.
    assert_eq!(longest_first_keep(4, 100, 20), (4, 16));
    assert_eq!(longest_first_keep(100, 4, 20), (16, 4));
    assert_eq!(longest_first_keep(100, 100, 20), (10, 10));
    assert_eq!(
        longest_first_keep(3, 4, 20),
        (3, 4),
        "a pair that already fits is untouched"
    );
    assert_eq!(
        longest_first_keep(5, 5, 0),
        (0, 0),
        "an exhausted budget keeps nothing rather than panicking"
    );
}
