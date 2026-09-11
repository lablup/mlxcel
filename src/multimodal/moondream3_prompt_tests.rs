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

use super::moondream3_prompt::{
    MOONDREAM3_CAPTION_NORMAL, MOONDREAM3_QUERY_PREFIX, MOONDREAM3_QUERY_SUFFIX,
    Moondream3PromptMode, prepare_moondream3_prompt_tokens,
};

#[test]
fn prepare_moondream3_prompt_tokens_uses_caption_template_for_empty_prompt() {
    let prepared = prepare_moondream3_prompt_tokens("", 1, |_text, _| vec![99]).unwrap();
    assert_eq!(prepared.mode, Moondream3PromptMode::Caption);
    assert_eq!(prepared.tokens, MOONDREAM3_CAPTION_NORMAL);
}

#[test]
fn prepare_moondream3_prompt_tokens_wraps_query_prompt_without_special_tokens() {
    let prepared = prepare_moondream3_prompt_tokens("What is this?", 1, |text, add_special| {
        assert_eq!(text, "What is this?");
        assert!(!add_special);
        vec![10, 11, 12]
    })
    .unwrap();

    let mut expected = MOONDREAM3_QUERY_PREFIX.to_vec();
    expected.extend([10, 11, 12]);
    expected.extend(MOONDREAM3_QUERY_SUFFIX);

    assert_eq!(prepared.mode, Moondream3PromptMode::Query);
    assert_eq!(prepared.tokens, expected);
}

#[test]
fn prepare_moondream3_prompt_tokens_rejects_non_single_image_requests() {
    let err = prepare_moondream3_prompt_tokens("hello", 2, |_text, _| vec![]).unwrap_err();
    assert!(err.contains("at most one image"));
}
