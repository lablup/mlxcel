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

//! Token-exact prompt parity with the checkpoints' own `LLMjpVLProcessor`.
//!
//! The synthetic gates in `llmjp_vl_prompt_tests.rs` pin the placement rule
//! against a stand-in tokenizer. This one pins the whole prompt against the
//! reference: the id sequences below were produced by running
//! `processing_llmjpvl.LLMjpVLProcessor.apply_chat_template(..., tokenize=True,
//! add_generation_prompt=True)` from each released checkpoint under
//! `transformers`, on a 512x512 solid image and the question quoted with each
//! case. Reproducing them here means mlxcel's rendered prompt, its
//! `<image>`-equivalent placement, its framing ids and its tokenization all
//! agree with the checkpoint's own pipeline, not merely with each other.
//!
//! A prompt-token *count* match is deliberately not the assertion. Two front
//! ends can agree on a count while rendering different text, which is how an
//! earlier VLM port in this tree shipped a wrong prompt with matching numbers.
//! The gate compares every id.

use std::path::PathBuf;

use super::llmjp_vl_prompt::insert_llmjp_image_tokens;
use crate::server::chat_template::{ChatMessage, ChatTemplateProcessor};

/// One checkpoint's reference prompt, factored into the parts around the
/// `<|image_pad|>` run so the 256-element run does not have to be written out.
struct ReferencePrompt {
    env_key: &'static str,
    repo: &'static str,
    local_dirs: &'static [&'static str],
    question: &'static str,
    /// `(<|image_pad|>, <|image_start|>, <|image_end|>)`.
    image_ids: (i32, i32, i32),
    /// Everything before the first `<|image_pad|>`, ending at
    /// `<|image_start|>`.
    prefix: &'static [i32],
    pads: usize,
    /// Everything from `<|image_end|>` to the end of the generation prompt.
    suffix: &'static [i32],
}

const JAGLE: ReferencePrompt = ReferencePrompt {
    env_key: "MLXCEL_TEST_JAGLE_VL_DIR",
    repo: "llm-jp/Jagle-VL-2.2B-Jagle-FineVision",
    local_dirs: &[
        "models/mlx/jagle-vl-2.2b-jagle-finevision",
        "models/Jagle-VL-2.2B-Jagle-FineVision",
    ],
    question: "この画像の色は何色ですか。",
    image_ids: (151655, 151669, 151670),
    prefix: &[
        151671, 8948, 151673, 2610, 525, 444, 10994, 13333, 79, 19625, 43, 11, 264, 22162, 318,
        57597, 444, 10994, 16176, 553, 444, 10994, 13333, 79, 13, 151672, 151671, 872, 151673,
        151669,
    ],
    pads: 256,
    suffix: &[
        151670, 50230, 116542, 15767, 38035, 136045, 38035, 131938, 1773, 151672, 151671, 77091,
        151674, 11822, 151673,
    ],
};

const LLMJP_9B: ReferencePrompt = ReferencePrompt {
    env_key: "MLXCEL_TEST_LLMJP_4VL_9B_DIR",
    repo: "llm-jp/llm-jp-4-vl-9B-beta",
    local_dirs: &[
        "models/mlx/llm-jp-4-vl-9b-beta",
        "models/llm-jp-4-vl-9B-beta",
    ],
    question: "この画像について説明してください。",
    image_ids: (14, 15, 16),
    prefix: &[
        10, 1116, 12, 989, 660, 22241, 900, 623, 3198, 623, 48378, 608, 616, 17176, 34423, 22241,
        900, 6074, 652, 22241, 900, 623, 3198, 611, 11, 10, 4358, 12, 15,
    ],
    pads: 256,
    suffix: &[16, 2158, 4005, 71911, 6943, 621, 11, 10, 12811, 9, 2520, 12],
};

impl ReferencePrompt {
    fn expected(&self) -> Vec<i32> {
        let mut out = Vec::with_capacity(self.prefix.len() + self.pads + self.suffix.len());
        out.extend_from_slice(self.prefix);
        out.extend(std::iter::repeat_n(self.image_ids.0, self.pads));
        out.extend_from_slice(self.suffix);
        out
    }

    fn checkpoint(&self) -> Option<PathBuf> {
        if let Ok(dir) = std::env::var(self.env_key) {
            let dir = PathBuf::from(dir);
            if dir.join("config.json").is_file() {
                return Some(dir);
            }
            return None;
        }
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let mut candidates: Vec<PathBuf> = crate::downloader::model_dir(self.repo)
            .into_iter()
            .collect();
        candidates.extend(self.local_dirs.iter().map(|dir| manifest.join(dir)));
        candidates
            .into_iter()
            .find(|dir| dir.join("config.json").is_file())
    }
}

/// Build the prompt the CLI and server build for a one-image request: render
/// the chat template from the checkpoint, tokenize it, then splice the framed
/// image block in.
fn build_prompt_tokens(case: &ReferencePrompt, dir: &std::path::Path) -> Vec<i32> {
    let processor = ChatTemplateProcessor::from_model_path(dir)
        .expect("chat template loads")
        .expect("checkpoint ships a chat template");
    let rendered = processor
        .apply(
            &[ChatMessage {
                role: "user".to_string(),
                content: case.question.to_string(),
            }],
            None,
        )
        .expect("template renders");

    let tokenizer = crate::tokenizer::load_tokenizer(dir).expect("tokenizer loads");
    let mut encode = |text: &str, add_special: bool| -> Vec<i32> {
        tokenizer
            .encode(text, add_special)
            .unwrap_or_default()
            .iter()
            .map(|&t| t as i32)
            .collect()
    };

    let mut prompt_tokens = encode(&rendered, false);
    let (pad, start, end) = case.image_ids;
    let stats = insert_llmjp_image_tokens(
        &rendered,
        &mut prompt_tokens,
        &[1],
        256,
        start,
        pad,
        end,
        &mut encode,
    )
    .expect("one image block is inserted");
    assert_eq!(stats.image_blocks, 1);
    assert_eq!(stats.total_image_tokens, 256);
    prompt_tokens
}

#[test]
fn one_image_prompts_are_token_exact_against_the_reference_processor() {
    let mut checked = 0usize;
    for case in [&JAGLE, &LLMJP_9B] {
        let Some(dir) = case.checkpoint() else {
            eprintln!("skipping real-checkpoint gate: {} not present", case.repo);
            continue;
        };
        checked += 1;
        let produced = build_prompt_tokens(case, &dir);
        let expected = case.expected();
        assert_eq!(
            produced.len(),
            expected.len(),
            "{}: prompt length (produced {} vs reference {})",
            case.repo,
            produced.len(),
            expected.len()
        );
        if let Some(index) = produced.iter().zip(&expected).position(|(a, b)| a != b) {
            panic!(
                "{}: first divergence at index {index}: produced {} vs reference {}",
                case.repo, produced[index], expected[index]
            );
        }
    }

    if checked == 0 {
        eprintln!("skipping: neither LLM-jp-VL checkpoint is downloaded");
    }
}
