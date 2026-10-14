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

//! Token-exact one-image prompt parity with mlx-vlm 0.6.17 on
//! `granite-vision-3.2-2b-4bit` (issue #1683).
//!
//! The reference ids were produced by mlx-vlm's own pipeline on the same
//! checkpoint: `prompt_utils.apply_chat_template(processor, config, question,
//! num_images=1)` followed by `utils.prepare_inputs(...)` with the 224x224
//! `tests/fixtures/test_image.png`. They are factored into the ids before the
//! `<image>` run, the run length, and the ids after it.
//!
//! The divergence this pins was in tokenization, not rendering: mlxcel rendered
//! the same text, but `<image>` (id 49155) is declared only in the checkpoint's
//! `tokenizer_config.json`, so it was split into `<`, `image`, `>`. With no
//! placeholder in the ids, the 1485 image tokens were spliced after the first
//! token, inside `<|system|>`, and the user turn kept the literal text. A
//! prompt-token count cannot catch that; this gate compares every id.

use std::path::PathBuf;

use super::granite_vision_prompt::insert_granite_vision_image_tokens;
use crate::server::chat_template::ChatTemplateProcessor;
use crate::vision::processors::anyres::AnyResProcessor;

const IMAGE_TOKEN: i32 = 49155;
const FEATURE_SIDE: i32 = 27;
const BASE_TOKENS: i32 = FEATURE_SIDE * FEATURE_SIDE;
/// One 224x224 image: base 27x27 plus one 384x384 tile of 27 rows of 27+1.
const IMAGE_TOKENS: usize = 1485;

/// `<|system|>\n` + the template's default system prompt + `<|user|>\n`.
const PREFIX: &[i32] = &[
    46, 110, 2946, 28318, 203, 51, 11210, 3733, 312, 39489, 1256, 461, 600, 5549, 31251, 629,
    21488, 47330, 32, 886, 47330, 13344, 17247, 30, 16360, 30, 461, 7743, 659, 19969, 372, 322,
    1256, 1182, 10017, 32, 203, 46, 110, 496, 28318, 203,
];

/// `(question, ids after the image run through the generation prompt)`.
const CASES: &[(&str, &[i32])] = &[
    (
        "What is in this image? Describe it briefly.",
        &[
            203, 8197, 438, 328, 458, 1778, 49, 11616, 561, 25585, 631, 32, 203, 46, 110, 17594,
            28318, 203,
        ],
    ),
    (
        "What color is this image? Answer with one word.",
        &[
            203, 8197, 1963, 438, 458, 1778, 49, 23574, 623, 1591, 3594, 32, 203, 46, 110, 17594,
            28318, 203,
        ],
    ),
];

fn checkpoint() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("MLXCEL_TEST_GRANITE_VISION_DIR") {
        let dir = PathBuf::from(dir);
        return dir.join("config.json").is_file().then_some(dir);
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut candidates: Vec<PathBuf> =
        crate::downloader::model_dir("mlx-community/granite-vision-3.2-2b-4bit")
            .into_iter()
            .collect();
    candidates.push(manifest.join("models/mlx/granite-vision-3.2-2b-4bit"));
    candidates
        .into_iter()
        .find(|dir| dir.join("config.json").is_file())
}

/// The prompt the CLI and server build for a one-image request: the checkpoint
/// template rendered with an image item ahead of the question, tokenized, then
/// the `<image>` placeholder expanded for a 224x224 image.
fn build_prompt_tokens(dir: &std::path::Path, question: &str) -> Vec<i32> {
    let processor = ChatTemplateProcessor::from_model_path(dir)
        .expect("chat template loads")
        .expect("checkpoint ships a chat template");
    let messages = serde_json::json!([{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": question}],
    }]);
    let rendered = processor
        .apply_raw(&messages, None)
        .expect("template renders");

    let tokenizer = crate::tokenizer::load_tokenizer(dir).expect("tokenizer loads");
    let mut prompt_tokens: Vec<i32> = tokenizer
        .encode(&rendered, false)
        .expect("prompt encodes")
        .iter()
        .map(|&t| t as i32)
        .collect();

    // A 224x224 image selects the 384x384 pinpoint from any Granite Vision
    // pinpoint list, so the leading entries suffice here.
    let anyres = AnyResProcessor::new(vec![(384, 384), (384, 768), (768, 384)], 384);
    let info = anyres.tile_info(224, 224);
    let stats = insert_granite_vision_image_tokens(
        &mut prompt_tokens,
        &[info],
        IMAGE_TOKEN,
        FEATURE_SIDE,
        BASE_TOKENS,
    )
    .expect("one image block is inserted");
    assert!(
        !stats.spliced,
        "the rendered prompt carried no <image> placeholder; the image block was spliced \
         after the first token instead of expanded in the user turn"
    );
    assert_eq!(stats.total_image_tokens as usize, IMAGE_TOKENS);
    prompt_tokens
}

#[test]
fn one_image_prompts_are_token_exact_against_mlx_vlm() {
    let Some(dir) = checkpoint() else {
        eprintln!("skipping real-checkpoint gate: granite-vision-3.2-2b-4bit not present");
        return;
    };
    for (question, suffix) in CASES {
        let produced = build_prompt_tokens(&dir, question);
        let mut expected = PREFIX.to_vec();
        expected.extend(std::iter::repeat_n(IMAGE_TOKEN, IMAGE_TOKENS));
        expected.extend_from_slice(suffix);

        if let Some(index) = produced.iter().zip(&expected).position(|(a, b)| a != b) {
            panic!(
                "{question:?}: first divergence at index {index}: produced {} vs reference {}",
                produced[index], expected[index]
            );
        }
        assert_eq!(
            produced.len(),
            expected.len(),
            "{question:?}: prompt length (produced {} vs reference {})",
            produced.len(),
            expected.len()
        );
    }
}
