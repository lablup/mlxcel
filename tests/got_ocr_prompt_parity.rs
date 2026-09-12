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

//! GOT-OCR 2.0 prompt parity against the reference tokenizer, on a real
//! checkpoint.
//!
//! The unit tests in `src/multimodal/got_ocr_prompt_tests.rs` pin the assembled
//! *string*. This pins the *ids*, which is the thing that actually has to be
//! right and the thing the two shipped special-token tables disagree about: the
//! tiktoken loader used to build only the HunYuan table, under which
//! `<|im_end|>` and `<imgpad>` resolve to nothing at all and the image block
//! silently tokenizes as literal text.
//!
//! The expected vectors were produced by the checkpoint's own
//! `tokenization_qwen.py` contract: `tiktoken.Encoding` over the 151643 ranks in
//! `qwen.tiktoken` with `SPECIAL_TOKENS + IMAGE_ST` numbered from
//! `len(mergeable_ranks)`, encoding the string `modeling_GOT.py::chat` builds.
//!
//! ## Invocation
//!
//! ```bash
//! cargo test --test got_ocr_prompt_parity --profile test-fast --features cuda -- --ignored --nocapture
//! ```
//!
//! `#[ignore]`-gated because it needs a checkpoint on disk; skips with a note
//! when none is present.

mod common;

use common::repo_model_dir;

use mlxcel::multimodal::got_ocr_prompt::build_got_prompt;
use mlxcel::tokenizer::load_tokenizer;

/// Any GOT checkpoint will do: all three released directories ship the same
/// `qwen.tiktoken` and the same `tokenizer_config.json`.
const MODEL_DIRS: &[&str] = &["got-ocr2_0-bf16", "got-ocr2_0-4bit", "got-ocr2_0-original"];

const IMAGE_TOKEN_LEN: usize = 256;

/// `<|im_start|>system\n        You should follow the instructions carefully and
/// explain your answers in detail.<|im_end|><|im_start|>user\n<img>` ...
const EXPECTED_HEAD: &[u32] = &[
    151644, 8948, 198, 286, 1446, 1265, 1795, 279, 11221, 15516, 323, 10339, 697, 11253, 304, 7716,
    13, 151645, 151644, 872, 198, 151857,
];

/// ... `</img>\nOCR: <|im_end|><|im_start|>assistant\n`
const EXPECTED_TAIL: &[u32] = &[151858, 198, 93495, 25, 220, 151645, 151644, 77091, 198];

/// ... `</img>\nOCR with format: <|im_end|><|im_start|>assistant\n`
const EXPECTED_FORMAT_TAIL: &[u32] = &[
    151858, 198, 93495, 448, 3561, 25, 220, 151645, 151644, 77091, 198,
];

const IMG_START_ID: u32 = 151857;
const IMG_END_ID: u32 = 151858;
const IMG_PAD_ID: u32 = 151859;
const IM_END_ID: u32 = 151645;
const ENDOFTEXT_ID: u32 = 151643;

fn resolve_checkpoint() -> Option<std::path::PathBuf> {
    MODEL_DIRS
        .iter()
        .map(|name| repo_model_dir(name))
        .find(|dir| dir.join("qwen.tiktoken").exists())
}

#[test]
#[ignore = "needs a GOT-OCR 2.0 checkpoint on disk"]
fn got_prompt_ids_match_the_reference_tokenizer() {
    let Some(model_dir) = resolve_checkpoint() else {
        eprintln!(
            "Skipping: no GOT-OCR 2.0 checkpoint found.\n\
             Fetch with: mlxcel download mlx-community/GOT-OCR2_0-bf16"
        );
        return;
    };
    let tokenizer = load_tokenizer(&model_dir).expect("load GOT tokenizer");

    // The special ids the checkpoint's `config.json` declares must be the ids
    // the tokenizer actually resolves. This is the check that fails outright
    // under the HunYuan table.
    for (spelling, expected) in [
        ("<|endoftext|>", ENDOFTEXT_ID),
        ("<|im_start|>", 151644),
        ("<|im_end|>", IM_END_ID),
        ("<img>", IMG_START_ID),
        ("</img>", IMG_END_ID),
        ("<imgpad>", IMG_PAD_ID),
    ] {
        assert_eq!(
            tokenizer.token_to_id(spelling),
            Some(expected),
            "{spelling} resolved to the wrong id"
        );
    }

    let (prompt, stats) = build_got_prompt("OCR: ", 1, IMAGE_TOKEN_LEN).expect("build prompt");
    assert_eq!(stats.total_image_tokens, IMAGE_TOKEN_LEN);
    let ids = tokenizer.encode(&prompt, false).expect("encode prompt");

    assert_eq!(ids.len(), 287, "prompt length");
    assert_eq!(&ids[..EXPECTED_HEAD.len()], EXPECTED_HEAD, "prompt head");
    assert_eq!(
        &ids[ids.len() - EXPECTED_TAIL.len()..],
        EXPECTED_TAIL,
        "prompt tail"
    );

    // The block between the framing tags is one uninterrupted run of exactly
    // `image_token_len` placeholders, which is what `merge_llava` pairs with
    // the tower's 256 feature rows.
    let start = ids
        .iter()
        .position(|&id| id == IMG_START_ID)
        .expect("<img>");
    let end = ids.iter().position(|&id| id == IMG_END_ID).expect("</img>");
    assert_eq!(end - start - 1, IMAGE_TOKEN_LEN, "placeholder run length");
    assert!(
        ids[start + 1..end].iter().all(|&id| id == IMG_PAD_ID),
        "placeholder run is not uniform"
    );
    assert_eq!(ids.iter().filter(|&&id| id == IMG_START_ID).count(), 1);
    assert_eq!(ids.iter().filter(|&&id| id == IMG_END_ID).count(), 1);
}

#[test]
#[ignore = "needs a GOT-OCR 2.0 checkpoint on disk"]
fn got_format_mode_prompt_ids_match_the_reference_tokenizer() {
    let Some(model_dir) = resolve_checkpoint() else {
        eprintln!("Skipping: no GOT-OCR 2.0 checkpoint found.");
        return;
    };
    let tokenizer = load_tokenizer(&model_dir).expect("load GOT tokenizer");
    let (prompt, _) =
        build_got_prompt("OCR with format: ", 1, IMAGE_TOKEN_LEN).expect("build prompt");
    let ids = tokenizer.encode(&prompt, false).expect("encode prompt");

    assert_eq!(ids.len(), 289, "prompt length");
    assert_eq!(&ids[..EXPECTED_HEAD.len()], EXPECTED_HEAD, "prompt head");
    assert_eq!(
        &ids[ids.len() - EXPECTED_FORMAT_TAIL.len()..],
        EXPECTED_FORMAT_TAIL,
        "prompt tail"
    );
}

/// The prompt a server render produces tokenizes to the identical ids as the
/// bare CLI instruction.
///
/// Matching *counts* between two front ends proves nothing; this compares every
/// id. The render is the builtin GOT template's output, which is what
/// `ChatTemplateProcessor` hands the runtime for a chat request.
#[test]
#[ignore = "needs a GOT-OCR 2.0 checkpoint on disk"]
fn server_render_and_cli_instruction_tokenize_identically() {
    let Some(model_dir) = resolve_checkpoint() else {
        eprintln!("Skipping: no GOT-OCR 2.0 checkpoint found.");
        return;
    };
    let tokenizer = load_tokenizer(&model_dir).expect("load GOT tokenizer");

    let (cli, _) = build_got_prompt("OCR: ", 1, IMAGE_TOKEN_LEN).expect("cli prompt");
    let rendered =
        mlxcel::server::chat_template::ChatTemplateProcessor::from_model_path(&model_dir)
            .expect("template lookup")
            .expect("GOT ships no template of its own, so the builtin must be selected")
            .apply(
                &[mlxcel::server::chat_template::ChatMessage {
                    role: "user".to_string(),
                    content: "OCR: ".to_string(),
                }],
                None,
            )
            .expect("render");
    let (server, stats) = build_got_prompt(&rendered, 1, IMAGE_TOKEN_LEN).expect("server prompt");
    assert!(
        stats.pre_templated,
        "the render must be recognized as already framed"
    );

    assert_eq!(server, cli, "server and CLI prompts differ");
    assert_eq!(
        tokenizer.encode(&server, false).expect("encode server"),
        tokenizer.encode(&cli, false).expect("encode cli"),
    );
}
