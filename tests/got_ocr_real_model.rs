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

//! GOT-OCR 2.0 real-checkpoint greedy parity, and the two-layout load gate.
//!
//! The reference token ids below were produced by running the original
//! `stepfun-ai/GOT-OCR2_0` weights through the checkpoint's own
//! `got_vision_b.py::build_GOT_vit_b()` tower, an `nn.Linear(1024, 1024)`
//! projector loaded from `model.mm_projector_vary.*`, and a transformers
//! `Qwen2Model` with the tied `lm_head`, spliced exactly as
//! `modeling_GOT.py::forward` does (features replace the run between `<img>`
//! and `</img>`). Input ids came from the checkpoint's `tokenization_qwen.py`
//! contract over `qwen.tiktoken`. The reference ran in fp32 on CPU with
//! torch 2.14.0; mlxcel runs bf16 on the GPU, so the claim under test is
//! greedy token-exactness, not bitwise agreement.
//!
//! What a failure here means is worth stating, because this model fails
//! quietly. A dropped stride in the compressor, a transposed feature grid, a
//! HunYuan special table, or a lost `<imgpad>` all yield fluent text; only
//! comparing it against the page proves the vision path ran.
//!
//! ## Invocation
//!
//! ```bash
//! cargo test --test got_ocr_real_model --profile test-fast --features cuda -- --ignored --nocapture
//! ```
//!
//! `#[ignore]`-gated; skips with a note when no checkpoint is present.

mod common;

use common::repo_model_dir;

use mlxcel_core::generate::LanguageModel;

/// The MLX conversion (about 1.4 GB) and the original `stepfun-ai` layout.
/// Running both is the two-key-layout gate: they differ in every prefix, in the
/// tower neck's names, and in whether a tied `lm_head` copy is present.
const MODEL_DIRS: &[&str] = &["got-ocr2_0-bf16", "got-ocr2_0-original"];

const FIXTURE: &str = "tests/fixtures/got_ocr_page.png";

/// Reference greedy continuation for `OCR: ` on [`FIXTURE`], terminator
/// included. Decodes to `"GOT OCR two point zero \nrenders this page\n"`.
const REF_GREEDY_IDS: &[i32] = &[
    38, 1793, 80577, 1378, 1459, 7168, 715, 54159, 419, 2150, 198,
];

/// `<|im_end|>`, the id the reference stops on.
const IM_END_ID: i32 = 151645;

/// `<|endoftext|>`, the only stop `config.json` declares. Never reached.
const ENDOFTEXT_ID: i32 = 151643;

const PROMPT_TOKENS: usize = 287;
const IMAGE_TOKENS: usize = 256;
const IMG_PAD_ID: i32 = 151859;
const MAX_NEW_TOKENS: usize = 64;

fn fixture_image() -> image::DynamicImage {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE);
    image::open(&path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()))
}

fn argmax_last_token(logits: &mlxcel_core::MlxArray) -> i32 {
    let shape = mlxcel_core::array_shape(logits);
    let last = mlxcel_core::slice(logits, &[0, shape[1] - 1, 0], &[1, shape[1], shape[2]]);
    let argmax = mlxcel_core::argmax_last_axis(&last);
    mlxcel_core::eval(&argmax);
    mlxcel_core::item_i32(&argmax)
}

/// Greedy-decode `OCR: ` over the fixture, returning `(prompt_tokens, generated)`.
/// `generated` includes the terminating stop id when one was reached.
fn greedy(model_dir: &std::path::Path) -> (Vec<i32>, Vec<i32>) {
    let (model, tokenizer) = mlxcel::load_model(model_dir).expect("load GOT-OCR 2.0");
    let images = vec![fixture_image()];
    let mut prompt_tokens: Vec<i32> = Vec::new();

    let prepared = mlxcel::vlm_runtime::prepare_and_compute_vlm_embeddings(
        &model,
        &mut prompt_tokens,
        "OCR: ",
        &images,
        |text, add_special| {
            tokenizer
                .encode(text, add_special)
                .unwrap_or_default()
                .iter()
                .map(|&t| t as i32)
                .collect()
        },
    )
    .expect("prepare VLM embeddings")
    .expect("an image request must produce embeddings");

    let stops = LanguageModel::eos_token_ids(&model);
    assert!(
        stops.contains(&IM_END_ID),
        "the stop set must carry <|im_end|>, got {stops:?}"
    );

    let input_ids = mlxcel_core::from_slice_i32(&prompt_tokens, &[1, prompt_tokens.len() as i32]);
    let mut caches = LanguageModel::make_caches(&model);
    assert!(
        !caches.is_empty(),
        "make_caches returned no per-layer cache; the decoder would run zero layers"
    );
    let mut logits = LanguageModel::forward_with_embeddings(
        &model,
        &input_ids,
        prepared.embeddings.inputs_embeds.as_ref(),
        &mut caches,
        prepared.embeddings.attention_mask_4d.as_deref(),
    );

    let mut generated = Vec::new();
    for _ in 0..MAX_NEW_TOKENS {
        let token = argmax_last_token(&logits);
        generated.push(token);
        if stops.contains(&token) {
            break;
        }
        let next = mlxcel_core::from_slice_i32(&[token], &[1, 1]);
        logits = LanguageModel::forward(&model, &next, &mut caches, None);
    }
    (prompt_tokens, generated)
}

/// Both released key layouts load, assemble the same 287-token prompt, and
/// greedy-decode the reference continuation.
#[test]
#[ignore = "requires a real checkpoint"]
fn got_greedy_decode_matches_the_reference_on_both_layouts() {
    let mut ran = 0usize;
    for name in MODEL_DIRS {
        let dir = repo_model_dir(name);
        if !dir.join("config.json").exists() {
            eprintln!("Skipping {name}: not found at {}", dir.display());
            continue;
        }
        ran += 1;

        let (prompt_tokens, generated) = greedy(&dir);

        assert_eq!(prompt_tokens.len(), PROMPT_TOKENS, "{name} prompt length");
        assert_eq!(
            prompt_tokens.iter().filter(|&&t| t == IMG_PAD_ID).count(),
            IMAGE_TOKENS,
            "{name} image placeholder count"
        );

        // Termination is asserted, not inferred: the run must end on the
        // separator strictly before the cap, and never on <|endoftext|>.
        assert!(
            generated.len() < MAX_NEW_TOKENS,
            "{name} did not terminate within {MAX_NEW_TOKENS} tokens: {generated:?}"
        );
        assert_eq!(
            generated.last().copied(),
            Some(IM_END_ID),
            "{name} must stop on <|im_end|>"
        );
        assert!(
            !generated[..generated.len() - 1].contains(&ENDOFTEXT_ID),
            "{name} emitted <|endoftext|> mid-stream"
        );

        let content = &generated[..generated.len() - 1];
        assert_eq!(
            content, REF_GREEDY_IDS,
            "{name} greedy decode diverged from the reference"
        );

        let text = mlxcel::tokenizer::load_tokenizer(&dir)
            .expect("tokenizer")
            .decode(&content.iter().map(|&t| t as u32).collect::<Vec<_>>(), true)
            .expect("decode");
        assert_eq!(
            text.trim(),
            "GOT OCR two point zero \nrenders this page",
            "{name} transcription"
        );
        eprintln!(
            "[{name}] {} prompt tokens, transcribed {text:?}",
            prompt_tokens.len()
        );
    }

    assert!(
        ran > 0,
        "no GOT-OCR 2.0 checkpoint found; fetch mlx-community/GOT-OCR2_0-bf16"
    );
}
