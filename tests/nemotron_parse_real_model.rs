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

//! Nemotron-Parse real-checkpoint greedy parity (issue #1369).
//!
//! The reference ids below come from the checkpoint's own `transformers`
//! implementation (`hf_nemotron_parse_modeling.py` with the `nvidia/C-RADIOv2-H`
//! tower code) run in fp32 on CPU with torch 2.14.0 and transformers 5.17.0,
//! on `tests/fixtures/nemotron_parse_page.png` with the default task prompt,
//! `num_beams=1`, `do_sample=False`, and `max_new_tokens=80`. The page is
//! 1240x1754, inside the 1664x2048 box, so preprocessing only pads (exact on
//! both sides) and the comparison exercises the model, not the resampler.
//!
//! The same reference run on the 8-bit export's weights dequantized back to
//! fp32 (so the quantization error is in the oracle too) produces the identical
//! sequence, so one pin serves both checkpoints.
//!
//! mlxcel runs the tower in the stored precision (f32 on the Hub checkpoint,
//! f16 on the conversions) and the decoder in f32 on the GPU, so the claim is
//! greedy token equality, not bitwise agreement. The reference's top-2 margin
//! is 0.04 to 0.07 at generated steps 2 and 60, so those are the steps where a
//! precision difference shows first (bf16 decoder activations flip both).
//!
//! The reference's 80th token is not the model's choice: `generation_config`
//! sets `forced_eos_token_id`, so `transformers` forces `</s>` on the step
//! that reaches `max_new_tokens`. mlxcel does not force it, so the comparison
//! covers the 79 tokens the model chose.
//!
//! ## Invocation
//!
//! ```bash
//! MLXCEL_NEMOTRON_PARSE_ROOT=/path/to/models \
//!   cargo test --release --test nemotron_parse_real_model -- --ignored --nocapture
//! ```
//!
//! `#[ignore]`-gated; skips with a note when no checkpoint is present.

mod common;

use std::path::PathBuf;

use mlxcel::models::NemotronParseVlmModel;
use mlxcel::models::nemotron_parse::DEFAULT_TASK_PROMPT;

const FIXTURE: &str = "tests/fixtures/nemotron_parse_page.png";

/// Greedy continuation (repetition penalty 1.0); the final `</s>` is the
/// reference's forced EOS at the token budget.
const REF_GREEDY_IDS: &[i32] = &[
    50204, 51226, 833, 62, 13716, 36993, 26070, 833, 50457, 51253, 52316, 221, 221, 50202, 51317,
    2113, 8701, 15829, 286, 4494, 10677, 6716, 299, 36993, 26070, 35, 47928, 36, 221, 592, 13142,
    343, 281, 352, 35, 30020, 8553, 6078, 74, 35, 62, 18277, 363, 281, 9831, 34, 221, 480, 286,
    14596, 343, 281, 610, 35, 7236, 310, 56, 384, 7342, 36, 50670, 51411, 52315, 221, 221, 50204,
    51467, 876, 9348, 850, 1412, 48, 243, 39, 34, 40, 41, 42, 36, 2,
];

/// The same run with `repetition_penalty=1.1` (the model card's default).
const REF_PENALIZED_IDS: &[i32] = &[
    50204, 51226, 833, 62, 13716, 36993, 26070, 833, 50457, 51253, 52316, 221, 221, 50202, 51317,
    2113, 8701, 15829, 286, 4494, 10677, 6716, 299, 36993, 26070, 35, 47928, 36, 381, 13142, 343,
    281, 352, 35, 30020, 8553, 6078, 74, 35, 62, 18277, 363, 281, 9831, 34, 312, 286, 14596, 343,
    281, 610, 35, 7236, 310, 56, 384, 7342, 36, 50670, 51411, 52315, 221, 221, 50204, 51467, 876,
    9348, 850, 1412, 48, 243, 39, 34, 40, 41, 42, 36, 43, 44, 2,
];

/// Number of leading greedy tokens the acceptance criterion pins.
const PARITY_TOKENS: usize = 64;

/// Tokens the reference model chose itself (everything before the forced EOS).
const CHOSEN_TOKENS: usize = 79;

fn model_dir(name: &str) -> Option<PathBuf> {
    if let Ok(root) = std::env::var("MLXCEL_NEMOTRON_PARSE_ROOT") {
        let p = PathBuf::from(root).join(name);
        return p.exists().then_some(p);
    }
    let p = common::repo_model_dir(name);
    p.exists().then_some(p)
}

fn page() -> image::DynamicImage {
    image::open(FIXTURE).expect("fixture page decodes")
}

/// Generated ids with the terminating EOS appended when the run hit it, so
/// they line up with the reference sequences.
fn greedy(model: &NemotronParseVlmModel, penalty: f32, max_new: usize) -> Vec<i32> {
    let seed = model.seed_ids(DEFAULT_TASK_PROMPT).unwrap();
    let pixels = model.image_processor().preprocess(&page());
    let out = model
        .model()
        .generate(&pixels, &seed, max_new, penalty, None)
        .expect("generation succeeds");
    let mut ids = out.tokens;
    if out.hit_eos {
        ids.push(model.model().config().eos_token_id);
    }
    ids
}

fn first_divergence(a: &[i32], b: &[i32]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y)
}

fn check_checkpoint(name: &str) {
    let Some(dir) = model_dir(name) else {
        eprintln!("skipping {name}: checkpoint not found");
        return;
    };
    let model = NemotronParseVlmModel::load(&dir).expect("checkpoint loads");

    // Seeding follows `transformers` generate: the default prompt already
    // starts with decoder_start, another prompt gets it prepended.
    assert_eq!(
        model.seed_ids(DEFAULT_TASK_PROMPT).unwrap(),
        vec![2, 0, 50004, 50008, 50001, 50010]
    );
    assert_eq!(
        model
            .seed_ids("<predict_bbox><predict_classes><output_markdown>")
            .unwrap(),
        vec![2, 50004, 50008, 50001]
    );

    let ids = greedy(&model, 1.0, 80);
    eprintln!("{name}: greedy ids {ids:?}");
    eprintln!(
        "{name}: first divergence from reference: {:?}",
        first_divergence(&ids, REF_GREEDY_IDS)
    );
    assert_eq!(
        &ids[..PARITY_TOKENS.min(ids.len())],
        &REF_GREEDY_IDS[..PARITY_TOKENS],
        "{name}: first {PARITY_TOKENS} greedy tokens differ from the reference"
    );
    assert_eq!(
        &ids[..CHOSEN_TOKENS],
        &REF_GREEDY_IDS[..CHOSEN_TOKENS],
        "{name}: greedy sequence differs"
    );
    let penalized = greedy(&model, 1.1, 80);
    assert_eq!(
        &penalized[..CHOSEN_TOKENS],
        &REF_PENALIZED_IDS[..CHOSEN_TOKENS],
        "{name}: repetition-penalty sequence differs"
    );

    let text = model
        .run(&page(), DEFAULT_TASK_PROMPT, 80, 1.0, None)
        .unwrap()
        .text;
    eprintln!("{name}: {text}");
    assert!(text.contains("Hello Nemotron"), "{text}");
    assert!(text.contains("<class_Title>"), "{text}");
}

/// The Hub original: f32 weights in the `encoder.*` / `decoder.*` layout.
#[test]
#[ignore = "requires the nvidia/NVIDIA-Nemotron-Parse-2.0 checkpoint"]
fn hub_checkpoint_matches_reference_greedy_tokens() {
    check_checkpoint("nvidia-nemotron-parse-2.0");
}

/// The MLX 8-bit conversion: `vision_tower.*` / `language_model.*`, dense
/// bf16 tower, 8-bit decoder.
#[test]
#[ignore = "requires mlx-community/Nemotron-Parse-2.0-8bit"]
fn eight_bit_checkpoint_matches_reference_greedy_tokens() {
    check_checkpoint("nemotron-parse-2.0-8bit");
}

/// The 4-bit conversion is lossier; it has to read the page, not match every
/// token.
#[test]
#[ignore = "requires mlx-community/Nemotron-Parse-2.0-4bit"]
fn four_bit_checkpoint_reads_the_page() {
    let Some(dir) = model_dir("nemotron-parse-2.0-4bit") else {
        eprintln!("skipping 4-bit: checkpoint not found");
        return;
    };
    let model = NemotronParseVlmModel::load(&dir).expect("checkpoint loads");
    let run = model
        .run(&page(), DEFAULT_TASK_PROMPT, 80, 1.0, None)
        .unwrap();
    eprintln!("4bit: {}", run.text);
    assert!(run.text.contains("Hello Nemotron"), "{}", run.text);
}
