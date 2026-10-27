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

use super::*;

#[test]
fn bucket_selection_matches_upstream() {
    let b = &DEFAULT_SEQ_LEN_BUCKETS;
    assert_eq!(pick_bucket(1, b, 512), 32);
    assert_eq!(pick_bucket(32, b, 512), 32);
    assert_eq!(pick_bucket(33, b, 512), 64);
    assert_eq!(pick_bucket(600, b, 512), 512);
    // A cap below the prompt length pads (and truncates) to the cap.
    assert_eq!(pick_bucket(400, b, 300), 300);
    assert_eq!(pick_bucket(100, b, 64), 64);
}

#[test]
fn text_ids_are_zero_except_the_token_axis() {
    let ids = prepare_text_ids(2, 3);
    assert_eq!(mlxcel_core::array_shape(&ids), [2, 3, 4]);
    let flat = mlxcel_core::utils::array_to_vec_f32(&mlxcel_core::astype(
        &ids,
        mlxcel_core::dtype::FLOAT32,
    ));
    let row: Vec<f32> = vec![0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 2.];
    assert_eq!(flat[..12], row[..]);
    assert_eq!(flat[12..], row[..]);
}

fn tokenizer_dir() -> Option<std::path::PathBuf> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("models/mlx/flux.2-klein-4b-bf16/tokenizer");
    dir.join("tokenizer.json").is_file().then_some(dir)
}

/// Ground truth from upstream `Flux2Tokenizer` (transformers fast tokenizer)
/// on `black-forest-labs/FLUX.2-klein-4B`'s `tokenizer/`. Skips when the
/// checkpoint is absent.
#[test]
fn chat_template_and_tokens_match_upstream_on_the_checkpoint() {
    let Some(dir) = tokenizer_dir() else {
        eprintln!("skipping: models/mlx/flux.2-klein-4b-bf16/tokenizer is absent");
        return;
    };
    let tok = Flux2Tokenizer::load(&dir).unwrap();
    let prompt = "A cinematic portrait of a glass bonsai tree on a moonlit desk";
    assert_eq!(
        tok.format_prompt(prompt).unwrap(),
        format!(
            "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
    );
    let want: [i32; 26] = [
        151644, 872, 198, 32, 64665, 33033, 315, 264, 8991, 81034, 2143, 4916, 389, 264, 17788,
        31635, 18010, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271,
    ];
    assert_eq!(tok.count_tokens(prompt).unwrap(), 26);
    let t = tok.tokenize(prompt, 32).unwrap();
    assert_eq!(t.input_ids[..26], want);
    assert_eq!(t.input_ids[26..], [151643; 6]);
    assert_eq!(t.attention_mask[..26], [1; 26]);
    assert_eq!(t.attention_mask[26..], [0; 6]);
    let truncated = tok.tokenize(prompt, 8).unwrap();
    assert_eq!(truncated.input_ids, want[..8]);
    assert_eq!(
        tok.format_prompt(NEGATIVE_PROMPT).unwrap(),
        "<|im_start|>user\n <|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
}
