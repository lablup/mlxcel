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

use mlxcel_core::UniquePtr;

use super::*;

#[test]
fn grid_scales_the_long_side_to_512() {
    assert_eq!(vq_grid(1024, 1024).unwrap(), (32, 32));
    assert_eq!(vq_grid(512, 512).unwrap(), (32, 32));
    // height 512, width 2048: 8 rows by 32 columns, 256 tokens.
    assert_eq!(vq_grid(512, 2048).unwrap(), (8, 32));
    assert_eq!(vq_grid(256, 384).unwrap(), (16, 24));
    assert!(vq_grid(16, 4096).is_err());
}

#[test]
fn transfer_schedule_spreads_the_remainder_first() {
    assert_eq!(transfer_schedule(32, 8), vec![4; 8]);
    assert_eq!(transfer_schedule(32, 5), vec![7, 7, 6, 6, 6]);
    assert_eq!(transfer_schedule(32, 32), vec![1; 32]);
}

fn bf16_scores(values: &[f32]) -> UniquePtr<MlxArray> {
    mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(values, &[1, values.len() as i32]),
        dtype::BFLOAT16,
    )
}

#[test]
fn threshold_reveals_everything_above_it() {
    let inf = f32::NEG_INFINITY;
    let s = bf16_scores(&[0.99, 0.5, 0.97, inf, 0.96, 0.98]);
    // Four clear 0.95 (bf16 0.949), at least the count of 3.
    let got = transfer_mask_top_count(&s, THRESHOLD, 3, 5);
    assert_eq!(got, vec![true, false, true, false, true, true]);
}

#[test]
fn too_few_above_threshold_reveal_the_top_count() {
    let inf = f32::NEG_INFINITY;
    let s = bf16_scores(&[0.2, 0.9, inf, 0.4, 0.96, 0.1]);
    let got = transfer_mask_top_count(&s, THRESHOLD, 3, 5);
    assert_eq!(got, vec![false, true, false, true, true, false]);
    // Never more than the masked positions.
    let s = bf16_scores(&[inf, 0.3, inf, inf]);
    assert_eq!(
        transfer_mask_top_count(&s, THRESHOLD, 4, 1),
        vec![false, true, false, false]
    );
}

#[test]
fn codebook_ids_reject_text_tokens() {
    let tokens = VqTokens {
        tokens: vec![1, 2, IMAGE_TOKEN_OFFSET + 5, IMAGE_TOKEN_OFFSET, 7],
        prompt_length: 2,
        gen_length: 2,
    };
    assert_eq!(tokens.codebook_ids(16384).unwrap(), vec![5, 0]);
    let bad = VqTokens {
        tokens: vec![1, 2, 3972, IMAGE_TOKEN_OFFSET],
        prompt_length: 2,
        gen_length: 2,
    };
    let err = bad.codebook_ids(16384).unwrap_err();
    assert!(format!("{err}").contains("outside the SigVQ codebook"));
}

#[test]
fn prior_prompts_past_the_bound_are_rejected() {
    assert!(check_vq_prompt_length(MAX_VQ_PROMPT_TOKENS).is_ok());
    let err = check_vq_prompt_length(MAX_VQ_PROMPT_TOKENS + 1).unwrap_err();
    assert!(format!("{err:#}").contains("at most 2048"), "{err:#}");
}

/// `vq_prompt_ids` reports rows * columns VQ tokens, and the conditional and
/// unconditional prompts share the `<|image|>` rows/columns tail (the final
/// `<|/image|>` is dropped). Reads the tokenizer only.
#[test]
#[ignore = "needs models/mlx/llada-image-turbo (tokenizer only)"]
fn prompt_ids_carry_the_grid_and_count() {
    let root = std::env::var_os("MLXCEL_LLADA_IMAGE_TURBO")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("models/mlx/llada-image-turbo")
        });
    let tokenizer =
        crate::tokenizer::load_tokenizer(&root.join("tokenizer")).expect("checkpoint tokenizer");
    let (square_cond, square_uncond, square_count) =
        vq_prompt_ids(&tokenizer, "a red panda", 1024, 1024).unwrap();
    assert_eq!(square_count, 32 * 32);
    // height 512, width 2048: 8 rows by 32 columns.
    let (wide_cond, wide_uncond, wide_count) =
        vq_prompt_ids(&tokenizer, "a red panda", 512, 2048).unwrap();
    assert_eq!(wide_count, 8 * 32);
    // Same token count (one reserved token per axis); only the ids differ.
    assert_eq!(square_cond.len(), wide_cond.len());
    assert_eq!(square_uncond.len(), wide_uncond.len());
    assert_ne!(square_cond, wide_cond);
    let tail = |ids: &[i32]| ids[ids.len() - 4..].to_vec();
    assert_eq!(tail(&square_cond), tail(&square_uncond));
    assert_eq!(tail(&wide_cond), tail(&wide_uncond));
    assert_ne!(tail(&square_cond), tail(&wide_cond));
}
