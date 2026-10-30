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

//! Gemma 3 under an external (VLM) prefill mask (issue #2760).
//!
//! A Gemma 3 image prefill hands the model one `[B, 1, S, S]` mask built by
//! `vision::merge` (causal plus bidirectional image blocks). transformers
//! uses that mask as is on the full-attention layers and intersects it with
//! the sliding window on the sliding layers. The external-mask branch used to
//! hand the same mask to every layer, so a sliding layer attended past its
//! window whenever the prompt was longer than the window.

use super::snapshot_prompt_cache::{VOCAB, build_wrapper_with_window, ids, to_vec_f32};
use crate::models::gemma3::external_mask_with_window_band;
use crate::vision::merge::{MultimodalMask, multimodal_attention_mask};
use mlxcel_core::MlxArray;
use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::LanguageModel;

const SEQ_BASE: u64 = 2_760_000;
const WINDOW: i32 = 4;
/// An image token id `ids` never produces (they stay below `VOCAB`), so the
/// Gemma 3 mask over the prompt is exactly causal.
const ABSENT_IMAGE_TOKEN: i32 = 1_000;

/// With no image tokens the Gemma 3 VLM mask is plain causal, so the
/// external-mask branch must reproduce the text prefill: the global layer
/// unbanded, the sliding layer banded to its 4-token window. 12 tokens put
/// the sliding layer well past the window.
#[test]
fn external_gemma3_mask_matches_text_prefill_past_the_window() {
    let tokens = ids(0..12);
    let seq_len = tokens.len() as i32;
    let prompt = mlxcel_core::from_slice_i32(&tokens, &[1, seq_len]);

    let text = build_wrapper_with_window(WINDOW);
    let seq_text = SequenceId::from_raw(SEQ_BASE + 1);
    text.prepare_sequence_state(seq_text);
    let want = text.forward_with_sequence_id(&prompt, Some(seq_text), &mut [], None);
    let want = to_vec_f32(want.as_ref().expect("text logits"));

    let vlm = build_wrapper_with_window(WINDOW);
    let seq_vlm = SequenceId::from_raw(SEQ_BASE + 2);
    vlm.prepare_sequence_state(seq_vlm);
    let embeds = vlm
        .embed_tokens(&prompt)
        .expect("Gemma 3 exposes embed_tokens");
    let ones = mlxcel_core::from_slice_i32(&vec![1; tokens.len()], &[1, seq_len]);
    let mask = multimodal_attention_mask(
        MultimodalMask::Gemma3ImageBlocks,
        ABSENT_IMAGE_TOKEN,
        &prompt,
        &ones,
    );
    let got = vlm.forward_with_embeddings_and_sequence_id(
        &prompt,
        Some(&embeds),
        Some(seq_vlm),
        &mut [],
        Some(&mask),
    );
    let got = to_vec_f32(got.as_ref().expect("VLM logits"));

    assert_eq!(got.len(), tokens.len() * VOCAB as usize, "logits shape");
    assert_eq!(got.len(), want.len(), "logits shape");
    for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        let (row, col) = (i / VOCAB as usize, i % VOCAB as usize);
        assert!(
            (g - w).abs() <= 1e-4,
            "row {row} logit {col}: external mask {g}, text prefill {w}"
        );
    }
}

/// Read an additive or boolean mask back as `allowed[q][k]` for batch row 0.
fn allowed(mask: &MlxArray) -> Vec<Vec<bool>> {
    let shape = mlxcel_core::array_shape(mask);
    let (q_len, k_len) = (
        shape[shape.len() - 2] as usize,
        shape[shape.len() - 1] as usize,
    );
    let as_f32 = mlxcel_core::astype(mask, mlxcel_core::dtype::FLOAT32);
    let values = to_vec_f32(&as_f32);
    let is_bool = mlxcel_core::array_dtype(mask) == mlxcel_core::dtype::BOOL;
    (0..q_len)
        .map(|q| {
            (0..k_len)
                .map(|k| {
                    let v = values[q * k_len + k];
                    if is_bool { v != 0.0 } else { v == 0.0 }
                })
                .collect()
        })
        .collect()
}

fn rows(spec: &[&str]) -> Vec<Vec<bool>> {
    spec.iter()
        .map(|row| row.bytes().map(|c| c == b'1').collect())
        .collect()
}

/// The sliding-layer mask is `band AND (causal OR same image block)`: an
/// image token keeps its later block-mates, and every key `window` or more
/// positions behind the query is dropped, block-mates included.
#[test]
fn sliding_layers_intersect_the_image_mask_with_the_window() {
    const IMG: i32 = 99;
    // bos, t, boi, img x3 (3..=5), eoi, t, boi, img x2 (9..=10), eoi, t
    let ids = [2, 5, 7, IMG, IMG, IMG, 8, 5, 7, IMG, IMG, 8, 5];
    let input_ids = mlxcel_core::from_slice_i32(&ids, &[1, ids.len() as i32]);
    let ones = mlxcel_core::from_slice_i32(&[1; 13], &[1, 13]);
    let full = multimodal_attention_mask(MultimodalMask::Gemma3ImageBlocks, IMG, &input_ids, &ones);
    let sliding = external_mask_with_window_band(&full, WINDOW)
        .expect("a 13-key mask is wider than a 4-token window");
    assert_eq!(
        mlxcel_core::array_shape(&sliding),
        mlxcel_core::array_shape(&full),
        "the sliding variant keeps the external mask's shape"
    );
    let expected = rows(&[
        "1000000000000",
        "1100000000000",
        "1110000000000",
        "1111110000000",
        "0111110000000",
        "0011110000000",
        "0001111000000",
        "0000111100000",
        "0000011110000",
        "0000001111100",
        "0000000111100",
        "0000000011110",
        "0000000001111",
    ]);
    assert_eq!(allowed(&sliding), expected);

    // The same band over a boolean mask stays boolean.
    let as_bool = mlxcel_core::equal(&full, &mlxcel_core::from_slice_f32(&[0.0], &[1]));
    let sliding_bool = external_mask_with_window_band(&as_bool, WINDOW)
        .expect("a 13-key mask is wider than a 4-token window");
    assert_eq!(
        mlxcel_core::array_dtype(&sliding_bool),
        mlxcel_core::dtype::BOOL
    );
    assert_eq!(allowed(&sliding_bool), expected);
}

/// A key axis that fits the window has nothing to band: the sliding layers
/// reuse the external mask.
#[test]
fn sliding_band_is_skipped_when_the_keys_fit_the_window() {
    let ids = [2, 5, 6, 5];
    let input_ids = mlxcel_core::from_slice_i32(&ids, &[1, 4]);
    let ones = mlxcel_core::from_slice_i32(&[1; 4], &[1, 4]);
    let full = multimodal_attention_mask(
        MultimodalMask::Gemma3ImageBlocks,
        ABSENT_IMAGE_TOKEN,
        &input_ids,
        &ones,
    );
    assert!(external_mask_with_window_band(&full, WINDOW).is_none());
}
