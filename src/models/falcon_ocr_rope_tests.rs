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

//! Falcon-OCR positional and mask tests.
//!
//! The expected values come from the checkpoint's own
//! `processing_falcon_ocr.py` / `attention.py`, evaluated on the same token
//! layout the loader produces.

use super::*;

fn ids() -> FalconOcrTokenIds {
    FalconOcrTokenIds {
        img_id: 227,
        image_cls_token_id: 244,
        img_end_id: 230,
        image_reg_token_ids: [245, 246, 247, 248],
    }
}

/// `[cls, reg1..4, img, img, img, img, end, t0, t1]` for a 2x2 patch grid.
fn prompt_2x2() -> Vec<i32> {
    vec![244, 245, 246, 247, 248, 227, 227, 227, 227, 230, 900, 901]
}

#[test]
fn an_image_block_collapses_onto_one_temporal_position() {
    let pos = temporal_positions(&prompt_2x2(), &ids());
    // CLS advances the counter; the four registers, the four patches and the
    // closing token all hold it. The text resumes from the next index.
    assert_eq!(pos, vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2]);
}

#[test]
fn a_text_only_prompt_gets_contiguous_positions() {
    let pos = temporal_positions(&[10, 11, 12, 13], &ids());
    assert_eq!(pos, vec![0, 1, 2, 3]);
}

/// Decode adds this delta to the KV-cache offset. With an image block the
/// temporal axis runs behind the cache axis, so the delta is negative.
#[test]
fn the_rope_delta_reconciles_cache_offset_with_temporal_position() {
    let pos = temporal_positions(&prompt_2x2(), &ids());
    let delta = rope_delta(&pos);
    assert_eq!(delta, 3 - 12);
    // The first generated token sits at cache offset == prompt length.
    assert_eq!(pos.len() as i32 + delta, pos[pos.len() - 1] + 1);
}

#[test]
fn a_text_only_prompt_has_a_zero_rope_delta() {
    assert_eq!(rope_delta(&temporal_positions(&[1, 2, 3], &ids())), 0);
}

#[test]
fn spatial_coordinates_are_row_major_and_centred() {
    let pos = spatial_positions(&prompt_2x2(), &ids(), &[(2, 2)]);
    // Square grid: both limits are sqrt(1) == 1, so the linspace is [-1, 1].
    let want: Vec<f32> = vec![
        0.0, 0.0, // cls
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // reg1..4
        -1.0, -1.0, // patch (0,0)
        -1.0, 1.0, // patch (0,1)
        1.0, -1.0, // patch (1,0)
        1.0, 1.0, // patch (1,1)
        0.0, 0.0, // end
        0.0, 0.0, 0.0, 0.0, // text
    ];
    assert_eq!(pos.len(), want.len());
    for (got, expect) in pos.iter().zip(want.iter()) {
        assert!((got - expect).abs() < 1e-6, "got {pos:?}");
    }
}

/// A wide image spreads the width axis further than the height axis: the
/// reference scales the limits by sqrt(cols/rows) and sqrt(rows/cols).
#[test]
fn a_non_square_grid_scales_the_two_axes_apart() {
    let tokens = vec![244, 245, 246, 247, 248, 227, 227, 230];
    let pos = spatial_positions(&tokens, &ids(), &[(1, 2)]);
    let ylim = (1.0f32 / 2.0).sqrt();
    let xlim = (2.0f32 / 1.0).sqrt();
    // rows == 1 makes the height linspace degenerate to its start value.
    assert!((pos[5 * 2] - -ylim).abs() < 1e-6);
    assert!((pos[5 * 2 + 1] - -xlim).abs() < 1e-6);
    assert!((pos[6 * 2] - -ylim).abs() < 1e-6);
    assert!((pos[6 * 2 + 1] - xlim).abs() < 1e-6);
}

/// Text tokens must land on (0, 0) so the golden rotation is the identity for
/// them and the whole-sequence application stays equivalent to the reference's
/// image-token-only scatter.
#[test]
fn text_tokens_get_the_identity_spatial_position() {
    let pos = spatial_positions(&prompt_2x2(), &ids(), &[(2, 2)]);
    for token_idx in [0usize, 9, 10, 11] {
        assert_eq!(pos[token_idx * 2], 0.0);
        assert_eq!(pos[token_idx * 2 + 1], 0.0);
    }
}

/// Reimplements `attention.py`'s mask composition independently so the shipped
/// builder is compared against the rule, not against itself.
fn reference_allowed(tokens: &[i32], ids: &FalconOcrTokenIds, q: usize, kv: usize) -> bool {
    // `acc_soi = cumsum(tok == soi)`, `acc_eoi = cumsum(tok == eoi)`, both
    // inclusive; `in_image = acc_soi - acc_eoi > 0`, `block = acc_soi * in_image`.
    let acc = |upto: usize, id: i32| tokens[..=upto].iter().filter(|&&t| t == id).count() as i32;
    let block = |i: usize| {
        let soi = acc(i, ids.image_cls_token_id);
        let eoi = acc(i, ids.img_end_id);
        if soi - eoi > 0 { soi } else { 0 }
    };
    let in_image = block(q) != 0 && block(q) == block(kv);
    q >= kv || in_image
}

#[test]
fn the_hybrid_mask_is_bidirectional_inside_the_image_and_causal_outside() {
    let tokens = prompt_2x2();
    let mask = build_hybrid_mask(&tokens, &ids());
    assert_eq!(
        mlxcel_core::array_shape(&mask),
        vec![1, 1, tokens.len() as i32, tokens.len() as i32]
    );

    let bytes = mlxcel_core::array_evaluated_bytes(&mask);
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let n = tokens.len();
    for q in 0..n {
        for kv in 0..n {
            let allowed = values[q * n + kv] == 0.0;
            assert_eq!(
                allowed,
                reference_allowed(&tokens, &ids(), q, kv),
                "mismatch at q={q} kv={kv}"
            );
        }
    }

    // Spot-check the property that distinguishes this model: a patch attends
    // forward to a later patch of the same image, but never to the text.
    assert_eq!(values[5 * n + 8], 0.0);
    assert!(values[5 * n + 10] < 0.0);
    // The closing token is outside the bidirectional region, so it is causal.
    assert!(values[5 * n + 9] < 0.0);
}

#[test]
fn the_hybrid_mask_is_plain_causal_without_images() {
    let tokens = vec![7, 8, 9];
    let mask = build_hybrid_mask(&tokens, &ids());
    let bytes = mlxcel_core::array_evaluated_bytes(&mask);
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    for q in 0..3 {
        for kv in 0..3 {
            assert_eq!(values[q * 3 + kv] == 0.0, q >= kv);
        }
    }
}

/// The blocked value must stay finite so a downcast to bf16/f16 cannot turn
/// the softmax into NaN.
#[test]
fn the_blocked_mask_value_is_finite() {
    assert!(mask_blocked_value().is_finite());
    assert!(mask_blocked_value() < -1e30);
}

#[test]
fn the_temporal_frequencies_match_the_reference_formula() {
    let inv = temporal_inv_freq(32, 10000.0);
    assert_eq!(inv.len(), 16);
    assert!((inv[0] - 1.0).abs() < 1e-6);
    for (j, w) in inv.iter().enumerate() {
        let want = 1.0f32 / 10000f32.powf((2 * j) as f32 / 32.0);
        assert!((w - want).abs() < 1e-6);
    }
}

/// The rotary must use interleaved pairs, not the half-split convention. A
/// quarter turn on pair 0 maps (1, 0) onto (0, 1) at indices 0 and 1; a
/// half-split implementation would move the energy to index D/2 instead.
#[test]
fn the_rotary_rotates_interleaved_pairs() {
    let dim = 8usize;
    let mut x = vec![0.0f32; dim];
    x[0] = 1.0;
    let x = mlxcel_core::from_slice_f32(&x, &[1, 1, 1, dim as i32]);

    let mut angles = vec![0.0f32; dim / 2];
    angles[0] = std::f32::consts::FRAC_PI_2;
    let theta = mlxcel_core::from_slice_f32(&angles, &[1, 1, 1, (dim / 2) as i32]);
    let (cos, sin) = (mlxcel_core::cos(&theta), mlxcel_core::sin(&theta));

    let out = rotate_interleaved(&x, &cos, &sin);
    let bytes = mlxcel_core::array_evaluated_bytes(&out);
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert!(values[0].abs() < 1e-6, "{values:?}");
    assert!((values[1] - 1.0).abs() < 1e-6, "{values:?}");
    assert!(
        values[dim / 2].abs() < 1e-6,
        "half-split leakage: {values:?}"
    );
}

/// A zero spatial position must leave the high half untouched, which is what
/// makes running the golden rotary over text tokens safe.
#[test]
fn a_zero_spatial_position_makes_the_golden_rotary_the_identity() {
    let heads = 2i32;
    let freqs_len = 2i32;
    let head_dim = 8i32;
    let freqs: Vec<f32> = (0..heads * freqs_len * 2)
        .map(|i| 0.1 * (i as f32 + 1.0))
        .collect();
    let freqs = mlxcel_core::from_slice_f32(&freqs, &[heads, freqs_len, 2]);
    let pos = mlxcel_core::from_slice_f32(&[0.0, 0.0], &[1, 1, 2]);

    let (cos_2d, sin_2d) = golden_cos_sin(&freqs, &pos);
    assert_eq!(
        mlxcel_core::array_shape(&cos_2d),
        vec![1, heads, 1, freqs_len]
    );

    let x: Vec<f32> = (0..heads * head_dim).map(|i| i as f32 + 1.0).collect();
    let x = mlxcel_core::from_slice_f32(&x, &[1, heads, 1, head_dim]);
    let (cos_1d, sin_1d) = temporal_cos_sin(&[0], &[1.0, 1.0]);
    let out = apply_3d_rotary(
        &x,
        cos_1d.as_ref().unwrap(),
        sin_1d.as_ref().unwrap(),
        Some((cos_2d.as_ref().unwrap(), sin_2d.as_ref().unwrap())),
    );

    let bytes = mlxcel_core::array_evaluated_bytes(&out);
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    // Position 0 makes both halves identity rotations.
    for (i, v) in values.iter().enumerate() {
        assert!((v - (i as f32 + 1.0)).abs() < 1e-5, "{values:?}");
    }
}
