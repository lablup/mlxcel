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

//! Tests for the padding-aware bidirectional and causal mask builders used by
//! the `/v1/embeddings` families.

use super::{
    array_to_vec_f32, create_bidirectional_padding_mask, create_bidirectional_window_mask,
    create_causal_mask, create_causal_padding_mask,
};
use crate::{dtype, ffi};

fn mask_i32(rows: &[&[i32]]) -> cxx::UniquePtr<ffi::MlxArray> {
    let b = rows.len() as i32;
    let l = rows[0].len() as i32;
    let flat: Vec<i32> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    ffi::from_slice_i32(&flat, &[b, l])
}

fn is_attend(v: f32) -> bool {
    v == 0.0
}

fn is_blocked(v: f32) -> bool {
    v == f32::NEG_INFINITY
}

#[test]
fn bidirectional_padding_mask_blocks_only_padding_keys() {
    // Row 0 right-padded, row 1 left-padded.
    let mask = mask_i32(&[&[1, 1, 1, 0], &[0, 0, 1, 1]]);
    let out = create_bidirectional_padding_mask(&mask);
    assert_eq!(ffi::array_shape(&out), vec![2, 1, 1, 4]);
    let values = array_to_vec_f32(&out);
    let expected_attend = [true, true, true, false, false, false, true, true];
    for (i, (&v, &attend)) in values.iter().zip(expected_attend.iter()).enumerate() {
        if attend {
            assert!(is_attend(v), "index {i}: expected attend, got {v}");
        } else {
            assert!(is_blocked(v), "index {i}: expected blocked, got {v}");
        }
    }
}

#[test]
fn causal_padding_mask_matches_create_causal_mask_when_unpadded() {
    let l = 5;
    let mask = mask_i32(&[&[1, 1, 1, 1, 1]]);
    let padded = array_to_vec_f32(&create_causal_padding_mask(&mask, 0));
    let plain = array_to_vec_f32(&create_causal_mask(l, 0));
    assert_eq!(padded.len(), plain.len());
    for (i, (p, c)) in padded.iter().zip(plain.iter()).enumerate() {
        assert_eq!(p.to_bits(), c.to_bits(), "mismatch at flat index {i}");
    }

    // With an offset the query rows are the last `total - offset` keys.
    let total = 6;
    let offset = 2;
    let mask = mask_i32(&[&[1, 1, 1, 1, 1, 1]]);
    let out = create_causal_padding_mask(&mask, offset);
    assert_eq!(ffi::array_shape(&out), vec![1, 1, total - offset, total]);
    let padded = array_to_vec_f32(&out);
    let plain = array_to_vec_f32(&create_causal_mask(total - offset, offset));
    for (i, (p, c)) in padded.iter().zip(plain.iter()).enumerate() {
        assert_eq!(
            p.to_bits(),
            c.to_bits(),
            "offset mismatch at flat index {i}"
        );
    }
}

#[test]
fn causal_padding_mask_keeps_diagonal_for_fully_padded_rows() {
    // Left padding: the first two query rows have no real causal key.
    let mask = mask_i32(&[&[0, 0, 1, 1]]);
    let out = create_causal_padding_mask(&mask, 0);
    let values = array_to_vec_f32(&out);
    let l = 4;
    let at = |q: usize, k: usize| values[q * l + k];

    // Padding rows keep exactly their diagonal column.
    for q in 0..2 {
        for k in 0..l {
            if k == q {
                assert!(is_attend(at(q, k)), "row {q} must keep its diagonal");
            } else {
                assert!(is_blocked(at(q, k)), "row {q} col {k} must be blocked");
            }
        }
    }
    // Real rows: causal over real keys only, padding keys stay blocked.
    assert!(is_blocked(at(2, 0)));
    assert!(is_blocked(at(2, 1)));
    assert!(is_attend(at(2, 2)));
    assert!(is_blocked(at(2, 3)));
    assert!(is_blocked(at(3, 0)));
    assert!(is_blocked(at(3, 1)));
    assert!(is_attend(at(3, 2)));
    assert!(is_attend(at(3, 3)));

    // Right padding: the trailing padding row still sees the real keys, so no
    // rescue is needed and the padding key stays blocked for every row.
    let mask = mask_i32(&[&[1, 1, 0]]);
    let values = array_to_vec_f32(&create_causal_padding_mask(&mask, 0));
    let at = |q: usize, k: usize| values[q * 3 + k];
    assert!(is_attend(at(2, 0)));
    assert!(is_attend(at(2, 1)));
    assert!(is_blocked(at(2, 2)));
    assert!(is_blocked(at(0, 2)));
    assert!(is_blocked(at(1, 2)));
}

#[test]
fn bidirectional_window_mask_blocks_beyond_window() {
    let mask = mask_i32(&[&[1, 1, 1, 1, 0]]);
    let window = 2;
    let out = create_bidirectional_window_mask(&mask, window);
    assert_eq!(ffi::array_shape(&out), vec![1, 1, 5, 5]);
    let values = array_to_vec_f32(&out);
    let at = |q: usize, k: usize| values[q * 5 + k];
    for q in 0..5 {
        for k in 0..5 {
            let distance = (q as i32 - k as i32).abs();
            let expected_attend = k < 4 && distance < window;
            // The padding query row (4) has real keys inside its window
            // (column 3), so nothing is rescued in this layout.
            if expected_attend {
                assert!(is_attend(at(q, k)), "({q},{k}) expected attend");
            } else {
                assert!(is_blocked(at(q, k)), "({q},{k}) expected blocked");
            }
        }
    }

    // A padding row with no real key inside its window keeps its diagonal.
    let mask = mask_i32(&[&[1, 0, 0, 0]]);
    let values = array_to_vec_f32(&create_bidirectional_window_mask(&mask, 1));
    let at = |q: usize, k: usize| values[q * 4 + k];
    for q in 1..4 {
        assert!(is_attend(at(q, q)), "row {q} must keep its diagonal");
        for k in 0..4 {
            if k != q {
                assert!(is_blocked(at(q, k)), "({q},{k}) expected blocked");
            }
        }
    }
}

#[test]
fn masks_stay_finite_after_f16_cast() {
    let mask = mask_i32(&[&[1, 1, 0], &[0, 1, 1]]);
    let masks = [
        create_bidirectional_padding_mask(&mask),
        create_causal_padding_mask(&mask, 0),
        create_bidirectional_window_mask(&mask, 2),
    ];
    for (i, m) in masks.iter().enumerate() {
        assert_eq!(ffi::array_dtype(m), dtype::FLOAT32, "mask {i} must be f32");
        for cast_dtype in [dtype::FLOAT16, dtype::BFLOAT16] {
            let cast = ffi::astype(m, cast_dtype);
            let values = array_to_vec_f32(&cast);
            assert!(
                values.iter().all(|&v| v == 0.0 || v == f32::NEG_INFINITY),
                "mask {i} cast to dtype {cast_dtype} must stay 0/-inf: {values:?}"
            );

            // Adding the cast mask to finite scores and taking the softmax
            // must not produce NaN in any row (the diagonal rescue guarantees
            // at least one finite entry per row).
            let shape = ffi::array_shape(&cast);
            let numel: usize = shape.iter().map(|&d| d as usize).product();
            let scores = ffi::astype(&ffi::from_slice_f32(&vec![0.25; numel], &shape), cast_dtype);
            let probs = ffi::softmax(&ffi::add(&scores, &cast), -1);
            let probs = array_to_vec_f32(&probs);
            assert!(
                probs.iter().all(|p| p.is_finite()),
                "mask {i} dtype {cast_dtype}: softmax produced non-finite values {probs:?}"
            );
        }
    }
}
