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

use super::{AudioLinear, causal_valid_mask};
use mlxcel_core::weights::WeightMap;

fn identity_weight(dim: usize) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let mut data = vec![0.0f32; dim * dim];
    for i in 0..dim {
        data[i * dim + i] = 1.0;
    }
    mlxcel_core::from_slice_f32(&data, &[dim as i32, dim as i32])
}

fn to_vec_f32(arr: &mlxcel_core::MlxArray) -> Vec<f32> {
    let arr = mlxcel_core::astype(arr, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&arr);
    mlxcel_core::array_to_raw_bytes(&arr)
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// The Gemma 4 audio checkpoints ship finite input/output clamp bounds for
/// every ClippableLinear, and the reference implementation applies them at
/// inference. Skipping them decorrelates the Conformer stack (issue #782):
/// the per-block error compounds from ~8% relative RMS after block 0 to ~95%
/// after block 11 on a real clip. This test locks the clamp semantics in.
#[test]
fn audio_linear_applies_checkpoint_clip_bounds() {
    let mut weights = WeightMap::new();
    weights.insert(
        "clip.linear.weight".to_string(),
        mlxcel_core::copy(&identity_weight(4)),
    );
    weights.insert(
        "clip.input_min".to_string(),
        mlxcel_core::from_slice_f32(&[-1.0], &[1]),
    );
    weights.insert(
        "clip.input_max".to_string(),
        mlxcel_core::from_slice_f32(&[1.0], &[1]),
    );
    weights.insert(
        "clip.output_min".to_string(),
        mlxcel_core::from_slice_f32(&[-0.5], &[1]),
    );
    weights.insert(
        "clip.output_max".to_string(),
        mlxcel_core::from_slice_f32(&[0.5], &[1]),
    );

    let layer = AudioLinear::from_weights(&weights, "clip", 64, 4).expect("clippable linear");
    let x = mlxcel_core::from_slice_f32(&[2.0, -3.0, 0.3, 0.9], &[1, 4]);
    let y = to_vec_f32(&layer.forward(&x));

    // Input clamps to [-1, 1] -> identity matmul -> output clamps to [-0.5, 0.5]:
    // 2.0 -> 1.0 -> 0.5; -3.0 -> -1.0 -> -0.5; 0.3 passes through; 0.9 -> 0.5.
    let expected = [0.5, -0.5, 0.3, 0.5];
    for (actual, expected) in y.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "clipped output {y:?} != expected {expected:?}"
        );
    }
}

/// Without bound tensors in the checkpoint the layer must behave as a plain
/// linear (clamping disabled), so non-clippable checkpoints keep working.
#[test]
fn audio_linear_without_bounds_is_plain_linear() {
    let mut weights = WeightMap::new();
    weights.insert(
        "plain.linear.weight".to_string(),
        mlxcel_core::copy(&identity_weight(4)),
    );

    let layer = AudioLinear::from_weights(&weights, "plain", 64, 4).expect("plain linear");
    let x = mlxcel_core::from_slice_f32(&[2.0, -3.0, 0.3, 0.9], &[1, 4]);
    let y = to_vec_f32(&layer.forward(&x));

    let expected = [2.0, -3.0, 0.3, 0.9];
    for (actual, expected) in y.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "passthrough output {y:?} != expected {expected:?}"
        );
    }
}

// ── Local attention mask (#2379) ─────────────────────────────────────────────

/// `(chunk, attention_context_left, attention_context_right)` sets from the
/// upstream test (Blaizzy/mlx-vlm#2385). `(12, 13, 0)` is the shipped Gemma 4
/// audio config.
const MASK_CASES: [(usize, usize, usize); 7] = [
    (12, 13, 0),
    (12, 1, 0),
    (8, 6, 4),
    (4, 2, 2),
    (6, 4, 1),
    (5, 1, 5),
    (10, 10, 3),
];

/// The reference predicate in plain Rust: query `q` of a chunk may see context
/// slot `c` when the query-minus-key distance is a past offset within
/// `left - 1` or a future offset within `right`.
fn reference_mask(chunk: usize, left: usize, right: usize) -> Vec<bool> {
    let max_past = left.saturating_sub(1) as i64;
    let max_future = right as i64;
    let ctx = chunk as i64 + max_past + max_future;
    let mut out = Vec::with_capacity(chunk * ctx as usize);
    for q in 0..chunk as i64 {
        for c in 0..ctx {
            let dist = q + max_past - c;
            out.push((0..max_past).contains(&dist) || (dist < 0 && -dist < max_future));
        }
    }
    out
}

/// The construction the fix replaced: `tril(ones(ctx, chunk)).T *
/// tril(ones(chunk, ctx), k = max_past + max_future)`, element by element.
fn previous_mask(chunk: usize, left: usize, right: usize) -> Vec<bool> {
    let max_past = left.saturating_sub(1) as i64;
    let max_future = right as i64;
    let ctx = chunk as i64 + max_past + max_future;
    let mut out = Vec::with_capacity(chunk * ctx as usize);
    for q in 0..chunk as i64 {
        for c in 0..ctx {
            // lower^T[q, c] = tril(ones(ctx, chunk))[c, q] = (q <= c);
            // upper[q, c] = (c <= q + max_past + max_future).
            out.push((q..=q + max_past + max_future).contains(&c));
        }
    }
    out
}

fn mask_values(chunk: usize, left: usize, right: usize) -> (Vec<i32>, Vec<bool>) {
    let mask = causal_valid_mask(chunk, left, right);
    mlxcel_core::eval(&mask);
    let shape = mlxcel_core::array_shape(&mask);
    let as_i32 = mlxcel_core::astype(&mask, mlxcel_core::dtype::INT32);
    mlxcel_core::eval(&as_i32);
    let values = mlxcel_core::array_to_raw_bytes(&as_i32)
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()) != 0)
        .collect();
    (shape, values)
}

#[test]
fn causal_valid_mask_matches_the_reference_predicate() {
    for (chunk, left, right) in MASK_CASES {
        let (shape, got) = mask_values(chunk, left, right);
        let ctx = chunk + left.saturating_sub(1) + right;
        assert_eq!(
            shape,
            vec![chunk as i32, ctx as i32],
            "{chunk},{left},{right}"
        );
        assert_eq!(
            got,
            reference_mask(chunk, left, right),
            "mask ({chunk},{left},{right})"
        );
    }
}

/// The predicate test above would pass vacuously if the old construction
/// already agreed with the reference. It does not on the shipped config: it
/// admits `dist == max_past`, one key further back than the reference.
#[test]
fn previous_mask_admits_one_key_too_far_back() {
    let (chunk, left, right) = (12, 13, 0);
    let old = previous_mask(chunk, left, right);
    let reference = reference_mask(chunk, left, right);
    assert_ne!(old, reference);
    let ctx = chunk + left - 1 + right;
    for q in 0..chunk {
        // Slot `q` is the key `max_past` positions back from query `q`.
        assert!(
            old[q * ctx + q],
            "old mask admits dist == max_past at q={q}"
        );
        assert!(!reference[q * ctx + q], "reference rejects it at q={q}");
    }
}
