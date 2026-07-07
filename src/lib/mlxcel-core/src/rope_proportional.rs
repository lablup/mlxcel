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

//! Proportional RoPE helper for Gemma 4 full-attention layers.
//!
//! The Gemma 4 reference (https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/rope_utils.py) defines a
//! `ProportionalRoPE` variant whose frequency exponents are normalized by the
//! FULL head dimension (`dims`) rather than the rotated-only slice
//! (`rotated_dims = int(dims * partial_rotary_factor // 2) * 2`). Only the
//! first `rotated_dims / 2` HF-style pairs actually rotate; the remaining
//! entries pass through unchanged.
//!
//! This module mirrors upstream's full-head `mx.fast.rope` call by padding the
//! custom frequency table with `inf` entries for the non-rotated tail. MLX's
//! `fast::rope` computes `inv_freqs = 1 / freqs` internally, so those tail
//! entries become zero phase and pass through unchanged.
//!
//! Used by: Gemma4

use crate::{
    MlxArray, UniquePtr, array_shape, compiled_proportional_rope, concatenate, copy,
    fast_rope_with_freqs, from_slice_f32, slice,
};

/// Compute the frequency table for proportional RoPE.
///
/// Returns a length-`head_dim / 2` tensor where
/// `freqs[i] = factor * base^(2 * i / head_dim)` for the rotated prefix and
/// `inf` for the non-rotated tail.
///
/// Note the denominator is the FULL `head_dim`, not `rotated_dims` — this is
/// the defining feature of "proportional" RoPE.
///
/// `rotated_dims = 2 * floor(partial_rotary_factor * head_dim / 2)`.
/// Returns `None` when `rotated_dims <= 0`, in which case
/// [`apply_proportional_rope`] treats the input as identity.
///
/// # Panics
///
/// Panics if `head_dim` is not a positive even integer, if `base <= 0`, or
/// if `factor <= 0`.
pub fn compute_proportional_rope_freqs(
    head_dim: i32,
    partial_rotary_factor: f32,
    base: f32,
    factor: f32,
) -> Option<UniquePtr<MlxArray>> {
    assert!(
        head_dim > 0 && head_dim % 2 == 0,
        "compute_proportional_rope_freqs: head_dim must be positive and even, got {head_dim}"
    );
    assert!(
        partial_rotary_factor.is_finite() && partial_rotary_factor >= 0.0,
        "compute_proportional_rope_freqs: partial_rotary_factor must be finite and \
         non-negative, got {partial_rotary_factor}"
    );
    assert!(
        base.is_finite() && base > 0.0,
        "compute_proportional_rope_freqs: base must be finite and positive, got {base}"
    );
    assert!(
        factor.is_finite() && factor > 0.0,
        "compute_proportional_rope_freqs: factor must be finite and positive, got {factor}"
    );

    let rope_angles = ((partial_rotary_factor as f64) * (head_dim as f64) / 2.0).floor() as i32;
    let rope_angles = rope_angles.clamp(0, head_dim / 2);
    if rope_angles <= 0 {
        return None;
    }

    let half = head_dim / 2;
    let mut freqs = Vec::with_capacity(half as usize);
    for i in 0..rope_angles {
        // exponent = (2 * i) / head_dim — denominator is FULL head_dim,
        // matching upstream's `arange(0, rotated_dims, 2) / dims` expression.
        let exponent = (2 * i) as f32 / head_dim as f32;
        freqs.push(factor * base.powf(exponent));
    }
    freqs.resize(half as usize, f32::INFINITY);
    Some(from_slice_f32(&freqs, &[half]))
}

/// Apply proportional RoPE to `x` along the last dimension.
///
/// The last dimension of `x` must equal `head_dim`. Only the first
/// `rotated_dims` entries are rotated — the remainder passes through
/// unchanged via the `inf` frequency tail — mirroring the upstream Python
/// reference (`mlx_lm.models.rope_utils.ProportionalRoPE.__call__`).
///
/// `freqs` must have been produced by [`compute_proportional_rope_freqs`]
/// with the same `head_dim` and `partial_rotary_factor`. `None` means no
/// rotation (identity), matching the reference behaviour when
/// `rotated_dims <= 0`.
pub fn apply_proportional_rope(
    x: &MlxArray,
    head_dim: i32,
    partial_rotary_factor: f32,
    offset: i32,
    freqs: Option<&MlxArray>,
) -> UniquePtr<MlxArray> {
    let rope_angles = ((partial_rotary_factor as f64) * (head_dim as f64) / 2.0).floor() as i32;
    let rotated_dims = 2 * rope_angles.max(0);

    if rotated_dims == 0 || freqs.is_none() {
        return copy(x);
    }

    let freqs = freqs.expect("freqs must be Some when rotated_dims > 0");
    let shape = array_shape(x);
    let rank = shape.len() as i32;
    assert!(rank >= 1, "apply_proportional_rope: x must have rank >= 1");
    let last_axis = rank - 1;
    let half = head_dim / 2;
    let last_dim = shape[last_axis as usize];
    assert!(
        last_dim >= head_dim,
        "apply_proportional_rope: last dim ({last_dim}) must be >= head_dim ({head_dim})"
    );

    // Fast path: when `last_dim == head_dim` (the standard Gemma 4 Q/K shape)
    // use the same full-head RoPE call as mlx-lm. The `inf` tail in `freqs`
    // produces zero phase for non-rotated pairs.
    if last_dim == head_dim {
        return compiled_proportional_rope(x, freqs, head_dim, rotated_dims, offset);
    }

    // head = x[..., :head_dim]; tail = x[..., head_dim:]
    let start_full = vec![0_i32; rank as usize];
    let mut stop_full = vec![i32::MAX; rank as usize];
    stop_full[last_axis as usize] = head_dim;
    let head = slice(x, &start_full, &stop_full);

    // left = head[..., :half]; right = head[..., half:]
    let mut stop_half = stop_full.clone();
    stop_half[last_axis as usize] = half;
    let left = slice(&head, &start_full, &stop_half);

    let mut start_half = start_full.clone();
    start_half[last_axis as usize] = half;
    let mut stop_head = stop_full.clone();
    stop_head[last_axis as usize] = head_dim;
    let right = slice(&head, &start_half, &stop_head);

    // rotated = concat([left[..., :rotated_dims/2], right[..., :rotated_dims/2]], -1)
    let rot_half = rotated_dims / 2;
    let mut stop_rot = stop_full.clone();
    stop_rot[last_axis as usize] = rot_half;
    let left_first = slice(&left, &start_full, &stop_rot);
    let right_first = slice(&right, &start_full, &stop_rot);
    let rotated = concatenate(&left_first, &right_first, last_axis);

    // Apply mx.fast.rope on the packed [..., rotated_dims] tensor.
    let rotated_out = fast_rope_with_freqs(&rotated, rotated_dims, false, 1.0, offset, freqs);

    // Split rotated_out back into its two halves.
    let mut start_rot = start_full.clone();
    let mut stop_rot_first = stop_full.clone();
    stop_rot_first[last_axis as usize] = rot_half;
    let rot_a = slice(&rotated_out, &start_rot, &stop_rot_first);
    start_rot[last_axis as usize] = rot_half;
    let mut stop_rot_full = stop_full.clone();
    stop_rot_full[last_axis as usize] = rotated_dims;
    let rot_b = slice(&rotated_out, &start_rot, &stop_rot_full);

    // Rebuild the two halves: replace the first rot_half entries in each.
    // left_new  = concat([rot_a, left [..., rot_half:half]], -1)
    // right_new = concat([rot_b, right[..., rot_half:half]], -1)
    let mut start_rest = start_full.clone();
    start_rest[last_axis as usize] = rot_half;
    let mut stop_rest = stop_full.clone();
    stop_rest[last_axis as usize] = half;
    let left_rest = slice(&left, &start_rest, &stop_rest);
    let right_rest = slice(&right, &start_rest, &stop_rest);
    let left_new = concatenate(&rot_a, &left_rest, last_axis);
    let right_new = concatenate(&rot_b, &right_rest, last_axis);

    let head_new = concatenate(&left_new, &right_new, last_axis);

    if last_dim == head_dim {
        return head_new;
    }

    // Preserve any elements past head_dim (e.g., deepseek-style padded
    // head_dims). In practice Gemma 4 never hits this branch.
    let mut start_tail = start_full.clone();
    start_tail[last_axis as usize] = head_dim;
    let stop_tail = stop_full.clone();
    let tail = slice(x, &start_tail, &stop_tail);
    concatenate(&head_new, &tail, last_axis)
}

/// Apply proportional RoPE with one independent offset per batch row.
///
/// This mirrors [`crate::fast_rope_batched`] for Gemma 4 full-attention
/// layers that use the proportional-frequency variant. A uniform-offset
/// batch collapses back to the scalar fast path; mixed offsets slice the
/// leading batch dimension, apply scalar proportional RoPE per row, and
/// concatenate the rows again.
///
/// Used by: Gemma 4 MTP assistant batched drafter when rows accept
/// different numbers of speculative tokens and therefore need per-row
/// frozen RoPE anchors.
pub fn apply_proportional_rope_batched(
    x: &MlxArray,
    head_dim: i32,
    partial_rotary_factor: f32,
    offsets: &[i32],
    freqs: Option<&MlxArray>,
) -> UniquePtr<MlxArray> {
    let shape = array_shape(x);
    let batch = shape.first().copied().unwrap_or(0).max(0) as usize;
    assert_eq!(
        batch,
        offsets.len(),
        "apply_proportional_rope_batched expected {batch} offsets, got {}",
        offsets.len()
    );

    if batch == 0 {
        return copy(x);
    }
    if batch == 1 {
        return apply_proportional_rope(x, head_dim, partial_rotary_factor, offsets[0], freqs);
    }

    let first_offset = offsets[0];
    if offsets[1..].iter().all(|&offset| offset == first_offset) {
        return apply_proportional_rope(x, head_dim, partial_rotary_factor, first_offset, freqs);
    }

    let rank = shape.len();
    let mut begin = vec![0; rank];
    let mut end = vec![i32::MAX; rank];
    end[0] = 1;

    let first = slice(x, &begin, &end);
    let mut result =
        apply_proportional_rope(&first, head_dim, partial_rotary_factor, offsets[0], freqs);

    for (batch_idx, &offset) in offsets.iter().enumerate().skip(1) {
        begin[0] = batch_idx as i32;
        end[0] = batch_idx as i32 + 1;
        let chunk = slice(x, &begin, &end);
        let chunk = apply_proportional_rope(&chunk, head_dim, partial_rotary_factor, offset, freqs);
        result = concatenate(&result, &chunk, 0);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{allclose, array_to_raw_bytes, astype, dtype, eval, item_bool};

    /// The compiled full-head fast path (`compiled_proportional_rope`, taken
    /// when `last_dim == head_dim`) must be numerically identical to the
    /// mlx-vlm / mlx-lm reference `ProportionalRoPE.__call__`, which packs the
    /// first `rotated_dims/2` slots of each head half into a `[.., rotated_dims]`
    /// tensor, applies `mx.fast.rope` with the finite freqs there, and stitches
    /// the halves back. Pins that the inf-tail full-head shortcut equals the
    /// packed reference algorithm for the exact Gemma-4 global-layer config
    /// (`head_dim = 512`, `partial_rotary_factor = 0.25`, `theta = 1e6`) at
    /// several positions, including well past the rotated slice's longest
    /// wavelength. Issue #686: a regression here would corrupt the 8
    /// full-attention layers' long-range phase and reproduce the position-
    /// dependent quality collapse that teacher-forced perplexity exposed but
    /// generation and self-consistency A/Bs never did.
    #[test]
    fn proportional_rope_full_head_matches_packed_reference() {
        let head_dim = 512_i32;
        let prf = 0.25_f32; // rope_angles = 64, rotated_dims = 128.
        let base = 1_000_000.0_f32;
        let rot_half = 64_i32; // rotated_dims / 2
        let half = head_dim / 2; // 256

        let freqs = compute_proportional_rope_freqs(head_dim, prf, base, 1.0)
            .expect("freqs must exist for prf=0.25");
        // The reference packs only the finite (rotated) frequencies.
        let freqs_rot = slice(&freqs, &[0], &[rot_half]);
        eval(&freqs);
        eval(&freqs_rot);

        // Deterministic non-trivial input [1, 2 heads, 3 positions, head_dim].
        let heads = 2_i32;
        let positions = 3_i32;
        let total = (heads * positions * head_dim) as usize;
        let data: Vec<f32> = (0..total)
            .map(|i| ((i as f32) * 0.013).sin() + 0.25)
            .collect();
        let x = from_slice_f32(&data, &[1, heads, positions, head_dim]);

        // Reference packed algorithm (mlx-vlm ProportionalRoPE, tail empty).
        let reference = |offset: i32| -> UniquePtr<MlxArray> {
            let left = slice(&x, &[0, 0, 0, 0], &[1, heads, positions, half]);
            let right = slice(&x, &[0, 0, 0, half], &[1, heads, positions, head_dim]);
            let left_first = slice(&left, &[0, 0, 0, 0], &[1, heads, positions, rot_half]);
            let right_first = slice(&right, &[0, 0, 0, 0], &[1, heads, positions, rot_half]);
            let packed = concatenate(&left_first, &right_first, 3);
            let rotated =
                fast_rope_with_freqs(&packed, 2 * rot_half, false, 1.0, offset, &freqs_rot);
            let rot_a = slice(&rotated, &[0, 0, 0, 0], &[1, heads, positions, rot_half]);
            let rot_b = slice(
                &rotated,
                &[0, 0, 0, rot_half],
                &[1, heads, positions, 2 * rot_half],
            );
            let left_rest = slice(&left, &[0, 0, 0, rot_half], &[1, heads, positions, half]);
            let right_rest = slice(&right, &[0, 0, 0, rot_half], &[1, heads, positions, half]);
            let left_new = concatenate(&rot_a, &left_rest, 3);
            let right_new = concatenate(&rot_b, &right_rest, 3);
            concatenate(&left_new, &right_new, 3)
        };

        for offset in [0_i32, 5, 130, 500] {
            let got = apply_proportional_rope(&x, head_dim, prf, offset, freqs.as_ref());
            let expected = reference(offset);
            let close = allclose(&got, &expected, 1e-5, 1e-5);
            eval(&close);
            assert!(
                item_bool(&close),
                "full-head proportional rope diverged from the packed reference at offset {offset}"
            );
        }
    }

    #[test]
    fn proportional_rope_freqs_match_python_formula() {
        // Gemma 4 full-attention layer: head_dim=256, partial_rotary_factor=0.25,
        // rope_theta=1_000_000 -> rope_angles = 32, followed by an `inf` tail.
        let head_dim = 256_i32;
        let prf = 0.25_f32;
        let base = 1_000_000.0_f32;
        let factor = 1.0_f32;
        let freqs = compute_proportional_rope_freqs(head_dim, prf, base, factor)
            .expect("proportional freqs must exist for prf=0.25");
        eval(&freqs);

        assert_eq!(
            array_shape(&freqs),
            vec![128],
            "freqs length must equal head_dim / 2"
        );

        let freqs_f32 = astype(&freqs, dtype::FLOAT32);
        eval(&freqs_f32);
        let bytes = array_to_raw_bytes(&freqs_f32);
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(values.len(), 128);

        for (i, &got) in values.iter().take(32).enumerate() {
            // Expected: base^(2*i / head_dim)
            let expected = base.powf((2 * i) as f32 / head_dim as f32);
            let rel = (got - expected).abs() / expected.max(1.0);
            assert!(
                rel < 1e-4,
                "freqs[{i}] expected {expected}, got {got} (rel {rel})"
            );
        }
        for (i, &got) in values.iter().enumerate().skip(32) {
            assert!(
                got.is_infinite() && got.is_sign_positive(),
                "freqs[{i}] must be +inf"
            );
        }
    }

    #[test]
    fn proportional_rope_zero_partial_factor_is_identity() {
        // partial_rotary_factor = 0 ⇒ rotated_dims = 0 ⇒ no-op.
        let head_dim = 128_i32;
        let freqs = compute_proportional_rope_freqs(head_dim, 0.0, 10_000.0, 1.0);
        assert!(freqs.is_none(), "freqs must be None when rope_angles = 0");

        let x = crate::ones(&[1, 2, 3, head_dim], dtype::FLOAT32);
        let out = apply_proportional_rope(&x, head_dim, 0.0, 0, None);
        eval(&out);
        let close = allclose(&out, &x, 1e-6, 1e-6);
        eval(&close);
        assert!(item_bool(&close), "zero-rotation path must be identity");
    }

    #[test]
    fn proportional_rope_nontrivial_offset_produces_nonidentity_rotation() {
        // Smoke test: for a non-zero offset and a non-trivial input, the
        // rotated output must differ from the input (at least on the rotated
        // portion). This catches accidental no-op regressions.
        let head_dim = 64_i32;
        let prf = 0.5_f32;
        let base = 10_000.0_f32;
        let total = (2 * 3) * head_dim as usize;
        let data: Vec<f32> = (0..total).map(|i| (i as f32 * 0.03).cos()).collect();
        let x = from_slice_f32(&data, &[1, 2, 3, head_dim]);

        let freqs =
            compute_proportional_rope_freqs(head_dim, prf, base, 1.0).expect("freqs must exist");
        let rotated = apply_proportional_rope(&x, head_dim, prf, 5, freqs.as_ref());
        eval(&rotated);
        let close = allclose(&rotated, &x, 1e-5, 1e-5);
        eval(&close);
        assert!(
            !item_bool(&close),
            "non-zero offset must rotate the first rotated_dims slots"
        );
    }

    #[test]
    fn proportional_rope_preserves_non_rotated_tail() {
        // Values in the non-rotated tail of each half (indices
        // [rotated_dims/2, head_dim/2) of `left` and `right`) must pass
        // through unchanged.
        let head_dim = 64_i32;
        let prf = 0.5_f32; // rope_angles = 16, rotated_dims = 32, rot_half = 16.
        let base = 10_000.0_f32;
        let half = (head_dim / 2) as usize;
        let rot_half = 16_usize;

        let total = head_dim as usize;
        let data: Vec<f32> = (0..total).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let x = from_slice_f32(&data, &[1, 1, 1, head_dim]);
        let freqs = compute_proportional_rope_freqs(head_dim, prf, base, 1.0).unwrap();
        let out = apply_proportional_rope(&x, head_dim, prf, 7, freqs.as_ref());
        eval(&out);

        let out_bytes = array_to_raw_bytes(&astype(&out, dtype::FLOAT32));
        let out_values: Vec<f32> = out_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // left tail: indices [rot_half, half) in the head — in flat layout
        // those map to positions [rot_half, half).
        for i in rot_half..half {
            let expected = data[i];
            let got = out_values[i];
            assert!(
                (got - expected).abs() < 1e-6,
                "left tail position {i}: expected {expected}, got {got}"
            );
        }
        // right tail: same sub-range but offset by `half` (start of right
        // half) → positions [half + rot_half, half + half) = [48, 64).
        for i in (half + rot_half)..(2 * half) {
            let expected = data[i];
            let got = out_values[i];
            assert!(
                (got - expected).abs() < 1e-6,
                "right tail position {i}: expected {expected}, got {got}"
            );
        }
    }
}
