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

//! Checkpoint-free parity tests for the shared single-step short-conv decode
//! fast path (`build_conv_decode_weight` + `short_conv_decode_step`, issue
//! #748 / #752).
//!
//! Each test builds a channel- and tap-asymmetric depthwise conv weight so a
//! transpose or axis bug cannot pass by symmetry, then asserts the elementwise
//! decode step is numerically identical to the stride-1 / no-pad / dilation-1 /
//! `groups == channels` `conv1d` it replaces. Kernel sizes 3 (LFM2) and 4 (the
//! mamba2 / falcon-h1 / granite-4.0-h / mamba / jamba / plamo2 / nemotron-h SSM
//! conv width) are both covered, in f32 and bf16 (the dtype the fast path
//! actually runs in on real checkpoints off Metal). The per-family test files
//! add the family-specific post-conv shaping (bias, SiLU); the numeric core of
//! the conv is pinned here.

use super::conv_decode::{build_conv_decode_weight, short_conv_decode_step};
use mlxcel_core::dtype;
use mlxcel_core::utils::silu;

/// Build an asymmetric `[channels, kernel, 1]` depthwise conv weight whose taps
/// differ per channel and per position (no symmetry to hide a transpose bug).
/// Values are kept in roughly `[-1, 1]` so that, like the LFM2 tests, bf16
/// rounding stays small and the parity tolerances remain meaningful (large
/// synthetic magnitudes would make bf16 error swamp any real logic bug).
fn asymmetric_conv_weight(channels: usize, kernel: usize) -> Vec<f32> {
    let mut data = Vec::with_capacity(channels * kernel);
    for c in 0..channels {
        for k in 0..kernel {
            // Distinct, non-monotone value per (channel, tap), bounded in ~[-1, 1).
            let v = (((c * 7 + k * 13 + 3) % 17) as f32) / 17.0 - 0.5;
            data.push(if (c + k) % 2 == 0 { v } else { -v });
        }
    }
    data
}

/// Deterministic asymmetric `[1, kernel, channels]` padded activation window,
/// also bounded in ~`[-1, 1]`.
fn asymmetric_padded(channels: usize, kernel: usize) -> Vec<f32> {
    let mut data = Vec::with_capacity(channels * kernel);
    for k in 0..kernel {
        for c in 0..channels {
            let v = (((k * 5 + c * 11 + 1) % 13) as f32) / 13.0 - 0.5;
            data.push(if (k + c) % 3 == 0 { -v } else { v });
        }
    }
    data
}

/// f32 parity: the elementwise decode step equals `conv1d` for a length-1
/// output, for the given channel count and kernel width.
fn assert_decode_matches_conv1d_f32(channels: usize, kernel: usize) {
    let ch = channels as i32;
    let k = kernel as i32;

    let weight_data = asymmetric_conv_weight(channels, kernel);
    let conv_weight = mlxcel_core::from_slice_f32(&weight_data, &[ch, k, 1]);

    let padded_data = asymmetric_padded(channels, kernel);
    let padded = mlxcel_core::from_slice_f32(&padded_data, &[1, k, ch]);

    let reference = mlxcel_core::conv1d(&padded, &conv_weight, 1, 0, 1, ch);
    assert_eq!(mlxcel_core::array_shape(&reference), vec![1, 1, ch]);

    let decode_weight = build_conv_decode_weight(&conv_weight);
    assert_eq!(mlxcel_core::array_shape(&decode_weight), vec![1, k, ch]);

    let elementwise = short_conv_decode_step(&padded, &decode_weight, dtype::FLOAT32);
    assert_eq!(mlxcel_core::array_shape(&elementwise), vec![1, 1, ch]);

    let diff = mlxcel_core::subtract(&reference, &elementwise);
    let max_abs = mlxcel_core::item_f32(&mlxcel_core::max_all(&mlxcel_core::abs(&diff)));
    assert!(
        max_abs < 1e-5,
        "f32 decode short-conv diverged from conv1d (channels={channels}, kernel={kernel}): max|diff| = {max_abs}"
    );
}

/// bf16 parity: same as the f32 case but in the dtype the fast path runs in on
/// real (bf16-activation) checkpoints. Values are built in f32 and cast to bf16
/// for both the reference and the fast path, isolating dtype rounding from
/// construction differences.
fn assert_decode_matches_conv1d_bf16(channels: usize, kernel: usize) {
    let ch = channels as i32;
    let k = kernel as i32;

    let weight_data = asymmetric_conv_weight(channels, kernel);
    let conv_weight_f32 = mlxcel_core::from_slice_f32(&weight_data, &[ch, k, 1]);
    let conv_weight = mlxcel_core::astype(&conv_weight_f32, dtype::BFLOAT16);

    let padded_data = asymmetric_padded(channels, kernel);
    let padded_f32 = mlxcel_core::from_slice_f32(&padded_data, &[1, k, ch]);
    let padded = mlxcel_core::astype(&padded_f32, dtype::BFLOAT16);

    let reference = mlxcel_core::conv1d(&padded, &conv_weight, 1, 0, 1, ch);
    assert_eq!(mlxcel_core::array_dtype(&reference), dtype::BFLOAT16);

    let decode_weight = build_conv_decode_weight(&conv_weight);
    let elementwise = short_conv_decode_step(&padded, &decode_weight, dtype::BFLOAT16);
    assert_eq!(mlxcel_core::array_shape(&elementwise), vec![1, 1, ch]);

    let diff = mlxcel_core::subtract(
        &mlxcel_core::astype(&reference, dtype::FLOAT32),
        &mlxcel_core::astype(&elementwise, dtype::FLOAT32),
    );
    let max_abs = mlxcel_core::item_f32(&mlxcel_core::max_all(&mlxcel_core::abs(&diff)));
    // bf16 carries ~3 decimal digits; the assertion takes a max over all
    // channels, and the expected magnitude of a max over more independent
    // bf16 rounding errors grows roughly like sqrt(channels) (an
    // order-statistic bound), so the tolerance scales with channel count too.
    let tol = 3e-2 * (channels as f32).sqrt();
    assert!(
        max_abs < tol,
        "bf16 decode short-conv diverged from bf16 conv1d (channels={channels}, kernel={kernel}): max|diff| = {max_abs} (tol {tol})"
    );
}

#[test]
fn conv_decode_matches_conv1d_kernel3_f32() {
    // LFM2 `conv_L_cache = 3` width, many channels.
    assert_decode_matches_conv1d_f32(48, 3);
}

#[test]
fn conv_decode_matches_conv1d_kernel4_f32() {
    // The mamba2 / falcon-h1 / granite-4.0-h / mamba / jamba / plamo2 /
    // nemotron-h SSM conv width (`conv_kernel = 4`).
    assert_decode_matches_conv1d_f32(64, 4);
}

#[test]
fn conv_decode_matches_conv1d_kernel3_bf16() {
    assert_decode_matches_conv1d_bf16(48, 3);
}

#[test]
fn conv_decode_matches_conv1d_kernel4_bf16() {
    assert_decode_matches_conv1d_bf16(64, 4);
}

#[test]
fn conv_decode_with_bias_and_silu_matches_conv1d_bf16() {
    // The SSM families post-process the conv as `silu(conv1d(x) + bias)`
    // (mamba2 / falcon-h1 / granite-4.0-h / mamba / jamba / nemotron-h). Assert
    // the whole shaped step is identical whether the conv came from the fast
    // decode path or from `conv1d`, so the adaptation is safe end-to-end and not
    // only at the bare-conv boundary.
    let channels = 64usize;
    let kernel = 4usize;
    let ch = channels as i32;
    let k = kernel as i32;

    let weight_data = asymmetric_conv_weight(channels, kernel);
    let conv_weight = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&weight_data, &[ch, k, 1]),
        dtype::BFLOAT16,
    );

    // Asymmetric per-channel bias, bounded in ~[-0.5, 0.5].
    let bias_data: Vec<f32> = (0..channels)
        .map(|c| (((c * 3 + 2) % 7) as f32) / 7.0 - 0.5 + if c % 2 == 0 { 0.1 } else { -0.1 })
        .collect();
    let bias = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&bias_data, &[ch]),
        dtype::BFLOAT16,
    );
    let bias_row = mlxcel_core::reshape(&bias, &[1, 1, -1]);

    let padded = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&asymmetric_padded(channels, kernel), &[1, k, ch]),
        dtype::BFLOAT16,
    );

    // Reference: conv1d -> + bias -> silu.
    let ref_conv = mlxcel_core::conv1d(&padded, &conv_weight, 1, 0, 1, ch);
    let ref_out = silu(&mlxcel_core::add(&ref_conv, &bias_row));

    // Fast path: decode step -> + bias -> silu.
    let decode_weight = build_conv_decode_weight(&conv_weight);
    let fast_conv = short_conv_decode_step(&padded, &decode_weight, dtype::BFLOAT16);
    let fast_out = silu(&mlxcel_core::add(&fast_conv, &bias_row));

    let diff = mlxcel_core::subtract(
        &mlxcel_core::astype(&ref_out, dtype::FLOAT32),
        &mlxcel_core::astype(&fast_out, dtype::FLOAT32),
    );
    let max_abs = mlxcel_core::item_f32(&mlxcel_core::max_all(&mlxcel_core::abs(&diff)));
    assert!(
        max_abs < 3e-2,
        "silu(bias + conv) diverged between fast decode and conv1d: max|diff| = {max_abs}"
    );
}
