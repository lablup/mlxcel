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

//! Convolutions cuDNN has no execution plan for still evaluate (issue #2739).
//!
//! With `MLX_ENABLE_TF32=0` (the test binary's default, see
//! `pin_full_precision_f32_matmuls_for_tests`), MLX deselects cuDNN's
//! tensor-core plans for f32 convolutions. On an NVIDIA GB10 (sm_121) that
//! leaves no plan for a dilated f32 conv1d, and upstream threw
//! "No execution plans support the graph" at eval, which broke Kokoro's
//! dilated generator resblocks. The CUDA `conv.cpp` overlay falls back to the
//! gemm convolution instead. The shapes below are Kokoro's; on other backends
//! the test checks the same values.
//!
//!   cargo test -p mlxcel-core --lib conv_no_plan_tests

use super::*;
use crate::streams::{DefaultDeviceGuard, lock_default_device};

fn ramp(shape: &[i32], scale: f32) -> UniquePtr<MlxArray> {
    let n: i32 = shape.iter().product();
    let values: Vec<f32> = (0..n).map(|i| ((i as f32) * scale).sin()).collect();
    from_slice_f32(&values, shape)
}

fn values(x: &MlxArray) -> Vec<f32> {
    let x = contiguous(x, false);
    eval(&x);
    crate::utils::array_to_vec_f32(&x)
}

/// `conv1d` on the default device against the same call on the CPU stream.
fn assert_matches_cpu(input: &[i32], weight: &[i32], padding: i32, dilation: i32) {
    let x = ramp(input, 0.37);
    let w = ramp(weight, 0.11);
    let device = values(&conv1d(&x, &w, 1, padding, dilation, 1));
    let reference = {
        let _cpu = DefaultDeviceGuard::cpu();
        values(&conv1d(&x, &w, 1, padding, dilation, 1))
    };
    assert_eq!(device.len(), reference.len());
    // Each output sums K * C_in products of values in [-1, 1].
    let bound = 1e-5 * (weight[1] * weight[2]) as f32;
    let worst = device
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst <= bound,
        "dilated conv1d {input:?} * {weight:?} (d={dilation}) differs from CPU by {worst}"
    );
}

#[test]
fn dilated_f32_conv1d_without_cudnn_plan_evaluates() {
    let _device_lock = lock_default_device();
    // Kokoro generator resblock convs1: kernel 3 dilation 3, kernel 11
    // dilation 5, at the 256- and 128-channel stages.
    assert_matches_cpu(&[1, 1200, 256], &[256, 3, 256], 3, 3);
    assert_matches_cpu(&[1, 1200, 256], &[256, 11, 256], 25, 5);
    assert_matches_cpu(&[1, 7200, 128], &[128, 11, 128], 25, 5);
}
