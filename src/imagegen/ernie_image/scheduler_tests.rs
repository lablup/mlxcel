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

//! Sigma literals printed by upstream `ErnieImageFlowMatchScheduler` (mlx 0.32.4
//! CUDA, mlx-vlm 0.7.8) as the shortest f32 round-trip strings.

use super::*;

const UPSTREAM_SIGMAS_8: [f32; 9] = [
    1.0, 0.9655172, 0.9230769, 0.8695652, 0.8, 0.7058824, 0.5714286, 0.36363637, 0.0,
];

const UPSTREAM_SIGMAS_50: [f32; 51] = [
    1.0, 0.9949239, 0.9896907, 0.9842932, 0.9787234, 0.972973, 0.96703297, 0.96089387, 0.9545455,
    0.9479769, 0.9411765, 0.93413174, 0.9268293, 0.91925466, 0.9113924, 0.90322584, 0.8947369,
    0.885906, 0.87671226, 0.86713284, 0.8571428, 0.84671533, 0.83582085, 0.8244275, 0.8125, 0.8,
    0.78688526, 0.77310926, 0.7586207, 0.74336284, 0.72727275, 0.7102804, 0.6923078, 0.6732673,
    0.6530612, 0.6315789, 0.6086956, 0.58426964, 0.55813956, 0.53012055, 0.5, 0.4675325, 0.4324325,
    0.39436617, 0.3529412, 0.30769238, 0.25806445, 0.20338982, 0.14285721, 0.07547163, 0.0,
];

fn assert_matches(steps: u32, expected: &[f32]) {
    let s = ErnieScheduler::new(steps).unwrap();
    assert_eq!(s.sigmas().len(), expected.len());
    assert_eq!(s.num_steps(), steps as usize);
    for (i, (got, want)) in s.sigmas().iter().zip(expected).enumerate() {
        assert!(
            got.to_bits() == want.to_bits(),
            "{steps} steps, sigma {i}: got {got:e}, upstream {want:e}"
        );
    }
    for (t, s) in s.timesteps().iter().zip(s.sigmas()) {
        assert_eq!(*t, s * NUM_TRAIN_TIMESTEPS);
    }
}

#[test]
fn eight_step_sigmas_match_upstream() {
    assert_matches(8, &UPSTREAM_SIGMAS_8);
}

#[test]
fn fifty_step_sigmas_match_upstream() {
    assert_matches(50, &UPSTREAM_SIGMAS_50);
}

#[test]
fn a_zero_step_schedule_is_rejected() {
    assert!(ErnieScheduler::new(0).is_err());
}

#[test]
fn the_step_accumulates_in_f32_and_keeps_the_sample_dtype() {
    let s = ErnieScheduler::new(8).unwrap();
    let sample = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&[1.0, -2.0], &[2]),
        dtype::BFLOAT16,
    );
    let out = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&[0.5, 0.25], &[2]),
        dtype::BFLOAT16,
    );
    let next = s.step(&out, 0, &sample).unwrap();
    assert_eq!(mlxcel_core::array_dtype(&next), dtype::BFLOAT16);
    let next = mlxcel_core::astype(&next, dtype::FLOAT32);
    mlxcel_core::eval(&next);
    let dt = UPSTREAM_SIGMAS_8[1] - UPSTREAM_SIGMAS_8[0];
    let want = [1.0 + dt * 0.5, -2.0 + dt * 0.25];
    let got: Vec<f32> = (0..2)
        .map(|i| mlxcel_core::item_f32(&mlxcel_core::slice(&next, &[i], &[i + 1])))
        .collect();
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() < 1e-2, "got {g}, want {w}");
    }
    assert!(s.step(&out, 8, &sample).is_err());
}
