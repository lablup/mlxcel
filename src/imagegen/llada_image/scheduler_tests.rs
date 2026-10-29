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
use crate::imagegen::llada_image::config::SchedulerConfig;

fn host(x: &MlxArray) -> Vec<f32> {
    let x = mlxcel_core::astype(x, dtype::FLOAT32);
    mlxcel_core::eval(&x);
    mlxcel_core::array_to_raw_bytes(&x)
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

const TURBO: SchedulerConfig = SchedulerConfig {
    use_uniform_sigmas: true,
    shift: 3.0,
    stochastic_sampling: true,
};
const BASE: SchedulerConfig = SchedulerConfig {
    use_uniform_sigmas: false,
    shift: 1.0,
    stochastic_sampling: false,
};

#[test]
fn turbo_sigmas_are_shifted_uniform_steps() {
    let s = LladaScheduler::new(TURBO, 4).unwrap();
    let got = host(s.sigmas());
    // 3s / (1 + 2s) at s = 1, 0.75, 0.5, 0.25, 0.
    let want = [1.0, 0.9, 0.75, 0.5, 0.0];
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() < 1e-7, "{got:?}");
    }
}

#[test]
fn base_schedule_matches_numpy_float64() {
    // numpy: 1 - (1 - (1 - linspace(0.001, 1, 51)[:-1] ** 1.17) ** 0.8) ** 1.1
    let sigmas = base_unshifted_sigmas(50);
    assert_eq!(sigmas.len(), 51);
    assert_eq!(sigmas[0].to_bits(), 0x3f7f_f8f0);
    assert_eq!(sigmas[1].to_bits(), 0x3f7e_9cba);
    assert_eq!(sigmas[2].to_bits(), 0x3f7c_b659);
    assert_eq!(sigmas[49].to_bits(), 0x3d5e_60c1);
    assert_eq!(sigmas[50], 0.0);
    // Shift 1 leaves them unchanged.
    let s = LladaScheduler::new(BASE, 50).unwrap();
    assert_eq!(host(s.sigmas()), sigmas);
}

#[test]
fn steps_follow_upstream_formulas() {
    let s = LladaScheduler::new(TURBO, 4).unwrap();
    let sample = mlxcel_core::from_slice_f32(&[1.0, -2.0], &[2]);
    let prediction = mlxcel_core::from_slice_f32(&[0.5, 0.25], &[2]);
    let noise = mlxcel_core::from_slice_f32(&[0.1, 0.2], &[2]);
    // sigma 0.9 -> 0.75: (1 - 0.75) * (x - 0.9 * v) + 0.75 * n.
    let got = host(&s.step(&prediction, &sample, 1, true, Some(&noise)).unwrap());
    let want = [0.25 * (1.0 - 0.45) + 0.075, 0.25 * (-2.0 - 0.225) + 0.15];
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() < 1e-6, "{got:?}");
    }
    assert!(s.step(&prediction, &sample, 1, true, None).is_err());
    // Euler: x + (sigma_next - sigma) * v.
    let got = host(&s.step(&prediction, &sample, 1, false, None).unwrap());
    let want = [1.0 - 0.15 * 0.5, -2.0 - 0.15 * 0.25];
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() < 1e-6, "{got:?}");
    }
    assert!(s.step(&prediction, &sample, 4, false, None).is_err());
    assert!(LladaScheduler::new(TURBO, 0).is_err());
}
