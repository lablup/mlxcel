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

use super::fixtures::{DYNAMIC_CASES, STATIC_CASES};
use super::*;

/// Relative tolerance against the upstream f32 arrays. The CPU math here and
/// MLX can differ by a few ulps in `exp` and `linspace`; the schedules are
/// otherwise identical.
const REL_TOL: f32 = 5e-6;
const ABS_TOL: f32 = 1e-6;

fn assert_close(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let tol = ABS_TOL.max(w.abs() * REL_TOL);
        assert!(
            (g - w).abs() <= tol,
            "{label}[{i}]: got {g:e}, want {w:e} (tol {tol:e})"
        );
    }
}

fn dynamic_params(shift_terminal: Option<f32>) -> DynamicShift {
    DynamicShift {
        shift_terminal,
        ..DynamicShift::default()
    }
}

#[test]
fn dynamic_schedule_matches_upstream() {
    assert_eq!(DYNAMIC_CASES.len(), 16);
    for (steps, seq_len, terminal, sigmas, timesteps) in DYNAMIC_CASES {
        let s =
            FlowMatchEulerDiscrete::dynamic(*seq_len, *steps, dynamic_params(*terminal)).unwrap();
        let label = format!("dynamic steps={steps} seq={seq_len} terminal={terminal:?}");
        assert_close(&format!("{label} sigmas"), s.sigmas(), sigmas);
        assert_close(&format!("{label} timesteps"), s.timesteps(), timesteps);
    }
}

#[test]
fn static_shift_schedule_matches_upstream() {
    assert_eq!(STATIC_CASES.len(), 8);
    for (steps, shift, sigmas, timesteps) in STATIC_CASES {
        let s = FlowMatchEulerDiscrete::static_shift(*steps, *shift).unwrap();
        let label = format!("static steps={steps} shift={shift}");
        assert_close(&format!("{label} sigmas"), s.sigmas(), sigmas);
        assert_close(&format!("{label} timesteps"), s.timesteps(), timesteps);
    }
}

#[test]
fn schedules_end_in_a_zero_sigma_and_are_strictly_decreasing() {
    for steps in [1, 2, 7, 30] {
        for s in [
            FlowMatchEulerDiscrete::dynamic(1024, steps, DynamicShift::default()).unwrap(),
            FlowMatchEulerDiscrete::static_shift(steps, 6.0).unwrap(),
        ] {
            assert_eq!(s.num_steps(), steps as usize);
            assert_eq!(s.sigmas().len(), steps as usize + 1);
            assert_eq!(*s.sigmas().last().unwrap(), 0.0);
            assert!(
                s.sigmas().windows(2).all(|w| w[0] > w[1]),
                "{:?}",
                s.sigmas()
            );
        }
    }
}

#[test]
fn one_step_schedule_starts_at_pure_noise_and_is_never_stretched() {
    let s = FlowMatchEulerDiscrete::dynamic(1024, 1, DynamicShift::default()).unwrap();
    assert_eq!(s.sigmas(), &[1.0, 0.0]);
    assert_eq!(s.timesteps(), &[1000.0]);
}

#[test]
fn shift_terminal_zero_or_none_disables_the_stretch() {
    let none = FlowMatchEulerDiscrete::dynamic(1024, 4, dynamic_params(None)).unwrap();
    let zero = FlowMatchEulerDiscrete::dynamic(1024, 4, dynamic_params(Some(0.0))).unwrap();
    assert_eq!(none, zero);
    let stretched = FlowMatchEulerDiscrete::dynamic(1024, 4, dynamic_params(Some(0.02))).unwrap();
    assert!((stretched.sigmas()[3] - 0.02).abs() < 1e-5);
    assert!(none.sigmas()[3] > 0.3);
}

#[test]
fn calculate_shift_is_linear_in_sequence_length() {
    let mu = |len| calculate_shift(len, 256, 8192, 0.5, 0.9);
    assert!((mu(256) - 0.5).abs() < 1e-12);
    assert!((mu(8192) - 0.9).abs() < 1e-12);
    assert!((mu(4224) - 0.7).abs() < 1e-12);
}

#[test]
fn zero_steps_and_bad_geometry_are_rejected() {
    assert!(FlowMatchEulerDiscrete::dynamic(1024, 0, DynamicShift::default()).is_err());
    assert!(FlowMatchEulerDiscrete::static_shift(0, 6.0).is_err());
    let bad = DynamicShift {
        base_image_seq_len: 256,
        max_image_seq_len: 256,
        ..DynamicShift::default()
    };
    assert!(FlowMatchEulerDiscrete::dynamic(1024, 4, bad).is_err());
}

#[test]
fn dt_is_the_sigma_difference_and_range_checked() {
    let s = FlowMatchEulerDiscrete::static_shift(4, 6.0).unwrap();
    for i in 0..4 {
        assert_eq!(s.dt(i).unwrap(), s.sigmas()[i + 1] - s.sigmas()[i]);
        assert!(s.dt(i).unwrap() < 0.0);
    }
    assert!(s.dt(4).is_err());
}

fn vec_f32(a: &MlxArray) -> Vec<f32> {
    mlxcel_core::utils::array_to_vec_f32(a)
}

#[test]
fn dynamic_step_accumulates_in_f32_and_returns_the_output_dtype() {
    let s = FlowMatchEulerDiscrete::dynamic(1024, 4, DynamicShift::default()).unwrap();
    let sample = mlxcel_core::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    let out = mlxcel_core::from_slice_f32(&[0.5, -0.5, 1.0, 0.0], &[1, 4]);
    let out16 = mlxcel_core::astype(&out, dtype::BFLOAT16);
    let next = s.step(&out16, 0, &sample).unwrap();
    assert_eq!(mlxcel_core::array_dtype(&next), dtype::BFLOAT16);
    let dt = s.dt(0).unwrap();
    let got = vec_f32(&next);
    for (g, (x, v)) in got
        .iter()
        .zip([(1.0, 0.5), (2.0, -0.5), (3.0, 1.0), (4.0, 0.0)])
    {
        // bf16 keeps ~3 significant digits.
        assert!(
            (g - (x + dt * v)).abs() < 0.05,
            "got {g}, want {}",
            x + dt * v
        );
    }
}

#[test]
fn static_step_runs_in_the_sample_dtype() {
    let s = FlowMatchEulerDiscrete::static_shift(4, 6.0).unwrap();
    let sample = mlxcel_core::astype(
        &mlxcel_core::from_slice_f32(&[1.0, 1.0], &[2]),
        dtype::FLOAT16,
    );
    let velocity = mlxcel_core::from_slice_f32(&[2.0, -2.0], &[2]);
    let next = s.step(&velocity, 1, &sample).unwrap();
    assert_eq!(mlxcel_core::array_dtype(&next), dtype::FLOAT16);
    let dt = s.dt(1).unwrap();
    let got = vec_f32(&next);
    assert!((got[0] - (1.0 + dt * 2.0)).abs() < 5e-3);
    assert!((got[1] - (1.0 - dt * 2.0)).abs() < 5e-3);
}

#[test]
fn step_out_of_range_is_an_error() {
    let s = FlowMatchEulerDiscrete::static_shift(2, 6.0).unwrap();
    let x = mlxcel_core::from_slice_f32(&[0.0], &[1]);
    assert!(s.step(&x, 2, &x).is_err());
}
