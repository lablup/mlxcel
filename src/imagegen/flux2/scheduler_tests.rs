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
use crate::imagegen::flux2::scheduler_fixtures::CASES;

/// Every table entry is within a few f32 ulps of upstream (MLX `exp` and
/// `linspace` run on the same backend family; most entries match exactly).
fn assert_ulps(label: &str, got: &[f32], want_bits: &[u32]) {
    assert_eq!(got.len(), want_bits.len(), "{label}: length");
    for (i, (g, w)) in got.iter().zip(want_bits).enumerate() {
        let w = f32::from_bits(*w);
        let ulps = (g.to_bits() as i64 - w.to_bits() as i64).abs();
        assert!(
            ulps <= 4,
            "{label}[{i}]: got {g:e}, want {w:e} ({ulps} ulps)"
        );
    }
}

#[test]
fn schedules_match_upstream_for_both_mu_branches() {
    assert_eq!(CASES.len(), 4);
    for (steps, seq_len, timesteps, sigmas) in CASES {
        let s = Flux2Scheduler::new(*seq_len, *steps).unwrap();
        let label = format!("steps={steps} seq={seq_len}");
        assert_eq!(s.len(), *steps as usize);
        assert_ulps(&format!("{label} timesteps"), s.timesteps(), timesteps);
        assert_ulps(&format!("{label} sigmas"), s.sigmas(), sigmas);
        assert_eq!(*s.sigmas().last().unwrap(), 0.0);
    }
}

#[test]
fn mu_branches_at_4300_tokens() {
    // Above the branch mu ignores the step count; below it does not.
    assert_eq!(empirical_mu(5000, 4), empirical_mu(5000, 50));
    assert_ne!(empirical_mu(1024, 4), empirical_mu(1024, 50));
    assert!((empirical_mu(1024, 4) - 2.030_689_707_949_945_5).abs() < 1e-12);
    assert!((empirical_mu(5000, 4) - 1.303_016_66).abs() < 1e-12);
}

#[test]
fn step_casts_dt_and_noise_to_the_latent_dtype() {
    let s = Flux2Scheduler::new(1024, 4).unwrap();
    let latents = mlxcel_core::astype(
        &mlxcel_core::full_f32(&[1, 2, 2], 1.0, mlxcel_core::dtype::FLOAT32),
        mlxcel_core::dtype::BFLOAT16,
    );
    let noise = mlxcel_core::full_f32(&[1, 2, 2], 2.0, mlxcel_core::dtype::FLOAT32);
    let out = s.step(&noise, 0, &latents).unwrap();
    assert_eq!(mlxcel_core::array_dtype(&out), mlxcel_core::dtype::BFLOAT16);
    let dt = s.sigmas()[1] - s.sigmas()[0];
    let got = mlxcel_core::utils::array_to_vec_f32(&out);
    assert!((got[0] - (1.0 + 2.0 * dt)).abs() < 1e-2, "{got:?}");
    assert!(s.step(&noise, 4, &latents).is_err());
    assert!(Flux2Scheduler::new(1024, 0).is_err());
}
