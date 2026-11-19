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

#[test]
fn linspace_hits_both_endpoints() {
    let values = torch_linspace(1.0, 0.25, 10);
    assert_eq!(values.len(), 10);
    assert!((values[0] - 1.0).abs() < 1e-6, "{values:?}");
    assert!((values[9] - 0.25).abs() < 1e-6, "{values:?}");
}

#[test]
fn the_schedule_decreases_strictly_and_ends_at_zero() {
    let sigmas = sigmas_for(50, 5.0);
    assert_eq!(sigmas.last().copied(), Some(0.0));
    assert!(sigmas.len() > 2, "{sigmas:?}");
    for pair in sigmas.windows(2) {
        assert!(pair[0] > pair[1], "sigmas must decrease strictly: {pair:?}");
    }
}

#[test]
fn every_sigma_sits_on_the_training_grid() {
    let mut training = grid(TRAINING_STEPS, 1.0, TRAINING_STEPS);
    training.reverse();
    for sigma in sigmas_for(20, 5.0) {
        if sigma == 0.0 {
            continue;
        }
        assert!(
            training.iter().any(|t| (t - sigma).abs() < 1e-6),
            "sigma {sigma} is not on the training grid"
        );
    }
}

#[test]
fn timesteps_are_sigmas_scaled_by_the_grid() {
    let sigmas = vec![1.0, 0.5, 0.0];
    assert_eq!(timesteps_for(&sigmas), vec![1000.0, 500.0]);
}

#[test]
fn zero_steps_yield_an_empty_schedule_instead_of_underflowing() {
    assert!(torch_linspace(1.0, 0.0, 0).is_empty());
    assert_eq!(sigmas_for(0, 5.0), vec![0.0]);
}
