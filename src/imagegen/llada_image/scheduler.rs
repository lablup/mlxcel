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

//! The LLaDA-Image flow schedule.
//!
//! A port of `LLaDAImageScheduler.sigmas` and `.step` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/llada_image/scheduler.py>.
//! Turbo uses `linspace(1, 0, steps + 1)` in f32 with shift 3 and re-noises
//! every step; the base checkpoint builds a nonlinear schedule in f64 on the
//! host (`1 - (1 - (1 - s^1.17)^0.8)^1.1` over `linspace(0.001, 1, steps +
//! 1)[:-1]`), rounds it to f32, appends 0 and takes Euler steps. The shift
//! `shift * s / (1 + (shift - 1) * s)` runs on the device with upstream's op
//! order, so every sigma matches bit for bit.

use anyhow::{Result, bail};
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::config::SchedulerConfig;

fn f32_scalar(value: f32) -> UniquePtr<MlxArray> {
    mlxcel_core::full_f32(&[], value, dtype::FLOAT32)
}

/// The base checkpoint's unshifted schedule: `steps` f64 values rounded to
/// f32, then 0 (numpy's `linspace` is `i * step + start`).
#[must_use]
pub fn base_unshifted_sigmas(steps: u32) -> Vec<f32> {
    let n = f64::from(steps);
    let step = (1.0 - 0.001) / n;
    let mut sigmas: Vec<f32> = (0..steps)
        .map(|i| {
            let s = f64::from(i) * step + 0.001;
            (1.0 - (1.0 - (1.0 - s.powf(1.17)).powf(0.8)).powf(1.1)) as f32
        })
        .collect();
    sigmas.push(0.0);
    sigmas
}

/// The schedule for one run.
pub struct LladaScheduler {
    config: SchedulerConfig,
    /// `[steps + 1]` f32, from 1 (or near it) down to 0.
    sigmas: UniquePtr<MlxArray>,
    steps: u32,
}

impl LladaScheduler {
    pub fn new(config: SchedulerConfig, steps: u32) -> Result<Self> {
        if steps < 1 {
            bail!("steps must be at least 1");
        }
        let n = i32::try_from(steps)? + 1;
        let unshifted = if config.use_uniform_sigmas {
            mlxcel_core::linspace(1.0, 0.0, n, dtype::FLOAT32)
        } else {
            mlxcel_core::from_slice_f32(&base_unshifted_sigmas(steps), &[n])
        };
        // `self.shift * sigma / (1.0 + (self.shift - 1.0) * sigma)`: Python
        // evaluates `shift - 1.0` in f64; both scalars are weak f32 operands.
        let shift = f64::from(config.shift);
        let numerator = mlxcel_core::multiply(&f32_scalar(config.shift), &unshifted);
        let denominator = mlxcel_core::add(
            &f32_scalar(1.0),
            &mlxcel_core::multiply(&f32_scalar((shift - 1.0) as f32), &unshifted),
        );
        let sigmas = mlxcel_core::divide(&numerator, &denominator);
        mlxcel_core::eval(&sigmas);
        Ok(Self {
            config,
            sigmas,
            steps,
        })
    }

    #[must_use]
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    #[must_use]
    pub fn steps(&self) -> u32 {
        self.steps
    }

    /// All `steps + 1` sigmas.
    #[must_use]
    pub fn sigmas(&self) -> &MlxArray {
        &self.sigmas
    }

    /// `sigmas[index]` as a 0-d f32 array.
    #[must_use]
    pub fn sigma(&self, index: u32) -> UniquePtr<MlxArray> {
        let i = index as i32;
        mlxcel_core::reshape(&mlxcel_core::slice(&self.sigmas, &[i], &[i + 1]), &[])
    }

    /// `sigmas[index : index + 1].astype(dt)`: the transformer's timestep.
    #[must_use]
    pub fn timestep(&self, index: u32, dt: i32) -> UniquePtr<MlxArray> {
        let i = index as i32;
        mlxcel_core::astype(&mlxcel_core::slice(&self.sigmas, &[i], &[i + 1]), dt)
    }

    /// Upstream `step` from `index` to `index + 1` in f32. `noise` is the
    /// stochastic re-noise draw; it is required exactly when the schedule
    /// samples stochastically.
    pub fn step(
        &self,
        prediction: &MlxArray,
        sample: &MlxArray,
        index: u32,
        stochastic: bool,
        noise: Option<&MlxArray>,
    ) -> Result<UniquePtr<MlxArray>> {
        if index >= self.steps {
            bail!(
                "scheduler step {index} is past the {}-step schedule",
                self.steps
            );
        }
        let sample = mlxcel_core::astype(sample, dtype::FLOAT32);
        let prediction = mlxcel_core::astype(prediction, dtype::FLOAT32);
        let sigma = self.sigma(index);
        let sigma_next = self.sigma(index + 1);
        if stochastic {
            let Some(noise) = noise else {
                bail!("stochastic sampling needs a noise draw");
            };
            let clean = mlxcel_core::subtract(&sample, &mlxcel_core::multiply(&sigma, &prediction));
            let keep = mlxcel_core::subtract(&f32_scalar(1.0), &sigma_next);
            Ok(mlxcel_core::add(
                &mlxcel_core::multiply(&keep, &clean),
                &mlxcel_core::multiply(&sigma_next, noise),
            ))
        } else {
            let delta = mlxcel_core::subtract(&sigma_next, &sigma);
            Ok(mlxcel_core::add(
                &sample,
                &mlxcel_core::multiply(&delta, &prediction),
            ))
        }
    }
}

#[cfg(test)]
#[path = "scheduler_tests.rs"]
mod tests;
