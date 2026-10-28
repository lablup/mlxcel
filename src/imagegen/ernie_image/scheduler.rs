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

//! `ErnieImageFlowMatchScheduler`.
//!
//! A port of
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/ernie_image/scheduler.py>:
//! linear sigmas `linspace(1, 0, n + 1)[:-1]` under the static shift
//! `shift * s / (1 + (shift - 1) * s)` (shift 4.0, the checkpoint's
//! `use_dynamic_shifting: false`), timesteps `sigma * 1000`, and an Euler step
//! accumulated in f32 and cast back to the sample dtype.
//!
//! It stays local rather than reusing
//! [`crate::imagegen::scheduler::FlowMatchEulerDiscrete::static_shift`]: that
//! schedule builds its sigmas as `linspace(1, 1 / n, n)`, which rounds
//! differently in f32 (50 steps differ in the last bit), and its step runs in
//! the sample dtype instead of f32.

use anyhow::{Result, bail};
use mlxcel_core::{MlxArray, UniquePtr, dtype};

/// ERNIE-Image's fixed scheduler shift.
pub const SHIFT: f32 = 4.0;
/// Number of training timesteps; timesteps are `sigma * NUM_TRAIN_TIMESTEPS`.
pub const NUM_TRAIN_TIMESTEPS: f32 = 1000.0;

/// A static-shift flow-match schedule: `num_steps + 1` sigmas ending in 0.
#[derive(Debug, Clone, PartialEq)]
pub struct ErnieScheduler {
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
}

impl ErnieScheduler {
    /// The schedule for `num_steps` denoising steps.
    pub fn new(num_steps: u32) -> Result<Self> {
        Self::with_shift(num_steps, SHIFT)
    }

    /// The schedule at an explicit `shift`.
    pub fn with_shift(num_steps: u32, shift: f32) -> Result<Self> {
        if num_steps < 1 {
            bail!("num_inference_steps must be >= 1, got {num_steps}");
        }
        // `mx.linspace(1.0, 0.0, n + 1)` in f32, which MLX evaluates as
        // `(1 - t) * start + t * stop` with `t = i / (num - 1)`. The
        // `start + i * step` form differs by one ulp at some indices. With
        // start 1 and stop 0 that is exactly `1 - t`.
        let mut sigmas: Vec<f32> = (0..num_steps)
            .map(|i| {
                let t = i as f32 / num_steps as f32;
                let s = 1.0_f32 - t;
                shift * s / (1.0 + (shift - 1.0) * s)
            })
            .collect();
        let timesteps = sigmas.iter().map(|s| s * NUM_TRAIN_TIMESTEPS).collect();
        sigmas.push(0.0);
        Ok(Self { sigmas, timesteps })
    }

    /// `num_steps + 1` sigmas, the last one 0.
    #[must_use]
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    /// `num_steps` timesteps (`sigma * 1000`).
    #[must_use]
    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    #[must_use]
    pub fn num_steps(&self) -> usize {
        self.timesteps.len()
    }

    /// One Euler step: `(sample.f32 + (sigma[i + 1] - sigma[i]) *
    /// model_output.f32).astype(sample.dtype)`.
    pub fn step(
        &self,
        model_output: &MlxArray,
        step_index: usize,
        sample: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>> {
        if step_index >= self.num_steps() {
            bail!(
                "step_index {step_index} out of range for a {}-step schedule",
                self.num_steps()
            );
        }
        let delta = self.sigmas[step_index + 1] - self.sigmas[step_index];
        let sample_dtype = mlxcel_core::array_dtype(sample);
        let delta = mlxcel_core::from_slice_f32(&[delta], &[]);
        let out = mlxcel_core::astype(model_output, dtype::FLOAT32);
        let sum = mlxcel_core::add(
            &mlxcel_core::astype(sample, dtype::FLOAT32),
            &mlxcel_core::multiply(&delta, &out),
        );
        Ok(mlxcel_core::astype(&sum, sample_dtype))
    }
}

#[cfg(test)]
#[path = "scheduler_tests.rs"]
mod tests;
