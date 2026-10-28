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

//! Flow-match Euler scheduler.
//!
//! Two sigma schedules share one Euler stepper:
//!
//! - [`FlowMatchEulerDiscrete::dynamic`] is the dynamic exponential-shift
//!   schedule of
//!   <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen_image/scheduler.py>
//!   (`calculate_shift` over the image sequence length, optional
//!   `shift_terminal` stretch). The step accumulates in f32 and casts back to
//!   the model-output dtype.
//! - [`FlowMatchEulerDiscrete::static_shift`] is the fixed-shift schedule of
//!   `mage_flow/scheduler.py`. The step runs in the latent dtype.
//!
//! Sigmas and timesteps are computed in f32 on the CPU with the same
//! operation order as the MLX graph upstream, so the schedule can be checked
//! without a device. Only [`FlowMatchEulerDiscrete::step`] touches MLX.
//!
//! Used by: QwenImage, MingImage (dynamic); Z-Image, Mage-Flow (static shift).
//! The FLUX.2 empirical-`mu` scheduler is not shared with this file.

use anyhow::{Result, bail};
use mlxcel_core::{MlxArray, UniquePtr, dtype};

/// Number of training timesteps; timesteps are `sigma * NUM_TRAIN_TIMESTEPS`.
pub const NUM_TRAIN_TIMESTEPS: f32 = 1000.0;

/// Linear mu over the image sequence length (`calculate_shift` upstream).
#[must_use]
pub fn calculate_shift(
    image_seq_len: u32,
    base_seq_len: u32,
    max_seq_len: u32,
    base_shift: f64,
    max_shift: f64,
) -> f64 {
    let slope = (max_shift - base_shift) / f64::from(max_seq_len - base_seq_len);
    slope * f64::from(image_seq_len) + (base_shift - slope * f64::from(base_seq_len))
}

/// Parameters of the dynamic exponential-shift schedule.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicShift {
    pub base_shift: f64,
    pub max_shift: f64,
    pub base_image_seq_len: u32,
    pub max_image_seq_len: u32,
    /// Stretch the schedule so the last sigma lands on this value; `None` or
    /// `0.0` disables it. A one-step schedule is never stretched.
    pub shift_terminal: Option<f32>,
}

impl Default for DynamicShift {
    fn default() -> Self {
        Self {
            base_shift: 0.5,
            max_shift: 0.9,
            base_image_seq_len: 256,
            max_image_seq_len: 8192,
            shift_terminal: Some(0.02),
        }
    }
}

/// How [`FlowMatchEulerDiscrete::step`] combines sample and model output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StepMode {
    /// `(sample.f32 + dt * out).astype(out.dtype)`, `dt` in the output dtype.
    AccumulateF32,
    /// `sample + dt * out`, all in the sample dtype.
    SampleDtype,
}

/// A flow-match Euler schedule: `num_steps + 1` sigmas ending in 0 and
/// `num_steps` timesteps.
#[derive(Debug, Clone, PartialEq)]
pub struct FlowMatchEulerDiscrete {
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
    mode: StepMode,
}

/// `mx.linspace(1.0, 1.0 / n, n)` in f32.
fn base_sigmas(num_steps: u32) -> Vec<f32> {
    let n = num_steps as usize;
    let start = 1.0_f32;
    let stop = 1.0_f32 / num_steps as f32;
    if n == 1 {
        return vec![start];
    }
    let step = (stop - start) / (n - 1) as f32;
    let mut sigmas: Vec<f32> = (0..n).map(|i| start + i as f32 * step).collect();
    sigmas[n - 1] = stop;
    sigmas
}

impl FlowMatchEulerDiscrete {
    fn from_sigmas(sigmas: Vec<f32>, mode: StepMode) -> Self {
        let timesteps = sigmas.iter().map(|s| s * NUM_TRAIN_TIMESTEPS).collect();
        let mut with_terminal = sigmas;
        with_terminal.push(0.0);
        Self {
            sigmas: with_terminal,
            timesteps,
            mode,
        }
    }

    /// The Qwen-Image / Ming-Image schedule.
    pub fn dynamic(image_seq_len: u32, num_steps: u32, params: DynamicShift) -> Result<Self> {
        if num_steps < 1 {
            bail!("num_inference_steps must be >= 1, got {num_steps}");
        }
        if params.max_image_seq_len <= params.base_image_seq_len {
            bail!("max_image_seq_len must exceed base_image_seq_len");
        }
        let mu = calculate_shift(
            image_seq_len,
            params.base_image_seq_len,
            params.max_image_seq_len,
            params.base_shift,
            params.max_shift,
        );
        // `mx.exp(mu)` takes the Python float as an f32 scalar.
        let exp_mu = (mu as f32).exp();
        let mut sigmas: Vec<f32> = base_sigmas(num_steps)
            .into_iter()
            .map(|s| exp_mu / (exp_mu + (1.0 / s - 1.0)))
            .collect();
        // Stretch before appending the final zero sigma.
        if let Some(terminal) = params.shift_terminal
            && terminal != 0.0
            && num_steps > 1
        {
            let one_minus: Vec<f32> = sigmas.iter().map(|s| 1.0 - s).collect();
            let scale = one_minus[one_minus.len() - 1] / (1.0 - terminal);
            sigmas = one_minus.iter().map(|m| 1.0 - m / scale).collect();
        }
        Ok(Self::from_sigmas(sigmas, StepMode::AccumulateF32))
    }

    /// The fixed-shift schedule: `shift * s / (1 + (shift - 1) * s)` over the
    /// linear sigmas `s`.
    pub fn static_shift(num_steps: u32, shift: f32) -> Result<Self> {
        if num_steps < 1 {
            bail!("num_inference_steps must be >= 1, got {num_steps}");
        }
        let sigmas = base_sigmas(num_steps)
            .into_iter()
            .map(|base| shift * base / (1.0 + (shift - 1.0) * base))
            .collect();
        Ok(Self::from_sigmas(sigmas, StepMode::SampleDtype))
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

    /// Number of denoising steps.
    #[must_use]
    pub fn num_steps(&self) -> usize {
        self.timesteps.len()
    }

    /// `sigma[i + 1] - sigma[i]`, the Euler step size at `step_index`.
    pub fn dt(&self, step_index: usize) -> Result<f32> {
        if step_index >= self.num_steps() {
            bail!(
                "step_index {step_index} out of range for a {}-step schedule",
                self.num_steps()
            );
        }
        Ok(self.sigmas[step_index + 1] - self.sigmas[step_index])
    }

    /// One Euler step: `sample + dt * model_output`.
    pub fn step(
        &self,
        model_output: &MlxArray,
        step_index: usize,
        sample: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>> {
        let dt = self.dt(step_index)?;
        let dt_scalar = mlxcel_core::from_slice_f32(&[dt], &[]);
        match self.mode {
            StepMode::AccumulateF32 => {
                let out_dtype = mlxcel_core::array_dtype(model_output);
                let dt = mlxcel_core::astype(&dt_scalar, out_dtype);
                let delta = mlxcel_core::multiply(&dt, model_output);
                let sample_f32 = mlxcel_core::astype(sample, dtype::FLOAT32);
                // Upstream's f32 + bf16 promotes to f32; the CUDA build of
                // mlxcel's MLX would round the sum to bf16, so widen first.
                let delta = mlxcel_core::astype(&delta, dtype::FLOAT32);
                let sum = mlxcel_core::add(&sample_f32, &delta);
                Ok(mlxcel_core::astype(&sum, out_dtype))
            }
            StepMode::SampleDtype => {
                let sample_dtype = mlxcel_core::array_dtype(sample);
                let dt = mlxcel_core::astype(&dt_scalar, sample_dtype);
                let out = mlxcel_core::astype(model_output, sample_dtype);
                Ok(mlxcel_core::add(sample, &mlxcel_core::multiply(&dt, &out)))
            }
        }
    }
}

#[cfg(test)]
#[path = "scheduler_fixtures.rs"]
mod fixtures;

#[cfg(test)]
#[path = "scheduler_tests.rs"]
mod tests;
