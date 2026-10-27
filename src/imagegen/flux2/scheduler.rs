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

//! The FLUX.2 flow-match Euler scheduler.
//!
//! A port of `FlowMatchEulerDiscreteScheduler` in
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/flux2/scheduler.py>,
//! line for line in f32 MLX ops so the tables carry upstream's bits: an
//! empirical `mu` fitted on the image sequence length and step count (one
//! line above 4300 tokens, an interpolation between the 10- and 200-step fits
//! below), and the exponential time shift `exp(mu) / (exp(mu) + (1/t - 1))`.
//! This is not the Qwen-Image scheduler in [`crate::imagegen::scheduler`].
//!
//! Used by: Flux2, Bonsai.

use anyhow::{Result, bail};
use mlxcel_core::{MlxArray, UniquePtr, dtype};

/// Above this image sequence length `mu` depends on the length alone.
pub const MU_BRANCH_SEQ_LEN: i32 = 4300;
const NUM_TRAIN_TIMESTEPS: f32 = 1000.0;

/// Upstream `_compute_empirical_mu`, in f64 as Python computes it.
#[must_use]
pub fn empirical_mu(image_seq_len: i32, num_steps: u32) -> f64 {
    let (a1, b1) = (8.738_095_24e-05_f64, 1.898_333_33_f64);
    let (a2, b2) = (0.000_169_27_f64, 0.456_666_66_f64);
    let len = f64::from(image_seq_len);
    if image_seq_len > MU_BRANCH_SEQ_LEN {
        return a2 * len + b2;
    }
    let m_200 = a2 * len + b2;
    let m_10 = a1 * len + b1;
    let a = (m_200 - m_10) / 190.0;
    let b = m_200 - 200.0 * a;
    a * f64::from(num_steps) + b
}

fn scalar(value: f32) -> UniquePtr<MlxArray> {
    mlxcel_core::full_f32(&[1], value, dtype::FLOAT32)
}

/// Sigma and timestep tables for one run.
#[derive(Debug, Clone)]
pub struct Flux2Scheduler {
    timesteps: Vec<f32>,
    /// `steps + 1` entries; the last is 0.
    sigmas: Vec<f32>,
}

impl Flux2Scheduler {
    /// Build the schedule for `steps` steps over `image_seq_len` tokens
    /// (`(height / 16) * (width / 16)`).
    pub fn new(image_seq_len: i32, steps: u32) -> Result<Self> {
        if steps < 1 {
            bail!("steps must be >= 1, got {steps}");
        }
        let n = i32::try_from(steps)?;
        // Python computes `1.0 / num_inference_steps` in f64; MLX takes it as f32.
        let stop = (1.0_f64 / f64::from(steps)) as f32;
        let base = mlxcel_core::linspace(1.0, stop, n, dtype::FLOAT32);
        let mu = empirical_mu(image_seq_len, steps) as f32;
        // exp(mu) / (exp(mu) + (1/t - 1) ** 1.0), as `_time_shift_exponential_array`.
        let exp_mu = mlxcel_core::exp(&scalar(mu));
        let inv = mlxcel_core::subtract(&mlxcel_core::divide(&scalar(1.0), &base), &scalar(1.0));
        let powered = mlxcel_core::power(&inv, &scalar(1.0));
        let sigmas = mlxcel_core::divide(&exp_mu, &mlxcel_core::add(&exp_mu, &powered));
        let timesteps = mlxcel_core::multiply(&sigmas, &scalar(NUM_TRAIN_TIMESTEPS));
        let mut sigma_values = mlxcel_core::utils::array_to_vec_f32(&sigmas);
        sigma_values.push(0.0);
        Ok(Self {
            timesteps: mlxcel_core::utils::array_to_vec_f32(&timesteps),
            sigmas: sigma_values,
        })
    }

    #[must_use]
    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    #[must_use]
    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    /// Number of denoising steps.
    #[must_use]
    pub fn len(&self) -> usize {
        self.timesteps.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.timesteps.is_empty()
    }

    /// `latents + dt * noise`, with `dt = sigma[i+1] - sigma[i]` taken in f32
    /// and both `dt` and `noise` cast to the latent dtype first, as upstream.
    pub fn step(
        &self,
        noise: &MlxArray,
        step_index: usize,
        latents: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>> {
        if step_index + 1 >= self.sigmas.len() {
            bail!(
                "step index {step_index} is out of range for a {}-step schedule",
                self.len()
            );
        }
        let latent_dtype = mlxcel_core::array_dtype(latents);
        let dt = mlxcel_core::subtract(
            &scalar(self.sigmas[step_index + 1]),
            &scalar(self.sigmas[step_index]),
        );
        let dt = mlxcel_core::astype(&dt, latent_dtype);
        let noise = mlxcel_core::astype(noise, latent_dtype);
        Ok(mlxcel_core::add(
            latents,
            &mlxcel_core::multiply(&dt, &noise),
        ))
    }
}

#[cfg(test)]
#[path = "scheduler_tests.rs"]
mod tests;
