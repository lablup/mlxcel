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

//! Ideogram 4 logit-normal sampling schedule and sampler presets.
//!
//! A port of
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/ideogram4/scheduler.py>.
//! The shared [`crate::imagegen::scheduler`] is a flow-match sigma table with
//! a fixed shift; this schedule instead maps step intervals through the
//! inverse normal CDF, so it lives here. All arithmetic is f64, as upstream's
//! Python floats are.

use anyhow::{Result, bail};

/// One sampler preset: step count, per-step guidance, schedule mean and std.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerPreset {
    pub name: &'static str,
    pub num_steps: u32,
    /// `guidance_schedule[i]` is used at loop index `i`; the loop runs from
    /// `num_steps - 1` down to 0, so the last entries are applied first.
    pub guidance_schedule: &'static [f64],
    pub mu: f64,
    pub std: f64,
}

const fn schedule<const N: usize>(low: usize) -> [f64; N] {
    let mut out = [7.0; N];
    let mut i = 0;
    while i < low {
        out[i] = 3.0;
        i += 1;
    }
    out
}

static QUALITY_48: [f64; 48] = schedule::<48>(3);
static DEFAULT_20: [f64; 20] = schedule::<20>(2);
static TURBO_12: [f64; 12] = schedule::<12>(1);

/// Upstream `PRESETS`, sorted by name (the order upstream's error lists).
pub const PRESETS: &[SamplerPreset] = &[
    SamplerPreset {
        name: "V4_DEFAULT_20",
        num_steps: 20,
        guidance_schedule: &DEFAULT_20,
        mu: 0.0,
        std: 1.75,
    },
    SamplerPreset {
        name: "V4_QUALITY_48",
        num_steps: 48,
        guidance_schedule: &QUALITY_48,
        mu: 0.0,
        std: 1.5,
    },
    SamplerPreset {
        name: "V4_TURBO_12",
        num_steps: 12,
        guidance_schedule: &TURBO_12,
        mu: 0.5,
        std: 1.75,
    },
];

/// The preset a request without `sampler_preset` uses.
pub const DEFAULT_PRESET: &str = "V4_DEFAULT_20";

/// Upstream `get_preset`: `None` is [`DEFAULT_PRESET`]; an unknown name errors
/// listing the valid ones.
pub fn get_preset(name: Option<&str>) -> Result<&'static SamplerPreset> {
    let key = name.unwrap_or(DEFAULT_PRESET);
    PRESETS.iter().find(|p| p.name == key).map_or_else(
        || {
            let names: Vec<String> = PRESETS.iter().map(|p| format!("'{}'", p.name)).collect();
            bail!(
                "Unknown Ideogram 4 sampler preset '{key}'; expected one of [{}]",
                names.join(", ")
            )
        },
        Ok,
    )
}

/// Upstream `LogitNormalSchedule`: maps `t` in `[0, 1]` to a noise level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogitNormalSchedule {
    pub mean: f64,
    pub std: f64,
    pub logsnr_min: f64,
    pub logsnr_max: f64,
}

impl LogitNormalSchedule {
    #[must_use]
    pub fn new(mean: f64, std: f64) -> Self {
        Self {
            mean,
            std,
            logsnr_min: -15.0,
            logsnr_max: 18.0,
        }
    }

    /// `1 - sigmoid(mean + std * Phi^-1(t))`, clamped to the log-SNR range.
    #[must_use]
    pub fn at(&self, t: f64) -> f64 {
        let t_min = 1.0 / (1.0 + (0.5 * self.logsnr_max).exp());
        let t_max = 1.0 / (1.0 + (0.5 * self.logsnr_min).exp());
        if t <= 0.0 {
            return t_max;
        }
        if t >= 1.0 {
            return t_min;
        }
        let z = inv_normal_cdf(t);
        let shifted = self.mean + self.std * z;
        let value = 1.0 - (1.0 / (1.0 + (-shifted).exp()));
        value.max(t_min).min(t_max)
    }
}

/// Upstream `get_schedule_for_resolution` with `known_resolution` 512x512:
/// the mean shifts by `0.5 * ln(pixels / 512^2)`.
#[must_use]
pub fn get_schedule_for_resolution(
    height: u32,
    width: u32,
    known_mean: f64,
    std: f64,
) -> LogitNormalSchedule {
    let pixels = f64::from(height) * f64::from(width);
    let known = 512.0 * 512.0;
    LogitNormalSchedule::new(known_mean + 0.5 * (pixels / known).ln(), std)
}

/// Upstream `make_step_intervals`: `i / n` for `i` in `0..=n`.
pub fn make_step_intervals(num_steps: u32) -> Result<Vec<f64>> {
    if num_steps < 1 {
        bail!("num_steps must be >= 1, got {num_steps}");
    }
    let n = f64::from(num_steps);
    Ok((0..=num_steps).map(|i| f64::from(i) / n).collect())
}

/// Inverse of the standard normal CDF.
///
/// Wichura's AS241 (`PPND16`), transcribed from CPython's
/// `statistics._normal_dist_inv_cdf` with the same branches and evaluation
/// order, so it matches `NormalDist().inv_cdf` to the last bits (relative
/// error about 1e-16). Defined for `0 < p < 1`.
#[must_use]
#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
pub fn inv_normal_cdf(p: f64) -> f64 {
    let q = p - 0.5;
    if q.abs() <= 0.425 {
        let r = 0.180625 - q * q;
        let num = (((((((2.5090809287301226727e+3 * r + 3.3430575583588128105e+4) * r
            + 6.7265770927008700853e+4)
            * r
            + 4.5921953931549871457e+4)
            * r
            + 1.3731693765509461125e+4)
            * r
            + 1.9715909503065514427e+3)
            * r
            + 1.3314166789178437745e+2)
            * r
            + 3.3871328727963666080e+0)
            * q;
        let den = ((((((5.2264952788528545610e+3 * r + 2.8729085735721942674e+4) * r
            + 3.9307895800092710610e+4)
            * r
            + 2.1213794301586595867e+4)
            * r
            + 5.3941960214247511077e+3)
            * r
            + 6.8718700749205790830e+2)
            * r
            + 4.2313330701600911252e+1)
            * r
            + 1.0;
        return num / den;
    }
    let r = if q <= 0.0 { p } else { 1.0 - p };
    let mut r = (-r.ln()).sqrt();
    let (num, den) = if r <= 5.0 {
        r -= 1.6;
        (
            ((((((7.7454501427834140764e-4 * r + 2.2723844989269184583e-2) * r
                + 2.4178072517745061177e-1)
                * r
                + 1.2704582524523683826e+0)
                * r
                + 3.6478483247632045505e+0)
                * r
                + 5.7694972214606914055e+0)
                * r
                + 4.6303378461565452959e+0)
                * r
                + 1.4234371107496835773e+0,
            ((((((1.0507500716444168432e-9 * r + 5.4759380849953449460e-4) * r
                + 1.5198666563616457197e-2)
                * r
                + 1.4810397642748007459e-1)
                * r
                + 6.8976733498510000455e-1)
                * r
                + 1.6763848301838038494e+0)
                * r
                + 2.0531916266377588219e+0)
                * r
                + 1.0,
        )
    } else {
        r -= 5.0;
        (
            ((((((2.0103343992922881327e-7 * r + 2.7115555687434875782e-5) * r
                + 1.2426609473880784386e-3)
                * r
                + 2.6532189526576123093e-2)
                * r
                + 2.9656057182850489123e-1)
                * r
                + 1.7848265399172913358e+0)
                * r
                + 5.4637849111641143699e+0)
                * r
                + 6.6579046435011037772e+0,
            ((((((2.0442631033899397856e-15 * r + 1.4215117583164458887e-7) * r
                + 1.8463183175100546818e-5)
                * r
                + 7.8686913114561325910e-4)
                * r
                + 1.4875361290850614853e-2)
                * r
                + 1.3692988092273580531e-1)
                * r
                + 5.9983220655588793769e-1)
                * r
                + 1.0,
        )
    };
    let x = num / den;
    if q < 0.0 { -x } else { x }
}

#[cfg(test)]
#[path = "scheduler_tests.rs"]
mod tests;
