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

//! MoGe3 image preprocessing and inference-option validation.

use anyhow::{Result, bail, ensure};
use image::DynamicImage;
use mlxcel_core::{MlxArray, UniquePtr};
use serde_json::{Map, Value};

use super::config::{MAX_REFINE_STEPS, MAX_RUNTIME_PIXELS, Moge3Config};

const MAX_OUTPUT_BYTES: usize = 512 << 20;

#[derive(Debug, Clone, Copy)]
pub struct InferOptions {
    pub num_tokens: usize,
    pub force_projection: bool,
    pub apply_mask: bool,
    pub fov_x: Option<f32>,
    pub refine_steps: usize,
    pub return_per_step: bool,
}

fn bool_opt(opts: &Map<String, Value>, name: &str, default: bool) -> Result<bool> {
    match opts.get(name) {
        None => Ok(default),
        Some(Value::Bool(v)) => Ok(*v),
        Some(_) => bail!("MoGe3 option `{name}` must be a boolean"),
    }
}
fn usize_opt(opts: &Map<String, Value>, name: &str) -> Result<Option<usize>> {
    match opts.get(name) {
        None => Ok(None),
        Some(v) => {
            let n = v.as_u64().ok_or_else(|| {
                anyhow::anyhow!("MoGe3 option `{name}` must be a non-negative integer")
            })?;
            usize::try_from(n)
                .map(Some)
                .map_err(|_| anyhow::anyhow!("MoGe3 option `{name}` is too large"))
        }
    }
}

impl InferOptions {
    pub fn validate(&self, config: &Moge3Config) -> Result<()> {
        let [min, max] = config.num_tokens_range;
        ensure!(
            (min..=max).contains(&self.num_tokens),
            "MoGe3 num_tokens must be in checkpoint range {min}..={max}"
        );
        ensure!(
            self.refine_steps <= MAX_REFINE_STEPS,
            "MoGe3 refine_steps must be at most {MAX_REFINE_STEPS}"
        );
        ensure!(
            self.refine_steps == 0 || config.refiner.is_some(),
            "MoGe3 refiner is disabled but refine_steps is positive"
        );
        if let Some(fov) = self.fov_x {
            ensure!(
                fov.is_finite() && fov > 0.0 && fov < 180.0,
                "MoGe3 fov_x must be finite and between 0 and 180 degrees"
            );
        }
        Ok(())
    }

    pub fn parse(config: &Moge3Config, opts: &Map<String, Value>) -> Result<Self> {
        const KNOWN: &[&str] = &[
            "task",
            "num_tokens",
            "resolution_level",
            "force_projection",
            "apply_mask",
            "fov_x",
            "refine_steps",
            "return_per_step",
        ];
        for key in opts.keys() {
            ensure!(
                KNOWN.contains(&key.as_str()),
                "unknown MoGe3 option `{key}`"
            );
        }
        if let Some(task) = opts.get("task") {
            ensure!(
                task.as_str() == Some("geometry"),
                "MoGe3 option `task` must be `geometry`"
            );
        }
        let num_tokens = usize_opt(opts, "num_tokens")?;
        let level = usize_opt(opts, "resolution_level")?.unwrap_or(9);
        ensure!(level <= 9, "MoGe3 resolution_level must be between 0 and 9");
        let [min, max] = config.num_tokens_range;
        let num_tokens = num_tokens.unwrap_or_else(|| min + (level * (max - min)) / 9);
        let refine_steps = usize_opt(opts, "refine_steps")?.unwrap_or(3);
        let fov_x = match opts.get("fov_x") {
            None => None,
            Some(v) => {
                let n = v
                    .as_f64()
                    .ok_or_else(|| anyhow::anyhow!("MoGe3 option `fov_x` must be a number"))?;
                ensure!(
                    n.is_finite() && n > 0.0 && n < 180.0,
                    "MoGe3 fov_x must be finite and between 0 and 180 degrees"
                );
                Some(n as f32)
            }
        };
        let result = Self {
            num_tokens,
            force_projection: bool_opt(opts, "force_projection", true)?,
            apply_mask: bool_opt(opts, "apply_mask", true)?,
            fov_x,
            refine_steps,
            return_per_step: bool_opt(opts, "return_per_step", false)?,
        };
        result.validate(config)?;
        Ok(result)
    }
}

pub(super) fn validate_output_budget(shape: &[i32], options: InferOptions) -> Result<()> {
    let (batch, height, width) = match shape {
        [h, w, 3] => (1usize, *h as usize, *w as usize),
        [b, h, w, 3] => (*b as usize, *h as usize, *w as usize),
        _ => bail!("MoGe3 float image must be HWC or NHWC RGB"),
    };
    let pixels = batch
        .checked_mul(height)
        .and_then(|v| v.checked_mul(width))
        .ok_or_else(|| anyhow::anyhow!("MoGe3 output pixel budget overflow"))?;
    let fixed = pixels
        .checked_mul(29)
        .ok_or_else(|| anyhow::anyhow!("MoGe3 fixed output budget overflow"))?;
    let per_step = if options.return_per_step {
        pixels
            .checked_mul(options.refine_steps + 1)
            .and_then(|v| v.checked_mul(16))
            .ok_or_else(|| anyhow::anyhow!("MoGe3 per-step output budget overflow"))?
    } else {
        0
    };
    let total = fixed
        .checked_add(per_step)
        .ok_or_else(|| anyhow::anyhow!("MoGe3 output budget overflow"))?;
    ensure!(
        total <= MAX_OUTPUT_BYTES,
        "MoGe3 retained outputs require {total} bytes, exceeding {MAX_OUTPUT_BYTES}"
    );
    Ok(())
}

pub fn image_to_array(image: &DynamicImage) -> Result<UniquePtr<MlxArray>> {
    ensure!(
        image.width() > 0 && image.height() > 0,
        "MoGe3 image dimensions must be positive"
    );
    let pixels = (image.width() as usize)
        .checked_mul(image.height() as usize)
        .ok_or_else(|| anyhow::anyhow!("MoGe3 image size overflow"))?;
    ensure!(
        pixels <= MAX_RUNTIME_PIXELS,
        "MoGe3 image exceeds {MAX_RUNTIME_PIXELS} pixels"
    );
    Ok(crate::vision::dense::preprocess::image_to_nhwc_f32(image))
}

pub fn validate_float_image(image: &MlxArray) -> Result<()> {
    let shape = mlxcel_core::array_shape(image);
    ensure!(
        shape.len() == 3 || shape.len() == 4,
        "MoGe3 float image must be HWC or NHWC"
    );
    ensure!(
        shape.iter().all(|&v| v > 0) && shape.last() == Some(&3),
        "MoGe3 float image must have positive RGB shape"
    );
    ensure!(
        matches!(
            mlxcel_core::array_dtype(image),
            mlxcel_core::dtype::FLOAT16
                | mlxcel_core::dtype::BFLOAT16
                | mlxcel_core::dtype::FLOAT32
        ),
        "MoGe3 array image must be floating point"
    );
    let min = mlxcel_core::min_all(image);
    let max = mlxcel_core::max_all(image);
    mlxcel_core::eval(&min);
    mlxcel_core::eval(&max);
    let lo = mlxcel_core::item_f32(&mlxcel_core::astype(&min, mlxcel_core::dtype::FLOAT32));
    let hi = mlxcel_core::item_f32(&mlxcel_core::astype(&max, mlxcel_core::dtype::FLOAT32));
    ensure!(
        lo.is_finite() && hi.is_finite() && lo >= 0.0 && hi <= 1.0,
        "MoGe3 floating-point image values must be finite and in [0, 1]"
    );
    Ok(())
}
