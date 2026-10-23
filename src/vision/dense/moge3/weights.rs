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

//! MoGe3 checkpoint access with exact upstream F32 preservation.

use anyhow::{Context, Result, ensure};
use mlxcel_core::weights::{WeightMap, load_weights_from_dir};
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::Moge3Config;
use super::conv_stack::sanitize_conv_transpose_weight;

pub fn load(path: &std::path::Path) -> Result<WeightMap> {
    let weights = load_weights_from_dir(path).map_err(anyhow::Error::msg)?;
    ensure!(!weights.is_empty(), "MoGe3 checkpoint contains no tensors");
    for (name, value) in &weights {
        ensure!(
            mlxcel_core::array_dtype(value) == mlxcel_core::dtype::FLOAT32,
            "MoGe3 tensor {name} must remain F32"
        );
    }
    Ok(weights)
}

pub fn array(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>> {
    weights
        .get(key)
        .map(|v| mlxcel_core::copy(v))
        .with_context(|| format!("MoGe3 weight not found: {key}"))
}

/// Match upstream's shape-based ConvTranspose2d sanitizer for every stack.
///
/// Published MLX checkpoints already use `[out, 2, 2, in]`; converted
/// PyTorch checkpoints may retain `[in, out, 2, 2]`. Comparing with the
/// configured target shape makes this operation idempotent.
pub fn sanitize_conv_transpose_weights(
    weights: &mut WeightMap,
    config: &Moge3Config,
) -> Result<()> {
    let stacks = [
        ("neck", Some(&config.neck)),
        ("points_head", config.points_head.as_ref()),
        ("mask_head", config.mask_head.as_ref()),
        ("normal_head", config.normal_head.as_ref()),
    ];
    for (prefix, stack) in stacks {
        let Some(stack) = stack else { continue };
        let kinds = stack
            .resamplers
            .resolve(stack.dim_res_blocks.len() - 1, "resamplers")?;
        for (index, kind) in kinds.iter().enumerate() {
            if kind != "conv_transpose" {
                continue;
            }
            let key = format!("{prefix}.resamplers.{index}.0.weight");
            let sanitized = sanitize_conv_transpose_weight(
                weights
                    .get(&key)
                    .with_context(|| format!("MoGe3 weight not found: {key}"))?,
                stack.dim_res_blocks[index],
                stack.dim_res_blocks[index + 1],
            )?;
            weights.insert(key, sanitized);
        }
    }
    Ok(())
}

pub struct Linear {
    weight: UniquePtr<MlxArray>,
    bias: UniquePtr<MlxArray>,
}
impl Linear {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
    ) -> Result<Self> {
        let weight = array(weights, &format!("{prefix}.weight"))?;
        let bias = array(weights, &format!("{prefix}.bias"))?;
        ensure!(
            mlxcel_core::array_shape(&weight) == [out_channels as i32, in_channels as i32],
            "MoGe3 linear {prefix} weight must be [{out_channels},{in_channels}]"
        );
        ensure!(
            mlxcel_core::array_shape(&bias) == [out_channels as i32],
            "MoGe3 linear {prefix} bias must be [{out_channels}]"
        );
        Ok(Self { weight, bias })
    }
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let wt = mlxcel_core::transpose(&self.weight);
        mlxcel_core::addmm(&self.bias, x, &wt, 1.0, 1.0)
    }
}

pub struct Conv2d {
    weight: UniquePtr<MlxArray>,
    bias: UniquePtr<MlxArray>,
    pub kernel: usize,
}
impl Conv2d {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        kernel: usize,
        in_channels: usize,
        out_channels: usize,
    ) -> Result<Self> {
        let weight = array(weights, &format!("{prefix}.weight"))?;
        let bias = array(weights, &format!("{prefix}.bias"))?;
        let shape = mlxcel_core::array_shape(&weight);
        ensure!(
            shape
                == [
                    out_channels as i32,
                    kernel as i32,
                    kernel as i32,
                    in_channels as i32,
                ],
            "MoGe3 convolution {prefix} has invalid shape {shape:?}"
        );
        ensure!(
            mlxcel_core::array_shape(&bias) == [out_channels as i32],
            "MoGe3 convolution {prefix} bias must be [{out_channels}]"
        );
        Ok(Self {
            weight,
            bias,
            kernel,
        })
    }
    pub fn forward(&self, x: &MlxArray, pad: i32) -> UniquePtr<MlxArray> {
        mlxcel_core::add(
            &mlxcel_core::conv2d(x, &self.weight, 1, 1, pad, pad, 1, 1, 1),
            &self.bias,
        )
    }
}
