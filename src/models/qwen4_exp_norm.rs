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

//! Qwen4-Exp normalization layers.

use mlxcel_core::dtype;
use mlxcel_core::utils::silu;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::OutputGate;
use super::shapes;

/// RMSNorm over zero-centered weights: `x * rsqrt(mean(x^2) + eps) * (1 + w)`,
/// computed in float32 and cast back.
///
/// With `group_size`, an `[.., groups * group_size]` input is normalized one
/// `group_size` chunk at a time, each chunk against its own slice of `w`.
///
/// `1 + w` is formed once in float32 at load: folding it into a bf16 weight
/// for the fused RMSNorm kernel would round it to 8 mantissa bits.
pub(crate) struct ZeroCenteredRmsNorm {
    scale: UniquePtr<MlxArray>,
    group_size: Option<i32>,
    eps: f32,
}

impl ZeroCenteredRmsNorm {
    pub(crate) fn new(
        weight: &MlxArray,
        group_size: Option<usize>,
        eps: f32,
    ) -> Result<Self, String> {
        let shape = mlxcel_core::array_shape(weight);
        if shape.len() != 1 {
            return Err(format!(
                "zero-centered norm weight must be 1-D, got {shape:?}"
            ));
        }
        let weight = mlxcel_core::astype(weight, dtype::FLOAT32);
        let one = mlxcel_core::full_f32(&[1], 1.0, dtype::FLOAT32);
        let mut scale = mlxcel_core::add(&weight, &one);
        let group_size = match group_size {
            None => None,
            Some(group) => {
                let group = group as i32;
                if group == 0 || shape[0] % group != 0 {
                    return Err(format!(
                        "zero-centered norm width {} is not divisible by group size {group}",
                        shape[0]
                    ));
                }
                scale = mlxcel_core::reshape(&scale, &[shape[0] / group, group]);
                Some(group)
            }
        };
        mlxcel_core::eval(&scale);
        Ok(Self {
            scale,
            group_size,
            eps,
        })
    }

    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        width: usize,
        group_size: Option<usize>,
        eps: f32,
    ) -> Result<Self, String> {
        let key = format!("{prefix}.weight");
        shapes::expect_vector(weights, &key, width)?;
        Self::new(&weights[&key], group_size, eps)
    }

    pub(crate) fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let mut y = mlxcel_core::copy(x);
        if let Some(group) = self.group_size {
            let mut grouped = shape[..shape.len() - 1].to_vec();
            grouped.extend([-1, group]);
            y = mlxcel_core::reshape(&y, &grouped);
        }
        let normed = mlxcel_core::compiled_scaled_rms_norm(&y, &self.scale, self.eps);
        mlxcel_core::reshape(&normed, &shape)
    }
}

/// GatedDeltaNet output norm: `rms_norm(x) * act(gate)` with an ordinary
/// (not zero-centered) weight, the product taken in float32.
pub(crate) struct GatedOutputNorm {
    weight: UniquePtr<MlxArray>,
    eps: f32,
    gate: OutputGate,
}

impl GatedOutputNorm {
    pub(crate) fn new(weight: UniquePtr<MlxArray>, eps: f32, gate: OutputGate) -> Self {
        Self { weight, eps, gate }
    }

    pub(crate) fn forward(&self, x: &MlxArray, gate: &MlxArray) -> UniquePtr<MlxArray> {
        let input_dtype = mlxcel_core::array_dtype(x);
        let normed = mlxcel_core::fast_rms_norm(x, &self.weight, self.eps);
        let normed = mlxcel_core::astype(&normed, dtype::FLOAT32);
        let gate = mlxcel_core::astype(gate, dtype::FLOAT32);
        let gate = match self.gate {
            OutputGate::Sigmoid => mlxcel_core::sigmoid(&gate),
            OutputGate::Silu => silu(&gate),
        };
        mlxcel_core::astype(&mlxcel_core::multiply(&normed, &gate), input_dtype)
    }
}
