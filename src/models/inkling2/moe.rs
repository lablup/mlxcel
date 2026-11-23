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

//! Inkling 2 sparse MoE: sigmoid router with a correction bias, top-k
//! renormalization, softcapped SwiGLU experts, f32 combine.
//!
//! Derived from mlx-vlm `SparseMoE` and `_route` / `_combine_routes`
//! (`mlx_vlm/models/inkling2/language.py`, MIT).
//!
//! The expert matvec runs through [`expert_projections`] (the gather path),
//! except inside a short speculative verify forward of a looped model, where
//! the grouped Metal kernel (mlx-vlm `grouped_experts.py`, issue #3359; gate
//! in `super::kernels`) decodes each expert's weights once for all its routes.

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr, dtype};

use super::config::Inkling2TextConfig;
use super::quant::{ExpertQuant, Inkling2QuantTable, MXFP8, NVFP4};
use crate::models::switch_layers::{SwitchGLU, SwitchGluActivation};

pub struct Inkling2Moe {
    /// f32 `[experts, hidden]`.
    gate_weight: UniquePtr<MlxArray>,
    /// f32 `[experts]`.
    bias: UniquePtr<MlxArray>,
    experts: SwitchGLU,
    top_k: i32,
    route_scale: f32,
    /// Grouped verify kernel (issue #3359): a looped model whose three planes
    /// the kernel decodes, kernels on at load.
    grouped: bool,
}

/// The quantization of one stacked expert plane, read from its tensors.
///
/// Affine planes carry zero-point biases. Block-float planes do not, and are
/// told apart by the packed width against the input width: NVFP4 is 4 bits in
/// groups of 16 (with an MLX `native_global_scale`), MXFP8 8 bits in groups
/// of 32. An unquantized plane has no scales.
fn plane_quant(
    weights: &WeightMap,
    prefix: &str,
    in_features: i32,
    affine: (i32, i32),
) -> Result<Option<ExpertQuant>, String> {
    let Some(scales) = weights.get(&format!("{prefix}.scales")) else {
        return Ok(None);
    };
    if weights.contains_key(&format!("{prefix}.biases")) {
        return Ok(Some(ExpertQuant {
            group_size: affine.0,
            bits: affine.1,
            mode: "affine",
        }));
    }
    let weight = weights
        .get(&format!("{prefix}.weight"))
        .ok_or_else(|| format!("Weight not found: {prefix}.weight"))?;
    let packed = *mlxcel_core::array_shape(weight).last().unwrap_or(&0);
    let groups = *mlxcel_core::array_shape(scales).last().unwrap_or(&0);
    let has_global = weights.contains_key(&format!("{prefix}.native_global_scale"));
    let quant = match (packed * 32, groups) {
        (bits_in, groups)
            if bits_in == 4 * in_features && groups * 16 == in_features && has_global =>
        {
            NVFP4
        }
        (bits_in, groups)
            if bits_in == 8 * in_features && groups * 32 == in_features && !has_global =>
        {
            MXFP8
        }
        _ => {
            return Err(format!(
                "{prefix}: packed width {packed} and {groups} scale groups match neither NVFP4 \
                 (with a global scale) nor MXFP8 for {in_features} input features"
            ));
        }
    };
    Ok(Some(quant))
}

impl Inkling2Moe {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &Inkling2TextConfig,
        layer: usize,
        affine: (i32, i32),
        table: Option<&Inkling2QuantTable>,
    ) -> Result<Self, String> {
        let gate_key = format!("{prefix}.gate_weight");
        let gate_weight = weights
            .get(&gate_key)
            .ok_or_else(|| format!("Weight not found: {gate_key}"))?;
        let experts = config.n_routed_experts as i32;
        let hidden = config.hidden_size as i32;
        if mlxcel_core::array_shape(gate_weight) != [experts, hidden] {
            return Err(format!(
                "{gate_key} must be [{experts}, {hidden}], got {:?}",
                mlxcel_core::array_shape(gate_weight)
            ));
        }
        let bias_key = format!("{prefix}.e_score_correction_bias");
        let bias = match weights.get(&bias_key) {
            Some(bias) if mlxcel_core::array_shape(bias) == [experts] => {
                mlxcel_core::astype(bias, dtype::FLOAT32)
            }
            Some(bias) => {
                return Err(format!(
                    "{bias_key} must be [{experts}], got {:?}",
                    mlxcel_core::array_shape(bias)
                ));
            }
            None => mlxcel_core::zeros(&[experts], dtype::FLOAT32),
        };
        let switch = format!("{prefix}.switch_mlp");
        let mut modes = Vec::with_capacity(3);
        for (projection, in_features) in [
            ("gate_proj", hidden),
            ("up_proj", hidden),
            ("down_proj", config.intermediate_size as i32),
        ] {
            let plane = format!("{switch}.{projection}");
            let quant = plane_quant(weights, &plane, in_features, affine)?;
            if let (Some(table), Some(found)) = (table, quant)
                && let Some(declared) = table.get(layer, projection)
                && declared != found
            {
                return Err(format!(
                    "{plane}: hf_quant_config.json declares {} but the tensors are {}",
                    declared.mode, found.mode
                ));
            }
            modes.push(quant);
        }
        if modes.iter().any(|mode| *mode != modes[0]) {
            return Err(format!(
                "{switch}: gate, up and down projections must share one quantization, got {modes:?}"
            ));
        }
        let experts_glu = match modes[0] {
            Some(quant) => SwitchGLU::from_weights_with_mode(
                weights,
                &switch,
                quant.group_size,
                quant.bits,
                quant.mode,
            )?,
            None => SwitchGLU::from_weights(weights, &switch, affine.0, affine.1)?,
        }
        // `tanh(g / 4) * 4 * sigmoid(g) * tanh(u / 25) * 25` in f32: SiTU with
        // both caps, mlx-vlm's `SoftcappedSwiGLU`.
        .with_activation(SwitchGluActivation::SiTU {
            beta: config.mlp_gate_softcap,
            linear_beta: Some(config.mlp_up_softcap),
        });
        let grouped = super::kernels::loop_kernels_wanted(config, super::kernels::enabled())
            && super::kernels::grouped_experts_supported(&experts_glu);
        Ok(Self {
            gate_weight: mlxcel_core::astype(gate_weight, dtype::FLOAT32),
            bias,
            experts: experts_glu,
            top_k: config.num_experts_per_tok as i32,
            route_scale: config.route_scale,
            grouped,
        })
    }

    /// `(indices [N, k] uint32-compatible, weights [N, k] f32)`.
    fn route(&self, x: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let scores = mlxcel_core::sigmoid(&mlxcel_core::matmul(
            &mlxcel_core::astype(x, dtype::FLOAT32),
            &mlxcel_core::transpose(&self.gate_weight),
        ));
        let biased = mlxcel_core::negative(&mlxcel_core::add(&scores, &self.bias));
        let order = mlxcel_core::argpartition(&biased, self.top_k - 1, -1);
        let rows = mlxcel_core::array_shape(&order)[0];
        let indices = mlxcel_core::slice(&order, &[0, 0], &[rows, self.top_k]);
        let picked = mlxcel_core::take_along_axis(&scores, &indices, -1);
        let total = mlxcel_core::add(
            &mlxcel_core::sum_axis(&picked, -1, true),
            &mlxcel_core::from_slice_f32(&[1e-20], &[1]),
        );
        let weights =
            mlxcel_core::multiply_scalar(&mlxcel_core::divide(&picked, &total), self.route_scale);
        (indices, weights)
    }

    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(x);
        let hidden = shape[shape.len() - 1];
        let tokens: i32 = shape[..shape.len() - 1].iter().product();
        let flat = mlxcel_core::reshape(x, &[tokens, hidden]);
        let (indices, weights) = self.route(&flat);
        let grouped = (self.grouped && super::kernels::grouped_experts_apply(x, self.top_k))
            .then(|| super::kernels::grouped_expert_projections(&self.experts, &flat, &indices))
            .flatten();
        let out = match grouped {
            Some(out) => out,
            None => expert_projections(&self.experts, &flat, &indices),
        };
        let out_dtype = mlxcel_core::array_dtype(&out);
        let weighted = mlxcel_core::multiply(
            &mlxcel_core::astype(&out, dtype::FLOAT32),
            &mlxcel_core::expand_dims(&weights, -1),
        );
        let combined = mlxcel_core::astype(&mlxcel_core::sum_axis(&weighted, -2, false), out_dtype);
        mlxcel_core::reshape(&combined, &shape)
    }
}

/// Routed expert outputs `[N, k, hidden]` for `x [N, hidden]`.
pub(crate) fn expert_projections(
    experts: &SwitchGLU,
    x: &MlxArray,
    indices: &MlxArray,
) -> UniquePtr<MlxArray> {
    experts.forward(x, indices)
}
