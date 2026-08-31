// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Inkling hierarchical-MLP vision tower.
//!
//! Inkling does not use a vision transformer. Each image tile is a
//! `[T=2, H=40, W=40, C=3]` temporal pair which is progressively folded into
//! the channel dimension and projected into one text-space soft token.

use mlxcel_core::layers::{RMSNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

fn default_model_type() -> String {
    "inkling_vision".into()
}
fn default_patch_size() -> usize {
    40
}
fn default_temporal_patch_size() -> usize {
    2
}
fn default_channels() -> usize {
    3
}
fn default_layers() -> usize {
    4
}
fn default_hidden_size() -> usize {
    6144
}
fn default_eps() -> f32 {
    1e-6
}

#[derive(Debug, Clone, Deserialize)]
pub struct InklingVisionConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: usize,
    #[serde(default = "default_channels", alias = "n_channels")]
    pub num_channels: usize,
    #[serde(default = "default_layers")]
    pub n_layers: usize,
    #[serde(default = "default_hidden_size", alias = "decoder_dmodel")]
    pub text_hidden_size: usize,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InklingHmlpLayerPlan {
    pub input_dim: usize,
    pub output_dim: usize,
    pub t_fold: usize,
    pub hw_fold: usize,
    pub add_norm: bool,
}

fn prime_factors(mut value: usize) -> Vec<usize> {
    let mut factors = Vec::new();
    while value.is_multiple_of(2) {
        factors.push(2);
        value /= 2;
    }
    let mut factor = 3;
    while factor * factor <= value {
        while value.is_multiple_of(factor) {
            factors.push(factor);
            value /= factor;
        }
        factor += 2;
    }
    if value > 1 {
        factors.push(value);
    }
    factors
}

fn cumulative_reversed(values: &[usize]) -> Vec<usize> {
    let mut product = 1usize;
    values
        .iter()
        .rev()
        .map(|&value| {
            product *= value;
            product
        })
        .collect()
}

fn assignment_search(
    row: usize,
    costs: &[Vec<f64>],
    used: &mut [bool],
    current: &mut Vec<usize>,
    current_cost: f64,
    best: &mut Option<(f64, Vec<usize>)>,
) {
    if row == costs.len() {
        if best.as_ref().is_none_or(|(cost, _)| current_cost < *cost) {
            *best = Some((current_cost, current.clone()));
        }
        return;
    }
    for column in 0..used.len() {
        if used[column] {
            continue;
        }
        let next_cost = current_cost + costs[row][column];
        if best.as_ref().is_some_and(|(cost, _)| next_cost >= *cost) {
            continue;
        }
        used[column] = true;
        current.push(column);
        assignment_search(row + 1, costs, used, current, next_cost, best);
        current.pop();
        used[column] = false;
    }
}

/// Compute the `(t, h, w, channels)` grids selected by the Inkling reference.
pub fn plan_out_scales(config: &InklingVisionConfig) -> Result<Vec<[i64; 4]>, String> {
    if config.patch_size == 0
        || config.temporal_patch_size == 0
        || config.num_channels == 0
        || config.n_layers == 0
    {
        return Err("Inkling vision dimensions and n_layers must be positive".into());
    }
    let spatial = cumulative_reversed(&prime_factors(config.patch_size));
    let temporal = cumulative_reversed(&prime_factors(config.temporal_patch_size));
    let final_h = *spatial
        .last()
        .ok_or_else(|| "Inkling patch_size must have a factorization".to_string())?;
    let mut scales = vec![[1_i64, 1, 1, config.num_channels as i64]];
    for &h in &spatial {
        let raw_channels = h
            .checked_mul(h)
            .and_then(|v| v.checked_mul(config.num_channels))
            .ok_or_else(|| "Inkling spatial channel plan overflowed".to_string())?;
        let channels = raw_channels.div_ceil(64) * 64;
        scales.push([1, h as i64, h as i64, channels as i64]);
    }
    for &t in &temporal {
        let raw_channels = final_h
            .checked_mul(final_h)
            .and_then(|v| v.checked_mul(config.num_channels))
            .and_then(|v| v.checked_mul(t))
            .ok_or_else(|| "Inkling temporal channel plan overflowed".to_string())?;
        let channels = raw_channels.div_ceil(64) * 64;
        scales.push([t as i64, final_h as i64, final_h as i64, channels as i64]);
    }

    let picks = config
        .n_layers
        .checked_add(1)
        .ok_or_else(|| "Inkling n_layers overflowed".to_string())?;
    if picks > scales.len() {
        return Err(format!(
            "Inkling n_layers={} requires {} HMLP grids, but the patch plan provides {}",
            config.n_layers,
            picks,
            scales.len()
        ));
    }
    let total_elements = config
        .patch_size
        .checked_mul(config.patch_size)
        .and_then(|v| v.checked_mul(config.temporal_patch_size))
        .and_then(|v| v.checked_mul(config.num_channels))
        .ok_or_else(|| "Inkling vision element count overflowed".to_string())?;
    let log_total = (total_elements as f64).ln();
    let costs = (0..picks)
        .map(|row| {
            let ideal = log_total * row as f64 / (picks - 1) as f64;
            scales
                .iter()
                .map(|scale| {
                    let reduction = scale[0] * scale[1] * scale[2];
                    (ideal - (reduction as f64).ln()).abs()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut best = None;
    assignment_search(
        0,
        &costs,
        &mut vec![false; scales.len()],
        &mut Vec::with_capacity(picks),
        0.0,
        &mut best,
    );
    let mut indices = best
        .ok_or_else(|| "Inkling HMLP scale assignment failed".to_string())?
        .1;
    indices[0] = 0;
    indices[picks - 1] = scales.len() - 1;
    Ok(indices.into_iter().map(|index| scales[index]).collect())
}

pub fn layer_plan(config: &InklingVisionConfig) -> Result<Vec<InklingHmlpLayerPlan>, String> {
    let scales = plan_out_scales(config)?;
    scales
        .windows(2)
        .enumerate()
        .map(|(index, pair)| {
            let start = pair[0];
            let end = pair[1];
            if end[0] % start[0] != 0 || end[1] % start[1] != 0 || end[2] % start[2] != 0 {
                return Err("Inkling HMLP plan contains non-integral folds".into());
            }
            let t_fold = (end[0] / start[0]) as usize;
            let hw_fold = (end[1] / start[1]) as usize;
            let input_dim = (start[3] as usize)
                .checked_mul(t_fold)
                .and_then(|v| v.checked_mul(hw_fold))
                .and_then(|v| v.checked_mul(hw_fold))
                .ok_or_else(|| "Inkling HMLP input width overflowed".to_string())?;
            Ok(InklingHmlpLayerPlan {
                input_dim,
                output_dim: if index + 1 == config.n_layers {
                    config.text_hidden_size
                } else {
                    end[3] as usize
                },
                t_fold,
                hw_fold,
                add_norm: index + 1 != config.n_layers,
            })
        })
        .collect()
}

/// Fold time, row, and column blocks into channels in reference order.
pub fn fold_timespace_to_depth(
    input: &MlxArray,
    t_fold: usize,
    hw_fold: usize,
) -> Result<UniquePtr<MlxArray>, String> {
    let shape = mlxcel_core::array_shape(input);
    if shape.len() != 5 {
        return Err(format!(
            "Inkling HMLP fold expects [B,T,H,W,C], got {shape:?}"
        ));
    }
    let [batch, time, height, width, channels] = [shape[0], shape[1], shape[2], shape[3], shape[4]];
    let t_fold = i32::try_from(t_fold).map_err(|_| "Inkling temporal fold exceeds i32")?;
    let hw_fold = i32::try_from(hw_fold).map_err(|_| "Inkling spatial fold exceeds i32")?;
    if time % t_fold != 0 || height % hw_fold != 0 || width % hw_fold != 0 {
        return Err(format!(
            "Inkling HMLP fold ({t_fold},{hw_fold}) does not divide input {shape:?}"
        ));
    }
    let folded = mlxcel_core::reshape(
        input,
        &[
            batch,
            time / t_fold,
            t_fold,
            height / hw_fold,
            hw_fold,
            width / hw_fold,
            hw_fold,
            channels,
        ],
    );
    let folded = mlxcel_core::transpose_axes(&folded, &[0, 1, 3, 5, 2, 4, 6, 7]);
    Ok(mlxcel_core::reshape(
        &folded,
        &[
            batch,
            time / t_fold,
            height / hw_fold,
            width / hw_fold,
            t_fold * hw_fold * hw_fold * channels,
        ],
    ))
}

struct InklingHmlpLayer {
    projection: UnifiedLinear,
    layer_norm: Option<RMSNorm>,
    plan: InklingHmlpLayerPlan,
}

fn validate_projection_shape(
    weights: &WeightMap,
    prefix: &str,
    plan: InklingHmlpLayerPlan,
    group_size: i32,
    bits: i32,
) -> Result<(), String> {
    let weight_name = format!("{prefix}.weight");
    let weight_shape = weights
        .get(&weight_name)
        .map(|weight| mlxcel_core::array_shape(weight))
        .ok_or_else(|| format!("Weight not found: {weight_name}"))?;
    let scales_name = format!("{prefix}.scales");
    let biases_name = format!("{prefix}.biases");
    let Some(scales) = weights.get(&scales_name) else {
        if weights.contains_key(&biases_name) {
            return Err(format!("{biases_name} exists without {scales_name}"));
        }
        let expected = [plan.output_dim as i32, plan.input_dim as i32];
        if weight_shape != expected {
            return Err(format!(
                "{weight_name}: expected {expected:?}, got {weight_shape:?}"
            ));
        }
        return Ok(());
    };

    if weight_shape.len() != 2 || weight_shape[0] != plan.output_dim as i32 {
        return Err(format!(
            "{weight_name}: expected {} output rows, got {weight_shape:?}",
            plan.output_dim
        ));
    }
    let scales_shape = mlxcel_core::array_shape(scales);
    let biases_shape = weights
        .get(&biases_name)
        .map(|biases| mlxcel_core::array_shape(biases));
    let mode =
        mlxcel_core::layers::infer_quantization_mode(biases_shape.is_some(), group_size, bits);
    mlxcel_core::layers::validate_quantized_packing(
        prefix,
        &mlxcel_core::layers::QuantizedTensorShapes {
            weight: &weight_shape,
            scales: &scales_shape,
            biases: biases_shape.as_deref(),
        },
        plan.input_dim,
        group_size,
        bits,
        mode,
    )
}

pub struct InklingHmlpEncoder {
    layers: Vec<InklingHmlpLayer>,
    final_norm: RMSNorm,
    hidden_size: usize,
}

impl InklingHmlpEncoder {
    pub fn from_weights(
        weights: &WeightMap,
        config: &InklingVisionConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        if config.model_type != "inkling_vision" {
            return Err(format!(
                "Inkling vision_config.model_type must be inkling_vision, got {}",
                config.model_type
            ));
        }
        if config.patch_size != 40 || config.temporal_patch_size != 2 || config.num_channels != 3 {
            return Err(format!(
                "Inkling HMLP requires [T=2,H=40,W=40,C=3], got T={}, H=W={}, C={}",
                config.temporal_patch_size, config.patch_size, config.num_channels
            ));
        }
        if config.text_hidden_size == 0 || config.text_hidden_size > i32::MAX as usize {
            return Err(format!(
                "Inkling vision text_hidden_size must be in 1..={}, got {}",
                i32::MAX,
                config.text_hidden_size
            ));
        }
        if !config.rms_norm_eps.is_finite() || config.rms_norm_eps <= 0.0 {
            return Err("Inkling vision rms_norm_eps must be finite and positive".into());
        }
        let plans = layer_plan(config)?;
        let mut layers = Vec::with_capacity(plans.len());
        for (index, plan) in plans.into_iter().enumerate() {
            let prefix = format!("vision_tower.encoder_layers.{index}.projection");
            if weights.contains_key(&format!("{prefix}.bias")) {
                return Err(format!("Inkling HMLP projection {index} must be bias-free"));
            }
            validate_projection_shape(weights, &prefix, plan, group_size, bits)?;
            let projection = UnifiedLinear::from_weights(weights, &prefix, group_size, bits)?;
            let layer_norm = if plan.add_norm {
                let name = format!("vision_tower.encoder_layers.{index}.layer_norm.weight");
                let weight = weights
                    .get(&name)
                    .map(|value| mlxcel_core::copy(value))
                    .ok_or_else(|| format!("Weight not found: {name}"))?;
                let shape = mlxcel_core::array_shape(&weight);
                if shape != [plan.output_dim as i32] {
                    return Err(format!(
                        "{name}: expected [{}], got {shape:?}",
                        plan.output_dim
                    ));
                }
                Some(RMSNorm::new(weight, config.rms_norm_eps))
            } else {
                None
            };
            layers.push(InklingHmlpLayer {
                projection,
                layer_norm,
                plan,
            });
        }
        let final_name = "vision_tower.final_norm.weight";
        let final_weight = weights
            .get(final_name)
            .map(|value| mlxcel_core::copy(value))
            .ok_or_else(|| format!("Weight not found: {final_name}"))?;
        let final_shape = mlxcel_core::array_shape(&final_weight);
        if final_shape != [config.text_hidden_size as i32] {
            return Err(format!(
                "{final_name}: expected [{}], got {final_shape:?}",
                config.text_hidden_size
            ));
        }
        Ok(Self {
            layers,
            final_norm: RMSNorm::new(final_weight, config.rms_norm_eps),
            hidden_size: config.text_hidden_size,
        })
    }

    pub fn forward(&self, pixel_values: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let shape = mlxcel_core::array_shape(pixel_values);
        if shape.len() != 5 || shape[1..] != [2, 40, 40, 3] {
            return Err(format!(
                "Inkling vision input must be [N,2,40,40,3], got {shape:?}"
            ));
        }
        let mut hidden = mlxcel_core::copy(pixel_values);
        for layer in &self.layers {
            if layer.plan.t_fold > 1 || layer.plan.hw_fold > 1 {
                hidden = fold_timespace_to_depth(&hidden, layer.plan.t_fold, layer.plan.hw_fold)?;
            }
            hidden = layer.projection.forward(&hidden);
            if let Some(norm) = &layer.layer_norm {
                hidden = mlxcel_core::gelu(&norm.forward(&hidden));
            }
        }
        hidden = self.final_norm.forward(&hidden);
        Ok(mlxcel_core::reshape(
            &hidden,
            &[shape[0], self.hidden_size as i32],
        ))
    }
}

#[cfg(test)]
#[path = "inkling_hmlp_tests.rs"]
mod tests;
