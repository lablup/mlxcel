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

//! Muse Glimmer post-tower visual-token fusion.

use crate::models::muse_glimmer::MuseGlimmerConfig;
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::qwen2_vl::concat_many;

pub const MUSE_GLIMMER_VISION_ADAPTER_ROOT: &str = "model.vision_adapter";
pub const MUSE_GLIMMER_VISION_PROJECTION_ROOT: &str = "model.vision_projection";

pub struct MuseGlimmerVisionFusion {
    adapter: MuseGlimmerVisionAdapter,
    projection: UnifiedLinear,
    merge_size: usize,
    tower_hidden_size: usize,
    norm_eps: f32,
}

impl MuseGlimmerVisionFusion {
    pub fn from_weights(weights: &WeightMap, config: &MuseGlimmerConfig) -> Result<Self, String> {
        config.validate()?;
        let tower_hidden_size = config.vision_config.hidden_size;
        let merged_hidden_size =
            tower_hidden_size * config.vision_config.merge_size * config.vision_config.merge_size;
        if config.out_hidden_size != merged_hidden_size {
            return Err(format!(
                "Muse Glimmer out_hidden_size ({}) must equal merged vision hidden size ({merged_hidden_size})",
                config.out_hidden_size
            ));
        }

        let adapter = MuseGlimmerVisionAdapter::from_weights(weights, config)?;
        let projection = load_fusion_linear(
            weights,
            MUSE_GLIMMER_VISION_PROJECTION_ROOT,
            config.text_config.group_size(),
            config.text_config.bits(),
        )?;
        let projection_shape = unified_linear_shape(&projection)?;
        let expected_projection = vec![
            config.text_config.hidden_size as i32,
            config.projector_hidden_size as i32,
        ];
        if projection_shape != expected_projection {
            return Err(format!(
                "Muse Glimmer vision projection weight must be {expected_projection:?}, got {projection_shape:?}"
            ));
        }

        Ok(Self {
            adapter,
            projection,
            merge_size: config.vision_config.merge_size,
            tower_hidden_size,
            norm_eps: config.text_config.rms_norm_eps,
        })
    }

    pub fn forward(
        &self,
        tower_features: &MlxArray,
        image_grid_thw: &[(i32, i32, i32)],
    ) -> Result<UniquePtr<MlxArray>, String> {
        let merged = pixel_shuffle_2x2(
            tower_features,
            image_grid_thw,
            self.merge_size,
            self.tower_hidden_size,
        )?;
        let adapted = self.adapter.forward(&merged);
        let projected = self.projection.forward(&adapted);
        Ok(weightless_perception_norm(&projected, self.norm_eps))
    }
}

pub struct MuseGlimmerVisionAdapter {
    fc1: UnifiedLinear,
    fc2: UnifiedLinear,
}

impl MuseGlimmerVisionAdapter {
    pub fn from_weights(weights: &WeightMap, config: &MuseGlimmerConfig) -> Result<Self, String> {
        let fc1 = load_fusion_linear(
            weights,
            &format!("{MUSE_GLIMMER_VISION_ADAPTER_ROOT}.fc1"),
            config.text_config.group_size(),
            config.text_config.bits(),
        )?;
        let fc2 = load_fusion_linear(
            weights,
            &format!("{MUSE_GLIMMER_VISION_ADAPTER_ROOT}.fc2"),
            config.text_config.group_size(),
            config.text_config.bits(),
        )?;
        check_linear_shape(
            &fc1,
            &[
                config.projector_hidden_size as i32,
                config.out_hidden_size as i32,
            ],
            "Muse Glimmer vision adapter fc1",
        )?;
        check_linear_shape(
            &fc2,
            &[
                config.projector_hidden_size as i32,
                config.projector_hidden_size as i32,
            ],
            "Muse Glimmer vision adapter fc2",
        )?;
        Ok(Self { fc1, fc2 })
    }

    pub fn forward(&self, hidden_states: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.fc1.forward(hidden_states);
        let h = mlxcel_core::gelu(&h);
        let h = self.fc2.forward(&h);
        mlxcel_core::gelu(&h)
    }
}

pub fn pixel_shuffle_2x2(
    hidden_states: &MlxArray,
    image_grid_thw: &[(i32, i32, i32)],
    merge_size: usize,
    expected_hidden_size: usize,
) -> Result<UniquePtr<MlxArray>, String> {
    if merge_size != 2 {
        return Err(format!(
            "Muse Glimmer pixel shuffle currently requires merge_size 2, got {merge_size}"
        ));
    }
    if image_grid_thw.is_empty() {
        return Err("Muse Glimmer pixel shuffle requires at least one image grid".to_string());
    }

    let shape = mlxcel_core::array_shape(hidden_states);
    if shape.len() != 2 || shape[1] as usize != expected_hidden_size {
        return Err(format!(
            "Muse Glimmer tower features must be [patches, {expected_hidden_size}], got {shape:?}"
        ));
    }

    let mut expected_rows = 0;
    for &grid in image_grid_thw {
        let (t, h, w) = grid;
        if t <= 0 || h <= 0 || w <= 0 {
            return Err(format!("Muse Glimmer grid must be positive, got {grid:?}"));
        }
        if h % 2 != 0 || w % 2 != 0 {
            return Err(format!(
                "Muse Glimmer grid {grid:?} must have h and w divisible by 2"
            ));
        }
        expected_rows += t * h * w;
    }
    if shape[0] != expected_rows {
        return Err(format!(
            "Muse Glimmer tower feature rows {} do not match image_grid_thw patch rows {expected_rows}",
            shape[0]
        ));
    }

    let hidden = shape[1];
    let mut offset = 0;
    let mut outputs = Vec::with_capacity(image_grid_thw.len());
    for &(t, h, w) in image_grid_thw {
        let mut indices = Vec::with_capacity((t * h * w) as usize);
        for frame in 0..t {
            let frame_base = offset + frame * h * w;
            for block_y in (0..h).step_by(2) {
                for block_x in (0..w).step_by(2) {
                    for dy in 0..2 {
                        for dx in 0..2 {
                            indices.push(frame_base + (block_y + dy) * w + block_x + dx);
                        }
                    }
                }
            }
        }
        let row_count = t * (h / 2) * (w / 2);
        let index = mlxcel_core::from_slice_i32(&indices, &[indices.len() as i32]);
        let chunk = mlxcel_core::take(hidden_states, &index, 0);
        let chunk = mlxcel_core::reshape(&chunk, &[row_count, 4, hidden]);
        let chunk = mlxcel_core::transpose_axes(&chunk, &[0, 2, 1]);
        outputs.push(mlxcel_core::reshape(&chunk, &[row_count, hidden * 4]));
        offset += t * h * w;
    }

    Ok(if outputs.len() == 1 {
        outputs.remove(0)
    } else {
        concat_many(&outputs, 0)
    })
}

pub fn weightless_perception_norm(hidden_states: &MlxArray, eps: f32) -> UniquePtr<MlxArray> {
    mlxcel_core::fast_rms_norm_no_weight(hidden_states, eps)
}

fn load_fusion_linear(
    weights: &WeightMap,
    prefix: &str,
    group_size: i32,
    bits: i32,
) -> Result<UnifiedLinear, String> {
    if weights.contains_key(&format!("{prefix}.bias")) {
        return Err(format!(
            "Muse Glimmer vision fusion linear must not have bias: {prefix}.bias"
        ));
    }
    UnifiedLinear::from_weights(weights, prefix, group_size, bits)
}

fn unified_linear_shape(linear: &UnifiedLinear) -> Result<Vec<i32>, String> {
    match linear {
        UnifiedLinear::Regular(linear) => Ok(mlxcel_core::array_shape(&linear.weight)),
        UnifiedLinear::Quantized { weight, .. } => {
            let packed = mlxcel_core::array_shape(&weight.weight);
            let scales = mlxcel_core::array_shape(&weight.scales);
            if packed.len() != 2 || scales.len() != 2 || packed[0] != scales[0] {
                return Err(format!(
                    "Muse Glimmer quantized vision fusion linear has inconsistent weight/scales shapes: weight={packed:?}, scales={scales:?}"
                ));
            }
            let logical_input = i64::from(scales[1]) * i64::from(weight.group_size);
            let logical_input = i32::try_from(logical_input).map_err(|_| {
                format!(
                    "Muse Glimmer quantized vision fusion input width overflows i32: {} * {}",
                    scales[1], weight.group_size
                )
            })?;
            Ok(vec![packed[0], logical_input])
        }
    }
}

fn check_linear_shape(linear: &UnifiedLinear, expected: &[i32], label: &str) -> Result<(), String> {
    let actual = unified_linear_shape(linear)?;
    if actual != expected {
        return Err(format!(
            "{label} weight must be {expected:?}, got {actual:?}"
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "muse_glimmer_fusion_tests.rs"]
mod tests;
