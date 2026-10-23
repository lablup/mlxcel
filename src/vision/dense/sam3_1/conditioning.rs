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

//! Retrieval and temporal encoding of SAM 3.1 tracker memories.

use mlxcel_core::{MlxArray, UniquePtr};

use super::multiplex::MULTIPLEX_COUNT;
use super::state::{FrameFeatures, FrameOutput, TrackerState, select_closest_conditioning};
use super::tracker::MultiplexTracker;

fn sine_position_1d(positions: &[usize], max_absolute_position: usize, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let denominator = max_absolute_position.saturating_sub(1).max(1) as f32;
    let mut encoded = Vec::with_capacity(positions.len() * dim);
    for &position in positions {
        let position = position as f32 / denominator;
        let mut angles = Vec::with_capacity(half);
        for lane in 0..half {
            let divisor = 10_000_f32.powf((2 * (lane / 2)) as f32 / half as f32);
            angles.push(position / divisor);
        }
        encoded.extend(angles.iter().map(|value| value.sin()));
        encoded.extend(angles.iter().map(|value| value.cos()));
    }
    encoded
}

impl MultiplexTracker {
    pub(super) fn memory_conditioned_features(
        &self,
        state: &TrackerState,
        frame_index: usize,
        total_frames: usize,
        features: &FrameFeatures,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let current = &features.propagation.vision;
        let position = features
            .propagation
            .position
            .as_deref()
            .ok_or_else(|| "SAM 3.1 propagation position is missing".to_owned())?;
        let shape = mlxcel_core::array_shape(current);
        let (buckets, height, width, channels) = (
            state.multiplex.num_buckets() as i32,
            shape[1],
            shape[2],
            shape[3],
        );
        let tokens = height * width;
        let image = mlxcel_core::reshape(current, &[1, tokens, channels]);
        let source = mlxcel_core::broadcast_to(&image, &[buckets, tokens, channels]);
        let source_position = mlxcel_core::broadcast_to(
            &mlxcel_core::reshape(position, &[1, tokens, channels]),
            &[buckets, tokens, channels],
        );

        let selected = select_closest_conditioning(
            frame_index,
            state.conditioning.keys().copied(),
            self.config.max_cond_frame_num,
        );
        let selected_set: std::collections::BTreeSet<usize> = selected.iter().copied().collect();

        let mut spatial_memory = Vec::new();
        let mut spatial_position = Vec::new();
        let mut image_memory = Vec::new();
        let mut image_position = Vec::new();

        for &index in &selected {
            if let Some(output) = state.conditioning.get(&index) {
                self.append_spatial_memory(
                    output,
                    frame_index.abs_diff(index),
                    true,
                    buckets,
                    tokens,
                    channels,
                    &mut spatial_memory,
                    &mut spatial_position,
                    &mut image_memory,
                    &mut image_position,
                );
            }
        }

        let stride = self.config.memory_temporal_stride_for_eval;
        for temporal_position in 1..self.config.num_maskmem {
            let relative = self.config.num_maskmem - temporal_position;
            let previous = if relative == 1 {
                frame_index.checked_sub(1)
            } else {
                frame_index
                    .checked_sub(2)
                    .map(|index| index / stride * stride)
                    .and_then(|index| index.checked_sub((relative - 2) * stride))
            };
            let Some(previous) = previous else {
                continue;
            };
            let output = state.non_conditioning.get(&previous).or_else(|| {
                (!selected_set.contains(&previous))
                    .then(|| state.conditioning.get(&previous))
                    .flatten()
            });
            if let Some(output) = output {
                self.append_spatial_memory(
                    output,
                    temporal_position,
                    false,
                    buckets,
                    tokens,
                    channels,
                    &mut spatial_memory,
                    &mut spatial_position,
                    &mut image_memory,
                    &mut image_position,
                );
            }
        }

        let mut pointer_positions = Vec::new();
        let mut pointers = Vec::new();
        let max_pointers = total_frames.min(self.config.max_object_pointers_in_encoder);
        for &index in &selected {
            if let Some(output) = state.conditioning.get(&index) {
                pointer_positions.push(frame_index.abs_diff(index));
                pointers.push(mlxcel_core::copy(&output.object_pointers));
            }
        }
        for difference in 1..max_pointers {
            let Some(index) = frame_index.checked_sub(difference) else {
                break;
            };
            let output = state.non_conditioning.get(&index).or_else(|| {
                (!selected_set.contains(&index))
                    .then(|| state.conditioning.get(&index))
                    .flatten()
            });
            if let Some(output) = output {
                pointer_positions.push(difference);
                pointers.push(mlxcel_core::copy(&output.object_pointers));
            }
        }
        let mut excluded_pointer_tokens = 0;
        if !pointers.is_empty() {
            let refs: Vec<&MlxArray> = pointers.iter().map(|value| &**value).collect();
            let pointers = mlxcel_core::concatenate_many(&refs, 1);
            excluded_pointer_tokens = mlxcel_core::array_shape(&pointers)[1];
            let pointer_position =
                self.temporal_pointer_position(&pointer_positions, max_pointers, buckets);
            spatial_memory.push(pointers);
            spatial_position.push(pointer_position);
        }

        if spatial_memory.is_empty() || image_memory.is_empty() {
            return Ok(mlxcel_core::reshape(
                &source,
                &[buckets, height, width, channels],
            ));
        }
        let memory_refs: Vec<&MlxArray> = spatial_memory.iter().map(|value| &**value).collect();
        let memory_position_refs: Vec<&MlxArray> =
            spatial_position.iter().map(|value| &**value).collect();
        let image_refs: Vec<&MlxArray> = image_memory.iter().map(|value| &**value).collect();
        let image_position_refs: Vec<&MlxArray> =
            image_position.iter().map(|value| &**value).collect();
        let attended = self.memory_attention.forward(
            &image,
            &source,
            &mlxcel_core::concatenate_many(&image_refs, 1),
            &mlxcel_core::concatenate_many(&memory_refs, 1),
            Some(&source_position),
            Some(&mlxcel_core::concatenate_many(&memory_position_refs, 1)),
            Some(&mlxcel_core::concatenate_many(&image_position_refs, 1)),
            excluded_pointer_tokens,
        )?;
        Ok(mlxcel_core::reshape(
            &attended,
            &[buckets, height, width, channels],
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn append_spatial_memory(
        &self,
        output: &FrameOutput,
        temporal_position: usize,
        conditioning: bool,
        buckets: i32,
        tokens: i32,
        channels: i32,
        memory: &mut Vec<UniquePtr<MlxArray>>,
        memory_position: &mut Vec<UniquePtr<MlxArray>>,
        image_memory: &mut Vec<UniquePtr<MlxArray>>,
        image_position: &mut Vec<UniquePtr<MlxArray>>,
    ) {
        let Some(features) = &output.memory_features else {
            return;
        };
        memory.push(mlxcel_core::reshape(features, &[buckets, tokens, channels]));
        let position = self.feature_position.forward(features);
        let temporal_index = if self.config.use_maskmem_tpos_v2 {
            if temporal_position > 0 && temporal_position < self.config.num_maskmem {
                self.config.num_maskmem - temporal_position - 1
            } else {
                self.config.num_maskmem - 1
            }
        } else {
            self.config.num_maskmem - if conditioning { 0 } else { temporal_position } - 1
        };
        let temporal = mlxcel_core::slice(
            &self.temporal_memory_position,
            &[temporal_index as i32, 0, 0, 0],
            &[temporal_index as i32 + 1, 1, 1, channels],
        );
        let temporal = mlxcel_core::reshape(&temporal, &[1, 1, channels]);
        memory_position.push(mlxcel_core::add(
            &mlxcel_core::reshape(&position, &[buckets, tokens, channels]),
            &temporal,
        ));
        image_memory.push(mlxcel_core::reshape(
            &output.image_features,
            &[1, tokens, channels],
        ));
        let image_spatial_position = self.feature_position.forward(&output.image_features);
        image_position.push(mlxcel_core::add(
            &mlxcel_core::reshape(&image_spatial_position, &[1, tokens, channels]),
            &temporal,
        ));
    }

    fn temporal_pointer_position(
        &self,
        positions: &[usize],
        max_absolute_position: usize,
        buckets: i32,
    ) -> UniquePtr<MlxArray> {
        let dim = self.config.memory_attention_hidden_size;
        let encoded = sine_position_1d(positions, max_absolute_position, dim);
        let position = mlxcel_core::from_slice_f32(&encoded, &[positions.len() as i32, dim as i32]);
        let position = self.temporal_position_projection.forward(&position);
        let position = mlxcel_core::repeat(&position, MULTIPLEX_COUNT as i32, 0);
        mlxcel_core::broadcast_to(
            &mlxcel_core::expand_dims(&position, 0),
            &[buckets, mlxcel_core::array_shape(&position)[0], dim as i32],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::sine_position_1d;

    #[test]
    fn temporal_pointer_sine_encoding_matches_pinned_upstream_literals() {
        let actual = sine_position_1d(&[0, 1, 3], 4, 8);
        let expected = [
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            0.3271947,
            0.3271947,
            0.0033333272,
            0.0033333272,
            0.94495696,
            0.94495696,
            0.99999446,
            0.99999446,
            0.84147096,
            0.84147096,
            0.009999833,
            0.009999833,
            0.5403023,
            0.5403023,
            0.99995,
            0.99995,
        ];
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1.0e-6,
                "temporal encoding mismatch at {index}: {actual} != {expected}"
            );
        }
    }
}
