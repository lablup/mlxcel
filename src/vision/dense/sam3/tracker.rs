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

//! SAM2-style spatial-memory tracker used by SAM 3 video propagation.

mod memory;

pub use memory::{MemoryAttention, MemoryEncoder};

use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::TrackerConfig;
use super::sam_components::{
    Conv2dLayer, PointPrompts, SamMaskDecoder, SamPromptEncoder, load_array,
};
use memory::nearest_resize_nhwc;

struct ObjectPointerMlp {
    proj_in: UnifiedLinear,
    layers: Vec<UnifiedLinear>,
    proj_out: UnifiedLinear,
}

impl ObjectPointerMlp {
    fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            proj_in: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.proj_in"),
                group_size,
                bits,
            )?,
            layers: vec![UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.layers.0"),
                group_size,
                bits,
            )?],
            proj_out: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.proj_out"),
                group_size,
                bits,
            )?,
        })
    }

    fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let mut x = mlxcel_core::relu(&self.proj_in.forward(x));
        for layer in &self.layers {
            x = mlxcel_core::relu(&layer.forward(&x));
        }
        self.proj_out.forward(&x)
    }
}

struct SharedImageEmbedding {
    positional_embedding: UniquePtr<MlxArray>,
}

impl SharedImageEmbedding {
    fn from_weights(weights: &WeightMap, prefix: &str) -> Result<Self, String> {
        Ok(Self {
            positional_embedding: load_array(weights, &format!("{prefix}.positional_embedding"))?,
        })
    }

    fn forward(&self, size: (i32, i32)) -> UniquePtr<MlxArray> {
        let mut coordinates = Vec::with_capacity((size.0 * size.1 * 2) as usize);
        for y in 0..size.0 {
            for x in 0..size.1 {
                coordinates.push(2.0 * x as f32 / size.1 as f32 - 1.0);
                coordinates.push(2.0 * y as f32 / size.0 as f32 - 1.0);
            }
        }
        let coordinates = mlxcel_core::from_slice_f32(&coordinates, &[size.0 * size.1, 2]);
        let angles = mlxcel_core::multiply_scalar(
            &mlxcel_core::matmul(&coordinates, &self.positional_embedding),
            2.0 * std::f32::consts::PI,
        );
        mlxcel_core::concatenate(&mlxcel_core::sin(&angles), &mlxcel_core::cos(&angles), -1)
    }
}

/// One propagation result plus the memory state needed by later frames.
pub struct TrackStepOutput {
    pub pred_masks: UniquePtr<MlxArray>,
    pub iou_scores: UniquePtr<MlxArray>,
    pub object_scores: UniquePtr<MlxArray>,
    pub object_pointer: UniquePtr<MlxArray>,
    pub memory: UniquePtr<MlxArray>,
}

pub struct TrackerModel {
    pub prompt_encoder: SamPromptEncoder,
    pub mask_decoder: SamMaskDecoder,
    pub memory_attention: MemoryAttention,
    pub memory_encoder: MemoryEncoder,
    object_pointer_proj: ObjectPointerMlp,
    temporal_positional_encoding_projection_layer: UnifiedLinear,
    mask_downsample: Conv2dLayer,
    shared_image_embedding: SharedImageEmbedding,
    no_memory_embedding: UniquePtr<MlxArray>,
    no_memory_positional_encoding: UniquePtr<MlxArray>,
    no_object_pointer: UniquePtr<MlxArray>,
    memory_temporal_positional_encoding: UniquePtr<MlxArray>,
    occlusion_spatial_embedding_parameter: Option<UniquePtr<MlxArray>>,
    hidden_size: i32,
    memory_dim: i32,
    num_maskmem: usize,
    max_object_pointers: usize,
    mask_downsampler_total_stride: i32,
}

impl TrackerModel {
    pub fn from_weights(
        weights: &WeightMap,
        config: &TrackerConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let prefix = "tracker_model";
        Ok(Self {
            prompt_encoder: SamPromptEncoder::from_weights(
                weights,
                &config.prompt_encoder_config,
                &format!("{prefix}.prompt_encoder"),
            )?,
            mask_decoder: SamMaskDecoder::from_weights(
                weights,
                &config.mask_decoder_config,
                &format!("{prefix}.mask_decoder"),
                group_size,
                bits,
            )?,
            memory_attention: MemoryAttention::from_weights(
                weights,
                config,
                &format!("{prefix}.memory_attention"),
                group_size,
                bits,
            )?,
            memory_encoder: MemoryEncoder::from_weights(
                weights,
                config,
                &format!("{prefix}.memory_encoder"),
                group_size,
                bits,
            )?,
            object_pointer_proj: ObjectPointerMlp::from_weights(
                weights,
                &format!("{prefix}.object_pointer_proj"),
                group_size,
                bits,
            )?,
            temporal_positional_encoding_projection_layer: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.temporal_positional_encoding_projection_layer"),
                group_size,
                bits,
            )?,
            mask_downsample: Conv2dLayer::from_weights(
                weights,
                &format!("{prefix}.mask_downsample"),
                4,
                0,
                1,
                true,
            )?,
            shared_image_embedding: SharedImageEmbedding::from_weights(
                weights,
                &format!("{prefix}.shared_image_embedding"),
            )?,
            no_memory_embedding: load_array(weights, &format!("{prefix}.no_memory_embedding"))?,
            no_memory_positional_encoding: load_array(
                weights,
                &format!("{prefix}.no_memory_positional_encoding"),
            )?,
            no_object_pointer: load_array(weights, &format!("{prefix}.no_object_pointer"))?,
            memory_temporal_positional_encoding: load_array(
                weights,
                &format!("{prefix}.memory_temporal_positional_encoding"),
            )?,
            occlusion_spatial_embedding_parameter: config
                .enable_occlusion_spatial_embedding
                .then(|| {
                    load_array(
                        weights,
                        &format!("{prefix}.occlusion_spatial_embedding_parameter"),
                    )
                })
                .transpose()?,
            hidden_size: config.memory_attention_hidden_size as i32,
            memory_dim: config.memory_encoder_output_channels as i32,
            num_maskmem: config.num_maskmem as usize,
            max_object_pointers: config.max_object_pointers_in_encoder as usize,
            mask_downsampler_total_stride: config.mask_downsampler_total_stride as i32,
        })
    }

    pub fn encode_image(&self, features: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let shape = mlxcel_core::array_shape(features);
        if shape.len() != 4 || shape[3] != self.hidden_size {
            return Err(format!(
                "SAM3 tracker expected NHWC features with {} channels, got {shape:?}",
                self.hidden_size
            ));
        }
        Ok(mlxcel_core::reshape(
            features,
            &[shape[0], shape[1] * shape[2], shape[3]],
        ))
    }

    #[must_use]
    pub fn shared_image_position_encoding(&self, size: (i32, i32)) -> UniquePtr<MlxArray> {
        self.shared_image_embedding.forward(size)
    }

    #[must_use]
    pub fn downsample_mask(&self, mask: &MlxArray) -> UniquePtr<MlxArray> {
        self.mask_downsample.forward(mask)
    }

    fn learned_memory_position(&self, age: usize, batch: i32, tokens: i32) -> UniquePtr<MlxArray> {
        let age = age.min(self.num_maskmem.saturating_sub(1)) as i32;
        let position = mlxcel_core::slice(
            &self.memory_temporal_positional_encoding,
            &[age, 0, 0, 0],
            &[age + 1, 1, 1, self.memory_dim],
        );
        mlxcel_core::broadcast_to(
            &mlxcel_core::reshape(&position, &[1, 1, self.memory_dim]),
            &[batch, tokens, self.memory_dim],
        )
    }

    fn pointer_temporal_encoding(&self, count: i32) -> UniquePtr<MlxArray> {
        let mut encoding = vec![0.0_f32; (count * self.hidden_size) as usize];
        for position in 0..count {
            for pair in 0..self.hidden_size / 2 {
                let angle = position as f32
                    / 10_000.0_f32.powf((2 * pair) as f32 / self.hidden_size as f32);
                let base = (position * self.hidden_size + 2 * pair) as usize;
                encoding[base] = angle.sin();
                encoding[base + 1] = angle.cos();
            }
        }
        mlxcel_core::from_slice_f32(&encoding, &[1, count, self.hidden_size])
    }

    fn prepare_memory(
        &self,
        memory_bank: &[&MlxArray],
        memory_positions: &[&MlxArray],
        object_pointers: &[&MlxArray],
    ) -> Result<Option<UniquePtr<MlxArray>>, String> {
        if memory_bank.is_empty() && object_pointers.is_empty() {
            return Ok(None);
        }
        let original_memory_count = memory_bank.len();
        let bank_offset = original_memory_count.saturating_sub(self.num_maskmem);
        let memory_bank = if memory_bank.len() > self.num_maskmem {
            &memory_bank[memory_bank.len() - self.num_maskmem..]
        } else {
            memory_bank
        };
        let mut pieces = Vec::with_capacity(memory_bank.len() + 1);
        for (index, memory) in memory_bank.iter().enumerate() {
            let shape = mlxcel_core::array_shape(memory);
            if shape.len() != 3 || shape[2] != self.memory_dim {
                return Err(format!(
                    "SAM3 memory entry must be [batch, tokens, {}], got {shape:?}",
                    self.memory_dim
                ));
            }
            let position_index =
                index + (memory_positions.len() == original_memory_count) as usize * bank_offset;
            let position = if let Some(position) = memory_positions.get(position_index) {
                mlxcel_core::broadcast_to(position, &shape)
            } else {
                self.learned_memory_position(memory_bank.len() - 1 - index, shape[0], shape[1])
            };
            pieces.push(mlxcel_core::add(memory, &position));
        }

        let pointer_count = object_pointers.len().min(self.max_object_pointers);
        if pointer_count > 0 {
            let pointers = &object_pointers[object_pointers.len() - pointer_count..];
            let first_shape = mlxcel_core::array_shape(pointers[0]);
            let batch = first_shape[0];
            let mut normalized = Vec::with_capacity(pointer_count);
            for pointer in pointers {
                normalized.push(mlxcel_core::reshape(pointer, &[batch, 1, self.hidden_size]));
            }
            let refs: Vec<&MlxArray> = normalized.iter().map(|value| &**value).collect();
            let pointers = mlxcel_core::concatenate_many(&refs, 1);
            if self.hidden_size % self.memory_dim != 0 {
                return Err(format!(
                    "SAM3 pointer width {} is not divisible by memory width {}",
                    self.hidden_size, self.memory_dim
                ));
            }
            let split = self.hidden_size / self.memory_dim;
            let pointer_tokens = mlxcel_core::reshape(
                &pointers,
                &[batch, pointer_count as i32 * split, self.memory_dim],
            );
            let temporal = self
                .temporal_positional_encoding_projection_layer
                .forward(&self.pointer_temporal_encoding(pointer_count as i32));
            let temporal =
                mlxcel_core::reshape(&temporal, &[1, pointer_count as i32, 1, self.memory_dim]);
            let temporal = mlxcel_core::broadcast_to(
                &temporal,
                &[batch, pointer_count as i32, split, self.memory_dim],
            );
            let temporal = mlxcel_core::reshape(
                &temporal,
                &[batch, pointer_count as i32 * split, self.memory_dim],
            );
            pieces.push(mlxcel_core::add(&pointer_tokens, &temporal));
        }
        let refs: Vec<&MlxArray> = pieces.iter().map(|value| &**value).collect();
        Ok(Some(mlxcel_core::concatenate_many(&refs, 1)))
    }

    /// Run one tracker frame with temporal memory positions and object pointers.
    #[allow(clippy::too_many_arguments)]
    pub fn track_step(
        &self,
        current_features: &MlxArray,
        memory_bank: &[&MlxArray],
        memory_positions: &[&MlxArray],
        object_pointers: &[&MlxArray],
        prompt_points: Option<PointPrompts<'_>>,
        prompt_boxes: Option<&MlxArray>,
        prompt_masks: Option<&MlxArray>,
        multimask_output: bool,
        high_res_features: Option<&[&MlxArray]>,
    ) -> Result<TrackStepOutput, String> {
        let memory = self.prepare_memory(memory_bank, memory_positions, object_pointers)?;
        self.decode_step(
            current_features,
            memory.as_deref(),
            prompt_points,
            prompt_boxes,
            prompt_masks,
            multimask_output,
            high_res_features,
        )
    }

    /// Match mlx-vlm propagation: direct memory concatenation, without temporal
    /// positions or pointer tokens. Callers maintain its binary-mask bank via
    /// `MemoryEncoder::forward`, as `Sam3VideoPredictor._update_memory` does.
    #[allow(clippy::too_many_arguments)]
    pub fn track_step_upstream(
        &self,
        current_features: &MlxArray,
        memory_bank: &[&MlxArray],
        prompt_points: Option<PointPrompts<'_>>,
        prompt_boxes: Option<&MlxArray>,
        prompt_masks: Option<&MlxArray>,
        multimask_output: bool,
        high_res_features: Option<&[&MlxArray]>,
    ) -> Result<TrackStepOutput, String> {
        let retained = if memory_bank.len() > self.num_maskmem {
            &memory_bank[memory_bank.len() - self.num_maskmem..]
        } else {
            memory_bank
        };
        let memory = (!retained.is_empty()).then(|| mlxcel_core::concatenate_many(retained, 1));
        self.decode_step(
            current_features,
            memory.as_deref(),
            prompt_points,
            prompt_boxes,
            prompt_masks,
            multimask_output,
            high_res_features,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_step(
        &self,
        current_features: &MlxArray,
        memory: Option<&MlxArray>,
        prompt_points: Option<PointPrompts<'_>>,
        prompt_boxes: Option<&MlxArray>,
        prompt_masks: Option<&MlxArray>,
        multimask_output: bool,
        high_res_features: Option<&[&MlxArray]>,
    ) -> Result<TrackStepOutput, String> {
        let feature_shape = mlxcel_core::array_shape(current_features);
        let (batch, height, width) = match feature_shape.as_slice() {
            [batch, height, width, channels] if *channels == self.hidden_size => {
                (*batch, *height, *width)
            }
            _ => {
                return Err(format!(
                    "SAM3 track_step expected NHWC features with {} channels, got {feature_shape:?}",
                    self.hidden_size
                ));
            }
        };
        let mut src = self.encode_image(current_features)?;
        if let Some(memory) = memory {
            src = self.memory_attention.forward(&src, memory);
        }
        let image_pe = mlxcel_core::broadcast_to(
            &self.prompt_encoder.dense_position_encoding(),
            &[batch, height * width, self.hidden_size],
        );
        let (mut sparse, mut dense) =
            self.prompt_encoder
                .forward(prompt_points, prompt_boxes, prompt_masks)?;
        let prompt_batch = mlxcel_core::array_shape(&dense)[0];
        if prompt_batch == 1 && batch != 1 && prompt_points.is_none() && prompt_boxes.is_none() {
            let sparse_shape = mlxcel_core::array_shape(&sparse);
            sparse = mlxcel_core::broadcast_to(&sparse, &[batch, sparse_shape[1], sparse_shape[2]]);
            dense = mlxcel_core::broadcast_to(&dense, &[batch, height * width, self.hidden_size]);
        } else if prompt_batch != batch {
            return Err(format!(
                "SAM3 prompt batch {prompt_batch} does not match feature batch {batch}"
            ));
        }
        let decoded = self.mask_decoder.forward(
            &src,
            &image_pe,
            &sparse,
            &dense,
            multimask_output,
            high_res_features,
        )?;
        let token_shape = mlxcel_core::array_shape(&decoded.sam_tokens);
        let first_token = mlxcel_core::reshape(
            &mlxcel_core::slice(&decoded.sam_tokens, &[0, 0, 0], &[batch, 1, token_shape[2]]),
            &[batch, token_shape[2]],
        );
        let object_pointer = self.object_pointer_proj.forward(&first_token);
        let mask_shape = mlxcel_core::array_shape(&decoded.masks);
        let mask = mlxcel_core::slice(
            &decoded.masks,
            &[0, 0, 0, 0],
            &[batch, 1, mask_shape[2], mask_shape[3]],
        );
        let mask = mlxcel_core::transpose_axes(&mask, &[0, 2, 3, 1]);
        let mask = nearest_resize_nhwc(
            &mask,
            height * self.mask_downsampler_total_stride,
            width * self.mask_downsampler_total_stride,
        );
        let memory = self.memory_encoder.forward(current_features, &mask)?;
        let memory_shape = mlxcel_core::array_shape(&memory);
        let memory = mlxcel_core::reshape(
            &memory,
            &[
                memory_shape[0],
                memory_shape[1] * memory_shape[2],
                memory_shape[3],
            ],
        );
        Ok(TrackStepOutput {
            pred_masks: decoded.masks,
            iou_scores: decoded.iou_scores,
            object_scores: decoded.object_score,
            object_pointer,
            memory,
        })
    }

    pub fn initial_state(&self) -> (&MlxArray, &MlxArray, &MlxArray, Option<&MlxArray>) {
        (
            &self.no_memory_embedding,
            &self.no_memory_positional_encoding,
            &self.no_object_pointer,
            self.occlusion_spatial_embedding_parameter.as_deref(),
        )
    }
}
