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

//! Multiplex spatial-memory encoding and object-pointer projection.

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::super::interpolate::resize_bilinear_nhwc;
use super::super::sam3::linear::UnifiedLinear;
use super::super::sam3::sam_components::Conv2dLayer;
use super::super::sam3::tracker::memory::{DownsampleConvBlock, MemoryFuser};
use super::config::TrackerConfig;
use super::multiplex::MultiplexState;
use super::state::TrackerState;
use super::tensor::add_scalar;
use super::tracker::MultiplexTracker;

pub(crate) struct ObjectPointerMlp {
    layers: [UnifiedLinear; 3],
}

impl ObjectPointerMlp {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            layers: [
                UnifiedLinear::from_weights(
                    weights,
                    &format!("{prefix}.layers.0"),
                    group_size,
                    bits,
                )?,
                UnifiedLinear::from_weights(
                    weights,
                    &format!("{prefix}.layers.1"),
                    group_size,
                    bits,
                )?,
                UnifiedLinear::from_weights(
                    weights,
                    &format!("{prefix}.layers.2"),
                    group_size,
                    bits,
                )?,
            ],
        })
    }

    pub(crate) fn forward(&self, input: &MlxArray) -> UniquePtr<MlxArray> {
        let hidden = mlxcel_core::relu(&self.layers[0].forward(input));
        let hidden = mlxcel_core::relu(&self.layers[1].forward(&hidden));
        self.layers[2].forward(&hidden)
    }
}

struct MaskDownsampler {
    input_size: usize,
    layers: Vec<DownsampleConvBlock>,
    final_conv: Conv2dLayer,
}

impl MaskDownsampler {
    fn from_weights(
        weights: &WeightMap,
        config: &TrackerConfig,
        prefix: &str,
    ) -> Result<Self, String> {
        let mut layers = Vec::with_capacity(4);
        for index in 0..4 {
            layers.push(DownsampleConvBlock::from_weights(
                weights,
                &format!("{prefix}.layers.{index}"),
                config.mask_downsampler_stride as i32,
                config.mask_downsampler_padding as i32,
            )?);
        }
        Ok(Self {
            input_size: config.mask_downsampler_input_size,
            layers,
            final_conv: Conv2dLayer::from_weights(
                weights,
                &format!("{prefix}.final_conv"),
                1,
                0,
                1,
                true,
            )?,
        })
    }

    fn forward(&self, masks: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let shape = mlxcel_core::array_shape(masks);
        if shape.len() != 4 || shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0 || shape[3] <= 0 {
            return Err(format!(
                "SAM 3.1 mask downsampler requires non-empty [B,H,W,C], got {shape:?}"
            ));
        }
        let mut output = if shape[1] != self.input_size as i32 || shape[2] != self.input_size as i32
        {
            resize_bilinear_nhwc(masks, (self.input_size, self.input_size), false, false)
                .map_err(|error| error.to_string())?
        } else {
            mlxcel_core::copy(masks)
        };
        for layer in &self.layers {
            output = layer.forward(&output);
        }
        Ok(self.final_conv.forward(&output))
    }
}

pub(crate) struct MultiplexMemoryEncoder {
    mask_downsampler: MaskDownsampler,
    fuser: MemoryFuser,
    feature_projection: Conv2dLayer,
}

impl MultiplexMemoryEncoder {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: &TrackerConfig,
        prefix: &str,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        Ok(Self {
            mask_downsampler: MaskDownsampler::from_weights(
                weights,
                config,
                &format!("{prefix}.mask_downsampler"),
            )?,
            fuser: MemoryFuser::from_parameters(
                weights,
                &format!("{prefix}.memory_fuser"),
                config.memory_fuser_num_layers,
                config.memory_fuser_embed_dim as i32,
                config.memory_fuser_padding as i32,
                group_size,
                bits,
            )?,
            feature_projection: Conv2dLayer::from_weights(
                weights,
                &format!("{prefix}.feature_projection"),
                1,
                0,
                1,
                true,
            )?,
        })
    }

    pub(crate) fn forward(
        &self,
        image_features: &MlxArray,
        mask_channels: &MlxArray,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let masks = self.mask_downsampler.forward(mask_channels)?;
        let projected = self.feature_projection.forward(image_features);
        let fused = self.fuser.forward(&mlxcel_core::add(&projected, &masks));
        Ok(fused)
    }
}

pub(crate) fn encode_mux_masks(
    state: &MultiplexState,
    masks_nchw: &MlxArray,
    conditioning: &[usize],
    config: &TrackerConfig,
) -> Result<UniquePtr<MlxArray>, String> {
    let shape = mlxcel_core::array_shape(masks_nchw);
    if shape.len() != 4 || shape[1] != 1 {
        return Err(format!(
            "SAM 3.1 memory masks must have [N,1,H,W], got {shape:?}"
        ));
    }
    let object_masks = mlxcel_core::squeeze_axis(masks_nchw, 1);
    let muxed = state.mux(&object_masks)?;
    let mut conditioning_values = vec![config.condition_as_mask_input_bg; state.num_objects()];
    for &index in conditioning {
        if index >= conditioning_values.len() {
            return Err(format!(
                "SAM 3.1 conditioning object {index} exceeds {} objects",
                conditioning_values.len()
            ));
        }
        conditioning_values[index] = config.condition_as_mask_input_fg;
    }
    let values =
        mlxcel_core::from_slice_f32(&conditioning_values, &[state.num_objects() as i32, 1, 1]);
    let embedded =
        mlxcel_core::broadcast_to(&values, &[state.num_objects() as i32, shape[2], shape[3]]);
    let conditioning = state.mux(&embedded)?;
    let channels = mlxcel_core::concatenate(&muxed, &conditioning, 1);
    Ok(mlxcel_core::transpose_axes(&channels, &[0, 2, 3, 1]))
}

impl MultiplexTracker {
    pub(super) fn apply_no_object_pointer(
        &self,
        pointer: &MlxArray,
        appearing: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        if !self.config.pred_obj_scores || !self.config.use_no_obj_ptr {
            return mlxcel_core::copy(pointer);
        }
        let lambda = mlxcel_core::astype(appearing, mlxcel_core::dtype::FLOAT32);
        if self.config.use_linear_no_obj_ptr {
            mlxcel_core::add(
                &mlxcel_core::multiply(&lambda, pointer),
                &mlxcel_core::multiply(
                    &mlxcel_core::subtract(&mlxcel_core::ones_like(&lambda), &lambda),
                    &self.no_object_pointer.forward(pointer),
                ),
            )
        } else if self.config.fixed_no_obj_ptr {
            mlxcel_core::multiply(&lambda, pointer)
        } else {
            mlxcel_core::copy(pointer)
        }
    }

    pub(super) fn encode_memory(
        &self,
        state: &TrackerState,
        image_features: &MlxArray,
        mask_logits: &MlxArray,
        object_scores: &MlxArray,
        conditioning: &[usize],
    ) -> Result<UniquePtr<MlxArray>, String> {
        let masks = if self.config.apply_sigmoid_to_mask_logits_for_mem_enc {
            add_scalar(
                &mlxcel_core::multiply_scalar(
                    &mlxcel_core::sigmoid(mask_logits),
                    self.config.sigmoid_scale_for_mem_enc,
                ),
                self.config.sigmoid_bias_for_mem_enc,
            )
        } else {
            mlxcel_core::copy(mask_logits)
        };
        let channels = encode_mux_masks(&state.multiplex, &masks, conditioning, &self.config)?;
        let memory = self.memory_encoder.forward(image_features, &channels)?;
        let appearing = mlxcel_core::astype(
            &mlxcel_core::greater(
                object_scores,
                &mlxcel_core::full_f32(
                    &[1],
                    self.config.object_score_logit_threshold,
                    mlxcel_core::array_dtype(object_scores),
                ),
            ),
            mlxcel_core::dtype::FLOAT32,
        );
        let appearing = state.multiplex.mux(&appearing)?;
        let missing = mlxcel_core::multiply(
            &mlxcel_core::subtract(&mlxcel_core::ones_like(&appearing), &appearing),
            &mlxcel_core::expand_dims(&self.no_object_spatial, 0),
        );
        let missing = mlxcel_core::sum_axis(&missing, 1, false);
        Ok(mlxcel_core::add(
            &memory,
            &mlxcel_core::expand_dims_multi(&missing, &[1, 2]),
        ))
    }
}
