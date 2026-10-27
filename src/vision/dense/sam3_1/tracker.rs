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

//! SAM 3.1 mask initialization and multiplex propagation.

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::super::interpolate::resize_bilinear_nhwc;
use super::super::sam3::linear::UnifiedLinear;
use super::super::sam3::position::PositionEmbeddingSine;
use super::super::sam3::sam_components::{
    Conv2dLayer, PointPrompts, PositionEmbedding, SamPromptEncoder, load_array,
};
use super::attention::MemoryAttention;
use super::config::TrackerConfig;
use super::decoder::{DecoderOutput, MultiplexMaskDecoder};
use super::memory::{MultiplexMemoryEncoder, ObjectPointerMlp};
use super::state::{BranchFeatures, FrameFeatures, FrameOutput, TrackPrediction, TrackerState};
use super::tensor::{add_scalar, gather_mask_candidate, mask_presence, resize_masks_nchw};

const NO_OBJECT_SCORE: f32 = -1024.0;

pub(crate) struct MultiplexTracker {
    pub(super) config: TrackerConfig,
    interactive_prompt_encoder: SamPromptEncoder,
    interactive_decoder: MultiplexMaskDecoder,
    propagation_decoder: MultiplexMaskDecoder,
    pub(super) memory_attention: MemoryAttention,
    pub(super) memory_encoder: MultiplexMemoryEncoder,
    object_pointer: ObjectPointerMlp,
    interactive_object_pointer: ObjectPointerMlp,
    pub(super) temporal_memory_position: UniquePtr<MlxArray>,
    pub(super) temporal_position_projection: UnifiedLinear,
    output_valid: UniquePtr<MlxArray>,
    output_invalid: UniquePtr<MlxArray>,
    pub(super) no_object_spatial: UniquePtr<MlxArray>,
    pub(super) no_object_pointer: UnifiedLinear,
    interactivity_no_memory: UniquePtr<MlxArray>,
    image_position: PositionEmbedding,
    interactive_mask_downsample: Conv2dLayer,
    pub(super) feature_position: PositionEmbeddingSine,
}

impl MultiplexTracker {
    pub(crate) fn from_weights(
        weights: &WeightMap,
        config: TrackerConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let root = "tracker_model";
        let mut interactive_config = config.mask_decoder_config.clone();
        interactive_config.multiplex_count = 1;
        interactive_config.multimask_outputs_only = false;
        interactive_config.use_multimask_token_for_obj_ptr = true;
        Ok(Self {
            interactive_prompt_encoder: SamPromptEncoder::from_weights(
                weights,
                &config.prompt_encoder_config,
                &format!("{root}.interactive_sam_prompt_encoder"),
                group_size,
                bits,
            )?,
            interactive_decoder: MultiplexMaskDecoder::from_weights(
                weights,
                &interactive_config,
                &format!("{root}.interactive_sam_mask_decoder"),
                group_size,
                bits,
            )?,
            propagation_decoder: MultiplexMaskDecoder::from_weights(
                weights,
                &config.mask_decoder_config,
                &format!("{root}.sam_mask_decoder"),
                group_size,
                bits,
            )?,
            memory_attention: MemoryAttention::from_weights(
                weights,
                &config,
                &format!("{root}.memory_attention"),
                group_size,
                bits,
            )?,
            memory_encoder: MultiplexMemoryEncoder::from_weights(
                weights,
                &config,
                &format!("{root}.memory_encoder"),
                group_size,
                bits,
            )?,
            object_pointer: ObjectPointerMlp::from_weights(
                weights,
                &format!("{root}.obj_ptr_proj"),
                group_size,
                bits,
            )?,
            interactive_object_pointer: ObjectPointerMlp::from_weights(
                weights,
                &format!("{root}.interactive_obj_ptr_proj"),
                group_size,
                bits,
            )?,
            temporal_memory_position: load_array(
                weights,
                &format!("{root}.memory_temporal_positional_encoding"),
            )?,
            temporal_position_projection: UnifiedLinear::from_weights(
                weights,
                &format!("{root}.temporal_positional_encoding_projection_layer"),
                group_size,
                bits,
            )?,
            output_valid: load_array(weights, &format!("{root}.output_valid_embed"))?,
            output_invalid: load_array(weights, &format!("{root}.output_invalid_embed"))?,
            no_object_spatial: load_array(weights, &format!("{root}.no_obj_embed_spatial"))?,
            no_object_pointer: UnifiedLinear::from_weights(
                weights,
                &format!("{root}.no_obj_ptr_linear"),
                group_size,
                bits,
            )?,
            interactivity_no_memory: load_array(
                weights,
                &format!("{root}.interactivity_no_mem_embed"),
            )?,
            image_position: PositionEmbedding::from_weights(
                weights,
                &format!("{root}.image_pe_layer"),
            )?,
            interactive_mask_downsample: Conv2dLayer::from_weights(
                weights,
                &format!("{root}.interactive_mask_downsample"),
                4,
                0,
                1,
                true,
            )?,
            feature_position: PositionEmbeddingSine::new(config.memory_attention_hidden_size / 2),
            config,
        })
    }

    pub(crate) fn prepare_features(
        &self,
        interactive_fpn: Option<Vec<UniquePtr<MlxArray>>>,
        propagation_fpn: Vec<UniquePtr<MlxArray>>,
    ) -> Result<FrameFeatures, String> {
        let propagation = self.prepare_branch(&self.propagation_decoder, propagation_fpn, true)?;
        let interactive = interactive_fpn
            .map(|fpn| self.prepare_branch(&self.interactive_decoder, fpn, false))
            .transpose()?;
        Ok(FrameFeatures {
            interactive,
            propagation,
        })
    }

    fn prepare_branch(
        &self,
        decoder: &MultiplexMaskDecoder,
        fpn: Vec<UniquePtr<MlxArray>>,
        with_position: bool,
    ) -> Result<BranchFeatures, String> {
        let vision = fpn
            .last()
            .map(|value| mlxcel_core::copy(value))
            .ok_or_else(|| "SAM 3.1 tracker received an empty FPN".to_owned())?;
        let high_resolution = decoder.project_high_resolution(&fpn)?;
        let position = with_position.then(|| self.feature_position.forward(&vision));
        Ok(BranchFeatures {
            vision,
            high_resolution,
            position,
        })
    }

    pub(crate) fn init_state(&self, num_objects: usize) -> Result<TrackerState, String> {
        TrackerState::new(num_objects)
    }

    pub(crate) fn track_init(
        &self,
        state: &mut TrackerState,
        features: &FrameFeatures,
        masks: &MlxArray,
    ) -> Result<TrackPrediction, String> {
        let interactive = features
            .interactive
            .as_ref()
            .ok_or_else(|| "SAM 3.1 initialization needs interactive features".to_owned())?;
        let shape = mlxcel_core::array_shape(masks);
        let masks = match shape.as_slice() {
            [_, _, _] => mlxcel_core::expand_dims(masks, 1),
            [_, 1, _, _] => mlxcel_core::copy(masks),
            _ => {
                return Err(format!(
                    "SAM 3.1 initialization masks must have [N,H,W] or [N,1,H,W], got {shape:?}"
                ));
            }
        };
        if mlxcel_core::array_shape(&masks)[0] != state.num_objects() as i32 {
            return Err("SAM 3.1 initialization mask count does not match the state".into());
        }
        let float_masks = mlxcel_core::astype(&masks, mlxcel_core::dtype::FLOAT32);
        let high_masks = add_scalar(&mlxcel_core::multiply_scalar(&float_masks, 20.0), -10.0);
        let pointer_and_score = self.interactive_pointer(interactive, &float_masks)?;
        let decoder_appearing = mlxcel_core::greater(
            &pointer_and_score.1,
            &mlxcel_core::full_f32(
                &[1],
                self.config.object_score_logit_threshold,
                mlxcel_core::array_dtype(&pointer_and_score.1),
            ),
        );
        let pointer = self.apply_no_object_pointer(&pointer_and_score.0, &decoder_appearing);
        let appearing = mask_presence(&float_masks);
        let pointer = self.apply_no_object_pointer(&pointer, &appearing);
        let object_scores = add_scalar(&mlxcel_core::multiply_scalar(&appearing, 20.0), -10.0);
        let conditioning: Vec<usize> = (0..state.num_objects()).collect();
        let memory = self.encode_memory(
            state,
            &features.propagation.vision,
            &high_masks,
            &object_scores,
            &conditioning,
        )?;
        let output = FrameOutput {
            object_pointers: state.multiplex.mux(&pointer)?,
            memory_features: Some(memory),
            image_features: mlxcel_core::copy(&features.propagation.vision),
        };
        state.insert_conditioning(0, output);
        Ok(TrackPrediction {
            masks: high_masks,
            object_score_logits: object_scores,
        })
    }

    fn interactive_pointer(
        &self,
        features: &BranchFeatures,
        masks: &MlxArray,
    ) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>), String> {
        let shape = mlxcel_core::array_shape(masks);
        let nhwc = mlxcel_core::transpose_axes(masks, &[0, 2, 3, 1]);
        let prompt = self.interactive_mask_downsample.forward(&nhwc);
        let prompt = resize_bilinear_nhwc(&prompt, (288, 288), false, true)
            .map_err(|error| error.to_string())?;
        let coordinates = mlxcel_core::zeros(&[shape[0], 2, 2], mlxcel_core::dtype::FLOAT32);
        let labels = mlxcel_core::full_f32(&[shape[0], 2], -1.0, mlxcel_core::dtype::INT32);
        let (sparse, dense) = self.interactive_prompt_encoder.forward(
            Some(PointPrompts {
                coordinates: &coordinates,
                labels: &labels,
            }),
            None,
            Some(&prompt),
        )?;
        let vision_shape = mlxcel_core::array_shape(&features.vision);
        let vision = mlxcel_core::add(&features.vision, &self.interactivity_no_memory);
        let vision = mlxcel_core::reshape(
            &vision,
            &[
                vision_shape[0],
                vision_shape[1] * vision_shape[2],
                vision_shape[3],
            ],
        );
        let decoded = self.interactive_decoder.forward(
            &vision,
            &self.interactive_prompt_encoder.dense_position_encoding(),
            &features.high_resolution,
            None,
            Some(&sparse),
            Some(&dense),
            false,
        )?;
        let tokens = mlxcel_core::reshape(
            &decoded.sam_tokens,
            &[shape[0], self.config.memory_attention_hidden_size as i32],
        );
        Ok((
            self.interactive_object_pointer.forward(&tokens),
            mlxcel_core::reshape(&decoded.object_scores, &[shape[0], 1]),
        ))
    }

    pub(crate) fn track_step(
        &self,
        state: &mut TrackerState,
        frame_index: usize,
        total_frames: usize,
        features: &FrameFeatures,
    ) -> Result<TrackPrediction, String> {
        let conditioned =
            self.memory_conditioned_features(state, frame_index, total_frames, features)?;
        let decoded = self.propagate_heads(state, features, &conditioned)?;
        let (output, prediction) = self.select_and_encode(state, features, decoded)?;
        let lookback = self
            .config
            .max_object_pointers_in_encoder
            .max(2 + (self.config.num_maskmem - 2) * self.config.memory_temporal_stride_for_eval);
        state.insert_propagated(frame_index, output, lookback);
        Ok(prediction)
    }

    fn propagate_heads(
        &self,
        state: &TrackerState,
        features: &FrameFeatures,
        conditioned: &MlxArray,
    ) -> Result<DecoderOutput, String> {
        let shape = mlxcel_core::array_shape(conditioned);
        let image = mlxcel_core::reshape(conditioned, &[shape[0], shape[1] * shape[2], shape[3]]);
        let valid =
            mlxcel_core::astype(&state.multiplex.valid_slots(), mlxcel_core::dtype::FLOAT32);
        let valid = mlxcel_core::expand_dims(&valid, -1);
        let extra = mlxcel_core::add(
            &mlxcel_core::multiply(&valid, &mlxcel_core::expand_dims(&self.output_valid, 0)),
            &mlxcel_core::multiply(
                &mlxcel_core::subtract(&mlxcel_core::ones_like(&valid), &valid),
                &mlxcel_core::expand_dims(&self.output_invalid, 0),
            ),
        );
        let side = self.config.mask_downsampler_input_size as i32 / 16;
        let pe = mlxcel_core::expand_dims(&self.image_position.grid((side, side)), 0);
        self.propagation_decoder.forward(
            &image,
            &pe,
            &features.propagation.high_resolution,
            Some(&extra),
            None,
            None,
            true,
        )
    }

    fn select_and_encode(
        &self,
        state: &TrackerState,
        features: &FrameFeatures,
        decoded: DecoderOutput,
    ) -> Result<(FrameOutput, TrackPrediction), String> {
        let mut low = state.multiplex.demux(&decoded.masks)?;
        let iou = state.multiplex.demux(&decoded.iou_scores)?;
        let scores = state.multiplex.demux(&decoded.object_scores)?;
        let tokens = state.multiplex.demux(&decoded.sam_tokens)?;
        let appearing = mlxcel_core::greater(
            &scores,
            &mlxcel_core::full_f32(
                &[1],
                self.config.object_score_logit_threshold,
                mlxcel_core::array_dtype(&scores),
            ),
        );
        let low_shape = mlxcel_core::array_shape(&low);
        let appear_mask = mlxcel_core::broadcast_to(
            &mlxcel_core::expand_dims_multi(&appearing, &[2, 3]),
            &low_shape,
        );
        low = mlxcel_core::where_cond(
            &appear_mask,
            &low,
            &mlxcel_core::full_f32(&low_shape, NO_OBJECT_SCORE, mlxcel_core::array_dtype(&low)),
        );
        let high = resize_masks_nchw(
            &low,
            (self.config.image_size, self.config.image_size),
            false,
        )?;
        let best = mlxcel_core::argmax(&iou, -1, false);
        let best_high = gather_mask_candidate(&high, &best);
        let token_shape = mlxcel_core::array_shape(&tokens);
        let token_indices = mlxcel_core::broadcast_to(
            &mlxcel_core::expand_dims_multi(&best, &[1, 2]),
            &[token_shape[0], 1, token_shape[2]],
        );
        let best_token =
            mlxcel_core::squeeze_axis(&mlxcel_core::take_along_axis(&tokens, &token_indices, 1), 1);
        let pointer = self.object_pointer.forward(&best_token);
        let appearing_f32 = mlxcel_core::astype(&appearing, mlxcel_core::dtype::FLOAT32);
        let pointer = self.apply_no_object_pointer(&pointer, &appearing_f32);
        let memory = self.encode_memory(
            state,
            &features.propagation.vision,
            &best_high,
            &scores,
            &[],
        )?;
        let prediction = TrackPrediction {
            masks: best_high,
            object_score_logits: mlxcel_core::astype(&scores, mlxcel_core::dtype::FLOAT32),
        };
        Ok((
            FrameOutput {
                object_pointers: state.multiplex.mux(&pointer)?,
                memory_features: Some(memory),
                image_features: mlxcel_core::copy(&features.propagation.vision),
            },
            prediction,
        ))
    }
}
