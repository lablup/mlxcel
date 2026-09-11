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

//! MiniCPM-o vision-language model.
//!
//! MiniCPM-o uses standard Qwen3 (NOT Qwen3-VL) as the text backbone, meaning
//! standard RoPE rather than interleaved MRoPE. `mlxcel` currently exposes the
//! image path; the upstream audio tower remains loaded from checkpoints but is
//! not wired into the CLI/server because the runtime has no audio input surface
//! yet.

use crate::LanguageModel;
use crate::vision::merge::InputEmbeddings;
use crate::vision::{encoders, processors};
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct MiniCPMOVLModel {
    pub text_model: crate::models::Qwen3Model,
    pub vision_tower: encoders::minicpmo::MiniCPMOVisionModel,
    pub resampler: encoders::minicpmo::MiniCPMOResampler,
    pub processor: processors::minicpmo::MiniCPMOProcessor,
    pub eos_token_ids: Vec<i32>,
}

impl MiniCPMOVLModel {
    fn fit_feature_span(
        feature_span: UniquePtr<MlxArray>,
        target_len: usize,
        hidden_size: i32,
    ) -> UniquePtr<MlxArray> {
        let current_len = mlxcel_core::array_shape(&feature_span)[1] as usize;
        if current_len == target_len {
            return feature_span;
        }

        if current_len > target_len {
            return mlxcel_core::slice(
                &feature_span,
                &[0, 0, 0],
                &[1, target_len as i32, hidden_size],
            );
        }

        let padding = mlxcel_core::zeros(
            &[
                1,
                (target_len.saturating_sub(current_len)) as i32,
                hidden_size,
            ],
            mlxcel_core::array_dtype(&feature_span),
        );
        mlxcel_core::concatenate(&feature_span, &padding, 1)
    }

    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        processed_images: &[processors::minicpmo::MiniCPMOImageInput],
        image_bounds: &[(usize, usize)],
    ) -> InputEmbeddings {
        let inputs_embeds = self.text_model.get_embed_tokens(input_ids);

        if processed_images.is_empty() || image_bounds.is_empty() {
            return InputEmbeddings {
                inputs_embeds,
                attention_mask_4d: None,
            };
        }

        let hidden_size = mlxcel_core::array_shape(&inputs_embeds)[2];
        let mut segments: Vec<UniquePtr<MlxArray>> = Vec::new();
        let mut previous_end = 0usize;

        for (image_idx, &(start, end)) in image_bounds.iter().enumerate() {
            if start > previous_end {
                let segment = mlxcel_core::slice(
                    &inputs_embeds,
                    &[0, previous_end as i32, 0],
                    &[1, start as i32, hidden_size],
                );
                segments.push(segment);
            }

            if let Some(processed) = processed_images.get(image_idx) {
                let pixel_values = mlxcel_core::from_slice_f32(
                    &processed.pixel_values,
                    &processed.pixel_values_shape,
                );
                let pixel_values =
                    mlxcel_core::astype(&pixel_values, mlxcel_core::array_dtype(&inputs_embeds));
                let vision_hidden_states = self
                    .vision_tower
                    .forward(&pixel_values, processed.spatial_shape);

                let resampled = self
                    .resampler
                    .forward(&vision_hidden_states, processed.spatial_shape);

                let fitted =
                    Self::fit_feature_span(resampled, end.saturating_sub(start), hidden_size);
                segments.push(fitted);
            }

            previous_end = end;
        }

        let total_len = mlxcel_core::array_shape(input_ids)[1] as usize;
        if previous_end < total_len {
            let tail = mlxcel_core::slice(
                &inputs_embeds,
                &[0, previous_end as i32, 0],
                &[1, total_len as i32, hidden_size],
            );
            segments.push(tail);
        }

        InputEmbeddings {
            inputs_embeds: encoders::minicpmo::concat_arrays(&segments, 1),
            attention_mask_4d: None,
        }
    }
}

impl LanguageModel for MiniCPMOVLModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model.forward_impl(input_ids, None, caches, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_impl(input_ids, input_embeddings, caches, mask)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.text_model.get_embed_tokens(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.text_model.make_caches()
    }

    fn num_layers(&self) -> usize {
        self.text_model.num_layers()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
}
