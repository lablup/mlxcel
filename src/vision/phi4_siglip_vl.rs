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

//! Phi4-SigLIP vision-language model.
//!
//! Used by: Phi4-SigLIP VLM

use crate::LanguageModel;
use crate::multimodal::phi4_siglip_prompt::PHI4_SIGLIP_IMAGE_TOKEN_INDEX;
use crate::vision::merge::InputEmbeddings;
use crate::vision::{encoders, processors};
use mlxcel_core::layers::{KVCache, UnifiedLinear};
use mlxcel_core::{MlxArray, UniquePtr};

pub struct Phi4SigLipVLModel {
    pub text_model: crate::models::Phi3Model,
    pub vision_tower: encoders::phi4_siglip::Phi4SigLipVisionEncoder,
    pub mm_projector_linear1: UnifiedLinear,
    pub mm_projector_linear2: UnifiedLinear,
    pub processor: processors::phi4_siglip::Phi4SigLipProcessor,
    pub select_layer: isize,
    pub eos_token_ids: Vec<i32>,
}

impl Phi4SigLipVLModel {
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        processed_images: &[processors::phi4_siglip::Phi4SigLipImageInput],
    ) -> InputEmbeddings {
        let ids_shape = mlxcel_core::array_shape(input_ids);
        let seq_len = ids_shape[1] as usize;
        let mut original_tokens = Vec::with_capacity(seq_len);
        let mut safe_tokens = Vec::with_capacity(seq_len);
        let mut image_positions = Vec::new();

        for token_idx in 0..seq_len {
            let token = mlxcel_core::slice(
                input_ids,
                &[0, token_idx as i32],
                &[1, token_idx as i32 + 1],
            );
            mlxcel_core::eval(&token);
            let value = mlxcel_core::item_i32(&token);
            original_tokens.push(value);
            if value == PHI4_SIGLIP_IMAGE_TOKEN_INDEX {
                image_positions.push(token_idx);
                safe_tokens.push(0);
            } else {
                safe_tokens.push(value);
            }
        }

        let safe_input_ids = mlxcel_core::from_slice_i32(&safe_tokens, &[1, seq_len as i32]);
        let inputs_embeds = self.text_model.embed_tokens.forward(&safe_input_ids);

        if processed_images.is_empty() || image_positions.is_empty() {
            return InputEmbeddings {
                inputs_embeds,
                attention_mask_4d: None,
            };
        }

        let embed_dtype = mlxcel_core::array_dtype(&inputs_embeds);
        let hidden_size = mlxcel_core::array_shape(&inputs_embeds)[2];
        let mut segments: Vec<UniquePtr<MlxArray>> = Vec::new();
        let mut previous_end = 0usize;

        for (image_idx, &position) in image_positions.iter().enumerate() {
            if position > previous_end {
                let segment = mlxcel_core::slice(
                    &inputs_embeds,
                    &[0, previous_end as i32, 0],
                    &[1, position as i32, hidden_size],
                );
                segments.push(segment);
            }

            if let Some(processed) = processed_images.get(image_idx) {
                let mut hidden_states = self
                    .vision_tower
                    .forward_hidden_states(&processed.pixel_values, processed.spatial_shape);
                let layer_count = hidden_states.len() as isize;
                let selected_index = if self.select_layer < 0 {
                    (layer_count + self.select_layer) as usize
                } else {
                    self.select_layer as usize
                };
                let selected = hidden_states.swap_remove(selected_index);
                let selected = mlxcel_core::astype(&selected, embed_dtype);
                let projected = self.mm_projector_linear1.forward(&selected);
                let projected = mlxcel_core::gelu_approx(&projected);
                let projected = self.mm_projector_linear2.forward(&projected);
                segments.push(projected);
            }

            previous_end = position + 1;
        }

        if previous_end < seq_len {
            let tail = mlxcel_core::slice(
                &inputs_embeds,
                &[0, previous_end as i32, 0],
                &[1, seq_len as i32, hidden_size],
            );
            segments.push(tail);
        }

        let inputs_embeds = encoders::phi4_siglip::concat_arrays(&segments, 1);
        InputEmbeddings {
            inputs_embeds,
            attention_mask_4d: None,
        }
    }
}

impl LanguageModel for Phi4SigLipVLModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model.forward(input_ids, caches, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_with_embeddings(input_ids, input_embeddings, caches, mask)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.text_model.embed_tokens.forward(input_ids))
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
