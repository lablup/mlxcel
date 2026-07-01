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

//! Moondream2 vision-language wrapper.
//!
//! Moondream2 shares Moondream3's vision tower (a linear-patch ViT with
//! overlap-crop reconstruction) and prompt structure, but pairs it with a
//! dense Phi-style text decoder instead of the sparse-MoE decoder. Like
//! Moondream3 it prepends a BOS token plus the projected image tokens ahead of
//! the text prompt, and applies a prefix-causal mask that lets the BOS + image
//! span attend bidirectionally while the prompt stays causal.

use crate::LanguageModel;
use crate::vision::merge::InputEmbeddings;
use crate::vision::{encoders, processors};
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

pub(crate) fn build_moondream2_attention_mask(
    prefix_tokens: usize,
    prompt_tokens: usize,
) -> UniquePtr<MlxArray> {
    let total_tokens = prefix_tokens + prompt_tokens;
    let mut values = vec![f32::NEG_INFINITY; total_tokens * total_tokens];

    for query_idx in 0..total_tokens {
        let allowed_keys = if query_idx < prefix_tokens {
            prefix_tokens
        } else {
            query_idx + 1
        };
        let row_start = query_idx * total_tokens;
        for key_idx in 0..allowed_keys {
            values[row_start + key_idx] = 0.0;
        }
    }

    let mask =
        mlxcel_core::from_slice_f32(&values, &[1, 1, total_tokens as i32, total_tokens as i32]);
    // Promote to bfloat16 so MLX SDPA can add the mask to the Q/K/V dtype.
    mlxcel_core::astype(&mask, mlxcel_core::dtype::BFLOAT16)
}

pub struct Moondream2VLModel {
    pub text_model: crate::models::Moondream2Model,
    pub vision_tower: encoders::moondream3::Moondream3VisionModel,
    pub processor: processors::moondream3::Moondream3Processor,
    pub eos_token_ids: Vec<i32>,
}

impl Moondream2VLModel {
    pub fn prefix_token_count(&self) -> usize {
        1 + self.vision_tower.output_token_count()
    }

    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        processed_image: &processors::moondream3::Moondream3ImageInput,
    ) -> InputEmbeddings {
        let prompt_embeddings = self.text_model.embed_tokens.forward(input_ids);
        let prompt_len = mlxcel_core::array_shape(&prompt_embeddings)[1] as usize;
        let embed_dtype = mlxcel_core::array_dtype(&prompt_embeddings);

        let pixel_values = mlxcel_core::from_slice_f32(
            &processed_image.pixel_values,
            &processed_image.pixel_values_shape,
        );
        let pixel_values = mlxcel_core::astype(&pixel_values, embed_dtype);
        let image_embeddings = self
            .vision_tower
            .encode_image_embeddings(&pixel_values, processed_image.tiling);

        let bos_ids = mlxcel_core::from_slice_i32(&[self.text_model.bos_token_id()], &[1, 1]);
        let bos_embeddings = self.text_model.embed_tokens.forward(&bos_ids);

        let prefix_embeddings = mlxcel_core::concatenate(&bos_embeddings, &image_embeddings, 1);
        let all_embeddings = mlxcel_core::concatenate(&prefix_embeddings, &prompt_embeddings, 1);
        let prefix_len = self.prefix_token_count();

        InputEmbeddings {
            inputs_embeds: all_embeddings,
            attention_mask_4d: Some(build_moondream2_attention_mask(prefix_len, prompt_len)),
        }
    }
}

impl LanguageModel for Moondream2VLModel {
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
        self.text_model.embed_tokens(input_ids)
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
