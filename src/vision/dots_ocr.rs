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

//! dots.ocr Vision-Language Model.
//!
//! Composes the `dots_vit` vision tower (which projects to the text width in its
//! own merger), the dynamic-resolution processor, and a plain Qwen2 text
//! decoder. There is no MRoPE and no separate connector: the merger output is
//! scattered LLaVA-style into the `<|imgpad|>` positions of the embedded prompt,
//! and the decoder runs standard 1D RoPE.
//!
//! Reference: mlx-vlm `mlx_vlm/models/dots_ocr/dots_ocr.py`.

use super::{encoders, merge, processors};
use crate::LanguageModel;
use crate::models::qwen2::Model as Qwen2Model;
use mlxcel_core::cache::SequenceId;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct DotsOcrVlModel {
    pub text_model: Qwen2Model,
    pub vision_encoder: encoders::dots_ocr::DotsVisionEncoder,
    pub processor: processors::dots_ocr::DotsOcrProcessor,
    pub image_token_id: i32,
    pub video_token_id: i32,
    pub vision_start_token_id: i32,
    pub spatial_merge_size: i32,
    pub eos_token_ids: Vec<i32>,
}

impl DotsOcrVlModel {
    /// Encode images and scatter the merged features at `<|imgpad|>` positions.
    pub fn input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> merge::InputEmbeddings {
        let inputs_embeds = self.text_model.get_embed_tokens(input_ids);
        let embed_dtype = mlxcel_core::array_dtype(&inputs_embeds);
        let pv = mlxcel_core::astype(pixel_values, embed_dtype);
        let vision_output = self.vision_encoder.forward_with_grid(&pv, grid_thw);
        merge::merge_llava(
            self.image_token_id,
            &vision_output.hidden_states,
            &inputs_embeds,
            input_ids,
        )
    }
}

impl LanguageModel for DotsOcrVlModel {
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
            .forward_with_embeddings_impl(input_ids, input_embeddings, caches, mask)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.text_model.get_embed_tokens(input_ids))
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        _seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model.forward(input_ids, caches, mask)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.text_model.make_caches()
    }

    fn num_layers(&self) -> usize {
        self.text_model.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }

    fn output_suppressed_token_ids(&self) -> Vec<i32> {
        // The image placeholder id must never be sampled during decode.
        vec![self.image_token_id]
    }
}
