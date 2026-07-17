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

//! MiniMax-M3-VL Vision-Language Model (`model_type: "minimax_m3_vl"`).
//!
//! Composes the CLIP-style `MiniMaxM3VisionEncoder` (tower + two-stage
//! projector, which already projects to the text hidden size) with the
//! MiniMax-M3 hybrid dense/MoE text backbone. There is no MRoPE and no separate
//! connector: the projector output is scattered LLaVA-style into the image
//! placeholder positions of the embedded prompt, and the decoder runs its
//! standard partial 1D RoPE.
//!
//! The 427B checkpoint cannot be loaded on the development machine, so the
//! validated surface is the synthetic reduced-config unit tests plus the
//! real-config parse test.

use super::{encoders, merge, processors};
use crate::LanguageModel;
use crate::models::MiniMaxM3Model;
use mlxcel_core::cache::SequenceId;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct MiniMaxM3VlModel {
    pub text_model: MiniMaxM3Model,
    pub vision_encoder: encoders::minimax_m3_vl::MiniMaxM3VisionEncoder,
    pub processor: processors::minimax_m3::MiniMaxM3Processor,
    /// `]<]image[>[` scatter target (200025 in the real checkpoint).
    pub image_token_id: i32,
    /// `]<]video[>[` (200026). Video is out of scope for this port.
    pub video_token_id: i32,
    /// `]<]start of image[>[` (200029).
    pub vision_start_token_id: i32,
    /// `]<]end of image[>[` (200030).
    pub vision_end_token_id: i32,
    pub spatial_merge_size: i32,
    pub eos_token_ids: Vec<i32>,
}

impl MiniMaxM3VlModel {
    /// Encode images and scatter the merged features at image-placeholder
    /// positions. The vision tower emits `grid.prod() / merge^2` tokens per
    /// image, which matches the placeholder expansion count produced by the
    /// shared Qwen-VL insertion helper (`t * (h/merge) * (w/merge)`).
    pub fn input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> merge::InputEmbeddings {
        let inputs_embeds = self.text_model.get_embed_tokens(input_ids);
        // forward_with_grid casts pixel_values to f32 internally; merge_llava
        // casts the projected features back to the text embedding dtype.
        let vision_output = self
            .vision_encoder
            .forward_with_grid(pixel_values, grid_thw);
        merge::merge_llava(
            self.image_token_id,
            &vision_output.hidden_states,
            &inputs_embeds,
            input_ids,
        )
    }
}

impl LanguageModel for MiniMaxM3VlModel {
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
        mlxcel_core::generate::LanguageModel::num_layers(&self.text_model)
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }

    fn output_suppressed_token_ids(&self) -> Vec<i32> {
        // Image/video placeholders and their vision framing markers are
        // input-alignment ids and must never be sampled during decode. Video
        // reuses the image vision_start/vision_end framing in MiniMax-M3-VL, so
        // there are no separate video framing tokens.
        vec![
            self.image_token_id,
            self.video_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
        ]
    }
}

#[cfg(test)]
#[path = "minimax_m3_vl_tests.rs"]
mod minimax_m3_vl_tests;
