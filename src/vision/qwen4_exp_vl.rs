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

//! Qwen3.8-Flash-Next (`qwen4_exp`) vision-language model: the Qwen3-VL
//! vision tower without deepstack features in front of the Qwen4-Exp text
//! backbone, with Qwen-VL prompt expansion and MRoPE positions.

use super::{encoders, merge, processors};
use crate::LanguageModel;
use crate::models::Qwen4ExpModel;
use crate::multimodal::qwen_vl::compute_qwen_vl_mrope_position_ids;
use mlxcel_core::cache::{KVCacheMode, SequenceId, SequenceStateLayout};
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct Qwen4ExpVLModel {
    pub text_model: Qwen4ExpModel,
    pub vision_encoder: encoders::qwen3_vl::Qwen3VLVisionEncoder,
    pub processor: processors::qwen2_vl::Qwen2VLProcessor,
    pub image_token_id: i32,
    pub video_token_id: i32,
    pub vision_start_token_id: i32,
    pub spatial_merge_size: usize,
}

impl Qwen4ExpVLModel {
    /// Token embeddings with the vision features scattered over the image and
    /// video placeholders. Also stores the prompt's MRoPE positions on the
    /// text model for the prefill that follows.
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> merge::InputEmbeddings {
        let inputs_embeds = self.text_model.get_embed_tokens(input_ids);
        let pixel_values =
            mlxcel_core::astype(pixel_values, mlxcel_core::array_dtype(&inputs_embeds));
        let vision_output = self
            .vision_encoder
            .forward_with_grid(&pixel_values, grid_thw);
        let merged = merge::merge_llava_any(
            &[self.image_token_id, self.video_token_id],
            &vision_output.hidden_states,
            &inputs_embeds,
            input_ids,
        );

        let position_ids = compute_qwen_vl_mrope_position_ids(
            input_ids,
            grid_thw,
            self.spatial_merge_size,
            self.image_token_id,
            self.video_token_id,
        );
        mlxcel_core::eval(&position_ids);
        let max_position = mlxcel_core::max_all(&position_ids);
        mlxcel_core::eval(&max_position);
        let seq_len = mlxcel_core::array_shape(input_ids)[1];
        let rope_deltas = mlxcel_core::item_i32(&max_position) + 1 - seq_len;
        self.text_model.set_mrope_state(position_ids, rope_deltas);
        merged
    }
}

impl LanguageModel for Qwen4ExpVLModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model.forward(input_ids, caches, mask)
    }

    fn forward_last_logits(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_last_logits(input_ids, caches, mask, last_pos)
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

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_with_sequence_id(input_ids, seq_id, caches, mask)
    }

    fn forward_last_logits_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_last_logits_with_sequence_id(input_ids, seq_id, caches, mask, last_pos)
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model.forward_with_embeddings_and_sequence_id(
            input_ids,
            input_embeddings,
            seq_id,
            caches,
            mask,
        )
    }

    fn forward_last_logits_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_last_logits_with_embeddings_and_sequence_id(
                input_ids,
                input_embeddings,
                seq_id,
                caches,
                mask,
                last_pos,
            )
    }

    fn forward_batched_with_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_batched_with_ids(input_ids, seq_ids, batch_caches, mask)
    }

    fn forward_batched(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_batched(input_ids, batch_caches, mask)
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
        self.text_model.eos_token_ids()
    }

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        self.text_model.sequence_state_layout()
    }

    fn supports_batching(&self) -> bool {
        self.text_model.supports_batching()
    }

    fn supports_batched_prefill(&self) -> bool {
        self.text_model.supports_batched_prefill()
    }

    fn supports_padded_prefill(&self) -> bool {
        self.text_model.supports_padded_prefill()
    }

    fn supports_maskless_padded_prefill(&self) -> bool {
        self.text_model.supports_maskless_padded_prefill()
    }

    fn supports_paged_decode_backend(&self) -> bool {
        self.text_model.supports_paged_decode_backend()
    }

    fn set_kv_cache_layer_modes(&self, modes: Vec<KVCacheMode>) {
        self.text_model.set_kv_cache_layer_modes(modes)
    }

    fn kv_cache_layer_modes(&self) -> Option<Vec<KVCacheMode>> {
        self.text_model.kv_cache_layer_modes()
    }

    fn reset_runtime_state(&self) {
        self.text_model.reset_runtime_state()
    }

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        self.text_model.prepare_sequence_state(seq_id)
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.text_model.release_sequence_state_by_id(seq_id)
    }
}
