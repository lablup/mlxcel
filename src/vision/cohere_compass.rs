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

//! Cohere Compass VLM (`cohere_compass`, CohereLabs North-Micro-Vision).
//!
//! A Qwen3-VL vision tower (Conv3d patch embed, interpolated 48x48 position
//! grid, three DeepStack mergers) in front of the Command-style Compass text
//! decoder. The vision half and the whole Qwen-VL prompt/MRoPE machinery are
//! reused verbatim; the only thing this family changes is the decoder, so the
//! wrapper mirrors [`super::qwen3_vl::Qwen3VLModel`] one for one.

use super::feature_cache::{CacheKey, DeepStackFeatures, VisionFeatureCache};
use super::{encoders, merge, processors};
use crate::LanguageModel;
use crate::multimodal::qwen_vl::{
    compute_qwen_vl_mrope_position_ids, forward_batched_with_seq_ids_dispatch,
};
use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::DecodeBatchContext;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct CohereCompassModel {
    pub text_model: crate::models::CohereCompassTextModel,
    pub vision_encoder: encoders::qwen3_vl::Qwen3VLVisionEncoder,
    pub processor: processors::qwen2_vl::Qwen2VLProcessor,
    pub image_token_id: i32,
    pub video_token_id: i32,
    pub vision_start_token_id: i32,
    pub spatial_merge_size: usize,
}

impl CohereCompassModel {
    /// Token embeddings with the vision features merged in at the image-pad
    /// positions, plus the DeepStack side branches handed to the text model.
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> merge::InputEmbeddings {
        self.get_input_embeddings_with_cache(input_ids, pixel_values, grid_thw, None, None)
    }

    /// Cache-aware variant. Both the post-merger hidden states and the three
    /// DeepStack branch outputs are cached together, keyed on the concatenated
    /// pixel tensor; the MRoPE state and the visual position mask depend on the
    /// current `input_ids` and are always recomputed.
    pub fn get_input_embeddings_with_cache(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
        cache_key: Option<&CacheKey>,
        vision_cache: Option<&std::sync::Mutex<VisionFeatureCache<DeepStackFeatures>>>,
    ) -> merge::InputEmbeddings {
        let inputs_embeds = self.text_model.get_embed_tokens(input_ids);

        let cached: Option<DeepStackFeatures> = match (cache_key, vision_cache) {
            (Some(key), Some(cache)) => cache.lock().ok().and_then(|mut guard| guard.get(key)),
            _ => None,
        };

        let (image_features, deepstack_features): (UniquePtr<MlxArray>, Vec<UniquePtr<MlxArray>>) =
            if let Some(cached) = cached {
                (cached.hidden_states, cached.deepstack)
            } else {
                let embed_dtype = mlxcel_core::array_dtype(&inputs_embeds);
                let pv = mlxcel_core::astype(pixel_values, embed_dtype);
                let vision_output = self.vision_encoder.forward_with_grid(&pv, grid_thw);

                if let (Some(key), Some(cache)) = (cache_key, vision_cache) {
                    mlxcel_core::eval(&vision_output.hidden_states);
                    for feat in &vision_output.deepstack_features {
                        mlxcel_core::eval(feat);
                    }
                    let hs_copy = mlxcel_core::copy(vision_output.hidden_states.as_ref().unwrap());
                    let ds_copy: Vec<UniquePtr<MlxArray>> = vision_output
                        .deepstack_features
                        .iter()
                        .map(|feat| mlxcel_core::copy(feat.as_ref().unwrap()))
                        .collect();
                    let snapshot = DeepStackFeatures::new(hs_copy, ds_copy);
                    if let Ok(mut guard) = cache.lock() {
                        guard.put(key.clone(), &snapshot);
                    }
                }

                (
                    vision_output.hidden_states,
                    vision_output.deepstack_features,
                )
            };

        let merged = merge::merge_llava_any(
            &[self.image_token_id, self.video_token_id],
            image_features.as_ref().unwrap(),
            &inputs_embeds,
            input_ids,
        );

        let visual_pos_masks = self.compute_visual_pos_masks(input_ids);
        mlxcel_core::eval(&visual_pos_masks);

        for feat in &deepstack_features {
            mlxcel_core::eval(feat);
        }

        if !deepstack_features.is_empty() {
            self.text_model
                .set_deepstack_state(visual_pos_masks, deepstack_features);
        }

        let position_ids = self.compute_rope_index(input_ids, grid_thw);
        let seq_len = mlxcel_core::array_shape(input_ids)[1];

        mlxcel_core::eval(&position_ids);
        let max_pos = mlxcel_core::max_all(&position_ids);
        mlxcel_core::eval(&max_pos);
        let rope_deltas = mlxcel_core::item_i32(&max_pos) + 1 - seq_len;

        self.text_model.set_mrope_state(position_ids, rope_deltas);

        merged
    }

    /// `[batch, seq_len]` bool mask of image/video token positions.
    fn compute_visual_pos_masks(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        let img_token = mlxcel_core::from_slice_i32(&[self.image_token_id], &[1]);
        let vid_token = mlxcel_core::from_slice_i32(&[self.video_token_id], &[1]);
        let img_mask = mlxcel_core::equal(input_ids, &img_token);
        let vid_mask = mlxcel_core::equal(input_ids, &vid_token);
        mlxcel_core::logical_or(&img_mask, &vid_mask)
    }

    /// `[3, batch, seq_len]` (T, H, W) position ids. Identical rule to
    /// Qwen2-VL/Qwen3-VL: text advances all three axes, an image block takes
    /// its merged-grid coordinates, and the next text token restarts at
    /// `max(previous) + 1`.
    fn compute_rope_index(
        &self,
        input_ids: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> UniquePtr<MlxArray> {
        compute_qwen_vl_mrope_position_ids(
            input_ids,
            grid_thw,
            self.spatial_merge_size,
            self.image_token_id,
            self.video_token_id,
        )
    }
}

impl LanguageModel for CohereCompassModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model.forward_impl(input_ids, None, caches)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_impl(input_ids, input_embeddings, caches)
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_for_sequence(input_ids, None, caches, seq_id)
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_for_sequence(input_ids, input_embeddings, caches, seq_id)
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.text_model.release_mrope_sequence(seq_id);
    }

    /// Per-row batched dispatch with seq_ids so each row's MRoPE state
    /// resolves correctly in a mixed VL + text server batch.
    fn forward_batched_with_context_and_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
        context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        forward_batched_with_seq_ids_dispatch(
            &self.text_model,
            input_ids,
            seq_ids,
            batch_caches,
            mask,
            context,
        )
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
        mlxcel_core::generate::LanguageModel::eos_token_ids(&self.text_model)
    }
}

#[cfg(test)]
#[path = "cohere_compass_tests.rs"]
mod cohere_compass_tests;
