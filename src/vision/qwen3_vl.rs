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

//! Qwen3-VL Vision-Language Model
//!
//! Vision encoder with DeepStack + Qwen3 language model with interleaved MRoPE

use super::feature_cache::{CacheKey, DeepStackFeatures, VisionFeatureCache};
use super::{encoders, merge, processors};
use crate::LanguageModel;
use crate::multimodal::qwen_vl::forward_batched_with_seq_ids_dispatch;
use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::DecodeBatchContext;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

/// Qwen3-VL VLM: Vision encoder with DeepStack + Qwen3 language model with interleaved MRoPE
///
/// Key differences from Qwen2/2.5-VL:
/// - Learned position embeddings with bilinear interpolation
/// - DeepStack multi-layer visual injection
/// - Interleaved MRoPE (not chunked)
/// - q_norm/k_norm in attention
/// - No windowed attention
pub struct Qwen3VLModel {
    pub text_model: crate::models::Qwen3VLModel,
    pub vision_encoder: encoders::qwen3_vl::Qwen3VLVisionEncoder,
    pub processor: processors::qwen2_vl::Qwen2VLProcessor,
    pub image_token_id: i32,
    pub video_token_id: i32,
    pub vision_start_token_id: i32,
    pub spatial_merge_size: usize,
}

impl Qwen3VLModel {
    /// Get input embeddings with vision features merged in
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> merge::InputEmbeddings {
        self.get_input_embeddings_with_cache(input_ids, pixel_values, grid_thw, None, None)
    }

    /// Get input embeddings with optional vision feature caching.
    ///
    /// Caches both the post-merger hidden states and the DeepStack side-branch
    /// features, since Qwen3-VL injects DeepStack outputs at selected language
    /// model layers. The cache key covers the concatenated pixel tensor for
    /// all images in the request (see [`Qwen25VLModel::get_input_embeddings_with_cache`]).
    ///
    /// MRoPE state and the visual position mask are recomputed on every call
    /// because they depend on the current `input_ids`.
    pub fn get_input_embeddings_with_cache(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
        cache_key: Option<&CacheKey>,
        vision_cache: Option<&std::sync::Mutex<VisionFeatureCache<DeepStackFeatures>>>,
    ) -> merge::InputEmbeddings {
        let inputs_embeds = self.text_model.get_embed_tokens(input_ids);

        // Cache lookup: reuse post-merger hidden states + DeepStack branch
        // outputs when the same pixel tensor has already been encoded.
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

                // Populate the cache on miss. We eval and deep-copy each
                // tensor to decouple the cached snapshot from the graph.
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

        // Merge vision features at image token positions (LLaVA-style)
        let merged = merge::merge_llava(
            self.image_token_id,
            image_features.as_ref().unwrap(),
            &inputs_embeds,
            input_ids,
        );

        // Compute visual_pos_masks: boolean mask of image/video token positions
        let visual_pos_masks = self.compute_visual_pos_masks(input_ids);
        mlxcel_core::eval(&visual_pos_masks);

        // Eval deepstack features before storing
        for feat in &deepstack_features {
            mlxcel_core::eval(feat);
        }

        // Store deepstack state in language model
        if !deepstack_features.is_empty() {
            self.text_model
                .set_deepstack_state(visual_pos_masks, deepstack_features);
        }

        // Compute MRoPE position IDs (same algorithm as Qwen2-VL)
        let position_ids = self.compute_rope_index(input_ids, grid_thw);
        let ids_shape = mlxcel_core::array_shape(input_ids);
        let seq_len = ids_shape[1];

        mlxcel_core::eval(&position_ids);
        let max_pos = mlxcel_core::max_all(&position_ids);
        mlxcel_core::eval(&max_pos);
        let max_pos_val = mlxcel_core::item_i32(&max_pos);
        let rope_deltas = max_pos_val + 1 - seq_len;

        self.text_model.set_mrope_state(position_ids, rope_deltas);

        merged
    }

    /// Compute boolean mask of image/video token positions
    /// Returns: [batch, seq_len]
    fn compute_visual_pos_masks(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        let img_token = mlxcel_core::from_slice_i32(&[self.image_token_id], &[1]);
        let vid_token = mlxcel_core::from_slice_i32(&[self.video_token_id], &[1]);

        let img_mask = mlxcel_core::equal(input_ids, &img_token);
        let vid_mask = mlxcel_core::equal(input_ids, &vid_token);

        // OR the two masks
        mlxcel_core::logical_or(&img_mask, &vid_mask)
    }

    /// Compute 3D position IDs [T, H, W] for mixed text+image sequences
    /// (Same algorithm as Qwen2-VL/2.5-VL)
    fn compute_rope_index(
        &self,
        input_ids: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> UniquePtr<MlxArray> {
        // Flatten to 1-D and eval once — avoids per-token eval() round-trips.
        let flat = mlxcel_core::reshape(input_ids, &[-1]);
        mlxcel_core::eval(&flat);
        let ids_shape = mlxcel_core::array_shape(input_ids);
        let seq_len = ids_shape[1] as usize;

        let mut tokens = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let tok = mlxcel_core::slice(&flat, &[i as i32], &[i as i32 + 1]);
            tokens.push(mlxcel_core::item_i32(&tok));
        }

        let merge = self.spatial_merge_size as i32;
        let mut pos_ids: Vec<Vec<i32>> = vec![Vec::new(); 3];
        let mut image_idx = 0usize;
        let mut st = 0usize;
        let mut current_pos = 0i32;

        let mut i = 0;
        while i < seq_len {
            if tokens[i] == self.image_token_id || tokens[i] == self.video_token_id {
                let vision_start = i;

                while i < seq_len
                    && (tokens[i] == self.image_token_id || tokens[i] == self.video_token_id)
                {
                    i += 1;
                }

                if vision_start > st {
                    let text_len = vision_start - st;
                    for p in current_pos..current_pos + text_len as i32 {
                        pos_ids[0].push(p);
                        pos_ids[1].push(p);
                        pos_ids[2].push(p);
                    }
                    current_pos += text_len as i32;
                }

                if image_idx < grid_thw.len() {
                    let (t, h, w) = grid_thw[image_idx];
                    let llm_h = h / merge;
                    let llm_w = w / merge;
                    let llm_t = t;

                    for ti in 0..llm_t {
                        for hi in 0..llm_h {
                            for wi in 0..llm_w {
                                pos_ids[0].push(current_pos + ti);
                                pos_ids[1].push(current_pos + hi);
                                pos_ids[2].push(current_pos + wi);
                            }
                        }
                    }
                    current_pos += llm_t.max(llm_h).max(llm_w);
                    image_idx += 1;
                }

                st = i;
                continue;
            }
            i += 1;
        }

        if st < seq_len {
            let text_len = seq_len - st;
            for p in current_pos..current_pos + text_len as i32 {
                pos_ids[0].push(p);
                pos_ids[1].push(p);
                pos_ids[2].push(p);
            }
        }

        let total_len = pos_ids[0].len() as i32;
        let t_arr = mlxcel_core::from_slice_i32(&pos_ids[0], &[1, 1, total_len]);
        let h_arr = mlxcel_core::from_slice_i32(&pos_ids[1], &[1, 1, total_len]);
        let w_arr = mlxcel_core::from_slice_i32(&pos_ids[2], &[1, 1, total_len]);

        let th = mlxcel_core::concatenate(t_arr.as_ref().unwrap(), h_arr.as_ref().unwrap(), 0);
        mlxcel_core::concatenate(th.as_ref().unwrap(), w_arr.as_ref().unwrap(), 0)
    }
}

impl LanguageModel for Qwen3VLModel {
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

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // route through the per-sequence MRoPE path so the
        // cached scalar delta cannot leak across requests.
        self.text_model
            .forward_for_sequence(input_ids, None, caches, mask, seq_id)
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_for_sequence(input_ids, input_embeddings, caches, mask, seq_id)
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.text_model.release_mrope_sequence(seq_id);
    }

    /// per-row batched dispatch with seq_ids so each row's
    /// MRoPE state resolves correctly in mixed VL+text batches.
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
