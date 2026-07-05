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

//! ERNIE-4.5 MoE VL Vision-Language Model.
//!
//! Composes the DFNRope dynamic-resolution vision tower, the
//! variable-resolution resampler, the smart-resize image processor, and the
//! modality-split ERNIE-4.5 MoE MRoPE text decoder. Follows the Qwen2-VL VLM
//! contract: packed vision sequences with `grid_thw`, LLaVA-style image-token
//! replacement at `<|IMAGE_PLACEHOLDER|>` positions, and 3D `[T, H, W]` MRoPE
//! position ids computed once at prefill (per-image placeholder count is
//! `t * (h / merge) * (w / merge)`, matching the resampler output rows).
//!
//! Reference: mlx-vlm
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/ernie4_5_moe_vl/ernie4_5_moe_vl.py>.

use super::{connectors, encoders, merge, processors};
use crate::LanguageModel;
use crate::models::ernie4_5_moe_vl::Ernie45MoeVlTextModel;
use crate::multimodal::batched_dispatch::forward_batched_with_seq_ids_dispatch;
use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::DecodeBatchContext;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct Ernie45MoeVlModel {
    pub text_model: Ernie45MoeVlTextModel,
    pub vision_encoder: encoders::ernie4_5_vl::Ernie45VlVisionEncoder,
    pub resampler: connectors::ernie4_5_vl::Ernie45VlResampler,
    pub processor: processors::ernie4_5_vl::Ernie45VlProcessor,
    pub image_token_id: i32,
    pub video_token_id: i32,
    pub vision_start_token_id: i32,
    pub vision_end_token_id: i32,
    pub video_start_token_id: i32,
    pub video_end_token_id: i32,
    pub spatial_merge_size: usize,
}

impl Ernie45MoeVlModel {
    /// Encode images, resample, merge features at image-token positions, and
    /// store the 3D MRoPE position ids on the text model (fallback slot).
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
        let image_features = self
            .resampler
            .forward_with_grid(&vision_output.hidden_states, grid_thw);

        let merged = merge::merge_llava(
            self.image_token_id,
            &image_features,
            &inputs_embeds,
            input_ids,
        );

        let position_ids = self.compute_rope_index(input_ids, grid_thw);
        let ids_shape = mlxcel_core::array_shape(input_ids);
        let seq_len = ids_shape[1];

        mlxcel_core::eval(&position_ids);
        let max_pos = mlxcel_core::max_all(&position_ids);
        mlxcel_core::eval(&max_pos);
        let rope_deltas = mlxcel_core::item_i32(&max_pos) + 1 - seq_len;

        self.text_model.set_mrope_state(position_ids, rope_deltas);
        merged
    }

    /// Compute 3D `[T, H, W]` position ids for a mixed text + image sequence
    /// (ERNIE `get_rope_index`, identical to the Qwen2-VL scheme: text runs
    /// share one scalar counter across all three axes; each vision block spans
    /// `(t, h/merge, w/merge)` index grids offset by the running counter).
    /// Returns `[3, 1, seq]`.
    fn compute_rope_index(
        &self,
        input_ids: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> UniquePtr<MlxArray> {
        mlxcel_core::eval(input_ids);
        let ids_shape = mlxcel_core::array_shape(input_ids);
        let seq_len = ids_shape[1] as usize;

        let mut tokens = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let tok = mlxcel_core::slice(input_ids, &[0, i as i32], &[1, i as i32 + 1]);
            mlxcel_core::eval(&tok);
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

    /// Move the fallback MRoPE state into the per-sequence map under `seq_id`.
    pub fn bind_mrope_state_to_sequence(&self, seq_id: SequenceId) {
        self.text_model.bind_mrope_state_to_sequence(seq_id);
    }
}

impl LanguageModel for Ernie45MoeVlModel {
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
        self.text_model.eos_token_ids()
    }

    fn output_suppressed_token_ids(&self) -> Vec<i32> {
        // Image/video placeholders and their frame markers are input-alignment
        // ids and must never be sampled during decode.
        vec![
            self.image_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
            self.video_start_token_id,
            self.video_end_token_id,
        ]
    }
}
