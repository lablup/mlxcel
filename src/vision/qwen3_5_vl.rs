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

//! Qwen3.5 Vision-Language Model
//!
//! Qwen3-VL vision encoder + Qwen3.5 hybrid text backbone (Transformer + GatedDeltaNet)
//! Uses interleaved MRoPE with partial rotary factor

use super::{encoders, merge, processors};
use crate::LanguageModel;
use crate::models::qwen3_5::{GdnRollbackSnapshot, VerifyOutput};
use crate::models::qwen3_next::Qwen3NextCache;
use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::DecodeBatchContext;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

/// Qwen3.5 VLM: Qwen3-VL vision encoder + Qwen3.5 hybrid text backbone
///
/// Key differences from Qwen3-VL:
/// - Hybrid Transformer + GatedDeltaNet (linear attention) text model
/// - Interleaved MRoPE with partial_rotary_factor=0.25
/// - Output gating in full attention layers
/// - No DeepStack visual injection (deepstack_visual_indexes = [])
pub struct Qwen35VLModel {
    pub text_model: crate::models::qwen3_5::Qwen35Model,
    pub vision_encoder: encoders::qwen3_vl::Qwen3VLVisionEncoder,
    pub processor: processors::qwen2_vl::Qwen2VLProcessor,
    pub image_token_id: i32,
    pub video_token_id: i32,
    pub vision_start_token_id: i32,
    pub spatial_merge_size: usize,
}

impl Qwen35VLModel {
    /// Get input embeddings with vision features merged in
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> merge::InputEmbeddings {
        let inputs_embeds = self.text_model.get_embed_tokens(input_ids);

        // Encode images through vision tower
        let embed_dtype = mlxcel_core::array_dtype(&inputs_embeds);
        let pv = mlxcel_core::astype(pixel_values, embed_dtype);
        let vision_output = self.vision_encoder.forward_with_grid(&pv, grid_thw);
        let image_features = &vision_output.hidden_states;

        // Merge vision features at image token positions (LLaVA-style)
        let merged = merge::merge_llava(
            self.image_token_id,
            image_features,
            &inputs_embeds,
            input_ids,
        );

        // Compute MRoPE position IDs (same algorithm as Qwen2-VL/Qwen3-VL)
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

    /// Speculative-decode verify pass that mirrors the text model's
    /// [`Qwen35Model::forward_speculative`].
    ///
    /// DFlash's drafter consumes per-layer hidden captures and
    /// per-GDN-layer rollback snapshots. The VLM wrapper exposes the same
    /// hooks so multimodal prefill + speculative tail can compose. The
    /// vision pathway runs in the standard prefill before this method is
    /// invoked, so the verify pass only needs the text-side caches.
    ///
    /// Used by: DFlash drafter round loop (sub-12).
    pub fn forward_speculative(
        &self,
        input_ids: &MlxArray,
        caches: &mut [Qwen3NextCache],
        capture_layer_ids: &[usize],
    ) -> VerifyOutput {
        self.text_model
            .forward_speculative(input_ids, caches, capture_layer_ids)
    }

    /// Rewind both attention and GDN caches to the accepted-prefix
    /// position. Delegates to the text model — the VLM wrapper holds no
    /// cache state of its own.
    pub fn rollback_speculative_cache(
        &self,
        caches: &mut [Qwen3NextCache],
        gdn_states: &[GdnRollbackSnapshot],
        accepted: &[i32],
        block_size: i32,
    ) -> i32 {
        self.text_model
            .rollback_speculative_cache(caches, gdn_states, accepted, block_size)
    }

    /// Compute 3D position IDs [T, H, W] for mixed text+image sequences
    /// (Same algorithm as Qwen2-VL/2.5-VL/Qwen3-VL)
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
}

/// DFlash speculative-decoding target adapter for the Qwen 3.5 VLM
/// wrapper.
///
/// Delegates every hook to the inner `text_model`. The VLM wrapper
/// itself holds no speculative state — vision is fully prefilled
/// before the verify/rollback loop begins, so the round loop interacts
/// only with the text backbone.
///
/// Used by: DFlash B=1 round loop.
impl mlxcel_core::drafter::dflash::SpeculativeTarget for Qwen35VLModel {
    type Cache = crate::models::qwen3_next::Qwen3NextCache;
    type VerifyOut = crate::models::qwen3_5::VerifyOutput;

    fn capture_layer_ids(&self) -> &[usize] {
        // The VLM wrapper follows the text model: the drafter checkpoint owns
        // the actual capture list, so callers pass it through
        // `verify_forward_with_capture_layers`.
        &[]
    }

    fn verify_forward(
        &self,
        verify_input: &MlxArray,
        caches: &mut [Self::Cache],
    ) -> Self::VerifyOut {
        self.text_model.forward_speculative(
            verify_input,
            caches,
            mlxcel_core::drafter::dflash::config::DEFAULT_TARGET_LAYER_IDS,
        )
    }

    fn verify_forward_with_capture_layers(
        &self,
        verify_input: &MlxArray,
        caches: &mut [Self::Cache],
        capture_layer_ids: &[usize],
    ) -> Self::VerifyOut {
        let capture_layer_ids = if capture_layer_ids.is_empty() {
            mlxcel_core::drafter::dflash::config::DEFAULT_TARGET_LAYER_IDS
        } else {
            capture_layer_ids
        };
        self.text_model
            .forward_speculative(verify_input, caches, capture_layer_ids)
    }

    fn rollback_partial(
        &self,
        caches: &mut [Self::Cache],
        verify_out: &Self::VerifyOut,
        accepted: i32,
        block_size: i32,
    ) {
        let _ = self.text_model.rollback_speculative_cache(
            caches,
            &verify_out.gdn_states,
            &[accepted],
            block_size,
        );
    }

    /// Batched B > 1 rollback for the VLM wrapper. — overrides
    /// the trait default (which panics for B > 1) so the batched DFlash
    /// round loop can drive a Qwen 3.5 VLM target with batched-prefill
    /// served by the continuous-batching scheduler.
    ///
    /// Mirrors the inner text-model `rollback_partial_batched`: the
    /// per-row K/V tail-zero and per-row GDN replay live inside
    /// `Qwen35Model::rollback_speculative_cache`, which already accepts a
    /// multi-element `accepted` slice.
    fn rollback_partial_batched(
        &self,
        caches: &mut [Self::Cache],
        verify_out: &Self::VerifyOut,
        accepted: &[i32],
        block_size: i32,
    ) {
        let _ = self.text_model.rollback_speculative_cache(
            caches,
            &verify_out.gdn_states,
            accepted,
            block_size,
        );
    }

    fn concat_hidden_for_drafter(&self, verify_out: &Self::VerifyOut) -> UniquePtr<MlxArray> {
        debug_assert!(
            !verify_out.hidden_states.is_empty(),
            "DFlash verify output must carry at least one captured hidden layer"
        );
        let mut acc = mlxcel_core::copy(verify_out.hidden_states[0].as_ref().unwrap());
        for slab in verify_out.hidden_states.iter().skip(1) {
            acc = mlxcel_core::concatenate(&acc, slab.as_ref().unwrap(), -1);
        }
        acc
    }

    fn verify_logits<'a>(&self, verify_out: &'a Self::VerifyOut) -> &'a MlxArray {
        verify_out
            .logits
            .as_ref()
            .expect("DFlash verify output must carry logits")
    }
}

impl LanguageModel for Qwen35VLModel {
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
        // forward through the underlying Qwen3.5 model's
        // seq-id-aware path so the per-sequence MRoPE entry resolves
        // for this specific request.
        mlxcel_core::generate::LanguageModel::forward_with_sequence_id(
            &self.text_model,
            input_ids,
            seq_id,
            caches,
            mask,
        )
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        mlxcel_core::generate::LanguageModel::forward_with_embeddings_and_sequence_id(
            &self.text_model,
            input_ids,
            input_embeddings,
            seq_id,
            caches,
            mask,
        )
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        mlxcel_core::generate::LanguageModel::release_sequence_state_by_id(
            &self.text_model,
            seq_id,
        );
    }

    fn reset_runtime_state(&self) {
        // Used by: CxxGenerator single-row generation paths. Reset the
        // hybrid text backbone's fallback mixed-cache slot while preserving
        // the MRoPE fallback written by VLM embedding preparation just before
        // generation starts.
        mlxcel_core::generate::LanguageModel::reset_runtime_state(&self.text_model);
    }

    /// forward `seq_ids` to the underlying Qwen3.5 model so
    /// each row's MRoPE state resolves correctly in mixed VL+text
    /// batches. `Qwen35Model::forward_batched_with_context_and_ids`
    /// already implements per-row dispatch and the batched-prefill fast
    /// path, so we forward straight through.
    fn forward_batched_with_context_and_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
        context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        mlxcel_core::generate::LanguageModel::forward_batched_with_context_and_ids(
            &self.text_model,
            input_ids,
            seq_ids,
            batch_caches,
            mask,
            context,
        )
    }

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        mlxcel_core::generate::LanguageModel::prepare_sequence_state(&self.text_model, seq_id);
    }

    fn sequence_state_layout(&self) -> mlxcel_core::cache::SequenceStateLayout {
        mlxcel_core::generate::LanguageModel::sequence_state_layout(&self.text_model)
    }

    fn supports_batching(&self) -> bool {
        mlxcel_core::generate::LanguageModel::supports_batching(&self.text_model)
    }

    fn supports_batched_prefill(&self) -> bool {
        mlxcel_core::generate::LanguageModel::supports_batched_prefill(&self.text_model)
    }

    fn supports_paged_decode_backend(&self) -> bool {
        mlxcel_core::generate::LanguageModel::supports_paged_decode_backend(&self.text_model)
    }

    fn supports_snapshot_reuse(&self) -> bool {
        // The vision wrapper owns no recurrent/text state of its own; the
        // Qwen3.5 text backbone carries the mixed attention / GatedDeltaNet
        // caches. Delegate snapshot capability so VLM-wrapped checkpoints
        // participate in the exact-prefix prompt cache just like the text
        // model.
        mlxcel_core::generate::LanguageModel::supports_snapshot_reuse(&self.text_model)
    }

    fn snapshot_sequence_state(
        &self,
        seq_id: SequenceId,
        token_len: usize,
    ) -> Option<mlxcel_core::generate::ModelStateSnapshot> {
        mlxcel_core::generate::LanguageModel::snapshot_sequence_state(
            &self.text_model,
            seq_id,
            token_len,
        )
    }

    fn restore_sequence_state(
        &self,
        seq_id: SequenceId,
        snapshot: &mlxcel_core::generate::ModelStateSnapshot,
    ) -> Result<(), String> {
        mlxcel_core::generate::LanguageModel::restore_sequence_state(
            &self.text_model,
            seq_id,
            snapshot,
        )
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.text_model.get_embed_tokens(input_ids))
    }

    /// Delegate to the inner text model so a DFlash drafter can lazy-bind
    /// the Qwen 3.5 embedding table even when the target is a VLM-wrapped
    /// Qwen 3.5 checkpoint. The vision tower owns no token
    /// embedding of its own — token embedding always lives on the text
    /// model.
    fn embed_tokens_module(&self) -> Option<mlxcel_core::layers::UnifiedEmbedding> {
        mlxcel_core::generate::LanguageModel::embed_tokens_module(&self.text_model)
    }

    fn lm_head_module(&self) -> Option<mlxcel_core::layers::UnifiedLinear> {
        mlxcel_core::generate::LanguageModel::lm_head_module(&self.text_model)
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
