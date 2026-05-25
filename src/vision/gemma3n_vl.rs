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

//! Gemma 3n Vision-Language Model
//!
//! MobileNetV5 vision encoder + Gemma3n language model

use super::{
    encoders, gemma3n_per_layer_inputs_state::Gemma3nPerLayerInputsState, merge, processors,
};
use crate::LanguageModel;
use mlxcel_core::cache::SequenceId;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

/// Gemma 3n VLM: MobileNetV5 vision encoder + Gemma3n language model
///
/// Unlike ViT-based VLMs, Gemma 3n uses a convolutional MobileNetV5 encoder
/// with Multi-Scale Fusion Adapter. The language model has a unique
/// `per_layer_inputs` mechanism that requires special handling: the
/// projected tensor is produced during `get_input_embeddings` and consumed
/// later by the prefill `forward_with_embeddings_and_sequence_id` call.
///
/// Issue #85: the `per_layer_inputs` tensor used to live in a single
/// `RefCell<Option<UniquePtr<MlxArray>>>` field on this struct. Under
/// concurrent VLM requests on `mlxcel-server`, two prepares could overwrite
/// the slot before either prefill consumed it, leaking one row's tensor
/// into another row's prefill (or panicking on `Option::unwrap()` when the
/// slot was already drained). The state is now held in
/// [`Gemma3nPerLayerInputsState`], a per-`SequenceId` map with a fallback
/// slot for legacy single-instance callers, mirroring the Gemma 4
/// [`Gemma4PerLayerInputsState`](crate::vision::gemma4_per_layer_inputs_state::Gemma4PerLayerInputsState)
/// container.
pub struct Gemma3nVLModel {
    pub text_model: crate::models::Gemma3nModel,
    pub vision_tower: encoders::gemma3n::Gemma3nVisionModel,
    pub embed_vision: crate::models::gemma3n::Gemma3nMultimodalEmbedder,
    pub processor: processors::siglip::SigLipProcessor, // 224x224
    pub image_token_id: i32,                            // 262_145 (<image_soft_token>)
    pub boi_token_id: i32,                              // 255_999 (<start_of_image>)
    pub eoi_token_id: i32,                              // 262_144 (<end_of_image>)
    pub vision_hidden_size: usize,                      // 2048
    /// Per-sequence storage for the projected `per_layer_inputs` tensor
    /// that is produced during embedding prep and consumed during the
    /// prefill `forward_with_embeddings_and_sequence_id` call. See module
    /// [`crate::vision::gemma3n_per_layer_inputs_state`] for the lifecycle.
    per_layer_inputs_state: Gemma3nPerLayerInputsState,
}

impl Gemma3nVLModel {
    pub fn new(
        text_model: crate::models::Gemma3nModel,
        vision_tower: encoders::gemma3n::Gemma3nVisionModel,
        embed_vision: crate::models::gemma3n::Gemma3nMultimodalEmbedder,
        processor: processors::siglip::SigLipProcessor,
        image_token_id: i32,
        boi_token_id: i32,
        eoi_token_id: i32,
        vision_hidden_size: usize,
    ) -> Self {
        Self {
            text_model,
            vision_tower,
            embed_vision,
            processor,
            image_token_id,
            boi_token_id,
            eoi_token_id,
            vision_hidden_size,
            per_layer_inputs_state: Gemma3nPerLayerInputsState::new(),
        }
    }

    /// Get input embeddings with vision features merged in.
    ///
    /// Side effect: writes the freshly projected `per_layer_inputs` tensor
    /// into the per-layer-inputs state container's fallback slot. The
    /// scheduler binds it to a `SequenceId` right after
    /// `prepare_request_vlm_embeddings` returns (see issue #85). Legacy
    /// single-row callers (CLI `mlxcel generate`, `mlxcel-bench-decode`,
    /// single-row tests) consume it via `take_fallback` on the next
    /// prefill.
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
    ) -> merge::InputEmbeddings {
        // 1. Text embeddings
        let inputs_embeds = self.text_model.language_model.get_embed_tokens(input_ids);

        // 2. Per-layer inputs (image_token_id >= vocab_size_per_layer, auto-zeroed)
        let per_layer_inputs = self
            .text_model
            .language_model
            .get_per_layer_inputs(input_ids);
        let per_layer_inputs = self
            .text_model
            .language_model
            .project_per_layer_inputs(&inputs_embeds, &per_layer_inputs);

        // 3. Vision: pixel_values → VisionTower → [B, H, W, C] (NHWC)
        let embed_dtype = mlxcel_core::array_dtype(&inputs_embeds);
        let pv = mlxcel_core::astype(pixel_values, embed_dtype);
        let vision_out = self.vision_tower.forward(&pv);

        // Reshape NHWC → [B, num_patches, hidden_size]
        let vo = mlxcel_core::transpose_axes(&vision_out, &[0, 3, 1, 2]);
        let vo_shape = mlxcel_core::array_shape(&vo);
        let b = vo_shape[0];
        let c = vo_shape[1]; // hidden_size (2048)
        let num_patches = vo_shape[2] * vo_shape[3]; // H*W
        let vo = mlxcel_core::reshape(&vo, &[b, c, num_patches]);
        let vo = mlxcel_core::transpose_axes(&vo, &[0, 2, 1]); // [B, num_patches, hidden_size]

        // Scale by sqrt(vision_hidden_size)
        let scale = mlxcel_core::full_f32(
            &[1],
            (self.vision_hidden_size as f32).sqrt(),
            mlxcel_core::dtype::FLOAT32,
        );
        let vo = mlxcel_core::multiply(&vo, &scale);

        // 4. Multimodal embedder: → [B, num_patches, text_hidden]
        let image_features = self.embed_vision.forward_soft(&vo);

        // 5. masked_scatter merge
        let merged = merge::merge_llava(
            self.image_token_id,
            &image_features,
            &inputs_embeds,
            input_ids,
        );

        // 6. Park the projected per_layer_inputs in the state container's
        //    fallback slot. The scheduler's
        //    `bind_gemma3n_per_layer_inputs_to_sequence` immediately
        //    transfers it into the per-sequence map under the request's
        //    `SequenceId`. Single-row callers (bench / CLI) consume it
        //    via `take_fallback` on the next prefill.
        self.per_layer_inputs_state
            .set_fallback(Some(per_layer_inputs));

        merged
    }

    // -- Per-sequence per_layer_inputs (issue #85) --------------------
    //
    // Thin wrappers around `Gemma3nPerLayerInputsState`. The scheduler
    // calls them via `LoadedModel::*` capability helpers (see
    // `loaded_model_capabilities.rs`) right after
    // `prepare_request_vlm_embeddings`, mirroring the Gemma 4 binding
    // flow in `gemma4_per_layer_inputs_state`.

    /// Drain the container's fallback slot into the per-`SequenceId` map
    /// under `seq_id`. No-op when the slot is empty (text-only request
    /// after a Gemma 3n VLM model load).
    pub fn bind_per_layer_inputs_to_sequence(&self, seq_id: SequenceId) {
        self.per_layer_inputs_state
            .bind_fallback_to_sequence(seq_id);
    }

    /// Drop a sequence's stored `per_layer_inputs`. Called from
    /// [`Self::release_sequence_state_by_id`] so the map drains in step
    /// with the scheduler's per-sequence cache cleanup.
    pub fn release_per_layer_inputs(&self, seq_id: SequenceId) {
        self.per_layer_inputs_state.release_sequence(seq_id);
    }

    /// Take a sequence's tensor out of the map without dropping the
    /// `UniquePtr`. Used by the scheduler's preemption path so the
    /// tensor can be carried across the eviction (which releases the old
    /// sequence id) and reinstalled under the freshly allocated id with
    /// [`Self::install_per_layer_inputs_for_sequence`].
    pub fn take_per_layer_inputs_for_sequence(
        &self,
        seq_id: SequenceId,
    ) -> Option<UniquePtr<MlxArray>> {
        self.per_layer_inputs_state.take_for_sequence(seq_id)
    }

    /// Re-install a previously taken tensor under `seq_id`. No-op when
    /// the snapshot is `None`.
    pub fn install_per_layer_inputs_for_sequence(
        &self,
        seq_id: SequenceId,
        snapshot: Option<UniquePtr<MlxArray>>,
    ) {
        if let Some(value) = snapshot {
            self.per_layer_inputs_state.bind_for_sequence(seq_id, value);
        }
    }

    /// Internal helper: resolve the active per_layer_inputs tensor for a
    /// VLM prefill. Prefers the per-`SequenceId` slot (server flow) and
    /// falls back to the fallback slot (CLI / bench / single-row).
    fn take_per_layer_inputs(&self, seq_id: Option<SequenceId>) -> Option<UniquePtr<MlxArray>> {
        match seq_id {
            Some(id) => self
                .per_layer_inputs_state
                .take_for_sequence(id)
                .or_else(|| self.per_layer_inputs_state.take_fallback()),
            None => self.per_layer_inputs_state.take_fallback(),
        }
    }

    fn align_per_layer_inputs_to_embeddings(
        per_layer_inputs: &MlxArray,
        input_embeddings: &MlxArray,
    ) -> Option<UniquePtr<MlxArray>> {
        let pli_shape = mlxcel_core::array_shape(per_layer_inputs);
        let embed_shape = mlxcel_core::array_shape(input_embeddings);
        let current_seq = pli_shape[1];
        let target_seq = embed_shape[1];

        if current_seq == target_seq {
            return None;
        }

        if current_seq > target_seq {
            return Some(mlxcel_core::slice(
                per_layer_inputs,
                &[0, 0, 0, 0],
                &[pli_shape[0], target_seq, pli_shape[2], pli_shape[3]],
            ));
        }

        let pad_rows = target_seq - current_seq;
        let dtype = mlxcel_core::array_dtype(per_layer_inputs);
        let padding =
            mlxcel_core::zeros(&[pli_shape[0], pad_rows, pli_shape[2], pli_shape[3]], dtype);
        Some(mlxcel_core::concatenate(per_layer_inputs, &padding, 1))
    }
}

impl LanguageModel for Gemma3nVLModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model.language_model.forward(input_ids, caches)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // No `seq_id` plumbed through (legacy CLI / bench / direct call).
        // Route to the per-sequence dispatch with `seq_id == None`, which
        // resolves the per_layer_inputs from the fallback slot.
        self.forward_with_embeddings_and_sequence_id(
            input_ids,
            input_embeddings,
            None,
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
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        if let Some(embeds) = input_embeddings {
            // VLM prefill: resolve the request's projected per_layer_inputs
            // from the per-`SequenceId` map (server flow) or the fallback
            // slot (CLI / bench). An absent tensor at this point is an
            // internal invariant violation: `get_input_embeddings` must
            // have populated the fallback slot before this consumer ran.
            let pli = self
                .take_per_layer_inputs(seq_id)
                .expect("gemma3n VLM prefill: per_layer_inputs missing for this sequence");
            // Issue #736: M5 tile-aligned prefill pads the token stream and
            // merged embeddings to an NA tile length. The projected
            // per-layer tensor is produced before that generic padding step,
            // so align it here before Gemma3n's per-layer blend.
            let aligned_pli = Self::align_per_layer_inputs_to_embeddings(&pli, embeds);
            let pli_ref = aligned_pli
                .as_ref()
                .map_or_else(|| pli.as_ref().unwrap(), |array| array.as_ref().unwrap());
            self.text_model
                .language_model
                .forward_with_inputs(embeds, pli_ref, caches)
        } else {
            self.text_model.language_model.forward(input_ids, caches)
        }
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.text_model.language_model.get_embed_tokens(input_ids))
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

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        // Issue #85: drop the per-sequence `per_layer_inputs` alongside
        // the text model's per-sequence cache release so the map cannot
        // grow without bound across long-running server sessions.
        self.per_layer_inputs_state.release_sequence(seq_id);
        // The text model (Gemma3nModel) uses the trait's default no-op
        // for release_sequence_state_by_id today; the call below is kept
        // for forward-compat when the text backbone gains per-seq state.
        mlxcel_core::generate::LanguageModel::release_sequence_state_by_id(
            &self.text_model,
            seq_id,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::Gemma3nVLModel;

    #[test]
    fn align_per_layer_inputs_pads_to_embedding_sequence_length() {
        let per_layer_inputs = mlxcel_core::zeros(&[1, 273, 2, 256], mlxcel_core::dtype::FLOAT32);
        let embeddings = mlxcel_core::zeros(&[1, 288, 128], mlxcel_core::dtype::FLOAT32);

        let aligned =
            Gemma3nVLModel::align_per_layer_inputs_to_embeddings(&per_layer_inputs, &embeddings)
                .expect("expected padding for shorter per_layer_inputs");

        assert_eq!(mlxcel_core::array_shape(&aligned), vec![1, 288, 2, 256]);
    }

    #[test]
    fn align_per_layer_inputs_slices_to_embedding_sequence_length() {
        let per_layer_inputs = mlxcel_core::zeros(&[1, 288, 2, 256], mlxcel_core::dtype::FLOAT32);
        let embeddings = mlxcel_core::zeros(&[1, 273, 128], mlxcel_core::dtype::FLOAT32);

        let aligned =
            Gemma3nVLModel::align_per_layer_inputs_to_embeddings(&per_layer_inputs, &embeddings)
                .expect("expected slicing for longer per_layer_inputs");

        assert_eq!(mlxcel_core::array_shape(&aligned), vec![1, 273, 2, 256]);
    }

    #[test]
    fn align_per_layer_inputs_leaves_matching_sequence_length_untouched() {
        let per_layer_inputs = mlxcel_core::zeros(&[1, 273, 2, 256], mlxcel_core::dtype::FLOAT32);
        let embeddings = mlxcel_core::zeros(&[1, 273, 128], mlxcel_core::dtype::FLOAT32);

        let aligned =
            Gemma3nVLModel::align_per_layer_inputs_to_embeddings(&per_layer_inputs, &embeddings);

        assert!(aligned.is_none());
    }
}
