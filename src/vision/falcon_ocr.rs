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

//! Falcon-OCR vision-language runtime.
//!
//! There is no vision tower to compose here: the "encoder" is one linear layer
//! that lives inside the decoder. This wrapper only owns the pieces that are
//! genuinely multimodal, namely image preprocessing, the patch projection and
//! scatter, and the per-request positional state (temporal positions, spatial
//! `(h, w)` coordinates, and the hybrid attention mask) that the decoder
//! consumes during prefill.

use crate::models::falcon_ocr::{FalconOcrPrefillState, FalconOcrTextModel};
use crate::models::falcon_ocr_rope::{
    FalconOcrTokenIds, build_hybrid_mask, rope_delta, spatial_positions, temporal_positions,
};
use crate::vision::merge;
use crate::vision::processors::falcon_ocr::FalconOcrProcessor;
use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct FalconOcrVlModel {
    pub text_model: FalconOcrTextModel,
    pub processor: FalconOcrProcessor,
    /// `<|OCR_PLAIN|>`, appended so a plain `-p "..."` still selects the OCR
    /// task head. `None` when the tokenizer does not carry the token.
    pub ocr_task_token_id: Option<i32>,
    pub eos_token_ids: Vec<i32>,
}

impl FalconOcrVlModel {
    pub fn token_ids(&self) -> FalconOcrTokenIds {
        self.text_model.config.token_ids()
    }

    /// Project image patches into the token stream and stash the positional
    /// state this prompt's prefill needs.
    ///
    /// `patches` is `[total_patches, patch_dim]` from
    /// [`FalconOcrProcessor::preprocess_with_grid`], `grids` the matching
    /// per-image `(rows, cols)`, and `tokens` the host copy of `input_ids`
    /// (positions and the hybrid mask are cheap to derive on the host and would
    /// otherwise force a device readback).
    pub fn input_embeddings(
        &self,
        input_ids: &MlxArray,
        tokens: &[i32],
        patches: &MlxArray,
        grids: &[(i32, i32)],
    ) -> merge::InputEmbeddings {
        let inputs_embeds = self.text_model.embed(input_ids);
        let embed_dtype = mlxcel_core::array_dtype(&inputs_embeds);
        let patches = mlxcel_core::astype(patches, embed_dtype);
        let features = self.text_model.project_patches(&patches);
        let merged = merge::merge_llava(
            self.text_model.config.img_id,
            &features,
            &inputs_embeds,
            input_ids,
        );

        let ids = self.token_ids();
        let positions = temporal_positions(tokens, &ids);
        let delta = rope_delta(&positions);
        let pos_hw = spatial_positions(tokens, &ids, grids);
        let pos_hw = mlxcel_core::from_slice_f32(&pos_hw, &[1, tokens.len() as i32, 2]);
        let mask = build_hybrid_mask(tokens, &ids);

        self.text_model.state.set_current(FalconOcrPrefillState {
            positions,
            pos_hw: Some(pos_hw),
            rope_delta: delta,
        });

        merge::InputEmbeddings {
            inputs_embeds: merged.inputs_embeds,
            attention_mask_4d: Some(mask),
        }
    }

    /// Move the pending prefill state onto a server sequence id so concurrent
    /// requests cannot consume each other's rope delta.
    pub fn bind_state_to_sequence(&self, seq_id: SequenceId) {
        self.text_model.state.bind_to_sequence(seq_id);
    }
}

/// Tokens that must never be sampled during decode.
///
/// The patch placeholder and the block framing tokens carry no text, and
/// emitting one would open a phantom image block that the hybrid mask on a
/// follow-up turn would then treat as bidirectional. `<|end_of_image|>` is
/// deliberately absent: it is outside the bidirectional region and is a
/// legitimate structural token in a multi-image prompt.
pub(crate) fn suppressed_token_ids(ids: &FalconOcrTokenIds) -> Vec<i32> {
    let mut out = vec![ids.img_id, ids.image_cls_token_id];
    out.extend_from_slice(&ids.image_reg_token_ids);
    out
}

impl LanguageModel for FalconOcrVlModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model.forward(input_ids, caches, mask)
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

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.text_model.embed(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.text_model.make_caches()
    }

    fn num_layers(&self) -> usize {
        LanguageModel::num_layers(&self.text_model)
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }

    fn output_suppressed_token_ids(&self) -> Vec<i32> {
        suppressed_token_ids(&self.token_ids())
    }

    fn supports_chunked_prefill(&self) -> bool {
        false
    }

    fn supports_padded_prefill(&self) -> bool {
        false
    }

    fn supports_batching(&self) -> bool {
        false
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.text_model.release_sequence_state_by_id(seq_id);
    }
}

#[cfg(test)]
#[path = "falcon_ocr_tests.rs"]
mod tests;
