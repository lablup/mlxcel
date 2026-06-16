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

//! Gemma 4 Unified (`gemma4_unified`) multimodal model.
//!
//! A single text + vision + audio model that reuses the shared Gemma 4 text
//! backbone ([`crate::models::Gemma4Wrapper`]). Unlike the ViT/Conformer-backed
//! `gemma4` VLM, this variant is **encoder-free**: images become soft tokens
//! via a small patch projector
//! ([`crate::vision::encoders::gemma4_unified::Gemma4UnifiedVisionEmbedder`]) and
//! audio via raw-waveform chunking, both projected into the language-model
//! hidden space by the shared
//! [`crate::vision::gemma4_multimodal_embedder::Gemma4MultimodalEmbedder`].
//!
//! The one decoder-side novelty is **blockwise bidirectional attention** over
//! image/video token spans during prefill (issue §6), driven by the per-token
//! vision block ids computed here and applied inside the text model.

use super::gemma4_multimodal_embedder::{Gemma4MultimodalEmbedder, masked_scatter};
use super::gemma4_per_layer_inputs_state::Gemma4PerLayerInputsState;
use super::processors::gemma4_unified::{Gemma4UnifiedImageInput, Gemma4UnifiedProcessor};
use super::{encoders, merge};
use crate::LanguageModel;
use mlxcel_core::cache::{SequenceId, SequenceStateLayout};
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

pub use super::gemma4_unified_mask::{
    UnifiedTokenIds, compute_vision_block_ids, derive_mm_token_type_ids, token_type,
};

/// Gemma 4 Unified model.
pub struct Gemma4UnifiedModel {
    pub text_model: crate::models::Gemma4Wrapper,
    pub vision_embedder: encoders::gemma4_unified::Gemma4UnifiedVisionEmbedder,
    pub embed_vision: Gemma4MultimodalEmbedder,
    pub embed_audio: Option<Gemma4MultimodalEmbedder>,
    pub processor: Gemma4UnifiedProcessor,
    pub image_token_id: i32,
    pub video_token_id: i32,
    pub audio_token_id: i32,
    pub boi_token_id: i32,
    pub eoi_token_id: i32,
    pub boa_token_id: i32,
    pub eoa_token_id: i32,
    /// Whether the checkpoint requests blockwise bidirectional vision attention.
    use_bidirectional_vision: bool,
    per_layer_inputs_state: Gemma4PerLayerInputsState,
    _weight_backing: crate::models::Gemma4WeightBacking,
}

impl Gemma4UnifiedModel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        text_model: crate::models::Gemma4Wrapper,
        vision_embedder: encoders::gemma4_unified::Gemma4UnifiedVisionEmbedder,
        embed_vision: Gemma4MultimodalEmbedder,
        processor: Gemma4UnifiedProcessor,
        image_token_id: i32,
        video_token_id: i32,
        boi_token_id: i32,
        eoi_token_id: i32,
    ) -> Self {
        let use_bidirectional_vision = text_model
            .text_config()
            .uses_bidirectional_vision_attention();
        Self {
            text_model,
            vision_embedder,
            embed_vision,
            embed_audio: None,
            processor,
            image_token_id,
            video_token_id,
            audio_token_id: 258_881,
            boi_token_id,
            eoi_token_id,
            boa_token_id: 256_000,
            eoa_token_id: 258_883,
            use_bidirectional_vision,
            per_layer_inputs_state: Gemma4PerLayerInputsState::new(),
            _weight_backing: crate::models::Gemma4WeightBacking::default(),
        }
    }

    pub(crate) fn set_weight_backing(&mut self, backing: crate::models::Gemma4WeightBacking) {
        self._weight_backing = backing;
    }

    /// Attach the audio feature embedder for audio-capable checkpoints.
    pub fn set_audio(
        &mut self,
        embed_audio: Gemma4MultimodalEmbedder,
        audio_token_id: i32,
        boa_token_id: i32,
        eoa_token_id: i32,
    ) {
        self.embed_audio = Some(embed_audio);
        self.audio_token_id = audio_token_id;
        self.boa_token_id = boa_token_id;
        self.eoa_token_id = eoa_token_id;
    }

    /// Compute merged input embeddings for a text/image/audio prompt.
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        images: &[Gemma4UnifiedImageInput],
    ) -> merge::InputEmbeddings {
        self.get_input_embeddings_with_audio(input_ids, images, None, None)
    }

    /// Compute merged input embeddings, optionally scattering audio frames.
    pub fn get_input_embeddings_with_audio(
        &self,
        input_ids: &MlxArray,
        images: &[Gemma4UnifiedImageInput],
        audio_features: Option<&MlxArray>,
        audio_mask: Option<&MlxArray>,
    ) -> merge::InputEmbeddings {
        // Text embeddings, scaled by sqrt(hidden) once. Vision/audio features
        // are already in the LM hidden space and must NOT be scaled again.
        let inputs_embeds = self.text_model.input_embeddings(input_ids);
        let inputs_embeds = mlxcel_core::multiply_scalar(
            &inputs_embeds,
            (self.text_model.hidden_size() as f32).sqrt(),
        );

        // Per-layer inputs are computed from text-only ids (image/video/audio
        // placeholders zeroed). No-op when the backbone has no per-layer table
        // (`hidden_size_per_layer_input == 0`, the reference 12B case).
        let per_layer_inputs = self.build_per_layer_inputs(input_ids, &inputs_embeds);

        // Encoder-free vision: project each image's patches, concat along the
        // token axis, and scatter into image_token_id placeholders.
        let mut result_embeds = if images.is_empty() {
            merge::InputEmbeddings {
                inputs_embeds: mlxcel_core::copy(&inputs_embeds),
                attention_mask_4d: None,
            }
        } else {
            let mut features = Vec::with_capacity(images.len());
            for image in images {
                let patch_feat = self
                    .vision_embedder
                    .forward(&image.patches, &image.positions);
                let projected = self.embed_vision.forward(&patch_feat);
                // Keep only the real (non-padding) patch rows so the count
                // aligns exactly with the image placeholder tokens the
                // processor emitted (`image.num_soft_tokens`). The padded rows
                // sit at the tail (indices >= num_soft_tokens) and are dropped.
                let shape = mlxcel_core::array_shape(&projected);
                let real = (image.num_soft_tokens as i32).min(shape[0]);
                let trimmed = mlxcel_core::slice(&projected, &[0, 0], &[real, shape[1]]);
                // [real, hidden] -> [1, real, hidden].
                features.push(mlxcel_core::reshape(&trimmed, &[1, real, shape[1]]));
            }
            let merged = if features.len() == 1 {
                mlxcel_core::astype(
                    features[0].as_ref().unwrap(),
                    mlxcel_core::array_dtype(&inputs_embeds),
                )
            } else {
                let cat = crate::vision::encoders::qwen2_vl::concat_many(&features, 1);
                mlxcel_core::astype(&cat, mlxcel_core::array_dtype(&inputs_embeds))
            };
            merge::merge_llava(self.image_token_id, &merged, &inputs_embeds, input_ids)
        };

        // Audio: chunk-projected frames scattered into audio_token_id slots.
        if let (Some(embed_audio), Some(audio_feat)) = (&self.embed_audio, audio_features) {
            let audio_encodings = embed_audio.forward(audio_feat);
            let current = &result_embeds.inputs_embeds;
            let audio_encodings =
                mlxcel_core::astype(&audio_encodings, mlxcel_core::array_dtype(current));

            let audio_token_arr = mlxcel_core::from_slice_i32(&[self.audio_token_id], &[1]);
            let is_audio = mlxcel_core::equal(input_ids, &audio_token_arr);
            // The validity mask is not needed for the scatter: every chunked
            // frame is a real audio soft token (zero-padded if partial), so the
            // placeholder count equals num_frames equals the projected-feature
            // count. The mask field is carried for callers that introspect frame
            // validity but is not consumed during embedding merge.
            let _ = audio_mask;
            let audio_mask_expanded = mlxcel_core::expand_dims(&is_audio, -1);
            let audio_mask_expanded =
                mlxcel_core::broadcast_to(&audio_mask_expanded, &mlxcel_core::array_shape(current));
            let scattered = masked_scatter(current, &audio_mask_expanded, &audio_encodings);
            result_embeds.inputs_embeds = scattered;
        }

        self.per_layer_inputs_state.set_fallback(per_layer_inputs);
        result_embeds
    }

    /// Build text-masked per-layer inputs (image/video/audio placeholders
    /// zeroed before the per-layer embedding lookup). Returns `None` when the
    /// backbone has no per-layer input table.
    fn build_per_layer_inputs(
        &self,
        input_ids: &MlxArray,
        inputs_embeds: &MlxArray,
    ) -> Option<UniquePtr<MlxArray>> {
        let is_mm = self.multimodal_token_mask(input_ids);
        let zero_ids = mlxcel_core::zeros_like(input_ids);
        let text_only_ids = mlxcel_core::where_cond(&is_mm, &zero_ids, input_ids);
        self.text_model
            .get_per_layer_inputs(&text_only_ids)
            .map(|per_layer| {
                self.text_model
                    .project_per_layer_inputs(inputs_embeds, Some(per_layer.as_ref().unwrap()))
                    .expect("Gemma4 Unified projected per-layer inputs")
            })
    }

    /// `input_ids == image | video | audio` bool mask.
    fn multimodal_token_mask(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        let image = mlxcel_core::from_slice_i32(&[self.image_token_id], &[1]);
        let video = mlxcel_core::from_slice_i32(&[self.video_token_id], &[1]);
        let is_image = mlxcel_core::equal(input_ids, &image);
        let is_video = mlxcel_core::equal(input_ids, &video);
        let mut mask = mlxcel_core::logical_or(&is_image, &is_video);
        if self.embed_audio.is_some() {
            let audio = mlxcel_core::from_slice_i32(&[self.audio_token_id], &[1]);
            let is_audio = mlxcel_core::equal(input_ids, &audio);
            mask = mlxcel_core::logical_or(&mask, &is_audio);
        }
        mask
    }

    /// Multimodal token ids used for type/block derivation.
    fn unified_token_ids(&self) -> UnifiedTokenIds {
        UnifiedTokenIds {
            image: self.image_token_id,
            video: self.video_token_id,
            // When audio is disabled, use an out-of-range sentinel so no token
            // is ever classified as audio.
            audio: if self.embed_audio.is_some() {
                self.audio_token_id
            } else {
                i32::MIN
            },
        }
    }

    /// Derive `mm_token_type_ids` (issue §6) from a host token-id slice.
    pub fn mm_token_type_ids(&self, input_ids_host: &[i32]) -> Vec<i32> {
        derive_mm_token_type_ids(input_ids_host, self.unified_token_ids())
    }

    /// Compute the per-position vision block-id vector for the blockwise
    /// bidirectional overlay from a host token-id slice (see
    /// [`compute_vision_block_ids`]).
    pub fn vision_block_ids(&self, input_ids_host: &[i32]) -> Option<Vec<i32>> {
        compute_vision_block_ids(
            input_ids_host,
            self.unified_token_ids(),
            self.use_bidirectional_vision,
        )
    }

    // -- per-sequence per_layer_inputs binding (mirrors Gemma4VLModel) --------

    pub fn bind_per_layer_inputs_to_sequence(&self, seq_id: SequenceId) {
        self.per_layer_inputs_state
            .bind_fallback_to_sequence(seq_id);
    }

    pub fn take_per_layer_inputs_for_sequence(
        &self,
        seq_id: SequenceId,
    ) -> Option<UniquePtr<MlxArray>> {
        self.per_layer_inputs_state.take_for_sequence(seq_id)
    }

    pub fn install_per_layer_inputs_for_sequence(
        &self,
        seq_id: SequenceId,
        snapshot: Option<UniquePtr<MlxArray>>,
    ) {
        if let Some(value) = snapshot {
            self.per_layer_inputs_state.bind_for_sequence(seq_id, value);
        }
    }

    // -- Gemma 4 MTP speculative-decoding hooks ------------------------------
    //
    // Pass-through methods that give the MTP drafter / generator the same
    // opt-in sink + rollback surface on the Unified-wrapped Gemma 4 path as on
    // the text-only and VLM paths. Mirrors `src/vision/gemma4_vl.rs` exactly:
    // the (optionally multimodal) prefill flows through the unified forward
    // that merges vision/audio features and seeds the cache + first hidden;
    // speculative decode then kicks in only AFTER the prefill tail, at which
    // point the rotating + dense KV caches owned by the inner `Gemma4Wrapper`
    // are the sole state the MTP loop touches. The encoder-free vision
    // embedder and `per_layer_inputs_state` do not participate.

    /// Sink-aware forward for the Unified-wrapped Gemma 4 text model.
    ///
    /// Delegates to [`crate::models::Gemma4Wrapper::forward_with_speculative_sinks`]
    /// on the inner text model. Like the VLM path, the caller routes the
    /// (optionally multimodal) prefill through the existing
    /// [`LanguageModel::forward_with_embeddings_and_sequence_id`] path FIRST;
    /// subsequent speculative decode steps consume `input_ids` only and pass
    /// `input_embeddings = None`, so no multimodal merge occurs during decode.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_speculative_sinks(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        per_layer_inputs: Option<&MlxArray>,
        mask: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        capture_layer_ids: Option<&[usize]>,
        sinks: Option<&mut crate::models::Gemma4SpeculativeSinks>,
        per_row_valid_end: Option<&[i32]>,
    ) -> UniquePtr<MlxArray> {
        self.text_model.forward_with_speculative_sinks(
            input_ids,
            input_embeddings,
            per_layer_inputs,
            mask,
            seq_id,
            capture_layer_ids,
            sinks,
            per_row_valid_end,
        )
    }

    /// Sink-aware forward against a **caller-owned** `[B, ...]` cache vector
    /// (batched MTP dispatch).
    ///
    /// Pure pass-through to
    /// [`crate::models::Gemma4Wrapper::forward_with_speculative_sinks_explicit_cache`].
    /// The Unified wrapper holds no batched speculative state of its own — the
    /// batched MTP target adapter owns the `[B, ...]` cache and the inner text
    /// model advances all rows through one forward.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_speculative_sinks_explicit_cache(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        per_layer_inputs: Option<&MlxArray>,
        mask: Option<&MlxArray>,
        caches: &mut [crate::models::gemma4::Cache],
        capture_layer_ids: Option<&[usize]>,
        sinks: Option<&mut crate::models::Gemma4SpeculativeSinks>,
        left_padding: Option<&[i32]>,
        per_row_valid_end: Option<&[i32]>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_with_speculative_sinks_explicit_cache(
                input_ids,
                input_embeddings,
                per_layer_inputs,
                mask,
                caches,
                capture_layer_ids,
                sinks,
                left_padding,
                per_row_valid_end,
            )
    }

    /// Rewind the inner text model's per-sequence KV caches after a Gemma 4
    /// MTP verify pass. Pure pass-through to
    /// [`crate::models::Gemma4Wrapper::rollback_speculative_cache`] — the
    /// Unified wrapper holds no additional speculative state to undo.
    pub fn rollback_speculative_cache(
        &self,
        seq_id: Option<SequenceId>,
        accepted: &[i32],
        block_size: i32,
    ) -> Result<(), String> {
        self.text_model
            .rollback_speculative_cache(seq_id, accepted, block_size)
    }

    /// Normalize a pre-norm hidden state with the Gemma 4 final norm before it
    /// is handed to the MTP assistant drafter. Pure pass-through to
    /// [`crate::models::Gemma4Wrapper::speculative_draft_hidden`].
    pub fn speculative_draft_hidden(&self, hidden: &MlxArray) -> UniquePtr<MlxArray> {
        self.text_model.speculative_draft_hidden(hidden)
    }

    /// Build the per-position vision block-id tensor from `input_ids` for the
    /// bidirectional overlay during the embeddings-driven prefill forward, or
    /// `None` when the overlay is disabled.
    ///
    /// Computed entirely with MLX ops (no full host readback): only the two
    /// gate scalars (`has_vision`, `has_audio`) are read back via `sum_all`.
    /// Returns a `[seq_len]` int32 array where each contiguous image/video run
    /// has a distinct non-negative id and every other position is `-1`.
    fn block_ids_array_for(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        if !self.use_bidirectional_vision {
            return None;
        }
        let shape = mlxcel_core::array_shape(input_ids);
        let seq_len = *shape.last()?;
        // Single-row prefill only (batch 1). Batched VLM prefill re-enters one
        // row at a time, so a >2-D or multi-row tensor is not handled here.
        if seq_len <= 1 || (shape.len() == 2 && shape[0] != 1) || shape.len() > 2 {
            return None;
        }

        let ids = mlxcel_core::astype(input_ids, mlxcel_core::dtype::INT32);
        let ids = mlxcel_core::reshape(&ids, &[seq_len]);

        let image = mlxcel_core::from_slice_i32(&[self.image_token_id], &[1]);
        let video = mlxcel_core::from_slice_i32(&[self.video_token_id], &[1]);
        let is_image = mlxcel_core::equal(&ids, &image);
        let is_video = mlxcel_core::equal(&ids, &video);
        let is_vision = mlxcel_core::logical_or(&is_image, &is_video);

        // Gate: vision present AND audio absent. Reduce to scalars.
        let is_vision_i32 = mlxcel_core::astype(&is_vision, mlxcel_core::dtype::INT32);
        let vision_count = mlxcel_core::item_i32(&mlxcel_core::sum_all(&is_vision_i32));
        if vision_count == 0 {
            return None;
        }
        let audio = mlxcel_core::from_slice_i32(&[self.audio_token_id], &[1]);
        let is_audio = mlxcel_core::equal(&ids, &audio);
        let is_audio_i32 = mlxcel_core::astype(&is_audio, mlxcel_core::dtype::INT32);
        let audio_count = mlxcel_core::item_i32(&mlxcel_core::sum_all(&is_audio_i32));
        if audio_count > 0 {
            return None;
        }

        // Block-start[i] = is_vision[i] && !is_vision[i-1]. Build the shifted
        // (previous-position) vision mask by padding a 0 in front and dropping
        // the last element.
        let prev_padded = mlxcel_core::pad(&is_vision_i32, &[1, 0], 0.0); // [seq_len + 1]
        let prev = mlxcel_core::slice(&prev_padded, &[0], &[seq_len]); // [seq_len]
        let zero = mlxcel_core::from_slice_i32(&[0], &[1]);
        let prev_off = mlxcel_core::equal(&prev, &zero); // !is_vision[i-1]
        let start = mlxcel_core::logical_and(&is_vision, &prev_off);
        let start_i32 = mlxcel_core::astype(&start, mlxcel_core::dtype::INT32);

        // block_num = cumsum(start) (1-based within vision runs).
        let block_num = mlxcel_core::cumsum(&start_i32, 0, false, true);
        let one = mlxcel_core::from_slice_i32(&[1], &[1]);
        let block_id_vision = mlxcel_core::subtract(&block_num, &one); // 0-based
        let neg_one = mlxcel_core::from_slice_i32(&[-1], &[1]);
        // Where not vision, use -1.
        let block_ids = mlxcel_core::where_cond(&is_vision, &block_id_vision, &neg_one);
        Some(mlxcel_core::astype(&block_ids, mlxcel_core::dtype::INT32))
    }
}

impl LanguageModel for Gemma4UnifiedModel {
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
        self.forward_with_embeddings_and_sequence_id(
            input_ids,
            input_embeddings,
            None,
            caches,
            mask,
        )
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

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        if let Some(embeds) = input_embeddings {
            let per_layer_inputs = match seq_id {
                Some(id) => self
                    .per_layer_inputs_state
                    .take_for_sequence(id)
                    .or_else(|| self.per_layer_inputs_state.take_fallback()),
                None => self.per_layer_inputs_state.take_fallback(),
            };
            // Blockwise bidirectional overlay: build per-position vision block
            // ids from the prompt token stream (image/video-only prefill).
            let block_ids = if mask.is_none() {
                self.block_ids_array_for(input_ids)
            } else {
                None
            };
            self.text_model.forward_unified_with_inputs_and_sequence_id(
                input_ids,
                Some(embeds),
                per_layer_inputs.as_ref().and_then(|arr| arr.as_ref()),
                mask,
                seq_id,
                block_ids.as_ref().and_then(|arr| arr.as_ref()),
            )
        } else {
            self.text_model
                .forward_with_sequence_id(input_ids, seq_id, caches, mask)
        }
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.text_model.input_embeddings(input_ids))
    }

    fn embed_tokens_module(&self) -> Option<mlxcel_core::layers::UnifiedEmbedding> {
        mlxcel_core::generate::LanguageModel::embed_tokens_module(&self.text_model)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.text_model.make_caches()
    }

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        self.text_model.sequence_state_layout()
    }

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        self.text_model.prepare_sequence_state(seq_id);
    }

    fn reset_runtime_state(&self) {
        self.text_model.reset_runtime_state();
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.per_layer_inputs_state.release_sequence(seq_id);
        self.text_model.release_sequence_state_by_id(seq_id);
    }

    fn supports_snapshot_reuse(&self) -> bool {
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

    fn num_layers(&self) -> usize {
        self.text_model.num_layers_value()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.text_model.eos_token_ids_value()
    }

    fn supports_padded_prefill(&self) -> bool {
        false
    }

    fn supports_batching(&self) -> bool {
        // Vision prefill needs single-chunk, single-row handling for the
        // bidirectional block mask; keep batched decode off for correctness.
        false
    }
}
