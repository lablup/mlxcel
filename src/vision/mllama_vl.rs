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

//! Llama 3.2 Vision (`mllama`) top-level runtime.
//!
//! Faithful port of `Model` in
//! `references/mlx-vlm/mlx_vlm/models/mllama/mllama.py`.
//!
//! Composition:
//! - [`MllamaVisionModel`] tower produces `cross_attention_states`.
//! - `multi_modal_projector` (a `UnifiedLinear` with bias, quantized in the
//!   `-4bit`/`-8bit` releases) maps the tower's `vision_output_dim` features
//!   into the text hidden size.
//! - [`MllamaTextModel`] is a Llama-3 decoder whose cross-attention layers
//!   attend to those projected features.
//!
//! Unlike the LLaVA-style VLMs, mllama does **not** merge image features into
//! the token stream. Instead the projected features are held as
//! `cross_attention_states` and consumed by the gated cross-attention layers.
//! The states keep only each image's REAL tiles: the tower runs on the full
//! zero-padded tile axis (its lanes are not separable, see the encoder tests),
//! but the reference's text-side `cross_attention_mask` gives every
//! padding-tile position an additive `-1e9`, i.e. exactly zero softmax weight,
//! so dropping those rows here is byte-equivalent to the reference while
//! shrinking the projector and cross-attention K/V work by
//! `max_num_tiles / num_real_tiles`.
//! Because [`crate::LanguageModel::forward`] carries no cross-attention slot,
//! the state computed by [`MllamaVLModel::prepare_cross_attention_states`] is
//! held in a [`MllamaImageStateStore`] keyed by the scheduler's `SequenceId`
//! (#2567): preparation writes the fallback slot, admission binds it to the
//! request's sequence, and every forward with that id attends to that image.
//! The CLI (`seq_id == None`) keeps using the fallback. With no image
//! (text-only, or a bound sequence with no entry), the cross-attention layers
//! are a pass-through.

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::{
    BatchedDecodeKind, LanguageModel, route_batched_decode, route_batched_decode_with_ids,
};
use mlxcel_core::layers::{KVCache, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::mllama::MllamaConfig;
use crate::models::mllama::text::{CrossAttentionInput, MllamaTextModel};
use crate::models::qwen_mrope_state::{
    concatenate_scattered, scatter_group_logits, select_cache_rows, take_rows,
};
use crate::multimodal::batched_dispatch::forward_batched_with_seq_ids_dispatch;
use crate::vision::encoders::mllama::MllamaVisionModel;
use crate::vision::mllama_image_state::{MllamaImageState, MllamaImageStateStore};
use crate::vision::processors::mllama::{MllamaImageInputs, MllamaImageProcessor};

/// The Llama 3.2 Vision runtime.
pub struct MllamaVLModel {
    pub text_model: MllamaTextModel,
    pub vision_tower: MllamaVisionModel,
    pub multi_modal_projector: UnifiedLinear,
    pub processor: MllamaImageProcessor,
    pub config: MllamaConfig,
    pub eos_token_ids: Vec<i32>,
    /// Per-request image state: the projected vision features
    /// `[B, kv_len, hidden]` and each cross-attention layer's fill-once K/V,
    /// keyed by `SequenceId` with a fallback slot for the CLI (#2567).
    image_state: MllamaImageStateStore,
}

impl MllamaVLModel {
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        text_model: MllamaTextModel,
        vision_tower: MllamaVisionModel,
        multi_modal_projector: UnifiedLinear,
        processor: MllamaImageProcessor,
        config: MllamaConfig,
        eos_token_ids: Vec<i32>,
    ) -> Self {
        Self {
            text_model,
            vision_tower,
            multi_modal_projector,
            processor,
            config,
            eos_token_ids,
            image_state: MllamaImageStateStore::new(),
        }
    }

    /// Load the `multi_modal_projector` linear (`vision_output_dim -> hidden`).
    ///
    /// Quantized in the `-4bit`/`-8bit` releases (`.scales`/`.biases` alongside
    /// a float `.bias`); [`UnifiedLinear`] auto-detects this and falls back to a
    /// plain linear on a dense checkpoint.
    pub fn load_projector(
        weights: &WeightMap,
        group_size: i32,
        bits: i32,
    ) -> Result<UnifiedLinear, String> {
        UnifiedLinear::from_weights(weights, "multi_modal_projector", group_size, bits)
    }

    /// Run the vision tower and projector to obtain the flattened
    /// `cross_attention_states` `[B, num_media * num_tiles * num_patches,
    /// hidden]`. Mirrors the vision branch of `Model.__call__` with an unknown
    /// per-image tile count: every tile lane of the padded tile axis (including
    /// the processor's zero-padding tiles) lands in the states.
    pub fn compute_cross_attention_states(
        &self,
        pixel_values: &MlxArray,
        aspect_ratio_ids: &MlxArray,
        aspect_ratio_mask: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        // An empty tile list never passes `select_real_tile_states`, so this
        // is exactly the legacy all-tiles path.
        self.compute_cross_attention_states_for_tiles(
            pixel_values,
            aspect_ratio_ids,
            aspect_ratio_mask,
            &[],
        )
    }

    /// Like [`Self::compute_cross_attention_states`], but keeps only the REAL
    /// tiles of each image in the resulting states, dropping the processor's
    /// zero-padding tile lanes (`num_tiles` is the per-image real tile count
    /// the processor reports).
    ///
    /// Correctness: the vision tower itself still runs on ALL tile lanes.
    /// Its aspect-ratio mask only blocks padding->padding attention, so the
    /// padding lanes are live register-like inputs to the real tiles' outputs
    /// and cannot be skipped (see the `padding_tile_content_reaches_real_tile
    /// _output` encoder test). The reference then masks every padding-tile
    /// position out of the text cross-attention with an additive `-1e9`
    /// (`cross_attention_mask` built from `num_tiles` in
    /// `processing_mllama.py`), which zeroes those positions' softmax weights
    /// exactly. This port passes no text-side cross mask, so dropping the
    /// padding-tile rows here reproduces the reference's tile-level masking
    /// exactly while shrinking the projector matmul and every cross-attention
    /// key/value computation by `max_num_tiles / num_real_tiles`.
    ///
    /// When the counts do not warrant (or do not safely permit) selection,
    /// this falls back to the legacy all-tiles states.
    pub fn compute_cross_attention_states_for_tiles(
        &self,
        pixel_values: &MlxArray,
        aspect_ratio_ids: &MlxArray,
        aspect_ratio_mask: &MlxArray,
        num_tiles: &[usize],
    ) -> UniquePtr<MlxArray> {
        let batch = mlxcel_core::array_shape(pixel_values)[0];
        let vision_output =
            self.vision_tower
                .forward(pixel_values, aspect_ratio_ids, aspect_ratio_mask);
        let hidden = self.config.text_config.hidden_size as i32;
        match select_real_tile_states(&vision_output, num_tiles) {
            Some(selected) => {
                // Project only the surviving rows (the projector is a
                // per-position linear, so slicing before projecting is exact).
                let projected = self.multi_modal_projector.forward(&selected);
                mlxcel_core::reshape(&projected, &[batch, -1, hidden])
            }
            None => {
                let projected = self.multi_modal_projector.forward(&vision_output);
                mlxcel_core::reshape(&projected, &[batch, -1, hidden])
            }
        }
    }

    /// Compute and stash the cross-attention states from preprocessed image
    /// inputs so subsequent [`LanguageModel::forward`] calls attend to them.
    /// Uses the processor's per-image real tile counts to keep only real-tile
    /// features (see [`Self::compute_cross_attention_states_for_tiles`]).
    pub fn prepare_cross_attention_states(&self, inputs: &MllamaImageInputs) {
        let states = self.compute_cross_attention_states_for_tiles(
            &inputs.pixel_values,
            &inputs.aspect_ratio_ids,
            &inputs.aspect_ratio_mask,
            &inputs.num_tiles,
        );
        self.set_cross_attention_states(states);
    }

    /// Stash externally-computed cross-attention states in the fallback slot.
    ///
    /// The slot carries fresh, unfilled K/V slots for every cross-attention
    /// layer, so the next forward derives the projected key/value from these
    /// states rather than from a previous image. The server moves the slot
    /// under the request's sequence id right after preparation
    /// ([`Self::bind_cross_attention_states_to_sequence`]).
    pub fn set_cross_attention_states(&self, states: UniquePtr<MlxArray>) {
        self.image_state
            .set_fallback(MllamaImageState::new(states, self.text_model.num_layers()));
    }

    /// Drop the fallback image state (revert the CLI path to text-only
    /// decoding). Per-sequence entries are untouched; the scheduler releases
    /// those with their sequences.
    pub fn clear_cross_attention_states(&self) {
        self.image_state.clear_fallback();
    }

    /// `true` when the fallback slot holds image state.
    pub fn has_cross_attention_states(&self) -> bool {
        self.image_state.has_fallback()
    }

    /// `true` when `seq_id` owns image state.
    pub fn has_image_state_for_sequence(&self, seq_id: SequenceId) -> bool {
        self.image_state.has_sequence(seq_id)
    }

    /// Move the image state prepared for the request that is about to prefill
    /// under the scheduler's `seq_id`, draining the fallback so the next
    /// request cannot inherit it. A text-only request (empty fallback) leaves
    /// the sequence without an entry, which runs the cross layers as the
    /// identity.
    pub fn bind_cross_attention_states_to_sequence(&self, seq_id: SequenceId) {
        self.image_state.bind_fallback_to_sequence(seq_id);
    }

    /// Remove and return `seq_id`'s image state (scheduler preemption: the
    /// vision tower does not run again on re-prefill).
    pub fn take_image_state_for_sequence(&self, seq_id: SequenceId) -> Option<MllamaImageState> {
        self.image_state.take_for_sequence(seq_id)
    }

    /// Re-install a taken image state under a (possibly new) `seq_id`.
    pub fn install_image_state_for_sequence(&self, seq_id: SequenceId, state: MllamaImageState) {
        self.image_state.bind_for_sequence(seq_id, state);
    }

    fn run_text(
        &self,
        input_ids: &MlxArray,
        input_embeds: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
    ) -> UniquePtr<MlxArray> {
        self.image_state.with_entry_mut(seq_id, |entry| {
            let cross = entry.map(|state| {
                let MllamaImageState {
                    cross_states,
                    cross_kv,
                } = state;
                CrossAttentionInput::new(cross_states, cross_kv.as_mut_slice())
            });
            self.text_model
                .forward(Some(input_ids), input_embeds, caches, mask, cross)
        })
    }

    /// Anonymous batch (no sequence ids) while the fallback slot holds image
    /// state: every row attends to that one image through the historical
    /// per-row graph. Only non-server callers reach this; the scheduler always
    /// passes ids and goes through [`Self::forward_batched_rows_by_sequence`].
    fn forward_batched_with_fallback_image_state(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
    ) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(input_ids);
        let row_len = shape.get(1).copied().unwrap_or(1);
        let mut rows = Vec::with_capacity(batch_caches.len());
        for (row, caches) in batch_caches.iter_mut().enumerate() {
            let row_input =
                mlxcel_core::slice(input_ids, &[row as i32, 0], &[row as i32 + 1, row_len]);
            rows.push(self.forward(&row_input, caches, None));
        }
        let refs: Vec<&MlxArray> = rows.iter().map(|row| &**row).collect();
        mlxcel_core::concatenate_many(&refs, 0)
    }

    /// Identified decode batch: rows whose sequence owns image state run one
    /// at a time through their own entry (the cross-attention K/V is per
    /// request and B = 1 shaped), while the rows without one batch natively
    /// through [`MllamaTextModel::forward_batched_text_only`]. Logits are
    /// scattered back to scheduler order.
    fn forward_batched_rows_by_sequence(
        &self,
        input_ids: &MlxArray,
        seq_ids: &[SequenceId],
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let (image_rows, text_rows): (Vec<usize>, Vec<usize>) =
            (0..seq_ids.len()).partition(|&row| self.has_image_state_for_sequence(seq_ids[row]));
        if image_rows.is_empty() {
            return self
                .text_model
                .forward_batched_text_only(input_ids, batch_caches, mask);
        }

        let mut outputs: Vec<Option<UniquePtr<MlxArray>>> = std::iter::repeat_with(|| None)
            .take(seq_ids.len())
            .collect();
        for &row in &image_rows {
            let ids = take_rows(input_ids, &[row], 0);
            let logits =
                self.forward_with_sequence_id(&ids, Some(seq_ids[row]), batch_caches[row], None);
            scatter_group_logits(&mut outputs, &[row], &logits);
        }
        if !text_rows.is_empty() {
            let ids = take_rows(input_ids, &text_rows, 0);
            let logits = if text_rows.len() == 1 {
                let row = text_rows[0];
                self.forward_with_sequence_id(&ids, Some(seq_ids[row]), batch_caches[row], None)
            } else {
                let mut caches = select_cache_rows(batch_caches, &text_rows);
                self.text_model
                    .forward_batched_text_only(&ids, &mut caches, None)
            };
            scatter_group_logits(&mut outputs, &text_rows, &logits);
        }
        concatenate_scattered(outputs, "Mllama")
    }
}

/// Select the real-tile rows of the vision tower output for the
/// cross-attention states.
///
/// `vision_output`: `[B, num_media, max_num_tiles, num_patches, dim]` (all
/// tile lanes, as produced by [`MllamaVisionModel::forward`]).
/// `num_tiles[m]`: the number of REAL tiles of image `m`; real tiles are
/// contiguous at tile indices `0..num_tiles[m]` (the processor appends the
/// zero-padding tiles).
///
/// Returns the media-major concatenation
/// `[B, sum(num_tiles) * num_patches, dim]`, which is the reference's
/// flattened `[B, media * tiles * patches, dim]` layout restricted to the
/// positions its text-side `cross_attention_mask` leaves unmasked. Ragged
/// per-image tile counts are handled exactly (each image contributes its own
/// real rows, order preserved).
///
/// Returns `None` (caller keeps the legacy all-tiles path) when:
/// - the batch dimension is not 1 (per-row selection would be ragged across
///   the shared kv axis; the current runtime only ever builds B == 1),
/// - `num_tiles` does not match the media count or holds an out-of-range
///   count (defensive; the processor cannot produce these), or
/// - every image already uses all `max_num_tiles` tiles (selection would be
///   a no-op; the legacy path is byte-identical and avoids extra reshapes).
fn select_real_tile_states(
    vision_output: &MlxArray,
    num_tiles: &[usize],
) -> Option<UniquePtr<MlxArray>> {
    let shape = mlxcel_core::array_shape(vision_output);
    if shape.len() != 5 || shape[0] != 1 {
        return None;
    }
    let (media, max_tiles, patches, dim) = (shape[1], shape[2], shape[3], shape[4]);
    if num_tiles.len() != media as usize
        || num_tiles.iter().any(|&n| n == 0 || n > max_tiles as usize)
        || num_tiles.iter().all(|&n| n == max_tiles as usize)
    {
        return None;
    }

    let mut selected: Option<UniquePtr<MlxArray>> = None;
    for (m, &n) in num_tiles.iter().enumerate() {
        let m = m as i32;
        let part = mlxcel_core::slice(
            vision_output,
            &[0, m, 0, 0, 0],
            &[1, m + 1, n as i32, patches, dim],
        );
        let part = mlxcel_core::reshape(&part, &[1, n as i32 * patches, dim]);
        selected = Some(match selected {
            None => part,
            Some(acc) => mlxcel_core::concatenate(&acc, &part, 1),
        });
    }
    selected
}

impl LanguageModel for MllamaVLModel {
    fn batched_decode_kind(&self) -> BatchedDecodeKind {
        // Text-only rows take the native path; rows with image state keep the
        // per-row graph over their own entry (#2567).
        BatchedDecodeKind::Native
    }

    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.run_text(input_ids, None, caches, mask, None)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.run_text(input_ids, input_embeddings, caches, mask, None)
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.run_text(input_ids, None, caches, mask, seq_id)
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.run_text(input_ids, input_embeddings, caches, mask, seq_id)
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        // The text tower keeps no per-sequence state of its own; the KV caches
        // live in the scheduler slice.
        self.image_state.release_sequence(seq_id);
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.text_model.embed_tokens(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.text_model.make_caches()
    }

    fn num_layers(&self) -> usize {
        self.text_model.num_layers()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }

    fn forward_batched(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        route_batched_decode(self, input_ids, batch_caches, |batch_caches| {
            if self.has_cross_attention_states() {
                self.forward_batched_with_fallback_image_state(input_ids, batch_caches)
            } else {
                self.text_model
                    .forward_batched_text_only(input_ids, batch_caches, mask)
            }
        })
    }

    fn forward_batched_with_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        route_batched_decode_with_ids(self, input_ids, seq_ids, batch_caches, |batch_caches| {
            let Some(seq_ids) = seq_ids.filter(|ids| ids.len() == batch_caches.len()) else {
                return self.forward_batched(input_ids, batch_caches, mask);
            };
            let shape = mlxcel_core::array_shape(input_ids);
            if shape.len() != 2 || shape[1] != 1 {
                return forward_batched_with_seq_ids_dispatch(
                    self,
                    input_ids,
                    Some(seq_ids),
                    batch_caches,
                    mask,
                );
            }
            self.forward_batched_rows_by_sequence(input_ids, seq_ids, batch_caches, mask)
        })
    }
}

#[cfg(test)]
#[path = "mllama_vl_tests.rs"]
mod tests;
