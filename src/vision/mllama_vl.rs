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
//! stashed in an interior-mutable cell (mirroring the Qwen-VL MRoPE-state
//! pattern) and threaded into every decode step until cleared. With no image
//! (text-only), the cell is empty and the cross-attention layers are a
//! pass-through.

use std::cell::RefCell;

use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::{KVCache, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::mllama::MllamaConfig;
use crate::models::mllama::text::MllamaTextModel;
use crate::vision::encoders::mllama::MllamaVisionModel;
use crate::vision::processors::mllama::{MllamaImageInputs, MllamaImageProcessor};

/// The Llama 3.2 Vision runtime.
pub struct MllamaVLModel {
    pub text_model: MllamaTextModel,
    pub vision_tower: MllamaVisionModel,
    pub multi_modal_projector: UnifiedLinear,
    pub processor: MllamaImageProcessor,
    pub config: MllamaConfig,
    pub eos_token_ids: Vec<i32>,
    /// Projected vision features `[B, kv_len, hidden]` for the current request,
    /// or `None` for a text-only request.
    cross_attention_states: RefCell<Option<UniquePtr<MlxArray>>>,
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
            cross_attention_states: RefCell::new(None),
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

    /// Stash externally-computed cross-attention states.
    ///
    /// The text cross-attention layers derive a per-layer key/value from these
    /// states and cache it for the whole generation, so a new image must
    /// invalidate that cache before the next forward rebuilds it.
    pub fn set_cross_attention_states(&self, states: UniquePtr<MlxArray>) {
        *self.cross_attention_states.borrow_mut() = Some(states);
        self.text_model.invalidate_cross_attention_cache();
    }

    /// Drop any stashed cross-attention states (revert to text-only decoding).
    ///
    /// Also drops the cached per-layer image key/value so a later image
    /// rebuilds it and the cross-attention layers stay pass-through meanwhile.
    pub fn clear_cross_attention_states(&self) {
        *self.cross_attention_states.borrow_mut() = None;
        self.text_model.invalidate_cross_attention_cache();
    }

    /// `true` when image cross-attention state is currently active.
    pub fn has_cross_attention_states(&self) -> bool {
        self.cross_attention_states.borrow().is_some()
    }

    fn run_text(
        &self,
        input_ids: &MlxArray,
        input_embeds: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let cross = self.cross_attention_states.borrow();
        let cross_ref = cross.as_deref();
        self.text_model.forward(
            Some(input_ids),
            input_embeds,
            caches,
            mask,
            cross_ref,
            None,
            None,
        )
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
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.run_text(input_ids, None, caches, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.run_text(input_ids, input_embeddings, caches, mask)
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
}

#[cfg(test)]
mod tests {
    use super::{MllamaVLModel, select_real_tile_states};
    use mlxcel_core::weights::WeightMap;
    use mlxcel_core::{MlxArray, UniquePtr};

    /// The real checkpoint stores `multi_modal_projector` quantized with a float
    /// `.bias` alongside the affine `.scales`/`.biases`. The loader must resolve
    /// all four and report a quantized projector (issue #527 follow-up).
    #[test]
    fn projector_loads_quantized_with_bias() {
        let mut w = WeightMap::new();
        // group_size 64 / bits 4: in=64 -> packed_in 8, num_groups 1.
        w.insert(
            "multi_modal_projector.weight".to_string(),
            mlxcel_core::from_slice_u32(&[0u32; 8 * 8], &[8, 8]),
        );
        w.insert(
            "multi_modal_projector.scales".to_string(),
            mlxcel_core::from_slice_f32(&[0.0; 8], &[8, 1]),
        );
        w.insert(
            "multi_modal_projector.biases".to_string(),
            mlxcel_core::from_slice_f32(&[0.0; 8], &[8, 1]),
        );
        w.insert(
            "multi_modal_projector.bias".to_string(),
            mlxcel_core::from_slice_f32(&[0.0; 8], &[8]),
        );

        let projector =
            MllamaVLModel::load_projector(&w, 64, 4).expect("quantized projector must load");
        assert!(
            projector.is_quantized(),
            "the 4-bit projector must load quantized, not as a raw packed linear"
        );
    }

    // --- Real-tile row selection (issue #527 perf follow-up). ---

    /// `[1, media, max_tiles, patches, dim]` filled with 0..N so every row is
    /// identifiable by value.
    fn arange_output(media: i32, max_tiles: i32, patches: i32, dim: i32) -> UniquePtr<MlxArray> {
        let n = (media * max_tiles * patches * dim) as usize;
        let vals: Vec<f32> = (0..n).map(|i| i as f32).collect();
        mlxcel_core::from_slice_f32(&vals, &[1, media, max_tiles, patches, dim])
    }

    fn max_abs_diff(a: &MlxArray, b: &MlxArray) -> f32 {
        let diff = mlxcel_core::subtract(a, b);
        let m = mlxcel_core::max_all(&mlxcel_core::abs(&diff));
        mlxcel_core::eval(&m);
        mlxcel_core::item_f32(&m)
    }

    /// Ragged multi-image selection keeps exactly the real-tile rows of every
    /// image, media-major, in the reference's flatten order.
    #[test]
    fn select_keeps_ragged_real_tile_rows_media_major() {
        // 2 media, 2 tiles, 2 patches, dim 3; media 0 has 1 real tile, media 1
        // has 2. Row layout (patch rows of dim 3):
        //   media 0: tile 0 -> values 0..6,   tile 1 -> 6..12 (padding)
        //   media 1: tile 0 -> values 12..18, tile 1 -> 18..24
        let output = arange_output(2, 2, 2, 3);
        let selected =
            select_real_tile_states(&output, &[1, 2]).expect("sub-max ragged counts must select");
        assert_eq!(mlxcel_core::array_shape(&selected), vec![1, 6, 3]);

        let expected: Vec<f32> = (0..6)
            .map(|i| i as f32)
            .chain((12..24).map(|i| i as f32))
            .collect();
        let expected = mlxcel_core::from_slice_f32(&expected, &[1, 6, 3]);
        assert_eq!(
            max_abs_diff(&selected, &expected),
            0.0,
            "selection must keep media 0 tile 0 and media 1 tiles 0..2, in order"
        );
    }

    /// Full tile counts are a no-op: the caller must keep the legacy all-tiles
    /// path so the states stay byte-identical to the pre-optimization output.
    #[test]
    fn select_declines_full_tile_counts() {
        let output = arange_output(2, 2, 2, 3);
        assert!(select_real_tile_states(&output, &[2, 2]).is_none());
    }

    /// Defensive fallbacks: count/media mismatch, zero counts, and
    /// out-of-range counts all keep the legacy path instead of guessing.
    #[test]
    fn select_declines_invalid_tile_counts() {
        let output = arange_output(2, 2, 2, 3);
        assert!(select_real_tile_states(&output, &[1]).is_none());
        assert!(select_real_tile_states(&output, &[1, 2, 1]).is_none());
        assert!(select_real_tile_states(&output, &[0, 2]).is_none());
        assert!(select_real_tile_states(&output, &[1, 3]).is_none());
        assert!(select_real_tile_states(&output, &[]).is_none());
    }
}
