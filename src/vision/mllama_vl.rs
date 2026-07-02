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
    /// hidden]`. Mirrors the vision branch of `Model.__call__`.
    pub fn compute_cross_attention_states(
        &self,
        pixel_values: &MlxArray,
        aspect_ratio_ids: &MlxArray,
        aspect_ratio_mask: &MlxArray,
    ) -> UniquePtr<MlxArray> {
        let batch = mlxcel_core::array_shape(pixel_values)[0];
        let vision_output =
            self.vision_tower
                .forward(pixel_values, aspect_ratio_ids, aspect_ratio_mask);
        let projected = self.multi_modal_projector.forward(&vision_output);
        let hidden = self.config.text_config.hidden_size as i32;
        mlxcel_core::reshape(&projected, &[batch, -1, hidden])
    }

    /// Compute and stash the cross-attention states from preprocessed image
    /// inputs so subsequent [`LanguageModel::forward`] calls attend to them.
    pub fn prepare_cross_attention_states(&self, inputs: &MllamaImageInputs) {
        let states = self.compute_cross_attention_states(
            &inputs.pixel_values,
            &inputs.aspect_ratio_ids,
            &inputs.aspect_ratio_mask,
        );
        self.set_cross_attention_states(states);
    }

    /// Stash externally-computed cross-attention states.
    pub fn set_cross_attention_states(&self, states: UniquePtr<MlxArray>) {
        *self.cross_attention_states.borrow_mut() = Some(states);
    }

    /// Drop any stashed cross-attention states (revert to text-only decoding).
    pub fn clear_cross_attention_states(&self) {
        *self.cross_attention_states.borrow_mut() = None;
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
    use super::MllamaVLModel;
    use mlxcel_core::weights::WeightMap;

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
}
