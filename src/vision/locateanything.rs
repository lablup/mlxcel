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

//! LocateAnything (`locateanything`) Vision-Language Model.
//!
//! Faithful port of the autoregressive path of upstream
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/locateanything/locateanything.py.
//!
//! Composition (verified against `mlx-community/LocateAnything-3B-4bit`):
//! - `vision_tower` — MoonViT-SO-400M, the same tower Kimi-VL uses, so
//!   [`crate::vision::encoders::kimi_vl::KimiVLVisionModel`] is reused. The two
//!   genuine deltas (LayerNorm eps `1e-5`, tanh-approximate GELU in the block
//!   MLP) are carried in the vision config. Plain bf16 in the released
//!   checkpoint even though the text stack is 4-bit.
//! - `multi_modal_projector` — [`LocateAnythingConnector`]:
//!   `LayerNorm(vision_hidden * merge_h * merge_w) -> Linear -> GELU -> Linear`,
//!   projecting each merged patch into the Qwen2 hidden size. Note the
//!   normalization runs over the **flattened merged patch** (4608 for the
//!   released 2x2 checkpoint), not over the 1152-wide vision hidden dim, which
//!   is where it differs from the Kimi-VL projector.
//! - `language_model` — Qwen2 backbone (mlxcel's [`crate::models::Qwen2Model`],
//!   a re-export of `Llama3Model`): 36 layers, hidden 2048, 16 heads, 2 kv
//!   heads, QKV bias, `tie_word_embeddings = true`.
//!
//! Vision/text fusion: the connector emits one feature row per merged patch,
//! and those rows replace the `<IMG_CONTEXT>` (`image_token_index`, 151665)
//! positions in the prompt embedding stream via
//! [`crate::vision::merge::merge_llava`] — the same scatter the upstream
//! `Model.get_input_embeddings` cumsum/`mx.where` pair performs.
//!
//! Grounding output: LocateAnything emits its boxes as ordinary text tokens
//! (`<ref>`/`</ref>`, `<box>`/`</box>`, and the 1001 coordinate tokens
//! `<0>`..`<1000>` at ids 151677..152677). Plain autoregressive decode is
//! therefore sufficient and no special detokenization is needed. The
//! checkpoint's parallel box-decoding head (`pbd`, `n_future_tokens = 6`) and
//! coordinate-token-to-box post-processing are deliberately out of scope here;
//! they are a follow-up.
//!
//! Used by: `loading::load_locateanything_vlm`, `multimodal::vlm_runtime`.

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::{DecodeBatchContext, LanguageModel};
use mlxcel_core::layers::{KVCache, LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::Qwen2Model;
use crate::vision::encoders::kimi_vl::{KimiMediaGrid, KimiVLVisionModel};
use crate::vision::merge::{self, InputEmbeddings};
use crate::vision::processors::locateanything::LocateAnythingProcessor;

/// Upstream fixes the projector LayerNorm at MLX's `nn.LayerNorm` default eps.
const CONNECTOR_LAYER_NORM_EPS: f32 = 1e-5;

/// The `multi_modal_projector` vision-to-language connector.
///
/// Input is the MoonViT patch-merger output `[total_merged, kh*kw,
/// vision_hidden]`. It is flattened to `[total_merged, vision_hidden*kh*kw]`,
/// normalized over that flattened width, and projected through
/// `Linear -> GELU -> Linear` into the text hidden size.
pub struct LocateAnythingConnector {
    layer_norm: LayerNorm,
    linear_1: UnifiedLinear,
    linear_2: UnifiedLinear,
    /// `vision_hidden * merge_h * merge_w` — the flattened merged-patch width.
    input_dim: i32,
}

impl LocateAnythingConnector {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        input_dim: i32,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let weight = weights
            .get(&format!("{prefix}.layer_norm.weight"))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {prefix}.layer_norm.weight"))?;
        let bias = weights
            .get(&format!("{prefix}.layer_norm.bias"))
            .map(|b| mlxcel_core::copy(b));
        let layer_norm = LayerNorm::new(weight, bias, CONNECTOR_LAYER_NORM_EPS);

        let linear_1 =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.linear_1"), group_size, bits)?;
        let linear_2 =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.linear_2"), group_size, bits)?;

        Ok(Self {
            layer_norm,
            linear_1,
            linear_2,
            input_dim,
        })
    }

    /// `image_features`: `[total_merged, kh*kw, vision_hidden]`. Returns
    /// `[total_merged, text_hidden]`.
    pub fn forward(&self, image_features: &MlxArray) -> UniquePtr<MlxArray> {
        let h = mlxcel_core::reshape(image_features, &[-1, self.input_dim]);
        let h = self.layer_norm.forward(&h);
        let h = self.linear_1.forward(&h);
        let h = mlxcel_core::gelu(&h);
        self.linear_2.forward(&h)
    }
}

/// Top-level LocateAnything (`locateanything`) VLM runtime.
pub struct LocateAnythingVLM {
    pub text_model: Qwen2Model,
    pub vision_model: KimiVLVisionModel,
    pub connector: LocateAnythingConnector,
    /// Native-resolution image processor (resize-up + patchify + per-image grid).
    pub processor: LocateAnythingProcessor,
    /// Token id whose positions receive image features (`image_token_index`,
    /// `<IMG_CONTEXT>` = 151665 in the released checkpoint).
    pub image_token_id: i32,
    /// Token id of the `<img>` opening frame (151666).
    pub img_start_token_id: i32,
    /// Token id of the `</img>` closing frame (151667).
    pub img_end_token_id: i32,
    /// `merge_kernel_size` from the vision config; the runtime divides each
    /// image's patch count by `merge_h * merge_w` to size its placeholder run.
    pub merge_kernel_size: [usize; 2],
    /// EOS/stop token ids resolved from the config at load time. The
    /// `Llama3Model` trait default returns Llama-3 ids, which are wrong here.
    pub eos_token_ids: Vec<i32>,
}

impl LocateAnythingVLM {
    /// Number of `<IMG_CONTEXT>` tokens (and connector feature rows) one
    /// `(grid_h, grid_w)` patch grid contributes.
    #[inline]
    pub fn merged_token_count(&self, grid: (i32, i32)) -> usize {
        crate::multimodal::locateanything_prompt::merged_token_count(grid, self.merge_kernel_size)
    }

    /// Compute merged input embeddings for a request that carries pixel values.
    /// Mirrors `Model.get_input_embeddings` in upstream `locateanything.py`.
    ///
    /// `pixel_values`: channels-first `[total_patches, C, p, p]` (the
    /// processor's native layout), packed in image order. MoonViT is
    /// channels-last, so we transpose once here, matching the reference
    /// `pixel_values.transpose(0, 2, 3, 1)`. `grids` carries one
    /// `(grid_h, grid_w)` per image, in the same order the patches are
    /// concatenated.
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grids: &[(i32, i32)],
    ) -> InputEmbeddings {
        let inputs_embeds = self.text_model.get_embed_tokens(input_ids);

        let embed_dtype = mlxcel_core::array_dtype(&inputs_embeds);
        let pv = mlxcel_core::astype(pixel_values, embed_dtype);
        let pv = mlxcel_core::transpose_axes(&pv, &[0, 2, 3, 1]);

        let media_grids: Vec<KimiMediaGrid> = grids
            .iter()
            .map(|&(h, w)| KimiMediaGrid::Image { h, w })
            .collect();

        // MoonViT -> [total_merged, kh*kw, vision_hidden].
        let vision_features = self.vision_model.forward_with_grid(&pv, &media_grids);
        // Connector -> [total_merged, text_hidden].
        let image_features = self.connector.forward(&vision_features);

        merge::merge_llava(
            self.image_token_id,
            &image_features,
            &inputs_embeds,
            input_ids,
        )
    }
}

// LanguageModel — text-only forward paths delegate straight to the Qwen2
// backbone, including the `forward_with_embeddings` path the VLM runtime uses
// to inject the merged image embeddings. EOS ids come from the LocateAnything
// config rather than the Llama-3 defaults the backbone would otherwise return.
impl LanguageModel for LocateAnythingVLM {
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
        self.text_model
            .forward_with_embeddings(input_ids, input_embeddings, caches, mask)
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        _seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model.forward(input_ids, caches, mask)
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        _seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_with_embeddings(input_ids, input_embeddings, caches, mask)
    }

    fn forward_batched(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_batched(input_ids, batch_caches, mask)
    }

    fn forward_batched_with_context(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
        context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_batched_with_context(input_ids, batch_caches, mask, context)
    }

    fn forward_batched_with_context_and_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
        context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        self.text_model.forward_batched_with_context_and_ids(
            input_ids,
            seq_ids,
            batch_caches,
            mask,
            context,
        )
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        self.text_model.embed_tokens(input_ids)
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

    fn supports_batched_prefill(&self) -> bool {
        self.text_model.supports_batched_prefill()
    }

    fn supports_maskless_padded_prefill(&self) -> bool {
        self.text_model.supports_maskless_padded_prefill()
    }

    fn supports_paged_decode_backend(&self) -> bool {
        self.text_model.supports_paged_decode_backend()
    }
}

#[cfg(test)]
#[path = "locateanything_tests.rs"]
mod tests;
