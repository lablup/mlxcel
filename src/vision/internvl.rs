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

//! InternVL (`internvl_chat`) Vision-Language Model.
//!
//! Faithful port of
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/internvl_chat/internvl_chat.py.
//!
//! Composition (for `internvl3-1b`):
//! - `vision_model` — [`InternVitVisionModel`] (non-quantized bf16, converted
//!   to f16 on Apple Silicon at load time).
//! - `mlp1` connector — `pixel_shuffle(downsample_ratio=0.5)` then
//!   `[LayerNorm(4096), Linear(4096->896), GELU, Linear(896->896)]`. The two
//!   Linears are 4-bit quantized in the released checkpoint; the LayerNorm is
//!   bf16 (-> f16 on Apple Silicon).
//! - `language_model` — Qwen2 backbone (mlxcel's [`crate::models::Qwen2Model`],
//!   a re-export of `Llama3Model`). InternVL3-1b uses a Qwen2 LM
//!   (`text_config.model_type == "qwen2"`), NOT InternLM, so we reuse the
//!   existing Qwen2/Llama backbone rather than reimplementing a transformer.
//!
//! Vision/text fusion: the connector produces `num_image_token` (256) feature
//! vectors per 448x448 tile in the Qwen2 hidden size (896). Those replace the
//! `<IMG_CONTEXT>` placeholder positions in the prompt embedding stream via
//! [`crate::vision::merge::merge_llava`] — exactly the upstream
//! `_merge_input_ids_with_image_features` semantics.
//!
//! Standard 1D RoPE is used (NOT Qwen2-VL's MRoPE), and the InternViT tower
//! uses no qk-norm, so this runtime is structurally a LLaVA-style VLM with a
//! tiled image preprocessor and a pixel-shuffle connector.
//!
//! Used by: `loading::load_internvl_vlm`, `multimodal::vlm_runtime`.

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::{DecodeBatchContext, LanguageModel};
use mlxcel_core::layers::{KVCache, LayerNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::Qwen2Model;
use crate::vision::encoders::VisionEncoder;
use crate::vision::encoders::internvl::InternVitVisionModel;
use crate::vision::merge::{self, InputEmbeddings};
use crate::vision::processors::internvl::InternVLProcessor;

/// The `mlp1` vision-to-language connector.
///
/// `pixel_shuffle(downsample_ratio)` reshapes `[B, N, C]` (N patches, channel
/// dim C) into `[B, N * r^2, C / r^2]` (here `[B, 256, 4096]`), then a small
/// MLP projects into the language hidden size:
/// `LayerNorm -> Linear -> GELU -> Linear`.
pub struct InternVLConnector {
    layer_norm: LayerNorm,
    linear_1: UnifiedLinear,
    linear_2: UnifiedLinear,
    downsample_ratio: f32,
}

impl InternVLConnector {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        layer_norm_eps: f32,
        downsample_ratio: f32,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        // mlp1 is a Sequential: index 0 = LayerNorm, 1 = Linear, 2 = GELU,
        // 3 = Linear (GELU has no parameters).
        let weight = weights
            .get(&format!("{prefix}.0.weight"))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| format!("Weight not found: {prefix}.0.weight"))?;
        let bias = weights
            .get(&format!("{prefix}.0.bias"))
            .map(|b| mlxcel_core::copy(b));
        let layer_norm = LayerNorm::new(weight, bias, layer_norm_eps);

        let linear_1 =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.1"), group_size, bits)?;
        let linear_2 =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.3"), group_size, bits)?;

        Ok(Self {
            layer_norm,
            linear_1,
            linear_2,
            downsample_ratio,
        })
    }

    /// `vision_features`: `[B, N, C]` (CLS already stripped) ->
    /// `[B, N*r^2, hidden]` projected image tokens.
    pub fn forward(&self, vision_features: &MlxArray) -> UniquePtr<MlxArray> {
        let x = pixel_shuffle(vision_features, self.downsample_ratio);
        let x = self.layer_norm.forward(&x);
        let x = self.linear_1.forward(&x);
        let x = mlxcel_core::gelu(&x);
        self.linear_2.forward(&x)
    }
}

/// Pixel shuffle (spatial -> channel) port of
/// https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/base.py#L311-L333.
///
/// Input `x`: `[B, N, C]` where `N = p*p` is a square patch grid.
/// Output: `[B, N*r^2, C/r^2]` (for the default `r = 0.5`, `[B, 4N, C*4]`).
fn pixel_shuffle(x: &MlxArray, shuffle_ratio: f32) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(x);
    let b = shape[0];
    let n = shape[1];
    let c = shape[2];
    let p = (n as f64).sqrt().round() as i32; // grid side length
    let r = shuffle_ratio; // 0.5

    // x = x.reshape(B, p, p, C)
    let x = mlxcel_core::reshape(x, &[b, p, p, c]);
    // x = x.reshape(B, p, p*r, C/r).transpose(0, 2, 1, 3)
    let new_w = (p as f32 * r) as i32;
    let new_c = (c as f32 / r) as i32;
    let x = mlxcel_core::reshape(&x, &[b, p, new_w, new_c]);
    let x = mlxcel_core::transpose_axes(&x, &[0, 2, 1, 3]);
    // x = x.reshape(B, p*r, p*r, C/r^2).transpose(0, 2, 1, 3)
    let new_h = (p as f32 * r) as i32;
    let new_c2 = (c as f32 / (r * r)) as i32;
    let x = mlxcel_core::reshape(&x, &[b, new_h, new_w, new_c2]);
    let x = mlxcel_core::transpose_axes(&x, &[0, 2, 1, 3]);
    // x = x.reshape(B, -1, C/r^2)
    mlxcel_core::reshape(&x, &[b, -1, new_c2])
}

/// Top-level InternVL (`internvl_chat`) VLM runtime.
pub struct InternVLChatVLM {
    pub text_model: Qwen2Model,
    pub vision_model: InternVitVisionModel,
    pub connector: InternVLConnector,
    pub processor: InternVLProcessor,
    /// Token id of the `<IMG_CONTEXT>` placeholder (resolved from the
    /// tokenizer; 151667 for `internvl3-1b`). Image features are merged at
    /// these positions.
    pub image_context_token_id: i32,
    /// Token id of the `<img>` opening frame (151665 for `internvl3-1b`).
    pub img_start_token_id: i32,
    /// Token id of the `</img>` closing frame (151666 for `internvl3-1b`).
    pub img_end_token_id: i32,
    /// Number of image feature tokens emitted per tile (256 for the default
    /// 448px tile with `downsample_ratio = 0.5`).
    pub num_image_token: usize,
    /// EOS/stop token ids consumed by the server stop path. Resolved at load
    /// time from the config (`<|im_end|>` / `<|endoftext|>` for Qwen2).
    pub eos_token_ids: Vec<i32>,
}

impl InternVLChatVLM {
    /// Compute merged input embeddings for a request that carries pixel
    /// values. Mirrors `Model.get_input_embeddings` in upstream
    /// `internvl_chat.py`.
    ///
    /// `pixel_values`: channels-first `[num_tiles, C, H, W]` (the processor's
    /// native layout). The InternViT tower is channels-last, so we transpose
    /// once here, matching the reference's `pixel_values.transpose(0, 2, 3, 1)`.
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
    ) -> InputEmbeddings {
        // Text embeddings (Qwen2 token embedding table).
        let inputs_embeds = self.text_model.get_embed_tokens(input_ids);

        // Match the dtype of the text embeddings when feeding pixels into the
        // vision tower (mirrors upstream `pixel_values.astype(dtype)`).
        let embed_dtype = mlxcel_core::array_dtype(&inputs_embeds);
        let pv = mlxcel_core::astype(pixel_values, embed_dtype);

        // [num_tiles, C, H, W] -> [num_tiles, H, W, C] for the conv patch embed.
        let pv = mlxcel_core::transpose_axes(&pv, &[0, 2, 3, 1]);

        // Vision tower -> [num_tiles, 1 + num_patches, hidden_vit].
        let vision_output = self.vision_model.forward(&pv);

        // Strip the CLS token: [:, 1:, :] (select_layer = -1).
        let hidden = &vision_output.hidden_states;
        let shape = mlxcel_core::array_shape(hidden);
        let stripped = mlxcel_core::slice(hidden, &[0, 1, 0], &[shape[0], shape[1], shape[2]]);

        // pixel_shuffle + mlp1 -> [num_tiles, num_image_token, hidden_lm].
        let image_features = self.connector.forward(&stripped);

        // Replace <IMG_CONTEXT> positions with the projected image features.
        merge::merge_llava(
            self.image_context_token_id,
            &image_features,
            &inputs_embeds,
            input_ids,
        )
    }
}

// LanguageModel — text-only forward paths delegate straight to the underlying
// Qwen2 backbone, including the `forward_with_embeddings` path used by the VLM
// runtime to inject pre-merged inputs. EOS ids are overridden so the server
// stop path uses InternVL's Qwen2 stop tokens rather than the Llama defaults.
impl LanguageModel for InternVLChatVLM {
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
mod tests {
    use super::pixel_shuffle;

    #[test]
    fn pixel_shuffle_collapses_spatial_into_channels() {
        // InternVL3-1b: 1024 patches (32x32 grid) of dim 1024, r=0.5.
        // Expected: 256 tokens of dim 4096 (= 1024 / 0.5^2).
        let n = 1024i32;
        let c = 1024i32;
        let data: Vec<f32> = (0..(n * c)).map(|i| i as f32).collect();
        let x = mlxcel_core::from_slice_f32(&data, &[1, n, c]);

        let out = pixel_shuffle(&x, 0.5);
        mlxcel_core::eval(&out);
        let shape = mlxcel_core::array_shape(&out);
        assert_eq!(shape, vec![1, 256, 4096], "pixel_shuffle output shape");

        // Element count is preserved (pure reshape/transpose, no data loss).
        let total: i32 = shape.iter().product();
        assert_eq!(total, n * c);
    }

    #[test]
    fn pixel_shuffle_preserves_first_row_values() {
        // The first 4096 output values must be a permutation of input values,
        // and the very first element maps from input index 0 (top-left patch,
        // channel 0) since both reshapes keep [0,0,..] at the origin.
        let n = 64i32; // 8x8 grid
        let c = 16i32;
        let data: Vec<f32> = (0..(n * c)).map(|i| i as f32).collect();
        let x = mlxcel_core::from_slice_f32(&data, &[1, n, c]);

        let out = pixel_shuffle(&x, 0.5);
        mlxcel_core::eval(&out);
        let shape = mlxcel_core::array_shape(&out);
        // 8x8 grid, r=0.5 -> 16 tokens of dim 64.
        assert_eq!(shape, vec![1, 16, 64]);

        let first = mlxcel_core::slice(&out, &[0, 0, 0], &[1, 1, 1]);
        mlxcel_core::eval(&first);
        assert_eq!(mlxcel_core::item_f32(&first), 0.0);
    }
}
