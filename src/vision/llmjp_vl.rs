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

//! LLM-jp-VL (`llmjpvl`) Vision-Language Model.
//!
//! Port of the `modeling_llmjpvl.py` / `processing_llmjpvl.py` pair shipped
//! inside `llm-jp/llm-jp-4-vl-9B-beta` and
//! `llm-jp/Jagle-VL-2.2B-Jagle-FineVision`. Both checkpoints declare
//! `model_type: "llmjpvl"` and share one architecture; they differ only in the
//! text backbone named by `llm_config.model_type`.
//!
//! Composition:
//! - `vision_backbone.vision_model`: a SigLIP2-so400m tower (512px, patch 16,
//!   1024 patches per tile, 27 layers of width 1152, `gelu_pytorch_tanh`),
//!   reusing [`crate::vision::encoders::siglip::SigLipVisionModel`]. The tower
//!   has no CLS token, so unlike InternViT nothing is stripped from the
//!   `post_layernorm` output. The checkpoint's attention-pooling `head.*`
//!   tensors are unused (`last_hidden_state` is the feature source).
//! - `mlp1` connector: `pixel_shuffle(0.5)` then
//!   `[LayerNorm(1152*4), Linear(4608 -> hidden), GELU, Linear(hidden, hidden)]`.
//!   This is byte-for-byte [`crate::vision::internvl::InternVLConnector`]; the
//!   only difference at the call site is the LayerNorm epsilon, which upstream
//!   leaves at the torch `nn.LayerNorm` default of 1e-5 rather than taking
//!   `vision_config.layer_norm_eps`.
//! - `language_model`: Llama (llm-jp-4-vl-9B, 196608-token vocab) or Qwen3
//!   (Jagle-VL-2.2B, per-head q/k RMSNorm), selected by `llm_config.model_type`
//!   and wrapped in [`LlmJpTextModel`].
//!
//! Vision/text fusion is the LLaVA scatter: the connector emits
//! `num_image_token` (256) vectors per 512x512 tile in the decoder width, and
//! those replace the `<|image_pad|>` positions in the prompt embedding stream
//! via [`crate::vision::merge::merge_llava`], exactly as upstream
//! `LLMjpVLModel.generate` does with `input_embeds[selected] = vit_embeds`.
//!
//! Used by: `loading::load_llmjp_vl`, `multimodal::vlm_runtime`.

use mlxcel_core::cache::{KVCacheMode, SequenceId, SequenceStateLayout};
use mlxcel_core::generate::{DecodeBatchContext, LanguageModel};
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::vision::encoders::VisionEncoder;
use crate::vision::encoders::siglip::SigLipVisionModel;
use crate::vision::internvl::InternVLConnector;
use crate::vision::llmjp_vl_text::LlmJpTextModel;
use crate::vision::merge::{self, InputEmbeddings};
use crate::vision::processors::internvl::InternVLProcessor;

/// Torch's `nn.LayerNorm` default epsilon.
///
/// Upstream builds the `mlp1` LayerNorm as a bare
/// `nn.LayerNorm(vit_hidden_size * 4)` and never passes `eps`, so the connector
/// normalizes at 1e-5 even though `vision_config.layer_norm_eps` is 1e-6. The
/// InternVL loader passes the vision config value at the equivalent call site;
/// this family must not, which is the one numerical difference between the two
/// connector call sites.
pub const LLMJP_MLP1_LAYER_NORM_EPS: f32 = 1e-5;

/// Upstream's default `tokenizer.model_max_length` for both checkpoints.
pub const LLMJP_DEFAULT_MODEL_MAX_LENGTH: usize = 4096;

/// Upstream's default `processor_config.json` `image_seq_length`.
pub const LLMJP_DEFAULT_IMAGE_SEQ_LENGTH: usize = 256;

/// Per-image tile budget, a port of the `max_num` computation in
/// `LLMjpVLProcessor.__call__`.
///
/// Upstream sizes the dynamic-tiling budget from what is left of the
/// tokenizer's context after the prompt text, because each tile costs
/// `image_seq_length` tokens and a framed block costs two more:
///
/// ```text
/// image_budget = model_max_length - text_tokens
/// max_num      = (image_budget // num_images - 2) // image_seq_length - 1
/// max_num      = max(1, min(max_dynamic_patch, max_num))
/// ```
///
/// `text_tokens` is the token count of the prompt with every `<image>`
/// placeholder removed. For a prompt under roughly 500 tokens with one image
/// this saturates at `max_dynamic_patch` (12).
///
/// Signed arithmetic with floor division reproduces Python's `//` for the
/// negative intermediates a very long prompt produces; every negative result
/// clamps to 1, which is also what upstream returns.
pub fn image_tile_budget(
    model_max_length: usize,
    text_tokens: usize,
    num_images: usize,
    image_seq_length: usize,
    max_dynamic_patch: usize,
) -> usize {
    if num_images == 0 || image_seq_length == 0 {
        return max_dynamic_patch.max(1);
    }
    let image_budget = model_max_length as i64 - text_tokens as i64;
    let per_image = image_budget.div_euclid(num_images as i64) - 2;
    let max_num = per_image.div_euclid(image_seq_length as i64) - 1;
    max_num.clamp(1, max_dynamic_patch.max(1) as i64) as usize
}

/// Top-level LLM-jp-VL (`llmjpvl`) VLM runtime.
pub struct LlmJpVlModel {
    pub text_model: LlmJpTextModel,
    pub vision_model: SigLipVisionModel,
    pub connector: InternVLConnector,
    pub processor: InternVLProcessor,
    /// Token id of the `<|image_pad|>` placeholder (14 for llm-jp-4-vl-9B,
    /// 151655 for Jagle-VL). Image features are merged at these positions.
    pub image_context_token_id: i32,
    /// Token id of `<|image_start|>` (15 / 151669).
    pub img_start_token_id: i32,
    /// Token id of `<|image_end|>` (16 / 151670).
    pub img_end_token_id: i32,
    /// Image feature tokens emitted per tile: `(512/16)^2 * 0.5^2 = 256`.
    pub num_image_token: usize,
    /// EOS/stop token ids for the CLI and server stop paths, resolved at load
    /// time from `generation_config.json` plus the checkpoint's own
    /// `<|return|>` / `<|end|>` / `<|im_end|>` ids.
    pub eos_token_ids: Vec<i32>,
    /// `processor_config.json` `image_seq_length` (256), the per-tile cost the
    /// tile budget divides by.
    pub image_seq_length: usize,
    /// `tokenizer_config.json` `model_max_length` (4096), the tile budget's
    /// context ceiling.
    pub model_max_length: usize,
    /// `config.json` `max_dynamic_patch` (12), the tile budget's cap.
    pub max_dynamic_patch: usize,
}

impl LlmJpVlModel {
    /// Tile budget for one request, given the token count of the prompt with
    /// image placeholders removed.
    pub fn tile_budget(&self, text_tokens: usize, num_images: usize) -> usize {
        image_tile_budget(
            self.model_max_length,
            text_tokens,
            num_images,
            self.image_seq_length,
            self.max_dynamic_patch,
        )
    }

    /// Merged input embeddings for a request carrying pixel values. Mirrors
    /// `LLMjpVLModel.extract_feature` followed by the `input_embeds[selected]`
    /// scatter in `LLMjpVLModel.generate`.
    ///
    /// `pixel_values`: channels-first `[num_tiles, C, H, W]` (the processor's
    /// native layout), transposed once here for the channels-last conv patch
    /// embedding.
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
    ) -> InputEmbeddings {
        let inputs_embeds = self.text_model.get_embed_tokens(input_ids);

        // Feed the tower in the text embedding dtype (upstream keeps the whole
        // vision stack in the checkpoint dtype).
        let embed_dtype = mlxcel_core::array_dtype(&inputs_embeds);
        let pv = mlxcel_core::astype(pixel_values, embed_dtype);

        // [num_tiles, C, H, W] -> [num_tiles, H, W, C].
        let pv = mlxcel_core::transpose_axes(&pv, &[0, 2, 3, 1]);

        // SigLIP has no CLS token, so `last_hidden_state` is used whole:
        // [num_tiles, 1024, 1152]. `select_layer != -1` is rejected at load.
        let vision_output = self.vision_model.forward(&pv);

        // pixel_shuffle + mlp1 -> [num_tiles, 256, hidden_lm].
        let image_features = self.connector.forward(&vision_output.hidden_states);

        merge::merge_llava(
            self.image_context_token_id,
            &image_features,
            &inputs_embeds,
            input_ids,
        )
    }
}

// LanguageModel: text-only forward paths delegate to the selected backbone.
// EOS ids come from the checkpoint rather than the backbone default, which
// would be Llama-3's or Qwen3's and is wrong for both of these vocabularies.
impl LanguageModel for LlmJpVlModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        LanguageModel::forward(&self.text_model, input_ids, caches, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        LanguageModel::forward_with_embeddings(
            &self.text_model,
            input_ids,
            input_embeddings,
            caches,
            mask,
        )
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        _seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        LanguageModel::forward(&self.text_model, input_ids, caches, mask)
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        _seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        LanguageModel::forward_with_embeddings(
            &self.text_model,
            input_ids,
            input_embeddings,
            caches,
            mask,
        )
    }

    fn forward_batched(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        LanguageModel::forward_batched(&self.text_model, input_ids, batch_caches, mask)
    }

    fn forward_batched_with_context(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
        context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        LanguageModel::forward_batched_with_context(
            &self.text_model,
            input_ids,
            batch_caches,
            mask,
            context,
        )
    }

    fn forward_batched_with_context_and_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
        context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        LanguageModel::forward_batched_with_context_and_ids(
            &self.text_model,
            input_ids,
            seq_ids,
            batch_caches,
            mask,
            context,
        )
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        LanguageModel::embed_tokens(&self.text_model, input_ids)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        LanguageModel::make_caches(&self.text_model)
    }

    fn set_kv_cache_layer_modes(&self, modes: Vec<KVCacheMode>) {
        LanguageModel::set_kv_cache_layer_modes(&self.text_model, modes)
    }

    fn kv_cache_layer_modes(&self) -> Option<Vec<KVCacheMode>> {
        LanguageModel::kv_cache_layer_modes(&self.text_model)
    }

    fn num_layers(&self) -> usize {
        LanguageModel::num_layers(&self.text_model)
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }

    /// The three framing ids are input-alignment scaffolding: emitting one as
    /// generated text corrupts the stream and, for `<|image_pad|>`, would also
    /// desynchronize a follow-up turn's feature scatter.
    fn output_suppressed_token_ids(&self) -> Vec<i32> {
        let mut ids = vec![
            self.image_context_token_id,
            self.img_start_token_id,
            self.img_end_token_id,
        ];
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Sequence state follows the text backbone rather than the trait default,
    /// so the server scheduler allocates one KV cache entry per decoder layer.
    /// Leaving this to the default would still work today, but only because
    /// `num_layers()` and `supports_batching()` both already delegate; pinning
    /// it makes the VLM's server-side cache shape explicit.
    fn sequence_state_layout(&self) -> SequenceStateLayout {
        LanguageModel::sequence_state_layout(&self.text_model)
    }

    fn supports_batching(&self) -> bool {
        LanguageModel::supports_batching(&self.text_model)
    }

    fn supports_batched_prefill(&self) -> bool {
        LanguageModel::supports_batched_prefill(&self.text_model)
    }

    fn supports_maskless_padded_prefill(&self) -> bool {
        LanguageModel::supports_maskless_padded_prefill(&self.text_model)
    }

    fn supports_paged_decode_backend(&self) -> bool {
        LanguageModel::supports_paged_decode_backend(&self.text_model)
    }
}

#[cfg(test)]
#[path = "llmjp_vl_tests.rs"]
mod tests;
