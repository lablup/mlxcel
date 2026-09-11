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

//! GOT-OCR 2.0 (`model_type: "GOT"`) Vision-Language Model.
//!
//! A 0.58B OCR model: `stepfun-ai/GOT-OCR2_0` plus the
//! `mlx-community/GOT-OCR2_0-{bf16,8bit,4bit}` conversions. The stack is short
//! and every piece already exists in the tree:
//!
//! - A SAM-style ViT-B tower on a fixed 1024x1024 input, reused verbatim as
//!   [`SamEncoder`] (the DeepSeek-OCR tower; `got_vision_b.py` and mlx-vlm's
//!   `sam.py` are the same network down to the `1e-6` neck LayerNorm and the
//!   bias-free stride-2 `net_2` / `net_3` compressor). It emits a 16x16 grid of
//!   width 1024, so exactly 256 feature rows per page.
//! - A single `Linear(1024, 1024)` projector (`mm_projector_vary` in the
//!   original, `multi_modal_projector` in the conversions).
//! - A Qwen2-0.5B decoder with q/k/v biases and tied embeddings, served by the
//!   Llama-family backbone the way [`crate::models::qwen2`] already does.
//!
//! Fusion is the LLaVA scatter: upstream splices the 256 rows in after the
//! `<img>` position, which is positionally identical to replacing the 256
//! `<imgpad>` rows between `<img>` and `</img>`, and that is what
//! [`merge_llava`] does. Because `<img>` / `<imgpad>` / `</img>` are real
//! vocabulary entries (151857 / 151859 / 151858) the block travels as ordinary
//! prompt text; see [`crate::multimodal::got_ocr_prompt`].
//!
//! The stop set is `[151643, 151645]`. `config.json` declares only 151643
//! (`<|endoftext|>`), which an OCR answer never emits: upstream stops on the
//! turn separator `<|im_end|>` (151645) through a `KeywordsStoppingCriteria`
//! rather than through `eos_token_id`. Without 151645 in the set a run does not
//! terminate, it fills `max_tokens`.
//!
//! Reference: `modeling_GOT.py` in <https://huggingface.co/stepfun-ai/GOT-OCR2_0>.

use mlxcel_core::cache::{KVCacheMode, SequenceId, SequenceStateLayout};
use mlxcel_core::generate::{DecodeBatchContext, LanguageModel};
use mlxcel_core::layers::{KVCache, UnifiedLinear};
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::Llama3Model;
use crate::vision::encoders::deepseekocr_sam::SamEncoder;
use crate::vision::merge::{InputEmbeddings, merge_llava};
use crate::vision::processors::got_ocr::GotOcrImageProcessor;

/// Top-level GOT-OCR 2.0 runtime.
pub struct GotOcrVlModel {
    /// Qwen2-0.5B decoder (the Llama-family backbone; `qwen2` is the same graph).
    pub text_model: Llama3Model,
    /// SAM ViT-B tower, shared with DeepSeek-OCR.
    pub vision_tower: SamEncoder,
    /// `Linear(1024, 1024)` with bias.
    pub projector: UnifiedLinear,
    pub processor: GotOcrImageProcessor,
    /// Feature rows one page produces (256).
    pub image_token_len: usize,
    /// `<img>` (151857).
    pub im_start_token_id: i32,
    /// `</img>` (151858).
    pub im_end_token_id: i32,
    /// `<imgpad>` (151859); the scatter target.
    pub im_patch_token_id: i32,
    /// `[151643, 151645]`.
    pub eos_token_ids: Vec<i32>,
}

impl GotOcrVlModel {
    /// Tower + projector for one preprocessed batch.
    ///
    /// `pixel_values` is channels-last `(n, 1024, 1024, 3)`, which is what the
    /// tower takes; the reference's `(n, 3, 1024, 1024)` is the same data with
    /// the transpose deferred into its first conv.
    ///
    /// The tower returns `(n, 16, 16, 1024)`. Upstream does
    /// `cnn_feature.flatten(2).permute(0, 2, 1)` on its NCHW `(n, 1024, 16, 16)`,
    /// which orders rows by `h * 16 + w`; a channels-last reshape to
    /// `(n, 256, 1024)` produces that same order without a transpose.
    pub fn image_features(&self, pixel_values: &MlxArray) -> UniquePtr<MlxArray> {
        let grid = self.vision_tower.forward(pixel_values);
        let shape = mlxcel_core::array_shape(&grid);
        let (n, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
        let flat = mlxcel_core::reshape(&grid, &[n, h * w, c]);
        self.projector.forward(&flat)
    }

    /// Embed the prompt and scatter the image features onto its `<imgpad>` rows.
    ///
    /// `merge_llava` consumes one feature row per placeholder in prompt order,
    /// so the caller must have emitted exactly `image_token_len` of them per
    /// image; the runtime arm checks that before calling this.
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
    ) -> InputEmbeddings {
        let inputs_embeds = self.text_model.get_embed_tokens(input_ids);
        let features = self.image_features(pixel_values);
        let shape = mlxcel_core::array_shape(&features);
        // Flatten the per-image batch into one prompt-ordered run. With the one
        // image the decoder was trained for this is a no-op reshape; keeping it
        // general means a future multi-image path scatters in the right order
        // rather than dropping images silently.
        let flat = mlxcel_core::reshape(&features, &[1, shape[0] * shape[1], shape[2]]);
        merge_llava(self.im_patch_token_id, &flat, &inputs_embeds, input_ids)
    }
}

impl LanguageModel for GotOcrVlModel {
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
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        LanguageModel::forward_with_sequence_id(&self.text_model, input_ids, seq_id, caches, mask)
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        LanguageModel::forward_with_embeddings_and_sequence_id(
            &self.text_model,
            input_ids,
            input_embeddings,
            seq_id,
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

    /// `[151643, 151645]`. See the module doc: without `<|im_end|>` a run never
    /// terminates on its own.
    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }

    /// The three framing ids are input scaffolding. Sampling `<imgpad>` would
    /// also desynchronize a follow-up turn's scatter, since the count of
    /// placeholders is what pairs the prompt with the tower's rows.
    fn output_suppressed_token_ids(&self) -> Vec<i32> {
        let mut ids = vec![
            self.im_start_token_id,
            self.im_end_token_id,
            self.im_patch_token_id,
        ];
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Delegated to the decoder, not left to the trait default.
    ///
    /// The default reports no per-layer sequence state, which hands the server
    /// scheduler an empty cache vector: the model then runs zero decoder layers
    /// and returns fluent-looking nonsense with no error anywhere. The CLI does
    /// not catch it because it builds its own caches from `make_caches`.
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
#[path = "got_ocr_tests.rs"]
mod tests;
