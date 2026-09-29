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

//! Mage-VL (`mage_vl`, `microsoft/Mage-VL`) Vision-Language Model.
//!
//! For still images Mage-VL is a Qwen2-VL-style patch processor (patch 16,
//! 2x2 merge, temporal patch 1, CLIP normalization) feeding the Mage-ViT tower
//! ([`crate::vision::encoders::mage_vl`]) and a stock Qwen3 decoder with plain
//! 1-D RoPE. There is no MRoPE: the decoder sees ordinary sequence positions,
//! so this runtime deliberately does not go through `QwenVlRuntime`.
//!
//! Fusion is the LLaVA scatter: the merger's `t*h*w/4` rows per image replace
//! the `<|image_pad|>` positions of the chat-templated prompt in order, as
//! upstream's `get_placeholder_mask` + `masked_scatter` does.
//!
//! The codec-native video path (hardware-codec frame groups and motion
//! vectors) is out of scope. A vision-stripped checkpoint loads with
//! `vision = None` and refuses image requests (#1367 convention).
//!
//! Used by: `loading::load_mage_vl`, `multimodal::vlm_runtime`.

use mlxcel_core::cache::{KVCacheMode, SequenceId, SequenceStateLayout};
use mlxcel_core::generate::{DecodeBatchContext, LanguageModel};
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::Qwen3Model;
use crate::vision::encoders::mage_vl::MageVlVisionEncoder;
use crate::vision::mage_vl_config::MageVlTokenIds;
use crate::vision::merge::{self, InputEmbeddings};
use crate::vision::processors::qwen2_vl::Qwen2VLProcessor;

/// Top-level Mage-VL runtime.
pub struct MageVlModel {
    pub text_model: Qwen3Model,
    /// `None` for a vision-stripped checkpoint (text-only load).
    pub vision: Option<MageVlVisionEncoder>,
    pub processor: Qwen2VLProcessor,
    pub token_ids: MageVlTokenIds,
    pub spatial_merge_size: usize,
    /// Stop ids resolved at load time (`<|im_end|>` and `<|endoftext|>`).
    pub eos_token_ids: Vec<i32>,
    /// Checkpoint directory, for the text-only refusal message.
    pub model_path: String,
}

impl MageVlModel {
    /// Merged token embeddings for a request carrying processor rows.
    ///
    /// `pixel_values`: `[sum(t*h*w), 3*16*16]` rows in merge-block order;
    /// `grid_thw`: one `(t, h, w)` patch grid per image.
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> anyhow::Result<InputEmbeddings> {
        let vision = self
            .vision
            .as_ref()
            .ok_or_else(|| crate::multimodal::qwen_vl::text_only_media_error(&self.model_path))?;
        let inputs_embeds = self.text_model.get_embed_tokens(input_ids);
        let features = vision
            .forward(pixel_values, grid_thw)
            .map_err(anyhow::Error::msg)?;
        Ok(merge::merge_llava_any(
            &[self.token_ids.image_token_id, self.token_ids.video_token_id],
            &features,
            &inputs_embeds,
            input_ids,
        ))
    }
}

// LanguageModel: every text path delegates to the Qwen3 decoder. EOS ids come
// from the checkpoint; the vision framing ids are suppressed from output.
impl LanguageModel for MageVlModel {
    fn forward_last_logits(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        LanguageModel::forward_last_logits(&self.text_model, input_ids, caches, mask, last_pos)
    }

    fn forward_last_logits_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        LanguageModel::forward_last_logits_with_sequence_id(
            &self.text_model,
            input_ids,
            seq_id,
            caches,
            mask,
            last_pos,
        )
    }

    fn forward_last_logits_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        LanguageModel::forward_last_logits_with_embeddings_and_sequence_id(
            &self.text_model,
            input_ids,
            input_embeddings,
            seq_id,
            caches,
            mask,
            last_pos,
        )
    }

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

    /// `<|vision_start|>`, `<|vision_end|>`, `<|image_pad|>` and
    /// `<|video_pad|>` are input scaffolding; emitting one would corrupt the
    /// stream and desynchronize a follow-up turn's feature scatter.
    fn output_suppressed_token_ids(&self) -> Vec<i32> {
        let ids = self.token_ids;
        let mut out = vec![
            ids.vision_start_token_id,
            ids.vision_end_token_id,
            ids.image_token_id,
            ids.video_token_id,
        ];
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Pinned to the decoder so the server scheduler allocates one KV entry
    /// per Qwen3 layer.
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
