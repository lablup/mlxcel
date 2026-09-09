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

//! The LLM-jp-VL text backbone selector.
//!
//! `llmjpvl` is one architecture over two decoders: `llm_config.model_type` is
//! `llama` for `llm-jp-4-vl-9B-beta` and `qwen3` for `Jagle-VL-2.2B`. This
//! module holds the enum that lets one VLM runtime carry either, and nothing
//! else; every method delegates and no computation happens here.
//!
//! Used by: [`crate::vision::llmjp_vl`].

use mlxcel_core::cache::{KVCacheMode, SequenceId, SequenceStateLayout};
use mlxcel_core::generate::{DecodeBatchContext, LanguageModel};
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::{Llama3Model, Qwen3Model};

/// The decoder behind an `llmjpvl` checkpoint.
///
/// `llm_config.model_type` picks the arm: `"llama"` (and the `"qwen2"` spelling
/// a converted checkpoint may carry, which mlxcel already serves with the same
/// Llama graph) selects [`Llama3Model`]; `"qwen3"` selects [`Qwen3Model`],
/// which adds the per-head q/k RMSNorm that Jagle-VL's Qwen3-1.7B backbone
/// needs and that the Llama graph does not have.
///
/// Every method delegates; this type adds no computation of its own. It exists
/// because the two backbones are distinct Rust types with no shared supertrait
/// beyond `LanguageModel`, and the VLM runtime needs `get_embed_tokens` (not
/// part of that trait) to build the text side of the merge.
pub enum LlmJpTextModel {
    Llama(Llama3Model),
    Qwen3(Qwen3Model),
}

/// Dispatch one expression over both backbone arms.
macro_rules! on_text_model {
    ($self:expr, $inner:ident => $body:expr) => {
        match $self {
            LlmJpTextModel::Llama($inner) => $body,
            LlmJpTextModel::Qwen3($inner) => $body,
        }
    };
}

impl LlmJpTextModel {
    /// Token embedding lookup, the text half of the multimodal merge.
    pub fn get_embed_tokens(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        on_text_model!(self, inner => inner.get_embed_tokens(input_ids))
    }

    /// Which `llm_config.model_type` this arm was built for. Diagnostics only.
    pub fn backbone_name(&self) -> &'static str {
        match self {
            LlmJpTextModel::Llama(_) => "llama",
            LlmJpTextModel::Qwen3(_) => "qwen3",
        }
    }
}

impl LanguageModel for LlmJpTextModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        on_text_model!(self, inner => LanguageModel::forward(inner, input_ids, caches, mask))
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        on_text_model!(self, inner => LanguageModel::forward_with_embeddings(
            inner,
            input_ids,
            input_embeddings,
            caches,
            mask,
        ))
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        on_text_model!(self, inner => LanguageModel::embed_tokens(inner, input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        on_text_model!(self, inner => LanguageModel::make_caches(inner))
    }

    fn set_kv_cache_layer_modes(&self, modes: Vec<KVCacheMode>) {
        on_text_model!(self, inner => LanguageModel::set_kv_cache_layer_modes(inner, modes))
    }

    fn kv_cache_layer_modes(&self) -> Option<Vec<KVCacheMode>> {
        on_text_model!(self, inner => LanguageModel::kv_cache_layer_modes(inner))
    }

    fn num_layers(&self) -> usize {
        on_text_model!(self, inner => LanguageModel::num_layers(inner))
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        on_text_model!(self, inner => LanguageModel::eos_token_ids(inner))
    }

    fn forward_batched(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        on_text_model!(self, inner => LanguageModel::forward_batched(
            inner,
            input_ids,
            batch_caches,
            mask,
        ))
    }

    fn forward_batched_with_context(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
        context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        on_text_model!(self, inner => LanguageModel::forward_batched_with_context(
            inner,
            input_ids,
            batch_caches,
            mask,
            context,
        ))
    }

    fn forward_batched_with_context_and_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
        context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        on_text_model!(self, inner => LanguageModel::forward_batched_with_context_and_ids(
            inner,
            input_ids,
            seq_ids,
            batch_caches,
            mask,
            context,
        ))
    }

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        on_text_model!(self, inner => LanguageModel::sequence_state_layout(inner))
    }

    fn supports_batching(&self) -> bool {
        on_text_model!(self, inner => LanguageModel::supports_batching(inner))
    }

    fn supports_batched_prefill(&self) -> bool {
        on_text_model!(self, inner => LanguageModel::supports_batched_prefill(inner))
    }

    fn supports_maskless_padded_prefill(&self) -> bool {
        on_text_model!(self, inner => LanguageModel::supports_maskless_padded_prefill(inner))
    }

    fn supports_paged_decode_backend(&self) -> bool {
        on_text_model!(self, inner => LanguageModel::supports_paged_decode_backend(inner))
    }
}
