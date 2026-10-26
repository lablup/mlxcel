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

//! Shared per-row batched dispatch for vision-language wrappers.
//!
//! Several VLM wrappers need an identical override on
//! [`LanguageModel::forward_batched_with_ids`] so each row of a
//! batched call reaches the text model's seq-aware forward path with its
//! own `seq_id`. Without the override, the trait default discards `seq_ids`
//! before they reach the model. Qwen VL now has a native identified decode
//! path and calls this helper only for conditional fallbacks. The GLM-4V,
//! GLM-4V-MoE, Hunyuan-VL, ERNIE-4.5-MoE-VL, PaddleOCR-VL and Cohere Compass
//! towers do the same. Gemma 4 VLM and Unified likewise retain it only for
//! multimodal prefill and conditional state fallbacks around native decode.
//!
//! The helper is generic over `M: LanguageModel` so families with different
//! per-sequence state (Qwen MRoPE deltas, Gemma 4 sliding-window caches)
//! reuse the same loop without coupling the families to each other.

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

/// Per-row batched dispatch helper. Loops over each row of `input_ids`
/// and calls `text_model.forward_with_sequence_id` with the corresponding
/// `seq_ids[i]`, then concatenates the per-row logits along axis 0 into
/// a `[B, T, V]` tensor.
///
/// When `seq_ids` is `None`, this falls back to the model's default
/// `forward_batched` so non-server callers (CLI, single-
/// process VLM batches) keep their existing behavior — they never had a
/// scheduler-allocated `SequenceId` to plumb through.
///
/// When the batch is empty, returns a zero-shaped placeholder so the
/// caller does not have to special-case the zero-batch path.
///
/// When the batch contains exactly one row, this still routes through
/// `forward_with_sequence_id` so the model's per-sequence state
/// (MRoPE entry, per-sequence KV cache, etc.) resolves to the right
/// slot — there is no fast-path skip back to `forward()`.
///
/// Used by:
/// - Qwen2VLModel, Qwen25VLModel, Qwen3VLModel, Qwen3VLMoeModel and
///   Qwen3OmniMoeModel (conditional fallback around their native decode).
/// - Glm4vTextModel (shared by GLM-4V and GLM-OCR), Glm4vMoeTextModel,
///   HunyuanVlTextModel, Ernie45MoeVlTextModel, PaddleOcrTextModel and
///   CohereCompassTextModel (prefill, missing-id and conditional state
///   fallbacks around their native decode).
/// - Gemma4VLModel and Gemma4UnifiedModel (multimodal prefill and pending
///   per-layer-input fallbacks around their native text decode).
///
/// Not used by Qwen35VLModel: that family's text model already implements
/// a native batched-with-ids fast path (per-row dispatch and a true batched-
/// prefill kernel) and wires it directly into the wrapper.
pub fn forward_batched_with_seq_ids_dispatch<M: LanguageModel + ?Sized>(
    text_model: &M,
    input_ids: &MlxArray,
    seq_ids: Option<&[SequenceId]>,
    batch_caches: &mut [&mut [KVCache]],
    mask: Option<&MlxArray>,
) -> UniquePtr<MlxArray> {
    let Some(seq_ids) = seq_ids else {
        return text_model.forward_batched(input_ids, batch_caches, mask);
    };

    let b = batch_caches.len();
    if b == 0 {
        return mlxcel_core::zeros(&[0, 1, 1], mlxcel_core::dtype::FLOAT32);
    }

    let shape = mlxcel_core::array_shape(input_ids);
    let row_len = if shape.len() >= 2 { shape[1] } else { 1 };

    if b == 1 {
        return text_model.forward_with_sequence_id(
            input_ids,
            seq_ids.first().copied(),
            batch_caches[0],
            mask,
        );
    }

    // Per-row dispatch: slice input_ids row-by-row and forward each row
    // with its own seq_id so the per-sequence model state resolves.
    let token_0 = mlxcel_core::slice(input_ids, &[0, 0], &[1, row_len]);
    let mut result = text_model.forward_with_sequence_id(
        &token_0,
        seq_ids.first().copied(),
        batch_caches[0],
        None,
    );
    for (i, caches) in batch_caches.iter_mut().enumerate().skip(1) {
        let token_i = mlxcel_core::slice(input_ids, &[i as i32, 0], &[i as i32 + 1, row_len]);
        let logits_i =
            text_model.forward_with_sequence_id(&token_i, seq_ids.get(i).copied(), caches, None);
        result = mlxcel_core::concatenate(&result, &logits_i, 0);
    }
    result
}
