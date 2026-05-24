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

//! MiniCPM-V 4.6 vision-language model.
//!
//! MiniCPM-V 4.6 uses Qwen3.5 (standard 1D RoPE, not Qwen3-VL / MRoPE) as
//! the text backbone.  The vision pipeline is:
//!   SigLIP ViT (27 layers) + VitMerger at layer 6 + post_layernorm + Merger
//!
//! The image embedding preparation reuses the MiniCPM-O prompt/processor
//! infrastructure (same token format: `<image><unk>...<unk></image>`).

use crate::LanguageModel;
use crate::vision::merge::InputEmbeddings;
use crate::vision::{encoders, processors};
use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::DecodeBatchContext;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct MiniCPMV46VLModel {
    pub text_model: crate::models::Qwen35Model,
    pub vision_model: encoders::minicpmv4_6::MiniCPMV46VisionModel,
    /// Shared processor for image resizing and normalization.
    /// MiniCPM-V 4.6 uses the same patch-based resizing as MiniCPM-O.
    pub processor: processors::minicpmo::MiniCPMOProcessor,
    pub eos_token_ids: Vec<i32>,
}

impl MiniCPMV46VLModel {
    pub fn image_feature_size_for_processed(
        &self,
        processed: &processors::minicpmo::MiniCPMOImageInput,
    ) -> Result<usize, String> {
        self.vision_model
            .output_token_count_for_spatial_shape(processed.spatial_shape)
    }

    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        processed_images: &[processors::minicpmo::MiniCPMOImageInput],
        image_bounds: &[(usize, usize)],
    ) -> InputEmbeddings {
        let inputs_embeds = self.text_model.get_embed_tokens(input_ids);

        if processed_images.is_empty() || image_bounds.is_empty() {
            return InputEmbeddings {
                inputs_embeds,
                attention_mask_4d: None,
            };
        }

        let hidden_size = mlxcel_core::array_shape(&inputs_embeds)[2];
        let mut segments: Vec<UniquePtr<MlxArray>> = Vec::new();
        let mut previous_end = 0usize;

        for (image_idx, &(start, end)) in image_bounds.iter().enumerate() {
            if start > previous_end {
                let segment = mlxcel_core::slice(
                    &inputs_embeds,
                    &[0, previous_end as i32, 0],
                    &[1, start as i32, hidden_size],
                );
                segments.push(segment);
            }

            if let Some(processed) = processed_images.get(image_idx) {
                let pixel_values = mlxcel_core::from_slice_f32(
                    &processed.pixel_values,
                    &processed.pixel_values_shape,
                );
                let pixel_values =
                    mlxcel_core::astype(&pixel_values, mlxcel_core::array_dtype(&inputs_embeds));

                let spatial_shape = processed.spatial_shape;
                // Run the full vision pipeline.
                let vision_tokens = self.vision_model.forward(&pixel_values, spatial_shape);

                // vision_tokens shape: [num_tokens, hidden_size]
                // We need [1, num_tokens, hidden_size] to slice into the embedding.
                let num_tokens = mlxcel_core::array_shape(&vision_tokens)[0];
                let span_len = end.saturating_sub(start);

                // Upstream (minicpmv4_6.py:388-393) raises a ValueError when the
                // emitted vision-token count does not match the reserved
                // placeholder span. We log a warning and clip/zero-pad instead of
                // hard-erroring: the validated 448×448 path produces exactly
                // `span_len` tokens, and a soft fallback avoids aborting otherwise
                // working multi-image / odd-resolution generations. A mismatch
                // here means the processor's `image_feature_size` and the vision
                // pipeline's effective merge geometry disagree.
                if num_tokens != span_len as i32 {
                    tracing::warn!(
                        image_index = image_idx,
                        emitted_vision_tokens = num_tokens,
                        reserved_placeholder_span = span_len,
                        "MiniCPM-V 4.6: vision-token count does not match image \
                         placeholder span; clipping/zero-padding to fit"
                    );
                }

                let use_tokens = num_tokens.min(span_len as i32) as i32;

                if use_tokens > 0 {
                    let clipped = if use_tokens < num_tokens {
                        mlxcel_core::slice(&vision_tokens, &[0, 0], &[use_tokens, hidden_size])
                    } else {
                        mlxcel_core::copy(&vision_tokens)
                    };
                    // Cast to embedding dtype.
                    let clipped =
                        mlxcel_core::astype(&clipped, mlxcel_core::array_dtype(&inputs_embeds));
                    // Expand batch dim.
                    let expanded = mlxcel_core::expand_dims(&clipped, 0);
                    segments.push(expanded);
                }

                // Fill any remaining placeholder slots with zeros so the total
                // sequence length stays consistent with what the tokenizer saw.
                let filled = use_tokens as usize;
                if filled < span_len {
                    let pad = mlxcel_core::zeros(
                        &[1, (span_len - filled) as i32, hidden_size],
                        mlxcel_core::array_dtype(&inputs_embeds),
                    );
                    segments.push(pad);
                }
            }

            previous_end = end;
        }

        let total_len = mlxcel_core::array_shape(input_ids)[1] as usize;
        if previous_end < total_len {
            let tail = mlxcel_core::slice(
                &inputs_embeds,
                &[0, previous_end as i32, 0],
                &[1, total_len as i32, hidden_size],
            );
            segments.push(tail);
        }

        let final_embeds = if segments.is_empty() {
            inputs_embeds
        } else {
            encoders::minicpmo::concat_arrays(&segments, 1)
        };

        InputEmbeddings {
            inputs_embeds: final_embeds,
            attention_mask_4d: None,
        }
    }
}

impl LanguageModel for MiniCPMV46VLModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model.forward_impl(input_ids, None, caches, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_impl(input_ids, input_embeddings, caches, mask)
    }

    // ── Qwen3.5 model-owned sequence-state forwarding ───────────────────────
    //
    // Qwen3.5 is a hybrid model (GatedDelta linear + full attention) that
    // keeps per-sequence recurrent state INSIDE the text model and ignores
    // the external KVCache for the linear layers. Every one of these methods
    // must be forwarded to `self.text_model`; inheriting the `LanguageModel`
    // trait defaults would yield the wrong cache layout, collide concurrent
    // sequences, and leave stale recurrent state across `generate` calls.
    // Mirrors `Qwen35VLModel` (see `qwen3_5_vl.rs`). The vision embeddings are
    // injected via `forward_with_embeddings_and_sequence_id`'s
    // `input_embeddings` argument exactly as in the non-seq-id path.

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        mlxcel_core::generate::LanguageModel::forward_with_sequence_id(
            &self.text_model,
            input_ids,
            seq_id,
            caches,
            mask,
        )
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        mlxcel_core::generate::LanguageModel::forward_with_embeddings_and_sequence_id(
            &self.text_model,
            input_ids,
            input_embeddings,
            seq_id,
            caches,
            mask,
        )
    }

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        mlxcel_core::generate::LanguageModel::prepare_sequence_state(&self.text_model, seq_id);
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        mlxcel_core::generate::LanguageModel::release_sequence_state_by_id(
            &self.text_model,
            seq_id,
        );
    }

    fn reset_runtime_state(&self) {
        mlxcel_core::generate::LanguageModel::reset_runtime_state(&self.text_model);
    }

    fn sequence_state_layout(&self) -> mlxcel_core::cache::SequenceStateLayout {
        mlxcel_core::generate::LanguageModel::sequence_state_layout(&self.text_model)
    }

    fn supports_batching(&self) -> bool {
        mlxcel_core::generate::LanguageModel::supports_batching(&self.text_model)
    }

    fn supports_batched_prefill(&self) -> bool {
        mlxcel_core::generate::LanguageModel::supports_batched_prefill(&self.text_model)
    }

    fn supports_paged_decode_backend(&self) -> bool {
        mlxcel_core::generate::LanguageModel::supports_paged_decode_backend(&self.text_model)
    }

    fn forward_batched_with_context_and_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
        context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        mlxcel_core::generate::LanguageModel::forward_batched_with_context_and_ids(
            &self.text_model,
            input_ids,
            seq_ids,
            batch_caches,
            mask,
            context,
        )
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.text_model.get_embed_tokens(input_ids))
    }

    /// Delegate to the inner text model so a DFlash drafter can lazy-bind the
    /// Qwen3.5 embedding table even when the target is the VLM-wrapped
    /// checkpoint. Token embedding always lives on the text model — the
    /// vision tower owns no token embedding. Mirrors `Qwen35VLModel`.
    fn embed_tokens_module(&self) -> Option<mlxcel_core::layers::UnifiedEmbedding> {
        mlxcel_core::generate::LanguageModel::embed_tokens_module(&self.text_model)
    }

    fn lm_head_module(&self) -> Option<mlxcel_core::layers::UnifiedLinear> {
        mlxcel_core::generate::LanguageModel::lm_head_module(&self.text_model)
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
