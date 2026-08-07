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

//! Jina VLM (`model_type: "jvlm"`) vision-language runtime.
//!
//! Composes the SigLIP-class tower plus VL connector
//! ([`crate::vision::encoders::jina_vlm`]) with the Qwen2-class decoder
//! ([`crate::models::jina_vlm`]) and scatters the connector output into the
//! prompt embeddings at the positions named by the processor's
//! `image_input_idx`.
//!
//! **The scatter is additive, and that is equivalent to the reference's
//! assignment.** Upstream `JinaVLMModel._encode_images` adds the `<im_patch>`
//! (151938) token embedding to every connector feature and then *assigns* the
//! sum over the target rows. Every target row is an `<im_patch>` token by
//! construction (`image_input_idx` is built from `nonzero(tokens ==
//! <im_patch>)`), so its embedding is already exactly the term being added, and
//! `embed + feature` equals `feature + embed`. Adding avoids a second embedding
//! lookup and a masked write.

use crate::models::jina_vlm::JinaVlmTextModel;
use crate::vision::encoders::jina_vlm::JinaVlmVisionModel;
use crate::vision::merge::InputEmbeddings;
use crate::vision::processors::jina_vlm::JinaVlmProcessor;
use mlxcel_core::cache::SequenceStateLayout;
use mlxcel_core::generate::{DecodeBatchContext, LanguageModel};
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

pub struct JinaVlmModel {
    pub text_model: JinaVlmTextModel,
    pub vision_model: JinaVlmVisionModel,
    pub processor: JinaVlmProcessor,
    /// `<|image|>` (151940): the placeholder the image-token block replaces.
    pub image_prompt_token_id: i32,
    /// `processor_config.json`'s `always_start_with_space`. The chat template
    /// emits a leading space before every turn, and for the first turn only
    /// when this is set; the released checkpoint sets it.
    pub always_start_with_space: bool,
}

impl JinaVlmModel {
    /// Merge vision features into the prompt embeddings.
    ///
    /// * `input_ids`: `[1, seq_len]`.
    /// * `pixel_values`: `[n_crops, n_patches, patch_dim]`.
    /// * `image_input_idx`: flat target positions, one per pooled patch, kept on
    ///   the host because the scatter is resolved there; negative entries are
    ///   padding and are skipped.
    /// * `image_masks`: `[n_crops, n_patches]` per-patch coverage.
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        image_input_idx: &[i32],
        image_masks: &MlxArray,
    ) -> InputEmbeddings {
        let x = self.text_model.embedding.forward(input_ids);
        let x_shape = mlxcel_core::array_shape(&x);

        let pv_shape = mlxcel_core::array_shape(pixel_values);
        let images =
            mlxcel_core::reshape(pixel_values, &[1, pv_shape[0], pv_shape[1], pv_shape[2]]);
        let masks = mlxcel_core::reshape(image_masks, &[1, pv_shape[0], pv_shape[1]]);

        // [1, n_crops, tokens_per_crop, text_hidden] -> [n_features, hidden]
        let features = self.vision_model.forward(&images, &masks);
        let feat_shape = mlxcel_core::array_shape(&features);
        let hidden = feat_shape[feat_shape.len() - 1];
        let features = mlxcel_core::reshape(&features, &[-1, hidden]);
        // Row count of the *reshaped* `[-1, hidden]` features. The gather below
        // is unchecked: MLX's `take` only wraps negative indices, so a row past
        // the end is an out-of-bounds device read, which is either silent
        // garbage or an illegal-address fault that poisons the CUDA context and
        // takes the whole process down rather than one request. The processor
        // and the tower agree on this count today, but they parse it from
        // separate config sources, so enforce it where the pairs are built.
        let feature_row_count = mlxcel_core::array_shape(&features)[0].max(0) as usize;
        let text_dtype = mlxcel_core::array_dtype(&x);
        let features = if mlxcel_core::array_dtype(&features) != text_dtype {
            mlxcel_core::astype(&features, text_dtype)
        } else {
            features
        };

        // Resolve the (feature row, target position) pairs. The processor
        // already produced these on the host, so this costs no device readback.
        let seq_len: i32 = x_shape.iter().take(x_shape.len() - 1).product();
        let mut target_positions: Vec<i32> = Vec::new();
        let mut feature_rows: Vec<i32> = Vec::new();
        for (row, &position) in image_input_idx.iter().enumerate() {
            if position >= 0 && position < seq_len && row < feature_row_count {
                target_positions.push(position);
                // `row < feature_row_count` and the count came from an `i32`
                // shape, so this cast cannot wrap.
                feature_rows.push(row as i32);
            }
        }

        if target_positions.is_empty() {
            return InputEmbeddings {
                inputs_embeds: x,
                attention_mask_4d: None,
            };
        }

        let h_dim = x_shape[x_shape.len() - 1];
        let flat_x = mlxcel_core::reshape(&x, &[seq_len, h_dim]);

        let n_targets = feature_rows.len() as i32;
        let feat_idx = mlxcel_core::from_slice_i32(&feature_rows, &[n_targets]);
        let active = mlxcel_core::take(&features, &feat_idx, 0);

        // Sparse row scatter into the target positions. This replaces a dense
        // `one_hot(target_positions) @ active`, whose cost was quadratic in the
        // image count because both `seq_len` and the target count grow with it:
        // a 16-image request built a multi-gigabyte one-hot matrix and ran a
        // teraflop-scale GEMM on the serialized model thread just to move a few
        // tens of thousands of rows. `scatter_add` is the same operation, since
        // the merge is additive and every non-selected one-hot term was an exact
        // `0.0 * x`, at the cost of the rows actually written.
        //
        // `features` was cast to `text_dtype` above, so `active` (and therefore
        // `updates`) already matches `flat_x`.
        let pos_arr = mlxcel_core::from_slice_i32(&target_positions, &[n_targets]);
        let updates = mlxcel_core::reshape(&active, &[n_targets, 1, h_dim]);
        let merged = mlxcel_core::scatter_add(&flat_x, &pos_arr, &updates, 0);

        InputEmbeddings {
            inputs_embeds: mlxcel_core::reshape(&merged, &x_shape),
            attention_mask_4d: None,
        }
    }
}

impl LanguageModel for JinaVlmModel {
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

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.text_model.embedding.forward(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.text_model.make_caches()
    }

    fn num_layers(&self) -> usize {
        self.text_model.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.text_model.eos_token_ids.clone()
    }

    /// The decoder keeps all of its state in the external KV cache, so declare
    /// the dense layout explicitly rather than letting the trait default infer
    /// it. The default reads `supports_batching()`, and a wrapper that ever
    /// returns `false` there would hand `forward` an empty cache slice on the
    /// server path and silently run zero decoder layers.
    fn sequence_state_layout(&self) -> SequenceStateLayout {
        LanguageModel::sequence_state_layout(&self.text_model)
    }

    fn supports_batching(&self) -> bool {
        LanguageModel::supports_batching(&self.text_model)
    }

    /// `<im_start>` / `<im_end>` / `<im_patch>` / `<im_col>` / `<|image|>` are
    /// structural: they only ever appear inside a processor built image block,
    /// and sampling one would open a block with no features behind it.
    fn output_suppressed_token_ids(&self) -> Vec<i32> {
        let tokens = &self.processor.tokens;
        vec![
            tokens.image_start_id,
            tokens.image_end_id,
            tokens.image_patch_id,
            tokens.image_col_id,
            self.image_prompt_token_id,
        ]
    }
}

#[cfg(test)]
#[path = "jina_vlm_tests.rs"]
mod tests;
