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

//! Molmo v1 Vision-Language Model.
//!
//! CLIP-style ViT vision encoder + attention pooling + SwiGLU projector +
//! OLMo-style text decoder. Like Molmo2 it uses an **additive** merge — vision
//! features are ADDED to the text embeddings — but the target positions come
//! from the processor-supplied `image_input_idx` rather than a scan for an
//! `image_patch_id` token (reference `molmo.py:get_input_embeddings`).

use super::{encoders, merge, processors};
use crate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

/// Molmo v1 Vision-Language Model.
pub struct MolmoVLModel {
    pub text_model: crate::models::molmo::MolmoModel,
    pub vision_tower: encoders::molmo::MolmoVisionModel,
    pub processor: processors::molmo::MolmoProcessor,
}

impl MolmoVLModel {
    /// Get input embeddings with vision features additively merged at the
    /// positions named by `image_input_idx`.
    ///
    /// * `input_ids`: `[1, seq_len]` combined token stream (image tokens first).
    /// * `pixel_values`: `[n_crops, n_patches, patch_dim]`.
    /// * `image_input_idx`: `[num_image * tokens_per_image]` flat positions
    ///   (negative entries are skipped).
    /// * `image_masks`: `[n_crops, n_patches]` per-patch coverage.
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        image_input_idx: &MlxArray,
        image_masks: &MlxArray,
    ) -> merge::InputEmbeddings {
        // 1. Embed the text tokens.
        let x = self.text_model.wte.forward(input_ids);
        let x_shape = mlxcel_core::array_shape(&x);

        // 2. Add batch dims for the vision tower (batch=1 inference path).
        let pv_shape = mlxcel_core::array_shape(pixel_values);
        let n_crops = pv_shape[0];
        let n_patches = pv_shape[1];
        let patch_dim = pv_shape[2];
        let images = mlxcel_core::reshape(pixel_values, &[1, n_crops, n_patches, patch_dim]);
        let masks = mlxcel_core::reshape(image_masks, &[1, n_crops, n_patches]);

        // 3. Encode -> [1, num_image, h*w, hidden].
        let image_features = self.vision_tower.forward(&images, &masks);
        let feat_shape = mlxcel_core::array_shape(&image_features);
        let hidden = feat_shape[feat_shape.len() - 1];
        // Flatten to [num_image * h*w, hidden].
        let image_features = mlxcel_core::reshape(&image_features, &[-1, hidden]);
        let img_dtype = mlxcel_core::array_dtype(&image_features);
        let txt_dtype = mlxcel_core::array_dtype(&x);
        let image_features = if img_dtype != txt_dtype {
            mlxcel_core::astype(&image_features, txt_dtype)
        } else {
            image_features
        };

        // 4. Resolve valid (feature_row, target_position) pairs on the host.
        mlxcel_core::eval(image_input_idx);
        let idx_len = mlxcel_core::array_shape(image_input_idx)[0];
        let mut target_positions: Vec<i32> = Vec::new();
        let mut feature_rows: Vec<i32> = Vec::new();
        for i in 0..idx_len {
            let slot = mlxcel_core::slice(image_input_idx, &[i], &[i + 1]);
            let v = mlxcel_core::item_i32(&slot);
            if v >= 0 {
                target_positions.push(v);
                feature_rows.push(i);
            }
        }

        if target_positions.is_empty() {
            return merge::InputEmbeddings {
                inputs_embeds: x,
                attention_mask_4d: None,
            };
        }

        // 5. Additive scatter: gather the active feature rows, then add them to
        //    the embedding rows at `target_positions`.
        let h_dim = x_shape[x_shape.len() - 1];
        let total_tokens: i32 = x_shape.iter().take(x_shape.len() - 1).product();
        let flat_x = mlxcel_core::reshape(&x, &[total_tokens, h_dim]);

        let feat_idx = mlxcel_core::from_slice_i32(&feature_rows, &[feature_rows.len() as i32]);
        let active_feats = mlxcel_core::take(&image_features, &feat_idx, 0);

        // spread = one_hot(target_positions) @ active_feats -> [total_tokens, h_dim]
        let num_active = target_positions.len() as i32;
        let row_indices: Vec<i32> = (0..total_tokens).collect();
        let row_arr = mlxcel_core::from_slice_i32(&row_indices, &[total_tokens, 1]);
        let pos_arr = mlxcel_core::from_slice_i32(&target_positions, &[1, num_active]);
        let one_hot = mlxcel_core::equal(&row_arr, &pos_arr);
        let one_hot_f = mlxcel_core::astype(&one_hot, mlxcel_core::array_dtype(&active_feats));
        let spread = mlxcel_core::matmul(&one_hot_f, &active_feats);
        let spread = mlxcel_core::astype(&spread, mlxcel_core::array_dtype(&flat_x));
        let merged = mlxcel_core::add(&flat_x, &spread);
        let x = mlxcel_core::reshape(&merged, &x_shape);

        merge::InputEmbeddings {
            inputs_embeds: x,
            attention_mask_4d: None,
        }
    }
}

impl LanguageModel for MolmoVLModel {
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

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.text_model.wte.forward(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.text_model.make_caches()
    }

    fn num_layers(&self) -> usize {
        self.text_model.blocks.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        mlxcel_core::generate::LanguageModel::eos_token_ids(&self.text_model)
    }
}
