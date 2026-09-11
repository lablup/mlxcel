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

//! Molmo-Point Vision-Language Model
//!
//! Custom ViT vision encoder + attention-based 2D pooling + SwiGLU projector
//! + Molmo2 text model + point prediction.
//!
//! Uses additive merge: vision features are ADDED to text embeddings at
//! image_patch_id positions, same as Molmo2.
//!
//! The point prediction layer extends the vocabulary during generation to
//! encode spatial coordinates as patch/subpatch/location tokens.

use super::{encoders, merge, processors};
use crate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use encoders::molmo_point::{MolmoPointConnector, PointPredictor};
use encoders::molmo2::Molmo2VisionTransformer;

/// Molmo-Point VLM configuration extracted from config.json
pub struct MolmoPointConfig {
    pub image_patch_id: i32,
    pub image_end_token_id: i32,
    pub image_non_indexable_patch_id: i32,
    pub patch_token_id: i32,
    pub subpatch_token_id: i32,
    pub location_token_id: i32,
    pub no_more_points_class: bool,
    pub norm_logits: bool,
    pub patch_location: Option<String>, // e.g. "3x3"
    pub vit_layers: Vec<i32>,
    pub hidden_size: i32,
}

/// Molmo-Point Vision-Language Model
pub struct MolmoPointVLModel {
    pub(crate) language_model: crate::models::molmo_point::MolmoPointLanguageModel,
    pub(crate) vision_model: Molmo2VisionTransformer,
    pub(crate) connector: MolmoPointConnector,
    #[allow(dead_code)] // Used for point coordinate extraction (future)
    pub(crate) point_predictor: PointPredictor,
    #[allow(dead_code)] // Used for ViT embedding construction (future)
    pub(crate) build_vit_embedding: mlxcel_core::layers::Linear, // vit_dim -> llm_dim
    pub(crate) processor: processors::molmo2::Molmo2Processor,
    pub(crate) config: MolmoPointConfig,
    pub(crate) vit_layers: Vec<usize>, // Resolved positive layer indices
}

impl MolmoPointVLModel {
    /// Get input embeddings with vision features additively merged.
    ///
    /// This is the prefill path: process images through ViT, pool, project,
    /// and add to text embeddings at image token positions.
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        image_token_pooling: &MlxArray,
        _image_grids: &MlxArray,
        _image_num_crops: &MlxArray,
    ) -> merge::InputEmbeddings {
        // 1. Embed all tokens (replace -1 with 0 for safe lookup)
        let ids_i32 = mlxcel_core::astype(input_ids, mlxcel_core::dtype::INT32);
        let neg_one = mlxcel_core::from_slice_i32(&[-1], &[1]);
        let zero = mlxcel_core::from_slice_i32(&[0], &[1]);
        let is_neg = mlxcel_core::equal(&ids_i32, &neg_one);
        let safe_ids = mlxcel_core::where_cond(&is_neg, &zero, &ids_i32);
        let x = self.language_model.model.wte.forward(&safe_ids);
        let x_shape = mlxcel_core::array_shape(&x);
        let batch_size = x_shape[0];
        let dim = x_shape[x_shape.len() - 1];

        // 2. Build batched images (batch=1 simplified)
        let (images, token_pooling) = self.build_batched_images(
            input_ids,
            pixel_values,
            image_token_pooling,
            _image_grids,
            _image_num_crops,
        );

        // 3. Identify image token positions
        let patch_token = mlxcel_core::from_slice_i32(&[self.config.image_patch_id], &[1]);
        let non_idx_token =
            mlxcel_core::from_slice_i32(&[self.config.image_non_indexable_patch_id], &[1]);
        let is_indexable = mlxcel_core::equal(input_ids, &patch_token);
        let is_non_indexable = mlxcel_core::equal(input_ids, &non_idx_token);
        // Combine: any image token position
        let is_image_i32 = mlxcel_core::add(
            &mlxcel_core::astype(&is_indexable, mlxcel_core::dtype::INT32),
            &mlxcel_core::astype(&is_non_indexable, mlxcel_core::dtype::INT32),
        );
        let is_image = mlxcel_core::greater(&is_image_i32, &mlxcel_core::zeros_like(&is_image_i32));

        // 4. Encode images through ViT
        let img_shape = mlxcel_core::array_shape(&images);
        let b = img_shape[0];
        let t = img_shape[1];
        let n = img_shape[2];
        let d = img_shape[3];
        let images_flat = mlxcel_core::reshape(&images, &[b * t, n, d]);
        // Cast to vision model dtype
        let vit_dtype = mlxcel_core::dtype::FLOAT16;
        let images_cast = mlxcel_core::astype(&images_flat, vit_dtype);
        let vit_hidden_states = self.vision_model.forward(&images_cast, None);

        // 5. Select and concatenate features from specified layers
        let mut features_list: Vec<&MlxArray> = Vec::new();
        for &layer_idx in &self.vit_layers {
            if let Some(arr) = vit_hidden_states[layer_idx].as_ref() {
                features_list.push(arr);
            }
        }
        let vit_features = if features_list.len() == 1 {
            mlxcel_core::copy(features_list[0])
        } else {
            let mut result = mlxcel_core::copy(features_list[0]);
            for &feat in &features_list[1..] {
                result = mlxcel_core::concatenate(&result, feat, -1);
            }
            result
        };

        let vit_feat_shape = mlxcel_core::array_shape(&vit_features);
        let vit_feature_dim = vit_feat_shape[vit_feat_shape.len() - 1];

        // Reshape to [batch, total_patches, dim]
        let vit_features = mlxcel_core::reshape(&vit_features, &[batch_size, -1, vit_feature_dim]);

        // 6. Gather features using token pooling indices
        let tp_shape = mlxcel_core::array_shape(&token_pooling);
        let n_pooled = tp_shape[1];
        let pool_size = tp_shape[2];

        let clamped_pooling = {
            let zero_arr = mlxcel_core::zeros_like(&token_pooling);
            let feat_len = mlxcel_core::array_shape(&vit_features)[1];
            let max_idx_data = vec![feat_len - 1; 1];
            let max_idx = mlxcel_core::from_slice_i32(&max_idx_data, &[1]);
            let clipped = mlxcel_core::maximum(&token_pooling, &zero_arr);
            mlxcel_core::clip(&clipped, &zero_arr, &max_idx)
        };

        let vit_features_gathered =
            self.batched_gather(&vit_features, &clamped_pooling, batch_size);
        // Mask invalid positions (where token_pooling < 0)
        let valid_mask =
            mlxcel_core::greater_equal(&token_pooling, &mlxcel_core::zeros_like(&token_pooling));
        let valid_4d = mlxcel_core::reshape(&valid_mask, &[tp_shape[0], n_pooled, pool_size, 1]);
        let valid_f =
            mlxcel_core::astype(&valid_4d, mlxcel_core::array_dtype(&vit_features_gathered));
        let vit_features_gathered = mlxcel_core::multiply(&vit_features_gathered, &valid_f);

        // 7. Build sparse features for connector
        let image_features_mask = mlxcel_core::any_axis(&valid_mask, -1, false);

        // Extract valid indices on host
        let flat_mask = mlxcel_core::reshape(&image_features_mask, &[-1]);
        mlxcel_core::eval(&flat_mask);
        let total_pooled_count = mlxcel_core::array_shape(&flat_mask)[0];
        let mut valid_indices: Vec<i32> = Vec::new();
        for i in 0..total_pooled_count {
            let idx_arr = mlxcel_core::from_slice_i32(&[i], &[1]);
            let val = mlxcel_core::take(&flat_mask, &idx_arr, 0);
            mlxcel_core::eval(&val);
            if mlxcel_core::item_bool(&val) {
                valid_indices.push(i);
            }
        }

        let vit_features_flat =
            mlxcel_core::reshape(&vit_features_gathered, &[-1, pool_size, vit_feature_dim]);
        let vit_mask_flat = mlxcel_core::reshape(&valid_mask, &[-1, pool_size]);

        let (vit_features_sparse, vit_mask_sparse) = if valid_indices.is_empty() {
            (
                mlxcel_core::zeros(
                    &[0, pool_size, vit_feature_dim],
                    mlxcel_core::array_dtype(&vit_features_flat),
                ),
                mlxcel_core::zeros(&[0, pool_size], mlxcel_core::dtype::BOOL),
            )
        } else {
            let idx_arr =
                mlxcel_core::from_slice_i32(&valid_indices, &[valid_indices.len() as i32]);
            (
                mlxcel_core::take(&vit_features_flat, &idx_arr, 0),
                mlxcel_core::take(&vit_mask_flat, &idx_arr, 0),
            )
        };

        // 8. Apply connector (attention pooling + projection)
        let image_features = self
            .connector
            .forward(&vit_features_sparse, &vit_mask_sparse);

        // 9. Find image token positions and add features
        let flat_is_image = mlxcel_core::reshape(&is_image, &[-1]);
        mlxcel_core::eval(&flat_is_image);
        let total_tokens = mlxcel_core::array_shape(&flat_is_image)[0];
        let mut image_positions: Vec<i32> = Vec::new();
        for i in 0..total_tokens {
            let idx_arr = mlxcel_core::from_slice_i32(&[i], &[1]);
            let val = mlxcel_core::take(&flat_is_image, &idx_arr, 0);
            mlxcel_core::eval(&val);
            if mlxcel_core::item_bool(&val) {
                image_positions.push(i);
            }
        }

        let x = if !image_positions.is_empty() {
            // Additive scatter: x_flat += one_hot(positions) @ image_features
            let x_flat = mlxcel_core::reshape(&x, &[-1, dim]);
            let x_flat_f32 = mlxcel_core::astype(&x_flat, mlxcel_core::dtype::FLOAT32);
            let num_pos = image_positions.len() as i32;

            let row_indices: Vec<i32> = (0..total_tokens).collect();
            let row_arr = mlxcel_core::from_slice_i32(&row_indices, &[total_tokens, 1]);
            let pos_arr = mlxcel_core::from_slice_i32(&image_positions, &[1, num_pos]);
            let one_hot = mlxcel_core::equal(&row_arr, &pos_arr);
            let one_hot_f = mlxcel_core::astype(&one_hot, mlxcel_core::dtype::FLOAT32);
            let img_feat_flat = mlxcel_core::reshape(&image_features, &[-1, dim]);
            let img_feat_f32 = mlxcel_core::astype(&img_feat_flat, mlxcel_core::dtype::FLOAT32);
            let spread = mlxcel_core::matmul(&one_hot_f, &img_feat_f32);
            let merged = mlxcel_core::add(&x_flat_f32, &spread);
            mlxcel_core::reshape(&merged, &x_shape)
        } else {
            x
        };

        merge::InputEmbeddings {
            inputs_embeds: x,
            attention_mask_4d: None,
        }
    }

    /// Build batched images from flattened inputs (batch=1 simplified path)
    fn build_batched_images(
        &self,
        _input_ids: &MlxArray,
        pixel_values: &MlxArray,
        image_token_pooling: &MlxArray,
        _image_grids: &MlxArray,
        _image_num_crops: &MlxArray,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let pv_shape = mlxcel_core::array_shape(pixel_values);
        let n_crops = pv_shape[0];
        let n_patches = pv_shape[1];
        let patch_dim = pv_shape[2];
        let images = mlxcel_core::reshape(pixel_values, &[1, n_crops, n_patches, patch_dim]);

        let tp_shape = mlxcel_core::array_shape(image_token_pooling);
        let total_pooled = tp_shape[0];
        let pool_size = tp_shape[1];
        let token_pooling =
            mlxcel_core::reshape(image_token_pooling, &[1, total_pooled, pool_size]);

        (images, token_pooling)
    }

    /// Batched gather: for each batch, gather from features using indices
    fn batched_gather(
        &self,
        features: &MlxArray,
        indices: &MlxArray,
        batch_size: i32,
    ) -> UniquePtr<MlxArray> {
        let idx_shape = mlxcel_core::array_shape(indices);
        let num_pooled = idx_shape[1];
        let pool_size = idx_shape[2];
        let feat_shape = mlxcel_core::array_shape(features);
        let dim = feat_shape[2];

        let flat_idx = mlxcel_core::reshape(indices, &[batch_size, num_pooled * pool_size]);

        let mut batch_idx_data = Vec::with_capacity((batch_size * num_pooled * pool_size) as usize);
        for b in 0..batch_size {
            for _ in 0..(num_pooled * pool_size) {
                batch_idx_data.push(b);
            }
        }
        let batch_idx =
            mlxcel_core::from_slice_i32(&batch_idx_data, &[batch_size, num_pooled * pool_size]);

        let batch_idx_flat = mlxcel_core::reshape(&batch_idx, &[-1]);
        let flat_idx_flat = mlxcel_core::reshape(&flat_idx, &[-1]);

        let features_2d = mlxcel_core::reshape(features, &[batch_size * feat_shape[1], dim]);
        let num_patches = mlxcel_core::from_slice_i32(&[feat_shape[1]], &[1]);
        let offset = mlxcel_core::multiply(&batch_idx_flat, &num_patches);
        let linear_idx = mlxcel_core::add(&offset, &flat_idx_flat);

        let gathered = mlxcel_core::take(&features_2d, &linear_idx, 0);
        mlxcel_core::reshape(&gathered, &[batch_size, num_pooled, pool_size, dim])
    }
}

impl LanguageModel for MolmoPointVLModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.language_model.forward(input_ids, caches, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.language_model
            .forward_with_embeddings(input_ids, input_embeddings, caches, mask)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.language_model.model.wte.forward(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.language_model.make_caches()
    }

    fn num_layers(&self) -> usize {
        self.language_model.model.blocks.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        // Molmo-Point uses Qwen2 tokenizer EOS tokens
        vec![151645, 151643] // <|im_end|>, <|endoftext|>
    }
}
