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

//! Phi-3V Vision-Language Model
//!
//! CLIP-ViT-Large encoder + HD tiling + custom projector + Phi3 text backbone

use super::encoders::VisionEncoder;
use super::{encoders, merge, processors};
use crate::LanguageModel;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

/// Phi-3V VLM: CLIP-ViT-Large encoder + HD tiling + custom projector + Phi3 text backbone
///
/// Unique features:
/// - HD dynamic resolution with 336x336 tiles
/// - 2x2 spatial pooling → 4C features + glb_GN/sub_GN separators
/// - Image positions marked with negative token IDs (-1, -2, etc.)
/// - Uses existing SigLipVisionModel configured as CLIP (CLS token, pre-layernorm, penultimate layer)
pub struct Phi3VLModel {
    pub text_model: crate::models::Phi3Model,
    pub vision_encoder: encoders::siglip::SigLipVisionModel,
    pub glb_gn: UniquePtr<MlxArray>, // [1, 1, 4*image_dim_out]
    pub sub_gn: UniquePtr<MlxArray>, // [1, 1, 1, 4*image_dim_out]
    pub img_proj_linear1: mlxcel_core::layers::UnifiedLinear, // 4*image_dim_out → hidden_size
    pub img_proj_linear2: mlxcel_core::layers::UnifiedLinear, // hidden_size → hidden_size
    pub processor: processors::phi3_v::Phi3VProcessor,
    pub image_dim_out: usize, // 1024
}

impl Phi3VLModel {
    /// Get input embeddings with vision features merged at negative token positions
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        image_sizes: &[(usize, usize)],
    ) -> merge::InputEmbeddings {
        // 1. Get text embeddings
        let inputs_embeds = self.text_model.embed_tokens.forward(input_ids);
        let embed_dtype = mlxcel_core::array_dtype(&inputs_embeds);

        // 2. Find positions where input_ids < 0
        mlxcel_core::eval(input_ids);
        let ids_shape = mlxcel_core::array_shape(input_ids);
        let seq_len = ids_shape[1] as usize;

        let mut tokens = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let tok = mlxcel_core::slice(input_ids, &[0, i as i32], &[1, i as i32 + 1]);
            mlxcel_core::eval(&tok);
            tokens.push(mlxcel_core::item_i32(&tok));
        }

        // Find negative token positions (batch_idx, start_idx)
        let mut positions: Vec<(usize, usize)> = Vec::new();
        let mut i = 0;
        while i < seq_len {
            if tokens[i] < 0 {
                positions.push((0, i));
                i += 1;
            } else {
                i += 1;
            }
        }

        if positions.is_empty() || image_sizes.is_empty() {
            return merge::InputEmbeddings {
                inputs_embeds,
                attention_mask_4d: None,
            };
        }

        // 3. Process all tiles through CLIP encoder
        // pixel_values shape: [B, num_tiles, C, H, W] or [num_tiles, C, H, W]
        let pv_shape = mlxcel_core::array_shape(pixel_values);
        let pv = if pv_shape.len() == 5 {
            // [B, T, C, H, W] → [B*T, C, H, W]
            let b = pv_shape[0];
            let t = pv_shape[1];
            mlxcel_core::reshape(
                pixel_values,
                &[b * t, pv_shape[2], pv_shape[3], pv_shape[4]],
            )
        } else {
            mlxcel_core::copy(pixel_values)
        };

        // Transpose [B*T, C, H, W] → [B*T, H, W, C] for vision encoder
        let pv = mlxcel_core::transpose_axes(&pv, &[0, 2, 3, 1]);
        let pv = mlxcel_core::astype(&pv, embed_dtype);

        // Run through CLIP encoder (returns penultimate layer, CLS dropped)
        let encoder_output = self.vision_encoder.forward(&pv);
        let img_features = &encoder_output.hidden_states;
        // Shape: [B*T, 576, 1024] (576 = 24*24 patches, CLS already dropped)

        // Reshape to [B, num_tiles, 576, C]
        let feat_shape = mlxcel_core::array_shape(img_features);
        let num_patches = feat_shape[1]; // 576
        let c = feat_shape[2]; // 1024

        let h_patches = (num_patches as f32).sqrt() as i32; // 24
        let c_dim = self.image_dim_out; // 1024

        // 4. HD spatial pooling and GN injection per image
        let mut merged_embeds = mlxcel_core::copy(&inputs_embeds);
        let mut pos_idx = 0usize;

        // Count total tiles for indexing into img_features
        let mut tile_offset = 0i32;

        for &(hd_h, hd_w) in image_sizes {
            let h_tiles = hd_h / 336;
            let w_tiles = hd_w / 336;
            let b_ = h_tiles * w_tiles; // number of sub-tiles

            // Extract this image's features: global (1) + sub-tiles (b_)
            let num_this_image = (b_ + 1) as i32; // +1 for global
            let img_feats = mlxcel_core::slice(
                img_features,
                &[tile_offset, 0, 0],
                &[tile_offset + num_this_image, num_patches, c],
            );

            // Global image features: [1, 576, C]
            let glb_feats = mlxcel_core::slice(&img_feats, &[0, 0, 0], &[1, num_patches, c]);
            // Sub-image features: [b_, 576, C]
            let sub_feats =
                mlxcel_core::slice(&img_feats, &[1, 0, 0], &[1 + b_ as i32, num_patches, c]);

            // HD spatial pooling: reshape [b_, 24, 24, C] → [b_, 12, 2, 12, 2, C] → [b_, 12, 12, 4C]
            let h2 = h_patches / 2; // 12

            // Global pooling
            let glb_pooled = self.hd_pool(&glb_feats, 1, h_patches, c_dim as i32);

            // Sub-image pooling (matches Python's _reshape_and_concatenate)
            let sub_pooled = if b_ > 0 {
                let sub_reshaped =
                    mlxcel_core::reshape(&sub_feats, &[b_ as i32, h2, 2, h2, 2, c_dim as i32]);
                let sub_transposed =
                    mlxcel_core::transpose_axes(&sub_reshaped, &[0, 1, 3, 2, 4, 5]);
                let sub_pooled_spatial = mlxcel_core::reshape(
                    &sub_transposed,
                    &[
                        1,
                        (h_tiles as i32) * h2,
                        (w_tiles as i32) * h2,
                        4 * c_dim as i32,
                    ],
                );

                // Inject sub_GN: tile sub_GN to [1, h_tiles*12, 1, 4C], concat per row
                let sub_gn_tiled = mlxcel_core::broadcast_to(
                    &self.sub_gn,
                    &[1, (h_tiles as i32) * h2, 1, 4 * c_dim as i32],
                );
                let sub_with_gn = mlxcel_core::concatenate(&sub_pooled_spatial, &sub_gn_tiled, 2);
                let sub_h = (h_tiles as i32) * h2;
                let sub_w = (w_tiles as i32) * h2 + 1;
                mlxcel_core::reshape(&sub_with_gn, &[1, sub_h * sub_w, 4 * c_dim as i32])
            } else {
                mlxcel_core::reshape(
                    &mlxcel_core::zeros(&[1], mlxcel_core::dtype::FLOAT32),
                    &[1, 0, 4 * c_dim as i32],
                )
            };

            // Combine: [sub_img, glb_GN(1 token), glb_img]
            let glb_gn_token = mlxcel_core::reshape(&self.glb_gn, &[1, 1, 4 * c_dim as i32]);
            let glb_gn_token = mlxcel_core::astype(&glb_gn_token, embed_dtype);
            let sub_pooled = mlxcel_core::astype(&sub_pooled, embed_dtype);
            let glb_pooled = mlxcel_core::astype(&glb_pooled, embed_dtype);
            let combined = mlxcel_core::concatenate(&sub_pooled, &glb_gn_token, 1);
            let combined = mlxcel_core::concatenate(&combined, &glb_pooled, 1);

            // Project through MLP: [1, N, 4C] → [1, N, hidden_size]
            let projected = self.img_proj_linear1.forward(&combined);
            let projected = mlxcel_core::gelu_approx(&projected);
            let projected = self.img_proj_linear2.forward(&projected);

            // Count expected tokens
            let cnt =
                (h_tiles * w_tiles + 1) * self.processor.num_img_tokens + 1 + (h_tiles + 1) * 12;

            // Replace negative-token positions in embeddings
            if pos_idx < positions.len() {
                let (_batch_idx, start_idx) = positions[pos_idx];
                let proj_shape = mlxcel_core::array_shape(&projected);
                let actual_cnt = proj_shape[1] as usize;
                let use_cnt = cnt.min(actual_cnt);

                let proj_slice =
                    mlxcel_core::slice(&projected, &[0, 0, 0], &[1, use_cnt as i32, proj_shape[2]]);
                let proj_slice = mlxcel_core::astype(&proj_slice, embed_dtype);

                merged_embeds = mlxcel_core::slice_update(
                    &merged_embeds,
                    &proj_slice,
                    &[0, start_idx as i32, 0],
                    &[1, (start_idx + use_cnt) as i32, proj_shape[2]],
                );

                pos_idx += cnt;
            }

            tile_offset += num_this_image;
        }

        merge::InputEmbeddings {
            inputs_embeds: merged_embeds,
            attention_mask_4d: None,
        }
    }

    /// HD 2x2 spatial pooling with sub_GN injection per row
    /// Input: [N, H*H, C] where H=24 (patches per side)
    /// Output: [1, H/2 * (H/2+1), 4*C] (with GN separator per row)
    fn hd_pool(&self, features: &MlxArray, n: i32, h: i32, c: i32) -> UniquePtr<MlxArray> {
        let h2 = h / 2; // 12

        // Reshape [N, H*H, C] → [N, H/2, 2, H/2, 2, C]
        let reshaped = mlxcel_core::reshape(features, &[n, h, h, c]);
        let reshaped = mlxcel_core::reshape(&reshaped, &[n, h2, 2, h2, 2, c]);
        // Transpose → [N, H/2, H/2, 2, 2, C]
        let transposed = mlxcel_core::transpose_axes(&reshaped, &[0, 1, 3, 2, 4, 5]);
        // Reshape → [N, H/2, H/2, 4*C]
        let pooled = mlxcel_core::reshape(&transposed, &[n, h2, h2, 4 * c]);

        // Inject sub_GN: tile [1, 1, 1, 4C] → [N, H/2, 1, 4C]
        let sub_gn_tiled = mlxcel_core::broadcast_to(&self.sub_gn, &[n, h2, 1, 4 * c]);
        // Concat per row: [N, H/2, H/2+1, 4C]
        let with_gn = mlxcel_core::concatenate(&pooled, &sub_gn_tiled, 2);
        // Flatten: [1, H/2*(H/2+1), 4C]
        mlxcel_core::reshape(&with_gn, &[1, h2 * (h2 + 1), 4 * c])
    }
}

impl LanguageModel for Phi3VLModel {
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
        if let Some(embeds) = input_embeddings {
            // VLM prefill: use pre-computed embeddings
            let mut h = mlxcel_core::copy(embeds);
            for (i, layer) in self.text_model.layers.iter().enumerate() {
                h = layer.forward(&h, &mut caches[i], mask);
            }
            let h = self.text_model.norm.forward(&h);
            self.text_model.lm_head.forward(&h)
        } else {
            self.text_model.forward(input_ids, caches, mask)
        }
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.text_model.embed_tokens.forward(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        self.text_model.make_caches()
    }

    fn num_layers(&self) -> usize {
        self.text_model.layers.len()
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        mlxcel_core::generate::LanguageModel::eos_token_ids(&self.text_model)
    }
}
