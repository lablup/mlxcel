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

//! Step-3.7 Vision-Language Model.
//!
//! Composes the `perception_encoder` ViT tower, the two-conv downsampler +
//! linear projector connector, the base+patch image processor, and the
//! Step-3.5 MoE text backbone (`Step3p5Model`).
//!
//! `input_embeddings()` runs the base pass (`728 -> conv1 -> 52x52 tokens ->
//! blocks -> downsamplers -> 13x13 -> projector -> 169x4096`) and the patch
//! pass (`504 -> 36x36 -> ... -> 9x9 -> 81x4096`), orders features
//! patches-first then base per image, and scatters them into
//! `image_token_index` placeholder positions with `merge::merge_llava`. The
//! total `<im_patch>` count MUST equal the total projected feature rows
//! (`169 + 81 * num_patches` per image); a mismatch is a hard error, never
//! silent truncation.
//!
//! The wrapper reuses `Step3p5Model`'s internal per-layer caches, so it
//! inherits `supports_batching() == false`.

use anyhow::Result;

use super::{connectors, encoders, merge, processors};
use crate::LanguageModel;
use crate::models::Step3p5Model;
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

/// Special-token ids for Step-3.7 image-placeholder expansion, resolved from
/// the tokenizer at load time.
#[derive(Debug, Clone, Copy)]
pub struct Step3p7TokenIds {
    /// `<im_patch>` scatter target (`image_token_index`, default 128001).
    pub image_token_index: i32,
    pub im_start: i32,
    pub im_end: i32,
    pub patch_start: i32,
    pub patch_end: i32,
    pub patch_newline: i32,
}

pub struct Step3p7VlModel {
    pub backbone: Step3p5Model,
    pub encoder: encoders::step3p7::Step3p7VisionEncoder,
    pub connector: connectors::step3p7::Step3p7Connector,
    pub processor: processors::step3p7::Step3p7Processor,
    pub tokens: Step3p7TokenIds,
    /// Base-image ViT grid side (`728 / patch_size` = 52).
    pub base_grid: i32,
    /// Patch-window ViT grid side (`504 / patch_size` = 36).
    pub patch_grid: i32,
    pub eos_token_ids: Vec<i32>,
}

impl Step3p7VlModel {
    /// Encode images (base + patch passes), order features patches-first then
    /// base per image, and scatter into `<im_patch>` positions.
    ///
    /// Hard-errors when the `<im_patch>` placeholder count does not equal the
    /// total projected feature rows.
    pub fn input_embeddings(
        &self,
        input_ids: &MlxArray,
        preprocessed: &processors::step3p7::Step3p7PreprocessOutput,
    ) -> Result<merge::InputEmbeddings> {
        // Base pass: (num_images, 3, 728, 728) -> (num_images, 169, text_hidden).
        let base_encoded =
            self.encoder
                .forward(preprocessed.base_pixel_values.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("step3p7 processor produced null base pixel values")
                })?);
        let base_features = self.connector.forward(
            base_encoded.as_ref().unwrap(),
            self.base_grid,
            self.base_grid,
        );

        // Patch pass (only when at least one image is windowed).
        let patch_features = match preprocessed.patch_pixel_values.as_ref() {
            Some(pv) => {
                let encoded = self.encoder.forward(pv.as_ref().unwrap());
                Some(self.connector.forward(
                    encoded.as_ref().unwrap(),
                    self.patch_grid,
                    self.patch_grid,
                ))
            }
            None => None,
        };

        // Order features PATCHES-FIRST then base, per image.
        let base_shape = mlxcel_core::array_shape(base_features.as_ref().unwrap());
        let base_tokens = base_shape[1];
        let hidden = base_shape[2];

        let mut blocks: Vec<UniquePtr<MlxArray>> = Vec::new();
        let mut patch_offset = 0i32;
        for (i, layout) in preprocessed.layouts.iter().enumerate() {
            if layout.num_patches > 0 {
                let patch_features = patch_features.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("step3p7 layout expects patches but none were produced")
                })?;
                let pshape = mlxcel_core::array_shape(patch_features.as_ref().unwrap());
                let n = layout.num_patches as i32;
                let per_patch = pshape[1];
                let slice = mlxcel_core::slice(
                    patch_features.as_ref().unwrap(),
                    &[patch_offset, 0, 0],
                    &[patch_offset + n, per_patch, hidden],
                );
                blocks.push(mlxcel_core::reshape(&slice, &[n * per_patch, hidden]));
                patch_offset += n;
            }

            let base_slice = mlxcel_core::slice(
                base_features.as_ref().unwrap(),
                &[i as i32, 0, 0],
                &[i as i32 + 1, base_tokens, hidden],
            );
            blocks.push(mlxcel_core::reshape(&base_slice, &[base_tokens, hidden]));
        }

        let features = concat_rows(blocks);
        let feat_rows = mlxcel_core::array_shape(features.as_ref().unwrap())[0];

        // Hard-error on placeholder/feature count mismatch (never truncate).
        let placeholder_count = count_image_tokens(input_ids, self.tokens.image_token_index);
        if placeholder_count != feat_rows {
            return Err(anyhow::anyhow!(
                "step3p7 image token/feature mismatch: {} <im_patch> placeholders vs {} projected feature rows (169 base + 81 per patch per image)",
                placeholder_count,
                feat_rows
            ));
        }

        let inputs_embeds = self.backbone.get_embed_tokens(input_ids);
        Ok(merge::merge_llava(
            self.tokens.image_token_index,
            features.as_ref().unwrap(),
            inputs_embeds.as_ref().unwrap(),
            input_ids,
        ))
    }
}

/// Concatenate a list of `(rows_i, hidden)` blocks along axis 0.
fn concat_rows(mut blocks: Vec<UniquePtr<MlxArray>>) -> UniquePtr<MlxArray> {
    let mut result = blocks.remove(0);
    for block in &blocks {
        result = mlxcel_core::concatenate(result.as_ref().unwrap(), block.as_ref().unwrap(), 0);
    }
    result
}

/// Count `<im_patch>` placeholder positions in `input_ids` via a device sum.
fn count_image_tokens(input_ids: &MlxArray, image_token_index: i32) -> i32 {
    let target = mlxcel_core::full_f32(&[1], image_token_index as f32, mlxcel_core::dtype::INT32);
    let target = mlxcel_core::astype(&target, mlxcel_core::dtype::INT32);
    let is_image = mlxcel_core::equal(input_ids, &target);
    let is_image = mlxcel_core::astype(&is_image, mlxcel_core::dtype::INT32);
    let count = mlxcel_core::sum_all(&is_image);
    mlxcel_core::eval(&count);
    mlxcel_core::item_i32(&count)
}

impl LanguageModel for Step3p7VlModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        // Step3p5Model uses internal per-layer caches and ignores the pool.
        LanguageModel::forward(&self.backbone, input_ids, caches, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        _caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.backbone
            .forward_with_embeddings_internal(input_ids, input_embeddings, mask)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.backbone.get_embed_tokens(input_ids))
    }

    fn make_caches(&self) -> Vec<KVCache> {
        LanguageModel::make_caches(&self.backbone)
    }

    fn num_layers(&self) -> usize {
        LanguageModel::num_layers(&self.backbone)
    }

    fn supports_batching(&self) -> bool {
        false
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        self.eos_token_ids.clone()
    }
}
