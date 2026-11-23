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

//! Inkling 2 image-language wrapper: vision features scattered into the
//! normalized text embeddings at the `<|unused_200054|>` placeholders.
//!
//! Derived from mlx-vlm `Model.get_input_embeddings`
//! (`mlx_vlm/models/inkling2/inkling2.py`, MIT).

use image::DynamicImage;
use mlxcel_core::cache::{CachePool, SequenceId, SequenceStateLayout};
use mlxcel_core::generate::{BatchedDecodeKind, LanguageModel, ModelStateSnapshot};
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::Inkling2Model;

use super::encoders::inkling2_vit::Inkling2VisionTower;
use super::merge::{InputEmbeddings, merge_llava};
use super::processors::inkling2::{Inkling2ImageProcessor, Inkling2ProcessedImages};

pub struct Inkling2VlModel {
    pub text: Inkling2Model,
    /// `None` for a text-only export.
    pub vision_tower: Option<Inkling2VisionTower>,
    image_token_id: i32,
}

impl Inkling2VlModel {
    pub fn new(
        text: Inkling2Model,
        vision_tower: Option<Inkling2VisionTower>,
        image_token_id: i32,
    ) -> Self {
        Self {
            text,
            vision_tower,
            image_token_id,
        }
    }

    pub fn image_token_id(&self) -> i32 {
        self.image_token_id
    }

    pub fn supports_images(&self) -> bool {
        self.vision_tower.is_some()
    }

    pub fn preprocess_image_refs(
        &self,
        images: &[&DynamicImage],
    ) -> Result<Inkling2ProcessedImages, String> {
        Inkling2ImageProcessor.preprocess_refs(images)
    }

    /// Normalized text embeddings with each image's `rows * cols` features
    /// written over its placeholders, in prompt order.
    pub fn prepare_input_embeddings(
        &self,
        input_ids: &MlxArray,
        images: &Inkling2ProcessedImages,
    ) -> Result<InputEmbeddings, String> {
        let tower = self
            .vision_tower
            .as_ref()
            .ok_or_else(|| "This Inkling 2 checkpoint has no vision tower".to_string())?;
        let features = tower.forward(&images.pixel_values, &images.grids)?;
        let feature_count = mlxcel_core::array_shape(&features)[0];
        let token = mlxcel_core::from_slice_i32(&[self.image_token_id], &[1]);
        let matches = mlxcel_core::astype(
            &mlxcel_core::equal(input_ids, &token),
            mlxcel_core::dtype::INT32,
        );
        let placeholders = mlxcel_core::item_i32(&mlxcel_core::sum_all(&matches));
        if placeholders != feature_count {
            return Err(format!(
                "Inkling 2 prompt has {placeholders} image placeholders but the vision tower produced {feature_count} features"
            ));
        }
        let embeddings = self.text.normalized_input_embeddings(input_ids);
        let features = mlxcel_core::astype(&features, mlxcel_core::array_dtype(&embeddings));
        Ok(merge_llava(
            self.image_token_id,
            &features,
            &embeddings,
            input_ids,
        ))
    }
}

impl LanguageModel for Inkling2VlModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        LanguageModel::forward(&self.text, input_ids, caches, mask)
    }

    fn last_row_logits(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> Option<UniquePtr<MlxArray>> {
        match input_embeddings {
            // Wrapper embeddings are already normalized and image-merged.
            Some(embeddings) => Some(self.text.forward_prepared_embeddings_with_sequence_id(
                embeddings,
                seq_id,
                Some(last_pos),
            )),
            None => self
                .text
                .last_row_logits(input_ids, None, seq_id, caches, mask, last_pos),
        }
    }

    fn make_caches(&self) -> Vec<KVCache> {
        LanguageModel::make_caches(&self.text)
    }

    fn num_layers(&self) -> usize {
        LanguageModel::num_layers(&self.text)
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        LanguageModel::eos_token_ids(&self.text)
    }

    fn output_suppressed_token_ids(&self) -> Vec<i32> {
        let mut ids = LanguageModel::output_suppressed_token_ids(&self.text);
        if !ids.contains(&self.image_token_id) {
            ids.push(self.image_token_id);
        }
        ids
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_with_embeddings_and_sequence_id(
            input_ids,
            input_embeddings,
            None,
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
        match input_embeddings {
            Some(embeddings) => self
                .text
                .forward_prepared_embeddings_with_sequence_id(embeddings, seq_id, None),
            None => {
                LanguageModel::forward_with_sequence_id(&self.text, input_ids, seq_id, caches, mask)
            }
        }
    }

    /// The normalized embedding matrix, so that the runtime's embeddings path
    /// and [`Self::prepare_input_embeddings`] agree on what they hand back.
    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.text.normalized_input_embeddings(input_ids))
    }

    fn reset_runtime_state(&self) {
        LanguageModel::reset_runtime_state(&self.text);
    }

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        LanguageModel::prepare_sequence_state(&self.text, seq_id);
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        LanguageModel::release_sequence_state_by_id(&self.text, seq_id);
    }

    fn release_sequence_state(&self, caches: &mut [KVCache]) {
        LanguageModel::release_sequence_state(&self.text, caches);
    }

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        LanguageModel::sequence_state_layout(&self.text)
    }

    fn kv_cache_storage_modes(&self) -> Option<Vec<Option<mlxcel_core::cache::KVCacheMode>>> {
        LanguageModel::kv_cache_storage_modes(&self.text)
    }

    fn sync_sequence_storage(
        &self,
        seq_id: SequenceId,
        cache_pool: &mut CachePool,
    ) -> Result<(), String> {
        LanguageModel::sync_sequence_storage(&self.text, seq_id, cache_pool)
    }

    fn batched_decode_kind(&self) -> BatchedDecodeKind {
        self.text.batched_decode_kind()
    }

    fn forward_batched(
        &self,
        input_ids: &MlxArray,
        caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.forward_batched_with_ids(input_ids, None, caches, mask)
    }

    fn forward_batched_with_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text
            .forward_batched_with_ids(input_ids, seq_ids, caches, mask)
    }

    fn supports_batching(&self) -> bool {
        LanguageModel::supports_batching(&self.text)
    }

    fn supports_padded_prefill(&self) -> bool {
        LanguageModel::supports_padded_prefill(&self.text)
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        LanguageModel::forward_with_sequence_id(&self.text, input_ids, seq_id, caches, mask)
    }

    fn supports_snapshot_reuse(&self) -> bool {
        LanguageModel::supports_snapshot_reuse(&self.text)
    }

    fn snapshot_sequence_state(
        &self,
        seq_id: SequenceId,
        token_len: usize,
    ) -> Option<ModelStateSnapshot> {
        LanguageModel::snapshot_sequence_state(&self.text, seq_id, token_len)
    }

    fn restore_sequence_state(
        &self,
        seq_id: SequenceId,
        snapshot: &ModelStateSnapshot,
    ) -> Result<(), String> {
        LanguageModel::restore_sequence_state(&self.text, seq_id, snapshot)
    }

    fn trim_state(&self, seq: Option<SequenceId>, excess: i32) -> Result<(), String> {
        LanguageModel::trim_state(&self.text, seq, excess)
    }
}
