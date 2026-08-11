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

//! In-process Muse Glimmer VLM model object.
//!
//! This owns the text decoder, vision tower, post-tower fusion, and host image
//! processor as one canonical checkpoint object. Request-time placeholder
//! expansion and feature scatter are handled by the shared multimodal runtime.

use image::DynamicImage;
use mlxcel_core::cache::{CachePool, SequenceId, SequenceStateLayout};
use mlxcel_core::generate::{DecodeBatchContext, LanguageModel, ModelStateSnapshot};
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::muse_glimmer_config::{
    DEFAULT_IMAGE_TOKEN_ID, DEFAULT_PAD_TOKEN_ID, DEFAULT_VIDEO_TOKEN_ID,
};
use crate::models::{MuseGlimmerConfig, MuseGlimmerTextWrapper};

use super::encoders::muse_glimmer::MuseGlimmerVisionTower;
use super::encoders::muse_glimmer_fusion::MuseGlimmerVisionFusion;
use super::processors::muse_glimmer::MuseGlimmerImageProcessor;

pub struct MuseGlimmerVlmModel {
    pub text: MuseGlimmerTextWrapper,
    pub vision_tower: MuseGlimmerVisionTower,
    pub vision_fusion: MuseGlimmerVisionFusion,
    pub image_processor: MuseGlimmerImageProcessor,
    image_token_id: i32,
    video_token_id: i32,
    pad_token_id: i32,
}

impl MuseGlimmerVlmModel {
    pub fn new(
        text: MuseGlimmerTextWrapper,
        vision_tower: MuseGlimmerVisionTower,
        vision_fusion: MuseGlimmerVisionFusion,
        image_processor: MuseGlimmerImageProcessor,
        config: &MuseGlimmerConfig,
    ) -> Result<Self, String> {
        if config.vision_config.patch_temporal != 2 {
            return Err(format!(
                "Muse Glimmer VLM supports static image duplication only; got patch_temporal {}",
                config.vision_config.patch_temporal
            ));
        }
        if config.vision_config.merge_size != 2 {
            return Err(format!(
                "Muse Glimmer VLM supports 2x2 visual token merge only; got merge_size {}",
                config.vision_config.merge_size
            ));
        }

        Ok(Self {
            text,
            vision_tower,
            vision_fusion,
            image_processor,
            image_token_id: config.image_token_id.unwrap_or(DEFAULT_IMAGE_TOKEN_ID),
            video_token_id: config.video_token_id.unwrap_or(DEFAULT_VIDEO_TOKEN_ID),
            pad_token_id: DEFAULT_PAD_TOKEN_ID,
        })
    }

    pub fn text_embeddings(&self, input_ids: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        LanguageModel::embed_tokens(&self.text, input_ids)
            .ok_or_else(|| "Muse Glimmer text decoder does not expose token embeddings".to_string())
    }

    pub fn preprocess_images(
        &self,
        images: &[DynamicImage],
    ) -> (UniquePtr<MlxArray>, Vec<(i32, i32, i32)>) {
        self.image_processor.preprocess_with_grid(images)
    }

    pub fn encode_and_fuse_images(
        &self,
        pixel_values: &MlxArray,
        image_grid_thw: &[(i32, i32, i32)],
    ) -> Result<UniquePtr<MlxArray>, String> {
        let tower_features = self.vision_tower.forward(pixel_values, image_grid_thw)?;
        let tower_features = tower_features
            .as_ref()
            .ok_or_else(|| "Muse Glimmer vision tower produced a null output".to_string())?;
        self.vision_fusion.forward(tower_features, image_grid_thw)
    }

    pub fn image_token_id(&self) -> i32 {
        self.image_token_id
    }

    pub fn video_token_id(&self) -> i32 {
        self.video_token_id
    }

    pub fn pad_token_id(&self) -> i32 {
        self.pad_token_id
    }

    pub fn reject_video_inputs(&self) -> Result<(), String> {
        Err(format!(
            "Muse Glimmer VLM does not support video inputs yet; token id {} is reserved only for fail-closed prompt validation",
            self.video_token_id
        ))
    }

    pub fn text_sequence_state_layout(&self) -> SequenceStateLayout {
        LanguageModel::sequence_state_layout(&self.text)
    }

    pub fn prepare_text_sequence_state(&self, seq_id: SequenceId) {
        LanguageModel::prepare_sequence_state(&self.text, seq_id);
    }

    pub fn release_text_sequence_state(&self, seq_id: SequenceId) {
        LanguageModel::release_sequence_state_by_id(&self.text, seq_id);
    }

    pub fn reset_text_runtime_state(&self) {
        LanguageModel::reset_runtime_state(&self.text);
    }
}

impl LanguageModel for MuseGlimmerVlmModel {
    fn num_layers(&self) -> usize {
        LanguageModel::num_layers(&self.text)
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        LanguageModel::eos_token_ids(&self.text)
    }

    fn output_suppressed_token_ids(&self) -> Vec<i32> {
        LanguageModel::output_suppressed_token_ids(&self.text)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        LanguageModel::make_caches(&self.text)
    }

    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        LanguageModel::forward(&self.text, input_ids, caches, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        LanguageModel::forward_with_embeddings(
            &self.text,
            input_ids,
            input_embeddings,
            caches,
            mask,
        )
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        LanguageModel::embed_tokens(&self.text, input_ids)
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

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        LanguageModel::sequence_state_layout(&self.text)
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

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        LanguageModel::forward_with_embeddings_and_sequence_id(
            &self.text,
            input_ids,
            input_embeddings,
            seq_id,
            caches,
            mask,
        )
    }

    fn release_sequence_state(&self, caches: &mut [KVCache]) {
        LanguageModel::release_sequence_state(&self.text, caches);
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

    fn sync_sequence_storage(
        &self,
        seq_id: SequenceId,
        cache_pool: &mut CachePool,
    ) -> Result<(), String> {
        LanguageModel::sync_sequence_storage(&self.text, seq_id, cache_pool)
    }

    fn forward_batched_with_context(
        &self,
        input_ids: &MlxArray,
        batch_caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
        context: Option<&DecodeBatchContext>,
    ) -> UniquePtr<MlxArray> {
        LanguageModel::forward_batched_with_context(
            &self.text,
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
            &self.text,
            input_ids,
            seq_ids,
            batch_caches,
            mask,
            context,
        )
    }
}
