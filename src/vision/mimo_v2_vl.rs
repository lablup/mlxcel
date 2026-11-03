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

//! MiMo-V2 vision-language wrapper.
//!
//! The text tower is the `mimo_v2_flash` decoder from issue #842, unchanged.
//! This wrapper adds the `mimovl` tower, the Qwen2.5-VL-compatible image and
//! video prompt population, and the `LanguageModel` forwarding the scheduler
//! needs. It deliberately implements [`QwenVlRuntime`]: MiMo-V2 uses the
//! Qwen2.5-VL processor, so its image/video path is the shared Qwen-VL one
//! (`<|vision_start|><|image_pad|><|vision_end|>`, dynamic resolution,
//! `grid_thw`), and routing it through the same trait keeps the CLI and server
//! media preparation identical to Qwen2.5-VL rather than a private copy.
//!
//! Audio is out of scope for this issue: the checkpoint's 20-channel codec
//! tokenizer has no mlxcel port, so an audio request must fail with a clear
//! message instead of silently dropping the clip.

use image::DynamicImage;
use mlxcel_core::cache::{CachePool, SequenceId, SequenceStateLayout};
use mlxcel_core::generate::{BatchedDecodeKind, LanguageModel, ModelStateSnapshot};
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::models::MiMoV2FlashModel;
use crate::multimodal::qwen_vl::{QwenVlMRopeSnapshot, QwenVlRuntime, QwenVlmPromptInfo};
use crate::vision::processors::qwen2_vl::Qwen2VLProcessor;

use super::encoders::mimovl::MimovlVisionEncoder;
use super::merge::{InputEmbeddings, merge_llava_any};

pub struct MimoV2VlModel {
    pub text_model: MiMoV2FlashModel,
    /// `None` for `skip_vision: true` or a vision-stripped checkpoint.
    pub vision_encoder: Option<MimovlVisionEncoder>,
    /// Model directory of a text-only load, used in the media rejection error.
    pub text_only_path: Option<String>,
    pub processor: Qwen2VLProcessor,
    pub image_token_id: i32,
    pub video_token_id: i32,
    pub vision_start_token_id: i32,
    pub vision_end_token_id: i32,
    pub audio_token_id: i32,
    pub spatial_merge_size: usize,
}

impl MimoV2VlModel {
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> InputEmbeddings {
        let embeddings = self.text_model.input_embeddings(input_ids);
        let Some(encoder) = self.vision_encoder.as_ref() else {
            panic!("{}", crate::multimodal::qwen_vl::NO_VISION_TOWER_GUARD);
        };
        let embed_dtype = mlxcel_core::array_dtype(&embeddings);
        let pixels = mlxcel_core::astype(pixel_values, embed_dtype);
        // The tower's patch embedding dtype is the one it was loaded with;
        // upstream casts the input the same way.
        let features = encoder
            .forward_with_grid(&pixels, grid_thw)
            .unwrap_or_else(|error| panic!("MiMo-V2 vision tower failed: {error}"))
            .hidden_states;

        let feature_shape = mlxcel_core::array_shape(&features);
        let feature_rows = feature_shape[0];
        let placeholders =
            count_visual_placeholders(input_ids, &[self.image_token_id, self.video_token_id]);
        if feature_rows != placeholders {
            panic!(
                "MiMo-V2 prompt has {placeholders} image/video placeholders but the vision tower \
                 produced {feature_rows} rows"
            );
        }
        let embedding_shape = mlxcel_core::array_shape(&embeddings);
        if feature_shape.len() != 2
            || embedding_shape.len() != 3
            || feature_shape[1] != embedding_shape[2]
        {
            panic!(
                "MiMo-V2 vision/text embedding shapes are incompatible: {feature_shape:?} vs \
                 {embedding_shape:?}"
            );
        }
        merge_llava_any(
            &[self.image_token_id, self.video_token_id],
            &features,
            &embeddings,
            input_ids,
        )
    }

    #[must_use]
    pub fn supports_audio(&self) -> bool {
        false
    }

    #[must_use]
    pub fn audio_token_id(&self) -> i32 {
        self.audio_token_id
    }
}

fn count_visual_placeholders(input_ids: &MlxArray, token_ids: &[i32]) -> i32 {
    let mut total = 0i32;
    for &token_id in token_ids {
        let token = mlxcel_core::from_slice_i32(&[token_id], &[1]);
        let matches = mlxcel_core::astype(
            &mlxcel_core::equal(input_ids, &token),
            mlxcel_core::dtype::INT32,
        );
        total += mlxcel_core::item_i32(&mlxcel_core::sum_all(&matches));
    }
    total
}

impl LanguageModel for MimoV2VlModel {
    fn batched_decode_kind(&self) -> BatchedDecodeKind {
        self.text_model.batched_decode_kind()
    }

    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        LanguageModel::forward(&self.text_model, input_ids, caches, mask)
    }

    fn forward_last_logits(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        LanguageModel::forward_last_logits(&self.text_model, input_ids, caches, mask, last_pos)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        LanguageModel::make_caches(&self.text_model)
    }

    fn num_layers(&self) -> usize {
        LanguageModel::num_layers(&self.text_model)
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        LanguageModel::eos_token_ids(&self.text_model)
    }

    fn output_suppressed_token_ids(&self) -> Vec<i32> {
        let mut ids = LanguageModel::output_suppressed_token_ids(&self.text_model);
        for id in [
            self.image_token_id,
            self.video_token_id,
            self.vision_start_token_id,
            self.vision_end_token_id,
            self.audio_token_id,
        ] {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        ids
    }

    fn supports_chunked_prefill(&self) -> bool {
        false
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        input_embeddings.map_or_else(
            || LanguageModel::forward(&self.text_model, input_ids, caches, mask),
            |embeddings| {
                self.text_model
                    .forward_prepared_embeddings_with_sequence_id(embeddings, None)
            },
        )
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.text_model.input_embeddings(input_ids))
    }

    fn reset_runtime_state(&self) {
        LanguageModel::reset_runtime_state(&self.text_model);
    }

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        LanguageModel::prepare_sequence_state(&self.text_model, seq_id);
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        LanguageModel::release_sequence_state_by_id(&self.text_model, seq_id);
    }

    fn release_sequence_state(&self, caches: &mut [KVCache]) {
        LanguageModel::release_sequence_state(&self.text_model, caches);
    }

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        LanguageModel::sequence_state_layout(&self.text_model)
    }

    fn kv_cache_storage_modes(&self) -> Option<Vec<Option<mlxcel_core::cache::KVCacheMode>>> {
        LanguageModel::kv_cache_storage_modes(&self.text_model)
    }

    fn sync_sequence_storage(
        &self,
        seq_id: SequenceId,
        cache_pool: &mut CachePool,
    ) -> Result<(), String> {
        LanguageModel::sync_sequence_storage(&self.text_model, seq_id, cache_pool)
    }

    fn forward_batched(
        &self,
        input_ids: &MlxArray,
        caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model.forward_batched(input_ids, caches, mask)
    }

    fn forward_batched_with_ids(
        &self,
        input_ids: &MlxArray,
        seq_ids: Option<&[SequenceId]>,
        caches: &mut [&mut [KVCache]],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.text_model
            .forward_batched_with_ids(input_ids, seq_ids, caches, mask)
    }

    fn supports_batching(&self) -> bool {
        LanguageModel::supports_batching(&self.text_model)
    }

    fn supports_padded_prefill(&self) -> bool {
        LanguageModel::supports_padded_prefill(&self.text_model)
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        LanguageModel::forward_with_sequence_id(&self.text_model, input_ids, seq_id, caches, mask)
    }

    fn forward_last_logits_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        LanguageModel::forward_last_logits_with_sequence_id(
            &self.text_model,
            input_ids,
            seq_id,
            caches,
            mask,
            last_pos,
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
        input_embeddings.map_or_else(
            || {
                LanguageModel::forward_with_sequence_id(
                    &self.text_model,
                    input_ids,
                    seq_id,
                    caches,
                    mask,
                )
            },
            |embeddings| {
                self.text_model
                    .forward_prepared_embeddings_with_sequence_id(embeddings, seq_id)
            },
        )
    }

    fn forward_last_logits_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        match input_embeddings {
            Some(embeddings) => {
                let logits = self
                    .text_model
                    .forward_prepared_embeddings_with_sequence_id(embeddings, seq_id);
                slice_last_logits(&logits, last_pos)
            }
            None => LanguageModel::forward_last_logits_with_sequence_id(
                &self.text_model,
                input_ids,
                seq_id,
                caches,
                mask,
                last_pos,
            ),
        }
    }

    fn supports_snapshot_reuse(&self) -> bool {
        LanguageModel::supports_snapshot_reuse(&self.text_model)
    }

    fn snapshot_sequence_state(
        &self,
        seq_id: SequenceId,
        token_len: usize,
    ) -> Option<ModelStateSnapshot> {
        LanguageModel::snapshot_sequence_state(&self.text_model, seq_id, token_len)
    }

    fn restore_sequence_state(
        &self,
        seq_id: SequenceId,
        snapshot: &ModelStateSnapshot,
    ) -> Result<(), String> {
        LanguageModel::restore_sequence_state(&self.text_model, seq_id, snapshot)
    }

    fn trim_state(&self, seq: Option<SequenceId>, excess: i32) -> Result<(), String> {
        LanguageModel::trim_state(&self.text_model, seq, excess)
    }
}

fn slice_last_logits(logits: &MlxArray, last_pos: usize) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(logits);
    let position = i32::try_from(last_pos).unwrap_or(shape[1] - 1);
    if position < 0 || position >= shape[1] {
        return mlxcel_core::copy(logits);
    }
    mlxcel_core::slice(
        logits,
        &[0, position, 0],
        &[shape[0], position + 1, shape[2]],
    )
}

impl QwenVlRuntime for MimoV2VlModel {
    fn prompt_info(&self) -> QwenVlmPromptInfo<'_> {
        QwenVlmPromptInfo {
            processor: &self.processor,
            spatial_merge_size: self.spatial_merge_size,
            vision_start_token_id: self.vision_start_token_id,
            image_token_id: self.image_token_id,
            video_token_id: self.video_token_id,
        }
    }

    fn text_only_source(&self) -> Option<&str> {
        self.text_only_path.as_deref()
    }

    fn input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        grid_thw: &[(i32, i32, i32)],
    ) -> InputEmbeddings {
        self.get_input_embeddings(input_ids, pixel_values, grid_thw)
    }

    /// MiMo-V2 uses 1-D text RoPE: there is no MRoPE state to bind or move.
    fn clear_pending_media_state(&self) {}

    fn take_mrope_entry_for_sequence(&self, _seq_id: SequenceId) -> QwenVlMRopeSnapshot {
        QwenVlMRopeSnapshot::mrope_only(None)
    }
}

/// Preprocess images through the checkpoint's Qwen2.5-VL processor.
impl MimoV2VlModel {
    pub fn preprocess_images(
        &self,
        images: &[DynamicImage],
    ) -> Result<(UniquePtr<MlxArray>, Vec<(i32, i32, i32)>), String> {
        self.processor.preprocess_media_with_grid(
            &images
                .iter()
                .map(crate::vision::processors::qwen2_vl::Qwen2VLMediaInput::Image)
                .collect::<Vec<_>>(),
        )
    }
}
