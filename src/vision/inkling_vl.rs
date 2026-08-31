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

//! Inkling image-language model wrapper.

use image::DynamicImage;
use mlxcel_core::cache::{SequenceId, SequenceStateLayout};
use mlxcel_core::generate::{LanguageModel, ModelStateSnapshot};
use mlxcel_core::layers::KVCache;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::audio::inkling_processor::InklingProcessorConfig;
use crate::models::InklingModel;

use super::encoders::inkling_hmlp::InklingHmlpEncoder;
use super::merge::{InputEmbeddings, merge_llava};
use super::processors::inkling::{InklingImageProcessor, InklingProcessedImages};

pub struct InklingVlModel {
    pub text: InklingModel,
    pub vision_tower: InklingHmlpEncoder,
    pub image_processor: InklingImageProcessor,
    pub audio_processor: InklingProcessorConfig,
    image_token_id: i32,
}

impl InklingVlModel {
    pub fn new(
        text: InklingModel,
        vision_tower: InklingHmlpEncoder,
        image_processor: InklingImageProcessor,
        audio_processor: InklingProcessorConfig,
        image_token_id: i32,
    ) -> Self {
        Self {
            text,
            vision_tower,
            image_processor,
            audio_processor,
            image_token_id,
        }
    }

    pub fn image_token_id(&self) -> i32 {
        self.image_token_id
    }

    #[must_use]
    pub fn supports_audio(&self) -> bool {
        self.text.supports_audio()
    }

    #[must_use]
    pub fn audio_token_id(&self) -> i32 {
        self.text.audio_token_id()
    }

    pub fn preprocess_images(
        &self,
        images: &[DynamicImage],
    ) -> Result<InklingProcessedImages, String> {
        self.image_processor.preprocess_with_counts(images)
    }

    pub fn preprocess_image_refs(
        &self,
        images: &[&DynamicImage],
    ) -> Result<InklingProcessedImages, String> {
        self.image_processor.preprocess_refs(images)
    }

    pub fn prepare_input_embeddings(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
    ) -> Result<InputEmbeddings, String> {
        let features = self.vision_tower.forward(pixel_values)?;
        let feature_shape = mlxcel_core::array_shape(&features);
        let feature_count = usize::try_from(feature_shape[0])
            .map_err(|_| "Inkling vision feature count is negative".to_string())?;
        let token = mlxcel_core::from_slice_i32(&[self.image_token_id], &[1]);
        let matches = mlxcel_core::equal(input_ids, &token);
        let matches = mlxcel_core::astype(&matches, mlxcel_core::dtype::INT32);
        let placeholder_count = mlxcel_core::item_i32(&mlxcel_core::sum_all(&matches));
        let placeholder_count = usize::try_from(placeholder_count)
            .map_err(|_| "Inkling image placeholder count is negative".to_string())?;
        if placeholder_count != feature_count {
            return Err(format!(
                "Inkling image placeholder count {placeholder_count} does not match {feature_count} HMLP features"
            ));
        }
        let embeddings = self.text.normalized_input_embeddings(input_ids)?;
        let embedding_shape = mlxcel_core::array_shape(&embeddings);
        if feature_shape.len() != 2
            || embedding_shape.len() != 3
            || feature_shape[1] != embedding_shape[2]
        {
            return Err(format!(
                "Inkling vision/text embedding shapes are incompatible: {feature_shape:?} vs {embedding_shape:?}"
            ));
        }
        Ok(merge_llava(
            self.image_token_id,
            &features,
            &embeddings,
            input_ids,
        ))
    }

    /// Prepare normalized text embeddings, scatter image rows first, and
    /// scatter compact valid dMel rows second.
    pub fn prepare_input_embeddings_with_audio(
        &self,
        input_ids: &MlxArray,
        pixel_values: Option<&MlxArray>,
        audio_input_ids: &MlxArray,
    ) -> Result<InputEmbeddings, String> {
        let image_embeddings = match pixel_values {
            Some(pixel_values) => self.prepare_input_embeddings(input_ids, pixel_values)?,
            None => InputEmbeddings {
                inputs_embeds: self.text.normalized_input_embeddings(input_ids)?,
                attention_mask_4d: None,
            },
        };
        self.text.merge_audio_embeddings(
            input_ids,
            &image_embeddings.inputs_embeds,
            audio_input_ids,
        )
    }

    /// Prepare Inkling video embeddings after replacing the second temporal
    /// plane of the final video-tile rows with adjacent-frame pixels.
    pub fn prepare_input_embeddings_with_video_slot1(
        &self,
        input_ids: &MlxArray,
        pixel_values: &MlxArray,
        video_slot1: &MlxArray,
    ) -> Result<InputEmbeddings, String> {
        let pixel_values = splice_video_slot1(pixel_values, video_slot1)?;
        self.prepare_input_embeddings(input_ids, &pixel_values)
    }
}

/// Replace temporal slot 1 only for the suffix that belongs to video frames.
///
/// `pixel_values` contains companion still-image tiles followed by the first
/// frame of every selected video pair. `video_slot1` contains the matching
/// second-frame tiles without a temporal axis. Keeping the prefix untouched is
/// essential: companion still images continue to use duplicated temporal
/// planes while only video tiles encode motion.
pub fn splice_video_slot1(
    pixel_values: &MlxArray,
    video_slot1: &MlxArray,
) -> Result<UniquePtr<MlxArray>, String> {
    let pixel_shape = mlxcel_core::array_shape(pixel_values);
    if pixel_shape.len() != 5 || pixel_shape[1..] != [2, 40, 40, 3] {
        return Err(format!(
            "Inkling video pixel values must have shape [N, 2, 40, 40, 3], got {pixel_shape:?}"
        ));
    }
    let slot_shape = mlxcel_core::array_shape(video_slot1);
    if slot_shape.len() != 4 || slot_shape[1..] != [40, 40, 3] {
        return Err(format!(
            "Inkling video slot-1 values must have shape [M, 40, 40, 3], got {slot_shape:?}"
        ));
    }
    let total_tiles = usize::try_from(pixel_shape[0])
        .map_err(|_| "Inkling video tile count is negative".to_string())?;
    let video_tiles = usize::try_from(slot_shape[0])
        .map_err(|_| "Inkling video slot-1 tile count is negative".to_string())?;
    if video_tiles == 0 || video_tiles > total_tiles {
        return Err(format!(
            "Inkling video slot-1 tile count {video_tiles} must be in 1..={total_tiles}"
        ));
    }

    let suffix_start = i32::try_from(total_tiles - video_tiles)
        .map_err(|_| "Inkling video suffix offset exceeds the MLX i32 limit".to_string())?;
    let total_tiles = i32::try_from(total_tiles)
        .map_err(|_| "Inkling video tile count exceeds the MLX i32 limit".to_string())?;
    let suffix_slot0 = mlxcel_core::slice(
        pixel_values,
        &[suffix_start, 0, 0, 0, 0],
        &[total_tiles, 1, 40, 40, 3],
    );
    let video_slot1 = mlxcel_core::astype(video_slot1, mlxcel_core::array_dtype(pixel_values));
    let video_slot1 = mlxcel_core::expand_dims(&video_slot1, 1);
    let suffix = mlxcel_core::concatenate(&suffix_slot0, &video_slot1, 1);
    if suffix_start == 0 {
        return Ok(suffix);
    }

    let prefix = mlxcel_core::slice(
        pixel_values,
        &[0, 0, 0, 0, 0],
        &[suffix_start, 2, 40, 40, 3],
    );
    Ok(mlxcel_core::concatenate(&prefix, &suffix, 0))
}

impl LanguageModel for InklingVlModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        LanguageModel::forward(&self.text, input_ids, caches, mask)
    }

    fn forward_last_logits(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        LanguageModel::forward_last_logits(&self.text, input_ids, caches, mask, last_pos)
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
        let audio_token_id = self.audio_token_id();
        if self.supports_audio() && !ids.contains(&audio_token_id) {
            ids.push(audio_token_id);
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
            || LanguageModel::forward(&self.text, input_ids, caches, mask),
            |embeddings| self.text.forward_prepared_embeddings(embeddings),
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

    fn release_sequence_state(&self, caches: &mut [KVCache]) {
        LanguageModel::release_sequence_state(&self.text, caches);
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

    fn forward_last_logits_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
        last_pos: usize,
    ) -> UniquePtr<MlxArray> {
        LanguageModel::forward_last_logits_with_sequence_id(
            &self.text, input_ids, seq_id, caches, mask, last_pos,
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
            || LanguageModel::forward_with_sequence_id(&self.text, input_ids, seq_id, caches, mask),
            |embeddings| {
                self.text
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
        input_embeddings.map_or_else(
            || {
                LanguageModel::forward_last_logits_with_sequence_id(
                    &self.text, input_ids, seq_id, caches, mask, last_pos,
                )
            },
            |embeddings| {
                self.text
                    .forward_last_prepared_embeddings_with_sequence_id(embeddings, seq_id, last_pos)
            },
        )
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

    fn trim_internal_caches(&self, excess: i32) {
        LanguageModel::trim_internal_caches(&self.text, excess);
    }
}

#[cfg(test)]
mod tests {
    use mlxcel_core::cache::SequenceId;
    use mlxcel_core::generate::LanguageModel;
    use mlxcel_core::utils::array_to_vec_f32;
    use mlxcel_core::weights::WeightMap;
    use mlxcel_core::{MlxArray, dtype};

    use super::*;
    use crate::vision::encoders::inkling_hmlp::{InklingVisionConfig, layer_plan};

    fn tiny_vlm() -> InklingVlModel {
        tiny_vlm_with_text(crate::models::inkling::tiny_model())
    }

    fn tiny_audio_vlm() -> InklingVlModel {
        tiny_vlm_with_text(crate::models::inkling::tiny_audio_model())
    }

    fn tiny_vlm_with_text(text: InklingModel) -> InklingVlModel {
        let vision_config: InklingVisionConfig = serde_json::from_value(serde_json::json!({
            "model_type": "inkling_vision",
            "patch_size": 40,
            "temporal_patch_size": 2,
            "num_channels": 3,
            "n_layers": 1,
            "text_hidden_size": 4,
            "rms_norm_eps": 1e-6
        }))
        .unwrap();
        let plan = layer_plan(&vision_config).unwrap().remove(0);
        let mut weights = WeightMap::new();
        weights.insert(
            "vision_tower.encoder_layers.0.projection.weight".into(),
            mlxcel_core::zeros(
                &[plan.output_dim as i32, plan.input_dim as i32],
                dtype::FLOAT32,
            ),
        );
        weights.insert(
            "vision_tower.final_norm.weight".into(),
            mlxcel_core::ones(&[vision_config.text_hidden_size as i32], dtype::FLOAT32),
        );
        let vision_tower =
            InklingHmlpEncoder::from_weights(&weights, &vision_config, 64, 4).unwrap();
        InklingVlModel::new(
            text,
            vision_tower,
            InklingImageProcessor::default(),
            InklingProcessorConfig::default(),
            7,
        )
    }

    fn arrays_equal(left: &MlxArray, right: &MlxArray) -> bool {
        let equal = mlxcel_core::allclose(left, right, 0.0, 0.0);
        mlxcel_core::eval(&equal);
        mlxcel_core::item_bool(&equal)
    }

    #[test]
    fn image_prefill_keeps_hmlp_prepared_embeddings_on_classic_wrapper_path() {
        let vlm = tiny_vlm();
        let input_ids = mlxcel_core::from_slice_i32(&[1, 7, 2], &[1, 3]);
        let pixel_values = mlxcel_core::zeros(&[1, 2, 40, 40, 3], dtype::FLOAT32);
        let prepared = vlm
            .prepare_input_embeddings(&input_ids, &pixel_values)
            .unwrap();
        assert_eq!(mlxcel_core::array_shape(&prepared.inputs_embeds), [1, 3, 4]);
        let last_pos = mlxcel_core::array_shape(&prepared.inputs_embeds)[1] as usize - 1;

        let mut wrapper_caches = Vec::new();
        let wrapper_logits = LanguageModel::forward_last_logits_with_embeddings_and_sequence_id(
            &vlm,
            &input_ids,
            Some(&prepared.inputs_embeds),
            Some(SequenceId::from_raw(901)),
            &mut wrapper_caches,
            None,
            last_pos,
        );
        let direct_logits = vlm.text.forward_last_prepared_embeddings_with_sequence_id(
            &prepared.inputs_embeds,
            Some(SequenceId::from_raw(902)),
            last_pos,
        );

        assert!(
            arrays_equal(&wrapper_logits, &direct_logits),
            "InklingVLM image prefill must feed HMLP-merged prepared embeddings directly \
             to vlm.text without replacing them with token embeddings or normalizing twice"
        );
    }

    #[test]
    fn mixed_prefill_scatter_order_is_normalized_text_then_image_then_audio() {
        let vlm = tiny_audio_vlm();
        let input_ids = mlxcel_core::from_slice_i32(&[1, 7, 6, 2], &[1, 4]);
        let pixel_values = mlxcel_core::zeros(&[1, 2, 40, 40, 3], dtype::FLOAT32);
        let audio_ids = mlxcel_core::zeros(&[1, 80], dtype::INT32);

        let image_first = vlm
            .prepare_input_embeddings(&input_ids, &pixel_values)
            .unwrap();
        let expected = vlm
            .text
            .merge_audio_embeddings(&input_ids, &image_first.inputs_embeds, &audio_ids)
            .unwrap();
        let actual = vlm
            .prepare_input_embeddings_with_audio(&input_ids, Some(&pixel_values), &audio_ids)
            .unwrap();

        assert!(arrays_equal(&actual.inputs_embeds, &expected.inputs_embeds));
        assert!(vlm.supports_audio());
        assert_eq!(vlm.audio_token_id(), 6);
        assert!(vlm.output_suppressed_token_ids().contains(&6));
    }

    #[test]
    fn vlm_mtp_adapter_is_a_text_backbone_target() {
        fn assert_mtp_target<T: mlxcel_core::speculative::mtp::target::MtpTarget>() {}
        let _ =
            assert_mtp_target::<crate::models::inkling_mtp_target::InklingVLMtpTargetAdapter<'_>>;
    }

    #[test]
    fn slot1_overwrite_touches_only_the_tail() {
        let pixels = mlxcel_core::zeros(&[5, 2, 40, 40, 3], mlxcel_core::dtype::FLOAT32);
        let seconds = mlxcel_core::ones(&[2, 40, 40, 3], mlxcel_core::dtype::FLOAT32);
        let spliced = splice_video_slot1(&pixels, &seconds).unwrap();
        mlxcel_core::eval(&spliced);
        let values = array_to_vec_f32(&spliced);
        let plane = 40 * 40 * 3;
        for tile in 0..5 {
            let slot0 = &values[tile * 2 * plane..tile * 2 * plane + plane];
            let slot1 = &values[tile * 2 * plane + plane..(tile + 1) * 2 * plane];
            assert!(slot0.iter().all(|&value| value == 0.0));
            let expected = if tile < 3 { 0.0 } else { 1.0 };
            assert!(slot1.iter().all(|&value| value == expected));
        }
    }

    #[test]
    fn video_slot1_splice_rejects_non_suffix_cardinality() {
        let pixels = mlxcel_core::zeros(&[1, 2, 40, 40, 3], mlxcel_core::dtype::FLOAT32);
        let seconds = mlxcel_core::zeros(&[2, 40, 40, 3], mlxcel_core::dtype::FLOAT32);
        let error = splice_video_slot1(&pixels, &seconds).err().unwrap();
        assert!(error.contains("must be in 1..=1"));
    }
}
