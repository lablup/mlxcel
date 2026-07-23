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

//! Phi4MM vision-language model with HD (high-definition) transform.
//!
//! Phi4MM uses a dynamic HD transform: the input image is split into a global
//! thumbnail and multiple sub-crops, each processed independently through
//! SigLIP.  Vision features are spatially compressed via AvgPool2d, then
//! assembled with learnable separator tokens (glb_GN, sub_GN) before
//! projection through a two-layer MLP.
//!
//! Used by: Phi4MM VLM

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use crate::LanguageModel;
use crate::audio::phi4mm::{
    Phi4MMAudioBatch, Phi4MMAudioEncoder, Phi4MMAudioFeatureExtractor, Phi4MMAudioProjection,
};
use crate::multimodal::phi4_siglip_prompt::PHI4_SIGLIP_IMAGE_TOKEN_INDEX;
use crate::multimodal::phi4mm_prompt::PHI4MM_AUDIO_TOKEN_ID;
use crate::vision::merge::InputEmbeddings;
use crate::vision::{encoders, processors};
use mlxcel_core::cache::{SequenceId, SequenceStateLayout};
use mlxcel_core::layers::{KVCache, UnifiedLinear};
use mlxcel_core::{MlxArray, UniquePtr};

pub struct Phi4MMVLModel {
    pub text_model: crate::models::Phi4MMModel,
    pub vision_tower: encoders::phi4_siglip::Phi4SigLipVisionEncoder,
    pub mm_projector_linear1: UnifiedLinear,
    pub mm_projector_linear2: UnifiedLinear,
    pub processor: processors::phi4mm::Phi4MMProcessor,
    pub audio_encoder: Phi4MMAudioEncoder,
    pub audio_projection: Phi4MMAudioProjection,
    pub audio_extractor: Phi4MMAudioFeatureExtractor,
    pub select_layer: isize,
    pub eos_token_ids: Vec<i32>,
    /// Learnable global separator: [1, 1, vision_dim]
    pub glb_gn: UniquePtr<MlxArray>,
    /// Learnable sub-image row separator: [1, 1, 1, vision_dim]
    pub sub_gn: UniquePtr<MlxArray>,
    /// hd_transform_order: "sub_glb" or "glb_sub"
    pub hd_transform_order: String,
    pub request_modes: Phi4MMRequestModes,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Phi4MMInputMode {
    #[default]
    Language,
    Vision,
    Speech,
    VisionSpeech,
}

impl Phi4MMInputMode {
    fn adapter(self) -> Option<&'static str> {
        match self {
            Self::Language => None,
            Self::Speech => Some("speech"),
            Self::Vision | Self::VisionSpeech => Some("vision"),
        }
    }
}

#[derive(Default)]
pub struct Phi4MMRequestModes {
    embedding_modes: RefCell<HashMap<usize, Phi4MMInputMode>>,
    sequence_modes: RefCell<HashMap<SequenceId, Phi4MMInputMode>>,
    fallback: Cell<Phi4MMInputMode>,
}

impl Phi4MMRequestModes {
    fn register_embeddings(&self, embeddings: &MlxArray, mode: Phi4MMInputMode) {
        self.embedding_modes
            .borrow_mut()
            .insert(embeddings as *const MlxArray as usize, mode);
    }

    fn begin_prefill(
        &self,
        embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
    ) -> Phi4MMInputMode {
        let mode = embeddings
            .and_then(|value| {
                self.embedding_modes
                    .borrow_mut()
                    .remove(&(value as *const MlxArray as usize))
            })
            .unwrap_or(Phi4MMInputMode::Language);
        if let Some(seq_id) = seq_id {
            self.sequence_modes.borrow_mut().insert(seq_id, mode);
        } else {
            self.fallback.set(mode);
        }
        mode
    }

    fn decode_mode(&self, seq_id: Option<SequenceId>) -> Phi4MMInputMode {
        match seq_id {
            Some(id) => self
                .sequence_modes
                .borrow()
                .get(&id)
                .copied()
                .unwrap_or(Phi4MMInputMode::Language),
            None => self.fallback.get(),
        }
    }

    fn prepare(&self, seq_id: SequenceId) {
        self.sequence_modes
            .borrow_mut()
            .insert(seq_id, Phi4MMInputMode::Language);
    }

    fn release(&self, seq_id: SequenceId) {
        self.sequence_modes.borrow_mut().remove(&seq_id);
    }

    fn reset(&self) {
        // `CxxGenerator` resets model runtime state after the caller has
        // already built and registered multimodal embeddings. Keep those
        // one-shot pointer registrations until `begin_prefill` consumes
        // them; only active/fallback sequence state belongs to the previous
        // generation.
        self.sequence_modes.borrow_mut().clear();
        self.fallback.set(Phi4MMInputMode::Language);
    }
}

impl Phi4MMVLModel {
    pub fn get_input_embeddings(
        &self,
        input_ids: &MlxArray,
        processed_images: &[processors::phi4mm::Phi4MMImageInput],
    ) -> Result<InputEmbeddings, String> {
        self.merge_input_embeddings(input_ids, processed_images, &[], Phi4MMInputMode::Vision)
    }

    pub fn extract_audio(&self, audios: &[(Vec<f32>, u32)]) -> Result<Phi4MMAudioBatch, String> {
        self.audio_extractor.extract_batch(audios)
    }

    pub fn get_input_embeddings_with_audio(
        &self,
        input_ids: &MlxArray,
        processed_images: &[processors::phi4mm::Phi4MMImageInput],
        audio: &Phi4MMAudioBatch,
    ) -> Result<InputEmbeddings, String> {
        let vision_mode = !processed_images.is_empty();
        let mut projected = Vec::with_capacity(audio.clips.len());
        for (index, features) in audio.clips.iter().enumerate() {
            let encoded = self.audio_encoder.forward(features)?;
            let encoded_shape = mlxcel_core::array_shape(&encoded);
            let encoded_len = encoded_shape[1] as usize;
            let needed = audio.embed_sizes[index];
            if encoded_len < needed {
                return Err(format!(
                    "Phi4MM audio encoder returned {encoded_len} rows for clip {}, expected at least {needed}",
                    index + 1
                ));
            }
            let encoded =
                mlxcel_core::slice(&encoded, &[0, 0, 0], &[1, needed as i32, encoded_shape[2]]);
            projected.push(self.audio_projection.forward(&encoded, vision_mode));
        }
        let mode = if vision_mode {
            Phi4MMInputMode::VisionSpeech
        } else {
            Phi4MMInputMode::Speech
        };
        self.merge_input_embeddings(input_ids, processed_images, &projected, mode)
    }

    fn merge_input_embeddings(
        &self,
        input_ids: &MlxArray,
        processed_images: &[processors::phi4mm::Phi4MMImageInput],
        audio_features: &[UniquePtr<MlxArray>],
        mode: Phi4MMInputMode,
    ) -> Result<InputEmbeddings, String> {
        let ids_shape = mlxcel_core::array_shape(input_ids);
        let seq_len = ids_shape[1] as usize;
        let mut safe_tokens = Vec::with_capacity(seq_len);
        let mut token_values = Vec::with_capacity(seq_len);

        for token_idx in 0..seq_len {
            let token = mlxcel_core::slice(
                input_ids,
                &[0, token_idx as i32],
                &[1, token_idx as i32 + 1],
            );
            mlxcel_core::eval(&token);
            let value = mlxcel_core::item_i32(&token);
            token_values.push(value);
            safe_tokens.push(
                if value == PHI4_SIGLIP_IMAGE_TOKEN_INDEX || value == PHI4MM_AUDIO_TOKEN_ID {
                    0
                } else {
                    value
                },
            );
        }

        let safe_input_ids = mlxcel_core::from_slice_i32(&safe_tokens, &[1, seq_len as i32]);
        let inputs_embeds = self.text_model.get_embed_tokens(&safe_input_ids);
        let embed_dtype = mlxcel_core::array_dtype(&inputs_embeds);
        let hidden_size = mlxcel_core::array_shape(&inputs_embeds)[2];

        let vision_features: Vec<UniquePtr<MlxArray>> = processed_images
            .iter()
            .map(|processed| self.hd_transform(processed, embed_dtype))
            .collect();

        // Partition sentinel rows by encoder output length, not by text
        // separators. Adjacent numbered tags expand to adjacent equal IDs but
        // still represent distinct media inputs.
        let mut media_segments: Vec<(i32, usize, usize, usize)> = Vec::new();
        let mut token_index = 0usize;
        let mut image_index = 0usize;
        let mut audio_index = 0usize;
        while token_index < seq_len {
            let kind = token_values[token_index];
            if kind != PHI4_SIGLIP_IMAGE_TOKEN_INDEX && kind != PHI4MM_AUDIO_TOKEN_ID {
                token_index += 1;
                continue;
            }
            let (feature_index, feature_len) = if kind == PHI4_SIGLIP_IMAGE_TOKEN_INDEX {
                let features = vision_features.get(image_index).ok_or_else(|| {
                    "Phi4MM prompt has more image placeholder rows than image inputs".to_string()
                })?;
                let current = image_index;
                image_index += 1;
                (current, mlxcel_core::array_shape(features)[1] as usize)
            } else {
                let features = audio_features.get(audio_index).ok_or_else(|| {
                    "Phi4MM prompt has more audio placeholder rows than audio inputs".to_string()
                })?;
                let current = audio_index;
                audio_index += 1;
                (current, mlxcel_core::array_shape(features)[1] as usize)
            };
            if feature_len == 0 || token_index + feature_len > seq_len {
                return Err(format!(
                    "Phi4MM media encoder produced invalid row count {feature_len} at prompt position {token_index}"
                ));
            }
            let end = token_index + feature_len;
            if token_values[token_index..end]
                .iter()
                .any(|token| *token != kind)
            {
                return Err(format!(
                    "Phi4MM media placeholder at position {token_index} does not contain the encoder's {feature_len} contiguous rows"
                ));
            }
            media_segments.push((kind, token_index, end, feature_index));
            token_index = end;
        }
        if image_index != vision_features.len() || audio_index != audio_features.len() {
            return Err(format!(
                "Phi4MM placeholder cardinality mismatch: prompt consumed {image_index} image/{audio_index} audio inputs, received {}/{}",
                vision_features.len(),
                audio_features.len()
            ));
        }

        let mut segments: Vec<UniquePtr<MlxArray>> = Vec::new();
        let mut previous_end = 0usize;
        for (kind, start, end, feature_index) in media_segments {
            if start > previous_end {
                segments.push(mlxcel_core::slice(
                    &inputs_embeds,
                    &[0, previous_end as i32, 0],
                    &[1, start as i32, hidden_size],
                ));
            }
            let features = if kind == PHI4_SIGLIP_IMAGE_TOKEN_INDEX {
                &vision_features[feature_index]
            } else {
                &audio_features[feature_index]
            };
            segments.push(mlxcel_core::astype(features, embed_dtype));
            previous_end = end;
        }
        if previous_end < seq_len {
            segments.push(mlxcel_core::slice(
                &inputs_embeds,
                &[0, previous_end as i32, 0],
                &[1, seq_len as i32, hidden_size],
            ));
        }
        let merged = if segments.is_empty() {
            inputs_embeds
        } else {
            encoders::phi4_siglip::concat_arrays(&segments, 1)
        };
        self.request_modes.register_embeddings(
            merged
                .as_ref()
                .ok_or("Phi4MM produced null input embeddings")?,
            mode,
        );
        Ok(InputEmbeddings {
            inputs_embeds: merged,
            attention_mask_4d: None,
        })
    }

    fn activate_input_mode(&self, mode: Phi4MMInputMode) {
        let adapter = mode.adapter();
        let select = |linear: &UnifiedLinear| {
            if let Err(error) = linear.select_lora(adapter) {
                panic!("validated Phi4MM adapter selection failed: {error}");
            }
        };
        for layer in &self.text_model.layers {
            select(&layer.self_attn.qkv_proj);
            select(&layer.self_attn.o_proj);
            select(&layer.mlp.gate_up_proj);
            select(&layer.mlp.down_proj);
        }
    }

    /// Perform HD transform for one image: encode all crops, pool, assemble with separators, project.
    fn hd_transform(
        &self,
        processed: &processors::phi4mm::Phi4MMImageInput,
        target_dtype: i32,
    ) -> UniquePtr<MlxArray> {
        let (h_crops, w_crops) = processed.image_grid;
        let pooled_grid = processed.pooled_grid_size; // 16

        // Encode each crop through vision tower
        let spatial_per_crop = (
            (self.processor.crop_size / self.processor.patch_size) as i32,
            (self.processor.crop_size / self.processor.patch_size) as i32,
        ); // (32, 32)

        let mut crop_features: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(processed.crops.len());
        for crop in &processed.crops {
            let pv = mlxcel_core::astype(&crop.pixel_values, target_dtype);
            let mut hidden_states = self
                .vision_tower
                .forward_hidden_states(&pv, spatial_per_crop);
            let layer_count = hidden_states.len() as isize;
            let selected_index = if self.select_layer < 0 {
                (layer_count + self.select_layer) as usize
            } else {
                self.select_layer as usize
            };
            let selected = hidden_states.swap_remove(selected_index);
            crop_features.push(selected);
        }

        // AvgPool2d: reduce each crop from 32×32 to 16×16
        let orig_grid = self.processor.crop_size / self.processor.patch_size; // 32
        let mut pooled_features: Vec<UniquePtr<MlxArray>> = Vec::with_capacity(crop_features.len());
        for feat in &crop_features {
            let pooled = avg_pool_2d(feat, orig_grid, pooled_grid);
            pooled_features.push(pooled);
        }

        let vision_dim = mlxcel_core::array_shape(&pooled_features[0])[2];

        // Global features (first crop): reshape to grid + add sub_GN column separators
        let global_feat = &pooled_features[0]; // [1, 256, D]
        let global_grid = mlxcel_core::reshape(
            global_feat,
            &[1, pooled_grid as i32, pooled_grid as i32, vision_dim],
        );
        // Add sub_GN separator column: repeat sub_GN for each row
        let sub_gn_col = self.make_sub_gn_column(pooled_grid as i32, vision_dim, target_dtype);
        let global_with_sep = mlxcel_core::concatenate(&global_grid, &sub_gn_col, 2);
        // Flatten: [1, pooled_grid, pooled_grid+1, D] → [1, pooled_grid*(pooled_grid+1), D]
        let global_tokens = mlxcel_core::reshape(
            &global_with_sep,
            &[1, (pooled_grid * (pooled_grid + 1)) as i32, vision_dim],
        );

        // Sub-image features: arrange spatially + add sub_GN separators
        let sub_tokens = self.assemble_sub_features(
            &pooled_features[1..],
            h_crops,
            w_crops,
            pooled_grid,
            vision_dim,
            processed.active_rows,
            processed.active_cols,
            target_dtype,
        );

        // Combine based on hd_transform_order
        let glb_gn_token = mlxcel_core::astype(&self.glb_gn, target_dtype);
        let glb_gn_3d = mlxcel_core::reshape(&glb_gn_token, &[1, 1, vision_dim]);

        let combined = if self.hd_transform_order == "sub_glb" {
            concat_3arrays(&sub_tokens, &glb_gn_3d, &global_tokens)
        } else {
            concat_3arrays(&global_tokens, &glb_gn_3d, &sub_tokens)
        };

        // Project through MLP: Linear → GELU → Linear
        let projected = self.mm_projector_linear1.forward(&combined);
        let projected = mlxcel_core::gelu_approx(&projected);
        self.mm_projector_linear2.forward(&projected)
    }

    /// Create a sub_GN separator column: [1, num_rows, 1, D]
    fn make_sub_gn_column(&self, num_rows: i32, dim: i32, dtype: i32) -> UniquePtr<MlxArray> {
        let sub_gn = mlxcel_core::astype(&self.sub_gn, dtype);
        // sub_gn shape: [1, 1, 1, D] → broadcast to [1, num_rows, 1, D]
        mlxcel_core::broadcast_to(&sub_gn, &[1, num_rows, 1, dim])
    }

    /// Assemble sub-crop features spatially with separators.
    fn assemble_sub_features(
        &self,
        sub_crop_features: &[UniquePtr<MlxArray>],
        h_crops: usize,
        w_crops: usize,
        pooled_grid: usize,
        vision_dim: i32,
        active_rows: usize,
        active_cols: usize,
        dtype: i32,
    ) -> UniquePtr<MlxArray> {
        let pg = pooled_grid as i32;
        // Each sub-crop: [1, pooled_grid^2, D]
        // Reshape each to [pooled_grid, pooled_grid, D]
        // Arrange spatially: (h_crops*pg, w_crops*pg, D)
        let total_h = h_crops * pooled_grid;
        let total_w = w_crops * pooled_grid;

        // Build the spatial grid
        let mut row_segments: Vec<UniquePtr<MlxArray>> = Vec::new();
        for ch in 0..h_crops {
            let mut col_segments: Vec<UniquePtr<MlxArray>> = Vec::new();
            for cw in 0..w_crops {
                let crop_idx = ch * w_crops + cw;
                let feat = &sub_crop_features[crop_idx];
                // [1, pg*pg, D] → [pg, pg, D]
                let reshaped = mlxcel_core::reshape(feat, &[pg, pg, vision_dim]);
                col_segments.push(reshaped);
            }
            // Concatenate columns: [pg, w_crops*pg, D]
            let row = concat_arrays_axis(&col_segments, 1);
            row_segments.push(row);
        }
        // Concatenate rows: [h_crops*pg, w_crops*pg, D]
        let spatial = concat_arrays_axis(&row_segments, 0);

        // Crop to active region (remove padding)
        let useful_h = active_rows.min(total_h) as i32;
        let useful_w = active_cols.min(total_w) as i32;
        let cropped = if useful_h < total_h as i32 || useful_w < total_w as i32 {
            mlxcel_core::slice(&spatial, &[0, 0, 0], &[useful_h, useful_w, vision_dim])
        } else {
            spatial
        };

        // Add sub_GN separator column per row: [useful_h, useful_w, D] → [1, useful_h, useful_w+1, D]
        let cropped_4d = mlxcel_core::reshape(&cropped, &[1, useful_h, useful_w, vision_dim]);
        let sub_gn_col = self.make_sub_gn_column(useful_h, vision_dim, dtype);
        let with_sep = mlxcel_core::concatenate(&cropped_4d, &sub_gn_col, 2);

        // Flatten: [1, useful_h * (useful_w + 1), D]
        let total_tokens = useful_h * (useful_w + 1);
        mlxcel_core::reshape(&with_sep, &[1, total_tokens, vision_dim])
    }
}

/// AvgPool2d: spatial 2×2 average pooling on vision features.
/// Input: [1, grid*grid, D] → Output: [1, (grid/2)*(grid/2), D]
fn avg_pool_2d(features: &MlxArray, orig_grid: usize, target_grid: usize) -> UniquePtr<MlxArray> {
    let shape = mlxcel_core::array_shape(features);
    let dim = shape[2];
    let og = orig_grid as i32;
    let tg = target_grid as i32;
    let pool_size = (orig_grid / target_grid) as i32;

    // [1, og*og, D] → [og, og, D]
    let spatial = mlxcel_core::reshape(features, &[og, og, dim]);
    // → [D, og, og] (NCHW-like for pooling)
    let transposed = mlxcel_core::transpose_axes(&spatial, &[2, 0, 1]);
    // [D, og, og] → [D, 1, og, og]
    let batched = mlxcel_core::reshape(&transposed, &[dim, 1, og, og]);

    // Manual 2×2 average pooling: take every 2×2 block and average
    // Reshape to [D, 1, tg, pool_size, tg, pool_size]
    let blocked = mlxcel_core::reshape(&batched, &[dim, 1, tg, pool_size, tg, pool_size]);
    // Mean over pool dimensions (axes 3 and 5)
    let mean1 = mlxcel_core::mean_axis(&blocked, 5, true);
    let mean2 = mlxcel_core::mean_axis(&mean1, 3, true);
    // [D, 1, tg, 1, tg, 1] → [D, tg, tg]
    let squeezed = mlxcel_core::reshape(&mean2, &[dim, tg, tg]);
    // → [tg, tg, D]
    let result = mlxcel_core::transpose_axes(&squeezed, &[1, 2, 0]);
    // → [1, tg*tg, D]
    mlxcel_core::reshape(&result, &[1, tg * tg, dim])
}

/// Concatenate three 3D arrays along axis 1.
fn concat_3arrays(a: &MlxArray, b: &MlxArray, c: &MlxArray) -> UniquePtr<MlxArray> {
    let ab = mlxcel_core::concatenate(a, b, 1);
    mlxcel_core::concatenate(&ab, c, 1)
}

/// Concatenate multiple arrays along a given axis.
fn concat_arrays_axis(arrays: &[UniquePtr<MlxArray>], axis: i32) -> UniquePtr<MlxArray> {
    if arrays.len() == 1 {
        return mlxcel_core::copy(&arrays[0]);
    }
    let mut result = mlxcel_core::copy(&arrays[0]);
    for arr in &arrays[1..] {
        result = mlxcel_core::concatenate(&result, arr, axis);
    }
    result
}

impl LanguageModel for Phi4MMVLModel {
    fn forward(
        &self,
        input_ids: &MlxArray,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.activate_input_mode(self.request_modes.decode_mode(None));
        self.text_model.forward(input_ids, caches, mask)
    }

    fn forward_with_embeddings(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mode = self.request_modes.begin_prefill(input_embeddings, None);
        self.activate_input_mode(mode);
        self.text_model
            .forward_impl(input_ids, input_embeddings, caches, mask)
    }

    fn forward_with_sequence_id(
        &self,
        input_ids: &MlxArray,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        self.activate_input_mode(self.request_modes.decode_mode(seq_id));
        self.text_model.forward(input_ids, caches, mask)
    }

    fn forward_with_embeddings_and_sequence_id(
        &self,
        input_ids: &MlxArray,
        input_embeddings: Option<&MlxArray>,
        seq_id: Option<SequenceId>,
        caches: &mut [KVCache],
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        let mode = self.request_modes.begin_prefill(input_embeddings, seq_id);
        self.activate_input_mode(mode);
        self.text_model
            .forward_impl(input_ids, input_embeddings, caches, mask)
    }

    fn embed_tokens(&self, input_ids: &MlxArray) -> Option<UniquePtr<MlxArray>> {
        Some(self.text_model.get_embed_tokens(input_ids))
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

    fn output_suppressed_token_ids(&self) -> Vec<i32> {
        vec![PHI4MM_AUDIO_TOKEN_ID]
    }

    fn supports_chunked_prefill(&self) -> bool {
        false
    }

    fn supports_padded_prefill(&self) -> bool {
        // Padding copies the merged embedding array and would sever the
        // request-mode registration used to choose the official Phi4MM LoRA.
        false
    }

    fn supports_batching(&self) -> bool {
        false
    }

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        // Phi4MM selects one checkpoint LoRA adapter per request, so decode
        // remains single-slot. Its text backbone still stores state in the
        // external Phi-3 KV cache slice; declaring that layout explicitly
        // prevents the scheduler from allocating the empty placeholder used
        // by model-owned SSM/recurrent runtimes.
        SequenceStateLayout::dense_kv_cache(self.text_model.num_layers())
    }

    fn prepare_sequence_state(&self, seq_id: SequenceId) {
        self.request_modes.prepare(seq_id);
    }

    fn release_sequence_state_by_id(&self, seq_id: SequenceId) {
        self.request_modes.release(seq_id);
    }

    fn reset_runtime_state(&self) {
        self.request_modes.reset();
        self.activate_input_mode(Phi4MMInputMode::Language);
    }
}

#[cfg(test)]
mod mode_tests {
    use super::*;

    #[test]
    fn request_modes_do_not_leak_between_sequences() {
        let modes = Phi4MMRequestModes::default();
        let audio = mlxcel_core::zeros(&[1, 1, 4], mlxcel_core::dtype::FLOAT32);
        let vision = mlxcel_core::zeros(&[1, 1, 4], mlxcel_core::dtype::FLOAT32);
        let audio_ref = audio.as_ref().unwrap();
        let vision_ref = vision.as_ref().unwrap();
        modes.register_embeddings(audio_ref, Phi4MMInputMode::Speech);
        modes.register_embeddings(vision_ref, Phi4MMInputMode::VisionSpeech);
        let first = SequenceId::from_raw(41);
        let second = SequenceId::from_raw(42);
        assert_eq!(
            modes.begin_prefill(Some(audio_ref), Some(first)),
            Phi4MMInputMode::Speech
        );
        assert_eq!(
            modes.begin_prefill(Some(vision_ref), Some(second)),
            Phi4MMInputMode::VisionSpeech
        );
        assert_eq!(modes.decode_mode(Some(first)), Phi4MMInputMode::Speech);
        assert_eq!(
            modes.decode_mode(Some(second)),
            Phi4MMInputMode::VisionSpeech
        );
        modes.release(first);
        assert_eq!(modes.decode_mode(Some(first)), Phi4MMInputMode::Language);
        assert_eq!(
            modes.decode_mode(Some(second)),
            Phi4MMInputMode::VisionSpeech
        );

        modes.fallback.set(Phi4MMInputMode::Speech);
        assert_eq!(
            modes.decode_mode(Some(first)),
            Phi4MMInputMode::Language,
            "a missing sequence must not inherit the legacy fallback mode"
        );

        modes.reset();
        assert_eq!(modes.decode_mode(Some(first)), Phi4MMInputMode::Language);
        assert_eq!(modes.decode_mode(Some(second)), Phi4MMInputMode::Language);

        let pending = mlxcel_core::zeros(&[1, 1, 4], mlxcel_core::dtype::FLOAT32);
        let pending_ref = pending.as_ref().unwrap();
        modes.register_embeddings(pending_ref, Phi4MMInputMode::Speech);
        modes.reset();
        assert_eq!(
            modes.begin_prefill(Some(pending_ref), None),
            Phi4MMInputMode::Speech,
            "a generator reset must preserve embeddings prepared for the new request"
        );
    }
}
