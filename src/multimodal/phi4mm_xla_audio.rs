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

//! Phi4MM host media/text-embedding producer paired with the IREE audio and
//! language runtimes.
//!
//! SpeechLib, the qualified image tower/projector, and the checkpoint embedding
//! table execute through MLX. The cascaded Conformer, audio projection,
//! modality-LoRA decoder, KV, and logits stay on the IREE path; no MLX decoder
//! or MLX audio encoder is constructed or retained.

use std::path::{Path, PathBuf};

use mlxcel_core::layers::UnifiedEmbedding;
use mlxcel_core::session::{
    OwnedTensor, PreparedAdapterMode, PreparedAttentionBias, PreparedModality, PreparedPositions,
    PreparedPrefill, PreparedTensorDType,
};
use mlxcel_xla::{Phi4AudioProjectionMode, Phi4AudioRuntime, phi4_audio_bucket_for_frames};

use crate::audio::phi4mm::Phi4MMAudioFeatureExtractor;
use crate::audio::{AudioPreprocessCheckpoint, AudioWaveformBatch};
use crate::loading::Phi4MMXlaVisionComponents;
use crate::multimodal::phi4_siglip_prompt::PHI4_SIGLIP_IMAGE_TOKEN_INDEX;
use crate::multimodal::phi4mm_prompt::{PHI4MM_AUDIO_TOKEN_ID, expand_phi4mm_placeholders};
use crate::vision::processors::phi4mm::Phi4MMImageInput;

#[doc(hidden)]
pub fn load_phi4mm_audio_policy(
    model_path: &Path,
) -> Result<crate::audio::AudioFamilyPolicy, String> {
    let config_text = std::fs::read_to_string(model_path.join("config.json"))
        .map_err(|error| format!("read Phi4MM config.json: {error}"))?;
    let config: serde_json::Value = serde_json::from_str(&config_text)
        .map_err(|error| format!("parse Phi4MM config.json: {error}"))?;
    let preprocessor_path = model_path.join("preprocessor_config.json");
    let preprocessor = if preprocessor_path.is_file() {
        let text = std::fs::read_to_string(&preprocessor_path)
            .map_err(|error| format!("read {}: {error}", preprocessor_path.display()))?;
        Some(
            serde_json::from_str::<serde_json::Value>(&text)
                .map_err(|error| format!("parse {}: {error}", preprocessor_path.display()))?,
        )
    } else {
        None
    };
    crate::audio::AudioFamilyPolicy::from_phi4mm_configs(&config, preprocessor.as_ref())
        .map_err(|error| error.to_string())
}

/// Thread-confined host producer for Phi4MM audio requests.
#[doc(hidden)]
pub struct Phi4MmXlaAudioProducer {
    model_path: PathBuf,
    device: String,
    extractor: Phi4MMAudioFeatureExtractor,
    text_embeddings: UnifiedEmbedding,
    hidden_size: usize,
    max_sequence_len: usize,
    runtime: Option<Phi4AudioRuntime>,
    vision: Option<Phi4MMXlaVisionComponents>,
}

impl Phi4MmXlaAudioProducer {
    #[doc(hidden)]
    pub fn load(model_path: &Path, device: &str, context_capacity: usize) -> Result<Self, String> {
        let (text_embeddings, hidden_size, checkpoint_capacity) =
            crate::loading::load_phi4mm_xla_text_embeddings(model_path)
                .map_err(|error| error.to_string())?;
        let max_sequence_len = context_capacity.min(checkpoint_capacity);
        if max_sequence_len == 0 {
            return Err("Phi4MM XLA audio context capacity must be positive".to_string());
        }
        Ok(Self {
            model_path: model_path.to_path_buf(),
            device: device.to_string(),
            extractor: Phi4MMAudioFeatureExtractor::new(),
            text_embeddings,
            hidden_size,
            max_sequence_len,
            runtime: None,
            vision: None,
        })
    }

    /// Load the same audio path plus the filtered #874 image tower/projector.
    ///
    /// This constructor is used by serving because one worker must be able to
    /// prepare image-only, audio-only, and mixed requests without changing
    /// process-global model state.
    #[doc(hidden)]
    pub fn load_multimodal(
        model_path: &Path,
        device: &str,
        context_capacity: usize,
    ) -> Result<Self, String> {
        let (text_embeddings, hidden_size, checkpoint_capacity, vision) =
            crate::loading::load_phi4mm_xla_media_components(model_path)
                .map_err(|error| error.to_string())?;
        let max_sequence_len = context_capacity.min(checkpoint_capacity);
        if max_sequence_len == 0 {
            return Err("Phi4MM XLA media context capacity must be positive".to_string());
        }
        Ok(Self {
            model_path: model_path.to_path_buf(),
            device: device.to_string(),
            extractor: Phi4MMAudioFeatureExtractor::new(),
            hidden_size,
            max_sequence_len,
            runtime: None,
            text_embeddings,
            vision: Some(vision),
        })
    }

    #[doc(hidden)]
    pub fn prepare_audio(
        &mut self,
        waveforms: AudioWaveformBatch,
        token_ids: Vec<i32>,
        cancelled: &std::sync::atomic::AtomicBool,
    ) -> Result<PreparedPrefill, String> {
        self.prepare_media(waveforms, token_ids, &[], cancelled)
    }

    #[doc(hidden)]
    pub fn prepare_images(
        &mut self,
        token_ids: Vec<i32>,
        images: &[image::DynamicImage],
        cancelled: &std::sync::atomic::AtomicBool,
    ) -> Result<PreparedPrefill, String> {
        self.prepare_media(
            AudioWaveformBatch {
                family: "phi4mm",
                clips: Vec::new(),
                boundaries: Vec::new(),
                total_source_samples: 0,
                total_samples: 0,
                total_source_duration_micros: 0,
                estimated_frames: 0,
                effective_audio_tokens: 0,
            },
            token_ids,
            images,
            cancelled,
        )
    }

    /// Prepare image-only, audio-only, or mixed Phi4MM embeddings.
    ///
    /// Any request containing an image selects the Vision decoder adapter and
    /// the vision branch of the audio projector. Audio-only requests select the
    /// Speech branch. This exactly matches the request mode contract qualified
    /// by #874.
    #[doc(hidden)]
    pub fn prepare_media(
        &mut self,
        waveforms: AudioWaveformBatch,
        token_ids: Vec<i32>,
        images: &[image::DynamicImage],
        cancelled: &std::sync::atomic::AtomicBool,
    ) -> Result<PreparedPrefill, String> {
        if waveforms.family != "phi4mm" {
            return Err(format!(
                "Phi4MM XLA producer received `{}` audio policy",
                waveforms.family
            ));
        }
        if !images.is_empty() && self.vision.is_none() {
            return Err(
                "Phi4MM image input requires the filtered XLA multimodal producer".to_string(),
            );
        }
        if cancelled.load(std::sync::atomic::Ordering::Acquire) {
            return Err("Phi4MM media preparation was cancelled before features".to_string());
        }

        let processed_images = self
            .vision
            .as_ref()
            .map(|vision| vision.processor.preprocess(images))
            .unwrap_or_default();
        let image_sizes = processed_images
            .iter()
            .map(|image| image.num_img_tokens)
            .collect::<Vec<_>>();

        let clips = waveforms
            .clips
            .into_iter()
            .map(|clip| (clip.samples, clip.sample_rate))
            .collect::<Vec<_>>();
        let (audio_features, audio_sizes, audio_frame_lengths) = if clips.is_empty() {
            (Vec::new(), Vec::new(), Vec::new())
        } else {
            let batch = self
                .extractor
                .extract_batch_cancellable(&clips, cancelled)?;
            (batch.clips, batch.embed_sizes, batch.frame_lengths)
        };

        let vision_mode = !processed_images.is_empty();
        let projection_mode = if vision_mode {
            Phi4AudioProjectionMode::Vision
        } else {
            Phi4AudioProjectionMode::Speech
        };
        let mut projected_audio = Vec::with_capacity(audio_features.len());
        for (index, features) in audio_features.iter().enumerate() {
            if cancelled.load(std::sync::atomic::Ordering::Acquire) {
                return Err(format!(
                    "Phi4MM audio preparation was cancelled at {:?}",
                    AudioPreprocessCheckpoint::Feature
                ));
            }
            let frame_len = audio_frame_lengths[index];
            let bucket = phi4_audio_bucket_for_frames(frame_len).ok_or_else(|| {
                format!(
                    "Phi4MM audio clip {} has unsupported {frame_len} frames",
                    index + 1
                )
            })?;
            if self
                .runtime
                .as_ref()
                .is_none_or(|runtime| runtime.frame_bucket() != bucket)
            {
                self.runtime = Some(Phi4AudioRuntime::load(
                    &self.model_path,
                    &self.device,
                    bucket,
                )?);
            }
            let features = mlxcel_core::astype(features, mlxcel_core::dtype::FLOAT32);
            let feature_values = raw_f32(&features, "Phi4MM SpeechLib features")?;
            let output = self
                .runtime
                .as_mut()
                .expect("runtime loaded for selected bucket")
                .project(&feature_values, frame_len, projection_mode)?;
            if output.valid_rows != audio_sizes[index] {
                return Err(format!(
                    "Phi4MM audio clip {} produced {} projection rows, expected {}",
                    index + 1,
                    output.valid_rows,
                    audio_sizes[index]
                ));
            }
            if output.hidden_size != self.hidden_size {
                return Err(format!(
                    "Phi4MM audio projection hidden size {} does not match text hidden size {}",
                    output.hidden_size, self.hidden_size
                ));
            }
            projected_audio.push(output.projected);
        }

        let logical_tokens = expand_phi4mm_placeholders(&token_ids, &image_sizes, &audio_sizes)?;
        if logical_tokens.len() > self.max_sequence_len {
            return Err(format!(
                "Phi4MM prepared sequence length {} exceeds XLA context capacity {}",
                logical_tokens.len(),
                self.max_sequence_len
            ));
        }
        let safe_tokens = logical_tokens
            .iter()
            .map(|&token| {
                if token == PHI4MM_AUDIO_TOKEN_ID || token == PHI4_SIGLIP_IMAGE_TOKEN_INDEX {
                    0
                } else {
                    token
                }
            })
            .collect::<Vec<_>>();
        let input_ids =
            mlxcel_core::from_slice_i32(&safe_tokens, &[1, logical_tokens.len() as i32]);
        let text = self.text_embeddings.forward(&input_ids);
        let shape = mlxcel_core::array_shape(&text);
        if shape != [1, logical_tokens.len() as i32, self.hidden_size as i32] {
            return Err(format!(
                "Phi4MM text embeddings have shape {shape:?}, expected [1, {}, {}]",
                logical_tokens.len(),
                self.hidden_size
            ));
        }
        let text_dtype = mlxcel_core::array_dtype(&text);
        let mut projected_images = Vec::with_capacity(processed_images.len());
        if let Some(vision) = self.vision.as_ref() {
            for (index, processed) in processed_images.iter().enumerate() {
                if cancelled.load(std::sync::atomic::Ordering::Acquire) {
                    return Err(format!(
                        "Phi4MM image preparation was cancelled before image {}",
                        index + 1
                    ));
                }
                let projected = vision.hd_transform(processed, text_dtype)?;
                let shape = mlxcel_core::array_shape(&projected);
                if shape != [1, image_sizes[index] as i32, self.hidden_size as i32] {
                    return Err(format!(
                        "Phi4MM image {} projection has shape {shape:?}, expected [1, {}, {}]",
                        index + 1,
                        image_sizes[index],
                        self.hidden_size
                    ));
                }
                let projected = mlxcel_core::astype(&projected, mlxcel_core::dtype::FLOAT32);
                projected_images.push(raw_f32(
                    &projected,
                    &format!("Phi4MM projected image {}", index + 1),
                )?);
            }
        }
        let text = mlxcel_core::astype(&text, mlxcel_core::dtype::FLOAT32);
        let mut merged = raw_f32(&text, "Phi4MM text embeddings")?;
        merge_media_rows(
            &logical_tokens,
            &projected_images,
            &projected_audio,
            self.hidden_size,
            &mut merged,
        )?;
        let embeddings = OwnedTensor::new(
            f32_bytes(&merged),
            PreparedTensorDType::Float32,
            vec![1, logical_tokens.len(), self.hidden_size],
        )
        .map_err(|error| error.to_string())?;
        let attention_bias = PreparedAttentionBias {
            tensor: OwnedTensor::new(
                vec![0; logical_tokens.len() * std::mem::size_of::<f32>()],
                PreparedTensorDType::Float32,
                vec![1, 1, 1, logical_tokens.len()],
            )
            .map_err(|error| error.to_string())?,
            causal: true,
        };
        let mut modalities = Vec::with_capacity(2);
        if !projected_images.is_empty() {
            modalities.push(PreparedModality {
                family: "phi4mm-image".to_string(),
                item_count: projected_images.len(),
                token_count: image_sizes.iter().sum(),
            });
        }
        if !projected_audio.is_empty() {
            modalities.push(PreparedModality {
                family: "phi4mm-audio".to_string(),
                item_count: projected_audio.len(),
                token_count: audio_sizes.iter().sum(),
            });
        }
        let adapter_mode = if vision_mode {
            PreparedAdapterMode::Vision
        } else if !projected_audio.is_empty() {
            PreparedAdapterMode::Speech
        } else {
            PreparedAdapterMode::Language
        };
        PreparedPrefill::new(
            logical_tokens,
            embeddings,
            PreparedPositions::Sequential {
                start: 0,
                length: safe_tokens.len(),
            },
            attention_bias,
            modalities,
        )
        .map(|prepared| prepared.with_adapter_mode(adapter_mode))
        .map_err(|error| error.to_string())
    }
}

impl Phi4MMXlaVisionComponents {
    /// Execute the exact #874 HD image transform while retaining only the
    /// filtered image tower/projector weights.
    fn hd_transform(
        &self,
        processed: &Phi4MMImageInput,
        target_dtype: i32,
    ) -> Result<mlxcel_core::UniquePtr<mlxcel_core::MlxArray>, String> {
        let (h_crops, w_crops) = processed.image_grid;
        let expected_crops = h_crops
            .checked_mul(w_crops)
            .and_then(|count| count.checked_add(1))
            .ok_or("Phi4MM image crop count overflow")?;
        if processed.crops.len() != expected_crops {
            return Err(format!(
                "Phi4MM image processor produced {} crops, expected {expected_crops}",
                processed.crops.len()
            ));
        }
        let pooled_grid = processed.pooled_grid_size;
        if pooled_grid == 0 {
            return Err("Phi4MM image pooled grid must be positive".to_string());
        }
        let spatial_per_crop = (
            (self.processor.crop_size / self.processor.patch_size) as i32,
            (self.processor.crop_size / self.processor.patch_size) as i32,
        );
        let mut crop_features = Vec::with_capacity(processed.crops.len());
        for crop in &processed.crops {
            let pixels = mlxcel_core::astype(&crop.pixel_values, target_dtype);
            let mut hidden_states = self
                .vision_tower
                .forward_hidden_states(&pixels, spatial_per_crop);
            let layer_count = hidden_states.len() as isize;
            let selected = if self.select_layer < 0 {
                layer_count.checked_add(self.select_layer)
            } else {
                Some(self.select_layer)
            }
            .filter(|&index| index >= 0 && index < layer_count)
            .ok_or_else(|| {
                format!(
                    "Phi4MM vision select layer {} is outside {layer_count} hidden states",
                    self.select_layer
                )
            })? as usize;
            crop_features.push(hidden_states.swap_remove(selected));
        }

        let original_grid = self.processor.crop_size / self.processor.patch_size;
        let pooled_features = crop_features
            .iter()
            .map(|features| avg_pool_2d(features, original_grid, pooled_grid))
            .collect::<Vec<_>>();
        let vision_dim = mlxcel_core::array_shape(&pooled_features[0])[2];
        let global_grid = mlxcel_core::reshape(
            &pooled_features[0],
            &[1, pooled_grid as i32, pooled_grid as i32, vision_dim],
        );
        let sub_gn_col = self.make_sub_gn_column(pooled_grid as i32, vision_dim, target_dtype);
        let global_with_sep = mlxcel_core::concatenate(&global_grid, &sub_gn_col, 2);
        let global_tokens = mlxcel_core::reshape(
            &global_with_sep,
            &[1, (pooled_grid * (pooled_grid + 1)) as i32, vision_dim],
        );
        let sub_tokens =
            self.assemble_sub_features(&pooled_features[1..], processed, vision_dim, target_dtype)?;
        let global_separator = mlxcel_core::astype(&self.glb_gn, target_dtype);
        let global_separator = mlxcel_core::reshape(&global_separator, &[1, 1, vision_dim]);
        let combined = if self.hd_transform_order == "sub_glb" {
            concat_3arrays(&sub_tokens, &global_separator, &global_tokens)
        } else {
            concat_3arrays(&global_tokens, &global_separator, &sub_tokens)
        };
        let projected = self.mm_projector_linear1.forward(&combined);
        let projected = mlxcel_core::gelu_approx(&projected);
        Ok(self.mm_projector_linear2.forward(&projected))
    }

    fn make_sub_gn_column(
        &self,
        rows: i32,
        dim: i32,
        dtype: i32,
    ) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
        let separator = mlxcel_core::astype(&self.sub_gn, dtype);
        mlxcel_core::broadcast_to(&separator, &[1, rows, 1, dim])
    }

    fn assemble_sub_features(
        &self,
        features: &[mlxcel_core::UniquePtr<mlxcel_core::MlxArray>],
        processed: &Phi4MMImageInput,
        vision_dim: i32,
        dtype: i32,
    ) -> Result<mlxcel_core::UniquePtr<mlxcel_core::MlxArray>, String> {
        let (h_crops, w_crops) = processed.image_grid;
        let pooled_grid = processed.pooled_grid_size;
        if features.len() != h_crops.saturating_mul(w_crops) {
            return Err(format!(
                "Phi4MM sub-crop feature count is {}, expected {}",
                features.len(),
                h_crops.saturating_mul(w_crops)
            ));
        }
        let grid = pooled_grid as i32;
        let total_h = h_crops * pooled_grid;
        let total_w = w_crops * pooled_grid;
        let mut row_segments = Vec::with_capacity(h_crops);
        for crop_row in 0..h_crops {
            let mut columns = Vec::with_capacity(w_crops);
            for crop_col in 0..w_crops {
                let index = crop_row * w_crops + crop_col;
                columns.push(mlxcel_core::reshape(
                    &features[index],
                    &[grid, grid, vision_dim],
                ));
            }
            row_segments.push(concat_arrays_axis(&columns, 1));
        }
        let spatial = concat_arrays_axis(&row_segments, 0);
        let useful_h = processed.active_rows.min(total_h) as i32;
        let useful_w = processed.active_cols.min(total_w) as i32;
        if useful_h <= 0 || useful_w <= 0 {
            return Err("Phi4MM image active region must be positive".to_string());
        }
        let cropped = if useful_h < total_h as i32 || useful_w < total_w as i32 {
            mlxcel_core::slice(&spatial, &[0, 0, 0], &[useful_h, useful_w, vision_dim])
        } else {
            spatial
        };
        let cropped = mlxcel_core::reshape(&cropped, &[1, useful_h, useful_w, vision_dim]);
        let separator = self.make_sub_gn_column(useful_h, vision_dim, dtype);
        let with_separator = mlxcel_core::concatenate(&cropped, &separator, 2);
        Ok(mlxcel_core::reshape(
            &with_separator,
            &[1, useful_h * (useful_w + 1), vision_dim],
        ))
    }
}

fn avg_pool_2d(
    features: &mlxcel_core::MlxArray,
    original_grid: usize,
    target_grid: usize,
) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let shape = mlxcel_core::array_shape(features);
    let dim = shape[2];
    let original_grid = original_grid as i32;
    let target_grid = target_grid as i32;
    let pool_size = original_grid / target_grid;
    let spatial = mlxcel_core::reshape(features, &[original_grid, original_grid, dim]);
    let transposed = mlxcel_core::transpose_axes(&spatial, &[2, 0, 1]);
    let batched = mlxcel_core::reshape(&transposed, &[dim, 1, original_grid, original_grid]);
    let blocked = mlxcel_core::reshape(
        &batched,
        &[dim, 1, target_grid, pool_size, target_grid, pool_size],
    );
    let mean = mlxcel_core::mean_axis(&blocked, 5, true);
    let mean = mlxcel_core::mean_axis(&mean, 3, true);
    let squeezed = mlxcel_core::reshape(&mean, &[dim, target_grid, target_grid]);
    let result = mlxcel_core::transpose_axes(&squeezed, &[1, 2, 0]);
    mlxcel_core::reshape(&result, &[1, target_grid * target_grid, dim])
}

fn concat_3arrays(
    first: &mlxcel_core::MlxArray,
    second: &mlxcel_core::MlxArray,
    third: &mlxcel_core::MlxArray,
) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let first_two = mlxcel_core::concatenate(first, second, 1);
    mlxcel_core::concatenate(&first_two, third, 1)
}

fn concat_arrays_axis(
    arrays: &[mlxcel_core::UniquePtr<mlxcel_core::MlxArray>],
    axis: i32,
) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let mut output = mlxcel_core::copy(&arrays[0]);
    for array in &arrays[1..] {
        output = mlxcel_core::concatenate(&output, array, axis);
    }
    output
}

fn raw_f32(array: &mlxcel_core::MlxArray, label: &str) -> Result<Vec<f32>, String> {
    let bytes = mlxcel_core::try_array_to_raw_bytes(array)
        .map_err(|error| format!("export {label}: {error}"))?;
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        return Err(format!("{label} has a non-f32 byte length {}", bytes.len()));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("four-byte f32")))
        .collect())
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn merge_media_rows(
    logical_tokens: &[i32],
    projected_images: &[Vec<f32>],
    projected_audio: &[Vec<f32>],
    hidden_size: usize,
    merged: &mut [f32],
) -> Result<(), String> {
    let expected = logical_tokens
        .len()
        .checked_mul(hidden_size)
        .ok_or("Phi4MM merged embedding size overflow")?;
    if merged.len() != expected {
        return Err(format!(
            "Phi4MM merged embedding buffer has {} elements, expected {expected}",
            merged.len()
        ));
    }
    let mut token_index = 0usize;
    let mut image_index = 0usize;
    let mut audio_index = 0usize;
    while token_index < logical_tokens.len() {
        let kind = logical_tokens[token_index];
        if kind != PHI4_SIGLIP_IMAGE_TOKEN_INDEX && kind != PHI4MM_AUDIO_TOKEN_ID {
            token_index += 1;
            continue;
        }
        let (rows, ordinal, label) = if kind == PHI4_SIGLIP_IMAGE_TOKEN_INDEX {
            let rows = projected_images
                .get(image_index)
                .ok_or("Phi4MM prompt has more image placeholder rows than image inputs")?;
            image_index += 1;
            (rows, image_index, "image")
        } else {
            let rows = projected_audio
                .get(audio_index)
                .ok_or("Phi4MM prompt has more audio placeholder rows than audio inputs")?;
            audio_index += 1;
            (rows, audio_index, "audio")
        };
        if rows.len() % hidden_size != 0 {
            return Err(format!(
                "Phi4MM projected {label} {ordinal} has {} elements not divisible by hidden size {hidden_size}",
                rows.len()
            ));
        }
        let row_count = rows.len() / hidden_size;
        let end = token_index
            .checked_add(row_count)
            .ok_or("Phi4MM media placeholder range overflow")?;
        if row_count == 0
            || end > logical_tokens.len()
            || logical_tokens[token_index..end]
                .iter()
                .any(|&token| token != kind)
        {
            return Err(format!(
                "Phi4MM {label} placeholder at position {token_index} does not contain {row_count} contiguous rows"
            ));
        }
        merged[token_index * hidden_size..end * hidden_size].copy_from_slice(rows);
        token_index = end;
    }
    if image_index != projected_images.len() || audio_index != projected_audio.len() {
        return Err(format!(
            "Phi4MM prompt consumed {image_index} image/{audio_index} audio inputs but {}/{} were prepared",
            projected_images.len(),
            projected_audio.len()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_rows_replace_exact_adjacent_placeholder_segments() {
        let tokens = vec![
            7,
            PHI4MM_AUDIO_TOKEN_ID,
            PHI4MM_AUDIO_TOKEN_ID,
            8,
            PHI4MM_AUDIO_TOKEN_ID,
            9,
        ];
        let projected_audio = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0]];
        let mut merged = vec![0.0; tokens.len() * 2];
        merge_media_rows(&tokens, &[], &projected_audio, 2, &mut merged).unwrap();
        assert_eq!(
            merged,
            vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0]
        );
    }

    #[test]
    fn image_and_audio_rows_preserve_mixed_prompt_order() {
        let tokens = vec![
            PHI4_SIGLIP_IMAGE_TOKEN_INDEX,
            PHI4_SIGLIP_IMAGE_TOKEN_INDEX,
            7,
            PHI4MM_AUDIO_TOKEN_ID,
            8,
        ];
        let images = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let audio = vec![vec![5.0, 6.0]];
        let mut merged = vec![0.0; tokens.len() * 2];
        merge_media_rows(&tokens, &images, &audio, 2, &mut merged).unwrap();
        assert_eq!(
            merged,
            vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0]
        );
    }
}
