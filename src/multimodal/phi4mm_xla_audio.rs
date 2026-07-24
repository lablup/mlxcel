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

//! Phi4MM host feature/text-embedding producer paired with the IREE audio and
//! language runtimes.
//!
//! Only the SpeechLib frontend and checkpoint embedding table execute through
//! MLX. The cascaded Conformer, projection, LoRA decoder, KV, and logits stay on
//! the IREE path; no MLX decoder is constructed or retained.

use std::path::{Path, PathBuf};

use mlxcel_core::layers::UnifiedEmbedding;
use mlxcel_core::session::{
    OwnedTensor, PreparedAdapterMode, PreparedAttentionBias, PreparedModality, PreparedPositions,
    PreparedPrefill, PreparedTensorDType,
};
use mlxcel_xla::{Phi4AudioProjectionMode, Phi4AudioRuntime, phi4_audio_bucket_for_frames};

use crate::audio::phi4mm::Phi4MMAudioFeatureExtractor;
use crate::audio::{AudioPreprocessCheckpoint, AudioWaveformBatch};
use crate::multimodal::phi4_siglip_prompt::PHI4_SIGLIP_IMAGE_TOKEN_INDEX;
use crate::multimodal::phi4mm_prompt::{PHI4MM_AUDIO_TOKEN_ID, expand_phi4mm_placeholders};

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
        })
    }

    #[doc(hidden)]
    pub fn prepare_audio(
        &mut self,
        waveforms: AudioWaveformBatch,
        token_ids: Vec<i32>,
        cancelled: &std::sync::atomic::AtomicBool,
    ) -> Result<PreparedPrefill, String> {
        if waveforms.family != "phi4mm" {
            return Err(format!(
                "Phi4MM XLA producer received `{}` audio policy",
                waveforms.family
            ));
        }
        if token_ids
            .iter()
            .any(|&token| token == PHI4_SIGLIP_IMAGE_TOKEN_INDEX)
        {
            return Err(
                "mixed Phi4MM image/audio preparation requires the combined vision producer"
                    .to_string(),
            );
        }
        if cancelled.load(std::sync::atomic::Ordering::Acquire) {
            return Err("Phi4MM audio preparation was cancelled before features".to_string());
        }
        let clips = waveforms
            .clips
            .into_iter()
            .map(|clip| (clip.samples, clip.sample_rate))
            .collect::<Vec<_>>();
        let batch = self
            .extractor
            .extract_batch_cancellable(&clips, cancelled)?;

        let mut projected = Vec::with_capacity(batch.clips.len());
        for (index, features) in batch.clips.iter().enumerate() {
            if cancelled.load(std::sync::atomic::Ordering::Acquire) {
                return Err(format!(
                    "Phi4MM audio preparation was cancelled at {:?}",
                    AudioPreprocessCheckpoint::Feature
                ));
            }
            let frame_len = batch.frame_lengths[index];
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
                .project(&feature_values, frame_len, Phi4AudioProjectionMode::Speech)?;
            if output.valid_rows != batch.embed_sizes[index] {
                return Err(format!(
                    "Phi4MM audio clip {} produced {} projection rows, expected {}",
                    index + 1,
                    output.valid_rows,
                    batch.embed_sizes[index]
                ));
            }
            if output.hidden_size != self.hidden_size {
                return Err(format!(
                    "Phi4MM audio projection hidden size {} does not match text hidden size {}",
                    output.hidden_size, self.hidden_size
                ));
            }
            projected.push(output.projected);
        }

        let logical_tokens = expand_phi4mm_placeholders(&token_ids, &[], &batch.embed_sizes)?;
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
                if token == PHI4MM_AUDIO_TOKEN_ID {
                    0
                } else {
                    token
                }
            })
            .collect::<Vec<_>>();
        let input_ids =
            mlxcel_core::from_slice_i32(&safe_tokens, &[1, logical_tokens.len() as i32]);
        let text = mlxcel_core::astype(
            &self.text_embeddings.forward(&input_ids),
            mlxcel_core::dtype::FLOAT32,
        );
        let shape = mlxcel_core::array_shape(&text);
        if shape != [1, logical_tokens.len() as i32, self.hidden_size as i32] {
            return Err(format!(
                "Phi4MM text embeddings have shape {shape:?}, expected [1, {}, {}]",
                logical_tokens.len(),
                self.hidden_size
            ));
        }
        let mut merged = raw_f32(&text, "Phi4MM text embeddings")?;
        merge_audio_rows(&logical_tokens, &projected, self.hidden_size, &mut merged)?;
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
        PreparedPrefill::new(
            logical_tokens,
            embeddings,
            PreparedPositions::Sequential {
                start: 0,
                length: safe_tokens.len(),
            },
            attention_bias,
            vec![PreparedModality {
                family: "phi4mm-audio".to_string(),
                item_count: projected.len(),
                token_count: batch.embed_sizes.iter().sum(),
            }],
        )
        .map(|prepared| prepared.with_adapter_mode(PreparedAdapterMode::Speech))
        .map_err(|error| error.to_string())
    }
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

fn merge_audio_rows(
    logical_tokens: &[i32],
    projected: &[Vec<f32>],
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
    let mut audio_index = 0usize;
    while token_index < logical_tokens.len() {
        if logical_tokens[token_index] != PHI4MM_AUDIO_TOKEN_ID {
            token_index += 1;
            continue;
        }
        let rows = projected
            .get(audio_index)
            .ok_or("Phi4MM prompt has more audio placeholder rows than audio inputs")?;
        if rows.len() % hidden_size != 0 {
            return Err(format!(
                "Phi4MM projected clip {} has {} elements not divisible by hidden size {hidden_size}",
                audio_index + 1,
                rows.len()
            ));
        }
        let row_count = rows.len() / hidden_size;
        let end = token_index
            .checked_add(row_count)
            .ok_or("Phi4MM audio placeholder range overflow")?;
        if row_count == 0
            || end > logical_tokens.len()
            || logical_tokens[token_index..end]
                .iter()
                .any(|&token| token != PHI4MM_AUDIO_TOKEN_ID)
        {
            return Err(format!(
                "Phi4MM audio placeholder at position {token_index} does not contain {row_count} contiguous rows"
            ));
        }
        merged[token_index * hidden_size..end * hidden_size].copy_from_slice(rows);
        token_index = end;
        audio_index += 1;
    }
    if audio_index != projected.len() {
        return Err(format!(
            "Phi4MM prompt consumed {audio_index} audio inputs but {} were prepared",
            projected.len()
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
        let projected = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0]];
        let mut merged = vec![0.0; tokens.len() * 2];
        merge_audio_rows(&tokens, &projected, 2, &mut merged).unwrap();
        assert_eq!(
            merged,
            vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0]
        );
    }
}
