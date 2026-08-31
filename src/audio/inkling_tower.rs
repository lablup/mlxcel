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

//! Inkling's summed-channel dMel embedding tower.

use mlxcel_core::layers::{RMSNorm, UnifiedEmbedding};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;

fn default_model_type() -> String {
    "inkling_audio".into()
}
fn default_mels() -> usize {
    80
}
fn default_vocab() -> usize {
    16
}
fn default_hidden() -> usize {
    6_144
}
fn default_eps() -> f32 {
    1e-6
}
fn default_chunk() -> usize {
    256
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct InklingAudioConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    #[serde(default = "default_mels")]
    pub n_mel_bins: usize,
    #[serde(default = "default_vocab")]
    pub mel_vocab_size: usize,
    #[serde(default = "default_hidden", alias = "decoder_dmodel")]
    pub text_hidden_size: usize,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_chunk")]
    pub max_frames_per_chunk: usize,
}

impl InklingAudioConfig {
    pub fn validate(&self, text_hidden_size: usize) -> Result<(), String> {
        if self.model_type != "inkling_audio" {
            return Err(format!(
                "Inkling audio model_type must be inkling_audio, got {}",
                self.model_type
            ));
        }
        if self.n_mel_bins != super::inkling_dmel::DEFAULT_N_MEL_BINS
            || self.mel_vocab_size != super::inkling_dmel::DEFAULT_MEL_VOCAB_SIZE
        {
            return Err(format!(
                "Inkling audio requires {} mel channels and {} dMel bins",
                super::inkling_dmel::DEFAULT_N_MEL_BINS,
                super::inkling_dmel::DEFAULT_MEL_VOCAB_SIZE
            ));
        }
        if self.max_frames_per_chunk == 0 || self.max_frames_per_chunk > default_chunk() {
            return Err(format!(
                "Inkling audio max_frames_per_chunk must be in 1..={}",
                default_chunk()
            ));
        }
        if self.text_hidden_size != text_hidden_size {
            return Err(format!(
                "Inkling audio text_hidden_size {} does not match text hidden_size {text_hidden_size}",
                self.text_hidden_size
            ));
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err("Inkling audio rms_norm_eps must be finite and positive".into());
        }
        self.n_mel_bins
            .checked_mul(self.mel_vocab_size)
            .and_then(|value| i32::try_from(value).ok())
            .ok_or_else(|| {
                "Inkling audio embedding vocabulary exceeds the MLX i32 limit".to_string()
            })?;
        i32::try_from(self.text_hidden_size)
            .map_err(|_| "Inkling audio hidden size exceeds the MLX i32 limit".to_string())?;
        Ok(())
    }
}

pub struct InklingAudioTower {
    embed: UnifiedEmbedding,
    norm: RMSNorm,
    n_mel_bins: usize,
    mel_vocab_size: usize,
    hidden_size: usize,
    max_frames_per_chunk: usize,
}

impl InklingAudioTower {
    pub fn from_weights(
        weights: &WeightMap,
        config: &InklingAudioConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let embed = UnifiedEmbedding::from_weights(
            weights,
            "audio_tower.embed_audio_tokens",
            group_size,
            bits,
        )?;
        let norm_weight = weights
            .get("audio_tower.norm.weight")
            .map(|weight| mlxcel_core::copy(weight))
            .ok_or_else(|| "Weight not found: audio_tower.norm.weight".to_string())?;
        Ok(Self {
            embed,
            norm: RMSNorm::new(norm_weight, config.rms_norm_eps),
            n_mel_bins: config.n_mel_bins,
            mel_vocab_size: config.mel_vocab_size,
            hidden_size: config.text_hidden_size,
            max_frames_per_chunk: config.max_frames_per_chunk,
        })
    }

    pub fn forward(&self, ids: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let shape = mlxcel_core::array_shape(ids);
        let [frames, channels] = shape.as_slice() else {
            return Err(format!(
                "Inkling audio ids must have shape [frames, {}], got {shape:?}",
                self.n_mel_bins
            ));
        };
        if *channels != self.n_mel_bins as i32 {
            return Err(format!(
                "Inkling audio ids have {channels} channels, expected {}",
                self.n_mel_bins
            ));
        }
        if *frames <= 0 {
            return Err("Inkling audio tower requires at least one valid frame".into());
        }
        if mlxcel_core::array_dtype(ids) != mlxcel_core::dtype::INT32 {
            return Err("Inkling audio ids must use int32 dtype".into());
        }
        let minimum = mlxcel_core::item_i32(&mlxcel_core::min_all(ids));
        let maximum = mlxcel_core::item_i32(&mlxcel_core::max_all(ids));
        if minimum < 0 || maximum >= self.mel_vocab_size as i32 {
            return Err(format!(
                "Inkling dMel ids must be in 0..{}, got values spanning {minimum}..={maximum}",
                self.mel_vocab_size
            ));
        }
        let offsets: Vec<i32> = (0..self.n_mel_bins)
            .map(|channel| (channel * self.mel_vocab_size) as i32)
            .collect();
        let offsets = mlxcel_core::from_slice_i32(&offsets, &[self.n_mel_bins as i32]);
        let offsets = mlxcel_core::reshape(&offsets, &[1, self.n_mel_bins as i32]);
        let mut output: Option<UniquePtr<MlxArray>> = None;
        let total = *frames as usize;
        for start in (0..total).step_by(self.max_frames_per_chunk) {
            let end = (start + self.max_frames_per_chunk).min(total);
            let chunk = mlxcel_core::slice(
                ids,
                &[start as i32, 0],
                &[end as i32, self.n_mel_bins as i32],
            );
            let indices = mlxcel_core::add(&chunk, &offsets);
            let embedded = self.embed.forward(&indices);
            let summed = mlxcel_core::sum_axis(&embedded, -2, false);
            let normalized = self.norm.forward(&summed);
            mlxcel_core::eval(&normalized);
            output = Some(match output {
                Some(previous) => mlxcel_core::concatenate(&previous, &normalized, 0),
                None => normalized,
            });
        }
        let output = output.ok_or_else(|| "Inkling audio tower produced no chunks".to_string())?;
        let output_shape = mlxcel_core::array_shape(&output);
        if output_shape != [*frames, self.hidden_size as i32] {
            return Err(format!(
                "Inkling audio tower produced {output_shape:?}, expected [{frames}, {}]",
                self.hidden_size
            ));
        }
        Ok(output)
    }
}

#[cfg(test)]
#[path = "inkling_tower_tests.rs"]
mod tests;
