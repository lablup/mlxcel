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

//! `SoundProjection` — audio embedding to text hidden size projection.
//!
//! Faithful Rust port of upstream
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/nemotron_h_nano_omni/audio.py (SoundProjection).
//!
//! Architecture:
//! `RMSNorm(hidden) -> Linear(hidden, projection_hidden) -> SquaredReLU
//!  -> Linear(projection_hidden, llm_hidden)`.
//!
//! The two `Linear` layers honour `projection_bias` from the audio
//! config (defaults to `false`, matching the released checkpoint).
//!
//! Used by: Nemotron H Nano Omni VLM (audio modality)

use mlxcel_core::layers::{RMSNorm, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::NemotronOmniAudioConfig;

/// Two-layer audio-to-text projector with RMSNorm + ReLU² activation.
pub struct NemotronOmniSoundProjection {
    norm: RMSNorm,
    linear1: UnifiedLinear,
    linear2: UnifiedLinear,
}

impl NemotronOmniSoundProjection {
    /// Build the projector from a checkpoint. `prefix` is typically
    /// `"sound_projection"`.
    ///
    /// Quantization controls (`group_size`, `bits`) are forwarded to the
    /// linear layers so this projector participates in the same
    /// quantization scheme as the rest of the model. Pure-fp16/bf16
    /// checkpoints set both to the model defaults (group_size=64, bits=4)
    /// and `UnifiedLinear` falls through to a non-quantized linear.
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &NemotronOmniAudioConfig,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let norm_weight = weights
            .get(&format!("{prefix}.norm.weight"))
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| {
                format!("Missing audio projector RMSNorm weight: {prefix}.norm.weight")
            })?;
        let norm = RMSNorm::new(norm_weight, config.rms_norm_eps);
        let linear1 =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.linear1"), group_size, bits)?;
        let linear2 =
            UnifiedLinear::from_weights(weights, &format!("{prefix}.linear2"), group_size, bits)?;
        Ok(Self {
            norm,
            linear1,
            linear2,
        })
    }

    /// `x: [B, T, hidden_size]` → `[B, T, llm_hidden_size]`.
    pub fn forward(&self, x: &MlxArray) -> UniquePtr<MlxArray> {
        let h = self.norm.forward(x);
        let h = self.linear1.forward(&h);
        let h = mlxcel_core::utils::relu_squared(&h);
        self.linear2.forward(&h)
    }

    /// Compute dtype of the projector — i.e. the dtype that audio
    /// embeddings are produced in and that `extract_audio_features`
    /// must cast its mel input to.
    ///
    /// Reads from the RMSNorm scale tensor which is loaded once from
    /// the checkpoint at the same compute precision as the rest of the
    /// model (no per-call allocation, unlike running an embedding
    /// lookup just to read the resulting dtype).
    pub fn compute_dtype(&self) -> i32 {
        mlxcel_core::array_dtype(&self.norm.weight)
    }
}
