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

//! Gated audio/text fusion feeding the EAR-TTS backbone.
//!
//! Ports `GatedProjectedSumRMSNorm` from
//! `mlx_vlm/models/nemotron_voicechat/tts.py`:
//! `final_norm(sigmoid(s) * (g * audio_proj(audio / Q) + (1 - g) * text_proj(text)))`
//! with `g = sigmoid(gate)`. The gate and residual scale are sigmoid-ed in
//! their stored dtype and then cast to the audio branch's dtype, which is
//! `f32` at runtime because the code embeddings are summed in `f32`. The
//! bf16 text branch is cast to that dtype as well before it is mixed in: the
//! reference promotes it there, and CUDA builds would otherwise resolve the
//! mix to bf16 (issue #2109). The cast is exact, so upstream promotion sees
//! the same values.

use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::norm_mlp::{OffsetRmsNorm, linear_forward, scalar, weight_with_shape};

/// `GatedProjectedSumRMSNorm` (key tree `gated_fusion_audio_text`).
pub struct GatedFusion {
    audio_proj: UnifiedLinear,
    text_proj: UnifiedLinear,
    gate: UniquePtr<MlxArray>,
    residual_scale: UniquePtr<MlxArray>,
    final_norm: OffsetRmsNorm,
    num_codebooks: usize,
}

impl GatedFusion {
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        hidden: usize,
        num_codebooks: usize,
        eps: f32,
        group_size: i32,
        bits: i32,
    ) -> Result<Self, String> {
        let lin = |name: &str| {
            UnifiedLinear::from_weights(weights, &format!("{prefix}.{name}"), group_size, bits)
        };
        Ok(Self {
            audio_proj: lin("audio_proj")?,
            text_proj: lin("text_proj")?,
            gate: weight_with_shape(weights, &format!("{prefix}.gate"), &[hidden as i32])?,
            residual_scale: weight_with_shape(weights, &format!("{prefix}.residual_scale"), &[])?,
            final_norm: OffsetRmsNorm::from_weights(
                weights,
                &format!("{prefix}.final_norm"),
                hidden,
                eps,
            )?,
            num_codebooks,
        })
    }

    /// `audio`: code embeddings `[B, L, H]`; `text`: subword condition
    /// `[B, L, H]`.
    pub fn forward(&self, audio: &MlxArray, text: &MlxArray) -> UniquePtr<MlxArray> {
        let act = mlxcel_core::array_dtype(audio);
        let audio = mlxcel_core::divide(audio, &scalar(self.num_codebooks as f64, act));
        let audio = linear_forward(&self.audio_proj, &audio);
        let act = mlxcel_core::array_dtype(&audio);
        let text = mlxcel_core::astype(&linear_forward(&self.text_proj, text), act);
        let gate = mlxcel_core::astype(&mlxcel_core::sigmoid(&self.gate), act);
        let scale = mlxcel_core::astype(&mlxcel_core::sigmoid(&self.residual_scale), act);
        let one_minus = mlxcel_core::subtract(&scalar(1.0, act), &gate);
        let mixed = mlxcel_core::add(
            &mlxcel_core::multiply(&gate, &audio),
            &mlxcel_core::multiply(&one_minus, &text),
        );
        self.final_norm
            .forward(&mlxcel_core::multiply(&scale, &mixed))
    }
}
