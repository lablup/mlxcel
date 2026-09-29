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

//! VoiceChat speech perception: FastConformer encoder plus the `proj` linear
//! into the LLM hidden width.
//!
//! Port of `Perception` in `mlx_vlm/models/nemotron_voicechat/model.py`. The
//! projected frames feed the LLM audio channel; the raw encoder frames feed the
//! RNNT transcript branch.

use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::{ConformerArgs, FastConformerEncoder};

/// Output of [`VoiceChatPerception::forward`].
pub struct PerceptionOutput {
    /// `[B, T', output_dim]` (4480 for VoiceChat), f32.
    pub projected: UniquePtr<MlxArray>,
    /// Valid encoder frames `T'`.
    pub length: usize,
    /// `[B, T', d_model]` raw encoder output (RNNT input), f32.
    pub encoded: UniquePtr<MlxArray>,
}

pub struct VoiceChatPerception {
    encoder: FastConformerEncoder,
    proj: UnifiedLinear,
}

impl VoiceChatPerception {
    /// Load `{prefix}.encoder.*` and `{prefix}.proj.*`
    /// (e.g. `prefix = "stt_model.perception"`). `quantization` is the
    /// checkpoint's `(group_size, bits)` for quantized linears.
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        args: &ConformerArgs,
        quantization: (i32, i32),
    ) -> Result<Self, String> {
        let (group_size, bits) = quantization;
        Ok(Self {
            encoder: FastConformerEncoder::from_weights(
                weights,
                &format!("{prefix}.encoder"),
                args,
                quantization,
            )?,
            proj: UnifiedLinear::from_weights(
                weights,
                &format!("{prefix}.proj"),
                group_size,
                bits,
            )?,
        })
    }

    pub fn encoder(&self) -> &FastConformerEncoder {
        &self.encoder
    }

    /// Project encoder frames `[B, T', d_model]` into the LLM width.
    pub fn project(&self, encoded: &MlxArray) -> UniquePtr<MlxArray> {
        self.proj.forward(encoded)
    }

    /// `mel: [B, T, feat_in]` with `length` valid mel frames.
    pub fn forward(&self, mel: &MlxArray, length: usize) -> Result<PerceptionOutput, String> {
        let (encoded, length) = self.encoder.forward(mel, length)?;
        Ok(PerceptionOutput {
            projected: self.project(&encoded),
            length,
            encoded,
        })
    }
}
