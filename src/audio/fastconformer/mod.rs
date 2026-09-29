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

//! Cache-aware FastConformer speech encoder (offline forward).
//!
//! Port of `Conformer` in `mlx_audio/stt/models/nemotron_asr/conformer.py`
//! as used by NemotronLabs VoiceChat (`stt_model.perception.encoder`): causal
//! depthwise-striding subsampling (8x in time), a Transformer-XL relative
//! positional encoding (`xscaling` off), and `n_layers` Conformer blocks with
//! the `chunked_limited` attention mask. The VoiceChat `Perception` wrapper
//! (encoder + `proj` linear into the LLM width) lives in [`perception`]; the
//! cache-aware incremental encoder is [`ConformerStreamingState`].
//!
//! Numerics follow the reference: weights stay in the checkpoint dtype (bf16)
//! and the f32 mel input promotes every op to f32, so the encoder output is f32.
//!
//! This is a separate module from `nemotron_h_nano_omni::encoder` because that
//! Parakeet port uses symmetric subsampling, BatchNorm, biased projections and
//! HF-style key names; only generic helpers are shared.
//!
//! Used by: NemotronLabs VoiceChat speech perception.

mod attention;
mod block;
pub mod config;
mod layers;
pub mod perception;
mod streaming;
mod subsampling;

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

pub use attention::{
    NEG_INF, RelPositionMultiHeadAttention, chunked_limited_mask, chunked_limited_visible,
    rel_pos_embedding, rel_shift,
};
pub use block::{BlockStreamOutput, ConformerBlock};
pub use config::{ConformerArgs, ConvContextSize};
pub use perception::VoiceChatPerception;
pub use streaming::{ConformerStreamingState, PRE_ENCODE_MEL_CACHE};
pub use subsampling::CausalDwStridingSubsampling;

pub struct FastConformerEncoder {
    pre_encode: CausalDwStridingSubsampling,
    layers: Vec<ConformerBlock>,
    args: ConformerArgs,
}

impl FastConformerEncoder {
    /// Load from `{prefix}.pre_encode.*` and `{prefix}.layers.{i}.*`
    /// (e.g. `prefix = "stt_model.perception.encoder"`).
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        args: &ConformerArgs,
    ) -> Result<Self, String> {
        args.validate()?;
        let pre_encode = CausalDwStridingSubsampling::from_weights(
            weights,
            &format!("{prefix}.pre_encode"),
            args,
        )?;
        let layers = (0..args.n_layers)
            .map(|i| ConformerBlock::from_weights(weights, &format!("{prefix}.layers.{i}"), args))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            pre_encode,
            layers,
            args: args.clone(),
        })
    }

    pub fn args(&self) -> &ConformerArgs {
        &self.args
    }

    /// Encoder frame count for `mel_frames` input frames.
    pub fn output_length(&self, mel_frames: usize) -> usize {
        self.pre_encode.output_length(mel_frames)
    }

    /// `mel: [B, T, feat_in]` (f32) to `([B, T', d_model], T')` using the
    /// first configured attention context (`[70, 0]` for VoiceChat).
    pub fn forward(
        &self,
        mel: &MlxArray,
        length: usize,
    ) -> Result<(UniquePtr<MlxArray>, usize), String> {
        self.forward_with_context(mel, length, self.args.default_att_context())
    }

    /// Like [`Self::forward`] with an explicit `[left, right]` attention context.
    pub fn forward_with_context(
        &self,
        mel: &MlxArray,
        length: usize,
        att_context: [i64; 2],
    ) -> Result<(UniquePtr<MlxArray>, usize), String> {
        let shape = mlxcel_core::array_shape(mel);
        if shape.len() != 3 || shape[1] < 1 || shape[2] != self.args.feat_in as i32 {
            return Err(format!(
                "FastConformer expects mel [B, T >= 1, {}], got {shape:?}",
                self.args.feat_in
            ));
        }
        let out_length = self.output_length(length);
        let mut x = self.pre_encode.forward(mel)?;
        if self.args.xscaling {
            x = mlxcel_core::multiply_scalar(&x, (self.args.d_model as f32).sqrt());
        }
        let dtype = mlxcel_core::array_dtype(&x);
        let frames = mlxcel_core::array_shape(&x)[1] as usize;
        let pos_emb = mlxcel_core::astype(&rel_pos_embedding(frames, self.args.d_model), dtype);
        let mask = if self.args.att_context_style == "chunked_limited" {
            let [left, right] = att_context;
            Some(mlxcel_core::astype(
                &chunked_limited_mask(frames, left, right),
                dtype,
            ))
        } else {
            None
        };
        for layer in &self.layers {
            x = layer.forward(&x, &pos_emb, mask.as_deref())?;
        }
        Ok((x, out_length))
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod streaming_tests;
