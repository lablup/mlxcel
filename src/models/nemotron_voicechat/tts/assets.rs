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

//! Speech-decoder buffers and the warmup prompt.
//!
//! Ports the non-module parts of `SpeechDecoder` (the `Aria` speaker latent,
//! `codec_silence_tokens`, `control_codes`) from
//! `mlx_vlm/models/nemotron_voicechat/tts.py`, and `_tts_prompt` /
//! `_replace_control_codes` from `session.py`.
//!
//! The only checkpoint conversion the reference applies to these keys is
//! the rename `_control_codes` -> `control_codes` (`SpeechDecoder.sanitize`);
//! both spellings are accepted. The integer buffers are stored as int64 in
//! the checkpoint and held as int32 here (their values are codec ids).

use mlxcel_core::dtype;
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::config::TtsConfig;
use super::norm_mlp::weight;
use super::subword::to_host_i32;

/// `SpeechDecoder` buffers under `tts_model.`.
pub struct SpeechDecoderAssets {
    /// `audio_prompt_latents.Aria`, `[1, frames, hidden]`.
    pub aria_latent: UniquePtr<MlxArray>,
    /// Codec codes of silence, `[num_quantizers]`.
    pub codec_silence_tokens: Vec<i32>,
    /// Codec ids that must never reach the codec decoder.
    pub control_codes: Vec<i32>,
}

impl SpeechDecoderAssets {
    /// Load from `{prefix}.*` (the checkpoint uses `tts_model`).
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &TtsConfig,
    ) -> Result<Self, String> {
        let aria_latent = weight(weights, &format!("{prefix}.audio_prompt_latents.Aria"))?;
        let shape = mlxcel_core::array_shape(&aria_latent);
        if shape.len() != 3
            || shape[0] != 1
            || shape[1] < 2
            || shape[2] != config.hidden_size as i32
        {
            return Err(format!(
                "{prefix}.audio_prompt_latents.Aria: expected [1, frames >= 2, {}], got {shape:?}",
                config.hidden_size
            ));
        }
        let silence_arr = weight(weights, &format!("{prefix}.codec_silence_tokens"))?;
        let silence = to_host_i32(&silence_arr);
        if silence.len() != config.num_quantizers {
            return Err(format!(
                "{prefix}.codec_silence_tokens: expected {} entries, got {}",
                config.num_quantizers,
                silence.len()
            ));
        }
        let control_key = [
            format!("{prefix}.control_codes"),
            format!("{prefix}._control_codes"),
        ]
        .into_iter()
        .find(|k| weights.contains_key(k))
        .ok_or_else(|| format!("Weight not found: {prefix}.control_codes"))?;
        let control_arr = weight(weights, &control_key)?;
        let control_codes = to_host_i32(&control_arr);
        Ok(Self {
            aria_latent,
            codec_silence_tokens: silence,
            control_codes,
        })
    }

    /// Speaker prompt length in frames.
    pub fn prompt_frames(&self) -> usize {
        mlxcel_core::array_shape(&self.aria_latent)[1] as usize
    }

    /// Silence codes as an int32 `[1, 1, Q]` array (the frame the session
    /// feeds back after the LLM emits EOS).
    pub fn silence_frame(&self) -> UniquePtr<MlxArray> {
        let q = self.codec_silence_tokens.len() as i32;
        mlxcel_core::from_slice_i32(&self.codec_silence_tokens, &[1, 1, q])
    }

    /// `_replace_control_codes`: every entry equal to a control code becomes
    /// the silence code of its codebook. `codes`: int `[..., Q]`.
    pub fn replace_control_codes(&self, codes: &MlxArray) -> UniquePtr<MlxArray> {
        let shape = mlxcel_core::array_shape(codes);
        let codes_dtype = mlxcel_core::array_dtype(codes);
        let mut mask = mlxcel_core::zeros(&shape, dtype::BOOL);
        for &token in &self.control_codes {
            let token = mlxcel_core::full_f32(&[], token as f32, codes_dtype);
            mask = mlxcel_core::logical_or(&mask, &mlxcel_core::equal(codes, &token));
        }
        let silence = mlxcel_core::astype(&self.silence_frame(), codes_dtype);
        let silence = mlxcel_core::broadcast_to(&silence, &shape);
        mlxcel_core::where_cond(&mask, &silence, codes)
    }
}

/// Inputs of [`super::RvqEarTtsModel::warmup`] for the cached speaker,
/// built the way `VoiceChatSession._tts_prompt` builds them.
pub struct TtsPrompt {
    /// int32 `[1, frames, Q]`.
    pub codes: UniquePtr<MlxArray>,
    pub subword_ids: Vec<i32>,
    pub subword_mask: Vec<bool>,
    pub audio_mask: Vec<bool>,
}

impl TtsPrompt {
    /// `codec_codes`: the codec's encoding of `frames + 1` frames of silence
    /// (`codec.encode(zeros)` transposed to `[1, frames + 1, Q]`).
    ///
    /// Frame 0 and frame `frames - 1` are overwritten with the mask code
    /// (`codebook_size`) and the trailing frame is dropped; the subword
    /// channel is `pad_token_id` everywhere with only the last two frames
    /// valid, and only the last frame is audio.
    pub fn from_codec_codes(
        codec_codes: &MlxArray,
        frames: usize,
        config: &TtsConfig,
        pad_token_id: i32,
    ) -> Result<Self, String> {
        let q = config.num_quantizers;
        let shape = mlxcel_core::array_shape(codec_codes);
        if frames < 2 || shape != [1, frames as i32 + 1, q as i32] {
            return Err(format!(
                "silent prompt codec produced unexpected shape {shape:?}, expected [1, {}, {q}]",
                frames + 1
            ));
        }
        let mut host = to_host_i32(codec_codes);
        host.truncate(frames * q);
        let mask_code = config.codebook_size as i32;
        for frame in [0, frames - 1] {
            host[frame * q..(frame + 1) * q].fill(mask_code);
        }
        let mut subword_mask = vec![false; frames];
        subword_mask[frames - 2..].fill(true);
        let mut audio_mask = vec![false; frames];
        audio_mask[frames - 1] = true;
        Ok(Self {
            codes: mlxcel_core::from_slice_i32(&host, &[1, frames as i32, q as i32]),
            subword_ids: vec![pad_token_id; frames],
            subword_mask,
            audio_mask,
        })
    }

    /// The last prompt frame, the `previous_code` of the first step.
    pub fn last_frame(&self) -> UniquePtr<MlxArray> {
        let s = mlxcel_core::array_shape(&self.codes);
        mlxcel_core::slice(&self.codes, &[0, s[1] - 1, 0], &[1, s[1], s[2]])
    }
}
