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

//! Offline (whole-utterance) Nemotron VoiceChat timeline.
//!
//! Port of `VoiceChatSession.generate` in upstream
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/nemotron_voicechat/session.py.
//! One 80 ms frame of user audio, or one system-prompt token, occupies one
//! timeline position; at every position the language model emits a text and
//! a function token and EAR-TTS emits 31 codec codes, so the answer audio is
//! exactly as long as the input timeline.
//!
//! The reference defaults its offline path to full-history recomputation of
//! the language model at every position; this port keeps the persistent
//! Nemotron-H caches instead (the reference's `use_language_cache=True`
//! mode), which produces identical tokens and codes on the reference and
//! bounds the per-position work.
//!
//! Used by: `mlxcel generate --audio` (see `commands/generate_voicechat.rs`)

use mlxcel_core::{MlxArray, UniquePtr};

use super::model::NemotronVoiceChatModel;
use super::tts::{TtsCaches, TtsPrompt};

/// Result of one offline VoiceChat turn.
#[derive(Debug, Clone)]
pub struct VoiceChatResult {
    /// Assistant text (pad, silence, BOS and EOS ids removed).
    pub text: String,
    /// Function-channel text, same filtering.
    pub function_text: String,
    /// RNNT transcript of the user audio, when the checkpoint ships a
    /// transcript vocabulary.
    pub user_transcript: Option<String>,
    /// Assistant speech, mono f32 at [`Self::sample_rate`].
    pub audio: Vec<f32>,
    pub sample_rate: u32,
    /// Text ids per audio frame (prompt prefix removed).
    pub text_tokens: Vec<i32>,
    /// Function ids per audio frame.
    pub function_tokens: Vec<i32>,
    /// Codec codes per audio frame, `num_quantizers` each, after control
    /// codes were replaced with silence.
    pub audio_codes: Vec<Vec<i32>>,
}

/// Longest silence `generate_offline` appends after the input.
pub const MAX_EXTRA_DECODING_SECONDS: f32 = 600.0;

/// Longest input (after the appended silence) one offline turn accepts. The
/// encoder builds a dense `[T, T]` attention mask over the whole utterance,
/// as the reference does, so the offline path is bounded; long sessions
/// belong to the streaming path.
pub const MAX_OFFLINE_INPUT_SECONDS: f32 = 1200.0;

pub(crate) use super::tts::host_i32;

pub(crate) fn host_f32(arr: &MlxArray) -> Vec<f32> {
    let arr = mlxcel_core::astype(arr, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&arr);
    mlxcel_core::array_to_raw_bytes(&arr)
        .chunks_exact(4)
        .map(|b| f32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Row `index` of a `[1, T, D]` array as `[1, 1, D]`.
pub(crate) fn frame_row(arr: &MlxArray, index: i32) -> UniquePtr<MlxArray> {
    let s = mlxcel_core::array_shape(arr);
    mlxcel_core::slice(arr, &[0, index, 0], &[1, index + 1, s[2]])
}

impl NemotronVoiceChatModel {
    /// Encode `(prompt_frames + 1)` codec frames of silence and build the
    /// Aria warmup prompt (`VoiceChatSession._tts_prompt`).
    pub(crate) fn tts_prompt(&self) -> Result<TtsPrompt, String> {
        let frames = self.assets.prompt_frames();
        let samples = (frames + 1) * self.codec.waveform_to_token_ratio();
        let zeros = mlxcel_core::zeros(&[1, 1, samples as i32], mlxcel_core::dtype::FLOAT32);
        let codes = self.codec.encode(&zeros)?;
        let codes = mlxcel_core::transpose_axes(&codes, &[0, 2, 1]);
        TtsPrompt::from_codec_codes(&codes, frames, &self.tts_config, self.config.pad_token_id)
    }

    /// Fresh TTS caches primed with the Aria prompt; returns them with the
    /// `previous_code` of the first step.
    pub(crate) fn warm_tts(&self) -> Result<(TtsCaches, UniquePtr<MlxArray>), String> {
        let prompt = self.tts_prompt()?;
        let mut caches = self.tts.make_caches();
        let hidden = self.tts.warmup(
            &prompt.codes,
            &prompt.subword_ids,
            &prompt.subword_mask,
            &prompt.audio_mask,
            Some(&self.assets.aria_latent),
            &mut caches,
        )?;
        mlxcel_core::eval(&hidden);
        Ok((caches, prompt.last_frame()))
    }

    /// `[BOS] + encode(prompt) + [EOS]`, or nothing for an empty prompt.
    pub(crate) fn system_prompt_ids(&self, prompt: Option<&str>) -> Result<Vec<i32>, String> {
        let text = prompt.unwrap_or(&self.config.default_system_prompt);
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        let ids = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| format!("nemotron_voicechat: system prompt tokenization: {e}"))?;
        let mut out = Vec::with_capacity(ids.len() + 2);
        out.push(self.config.bos_token_id);
        out.extend(ids.into_iter().map(|id| id as i32));
        out.push(self.config.eos_token_id);
        Ok(out)
    }

    /// Decode assistant or function ids with the special ids removed.
    pub(crate) fn decode_channel(&self, ids: &[i32]) -> Result<String, String> {
        let special = self.config.special_text_ids();
        let kept: Vec<u32> = ids
            .iter()
            .filter(|id| !special.contains(id))
            .map(|&id| id as u32)
            .collect();
        self.tokenizer
            .decode(&kept, false)
            .map_err(|e| format!("nemotron_voicechat: decode: {e}"))
    }

    /// Run one offline turn over 16 kHz mono `waveform`.
    ///
    /// `extra_decoding_seconds` of silence are appended so the model can
    /// finish answering; `system_prompt` `None` uses the checkpoint default
    /// (empty for the released checkpoints); `seed` seeds MLX's global RNG,
    /// which drives the EAR-TTS sampling noise.
    pub fn generate_offline(
        &self,
        waveform: &[f32],
        system_prompt: Option<&str>,
        extra_decoding_seconds: f32,
        seed: u64,
    ) -> Result<VoiceChatResult, String> {
        if !(extra_decoding_seconds.is_finite()
            && (0.0..=MAX_EXTRA_DECODING_SECONDS).contains(&extra_decoding_seconds))
        {
            return Err(format!(
                "extra_decoding_seconds must be between 0 and {MAX_EXTRA_DECODING_SECONDS}"
            ));
        }
        mlxcel_core::random_seed(seed);
        let rate = self.config.input_sample_rate as f32;
        let total_seconds = waveform.len() as f32 / rate + extra_decoding_seconds;
        if total_seconds > MAX_OFFLINE_INPUT_SECONDS {
            return Err(format!(
                "nemotron_voicechat: offline input of {total_seconds:.0} s exceeds the \
                 {MAX_OFFLINE_INPUT_SECONDS} s limit"
            ));
        }
        let mut samples = waveform.to_vec();
        samples.resize(
            samples.len() + (extra_decoding_seconds * rate).round() as usize,
            0.0,
        );

        let mel = crate::audio::nemotron_mel::log_mel_array(&samples, &self.mel_args)?;
        let mel_frames = mlxcel_core::array_shape(&mel)[1] as usize;
        let perception = self.perception.forward(&mel, mel_frames)?;
        let audio_frames = perception.length;
        if audio_frames == 0 {
            return Err("nemotron_voicechat: the input is shorter than one 80 ms frame".into());
        }
        mlxcel_core::eval(&perception.projected);
        mlxcel_core::eval(&perception.encoded);

        let prompt_ids = self.system_prompt_ids(system_prompt)?;
        let prompt_frames = prompt_ids.len();
        let prompt_embeds = if prompt_ids.is_empty() {
            None
        } else {
            let embeds = self.lm.embed(&prompt_ids);
            Some(mlxcel_core::astype(
                &embeds,
                mlxcel_core::array_dtype(&perception.projected),
            ))
        };
        let timeline = prompt_frames + audio_frames;
        let pad = self.config.pad_token_id;
        let eos = self.config.eos_token_id;
        let q = self.tts_config.num_quantizers;

        let (mut tts_caches, mut previous_code) = self.warm_tts()?;
        let silence = self.assets.silence_frame();
        let mut lm_caches = self.lm.make_caches();
        let mut text = vec![pad; timeline];
        let mut function = vec![pad; timeline];
        let mut codes = vec![vec![0_i32; q]; timeline];

        for t in 0..timeline {
            let audio_row = match &prompt_embeds {
                Some(embeds) if t < prompt_frames => frame_row(embeds, t as i32),
                _ => frame_row(&perception.projected, (t - prompt_frames) as i32),
            };
            let (prev_text, prev_function) = if t == 0 {
                (pad, pad)
            } else {
                (text[t - 1], function[t - 1])
            };
            let fused = self.lm.fused_input(prev_text, &audio_row, prev_function);
            let out = self.lm.step(&fused, &mut lm_caches);
            if t >= prompt_frames {
                text[t] = out.text;
                function[t] = out.function;
            }
            // Position 0 only warms the language model; every later
            // position, including the system-prompt prefix, advances EAR-TTS.
            if t == 0 {
                continue;
            }
            if text[t] == eos {
                previous_code = mlxcel_core::copy(&silence);
            }
            let step = self.tts.step(&previous_code, text[t], &mut tts_caches)?;
            codes[t] = host_i32(&step.codes);
            previous_code = step.codes;
        }

        let text = text.split_off(prompt_frames);
        let function = function.split_off(prompt_frames);
        let codes = codes.split_off(prompt_frames);
        let flat: Vec<i32> = codes.iter().flatten().copied().collect();
        let frames = codes.len() as i32;
        let code_arr = mlxcel_core::from_slice_i32(&flat, &[1, frames, q as i32]);
        let clean = self.assets.replace_control_codes(&code_arr);
        let clean_host = host_i32(&clean);
        let decoded = self
            .codec
            .decode(&mlxcel_core::transpose_axes(&clean, &[0, 2, 1]))?;
        let audio = host_f32(&decoded);

        let user_transcript = if self.rnnt_vocabulary.is_empty() {
            None
        } else {
            Some(self.rnnt.transcribe(
                &perception.encoded,
                audio_frames,
                self.max_symbols,
                &self.rnnt_vocabulary,
            )?)
        };
        Ok(VoiceChatResult {
            text: self.decode_channel(&text)?,
            function_text: self.decode_channel(&function)?,
            user_transcript,
            audio,
            sample_rate: self.config.output_sample_rate,
            text_tokens: text,
            function_tokens: function,
            audio_codes: clean_host.chunks(q).map(<[i32]>::to_vec).collect(),
        })
    }
}
