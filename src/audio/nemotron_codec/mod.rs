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

//! NemotronLabs VoiceChat neural audio codec (22.05 kHz, 31 codebooks).
//!
//! Faithful port of `NemotronVoiceChatCodec` from
//! `mlx_audio/codec/models/nemotron_voicechat/codec.py` (mlx-audio), the
//! codec embedded in the VoiceChat checkpoint under `tts_model.audio_codec`:
//!
//! - encode: waveform -> STFT (n_fft 16, hop 4) -> ConvNeXt encoder with
//!   strided downsampling (7, 7, 9) -> residual VQ -> `[B, 31, T]` codes,
//!   one frame per 1764 samples;
//! - decode: codes -> summed codebook rows -> ConvNeXt decoder with
//!   transposed-conv upsampling -> magnitude/phase -> iSTFT -> `[B, 1, T * 1764]`;
//! - streaming: [`NemotronCodec::decode_step`] with a [`CausalConv1dCache`]
//!   decodes new frames without replaying history.
//!
//! Streaming contract (from the reference, not index equality): each
//! non-flush step returns exactly `waveform_to_token_ratio` samples, the
//! flush step returns 8 more, and the concatenated stream equals
//! [`NemotronCodec::decode`] of all frames delayed by
//! [`NemotronCodec::stream_delay_samples`] (8) samples, except the first and
//! last 6 samples of the full decode, where the full decode's overlap-add
//! window envelope covers fewer frames than the stream's (which counts the
//! zero pre-roll and the flush tail). The reference behaves the same way.
//!
//! Used by: NemotronLabs VoiceChat TTS (offline decode and TTS prompt codes).

mod cache;
mod config;
mod layers;
mod network;
mod prvq;
mod stft;

#[cfg(test)]
mod codec_tests;

pub use cache::CausalConv1dCache;
pub use config::CodecConfig;

use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use cache::CacheKey;
use layers::scalar_like;
use network::{AudioDecoder, AudioEncoder};
use prvq::ResidualQuantizer;

/// The VoiceChat codec with loaded weights.
pub struct NemotronCodec {
    config: CodecConfig,
    encoder: AudioEncoder,
    decoder: AudioDecoder,
    prvq: ResidualQuantizer,
}

impl NemotronCodec {
    /// Load from a weight map whose codec tensors live under `prefix`
    /// (`tts_model.audio_codec` in the VoiceChat checkpoint). Conv kernels
    /// may be in PyTorch or MLX layout; variances are not required.
    pub fn from_weights(
        weights: &WeightMap,
        prefix: &str,
        config: &CodecConfig,
    ) -> Result<Self, String> {
        config.validate()?;
        let prefix = prefix.trim_end_matches('.');
        let join = |name: &str| {
            if prefix.is_empty() {
                name.to_string()
            } else {
                format!("{prefix}.{name}")
            }
        };
        Ok(Self {
            config: config.clone(),
            encoder: AudioEncoder::from_weights(weights, &join("encoder"), config)?,
            decoder: AudioDecoder::from_weights(weights, &join("decoder"), config)?,
            prvq: ResidualQuantizer::from_weights(weights, &join("prvq"), config)?,
        })
    }

    pub fn config(&self) -> &CodecConfig {
        &self.config
    }

    pub fn sample_rate(&self) -> usize {
        self.config.sample_rate
    }

    pub fn frame_rate(&self) -> f64 {
        self.config.frame_rate()
    }

    /// Waveform samples per codec frame (1764 for the released codec).
    pub fn waveform_to_token_ratio(&self) -> usize {
        self.config.waveform_to_token_ratio()
    }

    pub fn num_quantizers(&self) -> usize {
        self.config.num_quantizers
    }

    fn half_spec_padding(&self) -> usize {
        let half = (self.config.n_fft - self.config.hop_length) / 2;
        half.div_ceil(self.config.hop_length)
    }

    /// Samples by which the streaming output lags [`Self::decode`] (8).
    pub fn stream_delay_samples(&self) -> usize {
        self.half_spec_padding() * self.config.hop_length
    }

    /// Waveform `[B, 1, N]` or `[B, N]` -> f32 latents `[B, T, latent_dim]`.
    pub fn encode_latents(&self, waveform: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let shape = mlxcel_core::array_shape(waveform);
        let wave = match shape.len() {
            3 if shape[1] == 1 => mlxcel_core::reshape(waveform, &[shape[0], shape[2]]),
            3 => return Err("only mono waveforms are supported".to_string()),
            2 => mlxcel_core::copy(waveform),
            _ => {
                return Err(format!(
                    "waveform must be [B, 1, N] or [B, N], got {shape:?}"
                ));
            }
        };
        let (real, imag) = stft::spectrogram(&wave, self.config.n_fft, self.config.hop_length)?;
        let features = mlxcel_core::concatenate(&real, &imag, -1);
        self.encoder.forward(&features)
    }

    /// Waveform `[B, 1, N]` or `[B, N]` -> int32 codes `[B, Q, N / 1764]`.
    pub fn encode(&self, waveform: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let latents = self.encode_latents(waveform)?;
        Ok(self.prvq.encode(&latents))
    }

    /// Codes `[B, Q, T]` (Q <= num_quantizers, ids < codebook_size) ->
    /// f32 waveform `[B, 1, T * 1764]`.
    pub fn decode(&self, codes: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        self.decode_impl(codes, None, false)
    }

    /// Decode new frames continuing the stream held in `cache`. `flush`
    /// emits the trailing samples and drops the cache entries; see the
    /// module docs for the output contract.
    pub fn decode_with_cache(
        &self,
        codes: &MlxArray,
        cache: &mut CausalConv1dCache,
        flush: bool,
    ) -> Result<UniquePtr<MlxArray>, String> {
        self.decode_impl(codes, Some(cache), flush)
    }

    /// Streaming decode of newly generated codes (`[1, Q, 1]` per frame in
    /// the VoiceChat loop) -> `[B, 1, 1764]` (`[B, 1, 1772]` when `flush`).
    pub fn decode_step(
        &self,
        codes: &MlxArray,
        cache: &mut CausalConv1dCache,
        flush: bool,
    ) -> Result<UniquePtr<MlxArray>, String> {
        self.decode_impl(codes, Some(cache), flush)
    }

    fn decode_impl(
        &self,
        codes: &MlxArray,
        cache: Option<&mut CausalConv1dCache>,
        flush: bool,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let codes = self.validate_codes(codes)?;
        crate::audio::stage_probe::mark("codec.validate", &[]);
        let latents = self.prvq.decode(&codes);
        crate::audio::stage_probe::mark("codec.prvq", &[&latents]);
        let waveform = self.decode_latents(&latents, cache, flush)?;
        crate::audio::stage_probe::mark("codec.decoder_istft", &[&waveform]);
        Ok(waveform)
    }

    /// Check `[B, Q, T]` shape and code range on the host (codes are tiny),
    /// so an out-of-range id (e.g. the 1024 mask code) is an error instead
    /// of an undefined GPU gather. Returns the codes as int32.
    fn validate_codes(&self, codes: &MlxArray) -> Result<UniquePtr<MlxArray>, String> {
        let shape = mlxcel_core::array_shape(codes);
        if shape.len() != 3 {
            return Err(format!("codes must have shape (B, Q, T), got {shape:?}"));
        }
        if shape[1] as usize > self.prvq.num_quantizers() || shape[1] == 0 {
            return Err(format!(
                "received {} quantizers, expected 1..={}",
                shape[1],
                self.prvq.num_quantizers()
            ));
        }
        if shape[0] == 0 || shape[2] == 0 {
            return Err(format!("codes must be non-empty, got {shape:?}"));
        }
        let ids = mlxcel_core::astype(codes, mlxcel_core::dtype::INT32);
        mlxcel_core::try_eval(&ids).map_err(|e| format!("codec codes eval failed: {e}"))?;
        crate::audio::stage_probe::count_sync(1);
        let limit = self.config.codebook_size as i32;
        let bytes = mlxcel_core::array_to_raw_bytes(&ids);
        if let Some(bad) = bytes
            .chunks_exact(4)
            .map(|c| i32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
            .find(|&v| !(0..limit).contains(&v))
        {
            return Err(format!("code id {bad} outside codebook range 0..{limit}"));
        }
        Ok(ids)
    }

    /// Latents `[B, T, latent_dim]` -> waveform `[B, 1, samples]` (f32).
    pub(crate) fn decode_latents(
        &self,
        latents: &MlxArray,
        mut cache: Option<&mut CausalConv1dCache>,
        flush: bool,
    ) -> Result<UniquePtr<MlxArray>, String> {
        let cfg = &self.config;
        let features = self.decoder.forward(latents, cache.as_deref_mut(), flush)?;
        let fshape = mlxcel_core::array_shape(&features);
        let (batch, frames) = (fshape[0], fshape[1]);
        let bins = cfg.num_bins() as i32;
        let logits = mlxcel_core::slice(&features, &[0, 0, 0], &[batch, frames, bins]);
        let phase = mlxcel_core::slice(&features, &[0, 0, bins], &[batch, frames, 2 * bins]);

        // magnitude = 100 * exp(-softplus(-logits + ln 100)), weak scalars.
        let max_magnitude = 100.0f32;
        let shifted = mlxcel_core::add(
            &mlxcel_core::negative(&logits),
            &scalar_like(max_magnitude.ln(), &logits),
        );
        let decay = mlxcel_core::exp(&mlxcel_core::negative(&mlxcel_core::softplus(&shifted)));
        let magnitude = mlxcel_core::multiply(&scalar_like(max_magnitude, &decay), &decay);
        let real = mlxcel_core::multiply(&magnitude, &mlxcel_core::cos(&phase));
        let imag = mlxcel_core::multiply(&magnitude, &mlxcel_core::sin(&phase));
        // DC and Nyquist bins of an rFFT are real-valued.
        let inner = mlxcel_core::slice(&imag, &[0, 0, 1], &[batch, frames, bins - 1]);
        let edge = mlxcel_core::zeros(&[batch, frames, 1], mlxcel_core::array_dtype(&imag));
        let imag = mlxcel_core::concatenate_many(&[&edge, &inner, &edge], -1);

        let half_spec = self.half_spec_padding();
        let streaming = cache.is_some();
        let (real, imag) = match cache {
            Some(cache) => {
                let spec = 2 * half_spec;
                let real = cache.update(&real, CacheKey::IstftReal, spec, flush)?;
                let imag = cache.update(&imag, CacheKey::IstftImag, spec, flush)?;
                if flush {
                    let rshape = mlxcel_core::array_shape(&real);
                    let tail = mlxcel_core::zeros(
                        &[rshape[0], half_spec as i32, rshape[2]],
                        mlxcel_core::array_dtype(&real),
                    );
                    (
                        mlxcel_core::concatenate(&real, &tail, 1),
                        mlxcel_core::concatenate(&imag, &tail, 1),
                    )
                } else {
                    (real, imag)
                }
            }
            None => (real, imag),
        };
        let waveform = stft::istft(&real, &imag, cfg.n_fft, cfg.hop_length)?;

        let length = mlxcel_core::array_shape(&waveform)[1];
        let pad_left = ((cfg.n_fft - cfg.hop_length) / 2) as i32;
        let pad_right = (cfg.n_fft - cfg.hop_length) as i32 - pad_left;
        let trim = if streaming {
            (half_spec * cfg.hop_length) as i32
        } else {
            0
        };
        let (start, stop) = (pad_left + trim, length - pad_right - trim);
        if stop < start {
            return Err(format!("codec decode produced too few samples ({length})"));
        }
        let waveform = mlxcel_core::slice(&waveform, &[0, start], &[batch, stop]);
        Ok(mlxcel_core::reshape(&waveform, &[batch, 1, stop - start]))
    }
}
