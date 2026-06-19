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

//! Whisper-style encoder-decoder ASR model.
//!
//! Architecture: a convolutional + transformer audio encoder feeding an
//! autoregressive text decoder whose blocks cross-attend to the encoder output.
//! The decoder is steered by multilingual transcribe/translate task tokens.
//!
//! This first port targets non-quantized (fp16/f32) MLX checkpoints and greedy
//! decoding. The weight loader also accepts the HuggingFace key layout so either
//! export of a checkpoint loads. Quantized checkpoints, beam search, and
//! word-level timestamps are tracked as follow-ups.

mod decoder;
mod decoding;
mod encoder;
pub mod layers;
pub mod tokenizer;

use std::path::Path;

use anyhow::{Result, anyhow};
use serde_json::Value;

use mlxcel_core::weights::WeightMap;

use crate::audio::whisper_mel;

use decoder::TextDecoder;
use encoder::AudioEncoder;
use tokenizer::WhisperTokenizer;

/// Encoder/decoder shape parameters. Mirrors the reference `ModelDimensions`
/// and accepts both the native and HuggingFace config field names.
#[derive(Debug, Clone)]
pub struct WhisperDims {
    pub n_mels: i32,
    pub n_audio_ctx: i32,
    pub n_audio_state: i32,
    pub n_audio_head: i32,
    pub n_audio_layer: i32,
    pub n_vocab: i32,
    pub n_text_ctx: i32,
    pub n_text_state: i32,
    pub n_text_head: i32,
    pub n_text_layer: i32,
}

fn config_i32(config: &Value, keys: &[&str]) -> Option<i32> {
    keys.iter()
        .find_map(|k| config.get(*k).and_then(Value::as_i64))
        .map(|v| v as i32)
}

impl WhisperDims {
    /// Parse from a (sanitized) `config.json` value, accepting native MLX field
    /// names or the HuggingFace `WhisperConfig` names.
    pub fn from_config(config: &Value) -> Result<Self> {
        let require = |keys: &[&str]| -> Result<i32> {
            config_i32(config, keys)
                .ok_or_else(|| anyhow!("Whisper config missing field(s): {keys:?}"))
        };
        let n_audio_state = require(&["n_audio_state", "d_model"])?;
        let n_text_state =
            config_i32(config, &["n_text_state", "d_model"]).unwrap_or(n_audio_state);
        Ok(Self {
            n_mels: config_i32(config, &["n_mels", "num_mel_bins"]).unwrap_or(80),
            n_audio_ctx: config_i32(config, &["n_audio_ctx", "max_source_positions"])
                .unwrap_or(1500),
            n_audio_state,
            n_audio_head: require(&["n_audio_head", "encoder_attention_heads"])?,
            n_audio_layer: require(&["n_audio_layer", "encoder_layers"])?,
            n_vocab: require(&["n_vocab", "vocab_size"])?,
            n_text_ctx: config_i32(config, &["n_text_ctx", "max_target_positions"]).unwrap_or(448),
            n_text_state,
            n_text_head: require(&["n_text_head", "decoder_attention_heads"])?,
            n_text_layer: require(&["n_text_layer", "decoder_layers"])?,
        })
    }
}

/// Loaded Whisper ASR model: audio encoder, text decoder, and resolved
/// tokenizer. Holds MLX weight handles, so the owning provider serializes
/// access and asserts thread-safety (see `crate::server::whisper_stt`).
pub struct WhisperModel {
    dims: WhisperDims,
    dtype: i32,
    encoder: AudioEncoder,
    decoder: TextDecoder,
    tokenizer: WhisperTokenizer,
}

impl WhisperModel {
    /// Load a Whisper checkpoint directory (config.json + safetensors +
    /// tokenizer.json).
    pub fn load(model_path: &Path) -> Result<Self> {
        let config_path = model_path.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| anyhow!("Failed to read {config_path:?}: {e}"))?;
        let config_str = super::sanitize_config_json(&config_str);
        let config: Value = serde_json::from_str(&config_str)
            .map_err(|e| anyhow!("Failed to parse Whisper config: {e}"))?;
        let dims = WhisperDims::from_config(&config)?;

        let mut weights = mlxcel_core::weights::load_weights_from_dir(model_path)
            .map_err(|e| anyhow!("Failed to load Whisper weights: {e}"))?;
        sanitize_whisper_weights(&mut weights);
        // Apple Silicon precision policy: bf16 -> f16 for non-quantized weights.
        let _ = super::convert_bf16_weights(&mut weights);

        let dtype = weights
            .get("decoder.token_embedding.weight")
            .map(|w| mlxcel_core::array_dtype(w))
            .unwrap_or(mlxcel_core::dtype::FLOAT16);

        let encoder = AudioEncoder::from_weights(&weights, &dims, dtype)
            .map_err(|e| anyhow!("Failed to build Whisper encoder: {e}"))?;
        let decoder = TextDecoder::from_weights(&weights, &dims, dtype)
            .map_err(|e| anyhow!("Failed to build Whisper decoder: {e}"))?;
        let tokenizer = WhisperTokenizer::from_dir(model_path)?;

        Ok(Self {
            dims,
            dtype,
            encoder,
            decoder,
            tokenizer,
        })
    }

    /// Transcribe (or translate) 16 kHz mono audio.
    ///
    /// Returns the concatenated text and the language code that was used. Audio
    /// longer than 30 s is processed in consecutive 30 s windows.
    ///
    /// Errors if MLX graph evaluation fails. The graph is evaluated through the
    /// fallible [`mlxcel_core::try_eval`] boundary, so an MLX failure (for
    /// example a shape or allocation error) is returned as `Err` instead of
    /// aborting the process with an uncaught C++ exception.
    pub fn transcribe(
        &self,
        audio_16k: &[f32],
        language: Option<&str>,
        translate: bool,
    ) -> Result<(String, Option<String>)> {
        let n_mels = self.dims.n_mels as usize;
        // Pad the waveform to a whole number of 30 s windows before the log-mel
        // transform so the silent tail is normalized like trained silence rather
        // than left at a raw 0.0 (see `pad_audio_to_window_multiple`).
        let audio_padded = pad_audio_to_window_multiple(audio_16k);
        let (mel, frames) = whisper_mel::log_mel_spectrogram(&audio_padded, n_mels);
        if frames == 0 {
            return Ok((String::new(), language.map(String::from)));
        }

        let mut text = String::new();
        let mut used_language: Option<String> = language.map(String::from);
        let mut seek = 0usize;
        while seek < frames {
            let window = padded_window(&mel, frames, n_mels, seek);
            let mel_arr = mlxcel_core::from_slice_f32(
                &window,
                &[1, whisper_mel::WHISPER_N_FRAMES as i32, n_mels as i32],
            );
            let mel_arr = mlxcel_core::astype(&mel_arr, self.dtype);
            let features = self.encoder.forward(&mel_arr);

            let (tokens, detected) = decoding::transcribe_segment(
                &self.decoder,
                &features,
                &self.tokenizer,
                self.dims.n_vocab,
                self.dims.n_text_ctx,
                used_language.as_deref(),
                translate,
            )?;
            if used_language.is_none() {
                used_language = detected;
            }
            text.push_str(&self.tokenizer.decode_text(&tokens));
            seek += whisper_mel::WHISPER_N_FRAMES;
        }

        Ok((text, used_language))
    }
}

/// Pad the waveform with trailing silence (zeros) up to a whole number of 30 s
/// windows.
///
/// The padding is applied to the waveform, before the log-mel transform, so the
/// silent tail flows through `log10` and the global-max clamp and lands at the
/// same normalized log floor as trained silence. Zero-filling the mel frames
/// directly (after normalization) would instead leave the tail at a raw `0.0`, a
/// low-mid value the encoder reads as spurious broadband energy, which can
/// degrade or hallucinate transcriptions of clips shorter than one window (under
/// 30 s) where the padding dominates. An empty input stays empty; the caller
/// short-circuits on zero frames.
fn pad_audio_to_window_multiple(audio: &[f32]) -> Vec<f32> {
    if audio.is_empty() {
        return Vec::new();
    }
    let windows = audio.len().div_ceil(whisper_mel::WHISPER_N_SAMPLES);
    let target = windows * whisper_mel::WHISPER_N_SAMPLES;
    let mut padded = Vec::with_capacity(target);
    padded.extend_from_slice(audio);
    padded.resize(target, 0.0);
    padded
}

/// Copy one 30 s window of mel frames starting at `seek`. The waveform is padded
/// to a whole number of windows before the log-mel transform (see
/// `pad_audio_to_window_multiple`), so in practice every window is full; the
/// trailing zero-fill here is a safety net for any final-frame remainder. Output
/// is row-major `[N_FRAMES][n_mels]`.
fn padded_window(mel: &[f32], frames: usize, n_mels: usize, seek: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; whisper_mel::WHISPER_N_FRAMES * n_mels];
    let avail = (frames - seek).min(whisper_mel::WHISPER_N_FRAMES);
    for f in 0..avail {
        let src = (seek + f) * n_mels;
        let dst = f * n_mels;
        out[dst..dst + n_mels].copy_from_slice(&mel[src..src + n_mels]);
    }
    out
}

/// Normalize checkpoint weight keys to the native layout used by this port.
///
/// MLX-native checkpoints already use the target keys and pass through
/// unchanged. HuggingFace `WhisperForConditionalGeneration` exports use a
/// different key scheme and a `[out, in, kernel]` Conv1d layout; those are
/// remapped and the conv weights transposed to `[out, kernel, in]`.
fn sanitize_whisper_weights(weights: &mut WeightMap) {
    let is_hf = weights
        .keys()
        .any(|k| k.starts_with("model.") || k.contains("encoder.layers."));
    if !is_hf {
        return;
    }

    // Ordered so more specific patterns are applied before generic ones.
    let key_map: &[(&str, &str)] = &[
        (
            "decoder.embed_positions.weight",
            "decoder.positional_embedding",
        ),
        ("encoder.layer_norm.", "encoder.ln_post."),
        ("decoder.layer_norm.", "decoder.ln."),
        ("encoder.layers.", "encoder.blocks."),
        ("decoder.layers.", "decoder.blocks."),
        (".self_attn_layer_norm.", ".attn_ln."),
        (".final_layer_norm.", ".mlp_ln."),
        (".encoder_attn_layer_norm.", ".cross_attn_ln."),
        (".fc1.", ".mlp1."),
        (".fc2.", ".mlp2."),
        (".self_attn.q_proj.", ".attn.query."),
        (".self_attn.k_proj.", ".attn.key."),
        (".self_attn.v_proj.", ".attn.value."),
        (".self_attn.out_proj.", ".attn.out."),
        (".encoder_attn.q_proj.", ".cross_attn.query."),
        (".encoder_attn.k_proj.", ".cross_attn.key."),
        (".encoder_attn.v_proj.", ".cross_attn.value."),
        (".encoder_attn.out_proj.", ".cross_attn.out."),
        ("decoder.embed_tokens.", "decoder.token_embedding."),
    ];

    let old_keys: Vec<String> = weights.keys().cloned().collect();
    let mut remapped: WeightMap = WeightMap::with_capacity(old_keys.len());
    for key in old_keys {
        let Some(value) = weights.remove(&key) else {
            continue;
        };
        let mut k = key.strip_prefix("model.").unwrap_or(&key).to_string();
        // The encoder sinusoidal positions are recomputed, never loaded.
        if k == "encoder.embed_positions.weight" {
            continue;
        }
        for (from, to) in key_map {
            if k.contains(from) {
                k = k.replace(from, to);
            }
        }
        // HuggingFace Conv1d weights are [out, in, kernel]; MLX wants
        // [out, kernel, in].
        let value = if (k == "encoder.conv1.weight" || k == "encoder.conv2.weight")
            && mlxcel_core::array_ndim(&value) == 3
        {
            mlxcel_core::transpose_axes(&value, &[0, 2, 1])
        } else {
            value
        };
        remapped.insert(k, value);
    }
    *weights = remapped;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn dims_parse_native_fields() {
        let cfg = json!({
            "n_mels": 80, "n_audio_ctx": 1500, "n_audio_state": 384,
            "n_audio_head": 6, "n_audio_layer": 4, "n_vocab": 51865,
            "n_text_ctx": 448, "n_text_state": 384, "n_text_head": 6, "n_text_layer": 4
        });
        let dims = WhisperDims::from_config(&cfg).unwrap();
        assert_eq!(dims.n_mels, 80);
        assert_eq!(dims.n_audio_layer, 4);
        assert_eq!(dims.n_vocab, 51865);
        assert_eq!(dims.n_text_head, 6);
    }

    #[test]
    fn dims_parse_huggingface_fields() {
        let cfg = json!({
            "num_mel_bins": 128, "max_source_positions": 1500, "d_model": 1280,
            "encoder_attention_heads": 20, "encoder_layers": 32, "vocab_size": 51866,
            "max_target_positions": 448, "decoder_attention_heads": 20, "decoder_layers": 32
        });
        let dims = WhisperDims::from_config(&cfg).unwrap();
        assert_eq!(dims.n_mels, 128);
        assert_eq!(dims.n_audio_state, 1280);
        assert_eq!(dims.n_text_state, 1280);
        assert_eq!(dims.n_audio_layer, 32);
        assert_eq!(dims.n_vocab, 51866);
    }

    #[test]
    fn dims_missing_required_field_errors() {
        let cfg = json!({ "n_mels": 80 });
        assert!(WhisperDims::from_config(&cfg).is_err());
    }

    #[test]
    fn padded_window_zero_pads_short_tail() {
        let n_mels = 2;
        let frames = 3;
        // 3 frames of [1,1],[2,2],[3,3].
        let mel = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0];
        let window = padded_window(&mel, frames, n_mels, 0);
        assert_eq!(window.len(), whisper_mel::WHISPER_N_FRAMES * n_mels);
        assert_eq!(&window[0..6], &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        // Everything past the 3 available frames is zero-padded.
        assert!(window[6..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn pad_audio_rounds_up_to_window_multiple() {
        assert!(pad_audio_to_window_multiple(&[]).is_empty());

        let one = pad_audio_to_window_multiple(&vec![0.5f32; 100]);
        assert_eq!(one.len(), whisper_mel::WHISPER_N_SAMPLES);
        assert_eq!(one[0], 0.5);
        assert_eq!(one[100], 0.0);

        let two = pad_audio_to_window_multiple(&vec![0.1f32; whisper_mel::WHISPER_N_SAMPLES + 5]);
        assert_eq!(two.len(), 2 * whisper_mel::WHISPER_N_SAMPLES);
    }

    #[test]
    fn padded_audio_tail_sits_at_log_floor_not_zero() {
        // A short tone padded to a full window must land its silent tail at the
        // normalized log floor (the per-utterance minimum), not 0.0. Feeding 0.0
        // would read as spurious broadband energy to the encoder.
        let mut tone = vec![0.0f32; 8000]; // 0.5 s at 16 kHz
        for (i, s) in tone.iter_mut().enumerate() {
            *s = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin();
        }
        let padded = pad_audio_to_window_multiple(&tone);
        assert_eq!(padded.len(), whisper_mel::WHISPER_N_SAMPLES);

        let (mel, frames) = whisper_mel::log_mel_spectrogram(&padded, 80);
        assert_eq!(frames, whisper_mel::WHISPER_N_FRAMES);

        let global_min = mel.iter().copied().fold(f32::INFINITY, f32::min);
        let last = &mel[(frames - 1) * 80..frames * 80];
        for &v in last {
            assert!(
                (v - global_min).abs() < 1e-3,
                "silent tail frame should sit at the log floor, got {v} vs floor {global_min}"
            );
        }
        assert!(
            global_min < -1e-2,
            "log floor must differ from the 0.0 the buggy mel padding produced, got {global_min}"
        );
    }
}
