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

//! Nemotron H Nano Omni audio modality.
//!
//! Faithful Rust port of upstream
//! `references/mlx-vlm/mlx_vlm/models/nemotron_h_nano_omni/audio.py`
//! covering:
//! - [`feature_extractor::NemotronOmniFeatureExtractor`] — log-mel
//!   spectrogram preprocessing (pure DSP, no network weights). Mirrors
//!   upstream `SoundFeatureExtractor`.
//! - [`encoder::NemotronOmniSoundEncoder`] — Parakeet/Conformer-style
//!   encoder (subsampling + 24x ParakeetEncoderBlock with FFN-MHSA-Conv-FFN).
//! - [`projector::NemotronOmniSoundProjection`] — RMSNorm + Linear +
//!   ReLU² + Linear projection from audio hidden size to LLM hidden size.
//!
//! ## Relationship to Gemma 4 audio (`crate::audio::*`)
//!
//! Gemma 4's audio path uses a USM-style preprocessor and a custom
//! Conformer (chunked attention, gradient clipping, residual_weight, etc.).
//! Parakeet uses different DSP parameters (n_fft=512, periodic Hann,
//! Slaney mel norm, regular preemphasis, per-clip normalization) and a
//! different encoder topology (relative positional encoding, GLU +
//! depthwise conv module, BatchNorm, no chunked attention). They are
//! sibling implementations — kept separate by design — but share a few
//! DSP helpers via `super::dsp` (mel filterbank, FFT magnitude). Audio
//! merge into the text token stream is delegated to
//! `crate::vision::merge::merge_llava` so the audio path mirrors the
//! existing image-token-merge contract.
//!
//! Used by: Nemotron H Nano Omni VLM (audio modality)

pub mod config;
pub mod encoder;
pub mod feature_extractor;
pub mod projector;

#[cfg(test)]
mod encoder_tests;

pub use config::NemotronOmniAudioConfig;
pub use encoder::NemotronOmniSoundEncoder;
pub use feature_extractor::{
    NemotronOmniFeatureExtractor, NemotronOmniFeatureExtractorOutput,
    nemotron_subsampling_output_length,
};
pub use projector::NemotronOmniSoundProjection;
