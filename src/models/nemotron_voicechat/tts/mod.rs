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

//! Nemotron VoiceChat EAR-TTS speech decoder.
//!
//! Ports `mlx_vlm/models/nemotron_voicechat/tts.py` (everything except the
//! codec, which lives in its own module, and the torch-layout conversions:
//! the only rename the TTS keys need is `_control_codes`), plus the TTS
//! pieces of `session.py` (`_tts_prompt`, `_replace_control_codes`).
//!
//! Per 80 ms frame the decoder takes the previous frame's 31 RVQ codec codes
//! and the LLM's current text token, runs a 28-layer Gemma 3 backbone under
//! classifier-free guidance, and samples the next frame's codes with a
//! Mixture-of-Gaussians head followed by MaskGIT-style residual
//! quantization. A session is primed once with [`RvqEarTtsModel::warmup`]
//! on the cached speaker prompt ([`TtsPrompt`] + the `Aria` latent from
//! [`SpeechDecoderAssets`]) and then advanced with [`RvqEarTtsModel::step`].
//!
//! Sampling uses MLX's global RNG in the reference's call order, so a run
//! seeded with [`mlxcel_core::random_seed`] right before warmup reproduces
//! the Python reference's codes.

mod assets;
mod char_encoder;
mod config;
mod fusion;
mod model;
mod mog_head;
mod norm_mlp;
mod rvq;
mod subword;

pub use assets::{SpeechDecoderAssets, TtsPrompt};
pub use config::{CharEncoderConfig, MogConfig, TtsConfig};
pub use model::{RvqEarTtsModel, TtsCaches, TtsStepOutput};
pub use mog_head::top_p_logits;
pub use norm_mlp::OffsetRmsNorm;
pub(crate) use subword::to_host_i32 as host_i32;

#[cfg(test)]
#[path = "tts_tests.rs"]
mod tts_tests;

#[cfg(test)]
#[path = "tts_dtype_tests.rs"]
mod tts_dtype_tests;
