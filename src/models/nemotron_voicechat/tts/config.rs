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

//! EAR-TTS configuration (`tts_config` in the VoiceChat `config.json`).
//!
//! Ports `TTSConfig`, `CharacterEncoderConfig` and `MoGConfig` from
//! `mlx_vlm/models/nemotron_voicechat/config.py`. Every default equals the
//! value the NemotronLabs VoiceChat checkpoint ships (or, where the
//! checkpoint omits a key, the reference dataclass default), so a missing
//! key deserializes to the reference behaviour.
//!
//! The sampling knobs (`guidance_scale`, `top_p`, `noise_scale`, `exponent`)
//! are `f64` on purpose: the reference evaluates expressions such as
//! `1.0 - top_p` in Python double precision and MLX rounds the result to
//! `f32` once. Holding them as `f32` would round the operand first and shift
//! the result by an ulp, which is enough to move a top-p boundary.

use serde::Deserialize;

use crate::models::gemma3::{ModelArgs, Quantization};

/// T5Gemma character encoder hyper-parameters (`tts_config.character_encoder`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct CharEncoderConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub query_pre_attn_scalar: f64,
    pub attn_logit_softcapping: f64,
    pub rope_base: f32,
    pub char_vocab_size: usize,
}

impl Default for CharEncoderConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1152,
            intermediate_size: 4608,
            num_hidden_layers: 1,
            num_attention_heads: 16,
            num_key_value_heads: 16,
            head_dim: 72,
            rms_norm_eps: 1e-6,
            query_pre_attn_scalar: 256.0,
            attn_logit_softcapping: 50.0,
            rope_base: 10_000.0,
            char_vocab_size: 257,
        }
    }
}

/// Mixture-of-Gaussians head hyper-parameters (`tts_config.mog_head`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default)]
pub struct MogConfig {
    pub intermediate_size: usize,
    pub low_rank: usize,
    pub min_log_std: f64,
    pub num_layers: usize,
    pub num_predictions: usize,
    pub eps: f32,
}

impl Default for MogConfig {
    fn default() -> Self {
        Self {
            intermediate_size: 4608,
            low_rank: 64,
            min_log_std: -4.0,
            num_layers: 3,
            num_predictions: 1024,
            eps: 1e-6,
        }
    }
}

/// EAR-TTS decoder configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TtsConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub sliding_window: usize,
    pub sliding_window_pattern: usize,
    pub rms_norm_eps: f32,
    pub query_pre_attn_scalar: f32,
    pub rope_global_base_freq: f32,
    pub rope_local_base_freq: f32,
    pub latent_size: usize,
    pub num_quantizers: usize,
    pub codebook_size: usize,
    pub num_delay_speech_tokens: usize,
    pub num_iterations: usize,
    pub guidance_scale: f64,
    pub top_p: f64,
    pub noise_scale: f64,
    pub exponent: f64,
    pub disable_eos_prediction: bool,
    pub use_gated_fusion_for_text_audio: bool,
    pub use_subword_flag_emb: bool,
    pub use_bos_eos_emb: bool,
    pub use_audio_prompt_frozen_projection: bool,
    pub audio_prompt_duration: f64,
    pub character_encoder: CharEncoderConfig,
    pub mog_head: MogConfig,
    /// Quantization parameters for any quantized linear under the TTS tree.
    /// The published checkpoint keeps the TTS dense, and
    /// [`mlxcel_core::layers::UnifiedLinear`] only consults these when a
    /// `.scales` tensor is present, so `None` (64 / 4) is the safe default.
    /// The loader copies the checkpoint's top-level `quantization` here.
    pub quantization: Option<Quantization>,
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1152,
            intermediate_size: 4608,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 16,
            head_dim: 72,
            sliding_window: 7500,
            sliding_window_pattern: 6,
            rms_norm_eps: 1e-6,
            query_pre_attn_scalar: 256.0,
            rope_global_base_freq: 1_000_000.0,
            rope_local_base_freq: 10_000.0,
            latent_size: 512,
            num_quantizers: 31,
            codebook_size: 1024,
            num_delay_speech_tokens: 2,
            num_iterations: 8,
            guidance_scale: 0.2,
            top_p: 0.95,
            noise_scale: 0.001,
            exponent: 3.0,
            disable_eos_prediction: true,
            use_gated_fusion_for_text_audio: true,
            use_subword_flag_emb: true,
            use_bos_eos_emb: true,
            use_audio_prompt_frozen_projection: true,
            audio_prompt_duration: 3.0,
            character_encoder: CharEncoderConfig::default(),
            mog_head: MogConfig::default(),
            quantization: None,
        }
    }
}

impl TtsConfig {
    /// Gemma 3 arguments for the EAR-TTS backbone, mirroring the reference
    /// `_gemma_config(config)` (vocab 1, no rope scaling).
    pub fn gemma_args(&self) -> ModelArgs {
        ModelArgs {
            model_type: "gemma3_text".to_string(),
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            intermediate_size: self.intermediate_size,
            num_attention_heads: self.num_attention_heads,
            head_dim: self.head_dim,
            rms_norm_eps: self.rms_norm_eps,
            vocab_size: 1,
            num_key_value_heads: self.num_key_value_heads,
            rope_theta: self.rope_global_base_freq,
            rope_local_base_freq: self.rope_local_base_freq,
            query_pre_attn_scalar: self.query_pre_attn_scalar,
            sliding_window: self.sliding_window,
            sliding_window_pattern: self.sliding_window_pattern,
            max_position_embeddings: 4096,
            rope_scaling: None,
            quantization: self.quantization.clone(),
        }
    }

    /// Number of RVQ codebooks unmasked by each of the `num_iterations`
    /// MaskGIT-style refinement passes, in order.
    ///
    /// Mirrors `generate_codes`: `masked[i] = ceil((1 - (i/n)^e)^(1/e) * Q)`
    /// evaluated in double precision, and `counts[i] = masked[i] -
    /// masked[i + 1]` (the last pass unmasks everything still masked). The
    /// counts always sum to `num_quantizers`; passes with a zero count are
    /// skipped by the generator and draw no random numbers.
    pub fn mask_schedule(&self) -> Vec<usize> {
        let n = self.num_iterations;
        let q = self.num_quantizers as f64;
        let e = self.exponent;
        let masked: Vec<usize> = (0..n)
            .map(|i| {
                let rate = i as f64 / n as f64;
                (((1.0 - rate.powf(e)).powf(1.0 / e)) * q).ceil() as usize
            })
            .collect();
        (0..n)
            .map(|i| masked[i].saturating_sub(masked.get(i + 1).copied().unwrap_or(0)))
            .collect()
    }

    /// Group size used when a TTS linear turns out to be quantized.
    pub(crate) fn group_size(&self) -> i32 {
        self.quantization.as_ref().map_or(64, |q| q.group_size)
    }

    /// Bit width used when a TTS linear turns out to be quantized.
    pub(crate) fn bits(&self) -> i32 {
        self.quantization.as_ref().map_or(4, |q| q.bits)
    }
}
