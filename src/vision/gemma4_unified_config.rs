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

//! Gemma 4 Unified (`gemma4_unified`) configuration schema.
//!
//! Top-level [`Gemma4UnifiedConfig`] plus the vision / audio sub-configs. The
//! text sub-config reuses [`crate::models::gemma4::TextConfig`] (the same
//! transformer the `gemma4` family already runs), so only the vision and audio
//! front-end parameters and the multimodal token ids are defined here.
//!
//! Token-id note: the checkpoint emits `eoa_token_index` (not `eoa_token_id`);
//! [`Gemma4UnifiedConfig::eoa_token_id`] falls back to it.

use serde::Deserialize;
use serde_json::Value;

use crate::models::multimodal_placeholders::MultimodalPlaceholderTokens;

fn default_rms_norm_eps() -> f32 {
    1e-6
}

/// Vision sub-config (`gemma4_unified_vision`). Encoder-free patch projector.
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4UnifiedVisionConfig {
    /// Vision-tower patch size (informational; the projector uses
    /// `model_patch_size`). Default 16.
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    /// Spatial pooling kernel applied to the patch grid. Default 3.
    #[serde(default = "default_pooling_kernel_size")]
    pub pooling_kernel_size: usize,
    /// Side length (pixels) of each non-overlapping projector patch. The flat
    /// patch vector has `model_patch_size² · 3` elements. Default 48.
    #[serde(default = "default_model_patch_size")]
    pub model_patch_size: usize,
    /// Patch-embedding dimension (== `output_proj_dims` == text hidden size).
    #[serde(default = "default_mm_embed_dim")]
    pub mm_embed_dim: usize,
    /// Number of learned positional slots per axis (`pos_embedding` axis 0).
    #[serde(default = "default_mm_posemb_size")]
    pub mm_posemb_size: usize,
    /// Soft tokens emitted per image (== max patches). Default 280.
    #[serde(default = "default_num_soft_tokens")]
    pub num_soft_tokens: usize,
    /// Output projection dim consumed by `embed_vision`. Equals `mm_embed_dim`.
    #[serde(default = "default_mm_embed_dim")]
    pub output_proj_dims: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
}

fn default_patch_size() -> usize {
    16
}
fn default_pooling_kernel_size() -> usize {
    3
}
fn default_model_patch_size() -> usize {
    48
}
fn default_mm_embed_dim() -> usize {
    3840
}
fn default_mm_posemb_size() -> usize {
    1120
}
fn default_num_soft_tokens() -> usize {
    280
}

/// Audio sub-config (`gemma4_unified_audio`). Projection-only (no Conformer).
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4UnifiedAudioConfig {
    /// Raw waveform samples consumed per audio token (frame size). Default 640.
    #[serde(default = "default_audio_samples_per_token")]
    pub audio_samples_per_token: usize,
    /// Audio feature embedding dim (== frame size == `output_proj_dims`).
    #[serde(default = "default_audio_embed_dim")]
    pub audio_embed_dim: usize,
    /// Hidden size of the audio feature path (== `audio_embed_dim`).
    #[serde(default = "default_audio_embed_dim")]
    pub hidden_size: usize,
    /// Output projection dim consumed by `embed_audio`. Equals `audio_embed_dim`.
    #[serde(default = "default_audio_embed_dim")]
    pub output_proj_dims: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
}

fn default_audio_samples_per_token() -> usize {
    640
}
fn default_audio_embed_dim() -> usize {
    640
}

/// Top-level `gemma4_unified` config.
///
/// `text_config` is kept as a raw [`Value`] and parsed into
/// [`crate::models::gemma4::ModelArgs`] / `TextConfig` by the loader (sharing
/// the existing Gemma 4 text parse path, including quantization inheritance).
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4UnifiedConfig {
    pub model_type: String,
    pub text_config: Value,
    pub vision_config: Gemma4UnifiedVisionConfig,
    #[serde(default)]
    pub audio_config: Option<Gemma4UnifiedAudioConfig>,

    #[serde(default = "default_image_token_id")]
    pub image_token_id: i32,
    #[serde(default = "default_audio_token_id")]
    pub audio_token_id: i32,
    #[serde(default = "default_video_token_id")]
    pub video_token_id: i32,
    #[serde(default = "default_boi_token_id")]
    pub boi_token_id: i32,
    #[serde(default = "default_eoi_token_id")]
    pub eoi_token_id: i32,
    #[serde(default = "default_boa_token_id")]
    pub boa_token_id: i32,
    /// End-of-audio id. Checkpoints emit `eoa_token_index`; if `eoa_token_id`
    /// is absent we fall back to it (see [`Self::resolve_eoa_token_id`]).
    #[serde(default)]
    pub eoa_token_id: Option<i32>,
    #[serde(default)]
    pub eoa_token_index: Option<i32>,
    #[serde(default)]
    pub eos_token_id: Option<Value>,
    #[serde(default)]
    pub tie_word_embeddings: Option<bool>,
}

fn default_image_token_id() -> i32 {
    258_880
}
fn default_audio_token_id() -> i32 {
    258_881
}
fn default_video_token_id() -> i32 {
    258_884
}
fn default_boi_token_id() -> i32 {
    255_999
}
fn default_eoi_token_id() -> i32 {
    258_882
}
fn default_boa_token_id() -> i32 {
    256_000
}
fn default_eoa_token_index() -> i32 {
    258_883
}

impl Gemma4UnifiedConfig {
    /// Resolve the end-of-audio token id, preferring `eoa_token_id`, then
    /// `eoa_token_index`, then the documented default (258883).
    pub fn resolve_eoa_token_id(&self) -> i32 {
        self.eoa_token_id
            .or(self.eoa_token_index)
            .unwrap_or_else(default_eoa_token_index)
    }

    /// The reserved multimodal placeholder token ids (audio / image / video
    /// span markers) that must never appear in generated text output
    /// (issue #350).
    ///
    /// These are input-alignment placeholders; the runtime scatters encoded
    /// features into them during prefill, but they are illegal as generation
    /// output. The returned [`MultimodalPlaceholderTokens`] is fed through
    /// [`MultimodalPlaceholderTokens::suppressed_ids`] and masked to `-inf` at
    /// every decode step. Real EOS ids (`eos_token_id`) are intentionally
    /// excluded so end-of-sequence detection is unaffected.
    pub fn placeholder_tokens(&self) -> MultimodalPlaceholderTokens {
        MultimodalPlaceholderTokens {
            audio_token_id: Some(self.audio_token_id),
            image_token_id: Some(self.image_token_id),
            video_token_id: Some(self.video_token_id),
            boa_token_id: Some(self.boa_token_id),
            boi_token_id: Some(self.boi_token_id),
            eoa_token_id: Some(self.resolve_eoa_token_id()),
            eoi_token_id: Some(self.eoi_token_id),
        }
    }
}

#[cfg(test)]
#[path = "gemma4_unified_config_tests.rs"]
mod tests;
