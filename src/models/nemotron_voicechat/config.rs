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

//! Top-level `config.json` of a Nemotron VoiceChat checkpoint.
//!
//! Mirrors `ModelConfig` in upstream
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/nemotron_voicechat/config.py.
//! The four sub-networks keep their own typed configs next to their modules;
//! this struct holds the timeline-level ids and rates plus the raw
//! sub-config objects, which the loader deserializes into those types.
//!
//! Used by: Nemotron VoiceChat loader and sessions

use serde::Deserialize;
use serde_json::Value;

/// Timeline-level configuration of a VoiceChat checkpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct VoiceChatConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    pub text_config: Value,
    #[serde(default)]
    pub audio_config: Value,
    #[serde(default)]
    pub tts_config: Value,
    #[serde(default)]
    pub codec_config: Value,
    #[serde(default = "default_bos")]
    pub bos_token_id: i32,
    #[serde(default = "default_eos")]
    pub eos_token_id: i32,
    #[serde(default = "default_pad")]
    pub pad_token_id: i32,
    #[serde(default = "default_silence")]
    pub silence_token_id: i32,
    #[serde(default = "default_blank")]
    pub rnnt_blank_id: i32,
    #[serde(default = "default_input_rate")]
    pub input_sample_rate: u32,
    #[serde(default = "default_output_rate")]
    pub output_sample_rate: u32,
    #[serde(default = "default_frame_duration")]
    pub frame_duration: f32,
    #[serde(default = "default_function_weight")]
    pub function_channel_weight: f32,
    #[serde(default)]
    pub default_system_prompt: String,
    #[serde(default)]
    pub rnnt_vocabulary: Vec<String>,
    #[serde(default = "default_speaker")]
    pub speaker: String,
    #[serde(default)]
    pub quantization: Option<Value>,
}

fn default_model_type() -> String {
    "nemotron_voicechat".to_string()
}
fn default_bos() -> i32 {
    1
}
fn default_eos() -> i32 {
    2
}
fn default_pad() -> i32 {
    12
}
fn default_silence() -> i32 {
    11
}
fn default_blank() -> i32 {
    1024
}
fn default_input_rate() -> u32 {
    16_000
}
fn default_output_rate() -> u32 {
    22_050
}
fn default_frame_duration() -> f32 {
    0.08
}
fn default_function_weight() -> f32 {
    2.0
}
fn default_speaker() -> String {
    "Aria".to_string()
}

impl VoiceChatConfig {
    /// Parse and validate a `config.json` string.
    pub fn from_json(text: &str) -> Result<Self, String> {
        let config: Self =
            serde_json::from_str(text).map_err(|e| format!("nemotron_voicechat config: {e}"))?;
        config.validate()?;
        Ok(config)
    }

    /// Reject values this port does not implement.
    pub fn validate(&self) -> Result<(), String> {
        if self.speaker != "Aria" {
            return Err(format!(
                "nemotron_voicechat: speaker {:?} is not supported; only the built-in \"Aria\" \
                 prompt latents ship with the checkpoint",
                self.speaker
            ));
        }
        if self.input_sample_rate != 16_000 {
            return Err(format!(
                "nemotron_voicechat: input_sample_rate {} is not supported (expected 16000)",
                self.input_sample_rate
            ));
        }
        Ok(())
    }

    /// Input samples per timeline frame (1280 at 16 kHz / 80 ms).
    pub fn frame_samples(&self) -> usize {
        (self.input_sample_rate as f32 * self.frame_duration).round() as usize
    }

    /// Top-level default `(group_size, bits)` from the `quantization` map,
    /// when the checkpoint is quantized.
    pub fn default_quantization(&self) -> Option<(i32, i32)> {
        let q = self.quantization.as_ref()?.as_object()?;
        let group_size = q.get("group_size")?.as_i64()? as i32;
        let bits = q.get("bits")?.as_i64()? as i32;
        Some((group_size, bits))
    }

    /// Token ids that never reach the decoded text: pad, silence, BOS, EOS.
    pub fn special_text_ids(&self) -> [i32; 4] {
        [
            self.pad_token_id,
            self.silence_token_id,
            self.bos_token_id,
            self.eos_token_id,
        ]
    }
}
