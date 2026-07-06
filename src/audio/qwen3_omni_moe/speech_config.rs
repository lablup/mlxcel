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
// Portions of this file are derived from mlx-vlm
// (https://github.com/Blaizzy/mlx-vlm), Copyright 2025 Prince Canuma,
// licensed under the MIT License. See the top-level NOTICE file for the
// attribution carried forward under the MIT License.

//! Qwen3-Omni speech-output configuration (stage 2: talker + code2wav).
//!
//! `talker_config` and `code2wav_config` are sub-objects of `config.json` at
//! the ROOT (not under `thinker_config`), as are the TTS control token ids
//! (`tts_bos_token_id` / `tts_eos_token_id` / `tts_pad_token_id`) and the
//! chat-role token ids the talker segmenter needs. Defaults mirror the
//! reference dataclasses in mlx-vlm
//! <https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen3_omni_moe/config.py>.
//!
//! Used by: Qwen3-Omni MoE speech pipeline (talker.rs, code2wav.rs, speech.rs).

use serde::Deserialize;
use std::collections::HashMap;

fn default_true() -> bool {
    true
}

/// Talker MoE decoder text config (`talker_config.text_config`).
#[derive(Debug, Clone, Deserialize)]
pub struct TalkerTextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub vocab_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    #[serde(default = "default_shared_expert_intermediate_size")]
    pub shared_expert_intermediate_size: usize,
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
}

fn default_shared_expert_intermediate_size() -> usize {
    768
}

/// Residual-codebook code predictor config
/// (`talker_config.code_predictor_config`). All fields default per the
/// reference dataclass.
#[derive(Debug, Clone, Deserialize)]
pub struct CodePredictorConfig {
    #[serde(default = "default_cp_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_cp_hidden")]
    pub hidden_size: usize,
    #[serde(default = "default_cp_intermediate")]
    pub intermediate_size: usize,
    #[serde(default = "default_cp_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_cp_kv_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_cp_rms_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_cp_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_cp_vocab")]
    pub vocab_size: usize,
    #[serde(default = "default_num_code_groups")]
    pub num_code_groups: usize,
}

fn default_cp_layers() -> usize {
    5
}
fn default_cp_hidden() -> usize {
    1024
}
fn default_cp_intermediate() -> usize {
    3072
}
fn default_cp_heads() -> usize {
    16
}
fn default_cp_kv_heads() -> usize {
    8
}
fn default_head_dim() -> usize {
    128
}
fn default_cp_rms_eps() -> f32 {
    1e-6
}
fn default_cp_rope_theta() -> f32 {
    1_000_000.0
}
fn default_cp_vocab() -> usize {
    2048
}
fn default_num_code_groups() -> usize {
    16
}

fn default_speaker_id() -> HashMap<String, i32> {
    HashMap::from([
        ("chelsie".to_string(), 2301),
        ("ethan".to_string(), 2302),
        ("aiden".to_string(), 2303),
    ])
}

/// Talker top-level config (`talker_config` at the config.json root).
#[derive(Debug, Clone, Deserialize)]
pub struct TalkerConfig {
    pub text_config: TalkerTextConfig,
    pub code_predictor_config: CodePredictorConfig,
    #[serde(default = "default_accept_hidden_layer")]
    pub accept_hidden_layer: usize,
    #[serde(default = "default_num_code_groups")]
    pub num_code_groups: usize,
    #[serde(default = "default_thinker_hidden_size")]
    pub thinker_hidden_size: usize,
    #[serde(default = "default_codec_bos")]
    pub codec_bos_id: i32,
    #[serde(default = "default_codec_eos")]
    pub codec_eos_token_id: i32,
    #[serde(default = "default_codec_nothink")]
    pub codec_nothink_id: i32,
    #[serde(default = "default_codec_pad")]
    pub codec_pad_id: i32,
    #[serde(default = "default_codec_think_bos")]
    pub codec_think_bos_id: i32,
    #[serde(default = "default_codec_think_eos")]
    pub codec_think_eos_id: i32,
    #[serde(default = "default_speaker_id")]
    pub speaker_id: HashMap<String, i32>,
}

fn default_accept_hidden_layer() -> usize {
    24
}
fn default_thinker_hidden_size() -> usize {
    2048
}
fn default_codec_bos() -> i32 {
    2149
}
fn default_codec_eos() -> i32 {
    2150
}
fn default_codec_nothink() -> i32 {
    2155
}
fn default_codec_pad() -> i32 {
    2148
}
fn default_codec_think_bos() -> i32 {
    2156
}
fn default_codec_think_eos() -> i32 {
    2157
}

/// code2wav codec vocoder config (`code2wav_config` at the config.json root).
#[derive(Debug, Clone, Deserialize)]
pub struct Code2WavConfig {
    #[serde(default = "default_c2w_hidden")]
    pub hidden_size: usize,
    #[serde(default = "default_cp_intermediate")]
    pub intermediate_size: usize,
    #[serde(default = "default_c2w_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_cp_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_cp_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_decoder_dim")]
    pub decoder_dim: usize,
    #[serde(default = "default_c2w_rms_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_c2w_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_codebook_size")]
    pub codebook_size: usize,
    #[serde(default = "default_num_code_groups")]
    pub num_quantizers: usize,
    #[serde(default = "default_upsample_rates")]
    pub upsample_rates: Vec<usize>,
    #[serde(default = "default_upsampling_ratios")]
    pub upsampling_ratios: Vec<usize>,
}

fn default_c2w_hidden() -> usize {
    1024
}
fn default_c2w_layers() -> usize {
    8
}
fn default_decoder_dim() -> usize {
    1536
}
fn default_c2w_rms_eps() -> f32 {
    1e-5
}
fn default_c2w_rope_theta() -> f32 {
    10_000.0
}
fn default_codebook_size() -> usize {
    2048
}
fn default_upsample_rates() -> Vec<usize> {
    vec![8, 5, 4, 3]
}
fn default_upsampling_ratios() -> Vec<usize> {
    vec![2, 2]
}

impl Code2WavConfig {
    /// Waveform samples produced per codec frame:
    /// `prod(upsampling_ratios) * prod(upsample_rates)` (1920 for the
    /// released checkpoints, 80 ms at 24 kHz).
    pub fn samples_per_frame(&self) -> usize {
        self.upsampling_ratios.iter().product::<usize>()
            * self.upsample_rates.iter().product::<usize>()
    }

    fn checked_samples_per_frame(&self) -> Result<usize, String> {
        self.upsampling_ratios
            .iter()
            .chain(self.upsample_rates.iter())
            .try_fold(1usize, |acc, &v| {
                acc.checked_mul(v)
                    .ok_or_else(|| "code2wav upsampling product overflows usize".to_string())
            })
    }
}

/// code2wav output sample rate. Fixed by the codec (12.5 Hz frame rate x
/// 1920x upsampling); not carried in `config.json`.
pub const SPEECH_SAMPLE_RATE: u32 = 24_000;

/// Token ids the speech pipeline needs from the config.json root.
#[derive(Debug, Clone)]
pub struct SpeechTokenIds {
    pub tts_bos_token_id: i32,
    pub tts_eos_token_id: i32,
    pub tts_pad_token_id: i32,
    pub im_start_token_id: i32,
    pub system_token_id: i32,
    pub user_token_id: i32,
    pub assistant_token_id: i32,
    /// Multimodal placeholder ids (image/video/audio); their presence in a
    /// sequence means the text-only speech path cannot condition it.
    pub multimodal_token_ids: Vec<i32>,
}

/// Talker sampling defaults, from `generation_config.json` `talker_*` fields
/// when present. The mlx-vlm reference applies only temperature and top-p to
/// the talker head (its `talker_top_k` / `talker_repetition_penalty` fields
/// are accepted but unused), so only these are carried.
#[derive(Debug, Clone)]
pub struct TalkerSampling {
    pub max_frames: usize,
    pub temperature: f32,
    pub top_p: f32,
}

impl Default for TalkerSampling {
    fn default() -> Self {
        Self {
            max_frames: 4096,
            temperature: 0.9,
            top_p: 1.0,
        }
    }
}

/// Everything the speech bundle needs from config.json (+ optional
/// generation_config.json).
#[derive(Debug, Clone)]
pub struct Qwen3OmniSpeechConfig {
    pub talker: TalkerConfig,
    pub code2wav: Code2WavConfig,
    pub ids: SpeechTokenIds,
    pub sampling: TalkerSampling,
    pub enable_audio_output: bool,
    pub group_size: i32,
    pub bits: i32,
}

/// Strip explicit JSON nulls so serde defaults apply (HF exports carry
/// `"sliding_window": null` style entries in the nested sub-configs).
fn strip_nulls(value: &mut serde_json::Value) {
    if let Some(map) = value.as_object_mut() {
        map.retain(|_, v| !v.is_null());
        for v in map.values_mut() {
            strip_nulls(v);
        }
    }
}

impl Qwen3OmniSpeechConfig {
    /// Parse from the config.json ROOT object and an optional parsed
    /// generation_config.json.
    pub fn parse(
        root: &serde_json::Value,
        generation_config: Option<&serde_json::Value>,
    ) -> Result<Self, String> {
        let sub = |key: &str| -> Result<serde_json::Value, String> {
            let mut v = root
                .get(key)
                .cloned()
                .ok_or_else(|| format!("Missing {key} in Qwen3-Omni config.json"))?;
            strip_nulls(&mut v);
            Ok(v)
        };

        let talker: TalkerConfig = serde_json::from_value(sub("talker_config")?)
            .map_err(|e| format!("Failed to parse talker_config: {e}"))?;
        let code2wav: Code2WavConfig = serde_json::from_value(sub("code2wav_config")?)
            .map_err(|e| format!("Failed to parse code2wav_config: {e}"))?;

        let id = |key: &str, default: i64| -> i32 {
            root.get(key).and_then(|v| v.as_i64()).unwrap_or(default) as i32
        };
        let thinker_id = |key: &str, default: i64| -> i32 {
            root.get("thinker_config")
                .and_then(|t| t.get(key))
                .and_then(|v| v.as_i64())
                .unwrap_or(default) as i32
        };
        let ids = SpeechTokenIds {
            tts_bos_token_id: id("tts_bos_token_id", 151_672),
            tts_eos_token_id: id("tts_eos_token_id", 151_673),
            tts_pad_token_id: id("tts_pad_token_id", 151_671),
            im_start_token_id: id("im_start_token_id", 151_644),
            system_token_id: id("system_token_id", 8_948),
            user_token_id: id("user_token_id", 872),
            assistant_token_id: id("assistant_token_id", 77_091),
            multimodal_token_ids: vec![
                thinker_id("image_token_id", 151_655),
                thinker_id("video_token_id", 151_656),
                thinker_id("audio_token_id", 151_675),
            ],
        };

        let mut sampling = TalkerSampling::default();
        if let Some(g) = generation_config {
            if let Some(v) = g.get("talker_max_new_tokens").and_then(|v| v.as_u64()) {
                sampling.max_frames = v as usize;
            }
            if let Some(v) = g.get("talker_temperature").and_then(|v| v.as_f64()) {
                sampling.temperature = v as f32;
            }
            if let Some(v) = g.get("talker_top_p").and_then(|v| v.as_f64()) {
                sampling.top_p = v as f32;
            }
        }

        let quant = |key: &str, default: i64| -> i32 {
            root.get("quantization")
                .and_then(|q| q.get(key))
                .and_then(|v| v.as_i64())
                .unwrap_or(default) as i32
        };

        let cfg = Self {
            talker,
            code2wav,
            ids,
            sampling,
            enable_audio_output: root
                .get("enable_audio_output")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            group_size: quant("group_size", 64),
            bits: quant("bits", 4),
        };
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<(), String> {
        let talker_groups = self.talker.num_code_groups;
        if talker_groups < 2 {
            return Err(format!(
                "talker_config.num_code_groups must be at least 2, got {talker_groups}"
            ));
        }
        let predictor_groups = self.talker.code_predictor_config.num_code_groups;
        if predictor_groups != talker_groups {
            return Err(format!(
                "talker_config.code_predictor_config.num_code_groups ({predictor_groups}) must \
                 match talker_config.num_code_groups ({talker_groups})"
            ));
        }
        if self.code2wav.num_quantizers != talker_groups {
            return Err(format!(
                "code2wav_config.num_quantizers ({}) must match talker_config.num_code_groups \
                 ({talker_groups})",
                self.code2wav.num_quantizers
            ));
        }
        if self.code2wav.codebook_size == 0 {
            return Err("code2wav_config.codebook_size must be nonzero".to_string());
        }
        let table_size = self
            .code2wav
            .num_quantizers
            .checked_mul(self.code2wav.codebook_size)
            .ok_or_else(|| "code2wav embedding table size overflows usize".to_string())?;
        if table_size > i32::MAX as usize {
            return Err(format!(
                "code2wav embedding table size {table_size} exceeds i32 index range"
            ));
        }
        if self.code2wav.checked_samples_per_frame()? == 0 {
            return Err("code2wav upsampling product must be nonzero".to_string());
        }
        if self.sampling.max_frames == 0 {
            return Err("talker_max_new_tokens must be nonzero".to_string());
        }
        if !self.sampling.temperature.is_finite() || self.sampling.temperature < 0.0 {
            return Err(format!(
                "talker_temperature must be finite and non-negative, got {}",
                self.sampling.temperature
            ));
        }
        if !self.sampling.top_p.is_finite() || self.sampling.top_p <= 0.0 {
            return Err(format!(
                "talker_top_p must be finite and positive, got {}",
                self.sampling.top_p
            ));
        }
        Ok(())
    }
}
