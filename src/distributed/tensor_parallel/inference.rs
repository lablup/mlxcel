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

//! Inference-entrypoint helpers for tensor-parallel configuration.
//!
//! These helpers are used by the CLI generate path and the HTTP server startup
//! path so TP flags are parsed, validated, and resolved consistently before
//! the model-loading path begins.

use std::path::Path;

use anyhow::{Context, Result};
use serde_json::Value;

use crate::models::{ModelType, get_model_type, sanitize_config_json};

use super::{EmbeddingMode, ModelShardPlan, MoeShardMode, ShardConfig, generate_shard_plan};

#[derive(Debug, Clone)]
pub struct TensorParallelPlanSummary {
    pub model_type: ModelType,
    pub architecture: String,
    pub num_layers: usize,
    pub shard_config: ShardConfig,
    pub plan: ModelShardPlan,
}

impl TensorParallelPlanSummary {
    pub fn summary_line(&self) -> String {
        format!(
            "architecture={} model_type={:?} layers={} tp_size={} shard_rules={}",
            self.architecture,
            self.model_type,
            self.num_layers,
            self.shard_config.tp_size,
            self.plan.layer_plans.len()
        )
    }
}

pub fn shard_config_from_cli(
    tp_size: usize,
    tp_moe_mode: &str,
    tp_embedding_mode: &str,
    tp_lm_head_mode: &str,
) -> Result<ShardConfig> {
    let shard_config = ShardConfig {
        tp_size,
        moe_mode: tp_moe_mode.parse::<MoeShardMode>()?,
        embedding_mode: tp_embedding_mode.parse::<EmbeddingMode>()?,
        lm_head_mode: tp_lm_head_mode.parse::<EmbeddingMode>()?,
    };
    shard_config.validate()?;
    Ok(shard_config)
}

pub fn resolve_model_shard_plan(
    model_path: &Path,
    shard_config: ShardConfig,
) -> Result<TensorParallelPlanSummary> {
    let model_dir = if model_path.is_file() {
        model_path.parent().with_context(|| {
            format!(
                "missing model directory parent for {}",
                model_path.display()
            )
        })?
    } else {
        model_path
    };
    let config = read_model_config(model_dir)?;
    let model_type = get_model_type(model_dir)?;
    let architecture = detect_plan_architecture(&config, model_type);
    // Single-rank execution never consumes the layer count downstream
    // (`generate_shard_plan` returns a replicated plan immediately for
    // `tp_size == 1`, and `validate_supported_runtime` returns its default
    // support struct without inspecting the summary). VLM wrappers like
    // LLaVA ship a partial `text_config` that omits `num_hidden_layers` and
    // defers to a referenced base model, so strict detection would block
    // the entire single-rank generate path for those models. Tolerate a
    // missing layer count here and only fail when a real shard plan is
    // actually needed.
    let num_layers = match detect_num_layers(&config) {
        Ok(n) => n,
        Err(_) if shard_config.tp_size == 1 => 0,
        Err(err) => {
            return Err(err).with_context(|| {
                format!(
                    "failed to determine layer count for tensor-parallel planning in {}",
                    model_dir.display()
                )
            });
        }
    };
    let plan = generate_shard_plan(&architecture, num_layers, &shard_config)?;

    Ok(TensorParallelPlanSummary {
        model_type,
        architecture,
        num_layers,
        shard_config,
        plan,
    })
}

pub fn ensure_single_rank_runtime(
    summary: &TensorParallelPlanSummary,
    entrypoint: &str,
) -> Result<()> {
    if summary.shard_config.tp_size > 1 {
        anyhow::bail!(
            "{entrypoint} accepted tensor-parallel flags and generated a shard plan \
             ({summary}), but multi-rank inference is not wired into the active \
             execution path yet",
            summary = summary.summary_line()
        );
    }
    Ok(())
}

fn read_model_config(model_dir: &Path) -> Result<Value> {
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let config_str = sanitize_config_json(&config_str);
    serde_json::from_str::<Value>(&config_str)
        .with_context(|| format!("failed to parse {}", config_path.display()))
}

fn detect_plan_architecture(config: &Value, model_type: ModelType) -> String {
    config
        .get("text_config")
        .and_then(|text| text.get("model_type"))
        .and_then(Value::as_str)
        .or_else(|| config.get("model_type").and_then(Value::as_str))
        .map(str::to_string)
        .unwrap_or_else(|| fallback_architecture(model_type).to_string())
}

fn fallback_architecture(model_type: ModelType) -> &'static str {
    match model_type {
        ModelType::Llama | ModelType::Mistral3 | ModelType::Mistral3VLM => "llama",
        ModelType::Llama4 | ModelType::Llama4VLM => "llama4",
        ModelType::Qwen2 | ModelType::Qwen2VL | ModelType::Qwen25VL | ModelType::FastVLM => "qwen2",
        ModelType::Qwen3 | ModelType::Qwen3VL => "qwen3",
        ModelType::Qwen3Moe | ModelType::Qwen3VLMoe | ModelType::Qwen3OmniMoe => "qwen3_moe",
        ModelType::Qwen3Next => "qwen3_next",
        ModelType::Qwen35 | ModelType::Qwen35VLM => "qwen3_5",
        ModelType::Qwen35Moe | ModelType::Qwen35MoeVLM => "qwen3_5_moe",
        ModelType::Gemma | ModelType::PaliGemmaVLM => "gemma",
        ModelType::Gemma2 => "gemma2",
        ModelType::Gemma3 | ModelType::Gemma3VLM => "gemma3",
        ModelType::Gemma4 | ModelType::Gemma4VLM | ModelType::Gemma4Unified => "gemma4",
        ModelType::Gemma3n | ModelType::Gemma3nVLM => "gemma3n",
        ModelType::Phi => "phi",
        ModelType::Phi3 | ModelType::Phi3VLM => "phi3",
        ModelType::Phi4MMVLM | ModelType::Phi4SigLipVLM => "phi4mm",
        ModelType::Phi3Small => "phi3small",
        ModelType::PhiMoe => "phimoe",
        ModelType::GptOss => "gpt_oss",
        ModelType::MiniMax => "minimax",
        ModelType::Mixtral => "mixtral",
        ModelType::Qwen2Moe => "qwen2_moe",
        ModelType::OLMoE => "olmoe",
        ModelType::DeepSeek => "deepseek",
        ModelType::DeepSeekV2 => "deepseek_v2",
        ModelType::DeepSeekV3 => "deepseek_v3",
        ModelType::DeepSeekV32 => "deepseek_v32",
        ModelType::Dots1 => "dots1",
        ModelType::Cohere => "cohere",
        ModelType::Cohere2 => "cohere2",
        ModelType::Cohere2Moe => "cohere2_moe",
        ModelType::InternLM2 => "internlm2",
        ModelType::InternLM3 => "internlm3",
        ModelType::Baichuan => "baichuan",
        ModelType::Glm4 | ModelType::Glm4v | ModelType::GlmOcr => "glm4",
        ModelType::Glm4Moe | ModelType::Glm4vMoe => "glm4_moe",
        ModelType::Glm4MoeLite => "glm4_moe_lite",
        ModelType::GlmMoeDsa => "glm_moe_dsa",
        ModelType::Ernie45 | ModelType::PaddleOcrVL => "ernie4_5",
        ModelType::DotsOcrVL => "qwen2",
        ModelType::Ernie45Moe => "ernie4_5_moe",
        // ERNIE-4.5-VL text backbone; TP is refused for VLM-kind models earlier,
        // this keeps the dispatch table total.
        ModelType::Ernie45MoeVLM => "ernie4_5_moe",
        // Hunyuan-VL text backbone is the Hunyuan dense stack; TP is refused
        // for VLM-kind models earlier, this keeps the dispatch table total.
        ModelType::HunyuanVLM => "hunyuan_v1_dense",
        ModelType::HunyuanMoe => "hunyuan_moe",
        ModelType::HunyuanV1Dense => "hunyuan_v1_dense",
        ModelType::MiMo => "mimo",
        ModelType::Apertus => "apertus",
        ModelType::SeedOss => "seed_oss",
        ModelType::Granite => "granite",
        ModelType::BitNet => "bitnet",
        ModelType::ExaOne => "exaone",
        ModelType::ExaOne4 => "exaone4",
        ModelType::ExaOneMoe => "exaone_moe",
        ModelType::SolarOpen => "solar_open",
        ModelType::Olmo => "olmo",
        ModelType::Olmo2 => "olmo2",
        ModelType::Olmo3 => "olmo3",
        ModelType::StarCoder2 => "starcoder2",
        ModelType::Mellum => "mellum",
        ModelType::MiniCPM | ModelType::MiniCPMOVLM => "minicpm",
        ModelType::MiniCPM3 => "minicpm3",
        ModelType::StableLM => "stablelm",
        ModelType::SmolLM3 => "smollm3",
        ModelType::Ministral3 => "ministral3",
        ModelType::Mistral4 => "mistral4",
        ModelType::Nemotron => "nemotron",
        ModelType::Mamba => "mamba",
        ModelType::Mamba2 => "mamba2",
        ModelType::Jamba => "jamba",
        ModelType::FalconH1 => "falcon_h1",
        ModelType::Lfm2 => "lfm2",
        ModelType::Lfm2Moe => "lfm2_moe",
        // LFM2-VL's text backbone is LFM2; TP is refused for VLM-kind models
        // earlier, this keeps the dispatch table total.
        ModelType::Lfm2VL => "lfm2",
        ModelType::Plamo2 => "plamo2",
        ModelType::GraniteMoeHybrid => "granitemoehybrid",
        ModelType::NemotronH => "nemotron_h",
        ModelType::NemotronHNanoOmniVLM => "nemotron_h_nano_omni",
        ModelType::NemotronNAS => "nemotron_nas",
        ModelType::Rwkv7 => "rwkv7",
        ModelType::KimiLinear => "kimi_linear",
        // Kimi-VL / Kimi-VL 2.5 use a DeepSeek-V3-style MoE text backbone.
        ModelType::KimiVL | ModelType::KimiK25 => "deepseek_v3",
        ModelType::LongcatFlash | ModelType::LongcatFlashNgram => "longcat_flash",
        ModelType::Step3p5 => "step3p5",
        // Step-3.7's text backbone is the Step-3.5 MoE stack; TP is refused for
        // VLM-kind models earlier, this keeps the dispatch table total.
        ModelType::Step3p7 => "step3p5",
        ModelType::RecurrentGemma => "recurrent_gemma",
        ModelType::LlavaVLM
        | ModelType::LlavaBunnyVLM
        | ModelType::AyaVisionVLM
        | ModelType::PixtralVLM
        | ModelType::Moondream3VLM
        | ModelType::Moondream2VLM
        | ModelType::MolmoVLM
        | ModelType::Molmo2VLM
        | ModelType::MolmoPointVLM
        // InternVL's text backbone is Qwen2 (llama family). TP is not
        // supported for VLM-kind models (the loader refuses it earlier);
        // this keeps the planner dispatch table from panicking.
        | ModelType::InternVLChatVLM
        // Llama 3.2 Vision's text backbone is Llama-3; TP is not supported for
        // VLM-kind models (the loader refuses it earlier).
        | ModelType::MllamaVLM => "llama",
        // SmolVLM's text backbone is SmolLM2 (llama family). TP is refused for
        // VLM-kind models earlier; this keeps the dispatch table total.
        | ModelType::SmolVLM
        // Idefics2's text backbone is Mistral (llama family). TP is refused for
        // VLM-kind models earlier; this keeps the dispatch table total.
        | ModelType::Idefics2 => "llama",
        // Granite Vision's text backbone is Granite; TP is refused for VLM-kind
        // models earlier, this keeps the dispatch table total.
        ModelType::GraniteVisionVLM => "granite",
        // Granite 4 Vision's text backbone is granitemoehybrid; TP is refused for
        // VLM-kind models earlier, this keeps the dispatch table total.
        ModelType::Granite4VisionVLM => "granitemoehybrid",
        ModelType::DeepSeekOcrVLM => "deepseek",
        ModelType::DeepSeekOcr2VLM => "deepseek",
        // DeepSeek-VL2's text backbone is deepseek_v2; TP is refused for VLM-kind
        // models earlier, this keeps the dispatch table total.
        ModelType::DeepSeekVL2 => "deepseek_v2",
        // Youtu-VL is not currently supported by tensor-parallel inference;
        // we return a placeholder architecture string here so the planner
        // does not panic on the dispatch table lookup. The actual loader
        // refuses TP routing earlier than this for VLM-kind models.
        ModelType::YoutuVLM => "youtu_vl",
        // MiniCPM-V 4.6 uses Qwen3.5 backbone; TP is not supported for VLM-kind
        // models and the loader refuses TP routing earlier. Return placeholder.
        ModelType::MiniCPMV46VLM => "minicpmv4_6",
        // DiffusionGemma is not supported by tensor-parallel inference (the
        // planner's supported-architecture validation rejects this string
        // before any TP load is attempted).
        ModelType::DiffusionGemma => "diffusion_gemma",
        // LLaDA-2 MoE is a masked-diffusion model served through the diffusion
        // worker loop, never routed to tensor-parallel text inference; the
        // planner's supported-architecture validation rejects this string.
        ModelType::Llada2Moe => "llada2_moe",
        // Whisper is an ASR model served through the audio endpoints, never
        // routed to tensor-parallel text inference; the loader rejects it
        // earlier. Return a placeholder so the dispatch table stays total.
        ModelType::Whisper => "whisper",
        // Kokoro is a TTS model served through /v1/audio/speech, never routed
        // to tensor-parallel text inference; placeholder keeps the table total.
        ModelType::Kokoro => "kokoro",
    }
}

fn detect_num_layers(config: &Value) -> Result<usize> {
    for path in [
        &["text_config", "num_hidden_layers"][..],
        &["language_config", "num_hidden_layers"][..],
        &["num_hidden_layers"][..],
        &["n_layers"][..],
        &["num_layers"][..],
    ] {
        if let Some(value) = value_at_path(config, path).and_then(Value::as_u64) {
            return usize::try_from(value).context("num_layers exceeds usize");
        }
    }

    anyhow::bail!(
        "missing layer-count key; expected one of text_config.num_hidden_layers, \
         language_config.num_hidden_layers, num_hidden_layers, n_layers, or num_layers"
    )
}

fn value_at_path<'a>(value: &'a Value, path: &[&str]) -> Option<&'a Value> {
    let mut current = value;
    for key in path {
        current = current.get(*key)?;
    }
    Some(current)
}

#[cfg(test)]
#[path = "inference_tests.rs"]
mod tests;
