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

//! Model-type detection helpers.
//!
//! This module owns config-driven architecture classification and related
//! detection helpers so `models/mod.rs` can stay focused on the registry of
//! model implementations and exported types.

use anyhow::Result;
use serde_json::Value;
use std::path::Path;

use super::ModelType;
use super::sanitize::sanitize_config_json;

pub(crate) fn has_vision_config(config: &serde_json::Value) -> bool {
    config.get("vision_config").is_some()
}

fn gemma4_has_vision_weights(model_path: &Path) -> bool {
    let index_path = model_path.join("model.safetensors.index.json");
    if let Ok(index_str) = std::fs::read_to_string(&index_path)
        && let Ok(index) = serde_json::from_str::<Value>(&index_str)
        && let Some(weight_map) = index.get("weight_map").and_then(Value::as_object)
    {
        return weight_map
            .keys()
            .any(|key| key.starts_with("vision_tower.") || key.starts_with("embed_vision."));
    }

    model_path.join("processor_config.json").exists()
}

pub(crate) fn detect_text_or_vlm(
    config: &serde_json::Value,
    text_model: ModelType,
    vlm_model: ModelType,
) -> ModelType {
    if has_vision_config(config) {
        vlm_model
    } else {
        text_model
    }
}

pub(crate) fn detect_hunyuan_model_type(config: &serde_json::Value) -> ModelType {
    let num_experts = config["num_experts"].as_i64().unwrap_or(1);
    if num_experts > 1 {
        ModelType::HunyuanMoe
    } else {
        ModelType::HunyuanV1Dense
    }
}

/// Detect model type from config.json
pub fn get_model_type(model_path: &Path) -> Result<ModelType> {
    let config_path = model_path.join("config.json");
    let config_str = std::fs::read_to_string(config_path)?;
    let config_str = sanitize_config_json(&config_str);
    let v: serde_json::Value = serde_json::from_str(&config_str)?;

    // Kokoro TTS checkpoints carry no top-level `model_type`, so detect them by
    // architecture signal (the `istftnet` config block or the canonical weight
    // filename) before the `model_type`-based dispatch below would error.
    if super::kokoro::is_kokoro_checkpoint(model_path, &v) {
        return Ok(ModelType::Kokoro);
    }

    let model_type_raw = v["model_type"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("model_type not found"))?;
    // Normalize to lowercase so HuggingFace checkpoints that preserve the
    // upstream casing (e.g. `NemotronH_Nano_Omni_Reasoning_V3`) match the
    // same arm as their canonical lowercase form.
    let model_type = model_type_raw.to_ascii_lowercase();

    match model_type.as_str() {
        "llama" | "mistral" => Ok(ModelType::Llama),
        "llama4" => Ok(detect_text_or_vlm(
            &v,
            ModelType::Llama4,
            ModelType::Llama4VLM,
        )),
        // Llama 3.2 Vision. Always multimodal (Llama-3 text backbone with gated
        // cross-attention adapters attending to a tiled ViT tower).
        "mllama" => Ok(ModelType::MllamaVLM),
        "qwen2" => Ok(ModelType::Qwen2),
        "qwen3" => Ok(ModelType::Qwen3),
        "qwen3_moe" => Ok(ModelType::Qwen3Moe),
        "qwen3_next" | "qwen3next" => Ok(ModelType::Qwen3Next),
        "qwen3_5" => Ok(detect_text_or_vlm(
            &v,
            ModelType::Qwen35,
            ModelType::Qwen35VLM,
        )),
        "qwen3_5_moe" => Ok(detect_text_or_vlm(
            &v,
            ModelType::Qwen35Moe,
            ModelType::Qwen35MoeVLM,
        )),
        "qwen2_moe" => Ok(ModelType::Qwen2Moe),
        "gemma" => Ok(ModelType::Gemma),
        "gemma2" => Ok(ModelType::Gemma2),
        "gemma3" | "gemma3_text" => Ok(detect_text_or_vlm(
            &v,
            ModelType::Gemma3,
            ModelType::Gemma3VLM,
        )),
        "gemma4" | "gemma4_text" => Ok(if gemma4_has_vision_weights(model_path) {
            ModelType::Gemma4VLM
        } else {
            ModelType::Gemma4
        }),
        // DiffusionGemma (block-diffusion on the Gemma 4 MoE backbone). The
        // checkpoint always ships a vision tower, but phase 1 is text-only:
        // the loader skips the vision weights, so detection is by model_type
        // alone. `diffusion_gemma_text` is accepted for text-only exports.
        "diffusion_gemma" | "diffusion_gemma_text" => Ok(ModelType::DiffusionGemma),
        // Gemma 4 Unified is always multimodal (text + vision [+ audio]); it
        // carries `vision_embedder.*` patch-projector weights rather than the
        // `vision_tower.*` ViT used by `gemma4`/Gemma4VLM, so it is detected by
        // model_type alone and never misrouted to Gemma4VLM.
        "gemma4_unified" => Ok(ModelType::Gemma4Unified),
        "gemma3n" | "gemma3n_text" => Ok(detect_text_or_vlm(
            &v,
            ModelType::Gemma3n,
            ModelType::Gemma3nVLM,
        )),
        "phi" | "phi-msft" => Ok(ModelType::Phi),
        "phi3" => Ok(ModelType::Phi3),
        "phi4mm" => Ok(ModelType::Phi4MMVLM),
        "phi4-siglip" => Ok(ModelType::Phi4SigLipVLM),
        "phi3_v" => Ok(ModelType::Phi3VLM),
        "phi3small" => Ok(ModelType::Phi3Small),
        "phimoe" => Ok(ModelType::PhiMoe),
        "minimax" => Ok(ModelType::MiniMax),
        "gpt_oss" => Ok(ModelType::GptOss),
        "mixtral" => Ok(ModelType::Mixtral),
        "olmoe" => Ok(ModelType::OLMoE),
        "deepseek" => Ok(ModelType::DeepSeek),
        "deepseek_v2" => Ok(ModelType::DeepSeekV2),
        "deepseek_v3" => Ok(ModelType::DeepSeekV3),
        "deepseek_v32" | "deepseek_v3.2" => Ok(ModelType::DeepSeekV32),
        "dots1" => Ok(ModelType::Dots1),
        "cohere" => Ok(ModelType::Cohere),
        "cohere2" => Ok(ModelType::Cohere2),
        "internlm2" => Ok(ModelType::InternLM2),
        "internlm3" => Ok(ModelType::InternLM3),
        "baichuan_m1" => Ok(ModelType::Baichuan),
        "bitnet" => Ok(ModelType::BitNet),
        "glm4" => Ok(ModelType::Glm4),
        "glm4_moe" => Ok(ModelType::Glm4Moe),
        "solar_open" => Ok(ModelType::SolarOpen),
        "glm4_moe_lite" => Ok(ModelType::Glm4MoeLite),
        "glm_moe_dsa" => Ok(ModelType::GlmMoeDsa),
        "ernie4_5" | "ernie4.5" => Ok(ModelType::Ernie45),
        "ernie4_5_moe" | "ernie4.5_moe" => Ok(ModelType::Ernie45Moe),
        "hunyuan_v1_dense" | "hunyuan_dense" => Ok(ModelType::HunyuanV1Dense),
        "hunyuan" => Ok(detect_hunyuan_model_type(&v)),
        "mimo" => Ok(ModelType::MiMo),
        "apertus" => Ok(ModelType::Apertus),
        "seed_oss" => Ok(ModelType::SeedOss),
        "granite" => Ok(ModelType::Granite),
        "exaone" => Ok(ModelType::ExaOne),
        "exaone4" => Ok(ModelType::ExaOne4),
        "exaone_moe" => Ok(ModelType::ExaOneMoe),
        "olmo" => Ok(ModelType::Olmo),
        "olmo2" => Ok(ModelType::Olmo2),
        "olmo3" => Ok(ModelType::Olmo3),
        "starcoder2" => Ok(ModelType::StarCoder2),
        "mellum" => Ok(ModelType::Mellum),
        "minicpm" => Ok(ModelType::MiniCPM),
        "minicpm3" => Ok(ModelType::MiniCPM3),
        "stablelm" => Ok(ModelType::StableLM),
        "smollm3" => Ok(ModelType::SmolLM3),
        "ministral3" => Ok(ModelType::Ministral3),
        "mistral3" => Ok(detect_text_or_vlm(
            &v,
            ModelType::Mistral3,
            ModelType::Mistral3VLM,
        )),
        "mistral4" => Ok(ModelType::Mistral4),
        "nemotron" => Ok(ModelType::Nemotron),
        "mamba" | "falcon_mamba" => Ok(ModelType::Mamba),
        "mamba2" => Ok(ModelType::Mamba2),
        "jamba" => Ok(ModelType::Jamba),
        "falcon_h1" => Ok(ModelType::FalconH1),
        "lfm2" => Ok(ModelType::Lfm2),
        "lfm2_vl" | "lfm2-vl" => Ok(ModelType::Lfm2VL),
        "lfm2_moe" => Ok(ModelType::Lfm2Moe),
        "plamo2" => Ok(ModelType::Plamo2),
        "granitemoehybrid" => Ok(ModelType::GraniteMoeHybrid),
        "nemotron_h" => Ok(ModelType::NemotronH),
        "nemotron_h_nano_omni" | "nemotronh_nano_omni_reasoning_v3" => {
            Ok(ModelType::NemotronHNanoOmniVLM)
        }
        "nemotron-nas" => Ok(ModelType::NemotronNAS),
        "rwkv7" => Ok(ModelType::Rwkv7),
        "kimi_linear" => Ok(ModelType::KimiLinear),
        "kimi_vl" => Ok(ModelType::KimiVL),
        "kimi_k25" => Ok(ModelType::KimiK25),
        "longcat_flash" => Ok(ModelType::LongcatFlash),
        "longcat_flash_ngram" => Ok(ModelType::LongcatFlashNgram),
        "step3p5" => Ok(ModelType::Step3p5),
        "recurrent_gemma" | "griffin" => Ok(ModelType::RecurrentGemma),
        "qwen2_vl" => Ok(ModelType::Qwen2VL),
        "qwen2_5_vl" => Ok(ModelType::Qwen25VL),
        "qwen3_vl" => Ok(ModelType::Qwen3VL),
        "qwen3_vl_moe" => Ok(ModelType::Qwen3VLMoe),
        "paddleocr_vl" => Ok(ModelType::PaddleOcrVL),
        "dots_ocr" => Ok(ModelType::DotsOcrVL),
        "glm4v" => Ok(ModelType::Glm4v),
        "glm4v_moe" => Ok(ModelType::Glm4vMoe),
        "glm_ocr" => Ok(ModelType::GlmOcr),
        "youtu_vl" => Ok(ModelType::YoutuVLM),
        "internvl_chat" => Ok(ModelType::InternVLChatVLM),
        // SmolVLM2 ships as `smolvlm`/`smolvlm2`. SmolVLM-Instruct ships as an
        // Idefics3 checkpoint (`idefics3`, `Idefics3ForConditionalGeneration`):
        // a SigLIP vision tower + pixel-shuffle connector + Llama text backbone,
        // which is exactly what the SmolVLM runtime implements.
        "smolvlm" | "smolvlm2" | "idefics3" => Ok(ModelType::SmolVLM),
        // Idefics2 shares SmolVLM's SigLIP tower but uses a perceiver-resampler
        // connector and a Mistral text backbone, so it gets its own runtime.
        "idefics2" => Ok(ModelType::Idefics2),
        "minicpmo" => Ok(ModelType::MiniCPMOVLM),
        "minicpmv4_6" => Ok(ModelType::MiniCPMV46VLM),
        "moondream3" => Ok(ModelType::Moondream3VLM),
        "moondream2" | "moondream1" => Ok(ModelType::Moondream2VLM),
        "granite_vision" => Ok(ModelType::GraniteVisionVLM),
        "granite4_vision" => Ok(ModelType::Granite4VisionVLM),
        "deepseekocr" => Ok(ModelType::DeepSeekOcrVLM),
        "deepseekocr_2" => Ok(ModelType::DeepSeekOcr2VLM),
        "llava" | "llava_next" => {
            // The original IBM Granite Vision checkpoint ships as `llava_next`
            // with a `granite` text backbone; route it to the Granite VLM.
            let text_model_type = v
                .get("text_config")
                .and_then(|t| t.get("model_type"))
                .and_then(|m| m.as_str())
                .unwrap_or("");
            if text_model_type == "granite" {
                Ok(ModelType::GraniteVisionVLM)
            } else {
                Ok(ModelType::LlavaVLM)
            }
        }
        "llava_bunny" | "bunny-llama" | "llava-qwen2" => Ok(ModelType::LlavaBunnyVLM),
        "aya_vision" => Ok(ModelType::AyaVisionVLM),
        "paligemma" => Ok(ModelType::PaliGemmaVLM),
        "pixtral" => Ok(ModelType::PixtralVLM),
        "molmo" => Ok(ModelType::MolmoVLM),
        "molmo2" => Ok(ModelType::Molmo2VLM),
        "molmo_point" => Ok(ModelType::MolmoPointVLM),
        // Speech-to-text (encoder-decoder ASR).
        "whisper" => Ok(ModelType::Whisper),
        _ => Err(anyhow::anyhow!(
            "Unsupported model type: {}",
            model_type_raw
        )),
    }
}
