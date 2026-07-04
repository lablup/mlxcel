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

//! Centralized model descriptors and loading-policy metadata.
//!
//! This module is the single control-plane registry for model kind, directory
//! load family, weight load family, and adapter support. Loader code should
//! consume these descriptors instead of maintaining parallel support matches in
//! multiple modules.

use anyhow::Result;

use crate::models::ModelType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ModelKind {
    Text,
    Vlm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DirectoryRouteFamily {
    /// `Mistral3` wrapper with nested `text_config` that picks the final route.
    Mistral3Dynamic,
    /// Vision-language families routed through `src/loading/vlm*.rs`.
    Vlm,
    /// Text families with directory loaders outside the standard registry.
    Nonstandard,
    /// Standard config-backed text families.
    ConfigBacked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StaticModelDescriptor {
    pub(crate) model_type: ModelType,
    pub(crate) kind: ModelKind,
    pub(crate) directory_family: DirectoryRouteFamily,
    pub(crate) adapter_weight_route: Option<WeightLoadRoute>,
    pub(crate) adapter_unsupported_message: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ModelCapabilities {
    pub(crate) kind: ModelKind,
    pub(crate) adapter_unsupported_message: Option<&'static str>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DirectoryLoadRoute {
    /// `Mistral3` wrapper with `ministral3` text tower in `text_config`.
    Mistral3TextWrapper,
    /// `Mistral3` wrapper with `mistral4` (MLA) text tower in `text_config`.
    Mistral3Mistral4Wrapper,
    /// `Mistral3` wrapper whose inner text tower should fall back to `Llama`.
    Mistral3LlamaFallback,
    /// Vision-language model families routed through `src/loading/vlm*.rs`.
    Vlm,
    /// Text families with directory loaders that do not fit the standard registry.
    Nonstandard,
    /// Standard config-backed text families.
    ConfigBacked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WeightLoadRoute {
    /// `Llama` plus the `Mistral3` text-wrapper fallback path.
    LlamaFamily,
    /// Architectures that require owned weights or custom sanitization.
    Special,
    /// Standard config-backed text families.
    ConfigBacked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ModelLoadPolicy {
    pub(crate) descriptor: StaticModelDescriptor,
    pub(crate) capabilities: ModelCapabilities,
    pub(crate) directory_route: DirectoryLoadRoute,
    pub(crate) weight_route: Option<WeightLoadRoute>,
}

macro_rules! for_each_model_registration {
    ($macro:ident) => {
        $macro! {
            Llama => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::Llama3Model::load, args: models::llama3::ModelArgs, weight_builder: models::Llama3Model::from_weights, wrap: LoadedModel::Llama } };
            Llama4 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::Llama4CxxModel::load, args: models::llama4::TextArgs, weight_builder: models::Llama4CxxModel::from_weights, wrap: |m| LoadedModel::Llama4(models::Llama4Wrapper::new(m)) } };
            Llama4VLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Llama4 VLM cannot be loaded with LoRA adapters yet") };
            MllamaVLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Llama 3.2 Vision (mllama) does not support adapter loading; use load_model() instead") };
            Qwen2 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::Qwen2Model::load, args: models::llama3::ModelArgs, weight_builder: models::Qwen2Model::from_weights, wrap: LoadedModel::Qwen2 } };
            Qwen3 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::Qwen3Model::load, args: models::qwen3::ModelArgs, weight_builder: models::Qwen3Model::from_weights, wrap: LoadedModel::Qwen3 } };
            Qwen3Moe => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::Qwen3MoeModel::load, args: models::qwen3_moe::ModelArgs, weight_builder: models::Qwen3MoeModel::from_weights, wrap: LoadedModel::Qwen3Moe } };
            Qwen3Next => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::Qwen3NextModel::load, args: models::qwen3_next::Qwen3NextConfig, weight_builder: models::Qwen3NextModel::from_weights, wrap: LoadedModel::Qwen3Next } };
            Qwen35 => { kind: Text, directory: Nonstandard, weight: Some(WeightLoadRoute::Special), adapter: None };
            Qwen35VLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Qwen3.5 VLM does not support adapter loading") };
            Qwen35Moe => { kind: Text, directory: Nonstandard, weight: Some(WeightLoadRoute::Special), adapter: None };
            Qwen35MoeVLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Qwen3.5 VLM does not support adapter loading") };
            Gemma => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::GemmaModel::load, args: models::gemma::ModelArgs, weight_builder: models::GemmaModel::from_weights, wrap: LoadedModel::Gemma } };
            Gemma2 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::Gemma2Model::load, args: models::gemma2::ModelArgs, weight_builder: models::Gemma2Model::from_weights, wrap: LoadedModel::Gemma2 } };
            Gemma3 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::Gemma3Model::load, args: models::gemma3::ModelArgs, weight_builder: models::Gemma3Model::from_weights, wrap: |m| LoadedModel::Gemma3(models::Gemma3Wrapper::new(m)) } };
            Gemma4 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::Gemma4Model::load, args: models::gemma4::ModelArgs, weight_builder: models::Gemma4Model::from_weights, wrap: |m| LoadedModel::Gemma4(models::Gemma4Wrapper::new(m)) } };
            DiffusionGemma => { kind: Text, directory: Nonstandard, weight: None, adapter: Some("DiffusionGemma cannot be loaded with LoRA adapters yet") };
            Gemma3VLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Gemma3 VLM cannot be loaded with LoRA adapters yet") };
            Gemma4VLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Gemma4 VLM cannot be loaded with LoRA adapters yet") };
            Gemma4Unified => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Gemma4 Unified cannot be loaded with LoRA adapters yet") };
            LlavaVLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("LLaVA VLM cannot be loaded with LoRA adapters yet") };
            GraniteVisionVLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Granite Vision VLM cannot be loaded with LoRA adapters yet") };
            Granite4VisionVLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Granite 4 Vision VLM cannot be loaded with LoRA adapters yet") };
            DeepSeekOcrVLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("DeepSeek-OCR VLM cannot be loaded with LoRA adapters yet") };
            LlavaBunnyVLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("LLaVA VLM cannot be loaded with LoRA adapters yet") };
            AyaVisionVLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Aya Vision VLM cannot be loaded with LoRA adapters yet") };
            PaliGemmaVLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("PaliGemma VLM cannot be loaded with LoRA adapters yet") };
            PixtralVLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Pixtral/Mistral3 VLM cannot be loaded with LoRA adapters yet") };
            Mistral3VLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Pixtral/Mistral3 VLM cannot be loaded with LoRA adapters yet") };
            Qwen2VL => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Qwen2-VL cannot be loaded with LoRA adapters yet") };
            Qwen25VL => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Qwen VL models cannot be loaded with LoRA adapters yet") };
            Qwen3VL => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Qwen VL models cannot be loaded with LoRA adapters yet") };
            Qwen3VLMoe => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Qwen VL models cannot be loaded with LoRA adapters yet") };
            PaddleOcrVL => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("PaddleOCR-VL cannot be loaded with LoRA adapters yet") };
            Glm4v => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("GLM-4V cannot be loaded with LoRA adapters yet") };
            Glm4vMoe => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("GLM-4V MoE cannot be loaded with LoRA adapters yet") };
            GlmOcr => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("GLM-OCR cannot be loaded with LoRA adapters yet") };
            YoutuVLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Youtu-VL VLM does not support adapter loading; use load_model() instead") };
            InternVLChatVLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("InternVL VLM does not support adapter loading; use load_model() instead") };
            KimiVL => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Kimi-VL VLM does not support adapter loading; use load_model() instead") };
            KimiK25 => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Kimi-VL 2.5 VLM does not support adapter loading; use load_model() instead") };
            SmolVLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("SmolVLM VLM does not support adapter loading; use load_model() instead") };
            Idefics2 => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Idefics2 VLM does not support adapter loading; use load_model() instead") };
            Lfm2VL => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("LFM2-VL VLM does not support adapter loading; use load_model() instead") };
            MiniCPMOVLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("MiniCPM-o VLM does not support adapter loading; use load_model() instead") };
            MiniCPMV46VLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("MiniCPM-V 4.6 VLM does not support adapter loading; use load_model() instead") };
            Moondream3VLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Moondream3 VLM does not support adapter loading; use load_model() instead") };
            Moondream2VLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Moondream2 VLM does not support adapter loading; use load_model() instead") };
            GptOss => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::GptOssModel::load, args: models::gpt_oss::ModelArgs, weight_builder: models::GptOssModel::from_weights, wrap: |m| LoadedModel::GptOss(models::GptOssWrapper::new(m)) } };
            Qwen2Moe => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::Qwen2MoeModel::load, args: models::qwen2_moe::ModelArgs, weight_builder: models::Qwen2MoeModel::from_weights, wrap: LoadedModel::Qwen2Moe } };
            Gemma3n => { kind: Text, directory: Nonstandard, weight: Some(WeightLoadRoute::Special), adapter: None };
            Gemma3nVLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Gemma3n VLM cannot be loaded with LoRA adapters yet") };
            Phi => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::PhiModel::load, args: models::phi::ModelArgs, weight_builder: models::PhiModel::from_weights, wrap: LoadedModel::Phi } };
            Phi3 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::Phi3Model::load, args: models::phi3::ModelArgs, weight_builder: models::Phi3Model::from_weights, wrap: LoadedModel::Phi3 } };
            Phi4MMVLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Phi4MM VLM does not support adapter loading; use load_model() instead") };
            Phi4SigLipVLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Phi4-SigLIP VLM does not support adapter loading; use load_model() instead") };
            Phi3VLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Phi3V VLM does not support adapter loading; use load_model() instead") };
            MolmoVLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Molmo VLM does not support adapter loading; use load_model() instead") };
            Molmo2VLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Molmo2 VLM does not support adapter loading; use load_model() instead") };
            MolmoPointVLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Molmo-Point VLM does not support adapter loading; use load_model() instead") };
            Phi3Small => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::Phi3SmallModel::load, args: models::phi3small::ModelArgs, weight_builder: models::Phi3SmallModel::from_weights, wrap: LoadedModel::Phi3Small } };
            PhiMoe => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::PhiMoeModel::load, args: models::phimoe::ModelArgs, weight_builder: models::PhiMoeModel::from_weights, wrap: LoadedModel::PhiMoe } };
            MiniMax => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::MiniMaxModel::load, args: models::minimax::ModelArgs, weight_builder: models::MiniMaxModel::from_weights, wrap: LoadedModel::MiniMax } };
            Mixtral => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::MixtralModel::load, args: models::mixtral::ModelArgs, weight_builder: models::MixtralModel::from_weights, wrap: LoadedModel::Mixtral } };
            OLMoE => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::OlmoeModel::load, args: models::olmoe::ModelArgs, weight_builder: models::OlmoeModel::from_weights, wrap: LoadedModel::OLMoE } };
            DeepSeek => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::DeepSeekModel::load, args: models::deepseek::ModelArgs, weight_builder: models::DeepSeekModel::from_weights, wrap: LoadedModel::DeepSeek } };
            DeepSeekV2 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::DeepSeekV2Model::load, args: models::deepseek_v2::ModelArgs, weight_builder: models::DeepSeekV2Model::from_weights, wrap: LoadedModel::DeepSeekV2 } };
            DeepSeekV3 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::DeepSeekV3Model::load, args: models::deepseek_v3::DeepSeekV3Config, weight_builder: models::DeepSeekV3Model::from_weights, wrap: LoadedModel::DeepSeekV3 } };
            DeepSeekV32 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::DeepSeekV32Model::load, args: models::deepseek_v32::ModelArgs, weight_builder: models::DeepSeekV32Model::from_weights, wrap: LoadedModel::DeepSeekV32 } };
            Dots1 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::Dots1Model::load, args: models::dots1::ModelArgs, weight_builder: models::Dots1Model::from_weights, wrap: LoadedModel::Dots1 } };
            Cohere => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::CohereModel::load, args: models::cohere::ModelArgs, weight_builder: models::CohereModel::from_weights, wrap: LoadedModel::Cohere } };
            Cohere2 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::Cohere2Model::load, args: models::cohere2::Cohere2Config, weight_builder: models::Cohere2Model::from_weights, wrap: LoadedModel::Cohere2 } };
            InternLM2 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::InternLM2Model::load, args: models::internlm2::ModelArgs, weight_builder: models::InternLM2Model::from_weights, wrap: LoadedModel::InternLM2 } };
            InternLM3 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::InternLM3Model::load, args: models::internlm3::ModelArgs, weight_builder: models::InternLM3Model::from_weights, wrap: LoadedModel::InternLM3 } };
            Baichuan => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::BaichuanModel::load, args: models::baichuan::BaichuanConfig, weight_builder: models::BaichuanModel::from_weights, wrap: LoadedModel::Baichuan } };
            Glm4 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::Glm4Model::load, args: models::glm4::ModelArgs, weight_builder: models::Glm4Model::from_weights, wrap: LoadedModel::Glm4 } };
            Glm4Moe => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::Glm4MoeModel::load, args: models::glm4_moe::ModelArgs, weight_builder: models::Glm4MoeModel::from_weights, wrap: LoadedModel::Glm4Moe } };
            Glm4MoeLite => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::Glm4MoeLiteModel::load, args: models::glm4_moe_lite::ModelArgs, weight_builder: models::Glm4MoeLiteModel::from_weights, wrap: LoadedModel::Glm4MoeLite } };
            GlmMoeDsa => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::GlmMoeDsaModel::load, args: models::glm_moe_dsa::ModelArgs, weight_builder: models::GlmMoeDsaModel::from_weights, wrap: LoadedModel::GlmMoeDsa } };
            Ernie45 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::Ernie45Model::load, args: models::ernie4_5::ModelArgs, weight_builder: models::Ernie45Model::from_weights, wrap: LoadedModel::Ernie45 } };
            Ernie45Moe => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::Ernie45MoeModel::load, args: models::ernie4_5_moe::ModelArgs, weight_builder: models::Ernie45MoeModel::from_weights, wrap: LoadedModel::Ernie45Moe } };
            HunyuanMoe => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::HunyuanMoeModel::load, args: models::hunyuan_moe::ModelArgs, weight_builder: models::HunyuanMoeModel::from_weights, wrap: LoadedModel::HunyuanMoe } };
            HunyuanV1Dense => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::HunyuanV1DenseModel::load, args: models::hunyuan_v1_dense::ModelArgs, weight_builder: models::HunyuanV1DenseModel::from_weights, wrap: LoadedModel::HunyuanV1Dense } };
            MiMo => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::MiMoModel::load, args: models::mimo::ModelArgs, weight_builder: models::MiMoModel::from_weights, wrap: LoadedModel::MiMo } };
            Apertus => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::ApertusModel::load, args: models::apertus::ModelArgs, weight_builder: models::ApertusModel::from_weights, wrap: LoadedModel::Apertus } };
            SeedOss => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::SeedOssModel::load, args: models::seed_oss::ModelArgs, weight_builder: models::SeedOssModel::from_weights, wrap: LoadedModel::SeedOss } };
            Granite => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::GraniteModel::load, args: models::granite::ModelArgs, weight_builder: models::GraniteModel::from_weights, wrap: LoadedModel::Granite } };
            BitNet => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::BitNetModel::load, args: models::bitnet::BitNetConfig, weight_builder: models::BitNetModel::from_weights, wrap: LoadedModel::BitNet } };
            ExaOne => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::ExaOneModel::load, args: models::exaone::ExaOneConfig, weight_builder: models::ExaOneModel::from_weights, wrap: LoadedModel::ExaOne } };
            ExaOne4 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::ExaOne4Model::load, args: models::exaone4::ModelArgs, weight_builder: models::ExaOne4Model::from_weights, wrap: |m| LoadedModel::ExaOne4(models::ExaOne4Wrapper::new(m)) } };
            ExaOneMoe => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::ExaoneMoeModel::load, args: models::exaone_moe::ModelArgs, weight_builder: models::ExaoneMoeModel::from_weights, wrap: LoadedModel::ExaOneMoe } };
            SolarOpen => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::SolarOpenModel::load, args: models::solar_open::ModelArgs, weight_builder: models::SolarOpenModel::from_weights, wrap: LoadedModel::SolarOpen } };
            Olmo => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::OlmoModel::load, args: models::olmo::ModelArgs, weight_builder: models::OlmoModel::from_weights, wrap: LoadedModel::Olmo } };
            Olmo2 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::OLMo2Model::load, args: models::olmo2::ModelArgs, weight_builder: models::OLMo2Model::from_weights, wrap: LoadedModel::Olmo2 } };
            Olmo3 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::OLMo3Model::load, args: models::olmo3::OLMo3Config, weight_builder: models::OLMo3Model::from_weights, wrap: LoadedModel::Olmo3 } };
            StarCoder2 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::StarCoder2Model::load, args: models::starcoder2::StarCoder2Config, weight_builder: models::StarCoder2Model::from_weights, wrap: LoadedModel::StarCoder2 } };
            Mellum => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::MellumModel::load, args: models::mellum::ModelArgs, weight_builder: models::MellumModel::from_weights, wrap: |m| LoadedModel::Mellum(models::MellumWrapper::new(m)) } };
            MiniCPM => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::MiniCPMModel::load, args: models::minicpm::ModelArgs, weight_builder: models::MiniCPMModel::from_weights, wrap: LoadedModel::MiniCPM } };
            MiniCPM3 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::MiniCPM3Model::load, args: models::minicpm3::ModelArgs, weight_builder: models::MiniCPM3Model::from_weights, wrap: LoadedModel::MiniCPM3 } };
            StableLM => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::StableLMModel::load, args: models::stablelm::ModelArgs, weight_builder: models::StableLMModel::from_weights, wrap: LoadedModel::StableLM } };
            SmolLM3 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::SmolLM3Model::load, args: models::smollm3::ModelArgs, weight_builder: models::SmolLM3Model::from_weights, wrap: LoadedModel::SmolLM3 } };
            Ministral3 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::Ministral3Model::load, args: models::ministral3::ModelArgs, weight_builder: models::Ministral3Model::from_weights, wrap: |m| LoadedModel::Ministral3(models::Ministral3Wrapper::new(m)) } };
            Mistral3 => { kind: Text, directory: Mistral3Dynamic, weight: Some(WeightLoadRoute::LlamaFamily), adapter: None };
            Mistral4 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::Mistral4Model::load, args: models::mistral4::Mistral4Config, weight_builder: models::Mistral4Model::from_weights, wrap: LoadedModel::Mistral4 } };
            Nemotron => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::NemotronModel::load, args: models::nemotron::ModelArgs, weight_builder: models::NemotronModel::from_weights, wrap: LoadedModel::Nemotron } };
            Mamba => { kind: Text, directory: Nonstandard, weight: Some(WeightLoadRoute::Special), adapter: None };
            Mamba2 => { kind: Text, directory: Nonstandard, weight: Some(WeightLoadRoute::Special), adapter: None };
            Jamba => { kind: Text, directory: Nonstandard, weight: Some(WeightLoadRoute::Special), adapter: None };
            FalconH1 => { kind: Text, directory: Nonstandard, weight: Some(WeightLoadRoute::Special), adapter: None };
            Lfm2 => { kind: Text, directory: Nonstandard, weight: Some(WeightLoadRoute::Special), adapter: None };
            Lfm2Moe => { kind: Text, directory: Nonstandard, weight: Some(WeightLoadRoute::Special), adapter: None };
            Plamo2 => { kind: Text, directory: Nonstandard, weight: Some(WeightLoadRoute::Special), adapter: None };
            GraniteMoeHybrid => { kind: Text, directory: Nonstandard, weight: Some(WeightLoadRoute::Special), adapter: None };
            NemotronH => { kind: Text, directory: Nonstandard, weight: Some(WeightLoadRoute::Special), adapter: None };
            NemotronHNanoOmniVLM => { kind: Vlm, directory: Vlm, weight: None, adapter: Some("Nemotron H Nano Omni VLM does not support adapter loading; use load_model() instead") };
            NemotronNAS => { kind: Text, directory: Nonstandard, weight: Some(WeightLoadRoute::Special), adapter: None };
            KimiLinear => { kind: Text, directory: Nonstandard, weight: Some(WeightLoadRoute::Special), adapter: None };
            LongcatFlash => { kind: Text, directory: Nonstandard, weight: Some(WeightLoadRoute::Special), adapter: None };
            LongcatFlashNgram => { kind: Text, directory: Nonstandard, weight: Some(WeightLoadRoute::Special), adapter: None };
            Step3p5 => { kind: Text, directory: ConfigBacked, weight: Some(WeightLoadRoute::ConfigBacked), adapter: None, config_backed: { dir_loader: models::Step3p5Model::load, args: models::step3p5::Step3p5Config, weight_builder: models::Step3p5Model::from_weights, wrap: LoadedModel::Step3p5 } };
            Rwkv7 => { kind: Text, directory: Nonstandard, weight: Some(WeightLoadRoute::Special), adapter: None };
            RecurrentGemma => { kind: Text, directory: Nonstandard, weight: Some(WeightLoadRoute::Special), adapter: None };
            Whisper => { kind: Text, directory: Nonstandard, weight: None, adapter: Some("Whisper ASR checkpoints are served through the /v1/audio/* endpoints, not text generation or adapter loading") };
            Kokoro => { kind: Text, directory: Nonstandard, weight: None, adapter: Some("Kokoro TTS checkpoints are served through the /v1/audio/speech endpoint, not text generation or adapter loading") };
        }
    };
}

macro_rules! descriptor_match {
    ($( $variant:ident => { kind: $kind:ident, directory: $directory:ident, weight: $weight:expr, adapter: $adapter:expr $(, config_backed: { dir_loader: $dir_loader:path, args: $args_ty:ty, weight_builder: $weight_builder:path, wrap: $wrap:expr })? }; )*) => {
        pub(crate) fn static_model_descriptor(model_type: ModelType) -> StaticModelDescriptor {
            match model_type {
                $(
                    ModelType::$variant => StaticModelDescriptor {
                        model_type,
                        kind: ModelKind::$kind,
                        directory_family: DirectoryRouteFamily::$directory,
                        adapter_weight_route: $weight,
                        adapter_unsupported_message: $adapter,
                    },
                )*
            }
        }
    };
}

for_each_model_registration!(descriptor_match);
pub(crate) use for_each_model_registration;

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn has_config_backed_registration(model_type: ModelType) -> bool {
    is_config_backed_model_type(model_type)
}

pub(crate) fn is_ministral3_config(config: &serde_json::Value) -> bool {
    config
        .get("text_config")
        .and_then(|tc| tc.get("model_type"))
        .and_then(|mt| mt.as_str())
        .map(|mt| mt == "ministral3")
        .unwrap_or(false)
}

pub(crate) fn is_mistral4_config(config: &serde_json::Value) -> bool {
    config
        .get("text_config")
        .and_then(|tc| tc.get("model_type"))
        .and_then(|mt| mt.as_str())
        .map(|mt| mt == "mistral4")
        .unwrap_or(false)
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn is_vlm_model_type(model_type: ModelType) -> bool {
    static_model_descriptor(model_type).kind == ModelKind::Vlm
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn is_config_backed_model_type(model_type: ModelType) -> bool {
    static_model_descriptor(model_type).directory_family == DirectoryRouteFamily::ConfigBacked
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn is_nonstandard_model_type(model_type: ModelType) -> bool {
    static_model_descriptor(model_type).directory_family == DirectoryRouteFamily::Nonstandard
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn adapter_loading_unsupported_message(model_type: ModelType) -> Option<&'static str> {
    static_model_descriptor(model_type).adapter_unsupported_message
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn is_special_weight_model_type(model_type: ModelType) -> bool {
    static_model_descriptor(model_type).adapter_weight_route == Some(WeightLoadRoute::Special)
}

pub(crate) fn model_capabilities(model_type: ModelType) -> ModelCapabilities {
    let descriptor = static_model_descriptor(model_type);
    ModelCapabilities {
        kind: descriptor.kind,
        adapter_unsupported_message: descriptor.adapter_unsupported_message,
    }
}

pub(crate) fn directory_load_route(
    model_type: ModelType,
    config: Option<&serde_json::Value>,
) -> Result<DirectoryLoadRoute> {
    let descriptor = static_model_descriptor(model_type);
    Ok(match descriptor.directory_family {
        DirectoryRouteFamily::Mistral3Dynamic => {
            if config.is_some_and(is_ministral3_config) {
                DirectoryLoadRoute::Mistral3TextWrapper
            } else if config.is_some_and(is_mistral4_config) {
                DirectoryLoadRoute::Mistral3Mistral4Wrapper
            } else {
                DirectoryLoadRoute::Mistral3LlamaFallback
            }
        }
        DirectoryRouteFamily::Vlm => DirectoryLoadRoute::Vlm,
        DirectoryRouteFamily::Nonstandard => DirectoryLoadRoute::Nonstandard,
        DirectoryRouteFamily::ConfigBacked => DirectoryLoadRoute::ConfigBacked,
    })
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn weight_load_route(model_type: ModelType) -> Result<WeightLoadRoute> {
    static_model_descriptor(model_type)
        .adapter_weight_route
        .ok_or_else(|| anyhow::anyhow!("Missing weight loader for model type: {:?}", model_type))
}

pub(crate) fn model_load_policy(
    model_type: ModelType,
    config: Option<&serde_json::Value>,
) -> Result<ModelLoadPolicy> {
    let descriptor = static_model_descriptor(model_type);
    let capabilities = model_capabilities(model_type);
    let directory_route = directory_load_route(model_type, config)?;
    let weight_route = if capabilities.adapter_unsupported_message.is_some() {
        None
    } else {
        descriptor.adapter_weight_route
    };

    Ok(ModelLoadPolicy {
        descriptor,
        capabilities,
        directory_route,
        weight_route,
    })
}
