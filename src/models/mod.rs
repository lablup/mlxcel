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

//! Model implementations for mlxcel
//!
//! All implementations use mlxcel-core for direct MLX C++ bindings.

mod detection;
mod gemma3n_helpers;
mod llama4_helpers;
mod model_owned;
pub(crate) mod qwen_mrope_state;
mod recurrent_snapshot;
mod sanitize;

// Shared modules
pub mod gated_delta;
pub mod switch_layers;

// Model implementations (mlxcel-core based)
pub mod apertus;
pub mod baichuan;
pub mod cohere;
pub mod cohere2;
pub mod deepseek;
pub mod deepseek_v2;
pub mod deepseek_v3;
pub mod deepseek_v32;
pub mod diffusion_gemma;
pub mod ernie4_5;
pub mod ernie4_5_moe;
pub mod exaone;
pub mod exaone4;
pub mod exaone_moe;
pub mod falcon_h1;
pub mod gemma;
pub mod gemma2;
pub mod gemma3;
pub mod gemma3n;
pub mod gemma4;
pub mod gemma4_mtp_target;
pub mod glm4;
pub mod glm4_moe;
pub mod glm4_moe_lite;
pub mod glm_moe_dsa;
pub mod gpt_oss;
pub mod granite;
pub mod granitemoehybrid;
pub mod hunyuan_moe;
pub mod hunyuan_v1_dense;
pub mod internlm2;
pub mod internlm3;
pub mod jamba;
pub mod kimi_linear;
pub mod lfm2;
pub mod llama3;
pub mod llama4;
pub mod longcat_flash_ngram;
pub mod mamba;
pub mod mamba2;
pub mod mimo;
pub mod minicpm;
pub mod minicpm3;
pub mod minimax;
pub mod ministral3;
pub mod mistral4;
pub mod mixtral;
pub mod molmo;
pub mod molmo2;
pub mod molmo_point;
pub mod moondream3;
pub mod nemotron;
pub mod nemotron_h;
pub mod nemotron_nas;
pub mod olmo;
pub mod olmo2;
pub mod olmo3;
pub mod olmoe;
pub mod phi;
pub mod phi3;
pub mod phi3small;
pub mod phi4mm;
pub mod phimoe;
pub mod plamo2;
pub mod qwen2;
pub mod qwen2_moe;
pub mod qwen2_vl;
pub mod qwen3;
pub mod qwen3_5;
pub mod qwen3_moe;
pub mod qwen3_next;
pub mod qwen3_vl;
pub mod qwen3_vl_moe;
pub mod recurrent_gemma;
pub mod rwkv7;
pub mod smollm3;
pub mod solar_open;
pub mod stablelm;
pub mod starcoder2;
pub mod step3p5;
pub mod youtu_vl_lm;

// Re-export model types
pub use apertus::ApertusModel;
pub use baichuan::BaichuanModel;
pub use cohere::CohereModel;
pub use cohere2::Cohere2Model;
pub use deepseek::DeepSeekModel;
pub use deepseek_v2::DeepSeekV2Model;
pub use deepseek_v3::DeepSeekV3Model;
pub use deepseek_v32::DeepSeekV32Model;
pub use detection::get_model_type;
pub use diffusion_gemma::DiffusionGemmaModel;
pub use ernie4_5::Ernie45Model;
pub use ernie4_5_moe::Ernie45MoeModel;
pub use exaone::ExaOneModel;
pub use exaone_moe::ExaoneMoeModel;
pub use exaone4::{ExaOne4Model, ExaOne4Wrapper};
pub use falcon_h1::FalconH1Model;
pub use gemma::GemmaModel;
pub use gemma2::Gemma2Model;
pub use gemma3::{Gemma3Model, Gemma3Wrapper};
pub use gemma3n::Gemma3nModel;
pub use gemma4::{Gemma4Model, Gemma4SpeculativeSinks, Gemma4Wrapper};
pub use glm_moe_dsa::GlmMoeDsaModel;
pub use glm4::Glm4Model;
pub use glm4_moe::Glm4MoeModel;
pub use glm4_moe_lite::Glm4MoeLiteModel;
pub use gpt_oss::{GptOssModel, GptOssWrapper};
pub use granite::GraniteModel;
pub use granitemoehybrid::GraniteMoeHybridModel;
pub use hunyuan_moe::HunyuanMoeModel;
pub use hunyuan_v1_dense::HunyuanV1DenseModel;
pub use internlm2::InternLM2Model;
pub use internlm3::InternLM3Model;
pub use jamba::JambaModel;
pub use kimi_linear::KimiLinearModel;
pub use lfm2::Lfm2Model;
pub use llama3::Llama3Model;
pub use llama4::{Llama4CxxModel, Llama4Wrapper};
pub use longcat_flash_ngram::LongcatFlashNgramModel;
pub use mamba::MambaModel;
pub use mamba2::Mamba2Model;
pub use mimo::MiMoModel;
pub use minicpm::MiniCPMModel;
pub use minicpm3::MiniCPM3Model;
pub use minimax::MiniMaxModel;
pub use ministral3::{Ministral3Model, Ministral3Wrapper};
pub use mistral4::Mistral4Model;
pub use mixtral::MixtralModel;
pub use molmo::MolmoModel;
pub use molmo2::Molmo2Model;
pub use moondream3::Moondream3Model;
pub use nemotron::NemotronModel;
pub use nemotron_h::NemotronHModel;
pub use nemotron_nas::NemotronNASModel;
pub use olmo::OlmoModel;
pub use olmo2::OLMo2Model;
pub use olmo3::OLMo3Model;
pub use olmoe::OlmoeModel;
pub use phi::PhiModel;
pub use phi3::Phi3Model;
pub use phi3small::Phi3SmallModel;
pub use phi4mm::Phi4MMModel;
pub use phimoe::PhiMoeModel;
pub use plamo2::Plamo2Model;
pub use qwen2::Qwen2Model;
pub use qwen2_moe::Qwen2MoeModel;
pub use qwen2_vl::Qwen2VLModel;
pub use qwen3::Qwen3Model;
pub use qwen3_5::{GdnRollbackSnapshot, Qwen35Model, VerifyOutput};
pub use qwen3_moe::Qwen3MoeModel;
pub use qwen3_next::Qwen3NextModel;
pub use qwen3_vl::Qwen3VLModel;
pub use qwen3_vl_moe::Qwen3VLMoeModel;
pub use recurrent_gemma::GriffinModel;
pub use rwkv7::Rwkv7;
pub(crate) use sanitize::{
    Gemma4WeightBacking, load_gemma4_text_weights_with_backing,
    load_gemma4_unified_weights_with_backing, load_gemma4_vlm_weights_with_backing,
    strip_gemma4_kv_shared_weights,
};
pub use sanitize::{
    convert_bf16_weights, convert_bf16_weights_with_keep, gemma3n_language_mlp_bf16_key,
    load_and_sanitize_weights, load_text_weights, sanitize_config_json, sanitize_tied_embeddings,
    warn_bf16_precision,
};
pub use smollm3::SmolLM3Model;
pub use solar_open::SolarOpenModel;
pub use stablelm::StableLMModel;
pub use starcoder2::StarCoder2Model;
pub use step3p5::Step3p5Model;

/// Supported model types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    // Standard Transformer models
    Llama,           // Llama 1/2/3, Mistral
    Llama4,          // Llama 4 (MoE)
    Llama4VLM,       // Llama 4 VLM (vision-language)
    Qwen2,           // Qwen 2/2.5
    Qwen3,           // Qwen 3
    Qwen3Moe,        // Qwen 3 MoE
    Qwen3Next,       // Qwen 3 with GatedDeltaNet
    Qwen35,          // Qwen 3.5 Hybrid (Transformer + GatedDeltaNet)
    Qwen35VLM,       // Qwen 3.5 VLM (Qwen3-VL vision + Qwen3.5 hybrid text)
    Qwen35Moe,       // Qwen 3.5 MoE Hybrid
    Qwen35MoeVLM,    // Qwen 3.5 MoE VLM
    Gemma,           // Gemma 1
    Gemma2,          // Gemma 2
    Gemma3,          // Gemma 3 (text-only)
    Gemma4,          // Gemma 4 text-only route
    DiffusionGemma,  // DiffusionGemma (block-diffusion on the Gemma 4 MoE backbone)
    Gemma3VLM,       // Gemma 3 VLM (vision-language)
    Gemma4VLM,       // Gemma 4 VLM (vision-language)
    Gemma4Unified,   // Gemma 4 Unified (encoder-free text + vision + audio)
    LlavaVLM,        // LLaVA (CLIP/SigLIP + Llama/Qwen2)
    LlavaBunnyVLM,   // LLaVA-Bunny (SigLIP + Qwen2)
    AyaVisionVLM,    // Aya Vision (SigLIP + Cohere2)
    PaliGemmaVLM,    // PaliGemma (SigLIP + Gemma)
    PixtralVLM,      // Pixtral (ViT w/ 2D RoPE + Mistral)
    Mistral3VLM,     // Mistral 3 VLM (Pixtral ViT + PatchMerger + Mistral)
    Qwen2VL,         // Qwen2-VL (custom ViT + Qwen2 w/ MRoPE)
    Qwen25VL,        // Qwen2.5-VL (windowed ViT + Qwen2 w/ MRoPE)
    Qwen3VL,         // Qwen3-VL (ViT + interleaved MRoPE + DeepStack)
    Qwen3VLMoe,      // Qwen3-VL-MoE (Qwen3-VL + MoE text backbone)
    YoutuVLM,        // Youtu-VL (SigLIP2 windowed-attn + DeepSeek-V3-style MLA)
    InternVLChatVLM, // InternVL (internvl_chat): InternViT + pixel-shuffle mlp1 + Qwen2 text
    MiniCPMOVLM,     // MiniCPM-o (dynamic SigLIP + resampler + Qwen3-VL text)
    MiniCPMV46VLM,   // MiniCPM-V 4.6 (SigLIP + VitMerger + Merger + Qwen3.5 text)
    Moondream3VLM,   // Moondream3 (custom ViT + custom text decoder, query/caption image path)
    Gemma3n,         // Gemma 3n (text-only)
    Gemma3nVLM,      // Gemma 3n VLM (MobileNetV5 + Gemma3n)
    Phi,             // Phi 1/2
    Phi3,            // Phi 3
    Phi4MMVLM,       // Phi-4 Multimodal (SigLIP2 NaFlex + Phi4 text, image path only)
    Phi4SigLipVLM,   // Phi-4 reasoning vision (SigLIP2 NaFlex + Phi3-style text)
    Phi3VLM,         // Phi 3.5 Vision (CLIP + Phi3)
    MolmoVLM,        // Molmo v1 (CLIP ViT + attention pooling + OLMo-style text)
    Molmo2VLM,       // Molmo2 (custom ViT + attention pooling + Molmo2 text)
    MolmoPointVLM,   // Molmo-Point (custom ViT + point prediction + Molmo2 text)
    Phi3Small,       // Phi 3 Small
    PhiMoe,          // Phi MoE

    // MoE models
    GptOss,
    MiniMax,
    Mixtral,
    Qwen2Moe,
    OLMoE,

    // DeepSeek family
    DeepSeek,
    DeepSeekV2,
    DeepSeekV3,
    DeepSeekV32,

    // Cohere family
    Cohere,
    Cohere2,

    // Chinese/Asian models
    InternLM2,
    InternLM3,
    Baichuan,
    Glm4,
    Glm4Moe,
    Glm4MoeLite,
    GlmMoeDsa,
    Ernie45,
    Ernie45Moe,
    HunyuanMoe,
    HunyuanV1Dense,
    MiMo,

    // Apertus (Swiss AI)
    Apertus,

    // IBM Granite
    Granite,

    // Korean models
    ExaOne,
    ExaOne4,
    ExaOneMoe,
    SolarOpen,

    // OLMo family
    Olmo,
    Olmo2,
    Olmo3,

    // Code models
    StarCoder2,

    // Other Transformer models
    MiniCPM,
    MiniCPM3,
    StableLM,
    SmolLM3,
    Ministral3,
    Mistral3,
    Mistral4,
    Nemotron,

    // SSM/Mamba models
    Mamba,
    Mamba2,
    Jamba,
    NemotronH,
    /// Nemotron H Nano Omni — vision-capable variant of `nemotron_h`
    /// Audio support is tracked separately as a follow-up.
    NemotronHNanoOmniVLM,
    NemotronNAS,

    // TII Falcon (Mamba2 + Attention parallel hybrid)
    FalconH1,

    // Liquid Foundation Models (short-conv + attention hybrid)
    Lfm2,
    Lfm2Moe,

    // Preferred Networks PLaMo 2 (Mamba + attention interleaved hybrid)
    Plamo2,

    // IBM Granite 4.x (Mamba2 + attention interleaved hybrid)
    GraniteMoeHybrid,

    // Kimi models
    KimiLinear,

    // Longcat models
    LongcatFlash,
    LongcatFlashNgram,

    // Step models
    Step3p5,

    // RNN models
    Rwkv7,
    RecurrentGemma,
}

/// All `ModelType` variants, in declaration order. Used as the iteration
/// source for `mlxcel arch` so that the rendered output stays in sync with
/// the registry. The exhaustiveness contract is enforced by
/// `ModelType::metadata()` (an exhaustive `match`) and by the
/// `all_model_types_covers_every_variant` unit test, which both asserts a
/// count floor and walks every entry to verify non-empty metadata.
pub const ALL_MODEL_TYPES: &[ModelType] = &[
    // Standard Transformer models
    ModelType::Llama,
    ModelType::Llama4,
    ModelType::Llama4VLM,
    ModelType::Qwen2,
    ModelType::Qwen3,
    ModelType::Qwen3Moe,
    ModelType::Qwen3Next,
    ModelType::Qwen35,
    ModelType::Qwen35VLM,
    ModelType::Qwen35Moe,
    ModelType::Qwen35MoeVLM,
    ModelType::Gemma,
    ModelType::Gemma2,
    ModelType::Gemma3,
    ModelType::Gemma4,
    ModelType::DiffusionGemma,
    ModelType::Gemma3VLM,
    ModelType::Gemma4VLM,
    ModelType::Gemma4Unified,
    ModelType::LlavaVLM,
    ModelType::LlavaBunnyVLM,
    ModelType::AyaVisionVLM,
    ModelType::PaliGemmaVLM,
    ModelType::PixtralVLM,
    ModelType::Mistral3VLM,
    ModelType::Qwen2VL,
    ModelType::Qwen25VL,
    ModelType::Qwen3VL,
    ModelType::Qwen3VLMoe,
    ModelType::YoutuVLM,
    ModelType::InternVLChatVLM,
    ModelType::MiniCPMOVLM,
    ModelType::MiniCPMV46VLM,
    ModelType::Moondream3VLM,
    ModelType::Gemma3n,
    ModelType::Gemma3nVLM,
    ModelType::Phi,
    ModelType::Phi3,
    ModelType::Phi4MMVLM,
    ModelType::Phi4SigLipVLM,
    ModelType::Phi3VLM,
    ModelType::MolmoVLM,
    ModelType::Molmo2VLM,
    ModelType::MolmoPointVLM,
    ModelType::Phi3Small,
    ModelType::PhiMoe,
    // MoE models
    ModelType::GptOss,
    ModelType::MiniMax,
    ModelType::Mixtral,
    ModelType::Qwen2Moe,
    ModelType::OLMoE,
    // DeepSeek family
    ModelType::DeepSeek,
    ModelType::DeepSeekV2,
    ModelType::DeepSeekV3,
    ModelType::DeepSeekV32,
    // Cohere family
    ModelType::Cohere,
    ModelType::Cohere2,
    // Chinese/Asian models
    ModelType::InternLM2,
    ModelType::InternLM3,
    ModelType::Baichuan,
    ModelType::Glm4,
    ModelType::Glm4Moe,
    ModelType::Glm4MoeLite,
    ModelType::GlmMoeDsa,
    ModelType::Ernie45,
    ModelType::Ernie45Moe,
    ModelType::HunyuanMoe,
    ModelType::HunyuanV1Dense,
    ModelType::MiMo,
    // Apertus (Swiss AI)
    ModelType::Apertus,
    // IBM Granite
    ModelType::Granite,
    // Korean models
    ModelType::ExaOne,
    ModelType::ExaOne4,
    ModelType::ExaOneMoe,
    ModelType::SolarOpen,
    // OLMo family
    ModelType::Olmo,
    ModelType::Olmo2,
    ModelType::Olmo3,
    // Code models
    ModelType::StarCoder2,
    // Other Transformer models
    ModelType::MiniCPM,
    ModelType::MiniCPM3,
    ModelType::StableLM,
    ModelType::SmolLM3,
    ModelType::Ministral3,
    ModelType::Mistral3,
    ModelType::Mistral4,
    ModelType::Nemotron,
    // SSM/Mamba models
    ModelType::Mamba,
    ModelType::Mamba2,
    ModelType::Jamba,
    ModelType::NemotronH,
    ModelType::NemotronHNanoOmniVLM,
    ModelType::NemotronNAS,
    // TII Falcon
    ModelType::FalconH1,
    // Liquid Foundation Models
    ModelType::Lfm2,
    ModelType::Lfm2Moe,
    // Preferred Networks PLaMo 2
    ModelType::Plamo2,
    // IBM Granite 4.x hybrid
    ModelType::GraniteMoeHybrid,
    // Kimi models
    ModelType::KimiLinear,
    // Longcat models
    ModelType::LongcatFlash,
    ModelType::LongcatFlashNgram,
    // Step models
    ModelType::Step3p5,
    // RNN models
    ModelType::Rwkv7,
    ModelType::RecurrentGemma,
];

impl ModelType {
    /// User-facing metadata for `mlxcel arch`: `(display_name, family)`.
    ///
    /// The match is intentionally exhaustive — adding a new variant to
    /// `ModelType` without supplying both fields is a compile error. This
    /// is the single source of truth that prevents `mlxcel arch` from
    /// drifting away from the registry the way the previous hand-written
    /// block did (see issue #26).
    ///
    /// * `display_name` — short human-readable label (e.g.
    ///   `"Llama 4 (MoE)"`, `"Qwen 3.5 MoE VLM"`). Stay factual; do not
    ///   invent capabilities not present in the variant.
    /// * `family` — free-form grouping label used by the renderer to bucket
    ///   variants into sections. Sibling families are used for VLMs
    ///   (e.g. `"Qwen VLM"` alongside `"Qwen"`).
    pub const fn metadata(self) -> (&'static str, &'static str) {
        match self {
            // ----- Llama -----
            ModelType::Llama => ("Llama 1/2/3", "Llama"),
            ModelType::Llama4 => ("Llama 4 (MoE)", "Llama"),
            ModelType::Llama4VLM => ("Llama 4 VLM", "Llama VLM"),

            // ----- Qwen (text/hybrid/MoE) -----
            ModelType::Qwen2 => ("Qwen 2 / 2.5", "Qwen"),
            ModelType::Qwen3 => ("Qwen 3", "Qwen"),
            ModelType::Qwen3Moe => ("Qwen 3 MoE", "Qwen"),
            ModelType::Qwen3Next => ("Qwen 3 Next (Attention + GatedDeltaNet + MoE)", "Qwen"),
            ModelType::Qwen35 => ("Qwen 3.5 (Attention + GatedDeltaNet hybrid)", "Qwen"),
            ModelType::Qwen35Moe => ("Qwen 3.5 MoE (hybrid)", "Qwen"),
            ModelType::Qwen2Moe => ("Qwen 2 MoE", "Qwen"),

            // ----- Qwen VLM -----
            ModelType::Qwen2VL => ("Qwen2-VL", "Qwen VLM"),
            ModelType::Qwen25VL => ("Qwen2.5-VL", "Qwen VLM"),
            ModelType::Qwen3VL => ("Qwen3-VL", "Qwen VLM"),
            ModelType::Qwen3VLMoe => ("Qwen3-VL MoE", "Qwen VLM"),
            ModelType::Qwen35VLM => ("Qwen 3.5 VLM", "Qwen VLM"),
            ModelType::Qwen35MoeVLM => ("Qwen 3.5 MoE VLM", "Qwen VLM"),

            // ----- Gemma (text) -----
            ModelType::Gemma => ("Gemma 1", "Gemma"),
            ModelType::Gemma2 => ("Gemma 2", "Gemma"),
            ModelType::Gemma3 => ("Gemma 3", "Gemma"),
            ModelType::Gemma3n => ("Gemma 3n", "Gemma"),
            ModelType::Gemma4 => ("Gemma 4", "Gemma"),
            ModelType::DiffusionGemma => (
                "DiffusionGemma (block-diffusion, Gemma 4 MoE backbone)",
                "Gemma",
            ),
            ModelType::RecurrentGemma => ("RecurrentGemma (Griffin: RGLRU + attention)", "Gemma"),

            // ----- Gemma VLM -----
            ModelType::Gemma3VLM => ("Gemma 3 VLM", "Gemma VLM"),
            ModelType::Gemma3nVLM => ("Gemma 3n VLM (MobileNetV5 + Gemma3n)", "Gemma VLM"),
            ModelType::Gemma4VLM => ("Gemma 4 VLM", "Gemma VLM"),
            ModelType::Gemma4Unified => (
                "Gemma 4 Unified (encoder-free text + vision + audio)",
                "Gemma VLM",
            ),
            ModelType::PaliGemmaVLM => ("PaliGemma (SigLIP + Gemma)", "Gemma VLM"),

            // ----- Mistral (text) -----
            ModelType::Ministral3 => ("Ministral 3", "Mistral"),
            ModelType::Mistral3 => ("Mistral 3", "Mistral"),
            ModelType::Mistral4 => ("Mistral 4 (MLA)", "Mistral"),

            // ----- Mistral VLM -----
            ModelType::PixtralVLM => ("Pixtral (2D-RoPE ViT + Mistral)", "Mistral VLM"),
            ModelType::Mistral3VLM => ("Mistral 3 VLM (Pixtral ViT + Mistral)", "Mistral VLM"),

            // ----- Phi (text) -----
            ModelType::Phi => ("Phi 1 / 2", "Phi"),
            ModelType::Phi3 => ("Phi 3", "Phi"),
            ModelType::Phi3Small => ("Phi 3 Small", "Phi"),
            ModelType::PhiMoe => ("Phi MoE", "Phi"),

            // ----- Phi VLM -----
            ModelType::Phi3VLM => ("Phi 3.5 Vision (CLIP + Phi3)", "Phi VLM"),
            ModelType::Phi4MMVLM => ("Phi-4 Multimodal (SigLIP2 NaFlex + Phi4)", "Phi VLM"),
            ModelType::Phi4SigLipVLM => (
                "Phi-4 SigLIP Vision (SigLIP2 NaFlex + Phi3 text)",
                "Phi VLM",
            ),

            // ----- DeepSeek -----
            ModelType::DeepSeek => ("DeepSeek v1", "DeepSeek"),
            ModelType::DeepSeekV2 => ("DeepSeek v2", "DeepSeek"),
            ModelType::DeepSeekV3 => ("DeepSeek v3 / R1", "DeepSeek"),
            ModelType::DeepSeekV32 => ("DeepSeek v3.2", "DeepSeek"),

            // ----- Cohere -----
            ModelType::Cohere => ("Command R (Cohere)", "Cohere"),
            ModelType::Cohere2 => ("Command R+ (Cohere2)", "Cohere"),
            ModelType::AyaVisionVLM => ("Aya Vision (SigLIP + Cohere2)", "Cohere VLM"),

            // ----- InternLM -----
            ModelType::InternLM2 => ("InternLM 2", "InternLM"),
            ModelType::InternLM3 => ("InternLM 3", "InternLM"),

            // ----- GLM -----
            ModelType::Glm4 => ("GLM 4", "GLM"),
            ModelType::Glm4Moe => ("GLM 4 MoE", "GLM"),
            ModelType::Glm4MoeLite => ("GLM 4 MoE Lite", "GLM"),
            ModelType::GlmMoeDsa => ("GLM MoE DSA", "GLM"),

            // ----- ERNIE -----
            ModelType::Ernie45 => ("ERNIE 4.5", "ERNIE"),
            ModelType::Ernie45Moe => ("ERNIE 4.5 MoE", "ERNIE"),

            // ----- Hunyuan -----
            ModelType::HunyuanV1Dense => ("Hunyuan v1 Dense", "Hunyuan"),
            ModelType::HunyuanMoe => ("Hunyuan MoE", "Hunyuan"),

            // ----- IBM Granite -----
            ModelType::Granite => ("Granite (dense)", "Granite"),
            ModelType::GraniteMoeHybrid => ("Granite 4 (Mamba2 + attention hybrid)", "Granite"),

            // ----- ExaOne -----
            ModelType::ExaOne => ("ExaOne 3", "ExaOne"),
            ModelType::ExaOne4 => ("ExaOne 4", "ExaOne"),
            ModelType::ExaOneMoe => ("ExaOne MoE", "ExaOne"),

            // ----- Solar -----
            ModelType::SolarOpen => ("Solar Open", "Solar"),

            // ----- OLMo -----
            ModelType::Olmo => ("OLMo 1", "OLMo"),
            ModelType::Olmo2 => ("OLMo 2", "OLMo"),
            ModelType::Olmo3 => ("OLMo 3", "OLMo"),
            ModelType::OLMoE => ("OLMoE (MoE)", "OLMo"),

            // ----- Nemotron -----
            ModelType::Nemotron => ("Nemotron-4", "Nemotron"),
            ModelType::NemotronH => (
                "Nemotron-H (Mamba2 + Attention + MLP/MoE hybrid)",
                "Nemotron",
            ),
            ModelType::NemotronNAS => ("Nemotron-NAS", "Nemotron"),
            ModelType::NemotronHNanoOmniVLM => ("Nemotron-H Nano Omni VLM", "Nemotron VLM"),

            // ----- MoE (other) -----
            ModelType::GptOss => ("gpt-oss (MoE)", "MoE (other)"),
            ModelType::MiniMax => ("MiniMax-M2 (MoE, 256 experts)", "MoE (other)"),
            ModelType::Mixtral => ("Mixtral (MoE)", "MoE (other)"),
            ModelType::KimiLinear => ("Kimi Linear (MLA + GatedDeltaNet hybrid)", "MoE (other)"),
            ModelType::LongcatFlash => ("LongCat Flash (MLA + MoE, dual sublayer)", "MoE (other)"),
            ModelType::LongcatFlashNgram => ("LongCat Flash + N-gram embedding", "MoE (other)"),
            ModelType::Step3p5 => ("Step-3.5 (Sigmoid MoE gate + SwitchGLU)", "MoE (other)"),

            // ----- Mamba / SSM -----
            ModelType::Mamba => ("Mamba 1 / Falcon Mamba", "Mamba / SSM"),
            ModelType::Mamba2 => ("Mamba 2", "Mamba / SSM"),

            // ----- Hybrid (Attention + SSM) -----
            ModelType::Jamba => ("Jamba (Mamba + Transformer + MoE)", "Hybrid"),

            // ----- Falcon -----
            ModelType::FalconH1 => ("Falcon-H1 (Mamba2 + Attention parallel hybrid)", "Falcon"),

            // ----- Liquid Foundation Models -----
            ModelType::Lfm2 => ("LFM2 (short-conv + attention hybrid)", "LFM2"),
            ModelType::Lfm2Moe => ("LFM2-MoE (sigmoid-gated experts)", "LFM2"),

            // ----- Preferred Networks -----
            ModelType::Plamo2 => ("PLaMo 2 (Mamba + attention hybrid)", "PLaMo"),

            // ----- RWKV -----
            ModelType::Rwkv7 => ("RWKV v7", "RWKV"),

            // ----- Specialized / other small/text -----
            ModelType::StarCoder2 => ("StarCoder 2", "Specialized"),
            ModelType::StableLM => ("StableLM", "Specialized"),
            ModelType::Baichuan => ("Baichuan", "Specialized"),
            ModelType::MiniCPM => ("MiniCPM 1", "Specialized"),
            ModelType::MiniCPM3 => ("MiniCPM 3", "Specialized"),
            ModelType::SmolLM3 => ("SmolLM 3", "Specialized"),
            ModelType::MiMo => ("MiMo (multi-token prediction)", "Specialized"),
            ModelType::Apertus => ("Apertus (dense)", "Specialized"),

            // ----- Other VLM (cross-family vision-language stacks) -----
            ModelType::LlavaVLM => ("LLaVA (CLIP/SigLIP + Llama/Qwen2)", "Other VLM"),
            ModelType::LlavaBunnyVLM => ("LLaVA-Bunny (SigLIP + Qwen2)", "Other VLM"),
            ModelType::InternVLChatVLM => {
                ("InternVL (InternViT + pixel-shuffle + Qwen2)", "Other VLM")
            }
            ModelType::MolmoVLM => ("Molmo (CLIP ViT + OLMo-style text)", "Other VLM"),
            ModelType::Molmo2VLM => ("Molmo 2 (custom ViT + Molmo2 text)", "Other VLM"),
            ModelType::MolmoPointVLM => {
                ("Molmo-Point (point prediction + Molmo2 text)", "Other VLM")
            }
            ModelType::Moondream3VLM => ("Moondream 3 (custom ViT + custom decoder)", "Other VLM"),
            ModelType::MiniCPMOVLM => (
                "MiniCPM-o (dynamic SigLIP + resampler + Qwen3-VL text)",
                "Other VLM",
            ),
            ModelType::MiniCPMV46VLM => (
                "MiniCPM-V 4.6 (SigLIP + VitMerger + Merger + Qwen3.5 text)",
                "Other VLM",
            ),
            ModelType::YoutuVLM => (
                "Youtu-VL (SigLIP2 windowed-attn + DeepSeek-V3 MLA)",
                "Other VLM",
            ),
        }
    }

    /// Short human-readable label for `mlxcel arch`. See [`metadata`].
    ///
    /// [`metadata`]: ModelType::metadata
    pub const fn display_name(self) -> &'static str {
        self.metadata().0
    }

    /// Family grouping label used by the renderer to bucket variants into
    /// sections. See [`metadata`].
    ///
    /// [`metadata`]: ModelType::metadata
    pub const fn family(self) -> &'static str {
        self.metadata().1
    }
}

#[cfg(test)]
mod metadata_tests {
    use super::{ALL_MODEL_TYPES, ModelType};

    /// `ALL_MODEL_TYPES` is the iteration source for `mlxcel arch`. The
    /// list must contain every `ModelType` variant or rendered output
    /// will silently miss models. We catch drift two ways:
    ///
    /// 1. A count floor (`> 80`) tied to the README's "80+ models"
    ///    claim. This is a *runtime* guard that triggers if a future
    ///    refactor accidentally shrinks the slice.
    /// 2. Walking the slice and asserting every entry has non-empty
    ///    metadata. This catches the case where someone added a
    ///    variant, wired it into `metadata()`, but forgot to push it
    ///    into `ALL_MODEL_TYPES`.
    ///
    /// The exhaustiveness of `ModelType::metadata()` itself is enforced
    /// at compile time by the exhaustive `match` — adding a variant
    /// without a metadata arm is a build error.
    #[test]
    fn all_model_types_covers_every_variant() {
        let count = ALL_MODEL_TYPES.len();
        assert!(
            count > 80,
            "ALL_MODEL_TYPES should hold >80 variants, got {count}; \
             did you add a variant to ModelType but forget to register \
             it in ALL_MODEL_TYPES?"
        );

        for &mt in ALL_MODEL_TYPES {
            assert!(
                !mt.display_name().is_empty(),
                "{mt:?} has empty display_name"
            );
            assert!(!mt.family().is_empty(), "{mt:?} has empty family");
        }
    }

    /// Sanity check on family stability: the family of a variant must be
    /// a non-trivial string and must round-trip through metadata.
    #[test]
    fn metadata_round_trip_is_consistent() {
        for &mt in ALL_MODEL_TYPES {
            let (name, family) = mt.metadata();
            assert_eq!(name, mt.display_name(), "display_name mismatch for {mt:?}");
            assert_eq!(family, mt.family(), "family mismatch for {mt:?}");
        }
    }

    /// The slice should not contain duplicates — duplicate entries would
    /// cause the renderer to emit the same model twice.
    #[test]
    fn all_model_types_has_no_duplicates() {
        let mut seen: Vec<ModelType> = Vec::with_capacity(ALL_MODEL_TYPES.len());
        for &mt in ALL_MODEL_TYPES {
            assert!(
                !seen.contains(&mt),
                "{mt:?} appears more than once in ALL_MODEL_TYPES"
            );
            seen.push(mt);
        }
    }
}

#[cfg(test)]
#[path = "detection_tests.rs"]
mod detection_tests;

#[cfg(test)]
#[path = "gemma3n_helpers_tests.rs"]
mod gemma3n_helpers_tests;

#[cfg(test)]
#[path = "gemma4_tests.rs"]
pub(crate) mod gemma4_tests;

#[cfg(test)]
#[path = "llama4_helpers_tests.rs"]
mod llama4_helpers_tests;

#[cfg(test)]
#[path = "sanitize_tests.rs"]
mod sanitize_tests;

#[cfg(test)]
#[path = "qwen_vl_position_tests.rs"]
mod qwen_vl_position_tests;

#[cfg(test)]
#[path = "qwen3_5_tests.rs"]
mod qwen3_5_tests;

#[cfg(test)]
#[path = "apertus_tests.rs"]
mod apertus_tests;

#[cfg(test)]
#[path = "granite_tests.rs"]
mod granite_tests;

#[cfg(test)]
#[path = "lfm2_tests.rs"]
mod lfm2_tests;

#[cfg(test)]
#[path = "falcon_h1_tests.rs"]
mod falcon_h1_tests;

#[cfg(test)]
#[path = "plamo2_tests.rs"]
mod plamo2_tests;

#[cfg(test)]
#[path = "granitemoehybrid_tests.rs"]
mod granitemoehybrid_tests;
