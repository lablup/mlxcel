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

pub(crate) mod col_late_interaction;
mod detection;
pub(crate) mod embedding_sanitize;
#[cfg(test)]
pub(crate) mod embedding_test_support;
mod gemma3n_helpers;
pub(crate) mod headless_llama;
mod llama4_helpers;
pub(crate) mod model_owned;
pub mod multimodal_placeholders;
pub(crate) mod qwen_mrope_state;
mod recurrent_snapshot;
mod sanitize;

// Shared modules
// `config` holds shared serde defaults and the common `QuantizationArgs`. It
// was never declared here, so the file was orphaned: not compiled, not linted,
// not tested. That is why its `get_mode` could hand out an unvalidated
// quantization mode with nothing catching it (issue #973). Declaring it makes
// the bound it now carries real rather than notional, so the next family that
// reaches for the helper cannot inherit the hole.
pub mod config;
pub(crate) mod conv_decode;
// `dynamic_ntk_rope` is the shared dynamic-NTK / linear rotary schedule for
// the InternLM families (#1324). It exists because both of them got the
// schedule wrong in different halves: `internlm3` scaled positions by 2.0 on
// every dynamic and every absent block, and `internlm2` dropped its
// `rope_scaling` block at deserialization so the base never moved.
pub mod dynamic_ntk_rope;
pub mod gated_delta;
// `rope_overrides` carries llama-server b10621's `--rope-scaling`,
// `--rope-scale`, `--rope-freq-scale` and `--rope-freq-base` from the server
// CLI down to the RoPE seams in `rope_utils` and `dynamic_ntk_rope` (#1450).
// It is process-wide because every family reads `config.json` inside its own
// `load()`, so there is no argument to thread; the applications counter is what
// keeps that from becoming a silent no-op on a family that is not on the seam.
pub mod rope_overrides;
// `rope_utils` is the shared reader for the `rope_scaling` block plus the
// frequency tables it selects (#1355). It exists because two families needed
// the same decision and only one made it: Apertus computed the `llama3` table
// inline while the shared Llama attention parsed the block and dropped it.
pub mod rope_utils;
pub mod switch_layers;

pub(crate) fn parse_optional_eos_token_ids(value: &Option<serde_json::Value>) -> Vec<i32> {
    match value {
        Some(serde_json::Value::Number(n)) => n
            .as_i64()
            .and_then(|id| i32::try_from(id).ok())
            .map(|id| vec![id])
            .unwrap_or_default(),
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_i64())
            .filter_map(|id| i32::try_from(id).ok())
            .collect(),
        _ => Vec::new(),
    }
}

// Model implementations (mlxcel-core based)
pub mod afmoe;
pub mod apertus;
pub mod baichuan;
pub mod bailing_moe;
pub mod bailing_moe_linear;
pub mod bert;
// Facade: the two config types are re-exported from `bert`, so callers keep
// one public path (`models::bert::BertArgs`).
mod bert_config;
pub mod bert_heads;
pub mod bitnet;
pub mod cohere;
pub mod cohere2;
pub mod cohere2_moe;
pub mod colidefics3;
pub mod colqwen2_5;
pub mod dbrx;
pub mod deepseek;
pub mod deepseek_v2;
pub mod deepseek_v3;
pub mod deepseek_v32;
pub mod deepseek_v4;
pub mod diffusion_gemma;
pub mod dots1;
pub mod ernie4_5;
pub mod ernie4_5_moe;
pub mod ernie4_5_moe_vl;
pub mod exaone;
pub mod exaone4;
pub mod exaone_moe;
pub mod falcon_h1;
pub mod falcon_ocr;
pub mod falcon_ocr_rope;
pub mod florence2;
pub mod gemma;
pub mod gemma2;
pub mod gemma3;
pub mod gemma3_embedding;
pub mod gemma3n;
pub mod gemma4;
pub mod gemma4_mtp_target;
pub mod glm4;
pub mod glm4_moe;
pub mod glm4_moe_lite;
pub mod glm4v;
pub mod glm4v_moe;
pub mod glm_moe_dsa;
pub mod gpt2;
pub mod gpt_bigcode;
pub mod gpt_neox;
pub mod gpt_oss;
pub mod granite;
pub mod granitemoehybrid;
pub mod helium;
pub mod hunyuan_moe;
pub mod hunyuan_v1_dense;
pub mod hunyuan_vl;
pub mod inkling;
pub mod inkling_mtp_target;
pub mod internlm2;
pub mod internlm3;
pub mod jamba;
pub mod jina_vlm;
pub mod kimi_linear;
pub mod klear;
pub mod lfm2;
pub mod lfm2_embedding;
pub mod llada2_moe;
pub mod llama3;
pub mod llama4;
pub mod llama_bidirec;
pub mod llama_nemotron_vl_embedding;
pub(crate) mod llama_nemotron_vl_tiling;
pub mod longcat_flash_ngram;
pub mod mamba;
pub mod mamba2;
pub mod mellum;
pub mod mimo;
pub mod minicpm;
pub mod minicpm3;
pub mod minimax;
pub mod minimax_m3;
pub mod ministral3;
pub mod ministral3_embedding;
pub mod mistral4;
pub mod mixtral;
pub mod mllama;
pub mod modernbert;
pub mod modernbert_heads;
pub mod molmo;
pub mod molmo2;
pub mod molmo_point;
pub mod moondream2;
pub mod moondream3;
pub mod muse_glimmer;
pub(crate) mod muse_glimmer_cache;
pub mod muse_glimmer_config;
pub(crate) mod muse_glimmer_layers;
pub mod nemotron;
pub mod nemotron_h;
pub mod nemotron_nas;
pub mod olmo;
pub mod olmo2;
pub mod olmo3;
pub mod olmoe;
pub mod openelm;
pub mod paddleocr_vl;
pub mod phi;
pub mod phi3;
pub mod phi3small;
pub mod phi4mm;
pub mod phimoe;
pub mod phixtral;
pub mod plamo2;
pub mod qwen2;
pub mod qwen2_moe;
pub mod qwen2_vl;
pub mod qwen3;
pub mod qwen3_5;
pub mod qwen3_5_mtp_target;
pub mod qwen3_embedding;
pub mod qwen3_moe;
pub mod qwen3_next;
pub mod qwen3_vl;
pub mod qwen3_vl_embedding;
pub mod qwen3_vl_moe;
pub mod recurrent_gemma;
pub mod rwkv7;
pub mod seed_oss;
pub mod siglip_text;
pub mod smollm3;
pub mod solar_open;
/// Runtime block-vs-chain exactness gate shared by the MTP speculative paths.
pub mod speculative_exactness;
pub mod stablelm;
pub mod starcoder2;
pub mod step3p5;
pub mod telechat3;
#[cfg(test)]
pub(crate) mod vl_embedding_test_images;
pub mod whisper;
pub mod youtu_vl_lm;

// Text-to-speech (StyleTTS2 + iSTFTNet) and its grapheme-to-phoneme front-end.
pub mod g2p;
pub mod kokoro;

// Re-export model types
pub use afmoe::AfmoeModel;
pub use apertus::ApertusModel;
pub use baichuan::BaichuanModel;
pub use bailing_moe::BailingMoeModel;
pub use bailing_moe_linear::BailingMoeLinearModel;
pub use bitnet::BitNetModel;
pub use cohere::CohereModel;
pub use cohere2::Cohere2Model;
pub use cohere2_moe::Cohere2MoeModel;
pub use dbrx::DbrxModel;
pub use deepseek::DeepSeekModel;
pub use deepseek_v2::DeepSeekV2Model;
pub use deepseek_v3::DeepSeekV3Model;
pub use deepseek_v4::DeepSeekV4Model;
pub use deepseek_v32::DeepSeekV32Model;
pub use detection::get_model_type;
pub(crate) use detection::is_sequence_classification_architecture;
pub use diffusion_gemma::DiffusionGemmaModel;
pub use dots1::Dots1Model;
pub use ernie4_5::Ernie45Model;
pub use ernie4_5_moe::Ernie45MoeModel;
pub use exaone::ExaOneModel;
pub use exaone_moe::ExaoneMoeModel;
pub use exaone4::{ExaOne4Model, ExaOne4Wrapper};
pub use falcon_h1::FalconH1Model;
pub use falcon_ocr::{FalconOcrConfig, FalconOcrTextModel};
pub use falcon_ocr_rope::FalconOcrTokenIds;
pub use florence2::{
    FLORENCE2_LOC_TOKEN_BASE, FLORENCE2_VISION_PREFIX, Florence2BoundingBox, Florence2Config,
    Florence2DaViT, Florence2ImageSize, Florence2Model, Florence2Output, Florence2Polygon,
    Florence2PostProcessingType, Florence2Processor, Florence2QuadBox, Florence2Quantization,
    Florence2RunOutput, Florence2SeqCache, Florence2Task, Florence2TaskResult, Florence2TextConfig,
    Florence2TextModel, Florence2VisionConfig, Florence2VlmModel, florence2_loc_token_id,
};
pub use gemma::GemmaModel;
pub use gemma2::Gemma2Model;
pub use gemma3::{Gemma3Model, Gemma3Wrapper};
pub use gemma3n::Gemma3nModel;
pub use gemma4::{Gemma4Model, Gemma4SpeculativeSinks, Gemma4Wrapper};
pub use glm_moe_dsa::GlmMoeDsaModel;
pub use glm4::Glm4Model;
pub use glm4_moe::Glm4MoeModel;
pub use glm4_moe_lite::Glm4MoeLiteModel;
pub use glm4v::Glm4vTextModel;
pub use glm4v_moe::Glm4vMoeTextModel;
pub use gpt_bigcode::GptBigCodeModel;
pub use gpt_neox::GptNeoxModel;
pub use gpt_oss::{GptOssModel, GptOssWrapper};
pub use gpt2::Gpt2Model;
pub use granite::GraniteModel;
pub use granitemoehybrid::GraniteMoeHybridModel;
pub use helium::HeliumModel;
pub use hunyuan_moe::HunyuanMoeModel;
pub use hunyuan_v1_dense::HunyuanV1DenseModel;
pub use inkling::InklingModel;
pub use internlm2::InternLM2Model;
pub use internlm3::InternLM3Model;
pub use jamba::JambaModel;
pub use jina_vlm::{JinaVlmTextConfig, JinaVlmTextModel};
pub use kimi_linear::KimiLinearModel;
pub use klear::KlearModel;
pub use lfm2::Lfm2Model;
pub use llada2_moe::Llada2MoeModel;
pub use llama3::Llama3Model;
pub use llama4::{Llama4CxxModel, Llama4Wrapper};
pub use longcat_flash_ngram::LongcatFlashNgramModel;
pub use mamba::MambaModel;
pub use mamba2::Mamba2Model;
pub use mellum::{MellumModel, MellumWrapper};
pub use mimo::MiMoModel;
pub use minicpm::MiniCPMModel;
pub use minicpm3::MiniCPM3Model;
pub use minimax::MiniMaxModel;
pub use minimax_m3::MiniMaxM3Model;
pub use ministral3::{Ministral3Model, Ministral3Wrapper};
pub use mistral4::Mistral4Model;
pub use mixtral::MixtralModel;
pub use molmo::MolmoModel;
pub use molmo2::Molmo2Model;
pub use moondream2::Moondream2Model;
pub use moondream3::Moondream3Model;
pub use multimodal_placeholders::MultimodalPlaceholderTokens;
pub use muse_glimmer::{
    DEFAULT_IMAGE_END_TOKEN_ID, DEFAULT_IMAGE_PLACEHOLDER_TOKEN_ID, DEFAULT_IMAGE_START_TOKEN_ID,
    DEFAULT_IMAGE_TOKEN_ID, MuseGlimmerConfig, MuseGlimmerTextConfig, MuseGlimmerTextModel,
    MuseGlimmerTextWrapper, MuseGlimmerVisionConfig,
};
pub use nemotron::NemotronModel;
pub use nemotron_h::NemotronHModel;
pub use nemotron_nas::NemotronNASModel;
pub use olmo::OlmoModel;
pub use olmo2::OLMo2Model;
pub use olmo3::OLMo3Model;
pub use olmoe::OlmoeModel;
pub use openelm::OpenElmModel;
pub use paddleocr_vl::{PaddleOcrTextConfig, PaddleOcrTextModel};
pub use phi::PhiModel;
pub use phi3::Phi3Model;
pub use phi3small::Phi3SmallModel;
pub use phi4mm::Phi4MMModel;
pub use phimoe::PhiMoeModel;
pub use phixtral::PhixtralModel;
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
    Gemma4WeightBacking, config_has_quantization_metadata, load_gemma4_text_weights_with_backing,
    load_gemma4_unified_weights_with_backing, load_gemma4_vlm_weights_with_backing,
    sanitize_gemma4_nvfp4_weights, should_convert_bf16_to_f16, strip_gemma4_kv_shared_weights,
};
// The only consumer outside `sanitize` is the diagnostics-gated Molmo2 vision
// reference loader, so an unconditional re-export is dead in a default build
// and `-D warnings` rejects it.
#[cfg(any(test, feature = "xla-diagnostics", feature = "xla-diagnostics-cpu"))]
pub(crate) use sanitize::load_weights_from_dir_with_filter;
pub use sanitize::{
    convert_bf16_weights, convert_bf16_weights_with_keep, gemma3n_language_mlp_bf16_key,
    load_and_sanitize_weights, load_text_weights, sanitize_config_json, sanitize_tied_embeddings,
    warn_bf16_precision,
};
pub use seed_oss::SeedOssModel;
pub use siglip_text::{SigLipTextArgs, SigLipTextModel};
pub use smollm3::SmolLM3Model;
pub use solar_open::SolarOpenModel;
pub use stablelm::StableLMModel;
pub use starcoder2::StarCoder2Model;
pub use step3p5::Step3p5Model;
pub use telechat3::TeleChat3Model;
pub use whisper::WhisperModel;

pub use kokoro::KokoroModel;

/// Supported model types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    // Standard Transformer models
    Llama,             // Llama 1/2/3, Mistral
    Llama4,            // Llama 4 (MoE)
    Llama4VLM,         // Llama 4 VLM (vision-language)
    MllamaVLM,         // Llama 3.2 Vision (mllama): tiled ViT + gated cross-attention
    Qwen2,             // Qwen 2/2.5
    Qwen3,             // Qwen 3
    Qwen3Moe,          // Qwen 3 MoE
    Qwen3Next,         // Qwen 3 with GatedDeltaNet
    Qwen35,            // Qwen 3.5 Hybrid (Transformer + GatedDeltaNet)
    Qwen35VLM,         // Qwen 3.5 VLM (Qwen3-VL vision + Qwen3.5 hybrid text)
    Qwen35Moe,         // Qwen 3.5 MoE Hybrid
    Qwen35MoeVLM,      // Qwen 3.5 MoE VLM
    Gemma,             // Gemma 1
    Gemma2,            // Gemma 2
    Gemma3,            // Gemma 3 (text-only)
    Gemma4,            // Gemma 4 text-only route
    DiffusionGemma,    // DiffusionGemma (block-diffusion on the Gemma 4 MoE backbone)
    Llada2Moe,         // LLaDA-2 MoE (masked-diffusion LM, DeepSeek-V3-style MoE FFN)
    Gemma3VLM,         // Gemma 3 VLM (vision-language)
    Gemma4VLM,         // Gemma 4 VLM (vision-language)
    Gemma4Unified,     // Gemma 4 Unified (encoder-free text + vision + audio)
    LlavaVLM,          // LLaVA (CLIP/SigLIP + Llama/Qwen2)
    GraniteVisionVLM,  // Granite Vision (SigLIP multi-tap + Granite text, AnyRes)
    Granite4VisionVLM, // Granite 4 Vision (SigLIP + window-QFormer + Granite-4 hybrid)
    DeepSeekOcrVLM,    // DeepSeek-OCR (SAM + CLIP + DeepSeek MoE decoder)
    DeepSeekOcr2VLM,   // DeepSeek-OCR 2 (SAM + Qwen2 resampler + DeepSeek MoE decoder)
    UnlimitedOcrVLM,   // Unlimited-OCR (DeepSeek-OCR stack + ring sliding decode cache)
    DeepSeekVL2,       // DeepSeek-VL2 (SigLIP + downsample MLP + DeepSeek-V2 MoE decoder)
    LlavaBunnyVLM,     // LLaVA-Bunny (SigLIP + Qwen2)
    FastVLM,           // FastVLM (FastViTHD vision + Qwen2 text, mlp2x_gelu)
    Ernie45MoeVLM,     // ERNIE-4.5 MoE VL (DFNRope ViT + modality-split MoE + 3D MRoPE)
    HunyuanVLM,        // Hunyuan-VL (ViT + perceive merger + XD-RoPE decoder)
    AyaVisionVLM,      // Aya Vision (SigLIP + Cohere2)
    PaliGemmaVLM,      // PaliGemma (SigLIP + Gemma)
    PixtralVLM,        // Pixtral (ViT w/ 2D RoPE + Mistral)
    Mistral3VLM,       // Mistral 3 VLM (Pixtral ViT + PatchMerger + Mistral)
    Qwen2VL,           // Qwen2-VL (custom ViT + Qwen2 w/ MRoPE)
    Qwen25VL,          // Qwen2.5-VL (windowed ViT + Qwen2 w/ MRoPE)
    Qwen3VL,           // Qwen3-VL (ViT + interleaved MRoPE + DeepStack)
    Qwen3VLMoe,        // Qwen3-VL-MoE (Qwen3-VL + MoE text backbone)
    Qwen3OmniMoe,      // Qwen3-Omni MoE thinker (Qwen3-VL-MoE + audio tower)
    PaddleOcrVL,       // PaddleOCR-VL (NaViT vision + ERNIE-4.5 w/ MRoPE)
    DotsOcrVL,         // dots.ocr (dots_vit ViT + Qwen2 text decoder)
    FalconOcrVL,       // Falcon-OCR (early-fusion patch projector, no vision tower)
    JinaVLM,           // Jina VLM (SigLIP-class ViT + Molmo-style connector + Qwen2 text)
    Glm4v,             // GLM-4V (GLM-4V ViT + GLM-4 text w/ sectioned MRoPE)
    Glm4vMoe,          // GLM-4V MoE (GLM-4V ViT + GLM-4 MoE text w/ MRoPE)
    GlmOcr,            // GLM-OCR (GLM-OCR ViT + GLM-4 text w/ full-width MRoPE)
    YoutuVLM,          // Youtu-VL (SigLIP2 windowed-attn + DeepSeek-V3-style MLA)
    InternVLChatVLM,   // InternVL (internvl_chat): InternViT + pixel-shuffle mlp1 + Qwen2 text
    LocateAnythingVLM, // LocateAnything: MoonViT + MLP connector + Qwen2 text (grounding)
    SmolVLM,  // SmolVLM/SmolVLM2 (smolvlm): SigLIP + pixel-shuffle connector + SmolLM2 text
    Idefics2, // Idefics2 (idefics2): SigLIP + perceiver-resampler connector + Mistral text
    MiniCPMOVLM, // MiniCPM-o (dynamic SigLIP + resampler + Qwen3-VL text)
    MiniCPMV46VLM, // MiniCPM-V 4.6 (SigLIP + VitMerger + Merger + Qwen3.5 text)
    Moondream3VLM, // Moondream3 (custom ViT + custom text decoder, query/caption image path)
    Moondream2VLM, // Moondream2 (SigLIP-style ViT + Phi text decoder + crop tiling)
    /// Florence-2 (`florence2`): DaViT vision tower + BART encoder-decoder
    /// text stack. Encoder-decoder (seq2seq), so it is served through its own
    /// task pipeline (CLI early exit), not the autoregressive decode loop.
    Florence2VLM,
    Gemma3n,    // Gemma 3n (text-only)
    Gemma3nVLM, // Gemma 3n VLM (MobileNetV5 + Gemma3n)
    Phi,        // Phi 1/2
    /// Phixtral (`phi-msft` with `num_local_experts`): a Mixtral-style
    /// sparse MoE on the Phi-2 parallel-residual backbone. Shares the
    /// `phi-msft` model_type with dense Phi and is told apart by
    /// `num_local_experts`; see `detection::detect_phi_model_type`.
    Phixtral,
    Phi3,          // Phi 3
    Phi4MMVLM,     // Phi-4 Multimodal (SigLIP2 NaFlex + Conformer audio + Phi4 text)
    Phi4SigLipVLM, // Phi-4 reasoning vision (SigLIP2 NaFlex + Phi3-style text)
    Phi3VLM,       // Phi 3.5 Vision (CLIP + Phi3)
    MolmoVLM,      // Molmo v1 (CLIP ViT + attention pooling + OLMo-style text)
    Molmo2VLM,     // Molmo2 (custom ViT + attention pooling + Molmo2 text)
    MolmoPointVLM, // Molmo-Point (custom ViT + point prediction + Molmo2 text)
    Phi3Small,     // Phi 3 Small
    PhiMoe,        // Phi MoE

    // MoE models
    GptOss,
    MiniMax,
    MiniMaxM3,
    MiniMaxM3VL,    // MiniMax-M3-VL (CLIP ViT + M3 hybrid dense/MoE text)
    MuseGlimmerVLM, // Muse Glimmer (Meta VLM, Muse text decoder + vision tower)
    Mixtral,
    Qwen2Moe,
    OLMoE,
    Dbrx, // Databricks DBRX (fused clipped QKV, norm-attn-norm, w1/v1/w2 experts)

    // DeepSeek family
    DeepSeek,
    DeepSeekV2,
    DeepSeekV3,
    DeepSeekV32,
    /// DeepSeek-V4 (HyperConnections, pooled-KV compression, HiSA, hash-routed MoE).
    DeepSeekV4,
    /// rednote dots.llm1 (DeepSeek-V3-style MoE without MLA).
    Dots1,

    // Cohere family
    Cohere,
    Cohere2,
    /// Command MoE (Cohere2 MoE): the Cohere2 backbone with a sparse MoE FFN.
    Cohere2Moe,

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
    /// Ant Group Ling / Bailing MoE (`bailing_moe`): DeepSeek-shaped sparse
    /// decoder with a fused GQA `query_key_value` and a single wide shared
    /// expert.
    BailingMoe,
    /// Ant Group Ling / Ring linear-attention MoE (`bailing_moe_linear`): the
    /// same MoE block interleaved with gated-linear-attention layers whose decay
    /// is a fixed ALiBi schedule.
    BailingMoeLinear,

    /// Arcee AFMoE (`afmoe`, the Trinity family): hybrid sliding/full
    /// attention with NoPE global layers, a sigmoid attention gate,
    /// sandwich norms, muP embedding scale and a sparse MoE FFN.
    Afmoe,
    /// Kuaishou Klear (`Klear`): a Qwen3-shaped sparse MoE whose shared
    /// expert is blended with the routed mixture through a learned 2-way
    /// softmax rather than added.
    Klear,

    // Apertus (Swiss AI)
    Apertus,

    // ByteDance Seed-OSS (dense)
    SeedOss,

    // IBM Granite
    Granite,

    // BitNet (1.58-bit ternary)
    BitNet,

    // Korean models
    ExaOne,
    ExaOne4,
    ExaOneMoe,
    SolarOpen,

    // OLMo family
    Olmo,
    Olmo2,
    Olmo3,
    OpenElm, // Apple OpenELM (layer-wise scaling: per-layer head counts and FFN widths)

    // GPT-2 lineage
    Gpt2,       // GPT-2 (learned absolute position embeddings, Conv1D weight layout)
    GptBigCode, // GPT-BigCode (StarCoder / SantaCoder: GPT-2 block with multi-query attention)
    GptNeoX,    // GPT-NeoX (EleutherAI Pythia: interleaved per-head QKV, partial RoPE)

    // Code models
    StarCoder2,
    Mellum, // Mellum 2 (JetBrains hybrid-attention MoE code model)

    // Other Transformer models
    Helium,    // Kyutai Helium (Llama-shaped dense decoder with traditional RoPE)
    TeleChat3, // TeleAI TeleChat3 (Llama-shaped dense decoder with YaRN RoPE scaling)
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
    Lfm2VL, // LFM2-VL (lfm2_vl): packed-patch SigLIP2 ViT + LFM2 hybrid text

    // Thinking Machines Inkling text backbone
    Inkling,

    // Preferred Networks PLaMo 2 (Mamba + attention interleaved hybrid)
    Plamo2,

    // IBM Granite 4.x (Mamba2 + attention interleaved hybrid)
    GraniteMoeHybrid,

    // Kimi models
    KimiLinear,
    KimiVL,  // Kimi-VL (MoonViT vision encoder + DeepSeek-V3-style MoE text)
    KimiK25, // Kimi-VL 2.5 (MoonViT + DeepSeek-V3-style MoE, image path)

    // Longcat models
    LongcatFlash,
    LongcatFlashNgram,

    // Step models
    Step3p5,
    Step3p7, // Step-3.7 (perception_encoder ViT + Step-3.5 MoE text; VLM)

    // RNN models
    Rwkv7,
    RecurrentGemma,

    // Speech-to-text (encoder-decoder ASR)
    Whisper,

    // Text-to-speech (StyleTTS2 acoustic model + built-in iSTFTNet vocoder)
    Kokoro,

    // Embedding models served through /v1/embeddings (epic #1348). Detected by
    // encoder-only `model_type`, embedding `architectures[0]`, a
    // `modules.json` Pooling entry or a `1_Pooling/config.json`.
    Bert,                     // BERT / MiniLM encoders (bert)
    XlmRoberta,               // XLM-RoBERTa encoders (xlm-roberta)
    ModernBert,               // ModernBERT encoders (modernbert)
    SiglipText,               // SigLIP text tower (siglip)
    Gemma3Embedding,          // EmbeddingGemma (gemma3_text, bidirectional)
    Qwen3Embedding,           // Qwen3-Embedding (qwen3 + 1_Pooling lasttoken)
    Qwen3VLEmbedding,         // Qwen3-VL-Embedding (qwen3_vl)
    Lfm2Embedding,            // LFM2 bidirectional embedder (lfm2)
    Ministral3Embedding,      // Ministral3 bidirectional embedder (ministral3)
    LlamaBidirec,             // Llama bidirectional embedder (llama / llama_bidirec)
    LlamaNemotronVLEmbedding, // Llama-Nemotron-VL embedder (llama_nemotron_vl)
    ColIdefics3,              // ColIdefics3 late-interaction retriever (idefics3)
    ColQwen25,                // ColQwen2.5 late-interaction retriever (qwen2_5_vl)

    // Rerankers served through /v1/rerank (#1356). Detected by an
    // `architectures[0]` ending in `ForSequenceClassification` on one of the
    // encoder families; the generative rerankers are indistinguishable from
    // chat checkpoints and reach the reranker worker through
    // `--reranker-model` instead.
    SequenceClassifier, // BERT / XLM-RoBERTa / ModernBERT cross-encoder
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
    ModelType::MllamaVLM,
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
    ModelType::Llada2Moe,
    ModelType::Gemma3VLM,
    ModelType::Gemma4VLM,
    ModelType::Gemma4Unified,
    ModelType::LlavaVLM,
    ModelType::GraniteVisionVLM,
    ModelType::Granite4VisionVLM,
    ModelType::DeepSeekOcrVLM,
    ModelType::DeepSeekOcr2VLM,
    ModelType::UnlimitedOcrVLM,
    ModelType::DeepSeekVL2,
    ModelType::LlavaBunnyVLM,
    ModelType::FastVLM,
    ModelType::Ernie45MoeVLM,
    ModelType::HunyuanVLM,
    ModelType::AyaVisionVLM,
    ModelType::PaliGemmaVLM,
    ModelType::PixtralVLM,
    ModelType::Mistral3VLM,
    ModelType::Qwen2VL,
    ModelType::Qwen25VL,
    ModelType::Qwen3VL,
    ModelType::Qwen3VLMoe,
    ModelType::Qwen3OmniMoe,
    ModelType::PaddleOcrVL,
    ModelType::DotsOcrVL,
    ModelType::FalconOcrVL,
    ModelType::JinaVLM,
    ModelType::Glm4v,
    ModelType::Glm4vMoe,
    ModelType::GlmOcr,
    ModelType::YoutuVLM,
    ModelType::InternVLChatVLM,
    ModelType::LocateAnythingVLM,
    ModelType::SmolVLM,
    ModelType::Idefics2,
    ModelType::MiniCPMOVLM,
    ModelType::MiniCPMV46VLM,
    ModelType::Moondream3VLM,
    ModelType::Moondream2VLM,
    ModelType::Florence2VLM,
    ModelType::Gemma3n,
    ModelType::Gemma3nVLM,
    ModelType::Phi,
    ModelType::Phixtral,
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
    ModelType::MiniMaxM3,
    ModelType::MiniMaxM3VL,
    ModelType::MuseGlimmerVLM,
    ModelType::Mixtral,
    ModelType::Qwen2Moe,
    ModelType::OLMoE,
    ModelType::Dbrx,
    // DeepSeek family
    ModelType::DeepSeek,
    ModelType::DeepSeekV2,
    ModelType::DeepSeekV3,
    ModelType::DeepSeekV32,
    ModelType::DeepSeekV4,
    ModelType::Dots1,
    // Cohere family
    ModelType::Cohere,
    ModelType::Cohere2,
    ModelType::Cohere2Moe,
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
    ModelType::BailingMoe,
    ModelType::BailingMoeLinear,
    ModelType::Afmoe,
    ModelType::Klear,
    // Apertus (Swiss AI)
    ModelType::Apertus,
    // ByteDance Seed-OSS
    ModelType::SeedOss,
    // IBM Granite
    ModelType::Granite,
    // BitNet (1.58-bit ternary)
    ModelType::BitNet,
    // Korean models
    ModelType::ExaOne,
    ModelType::ExaOne4,
    ModelType::ExaOneMoe,
    ModelType::SolarOpen,
    // OLMo family
    ModelType::Olmo,
    ModelType::Olmo2,
    ModelType::Olmo3,
    ModelType::OpenElm,
    // GPT-2 lineage
    ModelType::Gpt2,
    ModelType::GptBigCode,
    ModelType::GptNeoX,
    // Code models
    ModelType::StarCoder2,
    ModelType::Mellum,
    // Other Transformer models
    ModelType::Helium,
    ModelType::TeleChat3,
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
    ModelType::Lfm2VL,
    ModelType::Inkling,
    // Preferred Networks PLaMo 2
    ModelType::Plamo2,
    // IBM Granite 4.x hybrid
    ModelType::GraniteMoeHybrid,
    // Kimi models
    ModelType::KimiLinear,
    ModelType::KimiVL,
    ModelType::KimiK25,
    // Longcat models
    ModelType::LongcatFlash,
    ModelType::LongcatFlashNgram,
    // Step models
    ModelType::Step3p5,
    ModelType::Step3p7,
    // RNN models
    ModelType::Rwkv7,
    ModelType::RecurrentGemma,
    // Speech-to-text
    ModelType::Whisper,
    // Text-to-speech
    ModelType::Kokoro,
    // Embedding models
    ModelType::Bert,
    ModelType::XlmRoberta,
    ModelType::ModernBert,
    ModelType::SiglipText,
    ModelType::Gemma3Embedding,
    ModelType::Qwen3Embedding,
    ModelType::Qwen3VLEmbedding,
    ModelType::Lfm2Embedding,
    ModelType::Ministral3Embedding,
    ModelType::LlamaBidirec,
    ModelType::LlamaNemotronVLEmbedding,
    ModelType::ColIdefics3,
    ModelType::ColQwen25,
    // Rerankers
    ModelType::SequenceClassifier,
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
            ModelType::MllamaVLM => (
                "Llama 3.2 Vision (tiled ViT + gated cross-attention)",
                "Llama VLM",
            ),

            // ----- Qwen (text/hybrid/MoE) -----
            ModelType::Qwen2 => ("Qwen 2 / 2.5", "Qwen"),
            ModelType::Qwen3 => ("Qwen 3", "Qwen"),
            ModelType::Qwen3Moe => ("Qwen 3 MoE", "Qwen"),
            ModelType::Qwen3Next => ("Qwen 3 Next (Attention + GatedDeltaNet + MoE)", "Qwen"),
            ModelType::Qwen35 => ("Qwen 3.5 / 3.8 (Attention + GatedDeltaNet hybrid)", "Qwen"),
            ModelType::Qwen35Moe => ("Qwen 3.5 / 3.6 MoE (hybrid)", "Qwen"),
            ModelType::Qwen2Moe => ("Qwen 2 MoE", "Qwen"),

            // ----- Qwen VLM -----
            ModelType::Qwen2VL => ("Qwen2-VL", "Qwen VLM"),
            ModelType::Qwen25VL => ("Qwen2.5-VL", "Qwen VLM"),
            ModelType::Qwen3VL => ("Qwen3-VL", "Qwen VLM"),
            ModelType::Qwen3VLMoe => ("Qwen3-VL MoE", "Qwen VLM"),
            ModelType::Qwen3OmniMoe => ("Qwen3-Omni MoE (thinker)", "Qwen VLM"),
            ModelType::PaddleOcrVL => ("PaddleOCR-VL", "PaddleOCR VLM"),
            ModelType::DotsOcrVL => ("dots.ocr (dots_vit + Qwen2)", "Other VLM"),
            ModelType::FalconOcrVL => ("Falcon-OCR (early fusion)", "Other VLM"),
            ModelType::JinaVLM => (
                "Jina VLM (SigLIP-so400m + 2x2 attention pooling + Qwen2 text)",
                "Other VLM",
            ),
            ModelType::Glm4v => ("GLM-4V", "GLM VLM"),
            ModelType::Glm4vMoe => ("GLM-4V MoE", "GLM VLM"),
            ModelType::GlmOcr => ("GLM-OCR", "GLM VLM"),
            ModelType::Qwen35VLM => ("Qwen 3.5 / 3.8 VLM", "Qwen VLM"),
            ModelType::Qwen35MoeVLM => ("Qwen 3.5 / 3.6 MoE VLM", "Qwen VLM"),

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
            ModelType::Llada2Moe => (
                "LLaDA-2 MoE (masked-diffusion LM, DeepSeek-V3-style MoE)",
                "Diffusion",
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
            ModelType::Phixtral => ("Phixtral (Phi-2 backbone + sparse MoE)", "Phi"),
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
            ModelType::DeepSeekV4 => ("DeepSeek v4", "DeepSeek"),

            // ----- Cohere -----
            ModelType::Cohere => ("Command R (Cohere)", "Cohere"),
            ModelType::Cohere2 => ("Command R+ (Cohere2)", "Cohere"),
            ModelType::Cohere2Moe => ("Command MoE (Cohere2 MoE)", "Cohere"),
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
            ModelType::BailingMoe => ("Ling / Bailing MoE (shared + routed experts)", "Bailing"),
            ModelType::Afmoe => ("Arcee AFMoE / Trinity (sliding + full hybrid MoE)", "Arcee"),
            ModelType::Klear => ("Klear MoE (coefficient-blended shared expert)", "Klear"),
            ModelType::BailingMoeLinear => (
                "Ling / Ring linear-attention MoE (GLA + full attention hybrid)",
                "Bailing",
            ),

            // ----- IBM Granite -----
            ModelType::Granite => ("Granite (dense)", "Granite"),
            ModelType::BitNet => ("BitNet b1.58 (ternary)", "BitNet"),
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
            ModelType::OpenElm => ("Apple OpenELM (layer-wise scaling)", "Specialized"),
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
            ModelType::MiniMaxM3 => (
                "MiniMax-M3 (hybrid dense/MoE, block-sparse attention)",
                "MoE (other)",
            ),
            ModelType::MiniMaxM3VL => (
                "MiniMax-M3-VL (CLIP ViT + M3 hybrid dense/MoE)",
                "MiniMax VLM",
            ),
            ModelType::MuseGlimmerVLM => ("Muse Glimmer 30B VLM", "Muse VLM"),
            ModelType::Mixtral => ("Mixtral (MoE)", "MoE (other)"),
            ModelType::Dbrx => ("Databricks DBRX (MoE)", "MoE (other)"),
            ModelType::KimiLinear => ("Kimi Linear (MLA + GatedDeltaNet hybrid)", "MoE (other)"),
            ModelType::KimiVL => ("Kimi-VL (MoonViT + DeepSeek-V3 MoE)", "Kimi VLM"),
            ModelType::KimiK25 => ("Kimi-VL 2.5 (MoonViT + DeepSeek-V3 MoE)", "Kimi VLM"),
            ModelType::LongcatFlash => ("LongCat Flash (MLA + MoE, dual sublayer)", "MoE (other)"),
            ModelType::LongcatFlashNgram => ("LongCat Flash + N-gram embedding", "MoE (other)"),
            ModelType::Step3p5 => ("Step-3.5 (Sigmoid MoE gate + SwitchGLU)", "MoE (other)"),
            ModelType::Step3p7 => ("Step-3.7 (ViT + Step-3.5 MoE)", "Step VLM"),
            ModelType::Dots1 => ("dots.llm1 (MoE)", "MoE (other)"),

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
            ModelType::Lfm2VL => ("LFM2-VL (packed-patch ViT + LFM2 hybrid text)", "LFM2"),
            ModelType::Inkling => (
                "Inkling (relative-bias attention + shared-expert MoE)",
                "Inkling",
            ),

            // ----- Preferred Networks -----
            ModelType::Plamo2 => ("PLaMo 2 (Mamba + attention hybrid)", "PLaMo"),

            // ----- RWKV -----
            ModelType::Rwkv7 => ("RWKV v7", "RWKV"),

            // ----- Speech-to-text (ASR) -----
            ModelType::Whisper => ("Whisper (encoder-decoder ASR)", "Speech-to-text"),

            // ----- Text-to-speech (TTS) -----
            ModelType::Kokoro => ("Kokoro (StyleTTS2 + iSTFTNet)", "Text-to-speech"),

            // ----- Embedding models (/v1/embeddings) -----
            ModelType::Bert => ("BERT / MiniLM encoder", "Embedding"),
            ModelType::XlmRoberta => ("XLM-RoBERTa encoder", "Embedding"),
            ModelType::ModernBert => ("ModernBERT encoder", "Embedding"),
            ModelType::SiglipText => ("SigLIP text tower", "Embedding"),
            ModelType::Gemma3Embedding => ("EmbeddingGemma (bidirectional Gemma 3)", "Embedding"),
            ModelType::Qwen3Embedding => ("Qwen3-Embedding (last-token)", "Embedding"),
            ModelType::Qwen3VLEmbedding => ("Qwen3-VL-Embedding (multimodal)", "Embedding"),
            ModelType::Lfm2Embedding => ("LFM2.5-Embedding (bidirectional LFM2)", "Embedding"),
            ModelType::Ministral3Embedding => {
                ("Nemotron-3-Embed (bidirectional Ministral 3)", "Embedding")
            }
            ModelType::LlamaBidirec => ("Bidirectional Llama / LLM2Vec embedder", "Embedding"),
            ModelType::LlamaNemotronVLEmbedding => {
                ("Llama-Nemotron-VL-Embed (multimodal)", "Embedding")
            }
            ModelType::ColIdefics3 => ("ColIdefics3 (late-interaction, multimodal)", "Embedding"),
            ModelType::ColQwen25 => ("ColQwen2.5 (late-interaction, multimodal)", "Embedding"),

            // ----- Rerankers (/v1/rerank) -----
            ModelType::SequenceClassifier => (
                "Cross-encoder sequence classifier (BERT / XLM-RoBERTa / ModernBERT)",
                "Reranker",
            ),

            // ----- Specialized / other small/text -----
            ModelType::Gpt2 => (
                "GPT-2 (learned absolute positions, Conv1D weights)",
                "Specialized",
            ),
            ModelType::GptBigCode => (
                "GPT-BigCode (StarCoder / SantaCoder, multi-query attention)",
                "Specialized",
            ),
            ModelType::GptNeoX => (
                "GPT-NeoX (EleutherAI Pythia, partial RoPE, parallel residual)",
                "Specialized",
            ),
            ModelType::Helium => (
                "Kyutai Helium (dense Llama shape, traditional RoPE)",
                "Specialized",
            ),
            ModelType::TeleChat3 => (
                "TeleAI TeleChat3 (dense Llama shape, YaRN RoPE)",
                "Specialized",
            ),
            ModelType::StarCoder2 => ("StarCoder 2", "Specialized"),
            ModelType::Mellum => ("Mellum 2 (JetBrains code)", "Specialized"),
            ModelType::StableLM => ("StableLM", "Specialized"),
            ModelType::Baichuan => ("Baichuan", "Specialized"),
            ModelType::MiniCPM => ("MiniCPM 1", "Specialized"),
            ModelType::MiniCPM3 => ("MiniCPM 3", "Specialized"),
            ModelType::SmolLM3 => ("SmolLM 3", "Specialized"),
            ModelType::MiMo => ("MiMo (multi-token prediction)", "Specialized"),
            ModelType::Apertus => ("Apertus (dense)", "Specialized"),
            ModelType::SeedOss => ("Seed-OSS", "Specialized"),

            // ----- Other VLM (cross-family vision-language stacks) -----
            ModelType::LlavaVLM => ("LLaVA (CLIP/SigLIP + Llama/Qwen2)", "Other VLM"),
            ModelType::GraniteVisionVLM => ("Granite Vision (SigLIP + Granite)", "Granite VLM"),
            ModelType::Granite4VisionVLM => (
                "Granite 4 Vision (SigLIP + Granite 4 hybrid)",
                "Granite VLM",
            ),
            ModelType::DeepSeekOcrVLM => ("DeepSeek-OCR (SAM + CLIP + DeepSeek MoE)", "Other VLM"),
            ModelType::DeepSeekOcr2VLM => (
                "DeepSeek-OCR 2 (SAM + Qwen2 resampler + DeepSeek MoE)",
                "Other VLM",
            ),
            ModelType::UnlimitedOcrVLM => (
                "Unlimited-OCR (SAM + CLIP + DeepSeek MoE, ring sliding decode)",
                "Other VLM",
            ),
            ModelType::DeepSeekVL2 => (
                "DeepSeek-VL2 (SigLIP + downsample MLP + DeepSeek-V2 MoE)",
                "Other VLM",
            ),
            ModelType::LlavaBunnyVLM => ("LLaVA-Bunny (SigLIP + Qwen2)", "Other VLM"),
            ModelType::FastVLM => ("FastVLM (FastViTHD + Qwen2)", "Other VLM"),
            ModelType::Ernie45MoeVLM => ("ERNIE 4.5 MoE VL (DFNRope + MoE)", "ERNIE"),
            ModelType::HunyuanVLM => ("Hunyuan-VL (ViT + XD-RoPE)", "Other VLM"),
            ModelType::InternVLChatVLM => {
                ("InternVL (InternViT + pixel-shuffle + Qwen2)", "Other VLM")
            }
            ModelType::LocateAnythingVLM => (
                "LocateAnything (MoonViT + MLP connector + Qwen2, grounding)",
                "Other VLM",
            ),
            ModelType::SmolVLM => ("SmolVLM (SigLIP + pixel-shuffle + SmolLM2)", "Other VLM"),
            ModelType::Idefics2 => (
                "Idefics2 (SigLIP + perceiver resampler + Mistral)",
                "Other VLM",
            ),
            ModelType::MolmoVLM => ("Molmo (CLIP ViT + OLMo-style text)", "Other VLM"),
            ModelType::Molmo2VLM => ("Molmo 2 (custom ViT + Molmo2 text)", "Other VLM"),
            ModelType::MolmoPointVLM => {
                ("Molmo-Point (point prediction + Molmo2 text)", "Other VLM")
            }
            ModelType::Moondream3VLM => ("Moondream 3 (custom ViT + custom decoder)", "Other VLM"),
            ModelType::Moondream2VLM => ("Moondream 2 (SigLIP-style ViT + Phi text)", "Other VLM"),
            ModelType::Florence2VLM => (
                "Florence-2 (DaViT + BART seq2seq, task prompts)",
                "Other VLM",
            ),
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

    /// Compiler-enforced completeness: every `ModelType` variant must appear in
    /// `ALL_MODEL_TYPES`, or it is silently absent from `mlxcel arch`.
    ///
    /// `all_variants!` lists each variant exactly once. A `match` guard inside
    /// it makes the compiler reject the list when a variant is missing — so
    /// adding a `ModelType` variant is a build error until it is listed here —
    /// and the same list is iterated to assert membership in `ALL_MODEL_TYPES`.
    /// A new model wired into the enum and `metadata()` but forgotten in
    /// `ALL_MODEL_TYPES` therefore fails this test (the prior `count > 80` check
    /// could not catch it).
    #[test]
    fn every_variant_is_registered_for_arch() {
        macro_rules! all_variants {
            ($($v:ident),+ $(,)?) => {{
                // Exhaustiveness guard: a missing variant is a build error here.
                fn _exhaustive(mt: ModelType) {
                    match mt {
                        $(ModelType::$v => {}),+
                    }
                }
                [$(ModelType::$v),+]
            }};
        }
        let variants = all_variants!(
            Llama,
            Llama4,
            Llama4VLM,
            MllamaVLM,
            Qwen2,
            Qwen3,
            Qwen3Moe,
            Qwen3Next,
            Qwen35,
            Qwen35VLM,
            Qwen35Moe,
            Qwen35MoeVLM,
            Gemma,
            Gemma2,
            Gemma3,
            Gemma4,
            DiffusionGemma,
            Llada2Moe,
            Gemma3VLM,
            Gemma4VLM,
            Gemma4Unified,
            LlavaVLM,
            GraniteVisionVLM,
            Granite4VisionVLM,
            DeepSeekOcrVLM,
            DeepSeekOcr2VLM,
            UnlimitedOcrVLM,
            DeepSeekVL2,
            LlavaBunnyVLM,
            FastVLM,
            Ernie45MoeVLM,
            HunyuanVLM,
            AyaVisionVLM,
            PaliGemmaVLM,
            PixtralVLM,
            Mistral3VLM,
            Qwen2VL,
            Qwen25VL,
            Qwen3VL,
            Qwen3VLMoe,
            Qwen3OmniMoe,
            PaddleOcrVL,
            DotsOcrVL,
            FalconOcrVL,
            JinaVLM,
            Glm4v,
            Glm4vMoe,
            GlmOcr,
            YoutuVLM,
            InternVLChatVLM,
            LocateAnythingVLM,
            SmolVLM,
            Idefics2,
            MiniCPMOVLM,
            MiniCPMV46VLM,
            Moondream3VLM,
            Moondream2VLM,
            Florence2VLM,
            Gemma3n,
            Gemma3nVLM,
            Phi,
            Phixtral,
            Phi3,
            Phi4MMVLM,
            Phi4SigLipVLM,
            Phi3VLM,
            MolmoVLM,
            Molmo2VLM,
            MolmoPointVLM,
            Phi3Small,
            PhiMoe,
            GptOss,
            MiniMax,
            MiniMaxM3,
            MiniMaxM3VL,
            MuseGlimmerVLM,
            Mixtral,
            Qwen2Moe,
            OLMoE,
            Dbrx,
            DeepSeek,
            DeepSeekV2,
            DeepSeekV3,
            DeepSeekV32,
            DeepSeekV4,
            Dots1,
            Cohere,
            Cohere2,
            Cohere2Moe,
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
            BailingMoe,
            BailingMoeLinear,
            Afmoe,
            Klear,
            Apertus,
            SeedOss,
            Granite,
            BitNet,
            ExaOne,
            ExaOne4,
            ExaOneMoe,
            SolarOpen,
            Olmo,
            Olmo2,
            Olmo3,
            OpenElm,
            Gpt2,
            GptBigCode,
            GptNeoX,
            StarCoder2,
            Mellum,
            Helium,
            TeleChat3,
            MiniCPM,
            MiniCPM3,
            StableLM,
            SmolLM3,
            Ministral3,
            Mistral3,
            Mistral4,
            Nemotron,
            Mamba,
            Mamba2,
            Jamba,
            NemotronH,
            NemotronHNanoOmniVLM,
            NemotronNAS,
            FalconH1,
            Lfm2,
            Lfm2Moe,
            Lfm2VL,
            Inkling,
            Plamo2,
            GraniteMoeHybrid,
            KimiLinear,
            KimiVL,
            KimiK25,
            LongcatFlash,
            LongcatFlashNgram,
            Step3p5,
            Step3p7,
            Rwkv7,
            RecurrentGemma,
            Whisper,
            Kokoro,
            Bert,
            XlmRoberta,
            ModernBert,
            SiglipText,
            Gemma3Embedding,
            Qwen3Embedding,
            Qwen3VLEmbedding,
            Lfm2Embedding,
            Ministral3Embedding,
            LlamaBidirec,
            LlamaNemotronVLEmbedding,
            ColIdefics3,
            ColQwen25,
            SequenceClassifier,
        );
        for mt in variants {
            assert!(
                ALL_MODEL_TYPES.contains(&mt),
                "{mt:?} is a ModelType variant but is missing from ALL_MODEL_TYPES; \
                 it will not appear in `mlxcel arch`. Add it to ALL_MODEL_TYPES."
            );
        }
        // Every variant is registered and the slice has no duplicates
        // (all_model_types_has_no_duplicates), so the lengths must match.
        assert_eq!(
            variants.len(),
            ALL_MODEL_TYPES.len(),
            "ALL_MODEL_TYPES has {} entries but there are {} ModelType variants",
            ALL_MODEL_TYPES.len(),
            variants.len(),
        );
    }

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
#[path = "seed_oss_tests.rs"]
mod seed_oss_tests;

#[cfg(test)]
#[path = "dots1_tests.rs"]
mod dots1_tests;

#[cfg(test)]
#[path = "conv_decode_tests.rs"]
mod conv_decode_tests;

#[cfg(test)]
#[path = "lfm2_tests.rs"]
mod lfm2_tests;

#[cfg(test)]
#[path = "phimoe_tests.rs"]
mod phimoe_tests;

#[cfg(test)]
#[path = "falcon_h1_tests.rs"]
mod falcon_h1_tests;

#[cfg(test)]
#[path = "plamo2_tests.rs"]
mod plamo2_tests;

#[cfg(test)]
#[path = "mellum_tests.rs"]
mod mellum_tests;

#[cfg(test)]
#[path = "granitemoehybrid_tests.rs"]
mod granitemoehybrid_tests;

#[cfg(test)]
#[path = "minimax_tests.rs"]
mod minimax_tests;

#[cfg(test)]
#[path = "modernbert_tests.rs"]
mod modernbert_tests;

#[cfg(test)]
#[path = "modernbert_real_checkpoint_tests.rs"]
mod modernbert_real_checkpoint_tests;
