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

use serde::Serialize;

use super::{ALL_MODEL_TYPES, ModelType};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Runtime {
    Generate,
    Serve,
    Embed,
    Rerank,
    Asr,
    Tts,
    Detect,
    Predict,
    ImageGenerate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Text,
    Image,
    Video,
    Audio,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputKind {
    Tokens,
    Embeddings,
    Scores,
    Audio,
    Boxes,
    DenseMap,
    Mesh,
    Keypoints,
    Masks,
    Image,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendStatus {
    Supported,
    Partial,
    Unsupported,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Drafter {
    Mtp,
    Dflash,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KvMode {
    Fp16,
    Int8,
    Turbo4,
}

#[derive(Debug, Clone, Copy)]
pub struct FamilyCapabilities {
    pub model_types: &'static [&'static str],
    pub aliases: &'static [&'static str],
    pub runtimes: &'static [Runtime],
    pub modalities_in: &'static [Modality],
    pub output: OutputKind,
    pub backends: BackendSupport,
    pub tensor_parallel: bool,
    pub pipeline_parallel: bool,
    pub drafters: &'static [Drafter],
    pub kv_modes: &'static [KvMode],
}

#[derive(Debug, Serialize)]
pub struct ArchitectureRegistry {
    pub mlxcel_version: &'static str,
    /// Complete schema vocabulary, including capabilities whose model rows
    /// have not landed yet.
    pub runtime_kinds: &'static [Runtime],
    pub output_kinds: &'static [OutputKind],
    pub families: Vec<ArchitectureFamily>,
}

const ALL_RUNTIMES: &[Runtime] = &[
    Runtime::Generate,
    Runtime::Serve,
    Runtime::Embed,
    Runtime::Rerank,
    Runtime::Asr,
    Runtime::Tts,
    Runtime::Detect,
    Runtime::Predict,
    Runtime::ImageGenerate,
];

const ALL_OUTPUT_KINDS: &[OutputKind] = &[
    OutputKind::Tokens,
    OutputKind::Embeddings,
    OutputKind::Scores,
    OutputKind::Audio,
    OutputKind::Boxes,
    OutputKind::DenseMap,
    OutputKind::Mesh,
    OutputKind::Keypoints,
    OutputKind::Masks,
    OutputKind::Image,
];

#[derive(Debug, Serialize)]
pub struct ArchitectureFamily {
    pub id: &'static str,
    pub display_name: &'static str,
    pub category: &'static str,
    pub model_types: Vec<&'static str>,
    pub aliases: &'static [&'static str],
    pub runtimes: &'static [Runtime],
    pub modalities_in: &'static [Modality],
    pub output: OutputKind,
    pub backends: BackendSupport,
    pub tensor_parallel: bool,
    pub pipeline_parallel: bool,
    pub drafters: &'static [Drafter],
    pub kv_modes: &'static [KvMode],
}

/// Per-backend support, one named column per GPU backend.
///
/// Named fields rather than a positional array, so a new column has to be
/// written at every construction site. Serialized in declaration order into
/// `mlxcel arch --json`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BackendSupport {
    pub metal: BackendStatus,
    pub cuda: BackendStatus,
    pub rocm: BackendStatus,
}

/// First-pass ROCm status, applied to every family (issue #1886).
///
/// ROCm is an experimental backend (docs/installation.md, "Linux with AMD ROCm
/// (experimental)", "Current status"). mlxcel's fused kernels run there as MLX
/// graph fallbacks until #1814, and only a handful of checkpoints have been
/// run on AMD: Qwen3 dense and MoE, gpt-oss MXFP4, and the audio paths. So
/// this is a starting value, not a per-family result. `Partial` keeps every
/// family runnable in the WebUI catalog, whose predicate admits
/// `Supported | Partial`, without claiming parity with Metal in
/// `mlxcel arch --json`. Refine it per family as each is validated on AMD
/// under epic #1801.
const ROCM_FIRST_PASS: BackendStatus = BackendStatus::Partial;

const GENERATE_SERVE: &[Runtime] = &[Runtime::Generate, Runtime::Serve];
const GENERATE_SERVE_RERANK: &[Runtime] = &[Runtime::Generate, Runtime::Serve, Runtime::Rerank];
const EMBED: &[Runtime] = &[Runtime::Embed];
const RERANK: &[Runtime] = &[Runtime::Rerank];
const ASR: &[Runtime] = &[Runtime::Asr];
const TTS: &[Runtime] = &[Runtime::Tts];
const DETECT: &[Runtime] = &[Runtime::Detect];
const PREDICT: &[Runtime] = &[Runtime::Predict];
const IMAGE_GENERATE: &[Runtime] = &[Runtime::ImageGenerate];

const TEXT: &[Modality] = &[Modality::Text];
const IMAGE: &[Modality] = &[Modality::Image];
const AUDIO: &[Modality] = &[Modality::Audio];
const TEXT_IMAGE: &[Modality] = &[Modality::Text, Modality::Image];
const IMAGE_VIDEO: &[Modality] = &[Modality::Image, Modality::Video];
const TEXT_IMAGE_VIDEO: &[Modality] = &[Modality::Text, Modality::Image, Modality::Video];
const TEXT_IMAGE_VIDEO_AUDIO: &[Modality] = &[
    Modality::Text,
    Modality::Image,
    Modality::Video,
    Modality::Audio,
];

const KV_FP16: &[KvMode] = &[KvMode::Fp16];
const KV_TURBO: &[KvMode] = &[KvMode::Fp16, KvMode::Int8, KvMode::Turbo4];
const NO_DRAFTERS: &[Drafter] = &[];
const MTP: &[Drafter] = &[Drafter::Mtp];
const MTP_DFLASH: &[Drafter] = &[Drafter::Mtp, Drafter::Dflash];
/// DFlash round loop only: the Muse Glimmer assistant drafter (issue #1343).
const DFLASH: &[Drafter] = &[Drafter::Dflash];

const EMPTY_KEYS: &[&str] = &[];
const QWEN35_ALIASES: &[&str] = &["Qwen 3.8"];
const QWEN35_MOE_ALIASES: &[&str] = &["Qwen 3.6"];
const QWEN35_VLM_ALIASES: &[&str] = &["Qwen 3.8 VL"];

#[derive(Debug, Clone, Copy)]
struct StandaloneArchitectureFamily {
    id: &'static str,
    display_name: &'static str,
    category: &'static str,
    model_types: &'static [&'static str],
    aliases: &'static [&'static str],
    runtimes: &'static [Runtime],
    modalities_in: &'static [Modality],
    output: OutputKind,
    backends: BackendSupport,
    tensor_parallel: bool,
    pipeline_parallel: bool,
    drafters: &'static [Drafter],
    kv_modes: &'static [KvMode],
}

const STANDALONE_ARCHITECTURE_FAMILIES: &[StandaloneArchitectureFamily] = &[
    StandaloneArchitectureFamily {
        id: "rt_detr_v2",
        display_name: "RT-DETRv2",
        category: "detection",
        model_types: &["rt_detr_v2"],
        aliases: EMPTY_KEYS,
        runtimes: DETECT,
        modalities_in: IMAGE,
        output: OutputKind::Boxes,
        backends: BackendSupport {
            metal: BackendStatus::Supported,
            cuda: BackendStatus::Supported,
            rocm: ROCM_FIRST_PASS,
        },
        tensor_parallel: false,
        pipeline_parallel: false,
        drafters: NO_DRAFTERS,
        kv_modes: KV_FP16,
    },
    StandaloneArchitectureFamily {
        id: "pp_doclayout_v3",
        display_name: "PP-DocLayoutV3",
        category: "detection",
        model_types: &["pp_doclayout_v3"],
        aliases: EMPTY_KEYS,
        runtimes: DETECT,
        modalities_in: IMAGE,
        output: OutputKind::Boxes,
        backends: BackendSupport {
            metal: BackendStatus::Supported,
            cuda: BackendStatus::Supported,
            rocm: ROCM_FIRST_PASS,
        },
        tensor_parallel: false,
        pipeline_parallel: false,
        drafters: NO_DRAFTERS,
        kv_modes: KV_FP16,
    },
    StandaloneArchitectureFamily {
        id: "rf-detr",
        display_name: "RF-DETR",
        category: "detection",
        model_types: &["rf-detr"],
        aliases: &["rfdetr"],
        runtimes: DETECT,
        modalities_in: IMAGE,
        output: OutputKind::Boxes,
        backends: BackendSupport {
            metal: BackendStatus::Partial,
            cuda: BackendStatus::Partial,
            rocm: BackendStatus::Supported,
        },
        tensor_parallel: false,
        pipeline_parallel: false,
        drafters: NO_DRAFTERS,
        kv_modes: KV_FP16,
    },
    StandaloneArchitectureFamily {
        id: "dinov2",
        display_name: "DINOv2",
        category: "vision encoder",
        model_types: &["dinov2", "dinov2_with_registers"],
        aliases: EMPTY_KEYS,
        runtimes: PREDICT,
        modalities_in: IMAGE,
        output: OutputKind::Embeddings,
        backends: BackendSupport {
            metal: BackendStatus::Supported,
            cuda: BackendStatus::Supported,
            rocm: ROCM_FIRST_PASS,
        },
        tensor_parallel: false,
        pipeline_parallel: false,
        drafters: NO_DRAFTERS,
        kv_modes: KV_FP16,
    },
    StandaloneArchitectureFamily {
        id: "sam3d_body",
        display_name: "SAM 3D Body",
        category: "dense_prediction",
        model_types: &["sam3d_body"],
        aliases: EMPTY_KEYS,
        runtimes: PREDICT,
        modalities_in: IMAGE,
        output: OutputKind::Mesh,
        backends: BackendSupport {
            metal: BackendStatus::Supported,
            cuda: BackendStatus::Supported,
            rocm: ROCM_FIRST_PASS,
        },
        tensor_parallel: false,
        pipeline_parallel: false,
        drafters: NO_DRAFTERS,
        kv_modes: KV_FP16,
    },
    StandaloneArchitectureFamily {
        id: "sam3d_objects",
        display_name: "SAM 3D Objects",
        category: "dense_prediction",
        model_types: &["sam3d_objects"],
        aliases: EMPTY_KEYS,
        runtimes: PREDICT,
        modalities_in: IMAGE,
        output: OutputKind::Mesh,
        backends: BackendSupport {
            metal: BackendStatus::Supported,
            cuda: BackendStatus::Supported,
            rocm: ROCM_FIRST_PASS,
        },
        tensor_parallel: false,
        pipeline_parallel: false,
        drafters: NO_DRAFTERS,
        kv_modes: KV_FP16,
    },
    StandaloneArchitectureFamily {
        id: "moge3",
        display_name: "MoGe3",
        category: "dense_prediction",
        model_types: &["moge3"],
        aliases: EMPTY_KEYS,
        runtimes: PREDICT,
        modalities_in: IMAGE,
        output: OutputKind::DenseMap,
        backends: BackendSupport {
            metal: BackendStatus::Supported,
            cuda: BackendStatus::Supported,
            rocm: ROCM_FIRST_PASS,
        },
        tensor_parallel: false,
        pipeline_parallel: false,
        drafters: NO_DRAFTERS,
        kv_modes: KV_FP16,
    },
    StandaloneArchitectureFamily {
        id: "sapiens2",
        display_name: "Sapiens2",
        category: "dense_prediction",
        model_types: &["sapiens2"],
        aliases: EMPTY_KEYS,
        runtimes: PREDICT,
        modalities_in: IMAGE,
        output: OutputKind::DenseMap,
        backends: BackendSupport {
            metal: BackendStatus::Supported,
            cuda: BackendStatus::Supported,
            rocm: ROCM_FIRST_PASS,
        },
        tensor_parallel: false,
        pipeline_parallel: false,
        drafters: NO_DRAFTERS,
        kv_modes: KV_FP16,
    },
    StandaloneArchitectureFamily {
        id: "video_depth_anything",
        display_name: "Video Depth Anything",
        category: "dense_prediction",
        model_types: &["video_depth_anything"],
        aliases: EMPTY_KEYS,
        runtimes: PREDICT,
        modalities_in: IMAGE_VIDEO,
        output: OutputKind::DenseMap,
        backends: BackendSupport {
            metal: BackendStatus::Supported,
            cuda: BackendStatus::Supported,
            rocm: ROCM_FIRST_PASS,
        },
        tensor_parallel: false,
        pipeline_parallel: false,
        drafters: NO_DRAFTERS,
        kv_modes: KV_FP16,
    },
    StandaloneArchitectureFamily {
        id: "sam3",
        display_name: "SAM 3",
        category: "segmentation",
        model_types: &["sam3_video", "sam3"],
        aliases: EMPTY_KEYS,
        runtimes: PREDICT,
        modalities_in: TEXT_IMAGE_VIDEO,
        output: OutputKind::Masks,
        backends: BackendSupport {
            metal: BackendStatus::Supported,
            cuda: BackendStatus::Supported,
            rocm: ROCM_FIRST_PASS,
        },
        tensor_parallel: false,
        pipeline_parallel: false,
        drafters: NO_DRAFTERS,
        kv_modes: KV_FP16,
    },
    StandaloneArchitectureFamily {
        id: "sam3_1",
        display_name: "SAM 3.1",
        category: "segmentation",
        model_types: &["sam3.1_video", "sam3.1"],
        aliases: &["sam31"],
        runtimes: PREDICT,
        modalities_in: TEXT_IMAGE_VIDEO,
        output: OutputKind::Masks,
        backends: BackendSupport {
            metal: BackendStatus::Supported,
            cuda: BackendStatus::Supported,
            rocm: ROCM_FIRST_PASS,
        },
        tensor_parallel: false,
        pipeline_parallel: false,
        drafters: NO_DRAFTERS,
        kv_modes: KV_FP16,
    },
    StandaloneArchitectureFamily {
        id: "flux2",
        display_name: "FLUX.2 Klein",
        category: "image generation",
        model_types: &["flux2"],
        aliases: EMPTY_KEYS,
        runtimes: IMAGE_GENERATE,
        modalities_in: TEXT_IMAGE,
        output: OutputKind::Image,
        backends: BackendSupport {
            metal: BackendStatus::Partial,
            cuda: BackendStatus::Supported,
            rocm: ROCM_FIRST_PASS,
        },
        tensor_parallel: false,
        pipeline_parallel: false,
        drafters: NO_DRAFTERS,
        kv_modes: KV_FP16,
    },
    StandaloneArchitectureFamily {
        id: "z_image",
        display_name: "Z-Image",
        category: "image generation",
        model_types: &["z_image"],
        aliases: &["z-image", "z-image-turbo"],
        runtimes: IMAGE_GENERATE,
        modalities_in: TEXT_IMAGE,
        output: OutputKind::Image,
        backends: BackendSupport {
            metal: BackendStatus::Supported,
            cuda: BackendStatus::Supported,
            rocm: ROCM_FIRST_PASS,
        },
        tensor_parallel: false,
        pipeline_parallel: false,
        drafters: NO_DRAFTERS,
        kv_modes: KV_FP16,
    },
    StandaloneArchitectureFamily {
        id: "bonsai",
        display_name: "Prism Bonsai (ternary FLUX.2 Klein)",
        category: "image generation",
        model_types: &["bonsai"],
        aliases: EMPTY_KEYS,
        runtimes: IMAGE_GENERATE,
        modalities_in: TEXT,
        output: OutputKind::Image,
        backends: BackendSupport {
            metal: BackendStatus::Partial,
            cuda: BackendStatus::Supported,
            rocm: ROCM_FIRST_PASS,
        },
        tensor_parallel: false,
        pipeline_parallel: false,
        drafters: NO_DRAFTERS,
        kv_modes: KV_FP16,
    },
    StandaloneArchitectureFamily {
        id: "mage_flow",
        display_name: "Mage-Flow",
        category: "image generation",
        model_types: &["mage_flow"],
        aliases: &["mage-flow", "mage-flow-turbo", "mage-flow-edit"],
        runtimes: IMAGE_GENERATE,
        modalities_in: TEXT_IMAGE,
        output: OutputKind::Image,
        backends: BackendSupport {
            metal: BackendStatus::Supported,
            cuda: BackendStatus::Supported,
            rocm: ROCM_FIRST_PASS,
        },
        tensor_parallel: false,
        pipeline_parallel: false,
        drafters: NO_DRAFTERS,
        kv_modes: KV_FP16,
    },
];

const TENSOR_PARALLEL_MODEL_TYPES: &[ModelType] = &[
    ModelType::Llama,
    ModelType::Qwen2,
    ModelType::Qwen3,
    ModelType::Qwen35,
    ModelType::Qwen35VLM,
    ModelType::Gemma3,
    ModelType::Gemma4,
    ModelType::Gemma4VLM,
    ModelType::Ernie45,
    ModelType::HunyuanV1Dense,
];

const PIPELINE_PARALLEL_MODEL_TYPES: &[ModelType] = &[
    ModelType::Llama,
    ModelType::IQuestCoder,
    ModelType::Llama4,
    ModelType::Llama4VLM,
    ModelType::Mixtral,
    ModelType::DeepSeekV3,
    ModelType::GptOss,
    ModelType::Gemma3,
    ModelType::Gemma4,
    ModelType::Gemma4VLM,
    ModelType::Glm4,
    ModelType::Glm4Moe,
    ModelType::Glm4MoeLite,
    ModelType::GlmMoeDsa,
    ModelType::Qwen3,
    ModelType::Qwen3Next,
    ModelType::Qwen35,
    ModelType::Qwen35VLM,
    ModelType::Qwen35Moe,
    ModelType::Qwen35MoeVLM,
    ModelType::Jamba,
    ModelType::NemotronH,
];

impl ModelType {
    pub const fn registry_id(self) -> &'static str {
        match self {
            ModelType::Llama => "llama",
            ModelType::IQuestCoder => "iquest_coder",
            ModelType::IQuestLoopCoder => "iquest_loop_coder",
            ModelType::Llama4 => "llama4",
            ModelType::Llama4VLM => "llama4_vlm",
            ModelType::MllamaVLM => "mllama_vlm",
            ModelType::Qwen2 => "qwen2",
            ModelType::Qwen3 => "qwen3",
            ModelType::Qwen3Moe => "qwen3_moe",
            ModelType::Qwen3Next => "qwen3_next",
            ModelType::Qwen35 => "qwen3_5",
            ModelType::Qwen35VLM => "qwen3_5_vlm",
            ModelType::Qwen35Moe => "qwen3_5_moe",
            ModelType::Qwen35MoeVLM => "qwen3_5_moe_vlm",
            ModelType::Qwen4ExpVLM => "qwen4_exp_vlm",
            ModelType::Gemma => "gemma",
            ModelType::Gemma2 => "gemma2",
            ModelType::Gemma3 => "gemma3",
            ModelType::Gemma4 => "gemma4",
            ModelType::DiffusionGemma => "diffusion_gemma",
            ModelType::Llada2Moe => "llada2_moe",
            ModelType::Gemma3VLM => "gemma3_vlm",
            ModelType::Gemma4VLM => "gemma4_vlm",
            ModelType::Gemma4Unified => "gemma4_unified",
            ModelType::LlavaVLM => "llava_vlm",
            ModelType::GraniteVisionVLM => "granite_vision_vlm",
            ModelType::Granite4VisionVLM => "granite4_vision_vlm",
            ModelType::DeepSeekOcrVLM => "deepseek_ocr_vlm",
            ModelType::DeepSeekOcr2VLM => "deepseek_ocr2_vlm",
            ModelType::UnlimitedOcrVLM => "unlimited_ocr_vlm",
            ModelType::DeepSeekVL2 => "deepseek_vl2",
            ModelType::LlavaBunnyVLM => "llava_bunny_vlm",
            ModelType::FastVLM => "fast_vlm",
            ModelType::Ernie45MoeVLM => "ernie4_5_moe_vlm",
            ModelType::HunyuanVLM => "hunyuan_vlm",
            ModelType::AyaVisionVLM => "aya_vision_vlm",
            ModelType::PaliGemmaVLM => "paligemma_vlm",
            ModelType::PixtralVLM => "pixtral_vlm",
            ModelType::Mistral3VLM => "mistral3_vlm",
            ModelType::Qwen2VL => "qwen2_vl",
            ModelType::Qwen25VL => "qwen2_5_vl",
            ModelType::Qwen3VL => "qwen3_vl",
            ModelType::Qwen3VLMoe => "qwen3_vl_moe",
            ModelType::Qwen3OmniMoe => "qwen3_omni_moe",
            ModelType::CohereCompassVLM => "cohere_compass_vlm",
            ModelType::PaddleOcrVL => "paddleocr_vl",
            ModelType::DotsOcrVL => "dots_ocr_vl",
            ModelType::FalconOcrVL => "falcon_ocr_vl",
            ModelType::JinaVLM => "jina_vlm",
            ModelType::Glm4v => "glm4v",
            ModelType::Glm4vMoe => "glm4v_moe",
            ModelType::GlmOcr => "glm_ocr",
            ModelType::YoutuLLM => "youtu_llm",
            ModelType::YoutuVLM => "youtu_vlm",
            ModelType::InternVLChatVLM => "internvl_chat_vlm",
            ModelType::LlmJpVLM => "llmjp_vlm",
            ModelType::MageVLM => "mage_vlm",
            ModelType::GotOcrVLM => "got_ocr_vlm",
            ModelType::LocateAnythingVLM => "locateanything_vlm",
            ModelType::SmolVLM => "smolvlm",
            ModelType::Idefics2 => "idefics2",
            ModelType::MiniCPMOVLM => "minicpmo_vlm",
            ModelType::MiniCPMV46VLM => "minicpmv4_6_vlm",
            ModelType::Moondream3VLM => "moondream3_vlm",
            ModelType::Moondream2VLM => "moondream2_vlm",
            ModelType::Florence2VLM => "florence2_vlm",
            ModelType::NemotronParseVLM => "nemotron_parse_vlm",
            ModelType::Gemma3n => "gemma3n",
            ModelType::Gemma3nVLM => "gemma3n_vlm",
            ModelType::Phi => "phi",
            ModelType::Phixtral => "phixtral",
            ModelType::Phi3 => "phi3",
            ModelType::Phi4MMVLM => "phi4_mm_vlm",
            ModelType::Phi4SigLipVLM => "phi4_siglip_vlm",
            ModelType::Phi3VLM => "phi3_vlm",
            ModelType::MolmoVLM => "molmo_vlm",
            ModelType::Molmo2VLM => "molmo2_vlm",
            ModelType::MolmoPointVLM => "molmo_point_vlm",
            ModelType::Phi3Small => "phi3small",
            ModelType::PhiMoe => "phimoe",
            ModelType::GptOss => "gpt_oss",
            ModelType::MiniMax => "minimax",
            ModelType::MiniMaxM3 => "minimax_m3",
            ModelType::MiniMaxM3VL => "minimax_m3_vl",
            ModelType::MuseGlimmer => "muse_glimmer",
            ModelType::MuseGlimmerVLM => "muse_glimmer_vlm",
            ModelType::Mixtral => "mixtral",
            ModelType::Qwen2Moe => "qwen2_moe",
            ModelType::OLMoE => "olmoe",
            ModelType::Dbrx => "dbrx",
            ModelType::DeepSeek => "deepseek",
            ModelType::DeepSeekV2 => "deepseek_v2",
            ModelType::DeepSeekV3 => "deepseek_v3",
            ModelType::DeepSeekV32 => "deepseek_v32",
            ModelType::DeepSeekV4 => "deepseek_v4",
            ModelType::Dots1 => "dots1",
            ModelType::Cohere => "cohere",
            ModelType::Cohere2 => "cohere2",
            ModelType::Cohere2Moe => "cohere2_moe",
            ModelType::InternLM2 => "internlm2",
            ModelType::InternLM3 => "internlm3",
            ModelType::Baichuan => "baichuan",
            ModelType::Glm4 => "glm4",
            ModelType::Glm4Moe => "glm4_moe",
            ModelType::Glm4MoeLite => "glm4_moe_lite",
            ModelType::GlmMoeDsa => "glm_moe_dsa",
            ModelType::Ernie45 => "ernie4_5",
            ModelType::Ernie45Moe => "ernie4_5_moe",
            ModelType::HunyuanMoe => "hunyuan_moe",
            ModelType::HunyuanV1Dense => "hunyuan_v1_dense",
            ModelType::MiMo => "mimo",
            ModelType::BailingMoe => "bailing_moe",
            ModelType::BailingMoeLinear => "bailing_moe_linear",
            ModelType::Afmoe => "afmoe",
            ModelType::Klear => "klear",
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
            ModelType::OpenElm => "openelm",
            ModelType::Gpt2 => "gpt2",
            ModelType::GptBigCode => "gpt_bigcode",
            ModelType::GptNeoX => "gpt_neox",
            ModelType::NanoChat => "nanochat",
            ModelType::StarCoder2 => "starcoder2",
            ModelType::Mellum => "mellum",
            ModelType::Laguna => "laguna",
            ModelType::Helium => "helium",
            ModelType::TeleChat3 => "telechat3",
            ModelType::MiniCPM => "minicpm",
            ModelType::MiniCPM3 => "minicpm3",
            ModelType::StableLM => "stablelm",
            ModelType::SmolLM3 => "smollm3",
            ModelType::Ministral3 => "ministral3",
            ModelType::Mistral3 => "mistral3",
            ModelType::Mistral4 => "mistral4",
            ModelType::Nemotron => "nemotron",
            ModelType::Mamba => "mamba",
            ModelType::Mamba2 => "mamba2",
            ModelType::Jamba => "jamba",
            ModelType::NemotronH => "nemotron_h",
            ModelType::NemotronHNanoOmniVLM => "nemotron_h_nano_omni_vlm",
            ModelType::NemotronNAS => "nemotron_nas",
            ModelType::FalconH1 => "falcon_h1",
            ModelType::Lfm2 => "lfm2",
            ModelType::Lfm2Moe => "lfm2_moe",
            ModelType::Lfm2VL => "lfm2_vl",
            ModelType::Inkling => "inkling",
            ModelType::InklingVLM => "inkling_vlm",
            ModelType::Plamo2 => "plamo2",
            ModelType::GraniteMoeHybrid => "granitemoehybrid",
            ModelType::KimiLinear => "kimi_linear",
            ModelType::KimiK3 => "kimi_k3",
            ModelType::KimiK3VLM => "kimi_k3_vlm",
            ModelType::KimiVL => "kimi_vl",
            ModelType::KimiK25 => "kimi_k25",
            ModelType::LongcatFlash => "longcat_flash",
            ModelType::LongcatFlashNgram => "longcat_flash_ngram",
            ModelType::Step3p5 => "step3p5",
            ModelType::Step3p7 => "step3p7",
            ModelType::Rwkv7 => "rwkv7",
            ModelType::RecurrentGemma => "recurrent_gemma",
            ModelType::Whisper => "whisper",
            ModelType::Kokoro => "kokoro",
            ModelType::NemotronVoiceChat => "nemotron_voicechat",
            ModelType::Bert => "bert",
            ModelType::XlmRoberta => "xlm_roberta",
            ModelType::ModernBert => "modernbert",
            ModelType::SiglipText => "siglip",
            ModelType::Gemma3Embedding => "gemma3_embedding",
            ModelType::EmbeddingGemma2 => "embedding_gemma2",
            ModelType::Qwen3Embedding => "qwen3_embedding",
            ModelType::Qwen3VLEmbedding => "qwen3_vl_embedding",
            ModelType::Lfm2Embedding => "lfm2_embedding",
            ModelType::Ministral3Embedding => "ministral3_embedding",
            ModelType::LlamaBidirec => "llama_bidirec",
            ModelType::LlamaNemotronVLEmbedding => "llama_nemotron_vl_embedding",
            ModelType::ColIdefics3 => "colidefics3",
            ModelType::ColQwen25 => "colqwen2_5",
            ModelType::SequenceClassifier => "sequence_classifier",
        }
    }

    pub fn capabilities(self) -> FamilyCapabilities {
        let runtimes = match self {
            ModelType::Bert
            | ModelType::XlmRoberta
            | ModelType::ModernBert
            | ModelType::SiglipText
            | ModelType::Gemma3Embedding
            | ModelType::EmbeddingGemma2
            | ModelType::Qwen3Embedding
            | ModelType::Qwen3VLEmbedding
            | ModelType::Lfm2Embedding
            | ModelType::Ministral3Embedding
            | ModelType::LlamaBidirec
            | ModelType::LlamaNemotronVLEmbedding
            | ModelType::ColIdefics3
            | ModelType::ColQwen25 => EMBED,
            ModelType::SequenceClassifier => RERANK,
            ModelType::Whisper => ASR,
            ModelType::Kokoro => TTS,
            // `mlxcel generate` runs an offline turn; `mlxcel serve` exposes
            // the `/v1/realtime` WebSocket (#1376).
            ModelType::NemotronVoiceChat => GENERATE_SERVE,
            ModelType::Qwen3 | ModelType::Qwen3VL => GENERATE_SERVE_RERANK,
            _ => GENERATE_SERVE,
        };
        let output = match self {
            ModelType::Bert
            | ModelType::XlmRoberta
            | ModelType::ModernBert
            | ModelType::SiglipText
            | ModelType::Gemma3Embedding
            | ModelType::EmbeddingGemma2
            | ModelType::Qwen3Embedding
            | ModelType::Qwen3VLEmbedding
            | ModelType::Lfm2Embedding
            | ModelType::Ministral3Embedding
            | ModelType::LlamaBidirec
            | ModelType::LlamaNemotronVLEmbedding
            | ModelType::ColIdefics3
            | ModelType::ColQwen25 => OutputKind::Embeddings,
            ModelType::SequenceClassifier => OutputKind::Scores,
            ModelType::Kokoro | ModelType::NemotronVoiceChat => OutputKind::Audio,
            _ => OutputKind::Tokens,
        };
        let modalities_in = modalities_for(self);
        let aliases = match self {
            ModelType::Qwen35 => QWEN35_ALIASES,
            ModelType::Qwen35Moe => QWEN35_MOE_ALIASES,
            ModelType::Qwen35VLM => QWEN35_VLM_ALIASES,
            _ => EMPTY_KEYS,
        };
        let drafters = match self {
            ModelType::Gemma4
            | ModelType::Gemma4VLM
            | ModelType::Gemma4Unified
            | ModelType::Inkling
            | ModelType::InklingVLM => MTP,
            ModelType::Qwen35
            | ModelType::Qwen35VLM
            | ModelType::Qwen35Moe
            | ModelType::Qwen35MoeVLM => MTP_DFLASH,
            ModelType::MuseGlimmerVLM => DFLASH,
            _ => NO_DRAFTERS,
        };
        let cuda = match self {
            ModelType::Kokoro | ModelType::Whisper | ModelType::Gemma4 => BackendStatus::Partial,
            _ => BackendStatus::Supported,
        };
        FamilyCapabilities {
            model_types: model_type_keys(self),
            aliases,
            runtimes,
            modalities_in,
            output,
            backends: BackendSupport {
                metal: BackendStatus::Supported,
                cuda,
                rocm: ROCM_FIRST_PASS,
            },
            tensor_parallel: TENSOR_PARALLEL_MODEL_TYPES.contains(&self),
            pipeline_parallel: PIPELINE_PARALLEL_MODEL_TYPES.contains(&self),
            drafters,
            kv_modes: if supports_turbo_kv(self) {
                KV_TURBO
            } else {
                KV_FP16
            },
        }
    }

    pub fn category(self) -> &'static str {
        match self {
            ModelType::Bert
            | ModelType::XlmRoberta
            | ModelType::ModernBert
            | ModelType::SiglipText
            | ModelType::Gemma3Embedding
            | ModelType::EmbeddingGemma2
            | ModelType::Qwen3Embedding
            | ModelType::Qwen3VLEmbedding
            | ModelType::Lfm2Embedding
            | ModelType::Ministral3Embedding
            | ModelType::LlamaBidirec
            | ModelType::LlamaNemotronVLEmbedding
            | ModelType::ColIdefics3
            | ModelType::ColQwen25 => "embedding",
            ModelType::SequenceClassifier => "reranker",
            ModelType::Whisper => "asr",
            ModelType::Kokoro => "tts",
            ModelType::NemotronVoiceChat => "speech_to_speech",
            ModelType::DiffusionGemma | ModelType::Llada2Moe => "diffusion",
            ModelType::BailingMoeLinear
            | ModelType::KimiLinear
            | ModelType::KimiK3
            | ModelType::KimiK3VLM
            | ModelType::LongcatFlash
            | ModelType::LongcatFlashNgram
            | ModelType::Qwen3Next => "linear_attention",
            ModelType::Mamba
            | ModelType::Mamba2
            | ModelType::Jamba
            | ModelType::GraniteMoeHybrid
            | ModelType::NemotronH
            | ModelType::FalconH1
            | ModelType::Lfm2
            | ModelType::Lfm2Moe
            | ModelType::Qwen35
            | ModelType::Plamo2
            | ModelType::Rwkv7
            | ModelType::RecurrentGemma => "hybrid_ssm",
            ModelType::Llama4
            | ModelType::Qwen3Moe
            | ModelType::Qwen35Moe
            | ModelType::GptOss
            | ModelType::MiniMax
            | ModelType::MiniMaxM3
            | ModelType::Mixtral
            | ModelType::Qwen2Moe
            | ModelType::OLMoE
            | ModelType::Dbrx
            | ModelType::DeepSeek
            | ModelType::DeepSeekV2
            | ModelType::DeepSeekV3
            | ModelType::DeepSeekV32
            | ModelType::DeepSeekV4
            | ModelType::Dots1
            | ModelType::Cohere2Moe
            | ModelType::Glm4Moe
            | ModelType::Glm4MoeLite
            | ModelType::GlmMoeDsa
            | ModelType::Ernie45Moe
            | ModelType::HunyuanMoe
            | ModelType::BailingMoe
            | ModelType::Afmoe
            | ModelType::Klear
            | ModelType::ExaOneMoe
            | ModelType::PhiMoe
            | ModelType::Phixtral
            | ModelType::Step3p5
            | ModelType::MiMo => "moe",
            _ => {
                let family = self.family();
                if matches!(
                    family,
                    "Llama VLM"
                        | "Qwen VLM"
                        | "Gemma VLM"
                        | "Mistral VLM"
                        | "Phi VLM"
                        | "Cohere VLM"
                        | "Granite VLM"
                        | "Nemotron VLM"
                        | "GLM VLM"
                        | "Kimi VLM"
                        | "PaddleOCR VLM"
                        | "MiniMax VLM"
                        | "Muse VLM"
                        | "Step VLM"
                        | "Other VLM"
                ) {
                    "vlm"
                } else {
                    "dense"
                }
            }
        }
    }
}

pub fn build_architecture_registry(mlxcel_version: &'static str) -> ArchitectureRegistry {
    let mut families: Vec<ArchitectureFamily> = ALL_MODEL_TYPES
        .iter()
        .copied()
        .map(|model_type| {
            let caps = model_type.capabilities();
            let model_types = if caps.model_types.is_empty() {
                vec![model_type.registry_id()]
            } else {
                caps.model_types.to_vec()
            };
            ArchitectureFamily {
                id: model_type.registry_id(),
                display_name: model_type.display_name(),
                category: model_type.category(),
                model_types,
                aliases: caps.aliases,
                runtimes: caps.runtimes,
                modalities_in: caps.modalities_in,
                output: caps.output,
                backends: caps.backends,
                tensor_parallel: caps.tensor_parallel,
                pipeline_parallel: caps.pipeline_parallel,
                drafters: caps.drafters,
                kv_modes: caps.kv_modes,
            }
        })
        .collect();

    families.extend(
        STANDALONE_ARCHITECTURE_FAMILIES
            .iter()
            .map(|family| ArchitectureFamily {
                id: family.id,
                display_name: family.display_name,
                category: family.category,
                model_types: family.model_types.to_vec(),
                aliases: family.aliases,
                runtimes: family.runtimes,
                modalities_in: family.modalities_in,
                output: family.output,
                backends: family.backends,
                tensor_parallel: family.tensor_parallel,
                pipeline_parallel: family.pipeline_parallel,
                drafters: family.drafters,
                kv_modes: family.kv_modes,
            }),
    );

    ArchitectureRegistry {
        mlxcel_version,
        runtime_kinds: ALL_RUNTIMES,
        output_kinds: ALL_OUTPUT_KINDS,
        families,
    }
}

fn model_type_keys(model_type: ModelType) -> &'static [&'static str] {
    match model_type {
        ModelType::Llama => &["llama", "mistral", "xverse"],
        ModelType::IQuestCoder => &["iquestcoder"],
        ModelType::IQuestLoopCoder => &["iquestloopcoder"],
        ModelType::Llama4 | ModelType::Llama4VLM => &["llama4"],
        ModelType::MllamaVLM => &["mllama"],
        ModelType::Qwen2 => &["qwen2"],
        ModelType::Qwen3 => &["qwen3"],
        ModelType::Qwen3Moe => &["qwen3_moe"],
        ModelType::Qwen3Next => &["qwen3_next", "qwen3next"],
        ModelType::Qwen35 | ModelType::Qwen35VLM => &["qwen3_5", "qwen3_5_text"],
        ModelType::Qwen35Moe | ModelType::Qwen35MoeVLM => &["qwen3_5_moe", "qwen3_5_moe_text"],
        ModelType::Qwen4ExpVLM => &["qwen4_exp"],
        ModelType::Qwen2Moe => &["qwen2_moe"],
        ModelType::Gemma => &["gemma"],
        ModelType::Gemma2 => &["gemma2"],
        ModelType::Gemma3 | ModelType::Gemma3VLM => &["gemma3", "gemma3_text"],
        ModelType::Gemma4 | ModelType::Gemma4VLM => &["gemma4", "gemma4_text"],
        ModelType::Gemma4Unified => &["gemma4_unified"],
        ModelType::Gemma3n | ModelType::Gemma3nVLM => &["gemma3n", "gemma3n_text"],
        ModelType::DiffusionGemma => &["diffusion_gemma", "diffusion_gemma_text"],
        ModelType::Llada2Moe => &["llada2_moe"],
        ModelType::Phi => &["phi", "phi-msft"],
        ModelType::Phixtral => &["phi", "phi-msft"],
        ModelType::Phi3 => &["phi3"],
        ModelType::Phi4MMVLM => &["phi4mm"],
        ModelType::Phi4SigLipVLM => &["phi4-siglip"],
        ModelType::Phi3VLM => &["phi3_v"],
        ModelType::Phi3Small => &["phi3small"],
        ModelType::PhiMoe => &["phimoe"],
        ModelType::MiniMax => &["minimax", "minimax_m2"],
        ModelType::MiniMaxM3 => &["minimax_m3"],
        ModelType::MiniMaxM3VL => &["minimax_m3_vl"],
        ModelType::MuseGlimmer | ModelType::MuseGlimmerVLM => &["muse_glimmer"],
        ModelType::GptOss => &["gpt_oss"],
        ModelType::Mixtral => &["mixtral"],
        ModelType::Dbrx => &["dbrx"],
        ModelType::OLMoE => &["olmoe"],
        ModelType::DeepSeek => &["deepseek"],
        ModelType::DeepSeekV2 => &["deepseek_v2"],
        ModelType::DeepSeekV3 => &["deepseek_v3", "kimi_k2", "joyai_llm_flash"],
        ModelType::DeepSeekV32 => &["deepseek_v32", "deepseek_v3.2"],
        ModelType::DeepSeekV4 => &["deepseek_v4"],
        ModelType::Dots1 => &["dots1"],
        ModelType::Cohere => &["cohere"],
        ModelType::Cohere2 => &["cohere2"],
        ModelType::Cohere2Moe => &["cohere2_moe"],
        ModelType::InternLM2 => &["internlm2"],
        ModelType::InternLM3 => &["internlm3"],
        ModelType::Baichuan => &["baichuan_m1"],
        ModelType::BitNet => &["bitnet"],
        ModelType::Glm4 => &["glm4"],
        ModelType::Glm4Moe => &["glm4_moe"],
        ModelType::Glm4MoeLite => &["glm4_moe_lite"],
        ModelType::GlmMoeDsa => &["glm_moe_dsa"],
        ModelType::Ernie45 => &["ernie4_5", "ernie4.5"],
        ModelType::Ernie45Moe => &["ernie4_5_moe", "ernie4.5_moe"],
        ModelType::Ernie45MoeVLM => &["ernie4_5_moe_vl", "ernie4.5_moe_vl"],
        ModelType::HunyuanV1Dense => &["hunyuan_v1_dense", "hunyuan_dense"],
        ModelType::HunyuanVLM => &["hunyuan_vl"],
        ModelType::HunyuanMoe => &["hunyuan"],
        ModelType::MiMo => &["mimo"],
        ModelType::BailingMoe => &["bailing_moe"],
        ModelType::BailingMoeLinear => &["bailing_moe_linear"],
        ModelType::Afmoe => &["afmoe"],
        ModelType::Klear => &["klear"],
        ModelType::Apertus => &["apertus"],
        ModelType::SeedOss => &["seed_oss"],
        ModelType::Granite => &["granite"],
        ModelType::ExaOne => &["exaone"],
        ModelType::ExaOne4 => &["exaone4"],
        ModelType::ExaOneMoe => &["exaone_moe"],
        ModelType::Olmo => &["olmo"],
        ModelType::Olmo2 => &["olmo2"],
        ModelType::Olmo3 => &["olmo3"],
        ModelType::OpenElm => &["openelm"],
        ModelType::Gpt2 => &["gpt2"],
        ModelType::GptBigCode => &["gpt_bigcode"],
        ModelType::GptNeoX => &["gpt_neox"],
        ModelType::NanoChat => &["nanochat"],
        ModelType::Helium => &["helium"],
        ModelType::TeleChat3 => &["telechat3"],
        ModelType::StarCoder2 => &["starcoder2"],
        ModelType::Mellum => &["mellum"],
        ModelType::Laguna => &["laguna"],
        ModelType::MiniCPM => &["minicpm"],
        ModelType::MiniCPM3 => &["minicpm3"],
        ModelType::StableLM => &["stablelm"],
        ModelType::SmolLM3 => &["smollm3"],
        ModelType::Ministral3 => &["ministral3"],
        ModelType::Mistral3 | ModelType::Mistral3VLM => &["mistral3"],
        ModelType::Mistral4 => &["mistral4"],
        ModelType::Nemotron => &["nemotron"],
        ModelType::Mamba => &["mamba", "falcon_mamba"],
        ModelType::Mamba2 => &["mamba2"],
        ModelType::Jamba => &["jamba"],
        ModelType::FalconH1 => &["falcon_h1"],
        ModelType::Lfm2 => &["lfm2"],
        ModelType::Lfm2VL => &["lfm2_vl", "lfm2-vl"],
        ModelType::Lfm2Moe => &["lfm2_moe"],
        ModelType::Inkling | ModelType::InklingVLM => &["inkling_mm_model", "inkling"],
        ModelType::Plamo2 => &["plamo2"],
        ModelType::GraniteMoeHybrid => &["granitemoehybrid"],
        ModelType::NemotronH => &["nemotron_h"],
        ModelType::NemotronHNanoOmniVLM => {
            &["nemotron_h_nano_omni", "nemotronh_nano_omni_reasoning_v3"]
        }
        ModelType::NemotronNAS => &["nemotron-nas"],
        ModelType::Rwkv7 => &["rwkv7"],
        ModelType::KimiLinear => &["kimi_linear"],
        ModelType::KimiK3 | ModelType::KimiK3VLM => &["kimi_k3"],
        ModelType::KimiVL => &["kimi_vl"],
        ModelType::KimiK25 => &["kimi_k25"],
        ModelType::LocateAnythingVLM => &["locateanything"],
        ModelType::LongcatFlash => &["longcat_flash"],
        ModelType::LongcatFlashNgram => &["longcat_flash_ngram"],
        ModelType::Step3p5 => &["step3p5"],
        ModelType::Step3p7 => &["step3p7"],
        ModelType::RecurrentGemma => &["recurrent_gemma", "griffin"],
        ModelType::Qwen2VL => &["qwen2_vl"],
        ModelType::Qwen25VL => &["qwen2_5_vl"],
        ModelType::Qwen3VL => &["qwen3_vl"],
        ModelType::Qwen3VLMoe => &["qwen3_vl_moe"],
        ModelType::Qwen3OmniMoe => &["qwen3_omni_moe"],
        ModelType::CohereCompassVLM => &["cohere_compass"],
        ModelType::PaddleOcrVL => &["paddleocr_vl"],
        ModelType::DotsOcrVL => &["dots_ocr"],
        ModelType::FalconOcrVL => &["falcon_ocr"],
        ModelType::Glm4v => &["glm4v"],
        ModelType::Glm4vMoe => &["glm4v_moe"],
        ModelType::GlmOcr => &["glm_ocr"],
        ModelType::YoutuLLM => &["youtu", "youtu_llm"],
        ModelType::YoutuVLM => &["youtu_vl"],
        ModelType::InternVLChatVLM => &["internvl_chat"],
        ModelType::LlmJpVLM => &["llmjpvl"],
        ModelType::MageVLM => &["mage_vl"],
        ModelType::GotOcrVLM => &["got"],
        ModelType::SmolVLM => &["smolvlm", "smolvlm2", "idefics3"],
        ModelType::Idefics2 => &["idefics2"],
        ModelType::MiniCPMOVLM => &["minicpmo"],
        ModelType::MiniCPMV46VLM => &["minicpmv4_6"],
        ModelType::Moondream3VLM => &["moondream3"],
        ModelType::Moondream2VLM => &["moondream2", "moondream1"],
        ModelType::GraniteVisionVLM => &["granite_vision"],
        ModelType::Granite4VisionVLM => &["granite4_vision"],
        ModelType::DeepSeekOcrVLM => &["deepseekocr"],
        ModelType::DeepSeekOcr2VLM => &["deepseekocr_2"],
        ModelType::UnlimitedOcrVLM => &["unlimited-ocr", "unlimited_ocr"],
        ModelType::DeepSeekVL2 => &["deepseek_vl_v2", "deepseek_vl2"],
        ModelType::LlavaVLM => &["llava", "llava_next"],
        ModelType::LlavaBunnyVLM => &["llava_bunny", "bunny-llama", "llava-qwen2"],
        ModelType::FastVLM => &["fastvlm", "llava_qwen2"],
        ModelType::AyaVisionVLM => &["aya_vision"],
        ModelType::PaliGemmaVLM => &["paligemma"],
        ModelType::PixtralVLM => &["pixtral"],
        ModelType::JinaVLM => &["jvlm", "jina_vlm"],
        ModelType::MolmoVLM => &["molmo"],
        ModelType::Molmo2VLM => &["molmo2"],
        ModelType::MolmoPointVLM => &["molmo_point"],
        ModelType::Florence2VLM => &["florence2"],
        ModelType::NemotronParseVLM => &["nemotron_parse"],
        ModelType::Whisper => &["whisper"],
        ModelType::Kokoro => &["kokoro"],
        ModelType::NemotronVoiceChat => &["nemotron_voicechat"],
        ModelType::Bert => &["bert"],
        ModelType::XlmRoberta => &["xlm-roberta", "xlm_roberta"],
        ModelType::ModernBert => &["modernbert"],
        ModelType::SiglipText => &["siglip", "siglip_text_model"],
        ModelType::Gemma3Embedding => &["gemma3_text", "gemma3"],
        ModelType::EmbeddingGemma2 => &["embedding_gemma2", "embedding_gemma2_text"],
        ModelType::Qwen3Embedding => &["qwen3"],
        ModelType::Qwen3VLEmbedding => &["qwen3_vl"],
        ModelType::Lfm2Embedding => &["lfm2"],
        ModelType::Ministral3Embedding => &["ministral3"],
        ModelType::LlamaBidirec => &["llama", "llama_bidirec"],
        ModelType::LlamaNemotronVLEmbedding => &["llama_nemotron_vl"],
        ModelType::ColIdefics3 => &["idefics3"],
        ModelType::ColQwen25 => &["qwen2_5_vl", "colqwen2"],
        ModelType::SequenceClassifier => &["bert", "xlm-roberta", "xlm_roberta", "modernbert"],
        ModelType::SolarOpen => &["solar_open"],
    }
}

fn modalities_for(model_type: ModelType) -> &'static [Modality] {
    match model_type {
        ModelType::Whisper | ModelType::NemotronVoiceChat => AUDIO,
        ModelType::EmbeddingGemma2
        | ModelType::Qwen3VLEmbedding
        | ModelType::LlamaNemotronVLEmbedding
        | ModelType::ColIdefics3
        | ModelType::ColQwen25 => TEXT_IMAGE,
        ModelType::Gemma4VLM
        | ModelType::Gemma4Unified
        | ModelType::Qwen3OmniMoe
        | ModelType::Phi4MMVLM
        | ModelType::InklingVLM
        | ModelType::Gemma3nVLM
        | ModelType::NemotronHNanoOmniVLM => TEXT_IMAGE_VIDEO_AUDIO,
        ModelType::Qwen2VL
        | ModelType::Qwen25VL
        | ModelType::Qwen3VL
        | ModelType::Qwen3VLMoe
        | ModelType::Qwen35VLM
        | ModelType::Qwen35MoeVLM
        | ModelType::Qwen4ExpVLM => TEXT_IMAGE_VIDEO,
        _ if is_vlm_registry_type(model_type) => TEXT_IMAGE,
        _ => TEXT,
    }
}

fn is_vlm_registry_type(model_type: ModelType) -> bool {
    matches!(
        model_type.family(),
        "Llama VLM"
            | "Qwen VLM"
            | "Gemma VLM"
            | "Mistral VLM"
            | "Phi VLM"
            | "Cohere VLM"
            | "Granite VLM"
            | "Nemotron VLM"
            | "GLM VLM"
            | "Kimi VLM"
            | "PaddleOCR VLM"
            | "MiniMax VLM"
            | "Muse VLM"
            | "Step VLM"
            | "Other VLM"
    )
}

fn supports_turbo_kv(model_type: ModelType) -> bool {
    matches!(
        model_type,
        ModelType::Llama
            | ModelType::IQuestCoder
            | ModelType::Llama4
            | ModelType::Llama4VLM
            | ModelType::Qwen2
            | ModelType::Qwen3
            | ModelType::Qwen3Moe
            | ModelType::Qwen3VL
            | ModelType::Qwen3VLMoe
            | ModelType::Qwen35
            | ModelType::Qwen35VLM
            | ModelType::Qwen35Moe
            | ModelType::Qwen35MoeVLM
            | ModelType::Gemma
            | ModelType::Gemma2
            | ModelType::Gemma3
            | ModelType::Gemma3VLM
            | ModelType::Gemma4
            | ModelType::Gemma4VLM
            | ModelType::Mistral3
            | ModelType::Mistral3VLM
            | ModelType::Mixtral
            | ModelType::DeepSeek
            | ModelType::DeepSeekV2
            | ModelType::Cohere
            | ModelType::Cohere2
            | ModelType::InternLM2
            | ModelType::InternLM3
            | ModelType::Glm4
            | ModelType::Ernie45
            | ModelType::HunyuanV1Dense
            | ModelType::Phi
            | ModelType::Phi3
            | ModelType::Phi3VLM
            | ModelType::Phi3Small
            | ModelType::Gpt2
            | ModelType::GptBigCode
            | ModelType::GptNeoX
            | ModelType::StarCoder2
            | ModelType::StableLM
            | ModelType::SmolLM3
            | ModelType::Ministral3
    )
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use mlxcel_core::cache::KVCacheMode;
    use mlxcel_core::cache::turbo::resolve_kv_cache_mode_for_model;

    use super::*;

    #[test]
    fn registry_has_one_family_per_model_type_with_unique_ids() {
        let registry = build_architecture_registry("test");
        assert_eq!(
            registry.families.len(),
            ALL_MODEL_TYPES.len() + STANDALONE_ARCHITECTURE_FAMILIES.len()
        );
        let mut ids = BTreeSet::new();
        for family in &registry.families {
            assert!(ids.insert(family.id), "duplicate registry id {}", family.id);
            assert!(
                !family.model_types.is_empty(),
                "{} has no model_type key",
                family.id
            );
            assert!(!family.runtimes.is_empty(), "{} has no runtime", family.id);
        }
    }

    #[test]
    fn standalone_registry_extensions_do_not_collide_with_model_registry() {
        let model_registry_ids: BTreeSet<_> = ALL_MODEL_TYPES
            .iter()
            .copied()
            .map(ModelType::registry_id)
            .collect();
        let model_registry_keys: BTreeSet<_> = ALL_MODEL_TYPES
            .iter()
            .copied()
            .flat_map(model_type_keys)
            .collect();
        let mut standalone_ids = BTreeSet::new();
        let mut standalone_keys = BTreeSet::new();

        for family in STANDALONE_ARCHITECTURE_FAMILIES {
            assert!(
                standalone_ids.insert(family.id),
                "duplicate standalone registry id {}",
                family.id
            );
            assert!(
                !model_registry_ids.contains(family.id),
                "standalone registry id {} collides with ALL_MODEL_TYPES",
                family.id
            );
            assert!(
                !family.model_types.is_empty(),
                "{} has no model_type key",
                family.id
            );
            for key in family.model_types {
                assert!(
                    standalone_keys.insert(*key),
                    "duplicate standalone model_type key {key}"
                );
                assert!(
                    !model_registry_keys.contains(key),
                    "standalone model_type key {key} collides with ALL_MODEL_TYPES"
                );
            }
        }
    }

    #[test]
    fn registry_json_round_trips() {
        let registry = build_architecture_registry("test");
        let value = serde_json::to_value(&registry).unwrap();
        assert_eq!(value["mlxcel_version"], "test");
        assert_eq!(
            value["families"].as_array().unwrap().len(),
            ALL_MODEL_TYPES.len() + STANDALONE_ARCHITECTURE_FAMILIES.len()
        );

        // Pins the `backends` shape `mlxcel arch --json` emits, `rocm` included
        // (issue #1886): one object per family, one key per GPU backend.
        let backends = |id: &str| {
            value["families"]
                .as_array()
                .unwrap()
                .iter()
                .find(|family| family["id"] == id)
                .unwrap_or_else(|| panic!("registry has no {id} family"))["backends"]
                .clone()
        };
        assert_eq!(
            backends("qwen3"),
            serde_json::json!({"metal": "supported", "cuda": "supported", "rocm": "partial"})
        );
        assert_eq!(
            backends("whisper"),
            serde_json::json!({"metal": "supported", "cuda": "partial", "rocm": "partial"})
        );
        assert_eq!(
            backends("rt_detr_v2"),
            serde_json::json!({"metal": "supported", "cuda": "supported", "rocm": "partial"})
        );
    }

    #[test]
    fn every_family_is_runnable_on_rocm() {
        // The WebUI catalog admits `Supported | Partial`. A family whose ROCm
        // column falls outside that is listed as not runnable on a ROCm host,
        // which is how issue #1886 reported every family.
        for family in build_architecture_registry("test").families {
            assert!(
                matches!(
                    family.backends.rocm,
                    BackendStatus::Supported | BackendStatus::Partial
                ),
                "{} is not runnable on ROCm: {:?}",
                family.id,
                family.backends.rocm
            );
        }
    }

    #[test]
    fn key_acceptance_families_report_expected_capabilities() {
        let qwen3 = family(ModelType::Qwen3);
        assert!(qwen3.runtimes.contains(&Runtime::Generate));
        assert!(qwen3.runtimes.contains(&Runtime::Serve));
        assert!(qwen3.runtimes.contains(&Runtime::Rerank));
        assert!(qwen3.tensor_parallel);
        assert!(qwen3.pipeline_parallel);

        let whisper = family(ModelType::Whisper);
        assert_eq!(whisper.runtimes, ASR);
        assert_eq!(whisper.modalities_in, AUDIO);
        assert_eq!(whisper.output, OutputKind::Tokens);

        let sequence = family(ModelType::SequenceClassifier);
        assert_eq!(sequence.runtimes, RERANK);
        assert_eq!(sequence.output, OutputKind::Scores);
    }

    #[test]
    fn standalone_detection_families_report_detect_only_capabilities() {
        let registry = build_architecture_registry("test");
        for (id, display_name, model_type) in [
            ("rt_detr_v2", "RT-DETRv2", "rt_detr_v2"),
            ("pp_doclayout_v3", "PP-DocLayoutV3", "pp_doclayout_v3"),
            ("rf-detr", "RF-DETR", "rf-detr"),
        ] {
            let family = registry
                .families
                .iter()
                .find(|family| family.id == id)
                .unwrap();

            assert_eq!(family.display_name, display_name);
            assert_eq!(family.category, "detection");
            assert_eq!(family.model_types, [model_type]);
            assert_eq!(family.runtimes, DETECT);
            assert!(!family.runtimes.contains(&Runtime::Generate));
            assert!(!family.runtimes.contains(&Runtime::Serve));
            assert_eq!(family.modalities_in, IMAGE);
            assert_eq!(family.output, OutputKind::Boxes);
            assert!(!family.tensor_parallel);
            assert!(!family.pipeline_parallel);
            assert_eq!(family.drafters, NO_DRAFTERS);
            assert_eq!(family.kv_modes, KV_FP16);
        }
    }

    #[test]
    fn standalone_dinov2_family_reports_feature_capabilities() {
        let dinov2 = build_architecture_registry("test")
            .families
            .into_iter()
            .find(|family| family.id == "dinov2")
            .unwrap();

        assert_eq!(dinov2.display_name, "DINOv2");
        assert_eq!(dinov2.category, "vision encoder");
        assert_eq!(dinov2.model_types, ["dinov2", "dinov2_with_registers"]);
        assert_eq!(dinov2.runtimes, PREDICT);
        assert_eq!(dinov2.modalities_in, IMAGE);
        assert_eq!(dinov2.output, OutputKind::Embeddings);
        assert!(!dinov2.tensor_parallel);
        assert!(!dinov2.pipeline_parallel);
        assert_eq!(dinov2.drafters, NO_DRAFTERS);
        assert_eq!(dinov2.kv_modes, KV_FP16);
    }

    #[test]
    fn video_depth_anything_reports_image_video_depth_capabilities() {
        let vda = build_architecture_registry("test")
            .families
            .into_iter()
            .find(|family| family.id == "video_depth_anything")
            .unwrap();
        assert_eq!(vda.display_name, "Video Depth Anything");
        assert_eq!(vda.category, "dense_prediction");
        assert_eq!(vda.model_types, ["video_depth_anything"]);
        assert_eq!(vda.runtimes, PREDICT);
        assert_eq!(vda.modalities_in, IMAGE_VIDEO);
        assert_eq!(vda.output, OutputKind::DenseMap);
        assert!(!vda.tensor_parallel);
        assert!(!vda.pipeline_parallel);
    }

    #[test]
    fn sam3_reports_promptable_mask_capabilities() {
        let sam3 = build_architecture_registry("test")
            .families
            .into_iter()
            .find(|family| family.id == "sam3")
            .unwrap();

        assert_eq!(sam3.display_name, "SAM 3");
        assert_eq!(sam3.category, "segmentation");
        assert_eq!(sam3.model_types, ["sam3_video", "sam3"]);
        assert_eq!(sam3.runtimes, PREDICT);
        assert_eq!(sam3.modalities_in, TEXT_IMAGE_VIDEO);
        assert_eq!(sam3.output, OutputKind::Masks);
        assert!(!sam3.tensor_parallel);
        assert!(!sam3.pipeline_parallel);
    }

    #[test]
    fn sam31_reports_multiplex_mask_capabilities() {
        let sam31 = build_architecture_registry("test")
            .families
            .into_iter()
            .find(|family| family.id == "sam3_1")
            .unwrap();

        assert_eq!(sam31.display_name, "SAM 3.1");
        assert_eq!(sam31.category, "segmentation");
        assert_eq!(sam31.model_types, ["sam3.1_video", "sam3.1"]);
        assert_eq!(sam31.aliases, ["sam31"]);
        assert_eq!(sam31.runtimes, PREDICT);
        assert_eq!(sam31.modalities_in, TEXT_IMAGE_VIDEO);
        assert_eq!(sam31.output, OutputKind::Masks);
        assert!(!sam31.tensor_parallel);
        assert!(!sam31.pipeline_parallel);
    }

    #[test]
    fn image_generation_runtime_and_output_kind_are_stable_json_values() {
        // `mlxcel arch --json` serializes these enums verbatim; the spellings
        // are what downstream site builds key on, so pin them. A family that
        // lists `ImageGenerate` / `Image` appears in the snapshot with them.
        assert_eq!(
            serde_json::to_value(Runtime::ImageGenerate).unwrap(),
            "image_generate"
        );
        assert_eq!(serde_json::to_value(OutputKind::Image).unwrap(), "image");
    }

    #[test]
    fn no_registered_family_claims_image_generation_before_a_model_ports_it() {
        // The runtime issue ships the foundation with an empty family table;
        // each model issue adds its row to the standalone family table and to
        // `imagegen::FAMILIES` together. A row in only one of the two would
        // advertise a family `mlxcel generate-image` cannot run.
        let advertised: Vec<_> = build_architecture_registry("test")
            .families
            .into_iter()
            .filter(|family| family.runtimes.contains(&Runtime::ImageGenerate))
            .map(|family| family.id)
            .collect();
        let runnable: Vec<_> = crate::imagegen::FAMILIES.iter().map(|f| f.name).collect();
        assert_eq!(
            advertised.len(),
            runnable.len(),
            "{advertised:?} vs {runnable:?}"
        );
        for id in advertised {
            assert!(
                runnable.contains(&id),
                "{id} is advertised but not runnable"
            );
        }
    }

    #[test]
    fn tensor_parallel_registry_set_matches_dispatch_contract() {
        let registry_set: BTreeSet<_> = ALL_MODEL_TYPES
            .iter()
            .copied()
            .filter(|mt| mt.capabilities().tensor_parallel)
            .map(ModelType::registry_id)
            .collect();
        let expected: BTreeSet<_> = TENSOR_PARALLEL_MODEL_TYPES
            .iter()
            .copied()
            .map(ModelType::registry_id)
            .collect();
        assert_eq!(registry_set, expected);
    }

    #[test]
    fn pipeline_parallel_registry_covers_stage_executor_families() {
        let registry_set: BTreeSet<_> = ALL_MODEL_TYPES
            .iter()
            .copied()
            .filter(|mt| mt.capabilities().pipeline_parallel)
            .map(ModelType::registry_id)
            .collect();
        for model_type in PIPELINE_PARALLEL_MODEL_TYPES {
            let id = model_type.registry_id();
            assert!(
                registry_set.contains(id),
                "{model_type:?} is not PP-enabled"
            );
        }
    }

    #[test]
    fn kv_modes_do_not_advertise_runtime_fp16_downgrades() {
        for model_type in ALL_MODEL_TYPES {
            let family = model_type.capabilities();
            for config_key in family.model_types {
                if family.kv_modes.contains(&KvMode::Int8) {
                    let (effective, _) =
                        resolve_kv_cache_mode_for_model(KVCacheMode::Int8, config_key);
                    assert_ne!(
                        effective,
                        KVCacheMode::Fp16,
                        "{model_type:?} advertises int8 for {config_key}, but runtime resolves to fp16"
                    );
                }
                if family.kv_modes.contains(&KvMode::Turbo4) {
                    let (effective, _) =
                        resolve_kv_cache_mode_for_model(KVCacheMode::Turbo4, config_key);
                    assert_ne!(
                        effective,
                        KVCacheMode::Fp16,
                        "{model_type:?} advertises turbo4 for {config_key}, but runtime resolves to fp16"
                    );
                }
            }
        }
    }

    #[test]
    fn mla_latent_cache_families_report_fp16_only_kv_modes() {
        for model_type in [
            ModelType::DeepSeekV3,
            ModelType::DeepSeekV32,
            ModelType::Glm4MoeLite,
            ModelType::GlmMoeDsa,
            ModelType::KimiLinear,
            ModelType::KimiK3,
            ModelType::KimiK3VLM,
            ModelType::LongcatFlashNgram,
        ] {
            assert_eq!(
                family(model_type).kv_modes,
                KV_FP16,
                "{model_type:?} should not advertise quantized KV cache modes"
            );
        }
    }

    fn family(model_type: ModelType) -> ArchitectureFamily {
        build_architecture_registry("test")
            .families
            .into_iter()
            .find(|family| family.id == model_type.registry_id())
            .unwrap()
    }
}
