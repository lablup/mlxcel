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
use mlxcel_core::drafter::dflash::is_dflash_drafter_config;
use serde_json::Value;
use std::path::Path;

use super::ModelType;
use super::sanitize::sanitize_config_json;

/// Canonical error for "this directory is a DFlash speculative drafter, not a
/// model you can load standalone" (#1168).
///
/// Every entry point that resolves a checkpoint directory goes through
/// [`get_model_type`], so raising the rejection there gives the offline
/// `mlxcel generate -m`, server startup, and the distributed stage loaders one
/// shared message instead of each one surfacing a different weight-map symptom.
///
/// Before this arm existed, a DFlash drafter passed to `-m` (or reached by the
/// offline `--draft-model` path, which loads the drafter as a full model) was
/// classified `ModelType::Qwen3` from its `"model_type": "qwen3"`, routed to
/// `Qwen3Model::load`, and died on its first weight lookup with
/// `Weight not found: model.embed_tokens.weight`. That message names a tensor,
/// not the problem.
fn dflash_drafter_not_standalone_error(model_path: &Path) -> anyhow::Error {
    anyhow::anyhow!(
        "{path} is a DFlash speculative drafter checkpoint, not a standalone model. \
         Its config.json declares the DFlashDraftModel architecture and/or a \
         dflash_config block, and its weights carry no embed_tokens and no lm_head \
         because a DFlash drafter borrows both from the target model when it binds. \
         Pass a full model to -m, and pass this directory to --draft-model on \
         `mlxcel-server` (with --draft-kind dflash) to use it as a drafter.",
        path = model_path.display(),
    )
}

pub(crate) fn has_vision_config(config: &serde_json::Value) -> bool {
    config.get("vision_config").is_some()
}

fn gemma4_has_vision_weights(model_path: &Path) -> bool {
    let index_path = model_path.join("model.safetensors.index.json");
    if let Ok(index_str) = std::fs::read_to_string(&index_path)
        && let Ok(index) = serde_json::from_str::<Value>(&index_str)
        && let Some(weight_map) = index.get("weight_map").and_then(Value::as_object)
    {
        // MLX-community checkpoints expose the vision front-end unprefixed
        // (`vision_tower.` / `embed_vision.`); ModelOpt NVFP4 exports nest it
        // under a leading `model.` (`model.vision_tower.` /
        // `model.embed_vision.`). Recognize both so an NVFP4 multimodal
        // checkpoint routes to Gemma4VLM instead of the text-only path, where
        // `normalize_nvfp4_keys` then strips the `model.` prefix (issue #749).
        return weight_map.keys().any(|key| {
            key.starts_with("vision_tower.")
                || key.starts_with("embed_vision.")
                || key.starts_with("model.vision_tower.")
                || key.starts_with("model.embed_vision.")
        });
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

/// Split the `phi` / `phi-msft` arm between the dense Phi decoder and Phixtral.
///
/// No phixtral checkpoint declares `model_type: "phixtral"`.
/// `mlabonne/phixtral-4x2_8` declares `phi-msft`, and upstream mlx-lm reaches
/// its phixtral implementation through `MODEL_REMAPPING` rather than the config
/// value, so an arm keyed on the string `"phixtral"` could never fire. The
/// discriminator is `num_local_experts`, which the sparse config carries and
/// the dense Phi-2 config does not.
///
/// A value of 1 is treated as dense: it describes one expert, which is a dense
/// MLP, and the phixtral block would be a needless indirection over it.
pub(crate) fn detect_phi_model_type(config: &serde_json::Value) -> ModelType {
    let num_local_experts = config["num_local_experts"].as_i64().unwrap_or(0);
    if num_local_experts > 1 {
        ModelType::Phixtral
    } else {
        ModelType::Phi
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

/// `model_type` values that are encoder-only and never generate text. A
/// `BertForMaskedLM` / `ModernBertForMaskedLM` checkpoint loads as an
/// embedder with its MLM head dropped.
const ENCODER_ONLY_MODEL_TYPES: &[&str] = &["bert", "xlm-roberta", "modernbert", "siglip"];

/// `architectures[0]` values that mark an embedding export outright.
const EMBEDDING_ARCHITECTURES: &[&str] = &[
    "BertModel",
    "XLMRobertaModel",
    "ModernBertModel",
    "SiglipModel",
    "SiglipTextModel",
    "LlamaBidirectionalModel",
    "LlamaNemotronVLModel",
    "Lfm2BidirectionalModel",
    "ColIdefics3",
    "ColQwen2_5",
    "ColQwen2ForRetrieval",
];

fn first_architecture(config: &Value) -> Option<&str> {
    config
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|arr| arr.first())
        .and_then(Value::as_str)
}

/// `config.architectures[0]` names an embedding export (including the two
/// flag-gated decoders: `Gemma3TextModel` with `use_bidirectional_attention`
/// and `Ministral3Model` with `is_causal: false`).
fn has_embedding_architecture(config: &Value) -> bool {
    let Some(arch) = first_architecture(config) else {
        return false;
    };
    if EMBEDDING_ARCHITECTURES.contains(&arch) {
        return true;
    }
    let flag = |key: &str| config.get(key).and_then(Value::as_bool);
    match arch {
        "Gemma3TextModel" => flag("use_bidirectional_attention") == Some(true),
        "Ministral3Model" => flag("is_causal") == Some(false),
        _ => false,
    }
}

/// `<model_dir>/modules.json` lists a sentence-transformers module whose
/// `type` ends with `.Pooling`. A file whose only extra module is
/// `1_LogitScore` (Qwen3-VL-Reranker) does not qualify.
fn modules_json_has_pooling(model_path: &Path) -> bool {
    let Ok(raw) = std::fs::read_to_string(model_path.join("modules.json")) else {
        return false;
    };
    let Ok(modules) = serde_json::from_str::<Value>(&raw) else {
        return false;
    };
    modules
        .as_array()
        .map(|entries| {
            entries.iter().any(|entry| {
                entry
                    .get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|ty| ty.ends_with(".Pooling"))
            })
        })
        .unwrap_or(false)
}

/// Map the `model_type` of a detected embedding layout to its family.
fn embedding_variant_for_model_type(model_type: &str) -> Option<ModelType> {
    Some(match model_type {
        "bert" => ModelType::Bert,
        "xlm-roberta" | "xlm_roberta" => ModelType::XlmRoberta,
        "modernbert" => ModelType::ModernBert,
        "siglip" | "siglip_text_model" => ModelType::SiglipText,
        "gemma3_text" | "gemma3" => ModelType::Gemma3Embedding,
        "qwen3" => ModelType::Qwen3Embedding,
        "qwen3_vl" => ModelType::Qwen3VLEmbedding,
        "lfm2" => ModelType::Lfm2Embedding,
        "ministral3" => ModelType::Ministral3Embedding,
        "llama" | "llama_bidirec" => ModelType::LlamaBidirec,
        "llama_nemotron_vl" => ModelType::LlamaNemotronVLEmbedding,
        "idefics3" => ModelType::ColIdefics3,
        "qwen2_5_vl" | "colqwen2" => ModelType::ColQwen25,
        _ => return None,
    })
}

/// Recognize an embedding checkpoint before the `model_type` dispatch.
///
/// Returns `Ok(Some(variant))` when the checkpoint is an embedding export:
/// its `model_type` is an encoder-only family, `config.architectures[0]` is
/// an embedding architecture, `modules.json` carries a `.Pooling` module, or
/// `1_Pooling/config.json` exists. A checkpoint whose `architectures[0]`
/// ends with `ForSequenceClassification` is a reranker, never an embedder.
/// `Ok(None)` means "not an embedding checkpoint, continue with the
/// generation dispatch"; `Err` means the layout says embedding but the
/// `model_type` has no embedding family, which is reported rather than
/// misrouted to a causal generator.
pub(crate) fn is_embedding_checkpoint(
    model_path: &Path,
    config: &Value,
) -> Result<Option<ModelType>> {
    let Some(model_type_raw) = config.get("model_type").and_then(Value::as_str) else {
        return Ok(None);
    };
    let model_type = model_type_raw.to_ascii_lowercase();

    if first_architecture(config).is_some_and(|arch| arch.ends_with("ForSequenceClassification")) {
        return Ok(None);
    }

    let encoder_only = ENCODER_ONLY_MODEL_TYPES.contains(&model_type.as_str());
    let layout_says_embedding = encoder_only
        || has_embedding_architecture(config)
        || modules_json_has_pooling(model_path)
        || model_path.join("1_Pooling").join("config.json").exists();
    if !layout_says_embedding {
        return Ok(None);
    }

    match embedding_variant_for_model_type(&model_type) {
        Some(variant) => Ok(Some(variant)),
        None => Err(anyhow::anyhow!(
            "{} is an embedding checkpoint (sentence-transformers pooling layout or embedding \
             architecture), but model_type `{model_type_raw}` has no embedding family in \
             mlxcel; see docs/embeddings.md for the supported families",
            model_path.display()
        )),
    }
}

/// Detect model type from config.json
pub fn get_model_type(model_path: &Path) -> Result<ModelType> {
    let config_path = model_path.join("config.json");
    let config_str = std::fs::read_to_string(config_path)?;
    let config_str = sanitize_config_json(&config_str);
    let v: serde_json::Value = serde_json::from_str(&config_str)?;

    // A DFlash speculative drafter is structurally not a standalone model, but
    // it declares an ordinary `"model_type": "qwen3"`, so it would otherwise
    // fall through to the Qwen 3 arm below (#1168). Reject it before any
    // `model_type` dispatch runs. The discriminator is structural (a
    // `dflash_config` block and/or the `DFlashDraftModel` architecture) rather
    // than the resolved `DrafterKind`, because `DEFAULT_DRAFTER_KIND` is
    // `Dflash`: keying on the resolved kind would also reject an ordinary small
    // full model used as a classic drafter, which loads and runs fine.
    if is_dflash_drafter_config(&v) {
        return Err(dflash_drafter_not_standalone_error(model_path));
    }

    // Kokoro TTS checkpoints carry no top-level `model_type`, so detect them by
    // architecture signal (the `istftnet` config block or the canonical weight
    // filename) before the `model_type`-based dispatch below would error.
    if super::kokoro::is_kokoro_checkpoint(model_path, &v) {
        return Ok(ModelType::Kokoro);
    }

    // Embedding exports reuse generator `model_type`s (`qwen3` for
    // Qwen3-Embedding, `gemma3_text` for EmbeddingGemma), so the layout and
    // architecture rules must run before the `model_type` match below.
    if let Some(embedding) = is_embedding_checkpoint(model_path, &v)? {
        return Ok(embedding);
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
        // LLaDA-2 MoE (masked-diffusion LM with a DeepSeek-V3-style MoE FFN).
        // Generates by iterative block-wise unmasking rather than autoregressive
        // decode; served on the shared diffusion worker loop.
        "llada2_moe" => Ok(ModelType::Llada2Moe),
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
        "phi" | "phi-msft" => Ok(detect_phi_model_type(&v)),
        "phi3" => Ok(ModelType::Phi3),
        "phi4mm" => Ok(ModelType::Phi4MMVLM),
        "phi4-siglip" => Ok(ModelType::Phi4SigLipVLM),
        "phi3_v" => Ok(ModelType::Phi3VLM),
        "phi3small" => Ok(ModelType::Phi3Small),
        "phimoe" => Ok(ModelType::PhiMoe),
        "minimax" => Ok(ModelType::MiniMax),
        "minimax_m3" => Ok(ModelType::MiniMaxM3),
        "minimax_m3_vl" => Ok(ModelType::MiniMaxM3VL),
        "muse_glimmer" => Ok(ModelType::MuseGlimmerVLM),
        "gpt_oss" => Ok(ModelType::GptOss),
        "mixtral" => Ok(ModelType::Mixtral),
        "dbrx" => Ok(ModelType::Dbrx),
        "olmoe" => Ok(ModelType::OLMoE),
        "deepseek" => Ok(ModelType::DeepSeek),
        "deepseek_v2" => Ok(ModelType::DeepSeekV2),
        "deepseek_v3" => Ok(ModelType::DeepSeekV3),
        "deepseek_v32" | "deepseek_v3.2" => Ok(ModelType::DeepSeekV32),
        "dots1" => Ok(ModelType::Dots1),
        "cohere" => Ok(ModelType::Cohere),
        "cohere2" => Ok(ModelType::Cohere2),
        "cohere2_moe" => Ok(ModelType::Cohere2Moe),
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
        "ernie4_5_moe_vl" | "ernie4.5_moe_vl" => Ok(ModelType::Ernie45MoeVLM),
        "hunyuan_v1_dense" | "hunyuan_dense" => Ok(ModelType::HunyuanV1Dense),
        "hunyuan_vl" => Ok(ModelType::HunyuanVLM),
        "hunyuan" => Ok(detect_hunyuan_model_type(&v)),
        "mimo" => Ok(ModelType::MiMo),
        "bailing_moe" => Ok(ModelType::BailingMoe),
        "bailing_moe_linear" => Ok(ModelType::BailingMoeLinear),
        "afmoe" => Ok(ModelType::Afmoe),
        // `Kwai-Klear/Klear-46B-A2.5B-Instruct` declares the CAPITALIZED
        // `"Klear"`. It matches this lowercase arm only because
        // `model_type_raw` is lowercased above; mlx-lm, which does not
        // normalize, has to ship `Klear.py` and a byte-identical `klear.py` to
        // cover both spellings. Without that normalization this arm would miss
        // every published checkpoint.
        "klear" => Ok(ModelType::Klear),
        "apertus" => Ok(ModelType::Apertus),
        "seed_oss" => Ok(ModelType::SeedOss),
        "granite" => Ok(ModelType::Granite),
        "exaone" => Ok(ModelType::ExaOne),
        "exaone4" => Ok(ModelType::ExaOne4),
        "exaone_moe" => Ok(ModelType::ExaOneMoe),
        "olmo" => Ok(ModelType::Olmo),
        "olmo2" => Ok(ModelType::Olmo2),
        "olmo3" => Ok(ModelType::Olmo3),
        "openelm" => Ok(ModelType::OpenElm),
        "gpt2" => Ok(ModelType::Gpt2),
        "gpt_bigcode" => Ok(ModelType::GptBigCode),
        "gpt_neox" => Ok(ModelType::GptNeoX),
        "helium" => Ok(ModelType::Helium),
        "telechat3" => Ok(ModelType::TeleChat3),
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
        // LocateAnything: MoonViT tower + MLP connector + Qwen2 text decoder.
        // The text sub-config also says "qwen2", so this arm must win at the
        // top level or the grounding VLM would load as a text-only Qwen2.
        "locateanything" => Ok(ModelType::LocateAnythingVLM),
        "longcat_flash" => Ok(ModelType::LongcatFlash),
        "longcat_flash_ngram" => Ok(ModelType::LongcatFlashNgram),
        "step3p5" => Ok(ModelType::Step3p5),
        "step3p7" => Ok(ModelType::Step3p7),
        "recurrent_gemma" | "griffin" => Ok(ModelType::RecurrentGemma),
        "qwen2_vl" => Ok(ModelType::Qwen2VL),
        "qwen2_5_vl" => Ok(ModelType::Qwen25VL),
        "qwen3_vl" => Ok(ModelType::Qwen3VL),
        "qwen3_vl_moe" => Ok(ModelType::Qwen3VLMoe),
        "qwen3_omni_moe" => Ok(ModelType::Qwen3OmniMoe),
        "paddleocr_vl" => Ok(ModelType::PaddleOcrVL),
        "dots_ocr" => Ok(ModelType::DotsOcrVL),
        "falcon_ocr" => Ok(ModelType::FalconOcrVL),
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
        "unlimited-ocr" | "unlimited_ocr" => Ok(ModelType::UnlimitedOcrVLM),
        "deepseek_vl_v2" | "deepseek_vl2" => Ok(ModelType::DeepSeekVL2),
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
        "fastvlm" | "llava_qwen2" => Ok(ModelType::FastVLM),
        "aya_vision" => Ok(ModelType::AyaVisionVLM),
        "paligemma" => Ok(ModelType::PaliGemmaVLM),
        "pixtral" => Ok(ModelType::PixtralVLM),
        // The released checkpoints (and both sub-configs) spell this `jvlm`;
        // `jina_vlm` is only the upstream mlx-vlm module name and is accepted
        // as an alias so a hand-edited config still routes.
        "jvlm" | "jina_vlm" => Ok(ModelType::JinaVLM),
        "molmo" => Ok(ModelType::MolmoVLM),
        "molmo2" => Ok(ModelType::Molmo2VLM),
        "molmo_point" => Ok(ModelType::MolmoPointVLM),
        // Florence-2 (DaViT tower + BART encoder-decoder text stack).
        "florence2" => Ok(ModelType::Florence2VLM),
        // Speech-to-text (encoder-decoder ASR).
        "whisper" => Ok(ModelType::Whisper),
        _ => Err(anyhow::anyhow!(
            "Unsupported model type: {}",
            model_type_raw
        )),
    }
}
