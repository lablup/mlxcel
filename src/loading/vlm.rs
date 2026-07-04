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

//! Shared VLM loading router and helper utilities.
//!
//! This module owns the common config/weight helpers used by family-specific
//! VLM loaders. Real construction logic stays in sibling `vlm_<family>.rs`
//! modules so this file can remain a router plus a small toolbox.
//!
//! Families:
//! - `vlm_gemma.rs`: Gemma3 / Gemma3n
//! - `vlm_llava.rs`: LLaVA / Bunny
//! - `vlm_pixtral.rs`: Pixtral / Mistral3
//! - `vlm_qwen.rs`: Qwen2 / 2.5 / 3 / 3.5-VL
//! - `vlm_siglip.rs`: Aya Vision / PaliGemma
//! - `vlm_special.rs`: Llama4 / MiniCPM-o / Phi4MM / Phi4-SigLIP / Phi3V / Molmo2

use anyhow::Result;
use mlxcel_core::weights::WeightMap;
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::vision;
use models::sanitize_config_json;

#[path = "vlm_deepseekocr.rs"]
mod deepseekocr;
#[path = "vlm_gemma.rs"]
mod gemma;
#[path = "vlm_gemma_unified.rs"]
mod gemma_unified;
#[path = "vlm_granite4_vision.rs"]
mod granite4_vision;
#[path = "vlm_granite_vision.rs"]
mod granite_vision;
#[path = "vlm_idefics2.rs"]
mod idefics2;
#[path = "vlm_internvl.rs"]
mod internvl;
#[path = "vlm_kimi_vl.rs"]
mod kimi_vl_loader;
#[path = "vlm_lfm2_vl.rs"]
mod lfm2_vl;
#[path = "vlm_llava.rs"]
mod llava;
#[path = "vlm_mllama.rs"]
mod mllama;
#[path = "vlm_nemotron_h_nano_omni.rs"]
mod nemotron_h_nano_omni;
#[path = "vlm_paddleocr.rs"]
mod paddleocr;
#[path = "vlm_pixtral.rs"]
mod pixtral;
#[path = "vlm_qwen.rs"]
mod qwen;
#[path = "vlm_siglip.rs"]
mod siglip;
#[path = "vlm_smolvlm.rs"]
mod smolvlm;
#[path = "vlm_special.rs"]
mod special;
#[path = "vlm_youtu_vl.rs"]
mod youtu_vl_loader;

pub(crate) use deepseekocr::load_deepseekocr_vlm;
pub(crate) use gemma::{load_gemma3_vlm, load_gemma3n_vlm, load_gemma4_vlm};
pub(crate) use gemma_unified::load_gemma4_unified;
pub(crate) use granite_vision::load_granite_vision_vlm;
pub(crate) use granite4_vision::load_granite4_vision_vlm;
pub(crate) use idefics2::load_idefics2_vlm;
pub(crate) use internvl::load_internvl_vlm;
pub(crate) use kimi_vl_loader::load_kimi_vl_vlm;
pub(crate) use lfm2_vl::load_lfm2_vl;
pub(crate) use llava::{load_llava_bunny_vlm, load_llava_vlm};
pub(crate) use mllama::load_mllama_vlm;
pub(crate) use nemotron_h_nano_omni::load_nemotron_h_nano_omni_vlm;
pub(crate) use paddleocr::load_paddleocr_vl;
pub(crate) use pixtral::{load_mistral3_vlm, load_pixtral_vlm};
pub(crate) use qwen::{
    load_glm_ocr, load_glm4v, load_glm4v_moe, load_qwen2_5_vl, load_qwen2_vl, load_qwen3_5_moe_vlm,
    load_qwen3_5_vlm, load_qwen3_vl, load_qwen3_vl_moe,
};
pub(crate) use siglip::{load_aya_vision_vlm, load_paligemma_vlm};
pub(crate) use smolvlm::load_smolvlm_vlm;
pub(crate) use special::{
    load_llama4_vlm, load_minicpmo_vlm, load_minicpmv4_6_vlm, load_molmo_point_vlm, load_molmo_vlm,
    load_molmo2_vlm, load_moondream2_vlm, load_moondream3_vlm, load_phi3_vlm, load_phi4_siglip_vlm,
    load_phi4mm_vlm,
};
pub(crate) use youtu_vl_loader::load_youtu_vl_vlm;

fn read_sanitized_vlm_config(model_path: &Path) -> Result<(String, Value)> {
    let config_path = model_path.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| anyhow::anyhow!("Failed to read config.json: {}", e))?;
    let config_str = sanitize_config_json(&config_str);
    let full_config: Value = parse_vlm_config(&config_str, "config")?;
    Ok((config_str, full_config))
}

fn parse_vlm_config<T: DeserializeOwned>(config_str: &str, label: &str) -> Result<T> {
    serde_json::from_str(config_str)
        .map_err(|e| anyhow::anyhow!("Failed to parse {}: {}", label, e))
}

fn parse_required_vlm_subconfig<T: DeserializeOwned>(
    full_config: &Value,
    key: &str,
    label: &str,
) -> Result<T> {
    let value = full_config
        .get(key)
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Missing {} in config.json", key))?;
    serde_json::from_value(value).map_err(|e| anyhow::anyhow!("Failed to parse {}: {}", label, e))
}

pub(super) fn require_object_mut<'a>(
    value: &'a mut Value,
    label: &str,
) -> Result<&'a mut serde_json::Map<String, Value>> {
    value
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("Expected {} to be a JSON object", label))
}

/// Consolidated VLM weight load entry point with optional
/// Axis A "weight-load surgery" hook.
///
/// All `src/loading/vlm_*.rs` family loaders funnel through this
/// function with `transform = None` by default. It reads safetensors,
/// applies Apple Silicon bf16 → f16 conversion for non-quantized
/// models, then invokes the optional
/// [`mlxcel_core::weights::WeightTransform`] hook.
///
/// When `transform` is `None`, behavior is bit-exact identical to the
/// pre-refactor `load_vlm_weights` path: no observable change to any
/// VLM load flow.
///
/// The bf16 → f16 conversion happens **before** the transform so
/// transforms operate on the final-precision tensors (consistent with
/// the text path in [`crate::models::load_text_weights`], where
/// transforms also see post-sanitization, pre-precision weights — for
/// VLMs the basic sanitization step is the family-specific weight
/// remap which runs in the caller after we return). Surgery operations
/// inspecting dtype therefore see the dtype the model graph will see.
///
/// ## Active-pipeline fallback (— A4)
///
/// When the explicit `transform` argument is `None` *and* the
/// `surgery` feature is enabled *and* the CLI has installed an active
/// pipeline via `crate::surgery::set_active_pipeline(...)`, the
/// installed pipeline is used as the transform. This is the integration
/// glue for `mlxcel generate --surgery foo.yaml` and
/// `mlxcel serve --surgery foo.yaml` on the VLM load path.
///
/// When no surgery is active (the default), the snapshot lookup
/// returns `None` and the call site falls through the existing
/// `if let Some(transform) = transform` arm exactly as before.
///
/// Used by: all VLM family loaders in src/loading/vlm_*.rs
/// (gemma, llava, nemotron_h_nano_omni, pixtral, qwen, siglip,
/// special, youtu_vl)
pub(crate) fn load_vlm_weights_common(
    model_path: &Path,
    transform: Option<&dyn mlxcel_core::weights::WeightTransform>,
) -> Result<WeightMap> {
    let mut weights = mlxcel_core::weights::load_weights_from_dir(model_path)
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    // On Apple Silicon, convert bf16 → f16 for performance (no Apple GPU has
    // native bf16 ALU).  Quantized models keep bf16 scales/biases as-is.
    let hw = mlxcel_core::hardware::get_hardware();
    if hw.silicon_gen != mlxcel_core::hardware::AppleSiliconGen::Unknown {
        let is_quantized = is_model_quantized(model_path);
        if !is_quantized {
            let had_bf16 = if is_gemma3n_model(model_path) {
                crate::models::convert_bf16_weights_with_keep(
                    &mut weights,
                    crate::models::gemma3n_language_mlp_bf16_key,
                )
            } else {
                crate::models::convert_bf16_weights(&mut weights)
            };
            if had_bf16 {
                crate::models::warn_bf16_precision();
            }
        }
    }

    // Axis A weight-load surgery hook. Invoked with the
    // parsed config.json so transforms can inspect quantization or
    // structural metadata. When both `transform` and the active
    // surgery slot are absent, the hook is bypassed entirely and the
    // load path matches the earlier baseline bit-for-bit.
    //
    // Resolution order (A4) mirrors `load_text_weights`:
    //   1. Explicit `transform` argument (test fixtures, programmatic).
    //   2. CLI-installed active pipeline (--surgery flag).
    //   3. Baseline — no transform applied.
    #[cfg(feature = "surgery")]
    let active_pipeline = transform
        .is_none()
        .then(crate::surgery::snapshot_active_pipeline)
        .flatten();
    let resolved_transform: Option<&dyn mlxcel_core::weights::WeightTransform> = match transform {
        Some(t) => Some(t),
        None => {
            #[cfg(feature = "surgery")]
            {
                active_pipeline
                    .as_deref()
                    .map(|p: &crate::surgery::SurgeryPipeline| {
                        p as &dyn mlxcel_core::weights::WeightTransform
                    })
            }
            #[cfg(not(feature = "surgery"))]
            {
                None
            }
        }
    };
    if let Some(transform) = resolved_transform {
        let cfg = std::fs::read_to_string(model_path.join("config.json"))
            .ok()
            .map(|s| crate::models::sanitize_config_json(&s))
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .unwrap_or(serde_json::Value::Null);
        transform
            .apply(&mut weights, &cfg)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
    }

    Ok(weights)
}

/// Check if a model is quantized by looking for `.scales` keys in weights
/// or a `quantization` field in config.json.
fn is_model_quantized(model_path: &Path) -> bool {
    let config_path = model_path.join("config.json");
    if let Ok(config_str) = std::fs::read_to_string(&config_path) {
        let config_str = crate::models::sanitize_config_json(&config_str);
        if let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_str) {
            if config.get("quantization").is_some() {
                return true;
            }
            if let Some(tc) = config.get("text_config")
                && tc.get("quantization").is_some()
            {
                return true;
            }
        }
    }
    false
}

fn is_gemma3n_model(model_path: &Path) -> bool {
    let Ok((_, config)) = read_sanitized_vlm_config(model_path) else {
        return false;
    };
    config
        .get("model_type")
        .and_then(Value::as_str)
        .is_some_and(|model_type| model_type == "gemma3n")
        || config
            .get("text_config")
            .and_then(|text_config| text_config.get("model_type"))
            .and_then(Value::as_str)
            .is_some_and(|model_type| model_type == "gemma3n" || model_type == "gemma3n_text")
}

fn read_optional_model_json(model_path: &Path, file_name: &str) -> Option<Value> {
    let path = model_path.join(file_name);
    let json = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&json).ok()
}

pub(super) fn strip_language_model_prefix(raw_weights: WeightMap) -> WeightMap {
    let mut weights = WeightMap::new();
    for (key, value) in raw_weights {
        let new_key = if let Some(stripped) = key.strip_prefix("language_model.") {
            stripped.to_string()
        } else {
            key
        };
        weights.insert(new_key, value);
    }
    weights
}

trait QwenVisionConfigExt {
    fn quant_group_size(&self) -> i32;
    fn quant_bits(&self) -> i32;
    fn set_quant_group_size(&mut self, value: i32);
    fn set_quant_bits(&mut self, value: i32);
    fn patch_size(&self) -> usize;
    fn temporal_patch_size(&self) -> usize;
    fn spatial_merge_size(&self) -> usize;
}

impl QwenVisionConfigExt for vision::encoders::qwen2_vl::Qwen2VLVisionConfig {
    fn quant_group_size(&self) -> i32 {
        self.quant_group_size
    }

    fn quant_bits(&self) -> i32 {
        self.quant_bits
    }

    fn set_quant_group_size(&mut self, value: i32) {
        self.quant_group_size = value;
    }

    fn set_quant_bits(&mut self, value: i32) {
        self.quant_bits = value;
    }

    fn patch_size(&self) -> usize {
        self.patch_size
    }

    fn temporal_patch_size(&self) -> usize {
        self.temporal_patch_size
    }

    fn spatial_merge_size(&self) -> usize {
        self.spatial_merge_size
    }
}

impl QwenVisionConfigExt for vision::encoders::qwen2_5_vl::Qwen25VLVisionConfig {
    fn quant_group_size(&self) -> i32 {
        self.quant_group_size
    }

    fn quant_bits(&self) -> i32 {
        self.quant_bits
    }

    fn set_quant_group_size(&mut self, value: i32) {
        self.quant_group_size = value;
    }

    fn set_quant_bits(&mut self, value: i32) {
        self.quant_bits = value;
    }

    fn patch_size(&self) -> usize {
        self.patch_size
    }

    fn temporal_patch_size(&self) -> usize {
        self.temporal_patch_size
    }

    fn spatial_merge_size(&self) -> usize {
        self.spatial_merge_size
    }
}

impl QwenVisionConfigExt for vision::encoders::qwen3_vl::Qwen3VLVisionConfig {
    fn quant_group_size(&self) -> i32 {
        self.quant_group_size
    }

    fn quant_bits(&self) -> i32 {
        self.quant_bits
    }

    fn set_quant_group_size(&mut self, value: i32) {
        self.quant_group_size = value;
    }

    fn set_quant_bits(&mut self, value: i32) {
        self.quant_bits = value;
    }

    fn patch_size(&self) -> usize {
        self.patch_size
    }

    fn temporal_patch_size(&self) -> usize {
        self.temporal_patch_size
    }

    fn spatial_merge_size(&self) -> usize {
        self.spatial_merge_size
    }
}

impl QwenVisionConfigExt for vision::encoders::glm4v::Glm4vVisionConfig {
    fn quant_group_size(&self) -> i32 {
        self.quant_group_size
    }

    fn quant_bits(&self) -> i32 {
        self.quant_bits
    }

    fn set_quant_group_size(&mut self, value: i32) {
        self.quant_group_size = value;
    }

    fn set_quant_bits(&mut self, value: i32) {
        self.quant_bits = value;
    }

    fn patch_size(&self) -> usize {
        self.patch_size
    }

    fn temporal_patch_size(&self) -> usize {
        self.temporal_patch_size
    }

    fn spatial_merge_size(&self) -> usize {
        self.spatial_merge_size
    }
}

fn inherit_qwen_vision_quantization<T: QwenVisionConfigExt>(
    vision_config: &mut T,
    full_config: &Value,
) {
    if (vision_config.quant_group_size() == 0 || vision_config.quant_bits() == 0)
        && let Some(q) = full_config.get("quantization")
    {
        vision_config.set_quant_group_size(
            q.get("group_size").and_then(|v| v.as_i64()).unwrap_or(64) as i32,
        );
        vision_config.set_quant_bits(q.get("bits").and_then(|v| v.as_i64()).unwrap_or(4) as i32);
    }
}

trait QwenTextQuantizationExt {
    fn has_quantization(&self) -> bool;
    fn set_quantization_from_value(&mut self, value: &Value);
}

impl QwenTextQuantizationExt for models::qwen3_vl::Qwen3VLConfig {
    fn has_quantization(&self) -> bool {
        self.quantization.is_some()
    }

    fn set_quantization_from_value(&mut self, value: &Value) {
        self.quantization = serde_json::from_value(value.clone()).ok();
    }
}

impl QwenTextQuantizationExt for models::qwen3_vl_moe::Qwen3VLMoeConfig {
    fn has_quantization(&self) -> bool {
        self.quantization.is_some()
    }

    fn set_quantization_from_value(&mut self, value: &Value) {
        self.quantization = serde_json::from_value(value.clone()).ok();
    }
}

impl QwenTextQuantizationExt for models::glm4v::Glm4vTextConfig {
    fn has_quantization(&self) -> bool {
        self.quantization.is_some()
    }

    fn set_quantization_from_value(&mut self, value: &Value) {
        self.quantization = serde_json::from_value(value.clone()).ok();
    }
}

fn inherit_qwen_text_quantization<T: QwenTextQuantizationExt>(
    text_config: &mut T,
    full_config: &Value,
) {
    if !text_config.has_quantization()
        && let Some(q) = full_config.get("quantization")
    {
        text_config.set_quantization_from_value(q);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QwenVisionTokenIds {
    image_token_id: i32,
    video_token_id: i32,
    vision_start_token_id: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Qwen35VlmVariant {
    Dense,
    Moe,
}

fn qwen_vl_token_ids(full_config: &Value, defaults: QwenVisionTokenIds) -> QwenVisionTokenIds {
    QwenVisionTokenIds {
        image_token_id: full_config
            .get("image_token_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(defaults.image_token_id as i64) as i32,
        video_token_id: full_config
            .get("video_token_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(defaults.video_token_id as i64) as i32,
        vision_start_token_id: full_config
            .get("vision_start_token_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(defaults.vision_start_token_id as i64) as i32,
    }
}

fn qwen35_vlm_token_defaults() -> QwenVisionTokenIds {
    QwenVisionTokenIds {
        image_token_id: 248056,
        video_token_id: 248057,
        vision_start_token_id: 248045,
    }
}

fn wrap_qwen35_vlm(vlm: vision::Qwen35VLModel, variant: Qwen35VlmVariant) -> LoadedModel {
    match variant {
        Qwen35VlmVariant::Dense => LoadedModel::Qwen35VLM(vlm),
        Qwen35VlmVariant::Moe => LoadedModel::Qwen35MoeVLM(vlm),
    }
}

fn qwen_vl_processor<T: QwenVisionConfigExt>(
    vision_config: &T,
) -> vision::processors::qwen2_vl::Qwen2VLProcessor {
    vision::processors::qwen2_vl::Qwen2VLProcessor::new(
        vision_config.patch_size(),
        vision_config.temporal_patch_size(),
        vision_config.spatial_merge_size(),
    )
}

fn qwen_vl_processor_with_norm<T: QwenVisionConfigExt>(
    vision_config: &T,
    mean: [f32; 3],
    std: [f32; 3],
) -> vision::processors::qwen2_vl::Qwen2VLProcessor {
    vision::processors::qwen2_vl::Qwen2VLProcessor::new_with_norm(
        vision_config.patch_size(),
        vision_config.temporal_patch_size(),
        vision_config.spatial_merge_size(),
        mean,
        std,
    )
}

fn rewrite_qwen3_vl_weight_key(key: String, moe_experts: bool) -> String {
    let new_key = if key.starts_with("model.language_model.") {
        key.replacen("model.language_model.", "model.", 1)
    } else if key.starts_with("model.visual.") {
        key.replacen("model.visual.", "vision_tower.", 1)
    } else if key.starts_with("language_model.") {
        key.replacen("language_model.", "", 1)
    } else {
        key
    };

    if moe_experts && new_key.contains(".mlp.experts.") {
        if let Some((_, after_experts)) = new_key.rsplit_once(".mlp.experts.") {
            if !after_experts.contains('.') {
                format!(
                    "{}.weight",
                    new_key.replace(".mlp.experts.", ".mlp.switch_mlp.")
                )
            } else {
                new_key.replace(".mlp.experts.", ".mlp.switch_mlp.")
            }
        } else {
            new_key
        }
    } else {
        new_key
    }
}

fn remap_qwen3_vl_weights(raw_weights: WeightMap, moe_experts: bool) -> WeightMap {
    let mut weights = WeightMap::new();
    for (key, value) in raw_weights {
        weights.insert(rewrite_qwen3_vl_weight_key(key, moe_experts), value);
    }
    weights
}

#[cfg(test)]
#[path = "vlm_tests.rs"]
mod tests;
