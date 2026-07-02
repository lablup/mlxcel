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

//! Llama 3.2 Vision (`mllama`) VLM loader.
//!
//! Checkpoint layout (Llama-3.2-11B-Vision):
//! - `language_model.model.*` / `language_model.lm_head.weight` — the Llama-3
//!   backbone with interleaved gated cross-attention layers.
//! - `vision_tower.*` — the tiled ViT tower (patch/class embeds, gated
//!   positional + pre/post tile embeddings, local + gated global transformers).
//! - `multi_modal_projector.{weight,bias}` — `vision_output_dim -> hidden`.
//!
//! On a quantized checkpoint (e.g. the `-4bit` release) the tower projections,
//! MLPs, the tile/positional embeddings, and the projector all carry
//! `.scales`/`.biases`; the loaders below auto-detect this via the shared
//! `Unified{Linear,Embedding}` primitives.
//!
//! Precision follows the standard Apple Silicon policy: bf16 tensors are
//! widened to f16 (no Apple GPU has a native bf16 ALU), while quantization
//! `.scales` / `.biases` stay bf16 for the `quantized_matmul` path.

use anyhow::Result;
use serde_json::Value;
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::models::mllama::MllamaConfig;
use crate::vision;

use super::{load_vlm_weights_common, read_sanitized_vlm_config, strip_language_model_prefix};

/// Llama 3.2 default stop tokens (`<|end_of_text|>`, `<|eom_id|>`, `<|eot_id|>`).
const DEFAULT_EOS_TOKEN_IDS: [i32; 3] = [128001, 128008, 128009];

fn resolve_eos_token_ids(full_config: &Value) -> Vec<i32> {
    let from_value = |v: &Value| -> Vec<i32> {
        match v {
            Value::Number(n) => n.as_i64().map(|id| vec![id as i32]).unwrap_or_default(),
            Value::Array(arr) => arr
                .iter()
                .filter_map(|x| x.as_i64().map(|n| n as i32))
                .collect(),
            _ => Vec::new(),
        }
    };

    if let Some(ids) = full_config
        .get("text_config")
        .and_then(|tc| tc.get("eos_token_id"))
        .map(from_value)
        .filter(|ids| !ids.is_empty())
    {
        return ids;
    }
    if let Some(ids) = full_config
        .get("eos_token_id")
        .map(from_value)
        .filter(|ids| !ids.is_empty())
    {
        return ids;
    }
    DEFAULT_EOS_TOKEN_IDS.to_vec()
}

/// Inherit a top-level `quantization` block into `text_config` and
/// `vision_config` when they do not carry their own. mllama quantizes the
/// whole model together (text backbone, vision tower, and projector), but the
/// exported `config.json` records the block only at the top level.
fn inherit_quantization(config_value: &mut Value, full_config: &Value) {
    let Some(q) = full_config.get("quantization") else {
        return;
    };
    for sub in ["text_config", "vision_config"] {
        let needs_quant = config_value
            .get(sub)
            .and_then(|c| c.get("quantization"))
            .is_none();
        if needs_quant && let Some(obj) = config_value.get_mut(sub).and_then(Value::as_object_mut) {
            obj.insert("quantization".to_string(), q.clone());
        }
    }
}

/// Load a Llama 3.2 Vision (`mllama`) VLM.
pub(crate) fn load_mllama_vlm(model_path: &Path) -> Result<LoadedModel> {
    use models::mllama::MllamaTextModel;
    use vision::MllamaVLModel;
    use vision::encoders::mllama::MllamaVisionModel;
    use vision::processors::mllama::MllamaImageProcessor;

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    let mut config_value = full_config.clone();
    inherit_quantization(&mut config_value, &full_config);
    let config: MllamaConfig = serde_json::from_value(config_value)
        .map_err(|e| anyhow::anyhow!("Failed to parse mllama config: {}", e))?;

    // Load all weights, rewrite `language_model.*` -> `model.*` / `lm_head.*`,
    // then widen bf16 -> f16 (keeping quantization scales/biases as bf16).
    let mut weights = strip_language_model_prefix(load_vlm_weights_common(model_path, None)?);
    let hw = mlxcel_core::hardware::get_hardware();
    if hw.silicon_gen != mlxcel_core::hardware::AppleSiliconGen::Unknown {
        let had_bf16 = models::convert_bf16_weights_with_keep(&mut weights, |key| {
            key.ends_with(".scales") || key.ends_with(".biases")
        });
        if had_bf16 {
            models::warn_bf16_precision();
        }
    }

    let text_model = MllamaTextModel::from_weights(&weights, &config.text_config)
        .map_err(|e| anyhow::anyhow!("Failed to load mllama text backbone: {}", e))?;
    // The real checkpoint stores the tower under `vision_tower.*` (HF/MLX
    // `MllamaForConditionalGeneration`), not `vision_model.*`.
    let vision_tower =
        MllamaVisionModel::from_weights(&weights, &config.vision_config, "vision_tower")
            .map_err(|e| anyhow::anyhow!("Failed to load mllama vision tower: {}", e))?;
    let projector = MllamaVLModel::load_projector(
        &weights,
        config.vision_config.quant_group_size(),
        config.vision_config.quant_bits(),
    )
    .map_err(|e| anyhow::anyhow!("Failed to load mllama multi_modal_projector: {}", e))?;

    let processor = MllamaImageProcessor::new(
        config.vision_config.image_size,
        config.vision_config.max_num_tiles,
    );
    let eos_token_ids = resolve_eos_token_ids(&full_config);

    let model = MllamaVLModel::from_parts(
        text_model,
        vision_tower,
        projector,
        processor,
        config,
        eos_token_ids,
    );
    Ok(LoadedModel::MllamaVLM(model))
}
