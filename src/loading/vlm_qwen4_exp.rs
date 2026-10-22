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

//! Qwen3.8-Flash-Next (`qwen4_exp`) loader.

use std::path::Path;

use anyhow::Result;
use mlxcel_core::weights::WeightMap;
use serde_json::Value;

use crate::LoadedModel;
use crate::models;
use crate::vision;

use super::qwen3_vision::load_qwen3_vl_vision_encoder;
use super::{
    inherit_qwen_vision_quantization, load_vlm_weights_common, parse_required_vlm_subconfig,
    qwen_vl_processor_with_norm, qwen35_vl_token_ids, read_sanitized_vlm_config,
};

/// Upstream's `VisionConfig.__post_init__` refuses deepstack features: the
/// text model has nowhere to inject them.
fn reject_deepstack(full_config: &Value) -> Result<()> {
    let indexes = full_config
        .get("vision_config")
        .and_then(|vision| vision.get("deepstack_visual_indexes"));
    match indexes {
        None | Some(Value::Null) => Ok(()),
        Some(Value::Array(items)) if items.is_empty() => Ok(()),
        Some(other) => Err(anyhow::anyhow!(
            "Qwen4-Exp does not use deepstack visual features, but \
             vision_config.deepstack_visual_indexes={other} was set"
        )),
    }
}

/// The merger's output rows are the width of the features scattered over the
/// placeholder embeddings. Any width but the text `hidden_size` makes that
/// scatter throw inside MLX on the first image or video request, which aborts
/// the process; a missing merger is left to the encoder loader to report.
fn check_merger_width(vision: &WeightMap, hidden_size: usize) -> Result<()> {
    const MERGER_OUTPUT: &str = "vision_tower.merger.linear_fc2.weight";
    let Some(weight) = vision.get(MERGER_OUTPUT) else {
        return Ok(());
    };
    let shape = mlxcel_core::array_shape(weight);
    if shape.first().map(|&rows| rows as usize) != Some(hidden_size) {
        anyhow::bail!(
            "Qwen4-Exp {MERGER_OUTPUT} has shape {shape:?}: the vision features must be \
             text_config.hidden_size = {hidden_size} wide"
        );
    }
    Ok(())
}

/// OptiQ's Qwen4 converter folds `1 + weight` into these stored norms.
/// Its `optiq_vision` metadata identifies the converted checkpoint; ordinary
/// HF and mlx-vlm weights remain zero-centered. Undo the storage conversion
/// in float32 so the model reconstructs the stored scale without bf16 loss.
/// `linear_attn.norm` is a conventional gated norm and must stay untouched.
fn restore_optiq_zero_centered_norms(weights: &mut WeightMap, config: &Value) {
    if !config.get("optiq_vision").is_some_and(Value::is_object) {
        return;
    }
    const FOLDED: &[&str] = &[
        "q_layernorm.weight",
        "k_layernorm.weight",
        "q_norm.weight",
        "k_norm.weight",
        "hc_norm.weight",
        "norm_key.weight",
        "norm_query.weight",
        "norm_conv.weight",
    ];
    for (name, weight) in weights.iter_mut() {
        if name.starts_with("model.") && FOLDED.iter().any(|suffix| name.ends_with(suffix)) {
            let scale = mlxcel_core::astype(weight, mlxcel_core::dtype::FLOAT32);
            *weight = mlxcel_core::subtract(
                &scale,
                &mlxcel_core::full_f32(&[], 1.0, mlxcel_core::dtype::FLOAT32),
            );
        }
    }
}

pub(crate) fn load_qwen4_exp_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::encoders::qwen3_vl::Qwen3VLVisionConfig;

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;
    let text_config = models::qwen4_exp::text_config_from_full(&full_config)
        .map_err(|error| anyhow::anyhow!(error))?;
    reject_deepstack(&full_config)?;
    let mut vision_config: Qwen3VLVisionConfig =
        parse_required_vlm_subconfig(&full_config, "vision_config", "Qwen4-Exp vision config")?;
    inherit_qwen_vision_quantization(&mut vision_config, &full_config);

    let weights = load_vlm_weights_common(model_path, None)?;
    let models::qwen4_exp::SanitizedWeights {
        mut text,
        mut vision,
    } = models::qwen4_exp::sanitize_weights(weights, &text_config)
        .map_err(|error| anyhow::anyhow!(error))?;
    restore_optiq_zero_centered_norms(&mut text, &full_config);
    let text_model = models::Qwen4ExpModel::from_weights(&text, &text_config)
        .map_err(|error| anyhow::anyhow!("Failed to load Qwen4-Exp text model: {error}"))?;
    drop(text);

    check_merger_width(&vision, text_config.raw.hidden_size)?;
    let vision_encoder = load_qwen3_vl_vision_encoder(&mut vision, &vision_config, "vision_tower")
        .map_err(|error| anyhow::anyhow!("Failed to load Qwen4-Exp vision encoder: {error}"))?;
    let processor =
        qwen_vl_processor_with_norm(model_path, &vision_config, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])?;
    let token_ids = qwen35_vl_token_ids(&full_config, text_config.raw.vocab_size)?;

    Ok(LoadedModel::Qwen4ExpVLM(vision::Qwen4ExpVLModel {
        text_model,
        vision_encoder,
        processor,
        image_token_id: token_ids.image_token_id,
        video_token_id: token_ids.video_token_id,
        vision_start_token_id: token_ids.vision_start_token_id,
        spatial_merge_size: vision_config.spatial_merge_size,
    }))
}

#[cfg(test)]
#[path = "vlm_qwen4_exp_tests.rs"]
mod tests;
