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

//! Gemma-family VLM loaders.
//!
//! Families:
//! - Gemma3 VLM
//! - Gemma3n VLM
//!
//! This file keeps Gemma-specific weight sanitation, metadata defaults, and
//! wrapper assembly out of the generic VLM router.

use anyhow::Result;
use mlxcel_core::weights::WeightMap;
use serde_json::Value;
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::vision;

use super::{
    load_vlm_weights_common, parse_required_vlm_subconfig, parse_vlm_config,
    read_optional_model_json, read_sanitized_vlm_config, strip_language_model_prefix,
};

struct Gemma3nMetadata {
    vision_hidden_size: usize,
    image_size: usize,
    image_token_id: i32,
    boi_token_id: i32,
    eoi_token_id: i32,
    vision_rms_eps: f32,
}

fn gemma3n_metadata(config: &Value) -> Gemma3nMetadata {
    Gemma3nMetadata {
        vision_hidden_size: config
            .get("vision_config")
            .and_then(|vc| vc.get("hidden_size"))
            .and_then(|v| v.as_u64())
            .unwrap_or(2048) as usize,
        image_size: config
            .get("vision_config")
            .and_then(|vc| vc.get("image_size"))
            .and_then(|v| v.as_u64())
            .unwrap_or(256) as usize,
        image_token_id: config
            .get("image_token_id")
            .or_else(|| config.get("image_token_index"))
            .and_then(|v| v.as_i64())
            .unwrap_or(262_145) as i32,
        boi_token_id: config
            .get("boi_token_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(255_999) as i32,
        eoi_token_id: config
            .get("eoi_token_id")
            .and_then(|v| v.as_i64())
            .unwrap_or(262_144) as i32,
        vision_rms_eps: config
            .get("vision_config")
            .and_then(|vc| vc.get("rms_norm_eps"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1e-6) as f32,
    }
}

fn gemma3n_needs_conv_transpose(raw_weights: &WeightMap) -> bool {
    raw_weights
        .get("model.vision_tower.timm_model.blocks.0.0.conv_exp.weight")
        .map(|w| {
            let shape = mlxcel_core::array_shape(w);
            shape.len() == 4 && shape[1] > shape[2]
        })
        .unwrap_or(false)
}

fn gemma3n_language_model_prefix(weights: &WeightMap) -> &'static str {
    if weights.contains_key("language_model.model.embed_tokens.weight") {
        "language_model.model"
    } else {
        "language_model"
    }
}

fn sanitize_gemma3n_weights(raw_weights: WeightMap) -> WeightMap {
    let needs_transpose = gemma3n_needs_conv_transpose(&raw_weights);
    // bf16→f16 conversion for M5 is handled by load_vlm_weights_common in vlm.rs.
    let mut weights = WeightMap::new();

    for (key, value) in raw_weights {
        let new_key = if let Some(stripped) = key.strip_prefix("model.") {
            stripped.to_string()
        } else {
            key
        };

        let value = if needs_transpose {
            let shape = mlxcel_core::array_shape(&value);
            if shape.len() == 4 {
                mlxcel_core::transpose_axes(&value, &[0, 2, 3, 1])
            } else {
                mlxcel_core::copy(&value)
            }
        } else {
            mlxcel_core::copy(&value)
        };

        weights.insert(new_key, value);
    }

    weights
}

/// Load a Gemma3 VLM model (text + vision tower + projector).
pub(crate) fn load_gemma3_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::config::VLMConfig;
    use vision::connectors::avg_pool::AvgPoolProjector;
    use vision::encoders::siglip::SigLipVisionModel;
    use vision::processors::siglip::SigLipProcessor;

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;
    let vlm_config: VLMConfig = serde_json::from_value(full_config.clone())
        .map_err(|e| anyhow::anyhow!("Failed to parse VLM config: {}", e))?;
    let text_config: models::gemma3::ModelArgs =
        serde_json::from_value(vlm_config.text_config.clone())
            .map_err(|e| anyhow::anyhow!("Failed to parse text_config: {}", e))?;

    let mut weights = strip_language_model_prefix(load_vlm_weights_common(model_path, None)?);
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    let text_model = models::Gemma3Model::from_weights(&weights, &text_config)
        .map_err(|e| anyhow::anyhow!("Failed to load text model: {}", e))?;
    let text_wrapper = models::Gemma3Wrapper::new(text_model);

    let vision_encoder = SigLipVisionModel::from_weights(
        &weights,
        &vlm_config.vision_config,
        "vision_tower.vision_model",
    )
    .map_err(|e| anyhow::anyhow!("Failed to load vision encoder: {}", e))?;

    let mm_tokens_per_image = vlm_config.get_mm_tokens_per_image();
    let connector = AvgPoolProjector::from_weights(
        &weights,
        "multi_modal_projector",
        vlm_config.vision_config.hidden_size,
        vlm_config.vision_config.image_size,
        vlm_config.vision_config.patch_size,
        mm_tokens_per_image,
        vlm_config.vision_config.layer_norm_eps,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load projector: {}", e))?;

    let processor = SigLipProcessor::new(vlm_config.vision_config.image_size);
    let vision_module = vision::VisionModule {
        encoder: Box::new(vision_encoder),
        connector: Box::new(connector),
        processor: Box::new(processor),
        image_token_id: vlm_config.image_token_index,
        pad_token_id: vlm_config.pad_token_id,
        hidden_size: if vlm_config.hidden_size > 0 {
            vlm_config.hidden_size
        } else {
            text_config.hidden_size
        },
        boi_token_id: vlm_config.boi_token_index,
        eoi_token_id: vlm_config.eoi_token_index,
        mm_tokens_per_image,
        merge_strategy: vision::MergeStrategy::Gemma3,
        has_bos: true,
        separator_token_id: None,
        suffix_tokens: Vec::new(),
        // Gemma3 VLM: Python Gemma3Processor wraps image tokens as:
        //   \n\n<start_of_image><image>x256<end_of_image>\n\n
        // Token 108 = "\n\n" in the Gemma3 vocabulary. However, the chat
        // template may already include surrounding \n tokens, so we only add
        // the extra \n\n wrapping when expanding BOI tokens from the template.
        block_prefix_tokens: vec![108],
        block_suffix_tokens: vec![108],
    };

    let vlm = vision::VisionLanguageModel {
        text_model: Box::new(LoadedModel::Gemma3(text_wrapper)),
        vision: vision_module,
    };

    Ok(LoadedModel::Gemma3VLM(vlm))
}

/// Load a Gemma3n VLM model.
pub(crate) fn load_gemma3n_vlm(model_path: &Path) -> Result<LoadedModel> {
    use vision::encoders::gemma3n::load_gemma3n_vision;
    use vision::processors::siglip::SigLipProcessor;

    let (config_str, full_config) = read_sanitized_vlm_config(model_path)?;
    let top_args: models::gemma3n::ModelArgs = parse_vlm_config(&config_str, "Gemma3n config")?;
    let text_config = top_args.text_args();
    let metadata = gemma3n_metadata(&full_config);

    let mut weights = sanitize_gemma3n_weights(load_vlm_weights_common(model_path, None)?);
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    let language_model = models::gemma3n::Gemma3nLanguageModel::from_weights(
        &weights,
        &text_config,
        gemma3n_language_model_prefix(&weights),
    )
    .map_err(|e| anyhow::anyhow!("Failed to load text model: {}", e))?;

    let text_model = models::Gemma3nModel {
        language_model,
        config: text_config.clone(),
    };

    let vision_tower = load_gemma3n_vision(&weights, "vision_tower.timm_model")
        .map_err(|e| anyhow::anyhow!("Failed to load vision tower: {}", e))?;

    let group_size = text_config
        .quantization
        .as_ref()
        .map(|q| q.group_size as i32)
        .unwrap_or(64);
    let bits = text_config
        .quantization
        .as_ref()
        .map(|q| q.bits as i32)
        .unwrap_or(4);

    let embed_vision = models::gemma3n::Gemma3nMultimodalEmbedder::from_weights(
        &weights,
        "embed_vision",
        metadata.vision_hidden_size,
        text_config.hidden_size,
        metadata.vision_rms_eps,
        group_size,
        bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load multimodal embedder: {}", e))?;

    let processor = SigLipProcessor::new_rescale_only(metadata.image_size);

    let vlm = vision::Gemma3nVLModel::new(
        text_model,
        vision_tower,
        embed_vision,
        processor,
        metadata.image_token_id,
        metadata.boi_token_id,
        metadata.eoi_token_id,
        metadata.vision_hidden_size,
    );

    Ok(LoadedModel::Gemma3nVLM(vlm))
}

/// Sanitize Gemma4 audio weights: transpose conv weights from PyTorch to MLX format.
fn sanitize_gemma4_audio_weights(weights: &mut mlxcel_core::weights::WeightMap) {
    let audio_keys: Vec<String> = weights
        .keys()
        .filter(|k| k.starts_with("audio_tower."))
        .cloned()
        .collect();

    for key in audio_keys {
        let shape = {
            let w = weights.get(&key).unwrap();
            mlxcel_core::array_shape(w)
        };

        // Conv2d: PyTorch [out, in, kH, kW] -> MLX [out, kH, kW, in].
        // Skip already-channel-last checkpoints to avoid double-converting the
        // shape (issue #428): the mlx-community gemma4 weights ship channel-last.
        if key.contains("subsample_conv_projection")
            && key.contains("conv.weight")
            && shape.len() == 4
            && !crate::loading::conv2d_weight_is_channel_last(&shape)
        {
            let w = weights.remove(&key).unwrap();
            let transposed = mlxcel_core::transpose_axes(&w, &[0, 2, 3, 1]);
            weights.insert(key, transposed);
        }
        // Conv1d depthwise: PyTorch [out, in, kW] -> MLX [out, kW, in].
        // The Conformer lconv1d is depthwise (in == 1), so the channel-last
        // form is [out, kW, 1]; skip it when already converted (issue #428).
        else if key.contains("depthwise_conv1d.weight")
            && shape.len() == 3
            && !crate::loading::conv1d_weight_is_channel_last(&shape)
        {
            let w = weights.remove(&key).unwrap();
            let transposed = mlxcel_core::transpose_axes(&w, &[0, 2, 1]);
            weights.insert(key, transposed);
        }
    }
}

/// Load a Gemma4 VLM model.
pub(crate) fn load_gemma4_vlm(model_path: &Path) -> Result<LoadedModel> {
    let (config_str, full_config) = read_sanitized_vlm_config(model_path)?;
    let args: models::gemma4::ModelArgs = parse_vlm_config(&config_str, "Gemma4 config")?;
    let (mut weights, weight_backing) = models::load_gemma4_vlm_weights_with_backing(model_path)
        .map_err(|e| anyhow::anyhow!("Failed to load Gemma4 VLM weights: {}", e))?;
    // Drop k_proj/v_proj/k_norm weight entries for KV-shared layers before
    // building the model.  The weight loader already materialized them via
    // eval_all, but releasing the entries here prevents the model constructor
    // from holding duplicate references and frees the backing VRAM after load.
    // Mirrors upstream mlx-lm PR #1240 (commit df1d3f3).
    models::strip_gemma4_kv_shared_weights(&mut weights, &full_config);
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    // Sanitize audio conv weights (PyTorch -> MLX format)
    sanitize_gemma4_audio_weights(&mut weights);

    let text_model = models::Gemma4Model::from_weights(&weights, &args)
        .map_err(|e| anyhow::anyhow!("Failed to load Gemma4 text model: {}", e))?;

    let vision_config: vision::encoders::gemma4::Gemma4VisionConfig =
        parse_required_vlm_subconfig(&full_config, "vision_config", "Gemma4 vision config")?;

    let text_quant = args.text_args().quantization;
    let group_size = text_quant
        .as_ref()
        .map(|q| q.group_size as i32)
        .unwrap_or(64);
    let bits = text_quant.as_ref().map(|q| q.bits as i32).unwrap_or(4);

    let vision_tower = vision::encoders::gemma4::Gemma4VisionModel::from_weights(
        &weights,
        "vision_tower",
        &vision_config,
        group_size,
        bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load Gemma4 vision tower: {}", e))?;

    // The RMS norm is applied BEFORE the projection on the vision encoder's
    // output dim (`vision_config.hidden_size`), matching upstream mlx-vlm
    // after the `embedding_post_projection_norm` -> `embedding_pre_projection_norm`
    // rename. This is a BREAKING change for pre-rename Gemma 4 VLM checkpoints;
    // users must re-download `mlx-community/gemma-4-*-it-4bit` or equivalent.
    let embed_vision = vision::gemma4_multimodal_embedder::Gemma4MultimodalEmbedder::from_weights(
        &weights,
        "embed_vision",
        vision_config.hidden_size,
        vision_config.rms_norm_eps(),
        group_size,
        bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load Gemma4 multimodal embedder: {}", e))?;

    let image_processor_config = read_optional_model_json(model_path, "processor_config.json")
        .and_then(|config| config.get("image_processor").cloned());
    let max_soft_tokens = image_processor_config
        .as_ref()
        .and_then(|cfg| cfg.get("max_soft_tokens"))
        .and_then(|value| value.as_u64())
        .unwrap_or(vision_config.default_output_length as u64) as usize;
    let patch_size = image_processor_config
        .as_ref()
        .and_then(|cfg| cfg.get("patch_size"))
        .and_then(|value| value.as_u64())
        .unwrap_or(vision_config.patch_size as u64) as usize;
    let pooling_kernel_size = image_processor_config
        .as_ref()
        .and_then(|cfg| cfg.get("pooling_kernel_size"))
        .and_then(|value| value.as_u64())
        .unwrap_or(vision_config.pooling_kernel_size as u64) as usize;

    let processor = vision::processors::gemma4::Gemma4Processor::new(
        patch_size,
        max_soft_tokens,
        pooling_kernel_size,
    );

    let mut vlm = vision::Gemma4VLModel::new(
        models::Gemma4Wrapper::new(text_model),
        vision_tower,
        embed_vision,
        processor,
        full_config
            .get("image_token_id")
            .and_then(|value| value.as_i64())
            .unwrap_or(258_880) as i32,
        full_config
            .get("boi_token_id")
            .and_then(|value| value.as_i64())
            .unwrap_or(255_999) as i32,
        full_config
            .get("eoi_token_id")
            .and_then(|value| value.as_i64())
            .unwrap_or(258_882) as i32,
    );
    vlm.set_weight_backing(weight_backing);

    // Load audio tower if audio_config is present
    if let Some(audio_config_val) = full_config.get("audio_config")
        && !audio_config_val.is_null()
    {
        let audio_config: crate::audio::AudioConfig =
            serde_json::from_value(audio_config_val.clone())
                .map_err(|e| anyhow::anyhow!("Failed to parse audio_config: {}", e))?;

        // Only load if audio weights actually exist
        if weights.contains_key("audio_tower.output_proj.weight") {
            let audio_tower = crate::audio::AudioEncoder::from_weights(
                &weights,
                "audio_tower",
                &audio_config,
                group_size,
                bits,
            )
            .map_err(|e| anyhow::anyhow!("Failed to load audio tower: {}", e))?;

            // Audio embedder: RMS norm on the audio encoder's output dim
            // (`audio_config.hidden_size`) BEFORE projection, matching the
            // reordered upstream Gemma 4 multimodal embedder. See the note on
            // `embed_vision` above for migration guidance.
            let embed_audio =
                vision::gemma4_multimodal_embedder::Gemma4MultimodalEmbedder::from_weights(
                    &weights,
                    "embed_audio",
                    audio_config.hidden_size,
                    audio_config.rms_norm_eps,
                    group_size,
                    bits,
                )
                .map_err(|e| anyhow::anyhow!("Failed to load audio embedder: {}", e))?;

            let audio_token_id = full_config
                .get("audio_token_id")
                .and_then(|v| v.as_i64())
                .unwrap_or(258_881) as i32;
            let boa_token_id = full_config
                .get("boa_token_id")
                .and_then(|v| v.as_i64())
                .unwrap_or(256_000) as i32;
            let eoa_token_id = full_config
                .get("eoa_token_id")
                .and_then(|v| v.as_i64())
                .unwrap_or(258_883) as i32;

            vlm.set_audio(
                audio_tower,
                embed_audio,
                audio_token_id,
                boa_token_id,
                eoa_token_id,
            );

            eprintln!(
                "Loaded Gemma4 audio encoder ({} Conformer layers)",
                audio_config.num_hidden_layers
            );
        }
    }

    Ok(LoadedModel::Gemma4VLM(vlm))
}

#[cfg(test)]
#[path = "vlm_gemma_tests.rs"]
mod tests;
