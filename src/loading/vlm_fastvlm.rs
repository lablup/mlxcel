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

//! FastVLM loader: FastViTHD vision tower + `mlp2x_gelu` projector + stock Qwen2
//! text decoder (`llava_qwen2` / `fastvlm`). Handles both the genuine
//! (`apple/FastVLM-0.5B`) and converted (`mlx-community/FastVLM-0.5B-bf16`)
//! weight layouts, normalizing to: vision under `vision_tower.vision_model.`,
//! projector under `mm_projector.linear_{1,2}`, text as bare `model.*` for the
//! Qwen2 builder. The `<image>` sentinel is fixed at `-200`; each image expands
//! to `(image_size / patch_size)^2 = 256` tokens (LLaVA merge).

use anyhow::Result;
use serde_json::Value;
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::vision;
use crate::vision::connectors::mlp::MLPProjector;
use crate::vision::encoders::fastvlm::{FastvlmVisionConfig, FastvlmVisionEncoder};
use crate::vision::processors::fastvlm::FastvlmProcessor;
use mlxcel_core::weights::WeightMap;

use super::{load_vlm_weights_common, read_sanitized_vlm_config};

const IMAGE_TOKEN_ID: i32 = -200;
const MM_TOKENS_PER_IMAGE: usize = 256;

/// Genuine-layout prefix; its presence selects the genuine sanitize path.
const GENUINE_VISION_PREFIX: &str = "model.vision_tower.vision_tower.model.";

fn is_4d(shape: &[i32]) -> bool {
    shape.len() == 4
}

/// Normalize both weight layouts to the internal scheme and apply drop rules.
/// Genuine checkpoints store conv weights channels-first `(O, I/G, kH, kW)` and
/// `layer_scale` as `(C, 1, 1)`; converted checkpoints are already channels-last
/// and `(1, 1, C)`, so their tensors pass through unchanged.
fn sanitize_fastvlm_weights(weights: WeightMap, tie_word_embeddings: bool) -> WeightMap {
    let genuine = weights.keys().any(|k| k.starts_with(GENUINE_VISION_PREFIX));

    // Decide the conv layout once from an unambiguous depthwise probe:
    // genuine `(C, 1, 3, 3)` (axis 1 == 1) vs converted `(C, 3, 3, 1)`.
    let permute_convs = if genuine {
        weights
            .get("model.vision_tower.vision_tower.model.patch_embed.1.reparam_conv.weight")
            .map(|w| {
                let s = mlxcel_core::array_shape(w);
                is_4d(&s) && s[1] == 1 && s[3] != 1
            })
            .unwrap_or(false)
    } else {
        false
    };

    let mut out = WeightMap::new();
    for (key, value) in weights {
        // Drop rules: BN counters, the classification head, and the redundant
        // lm_head when embeddings are tied.
        if key.ends_with(".num_batches_tracked") || key.ends_with("head.proj") {
            continue;
        }
        if tie_word_embeddings && (key == "lm_head.weight" || key == "model.lm_head.weight") {
            continue;
        }

        // Key normalization.
        let mut new_key = if genuine {
            if let Some(rest) = key.strip_prefix(GENUINE_VISION_PREFIX) {
                // `patch_embed.<n>` -> `patch_embed.blocks.<n>`.
                let rest = if let Some(n) = rest.strip_prefix("patch_embed.") {
                    if n.starts_with("blocks.") {
                        format!("patch_embed.{n}")
                    } else {
                        format!("patch_embed.blocks.{n}")
                    }
                } else {
                    rest.to_string()
                };
                format!("vision_tower.vision_model.{rest}")
            } else if let Some(rest) = key.strip_prefix("model.mm_projector.") {
                format!("mm_projector.{rest}")
            } else if let Some(rest) = key.strip_prefix("model.") {
                format!("model.{rest}")
            } else {
                key.clone()
            }
        } else if let Some(rest) = key.strip_prefix("language_model.") {
            rest.to_string()
        } else {
            key.clone()
        };

        // Projector rename for MLPProjector reuse.
        if let Some(rest) = new_key.strip_prefix("mm_projector.0.") {
            new_key = format!("mm_projector.linear_1.{rest}");
        } else if let Some(rest) = new_key.strip_prefix("mm_projector.2.") {
            new_key = format!("mm_projector.linear_2.{rest}");
        }

        // Layout fixes for genuine vision tensors.
        let is_vision = new_key.starts_with("vision_tower.vision_model.");
        let value = if genuine && is_vision {
            let shape = mlxcel_core::array_shape(&value);
            if new_key.ends_with("layer_scale")
                || new_key.ends_with("layer_scale_1")
                || new_key.ends_with("layer_scale_2")
            {
                // (C, 1, 1) -> (1, 1, C).
                if shape.len() == 3 && shape[2] == 1 {
                    mlxcel_core::transpose_axes(&value, &[1, 2, 0])
                } else {
                    value
                }
            } else if permute_convs && is_4d(&shape) {
                // (O, I/G, kH, kW) -> (O, kH, kW, I/G).
                mlxcel_core::transpose_axes(&value, &[0, 2, 3, 1])
            } else {
                value
            }
        } else {
            value
        };

        out.insert(new_key, value);
    }
    out
}

pub(crate) fn load_fastvlm_vlm(model_path: &Path) -> Result<LoadedModel> {
    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    let projector_type = full_config
        .get("mm_projector_type")
        .and_then(|v| v.as_str())
        .unwrap_or("mlp2x_gelu");
    if projector_type != "mlp2x_gelu" {
        anyhow::bail!(
            "FastVLM projector type {projector_type:?} is not supported (only mlp2x_gelu)"
        );
    }

    let tie_word_embeddings = full_config
        .get("tie_word_embeddings")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let mut weights = sanitize_fastvlm_weights(
        load_vlm_weights_common(model_path, None)?,
        tie_word_embeddings,
    );

    // Text: stock Qwen2 (biases auto-load from weight presence). FastVLM keeps
    // the text fields at the config root (no `text_config` block). Tie-embed
    // sanitize provides `lm_head` from `embed_tokens` when the head is dropped.
    models::sanitize_tied_embeddings(&mut weights, &full_config);
    let text_args: models::llama3::ModelArgs = serde_json::from_value(full_config.clone())
        .map_err(|e| anyhow::anyhow!("Failed to parse FastVLM text config: {}", e))?;
    let text_model = LoadedModel::Qwen2(
        models::Qwen2Model::from_weights(&weights, &text_args)
            .map_err(|e| anyhow::anyhow!("Failed to load FastVLM text model: {}", e))?,
    );
    let text_hidden_size = text_args.hidden_size;

    // Vision: FastViTHD (fixed MobileCLIP-L 1024 geometry across model sizes).
    let vcfg = FastvlmVisionConfig::default();
    let vision_encoder = FastvlmVisionEncoder::from_weights(&weights, &vcfg, "vision_tower")
        .map_err(|e| anyhow::anyhow!("Failed to load FastVLM vision tower: {}", e))?;

    let connector = MLPProjector::from_weights(&weights, "mm_projector", 64, 4)
        .map_err(|e| anyhow::anyhow!("Failed to load FastVLM projector: {}", e))?;

    let processor = fastvlm_processor(model_path, vcfg.image_size as usize);

    let vision_module = vision::VisionModule {
        encoder: Box::new(vision_encoder),
        connector: Box::new(connector),
        processor: Box::new(processor),
        image_token_id: IMAGE_TOKEN_ID,
        pad_token_id: 0,
        hidden_size: text_hidden_size,
        boi_token_id: 0,
        eoi_token_id: 0,
        mm_tokens_per_image: MM_TOKENS_PER_IMAGE,
        merge_strategy: vision::MergeStrategy::LLaVA,
        has_bos: true,
        separator_token_id: None,
        suffix_tokens: Vec::new(),
        block_prefix_tokens: Vec::new(),
        block_suffix_tokens: Vec::new(),
        pixtral_layout: None,
    };

    let vlm = vision::VisionLanguageModel {
        text_model: Box::new(text_model),
        vision: vision_module,
    };
    Ok(LoadedModel::FastVLM(vlm))
}

/// Build the processor, honoring `preprocessor_config.json` overrides
/// (`image_mean` / `image_std`) when present. Defaults are a plain `1/255`
/// rescale.
fn fastvlm_processor(model_path: &Path, image_size: usize) -> FastvlmProcessor {
    let mut processor = FastvlmProcessor {
        image_size,
        ..FastvlmProcessor::default()
    };
    let cfg_path = model_path.join("preprocessor_config.json");
    if let Ok(text) = std::fs::read_to_string(&cfg_path)
        && let Ok(cfg) = serde_json::from_str::<Value>(&text)
    {
        if let Some(m) = read_triplet(&cfg, "image_mean") {
            processor.mean = m;
        }
        if let Some(s) = read_triplet(&cfg, "image_std") {
            processor.std = s;
        }
    }
    processor
}

fn read_triplet(cfg: &Value, key: &str) -> Option<[f32; 3]> {
    let arr = cfg.get(key)?.as_array()?;
    if arr.len() != 3 {
        return None;
    }
    Some([
        arr[0].as_f64()? as f32,
        arr[1].as_f64()? as f32,
        arr[2].as_f64()? as f32,
    ])
}
