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

//! PLaMo 2.1 VL (`plamo2vl`) loader (issue #1380).
//!
//! Two key layouts are accepted and normalized to the HF one before any module
//! reads them:
//!
//! | tensor | HF export (`pfnet/plamo-2.1-2b-vl`) | mlx-vlm conversion |
//! | --- | --- | --- |
//! | text model | `model.*`, `lm_head.weight` | `language_model.model.*` (no head) |
//! | adapter biases | `image_proj.biasN._bias` | `image_proj.biasN.bias` |
//! | SigLIP head | `...model.head.*` (unused) | absent |
//!
//! Normalized form: `model.*` / `lm_head.*` for the text model and
//! `image_proj.biasN.bias`; the unused attention-pooling head is dropped. The
//! SigLIP patch embedding is `[1152, 3, 14, 14]` in the HF export and already
//! channels-last in conversions; the shared SigLIP loader transposes only the
//! former.
//!
//! The text config carries no `eos_token_id` (the top-level `1`, i.e.
//! `<|plamo:bos|>`, is the stop id) and, on quantized conversions, no
//! `quantization` block; both are inherited from the top level. On a quantized
//! checkpoint the bf16 tower and adapter tensors are cast to f16 on Apple
//! Silicon (quantization `.scales` / `.biases` stay bf16), mirroring the
//! InternVL loader; the text model's tensors are left exactly as the PLaMo 2
//! text loader would see them.

use anyhow::Result;
use mlxcel_core::weights::WeightMap;
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::vision;

use super::{load_vlm_weights_common, parse_required_vlm_subconfig, read_sanitized_vlm_config};

const VISION_PREFIX: &str = "vision_model.vision_encoder.model";

/// Rewrite mlx-vlm keys into the HF layout and drop the unused SigLIP head.
///
/// Used by: [`load_plamo2_vl`].
pub(crate) fn normalize_plamo2_vl_weight_keys(weights: WeightMap) -> WeightMap {
    let head_prefix = format!("{VISION_PREFIX}.head.");
    let mut out = WeightMap::new();
    for (key, value) in weights {
        if key.starts_with(&head_prefix) {
            continue;
        }
        let key = match key.strip_prefix("language_model.") {
            Some(rest) if rest.starts_with("model.") || rest.starts_with("lm_head.") => {
                rest.to_string()
            }
            _ => key,
        };
        let key = if key.starts_with("image_proj.") && key.ends_with("._bias") {
            format!("{}.bias", &key[..key.len() - "._bias".len()])
        } else {
            key
        };
        out.insert(key, value);
    }
    out
}

/// The `text_config` as PLaMo 2 `ModelArgs`, with the top-level
/// `eos_token_id` and `quantization` inherited when it has none.
fn plamo2_vl_text_args(full_config: &serde_json::Value) -> Result<models::plamo2::ModelArgs> {
    let mut text = full_config
        .get("text_config")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Missing text_config in PLaMo 2.1 VL config.json"))?;
    let obj = super::require_object_mut(&mut text, "PLaMo 2.1 VL text_config")?;
    obj.entry("model_type")
        .or_insert_with(|| serde_json::json!("plamo2"));
    for key in ["eos_token_id", "quantization"] {
        if obj.get(key).is_none_or(|v| v.is_null())
            && let Some(value) = full_config.get(key).filter(|v| !v.is_null())
        {
            obj.insert(key.to_string(), value.clone());
        }
    }
    serde_json::from_value(text)
        .map_err(|e| anyhow::anyhow!("Failed to parse PLaMo 2.1 VL text_config: {e}"))
}

/// Whether the checkpoint's tokenizer prepends `<|plamo:bos|>`.
///
/// `PlamoTokenizer.__init__` sets `add_bos_token = False` when the config omits
/// the key, so the default here is false too: the HF export sets it to true,
/// the mlx-vlm conversions drop it, and each checkpoint is served the prompt
/// its own processor builds.
fn plamo2_vl_adds_bos(model_path: &Path) -> bool {
    std::fs::read_to_string(model_path.join("tokenizer_config.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|config| config.get("add_bos_token").and_then(|v| v.as_bool()))
        .unwrap_or(false)
}

/// Load a PLaMo 2.1 VL checkpoint (SigLIP tiles + `image_proj` adapter +
/// PLaMo 2 hybrid decoder).
pub(crate) fn load_plamo2_vl(model_path: &Path) -> Result<LoadedModel> {
    use vision::connectors::plamo2_vl::Plamo2VlImageProjector;
    use vision::encoders::siglip::SigLipVisionModel;
    use vision::plamo2_vl::{
        PLAMO2_VL_BOS_TOKEN_ID, PLAMO2_VL_IMG_END_TOKEN_ID, PLAMO2_VL_IMG_START_TOKEN_ID,
        Plamo2VlModel,
    };
    use vision::plamo2_vl_config::{Plamo2VlProcessorConfig, Plamo2VlVisionConfig};
    use vision::processors::plamo2_vl::Plamo2VlProcessor;

    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;
    let vision_config: Plamo2VlVisionConfig =
        parse_required_vlm_subconfig(&full_config, "vision_config", "PLaMo 2.1 VL vision config")?;
    let siglip_config = vision_config
        .to_siglip_config()
        .map_err(|e| anyhow::anyhow!("PLaMo 2.1 VL vision config: {e}"))?;
    let text_args = plamo2_vl_text_args(&full_config)?;
    let image_token_id = full_config
        .get("image_token_index")
        .and_then(|v| v.as_i64())
        .map_or(vision_config.image_token_id, |id| id as i32);

    let mut weights = normalize_plamo2_vl_weight_keys(load_vlm_weights_common(model_path, None)?);
    if text_args.quantization.is_some() && mlxcel_core::hardware::get_hardware().is_apple_silicon()
    {
        let had_bf16 = models::convert_bf16_weights_with_keep(&mut weights, |key| {
            let multimodal = key.starts_with("vision_model.") || key.starts_with("image_proj.");
            !multimodal || key.ends_with(".scales") || key.ends_with(".biases")
        });
        // The tower and adapter are bf16 by design on a quantized
        // conversion, so the bf16-checkpoint advisory does not apply.
        let _ = had_bf16;
    }

    let group_size = text_args.group_size();
    let bits = text_args.bits();
    let patch_key = format!("{VISION_PREFIX}.embeddings.patch_embedding.weight");
    let vision_dtype = weights
        .get(&patch_key)
        .map(|w| mlxcel_core::array_dtype(w))
        .ok_or_else(|| anyhow::anyhow!("Weight not found: {patch_key}"))?;
    let vision_model = SigLipVisionModel::from_weights_with_quant_and_gelu(
        &weights,
        &siglip_config,
        VISION_PREFIX,
        group_size,
        bits,
        false,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load PLaMo 2.1 VL SigLIP tower: {e}"))?;
    let projector = Plamo2VlImageProjector::from_weights(
        &weights,
        "image_proj",
        text_args.rms_norm_eps,
        group_size,
        bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load PLaMo 2.1 VL image_proj adapter: {e}"))?;

    let processor_config =
        Plamo2VlProcessorConfig::from_model_dir(model_path).map_err(|e| anyhow::anyhow!("{e}"))?;
    let processor = Plamo2VlProcessor::new(&processor_config);
    if processor.image_size != siglip_config.image_size {
        anyhow::bail!(
            "PLaMo 2.1 VL tile size {} does not match the SigLIP tower's {}px position table",
            processor.image_size,
            siglip_config.image_size
        );
    }
    let num_image_token = processor.num_image_token();

    // Hand the text model only its own tensors.
    let text_weights: WeightMap = weights
        .into_iter()
        .filter(|(key, _)| key.starts_with("model.") || key.starts_with("lm_head."))
        .collect();
    let text_model = models::Plamo2Model::from_weights(text_args, text_weights)
        .map_err(|e| anyhow::anyhow!("Failed to load PLaMo 2.1 VL text model: {e}"))?;
    let eos_token_ids = mlxcel_core::generate::LanguageModel::eos_token_ids(&text_model);

    Ok(LoadedModel::Plamo2VLM(Plamo2VlModel {
        text_model,
        vision_model,
        projector,
        processor,
        vision_dtype,
        image_token_id,
        img_start_token_id: PLAMO2_VL_IMG_START_TOKEN_ID,
        img_end_token_id: PLAMO2_VL_IMG_END_TOKEN_ID,
        bos_token_id: plamo2_vl_adds_bos(model_path).then_some(PLAMO2_VL_BOS_TOKEN_ID),
        num_image_token,
        eos_token_ids,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_maps_both_layouts_to_the_hf_keys() {
        let _guard = crate::models::embedding_test_support::mlx_test_guard();
        let mut weights = WeightMap::new();
        for key in [
            "language_model.model.embed_tokens.weight",
            "language_model.lm_head.weight",
            "image_proj.bias0.bias",
            "image_proj.bias1._bias",
            "image_proj.linear1.biases",
            "vision_model.vision_encoder.model.head.probe",
            "vision_model.vision_encoder.model.post_layernorm.bias",
            "model.norm.weight",
        ] {
            weights.insert(key.to_string(), mlxcel_core::zeros(&[1], 0));
        }
        let mut keys: Vec<String> = normalize_plamo2_vl_weight_keys(weights)
            .into_keys()
            .collect();
        keys.sort();
        assert_eq!(
            keys,
            vec![
                "image_proj.bias0.bias",
                "image_proj.bias1.bias",
                "image_proj.linear1.biases",
                "lm_head.weight",
                "model.embed_tokens.weight",
                "model.norm.weight",
                "vision_model.vision_encoder.model.post_layernorm.bias",
            ]
        );
    }

    #[test]
    fn add_bos_follows_the_tokenizer_config_and_defaults_off() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!plamo2_vl_adds_bos(dir.path()));
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"bos_token": "<|plamo:bos|>"}"#,
        )
        .unwrap();
        assert!(!plamo2_vl_adds_bos(dir.path()));
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"add_bos_token": true}"#,
        )
        .unwrap();
        assert!(plamo2_vl_adds_bos(dir.path()));
    }

    #[test]
    fn text_args_inherit_top_level_eos_and_quantization() {
        let config = serde_json::json!({
            "model_type": "plamo2vl",
            "eos_token_id": 1,
            "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
            "text_config": {
                "hidden_size": 2048, "num_hidden_layers": 32, "vocab_size": 100032,
                "intermediate_size": 5632, "num_attention_heads": 32,
                "num_key_value_heads": 4, "rope_theta": 1000000.0,
                "rope_local_theta": 1000000.0, "full_attention_idx": [0, 2]
            }
        });
        let args = plamo2_vl_text_args(&config).unwrap();
        assert_eq!(args.eos_token_ids(), vec![1]);
        assert_eq!((args.group_size(), args.bits()), (64, 4));
        assert!(args.tie_word_embeddings);
        assert_eq!(args.rope_base(1), 1_000_000.0);
    }
}
