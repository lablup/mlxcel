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

//! Nemotron H Nano Omni VLM loader.
//!
//! Constructs [`crate::vision::NemotronHNanoOmniVlModel`] from a HF
//! `mlx-community` checkpoint. The vision path landed; the
//! audio path was added and only activates when the checkpoint
//! ships a `sound_config` block.
//!
//! Wire-up:
//! - `config.json` carries `text_config` (parsed by
//!   [`crate::models::nemotron_h::NemotronHConfig`]) plus `vision_config`
//!   (parsed by
//!   [`crate::vision::encoders::nemotron_h_nano_omni::NemotronHNanoOmniVisionConfig`])
//!   and the top-level projector knobs (`projector_hidden_size`,
//!   `vit_hidden_size`, `downsample_ratio`, `ps_version`,
//!   `img_context_token_id`).
//! - When `sound_config` is present, the loader additionally parses it
//!   into [`crate::audio::nemotron_h_nano_omni::NemotronOmniAudioConfig`]
//!   and builds the Parakeet/Conformer encoder + projection.
//! - Weight remap mirrors upstream `Model.sanitize` and
//!   `sanitize_audio_weights`:
//!   * rename `mlp1.{0,1,3}` → `mlp1.layers.{0,1,3}`,
//!   * strip the `language_model.` prefix from text weights,
//!   * skip `sound_encoder.encoder.feature_extractor.*` (training-only
//!     DSP reference weights),
//!   * skip `*.num_batches_tracked` BatchNorm scratch state,
//!   * transpose Conv1d weights `[O, I, K]` → `[O, K, I]` and Conv2d
//!     weights `[O, I, kh, kw]` → `[O, kh, kw, I]` so MLX channel-last
//!     conv ops can consume them directly.

use anyhow::Result;
use mlxcel_core::weights::WeightMap;
use serde_json::Value;
use std::path::Path;

use super::{load_vlm_weights_common, read_sanitized_vlm_config};
use crate::LoadedModel;
use crate::audio::nemotron_h_nano_omni::{
    NemotronOmniAudioConfig, NemotronOmniFeatureExtractor, NemotronOmniSoundEncoder,
    NemotronOmniSoundProjection,
};
use crate::models::NemotronHModel;
use crate::models::nemotron_h::{BlockType, NemotronHConfig};
use crate::vision::encoders::nemotron_h_nano_omni::{
    NemotronHNanoOmniVisionConfig, NemotronHNanoOmniVisionModel,
};
use crate::vision::nemotron_h_nano_omni_vl::{
    NemotronHNanoOmniProjector, NemotronHNanoOmniVlConfig, NemotronHNanoOmniVlModel,
    NemotronOmniAudioBundle,
};
use crate::vision::processors::nemotron_h_nano_omni::{
    NemotronHNanoOmniImageProcessor, NemotronHNanoOmniProcessorConfig,
};

/// Strip audio-only weights when the checkpoint has no `sound_config`.
///
/// dropped every `sound_*`/`audio_tower.*` weight at load
/// time. keeps them when the checkpoint advertises a
/// `sound_config` (so the audio encoder can pick them up) and only
/// strips them when audio is genuinely absent. This preserves
/// backwards-compatibility for older Nemotron H Nano Omni snapshots
/// that ship without a sound config.
fn filter_out_audio_weights(weights: WeightMap, drop_audio: bool) -> WeightMap {
    if !drop_audio {
        return weights;
    }
    weights
        .into_iter()
        .filter(|(key, _)| {
            !(key.starts_with("sound_encoder.")
                || key.starts_with("sound_projection.")
                || key.starts_with("sound_feature_extractor.")
                || key.starts_with("audio_tower."))
        })
        .collect()
}

/// Decide whether the `sound_encoder` conv weights are in PyTorch `[O, I, K]`
/// layout (need transposing to MLX channel-last) or are already channel-last
/// (must be left untouched).
///
/// The Parakeet conv module mixes depthwise and pointwise Conv1d whose
/// channel-last vs PyTorch shape signatures are mirror images (depthwise
/// channel-last `[O, kW>1, 1]` vs pointwise channel-last `[O, 1, in>1]`), so no
/// single per-weight shape test can classify both: the earlier per-weight guard
/// left pointwise on an unconditional transpose, which corrupted a channel-last
/// checkpoint's pointwise weights and crashed the audio conv (#428). Decide the
/// whole tower's layout ONCE from an unambiguous depthwise weight: PyTorch
/// depthwise is `[O, 1, kW]` (axis 1 == 1) while channel-last depthwise is
/// `[O, kW, 1]` (axis 2 == 1). mlx-community checkpoints (e.g.
/// Nemotron-3-Nano-Omni) ship the encoder already channel-last, so this returns
/// false and every `sound_encoder` conv transpose is skipped.
fn nemotron_audio_needs_conv_transpose(weights: &WeightMap) -> bool {
    for (key, value) in weights.iter() {
        if key.starts_with("sound_encoder.encoder.")
            && key.contains("depthwise")
            && key.ends_with(".weight")
        {
            let shape = mlxcel_core::array_shape(value);
            if shape.len() == 3 {
                return shape[1] == 1;
            }
        }
    }
    // No depthwise weight found: preserve the historical unconditional transpose.
    true
}

/// Mirror upstream `sanitize_audio_weights` from
/// https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/nemotron_h_nano_omni/audio.py:
/// - drop `sound_encoder.encoder.feature_extractor.*` (training-only),
/// - drop `*.num_batches_tracked` BatchNorm scratch state,
/// - transpose Conv1d `[O, I, K]` -> MLX `[O, K, I]` and Conv2d
///   `[O, I, kH, kW]` -> MLX `[O, kH, kW, I]`, but ONLY when the checkpoint is
///   in PyTorch layout. A checkpoint already stored channel-last (detected once
///   via [`nemotron_audio_needs_conv_transpose`]) is left untouched so the
///   pointwise conv1d weights are not corrupted by a double transpose (#428).
fn sanitize_audio_weights(weights: WeightMap) -> WeightMap {
    let needs_transpose = nemotron_audio_needs_conv_transpose(&weights);
    let mut out = WeightMap::new();
    for (key, value) in weights {
        if key.starts_with("sound_encoder.encoder.feature_extractor.") {
            continue;
        }
        if key.ends_with(".num_batches_tracked") {
            continue;
        }
        if needs_transpose && key.starts_with("sound_encoder.encoder.") && key.ends_with(".weight")
        {
            let shape = mlxcel_core::array_shape(&value);
            match shape.len() {
                3 => {
                    out.insert(key, mlxcel_core::transpose_axes(&value, &[0, 2, 1]));
                    continue;
                }
                4 => {
                    out.insert(key, mlxcel_core::transpose_axes(&value, &[0, 2, 3, 1]));
                    continue;
                }
                _ => {}
            }
        }
        out.insert(key, value);
    }
    out
}

/// Mirror upstream `Model.sanitize` mlp1 layer rename:
/// `mlp1.{0,1,3}.` → `mlp1.layers.{0,1,3}.`.
fn rename_projector_weights(raw: WeightMap) -> WeightMap {
    let mut out = WeightMap::new();
    for (key, value) in raw {
        let new_key = if key.starts_with("mlp1.0.") {
            key.replacen("mlp1.0.", "mlp1.layers.0.", 1)
        } else if key.starts_with("mlp1.1.") {
            key.replacen("mlp1.1.", "mlp1.layers.1.", 1)
        } else if key.starts_with("mlp1.3.") {
            key.replacen("mlp1.3.", "mlp1.layers.3.", 1)
        } else {
            key
        };
        out.insert(new_key, value);
    }
    out
}

/// Split the unified weight map into text-model weights (with
/// `language_model.` prefix removed) and the remainder (vision tower +
/// projector + everything else). The text path can then call
/// [`NemotronHModel::sanitize_weights`] and feed the result into the
/// existing config-backed builder.
fn split_text_and_others(raw: WeightMap) -> (WeightMap, WeightMap) {
    let mut text = WeightMap::new();
    let mut others = WeightMap::new();
    for (key, value) in raw {
        if let Some(rest) = key.strip_prefix("language_model.") {
            text.insert(rest.to_string(), value);
        } else {
            others.insert(key, value);
        }
    }
    (text, others)
}

fn parse_text_config(full_config: &Value) -> Result<NemotronHConfig> {
    let text_value = full_config
        .get("text_config")
        .or_else(|| full_config.get("llm_config"))
        .cloned()
        .ok_or_else(|| {
            anyhow::anyhow!("Missing text_config in nemotron_h_nano_omni config.json")
        })?;
    let mut config: NemotronHConfig = serde_json::from_value(text_value)
        .map_err(|e| anyhow::anyhow!("Failed to parse Nemotron-H text config: {e}"))?;
    config
        .post_init()
        .map_err(|e| anyhow::anyhow!("Nemotron-H text config post_init failed: {e}"))?;
    Ok(config)
}

fn parse_vision_config(full_config: &Value) -> Result<NemotronHNanoOmniVisionConfig> {
    let vision_value = full_config
        .get("vision_config")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    serde_json::from_value(vision_value)
        .map_err(|e| anyhow::anyhow!("Failed to parse Nemotron H Nano Omni vision config: {e}"))
}

fn parse_processor_config(
    model_path: &Path,
    full_config: &Value,
) -> NemotronHNanoOmniProcessorConfig {
    // The released checkpoint stores image-processor knobs in
    // `preprocessor_config.json` (or the nested `image_processor` block
    // in `processor_config.json` on older exports). Fall back to
    // sensible defaults when the file is missing — the loader is
    // resilient to partial metadata.
    let mut config = NemotronHNanoOmniProcessorConfig::default();

    let mut sources: Vec<Value> = Vec::new();
    if let Ok(raw) = std::fs::read_to_string(model_path.join("preprocessor_config.json"))
        && let Ok(value) = serde_json::from_str::<Value>(&raw)
    {
        sources.push(value);
    }
    if let Ok(raw) = std::fs::read_to_string(model_path.join("processor_config.json"))
        && let Ok(value) = serde_json::from_str::<Value>(&raw)
    {
        if let Some(nested) = value.get("image_processor").cloned() {
            sources.push(nested);
        }
        sources.push(value);
    }

    // Final fallback to the top-level config so global overrides like
    // `downsample_ratio` propagate to the processor.
    sources.push(full_config.clone());

    fn array_of_3<T: serde::de::DeserializeOwned + Copy + Default>(
        value: Option<&Value>,
    ) -> Option<[T; 3]> {
        let arr = value?.as_array()?;
        if arr.len() != 3 {
            return None;
        }
        let mut out = [T::default(); 3];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = serde_json::from_value(arr[i].clone()).ok()?;
        }
        Some(out)
    }

    for source in &sources {
        if let Some(value) = array_of_3::<f32>(source.get("norm_mean")) {
            config.norm_mean = value;
        }
        if let Some(value) = array_of_3::<f32>(source.get("norm_std")) {
            config.norm_std = value;
        }
        if let Some(value) = source.get("patch_size").and_then(Value::as_u64) {
            config.patch_size = value as usize;
        }
        if let Some(value) = source.get("downsample_ratio").and_then(Value::as_f64) {
            config.downsample_ratio = value as f32;
        }
        if let Some(value) = source.get("min_num_patches").and_then(Value::as_u64) {
            config.min_num_patches = value as usize;
        }
        if let Some(value) = source.get("max_num_patches").and_then(Value::as_u64) {
            config.max_num_patches = value as usize;
        }
        if let Some(value) = source.get("max_model_len").and_then(Value::as_u64) {
            config.max_model_len = value as usize;
        }
    }
    config
}

fn parse_eos_token_ids(model_path: &Path, full_config: &Value) -> Vec<i32> {
    if let Ok(raw) = std::fs::read_to_string(model_path.join("generation_config.json"))
        && let Ok(value) = serde_json::from_str::<Value>(&raw)
        && let Some(ids) = read_eos(&value)
    {
        return ids;
    }
    read_eos(full_config).unwrap_or_default()
}

fn read_eos(value: &Value) -> Option<Vec<i32>> {
    let entry = value.get("eos_token_id")?;
    match entry {
        Value::Array(arr) => Some(
            arr.iter()
                .filter_map(|v| v.as_i64().map(|n| n as i32))
                .collect(),
        ),
        Value::Number(n) => n.as_i64().map(|n| vec![n as i32]),
        _ => None,
    }
}

fn parse_block_types(config: &NemotronHConfig) -> Result<Vec<BlockType>> {
    let pattern = config.hybrid_override_pattern.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "NemotronH: hybrid_override_pattern must be set in text_config for nemotron_h_nano_omni"
        )
    })?;
    Ok(pattern
        .iter()
        .map(|name| BlockType::from_str(name))
        .collect())
}

pub(crate) fn load_nemotron_h_nano_omni_vlm(model_path: &Path) -> Result<LoadedModel> {
    let (_, full_config) = read_sanitized_vlm_config(model_path)?;

    // 1. Parse all configs up front so we can fail early on missing fields.
    let text_config = parse_text_config(&full_config)?;
    let block_types = parse_block_types(&text_config)?;
    let vision_config = parse_vision_config(&full_config)?;
    let processor_config = parse_processor_config(model_path, &full_config);
    let processor = NemotronHNanoOmniImageProcessor::new(processor_config);

    let vit_hidden_size = full_config
        .get("vit_hidden_size")
        .and_then(Value::as_u64)
        .unwrap_or(vision_config.hidden_size as u64) as usize;
    let projector_hidden_size = full_config
        .get("projector_hidden_size")
        .and_then(Value::as_u64)
        .unwrap_or(4096) as usize;
    let downsample_ratio = full_config
        .get("downsample_ratio")
        .and_then(Value::as_f64)
        .unwrap_or(0.5) as f32;
    let ps_version = full_config
        .get("ps_version")
        .and_then(Value::as_str)
        .unwrap_or("v1")
        .to_string();
    let img_context_token_id = full_config
        .get("img_context_token_id")
        .or_else(|| full_config.get("image_token_index"))
        .and_then(Value::as_i64)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Missing img_context_token_id (or image_token_index) in nemotron_h_nano_omni config"
            )
        })? as i32;
    let image_start_token_id = full_config
        .get("image_start_token_id")
        .and_then(Value::as_i64)
        .unwrap_or(0) as i32;
    let image_end_token_id = full_config
        .get("image_end_token_id")
        .and_then(Value::as_i64)
        .unwrap_or(0) as i32;

    // Parse audio config. Audio is opt-in: when
    // `sound_config` is absent the loader stays bit-for-bit identical
    // to the issue- path.
    let audio_config = full_config
        .get("sound_config")
        .map(|value| -> Result<NemotronOmniAudioConfig> {
            serde_json::from_value(value.clone()).map_err(|e| {
                anyhow::anyhow!("Failed to parse Nemotron H Nano Omni sound_config: {e}")
            })
        })
        .transpose()?;
    let sound_context_token_id = full_config
        .get("sound_context_token_id")
        .and_then(Value::as_i64)
        .map(|v| v as i32);
    let sound_start_token_id = full_config
        .get("sound_start_token_id")
        .and_then(Value::as_i64)
        .unwrap_or(0) as i32;
    let sound_end_token_id = full_config
        .get("sound_end_token_id")
        .and_then(Value::as_i64)
        .unwrap_or(0) as i32;

    let eos_token_ids = parse_eos_token_ids(model_path, &full_config);

    // 2. Load weights. When the checkpoint ships a sound_config we
    // keep audio weights and run the upstream `sanitize_audio_weights`
    // transpose pass; when audio is absent we drop them as before.
    let raw_weights = load_vlm_weights_common(model_path, None)?;
    let drop_audio = audio_config.is_none();
    let raw_weights = filter_out_audio_weights(raw_weights, drop_audio);
    let raw_weights = if audio_config.is_some() {
        sanitize_audio_weights(raw_weights)
    } else {
        raw_weights
    };
    let raw_weights = rename_projector_weights(raw_weights);
    let (text_weights, other_weights) = split_text_and_others(raw_weights);

    // 3. Build the text model using the existing Nemotron-H sanitizer
    // and `from_weights` builder. This guarantees parity with the
    // text-only Nemotron-H path — any future text-side improvement
    // automatically benefits the VLM.
    let text_weights = NemotronHModel::sanitize_weights(text_weights, &text_config);
    let text_model = NemotronHModel::from_weights(text_config, text_weights, block_types)
        .map_err(|err| anyhow::anyhow!("Failed to load Nemotron H Nano Omni text model: {err}"))?;

    // 4. Build the vision tower and the multimodal projector. The vision
    // tower lives at `vision_model.radio_model.*`; the projector at
    // `mlp1.*` (after the layer rename above).
    let group_size = group_size_from_config(&full_config);
    let bits = bits_from_config(&full_config);
    let vision_tower = NemotronHNanoOmniVisionModel::from_weights(
        &other_weights,
        "vision_model.radio_model",
        &vision_config,
        group_size,
        bits,
    )
    .map_err(|err| anyhow::anyhow!("Failed to load Nemotron H Nano Omni vision tower: {err}"))?;
    let projector =
        NemotronHNanoOmniProjector::from_weights(&other_weights, "mlp1", group_size, bits)
            .map_err(|err| {
                anyhow::anyhow!("Failed to load Nemotron H Nano Omni projector: {err}")
            })?;

    // 5. Compose the VLM wrapper. Audio bundle is attached only when
    // the checkpoint actually shipped it; failures here surface as
    // load errors, since a partial audio path is worse than a clean
    // text+vision-only fallback would be.
    let text_hidden_size = text_model.hidden_size();
    let vl_config = NemotronHNanoOmniVlConfig {
        vit_hidden_size,
        projector_hidden_size,
        text_hidden_size,
        downsample_ratio,
        ps_version,
        img_context_token_id,
        image_start_token_id,
        image_end_token_id,
        sound_context_token_id,
        sound_start_token_id,
        sound_end_token_id,
        eos_token_ids,
    };
    let mut model =
        NemotronHNanoOmniVlModel::new(text_model, vision_tower, projector, processor, vl_config);

    if let Some(audio_config) = audio_config {
        let encoder = NemotronOmniSoundEncoder::from_weights(
            &other_weights,
            "sound_encoder",
            &audio_config,
            group_size,
            bits,
        )
        .map_err(|err| {
            anyhow::anyhow!("Failed to load Nemotron H Nano Omni sound encoder: {err}")
        })?;
        let projection = NemotronOmniSoundProjection::from_weights(
            &other_weights,
            "sound_projection",
            &audio_config,
            group_size,
            bits,
        )
        .map_err(|err| {
            anyhow::anyhow!("Failed to load Nemotron H Nano Omni sound projection: {err}")
        })?;
        let feature_extractor = NemotronOmniFeatureExtractor::new(&audio_config);
        let bundle = NemotronOmniAudioBundle {
            config: audio_config,
            feature_extractor,
            encoder,
            projection,
        };
        model = model.with_audio(bundle);
    }

    Ok(LoadedModel::NemotronHNanoOmniVLM(model))
}

fn group_size_from_config(full_config: &Value) -> i32 {
    full_config
        .get("quantization")
        .and_then(|q| q.get("group_size"))
        .and_then(Value::as_i64)
        .unwrap_or(64) as i32
}

fn bits_from_config(full_config: &Value) -> i32 {
    full_config
        .get("quantization")
        .and_then(|q| q.get("bits"))
        .and_then(Value::as_i64)
        .unwrap_or(4) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ones(shape: &[i32]) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
        mlxcel_core::ones(shape, mlxcel_core::dtype::FLOAT32)
    }

    #[test]
    fn filter_out_audio_weights_keeps_audio_when_audio_enabled() {
        let mut weights = WeightMap::new();
        weights.insert("language_model.embed_tokens.weight".into(), ones(&[8, 8]));
        weights.insert(
            "sound_encoder.encoder.subsampling.layers.0.weight".into(),
            ones(&[4, 3, 3, 1]),
        );
        weights.insert("sound_projection.linear1.weight".into(), ones(&[16, 8]));
        weights.insert(
            "sound_encoder.encoder.feature_extractor.something.weight".into(),
            ones(&[1]),
        );

        // Audio enabled: only `feature_extractor.*` is left in the map for the sanitizer to drop later.
        let kept = filter_out_audio_weights(weights, false);
        assert!(kept.contains_key("sound_encoder.encoder.subsampling.layers.0.weight"));
        assert!(kept.contains_key("sound_projection.linear1.weight"));
        assert!(kept.contains_key("sound_encoder.encoder.feature_extractor.something.weight"));
    }

    #[test]
    fn filter_out_audio_weights_drops_audio_when_disabled() {
        let mut weights = WeightMap::new();
        weights.insert("language_model.embed_tokens.weight".into(), ones(&[8, 8]));
        weights.insert(
            "sound_encoder.encoder.subsampling.layers.0.weight".into(),
            ones(&[4, 3, 3, 1]),
        );
        weights.insert("sound_projection.linear1.weight".into(), ones(&[16, 8]));
        weights.insert("audio_tower.foo".into(), ones(&[1]));

        let kept = filter_out_audio_weights(weights, true);
        assert!(kept.contains_key("language_model.embed_tokens.weight"));
        assert!(!kept.contains_key("sound_encoder.encoder.subsampling.layers.0.weight"));
        assert!(!kept.contains_key("sound_projection.linear1.weight"));
        assert!(!kept.contains_key("audio_tower.foo"));
    }

    #[test]
    fn sanitize_audio_weights_drops_feature_extractor_and_num_batches_tracked() {
        let mut weights = WeightMap::new();
        weights.insert(
            "sound_encoder.encoder.feature_extractor.window".into(),
            ones(&[10]),
        );
        weights.insert(
            "sound_encoder.encoder.layers.0.conv.norm.num_batches_tracked".into(),
            ones(&[1]),
        );
        weights.insert(
            "sound_encoder.encoder.layers.0.conv.norm.running_mean".into(),
            ones(&[16]),
        );
        weights.insert("sound_projection.linear1.weight".into(), ones(&[16, 8]));

        let sanitized = sanitize_audio_weights(weights);
        assert!(!sanitized.contains_key("sound_encoder.encoder.feature_extractor.window"));
        assert!(
            !sanitized.contains_key("sound_encoder.encoder.layers.0.conv.norm.num_batches_tracked")
        );
        assert!(sanitized.contains_key("sound_encoder.encoder.layers.0.conv.norm.running_mean"));
        assert!(sanitized.contains_key("sound_projection.linear1.weight"));
    }

    #[test]
    fn sanitize_audio_weights_transposes_conv_weights() {
        let mut weights = WeightMap::new();
        // Conv1d depthwise in PyTorch layout: [out=4, in=1, kernel=3].
        weights.insert(
            "sound_encoder.encoder.layers.0.conv.depthwise_conv.weight".into(),
            ones(&[4, 1, 3]),
        );
        // Conv2d weight in PyTorch layout: [out=4, in=1, kh=3, kw=3].
        weights.insert(
            "sound_encoder.encoder.subsampling.layers.0.weight".into(),
            ones(&[4, 1, 3, 3]),
        );

        let sanitized = sanitize_audio_weights(weights);
        let conv1d_shape = mlxcel_core::array_shape(
            sanitized
                .get("sound_encoder.encoder.layers.0.conv.depthwise_conv.weight")
                .unwrap(),
        );
        // After transpose(0,2,1): [4, 3, 1].
        assert_eq!(conv1d_shape, vec![4, 3, 1]);

        let conv2d_shape = mlxcel_core::array_shape(
            sanitized
                .get("sound_encoder.encoder.subsampling.layers.0.weight")
                .unwrap(),
        );
        // After transpose(0,2,3,1): [4, 3, 3, 1].
        assert_eq!(conv2d_shape, vec![4, 3, 3, 1]);
    }

    fn audio_shape(weights: &WeightMap, key: &str) -> Vec<i32> {
        mlxcel_core::array_shape(weights.get(key).unwrap())
    }

    const DEPTHWISE: &str = "sound_encoder.encoder.layers.0.conv.depthwise_conv.weight";
    const POINTWISE: &str = "sound_encoder.encoder.layers.0.conv.pointwise_conv1.weight";
    const SUBSAMPLE_CONV2D: &str = "sound_encoder.encoder.subsampling.layers.0.weight";

    #[test]
    fn sanitize_audio_weights_skips_channel_last_depthwise_conv1d() {
        // A depthwise conv1d already in MLX channel-last [out, kW, in=1] layout
        // must be left untouched (issue #428).
        let mut weights = WeightMap::new();
        weights.insert(DEPTHWISE.into(), ones(&[1024, 5, 1]));
        let sanitized = sanitize_audio_weights(weights);
        assert_eq!(audio_shape(&sanitized, DEPTHWISE), vec![1024, 5, 1]);
    }

    #[test]
    fn sanitize_audio_weights_transposes_pytorch_depthwise_conv1d() {
        // PyTorch depthwise [out, in=1, kW] -> MLX [out, kW, 1].
        let mut weights = WeightMap::new();
        weights.insert(DEPTHWISE.into(), ones(&[1024, 1, 5]));
        let sanitized = sanitize_audio_weights(weights);
        assert_eq!(audio_shape(&sanitized, DEPTHWISE), vec![1024, 5, 1]);
    }

    #[test]
    fn sanitize_audio_weights_depthwise_conv1d_is_idempotent() {
        let mut once = WeightMap::new();
        once.insert(DEPTHWISE.into(), ones(&[1024, 1, 5]));
        let once = sanitize_audio_weights(once);

        let mut twice = WeightMap::new();
        twice.insert(DEPTHWISE.into(), ones(&[1024, 1, 5]));
        let twice = sanitize_audio_weights(sanitize_audio_weights(twice));

        assert_eq!(
            audio_shape(&once, DEPTHWISE),
            audio_shape(&twice, DEPTHWISE)
        );
        assert_eq!(audio_shape(&twice, DEPTHWISE), vec![1024, 5, 1]);
    }

    #[test]
    fn sanitize_audio_weights_channel_last_pointwise_left_untouched() {
        // Regression (#428): in a channel-last checkpoint (depthwise [O, kW, 1])
        // the pointwise conv1d [O, 1, in] must NOT be transposed. The earlier
        // per-weight guard transposed the pointwise and crashed the audio conv.
        // The depthwise anchors the checkpoint-level layout detection.
        let mut weights = WeightMap::new();
        weights.insert(DEPTHWISE.into(), ones(&[1024, 9, 1]));
        weights.insert(POINTWISE.into(), ones(&[2048, 1, 1024]));
        let sanitized = sanitize_audio_weights(weights);
        assert_eq!(audio_shape(&sanitized, DEPTHWISE), vec![1024, 9, 1]);
        assert_eq!(audio_shape(&sanitized, POINTWISE), vec![2048, 1, 1024]);
    }

    #[test]
    fn sanitize_audio_weights_transposes_pytorch_pointwise_conv1d() {
        // PyTorch checkpoint (depthwise [O, 1, kW]): pointwise [O, in, 1] is
        // transposed to channel-last [O, 1, in].
        let mut weights = WeightMap::new();
        weights.insert(DEPTHWISE.into(), ones(&[1024, 1, 9]));
        weights.insert(POINTWISE.into(), ones(&[2048, 1024, 1]));
        let sanitized = sanitize_audio_weights(weights);
        assert_eq!(audio_shape(&sanitized, POINTWISE), vec![2048, 1, 1024]);
    }

    #[test]
    fn sanitize_audio_weights_skips_channel_last_conv2d() {
        // Conv2d already channel-last [out, kH, kW, in] must be left untouched;
        // a channel-last depthwise anchors the checkpoint-level layout detection.
        let mut weights = WeightMap::new();
        weights.insert(DEPTHWISE.into(), ones(&[1024, 3, 1]));
        weights.insert(SUBSAMPLE_CONV2D.into(), ones(&[32, 3, 3, 1]));
        let sanitized = sanitize_audio_weights(weights);
        assert_eq!(audio_shape(&sanitized, SUBSAMPLE_CONV2D), vec![32, 3, 3, 1]);
    }

    #[test]
    fn sanitize_audio_weights_conv2d_is_idempotent() {
        // A PyTorch depthwise anchors detection; after the first pass it is
        // channel-last, so the second pass skips and the result is stable.
        let mut once = WeightMap::new();
        once.insert(DEPTHWISE.into(), ones(&[1024, 1, 3]));
        once.insert(SUBSAMPLE_CONV2D.into(), ones(&[4, 1, 3, 3]));
        let once = sanitize_audio_weights(once);

        let mut twice = WeightMap::new();
        twice.insert(DEPTHWISE.into(), ones(&[1024, 1, 3]));
        twice.insert(SUBSAMPLE_CONV2D.into(), ones(&[4, 1, 3, 3]));
        let twice = sanitize_audio_weights(sanitize_audio_weights(twice));

        assert_eq!(
            audio_shape(&once, SUBSAMPLE_CONV2D),
            audio_shape(&twice, SUBSAMPLE_CONV2D)
        );
        assert_eq!(audio_shape(&twice, SUBSAMPLE_CONV2D), vec![4, 3, 3, 1]);
    }
}
