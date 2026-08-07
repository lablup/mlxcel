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

//! Falcon-OCR loader (`model_type: falcon_ocr`).
//!
//! The checkpoint uses the original Falcon naming (`tok_embeddings`,
//! `layers.N.attention.*`, `layers.N.feed_forward.*`, `output`) rather than the
//! HF `model.layers.N.self_attn` scheme, and ships the gate/up projection fused
//! and row-interleaved. Both are resolved here so the decoder sees an ordinary
//! weight map.

use anyhow::Result;
use std::path::Path;

use crate::LoadedModel;
use crate::models::falcon_ocr::{FalconOcrConfig, FalconOcrTextModel};
use crate::vision;
use crate::vision::processors::falcon_ocr::FalconOcrProcessor;
use mlxcel_core::weights::WeightMap;

use super::{load_vlm_weights_common, parse_vlm_config, read_sanitized_vlm_config};

/// Default ids in the shipped tokenizer, used only when `tokenizer.json`
/// cannot be read.
const DEFAULT_OCR_PLAIN_TOKEN_ID: i32 = 257;
const DEFAULT_END_OF_QUERY_TOKEN_ID: i32 = 263;

/// Split the fused, row-interleaved gate/up projection.
///
/// `feed_forward.w13.weight` is `[2 * ffn_dim, dim]` with gate and up rows
/// alternating: row `2i` is gate row `i`, row `2i + 1` is up row `i`. That
/// ordering is what the reference triton kernel reads
/// (`gate_idx = 2 * col`, `up_idx = 2 * col + 1`) and what mlx-vlm's `sanitize`
/// undoes with `concatenate([v[0::2], v[1::2]])`.
///
/// The split is applied to every tensor under a `w13` prefix, not just
/// `.weight`, so an affine-quantized conversion (whose `scales` / `biases`
/// share the row axis, with groups running along the input axis) round-trips
/// correctly too.
pub(crate) fn sanitize_falcon_ocr_weights(weights: WeightMap) -> Result<WeightMap> {
    let mut out = WeightMap::new();
    for (key, value) in weights {
        let Some((prefix, suffix)) = split_w13_key(&key) else {
            out.insert(key, value);
            continue;
        };
        let shape = mlxcel_core::array_shape(&value);
        if shape.len() != 2 || shape[0] % 2 != 0 {
            return Err(anyhow::anyhow!(
                "Falcon-OCR {key} must be a 2-D tensor with an even row count, got {shape:?}"
            ));
        }
        let (rows, cols) = (shape[0] / 2, shape[1]);
        let paired = mlxcel_core::reshape(&value, &[rows, 2, cols]);
        let gate = mlxcel_core::reshape(
            &mlxcel_core::slice(&paired, &[0, 0, 0], &[rows, 1, cols]),
            &[rows, cols],
        );
        let up = mlxcel_core::reshape(
            &mlxcel_core::slice(&paired, &[0, 1, 0], &[rows, 2, cols]),
            &[rows, cols],
        );
        out.insert(format!("{prefix}.w1.{suffix}"), gate);
        out.insert(format!("{prefix}.w3.{suffix}"), up);
    }
    Ok(out)
}

/// Split `<prefix>.w13.<suffix>` into its two halves.
fn split_w13_key(key: &str) -> Option<(&str, &str)> {
    let idx = key.find(".w13.")?;
    Some((&key[..idx], &key[idx + ".w13.".len()..]))
}

/// Resolve a special token id from the shipped tokenizer.
///
/// Neither `<|OCR_PLAIN|>` nor `<|end_of_query|>` appears in `config.json`, and
/// both are load-bearing: without the task token the model describes the page
/// instead of transcribing it, and without the query terminator in the stop set
/// the decode runs one step past the end and prints the marker.
///
/// `tokenizer_config.json` names the token (`ocr_plain_token`,
/// `end_of_query_token`); `tokenizer.json` maps that name to an id.
fn read_special_token_id(model_path: &Path, config_key: &str, fallback_name: &str) -> Option<i32> {
    let name = std::fs::read_to_string(model_path.join("tokenizer_config.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| {
            v.get(config_key)
                .and_then(|t| t.as_str())
                .map(str::to_string)
        })
        .unwrap_or_else(|| fallback_name.to_string());

    let tokenizer = std::fs::read_to_string(model_path.join("tokenizer.json")).ok()?;
    let tokenizer: serde_json::Value = serde_json::from_str(&tokenizer).ok()?;
    tokenizer
        .get("added_tokens")?
        .as_array()?
        .iter()
        .find(|t| t.get("content").and_then(|c| c.as_str()) == Some(name.as_str()))
        .and_then(|t| t.get("id"))
        .and_then(|id| id.as_i64())
        .map(|id| id as i32)
}

pub(crate) fn load_falcon_ocr_vl(model_path: &Path) -> Result<LoadedModel> {
    let (config_str, _full_config) = read_sanitized_vlm_config(model_path)?;
    let config: FalconOcrConfig = parse_vlm_config(&config_str, "Falcon-OCR config")?;

    let weights = sanitize_falcon_ocr_weights(load_vlm_weights_common(model_path, None)?)?;
    let text_model = FalconOcrTextModel::from_weights(&weights, &config)
        .map_err(|e| anyhow::anyhow!("Failed to load Falcon-OCR model: {}", e))?;

    // The reference stops on `[eos_id, <|end_of_query|>]`; the OCR head emits
    // the query terminator first, so leaving it out prints the marker and burns
    // an extra decode step.
    let mut eos_token_ids = crate::loading::read_eos_token_ids(model_path);
    if eos_token_ids.is_empty() {
        eos_token_ids = vec![config.eos_id];
    }
    let end_of_query = read_special_token_id(model_path, "end_of_query_token", "<|end_of_query|>")
        .unwrap_or(DEFAULT_END_OF_QUERY_TOKEN_ID);
    if !eos_token_ids.contains(&end_of_query) {
        eos_token_ids.push(end_of_query);
    }

    let processor = FalconOcrProcessor::new(config.spatial_patch_size as u32, config.channel_size);
    let ocr_task_token_id = read_special_token_id(model_path, "ocr_plain_token", "<|OCR_PLAIN|>")
        .or(Some(DEFAULT_OCR_PLAIN_TOKEN_ID));

    Ok(LoadedModel::FalconOcrVL(vision::FalconOcrVlModel {
        text_model,
        processor,
        ocr_task_token_id,
        eos_token_ids,
    }))
}

#[cfg(test)]
#[path = "vlm_falcon_ocr_tests.rs"]
mod tests;
