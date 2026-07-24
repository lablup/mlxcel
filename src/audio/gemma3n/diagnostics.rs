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

//! Audio-only MLX oracle used by the opt-in Gemma3n XLA first-divergence gate.

use std::path::Path;

use mlxcel_core::layers::{RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};

use super::{GEMMA3N_AUDIO_SOFT_TOKENS, Gemma3nAudioConfig, Gemma3nAudioEncoder};
use crate::models::gemma3n::{Gemma3nAudioEmbedder, ModelArgs, TextConfig};

#[derive(Debug, Clone)]
pub struct Gemma3nAudioMlxDiagnosticOutput {
    pub sscp_conv_0: Vec<f32>,
    pub encoded_reduced: Vec<f32>,
    pub soft_norm: Vec<f32>,
    pub soft_linear: Vec<f32>,
    pub soft_post_norm: Vec<f32>,
    pub hard_embedding: Vec<f32>,
    pub hard_norm: Vec<f32>,
    pub hard_linear: Vec<f32>,
    pub hard_post_norm: Vec<f32>,
    pub projected_audio: Vec<f32>,
    pub hard_audio: Vec<f32>,
    pub embeddings: Vec<f32>,
    pub dense_ple: Vec<f32>,
    pub projected_lengths: Vec<usize>,
    pub hidden_size: usize,
    pub layers: usize,
    pub hidden_per_layer: usize,
}

fn canonical_name(name: &str) -> &str {
    name.strip_prefix("model.").unwrap_or(name)
}

fn keep_diagnostic_weight(name: &str) -> bool {
    let name = canonical_name(name);
    name.starts_with("audio_tower.")
        || name.starts_with("embed_audio.")
        || [
            "language_model.embed_tokens.",
            "language_model.embed_tokens_per_layer.",
            "language_model.per_layer_model_projection.",
            "language_model.per_layer_projection_norm.",
            "language_model.model.embed_tokens.",
            "language_model.model.embed_tokens_per_layer.",
            "language_model.model.per_layer_model_projection.",
            "language_model.model.per_layer_projection_norm.",
        ]
        .iter()
        .any(|prefix| name.starts_with(prefix))
}

fn load_diagnostic_weights(model_dir: &Path) -> Result<WeightMap, String> {
    let raw =
        mlxcel_core::weights::load_weights_from_dir_filtered(model_dir, keep_diagnostic_weight)?;
    let mut weights = WeightMap::new();
    for (name, value) in raw {
        weights.insert(canonical_name(&name).to_string(), mlxcel_core::copy(&value));
    }
    Ok(weights)
}

fn read_config(model_dir: &Path) -> Result<(TextConfig, Gemma3nAudioConfig), String> {
    let path = model_dir.join("config.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    let root: ModelArgs = serde_json::from_str(&text)
        .map_err(|error| format!("parse Gemma3n config {}: {error}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| format!("parse Gemma3n config {}: {error}", path.display()))?;
    let audio = value
        .get("audio_config")
        .ok_or_else(|| format!("{} has no audio_config", path.display()))
        .and_then(|value| {
            serde_json::from_value(value.clone())
                .map_err(|error| format!("parse Gemma3n audio_config: {error}"))
        })?;
    Ok((root.text_args(), audio))
}

fn array_f32(array: &MlxArray) -> Vec<f32> {
    let values = mlxcel_core::astype(array, mlxcel_core::dtype::FLOAT32);
    mlxcel_core::eval(&values);
    mlxcel_core::array_to_raw_bytes(&values)
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("f32 byte width")))
        .collect()
}

fn copy_weight(weights: &WeightMap, key: &str) -> Result<UniquePtr<MlxArray>, String> {
    weights
        .get(key)
        .map(|weight| mlxcel_core::copy(weight))
        .ok_or_else(|| format!("Gemma3n diagnostic weight not found: {key}"))
}

fn language_prefix(weights: &WeightMap) -> &'static str {
    if weights.contains_key("language_model.model.embed_tokens.weight") {
        "language_model.model"
    } else {
        "language_model"
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run_gemma3n_audio_mlx_diagnostics(
    model_dir: &Path,
    mel: &[f32],
    valid_mask: &[u8],
    frame_bucket: usize,
    clips: usize,
    token_ids: &[i32],
    audio_token_id: i32,
    context_capacity: usize,
) -> Result<Gemma3nAudioMlxDiagnosticOutput, String> {
    if token_ids.is_empty() || token_ids.len() > context_capacity {
        return Err(format!(
            "Gemma3n diagnostic token length {} is outside 1..={context_capacity}",
            token_ids.len()
        ));
    }
    let (text, audio) = read_config(model_dir)?;
    let expected_mel = clips
        .checked_mul(frame_bucket)
        .and_then(|value| value.checked_mul(audio.input_feat_size))
        .ok_or_else(|| "Gemma3n diagnostic mel size overflow".to_string())?;
    if mel.len() != expected_mel || valid_mask.len() != clips * frame_bucket {
        return Err(format!(
            "Gemma3n diagnostic input lengths {}/{} do not match clips/bucket/features {clips}/{frame_bucket}/{}",
            mel.len(),
            valid_mask.len(),
            audio.input_feat_size
        ));
    }

    let weights = load_diagnostic_weights(model_dir)?;
    let group_size = text
        .quantization
        .as_ref()
        .map(|quant| quant.group_size as i32)
        .unwrap_or(64);
    let bits = text
        .quantization
        .as_ref()
        .map(|quant| quant.bits as i32)
        .unwrap_or(4);
    let tower =
        Gemma3nAudioEncoder::from_weights(&weights, "audio_tower", &audio, group_size, bits)?;
    let embed_audio = Gemma3nAudioEmbedder::from_weights(
        &weights,
        "embed_audio",
        &audio,
        text.hidden_size,
        group_size,
        bits,
    )?;

    let mel = mlxcel_core::from_slice_f32(
        mel,
        &[
            clips as i32,
            frame_bucket as i32,
            audio.input_feat_size as i32,
        ],
    );
    let invalid = valid_mask
        .iter()
        .map(|valid| i32::from(*valid == 0))
        .collect::<Vec<_>>();
    let invalid = mlxcel_core::from_slice_i32(&invalid, &[clips as i32, frame_bucket as i32]);
    let invalid = mlxcel_core::astype(&invalid, mlxcel_core::dtype::BOOL);
    let (encoded, encoded_invalid, sscp_conv_0) =
        tower.forward_with_sscp_conv0_diagnostic(&mel, &invalid)?;
    let encoded_shape = mlxcel_core::array_shape(&encoded);
    let projected_lengths = (0..clips)
        .map(|clip| {
            let row = mlxcel_core::slice(
                &encoded_invalid,
                &[clip as i32, 0],
                &[(clip + 1) as i32, encoded_shape[1]],
            );
            let valid = mlxcel_core::logical_not(&row);
            let valid = mlxcel_core::astype(&valid, mlxcel_core::dtype::INT32);
            mlxcel_core::item_i32(&mlxcel_core::sum_all(&valid)) as usize
        })
        .collect::<Vec<_>>();

    let padding = embed_audio.padding_embedding();
    let (soft_norm, soft_linear, soft_post_norm) =
        embed_audio.diagnostic_soft_projection_stages(&encoded);
    let mut projected = mlxcel_core::copy(&soft_post_norm);
    let projected_shape = mlxcel_core::array_shape(&projected);
    let invalid = mlxcel_core::reshape(
        &encoded_invalid,
        &[projected_shape[0], projected_shape[1], 1],
    );
    projected = mlxcel_core::where_cond(&invalid, &padding, &projected);
    let missing = GEMMA3N_AUDIO_SOFT_TOKENS as i32 - projected_shape[1];
    if missing < 0 {
        return Err(format!(
            "Gemma3n MLX diagnostic projected {} rows, exceeding {}",
            projected_shape[1], GEMMA3N_AUDIO_SOFT_TOKENS
        ));
    }
    if missing > 0 {
        let tail =
            mlxcel_core::broadcast_to(&padding, &[clips as i32, missing, text.hidden_size as i32]);
        projected = mlxcel_core::concatenate(&projected, &tail, 1);
    }
    let (hard_embedding, hard_norm, hard_linear, hard_post_norm) =
        embed_audio.diagnostic_hard_projection_stages();

    let language = language_prefix(&weights);
    let text_embedding = UnifiedEmbedding::from_weights(
        &weights,
        &format!("{language}.embed_tokens"),
        group_size,
        bits,
    )?;
    let token_ple = UnifiedEmbedding::from_weights(
        &weights,
        &format!("{language}.embed_tokens_per_layer"),
        group_size,
        bits,
    )?;
    let model_projection = UnifiedLinear::from_weights(
        &weights,
        &format!("{language}.per_layer_model_projection"),
        group_size,
        bits,
    )?;
    let projection_norm = RMSNorm::new(
        copy_weight(
            &weights,
            &format!("{language}.per_layer_projection_norm.weight"),
        )?,
        text.rms_norm_eps,
    );

    let mut padded_tokens = token_ids.to_vec();
    padded_tokens.resize(context_capacity, 0);
    let token_array = mlxcel_core::from_slice_i32(&padded_tokens, &[1, context_capacity as i32]);
    let scaled_text = mlxcel_core::multiply_scalar(
        &text_embedding.forward(&token_array),
        (text.hidden_size as f32).sqrt(),
    );
    let hard_merged = embed_audio.merge_hard_tokens(&token_array, &scaled_text);
    let projected = mlxcel_core::reshape(
        &projected,
        &[
            clips as i32 * GEMMA3N_AUDIO_SOFT_TOKENS as i32,
            text.hidden_size as i32,
        ],
    );
    let merged =
        crate::vision::merge::merge_llava(audio_token_id, &projected, &hard_merged, &token_array)
            .inputs_embeds;

    let per_layer_limit =
        mlxcel_core::from_slice_i32(&[text.vocab_size_per_layer_input as i32], &[1]);
    let per_layer_mask = mlxcel_core::less(&token_array, &per_layer_limit);
    let per_layer_tokens = mlxcel_core::where_cond(
        &per_layer_mask,
        &token_array,
        &mlxcel_core::zeros(&[1, context_capacity as i32], mlxcel_core::dtype::INT32),
    );
    let token_ple = mlxcel_core::multiply_scalar(
        &token_ple.forward(&per_layer_tokens),
        (text.hidden_size_per_layer_input as f32).sqrt(),
    );
    let token_ple = mlxcel_core::reshape(
        &token_ple,
        &[
            context_capacity as i32,
            text.num_hidden_layers as i32,
            text.hidden_size_per_layer_input as i32,
        ],
    );
    let projected_ple = model_projection.forward(&merged);
    let projected_ple =
        mlxcel_core::multiply_scalar(&projected_ple, (text.hidden_size as f32).powf(-0.5));
    let projected_ple = mlxcel_core::reshape(
        &projected_ple,
        &[
            context_capacity as i32,
            text.num_hidden_layers as i32,
            text.hidden_size_per_layer_input as i32,
        ],
    );
    let projected_ple = projection_norm.forward(&projected_ple);
    let dense_ple = mlxcel_core::multiply_scalar(
        &mlxcel_core::add(&projected_ple, &token_ple),
        std::f32::consts::FRAC_1_SQRT_2,
    );

    let mut embeddings = array_f32(&merged);
    embeddings[token_ids.len() * text.hidden_size..].fill(0.0);
    let mut dense_ple = array_f32(&dense_ple);
    dense_ple[token_ids.len() * text.num_hidden_layers * text.hidden_size_per_layer_input..]
        .fill(0.0);

    Ok(Gemma3nAudioMlxDiagnosticOutput {
        sscp_conv_0: array_f32(&sscp_conv_0),
        encoded_reduced: array_f32(&encoded),
        soft_norm: array_f32(&soft_norm),
        soft_linear: array_f32(&soft_linear),
        soft_post_norm: array_f32(&soft_post_norm),
        hard_embedding: array_f32(&hard_embedding),
        hard_norm: array_f32(&hard_norm),
        hard_linear: array_f32(&hard_linear),
        hard_post_norm: array_f32(&hard_post_norm),
        projected_audio: array_f32(&projected),
        hard_audio: array_f32(&hard_post_norm),
        embeddings,
        dense_ple,
        projected_lengths,
        hidden_size: text.hidden_size,
        layers: text.num_hidden_layers,
        hidden_per_layer: text.hidden_size_per_layer_input,
    })
}
