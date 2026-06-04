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

//! Gemma 4 Unified (`gemma4_unified`) loader.
//!
//! Encoder-free multimodal loader: text backbone (shared Gemma 4 transformer) +
//! patch-projection vision embedder + waveform-chunk audio path. Kept separate
//! from `vlm_gemma.rs` (which hosts the ViT/Conformer-backed Gemma 3/3n/4 VLM
//! loaders) so each file stays under the project's 500-line cap.

use anyhow::Result;
use std::path::Path;

use crate::LoadedModel;
use crate::models;

use super::{parse_vlm_config, read_sanitized_vlm_config};

/// Sanitize `gemma4_unified` checkpoint weights to module param names.
///
/// Mirrors the upstream `gemma4_unified` `sanitize` (issue §7):
/// * drop `*rotary_emb*` (rotary inv-freq buffers) and `lm_head.weight`
///   (output is tied to `embed_tokens`);
/// * strip a leading `model.` prefix, then normalize `language_model.<x>` →
///   `language_model.model.<x>` unless it already starts with
///   `language_model.model.`;
/// * split fused MoE experts `…experts.gate_up_proj` into
///   `…experts.switch_glu.gate_proj.weight` / `…up_proj.weight` (swap the last
///   two axes, split the doubled dim in half) and rename
///   `…experts.down_proj` → `…experts.switch_glu.down_proj.weight`;
/// * drop `embed_audio*` when `has_audio` is false.
pub(crate) fn sanitize_gemma4_unified_weights(
    raw: mlxcel_core::weights::WeightMap,
    has_audio: bool,
) -> mlxcel_core::weights::WeightMap {
    let mut out = mlxcel_core::weights::WeightMap::new();

    for (key, value) in raw {
        // Drop rotary buffers and the (tied) lm_head weight.
        if key.contains("rotary_emb") || key == "lm_head.weight" || key.starts_with("lm_head.") {
            continue;
        }
        // Drop audio embedder tensors when the model has no audio config.
        if !has_audio && key.starts_with("embed_audio") {
            continue;
        }

        // Fused MoE experts: split gate_up_proj, rename down_proj. Only the
        // non-quantized `.weight` path is fused in upstream Gemma 4 Unified
        // MoE checkpoints; quantized scales/biases follow the same split when
        // present.
        if key.ends_with("experts.gate_up_proj") {
            let prefix = key.trim_end_matches("gate_up_proj");
            // value: [num_experts, in, 2*ffn]; swap last two axes →
            // [num_experts, 2*ffn, in], then split the doubled dim in half.
            let swapped = mlxcel_core::transpose_axes(&value, &[0, 2, 1]);
            let shape = mlxcel_core::array_shape(&swapped);
            let doubled = shape[1];
            let half = doubled / 2;
            let gate = mlxcel_core::slice(&swapped, &[0, 0, 0], &[shape[0], half, shape[2]]);
            let up = mlxcel_core::slice(&swapped, &[0, half, 0], &[shape[0], doubled, shape[2]]);
            out.insert(format!("{prefix}switch_glu.gate_proj.weight"), gate);
            out.insert(format!("{prefix}switch_glu.up_proj.weight"), up);
            continue;
        }
        if key.ends_with("experts.down_proj") {
            let prefix = key.trim_end_matches("down_proj");
            out.insert(format!("{prefix}switch_glu.down_proj.weight"), value);
            continue;
        }

        // Prefix normalization: strip a leading `model.`, then ensure the
        // language model lives under `language_model.model.`.
        let normalized = normalize_gemma4_unified_key(&key);
        out.insert(normalized, value);
    }

    out
}

/// Normalize a single `gemma4_unified` weight key (prefix handling only).
fn normalize_gemma4_unified_key(key: &str) -> String {
    // `model.language_model.X` → `language_model.model.X`.
    if let Some(rest) = key.strip_prefix("model.language_model.") {
        if rest.starts_with("model.") {
            return format!("language_model.{rest}");
        }
        return format!("language_model.model.{rest}");
    }
    // `model.embed_vision.X` / `model.vision_embedder.X` / `model.embed_audio.X`
    // → drop the leading `model.`.
    for mm_prefix in ["embed_vision.", "vision_embedder.", "embed_audio."] {
        if let Some(rest) = key.strip_prefix(&format!("model.{mm_prefix}")) {
            return format!("{mm_prefix}{rest}");
        }
    }
    // Bare `language_model.X` (not already `language_model.model.X`) →
    // `language_model.model.X`.
    if let Some(rest) = key.strip_prefix("language_model.")
        && !rest.starts_with("model.")
    {
        return format!("language_model.model.{rest}");
    }
    key.to_string()
}

/// Load a Gemma 4 Unified (`gemma4_unified`) multimodal model.
pub(crate) fn load_gemma4_unified(model_path: &Path) -> Result<LoadedModel> {
    use crate::vision::encoders::gemma4_unified::Gemma4UnifiedVisionEmbedder;
    use crate::vision::gemma4_multimodal_embedder::Gemma4MultimodalEmbedder;
    use crate::vision::gemma4_unified::Gemma4UnifiedModel;
    use crate::vision::gemma4_unified_config::Gemma4UnifiedConfig;
    use crate::vision::processors::gemma4_unified::Gemma4UnifiedProcessor;

    let (config_str, full_config) = read_sanitized_vlm_config(model_path)?;
    let unified_config: Gemma4UnifiedConfig =
        parse_vlm_config(&config_str, "Gemma4 Unified config")?;
    let has_audio = unified_config.audio_config.is_some();

    // Reuse the Gemma 4 text-config parse path (quantization inheritance,
    // layer_types defaults, etc.) by wrapping the full config as ModelArgs.
    let text_args: models::gemma4::ModelArgs =
        parse_vlm_config(&config_str, "Gemma4 Unified text config")?;
    let text_config = text_args.text_args();

    let (raw_weights, weight_backing) =
        models::load_gemma4_unified_weights_with_backing(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load Gemma4 Unified weights: {}", e))?;
    let mut weights = sanitize_gemma4_unified_weights(raw_weights, has_audio);
    models::strip_gemma4_kv_shared_weights(&mut weights, &full_config);
    models::sanitize_tied_embeddings(&mut weights, &full_config);

    let text_model = models::Gemma4Model::from_weights(&weights, &text_args)
        .map_err(|e| anyhow::anyhow!("Failed to load Gemma4 Unified text model: {}", e))?;

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

    let vision_config = &unified_config.vision_config;
    let vision_embedder = Gemma4UnifiedVisionEmbedder::from_weights(
        &weights,
        "vision_embedder",
        vision_config,
        group_size,
        bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load Gemma4 Unified vision embedder: {}", e))?;

    // embed_vision: output_proj_dims (== mm_embed_dim) → text hidden.
    let embed_vision = Gemma4MultimodalEmbedder::from_weights(
        &weights,
        "embed_vision",
        vision_config.output_proj_dims,
        vision_config.rms_norm_eps,
        group_size,
        bits,
    )
    .map_err(|e| anyhow::anyhow!("Failed to load Gemma4 Unified vision projector: {}", e))?;

    let processor = Gemma4UnifiedProcessor::new(
        vision_config.model_patch_size,
        vision_config.num_soft_tokens,
        unified_config
            .audio_config
            .as_ref()
            .map(|a| a.audio_samples_per_token)
            .unwrap_or(640),
    );

    let mut model = Gemma4UnifiedModel::new(
        models::Gemma4Wrapper::new(text_model),
        vision_embedder,
        embed_vision,
        processor,
        unified_config.image_token_id,
        unified_config.video_token_id,
        unified_config.boi_token_id,
        unified_config.eoi_token_id,
    );
    model.set_weight_backing(weight_backing);

    // Audio feature embedder: output_proj_dims (== audio_embed_dim, 640) →
    // text hidden. Only when audio_config + weights are present.
    if let Some(audio_config) = &unified_config.audio_config
        && weights.contains_key("embed_audio.embedding_projection.weight")
    {
        let embed_audio = Gemma4MultimodalEmbedder::from_weights(
            &weights,
            "embed_audio",
            audio_config.output_proj_dims,
            audio_config.rms_norm_eps,
            group_size,
            bits,
        )
        .map_err(|e| anyhow::anyhow!("Failed to load Gemma4 Unified audio projector: {}", e))?;
        model.set_audio(
            embed_audio,
            unified_config.audio_token_id,
            unified_config.boa_token_id,
            unified_config.resolve_eoa_token_id(),
        );
        eprintln!("Loaded Gemma4 Unified audio embedder (encoder-free waveform path)");
    }

    Ok(LoadedModel::Gemma4Unified(model))
}

#[cfg(test)]
#[path = "vlm_gemma_unified_tests.rs"]
mod tests;
