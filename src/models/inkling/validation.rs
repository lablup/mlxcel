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

use mlxcel_core::weights::WeightMap;

use super::InklingConfig;
use super::validation_shapes::{conv, exact, expert, matrix, vector};

pub(super) fn validate_config(config: &InklingConfig) -> Result<(), String> {
    let text = &config.text_config;
    for (name, value) in [
        ("hidden_size", text.hidden_size),
        ("num_hidden_layers", text.num_hidden_layers),
        ("vocab_size", text.vocab_size),
        ("num_attention_heads", text.num_attention_heads),
        ("num_key_value_heads", text.num_key_value_heads),
        ("head_dim", text.head_dim),
        ("swa_num_attention_heads", text.swa_num_attention_heads),
        ("swa_num_key_value_heads", text.swa_num_key_value_heads),
        ("swa_head_dim", text.swa_head_dim),
        ("sliding_window_size", text.sliding_window_size),
        ("d_rel", text.d_rel),
        ("rel_extent", text.rel_extent),
        ("sconv_kernel_size", text.sconv_kernel_size),
        ("n_routed_experts", text.n_routed_experts),
        ("n_shared_experts", text.n_shared_experts),
    ] {
        if value == 0 {
            return Err(format!("Inkling {name} must be positive"));
        }
    }
    if !text
        .num_attention_heads
        .is_multiple_of(text.num_key_value_heads)
        || !text
            .swa_num_attention_heads
            .is_multiple_of(text.swa_num_key_value_heads)
    {
        return Err("Inkling query-head counts must be divisible by KV-head counts".into());
    }
    if text.num_experts_per_tok == 0 || text.num_experts_per_tok > text.n_routed_experts {
        return Err("Inkling num_experts_per_tok must be in 1..=n_routed_experts".into());
    }
    if text.dense_mlp_idx > text.num_hidden_layers {
        return Err("Inkling dense_mlp_idx exceeds num_hidden_layers".into());
    }
    if let Some(types) = &text.layer_types
        && types.len() != text.num_hidden_layers
    {
        return Err("Inkling layer_types length must equal num_hidden_layers".into());
    }
    if text.layer_types.as_ref().is_some_and(|types| {
        types
            .iter()
            .any(|kind| !matches!(kind.as_str(), "hybrid_sliding" | "hybrid"))
    }) {
        return Err("Inkling layer_types entries must be 'hybrid_sliding' or 'hybrid'".into());
    }
    if let Some(types) = &text.mlp_layer_types
        && types.len() != text.num_hidden_layers
    {
        return Err("Inkling mlp_layer_types length must equal num_hidden_layers".into());
    }
    if text.mlp_layer_types.as_ref().is_some_and(|types| {
        types
            .iter()
            .any(|kind| !matches!(kind.as_str(), "dense" | "sparse"))
    }) {
        return Err("Inkling mlp_layer_types entries must be 'dense' or 'sparse'".into());
    }
    if text
        .local_layer_ids
        .as_ref()
        .is_some_and(|ids| ids.iter().any(|&i| i >= text.num_hidden_layers))
    {
        return Err("Inkling local_layer_ids contains an out-of-range layer".into());
    }
    let (dense, moe) = text.widths()?;
    if dense == 0 || moe == 0 {
        return Err("Inkling MLP widths must be positive".into());
    }
    for (name, value) in [
        ("rms_norm_eps", text.rms_norm_eps),
        (
            "logits_mup_width_multiplier",
            text.logits_mup_width_multiplier,
        ),
        ("route_scale", text.route_scale),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(format!("Inkling {name} must be finite and positive"));
        }
    }
    if !text.log_scaling_alpha.is_finite() || text.log_scaling_alpha < 0.0 {
        return Err("Inkling log_scaling_alpha must be finite and non-negative".into());
    }
    if text.log_scaling_n_floor == Some(0) {
        return Err("Inkling log_scaling_n_floor must be positive when present".into());
    }
    if text
        .unpadded_vocab_size
        .is_some_and(|v| v > text.vocab_size)
    {
        return Err("Inkling unpadded_vocab_size exceeds vocab_size".into());
    }
    for (name, value) in [
        ("hidden_size", text.hidden_size),
        ("vocab_size", text.vocab_size),
        ("dense_width", dense),
        ("moe_width", moe),
        ("num_attention_heads", text.num_attention_heads),
        ("num_key_value_heads", text.num_key_value_heads),
        ("swa_num_attention_heads", text.swa_num_attention_heads),
        ("swa_num_key_value_heads", text.swa_num_key_value_heads),
        ("head_dim", text.head_dim),
        ("swa_head_dim", text.swa_head_dim),
        ("d_rel", text.d_rel),
        ("n_routed_experts", text.n_routed_experts),
        ("n_shared_experts", text.n_shared_experts),
    ] {
        if i32::try_from(value).is_err() {
            return Err(format!(
                "Inkling {name} exceeds the MLX i32 dimension limit"
            ));
        }
    }
    for (name, lhs, rhs) in [
        (
            "global query width",
            text.num_attention_heads,
            text.head_dim,
        ),
        ("global KV width", text.num_key_value_heads, text.head_dim),
        (
            "sliding query width",
            text.swa_num_attention_heads,
            text.swa_head_dim,
        ),
        (
            "sliding KV width",
            text.swa_num_key_value_heads,
            text.swa_head_dim,
        ),
        (
            "relative projection width",
            text.num_attention_heads,
            text.d_rel,
        ),
        (
            "sliding relative projection width",
            text.swa_num_attention_heads,
            text.d_rel,
        ),
        ("shared expert width", text.n_shared_experts, moe),
    ] {
        if lhs
            .checked_mul(rhs)
            .and_then(|value| i32::try_from(value).ok())
            .is_none()
        {
            return Err(format!(
                "Inkling {name} exceeds the MLX i32 dimension limit"
            ));
        }
    }
    if text
        .n_routed_experts
        .checked_add(text.n_shared_experts)
        .and_then(|value| i32::try_from(value).ok())
        .is_none()
    {
        return Err("Inkling total expert count exceeds the MLX i32 dimension limit".into());
    }
    let (group, bits, _) = config.quantization();
    mlxcel_core::layers::validate_quantization_params(group, bits)
        .map_err(|e| format!("Inkling quantization: {e}"))?;
    if let Some(audio) = &config.audio_config {
        audio.validate(text.hidden_size)?;
    }
    Ok(())
}

pub(super) fn validate_weight_shapes(
    weights: &WeightMap,
    config: &InklingConfig,
) -> Result<(), String> {
    let text = &config.text_config;
    let (group_size, bits, _) = config.quantization();
    matrix(
        weights,
        "model.embed_tokens",
        text.vocab_size,
        text.hidden_size,
        group_size,
        bits,
    )?;
    vector(weights, "model.norm.weight", text.hidden_size)?;
    if text.use_embed_norm {
        vector(weights, "model.embed_norm.weight", text.hidden_size)?;
    }
    if !text.tie_word_embeddings {
        matrix(
            weights,
            "lm_head",
            text.vocab_size,
            text.hidden_size,
            group_size,
            bits,
        )?;
    }
    if let Some(audio) = &config.audio_config {
        let audio_vocab = audio
            .n_mel_bins
            .checked_mul(audio.mel_vocab_size)
            .ok_or_else(|| "Inkling audio embedding vocabulary overflowed usize".to_string())?;
        matrix(
            weights,
            "audio_tower.embed_audio_tokens",
            audio_vocab,
            text.hidden_size,
            group_size,
            bits,
        )?;
        vector(weights, "audio_tower.norm.weight", text.hidden_size)?;
    }
    let (dense_width, moe_width) = text.widths()?;
    for index in 0..text.num_hidden_layers {
        let layer = format!("model.layers.{index}");
        vector(
            weights,
            &format!("{layer}.input_layernorm.weight"),
            text.hidden_size,
        )?;
        vector(
            weights,
            &format!("{layer}.post_attention_layernorm.weight"),
            text.hidden_size,
        )?;
        conv(
            weights,
            &format!("{layer}.attn_sconv.conv.weight"),
            text.hidden_size,
            text.sconv_kernel_size,
        )?;
        conv(
            weights,
            &format!("{layer}.mlp_sconv.conv.weight"),
            text.hidden_size,
            text.sconv_kernel_size,
        )?;
        let sliding = text.layer_is_sliding(index);
        let (heads, kv_heads, head_dim, extent) = if sliding {
            (
                text.swa_num_attention_heads,
                text.swa_num_key_value_heads,
                text.swa_head_dim,
                text.sliding_window_size,
            )
        } else {
            (
                text.num_attention_heads,
                text.num_key_value_heads,
                text.head_dim,
                text.rel_extent,
            )
        };
        let attn = format!("{layer}.self_attn");
        matrix(
            weights,
            &format!("{attn}.q_proj"),
            heads * head_dim,
            text.hidden_size,
            group_size,
            bits,
        )?;
        matrix(
            weights,
            &format!("{attn}.k_proj"),
            kv_heads * head_dim,
            text.hidden_size,
            group_size,
            bits,
        )?;
        matrix(
            weights,
            &format!("{attn}.v_proj"),
            kv_heads * head_dim,
            text.hidden_size,
            group_size,
            bits,
        )?;
        matrix(
            weights,
            &format!("{attn}.r_proj"),
            heads * text.d_rel,
            text.hidden_size,
            group_size,
            bits,
        )?;
        matrix(
            weights,
            &format!("{attn}.o_proj"),
            text.hidden_size,
            heads * head_dim,
            group_size,
            bits,
        )?;
        vector(weights, &format!("{attn}.q_norm.weight"), head_dim)?;
        vector(weights, &format!("{attn}.k_norm.weight"), head_dim)?;
        conv(
            weights,
            &format!("{attn}.k_sconv.conv.weight"),
            kv_heads * head_dim,
            text.sconv_kernel_size,
        )?;
        conv(
            weights,
            &format!("{attn}.v_sconv.conv.weight"),
            kv_heads * head_dim,
            text.sconv_kernel_size,
        )?;
        exact(
            weights,
            &format!("{attn}.rel_proj"),
            &[text.d_rel as i32, extent as i32],
        )?;
        let mlp = format!("{layer}.mlp");
        exact(weights, &format!("{mlp}.global_scale"), &[1])?;
        if text.layer_is_dense(index) {
            matrix(
                weights,
                &format!("{mlp}.gate_proj"),
                dense_width,
                text.hidden_size,
                group_size,
                bits,
            )?;
            matrix(
                weights,
                &format!("{mlp}.up_proj"),
                dense_width,
                text.hidden_size,
                group_size,
                bits,
            )?;
            matrix(
                weights,
                &format!("{mlp}.down_proj"),
                text.hidden_size,
                dense_width,
                group_size,
                bits,
            )?;
        } else {
            exact(
                weights,
                &format!("{mlp}.gate_weight"),
                &[
                    (text.n_routed_experts + text.n_shared_experts) as i32,
                    text.hidden_size as i32,
                ],
            )?;
            exact(
                weights,
                &format!("{mlp}.e_score_correction_bias"),
                &[text.n_routed_experts as i32],
            )?;
            expert(
                weights,
                &format!("{mlp}.switch_mlp.gate_proj"),
                text.n_routed_experts,
                moe_width,
                text.hidden_size,
                group_size,
                bits,
            )?;
            expert(
                weights,
                &format!("{mlp}.switch_mlp.up_proj"),
                text.n_routed_experts,
                moe_width,
                text.hidden_size,
                group_size,
                bits,
            )?;
            expert(
                weights,
                &format!("{mlp}.switch_mlp.down_proj"),
                text.n_routed_experts,
                text.hidden_size,
                moe_width,
                group_size,
                bits,
            )?;
            matrix(
                weights,
                &format!("{mlp}.shared_experts.gate_proj"),
                text.n_shared_experts * moe_width,
                text.hidden_size,
                group_size,
                bits,
            )?;
            matrix(
                weights,
                &format!("{mlp}.shared_experts.up_proj"),
                text.n_shared_experts * moe_width,
                text.hidden_size,
                group_size,
                bits,
            )?;
            matrix(
                weights,
                &format!("{mlp}.shared_experts.down_proj"),
                text.hidden_size,
                text.n_shared_experts * moe_width,
                group_size,
                bits,
            )?;
            let switch = format!("{mlp}.switch_mlp");
            if weights.contains_key(&format!("{switch}.gate_proj.scales"))
                && !weights.contains_key(&format!("{switch}.gate_proj.biases"))
            {
                exact(
                    weights,
                    &format!("{switch}.gate_scale"),
                    &[text.n_routed_experts as i32],
                )?;
                exact(
                    weights,
                    &format!("{switch}.out_scale"),
                    &[text.n_routed_experts as i32],
                )?;
            }
        }
    }
    Ok(())
}
