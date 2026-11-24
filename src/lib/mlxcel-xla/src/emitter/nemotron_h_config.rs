// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
// SPDX-License-Identifier: Apache-2.0

//! Nemotron-H's independent attention, Mamba2, and feed-forward blocks.

use super::config::{
    Config, DenseMlpActivation, MixerKind, NormStyle, QuantConfig, RopeScaling,
    SSM_DEFAULT_CHUNK_SIZE, SsmConfig, WeightScheme, check_layer_count,
};
use super::recurrent_config::json_f32;
use serde_json::Value;

pub(super) fn from_json(v: &Value, quantization: Option<QuantConfig>) -> Result<Config, String> {
    let get = |key: &str| v.get(key).filter(|value| !value.is_null());
    let uint = |key: &str, default: Option<usize>| -> Result<usize, String> {
        match get(key) {
            Some(x) => x
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| format!("nemotron_h `{key}` must be a non-negative integer")),
            None => default.ok_or_else(|| format!("nemotron_h missing `{key}`")),
        }
    };
    let boolean = |key: &str, default: bool| -> Result<bool, String> {
        get(key).map_or(Ok(default), |x| {
            x.as_bool()
                .ok_or_else(|| format!("nemotron_h `{key}` must be a boolean"))
        })
    };
    // A non-null override wins over both block-name aliases, as on the MLX path.
    // Bound the schedule before allocating its normalized copy.
    for key in [
        "hybrid_override_pattern",
        "layers_block_type",
        "layer_types",
    ] {
        if let Some(value) = get(key) {
            if let Some(text) = value.as_str() {
                check_layer_count(text.len())?;
            }
            if let Some(entries) = value.as_array() {
                check_layer_count(entries.len())?;
            }
        }
    }
    let codes = if let Some(pattern) = get("hybrid_override_pattern") {
        match pattern {
            Value::String(text) => text.chars().map(|c| c.to_string()).collect::<Vec<_>>(),
            Value::Array(entries) => entries
                .iter()
                .map(|x| {
                    x.as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| "nemotron_h pattern entries must be strings".to_string())
                })
                .collect::<Result<Vec<_>, _>>()?,
            _ => return Err("nemotron_h hybrid_override_pattern must be a string or array".into()),
        }
    } else {
        get("layers_block_type")
            .or_else(|| get("layer_types"))
            .and_then(Value::as_array)
            .ok_or("nemotron_h requires hybrid_override_pattern, layers_block_type or layer_types")?
            .iter()
            .map(|x| match x.as_str() {
                Some("attention" | "full_attention") => Ok("*".to_string()),
                Some("conv" | "linear_attention" | "mamba") => Ok("M".to_string()),
                Some("mlp") => Ok("-".to_string()),
                Some("moe") => Ok("E".to_string()),
                _ => Err(format!("nemotron_h unknown block name {x}")),
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    let n_layers = check_layer_count(codes.len())?;
    if n_layers == 0 {
        return Err("nemotron_h requires at least one block".into());
    }
    // Saved transformers configs may omit num_hidden_layers. The schedule is
    // authoritative, matching NemotronHConfig::post_init on the MLX path.
    let layer_mixers = codes
        .iter()
        .map(|code| match code.as_str() {
            "M" => Ok(MixerKind::Mamba2),
            "*" => Ok(MixerKind::Attention),
            "-" => Ok(MixerKind::None),
            "E" => Err("nemotron_h MoE (E) blocks require the sigmoid router in #3345".to_string()),
            _ => Err(format!("nemotron_h unknown block code {code:?}")),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let hidden = uint("hidden_size", None)?;
    let inter = uint("intermediate_size", None)?;
    let vocab = uint("vocab_size", None)?;
    let n_q = uint("num_attention_heads", None)?;
    let n_kv = uint("num_key_value_heads", Some(n_q))?;
    for (name, x) in [
        ("hidden_size", hidden),
        ("intermediate_size", inter),
        ("vocab_size", vocab),
        ("num_attention_heads", n_q),
        ("num_key_value_heads", n_kv),
    ] {
        if x == 0 {
            return Err(format!("nemotron_h `{name}` must be positive"));
        }
    }
    let head_dim = uint(
        "attention_head_dim",
        Some(uint("head_dim", Some(hidden / n_q))?),
    )?;
    if head_dim == 0
        || !n_q.is_multiple_of(n_kv)
        || n_q.checked_mul(head_dim).is_none()
        || n_kv.checked_mul(head_dim).is_none()
    {
        return Err("nemotron_h invalid attention dimensions".into());
    }
    let n_heads = uint("mamba_num_heads", None)?;
    let ssm_head_dim = uint("mamba_head_dim", None)?;
    let n_groups = uint("n_groups", Some(1))?;
    let state_size = uint("ssm_state_size", None)?;
    let d_inner = n_heads
        .checked_mul(ssm_head_dim)
        .ok_or("nemotron_h d_inner overflow")?;
    let conv_dim = n_groups
        .checked_mul(state_size)
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| n.checked_add(d_inner))
        .ok_or("nemotron_h conv_dim overflow")?;
    if n_groups == 0 {
        return Err("nemotron_h n_groups must be positive".into());
    }
    let norm_eps =
        get("layer_norm_epsilon").map_or(Ok(1e-5), |x| json_f32(x, "layer_norm_epsilon"))?;
    let time_step_limit = match get("time_step_limit") {
        None => (0.0, f32::INFINITY),
        Some(Value::Array(pair)) if pair.len() == 2 => (
            json_f32(&pair[0], "time_step_limit[0]")?,
            json_f32(&pair[1], "time_step_limit[1]")?,
        ),
        _ => return Err("nemotron_h time_step_limit must be a pair".into()),
    };
    for (key, expected) in [("mamba_hidden_act", "silu"), ("mlp_hidden_act", "relu2")] {
        if let Some(act) = get(key)
            && act.as_str() != Some(expected)
        {
            return Err(format!("nemotron_h `{key}` must be {expected}"));
        }
    }
    let ssm = SsmConfig {
        n_heads,
        head_dim: ssm_head_dim,
        state_size,
        n_groups,
        conv_kernel: uint("conv_kernel", Some(4))?,
        conv_dim,
        use_conv_bias: boolean("use_conv_bias", true)?,
        use_bias: boolean("use_bias", false)?,
        time_step_limit,
        norm_eps,
        norm_group_size: d_inner / n_groups,
        chunk_size: SSM_DEFAULT_CHUNK_SIZE,
    };
    ssm.validate()?;
    Ok(Config {
        hidden,
        inter,
        vocab,
        n_layers,
        n_q,
        n_kv,
        head_dim,
        eps: norm_eps,
        rope_theta: 10000.0,
        rope: RopeScaling::Plain,
        use_rope_layers: Some(vec![false; n_layers]),
        tie_word_embeddings: boolean("tie_word_embeddings", false)?,
        qkv_bias: boolean("attention_bias", false)?,
        attn_o_bias: boolean("attention_bias", false)?,
        mlp_bias: boolean("mlp_bias", false)?,
        dense_mlp: true,
        dense_mlp_activation: DenseMlpActivation::Relu2,
        norm_style: NormStyle::Plain,
        weight_scheme: WeightScheme::NemotronH,
        ssm: Some(ssm),
        layer_mixers,
        quantization,
        ..Config::llama_3_2_1b()
    })
}
