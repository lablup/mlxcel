// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
// SPDX-License-Identifier: Apache-2.0

//! Raw Hugging Face Falcon-H1 dimensions and muP semantics.
use super::config::{
    Config, MixerKind, QuantConfig, SsmConfig, SsmKind, WeightScheme, check_layer_count,
};
use super::recurrent_config::json_f32;
use serde_json::Value;

/// Runtime factors for original HF weights, which have not folded muP scaling.
#[derive(Clone, Debug)]
pub struct FalconH1Config {
    pub attention_in: f32,
    pub attention_out: f32,
    pub key: f32,
    pub ssm_in: f32,
    pub ssm_out: f32,
    pub ssm_channels: [f32; 5],
    pub mlp: [f32; 2],
    pub rms_norm: bool,
    pub norm_before_gate: bool,
    pub out_bias: bool,
}

pub(super) fn from_json(v: &Value, quantization: Option<QuantConfig>) -> Result<Config, String> {
    let get = |key: &str| v.get(key).filter(|x| !x.is_null());
    let uint = |key: &str, default: Option<usize>| -> Result<usize, String> {
        get(key).map_or_else(
            || default.ok_or_else(|| format!("falcon_h1 missing `{key}`")),
            |x| {
                x.as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .filter(|&n| n > 0)
                    .ok_or_else(|| format!("falcon_h1 `{key}` must be a positive integer"))
            },
        )
    };
    let boolean = |key: &str, default: bool| -> Result<bool, String> {
        get(key).map_or(Ok(default), |x| {
            x.as_bool()
                .ok_or_else(|| format!("falcon_h1 `{key}` must be boolean"))
        })
    };
    let float = |key: &str, default: f32| -> Result<f32, String> {
        let f = get(key).map_or(Ok(default), |x| json_f32(x, key))?;
        if !f.is_finite() {
            return Err(format!("falcon_h1 `{key}` must be finite"));
        }
        Ok(f)
    };
    let factors = |key: &str, count: usize| -> Result<Vec<f32>, String> {
        match get(key) {
            None => Ok(vec![1.0; count]),
            Some(Value::Array(xs)) if xs.len() == count => xs
                .iter()
                .map(|x| {
                    let f = json_f32(x, key)?;
                    if f.is_finite() {
                        Ok(f)
                    } else {
                        Err(format!("falcon_h1 `{key}` must be finite"))
                    }
                })
                .collect(),
            _ => Err(format!("falcon_h1 `{key}` must have {count} multipliers")),
        }
    };
    let hidden = uint("hidden_size", None)?;
    let n_layers = check_layer_count(uint("num_hidden_layers", None)?)?;
    let n_q = uint("num_attention_heads", None)?;
    let n_kv = uint("num_key_value_heads", Some(n_q))?;
    let head_dim = uint("head_dim", Some(hidden / n_q))?;
    if head_dim == 0
        || !head_dim.is_multiple_of(2)
        || !n_q.is_multiple_of(n_kv)
        || n_q.checked_mul(head_dim).is_none()
        || n_kv.checked_mul(head_dim).is_none()
    {
        return Err("falcon_h1 invalid attention dimensions".into());
    }
    let d_inner = match get("mamba_d_ssm") {
        Some(_) => uint("mamba_d_ssm", None)?,
        None => hidden
            .checked_mul(uint("mamba_expand", Some(2))?)
            .ok_or("falcon_h1 SSM width overflow")?,
    };
    let n_heads = uint("mamba_n_heads", Some(128))?;
    let ssm_head_dim = match get("mamba_d_head") {
        Some(Value::String(text)) if text == "auto" => d_inner / n_heads,
        Some(_) => uint("mamba_d_head", None)?,
        None => uint("mamba_head_dim", Some(d_inner / n_heads))?,
    };
    if n_heads.checked_mul(ssm_head_dim) != Some(d_inner) || ssm_head_dim == 0 {
        return Err("falcon_h1 mamba_n_heads * mamba_d_head must equal mamba_d_ssm".into());
    }
    let n_groups = uint("mamba_n_groups", Some(1))?;
    let state_size = uint("mamba_d_state", Some(256))?;
    let conv_dim = n_groups
        .checked_mul(state_size)
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| n.checked_add(d_inner))
        .ok_or("falcon_h1 conv_dim overflow")?;
    let eps = float("rms_norm_eps", 1e-5)?;
    let channels: [f32; 5] = factors("ssm_multipliers", 5)?
        .try_into()
        .map_err(|_| "ssm_multipliers length")?;
    let mlp: [f32; 2] = factors("mlp_multipliers", 2)?
        .try_into()
        .map_err(|_| "mlp_multipliers length")?;
    let time_step_limit = match get("time_step_limit") {
        None => (0.0, f32::INFINITY),
        Some(Value::Array(pair)) if pair.len() == 2 => (
            json_f32(&pair[0], "time_step_limit[0]")?,
            json_f32(&pair[1], "time_step_limit[1]")?,
        ),
        _ => return Err("falcon_h1 time_step_limit must be a pair".into()),
    };
    let ssm = SsmConfig {
        kind: SsmKind::Mamba2,
        n_heads,
        head_dim: ssm_head_dim,
        state_size,
        n_groups,
        conv_kernel: uint("mamba_d_conv", Some(4))?,
        conv_dim,
        use_conv_bias: boolean("mamba_conv_bias", true)?,
        use_bias: boolean("mamba_proj_bias", false)?,
        time_step_limit,
        norm_eps: eps,
        norm_group_size: d_inner / n_groups,
        chunk_size: uint("mamba_chunk_size", Some(256))?,
    };
    ssm.validate()?;
    if get("hidden_act").is_some_and(|x| x.as_str() != Some("silu")) {
        return Err("falcon_h1 hidden_act must be silu".into());
    }
    // These optional modes change the reference topology and need their own contract.
    if get("attn_layer_indices").is_some()
        || get("sliding_window").is_some()
        || get("rope_scaling").is_some()
        || get("mup_vector").is_some()
    {
        return Err("falcon_h1 sparse attention, sliding_window, rope_scaling and explicit mup_vector are unsupported; use ssm_multipliers".into());
    }
    let inter = match get("intermediate_size") {
        Some(_) => uint("intermediate_size", None)?,
        None => hidden
            .checked_mul(uint("mlp_expansion_factor", Some(4))?)
            .ok_or("falcon_h1 MLP width overflow")?,
    };
    let rope_params = get("rope_parameters");
    if rope_params.is_some_and(|params| {
        params
            .get("rope_type")
            .and_then(Value::as_str)
            .unwrap_or("default")
            != "default"
    }) {
        return Err("falcon_h1 only default rope_parameters are supported".into());
    }
    let rope_theta = rope_params
        .and_then(|p| p.get("rope_theta"))
        .or_else(|| get("rope_theta"))
        .map_or(Ok(10000.0), |x| {
            x.as_f64().ok_or("falcon_h1 rope_theta must be numeric")
        })?;
    if !rope_theta.is_finite() || rope_theta <= 0.0 {
        return Err("falcon_h1 rope_theta must be positive".into());
    }
    Ok(Config {
        hidden,
        inter,
        n_layers,
        n_q,
        n_kv,
        head_dim,
        eps,
        rope_theta,
        vocab: uint("vocab_size", None)?,
        tie_word_embeddings: boolean("tie_word_embeddings", false)?,
        qkv_bias: boolean("attention_bias", false)?,
        attn_o_bias: boolean("attention_bias", false)?,
        mlp_bias: boolean("mlp_bias", false)?,
        embedding_multiplier: Some(float("embedding_multiplier", 1.0)?),
        logit_mul: Some(float("lm_head_multiplier", 1.0)?),
        weight_scheme: WeightScheme::FalconH1,
        layer_mixers: vec![MixerKind::ParallelMamba2Attention; n_layers],
        ssm: Some(ssm),
        quantization,
        falcon_h1: Some(FalconH1Config {
            attention_in: float("attention_in_multiplier", 1.0)?,
            attention_out: float("attention_out_multiplier", 1.0)?,
            key: float("key_multiplier", 1.0)?,
            ssm_in: float("ssm_in_multiplier", 1.0)?,
            ssm_out: float("ssm_out_multiplier", 1.0)?,
            ssm_channels: channels,
            mlp,
            rms_norm: boolean("mamba_rms_norm", false)?,
            norm_before_gate: boolean("mamba_norm_before_gate", true)?,
            out_bias: boolean("projectors_bias", false)?,
        }),
        ..Config::llama_3_2_1b()
    })
}
