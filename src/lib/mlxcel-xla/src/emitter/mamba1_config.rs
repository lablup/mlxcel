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

//! Mamba1 and Falcon-Mamba `config.json` parsing (issue #3348).

use super::config::{
    Config, NormStyle, QuantConfig, RopeScaling, SSM_DEFAULT_CHUNK_SIZE, SsmConfig, SsmKind,
    WeightScheme, check_layer_count,
};
use super::recurrent_config::MixerKind;

/// Build an attention-free Mamba1 config from either HF's field names or the
/// original state-spaces aliases retained by `*-hf` checkpoints.
pub(super) fn mamba1_from_json(
    v: &serde_json::Value,
    quantization: Option<QuantConfig>,
) -> Result<Config, String> {
    let model_type = v
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("mamba");
    let family = if model_type == "falcon_mamba" {
        "falcon_mamba"
    } else {
        "mamba"
    };
    let value = |keys: &[&str]| {
        keys.iter()
            .find_map(|key| v.get(*key).filter(|value| !value.is_null()))
    };
    let usize_key = |keys: &[&str]| -> Result<usize, String> {
        value(keys)
            .ok_or_else(|| format!("{family} config.json missing integer among {keys:?}"))?
            .as_u64()
            .and_then(|x| usize::try_from(x).ok())
            .ok_or_else(|| format!("{family} config.json {keys:?} must be a non-negative integer"))
    };
    let usize_or = |keys: &[&str], default: usize| -> Result<usize, String> {
        match value(keys) {
            None => Ok(default),
            Some(_) => usize_key(keys),
        }
    };
    let bool_keys_or = |keys: &[&str], default: bool| -> Result<bool, String> {
        match value(keys) {
            None => Ok(default),
            Some(value) => value
                .as_bool()
                .ok_or_else(|| format!("{family} config.json {keys:?} must be a boolean")),
        }
    };
    let bool_or = |key: &str, default: bool| bool_keys_or(&[key], default);
    let f32_or = |keys: &[&str], default: f32| -> Result<f32, String> {
        match value(keys) {
            None => Ok(default),
            Some(value) => value
                .as_f64()
                .map(|x| x as f32)
                .ok_or_else(|| format!("{family} config.json {keys:?} must be a number")),
        }
    };

    if !bool_or("rms_norm", true)? {
        return Err(format!(
            "{family} with rms_norm = false (LayerNorm blocks) is not supported"
        ));
    }
    if let Some(act) = value(&["hidden_act"])
        && act.as_str() != Some("silu")
    {
        return Err(format!(
            "{family} hidden_act {act} is not supported (the mixer is silu-gated)"
        ));
    }
    let hidden = usize_key(&["hidden_size", "d_model"])?;
    let n_layers = check_layer_count(usize_key(&["num_hidden_layers", "n_layer", "n_layers"])?)?;
    let vocab = usize_key(&["vocab_size"])?;
    let state_size = usize_key(&["state_size", "d_state"])?;
    let conv_kernel = usize_or(&["conv_kernel", "d_conv"], 4)?;
    let expand = usize_or(&["expand"], 2)?;
    let d_inner = match value(&["intermediate_size", "d_inner"]) {
        Some(_) => usize_key(&["intermediate_size", "d_inner"])?,
        None => hidden
            .checked_mul(expand)
            .ok_or_else(|| format!("{family} expand * hidden_size overflows usize"))?,
    };
    let dt_rank = match value(&["time_step_rank", "dt_rank"]) {
        None => hidden.div_ceil(16),
        Some(serde_json::Value::String(text)) if text == "auto" => hidden.div_ceil(16),
        Some(serde_json::Value::String(text)) => {
            return Err(format!("{family} time_step_rank {text:?} is not supported"));
        }
        Some(_) => usize_key(&["time_step_rank", "dt_rank"])?,
    };
    for (name, value) in [
        ("hidden_size", hidden),
        ("num_hidden_layers", n_layers),
        ("vocab_size", vocab),
        ("intermediate_size", d_inner),
        ("state_size", state_size),
        ("time_step_rank", dt_rank),
    ] {
        if value == 0 {
            return Err(format!("{family} config.json `{name}` must be positive"));
        }
    }
    let norm_eps = f32_or(&["layer_norm_epsilon", "rms_norm_eps"], 1e-5)?;
    let mixer_eps = f32_or(&["mixer_rms_eps"], 1e-6)?;
    let bcdt_norm = model_type == "falcon_mamba" || bool_or("use_bcdt_rms", false)?;
    if !norm_eps.is_finite() || norm_eps <= 0.0 {
        return Err(format!(
            "{family} block norm epsilon {norm_eps} must be finite and positive"
        ));
    }
    if bcdt_norm && (!mixer_eps.is_finite() || mixer_eps <= 0.0) {
        return Err(format!(
            "{family} mixer_rms_eps {mixer_eps} must be finite and positive"
        ));
    }
    let ssm = SsmConfig {
        kind: SsmKind::Mamba1 { dt_rank, bcdt_norm },
        // Reuse `n_heads * head_dim` as the common d_inner representation.
        n_heads: 1,
        head_dim: d_inner,
        state_size,
        n_groups: 1,
        conv_kernel,
        conv_dim: d_inner,
        // Transformers uses the `use_*` names while the original state-spaces
        // configs use `conv_bias` / `bias`. Prefer the canonical HF spellings
        // when a converted checkpoint happens to contain both.
        use_conv_bias: bool_keys_or(&["use_conv_bias", "conv_bias"], true)?,
        use_bias: bool_keys_or(&["use_bias", "bias"], false)?,
        time_step_limit: (0.0, f32::INFINITY),
        norm_eps: if bcdt_norm { mixer_eps } else { norm_eps },
        norm_group_size: d_inner,
        chunk_size: SSM_DEFAULT_CHUNK_SIZE,
    };
    ssm.validate()?;

    Ok(Config {
        hidden,
        inter: 0,
        n_layers,
        n_q: 0,
        n_kv: 0,
        head_dim: 0,
        eps: norm_eps,
        rope_theta: 0.0,
        vocab,
        rope: RopeScaling::Plain,
        tie_word_embeddings: bool_or("tie_word_embeddings", true)?,
        quantization,
        norm_style: NormStyle::Plain,
        weight_scheme: WeightScheme::Mamba1,
        ssm: Some(ssm),
        layer_mixers: vec![MixerKind::Mamba1; n_layers],
        ..Config::llama_3_2_1b()
    })
}
