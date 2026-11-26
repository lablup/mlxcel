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

//! Jamba `config.json` parsing (issue #3349).
//!
//! Jamba interleaves Mamba1 and attention blocks. Block `i` is attention when
//! `i % attn_layer_period == attn_layer_offset` and Mamba1 otherwise; every
//! block then runs `pre_ff_layernorm` and a feed-forward that is a softmax-routed
//! MoE when `num_experts > 1` and `i % expert_layer_period == expert_layer_offset`,
//! and a dense SwiGLU MLP otherwise. Attention applies no positional encoding.
//! The Mamba1 mixer normalizes the projected dt, B, and C with learned RMSNorm
//! weights ([`BcdtNorm::Weighted`]).
//!
//! The router follows Hugging Face `JambaSparseMoeBlock`: softmax over all
//! experts, then top-k, with no renormalization of the selected weights.

use serde_json::Value;

use super::config::{
    BcdtNorm, Config, ExpertAct, MoeConfig, MoeScoring, NormStyle, QuantConfig, RopeScaling,
    SSM_DEFAULT_CHUNK_SIZE, SsmConfig, SsmKind, WeightScheme, check_layer_count,
};
use super::recurrent_config::{MixerKind, json_f32};

/// Build a Jamba config from the Hugging Face `JambaConfig` field names.
pub(super) fn from_json(v: &Value, quantization: Option<QuantConfig>) -> Result<Config, String> {
    let get = |key: &str| v.get(key).filter(|value| !value.is_null());
    let uint = |key: &str, default: Option<usize>| -> Result<usize, String> {
        match get(key) {
            Some(x) => x
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| format!("jamba `{key}` must be a non-negative integer")),
            None => default.ok_or_else(|| format!("jamba config.json missing `{key}`")),
        }
    };
    let boolean = |key: &str, default: bool| -> Result<bool, String> {
        get(key).map_or(Ok(default), |x| {
            x.as_bool()
                .ok_or_else(|| format!("jamba `{key}` must be a boolean"))
        })
    };

    if let Some(act) = get("hidden_act")
        && act.as_str() != Some("silu")
    {
        return Err(format!(
            "jamba hidden_act {act} is not supported (the MLP and mixer are silu-gated)"
        ));
    }
    if get("sliding_window").is_some() {
        return Err("jamba sliding_window attention is not supported".into());
    }
    let hidden = uint("hidden_size", None)?;
    let inter = uint("intermediate_size", None)?;
    let n_layers = check_layer_count(uint("num_hidden_layers", None)?)?;
    let vocab = uint("vocab_size", None)?;
    let n_q = uint("num_attention_heads", None)?;
    let n_kv = uint("num_key_value_heads", Some(n_q))?;
    for (name, value) in [
        ("hidden_size", hidden),
        ("intermediate_size", inter),
        ("num_hidden_layers", n_layers),
        ("vocab_size", vocab),
        ("num_attention_heads", n_q),
        ("num_key_value_heads", n_kv),
    ] {
        if value == 0 {
            return Err(format!("jamba `{name}` must be positive"));
        }
    }
    let head_dim = uint("head_dim", Some(hidden / n_q))?;
    if head_dim == 0
        || !n_q.is_multiple_of(n_kv)
        || n_q.checked_mul(head_dim).is_none()
        || n_kv.checked_mul(head_dim).is_none()
    {
        return Err("jamba invalid attention dimensions".into());
    }
    let eps = get("rms_norm_eps").map_or(Ok(1e-6), |x| json_f32(x, "rms_norm_eps"))?;
    if !eps.is_finite() || eps <= 0.0 {
        return Err(format!(
            "jamba rms_norm_eps {eps} must be finite and positive"
        ));
    }

    // The block schedule, as `JambaConfig.layers_block_type` and
    // `layers_num_experts` derive it.
    let schedule = |period_key: &str, offset_key: &str, period: usize, offset: usize| {
        let period = uint(period_key, Some(period))?;
        let offset = uint(offset_key, Some(offset))?;
        if period == 0 || offset >= period {
            return Err(format!(
                "jamba `{offset_key}`={offset} must be less than a positive `{period_key}`={period}"
            ));
        }
        Ok::<_, String>(
            (0..n_layers)
                .map(|i| i % period == offset)
                .collect::<Vec<_>>(),
        )
    };
    let attention = schedule("attn_layer_period", "attn_layer_offset", 8, 4)?;
    let expert = schedule("expert_layer_period", "expert_layer_offset", 2, 1)?;
    let layer_mixers: Vec<MixerKind> = attention
        .iter()
        .map(|&is_attention| {
            if is_attention {
                MixerKind::Attention
            } else {
                MixerKind::Mamba1
            }
        })
        .collect();

    let n_experts = uint("num_experts", Some(16))?;
    let moe = if n_experts > 1 && expert.iter().any(|&is_moe| is_moe) {
        let top_k = uint("num_experts_per_tok", Some(2))?;
        if top_k == 0 || top_k > n_experts {
            return Err(format!(
                "jamba num_experts_per_tok={top_k} must be between 1 and num_experts={n_experts}"
            ));
        }
        Some(MoeConfig {
            n_experts,
            top_k,
            intermediate: inter,
            // HF selects the top-k of the full softmax and does not renormalize.
            norm_topk_prob: false,
            routed_scaling_factor: 1.0,
            scoring: MoeScoring::Softmax,
            expert_act: ExpertAct::SwiGlu,
            shared: None,
            first_k_dense: 0,
            layer_mask: Some(expert),
            weight_prefix: "feed_forward",
        })
    } else {
        None
    };

    let state_size = uint("mamba_d_state", Some(16))?;
    let conv_kernel = uint("mamba_d_conv", Some(4))?;
    let expand = uint("mamba_expand", Some(2))?;
    let d_inner = hidden
        .checked_mul(expand)
        .ok_or("jamba mamba_expand * hidden_size overflows usize")?;
    let dt_rank = match get("mamba_dt_rank") {
        None => hidden.div_ceil(16),
        Some(Value::String(text)) if text == "auto" => hidden.div_ceil(16),
        Some(Value::String(text)) => {
            return Err(format!("jamba mamba_dt_rank {text:?} is not supported"));
        }
        Some(_) => uint("mamba_dt_rank", None)?,
    };
    if d_inner == 0 || state_size == 0 || dt_rank == 0 {
        return Err("jamba Mamba dimensions must be positive".into());
    }
    let ssm = SsmConfig {
        kind: SsmKind::Mamba1 {
            dt_rank,
            bcdt_norm: BcdtNorm::Weighted,
        },
        n_heads: 1,
        head_dim: d_inner,
        state_size,
        n_groups: 1,
        conv_kernel,
        conv_dim: d_inner,
        use_conv_bias: boolean("mamba_conv_bias", true)?,
        use_bias: boolean("mamba_proj_bias", false)?,
        time_step_limit: (0.0, f32::INFINITY),
        // `dt_layernorm` / `b_layernorm` / `c_layernorm` use the block eps.
        norm_eps: eps,
        norm_group_size: d_inner,
        chunk_size: SSM_DEFAULT_CHUNK_SIZE,
    };
    ssm.validate()?;

    Ok(Config {
        hidden,
        inter,
        n_layers,
        n_q,
        n_kv,
        head_dim,
        eps,
        rope_theta: 10000.0,
        vocab,
        rope: RopeScaling::Plain,
        // Jamba attention has no positional encoding; the Mamba1 blocks order
        // the sequence.
        use_rope_layers: Some(vec![false; n_layers]),
        tie_word_embeddings: boolean("tie_word_embeddings", false)?,
        quantization,
        norm_style: NormStyle::Plain,
        weight_scheme: WeightScheme::Jamba,
        moe,
        ssm: Some(ssm),
        layer_mixers,
        ..Config::llama_3_2_1b()
    })
}
