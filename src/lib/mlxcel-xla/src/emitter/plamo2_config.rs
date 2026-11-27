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

//! PLaMo 2 (`model_type = "plamo2"`) config reader and checkpoint cross-check
//! (epic #564, issue #3347).
//!
//! A PLaMo 2 block is `x + post_mixer_norm(mixer(pre_mixer_norm(x)))` followed
//! by `x + post_mlp_norm(mlp(pre_mlp_norm(x)))`, where the mixer is attention or
//! the [`SsmKind::Plamo2`] Mamba variant by index (HF `is_mamba`). Every norm is
//! `(weight + offset) * rmsnorm(x)` with a per-site offset, so the emitter uses
//! the four-norm [`NormStyle::GemmaFf`] placement with raw weights and adds the
//! offset in the graph where each norm weight is taken. Attention layers are
//! local (sliding) unless listed in `full_attention_idx`; HF keeps keys whose
//! age is at most `attention_window_size`, one more than the emitter's
//! `age < sliding_window` band, so the window is stored as `W + 1`.

use std::collections::HashMap;

use super::config::{
    Config, MixerKind, NormStyle, QkNorm, QuantConfig, RopeScaling, SsmConfig, SsmKind,
    WeightScheme, check_layer_count,
};
use super::recurrent_config::json_f32;
use serde_json::Value;

/// `pre_mixer_norm` weight offset.
pub(crate) const PRE_MIXER_NORM_OFFSET: f32 = 1.0;
/// `pre_mlp_norm` weight offset.
pub(crate) const PRE_MLP_NORM_OFFSET: f32 = 1.0;
/// Final `model.norm` weight offset.
pub(crate) const FINAL_NORM_OFFSET: f32 = 1.0;
/// The Q/K RMSNorm eps; HF and MLX hardcode it instead of `rms_norm_eps`.
pub(crate) const QK_NORM_EPS: f32 = 1e-6;
/// MLX eager's `ssm_update` default limit, which its PLaMo 2 scan applies
/// (`src/models/plamo2.rs`); the HF CPU scan applies no clamp.
pub(crate) const TIME_STEP_LIMIT: (f32, f32) = (0.001, 100.0);

/// `post_mixer_norm` weight offset, `1 / 5`.
#[must_use]
pub(crate) fn post_mixer_norm_offset() -> f32 {
    1.0 / 5.0
}

/// `post_mlp_norm` weight offset, `1 / 5**1.5` (the same f32 value as HF's
/// double expression rounded once).
#[must_use]
pub(crate) fn post_mlp_norm_offset() -> f32 {
    1.0 / 5.0_f32.powf(1.5)
}

/// The PLaMo 2 parameters the shared [`Config`] fields do not express.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plamo2Config {
    /// Original layer indices that use full (unwindowed) attention and the
    /// global RoPE base; every other attention layer is local.
    pub full_attention: Vec<usize>,
    /// HF `attention_window_size`: a local layer keeps keys of age `<= W`.
    pub attention_window: usize,
}

/// HF `is_mamba(config, i)`.
fn is_mamba(enabled: bool, step: usize, n_layers: usize, i: usize) -> bool {
    if !enabled {
        return false;
    }
    if n_layers <= step / 2 {
        return i != n_layers - 1;
    }
    i % step != step / 2
}

pub(super) fn from_json(v: &Value, quantization: Option<QuantConfig>) -> Result<Config, String> {
    if quantization.is_some() {
        return Err(
            "plamo2 on the OpenXLA backend loads unquantized checkpoints only (its fused \
             qkv_proj / gate_up_proj are row-sliced at load)"
                .into(),
        );
    }
    let get = |key: &str| v.get(key).filter(|x| !x.is_null());
    let uint = |key: &str, default: Option<usize>| -> Result<usize, String> {
        get(key).map_or_else(
            || default.ok_or_else(|| format!("plamo2 config.json missing `{key}`")),
            |x| {
                x.as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .filter(|&n| n > 0)
                    .ok_or_else(|| format!("plamo2 `{key}` must be a positive integer"))
            },
        )
    };
    let boolean = |key: &str, default: bool| -> Result<bool, String> {
        get(key).map_or(Ok(default), |x| {
            x.as_bool()
                .ok_or_else(|| format!("plamo2 `{key}` must be a boolean"))
        })
    };
    let positive_float = |key: &str, default: f64| -> Result<f64, String> {
        let f = get(key).map_or(Ok(default), |x| {
            x.as_f64()
                .ok_or_else(|| format!("plamo2 `{key}` must be a number"))
        })?;
        if !f.is_finite() || f <= 0.0 {
            return Err(format!("plamo2 `{key}` must be finite and positive"));
        }
        Ok(f)
    };
    // MoE, FP8-evaluation and image variants change the block topology.
    for key in [
        "n_expert",
        "k_expert",
        "sparse_step",
        "eval_attention_n_bit",
        "eval_mlp_n_bit",
        "image_token_id",
        "rope_scaling",
    ] {
        if get(key).is_some() {
            return Err(format!(
                "plamo2 `{key}` is not supported on the OpenXLA backend"
            ));
        }
    }
    if boolean("use_predefined_initial_state", false)? {
        return Err("plamo2 use_predefined_initial_state = true is not supported".into());
    }

    let hidden = uint("hidden_size", None)?;
    let n_layers = check_layer_count(uint("num_hidden_layers", None)?)?;
    let vocab = uint("vocab_size", None)?;
    let inter = uint("intermediate_size", None)?;
    let n_q = uint("num_attention_heads", None)?;
    let n_kv = uint("num_key_value_heads", Some(4))?;
    let head_dim = uint("hidden_size_per_head", Some(128))?;
    if !head_dim.is_multiple_of(2)
        || !n_q.is_multiple_of(n_kv)
        || n_q.checked_mul(head_dim).is_none()
        || n_kv.checked_mul(head_dim).is_none()
        || inter.checked_mul(2).is_none()
    {
        return Err("plamo2 invalid attention or MLP dimensions".into());
    }
    let eps = get("rms_norm_eps").map_or(Ok(1e-6), |x| json_f32(x, "rms_norm_eps"))?;
    if !eps.is_finite() || eps <= 0.0 {
        return Err("plamo2 rms_norm_eps must be finite and positive".into());
    }
    let step = uint("mamba_step", Some(2))?;
    if step < 2 {
        return Err("plamo2 mamba_step must be greater than 1".into());
    }
    let enabled = boolean("mamba_enabled", true)?;
    let window = uint("attention_window_size", Some(2048))?;
    let band = window
        .checked_add(1)
        .filter(|&w| i32::try_from(w).is_ok())
        .ok_or("plamo2 attention_window_size + 1 must fit i32")?;
    let mut full_attention = Vec::new();
    match get("full_attention_idx") {
        None => {}
        Some(Value::Array(items)) => {
            for item in items {
                let li = item
                    .as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .filter(|&n| n < n_layers)
                    .ok_or("plamo2 full_attention_idx entries must be layer indices")?;
                if !full_attention.contains(&li) {
                    full_attention.push(li);
                }
            }
        }
        Some(_) => return Err("plamo2 full_attention_idx must be a list".into()),
    }
    let rope_theta = positive_float("rope_theta", 10000.0)?;
    let rope_local_theta = positive_float("rope_local_theta", 10000.0)?;

    let layer_mixers: Vec<MixerKind> = (0..n_layers)
        .map(|li| {
            if is_mamba(enabled, step, n_layers, li) {
                MixerKind::Plamo2Mamba
            } else {
                MixerKind::Attention
            }
        })
        .collect();
    let ssm = if layer_mixers.contains(&MixerKind::Plamo2Mamba) {
        let n_heads = uint("mamba_num_heads", Some(64))?;
        let d_inner = n_heads
            .checked_mul(head_dim)
            .ok_or("plamo2 mamba_num_heads * hidden_size_per_head overflows usize")?;
        let ssm = SsmConfig {
            kind: SsmKind::Plamo2 {
                // HF / MLX `dt_dim = max(64, hidden_size / 16)`.
                dt_rank: (hidden / 16).max(64),
            },
            n_heads,
            head_dim,
            state_size: uint("mamba_d_state", Some(64))?,
            n_groups: 1,
            conv_kernel: uint("mamba_d_conv", Some(4))?,
            conv_dim: d_inner,
            use_conv_bias: false,
            use_bias: false,
            time_step_limit: TIME_STEP_LIMIT,
            norm_eps: eps,
            norm_group_size: d_inner,
            chunk_size: uint("mamba_chunk_size", Some(256))?,
        };
        ssm.validate()?;
        Some(ssm)
    } else {
        None
    };
    // A distinct local base needs the local RoPE tables; equal bases share one.
    let uses_local = layer_mixers
        .iter()
        .enumerate()
        .any(|(li, m)| *m == MixerKind::Attention && !full_attention.contains(&li));
    let rope_local_base =
        (uses_local && rope_local_theta != rope_theta).then_some(rope_local_theta);
    Ok(Config {
        hidden,
        inter,
        n_layers,
        n_q,
        n_kv,
        head_dim,
        eps,
        rope_theta,
        // Plain RoPE: the Llama-3.2 base below carries Llama3 frequency scaling,
        // which only shows at long positions (the 2048-window probe caught it).
        rope: RopeScaling::Plain,
        vocab,
        tie_word_embeddings: boolean("tie_word_embeddings", true)?,
        quantization: None,
        norm_style: NormStyle::GemmaFf,
        qk_norm: Some(QkNorm {
            per_head: true,
            one_plus: false,
        }),
        rope_local_base,
        sliding_window: Some(band),
        fused_qkv: true,
        fused_gate_up: true,
        weight_scheme: WeightScheme::Plamo2,
        ssm,
        plamo2: Some(Plamo2Config {
            full_attention,
            attention_window: window,
        }),
        layer_mixers,
        ..Config::llama_3_2_1b()
    })
}

/// Cross-check every PLaMo 2 tensor shape against the checkpoint headers
/// (`shapes`, from [`Config::validate_recurrent_checkpoint`]).
pub(super) fn validate_checkpoint(
    c: &Config,
    shapes: &HashMap<String, Vec<usize>>,
) -> Result<(), String> {
    let ssm = c
        .ssm
        .as_ref()
        .ok_or("a recurrent plamo2 config has no SSM")?;
    let SsmKind::Plamo2 { dt_rank } = ssm.kind else {
        return Err("a plamo2 config must use the Plamo2 SSM kind".into());
    };
    let check = |name: &str, want: &[usize]| -> Result<(), String> {
        match shapes.get(name) {
            Some(got) if got.as_slice() == want => Ok(()),
            Some(got) => Err(format!(
                "checkpoint tensor {name} has shape {got:?}, config.json implies {want:?}"
            )),
            None => Err(format!("checkpoint has no tensor {name}")),
        }
    };
    let (h, di, n, hs) = (c.hidden, ssm.d_inner(), ssm.state_size, ssm.n_heads);
    // The state `H*P*N` is sized from the config alone; bound it by the
    // in_proj it is derived from, as the generic recurrent check does.
    let state = di.checked_mul(n);
    let in_proj = ssm.in_proj_dim().checked_mul(h);
    if !matches!((state, in_proj), (Some(s), Some(p)) if s <= p) {
        return Err(format!(
            "SSM state D*N = {di} * {n} exceeds the in_proj weight [{}, {h}]",
            ssm.in_proj_dim()
        ));
    }
    let names = crate::weight_names::scheme_names(WeightScheme::Plamo2);
    check(names.embed, &[c.vocab, h])?;
    check(names.final_norm, &[h])?;
    if !c.tie_word_embeddings {
        check(names.lm_head, &[c.vocab, h])?;
    }
    let (qd, kvd) = (c.n_q * c.head_dim, c.n_kv * c.head_dim);
    for li in 0..c.n_layers {
        let p = format!("{}{li}.", names.layer_stem);
        for norm in [
            names.input_layernorm,
            names.post_attention_layernorm,
            names.pre_ff_norm,
            names.post_ff_norm,
        ] {
            check(&format!("{p}{norm}"), &[h])?;
        }
        check(&format!("{p}mlp.gate_up_proj.weight"), &[2 * c.inter, h])?;
        check(&format!("{p}mlp.down_proj.weight"), &[h, c.inter])?;
        let m = format!("{p}mixer.");
        if c.layer_mixer(li) == MixerKind::Attention {
            check(&format!("{m}qkv_proj.weight"), &[qd + 2 * kvd, h])?;
            check(&format!("{m}o_proj.weight"), &[h, qd])?;
            check(&format!("{m}q_weight"), &[c.n_q, c.head_dim])?;
            check(&format!("{m}k_weight"), &[c.n_kv, c.head_dim])?;
            continue;
        }
        check(&format!("{m}in_proj.weight"), &[2 * di, h])?;
        check(&format!("{m}conv1d.weight"), &[di, 1, ssm.conv_kernel])?;
        check(&format!("{m}bcdt_proj.weight"), &[2 * n + dt_rank, di])?;
        check(&format!("{m}dt_proj.weight"), &[hs, dt_rank])?;
        check(&format!("{m}dt_norm_weight"), &[dt_rank])?;
        check(&format!("{m}B_norm_weight"), &[n])?;
        check(&format!("{m}C_norm_weight"), &[n])?;
        for name in ["A_log", "D", "dt_bias"] {
            check(&format!("{m}{name}"), &[hs])?;
        }
        check(&format!("{m}out_proj.weight"), &[h, di])?;
    }
    Ok(())
}
