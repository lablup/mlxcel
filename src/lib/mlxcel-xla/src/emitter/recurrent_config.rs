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

//! Per-layer mixer schedule and the threaded recurrent-state ABI (epic #564,
//! issue #3338).
//!
//! Every attention-only config carries `layer_mixers = [Attention; n_layers]`
//! and `ssm = None`, so it has no state tensors, a KV cache, and a FFN on every
//! layer: its graphs are unchanged. A Mamba2 config has no attention layer at
//! all, so it has no KV cache, and threads two f32 state tensors through its
//! graphs in the order [`Config::state_tensors`] lists them.

use super::config::{
    Config, NormStyle, QuantConfig, RopeScaling, SSM_DEFAULT_CHUNK_SIZE, SsmConfig, WeightScheme,
    check_layer_count,
};

/// The token mixer of one layer. Later units add more recurrent kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixerKind {
    /// Softmax attention with a KV cache (every pre-#564 family).
    Attention,
    /// Mamba2 selective state-space mixer ([`SsmConfig`]).
    Mamba2,
}

impl MixerKind {
    /// Whether the mixer carries recurrent state instead of a KV cache.
    #[must_use]
    pub fn is_recurrent(self) -> bool {
        !matches!(self, MixerKind::Attention)
    }

    /// The HF `model_type` of the family this mixer comes from, for errors.
    #[must_use]
    pub fn family(self) -> &'static str {
        match self {
            MixerKind::Attention => "attention",
            MixerKind::Mamba2 => "mamba2",
        }
    }
}

/// Element type of one threaded state tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateElem {
    F32,
}

/// One threaded recurrent-state tensor: its graph and shim name, its
/// per-sequence shape (no batch axis), and its element type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateTensorSpec {
    pub name: &'static str,
    pub dims: Vec<usize>,
    pub elem: StateElem,
}

impl Config {
    /// The mixer of layer `li`. A config whose schedule is shorter than
    /// `n_layers` (a test literal built from another config) reads as attention.
    #[must_use]
    pub fn layer_mixer(&self, li: usize) -> MixerKind {
        self.layer_mixers
            .get(li)
            .copied()
            .unwrap_or(MixerKind::Attention)
    }

    /// Whether layer `li` has a feed-forward sublayer. A pure Mamba2 block is
    /// `norm -> mixer -> residual` with no FFN.
    #[must_use]
    pub fn layer_has_ffn(&self, li: usize) -> bool {
        self.layer_mixer(li) == MixerKind::Attention
    }

    /// Number of layers that carry recurrent state.
    #[must_use]
    pub fn recurrent_layer_count(&self) -> usize {
        (0..self.n_layers)
            .filter(|&li| self.layer_mixer(li).is_recurrent())
            .count()
    }

    /// The ordinal of `li` among the recurrent layers (its row in every state
    /// tensor), or `None` for an attention layer.
    #[must_use]
    pub fn state_layer_index(&self, li: usize) -> Option<usize> {
        if li >= self.n_layers || !self.layer_mixer(li).is_recurrent() {
            return None;
        }
        Some(
            (0..li)
                .filter(|&lj| self.layer_mixer(lj).is_recurrent())
                .count(),
        )
    }

    /// Whether any layer is attention, i.e. whether the graphs carry a KV cache.
    #[must_use]
    pub fn has_kv_cache(&self) -> bool {
        (0..self.n_layers).any(|li| self.layer_mixer(li) == MixerKind::Attention)
    }

    /// The threaded state tensors, in graph-argument order. Empty for every
    /// attention-only config. Mamba2: `ssm_h [L_ssm, H, P, N]` then
    /// `ssm_conv [L_ssm, K-1, conv_dim]`, both f32.
    #[must_use]
    pub fn state_tensors(&self) -> Vec<StateTensorSpec> {
        let layers = self.recurrent_layer_count();
        match &self.ssm {
            Some(ssm) if layers > 0 => vec![
                StateTensorSpec {
                    name: "ssm_h",
                    dims: vec![layers, ssm.n_heads, ssm.head_dim, ssm.state_size],
                    elem: StateElem::F32,
                },
                StateTensorSpec {
                    name: "ssm_conv",
                    dims: vec![layers, ssm.conv_kernel - 1, ssm.conv_dim],
                    elem: StateElem::F32,
                },
            ],
            _ => Vec::new(),
        }
    }

    /// `!self.state_tensors().is_empty()`.
    #[must_use]
    pub fn has_recurrent_state(&self) -> bool {
        !self.state_tensors().is_empty()
    }

    /// The family name of the first recurrent mixer, for error messages.
    pub(crate) fn recurrent_family(&self) -> Option<&'static str> {
        (0..self.n_layers)
            .map(|li| self.layer_mixer(li))
            .find(|mixer| mixer.is_recurrent())
            .map(MixerKind::family)
    }

    /// `Err` naming the model type when this config threads recurrent state
    /// and `graph` has no state contract yet.
    pub(crate) fn reject_recurrent(&self, graph: &str) -> Result<(), String> {
        match self.recurrent_family() {
            Some(family) if self.has_recurrent_state() => Err(format!(
                "the OpenXLA {graph} graph does not support recurrent-state models \
                 (model_type {family}); only the token prefill and the \
                 single-sequence decode thread SSM state"
            )),
            _ => Ok(()),
        }
    }
}

/// Replace the bare `Infinity`, `-Infinity` and `NaN` tokens Python's `json`
/// writes (HF Mamba2 configs carry `"time_step_limit": [0.0, Infinity]`) with
/// strings, so `serde_json` accepts the document. Text inside string literals
/// is left alone. Returns `None` when there is nothing to replace.
pub(super) fn quote_non_finite_json(text: &str) -> Option<String> {
    let mut out = String::with_capacity(text.len() + 16);
    let mut in_string = false;
    let mut escaped = false;
    let mut changed = false;
    let mut rest = text;
    while let Some(ch) = rest.chars().next() {
        if in_string {
            out.push(ch);
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            rest = &rest[ch.len_utf8()..];
            continue;
        }
        if ch == '"' {
            in_string = true;
            out.push(ch);
            rest = &rest[1..];
            continue;
        }
        let token = ["-Infinity", "Infinity", "NaN"]
            .into_iter()
            .find(|token| rest.starts_with(token));
        if let Some(token) = token {
            out.push('"');
            out.push_str(token);
            out.push('"');
            rest = &rest[token.len()..];
            changed = true;
            continue;
        }
        out.push(ch);
        rest = &rest[ch.len_utf8()..];
    }
    changed.then_some(out)
}

/// A JSON number, or one of the quoted non-finite tokens
/// [`quote_non_finite_json`] produces, or `null` for +inf.
/// A config float. Besides plain numbers this accepts the spellings HF uses
/// for an unbounded limit: `null`, the bare `Infinity` token (quoted by
/// [`quote_non_finite_json`] before parsing), and transformers 5's
/// `{"__float__": "Infinity"}` object.
fn json_f32(value: &serde_json::Value, what: &str) -> Result<f32, String> {
    if let Some(inner) = value.as_object().and_then(|object| object.get("__float__")) {
        return json_f32(inner, what);
    }
    match value {
        serde_json::Value::Null => Ok(f32::INFINITY),
        serde_json::Value::String(text) if text == "Infinity" => Ok(f32::INFINITY),
        serde_json::Value::String(text) if text == "-Infinity" => Ok(f32::NEG_INFINITY),
        other => other
            .as_f64()
            .map(|x| x as f32)
            .ok_or_else(|| format!("config.json `{what}` must be a number")),
    }
}

/// Build the [`Config`] of an HF `Mamba2ForCausalLM` (`model_type = "mamba2"`).
///
/// The attention fields are zero (there is no attention layer, so nothing
/// reads them); `layer_mixers` is `Mamba2` on every layer. `chunk_size` is the
/// value the #3337 lowering spike chose, not the checkpoint's training chunk.
/// The dims are cross-checked against the checkpoint tensors by
/// [`Config::validate_recurrent_checkpoint`] when the config is read from a
/// model directory.
pub(super) fn mamba2_from_json(
    v: &serde_json::Value,
    quantization: Option<QuantConfig>,
) -> Result<Config, String> {
    let get = |key: &str| v.get(key).filter(|value| !value.is_null());
    let usize_key = |keys: &[&str]| -> Result<usize, String> {
        keys.iter()
            .find_map(|key| get(key))
            .ok_or_else(|| format!("mamba2 config.json missing integer among {keys:?}"))?
            .as_u64()
            .and_then(|x| usize::try_from(x).ok())
            .ok_or_else(|| format!("mamba2 config.json {keys:?} must be a non-negative integer"))
    };
    let usize_or = |key: &str, default: usize| -> Result<usize, String> {
        match get(key) {
            None => Ok(default),
            Some(_) => usize_key(&[key]),
        }
    };
    let bool_or = |key: &str, default: bool| -> Result<bool, String> {
        match get(key) {
            None => Ok(default),
            Some(value) => value
                .as_bool()
                .ok_or_else(|| format!("mamba2 config.json `{key}` must be a boolean")),
        }
    };

    if !bool_or("rms_norm", true)? {
        return Err("mamba2 with rms_norm = false (LayerNorm blocks) is not supported".into());
    }
    if let Some(act) = get("hidden_act")
        && act.as_str() != Some("silu")
    {
        return Err(format!(
            "mamba2 hidden_act {act} is not supported (the mixer is silu-gated)"
        ));
    }
    let hidden = usize_key(&["hidden_size"])?;
    let n_layers = check_layer_count(usize_key(&["num_hidden_layers"])?)?;
    let vocab = usize_key(&["vocab_size"])?;
    let n_heads = usize_key(&["num_heads"])?;
    let head_dim = usize_key(&["head_dim"])?;
    let state_size = usize_key(&["state_size", "ssm_state_size"])?;
    let n_groups = usize_or("n_groups", 1)?;
    let conv_kernel = usize_or("conv_kernel", 4)?;
    let expand = usize_or("expand", 2)?;
    for (name, value) in [
        ("hidden_size", hidden),
        ("num_hidden_layers", n_layers),
        ("vocab_size", vocab),
        ("expand", expand),
    ] {
        if value == 0 {
            return Err(format!("mamba2 config.json `{name}` must be positive"));
        }
    }
    let d_inner = n_heads
        .checked_mul(head_dim)
        .ok_or("mamba2 num_heads * head_dim overflows usize")?;
    // HF sizes the mixer by `expand * hidden_size` and reshapes it into
    // `num_heads` heads of `head_dim`; the two must agree.
    if expand.checked_mul(hidden) != Some(d_inner) {
        return Err(format!(
            "mamba2 num_heads * head_dim = {d_inner} != expand * hidden_size = {expand} * {hidden}"
        ));
    }
    let conv_dim = n_groups
        .checked_mul(state_size)
        .and_then(|gn| gn.checked_mul(2))
        .and_then(|bc| bc.checked_add(d_inner))
        .ok_or("mamba2 conv_dim = d_inner + 2*G*N overflows usize")?;
    let time_step_limit = match get("time_step_limit") {
        None => (0.0, f32::INFINITY),
        Some(serde_json::Value::Array(pair)) if pair.len() == 2 => (
            json_f32(&pair[0], "time_step_limit[0]")?,
            json_f32(&pair[1], "time_step_limit[1]")?,
        ),
        Some(_) => return Err("mamba2 config.json `time_step_limit` must be a pair".into()),
    };
    let norm_eps = match get("layer_norm_epsilon") {
        None => 1e-5,
        Some(value) => json_f32(value, "layer_norm_epsilon")?,
    };
    let ssm = SsmConfig {
        n_heads,
        head_dim,
        state_size,
        n_groups,
        conv_kernel,
        conv_dim,
        use_conv_bias: bool_or("use_conv_bias", true)?,
        use_bias: bool_or("use_bias", false)?,
        time_step_limit,
        norm_eps,
        // HF Mamba2 normalizes the gated output over all of d_inner at once.
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
        // HF `Mamba2Config` defaults to an untied head, so a config that omits
        // the key reads `lm_head.weight` like HF does (the checkpoint cross-check
        // then requires it) instead of silently reusing the embeddings.
        tie_word_embeddings: bool_or("tie_word_embeddings", false)?,
        quantization,
        norm_style: NormStyle::Plain,
        weight_scheme: WeightScheme::Mamba2,
        ssm: Some(ssm),
        layer_mixers: vec![MixerKind::Mamba2; n_layers],
        ..Config::llama_3_2_1b()
    })
}
