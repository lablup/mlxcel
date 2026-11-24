// Copyright 2025-2026 Lablup Inc.
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

//! `mtp_config` of an Inkling 2 checkpoint, validated as upstream
//! `mlx_vlm/speculative/drafters/inkling2_dspark/config.py` does.

use std::path::Path;

use serde_json::Value;

/// Top-level `model_type` values of the two Inkling 2 releases.
pub const INKLING2_MODEL_TYPES: [&str; 2] = ["inkling2", "inkling2_loop"];

/// The bundled drafter's weight file, next to the target shards.
pub const MTP_WEIGHTS_FILE: &str = "mtp.safetensors";

/// Shape of an Inkling 2 target that a bundled drafter must match.
///
/// The drafter reads it from its own `config.json`; the target reports its
/// own through [`super::Inkling2DsparkBinding`]. Bind refuses a mismatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inkling2TargetShape {
    /// Top-level `model_type` (`inkling2` or `inkling2_loop`).
    pub model_type: String,
    pub hidden_size: usize,
    /// Padded vocabulary of the LM head and embedding table.
    pub vocab_size: usize,
    /// Physical decoder layers.
    pub num_hidden_layers: usize,
    /// Layer visits (`len(layer_indices)`): one per physical layer, plus the
    /// repeated core passes of a looped checkpoint.
    pub visit_count: usize,
    /// `(core_start, core_end, num_passes)` of a looped checkpoint.
    pub loop_config: Option<(usize, usize, usize)>,
}

impl Inkling2TargetShape {
    /// The first field that differs from `other`, as `(name, self, other)`.
    pub fn first_mismatch(&self, other: &Self) -> Option<(&'static str, String, String)> {
        let fields: [(&'static str, String, String); 6] = [
            (
                "model_type",
                self.model_type.clone(),
                other.model_type.clone(),
            ),
            (
                "hidden_size",
                self.hidden_size.to_string(),
                other.hidden_size.to_string(),
            ),
            (
                "vocab_size",
                self.vocab_size.to_string(),
                other.vocab_size.to_string(),
            ),
            (
                "num_hidden_layers",
                self.num_hidden_layers.to_string(),
                other.num_hidden_layers.to_string(),
            ),
            (
                "loop_config (core_start, core_end, num_passes)",
                format!("{:?}", self.loop_config),
                format!("{:?}", other.loop_config),
            ),
            (
                "layer visit count",
                self.visit_count.to_string(),
                other.visit_count.to_string(),
            ),
        ];
        fields.into_iter().find(|(_, a, b)| a != b)
    }
}

/// Validated drafter configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct Inkling2DsparkConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub markov_rank: usize,
    pub sliding_window_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    /// Visit ids the target captures, in fusion order; the last one is the
    /// final normalized hidden (id `visit_count`).
    pub target_layer_ids: Vec<usize>,
    /// Drafter layers that attend over a sliding context window.
    pub local_layer_ids: Vec<usize>,
    /// Trained block: one anchor plus `block_size - 1` proposals. Also the
    /// widest verify block the drafter can serve.
    pub block_size: usize,
    pub vocab_size: usize,
    pub unpadded_vocab_size: usize,
    pub final_logit_softcapping: f32,
    /// What the drafter was trained against.
    pub target: Inkling2TargetShape,
}

fn field<'a>(object: &'a Value, name: &str) -> Result<&'a Value, String> {
    object
        .get(name)
        .ok_or_else(|| format!("Inkling 2 mtp_config is missing {name}"))
}

fn usize_field(object: &Value, name: &str) -> Result<usize, String> {
    let value = field(object, name)?;
    value
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| format!("Inkling 2 {name} must be a non-negative integer, got {value}"))
}

fn f32_field(object: &Value, name: &str) -> Result<f32, String> {
    let value = field(object, name)?;
    value
        .as_f64()
        .map(|v| v as f32)
        .ok_or_else(|| format!("Inkling 2 {name} must be a number, got {value}"))
}

/// A list of ids; negative or non-integer entries are an error naming them.
fn id_list(object: &Value, name: &str) -> Result<Vec<usize>, String> {
    let value = field(object, name)?;
    let items = value
        .as_array()
        .ok_or_else(|| format!("Inkling 2 mtp_config {name} must be a list, got {value}"))?;
    items
        .iter()
        .map(|item| {
            item.as_u64()
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| {
                    format!("Inkling 2 mtp_config {name} entries must be >= 0, got {value}")
                })
        })
        .collect()
}

fn flag(object: &Value, name: &str) -> bool {
    object.get(name).is_some_and(|v| v.as_bool() == Some(true))
}

/// The target's visit list length and loop triple, from its text config.
fn target_shape(top: &Value, text: &Value) -> Result<Inkling2TargetShape, String> {
    let model_type = top
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let num_hidden_layers = usize_field(text, "num_hidden_layers")?;
    let loop_config = match text.get("loop_config") {
        None | Some(Value::Null) => None,
        Some(lc) => Some((
            usize_field(lc, "core_start")?,
            usize_field(lc, "core_end")?,
            usize_field(lc, "num_passes")?,
        )),
    };
    let visit_count = match loop_config {
        None => num_hidden_layers,
        Some((start, end, passes)) => {
            if !(start < end && end <= num_hidden_layers && passes >= 1) {
                return Err(format!(
                    "Inkling 2 loop_config {loop_config:?} does not fit {num_hidden_layers} layers"
                ));
            }
            num_hidden_layers + (end - start) * (passes - 1)
        }
    };
    Ok(Inkling2TargetShape {
        model_type,
        hidden_size: usize_field(text, "hidden_size")?,
        vocab_size: usize_field(text, "vocab_size")?,
        num_hidden_layers,
        visit_count,
        loop_config,
    })
}

impl Inkling2DsparkConfig {
    /// Parse a whole Inkling 2 `config.json`. `mtp_config` sits at the top
    /// level of the native release and inside `text_config` of an mlx-vlm
    /// split; both are read.
    pub fn from_json(top: &Value) -> Result<Self, String> {
        let model_type = top.get("model_type").and_then(Value::as_str);
        if !model_type.is_some_and(|mt| INKLING2_MODEL_TYPES.contains(&mt)) {
            return Err(format!(
                "Inkling 2 DSpark needs model_type inkling2 or inkling2_loop, got {model_type:?}"
            ));
        }
        let text = top.get("text_config").unwrap_or(top);
        let draft = top
            .get("mtp_config")
            .or_else(|| text.get("mtp_config"))
            .filter(|v| v.as_object().is_some_and(|o| !o.is_empty()))
            .ok_or_else(|| "Inkling 2 DSpark requires mtp_config".to_string())?;
        let target = target_shape(top, text)?;
        let vocab_size = target.vocab_size;
        let unpadded_vocab_size = match text.get("unpadded_vocab_size") {
            None | Some(Value::Null) => vocab_size,
            Some(_) => usize_field(text, "unpadded_vocab_size")?.min(vocab_size),
        };
        let final_logit_softcapping = match text.get("final_logit_softcapping") {
            None | Some(Value::Null) => 30.0,
            Some(_) => f32_field(text, "final_logit_softcapping")?,
        };
        let config = Self {
            hidden_size: usize_field(draft, "hidden_size")?,
            intermediate_size: usize_field(draft, "intermediate_size")?,
            num_hidden_layers: usize_field(draft, "num_hidden_layers")?,
            num_attention_heads: usize_field(draft, "num_attention_heads")?,
            num_key_value_heads: usize_field(draft, "num_key_value_heads")?,
            head_dim: usize_field(draft, "head_dim")?,
            markov_rank: usize_field(draft, "markov_rank")?,
            sliding_window_size: usize_field(draft, "sliding_window_size")?,
            rope_theta: f32_field(draft, "rope_theta")?,
            rms_norm_eps: f32_field(draft, "rms_norm_eps")?,
            target_layer_ids: id_list(draft, "target_layer_ids")?,
            local_layer_ids: id_list(draft, "local_layer_ids")?,
            block_size: usize_field(draft, "block_size")?,
            vocab_size,
            unpadded_vocab_size,
            final_logit_softcapping,
            target,
        };
        let head_type = draft.get("markov_head_type").and_then(Value::as_str);
        config.validate(head_type, flag(draft, "q_bias"), flag(draft, "o_bias"))?;
        Ok(config)
    }

    /// Read `dir/config.json`; also requires `dir/mtp.safetensors`.
    pub fn from_dir(dir: &Path) -> Result<Self, String> {
        let path = dir.join("config.json");
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let json: Value =
            serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        if !dir.join(MTP_WEIGHTS_FILE).is_file() {
            return Err(format!(
                "{} has no {MTP_WEIGHTS_FILE}; this Inkling 2 export ships no DSpark drafter",
                dir.display()
            ));
        }
        Self::from_json(&json)
    }

    /// Upstream `ModelConfig.__post_init__`, with the offending value named.
    fn validate(&self, head_type: Option<&str>, q_bias: bool, o_bias: bool) -> Result<(), String> {
        let unsupported = |what: String| Err(format!("Unsupported Inkling 2 mtp_config: {what}"));
        if head_type != Some("rnn_hidden") {
            return unsupported(format!(
                "markov_head_type {head_type:?} (only \"rnn_hidden\" is implemented)"
            ));
        }
        if q_bias || o_bias {
            return unsupported(format!("q_bias = {q_bias}, o_bias = {o_bias} (no bias)"));
        }
        if self.hidden_size != self.target.hidden_size {
            return unsupported(format!(
                "drafter hidden_size {} differs from the target's {}",
                self.hidden_size, self.target.hidden_size
            ));
        }
        if self.block_size < 2 {
            return unsupported(format!("block_size {} (need >= 2)", self.block_size));
        }
        if self.num_hidden_layers < 1 || self.markov_rank < 1 {
            return unsupported(format!(
                "num_hidden_layers {} / markov_rank {} (need >= 1)",
                self.num_hidden_layers, self.markov_rank
            ));
        }
        if self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            return unsupported(format!(
                "num_attention_heads {} is not a multiple of num_key_value_heads {}",
                self.num_attention_heads, self.num_key_value_heads
            ));
        }
        if self.head_dim == 0 || !self.head_dim.is_multiple_of(2) {
            return unsupported(format!("head_dim {} (need even)", self.head_dim));
        }
        if self.sliding_window_size < 1 {
            return unsupported("sliding_window_size 0".to_string());
        }
        if let Some(bad) = self
            .local_layer_ids
            .iter()
            .find(|&&i| i >= self.num_hidden_layers)
        {
            return unsupported(format!(
                "local_layer_ids entry {bad} past num_hidden_layers {}",
                self.num_hidden_layers
            ));
        }
        let ids = &self.target_layer_ids;
        let increasing = ids.windows(2).all(|w| w[0] < w[1]);
        if ids.len() < 2 || !increasing || ids.last() != Some(&self.target.visit_count) {
            return Err(format!(
                "Invalid Inkling 2 DSpark target_layer_ids {ids:?}: need at least 2 strictly \
                 increasing ids ending at the target's layer visit count {}",
                self.target.visit_count
            ));
        }
        Ok(())
    }

    /// Default verify width: the trained block (one anchor and
    /// `block_size - 1` proposals).
    pub fn default_verify_width(&self) -> usize {
        self.block_size
    }

    /// `None` when a `width`-row verify block is servable, otherwise the
    /// refusal naming the accepted range.
    pub fn verify_width_error(&self, width: usize) -> Option<String> {
        (!(2..=self.block_size).contains(&width)).then(|| {
            format!(
                "Inkling 2 DSpark drafter serves verify blocks of 2..={} rows (one anchor and up \
                 to {} proposals, its trained block); got --draft-block-size {width}",
                self.block_size,
                self.block_size - 1
            )
        })
    }
}

/// Whether `dir` is an Inkling 2 checkpoint that bundles a DSpark drafter:
/// an Inkling 2 `model_type`, a non-empty `mtp_config` and `mtp.safetensors`.
/// Reads only `config.json` and checks the file exists.
///
/// Used by: drafter routing (`resolve_drafter_kind`, `load_drafter`,
/// `dflash::is_dflash_drafter_dir`).
pub fn is_inkling2_dspark_dir(dir: &Path) -> bool {
    if !dir.join(MTP_WEIGHTS_FILE).is_file() {
        return false;
    }
    let Ok(bytes) = std::fs::read(dir.join("config.json")) else {
        return false;
    };
    let Ok(json) = serde_json::from_slice::<Value>(&bytes) else {
        return false;
    };
    let inkling2 = json
        .get("model_type")
        .and_then(Value::as_str)
        .is_some_and(|mt| INKLING2_MODEL_TYPES.contains(&mt));
    let text = json.get("text_config").unwrap_or(&json);
    let has_mtp = json
        .get("mtp_config")
        .or_else(|| text.get("mtp_config"))
        .is_some_and(|v| v.as_object().is_some_and(|o| !o.is_empty()));
    inkling2 && has_mtp
}
