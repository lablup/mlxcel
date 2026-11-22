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

//! Limite `config.json` contract. The checkpoint documents its numerical
//! conventions in string fields; each one is an assertion about what this
//! port implements, so a value other than the implemented one is refused at
//! load instead of running a silently different model.

use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};

#[derive(Debug, Clone, Deserialize)]
pub struct ModelArgs {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    pub vocab_size: usize,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_mlp_type")]
    pub mlp_type: String,

    #[serde(default = "default_softmax_scale")]
    pub attention_softmax_scale: f32,
    #[serde(default = "default_attn_gate_channels")]
    pub attn_gate_channels: usize,
    #[serde(default = "default_gate_scale")]
    pub attn_gate_scale: f32,

    #[serde(default = "default_head_dim")]
    pub ve_dim: usize,
    #[serde(default = "default_ve_gate_channels")]
    pub ve_gate_channels: usize,
    #[serde(default = "default_gate_scale")]
    pub ve_gate_scale: f32,
    #[serde(default = "default_ve_stored_heads")]
    pub ve_stored_heads: usize,
    #[serde(default)]
    pub ve_layers: Vec<usize>,

    #[serde(default = "default_true")]
    pub xsa: bool,
    #[serde(default)]
    pub xsa_layers: Vec<usize>,
    #[serde(default = "default_xsa_eps")]
    pub xsa_normalize_eps: f32,

    #[serde(default)]
    pub global_layers: Vec<usize>,
    #[serde(default = "default_true")]
    pub global_nope: bool,
    #[serde(default = "default_global_window")]
    pub global_window: i64,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,

    #[serde(default = "default_rope_base")]
    pub rope_base_local: f32,
    #[serde(default = "default_rope_n_pairs")]
    pub rope_n_pairs: usize,
    #[serde(default)]
    pub rope_per_layer: bool,

    #[serde(default = "default_true")]
    pub mudd: bool,
    #[serde(default = "default_true")]
    pub mudd_mlp: bool,
    #[serde(default = "default_mudd_taps")]
    pub mudd_taps: usize,
    #[serde(default = "default_mudd_inter")]
    pub mudd_inter: usize,
    #[serde(default)]
    pub mudd_layers: Vec<usize>,
    #[serde(default)]
    pub mudd_tap_idx: HashMap<String, Vec<usize>>,

    #[serde(default)]
    pub final_softcap: f32,
    #[serde(default)]
    pub softcap_logits: Option<serde_json::Value>,

    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
    #[serde(default)]
    pub quantization: Option<serde_json::Value>,
    #[serde(default)]
    pub quantization_config: Option<serde_json::Value>,

    /// Convention strings, checked by [`ModelArgs::validate`].
    #[serde(flatten)]
    pub conventions: HashMap<String, serde_json::Value>,
}

fn default_true() -> bool {
    true
}
fn default_head_dim() -> usize {
    128
}
fn default_mlp_type() -> String {
    "swiglu".into()
}
fn default_softmax_scale() -> f32 {
    0.1
}
fn default_attn_gate_channels() -> usize {
    128
}
fn default_gate_scale() -> f32 {
    2.0
}
fn default_ve_gate_channels() -> usize {
    12
}
fn default_ve_stored_heads() -> usize {
    2
}
fn default_xsa_eps() -> f32 {
    1e-4
}
fn default_global_window() -> i64 {
    -1
}
fn default_sliding_window() -> usize {
    1024
}
fn default_rope_base() -> f32 {
    1024.0
}
fn default_rope_n_pairs() -> usize {
    32
}
fn default_mudd_taps() -> usize {
    3
}
fn default_mudd_inter() -> usize {
    32
}

/// The conventions this port implements, keyed by config field. A field the
/// config omits is accepted (the reference defaults to the same behavior).
const CONVENTIONS: &[(&str, &str)] = &[
    ("rope_style", "interleaved_pairs_odd_lane_sign_flip"),
    (
        "sliding_window_convention",
        "k >= q - sliding_window, inclusive of the query token (span = sliding_window + 1 keys)",
    ),
    (
        "mudd_hist_convention",
        "hist[0] = rms_norm(embedding); hist[j] = output of layer j-1",
    ),
    ("rms_norm_eps_mode", "torch_finfo_default"),
    ("mudd_accumulation", "ordered_left_to_right"),
    ("qk_norm", "rms_pre_rope"),
    ("rope_cos_sin_dtype", "bfloat16"),
    ("ve_head_slice", "first_num_key_value_heads"),
    ("attn_gate_applied", "per_head_before_o_proj"),
    ("mudd_r_site", "resid"),
    ("pos_mode", "rope"),
];

/// Boolean conventions and the value implemented.
const FLAG_CONVENTIONS: &[(&str, bool)] = &[
    ("rms_norm_has_weight", false),
    ("ve_applied_before_qk_norm", true),
];

impl ModelArgs {
    /// Refuse every config this port would run differently from the
    /// reference.
    pub fn validate(&self) -> Result<(), String> {
        let fail = |msg: String| Err(format!("Limite: {msg}"));
        for (key, want) in CONVENTIONS {
            if let Some(got) = self.conventions.get(*key)
                && got.as_str() != Some(want)
            {
                return fail(format!("{key} is {got}, only {want:?} is implemented"));
            }
        }
        for (key, want) in FLAG_CONVENTIONS {
            if let Some(got) = self.conventions.get(*key)
                && got.as_bool() != Some(*want)
            {
                return fail(format!("{key} is {got}, only {want} is implemented"));
            }
        }
        if self.rope_per_layer {
            return fail("rope_per_layer=true needs a per-layer frequency table".into());
        }
        if self.mlp_type != "swiglu" {
            return fail(format!("mlp_type {:?} is not supported", self.mlp_type));
        }
        if !self.tie_word_embeddings {
            return fail("only the tied embedding head is implemented".into());
        }
        if self.quantization.is_some() || self.quantization_config.is_some() {
            return fail("quantized checkpoints are not supported".into());
        }
        if self.global_window != -1 {
            return fail(format!(
                "global_window {} is not supported (only -1, full causal)",
                self.global_window
            ));
        }
        let n = self.num_hidden_layers;
        let h = self.num_attention_heads;
        let kv = self.num_key_value_heads;
        if h == 0 || kv == 0 || !h.is_multiple_of(kv) {
            return fail(format!("{h} heads do not group over {kv} KV heads"));
        }
        if !self.head_dim.is_multiple_of(2) || 2 * self.rope_n_pairs > self.head_dim {
            return fail(format!(
                "rope_n_pairs {} does not fit head_dim {}",
                self.rope_n_pairs, self.head_dim
            ));
        }
        if self.attn_gate_channels > self.hidden_size || self.ve_gate_channels > self.hidden_size {
            return fail("gate channels exceed hidden_size".into());
        }
        if !self.ve_layers.is_empty() && (self.ve_dim > self.head_dim || self.ve_stored_heads == 0)
        {
            return fail(format!(
                "value embeddings of {} x {} do not fit head_dim {}",
                self.ve_stored_heads, self.ve_dim, self.head_dim
            ));
        }
        for (name, layers) in [
            ("ve_layers", &self.ve_layers),
            ("xsa_layers", &self.xsa_layers),
            ("global_layers", &self.global_layers),
            ("mudd_layers", &self.mudd_layers),
        ] {
            if let Some(bad) = layers.iter().find(|&&l| l >= n) {
                return fail(format!("{name} names layer {bad} of {n}"));
            }
        }
        if self.mudd {
            self.tap_indices()?;
        }
        self.sigmoid_softcap()?;
        Ok(())
    }

    /// `mudd_tap_idx` with integer keys, checked against `mudd_layers`:
    /// every tap reads an earlier history slot (`hist[j]`, `j <= layer`).
    pub fn tap_indices(&self) -> Result<BTreeMap<usize, Vec<usize>>, String> {
        let mut taps = BTreeMap::new();
        for (key, idx) in &self.mudd_tap_idx {
            let layer: usize = key
                .parse()
                .map_err(|_| format!("Limite: mudd_tap_idx key {key:?} is not a layer"))?;
            if idx.is_empty() || idx.len() > self.mudd_taps {
                return Err(format!(
                    "Limite: layer {layer} has {} MUDD taps, between 1 and {} are allowed",
                    idx.len(),
                    self.mudd_taps
                ));
            }
            if let Some(bad) = idx.iter().find(|&&j| j > layer) {
                return Err(format!(
                    "Limite: layer {layer} taps history slot {bad}, which is not computed yet"
                ));
            }
            taps.insert(layer, idx.clone());
        }
        let mut declared: Vec<usize> = self.mudd_layers.clone();
        declared.sort_unstable();
        declared.dedup();
        if taps.keys().copied().collect::<Vec<_>>() != declared {
            return Err(format!(
                "Limite: mudd_tap_idx covers {:?} but mudd_layers is {:?}",
                taps.keys().collect::<Vec<_>>(),
                declared
            ));
        }
        Ok(taps)
    }

    /// `softcap_logits` as `(a, b, c)`; only `kind: sigmoid` is supported.
    pub fn sigmoid_softcap(&self) -> Result<Option<(f32, f32, f32)>, String> {
        let Some(cap) = self.softcap_logits.as_ref().filter(|c| !c.is_null()) else {
            return Ok(None);
        };
        let kind = cap.get("kind").and_then(|k| k.as_str());
        if kind != Some("sigmoid") {
            return Err(format!(
                "Limite: softcap_logits.kind={kind:?}; only sigmoid is supported"
            ));
        }
        let num = |k: &str, default: f32| {
            cap.get(k)
                .and_then(|v| v.as_f64())
                .map_or(default, |v| v as f32)
        };
        let c = num("c", 1.0);
        if c == 0.0 || !c.is_finite() {
            return Err(format!(
                "Limite: softcap_logits.c={c} must be finite and nonzero"
            ));
        }
        Ok(Some((num("a", 0.0), num("b", 0.0), c)))
    }

    pub fn eos_token_ids(&self) -> Vec<i32> {
        match &self.eos_token_id {
            Some(serde_json::Value::Number(n)) => {
                n.as_i64().map(|id| vec![id as i32]).unwrap_or_default()
            }
            Some(serde_json::Value::Array(arr)) => arr
                .iter()
                .filter_map(|v| v.as_i64().map(|n| n as i32))
                .collect(),
            _ => Vec::new(),
        }
    }
}
