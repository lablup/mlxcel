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

//! `config.json` of a Raon-1 mini checkpoint (`model_type: "raon1_mini"`).
//!
//! Field names and defaults follow `configuration_raon1_mini.py` at
//! `KRAFTON/Raon-1-mini@6691a076`. [`Raon1MiniConfig::from_json_str`] also
//! rejects the configurations the reference forward cannot run, by name, and
//! any quantization other than the original checkpoint's MXFP8 experts and the
//! two NVFP4 exports (`raon1_mini_quant.rs`, #3437).

use serde::Deserialize;
use serde_json::Value;

use super::reasoning_effort_steering::ReasoningEffortSteering;

/// `rda_config` block: the Raon Delta Attention geometry and the 1-based
/// layer lists that split the stack into RDA and MLA layers.
#[derive(Debug, Clone, Deserialize)]
pub struct RaonRdaConfig {
    pub num_heads: Option<usize>,
    pub head_dim: Option<usize>,
    #[serde(default = "default_conv_kernel")]
    pub conv_kernel_size: usize,
    #[serde(default = "default_true")]
    pub full_rank_gate: bool,
    #[serde(default = "default_gate_lower_bound")]
    pub gate_lower_bound: Option<f32>,
    #[serde(default)]
    pub rda_layers: Vec<usize>,
    #[serde(default)]
    pub mla_layers: Vec<usize>,
}

/// Text configuration of Raon-1 mini.
#[derive(Debug, Clone, Deserialize)]
pub struct Raon1MiniConfig {
    #[serde(default)]
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: Option<usize>,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default = "default_situ_beta")]
    pub activation_situ_beta: f32,
    #[serde(default = "default_situ_linear_beta")]
    pub activation_situ_linear_beta: f32,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_mla_latent_norm_eps")]
    pub mla_latent_norm_eps: f32,
    #[serde(default = "default_selector_eps")]
    pub raon_mhc_selector_eps: f32,
    #[serde(default = "default_streams")]
    pub raon_mhc_streams: usize,
    #[serde(default = "default_post_factor")]
    pub raon_mhc_post_factor: f32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub num_experts: usize,
    #[serde(default = "default_top_k")]
    pub num_experts_per_token: usize,
    pub moe_intermediate_size: usize,
    #[serde(default = "default_one")]
    pub num_shared_experts: usize,
    #[serde(default = "default_sigmoid")]
    pub moe_router_activation_func: String,
    #[serde(default = "default_true")]
    pub moe_renormalize: bool,
    #[serde(default = "default_one_f32")]
    pub routed_scaling_factor: f32,
    #[serde(default = "default_one")]
    pub first_k_dense_replace: usize,
    #[serde(default = "default_one")]
    pub moe_layer_freq: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    #[serde(default = "default_true")]
    pub mla_use_nope: bool,
    #[serde(default = "default_true")]
    pub mla_use_output_gate: bool,
    pub rda_config: RaonRdaConfig,
    #[serde(default = "default_decay_rank")]
    pub rda_decay_rank: usize,
    #[serde(default)]
    pub num_nextn_predict_layers: usize,
    #[serde(default)]
    pub eos_token_id: Option<Value>,
    /// Fields the reference rejects; carried only so a config that sets
    /// them fails by name instead of running a different model.
    #[serde(default)]
    pub use_grouped_topk: bool,
    #[serde(default)]
    pub routed_expert_hidden_size: Option<usize>,
    #[serde(default)]
    pub latent_moe_use_norm: bool,
    #[serde(default)]
    pub attn_res_block_size: Option<usize>,
    /// Reasoning-effort steering (#3435); `None` means alpha 0 everywhere.
    #[serde(default)]
    pub reasoning_effort_steering: Option<ReasoningEffortSteering>,
    /// Every other top-level key (`quantization_config`, the vendor MLX
    /// export's markers, ...), checked by [`Raon1MiniConfig::validate`].
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

fn default_conv_kernel() -> usize {
    4
}
fn default_true() -> bool {
    true
}
fn default_gate_lower_bound() -> Option<f32> {
    Some(-5.0)
}
fn default_hidden_act() -> String {
    "situ".to_string()
}
fn default_situ_beta() -> f32 {
    4.0
}
fn default_situ_linear_beta() -> f32 {
    25.0
}
fn default_rms_norm_eps() -> f32 {
    1e-5
}
fn default_mla_latent_norm_eps() -> f32 {
    1e-6
}
fn default_selector_eps() -> f32 {
    1e-5
}
fn default_streams() -> usize {
    4
}
fn default_post_factor() -> f32 {
    2.0
}
fn default_top_k() -> usize {
    6
}
fn default_one() -> usize {
    1
}
fn default_one_f32() -> f32 {
    1.0
}
fn default_sigmoid() -> String {
    "sigmoid".to_string()
}
fn default_decay_rank() -> usize {
    128
}

/// Generation EOS ids of the release (`generation_config.json`): `<|im_end|>`
/// and `<|endoftext|>`.
pub const DEFAULT_EOS_TOKEN_IDS: [i32; 2] = [259202, 259200];

impl Raon1MiniConfig {
    /// Parse and validate a `config.json`.
    pub fn from_json_str(json: &str) -> Result<Self, String> {
        let value: Value =
            serde_json::from_str(json).map_err(|e| format!("raon1_mini config.json: {e}"))?;
        // The quantization check runs on the raw JSON first, so an unsupported
        // export whose other fields would not even parse still fails by name.
        reject_unsupported_export(&value)?;
        let config: Self =
            serde_json::from_value(value).map_err(|e| format!("raon1_mini config.json: {e}"))?;
        config.validate()?;
        Ok(config)
    }

    /// `head_dim`, or `hidden_size / num_attention_heads` when unset.
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads.max(1))
    }

    pub fn rda_num_heads(&self) -> usize {
        self.rda_config
            .num_heads
            .unwrap_or(self.num_attention_heads)
    }

    pub fn rda_head_dim(&self) -> usize {
        self.rda_config.head_dim.unwrap_or_else(|| self.head_dim())
    }

    /// `num_heads * head_dim` of an RDA layer.
    pub fn rda_projection_dim(&self) -> usize {
        self.rda_num_heads() * self.rda_head_dim()
    }

    /// `qk_nope_head_dim + qk_rope_head_dim`, the MLA score width.
    pub fn q_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// Zero-based layer `i` is RDA when `i + 1` is in `rda_layers`.
    pub fn is_rda_layer(&self, layer_idx: usize) -> bool {
        self.rda_config.rda_layers.contains(&(layer_idx + 1))
    }

    /// Zero-based indices of the MLA layers.
    pub fn mla_layer_indices(&self) -> Vec<usize> {
        (0..self.num_hidden_layers)
            .filter(|&i| !self.is_rda_layer(i))
            .collect()
    }

    pub fn is_moe_layer(&self, layer_idx: usize) -> bool {
        self.num_experts > 0
            && layer_idx >= self.first_k_dense_replace
            && layer_idx.is_multiple_of(self.moe_layer_freq.max(1))
    }

    /// EOS ids from `eos_token_id`, else the release defaults.
    pub fn eos_token_ids(&self) -> Vec<i32> {
        let parsed = super::parse_optional_eos_token_ids(&self.eos_token_id);
        if parsed.is_empty() {
            DEFAULT_EOS_TOKEN_IDS.to_vec()
        } else {
            parsed
        }
    }

    /// The load errors the reference raises (or would silently run wrong).
    pub fn validate(&self) -> Result<(), String> {
        reject_unsupported_export(&Value::Object(self.extra.clone()))?;
        let fail = |msg: String| Err(format!("raon1_mini: {msg}"));
        if self.hidden_act != "situ" {
            return fail(format!(
                "hidden_act {:?} is not supported; Raon-1 mini uses situ",
                self.hidden_act
            ));
        }
        if self.moe_router_activation_func != "sigmoid" {
            return fail(format!(
                "moe_router_activation_func {:?} is not supported; the router is sigmoid",
                self.moe_router_activation_func
            ));
        }
        if self.use_grouped_topk {
            return fail("use_grouped_topk is not supported".into());
        }
        if self.routed_expert_hidden_size.is_some() {
            return fail("routed_expert_hidden_size (latent MoE) is not supported".into());
        }
        if self.latent_moe_use_norm {
            return fail("latent_moe_use_norm is not supported".into());
        }
        if !self.mla_use_nope {
            return fail("mla_use_nope false (rotary MLA) is not supported".into());
        }
        if self.attn_res_block_size.is_some() {
            return fail("attn_res_block_size (Attention Residuals) is not supported".into());
        }
        if self.raon_mhc_streams < 2 {
            return fail(format!(
                "raon_mhc_streams must be at least 2, got {}",
                self.raon_mhc_streams
            ));
        }
        if self.moe_layer_freq < 1 {
            return fail("moe_layer_freq must be at least 1".into());
        }
        if !self.rda_config.full_rank_gate {
            return fail("rda_config.full_rank_gate false is not supported".into());
        }
        if self.rda_config.conv_kernel_size < 1 {
            return fail("rda_config.conv_kernel_size must be at least 1".into());
        }
        if self.num_hidden_layers == 0 || self.hidden_size == 0 || self.num_attention_heads == 0 {
            return fail(
                "num_hidden_layers, hidden_size and num_attention_heads must be positive".into(),
            );
        }
        if self.num_experts > 0
            && (self.num_experts_per_token == 0 || self.num_experts_per_token > self.num_experts)
        {
            return fail(format!(
                "num_experts_per_token {} must be in 1..={}",
                self.num_experts_per_token, self.num_experts
            ));
        }
        self.validate_geometry()?;
        if let Some(steering) = &self.reasoning_effort_steering {
            steering
                .validate(self.num_hidden_layers)
                .map_err(|e| format!("raon1_mini: {e}"))?;
            if let Some(layer) = steering.layers.iter().find(|&&l| !self.is_moe_layer(l)) {
                return fail(format!(
                    "reasoning_effort_steering lists layer {layer}, which is not a MoE layer"
                ));
            }
        }
        validate_layer_partition(
            &self.rda_config.rda_layers,
            &self.rda_config.mla_layers,
            self.num_hidden_layers,
        )
    }
}

impl Raon1MiniConfig {
    /// Every dimension the loader turns into an MLX shape, as the `i32` MLX
    /// uses, with the products it forms. A `usize` above `i32::MAX` would
    /// otherwise truncate in an `as i32` cast and could match a crafted
    /// tensor, and an overflowing product panics (debug) or wraps (release)
    /// before any weight is read. Zero widths are refused too: a zero RDA
    /// `head_dim` makes the query scale `inf`, and `raon_mhc_streams` above
    /// [`MAX_MHC_STREAMS`] would size the `[n, n]` chart from the config
    /// alone.
    fn validate_geometry(&self) -> Result<(), String> {
        let fail = |msg: String| Err(format!("raon1_mini: {msg}"));
        let positive = [
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("moe_intermediate_size", self.moe_intermediate_size),
            ("kv_lora_rank", self.kv_lora_rank),
            ("qk_nope_head_dim", self.qk_nope_head_dim),
            ("v_head_dim", self.v_head_dim),
            ("rda_config.num_heads", self.rda_num_heads()),
            ("rda_config.head_dim", self.rda_head_dim()),
        ];
        if let Some((name, _)) = positive.iter().find(|(_, v)| *v == 0) {
            return fail(format!("{name} must be positive"));
        }
        if self.raon_mhc_streams > MAX_MHC_STREAMS {
            return fail(format!(
                "raon_mhc_streams {} is above the supported {MAX_MHC_STREAMS}",
                self.raon_mhc_streams
            ));
        }
        let n = self.raon_mhc_streams;
        let products: [(&str, &[usize]); 12] = [
            ("vocab_size", &[self.vocab_size]),
            ("num_experts", &[self.num_experts]),
            ("q_lora_rank", &[self.q_lora_rank]),
            ("intermediate_size", &[self.intermediate_size]),
            ("moe_intermediate_size", &[self.moe_intermediate_size]),
            (
                "rda_config.conv_kernel_size",
                &[self.rda_config.conv_kernel_size],
            ),
            ("raon_mhc_streams * hidden_size", &[n, self.hidden_size]),
            ("(raon_mhc_streams - 1)^2 selector rows", &[n - 1, n - 1, 1]),
            (
                "3 * rda num_heads * head_dim",
                &[3, self.rda_num_heads(), self.rda_head_dim()],
            ),
            (
                "num_attention_heads * (qk_nope_head_dim + qk_rope_head_dim)",
                &[
                    self.num_attention_heads,
                    self.qk_nope_head_dim.saturating_add(self.qk_rope_head_dim),
                ],
            ),
            (
                "num_attention_heads * v_head_dim",
                &[self.num_attention_heads, self.v_head_dim],
            ),
            (
                "kv_lora_rank + qk_rope_head_dim",
                &[self.kv_lora_rank.saturating_add(self.qk_rope_head_dim)],
            ),
        ];
        for (name, factors) in products {
            let product = factors
                .iter()
                .try_fold(1usize, |acc, &f| acc.checked_mul(f))
                .filter(|&p| p <= i32::MAX as usize);
            if product.is_none() {
                return fail(format!("{name} overflows an MLX (i32) dimension"));
            }
        }
        Ok(())
    }
}

/// Upper bound on `raon_mhc_streams` (the release uses 4). The mHC chart is
/// `[n, n]` and the selector `[2n + (n-1)^2, n * hidden]`, so the stream
/// count is bounded before any of them is built.
pub const MAX_MHC_STREAMS: usize = 64;

/// `rda_layers` and `mla_layers` (1-based) must partition `1..=layers`.
///
/// Memory and time are bounded by the entries the JSON actually lists, not
/// by `num_layers`: an untrusted `num_hidden_layers` of `usize::MAX` must
/// fail by name rather than size a bitmap or scan from it.
pub fn validate_layer_partition(
    rda_layers: &[usize],
    mla_layers: &[usize],
    num_layers: usize,
) -> Result<(), String> {
    let listed = rda_layers.len() + mla_layers.len();
    let mut seen = std::collections::HashSet::with_capacity(listed);
    for &layer in rda_layers.iter().chain(mla_layers) {
        if layer == 0 || layer > num_layers {
            return Err(format!(
                "raon1_mini: rda_config layer {layer} is outside 1..={num_layers} (the lists are 1-based)"
            ));
        }
        if !seen.insert(layer) {
            return Err(format!(
                "raon1_mini: rda_config layer {layer} is listed twice in rda_layers / mla_layers"
            ));
        }
    }
    // The entries are now distinct and in range, so fewer than `num_layers`
    // of them leave a gap within `1..=listed + 1`.
    if let Some(missing) = (1..=num_layers.min(listed + 1)).find(|l| !seen.contains(l)) {
        return Err(format!(
            "raon1_mini: rda_config rda_layers and mla_layers do not cover layer {missing} of 1..={num_layers}"
        ));
    }
    Ok(())
}

/// Refuse the checkpoints this port does not run, before any weight loads.
/// The original checkpoint and both NVFP4 exports load (#3437); any other
/// quantization fails by name (`raon1_mini_quant.rs`).
fn reject_unsupported_export(config: &Value) -> Result<(), String> {
    super::raon1_mini_quant::check_export(config)
}

#[cfg(test)]
#[path = "raon1_mini_config_tests.rs"]
mod tests;
