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

//! Configuration for the Qwen 3.5 MTP drafter (`model_type: "qwen3_5_mtp"`).
//!
//! Mirrors the upstream `Qwen3_5MTPConfig` shape: a small top level
//! (`block_size`, `tie_word_embeddings`) plus a nested `text_config` that is a
//! full copy of the target's text config. See
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/speculative/drafters/qwen3_5_mtp/config.py.
//!
//! The drafter exercises only a subset of the nested text config (the MTP
//! head is a single full-attention decoder layer sized on the target's
//! `hidden_size`), so [`Qwen35MtpTextConfig`] deserializes just that subset
//! and ignores the rest. The full target-side `Qwen35Config` lives in the
//! `mlxcel` binary crate, above `mlxcel-core` in the dependency graph, so it
//! cannot be reused here; field names, defaults, and the
//! `rope_parameters`-derived accessors mirror it exactly.

use serde::Deserialize;

/// Quantization arguments for a (hypothetical) quantized drafter checkpoint.
/// The published `mlx-community/Qwen3.8-27B-MTP-bf16` drafter is unquantized;
/// the field exists so a future quantized split loads without a config edit.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35MtpQuantization {
    pub group_size: i32,
    pub bits: i32,
}

/// Subset of the target-mirroring `text_config` the MTP drafter consumes.
///
/// Unknown fields (`layer_types`, `linear_*`, MoE knobs, …) are ignored: the
/// MTP layer is always full attention (upstream builds it with
/// `full_attention_interval=1`), so none of the linear-attention or layout
/// fields participate.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35MtpTextConfig {
    #[serde(default)]
    pub model_type: String,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    /// RoPE parameters in the same free-form dict shape the target's
    /// `Qwen35Config` parses (`rope_theta`, `partial_rotary_factor`, plus
    /// MRoPE keys the drafter ignores).
    #[serde(default)]
    pub rope_parameters: Option<serde_json::Value>,
    /// Number of MTP decoder layers. `1` on every published Qwen 3.5 / 3.6 /
    /// 3.8 checkpoint.
    #[serde(default = "default_mtp_num_hidden_layers")]
    pub mtp_num_hidden_layers: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization: Option<Qwen35MtpQuantization>,
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

fn default_mtp_num_hidden_layers() -> usize {
    1
}

impl Qwen35MtpTextConfig {
    /// Effective quantization group size (matches `Qwen35Config::group_size`).
    pub fn group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map(|q| q.group_size)
            .unwrap_or(64)
    }

    /// Effective quantization bit width (matches `Qwen35Config::bits`).
    pub fn bits(&self) -> i32 {
        self.quantization.as_ref().map(|q| q.bits).unwrap_or(4)
    }

    /// RoPE base frequency. Default mirrors the target-side
    /// `Qwen35Config::rope_theta` fallback.
    pub fn rope_theta(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .and_then(|rp| rp.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(100_000.0)
    }

    /// Partial rotary factor. Default mirrors the target-side
    /// `Qwen35Config::partial_rotary_factor` fallback.
    pub fn partial_rotary_factor(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .and_then(|rp| rp.get("partial_rotary_factor"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(0.25)
    }

    /// Per-head dimension, falling back to `hidden_size / num_attention_heads`.
    ///
    /// The fallback is lazy and division-safe. It used to be
    /// `unwrap_or(self.hidden_size / self.num_attention_heads)`, and
    /// `Option::unwrap_or` takes a *value*: the division was evaluated on every
    /// call, including when `head_dim` was `Some`. A `num_attention_heads: 0` in
    /// a drafter `config.json` therefore panicked here with "attempt to divide
    /// by zero" even for a config that otherwise specified the head geometry in
    /// full. That mattered because the drafter is loaded from disk inside the
    /// batch worker at the first speculative request rather than at startup, and
    /// every core inference worker runs under `run_core_thread_or_abort`, which
    /// converts an uncaught panic into `std::process::abort()`: a config scalar
    /// took down the whole server process.
    ///
    /// [`Qwen35MtpConfig::normalize`] is the real gate (it rejects a zero head
    /// count with a named error before any weight is touched). The `checked_div`
    /// here is defense in depth for any path that resolves head geometry without
    /// having normalized first: 0 is a benign sentinel because every downstream
    /// consumer of a head dim is a reshape or a projection load that rejects it
    /// with a shape error, which is recoverable where a panic is not.
    pub fn head_dim_resolved(&self) -> usize {
        self.head_dim.unwrap_or_else(|| {
            self.hidden_size
                .checked_div(self.num_attention_heads)
                .unwrap_or(0)
        })
    }

    /// Rotary dimensions per head (`head_dim * partial_rotary_factor`),
    /// matching the target-side `Qwen35Config::rope_dims`.
    pub fn rope_dims(&self) -> i32 {
        (self.head_dim_resolved() as f32 * self.partial_rotary_factor()) as i32
    }

    /// Whether the mirrored target family is a MoE variant. The dense MTP
    /// drafter does not implement the MoE decoder layer, so loaders reject
    /// this early with a named error instead of failing on a missing
    /// `switch_mlp` weight.
    pub fn is_moe(&self) -> bool {
        self.model_type.contains("moe")
    }

    /// Bounds-check the numeric fields the drafter actually consumes.
    ///
    /// Called from [`Qwen35MtpConfig::normalize`], which is the single gate
    /// every load path goes through, so a rejected value never reaches a
    /// division, a reshape, or an allocation. Checks are ordered so the most
    /// specific message wins: the zero checks on the two head counts precede the
    /// GQA divisibility check, whose `%` would itself divide by zero on
    /// `num_key_value_heads: 0`.
    fn validate_bounds(&self) -> Result<(), String> {
        if self.hidden_size == 0 {
            return Err(format!(
                "Qwen35MtpConfig.text_config.hidden_size ({}) must be > 0: it sizes every \
                 projection and the residual stream",
                self.hidden_size
            ));
        }
        if self.num_attention_heads == 0 {
            return Err(format!(
                "Qwen35MtpConfig.text_config.num_attention_heads ({}) must be > 0: it is the \
                 divisor in the head_dim fallback and a reshape extent in the MTP attention \
                 layer, so zero is an uncatchable abort at first draft rather than a load error",
                self.num_attention_heads
            ));
        }
        if self.num_key_value_heads == 0 {
            return Err(format!(
                "Qwen35MtpConfig.text_config.num_key_value_heads ({}) must be > 0: it is a \
                 reshape extent for the K/V projections in the MTP attention layer",
                self.num_key_value_heads
            ));
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(format!(
                "Qwen35MtpConfig.text_config.num_attention_heads ({}) must be a multiple of \
                 num_key_value_heads ({}): the MTP attention layer reshapes on both counts and \
                 repeats K/V across each query group, which has no meaning for a ragged layout",
                self.num_attention_heads, self.num_key_value_heads
            ));
        }
        if let Some(head_dim) = self.head_dim
            && head_dim == 0
        {
            return Err(
                "Qwen35MtpConfig.text_config.head_dim (0) must be > 0 when present: it is a \
                 reshape extent and the denominator of the attention scale"
                    .to_string(),
            );
        }
        if self.vocab_size == 0 {
            return Err(format!(
                "Qwen35MtpConfig.text_config.vocab_size ({}) must be > 0: it sizes the drafter's \
                 embedding table and LM head",
                self.vocab_size
            ));
        }
        if self.intermediate_size == 0 {
            return Err(format!(
                "Qwen35MtpConfig.text_config.intermediate_size ({}) must be > 0: it sizes the \
                 SwiGLU MLP in the MTP decoder layer",
                self.intermediate_size
            ));
        }
        if !(1..=MAX_MTP_NUM_HIDDEN_LAYERS).contains(&self.mtp_num_hidden_layers) {
            return Err(format!(
                "Qwen35MtpConfig.text_config.mtp_num_hidden_layers ({}) must be between 1 and \
                 {MAX_MTP_NUM_HIDDEN_LAYERS}: the count is used as an allocation size before any \
                 weight is read, so an unbounded value is an allocation-failure abort rather than \
                 a load error, and 0 layers builds a drafter that cannot draft",
                self.mtp_num_hidden_layers
            ));
        }
        Ok(())
    }
}

/// Drafter config for the Qwen 3.5 MTP head.
///
/// Mirrors upstream `Qwen3_5MTPConfig`: `block_size` defaults to
/// `mtp_num_hidden_layers + 2` when absent (the **total** verify-round budget
/// including the bonus token, so the published `block_size: 3` checkpoints
/// draft 2 tokens per round).
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35MtpConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,
    /// Total round budget including the bonus token. `None` in JSON resolves
    /// to `mtp_num_hidden_layers + 2` in [`Self::normalize`], mirroring
    /// upstream `from_dict`'s `flat.setdefault("block_size", mtp_depth + 2)`.
    #[serde(default)]
    pub block_size: Option<usize>,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    pub text_config: Option<Qwen35MtpTextConfig>,
}

fn default_model_type() -> String {
    "qwen3_5_mtp".to_string()
}

fn default_tie_word_embeddings() -> bool {
    true
}

/// Upper bound on `text_config.mtp_num_hidden_layers`.
///
/// This is an allocation bound, not a claim about how deep an MTP head can
/// usefully be. Every published Qwen 3.5 / 3.6 / 3.8 MTP checkpoint ships
/// `mtp_num_hidden_layers: 1`, and upstream derives `block_size` from it as
/// `mtp_num_hidden_layers + 2`, so the field is a draft depth rather than a
/// model dimension: it does not grow with parameter count the way
/// `num_hidden_layers` does.
///
/// The bound is needed because the count is consumed as an allocation size
/// *before* any weight is consulted. `Qwen35MtpDraftModel::from_weights` builds
/// an expected-weight inventory of eleven formatted `String`s per layer and a
/// `Vec::with_capacity` of this size, so a hostile `mtp_num_hidden_layers:
/// 100000000` asks for on the order of 1.1e9 allocations up front. Allocation
/// failure in Rust is an abort, not a catchable panic, and the drafter loads
/// inside the batch worker at first request, so that is a whole-server abort
/// driven by a config scalar.
///
/// 8 leaves eight times the headroom over every checkpoint that exists while
/// keeping the worst case at a few hundred small allocations. If a real
/// checkpoint ever ships a deeper head, raise this constant: that makes the
/// change one reviewed line rather than a silently unbounded capacity, and the
/// rejection names the field and the cap so the reason is obvious from the
/// error alone.
const MAX_MTP_NUM_HIDDEN_LAYERS: usize = 8;

impl Qwen35MtpConfig {
    /// Validate and apply the upstream post-init fixups:
    ///
    /// - `text_config` must be present (upstream raises `ValueError`).
    /// - `block_size` defaults to `mtp_num_hidden_layers + 2`.
    /// - `tie_word_embeddings` follows `text_config.tie_word_embeddings`
    ///   (upstream `__post_init__`).
    /// - MoE text configs are rejected: the dense drafter has no
    ///   `Qwen3_5MoeDecoderLayer` port.
    /// - Numeric fields are bounds-checked (see below).
    ///
    /// The numeric checks exist because two `text_config` scalars reach code
    /// that aborts the process rather than returning an error. A
    /// `num_attention_heads: 0` divides by zero in
    /// [`Qwen35MtpTextConfig::head_dim_resolved`], and an unbounded
    /// `mtp_num_hidden_layers` is used as an allocation size in
    /// `Qwen35MtpDraftModel::from_weights` before any weight is read (see
    /// [`MAX_MTP_NUM_HIDDEN_LAYERS`]). The drafter is loaded lazily inside the
    /// batch worker at the first speculative request, and core worker threads
    /// run under `run_core_thread_or_abort`, which turns an uncaught panic into
    /// `std::process::abort()`; an allocation failure aborts outright. So a
    /// malformed drafter `config.json` used to take down the whole server on
    /// first request instead of declining to classic decode with a named error.
    ///
    /// Rejecting at normalize time is the same policy the tree already applies
    /// to quantization scalars in `crate::layers::validate_quantization_params`
    /// (issue #929): refuse a config-driven value that can only fail
    /// uncatchably later, at parse time, with the field and the value in the
    /// message.
    pub fn normalize(mut self) -> Result<Self, String> {
        let text_cfg = self
            .text_config
            .as_ref()
            .ok_or_else(|| "Qwen35MtpConfig.text_config must be set".to_string())?;
        if text_cfg.is_moe() {
            return Err(format!(
                "qwen3_5_mtp drafter: MoE text_config (model_type {:?}) is not supported; \
                 only the dense MTP decoder layer is implemented",
                text_cfg.model_type
            ));
        }
        text_cfg.validate_bounds()?;
        if self.block_size.is_none() {
            self.block_size = Some(text_cfg.mtp_num_hidden_layers + 2);
        }
        self.tie_word_embeddings = text_cfg.tie_word_embeddings;
        Ok(self)
    }

    /// Nested text config accessor. Call after [`Self::normalize`].
    pub fn text_config(&self) -> &Qwen35MtpTextConfig {
        self.text_config
            .as_ref()
            .expect("Qwen35MtpConfig.text_config must be set (call normalize first)")
    }

    /// Resolved total round budget (bonus token included). Call after
    /// [`Self::normalize`].
    pub fn block_size(&self) -> usize {
        self.block_size
            .expect("Qwen35MtpConfig.block_size resolved by normalize")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published `mlx-community/Qwen3.8-27B-MTP-bf16` config shape:
    /// `block_size: 3` at top level, `text_config` mirroring the target
    /// (with keys the drafter ignores), `mtp_num_hidden_layers: 1`.
    #[test]
    fn parses_published_qwen38_drafter_config_shape() {
        let json = r#"{
            "block_size": 3,
            "model_type": "qwen3_5_mtp",
            "tie_word_embeddings": false,
            "text_config": {
                "model_type": "qwen3_5_text",
                "attn_output_gate": true,
                "full_attention_interval": 4,
                "head_dim": 256,
                "hidden_size": 5120,
                "intermediate_size": 17408,
                "layer_types": ["linear_attention", "full_attention"],
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": false,
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
                "num_hidden_layers": 64,
                "rms_norm_eps": 1e-06,
                "rope_parameters": {
                    "mrope_interleaved": true,
                    "mrope_section": [11, 11, 10],
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 10000000,
                    "rope_type": "default"
                },
                "tie_word_embeddings": false,
                "vocab_size": 248320
            }
        }"#;
        let cfg: Qwen35MtpConfig = serde_json::from_str(json).expect("parse config");
        let cfg = cfg.normalize().expect("normalize");
        assert_eq!(cfg.model_type, "qwen3_5_mtp");
        assert_eq!(cfg.block_size(), 3);
        assert!(!cfg.tie_word_embeddings);
        let tc = cfg.text_config();
        assert_eq!(tc.hidden_size, 5120);
        assert_eq!(tc.num_attention_heads, 24);
        assert_eq!(tc.num_key_value_heads, 4);
        assert_eq!(tc.head_dim_resolved(), 256);
        assert_eq!(tc.mtp_num_hidden_layers, 1);
        assert_eq!(tc.vocab_size, 248320);
        assert_eq!(tc.rope_theta(), 10_000_000.0);
        // 256 * 0.25 = 64 rotary dims — the partial-rotary geometry the
        // target's full-attention layers use.
        assert_eq!(tc.rope_dims(), 64);
    }

    /// `block_size` omitted resolves to `mtp_num_hidden_layers + 2`
    /// (upstream `from_dict` setdefault).
    #[test]
    fn block_size_defaults_to_mtp_depth_plus_two() {
        let json = r#"{
            "model_type": "qwen3_5_mtp",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 512,
                "mtp_num_hidden_layers": 1
            }
        }"#;
        let cfg: Qwen35MtpConfig = serde_json::from_str(json).expect("parse");
        let cfg = cfg.normalize().expect("normalize");
        assert_eq!(cfg.block_size(), 3);
    }

    #[test]
    fn normalize_rejects_missing_text_config() {
        let cfg: Qwen35MtpConfig =
            serde_json::from_str(r#"{"model_type": "qwen3_5_mtp"}"#).expect("parse");
        let err = cfg.normalize().expect_err("must reject");
        assert!(err.contains("text_config"));
    }

    #[test]
    fn normalize_rejects_moe_text_config() {
        let json = r#"{
            "model_type": "qwen3_5_mtp",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 512
            }
        }"#;
        let cfg: Qwen35MtpConfig = serde_json::from_str(json).expect("parse");
        let err = cfg.normalize().expect_err("must reject MoE");
        assert!(err.contains("MoE"), "got: {err}");
    }

    /// `num_attention_heads: 0` must be rejected at normalize time, not divided
    /// by at first draft.
    ///
    /// `head_dim` is explicitly `256` here, which is the point of the fixture:
    /// the config looks fully specified, so nothing should ever need the
    /// `hidden_size / num_attention_heads` fallback. The old
    /// `unwrap_or(hidden_size / num_attention_heads)` evaluated that division
    /// eagerly anyway, and `head_dim_resolved()` panicked with "attempt to
    /// divide by zero" on a config that never asked for the fallback.
    #[test]
    fn normalize_rejects_zero_attention_heads() {
        let json = r#"{
            "model_type": "qwen3_5_mtp",
            "text_config": {
                "model_type": "qwen3_5_text",
                "head_dim": 256,
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 0,
                "num_key_value_heads": 2,
                "vocab_size": 512
            }
        }"#;
        let cfg: Qwen35MtpConfig = serde_json::from_str(json).expect("parse");
        let err = cfg
            .normalize()
            .expect_err("must reject zero attention heads");
        assert!(err.contains("num_attention_heads"), "got: {err}");
    }

    /// [`Qwen35MtpTextConfig::head_dim_resolved`] must not panic even when
    /// called without a preceding `normalize()`. Reaching this assertion at all
    /// is the test: the pre-fix implementation aborted the process here.
    #[test]
    fn head_dim_resolved_does_not_divide_by_zero() {
        let tc = Qwen35MtpTextConfig {
            model_type: "qwen3_5_text".to_string(),
            hidden_size: 64,
            num_attention_heads: 0,
            num_key_value_heads: 2,
            head_dim: None,
            rms_norm_eps: 1e-6,
            intermediate_size: 128,
            vocab_size: 512,
            rope_parameters: None,
            mtp_num_hidden_layers: 1,
            tie_word_embeddings: false,
            quantization: None,
        };
        assert_eq!(tc.head_dim_resolved(), 0);
        assert_eq!(tc.rope_dims(), 0);
    }

    /// An unbounded `mtp_num_hidden_layers` is an allocation size the model
    /// loader consumes before reading any weight, so it is bounded here.
    #[test]
    fn normalize_rejects_absurd_mtp_layer_count() {
        let json = r#"{
            "model_type": "qwen3_5_mtp",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 512,
                "mtp_num_hidden_layers": 100000000
            }
        }"#;
        let cfg: Qwen35MtpConfig = serde_json::from_str(json).expect("parse");
        let err = cfg
            .normalize()
            .expect_err("must reject absurd mtp layer count");
        assert!(err.contains("mtp_num_hidden_layers"), "got: {err}");
        assert!(err.contains("100000000"), "got: {err}");
    }

    /// GQA layout: `Qwen35MtpAttention` reshapes on both head counts and
    /// repeats K/V across each query group, so a ragged layout is rejected.
    #[test]
    fn normalize_rejects_non_divisible_gqa_head_counts() {
        let json = r#"{
            "model_type": "qwen3_5_mtp",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_attention_heads": 6,
                "num_key_value_heads": 4,
                "vocab_size": 512
            }
        }"#;
        let cfg: Qwen35MtpConfig = serde_json::from_str(json).expect("parse");
        let err = cfg.normalize().expect_err("must reject ragged GQA layout");
        assert!(err.contains("num_attention_heads"), "got: {err}");
        assert!(err.contains("num_key_value_heads"), "got: {err}");
    }
}
