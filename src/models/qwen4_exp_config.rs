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

//! Qwen4-Exp text configuration.
//!
//! [`Qwen4ExpConfig::resolve`] mirrors upstream `TextConfig.__post_init__`
//! (`mlx_vlm/models/qwen4_exp/config.py`): it fills the derived defaults and
//! rejects every combination upstream rejects, each with an error naming the
//! field. It also rejects the two config features this port leaves out of
//! scope, external PLE storage and the FP8 checkpoint layout, rather than
//! loading them wrong.

use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;

pub const LINEAR_ATTENTION: &str = "linear_attention";
/// `layer_types` entry for a Qwen Sparse Attention layer. Checkpoints spell it
/// `full_attention`; upstream renames it on load and so does [`Qwen4ExpConfig::resolve`].
pub const QWEN_SPARSE_ATTENTION: &str = "qwen_sparse_attention";

const REQUIRED_ROPE_KEYS: [&str; 4] = [
    "mrope_section",
    "type",
    "rope_theta",
    "partial_rotary_factor",
];

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen4ExpConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: Option<usize>,
    pub linear_num_value_heads: usize,
    pub linear_num_key_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub shared_expert_intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_hc_count")]
    pub hc_count: usize,
    #[serde(default = "default_hc_lowrank")]
    pub hc_lowrank: usize,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    #[serde(default = "default_full_attention_interval")]
    pub full_attention_interval: usize,
    /// One-indexed decoder layers carrying a PLE n-gram embedding.
    #[serde(default)]
    pub ple_layer_ids: Vec<usize>,
    #[serde(default)]
    pub ple_embed_dim: Option<usize>,
    #[serde(default = "default_ple_conv_kernel_size")]
    pub ple_conv_kernel_size: usize,
    #[serde(default = "default_ngram_size")]
    pub ngram_size: usize,
    #[serde(default = "default_heads_per_ngram")]
    pub heads_per_ngram: usize,
    #[serde(default = "default_ngram_vocab_size_base")]
    pub ngram_vocab_size_base: u64,
    #[serde(default = "default_make_ngram_vocab_size_divisible_by")]
    pub make_ngram_vocab_size_divisible_by: u64,
    #[serde(default = "default_split_ngram_parts")]
    pub split_ngram_parts: usize,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default)]
    pub ple_storage: Option<Value>,
    #[serde(default = "default_indexer_n_heads")]
    pub indexer_n_heads: usize,
    #[serde(default = "default_indexer_kv_heads")]
    pub indexer_kv_heads: usize,
    #[serde(default = "default_indexer_head_dim")]
    pub indexer_head_dim: usize,
    #[serde(default = "default_indexer_budget")]
    pub indexer_budget: usize,
    #[serde(default = "default_indexer_compress_ratio")]
    pub indexer_compress_ratio: usize,
    #[serde(default = "default_output_gate_type")]
    pub output_gate_type: Option<String>,
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub eos_token_id: Option<Value>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub rope_parameters: Option<Value>,
    #[serde(default)]
    pub quantization: Option<Value>,
}

fn default_hc_count() -> usize {
    4
}
fn default_hc_lowrank() -> usize {
    320
}
fn default_full_attention_interval() -> usize {
    4
}
fn default_ple_conv_kernel_size() -> usize {
    4
}
fn default_ngram_size() -> usize {
    3
}
fn default_heads_per_ngram() -> usize {
    8
}
fn default_ngram_vocab_size_base() -> u64 {
    20_000_000
}
fn default_make_ngram_vocab_size_divisible_by() -> u64 {
    128
}
fn default_split_ngram_parts() -> usize {
    128
}
fn default_seed() -> u64 {
    1234
}
fn default_indexer_n_heads() -> usize {
    4
}
fn default_indexer_kv_heads() -> usize {
    1
}
fn default_indexer_head_dim() -> usize {
    128
}
fn default_indexer_budget() -> usize {
    2048
}
fn default_indexer_compress_ratio() -> usize {
    4
}
fn default_output_gate_type() -> Option<String> {
    Some("sigmoid".to_string())
}
fn default_true() -> bool {
    true
}

/// Activation applied to the GatedDeltaNet output-norm gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputGate {
    Sigmoid,
    Silu,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Qwen4ExpConfigError {
    #[error(
        "Qwen4-Exp layer_types must contain one entry per decoder layer: {got} != num_hidden_layers {expected}"
    )]
    LayerTypesLength { got: usize, expected: usize },
    #[error(
        "Qwen4-Exp layer_types contains unsupported entries {0:?}; only linear_attention and qwen_sparse_attention (full_attention) are defined"
    )]
    UnsupportedLayerTypes(Vec<String>),
    #[error("Qwen4-Exp requires hc_count > 1, got {0}")]
    HcCount(usize),
    #[error("Qwen4-Exp output_gate_type must be 'sigmoid' or 'silu', got {0:?}")]
    OutputGateType(Option<String>),
    #[error("Qwen4-Exp QSA requires indexer_kv_heads=1, got {0}")]
    IndexerKvHeads(usize),
    #[error("Qwen4-Exp indexer_compress_ratio must be positive")]
    IndexerCompressRatio,
    #[error(
        "Qwen4-Exp indexer_budget ({budget}) must be divisible by indexer_compress_ratio ({ratio})"
    )]
    IndexerBudget { budget: usize, ratio: usize },
    #[error(
        "Qwen4-Exp ple_embed_dim ({ple_embed_dim}) must be divisible by the total number of n-gram heads ((ngram_size - 1) * heads_per_ngram = {ngram_heads})"
    )]
    PleEmbedDim {
        ple_embed_dim: usize,
        ngram_heads: usize,
    },
    #[error(
        "Qwen4-Exp ple_layer_ids are one-indexed decoder layers in [1, {num_hidden_layers}], got {invalid:?}"
    )]
    PleLayerIds {
        invalid: Vec<usize>,
        num_hidden_layers: usize,
    },
    #[error("Qwen4-Exp PLE is only supported on linear-attention layers, got ple_layer_ids {0:?}")]
    PleOnNonLinearLayer(Vec<usize>),
    #[error(
        "Qwen4-Exp eos_token_id is required when ple_layer_ids is non-empty: the n-gram context resets at EOS"
    )]
    MissingEosForPle,
    #[error("Qwen4-Exp eos_token_id must be an integer or a non-empty list of integers, got {0}")]
    EosTokenId(String),
    #[error("Qwen4-Exp rope_parameters must contain keys {REQUIRED_ROPE_KEYS:?}; missing {0:?}")]
    RopeParameters(Vec<String>),
    #[error("Qwen4-Exp rope_parameters.{field} has the wrong type: {value}")]
    RopeParameterType { field: &'static str, value: String },
    #[error(
        "Qwen4-Exp ple_storage (external PLE n-gram storage) is not supported by mlxcel; load a checkpoint whose n-gram table is in the safetensors shards"
    )]
    PleStorage,
    #[error("Qwen4-Exp {field} must be positive")]
    Zero { field: &'static str },
    #[error("Qwen4-Exp {field} ({value}) exceeds the MLX i32 dimension limit")]
    I32Limit { field: &'static str, value: u128 },
    #[error(
        "Qwen4-Exp num_attention_heads ({heads}) must be divisible by num_key_value_heads ({kv_heads})"
    )]
    HeadGrouping { heads: usize, kv_heads: usize },
    #[error("Qwen4-Exp num_experts_per_tok ({per_token}) must not exceed num_experts ({experts})")]
    ExpertsPerToken { per_token: usize, experts: usize },
    #[error(
        "Qwen4-Exp rotary width head_dim * partial_rotary_factor = {rope_dims} must be even and in [2, head_dim = {head_dim}]"
    )]
    RopeWidth { rope_dims: usize, head_dim: usize },
    #[error(
        "Qwen4-Exp rope_parameters.mrope_section must hold three entries in [0, {half}], got {section:?}"
    )]
    MropeSection { section: Vec<i32>, half: usize },
    #[error(
        "Qwen4-Exp quantization.{module} declares group_size={group_size}, bits={bits}: {reason}"
    )]
    Quantization {
        module: String,
        group_size: i32,
        bits: i32,
        reason: String,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen4ExpRope {
    pub mrope_section: Vec<i32>,
    pub rope_theta: f32,
    pub partial_rotary_factor: f32,
}

/// [`Qwen4ExpConfig`] after [`Qwen4ExpConfig::resolve`]: every derived
/// default filled in and every invariant the model relies on checked.
#[derive(Debug, Clone)]
pub struct ResolvedQwen4ExpConfig {
    pub raw: Qwen4ExpConfig,
    pub head_dim: usize,
    pub ple_embed_dim: usize,
    /// Sorted, de-duplicated, one-indexed.
    pub ple_layer_ids: Vec<usize>,
    /// `None` when the kinds follow from `full_attention_interval`: a derived
    /// list would be `num_hidden_layers` long before any weight bounds it.
    pub layer_types: Option<Vec<String>>,
    pub output_gate: OutputGate,
    pub rope: Qwen4ExpRope,
    /// First entry when the config gives a list, as upstream's n-gram hash reads it.
    pub eos_token_id: Option<i64>,
    pub quantization: QuantizationTable,
}

impl Qwen4ExpConfig {
    pub fn resolve(self) -> Result<ResolvedQwen4ExpConfig, Qwen4ExpConfigError> {
        for (field, value) in [
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("linear_num_value_heads", self.linear_num_value_heads),
            ("linear_num_key_heads", self.linear_num_key_heads),
            ("linear_key_head_dim", self.linear_key_head_dim),
            ("linear_value_head_dim", self.linear_value_head_dim),
            ("linear_conv_kernel_dim", self.linear_conv_kernel_dim),
            ("num_experts", self.num_experts),
            ("num_experts_per_tok", self.num_experts_per_tok),
            ("vocab_size", self.vocab_size),
            ("hc_lowrank", self.hc_lowrank),
            ("full_attention_interval", self.full_attention_interval),
            ("indexer_n_heads", self.indexer_n_heads),
            ("indexer_head_dim", self.indexer_head_dim),
            ("indexer_budget", self.indexer_budget),
        ] {
            if value == 0 {
                return Err(Qwen4ExpConfigError::Zero { field });
            }
        }
        if self
            .ple_storage
            .as_ref()
            .is_some_and(|value| !value.is_null())
        {
            return Err(Qwen4ExpConfigError::PleStorage);
        }

        let head_dim = self
            .head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads);
        let ple_embed_dim = self.ple_embed_dim.unwrap_or(self.hidden_size);
        self.check_i32_dimensions(head_dim, ple_embed_dim)?;
        let mut ple_layer_ids = self.ple_layer_ids.clone();
        ple_layer_ids.sort_unstable();
        ple_layer_ids.dedup();

        let layer_types: Option<Vec<String>> = self.layer_types.as_ref().map(|kinds| {
            kinds
                .iter()
                .map(|kind| {
                    if kind == "full_attention" {
                        QWEN_SPARSE_ATTENTION.to_string()
                    } else {
                        kind.clone()
                    }
                })
                .collect()
        });
        if let Some(kinds) = &layer_types {
            if kinds.len() != self.num_hidden_layers {
                return Err(Qwen4ExpConfigError::LayerTypesLength {
                    got: kinds.len(),
                    expected: self.num_hidden_layers,
                });
            }
            let mut unsupported: Vec<String> = kinds
                .iter()
                .filter(|kind| *kind != LINEAR_ATTENTION && *kind != QWEN_SPARSE_ATTENTION)
                .cloned()
                .collect();
            if !unsupported.is_empty() {
                unsupported.sort();
                unsupported.dedup();
                return Err(Qwen4ExpConfigError::UnsupportedLayerTypes(unsupported));
            }
        }
        let is_linear = |layer_idx: usize| {
            is_linear_layer(
                layer_types.as_deref(),
                self.full_attention_interval,
                layer_idx,
            )
        };
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(Qwen4ExpConfigError::HeadGrouping {
                heads: self.num_attention_heads,
                kv_heads: self.num_key_value_heads,
            });
        }
        if self.num_experts_per_tok > self.num_experts {
            return Err(Qwen4ExpConfigError::ExpertsPerToken {
                per_token: self.num_experts_per_tok,
                experts: self.num_experts,
            });
        }
        if self.hc_count <= 1 {
            return Err(Qwen4ExpConfigError::HcCount(self.hc_count));
        }
        let output_gate = match self.output_gate_type.as_deref() {
            Some("sigmoid") => OutputGate::Sigmoid,
            Some("silu") => OutputGate::Silu,
            _ => {
                return Err(Qwen4ExpConfigError::OutputGateType(
                    self.output_gate_type.clone(),
                ));
            }
        };
        if self.indexer_kv_heads != 1 {
            return Err(Qwen4ExpConfigError::IndexerKvHeads(self.indexer_kv_heads));
        }
        if self.indexer_compress_ratio == 0 {
            return Err(Qwen4ExpConfigError::IndexerCompressRatio);
        }
        if !self
            .indexer_budget
            .is_multiple_of(self.indexer_compress_ratio)
        {
            return Err(Qwen4ExpConfigError::IndexerBudget {
                budget: self.indexer_budget,
                ratio: self.indexer_compress_ratio,
            });
        }
        let ngram_heads = self.ngram_size.saturating_sub(1) * self.heads_per_ngram;
        if ngram_heads == 0 || !ple_embed_dim.is_multiple_of(ngram_heads) {
            return Err(Qwen4ExpConfigError::PleEmbedDim {
                ple_embed_dim,
                ngram_heads,
            });
        }
        let invalid: Vec<usize> = ple_layer_ids
            .iter()
            .copied()
            .filter(|&id| id < 1 || id > self.num_hidden_layers)
            .collect();
        if !invalid.is_empty() {
            return Err(Qwen4ExpConfigError::PleLayerIds {
                invalid,
                num_hidden_layers: self.num_hidden_layers,
            });
        }
        let non_linear: Vec<usize> = ple_layer_ids
            .iter()
            .copied()
            .filter(|&id| !is_linear(id - 1))
            .collect();
        if !non_linear.is_empty() {
            return Err(Qwen4ExpConfigError::PleOnNonLinearLayer(non_linear));
        }
        let eos_token_id = parse_eos(self.eos_token_id.as_ref())?;
        if !ple_layer_ids.is_empty() && eos_token_id.is_none() {
            return Err(Qwen4ExpConfigError::MissingEosForPle);
        }
        if !ple_layer_ids.is_empty() {
            for (field, value) in [
                ("ple_conv_kernel_size", self.ple_conv_kernel_size),
                ("split_ngram_parts", self.split_ngram_parts),
            ] {
                if value == 0 {
                    return Err(Qwen4ExpConfigError::Zero { field });
                }
            }
            if self.ngram_vocab_size_base == 0 {
                return Err(Qwen4ExpConfigError::Zero {
                    field: "ngram_vocab_size_base",
                });
            }
            if self.make_ngram_vocab_size_divisible_by == 0 {
                return Err(Qwen4ExpConfigError::Zero {
                    field: "make_ngram_vocab_size_divisible_by",
                });
            }
        }
        let rope = parse_rope(self.rope_parameters.as_ref())?;
        check_rope_geometry(&rope, head_dim)?;
        let quantization = QuantizationTable::from_config(self.quantization.as_ref())?;

        Ok(ResolvedQwen4ExpConfig {
            raw: self,
            head_dim,
            ple_embed_dim,
            ple_layer_ids,
            layer_types,
            output_gate,
            rope,
            eos_token_id,
            quantization,
        })
    }
}

impl Qwen4ExpConfig {
    /// Every size the model hands to MLX as an `i32`, alone or as the product
    /// it is used as, must fit one.
    fn check_i32_dimensions(
        &self,
        head_dim: usize,
        ple_embed_dim: usize,
    ) -> Result<(), Qwen4ExpConfigError> {
        let product = |factors: &[usize]| {
            factors
                .iter()
                .fold(1u128, |acc, &factor| acc.saturating_mul(factor as u128))
        };
        let key_dim = product(&[self.linear_key_head_dim, self.linear_num_key_heads]);
        let value_dim = product(&[self.linear_value_head_dim, self.linear_num_value_heads]);
        for (field, value) in [
            ("hidden_size", self.hidden_size as u128),
            ("num_attention_heads", self.num_attention_heads as u128),
            ("num_key_value_heads", self.num_key_value_heads as u128),
            ("head_dim", head_dim as u128),
            (
                "linear_num_value_heads",
                self.linear_num_value_heads as u128,
            ),
            ("linear_num_key_heads", self.linear_num_key_heads as u128),
            ("linear_key_head_dim", self.linear_key_head_dim as u128),
            ("linear_value_head_dim", self.linear_value_head_dim as u128),
            (
                "linear_conv_kernel_dim",
                self.linear_conv_kernel_dim as u128,
            ),
            ("num_experts", self.num_experts as u128),
            ("num_experts_per_tok", self.num_experts_per_tok as u128),
            (
                "shared_expert_intermediate_size",
                self.shared_expert_intermediate_size as u128,
            ),
            ("moe_intermediate_size", self.moe_intermediate_size as u128),
            ("vocab_size", self.vocab_size as u128),
            ("hc_count", self.hc_count as u128),
            ("hc_lowrank", self.hc_lowrank as u128),
            ("ple_embed_dim", ple_embed_dim as u128),
            ("ple_conv_kernel_size", self.ple_conv_kernel_size as u128),
            ("ngram_size", self.ngram_size as u128),
            ("heads_per_ngram", self.heads_per_ngram as u128),
            ("split_ngram_parts", self.split_ngram_parts as u128),
            ("indexer_n_heads", self.indexer_n_heads as u128),
            ("indexer_head_dim", self.indexer_head_dim as u128),
            ("indexer_budget", self.indexer_budget as u128),
            (
                "indexer_compress_ratio",
                self.indexer_compress_ratio as u128,
            ),
            (
                "hc_count * hidden_size",
                product(&[self.hc_count, self.hidden_size]),
            ),
            (
                "2 * num_attention_heads * head_dim",
                product(&[2, self.num_attention_heads, head_dim]),
            ),
            (
                "num_key_value_heads * head_dim",
                product(&[self.num_key_value_heads, head_dim]),
            ),
            (
                "(indexer_n_heads + 1) * indexer_head_dim",
                product(&[
                    self.indexer_n_heads.saturating_add(1),
                    self.indexer_head_dim,
                ]),
            ),
            ("linear key dim", key_dim),
            ("linear value dim", value_dim),
            (
                "linear conv dim",
                key_dim.saturating_mul(2).saturating_add(value_dim),
            ),
            (
                "(ple_conv_kernel_size - 1) * ngram_size",
                product(&[self.ple_conv_kernel_size.saturating_sub(1), self.ngram_size]),
            ),
        ] {
            if value > i32::MAX as u128 {
                return Err(Qwen4ExpConfigError::I32Limit { field, value });
            }
        }
        Ok(())
    }
}

fn is_linear_layer(
    layer_types: Option<&[String]>,
    full_attention_interval: usize,
    layer_idx: usize,
) -> bool {
    match layer_types {
        Some(kinds) => kinds[layer_idx] == LINEAR_ATTENTION,
        None => !(layer_idx + 1).is_multiple_of(full_attention_interval),
    }
}

fn rope_dims(head_dim: usize, partial_rotary_factor: f32) -> usize {
    (head_dim as f32 * partial_rotary_factor) as usize
}

/// The rotary tables are `rope_dims` wide and slice the first `rope_dims`
/// channels of every head, and `InterleavedMRoPE` walks each section once
/// per forward and indexes the `[3, ..]` position rows by section number.
fn check_rope_geometry(rope: &Qwen4ExpRope, head_dim: usize) -> Result<(), Qwen4ExpConfigError> {
    let dims = rope_dims(head_dim, rope.partial_rotary_factor);
    if dims < 2 || dims > head_dim || !dims.is_multiple_of(2) {
        return Err(Qwen4ExpConfigError::RopeWidth {
            rope_dims: dims,
            head_dim,
        });
    }
    let half = dims / 2;
    let section = &rope.mrope_section;
    if section.len() != 3
        || section
            .iter()
            .any(|&entry| entry < 0 || entry as usize > half)
    {
        return Err(Qwen4ExpConfigError::MropeSection {
            section: section.clone(),
            half,
        });
    }
    Ok(())
}

impl ResolvedQwen4ExpConfig {
    pub fn is_linear_layer(&self, layer_idx: usize) -> bool {
        is_linear_layer(
            self.layer_types.as_deref(),
            self.raw.full_attention_interval,
            layer_idx,
        )
    }

    /// Zero-based index into `ple_layer_ids` of decoder layer `layer_idx`.
    pub fn ple_index(&self, layer_idx: usize) -> Option<usize> {
        self.ple_layer_ids
            .iter()
            .position(|&one_indexed| one_indexed == layer_idx + 1)
    }

    /// Rotary width shared by attention and the QSA indexer.
    pub fn rope_dims(&self) -> usize {
        rope_dims(self.head_dim, self.rope.partial_rotary_factor)
    }

    pub fn ngram_heads(&self) -> usize {
        (self.raw.ngram_size - 1) * self.raw.heads_per_ngram
    }
}

fn parse_eos(value: Option<&Value>) -> Result<Option<i64>, Qwen4ExpConfigError> {
    let invalid = |value: &Value| Qwen4ExpConfigError::EosTokenId(truncate(&value.to_string()));
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => number
            .as_i64()
            .map(Some)
            .ok_or_else(|| invalid(value.unwrap())),
        Some(Value::Array(items)) => match items.first() {
            Some(first) => first
                .as_i64()
                .map(Some)
                .ok_or_else(|| invalid(value.unwrap())),
            None => Err(invalid(value.unwrap())),
        },
        Some(other) => Err(invalid(other)),
    }
}

fn parse_rope(value: Option<&Value>) -> Result<Qwen4ExpRope, Qwen4ExpConfigError> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(Qwen4ExpRope {
            mrope_section: vec![11, 11, 10],
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
        });
    };
    let Some(map) = value.as_object() else {
        return Err(Qwen4ExpConfigError::RopeParameterType {
            field: "rope_parameters",
            value: truncate(&value.to_string()),
        });
    };
    let has = |key: &str| map.contains_key(key) || (key == "type" && map.contains_key("rope_type"));
    let missing: Vec<String> = REQUIRED_ROPE_KEYS
        .iter()
        .filter(|key| !has(key))
        .map(|key| key.to_string())
        .collect();
    if !missing.is_empty() {
        return Err(Qwen4ExpConfigError::RopeParameters(missing));
    }
    let type_error = |field: &'static str| Qwen4ExpConfigError::RopeParameterType {
        field,
        value: truncate(&map[field].to_string()),
    };
    let mrope_section = map["mrope_section"]
        .as_array()
        .and_then(|items| {
            items
                .iter()
                .map(|item| item.as_i64().and_then(|v| i32::try_from(v).ok()))
                .collect::<Option<Vec<i32>>>()
        })
        .ok_or_else(|| type_error("mrope_section"))?;
    let rope_theta = map["rope_theta"]
        .as_f64()
        .ok_or_else(|| type_error("rope_theta"))? as f32;
    let partial_rotary_factor = map["partial_rotary_factor"]
        .as_f64()
        .ok_or_else(|| type_error("partial_rotary_factor"))? as f32;
    Ok(Qwen4ExpRope {
        mrope_section,
        rope_theta,
        partial_rotary_factor,
    })
}

fn truncate(text: &str) -> String {
    const MAX_CHARS: usize = 64;
    if text.chars().count() <= MAX_CHARS {
        text.to_string()
    } else {
        format!("{}...", text.chars().take(MAX_CHARS).collect::<String>())
    }
}

/// The checkpoint's `quantization` block: one default pair plus per-module
/// overrides keyed by module path.
///
/// The released OptiQ quant mixes 2-, 4- and 8-bit affine modules under one
/// 4-bit default. `UnifiedLinear` re-derives a mismatched width from the
/// tensor shapes, but warns once per tensor, and the stacked expert loader
/// (`SwitchLinear`) hands the declared pair straight to `gather_qmm`, so each
/// module has to be loaded with its own declared pair.
#[derive(Debug, Clone, Default)]
pub struct QuantizationTable {
    default: Option<(i32, i32)>,
    overrides: HashMap<String, (i32, i32)>,
}

impl QuantizationTable {
    fn from_config(value: Option<&Value>) -> Result<Self, Qwen4ExpConfigError> {
        let Some(map) = value.and_then(Value::as_object) else {
            return Ok(Self::default());
        };
        let pair_of = |module: &str, entry: &serde_json::Map<String, Value>| {
            let group_size = entry
                .get("group_size")
                .and_then(Value::as_i64)
                .unwrap_or(64) as i32;
            let bits = entry.get("bits").and_then(Value::as_i64).unwrap_or(4) as i32;
            mlxcel_core::layers::validate_quantization_params(group_size, bits).map_err(
                |reason| Qwen4ExpConfigError::Quantization {
                    module: module.to_string(),
                    group_size,
                    bits,
                    reason,
                },
            )?;
            Ok((group_size, bits))
        };
        let default = if map.contains_key("group_size") || map.contains_key("bits") {
            Some(pair_of("<default>", map)?)
        } else {
            None
        };
        let mut overrides = HashMap::new();
        for (module, entry) in map {
            if let Some(entry) = entry.as_object() {
                overrides.insert(
                    super::sanitize::canonical_key(module),
                    pair_of(module, entry)?,
                );
            }
        }
        Ok(Self { default, overrides })
    }

    pub fn for_module(&self, prefix: &str) -> (i32, i32) {
        self.overrides
            .get(prefix)
            .copied()
            .or(self.default)
            .unwrap_or((64, 4))
    }
}

/// Parse and resolve the text config of a `qwen4_exp` checkpoint.
///
/// `full_config` is the whole `config.json`: the top-level `quantization`
/// block is inherited when `text_config` carries none.
pub fn text_config_from_full(full_config: &Value) -> Result<ResolvedQwen4ExpConfig, String> {
    let mut text = full_config
        .get("text_config")
        .cloned()
        .unwrap_or_else(|| full_config.clone());
    if let Some(object) = text.as_object_mut()
        && !object.contains_key("quantization")
        && let Some(quantization) = full_config.get("quantization")
    {
        object.insert("quantization".to_string(), quantization.clone());
    }
    let config: Qwen4ExpConfig = serde_json::from_value(text)
        .map_err(|error| format!("Failed to parse Qwen4-Exp text config: {error}"))?;
    config.resolve().map_err(|error| error.to_string())
}

#[cfg(test)]
#[path = "qwen4_exp_config_tests.rs"]
mod tests;
