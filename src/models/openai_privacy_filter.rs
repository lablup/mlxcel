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

//! OpenAI Privacy Filter (`openai_privacy_filter`): a bidirectional sparse-MoE
//! token classifier (8 layers, 128 experts, top-4) that tags PII with BIOES
//! labels. It is a GPT-OSS stack without the causal cache: biased GQA
//! attention with per-head sinks over a symmetric 128-token band, interleaved
//! YaRN RoPE scaled by `head_dim^-0.25` on both q and k, and clipped-SwiGLU
//! experts whose projections and accumulation run in f32. A `score` head
//! maps the final norm to the `id2label` tags.
//!
//! Differences from `src/models/gpt_oss.rs`: no KV cache, no layer schedule,
//! `traditional` (interleaved) RoPE, `truncate` read from `rope_parameters`,
//! q and k both scaled by `head_dim^-0.25` with SDPA at scale 1, the router
//! and experts in f32, and query-chunked attention (`attention_chunk_size`,
//! 4096) and token-chunked routing (`moe_chunk_size`, 2048).
//!
//! Reference: `mlx_vlm/models/openai_privacy_filter/openai_privacy_filter.py`
//! (Blaizzy/mlx-vlm#2131). The classifier pipeline and the Viterbi decoder
//! live in `src/token_classification/privacy_filter.rs`.

use std::collections::HashMap;
use std::path::Path;

use mlxcel_core::layers::{RMSNorm, UnifiedEmbedding, UnifiedLinear};
use mlxcel_core::weights::WeightMap;
use mlxcel_core::{MlxArray, UniquePtr};
use serde::Deserialize;
use serde_json::Value;

use crate::models::gpt_oss_common::{YarnRopeParams, yarn_rope_freqs};

use super::openai_privacy_filter_layers::PrivacyFilterLayer;

fn default_vocab_size() -> usize {
    200_064
}
fn default_hidden() -> usize {
    640
}
fn default_layers() -> usize {
    8
}
fn default_experts() -> usize {
    128
}
fn default_top_k() -> usize {
    4
}
fn default_head_dim() -> usize {
    64
}
fn default_heads() -> usize {
    14
}
fn default_kv_heads() -> usize {
    2
}
fn default_sliding_window() -> usize {
    128
}
fn default_n_ctx() -> usize {
    128_000
}
fn default_eps() -> f32 {
    1e-5
}
fn default_true() -> bool {
    true
}
fn default_rope_theta() -> f32 {
    150_000.0
}
fn default_attention_chunk() -> usize {
    4096
}
fn default_moe_chunk() -> usize {
    2048
}

/// `config.json` of an `openai_privacy_filter` checkpoint. Defaults are the
/// upstream `ModelConfig` dataclass; `rope_parameters` is lifted into
/// `rope_theta` / `rope_scaling` by the shared transformers-5 lift.
#[derive(Debug, Clone, Deserialize)]
#[serde(remote = "Self")]
pub struct PrivacyFilterArgs {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_hidden")]
    pub hidden_size: usize,
    #[serde(default = "default_hidden")]
    pub intermediate_size: usize,
    #[serde(default = "default_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_experts")]
    pub num_local_experts: usize,
    #[serde(default = "default_top_k")]
    pub num_experts_per_tok: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_kv_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,
    /// The classifier's window (`TokenClassifierModel::context_size`).
    #[serde(default = "default_n_ctx")]
    pub default_n_ctx: usize,
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_true")]
    pub attention_bias: bool,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// The YaRN block (`type`, `factor`, `beta_fast`, `beta_slow`,
    /// `original_max_position_embeddings`, `truncate`).
    #[serde(default)]
    pub rope_scaling: Option<HashMap<String, Value>>,
    /// Queries per attention chunk (upstream default 4096).
    #[serde(default = "default_attention_chunk")]
    pub attention_chunk_size: usize,
    /// Tokens per routing chunk (upstream default 2048).
    #[serde(default = "default_moe_chunk")]
    pub moe_chunk_size: usize,
    #[serde(default)]
    pub quantization: Option<HashMap<String, Value>>,
}

mlxcel_core::impl_rope_lifting_deserialize!(PrivacyFilterArgs);

impl PrivacyFilterArgs {
    /// Parse and validate a `config.json` body.
    pub fn from_json(config: &str) -> Result<Self, String> {
        let args: Self = serde_json::from_str(config)
            .map_err(|e| format!("Failed to parse openai_privacy_filter config.json: {e}"))?;
        args.validate()?;
        Ok(args)
    }

    fn validate(&self) -> Result<(), String> {
        if self.num_key_value_heads == 0 || self.num_attention_heads % self.num_key_value_heads != 0
        {
            return Err("num_attention_heads must be divisible by num_key_value_heads".into());
        }
        if self.attention_chunk_size == 0 {
            return Err("attention_chunk_size must be positive".into());
        }
        if self.moe_chunk_size == 0 {
            return Err("moe_chunk_size must be positive".into());
        }
        if self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.num_local_experts {
            return Err(format!(
                "num_experts_per_tok {} must be between 1 and num_local_experts {}",
                self.num_experts_per_tok, self.num_local_experts
            ));
        }
        if self.head_dim % 2 != 0 {
            return Err("head_dim must be even for RoPE".into());
        }
        let rope_type = self.rope_scaling.as_ref().map_or("yarn", |scaling| {
            scaling
                .get("type")
                .or_else(|| scaling.get("rope_type"))
                .and_then(Value::as_str)
                .unwrap_or("yarn")
        });
        if rope_type != "yarn" {
            return Err(format!(
                "OpenAI Privacy Filter currently requires YaRN RoPE, got rope type `{rope_type}`"
            ));
        }
        Ok(())
    }

    /// The YaRN frequencies and attention factor of this config
    /// (`PrivacyFilterRoPE`). Missing keys take the upstream defaults;
    /// `truncate` defaults to `false`.
    pub(crate) fn yarn(&self) -> (Vec<f32>, f32) {
        let scaling = self.rope_scaling.as_ref();
        let read = |key: &str, default: f64| -> f32 {
            scaling
                .and_then(|s| s.get(key))
                .and_then(Value::as_f64)
                .unwrap_or(default) as f32
        };
        let params = YarnRopeParams {
            dims: self.head_dim,
            base: self.rope_theta,
            factor: read("factor", 32.0),
            original_max_position_embeddings: read("original_max_position_embeddings", 4096.0),
            beta_fast: read("beta_fast", 32.0),
            beta_slow: read("beta_slow", 1.0),
            mscale: 1.0,
            mscale_all_dim: 0.0,
            truncate: scaling
                .and_then(|s| s.get("truncate"))
                .and_then(Value::as_bool)
                .unwrap_or(false),
        };
        yarn_rope_freqs(&params)
    }

    /// `(group_size, bits, mode)` for one projection: the per-prefix entry of
    /// `quantization` when the checkpoint has one (the router is quantized at
    /// 8 bits / group 64 by `quant_predicate`), else the top-level pair.
    pub(crate) fn quant_for(&self, prefix: &str) -> (i32, i32, String) {
        let Some(map) = self.quantization.as_ref() else {
            return (64, 4, "affine".to_string());
        };
        let entry = map.get(prefix).and_then(Value::as_object);
        let read_i32 = |key: &str, default: i64| -> i32 {
            entry
                .and_then(|e| e.get(key))
                .or_else(|| map.get(key))
                .and_then(Value::as_i64)
                .unwrap_or(default) as i32
        };
        let mode = entry
            .and_then(|e| e.get("mode"))
            .or_else(|| map.get("mode"))
            .and_then(Value::as_str)
            .unwrap_or("affine")
            .to_string();
        (read_i32("group_size", 64), read_i32("bits", 4), mode)
    }

    /// The top-level `(group_size, bits, mode)` the expert planes use.
    pub(crate) fn expert_quantization(&self) -> (i32, i32, String) {
        self.quant_for("")
    }
}

/// The encoder plus its `score` head: `[B, T]` token ids to `[B, T, labels]`
/// logits.
pub struct PrivacyFilterModel {
    embed_tokens: UnifiedEmbedding,
    layers: Vec<PrivacyFilterLayer>,
    norm: RMSNorm,
    score: UnifiedLinear,
    vocab_rows: usize,
    num_labels: usize,
    args: PrivacyFilterArgs,
}

/// Split `array` in two equal halves along `axis` (negative axes allowed).
fn split_half(array: &MlxArray, axis: i32) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    let shape = mlxcel_core::array_shape(array);
    let axis = if axis < 0 {
        shape.len() as i32 + axis
    } else {
        axis
    } as usize;
    let half = shape[axis] / 2;
    let starts = vec![0; shape.len()];
    let mut first_stop = shape.clone();
    first_stop[axis] = half;
    let mut second_start = starts.clone();
    second_start[axis] = half;
    (
        mlxcel_core::slice(array, &starts, &first_stop),
        mlxcel_core::slice(array, &second_start, &shape),
    )
}

/// Bring the Hugging Face expert layout to the split `gate_proj` / `up_proj`
/// / `down_proj` planes the loader reads, as upstream `Model.sanitize`: the
/// fused `gate_up_proj` `[E, hidden, 2 * inter]` is transposed and split in
/// two along the output axis, its bias split along the last axis, and
/// `down_proj` `[E, inter, hidden]` is transposed and renamed `.weight`. The
/// earlier fused MLX runtime layout (`gate_up_proj.<leaf>`, quantization
/// companions included) is split without a transpose. Published mlx-community
/// checkpoints are already split and pass through untouched.
pub fn sanitize_weights(weights: &mut WeightMap) {
    let keys: Vec<String> = weights.keys().cloned().collect();
    for key in keys {
        if key.ends_with(".mlp.experts.gate_up_proj") {
            let value = weights.remove(&key).expect("key listed");
            let prefix = key.trim_end_matches("gate_up_proj");
            let transposed = mlxcel_core::swap_axes(&value, -1, -2);
            let (gate, up) = split_half(&transposed, -2);
            weights.insert(format!("{prefix}gate_proj.weight"), gate);
            weights.insert(format!("{prefix}up_proj.weight"), up);
        } else if key.ends_with(".mlp.experts.gate_up_proj_bias") {
            let value = weights.remove(&key).expect("key listed");
            let prefix = key.trim_end_matches("gate_up_proj_bias");
            let (gate, up) = split_half(&value, -1);
            weights.insert(format!("{prefix}gate_proj.bias"), gate);
            weights.insert(format!("{prefix}up_proj.bias"), up);
        } else if let Some((prefix, leaf)) = key.split_once(".mlp.experts.gate_up_proj.") {
            let value = weights.remove(&key).expect("key listed");
            let axis = if leaf == "bias" { -1 } else { -2 };
            let (gate, up) = split_half(&value, axis);
            weights.insert(format!("{prefix}.mlp.experts.gate_proj.{leaf}"), gate);
            weights.insert(format!("{prefix}.mlp.experts.up_proj.{leaf}"), up);
        } else if key.ends_with(".mlp.experts.down_proj") {
            let value = weights.remove(&key).expect("key listed");
            weights.insert(
                format!("{key}.weight"),
                mlxcel_core::swap_axes(&value, -1, -2),
            );
        } else if key.ends_with(".mlp.experts.down_proj_bias") {
            let value = weights.remove(&key).expect("key listed");
            weights.insert(key.replace("down_proj_bias", "down_proj.bias"), value);
        }
    }
}

impl PrivacyFilterModel {
    /// Parse the checkpoint's `config.json`.
    pub fn read_args(model_dir: &Path) -> Result<PrivacyFilterArgs, String> {
        let config = std::fs::read_to_string(model_dir.join("config.json"))
            .map_err(|e| format!("Failed to read config.json: {e}"))?;
        PrivacyFilterArgs::from_json(&config)
    }

    /// Load a checkpoint directory through the shared text-weight funnel.
    pub fn load(model_dir: &Path) -> Result<Self, String> {
        let args = Self::read_args(model_dir)?;
        Self::load_with(model_dir, args)
    }

    /// Load the weights of `model_dir` under caller-supplied `args` (tests
    /// override the chunk sizes to compare chunked and unchunked forwards).
    pub fn load_with(model_dir: &Path, args: PrivacyFilterArgs) -> Result<Self, String> {
        let mut weights = crate::models::load_text_weights(model_dir, None)?;
        sanitize_weights(&mut weights);
        Self::from_weights(&weights, args)
    }

    /// Build from sanitized weights.
    pub fn from_weights(weights: &WeightMap, args: PrivacyFilterArgs) -> Result<Self, String> {
        let (embed_gs, embed_bits, _) = args.quant_for("model.embed_tokens");
        let vocab_rows = weights
            .get("model.embed_tokens.weight")
            .map(|w| mlxcel_core::array_shape(w)[0] as usize)
            .ok_or_else(|| "Missing weight: model.embed_tokens.weight".to_string())?;
        let embed_tokens =
            UnifiedEmbedding::from_weights(weights, "model.embed_tokens", embed_gs, embed_bits)?;

        let (freqs, attention_factor) = args.yarn();
        let rope_freqs = mlxcel_core::from_slice_f32(&freqs, &[freqs.len() as i32]);
        let layers = (0..args.num_hidden_layers)
            .map(|index| {
                PrivacyFilterLayer::from_weights(
                    weights,
                    &args,
                    index,
                    &rope_freqs,
                    attention_factor,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let norm_weight = weights
            .get("model.norm.weight")
            .map(|w| mlxcel_core::copy(w))
            .ok_or_else(|| "Missing weight: model.norm.weight".to_string())?;
        let norm = RMSNorm::new(norm_weight, args.rms_norm_eps);

        let num_labels = weights
            .get("score.weight")
            .map(|w| mlxcel_core::array_shape(w)[0] as usize)
            .ok_or_else(|| "Missing weight: score.weight".to_string())?;
        let (score_gs, score_bits, score_mode) = args.quant_for("score");
        let score = UnifiedLinear::from_weights_with_mode(
            weights,
            "score",
            score_gs,
            score_bits,
            &score_mode,
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            score,
            vocab_rows,
            num_labels,
            args,
        })
    }

    pub fn args(&self) -> &PrivacyFilterArgs {
        &self.args
    }

    /// Rows of the token embedding table (the ids `log_probs` may be given).
    pub fn vocab_rows(&self) -> usize {
        self.vocab_rows
    }

    /// Outputs of the `score` head.
    pub fn num_labels(&self) -> usize {
        self.num_labels
    }

    /// Final hidden states `[B, T, hidden]` for `[B, T]` int32 ids.
    pub fn hidden_states(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        let mut h = self.embed_tokens.forward(input_ids);
        for layer in &self.layers {
            h = layer.forward(&h);
        }
        self.norm.forward(&h)
    }

    /// Per-token label logits `[B, T, labels]` in the activation dtype.
    pub fn logits(&self, input_ids: &MlxArray) -> UniquePtr<MlxArray> {
        self.score.forward(&self.hidden_states(input_ids))
    }
}

#[cfg(test)]
#[path = "openai_privacy_filter_tests.rs"]
mod tests;
