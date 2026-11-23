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

//! Hy4 configuration: the DeepSeek-V3.2 args plus the Hy4 keys of
//! `config.json`, validated at parse time, and the parity arms.

use serde::Deserialize;

use super::super::ModelArgs;

/// Comma-separated parity arms, read once at model load. Each arm removes or
/// swaps one Hy4-specific term so a logit trace can show that the term is
/// load-bearing: `no_sinks`, `no_gate`, `no_swiglu_limit`, `rope_half_split`.
/// Largest `hc_mult` accepted; the shipped model uses 4.
const MAX_HC_MULT: usize = 64;

pub(crate) const ARM_ENV: &str = "MLXCEL_HY4_ARM";

/// The parity arms in force for one model. All `false` is the shipped model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ParityArms {
    pub(crate) no_sinks: bool,
    pub(crate) no_gate: bool,
    pub(crate) no_swiglu_limit: bool,
    pub(crate) rope_half_split: bool,
}

impl ParityArms {
    /// Parse an [`ARM_ENV`] value. An unknown arm is an error, so a typo
    /// cannot silently run the default model under a diagnostic label.
    pub(crate) fn parse(value: &str) -> Result<Self, String> {
        let mut arms = Self::default();
        for arm in value
            .split(',')
            .map(str::trim)
            .filter(|arm| !arm.is_empty())
        {
            match arm {
                "no_sinks" => arms.no_sinks = true,
                "no_gate" => arms.no_gate = true,
                "no_swiglu_limit" => arms.no_swiglu_limit = true,
                "rope_half_split" => arms.rope_half_split = true,
                other => {
                    return Err(format!(
                        "{ARM_ENV}: unknown arm `{other}` (expected a comma-separated list of \
                         no_sinks, no_gate, no_swiglu_limit, rope_half_split)"
                    ));
                }
            }
        }
        Ok(arms)
    }

    pub(super) fn from_env() -> Result<Self, String> {
        match std::env::var(ARM_ENV) {
            Ok(value) => {
                let arms = Self::parse(&value)?;
                if arms != Self::default() {
                    tracing::warn!(?arms, "{ARM_ENV} is set: this is not the shipped Hy4 model");
                }
                Ok(arms)
            }
            Err(std::env::VarError::NotPresent) => Ok(Self::default()),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err(format!("{ARM_ENV}: the value is not valid UTF-8"))
            }
        }
    }
}

/// Whether a layer owns a DSA indexer or reuses the previous one's selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexerType {
    Full,
    Shared,
}

/// The Hy4 keys of `config.json` that the DeepSeek-V3.2 args do not carry.
#[derive(Debug, Deserialize)]
struct RawExtras {
    #[serde(default = "default_hc_mult")]
    hc_mult: usize,
    #[serde(default = "default_hc_eps")]
    hc_eps: f32,
    #[serde(default = "default_hc_magnitude")]
    hc_magnitude: f32,
    #[serde(default)]
    indexer_types: Option<Vec<String>>,
    #[serde(default)]
    mlp_layer_types: Option<Vec<String>>,
    #[serde(default)]
    first_k_dense_replace: Option<usize>,
    #[serde(default)]
    swiglu_limit: Option<f32>,
    #[serde(default)]
    eos_token_id: Option<serde_json::Value>,
}

fn default_hc_mult() -> usize {
    4
}
fn default_hc_eps() -> f32 {
    1e-6
}
fn default_hc_magnitude() -> f32 {
    2.0
}

/// `eos_token_id` of the shipped config, used when `config.json` names none.
const DEFAULT_EOS_TOKEN_ID: i32 = 120025;

/// Validated Hy4 configuration: the DeepSeek-V3.2 args plus the Hy4 keys.
#[derive(Debug, Clone)]
pub struct HyV4Config {
    /// MLA, indexer and MoE dimensions, shared with DeepSeek-V3.2.
    /// `rope_parameters.rope_theta` is lifted into `rope_theta` (#2448).
    pub base: ModelArgs,
    pub hc_mult: usize,
    pub hc_eps: f32,
    pub hc_magnitude: f32,
    /// One entry per hidden layer.
    pub indexer_types: Vec<IndexerType>,
    /// Per layer: `true` for a MoE (`sparse`) MLP, `false` for a dense one.
    pub mlp_is_moe: Vec<bool>,
    pub swiglu_limit: Option<f32>,
    pub eos_token_ids: Vec<i32>,
}

impl HyV4Config {
    /// Parse and validate a `config.json` document.
    pub fn from_json_str(config: &str) -> Result<Self, String> {
        let value: serde_json::Value = serde_json::from_str(config)
            .map_err(|e| format!("Failed to parse config.json: {e}"))?;
        Self::from_value(value)
    }

    fn from_value(value: serde_json::Value) -> Result<Self, String> {
        let mut base: ModelArgs = serde_json::from_value(value.clone())
            .map_err(|e| format!("Failed to parse hy_v4 config: {e}"))?;
        let raw: RawExtras = serde_json::from_value(value)
            .map_err(|e| format!("Failed to parse hy_v4 config: {e}"))?;
        let layers = base.num_hidden_layers;

        if base.topk_method != "noaux_tc" {
            return Err(format!(
                "hy_v4: unsupported topk_method `{}` (only noaux_tc)",
                base.topk_method
            ));
        }
        if raw.hc_mult == 0 || raw.hc_mult > MAX_HC_MULT {
            return Err(format!(
                "hy_v4: hc_mult must be between 1 and {MAX_HC_MULT}, got {}",
                raw.hc_mult
            ));
        }
        // The hyper-connection mixers are sized 2 * hc_mult and
        // hc_mult * hidden_size; both must fit the i32 shapes MLX takes.
        let widest = i32::try_from(raw.hc_mult)
            .ok()
            .and_then(|m| m.checked_mul(2))
            .and_then(|m| m.checked_mul(i32::try_from(base.hidden_size).ok()?));
        if widest.is_none() {
            return Err(format!(
                "hy_v4: 2 * hc_mult ({}) * hidden_size ({}) does not fit in i32",
                raw.hc_mult, base.hidden_size
            ));
        }

        // Upstream default when the config ships no list.
        let indexer_names = raw.indexer_types.unwrap_or_else(|| {
            (0..layers)
                .map(|i| {
                    if i < 2 || i % 4 == 1 {
                        "full"
                    } else {
                        "shared"
                    }
                    .to_string()
                })
                .collect()
        });
        if indexer_names.len() != layers {
            return Err(format!(
                "hy_v4: `indexer_types` must have one entry per hidden layer, got {} for {layers} \
                 layers",
                indexer_names.len()
            ));
        }
        let indexer_types = indexer_names
            .iter()
            .enumerate()
            .map(|(layer, name)| match name.as_str() {
                "full" => Ok(IndexerType::Full),
                "shared" => Ok(IndexerType::Shared),
                other => Err(format!(
                    "hy_v4: unsupported indexer type `{other}` at indexer_types[{layer}] \
                     (expected `full` or `shared`)"
                )),
            })
            .collect::<Result<Vec<_>, String>>()?;
        if indexer_types.first() == Some(&IndexerType::Shared) {
            return Err(
                "hy_v4: indexer_types[0] is `shared`, but there is no earlier `full` layer whose \
                 top-k selection it could reuse"
                    .to_string(),
            );
        }

        let mlp_is_moe = match raw.mlp_layer_types {
            Some(names) => {
                if names.len() != layers {
                    return Err(format!(
                        "hy_v4: `mlp_layer_types` must have one entry per hidden layer, got {} \
                         for {layers} layers",
                        names.len()
                    ));
                }
                names
                    .iter()
                    .enumerate()
                    .map(|(layer, name)| match name.as_str() {
                        "sparse" => Ok(true),
                        "dense" => Ok(false),
                        other => Err(format!(
                            "hy_v4: unsupported MLP type `{other}` at mlp_layer_types[{layer}] \
                             (expected `dense` or `sparse`)"
                        )),
                    })
                    .collect::<Result<Vec<_>, String>>()?
            }
            // Upstream `first_k_dense_replace` default is 1 for this family.
            None => {
                let dense = raw.first_k_dense_replace.unwrap_or(1);
                (0..layers).map(|layer| layer >= dense).collect()
            }
        };
        if mlp_is_moe.iter().any(|&moe| moe) && base.n_routed_experts.is_none() {
            return Err("hy_v4: a sparse MLP layer needs `n_routed_experts`".to_string());
        }

        // mlx-vlm's indexer rotates interleaved pairs for this family.
        base.indexer_rope_interleave = true;

        let eos_id = |id: &serde_json::Value| -> Result<i32, String> {
            id.as_i64()
                .and_then(|id| i32::try_from(id).ok())
                .filter(|id| *id >= 0)
                .ok_or_else(|| format!("hy_v4: invalid eos_token_id entry {id}"))
        };
        let eos_token_ids = match &raw.eos_token_id {
            Some(id @ serde_json::Value::Number(_)) => vec![eos_id(id)?],
            Some(serde_json::Value::Array(ids)) => {
                ids.iter().map(eos_id).collect::<Result<Vec<_>, _>>()?
            }
            _ => vec![DEFAULT_EOS_TOKEN_ID],
        };

        Ok(Self {
            base,
            hc_mult: raw.hc_mult,
            hc_eps: raw.hc_eps,
            hc_magnitude: raw.hc_magnitude,
            indexer_types,
            mlp_is_moe,
            swiglu_limit: raw.swiglu_limit.filter(|limit| *limit > 0.0),
            eos_token_ids,
        })
    }
}

impl<'de> Deserialize<'de> for HyV4Config {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        Self::from_value(value).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
#[path = "hy_v4_config_tests.rs"]
mod tests;
