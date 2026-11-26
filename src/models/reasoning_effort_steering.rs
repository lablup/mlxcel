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

//! The `reasoning_effort_steering` config block (Raon-1 mini release, #3435).
//!
//! ```json
//! "reasoning_effort_steering": {
//!   "alpha": {"none": 0, "low": 0.3, "medium": 0.1, "high": 0.05, "xhigh": 0.02, "max": 0},
//!   "default": "high",
//!   "layers": [12, 14, 16, 18, 20]
//! }
//! ```
//!
//! Each listed (zero-based) layer adds `alpha * c` to its MoE output in
//! float32 before the mHC mix, where `c` is the layer's
//! `moe.reasoning_effort_steering` vector and `alpha` depends on the
//! request's reasoning effort. Alpha resolution follows
//! `Raon1MiniForCausalLM.set_reasoning_effort` in the reference
//! `modeling_raon1_mini.py` (revision `6691a076`):
//!
//! * no effort selects `default`;
//! * `none` (thinking off) uses the config's `none` alpha when it is above 0,
//!   else [`NO_THINKING_ALPHA`];
//! * any other level uses its own alpha (`max` is 0: steering off);
//! * a level outside `none` plus the alpha keys is an error.
//!
//! This type is shared by the model (per-sequence alpha) and the server
//! (request validation, the 400 message, and the prompt-cache key), so both
//! accept exactly the same set.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

/// The thinking-off level.
pub const NONE_LEVEL: &str = "none";

/// Alpha for `none` when the config's own `none` entry is 0 (the
/// reference's `NO_THINKING_ALPHA`).
pub const NO_THINKING_ALPHA: f32 = 0.1;

/// The published levels, in the order the model card lists them. Accepted
/// levels are reported in this order, then any other config key sorted.
const CANONICAL_ORDER: [&str; 6] = ["none", "low", "medium", "high", "xhigh", "max"];

/// The parsed `reasoning_effort_steering` block.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ReasoningEffortSteering {
    /// Alpha per level, `none` included when the config lists it.
    pub alpha: BTreeMap<String, f64>,
    /// The level a request without one uses.
    pub default: String,
    /// Zero-based decoder layers that carry a steering vector.
    pub layers: Vec<usize>,
}

impl ReasoningEffortSteering {
    /// The block from a parsed `config.json`, or `None` when the config has
    /// none (alpha 0 everywhere).
    ///
    /// # Errors
    /// Returns the message for a block that does not parse.
    pub fn from_config_value(config: &serde_json::Value) -> Result<Option<Self>, String> {
        match config.get("reasoning_effort_steering") {
            None | Some(serde_json::Value::Null) => Ok(None),
            Some(block) => serde_json::from_value(block.clone())
                .map(Some)
                .map_err(|e| format!("reasoning_effort_steering: {e}")),
        }
    }

    /// The validated block from `<model_dir>/config.json`, or `None` when the
    /// file is missing or declares no block.
    ///
    /// # Errors
    /// Returns the message for an unreadable config or an invalid block.
    pub fn from_model_dir(model_dir: &Path) -> Result<Option<Self>, String> {
        let path = model_dir.join("config.json");
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Ok(None);
        };
        let value: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let Some(steering) = Self::from_config_value(&value)? else {
            return Ok(None);
        };
        let num_layers = value
            .get("num_hidden_layers")
            .and_then(serde_json::Value::as_u64)
            .map_or(usize::MAX, |n| n as usize);
        steering.validate(num_layers)?;
        Ok(Some(steering))
    }

    /// The config rules: `default` is an alpha key and not `none`, every
    /// alpha is finite and non-negative, and `layers` is a non-empty list
    /// of distinct indices below `num_layers`.
    ///
    /// # Errors
    /// Returns the first rule the block breaks.
    pub fn validate(&self, num_layers: usize) -> Result<(), String> {
        let fail = |msg: String| Err(format!("reasoning_effort_steering: {msg}"));
        if self.default == NONE_LEVEL {
            return fail("default cannot be none".into());
        }
        if !self.alpha.contains_key(&self.default) {
            return fail(format!(
                "default {:?} is not one of the alpha keys ({})",
                self.default,
                self.alpha.keys().cloned().collect::<Vec<_>>().join(", ")
            ));
        }
        for (level, alpha) in &self.alpha {
            if !alpha.is_finite() || *alpha < 0.0 {
                return fail(format!(
                    "alpha for {level:?} must be finite and non-negative, got {alpha}"
                ));
            }
            if level.trim() != level || level.to_ascii_lowercase() != *level || level.is_empty() {
                return fail(format!(
                    "alpha key {level:?} must be a lowercase level name"
                ));
            }
        }
        if self.layers.is_empty() {
            return fail("layers must not be empty".into());
        }
        let mut seen = std::collections::BTreeSet::new();
        for &layer in &self.layers {
            if layer >= num_layers {
                return fail(format!(
                    "layer {layer} is out of range for {num_layers} decoder layers \
                     (layers are zero-based)"
                ));
            }
            if !seen.insert(layer) {
                return fail(format!("layer {layer} is listed twice"));
            }
        }
        Ok(())
    }

    /// Whether `layer` (zero-based) is steered.
    #[must_use]
    pub fn steers_layer(&self, layer: usize) -> bool {
        self.layers.contains(&layer)
    }

    /// `none` plus every alpha key, in the model card's order.
    #[must_use]
    pub fn accepted_levels(&self) -> Vec<String> {
        let mut levels: Vec<String> = CANONICAL_ORDER
            .iter()
            .filter(|level| **level == NONE_LEVEL || self.alpha.contains_key(**level))
            .map(|level| (*level).to_string())
            .collect();
        levels.extend(
            self.alpha
                .keys()
                .filter(|key| !CANONICAL_ORDER.contains(&key.as_str()))
                .cloned(),
        );
        levels
    }

    /// `raw` trimmed and lowercased, when it is an accepted level.
    ///
    /// No translation: OpenAI's `minimal` is not `low`, and `off` is not
    /// `none`. An unknown spelling is an error naming the accepted set.
    ///
    /// # Errors
    /// Returns the client-facing message for an unknown level.
    pub fn normalize_level(&self, raw: &str) -> Result<String, String> {
        let level = raw.trim().to_ascii_lowercase();
        if self.accepted_levels().contains(&level) {
            Ok(level)
        } else {
            Err(format!(
                "reasoning_effort must be one of {}; got {raw:?}",
                self.accepted_levels().join(", ")
            ))
        }
    }

    /// The alpha a request at `level` runs with (`None`: the default level).
    ///
    /// # Errors
    /// Returns the message for an unknown level.
    pub fn alpha_for(&self, level: Option<&str>) -> Result<f32, String> {
        let level = match level {
            Some(raw) => self.normalize_level(raw)?,
            None => self.default.clone(),
        };
        if level == NONE_LEVEL {
            let configured = self.alpha.get(NONE_LEVEL).copied().unwrap_or(0.0);
            return Ok(if configured > 0.0 {
                configured as f32
            } else {
                NO_THINKING_ALPHA
            });
        }
        self.alpha
            .get(&level)
            .map(|alpha| *alpha as f32)
            .ok_or_else(|| format!("reasoning_effort {level:?} has no alpha"))
    }
}

#[cfg(test)]
#[path = "reasoning_effort_steering_tests.rs"]
mod tests;
