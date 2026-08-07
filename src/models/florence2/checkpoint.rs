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

//! Florence-2 checkpoint plumbing: whole-model config parsing and the
//! weight-key normalization that has to happen before either half of the
//! model is built.
//!
//! Kept apart from `model.rs` because these two run once at load time and
//! have nothing to do with the forward path.
//!
//! Reference:
//! https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/florence2/florence2.py

use anyhow::Result;
use serde_json::Value;

use mlxcel_core::weights::WeightMap;

use crate::vision::encoders::florence2_davit;

use super::{Florence2TextConfig, Florence2VisionConfig};

/// Weight-key prefix of the language model inside a Florence-2 checkpoint.
pub(crate) const FLORENCE2_TEXT_PREFIX: &str = "language_model.";

/// Both halves of a Florence-2 `config.json` plus the fusion-stage fields.
#[derive(Debug, Clone)]
pub struct Florence2Config {
    pub text: Florence2TextConfig,
    pub vision: Florence2VisionConfig,
    /// Placeholder id the processor may emit for the image. Florence-2 does
    /// not actually splice image features into placeholder slots, so any
    /// occurrence is dropped before the prompt is embedded. Upstream defaults
    /// this to the text vocabulary size (51289), one past the last real id.
    pub image_token_id: i32,
}

impl Florence2Config {
    /// Parse a full Florence-2 `config.json` (`model_type: florence2`).
    pub fn from_model_config(config: &Value) -> Result<Self> {
        let text = Florence2TextConfig::from_model_config(config)?;
        let vision = Florence2VisionConfig::from_model_config(config)?;
        let image_token_id = config
            .get("image_token_id")
            .or_else(|| config.get("image_token_index"))
            .and_then(Value::as_i64)
            .map(|v| v as i32)
            .unwrap_or(text.vocab_size);
        Ok(Self {
            text,
            vision,
            image_token_id,
        })
    }
}

/// Normalize a full Florence-2 checkpoint's weight keys.
///
/// Three passes, in order:
///
/// 1. The DaViT conv channels-last remap and `position_ids` drop from
///    [`florence2_davit::sanitize`]. Both of its key guards are specific
///    enough (`convs`/`proj.weight`, `blocks`/`dw.weight`) that running them
///    over the whole checkpoint leaves `language_model.*` untouched, and both
///    are idempotent, so this is a no-op on the `mlx-community` exports that
///    already ship channels-last conv weights.
/// 2. Drop `final_logits_bias`. HuggingFace BART carries it as a registered
///    buffer of zeros; it is not a parameter and there is no LM-head bias in
///    this implementation.
/// 3. BART shared-embedding naming. The encoder and decoder token tables are
///    tied to `model.shared` and exports vary in which of the three they
///    materialize, so fill `model.shared.weight` from an `embed_tokens` copy
///    when only the latter is present.
pub fn sanitize(weights: WeightMap) -> WeightMap {
    let weights = florence2_davit::sanitize(weights);

    let mut out = WeightMap::with_capacity(weights.len());
    for (key, value) in weights {
        if key.contains("final_logits_bias") {
            continue;
        }
        out.insert(key, value);
    }

    let shared = format!("{FLORENCE2_TEXT_PREFIX}model.shared.weight");
    if !out.contains_key(&shared) {
        let source = [
            format!("{FLORENCE2_TEXT_PREFIX}model.encoder.embed_tokens.weight"),
            format!("{FLORENCE2_TEXT_PREFIX}model.decoder.embed_tokens.weight"),
        ]
        .into_iter()
        .find_map(|key| out.get(&key).map(|w| mlxcel_core::copy(w)));
        if let Some(tensor) = source {
            out.insert(shared, tensor);
        }
    }
    out
}
