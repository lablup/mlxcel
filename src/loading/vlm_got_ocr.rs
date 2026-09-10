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

//! GOT-OCR 2.0 (`model_type: "GOT"`) loader.
//!
//! One architecture, two key layouts in the wild:
//!
//! - `stepfun-ai/GOT-OCR2_0` (the original): `model.vision_tower_high.*` with
//!   the tower's neck as the `nn.Sequential` indices `neck.0..3`,
//!   `model.mm_projector_vary.*`, `model.*` for the Qwen2 decoder, and a tied
//!   `lm_head.weight` copy. Conv weights are torch `[O, I, kH, kW]`.
//! - `mlx-community/GOT-OCR2_0-{bf16,8bit,4bit}` (the conversions):
//!   `vision_tower.*` with the neck renamed to `conv1 / norm1 / conv2 / norm2`,
//!   `multi_modal_projector.*`, `language_model.model.*`, no `lm_head`. Conv
//!   weights are already `[O, kH, kW, I]`.
//!
//! [`canonicalize_got_keys`] folds both onto one naming so a single loader
//! serves them. It targets the *original* spelling for the tower, because
//! [`SamEncoder::from_weights`] already reads `neck.0..3` for DeepSeek-OCR and
//! renaming four keys is smaller than threading a neck-key parameter through
//! the shared encoder.
//!
//! The tower itself is the DeepSeek-OCR SAM ViT-B verbatim (`got_vision_b.py`
//! and `sam.py` agree on geometry, the `1e-6` neck LayerNorm epsilon, and the
//! bias-free `net_2` / `net_3` stride-2 compressor), so
//! [`SamConfig::default`] describes it exactly and the shared
//! `conv_channels_last` shape gate absorbs the layout difference.

use anyhow::Result;
use serde_json::{Value, json};
use std::path::Path;

use crate::LoadedModel;
use crate::models;
use crate::vision;
use crate::vision::encoders::deepseekocr_sam::{SamConfig, SamEncoder};
use crate::vision::processors::got_ocr::GotOcrImageProcessor;
use mlxcel_core::layers::UnifiedLinear;
use mlxcel_core::weights::WeightMap;

use super::{load_vlm_weights_common, read_sanitized_vlm_config, strip_language_model_prefix};

/// Canonical tower prefix after [`canonicalize_got_keys`].
pub(crate) const VISION_PREFIX: &str = "vision_tower";

/// Canonical projector prefix after [`canonicalize_got_keys`].
pub(crate) const PROJECTOR_PREFIX: &str = "multi_modal_projector";

/// `config.json` defaults, used when a checkpoint omits the field. Every
/// released GOT checkpoint states all four, but the tower geometry is fixed in
/// `got_vision_b.py` rather than configurable, so defaulting is safe.
const DEFAULT_IMAGE_TOKEN_LEN: usize = 256;
const DEFAULT_IM_START_TOKEN: i32 = 151857;
const DEFAULT_IM_END_TOKEN: i32 = 151858;
const DEFAULT_IM_PATCH_TOKEN: i32 = 151859;

/// `<|endoftext|>`, the only stop id `config.json` declares.
const DEFAULT_EOS_TOKEN_ID: i32 = 151643;

/// `<|im_end|>`, the turn separator upstream's `KeywordsStoppingCriteria`
/// watches for.
///
/// `config.json` names only 151643 and the model does not emit it at the end of
/// an OCR answer, so a run that stops on the config alone never terminates: it
/// generates until `max_tokens`. The separator has to join the stop set.
const IM_END_STOP_TOKEN_ID: i32 = 151645;

/// The four `nn.Sequential` neck indices and the names the MLX conversion gave
/// them, in `(converted, original)` order.
const NECK_RENAMES: [(&str, &str); 4] = [
    ("conv1", "neck.0"),
    ("norm1", "neck.1"),
    ("conv2", "neck.2"),
    ("norm2", "neck.3"),
];

/// Rewrite one weight key onto the canonical naming.
///
/// Returns the key unchanged when it is already canonical, which is what makes
/// [`canonicalize_got_keys`] idempotent: the converted layout's tower and
/// projector prefixes are the canonical ones, and re-running only has to leave
/// them alone.
fn canonicalize_got_key(key: &str) -> String {
    // Original tower: `model.vision_tower_high.X` -> `vision_tower.X`, with the
    // neck kept at its `neck.N` spelling (which SamEncoder reads directly).
    if let Some(rest) = key.strip_prefix("model.vision_tower_high.") {
        return format!("{VISION_PREFIX}.{rest}");
    }
    // Converted tower: rename the neck back to the indices, pass the rest.
    if let Some(rest) = key.strip_prefix("vision_tower.") {
        for (converted, original) in NECK_RENAMES {
            if let Some(tail) = rest.strip_prefix(&format!("{converted}.")) {
                return format!("{VISION_PREFIX}.{original}.{tail}");
            }
        }
        return key.to_string();
    }
    if let Some(rest) = key.strip_prefix("model.mm_projector_vary.") {
        return format!("{PROJECTOR_PREFIX}.{rest}");
    }
    if key.starts_with("language_model.") || key.starts_with(PROJECTOR_PREFIX) {
        return key.to_string();
    }
    // Original decoder: `model.X` -> `language_model.model.X`, `lm_head.X` ->
    // `language_model.lm_head.X`.
    if key.starts_with("model.") {
        return format!("language_model.{key}");
    }
    if key.starts_with("lm_head.") {
        return format!("language_model.{key}");
    }
    key.to_string()
}

/// Fold either released key layout onto one naming.
///
/// `drop_tied_lm_head` removes the tied `lm_head.weight` copy the original
/// checkpoint ships (151860 x 1024 bf16, roughly 311 MB) when
/// `tie_word_embeddings` is set. The Llama-family backbone ties the head to the
/// embedding itself, so the copy is never read; keeping it would only cost
/// resident memory and let a divergent copy silently win.
pub(crate) fn canonicalize_got_keys(weights: WeightMap, drop_tied_lm_head: bool) -> WeightMap {
    let mut out = WeightMap::new();
    for (key, value) in weights {
        let key = canonicalize_got_key(&key);
        if drop_tied_lm_head && key.starts_with("language_model.lm_head.") {
            continue;
        }
        out.insert(key, value);
    }
    out
}

/// Build the Qwen2 decoder args from GOT's flat `config.json`.
///
/// The text config *is* the root: there is no `text_config` and no
/// `language_config`. Two fixups make it parse as a Llama-family model:
/// `model_type` is rewritten from `"GOT"` to `"qwen2"` so the backbone takes
/// the attention-bias branch (GOT's q/k/v carry biases, like every Qwen2), and
/// the root `quantization` block is left in place for a converted checkpoint.
///
/// `vision_config` is dropped when present. The 4-bit conversion ships it as an
/// empty object and the tower geometry is fixed in code, but leaving an unknown
/// key in the value handed to a `deny_unknown_fields`-free struct is harmless
/// only by accident; removing it keeps the decoder args to decoder fields.
pub(crate) fn got_text_config(full_config: &Value) -> Result<Value> {
    let mut text_config = full_config.clone();
    let obj = super::require_object_mut(&mut text_config, "GOT-OCR config.json")?;
    obj.insert("model_type".to_string(), json!("qwen2"));
    obj.remove("vision_config");
    obj.remove("quantization_config");
    Ok(text_config)
}

/// The stop set: `eos_token_id` from the config plus `<|im_end|>`.
///
/// `eos_token_id` may be a scalar or a list; both spellings appear across the
/// family's `config.json` and `generation_config.json`.
pub(crate) fn got_eos_token_ids(
    full_config: &Value,
    generation_config: Option<&Value>,
) -> Vec<i32> {
    fn push(ids: &mut Vec<i32>, id: i32) {
        if !ids.contains(&id) {
            ids.push(id);
        }
    }

    let mut ids: Vec<i32> = Vec::new();
    for source in [generation_config, Some(full_config)].into_iter().flatten() {
        match source.get("eos_token_id") {
            Some(Value::Number(n)) => {
                if let Some(id) = n.as_i64() {
                    push(&mut ids, id as i32);
                }
            }
            Some(Value::Array(items)) => {
                for id in items.iter().filter_map(Value::as_i64) {
                    push(&mut ids, id as i32);
                }
            }
            _ => {}
        }
    }
    if ids.is_empty() {
        push(&mut ids, DEFAULT_EOS_TOKEN_ID);
    }
    push(&mut ids, IM_END_STOP_TOKEN_ID);
    ids
}

fn config_usize(full_config: &Value, key: &str, default: usize) -> usize {
    full_config
        .get(key)
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .unwrap_or(default)
}

fn config_i32(full_config: &Value, key: &str, default: i32) -> i32 {
    full_config
        .get(key)
        .and_then(Value::as_i64)
        .map(|v| v as i32)
        .unwrap_or(default)
}

/// Load a GOT-OCR 2.0 (`GOT`) VLM.
pub(crate) fn load_got_ocr_vlm(model_path: &Path) -> Result<LoadedModel> {
    let (_config_str, full_config) = read_sanitized_vlm_config(model_path)?;

    let mut args: models::llama3::ModelArgs =
        serde_json::from_value(got_text_config(&full_config)?)
            .map_err(|e| anyhow::anyhow!("Failed to parse GOT-OCR config as qwen2: {}", e))?;
    args.set_checkpoint_label(model_path);
    let (group_size, bits) = (args.group_size(), args.bits());
    let tie_word_embeddings = full_config
        .get("tie_word_embeddings")
        .and_then(Value::as_bool)
        .unwrap_or(true);

    let weights = canonicalize_got_keys(
        load_vlm_weights_common(model_path, None)?,
        tie_word_embeddings,
    );

    // The tower is never quantized in either released layout, so it loads
    // through the plain `SamEncoder` path regardless of the decoder's bits.
    let vision_tower = SamEncoder::from_weights(&weights, VISION_PREFIX, SamConfig::default())
        .map_err(|e| anyhow::anyhow!("Failed to load GOT-OCR SAM vision tower: {}", e))?;
    let projector = UnifiedLinear::from_weights(&weights, PROJECTOR_PREFIX, group_size, bits)
        .map_err(|e| anyhow::anyhow!("Failed to load GOT-OCR projector: {}", e))?;

    let text_weights = strip_language_model_prefix(weights);
    let text_model = models::Llama3Model::from_weights(&text_weights, &args)
        .map_err(|e| anyhow::anyhow!("Failed to load GOT-OCR Qwen2 text model: {}", e))?;
    drop(text_weights);

    let generation_config = std::fs::read_to_string(model_path.join("generation_config.json"))
        .ok()
        .and_then(|content| serde_json::from_str::<Value>(&content).ok());

    let vlm = vision::got_ocr::GotOcrVlModel {
        text_model,
        vision_tower,
        projector,
        processor: GotOcrImageProcessor::default(),
        image_token_len: config_usize(&full_config, "image_token_len", DEFAULT_IMAGE_TOKEN_LEN),
        im_start_token_id: config_i32(&full_config, "im_start_token", DEFAULT_IM_START_TOKEN),
        im_end_token_id: config_i32(&full_config, "im_end_token", DEFAULT_IM_END_TOKEN),
        im_patch_token_id: config_i32(&full_config, "im_patch_token", DEFAULT_IM_PATCH_TOKEN),
        eos_token_ids: got_eos_token_ids(&full_config, generation_config.as_ref()),
    };
    Ok(LoadedModel::GotOcrVLM(vlm))
}

#[cfg(test)]
#[path = "vlm_got_ocr_tests.rs"]
mod tests;
