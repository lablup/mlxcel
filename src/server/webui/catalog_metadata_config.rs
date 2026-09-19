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

use serde_json::Value;

use super::catalog_types::{
    CatalogMetadataUnknownReasons, MAX_DECLARED_ARCHITECTURE_BYTES, MAX_DECLARED_ARCHITECTURES,
    MAX_DTYPE_BYTES, MAX_MODEL_TYPE_BYTES,
};

pub(super) fn declared_model_type_from_config(
    config: Option<&Value>,
    config_error: Option<&str>,
    reasons: &mut CatalogMetadataUnknownReasons,
) -> Option<String> {
    let Some(config) = config else {
        reasons.model_type = Some(
            config_error
                .unwrap_or("config.json is unavailable in bounded metadata")
                .to_string(),
        );
        return None;
    };
    let Some(value) = config.get("model_type") else {
        reasons.model_type = Some("config.json has no string model_type".to_string());
        return None;
    };
    let Some(raw) = value.as_str() else {
        reasons.model_type = Some("model_type must be a string".to_string());
        return None;
    };
    if raw.len() > MAX_MODEL_TYPE_BYTES {
        reasons.model_type = Some(format!("model_type exceeds {MAX_MODEL_TYPE_BYTES} bytes"));
        return None;
    }
    Some(raw.to_string())
}

pub(super) fn declared_architectures_from_config(
    config: Option<&Value>,
    config_error: Option<&str>,
    reasons: &mut CatalogMetadataUnknownReasons,
) -> Option<Vec<String>> {
    let Some(config) = config else {
        reasons.declared_architectures = Some(
            config_error
                .unwrap_or("config.json is unavailable in bounded metadata")
                .to_string(),
        );
        return None;
    };
    let Some(value) = config.get("architectures") else {
        reasons.declared_architectures = Some("config.json has no architectures array".to_string());
        return None;
    };
    let Some(items) = value.as_array() else {
        reasons.declared_architectures =
            Some("architectures must be an array of strings".to_string());
        return None;
    };
    if items.len() > MAX_DECLARED_ARCHITECTURES {
        reasons.declared_architectures = Some(format!(
            "architectures has more than {MAX_DECLARED_ARCHITECTURES} entries"
        ));
        return None;
    }
    let mut architectures = Vec::with_capacity(items.len());
    for item in items {
        let Some(raw) = item.as_str() else {
            reasons.declared_architectures =
                Some("architectures must contain only strings".to_string());
            return None;
        };
        if raw.len() > MAX_DECLARED_ARCHITECTURE_BYTES {
            reasons.declared_architectures = Some(format!(
                "architectures entry exceeds {MAX_DECLARED_ARCHITECTURE_BYTES} bytes"
            ));
            return None;
        }
        architectures.push(raw.to_string());
    }
    Some(architectures)
}

/// The declared weight dtype, normalized to the short name the library prints.
///
/// Reads `torch_dtype`, then `dtype` (the key newer `transformers` exports write),
/// at the top level and then under `text_config`, the layout multimodal exports
/// use. A `torch.` prefix is dropped and the common floating-point names shorten
/// (`bfloat16` to `bf16`); any other value passes through lowercased only when it
/// is a bounded ASCII identifier, so a config cannot put arbitrary text into the
/// catalog. `auto` is a loader directive, not a dtype, and reads as unknown.
pub(super) fn dtype_from_config(config: &Value) -> Option<String> {
    [Some(config), config.get("text_config")]
        .into_iter()
        .flatten()
        .flat_map(|scope| ["torch_dtype", "dtype"].map(|key| scope.get(key)))
        .flatten()
        .find_map(|value| normalize_dtype(value.as_str()?))
}

fn normalize_dtype(raw: &str) -> Option<String> {
    let name = raw.trim();
    let name = name.strip_prefix("torch.").unwrap_or(name);
    if name.is_empty()
        || name.len() > MAX_DTYPE_BYTES
        || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
    {
        return None;
    }
    let lower = name.to_ascii_lowercase();
    let short = match lower.as_str() {
        "auto" => return None,
        "bfloat16" | "bf16" => "bf16",
        "float16" | "half" | "fp16" | "f16" => "fp16",
        "float32" | "float" | "fp32" | "f32" => "fp32",
        "float64" | "double" | "fp64" | "f64" => "fp64",
        _ => return Some(lower),
    };
    Some(short.to_string())
}
