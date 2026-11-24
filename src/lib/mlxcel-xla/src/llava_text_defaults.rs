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

//! Hugging Face text-config defaults for LLaVA wrappers (issue #3366).
//!
//! `llava-hf` and `mlx-community` conversions of LLaVA-1.5-7B declare
//! `text_config.model_type = "llama"` and leave the architecture fields at the
//! `LlamaConfig` defaults implicit. The eager loader recovers those from the
//! checkpoint tensors; the OpenXLA path reads only `config.json`, so it applies
//! the same Hugging Face class defaults here. One table serves the text-graph
//! emitter, the IREE vision config, and the host preprocessor so they cannot
//! disagree. Ungated: the host preprocessor needs it without the `iree` feature.

use serde_json::{Map, Value, json};

/// The `transformers.LlamaConfig` constructor defaults for the fields the
/// OpenXLA text graph and vision preprocessor read. `num_key_value_heads` is
/// resolved separately because it defaults to `num_attention_heads`.
const LLAMA_DEFAULTS: &[(&str, fn() -> Value)] = &[
    ("vocab_size", || json!(32_000)),
    ("hidden_size", || json!(4096)),
    ("intermediate_size", || json!(11_008)),
    ("num_hidden_layers", || json!(32)),
    ("num_attention_heads", || json!(32)),
    ("max_position_embeddings", || json!(2048)),
    ("rms_norm_eps", || json!(1.0e-6)),
    ("rope_theta", || json!(10_000.0)),
    // `LlamaConfig` ties nothing by default. The XLA emitter's own fallback for
    // an absent key is tied, which silently replaces `lm_head` with the embedding
    // table (garbage text on llava-1.5-7b-4bit).
    ("tie_word_embeddings", || json!(false)),
];

/// Whether `model_type` names a Hugging Face LLaVA wrapper (`LlavaConfig` or
/// `LlavaNextConfig`) whose `text_config` follows the llama defaults when it
/// leaves fields implicit. The host preprocessor, the text-graph emitter, and
/// the IREE vision config all gate on this one predicate so a wrapper resolves
/// identically on every path.
#[must_use]
pub fn is_llava_text_defaults_wrapper(model_type: Option<&str>) -> bool {
    matches!(model_type, Some("llava" | "llava_next"))
}

/// Fill the missing architecture fields of a LLaVA `text_config` from the
/// defaults of its declared `model_type` (an absent `model_type` is `llama`,
/// matching the host preprocessor's own fallback). Explicit values are never
/// overwritten. A model type without a known default table is left untouched,
/// so a missing field still fails closed at its reader.
pub fn apply_llava_text_config_defaults(text_config: &mut Map<String, Value>) {
    let model_type = text_config
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("llama");
    if model_type != "llama" {
        return;
    }
    for (name, default) in LLAMA_DEFAULTS {
        text_config
            .entry((*name).to_string())
            .or_insert_with(default);
    }
    if !text_config.contains_key("num_key_value_heads")
        && let Some(heads) = text_config.get("num_attention_heads").cloned()
    {
        text_config.insert("num_key_value_heads".to_string(), heads);
    }
}

/// Error text for a required text field that is absent and has no default.
#[must_use]
pub fn llava_text_field_missing(field: &str, text_config: &Map<String, Value>) -> String {
    let model_type = text_config
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or("llama");
    format!(
        "LLaVA text_config.{field} is required: text_config.model_type={model_type:?} has no \
         known Hugging Face default for it, so it must be spelled out in config.json"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(value: Value) -> Map<String, Value> {
        value.as_object().cloned().unwrap()
    }

    #[test]
    fn llama_declaration_resolves_hf_defaults() {
        let mut text =
            map(json!({"model_type": "llama", "rms_norm_eps": 1e-5, "vocab_size": 32064}));
        apply_llava_text_config_defaults(&mut text);
        assert_eq!(text["hidden_size"], json!(4096));
        assert_eq!(text["intermediate_size"], json!(11_008));
        assert_eq!(text["num_hidden_layers"], json!(32));
        assert_eq!(text["num_attention_heads"], json!(32));
        assert_eq!(text["num_key_value_heads"], json!(32));
        assert_eq!(text["tie_word_embeddings"], json!(false));
        // Declared values win over the table.
        assert_eq!(text["rms_norm_eps"], json!(1e-5));
        assert_eq!(text["vocab_size"], json!(32064));
    }

    #[test]
    fn kv_heads_default_follows_declared_attention_heads() {
        let mut text = map(json!({"model_type": "llama", "num_attention_heads": 40}));
        apply_llava_text_config_defaults(&mut text);
        assert_eq!(text["num_key_value_heads"], json!(40));
    }

    #[test]
    fn fully_spelled_config_is_unchanged() {
        let original = map(json!({
            "model_type": "llama", "hidden_size": 5120, "intermediate_size": 13824,
            "num_hidden_layers": 40, "num_attention_heads": 40, "num_key_value_heads": 8,
            "max_position_embeddings": 4096, "rms_norm_eps": 1e-5, "rope_theta": 5e5,
            "vocab_size": 32064, "tie_word_embeddings": true
        }));
        let mut text = original.clone();
        apply_llava_text_config_defaults(&mut text);
        assert_eq!(text, original);
    }

    #[test]
    fn unknown_model_type_stays_fail_closed() {
        let mut text = map(json!({"model_type": "mystery"}));
        apply_llava_text_config_defaults(&mut text);
        assert!(!text.contains_key("hidden_size"));
        let message = llava_text_field_missing("hidden_size", &text);
        assert!(
            message.contains("hidden_size") && message.contains("mystery"),
            "{message}"
        );
    }

    #[test]
    fn wrapper_predicate_covers_llava_and_llava_next_only() {
        assert!(is_llava_text_defaults_wrapper(Some("llava")));
        assert!(is_llava_text_defaults_wrapper(Some("llava_next")));
        assert!(!is_llava_text_defaults_wrapper(Some("qwen2_vl")));
        assert!(!is_llava_text_defaults_wrapper(None));
    }

    #[test]
    fn qwen2_declaration_is_not_defaulted() {
        let mut text = map(json!({"model_type": "qwen2"}));
        apply_llava_text_config_defaults(&mut text);
        assert!(text.get("hidden_size").is_none());
    }
}
