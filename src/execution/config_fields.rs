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

//! Single source of truth for the `config.json` field-name aliases the
//! memory / KV estimators read.
//!
//! Checkpoints spell the same architectural quantity several ways: the modern
//! HuggingFace `num_hidden_layers` / `hidden_size`, the MPT / OLMo `d_model`,
//! and the OpenAI-era `n_layer` / `n_embd` that GPT-2 and GPT-BigCode still
//! use. Every estimator has to accept all of them or it silently reports zero
//! for a family whose config it cannot parse.
//!
//! Before this module the alias lists were copied into five independent
//! `.or_else()` chains across `kv_arch`, `memory_estimate` (twice),
//! `kv_cache_advisor`, and `quant_advisor`. A spelling added to one was not
//! added to the others, which is exactly how the GPT-2 / GPT-BigCode gap
//! (#927) reproduced in every one of them. Add a new alias here, once.
//!
//! Ordering inside each list is significant: the first present key wins, so
//! the modern spellings lead and the legacy ones trail. Appending an alias
//! therefore cannot change what an already-resolving config resolves to.

use serde_json::Value;

/// Transformer layer count.
///
/// `n_layer` (singular) is the OpenAI-era spelling carried by GPT-2 and
/// GPT-BigCode configs; without it `classify` short-circuits and the whole KV
/// estimate is reported as unavailable.
pub(crate) const LAYER_COUNT_KEYS: &[&str] =
    &["num_hidden_layers", "n_layers", "num_layers", "n_layer"];

/// Model hidden size (embedding width).
///
/// `n_embd` is the matching OpenAI-era spelling. It is the fallback source for
/// `head_dim` (`hidden / num_heads`) and the basis of the activation reserve.
pub(crate) const HIDDEN_SIZE_KEYS: &[&str] =
    &["hidden_size", "d_model", "dim", "model_dim", "n_embd"];

/// Query (attention) head count.
pub(crate) const NUM_HEADS_KEYS: &[&str] =
    &["num_attention_heads", "num_heads", "n_heads", "n_head"];

/// Numeric key/value head count, for grouped-query and multi-query attention.
///
/// This list is numeric only. GPT-BigCode carries none of these keys and
/// instead signals multi-query attention with the boolean [`MULTI_QUERY_KEY`];
/// use [`resolve_num_kv_heads`] rather than reading this list directly.
pub(crate) const NUM_KV_HEADS_KEYS: &[&str] = &[
    "num_key_value_heads",
    "num_kv_heads",
    "n_kv_heads",
    "n_head_kv",
    "multi_query_group_num",
];

/// Explicit per-head dimension, when the config states it outright.
pub(crate) const HEAD_DIM_KEYS: &[&str] = &["head_dim", "head_size"];

/// FFN intermediate width.
pub(crate) const INTERMEDIATE_SIZE_KEYS: &[&str] =
    &["intermediate_size", "ffn_dim", "ffn_hidden_size"];

/// Boolean multi-query-attention flag (GPT-BigCode, Falcon).
///
/// `true` means exactly one key/value head shared across every query head.
pub(crate) const MULTI_QUERY_KEY: &str = "multi_query";

/// The decoder sub-object of a VLM config, or the config itself for text
/// models. VLMs nest the language model under `text_config`.
pub(crate) fn text_config(config: &Value) -> &Value {
    config.get("text_config").unwrap_or(config)
}

/// First present `u64` among `keys` in `obj`.
pub(crate) fn get_u64(obj: &Value, keys: &[&str]) -> Option<u64> {
    keys.iter()
        .find_map(|k| obj.get(*k).and_then(|v| v.as_u64()))
}

/// First present `f64` among `keys` in `obj`.
pub(crate) fn get_f64(obj: &Value, keys: &[&str]) -> Option<f64> {
    keys.iter()
        .find_map(|k| obj.get(*k).and_then(|v| v.as_f64()))
}

/// Whether the config declares boolean multi-query attention.
///
/// Only a literal `true` counts. An absent flag, `false`, or a non-boolean
/// value all mean "not multi-query", so a malformed config keeps the ordinary
/// head-count behaviour instead of silently collapsing to one KV head.
pub(crate) fn is_multi_query(text: &Value) -> bool {
    text.get(MULTI_QUERY_KEY).and_then(Value::as_bool) == Some(true)
}

/// Resolve the number of key/value heads, honouring both the numeric fields
/// and the boolean multi-query flag.
///
/// Precedence, highest first:
///
/// 1. An explicit numeric field from [`NUM_KV_HEADS_KEYS`]. A config that
///    states a count means it, even if it also carries `multi_query`.
/// 2. `multi_query: true`, which means exactly one shared KV head.
/// 3. `num_heads`, the plain MHA fallback.
///
/// Step 2 exists because GPT-BigCode expresses MQA only as a boolean.
/// `GptBigCodeArgs::n_kv_heads` (`src/models/gpt_bigcode.rs`) returns
/// `if multi_query { 1 } else { n_head }`, and the runtime caches accordingly.
/// Falling straight through to step 3 would claim `n_head` KV heads where one
/// is cached and over-reserve the KV estimate by that whole factor: 16x for
/// santacoder, 48x for a StarCoder-sized checkpoint.
///
/// Safe for the degenerate `multi_query: true` with `n_head: 0` shape that the
/// model-side `validate` rejects at load: the result is 1 regardless of
/// `num_heads`, and no caller divides by the returned value.
pub(crate) fn resolve_num_kv_heads(text: &Value, num_heads: u64) -> u64 {
    if let Some(kv) = get_u64(text, NUM_KV_HEADS_KEYS) {
        return kv;
    }
    if is_multi_query(text) {
        return 1;
    }
    num_heads
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn alias_lists_prefer_modern_spellings_over_legacy() {
        // A config carrying both spellings must resolve to the modern one, so
        // appending a legacy alias cannot move an existing estimate.
        let cfg = json!({
            "num_hidden_layers": 32,
            "n_layer": 12,
            "hidden_size": 4096,
            "n_embd": 768,
        });
        assert_eq!(get_u64(&cfg, LAYER_COUNT_KEYS), Some(32));
        assert_eq!(get_u64(&cfg, HIDDEN_SIZE_KEYS), Some(4096));
    }

    #[test]
    fn openai_era_spellings_resolve() {
        let cfg = json!({ "n_layer": 12, "n_embd": 768, "n_head": 12 });
        assert_eq!(get_u64(&cfg, LAYER_COUNT_KEYS), Some(12));
        assert_eq!(get_u64(&cfg, HIDDEN_SIZE_KEYS), Some(768));
        assert_eq!(get_u64(&cfg, NUM_HEADS_KEYS), Some(12));
    }

    #[test]
    fn multi_query_true_is_one_kv_head() {
        let cfg = json!({ "n_head": 16, "multi_query": true });
        assert!(is_multi_query(&cfg));
        assert_eq!(resolve_num_kv_heads(&cfg, 16), 1);
    }

    #[test]
    fn multi_query_false_or_absent_falls_back_to_mha() {
        let explicit_false = json!({ "n_head": 16, "multi_query": false });
        assert!(!is_multi_query(&explicit_false));
        assert_eq!(resolve_num_kv_heads(&explicit_false, 16), 16);

        let absent = json!({ "n_head": 16 });
        assert!(!is_multi_query(&absent));
        assert_eq!(resolve_num_kv_heads(&absent, 16), 16);
    }

    #[test]
    fn numeric_kv_head_field_outranks_multi_query() {
        let cfg = json!({
            "num_attention_heads": 71,
            "num_kv_heads": 8,
            "multi_query": true,
        });
        assert_eq!(resolve_num_kv_heads(&cfg, 71), 8);
    }

    #[test]
    fn non_boolean_multi_query_is_not_multi_query() {
        let cfg = json!({ "n_head": 16, "multi_query": "true" });
        assert!(!is_multi_query(&cfg));
        assert_eq!(resolve_num_kv_heads(&cfg, 16), 16);
    }

    #[test]
    fn multi_query_with_zero_heads_stays_at_one() {
        let cfg = json!({ "n_head": 0, "multi_query": true });
        assert_eq!(resolve_num_kv_heads(&cfg, 0), 1);
    }

    #[test]
    fn text_config_unwraps_vlm_nesting() {
        let vlm = json!({ "text_config": { "n_layer": 4 } });
        assert_eq!(get_u64(text_config(&vlm), LAYER_COUNT_KEYS), Some(4));

        let text_only = json!({ "n_layer": 4 });
        assert_eq!(get_u64(text_config(&text_only), LAYER_COUNT_KEYS), Some(4));
    }
}
