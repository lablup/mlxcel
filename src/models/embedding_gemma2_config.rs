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

//! Configuration parsing, per-layer geometry resolution, per-prefix
//! quantization lookup, and weight-key normalization for EmbeddingGemma 2
//! (`embedding_gemma2`, `embedding_gemma2_text`).

use mlxcel_core::weights::WeightMap;
use serde_json::Value;

/// Resolved geometry for a single EmbeddingGemma 2 transformer layer.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingGemma2LayerConfig {
    /// Layer index (`0..num_hidden_layers`).
    pub layer_idx: usize,
    /// Query attention head count (`num_attention_heads`, default 4).
    pub num_attention_heads: i32,
    /// Key/value head count (`2` on local layers, `1` on global layers).
    pub num_key_value_heads: i32,
    /// Per-head dimension (`256` on local layers, `512` on global layers).
    pub head_dim: i32,
    /// RoPE base frequency (`10_000.0` on local layers, `1_000_000.0` on global layers).
    pub rope_theta: f32,
    /// True when this layer uses sliding-window bidirectional attention (`|q - k| <= sliding_window`).
    pub is_sliding: bool,
}

/// Resolved configuration for an `EmbeddingGemma2Model` checkpoint.
#[derive(Debug, Clone)]
pub struct EmbeddingGemma2Config {
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: usize,
    pub ple_dim: i32,
    pub embedding_dim: usize,
    pub rms_norm_eps: f32,
    pub sliding_window: i32,
    pub image_token_id: i32,
    pub boi_token_id: i32,
    pub eoi_token_id: i32,
    pub audio_token_id: i32,
    pub boa_token_id: i32,
    pub eoa_token_id: i32,
    pub layers: Vec<EmbeddingGemma2LayerConfig>,
}

impl EmbeddingGemma2Config {
    /// Parse an `EmbeddingGemma2Config` from `config.json` (accepting either a
    /// nested `text_config` block or flat top-level fields).
    ///
    /// Fails when `sliding_window` cannot back the bidirectional band
    /// (issue #2798): `0`, a negative value, or anything above `i32::MAX`
    /// would silently mean full attention or a wrapped, narrowed window.
    pub fn from_json(config: &Value, weights: Option<&WeightMap>) -> Result<Self, String> {
        let tc = config.get("text_config").unwrap_or(config);

        let vocab_size = i32_field(tc, "vocab_size")
            .or_else(|| i32_field(config, "vocab_size"))
            .unwrap_or(262_144)
            .max(1);
        let hidden_size = i32_field(tc, "hidden_size")
            .or_else(|| i32_field(config, "hidden_size"))
            .unwrap_or(512)
            .max(1);
        let intermediate_size = i32_field(tc, "intermediate_size")
            .or_else(|| i32_field(config, "intermediate_size"))
            .unwrap_or(2048)
            .max(1);
        let num_hidden_layers = i32_field(tc, "num_hidden_layers")
            .or_else(|| i32_field(config, "num_hidden_layers"))
            .unwrap_or(24)
            .max(1) as usize;
        let default_n_heads = i32_field(tc, "num_attention_heads")
            .or_else(|| i32_field(config, "num_attention_heads"))
            .unwrap_or(4)
            .max(1);
        let default_n_kv_heads = i32_field(tc, "num_key_value_heads")
            .or_else(|| i32_field(config, "num_key_value_heads"))
            .unwrap_or(2)
            .max(1);
        let default_head_dim = i32_field(tc, "head_dim")
            .or_else(|| i32_field(config, "head_dim"))
            .unwrap_or(256)
            .max(1);
        let default_rope_theta = f32_field(tc, "rope_theta")
            .or_else(|| f32_field(config, "rope_theta"))
            .or_else(|| {
                tc.get("rope_parameters")
                    .or_else(|| config.get("rope_parameters"))?
                    .get("sliding_attention")?
                    .get("rope_theta")?
                    .as_f64()
                    .map(|v| v as f32)
            })
            .unwrap_or(10_000.0);
        let rms_norm_eps = f32_field(tc, "rms_norm_eps")
            .or_else(|| f32_field(config, "rms_norm_eps"))
            .unwrap_or(1e-6);
        // Parsed from the raw JSON rather than through `i32_field` so a value
        // above `i32::MAX` reaches the validator instead of wrapping (issue
        // #2798). The band's consumer reads `sliding_window + 1`, so the raw
        // value is validated before any clamp. A JSON null, or a value that is
        // not an integer at all, is not a window: it falls through to the
        // default as `i32_field` did, rather than failing a config that loaded
        // before.
        let sliding_window_value = tc
            .get("sliding_window")
            .or_else(|| config.get("sliding_window"));
        // `as_i64` keeps a negative value visible to the error below instead of
        // hiding it behind a saturating conversion.
        let sliding_window_int = sliding_window_value.and_then(Value::as_i64);
        // `as_u64` first so a positive integer above `i64::MAX` still reaches
        // the validator instead of falling back to the default.
        let sliding_window_raw = match sliding_window_value {
            Some(value) => value
                .as_u64()
                .map(|window| usize::try_from(window).unwrap_or(usize::MAX))
                .or_else(|| {
                    sliding_window_int.map(|window| usize::try_from(window).unwrap_or(usize::MAX))
                })
                .unwrap_or(512),
            None => 512,
        };

        let embedding_dim = i32_field(tc, "embedding_dim")
            .or_else(|| i32_field(config, "embedding_dim"))
            .or_else(|| {
                weights
                    .and_then(|w| w.get("language_model.embedding_projection.weight"))
                    .and_then(|arr| mlxcel_core::array_shape(arr).first().copied())
            })
            .unwrap_or(768)
            .max(1) as usize;

        let ple_dim = i32_field(tc, "hidden_size_per_layer_input")
            .or_else(|| i32_field(config, "hidden_size_per_layer_input"))
            .filter(|&d| d > 0)
            .or_else(|| {
                weights
                    .and_then(|w| w.get("language_model.ple.per_layer_projection_norm.weight"))
                    .and_then(|arr| mlxcel_core::array_shape(arr).first().copied())
            })
            .unwrap_or(hidden_size)
            .max(1);

        let layer_types: Option<Vec<&str>> = tc
            .get("layer_types")
            .or_else(|| config.get("layer_types"))
            .and_then(Value::as_array)
            .map(|arr| arr.iter().filter_map(Value::as_str).collect());

        let per_layer_map = tc
            .get("per_layer_config")
            .or_else(|| config.get("per_layer_config"))
            .and_then(Value::as_object);
        let fallback_global_head_dim = i32_field(tc, "global_head_dim")
            .or_else(|| i32_field(config, "global_head_dim"))
            .unwrap_or(512)
            .max(1);
        let fallback_global_kv_heads = i32_field(tc, "global_num_key_value_heads")
            .or_else(|| i32_field(config, "global_num_key_value_heads"))
            .unwrap_or(1)
            .max(1);
        let fallback_global_rope_theta = tc
            .get("rope_parameters")
            .or_else(|| config.get("rope_parameters"))
            .and_then(|rp| rp.get("full_attention"))
            .and_then(|fa| fa.get("rope_theta"))
            .and_then(Value::as_f64)
            .map(|v| v as f32)
            .unwrap_or(1_000_000.0);

        let mut layers = Vec::with_capacity(num_hidden_layers);
        for i in 0..num_hidden_layers {
            let override_entry = per_layer_map.and_then(|map| {
                map.get(&format!("{i:02}"))
                    .or_else(|| map.get(&format!("{i}")))
            });

            let is_sliding = match layer_types.as_ref().and_then(|lt| lt.get(i).copied()) {
                Some(kind) => kind == "sliding_attention",
                None => match override_entry
                    .and_then(|e| e.get("attention_type").or_else(|| e.get("layer_type")))
                    .and_then(Value::as_str)
                {
                    Some(kind) => kind == "sliding_attention",
                    None => override_entry.is_none() && (i + 1) % 6 != 0,
                },
            };

            let num_attention_heads = override_entry
                .and_then(|e| i32_field(e, "num_attention_heads"))
                .unwrap_or(default_n_heads)
                .max(1);

            let num_key_value_heads = override_entry
                .and_then(|e| i32_field(e, "num_key_value_heads"))
                .unwrap_or(if is_sliding {
                    default_n_kv_heads
                } else {
                    fallback_global_kv_heads
                })
                .max(1);

            let head_dim = override_entry
                .and_then(|e| i32_field(e, "head_dim"))
                .unwrap_or(if is_sliding {
                    default_head_dim
                } else {
                    fallback_global_head_dim
                })
                .max(1);

            let rope_theta = override_entry
                .and_then(|e| {
                    f32_field(e, "rope_theta").or_else(|| {
                        e.get("rope_parameters")?
                            .get("rope_theta")?
                            .as_f64()
                            .map(|v| v as f32)
                    })
                })
                .unwrap_or(if is_sliding {
                    default_rope_theta
                } else {
                    fallback_global_rope_theta
                });

            layers.push(EmbeddingGemma2LayerConfig {
                layer_idx: i,
                num_attention_heads,
                num_key_value_heads,
                head_dim,
                rope_theta,
                is_sliding,
            });
        }

        let has_sliding_layer = layers.iter().any(|layer| layer.is_sliding);
        // A negative value saturates to `usize::MAX` on the way to the
        // validator, so report the config's own value here instead of the
        // saturated one (issue #2798).
        if let Some(window) = sliding_window_int
            && window < 1
            && has_sliding_layer
        {
            return Err(format!(
                "EmbeddingGemma 2: sliding_window must be between 1 and {}, got {window}",
                i32::MAX
            ));
        }
        let sliding_window = mlxcel_core::cache::cache_window(
            "EmbeddingGemma 2",
            sliding_window_raw,
            has_sliding_layer,
        )?;

        Ok(Self {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            ple_dim,
            embedding_dim,
            rms_norm_eps,
            sliding_window,
            image_token_id: i32_field(config, "image_token_id")
                .or_else(|| i32_field(tc, "image_token_id"))
                .unwrap_or(258_880),
            boi_token_id: i32_field(config, "boi_token_id")
                .or_else(|| i32_field(tc, "boi_token_id"))
                .unwrap_or(255_999),
            eoi_token_id: i32_field(config, "eoi_token_id")
                .or_else(|| i32_field(tc, "eoi_token_id"))
                .unwrap_or(258_882),
            audio_token_id: i32_field(config, "audio_token_id")
                .or_else(|| i32_field(tc, "audio_token_id"))
                .unwrap_or(258_881),
            boa_token_id: i32_field(config, "boa_token_id")
                .or_else(|| i32_field(tc, "boa_token_id"))
                .unwrap_or(256_000),
            eoa_token_id: i32_field(config, "eoa_token_id")
                .or_else(|| i32_field(tc, "eoa_token_id"))
                .unwrap_or(258_883),
            layers,
        })
    }
}

fn i32_field(v: &Value, key: &str) -> Option<i32> {
    v.get(key)?.as_i64().map(|n| n as i32)
}

fn f32_field(v: &Value, key: &str) -> Option<f32> {
    v.get(key)?.as_f64().map(|n| n as f32)
}

/// Resolve `(group_size, bits)` for a specific weight prefix, honoring per-prefix
/// overrides inside `config.json` `quantization` / `quantization_config`.
pub fn quant_params_for_prefix(config: &Value, prefix: &str) -> (i32, i32) {
    let default_gs = config
        .get("quantization")
        .or_else(|| config.get("quantization_config"))
        .and_then(|q| q.get("group_size"))
        .and_then(Value::as_i64)
        .unwrap_or(64) as i32;
    let default_bits = config
        .get("quantization")
        .or_else(|| config.get("quantization_config"))
        .and_then(|q| q.get("bits"))
        .and_then(Value::as_i64)
        .unwrap_or(4) as i32;

    let Some(quant_obj) = config
        .get("quantization")
        .or_else(|| config.get("quantization_config"))
        .and_then(Value::as_object)
    else {
        return (default_gs, default_bits);
    };

    let mut best_match: Option<(usize, &Value)> = None;
    for (key, val) in quant_obj {
        if matches!(key.as_str(), "group_size" | "bits" | "mode") {
            continue;
        }
        let clean_key = key.strip_prefix("model.").unwrap_or(key.as_str());
        let matches_prefix = prefix == clean_key
            || prefix.starts_with(&format!("{clean_key}."))
            || prefix.strip_prefix("language_model.").is_some_and(|rest| {
                rest == clean_key || rest.starts_with(&format!("{clean_key}."))
            })
            || ((clean_key == "ple" || clean_key == "language_model.ple")
                && (prefix.starts_with("language_model.ple.") || prefix.contains(".ple_block")));
        if matches_prefix && best_match.is_none_or(|(len, _)| clean_key.len() > len) {
            best_match = Some((clean_key.len(), val));
        }
    }

    if let Some((_, override_val)) = best_match {
        if override_val.as_bool() == Some(false) {
            return (0, 0);
        }
        if let Some(obj) = override_val.as_object() {
            let gs = obj
                .get("group_size")
                .and_then(Value::as_i64)
                .map(|v| v as i32)
                .unwrap_or(default_gs);
            let bits = obj
                .get("bits")
                .and_then(Value::as_i64)
                .map(|v| v as i32)
                .unwrap_or(default_bits);
            return (gs, bits);
        }
    }

    (default_gs, default_bits)
}

/// Normalize weight keys from either HuggingFace `google/embeddinggemma-2` or
/// `mlx-community/embeddinggemma-2-*` into the canonical `language_model.*`
/// and `{vision_tower,embed_vision,audio_tower,embed_audio}.*` layout.
pub fn sanitize_embedding_gemma2_weights(weights: &mut WeightMap) {
    weights.retain(|key, _| {
        let stripped = key.strip_prefix("model.").unwrap_or(key);
        let stripped = stripped.strip_prefix("language_model.").unwrap_or(stripped);
        !stripped.starts_with("lm_head.")
            && !key.ends_with(".rotary_emb.inv_freq")
            && !key.ends_with(".position_ids")
    });

    let keys: Vec<String> = weights.keys().cloned().collect();
    for key in keys {
        let canonical = canonicalize_weight_key(&key);
        if canonical != key
            && let Some(tensor) = weights.remove(&key)
        {
            weights.insert(canonical, tensor);
        }
    }
}

fn canonicalize_weight_key(key: &str) -> String {
    let stripped = key.strip_prefix("model.").unwrap_or(key);

    for tower in [
        "vision_tower.",
        "embed_vision.",
        "audio_tower.",
        "embed_audio.",
    ] {
        if stripped.starts_with(tower) {
            return stripped.to_string();
        }
    }

    let lm_rest = stripped
        .strip_prefix("language_model.model.")
        .or_else(|| stripped.strip_prefix("language_model."))
        .unwrap_or(stripped);

    if let Some(rest) = lm_rest
        .strip_prefix("per_layer_model_projection.")
        .map(|s| format!("language_model.ple.per_layer_model_projection.{s}"))
        .or_else(|| {
            lm_rest
                .strip_prefix("per_layer_projection_norm.")
                .map(|s| format!("language_model.ple.per_layer_projection_norm.{s}"))
        })
    {
        return rest;
    }

    if let Some(after_layers) = lm_rest.strip_prefix("layers.")
        && let Some((idx_str, layer_rest)) = after_layers.split_once('.')
    {
        for ple_leaf in [
            "per_layer_input_gate.",
            "per_layer_projection.",
            "post_per_layer_input_norm.",
        ] {
            if layer_rest.starts_with(ple_leaf) {
                return format!("language_model.layers.{idx_str}.ple_block.{layer_rest}");
            }
        }
        return format!("language_model.layers.{idx_str}.{layer_rest}");
    }

    for root in ["embed_tokens.", "norm.", "ple.", "embedding_projection."] {
        if lm_rest.starts_with(root) {
            return format!("language_model.{lm_rest}");
        }
    }

    stripped.to_string()
}
