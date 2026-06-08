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

//! Hybrid SSM / linear-attention opt-out for Automatic Prefix Caching.
//!
//! APC's block-hash chain assumes the per-token state is fully captured by the
//! KV cache. Hybrid models that interleave Mamba / SSM / linear-attention
//! layers carry a recurrent hidden state that cannot be reconstructed from a
//! token-prefix hash alone — replaying the prefix from a saved KV slab would
//! restore the attention layers but leave the recurrent layers in a stale or
//! zero state, producing garbage outputs.
//!
//! Rather than risk a silent correctness bug, the server detects these model
//! families at startup (by inspecting `model_type` / `architectures` in
//! `config.json`) and force-disables APC for them. The whole-prefix prompt
//! cache (`PromptCacheStore`) remains available because it adopts the entire
//! cache as a unit, so its full-state semantics are unaffected.
//!
//! ## Coverage
//!
//! [`HYBRID_SSM_MODEL_TYPES`] is authoritative. New hybrid families added in
//! future MLX upstream syncs must be added there at the same time the model is
//! wired in. The `all_hybrid_ssm_model_types_round_trip` test iterates the
//! constant itself so the table below can never silently drift from it.
//!
//! | model_type             | Architecture                              |
//! |------------------------|-------------------------------------------|
//! | `jamba`                | Mamba + Transformer + MoE                 |
//! | `mamba`                | Mamba 1                                   |
//! | `mamba2`               | Mamba 2                                   |
//! | `nemotron_h`           | Mamba2 + Attention + MLP/MoE              |
//! | `gated_delta`          | Gated DeltaNet (Qwen3-Next family)        |
//! | `kimi_linear`          | Kimi linear-attention hybrid              |
//! | `qwen3_next`           | Full Attention + GatedDeltaNet + MoE      |
//! | `falcon_mamba`         | Falcon Mamba (HF `model_type` alias)      |
//! | `longcat_flash`        | LongCat-Flash linear hybrid               |
//! | `longcat_flash_ngram`  | LongCat-Flash with n-gram speculation     |
//! | `rwkv7`                | RWKV-7 recurrent                          |
//! | `recurrent_gemma`      | Griffin (RecurrentGemma) local+recurrent  |
//!
//! ## Why this is safe under the unified paged cache (#124)
//!
//! Two independent mechanisms keep these families out of cross-request block
//! sharing, so the unified paged store inherits the carve-out for free:
//!
//! 1. **APC opt-out (this module).** [`detect_hybrid_ssm`] runs at startup and
//!    force-disables Automatic Prefix Caching, so no block-hash chain is ever
//!    computed for a recurrent prompt.
//! 2. **`ModelOwned` backend (the load-bearing guarantee).** Every family here
//!    overrides `LanguageModel::supports_batching()` to `false`, which routes
//!    the default `sequence_state_layout()` to
//!    `SequenceStateBackend::ModelOwned`. A `ModelOwned` sequence is rejected by
//!    both `CachePool::detach` and `CachePool::detach_paged` (they return
//!    `None`), so the scheduler's donate-back path can never hand a recurrent
//!    sequence to the prompt-cache store, so its store stays empty and every
//!    adopt attempt misses. The same `supports_batching() == false` also forces
//!    `effective_decode_storage_backend` to `Dense`, so `--decode-storage-backend
//!    paged` is silently ignored for these models rather than corrupting their
//!    recurrent state. The net effect is bit-identical output to the pre-epic
//!    dense path regardless of the requested decode backend.
//!
//! ## Detection precedence
//!
//! 1. Top-level `config.model_type` string.
//! 2. Nested `config.text_config.model_type` (used by VLMs).
//! 3. The first entry in `config.architectures[]` (lower-case-matched against
//!    a known set of hybrid arch names).
//!
//! If none of these match the known list, APC stays at whatever the operator
//! requested.

use std::path::Path;

use serde_json::Value;

/// Authoritative list of hybrid SSM / linear-attention `model_type` strings
/// that must opt out of APC.
pub const HYBRID_SSM_MODEL_TYPES: &[&str] = &[
    "jamba",
    "mamba",
    "mamba2",
    "nemotron_h",
    "gated_delta",
    "kimi_linear",
    "qwen3_next",
    // Aliases sometimes seen in HF configs:
    "falcon_mamba",
    "longcat_flash",
    "longcat_flash_ngram",
    "rwkv7",
    "recurrent_gemma",
];

/// Architecture-name fragments (matched case-insensitively against entries in
/// the HF `architectures` array) that imply a hybrid SSM model. Used as a
/// fallback when `model_type` is missing or non-canonical.
const HYBRID_SSM_ARCH_FRAGMENTS: &[&str] = &[
    "jamba",
    "mamba",
    "nemotronh",
    "gateddelta",
    "kimilinear",
    "qwen3next",
    "falconmamba",
    "longcatflash",
    "rwkv7",
    "recurrentgemma",
];

/// Return `true` when `model_type` (a value from `config.json`) corresponds to
/// a hybrid SSM / linear-attention family that cannot use APC.
///
/// Matching is case-insensitive and trims whitespace. Empty input returns
/// `false`.
#[must_use]
pub fn is_hybrid_ssm_model_type(model_type: &str) -> bool {
    let s = model_type.trim().to_ascii_lowercase();
    if s.is_empty() {
        return false;
    }
    HYBRID_SSM_MODEL_TYPES
        .iter()
        .any(|&known| known.eq_ignore_ascii_case(&s))
}

/// Inspect the loaded `config.json` JSON and decide whether APC must be
/// disabled. Returns `Some(model_type_str)` identifying the offending family
/// when a hybrid model is detected, `None` otherwise.
///
/// Detection precedence:
/// 1. `config.model_type` (top-level).
/// 2. `config.text_config.model_type` (VLM nested config).
/// 3. `config.architectures[0]` (case-insensitive substring match).
#[must_use]
pub fn detect_hybrid_ssm(config: &Value) -> Option<String> {
    // 1) Top-level model_type
    if let Some(mt) = config.get("model_type").and_then(Value::as_str)
        && is_hybrid_ssm_model_type(mt)
    {
        return Some(mt.to_string());
    }
    // 2) Nested text_config.model_type (VLM)
    if let Some(mt) = config
        .get("text_config")
        .and_then(|tc| tc.get("model_type"))
        .and_then(Value::as_str)
        && is_hybrid_ssm_model_type(mt)
    {
        return Some(mt.to_string());
    }
    // 3) Architecture fragments — match against the lower-cased arch name with
    //    underscores stripped so "NemotronHForCausalLM" hits the "nemotronh"
    //    fragment.
    if let Some(arches) = config.get("architectures").and_then(Value::as_array) {
        for arch in arches {
            if let Some(name) = arch.as_str() {
                let normalized: String = name
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
                    .to_ascii_lowercase();
                for &frag in HYBRID_SSM_ARCH_FRAGMENTS {
                    if normalized.contains(frag) {
                        return Some(name.to_string());
                    }
                }
            }
        }
    }
    None
}

/// Read `<model_path>/config.json` and run [`detect_hybrid_ssm`].
///
/// Returns `Ok(None)` when the config is readable but the model is not
/// hybrid. Returns `Ok(Some(name))` when a hybrid family is detected.
/// Returns `Err` when the file is missing or cannot be parsed — the caller
/// should treat this as "do not silently disable APC" and let the operator's
/// flag stand (the model loader will surface a clearer error a few lines
/// later anyway).
pub fn detect_hybrid_ssm_from_path(model_path: &Path) -> std::io::Result<Option<String>> {
    let config_path = model_path.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)?;
    let config: Value = serde_json::from_str(&config_str)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    Ok(detect_hybrid_ssm(&config))
}

#[cfg(test)]
#[path = "hybrid_ssm_tests.rs"]
mod tests;
