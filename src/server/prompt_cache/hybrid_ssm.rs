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
//! | `kimi_k3`              | Kimi K3 (KDA + MLA hybrid text backbone)  |
//! | `qwen3_next`           | Full Attention + GatedDeltaNet + MoE      |
//! | `falcon_mamba`         | Falcon Mamba (HF `model_type` alias)      |
//! | `longcat_flash`        | LongCat-Flash linear hybrid               |
//! | `longcat_flash_ngram`  | LongCat-Flash with n-gram speculation     |
//! | `longcat_flash_sparse` | LongCat-Flash-Lite-Sparse (LSA, n-gram)   |
//! | `rwkv7`                | RWKV-7 recurrent                          |
//! | `recurrent_gemma`      | Griffin (RecurrentGemma) local+recurrent  |
//! | `z1t`                  | Z1T AFT-conv pool sums + conv window      |
//! | `baichuan_m1`          | Dense attention with a per-token K/V       |
//! |                        | convolution carrying recurrent state      |
//! | `falcon_h1`            | Parallel Mamba2 + Attention (Falcon-H1)   |
//! | `plamo2`, `plamo2vl`   | Mamba + Attention (PLaMo 2, and its VLM)  |
//! | `qwen3_5`, `_text`     | Gated DeltaNet + Attention (Qwen3.5)      |
//! | `qwen3_5_moe`, `_text` | Qwen3.5 MoE, same recurrent layers        |
//! | `granitemoehybrid`     | Mamba2 + Attention + MoE (Granite 4.0-H)  |
//!
//! The last five rows were added by #3341 (epic #564). On the MLX path the
//! only effect of membership is the APC opt-out at startup: these families
//! already keep their state outside the `KVCache` slice (model-owned for
//! Falcon-H1, PLaMo 2 and Granite 4.0-H, scheduler-owned `LayerState` for
//! Qwen3.5), so they never reached the block-hash store, and their whole
//! snapshots and turn-boundary state checkpoints (#3047) never consult the
//! APC flag.
//!
//! ## Why this is safe under the unified paged cache (#124)
//!
//! Two independent mechanisms keep these families out of cross-request block
//! sharing, so the unified paged store inherits the carve-out for free:
//!
//! 1. **APC opt-out (this module).** [`detect_hybrid_ssm`] runs at startup and
//!    force-disables Automatic Prefix Caching, so no block-hash chain is ever
//!    computed for a recurrent prompt.
//! 2. **State outside the `KVCache` slice (the load-bearing guarantee).** Every
//!    family here except Baichuan-M1 reports a `sequence_state_layout()` whose
//!    backend is `ModelOwned` or `SchedulerOwned`
//!    (`SequenceStateBackend::keeps_state_outside_kv_slice`). That holds whether
//!    the family batches or not: Mamba2, Nemotron-H, Falcon-H1, PLaMo 2,
//!    Granite 4.0-H and Qwen3.5 return `supports_batching() == true` and still
//!    keep that layout. The scheduler's dense adopt and donate paths both
//!    decline such a sequence, so a recurrent sequence never reaches the
//!    block-hash store; it is restored only from whole-state snapshots and
//!    turn-boundary checkpoints, which never truncate a recurrent state to a
//!    shorter prefix.
//!
//!    Baichuan-M1 is the one dense exception (#2567): its KV caches live in the
//!    scheduler slice and it batches per row, while the K/V convolution state
//!    is model-owned and keyed by `SequenceId`. Neither carve-out above applies
//!    to it, so the whole-prefix store's adopt and donate paths consult
//!    `LanguageModel::kv_cache_captures_sequence_state()` and decline the family
//!    there: a restored KV prefix would arrive without the conv state of its
//!    last token.
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

use super::policy::PromptCacheConfig;

/// Authoritative list of hybrid SSM / linear-attention `model_type` strings
/// that must opt out of APC.
pub const HYBRID_SSM_MODEL_TYPES: &[&str] = &[
    "jamba",
    "mamba",
    "mamba2",
    "nemotron_h",
    "gated_delta",
    "kimi_linear",
    "kimi_k3",
    "qwen3_next",
    // Aliases sometimes seen in HF configs:
    "falcon_mamba",
    "longcat_flash",
    "longcat_flash_ngram",
    // Model-owned caches and n-gram tail, like the other LongCat types (#2362).
    "longcat_flash_sparse",
    "rwkv7",
    "recurrent_gemma",
    // AFT-conv running pool sums and conv window (Z1T, #2364).
    "z1t",
    // Dense attention, but every layer convolves K/V with the previous token's
    // pre-convolution K/V, which is recurrent state the KV cache does not hold
    // (#2567).
    "baichuan_m1",
    // Hybrids whose MLX state already lives outside the KV slice; listed so
    // APC is off for every recurrent family on both backends (#3341).
    "falcon_h1",
    "plamo2",
    "plamo2vl",
    "qwen3_5",
    "qwen3_5_text",
    "qwen3_5_moe",
    "qwen3_5_moe_text",
    "granitemoehybrid",
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
    "kimik3",
    "qwen3next",
    "falconmamba",
    "longcatflash",
    // LongCat-Flash-Lite-Sparse's original config has no `model_type` (#2362).
    "longcatcausallm",
    "rwkv7",
    "recurrentgemma",
    "z1t",
    "baichuanm1",
    "falconh1",
    "plamo2",
    // `Qwen3_5ForConditionalGeneration`, `Qwen3_5MoeForCausalLM`, ...
    "qwen35",
    "granitemoehybrid",
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
    // HF writes Python's bare `Infinity` into Mamba2 and Nemotron-H configs
    // (`time_step_limit`); a strict parse would fail and leave APC on for
    // exactly the families this check exists for.
    let config: Value = serde_json::from_str(&crate::models::sanitize_config_json(&config_str))
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    Ok(detect_hybrid_ssm(&config))
}

/// Startup's APC opt-out: when the operator requested APC and `model_path`
/// is a [`HYBRID_SSM_MODEL_TYPES`] family, turn APC off and log one warning.
/// Returns the detected family. The whole-prefix store, its snapshots and the
/// turn-boundary state checkpoints (#3047) stay as configured.
///
/// Used by: `startup::resolve_prompt_cache_store`.
pub fn disable_apc_for_hybrid_ssm(
    prompt_cache: &mut PromptCacheConfig,
    model_path: &Path,
) -> Option<String> {
    if !prompt_cache.apc.enabled {
        return None;
    }
    let family = detect_hybrid_ssm_from_path(model_path).ok().flatten()?;
    tracing::warn!(
        model_type = %family,
        "Detected hybrid SSM / linear-attention model family ({family}); \
         auto-disabling APC because recurrent state cannot decompose \
         into hashable blocks. Whole-prefix prompt cache is unaffected."
    );
    prompt_cache.apc.enabled = false;
    Some(family)
}

/// Decide whether a model must stay out of prefix-hash sharing entirely on a
/// backend that reports its own recurrent state (#3341).
///
/// `backend_threads_recurrent_state` is the backend's answer for this
/// checkpoint (for OpenXLA, `Config::has_recurrent_state()`). The shared
/// [`detect_hybrid_ssm`] list is consulted as well, so a recurrent family the
/// backend cannot load yet is still refused. Returns the family name to log,
/// or `None` when the model may share prefixes.
#[must_use]
pub fn recurrent_prefix_sharing_family(
    config: &Value,
    backend_threads_recurrent_state: bool,
) -> Option<String> {
    if let Some(family) = detect_hybrid_ssm(config) {
        return Some(family);
    }
    backend_threads_recurrent_state.then(|| {
        config
            .get("model_type")
            .and_then(Value::as_str)
            .unwrap_or("recurrent")
            .to_string()
    })
}

#[cfg(test)]
#[path = "hybrid_ssm_tests.rs"]
mod tests;
