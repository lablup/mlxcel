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
//
// Portions of this file are derived from turboquant_plus
// (https://github.com/TheTom/turboquant_plus), Copyright 2026 Tom Turney,
// licensed under the Apache License, Version 2.0. See the top-level NOTICE
// file for the attribution carried forward under Apache-2.0 Section 4(d).

//! Per-model allowlist for **symmetric** `KVCacheMode::Turbo4`
//!
//! # Why an allowlist?
//!
//! Symmetric Turbo4 (4-bit K + 4-bit V) is **catastrophic on dense Q4_K_M
//! weights**. The TurboQuant+ team measured PPL 218 on `Qwen2.5-7B-Q4_K_M`
//! with `turbo4-K + turbo4-V` versus a baseline PPL of 6.6 — a 33×
//! regression. The math is unforgiving: the softmax in attention
//! exponentially amplifies K-side quantization error, and Q4_K_M weights
//! already eat most of the precision budget.
//!
//! The asymmetric path (`KVCacheMode::Turbo4Asym`, FP16-K + Turbo4-V)
//! sidesteps this entirely because softmax never sees a quantized K. That
//! is the *recommended default* for any Q4_K_M model and is what
//! non-allowlisted models fall back to when the user requests
//! `--kv-cache-mode turbo4`.
//!
//! # When is symmetric Turbo4 safe?
//!
//! TurboQuant+ identified four model classes where symmetric Turbo4 stays
//! within the +2.0% PPL budget:
//!
//! 1. **Large dense models on Q4_K_M** (≥70B). The extra parameters absorb
//!    the K-side quantization noise. Llama-3.1-70B-Q4_K_M measured at
//!    +6.3% PPL (still over the +2% gate, but *not* catastrophic — the gate
//!    is per-model, see `tests/turbo_kv_e2e.rs`). Mistral-Small-24B and
//!    Command-R+ 104B were measured "healthy" by TurboQuant+.
//! 2. **Q8 weights regardless of size.** Higher-precision weights leave
//!    enough headroom for K-side compression. Any Q8 model is presumed
//!    safe; the allowlist marks the `model_type` keys without a
//!    `:q4_k_m` qualifier.
//! 3. **Hybrid MoE / delta-net models** (Qwen3.5 family, Qwen3-Next).
//!    Only some layers use a traditional KV cache — the rest use
//!    delta-net or linear attention paths that bypass the cache
//!    entirely — so the K-side error never accumulates to catastrophic
//!    levels.
//! 4. **Models explicitly validated** by extending the B3 quality gate
//!    in `tests/turbo_kv_e2e.rs` and demonstrating PPL within +2.0% of
//!    the FP16 baseline.
//!
//! # How to extend the allowlist
//!
//! Adding a new entry **requires** a B3 quality-gate pass for that model
//! family. The workflow is:
//!
//! 1. Run `cargo test --test turbo_kv_e2e --release -- --ignored
//!    test_<family>_symmetric_turbo4_quality_gate --nocapture` against the
//!    candidate model checkout.
//! 2. Confirm `(ppl_turbo4_sym - ppl_fp16) / ppl_fp16 ≤ 0.02`.
//! 3. Add the `model_type` (the value of the `model_type` field in
//!    `config.json`) to `ALLOWED_SYMMETRIC_TURBO_FAMILIES` below with a
//!    one-line comment citing the measured PPL delta and the date.
//!
//! # Lookup contract
//!
//! [`is_symmetric_turbo_allowed`] takes the `model_type` string read from
//! `config.json` (canonical lowercase, e.g. `"qwen3_5"` or `"llama"`) and
//! returns whether symmetric Turbo4 is on the allowlist. The check is a
//! straight string-prefix match so config variants like
//! `"qwen3_5_moe"` still match the `"qwen3_5"` family entry.
//!
//! Used by: `src/commands/generate.rs` (CLI fallback warning),
//! `KVCache::new_with_mode_and_seed` (cache initialization), and
//! `tests/turbo_kv_e2e.rs` (allowlist regression tests).

/// Hard-coded list of `model_type` values where symmetric `KVCacheMode::Turbo4`
/// is known to stay within +2.0% PPL of the FP16 baseline.
///
/// Each entry is the `model_type` string read from `config.json`, stored
/// lowercase to match the canonical detection path (`models::detection::
/// get_model_type`). Adding entries requires a B3 quality-gate pass —
/// see the module documentation for the workflow.
///
/// **DO NOT** widen this list without running the quality gate first. The
/// PPL >200 regressions documented in the TurboQuant+ paper happen
/// silently — the model still produces fluent-looking text, it just
/// hallucinates aggressively. The gate is the only meaningful safety net.
pub static ALLOWED_SYMMETRIC_TURBO_FAMILIES: &[&str] = &[
    // Hybrid MoE / delta-net models — only a subset of layers use a
    // traditional KV cache, so K-side compression is partial by
    // construction. TurboQuant+ documented this as "accidentally safe"
    // (validation report, October 2025).
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_next",
    // The bare `model_type` strings used by future B4 follow-up work
    // (large dense models on Q4_K_M and Q8 weights) live here once their
    // per-model quality gates land. They are intentionally NOT in this
    // initial list because mlxcel does not yet read the weight quantization
    // tier from `config.json` — adding a `model_type` entry would also
    // greenlight Q4_K_M variants of the same family, which is exactly what
    // the safety story forbids. Re-evaluate when (B12 docs)
    // wires up a richer model-fingerprint lookup.
];

/// Check whether a model family is on the symmetric Turbo4 allowlist.
///
/// `model_type` is the lowercase string from `config.json`'s `model_type`
/// field (matches the canonical key used by `models::detection::
/// get_model_type`). Returns `true` if the family is allowlisted, `false`
/// otherwise.
///
/// The match is a string-prefix walk so config variants of an allowlisted
/// family still match: `"qwen3_5_moe"` matches the `"qwen3_5"` entry. This
/// keeps the allowlist short while staying safe — TurboQuant+ confirmed
/// the entire Qwen3.5 family (dense + MoE + VLM) shares the same
/// hybrid-cache property.
///
/// Used by: CLI fallback warning in `src/commands/generate.rs`, allowlist
/// regression tests in `tests/turbo_kv_e2e.rs`.
pub fn is_symmetric_turbo_allowed(model_type: &str) -> bool {
    let needle = model_type.trim().to_ascii_lowercase();
    if needle.is_empty() {
        return false;
    }
    ALLOWED_SYMMETRIC_TURBO_FAMILIES
        .iter()
        .any(|&family| needle == family || needle.starts_with(&format!("{family}_")))
}

/// Build the user-facing warning message for a non-allowlisted model.
///
/// The message is identical across CLI / server / library entry points so
/// users who hit the warning in one context can search it directly.
/// Includes the rejected `model_type` and the recommended fallback.
///
/// Used by: `src/commands/generate.rs::resolve_kv_cache_mode_with_allowlist`.
pub fn symmetric_turbo_warning_message(model_type: &str) -> String {
    format!(
        "warning: symmetric turbo4 is risky on this model family \
         (model_type=\"{model_type}\"; Q4_K_M dense models can produce PPL 200+).\n\
         \tFalling back to --kv-cache-mode fp16+turbo4. Override with --force."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowlisted_family_passes() {
        assert!(is_symmetric_turbo_allowed("qwen3_5"));
        assert!(is_symmetric_turbo_allowed("qwen3_5_moe"));
        assert!(is_symmetric_turbo_allowed("qwen3_next"));
    }

    #[test]
    fn case_insensitive_lookup() {
        assert!(is_symmetric_turbo_allowed("Qwen3_5"));
        assert!(is_symmetric_turbo_allowed("QWEN3_5"));
    }

    #[test]
    fn whitespace_is_trimmed() {
        assert!(is_symmetric_turbo_allowed("  qwen3_5  "));
    }

    #[test]
    fn non_allowlisted_family_rejected() {
        assert!(!is_symmetric_turbo_allowed("llama"));
        assert!(!is_symmetric_turbo_allowed("qwen2"));
        assert!(!is_symmetric_turbo_allowed("gemma3"));
        assert!(!is_symmetric_turbo_allowed("mistral"));
    }

    #[test]
    fn empty_string_rejected() {
        assert!(!is_symmetric_turbo_allowed(""));
        assert!(!is_symmetric_turbo_allowed("   "));
    }

    /// Prefix-matching must NOT accept an arbitrary substring — `qwen3_5`
    /// is allowed but `qwen3` (the parent dense family without the .5
    /// hybrid cache) must not be allowed by accident.
    #[test]
    fn parent_family_does_not_match_child_entry() {
        assert!(!is_symmetric_turbo_allowed("qwen3"));
        // And vice-versa: the prefix walk requires the entry to be a true
        // prefix terminated by `_` or end-of-string, not just any substring.
        assert!(!is_symmetric_turbo_allowed("qwen3_5xyz_unrelated"));
    }

    #[test]
    fn warning_message_includes_model_type_and_fallback() {
        let msg = symmetric_turbo_warning_message("llama");
        assert!(msg.contains("model_type=\"llama\""));
        assert!(msg.contains("fp16+turbo4"));
        assert!(msg.contains("--force"));
        // We want the text "warning:" so users grepping logs find it.
        assert!(msg.starts_with("warning:"));
    }
}
