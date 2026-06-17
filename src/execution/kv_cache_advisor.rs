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

//! Advisory KV-cache-mode recommendations (issue #327).
//!
//! Extends the quant/cache advisor so `--recommend-quant` also suggests a
//! TurboQuant KV-cache mode per model family and context range. The output is
//! purely advisory: it prints suggestions to benchmark and never changes the
//! default inference path. The runtime still resolves the KV-cache mode from
//! the CLI/server flags (`--kv-cache-mode`, `--cache-type-k/--cache-type-v`),
//! whose default stays [`KVCacheMode::Fp16`]
//! (`crate::cli::turbo_args::resolve_kv_cache_mode`).
//!
//! # Why this cannot reintroduce the #289 bf16 to f16 promotion
//!
//! The #289 regression came from promoting bf16 quantized-weight scales/biases
//! to f16 in the weight-loading path (`sanitize.rs`). This module only ever
//! produces a [`KVCacheMode`] value. KV-cache modes quantize the K/V *cache*
//! tensors (the per-token attention state) and dequantize back to FP16 for
//! SDPA; they never touch the model weights or their scales/biases. So no
//! recommendation here can imply a weight-dtype change. The unit tests assert
//! the recommendation is always one of the five KV-cache modes named in the
//! issue and never anything that interacts with weight storage.
//!
//! # How the recommendation keys on family and context
//!
//! - **Family** is read from the architecture classification in
//!   [`crate::execution::kv_arch`] ([`KvArchKind`]) plus the raw `model_type`
//!   for the symmetric-Turbo4 PPL allowlist
//!   ([`mlxcel_core::cache::turbo::is_symmetric_turbo_allowed`]).
//! - **Context range** is bucketed by [`KvContextRange`]: short single-request
//!   decode versus long-context serving. Long context and memory-constrained
//!   serving are prioritized over raw short-decode tok/s.
//!
//! Used by: `quant_advisor::advise_quantization` (populates
//! `QuantAdvice::kv_cache_advice`) and `quant_advisor::print_quant_advice`
//! (renders the advisory section under `--recommend-quant`).

use std::path::Path;

use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::cache::turbo::is_symmetric_turbo_allowed;

use crate::execution::kv_arch::{KvArchKind, estimate_kv_arch_from_config};
use crate::execution::memory_estimate::DEFAULT_CTX_LEN;

// ── Context range buckets ───────────────────────────────────────────────────

/// Upper bound (inclusive) of the short context bucket, in tokens.
pub const SHORT_CTX_MAX: u64 = 4_096;
/// Upper bound (inclusive) of the medium context bucket, in tokens.
pub const MEDIUM_CTX_MAX: u64 = 32_768;

/// Context-length bucket the recommendation is keyed on.
///
/// The boundaries mirror the `scripts/bench_kv_cache.sh` sweep cells
/// (4K / 16K / 32K): short single-request decode, medium serving, and the
/// long-context regime where KV-cache footprint dominates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvContextRange {
    /// `ctx <= SHORT_CTX_MAX`. Interactive / single-request decode.
    Short,
    /// `SHORT_CTX_MAX < ctx <= MEDIUM_CTX_MAX`. Medium serving context.
    Medium,
    /// `ctx > MEDIUM_CTX_MAX`. Long-context serving; KV footprint dominates.
    Long,
}

impl KvContextRange {
    /// Bucket a concrete context length (in tokens).
    #[must_use]
    pub fn from_ctx_len(ctx_len: u64) -> Self {
        if ctx_len <= SHORT_CTX_MAX {
            KvContextRange::Short
        } else if ctx_len <= MEDIUM_CTX_MAX {
            KvContextRange::Medium
        } else {
            KvContextRange::Long
        }
    }

    /// All buckets in increasing-context order.
    #[must_use]
    pub fn all() -> [KvContextRange; 3] {
        [
            KvContextRange::Short,
            KvContextRange::Medium,
            KvContextRange::Long,
        ]
    }

    /// Human-readable description used in CLI output.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            KvContextRange::Short => "short  (<=4K tokens, interactive)",
            KvContextRange::Medium => "medium (4K-32K tokens)",
            KvContextRange::Long => "long   (>32K tokens, serving)",
        }
    }
}

// ── Advice value ─────────────────────────────────────────────────────────────

/// A single advisory KV-cache-mode suggestion for one (family, context-range).
///
/// This is data only. Holding one of these never changes the running cache
/// mode; the caller prints it and the user opts in via the CLI/server flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheModeAdvice {
    /// The KV-cache mode suggested as the first thing to benchmark.
    pub suggested: KVCacheMode,
    /// An optional second mode worth comparing (usually more aggressive or a
    /// lower-risk fallback). `None` when a single suggestion is enough.
    pub also_consider: Option<KVCacheMode>,
    /// The detected KV architecture class this advice was computed for.
    pub arch_kind: KvArchKind,
    /// The context-range bucket this advice applies to.
    pub context_range: KvContextRange,
    /// One-sentence rationale, safe to print verbatim.
    pub rationale: &'static str,
}

impl KvCacheModeAdvice {
    /// Render the one-line `range: mode (also: mode2)` header for CLI output.
    #[must_use]
    pub fn headline(&self) -> String {
        let also = match self.also_consider {
            Some(mode) => format!("  (also benchmark: {mode})"),
            None => String::new(),
        };
        format!("{}: {}{}", self.context_range.label(), self.suggested, also)
    }
}

// ── Recommendation core (pure) ─────────────────────────────────────────────────

/// Recommend a KV-cache mode for a model family and context range.
///
/// Pure function: identical inputs always produce identical output and no I/O
/// is performed. The result is one of the five issue-#327 modes: `fp16`,
/// `int8`, `fp16+turbo4` (asymmetric), `turbo4` (symmetric), or `fp16+turbo3`.
///
/// Safety invariants enforced here and covered by tests:
/// - Symmetric [`KVCacheMode::Turbo4`] is suggested **only** for families on
///   the PPL allowlist ([`is_symmetric_turbo_allowed`]). Dense Q4_K_M models
///   off the allowlist can degrade to PPL 200+ under symmetric Turbo4.
/// - [`KVCacheMode::Turbo4Delegated`] is never suggested; it is a decode-speed
///   experiment, not one of the five advisory modes.
/// - MLA and pure-SSM families never receive a Turbo (Walsh-Hadamard) mode,
///   because their cache dimension is not a power of two (MLA) or absent
///   (SSM).
#[must_use]
pub fn recommend_kv_cache_mode(
    arch_kind: KvArchKind,
    model_type: &str,
    range: KvContextRange,
) -> KvCacheModeAdvice {
    use KVCacheMode::{Fp16, Int8, Turbo3Asym, Turbo4, Turbo4Asym};
    use KvArchKind::{Hybrid, MlaCompressed, MlaDecompressed, PureSsm, SlidingWindow, Standard};
    use KvContextRange::{Long, Medium, Short};

    let (suggested, also_consider, rationale): (KVCacheMode, Option<KVCacheMode>, &'static str) =
        match arch_kind {
            // Pure SSM (Mamba/Mamba2): no context-proportional KV cache, so a
            // Turbo KV mode saves almost nothing.
            PureSsm => (
                Fp16,
                None,
                "Pure SSM keeps an O(1) recurrent state and no context-proportional KV cache, so a Turbo KV mode saves almost nothing. Keep fp16.",
            ),
            // MLA (DeepSeek): the cached latent is small and its dimension is
            // not a power of two, so the Turbo Walsh-Hadamard V path does not
            // apply. Per-token INT8 has no head-dim constraint.
            MlaCompressed | MlaDecompressed => match range {
                Short => (
                    Fp16,
                    None,
                    "MLA already caches a compact low-rank latent, so at short context the KV footprint is small. Keep fp16.",
                ),
                Medium | Long => (
                    Int8,
                    None,
                    "MLA caches a low-rank latent whose dimension is not a power of two, so the Turbo Walsh-Hadamard V path does not apply; per-token int8 is the safe KV compression to benchmark here.",
                ),
            },
            // Standard / sliding-window / hybrid attention: the Turbo target.
            Standard | SlidingWindow | Hybrid => match range {
                Short => (
                    Fp16,
                    None,
                    "Short context keeps the KV cache small, so fp16 preserves baseline quality and decode speed.",
                ),
                Medium => (
                    Turbo4Asym,
                    Some(Int8),
                    "fp16+turbo4 keeps K in fp16 (softmax never sees a quantized K) and compresses V, the safest Turbo starting point; int8 is a simpler ~50% alternative. Benchmark both against fp16.",
                ),
                Long => {
                    if is_symmetric_turbo_allowed(model_type) {
                        (
                            Turbo4,
                            Some(Turbo4Asym),
                            "This family is on the symmetric-Turbo4 PPL allowlist, so turbo4 (4-bit K and V) offers the largest KV savings; fp16+turbo4 is the lower-risk fallback. Benchmark both for quality and throughput.",
                        )
                    } else {
                        (
                            Turbo4Asym,
                            Some(Turbo3Asym),
                            "fp16+turbo4 (K stays fp16) is the safest large KV saver. Symmetric turbo4 is withheld because this family is not on the PPL allowlist (dense Q4_K_M can reach PPL 200+); fp16+turbo3 trades more V error for further savings when memory is tight.",
                        )
                    }
                }
            },
        };

    KvCacheModeAdvice {
        suggested,
        also_consider,
        arch_kind,
        context_range: range,
        rationale,
    }
}

// ── Config-driven advice ────────────────────────────────────────────────────

/// Read `config.json` and produce advisory KV-cache-mode suggestions for every
/// context-range bucket.
///
/// Returns an empty vector when `config.json` cannot be read/parsed or when the
/// architecture cannot be classified, so callers can treat "no advice" as a
/// soft, non-fatal condition. Reads only `config.json`; never loads weights.
#[must_use]
pub fn advise_kv_cache_modes(model_path: &Path) -> Vec<KvCacheModeAdvice> {
    let Some((arch_kind, model_type)) = arch_and_type_from_path(model_path) else {
        return Vec::new();
    };
    KvContextRange::all()
        .into_iter()
        .map(|range| recommend_kv_cache_mode(arch_kind, &model_type, range))
        .collect()
}

fn arch_and_type_from_path(model_path: &Path) -> Option<(KvArchKind, String)> {
    let config_str = std::fs::read_to_string(model_path.join("config.json")).ok()?;
    let config: serde_json::Value = serde_json::from_str(&config_str).ok()?;
    arch_and_type_from_config(&config)
}

fn arch_and_type_from_config(config: &serde_json::Value) -> Option<(KvArchKind, String)> {
    // Reuse the architecture classifier so the advisor and the memory
    // estimator never disagree on what family a model is. Batch/dtype/ctx do
    // not affect the detected `kind`, so the defaults are arbitrary here.
    let estimate = estimate_kv_arch_from_config(config, DEFAULT_CTX_LEN, false, 1)?;
    Some((estimate.kind, read_model_type(config)))
}

/// Read the canonical lowercase `model_type` string, matching the allowlist's
/// lookup contract. VLMs may nest it under `text_config`.
fn read_model_type(config: &serde_json::Value) -> String {
    let text = config.get("text_config").unwrap_or(config);
    text.get("model_type")
        .and_then(|v| v.as_str())
        .or_else(|| config.get("model_type").and_then(|v| v.as_str()))
        .unwrap_or("")
        .to_ascii_lowercase()
}

// ── Rendering ──────────────────────────────────────────────────────────────────

/// Render the advisory KV-cache section as a printable block.
///
/// Returns an empty string when there is nothing to advise, so the caller can
/// skip printing entirely. The block is explicit that the advice is opt-in,
/// that the default path is unchanged, and that KV-cache modes never touch the
/// model weights.
#[must_use]
pub fn render_kv_cache_advice(advices: &[KvCacheModeAdvice]) -> String {
    if advices.is_empty() {
        return String::new();
    }

    let mut out = String::new();
    out.push_str("  KV-cache mode suggestions (advisory, opt-in):\n");
    if let Some(first) = advices.first() {
        out.push_str(&format!("    Architecture: {}\n", first.arch_kind.label()));
    }
    out.push_str(
        "    These are suggestions to benchmark, not validated defaults. The default\n\
         \x20   inference path is unchanged (fp16); opt in with --kv-cache-mode or\n\
         \x20   --cache-type-k/--cache-type-v. KV-cache modes quantize only the K/V cache\n\
         \x20   tensors, never the model weights, so they do not change quantized-weight\n\
         \x20   dtype. Long context and memory-constrained serving are prioritized over\n\
         \x20   raw short-decode tok/s.\n\n",
    );

    for advice in advices {
        out.push_str(&format!("    {}\n", advice.headline()));
        out.push_str(&format!("        {}\n", advice.rationale));
    }

    out.push_str(
        "\n    Validate per family before adopting; see docs/turbo-kv-cache.md for the\n\
         \x20   quality and throughput checklist.\n",
    );
    out
}

/// Print [`render_kv_cache_advice`] to stdout, skipping output when empty.
pub fn print_kv_cache_advice(advices: &[KvCacheModeAdvice]) {
    let block = render_kv_cache_advice(advices);
    if !block.is_empty() {
        println!();
        print!("{block}");
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// The five KV-cache modes the issue scopes the advisor to. `Turbo4Delegated`
    /// is intentionally excluded.
    const ALLOWED_MODES: [KVCacheMode; 5] = [
        KVCacheMode::Fp16,
        KVCacheMode::Int8,
        KVCacheMode::Turbo4Asym,
        KVCacheMode::Turbo4,
        KVCacheMode::Turbo3Asym,
    ];

    const ALL_ARCH_KINDS: [KvArchKind; 6] = [
        KvArchKind::Standard,
        KvArchKind::SlidingWindow,
        KvArchKind::MlaCompressed,
        KvArchKind::MlaDecompressed,
        KvArchKind::Hybrid,
        KvArchKind::PureSsm,
    ];

    #[test]
    fn context_range_bucketing() {
        assert_eq!(KvContextRange::from_ctx_len(0), KvContextRange::Short);
        assert_eq!(KvContextRange::from_ctx_len(4_096), KvContextRange::Short);
        assert_eq!(KvContextRange::from_ctx_len(4_097), KvContextRange::Medium);
        assert_eq!(KvContextRange::from_ctx_len(16_384), KvContextRange::Medium);
        assert_eq!(KvContextRange::from_ctx_len(32_768), KvContextRange::Medium);
        assert_eq!(KvContextRange::from_ctx_len(32_769), KvContextRange::Long);
        assert_eq!(KvContextRange::from_ctx_len(131_072), KvContextRange::Long);
    }

    #[test]
    fn short_context_is_always_fp16() {
        for kind in ALL_ARCH_KINDS {
            let advice = recommend_kv_cache_mode(kind, "llama", KvContextRange::Short);
            assert_eq!(
                advice.suggested,
                KVCacheMode::Fp16,
                "short context for {kind:?} should keep fp16"
            );
            assert_eq!(advice.also_consider, None);
        }
    }

    #[test]
    fn standard_medium_suggests_turbo4_asym() {
        let advice = recommend_kv_cache_mode(KvArchKind::Standard, "llama", KvContextRange::Medium);
        assert_eq!(advice.suggested, KVCacheMode::Turbo4Asym);
        assert_eq!(advice.also_consider, Some(KVCacheMode::Int8));
    }

    #[test]
    fn standard_long_non_allowlisted_never_symmetric_turbo4() {
        // Llama is a dense Q4_K_M-style family and is NOT on the allowlist.
        let advice = recommend_kv_cache_mode(KvArchKind::Standard, "llama", KvContextRange::Long);
        assert_eq!(advice.suggested, KVCacheMode::Turbo4Asym);
        assert_ne!(advice.suggested, KVCacheMode::Turbo4);
        assert_eq!(advice.also_consider, Some(KVCacheMode::Turbo3Asym));
    }

    #[test]
    fn allowlisted_family_long_suggests_symmetric_turbo4() {
        for mt in ["qwen3_5", "qwen3_5_moe", "qwen3_next"] {
            let advice = recommend_kv_cache_mode(KvArchKind::Standard, mt, KvContextRange::Long);
            assert_eq!(
                advice.suggested,
                KVCacheMode::Turbo4,
                "allowlisted family {mt} at long context should suggest turbo4"
            );
            assert_eq!(advice.also_consider, Some(KVCacheMode::Turbo4Asym));
        }
    }

    /// The most important safety invariant: symmetric Turbo4 is suggested only
    /// for allowlisted families, never otherwise, in any field, at any range.
    #[test]
    fn symmetric_turbo4_only_for_allowlisted_families() {
        let off_allowlist = ["llama", "qwen2", "gemma3", "mistral", "qwen3", "phi3", ""];
        for kind in ALL_ARCH_KINDS {
            for range in KvContextRange::all() {
                for mt in off_allowlist {
                    let advice = recommend_kv_cache_mode(kind, mt, range);
                    assert_ne!(
                        advice.suggested,
                        KVCacheMode::Turbo4,
                        "{kind:?}/{mt}/{range:?} must not suggest symmetric turbo4"
                    );
                    assert_ne!(
                        advice.also_consider,
                        Some(KVCacheMode::Turbo4),
                        "{kind:?}/{mt}/{range:?} must not list symmetric turbo4"
                    );
                }
            }
        }
    }

    #[test]
    fn pure_ssm_is_always_fp16() {
        for range in KvContextRange::all() {
            let advice = recommend_kv_cache_mode(KvArchKind::PureSsm, "mamba", range);
            assert_eq!(advice.suggested, KVCacheMode::Fp16);
            assert_eq!(advice.also_consider, None);
        }
    }

    #[test]
    fn mla_uses_int8_never_turbo() {
        for kind in [KvArchKind::MlaCompressed, KvArchKind::MlaDecompressed] {
            // Short stays fp16.
            assert_eq!(
                recommend_kv_cache_mode(kind, "deepseek_v3", KvContextRange::Short).suggested,
                KVCacheMode::Fp16
            );
            for range in [KvContextRange::Medium, KvContextRange::Long] {
                let advice = recommend_kv_cache_mode(kind, "deepseek_v3", range);
                assert_eq!(advice.suggested, KVCacheMode::Int8);
                // MLA must never get a Walsh-Hadamard Turbo mode.
                for mode in [
                    KVCacheMode::Turbo4,
                    KVCacheMode::Turbo4Asym,
                    KVCacheMode::Turbo3Asym,
                ] {
                    assert_ne!(advice.suggested, mode);
                    assert_ne!(advice.also_consider, Some(mode));
                }
            }
        }
    }

    /// Every recommendation is one of the five KV-cache modes named in the
    /// issue, and never `Turbo4Delegated`. Because the output is always a
    /// `KVCacheMode` (a KV-cache-only storage setting), it cannot imply a
    /// bf16 to f16 promotion of quantized model weights (the #289 landmine).
    #[test]
    fn recommendation_is_always_an_allowed_kv_cache_mode() {
        let model_types = ["llama", "qwen3_5", "deepseek_v3", "mamba", "gemma3", ""];
        for kind in ALL_ARCH_KINDS {
            for range in KvContextRange::all() {
                for mt in model_types {
                    let advice = recommend_kv_cache_mode(kind, mt, range);
                    assert!(
                        ALLOWED_MODES.contains(&advice.suggested),
                        "{kind:?}/{mt}/{range:?} suggested an out-of-scope mode: {:?}",
                        advice.suggested
                    );
                    assert_ne!(advice.suggested, KVCacheMode::Turbo4Delegated);
                    if let Some(also) = advice.also_consider {
                        assert!(
                            ALLOWED_MODES.contains(&also),
                            "{kind:?}/{mt}/{range:?} also_consider out-of-scope: {also:?}"
                        );
                        assert_ne!(also, KVCacheMode::Turbo4Delegated);
                    }
                }
            }
        }
    }

    #[test]
    fn advise_returns_empty_for_missing_config() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(advise_kv_cache_modes(tmp.path()).is_empty());
    }

    #[test]
    fn advise_classifies_standard_config_for_all_ranges() {
        let cfg = serde_json::json!({
            "model_type": "llama",
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128
        });
        let (kind, mt) = arch_and_type_from_config(&cfg).unwrap();
        assert_eq!(kind, KvArchKind::Standard);
        assert_eq!(mt, "llama");

        let advices: Vec<_> = KvContextRange::all()
            .into_iter()
            .map(|r| recommend_kv_cache_mode(kind, &mt, r))
            .collect();
        assert_eq!(advices.len(), 3);
        assert_eq!(advices[0].context_range, KvContextRange::Short);
        assert_eq!(advices[0].suggested, KVCacheMode::Fp16);
        assert_eq!(advices[1].suggested, KVCacheMode::Turbo4Asym);
        assert_eq!(advices[2].suggested, KVCacheMode::Turbo4Asym);
    }

    /// Computing advice must not change the runtime default. With no opt-in
    /// flags, the resolver still returns `Fp16`.
    #[test]
    fn default_kv_cache_mode_unchanged_without_opt_in() {
        use crate::cli::turbo_args::resolve_kv_cache_mode;

        // Build advice (the new behavior) ...
        let _ = recommend_kv_cache_mode(KvArchKind::Standard, "llama", KvContextRange::Long);
        // ... the default resolution is still fp16.
        let resolved = resolve_kv_cache_mode(None, None, None).unwrap();
        assert_eq!(resolved, KVCacheMode::Fp16);
    }

    #[test]
    fn render_block_states_advisory_and_no_weight_change() {
        let advices = vec![
            recommend_kv_cache_mode(KvArchKind::Standard, "llama", KvContextRange::Short),
            recommend_kv_cache_mode(KvArchKind::Standard, "llama", KvContextRange::Medium),
            recommend_kv_cache_mode(KvArchKind::Standard, "llama", KvContextRange::Long),
        ];
        let block = render_kv_cache_advice(&advices);
        assert!(block.contains("advisory"));
        assert!(block.contains("opt-in") || block.contains("opt in"));
        assert!(block.contains("default"));
        // The bf16->f16 guarantee must be visible to the user.
        assert!(block.contains("never the model weights"));
        // All three ranges appear.
        assert!(block.contains("short"));
        assert!(block.contains("medium"));
        assert!(block.contains("long"));
        // No em dashes in user-facing output.
        assert!(!block.contains('\u{2014}'));
    }

    #[test]
    fn render_block_empty_for_no_advice() {
        assert_eq!(render_kv_cache_advice(&[]), "");
    }

    #[test]
    fn headline_includes_also_consider_when_present() {
        let advice = recommend_kv_cache_mode(KvArchKind::Standard, "llama", KvContextRange::Medium);
        let line = advice.headline();
        assert!(line.contains("fp16+turbo4"));
        assert!(line.contains("also benchmark"));
        assert!(line.contains("int8"));
    }
}
