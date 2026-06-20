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
//! - **Head-dimension guard**: Turbo (Walsh-Hadamard) modes require a
//!   power-of-two attention head dimension. When the head dim is derivable from
//!   `config.json` and is not a power of two (for example Phi-2 at 80),
//!   Turbo suggestions are downgraded to `int8` or `fp16`, matching the
//!   treatment of MLA families.
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
/// is performed. The result is one of six KV-cache modes: `fp16`, `int8`,
/// `turbo4-delegated`, `fp16+turbo4` (asymmetric), `turbo4` (symmetric), or
/// `fp16+turbo3`.
///
/// The advice keys on the measured Apple-Silicon decode/prefill sweep
/// (`benchmarks/turbo_kv/`, M1 Ultra, post-#369): every quantized KV mode
/// decodes slower than fp16, so the suggestions trade KV footprint for memory,
/// not throughput. The fast quantized picks are `int8` (about 2x compression,
/// robust on both dense and MoE) and `turbo4-delegated` (about 4x V, the
/// fastest of the Turbo codecs); `fp16+turbo4` keeps K exact at a higher decode
/// cost; symmetric `turbo4` maximizes compression but is the slowest and
/// quality-sensitive.
///
/// Safety invariants enforced here and covered by tests:
/// - Symmetric [`KVCacheMode::Turbo4`] is suggested **only** for families on
///   the PPL allowlist ([`is_symmetric_turbo_allowed`]). Dense Q4_K_M models
///   off the allowlist can degrade to PPL 200+ under symmetric Turbo4.
/// - [`KVCacheMode::Turbo3Asym`] is never the suggested or also-benchmark mode.
///   Its measured decode throughput (about 0.01-0.07x of fp16) makes it a
///   memory-extremis last resort documented in `docs/turbo-kv-cache.md`, not an
///   advisory pick.
/// - MLA and pure-SSM families never receive a Turbo (Walsh-Hadamard) mode,
///   because their cache dimension is not a power of two (MLA) or absent
///   (SSM).
#[must_use]
pub fn recommend_kv_cache_mode(
    arch_kind: KvArchKind,
    model_type: &str,
    range: KvContextRange,
) -> KvCacheModeAdvice {
    use KVCacheMode::{Fp16, Int8, Turbo4, Turbo4Asym, Turbo4Delegated};
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
                    Int8,
                    Some(Turbo4Delegated),
                    "int8 is the fastest quantized KV mode here (about 2x compression, robust on dense and MoE). turbo4-delegated reaches about 4x V compression at similar dense decode speed; fp16+turbo4 keeps K exact for the same V savings at a higher decode cost. Every quantized mode decodes slower than fp16, so adopt only when KV memory matters and benchmark first.",
                ),
                Long => {
                    if is_symmetric_turbo_allowed(model_type) {
                        (
                            Turbo4,
                            Some(Turbo4Delegated),
                            "This family is on the symmetric-Turbo4 PPL allowlist, so turbo4 (4-bit K and V) gives the largest KV footprint savings for long-context serving; turbo4-delegated is the faster fallback (4-bit cold V, exact recent V, fp16 K). Both decode slower than fp16, so adopt only when KV memory is the binding constraint.",
                        )
                    } else {
                        (
                            Turbo4Delegated,
                            Some(Turbo4Asym),
                            "turbo4-delegated gives about 4x V compression at the fastest quantized decode speed (4-bit cold V, exact recent V, fp16 K); fp16+turbo4 is the exact-K alternative at a higher decode cost. Symmetric turbo4 is withheld off the PPL allowlist (dense Q4_K_M can reach PPL 200+). All quantized KV modes trade memory for footprint, not decode speed.",
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
///
/// When the attention head dimension is derivable from the config and is not a
/// power of two (for example Phi-2 at head_dim 80), any Turbo
/// (Walsh-Hadamard) suggestion is downgraded to `int8` or `fp16`. The Turbo
/// path panics on non-power-of-two head dims via
/// `TurboQuantParams::new`; this guard prevents that panic from reaching a
/// user-initiated benchmark run.
#[must_use]
pub fn advise_kv_cache_modes(model_path: &Path) -> Vec<KvCacheModeAdvice> {
    let config_str = match std::fs::read_to_string(model_path.join("config.json")) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let config: serde_json::Value = match serde_json::from_str(&config_str) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    advise_kv_cache_modes_from_config(&config)
}

/// Pure-data core of [`advise_kv_cache_modes`], operating on an already-parsed
/// config value. Separated so unit tests can drive it without touching the
/// filesystem.
fn advise_kv_cache_modes_from_config(config: &serde_json::Value) -> Vec<KvCacheModeAdvice> {
    let Some((arch_kind, model_type)) = arch_and_type_from_config(config) else {
        return Vec::new();
    };
    // Turbo modes require a power-of-two head dimension. When the config lets
    // us derive the head dim and it is not a power of two, downgrade all Turbo
    // suggestions to head-dim-agnostic alternatives.
    let turbo_ok = read_head_dim(config).is_none_or(|d| d > 0 && d.is_power_of_two());
    KvContextRange::all()
        .into_iter()
        .map(|range| {
            let advice = recommend_kv_cache_mode(arch_kind, &model_type, range);
            if turbo_ok {
                advice
            } else {
                downgrade_for_non_power_of_two(advice)
            }
        })
        .collect()
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

/// Derive the attention head dimension from `config.json`.
///
/// Checks `text_config` first, then top-level. Returns an explicit `head_dim`
/// or `head_size` when present; otherwise divides the hidden size by the
/// attention-head count. Returns `None` when the required fields are absent or
/// heads is zero.
///
/// The hidden-size and head-count field-name lists mirror
/// [`crate::execution::kv_arch`]'s `attn_dims` so the non-power-of-two guard
/// derives the same head dim the classifier used. Without this, alternate
/// config naming (OLMo / MPT-style `d_model` + `n_heads`) would make this
/// return `None`, defaulting `turbo_ok` to `true` and slipping a
/// non-power-of-two head dim past the guard.
fn read_head_dim(config: &serde_json::Value) -> Option<u64> {
    let text = config.get("text_config").unwrap_or(config);
    if let Some(d) = text.get("head_dim").and_then(|v| v.as_u64()) {
        return Some(d);
    }
    if let Some(d) = text.get("head_size").and_then(|v| v.as_u64()) {
        return Some(d);
    }
    let hidden = text
        .get("hidden_size")
        .or_else(|| text.get("d_model"))
        .or_else(|| text.get("dim"))
        .or_else(|| text.get("model_dim"))
        .and_then(|v| v.as_u64())?;
    let heads = text
        .get("num_attention_heads")
        .or_else(|| text.get("num_heads"))
        .or_else(|| text.get("n_heads"))
        .or_else(|| text.get("n_head"))
        .and_then(|v| v.as_u64())?;
    if heads == 0 {
        return None;
    }
    Some(hidden / heads)
}

/// The rationale emitted when a Turbo suggestion is downgraded because the
/// model's head dimension is not a power of two.
const NON_POW2_RATIONALE: &str = "This family's attention head dimension is not \
    a power of two, so the Turbo Walsh-Hadamard V path does not apply; \
    per-token int8 is the head-dim-agnostic KV compression to benchmark here.";

/// Returns `true` for the modes that use the Walsh-Hadamard transform and so
/// require a power-of-two head dimension. `Turbo4Delegated` is included because
/// it is a hot/cold split built on the same `Turbo4Asym` codec.
fn is_walsh_hadamard_turbo(mode: KVCacheMode) -> bool {
    matches!(
        mode,
        KVCacheMode::Turbo4Asym
            | KVCacheMode::Turbo4
            | KVCacheMode::Turbo3Asym
            | KVCacheMode::Turbo4Delegated
    )
}

/// Downgrade any Turbo (Walsh-Hadamard) suggestion to `int8` when the model's
/// attention head dimension is not a power of two.
///
/// If `suggested` is a Turbo mode, it is replaced by `int8` and `also_consider`
/// is cleared. If only `also_consider` is a Turbo mode, that field is cleared.
/// Non-Turbo suggestions (fp16, int8) pass through unchanged.
fn downgrade_for_non_power_of_two(advice: KvCacheModeAdvice) -> KvCacheModeAdvice {
    if is_walsh_hadamard_turbo(advice.suggested) {
        KvCacheModeAdvice {
            suggested: KVCacheMode::Int8,
            also_consider: None,
            rationale: NON_POW2_RATIONALE,
            ..advice
        }
    } else if advice.also_consider.is_some_and(is_walsh_hadamard_turbo) {
        KvCacheModeAdvice {
            also_consider: None,
            ..advice
        }
    } else {
        advice
    }
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

#[cfg(test)]
#[path = "kv_cache_advisor_tests.rs"]
mod tests;
