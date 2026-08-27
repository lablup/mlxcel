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

//! TurboQuant KV-cache CLI flag group.
//!
//! This module owns the single canonical clap definition for all
//! TurboQuant-family KV-cache options. Every mlxcel binary that exposes a
//! generation surface flattens [`TurboKvCacheArgs`] via
//! `#[command(flatten)]`, which means:
//!
//! - The user-visible `--help` text is identical across `mlxcel generate`,
//!   `mlxcel serve`, and `mlxcel-server`.
//! - Adding, renaming, or extending a flag only requires editing this file.
//! - The shared resolution helpers ([`resolve_kv_cache_mode`],
//!   [`env_fallback_cache_type_k`], [`env_fallback_cache_type_v`]) live
//!   alongside the flag definitions, eliminating drift between the
//!   binary entry points.
//!
//! Used by: mlxcel generate, mlxcel serve, mlxcel-server.
//!
//! See `docs/turbo-kv-cache.md` for the operator-facing reference.

use std::path::Path;

use clap::Args;
use mlxcel_core::cache::KVCacheMode;
use mlxcel_core::cache::turbo::resolve_kv_cache_mode_for_model;

/// Shared TurboQuant KV-cache flag group.
///
/// Flattened into the clap `Args` struct of every binary that exposes a
/// generation surface. The four flags below MUST stay in sync across all
/// callers; see the integration test `tests/cli_help_consistency.rs`.
#[derive(Args, Debug, Clone, Default)]
#[command(next_help_heading = "KV Cache (TurboQuant) Options")]
pub struct TurboKvCacheArgs {
    /// K-side KV cache quantization type.
    ///
    /// Accepted values:
    ///   fp16 / f16         Standard half-precision storage (default, no overhead).
    ///   int8               Per-token INT8 absmax quantization. ~50% KV memory
    ///                      savings with small per-token quantization error.
    ///   fp16+turbo4        Asymmetric Fp16-K + Turbo4-V (alias: turbo4-asym).
    ///                      K side stays FP16; V side uses 4-bit PolarQuant
    ///                      with Walsh-Hadamard rotation. ~26% net KV savings
    ///                      at long context with negligible quality loss on
    ///                      Q4_K_M dense weights.
    ///   fp16+turbo3        Asymmetric Fp16-K + Turbo3-V (alias: turbo3-asym).
    ///                      V side uses a 3-bit codebook for ~5.1x total KV
    ///                      savings at slightly higher V reconstruction error.
    ///                      Symmetric Turbo3 is not offered.
    ///   turbo4             Symmetric Turbo4-K + Turbo4-V (alias: turbo4-sym).
    ///                      Allowlisted dense models only; non-allowlisted
    ///                      models fall back to turbo4-asym. ~73% net KV
    ///                      savings.
    ///   turbo4-delegated   Hot/cold split on top of fp16+turbo4. FP16 hot
    ///                      tail + packed turbo cold body. Targets >= 97% of
    ///                      FP16 decode speed at 4K and >= 95% at 16K on
    ///                      M5 Max.
    ///
    /// When only one of --cache-type-k / --cache-type-v is specified, the
    /// other side defaults to fp16. Takes precedence over --kv-cache-mode
    /// when both are supplied (a warning is emitted). Unsupported K/V
    /// combinations are rejected at startup with a clear error.
    ///
    /// Also read from LLAMA_ARG_CACHE_TYPE_K.
    #[arg(
        long = "cache-type-k",
        env = "LLAMA_ARG_CACHE_TYPE_K",
        value_name = "TYPE",
        verbatim_doc_comment
    )]
    pub cache_type_k: Option<String>,

    /// V-side KV cache quantization type.
    ///
    /// Accepts the same value set as --cache-type-k. When only one of
    /// --cache-type-k / --cache-type-v is specified, the other side defaults
    /// to fp16. Takes precedence over --kv-cache-mode when both are supplied.
    ///
    /// Also read from LLAMA_ARG_CACHE_TYPE_V.
    #[arg(
        long = "cache-type-v",
        env = "LLAMA_ARG_CACHE_TYPE_V",
        value_name = "TYPE"
    )]
    pub cache_type_v: Option<String>,

    /// KV cache mode shorthand (legacy; prefer --cache-type-k / --cache-type-v).
    ///
    /// Sets both K and V to the same mode. Accepted values: fp16 (default),
    /// int8, fp16+turbo4 (alias turbo4-asym), fp16+turbo3 (alias
    /// turbo3-asym / turbo3), turbo4 (alias turbo4-sym), turbo4-delegated.
    ///
    /// When --cache-type-k or --cache-type-v are also supplied, the split
    /// flags win and this flag is ignored (with a warning).
    #[arg(long = "kv-cache-mode", value_name = "MODE")]
    pub kv_cache_mode: Option<String>,

    /// Number of boundary transformer layers to keep at higher precision when
    /// a Turbo4* KV cache mode is active.
    ///
    /// The first N and last N V layers contribute disproportionately to
    /// quality loss under aggressive V quantization. Keeping them at FP16
    /// recovers a large fraction of the perplexity gap at zero speed cost.
    ///
    /// 0 disables the policy entirely. The count is clamped to
    /// min(value, n_layers / 2) so a too-large value on a shallow model
    /// degrades gracefully into "every layer FP16". Inert when the resolved
    /// KV cache mode is fp16 or int8.
    ///
    /// Equivalent to setting MLXCEL_KV_BOUNDARY_V_LAYERS in the environment;
    /// the CLI flag wins when both are present.
    #[arg(long = "turbo-boundary-v", value_name = "COUNT")]
    pub turbo_boundary_v: Option<i32>,
}

impl TurboKvCacheArgs {
    /// Wire the resolved boundary-V count into the process environment so
    /// `mlxcel-core` picks it up at first cache instantiation.
    ///
    /// `mlxcel-core` reads `MLXCEL_KV_BOUNDARY_V_LAYERS` (constant
    /// [`mlxcel_core::cache::turbo::BOUNDARY_V_ENV`]) inside
    /// `boundary_v_layers_from_env()` when constructing the per-layer KV
    /// cache table. The CLI flag is just a more discoverable spelling of
    /// that env var; this helper performs the translation. Clamping against
    /// `n_layers / 2` and the negative-input "treat as 0" rule are enforced
    /// downstream by `mlxcel_core::cache::turbo::resolve_boundary_count` and
    /// `parse_boundary_v_str`. The CLI surface forwards the raw integer
    /// verbatim and lets the runtime own the validation contract.
    ///
    /// Both `mlxcel serve` and `mlxcel-server` accept the flag indirectly
    /// through this helper; `mlxcel generate` already calls it in the
    /// startup path. When `turbo_boundary_v` is `None` this is a no-op (the
    /// process environment is left untouched, so any caller-set
    /// `MLXCEL_KV_BOUNDARY_V_LAYERS` continues to win).
    ///
    /// SAFETY: this function calls `std::env::set_var`, which is
    /// process-global and is not thread-safe with concurrent `getenv` from
    /// other threads (Rust 1.84 marks it `unsafe` for that reason). It MUST
    /// be invoked while:
    ///
    /// 1. No `mlxcel-core` generator, server worker, or MLX background
    ///    stream has been constructed yet   those read
    ///    `MLXCEL_KV_BOUNDARY_V_LAYERS` at first cache instantiation; AND
    /// 2. No other thread is concurrently reading the process environment.
    ///
    /// On the `mlxcel generate` path the caller is fully synchronous, so
    /// (2) is trivially satisfied. On the `mlxcel serve` / `mlxcel-server`
    /// paths the `#[tokio::main]` runtime has already been built (its
    /// worker threads are spawned but parked, with no tasks scheduled
    /// yet), and the only env reads that follow run on the same thread
    /// that called `set_var`. Future refactors that spawn tasks before
    /// this call site, or that read env from spawned tasks during
    /// pre-startup, must keep the write upstream of the runtime
    /// bring-up   or switch to a typed config rather than a process env
    /// var.
    pub fn apply_to_environment(&self) {
        if let Some(boundary) = self.turbo_boundary_v {
            // SAFETY: see the function-level SAFETY comment for the full
            // precondition. The env var name is a fixed compile-time
            // constant from mlxcel_core, eliminating any chance of
            // writing to the wrong key.
            unsafe {
                std::env::set_var(
                    mlxcel_core::cache::turbo::BOUNDARY_V_ENV,
                    boundary.to_string(),
                );
            }
        }
    }
}

// ── (B11)   KV cache type split flags ─────────────────────────────
// (Comment kept in non-doc form for code-archeology only; the user-facing
// help text intentionally omits closed-repo issue numbers.)

/// Supported K/V combinations and their corresponding `KVCacheMode`.
///
/// | K      | V                    | Mode              |
/// |--------|----------------------|-------------------|
/// | fp16   | fp16                 | `Fp16`            |
/// | int8   | int8                 | `Int8`            |
/// | fp16   | turbo4 / turbo4-asym | `Turbo4Asym`      |
/// | turbo4 | turbo4               | `Turbo4`          |
/// | fp16   | turbo4-delegated     | `Turbo4Delegated` |
/// | fp16   | turbo3 / turbo3-asym | `Turbo3Asym`      |
///
/// Any other combination returns an error with a description of valid pairs.
pub fn resolve_kv_cache_mode(
    cache_type_k: Option<&str>,
    cache_type_v: Option<&str>,
    kv_cache_mode_legacy: Option<&str>,
) -> Result<KVCacheMode, String> {
    let have_split = cache_type_k.is_some() || cache_type_v.is_some();
    let have_legacy = kv_cache_mode_legacy.is_some();

    if have_split && have_legacy {
        tracing::warn!(
            "--cache-type-k/--cache-type-v and --kv-cache-mode are both set; \
             --cache-type-k/--cache-type-v take precedence (use one or the other)"
        );
    }

    if have_split {
        // Resolve each side, defaulting the unspecified side to fp16.
        let k_str = cache_type_k.unwrap_or("fp16");
        let v_str = cache_type_v.unwrap_or("fp16");

        let k_mode =
            parse_split_cache_type(k_str).map_err(|_| cache_type_error("--cache-type-k", k_str))?;
        let v_mode =
            parse_split_cache_type(v_str).map_err(|_| cache_type_error("--cache-type-v", v_str))?;

        return map_kv_modes_to_cache_mode(k_mode, v_mode);
    }

    if let Some(legacy) = kv_cache_mode_legacy {
        return legacy
            .parse::<KVCacheMode>()
            .map_err(|_| format!("unrecognised --kv-cache-mode value \"{legacy}\""));
    }

    // Default: FP16 (bit-exact baseline).
    Ok(KVCacheMode::Fp16)
}

/// Parse one split cache type, including llama.cpp's exact `f16` spelling for
/// unquantized FP16 storage. GGML quantizer names are deliberately not
/// translated here: accepting `q8_0` or `q4_0` as a mathematically different
/// mlxcel mode would make a deployment appear compatible while changing its
/// cache arithmetic.
fn parse_split_cache_type(value: &str) -> Result<KVCacheMode, ()> {
    if value.eq_ignore_ascii_case("f16") {
        return Ok(KVCacheMode::Fp16);
    }
    value.parse::<KVCacheMode>().map_err(|_| ())
}

/// b10621's quantized KV cache types.
///
/// From its `kv_cache_types` table
/// (<https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/common/arg.cpp>).
/// `--cache-type-k`'s help advertises `f32, f16, bf16, q8_0, q4_0, q4_1,
/// iq4_nl, q5_0, q5_1`; these are the quantizers among them.
const GGML_KV_QUANTIZERS: [&str; 6] = ["q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"];

/// b10621's unquantized KV cache types other than `f16`.
///
/// `GGML_TYPE_F32` and `GGML_TYPE_BF16` are plain float types, not quantizers,
/// so they are refused for a different reason and get their own sentence.
const GGML_KV_FLOAT_TYPES: [&str; 2] = ["f32", "bf16"];

/// Build the rejection for an unrecognized `--cache-type-k` / `--cache-type-v`
/// value (issue #1445).
///
/// The two vocabularies overlap on exactly one spelling. `f16` names the same
/// thing on both sides, unquantized half-precision KV storage, so it is
/// aliased onto [`KVCacheMode::Fp16`]. Every other GGML name is a *different
/// quantizer*: `q8_0` is a block-wise 8-bit format with a per-block scale,
/// while mlxcel's `int8` is per-token absmax, and the Turbo modes are
/// PolarQuant with a Walsh-Hadamard rotation. Mapping one onto the other would
/// leave a deployment believing its cache arithmetic is something it is not,
/// which is why they are named here rather than translated.
///
/// A GGML name gets its own message so the operator learns *why* the spelling
/// they copied from a llama-server command line does not carry over; anything
/// else gets the plain list of what mlxcel accepts.
fn cache_type_error(option: &str, value: &str) -> String {
    const ALTERNATIVES: &str = "Use instead: f16 (identical storage), int8 (per-token absmax), \
                                fp16+turbo4, fp16+turbo3, turbo4, or turbo4-delegated. See \
                                docs/turbo-kv-cache.md for what each one costs and measures.";
    let lowered = value.to_ascii_lowercase();
    if GGML_KV_QUANTIZERS.contains(&lowered.as_str()) {
        return format!(
            "{option} {value} is not supported: `{lowered}` is a GGML KV cache quantizer, and \
             mlxcel stores its KV cache in MLX arrays with no numerically equivalent \
             representation. `f16` is the one spelling the two vocabularies share, and it is \
             accepted.\n{ALTERNATIVES}"
        );
    }
    if GGML_KV_FLOAT_TYPES.contains(&lowered.as_str()) {
        return format!(
            "{option} {value} is not supported: `{lowered}` is an unquantized GGML float type, \
             and mlxcel's KV cache stores f16 or one of its own quantized modes; there is no f32 \
             or bf16 KV storage to select. `f16` is the one spelling the two vocabularies share, \
             and it is accepted.\n{ALTERNATIVES}"
        );
    }
    format!(
        "{option} {value} is not a recognised KV cache type.\n\
         Use instead: f16 (alias fp16), int8, fp16+turbo4 (alias turbo4-asym), fp16+turbo3 \
         (alias turbo3-asym), turbo4 (alias turbo4-sym), or turbo4-delegated. See \
         docs/turbo-kv-cache.md."
    )
}

/// Map a (K-mode, V-mode) pair to the combined `KVCacheMode`.
///
/// Not all combinations are supported. Returns an error with a human-readable
/// message when the pair is unsupported.
fn map_kv_modes_to_cache_mode(k: KVCacheMode, v: KVCacheMode) -> Result<KVCacheMode, String> {
    use KVCacheMode::{Fp16, Int8, Turbo3Asym, Turbo4, Turbo4Asym, Turbo4Delegated};
    match (k, v) {
        (Fp16, Fp16) => Ok(Fp16),
        (Int8, Int8) => Ok(Int8),
        // Asymmetric: FP16 K + Turbo4 V → Turbo4Asym (covers turbo4-asym input on V side)
        (Fp16, Turbo4Asym) | (Fp16, Turbo4) => Ok(Turbo4Asym),
        // Symmetric: Turbo4 K + Turbo4 V → Turbo4 (allowlist-gated inside mlxcel-core)
        (Turbo4, Turbo4) => Ok(Turbo4),
        // Delegated hot/cold: FP16 K + Turbo4Delegated V → Turbo4Delegated
        (Fp16, Turbo4Delegated) => Ok(Turbo4Delegated),
        // Asymmetric 3-bit: FP16 K + Turbo3 V → Turbo3Asym. Symmetric Turbo3
        // is intentionally not offered (see KVCacheMode::Turbo3Asym docs and
        // the help text for `--cache-type-v`). The legacy
        // `--kv-cache-mode fp16+turbo3` shorthand routed through
        // `KVCacheMode::from_str` already accepted this pair, so the split
        // flags must accept it too for parity.
        (Fp16, Turbo3Asym) => Ok(Turbo3Asym),
        // Anything else is unsupported.
        (k, v) => Err(format!(
            "unsupported --cache-type-k={k} / --cache-type-v={v} combination; \
             supported pairs:\n  \
             fp16   / fp16              -> fp16 (default)\n  \
             int8   / int8              -> int8\n  \
             fp16   / turbo4            -> fp16+turbo4 (Turbo4Asym)\n  \
             fp16   / turbo4-asym       -> fp16+turbo4 (Turbo4Asym)\n  \
             turbo4 / turbo4            -> turbo4 (symmetric, allowlist-gated)\n  \
             fp16   / turbo4-delegated  -> turbo4-delegated\n  \
             fp16   / turbo3            -> fp16+turbo3 (Turbo3Asym)\n  \
             fp16   / turbo3-asym       -> fp16+turbo3 (Turbo3Asym)"
        )),
    }
}

/// Apply `LLAMA_ARG_CACHE_TYPE_K` env var fallback to the raw
/// `--cache-type-k` CLI value.
///
/// Precedence: CLI flag beats env var. When the CLI flag was not provided
/// (value is `None`) and the env var is set, the env var value is applied.
/// When both are present, the CLI value is kept and an INFO log is emitted.
pub fn env_fallback_cache_type_k(value: &mut Option<String>) {
    apply_optional_string_env_fallback(value, "LLAMA_ARG_CACHE_TYPE_K", "cache-type-k");
}

/// Apply `LLAMA_ARG_CACHE_TYPE_V` env var fallback to the raw
/// `--cache-type-v` CLI value.
///
/// Precedence: CLI flag beats env var. When the CLI flag was not provided
/// (value is `None`) and the env var is set, the env var value is applied.
/// When both are present, the CLI value is kept and an INFO log is emitted.
pub fn env_fallback_cache_type_v(value: &mut Option<String>) {
    apply_optional_string_env_fallback(value, "LLAMA_ARG_CACHE_TYPE_V", "cache-type-v");
}

/// Shared helper: if `value` is `None` and the named env var is set, fill
/// `value` from the env var. If `value` is `Some` (CLI was set) and the env
/// var is also present and differs, log an INFO and keep the CLI value.
/// When `value` already equals the env var string (because clap's `env = "..."`
/// injected it), no conflict log is emitted since there is no real conflict.
fn apply_optional_string_env_fallback(value: &mut Option<String>, key: &str, flag_name: &str) {
    if value.is_some() {
        if let Ok(raw) = std::env::var(key) {
            let trimmed = raw.trim();
            // Only log a conflict when the values genuinely differ. When
            // clap injected the env var as the CLI value they are equal
            // and logging would be misleading.
            if value.as_deref() != Some(trimmed) {
                tracing::info!(
                    "{key} env var is set but --{flag_name} CLI flag takes precedence; ignoring {key}"
                );
            }
        }
        return;
    }
    if let Ok(raw) = std::env::var(key) {
        let trimmed = raw.trim().to_string();
        if !trimmed.is_empty() {
            *value = Some(trimmed);
        }
    }
}

/// Second stage of KV-cache mode resolution: turn the mode the operator asked
/// for into the mode this particular model can actually run.
///
/// [`resolve_kv_cache_mode`] turns flags into a [`KVCacheMode`] without knowing
/// which model is being loaded. Two rules need the model, and both were
/// documented long before anything performed them (issue #1350):
///
/// * The MLA-latent families (`glm4_moe_lite`, `deepseek_v3`, `kimi_linear`,
///   `longcat_flash_ngram`) pack a `(kv_latent, k_pe)` pair into one cache, so
///   no quantized mode is calibrated for what they store. Symmetric `turbo4`
///   made the K quantizer read past its sign vectors and every generated token
///   came out `!`; the asymmetric modes quietly compressed the RoPE key stream
///   instead of a value. They resolve to `fp16`.
/// * `turbo4` on a family outside `ALLOWED_SYMMETRIC_TURBO_FAMILIES` resolves
///   to `fp16+turbo4`, which is the fallback the CLI banner has always
///   described.
///
/// Every generation entry point (CLI `generate`, the chat REPL, the memory
/// preflight, `mlxcel-server`, and `bench_decode`) calls this so the mode that
/// is announced is the mode the caches are built with. Returns the effective
/// mode plus, when it differs from the request, one line for the operator.
///
/// `model_path` is a resolved local model directory. A missing or unparsable
/// `config.json` yields `(requested, None)`: this stage must not invent a
/// downgrade from a model it could not identify, and the unconditional head-dim
/// assertions in `cache::turbo::quant` remain the backstop for that case.
///
/// Used by: `commands::generate`, `commands::chat`, `server::cli_input`,
/// `bin/bench_decode`.
#[must_use]
pub fn resolve_effective_kv_cache_mode(
    requested: KVCacheMode,
    model_path: &Path,
) -> (KVCacheMode, Option<String>) {
    let Some(model_type) = read_config_model_type(model_path) else {
        return (requested, None);
    };
    resolve_kv_cache_mode_for_model(requested, &model_type)
}

/// Apply [`resolve_effective_kv_cache_mode`] and report the substitution once
/// on stderr.
///
/// stderr rather than stdout so a downgrade notice never lands in piped
/// generation output, and `tracing::warn!` as well so it is captured in the
/// server's structured log.
///
/// Used by: the CLI generation entry points; the server logs through
/// `tracing` alone and calls [`resolve_effective_kv_cache_mode`] directly.
#[must_use]
pub fn resolve_and_announce_kv_cache_mode(
    requested: KVCacheMode,
    model_path: &Path,
) -> KVCacheMode {
    let (effective, warning) = resolve_effective_kv_cache_mode(requested, model_path);
    if let Some(warning) = warning {
        eprintln!("{warning}");
        tracing::warn!(
            requested = %requested,
            effective = %effective,
            "KV cache mode substituted for this model family"
        );
    }
    effective
}

/// Read the raw `model_type` string from a model directory's `config.json`.
///
/// Deliberately the raw string rather than `models::detection::get_model_type`:
/// the allowlist and the latent-family list are both keyed on the `config.json`
/// value, and detection maps several distinct `model_type` values onto one
/// enum variant.
fn read_config_model_type(model_path: &Path) -> Option<String> {
    let content = std::fs::read_to_string(model_path.join("config.json")).ok()?;
    let config = serde_json::from_str::<serde_json::Value>(&content).ok()?;
    config
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}

#[cfg(test)]
#[path = "turbo_args_tests.rs"]
mod tests;
