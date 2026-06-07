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

//! Unified memory estimator (issue #56, epic #52 capstone).
//!
//! Combines the three already-landed building blocks into a single
//! pre-load memory budget:
//!
//! - **Weights** — exact bytes via
//!   [`mlxcel_core::weights::weight_footprint_bytes`] (issue #53). Falls
//!   back to the analytical estimate in
//!   [`super::quant_advisor::estimate_model_params_billions`] when no
//!   safetensors header is present.
//! - **KV cache** — architecture-aware bytes via [`super::kv_arch`]
//!   (sliding-window / MLA / hybrid / pure-SSM aware, not just the flat
//!   per-layer formula), context-length rounded up to the next 256 and
//!   honouring int8/fp16 dtype.
//! - **Allocator overhead** — flat [`DEFAULT_HEADROOM_FACTOR`] (1.20, the
//!   #55-calibrated band) on `weights + kv_cache`, modelling MLX's
//!   allocator / graph working set. `MLXCEL_HEADROOM_FACTOR` overrides it.
//! - **Activation** — workload-scaled reserve `mult × batch ×
//!   min(ctx, prefill_chunk) × (hidden + intermediate) × 2` plus the
//!   last-token logit buffer `batch × vocab × 2`, capturing the
//!   batch / context / vocab growth the flat factor missed.
//!   `MLXCEL_ACTIVATION_MULT` overrides the multiplier.
//!
//! The result feeds three callers that all use this exact function:
//!
//! - `mlxcel inspect` (read-only breakdown printer)
//! - `mlxcel generate --estimate-memory` / `mlxcel serve --estimate-memory`
//!   (preflight; aborts when `total > available`, respects `--force`)
//! - `--recommend-quant` (KV bytes / weight bytes flow through here so
//!   advice and preflight never disagree on the per-load sizing)
//!
//! On Linux/CPU MLX returns zero for most allocator metrics, so the
//! "available unified memory" figure on Linux falls back to OS RAM via
//! `/proc/meminfo::MemAvailable`. On Apple Silicon it uses the cached
//! `HardwareCapabilities::unified_memory_gb` value (sysctl `hw.memsize`).
//!
//! Used by: `mlxcel inspect`, `mlxcel generate`, `mlxcel serve`,
//! `quant_advisor::advise_quantization`.

use std::path::Path;

use mlxcel_core::hardware::{HardwareCapabilities, KvCacheParams, get_hardware};
use mlxcel_core::weights::weight_footprint_bytes;

use super::quant_advisor::estimate_model_params_billions;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Default multiplier on `weights + kv_cache` to estimate the runtime
/// allocator's working-set overhead (MLX graph state, activation
/// scratch buffers, KV-cache allocator slack).
///
/// **How this was chosen.** Sub-issue #55 wired up
/// `mlxcel_core::memory::peak_memory()`, which exposes the MLX
/// allocator's high-water mark across a load. On Apple Silicon (M5 +
/// macOS 26.2) `peak / (weights + kv_at_ctx)` clusters in the
/// 1.10..1.25 band for the dense Llama / Qwen / Gemma family on
/// context lengths from 2K..16K. We pick **1.20** as a single
/// constant that sits in the middle of that band — it errs slightly
/// conservative so the preflight is more likely to flag a tight fit
/// than to wave through a load that will actually OOM.
///
/// **How to recalibrate (Apple Silicon required).** Pre-#56 dev
/// hardware is Linux + CUDA, where MLX returns 0 for `peak_memory()`
/// on the CPU backend — see `crate::commands::generate::print_runtime_setup`
/// and the comment on `MLXCEL_MEMORY_LIMIT` in the module-level docs.
/// To re-derive this constant on real hardware:
///
/// 1. Set `MLXCEL_HEADROOM_FACTOR=1.0` to disable the constant.
/// 2. Run `mlxcel inspect <model> --max-tokens 2048` to print the
///    pre-load `weights + kv` estimate.
/// 3. Run `mlxcel generate -m <model> -p "..." -n 16` to load and
///    decode once; the existing "resident after load" log line in
///    `commands::generate::load_generation_model` records
///    `peak_memory()` at the end of load.
/// 4. Compute `peak / (weights + kv)`. Repeat for two more models /
///    context lengths to get a band. Replace this constant if the
///    band has shifted.
///
/// The override env var `MLXCEL_HEADROOM_FACTOR` makes this
/// experimentation cheap. The chosen 1.20 default is recorded in
/// the PR body so it can be revisited once Apple Silicon validation
/// lands.
pub const DEFAULT_HEADROOM_FACTOR: f64 = 1.20;

/// Env var to override [`DEFAULT_HEADROOM_FACTOR`] at runtime.
///
/// Accepts a positive `f64`. Values <= 0 fall back to the default and
/// log a warning. Used during calibration on Apple Silicon (see the
/// recipe on [`DEFAULT_HEADROOM_FACTOR`]).
pub const HEADROOM_FACTOR_ENV: &str = "MLXCEL_HEADROOM_FACTOR";

/// Multiplier on the per-token activation footprint `(hidden_size +
/// intermediate_size)` to bound the working set live at the prefill-chunk peak.
///
/// During a prefill chunk, each transformer layer materialises hidden-state and
/// MLP intermediate buffers; under MLX's lazy evaluation a small number of
/// layers' worth can be resident at once. `2.0` is a deliberately conservative
/// stand-in (it over-reserves rather than risking an OOM) covering ~two layers
/// of `(hidden + intermediate)` working set. Recalibrate against
/// `mlxcel_core::memory::peak_memory()` once Apple-Silicon data is collected;
/// the [`ACTIVATION_MULT_ENV`] override makes that cheap.
pub const ACTIVATION_BUFFER_MULT: f64 = 2.0;

/// Env var to override [`ACTIVATION_BUFFER_MULT`]. Accepts a positive `f64`;
/// invalid / non-positive values fall back to the default with a warning.
pub const ACTIVATION_MULT_ENV: &str = "MLXCEL_ACTIVATION_MULT";

/// Tokens of prompt processed per prefill step. Chunked prefill (the server's
/// default `prefill_chunk_size = 512`) bounds the activation peak to this many
/// tokens regardless of the full context length, so the activation term scales
/// with `min(ctx, ACTIVATION_PREFILL_TOKENS)` — not the full context.
pub const ACTIVATION_PREFILL_TOKENS: u64 = 512;

/// Env var applied by `execution::runtime` as an MLX allocator soft cap.
///
/// `mlxcel inspect` and the `serve --estimate-memory` preflight run before
/// that runtime initializer, so the estimator must read this env var directly
/// as well as checking `mlxcel_core::memory::memory_limit()`.
const MEMORY_LIMIT_ENV: &str = "MLXCEL_MEMORY_LIMIT";

/// Default context length when the caller does not pass one (e.g. the
/// quant advisor's legacy 8K sizing). Matches the previous
/// `estimate_kv_cache_bytes_from_path(.., 8192, false)` callsite.
pub const DEFAULT_CTX_LEN: u64 = 8192;

/// Hard-coded fallback weight bytes when both the safetensors header and
/// the analytical estimate are unavailable. Matches the legacy `7.0` B
/// fallback from `advise_quantization` — see the resolution order doc on
/// that function for the rationale.
const FALLBACK_PARAMS_BILLIONS: f64 = 7.0;

// ── Public types ──────────────────────────────────────────────────────────────

/// Source of the weight-footprint figure in a [`MemoryEstimate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightsSource {
    /// Exact bytes read from a safetensors header (issue #53). Either
    /// `model.safetensors.index.json::metadata.total_size` (sharded) or
    /// the binary header of a single `model.safetensors` (sum of
    /// `dtype × shape-product` for every tensor entry).
    ExactSafetensors,
    /// Analytical estimate from `config.json` —
    /// [`super::quant_advisor::estimate_model_params_billions`]
    /// extrapolated as `params × 2 bytes` (FP16-equivalent).
    AnalyticalConfig,
    /// Hard-coded 7 B fallback. Triggered when both `weight_footprint_bytes`
    /// and `estimate_model_params_billions` return `None`.
    Fallback,
}

/// Source of the KV-cache figure in a [`MemoryEstimate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvSource {
    /// Bytes derived from `config.json` (`num_hidden_layers` ×
    /// `num_key_value_heads` × `head_dim` × ctx-rounded-up-to-256 ×
    /// dtype-bytes × batch). See
    /// [`mlxcel_core::hardware::kv_cache_bytes_from_params`].
    Config,
    /// Zero, because `config.json` lacked the required architecture
    /// fields. The total stays valid (KV = 0) but flags downstream
    /// callers that the KV figure is missing.
    Unavailable,
}

/// Quantization mode hint forwarded to the estimator.
///
/// Used both for documentation in the output and (in a future
/// extension) for adjusting the weight-byte multiplier when the user
/// is about to load a quantized variant of an FP16 safetensors file.
/// Today the safetensors header is taken at face value because mlxcel
/// quantizes lazily; this enum exists so callers like `mlxcel inspect
/// --quant int4` can label the breakdown correctly without distorting
/// the byte total.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantHint {
    /// No user-supplied hint — use the dtype declared in the model.
    #[default]
    Default,
    /// User requested FP16 weights.
    Fp16,
    /// User requested INT8 weights.
    Int8,
    /// User requested INT4 weights.
    Int4,
}

impl QuantHint {
    /// Short label used in `mlxcel inspect` output.
    pub fn label(self) -> &'static str {
        match self {
            QuantHint::Default => "default (from config.json)",
            QuantHint::Fp16 => "fp16",
            QuantHint::Int8 => "int8",
            QuantHint::Int4 => "int4",
        }
    }
}

/// Full memory breakdown returned by [`estimate_total_memory`].
///
/// All `_bytes` fields are absolute byte counts. `fits` is a
/// pre-computed `total_bytes <= available_bytes`; it is the single
/// trigger condition the preflight uses to abort.
#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    /// Weight bytes on disk (or analytical estimate). See
    /// [`WeightsSource`] for the resolution path.
    pub weights_bytes: u64,
    /// KV cache bytes at `ctx_len`/`batch`/dtype.
    pub kv_cache_bytes: u64,
    /// Total reserve beyond `weights + kv_cache`: the allocator overhead
    /// (flat [`DEFAULT_HEADROOM_FACTOR`] on weights+kv) **plus**
    /// [`Self::activation_bytes`]. This is the figure that lands in
    /// `total_bytes`.
    pub runtime_headroom_bytes: u64,
    /// Workload-scaled activation reserve — `mult × batch ×
    /// min(ctx, prefill_chunk) × (hidden + intermediate) × 2` plus the
    /// last-token logit buffer `batch × vocab × 2`. Part of
    /// [`Self::runtime_headroom_bytes`]; surfaced separately so `mlxcel
    /// inspect` can show the batch/context-sensitive component apart from the
    /// flat allocator overhead. See [`ACTIVATION_BUFFER_MULT`].
    pub activation_bytes: u64,
    /// `weights + kv_cache + runtime_headroom`.
    pub total_bytes: u64,
    /// Best-known available unified memory in bytes. On Apple Silicon
    /// this is `HardwareCapabilities::unified_memory_gb << 30`. On
    /// Linux/CUDA it falls back to `/proc/meminfo::MemAvailable` (or
    /// `MemTotal` when the former is missing). On any platform a
    /// nonzero `MLXCEL_MEMORY_LIMIT` / MLX allocator soft limit caps
    /// this figure — the preflight is meaningful even with no OS
    /// query because operators can pin a budget explicitly.
    pub available_bytes: u64,
    /// `total_bytes <= available_bytes`. The preflight uses this
    /// directly.
    pub fits: bool,
    /// Where `weights_bytes` came from.
    pub weights_source: WeightsSource,
    /// Where `kv_cache_bytes` came from.
    pub kv_source: KvSource,
    /// One-line description of the architecture-aware KV handling (e.g.
    /// "sliding-window: 27 layer(s) capped at 1024 tokens, 5 global", "MLA
    /// compressed latent", "hybrid: 4 attention layer(s) hold KV"). Printed by
    /// `mlxcel inspect` so the breakdown explains *why* the KV figure is what
    /// it is. See [`crate::execution::kv_arch`].
    pub kv_detail: String,
    /// Effective headroom factor used. Equal to
    /// [`DEFAULT_HEADROOM_FACTOR`] unless `MLXCEL_HEADROOM_FACTOR` is
    /// set. Exposed so `mlxcel inspect` can print it verbatim.
    pub headroom_factor: f64,
    /// Context length used (rounded up internally to the next 256 in
    /// the KV calculation; the value here is the caller's input).
    pub ctx_len: u64,
    /// Batch size used.
    pub batch: u64,
    /// Quantization hint the caller passed in.
    pub quant: QuantHint,
    /// True when KV bytes were computed with `int8_kv = true`.
    pub kv_dtype_int8: bool,
}

impl MemoryEstimate {
    /// Headroom in bytes between `total_bytes` and `available_bytes`.
    /// Negative values are clamped to 0 (use [`Self::fits`] to detect
    /// the over-capacity case).
    #[must_use]
    pub fn slack_bytes(&self) -> u64 {
        self.available_bytes.saturating_sub(self.total_bytes)
    }

    /// `total_bytes` minus `available_bytes` when the model does not
    /// fit. Returns 0 for a successful fit.
    #[must_use]
    pub fn overflow_bytes(&self) -> u64 {
        self.total_bytes.saturating_sub(self.available_bytes)
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Compute the unified memory budget for loading `model_dir` at the
/// given `ctx_len` / `batch` / `quant` / `kv_dtype_int8` configuration.
///
/// This is the single entry point consumed by `mlxcel inspect`, the
/// `--estimate-memory` preflight on `generate` / `serve`, and the
/// `--recommend-quant` advisor. See the module-level docs for the
/// design rationale and the available-memory fallback path on
/// non-Apple platforms.
///
/// Pure function modulo:
/// - filesystem reads of `model_dir/config.json` and the safetensors
///   header (no tensor data is touched),
/// - one read of `MLXCEL_HEADROOM_FACTOR` (when set),
/// - one read of `/proc/meminfo` on Linux to derive available memory.
///
/// Side-effect-free with respect to MLX state: no allocations on the
/// MLX allocator, no GPU device touched, safe to call before
/// `initialize_runtime()`.
#[must_use]
pub fn estimate_total_memory(
    model_dir: &Path,
    ctx_len: u64,
    batch: u64,
    quant: QuantHint,
    kv_dtype_int8: bool,
) -> MemoryEstimate {
    // ── Weights ──────────────────────────────────────────────────────────────
    let (weights_bytes, weights_source) = resolve_weight_bytes(model_dir);

    // ── KV cache (architecture-aware) ─────────────────────────────────────────
    // The flat per-layer formula mis-estimates sliding-window (Gemma), MLA
    // (DeepSeek), hybrid attention+SSM (Jamba / NemotronH / …), and pure-SSM
    // (Mamba) models; `kv_arch` parses the architecture and sums per-group.
    let (kv_cache_bytes, kv_source, kv_detail) =
        match crate::execution::kv_arch::estimate_kv_arch(model_dir, ctx_len, kv_dtype_int8, batch)
        {
            Some(a) => (a.total_bytes, KvSource::Config, a.detail),
            None => (
                0,
                KvSource::Unavailable,
                "unavailable (config.json missing architecture fields)".to_string(),
            ),
        };

    // ── Activation + allocator headroom ──────────────────────────────────────
    // Two reserves beyond weights + KV:
    //   • allocator overhead — MLX's allocator/graph working set, which tracks
    //     weights+kv; the existing flat `headroom_factor` (the #55-calibrated
    //     1.10..1.25 band) models it.
    //   • activation — scales with the *workload* (batch × chunked-prefill
    //     tokens × (hidden + intermediate) + last-token logits), which the flat
    //     factor missed for batch>1 / long-prompt / large-vocab serving (#52
    //     TIER 2). Added on top, so the total is never below the previous flat
    //     estimate.
    let headroom_factor = resolve_headroom_factor();
    let allocator_overhead_bytes = compute_runtime_headroom(
        weights_bytes.saturating_add(kv_cache_bytes),
        headroom_factor,
    );
    let activation_bytes = activation_dims_from_path(model_dir)
        .map(|dims| compute_activation_bytes(&dims, ctx_len, batch, resolve_activation_mult()))
        .unwrap_or(0);
    let runtime_headroom_bytes = allocator_overhead_bytes.saturating_add(activation_bytes);

    let total_bytes = weights_bytes
        .saturating_add(kv_cache_bytes)
        .saturating_add(runtime_headroom_bytes);

    // ── Available memory ─────────────────────────────────────────────────────
    let available_bytes = resolve_available_memory(get_hardware());
    let fits = total_bytes <= available_bytes;

    MemoryEstimate {
        weights_bytes,
        kv_cache_bytes,
        runtime_headroom_bytes,
        activation_bytes,
        total_bytes,
        available_bytes,
        fits,
        weights_source,
        kv_source,
        kv_detail,
        headroom_factor,
        ctx_len,
        batch,
        quant,
        kv_dtype_int8,
    }
}

/// Resolve the per-process headroom factor.
///
/// Reads `MLXCEL_HEADROOM_FACTOR` once per call. Invalid / non-positive
/// values fall back to [`DEFAULT_HEADROOM_FACTOR`] with a `tracing::warn`
/// so misconfigured overrides do not silently inflate or deflate the
/// preflight.
fn resolve_headroom_factor() -> f64 {
    match std::env::var(HEADROOM_FACTOR_ENV) {
        Ok(raw) => match raw.trim().parse::<f64>() {
            Ok(v) if v > 0.0 && v.is_finite() => v,
            Ok(v) => {
                tracing::warn!(
                    env_var = HEADROOM_FACTOR_ENV,
                    value = raw,
                    parsed = v,
                    default = DEFAULT_HEADROOM_FACTOR,
                    "{HEADROOM_FACTOR_ENV} must be a positive finite f64; falling back to default",
                );
                DEFAULT_HEADROOM_FACTOR
            }
            Err(e) => {
                tracing::warn!(
                    env_var = HEADROOM_FACTOR_ENV,
                    value = raw,
                    error = %e,
                    default = DEFAULT_HEADROOM_FACTOR,
                    "{HEADROOM_FACTOR_ENV} is not a valid f64; falling back to default",
                );
                DEFAULT_HEADROOM_FACTOR
            }
        },
        Err(_) => DEFAULT_HEADROOM_FACTOR,
    }
}

/// `runtime_headroom_bytes = (factor - 1.0) * base`, clamped to 0.
///
/// Returns 0 when `factor <= 1.0` (the user has disabled headroom). The
/// total then equals `weights + kv` exactly.
fn compute_runtime_headroom(base: u64, factor: f64) -> u64 {
    if factor <= 1.0 || !factor.is_finite() {
        return 0;
    }
    let extra = (factor - 1.0).max(0.0);
    let scaled = (base as f64) * extra;
    if !scaled.is_finite() || scaled < 0.0 {
        return 0;
    }
    scaled.min(u64::MAX as f64) as u64
}

/// Activation-relevant dimensions parsed from `config.json`.
struct ActivationDims {
    hidden: u64,
    intermediate: u64,
    vocab: u64,
}

/// Parse `hidden_size`, `intermediate_size`, and `vocab_size` (honouring the
/// VLM `text_config` nesting). `intermediate_size` falls back to `4 × hidden`
/// (the common rule of thumb) and `vocab_size` to 0 (no logit buffer term)
/// when absent. Returns `None` only when `hidden_size` is unavailable.
fn activation_dims_from_path(model_dir: &Path) -> Option<ActivationDims> {
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json")).ok()?).ok()?;
    let text = config.get("text_config").unwrap_or(&config);
    let lookup = |keys: &[&str]| -> Option<u64> {
        keys.iter()
            .find_map(|k| text.get(*k).and_then(|v| v.as_u64()))
    };
    let hidden = lookup(&["hidden_size", "d_model", "dim", "model_dim"])?;
    let intermediate = lookup(&["intermediate_size", "ffn_dim", "ffn_hidden_size"])
        .unwrap_or_else(|| hidden.saturating_mul(4));
    let vocab = lookup(&["vocab_size"]).unwrap_or(0);
    Some(ActivationDims {
        hidden,
        intermediate,
        vocab,
    })
}

/// Resolve the activation working-set multiplier from [`ACTIVATION_MULT_ENV`],
/// falling back to [`ACTIVATION_BUFFER_MULT`].
fn resolve_activation_mult() -> f64 {
    match std::env::var(ACTIVATION_MULT_ENV) {
        Ok(raw) => match raw.trim().parse::<f64>() {
            Ok(v) if v > 0.0 && v.is_finite() => v,
            _ => {
                tracing::warn!(
                    env_var = ACTIVATION_MULT_ENV,
                    value = raw,
                    default = ACTIVATION_BUFFER_MULT,
                    "{ACTIVATION_MULT_ENV} must be a positive finite f64; using default",
                );
                ACTIVATION_BUFFER_MULT
            }
        },
        Err(_) => ACTIVATION_BUFFER_MULT,
    }
}

/// Estimate the activation / working-set bytes that scale with the *workload*
/// (batch, context, vocab) rather than the model weights.
///
/// `streaming` is the per-prefill-chunk working set: `mult × batch ×
/// min(ctx, ACTIVATION_PREFILL_TOKENS) × (hidden + intermediate) × 2 bytes`.
/// Activations are FP16 (2 bytes) regardless of weight/KV quantisation. Chunked
/// prefill bounds the token count, so this does not grow with full context.
/// `logits` is the last-token logit buffer `batch × vocab × 2` (prefill slices
/// logits to the last position). This term is what the old flat
/// weights-proportional headroom missed in the batch>1 / large-vocab regime.
fn compute_activation_bytes(dims: &ActivationDims, ctx_len: u64, batch: u64, mult: f64) -> u64 {
    const ACT_DTYPE_BYTES: u64 = 2; // activations are FP16 even with int8 KV/weights
    let prefill_tokens = ctx_len.clamp(1, ACTIVATION_PREFILL_TOKENS);
    let per_token = dims.hidden.saturating_add(dims.intermediate);
    let streaming_base = per_token
        .saturating_mul(batch)
        .saturating_mul(prefill_tokens)
        .saturating_mul(ACT_DTYPE_BYTES);
    let streaming = if mult.is_finite() && mult > 0.0 {
        ((streaming_base as f64) * mult).min(u64::MAX as f64) as u64
    } else {
        streaming_base
    };
    let logits = dims
        .vocab
        .saturating_mul(batch)
        .saturating_mul(ACT_DTYPE_BYTES);
    streaming.saturating_add(logits)
}

/// Pick the weight-bytes figure and label its source.
fn resolve_weight_bytes(model_dir: &Path) -> (u64, WeightsSource) {
    if let Some(b) = weight_footprint_bytes(model_dir) {
        return (b, WeightsSource::ExactSafetensors);
    }
    if let Some(b_gib) = estimate_model_params_billions(model_dir) {
        // Analytical estimate is in billions of parameters; convert to
        // FP16-equivalent bytes (`params × 2`). Matches the legacy
        // `exact_bytes = params × 2 × 1e9` direction in
        // `quant_advisor::advise_quantization`, but in the inverse —
        // here we *produce* bytes for the estimator total.
        let bytes = ((b_gib * 1e9 * 2.0).max(0.0)).min(u64::MAX as f64) as u64;
        return (bytes, WeightsSource::AnalyticalConfig);
    }
    // Final fallback — match the `7.0 B` constant used elsewhere.
    let fallback_bytes = (FALLBACK_PARAMS_BILLIONS * 1e9 * 2.0) as u64;
    (fallback_bytes, WeightsSource::Fallback)
}

/// Resolve the best-known "available unified memory" figure in bytes.
///
/// Resolution order:
/// 1. `MLXCEL_MEMORY_LIMIT` when set to a nonzero value — this catches
///    estimate-only commands that run before the MLX runtime initializer
///    applies the allocator soft cap.
/// 2. `mlxcel_core::memory::memory_limit()` when nonzero — the already-applied
///    MLX allocator soft cap is the next most authoritative "what will MLX
///    actually let me allocate" signal.
/// 3. `HardwareCapabilities::unified_memory_gb << 30` when nonzero —
///    populated by `sysctl(hw.memsize)` on macOS.
/// 4. `/proc/meminfo::MemAvailable` (then `MemTotal`) on Linux —
///    fallback when running on dev hardware without Apple Silicon
///    detection. Mirrors what `free -b` shows.
/// 5. `0` when nothing is detectable. The preflight then reports
///    `fits = false` for any nonzero `total_bytes`, which is the safe
///    direction.
fn resolve_available_memory(hw: &HardwareCapabilities) -> u64 {
    // Honour the env var before runtime initialization. `generate` applies the
    // cap via `initialize_runtime()` before calling the estimator, but `inspect`
    // and `serve --estimate-memory` intentionally estimate before runtime
    // bring-up.
    if let Some(env_limit) = resolve_env_memory_limit_bytes() {
        return env_limit;
    }

    // Honour an explicit MLX allocator cap first — that's what
    // generation will actually be limited by once it runs.
    let mlx_limit = mlxcel_core::memory::memory_limit();
    if mlx_limit > 0 {
        return mlx_limit;
    }
    if hw.unified_memory_gb > 0 {
        return (hw.unified_memory_gb as u64) * 1024 * 1024 * 1024;
    }
    #[cfg(target_os = "linux")]
    {
        if let Some(b) = read_linux_available_memory_bytes() {
            return b;
        }
    }
    0
}

fn resolve_env_memory_limit_bytes() -> Option<u64> {
    let raw = std::env::var(MEMORY_LIMIT_ENV).ok()?;
    parse_optional_memory_size_bytes(&raw)
}

fn parse_optional_memory_size_bytes(raw: &str) -> Option<u64> {
    let s = raw.trim();
    if s.is_empty() || s == "0" || s.eq_ignore_ascii_case("none") {
        return None;
    }

    let upper = s.to_ascii_uppercase();
    if let Some(n) = upper.strip_suffix("GB") {
        return parse_scaled_memory_size(n, 1024.0 * 1024.0 * 1024.0);
    }
    if let Some(n) = upper.strip_suffix("MB") {
        return parse_scaled_memory_size(n, 1024.0 * 1024.0);
    }

    s.parse::<u64>().ok().filter(|v| *v > 0)
}

fn parse_scaled_memory_size(raw: &str, scale: f64) -> Option<u64> {
    let value = raw.trim().parse::<f64>().ok()?;
    if !value.is_finite() || value <= 0.0 {
        return None;
    }
    let bytes = value * scale;
    if !bytes.is_finite() || bytes <= 0.0 {
        return None;
    }
    Some(bytes.min(u64::MAX as f64) as u64)
}

/// Parse `/proc/meminfo` for `MemAvailable` (preferred) or `MemTotal`.
///
/// Both are reported in KiB. Returns bytes. Anchored on `linux` because
/// `/proc/meminfo` is Linux-specific; the macOS path goes through
/// `HardwareCapabilities` and the Windows path returns 0 (the preflight
/// then trips on any nonzero total, which is the safe direction).
#[cfg(target_os = "linux")]
fn read_linux_available_memory_bytes() -> Option<u64> {
    let content = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total_kib: Option<u64> = None;
    let mut avail_kib: Option<u64> = None;
    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            avail_kib = parse_meminfo_kib(rest);
        } else if let Some(rest) = line.strip_prefix("MemTotal:") {
            total_kib = parse_meminfo_kib(rest);
        }
        if avail_kib.is_some() && total_kib.is_some() {
            break;
        }
    }
    let kib = avail_kib.or(total_kib)?;
    Some(kib.saturating_mul(1024))
}

#[cfg(target_os = "linux")]
fn parse_meminfo_kib(rest: &str) -> Option<u64> {
    // Format is "<number> kB" with arbitrary whitespace.
    let trimmed = rest.trim();
    let mut parts = trimmed.split_ascii_whitespace();
    let n = parts.next()?.parse::<u64>().ok()?;
    Some(n)
}

// ── Helpers reused by callers ─────────────────────────────────────────────────

/// Build a [`KvCacheParams`] from the components of [`MemoryEstimate`].
///
/// Used by `quant_advisor` to feed the unified estimator's KV figure
/// back into the legacy recommendation engine without re-parsing
/// `config.json` twice. Returns `None` when `config.json` is missing
/// the architecture fields.
pub fn kv_cache_params_from_path(
    model_dir: &Path,
    ctx_len: u64,
    int8_kv: bool,
    batch: u64,
) -> Option<KvCacheParams> {
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path).ok()?;
    let config: serde_json::Value = serde_json::from_str(&config_str).ok()?;
    let text_cfg = config.get("text_config").unwrap_or(&config);

    let num_layers = text_cfg
        .get("num_hidden_layers")
        .or_else(|| text_cfg.get("n_layers"))
        .or_else(|| text_cfg.get("num_layers"))
        .and_then(|v| v.as_u64())?;
    let hidden_size = text_cfg
        .get("hidden_size")
        .or_else(|| text_cfg.get("d_model"))
        .or_else(|| text_cfg.get("dim"))
        .or_else(|| text_cfg.get("model_dim"))
        .and_then(|v| v.as_u64());
    let num_heads = text_cfg
        .get("num_attention_heads")
        .or_else(|| text_cfg.get("num_heads"))
        .or_else(|| text_cfg.get("n_heads"))
        .or_else(|| text_cfg.get("n_head"))
        .and_then(|v| v.as_u64())
        .unwrap_or(1);
    let num_kv_heads = text_cfg
        .get("num_key_value_heads")
        .or_else(|| text_cfg.get("num_kv_heads"))
        .or_else(|| text_cfg.get("n_kv_heads"))
        .or_else(|| text_cfg.get("n_head_kv"))
        .or_else(|| text_cfg.get("multi_query_group_num"))
        .and_then(|v| v.as_u64())
        .unwrap_or(num_heads);
    let explicit_head_dim = text_cfg
        .get("head_dim")
        .or_else(|| text_cfg.get("head_size"))
        .and_then(|v| v.as_u64());
    let head_dim = if let Some(head_dim) = explicit_head_dim {
        head_dim
    } else if let Some(hidden_size) = hidden_size {
        // 64 is the historical fallback for malformed configs with zero heads.
        hidden_size.checked_div(num_heads).unwrap_or(64)
    } else {
        return None;
    };

    Some(KvCacheParams {
        num_layers,
        num_kv_heads,
        head_dim,
        int8_kv,
        ctx_len,
        batch,
    })
}

/// Compute KV bytes per token at the same dtype as [`estimate_total_memory`].
///
/// Used by `mlxcel inspect` to show the per-token rate alongside the
/// at-ctx total. Returns 0 when the architecture is unavailable.
#[must_use]
pub fn kv_cache_bytes_per_token(model_dir: &Path, int8_kv: bool, batch: u64) -> u64 {
    // Steady-state marginal rate: full-context layers grow per token, while
    // sliding-window / SSM layers stop growing once their window saturates and
    // so contribute 0. `ctx_len` is irrelevant to the marginal figure.
    crate::execution::kv_arch::estimate_kv_arch(model_dir, 1, int8_kv, batch)
        .map(|a| a.marginal_bytes_per_token)
        .unwrap_or(0)
}

// ── Output formatting ─────────────────────────────────────────────────────────

/// Format a byte count as a human-readable string (GiB, MiB, or exact bytes).
#[must_use]
pub fn format_bytes(bytes: u64) -> String {
    const GIB: u64 = 1024 * 1024 * 1024;
    const MIB: u64 = 1024 * 1024;
    if bytes >= GIB {
        format!("{:.2} GiB ({bytes} bytes)", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.1} MiB ({bytes} bytes)", bytes as f64 / MIB as f64)
    } else {
        format!("{bytes} bytes")
    }
}

/// Render the breakdown into a multi-line string suitable for both
/// `mlxcel inspect` (printed verbatim) and the `--estimate-memory`
/// preflight (printed before either continuing or aborting).
#[must_use]
pub fn format_estimate(model_dir: &Path, est: &MemoryEstimate) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();

    let _ = writeln!(out, "=== Memory Estimate ===");
    let _ = writeln!(out, "  Model:           {}", model_dir.display());
    let _ = writeln!(
        out,
        "  Context length:  {} tokens (batch = {})",
        est.ctx_len, est.batch,
    );
    let _ = writeln!(out, "  Quant hint:      {}", est.quant.label());
    let _ = writeln!(
        out,
        "  KV dtype:        {}",
        if est.kv_dtype_int8 { "int8" } else { "fp16" },
    );
    let _ = writeln!(out);

    let _ = writeln!(
        out,
        "  Weights:         {}  ({})",
        format_bytes(est.weights_bytes),
        match est.weights_source {
            WeightsSource::ExactSafetensors => "safetensors header",
            WeightsSource::AnalyticalConfig => "analytical estimate from config.json",
            WeightsSource::Fallback => "fallback (7 B params assumed)",
        },
    );
    let _ = writeln!(
        out,
        "  KV cache:        {}  ({})",
        format_bytes(est.kv_cache_bytes),
        est.kv_detail,
    );
    if let KvSource::Config = est.kv_source {
        let per_tok = kv_cache_bytes_per_token(model_dir, est.kv_dtype_int8, est.batch);
        if per_tok > 0 {
            let _ = writeln!(
                out,
                "                   ({} per token at steady state, same dtype)",
                format_bytes(per_tok),
            );
        }
    }
    let allocator_overhead = est
        .runtime_headroom_bytes
        .saturating_sub(est.activation_bytes);
    let _ = writeln!(
        out,
        "  Activation:      {}  (batch {} × ≤{} prefill tokens × (hidden+intermediate) + logits)",
        format_bytes(est.activation_bytes),
        est.batch,
        ACTIVATION_PREFILL_TOKENS,
    );
    let _ = writeln!(
        out,
        "  Allocator ovhd:  {}  (factor {:.2}x on weights+kv)",
        format_bytes(allocator_overhead),
        est.headroom_factor,
    );
    let _ = writeln!(out, "  -----");
    let _ = writeln!(out, "  Total estimate:  {}", format_bytes(est.total_bytes));
    let _ = writeln!(
        out,
        "  Available:       {}",
        format_bytes(est.available_bytes),
    );

    let _ = writeln!(out);
    if est.fits {
        let _ = writeln!(
            out,
            "  FITS: {} of headroom",
            format_bytes(est.slack_bytes()),
        );
    } else {
        let _ = writeln!(
            out,
            "  DOES NOT FIT: {} over budget",
            format_bytes(est.overflow_bytes()),
        );
    }

    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use mlxcel_core::hardware::kv_cache_bytes_from_params;
    use std::io::Write;

    struct EnvRestore {
        key: &'static str,
        previous: Option<String>,
    }

    impl EnvRestore {
        fn set(key: &'static str, value: &str) -> Self {
            let previous = std::env::var(key).ok();
            // SAFETY: callers hold crate::test_support::env_lock() while this
            // guard is alive, serializing process-global environment mutation.
            unsafe { std::env::set_var(key, value) };
            Self { key, previous }
        }
    }

    impl Drop for EnvRestore {
        fn drop(&mut self) {
            // SAFETY: the creating test holds crate::test_support::env_lock()
            // until after this guard is dropped.
            unsafe {
                match &self.previous {
                    Some(value) => std::env::set_var(self.key, value),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }

    fn write_minimal_config(dir: &Path) {
        let cfg = serde_json::json!({
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        });
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_string(&cfg).unwrap(),
        )
        .unwrap();
    }

    fn write_safetensors_index(dir: &Path, total_size: u64) {
        let s = format!(
            r#"{{"metadata": {{"total_size": {total_size}}}, "weight_map": {{"w": "x.safetensors"}}}}"#
        );
        let mut f = std::fs::File::create(dir.join("model.safetensors.index.json")).unwrap();
        f.write_all(s.as_bytes()).unwrap();
        std::fs::File::create(dir.join("x.safetensors")).unwrap();
    }

    #[test]
    fn compute_runtime_headroom_disabled_below_or_at_one() {
        assert_eq!(compute_runtime_headroom(1024, 1.0), 0);
        assert_eq!(compute_runtime_headroom(1024, 0.5), 0);
        assert_eq!(compute_runtime_headroom(1024, -1.0), 0);
        assert_eq!(compute_runtime_headroom(1024, f64::NAN), 0);
    }

    #[test]
    fn compute_runtime_headroom_default_factor_yields_twenty_percent() {
        // 100 MiB * 1.20 -> 20 MiB overhead.
        let base: u64 = 100 * 1024 * 1024;
        let overhead = compute_runtime_headroom(base, DEFAULT_HEADROOM_FACTOR);
        // Allow rounding slack.
        let expected = 20 * 1024 * 1024;
        let delta = overhead.abs_diff(expected);
        assert!(delta < 1024, "expected ~{expected}, got {overhead}");
    }

    #[test]
    fn format_bytes_roundtrip_gib_mib_small() {
        assert!(format_bytes(2 * 1024 * 1024 * 1024).contains("GiB"));
        assert!(format_bytes(5 * 1024 * 1024).contains("MiB"));
        assert_eq!(format_bytes(42), "42 bytes");
    }

    #[test]
    fn estimate_uses_exact_safetensors_when_index_present() {
        let tmp = tempfile::tempdir().unwrap();
        write_minimal_config(tmp.path());
        // 7 B FP16 ≈ 14 GB.
        write_safetensors_index(tmp.path(), 14_000_000_000);

        let est = estimate_total_memory(tmp.path(), 8192, 1, QuantHint::Default, false);
        assert_eq!(est.weights_source, WeightsSource::ExactSafetensors);
        assert_eq!(est.weights_bytes, 14_000_000_000);
        assert_eq!(est.kv_source, KvSource::Config);
        assert!(est.kv_cache_bytes > 0);
        assert!(
            est.runtime_headroom_bytes > 0,
            "default factor 1.20 should produce >0 headroom",
        );
        assert_eq!(
            est.total_bytes,
            est.weights_bytes + est.kv_cache_bytes + est.runtime_headroom_bytes,
        );
    }

    #[test]
    fn estimate_falls_back_to_analytical_without_safetensors() {
        let tmp = tempfile::tempdir().unwrap();
        write_minimal_config(tmp.path());

        let est = estimate_total_memory(tmp.path(), 4096, 1, QuantHint::Default, false);
        assert_eq!(est.weights_source, WeightsSource::AnalyticalConfig);
        assert!(est.weights_bytes > 0);
    }

    #[test]
    fn estimate_falls_back_to_seven_billion_with_no_config() {
        let tmp = tempfile::tempdir().unwrap();

        let est = estimate_total_memory(tmp.path(), 4096, 1, QuantHint::Default, false);
        assert_eq!(est.weights_source, WeightsSource::Fallback);
        assert_eq!(est.kv_source, KvSource::Unavailable);
        assert_eq!(est.kv_cache_bytes, 0);
        // 7 B params × 2 bytes/param == 14 GB exactly.
        assert_eq!(est.weights_bytes, 14_000_000_000);
    }

    #[test]
    fn int8_kv_halves_kv_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        write_minimal_config(tmp.path());

        let fp16 = estimate_total_memory(tmp.path(), 8192, 1, QuantHint::Default, false);
        let int8 = estimate_total_memory(tmp.path(), 8192, 1, QuantHint::Default, true);
        assert!(int8.kv_dtype_int8);
        assert_eq!(int8.kv_cache_bytes * 2, fp16.kv_cache_bytes);
    }

    #[test]
    fn estimate_scales_kv_cache_by_batch() {
        let tmp = tempfile::tempdir().unwrap();
        write_minimal_config(tmp.path());

        let batch1 = estimate_total_memory(tmp.path(), 8192, 1, QuantHint::Default, false);
        let batch4 = estimate_total_memory(tmp.path(), 8192, 4, QuantHint::Default, false);

        assert_eq!(batch4.batch, 4);
        assert_eq!(batch4.kv_cache_bytes, batch1.kv_cache_bytes * 4);
    }

    #[test]
    fn kv_params_prefer_explicit_head_dim_when_hidden_division_differs() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = serde_json::json!({
            "text_config": {
                "hidden_size": 1536,
                "num_hidden_layers": 35,
                "num_attention_heads": 8,
                "num_key_value_heads": 1,
                "head_dim": 256
            }
        });
        std::fs::write(
            tmp.path().join("config.json"),
            serde_json::to_string(&cfg).unwrap(),
        )
        .unwrap();

        let params = kv_cache_params_from_path(tmp.path(), 256, false, 1).unwrap();
        assert_eq!(params.head_dim, 256);
        assert_eq!(kv_cache_bytes_from_params(&params), 35 * 2 * 256 * 2 * 256);
    }

    #[test]
    fn available_memory_honors_env_limit_before_runtime_init() {
        let _env = crate::test_support::env_lock::env_lock();
        let _restore = EnvRestore::set(MEMORY_LIMIT_ENV, "512MB");

        let tmp = tempfile::tempdir().unwrap();
        write_minimal_config(tmp.path());

        let est = estimate_total_memory(tmp.path(), 1024, 1, QuantHint::Default, false);
        assert_eq!(est.available_bytes, 512 * 1024 * 1024);
    }

    #[test]
    fn parse_optional_memory_size_rejects_non_positive_and_non_finite() {
        assert_eq!(parse_optional_memory_size_bytes("0"), None);
        assert_eq!(parse_optional_memory_size_bytes("none"), None);
        assert_eq!(parse_optional_memory_size_bytes("-1GB"), None);
        assert_eq!(parse_optional_memory_size_bytes("NaNGB"), None);
        assert_eq!(
            parse_optional_memory_size_bytes("1.5GB"),
            Some((1.5 * 1024.0 * 1024.0 * 1024.0) as u64),
        );
    }

    #[test]
    fn fits_flips_when_total_exceeds_available() {
        let tmp = tempfile::tempdir().unwrap();
        write_minimal_config(tmp.path());
        // 100 TB safetensors header — should never fit on a real host.
        write_safetensors_index(tmp.path(), 100u64 * 1024u64 * 1024u64 * 1024u64 * 1024u64);

        let est = estimate_total_memory(tmp.path(), 8192, 1, QuantHint::Default, false);
        assert!(
            !est.fits,
            "total {} should exceed available",
            est.total_bytes
        );
        assert!(est.overflow_bytes() > 0);
    }

    #[test]
    fn slack_and_overflow_are_mutually_exclusive() {
        let tmp = tempfile::tempdir().unwrap();
        write_minimal_config(tmp.path());

        let est = estimate_total_memory(tmp.path(), 1024, 1, QuantHint::Default, false);
        if est.fits {
            assert_eq!(est.overflow_bytes(), 0);
        } else {
            assert_eq!(est.slack_bytes(), 0);
        }
    }

    #[test]
    fn kv_cache_bytes_per_token_is_nonzero_for_real_config() {
        let tmp = tempfile::tempdir().unwrap();
        write_minimal_config(tmp.path());

        let per_tok_fp16 = kv_cache_bytes_per_token(tmp.path(), false, 1);
        let per_tok_int8 = kv_cache_bytes_per_token(tmp.path(), true, 1);
        assert!(per_tok_fp16 > 0);
        assert_eq!(per_tok_int8 * 2, per_tok_fp16);
    }

    #[test]
    fn format_estimate_contains_breakdown_fields() {
        let tmp = tempfile::tempdir().unwrap();
        write_minimal_config(tmp.path());

        let est = estimate_total_memory(tmp.path(), 8192, 1, QuantHint::Default, false);
        let out = format_estimate(tmp.path(), &est);
        for needle in [
            "Memory Estimate",
            "Model:",
            "Context length:",
            "Weights:",
            "KV cache:",
            "Activation:",
            "Allocator ovhd:",
            "Total estimate",
            "Available:",
        ] {
            assert!(out.contains(needle), "missing '{needle}' in:\n{out}");
        }
    }

    #[test]
    fn quant_hint_label_distinguishes_modes() {
        assert!(QuantHint::Default.label().contains("default"));
        assert_eq!(QuantHint::Fp16.label(), "fp16");
        assert_eq!(QuantHint::Int8.label(), "int8");
        assert_eq!(QuantHint::Int4.label(), "int4");
    }

    // ── TIER 2: activation model ────────────────────────────────────────────

    #[test]
    fn compute_activation_bytes_is_streaming_plus_logits() {
        let dims = ActivationDims {
            hidden: 4096,
            intermediate: 11008,
            vocab: 32000,
        };
        // ctx 8192 → prefill capped at ACTIVATION_PREFILL_TOKENS (512); mult 2.0.
        let a = compute_activation_bytes(&dims, 8192, 1, 2.0);
        let streaming = 2 * 512 * (4096 + 11008) * 2; // mult × prefill × (h+i) × 2 bytes
        let logits = 32000 * 2; // vocab × batch(1) × 2 bytes
        assert_eq!(a, streaming + logits);
    }

    #[test]
    fn activation_scales_linearly_with_batch() {
        let dims = ActivationDims {
            hidden: 2048,
            intermediate: 5632,
            vocab: 50000,
        };
        let b1 = compute_activation_bytes(&dims, 4096, 1, 2.0);
        let b4 = compute_activation_bytes(&dims, 4096, 4, 2.0);
        // Both the streaming and logit terms scale with batch.
        assert_eq!(b4, b1 * 4);
    }

    #[test]
    fn activation_is_capped_by_prefill_chunk() {
        let dims = ActivationDims {
            hidden: 2048,
            intermediate: 5632,
            vocab: 0,
        };
        // Past the prefill chunk, activation does not grow with context.
        let at_8k = compute_activation_bytes(&dims, 8192, 1, 2.0);
        let at_32k = compute_activation_bytes(&dims, 32768, 1, 2.0);
        assert_eq!(at_8k, at_32k);
        // Below the chunk, it is smaller (prefill = ctx).
        let at_256 = compute_activation_bytes(&dims, 256, 1, 2.0);
        assert!(at_256 < at_8k);
        assert_eq!(at_256 * (ACTIVATION_PREFILL_TOKENS / 256), at_8k);
    }

    #[test]
    fn estimate_total_includes_activation_reserve() {
        let tmp = tempfile::tempdir().unwrap();
        write_minimal_config(tmp.path());
        let est = estimate_total_memory(tmp.path(), 8192, 4, QuantHint::Default, false);
        assert!(
            est.activation_bytes > 0,
            "a config with hidden_size must yield a nonzero activation reserve"
        );
        // runtime_headroom_bytes = allocator overhead + activation; both included
        // in the total.
        assert!(est.runtime_headroom_bytes >= est.activation_bytes);
        assert_eq!(
            est.total_bytes,
            est.weights_bytes + est.kv_cache_bytes + est.runtime_headroom_bytes
        );
    }

    #[test]
    fn activation_grows_with_batch_through_the_full_estimate() {
        let tmp = tempfile::tempdir().unwrap();
        write_minimal_config(tmp.path());
        let b1 = estimate_total_memory(tmp.path(), 8192, 1, QuantHint::Default, false);
        let b8 = estimate_total_memory(tmp.path(), 8192, 8, QuantHint::Default, false);
        // The old flat headroom was batch-blind; the activation term now makes
        // the reserve grow with batch.
        assert!(b8.activation_bytes > b1.activation_bytes);
        assert_eq!(b8.activation_bytes, b1.activation_bytes * 8);
    }

    #[test]
    fn activation_dims_default_intermediate_and_vocab() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = serde_json::json!({ "hidden_size": 1024 });
        std::fs::write(
            tmp.path().join("config.json"),
            serde_json::to_string(&cfg).unwrap(),
        )
        .unwrap();
        let dims = activation_dims_from_path(tmp.path()).unwrap();
        assert_eq!(dims.hidden, 1024);
        assert_eq!(dims.intermediate, 4096); // 4 × hidden fallback
        assert_eq!(dims.vocab, 0); // no logit term when absent
    }
}
