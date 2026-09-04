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
use serde::Serialize;

use super::config_fields;
use super::quant_advisor::estimate_model_params_billions;
use crate::models::{ModelType, get_model_type};

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

    /// Stable short label used by machine-readable inspect output.
    pub fn json_label(self) -> &'static str {
        match self {
            QuantHint::Default => "default",
            QuantHint::Fp16 => "fp16",
            QuantHint::Int8 => "int8",
            QuantHint::Int4 => "int4",
        }
    }
}

impl WeightsSource {
    fn json_label(self) -> &'static str {
        match self {
            WeightsSource::ExactSafetensors => "safetensors_header",
            WeightsSource::AnalyticalConfig => "analytical_config",
            WeightsSource::Fallback => "fallback",
        }
    }
}

/// How an operator asked the paged KV pool's block budget to be sized
/// (the `--kv-cache-budget` server knob, epic #116 #122 b3).
///
/// Resolved to a concrete block count by [`resolve_paged_block_budget`]
/// once the model geometry is known. `None` (the flag absent) keeps the
/// pool unbounded — the default, behaviour-preserving path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagedBudgetDirective {
    /// An explicit byte cap on the paged KV pool, converted to a block
    /// count via the model's per-block byte cost. Raw bytes, matching the
    /// other byte-valued server knobs (e.g. `--prompt-cache-capacity-bytes`).
    Bytes(u64),
    /// Derive the cap from [`estimate_total_memory`]: the unified-memory
    /// headroom left for KV after weights, activation, and the allocator
    /// safety factor (`--kv-cache-budget auto`). This is the server default.
    Auto,
    /// Explicit opt-out: leave the paged KV pool unbounded. Spelled `none`,
    /// `off`, `disabled`, `unbounded`, or `0` on the command line. Restores the
    /// pre-default behaviour where no admission cap is installed.
    Disabled,
}

impl std::str::FromStr for PagedBudgetDirective {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed = s.trim();
        if trimmed.eq_ignore_ascii_case("auto") {
            return Ok(Self::Auto);
        }
        // Explicit opt-out keywords leave the pool unbounded.
        if matches!(
            trimmed.to_ascii_lowercase().as_str(),
            "none" | "off" | "disabled" | "unbounded"
        ) {
            return Ok(Self::Disabled);
        }
        // A plain byte count. Human-readable suffixes (8GiB) are
        // intentionally not parsed, to stay consistent with the other
        // byte-valued server knobs which all take raw bytes. A budget of `0`
        // is meaningless as a cap, so it is treated as the explicit opt-out.
        trimmed
            .parse::<u64>()
            .map(|bytes| {
                if bytes == 0 {
                    Self::Disabled
                } else {
                    Self::Bytes(bytes)
                }
            })
            .map_err(|_| format!("expected a byte count, 'auto', or 'none', got '{s}'"))
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

/// JSON payload emitted by `mlxcel inspect --json`.
///
/// The byte fields are copied from [`MemoryEstimate`] so scripts do not have
/// to scrape the human-readable banner. Optional fields serialize as `null`
/// when the estimator or model config cannot provide them.
#[derive(Debug, Clone, Serialize)]
pub struct InspectReport {
    pub mlxcel_version: &'static str,
    pub model: String,
    pub model_type: Option<String>,
    pub family: Option<String>,
    pub inputs: InspectReportInputs,
    pub weights_bytes: u64,
    pub weights_source: &'static str,
    pub kv_bytes_per_token: InspectKvBytesPerToken,
    pub kv_bytes_total: u64,
    pub kv_detail: String,
    pub per_slot_overhead_bytes: Option<u64>,
    pub activation_bytes: u64,
    pub headroom_bytes: u64,
    pub headroom_factor: f64,
    pub budget_bytes: u64,
    pub total_bytes: u64,
    pub fits: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct InspectReportInputs {
    pub max_tokens: u64,
    pub batch: u64,
    pub kv_cache_mode: String,
    pub quant: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct InspectKvBytesPerToken {
    pub fp16: Option<u64>,
    pub int8: Option<u64>,
    pub turbo4: Option<u64>,
}

impl InspectReport {
    /// Build the machine-readable inspect report from the same estimator state
    /// that backs the text banner.
    #[must_use]
    pub fn from_estimate(
        model_dir: &Path,
        est: &MemoryEstimate,
        kv_cache_mode: String,
        model_type: Option<String>,
        family: Option<String>,
        per_slot_overhead_bytes: Option<u64>,
    ) -> Self {
        let kv_bytes_per_token = if matches!(est.kv_source, KvSource::Config) {
            InspectKvBytesPerToken {
                fp16: Some(kv_cache_bytes_per_token(model_dir, false, 1)),
                int8: Some(kv_cache_bytes_per_token(model_dir, true, 1)),
                turbo4: None,
            }
        } else {
            InspectKvBytesPerToken {
                fp16: None,
                int8: None,
                turbo4: None,
            }
        };

        Self {
            mlxcel_version: env!("CARGO_PKG_VERSION"),
            model: model_dir.display().to_string(),
            model_type,
            family,
            inputs: InspectReportInputs {
                max_tokens: est.ctx_len,
                batch: est.batch,
                kv_cache_mode,
                quant: est.quant.json_label(),
            },
            weights_bytes: est.weights_bytes,
            weights_source: est.weights_source.json_label(),
            kv_bytes_per_token,
            kv_bytes_total: est.kv_cache_bytes,
            kv_detail: est.kv_detail.clone(),
            per_slot_overhead_bytes,
            activation_bytes: est.activation_bytes,
            headroom_bytes: est.runtime_headroom_bytes,
            headroom_factor: est.headroom_factor,
            budget_bytes: est.available_bytes,
            total_bytes: est.total_bytes,
            fits: est.fits,
        }
    }
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
///
/// The alias lists come from [`crate::execution::config_fields`], shared with
/// the KV classifier: a hidden-size spelling the classifier accepts but this
/// function does not means a model reports a real KV figure next to a zero
/// activation reserve.
fn activation_dims_from_path(model_dir: &Path) -> Option<ActivationDims> {
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json")).ok()?).ok()?;
    let text = config_fields::text_config(&config);
    let hidden = config_fields::get_u64(text, config_fields::HIDDEN_SIZE_KEYS)?;
    let intermediate = config_fields::get_u64(text, config_fields::INTERMEDIATE_SIZE_KEYS)
        .unwrap_or_else(|| hidden.saturating_mul(4));
    let vocab = config_fields::get_u64(text, &["vocab_size"]).unwrap_or(0);
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

/// Read `MLXCEL_MEMORY_LIMIT` the way the runtime allocator cap reads it.
///
/// The `0` / `none` / empty spellings mean "unset" and belong here, next to
/// the preflight that consumes the result; everything else goes through
/// [`crate::execution::runtime::parse_memory_size`], so the preflight and the
/// allocator cap resolve one string to one number. Issue #1317: this used to
/// be a second parser that accepted only `GB` and `MB`, which made
/// `MLXCEL_MEMORY_LIMIT=4G` cap the allocator while the preflight ignored it
/// and reported the machine's total memory.
fn parse_optional_memory_size_bytes(raw: &str) -> Option<u64> {
    let s = raw.trim();
    if s.is_empty() || s == "0" || s.eq_ignore_ascii_case("none") {
        return None;
    }
    crate::execution::runtime::parse_memory_size(s).filter(|bytes| *bytes > 0)
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
///
/// Field names and the KV-head resolution come from
/// [`crate::execution::config_fields`], the same source
/// [`crate::execution::kv_arch`] classifies from, so this cannot drift out of
/// agreement with the classifier.
pub fn kv_cache_params_from_path(
    model_dir: &Path,
    ctx_len: u64,
    int8_kv: bool,
    batch: u64,
) -> Option<KvCacheParams> {
    let config_path = model_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path).ok()?;
    let config: serde_json::Value = serde_json::from_str(&config_str).ok()?;
    let text_cfg = config_fields::text_config(&config);

    let num_layers = config_fields::get_u64(text_cfg, config_fields::LAYER_COUNT_KEYS)?;
    let hidden_size = config_fields::get_u64(text_cfg, config_fields::HIDDEN_SIZE_KEYS);
    let num_heads = config_fields::get_u64(text_cfg, config_fields::NUM_HEADS_KEYS).unwrap_or(1);
    let num_kv_heads = config_fields::resolve_num_kv_heads(text_cfg, num_heads);
    let explicit_head_dim = config_fields::get_u64(text_cfg, config_fields::HEAD_DIM_KEYS);
    let head_dim = if let Some(head_dim) = explicit_head_dim {
        head_dim
    } else {
        // 64 is the historical fallback for malformed configs with zero heads.
        hidden_size?.checked_div(num_heads).unwrap_or(64)
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

/// Read the raw top-level `model_type` string from `config.json`.
#[must_use]
pub fn raw_model_type_from_config(model_dir: &Path) -> Option<String> {
    let config = read_model_config(model_dir)?;
    config
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .map(ToOwned::to_owned)
}

/// Best-effort family slug for `mlxcel inspect --json`.
///
/// TODO(#1508): replace this local fallback with the `mlxcel arch --json`
/// registry id once that contract is available on `main`.
#[must_use]
pub fn inspect_family_slug(model_dir: &Path) -> Option<String> {
    get_model_type(model_dir)
        .ok()
        .map(|model_type| inspect_registry_id(model_type).to_string())
}

/// Fused paged-decode workspace reserve for families that select the paged
/// backend by default. Families outside this set keep the field `null`.
#[must_use]
pub fn inspect_per_slot_overhead_bytes(model_dir: &Path, batch: u64) -> Option<u64> {
    let model_type = get_model_type(model_dir).ok()?;
    if !model_type_uses_default_paged_decode(model_type) {
        return None;
    }
    let num_layers = kv_cache_params_from_path(model_dir, DEFAULT_CTX_LEN, false, 1)?.num_layers;
    Some(paged_v2_workspace_reserve_bytes(
        model_dir,
        num_layers as usize,
        batch,
    ))
}

fn read_model_config(model_dir: &Path) -> Option<serde_json::Value> {
    let config_str = std::fs::read_to_string(model_dir.join("config.json")).ok()?;
    serde_json::from_str(&config_str).ok()
}

fn model_type_uses_default_paged_decode(model_type: ModelType) -> bool {
    matches!(
        model_type,
        ModelType::Llama
            | ModelType::Llama4
            | ModelType::Qwen3
            | ModelType::Qwen35
            | ModelType::Gemma3
    )
}

fn inspect_registry_id(model_type: ModelType) -> &'static str {
    match model_type {
        ModelType::Llama => "llama",
        ModelType::IQuestCoder => "iquest_coder",
        ModelType::Llama4 => "llama4",
        ModelType::Llama4VLM => "llama4_vlm",
        ModelType::MllamaVLM => "mllama_vlm",
        ModelType::Qwen2 => "qwen2",
        ModelType::Qwen3 => "qwen3",
        ModelType::Qwen3Moe => "qwen3_moe",
        ModelType::Qwen3Next => "qwen3_next",
        ModelType::Qwen35 => "qwen3_5",
        ModelType::Qwen35VLM => "qwen3_5_vlm",
        ModelType::Qwen35Moe => "qwen3_5_moe",
        ModelType::Qwen35MoeVLM => "qwen3_5_moe_vlm",
        ModelType::Gemma => "gemma",
        ModelType::Gemma2 => "gemma2",
        ModelType::Gemma3 => "gemma3",
        ModelType::Gemma4 => "gemma4",
        ModelType::DiffusionGemma => "diffusion_gemma",
        ModelType::Llada2Moe => "llada2_moe",
        ModelType::Gemma3VLM => "gemma3_vlm",
        ModelType::Gemma4VLM => "gemma4_vlm",
        ModelType::Gemma4Unified => "gemma4_unified",
        ModelType::LlavaVLM => "llava_vlm",
        ModelType::GraniteVisionVLM => "granite_vision_vlm",
        ModelType::Granite4VisionVLM => "granite4_vision_vlm",
        ModelType::DeepSeekOcrVLM => "deepseek_ocr_vlm",
        ModelType::DeepSeekOcr2VLM => "deepseek_ocr2_vlm",
        ModelType::UnlimitedOcrVLM => "unlimited_ocr_vlm",
        ModelType::DeepSeekVL2 => "deepseek_vl2",
        ModelType::LlavaBunnyVLM => "llava_bunny_vlm",
        ModelType::FastVLM => "fast_vlm",
        ModelType::Ernie45MoeVLM => "ernie4_5_moe_vlm",
        ModelType::HunyuanVLM => "hunyuan_vlm",
        ModelType::AyaVisionVLM => "aya_vision_vlm",
        ModelType::PaliGemmaVLM => "paligemma_vlm",
        ModelType::PixtralVLM => "pixtral_vlm",
        ModelType::Mistral3VLM => "mistral3_vlm",
        ModelType::Qwen2VL => "qwen2_vl",
        ModelType::Qwen25VL => "qwen2_5_vl",
        ModelType::Qwen3VL => "qwen3_vl",
        ModelType::Qwen3VLMoe => "qwen3_vl_moe",
        ModelType::Qwen3OmniMoe => "qwen3_omni_moe",
        ModelType::PaddleOcrVL => "paddleocr_vl",
        ModelType::DotsOcrVL => "dots_ocr_vl",
        ModelType::FalconOcrVL => "falcon_ocr_vl",
        ModelType::JinaVLM => "jina_vlm",
        ModelType::Glm4v => "glm4v",
        ModelType::Glm4vMoe => "glm4v_moe",
        ModelType::GlmOcr => "glm_ocr",
        ModelType::YoutuLLM => "youtu_llm",
        ModelType::YoutuVLM => "youtu_vlm",
        ModelType::InternVLChatVLM => "internvl_chat_vlm",
        ModelType::LocateAnythingVLM => "locateanything_vlm",
        ModelType::SmolVLM => "smolvlm",
        ModelType::Idefics2 => "idefics2",
        ModelType::MiniCPMOVLM => "minicpmo_vlm",
        ModelType::MiniCPMV46VLM => "minicpmv4_6_vlm",
        ModelType::Moondream3VLM => "moondream3_vlm",
        ModelType::Moondream2VLM => "moondream2_vlm",
        ModelType::Florence2VLM => "florence2_vlm",
        ModelType::Gemma3n => "gemma3n",
        ModelType::Gemma3nVLM => "gemma3n_vlm",
        ModelType::Phi => "phi",
        ModelType::Phixtral => "phixtral",
        ModelType::Phi3 => "phi3",
        ModelType::Phi4MMVLM => "phi4_mm_vlm",
        ModelType::Phi4SigLipVLM => "phi4_siglip_vlm",
        ModelType::Phi3VLM => "phi3_vlm",
        ModelType::MolmoVLM => "molmo_vlm",
        ModelType::Molmo2VLM => "molmo2_vlm",
        ModelType::MolmoPointVLM => "molmo_point_vlm",
        ModelType::Phi3Small => "phi3small",
        ModelType::PhiMoe => "phimoe",
        ModelType::GptOss => "gpt_oss",
        ModelType::MiniMax => "minimax",
        ModelType::MiniMaxM3 => "minimax_m3",
        ModelType::MiniMaxM3VL => "minimax_m3_vl",
        ModelType::MuseGlimmerVLM => "muse_glimmer_vlm",
        ModelType::Mixtral => "mixtral",
        ModelType::Qwen2Moe => "qwen2_moe",
        ModelType::OLMoE => "olmoe",
        ModelType::Dbrx => "dbrx",
        ModelType::DeepSeek => "deepseek",
        ModelType::DeepSeekV2 => "deepseek_v2",
        ModelType::DeepSeekV3 => "deepseek_v3",
        ModelType::DeepSeekV32 => "deepseek_v32",
        ModelType::DeepSeekV4 => "deepseek_v4",
        ModelType::Dots1 => "dots1",
        ModelType::Cohere => "cohere",
        ModelType::Cohere2 => "cohere2",
        ModelType::Cohere2Moe => "cohere2_moe",
        ModelType::InternLM2 => "internlm2",
        ModelType::InternLM3 => "internlm3",
        ModelType::Baichuan => "baichuan",
        ModelType::Glm4 => "glm4",
        ModelType::Glm4Moe => "glm4_moe",
        ModelType::Glm4MoeLite => "glm4_moe_lite",
        ModelType::GlmMoeDsa => "glm_moe_dsa",
        ModelType::Ernie45 => "ernie4_5",
        ModelType::Ernie45Moe => "ernie4_5_moe",
        ModelType::HunyuanMoe => "hunyuan_moe",
        ModelType::HunyuanV1Dense => "hunyuan_v1_dense",
        ModelType::MiMo => "mimo",
        ModelType::BailingMoe => "bailing_moe",
        ModelType::BailingMoeLinear => "bailing_moe_linear",
        ModelType::Afmoe => "afmoe",
        ModelType::Klear => "klear",
        ModelType::Apertus => "apertus",
        ModelType::SeedOss => "seed_oss",
        ModelType::Granite => "granite",
        ModelType::BitNet => "bitnet",
        ModelType::ExaOne => "exaone",
        ModelType::ExaOne4 => "exaone4",
        ModelType::ExaOneMoe => "exaone_moe",
        ModelType::SolarOpen => "solar_open",
        ModelType::Olmo => "olmo",
        ModelType::Olmo2 => "olmo2",
        ModelType::Olmo3 => "olmo3",
        ModelType::OpenElm => "openelm",
        ModelType::Gpt2 => "gpt2",
        ModelType::GptBigCode => "gpt_bigcode",
        ModelType::GptNeoX => "gpt_neox",
        ModelType::StarCoder2 => "starcoder2",
        ModelType::Mellum => "mellum",
        ModelType::Helium => "helium",
        ModelType::TeleChat3 => "telechat3",
        ModelType::MiniCPM => "minicpm",
        ModelType::MiniCPM3 => "minicpm3",
        ModelType::StableLM => "stablelm",
        ModelType::SmolLM3 => "smollm3",
        ModelType::Ministral3 => "ministral3",
        ModelType::Mistral3 => "mistral3",
        ModelType::Mistral4 => "mistral4",
        ModelType::Nemotron => "nemotron",
        ModelType::Mamba => "mamba",
        ModelType::Mamba2 => "mamba2",
        ModelType::Jamba => "jamba",
        ModelType::NemotronH => "nemotron_h",
        ModelType::NemotronHNanoOmniVLM => "nemotron_h_nano_omni_vlm",
        ModelType::NemotronNAS => "nemotron_nas",
        ModelType::FalconH1 => "falcon_h1",
        ModelType::Lfm2 => "lfm2",
        ModelType::Lfm2Moe => "lfm2_moe",
        ModelType::Lfm2VL => "lfm2_vl",
        ModelType::Inkling => "inkling",
        ModelType::InklingVLM => "inkling_vlm",
        ModelType::Plamo2 => "plamo2",
        ModelType::GraniteMoeHybrid => "granitemoehybrid",
        ModelType::KimiLinear => "kimi_linear",
        ModelType::KimiVL => "kimi_vl",
        ModelType::KimiK25 => "kimi_k25",
        ModelType::LongcatFlash => "longcat_flash",
        ModelType::LongcatFlashNgram => "longcat_flash_ngram",
        ModelType::Step3p5 => "step3p5",
        ModelType::Step3p7 => "step3p7",
        ModelType::Rwkv7 => "rwkv7",
        ModelType::RecurrentGemma => "recurrent_gemma",
        ModelType::Whisper => "whisper",
        ModelType::Kokoro => "kokoro",
        ModelType::Bert => "bert",
        ModelType::XlmRoberta => "xlm_roberta",
        ModelType::ModernBert => "modernbert",
        ModelType::SiglipText => "siglip",
        ModelType::Gemma3Embedding => "gemma3_embedding",
        ModelType::Qwen3Embedding => "qwen3_embedding",
        ModelType::Qwen3VLEmbedding => "qwen3_vl_embedding",
        ModelType::Lfm2Embedding => "lfm2_embedding",
        ModelType::Ministral3Embedding => "ministral3_embedding",
        ModelType::LlamaBidirec => "llama_bidirec",
        ModelType::LlamaNemotronVLEmbedding => "llama_nemotron_vl_embedding",
        ModelType::ColIdefics3 => "colidefics3",
        ModelType::ColQwen25 => "colqwen2_5",
        ModelType::SequenceClassifier => "sequence_classifier",
    }
}

// ── Paged KV block-budget resolution (epic #116 #122 b3) ──────────────────────

/// Real byte cost of a single paged KV block: `block_size` tokens of one
/// layer's K+V at the pool's storage dtype.
///
/// The paged pool counts its budget in **per-layer blocks** — one block holds
/// `block_size` tokens for a single layer of a single sequence (see
/// `BatchScheduler::estimate_prefill_blocks`, which sizes a prefill as
/// `ceil(prompt / block_size) × num_layers`). This is therefore the unit a
/// byte budget must be divided by.
///
/// Derived from the same architecture-aware geometry as
/// [`estimate_total_memory`]: the steady-state per-token KV rate
/// ([`kv_cache_bytes_per_token`] at `batch = 1`) summed across attention
/// layers, divided by `num_layers` to recover the per-layer rate, times
/// `block_size`. For the pure-attention Fp16 models that are pool-backed today
/// (Llama, Qwen3) every layer is a full-attention layer, so the division is
/// exact. For sliding-window / hybrid / MLA models the marginal rate counts
/// only the layers that grow, so dividing by the total layer count
/// *under*-estimates the per-block cost — but those families keep dense caches
/// and never touch the pool, so the figure is inert for them. If such a family
/// is ever pool-backed, swap this for the per-layer K/V geometry directly.
///
/// Returns `None` when the architecture is unavailable or
/// `num_layers` / `block_size` is zero.
#[must_use]
pub fn paged_block_bytes(
    model_dir: &Path,
    num_layers: usize,
    block_size: usize,
    kv_dtype_int8: bool,
) -> Option<u64> {
    if num_layers == 0 || block_size == 0 {
        return None;
    }
    let per_token_all_layers = kv_cache_bytes_per_token(model_dir, kv_dtype_int8, 1);
    let per_layer_per_token = per_token_all_layers / num_layers as u64;
    if per_layer_per_token == 0 {
        return None;
    }
    Some(per_layer_per_token.saturating_mul(block_size as u64))
}

/// Unified-memory headroom (bytes) available for the paged KV pool under the
/// `--kv-cache-budget auto` policy.
///
/// Inverts the [`estimate_total_memory`] fit inequality. Recall that
/// `total = headroom_factor × (weights + kv) + activation` (the allocator
/// overhead is `(factor − 1) × (weights + kv)`); requiring `total ≤ available`
/// and solving for the KV term gives
/// `kv ≤ (available − activation) / factor − weights`. Returns the clamped
/// non-negative headroom; `0` when the model leaves no room for KV.
fn auto_kv_budget_bytes(est: &MemoryEstimate) -> u64 {
    let factor = if est.headroom_factor.is_finite() && est.headroom_factor > 1.0 {
        est.headroom_factor
    } else {
        1.0
    };
    let after_activation = est.available_bytes.saturating_sub(est.activation_bytes);
    // `after_activation / factor` in f64; byte magnitudes (≤ ~10^12) sit far
    // inside f64's exact-integer range, and factor ≥ 1.0 only shrinks the value.
    let scaled = (after_activation as f64 / factor).floor();
    let scaled = if scaled.is_finite() && scaled >= 0.0 {
        scaled.min(u64::MAX as f64) as u64
    } else {
        0
    };
    scaled.saturating_sub(est.weights_bytes)
}

/// Resolve a [`PagedBudgetDirective`] into a concrete paged-block count for the
/// server's `CachePool::set_paged_block_budget`.
///
/// `batch` is the server's configured active-sequence count (it scales the
/// activation reserve under [`PagedBudgetDirective::Auto`]). `block_size` is
/// the pool's block size (`DEFAULT_PAGED_BLOCK_SIZE`). Returns the number of
/// blocks the pool may mint, or `None` when the model geometry is unavailable
/// (the caller should then leave the pool unbounded). A returned `Some(0)`
/// means the budget rounds below one block — the caller decides whether to
/// reject that config or leave the pool unbounded; it must not install a zero
/// budget, which would wedge every request.
#[must_use]
pub fn resolve_paged_block_budget(
    model_dir: &Path,
    num_layers: usize,
    block_size: usize,
    batch: u64,
    kv_dtype_int8: bool,
    directive: PagedBudgetDirective,
) -> Option<usize> {
    let per_block = paged_block_bytes(model_dir, num_layers, block_size, kv_dtype_int8)?;
    let budget_bytes = match directive {
        // Explicit opt-out: no cap, leave the pool unbounded (mirrors the
        // absent flag / the pre-default behaviour).
        PagedBudgetDirective::Disabled => return None,
        PagedBudgetDirective::Bytes(bytes) => bytes,
        PagedBudgetDirective::Auto => {
            let est = estimate_total_memory(
                model_dir,
                DEFAULT_CTX_LEN,
                batch.max(1),
                QuantHint::Default,
                kv_dtype_int8,
            );
            auto_kv_budget_bytes(&est)
        }
    };
    // #899: the fused decode v2 workspace lives outside the block pool but
    // inside the same memory, so charge it to the KV budget before converting
    // the remainder into blocks. Admission then reserves blocks it can actually
    // back with memory.
    //
    // The subtraction has to saturate (#1091). The reserve is device-derived, so
    // it is not bounded by anything the caller asked for: it scales with
    // [`mlxcel_core::paged_v2::device_target_ctas`], which is 512 on every
    // non-Metal host, putting it at 16.25 MiB for the common 8-kv-head /
    // 128-head-dim geometry. A `--kv-cache-budget` in the low megabytes is
    // therefore smaller than the reserve on any Linux or CUDA box, and a
    // wrapping `-` would turn that shortfall into a budget near `u64::MAX`. That
    // divides down to roughly 2^47 blocks for a 128 KiB block, not to
    // `usize::MAX`: the `unwrap_or` below only fires where the `try_from` can
    // fail, which is a 32-bit target. Either way it is an admission cap that
    // admits everything, the exact opposite of what was asked for. Saturating
    // yields `Some(0)`, which `resolve_worker_paged_block_budget` maps to "leave
    // the pool unbounded" with a warning naming the reserve.
    let workspace = paged_v2_workspace_reserve_bytes(model_dir, num_layers, batch);
    let budget_bytes = budget_bytes.saturating_sub(workspace);
    Some(usize::try_from(budget_bytes / per_block).unwrap_or(usize::MAX))
}

/// Concurrent v2 launches whose workspaces can be alive at once.
///
/// The decode lookahead pipeline (#632) can have the current step's forward and
/// the next step's prime in flight together, and each holds one layer's
/// workspace; nothing holds more.
const PAGED_V2_CONCURRENT_LAUNCHES: u64 = 2;

/// Upper bound on the GQA replication factor used for the workspace reserve.
///
/// The workspace scales with the query-head count, which
/// [`mlxcel_core::hardware::KvCacheParams`] does not carry (it only needs the
/// KV side). Bounding `Hq / Hkv` at 16 covers every family in
/// `docs/supported-models.md` (Llama 3 is 4, Qwen 2.5 7B is 7, the widest MQA
/// ports are 8) with headroom, and over-reserving a few megabytes is the safe
/// direction for an admission bound.
const PAGED_V2_MAX_N_REP: u64 = 16;

/// Bytes to reserve for the fused decode v2 workspace (issue #899).
///
/// The workspace is the partial kernel's `(partial_v, lse)` output pair, sized
/// `num_chunks * Hq * (head_dim + 1) * 4` bytes
/// ([`mlxcel_core::paged_v2::PagedDecodePlan::workspace_bytes`]). The plan's
/// binary search stops at the largest chunk size still reaching the device CTA
/// target, so `num_chunks * ctas_per_chunk` lands within a factor of two of
/// that target and
///
/// ```text
/// num_chunks <= 2 * target_ctas / Hkv + batch
/// ```
///
/// since `ctas_per_chunk = Hkv * q_groups >= Hkv`. Substituting
/// `Hq = Hkv * n_rep` makes the dominant term independent of the head counts:
/// `2 * target_ctas * n_rep * (head_dim + 1) * 4`.
///
/// [`PAGED_V2_MAX_N_REP`] makes the result a few times larger than any real
/// plan needs, which is the safe direction for an admission bound. Even so it
/// lands in the low tens of megabytes at most, well under a tenth of a percent
/// of a serving KV budget, which is why it was not accounted for before v2
/// became the production path. It is charged now so the accounting is truthful,
/// not because it is large.
#[must_use]
pub fn paged_v2_workspace_reserve_bytes(model_dir: &Path, num_layers: usize, batch: u64) -> u64 {
    let Some(params) = kv_cache_params_from_path(model_dir, DEFAULT_CTX_LEN, false, 1) else {
        return 0;
    };
    if num_layers == 0 || params.num_kv_heads == 0 || params.head_dim == 0 {
        return 0;
    }
    let target = mlxcel_core::paged_v2::device_target_ctas() as u64;
    let batch = batch.max(1);
    let chunks = target
        .saturating_mul(2)
        .div_ceil(params.num_kv_heads)
        .saturating_add(batch);
    let hq = params.num_kv_heads.saturating_mul(PAGED_V2_MAX_N_REP);
    let per_launch = chunks
        .saturating_mul(hq)
        .saturating_mul(params.head_dim.saturating_add(1))
        .saturating_mul(std::mem::size_of::<f32>() as u64);
    per_launch.saturating_mul(PAGED_V2_CONCURRENT_LAUNCHES)
}

// ── Paged pool slab sizing (issue #899) ──────────────────────────────────────

/// Environment override for the paged pool's slab size, in blocks.
///
/// `0` pins the pool default ([`mlxcel_core::cache::POOL_SLAB_BLOCKS`]), which
/// keeps the pre-#899 allocation behaviour and, as a side effect, keeps the
/// fused decode path unreachable for anything past one slab. Any other positive
/// integer is used verbatim, neither floored nor budget-capped, which is how
/// a benchmark sweeps the setting. Anything else warns and is ignored, so the
/// slab is derived exactly as if the variable were unset (#1137).
pub const PAGED_SLAB_BLOCKS_ENV: &str = "MLXCEL_PAGED_SLAB_BLOCKS";

/// Resolve the paged pool's slab size in blocks (issue #899).
///
/// ## Why this exists
///
/// The fused paged-attention decode kernels read **one contiguous pool buffer
/// per side**, so they can only serve a layer whose physical rows all live in
/// the pool's first slab. With the historical 32-block slab that caps them at
/// 1024 tokens across the entire batch (at `block_size` 32), which is below
/// every context the #899 dispatch policy would pick them for. Sizing the slab
/// to the workload is what makes the fused path reachable.
///
/// ## The policy
///
/// One slab per layer big enough for the batch the server was configured to
/// run at the context it was configured to serve:
///
/// ```text
/// blocks = ceil(per_slot_ctx / block_size) * batch
/// ```
///
/// clamped below by the pool default and above by the per-layer share of the
/// paged block budget, so the eager allocation can never exceed what
/// `--kv-cache-budget` already reserved. `per_slot_ctx` is the effective
/// per-slot `--ctx-size` (the same figure `--max-kv-size` is resolved from);
/// [`DEFAULT_CTX_LEN`] stands in when the operator did not set one, which is
/// also what [`estimate_total_memory`] assumes.
///
/// ## What it costs
///
/// The slab is allocated lazily, per layer, on that layer's first write, and
/// the whole slab is allocated at once. So the first prefill front-loads
/// `blocks * paged_block_bytes * num_layers` bytes: exactly the KV cache for
/// `batch` sequences at `per_slot_ctx` tokens, which is the figure the startup
/// memory estimate already reports. A layer that outgrows its slab appends a
/// second one exactly as before (no copy, #235) and simply stops being eligible
/// for the fused path, so an under-sized slab degrades to the pre-#899 gather
/// behaviour rather than failing.
///
/// Returns `None` when the geometry is unavailable or the operator pinned the
/// pool default, in which case the caller should not touch the pool's slab
/// size.
#[must_use]
pub fn resolve_paged_slab_blocks(
    model_dir: &Path,
    num_layers: usize,
    block_size: usize,
    batch: u64,
    per_slot_ctx: u64,
    kv_dtype_int8: bool,
    block_budget: Option<usize>,
) -> Option<usize> {
    if num_layers == 0 || block_size == 0 {
        return None;
    }
    if let Ok(raw) = std::env::var(PAGED_SLAB_BLOCKS_ENV) {
        match raw.trim().parse::<usize>() {
            Ok(0) => return None,
            Ok(n) => return Some(n),
            Err(_) => tracing::warn!(
                env_var = PAGED_SLAB_BLOCKS_ENV,
                value = raw,
                "{PAGED_SLAB_BLOCKS_ENV} must be a non-negative integer; ignoring it and \
                 deriving the slab as if unset (0 pins the pool default)",
            ),
        }
    }
    // The geometry probe doubles as the "is this model pool-eligible at all"
    // check: without it the byte clamp below would be meaningless.
    paged_block_bytes(model_dir, num_layers, block_size, kv_dtype_int8)?;

    let ctx = if per_slot_ctx == 0 {
        DEFAULT_CTX_LEN
    } else {
        per_slot_ctx
    };
    let per_seq_blocks = ctx.div_ceil(block_size as u64);
    let want = per_seq_blocks.saturating_mul(batch.max(1));
    let want = usize::try_from(want).unwrap_or(usize::MAX);

    let capped = match block_budget {
        // The budget is a global block count across every layer, so a layer may
        // claim at most its even share without the eager allocation being able
        // to exceed the budget the operator approved.
        Some(budget) => want.min((budget / num_layers).max(1)),
        None => want,
    };
    Some(capped.max(mlxcel_core::cache::POOL_SLAB_BLOCKS))
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

        fn unset(key: &'static str) -> Self {
            let previous = std::env::var(key).ok();
            // SAFETY: as for `set`.
            unsafe { std::env::remove_var(key) };
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
    fn inspect_report_copies_estimate_contract() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = serde_json::json!({
            "model_type": "llama",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        });
        write_config(tmp.path(), &cfg);
        write_safetensors_index(tmp.path(), 2_415_919_104);

        let est = estimate_total_memory(tmp.path(), 8192, 1, QuantHint::Default, false);
        let report = InspectReport::from_estimate(
            tmp.path(),
            &est,
            "fp16".to_string(),
            raw_model_type_from_config(tmp.path()),
            inspect_family_slug(tmp.path()),
            None,
        );

        assert_eq!(report.model_type.as_deref(), Some("llama"));
        assert_eq!(report.family.as_deref(), Some("llama"));
        assert_eq!(
            report.weights_bytes + report.kv_bytes_total + report.headroom_bytes,
            report.total_bytes
        );
        assert_eq!(report.budget_bytes, est.available_bytes);
        assert_eq!(report.fits, report.total_bytes <= report.budget_bytes);
        assert_eq!(report.inputs.max_tokens, est.ctx_len);
        assert_eq!(report.inputs.batch, est.batch);
        assert_eq!(report.inputs.kv_cache_mode, "fp16");
        assert_eq!(report.inputs.quant, "default");
        assert_eq!(report.weights_source, "safetensors_header");
        assert_eq!(
            report.kv_bytes_per_token.fp16,
            Some(kv_cache_bytes_per_token(tmp.path(), false, 1))
        );
        assert_eq!(
            report.kv_bytes_per_token.int8,
            Some(kv_cache_bytes_per_token(tmp.path(), true, 1))
        );
        assert_eq!(report.kv_bytes_per_token.turbo4, None);
        assert!(format_estimate(tmp.path(), &est).contains(&format_bytes(report.total_bytes)));
    }

    #[test]
    fn inspect_family_slug_matches_arch_registry_id() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = serde_json::json!({
            "model_type": "qwen3_5",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "intermediate_size": 11008,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
        });
        write_config(tmp.path(), &cfg);

        assert_eq!(
            raw_model_type_from_config(tmp.path()).as_deref(),
            Some("qwen3_5")
        );
        assert_eq!(inspect_family_slug(tmp.path()).as_deref(), Some("qwen3_5"));
    }

    #[test]
    fn inspect_family_slug_is_classifier_derived_not_raw_model_type() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = serde_json::json!({
            "model_type": "gemma3_text",
            "hidden_size": 2048,
            "num_hidden_layers": 18,
            "vocab_size": 256000,
            "intermediate_size": 8192,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
        });
        write_config(tmp.path(), &cfg);

        assert_eq!(
            raw_model_type_from_config(tmp.path()).as_deref(),
            Some("gemma3_text")
        );
        assert_eq!(inspect_family_slug(tmp.path()).as_deref(), Some("gemma3"));
    }

    #[test]
    fn inspect_report_serializes_nulls_and_stable_key_order() {
        let tmp = tempfile::tempdir().unwrap();
        let est = MemoryEstimate {
            weights_bytes: 100,
            kv_cache_bytes: 0,
            runtime_headroom_bytes: 20,
            activation_bytes: 4,
            total_bytes: 120,
            available_bytes: 128,
            fits: true,
            weights_source: WeightsSource::Fallback,
            kv_source: KvSource::Unavailable,
            kv_detail: "unavailable".to_string(),
            headroom_factor: DEFAULT_HEADROOM_FACTOR,
            ctx_len: 256,
            batch: 2,
            quant: QuantHint::Int8,
            kv_dtype_int8: true,
        };
        let report =
            InspectReport::from_estimate(tmp.path(), &est, "int8".to_string(), None, None, None);
        let json = serde_json::to_string_pretty(&report).unwrap();

        assert!(json.contains(r#""model_type": null"#));
        assert!(json.contains(r#""family": null"#));
        assert!(json.contains(r#""turbo4": null"#));
        assert!(json.contains(r#""per_slot_overhead_bytes": null"#));
        assert!(json.find(r#""mlxcel_version""#).unwrap() < json.find(r#""model""#).unwrap());
        assert!(json.find(r#""model""#).unwrap() < json.find(r#""model_type""#).unwrap());
        assert!(
            json.find(r#""weights_bytes""#).unwrap() < json.find(r#""kv_bytes_total""#).unwrap()
        );
        assert_eq!(report.kv_bytes_per_token.fp16, None);
        assert_eq!(report.kv_bytes_per_token.int8, None);
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

    // ── OpenAI-era field naming: GPT-2 / GPT-BigCode (#927) ──────────────────

    fn write_config(dir: &Path, cfg: &serde_json::Value) {
        std::fs::write(dir.join("config.json"), serde_json::to_string(cfg).unwrap()).unwrap();
    }

    /// `models/gpt2` verbatim.
    fn gpt2_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "gpt2",
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "vocab_size": 50257,
        })
    }

    /// `models/gpt_bigcode-santacoder` verbatim.
    fn gpt_bigcode_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "gpt_bigcode",
            "n_layer": 24,
            "n_head": 16,
            "n_embd": 2048,
            "n_inner": 8192,
            "multi_query": true,
            "vocab_size": 49280,
        })
    }

    /// Before #927 both families reported `KvSource::Unavailable` with 0 KV
    /// bytes AND a 0-byte activation reserve, because the layer-count and
    /// hidden-size lookups did not alias `n_layer` / `n_embd`.
    #[test]
    fn openai_era_naming_yields_nonzero_kv_and_activation() {
        for (name, cfg, expected_kv) in [
            ("gpt2", gpt2_config(), 301_989_888u64),
            ("gpt_bigcode", gpt_bigcode_config(), 100_663_296u64),
        ] {
            let tmp = tempfile::tempdir().unwrap();
            write_config(tmp.path(), &cfg);

            let est = estimate_total_memory(tmp.path(), 8192, 1, QuantHint::Default, false);
            assert_eq!(est.kv_source, KvSource::Config, "{name} kv source");
            assert_eq!(est.kv_cache_bytes, expected_kv, "{name} kv bytes");
            assert!(
                !est.kv_detail.contains("unavailable"),
                "{name} detail should not say unavailable, got {}",
                est.kv_detail
            );
            assert!(
                est.activation_bytes > 0,
                "{name} activation reserve should be non-zero"
            );
        }
    }

    /// `kv_cache_params_from_path` duplicates the classifier's geometry for the
    /// legacy recommendation engine; it must agree, boolean `multi_query`
    /// included, or the two surfaces disagree by a factor of `n_head`.
    #[test]
    fn kv_params_agree_with_classifier_for_openai_era_naming() {
        for (name, cfg, expected_kv_heads, expected_head_dim) in [
            ("gpt2", gpt2_config(), 12u64, 64u64),
            ("gpt_bigcode", gpt_bigcode_config(), 1u64, 128u64),
        ] {
            let tmp = tempfile::tempdir().unwrap();
            write_config(tmp.path(), &cfg);

            let params = kv_cache_params_from_path(tmp.path(), 8192, false, 1)
                .unwrap_or_else(|| panic!("{name} params"));
            assert_eq!(params.num_kv_heads, expected_kv_heads, "{name} kv heads");
            assert_eq!(params.head_dim, expected_head_dim, "{name} head dim");

            let est = estimate_total_memory(tmp.path(), 8192, 1, QuantHint::Default, false);
            assert_eq!(
                kv_cache_bytes_from_params(&params),
                est.kv_cache_bytes,
                "{name}: params and classifier must agree"
            );
        }
    }

    /// `paged_block_bytes` fed the server's `--kv-cache-budget auto` ceiling a
    /// `None` for both families, leaving the pool unbounded.
    #[test]
    fn paged_block_bytes_resolves_for_openai_era_naming() {
        let tmp = tempfile::tempdir().unwrap();
        write_config(tmp.path(), &gpt_bigcode_config());
        // 24 layers x 2 (K+V) x 1 kv head x 128 head_dim x 2 bytes = 12288/token
        // across all layers, so 512 bytes per layer per token, x 16 tokens.
        assert_eq!(paged_block_bytes(tmp.path(), 24, 16, false), Some(8192));
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

    /// Issue #1317: the short suffix reaches the preflight too. Before the two
    /// parsers were merged this returned the machine's total memory, because
    /// the preflight's own parser took `MB` but not `M`.
    #[test]
    fn available_memory_honors_short_suffix_env_limit() {
        let _env = crate::test_support::env_lock::env_lock();
        let _restore = EnvRestore::set(MEMORY_LIMIT_ENV, "512M");

        let tmp = tempfile::tempdir().unwrap();
        write_minimal_config(tmp.path());

        let est = estimate_total_memory(tmp.path(), 1024, 1, QuantHint::Default, false);
        assert_eq!(est.available_bytes, 512 * 1024 * 1024);
    }

    /// Issue #1137: the warning for a malformed `MLXCEL_PAGED_SLAB_BLOCKS`
    /// says the derived slab size is used, and the code used to return the
    /// `0` pin instead. The three arms must stay distinct: unset and
    /// malformed derive, `0` pins the pool default (`None`), a number wins
    /// verbatim.
    #[test]
    fn paged_slab_blocks_malformed_env_derives_like_unset() {
        let _env = crate::test_support::env_lock::env_lock();
        let tmp = tempfile::tempdir().unwrap();
        write_minimal_config(tmp.path());
        let resolve = || resolve_paged_slab_blocks(tmp.path(), 32, 32, 4, 8192, false, None);

        let derived = {
            let _unset = EnvRestore::unset(PAGED_SLAB_BLOCKS_ENV);
            resolve()
        };
        assert_eq!(derived, Some(8192 / 32 * 4));

        let _malformed = EnvRestore::set(PAGED_SLAB_BLOCKS_ENV, "lots");
        assert_eq!(resolve(), derived);

        let _zero = EnvRestore::set(PAGED_SLAB_BLOCKS_ENV, "0");
        assert_eq!(resolve(), None);

        // Trimmed and verbatim: not floored at the 32-block pool default the
        // derived path applies.
        let _explicit = EnvRestore::set(PAGED_SLAB_BLOCKS_ENV, " 8 ");
        assert_eq!(resolve(), Some(8));
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

    /// Issue #1317: one grammar, so the preflight resolves `4G` and `4GB` to
    /// the same number the allocator cap uses.
    #[test]
    fn parse_optional_memory_size_accepts_the_runtime_grammar() {
        let four_gib = Some(4 * 1024 * 1024 * 1024);
        assert_eq!(parse_optional_memory_size_bytes("4G"), four_gib);
        assert_eq!(parse_optional_memory_size_bytes("4GB"), four_gib);
        assert_eq!(parse_optional_memory_size_bytes("4gb"), four_gib);
        assert_eq!(
            parse_optional_memory_size_bytes("512M"),
            parse_optional_memory_size_bytes("512MB"),
        );
        assert_eq!(parse_optional_memory_size_bytes("8K"), Some(8192));
        assert_eq!(parse_optional_memory_size_bytes("1024"), Some(1024));
        // `0` after scaling is still "unset" to the preflight.
        assert_eq!(parse_optional_memory_size_bytes("0GB"), None);
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

    // ── #122 b3: paged KV block-budget resolution ───────────────────────────

    #[test]
    fn paged_budget_directive_parses_auto_and_bytes() {
        assert_eq!(
            "auto".parse::<PagedBudgetDirective>().unwrap(),
            PagedBudgetDirective::Auto,
        );
        assert_eq!(
            "AUTO".parse::<PagedBudgetDirective>().unwrap(),
            PagedBudgetDirective::Auto,
        );
        assert_eq!(
            " 8589934592 ".parse::<PagedBudgetDirective>().unwrap(),
            PagedBudgetDirective::Bytes(8_589_934_592),
        );
        assert!("8GiB".parse::<PagedBudgetDirective>().is_err());
        assert!("-5".parse::<PagedBudgetDirective>().is_err());
    }

    #[test]
    fn paged_budget_directive_parses_disable_keywords() {
        // #628: `none` / `off` / `disabled` / `unbounded` / `0` all mean the
        // explicit opt-out (leave the pool unbounded).
        for spelling in ["none", "NONE", "off", "disabled", "unbounded", " none "] {
            assert_eq!(
                spelling.parse::<PagedBudgetDirective>().unwrap(),
                PagedBudgetDirective::Disabled,
                "spelling {spelling:?} should disable the budget",
            );
        }
        assert_eq!(
            "0".parse::<PagedBudgetDirective>().unwrap(),
            PagedBudgetDirective::Disabled,
        );
    }

    #[test]
    fn resolve_paged_block_budget_disabled_is_unbounded() {
        // A `Disabled` directive resolves to no cap (None) regardless of model
        // geometry, mirroring the absent flag.
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(
            tmp.path().join("config.json"),
            br#"{"num_hidden_layers": 32, "hidden_size": 4096, "num_attention_heads": 32}"#,
        )
        .unwrap();
        assert_eq!(
            resolve_paged_block_budget(
                tmp.path(),
                32,
                32,
                4,
                false,
                PagedBudgetDirective::Disabled
            ),
            None,
        );
    }

    #[test]
    fn paged_block_bytes_matches_uniform_geometry() {
        let tmp = tempfile::tempdir().unwrap();
        write_minimal_config(tmp.path());
        // 32 layers, 8 kv heads, head_dim 128, fp16:
        //   per-token (all layers) = 32 × 2 × 8 × 128 × 2 = 131072
        //   per-layer per-token    = 131072 / 32 = 4096
        //   per-block              = 4096 × 32 (block_size) = 131072
        assert_eq!(paged_block_bytes(tmp.path(), 32, 32, false), Some(131_072));
        // int8 halves the per-block cost.
        assert_eq!(paged_block_bytes(tmp.path(), 32, 32, true), Some(65_536));
    }

    #[test]
    fn paged_block_bytes_none_on_zero_or_missing_geometry() {
        let tmp = tempfile::tempdir().unwrap();
        write_minimal_config(tmp.path());
        assert_eq!(paged_block_bytes(tmp.path(), 0, 32, false), None);
        assert_eq!(paged_block_bytes(tmp.path(), 32, 0, false), None);
        // No config.json ⇒ no architecture ⇒ None.
        let empty = tempfile::tempdir().unwrap();
        assert_eq!(paged_block_bytes(empty.path(), 32, 32, false), None);
    }

    #[test]
    fn resolve_block_budget_explicit_bytes_floors_to_block_count() {
        let tmp = tempfile::tempdir().unwrap();
        write_minimal_config(tmp.path());
        let per_block = paged_block_bytes(tmp.path(), 32, 32, false).unwrap(); // 131072
        // The paged decode v2 workspace is charged to the byte budget before it
        // is divided into blocks (#899), so a caller asking for exactly N blocks
        // of KV has to ask for the workspace too.
        let workspace = paged_v2_workspace_reserve_bytes(tmp.path(), 32, 1);
        assert!(workspace > 0, "the minimal config has a derivable geometry");
        // Every request below is derived from the device-measured reserve, and
        // the `test-fast` profile inherits `release`, where overflow checks are
        // off. A plain `*` or `+` would wrap silently into a wrong expectation
        // rather than panicking on a device with an extreme CTA target, which is
        // the class of defect this test exists to pin, so the arithmetic
        // saturates throughout.
        let blocks_worth = |n: u64| per_block.saturating_mul(n);

        // Exactly 100 blocks.
        assert_eq!(
            resolve_paged_block_budget(
                tmp.path(),
                32,
                32,
                1,
                false,
                PagedBudgetDirective::Bytes(blocks_worth(100).saturating_add(workspace)),
            ),
            Some(100),
        );
        // 100 blocks + a partial ⇒ floors to 100.
        assert_eq!(
            resolve_paged_block_budget(
                tmp.path(),
                32,
                32,
                1,
                false,
                PagedBudgetDirective::Bytes(
                    blocks_worth(100)
                        .saturating_add(per_block / 2)
                        .saturating_add(workspace),
                ),
            ),
            Some(100),
        );
        // Below one block ⇒ 0 (caller leaves the pool unbounded rather than
        // installing a wedging zero budget).
        assert_eq!(
            resolve_paged_block_budget(
                tmp.path(),
                32,
                32,
                1,
                false,
                PagedBudgetDirective::Bytes(per_block.saturating_sub(1).saturating_add(workspace)),
            ),
            Some(0),
        );
        // A budget that did not account for the workspace comes back short by
        // exactly the reserve, which is the point: the blocks it hands out are
        // blocks it can actually back.
        //
        // The reserve is device-derived, so neither the request nor the
        // expectation may assume it is small (#1091). `device_target_ctas()` is
        // 512 on every non-Metal host and `gpu_core_count * 8` on Apple ones,
        // which puts the reserve at 16.25 MiB for this geometry on a CUDA box
        // against a 12.5 MiB 100-block budget. The expectation therefore mirrors
        // the implementation's `saturating_sub` instead of re-deriving it with a
        // `-` that wraps once the reserve exceeds the whole budget.
        let reserve_blocks = workspace.div_ceil(per_block);
        let shortfall =
            usize::try_from(blocks_worth(100).saturating_sub(workspace) / per_block).unwrap();
        assert_eq!(
            resolve_paged_block_budget(
                tmp.path(),
                32,
                32,
                1,
                false,
                PagedBudgetDirective::Bytes(blocks_worth(100)),
            ),
            Some(shortfall),
        );
        // The reserve costs exactly `ceil(workspace / per_block)` blocks, capped
        // at the 100 on offer when it swallows the budget outright.
        assert_eq!(
            100 - shortfall,
            usize::try_from(reserve_blocks.min(100)).unwrap(),
            "a {workspace}-byte reserve is {reserve_blocks} blocks, so it should cost \
             min({reserve_blocks}, 100) of the 100 blocks requested",
        );
        // The same contract without the cap, so the exact-reserve case is
        // exercised on large-CTA devices too: paying for the reserve in whole
        // blocks on top of the 100 hands back exactly the 100 blocks asked for.
        assert_eq!(
            resolve_paged_block_budget(
                tmp.path(),
                32,
                32,
                1,
                false,
                PagedBudgetDirective::Bytes(blocks_worth(reserve_blocks.saturating_add(100))),
            ),
            Some(100),
        );
    }

    /// Regression for #1091: a byte budget smaller than the device-derived
    /// paged decode v2 workspace reserve resolves to zero blocks, never to a
    /// wrapped block count.
    ///
    /// This pins the `saturating_sub` in `resolve_paged_block_budget` directly.
    /// The reserve is 16.25 MiB for this geometry on any non-Metal host
    /// (`device_target_ctas() == 512`), so an explicit `--kv-cache-budget` in
    /// the low megabytes reaches this path on any Linux or CUDA box. A wrapping
    /// `-` would turn a one-byte shortfall into a budget near `u64::MAX`, which
    /// divides down to roughly 2^47 blocks here: an admission cap that admits
    /// everything instead of capping the pool.
    #[test]
    fn resolve_block_budget_below_the_workspace_reserve_is_zero_blocks() {
        let tmp = tempfile::tempdir().unwrap();
        write_minimal_config(tmp.path());
        let per_block = paged_block_bytes(tmp.path(), 32, 32, false).unwrap();
        let workspace = paged_v2_workspace_reserve_bytes(tmp.path(), 32, 1);
        assert!(workspace > 0, "the minimal config has a derivable geometry");

        for budget in [0, 1, workspace / 2, workspace.saturating_sub(1), workspace] {
            assert_eq!(
                resolve_paged_block_budget(
                    tmp.path(),
                    32,
                    32,
                    1,
                    false,
                    PagedBudgetDirective::Bytes(budget),
                ),
                Some(0),
                "a {budget}-byte budget under a {workspace}-byte reserve must \
                 resolve to 0 blocks, not a wrapped count",
            );
        }
        // One whole block past the reserve is the first budget that mints
        // anything, so the boundary is pinned from both sides.
        assert_eq!(
            resolve_paged_block_budget(
                tmp.path(),
                32,
                32,
                1,
                false,
                PagedBudgetDirective::Bytes(workspace.saturating_add(per_block).saturating_sub(1)),
            ),
            Some(0),
        );
        assert_eq!(
            resolve_paged_block_budget(
                tmp.path(),
                32,
                32,
                1,
                false,
                PagedBudgetDirective::Bytes(workspace.saturating_add(per_block)),
            ),
            Some(1),
        );
    }

    #[test]
    fn the_v2_workspace_reserve_is_small_and_scales_with_the_head_dim() {
        let tmp = tempfile::tempdir().unwrap();
        write_minimal_config(tmp.path());
        let reserve = paged_v2_workspace_reserve_bytes(tmp.path(), 32, 4);
        // Single-digit megabytes: the reserve exists so admission is truthful,
        // not because it is a material share of a KV budget.
        assert!(reserve > 0, "expected a non-zero reserve, got {reserve}");
        assert!(
            reserve < 64 * 1024 * 1024,
            "the reserve should stay small, got {reserve} bytes"
        );
        // A model with no derivable geometry reserves nothing rather than
        // guessing.
        let empty = tempfile::tempdir().unwrap();
        assert_eq!(paged_v2_workspace_reserve_bytes(empty.path(), 32, 4), 0);
    }

    #[test]
    fn auto_kv_budget_inverts_the_fit_inequality() {
        // factor × (weights + kv) + activation ≤ available, solved for kv.
        let est = MemoryEstimate {
            weights_bytes: 10_000_000_000,
            kv_cache_bytes: 0,
            runtime_headroom_bytes: 0,
            activation_bytes: 1_000_000_000,
            total_bytes: 0,
            available_bytes: 25_000_000_000,
            fits: true,
            weights_source: WeightsSource::AnalyticalConfig,
            kv_source: KvSource::Config,
            kv_detail: String::new(),
            headroom_factor: 1.20,
            ctx_len: DEFAULT_CTX_LEN,
            batch: 1,
            quant: QuantHint::Default,
            kv_dtype_int8: false,
        };
        // (25e9 − 1e9) / 1.2 − 10e9 = 20e9 − 10e9 = 10e9.
        let budget = auto_kv_budget_bytes(&est);
        assert_eq!(budget, 10_000_000_000);
        // The result preserves the fit (with equality here).
        let reconstructed =
            (1.20_f64 * (est.weights_bytes + budget) as f64) as u64 + est.activation_bytes;
        assert!(reconstructed <= est.available_bytes);
    }

    #[test]
    fn auto_kv_budget_saturates_to_zero_when_overcommitted() {
        let est = MemoryEstimate {
            weights_bytes: 30_000_000_000,
            kv_cache_bytes: 0,
            runtime_headroom_bytes: 0,
            activation_bytes: 1_000_000_000,
            total_bytes: 0,
            available_bytes: 16_000_000_000,
            fits: false,
            weights_source: WeightsSource::AnalyticalConfig,
            kv_source: KvSource::Config,
            kv_detail: String::new(),
            headroom_factor: 1.20,
            ctx_len: DEFAULT_CTX_LEN,
            batch: 1,
            quant: QuantHint::Default,
            kv_dtype_int8: false,
        };
        assert_eq!(auto_kv_budget_bytes(&est), 0);
    }

    #[test]
    fn resolve_block_budget_auto_scales_with_available_memory() {
        let _env = crate::test_support::env_lock::env_lock();
        let tmp = tempfile::tempdir().unwrap();
        write_minimal_config(tmp.path());

        let auto = || {
            resolve_paged_block_budget(tmp.path(), 32, 32, 1, false, PagedBudgetDirective::Auto)
                .unwrap()
        };

        let big = {
            let _r = EnvRestore::set(MEMORY_LIMIT_ENV, "256GB");
            auto()
        };
        let small = {
            let _r = EnvRestore::set(MEMORY_LIMIT_ENV, "32GB");
            auto()
        };
        // More available memory ⇒ strictly more acquirable KV blocks.
        assert!(
            big > small,
            "256GB budget {big} should exceed 32GB budget {small}"
        );
        assert!(big > 0);

        // A limit below the (~12 GB) weight footprint leaves no KV room ⇒ 0.
        let starved = {
            let _r = EnvRestore::set(MEMORY_LIMIT_ENV, "1GB");
            auto()
        };
        assert_eq!(starved, 0);
    }
}
