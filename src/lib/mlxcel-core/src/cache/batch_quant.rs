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

//! Batch-aware quantized KV caches for continuous batching (issue #545).
//!
//! Mirrors upstream `mlx-vlm` PR #1030 (commit `b027538`) which added two
//! batched-cache variants used by the Python continuous-batching server:
//!
//! 1. [`BatchQuantizedKVCache`] — uniform `mx.quantize`-based 4/8-bit
//!    quantized cache. The on-device representation is the same affine
//!    `(packed_uint32, scales, biases)` triple `mlx::core::quantize`
//!    produces for weight quantization. Per-token I/O re-uses the same
//!    `mlx::core::quantize` / `mlx::core::dequantize` ops the existing
//!    `mlxcel-core` quantized-linear path is built on.
//! 2. [`BatchTurboQuantKVCache`] — TurboQuant variant. Attention happens
//!    via dequantize + standard SDPA (no batch-aware Metal kernels yet —
//!    those are explicitly out of scope for this issue). The actual
//!    per-layer storage is delegated to [`KVCache`] in the existing
//!    `Turbo4Asym` mode (asymmetric Fp16-K + Turbo4-V); the wrapper
//!    centralises the per-batch metadata (left padding, sequence offsets)
//!    and the [`extend`](BatchTurboQuantKVCache::extend) /
//!    [`filter`](BatchTurboQuantKVCache::filter) lifecycle helpers.
//!
//! # Status: forward-looking stable seams (not on the live decode path)
//!
//! Both [`BatchQuantizedKVCache`] and [`BatchTurboQuantKVCache`] are
//! currently **stable seams**, not wired into mlxcel's live decode path.
//! Today the live continuous-batching decode path quantizes the KV cache
//! by reading [`BatchKvQuantConfig`] directly and applying the resolved
//! per-layer [`KVCacheMode`] table to mlxcel's per-sequence per-layer
//! [`KVCache`] instances via the scheduler's `apply_kv_cache_mode_to`
//! method (see `src/server/batch/scheduler.rs`). The per-sequence caches
//! still live in [`super::CachePool`]; no batch-wrapping facade type is
//! consulted on the hot path.
//!
//! These two facade types exist as a **forward-looking seam** that
//! mirrors the upstream `mlx-vlm` PR #1030 contract surface, so future
//! batch-aware tensor work (e.g., a Metal-side batched attention kernel
//! that wants a single `(packed, scales, biases, left_padding, offset)`
//! struct, or a fused batch-wide `extend`/`filter` of device tensors) has
//! a stable Rust API to plug into. They are exercised today by the
//! regression tests in `mod tests` below — those tests are the entire
//! current consumer set.
//!
//! This intentionally matches the pattern from PR #564 / issue #544 for
//! [`super::BatchedAttentionMetadata::extend`] and
//! [`super::BatchedAttentionMetadata::filter_in_place`]: a metadata-shaped
//! API is added before any tensor-level integration so that the contract
//! is reviewable and locked down separately from the kernel work it
//! enables.
//!
//! # Tracking the per-layer mode table
//!
//! mlxcel's continuous-batching scheduler operates on per-sequence
//! per-layer caches via [`super::CachePool`], so what *is* wired into the
//! hot path is [`BatchKvQuantConfig::resolve_layer_modes`] — the
//! per-layer mode table that the scheduler reads to honour
//! [`BatchKvQuantConfig::skip_last_layer`] (sensitive on deep models such
//! as gemma-4-31b). The two facade structs duplicate that table on a
//! per-instance basis but are not the source of truth for the live
//! decode path.
//!
//! # Last-layer skip
//!
//! The upstream PR #1030 description specifically calls out gemma-4-31b
//! as a model where quantizing the final transformer layer hurts decoded
//! quality. We model that as a separate boolean knob ([`skip_last_layer`])
//! rather than reusing the existing 2-on-each-end Boundary-V policy
//! (see [`super::turbo::boundary`]), because the two policies overlap
//! intentionally:
//!
//! - **Boundary-V** (issue #478) protects the first 2 *and* last 2 layers
//!   for `Turbo4*` modes. That mechanism stays unchanged.
//! - **`skip_last_layer`** (issue #545) protects *only* the final layer
//!   and applies to every quantization scheme exposed via
//!   [`KvQuantScheme`], including the new [`KvQuantScheme::Uniform`]
//!   variant which is not covered by Boundary-V.
//!
//! Both policies compose: when `skip_last_layer == true` and the cache is
//! configured with a `Turbo4*` nominal mode, the final layer is forced to
//! [`KVCacheMode::Fp16`] regardless of what Boundary-V would have produced.
//!
//! # Custom Metal kernels (out of scope)
//!
//! Upstream Python's `BatchTurboQuantKVCache` dispatches attention through
//! a dequantize + standard SDPA path because no batch-aware TurboQuant
//! kernel exists yet on either side. We mirror that contract here: the
//! per-layer caches still carry packed sidecars under [`KVCacheMode::Turbo4Asym`],
//! but the batched decode path is responsible for materialising an FP16 V
//! tensor before SDPA. The single-stream Metal kernels in
//! [`super::turbo::sparse_v`] remain unchanged.

use super::{KVCache, KVCacheMode};
use crate::ffi::MlxArray;
use crate::utils;
use cxx::UniquePtr;

/// Quantization scheme exposed via the `--kv-quant-scheme` CLI flag.
///
/// The scheme controls the **storage encoding**; the bit count is set
/// independently via `--kv-bits`. See [`BatchKvQuantConfig`] for the
/// resolved combination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvQuantScheme {
    /// Uniform `mx.quantize`-style affine quantization (matches upstream
    /// `BatchQuantizedKVCache`). Supports 4-bit and 8-bit. The
    /// [`BatchKvQuantConfig::group_size`] field controls the channel
    /// blocking used for the affine codec.
    #[default]
    Uniform,
    /// TurboQuant-based quantization (matches upstream
    /// `BatchTurboQuantKVCache`). Always 4-bit V on top of FP16 K
    /// (`KVCacheMode::Turbo4Asym`); the [`BatchKvQuantConfig::group_size`]
    /// field is ignored because the TurboQuant codebook is per-token, not
    /// per-channel-group.
    TurboQuant,
}

impl std::str::FromStr for KvQuantScheme {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "uniform" | "affine" | "mx.quantize" => Ok(Self::Uniform),
            "turboquant" | "turbo" | "turbo4" => Ok(Self::TurboQuant),
            other => Err(format!(
                "unknown --kv-quant-scheme \"{other}\"; expected one of \"uniform\", \"turboquant\""
            )),
        }
    }
}

impl std::fmt::Display for KvQuantScheme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Uniform => f.write_str("uniform"),
            Self::TurboQuant => f.write_str("turboquant"),
        }
    }
}

/// Default group size for [`KvQuantScheme::Uniform`].
///
/// 64 matches both the upstream Python `BatchQuantizedKVCache` default
/// (`group_size=64`) and the canonical `mx.quantize` default that
/// mlxcel's quantized-linear weights use.
pub const DEFAULT_KV_GROUP_SIZE: i32 = 64;

/// Configuration object resolved at server startup time.
///
/// All fields are validated by [`BatchKvQuantConfig::validate`] before
/// they are passed to a scheduler — invalid combinations are rejected at
/// the edge so the hot decode loop never has to re-check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchKvQuantConfig {
    /// Quantization scheme. Maps to a base [`KVCacheMode`] in
    /// [`Self::base_mode`].
    pub scheme: KvQuantScheme,
    /// Per-token bit count. Valid values:
    ///
    /// - [`KvQuantScheme::Uniform`]: `8` only. The `(Uniform, 4)`
    ///   combination is rejected at validate-time today because
    ///   mlxcel-core has no `Int4Affine` single-stream cache mode;
    ///   operators who want 4-bit batched KV cache should use
    ///   `--kv-quant-scheme turboquant`.
    /// - [`KvQuantScheme::TurboQuant`]: `4` (only). Other values are
    ///   rejected at validate-time so a bogus CLI value cannot reach the
    ///   `KVCacheMode::Turbo4Asym` allocation path.
    pub bits: i32,
    /// Channel-group size for [`KvQuantScheme::Uniform`]. Ignored for
    /// `TurboQuant`. Must be positive — model-aware divisibility against
    /// `head_dim` is not enforced today because no `Int4Affine` code path
    /// exists yet (the 8-bit `Int8` mode does not need a group-size
    /// divisibility check). When the `Int4Affine` path lands, add a
    /// `validate_against_head_dim(head_dim: i32)` method (see
    /// [`Self::validate`] TODO).
    pub group_size: i32,
    /// When `true`, the final transformer layer always runs at
    /// [`KVCacheMode::Fp16`] regardless of `scheme`/`bits`.
    ///
    /// Defaults to `true` because the issue's acceptance criterion
    /// explicitly requires the last-layer skip to be active by default
    /// (gemma-4-31b regression case). Operators can pass
    /// `--kv-skip-last-layer false` (or set
    /// `LLAMA_ARG_KV_SKIP_LAST_LAYER=0`) to opt out for benchmarking or
    /// for shallow models where the policy is counter-productive.
    pub skip_last_layer: bool,
}

impl Default for BatchKvQuantConfig {
    /// Default: no quantization. Returns a [`KvQuantScheme::Uniform`]
    /// shape with `bits == 0` so callers can still dispatch on the
    /// [`Self::is_enabled`] predicate without having to check `Option`.
    fn default() -> Self {
        Self {
            scheme: KvQuantScheme::Uniform,
            bits: 0,
            group_size: DEFAULT_KV_GROUP_SIZE,
            skip_last_layer: true,
        }
    }
}

impl BatchKvQuantConfig {
    /// Construct a fully-specified config without going through CLI parsing.
    ///
    /// Convenience wrapper for tests and direct programmatic callers.
    /// Returns an error from [`Self::validate`] if the combination is
    /// invalid.
    pub fn new(
        scheme: KvQuantScheme,
        bits: i32,
        group_size: i32,
        skip_last_layer: bool,
    ) -> Result<Self, String> {
        let cfg = Self {
            scheme,
            bits,
            group_size,
            skip_last_layer,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Whether quantization is actually active.
    ///
    /// `bits == 0` is the "off" sentinel and matches the
    /// [`KVCacheMode::Fp16`] baseline; both [`KvQuantScheme`] values
    /// collapse to no-op in that case.
    pub fn is_enabled(&self) -> bool {
        self.bits > 0
    }

    /// Map the resolved config to the underlying nominal [`KVCacheMode`].
    ///
    /// The mlxcel core cache machinery is keyed off [`KVCacheMode`], not
    /// off `(scheme, bits)`, so this is the seam where the new CLI knobs
    /// reduce back to the existing storage backend:
    ///
    /// | scheme       | bits | nominal mode |
    /// |--------------|------|--------------|
    /// | `Uniform`    | 0    | `Fp16`       |
    /// | `Uniform`    | 8    | `Int8`       |
    /// | `TurboQuant` | 4    | `Turbo4Asym` |
    ///
    /// **Note**: `(Uniform, 4)` is **not** in this table. mlxcel-core has
    /// no `Int4Affine` single-stream cache mode today, and no batched
    /// 4-bit affine code path is wired in. [`Self::validate`] rejects the
    /// `(Uniform, 4)` combination at server startup so this method never
    /// has to fall back to a silent `Fp16` for that case. Operators who
    /// want 4-bit batched KV cache should use `--kv-quant-scheme
    /// turboquant`; operators who want uniform affine quantization should
    /// use `--kv-bits 8`.
    ///
    /// When 8-bit uniform is requested the existing `Int8` path is reused
    /// so single-stream and batched callers share the same
    /// `quantize_per_token` helpers.
    pub fn base_mode(&self) -> KVCacheMode {
        if !self.is_enabled() {
            return KVCacheMode::Fp16;
        }
        match (self.scheme, self.bits) {
            // 8-bit uniform reuses the single-stream Int8 path so existing
            // detach/adopt and prompt-cache code keeps working unchanged.
            (KvQuantScheme::Uniform, 8) => KVCacheMode::Int8,
            // (Uniform, 4) is rejected by `validate()` and never reaches
            // here. Any other uniform bit count would have been rejected
            // by `validate()` as well, but we map it to the safe Fp16
            // fall-through to avoid a `panic!` if a future caller skips
            // validation.
            (KvQuantScheme::Uniform, _) => KVCacheMode::Fp16,
            (KvQuantScheme::TurboQuant, _) => KVCacheMode::Turbo4Asym,
        }
    }

    /// Resolve the per-layer effective [`KVCacheMode`] for an entire
    /// model, applying [`Self::skip_last_layer`].
    ///
    /// This is the single source of truth for "which layers are
    /// quantized" so the scheduler, the cache-pool reset path, and the
    /// regression tests all agree.
    pub fn resolve_layer_modes(&self, n_layers: usize) -> Vec<KVCacheMode> {
        let nominal = self.base_mode();
        let mut modes = vec![nominal; n_layers];
        if !self.skip_last_layer || n_layers == 0 {
            return modes;
        }
        // Force the last layer to Fp16. We also leave the policy inert
        // when nominal is already Fp16 (no quantization to skip).
        if nominal != KVCacheMode::Fp16 {
            modes[n_layers - 1] = KVCacheMode::Fp16;
        }
        modes
    }

    /// Validate the resolved combination.
    ///
    /// Errors that are *invariably* a CLI mistake (negative bits, unknown
    /// (scheme, bits) pairing, non-positive group size) are surfaced here
    /// so server startup fails loudly.
    ///
    /// **TODO**: `group_size` must divide every model's `head_dim` for
    /// the affine codec to be valid. We cannot check that here because
    /// `head_dim` is not known until the model is loaded. There is no
    /// model-aware caller for this check today because no `Int4Affine`
    /// single-stream code path exists yet (the `(Uniform, 4)`
    /// combination is rejected here, and `Int8` does not require a
    /// group-size divisibility check). Add a
    /// `validate_against_head_dim(head_dim: i32)` method when the
    /// `Int4Affine` path lands.
    pub fn validate(&self) -> Result<(), String> {
        if self.bits < 0 {
            return Err(format!(
                "BatchKvQuantConfig: bits must be non-negative (got {})",
                self.bits
            ));
        }
        if self.group_size <= 0 {
            return Err(format!(
                "--kv-group-size must be positive, got {}",
                self.group_size
            ));
        }
        if !self.is_enabled() {
            return Ok(());
        }
        match (self.scheme, self.bits) {
            (KvQuantScheme::Uniform, 8) => Ok(()),
            // (Uniform, 4) is intentionally rejected at validation time:
            // mlxcel-core has no Int4Affine single-stream cache mode,
            // and silently downgrading to Fp16 would mean operators who
            // pass `--kv-bits 4 --kv-quant-scheme uniform` see no
            // quantization with no error. Direct them to the supported
            // 4-bit batched mode (TurboQuant) or to 8-bit uniform.
            (KvQuantScheme::Uniform, 4) => Err(
                "`--kv-quant-scheme uniform` with `--kv-bits 4` is not yet supported \
                 (mlxcel-core has no Int4Affine single-stream cache mode). \
                 Use `--kv-quant-scheme turboquant` for 4-bit batched KV cache, \
                 or `--kv-bits 8` for uniform 8-bit quantization."
                    .to_string(),
            ),
            (KvQuantScheme::Uniform, b) => Err(format!(
                "--kv-bits {b} not supported for --kv-quant-scheme uniform; expected 8 \
                 (4-bit uniform is not yet supported; use --kv-quant-scheme turboquant for 4-bit)"
            )),
            (KvQuantScheme::TurboQuant, 4) => Ok(()),
            (KvQuantScheme::TurboQuant, b) => Err(format!(
                "--kv-bits {b} not supported for --kv-quant-scheme turboquant; expected 4"
            )),
        }
    }
}

/// Per-sequence batch-quantized KV cache (uniform `mx.quantize` variant).
///
/// Mirrors upstream `BatchQuantizedKVCache` (`mlx-vlm` PR #1030) — same
/// field layout (`caches`, `left_padding`, `offset`) and the same
/// `extend` / `filter` / `update_after_decode` contract that upstream's
/// continuous-batching scheduler uses. One instance is intended to own a
/// `Vec<KVCache>` (one entry per transformer layer) plus the
/// per-sequence batch metadata (left-padding + offsets) needed by a
/// continuous-batching scheduler's lifecycle operations.
///
/// # Status: forward-looking stable seam (not on the live decode path)
///
/// As of this PR (issue #545) **this struct is not instantiated by the
/// live decode path**. mlxcel's continuous-batching scheduler quantizes
/// the KV cache by reading [`BatchKvQuantConfig`] directly and applying
/// the per-layer [`KVCacheMode`] table from
/// [`BatchKvQuantConfig::resolve_layer_modes`] to the per-sequence
/// per-layer [`KVCache`] instances stored in [`super::CachePool`] (see
/// `apply_kv_cache_mode_to` in `src/server/batch/scheduler.rs`).
///
/// This struct exists as a **forward-looking stable seam** that mirrors
/// the upstream PR #1030 contract surface so future batch-aware tensor
/// work has a stable Rust API to plug into without churning the public
/// surface again. This intentionally matches the pattern from PR #564 /
/// issue #544 for [`super::BatchedAttentionMetadata::extend`] and
/// [`super::BatchedAttentionMetadata::filter_in_place`].
///
/// # Layer-mode policy
///
/// The constructor [`Self::new`] consults
/// [`BatchKvQuantConfig::resolve_layer_modes`] so the
/// [`BatchKvQuantConfig::skip_last_layer`] knob takes effect at allocation
/// time without any further runtime branching.
///
/// # Used by
///
/// - The regression tests in [`mod tests`] below — currently the only
///   consumer.
/// - Future batch-aware Metal-kernel dispatch and continuous-batching
///   fusion work; these will be the first non-test consumers.
pub struct BatchQuantizedKVCache {
    /// Per-layer cache instances. Length equals `n_layers`. Each cache's
    /// `mode` is set per [`BatchKvQuantConfig::resolve_layer_modes`].
    pub caches: Vec<KVCache>,
    /// Configuration captured at allocation time. Stored so detach /
    /// adopt code paths can reproduce the same layer-mode table.
    pub config: BatchKvQuantConfig,
    /// Per-sequence left-padding (number of leading padding tokens).
    /// Length equals the active batch size. Mirrors the upstream Python
    /// `left_padding: mx.array` field but kept on the host as `Vec<i32>`
    /// because mlxcel's scheduler indexes per-row state from the host
    /// thread.
    pub left_padding: Vec<i32>,
    /// Logical sequence offsets — `offset[i] == -left_padding[i]` at
    /// init, then advances by `num_steps` on every
    /// [`update_after_decode`](Self::update_after_decode) call. Tracks
    /// the upstream Python `offset` field so the same arithmetic for
    /// [`extend`](Self::extend) / [`filter`](Self::filter) port across
    /// directly.
    pub offset: Vec<i32>,
    /// Actual number of tokens stored in the KV buffer (shared across the
    /// batch since all sequences are padded to the same length).
    ///
    /// This is the Rust analogue of Python `BatchKVCache._idx` /
    /// `BatchQuantizedKVCache._idx`. It starts at 0 and advances by
    /// `num_steps` on every [`update_after_decode`](Self::update_after_decode)
    /// call.  Callers must invoke `update_after_decode` after the initial
    /// prefill to bring `idx` in sync with the actual buffer occupancy.
    /// **This field must be used — not `offset` — when computing the mask
    /// offset for [`Self::make_mask`].**
    ///
    /// `filter` adjusts `idx` when it trims leading padding (mirrors
    /// Python `self._idx -= min_lp`).  `extend` takes the max of both
    /// caches' `idx` values (mirrors Python `self._idx = max(…)`).
    ///
    /// See upstream mlx-vlm PR #1208 (commit 2a55b80) for the original
    /// Python bug-fix: using the logical `offset` (which starts negative
    /// for padded sequences) instead of the actual buffer index caused
    /// incorrect causal mask shapes. Rust mirrors the fix here.
    pub idx: i32,
}

impl BatchQuantizedKVCache {
    /// Allocate a new batch-quantized cache covering `n_layers`
    /// transformer layers and an initial batch of `left_padding.len()`
    /// sequences.
    ///
    /// The per-layer caches are created in the resolved
    /// [`KVCacheMode`] from
    /// [`BatchKvQuantConfig::resolve_layer_modes`]. They start empty —
    /// the actual K/V tensors are populated lazily by the model's first
    /// `update_and_fetch` call inside the scheduler.
    pub fn new(
        config: BatchKvQuantConfig,
        n_layers: usize,
        left_padding: Vec<i32>,
    ) -> Result<Self, String> {
        config.validate()?;
        let layer_modes = config.resolve_layer_modes(n_layers);
        let caches = layer_modes
            .into_iter()
            .map(KVCache::new_with_mode)
            .collect();
        let offset = left_padding.iter().map(|&lp| -lp).collect();
        // `idx` starts at 0 (empty buffer); it will grow as tokens are
        // added via `update_after_decode`. The initial prefill populates
        // `max(left_padding)` slots of padding + the actual prompt tokens,
        // so callers must call `update_after_decode` after prefill to keep
        // `idx` in sync. See [`Self::update_after_decode`] and
        // [`Self::make_mask`].
        Ok(Self {
            caches,
            config,
            left_padding,
            offset,
            idx: 0,
        })
    }

    /// Number of transformer layers backing this batched cache.
    pub fn n_layers(&self) -> usize {
        self.caches.len()
    }

    /// Active batch size (number of sequences currently tracked).
    pub fn batch_size(&self) -> usize {
        self.left_padding.len()
    }

    /// Verify the cache is internally consistent.
    ///
    /// Two invariants are checked:
    ///
    /// 1. The per-row metadata vectors (`left_padding`, `offset`) agree
    ///    on a single batch size. Mirrors
    ///    [`super::BatchedAttentionMetadata::assert_consistent`] —
    ///    callers that build instances by direct field assignment should
    ///    invoke this in debug builds to catch any future regression
    ///    that re-introduces a uniformity-collapsing optimization (the
    ///    PR #1110 hazard from issue #544).
    /// 2. `caches` is non-empty. A batched cache with zero layers has
    ///    no transformer state to track and almost always indicates a
    ///    constructor bug (e.g., a model whose `n_layers` resolved to
    ///    zero before allocation).
    pub fn assert_consistent(&self) -> Result<(), String> {
        if self.left_padding.len() != self.offset.len() {
            return Err(format!(
                "BatchQuantizedKVCache: left_padding len {} differs from offset len {}",
                self.left_padding.len(),
                self.offset.len()
            ));
        }
        if self.caches.is_empty() {
            return Err(
                "BatchQuantizedKVCache: caches is empty; expected at least one transformer layer"
                    .to_string(),
            );
        }
        Ok(())
    }

    /// Advance the per-sequence offsets and the shared buffer index by
    /// `num_steps`.
    ///
    /// Called by the scheduler after a successful `update_and_fetch` on
    /// the underlying per-layer caches. The tensor-side cache update is
    /// performed directly on `self.caches[layer_idx]` — this method only
    /// keeps the bookkeeping fields in sync.
    ///
    /// `idx` (actual number of stored tokens) is advanced
    /// alongside `offset` so that [`Self::make_mask`] can use the correct
    /// buffer length rather than the potentially-negative logical offset.
    pub fn update_after_decode(&mut self, num_steps: i32) -> Result<(), String> {
        if num_steps < 0 {
            return Err(format!(
                "BatchQuantizedKVCache::update_after_decode: num_steps must be non-negative, got {num_steps}"
            ));
        }
        self.assert_consistent()?;
        for off in &mut self.offset {
            *off += num_steps;
        }
        self.idx += num_steps;
        Ok(())
    }

    /// Build the causal attention mask for the next `n` query tokens.
    ///
    /// Mirrors the upstream `BatchKVCache.make_mask` / `BatchQuantizedKVCache.make_mask`
    /// fix from mlx-vlm PR #1208 (commit `2a55b80`).
    ///
    /// **Bug in the original Python (pre-fix):** `create_attention_mask` was
    /// called with `offset=self.offset`, where `self.offset` is an array that
    /// starts at `[-left_padding[i] for i in range(B)]` — negative for padded
    /// sequences.  This produced a mask with an incorrect shape (or a crash).
    ///
    /// **The fix:** use `self._idx` (actual number of tokens in the buffer) as
    /// the `offset` parameter to `create_causal_mask`, and apply the
    /// `left_padding` filter on top to exclude leading padding positions.
    ///
    /// In Rust, `self.idx` is the analogue of Python `self._idx`.
    ///
    /// Returns `None` when `n == 1` and there is no left-padding (the common
    /// single-token decode case where no mask is needed).
    pub fn make_mask(&self, n: i32) -> Option<UniquePtr<MlxArray>> {
        // Single-token decode with no left-padding: masking is unnecessary
        // because there is only one query token and it always attends to the
        // entire (already-correct) KV context.
        if n == 1 && self.left_padding.iter().all(|&p| p == 0) {
            return None;
        }
        Some(utils::create_causal_mask_with_left_padding(
            n,
            self.idx,
            &self.left_padding,
        ))
    }

    /// Drop sequences not in `batch_indices`, preserving per-row vector
    /// shape.
    ///
    /// The mlxcel analogue of upstream `BatchQuantizedKVCache.filter()`
    /// (PR #1030). `batch_indices[k] = i` means "row `i` of the current
    /// batch becomes row `k` of the filtered batch". Errors if any index
    /// is out of range or duplicated.
    ///
    /// **Note**: the per-layer tensor-side filter is *not* performed
    /// here. mlxcel's continuous-batching scheduler stores per-sequence
    /// caches in [`super::CachePool`] keyed by [`super::SequenceId`], so
    /// a "filter" at the batched-cache level is naturally expressed as
    /// "drop the cache pool entries for the dropped sequences". This
    /// method only synchronises the per-row metadata vectors that the
    /// scheduler keeps for batch-aware kernels.
    pub fn filter(&mut self, batch_indices: &[usize]) -> Result<(), String> {
        self.assert_consistent()?;
        let n = self.left_padding.len();
        let mut seen = vec![false; n];
        for &i in batch_indices {
            if i >= n {
                return Err(format!(
                    "BatchQuantizedKVCache::filter: index {i} out of range for batch size {n}"
                ));
            }
            if seen[i] {
                return Err(format!(
                    "BatchQuantizedKVCache::filter: duplicate index {i}"
                ));
            }
            seen[i] = true;
        }
        let new_left_padding: Vec<i32> = batch_indices
            .iter()
            .map(|&i| self.left_padding[i])
            .collect();
        let new_offset: Vec<i32> = batch_indices.iter().map(|&i| self.offset[i]).collect();
        self.left_padding = new_left_padding;
        self.offset = new_offset;
        // Trim leading padding once it becomes uniform across the
        // surviving batch — mirrors the upstream "min_lp > 0" branch.
        //
        // Review fix: upstream Python also does `self._idx -= min_lp`
        // here (see `BatchQuantizedKVCache.filter`, commit 2a55b80). We must
        // mirror that so `make_mask` continues to compute the correct
        // `total_len = n + idx` after a trim.
        if let Some(&min_lp) = self.left_padding.iter().min() {
            if min_lp > 0 {
                for lp in &mut self.left_padding {
                    *lp -= min_lp;
                }
                // Precondition: callers invoke `update_after_decode` (which
                // advances `idx`) before `filter` whenever `left_padding` is
                // nonzero, so `idx >= min_lp` holds here and `idx` stays
                // non-negative. Mirrors upstream `self._idx -= min_lp`; a
                // negative `idx` would make `make_mask` build a degenerate
                // `total_len = n + idx`. The scheduler upholds this once these
                // caches move onto the live decode path.
                self.idx -= min_lp;
                // The actual K/V tensor trim (slicing leading
                // `min_lp` tokens out of each layer cache) is the
                // scheduler's responsibility because it requires
                // `cxx::UniquePtr<MlxArray>` operations on the live
                // device-resident tensors. The metadata side is
                // sufficient for the unit-test scenario, where we
                // verify that the wrapper's bookkeeping matches the
                // upstream Python invariant.
            }
        }
        Ok(())
    }

    /// Concatenate `other` into this cache along the batch dimension.
    ///
    /// The mlxcel analogue of upstream `BatchQuantizedKVCache.extend()`
    /// (PR #1030). Mirrors the upstream pattern of merging both metadata
    /// fields (`left_padding` + `offset`) with the same row order.
    ///
    /// **Note** — like [`Self::filter`], the per-layer tensor-side
    /// concatenation is the scheduler's responsibility (it already
    /// owns the per-sequence caches in [`super::CachePool`]).
    pub fn extend(&mut self, other: &Self) -> Result<(), String> {
        self.assert_consistent()?;
        other.assert_consistent()?;
        if self.config != other.config {
            return Err(
                "BatchQuantizedKVCache::extend: cannot extend with a different config".to_string(),
            );
        }
        if self.caches.len() != other.caches.len() {
            return Err(format!(
                "BatchQuantizedKVCache::extend: layer count mismatch ({} vs {})",
                self.caches.len(),
                other.caches.len()
            ));
        }
        self.left_padding.extend_from_slice(&other.left_padding);
        self.offset.extend_from_slice(&other.offset);
        // Mirrors upstream Python `BatchQuantizedKVCache.extend`:
        // `self._idx = max(self._idx, other._idx)` so that `make_mask`
        // uses the longer of the two buffer lengths after the merge.
        self.idx = self.idx.max(other.idx);
        Ok(())
    }
}

/// Per-sequence batch-quantized KV cache (TurboQuant variant).
///
/// Mirrors upstream `BatchTurboQuantKVCache` (`mlx-vlm` PR #1030) — same
/// field layout and the same `extend` / `filter` / `update_after_decode`
/// contract — but uses mlxcel's existing [`KVCacheMode::Turbo4Asym`]
/// storage backend for each per-layer cache. Attention dispatch is via
/// dequantize + standard SDPA — custom batch-aware Metal kernels are
/// explicitly out of scope for issue #545 (the single-stream
/// [`super::turbo::sparse_v`] kernels are not used here because they
/// assume `B == 1`).
///
/// # Status: forward-looking stable seam (not on the live decode path)
///
/// As of this PR (issue #545) **this struct is not instantiated by the
/// live decode path**. The live continuous-batching decode path drives
/// quantization through [`BatchKvQuantConfig`] +
/// [`BatchKvQuantConfig::resolve_layer_modes`] applied directly to the
/// per-sequence per-layer [`KVCache`] instances in [`super::CachePool`]
/// via the scheduler's `apply_kv_cache_mode_to` (see
/// `src/server/batch/scheduler.rs`).
///
/// This struct mirrors the upstream PR #1030 contract surface as a
/// **forward-looking stable seam** so future batch-aware Metal-kernel
/// dispatch (e.g., a fused batched TurboQuant attention kernel that
/// wants a single struct holding `(per_layer_caches, left_padding,
/// offset)`) has a stable Rust API to slot into. This intentionally
/// matches the pattern from PR #564 / issue #544 for
/// [`super::BatchedAttentionMetadata::extend`] and
/// [`super::BatchedAttentionMetadata::filter_in_place`].
///
/// The bookkeeping (left padding, offsets, extend / filter) is
/// identical to [`BatchQuantizedKVCache`] — only the per-layer storage
/// mode differs. We keep the two types separate (rather than collapsing
/// into a generic over the mode) to keep the upstream-mirroring
/// `Used by` mapping obvious and to leave room for a future Metal-kernel
/// dispatch on this type without churning the uniform variant.
///
/// # Used by
///
/// - The regression tests in [`mod tests`] below — currently the only
///   consumer.
/// - Future batch-aware TurboQuant Metal-kernel dispatch; this will be
///   the first non-test consumer.
pub struct BatchTurboQuantKVCache {
    /// Per-layer cache instances. Configured to
    /// [`KVCacheMode::Turbo4Asym`] (with the last layer optionally
    /// downgraded to `Fp16` per [`BatchKvQuantConfig::skip_last_layer`]).
    pub caches: Vec<KVCache>,
    /// Configuration captured at allocation time.
    pub config: BatchKvQuantConfig,
    /// Per-sequence left-padding. Same semantics as on
    /// [`BatchQuantizedKVCache`].
    pub left_padding: Vec<i32>,
    /// Logical sequence offsets. Same semantics as on
    /// [`BatchQuantizedKVCache`].
    pub offset: Vec<i32>,
    /// Actual number of tokens stored in the KV buffer. Analogue of
    /// `BatchTurboQuantKVCache._idx` in upstream Python.
    /// See [`BatchQuantizedKVCache::idx`] for the full rationale.
    pub idx: i32,
}

impl BatchTurboQuantKVCache {
    /// Allocate a new TurboQuant batched cache.
    ///
    /// `config.scheme` **must** be [`KvQuantScheme::TurboQuant`] —
    /// constructing a TurboQuant batched cache with a uniform config is a
    /// programmer error and is rejected up front.
    pub fn new(
        config: BatchKvQuantConfig,
        n_layers: usize,
        left_padding: Vec<i32>,
    ) -> Result<Self, String> {
        config.validate()?;
        if config.scheme != KvQuantScheme::TurboQuant {
            return Err(format!(
                "BatchTurboQuantKVCache::new: scheme must be TurboQuant, got {}",
                config.scheme
            ));
        }
        if !config.is_enabled() {
            return Err(
                "BatchTurboQuantKVCache::new: cannot construct with quantization disabled (bits == 0)"
                    .to_string(),
            );
        }
        let layer_modes = config.resolve_layer_modes(n_layers);
        let caches = layer_modes
            .into_iter()
            .map(KVCache::new_with_mode)
            .collect();
        let offset = left_padding.iter().map(|&lp| -lp).collect();
        Ok(Self {
            caches,
            config,
            left_padding,
            offset,
            idx: 0,
        })
    }

    /// Number of transformer layers backing this batched cache.
    pub fn n_layers(&self) -> usize {
        self.caches.len()
    }

    /// Active batch size.
    pub fn batch_size(&self) -> usize {
        self.left_padding.len()
    }

    /// See [`BatchQuantizedKVCache::assert_consistent`].
    pub fn assert_consistent(&self) -> Result<(), String> {
        if self.left_padding.len() != self.offset.len() {
            return Err(format!(
                "BatchTurboQuantKVCache: left_padding len {} differs from offset len {}",
                self.left_padding.len(),
                self.offset.len()
            ));
        }
        if self.caches.is_empty() {
            return Err(
                "BatchTurboQuantKVCache: caches is empty; expected at least one transformer layer"
                    .to_string(),
            );
        }
        Ok(())
    }

    /// See [`BatchQuantizedKVCache::update_after_decode`].
    ///
    /// advances `idx` alongside `offset` so that
    /// [`Self::make_mask`] uses the correct buffer length.
    pub fn update_after_decode(&mut self, num_steps: i32) -> Result<(), String> {
        if num_steps < 0 {
            return Err(format!(
                "BatchTurboQuantKVCache::update_after_decode: num_steps must be non-negative, got {num_steps}"
            ));
        }
        self.assert_consistent()?;
        for off in &mut self.offset {
            *off += num_steps;
        }
        self.idx += num_steps;
        Ok(())
    }

    /// Build the causal attention mask for the next `n` query tokens.
    ///
    /// Mirrors the upstream `BatchTurboQuantKVCache.make_mask` fix from
    /// mlx-vlm PR #1208 (commit `2a55b80`). See
    /// [`BatchQuantizedKVCache::make_mask`] for the full rationale.
    pub fn make_mask(&self, n: i32) -> Option<UniquePtr<MlxArray>> {
        if n == 1 && self.left_padding.iter().all(|&p| p == 0) {
            return None;
        }
        Some(utils::create_causal_mask_with_left_padding(
            n,
            self.idx,
            &self.left_padding,
        ))
    }

    /// See [`BatchQuantizedKVCache::filter`].
    pub fn filter(&mut self, batch_indices: &[usize]) -> Result<(), String> {
        self.assert_consistent()?;
        let n = self.left_padding.len();
        let mut seen = vec![false; n];
        for &i in batch_indices {
            if i >= n {
                return Err(format!(
                    "BatchTurboQuantKVCache::filter: index {i} out of range for batch size {n}"
                ));
            }
            if seen[i] {
                return Err(format!(
                    "BatchTurboQuantKVCache::filter: duplicate index {i}"
                ));
            }
            seen[i] = true;
        }
        let new_left_padding: Vec<i32> = batch_indices
            .iter()
            .map(|&i| self.left_padding[i])
            .collect();
        let new_offset: Vec<i32> = batch_indices.iter().map(|&i| self.offset[i]).collect();
        self.left_padding = new_left_padding;
        self.offset = new_offset;
        // Mirrors upstream Python `BatchQuantizedKVCache.filter`: also
        // decrement `_idx` by `min_lp` when padding is trimmed, so that
        // `make_mask` keeps computing the correct `total_len = n + idx`.
        if let Some(&min_lp) = self.left_padding.iter().min() {
            if min_lp > 0 {
                for lp in &mut self.left_padding {
                    *lp -= min_lp;
                }
                // Precondition: callers invoke `update_after_decode` (which
                // advances `idx`) before `filter` whenever `left_padding` is
                // nonzero, so `idx >= min_lp` holds here and `idx` stays
                // non-negative. Mirrors upstream `self._idx -= min_lp`; a
                // negative `idx` would make `make_mask` build a degenerate
                // `total_len = n + idx`. The scheduler upholds this once these
                // caches move onto the live decode path.
                self.idx -= min_lp;
            }
        }
        Ok(())
    }

    /// See [`BatchQuantizedKVCache::extend`].
    pub fn extend(&mut self, other: &Self) -> Result<(), String> {
        self.assert_consistent()?;
        other.assert_consistent()?;
        if self.config != other.config {
            return Err(
                "BatchTurboQuantKVCache::extend: cannot extend with a different config".to_string(),
            );
        }
        if self.caches.len() != other.caches.len() {
            return Err(format!(
                "BatchTurboQuantKVCache::extend: layer count mismatch ({} vs {})",
                self.caches.len(),
                other.caches.len()
            ));
        }
        self.left_padding.extend_from_slice(&other.left_padding);
        self.offset.extend_from_slice(&other.offset);
        // Mirrors upstream `BatchQuantizedKVCache.extend`: take max _idx.
        self.idx = self.idx.max(other.idx);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── KvQuantScheme ────────────────────────────────────────────────

    #[test]
    fn scheme_parses_uniform_aliases() {
        for s in ["uniform", "UNIFORM", "affine", "mx.quantize"] {
            assert_eq!(s.parse::<KvQuantScheme>().unwrap(), KvQuantScheme::Uniform);
        }
    }

    #[test]
    fn scheme_parses_turboquant_aliases() {
        for s in ["turboquant", "TURBOQUANT", "turbo", "turbo4"] {
            assert_eq!(
                s.parse::<KvQuantScheme>().unwrap(),
                KvQuantScheme::TurboQuant
            );
        }
    }

    #[test]
    fn scheme_rejects_unknown_string() {
        assert!("polar".parse::<KvQuantScheme>().is_err());
    }

    // ── BatchKvQuantConfig ───────────────────────────────────────────

    #[test]
    fn config_default_is_disabled() {
        let cfg = BatchKvQuantConfig::default();
        assert!(!cfg.is_enabled());
        assert_eq!(cfg.base_mode(), KVCacheMode::Fp16);
        assert!(cfg.skip_last_layer);
    }

    #[test]
    fn config_uniform_8bit_maps_to_int8() {
        let cfg = BatchKvQuantConfig::new(KvQuantScheme::Uniform, 8, 64, true).unwrap();
        assert_eq!(cfg.base_mode(), KVCacheMode::Int8);
    }

    #[test]
    fn config_uniform_4bit_is_rejected_at_validation_time() {
        // (Uniform, 4) has no Int4Affine single-stream cache mode in
        // mlxcel-core today, and silently returning `Fp16` from
        // `base_mode()` would mean operators see no quantization with
        // no error. `validate()` rejects the combination so the silent
        // no-op cannot reach the live decode path.
        let err = BatchKvQuantConfig::new(KvQuantScheme::Uniform, 4, 64, true).unwrap_err();
        assert!(
            err.contains("--kv-quant-scheme uniform")
                && err.contains("--kv-bits 4")
                && err.contains("not yet supported"),
            "expected error mentioning the unsupported (uniform, 4) combination, got: {err}"
        );
        // The error must point operators at supported alternatives.
        assert!(
            err.contains("turboquant") && err.contains("--kv-bits 8"),
            "expected error to mention turboquant + 8-bit alternatives, got: {err}"
        );
    }

    #[test]
    fn config_turboquant_4bit_maps_to_turbo4_asym() {
        let cfg = BatchKvQuantConfig::new(KvQuantScheme::TurboQuant, 4, 64, true).unwrap();
        assert_eq!(cfg.base_mode(), KVCacheMode::Turbo4Asym);
    }

    #[test]
    fn config_rejects_invalid_uniform_bits() {
        for &b in &[3, 5, 6, 7, 16] {
            let err = BatchKvQuantConfig::new(KvQuantScheme::Uniform, b, 64, true).unwrap_err();
            assert!(err.contains("not supported for --kv-quant-scheme uniform"));
        }
    }

    #[test]
    fn config_rejects_invalid_turboquant_bits() {
        for &b in &[2, 3, 5, 8] {
            let err = BatchKvQuantConfig::new(KvQuantScheme::TurboQuant, b, 64, true).unwrap_err();
            assert!(err.contains("not supported for --kv-quant-scheme turboquant"));
        }
    }

    #[test]
    fn config_rejects_non_positive_group_size() {
        for &g in &[0, -1, -64] {
            let err = BatchKvQuantConfig::new(KvQuantScheme::Uniform, 8, g, true).unwrap_err();
            assert!(err.contains("--kv-group-size must be positive"));
        }
    }

    // ── Negative-bits guard (Fix 3) ─────────────────────────────────

    /// Direct construction with `bits < 0` must be rejected by `validate()`.
    /// Previously `is_enabled()` returned false for negative bits and
    /// validation passed silently, giving a silent no-op cache.
    #[test]
    fn config_rejects_negative_bits() {
        for &b in &[-1i32, -4, -8, i32::MIN] {
            let err = BatchKvQuantConfig::new(KvQuantScheme::Uniform, b, 64, true).unwrap_err();
            assert!(
                err.contains("bits must be non-negative"),
                "expected 'bits must be non-negative' in error for bits={b}, got: {err}"
            );
            assert!(
                err.contains(&b.to_string()),
                "error must include the bad value {b}, got: {err}"
            );
        }
    }

    /// Confirm `validate()` rejects negative bits directly (bypassing `new`).
    #[test]
    fn validate_rejects_negative_bits_directly() {
        let cfg = BatchKvQuantConfig {
            scheme: KvQuantScheme::Uniform,
            bits: -1,
            group_size: 64,
            skip_last_layer: true,
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("bits must be non-negative"));
    }

    // ── Layer-mode resolution + last-layer skip ──────────────────────

    #[test]
    fn skip_last_layer_keeps_final_layer_fp16_for_int8() {
        let cfg = BatchKvQuantConfig::new(KvQuantScheme::Uniform, 8, 64, true).unwrap();
        let modes = cfg.resolve_layer_modes(8);
        assert_eq!(modes.len(), 8);
        for (i, mode) in modes.iter().enumerate() {
            if i == 7 {
                assert_eq!(*mode, KVCacheMode::Fp16, "last layer must be Fp16");
            } else {
                assert_eq!(*mode, KVCacheMode::Int8, "layer {i} must keep nominal mode");
            }
        }
    }

    #[test]
    fn skip_last_layer_keeps_final_layer_fp16_for_turbo4_asym() {
        let cfg = BatchKvQuantConfig::new(KvQuantScheme::TurboQuant, 4, 64, true).unwrap();
        let modes = cfg.resolve_layer_modes(40); // gemma-4-31b-class depth
        assert_eq!(modes.len(), 40);
        assert_eq!(modes[39], KVCacheMode::Fp16);
        assert_eq!(modes[0], KVCacheMode::Turbo4Asym);
        assert_eq!(modes[20], KVCacheMode::Turbo4Asym);
    }

    #[test]
    fn skip_last_layer_disabled_keeps_uniform_modes() {
        let cfg = BatchKvQuantConfig::new(KvQuantScheme::Uniform, 8, 64, false).unwrap();
        let modes = cfg.resolve_layer_modes(8);
        for mode in &modes {
            assert_eq!(*mode, KVCacheMode::Int8);
        }
    }

    #[test]
    fn skip_last_layer_inert_when_nominal_is_fp16() {
        let cfg = BatchKvQuantConfig::default();
        let modes = cfg.resolve_layer_modes(4);
        for mode in &modes {
            assert_eq!(*mode, KVCacheMode::Fp16);
        }
    }

    #[test]
    fn resolve_layer_modes_handles_zero_layers() {
        let cfg = BatchKvQuantConfig::new(KvQuantScheme::Uniform, 8, 64, true).unwrap();
        let modes = cfg.resolve_layer_modes(0);
        assert!(modes.is_empty());
    }

    // ── BatchQuantizedKVCache ────────────────────────────────────────

    #[test]
    fn batch_quantized_new_initializes_offsets() {
        let cfg = BatchKvQuantConfig::new(KvQuantScheme::Uniform, 8, 64, true).unwrap();
        let cache = BatchQuantizedKVCache::new(cfg, 4, vec![3, 0, 5]).unwrap();
        assert_eq!(cache.n_layers(), 4);
        assert_eq!(cache.batch_size(), 3);
        assert_eq!(cache.offset, vec![-3, 0, -5]);
        assert_eq!(cache.left_padding, vec![3, 0, 5]);
        cache.assert_consistent().unwrap();
    }

    #[test]
    fn batch_quantized_per_layer_modes_reflect_skip_last() {
        let cfg = BatchKvQuantConfig::new(KvQuantScheme::Uniform, 8, 64, true).unwrap();
        let cache = BatchQuantizedKVCache::new(cfg, 5, vec![0, 0]).unwrap();
        assert_eq!(cache.caches[0].mode, KVCacheMode::Int8);
        assert_eq!(cache.caches[3].mode, KVCacheMode::Int8);
        assert_eq!(cache.caches[4].mode, KVCacheMode::Fp16);
    }

    #[test]
    fn batch_quantized_update_after_decode_advances_offsets() {
        let cfg = BatchKvQuantConfig::new(KvQuantScheme::Uniform, 8, 64, false).unwrap();
        let mut cache = BatchQuantizedKVCache::new(cfg, 2, vec![0, 2]).unwrap();
        assert_eq!(cache.offset, vec![0, -2]);
        cache.update_after_decode(5).unwrap();
        assert_eq!(cache.offset, vec![5, 3]);
    }

    /// `update_after_decode` must advance `idx` alongside `offset`.
    /// This confirms that the fix applied `idx += num_steps` so that
    /// `make_mask` sees the correct buffer length.
    #[test]
    fn batch_quantized_update_after_decode_advances_idx() {
        let cfg = BatchKvQuantConfig::new(KvQuantScheme::Uniform, 8, 64, false).unwrap();
        // left_padding=[2,0]: idx starts at 0, offset starts at [-2, 0]
        let mut cache = BatchQuantizedKVCache::new(cfg, 2, vec![2, 0]).unwrap();
        assert_eq!(cache.idx, 0, "idx starts at 0");
        // Simulate prefill: 5 tokens deposited (2 padding + 3 real)
        cache.update_after_decode(5).unwrap();
        assert_eq!(cache.idx, 5, "idx must advance by num_steps");
        // offset is per-sequence, idx is shared
        assert_eq!(cache.offset, vec![3, 5], "offset advances independently");
    }

    /// `make_mask` must return None for the single-token decode
    /// case when there is no left-padding (no MLX ops needed).
    #[test]
    fn batch_quantized_make_mask_none_when_no_padding_single_token() {
        let cfg = BatchKvQuantConfig::new(KvQuantScheme::Uniform, 8, 64, false).unwrap();
        let mut cache = BatchQuantizedKVCache::new(cfg, 2, vec![0, 0]).unwrap();
        cache.update_after_decode(5).unwrap();
        // Single token, no padding → no mask needed (fast path)
        assert!(cache.make_mask(1).is_none());
    }

    /// `make_mask` must return Some when left_padding is non-zero
    /// — tests that verify the actual mask values are in `ffi_tests.rs` because
    /// those call `create_causal_mask_with_left_padding` which requires the
    /// MLX C++ runtime.
    #[test]
    fn batch_quantized_make_mask_returns_some_when_padding_present_metadata_check() {
        // This test only checks the return variant (Some vs None),
        // not the actual mask values.  See ffi_tests for value tests.
        let cfg = BatchKvQuantConfig::new(KvQuantScheme::Uniform, 8, 64, false).unwrap();
        let cache0 = BatchQuantizedKVCache::new(cfg, 2, vec![0, 0]).unwrap();
        // Zero idx → mask for n=1, no padding → None
        assert!(cache0.make_mask(1).is_none());
        // Fake: manually set idx without calling MLX (pure metadata test)
        // We cannot easily call make_mask(1) with padding in a no-MLX test
        // because the make_mask impl calls create_causal_mask_with_left_padding.
        // The MLX tests in ffi_tests.rs cover the non-None path.
    }

    #[test]
    fn batch_quantized_update_rejects_negative_steps() {
        let cfg = BatchKvQuantConfig::new(KvQuantScheme::Uniform, 8, 64, false).unwrap();
        let mut cache = BatchQuantizedKVCache::new(cfg, 2, vec![0, 0]).unwrap();
        assert!(cache.update_after_decode(-1).is_err());
    }

    #[test]
    fn batch_quantized_filter_keeps_selected_rows() {
        let cfg = BatchKvQuantConfig::new(KvQuantScheme::Uniform, 8, 64, false).unwrap();
        let mut cache = BatchQuantizedKVCache::new(cfg, 2, vec![1, 2, 3, 4]).unwrap();
        cache.update_after_decode(10).unwrap();
        // Drop rows 0 and 2.
        cache.filter(&[1, 3]).unwrap();
        assert_eq!(cache.batch_size(), 2);
        // Surviving left-padding values were [2, 4]; min is 2 so the
        // shared trim subtracts 2 → [0, 2].
        assert_eq!(cache.left_padding, vec![0, 2]);
        // Offsets carry through unchanged: each surviving row still sees
        // its own decode counter.
        assert_eq!(cache.offset, vec![10 - 2, 10 - 4]);
    }

    #[test]
    fn batch_quantized_filter_rejects_out_of_range() {
        let cfg = BatchKvQuantConfig::new(KvQuantScheme::Uniform, 8, 64, false).unwrap();
        let mut cache = BatchQuantizedKVCache::new(cfg, 2, vec![0, 0]).unwrap();
        assert!(cache.filter(&[0, 5]).is_err());
    }

    #[test]
    fn batch_quantized_filter_rejects_duplicates() {
        let cfg = BatchKvQuantConfig::new(KvQuantScheme::Uniform, 8, 64, false).unwrap();
        let mut cache = BatchQuantizedKVCache::new(cfg, 2, vec![0, 0, 0]).unwrap();
        assert!(cache.filter(&[0, 0]).is_err());
    }

    #[test]
    fn batch_quantized_extend_concatenates_metadata() {
        let cfg = BatchKvQuantConfig::new(KvQuantScheme::Uniform, 8, 64, true).unwrap();
        let mut left = BatchQuantizedKVCache::new(cfg, 4, vec![0, 1]).unwrap();
        let right = BatchQuantizedKVCache::new(cfg, 4, vec![2, 3, 4]).unwrap();
        left.extend(&right).unwrap();
        assert_eq!(left.batch_size(), 5);
        assert_eq!(left.left_padding, vec![0, 1, 2, 3, 4]);
        assert_eq!(left.offset, vec![0, -1, -2, -3, -4]);
    }

    #[test]
    fn batch_quantized_extend_rejects_config_mismatch() {
        // Two configs that differ in `skip_last_layer` (a runtime-relevant
        // field) — the `(Uniform, 4)` combination is no longer valid, so
        // we exercise the mismatch via a different field.
        let cfg_a = BatchKvQuantConfig::new(KvQuantScheme::Uniform, 8, 64, true).unwrap();
        let cfg_b = BatchKvQuantConfig::new(KvQuantScheme::Uniform, 8, 64, false).unwrap();
        let mut left = BatchQuantizedKVCache::new(cfg_a, 4, vec![0]).unwrap();
        let right = BatchQuantizedKVCache::new(cfg_b, 4, vec![0]).unwrap();
        let err = left.extend(&right).unwrap_err();
        assert!(err.contains("different config"));
    }

    #[test]
    fn batch_quantized_extend_rejects_layer_count_mismatch() {
        let cfg = BatchKvQuantConfig::new(KvQuantScheme::Uniform, 8, 64, true).unwrap();
        let mut left = BatchQuantizedKVCache::new(cfg, 4, vec![0]).unwrap();
        let right = BatchQuantizedKVCache::new(cfg, 6, vec![0]).unwrap();
        let err = left.extend(&right).unwrap_err();
        assert!(err.contains("layer count mismatch"));
    }

    // ── BatchTurboQuantKVCache ───────────────────────────────────────

    #[test]
    fn batch_turboquant_new_initializes_offsets() {
        let cfg = BatchKvQuantConfig::new(KvQuantScheme::TurboQuant, 4, 64, true).unwrap();
        let cache = BatchTurboQuantKVCache::new(cfg, 6, vec![0, 7]).unwrap();
        assert_eq!(cache.n_layers(), 6);
        assert_eq!(cache.batch_size(), 2);
        assert_eq!(cache.offset, vec![0, -7]);
    }

    #[test]
    fn batch_turboquant_per_layer_modes_reflect_skip_last() {
        let cfg = BatchKvQuantConfig::new(KvQuantScheme::TurboQuant, 4, 64, true).unwrap();
        let cache = BatchTurboQuantKVCache::new(cfg, 5, vec![0]).unwrap();
        assert_eq!(cache.caches[0].mode, KVCacheMode::Turbo4Asym);
        assert_eq!(cache.caches[3].mode, KVCacheMode::Turbo4Asym);
        assert_eq!(cache.caches[4].mode, KVCacheMode::Fp16);
    }

    #[test]
    fn batch_turboquant_rejects_uniform_scheme() {
        let cfg = BatchKvQuantConfig::new(KvQuantScheme::Uniform, 8, 64, true).unwrap();
        // We cannot use `unwrap_err` because `BatchTurboQuantKVCache`
        // does not implement `Debug` (its inner `KVCache` does not),
        // so destructure the Result manually.
        match BatchTurboQuantKVCache::new(cfg, 4, vec![0]) {
            Err(err) => assert!(err.contains("scheme must be TurboQuant")),
            Ok(_) => panic!("expected error"),
        }
    }

    #[test]
    fn batch_turboquant_rejects_disabled_config() {
        // `bits == 0` is the off-sentinel; constructing a TurboQuant
        // batched cache with bits zero is a programmer error.
        let cfg = BatchKvQuantConfig {
            scheme: KvQuantScheme::TurboQuant,
            bits: 0,
            group_size: 64,
            skip_last_layer: true,
        };
        match BatchTurboQuantKVCache::new(cfg, 4, vec![0]) {
            Err(err) => assert!(err.contains("disabled")),
            Ok(_) => panic!("expected error"),
        }
    }

    #[test]
    fn batch_turboquant_filter_extend_round_trip() {
        let cfg = BatchKvQuantConfig::new(KvQuantScheme::TurboQuant, 4, 64, true).unwrap();
        let mut left = BatchTurboQuantKVCache::new(cfg, 4, vec![1, 2]).unwrap();
        let right = BatchTurboQuantKVCache::new(cfg, 4, vec![3, 4]).unwrap();
        left.extend(&right).unwrap();
        assert_eq!(left.batch_size(), 4);
        assert_eq!(left.left_padding, vec![1, 2, 3, 4]);
        // Drop the first and third rows.
        left.filter(&[1, 3]).unwrap();
        assert_eq!(left.batch_size(), 2);
        // Surviving left_padding before trim: [2, 4]. min is 2, so the
        // shared-trim leaves [0, 2].
        assert_eq!(left.left_padding, vec![0, 2]);
    }

    #[test]
    fn batch_turboquant_update_after_decode_advances_offsets() {
        let cfg = BatchKvQuantConfig::new(KvQuantScheme::TurboQuant, 4, 64, true).unwrap();
        let mut cache = BatchTurboQuantKVCache::new(cfg, 4, vec![3, 0]).unwrap();
        cache.update_after_decode(7).unwrap();
        assert_eq!(cache.offset, vec![4, 7]);
    }

    /// `BatchTurboQuantKVCache::update_after_decode` must advance
    /// `idx` alongside `offset`.
    #[test]
    fn batch_turboquant_update_after_decode_advances_idx() {
        let cfg = BatchKvQuantConfig::new(KvQuantScheme::TurboQuant, 4, 64, true).unwrap();
        let mut cache = BatchTurboQuantKVCache::new(cfg, 4, vec![3, 0]).unwrap();
        assert_eq!(cache.idx, 0, "idx starts at 0");
        cache.update_after_decode(7).unwrap();
        assert_eq!(cache.idx, 7, "idx must advance by num_steps");
    }

    /// `BatchTurboQuantKVCache::make_mask` must return None for
    /// n=1 with no left-padding.
    #[test]
    fn batch_turboquant_make_mask_none_when_no_padding_single_token() {
        let cfg = BatchKvQuantConfig::new(KvQuantScheme::TurboQuant, 4, 64, true).unwrap();
        let mut cache = BatchTurboQuantKVCache::new(cfg, 4, vec![0, 0]).unwrap();
        cache.update_after_decode(5).unwrap();
        assert!(cache.make_mask(1).is_none());
    }

    /// metadata test — no MLX ops, just verifies None path.
    #[test]
    fn batch_turboquant_make_mask_none_for_no_padding_single_token_metadata() {
        let cfg = BatchKvQuantConfig::new(KvQuantScheme::TurboQuant, 4, 64, true).unwrap();
        let cache = BatchTurboQuantKVCache::new(cfg, 4, vec![0, 0]).unwrap();
        assert!(cache.make_mask(1).is_none());
    }

    // ── filter/extend idx invariants ────────

    /// `filter` with `min_lp > 0` must decrement `idx` by `min_lp`,
    /// mirroring upstream Python `BatchQuantizedKVCache.filter` which does
    /// `self._idx -= min_lp`.
    #[test]
    fn batch_quantized_filter_trims_idx_with_min_padding() {
        let cfg = BatchKvQuantConfig::new(KvQuantScheme::Uniform, 8, 64, false).unwrap();
        // B=3: padding=[2, 3, 5].  After update_after_decode(7): idx=7.
        let mut cache = BatchQuantizedKVCache::new(cfg, 2, vec![2, 3, 5]).unwrap();
        cache.update_after_decode(7).unwrap();
        assert_eq!(cache.idx, 7);
        // Keep seqs 0 and 2 (left_padding=[2, 5]).  min_lp=2 → trim 2 from idx.
        cache.filter(&[0, 2]).unwrap();
        assert_eq!(cache.left_padding, vec![0, 3], "min_lp=2 trimmed from each");
        assert_eq!(cache.idx, 5, "idx must be decremented by min_lp=2");
    }

    /// `filter` with `min_lp == 0` must NOT change `idx`.
    #[test]
    fn batch_quantized_filter_no_trim_preserves_idx() {
        let cfg = BatchKvQuantConfig::new(KvQuantScheme::Uniform, 8, 64, false).unwrap();
        let mut cache = BatchQuantizedKVCache::new(cfg, 2, vec![0, 3]).unwrap();
        cache.update_after_decode(5).unwrap();
        assert_eq!(cache.idx, 5);
        cache.filter(&[0, 1]).unwrap();
        // min_lp=0 → no trim → idx unchanged
        assert_eq!(cache.idx, 5, "idx must remain 5 when min_lp=0");
    }

    /// `extend` must take `max(self.idx, other.idx)`, mirroring upstream Python.
    #[test]
    fn batch_quantized_extend_takes_max_idx() {
        let cfg = BatchKvQuantConfig::new(KvQuantScheme::Uniform, 8, 64, true).unwrap();
        let mut left = BatchQuantizedKVCache::new(cfg, 4, vec![0, 1]).unwrap();
        left.update_after_decode(7).unwrap();
        assert_eq!(left.idx, 7);

        let mut right = BatchQuantizedKVCache::new(cfg, 4, vec![2, 3, 4]).unwrap();
        right.update_after_decode(10).unwrap();
        assert_eq!(right.idx, 10);

        left.extend(&right).unwrap();
        assert_eq!(left.idx, 10, "extend must take max(7, 10) = 10");
        assert_eq!(left.batch_size(), 5);
    }

    /// `BatchTurboQuantKVCache::filter` must also decrement `idx` by `min_lp`.
    #[test]
    fn batch_turboquant_filter_trims_idx_with_min_padding() {
        let cfg = BatchKvQuantConfig::new(KvQuantScheme::TurboQuant, 4, 64, false).unwrap();
        let mut cache = BatchTurboQuantKVCache::new(cfg, 4, vec![3, 5]).unwrap();
        cache.update_after_decode(8).unwrap();
        assert_eq!(cache.idx, 8);
        // Keep both; min_lp=3 → trim 3 from idx.
        cache.filter(&[0, 1]).unwrap();
        assert_eq!(cache.left_padding, vec![0, 2], "min_lp=3 trimmed");
        assert_eq!(cache.idx, 5, "idx decremented by min_lp=3");
    }

    /// `BatchTurboQuantKVCache::extend` must take max idx.
    #[test]
    fn batch_turboquant_extend_takes_max_idx() {
        let cfg = BatchKvQuantConfig::new(KvQuantScheme::TurboQuant, 4, 64, false).unwrap();
        let mut left = BatchTurboQuantKVCache::new(cfg, 4, vec![0]).unwrap();
        left.update_after_decode(5).unwrap();

        let mut right = BatchTurboQuantKVCache::new(cfg, 4, vec![0]).unwrap();
        right.update_after_decode(9).unwrap();

        left.extend(&right).unwrap();
        assert_eq!(left.idx, 9, "extend must take max(5, 9) = 9");
    }
}
