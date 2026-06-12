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

//! Attention cache state machines shared by text and VLM families.
//!
//! These types keep cache growth, rewinding, and sliding-window semantics in
//! one place so `layers.rs` can focus on layer math while models continue to
//! import the same cache types via `mlxcel_core::layers`.
//!
//! # KV Cache Quantization
//!
//! `KVCache` optionally stores keys/values in INT8 to reduce memory by ~50%.
//! Enable via `KVCacheMode::Int8` at construction time. The `update_and_fetch`
//! method always returns FP16 tensors (dequantized on read), so the attention
//! computation is unaffected.
//!
//! `KVCacheMode::Turbo4Asym` keeps the K side as FP16
//! and compresses the V side to 4-bit PolarQuant indices plus per-token norms,
//! reducing total KV memory by ~26% at long context. The compressed V buffers
//! live in dedicated sidecar fields (`v_packed`, `v_norms`) and the
//! quantize/dequantize helpers in [`turbo::quant`] handle the math.
//!
//! `KVCacheMode::Turbo3Asym` is the 3-bit sibling of
//! `Turbo4Asym` — same Fp16-K + asymmetric layout, but the V side uses the
//! 8-centroid (3-bit) Lloyd-Max codebook and the 24-bit-grouped packing
//! layout from [`turbo::pack3`]. Compression climbs to ~5.1× total KV
//! savings (vs ~3.8× for `Turbo4Asym`) at the cost of a slightly larger
//! V-reconstruction error. Symmetric Turbo3 is an explicit non-goal of
//! this PR — see [`turbo::quant3`] for rationale.
//!
//! `KVCacheMode::Turbo4` extends Turbo4Asym to a
//! **symmetric** 4-bit K + 4-bit V layout, reducing KV memory by ~73% at
//! long context. The K side mirrors the V side bit-for-bit but uses an
//! independent pair of sign vectors (see [`turbo::quant::K_SEED_OFFSET`]).
//! Symmetric Turbo4 is **dangerous** on dense Q4_K_M weights — see the
//! [`turbo::allowlist`] module for the per-model allowlist that gates
//! end-user opt-in.
//!
//! `KVCacheMode::Turbo4Delegated` extends the
//! asymmetric mode with a V-only hot/cold split that recovers 97–100% of FP16
//! decode speed at long context. The K side is a single unified FP16 buffer
//! that grows in lockstep with `offset` (identical shape contract to
//! `KVCacheMode::Fp16`); only V is split into a packed cold body plus an FP16
//! hot ring. During prefill tokens accumulate in the standard
//! FP16 `keys`/`values` buffers (zero overhead). On the first decode step the
//! V hot body is "folded" into cold storage by quantizing it into packed
//! Turbo4 in `v_packed`/`v_norms`. The K side does *not* move — it stays in
//! the unified `keys` buffer at positions `[0..offset]`. Subsequent decode
//! tokens flow through the unified K buffer (slice-update at `offset`) and a
//! small pre-allocated FP16 V hot ring. When the V hot ring crosses
//! [`turbo::DELEGATED_HOT_THRESHOLD`] tokens, the oldest hot V block is
//! quantized and appended to the cold V sidecars (the K side just observes
//! `cold_offset += block`). SDPA always reads FP16: reads return
//! `slice(keys, 0, offset)` for K and `concat(dequant(v_packed), hot_V)` for
//! V. The K side has no per-step concat — that was the dominant residual
//! cost vs FP16 mode (~7 ms/step at 4 K context). See
//! https://github.com/TheTom/turboquant_plus/blob/main/README.md §"MLX Framework Port" for the
//! original architecture.
//!
//! Setting `MLXCEL_TURBO4_DELEGATED_FP16_FAST_PATH=1` switches this mode to a
//! delegated FP16 working-set variant: packed cold V sidecars are still built,
//! but `values` remains a unified FP16 V buffer and attention reads it through
//! the native SDPA path. This mirrors TurboQuant+'s delegated KVCache speed
//! path and is intended as an opt-in performance comparison, not the default
//! compressed-only memory target.

pub mod batch_quant;
mod detach;
mod paged;
mod paged_detach;
#[cfg(test)]
#[path = "cache/paged_pool_tests.rs"]
mod paged_pool_tests;
#[cfg(test)]
#[path = "cache/paged_turbo_tests.rs"]
mod paged_turbo_tests;
#[cfg(test)]
#[path = "cache/sparse_v_tests.rs"]
mod sparse_v_tests;
pub mod turbo;
#[cfg(test)]
#[path = "cache/turbo_tests.rs"]
mod turbo_tests;

pub use batch_quant::{
    BatchKvQuantConfig, BatchQuantizedKVCache, BatchTurboQuantKVCache, KvQuantScheme,
    DEFAULT_KV_GROUP_SIZE,
};
pub use detach::{DetachedCacheSet, DetachedHandle, DetachedKVCache, DetachedRotatingKVCache};
pub use paged::{
    GatheredKv, PagedBlockId, PagedBlockPool, PagedCacheStats, PagedKvLayout, PagedLayerState,
    PagedSequenceState,
};
pub use paged_detach::DetachedPagedCacheSet;

use std::cell::{Ref, RefCell, RefMut};
use std::rc::Rc;

use crate::concatenate;
use crate::dtype;
use crate::ffi;
use crate::ffi::MlxArray;
use crate::ops::divide_scalar;
use cxx::UniquePtr;

/// One paged pool block's live K/V contents, used to ferry block data across
/// the distributed serialization boundary (#125). Carries the ORIGIN node's
/// physical block id and layer; the decode node remaps to fresh ids on restore.
pub struct PagedBlockContents {
    pub block_id: PagedBlockId,
    pub layer_idx: usize,
    pub keys: UniquePtr<MlxArray>,
    pub values: UniquePtr<MlxArray>,
}

fn direct_prefill_cache_store_enabled() -> bool {
    std::env::var("MLXCEL_ENABLE_DIRECT_PREFILL_CACHE_STORE").is_ok()
}

/// Check that all `KVCache` entries in `caches` support cache trimming.
///
/// Mirrors the upstream mlx-lm `can_trim_prompt_cache` function
/// (https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py). Speculative decoding requires a trimmable
/// cache so it can rewind cache entries after a draft-token rejection.
///
/// All current `KVCache` mode variants (Fp16, Int8, Turbo4Asym, Turbo4,
/// Turbo3Asym, Turbo4Delegated) return `true` from [`KVCache::is_trimmable`].
/// This function exists so new non-trimmable cache types can be detected
/// early — before the speculative decode loop silently corrupts the cache.
///
/// Used by: speculative decoding entry-point validation (`speculative.rs`)
pub fn can_trim_prompt_cache(caches: &[KVCache]) -> bool {
    caches.iter().all(|c| c.is_trimmable())
}

/// Storage mode for KV cache tensors.
///
/// Controls the on-device representation of accumulated key/value tensors.
/// The public `update_and_fetch` interface always returns FP16 regardless of
/// the chosen mode, so attention kernels are unaffected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KVCacheMode {
    /// Standard half-precision storage (default). No quantization overhead.
    #[default]
    Fp16,
    /// Per-token INT8 absmax quantization. Reduces KV cache memory by ~50%
    /// at the cost of small quantization error per token.
    Int8,
    /// Asymmetric Fp16-K + Turbo4-V. K side stays in FP16; V side uses 4-bit
    /// PolarQuant with Walsh–Hadamard rotation for ~26% net KV memory savings
    /// at long context..
    Turbo4Asym,
    /// Asymmetric Fp16-K + Turbo3-V. K side stays in FP16; V side uses 3-bit
    /// PolarQuant with Walsh–Hadamard rotation for ~5.1× total KV memory
    /// savings (vs ~3.8× for `Turbo4Asym`) at the cost of a slightly higher
    /// V-reconstruction error. The 3-bit packing is awkward (8 coords share
    /// 3 bytes / 24 bits — see [`turbo::pack3`]), so the dequant path runs
    /// a host round-trip rather than the 4-bit path's pure on-device unpack.
    /// Symmetric Turbo3 is **not** offered in this mode ('s "Quality–compression tradeoff control" explicitly defers it). See
    Turbo3Asym,
    /// Symmetric Turbo4-K + Turbo4-V. Both K and V use 4-bit PolarQuant with
    /// independent Walsh–Hadamard rotations for ~73% net KV memory savings
    /// at long context. **Dangerous on dense Q4_K_M weights** — gated by
    /// the per-model allowlist in [`turbo::allowlist`]. See
    Turbo4,
    /// Delegated hot/cold split on top of `Turbo4Asym`. Prefill stores raw
    /// FP16; on the first decode step the prefilled body is folded into cold
    /// storage (FP16 cold K + Turbo4 packed cold V); subsequent decode tokens
    /// flow through a small FP16 hot tail with zero-alloc slice-update. SDPA
    /// always reads FP16. Targets ≥97%-of-FP16 decode speed at 4K and ≥95%
    /// at 16K on M5 Max..
    /// `MLXCEL_TURBO4_DELEGATED_FP16_FAST_PATH=1` keeps an FP16 V working set
    /// for native-SDPA decode while still maintaining packed sidecars.
    Turbo4Delegated,
}

impl std::str::FromStr for KVCacheMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "fp16" | "float16" => Ok(Self::Fp16),
            "int8" | "i8" => Ok(Self::Int8),
            // Both spellings are accepted: the canonical user-facing string
            // ("fp16+turbo4") makes the asymmetric K/V split explicit, while
            // "turbo4-asym" is a shorter alias for scripts and tests.
            "turbo4-asym" | "fp16+turbo4" => Ok(Self::Turbo4Asym),
            // Turbo3Asym (3-bit V, asymmetric only —). Same alias
            // pattern as Turbo4Asym: "fp16+turbo3" is the canonical string,
            // "turbo3-asym" / "turbo3" are shorter forms used in scripts and
            // tests. There is intentionally no symmetric "turbo3" alias —
            // symmetric 3-bit on dense Q4_K_M weights is catastrophic and
            // explicitly out of scope for this PR (see [`turbo::quant3`]).
            "turbo3-asym" | "fp16+turbo3" | "turbo3" => Ok(Self::Turbo3Asym),
            // Symmetric Turbo4 (4-bit K + 4-bit V). The canonical user-facing
            // string is "turbo4"; "turbo4-sym" is an explicit alias used in
            // tests/scripts when readers benefit from the K/V symmetry being
            // spelled out.
            "turbo4" | "turbo4-sym" => Ok(Self::Turbo4),
            // Delegated hot/cold split (B7). Same K=FP16 +
            // V=Turbo4 contract as Turbo4Asym but uses the delegated KVCache
            // architecture for decode speed at long context.
            "turbo4-delegated" | "fp16+turbo4-delegated" => Ok(Self::Turbo4Delegated),
            other => Err(format!(
                "unknown kv-cache-mode \"{other}\"; expected one of \
                 \"fp16\", \"int8\", \"fp16+turbo4\" (alias \"turbo4-asym\"), \
                 \"turbo4\" (alias \"turbo4-sym\"), \
                 \"turbo4-delegated\" (alias \"fp16+turbo4-delegated\"), \
                 \"fp16+turbo3\" (aliases \"turbo3-asym\" / \"turbo3\")"
            )),
        }
    }
}

impl std::fmt::Display for KVCacheMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fp16 => f.write_str("fp16"),
            Self::Int8 => f.write_str("int8"),
            Self::Turbo4Asym => f.write_str("fp16+turbo4"),
            Self::Turbo4 => f.write_str("turbo4"),
            Self::Turbo4Delegated => f.write_str("turbo4-delegated"),
            Self::Turbo3Asym => f.write_str("fp16+turbo3"),
        }
    }
}

// ---------------------------------------------------------------------------
// INT8 quantization helpers
// ---------------------------------------------------------------------------

/// Quantize a tensor to INT8 using per-token absmax scaling.
///
/// `x` has shape `[B, H, T, D]` where T is typically 1 (one new token).
/// Returns `(x_int8, scale)` where:
/// - `x_int8`: `[B, H, T, D]` INT8
/// - `scale`:  `[B, H, T, 1]` FP16 — the absmax / 127.0 for each token
///
/// Used by: QuantizedKVCache (INT8 mode of KVCache)
fn quantize_per_token(x: &MlxArray) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
    // Compute per-token absmax: reduce over last dim (head_dim), keepdims
    let abs_x = ffi::abs(x);
    let absmax = ffi::max_axis(&abs_x, -1, true); // [B, H, T, 1]

    // scale = absmax / 127.0  (FP16 to match cache dtype)
    let scale = divide_scalar(&absmax, 127.0); // [B, H, T, 1]

    // Avoid divide-by-zero: replace zero scales with 1.0
    let one = ffi::full_f32(&[1], 1.0, dtype::FLOAT16);
    let safe_scale = ffi::maximum(&scale, &one);

    // x_int8 = round(x / safe_scale).clamp(-128, 127)
    let x_div = ffi::divide(x, &safe_scale);
    let x_rounded = ffi::round(&x_div);
    let lo = ffi::full_f32(&[1], -128.0, ffi::array_dtype(x));
    let hi = ffi::full_f32(&[1], 127.0, ffi::array_dtype(x));
    let x_clipped = ffi::clip(&x_rounded, &lo, &hi);
    let x_int8 = ffi::astype(&x_clipped, dtype::INT8);

    (x_int8, safe_scale)
}

/// Dequantize INT8 tensor back to FP16 for attention computation.
///
/// `x_int8`: `[B, H, L, D]` INT8
/// `scale`:  `[B, H, L, 1]` FP16
/// Returns:  `[B, H, L, D]` FP16
///
/// Used by: QuantizedKVCache (INT8 mode of KVCache)
fn dequantize(x_int8: &MlxArray, scale: &MlxArray) -> UniquePtr<MlxArray> {
    let x_fp16 = ffi::astype(x_int8, dtype::FLOAT16);
    ffi::multiply(&x_fp16, scale)
}

/// Default Turbo4Asym sign-vector seed when the caller does not supply one.
///
/// Hard-coded so two cache instances built independently with the same V
/// `head_dim` yield identical rotations — important for reproducibility and
/// for the detach/adopt round-trip when the Turbo4 sidecars travel without
/// the originating cache.
pub(crate) const TURBO_DEFAULT_SEED: u32 = 0x7B4_70404; // "TUR" 0x474 + B2 issue 474

/// KV Cache for attention layers.
///
/// Uses pre-allocated buffers with slice_update for O(1) per-token updates,
/// matching Python mlx-lm's KVCache implementation. Buffers grow by `step`
/// slots at a time (default 256) to amortize allocation cost.
///
/// When `mode` is `KVCacheMode::Int8`, keys and values are stored as INT8
/// tensors with per-token scale factors. `update_and_fetch` always returns
/// FP16 (dequantized) so attention kernels see standard tensors.
///
/// When `mode` is `KVCacheMode::Turbo4Asym`, the keys buffer stays FP16
/// while the values buffer is replaced by `v_packed` (nibble-packed 4-bit
/// PolarQuant indices) plus `v_norms` (per-token fp16 L2 norms). The
/// per-cache `turbo_params` field carries the deterministic sign vectors
/// and shared codebook used by the quantize/dequantize helpers in
/// [`turbo::quant`]. The standard `values` field stays `None` in this mode.
///
/// When `mode` is `KVCacheMode::Turbo4` (symmetric), the K
/// buffers are *also* replaced by `k_packed` + `k_norms` sidecars; the
/// `keys` field stays `None`. The same `turbo_params` instance carries an
/// independent K-side sign-vector pair so the K and V quantization noise
/// is uncorrelated.
///
/// Used by: All transformer models (Llama, Qwen, Gemma, etc.). The shared
/// `update`, `update_and_fetch`, `trim`, `nbytes`, and
/// `bytes_per_reserved_token` methods dispatch over `mode` and support
/// `KVCacheMode::Fp16`, `KVCacheMode::Int8`, `KVCacheMode::Turbo4Asym`
/// and `KVCacheMode::Turbo4` without per-model branching.
pub struct KVCache {
    pub keys: Option<UniquePtr<MlxArray>>,
    pub values: Option<UniquePtr<MlxArray>>,
    /// Monotonically increasing absolute write position. Used as the RoPE
    /// position for new K/Q tokens and as the upper bound of the live window.
    /// Once a token has been written at position `p`, this value never
    /// decreases through any operation that preserves on-device K — in
    /// particular `trim_front` increments [`Self::live_start`]
    /// instead of decrementing `offset`, so the relative position
    /// `offset - i` between a cached K at monotonic slot `i` and a freshly
    /// rotated Q stays invariant after the live window is bounded.
    pub offset: i32,
    /// Monotonically increasing start position of the live window inside the
    /// monotonic position space. Slot `i` in the on-device `keys` / `values`
    /// buffer corresponds to monotonic position `live_start + i`. The live
    /// window length is `offset - live_start`; attention reads only this
    /// slice. Always `0` for callers that never trim the head — and **must**
    /// stay `0` for Turbo4*/Turbo3* modes since the packed sidecars carry
    /// per-token rotation state that head-trim cannot safely rebuild
    /// (see [`Self::trim_front`]).
    ///
    /// Together with `offset`, this preserves the RoPE invariant under
    /// the `--max-kv-size` cap: K stored at buffer slot `i` was
    /// rotated at monotonic position `live_start + i` *at write time*, Q is
    /// rotated at the current monotonic `offset`, and attention sees the
    /// correct relative position `offset - (live_start + i)`.
    pub(crate) live_start: i32,
    step: i32,
    /// Quantization mode for stored keys/values.
    pub mode: KVCacheMode,
    // INT8-mode scale factors: [B, H, L, 1] FP16, None when mode == Fp16
    pub(crate) key_scales: Option<UniquePtr<MlxArray>>,
    pub(crate) val_scales: Option<UniquePtr<MlxArray>>,
    // Turbo4Asym/Turbo4-mode V-side packed indices: [B, H, L, head_dim/2] u8
    pub(crate) v_packed: Option<UniquePtr<MlxArray>>,
    // Turbo4Asym/Turbo4-mode V-side per-token norms: [B, H, L, 1] fp16
    pub(crate) v_norms: Option<UniquePtr<MlxArray>>,
    // Turbo4-V precomputed kernel rescale `norm[t] / |y_hat[t]|`.
    // Same `[B, H, L, 1]` fp16 shape as `v_norms` and slice/concat/trimmed
    // lockstep with it. Populated only on Turbo4Asym / Turbo4 / Turbo4Delegated
    // updates — Turbo3Asym leaves this `None` because the 3-bit V kernel does
    // not exist. Consumed by `attention_sparse_v_turbo4_fused` to skip the
    // per-cache-token threadgroup tree reduction that previously dominated
    // decode latency at 4 K context (A/B).
    pub(crate) v_rescale: Option<UniquePtr<MlxArray>>,
    // Turbo4-mode (symmetric) K-side packed indices: [B, H, L, head_dim/2] u8
    pub(crate) k_packed: Option<UniquePtr<MlxArray>>,
    // Turbo4-mode (symmetric) K-side per-token norms: [B, H, L, 1] fp16
    pub(crate) k_norms: Option<UniquePtr<MlxArray>>,
    /// Cached PolarQuant params (sign vectors + codebook) for Turbo4* modes
    /// (Turbo4Asym / Turbo4 symmetric / Turbo4Delegated).
    ///
    /// Lazily initialised on the first Turbo4 update once the V `head_dim` is
    /// known. The deterministic seed is derived from the cache's `turbo_seed`
    /// so detach/adopt and re-construction reproduce the same rotation. In
    /// symmetric `Turbo4` mode the same params carry both V and K sign
    /// vectors (`signs1`/`signs2` for V, `k_signs1`/`k_signs2` for K).
    pub(crate) turbo_params: Option<turbo::TurboQuantParams>,
    /// Cached PolarQuant params for `Turbo3Asym`. The 3-bit
    /// codebook (8 centroids) is incompatible with the 4-bit `turbo_params`
    /// codebook so a separate field is required. Lazily initialised on the
    /// first `Turbo3Asym` update once the V `head_dim` is known. Stays
    /// `None` for non-Turbo3 modes.
    pub(crate) turbo3_params: Option<turbo::quant3::TurboQuantParams3>,
    /// Deterministic seed for the Turbo4 sign vectors. Set at construction
    /// time so detach/adopt round-trip without recomputing rotations.
    pub(crate) turbo_seed: u32,
    /// Number of tokens currently folded into cold V storage (Turbo4Delegated
    /// only).
    ///
    /// **** — K side is no longer split into hot/cold. The unified
    /// `keys` buffer holds all `offset` tokens at FP16 (same contract as
    /// `KVCacheMode::Fp16`). Only V has a hot/cold split: the cold body lives
    /// in the packed `v_packed`/`v_norms`/`v_rescale` sidecars (length
    /// `cold_offset`) and the hot ring lives in `values` (length
    /// `offset - cold_offset`).
    ///
    /// Invariant: `cold_offset >= 0 && cold_offset <= offset`. In all
    /// non-delegated modes `cold_offset` stays at 0 and `v_packed`/`v_norms`
    /// behave per the per-mode invariants documented above.
    pub(crate) cold_offset: i32,
    /// Hot-tail fold threshold (Turbo4Delegated only). Mutable so test harness
    /// and benchmarks can configure it without going through the env var.
    pub(crate) hot_threshold: i32,
    /// Opt-in TurboQuant+ delegated-style path for `Turbo4Delegated`.
    ///
    /// When true, `values` is a unified FP16 V buffer rather than a hot ring.
    /// Packed cold V sidecars still advance with `cold_offset`, but decode
    /// attention reads the FP16 V slice through native SDPA.
    pub(crate) delegated_fp16_fast_path: bool,
    /// Packed sidecar maintenance cadence for the opt-in delegated FP16 fast
    /// path. Ignored unless `delegated_fp16_fast_path` is true.
    pub(crate) delegated_fp16_sidecar_policy: turbo::DelegatedFp16SidecarPolicy,
    /// Optional transparent pool backing.
    ///
    /// `None` for every dense cache (the default for all existing
    /// constructors), so the dense `update`/`update_and_fetch` path and every
    /// existing call site are byte-for-byte unchanged. When `Some`,
    /// [`Self::update_and_fetch`] routes K/V writes into a shared
    /// [`PagedBlockPool`] (via `write_prefill`) and reads the visible window
    /// back (via `gather_visible`) instead of touching the dense `keys`/
    /// `values` buffers. Built only by [`Self::new_paged`].
    ///
    /// This is the single-stream groundwork for the scheduler-driven paged KV
    /// cache (#121): one sequence, one pool, caches visited sequentially per
    /// layer, so the simple `Rc<RefCell<…>>` sharing is sound (see the Step-0
    /// `Send` analysis in the introducing PR — `KVCache` is never required to
    /// be `Send`/`Sync`).
    pub(crate) paged_backing: Option<PagedBacking>,
}

/// Shared handle that makes one [`KVCache`] write/read through a pooled paged
/// KV store instead of its own dense buffers.
///
/// All layers of a single sequence share the same `pool` and `state` handles
/// (cheap `Rc` clones); `layer_idx` selects this cache's slice of the
/// per-sequence [`PagedSequenceState`]. Interior mutability via `RefCell` is
/// required because [`KVCache::update_and_fetch`] needs `&mut` access to both
/// the pool (to append/allocate blocks) and the per-sequence state while the
/// model only hands the cache out behind `&mut [KVCache]` one layer at a time.
///
/// # Why `Rc<RefCell<…>>` and not raw pointers
///
/// `KVCache` is not required to be `Send`/`Sync` anywhere in the tree (no
/// `unsafe impl`, no `assert_send_sync::<KVCache>()`, and the only async-context
/// `Vec<KVCache>` — the pipeline stage service — is built and driven entirely
/// inside a single OS thread via `block_on` on a current-thread runtime, never
/// crossing a `tokio::spawn` boundary). The safe shared-ownership form is
/// therefore preferred over raw pointers.
#[derive(Clone)]
pub(crate) struct PagedBacking {
    pub(crate) pool: Rc<RefCell<PagedBlockPool>>,
    pub(crate) state: Rc<RefCell<PagedSequenceState>>,
    pub(crate) layer_idx: usize,
}

impl std::fmt::Debug for PagedBacking {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PagedBacking")
            .field("layer_idx", &self.layer_idx)
            .finish_non_exhaustive()
    }
}

impl KVCache {
    /// Create a new empty KV cache with default step size (256) and FP16 mode.
    pub fn new() -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
            live_start: 0,
            step: 256,
            mode: KVCacheMode::Fp16,
            key_scales: None,
            val_scales: None,
            v_packed: None,
            v_norms: None,
            v_rescale: None,
            k_packed: None,
            k_norms: None,
            turbo_params: None,
            turbo3_params: None,
            turbo_seed: TURBO_DEFAULT_SEED,
            cold_offset: 0,
            hot_threshold: turbo::DELEGATED_HOT_THRESHOLD,
            delegated_fp16_fast_path: turbo::delegated_fp16_fast_path_enabled(),
            delegated_fp16_sidecar_policy: turbo::delegated_fp16_sidecar_policy(),
            paged_backing: None,
        }
    }

    /// Create a new empty KV cache with the specified quantization mode.
    ///
    /// Use `KVCacheMode::Int8` to store accumulated keys/values in INT8 format.
    /// Use `KVCacheMode::Turbo4Asym` for asymmetric Fp16-K + Turbo4-V
    /// compression. Use `KVCacheMode::Turbo4` for symmetric
    /// Turbo4-K + Turbo4-V compression — note that this mode
    /// is dangerous on dense Q4_K_M weights and must be gated by the
    /// per-model allowlist in [`turbo::allowlist`] before being exposed
    /// to end users. The `update_and_fetch` method will transparently
    /// quantize incoming tensors and dequantize them on read, so callers
    /// always receive FP16.
    pub fn new_with_mode(mode: KVCacheMode) -> Self {
        Self::new_with_mode_and_seed(mode, TURBO_DEFAULT_SEED)
    }

    /// Like [`KVCache::new_with_mode`] but lets the caller pin a specific
    /// Turbo4 sign-vector seed.
    ///
    /// Production callers should prefer `new_with_mode` and let the cache
    /// pool layer disambiguate seeds across layers (see `make_caches` paths
    /// per model). The explicit seed is exposed for tests, benchmarks, and
    /// the detach/adopt path which must preserve the original seed.
    pub fn new_with_mode_and_seed(mode: KVCacheMode, turbo_seed: u32) -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
            live_start: 0,
            step: 256,
            mode,
            key_scales: None,
            val_scales: None,
            v_packed: None,
            v_norms: None,
            v_rescale: None,
            k_packed: None,
            k_norms: None,
            turbo_params: None,
            turbo3_params: None,
            turbo_seed,
            cold_offset: 0,
            hot_threshold: turbo::DELEGATED_HOT_THRESHOLD,
            delegated_fp16_fast_path: turbo::delegated_fp16_fast_path_enabled(),
            delegated_fp16_sidecar_policy: turbo::delegated_fp16_sidecar_policy(),
            paged_backing: None,
        }
    }

    /// Create a transparently pool-backed empty KV cache for one layer.
    ///
    /// The returned cache is `KVCacheMode::Fp16` with default step size, but
    /// instead of accumulating K/V in its own dense `keys`/`values` buffers it
    /// writes new tokens into the shared [`PagedBlockPool`] (`write_prefill`)
    /// and reads the visible window back (`gather_visible`) inside
    /// [`Self::update_and_fetch`]. All other dense state (`offset`,
    /// `live_start`, …) is still tracked so RoPE positions stay correct; the
    /// `keys`/`values` buffers simply remain `None`.
    ///
    /// `pool` and `state` are shared (cheap `Rc` clones) across every layer of
    /// the same sequence; `layer_idx` selects this cache's slice of the
    /// per-sequence [`PagedSequenceState`]. This is the single-stream (`B == 1`)
    /// verification path for the paged KV cache (#118/#119/#120) — the
    /// scheduler-driven multi-sequence wiring is #121.
    ///
    /// # Panics / errors
    ///
    /// This constructor never fails. Geometry mismatches (wrong `layer_idx`,
    /// non-`[1, n_kv_heads, n_new, head_dim]` K/V, `B != 1`) surface as a
    /// panic from [`Self::update_and_fetch`] on first write, because the model
    /// forward signature cannot return a `Result`.
    pub fn new_paged(
        pool: Rc<RefCell<PagedBlockPool>>,
        state: Rc<RefCell<PagedSequenceState>>,
        layer_idx: usize,
    ) -> Self {
        let mut cache = Self::new();
        cache.paged_backing = Some(PagedBacking {
            pool,
            state,
            layer_idx,
        });
        cache
    }

    /// Whether this cache writes/reads through a shared [`PagedBlockPool`]
    /// (built via [`Self::new_paged`]) instead of its own dense buffers.
    ///
    /// The batched-decode dispatch in each transformer model uses this to skip
    /// the native dense-pointer paged kernel (which reads `keys`/`values`, both
    /// `None` here) and fall through to the per-sequence `update_and_fetch`
    /// loop, whose pool intercept transparently writes to the pool and gathers
    /// the visible window. See the model `forward_split_attention` dispatch.
    #[inline]
    pub fn is_paged_backed(&self) -> bool {
        self.paged_backing.is_some()
    }

    /// Override the Turbo4Delegated hot-tail fold threshold for this cache.
    ///
    /// Test-only entry point for the unit/integration tests in `turbo_tests.rs`
    /// and the speed benchmarks. Production callers should leave the default
    /// [`turbo::DELEGATED_HOT_THRESHOLD`] in place; tuning this changes the
    /// fold cadence and therefore the speed/quality trade-off documented in
    /// https://github.com/TheTom/turboquant_plus/blob/main/README.md. Setting `threshold <= 0` is
    /// rejected so the fold path stays well-defined.
    pub fn set_hot_threshold(&mut self, threshold: i32) {
        if threshold > 0 {
            self.hot_threshold = threshold;
        }
    }

    /// Read the configured hot-tail fold threshold (Turbo4Delegated only;
    /// returns the default for other modes).
    pub fn hot_threshold(&self) -> i32 {
        self.hot_threshold
    }

    /// Number of tokens currently held in cold packed storage. Always 0 unless
    /// `mode == Turbo4Delegated`.
    pub fn cold_offset(&self) -> i32 {
        self.cold_offset
    }

    /// Number of tokens currently held in the FP16 hot tail. Equals
    /// `offset - cold_offset` and matches `seq_len()` when no folds have
    /// happened (i.e. during prefill or before the prefill→decode transition).
    pub fn hot_offset(&self) -> i32 {
        self.offset - self.cold_offset
    }

    /// Build all missing Turbo4Delegated FP16 fast-path packed V sidecars.
    ///
    /// The FP16 fast path keeps `values` as a unified V buffer for SDPA, so
    /// packed cold sidecars are not required to answer decode attention. This
    /// method exists for explicit preservation paths: pre-decode handoff under
    /// the conservative policy, detach / prompt-cache donation under the lazy
    /// policy, and future memory recovery hooks.
    ///
    /// Returns `true` only when a fold was actually performed.
    ///
    /// Used by: [`Self::compact_turbo4_delegated_fp16_sidecars_for_decode`]
    /// and [`Self::clone_handle`].
    pub fn compact_turbo4_delegated_fp16_sidecars(&mut self) -> bool {
        if self.mode != KVCacheMode::Turbo4Delegated || !self.delegated_fp16_fast_path {
            return false;
        }
        if self.offset <= self.cold_offset || self.values.is_none() {
            return false;
        }

        self.fold_unified_v_range_to_cold(self.cold_offset, self.offset - self.cold_offset);
        true
    }

    /// Pre-build Turbo4Delegated FP16 fast-path packed V sidecars after
    /// prefill and before the first decode forward when the sidecar policy is
    /// `predecode`.
    ///
    /// Used by: generator prefill→decode handoff paths and server batch
    /// scheduler `finish_prefill`.
    pub fn compact_turbo4_delegated_fp16_sidecars_for_decode(&mut self) -> bool {
        if self.delegated_fp16_sidecar_policy != turbo::DelegatedFp16SidecarPolicy::Predecode {
            return false;
        }
        self.compact_turbo4_delegated_fp16_sidecars()
    }

    /// Prepare Turbo4Delegated cache state before the first decode forward.
    ///
    /// In the compressed packed-V path this moves the initial prefill body
    /// fold out of the first single-token decode `update()`: V is quantized
    /// into cold packed sidecars immediately after prefill, the FP16 hot ring
    /// is reset empty, and the first decode step only appends its new token.
    /// Cold V therefore remains compressed for attention; the one-time fold
    /// is simply charged to the prefill->decode handoff instead of decode.
    ///
    /// In the opt-in FP16 working-set path, this preserves the existing
    /// sidecar-policy behavior: `predecode` folds packed sidecars here while
    /// `lazy` skips foreground compaction.
    ///
    /// Returns `true` only when a fold was performed.
    ///
    /// Used by: generator prefill->decode handoff paths and server batch
    /// scheduler `finish_prefill`.
    pub fn prepare_turbo4_delegated_for_decode(&mut self) -> bool {
        if self.mode != KVCacheMode::Turbo4Delegated {
            return false;
        }
        if self.delegated_fp16_fast_path {
            return self.compact_turbo4_delegated_fp16_sidecars_for_decode();
        }
        if self.cold_offset == 0 && self.offset > 0 && self.values.is_some() && self.keys.is_some()
        {
            self.fold_hot_to_cold_full();
            return true;
        }
        false
    }

    /// Check if cache is empty.
    ///
    /// Returns `true` only when no token has ever been written into the
    /// cache (`self.offset == 0`). After [`Self::trim_front`] drops the
    /// entire live window, the on-device buffers may be `None` but
    /// `self.offset` stays at its monotonic value so callers that branch
    /// on "is this the initial prefill?" (e.g. the fused causal prefill
    /// fast path in dense models, which hard-codes RoPE positions to
    /// `[0, l)`) do not silently engage on a post-trim cache where the
    /// next Q is rotated at a non-zero position.
    pub fn is_empty(&self) -> bool {
        // Symmetric Turbo4 has no `keys` buffer (K is packed into
        // `k_packed`), so check both. The cache is empty iff neither the
        // dense K buffer nor the packed K buffer is allocated AND no
        // monotonic write has happened.
        self.offset == 0 && self.keys.is_none() && self.k_packed.is_none()
    }

    /// Get current sequence length in cache.
    ///
    /// Reports the **live window length** — the number of tokens attention
    /// will actually see on the next forward pass. After [`Self::trim_front`]
    /// drops the oldest `n` tokens, this returns `offset - live_start`
    /// rather than the monotonic `offset` so callers that build masks /
    /// allocate per-token sidecars (e.g., paged accounting, server gauges,
    /// detach handles) see the post-trim length. Pre- callers that never
    /// trim observe identical behaviour because `live_start == 0`.
    pub fn seq_len(&self) -> i32 {
        self.offset - self.live_start
    }

    /// Number of live tokens currently visible in the cache (alias of
    /// [`Self::seq_len`], kept for call-sites that want to underline they
    /// are interested in the post-trim length rather than the monotonic
    /// position).
    #[inline]
    pub fn live_len(&self) -> i32 {
        self.offset - self.live_start
    }

    /// Buffer slot index where the next K/V token will be written.
    ///
    /// Slot `i` in the on-device buffer maps to monotonic position
    /// `live_start + i`; the next write lands at monotonic position
    /// `self.offset`, i.e. buffer slot `self.offset - self.live_start`.
    /// Pre- callers (those that never trim) see this equal to
    /// `self.offset` because `live_start == 0`.
    #[inline]
    fn buffer_idx(&self) -> i32 {
        self.offset - self.live_start
    }

    /// Read-only view of the live K/V window without writing to the cache.
    ///
    /// Returns the `[B, n_kv_heads, live_len, head_dim]` slices of the dense
    /// Fp16 buffers — exactly what [`Self::update_and_fetch`] would have
    /// returned for a zero-token update, but as a `&self` accessor so callers
    /// can concatenate the cached prefix with externally computed K/V
    /// without mutating cache state.
    ///
    /// Returns `None` when the cache is empty, pool-backed (`new_paged`), or
    /// not in plain `KVCacheMode::Fp16` — the only storage layout whose raw
    /// buffers are directly attention-ready without dequantization.
    ///
    /// Used by: DiffusionGemma canvas (decoder-mode) attention (issue #217),
    /// which reads the committed encoder prefix as a read-only context for
    /// every denoising step.
    pub fn visible_state(&self) -> Option<(UniquePtr<MlxArray>, UniquePtr<MlxArray>)> {
        if self.mode != KVCacheMode::Fp16 || self.paged_backing.is_some() {
            return None;
        }
        let keys = self.keys.as_ref()?;
        let values = self.values.as_ref()?;
        let live_len = self.buffer_idx();
        if live_len <= 0 {
            return None;
        }
        let ks = ffi::array_shape(keys);
        let vs = ffi::array_shape(values);
        Some((
            ffi::slice(keys, &[0, 0, 0, 0], &[ks[0], ks[1], live_len, ks[3]]),
            ffi::slice(values, &[0, 0, 0, 0], &[vs[0], vs[1], live_len, vs[3]]),
        ))
    }

    /// Get the allocated buffer size (sequence dimension)
    fn buffer_seq_len(&self) -> i32 {
        // In symmetric Turbo4 mode the `keys` field stays None and the
        // step-grown buffer lives in `k_packed`; consult that instead. All
        // other modes (Fp16, Int8, Turbo4Asym) keep the buffer in `keys`.
        let buf = self.keys.as_ref().or(self.k_packed.as_ref());
        match buf {
            Some(k) => {
                let shape = ffi::array_shape(k);
                if shape.len() >= 3 {
                    shape[2]
                } else {
                    0
                }
            }
            None => 0,
        }
    }

    /// Update cache with new key/value using pre-allocated buffer + slice_update.
    ///
    /// In `KVCacheMode::Int8` the incoming tensors are quantized to INT8 before
    /// storage; scale factors are accumulated in a parallel `[B, H, L, 1]`
    /// buffer. In `KVCacheMode::Turbo4Asym` the K side stays FP16 while the V
    /// side is quantized to packed 4-bit indices plus per-token norms via
    /// [`turbo::quant::quantize_v_turbo4`]. In `KVCacheMode::Turbo4` **both** K and V are quantized via the symmetric Turbo4 path
    /// using independent sign-vector pairs. In `KVCacheMode::Fp16` this
    /// behaves identically to the original implementation.
    pub fn update(&mut self, new_keys: UniquePtr<MlxArray>, new_values: UniquePtr<MlxArray>) {
        // Transparent pool-backed path (`new_paged`): append straight into the
        // shared `PagedBlockPool` instead of the dense buffers, mirroring the
        // `update_and_fetch` intercept. This is reached by callers that write
        // the cache without needing the gathered window back — notably the
        // single-sequence fused causal prefill path (llama3), where the fused
        // Metal kernel has already produced the attention output. Returns early
        // so the dense `update_*` below never runs and no dense buffer is
        // allocated. See `new_paged` / `write_paged`.
        if self.paged_backing.is_some() {
            self.write_paged(&new_keys, &new_values);
            return;
        }
        match self.mode {
            KVCacheMode::Int8 => self.update_int8(new_keys, new_values),
            KVCacheMode::Turbo4Asym => self.update_turbo4_asym(new_keys, new_values),
            KVCacheMode::Turbo4 => self.update_turbo4_sym(new_keys, new_values),
            KVCacheMode::Turbo4Delegated => self.update_turbo4_delegated(new_keys, new_values),
            KVCacheMode::Turbo3Asym => self.update_turbo3_asym(new_keys, new_values),
            KVCacheMode::Fp16 => self.update_fp16(new_keys, new_values),
        }
    }

    /// FP16 (standard) update path — original pre-allocated buffer logic.
    ///
    /// Operates in **buffer-slot** coordinates: `prev = self.offset
    /// - self.live_start` is the slot index where the next K/V token is
    /// written, and the slice-update upper bound is `prev + new_seq_len`
    /// (i.e. `buffer_idx()` after the monotonic `self.offset` has been
    /// advanced). The monotonic `self.offset` is incremented unconditionally
    /// so RoPE positions for subsequent Q tokens stay correct after a
    /// [`Self::trim_front`] has shifted `self.live_start` forward.
    fn update_fp16(&mut self, new_keys: UniquePtr<MlxArray>, new_values: UniquePtr<MlxArray>) {
        let key_shape = ffi::array_shape(&new_keys);
        let new_seq_len = key_shape[2];
        let prev = self.buffer_idx();

        if prev == 0 && self.keys.is_none() && direct_prefill_cache_store_enabled() {
            self.keys = Some(ffi::contiguous(&new_keys, false));
            self.values = Some(ffi::contiguous(&new_values, false));
            self.offset += new_seq_len;
            return;
        }

        if self.keys.is_none() || (prev + new_seq_len) > self.buffer_seq_len() {
            let b = key_shape[0];
            let n_kv_heads = key_shape[1];
            let k_head_dim = key_shape[3];
            let val_shape = ffi::array_shape(&new_values);
            let v_head_dim = val_shape[3];

            let n_steps = (self.step + new_seq_len - 1) / self.step;
            let buf_size = n_steps * self.step;

            let k_dtype = ffi::array_dtype(&new_keys);
            let v_dtype = ffi::array_dtype(&new_values);
            let new_k = ffi::zeros(&[b, n_kv_heads, buf_size, k_head_dim], k_dtype);
            let new_v = ffi::zeros(&[b, n_kv_heads, buf_size, v_head_dim], v_dtype);

            if self.keys.is_some() {
                if prev % self.step != 0 {
                    self.keys = Some(ffi::slice(
                        self.keys.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, k_head_dim],
                    ));
                    self.values = Some(ffi::slice(
                        self.values.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, v_head_dim],
                    ));
                }
                self.keys = Some(concatenate(self.keys.as_ref().unwrap(), &new_k, 2));
                self.values = Some(concatenate(self.values.as_ref().unwrap(), &new_v, 2));
            } else {
                self.keys = Some(new_k);
                self.values = Some(new_v);
            }
        }

        self.offset += new_seq_len;
        let live_len = self.buffer_idx();

        let k_shape = ffi::array_shape(self.keys.as_ref().unwrap());
        let v_shape = ffi::array_shape(self.values.as_ref().unwrap());
        self.keys = Some(ffi::slice_update(
            self.keys.as_ref().unwrap(),
            &new_keys,
            &[0, 0, prev, 0],
            &[k_shape[0], k_shape[1], live_len, k_shape[3]],
        ));
        self.values = Some(ffi::slice_update(
            self.values.as_ref().unwrap(),
            &new_values,
            &[0, 0, prev, 0],
            &[v_shape[0], v_shape[1], live_len, v_shape[3]],
        ));
    }

    /// INT8 update path — quantizes incoming K/V tokens and accumulates into
    /// INT8 key/value buffers alongside FP16 per-token scale buffers.
    ///
    /// Layout of stored buffers (step-aligned, grown lazily):
    /// - `keys`/`values`: `[B, H, capacity, D]` INT8
    /// - `key_scales`/`val_scales`: `[B, H, capacity, 1]` FP16
    fn update_int8(&mut self, new_keys: UniquePtr<MlxArray>, new_values: UniquePtr<MlxArray>) {
        // Cast incoming tensors to FP16 before quantization so scale
        // computation operates in a consistent dtype.
        let new_keys_f16 = ffi::astype(&new_keys, dtype::FLOAT16);
        let new_values_f16 = ffi::astype(&new_values, dtype::FLOAT16);

        let (k_int8, k_scale) = quantize_per_token(&new_keys_f16);
        let (v_int8, v_scale) = quantize_per_token(&new_values_f16);

        let key_shape = ffi::array_shape(&k_int8);
        let new_seq_len = key_shape[2];
        // Use buffer-slot coordinates so `--max-kv-size` trim_front stays
        // RoPE-correct: `prev` is the slot where the next token is
        // written, not the monotonic position.
        let prev = self.buffer_idx();

        if prev == 0 && self.keys.is_none() && direct_prefill_cache_store_enabled() {
            self.keys = Some(ffi::contiguous(&k_int8, false));
            self.values = Some(ffi::contiguous(&v_int8, false));
            self.key_scales = Some(ffi::contiguous(&k_scale, false));
            self.val_scales = Some(ffi::contiguous(&v_scale, false));
            self.offset += new_seq_len;
            return;
        }

        if self.keys.is_none() || (prev + new_seq_len) > self.buffer_seq_len() {
            let b = key_shape[0];
            let n_kv_heads = key_shape[1];
            let k_head_dim = key_shape[3];
            let val_shape = ffi::array_shape(&v_int8);
            let v_head_dim = val_shape[3];

            let n_steps = (self.step + new_seq_len - 1) / self.step;
            let buf_size = n_steps * self.step;

            let new_k_buf = ffi::zeros(&[b, n_kv_heads, buf_size, k_head_dim], dtype::INT8);
            let new_v_buf = ffi::zeros(&[b, n_kv_heads, buf_size, v_head_dim], dtype::INT8);
            let new_ks_buf = ffi::zeros(&[b, n_kv_heads, buf_size, 1], dtype::FLOAT16);
            let new_vs_buf = ffi::zeros(&[b, n_kv_heads, buf_size, 1], dtype::FLOAT16);

            if self.keys.is_some() {
                if prev % self.step != 0 {
                    self.keys = Some(ffi::slice(
                        self.keys.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, k_head_dim],
                    ));
                    self.values = Some(ffi::slice(
                        self.values.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, v_head_dim],
                    ));
                    self.key_scales = Some(ffi::slice(
                        self.key_scales.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, 1],
                    ));
                    self.val_scales = Some(ffi::slice(
                        self.val_scales.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, 1],
                    ));
                }
                self.keys = Some(concatenate(self.keys.as_ref().unwrap(), &new_k_buf, 2));
                self.values = Some(concatenate(self.values.as_ref().unwrap(), &new_v_buf, 2));
                self.key_scales = Some(concatenate(
                    self.key_scales.as_ref().unwrap(),
                    &new_ks_buf,
                    2,
                ));
                self.val_scales = Some(concatenate(
                    self.val_scales.as_ref().unwrap(),
                    &new_vs_buf,
                    2,
                ));
            } else {
                self.keys = Some(new_k_buf);
                self.values = Some(new_v_buf);
                self.key_scales = Some(new_ks_buf);
                self.val_scales = Some(new_vs_buf);
            }
        }

        self.offset += new_seq_len;
        let live_len = self.buffer_idx();

        let k_shape = ffi::array_shape(self.keys.as_ref().unwrap());
        let v_shape = ffi::array_shape(self.values.as_ref().unwrap());
        let ks_shape = ffi::array_shape(self.key_scales.as_ref().unwrap());
        let vs_shape = ffi::array_shape(self.val_scales.as_ref().unwrap());

        self.keys = Some(ffi::slice_update(
            self.keys.as_ref().unwrap(),
            &k_int8,
            &[0, 0, prev, 0],
            &[k_shape[0], k_shape[1], live_len, k_shape[3]],
        ));
        self.values = Some(ffi::slice_update(
            self.values.as_ref().unwrap(),
            &v_int8,
            &[0, 0, prev, 0],
            &[v_shape[0], v_shape[1], live_len, v_shape[3]],
        ));
        self.key_scales = Some(ffi::slice_update(
            self.key_scales.as_ref().unwrap(),
            &k_scale,
            &[0, 0, prev, 0],
            &[ks_shape[0], ks_shape[1], live_len, 1],
        ));
        self.val_scales = Some(ffi::slice_update(
            self.val_scales.as_ref().unwrap(),
            &v_scale,
            &[0, 0, prev, 0],
            &[vs_shape[0], vs_shape[1], live_len, 1],
        ));
    }

    /// Turbo4Asym update path — keeps the K side FP16 (mirroring `update_fp16`)
    /// and quantizes the V side via PolarQuant + Walsh–Hadamard rotation.
    ///
    /// Layout of stored buffers (step-aligned, grown lazily):
    /// - `keys`:    `[B, H, capacity, K_dim]` FP16, identical to the FP16 path.
    /// - `v_packed`: `[B, H, capacity, V_dim/2]` UINT8 (nibble-packed indices).
    /// - `v_norms`:  `[B, H, capacity, 1]` FP16 (per-token L2 of original V).
    ///
    /// `values` stays `None` in this mode — the FP16 values tensor is
    /// reconstructed lazily on `update_and_fetch` via
    /// [`turbo::quant::dequantize_v_turbo4`].
    ///
    /// Used by: `KVCache::update` dispatch when `mode == KVCacheMode::Turbo4Asym`
    fn update_turbo4_asym(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) {
        // Cast incoming K/V to FP16 to match the cache contract.
        let new_keys_f16 = ffi::astype(&new_keys, dtype::FLOAT16);
        let new_values_f16 = ffi::astype(&new_values, dtype::FLOAT16);

        // Lazy-init TurboQuantParams once we know the V head_dim.
        if self.turbo_params.is_none() {
            let v_shape = ffi::array_shape(&new_values_f16);
            let v_head_dim = v_shape[3] as u32;
            self.turbo_params = Some(turbo::TurboQuantParams::new(v_head_dim, self.turbo_seed));
        }
        let params = self
            .turbo_params
            .as_ref()
            .expect("turbo_params just initialised");

        let (v_packed_new, v_norms_new, v_rescale_new) =
            turbo::quant::quantize_v_turbo4(&new_values_f16, params);

        let key_shape = ffi::array_shape(&new_keys_f16);
        let new_seq_len = key_shape[2];
        let prev = self.offset;
        let v_packed_shape = ffi::array_shape(&v_packed_new);

        if prev == 0 && self.keys.is_none() && direct_prefill_cache_store_enabled() {
            self.keys = Some(ffi::contiguous(&new_keys_f16, false));
            self.v_packed = Some(ffi::contiguous(&v_packed_new, false));
            self.v_norms = Some(ffi::contiguous(&v_norms_new, false));
            self.v_rescale = Some(ffi::contiguous(&v_rescale_new, false));
            self.offset = new_seq_len;
            return;
        }

        if self.keys.is_none() || (prev + new_seq_len) > self.buffer_seq_len() {
            let b = key_shape[0];
            let n_kv_heads = key_shape[1];
            let k_head_dim = key_shape[3];
            let v_packed_dim = v_packed_shape[3];

            let n_steps = (self.step + new_seq_len - 1) / self.step;
            let buf_size = n_steps * self.step;

            let new_k_buf = ffi::zeros(&[b, n_kv_heads, buf_size, k_head_dim], dtype::FLOAT16);
            let new_vp_buf = ffi::zeros(&[b, n_kv_heads, buf_size, v_packed_dim], dtype::UINT8);
            let new_vn_buf = ffi::zeros(&[b, n_kv_heads, buf_size, 1], dtype::FLOAT16);
            let new_vr_buf = ffi::zeros(&[b, n_kv_heads, buf_size, 1], dtype::FLOAT16);

            if self.keys.is_some() {
                if prev % self.step != 0 {
                    self.keys = Some(ffi::slice(
                        self.keys.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, k_head_dim],
                    ));
                    self.v_packed = Some(ffi::slice(
                        self.v_packed.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, v_packed_dim],
                    ));
                    self.v_norms = Some(ffi::slice(
                        self.v_norms.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, 1],
                    ));
                    if let Some(ref vr) = self.v_rescale {
                        self.v_rescale =
                            Some(ffi::slice(vr, &[0, 0, 0, 0], &[b, n_kv_heads, prev, 1]));
                    }
                }
                self.keys = Some(concatenate(self.keys.as_ref().unwrap(), &new_k_buf, 2));
                self.v_packed = Some(concatenate(self.v_packed.as_ref().unwrap(), &new_vp_buf, 2));
                self.v_norms = Some(concatenate(self.v_norms.as_ref().unwrap(), &new_vn_buf, 2));
                self.v_rescale = match self.v_rescale.as_ref() {
                    Some(vr) => Some(concatenate(vr, &new_vr_buf, 2)),
                    None => Some(new_vr_buf),
                };
            } else {
                self.keys = Some(new_k_buf);
                self.v_packed = Some(new_vp_buf);
                self.v_norms = Some(new_vn_buf);
                self.v_rescale = Some(new_vr_buf);
            }
        }

        self.offset += new_seq_len;

        let k_shape = ffi::array_shape(self.keys.as_ref().unwrap());
        let vp_shape = ffi::array_shape(self.v_packed.as_ref().unwrap());
        let vn_shape = ffi::array_shape(self.v_norms.as_ref().unwrap());

        self.keys = Some(ffi::slice_update(
            self.keys.as_ref().unwrap(),
            &new_keys_f16,
            &[0, 0, prev, 0],
            &[k_shape[0], k_shape[1], self.offset, k_shape[3]],
        ));
        self.v_packed = Some(ffi::slice_update(
            self.v_packed.as_ref().unwrap(),
            &v_packed_new,
            &[0, 0, prev, 0],
            &[vp_shape[0], vp_shape[1], self.offset, vp_shape[3]],
        ));
        self.v_norms = Some(ffi::slice_update(
            self.v_norms.as_ref().unwrap(),
            &v_norms_new,
            &[0, 0, prev, 0],
            &[vn_shape[0], vn_shape[1], self.offset, 1],
        ));
        let vr_buf = self
            .v_rescale
            .as_ref()
            .expect("v_rescale lockstep with v_norms");
        let vr_shape = ffi::array_shape(vr_buf);
        self.v_rescale = Some(ffi::slice_update(
            vr_buf,
            &v_rescale_new,
            &[0, 0, prev, 0],
            &[vr_shape[0], vr_shape[1], self.offset, 1],
        ));
    }

    /// Turbo3Asym update path — keeps the K side FP16 (mirroring `update_fp16`)
    /// and quantizes the V side via 3-bit PolarQuant + Walsh–Hadamard rotation.
    ///
    /// Layout of stored buffers (step-aligned, grown lazily):
    /// - `keys`:    `[B, H, capacity, K_dim]` FP16, identical to the FP16 path.
    /// - `v_packed`: `[B, H, capacity, V_dim*3/8]` UINT8 (24-bit-grouped indices).
    /// - `v_norms`:  `[B, H, capacity, 1]` FP16 (per-token L2 of original V).
    ///
    /// `values` stays `None` in this mode — the FP16 values tensor is
    /// reconstructed lazily on `update_and_fetch` via
    /// [`turbo::quant3::dequantize_v_turbo3`].
    ///
    /// Mirrors `update_turbo4_asym` bit-for-bit except the V quantizer is
    /// `quantize_v_turbo3` (3-bit codebook + 24-bit packing) and the
    /// dedicated `turbo3_params` field caches the 3-bit codebook so the
    /// 4-bit `turbo_params` is not perturbed.
    ///
    /// Used by: `KVCache::update` dispatch when `mode == KVCacheMode::Turbo3Asym`
    fn update_turbo3_asym(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) {
        // Cast incoming K/V to FP16 to match the cache contract.
        let new_keys_f16 = ffi::astype(&new_keys, dtype::FLOAT16);
        let new_values_f16 = ffi::astype(&new_values, dtype::FLOAT16);

        // Lazy-init TurboQuantParams3 once we know the V head_dim.
        if self.turbo3_params.is_none() {
            let v_shape = ffi::array_shape(&new_values_f16);
            let v_head_dim = v_shape[3] as u32;
            self.turbo3_params = Some(turbo::quant3::TurboQuantParams3::new(
                v_head_dim,
                self.turbo_seed,
            ));
        }
        let params = self
            .turbo3_params
            .as_ref()
            .expect("turbo3_params just initialised");

        let (v_packed_new, v_norms_new) = turbo::quant3::quantize_v_turbo3(&new_values_f16, params);

        let key_shape = ffi::array_shape(&new_keys_f16);
        let new_seq_len = key_shape[2];
        let prev = self.offset;
        let v_packed_shape = ffi::array_shape(&v_packed_new);

        if prev == 0 && self.keys.is_none() && direct_prefill_cache_store_enabled() {
            self.keys = Some(ffi::contiguous(&new_keys_f16, false));
            self.v_packed = Some(ffi::contiguous(&v_packed_new, false));
            self.v_norms = Some(ffi::contiguous(&v_norms_new, false));
            self.offset = new_seq_len;
            return;
        }

        if self.keys.is_none() || (prev + new_seq_len) > self.buffer_seq_len() {
            let b = key_shape[0];
            let n_kv_heads = key_shape[1];
            let k_head_dim = key_shape[3];
            let v_packed_dim = v_packed_shape[3];

            let n_steps = (self.step + new_seq_len - 1) / self.step;
            let buf_size = n_steps * self.step;

            let new_k_buf = ffi::zeros(&[b, n_kv_heads, buf_size, k_head_dim], dtype::FLOAT16);
            let new_vp_buf = ffi::zeros(&[b, n_kv_heads, buf_size, v_packed_dim], dtype::UINT8);
            let new_vn_buf = ffi::zeros(&[b, n_kv_heads, buf_size, 1], dtype::FLOAT16);

            if self.keys.is_some() {
                if prev % self.step != 0 {
                    self.keys = Some(ffi::slice(
                        self.keys.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, k_head_dim],
                    ));
                    self.v_packed = Some(ffi::slice(
                        self.v_packed.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, v_packed_dim],
                    ));
                    self.v_norms = Some(ffi::slice(
                        self.v_norms.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, 1],
                    ));
                }
                self.keys = Some(concatenate(self.keys.as_ref().unwrap(), &new_k_buf, 2));
                self.v_packed = Some(concatenate(self.v_packed.as_ref().unwrap(), &new_vp_buf, 2));
                self.v_norms = Some(concatenate(self.v_norms.as_ref().unwrap(), &new_vn_buf, 2));
            } else {
                self.keys = Some(new_k_buf);
                self.v_packed = Some(new_vp_buf);
                self.v_norms = Some(new_vn_buf);
            }
        }

        self.offset += new_seq_len;

        let k_shape = ffi::array_shape(self.keys.as_ref().unwrap());
        let vp_shape = ffi::array_shape(self.v_packed.as_ref().unwrap());
        let vn_shape = ffi::array_shape(self.v_norms.as_ref().unwrap());

        self.keys = Some(ffi::slice_update(
            self.keys.as_ref().unwrap(),
            &new_keys_f16,
            &[0, 0, prev, 0],
            &[k_shape[0], k_shape[1], self.offset, k_shape[3]],
        ));
        self.v_packed = Some(ffi::slice_update(
            self.v_packed.as_ref().unwrap(),
            &v_packed_new,
            &[0, 0, prev, 0],
            &[vp_shape[0], vp_shape[1], self.offset, vp_shape[3]],
        ));
        self.v_norms = Some(ffi::slice_update(
            self.v_norms.as_ref().unwrap(),
            &v_norms_new,
            &[0, 0, prev, 0],
            &[vn_shape[0], vn_shape[1], self.offset, 1],
        ));
    }

    /// Symmetric Turbo4 update path — quantizes both K and V to 4-bit
    /// PolarQuant indices using independent sign-vector pairs.
    ///
    /// Layout of stored buffers (step-aligned, grown lazily):
    /// - `k_packed`: `[B, H, capacity, K_dim/2]` UINT8 (K-side nibble-packed indices).
    /// - `k_norms`:  `[B, H, capacity, 1]` FP16 (per-token L2 of original K).
    /// - `v_packed`: `[B, H, capacity, V_dim/2]` UINT8 (V-side nibble-packed indices).
    /// - `v_norms`:  `[B, H, capacity, 1]` FP16 (per-token L2 of original V).
    ///
    /// Both `keys` and `values` stay `None` in this mode — the FP16 K/V
    /// tensors are reconstructed lazily on `update_and_fetch` via
    /// [`turbo::quant::dequantize_k_turbo4`] / [`turbo::quant::dequantize_v_turbo4`].
    ///
    /// **Safety**: callers must consult [`turbo::is_symmetric_turbo_allowed`]
    /// before constructing a cache in this mode for an arbitrary model — see
    /// [`turbo::allowlist`] for the rationale.
    ///
    /// Used by: `KVCache::update` dispatch when `mode == KVCacheMode::Turbo4`
    fn update_turbo4_sym(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) {
        // Cast incoming K/V to FP16 to match the cache contract.
        let new_keys_f16 = ffi::astype(&new_keys, dtype::FLOAT16);
        let new_values_f16 = ffi::astype(&new_values, dtype::FLOAT16);

        // Lazy-init TurboQuantParams once we know the V head_dim. Both K and
        // V must share the same head_dim because attention requires Q·Kᵀ to
        // be a valid inner product — if the model used different head_dims
        // for K and V it would be a different architecture entirely. The
        // assertion below catches any future architecture that violates
        // this assumption.
        let v_shape = ffi::array_shape(&new_values_f16);
        let k_shape_in = ffi::array_shape(&new_keys_f16);
        debug_assert_eq!(
            k_shape_in[3], v_shape[3],
            "symmetric Turbo4 requires K and V head_dim to match; \
             got K head_dim={} V head_dim={} — this model likely uses \
             differently-sized K/V projections and is incompatible with \
             this mode.",
            k_shape_in[3], v_shape[3],
        );
        if self.turbo_params.is_none() {
            let v_head_dim = v_shape[3] as u32;
            self.turbo_params = Some(turbo::TurboQuantParams::new(v_head_dim, self.turbo_seed));
        }
        let params = self
            .turbo_params
            .as_ref()
            .expect("turbo_params just initialised");

        let (k_packed_new, k_norms_new) = turbo::quant::quantize_k_turbo4(&new_keys_f16, params);
        let (v_packed_new, v_norms_new, v_rescale_new) =
            turbo::quant::quantize_v_turbo4(&new_values_f16, params);

        let key_shape = ffi::array_shape(&new_keys_f16);
        let new_seq_len = key_shape[2];
        let prev = self.offset;
        let k_packed_shape = ffi::array_shape(&k_packed_new);
        let v_packed_shape = ffi::array_shape(&v_packed_new);

        if prev == 0 && self.is_empty() && direct_prefill_cache_store_enabled() {
            self.k_packed = Some(ffi::contiguous(&k_packed_new, false));
            self.k_norms = Some(ffi::contiguous(&k_norms_new, false));
            self.v_packed = Some(ffi::contiguous(&v_packed_new, false));
            self.v_norms = Some(ffi::contiguous(&v_norms_new, false));
            self.v_rescale = Some(ffi::contiguous(&v_rescale_new, false));
            self.offset = new_seq_len;
            return;
        }

        if self.k_packed.is_none() || (prev + new_seq_len) > self.buffer_seq_len() {
            let b = key_shape[0];
            let n_kv_heads = key_shape[1];
            let k_packed_dim = k_packed_shape[3];
            let v_packed_dim = v_packed_shape[3];

            let n_steps = (self.step + new_seq_len - 1) / self.step;
            let buf_size = n_steps * self.step;

            let new_kp_buf = ffi::zeros(&[b, n_kv_heads, buf_size, k_packed_dim], dtype::UINT8);
            let new_kn_buf = ffi::zeros(&[b, n_kv_heads, buf_size, 1], dtype::FLOAT16);
            let new_vp_buf = ffi::zeros(&[b, n_kv_heads, buf_size, v_packed_dim], dtype::UINT8);
            let new_vn_buf = ffi::zeros(&[b, n_kv_heads, buf_size, 1], dtype::FLOAT16);
            let new_vr_buf = ffi::zeros(&[b, n_kv_heads, buf_size, 1], dtype::FLOAT16);

            if self.k_packed.is_some() {
                if prev % self.step != 0 {
                    self.k_packed = Some(ffi::slice(
                        self.k_packed.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, k_packed_dim],
                    ));
                    self.k_norms = Some(ffi::slice(
                        self.k_norms.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, 1],
                    ));
                    self.v_packed = Some(ffi::slice(
                        self.v_packed.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, v_packed_dim],
                    ));
                    self.v_norms = Some(ffi::slice(
                        self.v_norms.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev, 1],
                    ));
                    if let Some(ref vr) = self.v_rescale {
                        self.v_rescale =
                            Some(ffi::slice(vr, &[0, 0, 0, 0], &[b, n_kv_heads, prev, 1]));
                    }
                }
                self.k_packed = Some(concatenate(self.k_packed.as_ref().unwrap(), &new_kp_buf, 2));
                self.k_norms = Some(concatenate(self.k_norms.as_ref().unwrap(), &new_kn_buf, 2));
                self.v_packed = Some(concatenate(self.v_packed.as_ref().unwrap(), &new_vp_buf, 2));
                self.v_norms = Some(concatenate(self.v_norms.as_ref().unwrap(), &new_vn_buf, 2));
                self.v_rescale = match self.v_rescale.as_ref() {
                    Some(vr) => Some(concatenate(vr, &new_vr_buf, 2)),
                    None => Some(new_vr_buf),
                };
            } else {
                self.k_packed = Some(new_kp_buf);
                self.k_norms = Some(new_kn_buf);
                self.v_packed = Some(new_vp_buf);
                self.v_norms = Some(new_vn_buf);
                self.v_rescale = Some(new_vr_buf);
            }
        }

        self.offset += new_seq_len;

        let kp_shape = ffi::array_shape(self.k_packed.as_ref().unwrap());
        let kn_shape = ffi::array_shape(self.k_norms.as_ref().unwrap());
        let vp_shape = ffi::array_shape(self.v_packed.as_ref().unwrap());
        let vn_shape = ffi::array_shape(self.v_norms.as_ref().unwrap());

        self.k_packed = Some(ffi::slice_update(
            self.k_packed.as_ref().unwrap(),
            &k_packed_new,
            &[0, 0, prev, 0],
            &[kp_shape[0], kp_shape[1], self.offset, kp_shape[3]],
        ));
        self.k_norms = Some(ffi::slice_update(
            self.k_norms.as_ref().unwrap(),
            &k_norms_new,
            &[0, 0, prev, 0],
            &[kn_shape[0], kn_shape[1], self.offset, 1],
        ));
        self.v_packed = Some(ffi::slice_update(
            self.v_packed.as_ref().unwrap(),
            &v_packed_new,
            &[0, 0, prev, 0],
            &[vp_shape[0], vp_shape[1], self.offset, vp_shape[3]],
        ));
        self.v_norms = Some(ffi::slice_update(
            self.v_norms.as_ref().unwrap(),
            &v_norms_new,
            &[0, 0, prev, 0],
            &[vn_shape[0], vn_shape[1], self.offset, 1],
        ));
        let vr_buf = self
            .v_rescale
            .as_ref()
            .expect("v_rescale lockstep with v_norms");
        let vr_shape = ffi::array_shape(vr_buf);
        self.v_rescale = Some(ffi::slice_update(
            vr_buf,
            &v_rescale_new,
            &[0, 0, prev, 0],
            &[vr_shape[0], vr_shape[1], self.offset, 1],
        ));
    }

    /// Turbo4Delegated update path — V-only hot/cold split with unified K
    ///
    /// **Phase 1: prefill / single-token append into hot tail.** Tokens land
    /// directly in the FP16 `keys`/`values` buffers, identical to the
    /// `update_fp16` path. No quantization runs.
    ///
    /// **Phase 2: prefill→decode transition.** Detected when `cold_offset == 0`
    /// and the incoming step is a single-token decode (`new_seq_len == 1`)
    /// against a populated cache (`offset > 0`). The current FP16 V body is
    /// **folded** into cold V storage: V is quantized into
    /// `v_packed`/`v_norms`/`v_rescale` via [`turbo::quant::quantize_v_turbo4`].
    /// The K side does *not* move — it stays in the unified `keys` buffer at
    /// `[0..offset]`. The hot V ring is then freshly allocated as a small ring
    /// (capacity = `step` × `ceil(hot_threshold / step)`) and the new decode
    /// token's V is appended. K just slice-updates into `keys[offset]` like
    /// FP16 mode.
    ///
    /// **Phase 3: subsequent decode.** K tokens append to the unified K
    /// buffer at `keys[offset]`; V tokens append to the V hot ring at
    /// `values[hot_offset()]`. When `hot_offset() > hot_threshold`, the
    /// oldest [`turbo::DELEGATED_FOLD_BLOCK`]-token block of V is folded
    /// into cold V storage (the K side just observes `cold_offset += block`
    /// — no K data movement). If the hot V ring ever exceeds
    /// [`turbo::DELEGATED_HOT_MAX`], an extra fold runs synchronously.
    ///
    /// Layout of stored buffers in delegated mode:
    /// - `keys`: unified FP16 K buffer, `[B, H, k_capacity, K_dim]`. Visible
    ///   length is `offset` — same shape contract as `KVCacheMode::Fp16`.
    ///   No cold/hot split for K.
    /// - `values`: V hot ring FP16, `[B, H, v_hot_capacity, V_dim]`.
    ///   Visible length is `hot_offset() = offset - cold_offset`.
    ///   In the opt-in FP16 fast path this same field is a unified FP16 V
    ///   buffer with visible length `offset`; the packed sidecars are retained
    ///   but are not read by attention.
    /// - `v_packed`: cold V Turbo4 indices, `[B, H, cold_capacity, V_dim/2]` u8.
    /// - `v_norms`: cold V per-token norms, `[B, H, cold_capacity, 1]` fp16.
    ///
    /// Used by: `KVCache::update` dispatch when `mode == KVCacheMode::Turbo4Delegated`
    /// (K unification:).
    fn update_turbo4_delegated(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) {
        let new_keys_f16 = ffi::astype(&new_keys, dtype::FLOAT16);
        let new_values_f16 = ffi::astype(&new_values, dtype::FLOAT16);

        if self.delegated_fp16_fast_path {
            self.update_turbo4_delegated_fp16_fast_path(&new_keys_f16, &new_values_f16);
            return;
        }

        let key_shape = ffi::array_shape(&new_keys_f16);
        let new_seq_len = key_shape[2];

        // Detect the prefill→decode transition: cold is empty, hot V already
        // has tokens, and this is a single-token decode step. We fold the
        // entire FP16 V body into cold V storage *before* appending the new
        // token so the first decode step sees the fully compressed V state.
        // The K side does not move — it stays unified in `keys`.
        if self.cold_offset == 0
            && self.offset > 0
            && new_seq_len == 1
            && self.values.is_some()
            && self.keys.is_some()
        {
            self.fold_hot_to_cold_full();
            // After fold_hot_to_cold_full(), `values` is reset to a fresh hot
            // V ring of capacity = ceil_step(hot_threshold); `keys` is
            // untouched (unified-K invariant).
        }

        // K side: ensure the unified `keys` buffer can hold `offset + new_seq_len`.
        // Growth follows the same step-aligned policy as `update_fp16`.
        let prev_offset = self.offset;
        let needed_k = prev_offset + new_seq_len;
        if self.keys.is_none() || needed_k > self.k_buffer_seq_len() {
            self.grow_unified_k_buffer(&new_keys_f16, needed_k);
        }

        // V side: ensure the hot V ring can hold `prev_hot + new_seq_len`.
        // Hot ring is anchored at `hot_threshold` so the first
        // decode-time allocation absorbs roughly one fold-period of tokens.
        let prev_hot = self.offset - self.cold_offset;
        let needed_v = prev_hot + new_seq_len;
        if self.values.is_none() || needed_v > self.v_hot_buffer_seq_len() {
            self.grow_hot_v_buffer(&new_values_f16, needed_v);
        }

        // Slice-update K at position `prev_offset` (unified buffer) and V at
        // position `prev_hot` (hot ring).
        let k_shape = ffi::array_shape(self.keys.as_ref().unwrap());
        let v_shape = ffi::array_shape(self.values.as_ref().unwrap());
        self.keys = Some(ffi::slice_update(
            self.keys.as_ref().unwrap(),
            &new_keys_f16,
            &[0, 0, prev_offset, 0],
            &[
                k_shape[0],
                k_shape[1],
                prev_offset + new_seq_len,
                k_shape[3],
            ],
        ));
        self.values = Some(ffi::slice_update(
            self.values.as_ref().unwrap(),
            &new_values_f16,
            &[0, 0, prev_hot, 0],
            &[v_shape[0], v_shape[1], prev_hot + new_seq_len, v_shape[3]],
        ));
        self.offset += new_seq_len;

        // Periodic fold: if hot V tail exceeded the threshold (and cold is now
        // populated, so we are in steady-state decode), fold one block.
        // Repeat synchronously while we are still over the hard `HOT_MAX` cap;
        // a single fold of `DELEGATED_FOLD_BLOCK` tokens is normally enough to
        // bring us back inside the threshold.
        if self.cold_offset > 0 {
            while self.hot_offset() > self.hot_threshold {
                let block = turbo::DELEGATED_FOLD_BLOCK.min(self.hot_offset());
                self.fold_hot_block_to_cold(block);
                if self.hot_offset() <= turbo::DELEGATED_HOT_MAX {
                    break;
                }
            }
        }
    }

    /// Turbo4Delegated FP16 working-set variant.
    ///
    /// This mirrors TurboQuant+'s delegated KVCache optimization: the packed V
    /// sidecars are maintained for measurement / future memory recovery, but
    /// decode attention reads a unified FP16 V buffer through native SDPA. The
    /// default Turbo4Delegated path below remains compressed-only for cold V.
    ///
    /// Used by: `update_turbo4_delegated` when
    /// `MLXCEL_TURBO4_DELEGATED_FP16_FAST_PATH=1`.
    fn update_turbo4_delegated_fp16_fast_path(
        &mut self,
        new_keys_f16: &MlxArray,
        new_values_f16: &MlxArray,
    ) {
        let key_shape = ffi::array_shape(new_keys_f16);
        let new_seq_len = key_shape[2];
        let prev_offset = self.offset;
        let needed = prev_offset + new_seq_len;

        if self.keys.is_none() || needed > self.k_buffer_seq_len() {
            self.grow_unified_k_buffer(new_keys_f16, needed);
        }
        if self.values.is_none() || needed > self.v_hot_buffer_seq_len() {
            self.grow_unified_v_buffer(new_values_f16, needed);
        }

        let k_shape = ffi::array_shape(self.keys.as_ref().unwrap());
        let v_shape = ffi::array_shape(self.values.as_ref().unwrap());
        self.keys = Some(ffi::slice_update(
            self.keys.as_ref().unwrap(),
            new_keys_f16,
            &[0, 0, prev_offset, 0],
            &[k_shape[0], k_shape[1], needed, k_shape[3]],
        ));
        self.values = Some(ffi::slice_update(
            self.values.as_ref().unwrap(),
            new_values_f16,
            &[0, 0, prev_offset, 0],
            &[v_shape[0], v_shape[1], needed, v_shape[3]],
        ));
        self.offset = needed;

        if self.delegated_fp16_sidecar_policy == turbo::DelegatedFp16SidecarPolicy::Predecode {
            if self.cold_offset == 0 && prev_offset > 0 && new_seq_len == 1 {
                self.fold_unified_v_range_to_cold(0, prev_offset);
            }

            while self.cold_offset > 0 && self.hot_offset() > self.hot_threshold {
                let block = turbo::DELEGATED_FOLD_BLOCK.min(self.hot_offset());
                self.fold_unified_v_range_to_cold(self.cold_offset, block);
                if self.hot_offset() <= turbo::DELEGATED_HOT_MAX {
                    break;
                }
            }
        }
    }

    /// Capacity (sequence dimension) of the unified `keys` buffer in
    /// Turbo4Delegated mode. Returns 0 when `keys` is None.
    fn k_buffer_seq_len(&self) -> i32 {
        match self.keys.as_ref() {
            Some(k) => {
                let s = ffi::array_shape(k);
                if s.len() >= 3 {
                    s[2]
                } else {
                    0
                }
            }
            None => 0,
        }
    }

    /// Capacity (sequence dimension) of the V hot ring `values` buffer in
    /// Turbo4Delegated mode. Returns 0 when `values` is None.
    fn v_hot_buffer_seq_len(&self) -> i32 {
        match self.values.as_ref() {
            Some(v) => {
                let s = ffi::array_shape(v);
                if s.len() >= 3 {
                    s[2]
                } else {
                    0
                }
            }
            None => 0,
        }
    }

    /// Allocate (or grow) the unified FP16 K buffer for Turbo4Delegated mode
    ///
    /// Capacity is rounded up to a multiple of `step`, identical to
    /// `update_fp16`'s growth policy. Existing visible `offset` K tokens are
    /// copied into the new buffer so subsequent slice-updates at `offset`
    /// land in the right place.
    ///
    /// Used by: `update_turbo4_delegated` when the K buffer needs to grow.
    fn grow_unified_k_buffer(&mut self, new_keys_f16: &MlxArray, needed_seq_len: i32) {
        let key_shape = ffi::array_shape(new_keys_f16);
        let b = key_shape[0];
        let n_kv_heads = key_shape[1];
        let k_head_dim = key_shape[3];

        let n_steps = (needed_seq_len + self.step - 1) / self.step;
        let buf_size = (n_steps * self.step).max(self.step);

        let new_k = ffi::zeros(&[b, n_kv_heads, buf_size, k_head_dim], dtype::FLOAT16);
        let prev = self.offset;
        if prev > 0 && self.keys.is_some() {
            // Copy existing K tokens into the fresh buffer at position 0.
            let old_k = self.keys.as_ref().unwrap();
            let old_k_shape = ffi::array_shape(old_k);
            let old_k_slice = ffi::slice(
                old_k,
                &[0, 0, 0, 0],
                &[old_k_shape[0], old_k_shape[1], prev, old_k_shape[3]],
            );
            self.keys = Some(ffi::slice_update(
                &new_k,
                &old_k_slice,
                &[0, 0, 0, 0],
                &[b, n_kv_heads, prev, k_head_dim],
            ));
        } else {
            self.keys = Some(new_k);
        }
    }

    /// Allocate (or grow) the FP16 V hot ring for Turbo4Delegated mode.
    ///
    /// Capacity is rounded up to a multiple of `step` and anchored at
    /// `hot_threshold` so the first decode-time allocation absorbs roughly
    /// one fold-period of V tokens before growing. Existing visible
    /// `hot_offset()` V tokens are copied into the new buffer so subsequent
    /// slice-updates land on the right tokens.
    ///
    /// Used by: `update_turbo4_delegated` when the V hot ring needs to grow.
    fn grow_hot_v_buffer(&mut self, new_values_f16: &MlxArray, needed_seq_len: i32) {
        let val_shape = ffi::array_shape(new_values_f16);
        let b = val_shape[0];
        let n_kv_heads = val_shape[1];
        let v_head_dim = val_shape[3];

        let target = needed_seq_len.max(self.hot_threshold);
        let n_steps = (target + self.step - 1) / self.step;
        let buf_size = (n_steps * self.step).max(self.step);

        let new_v = ffi::zeros(&[b, n_kv_heads, buf_size, v_head_dim], dtype::FLOAT16);

        let prev_hot = self.offset - self.cold_offset;
        if prev_hot > 0 && self.values.is_some() {
            let old_v = self.values.as_ref().unwrap();
            let old_v_shape = ffi::array_shape(old_v);
            let old_v_slice = ffi::slice(
                old_v,
                &[0, 0, 0, 0],
                &[old_v_shape[0], old_v_shape[1], prev_hot, old_v_shape[3]],
            );
            self.values = Some(ffi::slice_update(
                &new_v,
                &old_v_slice,
                &[0, 0, 0, 0],
                &[b, n_kv_heads, prev_hot, v_head_dim],
            ));
        } else {
            self.values = Some(new_v);
        }
    }

    /// Allocate (or grow) the unified FP16 V buffer used by the opt-in
    /// Turbo4Delegated fast path.
    ///
    /// Used by: `update_turbo4_delegated_fp16_fast_path`.
    fn grow_unified_v_buffer(&mut self, new_values_f16: &MlxArray, needed_seq_len: i32) {
        let val_shape = ffi::array_shape(new_values_f16);
        let b = val_shape[0];
        let n_kv_heads = val_shape[1];
        let v_head_dim = val_shape[3];

        let n_steps = (needed_seq_len + self.step - 1) / self.step;
        let buf_size = (n_steps * self.step).max(self.step);

        let new_v = ffi::zeros(&[b, n_kv_heads, buf_size, v_head_dim], dtype::FLOAT16);
        let prev = self.offset;
        if prev > 0 && self.values.is_some() {
            let old_v = self.values.as_ref().unwrap();
            let old_v_shape = ffi::array_shape(old_v);
            let old_v_slice = ffi::slice(
                old_v,
                &[0, 0, 0, 0],
                &[old_v_shape[0], old_v_shape[1], prev, old_v_shape[3]],
            );
            self.values = Some(ffi::slice_update(
                &new_v,
                &old_v_slice,
                &[0, 0, 0, 0],
                &[b, n_kv_heads, prev, v_head_dim],
            ));
        } else {
            self.values = Some(new_v);
        }
    }

    /// Fold the entire FP16 V hot body into cold V storage.
    ///
    /// Used for the prefill→decode transition: the V hot body is quantized
    /// into the packed `v_packed`/`v_norms`/`v_rescale` sidecars and the V
    /// hot ring is then re-allocated empty so subsequent decode tokens see a
    /// fresh ring. The K side does *not* move — the unified `keys` buffer
    /// stays untouched (unifies K storage and removes the cold/hot split for K). `cold_offset` advances by `prev_hot`; the K data at
    /// positions `[0..cold_offset]` is logically the "cold K" but lives in
    /// the same buffer as the hot K tokens.
    ///
    /// Used by: `update_turbo4_delegated` on the prefill→decode transition.
    fn fold_hot_to_cold_full(&mut self) {
        let prev_hot = self.offset - self.cold_offset;
        if prev_hot <= 0 || self.values.is_none() {
            return;
        }
        // Pull the visible V hot region out as a fresh fp16 slice.
        let hot_v = self.values.as_ref().unwrap();
        let hv_shape = ffi::array_shape(hot_v);
        let hot_v_slice = ffi::slice(
            hot_v,
            &[0, 0, 0, 0],
            &[hv_shape[0], hv_shape[1], prev_hot, hv_shape[3]],
        );

        // Quantize V before consuming the slice. K is not touched here.
        if self.turbo_params.is_none() {
            let v_head_dim = hv_shape[3] as u32;
            self.turbo_params = Some(turbo::TurboQuantParams::new(v_head_dim, self.turbo_seed));
        }
        let params = self
            .turbo_params
            .as_ref()
            .expect("turbo_params just initialised");
        let (v_packed_new, v_norms_new, v_rescale_new) =
            turbo::quant::quantize_v_turbo4(&hot_v_slice, params);

        // Append the produced cold V buffers (cold V is empty before the full
        // fold). `cold_offset` advances by `prev_hot` inside `append_cold_block`.
        self.append_cold_block(&v_packed_new, &v_norms_new, &v_rescale_new, prev_hot);

        // Reset the V hot ring to a freshly-allocated buffer sized for the
        // configured hot_threshold (rounded up to step). K is unified —
        // `keys` already holds all `offset` tokens at FP16 and is left
        // untouched.
        let b = hv_shape[0];
        let n_kv_heads = hv_shape[1];
        let v_head_dim = hv_shape[3];
        let target = self.hot_threshold;
        let n_steps = (target + self.step - 1) / self.step;
        let buf_size = (n_steps * self.step).max(self.step);
        self.values = Some(ffi::zeros(
            &[b, n_kv_heads, buf_size, v_head_dim],
            dtype::FLOAT16,
        ));
        // self.offset stays where it was (offset = cold_offset + 0).
    }

    /// Fold the oldest `block` tokens of the V hot ring into cold V storage
    /// and shift the remaining hot V tokens left by `block` positions
    ///
    /// The K side is unified — `keys[cold_offset..cold_offset+block]` already
    /// holds the K tokens that are logically being "folded". `cold_offset`
    /// advances by `block` inside `append_cold_block` and the K data simply
    /// stays where it is.
    ///
    /// Used by: steady-state decode in `update_turbo4_delegated` whenever
    /// `hot_offset() > hot_threshold`.
    fn fold_hot_block_to_cold(&mut self, block: i32) {
        if block <= 0 {
            return;
        }
        let prev_hot = self.offset - self.cold_offset;
        let block = block.min(prev_hot);
        if block == 0 {
            return;
        }

        let hot_v = self
            .values
            .as_ref()
            .expect("hot values must exist for fold");
        let hv_shape = ffi::array_shape(hot_v);
        let b = hv_shape[0];
        let n_kv_heads = hv_shape[1];
        let v_head_dim = hv_shape[3];

        // Slice the [0:block] window — these are the oldest hot V tokens.
        let hot_v_old = ffi::slice(hot_v, &[0, 0, 0, 0], &[b, n_kv_heads, block, v_head_dim]);

        // Quantize V. Inline the `turbo_params` lookup so the immutable
        // borrow ends before `append_cold_block` (which takes `&mut self`).
        let (v_packed_new, v_norms_new, v_rescale_new) = turbo::quant::quantize_v_turbo4(
            &hot_v_old,
            self.turbo_params
                .as_ref()
                .expect("turbo_params must be set after first fold_hot_to_cold_full"),
        );

        // Pre-compute the V hot-ring keep slice BEFORE any mutable borrow on
        // self. Once `hot_v_keep` is owned, the immutable borrow on
        // `self.values` (held via `hot_v`) can be released by NLL, freeing
        // the path for `&mut self` calls.
        let remaining = prev_hot - block;
        let hot_v_keep = if remaining > 0 {
            Some(ffi::slice(
                hot_v,
                &[0, 0, block, 0],
                &[b, n_kv_heads, prev_hot, v_head_dim],
            ))
        } else {
            None
        };
        // hot_v is no longer used past this point — NLL releases the
        // immutable borrow on self.values here.

        // Append V into cold storage (which is non-empty here — the cold V
        // prefix was already populated by the prefill→decode full fold).
        // `cold_offset` advances by `block`; the K side is unified and needs
        // no data movement.
        self.append_cold_block(&v_packed_new, &v_norms_new, &v_rescale_new, block);

        // Shift the remaining V hot tail [block..prev_hot] left into
        // [0..prev_hot-block] so subsequent slice-updates see contiguous
        // tokens at the front of the ring.
        if let Some(hot_v_keep) = hot_v_keep {
            self.values = Some(ffi::slice_update(
                self.values.as_ref().unwrap(),
                &hot_v_keep,
                &[0, 0, 0, 0],
                &[b, n_kv_heads, remaining, v_head_dim],
            ));
        }
        // self.offset is unchanged: cold_offset += block, hot_len -= block.
        // The total length stays the same. K side is unaffected — `keys`
        // already holds all `offset` tokens at FP16.
    }

    /// Fold an absolute `[start, start + block)` window from the unified FP16
    /// V buffer into packed cold V sidecars without dropping the FP16 source.
    ///
    /// Used by: `update_turbo4_delegated_fp16_fast_path`.
    fn fold_unified_v_range_to_cold(&mut self, start: i32, block: i32) {
        if block <= 0 {
            return;
        }
        debug_assert_eq!(
            start, self.cold_offset,
            "unified V folds must append exactly at the cold boundary"
        );

        let values = self
            .values
            .as_ref()
            .expect("unified values must exist for FP16 fast-path fold");
        let v_shape = ffi::array_shape(values);
        let b = v_shape[0];
        let n_kv_heads = v_shape[1];
        let v_head_dim = v_shape[3];
        let end = (start + block).min(self.offset);
        if end <= start {
            return;
        }
        let block = end - start;
        let v_slice = ffi::slice(values, &[0, 0, start, 0], &[b, n_kv_heads, end, v_head_dim]);

        if self.turbo_params.is_none() {
            self.turbo_params = Some(turbo::TurboQuantParams::new(
                v_head_dim as u32,
                self.turbo_seed,
            ));
        }
        let params = self
            .turbo_params
            .as_ref()
            .expect("turbo_params just initialised");
        let (v_packed_new, v_norms_new, v_rescale_new) =
            turbo::quant::quantize_v_turbo4(&v_slice, params);

        self.append_cold_block(&v_packed_new, &v_norms_new, &v_rescale_new, block);
    }

    /// Append a `block`-token chunk to the cold V storage buffers, growing
    /// them (in `step`-sized increments) when the existing capacity does not
    /// fit (V-only — K is unified, no cold-K buffer to update).
    ///
    /// Inputs are the freshly-prepared cold V tensors:
    /// - `v_packed_block`:  `[B, H, block, V_dim/2]` u8   — appended to `v_packed`.
    /// - `v_norms_block`:   `[B, H, block, 1]`      fp16  — appended to `v_norms`.
    /// - `v_rescale_block`: `[B, H, block, 1]`      fp16  — appended to `v_rescale`
    ///   (precomputed Sparse-V kernel rescale, lockstep with `v_norms`).
    ///
    /// Updates `cold_offset` by `block`. Used by both the full prefill→decode
    /// fold and the steady-state per-block fold.
    fn append_cold_block(
        &mut self,
        v_packed_block: &MlxArray,
        v_norms_block: &MlxArray,
        v_rescale_block: &MlxArray,
        block: i32,
    ) {
        let vp_shape = ffi::array_shape(v_packed_block);
        let _vn_shape = ffi::array_shape(v_norms_block);
        let b = vp_shape[0];
        let n_kv_heads = vp_shape[1];
        let v_packed_dim = vp_shape[3];

        let prev_cold = self.cold_offset;
        let needed = prev_cold + block;

        // Grow cold V buffers if needed.
        let cold_cap = match &self.v_packed {
            Some(vp) => {
                let s = ffi::array_shape(vp);
                if s.len() >= 3 {
                    s[2]
                } else {
                    0
                }
            }
            None => 0,
        };
        if needed > cold_cap {
            let n_steps = (needed + self.step - 1) / self.step;
            let buf_size = n_steps * self.step;
            let new_vp = ffi::zeros(&[b, n_kv_heads, buf_size, v_packed_dim], dtype::UINT8);
            let new_vn = ffi::zeros(&[b, n_kv_heads, buf_size, 1], dtype::FLOAT16);
            let new_vr = ffi::zeros(&[b, n_kv_heads, buf_size, 1], dtype::FLOAT16);
            if self.v_packed.is_some() && prev_cold > 0 {
                let old_vp = self.v_packed.as_ref().unwrap();
                let old_vn = self.v_norms.as_ref().unwrap();
                let old_vp_slice = ffi::slice(
                    old_vp,
                    &[0, 0, 0, 0],
                    &[b, n_kv_heads, prev_cold, v_packed_dim],
                );
                let old_vn_slice =
                    ffi::slice(old_vn, &[0, 0, 0, 0], &[b, n_kv_heads, prev_cold, 1]);
                let old_vr_slice = self
                    .v_rescale
                    .as_ref()
                    .map(|vr| ffi::slice(vr, &[0, 0, 0, 0], &[b, n_kv_heads, prev_cold, 1]));
                self.v_packed = Some(ffi::slice_update(
                    &new_vp,
                    &old_vp_slice,
                    &[0, 0, 0, 0],
                    &[b, n_kv_heads, prev_cold, v_packed_dim],
                ));
                self.v_norms = Some(ffi::slice_update(
                    &new_vn,
                    &old_vn_slice,
                    &[0, 0, 0, 0],
                    &[b, n_kv_heads, prev_cold, 1],
                ));
                self.v_rescale = Some(match old_vr_slice {
                    Some(s) => ffi::slice_update(
                        &new_vr,
                        &s,
                        &[0, 0, 0, 0],
                        &[b, n_kv_heads, prev_cold, 1],
                    ),
                    None => new_vr,
                });
            } else {
                self.v_packed = Some(new_vp);
                self.v_norms = Some(new_vn);
                self.v_rescale = Some(new_vr);
            }
        }

        // Slice-update the new V block at position prev_cold.
        let after = prev_cold + block;
        let vp_buf = self.v_packed.as_ref().unwrap();
        let vn_buf = self.v_norms.as_ref().unwrap();
        let vp_buf_shape = ffi::array_shape(vp_buf);
        let vn_buf_shape = ffi::array_shape(vn_buf);
        self.v_packed = Some(ffi::slice_update(
            vp_buf,
            v_packed_block,
            &[0, 0, prev_cold, 0],
            &[vp_buf_shape[0], vp_buf_shape[1], after, vp_buf_shape[3]],
        ));
        self.v_norms = Some(ffi::slice_update(
            vn_buf,
            v_norms_block,
            &[0, 0, prev_cold, 0],
            &[vn_buf_shape[0], vn_buf_shape[1], after, 1],
        ));
        let vr_buf = self
            .v_rescale
            .as_ref()
            .expect("v_rescale lockstep with v_norms in append_cold_block");
        let vr_buf_shape = ffi::array_shape(vr_buf);
        self.v_rescale = Some(ffi::slice_update(
            vr_buf,
            v_rescale_block,
            &[0, 0, prev_cold, 0],
            &[vr_buf_shape[0], vr_buf_shape[1], after, 1],
        ));

        self.cold_offset += block;
        // retired the `cold_v_dequant_cache` memo: the fused
        // kernel reads packed cold V directly, so the cold body changing has
        // no host-side cached state to invalidate.
    }

    /// Returns `true` if this cache entry supports trimming via [`KVCache::trim`].
    ///
    /// All `KVCache` variants (Fp16, Int8, Turbo4*, Turbo3Asym) support trim.
    /// This method exists as a mirror of the upstream mlx-lm `is_trimmable()`
    /// (`models/cache.py`) and is consumed by `can_trim_prompt_cache` to let
    /// speculative decoding fail fast when a non-trimmable cache type would
    /// otherwise silently corrupt the cache rewind logic.
    ///
    /// Used by: `can_trim_prompt_cache` (speculative decoding validation)
    #[inline]
    pub fn is_trimmable(&self) -> bool {
        true
    }

    /// Trim the last `n` entries from the cache.
    ///
    /// Returns the number of entries actually trimmed.
    /// In INT8 mode the corresponding scale buffers are also trimmed.
    /// In Turbo4Asym mode the V-packed and V-norms buffers are also trimmed.
    /// In Turbo4 (symmetric) mode both the K- and V-side packed sidecars are
    /// trimmed.
    /// In Turbo4Delegated mode the hot tail is shrunk first; only when the
    /// requested trim depth exceeds the hot tail does the cold storage get
    /// touched. This mirrors speculative decoding's "rewind one block" pattern
    /// from `update_turbo4_asym` and avoids paying for a re-quantize on the
    /// common short-rewind case.
    /// Used by: speculative decoding cache rewinds
    pub fn trim(&mut self, n: i32) -> i32 {
        // Clamp against the live window length, not the monotonic offset:
        // after a `trim_front`-induced live_start advance we must not roll
        // `self.offset` below `self.live_start`, otherwise the slot
        // arithmetic in subsequent `update_*` calls breaks. With
        // `live_start == 0` (the common case) this is bit-identical to
        // clamping against `self.offset`.
        let n = n.min(self.live_len());
        if n <= 0 {
            return 0;
        }
        // Pool-backed caches keep no dense `keys`/`values` buffers (#121); the
        // block table is the authoritative store and is trimmed through the
        // pool API (`CachePool::trim_paged_tokens` / `rewind_paged_tokens`),
        // never the dense buffer slicing below (which would `unwrap` a `None`
        // buffer). Treat a dense-side trim as a no-op for them.
        if self.paged_backing.is_some() {
            return 0;
        }
        // Turbo4Delegated: hot-first trim. Tokens to remove from cold = max(0, n - hot_len).
        // We adjust cold_offset and offset, then fall through to the per-mode buffer slicing
        // logic below to keep the buffers consistent with the new offsets.
        let mut hot_trim = 0_i32;
        let mut cold_trim = 0_i32;
        if self.mode == KVCacheMode::Turbo4Delegated {
            let hot_len = self.offset - self.cold_offset;
            hot_trim = n.min(hot_len);
            cold_trim = (n - hot_trim).max(0);
        }
        self.offset -= n;
        if self.mode == KVCacheMode::Turbo4Delegated {
            self.cold_offset -= cold_trim;
        }
        // After `self.offset -= n`, the live window length is
        // `self.offset - self.live_start`. Use that for the buffer-slot
        // slice upper bound so trim stays consistent under a non-zero
        // `live_start` (+ speculative-decode rewind composing).
        // `live_start` stays at `0` for Turbo modes because `trim_front`
        // refuses to advance it for them, so the Turbo branches below that
        // still use `self.offset` continue to be correct.
        let live_len_after = self.offset - self.live_start;
        if live_len_after == 0 {
            self.keys = None;
            self.values = None;
            self.key_scales = None;
            self.val_scales = None;
            self.v_packed = None;
            self.v_norms = None;
            self.v_rescale = None;
            self.k_packed = None;
            self.k_norms = None;
            self.cold_offset = 0;
            // Clear turbo_params so the next quantize call rebuilds it from
            // scratch (required if the caller reuses this cache slot with a
            // different head_dim). LOW-1 fix. The 3-bit
            // `turbo3_params` follows the same contract.
            self.turbo_params = None;
            self.turbo3_params = None;
            // retired the cold-V dequant memo — nothing to drop.
        } else if self.mode == KVCacheMode::Turbo4Delegated {
            // K is unified — slice `keys` to `self.offset` like
            // `Fp16` mode. V hot ring trims to the new hot length first; cold
            // V sidecars trim only when the requested depth exceeded the hot
            // V tail (`cold_trim > 0`).
            let new_hot_len = self.offset - self.cold_offset;
            if let Some(ref k) = self.keys {
                let k_shape = ffi::array_shape(k);
                self.keys = Some(ffi::slice(
                    k,
                    &[0, 0, 0, 0],
                    &[k_shape[0], k_shape[1], self.offset, k_shape[3]],
                ));
            }
            if let Some(ref v) = self.values {
                let v_shape = ffi::array_shape(v);
                let visible_v_len = if self.delegated_fp16_fast_path {
                    self.offset
                } else {
                    new_hot_len
                };
                if visible_v_len > 0 {
                    self.values = Some(ffi::slice(
                        v,
                        &[0, 0, 0, 0],
                        &[v_shape[0], v_shape[1], visible_v_len, v_shape[3]],
                    ));
                }
            }
            // If the cold portion shrank (n exceeded the hot V tail), re-slice
            // the V cold sidecars to the new cold_offset. K is unified — the
            // already-sliced `keys` covers the new total length, no separate
            // cold-K trim is needed.
            if cold_trim > 0 {
                if let Some(ref vp) = self.v_packed {
                    let vp_shape = ffi::array_shape(vp);
                    self.v_packed = Some(ffi::slice(
                        vp,
                        &[0, 0, 0, 0],
                        &[vp_shape[0], vp_shape[1], self.cold_offset, vp_shape[3]],
                    ));
                }
                if let Some(ref vn) = self.v_norms {
                    let vn_shape = ffi::array_shape(vn);
                    self.v_norms = Some(ffi::slice(
                        vn,
                        &[0, 0, 0, 0],
                        &[vn_shape[0], vn_shape[1], self.cold_offset, 1],
                    ));
                }
                if let Some(ref vr) = self.v_rescale {
                    let vr_shape = ffi::array_shape(vr);
                    self.v_rescale = Some(ffi::slice(
                        vr,
                        &[0, 0, 0, 0],
                        &[vr_shape[0], vr_shape[1], self.cold_offset, 1],
                    ));
                }
                // retired the cold-V dequant memo — nothing to
                // drop when the cold V body shrinks.
            }
            // Suppress unused-variable warnings in non-delegated branches.
            let _ = hot_trim;
        } else {
            // Non-delegated modes: simple buffer-prefix trim.
            // The slice upper bound is the post-trim *live* window length
            // (`live_len_after`), not the monotonic `self.offset`. For
            // Turbo modes `live_start == 0` so `live_len_after == self.offset`
            // and behaviour is bit-identical to the earlier implementation.
            // Trim keys (always present except in Turbo4Asym pre-init).
            if let Some(ref k) = self.keys {
                let k_shape = ffi::array_shape(k);
                self.keys = Some(ffi::slice(
                    k,
                    &[0, 0, 0, 0],
                    &[k_shape[0], k_shape[1], live_len_after, k_shape[3]],
                ));
            }
            // Trim values for FP16/INT8 modes (Turbo4Asym keeps values None).
            if let Some(ref v) = self.values {
                let v_shape = ffi::array_shape(v);
                self.values = Some(ffi::slice(
                    v,
                    &[0, 0, 0, 0],
                    &[v_shape[0], v_shape[1], live_len_after, v_shape[3]],
                ));
            }
            // Trim INT8 scale sidecars.
            if self.mode == KVCacheMode::Int8 {
                if let Some(ref ks) = self.key_scales {
                    let ks_shape = ffi::array_shape(ks);
                    self.key_scales = Some(ffi::slice(
                        ks,
                        &[0, 0, 0, 0],
                        &[ks_shape[0], ks_shape[1], live_len_after, 1],
                    ));
                }
                if let Some(ref vs) = self.val_scales {
                    let vs_shape = ffi::array_shape(vs);
                    self.val_scales = Some(ffi::slice(
                        vs,
                        &[0, 0, 0, 0],
                        &[vs_shape[0], vs_shape[1], live_len_after, 1],
                    ));
                }
            }
            // Trim Turbo4* V sidecars (per-token; speculative-decode rewinds
            // <1 block at a time so we never need to re-quantize a partial
            // block — the trimmed tail is already block-aligned in the buffer).
            // `Turbo4Asym`, `Turbo4`, and `Turbo3Asym` all carry
            // the V sidecars; only `Turbo4` (symmetric) additionally carries
            // the K-side sidecars.
            if matches!(
                self.mode,
                KVCacheMode::Turbo4Asym | KVCacheMode::Turbo4 | KVCacheMode::Turbo3Asym
            ) {
                if let Some(ref vp) = self.v_packed {
                    let vp_shape = ffi::array_shape(vp);
                    self.v_packed = Some(ffi::slice(
                        vp,
                        &[0, 0, 0, 0],
                        &[vp_shape[0], vp_shape[1], self.offset, vp_shape[3]],
                    ));
                }
                if let Some(ref vn) = self.v_norms {
                    let vn_shape = ffi::array_shape(vn);
                    self.v_norms = Some(ffi::slice(
                        vn,
                        &[0, 0, 0, 0],
                        &[vn_shape[0], vn_shape[1], self.offset, 1],
                    ));
                }
                // v_rescale: tracks v_norms lockstep. Turbo3Asym
                // never populates v_rescale (3-bit V kernel does not exist),
                // so the `if let Some(...)` guard handles that case.
                if let Some(ref vr) = self.v_rescale {
                    let vr_shape = ffi::array_shape(vr);
                    self.v_rescale = Some(ffi::slice(
                        vr,
                        &[0, 0, 0, 0],
                        &[vr_shape[0], vr_shape[1], self.offset, 1],
                    ));
                }
            }
            // K-side sidecars are present only in symmetric Turbo4.
            if self.mode == KVCacheMode::Turbo4 {
                if let Some(ref kp) = self.k_packed {
                    let kp_shape = ffi::array_shape(kp);
                    self.k_packed = Some(ffi::slice(
                        kp,
                        &[0, 0, 0, 0],
                        &[kp_shape[0], kp_shape[1], self.offset, kp_shape[3]],
                    ));
                }
                if let Some(ref kn) = self.k_norms {
                    let kn_shape = ffi::array_shape(kn);
                    self.k_norms = Some(ffi::slice(
                        kn,
                        &[0, 0, 0, 0],
                        &[kn_shape[0], kn_shape[1], self.offset, 1],
                    ));
                }
            }
        }
        n
    }

    /// Drop the oldest `n` entries from the live window of the cache.
    ///
    /// This is the dual of [`KVCache::trim`] (which removes the *newest* `n`
    /// entries for speculative-decode rewinds). `trim_front` is used by the
    /// batch scheduler's `--max-kv-size` bound to cap the
    /// memory footprint of a plain `KVCache` by evicting the oldest tokens
    /// once the live size exceeds the configured cap. Sliding-window models
    /// that already use [`RotatingKVCache`] manage their own circular
    /// buffer and never go through this path.
    ///
    /// # Position invariant (the load-bearing reason for the live_start split)
    ///
    /// Unlike the previous implementation, this method **does not decrement
    /// `self.offset`**. Decrementing the monotonic position would silently
    /// break RoPE: K vectors at buffer slot `i` were rotated at their
    /// original write-time monotonic position, and Q gets rotated at the
    /// current `self.offset` on the next forward pass. If `offset` were
    /// rolled back by `n`, the relative position `offset - i_rope` seen by
    /// attention would shift by `-n` and the model output collapses the
    /// moment the cap kicks in.
    ///
    /// Instead, this method:
    /// 1. Physically slices the oldest `n` slots off the on-device buffers
    ///    so memory shrinks (the actual point of `--max-kv-size`),
    /// 2. **Advances `self.live_start` by `n`** so subsequent updates write
    ///    at the right slot (`buffer_idx() = offset - live_start`) and
    ///    [`Self::update_and_fetch`] returns only the live window
    ///    `[live_start .. offset]`,
    /// 3. **Leaves `self.offset` unchanged** so RoPE for the next Q stays
    ///    rotated at the correct monotonic position.
    ///
    /// Upstream `RotatingKVCache` (https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py#L410-L510) uses
    /// the same `offset` monotonic / `_idx` rotating split for the same
    /// reason; the names differ but the invariant is identical.
    ///
    /// Returns the number of entries actually trimmed (clamped to
    /// `[0, live_len]`).
    ///
    /// # Supported modes
    ///
    /// Only `KVCacheMode::Fp16` and `KVCacheMode::Int8` are supported by
    /// this method. Calling `trim_front` on a `Turbo4*` / `Turbo3*` cache
    /// is a no-op that returns `0` — the Turbo modes maintain per-token
    /// rotation state in their sidecars (`turbo_params`, `v_packed`,
    /// `v_norms`, `cold_offset`, etc.) that is not safe to truncate from
    /// the head, and `--max-kv-size` is not supported in combination with
    /// Turbo KV quantization in v1. The scheduler is expected to log a
    /// warning when this combination is requested rather than silently
    /// producing incorrect output. The `live_start` field is guaranteed
    /// to stay at `0` for Turbo modes so the Turbo fetch paths that still
    /// slice `[0..self.offset]` continue to be correct.
    ///
    /// Used by: [`crate::cache::CachePool`] consumers that need to enforce
    /// a max KV size on otherwise-unbounded `KVCache` instances (server batch scheduler).
    pub fn trim_front(&mut self, n: i32) -> i32 {
        // Clamp against the live window length, not the monotonic offset:
        // the live window is what attention sees, and that is what we are
        // capping.
        let live_len = self.buffer_idx();
        let n = n.min(live_len).max(0);
        if n == 0 {
            return 0;
        }

        // Pool-backed caches (#121) hold no dense head buffer to slice from the
        // front, and advancing `live_start` here would desync the cache from the
        // pool's authoritative block table (which the gather reads via the
        // per-sequence state, not `live_start`). The `--max-kv-size` cap for
        // paged sequences is a pool-side concern; treat the dense-side front
        // trim as a no-op so it never silently diverges.
        if self.paged_backing.is_some() {
            return 0;
        }

        // Only Fp16 and Int8 are supported; Turbo modes carry rotation
        // state in their sidecars that is not safe to truncate from the
        // head. `live_start` must remain `0` for those modes so the Turbo
        // fetch paths that still slice `[0..self.offset]` continue to be
        // correct.
        if !matches!(self.mode, KVCacheMode::Fp16 | KVCacheMode::Int8) {
            return 0;
        }

        let new_live_len = live_len - n;

        if new_live_len == 0 {
            // Whole live window is being dropped — reset the buffers so
            // subsequent updates reallocate from scratch with the correct
            // `step` granularity. `self.offset` STAYS monotonic; only
            // `live_start` jumps forward so the next-write slot is `0`
            // (matching `offset - live_start == 0`).
            self.live_start = self.offset;
            self.keys = None;
            self.values = None;
            self.key_scales = None;
            self.val_scales = None;
            return n;
        }

        // Slice keys: drop buffer slots `[0, n)` (i.e. monotonic positions
        // `[live_start, live_start + n)`). The new buffer slot `0` now
        // corresponds to monotonic position `live_start + n`, which is the
        // new `live_start`.
        if let Some(ref k) = self.keys {
            let k_shape = ffi::array_shape(k);
            self.keys = Some(ffi::slice(
                k,
                &[0, 0, n, 0],
                &[k_shape[0], k_shape[1], live_len, k_shape[3]],
            ));
        }
        if let Some(ref v) = self.values {
            let v_shape = ffi::array_shape(v);
            self.values = Some(ffi::slice(
                v,
                &[0, 0, n, 0],
                &[v_shape[0], v_shape[1], live_len, v_shape[3]],
            ));
        }
        // Slice INT8 scale sidecars in lockstep.
        if self.mode == KVCacheMode::Int8 {
            if let Some(ref ks) = self.key_scales {
                let ks_shape = ffi::array_shape(ks);
                self.key_scales = Some(ffi::slice(
                    ks,
                    &[0, 0, n, 0],
                    &[ks_shape[0], ks_shape[1], live_len, 1],
                ));
            }
            if let Some(ref vs) = self.val_scales {
                let vs_shape = ffi::array_shape(vs);
                self.val_scales = Some(ffi::slice(
                    vs,
                    &[0, 0, n, 0],
                    &[vs_shape[0], vs_shape[1], live_len, 1],
                ));
            }
        }

        // CRITICAL: do NOT modify `self.offset`. Advance `live_start` only.
        // See the top-level doc comment for the RoPE rationale.
        self.live_start += n;
        n
    }

    /// Update cache and return view of filled portion.
    ///
    /// In `KVCacheMode::Fp16` returns sliced FP16 keys/values directly.
    /// In `KVCacheMode::Int8` dequantizes the accumulated INT8 buffers back to
    /// FP16 before returning, so attention kernels always receive FP16 tensors.
    /// In `KVCacheMode::Turbo4Asym` returns FP16 keys (untouched) and a V tensor
    /// reconstructed from the packed 4-bit indices + per-token norms.
    /// In `KVCacheMode::Turbo4` (symmetric) **both** K and V are reconstructed
    /// from packed 4-bit indices using independent sign-vector pairs.
    /// In `KVCacheMode::Turbo4Delegated` (K unified) returns
    /// `slice(keys, 0, offset)` for K (no concat — same shape contract as
    /// `Fp16` mode) and
    /// `[dequant(v_packed[:cold_offset]); hot_values[:hot_offset]]` for V.
    /// SDPA always sees FP16; the packed cold V storage is internal. With
    /// `MLXCEL_TURBO4_DELEGATED_FP16_FAST_PATH=1`, V is already a unified FP16
    /// working set and this returns `slice(values, 0, offset)`.
    pub fn update_and_fetch(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        // Transparent pool-backed path (`new_paged`). Writes the new K/V into
        // the shared `PagedBlockPool` and returns the gathered visible window,
        // so the model `forward` is unchanged. Returns early; the dense
        // `self.update(...)` below is intentionally skipped so we never
        // double-write or allocate dense buffers. See `new_paged`.
        if self.paged_backing.is_some() {
            return self.update_and_fetch_paged(&new_keys, &new_values);
        }

        self.update(new_keys, new_values);

        // After `update`, the live window length equals `buffer_idx()` —
        // i.e. `self.offset - self.live_start`. For all callers that never
        // trim (`live_start == 0`) this is bit-identical to the original
        // `self.offset`-based slicing.
        let live_len = self.buffer_idx();
        match self.mode {
            KVCacheMode::Int8 => {
                // Dequantize the filled portion of the INT8 buffers
                let k_int8 = self.keys.as_ref().unwrap();
                let v_int8 = self.values.as_ref().unwrap();
                let k_scales = self.key_scales.as_ref().unwrap();
                let v_scales = self.val_scales.as_ref().unwrap();

                let ks = ffi::array_shape(k_int8);
                let vs = ffi::array_shape(v_int8);
                let kss = ffi::array_shape(k_scales);
                let vss = ffi::array_shape(v_scales);

                let k_slice = ffi::slice(k_int8, &[0, 0, 0, 0], &[ks[0], ks[1], live_len, ks[3]]);
                let v_slice = ffi::slice(v_int8, &[0, 0, 0, 0], &[vs[0], vs[1], live_len, vs[3]]);
                let ks_slice = ffi::slice(k_scales, &[0, 0, 0, 0], &[kss[0], kss[1], live_len, 1]);
                let vs_slice = ffi::slice(v_scales, &[0, 0, 0, 0], &[vss[0], vss[1], live_len, 1]);

                (
                    dequantize(&k_slice, &ks_slice),
                    dequantize(&v_slice, &vs_slice),
                )
            }
            KVCacheMode::Turbo4Asym => {
                let k = self.keys.as_ref().unwrap();
                let vp = self.v_packed.as_ref().unwrap();
                let vn = self.v_norms.as_ref().unwrap();
                let params = self
                    .turbo_params
                    .as_ref()
                    .expect("turbo_params must be initialised after first update_turbo4_asym");

                let ks = ffi::array_shape(k);
                let vps = ffi::array_shape(vp);
                let vns = ffi::array_shape(vn);

                let k_slice = ffi::slice(k, &[0, 0, 0, 0], &[ks[0], ks[1], self.offset, ks[3]]);
                let vp_slice =
                    ffi::slice(vp, &[0, 0, 0, 0], &[vps[0], vps[1], self.offset, vps[3]]);
                let vn_slice = ffi::slice(vn, &[0, 0, 0, 0], &[vns[0], vns[1], self.offset, 1]);

                let v_dequantized = turbo::quant::dequantize_v_turbo4(&vp_slice, &vn_slice, params);
                (k_slice, v_dequantized)
            }
            KVCacheMode::Turbo4 => {
                let kp = self.k_packed.as_ref().unwrap();
                let kn = self.k_norms.as_ref().unwrap();
                let vp = self.v_packed.as_ref().unwrap();
                let vn = self.v_norms.as_ref().unwrap();
                let params = self
                    .turbo_params
                    .as_ref()
                    .expect("turbo_params must be initialised after first update_turbo4_sym");

                let kps = ffi::array_shape(kp);
                let kns = ffi::array_shape(kn);
                let vps = ffi::array_shape(vp);
                let vns = ffi::array_shape(vn);

                let kp_slice =
                    ffi::slice(kp, &[0, 0, 0, 0], &[kps[0], kps[1], self.offset, kps[3]]);
                let kn_slice = ffi::slice(kn, &[0, 0, 0, 0], &[kns[0], kns[1], self.offset, 1]);
                let vp_slice =
                    ffi::slice(vp, &[0, 0, 0, 0], &[vps[0], vps[1], self.offset, vps[3]]);
                let vn_slice = ffi::slice(vn, &[0, 0, 0, 0], &[vns[0], vns[1], self.offset, 1]);

                let k_dequantized = turbo::quant::dequantize_k_turbo4(&kp_slice, &kn_slice, params);
                let v_dequantized = turbo::quant::dequantize_v_turbo4(&vp_slice, &vn_slice, params);
                (k_dequantized, v_dequantized)
            }
            KVCacheMode::Turbo4Delegated => self.fetch_turbo4_delegated(),
            KVCacheMode::Turbo3Asym => {
                let k = self.keys.as_ref().unwrap();
                let vp = self.v_packed.as_ref().unwrap();
                let vn = self.v_norms.as_ref().unwrap();
                let params = self
                    .turbo3_params
                    .as_ref()
                    .expect("turbo3_params must be initialised after first update_turbo3_asym");

                let ks = ffi::array_shape(k);
                let vps = ffi::array_shape(vp);
                let vns = ffi::array_shape(vn);

                let k_slice = ffi::slice(k, &[0, 0, 0, 0], &[ks[0], ks[1], self.offset, ks[3]]);
                let vp_slice =
                    ffi::slice(vp, &[0, 0, 0, 0], &[vps[0], vps[1], self.offset, vps[3]]);
                let vn_slice = ffi::slice(vn, &[0, 0, 0, 0], &[vns[0], vns[1], self.offset, 1]);

                let v_dequantized =
                    turbo::quant3::dequantize_v_turbo3(&vp_slice, &vn_slice, params);
                (k_slice, v_dequantized)
            }
            KVCacheMode::Fp16 => {
                let k = self.keys.as_ref().unwrap();
                let v = self.values.as_ref().unwrap();
                let ks = ffi::array_shape(k);
                let vs = ffi::array_shape(v);
                (
                    ffi::slice(k, &[0, 0, 0, 0], &[ks[0], ks[1], live_len, ks[3]]),
                    ffi::slice(v, &[0, 0, 0, 0], &[vs[0], vs[1], live_len, vs[3]]),
                )
            }
        }
    }

    /// Transparent pool-backed `update_and_fetch` for single-stream
    /// (`B == 1`) sequences created with [`Self::new_paged`].
    ///
    /// Appends the new K/V tokens to this layer's tail inside the shared
    /// [`PagedBlockPool`] (`write_prefill` handles both bulk prefill and a
    /// single decode token), advances the monotonic `offset`, and returns the
    /// gathered visible window in the SDPA-ready shape
    /// `[1, n_kv_heads, visible_len, head_dim]` — byte-identical to what the
    /// dense Fp16 path would have returned. The dense `keys`/`values` buffers
    /// stay `None`.
    ///
    /// `offset` is advanced by `n_new` here for the same reason
    /// [`Self::update_fp16`] advances it: RoPE reads `cache.offset` *before*
    /// the next step's `update_and_fetch`, so it must reflect the
    /// pre-call position during RoPE and the post-call position afterward. The
    /// paged `state.layers[layer_idx].len` is bumped in lockstep by
    /// `write_prefill`, so with `live_start == 0` (no head-trim on this path)
    /// `buffer_idx() == offset == state len`.
    ///
    /// # Panics
    ///
    /// Panics (with a descriptive message) if the backing is missing, the K/V
    /// batch dim is not `1`, the pool `write_prefill`/`gather_visible` calls
    /// fail (e.g. geometry mismatch), or the gather is unexpectedly empty after
    /// a non-empty write. The model `forward` signature returns a plain tuple,
    /// so there is no `Result` to thread these through; a misuse is a hard bug,
    /// not a recoverable condition.
    fn update_and_fetch_paged(
        &mut self,
        new_keys: &MlxArray,
        new_values: &MlxArray,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        // Append the new tokens (no-op for a zero-token call), then gather the
        // full visible window. Splitting the write into `write_paged` lets the
        // no-fetch `update` path reuse the exact same pool-append logic.
        self.write_paged(new_keys, new_values);

        let backing = self
            .paged_backing
            .clone()
            .expect("update_and_fetch_paged called without paged backing");
        let state = backing.state.borrow();
        let pool = backing.pool.borrow();
        pool.gather_visible(&state, backing.layer_idx)
            .expect("PagedBlockPool::gather_visible failed for pool-backed cache")
            .expect("gather_visible returned None for a pool-backed cache")
    }

    /// Pool-write half of the pool-backed cache path: append the new K/V
    /// tokens to this layer's tail inside the shared [`PagedBlockPool`]
    /// (`write_prefill` handles both bulk prefill and a single decode token)
    /// and advance the monotonic `offset`.
    ///
    /// Shared by [`Self::update`] (no fetch) and
    /// [`Self::update_and_fetch_paged`] (which gathers the visible window
    /// afterward). A zero-token call is a no-op so callers can pass an empty
    /// update without special-casing. The pool/state borrows are scoped tightly
    /// to this method so they never nest with the gather in
    /// `update_and_fetch_paged` or any scheduler-side borrow.
    ///
    /// `offset` is advanced by `n_new` for the same reason
    /// [`Self::update_fp16`] advances it: RoPE reads `cache.offset` *before*
    /// the next step's update, so it must reflect the pre-call position during
    /// RoPE and the post-call position afterward. The paged
    /// `state.layers[layer_idx].len` is bumped in lockstep by `write_prefill`.
    ///
    /// # Panics
    ///
    /// Panics if the backing is missing, the K/V batch dim is not `1`, or the
    /// pool `write_prefill` fails (geometry mismatch). The model `forward`
    /// signature returns plain tuples, so a misuse is a hard bug, not a
    /// recoverable condition.
    fn write_paged(&mut self, new_keys: &MlxArray, new_values: &MlxArray) {
        let backing = self
            .paged_backing
            .clone()
            .expect("write_paged called without paged backing");
        let layer_idx = backing.layer_idx;

        let k_shape = ffi::array_shape(new_keys);
        assert_eq!(
            k_shape.len(),
            4,
            "write_paged: K must be [1, n_kv_heads, n_new, head_dim], got shape {k_shape:?}"
        );
        assert_eq!(
            k_shape[0], 1,
            "write_paged: only single-stream (B == 1) is supported, got batch {} (shape {k_shape:?})",
            k_shape[0]
        );
        let n_new = k_shape[2];
        if n_new <= 0 {
            // Nothing to append; leave the offset and pool untouched.
            return;
        }

        {
            let mut pool = backing.pool.borrow_mut();
            let mut state = backing.state.borrow_mut();
            pool.write_prefill(&mut state, layer_idx, new_keys, new_values)
                .expect("PagedBlockPool::write_prefill failed");
        }

        // Advance the monotonic write position. RoPE for the *next* step reads
        // `cache.offset` before calling back into the cache, so this mirrors
        // the dense `update_fp16` bump exactly.
        self.offset += n_new;
    }

    /// Read path for `KVCacheMode::Turbo4Delegated` (K unified; cold-V dequant memo retired).
    ///
    /// K is a single unified FP16 buffer — returns
    /// `slice(keys, 0, offset)`, identical to `KVCacheMode::Fp16`. No
    /// per-step concat on the K side; that was the dominant residual cost
    /// vs FP16 mode (~7 ms/step at 4 K context).
    ///
    /// V still uses a packed cold body + FP16 hot ring: returns
    /// `concat(dequant(v_packed[:cold_offset]), hot_V)`. When
    /// `cold_offset == 0` (still in prefill or at the boundary before any
    /// fold has happened) this degrades to a plain hot V slice; when
    /// `hot_offset() == 0` (just after a full fold with no decode token yet)
    /// this returns just the dequantized cold V body.
    ///
    /// **Per-step cost.** This fallback rebuilds the full
    /// `dequantize_v_turbo4(v_packed[:cold_offset], v_norms[:cold_offset])`
    /// graph on every call — the earlier `cold_v_dequant_cache` memo was
    /// retired because the fused kernel path
    /// ([`Self::update_and_turbo4_delegated_attention`]) reads packed cold V
    /// directly without ever materialising the FP16 cold V tensor. This
    /// fallback is now only reached on the prefill-shaped path (multi-token
    /// `update_and_fetch`) or when callers have not been ported to the
    /// fused-attention API; on the decode hot path the fused kernel is the
    /// only consumer.
    ///
    /// With `delegated_fp16_fast_path`, `values` is a unified FP16 V buffer.
    /// In that mode this read path returns `slice(values, 0, offset)` and never
    /// materializes cold V from packed sidecars.
    fn fetch_turbo4_delegated(&mut self) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        // K side: unified buffer — same path as FP16 mode.
        let k = self.keys.as_ref().expect("unified keys must exist");
        let ks = ffi::array_shape(k);
        let k_slice = ffi::slice(k, &[0, 0, 0, 0], &[ks[0], ks[1], self.offset, ks[3]]);

        if self.delegated_fp16_fast_path {
            let v = self
                .values
                .as_ref()
                .expect("unified values must exist in Turbo4Delegated FP16 fast path");
            let vs = ffi::array_shape(v);
            let v_slice = ffi::slice(v, &[0, 0, 0, 0], &[vs[0], vs[1], self.offset, vs[3]]);
            return (k_slice, v_slice);
        }

        // V side: packed cold body + FP16 hot ring. Hot slice takes `hot_len`
        // tokens from the front of the hot ring.
        let hot_len = self.offset - self.cold_offset;
        let hot_v_slice = if hot_len > 0 {
            let v = self.values.as_ref().expect("hot values must exist");
            let vs = ffi::array_shape(v);
            Some(ffi::slice(
                v,
                &[0, 0, 0, 0],
                &[vs[0], vs[1], hot_len, vs[3]],
            ))
        } else {
            None
        };

        // No cold V body yet — just return the hot V slice (paired with the
        // unified K slice).
        if self.cold_offset == 0 {
            return (
                k_slice,
                hot_v_slice.expect("hot V must exist when cold_offset == 0 and offset > 0"),
            );
        }

        // Cold V dequant: rebuild from packed every call (retired the memo). On the fused-attention decode path the kernel reads
        // `v_packed` directly — this fallback only runs for prefill-shaped
        // calls and unported call sites.
        let vp = self.v_packed.as_ref().expect("v_packed must exist");
        let vn = self.v_norms.as_ref().expect("v_norms must exist");
        let params = self
            .turbo_params
            .as_ref()
            .expect("turbo_params must be set after first fold");
        let vp_shape = ffi::array_shape(vp);
        let vn_shape = ffi::array_shape(vn);
        let vp_slice = ffi::slice(
            vp,
            &[0, 0, 0, 0],
            &[vp_shape[0], vp_shape[1], self.cold_offset, vp_shape[3]],
        );
        let vn_slice = ffi::slice(
            vn,
            &[0, 0, 0, 0],
            &[vn_shape[0], vn_shape[1], self.cold_offset, 1],
        );
        let cold_v_dequant = turbo::quant::dequantize_v_turbo4(&vp_slice, &vn_slice, params);

        // V side: concatenate cold + hot. K is already a unified slice.
        let full_v = match hot_v_slice {
            Some(hot_v) => concatenate(&cold_v_dequant, &hot_v, 2),
            None => cold_v_dequant,
        };
        (k_slice, full_v)
    }

    /// Get the total memory size of the cached keys and values in bytes.
    ///
    /// In INT8 mode this includes both the INT8 buffers and the scale tensors.
    /// In Turbo4Asym mode this counts the FP16 keys plus the packed-V and
    /// V-norm sidecars; the original FP16 values tensor is not stored.
    /// In Turbo4 (symmetric) mode this counts only the K- and V-side packed
    /// sidecars; both the FP16 keys and FP16 values tensors are absent.
    /// In Turbo4Delegated mode (K unified; cold-V dequant memo retired) this counts the unified FP16 K buffer (under
    /// `keys`, identical footprint to `Fp16` mode), the FP16 V hot ring
    /// (under `values`), and the packed cold V sidecars
    /// (`v_packed`/`v_norms`/`v_rescale`). There is no separate cold-K
    /// buffer (removed it) and no FP16 cold-V working set (replaced the earlier memo with a fused kernel that reads packed cold V directly). When `MLXCEL_TURBO4_DELEGATED_FP16_FAST_PATH=1`,
    /// `values` is instead a unified FP16 V working set and `nbytes()` counts
    /// both that buffer and the packed sidecars by design.
    /// In Turbo3Asym mode this counts the FP16 keys plus the 3-bit packed-V
    /// and V-norm sidecars; the underlying `array_nbytes` already accounts
    /// for the smaller per-token byte count (`head_dim * 3 / 8` u8 vs the
    /// 4-bit path's `head_dim / 2`), so the headline number reflects the
    /// ~5.1× total compression vs FP16.
    pub fn nbytes(&self) -> usize {
        let k_bytes = self.keys.as_ref().map_or(0, |k| ffi::array_nbytes(k));
        let v_bytes = self.values.as_ref().map_or(0, |v| ffi::array_nbytes(v));
        let ks_bytes = self.key_scales.as_ref().map_or(0, |k| ffi::array_nbytes(k));
        let vs_bytes = self.val_scales.as_ref().map_or(0, |v| ffi::array_nbytes(v));
        let vp_bytes = self.v_packed.as_ref().map_or(0, |v| ffi::array_nbytes(v));
        let vn_bytes = self.v_norms.as_ref().map_or(0, |v| ffi::array_nbytes(v));
        let vr_bytes = self.v_rescale.as_ref().map_or(0, |v| ffi::array_nbytes(v));
        let kp_bytes = self.k_packed.as_ref().map_or(0, |v| ffi::array_nbytes(v));
        let kn_bytes = self.k_norms.as_ref().map_or(0, |v| ffi::array_nbytes(v));
        // retired the earlier `cold_v_dequant_cache` memo: the
        // fused kernel reads packed cold V directly so there is no longer a
        // FP16 cold-V working set to count here.
        k_bytes
            + v_bytes
            + ks_bytes
            + vs_bytes
            + vp_bytes
            + vn_bytes
            + vr_bytes
            + kp_bytes
            + kn_bytes
    }

    /// Force MLX to materialise the KV cache state without touching the logit
    /// tensors returned by the model's `forward` pass.
    ///
    /// Upstream mlx-lm evaluates `[c.state for c in cache]` after each
    /// chunked-prefill step (see https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py ~line 583). This method
    /// mirrors that pattern: it calls `ffi::eval` on every non-`None` tensor
    /// field that contributes to the cache state (`keys`, `values`,
    /// `key_scales`, `val_scales`, `v_packed`, `v_norms`, `v_rescale`,
    /// `k_packed`, `k_norms`), then returns. Evaluating only the cache state
    /// avoids forcing the LM-head matmul (and the resulting peak allocation)
    /// that would follow from evaluating the full `[1, step, vocab_size]`
    /// logits tensor.
    ///
    /// Used by: `SpeculativeGenerator::generate` (chunked prefill loop)
    pub fn eval_state(&self) {
        if let Some(k) = self.keys.as_ref() {
            ffi::eval(k);
        }
        if let Some(v) = self.values.as_ref() {
            ffi::eval(v);
        }
        if let Some(ks) = self.key_scales.as_ref() {
            ffi::eval(ks);
        }
        if let Some(vs) = self.val_scales.as_ref() {
            ffi::eval(vs);
        }
        if let Some(vp) = self.v_packed.as_ref() {
            ffi::eval(vp);
        }
        if let Some(vn) = self.v_norms.as_ref() {
            ffi::eval(vn);
        }
        if let Some(vr) = self.v_rescale.as_ref() {
            ffi::eval(vr);
        }
        if let Some(kp) = self.k_packed.as_ref() {
            ffi::eval(kp);
        }
        if let Some(kn) = self.k_norms.as_ref() {
            ffi::eval(kn);
        }
    }

    /// Returns `true` iff this cache holds a packed V (the Turbo4 family)
    /// and the sparse-V threshold is enabled (see
    /// [`turbo::sparse_v::is_enabled`]).
    ///
    /// Sparse-V is currently supported only for `KVCacheMode::Turbo4Asym`.
    /// `Turbo4Delegated` is *not* yet wired through `sparse_v_attention`
    /// because that mode splits the visible token range across cold packed
    /// V (`v_packed[0..cold_offset]`) and hot FP16 V (`hot_values[..]`),
    /// while the current dispatcher in [`Self::sparse_v_attention`] slices a
    /// single contiguous `0..self.offset` range. Wiring delegated mode
    /// through sparse-V requires a hot+cold composition pass and is
    /// deferred to a follow-up issue. Symmetric `Turbo4` (K also packed) is
    /// also not yet wired through the split-SDPA path. Callers that want to
    /// opt into the attention-gated dequant path should check this accessor
    /// and, if true, use [`Self::sparse_v_attention`] instead of the
    /// standard [`Self::update_and_fetch`] + `attention()` pair.
    ///
    /// Used by: future model attention call sites (integration deferred —
    /// see `cache/turbo/sparse_v.rs` module docs).
    pub fn sparse_v_available(&self) -> bool {
        if !turbo::sparse_v::is_enabled() {
            return false;
        }
        matches!(self.mode, KVCacheMode::Turbo4Asym)
    }

    /// Borrow the packed V indices for sparse-V attention.
    ///
    /// Returns `None` when sparse-V is not active (see
    /// [`Self::sparse_v_available`]) or before the first
    /// `update_and_fetch` call has populated the V sidecars.
    ///
    /// Used by: [`Self::sparse_v_attention`].
    pub fn v_packed(&self) -> Option<&MlxArray> {
        self.v_packed.as_deref()
    }

    /// Borrow the per-token V norms paired with [`Self::v_packed`].
    pub fn v_norms(&self) -> Option<&MlxArray> {
        self.v_norms.as_deref()
    }

    /// Borrow the per-token V Sparse-V kernel rescale paired with
    /// [`Self::v_packed`].
    ///
    /// Stores `norm[t] / max(|y_hat[t]|, 1e-10)` in fp16 — the exact value the
    /// fused kernel previously derived per-token via a threadgroup tree
    /// reduction. Populated by Turbo4Asym / Turbo4 / Turbo4Delegated update
    /// paths; `None` for non-Turbo4 modes.
    pub fn v_rescale(&self) -> Option<&MlxArray> {
        self.v_rescale.as_deref()
    }

    /// Borrow the cached `TurboQuantParams` (sign vectors + codebook).
    ///
    /// Returns `None` until the first Turbo4 update has lazy-initialised
    /// the params (head_dim is known only at that point).
    pub fn turbo_params(&self) -> Option<&turbo::TurboQuantParams> {
        self.turbo_params.as_ref()
    }

    /// Attention-gated SDPA dispatch: routes to the fused Sparse-V Metal
    /// kernel when available, otherwise falls back to the graph reference
    /// [`turbo::sparse_v::attention_sparse_v_turbo4`] path. Returns `None`
    /// when sparse-V is inactive so the caller can use the standard
    /// `attention()` path.
    ///
    /// **Contract.** This preserves the full-dequant attention result within
    /// FP16 round-off at `threshold=0`. At positive thresholds, positions with
    /// attention weight below the configured cutoff are skipped. On macOS with
    /// a supported power-of-two head dimension the skip happens inside the
    /// fused Metal kernel; elsewhere the graph fallback remains a correctness
    /// reference and still pays full V dequant cost.
    ///
    /// # Inputs
    ///
    /// - `q`: `[B, Hq, Tq, D]` query tensor
    /// - `k`: `[B, Hkv, Tk, D]` key tensor (already FP16 — Turbo4Asym
    ///   keeps K in FP16, Turbo4Delegated returns FP16 from the read path)
    /// - `scale`: attention scale factor (typically `1 / sqrt(d)`)
    /// - `mask`: optional additive mask
    ///
    /// # Output
    ///
    /// `Some([B, Hq, Tq, D])` FP16 attention output when sparse-V was
    /// applied, `None` otherwise (caller falls back to standard SDPA).
    ///
    /// Used by: future model attention call sites that have been ported
    /// to the split-SDPA path. Standard call sites continue to use
    /// `cache.update_and_fetch(...)` followed by `attention(...)`.
    /// Combined update + sparse-V attention dispatch.
    ///
    /// The standard `update_and_fetch + attention()` pair pays the full V
    /// dequant cost inside `update_and_fetch` even when sparse-V is enabled.
    /// This method side-steps that by:
    ///
    /// 1. Calling [`Self::update`] to fill the packed buffers (no V dequant).
    /// 2. Reading the K side (FP16 for `Turbo4Asym`).
    /// 3. Dispatching [`Self::sparse_v_attention`] directly with the packed
    ///    V buffers — the fused Metal kernel does the per-thread dequant +
    ///    skip in one pass.
    ///
    /// Returns:
    /// - `Some(attn_out)` when sparse-V is active and the kernel/graph path
    ///   handled the dispatch. The caller uses this output and skips the
    ///   standard `attention()` call.
    /// - `None` when sparse-V is *not* active. The caller falls back to
    ///   `update_and_fetch + attention()`.
    ///
    /// The Q tensor must already have RoPE / Q-norm applied; the caller is
    /// responsible for that, just as with the standard path.
    ///
    /// Used by: per-model attention call sites (Llama 3, Qwen 3, etc.)
    /// when the cache is in `Turbo4Asym` mode and
    /// `MLXCEL_SPARSE_V_THRESHOLD > 0`. `Turbo4Delegated` integration is
    /// deferred — see [`Self::sparse_v_available`].
    pub fn update_and_sparse_v_attention(
        &mut self,
        q: &MlxArray,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
        scale: f32,
        mask: Option<&MlxArray>,
    ) -> Option<UniquePtr<MlxArray>> {
        if !self.sparse_v_available() {
            return None;
        }
        // Fill the packed buffers; this also advances `self.offset`.
        self.update(new_keys, new_values);

        // For Turbo4Asym K stays FP16 in `self.keys` and the visible token
        // range is contiguous (`0..self.offset`). Slice it directly.
        // (`Turbo4` symmetric mode would need a packed K dequant here, and
        // `Turbo4Delegated` would need a hot+cold composition; both are
        // excluded from `sparse_v_available()`.)
        let k_buf = self.keys.as_ref()?;
        let ks = ffi::array_shape(k_buf);
        let k_slice = ffi::slice(k_buf, &[0, 0, 0, 0], &[ks[0], ks[1], self.offset, ks[3]]);

        self.sparse_v_attention(q, &k_slice, scale, mask)
    }

    pub fn sparse_v_attention(
        &self,
        q: &MlxArray,
        k: &MlxArray,
        scale: f32,
        mask: Option<&MlxArray>,
    ) -> Option<UniquePtr<MlxArray>> {
        if !self.sparse_v_available() {
            return None;
        }
        let v_packed_buf = self.v_packed()?;
        let v_norms_buf = self.v_norms()?;
        let v_rescale_buf = self.v_rescale()?;
        let params = self.turbo_params()?;
        let threshold = turbo::sparse_v::threshold();

        // Slice the packed V buffers down to the visible token range. The
        // raw `v_packed`/`v_norms`/`v_rescale` accessors return the full
        // buffer, which includes pre-allocated capacity beyond `self.offset`.
        // Without the slice the kernel would sweep `Tk = buffer_seq_len`,
        // paying for dead capacity slots and producing wrong contributions.
        //
        // For Turbo4Asym the packed shapes are:
        //   v_packed:  [B, H, capacity, D/2] u8
        //   v_norms:   [B, H, capacity, 1]   f16
        //   v_rescale: [B, H, capacity, 1]   f16
        // We slice axis 2 (the token axis) to [0, self.offset).
        let vp_shape = ffi::array_shape(v_packed_buf);
        let vn_shape = ffi::array_shape(v_norms_buf);
        let vr_shape = ffi::array_shape(v_rescale_buf);
        let v_packed_owned = ffi::slice(
            v_packed_buf,
            &[0, 0, 0, 0],
            &[vp_shape[0], vp_shape[1], self.offset, vp_shape[3]],
        );
        let v_norms_owned = ffi::slice(
            v_norms_buf,
            &[0, 0, 0, 0],
            &[vn_shape[0], vn_shape[1], self.offset, 1],
        );
        let v_rescale_owned = ffi::slice(
            v_rescale_buf,
            &[0, 0, 0, 0],
            &[vr_shape[0], vr_shape[1], self.offset, 1],
        );

        // Prefer the fused Metal kernel path (+) when
        // available. Falls through to the graph-level reference path on
        // non-macOS, when the kernel is disabled via
        // `MLXCEL_SPARSE_V_KERNEL=0`, or when the model uses a non-power-of-2
        // head_dim (Gemma 4 192-dim heads).
        //
        // The fused kernel reads the precomputed `v_rescale` (norm/|y_hat|)
        // directly, eliminating the per-token threadgroup tree reduction
        // that previously dominated decode latency on M5 Max at 4 K context
        // The graph fallback continues to use `v_norms` only.
        if let Some(out) = turbo::sparse_v::attention_sparse_v_turbo4_fused(
            q,
            k,
            &v_packed_owned,
            &v_rescale_owned,
            params,
            scale,
            mask,
            threshold,
        ) {
            return Some(out);
        }
        Some(turbo::sparse_v::attention_sparse_v_turbo4(
            q,
            k,
            &v_packed_owned,
            &v_norms_owned,
            params,
            scale,
            mask,
            threshold,
        ))
    }

    /// Returns `true` iff this cache can route symmetric `Turbo4` through the
    /// Swift-LM-style dequant-first SDPA path.
    pub fn turbo4_dequant_sdpa_available(&self) -> bool {
        self.mode == KVCacheMode::Turbo4
    }

    /// Combined update + symmetric Turbo4 dequant-first SDPA dispatch.
    ///
    /// This mirrors `references/mlx-swift-lm`'s default compressed K/V route
    /// for the Rust symmetric `Turbo4` mode: cache state stays packed, K/V are
    /// transiently dequantized in their rotated codec bases for the current
    /// attention call, Q is forward-rotated into the K basis, and only the
    /// small output tensor is inverse-rotated through the V basis.
    ///
    /// Used by: model attention call sites when `mode == KVCacheMode::Turbo4`
    /// and [`turbo::sparse_v::turbo4_dequant_sdpa_enabled`] is true.
    pub fn update_and_turbo4_dequant_sdpa_attention(
        &mut self,
        q: &MlxArray,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
        scale: f32,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        assert!(
            self.turbo4_dequant_sdpa_available(),
            "update_and_turbo4_dequant_sdpa_attention called on a cache that is not in \
             Turbo4 mode (mode={:?})",
            self.mode
        );
        self.update(new_keys, new_values);

        let kp = self.k_packed.as_ref().expect("k_packed must exist");
        let kn = self.k_norms.as_ref().expect("k_norms must exist");
        let vp = self.v_packed.as_ref().expect("v_packed must exist");
        let vr = self.v_rescale.as_ref().expect("v_rescale must exist");
        let params = self
            .turbo_params
            .as_ref()
            .expect("turbo_params must be initialised after first update_turbo4_sym");

        let kp_shape = ffi::array_shape(kp);
        let kn_shape = ffi::array_shape(kn);
        let vp_shape = ffi::array_shape(vp);
        let vr_shape = ffi::array_shape(vr);
        let kp_slice = ffi::slice(
            kp,
            &[0, 0, 0, 0],
            &[kp_shape[0], kp_shape[1], self.offset, kp_shape[3]],
        );
        let kn_slice = ffi::slice(
            kn,
            &[0, 0, 0, 0],
            &[kn_shape[0], kn_shape[1], self.offset, 1],
        );
        let vp_slice = ffi::slice(
            vp,
            &[0, 0, 0, 0],
            &[vp_shape[0], vp_shape[1], self.offset, vp_shape[3]],
        );
        let vr_slice = ffi::slice(
            vr,
            &[0, 0, 0, 0],
            &[vr_shape[0], vr_shape[1], self.offset, 1],
        );

        turbo::sparse_v::attention_turbo4_dequant_sdpa(
            q, &kp_slice, &kn_slice, &vp_slice, &vr_slice, params, scale, mask,
        )
    }

    /// Returns `true` iff this cache is in `KVCacheMode::Turbo4Delegated` mode
    /// and the fused dequant + SDPA decode path is reachable.
    ///
    /// The fused kernel is Apple-Silicon-only (the runtime JIT requires Metal)
    /// and supports the Turbo4Delegated mode regardless of whether
    /// `MLXCEL_SPARSE_V_THRESHOLD` is set — the threshold is an additional
    /// per-token skip optimisation, not a gate. Callers that want the kernel
    /// path should check this accessor and, if true, use
    /// [`Self::update_and_turbo4_delegated_attention`] in place of the
    /// standard [`Self::update_and_fetch`] + `attention()` pair.
    ///
    /// Used by: per-model attention call sites that have been ported to the
    /// fused-kernel path. Standard call sites continue to use
    /// `cache.update_and_fetch(...)` followed by `attention(...)`.
    pub fn turbo4_delegated_available(&self) -> bool {
        self.mode == KVCacheMode::Turbo4Delegated
    }

    /// Combined update + Turbo4Delegated attention dispatch.
    ///
    /// The standard `update_and_fetch + attention()` pair pays the full
    /// `dequantize_v_turbo4(v_packed[:cold_offset])` graph cost on every
    /// decode step, plus a `concat(cold_v_dequant, hot_v, axis=2)` that
    /// scales with `cold_offset`. The default compressed path now mirrors
    /// `references/mlx-swift-lm`: it dequantizes cold V only into the rotated
    /// codec basis, runs native SDPA, and inverse-rotates the small output.
    /// If `MLXCEL_TURBO4_DELEGATED_DEQUANT_SDPA=0`, the function falls back
    /// to the custom packed-V Metal kernels:
    ///
    /// 1. [`Self::update`] fills the unified K buffer, the V hot ring, and
    ///    (when `hot_offset > hot_threshold`) the packed cold V sidecars.
    /// 2. [`turbo::sparse_v::attention_turbo4_delegated_steel`] or
    ///    [`turbo::sparse_v::attention_turbo4_delegated_fused`] runs Q·K
    ///    against the unified K, softmax, the packed cold-V contribution, a
    ///    small hot-V contribution, and sums the two.
    ///
    /// When the fused kernel is gated off (non-macOS, non-power-of-2 head
    /// dim, or `MLXCEL_SPARSE_V_KERNEL=0`) this function falls through to a
    /// graph-only reference path inside the same call: it dequantises the
    /// cold V tokens transiently (no memo retention — the V-budget guard in
    /// [`crate::cache::turbo_tests::delegated_per_token_v_budget_after_issue_528`]
    /// continues to pass) and runs standard SDPA on the resulting FP16
    /// `[B, Hkv, T_total, D]` V tensor. Either way the cache state after the
    /// call is identical to the kernel-path state.
    ///
    /// **Panics.** This function is intended for callers that have already
    /// verified [`Self::turbo4_delegated_available`] returns `true`. It panics
    /// otherwise — that is a programming error.
    ///
    /// The Q tensor must already have RoPE / Q-norm applied; the caller is
    /// responsible for that, just as with the standard path.
    ///
    /// Used by: per-model attention call sites (Llama 3, Qwen 3, etc.) when
    /// the cache is in `Turbo4Delegated` mode and
    /// [`turbo::sparse_v::turbo4_delegated_compressed_attention_enabled`]
    /// returns `true`.
    pub fn update_and_turbo4_delegated_attention(
        &mut self,
        q: &MlxArray,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
        scale: f32,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        assert!(
            self.turbo4_delegated_available(),
            "update_and_turbo4_delegated_attention called on a cache that is not in \
             Turbo4Delegated mode (mode={:?})",
            self.mode
        );
        // Fill the unified K buffer + V hot ring (and possibly fold hot V
        // into cold packed storage). After this call `self.offset` and
        // `self.cold_offset` are up to date.
        self.update(new_keys, new_values);

        if self.delegated_fp16_fast_path {
            return self.delegated_graph_attention(q, scale, mask);
        }

        let cold_offset = self.cold_offset;
        let hot_offset = self.offset - cold_offset;
        debug_assert!(
            hot_offset >= 0,
            "hot_offset must be non-negative (offset={}, cold_offset={cold_offset})",
            self.offset
        );

        // turbo_params is initialised lazily on the first fold (inside
        // `update_turbo4_delegated`). Before the first fold (cold_offset == 0)
        // the kernel is not dispatched on the cold side but the rotation /
        // codebook references are still part of the call signature. Eagerly
        // initialise params here so we can pass them down without juggling
        // mutable borrows below; subsequent folds will reuse this instance.
        if self.turbo_params.is_none() {
            // Use the new_values head_dim we just stored. After `update` the
            // unified K head_dim equals the V head_dim only on Turbo4
            // (symmetric); for delegated the K and V head_dims may differ in
            // principle, so probe V from the hot ring or packed sidecars.
            let head_dim_u32 = self
                .values
                .as_ref()
                .map(|v| ffi::array_shape(v)[3] as u32)
                .or_else(|| {
                    self.v_packed
                        .as_ref()
                        .map(|vp| (ffi::array_shape(vp)[3] as u32) * 2)
                })
                .expect("either hot V ring or v_packed must exist after first update");
            self.turbo_params = Some(turbo::TurboQuantParams::new(head_dim_u32, self.turbo_seed));
        }

        // Slice the unified K buffer down to the visible token range
        // [0, offset). Same shape contract as `KVCacheMode::Fp16`.
        let k_buf = self
            .keys
            .as_ref()
            .expect("unified keys must exist after update on Turbo4Delegated");
        let ks = ffi::array_shape(k_buf);
        let k_slice = ffi::slice(k_buf, &[0, 0, 0, 0], &[ks[0], ks[1], self.offset, ks[3]]);

        // Slice the cold V packed sidecars down to [0, cold_offset). When
        // `cold_offset == 0` we skip these — the fused kernel only sweeps
        // the cold range, so empty cold means the kernel call is skipped on
        // the host side.
        let v_packed_owned = if cold_offset > 0 {
            let vp = self
                .v_packed
                .as_ref()
                .expect("v_packed must exist when cold_offset > 0");
            let vp_shape = ffi::array_shape(vp);
            Some(ffi::slice(
                vp,
                &[0, 0, 0, 0],
                &[vp_shape[0], vp_shape[1], cold_offset, vp_shape[3]],
            ))
        } else {
            None
        };
        let v_rescale_owned = if cold_offset > 0 {
            let vr = self
                .v_rescale
                .as_ref()
                .expect("v_rescale must exist when cold_offset > 0");
            let vr_shape = ffi::array_shape(vr);
            Some(ffi::slice(
                vr,
                &[0, 0, 0, 0],
                &[vr_shape[0], vr_shape[1], cold_offset, 1],
            ))
        } else {
            None
        };

        // Slice the V hot ring down to [0, hot_offset).
        let hot_v_owned = if hot_offset > 0 {
            let hv = self
                .values
                .as_ref()
                .expect("hot values must exist when hot_offset > 0");
            let hv_shape = ffi::array_shape(hv);
            Some(ffi::slice(
                hv,
                &[0, 0, 0, 0],
                &[hv_shape[0], hv_shape[1], hot_offset, hv_shape[3]],
            ))
        } else {
            None
        };

        let params = self
            .turbo_params
            .as_ref()
            .expect("turbo_params eagerly initialised above");

        let threshold = turbo::sparse_v::threshold();
        let v_packed_ref = v_packed_owned.as_deref();
        let v_rescale_ref = v_rescale_owned.as_deref();
        let hot_v_ref = hot_v_owned.as_deref();

        // Swift-LM reference path — dequant cold packed V in rotated value
        // basis, run native SDPA, then inverse-rotate only the small output.
        // This keeps persistent cold V compressed while avoiding an inverse
        // WHT over every cold token on each graph fallback. It is the default
        // compressed path; set `MLXCEL_TURBO4_DELEGATED_DEQUANT_SDPA=0` to
        // compare against the custom steel/cold-only order below.
        if turbo::sparse_v::turbo4_delegated_dequant_sdpa_enabled() {
            if let Some(out) = turbo::sparse_v::attention_turbo4_delegated_dequant_sdpa(
                q,
                &k_slice,
                v_packed_ref,
                v_rescale_ref,
                hot_v_ref,
                params,
                cold_offset,
                hot_offset,
                scale,
                mask,
            ) {
                return out;
            }
        }

        // Prefer the steel-attention-envelope kernel when
        // available. It collapses softmax + cold-V dequant + hot-V accum into
        // a single Metal dispatch, eliminating the per-step FFI / dispatch
        // overhead of the host composition path. The cold-only kernel from
        // remains as a fallback for the same correctness contract.
        if let Some(out) = turbo::sparse_v::attention_turbo4_delegated_steel(
            q,
            &k_slice,
            v_packed_ref,
            v_rescale_ref,
            hot_v_ref,
            params,
            cold_offset,
            hot_offset,
            scale,
            mask,
            threshold,
        ) {
            return out;
        }

        // Steel envelope unavailable (non-macOS, non-power-of-2 head_dim, or
        // `MLXCEL_SPARSE_V_KERNEL=0`). Try the cold-only fused composition
        // before falling all the way back to the graph SDPA.
        // The cold-only path keeps the V-budget guarantee and is still
        // measurably faster than the full graph fallback for some
        // configurations.
        if let Some(out) = turbo::sparse_v::attention_turbo4_delegated_fused(
            q,
            &k_slice,
            v_packed_ref,
            v_rescale_ref,
            hot_v_ref,
            params,
            cold_offset,
            hot_offset,
            scale,
            mask,
            threshold,
        ) {
            return out;
        }

        // All fused kernel paths rejected the dispatch. Cache state is
        // already up to date from the `self.update(...)` above, so we run the
        // same graph SDPA reference path the legacy `update_and_fetch +
        // attention()` pair would produce. The transient FP16 cold V dequant
        // materialised here is thrown away as soon as the SDPA call finishes
        // — no memo is retained.
        self.delegated_graph_attention(q, scale, mask)
    }

    /// Graph-only Turbo4Delegated attention reference (fallback).
    ///
    /// Computes attention against the **already-updated** Turbo4Delegated cache
    /// using the legacy compositional path:
    ///
    /// 1. [`Self::fetch_turbo4_delegated`] returns the unified K slice and a
    ///    fresh `concat(cold_v_dequant, hot_v, axis=2)` FP16 V tensor. The
    ///    cold dequant is materialised transiently for this call and not
    ///    retained anywhere on `self` — the V-budget guard
    ///    (`delegated_per_token_v_budget_after_issue_528`) continues to pass.
    /// 2. [`crate::layers::attention`] runs SDPA against that K/V pair with
    ///    the supplied scale and optional additive mask.
    ///
    /// Bit-equivalent (within FP16 round-off) to running
    /// `let (k, v) = cache.update_and_fetch(...); attention(&q, &k, &v, ...)`
    /// after a `cache.update(...)` has already been applied.
    ///
    /// Used by: [`Self::update_and_turbo4_delegated_attention`] when the
    /// fused Metal kernel is gated off, and by tests that need to verify the
    /// graph-fallback path without manipulating
    /// `MLXCEL_SPARSE_V_KERNEL` (which is cached in a `OnceLock` and cannot
    /// be flipped between cargo-test cases reliably).
    pub fn delegated_graph_attention(
        &mut self,
        q: &MlxArray,
        scale: f32,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        assert!(
            self.turbo4_delegated_available(),
            "delegated_graph_attention called on a cache that is not in \
             Turbo4Delegated mode (mode={:?})",
            self.mode
        );
        let (k_full, v_full) = self.fetch_turbo4_delegated();
        crate::layers::attention(q, &k_full, &v_full, scale, mask, 0.0, 0)
    }

    /// Estimated storage bytes per reserved token slot in the backing buffer.
    ///
    /// This uses the allocated buffer capacity rather than the visible offset
    /// so callers can mirror dense-cache physical storage into paged block
    /// accounting even when the buffer is step-allocated.
    pub fn bytes_per_reserved_token(&self) -> usize {
        let capacity = self.buffer_seq_len();
        if capacity <= 0 {
            return 0;
        }
        self.nbytes() / capacity as usize
    }
}

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Rotating KV Cache for sliding window attention (e.g. Gemma 3, Ministral 3).
///
/// Maintains a fixed-size circular buffer for keys/values. Oversized prefill
/// is linearized before single-token decode so wraparound stays well-defined.
///
/// # KV Cache Quantization
///
/// `RotatingKVCache` supports the same `KVCacheMode` set as `KVCache`:
/// `Fp16` (default), `Int8`, and `Turbo4Asym`. The mode is selected at
/// construction time via [`RotatingKVCache::new_with_mode`].
///
/// In `KVCacheMode::Turbo4Asym` the V buffer is replaced by `v_packed`
/// (`[B, H, max_size, head_dim/2]` UINT8) plus `v_norms`
/// (`[B, H, max_size, 1]` FP16); the `values` field stays `None`. K stays
/// FP16 in the regular `keys` buffer.
///
/// ## Block alignment invariant (B9)
///
/// TurboQuant V quantization is **per-token**: each token's `head_dim` vector
/// is rotated and quantized independently along the last axis. There is no
/// cross-token shared state in the rotation pipeline (the `BLOCK_SIZE` = 32
/// constant only governs buffer growth granularity for the dense `KVCache`).
///
/// As a result, the ring-buffer wraparound at position `idx` is correct
/// **for any** `max_size` — single-token writes simply overwrite the packed
/// bytes and the per-token norm at `idx`, regardless of where `idx` falls.
///
/// However, we still require `max_size` to be a multiple of
/// [`turbo::BLOCK_SIZE`] (32) when `mode == Turbo4Asym` for two reasons:
///
/// 1. **Future-proofing**: B5 (3-bit packing) introduces 24-bit
///    groups that span byte boundaries. A non-32-aligned ring would force
///    partial-block re-quantization on every wraparound. Locking the
///    invariant here means B5 only has to revisit pack/unpack, not the cache.
/// 2. **Trim path simplicity**: the dense `KVCache::trim` path already
///    exploits 32-alignment to avoid mid-block re-quantize work. Holding the
///    same invariant in `RotatingKVCache` keeps speculative-decode rewinds
///    bit-identical between dense and rotating caches.
///
/// All sliding-window models we ship today (Gemma 3 4 K, Gemma 4 8 K,
/// Ministral 3 8 K, GPT-OSS 4 K / 16 K, RecurrentGemma 2 K, Exaone 4 K)
/// already use 32-divisible window sizes, so this constraint is non-binding
/// in practice. The constructor enforces it with a clear error message
/// rather than silently accepting an invalid configuration.
///
/// Used by: Gemma3, Gemma4, Ministral3, GPT-OSS, RecurrentGemma, Exaone
pub struct RotatingKVCache {
    pub keys: Option<UniquePtr<MlxArray>>,
    pub values: Option<UniquePtr<MlxArray>>,
    /// Logical sliding-window length used by attention.
    pub max_size: i32,
    /// Extra speculative rollback slack. When non-zero, the cache keeps a
    /// temporal prefix of up to `max_size + buffer_size` tokens and compacts
    /// from the front instead of destructively wrapping as soon as the logical
    /// sliding window is full.
    pub buffer_size: i32,
    pub offset: i32,
    /// Absolute logical position of `keys[..., 0, :]` when `buffer_size > 0`.
    start_position: i32,
    /// Current write position in the buffer (separate from offset to handle trim correctly)
    idx: i32,
    step: i32,
    /// Quantization mode for stored keys/values.
    pub mode: KVCacheMode,
    /// INT8-mode per-token K scale buffer: `[B, H, max_size, 1]` FP16.
    pub(crate) key_scales: Option<UniquePtr<MlxArray>>,
    /// INT8-mode per-token V scale buffer: `[B, H, max_size, 1]` FP16.
    pub(crate) val_scales: Option<UniquePtr<MlxArray>>,
    /// Turbo4Asym-mode V-side packed indices: `[B, H, max_size, head_dim/2]` u8.
    pub(crate) v_packed: Option<UniquePtr<MlxArray>>,
    /// Turbo4Asym-mode V-side per-token norms: `[B, H, max_size, 1]` fp16.
    pub(crate) v_norms: Option<UniquePtr<MlxArray>>,
    /// Turbo4-V precomputed Sparse-V kernel rescale `norm[t] / |y_hat[t]|`
    /// Same shape and lockstep lifecycle as `v_norms`. See the
    /// dense `KVCache::v_rescale` docs for the rationale.
    pub(crate) v_rescale: Option<UniquePtr<MlxArray>>,
    /// Cached PolarQuant params (sign vectors + codebook) for Turbo4Asym.
    /// Lazily initialised on the first Turbo4 update once V `head_dim` is
    /// known; deterministic given `turbo_seed` so detach/adopt round-trips
    /// without recomputing rotations.
    pub(crate) turbo_params: Option<turbo::TurboQuantParams>,
    /// Deterministic seed for the Turbo4 sign vectors. Set at construction
    /// time so detach/adopt round-trip without recomputing rotations.
    pub(crate) turbo_seed: u32,
}

impl RotatingKVCache {
    /// Create a new rotating KV cache with specified maximum size (FP16 mode).
    pub fn new(max_size: i32) -> Self {
        Self::new_with_mode_and_seed(max_size, KVCacheMode::Fp16, TURBO_DEFAULT_SEED)
    }

    /// Create a new rotating KV cache with the specified maximum size and
    /// quantization mode.
    ///
    /// `Turbo4Asym` requires `max_size` to be a positive multiple of
    /// [`turbo::BLOCK_SIZE`] — see the type-level docs for rationale.
    ///
    /// # Panics
    ///
    /// Panics if `mode == Turbo4Asym` and `max_size` is not a positive
    /// multiple of `turbo::BLOCK_SIZE`. Misconfiguring this would leave the
    /// trim path silently broken; failing fast at construction is the only
    /// correct contract.
    pub fn new_with_mode(max_size: i32, mode: KVCacheMode) -> Self {
        Self::new_with_mode_and_seed(max_size, mode, TURBO_DEFAULT_SEED)
    }

    /// Like [`RotatingKVCache::new_with_mode`] but lets the caller pin a
    /// specific Turbo4 sign-vector seed.
    ///
    /// Production callers should prefer `new_with_mode`; the explicit seed is
    /// exposed for tests, benchmarks, and the detach/adopt path which must
    /// preserve the original seed.
    pub fn new_with_mode_and_seed(max_size: i32, mode: KVCacheMode, turbo_seed: u32) -> Self {
        if mode == KVCacheMode::Turbo4Asym {
            assert!(
                max_size > 0 && max_size % turbo::BLOCK_SIZE == 0,
                "RotatingKVCache::new_with_mode: max_size must be a positive multiple of \
                 turbo::BLOCK_SIZE ({}); got {max_size}. See type docs for rationale.",
                turbo::BLOCK_SIZE
            );
        }
        Self {
            keys: None,
            values: None,
            max_size,
            buffer_size: 0,
            offset: 0,
            start_position: 0,
            idx: 0,
            step: 256,
            mode,
            key_scales: None,
            val_scales: None,
            v_packed: None,
            v_norms: None,
            v_rescale: None,
            turbo_params: None,
            turbo_seed,
        }
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.keys.is_none()
    }

    /// Get current sequence length in cache
    pub fn seq_len(&self) -> i32 {
        if self.buffer_size > 0 {
            return self.idx.max(0);
        }
        if let Some(ref keys) = self.keys {
            let shape = ffi::array_shape(keys);
            if shape.len() >= 3 {
                shape[2]
            } else {
                0
            }
        } else {
            0
        }
    }

    /// Update cache with new key/value, rotating if necessary.
    ///
    /// Returns the full cached keys/values, always in FP16 — the public
    /// contract is mode-independent. The `Turbo4Asym` path quantizes V on
    /// write and dequantizes the visible window on read so attention kernels
    /// see standard FP16 tensors regardless of stored representation.
    pub fn update_and_fetch(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        match self.mode {
            KVCacheMode::Fp16 => self.update_and_fetch_fp16(new_keys, new_values),
            KVCacheMode::Int8 => {
                // INT8 support for RotatingKVCache is not part of B9 / issue
                // Fall back to FP16 storage so the path is correct, even
                // if mis-configured. A future sub-issue can wire INT8 in.
                self.update_and_fetch_fp16(new_keys, new_values)
            }
            KVCacheMode::Turbo4Asym => self.update_and_fetch_turbo4_asym(new_keys, new_values),
            KVCacheMode::Turbo4 => {
                // Symmetric Turbo4 is not wired into RotatingKVCache
                // by B9 / (RotatingKVCache currently supports only
                // Fp16/Int8/Turbo4Asym). Fall back to FP16 so a mis-configured
                // sliding-window model does not panic; a future sub-issue can
                // wire symmetric K/V quantization in.
                self.update_and_fetch_fp16(new_keys, new_values)
            }
            KVCacheMode::Turbo4Delegated => {
                // Delegated hot/cold is not wired into RotatingKVCache
                // by B7 / (the delegated path targets dense caches).
                // Fall back to FP16 so a mis-configured sliding-window model does
                // not panic.
                self.update_and_fetch_fp16(new_keys, new_values)
            }
            KVCacheMode::Turbo3Asym => {
                // Turbo3Asym is not wired into RotatingKVCache in
                // this PR — wraparound + 3-bit re-pack alignment requires its
                // own analysis (mirrors B9 / for Turbo4Asym). Fall
                // back to FP16 so sliding-window models with `--kv-cache-mode
                // fp16+turbo3` do not panic; a follow-up sub-issue can wire
                // Turbo3 into the rotating path once dense Turbo3 is validated.
                self.update_and_fetch_fp16(new_keys, new_values)
            }
        }
    }

    fn update_and_fetch_fp16(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        if self.buffer_size > 0 {
            return self.update_and_fetch_buffered_fp16(new_keys, new_values);
        }

        let new_seq_len = {
            let shape = ffi::array_shape(&new_keys);
            shape[2]
        };

        if new_seq_len > 1 {
            return self.update_concat(new_keys, new_values, new_seq_len);
        }

        self.update_in_place(new_keys, new_values)
    }

    /// Enable upstream-style buffered rotating semantics for speculative
    /// verification bursts.
    ///
    /// This mirrors mlx-vlm's `BufferedRotatingKVCache`: the logical attention
    /// window remains `max_size`, but the backing buffer receives extra
    /// temporary slack (`buffer_size`) so an MTP verify block can append and
    /// then roll back without overwriting the oldest still-visible window
    /// entries. Existing ring contents are converted into temporal order.
    ///
    /// Used by: Gemma 4 MTP target caches near sliding-window rollover.
    pub fn enable_speculative_buffer(&mut self, buffer_size: i32) -> Result<(), String> {
        let buffer_size = buffer_size.max(0);
        if buffer_size <= self.buffer_size && self.buffer_size > 0 {
            return Ok(());
        }
        if self.mode != KVCacheMode::Fp16 {
            return Err(format!(
                "RotatingKVCache::enable_speculative_buffer only supports FP16 storage; \
                 got {:?}",
                self.mode
            ));
        }

        if self.keys.is_none() {
            self.buffer_size = self.buffer_size.max(buffer_size);
            self.idx = 0;
            self.start_position = self.offset;
            return Ok(());
        }
        if self.values.is_none() {
            return Err(
                "RotatingKVCache::enable_speculative_buffer requires a value buffer".into(),
            );
        }

        let (keys, values, visible_len) = self.visible_fp16_prefix_for_concat();
        self.buffer_size = self.buffer_size.max(buffer_size);
        let keep = visible_len.min(self.max_size).max(0);
        let start = visible_len - keep;
        let k_shape = ffi::array_shape(&keys);
        let v_shape = ffi::array_shape(&values);
        let kept_keys = ffi::slice(
            &keys,
            &[0, 0, start, 0],
            &[k_shape[0], k_shape[1], visible_len, k_shape[3]],
        );
        let kept_values = ffi::slice(
            &values,
            &[0, 0, start, 0],
            &[v_shape[0], v_shape[1], visible_len, v_shape[3]],
        );

        let capacity = self.buffered_capacity_for(keep, 0);
        let k_zeros = ffi::zeros(
            &[k_shape[0], k_shape[1], capacity, k_shape[3]],
            ffi::array_dtype(&kept_keys),
        );
        let v_zeros = ffi::zeros(
            &[v_shape[0], v_shape[1], capacity, v_shape[3]],
            ffi::array_dtype(&kept_values),
        );

        self.keys = Some(if keep > 0 {
            ffi::slice_update(
                &k_zeros,
                &kept_keys,
                &[0, 0, 0, 0],
                &[k_shape[0], k_shape[1], keep, k_shape[3]],
            )
        } else {
            k_zeros
        });
        self.values = Some(if keep > 0 {
            ffi::slice_update(
                &v_zeros,
                &kept_values,
                &[0, 0, 0, 0],
                &[v_shape[0], v_shape[1], keep, v_shape[3]],
            )
        } else {
            v_zeros
        });
        self.idx = keep;
        self.start_position = self.offset - keep;
        Ok(())
    }

    fn buffered_target_size(&self, incoming: i32) -> i32 {
        self.max_size + self.buffer_size.max(incoming).max(0)
    }

    fn buffered_capacity_for(&self, needed: i32, incoming: i32) -> i32 {
        let target = needed.max(self.buffered_target_size(incoming));
        ((target + self.step - 1) / self.step) * self.step
    }

    fn buffered_planned_drop(&self, incoming: i32) -> i32 {
        let needed = self.idx + incoming;
        let target_size = self.buffered_target_size(incoming);
        if needed <= target_size {
            return 0;
        }
        let target_start = (self.offset - self.max_size + 1).max(0);
        (target_start - self.start_position).max(0).min(self.idx)
    }

    fn compact_buffered_prefix(&mut self, drop: i32) {
        if drop <= 0 || self.keys.is_none() {
            return;
        }
        let keep = self.idx - drop;
        let keys = self.keys.as_ref().unwrap();
        let values = self
            .values
            .as_ref()
            .expect("buffered rotating cache keeps values with keys");
        let k_shape = ffi::array_shape(keys);
        let v_shape = ffi::array_shape(values);
        let capacity = k_shape[2];
        let kept_keys = ffi::slice(
            keys,
            &[0, 0, drop, 0],
            &[k_shape[0], k_shape[1], self.idx, k_shape[3]],
        );
        let kept_values = ffi::slice(
            values,
            &[0, 0, drop, 0],
            &[v_shape[0], v_shape[1], self.idx, v_shape[3]],
        );
        let k_zeros = ffi::zeros(
            &[k_shape[0], k_shape[1], capacity, k_shape[3]],
            ffi::array_dtype(keys),
        );
        let v_zeros = ffi::zeros(
            &[v_shape[0], v_shape[1], capacity, v_shape[3]],
            ffi::array_dtype(values),
        );
        self.keys = Some(if keep > 0 {
            ffi::slice_update(
                &k_zeros,
                &kept_keys,
                &[0, 0, 0, 0],
                &[k_shape[0], k_shape[1], keep, k_shape[3]],
            )
        } else {
            k_zeros
        });
        self.values = Some(if keep > 0 {
            ffi::slice_update(
                &v_zeros,
                &kept_values,
                &[0, 0, 0, 0],
                &[v_shape[0], v_shape[1], keep, v_shape[3]],
            )
        } else {
            v_zeros
        });
        self.start_position += drop;
        self.idx = keep;
    }

    fn ensure_buffered_capacity(
        &mut self,
        new_keys: &MlxArray,
        new_values: &MlxArray,
        needed: i32,
        incoming: i32,
    ) {
        if let Some(keys) = self.keys.as_ref() {
            if needed <= ffi::array_shape(keys)[2] {
                return;
            }
        }

        let new_k_shape = ffi::array_shape(new_keys);
        let new_v_shape = ffi::array_shape(new_values);
        let capacity = self.buffered_capacity_for(needed, incoming);
        let k_zeros = ffi::zeros(
            &[new_k_shape[0], new_k_shape[1], capacity, new_k_shape[3]],
            ffi::array_dtype(new_keys),
        );
        let v_zeros = ffi::zeros(
            &[new_v_shape[0], new_v_shape[1], capacity, new_v_shape[3]],
            ffi::array_dtype(new_values),
        );

        if self.keys.is_none() || self.idx <= 0 {
            self.keys = Some(k_zeros);
            self.values = Some(v_zeros);
            return;
        }

        let keys = self.keys.as_ref().unwrap();
        let values = self
            .values
            .as_ref()
            .expect("buffered rotating cache keeps values with keys");
        let k_shape = ffi::array_shape(keys);
        let v_shape = ffi::array_shape(values);
        let live_keys = ffi::slice(
            keys,
            &[0, 0, 0, 0],
            &[k_shape[0], k_shape[1], self.idx, k_shape[3]],
        );
        let live_values = ffi::slice(
            values,
            &[0, 0, 0, 0],
            &[v_shape[0], v_shape[1], self.idx, v_shape[3]],
        );
        self.keys = Some(ffi::slice_update(
            &k_zeros,
            &live_keys,
            &[0, 0, 0, 0],
            &[k_shape[0], k_shape[1], self.idx, k_shape[3]],
        ));
        self.values = Some(ffi::slice_update(
            &v_zeros,
            &live_values,
            &[0, 0, 0, 0],
            &[v_shape[0], v_shape[1], self.idx, v_shape[3]],
        ));
    }

    fn update_and_fetch_buffered_fp16(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let incoming = ffi::array_shape(&new_keys)[2];
        let drop = self.buffered_planned_drop(incoming);
        self.compact_buffered_prefix(drop);

        let needed = self.idx + incoming;
        self.ensure_buffered_capacity(
            new_keys.as_ref().unwrap(),
            new_values.as_ref().unwrap(),
            needed,
            incoming,
        );

        let mut k_buffer = self.keys.take().unwrap();
        let mut v_buffer = self.values.take().unwrap();
        let k_shape = ffi::array_shape(&k_buffer);
        let v_shape = ffi::array_shape(&v_buffer);
        let pos = self.idx;
        k_buffer = ffi::slice_update(
            &k_buffer,
            &new_keys,
            &[0, 0, pos, 0],
            &[k_shape[0], k_shape[1], needed, k_shape[3]],
        );
        v_buffer = ffi::slice_update(
            &v_buffer,
            &new_values,
            &[0, 0, pos, 0],
            &[v_shape[0], v_shape[1], needed, v_shape[3]],
        );

        self.idx = needed;
        self.offset += incoming;

        let k_out = ffi::slice(
            &k_buffer,
            &[0, 0, 0, 0],
            &[k_shape[0], k_shape[1], self.idx, k_shape[3]],
        );
        let v_out = ffi::slice(
            &v_buffer,
            &[0, 0, 0, 0],
            &[v_shape[0], v_shape[1], self.idx, v_shape[3]],
        );
        self.keys = Some(k_buffer);
        self.values = Some(v_buffer);
        (k_out, v_out)
    }

    fn update_concat(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
        new_seq_len: i32,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        if self.keys.is_none() {
            self.offset += new_seq_len;
            self.idx = new_seq_len;
            self.keys = Some(ffi::contiguous(&new_keys, false));
            self.values = Some(ffi::contiguous(&new_values, false));
            return (new_keys, new_values);
        }

        let (base_k, base_v, current_seq_len) = self.visible_fp16_prefix_for_concat();

        let concat_k = concatenate(&base_k, &new_keys, 2);
        let concat_v = concatenate(&base_v, &new_values, 2);

        let total_len = current_seq_len + new_seq_len;
        self.offset += new_seq_len;

        if total_len > self.max_size {
            let start = total_len - self.max_size;
            let k = ffi::slice(
                &concat_k,
                &[0, 0, start, 0],
                &[i32::MAX, i32::MAX, total_len, i32::MAX],
            );
            let v = ffi::slice(
                &concat_v,
                &[0, 0, start, 0],
                &[i32::MAX, i32::MAX, total_len, i32::MAX],
            );
            self.idx = self.max_size;
            self.keys = Some(ffi::contiguous(&k, false));
            self.values = Some(ffi::contiguous(&v, false));
            (k, v)
        } else {
            self.idx = total_len;
            self.keys = Some(ffi::contiguous(&concat_k, false));
            self.values = Some(ffi::contiguous(&concat_v, false));
            (concat_k, concat_v)
        }
    }

    /// Return the visible FP16 K/V prefix in chronological order for a
    /// multi-token append.
    ///
    /// Used by: [`Self::update_concat`] after speculative decode rewinds.
    /// `RotatingKVCache::trim` mirrors upstream and only rewinds
    /// `offset`/`idx`; it intentionally leaves the backing buffer at its
    /// previous capacity. A later multi-token verify append must therefore
    /// concatenate onto the visible prefix (`visible_len()`), not onto stale
    /// tail slots still present in the physical allocation.
    fn visible_fp16_prefix_for_concat(&self) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>, i32) {
        let keys = self
            .keys
            .as_ref()
            .expect("visible_fp16_prefix_for_concat requires initialized keys");
        let values = self
            .values
            .as_ref()
            .expect("visible_fp16_prefix_for_concat requires initialized values");
        let k_shape = ffi::array_shape(keys);
        let v_shape = ffi::array_shape(values);
        let physical_len = k_shape[2];
        let visible_len = self.visible_len().min(physical_len).max(0);
        let logical_start = self.logical_start().min(physical_len).max(0);

        let slice_range = |arr: &MlxArray, shape: &[i32], start: i32, stop: i32| {
            ffi::slice(
                arr,
                &[0, 0, start, 0],
                &[shape[0], shape[1], stop, shape[3]],
            )
        };

        if visible_len == 0 || logical_start == 0 {
            return (
                slice_range(keys, &k_shape, 0, visible_len),
                slice_range(values, &v_shape, 0, visible_len),
                visible_len,
            );
        }

        let tail_len = (physical_len - logical_start).min(visible_len);
        let k_tail = slice_range(keys, &k_shape, logical_start, logical_start + tail_len);
        let v_tail = slice_range(values, &v_shape, logical_start, logical_start + tail_len);
        if tail_len == visible_len {
            return (k_tail, v_tail, visible_len);
        }

        let head_len = visible_len - tail_len;
        let k_head = slice_range(keys, &k_shape, 0, head_len);
        let v_head = slice_range(values, &v_shape, 0, head_len);
        (
            concatenate(&k_tail, &k_head, 2),
            concatenate(&v_tail, &v_head, 2),
            visible_len,
        )
    }

    fn update_in_place(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        if let Some(ref keys) = self.keys {
            let shape = ffi::array_shape(keys);
            let buffer_size = shape[2];
            if buffer_size > self.max_size {
                let start = buffer_size - self.max_size;
                let ks = ffi::array_shape(self.keys.as_ref().unwrap());
                let vs = ffi::array_shape(self.values.as_ref().unwrap());
                self.keys = Some(ffi::contiguous(
                    &ffi::slice(
                        self.keys.as_ref().unwrap(),
                        &[0, 0, start, 0],
                        &[ks[0], ks[1], buffer_size, ks[3]],
                    ),
                    false,
                ));
                self.values = Some(ffi::contiguous(
                    &ffi::slice(
                        self.values.as_ref().unwrap(),
                        &[0, 0, start, 0],
                        &[vs[0], vs[1], buffer_size, vs[3]],
                    ),
                    false,
                ));
                self.idx = self.max_size;
            }
        }

        if self.keys.is_none() {
            let shape = ffi::array_shape(&new_keys);
            let batch = shape[0];
            let heads = shape[1];
            let head_dim = shape[3];
            let value_shape = ffi::array_shape(&new_values);
            let value_head_dim = value_shape[3];

            let k_zeros = ffi::zeros(
                &[batch, heads, self.max_size, head_dim],
                ffi::array_dtype(&new_keys),
            );
            let v_zeros = ffi::zeros(
                &[batch, heads, self.max_size, value_head_dim],
                ffi::array_dtype(&new_values),
            );

            let k = ffi::slice_update(
                &k_zeros,
                &new_keys,
                &[0, 0, 0, 0],
                &[batch, heads, 1, head_dim],
            );
            let v = ffi::slice_update(
                &v_zeros,
                &new_values,
                &[0, 0, 0, 0],
                &[batch, heads, 1, value_head_dim],
            );

            self.offset = 1;
            self.idx = 1;
            self.keys = Some(ffi::contiguous(&k, false));
            self.values = Some(ffi::contiguous(&v, false));

            let k_out = ffi::slice(&k, &[0, 0, 0, 0], &[batch, heads, 1, head_dim]);
            let v_out = ffi::slice(&v, &[0, 0, 0, 0], &[batch, heads, 1, value_head_dim]);
            return (k_out, v_out);
        }

        let mut k_buffer = self.keys.take().unwrap();
        let mut v_buffer = self.values.take().unwrap();

        let shape = ffi::array_shape(&k_buffer);
        let batch = shape[0];
        let heads = shape[1];
        let buffer_size = shape[2];
        let head_dim = shape[3];
        let value_shape = ffi::array_shape(&v_buffer);
        let value_head_dim = value_shape[3];

        if self.idx >= buffer_size && buffer_size < self.max_size {
            let grow_by = self.step.min(self.max_size - buffer_size).max(0);
            if grow_by > 0 {
                let k_zeros = ffi::zeros(
                    &[batch, heads, grow_by, head_dim],
                    ffi::array_dtype(&new_keys),
                );
                let v_zeros = ffi::zeros(
                    &[batch, heads, grow_by, value_head_dim],
                    ffi::array_dtype(&new_values),
                );
                k_buffer = concatenate(&k_buffer, &k_zeros, 2);
                v_buffer = concatenate(&v_buffer, &v_zeros, 2);
            }
        }

        if self.idx >= self.max_size {
            self.idx = 0;
        }

        let pos = self.idx;
        let k_buffer = ffi::slice_update(
            &k_buffer,
            &new_keys,
            &[0, 0, pos, 0],
            &[batch, heads, pos + 1, head_dim],
        );
        let v_buffer = ffi::slice_update(
            &v_buffer,
            &new_values,
            &[0, 0, pos, 0],
            &[batch, heads, pos + 1, value_head_dim],
        );

        self.offset += 1;
        self.idx += 1;

        if self.offset < self.max_size {
            let k_out = ffi::slice(
                &k_buffer,
                &[0, 0, 0, 0],
                &[batch, heads, self.offset, head_dim],
            );
            let v_out = ffi::slice(
                &v_buffer,
                &[0, 0, 0, 0],
                &[batch, heads, self.offset, value_head_dim],
            );
            self.keys = Some(k_buffer);
            self.values = Some(v_buffer);
            (k_out, v_out)
        } else {
            let k_out = ffi::contiguous(&k_buffer, false);
            let v_out = ffi::contiguous(&v_buffer, false);
            self.keys = Some(k_buffer);
            self.values = Some(v_buffer);
            (k_out, v_out)
        }
    }

    // -----------------------------------------------------------------------
    // Turbo4Asym update path (B9)
    //
    // Storage layout when `mode == KVCacheMode::Turbo4Asym`:
    //   - keys     : [B, H, max_size, K_dim]    FP16 — same as Fp16 path
    //   - v_packed : [B, H, max_size, V_dim/2]  UINT8 — nibble-packed indices
    //   - v_norms  : [B, H, max_size, 1]        FP16 — per-token V L2 norms
    //   - values   : None — Turbo4Asym never stores fp16 V
    //
    // Block alignment invariant: `max_size % BLOCK_SIZE == 0` (asserted by
    // `new_with_mode`). Quantization is per-token along `head_dim`, so each
    // ring-buffer slot is independent — wraparound at any 32-aligned `idx`
    // lands on a fresh slot whose packed bytes can be overwritten without
    // disturbing neighbours.
    // -----------------------------------------------------------------------

    fn update_and_fetch_turbo4_asym(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let new_seq_len = {
            let shape = ffi::array_shape(&new_keys);
            shape[2]
        };

        // Cast incoming K/V to FP16 to match the cache contract, then quantize
        // V before any cache mutation.
        let new_keys_f16 = ffi::astype(&new_keys, dtype::FLOAT16);
        let new_values_f16 = ffi::astype(&new_values, dtype::FLOAT16);

        // Lazy-init TurboQuantParams once V head_dim is known.
        if self.turbo_params.is_none() {
            let v_shape = ffi::array_shape(&new_values_f16);
            let v_head_dim = v_shape[3] as u32;
            self.turbo_params = Some(turbo::TurboQuantParams::new(v_head_dim, self.turbo_seed));
        }
        let params = self
            .turbo_params
            .as_ref()
            .expect("turbo_params just initialised");
        let (v_packed_new, v_norms_new, v_rescale_new) =
            turbo::quant::quantize_v_turbo4(&new_values_f16, params);

        if new_seq_len > 1 {
            self.update_turbo4_concat(
                new_keys_f16,
                v_packed_new,
                v_norms_new,
                v_rescale_new,
                new_seq_len,
            )
        } else {
            self.update_turbo4_in_place(new_keys_f16, v_packed_new, v_norms_new, v_rescale_new)
        }
    }

    /// Multi-token (prefill) path for Turbo4Asym. Mirrors
    /// `update_concat` but operates on the V sidecars instead of an FP16 V
    /// buffer. K side is identical to the FP16 path.
    fn update_turbo4_concat(
        &mut self,
        new_keys_f16: UniquePtr<MlxArray>,
        v_packed_new: UniquePtr<MlxArray>,
        v_norms_new: UniquePtr<MlxArray>,
        v_rescale_new: UniquePtr<MlxArray>,
        new_seq_len: i32,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        // First update: just install (Same shape semantics as Fp16 path.)
        if self.keys.is_none() {
            self.offset += new_seq_len;
            self.idx = new_seq_len;
            self.keys = Some(ffi::contiguous(&new_keys_f16, false));
            self.v_packed = Some(ffi::contiguous(&v_packed_new, false));
            self.v_norms = Some(ffi::contiguous(&v_norms_new, false));
            self.v_rescale = Some(ffi::contiguous(&v_rescale_new, false));
            // Build fp16 V output by dequantizing the packed slice we just stored.
            let params = self
                .turbo_params
                .as_ref()
                .expect("turbo_params populated by caller");
            let v_out = turbo::quant::dequantize_v_turbo4(&v_packed_new, &v_norms_new, params);
            return (new_keys_f16, v_out);
        }

        let current_seq_len = {
            let shape = ffi::array_shape(self.keys.as_ref().unwrap());
            shape[2]
        };

        let concat_k = concatenate(self.keys.as_ref().unwrap(), &new_keys_f16, 2);
        let concat_vp = concatenate(self.v_packed.as_ref().unwrap(), &v_packed_new, 2);
        let concat_vn = concatenate(self.v_norms.as_ref().unwrap(), &v_norms_new, 2);
        let concat_vr = match self.v_rescale.as_ref() {
            Some(vr) => concatenate(vr, &v_rescale_new, 2),
            None => ffi::contiguous(&v_rescale_new, false),
        };

        let total_len = current_seq_len + new_seq_len;
        self.offset += new_seq_len;

        // Block alignment is preserved: per-token quantization means slicing
        // a packed buffer along axis=2 always yields valid packed tokens. No
        // partial-block re-quantization is needed regardless of `start`.
        if total_len > self.max_size {
            let start = total_len - self.max_size;
            let k = ffi::slice(
                &concat_k,
                &[0, 0, start, 0],
                &[i32::MAX, i32::MAX, total_len, i32::MAX],
            );
            let vp = ffi::slice(
                &concat_vp,
                &[0, 0, start, 0],
                &[i32::MAX, i32::MAX, total_len, i32::MAX],
            );
            let vn = ffi::slice(
                &concat_vn,
                &[0, 0, start, 0],
                &[i32::MAX, i32::MAX, total_len, i32::MAX],
            );
            let vr = ffi::slice(
                &concat_vr,
                &[0, 0, start, 0],
                &[i32::MAX, i32::MAX, total_len, i32::MAX],
            );
            self.idx = self.max_size;
            self.keys = Some(ffi::contiguous(&k, false));
            self.v_packed = Some(ffi::contiguous(&vp, false));
            self.v_norms = Some(ffi::contiguous(&vn, false));
            self.v_rescale = Some(ffi::contiguous(&vr, false));
            let params = self
                .turbo_params
                .as_ref()
                .expect("turbo_params populated by caller");
            let v_out = turbo::quant::dequantize_v_turbo4(&vp, &vn, params);
            (k, v_out)
        } else {
            self.idx = total_len;
            self.keys = Some(ffi::contiguous(&concat_k, false));
            self.v_packed = Some(ffi::contiguous(&concat_vp, false));
            self.v_norms = Some(ffi::contiguous(&concat_vn, false));
            self.v_rescale = Some(ffi::contiguous(&concat_vr, false));
            let params = self
                .turbo_params
                .as_ref()
                .expect("turbo_params populated by caller");
            let v_out = turbo::quant::dequantize_v_turbo4(&concat_vp, &concat_vn, params);
            (concat_k, v_out)
        }
    }

    /// Single-token (decode) path for Turbo4Asym. Mirrors `update_in_place`,
    /// writing the new packed token + norm into the ring buffer at `self.idx`.
    /// Wraparound at `idx == max_size` resets `idx` to 0 — because each token
    /// is independently quantized, the overwrite is byte-correct without any
    /// re-quantize work.
    fn update_turbo4_in_place(
        &mut self,
        new_keys_f16: UniquePtr<MlxArray>,
        v_packed_new: UniquePtr<MlxArray>,
        v_norms_new: UniquePtr<MlxArray>,
        v_rescale_new: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        // First-call init mirrors the FP16 path: pre-allocate the ring buffer.
        if self.keys.is_none() {
            let shape = ffi::array_shape(&new_keys_f16);
            let batch = shape[0];
            let heads = shape[1];
            let head_dim = shape[3];
            let vp_shape = ffi::array_shape(&v_packed_new);
            let v_packed_dim = vp_shape[3];

            let k_zeros = ffi::zeros(
                &[batch, heads, self.max_size, head_dim],
                ffi::array_dtype(&new_keys_f16),
            );
            let vp_zeros = ffi::zeros(&[batch, heads, self.max_size, v_packed_dim], dtype::UINT8);
            let vn_zeros = ffi::zeros(&[batch, heads, self.max_size, 1], dtype::FLOAT16);
            let vr_zeros = ffi::zeros(&[batch, heads, self.max_size, 1], dtype::FLOAT16);

            let k = ffi::slice_update(
                &k_zeros,
                &new_keys_f16,
                &[0, 0, 0, 0],
                &[batch, heads, 1, head_dim],
            );
            let vp = ffi::slice_update(
                &vp_zeros,
                &v_packed_new,
                &[0, 0, 0, 0],
                &[batch, heads, 1, v_packed_dim],
            );
            let vn = ffi::slice_update(
                &vn_zeros,
                &v_norms_new,
                &[0, 0, 0, 0],
                &[batch, heads, 1, 1],
            );
            let vr = ffi::slice_update(
                &vr_zeros,
                &v_rescale_new,
                &[0, 0, 0, 0],
                &[batch, heads, 1, 1],
            );

            self.offset = 1;
            self.idx = 1;
            self.keys = Some(ffi::contiguous(&k, false));
            self.v_packed = Some(ffi::contiguous(&vp, false));
            self.v_norms = Some(ffi::contiguous(&vn, false));
            self.v_rescale = Some(ffi::contiguous(&vr, false));

            // Build the FP16 V view from the single packed token.
            let params = self
                .turbo_params
                .as_ref()
                .expect("turbo_params populated by caller");
            let vp_slice = ffi::slice(&vp, &[0, 0, 0, 0], &[batch, heads, 1, v_packed_dim]);
            let vn_slice = ffi::slice(&vn, &[0, 0, 0, 0], &[batch, heads, 1, 1]);
            let v_out = turbo::quant::dequantize_v_turbo4(&vp_slice, &vn_slice, params);
            let k_out = ffi::slice(&k, &[0, 0, 0, 0], &[batch, heads, 1, head_dim]);
            return (k_out, v_out);
        }

        let mut k_buffer = self.keys.take().unwrap();
        let mut vp_buffer = self.v_packed.take().unwrap();
        let mut vn_buffer = self.v_norms.take().unwrap();
        let mut vr_buffer = self
            .v_rescale
            .take()
            .expect("v_rescale lockstep with v_norms in rotating cache");

        let shape = ffi::array_shape(&k_buffer);
        let batch = shape[0];
        let heads = shape[1];
        let buffer_size = shape[2];
        let head_dim = shape[3];
        let vp_shape = ffi::array_shape(&vp_buffer);
        let v_packed_dim = vp_shape[3];

        // Lazy buffer growth (mirrors fp16 path). For Turbo4Asym this only
        // fires before the first wraparound; once `buffer_size == max_size`
        // we stay there forever.
        if self.idx >= buffer_size && buffer_size < self.max_size {
            let grow_by = self.step.min(self.max_size - buffer_size).max(0);
            if grow_by > 0 {
                let k_zeros = ffi::zeros(
                    &[batch, heads, grow_by, head_dim],
                    ffi::array_dtype(&new_keys_f16),
                );
                let vp_zeros = ffi::zeros(&[batch, heads, grow_by, v_packed_dim], dtype::UINT8);
                let vn_zeros = ffi::zeros(&[batch, heads, grow_by, 1], dtype::FLOAT16);
                let vr_zeros = ffi::zeros(&[batch, heads, grow_by, 1], dtype::FLOAT16);
                k_buffer = concatenate(&k_buffer, &k_zeros, 2);
                vp_buffer = concatenate(&vp_buffer, &vp_zeros, 2);
                vn_buffer = concatenate(&vn_buffer, &vn_zeros, 2);
                vr_buffer = concatenate(&vr_buffer, &vr_zeros, 2);
            }
        }

        // Wraparound. Per-token independence makes this a simple in-place
        // overwrite; no block-edge re-quantization is required.
        if self.idx >= self.max_size {
            self.idx = 0;
        }

        let pos = self.idx;
        let k_buffer = ffi::slice_update(
            &k_buffer,
            &new_keys_f16,
            &[0, 0, pos, 0],
            &[batch, heads, pos + 1, head_dim],
        );
        let vp_buffer = ffi::slice_update(
            &vp_buffer,
            &v_packed_new,
            &[0, 0, pos, 0],
            &[batch, heads, pos + 1, v_packed_dim],
        );
        let vn_buffer = ffi::slice_update(
            &vn_buffer,
            &v_norms_new,
            &[0, 0, pos, 0],
            &[batch, heads, pos + 1, 1],
        );
        let vr_buffer = ffi::slice_update(
            &vr_buffer,
            &v_rescale_new,
            &[0, 0, pos, 0],
            &[batch, heads, pos + 1, 1],
        );

        self.offset += 1;
        self.idx += 1;

        // Build the FP16 V view from the visible window and return.
        let params = self
            .turbo_params
            .as_ref()
            .expect("turbo_params populated by caller");
        let result = if self.offset < self.max_size {
            let visible = self.offset;
            let k_out = ffi::slice(&k_buffer, &[0, 0, 0, 0], &[batch, heads, visible, head_dim]);
            let vp_view = ffi::slice(
                &vp_buffer,
                &[0, 0, 0, 0],
                &[batch, heads, visible, v_packed_dim],
            );
            let vn_view = ffi::slice(&vn_buffer, &[0, 0, 0, 0], &[batch, heads, visible, 1]);
            let v_out = turbo::quant::dequantize_v_turbo4(&vp_view, &vn_view, params);
            (k_out, v_out)
        } else {
            // Ring is full — return the contiguous full buffer view (matches
            // fp16 semantics: callers see the complete sliding window).
            let k_out = ffi::contiguous(&k_buffer, false);
            let v_out = turbo::quant::dequantize_v_turbo4(&vp_buffer, &vn_buffer, params);
            (k_out, v_out)
        };
        self.keys = Some(k_buffer);
        self.v_packed = Some(vp_buffer);
        self.v_norms = Some(vn_buffer);
        self.v_rescale = Some(vr_buffer);
        result
    }

    /// Get the current offset
    pub fn get_offset(&self) -> i32 {
        self.offset
    }

    /// Visible length exposed to decode attention.
    pub fn visible_len(&self) -> i32 {
        if self.buffer_size > 0 {
            return self.idx.max(0);
        }
        self.seq_len().min(self.offset).max(0)
    }

    /// Physical start index of the logical oldest token in the ring buffer.
    ///
    /// Before the cache wraps, the visible region starts at index 0. After
    /// wrapping, `idx` tracks the next write position, which is also the
    /// oldest logical token in the ring.
    pub fn logical_start(&self) -> i32 {
        if self.buffer_size > 0 {
            return 0;
        }
        let visible_len = self.visible_len();
        if visible_len == 0 || self.offset <= visible_len {
            0
        } else {
            self.idx.rem_euclid(visible_len)
        }
    }

    /// Internal write position in the ring buffer.
    ///
    /// Mirrors Python `mlx_lm.models.cache.RotatingKVCache._idx`. Used by the
    /// Gemma 4 MTP `rollback_speculative_cache` hook to compute
    /// `kv_len` for per-row tail zeroing of partial-accept verify passes.
    /// Equal to `self.offset` while the cache has not yet wrapped, and
    /// otherwise tracks the next physical write slot in the rotating buffer.
    pub fn buffer_write_idx(&self) -> i32 {
        self.idx
    }

    /// Trim the last `n` entries from the rotating cache by rewinding the
    /// monotonic offset and the buffer write index.
    ///
    /// Mirrors Python `mlx_lm.models.cache.RotatingKVCache.trim`
    /// (`n = min(self.offset, n); self.offset -= n; self._idx -= n`). The
    /// stored buffer is intentionally not re-sliced: subsequent
    /// `update_and_fetch` calls overwrite the trimmed window in place at the
    /// rewound `idx`, exactly matching the rotating-cache semantics in
    /// upstream mlx-lm. The fetched `[..., :self.offset, :]` view returned
    /// by the next update reflects only the live region.
    ///
    /// The function is bit-identical to Python's helper for non-wrapped
    /// caches (the typical Gemma 4 MTP rollback case, where `n` is
    /// `block_size - accepted - 1`, far below the sliding window's
    /// `max_size`).
    ///
    /// Used by: Gemma 4 MTP `rollback_speculative_cache`.
    pub fn trim(&mut self, n: i32) -> i32 {
        let n = if self.buffer_size > 0 {
            n.min(self.idx).min(self.offset)
        } else {
            n.min(self.offset)
        };
        if n <= 0 {
            return 0;
        }
        self.offset -= n;
        self.idx -= n;
        n
    }
}

impl Default for RotatingKVCache {
    fn default() -> Self {
        Self::new(4096)
    }
}

/// Chunked KV Cache for Llama 4's iGQA (Interleaved GQA) attention.
///
/// Maintains a sliding window cache that trims from the front when exceeding
/// `chunk_size`, while still tracking the global start position for mask logic.
pub struct ChunkedKVCache {
    pub keys: Option<UniquePtr<MlxArray>>,
    pub values: Option<UniquePtr<MlxArray>>,
    pub chunk_size: i32,
    pub offset: i32,
    pub start_position: i32,
    step: i32,
}

impl ChunkedKVCache {
    /// Create a new chunked KV cache with specified chunk size
    pub fn new(chunk_size: i32) -> Self {
        Self {
            keys: None,
            values: None,
            chunk_size,
            offset: 0,
            start_position: 0,
            step: 256,
        }
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.keys.is_none()
    }

    /// Get the global offset (total tokens processed)
    pub fn get_offset(&self) -> i32 {
        self.offset
    }

    /// Get the start position (where visible window begins)
    pub fn get_start_position(&self) -> i32 {
        self.start_position
    }

    /// Trim the front of the cache if it exceeds chunk_size.
    ///
    /// This should be called before processing each layer.
    pub fn maybe_trim_front(&mut self) {
        if let Some(ref keys) = self.keys {
            let shape = ffi::array_shape(keys);
            let seq_len = (self.offset - self.start_position).min(shape[2]);

            if seq_len > self.chunk_size {
                let trim_amount = seq_len - self.chunk_size;
                self.start_position += trim_amount;

                let k_shape = ffi::array_shape(self.keys.as_ref().unwrap());
                let v_shape = ffi::array_shape(self.values.as_ref().unwrap());

                self.keys = Some(ffi::slice(
                    self.keys.as_ref().unwrap(),
                    &[0, 0, trim_amount, 0],
                    &[k_shape[0], k_shape[1], seq_len, k_shape[3]],
                ));
                self.values = Some(ffi::slice(
                    self.values.as_ref().unwrap(),
                    &[0, 0, trim_amount, 0],
                    &[v_shape[0], v_shape[1], seq_len, v_shape[3]],
                ));
            }
        }
    }

    /// Update cache with new key/value and return the visible portion
    pub fn update_and_fetch(
        &mut self,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
    ) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
        let new_shape = ffi::array_shape(&new_keys);
        let new_seq_len = new_shape[2];
        let prev = self.offset - self.start_position;

        if self.keys.is_none() || (prev + new_seq_len) > self.get_buffer_size() {
            let b = new_shape[0];
            let n_kv_heads = new_shape[1];
            let k_head_dim = new_shape[3];
            let v_shape = ffi::array_shape(&new_values);
            let v_head_dim = v_shape[3];

            let n_steps = (self.step + new_seq_len - 1) / self.step;
            let new_buffer_size = n_steps * self.step;

            let new_k = ffi::zeros(
                &[b, n_kv_heads, new_buffer_size, k_head_dim],
                ffi::array_dtype(&new_keys),
            );
            let new_v = ffi::zeros(
                &[b, n_kv_heads, new_buffer_size, v_head_dim],
                ffi::array_dtype(&new_values),
            );

            if self.keys.is_some() {
                if prev % self.step != 0 {
                    let k_shape = ffi::array_shape(self.keys.as_ref().unwrap());
                    let v_shape = ffi::array_shape(self.values.as_ref().unwrap());
                    self.keys = Some(ffi::slice(
                        self.keys.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[k_shape[0], k_shape[1], prev, k_shape[3]],
                    ));
                    self.values = Some(ffi::slice(
                        self.values.as_ref().unwrap(),
                        &[0, 0, 0, 0],
                        &[v_shape[0], v_shape[1], prev, v_shape[3]],
                    ));
                }
                self.keys = Some(concatenate(self.keys.as_ref().unwrap(), &new_k, 2));
                self.values = Some(concatenate(self.values.as_ref().unwrap(), &new_v, 2));
            } else {
                self.keys = Some(new_k);
                self.values = Some(new_v);
            }
        }

        self.offset += new_seq_len;
        let end = self.offset - self.start_position;

        let k_shape = ffi::array_shape(self.keys.as_ref().unwrap());
        let v_shape = ffi::array_shape(self.values.as_ref().unwrap());

        if prev > 0 {
            let k_before = ffi::slice(
                self.keys.as_ref().unwrap(),
                &[0, 0, 0, 0],
                &[k_shape[0], k_shape[1], prev, k_shape[3]],
            );
            let v_before = ffi::slice(
                self.values.as_ref().unwrap(),
                &[0, 0, 0, 0],
                &[v_shape[0], v_shape[1], prev, v_shape[3]],
            );

            if end < k_shape[2] {
                let k_after = ffi::slice(
                    self.keys.as_ref().unwrap(),
                    &[0, 0, end, 0],
                    &[k_shape[0], k_shape[1], k_shape[2], k_shape[3]],
                );
                let v_after = ffi::slice(
                    self.values.as_ref().unwrap(),
                    &[0, 0, end, 0],
                    &[v_shape[0], v_shape[1], v_shape[2], v_shape[3]],
                );
                self.keys = Some(concatenate(
                    &concatenate(&k_before, &new_keys, 2),
                    &k_after,
                    2,
                ));
                self.values = Some(concatenate(
                    &concatenate(&v_before, &new_values, 2),
                    &v_after,
                    2,
                ));
            } else {
                self.keys = Some(concatenate(&k_before, &new_keys, 2));
                self.values = Some(concatenate(&v_before, &new_values, 2));
            }
        } else if end < k_shape[2] {
            let k_after = ffi::slice(
                self.keys.as_ref().unwrap(),
                &[0, 0, end, 0],
                &[k_shape[0], k_shape[1], k_shape[2], k_shape[3]],
            );
            let v_after = ffi::slice(
                self.values.as_ref().unwrap(),
                &[0, 0, end, 0],
                &[v_shape[0], v_shape[1], v_shape[2], v_shape[3]],
            );
            self.keys = Some(concatenate(&new_keys, &k_after, 2));
            self.values = Some(concatenate(&new_values, &v_after, 2));
        } else {
            self.keys = Some(ffi::contiguous(&new_keys, false));
            self.values = Some(ffi::contiguous(&new_values, false));
        }

        (
            ffi::slice(
                self.keys.as_ref().unwrap(),
                &[0, 0, 0, 0],
                &[k_shape[0], k_shape[1], end, k_shape[3]],
            ),
            ffi::slice(
                self.values.as_ref().unwrap(),
                &[0, 0, 0, 0],
                &[v_shape[0], v_shape[1], end, v_shape[3]],
            ),
        )
    }

    fn get_buffer_size(&self) -> i32 {
        if let Some(ref keys) = self.keys {
            let shape = ffi::array_shape(keys);
            shape[2]
        } else {
            0
        }
    }
}

/// Structure-of-arrays metadata for batched decode position handling.
///
/// This keeps per-sequence offsets, query lengths, visible KV lengths, and
/// window sizes in a kernel-friendly representation instead of relying on
/// scalar `cache.offset` assumptions inside batched model code.
///
/// # Invariant
///
/// All four `Vec<i32>` fields **must** have the same length, equal to the
/// active batch size. This holds even when the batch is uniform (e.g. every
/// sequence has the same offset) — there is no scalar fast-path that
/// collapses the per-row state to a single value. This invariant exists for
/// two reasons:
///
/// 1. Downstream kernel and graph code (paged decode, fused RoPE, padded
///    SDPA) indexes per batch row directly. A scalar fallback would break
///    those callers silently.
/// 2. It mirrors the upstream `mlx-vlm` PR #1110 fix pattern (in this repo) for `BatchTurboQuantKVCache`. There, an "init-time scalar
///    fast-path when all `left_padding` entries are equal" caused
///    `extend()` and `filter()` to crash later when the batch composition
///    changed mid-decode and the still-scalar `offset` was passed to
///    `mx.concatenate`. Rust's static typing prevents the literal Python
///    crash, but the *semantic* hazard — code that assumes uniform batches
///    and silently degrades when composition changes — is what the
///    `extend` / `filter` / `assert_consistent` helpers below guard
///    against.
///
/// Use [`assert_consistent`](Self::assert_consistent) at debug time when
/// constructing instances directly via the public fields to catch any
/// future regression that re-introduces a uniformity-collapsing
/// optimization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchedAttentionMetadata {
    pub rope_offsets: Vec<i32>,
    pub query_lens: Vec<i32>,
    pub kv_lens: Vec<i32>,
    pub window_sizes: Vec<i32>,
}

impl BatchedAttentionMetadata {
    /// Build heterogeneous per-sequence metadata from standard KV caches.
    ///
    /// Always emits per-row `Vec<i32>` arrays of length `caches.len()`,
    /// even when every cache has the same offset. See the struct-level docs
    /// for the rationale (upstream `mlx-vlm` PR #1110).
    pub fn from_kv_caches(
        caches: &[&mut KVCache],
        query_lens: &[i32],
        window_sizes: &[i32],
    ) -> Result<Self, String> {
        let batch = caches.len();
        if query_lens.len() != batch {
            return Err(format!(
                "expected {} query lengths for batched attention metadata, got {}",
                batch,
                query_lens.len()
            ));
        }
        if window_sizes.len() != batch {
            return Err(format!(
                "expected {} window sizes for batched attention metadata, got {}",
                batch,
                window_sizes.len()
            ));
        }

        let mut rope_offsets = Vec::with_capacity(batch);
        let mut kv_lens = Vec::with_capacity(batch);
        for (cache, &query_len) in caches.iter().zip(query_lens.iter()) {
            if query_len < 0 {
                return Err(format!(
                    "query length must be non-negative for batched attention metadata, got {query_len}"
                ));
            }
            // `rope_offsets` must stay on the monotonic position axis so the
            // newly-projected Q/K rows receive the same RoPE positions they
            // would have seen before any `--max-kv-size` trim. `kv_lens`,
            // however, describes the **visible** KV window passed to the
            // attention kernel. After `KVCache::trim_front` advances
            // `live_start`, the visible prefix is `live_len()` rather than the
            // monotonic offset. Keeping `kv_lens = offset + query_len` after a
            // trim would make paged/dense-compatible batched attention believe
            // the cache arrays are much longer than they actually are.
            let offset = cache.offset;
            let live_len = cache.live_len();
            rope_offsets.push(offset);
            kv_lens.push(live_len + query_len);
        }

        Ok(Self {
            rope_offsets,
            query_lens: query_lens.to_vec(),
            kv_lens,
            window_sizes: window_sizes.to_vec(),
        })
    }

    /// Build uniform metadata for full-attention batched decode/prefill paths.
    ///
    /// Despite the "uniform" name, this still produces full per-row
    /// `Vec<i32>` arrays of length `caches.len()`. The "uniform" refers
    /// only to the caller-supplied `query_len` and `window_size` being the
    /// same for every batch row; per-row offsets are still read from each
    /// cache. **Do not** add a scalar fast-path here — see the
    /// [`BatchedAttentionMetadata`] struct-level docs for
    /// why.
    pub fn uniform_kv_caches(
        caches: &[&mut KVCache],
        query_len: i32,
        window_size: i32,
    ) -> Result<Self, String> {
        let query_lens = vec![query_len; caches.len()];
        let window_sizes = vec![window_size; caches.len()];
        Self::from_kv_caches(caches, &query_lens, &window_sizes)
    }

    pub fn len(&self) -> usize {
        self.rope_offsets.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rope_offsets.is_empty()
    }

    /// Verify the invariant that all per-row vectors have the same length.
    ///
    /// Returns `Ok(())` when the four `Vec<i32>` fields agree on a single
    /// batch size, and an error message identifying the mismatched lengths
    /// otherwise. The struct fields are public for kernel-friendly slice
    /// access, so any code path that constructs a metadata instance
    /// directly (rather than going through [`Self::from_kv_caches`] or
    /// [`Self::uniform_kv_caches`]) should call this method in debug
    /// builds.
    ///
    /// This mirrors the upstream `mlx-vlm` PR #1110 pattern of
    /// keeping per-row offset state shape-stable across batch grow / shrink
    /// transitions. See the struct-level docs for the full rationale.
    ///
    /// Used by: [`Self::extend`], [`Self::filter_in_place`], and any model
    /// code that constructs metadata via direct field assignment.
    pub fn assert_consistent(&self) -> Result<(), String> {
        let n = self.rope_offsets.len();
        if self.query_lens.len() != n {
            return Err(format!(
                "BatchedAttentionMetadata: query_lens length {} differs from rope_offsets length {n}",
                self.query_lens.len()
            ));
        }
        if self.kv_lens.len() != n {
            return Err(format!(
                "BatchedAttentionMetadata: kv_lens length {} differs from rope_offsets length {n}",
                self.kv_lens.len()
            ));
        }
        if self.window_sizes.len() != n {
            return Err(format!(
                "BatchedAttentionMetadata: window_sizes length {} differs from rope_offsets length {n}",
                self.window_sizes.len()
            ));
        }
        Ok(())
    }

    /// Extend the metadata with another batch's per-row state, preserving
    /// the per-row vector shape.
    ///
    /// This is the mlxcel analogue of upstream `mlx-vlm` PR #1110's
    /// `BatchTurboQuantKVCache.extend()`: when the batch
    /// scheduler grows the active batch by adopting one or more newly
    /// prefilled sequences, the resulting metadata must remain a per-row
    /// `Vec<i32>` even when both halves were uniform. The Python upstream
    /// crash mode — `mx.concatenate([scalar_int, mx_array])` — is
    /// statically impossible in Rust because the field type is already
    /// `Vec<i32>`, but this method centralises the merge so callers do not
    /// reach for ad-hoc `extend_from_slice` calls that could miss one of
    /// the four parallel vectors.
    ///
    /// Used by: future scheduler paths that compose two batches' attention
    /// metadata before a fused decode call. The current decode path
    /// rebuilds metadata from scratch on every step via
    /// [`Self::from_kv_caches`], which already preserves the invariant; this
    /// helper is provided as a stable seam for upcoming continuous-batching
    /// fusion work and as a regression target.
    pub fn extend(&mut self, other: &Self) -> Result<(), String> {
        self.assert_consistent()?;
        other.assert_consistent()?;
        self.rope_offsets.extend_from_slice(&other.rope_offsets);
        self.query_lens.extend_from_slice(&other.query_lens);
        self.kv_lens.extend_from_slice(&other.kv_lens);
        self.window_sizes.extend_from_slice(&other.window_sizes);
        Ok(())
    }

    /// Drop rows not in `indices`, preserving the per-row vector shape.
    ///
    /// The mlxcel analogue of upstream `mlx-vlm` PR #1110's
    /// `BatchTurboQuantKVCache.filter()`. Indexing is
    /// position-based: `indices[k] = i` means "row `i` of the current
    /// batch becomes row `k` of the filtered batch".
    ///
    /// Errors if any index is out of range or duplicated, or if the
    /// per-row vectors are not already consistent (the latter would mean
    /// somebody bypassed the public constructors and produced an
    /// inconsistent state).
    ///
    /// Used by: regression coverage for batch-shrink transitions
    /// not yet wired into the live decode path because the
    /// scheduler currently rebuilds metadata each step rather than
    /// filtering in place. Provided as a stable seam for upcoming
    /// continuous-batching fusion work.
    pub fn filter_in_place(&mut self, indices: &[usize]) -> Result<(), String> {
        self.assert_consistent()?;
        let n = self.rope_offsets.len();
        // Reject out-of-range and duplicate indices up front so the merge
        // cannot leave the four parallel vectors in inconsistent partial
        // states on error.
        let mut seen = vec![false; n];
        for &i in indices {
            if i >= n {
                return Err(format!(
                    "BatchedAttentionMetadata::filter_in_place: index {i} out of range for batch size {n}"
                ));
            }
            if seen[i] {
                return Err(format!(
                    "BatchedAttentionMetadata::filter_in_place: duplicate index {i}"
                ));
            }
            seen[i] = true;
        }

        let new_rope_offsets: Vec<i32> = indices.iter().map(|&i| self.rope_offsets[i]).collect();
        let new_query_lens: Vec<i32> = indices.iter().map(|&i| self.query_lens[i]).collect();
        let new_kv_lens: Vec<i32> = indices.iter().map(|&i| self.kv_lens[i]).collect();
        let new_window_sizes: Vec<i32> = indices.iter().map(|&i| self.window_sizes[i]).collect();

        self.rope_offsets = new_rope_offsets;
        self.query_lens = new_query_lens;
        self.kv_lens = new_kv_lens;
        self.window_sizes = new_window_sizes;
        Ok(())
    }
}

/// Decode-only paged attention metadata derived from per-sequence KV lengths.
///
/// The current dense-compat kernel treats `block_tables` as logical block
/// indices (`0..num_blocks`) for each sequence. A future physical paged-KV
/// backend can reuse the same shape while replacing these entries with actual
/// physical block identifiers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedDecodeMetadata {
    pub block_size: i32,
    pub kv_lens: Vec<i32>,
    pub block_table_offsets: Vec<i32>,
    pub block_tables: Vec<i32>,
}

/// Decode-only paged attention metadata for ring-buffer-backed rotating caches.
///
/// `logical_starts[i]` identifies the physical buffer index of the oldest
/// visible token for sequence `i`, allowing native paged decode kernels to
/// gather wrapped sliding-window buffers without first materializing a dense
/// linearized copy in Rust.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RotatingPagedDecodeMetadata {
    pub block_size: i32,
    pub kv_lens: Vec<i32>,
    pub logical_starts: Vec<i32>,
}

impl RotatingPagedDecodeMetadata {
    pub fn from_parts(
        kv_lens: &[i32],
        logical_starts: &[i32],
        block_size: i32,
    ) -> Result<Self, String> {
        if block_size <= 0 {
            return Err(format!(
                "rotating paged decode metadata requires block_size > 0, got {block_size}"
            ));
        }
        if kv_lens.len() != logical_starts.len() {
            return Err(format!(
                "rotating paged decode metadata length mismatch: {} kv_lens vs {} logical_starts",
                kv_lens.len(),
                logical_starts.len()
            ));
        }
        for (&kv_len, &logical_start) in kv_lens.iter().zip(logical_starts.iter()) {
            if kv_len < 0 {
                return Err(format!(
                    "rotating paged decode metadata requires non-negative kv lengths, got {kv_len}"
                ));
            }
            if logical_start < 0 {
                return Err(format!(
                    "rotating paged decode metadata requires non-negative logical starts, got {logical_start}"
                ));
            }
            if kv_len > 0 && logical_start >= kv_len {
                return Err(format!(
                    "rotating paged decode metadata requires logical_start < kv_len when kv_len > 0, got logical_start={logical_start}, kv_len={kv_len}"
                ));
            }
        }

        Ok(Self {
            block_size,
            kv_lens: kv_lens.to_vec(),
            logical_starts: logical_starts.to_vec(),
        })
    }

    pub fn len(&self) -> usize {
        self.kv_lens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kv_lens.is_empty()
    }
}

impl PagedDecodeMetadata {
    pub fn from_attention_metadata(
        metadata: &BatchedAttentionMetadata,
        block_size: i32,
    ) -> Result<Self, String> {
        Self::from_visible_lengths(&metadata.kv_lens, block_size)
    }

    pub fn from_visible_lengths(kv_lens: &[i32], block_size: i32) -> Result<Self, String> {
        if block_size <= 0 {
            return Err(format!(
                "paged decode metadata requires block_size > 0, got {block_size}"
            ));
        }

        let mut block_table_offsets = Vec::with_capacity(kv_lens.len() + 1);
        let mut block_tables = Vec::new();
        block_table_offsets.push(0);

        for &kv_len in kv_lens {
            if kv_len < 0 {
                return Err(format!(
                    "paged decode metadata requires non-negative kv lengths, got {kv_len}"
                ));
            }

            let block_count = if kv_len == 0 {
                0
            } else {
                (kv_len + block_size - 1) / block_size
            };
            for logical_block in 0..block_count {
                block_tables.push(logical_block);
            }
            block_table_offsets.push(block_tables.len() as i32);
        }

        Ok(Self {
            block_size,
            kv_lens: kv_lens.to_vec(),
            block_table_offsets,
            block_tables,
        })
    }

    pub fn len(&self) -> usize {
        self.kv_lens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.kv_lens.is_empty()
    }
}

impl Default for ChunkedKVCache {
    fn default() -> Self {
        Self::new(8192)
    }
}

// --- Per-sequence cache isolation for continuous batching ---

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Unique identifier for a sequence in the batch.
///
/// Each active generation sequence receives a unique monotonically increasing
/// ID from the owning `CachePool`. The inner `u64` never wraps within any
/// reasonable server lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SequenceId(u64);

impl SequenceId {
    /// Construct a `SequenceId` from a raw `u64` value.
    ///
    /// In production code, IDs are assigned by `CachePool::allocate`. This
    /// constructor is provided for tests, builders, and deserialization.
    pub fn from_raw(id: u64) -> Self {
        Self(id)
    }

    /// Return the raw numeric identifier.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for SequenceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "seq-{}", self.0)
    }
}

/// Logical owner/backend for one sequence's runtime state.
///
/// Phase 0 keeps all backends represented through the existing `Vec<KVCache>`
/// surface so behavior stays unchanged while the control plane gains an
/// explicit seam for future paged and model-owned state backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStateBackend {
    /// Standard per-layer external KV caches stored directly in the sequence.
    DenseKvCache,
    /// Paged block tables plus logical sequence metadata.
    PagedKvCache,
    /// Model-owned/internal state. The exposed `Vec<KVCache>` acts only as a
    /// compatibility placeholder for existing generation and scheduler paths.
    ModelOwned,
}

/// Backend/layout descriptor for allocating one sequence's runtime state.
///
/// Used by: `LanguageModel::sequence_state_layout()`, `CachePool::allocate()`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceStateLayout {
    pub backend: SequenceStateBackend,
    pub num_layers: usize,
    pub paged_layout: Option<PagedKvLayout>,
}

impl SequenceStateLayout {
    /// Allocate per-layer dense external KV caches for this sequence.
    pub const fn dense_kv_cache(num_layers: usize) -> Self {
        Self {
            backend: SequenceStateBackend::DenseKvCache,
            num_layers,
            paged_layout: None,
        }
    }

    /// Allocate per-layer paged KV state for this sequence.
    pub fn paged_kv_cache(paged_layout: PagedKvLayout) -> Self {
        Self {
            backend: SequenceStateBackend::PagedKvCache,
            num_layers: paged_layout.num_layers,
            paged_layout: Some(paged_layout),
        }
    }

    /// Allocate model-owned/internal sequence state with placeholder KV slots.
    pub const fn model_owned(num_layers: usize) -> Self {
        Self {
            backend: SequenceStateBackend::ModelOwned,
            num_layers,
            paged_layout: None,
        }
    }
}

/// One sequence's full set of layer caches.
///
/// Created by `CachePool::allocate` and tied to a single generation request.
/// The caller owns mutable access while the sequence is active and must call
/// `CachePool::release` when generation finishes.
pub struct SequenceCacheSet {
    /// Logical owner/backend for this sequence's state.
    pub backend: SequenceStateBackend,
    /// Per-layer KV caches (one entry per model layer).
    pub caches: Vec<KVCache>,
    /// Paged block-table state when `backend == PagedKvCache`.
    ///
    /// Wrapped in `Rc<RefCell<…>>` so a pool-backed sequence can share the
    /// SAME state handle with every per-layer [`KVCache`] (each cache's
    /// [`PagedBacking`] holds a cheap `Rc` clone). The pool-backed caches and
    /// this set therefore observe one authoritative block table. Dense and
    /// model-owned sequences leave this `None`. The interior mutability is only
    /// ever exercised at request boundaries (allocate / sync / release) and
    /// inside `update_and_fetch` during forward — never both at once — so the
    /// borrows stay non-overlapping (see the borrow-discipline notes on the
    /// `CachePool` paged methods).
    pub paged: Option<Rc<RefCell<PagedSequenceState>>>,
    /// Unique identifier assigned by the pool.
    pub seq_id: SequenceId,
    /// Number of prompt tokens originally prefilled.
    pub prompt_len: usize,
    /// Current generation position (incremented during decode).
    pub current_offset: i32,
    /// Wall-clock time when this cache set was allocated.
    pub created_at: Instant,
    paged_layout: Option<PagedKvLayout>,
}

impl SequenceCacheSet {
    fn with_backend(
        seq_id: SequenceId,
        backend: SequenceStateBackend,
        caches: Vec<KVCache>,
        paged: Option<Rc<RefCell<PagedSequenceState>>>,
        paged_layout: Option<PagedKvLayout>,
    ) -> Self {
        Self {
            backend,
            caches,
            paged,
            seq_id,
            prompt_len: 0,
            current_offset: 0,
            created_at: Instant::now(),
            paged_layout,
        }
    }

    /// Allocate a sequence state backed by standard external KV caches.
    pub fn dense_external(seq_id: SequenceId, caches: Vec<KVCache>) -> Self {
        Self::with_backend(
            seq_id,
            SequenceStateBackend::DenseKvCache,
            caches,
            None,
            None,
        )
    }

    /// Allocate a sequence state backed by paged block tables.
    pub fn paged(seq_id: SequenceId, paged_layout: PagedKvLayout) -> Self {
        let paged = PagedSequenceState::new(&paged_layout);
        Self::with_backend(
            seq_id,
            SequenceStateBackend::PagedKvCache,
            Vec::new(),
            Some(Rc::new(RefCell::new(paged))),
            Some(paged_layout),
        )
    }

    /// Allocate a sequence state for model-owned/internal caches.
    pub fn model_owned(seq_id: SequenceId) -> Self {
        Self::with_backend(
            seq_id,
            SequenceStateBackend::ModelOwned,
            Vec::new(),
            None,
            None,
        )
    }

    /// Total memory footprint of all layer caches in bytes.
    pub fn nbytes(&self) -> usize {
        let dense_bytes: usize = self.caches.iter().map(|c| c.nbytes()).sum();
        let paged_bytes = self
            .paged
            .as_ref()
            .zip(self.paged_layout.as_ref())
            .map_or(0, |(state, layout)| state.borrow().used_bytes(layout));
        dense_bytes + paged_bytes
    }

    /// Borrow this sequence's paged block-table state for reading.
    ///
    /// Returns a [`Ref`] guard; the borrow must be released before any code
    /// path mutates the same state (e.g. `update_and_fetch` during forward or
    /// `CachePool::sync_paged_state_with_*`). `None` for dense / model-owned
    /// sequences.
    pub fn paged_state(&self) -> Option<Ref<'_, PagedSequenceState>> {
        self.paged.as_ref().map(|rc| rc.borrow())
    }

    /// Mutably borrow this sequence's paged block-table state.
    ///
    /// Returns a [`RefMut`] guard. See [`Self::paged_state`] for the borrow
    /// discipline; never hold this across a call that re-borrows the same
    /// state.
    pub fn paged_state_mut(&mut self) -> Option<RefMut<'_, PagedSequenceState>> {
        self.paged.as_ref().map(|rc| rc.borrow_mut())
    }

    pub fn paged_stats(&self) -> Option<PagedCacheStats> {
        self.paged
            .as_ref()
            .zip(self.paged_layout.as_ref())
            .map(|(state, layout)| {
                let state = state.borrow();
                PagedCacheStats {
                    allocated_blocks: state.reserved_blocks(),
                    live_blocks: state.reserved_blocks(),
                    free_blocks: 0,
                    bytes_reserved: state.reserved_bytes(layout),
                    bytes_in_use: state.used_bytes(layout),
                }
            })
    }

    pub fn paged_layout(&self) -> Option<&PagedKvLayout> {
        self.paged_layout.as_ref()
    }
}

/// Pool that allocates and recycles per-sequence cache sets.
///
/// Designed for use by a continuous-batching scheduler. The pool assigns
/// monotonically increasing `SequenceId` values and enforces a hard upper
/// bound on concurrent active sequences.
///
/// Thread safety: `CachePool` itself is **not** `Sync`; callers in async
/// server code should wrap it in an appropriate lock (`Mutex` or `RwLock`).
pub struct CachePool {
    next_id: AtomicU64,
    active: HashMap<SequenceId, SequenceCacheSet>,
    max_sequences: usize,
    /// Shared paged block pool, lazily created on first paged allocation.
    ///
    /// Wrapped in `Rc<RefCell<…>>` so pool-backed [`KVCache`]s can hold a cheap
    /// `Rc` clone and write/read the pool transparently from inside
    /// `update_and_fetch` during forward, while the scheduler borrows the same
    /// pool at request boundaries. The two never nest: the per-request pool
    /// borrows here all drop before the model forward runs, and the forward's
    /// `update_and_fetch` borrows drop before control returns to the scheduler.
    paged_pool: Option<Rc<RefCell<PagedBlockPool>>>,
    /// Detached cache sets parked inside the pool during cross-request
    /// handoffs. See [`detach`] for the full design.
    detached: detach::DetachedMap,
    /// Paged block budget remembered across lazy pool creation. The pool is
    /// built on the first paged allocation, which may happen after the
    /// scheduler sets the budget, so the value is stored here and applied in
    /// [`CachePool::ensure_paged_pool`] when the pool is born (and immediately
    /// to a live pool in [`CachePool::set_paged_block_budget`]). `None` =
    /// unbounded (the default).
    paged_block_budget: Option<usize>,
}

impl CachePool {
    /// Create a new pool allowing up to `max_sequences` concurrent cache sets.
    pub fn new(max_sequences: usize) -> Self {
        Self {
            next_id: AtomicU64::new(0),
            active: HashMap::new(),
            max_sequences,
            paged_pool: None,
            detached: HashMap::new(),
            paged_block_budget: None,
        }
    }

    /// Allocate a fresh cache set for a new sequence.
    ///
    /// For batching models, calls `model.make_caches()` to build per-layer
    /// caches and enforces the `max_sequences` capacity limit.
    ///
    /// For non-batching models (internal RefCell/SSM caches), allocates a
    /// lightweight placeholder entry with dummy caches — without calling
    /// `make_caches()` — so that requests can be queued while another
    /// sequence is still generating.  The scheduler resets the model's
    /// internal caches at prefill time, not at enqueue time.
    pub fn allocate(
        &mut self,
        model: &dyn crate::generate::LanguageModel,
    ) -> Result<SequenceId, String> {
        self.allocate_with_layout(model, None)
    }

    /// Allocate a fresh cache set using either the model default layout or an
    /// explicit server-side override.
    pub fn allocate_with_layout(
        &mut self,
        model: &dyn crate::generate::LanguageModel,
        layout_override: Option<SequenceStateLayout>,
    ) -> Result<SequenceId, String> {
        let layout = layout_override.unwrap_or_else(|| model.sequence_state_layout());
        if layout.backend == SequenceStateBackend::DenseKvCache
            && self.active.len() >= self.max_sequences
        {
            return Err(format!(
                "CachePool: max capacity ({}) reached, cannot allocate new sequence",
                self.max_sequences
            ));
        }

        let id = SequenceId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let entry = match layout.backend {
            SequenceStateBackend::DenseKvCache => {
                SequenceCacheSet::dense_external(id, model.make_caches())
            }
            SequenceStateBackend::PagedKvCache => {
                let paged_layout = layout.paged_layout.ok_or_else(|| {
                    "CachePool: paged backend requires a paged layout".to_string()
                })?;
                self.ensure_paged_pool(&paged_layout)?;
                let state_rc = Rc::new(RefCell::new(PagedSequenceState::new(&paged_layout)));

                // Build the per-layer caches. Pool-backed caches are wired only
                // when BOTH hold:
                //
                //  * the model's NATURAL backend is the dense external KVCache
                //    slice (`supports_batching()` transformers — qwen3 / llama3).
                //    These actually read/write the caches handed to `forward`, so
                //    pool-backing routes their `update_and_fetch` straight into
                //    the shared `PagedBlockPool` (`write_prefill` + `gather_visible`).
                //    Model-owned families (gemma3 / llama4 / qwen3_5) ignore the
                //    caller's slice and drive their own `ModelOwnedSequenceState`;
                //    the scheduler still routes them through the paged backend for
                //    shadow block-table accounting (`sync_paged_state_with_lengths`),
                //    so their placeholders stay dense — byte-identical to before.
                //    Paged-natural test stubs likewise keep dense placeholders.
                //
                //  * the paged layout is Fp16. The pool-backed `update_and_fetch`
                //    intercept stores raw Fp16 K/V (the #152-validated path);
                //    Turbo4 paged layouts carry their own `cache_mode` and keep
                //    the existing dense quantized path untouched.
                let natural_backend = model.sequence_state_layout().backend;
                let pool_backed = natural_backend == SequenceStateBackend::DenseKvCache
                    && paged_layout.cache_mode == KVCacheMode::Fp16;
                let caches = if pool_backed {
                    let pool_rc = self
                        .paged_pool
                        .as_ref()
                        .expect("paged pool exists after ensure_paged_pool")
                        .clone();
                    (0..paged_layout.num_layers)
                        .map(|layer_idx| {
                            KVCache::new_paged(pool_rc.clone(), state_rc.clone(), layer_idx)
                        })
                        .collect()
                } else {
                    model.make_caches()
                };

                SequenceCacheSet::with_backend(
                    id,
                    SequenceStateBackend::PagedKvCache,
                    caches,
                    Some(state_rc),
                    Some(paged_layout),
                )
            }
            SequenceStateBackend::ModelOwned => SequenceCacheSet::model_owned(id),
        };
        self.active.insert(id, entry);
        Ok(id)
    }

    /// Return an immutable reference to the full `SequenceCacheSet` for the
    /// given sequence, or `None` if the ID is not active.
    pub fn get(&self, id: SequenceId) -> Option<&SequenceCacheSet> {
        self.active.get(&id)
    }

    /// Return a mutable reference to the full `SequenceCacheSet` for the
    /// given sequence, or `None` if the ID is not active.
    pub fn get_mut(&mut self, id: SequenceId) -> Option<&mut SequenceCacheSet> {
        self.active.get_mut(&id)
    }

    pub fn get_paged_state(&self, id: SequenceId) -> Option<Ref<'_, PagedSequenceState>> {
        self.active.get(&id)?.paged_state()
    }

    pub fn get_paged_state_mut(
        &mut self,
        id: SequenceId,
    ) -> Option<RefMut<'_, PagedSequenceState>> {
        self.active.get_mut(&id)?.paged_state_mut()
    }

    /// Return a mutable slice of the per-layer KV caches for direct use
    /// in `model.forward()`, or `None` if the ID is not active.
    pub fn get_caches_mut(&mut self, id: SequenceId) -> Option<&mut [KVCache]> {
        self.active.get_mut(&id).map(|s| s.caches.as_mut_slice())
    }

    /// Return cache slices for multiple active sequences in one call.
    ///
    /// This centralizes the aliasing/unsafe boundary so scheduler code does
    /// not need to reconstruct `&mut [KVCache]` slices from raw pointers.
    pub fn get_batch_caches_mut<'a>(
        &'a mut self,
        ids: &[SequenceId],
    ) -> Result<Vec<&'a mut [KVCache]>, String> {
        let mut cache_ptrs: Vec<(*mut KVCache, usize)> = Vec::with_capacity(ids.len());
        for &id in ids {
            let (ptr, len) = {
                let caches = self
                    .get_caches_mut(id)
                    .ok_or_else(|| format!("CachePool: sequence {id} not found"))?;
                (caches.as_mut_ptr(), caches.len())
            };
            cache_ptrs.push((ptr, len));
        }

        // SAFETY: each `SequenceId` maps to a distinct `SequenceCacheSet`
        // allocation inside the HashMap, and callers ensure the same id is not
        // requested twice in one batch. The returned slices are tied to the
        // lifetime of `&mut self` and no mutation of `self.active` occurs
        // between pointer extraction and slice reconstruction.
        Ok(cache_ptrs
            .iter()
            .map(|&(ptr, len)| unsafe { std::slice::from_raw_parts_mut(ptr, len) })
            .collect())
    }

    /// Release a sequence's caches, reclaiming the pool slot.
    ///
    /// This is a no-op if `id` is not currently active.
    pub fn release(&mut self, id: SequenceId) {
        if let Some(sequence) = self.active.remove(&id) {
            // Borrow the pool and this sequence's state through their shared
            // cells. The released sequence's pool-backed caches (if any) hold
            // sibling `Rc` clones of `state`, but they are never forwarded at
            // release time, so `borrow_mut` here cannot collide. Both borrows
            // drop before `sequence` (and thus those caches) drop.
            if let (Some(pool), Some(state)) = (self.paged_pool.as_ref(), sequence.paged.as_ref()) {
                let _ = pool.borrow_mut().release_sequence(&mut state.borrow_mut());
            }
        }
    }

    /// Number of sequences currently holding active cache sets.
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// Sum of `nbytes()` across all active cache sets, plus any detached
    /// cache sets currently parked inside the pool.
    ///
    /// Parked sets are tensors the pool still physically holds in-flight
    /// during cross-request handoffs (see `detach` / `adopt`), so including
    /// them here keeps memory accounting consistent for schedulers that
    /// admission-control on `memory_usage_bytes()`.
    ///
    /// Used by: `BatchScheduler::update_gauges` (server observability), any
    /// admission-control path that caps KV memory before accepting new work.
    pub fn memory_usage_bytes(&self) -> usize {
        let active_bytes: usize = self.active.values().map(|s| s.nbytes()).sum();
        let parked_bytes: usize = self.detached.values().map(|d| d.nbytes()).sum();
        // Per-page Turbo4 sidecars and physical main-K/V pool tensors in the
        // paged pool are owned by the pool itself rather than by any individual
        // `SequenceCacheSet`, so they are not visible to the per-sequence
        // `nbytes()` walk above. Add them explicitly so admission-control sees
        // the true KV footprint for paged deployments. The main-K/V pool
        // tensors are lazily allocated, so this contributes 0 until a writer is
        // wired (#120) and never perturbs the layout-derived scheduling budgets
        // (`reserved_bytes`/`used_bytes`).
        let pool_bytes: usize = self
            .paged_pool
            .as_ref()
            .map(|pool| {
                let pool = pool.borrow();
                pool.turbo_sidecar_bytes() + pool.pool_tensor_bytes()
            })
            .unwrap_or(0);
        active_bytes + parked_bytes + pool_bytes
    }

    /// Run `f` with mutable access to both the shared block pool and one
    /// sequence's paged state, borrowing each cell exactly once.
    ///
    /// Centralizes the dual `borrow_mut` so the public paged-token mutators
    /// keep tight, non-overlapping borrow scopes (the pool and the per-sequence
    /// state are distinct `RefCell`s, so the two `borrow_mut`s never collide).
    /// The closure must not re-borrow either cell.
    fn with_pool_and_state<R>(
        &self,
        id: SequenceId,
        f: impl FnOnce(&mut PagedBlockPool, &mut PagedSequenceState) -> Result<R, String>,
    ) -> Result<R, String> {
        let pool = self
            .paged_pool
            .as_ref()
            .ok_or_else(|| "CachePool: paged backend is not initialized".to_string())?;
        let state = self
            .active
            .get(&id)
            .ok_or_else(|| format!("CachePool: sequence {id} not found"))?
            .paged
            .as_ref()
            .ok_or_else(|| format!("CachePool: sequence {id} is not paged"))?;
        let mut pool = pool.borrow_mut();
        let mut state = state.borrow_mut();
        f(&mut pool, &mut state)
    }

    pub fn append_paged_tokens(
        &mut self,
        id: SequenceId,
        layer_idx: usize,
        token_count: usize,
    ) -> Result<(), String> {
        self.with_pool_and_state(id, |pool, state| {
            pool.append_tokens(state, layer_idx, token_count)
        })
    }

    pub fn trim_paged_tokens(
        &mut self,
        id: SequenceId,
        layer_idx: usize,
        token_count: usize,
    ) -> Result<usize, String> {
        self.with_pool_and_state(id, |pool, state| {
            pool.trim_tokens(state, layer_idx, token_count)
        })
    }

    pub fn rewind_paged_tokens(
        &mut self,
        id: SequenceId,
        layer_idx: usize,
        token_count: usize,
    ) -> Result<usize, String> {
        self.with_pool_and_state(id, |pool, state| {
            pool.rewind_tokens(state, layer_idx, token_count)
        })
    }

    pub fn paged_stats(&self) -> Option<PagedCacheStats> {
        // Pool-wide stats (#226): block counts and REAL slab bytes come from
        // the pool itself, covering active sequences and parked prompt-cache
        // pins alike, so no per-sequence borrows are needed anymore.
        self.paged_pool.as_ref().map(|pool| pool.borrow().stats())
    }

    pub fn paged_block_size(&self) -> Option<usize> {
        self.paged_pool
            .as_ref()
            .map(|pool| pool.borrow().layout().block_size)
    }

    /// Set (or clear, with `None`) the paged pool's global block budget — the
    /// cap on distinct physical blocks the pool may allocate. Opt-in; `None`
    /// (the default) leaves the pool unbounded. No-op when there is no paged
    /// pool (dense-only configuration). The scheduler derives the value from
    /// the configured / estimated KV byte budget (#122).
    pub fn set_paged_block_budget(&mut self, max_blocks: Option<usize>) {
        self.paged_block_budget = max_blocks;
        if let Some(pool) = self.paged_pool.as_ref() {
            pool.borrow_mut().set_block_budget(max_blocks);
        }
    }

    /// The configured paged block budget, or `None` when unbounded. This is the
    /// stored intent, so it is correct even before the pool is lazily created
    /// (the value is applied to the pool on creation).
    pub fn paged_block_budget(&self) -> Option<usize> {
        self.paged_block_budget
    }

    /// Blocks still **acquirable** before the paged budget is hit
    /// (`budget − live`), or `None` when unbounded / no paged pool. `Some(0)`
    /// means every budgeted block is in use; the admission gate must reclaim
    /// (evict cold prefixes, then preempt) or defer. Eviction raises this even
    /// though allocated rows are retained, because freed blocks are reusable.
    pub fn free_paged_block_budget(&self) -> Option<usize> {
        self.paged_pool
            .as_ref()
            .and_then(|pool| pool.borrow().free_block_budget())
    }

    /// Read-only access to the underlying [`PagedBlockPool`].
    ///
    /// Returns a [`Ref`] guard; release it before any path that mutates the
    /// pool. Used by: tests and read-only diagnostics that need to peek at
    /// per-page Turbo4 sidecar state without going through the higher-level
    /// `CachePool` API.
    pub fn paged_pool_ref(&self) -> Option<Ref<'_, PagedBlockPool>> {
        self.paged_pool.as_ref().map(|pool| pool.borrow())
    }

    /// Mutable access to the underlying [`PagedBlockPool`].
    ///
    /// Returns a [`RefMut`] guard tied to `&mut self`, so the compiler still
    /// enforces exclusive pool access at the borrow site. Used by: scheduler /
    /// model code that installs Turbo4 sidecar pages directly on the pool, and
    /// by unit tests for the same purpose.
    pub fn paged_pool_mut(&mut self) -> Option<RefMut<'_, PagedBlockPool>> {
        self.paged_pool.as_ref().map(|pool| pool.borrow_mut())
    }

    /// Mirror the visible dense-cache offsets into the paged backend state for
    /// one sequence.
    ///
    /// This keeps server decode/pre-fill lifecycle bookkeeping aligned while
    /// the actual model execution still runs on dense compatibility caches.
    pub fn sync_paged_state_with_dense(&mut self, id: SequenceId) -> Result<(), String> {
        let target_lens = {
            let sequence = self
                .active
                .get(&id)
                .ok_or_else(|| format!("CachePool: sequence {id} not found"))?;
            // Pool-backed paged sequences are authoritative: `write_prefill`
            // advances each layer's `len` in lockstep with the cache offset
            // during forward, so mirroring the dense-cache lengths here is
            // redundant — and the dense placeholder buffers are empty, so the
            // mirror would trim the live block table to zero. Skip them.
            // Model-owned families never reach this path; they override
            // `sync_sequence_storage` to call `sync_paged_state_with_lengths`
            // with their own model-owned cache lengths.
            if sequence.caches.iter().any(|cache| cache.is_paged_backed()) {
                return Ok(());
            }
            sequence
                .caches
                .iter()
                .map(|cache| cache.seq_len().max(0) as usize)
                .collect::<Vec<usize>>()
        };
        self.sync_paged_state_with_lengths(id, &target_lens)
    }

    /// Mirror explicit visible lengths into the paged backend state for one sequence.
    pub fn sync_paged_state_with_lengths(
        &mut self,
        id: SequenceId,
        target_lens: &[usize],
    ) -> Result<(), String> {
        let pool = match self.paged_pool.as_ref() {
            Some(pool) => pool,
            None => return Ok(()),
        };
        let state_cell = match self.active.get(&id) {
            Some(sequence) => match sequence.paged.as_ref() {
                Some(state) => state,
                None => return Ok(()),
            },
            None => return Err(format!("CachePool: sequence {id} not found")),
        };

        // The pool and the per-sequence state live in distinct `RefCell`s, so
        // both `borrow_mut`s coexist. For model-owned models (gemma3/llama4/
        // qwen3_5) this is the authoritative length sync; for the pool-backed
        // dense-natural path (qwen3/llama3) the lengths already match because
        // `write_prefill` advanced them in lockstep, so the loop is a no-op.
        let mut pool = pool.borrow_mut();
        let mut state = state_cell.borrow_mut();

        if target_lens.len() != state.layers.len() {
            return Err(format!(
                "CachePool: expected {} paged layer lengths for {id}, got {}",
                state.layers.len(),
                target_lens.len()
            ));
        }

        for (layer_idx, target_len) in target_lens.iter().copied().enumerate() {
            let current_len = state.layers[layer_idx].len;
            if target_len > current_len {
                pool.append_tokens(&mut state, layer_idx, target_len - current_len)?;
            } else if target_len < current_len {
                pool.trim_tokens(&mut state, layer_idx, current_len - target_len)?;
            }
        }
        Ok(())
    }

    /// Restore externally serialized paged state into an active sequence and
    /// register its blocks with the shared allocator.
    pub fn restore_paged_state(
        &mut self,
        id: SequenceId,
        restored: PagedSequenceState,
    ) -> Result<(), String> {
        let layout = {
            let sequence = self
                .active
                .get(&id)
                .ok_or_else(|| format!("CachePool: sequence {id} not found"))?;
            if sequence.backend != SequenceStateBackend::PagedKvCache {
                return Err(format!(
                    "CachePool: sequence {id} is not using the paged backend"
                ));
            }

            sequence
                .paged_layout
                .clone()
                .ok_or_else(|| format!("CachePool: sequence {id} is missing paged layout"))?
        };
        if restored.block_size != layout.block_size {
            return Err(format!(
                "CachePool: restored block size {} does not match layout block size {}",
                restored.block_size, layout.block_size
            ));
        }
        if restored.layers.len() != layout.num_layers {
            return Err(format!(
                "CachePool: restored layer count {} does not match layout layer count {}",
                restored.layers.len(),
                layout.num_layers
            ));
        }

        let existing = {
            let sequence = self
                .active
                .get_mut(&id)
                .ok_or_else(|| format!("CachePool: sequence {id} not found"))?;
            sequence.paged.take()
        };

        self.ensure_paged_pool(&layout)?;
        {
            let pool = self
                .paged_pool
                .as_ref()
                .expect("paged pool must exist after ensure_paged_pool");
            let mut pool = pool.borrow_mut();
            if let Some(existing) = existing.as_ref() {
                pool.release_sequence(&mut existing.borrow_mut())?;
            }
            pool.restore_sequence(&restored)?;
        }

        let sequence = self
            .active
            .get_mut(&id)
            .ok_or_else(|| format!("CachePool: sequence {id} not found"))?;
        sequence.paged = Some(Rc::new(RefCell::new(restored)));
        Ok(())
    }

    /// Extract the live K/V contents of every pool block backing a paged
    /// sequence, for distributed transfer (#125).
    ///
    /// Returns one [`PagedBlockContents`] per distinct physical block in the
    /// sequence's block table, carrying the ORIGIN node's block id + layer and
    /// the block's full `[block_size, n_kv_heads, head_dim]` K/V slabs. The
    /// decode node remaps these to fresh ids in
    /// [`Self::restore_paged_state_with_contents`].
    ///
    /// Returns an empty vector for dense / model-owned sequences, for a paged
    /// sequence with no block table, or when no paged pool exists.
    pub fn extract_paged_blocks(&self, id: SequenceId) -> Result<Vec<PagedBlockContents>, String> {
        let sequence = self
            .active
            .get(&id)
            .ok_or_else(|| format!("CachePool: sequence {id} not found"))?;
        if sequence.backend != SequenceStateBackend::PagedKvCache || sequence.paged.is_none() {
            return Ok(Vec::new());
        }
        // Only POOL-BACKED (Fp16 paged) sequences keep their K/V in the shared
        // pool. A model-owned paged sequence (gemma3 / llama4 / qwen3_5) holds
        // dense placeholder caches and keeps its KV model-internal, with the pool
        // tracking a shadow block table only, so there is nothing pool-paged to
        // transfer. `caches.is_empty()` (a paged-natural stub) is vacuously
        // pool-backed and falls through to the empty block-table fast path below.
        if !sequence.caches.iter().all(|cache| cache.is_paged_backed()) {
            return Ok(Vec::new());
        }
        let pool = match self.paged_pool.as_ref() {
            Some(pool) => pool.borrow(),
            None => return Ok(Vec::new()),
        };
        let state = sequence
            .paged_state()
            .expect("paged state present (checked above)");

        let mut blocks = Vec::new();
        // Block ids are globally unique and each belongs to exactly one layer, so
        // a sequence-wide dedup never drops a needed block and prevents
        // transferring the same physical block twice.
        let mut seen: HashSet<u64> = HashSet::new();
        for (layer_idx, layer) in state.layers.iter().enumerate() {
            for &block_id in &layer.block_ids {
                if !seen.insert(block_id.as_u64()) {
                    continue;
                }
                let (keys, values) = pool.read_block_contents(block_id, layer_idx)?;
                blocks.push(PagedBlockContents {
                    block_id,
                    layer_idx,
                    keys,
                    values,
                });
            }
        }
        Ok(blocks)
    }

    /// Restore externally serialized paged state INTO an active pool-backed
    /// sequence, materializing transferred block CONTENTS on fresh physical
    /// rows (#125).
    ///
    /// Unlike [`Self::restore_paged_state`] (which only re-registers the origin
    /// node's block ids as metadata and assumes the contents already live in the
    /// pool), this path is for a true cross-node handoff: it acquires a FRESH
    /// block for every transferred [`PagedBlockContents`], writes the K/V slab,
    /// and rebuilds the sequence's block table over the fresh ids. Each fresh
    /// block starts at refcount 1, so block accounting matches the origin node.
    ///
    /// Requires the sequence to already be allocated with POOL-BACKED caches
    /// (Fp16 paged, the #121 `new_paged` path), because the contents are stored
    /// in the shared [`PagedBlockPool`]. The existing per-layer caches keep their
    /// shared block-table cell (its contents are replaced IN PLACE, never the
    /// `Rc` itself) so the restored state stays visible to them, and each cache's
    /// RoPE `offset` is advanced to the restored prefix length (mirrors
    /// `adopt_paged_preserving`; assumes the post-prefill `logical_start == 0`
    /// invariant so `len` is the prefix length).
    pub fn restore_paged_state_with_contents(
        &mut self,
        id: SequenceId,
        restored: PagedSequenceState,
        contents: Vec<PagedBlockContents>,
    ) -> Result<(), String> {
        let layout = {
            let sequence = self
                .active
                .get(&id)
                .ok_or_else(|| format!("CachePool: sequence {id} not found"))?;
            if sequence.backend != SequenceStateBackend::PagedKvCache {
                return Err(format!(
                    "CachePool: sequence {id} is not using the paged backend"
                ));
            }
            if !sequence.caches.iter().all(|c| c.is_paged_backed()) {
                return Err(format!(
                    "CachePool: restore_paged_state_with_contents requires a pool-backed sequence (sequence {id} has non-pool-backed caches)"
                ));
            }
            sequence
                .paged_layout
                .clone()
                .ok_or_else(|| format!("CachePool: sequence {id} is missing paged layout"))?
        };
        if restored.block_size != layout.block_size {
            return Err(format!(
                "CachePool: restored block size {} does not match layout block size {}",
                restored.block_size, layout.block_size
            ));
        }
        if restored.layers.len() != layout.num_layers {
            return Err(format!(
                "CachePool: restored layer count {} does not match layout layer count {}",
                restored.layers.len(),
                layout.num_layers
            ));
        }

        self.ensure_paged_pool(&layout)?;

        // Materialize the transferred contents on fresh rows and rebuild the
        // block table over the new ids. The pool and the per-sequence state live
        // in distinct `RefCell`s (and `pool_rc` is a cloned handle independent of
        // `self`), so the pool `borrow_mut` here coexists with the state borrow.
        let pool_rc = self
            .paged_pool
            .as_ref()
            .expect("paged pool must exist after ensure_paged_pool")
            .clone();
        let (remapped, offsets) = {
            let mut pool = pool_rc.borrow_mut();

            // Release any blocks the sequence already holds (a freshly allocated
            // sequence has none, so this is normally a no-op) so re-restoring
            // never leaks. The shared state cell is KEPT (the caches clone it);
            // only its contents are replaced below.
            {
                let sequence = self
                    .active
                    .get(&id)
                    .ok_or_else(|| format!("CachePool: sequence {id} not found"))?;
                if let Some(state_rc) = sequence.paged.as_ref() {
                    pool.release_sequence(&mut state_rc.borrow_mut())?;
                }
            }

            // Acquire + write a fresh block for every transferred origin block.
            // Track the fresh ids so any failure mid-restore releases them
            // instead of leaking pool blocks: a malformed/oversized transferred
            // slab, an out-of-range layer, a write that exceeds the block
            // budget, or a block-table entry with no matching contents must not
            // strand already-acquired blocks (#125 hardening).
            let mut acquired: Vec<PagedBlockId> = Vec::with_capacity(contents.len());
            let mut map: HashMap<u64, PagedBlockId> = HashMap::new();
            for content in &contents {
                match pool.acquire_and_write_block(
                    content.layer_idx,
                    &content.keys,
                    &content.values,
                ) {
                    Ok(fresh) => {
                        acquired.push(fresh);
                        map.insert(content.block_id.as_u64(), fresh);
                    }
                    Err(e) => {
                        for id in acquired.drain(..) {
                            let _ = pool.release_block(id);
                        }
                        return Err(e);
                    }
                }
            }

            // Remap the block table over the fresh ids, preserving len /
            // logical_start.
            let mut layers = Vec::with_capacity(restored.layers.len());
            for layer in &restored.layers {
                let mut block_ids = Vec::with_capacity(layer.block_ids.len());
                for origin in &layer.block_ids {
                    let fresh = match map.get(&origin.as_u64()).copied() {
                        Some(fresh) => fresh,
                        None => {
                            for id in acquired.drain(..) {
                                let _ = pool.release_block(id);
                            }
                            return Err(format!(
                                "CachePool: missing transferred contents for block {origin}"
                            ));
                        }
                    };
                    block_ids.push(fresh);
                }
                layers.push(PagedLayerState {
                    block_ids,
                    len: layer.len,
                    logical_start: layer.logical_start,
                });
            }
            let offsets: Vec<i32> = layers.iter().map(|layer| layer.len as i32).collect();
            (
                PagedSequenceState {
                    block_size: restored.block_size,
                    layers,
                },
                offsets,
            )
        };

        // Install the remapped state IN PLACE (keep the shared cell the
        // pool-backed caches clone) and advance each cache's RoPE offset to the
        // restored prefix length.
        let sequence = self
            .active
            .get_mut(&id)
            .ok_or_else(|| format!("CachePool: sequence {id} not found"))?;
        {
            let state_rc = sequence
                .paged
                .as_ref()
                .ok_or_else(|| format!("CachePool: sequence {id} has no paged state cell"))?;
            *state_rc.borrow_mut() = remapped;
        }
        for (layer_idx, cache) in sequence.caches.iter_mut().enumerate() {
            if let Some(&offset) = offsets.get(layer_idx) {
                cache.offset = offset;
            }
        }
        Ok(())
    }

    /// Maximum number of concurrent sequences this pool allows.
    pub fn max_sequences(&self) -> usize {
        self.max_sequences
    }

    fn ensure_paged_pool(&mut self, layout: &PagedKvLayout) -> Result<(), String> {
        if let Some(pool) = self.paged_pool.as_ref() {
            if pool.borrow().layout() != layout {
                return Err("CachePool: paged layout mismatch for active paged backend".to_string());
            }
            return Ok(());
        }
        let mut pool = PagedBlockPool::new(layout.clone());
        // Apply the budget remembered before the pool existed (set via
        // `set_paged_block_budget` before the first paged allocation).
        pool.set_block_budget(self.paged_block_budget);
        self.paged_pool = Some(Rc::new(RefCell::new(pool)));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_cache_trim_clears_storage_when_fully_rewound() {
        let mut cache = KVCache::new();
        let keys = ffi::from_slice_f32(&[1.0, 2.0], &[1, 1, 2, 1]);
        let values = ffi::from_slice_f32(&[3.0, 4.0], &[1, 1, 2, 1]);
        cache.update(keys, values);

        assert_eq!(cache.seq_len(), 2);
        assert!(!cache.is_empty());
        assert_eq!(cache.trim(5), 2);
        assert_eq!(cache.seq_len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn rotating_cache_multi_token_append_after_trim_uses_visible_prefix() {
        let mut cache = RotatingKVCache::new(16);
        let keys = ffi::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4, 1]);
        let values = ffi::from_slice_f32(&[10.0, 20.0, 30.0, 40.0], &[1, 1, 4, 1]);
        let _ = cache.update_and_fetch(keys, values);

        assert_eq!(cache.offset, 4);
        assert_eq!(cache.idx, 4);
        assert_eq!(cache.seq_len(), 4);

        // Speculative rollback rewinds the logical write point but leaves
        // the backing allocation intact, matching upstream rotating-cache
        // semantics.
        assert_eq!(cache.trim(3), 3);
        assert_eq!(cache.offset, 1);
        assert_eq!(cache.idx, 1);
        assert_eq!(cache.seq_len(), 4);

        let new_keys = ffi::from_slice_f32(&[5.0, 6.0, 7.0, 8.0], &[1, 1, 4, 1]);
        let new_values = ffi::from_slice_f32(&[50.0, 60.0, 70.0, 80.0], &[1, 1, 4, 1]);
        let (visible_keys, visible_values) = cache.update_and_fetch(new_keys, new_values);

        assert_eq!(cache.offset, 5);
        assert_eq!(cache.idx, 5);
        assert_eq!(ffi::array_shape(&visible_keys), vec![1, 1, 5, 1]);
        assert_eq!(ffi::array_shape(&visible_values), vec![1, 1, 5, 1]);

        let to_f32 = |arr: &MlxArray| {
            ffi::eval(arr);
            ffi::array_to_raw_bytes(arr)
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<_>>()
        };
        assert_eq!(to_f32(&visible_keys), vec![1.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(to_f32(&visible_values), vec![10.0, 50.0, 60.0, 70.0, 80.0]);
    }

    #[test]
    fn rotating_cache_speculative_buffer_preserves_temporal_prefix() {
        let mut cache = RotatingKVCache::new(4);
        cache
            .enable_speculative_buffer(2)
            .expect("fp16 rotating cache supports speculative buffering");

        let token = |x: f32| ffi::from_slice_f32(&[x], &[1, 1, 1, 1]);
        let mut visible = None;
        for i in 1..=6 {
            let (keys, _) = cache.update_and_fetch(token(i as f32), token((i * 10) as f32));
            visible = Some(keys);
        }

        let to_f32 = |arr: &MlxArray| {
            ffi::eval(arr);
            ffi::array_to_raw_bytes(arr)
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<_>>()
        };

        let visible = visible.expect("updates produced a visible prefix");
        assert_eq!(cache.get_offset(), 6);
        assert_eq!(cache.buffer_write_idx(), 6);
        assert_eq!(ffi::array_shape(&visible), vec![1, 1, 6, 1]);
        assert_eq!(to_f32(&visible), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // The first overflow past `max_size + buffer_size` compacts the
        // prefix instead of ring-wrapping, so the returned state stays in
        // chronological order and rollback can safely rewind from the tail.
        let (visible, _) = cache.update_and_fetch(token(7.0), token(70.0));
        assert_eq!(cache.get_offset(), 7);
        assert_eq!(cache.buffer_write_idx(), 4);
        assert_eq!(ffi::array_shape(&visible), vec![1, 1, 4, 1]);
        assert_eq!(to_f32(&visible), vec![4.0, 5.0, 6.0, 7.0]);

        assert_eq!(cache.trim(2), 2);
        let new_keys = ffi::from_slice_f32(&[8.0, 9.0], &[1, 1, 2, 1]);
        let new_values = ffi::from_slice_f32(&[80.0, 90.0], &[1, 1, 2, 1]);
        let (visible, _) = cache.update_and_fetch(new_keys, new_values);
        assert_eq!(cache.get_offset(), 7);
        assert_eq!(cache.buffer_write_idx(), 4);
        assert_eq!(to_f32(&visible), vec![4.0, 5.0, 8.0, 9.0]);
    }

    #[test]
    fn rotating_cache_speculative_buffer_conversion_keeps_last_window() {
        let mut cache = RotatingKVCache::new(4);
        let keys = ffi::from_slice_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 1, 6, 1]);
        let values = ffi::from_slice_f32(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[1, 1, 6, 1]);
        let _ = cache.update_and_fetch(keys, values);

        cache
            .enable_speculative_buffer(2)
            .expect("fp16 rotating cache supports speculative buffering");

        let token = |x: f32| ffi::from_slice_f32(&[x], &[1, 1, 1, 1]);
        let (visible, _) = cache.update_and_fetch(token(7.0), token(70.0));
        let to_f32 = |arr: &MlxArray| {
            ffi::eval(arr);
            ffi::array_to_raw_bytes(arr)
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<_>>()
        };

        assert_eq!(cache.get_offset(), 7);
        assert_eq!(cache.buffer_write_idx(), 5);
        assert_eq!(ffi::array_shape(&visible), vec![1, 1, 5, 1]);
        assert_eq!(to_f32(&visible), vec![3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn rotating_cache_speculative_buffer_conversion_reorders_wrapped_ring() {
        let mut cache = RotatingKVCache::new(4);
        let token = |x: f32| ffi::from_slice_f32(&[x], &[1, 1, 1, 1]);
        for i in 1..=6 {
            cache.update_and_fetch(token(i as f32), token((i * 10) as f32));
        }
        assert_eq!(cache.get_offset(), 6);
        assert_eq!(cache.buffer_write_idx(), 2);

        cache
            .enable_speculative_buffer(2)
            .expect("fp16 rotating cache supports speculative buffering");
        let (visible, _) = cache.update_and_fetch(token(7.0), token(70.0));
        let to_f32 = |arr: &MlxArray| {
            ffi::eval(arr);
            ffi::array_to_raw_bytes(arr)
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<_>>()
        };

        assert_eq!(cache.get_offset(), 7);
        assert_eq!(cache.buffer_write_idx(), 5);
        assert_eq!(to_f32(&visible), vec![3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    // `trim_front` drops the oldest `n` tokens from a KVCache's
    // live window so the server scheduler can enforce a `--max-kv-size`
    // bound on otherwise unbounded plain `KVCache` instances.
    //
    // CRITICAL: `self.offset` is the monotonic RoPE position and **never**
    // decreases. `trim_front` advances `self.live_start` instead. The tests
    // below assert the post-trim invariant pair `(offset, live_start)` so a
    // future regression to "decrement offset on trim" (which silently breaks
    // RoPE relative positions) is caught immediately. See
    // [`KVCache::trim_front`] doc-comment for the full position invariant.
    #[test]
    fn kv_cache_trim_front_drops_oldest_entries_fp16() {
        // Helper: read an MlxArray as Vec<f32>. Uses the same astype +
        // raw-bytes pattern as `flatten_fp32` in `turbo_tests.rs` so cache
        // tests stay independent of the FFI surface beyond the slice helpers
        // already exercised here.
        fn read_f32(arr: &ffi::MlxArray) -> Vec<f32> {
            let a = ffi::astype(arr, dtype::FLOAT32);
            ffi::eval(&a);
            ffi::array_to_raw_bytes(&a)
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }

        let mut cache = KVCache::new();
        let keys = ffi::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4, 1]);
        let values = ffi::from_slice_f32(&[5.0, 6.0, 7.0, 8.0], &[1, 1, 4, 1]);
        cache.update(keys, values);
        assert_eq!(cache.seq_len(), 4);
        assert_eq!(cache.offset, 4);
        assert_eq!(cache.live_start, 0);

        // Drop the two oldest tokens; the visible window must keep the
        // tail (tokens 3, 4 for keys / 7, 8 for values).
        assert_eq!(cache.trim_front(2), 2);
        assert_eq!(cache.seq_len(), 2);
        // RoPE invariant: monotonic offset is unchanged after trim.
        assert_eq!(cache.offset, 4);
        // The dropped tokens are accounted for by advancing the live
        // window start, not by rolling back the monotonic offset.
        assert_eq!(cache.live_start, 2);
        assert_eq!(cache.live_len(), 2);

        let k_data = read_f32(cache.keys.as_ref().unwrap());
        let v_data = read_f32(cache.values.as_ref().unwrap());
        assert_eq!(k_data, vec![3.0, 4.0]);
        assert_eq!(v_data, vec![7.0, 8.0]);
    }

    #[test]
    fn kv_cache_trim_front_clears_buffers_when_dropping_all_entries() {
        let mut cache = KVCache::new();
        let keys = ffi::from_slice_f32(&[1.0, 2.0], &[1, 1, 2, 1]);
        let values = ffi::from_slice_f32(&[3.0, 4.0], &[1, 1, 2, 1]);
        cache.update(keys, values);

        assert_eq!(cache.trim_front(2), 2);
        // Live window is empty but the monotonic position is preserved.
        assert_eq!(cache.live_len(), 0);
        assert_eq!(cache.offset, 2);
        assert_eq!(cache.live_start, 2);
        // On-device buffers are reset to free memory.
        assert!(cache.keys.is_none());
        assert!(cache.values.is_none());
        // `is_empty` reports `false` because the monotonic offset is non-zero
        // — this is load-bearing for the dense fused-causal-prefill fast
        // path in `Llama3Attention` which hard-codes RoPE to `[0, l)` and
        // therefore must not re-engage on a post-trim cache where the next
        // Q is rotated at a non-zero position.
        assert!(!cache.is_empty());
    }

    #[test]
    fn kv_cache_trim_front_clamps_negative_and_zero() {
        let mut cache = KVCache::new();
        let keys = ffi::from_slice_f32(&[1.0, 2.0], &[1, 1, 2, 1]);
        let values = ffi::from_slice_f32(&[3.0, 4.0], &[1, 1, 2, 1]);
        cache.update(keys, values);

        assert_eq!(cache.trim_front(0), 0);
        assert_eq!(cache.trim_front(-5), 0);
        // Original window is preserved; live_start is still 0.
        assert_eq!(cache.offset, 2);
        assert_eq!(cache.live_start, 0);
        assert_eq!(cache.live_len(), 2);
    }

    #[test]
    fn kv_cache_trim_front_clamps_to_live_len_when_n_exceeds_size() {
        let mut cache = KVCache::new();
        let keys = ffi::from_slice_f32(&[1.0, 2.0, 3.0], &[1, 1, 3, 1]);
        let values = ffi::from_slice_f32(&[4.0, 5.0, 6.0], &[1, 1, 3, 1]);
        cache.update(keys, values);

        // Requesting more than the live size clamps and clears the live
        // window. Monotonic `offset` stays at 3; `live_start` jumps to 3.
        assert_eq!(cache.trim_front(10), 3);
        assert_eq!(cache.offset, 3);
        assert_eq!(cache.live_start, 3);
        assert_eq!(cache.live_len(), 0);
        // Buffers freed (live window empty); `is_empty()` is still false
        // because the monotonic offset is non-zero.
        assert!(cache.keys.is_none());
        assert!(!cache.is_empty());
    }

    #[test]
    fn kv_cache_trim_front_is_noop_for_turbo_modes() {
        // Turbo modes maintain per-token rotation state in sidecars
        // (`turbo_params`, `v_packed`, `v_norms`, …); head-trim is
        // unsupported and must be a safe no-op. `--max-kv-size` is
        // documented as incompatible with Turbo KV quantization in v1.
        // `live_start` must stay at `0` for these modes so the Turbo fetch
        // paths that still slice `[0..self.offset]` continue to be correct.
        let mut cache = KVCache::new();
        cache.mode = KVCacheMode::Turbo4Asym;
        cache.offset = 5;
        // Fabricate visible offset; no buffers attached because we never
        // call update(). trim_front should refuse and return 0.
        assert_eq!(cache.trim_front(2), 0);
        assert_eq!(cache.offset, 5);
        assert_eq!(cache.live_start, 0);
    }

    // `trim_front` must preserve the RoPE relative-position
    // invariant after the live window is bounded — K at buffer slot `i`
    // was rotated at monotonic position `live_start + i` at write time,
    // and Q for the next decode step is rotated at the *current*
    // monotonic `offset`. A naive `trim_front` that decrements `offset`
    // would shift relative positions by `-n` and silently collapse the
    // model output the moment the cap kicks in.
    //
    // This test exercises the invariant by writing N tokens, trimming `n`,
    // writing one more token, and verifying:
    //  - `cache.offset` is `N + 1` (monotonic, never rolled back).
    //  - `cache.live_start` is `n` (advanced by trim_front).
    //  - `cache.live_len()` is `(N + 1) - n` — the live window is
    //    [trim..N+1] in monotonic-position space.
    //  - The buffer holds the surviving live tokens plus the newest write,
    //    written at slot `(live_len - 1)` (not at slot `offset`).
    #[test]
    fn kv_cache_trim_front_preserves_monotonic_offset_under_subsequent_writes() {
        fn read_f32(arr: &ffi::MlxArray) -> Vec<f32> {
            let a = ffi::astype(arr, dtype::FLOAT32);
            ffi::eval(&a);
            ffi::array_to_raw_bytes(&a)
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }

        let mut cache = KVCache::new();
        let keys = ffi::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4, 1]);
        let values = ffi::from_slice_f32(&[5.0, 6.0, 7.0, 8.0], &[1, 1, 4, 1]);
        cache.update(keys, values);
        // Drop the two oldest tokens — live window is now [3, 4] / [7, 8].
        assert_eq!(cache.trim_front(2), 2);
        assert_eq!(cache.offset, 4);
        assert_eq!(cache.live_start, 2);
        assert_eq!(cache.live_len(), 2);

        // Write one more token — must land at buffer slot 2 (live_len), not
        // at slot 4 (monotonic offset). After the write `offset` becomes 5
        // and `live_len` becomes 3.
        let new_k = ffi::from_slice_f32(&[9.0], &[1, 1, 1, 1]);
        let new_v = ffi::from_slice_f32(&[10.0], &[1, 1, 1, 1]);
        cache.update(new_k, new_v);
        assert_eq!(cache.offset, 5);
        assert_eq!(cache.live_start, 2);
        assert_eq!(cache.live_len(), 3);

        let k_data = read_f32(cache.keys.as_ref().unwrap());
        let v_data = read_f32(cache.values.as_ref().unwrap());
        // Buffer slots [0, 1, 2] hold the live tokens at monotonic
        // positions [2, 3, 4] respectively.
        assert_eq!(&k_data[..3], &[3.0, 4.0, 9.0]);
        assert_eq!(&v_data[..3], &[7.0, 8.0, 10.0]);
    }

    // correctness regression test — RoPE relative-position
    // invariant under `--max-kv-size`.
    //
    // Builds two caches that should produce identical attention outputs:
    //
    //   - **Reference**: a fresh cache holding M tokens, with Q at
    //     monotonic position `M` attending over K rotated at positions
    //     `[0, M)`.
    //   - **Capped**:  a cache that wrote N + M tokens and then trimmed
    //     the oldest N. The surviving live window holds the same `M`
    //     pre-rotation inputs but rotated at monotonic positions
    //     `[N, N + M)`. Q is rotated at monotonic position `N + M`.
    //
    // Both setups see the **same relative positions** between Q and each K
    // (`[M, M-1, ..., 1]`) because RoPE is a function of `q_pos - k_pos`
    // and the absolute base `N` cancels. The attention outputs must
    // therefore match within FP16 round-off.
    //
    // The pre-fix `trim_front` decremented `cache.offset` by N, which
    // shifted Q's rotation position from `N + M` down to `M`. K vectors
    // in the buffer were still rotated at their *original* monotonic
    // positions `[N, N + M)`, so attention saw relative positions
    // `[M - N, M - N - 1, ..., 1 - N]` — wrong sign at the head of the
    // window, wrong magnitude everywhere. This test fails noisily on that
    // regression (the RMS would be well above 0.1).
    #[test]
    fn kv_cache_trim_front_preserves_rope_relative_positions_under_attention() {
        use crate as mlxcel_core;
        // Tiny shapes — keep this test cheap so it can run on CI.
        // [B, H, T, D] = [1, 1, M, head_dim]. head_dim must be even for RoPE.
        const N: i32 = 3;
        const M: i32 = 4;
        const HEAD_DIM: i32 = 8;
        let scale = 1.0_f32 / (HEAD_DIM as f32).sqrt();

        // Helper: fixed pseudo-random K/V/Q vectors for a single token at
        // the given seed. Deterministic so two callers produce identical
        // pre-rotation inputs (the load-bearing property of this test).
        fn unit_token(seed: u32) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
            // [1, 1, 1, HEAD_DIM] -- one token, one batch, one KV head.
            let k_data: Vec<f32> = (0..HEAD_DIM as u32)
                .map(|i| ((seed.wrapping_mul(7919).wrapping_add(i)) as f32 * 1.0e-3).sin())
                .collect();
            let v_data: Vec<f32> = (0..HEAD_DIM as u32)
                .map(|i| ((seed.wrapping_mul(7901).wrapping_add(i)) as f32 * 1.0e-3).cos())
                .collect();
            (
                ffi::from_slice_f32(&k_data, &[1, 1, 1, HEAD_DIM]),
                ffi::from_slice_f32(&v_data, &[1, 1, 1, HEAD_DIM]),
            )
        }

        // Apply rotation to a single-token K at `rope_pos`. Uses the same
        // `fast_rope` primitive every dense model goes through, so this
        // exercises the production path end-to-end.
        fn rotate_at(arr: &MlxArray, rope_pos: i32) -> UniquePtr<MlxArray> {
            mlxcel_core::fast_rope(arr, HEAD_DIM, false, 10_000.0, 1.0, rope_pos)
        }

        // Read an MlxArray (post-eval) to FP32 for RMS comparison.
        fn to_f32(arr: &MlxArray) -> Vec<f32> {
            let a = ffi::astype(arr, dtype::FLOAT32);
            ffi::eval(&a);
            ffi::array_to_raw_bytes(&a)
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }

        // The same `M` pre-rotation K/V inputs feed both caches. Seeds are
        // chosen so the rotated K/V is non-trivial (avoids degenerate
        // all-zero attention outputs that would mask a regression).
        let live_seeds: Vec<u32> = (0..M as u32).map(|i| 1_000 + i).collect();

        // ──────────────────────────────────────────────────────────────
        // Reference: a fresh cache holding M tokens at positions [0, M).
        let mut cache_ref = KVCache::new();
        for (i, &seed) in live_seeds.iter().enumerate() {
            let (k, v) = unit_token(seed);
            // Write at monotonic position `i`.
            let k_rot = rotate_at(&k, i as i32);
            cache_ref.update(k_rot, v);
        }
        assert_eq!(cache_ref.offset, M);
        assert_eq!(cache_ref.live_start, 0);

        // Q for the next decode step (a single fixed Q vector rotated at
        // monotonic position `M`).
        let (q_unrot, _) = unit_token(42);
        let q_ref = rotate_at(&q_unrot, M);
        let (k_ref, v_ref) = cache_ref.update_and_fetch(
            rotate_at(&unit_token(99).0, M),
            unit_token(99).1, // V is not rotated
        );
        // Recover the M live K/V (we wrote M+1 tokens; for attention parity
        // we only care that the slice consumed by attention is consistent
        // between ref and capped).
        let out_ref = mlxcel_core::causal_attention(&q_ref, &k_ref, &v_ref, scale, 0.0, 0);
        let out_ref_f32 = to_f32(&out_ref);

        // ──────────────────────────────────────────────────────────────
        // Capped: write N filler tokens at positions [0, N), then the same
        // M live tokens at positions [N, N + M), then trim the oldest N.
        let mut cache_cap = KVCache::new();
        for i in 0..N as u32 {
            // Filler seeds disjoint from `live_seeds` so the filler K/V
            // cannot accidentally line up with the trimmed window.
            let (k, v) = unit_token(50_000 + i);
            let k_rot = rotate_at(&k, i as i32);
            cache_cap.update(k_rot, v);
        }
        for (i, &seed) in live_seeds.iter().enumerate() {
            let (k, v) = unit_token(seed);
            let rope_pos = N + i as i32;
            let k_rot = rotate_at(&k, rope_pos);
            cache_cap.update(k_rot, v);
        }
        assert_eq!(cache_cap.offset, N + M);
        assert_eq!(cache_cap.live_start, 0);

        // Trim the oldest N tokens. Post-trim:
        //   offset = N + M (monotonic, unchanged — load-bearing for RoPE)
        //   live_start = N
        //   live_len = M
        // K @ buffer slot i was rotated at monotonic position `N + i`.
        assert_eq!(cache_cap.trim_front(N), N);
        assert_eq!(cache_cap.offset, N + M);
        assert_eq!(cache_cap.live_start, N);
        assert_eq!(cache_cap.live_len(), M);

        // Q for the next decode step is rotated at monotonic position
        // `N + M` (the current `cache_cap.offset`). The same Q-input as the
        // reference, but rotated at a different absolute position.
        let q_cap = rotate_at(&q_unrot, cache_cap.offset);
        // Equivalent decode-step write to keep the symmetry with the
        // reference (same M+1 tokens flow through update_and_fetch).
        let (k_cap, v_cap) = cache_cap.update_and_fetch(
            rotate_at(&unit_token(99).0, cache_cap.offset),
            unit_token(99).1,
        );
        let out_cap = mlxcel_core::causal_attention(&q_cap, &k_cap, &v_cap, scale, 0.0, 0);
        let out_cap_f32 = to_f32(&out_cap);

        // ──────────────────────────────────────────────────────────────
        // Compare attention outputs. RoPE is purely a function of
        // `q_pos - k_pos`, so the two outputs should match within FP16
        // round-off if and only if `trim_front` preserved the position
        // invariant.
        assert_eq!(
            out_ref_f32.len(),
            out_cap_f32.len(),
            "attention output shape mismatch — different live window sizes?"
        );
        let mut sq_err = 0.0_f64;
        for (a, b) in out_ref_f32.iter().zip(out_cap_f32.iter()) {
            let d = (*a as f64) - (*b as f64);
            sq_err += d * d;
        }
        let rms = (sq_err / out_ref_f32.len() as f64).sqrt();
        // `1e-3` is the tolerance the project uses for fused-kernel parity
        // tests on Apple Silicon FP16 (see e.g.
        // `delegated_fused_kernel_matches_reference_over_200_steps`).
        assert!(
            rms < 1e-3,
            "RoPE relative-position regression: trim_front shifted Q's rotation \
             position away from K's write-time position. attention RMS = {rms}; \
             expected < 1e-3. This usually means `trim_front` decremented \
             `self.offset` instead of advancing `self.live_start`."
        );
    }

    // Negative control for the regression test above: confirm that the
    // RMS check actually fires when the position invariant is broken.
    //
    // We simulate the pre-fix `trim_front` by hand: write the trimmed
    // cache exactly as before, but rotate Q at the (wrong) monotonic
    // position `M` instead of `N + M`. Stored K's are still rotated at
    // positions `[N, N + M)`. Q at `M` and K at `N + i` produce relative
    // position `M - N - i` — wrong sign at the head of the window. If
    // the RMS check ever stops catching this (e.g. tolerance loosened
    // beyond the defect signal), this test fails.
    #[test]
    fn kv_cache_trim_front_simulated_offset_decrement_breaks_rope_attention() {
        use crate as mlxcel_core;
        const N: i32 = 3;
        const M: i32 = 4;
        const HEAD_DIM: i32 = 8;
        let scale = 1.0_f32 / (HEAD_DIM as f32).sqrt();

        fn unit_token(seed: u32) -> (UniquePtr<MlxArray>, UniquePtr<MlxArray>) {
            let k_data: Vec<f32> = (0..HEAD_DIM as u32)
                .map(|i| ((seed.wrapping_mul(7919).wrapping_add(i)) as f32 * 1.0e-3).sin())
                .collect();
            let v_data: Vec<f32> = (0..HEAD_DIM as u32)
                .map(|i| ((seed.wrapping_mul(7901).wrapping_add(i)) as f32 * 1.0e-3).cos())
                .collect();
            (
                ffi::from_slice_f32(&k_data, &[1, 1, 1, HEAD_DIM]),
                ffi::from_slice_f32(&v_data, &[1, 1, 1, HEAD_DIM]),
            )
        }
        fn rotate_at(arr: &MlxArray, rope_pos: i32) -> UniquePtr<MlxArray> {
            mlxcel_core::fast_rope(arr, HEAD_DIM, false, 10_000.0, 1.0, rope_pos)
        }
        fn to_f32(arr: &MlxArray) -> Vec<f32> {
            let a = ffi::astype(arr, dtype::FLOAT32);
            ffi::eval(&a);
            ffi::array_to_raw_bytes(&a)
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }

        let live_seeds: Vec<u32> = (0..M as u32).map(|i| 1_000 + i).collect();

        // Reference: K's rotated at positions `[0, M)`, Q rotated at `M`.
        let mut cache_ref = KVCache::new();
        for (i, &seed) in live_seeds.iter().enumerate() {
            let (k, v) = unit_token(seed);
            cache_ref.update(rotate_at(&k, i as i32), v);
        }
        let (q_unrot, _) = unit_token(42);
        let q_ref = rotate_at(&q_unrot, M);
        let (k_ref, v_ref) =
            cache_ref.update_and_fetch(rotate_at(&unit_token(99).0, M), unit_token(99).1);
        let out_ref = mlxcel_core::causal_attention(&q_ref, &k_ref, &v_ref, scale, 0.0, 0);
        let out_ref_f32 = to_f32(&out_ref);

        // Simulated broken cache: K's rotated at `[N, N + M)` (correct
        // write-time positions), but Q rotated at the **wrong** position
        // `M` — this is what `cache.offset` would be after the pre-fix
        // `trim_front` decremented it from `N + M` to `M`.
        let mut cache_broken = KVCache::new();
        for i in 0..N as u32 {
            let (k, v) = unit_token(50_000 + i);
            cache_broken.update(rotate_at(&k, i as i32), v);
        }
        for (i, &seed) in live_seeds.iter().enumerate() {
            let (k, v) = unit_token(seed);
            let rope_pos = N + i as i32;
            cache_broken.update(rotate_at(&k, rope_pos), v);
        }
        // Real trim_front to get the right buffer contents (the post-fix
        // behaviour). Then deliberately rotate Q at the (wrong) monotonic
        // position `M` to simulate the pre-fix offset decrement.
        assert_eq!(cache_broken.trim_front(N), N);
        let q_broken = rotate_at(&q_unrot, M);
        let (k_broken, v_broken) =
            cache_broken.update_and_fetch(rotate_at(&unit_token(99).0, M), unit_token(99).1);
        let out_broken =
            mlxcel_core::causal_attention(&q_broken, &k_broken, &v_broken, scale, 0.0, 0);
        let out_broken_f32 = to_f32(&out_broken);

        let mut sq_err = 0.0_f64;
        for (a, b) in out_ref_f32.iter().zip(out_broken_f32.iter()) {
            let d = (*a as f64) - (*b as f64);
            sq_err += d * d;
        }
        let rms = (sq_err / out_ref_f32.len() as f64).sqrt();
        assert!(
            rms > 1e-3,
            "RoPE-defect simulator produced RMS = {rms}, which means the regression \
             test's tolerance (< 1e-3) cannot distinguish the defect signal from \
             FP16 round-off. Tighten the tolerance in \
             `kv_cache_trim_front_preserves_rope_relative_positions_under_attention` \
             or pick a more aggressive RoPE base to widen the position-vs-position \
             differential."
        );
    }

    #[test]
    fn rotating_cache_wraps_single_token_updates_to_window_size() {
        let mut cache = RotatingKVCache::new(2);
        let first = ffi::from_slice_f32(&[1.0], &[1, 1, 1, 1]);
        let second = ffi::from_slice_f32(&[2.0], &[1, 1, 1, 1]);
        let third = ffi::from_slice_f32(&[3.0], &[1, 1, 1, 1]);
        let values = |x| ffi::from_slice_f32(&[x], &[1, 1, 1, 1]);

        cache.update_and_fetch(first, values(1.0));
        cache.update_and_fetch(second, values(2.0));
        let (keys, _values) = cache.update_and_fetch(third, values(3.0));

        assert_eq!(cache.get_offset(), 3);
        assert_eq!(cache.seq_len(), 2);
        assert_eq!(ffi::array_shape(&keys), vec![1, 1, 2, 1]);
    }

    #[test]
    fn chunked_cache_trim_front_advances_visible_window() {
        let mut cache = ChunkedKVCache::new(2);
        let keys = ffi::from_slice_f32(&[1.0, 2.0, 3.0], &[1, 1, 3, 1]);
        let values = ffi::from_slice_f32(&[4.0, 5.0, 6.0], &[1, 1, 3, 1]);
        cache.update_and_fetch(keys, values);
        cache.maybe_trim_front();

        assert_eq!(cache.get_offset(), 3);
        assert_eq!(cache.get_start_position(), 1);
        assert_eq!(
            ffi::array_shape(cache.keys.as_ref().unwrap()),
            vec![1, 1, 2, 1]
        );
    }

    // --- CachePool tests ---

    /// Minimal model stub for CachePool tests. Produces N empty KVCaches.
    struct StubModel {
        num_layers: usize,
    }

    impl crate::generate::LanguageModel for StubModel {
        fn forward(
            &self,
            _input_ids: &MlxArray,
            _caches: &mut [KVCache],
            _mask: Option<&MlxArray>,
        ) -> UniquePtr<MlxArray> {
            ffi::zeros(&[1], 0)
        }

        fn make_caches(&self) -> Vec<KVCache> {
            (0..self.num_layers).map(|_| KVCache::new()).collect()
        }

        fn num_layers(&self) -> usize {
            self.num_layers
        }

        fn eos_token_ids(&self) -> Vec<i32> {
            vec![0]
        }
    }

    /// Paged sequence whose per-layer caches are dense placeholders driven by
    /// the model (mirrors the gemma3 / llama4 model-owned + paged shape). Its
    /// natural backend is `ModelOwned`, so the `allocate` pool-backed gate keeps
    /// `make_caches()` dense — exercising the dense-length mirror path of
    /// `sync_paged_state_with_dense` that #121 left in place for non-pool-backed
    /// paged sequences.
    struct ShadowDenseModel {
        num_layers: usize,
    }

    impl crate::generate::LanguageModel for ShadowDenseModel {
        fn forward(
            &self,
            _input_ids: &MlxArray,
            _caches: &mut [KVCache],
            _mask: Option<&MlxArray>,
        ) -> UniquePtr<MlxArray> {
            ffi::zeros(&[1], 0)
        }

        fn make_caches(&self) -> Vec<KVCache> {
            (0..self.num_layers).map(|_| KVCache::new()).collect()
        }

        fn num_layers(&self) -> usize {
            self.num_layers
        }

        fn eos_token_ids(&self) -> Vec<i32> {
            vec![0]
        }

        fn sequence_state_layout(&self) -> SequenceStateLayout {
            SequenceStateLayout::model_owned(self.num_layers)
        }
    }

    struct PagedModel {
        layout: PagedKvLayout,
    }

    impl crate::generate::LanguageModel for PagedModel {
        fn forward(
            &self,
            _input_ids: &MlxArray,
            _caches: &mut [KVCache],
            _mask: Option<&MlxArray>,
        ) -> UniquePtr<MlxArray> {
            ffi::zeros(&[1], 0)
        }

        fn make_caches(&self) -> Vec<KVCache> {
            Vec::new()
        }

        fn num_layers(&self) -> usize {
            self.layout.num_layers
        }

        fn eos_token_ids(&self) -> Vec<i32> {
            vec![0]
        }

        fn sequence_state_layout(&self) -> SequenceStateLayout {
            SequenceStateLayout::paged_kv_cache(self.layout.clone())
        }
    }

    #[test]
    fn cache_pool_allocate_and_release() {
        let model = StubModel { num_layers: 4 };
        let mut pool = CachePool::new(8);

        let id1 = pool.allocate(&model).expect("should allocate");
        let id2 = pool.allocate(&model).expect("should allocate");

        assert_ne!(id1, id2);
        assert_eq!(pool.active_count(), 2);
        assert_eq!(
            pool.get_mut(id1).unwrap().backend,
            SequenceStateBackend::DenseKvCache
        );
        assert_eq!(
            pool.get_mut(id2).unwrap().backend,
            SequenceStateBackend::DenseKvCache
        );

        // Each sequence should have 4 layer caches
        assert_eq!(pool.get_caches_mut(id1).unwrap().len(), 4);
        assert_eq!(pool.get_caches_mut(id2).unwrap().len(), 4);

        pool.release(id1);
        assert_eq!(pool.active_count(), 1);
        assert!(pool.get_mut(id1).is_none());
        assert!(pool.get_mut(id2).is_some());

        pool.release(id2);
        assert_eq!(pool.active_count(), 0);
    }

    #[test]
    fn cache_pool_refuses_allocation_when_full() {
        let model = StubModel { num_layers: 2 };
        let mut pool = CachePool::new(2);

        pool.allocate(&model).expect("first");
        pool.allocate(&model).expect("second");

        let result = pool.allocate(&model);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("max capacity"));
    }

    #[test]
    fn cache_pool_release_reopens_slot() {
        let model = StubModel { num_layers: 1 };
        let mut pool = CachePool::new(1);

        let id = pool.allocate(&model).expect("first");
        assert!(pool.allocate(&model).is_err());

        pool.release(id);
        assert_eq!(pool.active_count(), 0);

        // Slot should be available again
        let id2 = pool.allocate(&model).expect("after release");
        assert_ne!(id, id2); // IDs are monotonic, never reused
        assert_eq!(pool.active_count(), 1);
    }

    #[test]
    fn cache_pool_independent_state() {
        let model = StubModel { num_layers: 2 };
        let mut pool = CachePool::new(4);

        let id1 = pool.allocate(&model).unwrap();
        let id2 = pool.allocate(&model).unwrap();

        // Mutate caches for sequence 1 only
        {
            let caches = pool.get_caches_mut(id1).unwrap();
            let keys = ffi::from_slice_f32(&[1.0, 2.0], &[1, 1, 2, 1]);
            let values = ffi::from_slice_f32(&[3.0, 4.0], &[1, 1, 2, 1]);
            caches[0].update(keys, values);
        }

        // Sequence 2 caches should still be empty
        {
            let caches = pool.get_caches_mut(id2).unwrap();
            assert!(caches[0].is_empty());
            assert!(caches[1].is_empty());
        }

        // Sequence 1 cache should have data
        {
            let caches = pool.get_caches_mut(id1).unwrap();
            assert!(!caches[0].is_empty());
            assert_eq!(caches[0].seq_len(), 2);
            // Second layer still empty
            assert!(caches[1].is_empty());
        }
    }

    #[test]
    fn cache_pool_collects_batch_cache_slices() {
        let model = StubModel { num_layers: 2 };
        let mut pool = CachePool::new(4);

        let id1 = pool.allocate(&model).unwrap();
        let id2 = pool.allocate(&model).unwrap();

        let mut batch = pool.get_batch_caches_mut(&[id1, id2]).unwrap();
        assert_eq!(batch.len(), 2);
        batch[0][0].offset = 3;
        batch[1][1].offset = 5;
        drop(batch);

        assert_eq!(pool.get_caches_mut(id1).unwrap()[0].offset, 3);
        assert_eq!(pool.get_caches_mut(id2).unwrap()[1].offset, 5);
    }

    #[test]
    fn cache_pool_memory_usage() {
        let model = StubModel { num_layers: 2 };
        let mut pool = CachePool::new(4);

        // Empty pool
        assert_eq!(pool.memory_usage_bytes(), 0);

        let id1 = pool.allocate(&model).unwrap();

        // Freshly allocated caches have no data
        assert_eq!(pool.memory_usage_bytes(), 0);

        // Add some data to one cache
        {
            let caches = pool.get_caches_mut(id1).unwrap();
            let keys = ffi::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4, 1]);
            let values = ffi::from_slice_f32(&[5.0, 6.0, 7.0, 8.0], &[1, 1, 4, 1]);
            caches[0].update(keys, values);
        }

        let mem_after = pool.memory_usage_bytes();
        assert!(mem_after > 0);

        // Release should bring memory tracking back to zero
        pool.release(id1);
        assert_eq!(pool.memory_usage_bytes(), 0);
    }

    #[test]
    fn cache_pool_sequence_metadata() {
        let model = StubModel { num_layers: 1 };
        let mut pool = CachePool::new(4);

        let id = pool.allocate(&model).unwrap();
        let entry = pool.get_mut(id).unwrap();

        assert_eq!(entry.seq_id, id);
        assert_eq!(entry.prompt_len, 0);
        assert_eq!(entry.current_offset, 0);

        // Simulate prefill state update
        entry.prompt_len = 42;
        entry.current_offset = 42;

        let entry = pool.get_mut(id).unwrap();
        assert_eq!(entry.prompt_len, 42);
        assert_eq!(entry.current_offset, 42);
    }

    #[test]
    fn cache_pool_release_nonexistent_is_noop() {
        let mut pool = CachePool::new(4);
        let fake_id = SequenceId(9999);
        pool.release(fake_id); // should not panic
        assert_eq!(pool.active_count(), 0);
    }

    #[test]
    fn sequence_id_display() {
        let id = SequenceId(42);
        assert_eq!(format!("{id}"), "seq-42");
        assert_eq!(id.as_u64(), 42);
    }

    #[test]
    fn cache_pool_rejects_non_batching_model() {
        struct NonBatchModel;

        impl crate::generate::LanguageModel for NonBatchModel {
            fn forward(
                &self,
                _input_ids: &MlxArray,
                _caches: &mut [KVCache],
                _mask: Option<&MlxArray>,
            ) -> UniquePtr<MlxArray> {
                ffi::zeros(&[1], 0)
            }

            fn make_caches(&self) -> Vec<KVCache> {
                vec![KVCache::new()]
            }

            fn num_layers(&self) -> usize {
                1
            }

            fn eos_token_ids(&self) -> Vec<i32> {
                vec![0]
            }

            fn supports_batching(&self) -> bool {
                false
            }
        }

        let model = NonBatchModel;
        let mut pool = CachePool::new(8);

        // Non-batching models use lightweight placeholders — multiple
        // allocations are allowed so requests can be queued while another
        // sequence is generating.
        let first = pool.allocate(&model);
        assert!(first.is_ok());
        let first = first.unwrap();
        assert_eq!(pool.active_count(), 1);
        assert_eq!(
            pool.get_mut(first).unwrap().backend,
            SequenceStateBackend::ModelOwned
        );

        let second = pool.allocate(&model);
        assert!(second.is_ok());
        let second = second.unwrap();
        assert_eq!(pool.active_count(), 2);

        // Release both
        pool.release(first);
        pool.release(second);
        assert_eq!(pool.active_count(), 0);
    }

    #[test]
    fn paged_layout_validates_block_geometry() {
        assert!(PagedKvLayout::uniform(2, 0, 128).is_err());
        assert!(PagedKvLayout::uniform(2, 4, 130).is_err());
        assert!(PagedKvLayout::new(4, Vec::new()).is_err());
    }

    #[test]
    fn cache_pool_allocates_paged_sequence_state() {
        let layout = PagedKvLayout::uniform(2, 4, 128).unwrap();
        let model = PagedModel {
            layout: layout.clone(),
        };
        let mut pool = CachePool::new(4);

        let id = pool.allocate(&model).unwrap();
        let entry = pool.get_mut(id).unwrap();
        assert_eq!(entry.backend, SequenceStateBackend::PagedKvCache);
        assert!(entry.caches.is_empty());
        assert_eq!(entry.paged_state().unwrap().layers.len(), layout.num_layers);
        assert_eq!(
            pool.paged_stats().unwrap(),
            PagedCacheStats {
                allocated_blocks: 0,
                live_blocks: 0,
                free_blocks: 0,
                bytes_reserved: 0,
                bytes_in_use: 0,
            }
        );
    }

    #[test]
    fn cache_pool_restores_paged_sequence_state_into_allocator() {
        let layout = PagedKvLayout::uniform(2, 4, 128).unwrap();
        let model = PagedModel {
            layout: layout.clone(),
        };
        let mut pool = CachePool::new(4);
        let id = pool.allocate(&model).unwrap();

        let restored = PagedSequenceState {
            block_size: layout.block_size,
            layers: vec![
                PagedLayerState {
                    block_ids: vec![PagedBlockId::from_raw(7), PagedBlockId::from_raw(8)],
                    len: 6,
                    logical_start: 0,
                },
                PagedLayerState {
                    block_ids: vec![PagedBlockId::from_raw(42)],
                    len: 3,
                    logical_start: 0,
                },
            ],
        };

        pool.restore_paged_state(id, restored).unwrap();

        assert_eq!(
            pool.paged_stats(),
            Some(PagedCacheStats {
                allocated_blocks: 3,
                live_blocks: 3,
                free_blocks: 0,
                bytes_reserved: 0,
                bytes_in_use: 0,
            })
        );

        pool.append_paged_tokens(id, 1, 2).unwrap();
        assert_eq!(
            pool.paged_stats(),
            Some(PagedCacheStats {
                allocated_blocks: 4,
                live_blocks: 4,
                free_blocks: 0,
                bytes_reserved: 0,
                bytes_in_use: 0,
            })
        );
    }

    #[test]
    fn cache_pool_extract_and_restore_with_contents_round_trips() {
        // Drive a pool-backed (dense-natural StubModel + Fp16 paged layout)
        // sequence on an origin CachePool, transfer its pool block CONTENTS to a
        // fresh decode CachePool via extract_paged_blocks +
        // restore_paged_state_with_contents, and assert content + block
        // accounting parity (#125).
        const H: i32 = 2;
        const D: i32 = 3;
        let block_size = 4usize;
        let num_layers = 2usize;
        let n_new = 6i32; // 2 blocks per layer (block 0 full, block 1 half-full).

        let layout =
            PagedKvLayout::uniform(num_layers, block_size, block_size * (H * D) as usize * 2)
                .unwrap();
        let model = StubModel { num_layers };

        // FP32 [1, H, n, D] K/V whose values encode (layer, head, slot, dim) so
        // every layer's blocks are distinct and any mis-slotting is caught. FP32
        // is preserved through the pool (no astype), so a raw byte compare is
        // meaningful.
        let make_kv = |layer: usize, salt: f32| -> UniquePtr<MlxArray> {
            let mut vals = Vec::with_capacity((H * n_new * D) as usize);
            for head in 0..H {
                for slot in 0..n_new {
                    for dim in 0..D {
                        vals.push(
                            salt + layer as f32 * 10_000.0
                                + head as f32 * 1000.0
                                + slot as f32 * 10.0
                                + dim as f32,
                        );
                    }
                }
            }
            ffi::from_slice_f32(&vals, &[1, H, n_new, D])
        };
        let raw = |arr: &MlxArray| -> Vec<u8> {
            ffi::eval(arr);
            ffi::array_to_raw_bytes(arr)
        };

        // --- Origin pool: allocate a pool-backed paged seq + write content. ---
        let mut origin = CachePool::new(4);
        let id_a = origin
            .allocate_with_layout(
                &model,
                Some(SequenceStateLayout::paged_kv_cache(layout.clone())),
            )
            .unwrap();
        {
            let caches = origin.get_caches_mut(id_a).unwrap();
            assert!(caches.iter().all(|c| c.is_paged_backed()));
            for (layer_idx, cache) in caches.iter_mut().enumerate() {
                let k = make_kv(layer_idx, 100.0);
                let v = make_kv(layer_idx, 500.0);
                let _ = cache.update_and_fetch(k, v);
            }
        }

        let contents_a = origin.extract_paged_blocks(id_a).unwrap();
        // 2 layers * 2 blocks each = 4 transferred blocks.
        assert_eq!(contents_a.len(), 4);
        let origin_stats = origin.paged_stats();
        let origin_live = origin.paged_pool_ref().unwrap().live_block_count();
        // Runtime block table to ferry alongside the contents.
        let runtime_a = (*origin.get_paged_state(id_a).unwrap()).clone();

        // --- Decode pool: allocate a matching pool-backed seq + restore. ---
        let mut decode = CachePool::new(4);
        let id_b = decode
            .allocate_with_layout(
                &model,
                Some(SequenceStateLayout::paged_kv_cache(layout.clone())),
            )
            .unwrap();
        decode
            .restore_paged_state_with_contents(id_b, runtime_a, contents_a)
            .unwrap();

        // Block accounting after restore matches the origin (no leak, no double).
        assert_eq!(decode.paged_stats(), origin_stats);
        assert_eq!(
            decode.paged_pool_ref().unwrap().live_block_count(),
            origin_live
        );

        // Restored caches are pool-backed and their RoPE offset is the prefix len.
        {
            let caches = decode.get_caches_mut(id_b).unwrap();
            assert!(caches.iter().all(|c| c.is_paged_backed()));
            for cache in caches.iter() {
                assert_eq!(cache.offset, n_new);
            }
        }

        // Content parity: extract iterates layers/blocks in the same order on
        // both pools and the decode block table is the origin's remapped 1:1, so
        // block i is the same logical (layer, position). The bytes are identical.
        let check_a = origin.extract_paged_blocks(id_a).unwrap();
        let check_b = decode.extract_paged_blocks(id_b).unwrap();
        assert_eq!(check_a.len(), check_b.len());
        for (a, b) in check_a.iter().zip(check_b.iter()) {
            assert_eq!(a.layer_idx, b.layer_idx);
            assert_eq!(
                raw(&a.keys),
                raw(&b.keys),
                "layer {} K mismatch",
                a.layer_idx
            );
            assert_eq!(
                raw(&a.values),
                raw(&b.values),
                "layer {} V mismatch",
                a.layer_idx
            );
        }
    }

    #[test]
    fn restore_paged_state_with_contents_releases_blocks_on_missing_remap() {
        // A handoff whose block table references a block with no matching
        // transferred contents must fail the restore AND release every block the
        // acquire loop already minted, never leaking pool blocks (#125 security
        // hardening). Drive a pool-backed origin, drop one transferred block, and
        // assert the decode pool's live block count is back to zero after the
        // failed restore.
        const H: i32 = 2;
        const D: i32 = 3;
        let block_size = 4usize;
        let num_layers = 2usize;
        let n_new = 6i32; // 2 blocks per layer.

        let layout =
            PagedKvLayout::uniform(num_layers, block_size, block_size * (H * D) as usize * 2)
                .unwrap();
        let model = StubModel { num_layers };

        let make_kv = |layer: usize, salt: f32| -> UniquePtr<MlxArray> {
            let mut vals = Vec::with_capacity((H * n_new * D) as usize);
            for head in 0..H {
                for slot in 0..n_new {
                    for dim in 0..D {
                        vals.push(
                            salt + layer as f32 * 10_000.0
                                + head as f32 * 1000.0
                                + slot as f32 * 10.0
                                + dim as f32,
                        );
                    }
                }
            }
            ffi::from_slice_f32(&vals, &[1, H, n_new, D])
        };

        let mut origin = CachePool::new(4);
        let id_a = origin
            .allocate_with_layout(
                &model,
                Some(SequenceStateLayout::paged_kv_cache(layout.clone())),
            )
            .unwrap();
        {
            let caches = origin.get_caches_mut(id_a).unwrap();
            for (layer_idx, cache) in caches.iter_mut().enumerate() {
                let _ =
                    cache.update_and_fetch(make_kv(layer_idx, 100.0), make_kv(layer_idx, 500.0));
            }
        }
        let mut contents = origin.extract_paged_blocks(id_a).unwrap();
        assert_eq!(contents.len(), 4);
        let runtime = (*origin.get_paged_state(id_a).unwrap()).clone();

        // Drop the last block's contents: the block table still references it, so
        // the remap misses it AFTER the loop has already acquired the other three.
        contents.pop();

        let mut decode = CachePool::new(4);
        let id_b = decode
            .allocate_with_layout(
                &model,
                Some(SequenceStateLayout::paged_kv_cache(layout.clone())),
            )
            .unwrap();
        assert_eq!(decode.paged_pool_ref().unwrap().live_block_count(), 0);

        let result = decode.restore_paged_state_with_contents(id_b, runtime, contents);
        assert!(
            result.is_err(),
            "restore must fail when a referenced block has no transferred contents"
        );
        assert_eq!(
            decode.paged_pool_ref().unwrap().live_block_count(),
            0,
            "a failed restore must release every acquired block (no leak)"
        );
    }

    #[test]
    fn cache_pool_paged_append_trim_release_and_reuse() {
        let layout = PagedKvLayout::uniform(2, 4, 128).unwrap();
        let model = PagedModel {
            layout: layout.clone(),
        };
        let mut pool = CachePool::new(4);

        let id1 = pool.allocate(&model).unwrap();
        pool.append_paged_tokens(id1, 0, 6).unwrap();

        let (first_block, second_block) = {
            let state = pool.get_paged_state(id1).unwrap();
            let layer = state.layer(0).unwrap();
            assert_eq!(layer.len, 6);
            assert_eq!(layer.visible_len(), 6);
            assert_eq!(layer.reserved_blocks(), 2);
            (layer.block_ids[0], layer.block_ids[1])
        };
        assert_eq!(
            pool.paged_stats().unwrap(),
            PagedCacheStats {
                allocated_blocks: 2,
                live_blocks: 2,
                free_blocks: 0,
                bytes_reserved: 0,
                bytes_in_use: 0,
            }
        );
        assert_eq!(pool.memory_usage_bytes(), 192);

        assert_eq!(pool.trim_paged_tokens(id1, 0, 1).unwrap(), 1);
        assert_eq!(
            pool.paged_stats().unwrap(),
            PagedCacheStats {
                allocated_blocks: 2,
                live_blocks: 2,
                free_blocks: 0,
                bytes_reserved: 0,
                bytes_in_use: 0,
            }
        );

        assert_eq!(pool.rewind_paged_tokens(id1, 0, 2).unwrap(), 2);
        {
            let state = pool.get_paged_state(id1).unwrap();
            let layer = state.layer(0).unwrap();
            assert_eq!(layer.len, 3);
            assert_eq!(layer.block_ids, vec![first_block]);
        }
        assert_eq!(
            pool.paged_stats().unwrap(),
            PagedCacheStats {
                allocated_blocks: 2,
                live_blocks: 1,
                free_blocks: 1,
                bytes_reserved: 0,
                bytes_in_use: 0,
            }
        );

        pool.append_paged_tokens(id1, 1, 4).unwrap();
        assert_eq!(
            pool.paged_stats().unwrap(),
            PagedCacheStats {
                allocated_blocks: 3,
                live_blocks: 2,
                free_blocks: 1,
                bytes_reserved: 0,
                bytes_in_use: 0,
            }
        );

        pool.release(id1);
        assert_eq!(
            pool.paged_stats().unwrap(),
            PagedCacheStats {
                allocated_blocks: 3,
                live_blocks: 0,
                free_blocks: 3,
                bytes_reserved: 0,
                bytes_in_use: 0,
            }
        );

        let id2 = pool.allocate(&model).unwrap();
        pool.append_paged_tokens(id2, 0, 4).unwrap();
        let reused_block = pool
            .get_paged_state(id2)
            .unwrap()
            .layer(0)
            .unwrap()
            .block_ids[0];
        assert_eq!(reused_block, first_block);
        assert_ne!(reused_block, second_block);
    }

    #[test]
    fn cache_pool_can_override_dense_model_with_paged_sequence_state() {
        let model = StubModel { num_layers: 2 };
        let mut pool = CachePool::new(4);
        let layout = SequenceStateLayout::paged_kv_cache(PagedKvLayout::uniform(2, 4, 4).unwrap());

        let id = pool.allocate_with_layout(&model, Some(layout)).unwrap();
        let entry = pool.get_mut(id).unwrap();

        assert_eq!(entry.backend, SequenceStateBackend::PagedKvCache);
        assert_eq!(entry.caches.len(), 2);
        assert!(entry.paged_state().is_some());
    }

    #[test]
    fn sync_paged_state_with_dense_cache_offsets_tracks_rewinds() {
        // `ShadowDenseModel` is model-owned, so the `allocate` pool-backed gate
        // keeps its caches dense — this is the path where the dense-length
        // mirror is the authoritative length source (#121 keeps it for
        // non-pool-backed paged sequences).
        let model = ShadowDenseModel { num_layers: 1 };
        let mut pool = CachePool::new(4);
        let layout = SequenceStateLayout::paged_kv_cache(PagedKvLayout::uniform(1, 4, 4).unwrap());
        let id = pool.allocate_with_layout(&model, Some(layout)).unwrap();
        // Sanity: this sequence must NOT be pool-backed, otherwise the dense
        // mirror below would be (correctly) skipped.
        assert!(
            !pool.get_caches_mut(id).unwrap()[0].is_paged_backed(),
            "model-owned paged sequence must keep dense placeholder caches"
        );

        {
            let caches = pool.get_caches_mut(id).unwrap();
            let keys = ffi::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4, 1]);
            let values = ffi::from_slice_f32(&[5.0, 6.0, 7.0, 8.0], &[1, 1, 4, 1]);
            caches[0].update(keys, values);
        }
        pool.sync_paged_state_with_dense(id).unwrap();
        assert_eq!(pool.get_paged_state(id).unwrap().layer(0).unwrap().len, 4);

        {
            let caches = pool.get_caches_mut(id).unwrap();
            assert_eq!(caches[0].trim(2), 2);
        }
        pool.sync_paged_state_with_dense(id).unwrap();
        assert_eq!(pool.get_paged_state(id).unwrap().layer(0).unwrap().len, 2);
    }

    #[test]
    fn sync_paged_state_with_dense_skips_pool_backed_sequences() {
        // A dense-natural Fp16 model gets pool-backed paged caches (#121). The
        // pool is authoritative: `update` writes straight into it and advances
        // `state.len` in lockstep, so `sync_paged_state_with_dense` must be a
        // no-op and must NOT mirror (and thus zero out) the live block table
        // from the empty dense placeholder buffers.
        let model = StubModel { num_layers: 1 };
        let mut pool = CachePool::new(4);
        let layout = SequenceStateLayout::paged_kv_cache(PagedKvLayout::uniform(1, 4, 4).unwrap());
        let id = pool.allocate_with_layout(&model, Some(layout)).unwrap();
        assert!(
            pool.get_caches_mut(id).unwrap()[0].is_paged_backed(),
            "dense-natural Fp16 paged sequence must be pool-backed"
        );

        {
            let caches = pool.get_caches_mut(id).unwrap();
            let keys = ffi::from_slice_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4, 1]);
            let values = ffi::from_slice_f32(&[5.0, 6.0, 7.0, 8.0], &[1, 1, 4, 1]);
            // Pool-backed `update` routes into the pool and bumps `state.len`.
            caches[0].update(keys, values);
        }
        assert_eq!(pool.get_paged_state(id).unwrap().layer(0).unwrap().len, 4);

        // The sync must leave the pool-driven length untouched (skip), rather
        // than mirroring the empty dense buffers down to 0.
        pool.sync_paged_state_with_dense(id).unwrap();
        assert_eq!(pool.get_paged_state(id).unwrap().layer(0).unwrap().len, 4);
    }

    #[test]
    fn batched_decode_metadata_tracks_heterogeneous_lengths() {
        let mut cache_a = KVCache::new();
        cache_a.offset = 3;
        let mut cache_b = KVCache::new();
        cache_b.offset = 9;
        let caches = vec![&mut cache_a, &mut cache_b];

        let metadata =
            BatchedAttentionMetadata::from_kv_caches(&caches, &[1, 4], &[0, 32]).unwrap();

        assert_eq!(metadata.rope_offsets, vec![3, 9]);
        assert_eq!(metadata.query_lens, vec![1, 4]);
        assert_eq!(metadata.kv_lens, vec![4, 13]);
        assert_eq!(metadata.window_sizes, vec![0, 32]);
    }

    #[test]
    fn batched_attention_metadata_rejects_mismatched_lengths() {
        let mut cache = KVCache::new();
        let caches = vec![&mut cache];

        assert!(BatchedAttentionMetadata::from_kv_caches(&caches, &[1, 2], &[0]).is_err());
        assert!(BatchedAttentionMetadata::from_kv_caches(&caches, &[1], &[0, 1]).is_err());
    }

    #[test]
    fn paged_decode_metadata_builds_logical_block_tables() {
        let metadata = BatchedAttentionMetadata {
            rope_offsets: vec![0, 4],
            query_lens: vec![1, 1],
            kv_lens: vec![3, 5],
            window_sizes: vec![0, 0],
        };

        let paged = PagedDecodeMetadata::from_attention_metadata(&metadata, 2).unwrap();

        assert_eq!(paged.block_size, 2);
        assert_eq!(paged.kv_lens, vec![3, 5]);
        assert_eq!(paged.block_table_offsets, vec![0, 2, 5]);
        assert_eq!(paged.block_tables, vec![0, 1, 0, 1, 2]);
    }

    #[test]
    fn paged_decode_metadata_rejects_invalid_block_size() {
        let metadata = BatchedAttentionMetadata {
            rope_offsets: vec![0],
            query_lens: vec![1],
            kv_lens: vec![1],
            window_sizes: vec![0],
        };

        assert!(PagedDecodeMetadata::from_attention_metadata(&metadata, 0).is_err());
    }

    // -----------------------------------------------------------------
    // regression coverage — TurboQuant continuous-batching
    // batch-grow / batch-shrink mid-decode.
    //
    // These tests mirror upstream `mlx-vlm` PR #1110's
    // `test_turboquant.py` additions for `BatchTurboQuantKVCache.extend`
    // / `filter`, applied to mlxcel's per-row metadata struct. The
    // upstream Python crash mode (`mx.concatenate([scalar_int, mx_array])`)
    // is statically impossible in Rust because every per-row field is
    // typed `Vec<i32>`, but the *semantic* hazard — code that assumes a
    // uniform batch and silently degrades when composition changes — is
    // what these tests guard against.
    //
    // Each Rust test corresponds to one Python upstream test:
    //   - test_batch_turboquant_extend_supports_uniform_single_item_offsets
    //     -> batched_metadata_extend_uniform_single_item_offsets
    //   - test_batch_turboquant_extend_supports_empty_uniform_offsets
    //     -> batched_metadata_extend_empty_uniform_offsets
    //   - test_batch_turboquant_filter_supports_uniform_single_item_offsets
    //     -> batched_metadata_filter_uniform_single_item_offsets
    //   - test_batch_turboquant_extend_pads_shorter_uniform_batch
    //     -> batched_metadata_extend_grow_then_shrink_preserves_per_row_state
    // -----------------------------------------------------------------

    /// Two single-sequence caches with uniform offsets (the upstream
    /// "scalar fast-path" trigger condition) extend correctly into a
    /// 2-row batch with per-row offset arrays of length 2.
    #[test]
    fn batched_metadata_extend_uniform_single_item_offsets() {
        let mut cache_a = KVCache::new();
        cache_a.offset = 3;
        let mut cache_b = KVCache::new();
        cache_b.offset = 3;

        let mut first = {
            let caches = vec![&mut cache_a];
            BatchedAttentionMetadata::from_kv_caches(&caches, &[1], &[0]).unwrap()
        };
        let second = {
            let caches = vec![&mut cache_b];
            BatchedAttentionMetadata::from_kv_caches(&caches, &[1], &[0]).unwrap()
        };

        // Trigger condition: both halves are individually "uniform" (every
        // row has offset 3). Extending must produce a per-row batch of
        // size 2, mirroring upstream `assert first.offset.tolist() == [3, 3]`.
        first.extend(&second).unwrap();

        assert_eq!(first.rope_offsets, vec![3, 3]);
        assert_eq!(first.query_lens, vec![1, 1]);
        assert_eq!(first.kv_lens, vec![4, 4]);
        assert_eq!(first.window_sizes, vec![0, 0]);
        assert_eq!(first.len(), 2);
        first.assert_consistent().unwrap();
    }

    /// Two empty single-sequence metadata objects (the upstream
    /// "scalar fast-path with empty offsets" trigger condition) extend
    /// correctly into a 2-row batch.
    #[test]
    fn batched_metadata_extend_empty_uniform_offsets() {
        let mut cache_a = KVCache::new();
        cache_a.offset = 0;
        let mut cache_b = KVCache::new();
        cache_b.offset = 0;

        let mut first = {
            let caches = vec![&mut cache_a];
            BatchedAttentionMetadata::from_kv_caches(&caches, &[0], &[0]).unwrap()
        };
        let second = {
            let caches = vec![&mut cache_b];
            BatchedAttentionMetadata::from_kv_caches(&caches, &[0], &[0]).unwrap()
        };

        first.extend(&second).unwrap();

        assert_eq!(first.rope_offsets, vec![0, 0]);
        assert_eq!(first.kv_lens, vec![0, 0]);
        assert_eq!(first.len(), 2);
        first.assert_consistent().unwrap();
    }

    /// `filter_in_place` over a uniform single-item batch keeps the per-row
    /// vector shape and selects the correct row.
    #[test]
    fn batched_metadata_filter_uniform_single_item_offsets() {
        let mut cache = KVCache::new();
        cache.offset = 3;

        let mut metadata = {
            let caches = vec![&mut cache];
            BatchedAttentionMetadata::from_kv_caches(&caches, &[1], &[0]).unwrap()
        };

        // Identity filter on a single-row batch.
        metadata.filter_in_place(&[0]).unwrap();
        assert_eq!(metadata.rope_offsets, vec![3]);
        assert_eq!(metadata.kv_lens, vec![4]);
        assert_eq!(metadata.len(), 1);
        metadata.assert_consistent().unwrap();
    }

    /// follow-up: `rope_offsets` live on the monotonic RoPE
    /// position axis, but `kv_lens` must describe the visible cache window
    /// after `KVCache::trim_front` advances `live_start`.
    #[test]
    fn batched_metadata_uses_live_len_for_kv_lens_after_trim_front() {
        let mut cache = KVCache::new();
        cache.offset = 10;
        cache.live_start = 7; // visible window is 3 tokens, not 10.

        let metadata = {
            let caches = vec![&mut cache];
            BatchedAttentionMetadata::uniform_kv_caches(&caches, 1, 0).unwrap()
        };

        assert_eq!(
            metadata.rope_offsets,
            vec![10],
            "RoPE offset must remain monotonic after trim_front"
        );
        assert_eq!(
            metadata.kv_lens,
            vec![4],
            "visible KV length must be live_len + query_len, not offset + query_len"
        );
        metadata.assert_consistent().unwrap();
    }

    /// End-to-end batch grow + shrink sequence: start with one sequence,
    /// admit a second mid-decode (extend), then evict the first
    /// (filter_in_place). Per-row offsets must remain coherent throughout.
    /// This is the strongest unit-level regression.
    #[test]
    fn batched_metadata_extend_grow_then_shrink_preserves_per_row_state() {
        let mut cache_a = KVCache::new();
        cache_a.offset = 5; // ongoing decode at position 5
        let mut cache_b = KVCache::new();
        cache_b.offset = 3; // freshly prefilled at position 3

        // Round 1: only cache_a in the batch.
        let mut metadata = {
            let caches = vec![&mut cache_a];
            BatchedAttentionMetadata::uniform_kv_caches(&caches, 1, 0).unwrap()
        };
        assert_eq!(metadata.rope_offsets, vec![5]);
        assert_eq!(metadata.kv_lens, vec![6]);

        // Mid-decode admission of cache_b: extend metadata with the new
        // sequence's per-row state.
        let new_metadata = {
            let caches = vec![&mut cache_b];
            BatchedAttentionMetadata::uniform_kv_caches(&caches, 1, 0).unwrap()
        };
        metadata.extend(&new_metadata).unwrap();

        // Per-row vectors must reflect both sequences' actual offsets.
        // This is the core property: even when the second batch was
        // "uniform" by itself (one row), extension preserves per-row
        // shape rather than collapsing to a scalar fast-path.
        assert_eq!(metadata.rope_offsets, vec![5, 3]);
        assert_eq!(metadata.kv_lens, vec![6, 4]);
        assert_eq!(metadata.len(), 2);
        metadata.assert_consistent().unwrap();

        // Mid-decode eviction of cache_a: filter to keep only row 1.
        // After this, the batch is "uniform" again (single row), but
        // the per-row vector shape must still hold.
        metadata.filter_in_place(&[1]).unwrap();
        assert_eq!(metadata.rope_offsets, vec![3]);
        assert_eq!(metadata.kv_lens, vec![4]);
        assert_eq!(metadata.len(), 1);
        metadata.assert_consistent().unwrap();
    }

    /// `extend` rejects metadata with mismatched per-row vector lengths,
    /// catching any regression that re-introduces a uniformity-collapsing
    /// optimization through a public field assignment.
    #[test]
    fn batched_metadata_extend_rejects_inconsistent_input() {
        let mut left = BatchedAttentionMetadata {
            rope_offsets: vec![0],
            query_lens: vec![1],
            kv_lens: vec![1],
            window_sizes: vec![0],
        };
        let inconsistent_right = BatchedAttentionMetadata {
            rope_offsets: vec![0, 0],
            query_lens: vec![1], // length 1 instead of 2 — bug
            kv_lens: vec![1, 1],
            window_sizes: vec![0, 0],
        };

        let err = left.extend(&inconsistent_right).unwrap_err();
        assert!(
            err.contains("query_lens"),
            "extend must reject mismatched per-row vectors: {err}"
        );

        // The left side must remain unchanged on error.
        assert_eq!(left.rope_offsets, vec![0]);
        assert_eq!(left.kv_lens, vec![1]);
        left.assert_consistent().unwrap();
    }

    /// `filter_in_place` rejects out-of-range and duplicate indices and
    /// preserves the original state on error.
    #[test]
    fn batched_metadata_filter_rejects_invalid_indices() {
        let original = BatchedAttentionMetadata {
            rope_offsets: vec![1, 2, 3],
            query_lens: vec![1, 1, 1],
            kv_lens: vec![2, 3, 4],
            window_sizes: vec![0, 0, 0],
        };

        let mut m = original.clone();
        let err = m.filter_in_place(&[0, 5]).unwrap_err();
        assert!(err.contains("out of range"), "expected range error: {err}");
        assert_eq!(m, original, "filter must not partially mutate on error");

        let mut m = original.clone();
        let err = m.filter_in_place(&[0, 0]).unwrap_err();
        assert!(err.contains("duplicate"), "expected duplicate error: {err}");
        assert_eq!(m, original, "filter must not partially mutate on error");
    }

    /// `assert_consistent` fails when per-row vectors disagree on length,
    /// even though all four fields are public. This protects future
    /// callers from constructing inconsistent metadata via direct field
    /// assignment — the upstream `mlx-vlm` PR #1110 hazard pattern in
    /// disguise.
    #[test]
    fn batched_metadata_assert_consistent_catches_field_skew() {
        let m = BatchedAttentionMetadata {
            rope_offsets: vec![0, 1],
            query_lens: vec![1], // inconsistent
            kv_lens: vec![1, 2],
            window_sizes: vec![0, 0],
        };
        let err = m.assert_consistent().unwrap_err();
        assert!(
            err.contains("query_lens"),
            "expected query_lens error: {err}"
        );

        let m = BatchedAttentionMetadata {
            rope_offsets: vec![0, 1],
            query_lens: vec![1, 1],
            kv_lens: vec![1], // inconsistent
            window_sizes: vec![0, 0],
        };
        let err = m.assert_consistent().unwrap_err();
        assert!(err.contains("kv_lens"), "expected kv_lens error: {err}");

        let m = BatchedAttentionMetadata {
            rope_offsets: vec![0, 1],
            query_lens: vec![1, 1],
            kv_lens: vec![1, 2],
            window_sizes: vec![0], // inconsistent
        };
        let err = m.assert_consistent().unwrap_err();
        assert!(
            err.contains("window_sizes"),
            "expected window_sizes error: {err}"
        );
    }

    /// Integration coverage: build metadata from caches that include a
    /// `Turbo4Asym`-mode KVCache alongside an `Fp16` KVCache. Their
    /// scalar `cache.offset` semantics are identical, so the per-row
    /// metadata stays correct across modes — the property that makes
    /// the upstream Python crash impossible to reach in mlxcel's
    /// architecture.
    #[test]
    fn batched_metadata_handles_mixed_mode_caches() {
        let mut fp16_cache = KVCache::new();
        fp16_cache.offset = 7;
        let mut turbo_cache = KVCache::new_with_mode(KVCacheMode::Turbo4Asym);
        turbo_cache.offset = 11;

        let caches = vec![&mut fp16_cache, &mut turbo_cache];
        let metadata = BatchedAttentionMetadata::from_kv_caches(&caches, &[1, 1], &[0, 0]).unwrap();

        assert_eq!(metadata.rope_offsets, vec![7, 11]);
        assert_eq!(metadata.kv_lens, vec![8, 12]);
        assert_eq!(metadata.len(), 2);
        metadata.assert_consistent().unwrap();
    }
}
