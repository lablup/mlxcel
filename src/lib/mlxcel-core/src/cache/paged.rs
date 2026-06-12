use std::collections::HashMap;

use cxx::UniquePtr;

use crate::ffi;
use crate::ffi::MlxArray;

use super::KVCacheMode;

/// Opaque identifier for one physical paged-KV block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PagedBlockId(u64);

impl PagedBlockId {
    pub fn from_raw(id: u64) -> Self {
        Self(id)
    }

    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for PagedBlockId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "block-{}", self.0)
    }
}

/// Static paged-KV layout shared by every sequence in one cache pool.
///
/// The layout doubles as a contract between the scheduler and the underlying
/// [`PagedBlockPool`]: it carries the per-block byte budget for the FP16/INT8
/// main K and V buffers (`bytes_per_block`) and, when the cache mode is one of
/// the Turbo4 variants, the per-block byte budget for the packed sidecars
/// (`turbo_sidecar_bytes_per_block`). The dense `KVCache::nbytes` accounting
/// in turbo modes always reflects packed storage; see B10.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedKvLayout {
    pub num_layers: usize,
    pub block_size: usize,
    pub bytes_per_block: Vec<usize>,
    /// Cache mode under which this layout was built. Controls per-page sidecar
    /// allocation and accounting in `PagedBlockPool`.
    ///
    /// `KVCacheMode::Fp16` and `KVCacheMode::Int8` keep the historical
    /// behavior (no sidecars, `bytes_per_block` only).
    pub cache_mode: KVCacheMode,
    /// Optional per-layer sidecar byte budget. Populated only when
    /// `cache_mode` is one of the Turbo4 variants. Used by `nbytes` accounting
    /// so paged Turbo4 caches reflect packed storage instead of FP16.
    pub turbo_sidecar_bytes_per_block: Vec<usize>,
}

impl PagedKvLayout {
    pub fn new(block_size: usize, bytes_per_block: Vec<usize>) -> Result<Self, String> {
        Self::new_with_mode(block_size, bytes_per_block, KVCacheMode::Fp16, Vec::new())
    }

    /// Construct a paged-KV layout with an explicit cache mode and optional
    /// per-layer sidecar byte budget.
    ///
    /// `turbo_sidecar_bytes_per_block` is required (length == bytes_per_block.len())
    /// when `cache_mode` is one of the Turbo4 variants and forbidden otherwise.
    pub fn new_with_mode(
        block_size: usize,
        bytes_per_block: Vec<usize>,
        cache_mode: KVCacheMode,
        turbo_sidecar_bytes_per_block: Vec<usize>,
    ) -> Result<Self, String> {
        if block_size == 0 {
            return Err("PagedKvLayout: block_size must be > 0".to_string());
        }
        if bytes_per_block.is_empty() {
            return Err("PagedKvLayout: bytes_per_block must not be empty".to_string());
        }
        if let Some((layer_idx, bytes)) = bytes_per_block
            .iter()
            .copied()
            .enumerate()
            .find(|(_, bytes)| *bytes == 0 || *bytes % block_size != 0)
        {
            return Err(format!(
                "PagedKvLayout: layer {layer_idx} bytes_per_block ({bytes}) must be a positive multiple of block_size ({block_size})"
            ));
        }

        let turbo = matches!(
            cache_mode,
            KVCacheMode::Turbo4Asym | KVCacheMode::Turbo4 | KVCacheMode::Turbo4Delegated
        );
        if turbo {
            if turbo_sidecar_bytes_per_block.len() != bytes_per_block.len() {
                return Err(format!(
                    "PagedKvLayout: turbo_sidecar_bytes_per_block length {} must match bytes_per_block length {} for cache_mode {:?}",
                    turbo_sidecar_bytes_per_block.len(),
                    bytes_per_block.len(),
                    cache_mode
                ));
            }
            if let Some((layer_idx, bytes)) = turbo_sidecar_bytes_per_block
                .iter()
                .copied()
                .enumerate()
                .find(|(_, bytes)| *bytes != 0 && *bytes % block_size != 0)
            {
                return Err(format!(
                    "PagedKvLayout: layer {layer_idx} turbo_sidecar_bytes_per_block ({bytes}) must be a multiple of block_size ({block_size})"
                ));
            }
        } else if !turbo_sidecar_bytes_per_block.is_empty() {
            return Err(format!(
                "PagedKvLayout: turbo_sidecar_bytes_per_block must be empty for cache_mode {cache_mode:?}"
            ));
        }

        Ok(Self {
            num_layers: bytes_per_block.len(),
            block_size,
            bytes_per_block,
            cache_mode,
            turbo_sidecar_bytes_per_block,
        })
    }

    pub fn uniform(
        num_layers: usize,
        block_size: usize,
        bytes_per_block: usize,
    ) -> Result<Self, String> {
        if num_layers == 0 {
            return Err("PagedKvLayout: num_layers must be > 0".to_string());
        }
        Self::new(block_size, vec![bytes_per_block; num_layers])
    }

    /// Build a Turbo4-aware uniform layout. `sidecar_bytes_per_block` is
    /// applied to every layer; pass `0` for layers that should not allocate
    /// per-page sidecars.
    pub fn uniform_with_mode(
        num_layers: usize,
        block_size: usize,
        bytes_per_block: usize,
        cache_mode: KVCacheMode,
        sidecar_bytes_per_block: usize,
    ) -> Result<Self, String> {
        if num_layers == 0 {
            return Err("PagedKvLayout: num_layers must be > 0".to_string());
        }
        let turbo = matches!(
            cache_mode,
            KVCacheMode::Turbo4Asym | KVCacheMode::Turbo4 | KVCacheMode::Turbo4Delegated
        );
        let sidecar = if turbo {
            vec![sidecar_bytes_per_block; num_layers]
        } else {
            Vec::new()
        };
        Self::new_with_mode(
            block_size,
            vec![bytes_per_block; num_layers],
            cache_mode,
            sidecar,
        )
    }

    pub fn bytes_per_token(&self, layer_idx: usize) -> Option<usize> {
        self.bytes_per_block
            .get(layer_idx)
            .map(|bytes| bytes / self.block_size)
    }

    pub fn bytes_per_block_for_layer(&self, layer_idx: usize) -> Option<usize> {
        self.bytes_per_block.get(layer_idx).copied()
    }

    /// Per-layer Turbo4 sidecar byte budget. Returns `0` for `Fp16`/`Int8`
    /// modes or for layers that did not opt into sidecar storage.
    pub fn turbo_sidecar_bytes_per_block_for_layer(&self, layer_idx: usize) -> usize {
        self.turbo_sidecar_bytes_per_block
            .get(layer_idx)
            .copied()
            .unwrap_or(0)
    }

    /// Whether the underlying cache mode requires per-page Turbo4 sidecar
    /// buffers.
    pub fn is_turbo_mode(&self) -> bool {
        matches!(
            self.cache_mode,
            KVCacheMode::Turbo4Asym | KVCacheMode::Turbo4 | KVCacheMode::Turbo4Delegated
        )
    }
}

/// Per-layer logical-to-physical mapping for paged KV storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedLayerState {
    pub block_ids: Vec<PagedBlockId>,
    pub len: usize,
    pub logical_start: usize,
}

impl PagedLayerState {
    pub fn new() -> Self {
        Self {
            block_ids: Vec::new(),
            len: 0,
            logical_start: 0,
        }
    }

    pub fn visible_len(&self) -> usize {
        self.len.saturating_sub(self.logical_start)
    }

    pub fn reserved_blocks(&self) -> usize {
        self.block_ids.len()
    }

    pub fn reserved_bytes(&self, layout: &PagedKvLayout, layer_idx: usize) -> usize {
        let main = self.reserved_blocks()
            * layout
                .bytes_per_block_for_layer(layer_idx)
                .unwrap_or_default();
        let sidecar =
            self.reserved_blocks() * layout.turbo_sidecar_bytes_per_block_for_layer(layer_idx);
        main + sidecar
    }

    pub fn used_bytes(&self, layout: &PagedKvLayout, layer_idx: usize) -> usize {
        let main = self.visible_len() * layout.bytes_per_token(layer_idx).unwrap_or_default();
        // Sidecar bytes are charged per allocated block, not per token, since
        // the packed buffer is allocated whole-block at write time.
        let sidecar_bytes_per_block = layout.turbo_sidecar_bytes_per_block_for_layer(layer_idx);
        let sidecar = if sidecar_bytes_per_block > 0 {
            // Pages that wholly contain visible tokens carry their full
            // sidecar; the partially-filled tail page also carries its full
            // sidecar because it is allocated up-front.
            self.visible_len().div_ceil(layout.block_size) * sidecar_bytes_per_block
        } else {
            0
        };
        main + sidecar
    }
}

impl Default for PagedLayerState {
    fn default() -> Self {
        Self::new()
    }
}

/// Per-sequence paged cache state spanning all transformer layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagedSequenceState {
    pub block_size: usize,
    pub layers: Vec<PagedLayerState>,
}

impl PagedSequenceState {
    pub fn new(layout: &PagedKvLayout) -> Self {
        Self {
            block_size: layout.block_size,
            layers: vec![PagedLayerState::default(); layout.num_layers],
        }
    }

    pub fn layer(&self, layer_idx: usize) -> Option<&PagedLayerState> {
        self.layers.get(layer_idx)
    }

    pub fn layer_mut(&mut self, layer_idx: usize) -> Option<&mut PagedLayerState> {
        self.layers.get_mut(layer_idx)
    }

    pub fn reserved_blocks(&self) -> usize {
        self.layers
            .iter()
            .map(PagedLayerState::reserved_blocks)
            .sum()
    }

    pub fn reserved_bytes(&self, layout: &PagedKvLayout) -> usize {
        self.layers
            .iter()
            .enumerate()
            .map(|(layer_idx, layer)| layer.reserved_bytes(layout, layer_idx))
            .sum()
    }

    pub fn used_bytes(&self, layout: &PagedKvLayout) -> usize {
        self.layers
            .iter()
            .enumerate()
            .map(|(layer_idx, layer)| layer.used_bytes(layout, layer_idx))
            .sum()
    }
}

/// Book-keeping for a single physical block owned by [`PagedBlockPool`].
///
/// `refcount` counts every live logical reference to the block: each
/// `PagedSequenceState::block_ids` entry and each detached cache handle that
/// retains the block contributes `1`. A block is considered free when
/// `refcount == 0` and is not eligible for recycling as long as `refcount > 0`.
///
/// `in_use` is retained as a derived mirror of `refcount > 0` so the existing
/// `stats` and internal debug assertions stay backwards compatible with call
/// sites that inspected the flag directly before refcounts were introduced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PagedBlockRecord {
    layer_idx: usize,
    refcount: u32,
}

impl PagedBlockRecord {
    fn new(layer_idx: usize) -> Self {
        Self {
            layer_idx,
            refcount: 1,
        }
    }

    fn is_in_use(&self) -> bool {
        self.refcount > 0
    }
}

/// Aggregated allocator/storage counters for paged KV state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PagedCacheStats {
    pub allocated_blocks: usize,
    pub live_blocks: usize,
    pub free_blocks: usize,
    pub bytes_reserved: usize,
    pub bytes_in_use: usize,
}

/// Per-page Turbo4 sidecar storage. Each field is keyed by [`PagedBlockId`]
/// and lazily populated when a sequence using a Turbo4 cache mode allocates
/// the page. Pages allocated for `Fp16`/`Int8` sequences leave every entry
/// `None`, preserving bit-identical accounting for the legacy paths.
///
/// The fields mirror the dense [`super::detach::DetachedKVCache`] sidecar
/// surface so a sequence can round-trip through detach/adopt without losing
/// any quantization state.
#[derive(Default)]
pub(crate) struct PagedTurboPageSidecars {
    pub v_packed: HashMap<PagedBlockId, UniquePtr<MlxArray>>,
    pub v_norms: HashMap<PagedBlockId, UniquePtr<MlxArray>>,
    pub k_packed: HashMap<PagedBlockId, UniquePtr<MlxArray>>,
    pub k_norms: HashMap<PagedBlockId, UniquePtr<MlxArray>>,
    pub cold_keys: HashMap<PagedBlockId, UniquePtr<MlxArray>>,
}

impl std::fmt::Debug for PagedTurboPageSidecars {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PagedTurboPageSidecars")
            .field("v_packed_pages", &self.v_packed.len())
            .field("v_norms_pages", &self.v_norms.len())
            .field("k_packed_pages", &self.k_packed.len())
            .field("k_norms_pages", &self.k_norms.len())
            .field("cold_keys_pages", &self.cold_keys.len())
            .finish()
    }
}

impl PagedTurboPageSidecars {
    /// Total bytes of every MLX sidecar tensor currently held.
    pub fn nbytes(&self) -> usize {
        let sum_map = |m: &HashMap<PagedBlockId, UniquePtr<MlxArray>>| -> usize {
            m.values().map(|a| ffi::array_nbytes(a)).sum()
        };
        sum_map(&self.v_packed)
            + sum_map(&self.v_norms)
            + sum_map(&self.k_packed)
            + sum_map(&self.k_norms)
            + sum_map(&self.cold_keys)
    }

    /// Drop all sidecar tensors for a specific block id.
    pub fn forget(&mut self, block_id: PagedBlockId) {
        self.v_packed.remove(&block_id);
        self.v_norms.remove(&block_id);
        self.k_packed.remove(&block_id);
        self.k_norms.remove(&block_id);
        self.cold_keys.remove(&block_id);
    }
}

/// Inferred per-layer geometry of a physical main-K/V pool tensor.
///
/// `PagedKvLayout` only carries the per-block byte budget; the head count,
/// head dim, and element dtype are not known until the first block is written.
/// They are captured here on the first [`PagedBlockPool::write_block`] for the
/// layer and then validated against on every subsequent write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PagedPoolMeta {
    n_kv_heads: i32,
    head_dim: i32,
    /// MLX dtype code of the stored K/V (e.g. `dtype::FLOAT16`, `dtype::INT8`).
    dtype: i32,
    /// Number of physical block rows the current pool tensor can hold along
    /// axis 0. Grown in chunks as new rows are assigned.
    capacity_blocks: usize,
}

/// Physical block allocator shared across every active paged sequence.
///
/// The pool can own the physical main K and V tensors (layout A from
/// ADR 0001: each block occupies `[block_size, n_kv_heads, head_dim]` and a
/// layer's pool tensor is `[capacity_blocks, block_size, n_kv_heads, head_dim]`,
/// separate tensors for K and V per layer). Storage is lazily allocated on the
/// first [`PagedBlockPool::write_block`] for a layer, so an `Fp16`/`Int8`
/// sequence that never writes (the current live flow, where dense
/// [`super::KVCache`] placeholders still carry live K/V) adds zero tensor or
/// row overhead. Removal of the dense placeholders from the live flow lands
/// with the decode-read (#119) / prefill-write (#120) reader/writer switch;
/// until then the pool storage added here is exercised only by unit tests.
///
/// In Turbo4 modes the pool also owns per-page sidecar MLX arrays (`v_packed`,
/// `v_norms`, optionally `k_packed`, `k_norms`, `cold_keys`) so packed
/// quantization state survives across detach/adopt round-trips and
/// prefix-cache handoffs. INT8 quantization scales stay in the dense
/// `DetachedKVCache` path; only the INT8 main K/V int-typed array flows
/// through the pool.
pub struct PagedBlockPool {
    layout: PagedKvLayout,
    next_block_id: u64,
    blocks: HashMap<PagedBlockId, PagedBlockRecord>,
    free_lists: Vec<Vec<PagedBlockId>>,
    /// Per-page Turbo4 sidecar tensors, keyed by `PagedBlockId`. Empty for
    /// `Fp16`/`Int8` cache modes; populated by Turbo4 sequences via
    /// [`PagedBlockPool::install_turbo_sidecar`].
    turbo_sidecars: PagedTurboPageSidecars,
    /// Per-layer physical K pool tensor (layout A). `None` until the first
    /// write to that layer lazily allocates it.
    pool_k: Vec<Option<UniquePtr<MlxArray>>>,
    /// Per-layer physical V pool tensor (layout A). `None` until first write.
    pool_v: Vec<Option<UniquePtr<MlxArray>>>,
    /// Per-layer inferred geometry/capacity of the pool tensors. `None` until
    /// first write.
    pool_meta: Vec<Option<PagedPoolMeta>>,
    /// Per-layer `block_id -> physical row` map keeping the pool tensors
    /// compact. Global `PagedBlockId`s remain unique across layers; rows are
    /// layer-local and assigned lazily on first write to a block.
    block_rows: Vec<HashMap<PagedBlockId, usize>>,
    /// Per-layer free row list. A row is pushed here when its block hits
    /// refcount 0 and is reused by the next first-write to a fresh block.
    free_rows: Vec<Vec<usize>>,
    /// Per-layer monotonic next-row cursor for rows never yet handed out.
    next_row: Vec<usize>,
    /// Optional cap on the total number of distinct physical blocks the pool
    /// may allocate (summed across all layers). `None` = unbounded (the pool
    /// lazily grows on demand, the historical behaviour). When `Some(max)`,
    /// [`PagedBlockPool::acquire_block`] refuses to mint a NEW block once
    /// `blocks.len() >= max` (reusing a freed block is always allowed, since it
    /// adds no memory). This is the global KV block budget #122 admits and
    /// evicts against; the scheduler sets it from the configured / estimated KV
    /// byte budget divided by the per-block byte size.
    block_budget: Option<usize>,
    /// Number of [`Self::grow_pool`] reallocations that actually grew a slab
    /// (across all layers). Each one copies an entire layer tensor, so this is
    /// the observable cost #224's presize exists to avoid; tests pin it and
    /// observability can surface it.
    grow_events: u64,
}

/// Number of physical block rows the pool tensor grows by when a freshly
/// assigned row exceeds the current capacity. Mirrors the chunked-growth
/// discipline of the dense `KVCache` (which pre-allocates in steps) so an
/// append is O(block) amortised rather than O(rows) on every write.
const POOL_GROW_CHUNK_BLOCKS: usize = 32;

/// Gathered visible K and V tensors for one layer, each shaped
/// `[1, n_kv_heads, visible_len, head_dim]` (SDPA-ready). Returned by
/// [`PagedBlockPool::gather_visible`].
pub type GatheredKv = (UniquePtr<MlxArray>, UniquePtr<MlxArray>);

/// A normalized layout-A slot update (`[1, n_slots, n_kv_heads, head_dim]`)
/// paired with its `(n_slots, n_kv_heads, head_dim)` geometry.
type NormalizedBlock = (UniquePtr<MlxArray>, (i32, i32, i32));

impl PagedBlockPool {
    pub fn new(layout: PagedKvLayout) -> Self {
        let num_layers = layout.num_layers;
        Self {
            free_lists: vec![Vec::new(); num_layers],
            layout,
            next_block_id: 0,
            blocks: HashMap::new(),
            turbo_sidecars: PagedTurboPageSidecars::default(),
            pool_k: (0..num_layers).map(|_| None).collect(),
            pool_v: (0..num_layers).map(|_| None).collect(),
            pool_meta: vec![None; num_layers],
            block_rows: vec![HashMap::new(); num_layers],
            free_rows: vec![Vec::new(); num_layers],
            next_row: vec![0; num_layers],
            block_budget: None,
            grow_events: 0,
        }
    }

    /// Number of slab-copy pool growths since construction (see
    /// [`Self::grow_pool`]). A long prefill should presize instead of
    /// accumulating these.
    pub fn pool_grow_events(&self) -> u64 {
        self.grow_events
    }

    pub fn layout(&self) -> &PagedKvLayout {
        &self.layout
    }

    /// Total distinct physical blocks the pool has ever allocated (live +
    /// freed-but-retained rows). This is the figure the [`block_budget`] caps
    /// and the proxy for the pool tensors' peak row count.
    ///
    /// [`block_budget`]: PagedBlockPool::block_budget
    pub fn allocated_block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Number of blocks currently held by at least one sequence / detached set
    /// (refcount > 0).
    pub fn live_block_count(&self) -> usize {
        self.blocks.values().filter(|r| r.is_in_use()).count()
    }

    /// The current global block budget, or `None` when unbounded.
    pub fn block_budget(&self) -> Option<usize> {
        self.block_budget
    }

    /// Set (or clear, with `None`) the global cap on distinct physical blocks.
    ///
    /// Opt-in: the default is `None` (unbounded, the historical lazy-grow
    /// behaviour). When set below the number already allocated the pool does not
    /// shrink — it simply refuses to mint further new blocks until eviction
    /// brings `allocated_block_count` back under the cap (freed rows are reused
    /// without minting). The scheduler owns deriving the value from the KV byte
    /// budget and reacting to the resulting allocation failures.
    pub fn set_block_budget(&mut self, max_blocks: Option<usize>) {
        self.block_budget = max_blocks;
    }

    /// Physical blocks still **acquirable** before the budget is hit, or `None`
    /// when unbounded. This is `budget − live_block_count`: a fresh
    /// `acquire_block` succeeds whenever fewer than `budget` blocks are live,
    /// because it can either reuse a freed (allocated-but-not-live) block or
    /// mint a new one while `allocated < budget`. Eviction / preemption that
    /// drops a block's refcount to 0 therefore *raises* this figure even though
    /// `allocated_block_count` is unchanged (the freed row is retained for
    /// reuse). The scheduler gates prefill admission on this value and reclaims
    /// (evict cold prefixes, then preempt) until it covers the sequence's need.
    /// `Some(0)` means every budgeted block is in use — admission must reclaim
    /// or defer.
    pub fn free_block_budget(&self) -> Option<usize> {
        self.block_budget
            .map(|max| max.saturating_sub(self.live_block_count()))
    }

    /// Whether the pool's cache mode requires Turbo4 sidecar storage.
    ///
    /// Used by: paged detach/adopt to decide whether to round-trip sidecars,
    /// nbytes accounting, and unit tests asserting the FP16/INT8 dispatch
    /// stays bit-identical.
    pub fn is_turbo_mode(&self) -> bool {
        self.layout.is_turbo_mode()
    }

    pub fn append_tokens(
        &mut self,
        state: &mut PagedSequenceState,
        layer_idx: usize,
        token_count: usize,
    ) -> Result<(), String> {
        let num_layers = state.layers.len();
        let layer = state.layer_mut(layer_idx).ok_or_else(|| {
            format!(
                "PagedBlockPool: layer {layer_idx} out of range for {} layers",
                num_layers
            )
        })?;
        if token_count == 0 {
            return Ok(());
        }

        // `block_ids` is indexed by ABSOLUTE position (`abs / block_size`),
        // the convention `write_prefill` and `gather_visible` use, so the
        // block count must cover the absolute length. With
        // `logical_start == 0` this is identical to the old visible-length
        // sizing; with `logical_start > 0` (a sliding-window owner,
        // issue #196) visible-length sizing under-allocated and the next
        // write indexed past the table.
        let new_len = layer.len + token_count;
        let required_blocks = new_len.div_ceil(self.layout.block_size);
        while layer.block_ids.len() < required_blocks {
            layer.block_ids.push(self.acquire_block(layer_idx)?);
        }
        layer.len = new_len;
        Ok(())
    }

    pub fn trim_tokens(
        &mut self,
        state: &mut PagedSequenceState,
        layer_idx: usize,
        token_count: usize,
    ) -> Result<usize, String> {
        let num_layers = state.layers.len();
        let layer = state.layer_mut(layer_idx).ok_or_else(|| {
            format!(
                "PagedBlockPool: layer {layer_idx} out of range for {} layers",
                num_layers
            )
        })?;
        if token_count == 0 || layer.len == 0 {
            return Ok(0);
        }

        let min_len = layer.logical_start.min(layer.len);
        let trimmed = token_count.min(layer.len - min_len);
        if trimmed == 0 {
            return Ok(0);
        }

        layer.len -= trimmed;
        if layer.len == layer.logical_start {
            // The back-trim consumed the whole visible window: the layer is
            // logically empty, so reset to the origin and release everything
            // (the old code only reset `logical_start`, which resurrected the
            // slid-away prefix as visible, issue #196).
            layer.len = 0;
            layer.logical_start = 0;
        }

        // `block_ids` is indexed by ABSOLUTE position; release only the tail
        // blocks past the absolute length. Sizing by visible length here
        // released tail blocks that still held visible tokens once
        // `logical_start` crossed a block boundary (issue #196). Head blocks
        // wholly before `logical_start` stay allocated; reclaiming them needs
        // a block-base offset and is deferred until a sliding-window paged
        // path exists.
        let required_blocks = layer.len.div_ceil(self.layout.block_size);
        while layer.block_ids.len() > required_blocks {
            if let Some(block_id) = layer.block_ids.pop() {
                self.release_block(block_id)?;
            }
        }
        Ok(trimmed)
    }

    pub fn rewind_tokens(
        &mut self,
        state: &mut PagedSequenceState,
        layer_idx: usize,
        token_count: usize,
    ) -> Result<usize, String> {
        self.trim_tokens(state, layer_idx, token_count)
    }

    pub fn release_sequence(&mut self, state: &mut PagedSequenceState) -> Result<(), String> {
        for layer in &mut state.layers {
            while let Some(block_id) = layer.block_ids.pop() {
                self.release_block(block_id)?;
            }
            layer.len = 0;
            layer.logical_start = 0;
        }
        Ok(())
    }

    /// Register an externally restored paged sequence state with this pool.
    ///
    /// Used by: distributed cache deserialization on decode nodes.
    pub fn restore_sequence(&mut self, state: &PagedSequenceState) -> Result<(), String> {
        if state.block_size != self.layout.block_size {
            return Err(format!(
                "PagedBlockPool: restored block size {} does not match pool block size {}",
                state.block_size, self.layout.block_size
            ));
        }
        if state.layers.len() != self.layout.num_layers {
            return Err(format!(
                "PagedBlockPool: restored layer count {} does not match pool layer count {}",
                state.layers.len(),
                self.layout.num_layers
            ));
        }

        for (layer_idx, layer) in state.layers.iter().enumerate() {
            // Absolute-position indexing: the block table must cover the full
            // absolute length, not just the visible window (issue #196).
            let required_blocks = layer.len.div_ceil(self.layout.block_size);
            if layer.block_ids.len() < required_blocks {
                return Err(format!(
                    "PagedBlockPool: restored layer {layer_idx} has {} blocks for length {}, requires at least {}",
                    layer.block_ids.len(),
                    layer.len,
                    required_blocks
                ));
            }

            for &block_id in &layer.block_ids {
                self.restore_block(block_id, layer_idx)?;
            }
        }
        Ok(())
    }

    pub fn stats(&self) -> PagedCacheStats {
        let allocated_blocks = self.blocks.len();
        let free_blocks = self
            .blocks
            .values()
            .filter(|record| !record.is_in_use())
            .count();
        let live_blocks = allocated_blocks.saturating_sub(free_blocks);
        // Real pool memory, not the layout's nominal scheduling placeholder
        // (#226): reserved = every allocated slab byte (capacity, including
        // grow slack); in use = bytes of the rows mapped to a live block,
        // which covers ACTIVE sequences and PARKED prompt-cache pins alike
        // (release_block unmaps a row at refcount 0). The per-sequence
        // nominal sums these replaced systematically under-reported
        // (32 B/block) and missed parked retention entirely.
        let bytes_reserved = self.pool_tensor_bytes();
        let bytes_in_use = (0..self.block_rows.len())
            .map(|layer_idx| {
                let per_block = self.real_block_bytes(layer_idx);
                // A layer with mapped rows must have written geometry; an
                // unknown dtype silently zeroing its bytes would be a future
                // regression, so surface it in debug builds.
                debug_assert!(
                    per_block.is_some() || self.block_rows[layer_idx].is_empty(),
                    "layer {layer_idx} has mapped rows but no real block size"
                );
                self.block_rows[layer_idx].len() * per_block.unwrap_or_default()
            })
            .sum();

        PagedCacheStats {
            allocated_blocks,
            live_blocks,
            free_blocks,
            bytes_reserved,
            bytes_in_use,
        }
    }

    /// REAL bytes one physical block of `layer_idx` occupies in the pool's K
    /// and V slabs combined: `block_size x n_kv_heads x head_dim x
    /// element_size x 2`. `None` until the layer's first write captures its
    /// geometry (or for an unknown dtype). This is the actual memory cost a
    /// pinned block imposes, unlike the layout's nominal `bytes_per_block`
    /// scheduling placeholder (#226).
    pub fn real_block_bytes(&self, layer_idx: usize) -> Option<usize> {
        let meta = self.pool_meta.get(layer_idx).copied().flatten()?;
        let esize = crate::dtype::size_bytes(meta.dtype)?;
        Some(self.layout.block_size * meta.n_kv_heads as usize * meta.head_dim as usize * esize * 2)
    }

    /// Current refcount of `block_id`, or `0` if the block is free or unknown.
    ///
    /// Used by: paged detach/adopt invariants, diagnostics.
    pub fn refcount(&self, block_id: PagedBlockId) -> u32 {
        self.blocks
            .get(&block_id)
            .map(|record| record.refcount)
            .unwrap_or(0)
    }

    /// Increment the refcount on an existing block so additional logical owners
    /// (e.g. a detached cache handle) keep the underlying block alive.
    ///
    /// Fails if `block_id` is unknown or is currently on the free list (i.e.
    /// `refcount == 0`), because pinning a freshly-reusable block would let
    /// callers observe stale contents.
    ///
    /// Used by: `DetachedPagedCacheSet` construction in `cache/paged_detach.rs`.
    pub fn retain_block(&mut self, block_id: PagedBlockId) -> Result<(), String> {
        let record = self
            .blocks
            .get_mut(&block_id)
            .ok_or_else(|| format!("PagedBlockPool: unknown block {block_id}"))?;
        if record.refcount == 0 {
            return Err(format!(
                "PagedBlockPool: cannot retain released block {block_id}"
            ));
        }
        record.refcount = record.refcount.saturating_add(1);
        Ok(())
    }

    /// Drop one logical reference to `block_id`. The block returns to the free
    /// list (and becomes eligible for recycling) only once the refcount
    /// reaches zero. Per-page Turbo4 sidecars (if any) and the block's main
    /// K/V pool row (if assigned) are released at the same moment so a recycled
    /// block can never serve stale data to a new sequence. The recycled row is
    /// overwritten before any read, so it needs no zeroing — only to become
    /// reusable.
    ///
    /// Used by: paged detach/adopt cleanup, internal sequence tear-down.
    pub fn release_block(&mut self, block_id: PagedBlockId) -> Result<(), String> {
        let layer_idx = {
            let record = self
                .blocks
                .get_mut(&block_id)
                .ok_or_else(|| format!("PagedBlockPool: unknown block {block_id}"))?;
            if record.refcount == 0 {
                return Err(format!(
                    "PagedBlockPool: block {block_id} was already released"
                ));
            }
            record.refcount -= 1;
            if record.refcount > 0 {
                return Ok(());
            }
            record.layer_idx
        };
        // Block has hit refcount 0 — drop any associated Turbo4 sidecars so
        // recycle-time aliasing cannot leak old packed contents.
        self.turbo_sidecars.forget(block_id);
        // Free the main-K/V pool row, if one was assigned, so it can be reused.
        if let Some(row) = self.block_rows[layer_idx].remove(&block_id) {
            self.free_rows[layer_idx].push(row);
        }
        self.free_lists[layer_idx].push(block_id);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Turbo4 sidecar installation API
    // -----------------------------------------------------------------------

    /// Install or replace the per-page packed-V tensor for `block_id`.
    ///
    /// Fails if the pool was not built with a Turbo4 cache mode or if
    /// `block_id` is unknown.
    pub fn install_v_packed(
        &mut self,
        block_id: PagedBlockId,
        v_packed: UniquePtr<MlxArray>,
    ) -> Result<(), String> {
        self.assert_turbo_mode("install_v_packed")?;
        if !self.blocks.contains_key(&block_id) {
            return Err(format!("PagedBlockPool: unknown block {block_id}"));
        }
        self.turbo_sidecars.v_packed.insert(block_id, v_packed);
        Ok(())
    }

    /// Install or replace the per-page V-norms tensor for `block_id`.
    pub fn install_v_norms(
        &mut self,
        block_id: PagedBlockId,
        v_norms: UniquePtr<MlxArray>,
    ) -> Result<(), String> {
        self.assert_turbo_mode("install_v_norms")?;
        if !self.blocks.contains_key(&block_id) {
            return Err(format!("PagedBlockPool: unknown block {block_id}"));
        }
        self.turbo_sidecars.v_norms.insert(block_id, v_norms);
        Ok(())
    }

    /// Install or replace the per-page packed-K tensor for `block_id`
    /// (symmetric Turbo4 only).
    pub fn install_k_packed(
        &mut self,
        block_id: PagedBlockId,
        k_packed: UniquePtr<MlxArray>,
    ) -> Result<(), String> {
        self.assert_turbo_mode("install_k_packed")?;
        if self.layout.cache_mode != KVCacheMode::Turbo4 {
            return Err(format!(
                "PagedBlockPool::install_k_packed: cache_mode {:?} does not store packed K",
                self.layout.cache_mode
            ));
        }
        if !self.blocks.contains_key(&block_id) {
            return Err(format!("PagedBlockPool: unknown block {block_id}"));
        }
        self.turbo_sidecars.k_packed.insert(block_id, k_packed);
        Ok(())
    }

    /// Install or replace the per-page K-norms tensor for `block_id`
    /// (symmetric Turbo4 only).
    pub fn install_k_norms(
        &mut self,
        block_id: PagedBlockId,
        k_norms: UniquePtr<MlxArray>,
    ) -> Result<(), String> {
        self.assert_turbo_mode("install_k_norms")?;
        if self.layout.cache_mode != KVCacheMode::Turbo4 {
            return Err(format!(
                "PagedBlockPool::install_k_norms: cache_mode {:?} does not store K norms",
                self.layout.cache_mode
            ));
        }
        if !self.blocks.contains_key(&block_id) {
            return Err(format!("PagedBlockPool: unknown block {block_id}"));
        }
        self.turbo_sidecars.k_norms.insert(block_id, k_norms);
        Ok(())
    }

    /// Install or replace the per-page cold-K FP16 tensor for `block_id`
    /// (delegated mode only).
    pub fn install_cold_keys(
        &mut self,
        block_id: PagedBlockId,
        cold_keys: UniquePtr<MlxArray>,
    ) -> Result<(), String> {
        self.assert_turbo_mode("install_cold_keys")?;
        if self.layout.cache_mode != KVCacheMode::Turbo4Delegated {
            return Err(format!(
                "PagedBlockPool::install_cold_keys: cache_mode {:?} does not store cold keys",
                self.layout.cache_mode
            ));
        }
        if !self.blocks.contains_key(&block_id) {
            return Err(format!("PagedBlockPool: unknown block {block_id}"));
        }
        self.turbo_sidecars.cold_keys.insert(block_id, cold_keys);
        Ok(())
    }

    /// Read-only access to the per-page packed-V tensor for `block_id`.
    pub fn v_packed_for(&self, block_id: PagedBlockId) -> Option<&MlxArray> {
        self.turbo_sidecars.v_packed.get(&block_id).map(|u| &**u)
    }

    /// Read-only access to the per-page V-norms tensor for `block_id`.
    pub fn v_norms_for(&self, block_id: PagedBlockId) -> Option<&MlxArray> {
        self.turbo_sidecars.v_norms.get(&block_id).map(|u| &**u)
    }

    /// Read-only access to the per-page packed-K tensor for `block_id`.
    pub fn k_packed_for(&self, block_id: PagedBlockId) -> Option<&MlxArray> {
        self.turbo_sidecars.k_packed.get(&block_id).map(|u| &**u)
    }

    /// Read-only access to the per-page K-norms tensor for `block_id`.
    pub fn k_norms_for(&self, block_id: PagedBlockId) -> Option<&MlxArray> {
        self.turbo_sidecars.k_norms.get(&block_id).map(|u| &**u)
    }

    /// Read-only access to the per-page cold-K tensor for `block_id`.
    pub fn cold_keys_for(&self, block_id: PagedBlockId) -> Option<&MlxArray> {
        self.turbo_sidecars.cold_keys.get(&block_id).map(|u| &**u)
    }

    /// Move the per-page packed-V tensor out of the pool. Called by paged
    /// detach to transfer Turbo4 sidecar ownership into a `DetachedPagedCacheSet`.
    pub fn take_v_packed(&mut self, block_id: PagedBlockId) -> Option<UniquePtr<MlxArray>> {
        self.turbo_sidecars.v_packed.remove(&block_id)
    }

    /// Move the per-page V-norms tensor out of the pool.
    pub fn take_v_norms(&mut self, block_id: PagedBlockId) -> Option<UniquePtr<MlxArray>> {
        self.turbo_sidecars.v_norms.remove(&block_id)
    }

    /// Move the per-page packed-K tensor out of the pool.
    pub fn take_k_packed(&mut self, block_id: PagedBlockId) -> Option<UniquePtr<MlxArray>> {
        self.turbo_sidecars.k_packed.remove(&block_id)
    }

    /// Move the per-page K-norms tensor out of the pool.
    pub fn take_k_norms(&mut self, block_id: PagedBlockId) -> Option<UniquePtr<MlxArray>> {
        self.turbo_sidecars.k_norms.remove(&block_id)
    }

    /// Move the per-page cold-K tensor out of the pool.
    pub fn take_cold_keys(&mut self, block_id: PagedBlockId) -> Option<UniquePtr<MlxArray>> {
        self.turbo_sidecars.cold_keys.remove(&block_id)
    }

    /// Sum of every per-page Turbo4 sidecar tensor currently held by the pool.
    /// Used by: nbytes accounting in `CachePool::memory_usage_bytes`.
    pub fn turbo_sidecar_bytes(&self) -> usize {
        self.turbo_sidecars.nbytes()
    }

    /// Whether any per-page sidecar tensor is installed for `block_id`.
    pub fn has_turbo_sidecar(&self, block_id: PagedBlockId) -> bool {
        self.turbo_sidecars.v_packed.contains_key(&block_id)
            || self.turbo_sidecars.v_norms.contains_key(&block_id)
            || self.turbo_sidecars.k_packed.contains_key(&block_id)
            || self.turbo_sidecars.k_norms.contains_key(&block_id)
            || self.turbo_sidecars.cold_keys.contains_key(&block_id)
    }

    // -----------------------------------------------------------------------
    // Physical main K/V pool storage (layout A)
    // -----------------------------------------------------------------------

    /// Write one block's worth of K/V into the layer's physical pool tensors.
    ///
    /// Routes both `Fp16` and `Int8` main K/V through pool storage (INT8 is
    /// just an int-typed array to `slice_update`; quantization scales stay in
    /// the dense path and are not touched here). The row for `block_id` is
    /// assigned lazily on the first write (reusing a freed row when available),
    /// and the pool tensor is grown in [`POOL_GROW_CHUNK_BLOCKS`] chunks when a
    /// fresh row exceeds capacity. The write reassigns the pool tensor via
    /// `slice_update` so MLX donates the buffer (O(block), per ADR 0001's
    /// append discipline).
    ///
    /// Accepted `k_block` / `v_block` shapes (the same for both):
    /// - `[1, n_kv_heads, n_slots, head_dim]` — the SDPA-style layout the
    ///   attention path produces (primary convention), or
    /// - `[n_slots, n_kv_heads, head_dim]` — the bare layout-A row slab.
    ///
    /// `slot_start` is the first slot within the block; `slot_start + n_slots`
    /// must not exceed `block_size`. dtype/geometry are validated against the
    /// layer's `PagedPoolMeta` (captured on the first write) on every call.
    pub fn write_block(
        &mut self,
        block_id: PagedBlockId,
        layer_idx: usize,
        slot_start: usize,
        k_block: &MlxArray,
        v_block: &MlxArray,
    ) -> Result<(), String> {
        self.validate_layer(layer_idx)?;
        if !self.blocks.contains_key(&block_id) {
            return Err(format!("PagedBlockPool: unknown block {block_id}"));
        }

        // Normalize both inputs to a `[1, n_slots, n_kv_heads, head_dim]` slot
        // update matching layout A, and extract (n_slots, n_kv_heads, head_dim).
        let (k_slot, k_geom) = normalize_block_for_layout_a(k_block, "k_block")?;
        let (v_slot, v_geom) = normalize_block_for_layout_a(v_block, "v_block")?;
        if k_geom != v_geom {
            return Err(format!(
                "PagedBlockPool::write_block: K geometry {k_geom:?} does not match V geometry {v_geom:?}"
            ));
        }
        let (n_slots, n_kv_heads, head_dim) = k_geom;
        let k_dtype = ffi::array_dtype(k_block);
        let v_dtype = ffi::array_dtype(v_block);
        if k_dtype != v_dtype {
            return Err(format!(
                "PagedBlockPool::write_block: K dtype {k_dtype} does not match V dtype {v_dtype}"
            ));
        }

        let block_size = self.layout.block_size as i32;
        if n_slots <= 0 || slot_start as i32 + n_slots > block_size {
            return Err(format!(
                "PagedBlockPool::write_block: slot range [{slot_start}, {}) out of bounds for block_size {block_size}",
                slot_start as i32 + n_slots
            ));
        }

        // Capture or validate the layer's pool geometry.
        match self.pool_meta[layer_idx] {
            Some(meta) => {
                if meta.n_kv_heads != n_kv_heads
                    || meta.head_dim != head_dim
                    || meta.dtype != k_dtype
                {
                    return Err(format!(
                        "PagedBlockPool::write_block: layer {layer_idx} expects (n_kv_heads={}, head_dim={}, dtype={}); got (n_kv_heads={n_kv_heads}, head_dim={head_dim}, dtype={k_dtype})",
                        meta.n_kv_heads, meta.head_dim, meta.dtype
                    ));
                }
            }
            None => {
                let capacity = POOL_GROW_CHUNK_BLOCKS;
                let shape = [capacity as i32, block_size, n_kv_heads, head_dim];
                self.pool_k[layer_idx] = Some(ffi::zeros(&shape, k_dtype));
                self.pool_v[layer_idx] = Some(ffi::zeros(&shape, k_dtype));
                self.pool_meta[layer_idx] = Some(PagedPoolMeta {
                    n_kv_heads,
                    head_dim,
                    dtype: k_dtype,
                    capacity_blocks: capacity,
                });
            }
        }

        // Resolve (and grow for) the physical row for this block.
        let row = self.assign_row(block_id, layer_idx)?;

        // Reassign the pool tensors with the slot update so MLX donates the
        // buffer (in-place append, O(block)).
        let starts = [row as i32, slot_start as i32, 0, 0];
        let stops = [
            row as i32 + 1,
            slot_start as i32 + n_slots,
            n_kv_heads,
            head_dim,
        ];
        let old_k = self.pool_k[layer_idx]
            .take()
            .expect("pool_k allocated above");
        self.pool_k[layer_idx] = Some(ffi::slice_update(&old_k, &k_slot, &starts, &stops));
        let old_v = self.pool_v[layer_idx]
            .take()
            .expect("pool_v allocated above");
        self.pool_v[layer_idx] = Some(ffi::slice_update(&old_v, &v_slot, &starts, &stops));
        Ok(())
    }

    /// Write a whole prefill's worth of K/V for one layer into the pool.
    ///
    /// `k_prefill` / `v_prefill` are `[1, n_kv_heads, n_new_tokens, head_dim]`
    /// (the SDPA layout the attention path produces, identical to the primary
    /// convention [`write_block`] accepts). The `n_new_tokens` tokens are
    /// written starting at the sequence's current absolute tail (`layer.len`),
    /// so this also serves the shared-prefix SUFFIX case: when called on a
    /// sequence that already references a shared prefix, it appends only the new
    /// (divergent) tokens.
    ///
    /// Steps:
    /// 1. Validate shapes/dtype/geometry against the layer's pool metadata.
    /// 2. [`append_tokens`] to grow the logical length and allocate the trailing
    ///    blocks the new tokens need (these are always fresh, refcount 1).
    /// 3. For every physical block the new tokens span, slice the matching
    ///    `[.., tok_start..tok_end, ..]` window out of `k_prefill`/`v_prefill`
    ///    and [`write_block`] it at the right `slot_start`. A partial first
    ///    block (the prefix ended mid-block) and a partial last block are both
    ///    handled by the per-block slot arithmetic.
    ///
    /// ## Copy-on-write of a shared partial tail block
    ///
    /// Block sharing in this pool is *not* always block-granular: a shared
    /// prefix can end mid-block (e.g. a 6-token prefix with `block_size == 4`
    /// leaves the second block half-full), and `CachePool::detach_paged`
    /// captures that partially-filled tail block and shares it. Writing the
    /// suffix's first tokens into that block would corrupt every sibling
    /// sequence that still references it. So before writing into any spanned
    /// block whose `refcount > 1`, this method performs a real copy-on-write:
    /// it copies the block's current contents into a freshly acquired block,
    /// repoints `state.block_ids[block_index]` at the copy, and drops this
    /// sequence's reference to the shared original (leaving the sibling's
    /// reference intact). Block-aligned prefixes start the suffix on a fresh
    /// `append_tokens` block (refcount 1), so the COW check is a no-op there.
    ///
    /// The stored bytes are byte-identical to the dense prefill path: each
    /// token lands at the same absolute `[.., abs, ..]` slot a dense
    /// `slice_update` from `[0, 0, prev, 0]` would target, so a later
    /// [`gather_visible`] returns the same bytes as the equivalent dense buffer.
    pub fn write_prefill(
        &mut self,
        state: &mut PagedSequenceState,
        layer_idx: usize,
        k_prefill: &MlxArray,
        v_prefill: &MlxArray,
    ) -> Result<(), String> {
        self.validate_layer(layer_idx)?;

        // Both inputs must be the 4D SDPA layout [1, n_kv_heads, n_new, head_dim].
        let k_shape = ffi::array_shape(k_prefill);
        let v_shape = ffi::array_shape(v_prefill);
        let (n_kv_heads, n_new, head_dim) = match k_shape.as_slice() {
            [batch, n_kv_heads, n_new, head_dim] => {
                if *batch != 1 {
                    return Err(format!(
                        "PagedBlockPool::write_prefill: k_prefill batch dim must be 1, got {batch} (shape {k_shape:?})"
                    ));
                }
                (*n_kv_heads, *n_new, *head_dim)
            }
            _ => {
                return Err(format!(
                    "PagedBlockPool::write_prefill: k_prefill must be [1, n_kv_heads, n_new_tokens, head_dim], got shape {k_shape:?}"
                ));
            }
        };
        if v_shape != k_shape {
            return Err(format!(
                "PagedBlockPool::write_prefill: k_prefill shape {k_shape:?} does not match v_prefill shape {v_shape:?}"
            ));
        }
        let k_dtype = ffi::array_dtype(k_prefill);
        let v_dtype = ffi::array_dtype(v_prefill);
        if k_dtype != v_dtype {
            return Err(format!(
                "PagedBlockPool::write_prefill: K dtype {k_dtype} does not match V dtype {v_dtype}"
            ));
        }
        if n_new <= 0 {
            return Ok(());
        }

        let block_size = self.layout.block_size as i32;
        // Absolute tail before the append — the first new token lands here.
        let first_abs = {
            let layer = state.layer(layer_idx).ok_or_else(|| {
                format!(
                    "PagedBlockPool::write_prefill: layer {layer_idx} out of range for {} layers",
                    state.layers.len()
                )
            })?;
            layer.len as i32
        };
        let last_abs = first_abs + n_new; // exclusive

        // Allocate the trailing blocks the new tokens need (and bump len). The
        // existing partial tail block, if any, is left untouched by append.
        self.append_tokens(state, layer_idx, n_new as usize)?;

        // Walk every block the new tokens span and write its slice.
        let first_block = (first_abs / block_size) as usize;
        let last_block = ((last_abs - 1) / block_size) as usize;

        // Size the layer's pool once for the whole span before any block
        // write. Without this, assign_row grows the slab in
        // POOL_GROW_CHUNK_BLOCKS steps and every step reallocates and copies
        // the entire layer tensor, transiently holding each step's old+new
        // slab pair in one lazy graph on long prefills (#224).
        self.presize_for_span(
            state,
            layer_idx,
            first_block,
            last_block,
            n_kv_heads,
            head_dim,
            k_dtype,
        )?;

        for block_index in first_block..=last_block {
            let block_start_abs = block_index as i32 * block_size;
            let slot_start = (first_abs.max(block_start_abs) - block_start_abs) as usize;
            let abs_begin = first_abs.max(block_start_abs);
            let abs_end = last_abs.min(block_start_abs + block_size);
            let n_slots = abs_end - abs_begin;
            if n_slots <= 0 {
                continue;
            }
            let tok_start = abs_begin - first_abs;
            let tok_end = tok_start + n_slots;

            // Copy-on-write the target block if a sibling sequence still shares
            // it (only ever the partial prefix tail; fresh append blocks have
            // refcount 1 and skip this).
            let block_id = state.layers[layer_idx].block_ids[block_index];
            let target_id = if self.refcount(block_id) > 1 {
                let fresh = self.copy_on_write_block(layer_idx, block_id)?;
                state.layers[layer_idx].block_ids[block_index] = fresh;
                fresh
            } else {
                block_id
            };

            // Slice the [1, H, n_slots, D] window out of the prefill tensors.
            let starts = [0, 0, tok_start, 0];
            let stops = [1, n_kv_heads, tok_end, head_dim];
            let k_slice = ffi::slice(k_prefill, &starts, &stops);
            let v_slice = ffi::slice(v_prefill, &starts, &stops);

            self.write_block(target_id, layer_idx, slot_start, &k_slice, &v_slice)?;
        }
        Ok(())
    }

    /// Ensure the layer's pool tensors can hold every span block in
    /// `first_block..=last_block` of `state` without incremental growth.
    ///
    /// Counts the span blocks that still need a physical row, nets out the
    /// reusable free rows, and either allocates the pool at the right size
    /// (first write to the layer) or grows it once. [`Self::assign_row`]
    /// keeps its grow fallback for rows minted outside the span (e.g. a
    /// copy-on-write fork), so undershooting here stays correct.
    #[allow(clippy::too_many_arguments)]
    fn presize_for_span(
        &mut self,
        state: &PagedSequenceState,
        layer_idx: usize,
        first_block: usize,
        last_block: usize,
        n_kv_heads: i32,
        head_dim: i32,
        dtype: i32,
    ) -> Result<(), String> {
        let mut unassigned = 0usize;
        for block_index in first_block..=last_block {
            let block_id = state.layers[layer_idx].block_ids[block_index];
            if !self.block_rows[layer_idx].contains_key(&block_id) {
                unassigned += 1;
            }
        }
        let minted = unassigned.saturating_sub(self.free_rows[layer_idx].len());
        if minted == 0 {
            return Ok(());
        }
        let target = self.next_row[layer_idx].saturating_add(minted);
        match self.pool_meta[layer_idx] {
            Some(meta) if target <= meta.capacity_blocks => Ok(()),
            Some(_) => self.grow_pool(layer_idx, target),
            None => {
                let capacity = target
                    .div_ceil(POOL_GROW_CHUNK_BLOCKS)
                    .saturating_mul(POOL_GROW_CHUNK_BLOCKS)
                    .max(POOL_GROW_CHUNK_BLOCKS);
                let block_size = self.layout.block_size as i32;
                let shape = [capacity as i32, block_size, n_kv_heads, head_dim];
                self.pool_k[layer_idx] = Some(ffi::zeros(&shape, dtype));
                self.pool_v[layer_idx] = Some(ffi::zeros(&shape, dtype));
                self.pool_meta[layer_idx] = Some(PagedPoolMeta {
                    n_kv_heads,
                    head_dim,
                    dtype,
                    capacity_blocks: capacity,
                });
                Ok(())
            }
        }
    }

    /// Copy the current contents of `src_block_id` into a freshly acquired block
    /// on the same layer and return the new block id. Used by [`write_prefill`]
    /// to fork a shared partial tail block before mutating it (copy-on-write).
    ///
    /// The source's full `[block_size, n_kv_heads, head_dim]` K and V slabs are
    /// sliced out of the layer's pool tensors and written into the fresh block
    /// at `slot_start = 0`, so the new block is a byte-identical copy (the
    /// caller then overwrites only the divergent suffix slots). The new block
    /// starts at refcount 1; the caller is responsible for releasing the
    /// reference to the shared original.
    fn copy_on_write_block(
        &mut self,
        layer_idx: usize,
        src_block_id: PagedBlockId,
    ) -> Result<PagedBlockId, String> {
        let meta = self.pool_meta[layer_idx].ok_or_else(|| {
            format!(
                "PagedBlockPool::copy_on_write_block: layer {layer_idx} has no pool tensors to copy from"
            )
        })?;
        let src_row = *self.block_rows[layer_idx]
            .get(&src_block_id)
            .ok_or_else(|| {
                format!(
                    "PagedBlockPool::copy_on_write_block: shared block {src_block_id} on layer {layer_idx} has no pool row (was it written?)"
                )
            })? as i32;
        let block_size = self.layout.block_size as i32;

        // Slice the source row's [block_size, H, D] slab out of K and V (the
        // bare layout-A slab write_block accepts via its 3D convention).
        let slab = |pool: &MlxArray| -> UniquePtr<MlxArray> {
            let row = ffi::slice(
                pool,
                &[src_row, 0, 0, 0],
                &[src_row + 1, block_size, meta.n_kv_heads, meta.head_dim],
            );
            ffi::reshape(&row, &[block_size, meta.n_kv_heads, meta.head_dim])
        };
        let k_slab = {
            let pool_k = self.pool_k[layer_idx]
                .as_ref()
                .expect("pool_k present when pool_meta present");
            slab(pool_k)
        };
        let v_slab = {
            let pool_v = self.pool_v[layer_idx]
                .as_ref()
                .expect("pool_v present when pool_meta present");
            slab(pool_v)
        };

        let fresh = self.acquire_block(layer_idx)?;
        self.write_block(fresh, layer_idx, 0, &k_slab, &v_slab)?;
        // Drop this sequence's reference to the shared original; the sibling
        // that still owns it keeps it alive.
        self.release_block(src_block_id)?;
        Ok(fresh)
    }

    /// Read one block's full `[block_size, n_kv_heads, head_dim]` K and V slabs
    /// out of the layer's physical pool tensors, for distributed transfer (#125).
    ///
    /// Mirrors the slab-read in [`Self::copy_on_write_block`]: it resolves the
    /// block's physical row, slices the `[1, block_size, n_kv_heads, head_dim]`
    /// window out of `pool_k`/`pool_v`, and reshapes each to the bare layout-A
    /// `[block_size, n_kv_heads, head_dim]` slab that [`Self::write_block`]
    /// accepts. The WHOLE block (including any trailing padding slots) is
    /// returned, so a decode node can reconstruct a byte-identical block via
    /// [`Self::acquire_and_write_block`].
    pub fn read_block_contents(
        &self,
        block_id: PagedBlockId,
        layer_idx: usize,
    ) -> Result<(UniquePtr<MlxArray>, UniquePtr<MlxArray>), String> {
        self.validate_layer(layer_idx)?;
        let meta = self.pool_meta[layer_idx].ok_or_else(|| {
            format!(
                "PagedBlockPool::read_block_contents: layer {layer_idx} has no pool tensors to read from"
            )
        })?;
        let src_row = *self.block_rows[layer_idx].get(&block_id).ok_or_else(|| {
            format!(
                "PagedBlockPool::read_block_contents: block {block_id} on layer {layer_idx} has no pool row (never written)"
            )
        })? as i32;
        let block_size = self.layout.block_size as i32;

        // Slice the source row's [block_size, H, D] slab out of K and V (the bare
        // layout-A slab convention `write_block`/`acquire_and_write_block` accept).
        let slab = |pool: &MlxArray| -> UniquePtr<MlxArray> {
            let window = ffi::slice(
                pool,
                &[src_row, 0, 0, 0],
                &[src_row + 1, block_size, meta.n_kv_heads, meta.head_dim],
            );
            ffi::reshape(&window, &[block_size, meta.n_kv_heads, meta.head_dim])
        };
        let k_slab = {
            let pool_k = self.pool_k[layer_idx]
                .as_ref()
                .expect("pool_k present when pool_meta present");
            slab(pool_k)
        };
        let v_slab = {
            let pool_v = self.pool_v[layer_idx]
                .as_ref()
                .expect("pool_v present when pool_meta present");
            slab(pool_v)
        };
        Ok((k_slab, v_slab))
    }

    /// Acquire a fresh block on `layer_idx` and write `k_block`/`v_block` (bare
    /// layout-A `[block_size, n_kv_heads, head_dim]` slabs, as produced by
    /// [`Self::read_block_contents`]) into it at slot 0. The returned block
    /// starts at refcount 1. Used by the decode node to materialize a
    /// transferred block on a fresh physical row (#125).
    pub fn acquire_and_write_block(
        &mut self,
        layer_idx: usize,
        k_block: &MlxArray,
        v_block: &MlxArray,
    ) -> Result<PagedBlockId, String> {
        let id = self.acquire_block(layer_idx)?;
        if let Err(e) = self.write_block(id, layer_idx, 0, k_block, v_block) {
            // `acquire_block` already minted the block; if the write fails (e.g.
            // a malformed or oversized transferred slab on the #125 restore
            // path), release it so a failed write never leaks a pool block.
            let _ = self.release_block(id);
            return Err(e);
        }
        Ok(id)
    }

    /// Gather the visible K/V window for one layer of a sequence into the
    /// SDPA-ready shape `[1, n_kv_heads, visible_len, head_dim]`.
    ///
    /// Builds the physical-row index array from the layer's `block_ids` (in
    /// block-table order — works for fragmented / out-of-order rows), takes
    /// those rows out of the pool on axis 0, flattens the block dimension,
    /// slices the visible window `[logical_start, len)` (dropping any trimmed
    /// prefix and trailing padding), and transposes into the fused-SDPA layout.
    /// The result is byte-identical to the equivalent dense contiguous buffer's
    /// visible slice. Returns `Ok(None)` when the layer has no visible tokens
    /// or no pool storage yet.
    pub fn gather_visible(
        &self,
        state: &PagedSequenceState,
        layer_idx: usize,
    ) -> Result<Option<GatheredKv>, String> {
        let layer = state.layer(layer_idx).ok_or_else(|| {
            format!(
                "PagedBlockPool::gather_visible: layer {layer_idx} out of range for {} layers",
                state.layers.len()
            )
        })?;
        let visible_len = layer.visible_len();
        if visible_len == 0 || layer.block_ids.is_empty() {
            return Ok(None);
        }

        let (pool_k, pool_v) = match (
            self.pool_k.get(layer_idx).and_then(|p| p.as_ref()),
            self.pool_v.get(layer_idx).and_then(|p| p.as_ref()),
        ) {
            (Some(k), Some(v)) => (k, v),
            _ => return Ok(None),
        };
        let meta = self.pool_meta[layer_idx]
            .expect("pool_meta present whenever pool tensors are allocated");

        // Map each block id (in block-table order) to its physical row.
        let mut rows_i32 = Vec::with_capacity(layer.block_ids.len());
        for block_id in &layer.block_ids {
            let row = *self.block_rows[layer_idx].get(block_id).ok_or_else(|| {
                format!(
                    "PagedBlockPool::gather_visible: block {block_id} on layer {layer_idx} has no pool row (was it written?)"
                )
            })?;
            rows_i32.push(row as i32);
        }

        let block_size = self.layout.block_size as i32;
        let n_blocks = rows_i32.len() as i32;
        let logical_start = layer.logical_start as i32;
        let len = layer.len as i32;
        if len > n_blocks * block_size {
            return Err(format!(
                "PagedBlockPool::gather_visible: layer {layer_idx} len {len} exceeds {n_blocks} blocks * block_size {block_size}"
            ));
        }

        let idx = ffi::from_slice_i32(&rows_i32, &[n_blocks]);

        let gather = |pool: &MlxArray| -> UniquePtr<MlxArray> {
            // [n_blocks, block_size, H, D]
            let gathered = ffi::take(pool, &idx, 0);
            // [n_blocks * block_size, H, D]
            let flat = ffi::reshape(
                &gathered,
                &[n_blocks * block_size, meta.n_kv_heads, meta.head_dim],
            );
            // [visible_len, H, D] — drop trimmed prefix and trailing padding.
            let window = ffi::slice(
                &flat,
                &[logical_start, 0, 0],
                &[len, meta.n_kv_heads, meta.head_dim],
            );
            // [1, visible_len, H, D]
            let batched = ffi::reshape(
                &window,
                &[1, len - logical_start, meta.n_kv_heads, meta.head_dim],
            );
            // [1, H, visible_len, D]
            ffi::transpose_axes(&batched, &[0, 2, 1, 3])
        };

        Ok(Some((gather(pool_k), gather(pool_v))))
    }

    /// Fused paged-attention decode over the pool via the native Metal kernel
    /// (epic #116 Phase 6, #123).
    ///
    /// Strategy (B) from ADR 0001: instead of gathering each sequence's visible
    /// KV into a contiguous tensor and calling SDPA (what
    /// [`Self::gather_visible`] feeds), this builds the flattened block-table
    /// metadata and hands the pool tensors straight to
    /// [`crate::paged_attention_decode`], which reads the scattered blocks
    /// inside the attention kernel with no gather copy.
    ///
    /// `q` is `[B, Hq, 1, head_dim]`; `states[b]` is sequence b's per-layer
    /// state (`states.len()` must equal `B`). Returns `[B, Hq, 1, head_dim]` in
    /// `q`'s dtype, or `None` when the layer's pool tensors are not yet
    /// allocated or no sequence has visible tokens (caller falls back to the
    /// gather path). The output is a drop-in for
    /// [`crate::layers::paged_decode_attention_pooled_fallback`].
    pub fn paged_decode_fused(
        &self,
        q: &MlxArray,
        states: &[&PagedSequenceState],
        layer_idx: usize,
        scale: f32,
    ) -> Result<Option<UniquePtr<MlxArray>>, String> {
        let (pool_k, pool_v) = match (
            self.pool_k.get(layer_idx).and_then(|p| p.as_ref()),
            self.pool_v.get(layer_idx).and_then(|p| p.as_ref()),
        ) {
            (Some(k), Some(v)) => (k, v),
            _ => return Ok(None),
        };
        let block_size = self.layout.block_size as i32;

        // Flatten every sequence's physical pool rows (block-table order) into
        // one `rows` array, with `row_offsets[b]` the start of sequence b.
        let mut rows: Vec<i32> = Vec::new();
        let mut row_offsets: Vec<i32> = Vec::with_capacity(states.len() + 1);
        let mut logical_starts: Vec<i32> = Vec::with_capacity(states.len());
        let mut visible_lens: Vec<i32> = Vec::with_capacity(states.len());
        row_offsets.push(0);

        let mut any_visible = false;
        for state in states {
            let layer = state.layer(layer_idx).ok_or_else(|| {
                format!(
                    "PagedBlockPool::paged_decode_fused: layer {layer_idx} out of range for {} layers",
                    state.layers.len()
                )
            })?;
            if layer.visible_len() > 0 {
                any_visible = true;
            }
            // Same front/back asymmetry guard as `gather_visible`: a slid
            // sequence whose `len` outruns its retained blocks would read past
            // the block table.
            let n_blocks = layer.block_ids.len() as i32;
            if layer.len as i32 > n_blocks * block_size {
                return Err(format!(
                    "PagedBlockPool::paged_decode_fused: layer {layer_idx} len {} exceeds {n_blocks} blocks * block_size {block_size}",
                    layer.len
                ));
            }
            for block_id in &layer.block_ids {
                let row = *self.block_rows[layer_idx].get(block_id).ok_or_else(|| {
                    format!(
                        "PagedBlockPool::paged_decode_fused: block {block_id} on layer {layer_idx} has no pool row (was it written?)"
                    )
                })?;
                rows.push(row as i32);
            }
            row_offsets.push(rows.len() as i32);
            logical_starts.push(layer.logical_start as i32);
            visible_lens.push(layer.visible_len() as i32);
        }

        if !any_visible {
            return Ok(None);
        }

        let n_states = states.len() as i32;
        let rows_arr = ffi::from_slice_i32(&rows, &[rows.len() as i32]);
        let off_arr = ffi::from_slice_i32(&row_offsets, &[n_states + 1]);
        let ls_arr = ffi::from_slice_i32(&logical_starts, &[n_states]);
        let vl_arr = ffi::from_slice_i32(&visible_lens, &[n_states]);

        // The kernel reads Q and writes its output in f32 deterministically (it
        // never re-specialises by dtype). Cast Q in and the result back to Q's
        // dtype so the fused path is byte-comparable with the gather fallback.
        let q_dtype = ffi::array_dtype(q);
        let q_f32 = if q_dtype == crate::dtype::FLOAT32 {
            None
        } else {
            Some(ffi::astype(q, crate::dtype::FLOAT32))
        };
        let q_in: &MlxArray = q_f32.as_deref().unwrap_or(q);

        let out_f32 = ffi::paged_attention_decode(
            q_in, pool_k, pool_v, &rows_arr, &off_arr, &ls_arr, &vl_arr, scale,
        );
        let out = if q_dtype == crate::dtype::FLOAT32 {
            out_f32
        } else {
            ffi::astype(&out_f32, q_dtype)
        };
        Ok(Some(out))
    }

    /// Sum of every allocated physical main-K/V pool tensor (K and V, all
    /// layers). Additive to the layout-derived scheduling budgets in
    /// `reserved_bytes`/`used_bytes`, which are unchanged.
    ///
    /// Used by: `CachePool::memory_usage_bytes` to reflect the true pool
    /// footprint once the live writer is wired (#120).
    pub fn pool_tensor_bytes(&self) -> usize {
        let sum = |pools: &[Option<UniquePtr<MlxArray>>]| -> usize {
            pools
                .iter()
                .filter_map(|p| p.as_ref())
                .map(|a| ffi::array_nbytes(a))
                .sum()
        };
        sum(&self.pool_k) + sum(&self.pool_v)
    }

    /// Resolve the physical pool row for `block_id` on `layer_idx`, assigning
    /// one lazily on first write (reusing a freed row when available) and
    /// growing the pool tensors if the new row exceeds capacity.
    fn assign_row(&mut self, block_id: PagedBlockId, layer_idx: usize) -> Result<usize, String> {
        if let Some(&row) = self.block_rows[layer_idx].get(&block_id) {
            return Ok(row);
        }
        let row = match self.free_rows[layer_idx].pop() {
            Some(row) => row,
            None => {
                let row = self.next_row[layer_idx];
                self.next_row[layer_idx] += 1;
                row
            }
        };
        if row
            >= self.pool_meta[layer_idx]
                .map(|m| m.capacity_blocks)
                .unwrap_or(0)
        {
            self.grow_pool(layer_idx, row + 1)?;
        }
        self.block_rows[layer_idx].insert(block_id, row);
        Ok(row)
    }

    /// Grow the layer's K and V pool tensors so they hold at least
    /// `min_capacity` rows, rounding up to the next [`POOL_GROW_CHUNK_BLOCKS`]
    /// multiple. Existing rows are copied into the larger buffer via
    /// `slice_update`.
    fn grow_pool(&mut self, layer_idx: usize, min_capacity: usize) -> Result<(), String> {
        let meta = self.pool_meta[layer_idx].ok_or_else(|| {
            format!("PagedBlockPool::grow_pool: layer {layer_idx} has no pool tensors")
        })?;
        if min_capacity <= meta.capacity_blocks {
            return Ok(());
        }
        let new_capacity = min_capacity
            .div_ceil(POOL_GROW_CHUNK_BLOCKS)
            .saturating_mul(POOL_GROW_CHUNK_BLOCKS)
            .max(POOL_GROW_CHUNK_BLOCKS);
        let block_size = self.layout.block_size as i32;
        let new_shape = [
            new_capacity as i32,
            block_size,
            meta.n_kv_heads,
            meta.head_dim,
        ];
        let copy_stops = [
            meta.capacity_blocks as i32,
            block_size,
            meta.n_kv_heads,
            meta.head_dim,
        ];
        let starts = [0, 0, 0, 0];

        let old_k = self.pool_k[layer_idx]
            .take()
            .expect("pool_k present when meta present");
        let mut new_k = ffi::zeros(&new_shape, meta.dtype);
        new_k = ffi::slice_update(&new_k, &old_k, &starts, &copy_stops);
        self.pool_k[layer_idx] = Some(new_k);

        let old_v = self.pool_v[layer_idx]
            .take()
            .expect("pool_v present when meta present");
        let mut new_v = ffi::zeros(&new_shape, meta.dtype);
        new_v = ffi::slice_update(&new_v, &old_v, &starts, &copy_stops);
        self.pool_v[layer_idx] = Some(new_v);

        self.pool_meta[layer_idx] = Some(PagedPoolMeta {
            capacity_blocks: new_capacity,
            ..meta
        });
        self.grow_events += 1;
        // Materialize the grown copies immediately. The old slabs then return
        // to the MLX buffer cache before the next layer grows, so a
        // multi-layer growth episode transiently holds one layer's old+new
        // pair instead of accumulating every layer's pair in a single lazy
        // graph (#224).
        ffi::eval(
            self.pool_k[layer_idx]
                .as_ref()
                .expect("pool_k present after grow"),
        );
        ffi::eval(
            self.pool_v[layer_idx]
                .as_ref()
                .expect("pool_v present after grow"),
        );
        Ok(())
    }

    fn assert_turbo_mode(&self, op: &str) -> Result<(), String> {
        if !self.layout.is_turbo_mode() {
            return Err(format!(
                "PagedBlockPool::{op}: pool cache_mode {:?} is not a Turbo4 variant",
                self.layout.cache_mode
            ));
        }
        Ok(())
    }

    fn acquire_block(&mut self, layer_idx: usize) -> Result<PagedBlockId, String> {
        self.validate_layer(layer_idx)?;
        if let Some(block_id) = self.free_lists[layer_idx].pop() {
            let record = self
                .blocks
                .get_mut(&block_id)
                .expect("free-list block must exist in registry");
            debug_assert_eq!(
                record.refcount, 0,
                "free-list block must have zero refcount"
            );
            record.refcount = 1;
            return Ok(block_id);
        }

        // No freed block to reuse — minting a new one grows the pool, so it is
        // subject to the global block budget (opt-in; `None` = unbounded).
        if let Some(max) = self.block_budget {
            if self.blocks.len() >= max {
                return Err(format!(
                    "PagedBlockPool: block budget exhausted ({} of {max} blocks allocated)",
                    self.blocks.len()
                ));
            }
        }

        let block_id = PagedBlockId(self.next_block_id);
        self.next_block_id += 1;
        self.blocks
            .insert(block_id, PagedBlockRecord::new(layer_idx));
        Ok(block_id)
    }

    fn restore_block(&mut self, block_id: PagedBlockId, layer_idx: usize) -> Result<(), String> {
        self.validate_layer(layer_idx)?;

        if let Some(record) = self.blocks.get_mut(&block_id) {
            if record.layer_idx != layer_idx {
                return Err(format!(
                    "PagedBlockPool: restored block {block_id} belongs to layer {}, not {}",
                    record.layer_idx, layer_idx
                ));
            }
            if record.is_in_use() {
                return Err(format!(
                    "PagedBlockPool: restored block {block_id} is already marked in use"
                ));
            }
            if let Some(pos) = self.free_lists[layer_idx]
                .iter()
                .position(|candidate| *candidate == block_id)
            {
                self.free_lists[layer_idx].swap_remove(pos);
            }
            record.refcount = 1;
        } else {
            self.blocks
                .insert(block_id, PagedBlockRecord::new(layer_idx));
        }

        self.next_block_id = self.next_block_id.max(block_id.as_u64().saturating_add(1));
        Ok(())
    }

    fn validate_layer(&self, layer_idx: usize) -> Result<(), String> {
        if layer_idx >= self.layout.num_layers {
            return Err(format!(
                "PagedBlockPool: layer {layer_idx} out of range for {} layers",
                self.layout.num_layers
            ));
        }
        Ok(())
    }
}

/// Normalize a write block into the layout-A slot update shape
/// `[1, n_slots, n_kv_heads, head_dim]` and return its geometry
/// `(n_slots, n_kv_heads, head_dim)`.
///
/// Accepts either the SDPA-style `[1, n_kv_heads, n_slots, head_dim]` (4D) or
/// the bare layout-A `[n_slots, n_kv_heads, head_dim]` (3D). No dtype
/// conversion is performed: `reshape`/`transpose_axes` preserve dtype, so an
/// FP16 block stays FP16 and an INT8 block stays INT8.
fn normalize_block_for_layout_a(block: &MlxArray, label: &str) -> Result<NormalizedBlock, String> {
    let shape = ffi::array_shape(block);
    match shape.as_slice() {
        // [1, n_kv_heads, n_slots, head_dim] -> [1, n_slots, n_kv_heads, head_dim]
        [batch, n_kv_heads, n_slots, head_dim] => {
            if *batch != 1 {
                return Err(format!(
                    "PagedBlockPool::write_block: {label} 4D batch dim must be 1, got {batch} (shape {shape:?})"
                ));
            }
            let slot = ffi::transpose_axes(block, &[0, 2, 1, 3]);
            Ok((slot, (*n_slots, *n_kv_heads, *head_dim)))
        }
        // [n_slots, n_kv_heads, head_dim] -> [1, n_slots, n_kv_heads, head_dim]
        [n_slots, n_kv_heads, head_dim] => {
            let slot = ffi::reshape(block, &[1, *n_slots, *n_kv_heads, *head_dim]);
            Ok((slot, (*n_slots, *n_kv_heads, *head_dim)))
        }
        _ => Err(format!(
            "PagedBlockPool::write_block: {label} must be [1, n_kv_heads, n_slots, head_dim] or [n_slots, n_kv_heads, head_dim], got shape {shape:?}"
        )),
    }
}
