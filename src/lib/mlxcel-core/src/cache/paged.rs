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
/// `stats_for_sequences` and internal debug assertions stay backwards
/// compatible with call sites that inspected the flag directly before
/// refcounts were introduced.
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

/// Physical block allocator shared across every active paged sequence.
///
/// In `Fp16`/`Int8` modes the pool tracks block-table metadata only — the
/// actual KV tensors live in dense [`super::KVCache`] placeholders attached to
/// each [`super::SequenceCacheSet`]. In Turbo4 modes the pool also owns per-
/// page sidecar MLX arrays (`v_packed`, `v_norms`, optionally `k_packed`,
/// `k_norms`, `cold_keys`) so packed quantization state survives across
/// detach/adopt round-trips and prefix-cache handoffs.
pub struct PagedBlockPool {
    layout: PagedKvLayout,
    next_block_id: u64,
    blocks: HashMap<PagedBlockId, PagedBlockRecord>,
    free_lists: Vec<Vec<PagedBlockId>>,
    /// Per-page Turbo4 sidecar tensors, keyed by `PagedBlockId`. Empty for
    /// `Fp16`/`Int8` cache modes; populated by Turbo4 sequences via
    /// [`PagedBlockPool::install_turbo_sidecar`].
    turbo_sidecars: PagedTurboPageSidecars,
}

impl PagedBlockPool {
    pub fn new(layout: PagedKvLayout) -> Self {
        Self {
            free_lists: vec![Vec::new(); layout.num_layers],
            layout,
            next_block_id: 0,
            blocks: HashMap::new(),
            turbo_sidecars: PagedTurboPageSidecars::default(),
        }
    }

    pub fn layout(&self) -> &PagedKvLayout {
        &self.layout
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

        let new_visible_len = layer.visible_len() + token_count;
        let required_blocks = new_visible_len.div_ceil(self.layout.block_size);
        while layer.block_ids.len() < required_blocks {
            layer.block_ids.push(self.acquire_block(layer_idx)?);
        }
        layer.len += token_count;
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
            layer.logical_start = 0;
        }

        let required_blocks = layer.visible_len().div_ceil(self.layout.block_size);
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
            let required_blocks = layer.visible_len().div_ceil(self.layout.block_size);
            if layer.block_ids.len() < required_blocks {
                return Err(format!(
                    "PagedBlockPool: restored layer {layer_idx} has {} blocks for visible length {}, requires at least {}",
                    layer.block_ids.len(),
                    layer.visible_len(),
                    required_blocks
                ));
            }

            for &block_id in &layer.block_ids {
                self.restore_block(block_id, layer_idx)?;
            }
        }
        Ok(())
    }

    pub fn stats_for_sequences<'a>(
        &self,
        sequences: impl IntoIterator<Item = &'a PagedSequenceState>,
    ) -> PagedCacheStats {
        let allocated_blocks = self.blocks.len();
        let free_blocks = self
            .blocks
            .values()
            .filter(|record| !record.is_in_use())
            .count();
        let live_blocks = allocated_blocks.saturating_sub(free_blocks);
        let states: Vec<&PagedSequenceState> = sequences.into_iter().collect();
        let bytes_reserved = states
            .iter()
            .map(|state| state.reserved_bytes(&self.layout))
            .sum();
        let bytes_in_use = states
            .iter()
            .map(|state| state.used_bytes(&self.layout))
            .sum();

        PagedCacheStats {
            allocated_blocks,
            live_blocks,
            free_blocks,
            bytes_reserved,
            bytes_in_use,
        }
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
    /// reaches zero. Per-page Turbo4 sidecars (if any) are dropped at the same
    /// moment so a recycled block can never serve stale packed data to a new
    /// sequence.
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
