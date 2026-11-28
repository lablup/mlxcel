// Copyright 2025-2026 Lablup Inc.
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

//! Scheduler-owned per-layer sequence state (issue #3254, ADR 0010).
//!
//! A family that reports [`LanguageModel::layer_state_kinds`] no longer keeps
//! a per-sequence cache map inside its wrapper. The pool builds one
//! [`LayerState`] per layer from the family's [`LayerStateKind`] table, keeps
//! it in the sequence's [`SequenceCacheSet`], and hands it to the model's
//! forward. Save, restore, truncation, pad trim and decode rewind are then
//! written once over `[LayerState]` instead of once per family.
//!
//! The storage behind a kind is a property of this module: [`LayerState::new`]
//! is the only place that maps `Sliding` to a [`RotatingKVCache`], so a later
//! storage change (window-bounded paged blocks) stays local to this file and
//! the snapshot encoders that read the variant.
//!
//! [`LanguageModel::layer_state_kinds`]: crate::generate::LanguageModel::layer_state_kinds
//! [`SequenceCacheSet`]: super::SequenceCacheSet

use cxx::UniquePtr;

use super::{
    ChunkedKVCache, KVCache, KVCacheMode, KvAttention, RotatingKVCache, flush_decode_undo_rows,
};
use crate::ffi::{self, MlxArray};

/// What one layer keeps per sequence.
///
/// A family reports one kind per layer with the same predicate its forward
/// uses, so storage reporting ([`Self::stored_mode`]), allocation
/// ([`LayerState::new`]) and the snapshot policy all read one table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerStateKind {
    /// Full attention: a [`KVCache`] built with the injected mode.
    Full,
    /// Sliding-window attention over `window` positions.
    Sliding { window: i32 },
    /// Chunked attention with `chunk`-position chunks, stored FP16 whatever
    /// the injected mode is.
    Chunked { chunk: i32 },
    /// Recurrent, convolution or linear-attention state: a summary of every
    /// token consumed, not per-token rows.
    Recurrent,
    /// Nothing (a KV-shared layer or a skipped attention block).
    None,
}

impl LayerStateKind {
    /// `Sliding { window }`, for call sites that map a window to a kind.
    pub const fn sliding(window: i32) -> Self {
        Self::Sliding { window }
    }

    /// The KV mode a layer of this kind stores when `mode` is injected, or
    /// `None` for a layer that holds no attention KV.
    pub const fn stored_mode(self, mode: KVCacheMode) -> Option<KVCacheMode> {
        match self {
            Self::Full => Some(mode),
            Self::Sliding { window } => Some(RotatingKVCache::stored_mode(window, mode)),
            Self::Chunked { .. } => Some(KVCacheMode::Fp16),
            Self::Recurrent | Self::None => None,
        }
    }

    /// The mode to build this layer's cache with.
    ///
    /// A `Turbo4Asym` sliding layer whose window is not a positive multiple of
    /// the Turbo block size falls back to FP16, because
    /// [`RotatingKVCache::new_with_mode`] panics for it. Every other mode
    /// passes through unchanged: the rotating cache already stores FP16 for
    /// them and keeps the label it was given. Snapshots and `can_truncate_to`
    /// read the stored mode, not that label (issue #3250).
    pub const fn build_mode(self, mode: KVCacheMode) -> KVCacheMode {
        match self {
            Self::Sliding { window }
                if matches!(mode, KVCacheMode::Turbo4Asym)
                    && !matches!(
                        RotatingKVCache::stored_mode(window, mode),
                        KVCacheMode::Turbo4Asym
                    ) =>
            {
                KVCacheMode::Fp16
            }
            _ => mode,
        }
    }
}

/// Named tensors of one recurrent layer: recurrent and convolution state and
/// any family-specific tensors.
///
/// Slots keep their insertion order, which is the order a snapshot stores
/// them in. `offset` is the number of tokens the state has consumed.
#[derive(Default)]
pub struct RecurrentState {
    tensors: Vec<(&'static str, Option<UniquePtr<MlxArray>>)>,
    pub offset: i32,
}

impl RecurrentState {
    pub fn new() -> Self {
        Self::default()
    }

    /// An empty state with `names` declared in snapshot order.
    pub fn with_slots(names: &[&'static str]) -> Self {
        Self {
            tensors: names.iter().map(|name| (*name, None)).collect(),
            offset: 0,
        }
    }

    /// Borrow the tensor stored under `name`.
    pub fn get(&self, name: &str) -> Option<&MlxArray> {
        self.tensors
            .iter()
            .find(|(slot, _)| *slot == name)
            .and_then(|(_, tensor)| tensor.as_deref())
    }

    /// Move the tensor stored under `name` out, leaving its slot empty.
    pub fn take(&mut self, name: &str) -> Option<UniquePtr<MlxArray>> {
        self.tensors
            .iter_mut()
            .find(|(slot, _)| *slot == name)
            .and_then(|(_, tensor)| tensor.take())
    }

    /// Store `tensor` under `name`, keeping the slot's position when it
    /// already exists and appending it otherwise.
    pub fn put(&mut self, name: &'static str, tensor: Option<UniquePtr<MlxArray>>) {
        match self.tensors.iter_mut().find(|(slot, _)| *slot == name) {
            Some((_, slot)) => *slot = tensor,
            None => self.tensors.push((name, tensor)),
        }
    }

    /// Every slot in snapshot order, empty ones included.
    pub fn slots(&self) -> impl Iterator<Item = (&'static str, Option<&MlxArray>)> {
        self.tensors
            .iter()
            .map(|(name, tensor)| (*name, tensor.as_deref()))
    }

    /// Slot names in snapshot order.
    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.tensors.iter().map(|(name, _)| *name)
    }

    /// Drop every tensor and reset the offset, keeping the declared slots.
    pub fn clear(&mut self) {
        for (_, tensor) in &mut self.tensors {
            *tensor = None;
        }
        self.offset = 0;
    }

    /// Bytes held by the stored tensors.
    pub fn nbytes(&self) -> usize {
        self.tensors
            .iter()
            .filter_map(|(_, tensor)| tensor.as_deref())
            .map(ffi::array_nbytes)
            .sum()
    }
}

/// One layer's per-sequence state, owned by the scheduler's pool entry.
///
/// Attention goes through [`KvAttention`] by matching to the inner cache
/// (ADR 0008, ADR 0009), so the storage behind a layer picks its kernel.
pub enum LayerState {
    Kv(KVCache),
    Rotating(RotatingKVCache),
    Chunked(ChunkedKVCache),
    Recurrent(RecurrentState),
    Empty,
}

impl LayerState {
    /// The fresh state of a layer of `kind` under the injected `mode`.
    pub fn new(kind: LayerStateKind, mode: KVCacheMode) -> Self {
        match kind {
            LayerStateKind::Full => Self::Kv(KVCache::new_with_mode(mode)),
            LayerStateKind::Sliding { window } => Self::Rotating(RotatingKVCache::new_with_mode(
                window,
                kind.build_mode(mode),
            )),
            LayerStateKind::Chunked { chunk } => Self::Chunked(ChunkedKVCache::new(chunk)),
            LayerStateKind::Recurrent => Self::Recurrent(RecurrentState::new()),
            LayerStateKind::None => Self::Empty,
        }
    }

    /// Fresh state for every layer of `kinds`. `modes[i]` is layer `i`'s
    /// injected mode; a missing table or a short one falls back to FP16.
    ///
    /// With `decode_undo_depth > 0` every sliding layer logs that many of its
    /// newest single-token writes, so a decode lookahead teardown can restore
    /// a wrapped window (issue #2159).
    pub fn for_kinds(
        kinds: &[LayerStateKind],
        modes: Option<&[KVCacheMode]>,
        decode_undo_depth: usize,
    ) -> Vec<Self> {
        kinds
            .iter()
            .enumerate()
            .map(|(layer_idx, &kind)| {
                let mode = modes
                    .and_then(|modes| modes.get(layer_idx).copied())
                    .unwrap_or(KVCacheMode::Fp16);
                let mut state = Self::new(kind, mode);
                if decode_undo_depth > 0
                    && let Self::Rotating(rotating) = &mut state
                {
                    rotating.set_decode_undo_depth(decode_undo_depth);
                }
                state
            })
            .collect()
    }

    /// Whether this state is the storage [`Self::new`] builds for `kind`.
    pub fn matches_kind(&self, kind: LayerStateKind) -> bool {
        match (self, kind) {
            (Self::Kv(_), LayerStateKind::Full)
            | (Self::Recurrent(_), LayerStateKind::Recurrent)
            | (Self::Empty, LayerStateKind::None) => true,
            (Self::Rotating(cache), LayerStateKind::Sliding { window }) => cache.max_size == window,
            (Self::Chunked(_), LayerStateKind::Chunked { .. }) => true,
            _ => false,
        }
    }

    /// Short name of the storage variant, for error messages.
    pub const fn storage_name(&self) -> &'static str {
        match self {
            Self::Kv(_) => "full",
            Self::Rotating(_) => "sliding",
            Self::Chunked(_) => "chunked",
            Self::Recurrent(_) => "recurrent",
            Self::Empty => "empty",
        }
    }

    /// Tokens this layer has consumed (the RoPE position of the next token).
    pub fn offset(&self) -> i32 {
        match self {
            Self::Kv(cache) => cache.offset,
            Self::Rotating(cache) => cache.offset,
            Self::Chunked(cache) => cache.offset,
            Self::Recurrent(state) => state.offset,
            Self::Empty => 0,
        }
    }

    /// Keys the next decode step attends: the live length of a full layer,
    /// the visible window of a sliding one, and zero for a layer with no
    /// attention KV, so paged shadow accounting reserves no blocks for it.
    pub fn visible_len(&self) -> usize {
        match self {
            Self::Kv(cache) => cache.seq_len().max(0) as usize,
            Self::Rotating(cache) => cache.visible_len().max(0) as usize,
            Self::Chunked(cache) => (cache.offset - cache.get_start_position()).max(0) as usize,
            Self::Recurrent(_) | Self::Empty => 0,
        }
    }

    /// Bytes held by this layer.
    pub fn nbytes(&self) -> usize {
        let pair = |keys: &Option<UniquePtr<MlxArray>>, values: &Option<UniquePtr<MlxArray>>| {
            keys.as_deref().map_or(0, ffi::array_nbytes)
                + values.as_deref().map_or(0, ffi::array_nbytes)
        };
        match self {
            Self::Kv(cache) => cache.nbytes(),
            Self::Rotating(cache) => pair(&cache.keys, &cache.values),
            Self::Chunked(cache) => pair(&cache.keys, &cache.values),
            Self::Recurrent(state) => state.nbytes(),
            Self::Empty => 0,
        }
    }

    /// Drop the trailing `excess` positions a tile-aligned padded prefill
    /// wrote, leaving the state an unpadded prefill of the same tokens would
    /// have left (issue #1755).
    ///
    /// # Errors
    ///
    /// A recurrent layer absorbed the pad positions and a chunked layer needs
    /// a chunk-boundary rewind; neither can drop them. Families that hold
    /// such a layer decline padded prefill.
    pub fn rewind_padded_prefill(&mut self, excess: i32) -> Result<(), String> {
        match self {
            Self::Kv(cache) => {
                let trimmed = cache.trim(excess);
                if trimmed == excess {
                    Ok(())
                } else {
                    Err(format!(
                        "standard cache dropped {trimmed} of {excess} pad positions"
                    ))
                }
            }
            Self::Rotating(cache) => cache.rewind_padded_prefill(excess),
            Self::Empty => Ok(()),
            Self::Chunked(_) | Self::Recurrent(_) => Err(format!(
                "a {} layer cannot drop {excess} pad positions",
                self.storage_name()
            )),
        }
    }

    /// Unwind the last `n` single-token decode appends (issue #2159). A full
    /// layer's cursor trim is exact, and a sliding layer restores the rows
    /// its undo log kept.
    ///
    /// # Errors
    ///
    /// A recurrent or chunked layer cannot be rewound, and a sliding layer
    /// fails when its undo log does not cover `n` writes.
    pub fn rewind_decode_appends(&mut self, n: i32) -> Result<(), String> {
        match self {
            Self::Kv(cache) => {
                let offset = cache.offset;
                let trimmed = cache.trim(n);
                if trimmed == n {
                    Ok(())
                } else {
                    Err(format!(
                        "global decode rewind of {n}: trimmed {trimmed} (offset was {offset})"
                    ))
                }
            }
            Self::Rotating(cache) => cache.rewind_decode_writes(n),
            Self::Empty => Ok(()),
            Self::Chunked(_) | Self::Recurrent(_) => Err(format!(
                "a {} layer cannot rewind {n} decode appends",
                self.storage_name()
            )),
        }
    }

    /// Schedule the pending undo-row copies of every sliding layer in
    /// `layers` (see [`flush_decode_undo_rows`]). No MLX call when nothing is
    /// pending, which is always the case without an undo log.
    pub fn flush_decode_undo<'a>(layers: impl IntoIterator<Item = &'a mut LayerState>) {
        flush_decode_undo_rows(layers.into_iter().filter_map(|layer| match layer {
            Self::Rotating(rotating) => Some(rotating),
            _ => None,
        }));
    }

    pub fn as_kv(&self) -> Option<&KVCache> {
        match self {
            Self::Kv(cache) => Some(cache),
            _ => None,
        }
    }

    pub fn as_kv_mut(&mut self) -> Option<&mut KVCache> {
        match self {
            Self::Kv(cache) => Some(cache),
            _ => None,
        }
    }

    pub fn as_rotating(&self) -> Option<&RotatingKVCache> {
        match self {
            Self::Rotating(cache) => Some(cache),
            _ => None,
        }
    }

    pub fn as_rotating_mut(&mut self) -> Option<&mut RotatingKVCache> {
        match self {
            Self::Rotating(cache) => Some(cache),
            _ => None,
        }
    }

    pub fn as_recurrent(&self) -> Option<&RecurrentState> {
        match self {
            Self::Recurrent(state) => Some(state),
            _ => None,
        }
    }

    pub fn as_recurrent_mut(&mut self) -> Option<&mut RecurrentState> {
        match self {
            Self::Recurrent(state) => Some(state),
            _ => None,
        }
    }
}

/// Attention through the storage behind the layer (ADR 0008, #2172). A layer
/// with no attention KV is never attended; reaching one here means the
/// family's forward and its [`LayerStateKind`] table disagree.
impl KvAttention for LayerState {
    fn attend(
        &mut self,
        q: &MlxArray,
        new_keys: UniquePtr<MlxArray>,
        new_values: UniquePtr<MlxArray>,
        scale: f32,
        mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        match self {
            Self::Kv(cache) => KVCache::attend(cache, q, new_keys, new_values, scale, mask),
            Self::Rotating(cache) => {
                RotatingKVCache::attend(cache, q, new_keys, new_values, scale, mask)
            }
            Self::Chunked(cache) => {
                KvAttention::attend(cache, q, new_keys, new_values, scale, mask)
            }
            Self::Recurrent(_) | Self::Empty => panic!(
                "attention over a {} layer: the forward and layer_state_kinds() disagree",
                self.storage_name()
            ),
        }
    }
}

#[cfg(test)]
#[path = "layer_state_tests.rs"]
mod tests;
