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

//! Serialization of KV cache state into the versioned binary wire format.
//!
//! # Wire Format (little-endian)
//!
//! ```text
//! [version: u8]           // Cache serialization version (2 current, 1 legacy)
//! [cache_type: u8]        // Standard=0, Rotating=1, Chunked=2
//! [num_layers: u32 LE]    // Number of cache layers
//! [metadata_len: u32 LE]  // JSON metadata length
//! [metadata: bytes]       // JSON-encoded SerializableCacheState (includes tensor data,
//!                         // backend metadata, and optional paged sequence state)
//! For each layer:
//!   [has_data: u8]        // 0 = empty layer, 1 = has key/value tensors
//! ```
//!
//! Used by: disaggregated serving pipeline (prefill -> decode handoff)

use anyhow::{Context, Result};

use super::types::{
    CACHE_FORMAT_VERSION_V2, CacheMetadata, CacheType, RawTensorData, SerializableCacheEntry,
    SerializableCacheState, SerializablePagedBlock, SerializablePagedSequenceState,
    SerializableSequenceBackend,
};

/// Current cache serialization format version.
pub const CACHE_FORMAT_VERSION: u8 = CACHE_FORMAT_VERSION_V2;

/// Serialize a `SerializableCacheState` into the binary wire format.
///
/// Returns the complete byte buffer ready for network transfer.
pub fn serialize_cache_state(state: &SerializableCacheState) -> Result<Vec<u8>> {
    let metadata_json =
        serde_json::to_vec(state).context("failed to serialize cache state metadata to JSON")?;

    let metadata_len = u32::try_from(metadata_json.len())
        .map_err(|_| anyhow::anyhow!("metadata JSON exceeds u32 length"))?;

    let num_layers = u32::try_from(state.entries.len())
        .map_err(|_| anyhow::anyhow!("num_layers exceeds u32"))?;

    // Header (10 bytes) + metadata + per-layer presence flags (1 byte each)
    let estimated_size = 10 + metadata_json.len() + state.entries.len();
    let mut buf = Vec::with_capacity(estimated_size);

    // Write header
    buf.push(CACHE_FORMAT_VERSION);
    buf.push(state.cache_type as u8);
    buf.extend_from_slice(&num_layers.to_le_bytes());
    buf.extend_from_slice(&metadata_len.to_le_bytes());
    buf.extend_from_slice(&metadata_json);

    // Write per-layer presence flags.
    // The actual tensor data is carried inside the JSON metadata
    // (via SerializableCacheState -> RawTensorData), so no separate
    // binary tensor frames are needed. The flags are retained for
    // forward-compatible framing.
    for entry in &state.entries {
        if entry.keys.is_some() && entry.values.is_some() {
            buf.push(1); // has_data
        } else {
            buf.push(0); // empty layer
        }
    }

    Ok(buf)
}

/// Extract a `SerializableCacheEntry` from a live `KVCache`.
///
/// Evaluates the MLX arrays and copies their data to Rust-owned buffers.
/// Only the filled portion (up to `cache.offset`) is serialized, not
/// the full pre-allocated buffer.
///
/// Used by: prefill node when preparing cache state for transfer.
pub fn extract_kv_cache_entry(cache: &mlxcel_core::cache::KVCache) -> SerializableCacheEntry {
    // Use `live_len() = offset - live_start` (not the monotonic `offset`) so
    // this remains correct after's `trim_front` has advanced
    // `live_start`. With `live_start == 0` this is bit-identical to slicing
    // at `cache.offset`.
    let live_len = cache.live_len();
    match (&cache.keys, &cache.values) {
        (Some(keys), Some(values)) if live_len > 0 => {
            // Slice to the filled portion only (buffer may be larger due to step allocation)
            let ks = mlxcel_core::array_shape(keys);
            let vs = mlxcel_core::array_shape(values);
            let filled_k =
                mlxcel_core::slice(keys, &[0, 0, 0, 0], &[ks[0], ks[1], live_len, ks[3]]);
            let filled_v =
                mlxcel_core::slice(values, &[0, 0, 0, 0], &[vs[0], vs[1], live_len, vs[3]]);
            let k_data = extract_mlx_array_data(&filled_k);
            let v_data = extract_mlx_array_data(&filled_v);
            SerializableCacheEntry {
                keys: Some(k_data),
                values: Some(v_data),
            }
        }
        _ => SerializableCacheEntry {
            keys: None,
            values: None,
        },
    }
}

/// Extract a `SerializableCacheEntry` from a live `RotatingKVCache`.
pub fn extract_rotating_cache_entry(
    cache: &mlxcel_core::cache::RotatingKVCache,
) -> SerializableCacheEntry {
    match (&cache.keys, &cache.values) {
        (Some(keys), Some(values)) => {
            let k_data = extract_mlx_array_data(keys);
            let v_data = extract_mlx_array_data(values);
            SerializableCacheEntry {
                keys: Some(k_data),
                values: Some(v_data),
            }
        }
        _ => SerializableCacheEntry {
            keys: None,
            values: None,
        },
    }
}

/// Extract a `SerializableCacheEntry` from a live `ChunkedKVCache`.
pub fn extract_chunked_cache_entry(
    cache: &mlxcel_core::cache::ChunkedKVCache,
) -> SerializableCacheEntry {
    match (&cache.keys, &cache.values) {
        (Some(keys), Some(values)) => {
            let k_data = extract_mlx_array_data(keys);
            let v_data = extract_mlx_array_data(values);
            SerializableCacheEntry {
                keys: Some(k_data),
                values: Some(v_data),
            }
        }
        _ => SerializableCacheEntry {
            keys: None,
            values: None,
        },
    }
}

/// Extract raw tensor data from a `UniquePtr<MlxArray>`.
fn extract_mlx_array_data(arr: &mlxcel_core::UniquePtr<mlxcel_core::MlxArray>) -> RawTensorData {
    let shape = mlxcel_core::array_shape(arr);
    let dtype = mlxcel_core::array_dtype(arr);
    let data = mlxcel_core::array_to_raw_bytes(arr);

    RawTensorData { data, shape, dtype }
}

/// Serialize all layer caches from a `SequenceCacheSet` into a
/// `SerializableCacheState`.
///
/// This is the primary entry point for the prefill node. It extracts
/// all layer caches, metadata, and packages them for transfer.
pub fn serialize_sequence_cache_set(
    cache_set: &mlxcel_core::cache::SequenceCacheSet,
    sampling_state: Option<super::types::SerializableSamplingState>,
    token_history: Vec<i32>,
) -> SerializableCacheState {
    let num_layers = cache_set.caches.len();
    let mut entries = Vec::with_capacity(num_layers);
    let mut layer_offsets = Vec::with_capacity(num_layers);

    for cache in &cache_set.caches {
        entries.push(extract_kv_cache_entry(cache));
        // Re-base the serialized offset to the live window length so the
        // receiving node restores `(offset=live_len, live_start=0)` —
        // equivalent in RoPE relative positions to the original
        // `(offset, live_start)` pair, but without leaking the post-
        // monotonic position through the wire format. With `live_start == 0`
        // this is bit-identical to recording `cache.offset` directly.
        layer_offsets.push(cache.live_len());
    }

    let metadata = CacheMetadata {
        prompt_len: cache_set.prompt_len,
        current_offset: cache_set.current_offset,
        num_layers,
        layer_offsets,
        max_size: None,
        layer_indices: None,
        chunk_size: None,
        start_positions: None,
    };

    SerializableCacheState {
        cache_type: CacheType::Standard,
        entries,
        metadata,
        sampling_state,
        token_history,
        sequence_id: cache_set.seq_id.as_u64(),
        sequence_backend: SerializableSequenceBackend::from_runtime(cache_set.backend),
        paged_state: cache_set
            .paged_state()
            .zip(cache_set.paged_layout())
            .map(|(state, layout)| SerializablePagedSequenceState::from_runtime(&state, layout)),
        // Pool block CONTENTS are not reachable from a bare `SequenceCacheSet`
        // (they live in the shared `PagedBlockPool`). They are filled in by
        // `serialize_cache_pool_sequence`, which has the owning `CachePool`.
        paged_blocks: Vec::new(),
    }
}

/// Serialize a sequence directly from its owning [`CachePool`], including the
/// paged pool block CONTENTS for a pool-backed (Fp16 paged) sequence (#125).
///
/// This is the pool-aware superset of [`serialize_sequence_cache_set`]: it
/// assembles the dense entries, metadata, and paged block-table snapshot exactly
/// as the cache-set path does, then attaches the live K/V slab of every physical
/// pool block backing the sequence (via [`CachePool::extract_paged_blocks`]). For
/// a dense / metadata-only sequence the block list is empty and the result is
/// identical to [`serialize_sequence_cache_set`], so the wire format stays
/// backward compatible.
///
/// Used by: the prefill node when handing a pool-backed sequence to a decode
/// node. The scheduler / transport wiring that calls this is the #126 capstone.
pub fn serialize_cache_pool_sequence(
    cache_pool: &mlxcel_core::cache::CachePool,
    id: mlxcel_core::cache::SequenceId,
    sampling_state: Option<super::types::SerializableSamplingState>,
    token_history: Vec<i32>,
) -> Result<SerializableCacheState> {
    let cache_set = cache_pool
        .get(id)
        .ok_or_else(|| anyhow::anyhow!("CachePool: sequence {id} not found"))?;
    let mut state = serialize_sequence_cache_set(cache_set, sampling_state, token_history);

    state.paged_blocks = cache_pool
        .extract_paged_blocks(id)
        .map_err(anyhow::Error::msg)?
        .into_iter()
        .map(|content| SerializablePagedBlock {
            block_id: content.block_id.as_u64(),
            layer_idx: content.layer_idx,
            keys: extract_mlx_array_data(&content.keys),
            values: extract_mlx_array_data(&content.values),
        })
        .collect();

    Ok(state)
}
