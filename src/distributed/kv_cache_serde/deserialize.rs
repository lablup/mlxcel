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

//! Deserialization of KV cache state from the binary wire format.
//!
//! Reconstructs `KVCache`, `RotatingKVCache`, or `ChunkedKVCache` instances
//! from serialized bytes, writing into pre-allocated `CachePool` slots.
//!
//! Used by: decode node in disaggregated serving pipeline

use anyhow::{Context, Result, bail};

use super::serialize::CACHE_FORMAT_VERSION;
use super::types::{
    CACHE_FORMAT_VERSION_V1, CacheType, RawTensorData, SerializableCacheState,
    SerializableSequenceBackend, validate_raw_tensor,
};

/// Deserialize a `SerializableCacheState` from the binary wire format.
///
/// Returns the deserialized state ready for reconstruction into live
/// cache objects.
pub fn deserialize_cache_state(buf: &[u8]) -> Result<SerializableCacheState> {
    if buf.len() < 10 {
        bail!(
            "cache state buffer too short: need at least 10 bytes, got {}",
            buf.len()
        );
    }

    let version = buf[0];
    if version != CACHE_FORMAT_VERSION_V1 && version != CACHE_FORMAT_VERSION {
        bail!(
            "unsupported cache format version: {version} (expected {CACHE_FORMAT_VERSION_V1} or {CACHE_FORMAT_VERSION})"
        );
    }

    let _cache_type = CacheType::try_from(buf[1]).context("invalid cache type discriminant")?;

    let num_layers = u32::from_le_bytes(buf[2..6].try_into()?) as usize;
    let metadata_len = u32::from_le_bytes(buf[6..10].try_into()?) as usize;

    let metadata_end = 10 + metadata_len;
    if buf.len() < metadata_end {
        bail!(
            "buffer too short for metadata: need {} bytes, got {}",
            metadata_end,
            buf.len()
        );
    }

    let state: SerializableCacheState = serde_json::from_slice(&buf[10..metadata_end])
        .context("failed to deserialize cache state JSON")?;

    // Validate consistency
    if state.entries.len() != num_layers {
        bail!(
            "layer count mismatch: header says {num_layers}, JSON has {}",
            state.entries.len()
        );
    }

    if state.metadata.num_layers != num_layers {
        bail!(
            "metadata layer count mismatch: header says {num_layers}, metadata has {}",
            state.metadata.num_layers
        );
    }

    if let Some(paged_state) = state.paged_state.as_ref() {
        paged_state
            .validate()
            .context("invalid paged sequence state in cache payload")?;
    }

    // Tensor data is embedded in the JSON via RawTensorData.
    // Validation of individual tensors happens in reconstruct_mlx_array().

    Ok(state)
}

/// Reconstruct a `RawTensorData` into a live `UniquePtr<MlxArray>`.
///
/// Validates shape/dtype/data consistency before calling `ffi::from_bytes`
/// to prevent out-of-bounds reads from malformed payloads.
pub fn reconstruct_mlx_array(
    tensor: &RawTensorData,
) -> Result<mlxcel_core::UniquePtr<mlxcel_core::MlxArray>> {
    validate_raw_tensor(tensor)?;
    Ok(mlxcel_core::from_bytes(
        &tensor.data,
        &tensor.shape,
        tensor.dtype,
    ))
}

/// Restore serialized cache entries into a pre-allocated `KVCache` slice.
///
/// Each cache in `caches` is replaced with the deserialized state.
/// The caller is responsible for ensuring `caches.len()` matches the
/// number of entries in `state`.
///
/// Used by: decode node after receiving cache state from prefill node.
pub fn restore_into_kv_caches(
    state: &SerializableCacheState,
    caches: &mut [mlxcel_core::cache::KVCache],
) -> Result<()> {
    if state.entries.len() != caches.len() {
        bail!(
            "layer count mismatch: state has {} entries, target has {} caches",
            state.entries.len(),
            caches.len()
        );
    }

    for (i, (entry, cache)) in state.entries.iter().zip(caches.iter_mut()).enumerate() {
        match (&entry.keys, &entry.values) {
            (Some(keys), Some(values)) => {
                let k = reconstruct_mlx_array(keys).with_context(|| format!("layer {i} keys"))?;
                let v =
                    reconstruct_mlx_array(values).with_context(|| format!("layer {i} values"))?;
                cache.keys = Some(k);
                cache.values = Some(v);
                cache.offset = state.metadata.layer_offsets.get(i).copied().unwrap_or(0);
            }
            (None, None) => {
                cache.keys = None;
                cache.values = None;
                cache.offset = 0;
            }
            _ => {
                bail!(
                    "layer {i}: inconsistent state (keys and values must both be present or absent)"
                );
            }
        }
    }

    Ok(())
}

/// Restore serialized cache state into a `SequenceCacheSet`.
///
/// Updates the cache set's per-layer caches and metadata fields.
/// The `SequenceCacheSet` must already be allocated with the correct
/// number of layers.
pub fn restore_into_sequence_cache_set(
    state: &SerializableCacheState,
    cache_set: &mut mlxcel_core::cache::SequenceCacheSet,
) -> Result<()> {
    if !state.entries.is_empty() || !cache_set.caches.is_empty() {
        restore_into_kv_caches(state, &mut cache_set.caches)?;
    }
    cache_set.prompt_len = state.metadata.prompt_len;
    cache_set.current_offset = state.metadata.current_offset;

    match (&state.paged_state, cache_set.backend) {
        (Some(serialized), mlxcel_core::cache::SequenceStateBackend::PagedKvCache) => {
            let runtime = serialized.to_runtime()?;
            // `SequenceCacheSet::paged` is a shared `Rc<RefCell<…>>` so pool-backed
            // sequences can alias one block table across their per-layer caches
            // (#121). A deserialized sequence is freshly restored with no live
            // caches yet, so it is the sole owner.
            cache_set.paged = Some(std::rc::Rc::new(std::cell::RefCell::new(runtime)));
        }
        (Some(_), _) => {
            bail!("cannot restore paged state into a non-paged sequence cache set");
        }
        (None, mlxcel_core::cache::SequenceStateBackend::PagedKvCache)
            if state.sequence_backend == SerializableSequenceBackend::PagedKvCache =>
        {
            bail!("paged sequence payload is missing paged_state metadata");
        }
        _ => {}
    }

    Ok(())
}

/// Restore serialized cache state directly into an active `CachePool` slot.
///
/// Used by: decode-side distributed cache ingestion where paged allocator
/// bookkeeping must be rebuilt alongside the sequence state.
pub fn restore_into_cache_pool_sequence(
    state: &SerializableCacheState,
    cache_pool: &mut mlxcel_core::cache::CachePool,
    seq_id: mlxcel_core::cache::SequenceId,
) -> Result<()> {
    {
        let cache_set = cache_pool
            .get_mut(seq_id)
            .ok_or_else(|| anyhow::anyhow!("CachePool: sequence {seq_id} not found"))?;
        if !state.entries.is_empty() || !cache_set.caches.is_empty() {
            restore_into_kv_caches(state, &mut cache_set.caches)?;
        }
        cache_set.prompt_len = state.metadata.prompt_len;
        cache_set.current_offset = state.metadata.current_offset;
    }

    if let Some(serialized) = state.paged_state.as_ref() {
        cache_pool
            .restore_paged_state(seq_id, serialized.to_runtime()?)
            .map_err(anyhow::Error::msg)?;
    } else if state.sequence_backend == SerializableSequenceBackend::PagedKvCache {
        bail!("paged sequence payload is missing paged_state metadata");
    }

    Ok(())
}
