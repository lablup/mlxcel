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

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, bail};

use super::serialize::CACHE_FORMAT_VERSION;
use super::types::{
    CACHE_FORMAT_VERSION_V1, CacheIngestLimits, CacheType, ExpectedBlockGeometry, RawTensorData,
    SerializableCacheState, SerializableSequenceBackend, validate_raw_tensor,
};

/// Deserialize a `SerializableCacheState` from the binary wire format, applying
/// the default [`CacheIngestLimits`] frame cap.
///
/// Returns the deserialized state ready for reconstruction into live
/// cache objects.
pub fn deserialize_cache_state(buf: &[u8]) -> Result<SerializableCacheState> {
    deserialize_cache_state_with_limits(buf, &CacheIngestLimits::default())
}

/// Deserialize a `SerializableCacheState`, rejecting any frame larger than
/// `limits.max_frame_bytes` before parsing.
///
/// The cap is the first line of defense at the deserialization boundary: an
/// untrusted peer cannot force a multi-gigabyte allocation by sending an
/// oversized buffer or a forged metadata length. The disaggregated decode path
/// passes a config-derived cap; [`deserialize_cache_state`] uses the generous
/// default backstop.
pub fn deserialize_cache_state_with_limits(
    buf: &[u8],
    limits: &CacheIngestLimits,
) -> Result<SerializableCacheState> {
    if buf.len() > limits.max_frame_bytes {
        bail!(
            "cache state frame is {} bytes, exceeds limit {}",
            buf.len(),
            limits.max_frame_bytes
        );
    }
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

    if metadata_len > limits.max_frame_bytes {
        bail!(
            "cache metadata length {} exceeds frame limit {}",
            metadata_len,
            limits.max_frame_bytes
        );
    }

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

/// Validate a deserialized paged handoff payload against ingest limits and,
/// when supplied, the decode model's real KV geometry, BEFORE any pool block is
/// acquired. This is the structural gate for a cross-node paged restore.
///
/// Always enforced (no live model needed):
/// - the transferred block count and each slab's byte size stay within `limits`;
/// - every block the restored table references has matching transferred
///   contents on the SAME layer, and every transferred block is referenced by
///   the table on its declared layer (no orphan contents, which would leak a
///   pool block; no cross-layer aliasing; no duplicate id disagreeing on layer);
/// - a paged sequence carries a coherent, non-empty content path (an empty
///   block table, or a table that references blocks with no contents, is
///   rejected; a metadata-only paged payload is not a valid cross-node handoff
///   because a fresh decode pool holds no pre-existing block contents).
///
/// Enforced when `expected` is provided:
/// - every slab's shape is exactly `[block_size, n_kv_heads, head_dim]` and its
///   dtype matches the decode model, anchoring pool geometry before the first
///   write captures it from the (untrusted) slab.
///
/// Dense payloads (no paged state, no blocks) pass unchanged.
pub fn validate_paged_payload(
    state: &SerializableCacheState,
    limits: &CacheIngestLimits,
    expected: Option<&ExpectedBlockGeometry>,
) -> Result<()> {
    let is_paged = state.sequence_backend == SerializableSequenceBackend::PagedKvCache
        || state.paged_state.is_some()
        || !state.paged_blocks.is_empty();
    if !is_paged {
        return Ok(());
    }

    if state.paged_blocks.len() > limits.max_paged_blocks {
        bail!(
            "paged handoff carries {} blocks, exceeds limit {}",
            state.paged_blocks.len(),
            limits.max_paged_blocks
        );
    }

    let num_layers = expected
        .map(|g| g.num_layers)
        .or_else(|| state.paged_state.as_ref().map(|s| s.layers.len()));

    // Per-block: byte caps, layer range, optional geometry anchor. Index each
    // supplied block id to its declared layer for the table cross-check below.
    let mut block_layer: HashMap<u64, usize> = HashMap::new();
    for (i, block) in state.paged_blocks.iter().enumerate() {
        for (name, slab) in [("keys", &block.keys), ("values", &block.values)] {
            if slab.data.len() > limits.max_block_slab_bytes {
                bail!(
                    "paged block {i} {name} slab is {} bytes, exceeds limit {}",
                    slab.data.len(),
                    limits.max_block_slab_bytes
                );
            }
        }
        if let Some(n) = num_layers
            && block.layer_idx >= n
        {
            bail!(
                "paged block {i} layer_idx {} out of range for {n} layers",
                block.layer_idx
            );
        }
        if let Some(g) = expected {
            let want = g.slab_shape();
            for (name, slab) in [("keys", &block.keys), ("values", &block.values)] {
                if slab.shape != want {
                    bail!(
                        "paged block {i} {name} shape {:?} does not match decode model geometry {:?}",
                        slab.shape,
                        want
                    );
                }
                if slab.dtype != g.dtype {
                    bail!(
                        "paged block {i} {name} dtype {} does not match decode model dtype {}",
                        slab.dtype,
                        g.dtype
                    );
                }
            }
        }
        if let Some(prev) = block_layer.insert(block.block_id, block.layer_idx)
            && prev != block.layer_idx
        {
            bail!(
                "paged block id {} supplied for both layer {prev} and layer {} (block ids are layer-unique)",
                block.block_id,
                block.layer_idx
            );
        }
    }

    let Some(paged_state) = state.paged_state.as_ref() else {
        bail!("paged sequence payload is missing paged_state metadata");
    };

    let has_blocks = !state.paged_blocks.is_empty();
    let mut referenced: HashSet<u64> = HashSet::new();
    let mut referenced_count = 0usize;
    for (layer_idx, layer) in paged_state.layers.iter().enumerate() {
        for &block_id in &layer.block_ids {
            referenced_count += 1;
            referenced.insert(block_id);
            if has_blocks {
                match block_layer.get(&block_id) {
                    Some(&l) if l == layer_idx => {}
                    Some(&l) => bail!(
                        "block {block_id} is referenced by layer {layer_idx} but its transferred contents declare layer {l}"
                    ),
                    None => bail!(
                        "block {block_id} referenced by layer {layer_idx} has no transferred contents"
                    ),
                }
            }
        }
    }

    if referenced_count == 0 {
        bail!(
            "paged handoff carries an empty block table; a cross-node decode handoff must carry a prefilled sequence"
        );
    }
    if !has_blocks {
        bail!(
            "paged handoff references {referenced_count} blocks but carries no block contents; metadata-only paged restore is invalid on a fresh decode pool"
        );
    }
    for block in &state.paged_blocks {
        if !referenced.contains(&block.block_id) {
            bail!(
                "paged block id {} has transferred contents but is not referenced by any layer table",
                block.block_id
            );
        }
    }

    Ok(())
}

/// Restore serialized cache state directly into an active `CachePool` slot,
/// applying the default [`CacheIngestLimits`] and no model geometry anchor.
///
/// Used by: in-process round-trips and the decode-side ingestion that does not
/// (yet) supply the decode model's geometry. Prefer
/// [`restore_into_cache_pool_sequence_anchored`] on the live disaggregated path
/// so transferred slabs are pinned to the decode model.
pub fn restore_into_cache_pool_sequence(
    state: &SerializableCacheState,
    cache_pool: &mut mlxcel_core::cache::CachePool,
    seq_id: mlxcel_core::cache::SequenceId,
) -> Result<()> {
    restore_into_cache_pool_sequence_anchored(
        state,
        cache_pool,
        seq_id,
        &CacheIngestLimits::default(),
        None,
    )
}

/// Restore serialized cache state into an active `CachePool` slot, validating
/// the payload against `limits` and (when supplied) the decode model's real KV
/// geometry before any pool block is acquired.
///
/// Used by: decode-side distributed cache ingestion where paged allocator
/// bookkeeping must be rebuilt alongside the sequence state, and the untrusted
/// wire payload must be anchored to the local model.
pub fn restore_into_cache_pool_sequence_anchored(
    state: &SerializableCacheState,
    cache_pool: &mut mlxcel_core::cache::CachePool,
    seq_id: mlxcel_core::cache::SequenceId,
    limits: &CacheIngestLimits,
    expected: Option<&ExpectedBlockGeometry>,
) -> Result<()> {
    validate_paged_payload(state, limits, expected)?;

    // A true pool-backed cross-node handoff carries the pool block CONTENTS in
    // `paged_blocks` (#125). That path materializes fresh pool blocks rather
    // than restoring dense per-layer buffers (empty for pool-backed sequences)
    // or merely re-registering origin block-id metadata.
    let content_path = !state.paged_blocks.is_empty();

    {
        let cache_set = cache_pool
            .get_mut(seq_id)
            .ok_or_else(|| anyhow::anyhow!("CachePool: sequence {seq_id} not found"))?;
        // Skip the dense per-layer restore on the content path: pool-backed
        // caches keep their K/V in the shared pool (their dense entries are
        // empty) and `restore_paged_state_with_contents` sets their RoPE
        // offsets, so restoring dense buffers here would only zero those offsets.
        if !content_path && (!state.entries.is_empty() || !cache_set.caches.is_empty()) {
            restore_into_kv_caches(state, &mut cache_set.caches)?;
        }
        cache_set.prompt_len = state.metadata.prompt_len;
        cache_set.current_offset = state.metadata.current_offset;
    }

    if content_path {
        let runtime = state
            .paged_state
            .as_ref()
            .ok_or_else(|| {
                anyhow::anyhow!("paged block contents present but paged_state metadata is missing")
            })?
            .to_runtime()?;
        let mut contents = Vec::with_capacity(state.paged_blocks.len());
        for (i, block) in state.paged_blocks.iter().enumerate() {
            let keys = reconstruct_mlx_array(&block.keys)
                .with_context(|| format!("paged block {i} keys"))?;
            let values = reconstruct_mlx_array(&block.values)
                .with_context(|| format!("paged block {i} values"))?;
            contents.push(mlxcel_core::cache::PagedBlockContents {
                block_id: mlxcel_core::cache::PagedBlockId::from_raw(block.block_id),
                layer_idx: block.layer_idx,
                keys,
                values,
            });
        }
        cache_pool
            .restore_paged_state_with_contents(seq_id, runtime, contents)
            .map_err(anyhow::Error::msg)?;
    } else if let Some(serialized) = state.paged_state.as_ref() {
        cache_pool
            .restore_paged_state(seq_id, serialized.to_runtime()?)
            .map_err(anyhow::Error::msg)?;
    } else if state.sequence_backend == SerializableSequenceBackend::PagedKvCache {
        bail!("paged sequence payload is missing paged_state metadata");
    }

    Ok(())
}
