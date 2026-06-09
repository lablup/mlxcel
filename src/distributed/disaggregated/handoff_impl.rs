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

//! Cross-node paged KV handoff mechanism: the seam that joins the serde layer
//! (`kv_cache_serde`) to the transport layer for disaggregated serving (#126).
//!
//! Before this module the two halves never referenced each other: the #125
//! serde functions (`serialize_cache_pool_sequence` /
//! `restore_into_cache_pool_sequence_anchored`) round-tripped a pool-backed
//! paged sequence at the `CachePool` level, and the [`Transport`] trait moved
//! raw bytes between nodes, but nothing wired one into the other. This module
//! provides the three primitives a serving-role scheduler needs:
//!
//! 1. [`extract_sequence_handoff`] (prefill side): serialize a finished,
//!    pool-backed sequence (dense metadata + paged block table + the referenced
//!    pool block CONTENTS) into a single wire frame.
//! 2. [`ingest_sequence_handoff`] (decode side): validate an incoming frame,
//!    allocate a fresh pool-backed sequence, and reconstruct its KV on local
//!    pool blocks, anchored to the decode model's real block geometry.
//! 3. [`send_handoff_payload`] / [`recv_handoff_payload`]: the async
//!    serde<->transport byte bridge over any [`Transport`].
//!
//! The decode-side anchor needs the local model's exact paged block geometry
//! (`n_kv_heads`, `head_dim`, and the runtime KV dtype). Only `num_layers` and
//! the block size are statically known; the KV dtype in particular is the
//! model's runtime activation dtype (e.g. bf16 for a 4-bit checkpoint, not
//! fp16), which no static accessor exposes. [`probe_block_geometry`] derives all
//! three by running a single one-token forward through a throwaway pool-backed
//! sequence and reading the gathered window's shape and dtype, model-agnostic
//! and exact, with no per-model code.
//!
//! ## Scope
//!
//! Handoff applies only to pool-backed **Fp16 paged** sequences (the dense-
//! natural-backend families wired to the shared pool: qwen3, llama3). Model-
//! owned-state families and recurrent/hybrid SSM models keep dense or
//! model-owned caches and are structurally excluded upstream
//! (`supports_batching()` / the paged-backend gate), so they never reach this
//! path. This is the in-process building block (B1) for the disaggregated
//! serving role wiring; the CLI role switch and a real network transport land in
//! later steps. The synchronous scaffolding [`HandoffProtocol`] trait
//! (`prefill_scheduler`) is intentionally left for that wiring: it predates the
//! async [`Transport`] and would force a `block_on`, so the async byte bridge
//! here is the primitive the real serve loop should build on.
//!
//! [`HandoffProtocol`]: super::prefill_scheduler::HandoffProtocol

use anyhow::{Result, anyhow, bail};
use bytes::Bytes;

use mlxcel_core::cache::{CachePool, PagedKvLayout, SequenceId, SequenceStateLayout};
use mlxcel_core::generate::LanguageModel;

use crate::distributed::kv_cache_serde::{
    CacheIngestLimits, ExpectedBlockGeometry, SerializableCacheState, SerializableSamplingState,
    deserialize_cache_state_with_limits, restore_into_cache_pool_sequence_anchored,
    serialize_cache_pool_sequence, serialize_cache_state,
};
use crate::distributed::transport::{Transport, TransportMessage};

/// Tensor-id tag carried on the [`TransportMessage::TensorData`] frame that
/// transports a serialized KV cache handoff. The receiver checks it so a
/// mismatched message kind fails loudly instead of being mis-restored.
pub const HANDOFF_TENSOR_ID: &str = "kv-cache-handoff";

/// Build the pool-backed Fp16 paged layout for `num_layers` layers at
/// `block_size` tokens per block, the same `PagedKvLayout::uniform` layout the
/// batch scheduler builds for a dense-natural Fp16 sequence, so an allocated
/// sequence is pool-backed and the handoff restore can materialize real pool
/// blocks.
fn handoff_paged_layout(num_layers: usize, block_size: usize) -> Result<SequenceStateLayout> {
    let layout = PagedKvLayout::uniform(num_layers, block_size, block_size)
        .map_err(|e| anyhow!("handoff: build paged layout: {e}"))?;
    Ok(SequenceStateLayout::paged_kv_cache(layout))
}

/// Derive the decode model's exact per-layer paged KV block geometry by probing
/// it once.
///
/// `ExpectedBlockGeometry` pins every transferred slab to the local model's real
/// `(block_size, n_kv_heads, head_dim, dtype)` before a single pool block is
/// acquired (see [`ExpectedBlockGeometry`]). The dims could be read from config,
/// but the KV dtype is the model's runtime activation dtype and has no static
/// accessor, so this runs a single one-token forward through a throwaway
/// pool-backed sequence and reads the geometry straight off the gathered window
/// (`[1, n_kv_heads, 1, head_dim]`) and its dtype. The throwaway pool is dropped
/// before returning, so there is no lasting cache state; the model is borrowed
/// immutably and its `forward` is pure with respect to the caches passed in, so
/// the probe has no side effects on the caller's model.
///
/// Only valid for pool-backed Fp16 families (the handoff scope); callers must
/// not probe a model-owned or recurrent model.
pub fn probe_block_geometry(
    model: &dyn LanguageModel,
    block_size: usize,
) -> Result<ExpectedBlockGeometry> {
    let num_layers = model.num_layers();
    let layout = handoff_paged_layout(num_layers, block_size)?;

    let mut pool = CachePool::new(1);
    let id = pool
        .allocate_with_layout(model, Some(layout))
        .map_err(|e| anyhow!("handoff geometry probe: allocate: {e}"))?;

    // One-token forward writes layer 0's first pool block, establishing the
    // pool's real geometry and dtype. Skip the forward (and release the slot)
    // if the model is not pool-backed for this layout, so the failure is a clean
    // error rather than a dense write the gather below cannot read.
    let input = mlxcel_core::from_slice_i32(&[0_i32], &[1, 1]);
    let mask = mlxcel_core::utils::create_causal_mask(1, 0);
    let pool_backed = {
        let caches = pool
            .get_caches_mut(id)
            .ok_or_else(|| anyhow!("handoff geometry probe: probe sequence vanished"))?;
        if caches.iter().all(|c| c.is_paged_backed()) {
            let logits = model.forward(&input, caches, Some(&mask));
            mlxcel_core::eval(&logits);
            true
        } else {
            false
        }
    };
    if !pool_backed {
        pool.release(id);
        bail!(
            "handoff geometry probe: model is not pool-backed for the Fp16 paged layout; \
             only dense-natural Fp16 families support paged handoff"
        );
    }

    let geometry = {
        let block_pool = pool
            .paged_pool_ref()
            .ok_or_else(|| anyhow!("handoff geometry probe: paged pool missing"))?;
        let seq = pool
            .get(id)
            .ok_or_else(|| anyhow!("handoff geometry probe: probe sequence vanished"))?;
        let state = seq
            .paged_state()
            .ok_or_else(|| anyhow!("handoff geometry probe: probe sequence has no paged state"))?;
        let (k, _v) = block_pool
            .gather_visible(&state, 0)
            .map_err(|e| anyhow!("handoff geometry probe: gather_visible: {e}"))?
            .ok_or_else(|| anyhow!("handoff geometry probe: gather_visible returned no window"))?;
        // `[1, n_kv_heads, visible_len, head_dim]`.
        let shape = mlxcel_core::array_shape(&k);
        if shape.len() != 4 {
            bail!(
                "handoff geometry probe: unexpected gathered window rank {} (shape {shape:?})",
                shape.len()
            );
        }
        ExpectedBlockGeometry {
            num_layers,
            block_size,
            n_kv_heads: shape[1],
            head_dim: shape[3],
            dtype: mlxcel_core::array_dtype(&k),
        }
    };

    pool.release(id);
    Ok(geometry)
}

/// Prefill side: serialize a finished pool-backed sequence into a single wire
/// frame ready for transport.
///
/// Captures the dense metadata, the paged block table, and the referenced pool
/// block CONTENTS (via [`serialize_cache_pool_sequence`]), then encodes the
/// whole `SerializableCacheState` to bytes. `token_history` is the sequence's
/// prompt token ids (needed so the decode node can continue sampling with the
/// same context); `sampling` carries the request's sampling parameters when the
/// caller tracks them; `generated_tokens` carries the prefill node's first
/// sampled token(s) so the decode node seeds its continuation correctly (#126
/// B2b) and the emitted stream matches a single-node run.
pub fn extract_sequence_handoff(
    cache_pool: &CachePool,
    id: SequenceId,
    sampling: Option<SerializableSamplingState>,
    token_history: Vec<i32>,
    generated_tokens: Vec<i32>,
) -> Result<Vec<u8>> {
    let mut state = serialize_cache_pool_sequence(cache_pool, id, sampling, token_history)?;
    state.generated_tokens = generated_tokens;
    serialize_cache_state(&state)
}

/// Decode side: validate an incoming handoff frame and reconstruct the sequence
/// on a fresh pool-backed slot, returning the new local sequence id.
///
/// Steps: reject oversized frames up front (`limits`), allocate a pool-backed
/// Fp16 paged sequence for the local model, then restore the KV onto fresh pool
/// blocks anchored to the local model's `geometry` (rejecting any slab that does
/// not match the decode model before a pool block is acquired). On restore
/// failure the freshly allocated slot is released so a rejected handoff leaves
/// no leaked sequence or blocks.
pub fn ingest_sequence_handoff(
    cache_pool: &mut CachePool,
    model: &dyn LanguageModel,
    bytes: &[u8],
    limits: &CacheIngestLimits,
    geometry: &ExpectedBlockGeometry,
    block_size: usize,
) -> Result<SequenceId> {
    let state = deserialize_cache_state_with_limits(bytes, limits)?;
    ingest_sequence_handoff_state(cache_pool, model, &state, limits, geometry, block_size)
}

/// Decode side, pre-deserialized variant: reconstruct a sequence from an already
/// parsed [`SerializableCacheState`] (otherwise identical to
/// [`ingest_sequence_handoff`]).
///
/// A serving-role scheduler that also needs the handoff's request context (the
/// prompt token history and the prefill node's generated tokens, to seed a live
/// decode sequence) deserializes the frame once, reads that context, and calls
/// this so the heavy paged-block restore is not paid for twice.
pub fn ingest_sequence_handoff_state(
    cache_pool: &mut CachePool,
    model: &dyn LanguageModel,
    state: &SerializableCacheState,
    limits: &CacheIngestLimits,
    geometry: &ExpectedBlockGeometry,
    block_size: usize,
) -> Result<SequenceId> {
    let layout = handoff_paged_layout(model.num_layers(), block_size)?;
    let seq_id = cache_pool
        .allocate_with_layout(model, Some(layout))
        .map_err(|e| anyhow!("handoff ingest: allocate decode sequence: {e}"))?;

    if let Err(e) =
        restore_into_cache_pool_sequence_anchored(state, cache_pool, seq_id, limits, Some(geometry))
    {
        // Release the just-allocated slot so a rejected handoff does not leak a
        // sequence (or any pool blocks the partial restore acquired).
        cache_pool.release(seq_id);
        return Err(e);
    }

    Ok(seq_id)
}

/// Async byte bridge (send): frame `payload` as a handoff [`TransportMessage`]
/// and send it to `peer` over `transport`.
pub async fn send_handoff_payload(
    transport: &dyn Transport,
    peer: &str,
    payload: &[u8],
) -> Result<()> {
    let message = TransportMessage::TensorData {
        tensor_id: HANDOFF_TENSOR_ID.to_string(),
        shape: vec![payload.len()],
        data: Bytes::copy_from_slice(payload),
    };
    transport.send(peer, message).await
}

/// Async byte bridge (receive): take the next inbound message from `transport`
/// and return the sender address with the raw handoff payload, rejecting any
/// message that is not a handoff frame.
pub async fn recv_handoff_payload(transport: &dyn Transport) -> Result<(String, Vec<u8>)> {
    let (from, message) = transport.recv().await?;
    match message {
        TransportMessage::TensorData {
            tensor_id, data, ..
        } => {
            if tensor_id == HANDOFF_TENSOR_ID {
                Ok((from, data.to_vec()))
            } else {
                bail!(
                    "expected a '{HANDOFF_TENSOR_ID}' handoff frame, got TensorData('{tensor_id}')"
                )
            }
        }
        TransportMessage::Control { operation, .. } => {
            bail!("expected a '{HANDOFF_TENSOR_ID}' handoff frame, got Control('{operation}')")
        }
    }
}

#[cfg(test)]
#[path = "handoff_impl_tests.rs"]
mod tests;
