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

//! Rust-native serializable representations of KV cache state.
//!
//! These types bridge the gap between the FFI-bound cache types
//! (`KVCache`, `RotatingKVCache`, `ChunkedKVCache`) and the wire format.
//! They contain only plain Rust data (no `UniquePtr<MlxArray>`) and can
//! be freely serialized, sent over the network, and deserialized.

use serde::{Deserialize, Serialize};

use super::super::tensor_protocol::TensorDtype;
use mlxcel_core::cache::{
    PagedBlockId, PagedKvLayout, PagedLayerState, PagedSequenceState, SequenceStateBackend,
};

/// Legacy cache serialization format version.
pub const CACHE_FORMAT_VERSION_V1: u8 = 1;
/// Current cache serialization format version.
pub const CACHE_FORMAT_VERSION_V2: u8 = 2;

/// Discriminant for the cache variant being serialized.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CacheType {
    /// Standard KV cache (pre-allocated buffer with offset).
    Standard = 0,
    /// Rotating (sliding-window) KV cache.
    Rotating = 1,
    /// Chunked KV cache (Llama 4 iGQA).
    Chunked = 2,
}

/// Runtime storage backend associated with one serialized sequence.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum SerializableSequenceBackend {
    /// Standard dense per-layer KV cache storage.
    #[default]
    DenseKvCache = 0,
    /// Dense compatibility caches plus mirrored paged block-table state.
    PagedKvCache = 1,
    /// Model-owned/internal state (no external dense KV ownership guarantee).
    ModelOwned = 2,
}

impl SerializableSequenceBackend {
    pub fn from_runtime(backend: SequenceStateBackend) -> Self {
        match backend {
            SequenceStateBackend::DenseKvCache => Self::DenseKvCache,
            SequenceStateBackend::PagedKvCache => Self::PagedKvCache,
            SequenceStateBackend::ModelOwned => Self::ModelOwned,
        }
    }
}

/// One paged layer's logical-to-physical mapping in transfer-safe form.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SerializablePagedLayerState {
    pub block_ids: Vec<u64>,
    pub len: usize,
    pub logical_start: usize,
}

impl SerializablePagedLayerState {
    pub fn from_runtime(layer: &PagedLayerState) -> Self {
        Self {
            block_ids: layer.block_ids.iter().map(|block| block.as_u64()).collect(),
            len: layer.len,
            logical_start: layer.logical_start,
        }
    }

    pub fn to_runtime(&self) -> PagedLayerState {
        PagedLayerState {
            block_ids: self
                .block_ids
                .iter()
                .copied()
                .map(PagedBlockId::from_raw)
                .collect(),
            len: self.len,
            logical_start: self.logical_start.min(self.len),
        }
    }
}

/// Paged KV metadata that must survive prefill/decode transfer boundaries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SerializablePagedSequenceState {
    pub block_size: usize,
    pub bytes_per_block: Vec<usize>,
    pub layers: Vec<SerializablePagedLayerState>,
}

impl SerializablePagedSequenceState {
    pub fn from_runtime(state: &PagedSequenceState, layout: &PagedKvLayout) -> Self {
        Self {
            block_size: state.block_size,
            bytes_per_block: layout.bytes_per_block.clone(),
            layers: state
                .layers
                .iter()
                .map(SerializablePagedLayerState::from_runtime)
                .collect(),
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        let layout = self.layout()?;
        if self.layers.len() != layout.num_layers {
            anyhow::bail!(
                "paged sequence layer count mismatch: state has {}, layout has {}",
                self.layers.len(),
                layout.num_layers
            );
        }

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            if layer.logical_start > layer.len {
                anyhow::bail!(
                    "paged layer {layer_idx} logical_start {} exceeds len {}",
                    layer.logical_start,
                    layer.len
                );
            }
            let visible_len = layer.len.saturating_sub(layer.logical_start);
            let required_blocks = visible_len.div_ceil(layout.block_size);
            if layer.block_ids.len() < required_blocks {
                anyhow::bail!(
                    "paged layer {layer_idx} has {} blocks for visible length {}, requires at least {}",
                    layer.block_ids.len(),
                    visible_len,
                    required_blocks
                );
            }
        }

        Ok(())
    }

    pub fn layout(&self) -> anyhow::Result<PagedKvLayout> {
        PagedKvLayout::new(self.block_size, self.bytes_per_block.clone())
            .map_err(anyhow::Error::msg)
    }

    pub fn to_runtime(&self) -> anyhow::Result<PagedSequenceState> {
        self.validate()?;
        Ok(PagedSequenceState {
            block_size: self.block_size,
            layers: self
                .layers
                .iter()
                .map(SerializablePagedLayerState::to_runtime)
                .collect(),
        })
    }
}

impl TryFrom<u8> for CacheType {
    type Error = anyhow::Error;

    fn try_from(value: u8) -> anyhow::Result<Self> {
        match value {
            0 => Ok(Self::Standard),
            1 => Ok(Self::Rotating),
            2 => Ok(Self::Chunked),
            other => anyhow::bail!("unknown cache type: {other}"),
        }
    }
}

/// Raw tensor data extracted from an `MlxArray`.
///
/// Contains the byte contents, shape, and dtype needed to reconstruct
/// the tensor on the receiving side via `ffi::from_bytes`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawTensorData {
    /// Raw bytes of the evaluated, contiguous tensor.
    pub data: Vec<u8>,
    /// Shape of the tensor (e.g. `[batch, n_kv_heads, seq_len, head_dim]`).
    pub shape: Vec<i32>,
    /// MLX dtype code (matches `mlxcel_core::dtype` constants).
    pub dtype: i32,
}

/// One layer's key/value pair in serializable form.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableCacheEntry {
    /// Serialized key tensor (None if cache layer is empty).
    pub keys: Option<RawTensorData>,
    /// Serialized value tensor (None if cache layer is empty).
    pub values: Option<RawTensorData>,
}

/// One paged pool block's K/V contents in serializable form (#125).
///
/// Carries the ORIGIN node's physical `block_id` + `layer_idx` and the block's
/// full `[block_size, n_kv_heads, head_dim]` K and V slabs as [`RawTensorData`].
/// The decode node reconstructs each slab, acquires a FRESH pool block, and
/// remaps `block_id` to the new physical id so block accounting matches the
/// origin. Like the dense `entries`, the tensor bytes ride inside the JSON
/// metadata frame; a compact binary framing is deferred to the #126 capstone.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializablePagedBlock {
    /// Origin-node physical block id (remapped on restore).
    pub block_id: u64,
    /// Layer the block belongs to.
    pub layer_idx: usize,
    /// Block's full K slab `[block_size, n_kv_heads, head_dim]`.
    pub keys: RawTensorData,
    /// Block's full V slab `[block_size, n_kv_heads, head_dim]`.
    pub values: RawTensorData,
}

/// Metadata for the serialized cache state.
///
/// Serialized as JSON within the wire format so it can evolve without
/// breaking the binary framing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetadata {
    /// Number of prompt tokens that were originally prefilled.
    pub prompt_len: usize,
    /// Current generation offset (total tokens processed).
    pub current_offset: i32,
    /// Number of layers in the cache.
    pub num_layers: usize,
    /// Per-layer offsets (for Standard/Chunked caches).
    pub layer_offsets: Vec<i32>,

    // -- Rotating-specific fields --
    /// Maximum window size (only meaningful for Rotating caches).
    pub max_size: Option<i32>,
    /// Per-layer write indices (only meaningful for Rotating caches).
    pub layer_indices: Option<Vec<i32>>,

    // -- Chunked-specific fields --
    /// Chunk size (only meaningful for Chunked caches).
    pub chunk_size: Option<i32>,
    /// Per-layer start positions (only meaningful for Chunked caches).
    pub start_positions: Option<Vec<i32>>,
}

/// Serializable sampling state, mirroring the live `SamplingConfig` fields
/// that are relevant to decode continuation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableSamplingState {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: Option<u64>,
    pub repetition_penalty: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: usize,
    pub dry_penalty_last_n: usize,
    pub dry_sequence_breakers: Vec<i32>,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub stop_token_ids: Vec<i32>,
}

/// Complete serializable cache state for one sequence.
///
/// This is the top-level type that gets serialized to the wire format
/// and transferred from prefill node to decode node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableCacheState {
    /// Cache variant discriminant.
    pub cache_type: CacheType,
    /// Per-layer cache entries.
    pub entries: Vec<SerializableCacheEntry>,
    /// Cache and sequence metadata.
    pub metadata: CacheMetadata,
    /// Sampling parameters for decode continuation.
    pub sampling_state: Option<SerializableSamplingState>,
    /// Token history for repetition/DRY penalties.
    pub token_history: Vec<i32>,
    /// Unique sequence ID (from CachePool).
    pub sequence_id: u64,
    /// Logical runtime backend that owned this sequence at transfer time.
    #[serde(default)]
    pub sequence_backend: SerializableSequenceBackend,
    /// Mirrored paged sequence state for paged-backed decode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paged_state: Option<SerializablePagedSequenceState>,
    /// Transferred paged pool block CONTENTS for a pool-backed cross-node
    /// handoff (#125). Empty for dense / metadata-only payloads (kept
    /// wire-compatible via `serde(default)` + `skip_serializing_if`), populated
    /// when a true pool-backed sequence is serialized. The tensor bytes ride
    /// inside this JSON metadata frame, same as the dense `entries`; a compact
    /// binary framing is deferred to #126.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub paged_blocks: Vec<SerializablePagedBlock>,
}

/// Convert MLX dtype code to `TensorDtype` for the tensor protocol.
///
/// Used when serializing cache tensors through the existing tensor
/// wire format.
pub fn mlx_dtype_to_tensor_dtype(mlx_dtype: i32) -> anyhow::Result<TensorDtype> {
    // MLX dtype codes from mlxcel_core::dtype
    match mlx_dtype {
        0 => Ok(TensorDtype::Bool),      // BOOL
        5 => Ok(TensorDtype::Int8),      // INT8
        6 => Ok(TensorDtype::Int16),     // INT16
        7 => Ok(TensorDtype::Int32),     // INT32
        9 => Ok(TensorDtype::Float16),   // FLOAT16
        10 => Ok(TensorDtype::Float32),  // FLOAT32
        12 => Ok(TensorDtype::BFloat16), // BFLOAT16
        other => anyhow::bail!("unsupported MLX dtype code for serialization: {other}"),
    }
}

/// Return the element size in bytes for an MLX dtype code.
///
/// Used by: deserialization validation to verify tensor data buffer sizes.
pub fn mlx_dtype_element_size(mlx_dtype: i32) -> anyhow::Result<usize> {
    match mlx_dtype {
        0 => Ok(1),              // BOOL
        1 => Ok(1),              // UINT8
        5 => Ok(1),              // INT8
        2 | 6 | 9 | 12 => Ok(2), // UINT16, INT16, FLOAT16, BFLOAT16
        3 | 7 | 10 => Ok(4),     // UINT32, INT32, FLOAT32
        4 | 8 | 11 => Ok(8),     // UINT64, INT64, FLOAT64
        13 => Ok(8),             // COMPLEX64
        other => anyhow::bail!("unknown MLX dtype code: {other}"),
    }
}

/// Validate that a `RawTensorData` is internally consistent.
///
/// Checks:
/// - All shape dimensions are non-negative.
/// - `data.len()` matches the expected byte count from shape and dtype.
///
/// Used by: deserialization to reject malformed tensors before passing
/// to `ffi::from_bytes`.
pub fn validate_raw_tensor(tensor: &RawTensorData) -> anyhow::Result<()> {
    // Reject negative shape dimensions
    for (i, &dim) in tensor.shape.iter().enumerate() {
        if dim < 0 {
            anyhow::bail!("invalid shape dimension at axis {i}: {dim} (must be non-negative)");
        }
    }

    let element_size = mlx_dtype_element_size(tensor.dtype)?;

    let num_elements: usize = tensor.shape.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim as usize)
            .ok_or_else(|| anyhow::anyhow!("shape overflow computing total elements"))
    })?;

    let expected_bytes = num_elements
        .checked_mul(element_size)
        .ok_or_else(|| anyhow::anyhow!("byte count overflow for tensor"))?;

    if tensor.data.len() != expected_bytes {
        anyhow::bail!(
            "tensor data size mismatch: shape {:?} with dtype {} expects {} bytes, got {}",
            tensor.shape,
            tensor.dtype,
            expected_bytes,
            tensor.data.len()
        );
    }

    Ok(())
}

/// Bounds applied to a cache payload accepted from an (untrusted) peer node, so
/// a malformed or hostile handoff cannot exhaust decode-node memory or smuggle
/// oversized blocks past the frame reader.
///
/// The live disaggregated decode path derives tight values from the model and
/// serving config; the [`Default`] values are generous backstops sized for
/// in-process and test round-trips, not a security boundary on their own. The
/// frame cap is the JSON-bloated wire size (the tensor bytes ride inside the
/// metadata frame as number arrays, roughly 4x their raw size).
#[derive(Debug, Clone, Copy)]
pub struct CacheIngestLimits {
    /// Maximum accepted wire frame size in bytes (whole buffer + metadata len).
    pub max_frame_bytes: usize,
    /// Maximum number of transferred paged blocks in one handoff.
    pub max_paged_blocks: usize,
    /// Maximum bytes for a single block slab (one K or one V slab).
    pub max_block_slab_bytes: usize,
}

impl Default for CacheIngestLimits {
    fn default() -> Self {
        Self {
            // 16 GiB: rejects absurd payloads while leaving headroom for a
            // large model / long context handoff under the bloated JSON framing.
            max_frame_bytes: 16 << 30,
            // 1,048,576 per-layer blocks: far above any realistic single-sequence
            // block table; the decode pool's block budget is the real bound.
            max_paged_blocks: 1 << 20,
            // 64 MiB: one [block_size, n_kv_heads, head_dim] slab is small even
            // for wide-GQA models (e.g. 32 * 128 * 256 * 4 = 4 MiB).
            max_block_slab_bytes: 64 << 20,
        }
    }
}

/// The decode model's real per-layer paged KV block geometry, used to anchor
/// every transferred block slab at the deserialization boundary.
///
/// A fresh decode pool captures its geometry from the FIRST slab it is asked to
/// write ([`mlxcel_core::cache`]'s `PagedBlockPool::write_block` only validates
/// subsequent writes against the captured geometry), so without this anchor a
/// peer could establish a wrong-shaped pool on the first write. Supplying the
/// model's real `(block_size, n_kv_heads, head_dim, dtype)` rejects any slab
/// that does not match before a single pool block is acquired.
#[derive(Debug, Clone, Copy)]
pub struct ExpectedBlockGeometry {
    /// Number of layers the decode model exposes.
    pub num_layers: usize,
    /// Paged pool block size (slots per block).
    pub block_size: usize,
    /// Key/value head count.
    pub n_kv_heads: i32,
    /// Per-head dimension.
    pub head_dim: i32,
    /// MLX dtype code for the pool tensors (e.g. 9 = float16, 12 = bfloat16).
    pub dtype: i32,
}

impl ExpectedBlockGeometry {
    /// The canonical bare layout-A slab shape `[block_size, n_kv_heads, head_dim]`
    /// that `read_block_contents` produces and `acquire_and_write_block` accepts.
    pub fn slab_shape(&self) -> [i32; 3] {
        [self.block_size as i32, self.n_kv_heads, self.head_dim]
    }
}
