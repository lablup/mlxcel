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

//! KV cache serialization and deserialization for disaggregated inference.
//!
//! This module provides the serialization layer that enables transferring
//! KV cache state from prefill nodes to decode nodes in a disaggregated
//! serving pipeline.
//!
//! # Architecture
//!
//! The FFI-bound cache types (`KVCache`, `RotatingKVCache`, `ChunkedKVCache`)
//! contain `UniquePtr<MlxArray>` which cannot be directly serialized. This
//! module defines Rust-native serializable representations and provides
//! conversion functions in both directions:
//!
//! ```text
//! Prefill Node                      Decode Node
//! ┌──────────────┐    wire     ┌──────────────┐
//! │  KVCache     │  ──────>   │  KVCache     │
//! │  (FFI types) │  format    │  (FFI types) │
//! └──────┬───────┘            └──────▲───────┘
//!        │ extract                   │ restore
//! ┌──────▼───────┐            ┌──────┴───────┐
//! │ Serializable │  serialize │ Serializable │
//! │ CacheState   │ ────────> │ CacheState   │
//! └──────────────┘  bytes    └──────────────┘
//! ```
//!
//! # Wire Format
//!
//! See [`serialize`] module documentation for the binary format specification.
//! Version `1` remains accepted for dense-only payloads, while version `2`
//! adds explicit paged sequence snapshots and backend metadata.
//!
//! # Supported Cache Types
//!
//! - [`KVCache`](mlxcel_core::cache::KVCache) — standard pre-allocated buffer
//! - [`RotatingKVCache`](mlxcel_core::cache::RotatingKVCache) — sliding window
//! - [`ChunkedKVCache`](mlxcel_core::cache::ChunkedKVCache) — Llama 4 iGQA
//! - paged-backed sequence state mirrored alongside dense compatibility caches

pub mod deserialize;
pub mod serialize;
pub mod types;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

pub use deserialize::{
    deserialize_cache_state, reconstruct_mlx_array, restore_into_cache_pool_sequence,
    restore_into_kv_caches, restore_into_sequence_cache_set,
};
pub use serialize::{
    CACHE_FORMAT_VERSION, extract_chunked_cache_entry, extract_kv_cache_entry,
    extract_rotating_cache_entry, serialize_cache_pool_sequence, serialize_cache_state,
    serialize_sequence_cache_set,
};
pub use types::{
    CACHE_FORMAT_VERSION_V1, CACHE_FORMAT_VERSION_V2, CacheMetadata, CacheType, RawTensorData,
    SerializableCacheEntry, SerializableCacheState, SerializablePagedBlock,
    SerializablePagedLayerState, SerializablePagedSequenceState, SerializableSamplingState,
    SerializableSequenceBackend, validate_raw_tensor,
};
