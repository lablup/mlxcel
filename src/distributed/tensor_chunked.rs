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

//! Chunked streaming transfer for large tensors.
//!
//! When a tensor is too large to transfer in a single message (or when
//! streaming is desired for pipeline overlap), this module splits the
//! serialized tensor into fixed-size chunks that can be sent individually
//! and reassembled on the receiver side.
//!
//! Used by: distributed inference (large weight shard transfers,
//! KV cache migration).

use anyhow::{Result, bail};

use super::tensor_protocol::{
    ChunkFrame, DEFAULT_CHUNK_SIZE, PROTOCOL_VERSION, TensorDtype, TensorFlags, TensorHeader,
    TensorKind,
};

/// Configuration for chunked tensor transfer.
#[derive(Debug, Clone)]
pub struct ChunkedTransferConfig {
    /// Size of each chunk in bytes.
    pub chunk_size: usize,
    /// Whether to attempt LZ4 compression on individual chunks.
    pub compress_chunks: bool,
}

impl Default for ChunkedTransferConfig {
    fn default() -> Self {
        Self {
            chunk_size: DEFAULT_CHUNK_SIZE,
            compress_chunks: false,
        }
    }
}

/// A chunked tensor ready for streaming transfer.
///
/// Contains the header (sent once) and a sequence of chunk frames.
#[derive(Debug)]
pub struct ChunkedTensor {
    /// The tensor header with the CHUNKED flag set.
    pub header: TensorHeader,
    /// Encoded header bytes (sent as the first message).
    pub header_bytes: Vec<u8>,
    /// Chunk frames to be sent sequentially.
    pub chunks: Vec<ChunkFrame>,
}

impl ChunkedTensor {
    /// Total number of chunks.
    pub fn num_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Total data bytes across all chunks.
    pub fn total_data_bytes(&self) -> usize {
        self.chunks.iter().map(|c| c.data.len()).sum()
    }
}

/// Split a serialized tensor payload into chunks for streaming transfer.
///
/// The `data` must be the raw (possibly quantized/compressed) tensor payload.
/// The function creates a header with the CHUNKED flag and splits the data
/// into fixed-size chunk frames.
pub fn split_into_chunks(
    dtype: TensorDtype,
    shape: &[u64],
    data: &[u8],
    kind: TensorKind,
    config: &ChunkedTransferConfig,
) -> Result<ChunkedTensor> {
    if config.chunk_size == 0 {
        bail!("chunk size must be > 0");
    }

    let mut flags = TensorFlags::new();
    flags.set(TensorFlags::CHUNKED);

    let header = TensorHeader {
        version: PROTOCOL_VERSION,
        flags,
        kind,
        dtype,
        shape: shape.to_vec(),
        metadata: vec![],
        data_len: data.len() as u64,
    };

    let header_bytes = header.encode();

    let total_chunks = data.len().div_ceil(config.chunk_size);
    let total_chunks = total_chunks.max(1); // At least one chunk even for empty data.

    let mut chunks = Vec::with_capacity(total_chunks);
    for i in 0..total_chunks {
        let start = i * config.chunk_size;
        let end = (start + config.chunk_size).min(data.len());
        let chunk_data = if start < data.len() {
            data[start..end].to_vec()
        } else {
            vec![]
        };

        chunks.push(ChunkFrame {
            chunk_index: i as u32,
            total_chunks: total_chunks as u32,
            data: chunk_data,
        });
    }

    Ok(ChunkedTensor {
        header,
        header_bytes,
        chunks,
    })
}

/// Reassembly buffer for receiving chunked tensor transfers.
#[derive(Debug)]
pub struct ChunkAssembler {
    /// The header received at the start of the transfer.
    header: TensorHeader,
    /// Expected total number of chunks.
    total_chunks: u32,
    /// Received chunks indexed by chunk_index.
    received: Vec<Option<Vec<u8>>>,
    /// Number of chunks received so far.
    count: u32,
}

impl ChunkAssembler {
    /// Create a new assembler from the header of a chunked transfer.
    ///
    /// The `total_chunks` is typically learned from the first ChunkFrame
    /// received. Call [`set_total_chunks`] once known.
    pub fn new(header: TensorHeader) -> Self {
        Self {
            header,
            total_chunks: 0,
            received: Vec::new(),
            count: 0,
        }
    }

    /// Set the total expected chunks (learned from the first ChunkFrame).
    pub fn set_total_chunks(&mut self, total: u32) {
        self.total_chunks = total;
        self.received.resize_with(total as usize, || None);
    }

    /// Insert a received chunk. Returns `true` if all chunks are now present.
    pub fn insert(&mut self, frame: ChunkFrame) -> Result<bool> {
        if self.total_chunks == 0 {
            self.set_total_chunks(frame.total_chunks);
        }

        if frame.chunk_index >= self.total_chunks {
            bail!(
                "chunk index {} out of range (total: {})",
                frame.chunk_index,
                self.total_chunks
            );
        }

        let idx = frame.chunk_index as usize;
        if self.received[idx].is_none() {
            self.received[idx] = Some(frame.data);
            self.count += 1;
        }

        Ok(self.is_complete())
    }

    /// Whether all chunks have been received.
    pub fn is_complete(&self) -> bool {
        self.total_chunks > 0 && self.count >= self.total_chunks
    }

    /// Number of chunks received so far.
    pub fn received_count(&self) -> u32 {
        self.count
    }

    /// Progress as a fraction [0.0, 1.0].
    pub fn progress(&self) -> f64 {
        if self.total_chunks == 0 {
            0.0
        } else {
            self.count as f64 / self.total_chunks as f64
        }
    }

    /// Assemble all chunks into the complete tensor data.
    ///
    /// Returns an error if not all chunks have been received.
    pub fn assemble(self) -> Result<(TensorHeader, Vec<u8>)> {
        if !self.is_complete() {
            bail!(
                "cannot assemble: only {}/{} chunks received",
                self.count,
                self.total_chunks
            );
        }

        let total_len: usize = self
            .received
            .iter()
            .map(|c| c.as_ref().map(|d| d.len()).unwrap_or(0))
            .sum();

        let mut data = Vec::with_capacity(total_len);
        for d in self.received.iter().flatten() {
            data.extend_from_slice(d);
        }

        Ok((self.header, data))
    }
}

/// Convenience function: check whether a tensor should use chunked transfer
/// based on its size.
///
/// Returns `true` if the data exceeds the default chunk size threshold
/// (i.e., would require more than one chunk).
pub fn should_chunk(data_len: usize, chunk_size: usize) -> bool {
    data_len > chunk_size
}

#[cfg(test)]
#[path = "tensor_chunked_tests.rs"]
mod tests;
