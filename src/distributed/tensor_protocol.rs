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

//! Tensor wire-format definitions for the distributed transfer protocol.
//!
//! # Wire Format (little-endian)
//!
//! ```text
//! [version: u8]           // Protocol version (currently 1)
//! [flags: u8]             // Bit flags (see TensorFlags)
//! [tensor_kind: u8]       // KVCache=0, Activation=1, WeightShard=2
//! [dtype: u8]             // Original dtype of the tensor
//! [ndim: u32 LE]          // Number of dimensions
//! [shape: ndim * u64 LE]  // Shape dimensions
//! [metadata_len: u32 LE]  // Optional metadata length (0 if none)
//! [metadata: bytes]       // Optional JSON metadata
//! [data_len: u64 LE]      // Payload length (after compression/quantization)
//! [data: bytes]           // Tensor data (raw, compressed, or quantized)
//! ```
//!
//! For chunked transfers, the `CHUNKED` flag is set and the data section is
//! replaced by a sequence of chunk frames:
//!
//! ```text
//! [chunk_index: u32 LE]   // 0-based chunk index
//! [total_chunks: u32 LE]  // Total number of chunks
//! [chunk_len: u32 LE]     // Length of this chunk's data
//! [chunk_data: bytes]     // Chunk payload
//! ```

use std::fmt;

use anyhow::{Context, Result, bail};

/// Current protocol version.
pub const PROTOCOL_VERSION: u8 = 1;

/// Default chunk size for large tensor transfers (1 MiB).
pub const DEFAULT_CHUNK_SIZE: usize = 1024 * 1024;

/// Maximum allowed number of dimensions in a tensor header.
/// Guards against malicious inputs that would cause integer overflow
/// in size calculations (ndim * 8 must fit in usize).
const MAX_NDIM: usize = 32;

/// Maximum allowed metadata length in bytes (16 MiB).
/// Prevents allocation of excessively large buffers from crafted headers.
const MAX_METADATA_LEN: usize = 16 * 1024 * 1024;

/// Minimum compression ratio (compressed / original) below which we apply
/// LZ4. If compression yields a ratio above this threshold the data is sent
/// uncompressed.
pub const COMPRESSION_RATIO_THRESHOLD: f64 = 0.90;

/// Bit flags encoded in the header's `flags` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TensorFlags(u8);

impl TensorFlags {
    /// Data is LZ4-compressed.
    pub const COMPRESSED: u8 = 0b0000_0001;
    /// Data has been quantized during transfer.
    pub const QUANTIZED: u8 = 0b0000_0010;
    /// Transfer uses chunked streaming.
    pub const CHUNKED: u8 = 0b0000_0100;

    pub fn new() -> Self {
        Self(0)
    }

    pub fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    pub fn bits(self) -> u8 {
        self.0
    }

    pub fn set(&mut self, flag: u8) {
        self.0 |= flag;
    }

    pub fn is_set(self, flag: u8) -> bool {
        self.0 & flag != 0
    }

    pub fn is_compressed(self) -> bool {
        self.is_set(Self::COMPRESSED)
    }

    pub fn is_quantized(self) -> bool {
        self.is_set(Self::QUANTIZED)
    }

    pub fn is_chunked(self) -> bool {
        self.is_set(Self::CHUNKED)
    }
}

/// The kind of tensor being transferred, which determines how the receiver
/// should handle reconstruction.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TensorKind {
    /// KV cache tensor for disaggregated inference (DI).
    KVCache = 0,
    /// Activation tensor for pipeline parallelism (PP).
    Activation = 1,
    /// Weight shard tensor for tensor parallelism (TP).
    WeightShard = 2,
}

impl TryFrom<u8> for TensorKind {
    type Error = anyhow::Error;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::KVCache),
            1 => Ok(Self::Activation),
            2 => Ok(Self::WeightShard),
            other => bail!("unknown tensor kind: {other}"),
        }
    }
}

impl fmt::Display for TensorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KVCache => write!(f, "KVCache"),
            Self::Activation => write!(f, "Activation"),
            Self::WeightShard => write!(f, "WeightShard"),
        }
    }
}

/// Data type of tensor elements, matching common MLX dtypes.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TensorDtype {
    Float32 = 0,
    Float16 = 1,
    BFloat16 = 2,
    Int32 = 3,
    Int16 = 4,
    Int8 = 5,
    UInt8 = 6,
    Bool = 7,
    /// 4-bit quantized (packed, 2 elements per byte).
    Int4 = 8,
}

impl TensorDtype {
    /// Size of one element in bytes. For `Int4` returns 0 since elements
    /// are sub-byte packed.
    pub fn element_size(self) -> usize {
        match self {
            Self::Float32 | Self::Int32 => 4,
            Self::Float16 | Self::BFloat16 | Self::Int16 => 2,
            Self::Int8 | Self::UInt8 | Self::Bool => 1,
            Self::Int4 => 0, // Sub-byte; callers must handle packing.
        }
    }
}

impl TryFrom<u8> for TensorDtype {
    type Error = anyhow::Error;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            0 => Ok(Self::Float32),
            1 => Ok(Self::Float16),
            2 => Ok(Self::BFloat16),
            3 => Ok(Self::Int32),
            4 => Ok(Self::Int16),
            5 => Ok(Self::Int8),
            6 => Ok(Self::UInt8),
            7 => Ok(Self::Bool),
            8 => Ok(Self::Int4),
            other => bail!("unknown tensor dtype: {other}"),
        }
    }
}

impl fmt::Display for TensorDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Float32 => write!(f, "float32"),
            Self::Float16 => write!(f, "float16"),
            Self::BFloat16 => write!(f, "bfloat16"),
            Self::Int32 => write!(f, "int32"),
            Self::Int16 => write!(f, "int16"),
            Self::Int8 => write!(f, "int8"),
            Self::UInt8 => write!(f, "uint8"),
            Self::Bool => write!(f, "bool"),
            Self::Int4 => write!(f, "int4"),
        }
    }
}

/// Quantization mode for on-the-fly transfer quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMode {
    /// No quantization (bit-exact transfer).
    None,
    /// Quantize float16 to int8 with per-group absmax scaling.
    Int8,
    /// Quantize float16 to int4 with per-group absmax scaling (packed).
    Int4,
}

/// Fixed-size header preceding tensor data on the wire.
///
/// After the header, `metadata_len` bytes of optional JSON metadata follow,
/// then `data_len` bytes of payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorHeader {
    pub version: u8,
    pub flags: TensorFlags,
    pub kind: TensorKind,
    pub dtype: TensorDtype,
    pub shape: Vec<u64>,
    pub metadata: Vec<u8>,
    pub data_len: u64,
}

impl TensorHeader {
    /// Size of the fixed portion of the header (before variable-length shape,
    /// metadata, and data).
    ///
    /// `version(1) + flags(1) + kind(1) + dtype(1) + ndim(4) = 8 bytes`
    pub const FIXED_SIZE: usize = 8;

    /// Compute the total number of elements described by the shape.
    ///
    /// Returns 0 for an empty (scalar) shape. Uses checked arithmetic
    /// to avoid silent overflow on malicious inputs.
    pub fn num_elements(&self) -> u64 {
        if self.shape.is_empty() {
            return 0;
        }
        self.shape
            .iter()
            .copied()
            .try_fold(1u64, u64::checked_mul)
            .unwrap_or(u64::MAX)
    }

    /// Encode the header into wire bytes (little-endian).
    pub fn encode(&self) -> Vec<u8> {
        let ndim = u32::try_from(self.shape.len()).expect("shape ndim exceeds u32");
        let metadata_len = u32::try_from(self.metadata.len()).expect("metadata length exceeds u32");
        // Fixed (8) + shape (ndim*8) + metadata_len (4) + metadata + data_len (8)
        let total = Self::FIXED_SIZE + (ndim as usize) * 8 + 4 + self.metadata.len() + 8;
        let mut buf = Vec::with_capacity(total);

        buf.push(self.version);
        buf.push(self.flags.bits());
        buf.push(self.kind as u8);
        buf.push(self.dtype as u8);
        buf.extend_from_slice(&ndim.to_le_bytes());
        for &dim in &self.shape {
            buf.extend_from_slice(&dim.to_le_bytes());
        }
        buf.extend_from_slice(&metadata_len.to_le_bytes());
        buf.extend_from_slice(&self.metadata);
        buf.extend_from_slice(&self.data_len.to_le_bytes());

        buf
    }

    /// Decode a header from wire bytes (little-endian).
    ///
    /// Returns the header and the number of bytes consumed.
    pub fn decode(buf: &[u8]) -> Result<(Self, usize)> {
        if buf.len() < Self::FIXED_SIZE {
            bail!(
                "header too short: need at least {} bytes, got {}",
                Self::FIXED_SIZE,
                buf.len()
            );
        }

        let version = buf[0];
        if version != PROTOCOL_VERSION {
            bail!("unsupported protocol version: {version} (expected {PROTOCOL_VERSION})");
        }

        let flags = TensorFlags::from_bits(buf[1]);
        let kind = TensorKind::try_from(buf[2]).context("invalid tensor kind")?;
        let dtype = TensorDtype::try_from(buf[3]).context("invalid tensor dtype")?;
        let ndim = u32::from_le_bytes(buf[4..8].try_into()?) as usize;

        if ndim > MAX_NDIM {
            bail!("ndim {ndim} exceeds maximum allowed ({MAX_NDIM})");
        }

        // Safe: ndim <= MAX_NDIM (32), so ndim * 8 <= 256, no overflow.
        let shape_end = Self::FIXED_SIZE + ndim * 8;
        if buf.len() < shape_end + 4 {
            bail!("header too short for shape + metadata_len");
        }

        let mut shape = Vec::with_capacity(ndim);
        for i in 0..ndim {
            let offset = Self::FIXED_SIZE + i * 8;
            let dim = u64::from_le_bytes(buf[offset..offset + 8].try_into()?);
            shape.push(dim);
        }

        let metadata_len = u32::from_le_bytes(buf[shape_end..shape_end + 4].try_into()?) as usize;
        if metadata_len > MAX_METADATA_LEN {
            bail!("metadata length {metadata_len} exceeds maximum allowed ({MAX_METADATA_LEN})");
        }
        let metadata_end = shape_end
            .checked_add(4)
            .and_then(|v| v.checked_add(metadata_len))
            .ok_or_else(|| anyhow::anyhow!("metadata offset overflow"))?;
        if buf.len() < metadata_end + 8 {
            bail!("header too short for metadata + data_len");
        }

        let metadata = buf[shape_end + 4..metadata_end].to_vec();
        let data_len = u64::from_le_bytes(buf[metadata_end..metadata_end + 8].try_into()?);

        let consumed = metadata_end + 8;

        Ok((
            Self {
                version,
                flags,
                kind,
                dtype,
                shape,
                metadata,
                data_len,
            },
            consumed,
        ))
    }
}

/// A single chunk in a chunked transfer.
#[derive(Debug, Clone)]
pub struct ChunkFrame {
    pub chunk_index: u32,
    pub total_chunks: u32,
    pub data: Vec<u8>,
}

impl ChunkFrame {
    /// Encode this chunk frame to wire bytes.
    pub fn encode(&self) -> Vec<u8> {
        let chunk_len = self.data.len() as u32;
        let mut buf = Vec::with_capacity(12 + self.data.len());
        buf.extend_from_slice(&self.chunk_index.to_le_bytes());
        buf.extend_from_slice(&self.total_chunks.to_le_bytes());
        buf.extend_from_slice(&chunk_len.to_le_bytes());
        buf.extend_from_slice(&self.data);
        buf
    }

    /// Decode a chunk frame from wire bytes.
    pub fn decode(buf: &[u8]) -> Result<Self> {
        if buf.len() < 12 {
            bail!("chunk frame too short: need 12 bytes, got {}", buf.len());
        }
        let chunk_index = u32::from_le_bytes(buf[0..4].try_into()?);
        let total_chunks = u32::from_le_bytes(buf[4..8].try_into()?);
        let chunk_len = u32::from_le_bytes(buf[8..12].try_into()?) as usize;
        if buf.len() < 12 + chunk_len {
            bail!(
                "chunk frame data truncated: expected {chunk_len} bytes, got {}",
                buf.len() - 12
            );
        }
        Ok(Self {
            chunk_index,
            total_chunks,
            data: buf[12..12 + chunk_len].to_vec(),
        })
    }
}

#[cfg(test)]
#[path = "tensor_protocol_tests.rs"]
mod tests;
