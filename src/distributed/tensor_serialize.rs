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

//! Tensor serialization and deserialization.
//!
//! Provides [`serialize_tensor`] and [`deserialize_tensor`] for encoding and
//! decoding tensors using the wire format defined in [`super::tensor_protocol`].
//!
//! Used by: distributed inference (KV cache transfer, activation relay,
//! weight shard distribution).

use anyhow::{Result, bail};
use bytes::Bytes;

use super::tensor_compress::{compress_if_beneficial, decompress};
use super::tensor_protocol::{
    PROTOCOL_VERSION, QuantizationMode, TensorDtype, TensorFlags, TensorHeader, TensorKind,
};
use super::tensor_quantize::{dequantize_int4, dequantize_int8, quantize_int4, quantize_int8};

/// Options controlling how a tensor is serialized for transfer.
#[derive(Debug, Clone)]
pub struct SerializeOptions {
    /// Kind of tensor (KVCache, Activation, WeightShard).
    pub kind: TensorKind,
    /// Quantization mode to apply during serialization.
    pub quantization: QuantizationMode,
    /// Whether to attempt LZ4 compression.
    pub compress: bool,
    /// Optional JSON metadata (e.g., layer index, cache sequence length).
    pub metadata: Option<String>,
}

impl Default for SerializeOptions {
    fn default() -> Self {
        Self {
            kind: TensorKind::Activation,
            quantization: QuantizationMode::None,
            compress: false,
            metadata: None,
        }
    }
}

/// A deserialized tensor with its metadata.
#[derive(Debug, Clone)]
pub struct DeserializedTensor {
    /// Original dtype of the tensor before any quantization.
    pub dtype: TensorDtype,
    /// Tensor shape.
    pub shape: Vec<u64>,
    /// Kind of tensor.
    pub kind: TensorKind,
    /// Optional metadata.
    pub metadata: Option<String>,
    /// Raw tensor data in the original dtype (dequantized if necessary).
    pub data: Vec<u8>,
}

/// Serialize a tensor into the wire format.
///
/// The `data` buffer must contain raw element bytes in the given `dtype`
/// layout. The shape must be consistent with the data length.
///
/// Returns the complete wire-format bytes (header + payload).
pub fn serialize_tensor(
    dtype: TensorDtype,
    shape: &[u64],
    data: &[u8],
    options: &SerializeOptions,
) -> Result<Vec<u8>> {
    validate_data_size(dtype, shape, data)?;

    let mut flags = TensorFlags::new();
    let mut payload = data.to_vec();
    let mut wire_dtype = dtype;

    // Step 1: Apply quantization if requested.
    match options.quantization {
        QuantizationMode::None => {}
        QuantizationMode::Int8 => {
            if dtype != TensorDtype::Float16 {
                bail!("int8 quantization requires float16 input, got {dtype}");
            }
            payload = quantize_int8(&payload);
            wire_dtype = TensorDtype::Int8;
            flags.set(TensorFlags::QUANTIZED);
        }
        QuantizationMode::Int4 => {
            if dtype != TensorDtype::Float16 {
                bail!("int4 quantization requires float16 input, got {dtype}");
            }
            payload = quantize_int4(&payload);
            wire_dtype = TensorDtype::Int4;
            flags.set(TensorFlags::QUANTIZED);
        }
    }

    // Step 2: Apply compression if requested and beneficial.
    if options.compress
        && let Some(compressed) = compress_if_beneficial(&payload)
    {
        payload = compressed;
        flags.set(TensorFlags::COMPRESSED);
    }

    let metadata_bytes = options
        .metadata
        .as_ref()
        .map(|m| m.as_bytes().to_vec())
        .unwrap_or_default();

    let header = TensorHeader {
        version: PROTOCOL_VERSION,
        flags,
        kind: options.kind,
        dtype: wire_dtype,
        shape: shape.to_vec(),
        metadata: metadata_bytes,
        data_len: payload.len() as u64,
    };

    let header_bytes = header.encode();
    let mut result = Vec::with_capacity(header_bytes.len() + payload.len());
    result.extend_from_slice(&header_bytes);
    result.extend_from_slice(&payload);

    Ok(result)
}

/// Deserialize a tensor from wire-format bytes.
///
/// If the tensor was quantized during transfer, the data is dequantized
/// back to float16. If compressed, it is decompressed first.
///
/// Returns the deserialized tensor and the number of bytes consumed.
pub fn deserialize_tensor(buf: &[u8]) -> Result<(DeserializedTensor, usize)> {
    let (header, header_len) = TensorHeader::decode(buf)?;
    let data_start = header_len;
    let data_len_usize = usize::try_from(header.data_len)
        .map_err(|_| anyhow::anyhow!("data_len {} exceeds addressable range", header.data_len))?;
    let data_end = data_start
        .checked_add(data_len_usize)
        .ok_or_else(|| anyhow::anyhow!("data offset overflow"))?;

    if buf.len() < data_end {
        bail!(
            "buffer too short for tensor data: need {} bytes, got {}",
            data_end,
            buf.len()
        );
    }

    let mut payload = buf[data_start..data_end].to_vec();

    // Step 1: Decompress if compressed.
    if header.flags.is_compressed() {
        payload = decompress(&payload)?;
    }

    // Step 2: Dequantize if quantized.
    let original_dtype = if header.flags.is_quantized() {
        let num_elements = header.num_elements() as usize;
        match header.dtype {
            TensorDtype::Int8 => {
                payload = dequantize_int8(&payload, num_elements);
                TensorDtype::Float16
            }
            TensorDtype::Int4 => {
                payload = dequantize_int4(&payload, num_elements);
                TensorDtype::Float16
            }
            other => bail!("unexpected quantized dtype: {other}"),
        }
    } else {
        header.dtype
    };

    let metadata = if header.metadata.is_empty() {
        None
    } else {
        Some(
            String::from_utf8(header.metadata)
                .map_err(|e| anyhow::anyhow!("invalid UTF-8 in metadata: {e}"))?,
        )
    };

    Ok((
        DeserializedTensor {
            dtype: original_dtype,
            shape: header.shape,
            kind: header.kind,
            metadata,
            data: payload,
        },
        data_end,
    ))
}

/// Validate that the data buffer size matches the shape and dtype.
///
/// Uses checked arithmetic to prevent integer overflow on crafted inputs.
fn validate_data_size(dtype: TensorDtype, shape: &[u64], data: &[u8]) -> Result<()> {
    let num_elements: u64 = if shape.is_empty() {
        0
    } else {
        shape
            .iter()
            .copied()
            .try_fold(1u64, u64::checked_mul)
            .ok_or_else(|| anyhow::anyhow!("shape product overflow for {shape:?}"))?
    };
    let elem_size = dtype.element_size();

    if elem_size == 0 {
        // Sub-byte dtype (Int4): expect ceil(num_elements / 2) bytes.
        let expected = num_elements.div_ceil(2) as usize;
        if data.len() != expected {
            bail!(
                "data size mismatch for {dtype}: shape {shape:?} ({num_elements} elements) \
                 expects {expected} bytes (packed), got {}",
                data.len()
            );
        }
    } else {
        let expected = num_elements
            .checked_mul(elem_size as u64)
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| anyhow::anyhow!(
                "data size overflow for {dtype}: shape {shape:?} ({num_elements} elements,                  elem_size {elem_size})"
            ))?;
        if data.len() != expected {
            bail!(
                "data size mismatch for {dtype}: shape {shape:?} ({num_elements} elements) \
                 expects {expected} bytes, got {}",
                data.len()
            );
        }
    }

    Ok(())
}

/// Convenience: serialize a tensor and return it as `Bytes`.
pub fn serialize_tensor_to_bytes(
    dtype: TensorDtype,
    shape: &[u64],
    data: &[u8],
    options: &SerializeOptions,
) -> Result<Bytes> {
    serialize_tensor(dtype, shape, data, options).map(Bytes::from)
}

#[cfg(test)]
#[path = "tensor_serialize_tests.rs"]
mod tests;
