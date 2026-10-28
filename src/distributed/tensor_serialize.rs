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

use std::borrow::Cow;
use std::fmt;
use std::ops::RangeInclusive;

use anyhow::{Result, anyhow, bail};
use bytes::Bytes;

use super::tensor_compress::{compress_if_beneficial, decompress};
use super::tensor_protocol::{
    PROTOCOL_VERSION, QuantizationMode, TensorDtype, TensorFlags, TensorHeader, TensorKind,
    max_decoded_tensor_bytes,
};
use super::tensor_quantize::{
    dequantize_int4, dequantize_int8, int4_payload_max_len, int8_payload_max_len, quantize_int4,
    quantize_int8,
};

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

/// Bound on one axis of a tensor a receiver is willing to decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DimBound {
    /// The axis must have exactly this length.
    Exact(u64),
    /// The axis may have any length up to and including this one.
    AtMost(u64),
}

impl DimBound {
    fn admits(self, dim: u64) -> bool {
        match self {
            Self::Exact(n) => dim == n,
            Self::AtMost(n) => dim <= n,
        }
    }
}

impl fmt::Display for DimBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact(n) => write!(f, "exactly {n}"),
            Self::AtMost(n) => write!(f, "at most {n}"),
        }
    }
}

/// Shape a receiver is willing to decode.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ShapeBound {
    /// One bound per axis; the rank must equal the number of bounds.
    Axes(Vec<DimBound>),
    /// A rank within `rank`, every axis at most `max_dim`.
    Uniform {
        rank: RangeInclusive<usize>,
        max_dim: u64,
    },
}

/// What a receiver accepts from a peer's tensor header, checked by
/// [`deserialize_tensor_with_limits`] right after the header is decoded and
/// before the payload is sliced, copied, or allocated.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorDecodeLimits {
    /// Accept an LZ4-compressed payload.
    pub allow_compressed: bool,
    /// Accept a quantized payload.
    pub allow_quantized: bool,
    /// Accepted header dtypes (the wire dtype, so `Int8`/`Int4` for a
    /// quantized frame). Empty accepts any dtype.
    pub dtypes: Vec<TensorDtype>,
    /// Accepted shape.
    pub shape: ShapeBound,
}

impl TensorDecodeLimits {
    /// Accept every flag, dtype, and shape the wire format can express. The
    /// header-derived size checks of [`deserialize_tensor_with_limits`]
    /// still apply.
    pub fn unrestricted() -> Self {
        Self {
            allow_compressed: true,
            allow_quantized: true,
            dtypes: Vec::new(),
            shape: ShapeBound::Uniform {
                rank: 0..=usize::MAX,
                max_dim: u64::MAX,
            },
        }
    }

    /// Reject a header whose flags, dtype, or shape fall outside the limits.
    pub fn check_header(&self, header: &TensorHeader) -> Result<()> {
        if header.flags.is_compressed() && !self.allow_compressed {
            bail!("tensor flag COMPRESSED is not accepted for this tensor");
        }
        if header.flags.is_quantized() && !self.allow_quantized {
            bail!("tensor flag QUANTIZED is not accepted for this tensor");
        }
        if !self.dtypes.is_empty() && !self.dtypes.contains(&header.dtype) {
            bail!(
                "tensor dtype {} is not accepted for this tensor (expected one of {:?})",
                header.dtype,
                self.dtypes
            );
        }
        let shape = &header.shape;
        match &self.shape {
            ShapeBound::Axes(axes) => {
                if shape.len() != axes.len() {
                    bail!(
                        "tensor shape {shape:?} has rank {}, expected rank {}",
                        shape.len(),
                        axes.len()
                    );
                }
                for (axis, (&dim, bound)) in shape.iter().zip(axes).enumerate() {
                    if !bound.admits(dim) {
                        bail!("tensor shape {shape:?}: axis {axis} is {dim}, expected {bound}");
                    }
                }
            }
            ShapeBound::Uniform { rank, max_dim } => {
                if !rank.contains(&shape.len()) {
                    bail!(
                        "tensor shape {shape:?} has rank {}, expected rank {}..={}",
                        shape.len(),
                        rank.start(),
                        rank.end()
                    );
                }
                if let Some((axis, dim)) = shape.iter().enumerate().find(|(_, d)| **d > *max_dim) {
                    bail!(
                        "tensor shape {shape:?}: axis {axis} is {dim}, expected at most {max_dim}"
                    );
                }
            }
        }
        Ok(())
    }
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
/// Returns the deserialized tensor and the number of bytes consumed. Accepts
/// any header the wire format can express; a receiver that knows what to
/// expect uses [`deserialize_tensor_with_limits`].
pub fn deserialize_tensor(buf: &[u8]) -> Result<(DeserializedTensor, usize)> {
    deserialize_tensor_with_limits(buf, &TensorDecodeLimits::unrestricted())
}

/// [`deserialize_tensor`], refusing a header outside `limits` before the
/// payload is sliced, copied, decompressed, or allocated.
pub fn deserialize_tensor_with_limits(
    buf: &[u8],
    limits: &TensorDecodeLimits,
) -> Result<(DeserializedTensor, usize)> {
    let (header, header_len) = TensorHeader::decode(buf)?;
    limits.check_header(&header)?;
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

    // Decompress and dequantize straight from the received bytes; only a
    // payload with neither flag is copied, once, into the returned tensor.
    let mut payload = Cow::Borrowed(&buf[data_start..data_end]);

    // The header comes from a peer: an overflowing shape is an error, and the
    // element count bounds every buffer sized below.
    let num_elements = usize::try_from(header.num_elements()?).map_err(|_| {
        anyhow!(
            "element count for shape {:?} exceeds addressable range",
            header.shape
        )
    })?;

    // Step 1: Decompress if compressed, never past what the header allows.
    if header.flags.is_compressed() {
        let limit = max_decoded_payload_len(&header, num_elements)?.min(max_decoded_tensor_bytes());
        payload = Cow::Owned(decompress(&payload, limit)?);
    }

    // Step 2: Dequantize if quantized.
    let original_dtype = if header.flags.is_quantized() {
        match header.dtype {
            TensorDtype::Int8 => {
                payload = Cow::Owned(dequantize_int8(&payload, num_elements)?);
                TensorDtype::Float16
            }
            TensorDtype::Int4 => {
                payload = Cow::Owned(dequantize_int4(&payload, num_elements)?);
                TensorDtype::Float16
            }
            other => bail!("unexpected quantized dtype: {other}"),
        }
    } else {
        header.dtype
    };

    // Every decoded tensor, quantized or not, must match its shape exactly;
    // `serialize_tensor` enforces the same rule on the way out.
    validate_data_size(original_dtype, &header.shape, &payload)?;

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
            data: payload.into_owned(),
        },
        data_end,
    ))
}

/// Upper bound on the payload size after decompression, derived from the
/// header: the exact data size for an unquantized tensor, and the largest
/// valid quantized layout otherwise (the exact layout is checked when
/// dequantizing).
fn max_decoded_payload_len(header: &TensorHeader, num_elements: usize) -> Result<usize> {
    let bound = if header.flags.is_quantized() {
        match header.dtype {
            TensorDtype::Int8 => int8_payload_max_len(num_elements),
            TensorDtype::Int4 => int4_payload_max_len(num_elements),
            other => bail!("unexpected quantized dtype: {other}"),
        }
    } else {
        match header.dtype.element_size() {
            // Sub-byte dtype (Int4): two elements per byte.
            0 => Some(num_elements.div_ceil(2)),
            elem_size => num_elements.checked_mul(elem_size),
        }
    };
    bound.ok_or_else(|| {
        anyhow!(
            "decoded size overflow for {} shape {:?}",
            header.dtype,
            header.shape
        )
    })
}

/// Validate that the data buffer size matches the shape and dtype.
///
/// Uses checked arithmetic to prevent integer overflow on crafted inputs.
fn validate_data_size(dtype: TensorDtype, shape: &[u64], data: &[u8]) -> Result<()> {
    // An empty shape is a 1-element scalar, as in `TensorHeader::num_elements`.
    let num_elements = TensorHeader::shape_num_elements(shape)?;
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
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "data size overflow for {dtype}: shape {shape:?} ({num_elements} elements, \
                 elem_size {elem_size})"
                )
            })?;
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

#[cfg(test)]
#[path = "tensor_serialize_limits_tests.rs"]
mod limits_tests;
