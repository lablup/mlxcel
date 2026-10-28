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

use super::super::tensor_protocol::{
    PROTOCOL_VERSION, QuantizationMode, TensorDtype, TensorFlags, TensorHeader, TensorKind,
};
use super::*;

/// Helper: create float32 raw bytes.
fn make_f32_data(values: &[f32]) -> Vec<u8> {
    let mut data = Vec::with_capacity(values.len() * 4);
    for &v in values {
        data.extend_from_slice(&v.to_le_bytes());
    }
    data
}

/// Helper: create float16 raw bytes from f32 values.
fn make_f16_data(values: &[f32]) -> Vec<u8> {
    let mut data = Vec::with_capacity(values.len() * 2);
    for &v in values {
        let bits = f32_to_f16_helper(v);
        data.extend_from_slice(&bits.to_le_bytes());
    }
    data
}

/// Minimal f32-to-f16 helper for tests.
fn f32_to_f16_helper(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7FFFFF;

    if exponent == 255 {
        if mantissa == 0 {
            (sign << 15) | 0x7C00
        } else {
            (sign << 15) | 0x7C00 | ((mantissa >> 13) as u16).max(1)
        }
    } else if exponent > 142 {
        (sign << 15) | 0x7C00
    } else if exponent < 113 {
        sign << 15
    } else {
        let f16_exp = (exponent - 112) as u16;
        let f16_man = (mantissa >> 13) as u16;
        (sign << 15) | (f16_exp << 10) | f16_man
    }
}

#[test]
fn test_serialize_deserialize_float32_bitexact() {
    let values: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
    let data = make_f32_data(&values);
    let options = SerializeOptions {
        kind: TensorKind::Activation,
        quantization: QuantizationMode::None,
        compress: false,
        metadata: None,
    };

    let wire = serialize_tensor(TensorDtype::Float32, &[64], &data, &options).unwrap();
    let (tensor, consumed) = deserialize_tensor(&wire).unwrap();

    assert_eq!(consumed, wire.len());
    assert_eq!(tensor.dtype, TensorDtype::Float32);
    assert_eq!(tensor.shape, vec![64]);
    assert_eq!(tensor.kind, TensorKind::Activation);
    assert_eq!(
        tensor.data, data,
        "non-quantized transfer must be bit-exact"
    );
}

#[test]
fn test_serialize_deserialize_float16_bitexact() {
    let data = make_f16_data(&[1.0, -1.0, 0.5, 0.0]);
    let options = SerializeOptions {
        kind: TensorKind::KVCache,
        quantization: QuantizationMode::None,
        compress: false,
        metadata: Some("{\"layer\":0}".to_string()),
    };

    let wire = serialize_tensor(TensorDtype::Float16, &[4], &data, &options).unwrap();
    let (tensor, _) = deserialize_tensor(&wire).unwrap();

    assert_eq!(tensor.dtype, TensorDtype::Float16);
    assert_eq!(
        tensor.data, data,
        "non-quantized transfer must be bit-exact"
    );
    assert_eq!(tensor.metadata.as_deref(), Some("{\"layer\":0}"));
}

#[test]
fn test_serialize_deserialize_multidim() {
    let data = make_f32_data(&[0.0; 24]);
    let options = SerializeOptions {
        kind: TensorKind::WeightShard,
        quantization: QuantizationMode::None,
        compress: false,
        metadata: None,
    };

    let wire = serialize_tensor(TensorDtype::Float32, &[2, 3, 4], &data, &options).unwrap();
    let (tensor, _) = deserialize_tensor(&wire).unwrap();

    assert_eq!(tensor.shape, vec![2, 3, 4]);
    assert_eq!(tensor.data, data);
}

#[test]
fn test_serialize_with_compression() {
    // Sparse data: mostly zeros.
    let data = vec![0u8; 4096]; // 1024 float32 zeros.
    let options = SerializeOptions {
        kind: TensorKind::Activation,
        quantization: QuantizationMode::None,
        compress: true,
        metadata: None,
    };

    let wire = serialize_tensor(TensorDtype::Float32, &[1024], &data, &options).unwrap();
    // Compressed wire should be smaller than uncompressed.
    assert!(
        wire.len() < data.len(),
        "compressed wire ({}) should be smaller than raw data ({})",
        wire.len(),
        data.len()
    );

    let (tensor, _) = deserialize_tensor(&wire).unwrap();
    assert_eq!(tensor.data, data, "compressed transfer must be bit-exact");
}

#[test]
fn test_serialize_with_int8_quantization() {
    let values: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
    let data = make_f16_data(&values);
    let options = SerializeOptions {
        kind: TensorKind::Activation,
        quantization: QuantizationMode::Int8,
        compress: false,
        metadata: None,
    };

    let wire = serialize_tensor(TensorDtype::Float16, &[256], &data, &options).unwrap();
    // Quantized wire should be smaller (roughly half for int8).
    assert!(
        wire.len() < data.len(),
        "int8 quantized wire ({}) should be smaller than original ({})",
        wire.len(),
        data.len()
    );

    let (tensor, _) = deserialize_tensor(&wire).unwrap();
    assert_eq!(tensor.dtype, TensorDtype::Float16);
    assert_eq!(tensor.data.len(), data.len());
    // Quantized transfer is lossy, but should be close.
}

#[test]
fn test_serialize_with_int4_quantization() {
    let values: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
    let data = make_f16_data(&values);
    let options = SerializeOptions {
        kind: TensorKind::WeightShard,
        quantization: QuantizationMode::Int4,
        compress: false,
        metadata: None,
    };

    let wire = serialize_tensor(TensorDtype::Float16, &[256], &data, &options).unwrap();
    // Int4 should be roughly 1/4 of original.
    assert!(
        wire.len() < data.len() / 2,
        "int4 quantized wire ({}) should be much smaller than original ({})",
        wire.len(),
        data.len()
    );

    let (tensor, _) = deserialize_tensor(&wire).unwrap();
    assert_eq!(tensor.dtype, TensorDtype::Float16);
    assert_eq!(tensor.data.len(), data.len());
}

#[test]
fn test_serialize_quantization_wrong_dtype() {
    let data = make_f32_data(&[1.0; 16]);
    let options = SerializeOptions {
        kind: TensorKind::Activation,
        quantization: QuantizationMode::Int8,
        compress: false,
        metadata: None,
    };

    let result = serialize_tensor(TensorDtype::Float32, &[16], &data, &options);
    assert!(result.is_err(), "quantization should require float16 input");
}

#[test]
fn test_serialize_data_size_mismatch() {
    let data = vec![0u8; 100]; // Wrong size for shape [16] of float32.
    let options = SerializeOptions::default();

    let result = serialize_tensor(TensorDtype::Float32, &[16], &data, &options);
    assert!(result.is_err());
}

#[test]
fn test_deserialize_truncated_data() {
    let data = make_f32_data(&[1.0; 16]);
    let options = SerializeOptions::default();

    let mut wire = serialize_tensor(TensorDtype::Float32, &[16], &data, &options).unwrap();
    wire.truncate(wire.len() / 2); // Corrupt.

    let result = deserialize_tensor(&wire);
    assert!(result.is_err());
}

#[test]
fn test_serialize_to_bytes() {
    let data = make_f32_data(&[1.0; 4]);
    let options = SerializeOptions::default();

    let bytes = serialize_tensor_to_bytes(TensorDtype::Float32, &[4], &data, &options).unwrap();
    assert!(!bytes.is_empty());
}

#[test]
fn test_all_tensor_kinds() {
    let data = make_f32_data(&[1.0; 4]);

    for kind in [
        TensorKind::KVCache,
        TensorKind::Activation,
        TensorKind::WeightShard,
    ] {
        let options = SerializeOptions {
            kind,
            quantization: QuantizationMode::None,
            compress: false,
            metadata: None,
        };

        let wire = serialize_tensor(TensorDtype::Float32, &[4], &data, &options).unwrap();
        let (tensor, _) = deserialize_tensor(&wire).unwrap();
        assert_eq!(tensor.kind, kind);
        assert_eq!(tensor.data, data, "all tensor kinds must be bit-exact");
    }
}

#[test]
fn test_serialize_with_compression_and_quantization() {
    // Sparse float16 data that should compress well after quantization.
    let mut values = vec![0.0f32; 256];
    values[0] = 1.0;
    values[128] = -1.0;
    let data = make_f16_data(&values);

    let options = SerializeOptions {
        kind: TensorKind::KVCache,
        quantization: QuantizationMode::Int8,
        compress: true,
        metadata: Some("{\"seq_len\":256}".to_string()),
    };

    let wire = serialize_tensor(TensorDtype::Float16, &[256], &data, &options).unwrap();
    let (tensor, _) = deserialize_tensor(&wire).unwrap();
    assert_eq!(tensor.dtype, TensorDtype::Float16);
    assert_eq!(tensor.data.len(), data.len());
    assert_eq!(tensor.metadata.as_deref(), Some("{\"seq_len\":256}"));
}

/// Build a wire buffer by hand so the header can disagree with the payload,
/// which `serialize_tensor` refuses to emit.
fn hand_built_wire(
    flags: TensorFlags,
    dtype: TensorDtype,
    shape: &[u64],
    payload: &[u8],
) -> Vec<u8> {
    let header = TensorHeader {
        version: PROTOCOL_VERSION,
        flags,
        kind: TensorKind::Activation,
        dtype,
        shape: shape.to_vec(),
        metadata: Vec::new(),
        data_len: payload.len() as u64,
    };
    let mut wire = header.encode();
    wire.extend_from_slice(payload);
    wire
}

fn flags_with(bits: &[u8]) -> TensorFlags {
    let mut flags = TensorFlags::new();
    for &bit in bits {
        flags.set(bit);
    }
    flags
}

/// Return the deserialize error; an `Ok` fails the test without printing the
/// decoded buffer (which can be gigabytes on unfixed code).
fn expect_deserialize_err(wire: &[u8]) -> String {
    match deserialize_tensor(wire) {
        Ok((tensor, _)) => panic!(
            "deserialize_tensor must reject this frame, got {} data bytes",
            tensor.data.len()
        ),
        Err(err) => format!("{err:#}"),
    }
}

/// An int8 payload header (`num_groups = 0`, `group_size = 128`) with no
/// scales or data.
fn empty_int8_payload() -> Vec<u8> {
    let mut payload = 0u32.to_le_bytes().to_vec();
    payload.extend_from_slice(&128u32.to_le_bytes());
    payload
}

/// A QUANTIZED shape of `1 << 31` elements with an 8-byte payload used to
/// return `Ok` with 4 GiB of zeros; the layout check rejects it before the
/// shape-sized output is allocated.
#[test]
fn deserialize_rejects_quantized_shape_larger_than_payload() {
    let wire = hand_built_wire(
        flags_with(&[TensorFlags::QUANTIZED]),
        TensorDtype::Int8,
        &[1 << 31],
        &empty_int8_payload(),
    );
    expect_deserialize_err(&wire);
}

/// A shape whose element count overflows `u64` is an error, not a saturated
/// count that a later `* 2` overflows.
#[test]
fn deserialize_rejects_quantized_shape_overflow() {
    let wire = hand_built_wire(
        flags_with(&[TensorFlags::QUANTIZED]),
        TensorDtype::Int8,
        &[u64::MAX, 2],
        &empty_int8_payload(),
    );
    let err = expect_deserialize_err(&wire);
    assert!(err.contains("overflow"), "unexpected error: {err}");
}

#[test]
fn deserialize_rejects_unquantized_size_mismatch() {
    let wire = hand_built_wire(TensorFlags::new(), TensorDtype::Float32, &[4], &[0u8; 12]);
    let err = expect_deserialize_err(&wire);
    assert!(err.contains("size mismatch"), "unexpected error: {err}");
}

/// A COMPRESSED float32 `[4]` tensor (16 bytes) whose LZ4 prefix declares
/// 3 GiB is refused on the size check instead of allocating 3 GiB.
#[test]
fn deserialize_rejects_compressed_size_beyond_shape() {
    let declared: u32 = 3 << 30;
    let mut payload = u64::from(declared).to_le_bytes().to_vec();
    payload.extend_from_slice(&declared.to_le_bytes());
    payload.push(0x00);
    let wire = hand_built_wire(
        flags_with(&[TensorFlags::COMPRESSED]),
        TensorDtype::Float32,
        &[4],
        &payload,
    );
    let err = expect_deserialize_err(&wire);
    assert!(err.contains("exceeds limit"), "unexpected error: {err}");
}

/// A compressed payload that decompresses to the wrong size for the shape is
/// rejected (the decompression limit allows it; the final size check does
/// not).
#[test]
fn deserialize_rejects_compressed_payload_smaller_than_shape() {
    let compressed = super::super::tensor_compress::compress(&[0u8; 4096]);
    let wire = hand_built_wire(
        flags_with(&[TensorFlags::COMPRESSED]),
        TensorDtype::Float32,
        &[2048],
        &compressed,
    );
    let err = expect_deserialize_err(&wire);
    assert!(err.contains("size mismatch"), "unexpected error: {err}");
}

#[test]
fn deserialize_round_trips_compressed_int4_quantized_tensor() {
    let mut values = vec![0.0f32; 300];
    values[7] = 1.0;
    let data = make_f16_data(&values);
    let options = SerializeOptions {
        kind: TensorKind::KVCache,
        quantization: QuantizationMode::Int4,
        compress: true,
        metadata: None,
    };
    let wire = serialize_tensor(TensorDtype::Float16, &[3, 100], &data, &options).unwrap();
    let (tensor, consumed) = deserialize_tensor(&wire).unwrap();
    assert_eq!(consumed, wire.len());
    assert_eq!(tensor.shape, vec![3, 100]);
    assert_eq!(tensor.data.len(), data.len());
}
