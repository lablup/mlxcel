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

//! Receiver-side decode limits and scalar semantics (issue #2716).

use super::super::tensor_protocol::{PROTOCOL_VERSION, TensorDtype, TensorFlags, TensorHeader};
use super::*;
use crate::distributed::kv_cache_serde::{RawTensorData, reconstruct_mlx_array};

/// A header with no payload bytes behind it: `data_len` claims `data_len`
/// bytes, so any check that runs only after the payload is sliced reports
/// "buffer too short" instead of the limit.
fn header_only(flags: u8, dtype: TensorDtype, shape: &[u64], data_len: u64) -> Vec<u8> {
    TensorHeader {
        version: PROTOCOL_VERSION,
        flags: TensorFlags::from_bits(flags),
        kind: TensorKind::Activation,
        dtype,
        shape: shape.to_vec(),
        metadata: Vec::new(),
        data_len,
    }
    .encode()
}

fn limits(shape: ShapeBound) -> TensorDecodeLimits {
    TensorDecodeLimits {
        allow_compressed: false,
        allow_quantized: false,
        dtypes: vec![TensorDtype::Float16],
        shape,
    }
}

fn expect_limit_err(buf: &[u8], limits: &TensorDecodeLimits, needle: &str) {
    let err = match deserialize_tensor_with_limits(buf, limits) {
        Ok((tensor, _)) => panic!("must be rejected, got {} data bytes", tensor.data.len()),
        Err(err) => format!("{err:#}"),
    };
    assert!(
        !err.contains("buffer too short"),
        "the limit must be checked before the payload: {err}"
    );
    assert!(err.contains(needle), "expected {needle:?} in: {err}");
}

#[test]
fn limits_refuse_flags_dtype_and_shape_before_the_payload() {
    let axes = limits(ShapeBound::Axes(vec![
        DimBound::Exact(1),
        DimBound::AtMost(8),
    ]));
    let f16 = TensorDtype::Float16;
    expect_limit_err(
        &header_only(TensorFlags::COMPRESSED, f16, &[1, 4], 8),
        &axes,
        "COMPRESSED",
    );
    expect_limit_err(
        &header_only(TensorFlags::QUANTIZED, TensorDtype::Int8, &[1, 4], 8),
        &axes,
        "QUANTIZED",
    );
    expect_limit_err(
        &header_only(0, TensorDtype::Float32, &[1, 4], 16),
        &axes,
        "dtype float32",
    );
    expect_limit_err(&header_only(0, f16, &[1, 4, 1], 8), &axes, "rank 2");
    expect_limit_err(&header_only(0, f16, &[2, 4], 16), &axes, "axis 0 is 2");
    expect_limit_err(&header_only(0, f16, &[1, 9], 18), &axes, "axis 1 is 9");

    let uniform = limits(ShapeBound::Uniform {
        rank: 2..=3,
        max_dim: 8,
    });
    expect_limit_err(&header_only(0, f16, &[4], 8), &uniform, "rank 2..=3");
    expect_limit_err(
        &header_only(0, f16, &[1, 1, 9], 18),
        &uniform,
        "axis 2 is 9",
    );
}

/// A header inside the limits proceeds to the payload checks, and a frame
/// that satisfies both decodes.
#[test]
fn limits_admit_a_matching_frame() {
    let axes = limits(ShapeBound::Axes(vec![
        DimBound::Exact(1),
        DimBound::AtMost(8),
    ]));
    let err = match deserialize_tensor_with_limits(
        &header_only(0, TensorDtype::Float16, &[1, 4], 8),
        &axes,
    ) {
        Ok(_) => panic!("a header without its payload must be rejected"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("buffer too short"), "unexpected error: {err}");

    let data = [0u8; 8];
    let wire = serialize_tensor(
        TensorDtype::Float16,
        &[1, 4],
        &data,
        &SerializeOptions::default(),
    )
    .unwrap();
    let (tensor, consumed) = deserialize_tensor_with_limits(&wire, &axes).unwrap();
    assert_eq!(consumed, wire.len());
    assert_eq!(tensor.data, data);
}

/// Issue #2716 item 4: an empty shape is a 1-element scalar, so a float16
/// scalar survives the whole wire path to an MLX array, and `[]` with no
/// bytes is rejected at decode instead of only in `validate_raw_tensor`.
#[test]
fn float16_scalar_round_trips_and_empty_scalar_is_rejected() {
    let bits = 0x3C00u16.to_le_bytes(); // 1.0 in float16
    let wire = serialize_tensor(
        TensorDtype::Float16,
        &[],
        &bits,
        &SerializeOptions::default(),
    )
    .unwrap();
    let (tensor, _) = deserialize_tensor(&wire).unwrap();
    assert!(tensor.shape.is_empty());
    assert_eq!(tensor.data, bits);
    let arr = reconstruct_mlx_array(&RawTensorData {
        data: tensor.data,
        shape: Vec::new(),
        dtype: mlxcel_core::dtype::FLOAT16,
    })
    .unwrap();
    assert!(mlxcel_core::array_shape(&arr).is_empty());
    assert_eq!(mlxcel_core::array_to_raw_bytes(&arr), bits);

    let empty = header_only(0, TensorDtype::Float16, &[], 0);
    let err = match deserialize_tensor(&empty) {
        Ok((tensor, _)) => panic!("[] with 0 bytes must be rejected, got {:?}", tensor.data),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("size mismatch"), "unexpected error: {err}");
}

/// Issue #2716 item 5: the payload is decompressed or dequantized from the
/// received bytes. The only copy in the decode body is the final
/// `into_owned`, which copies nothing when a flag already produced an owned
/// buffer; a `to_vec()` there means the payload is copied before decoding.
#[test]
fn decode_does_not_copy_the_payload_before_decompressing() {
    let source = include_str!("tensor_serialize.rs");
    let start = source
        .find("pub fn deserialize_tensor_with_limits(")
        .expect("decode function present");
    let body = &source[start..];
    let body = &body[..body.find("\n}\n").expect("decode function ends")];
    let decompress_at = body.find("decompress(").expect("decode decompresses");
    assert!(
        !body[..decompress_at].contains("to_vec()"),
        "the payload is copied before decompress"
    );
    assert!(
        !body.contains("to_vec()"),
        "the decode body must not copy the payload"
    );
}
