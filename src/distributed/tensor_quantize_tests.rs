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

use super::*;

/// Helper to create float16 bytes from f32 values.
fn make_f16_data(values: &[f32]) -> Vec<u8> {
    let mut data = Vec::with_capacity(values.len() * 2);
    for &v in values {
        data.extend_from_slice(&f32_to_f16(v).to_le_bytes());
    }
    data
}

/// Helper to read float16 bytes back to f32 values.
fn read_f16_data(data: &[u8]) -> Vec<f32> {
    (0..data.len() / 2)
        .map(|i| {
            let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
            f16_to_f32(bits)
        })
        .collect()
}

#[test]
fn test_f16_roundtrip_basic() {
    let test_values = [0.0f32, 1.0, -1.0, 0.5, -0.5, 65504.0, -65504.0];
    for &v in &test_values {
        let bits = f32_to_f16(v);
        let back = f16_to_f32(bits);
        assert!(
            (back - v).abs() < 1e-3 || (v.abs() > 1000.0 && (back - v).abs() / v.abs() < 0.01),
            "f16 roundtrip failed for {v}: got {back}"
        );
    }
}

#[test]
fn test_f16_zero() {
    assert_eq!(f16_to_f32(f32_to_f16(0.0)), 0.0);
    assert_eq!(f16_to_f32(f32_to_f16(-0.0)), -0.0);
}

#[test]
fn test_f16_infinity() {
    let pos_inf = f32_to_f16(f32::INFINITY);
    assert!(f16_to_f32(pos_inf).is_infinite());
    assert!(f16_to_f32(pos_inf).is_sign_positive());

    let neg_inf = f32_to_f16(f32::NEG_INFINITY);
    assert!(f16_to_f32(neg_inf).is_infinite());
    assert!(f16_to_f32(neg_inf).is_sign_negative());
}

#[test]
fn test_quantize_dequantize_int8_zeros() {
    let data = make_f16_data(&[0.0; 256]);
    let quantized = quantize_int8(&data);
    let dequantized = dequantize_int8(&quantized, 256).unwrap();
    let values = read_f16_data(&dequantized);
    for v in values {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn test_quantize_dequantize_int8_accuracy() {
    // Create a range of values.
    let original: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) / 128.0).collect();
    let data = make_f16_data(&original);
    let quantized = quantize_int8(&data);
    let dequantized = dequantize_int8(&quantized, 256).unwrap();
    let result = read_f16_data(&dequantized);

    // Int8 quantization should be within ~1% of absmax per group.
    for (i, (&orig, &deq)) in original.iter().zip(result.iter()).enumerate() {
        let tolerance = 0.02; // 2% tolerance for quantization error.
        let err = (orig - deq).abs();
        assert!(
            err < tolerance,
            "int8 quantization error at index {i}: orig={orig}, deq={deq}, err={err}"
        );
    }
}

#[test]
fn test_quantize_dequantize_int8_bandwidth() {
    let data = make_f16_data(&[1.0; 1024]);
    let quantized = quantize_int8(&data);
    // Original: 1024 * 2 = 2048 bytes.
    // Quantized: 8 (header) + scales + data. Should be roughly half.
    assert!(
        quantized.len() < data.len(),
        "int8 quantized should be smaller: {} vs {}",
        quantized.len(),
        data.len()
    );
}

#[test]
fn test_quantize_dequantize_int4_zeros() {
    let data = make_f16_data(&[0.0; 256]);
    let quantized = quantize_int4(&data);
    let dequantized = dequantize_int4(&quantized, 256).unwrap();
    let values = read_f16_data(&dequantized);
    for v in values {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn test_quantize_dequantize_int4_accuracy() {
    // Int4 has lower precision, use a smaller range.
    let original: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) / 64.0).collect();
    let data = make_f16_data(&original);
    let quantized = quantize_int4(&data);
    let dequantized = dequantize_int4(&quantized, 128).unwrap();
    let result = read_f16_data(&dequantized);

    // Int4 has much larger quantization error (~15% of range per group).
    for (i, (&orig, &deq)) in original.iter().zip(result.iter()).enumerate() {
        let tolerance = 0.20; // 20% tolerance for 4-bit quantization.
        let err = (orig - deq).abs();
        assert!(
            err < tolerance,
            "int4 quantization error at index {i}: orig={orig}, deq={deq}, err={err}"
        );
    }
}

#[test]
fn test_quantize_dequantize_int4_bandwidth() {
    let data = make_f16_data(&[1.0; 1024]);
    let quantized = quantize_int4(&data);
    // Quantized should be roughly 1/4 of original.
    assert!(
        quantized.len() < data.len() / 2,
        "int4 quantized should be much smaller: {} vs {}",
        quantized.len(),
        data.len()
    );
}

#[test]
fn test_bandwidth_ratio() {
    use super::super::tensor_protocol::QuantizationMode;
    assert_eq!(bandwidth_ratio(&QuantizationMode::None), 1.0);
    assert!(bandwidth_ratio(&QuantizationMode::Int8) < 0.6);
    assert!(bandwidth_ratio(&QuantizationMode::Int4) < 0.3);
}

#[test]
fn test_quantize_int8_non_group_aligned() {
    // 200 elements: not a multiple of group_size (128).
    let original: Vec<f32> = (0..200).map(|i| i as f32 / 100.0).collect();
    let data = make_f16_data(&original);
    let quantized = quantize_int8(&data);
    let dequantized = dequantize_int8(&quantized, 200).unwrap();
    let result = read_f16_data(&dequantized);
    assert_eq!(result.len(), 200);
}

#[test]
fn test_quantize_int4_non_group_aligned() {
    let original: Vec<f32> = (0..200).map(|i| i as f32 / 100.0).collect();
    let data = make_f16_data(&original);
    let quantized = quantize_int4(&data);
    let dequantized = dequantize_int4(&quantized, 200).unwrap();
    let result = read_f16_data(&dequantized);
    assert_eq!(result.len(), 200);
}

#[test]
fn test_quantize_int4_odd_elements() {
    // Odd number of elements tests nibble packing edge case.
    let original: Vec<f32> = (0..129).map(|i| i as f32 / 64.0).collect();
    let data = make_f16_data(&original);
    let quantized = quantize_int4(&data);
    let dequantized = dequantize_int4(&quantized, 129).unwrap();
    let result = read_f16_data(&dequantized);
    assert_eq!(result.len(), 129);
}

/// Hand-built quantized payload header: `[num_groups][group_size][rest]`.
fn quantized_payload(num_groups: u32, group_size: u32, rest_len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + rest_len);
    out.extend_from_slice(&num_groups.to_le_bytes());
    out.extend_from_slice(&group_size.to_le_bytes());
    out.resize(8 + rest_len, 0);
    out
}

/// Return the dequantize error; an `Ok` fails without printing the output.
fn expect_err(result: anyhow::Result<Vec<u8>>, what: &str) -> String {
    match result {
        Ok(out) => panic!("{what} must be rejected, got {} output bytes", out.len()),
        Err(err) => err.to_string(),
    }
}

type Dequantize = fn(&[u8], usize) -> anyhow::Result<Vec<u8>>;

/// Both dequantizers, each with the size of its data section for
/// `num_elements` values.
fn dequantizers() -> [(&'static str, Dequantize, fn(usize) -> usize); 2] {
    [
        ("int8", dequantize_int8, |n| n),
        ("int4", dequantize_int4, |n| n.div_ceil(2)),
    ]
}

#[test]
fn dequantize_rejects_payload_shorter_than_header() {
    for (name, dequantize, _) in dequantizers() {
        let err = expect_err(dequantize(&[0u8; 7], 4), name);
        assert!(err.contains("too short"), "{name}: unexpected error: {err}");
    }
}

#[test]
fn dequantize_rejects_zero_group_size() {
    for (name, dequantize, data_len) in dequantizers() {
        let payload = quantized_payload(1, 0, 2 + data_len(4));
        let err = expect_err(dequantize(&payload, 4), name);
        assert!(
            err.contains("group_size"),
            "{name}: unexpected error: {err}"
        );
    }
}

/// A `num_groups` that does not match `ceil(num_elements / group_size)` used
/// to dequantize only the declared groups and return a short tensor.
#[test]
fn dequantize_rejects_num_groups_inconsistent_with_num_elements() {
    for (name, dequantize, data_len) in dequantizers() {
        // 256 elements at group size 128 need 2 groups; claim 1.
        let payload = quantized_payload(1, 128, 2 + data_len(256));
        let err = expect_err(dequantize(&payload, 256), name);
        assert!(
            err.contains("num_groups"),
            "{name}: unexpected error: {err}"
        );
    }
}

/// A header that promises more elements than the payload carries is rejected
/// instead of zero-filling a buffer sized from the header.
#[test]
fn dequantize_rejects_payload_shorter_than_layout() {
    for (name, dequantize, data_len) in dequantizers() {
        let payload = quantized_payload(2, 128, 4 + data_len(256) - 1);
        let err = expect_err(dequantize(&payload, 256), name);
        assert!(err.contains("expected"), "{name}: unexpected error: {err}");
    }
}

#[test]
fn dequantize_rejects_trailing_bytes() {
    for (name, dequantize, data_len) in dequantizers() {
        let payload = quantized_payload(2, 128, 4 + data_len(256) + 1);
        let err = expect_err(dequantize(&payload, 256), name);
        assert!(err.contains("expected"), "{name}: unexpected error: {err}");
    }
}

/// A shape far larger than the payload must fail on the layout check, never
/// allocate the shape-sized output (`1 << 40` elements would be 2 TiB).
#[test]
fn dequantize_rejects_huge_element_count_without_allocating() {
    for (name, dequantize, _) in dequantizers() {
        let payload = quantized_payload(0, 128, 0);
        expect_err(dequantize(&payload, 1 << 40), name);
        expect_err(dequantize(&payload, usize::MAX), name);
    }
}

#[test]
fn dequantize_accepts_empty_tensor() {
    let int8 = quantize_int8(&[]);
    assert_eq!(dequantize_int8(&int8, 0).unwrap(), Vec::<u8>::new());
    let int4 = quantize_int4(&[]);
    assert_eq!(dequantize_int4(&int4, 0).unwrap(), Vec::<u8>::new());
}

#[test]
fn payload_max_len_bounds_every_quantized_payload() {
    for n in [0usize, 1, 2, 127, 128, 129, 200, 1024] {
        let data = make_f16_data(&vec![0.5; n]);
        assert!(quantize_int8(&data).len() <= int8_payload_max_len(n).unwrap());
        assert!(quantize_int4(&data).len() <= int4_payload_max_len(n).unwrap());
    }
    assert_eq!(int8_payload_max_len(usize::MAX), None);
    assert_eq!(int4_payload_max_len(usize::MAX), None);
}
