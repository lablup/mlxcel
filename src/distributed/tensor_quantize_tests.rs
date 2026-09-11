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
    let dequantized = dequantize_int8(&quantized, 256);
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
    let dequantized = dequantize_int8(&quantized, 256);
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
    let dequantized = dequantize_int4(&quantized, 256);
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
    let dequantized = dequantize_int4(&quantized, 128);
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
    let dequantized = dequantize_int8(&quantized, 200);
    let result = read_f16_data(&dequantized);
    assert_eq!(result.len(), 200);
}

#[test]
fn test_quantize_int4_non_group_aligned() {
    let original: Vec<f32> = (0..200).map(|i| i as f32 / 100.0).collect();
    let data = make_f16_data(&original);
    let quantized = quantize_int4(&data);
    let dequantized = dequantize_int4(&quantized, 200);
    let result = read_f16_data(&dequantized);
    assert_eq!(result.len(), 200);
}

#[test]
fn test_quantize_int4_odd_elements() {
    // Odd number of elements tests nibble packing edge case.
    let original: Vec<f32> = (0..129).map(|i| i as f32 / 64.0).collect();
    let data = make_f16_data(&original);
    let quantized = quantize_int4(&data);
    let dequantized = dequantize_int4(&quantized, 129);
    let result = read_f16_data(&dequantized);
    assert_eq!(result.len(), 129);
}
