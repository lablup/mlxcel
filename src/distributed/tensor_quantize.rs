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

//! On-the-fly quantization for tensor transfer bandwidth reduction.
//!
//! Provides float16-to-int8 and float16-to-int4 quantization with per-group
//! absmax scaling. The quantized payload includes the scale factors so the
//! receiver can dequantize without extra metadata.
//!
//! # Wire Layout (int8)
//!
//! ```text
//! [num_groups: u32 LE]
//! [group_size: u32 LE]
//! [scales: num_groups * f16]     // Per-group absmax scale factors
//! [quantized: num_elements * i8] // Quantized values
//! ```
//!
//! # Wire Layout (int4, packed)
//!
//! ```text
//! [num_groups: u32 LE]
//! [group_size: u32 LE]
//! [scales: num_groups * f16]          // Per-group absmax scale factors
//! [quantized: ceil(num_elements/2)]   // Packed 4-bit values (2 per byte)
//! ```
//!
//! Used by: distributed inference tensor transfers (KV cache, activations,
//! weight shards).

/// Default group size for absmax quantization.
pub const DEFAULT_GROUP_SIZE: usize = 128;

/// Quantize float16 data to int8 with per-group absmax scaling.
///
/// Input: raw bytes of float16 elements.
/// Output: serialized quantized payload (see wire layout above).
pub fn quantize_int8(f16_data: &[u8]) -> Vec<u8> {
    let num_elements = f16_data.len() / 2;
    let group_size = DEFAULT_GROUP_SIZE;
    let num_groups = num_elements.div_ceil(group_size);

    // Parse float16 values.
    let values: Vec<f32> = (0..num_elements)
        .map(|i| {
            let bits = u16::from_le_bytes([f16_data[i * 2], f16_data[i * 2 + 1]]);
            f16_to_f32(bits)
        })
        .collect();

    // Compute per-group scales and quantized values.
    let mut scales = Vec::with_capacity(num_groups);
    let mut quantized = Vec::with_capacity(num_elements);

    for g in 0..num_groups {
        let start = g * group_size;
        let end = (start + group_size).min(num_elements);
        let group = &values[start..end];

        let absmax = group.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if absmax == 0.0 { 1.0 } else { absmax / 127.0 };
        scales.push(f32_to_f16(scale));

        for &val in group {
            let q = (val / scale).round().clamp(-127.0, 127.0) as i8;
            quantized.push(q as u8);
        }
    }

    // Build output buffer.
    let mut out = Vec::with_capacity(8 + num_groups * 2 + num_elements);
    out.extend_from_slice(&(num_groups as u32).to_le_bytes());
    out.extend_from_slice(&(group_size as u32).to_le_bytes());
    for &s in &scales {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out.extend_from_slice(&quantized);

    out
}

/// Dequantize int8 data back to float16.
///
/// Input: serialized quantized payload.
/// Output: raw bytes of float16 elements.
///
/// # Panics
///
/// Panics if the payload is malformed (too short for declared groups/data).
/// Callers receiving untrusted data should validate payload length first.
pub fn dequantize_int8(payload: &[u8], num_elements: usize) -> Vec<u8> {
    if payload.len() < 8 {
        return vec![0u8; num_elements * 2];
    }

    let num_groups = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
    let group_size = u32::from_le_bytes(payload[4..8].try_into().unwrap()) as usize;

    if group_size == 0 || num_groups == 0 {
        return vec![0u8; num_elements * 2];
    }

    let scales_start: usize = 8;
    let scales_end = scales_start.saturating_add(num_groups.saturating_mul(2));
    let data_start = scales_end;

    // Validate payload has enough data for scales and quantized values.
    let required_len = data_start.saturating_add(num_elements);
    if payload.len() < required_len {
        return vec![0u8; num_elements * 2];
    }

    let mut result = Vec::with_capacity(num_elements * 2);

    for g in 0..num_groups {
        let scale_offset = scales_start + g * 2;
        let scale_bits =
            u16::from_le_bytes(payload[scale_offset..scale_offset + 2].try_into().unwrap());
        let scale = f16_to_f32(scale_bits);

        let start = g * group_size;
        let end = (start + group_size).min(num_elements);
        for i in start..end {
            let q = payload[data_start + i] as i8;
            let val = q as f32 * scale;
            result.extend_from_slice(&f32_to_f16(val).to_le_bytes());
        }
    }

    result
}

/// Quantize float16 data to int4 (packed) with per-group absmax scaling.
///
/// Input: raw bytes of float16 elements.
/// Output: serialized quantized payload (see wire layout above).
pub fn quantize_int4(f16_data: &[u8]) -> Vec<u8> {
    let num_elements = f16_data.len() / 2;
    let group_size = DEFAULT_GROUP_SIZE;
    let num_groups = num_elements.div_ceil(group_size);

    // Parse float16 values.
    let values: Vec<f32> = (0..num_elements)
        .map(|i| {
            let bits = u16::from_le_bytes([f16_data[i * 2], f16_data[i * 2 + 1]]);
            f16_to_f32(bits)
        })
        .collect();

    // Compute per-group scales and quantized values.
    let mut scales = Vec::with_capacity(num_groups);
    let mut quantized_nibbles = Vec::with_capacity(num_elements);

    for g in 0..num_groups {
        let start = g * group_size;
        let end = (start + group_size).min(num_elements);
        let group = &values[start..end];

        let absmax = group.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = if absmax == 0.0 { 1.0 } else { absmax / 7.0 };
        scales.push(f32_to_f16(scale));

        for &val in group {
            // Quantize to signed 4-bit range [-7, 7], stored as unsigned [0, 15].
            let q = (val / scale).round().clamp(-7.0, 7.0) as i8;
            quantized_nibbles.push(((q + 8) as u8) & 0x0F);
        }
    }

    // Pack two nibbles per byte (low nibble first).
    let packed_len = num_elements.div_ceil(2);
    let mut packed = Vec::with_capacity(packed_len);
    for i in (0..num_elements).step_by(2) {
        let lo = quantized_nibbles[i];
        let hi = if i + 1 < num_elements {
            quantized_nibbles[i + 1]
        } else {
            0
        };
        packed.push(lo | (hi << 4));
    }

    // Build output buffer.
    let mut out = Vec::with_capacity(8 + num_groups * 2 + packed_len);
    out.extend_from_slice(&(num_groups as u32).to_le_bytes());
    out.extend_from_slice(&(group_size as u32).to_le_bytes());
    for &s in &scales {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out.extend_from_slice(&packed);

    out
}

/// Dequantize int4 (packed) data back to float16.
///
/// Input: serialized quantized payload.
/// Output: raw bytes of float16 elements.
///
/// # Panics
///
/// Panics if the payload is malformed (too short for declared groups/data).
/// Callers receiving untrusted data should validate payload length first.
pub fn dequantize_int4(payload: &[u8], num_elements: usize) -> Vec<u8> {
    if payload.len() < 8 {
        return vec![0u8; num_elements * 2];
    }

    let num_groups = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
    let group_size = u32::from_le_bytes(payload[4..8].try_into().unwrap()) as usize;

    if group_size == 0 || num_groups == 0 {
        return vec![0u8; num_elements * 2];
    }

    let scales_start: usize = 8;
    let scales_end = scales_start.saturating_add(num_groups.saturating_mul(2));
    let data_start = scales_end;
    let packed_len = num_elements.div_ceil(2);

    // Validate payload has enough data for scales and packed values.
    let required_len = data_start.saturating_add(packed_len);
    if payload.len() < required_len {
        return vec![0u8; num_elements * 2];
    }

    // Unpack nibbles.
    let packed = &payload[data_start..];
    let mut nibbles = Vec::with_capacity(num_elements);
    for &byte in packed {
        nibbles.push(byte & 0x0F);
        nibbles.push((byte >> 4) & 0x0F);
    }
    nibbles.truncate(num_elements);

    let mut result = Vec::with_capacity(num_elements * 2);

    for g in 0..num_groups {
        let scale_offset = scales_start + g * 2;
        let scale_bits =
            u16::from_le_bytes(payload[scale_offset..scale_offset + 2].try_into().unwrap());
        let scale = f16_to_f32(scale_bits);

        let start = g * group_size;
        let end = (start + group_size).min(num_elements);
        for &nibble in &nibbles[start..end] {
            let q = nibble as i8 - 8; // Convert back from [0,15] to [-8,7].
            let val = q as f32 * scale;
            result.extend_from_slice(&f32_to_f16(val).to_le_bytes());
        }
    }

    result
}

/// Compute the bandwidth reduction ratio for a given quantization mode.
///
/// Returns the approximate ratio of quantized size to original size.
pub fn bandwidth_ratio(mode: &super::tensor_protocol::QuantizationMode) -> f64 {
    match mode {
        super::tensor_protocol::QuantizationMode::None => 1.0,
        // int8: 1 byte per element + scales overhead (~1.5%)
        super::tensor_protocol::QuantizationMode::Int8 => 0.51,
        // int4: 0.5 bytes per element + scales overhead (~1.5%)
        super::tensor_protocol::QuantizationMode::Int4 => 0.26,
    }
}

// --- IEEE 754 half-precision helpers ---

/// Convert a float16 bit pattern to float32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            // Zero.
            f32::from_bits(sign << 31)
        } else {
            // Subnormal: normalize.
            let mut e = exponent;
            let mut m = mantissa;
            while m & 0x400 == 0 {
                m <<= 1;
                e = e.wrapping_sub(1);
            }
            m &= 0x3FF;
            let f32_exp = (127u32 - 15 + 1).wrapping_add(e);
            f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13))
        }
    } else if exponent == 31 {
        // Inf or NaN.
        if mantissa == 0 {
            f32::from_bits((sign << 31) | (0xFF << 23))
        } else {
            f32::from_bits((sign << 31) | (0xFF << 23) | (mantissa << 13))
        }
    } else {
        // Normal.
        let f32_exp = exponent + (127 - 15);
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mantissa << 13))
    }
}

/// Convert a float32 value to a float16 bit pattern.
fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7FFFFF;

    if exponent == 255 {
        // Inf or NaN.
        if mantissa == 0 {
            (sign << 15) | 0x7C00
        } else {
            (sign << 15) | 0x7C00 | ((mantissa >> 13) as u16).max(1)
        }
    } else if exponent > 142 {
        // Overflow to infinity.
        (sign << 15) | 0x7C00
    } else if exponent < 113 {
        // Underflow to zero or subnormal.
        if exponent < 103 {
            sign << 15
        } else {
            let shift = 125 - exponent;
            let m = (mantissa | 0x800000) >> (shift + 1);
            (sign << 15) | (m >> 13) as u16
        }
    } else {
        let f16_exp = (exponent - 112) as u16;
        let f16_man = (mantissa >> 13) as u16;
        (sign << 15) | (f16_exp << 10) | f16_man
    }
}

#[cfg(test)]
#[path = "tensor_quantize_tests.rs"]
mod tests;
