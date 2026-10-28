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

use anyhow::{Result, anyhow, bail};

/// Default group size for absmax quantization.
pub const DEFAULT_GROUP_SIZE: usize = 128;

/// Upper bound on the int8 quantized payload size for `num_elements` values.
///
/// The layout is `8 + 2 * num_groups + num_elements` bytes, and any group
/// size of at least one gives `num_groups <= num_elements`, so
/// `8 + 3 * num_elements` covers every valid payload. Receivers use it to cap
/// how far an LZ4 block may decompress before [`dequantize_int8`] checks the
/// exact layout. Returns `None` on overflow.
pub fn int8_payload_max_len(num_elements: usize) -> Option<usize> {
    num_elements.checked_mul(3)?.checked_add(8)
}

/// Upper bound on the int4 (packed) quantized payload size for
/// `num_elements` values: `8 + 2 * num_elements + ceil(num_elements / 2)`,
/// by the same reasoning as [`int8_payload_max_len`]. Returns `None` on
/// overflow.
pub fn int4_payload_max_len(num_elements: usize) -> Option<usize> {
    num_elements
        .checked_mul(2)?
        .checked_add(num_elements.div_ceil(2))?
        .checked_add(8)
}

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

/// Header and section offsets of a quantized payload whose layout has been
/// checked against the element count.
struct QuantizedLayout {
    num_groups: usize,
    group_size: usize,
    data_start: usize,
}

/// Validate a quantized payload against `num_elements` before anything is
/// allocated from it.
///
/// The payload must be exactly `8 + 2 * num_groups + data_len` bytes with
/// `group_size >= 1` and `num_groups == ceil(num_elements / group_size)`, all
/// computed with checked arithmetic. `data_len` is the size of the quantized
/// value section (`num_elements` for int8, `ceil(num_elements / 2)` for int4).
fn validate_layout(
    payload: &[u8],
    num_elements: usize,
    data_len: usize,
    kind: &str,
) -> Result<QuantizedLayout> {
    let Some((header, _)) = payload.split_first_chunk::<8>() else {
        bail!(
            "{kind} quantized payload too short for its header: expected at least 8 bytes, got {}",
            payload.len()
        );
    };
    let num_groups = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    let group_size = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
    let num_groups = usize::try_from(num_groups)
        .map_err(|_| anyhow!("{kind} quantized num_groups {num_groups} exceeds usize"))?;
    let group_size = usize::try_from(group_size)
        .map_err(|_| anyhow!("{kind} quantized group_size {group_size} exceeds usize"))?;
    if group_size == 0 {
        bail!("{kind} quantized payload has group_size 0");
    }
    let expected_groups = num_elements.div_ceil(group_size);
    if num_groups != expected_groups {
        bail!(
            "{kind} quantized payload declares num_groups {num_groups}, expected {expected_groups} \
             for {num_elements} elements at group_size {group_size}"
        );
    }
    let expected_len = num_groups
        .checked_mul(2)
        .and_then(|scales| scales.checked_add(8))
        .and_then(|data_start| data_start.checked_add(data_len))
        .ok_or_else(|| {
            anyhow!("{kind} quantized payload size overflows for {num_elements} elements")
        })?;
    if payload.len() != expected_len {
        bail!(
            "{kind} quantized payload length mismatch: expected {expected_len} bytes for \
             {num_elements} elements in {num_groups} groups, got {}",
            payload.len()
        );
    }
    Ok(QuantizedLayout {
        num_groups,
        group_size,
        // Cannot overflow: bounded by `expected_len` above.
        data_start: 8 + 2 * num_groups,
    })
}

/// Read the float16 scale of group `g` from a validated payload.
fn group_scale(payload: &[u8], g: usize) -> f32 {
    let offset = 8 + g * 2;
    f16_to_f32(u16::from_le_bytes([payload[offset], payload[offset + 1]]))
}

/// Dequantize int8 data back to float16.
///
/// Input: serialized quantized payload. Output: raw bytes of `num_elements`
/// float16 values.
///
/// # Errors
///
/// The payload usually comes from a peer, so its layout is validated against
/// `num_elements` before the output is allocated (see the wire layout in the
/// module docs). A short or over-long payload, `group_size == 0`, or a
/// `num_groups` inconsistent with `num_elements` is an error; nothing is
/// zero-filled. Because the payload length must match exactly, the output is
/// at most twice the payload size.
pub fn dequantize_int8(payload: &[u8], num_elements: usize) -> Result<Vec<u8>> {
    let layout = validate_layout(payload, num_elements, num_elements, "int8")?;
    let out_len = num_elements
        .checked_mul(2)
        .ok_or_else(|| anyhow!("int8 dequantized size overflows for {num_elements} elements"))?;
    let values = &payload[layout.data_start..];
    let mut result = Vec::with_capacity(out_len);

    for g in 0..layout.num_groups {
        let scale = group_scale(payload, g);
        let start = g * layout.group_size;
        let end = start.saturating_add(layout.group_size).min(num_elements);
        for &q in &values[start..end] {
            let val = q as i8 as f32 * scale;
            result.extend_from_slice(&f32_to_f16(val).to_le_bytes());
        }
    }

    Ok(result)
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
/// Input: serialized quantized payload. Output: raw bytes of `num_elements`
/// float16 values.
///
/// # Errors
///
/// Validated exactly like [`dequantize_int8`], with a packed value section of
/// `ceil(num_elements / 2)` bytes, so the output is at most four times the
/// payload size.
pub fn dequantize_int4(payload: &[u8], num_elements: usize) -> Result<Vec<u8>> {
    let layout = validate_layout(payload, num_elements, num_elements.div_ceil(2), "int4")?;
    let out_len = num_elements
        .checked_mul(2)
        .ok_or_else(|| anyhow!("int4 dequantized size overflows for {num_elements} elements"))?;
    let packed = &payload[layout.data_start..];
    let mut result = Vec::with_capacity(out_len);

    for g in 0..layout.num_groups {
        let scale = group_scale(payload, g);
        let start = g * layout.group_size;
        let end = start.saturating_add(layout.group_size).min(num_elements);
        for i in start..end {
            let byte = packed[i / 2];
            let nibble = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            let q = nibble as i8 - 8; // Convert back from [0,15] to [-8,7].
            let val = q as f32 * scale;
            result.extend_from_slice(&f32_to_f16(val).to_le_bytes());
        }
    }

    Ok(result)
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
