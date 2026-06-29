// Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
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

//! Widen safetensors weight bytes to f32 (the dtype the emitted StableHLO graphs
//! take), for the IREE loader (issue #449 M3 Stage 2d). bf16 and f16 are the
//! common checkpoint dtypes; f32 is a passthrough. Every conversion is exact
//! (f32 represents every bf16/f16 value), so the widened weights match HF's own
//! f32 cast, which the token-exact oracle gate depends on.

/// bf16 little-endian bytes -> f32 (bf16 is the high 16 bits of f32).
pub(crate) fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

/// One IEEE 754 half (f16) -> f32. The arithmetic forms are exact: a normal's
/// `1 + mant/1024` is a dyadic with denominator 2^10 and the `2^(exp-15)` / `2^-24`
/// scales are exact powers of two, so the widening is bit-for-bit.
pub(crate) fn half_to_f32(h: u16) -> f32 {
    let sign = if h >> 15 == 1 { -1.0 } else { 1.0 };
    let exp = (h >> 10) & 0x1f;
    let mant = (h & 0x3ff) as f32;
    match exp {
        0 => sign * mant * 2f32.powi(-24),           // zero / subnormal
        0x1f if mant == 0.0 => sign * f32::INFINITY, // +/- inf
        0x1f => f32::NAN,                            // nan
        _ => sign * (1.0 + mant / 1024.0) * 2f32.powi(exp as i32 - 15), // normal
    }
}

/// f16 little-endian bytes -> f32, via a 65536-entry `u16 -> f32` lookup table.
/// The table is built once (every f16 bit pattern, exact) and then each element
/// is a single index, so widening a multi-GB checkpoint is memory-bound rather
/// than arithmetic-bound (an 8B-param checkpoint otherwise spends minutes in
/// per-element `powi`).
pub(crate) fn f16_to_f32(bytes: &[u8]) -> Vec<f32> {
    let table: Vec<f32> = (0..=u16::MAX).map(half_to_f32).collect();
    bytes
        .chunks_exact(2)
        .map(|c| table[u16::from_le_bytes([c[0], c[1]]) as usize])
        .collect()
}

/// f32 little-endian bytes -> f32 (a plain reinterpret, for f32 checkpoints).
pub(crate) fn f32_le_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f16 widening is exact against `f32 as` for representative values: zero, one,
    /// a fraction, a negative, the max normal, and a subnormal.
    #[test]
    fn half_to_f32_matches_reference_values() {
        // (f16 bits, expected f32) pairs.
        let cases: [(u16, f32); 7] = [
            (0x0000, 0.0),            // +0
            (0x8000, -0.0),           // -0
            (0x3c00, 1.0),            // 1.0
            (0x3800, 0.5),            // 0.5
            (0xc000, -2.0),           // -2.0
            (0x7bff, 65504.0),        // max normal f16
            (0x0001, 2f32.powi(-24)), // smallest positive subnormal
        ];
        for (bits, want) in cases {
            let got = half_to_f32(bits);
            assert_eq!(got, want, "f16 {bits:#06x} -> {got} != {want}");
        }
    }

    /// inf / nan f16 encodings widen to f32 inf / nan.
    #[test]
    fn half_to_f32_handles_inf_and_nan() {
        assert!(half_to_f32(0x7c00).is_infinite() && half_to_f32(0x7c00) > 0.0);
        assert!(half_to_f32(0xfc00).is_infinite() && half_to_f32(0xfc00) < 0.0);
        assert!(half_to_f32(0x7e00).is_nan());
    }

    /// The byte converters round-trip a little-endian buffer of two values.
    #[test]
    fn f16_byte_buffer_widens_both_lanes() {
        // 1.0 (0x3c00) then -2.0 (0xc000), little-endian.
        let bytes = [0x00, 0x3c, 0x00, 0xc0];
        assert_eq!(f16_to_f32(&bytes), vec![1.0, -2.0]);
    }

    /// bf16 widening keeps the high 16 bits (1.0 -> 0x3f80).
    #[test]
    fn bf16_byte_buffer_widens() {
        let bytes = [0x80, 0x3f]; // bf16 1.0, little-endian
        assert_eq!(bf16_to_f32(&bytes), vec![1.0]);
    }

    /// f32 passthrough reinterprets 4-byte lanes.
    #[test]
    fn f32_passthrough_reinterprets() {
        let bytes = 1.5f32.to_le_bytes();
        assert_eq!(f32_le_to_f32(&bytes), vec![1.5]);
    }
}
