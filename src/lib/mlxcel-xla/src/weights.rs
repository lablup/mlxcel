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

/// Dequantize one MLX affine-quantized weight to row-major `[out, in]` f32.
///
/// `packed` is the row-major `[out, in_packed]` u32 weight (little-endian bytes,
/// `in_packed = in * bits / 32`); `scales` / `biases` are the row-major
/// `[out, in/group_size]` f16 buffers. Each weight is recovered as
/// `w[o,i] = q[o,i] * scale[o, i/group_size] + bias[o, i/group_size]`, where `q`
/// is the `bits`-wide value unpacked low-order-first from `packed[o, i/(32/bits)]`
/// (the MLX affine layout). The graph runs in f32, so the packed weights are
/// widened here once at load.
pub(crate) fn dequantize_affine(
    packed: &[u8],
    scales: &[u8],
    biases: &[u8],
    out: usize,
    in_packed: usize,
    bits: usize,
    group_size: usize,
) -> Result<Vec<f32>, String> {
    if !(bits == 4 || bits == 8) {
        return Err(format!(
            "unsupported quantization bits {bits} (expected 4 or 8)"
        ));
    }
    let per_u32 = 32 / bits; // values packed per u32
    let in_ = in_packed * per_u32;
    if group_size == 0 || !in_.is_multiple_of(group_size) {
        return Err(format!(
            "quantization group_size {group_size} does not divide in dimension {in_}"
        ));
    }
    let n_groups = in_ / group_size;
    if packed.len() != out * in_packed * 4 {
        return Err(format!(
            "packed weight is {} bytes, expected {} ([{out}, {in_packed}] u32)",
            packed.len(),
            out * in_packed * 4
        ));
    }
    let scales = f16_to_f32(scales);
    let biases = f16_to_f32(biases);
    if scales.len() != out * n_groups || biases.len() != out * n_groups {
        return Err(format!(
            "scales/biases have {}/{} elements, expected {} ([{out}, {n_groups}])",
            scales.len(),
            biases.len(),
            out * n_groups
        ));
    }
    let mask: u32 = (1u32 << bits) - 1;
    let mut w = vec![0f32; out * in_];
    for o in 0..out {
        let row = &packed[o * in_packed * 4..(o + 1) * in_packed * 4];
        let grow = o * n_groups;
        let wrow = o * in_;
        for p in 0..in_packed {
            let u =
                u32::from_le_bytes([row[p * 4], row[p * 4 + 1], row[p * 4 + 2], row[p * 4 + 3]]);
            for j in 0..per_u32 {
                let i = p * per_u32 + j;
                let q = ((u >> (bits * j)) & mask) as f32;
                let g = i / group_size;
                w[wrow + i] = q * scales[grow + g] + biases[grow + g];
            }
        }
    }
    Ok(w)
}

/// Dequantize a STACKED mlx-lm affine-quantized expert weight (issue #500) to
/// row-major `[experts, out, in]` f32. The MoE `switch_mlp` projections pack all
/// `experts` into one `[experts, out, in_packed]` U32 tensor with companion
/// `[experts, out, in/group_size]` f16 `scales` / `biases`; this dequantizes each
/// expert's `[out, in_packed]` slab with [`dequantize_affine`] and concatenates
/// them, so the loader hands the emitter's `[E, out, in]` expert arg one f32
/// buffer. Byte-for-byte identical to dequantizing each expert separately.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dequantize_affine_stacked(
    packed: &[u8],
    scales: &[u8],
    biases: &[u8],
    experts: usize,
    out: usize,
    in_packed: usize,
    bits: usize,
    group_size: usize,
) -> Result<Vec<f32>, String> {
    if experts == 0 {
        return Err("stacked expert weight has 0 experts".to_string());
    }
    if !(bits == 4 || bits == 8) {
        return Err(format!(
            "unsupported quantization bits {bits} (expected 4 or 8)"
        ));
    }
    let per_u32 = 32 / bits;
    let in_ = in_packed * per_u32;
    if group_size == 0 || !in_.is_multiple_of(group_size) {
        return Err(format!(
            "quantization group_size {group_size} does not divide in dimension {in_}"
        ));
    }
    let n_groups = in_ / group_size;
    // Per-expert strides: the U32 weight is 4 bytes/element, the f16 scales/biases
    // 2 bytes/element. `dequantize_affine` re-validates each slab's exact sizes.
    let packed_stride = out * in_packed * 4;
    let sb_stride = out * n_groups * 2;
    if packed.len() != experts * packed_stride {
        return Err(format!(
            "stacked packed weight is {} bytes, expected {} ([{experts}, {out}, {in_packed}] u32)",
            packed.len(),
            experts * packed_stride
        ));
    }
    if scales.len() != experts * sb_stride || biases.len() != experts * sb_stride {
        return Err(format!(
            "stacked scales/biases have {}/{} bytes, expected {} ([{experts}, {out}, {n_groups}] f16)",
            scales.len(),
            biases.len(),
            experts * sb_stride
        ));
    }
    let mut w = Vec::with_capacity(experts * out * in_);
    for e in 0..experts {
        let p = &packed[e * packed_stride..(e + 1) * packed_stride];
        let s = &scales[e * sb_stride..(e + 1) * sb_stride];
        let bi = &biases[e * sb_stride..(e + 1) * sb_stride];
        let slab = dequantize_affine(p, s, bi, out, in_packed, bits, group_size)?;
        w.extend_from_slice(&slab);
    }
    Ok(w)
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

    /// 4-bit affine dequant on a hand-built row: one u32 packs eight nibbles
    /// 1..=8 (low-order first), two groups of 4 with scale/bias (2.0, +10) and
    /// (0.5, -1), so `q*scale + bias` is exact.
    #[test]
    fn dequantize_affine_recovers_hand_example() {
        // u32 = 0x8765_4321 -> nibbles [1,2,3,4,5,6,7,8] low-order first.
        let packed = [0x21u8, 0x43, 0x65, 0x87];
        let scales = [0x00u8, 0x40, 0x00, 0x38]; // f16 [2.0, 0.5]
        let biases = [0x00u8, 0x49, 0x00, 0xBC]; // f16 [10.0, -1.0]
        let w = dequantize_affine(&packed, &scales, &biases, 1, 1, 4, 4).unwrap();
        assert_eq!(w, vec![12.0, 14.0, 16.0, 18.0, 1.5, 2.0, 2.5, 3.0]);
    }

    /// 8-bit affine dequant: one u32 packs four bytes 10/20/30/40 (low-order
    /// first), two groups of 2 with scale/bias (2.0, +10) and (0.5, -1), so
    /// `q*scale + bias` is exact. Exercises the `bits = 8` (`per_u32 = 4`) path.
    #[test]
    fn dequantize_affine_8bit_recovers_hand_example() {
        // u32 = 0x281E_140A -> bytes [10, 20, 30, 40] low-order first.
        let packed = [0x0Au8, 0x14, 0x1E, 0x28];
        let scales = [0x00u8, 0x40, 0x00, 0x38]; // f16 [2.0, 0.5]
        let biases = [0x00u8, 0x49, 0x00, 0xBC]; // f16 [10.0, -1.0]
        let w = dequantize_affine(&packed, &scales, &biases, 1, 1, 8, 2).unwrap();
        assert_eq!(w, vec![30.0, 50.0, 14.0, 19.0]);
    }

    /// A packed buffer whose size disagrees with `[out, in_packed]` is rejected.
    #[test]
    fn dequantize_affine_rejects_size_mismatch() {
        let packed = [0u8; 4];
        let sb = [0u8; 4];
        assert!(dequantize_affine(&packed, &sb, &sb, 2, 1, 4, 4).is_err());
    }

    /// The stacked (rank-3) expert dequant is exactly the per-expert dequant
    /// concatenated: two experts, each the 4-bit hand example, yield that row
    /// twice. Exercises the `[experts, out, in_packed]` slab strides (issue #500).
    #[test]
    fn dequantize_affine_stacked_concatenates_expert_slabs() {
        // One expert's inputs (the `dequantize_affine_recovers_hand_example` row).
        let packed1 = [0x21u8, 0x43, 0x65, 0x87];
        let scales1 = [0x00u8, 0x40, 0x00, 0x38]; // f16 [2.0, 0.5]
        let biases1 = [0x00u8, 0x49, 0x00, 0xBC]; // f16 [10.0, -1.0]
        let one = dequantize_affine(&packed1, &scales1, &biases1, 1, 1, 4, 4).unwrap();

        // Stack two identical experts.
        let packed: Vec<u8> = packed1.iter().chain(&packed1).copied().collect();
        let scales: Vec<u8> = scales1.iter().chain(&scales1).copied().collect();
        let biases: Vec<u8> = biases1.iter().chain(&biases1).copied().collect();
        let stacked = dequantize_affine_stacked(&packed, &scales, &biases, 2, 1, 1, 4, 4).unwrap();

        let mut expected = one.clone();
        expected.extend_from_slice(&one);
        assert_eq!(stacked, expected);
        assert_eq!(stacked.len(), 2 * 8, "two experts x eight recovered values");
    }

    /// A stacked buffer whose size disagrees with `[experts, out, in_packed]` is
    /// rejected, so a mis-shaped expert bank fails loudly rather than mis-loading.
    #[test]
    fn dequantize_affine_stacked_rejects_size_mismatch() {
        let packed = [0u8; 4]; // one expert's worth, but experts = 2 declared
        let sb = [0u8; 4];
        assert!(dequantize_affine_stacked(&packed, &sb, &sb, 2, 1, 1, 4, 4).is_err());
    }
}
