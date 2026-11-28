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
//
// Portions of this file are derived from turboquant_plus
// (https://github.com/TheTom/turboquant_plus), Copyright 2026 Tom Turney,
// licensed under the Apache License, Version 2.0. See the top-level NOTICE
// file for the attribution carried forward under Apache-2.0 Section 4(d).

//! 3-bit pack/unpack helpers for `KVCacheMode::Turbo3Asym`.
//!
//! 3 bits do not divide a byte cleanly, so we pack **8 coordinates into 3
//! bytes** (8 × 3 = 24 bits = 3 bytes). The cache calls into these helpers
//! one (b, h, t) row at a time; for `head_dim ∈ {64, 80, 96, 128, 192, 256}`
//! the per-token byte count is `head_dim * 3 / 8 ∈ {24, 30, 36, 48, 72, 96}`,
//! all integer-aligned.
//!
//! # Bit layout
//!
//! Each group of 8 consecutive 3-bit indices (`i0..=i7`) is packed
//! little-endian-bit into 3 bytes (`b0`, `b1`, `b2`):
//!
//! ```text
//!   bit:        7 6 5 4 3 2 1 0
//!   b0:         i2[0] i1[2..0] i0[2..0]
//!   b1:         i5[1..0] i4[2..0] i3[2..0] i2[2..1]
//!   b2:         i7[2..0] i6[2..0] i5[2]
//! ```
//!
//! Equivalently, the packed bit position for coordinate `k` (within the
//! 8-coord group) is `bit_off = k * 3` and the index occupies bits
//! `bit_off..bit_off+3` in the 24-bit little-endian word
//! `b0 | (b1 << 8) | (b2 << 16)`.
//!
//! This layout is the conventional "stream the bits little-endian" packing
//! used by every other 3-bit coder (e.g. q3_K_M in llama.cpp). Picking it
//! means an on-device unpack can be expressed as
//! `(word >> (k * 3)) & 0b111` once the three bytes have been loaded into a
//! single 32-bit lane.
//!
//! # Group alignment
//!
//! Callers must ensure `head_dim` is a multiple of 8 — the head-dim grid
//! in this codebase (64, 80, 96, 128, 192, 256) is all multiples of 8, but
//! the assertion below catches any future architecture that breaks this.
//! Within a single `(b, h, t)` row the groups are stored back-to-back, so
//! offsets stay byte-aligned regardless of which group you are reading.
//!
//! # Host and device forms
//!
//! The cache packs and unpacks on the device: `pack_3bit_on_device` and
//! `unpack_3bit_on_device` express the layout above in MLX integer ops
//! (shift, or, and) over a 32-bit lane per group, so neither the quantize nor
//! the dequantize path evaluates the graph or copies the packed bytes to host
//! memory (#3274). They are lazy like every other graph op.
//!
//! The host functions ([`pack_3bit_indices`], [`unpack_3bit_indices`] and the
//! per-token wrappers) define the layout and are the reference the device
//! forms are tested against, byte for byte. Production code does not call
//! them; the host reference codec in `quant3`'s tests and
//! `examples/turbo3_readback_share_bench.rs` do.
//!
//! Used by: [`super::quant3`] (3-bit V-side compression for
//! `KVCacheMode::Turbo3Asym`).

use cxx::UniquePtr;

use crate::dtype;
use crate::ffi;
use crate::ffi::MlxArray;

/// Number of coordinates packed into a single 24-bit "group". Fixed at 8 so
/// each group occupies exactly 3 bytes (no partial-byte tails).
pub const COORDS_PER_GROUP: usize = 8;

/// Number of bytes per group of [`COORDS_PER_GROUP`] indices. Always 3.
pub const BYTES_PER_GROUP: usize = 3;

/// V-side bit width for `KVCacheMode::Turbo3Asym`.
pub const V_BIT_WIDTH_3: u8 = 3;

/// Compute the number of packed bytes needed to store `head_dim` 3-bit
/// indices for one token.
///
/// Panics in every build profile if `head_dim` is not a positive multiple of
/// [`COORDS_PER_GROUP`] (8): the result sizes downstream packed buffers, so a
/// silently truncated count must never escape (#234). Callers are expected to
/// have validated `head_dim` against the supported grid before calling;
/// `TurboQuantParams3::new` already enforces it at init.
#[inline]
pub fn packed_bytes_per_token_3bit(head_dim: i32) -> i32 {
    assert!(
        head_dim > 0 && (head_dim as usize).is_multiple_of(COORDS_PER_GROUP),
        "packed_bytes_per_token_3bit: head_dim must be a positive multiple of 8; \
         got {head_dim}"
    );
    head_dim * V_BIT_WIDTH_3 as i32 / 8
}

/// Pack a slice of 3-bit indices (each in `0..8`) into a contiguous byte buffer.
///
/// `indices.len()` must be a multiple of [`COORDS_PER_GROUP`] (8). The output
/// has length `indices.len() * 3 / 8`.
///
/// Each input value is masked to its low 3 bits before packing — out-of-range
/// values silently lose their high bits rather than triggering a panic, so
/// callers must enforce `idx < 8` themselves if exact rejection is required.
/// (In the Turbo3 path the centroid lookup already returns indices in
/// `0..2^bit_width`, so the mask is defensive rather than load-bearing.)
///
/// Used by: [`pack_3bit_per_token`] (the host reference packing).
pub fn pack_3bit_indices(indices: &[u8], out: &mut [u8]) {
    assert!(
        indices.len().is_multiple_of(COORDS_PER_GROUP),
        "pack_3bit_indices: index count ({}) must be a multiple of {}",
        indices.len(),
        COORDS_PER_GROUP
    );
    let n_groups = indices.len() / COORDS_PER_GROUP;
    assert_eq!(
        out.len(),
        n_groups * BYTES_PER_GROUP,
        "pack_3bit_indices: output buffer must be {} bytes, got {}",
        n_groups * BYTES_PER_GROUP,
        out.len()
    );
    for g in 0..n_groups {
        let in_off = g * COORDS_PER_GROUP;
        let out_off = g * BYTES_PER_GROUP;
        // Build the 24-bit little-endian word, then split into bytes.
        let mut word: u32 = 0;
        for k in 0..COORDS_PER_GROUP {
            let v = (indices[in_off + k] & 0x07) as u32;
            word |= v << (k as u32 * 3);
        }
        out[out_off] = (word & 0xFF) as u8;
        out[out_off + 1] = ((word >> 8) & 0xFF) as u8;
        out[out_off + 2] = ((word >> 16) & 0xFF) as u8;
    }
}

/// Inverse of [`pack_3bit_indices`]: expand 3-byte groups back into 8-byte
/// runs of u8 indices in `0..8`.
///
/// `packed.len()` must be a multiple of [`BYTES_PER_GROUP`] (3). Each output
/// byte is in `0..8` (only the low 3 bits are written).
///
/// Used by: [`unpack_3bit_per_token`] (the host reference unpacking).
pub fn unpack_3bit_indices(packed: &[u8], out: &mut [u8]) {
    assert!(
        packed.len().is_multiple_of(BYTES_PER_GROUP),
        "unpack_3bit_indices: packed byte count ({}) must be a multiple of {}",
        packed.len(),
        BYTES_PER_GROUP
    );
    let n_groups = packed.len() / BYTES_PER_GROUP;
    assert_eq!(
        out.len(),
        n_groups * COORDS_PER_GROUP,
        "unpack_3bit_indices: output buffer must be {} bytes, got {}",
        n_groups * COORDS_PER_GROUP,
        out.len()
    );
    for g in 0..n_groups {
        let in_off = g * BYTES_PER_GROUP;
        let out_off = g * COORDS_PER_GROUP;
        let word = (packed[in_off] as u32)
            | ((packed[in_off + 1] as u32) << 8)
            | ((packed[in_off + 2] as u32) << 16);
        for k in 0..COORDS_PER_GROUP {
            out[out_off + k] = ((word >> (k as u32 * 3)) & 0x07) as u8;
        }
    }
}

/// Pack `total_tokens` × `head_dim` indices (laid out as a flat row-major
/// `[token, coord]` slice) into a flat row-major `[token, packed_byte]` buffer.
///
/// Convenience wrapper around [`pack_3bit_indices`] that handles the per-token
/// stride bookkeeping. Output buffer must be sized
/// `total_tokens * packed_bytes_per_token_3bit(head_dim)`.
///
/// Used by: the host reference quantizer in `quant3`'s tests and the parity
/// tests of `pack_3bit_on_device`. The cache packs on the device.
pub fn pack_3bit_per_token(indices: &[u8], head_dim: i32, total_tokens: usize) -> Vec<u8> {
    let bytes_per_token = packed_bytes_per_token_3bit(head_dim) as usize;
    let coords_per_token = head_dim as usize;
    assert_eq!(
        indices.len(),
        total_tokens * coords_per_token,
        "pack_3bit_per_token: index count mismatch"
    );
    let mut out = vec![0u8; total_tokens * bytes_per_token];
    for tok in 0..total_tokens {
        let in_off = tok * coords_per_token;
        let out_off = tok * bytes_per_token;
        pack_3bit_indices(
            &indices[in_off..in_off + coords_per_token],
            &mut out[out_off..out_off + bytes_per_token],
        );
    }
    out
}

/// Unpack a `total_tokens` × `bytes_per_token` packed buffer back into a flat
/// `[token, coord]` index array.
///
/// Output values are in `0..8`.
///
/// Used by: the host reference dequantizer in `quant3`'s tests, the parity
/// tests of `unpack_3bit_on_device` and
/// `examples/turbo3_readback_share_bench.rs`. The cache unpacks on the device.
pub fn unpack_3bit_per_token(packed: &[u8], head_dim: i32, total_tokens: usize) -> Vec<u8> {
    let bytes_per_token = packed_bytes_per_token_3bit(head_dim) as usize;
    let coords_per_token = head_dim as usize;
    assert_eq!(
        packed.len(),
        total_tokens * bytes_per_token,
        "unpack_3bit_per_token: packed byte count mismatch"
    );
    let mut out = vec![0u8; total_tokens * coords_per_token];
    for tok in 0..total_tokens {
        let in_off = tok * bytes_per_token;
        let out_off = tok * coords_per_token;
        unpack_3bit_indices(
            &packed[in_off..in_off + bytes_per_token],
            &mut out[out_off..out_off + coords_per_token],
        );
    }
    out
}

// ---------------------------------------------------------------------------
// Device forms
// ---------------------------------------------------------------------------

/// Bit offset of each of the 8 indices inside its 24-bit group word.
const GROUP_SHIFTS: [u32; COORDS_PER_GROUP] = [0, 3, 6, 9, 12, 15, 18, 21];

/// A one-element UINT32 constant that broadcasts against any lane.
fn u32_scalar(value: u32) -> UniquePtr<MlxArray> {
    ffi::from_slice_u32(&[value], &[1])
}

/// Checks a `[B, H, T, last]` tensor against the expected last axis and
/// returns `(b, h, t)`. Unconditional: the reshapes below are computed from
/// `head_dim`, and a mismatch would otherwise surface as an MLX throw across a
/// `noexcept` bridge call, which aborts the process instead of unwinding.
fn checked_bht(name: &str, shape: &[i32], last: i32, what: &str) -> (i32, i32, i32) {
    assert_eq!(
        shape.len(),
        4,
        "{name}: input must be 4-D [B, H, T, {what}]"
    );
    assert_eq!(
        shape[3], last,
        "{name}: last axis must be {what} = {last}; got shape {shape:?}"
    );
    (shape[0], shape[1], shape[2])
}

/// Device form of [`pack_3bit_per_token`]: packs a `[B, H, T, head_dim]` UINT8
/// index tensor into `[B, H, T, head_dim * 3 / 8]` UINT8, byte for byte the
/// host layout documented in the module header.
///
/// Each group of 8 indices is widened to UINT32, shifted to its bit offset
/// (`k * 3`) and OR-ed into one 24-bit word, whose three bytes are then split
/// out. OR is used instead of a sum so the word stays UINT32 with no dtype
/// promotion. As on the host, each index is masked to its low 3 bits first, so
/// an out-of-range value loses its high bits instead of corrupting a
/// neighbour.
///
/// The result is lazy. `T == 0` yields a `[B, H, 0, head_dim * 3 / 8]` tensor.
///
/// # Panics
///
/// Panics if `head_dim` is not a positive multiple of 8 or if `idx_u8` is not
/// `[B, H, T, head_dim]`.
///
/// Used by: [`super::quant3::quantize_v_turbo3`].
pub(crate) fn pack_3bit_on_device(idx_u8: &MlxArray, head_dim: i32) -> UniquePtr<MlxArray> {
    let bytes_per_token = packed_bytes_per_token_3bit(head_dim);
    let shape = ffi::array_shape(idx_u8);
    let (b, h, t) = checked_bht("pack_3bit_on_device", &shape, head_dim, "head_dim");
    let groups = head_dim / COORDS_PER_GROUP as i32;

    let lanes = ffi::astype(
        &ffi::reshape(idx_u8, &[b, h, t, groups, COORDS_PER_GROUP as i32]),
        dtype::UINT32,
    );
    let masked = ffi::bitwise_and(&lanes, &u32_scalar(0x07));
    let shifts = ffi::from_slice_u32(&GROUP_SHIFTS, &[COORDS_PER_GROUP as i32]);
    let shifted = ffi::left_shift(&masked, &shifts);
    // The 8 lanes occupy disjoint bits of the word, so OR assembles it.
    let lane = |k: i32| {
        ffi::reshape(
            &ffi::slice(&shifted, &[0, 0, 0, 0, k], &[b, h, t, groups, k + 1]),
            &[b, h, t, groups],
        )
    };
    let mut word = lane(0);
    for k in 1..COORDS_PER_GROUP as i32 {
        word = ffi::bitwise_or(&word, &lane(k));
    }
    let byte_mask = u32_scalar(0xFF);
    let b0 = ffi::bitwise_and(&word, &byte_mask);
    let b1 = ffi::bitwise_and(&ffi::right_shift(&word, &u32_scalar(8)), &byte_mask);
    let b2 = ffi::right_shift(&word, &u32_scalar(16));
    let bytes = crate::ops::stack_owned(&[b0, b1, b2], -1);
    ffi::astype(
        &ffi::reshape(&bytes, &[b, h, t, bytes_per_token]),
        dtype::UINT8,
    )
}

/// Device form of [`unpack_3bit_per_token`]: expands a
/// `[B, H, T, head_dim * 3 / 8]` UINT8 packed tensor into `[B, H, T, head_dim]`
/// UINT8 indices in `0..8`, element for element the host result.
///
/// Each 3-byte group is widened to UINT32 and assembled into the 24-bit word
/// `b0 | b1 << 8 | b2 << 16`; index `k` is `(word >> (k * 3)) & 7`, with the
/// mask applied after the lanes are narrowed back to UINT8.
///
/// The result is lazy. `T == 0` yields a `[B, H, 0, head_dim]` tensor.
///
/// # Panics
///
/// Panics if `head_dim` is not a positive multiple of 8 or if `packed_u8` is
/// not `[B, H, T, head_dim * 3 / 8]`.
///
/// Used by: [`super::quant3::dequantize_v_turbo3`].
pub(crate) fn unpack_3bit_on_device(packed_u8: &MlxArray, head_dim: i32) -> UniquePtr<MlxArray> {
    let bytes_per_token = packed_bytes_per_token_3bit(head_dim);
    let shape = ffi::array_shape(packed_u8);
    let (b, h, t) = checked_bht(
        "unpack_3bit_on_device",
        &shape,
        bytes_per_token,
        "head_dim * 3 / 8",
    );
    let groups = head_dim / COORDS_PER_GROUP as i32;

    let bytes = ffi::astype(
        &ffi::reshape(packed_u8, &[b, h, t, groups, BYTES_PER_GROUP as i32]),
        dtype::UINT32,
    );
    let byte = |i: i32| ffi::slice(&bytes, &[0, 0, 0, 0, i], &[b, h, t, groups, i + 1]);
    // `[B, H, T, groups, 1]`: one 24-bit word per group, kept with a trailing
    // axis so the 8 shifts broadcast against it.
    let word = ffi::bitwise_or(
        &ffi::bitwise_or(&byte(0), &ffi::left_shift(&byte(1), &u32_scalar(8))),
        &ffi::left_shift(&byte(2), &u32_scalar(16)),
    );
    let shifts = ffi::from_slice_u32(&GROUP_SHIFTS, &[COORDS_PER_GROUP as i32]);
    // Narrow to UINT8 before the 3-bit mask: the mask only reads the low three
    // bits, so the result is the same, and the `[B, H, T, groups, 8]` lanes are
    // masked as bytes instead of as a second UINT32 tensor of that shape.
    let lanes = ffi::astype(&ffi::right_shift(&word, &shifts), dtype::UINT8);
    let indices = ffi::bitwise_and(&lanes, &ffi::full_f32(&[1], 7.0, dtype::UINT8));
    ffi::reshape(&indices, &[b, h, t, head_dim])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_bytes_matches_supported_head_dims() {
        // spec: head_dim ∈ {64, 80, 96, 128, 192, 256} → bytes
        // {24, 30, 36, 48, 72, 96}.
        assert_eq!(packed_bytes_per_token_3bit(64), 24);
        assert_eq!(packed_bytes_per_token_3bit(80), 30);
        assert_eq!(packed_bytes_per_token_3bit(96), 36);
        assert_eq!(packed_bytes_per_token_3bit(128), 48);
        assert_eq!(packed_bytes_per_token_3bit(192), 72);
        assert_eq!(packed_bytes_per_token_3bit(256), 96);
    }

    #[test]
    #[should_panic(expected = "must be a positive multiple of 8")]
    fn packed_bytes_rejects_non_multiple_of_8() {
        // 7 is not a multiple of 8 → must panic in every build profile: the
        // contract sizes downstream packed buffers, and a release build that
        // silently truncated (7 * 3 / 8 = 2) would mis-size them (#234).
        let _ = packed_bytes_per_token_3bit(7);
    }

    /// All-zero indices must round-trip to all-zero packed bytes and back.
    #[test]
    fn round_trip_zero_indices() {
        let indices = vec![0u8; 16]; // two groups
        let mut packed = vec![0u8; 6];
        pack_3bit_indices(&indices, &mut packed);
        assert_eq!(packed, vec![0u8; 6]);
        let mut recovered = vec![0u8; 16];
        unpack_3bit_indices(&packed, &mut recovered);
        assert_eq!(recovered, indices);
    }

    /// All-7 (max representable) indices round-trip exactly. Catches accidental
    /// sign-extension or off-by-one mask errors.
    #[test]
    fn round_trip_max_indices() {
        let indices = vec![7u8; 8];
        let mut packed = vec![0u8; 3];
        pack_3bit_indices(&indices, &mut packed);
        // 8 × 0b111 = 0xFFFFFF in 24 bits.
        assert_eq!(packed, vec![0xFF, 0xFF, 0xFF]);
        let mut recovered = vec![0u8; 8];
        unpack_3bit_indices(&packed, &mut recovered);
        assert_eq!(recovered, indices);
    }

    /// A staircase of indices `0..8` round-trips exactly. Verifies the
    /// little-endian-bit ordering documented in the module header.
    #[test]
    fn round_trip_staircase_indices() {
        let indices: Vec<u8> = (0..8).collect();
        let mut packed = vec![0u8; 3];
        pack_3bit_indices(&indices, &mut packed);
        // Hand-derived: word = 0|1<<3|2<<6|3<<9|4<<12|5<<15|6<<18|7<<21
        //             = 0b111110_101100_011010_001000_000 (read low→high)
        // Easier: compute the expected word in code and compare.
        let mut expected_word: u32 = 0;
        for k in 0..8u32 {
            expected_word |= k << (k * 3);
        }
        let expected = [
            (expected_word & 0xFF) as u8,
            ((expected_word >> 8) & 0xFF) as u8,
            ((expected_word >> 16) & 0xFF) as u8,
        ];
        assert_eq!(packed, expected);

        let mut recovered = vec![0u8; 8];
        unpack_3bit_indices(&packed, &mut recovered);
        assert_eq!(recovered, indices);
    }

    /// Multi-group round-trip with random data exercises the per-group
    /// stride logic.
    #[test]
    fn round_trip_random_multi_group() {
        // 32 indices = 4 groups = 12 bytes packed. Use a deterministic LCG.
        let mut state: u32 = 0xC0FF_EE42;
        let mut indices = vec![0u8; 32];
        for x in &mut indices {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *x = ((state >> 17) & 0x07) as u8; // top 3 bits → 0..8
        }
        let mut packed = vec![0u8; 12];
        pack_3bit_indices(&indices, &mut packed);
        let mut recovered = vec![0u8; 32];
        unpack_3bit_indices(&packed, &mut recovered);
        assert_eq!(recovered, indices, "random round-trip failed");
    }

    /// Per-token packing for a realistic head_dim=128 → 48 bytes/token, and
    /// 4 tokens.
    #[test]
    fn round_trip_per_token_head_dim_128() {
        let head_dim = 128_i32;
        let total_tokens = 4_usize;
        let mut state: u32 = 0xBADC_AFE0;
        let total_coords = total_tokens * head_dim as usize;
        let mut indices = vec![0u8; total_coords];
        for x in &mut indices {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *x = ((state >> 17) & 0x07) as u8;
        }
        let packed = pack_3bit_per_token(&indices, head_dim, total_tokens);
        assert_eq!(packed.len(), total_tokens * 48);
        let recovered = unpack_3bit_per_token(&packed, head_dim, total_tokens);
        assert_eq!(recovered, indices);
    }

    /// head_dim=80 (Mistral-class). 80 * 3 = 240 bits = exactly 30 bytes per
    /// token; verifies the awkward but valid alignment's
    /// design notes.
    #[test]
    fn round_trip_per_token_head_dim_80() {
        let head_dim = 80_i32;
        assert_eq!(packed_bytes_per_token_3bit(head_dim), 30);
        let total_tokens = 3_usize;
        let total_coords = total_tokens * head_dim as usize;
        let indices: Vec<u8> = (0..total_coords)
            .map(|i| ((i * 31 + 5) % 8) as u8)
            .collect();
        let packed = pack_3bit_per_token(&indices, head_dim, total_tokens);
        assert_eq!(packed.len(), total_tokens * 30);
        let recovered = unpack_3bit_per_token(&packed, head_dim, total_tokens);
        assert_eq!(recovered, indices);
    }

    /// Out-of-range indices (>= 8) must be silently masked to their low 3 bits
    /// rather than triggering UB. Documents the defensive-mask contract.
    #[test]
    fn out_of_range_indices_are_masked() {
        let indices = vec![0xFFu8; 8]; // would overflow the 3-bit field
        let mut packed = vec![0u8; 3];
        pack_3bit_indices(&indices, &mut packed);
        // After masking each input to 0x07, we expect the same bits as
        // pack(vec![7; 8]).
        assert_eq!(packed, vec![0xFF, 0xFF, 0xFF]);
        let mut recovered = vec![0u8; 8];
        unpack_3bit_indices(&packed, &mut recovered);
        assert_eq!(recovered, vec![7u8; 8]);
    }

    // -- device forms (#3274) --

    /// Deterministic bytes from the LCG the other tests here use.
    fn lcg_bytes(seed: u32, n: usize, mask: u8) -> Vec<u8> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 17) as u8) & mask
            })
            .collect()
    }

    fn u8_tensor(data: &[u8], shape: &[i32]) -> UniquePtr<MlxArray> {
        ffi::from_bytes(data, shape, dtype::UINT8)
    }

    fn evaluated_bytes(arr: &MlxArray) -> Vec<u8> {
        ffi::eval(arr);
        ffi::array_to_raw_bytes(arr)
    }

    /// `pack_3bit_on_device` is `pack_3bit_per_token` byte for byte, and
    /// `unpack_3bit_on_device` inverts it, over every head dim in the
    /// supported grid (including 80, 96 and 192, which are multiples of 8 but
    /// not powers of two) and over B > 1 and H > 1.
    #[test]
    fn device_pack_matches_host_and_round_trips() {
        for &head_dim in &[8_i32, 64, 80, 96, 128, 192, 256] {
            let (b, h, t) = (2_i32, 3_i32, 5_i32);
            let total_tokens = (b * h * t) as usize;
            let indices = lcg_bytes(
                0xC0FF_EE42 ^ head_dim as u32,
                total_tokens * head_dim as usize,
                0x07,
            );
            let idx = u8_tensor(&indices, &[b, h, t, head_dim]);

            let packed = pack_3bit_on_device(&idx, head_dim);
            assert_eq!(ffi::array_dtype(&packed), dtype::UINT8);
            assert_eq!(
                ffi::array_shape(&packed),
                vec![b, h, t, packed_bytes_per_token_3bit(head_dim)]
            );
            assert_eq!(
                evaluated_bytes(&packed),
                pack_3bit_per_token(&indices, head_dim, total_tokens),
                "device pack differs from the host pack at head_dim {head_dim}"
            );

            let unpacked = unpack_3bit_on_device(&packed, head_dim);
            assert_eq!(ffi::array_dtype(&unpacked), dtype::UINT8);
            assert_eq!(ffi::array_shape(&unpacked), vec![b, h, t, head_dim]);
            assert_eq!(
                evaluated_bytes(&unpacked),
                indices,
                "device unpack(pack(idx)) != idx at head_dim {head_dim}"
            );
        }
    }

    /// `unpack_3bit_on_device` is `unpack_3bit_per_token` element for element
    /// on arbitrary packed bytes: every byte value occurs, so every bit of
    /// every lane is exercised, not only the patterns a quantizer produces.
    #[test]
    fn device_unpack_matches_host_on_arbitrary_bytes() {
        for &head_dim in &[64_i32, 128, 256] {
            let (b, h, t) = (1_i32, 2_i32, 33_i32);
            let total_tokens = (b * h * t) as usize;
            let bytes_per_token = packed_bytes_per_token_3bit(head_dim);
            let packed = lcg_bytes(
                0xBADC_AFE0 ^ head_dim as u32,
                total_tokens * bytes_per_token as usize,
                0xFF,
            );
            for value in 0..=255_u8 {
                assert!(packed.contains(&value), "fixture misses byte {value}");
            }
            let unpacked =
                unpack_3bit_on_device(&u8_tensor(&packed, &[b, h, t, bytes_per_token]), head_dim);
            assert_eq!(
                evaluated_bytes(&unpacked),
                unpack_3bit_per_token(&packed, head_dim, total_tokens),
                "device unpack differs from the host unpack at head_dim {head_dim}"
            );
        }
    }

    /// The extremes of the index range and of the layout: all zero, all 7, the
    /// staircase that pins the bit order, and a single token in a single head.
    #[test]
    fn device_pack_extreme_patterns_match_host() {
        let head_dim = 64_i32;
        let staircase: Vec<u8> = (0..head_dim as usize).map(|k| (k % 8) as u8).collect();
        for (name, indices) in [
            ("zeros", vec![0_u8; head_dim as usize]),
            ("sevens", vec![7_u8; head_dim as usize]),
            ("staircase", staircase),
        ] {
            let packed = pack_3bit_on_device(&u8_tensor(&indices, &[1, 1, 1, head_dim]), head_dim);
            let host = pack_3bit_per_token(&indices, head_dim, 1);
            assert_eq!(evaluated_bytes(&packed), host, "{name}: device pack");
            let unpacked = unpack_3bit_on_device(&packed, head_dim);
            assert_eq!(evaluated_bytes(&unpacked), indices, "{name}: device unpack");
        }
        // The staircase group is the hand-derived word of the host test above.
        let staircase_group: Vec<u8> = (0..8).collect();
        let packed = pack_3bit_on_device(&u8_tensor(&staircase_group, &[1, 1, 1, 8]), 8);
        assert_eq!(evaluated_bytes(&packed), vec![0x88, 0xC6, 0xFA]);
    }

    /// Out-of-range indices lose their high bits on the device exactly as on
    /// the host (the defensive mask), instead of spilling into a neighbour.
    #[test]
    fn device_pack_masks_out_of_range_indices_like_the_host() {
        let head_dim = 64_i32;
        let total_tokens = 4_usize;
        let indices = lcg_bytes(0x1234_5678, total_tokens * head_dim as usize, 0xFF);
        assert!(indices.iter().any(|&i| i >= 8));
        let packed = pack_3bit_on_device(
            &u8_tensor(&indices, &[1, 1, total_tokens as i32, head_dim]),
            head_dim,
        );
        assert_eq!(
            evaluated_bytes(&packed),
            pack_3bit_per_token(&indices, head_dim, total_tokens)
        );
        let all_ff = vec![0xFF_u8; 8];
        let packed = pack_3bit_on_device(&u8_tensor(&all_ff, &[1, 1, 1, 8]), 8);
        assert_eq!(evaluated_bytes(&packed), vec![0xFF, 0xFF, 0xFF]);
    }

    /// Zero tokens flow through both device forms as zero-size tensors of the
    /// right shape, for B and H above one as well.
    #[test]
    fn device_forms_accept_zero_tokens() {
        for &(b, h, head_dim) in &[(1_i32, 1_i32, 64_i32), (2, 8, 128), (1, 2, 256)] {
            let bytes_per_token = packed_bytes_per_token_3bit(head_dim);
            let no_indices = ffi::zeros(&[b, h, 0, head_dim], dtype::UINT8);
            let packed = pack_3bit_on_device(&no_indices, head_dim);
            assert_eq!(ffi::array_shape(&packed), vec![b, h, 0, bytes_per_token]);
            assert_eq!(ffi::array_dtype(&packed), dtype::UINT8);
            assert!(evaluated_bytes(&packed).is_empty());

            let no_bytes = ffi::zeros(&[b, h, 0, bytes_per_token], dtype::UINT8);
            let unpacked = unpack_3bit_on_device(&no_bytes, head_dim);
            assert_eq!(ffi::array_shape(&unpacked), vec![b, h, 0, head_dim]);
            assert_eq!(ffi::array_dtype(&unpacked), dtype::UINT8);
            assert!(evaluated_bytes(&unpacked).is_empty());
        }
    }

    /// A decode-sized window (32768 tokens, 8 heads, head_dim 128: 33.5 M
    /// indices) packs and unpacks to the host result. The device forms are a
    /// fixed number of elementwise ops, so the cost is linear in the window;
    /// this pins the result at the largest context the issue measures.
    #[test]
    fn device_forms_match_host_at_a_32k_window() {
        let (b, h, t, head_dim) = (1_i32, 8_i32, 32_768_i32, 128_i32);
        let total_tokens = (b * h * t) as usize;
        let indices = lcg_bytes(0x0BAD_F00D, total_tokens * head_dim as usize, 0x07);
        let host_packed = pack_3bit_per_token(&indices, head_dim, total_tokens);

        let packed = pack_3bit_on_device(&u8_tensor(&indices, &[b, h, t, head_dim]), head_dim);
        assert!(evaluated_bytes(&packed) == host_packed, "32k device pack");
        let unpacked = unpack_3bit_on_device(&packed, head_dim);
        assert!(evaluated_bytes(&unpacked) == indices, "32k device unpack");
    }

    #[test]
    #[should_panic(expected = "pack_3bit_on_device: last axis must be head_dim = 64")]
    fn device_pack_rejects_a_head_dim_mismatch() {
        let idx = u8_tensor(&[0_u8; 128], &[1, 1, 1, 128]);
        let _ = pack_3bit_on_device(&idx, 64);
    }

    #[test]
    #[should_panic(expected = "unpack_3bit_on_device: last axis must be head_dim * 3 / 8 = 24")]
    fn device_unpack_rejects_a_packed_width_mismatch() {
        let packed = u8_tensor(&[0_u8; 48], &[1, 1, 1, 48]);
        let _ = unpack_3bit_on_device(&packed, 64);
    }

    #[test]
    #[should_panic(expected = "pack_3bit_on_device: input must be 4-D")]
    fn device_pack_rejects_a_non_4d_input() {
        let idx = u8_tensor(&[0_u8; 64], &[1, 1, 64]);
        let _ = pack_3bit_on_device(&idx, 64);
    }

    #[test]
    #[should_panic(expected = "must be a positive multiple of 8")]
    fn device_unpack_rejects_a_head_dim_that_is_not_a_multiple_of_8() {
        let packed = u8_tensor(&[0_u8; 3], &[1, 1, 1, 3]);
        let _ = unpack_3bit_on_device(&packed, 12);
    }
}
