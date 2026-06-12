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
//! # On-device dequant
//!
//! Unlike the 4-bit (`turbo4`) path, the 3-bit path performs an
//! eval + readback to host memory, expands each packed group to 8 u8s on
//! CPU, and ships the result back as a UINT8 array. This is the same
//! readback pattern the quantize side uses for nearest-centroid lookup,
//! and matches the speed/correctness trade-off documented for
//! `quantize_into_packed` in [`super::quant`]. A pure on-device unpack
//! using `bitwise_and` + `right_shift` + `take` is feasible (the layout is
//! GPU-friendly: 8 coords → 24 bits → one 32-bit lane), but is deferred
//! until after the dense path is validated end-to-end.
//!
//! Used by: [`super::quant3`] (3-bit V-side compression for
//! `KVCacheMode::Turbo3Asym`).

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
/// Used by: [`pack_3bit_per_token`] (per-row packing in the V-side quant
/// pipeline).
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
/// Used by: [`unpack_3bit_per_token`] (per-row unpacking on the V-side
/// dequant path).
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
/// Used by: [`super::quant3::quantize_v_turbo3`].
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
/// Output values are in `0..8`. Used by the dequantize path before the
/// centroid `take()` lookup.
///
/// Used by: [`super::quant3::dequantize_v_turbo3`].
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
}
