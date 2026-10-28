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

#[test]
fn test_compress_decompress_roundtrip() {
    let data = vec![0xABu8; 4096];
    let compressed = compress(&data);
    let decompressed = decompress(&compressed, data.len()).unwrap();
    assert_eq!(decompressed, data);
}

#[test]
fn test_compress_decompress_varied_data() {
    let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
    let compressed = compress(&data);
    let decompressed = decompress(&compressed, data.len()).unwrap();
    assert_eq!(decompressed, data);
}

#[test]
fn test_compress_if_beneficial_sparse() {
    // Highly compressible data (all zeros).
    let data = vec![0u8; 8192];
    let result = compress_if_beneficial(&data);
    assert!(
        result.is_some(),
        "sparse data should be deemed beneficial to compress"
    );
    let compressed = result.unwrap();
    assert!(compressed.len() < data.len());

    // Verify roundtrip.
    let decompressed = decompress(&compressed, data.len()).unwrap();
    assert_eq!(decompressed, data);
}

#[test]
fn test_compress_if_beneficial_random() {
    // Pseudo-random data should not compress well.
    // Use a deterministic "random" pattern.
    let mut data = vec![0u8; 8192];
    let mut state: u32 = 42;
    for byte in data.iter_mut() {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        *byte = ((state >> 16) & 0xFF) as u8;
    }
    let result = compress_if_beneficial(&data);
    // Random data should not compress below threshold.
    assert!(
        result.is_none(),
        "random data should not be beneficial to compress"
    );
}

#[test]
fn test_compress_if_beneficial_empty() {
    assert!(compress_if_beneficial(&[]).is_none());
}

#[test]
fn test_decompress_invalid() {
    assert!(decompress(&[0, 0, 0, 0], 1 << 20).is_err());
}

#[test]
fn test_decompress_corrupted() {
    let data = vec![42u8; 256];
    let mut compressed = compress(&data);
    // Corrupt the LZ4 data.
    if compressed.len() > 12 {
        compressed[12] ^= 0xFF;
    }
    assert!(decompress(&compressed, data.len()).is_err());
}

#[test]
fn test_likely_compressible_zeros() {
    let data = vec![0u8; 4096];
    assert!(likely_compressible(&data));
}

#[test]
fn test_likely_compressible_repeated_pattern() {
    let data: Vec<u8> = (0..4096).map(|i| (i % 4) as u8).collect();
    assert!(likely_compressible(&data));
}

#[test]
fn test_likely_compressible_random() {
    let mut data = vec![0u8; 4096];
    let mut state: u32 = 42;
    for byte in data.iter_mut() {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        *byte = ((state >> 16) & 0xFF) as u8;
    }
    assert!(!likely_compressible(&data));
}

#[test]
fn test_likely_compressible_short() {
    assert!(!likely_compressible(&[0; 32]));
}

#[test]
fn test_compression_ratio_sparse_tensor() {
    // Simulate a sparse attention mask (mostly zeros with a few ones).
    let mut data = vec![0u8; 16384];
    for i in (0..data.len()).step_by(128) {
        data[i] = 1;
    }
    let compressed = compress(&data);
    let ratio = compressed.len() as f64 / data.len() as f64;
    assert!(
        ratio < 0.1,
        "sparse tensor should compress to <10%: ratio={ratio:.3}"
    );

    // Verify roundtrip.
    let decompressed = decompress(&compressed, data.len()).unwrap();
    assert_eq!(decompressed, data);
}

/// Build a payload in the [`compress`] layout with chosen prefixes and body.
fn framed(outer_len: u64, inner_len: u32, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(12 + body.len());
    out.extend_from_slice(&outer_len.to_le_bytes());
    out.extend_from_slice(&inner_len.to_le_bytes());
    out.extend_from_slice(body);
    out
}

/// Run `decompress` and return its error message; an `Ok` fails the test
/// without printing the (possibly huge) output buffer.
fn decompress_err(data: &[u8], max_len: usize) -> String {
    match decompress(data, max_len) {
        Ok(out) => panic!(
            "decompress must reject this payload, got {} bytes",
            out.len()
        ),
        Err(err) => format!("{err:#}"),
    }
}

/// A 13-byte payload declaring 4 GiB must fail on the size check, before any
/// buffer of the declared size is allocated (lz4_flex's
/// `decompress_size_prepended` allocated it first and only then failed).
#[test]
fn decompress_rejects_oversized_length_prefix_before_allocating() {
    let payload = framed(0xFFFF_FFFF, 0xFFFF_FFFF, &[0x00]);
    let err = decompress_err(&payload, 1 << 20);
    assert!(err.contains("exceeds"), "unexpected error: {err}");
}

#[test]
fn decompress_rejects_mismatched_size_prefixes() {
    let data = vec![7u8; 1024];
    let mut compressed = compress(&data);
    compressed[0..8].copy_from_slice(&(data.len() as u64 + 1).to_le_bytes());
    let err = decompress_err(&compressed, 1 << 20);
    assert!(
        err.contains("LZ4 size prefix mismatch"),
        "unexpected error: {err}"
    );
}

/// LZ4 cannot expand a block by more than about 255:1, so a declared size
/// beyond 256 times the received body is rejected before allocating.
#[test]
fn decompress_rejects_declared_size_beyond_expansion_bound() {
    let declared: u32 = 256 * 4 + 1;
    let payload = framed(u64::from(declared), declared, &[0x00, 0x00, 0x00, 0x00]);
    let err = decompress_err(&payload, 1 << 20);
    assert!(err.contains("expansion bound"), "unexpected error: {err}");
}

/// A block that decodes to fewer bytes than its prefix declares is rejected
/// instead of being returned short.
#[test]
fn decompress_rejects_block_shorter_than_declared_size() {
    let data = vec![3u8; 4096];
    let mut compressed = compress(&data);
    let declared = data.len() + 1;
    compressed[0..8].copy_from_slice(&(declared as u64).to_le_bytes());
    compressed[8..12].copy_from_slice(&(declared as u32).to_le_bytes());
    let err = decompress_err(&compressed, 1 << 20);
    assert!(err.contains("LZ4"), "unexpected error: {err}");
}

#[test]
fn decompress_rejects_payload_shorter_than_both_prefixes() {
    let err = decompress_err(&[0u8; 11], 1 << 20);
    assert!(err.contains("too short"), "unexpected error: {err}");
}

/// The expansion bound must not reject maximally compressible data: 64 MiB
/// of zeros compresses at close to the LZ4 ceiling and still round-trips.
#[test]
fn decompress_round_trips_64_mib_of_zeros() {
    let len = 64 << 20;
    let data = vec![0u8; len];
    let compressed = compress(&data);
    let decompressed = decompress(&compressed, len).unwrap();
    assert!(decompressed.len() == len && decompressed.iter().all(|&b| b == 0));
    // One byte under the declared size is over the caller's limit.
    let err = decompress_err(&compressed, len - 1);
    assert!(err.contains("exceeds limit"), "unexpected error: {err}");
}
