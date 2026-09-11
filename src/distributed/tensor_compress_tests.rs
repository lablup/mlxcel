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
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, data);
}

#[test]
fn test_compress_decompress_varied_data() {
    let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
    let compressed = compress(&data);
    let decompressed = decompress(&compressed).unwrap();
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
    let decompressed = decompress(&compressed).unwrap();
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
    assert!(decompress(&[0, 0, 0, 0]).is_err());
}

#[test]
fn test_decompress_corrupted() {
    let data = vec![42u8; 256];
    let mut compressed = compress(&data);
    // Corrupt the LZ4 data.
    if compressed.len() > 12 {
        compressed[12] ^= 0xFF;
    }
    assert!(decompress(&compressed).is_err());
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
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, data);
}
