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

//! LZ4 compression for tensor transfer bandwidth reduction.
//!
//! Compression is applied selectively: only when the compressed output is
//! meaningfully smaller than the input (below [`COMPRESSION_RATIO_THRESHOLD`]).
//! This avoids wasting CPU cycles on dense float data that compresses poorly.
//!
//! Sparse tensors (e.g., attention masks, one-hot activations) compress well,
//! often achieving 10-50x reduction.
//!
//! Used by: distributed inference tensor transfers (KV cache, activations,
//! weight shards).

use anyhow::{Context, Result};

use super::tensor_protocol::COMPRESSION_RATIO_THRESHOLD;

/// Compress data with LZ4 block compression.
///
/// Returns the compressed bytes prefixed with the original uncompressed
/// length as a u64 LE (needed for decompression buffer sizing).
pub fn compress(data: &[u8]) -> Vec<u8> {
    let compressed = lz4_flex::compress_prepend_size(data);
    let mut out = Vec::with_capacity(8 + compressed.len());
    out.extend_from_slice(&(data.len() as u64).to_le_bytes());
    out.extend_from_slice(&compressed);
    out
}

/// Decompress LZ4-compressed data.
///
/// Expects the format produced by [`compress`]: a u64 LE original length
/// prefix followed by LZ4 block data.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() < 8 {
        anyhow::bail!("compressed data too short for length prefix");
    }
    let _original_len = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
    let decompressed = lz4_flex::decompress_size_prepended(&data[8..])
        .map_err(|e| anyhow::anyhow!("LZ4 decompression failed: {e}"))
        .context("decompressing tensor data")?;
    Ok(decompressed)
}

/// Compress data only if the result is meaningfully smaller.
///
/// Returns `Some(compressed)` if the compression ratio is below the
/// threshold, or `None` if compression is not beneficial.
pub fn compress_if_beneficial(data: &[u8]) -> Option<Vec<u8>> {
    if data.is_empty() {
        return None;
    }

    let compressed = compress(data);

    // The compressed output includes our 8-byte length prefix.
    let ratio = compressed.len() as f64 / data.len() as f64;
    if ratio < COMPRESSION_RATIO_THRESHOLD {
        Some(compressed)
    } else {
        None
    }
}

/// Estimate whether data is likely to compress well based on a small sample.
///
/// This is a fast heuristic: it checks the entropy of the first 4 KiB.
/// Useful for deciding whether to attempt compression without actually
/// running the compressor on the full data.
pub fn likely_compressible(data: &[u8]) -> bool {
    if data.len() < 64 {
        return false;
    }

    // Sample the first 4 KiB.
    let sample_len = data.len().min(4096);
    let sample = &data[..sample_len];

    // Count unique byte values as a rough entropy proxy.
    let mut seen = [false; 256];
    for &b in sample {
        seen[b as usize] = true;
    }
    let unique_bytes = seen.iter().filter(|&&v| v).count();

    // If fewer than 60% of byte values appear in the sample, the data
    // likely has patterns that compress well.
    unique_bytes < 154
}

#[cfg(test)]
#[path = "tensor_compress_tests.rs"]
mod tests;
