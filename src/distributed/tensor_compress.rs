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

use anyhow::{Context, Result, anyhow, bail};

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

/// Upper bound on how far one LZ4 block can expand.
///
/// The LZ4 block format spends at least one byte per 255 bytes of match
/// length, so no block decodes to more than about 255 times its size. The
/// bound ties the output allocation to the bytes actually received without
/// rejecting any valid block.
const LZ4_MAX_EXPANSION: usize = 256;

/// Length of the two size prefixes ahead of the LZ4 block: the u64 written by
/// [`compress`] and the u32 written by `lz4_flex::compress_prepend_size`.
const PREFIX_LEN: usize = 12;

/// Decompress LZ4-compressed data.
///
/// Expects the format produced by [`compress`]:
/// `[u64 LE original_len][u32 LE original_len][LZ4 block]`.
///
/// `max_len` is the largest decompressed size the caller accepts, normally
/// derived from the tensor header the payload arrived with. The size prefix
/// comes from a peer, so before any output is allocated the two prefixes must
/// agree, the declared size must not exceed `max_len`, and it must not exceed
/// [`LZ4_MAX_EXPANSION`] times the received block. The output buffer is then
/// exactly the declared size, and a block that decodes to any other length is
/// rejected. (Issue #2691 added `max_len`; the wire format is unchanged.)
pub fn decompress(data: &[u8], max_len: usize) -> Result<Vec<u8>> {
    let Some((outer, rest)) = data.split_first_chunk::<8>() else {
        bail!(
            "compressed data too short for length prefixes: {} bytes, need at least {PREFIX_LEN}",
            data.len()
        );
    };
    let Some((inner, body)) = rest.split_first_chunk::<4>() else {
        bail!(
            "compressed data too short for length prefixes: {} bytes, need at least {PREFIX_LEN}",
            data.len()
        );
    };
    let outer = u64::from_le_bytes(*outer);
    let inner = u32::from_le_bytes(*inner);
    if outer != u64::from(inner) {
        bail!("LZ4 size prefix mismatch: outer prefix {outer}, inner prefix {inner}");
    }
    let declared = usize::try_from(outer)
        .map_err(|_| anyhow!("decompressed size {outer} exceeds addressable range"))?;
    if declared > max_len {
        bail!("decompressed size {declared} exceeds limit {max_len}");
    }
    if declared > body.len().saturating_mul(LZ4_MAX_EXPANSION) {
        bail!(
            "decompressed size {declared} exceeds the LZ4 expansion bound for a {}-byte block",
            body.len()
        );
    }

    let mut out = vec![0u8; declared];
    let written = lz4_flex::block::decompress_into(body, &mut out)
        .map_err(|e| anyhow!("LZ4 decompression failed: {e}"))
        .context("decompressing tensor data")?;
    if written != declared {
        bail!("LZ4 block decoded to {written} bytes, but its size prefix declares {declared}");
    }
    Ok(out)
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
