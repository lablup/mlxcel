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

//! Automatic Prefix Caching (APC) — request-path helpers used by
//! [`super::store::PromptCacheStore`].
//!
//! The store delegates two pieces of APC-specific logic here so its main file
//! stays focused on locking + LRU bookkeeping:
//!
//! - [`apc_consistent_prefix_len`]: at lookup time, recompute the request's
//!   block-hash chain on the fly and clamp the matched-prefix length to the
//!   last block boundary where both the request and the candidate entry's
//!   stored chain agree. This is the core safety property of APC — a
//!   candidate is never adopted past a divergence.
//! - [`ApcStoreStats`]: a small struct carrying the three aggregate counters
//!   the `/v1/cache/stats` route surfaces (total blocks, unique blocks,
//!   active entries).
//!
//! Both helpers are no-ops when the store has APC disabled — they are only
//! invoked from the APC-on branch in `lookup_longest_prefix` and `apc_stats`.

use super::block_hash::{ApcBlockHash, ApcHashAlgo, BlockHashChain};

/// Compute the number of leading tokens of the request that agree with the
/// candidate entry's stored block-hash chain at block granularity.
///
/// The function recomputes the request's block hashes for the first
/// `floor(matched_len / block_size)` blocks using the same `block_size`,
/// `algo`, and `extra_hash` that produced the candidate's chain. Because the
/// chain is constructed as a Merkle-DAG (each block's hash feeds the next as
/// `parent`), a divergence at block `k` invalidates every block from `k`
/// onward — so the function returns `k * block_size`, the last block boundary
/// where both chains agreed.
///
/// If every covered block agrees, the input `matched_len` is preserved.
/// `matched_len` smaller than `block_size` short-circuits to `0` because no
/// full block can be verified without `block_size` tokens of agreement.
pub(super) fn apc_consistent_prefix_len(
    request_tokens: &[i32],
    candidate_hashes: &[ApcBlockHash],
    block_size: usize,
    algo: ApcHashAlgo,
    extra_hash: &[u8; 32],
    matched_len: usize,
) -> usize {
    if block_size == 0 || matched_len < block_size {
        // No full block of agreement to verify — be conservative and report
        // zero matched tokens. The caller will treat this as a miss.
        return 0;
    }
    let coverable_blocks = matched_len / block_size;
    if coverable_blocks == 0 {
        return 0;
    }
    let coverable_blocks = coverable_blocks.min(candidate_hashes.len());
    if coverable_blocks == 0 {
        return 0;
    }
    let cap_tokens = (coverable_blocks * block_size).min(request_tokens.len());
    let request_chain =
        BlockHashChain::compute(&request_tokens[..cap_tokens], block_size, algo, extra_hash);

    let upper = coverable_blocks.min(request_chain.hashes.len());
    let consistent_blocks = request_chain
        .hashes
        .iter()
        .zip(candidate_hashes.iter())
        .take(upper)
        .take_while(|(req, cand)| req == cand)
        .count();
    consistent_blocks * block_size
}

/// Aggregate APC-specific statistics gathered by walking every live entry.
///
/// Returned as a flat triple so callers (notably the `/v1/cache/stats`
/// handler) can consume it without reaching into the store's internal
/// types. When APC is disabled or no entries carry a chain, every field is
/// zero — operators can rely on a stable zero-payload shape.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ApcStoreStats {
    /// Total number of block hashes recorded across all entries that have an
    /// APC chain. Effectively `sum(entry.apc_block_hashes.len())` over
    /// APC-active entries.
    pub total_blocks_stored: usize,
    /// Number of distinct `ApcBlockHash` values across all entries — a
    /// measure of dedup potential. `total_blocks_stored - unique_block_hashes`
    /// approximates the count of cross-entry block reuse opportunities.
    pub unique_block_hashes: usize,
    /// Number of entries that carry a populated APC block-hash chain. When
    /// APC is disabled at the store level, this is `0`.
    pub apc_active_entries: usize,
}

#[cfg(test)]
#[path = "apc_lookup_tests.rs"]
mod tests;
