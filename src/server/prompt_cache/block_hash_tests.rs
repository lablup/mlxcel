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

//! Unit tests for the APC block hash chain.
//!
//! Coverage:
//! - Stable digests for identical inputs (run-to-run determinism).
//! - Multimodal collision avoidance (different images break the chain).
//! - Block boundary behaviour (full blocks, partial tail blocks, empty input).
//! - Chain propagation (modifying token N invalidates all downstream blocks).
//! - Hash-algorithm parsing and round-trip.

use super::*;
use crate::server::prompt_cache::key::{MultimodalDigest, multimodal_digest};

fn tokens(n: usize) -> Vec<i32> {
    (0..n as i32).collect()
}

// ---------------------------------------------------------------------------
// Stability and determinism
// ---------------------------------------------------------------------------

#[test]
fn chain_is_stable_for_identical_inputs() {
    let toks = tokens(48);
    let mm = MultimodalDigest::empty();

    let a = BlockHashChain::compute_with_mm(&toks, 16, ApcHashAlgo::Sha256, &mm);
    let b = BlockHashChain::compute_with_mm(&toks, 16, ApcHashAlgo::Sha256, &mm);

    assert_eq!(a.hashes, b.hashes);
    assert_eq!(a.len(), 3);
    assert_eq!(a.full_blocks(), 3);
}

#[test]
fn chain_changes_when_tokens_differ() {
    let mm = MultimodalDigest::empty();
    let a = BlockHashChain::compute_with_mm(&tokens(32), 16, ApcHashAlgo::Sha256, &mm);
    let mut shifted = tokens(32);
    shifted[0] = 999;
    let b = BlockHashChain::compute_with_mm(&shifted, 16, ApcHashAlgo::Sha256, &mm);

    // First block diverges, and because the chain is built on parent=prev,
    // every downstream block must also diverge.
    for i in 0..a.len() {
        assert_ne!(
            a.hashes[i], b.hashes[i],
            "block {i} should differ when tokens differ at index 0"
        );
    }
}

#[test]
fn chain_propagates_late_token_change_only_downstream() {
    let mm = MultimodalDigest::empty();
    let a = BlockHashChain::compute_with_mm(&tokens(48), 16, ApcHashAlgo::Sha256, &mm);
    // Mutate a token in the second block (index 16-31 inclusive).
    let mut mutated = tokens(48);
    mutated[20] = 7777;
    let b = BlockHashChain::compute_with_mm(&mutated, 16, ApcHashAlgo::Sha256, &mm);

    // Block 0 must be identical (untouched tokens).
    assert_eq!(a.hashes[0], b.hashes[0]);
    // Blocks 1 and 2 must diverge (block 1 has the change, block 2 has the
    // mutated parent hash from block 1).
    assert_ne!(a.hashes[1], b.hashes[1]);
    assert_ne!(a.hashes[2], b.hashes[2]);
}

// ---------------------------------------------------------------------------
// Multimodal isolation (the core APC value-add)
// ---------------------------------------------------------------------------

#[test]
fn chain_diverges_at_first_block_when_mm_digest_differs() {
    // Two requests with identical tokens but different image content. The APC
    // chain must diverge at block 0, otherwise we would share KV-cache entries
    // for visually different prompts that happen to share placeholder tokens.
    let toks = tokens(32);
    let img_a: &[&[u8]] = &[b"image-bytes-A"];
    let img_b: &[&[u8]] = &[b"image-bytes-B"];

    let mm_a = multimodal_digest(img_a, &[]);
    let mm_b = multimodal_digest(img_b, &[]);
    assert_ne!(mm_a, mm_b);

    let chain_a = BlockHashChain::compute_with_mm(&toks, 16, ApcHashAlgo::Sha256, &mm_a);
    let chain_b = BlockHashChain::compute_with_mm(&toks, 16, ApcHashAlgo::Sha256, &mm_b);

    assert_eq!(chain_a.len(), 2);
    assert_eq!(chain_b.len(), 2);
    // Critical: divergence at block 0, the very first block.
    assert_ne!(
        chain_a.hashes[0], chain_b.hashes[0],
        "block 0 must diverge across different images even with identical tokens"
    );
    // And subsequent blocks too (parent propagation).
    assert_ne!(chain_a.hashes[1], chain_b.hashes[1]);
}

#[test]
fn chain_text_only_matches_empty_mm_digest() {
    let toks = tokens(16);
    let chain_via_helper =
        BlockHashChain::compute_with_mm(&toks, 16, ApcHashAlgo::Sha256, &MultimodalDigest::empty());
    let chain_via_raw = BlockHashChain::compute(
        &toks,
        16,
        ApcHashAlgo::Sha256,
        MultimodalDigest::empty().as_bytes(),
    );
    assert_eq!(chain_via_helper.hashes, chain_via_raw.hashes);
}

// ---------------------------------------------------------------------------
// Block boundary behaviour
// ---------------------------------------------------------------------------

#[test]
fn chain_is_empty_for_empty_token_sequence() {
    let chain =
        BlockHashChain::compute_with_mm(&[], 16, ApcHashAlgo::Sha256, &MultimodalDigest::empty());
    assert!(chain.is_empty());
    assert_eq!(chain.len(), 0);
    assert_eq!(chain.full_blocks(), 0);
    assert_eq!(chain.tail(), None);
}

#[test]
fn partial_tail_block_is_recorded_and_distinct() {
    // 40 tokens = 2 full blocks of 16 + 1 partial block of 8.
    let toks = tokens(40);
    let chain =
        BlockHashChain::compute_with_mm(&toks, 16, ApcHashAlgo::Sha256, &MultimodalDigest::empty());
    assert_eq!(chain.len(), 3);
    assert_eq!(chain.full_blocks(), 2);

    // The same first 32 tokens should produce identical first two block hashes
    // (parent propagation is deterministic and the hashed content is identical).
    let toks32 = tokens(32);
    let chain32 = BlockHashChain::compute_with_mm(
        &toks32,
        16,
        ApcHashAlgo::Sha256,
        &MultimodalDigest::empty(),
    );
    assert_eq!(chain32.len(), 2);
    assert_eq!(chain32.full_blocks(), 2);
    assert_eq!(chain32.hashes[0], chain.hashes[0]);
    assert_eq!(chain32.hashes[1], chain.hashes[1]);
}

#[test]
fn block_size_one_is_supported() {
    // Degenerate case: each token is its own block. Useful to verify the
    // hashing does not assume block_size > 1.
    let toks = tokens(4);
    let chain =
        BlockHashChain::compute_with_mm(&toks, 1, ApcHashAlgo::Sha256, &MultimodalDigest::empty());
    assert_eq!(chain.len(), 4);
    assert_eq!(chain.full_blocks(), 4);
    // All-distinct hashes (different parents and different tokens at each step).
    for i in 0..4 {
        for j in (i + 1)..4 {
            assert_ne!(chain.hashes[i], chain.hashes[j]);
        }
    }
}

#[test]
fn block_size_zero_falls_back_to_default_in_release() {
    // In debug builds this asserts and panics; here we just confirm the
    // release-build fallback path exists by invoking the method via the
    // public surface. We cannot easily exercise the release-build path under
    // `cfg(debug_assertions)` so the assertion is left as a documented
    // contract — see the function docs.
    if !cfg!(debug_assertions) {
        let chain = BlockHashChain::compute_with_mm(
            &tokens(32),
            0,
            ApcHashAlgo::Sha256,
            &MultimodalDigest::empty(),
        );
        // 32 tokens / DEFAULT_APC_BLOCK_SIZE (16) = 2 blocks
        assert_eq!(chain.block_size, DEFAULT_APC_BLOCK_SIZE);
        assert_eq!(chain.len(), 2);
    }
}

// ---------------------------------------------------------------------------
// Algorithm parsing and round-trip
// ---------------------------------------------------------------------------

#[test]
fn apc_hash_algo_default_is_sha256() {
    assert_eq!(ApcHashAlgo::default(), ApcHashAlgo::Sha256);
    assert_eq!(ApcHashAlgo::Sha256.as_str(), "sha256");
    assert_eq!(ApcHashAlgo::Blake3.as_str(), "blake3");
}

#[test]
fn apc_hash_algo_parses_common_spellings() {
    assert_eq!(
        "sha256".parse::<ApcHashAlgo>().unwrap(),
        ApcHashAlgo::Sha256
    );
    assert_eq!(
        "SHA256".parse::<ApcHashAlgo>().unwrap(),
        ApcHashAlgo::Sha256
    );
    assert_eq!(
        "sha-256".parse::<ApcHashAlgo>().unwrap(),
        ApcHashAlgo::Sha256
    );
    assert_eq!(
        "blake3".parse::<ApcHashAlgo>().unwrap(),
        ApcHashAlgo::Blake3
    );
    assert_eq!(
        "Blake3".parse::<ApcHashAlgo>().unwrap(),
        ApcHashAlgo::Blake3
    );
    assert_eq!(
        "blake-3".parse::<ApcHashAlgo>().unwrap(),
        ApcHashAlgo::Blake3
    );
}

#[test]
fn apc_hash_algo_rejects_unknown_strings() {
    assert!("md5".parse::<ApcHashAlgo>().is_err());
    assert!("".parse::<ApcHashAlgo>().is_err());
    assert!("xxhash".parse::<ApcHashAlgo>().is_err());
}

#[test]
fn sha256_and_blake3_produce_different_chains_for_same_input() {
    // Sanity: switching the algorithm should change the hash bytes for the
    // same logical input. This guards against an accidental implementation
    // where both branches pick the same hasher.
    let toks = tokens(32);
    let mm = MultimodalDigest::empty();
    let sha = BlockHashChain::compute_with_mm(&toks, 16, ApcHashAlgo::Sha256, &mm);
    let bl3 = BlockHashChain::compute_with_mm(&toks, 16, ApcHashAlgo::Blake3, &mm);
    assert_eq!(sha.len(), bl3.len());
    for i in 0..sha.len() {
        assert_ne!(
            sha.hashes[i], bl3.hashes[i],
            "block {i} must differ between sha256 and blake3"
        );
    }
}

// ---------------------------------------------------------------------------
// Display / hex helpers
// ---------------------------------------------------------------------------

#[test]
fn block_hash_zero_short_hex_is_all_zeros() {
    let zero = ApcBlockHash::ZERO;
    assert_eq!(zero.short_hex(), "0000000000000000");
    assert_eq!(
        zero.to_hex(),
        "0000000000000000000000000000000000000000000000000000000000000000"
    );
}

#[test]
fn block_hash_to_hex_round_trips_via_as_bytes() {
    let chain = BlockHashChain::compute_with_mm(
        &tokens(16),
        16,
        ApcHashAlgo::Sha256,
        &MultimodalDigest::empty(),
    );
    let h = chain.hashes[0];
    let hex = h.to_hex();
    assert_eq!(hex.len(), 64);
    // Verify hex characters are lowercase hex.
    assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
    // Same display impl.
    assert_eq!(format!("{h}"), hex);
}
