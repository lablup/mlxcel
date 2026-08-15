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

//! Unit tests for the APC lookup-time helpers.

use super::apc_consistent_prefix_len;
use crate::server::prompt_cache::block_hash::{ApcHashAlgo, BlockHashChain};
use crate::server::prompt_cache::key::MultimodalDigest;

const BLOCK: usize = 16;

fn empty_extra() -> [u8; 32] {
    *MultimodalDigest::empty().as_bytes()
}

#[test]
fn aligned_matching_chains_return_the_block_floored_len() {
    // Both sides see the same tokens and same extra_hash → every block hash
    // agrees. matched_len (64) is already block-aligned to BLOCK (16), so
    // the block-floored result happens to equal matched_len here, so this
    // test alone cannot distinguish "matched_len preserved" from "floored to
    // block granularity". See `non_aligned_matched_len_floors_to_covered_blocks`
    // below for a case that tells the two apart.
    let tokens: Vec<i32> = (0..64).collect();
    let extra = empty_extra();
    let candidate_chain = BlockHashChain::compute(&tokens, BLOCK, ApcHashAlgo::Sha256, &extra);
    let consistent = apc_consistent_prefix_len(
        &tokens,
        &candidate_chain.hashes,
        BLOCK,
        ApcHashAlgo::Sha256,
        &extra,
        tokens.len(),
    );
    assert_eq!(consistent, tokens.len());
}

#[test]
fn non_aligned_matched_len_floors_to_covered_blocks() {
    // 73 tokens, BLOCK = 16, matched_len = 73, fully agreeing chains. The
    // candidate chain has ceil(73/16) = 5 hashes (the 5th is a 9-token
    // partial trailing block), but coverable_blocks = 73/16 = 4 caps the
    // comparison at 4 blocks, so that 5th, partial block is never compared.
    // The result is 4 * 16 = 64, not 73: even with full agreement across
    // every block that gets checked, the non-block-aligned tail of
    // matched_len is dropped.
    let tokens: Vec<i32> = (0..73).collect();
    let extra = empty_extra();
    let candidate_chain = BlockHashChain::compute(&tokens, BLOCK, ApcHashAlgo::Sha256, &extra);
    assert_eq!(candidate_chain.hashes.len(), 5);
    let consistent = apc_consistent_prefix_len(
        &tokens,
        &candidate_chain.hashes,
        BLOCK,
        ApcHashAlgo::Sha256,
        &extra,
        tokens.len(),
    );
    assert_eq!(consistent, 64);
    assert_ne!(consistent, tokens.len());
}

#[test]
fn divergent_extra_hash_truncates_at_first_block() {
    // Same tokens, different extra_hash (e.g. different image). The chain
    // diverges at block 0, so the helper must report 0 matched tokens.
    let tokens: Vec<i32> = (0..64).collect();
    let candidate_extra = [0xAAu8; 32];
    let request_extra = [0xBBu8; 32];
    let candidate_chain =
        BlockHashChain::compute(&tokens, BLOCK, ApcHashAlgo::Sha256, &candidate_extra);
    let consistent = apc_consistent_prefix_len(
        &tokens,
        &candidate_chain.hashes,
        BLOCK,
        ApcHashAlgo::Sha256,
        &request_extra,
        tokens.len(),
    );
    assert_eq!(consistent, 0);
}

#[test]
fn divergence_at_block_two_truncates_to_block_boundary() {
    // Block 0 and 1 agree, block 2 disagrees because we mutate one token
    // at index 32 (start of block 2). The helper must clamp to 32 tokens.
    let mut request: Vec<i32> = (0..64).collect();
    let candidate = request.clone();
    request[32] = 9999;
    let extra = empty_extra();
    let candidate_chain = BlockHashChain::compute(&candidate, BLOCK, ApcHashAlgo::Sha256, &extra);
    let consistent = apc_consistent_prefix_len(
        &request,
        &candidate_chain.hashes,
        BLOCK,
        ApcHashAlgo::Sha256,
        &extra,
        request.len(),
    );
    assert_eq!(consistent, 32);
}

#[test]
fn matched_len_below_block_size_returns_zero() {
    // The trie says 8 tokens match but block_size is 16 → no full block
    // of agreement is verifiable, so APC reports 0.
    let tokens: Vec<i32> = (0..64).collect();
    let extra = empty_extra();
    let candidate_chain = BlockHashChain::compute(&tokens, BLOCK, ApcHashAlgo::Sha256, &extra);
    let consistent = apc_consistent_prefix_len(
        &tokens,
        &candidate_chain.hashes,
        BLOCK,
        ApcHashAlgo::Sha256,
        &extra,
        8,
    );
    assert_eq!(consistent, 0);
}

#[test]
fn matched_len_floors_to_block_boundary() {
    // The trie says 33 tokens match but only block 0 (0..16) and block 1
    // (16..32) are full blocks. matched_len floors to 32.
    let tokens: Vec<i32> = (0..64).collect();
    let extra = empty_extra();
    let candidate_chain = BlockHashChain::compute(&tokens, BLOCK, ApcHashAlgo::Sha256, &extra);
    let consistent = apc_consistent_prefix_len(
        &tokens,
        &candidate_chain.hashes,
        BLOCK,
        ApcHashAlgo::Sha256,
        &extra,
        33,
    );
    assert_eq!(consistent, 32);
}

#[test]
fn empty_candidate_chain_returns_zero() {
    let tokens: Vec<i32> = (0..64).collect();
    let extra = empty_extra();
    let consistent = apc_consistent_prefix_len(
        &tokens,
        &[],
        BLOCK,
        ApcHashAlgo::Sha256,
        &extra,
        tokens.len(),
    );
    assert_eq!(consistent, 0);
}

#[test]
fn block_size_zero_returns_zero_safely() {
    // Defensive: if a misconfiguration ever produced block_size=0, the
    // helper must not panic and must report zero matched tokens.
    let tokens: Vec<i32> = (0..16).collect();
    let extra = empty_extra();
    let consistent = apc_consistent_prefix_len(&tokens, &[], 0, ApcHashAlgo::Sha256, &extra, 16);
    assert_eq!(consistent, 0);
}

#[test]
fn empty_request_tokens_returns_zero() {
    // An empty request token list cannot agree with any candidate block, so
    // the helper must return 0 without panicking. This covers the case where
    // a caller passes an empty slice — e.g. a zero-token request that somehow
    // reached the APC path after a trie match of length 0.
    let candidate_tokens: Vec<i32> = (0..32).collect();
    let extra = empty_extra();
    let candidate_chain =
        BlockHashChain::compute(&candidate_tokens, BLOCK, ApcHashAlgo::Sha256, &extra);
    let consistent = apc_consistent_prefix_len(
        &[],
        &candidate_chain.hashes,
        BLOCK,
        ApcHashAlgo::Sha256,
        &extra,
        0,
    );
    assert_eq!(consistent, 0);
}

#[test]
fn matched_len_zero_with_nonempty_candidate_returns_zero() {
    // matched_len=0 is a degenerate input the caller might produce when the
    // trie returned a zero-length prefix hit. The helper must clamp to 0
    // without attempting to verify any block.
    let tokens: Vec<i32> = (0..64).collect();
    let extra = empty_extra();
    let candidate_chain = BlockHashChain::compute(&tokens, BLOCK, ApcHashAlgo::Sha256, &extra);
    let consistent = apc_consistent_prefix_len(
        &tokens,
        &candidate_chain.hashes,
        BLOCK,
        ApcHashAlgo::Sha256,
        &extra,
        0,
    );
    assert_eq!(consistent, 0);
}
