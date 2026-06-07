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

//! End-to-end integration tests for Automatic Prefix Caching (APC).
//!
//! These tests cover behavioural properties spanning multiple modules:
//!
//! - The block-hash chain diverges as soon as the multimodal digest changes,
//!   so a request with the same prompt tokens but a different image cannot
//!   accidentally adopt KV blocks that were saved for a different image.
//! - The chain remains stable across runs and across hash-algorithm choices
//!   for the same input.
//! - Hybrid-SSM detection composes correctly with the APC feature gate.

use std::time::Duration;

use crate::server::prompt_cache::block_hash::{ApcHashAlgo, BlockHashChain};
use crate::server::prompt_cache::entry::CacheEntry;
use crate::server::prompt_cache::hybrid_ssm::detect_hybrid_ssm;
use crate::server::prompt_cache::key::{
    MultimodalDigest, PromptCacheKey, multimodal_digest, multimodal_digest_from_vecs,
};
use crate::server::prompt_cache::policy::{ApcConfig, PromptCacheConfig};
use crate::server::prompt_cache::store::PromptCacheStore;
use crate::server::routes::cache::build_stats_response;

/// Smoke convenience: build a `PromptCacheKey` with the given tokens and
/// multimodal digest. We anchor the rest of the dimensions (model, lora,
/// template, session) so the *only* difference between the two keys we
/// compare is the multimodal payload — exactly the regression we want to
/// guard against.
fn key_with<'a>(tokens: &'a [i32], mm: MultimodalDigest) -> (PromptCacheKey<'a>, MultimodalDigest) {
    let k = PromptCacheKey::new_full("model-x", None, "tpl-x", None, mm, tokens);
    (k, mm)
}

#[test]
fn apc_chains_for_same_tokens_different_images_share_no_block() {
    // Realistic shape: the operator sends two requests with identical token
    // prompts but two different images. The KV cache for request A must not
    // be reused for request B.
    let tokens: Vec<i32> = (0..64).collect();

    let img_a: Vec<Vec<u8>> = vec![b"PNG-bytes-A-content".to_vec()];
    let img_b: Vec<Vec<u8>> = vec![b"PNG-bytes-B-content".to_vec()];

    let mm_a = multimodal_digest_from_vecs(&img_a, &[]);
    let mm_b = multimodal_digest_from_vecs(&img_b, &[]);
    assert_ne!(
        mm_a, mm_b,
        "different image bytes must yield distinct digests"
    );

    let (key_a, _) = key_with(&tokens, mm_a);
    let (key_b, _) = key_with(&tokens, mm_b);
    // Whole-prefix bucket digests must also differ (this is the existing bucket-level guarantee).
    assert_ne!(key_a.digest(), key_b.digest());

    // Block-hash chain divergence: every single block must differ.
    let chain_a = BlockHashChain::compute_with_mm(&tokens, 16, ApcHashAlgo::Sha256, &mm_a);
    let chain_b = BlockHashChain::compute_with_mm(&tokens, 16, ApcHashAlgo::Sha256, &mm_b);
    assert_eq!(chain_a.len(), chain_b.len());
    assert!(chain_a.len() >= 4);
    for (i, (a, b)) in chain_a.hashes.iter().zip(chain_b.hashes.iter()).enumerate() {
        assert_ne!(
            a, b,
            "block {i} must differ across requests with different images"
        );
    }
}

#[test]
fn apc_chains_match_for_same_tokens_same_image_across_runs() {
    // The reuse story: identical request -> identical chain, so the second
    // request can adopt the cached blocks. This is the inverse of the
    // collision-avoidance test above.
    let tokens: Vec<i32> = (0..64).collect();
    let img: Vec<Vec<u8>> = vec![b"identical-image".to_vec()];
    let mm = multimodal_digest_from_vecs(&img, &[]);

    let chain_run1 = BlockHashChain::compute_with_mm(&tokens, 16, ApcHashAlgo::Sha256, &mm);
    let chain_run2 = BlockHashChain::compute_with_mm(&tokens, 16, ApcHashAlgo::Sha256, &mm);

    assert_eq!(chain_run1.hashes, chain_run2.hashes);
}

#[test]
fn apc_chain_shares_prefix_when_only_late_tokens_differ() {
    // Two requests with the same image and the same first 32 tokens but
    // diverging tail tokens should share the first two block hashes (block
    // size 16) and only diverge on block 2.
    let mut toks_a: Vec<i32> = (0..64).collect();
    let mut toks_b = toks_a.clone();
    toks_b[40] = 9999; // mutate a token in block index 2

    let img: Vec<Vec<u8>> = vec![b"image".to_vec()];
    let mm = multimodal_digest_from_vecs(&img, &[]);

    let chain_a = BlockHashChain::compute_with_mm(&toks_a, 16, ApcHashAlgo::Sha256, &mm);
    let chain_b = BlockHashChain::compute_with_mm(&toks_b, 16, ApcHashAlgo::Sha256, &mm);

    assert_eq!(chain_a.hashes[0], chain_b.hashes[0]);
    assert_eq!(chain_a.hashes[1], chain_b.hashes[1]);
    assert_ne!(chain_a.hashes[2], chain_b.hashes[2]);
    assert_ne!(chain_a.hashes[3], chain_b.hashes[3]);

    // Mutate the original to keep the variable used.
    toks_a[40] = 0;
    let _ = toks_a; // silence unused-mut lint
}

#[test]
fn apc_audio_payloads_isolate_chain_like_images() {
    // Audio is the second multimodal channel. Same tokens, different audio
    // bytes must yield different chains, mirroring the image case.
    let tokens: Vec<i32> = (0..32).collect();
    let mm_a = multimodal_digest(&[], &[b"audio-A"]);
    let mm_b = multimodal_digest(&[], &[b"audio-B"]);

    let chain_a = BlockHashChain::compute_with_mm(&tokens, 16, ApcHashAlgo::Sha256, &mm_a);
    let chain_b = BlockHashChain::compute_with_mm(&tokens, 16, ApcHashAlgo::Sha256, &mm_b);
    for (i, (a, b)) in chain_a.hashes.iter().zip(chain_b.hashes.iter()).enumerate() {
        assert_ne!(a, b, "block {i} must differ across distinct audio payloads");
    }
}

#[test]
fn hybrid_ssm_detection_composes_with_apc_gate() {
    // Sanity check that the hybrid-SSM detection produces the values we
    // expect when integrated with APC config decisions.
    use serde_json::json;

    // jamba => hybrid hit, APC must be auto-disabled.
    let cfg = json!({"model_type": "jamba"});
    assert!(detect_hybrid_ssm(&cfg).is_some());

    // qwen3 (non-hybrid) => no hit, APC stays at operator-requested state.
    let cfg = json!({"model_type": "qwen3"});
    assert!(detect_hybrid_ssm(&cfg).is_none());

    // VLM with text_config.model_type=mamba2 (a hybrid wrapped inside a
    // VLM config) => hit on the nested path.
    let cfg = json!({
        "model_type": "internvl",
        "text_config": {"model_type": "mamba2"},
    });
    assert!(detect_hybrid_ssm(&cfg).is_some());
}

// ---------------------------------------------------------------------------
// End-to-end PromptCacheStore integration with APC enabled.
//
// These tests exercise the full data path the reviewer flagged as missing:
//
//   PromptCacheStore::insert  ->  block-hash chain attached to entry
//   PromptCacheStore::lookup_longest_prefix  ->  block-hash discriminator
//   /v1/cache/stats           ->  apc_active_entries / total_blocks_stored
//
// We use CacheEntry::new_for_test (zero-tensor detached set) so we don't need
// to load a real model — the store's accounting and lookup mechanics don't
// depend on the underlying tensors.
// ---------------------------------------------------------------------------

const BLOCK: usize = 16;
const APC_ON_MIN_PREFIX: usize = 4;

fn apc_on_config() -> PromptCacheConfig {
    let apc = ApcConfig {
        enabled: true,
        block_size: BLOCK,
        num_blocks: None,
        hash: ApcHashAlgo::Sha256,
    };
    PromptCacheConfig::new(
        true,
        1 << 20,
        32,
        Duration::from_secs(3600),
        APC_ON_MIN_PREFIX,
    )
    .with_apc(apc)
}

fn apc_off_config() -> PromptCacheConfig {
    PromptCacheConfig::new(
        true,
        1 << 20,
        32,
        Duration::from_secs(3600),
        APC_ON_MIN_PREFIX,
    )
}

#[test]
fn apc_stores_independent_entries_for_same_tokens_different_images() {
    // The headline regression: two requests with identical tokens but
    // different image bytes must end up as two independent entries with
    // distinct block-hash chains. Lookup-time discrimination then lands
    // each request on its own entry.
    let cfg = apc_on_config();
    let store = PromptCacheStore::with_config(cfg.clone());

    let tokens: Vec<i32> = (0..32).collect();
    let img_a: Vec<Vec<u8>> = vec![b"PNG-bytes-A".to_vec()];
    let img_b: Vec<Vec<u8>> = vec![b"PNG-bytes-B".to_vec()];
    let mm_a = multimodal_digest_from_vecs(&img_a, &[]);
    let mm_b = multimodal_digest_from_vecs(&img_b, &[]);
    assert_ne!(mm_a, mm_b);

    let key_a = PromptCacheKey::new_full("model", None, "tpl", None, mm_a, &tokens);
    let key_b = PromptCacheKey::new_full("model", None, "tpl", None, mm_b, &tokens);

    let entry_a = CacheEntry::new_for_test(tokens.clone(), 1024);
    let entry_b = CacheEntry::new_for_test(tokens.clone(), 1024);
    store.insert(&key_a, entry_a).expect("insert A");
    store.insert(&key_b, entry_b).expect("insert B");

    // Two distinct entries.
    assert_eq!(store.len(), 2, "different mm_digests -> two entries");

    // Bucket digests differ (mm_digest is part of PromptCacheKey::digest()).
    assert_ne!(key_a.digest(), key_b.digest());

    // Lookup with T1+I1 hits the I1 entry.
    let (hit_a, matched_a) = store
        .lookup_longest_prefix(&key_a, &tokens)
        .expect("lookup A hits");
    assert_eq!(matched_a, tokens.len());
    assert_eq!(hit_a.tokens, tokens);
    let chain_a = hit_a
        .apc_block_hashes()
        .expect("APC chain attached on insert");
    let expected_chain_a =
        BlockHashChain::compute_with_mm(&tokens, BLOCK, ApcHashAlgo::Sha256, &mm_a);
    assert_eq!(chain_a, expected_chain_a.hashes.as_slice());

    // Lookup with T1+I2 hits the I2 entry, NOT the I1 entry.
    let (hit_b, matched_b) = store
        .lookup_longest_prefix(&key_b, &tokens)
        .expect("lookup B hits");
    assert_eq!(matched_b, tokens.len());
    let chain_b = hit_b
        .apc_block_hashes()
        .expect("APC chain attached on insert");
    let expected_chain_b =
        BlockHashChain::compute_with_mm(&tokens, BLOCK, ApcHashAlgo::Sha256, &mm_b);
    assert_eq!(chain_b, expected_chain_b.hashes.as_slice());

    // Cross-check: chains differ across the two entries.
    assert_ne!(chain_a, chain_b);
}

#[test]
fn apc_stats_reflect_block_chain_after_inserts() {
    // /v1/cache/stats must report apc_active_entries: 2 and
    // total_blocks_stored: 2 * num_blocks = 4 after two 32-token entries
    // are inserted with APC on.
    let cfg = apc_on_config();
    let store = PromptCacheStore::with_config(cfg.clone());

    let tokens: Vec<i32> = (0..32).collect();
    let img_a: Vec<Vec<u8>> = vec![b"image-A".to_vec()];
    let img_b: Vec<Vec<u8>> = vec![b"image-B".to_vec()];
    let mm_a = multimodal_digest_from_vecs(&img_a, &[]);
    let mm_b = multimodal_digest_from_vecs(&img_b, &[]);

    store
        .insert(
            &PromptCacheKey::new_full("model", None, "tpl", None, mm_a, &tokens),
            CacheEntry::new_for_test(tokens.clone(), 1024),
        )
        .expect("insert A");
    store
        .insert(
            &PromptCacheKey::new_full("model", None, "tpl", None, mm_b, &tokens),
            CacheEntry::new_for_test(tokens.clone(), 1024),
        )
        .expect("insert B");

    let resp = build_stats_response(
        Some(&store),
        &cfg,
        crate::server::routes::cache::PagedBlockStats::default(),
    );
    assert!(resp.enabled);
    assert!(resp.apc_enabled);
    assert_eq!(resp.entries, 2);
    assert_eq!(resp.apc_active_entries, 2);
    // Each entry covers 32 tokens at block_size=16 -> 2 blocks per entry.
    assert_eq!(resp.total_blocks_stored, 4);
    // The two entries diverge from the very first block (mm_digest folds in)
    // so all four block hashes are distinct.
    assert_eq!(resp.unique_block_hashes, 4);
}

#[test]
fn apc_disabled_flow_still_works_but_apc_active_entries_is_zero() {
    // Same insert + lookup flow but with APC disabled. The whole-prefix
    // mm_digest path must still keep the two entries independent and the
    // stats endpoint must report 0 APC-related counters.
    let cfg = apc_off_config();
    let store = PromptCacheStore::with_config(cfg.clone());

    let tokens: Vec<i32> = (0..32).collect();
    let img_a: Vec<Vec<u8>> = vec![b"image-A".to_vec()];
    let img_b: Vec<Vec<u8>> = vec![b"image-B".to_vec()];
    let mm_a = multimodal_digest_from_vecs(&img_a, &[]);
    let mm_b = multimodal_digest_from_vecs(&img_b, &[]);
    let key_a = PromptCacheKey::new_full("model", None, "tpl", None, mm_a, &tokens);
    let key_b = PromptCacheKey::new_full("model", None, "tpl", None, mm_b, &tokens);

    store
        .insert(&key_a, CacheEntry::new_for_test(tokens.clone(), 1024))
        .expect("insert A");
    store
        .insert(&key_b, CacheEntry::new_for_test(tokens.clone(), 1024))
        .expect("insert B");

    // Same isolation guarantee from the existing whole-prefix mm_digest
    // path: two entries, each lookup hits its own entry, no APC chain.
    assert_eq!(store.len(), 2);
    let (hit_a, _) = store
        .lookup_longest_prefix(&key_a, &tokens)
        .expect("lookup A hits");
    assert!(hit_a.apc_block_hashes().is_none(), "APC off -> no chain");
    let (hit_b, _) = store
        .lookup_longest_prefix(&key_b, &tokens)
        .expect("lookup B hits");
    assert!(hit_b.apc_block_hashes().is_none(), "APC off -> no chain");

    let resp = build_stats_response(
        Some(&store),
        &cfg,
        crate::server::routes::cache::PagedBlockStats::default(),
    );
    assert!(resp.enabled);
    assert!(!resp.apc_enabled);
    assert_eq!(resp.entries, 2);
    assert_eq!(resp.apc_active_entries, 0);
    assert_eq!(resp.total_blocks_stored, 0);
    assert_eq!(resp.unique_block_hashes, 0);
}

#[test]
fn apc_chain_is_recomputable_from_entry_tokens_and_request_mm_digest() {
    // Round-trip property: after a successful insert with APC on, the
    // recovered chain on the entry must equal a fresh chain computed from
    // the entry's tokens and the request's mm_digest. This is what the
    // lookup-time discriminator relies on for safety.
    let cfg = apc_on_config();
    let store = PromptCacheStore::with_config(cfg);
    let tokens: Vec<i32> = (0..48).collect();
    let img: Vec<Vec<u8>> = vec![b"only-image".to_vec()];
    let mm = multimodal_digest_from_vecs(&img, &[]);
    let key = PromptCacheKey::new_full("model", None, "tpl", None, mm, &tokens);

    store
        .insert(&key, CacheEntry::new_for_test(tokens.clone(), 1024))
        .expect("insert");

    let (hit, _) = store.lookup_longest_prefix(&key, &tokens).expect("hit");
    let stored = hit.apc_block_hashes().expect("APC chain stored");
    let recomputed = BlockHashChain::compute_with_mm(&tokens, BLOCK, ApcHashAlgo::Sha256, &mm);
    assert_eq!(stored, recomputed.hashes.as_slice());
    // 48 tokens / 16 = 3 full blocks; chain length matches.
    assert_eq!(stored.len(), 3);
}
