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

//! Real-model parity for the unified radix prompt-prefix cache over the paged
//! block pool (#121).
//!
//! Proves the headline behaviour: two requests sharing a prompt prefix store
//! that prefix's KV **once** in the paged pool and the second request reuses it
//! without re-prefilling — and still decodes bit-identically to a cold
//! (no-cache) paged run. The case runs for each pool-backed family with a small
//! checkpoint — **qwen3** and **llama3** (Fp16 dense-natural backend). The
//! adopt/donate machinery operates on the pool + block table, not model
//! internals, so it is otherwise model-agnostic.
//!
//! ## What it drives
//!
//! The radix-store + `CachePool` path the scheduler uses (`BatchScheduler::new`
//! is `pub(crate)`, so an integration test drives its cache core directly, same
//! as `paged_scheduler_parity`):
//!
//! 1. **Cold reference** — a pool-backed paged sequence prefills the FULL prompt
//!    `[prefix + suffix]` and greedily decodes `DECODE_STEPS` tokens.
//! 2. **Cached path** — sequence A prefills ONLY the shared `prefix`, then
//!    `detach_paged` → a [`DetachedKvSet::Paged`] [`CacheEntry`] in a real
//!    [`PromptCacheStore`]. A second request B looks the prefix up, `take_detached`
//!    → `adopt_paged` (sharing A's refcount-pinned prefix blocks), prefills ONLY
//!    its divergent `suffix` at the right offset, and greedily decodes.
//!
//! The cached decode must equal the cold decode, and B's block table must reuse
//! A's prefix blocks (no second copy of the prefix).
//!
//! ## Running
//!
//! `#[ignore]` (loads a real checkpoint and runs real GPU forwards). Run with:
//!
//! ```text
//! cargo test --test paged_prefix_share_parity --release \
//!     --features metal,accelerate -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Each case soft-skips when its model directory is absent. Fetch with:
//!
//! ```text
//! ./target/release/mlxcel download mlx-community/Qwen3-0.6B-4bit
//! ./target/release/mlxcel download mlx-community/Llama-3.2-1B-Instruct-4bit
//! ```

mod common;
use common::repo_model_dir;

use std::time::Duration;

use mlxcel::server::prompt_cache::key::{MultimodalDigest, PromptCacheKey};
use mlxcel::server::prompt_cache::{
    CacheEntry, DetachedKvSet, PromptCacheConfig, PromptCacheStore,
};
use mlxcel::{LanguageModel, initialize_runtime, load_model};
use mlxcel_core::cache::{CachePool, PagedKvLayout, SequenceStateLayout};

/// qwen3 checkpoint directory name (pool-backed family).
const QWEN3_DIR: &str = "qwen3-0.6b-4bit";
/// llama3 checkpoint directory name (pool-backed family).
const LLAMA3_DIR: &str = "llama-3.2-1b-4bit";

/// Paged block size (tokens per physical block) — matches the scheduler's
/// `DEFAULT_PAGED_BLOCK_SIZE`.
const BLOCK_SIZE: usize = 32;

/// Number of greedy decode steps to compare after prefill.
const DECODE_STEPS: usize = 16;

/// Shared prompt prefix — 40 tokens (> one 32-token block) so the prefix spans
/// two physical blocks and the "stored once / no re-prefill" claim is concrete.
/// Deterministic plausible ids; no tokenizer needed.
const PREFIX_TOKENS: &[i32] = &[
    9707, 11, 358, 1079, 264, 4128, 1614, 13, 5209, 3291, 752, 911, 697, 7990, 13, 358, 2776, 264,
    10950, 17847, 13, 6771, 594, 1438, 419, 1495, 3019, 553, 3019, 11, 323, 1473, 697, 975, 13,
    5209, 387, 2797, 624, 14374,
];

/// B's divergent suffix appended after the shared prefix.
const SUFFIX_TOKENS: &[i32] = &[14582, 25, 3555, 374, 220, 17, 488, 220, 17, 30];

/// Greedy argmax over the vocab at sequence position `pos` of a
/// `[batch, seq_len, vocab]` logits tensor, for batch row `row`.
fn greedy_token(logits: &mlxcel_core::MlxArray, row: i32, pos: i32) -> i32 {
    let shape = mlxcel_core::array_shape(logits);
    let vocab = shape[2];
    let at_pos = mlxcel_core::slice(logits, &[row, pos, 0], &[row + 1, pos + 1, vocab]);
    let flat = mlxcel_core::reshape(&at_pos, &[vocab]);
    mlxcel_core::eval(&flat);
    mlxcel_core::item_i32(&mlxcel_core::argmax_last_axis(&flat))
}

/// The paged layout the scheduler builds for the default Fp16 KV mode.
fn scheduler_paged_layout(num_layers: usize) -> SequenceStateLayout {
    SequenceStateLayout::paged_kv_cache(
        PagedKvLayout::uniform(num_layers, BLOCK_SIZE, BLOCK_SIZE).expect("valid paged layout"),
    )
}

/// Prefill `prefill` (logical positions `[offset, offset + prefill.len())`)
/// through `caches`, then greedily decode `DECODE_STEPS` tokens. `offset == 0`
/// is a cold full prefill; `offset > 0` continues after an adopted prefix.
fn prefill_and_decode(
    model: &mlxcel::LoadedModel,
    caches: &mut [mlxcel_core::cache::KVCache],
    prefill: &[i32],
    offset: i32,
) -> Vec<i32> {
    let len = prefill.len() as i32;
    let prompt = mlxcel_core::from_slice_i32(prefill, &[1, len]);
    let mask = mlxcel_core::utils::create_causal_mask(len, offset);
    let logits = model.forward(&prompt, caches, Some(&mask));
    mlxcel_core::eval(&logits);

    let mut next = greedy_token(&logits, 0, len - 1);
    let mut decoded = Vec::with_capacity(DECODE_STEPS);
    for _ in 0..DECODE_STEPS {
        decoded.push(next);
        let step_input = mlxcel_core::from_slice_i32(&[next], &[1, 1]);
        let logits = model.forward(&step_input, caches, None);
        mlxcel_core::eval(&logits);
        next = greedy_token(&logits, 0, 0);
    }
    decoded
}

fn load_or_skip(model_dir_name: &str, fetch_repo: &str) -> Option<mlxcel::LoadedModel> {
    let model_dir = repo_model_dir(model_dir_name);
    if !model_dir.exists() {
        eprintln!(
            "Skipping {model_dir_name}: model directory not found at {}.\n\
             Fetch with: ./target/release/mlxcel download {fetch_repo}",
            model_dir.display()
        );
        return None;
    }
    let (model, _tokenizer) =
        load_model(&model_dir).unwrap_or_else(|e| panic!("load {model_dir_name}: {e:?}"));
    Some(model)
}

/// Sequence A stores its prefix in the radix store; sequence B adopts it via the
/// paged pool, shares A's prefix blocks (no re-prefill), and decodes identically
/// to a cold no-cache run of the same `prefix + suffix` prompt. `cache_model_key`
/// is the `PromptCacheKey` model field (arbitrary, but must match between A's
/// insert and B's lookup).
fn assert_prefix_share_parity(model: &mlxcel::LoadedModel, label: &str, cache_model_key: &str) {
    let num_layers = model.num_layers();
    let prefix_len = PREFIX_TOKENS.len() as i32;

    let mut full_prompt = PREFIX_TOKENS.to_vec();
    full_prompt.extend_from_slice(SUFFIX_TOKENS);

    eprintln!(
        "\n=== paged prefix-share parity: {label} ({num_layers} layers, \
         prefix={prefix_len}, suffix={}) ===",
        SUFFIX_TOKENS.len()
    );

    // ---- COLD reference: full cold prefill (no cache) + decode. ----
    let cold_tokens = {
        let mut pool = CachePool::new(2);
        let id = pool
            .allocate_with_layout(model, Some(scheduler_paged_layout(num_layers)))
            .expect("cold paged allocate");
        let caches = pool.get_caches_mut(id).unwrap();
        prefill_and_decode(model, caches, &full_prompt, 0)
    };
    eprintln!("cold   decoded: {cold_tokens:?}");

    // ---- CACHED path through the real radix store. ----
    let store = PromptCacheStore::with_config(PromptCacheConfig::new(
        true,
        1 << 30,
        64,
        Duration::from_secs(3600),
        1,
    ));
    let mut pool = CachePool::new(4);

    // A prefills ONLY the shared prefix (positions [0, prefix_len)).
    let seq_a = pool
        .allocate_with_layout(model, Some(scheduler_paged_layout(num_layers)))
        .expect("A paged allocate");
    {
        let caches = pool.get_caches_mut(seq_a).unwrap();
        let prompt = mlxcel_core::from_slice_i32(PREFIX_TOKENS, &[1, prefix_len]);
        let mask = mlxcel_core::utils::create_causal_mask(prefix_len, 0);
        let logits = model.forward(&prompt, caches, Some(&mask));
        mlxcel_core::eval(&logits);
    }
    let a_blocks = pool
        .get_paged_state(seq_a)
        .unwrap()
        .layer(0)
        .unwrap()
        .block_ids
        .clone();
    assert!(
        a_blocks.len() >= 2,
        "40-token prefix should span >= 2 blocks (got {})",
        a_blocks.len()
    );

    // A finishes → donate: detach_paged → DetachedKvSet::Paged → store.
    let set_a = pool.detach_paged(seq_a).expect("A detach_paged");
    // The detached set pins every prefix block (refcount > 1: the set's block
    // table + the detach refcount bump), so the pool cannot recycle the prefix.
    assert_eq!(
        pool.paged_pool_ref().unwrap().refcount(a_blocks[0]),
        2,
        "detached prefix block must be pinned (refcount > 1)"
    );
    let entry = CacheEntry::new(PREFIX_TOKENS.to_vec(), DetachedKvSet::Paged(set_a));
    let insert_key = PromptCacheKey::new_full(
        cache_model_key,
        None,
        "tpl-v1",
        Some("sess"),
        MultimodalDigest::empty(),
        PREFIX_TOKENS,
    );
    store.insert(&insert_key, entry).expect("store insert");

    // B arrives with [prefix + suffix]; the store returns A's prefix entry.
    let lookup_key = PromptCacheKey::new_full(
        cache_model_key,
        None,
        "tpl-v1",
        Some("sess"),
        MultimodalDigest::empty(),
        &full_prompt,
    );
    let (hit_entry, matched_len) = store
        .lookup_longest_prefix(&lookup_key, &full_prompt)
        .expect("B must hit A's cached prefix");
    assert_eq!(
        matched_len,
        PREFIX_TOKENS.len(),
        "B must match the full stored prefix"
    );
    let detached_b = hit_entry
        .take_detached()
        .expect("first take yields the set");
    let set_b = match detached_b {
        DetachedKvSet::Paged(p) => p,
        DetachedKvSet::Dense(_) => panic!("paged backend must store a paged set"),
    };
    let seq_b = pool.adopt_paged(model, set_b).expect("B adopt_paged");

    // B SHARES A's prefix blocks (the prefix is stored once, not re-prefilled).
    let b_blocks = pool
        .get_paged_state(seq_b)
        .unwrap()
        .layer(0)
        .unwrap()
        .block_ids
        .clone();
    assert_eq!(
        b_blocks, a_blocks,
        "B must reuse A's prefix block ids (no re-prefill / second copy)"
    );
    // adopt transferred the pin onto B and released the detach bump → refcount 1.
    assert_eq!(
        pool.paged_pool_ref().unwrap().refcount(a_blocks[0]),
        1,
        "after adopt the prefix block is solely owned by B"
    );

    // B prefills ONLY its suffix at offset = prefix_len, then decodes.
    let cached_tokens = {
        let caches = pool.get_caches_mut(seq_b).unwrap();
        assert!(
            caches.iter().all(|c| c.is_paged_backed()),
            "adopted B must have pool-backed caches"
        );
        prefill_and_decode(model, caches, SUFFIX_TOKENS, prefix_len)
    };
    eprintln!("cached decoded: {cached_tokens:?}");

    assert_eq!(
        cached_tokens, cold_tokens,
        "shared-prefix (adopted) decode must equal the cold no-cache run\n\
         cold:   {cold_tokens:?}\ncached: {cached_tokens:?}"
    );
    eprintln!(
        "OK: B reused A's {}-block prefix and decoded {DECODE_STEPS} tokens identically to cold.",
        a_blocks.len()
    );
}

#[test]
#[ignore = "loads qwen3-0.6b-4bit and runs real GPU forwards; run with --ignored"]
fn paged_prefix_share_matches_cold_run_qwen3() {
    let _runtime = initialize_runtime();
    let Some(model) = load_or_skip(QWEN3_DIR, "mlx-community/Qwen3-0.6B-4bit") else {
        return;
    };
    assert_prefix_share_parity(&model, QWEN3_DIR, "qwen3");
}

#[test]
#[ignore = "loads llama-3.2-1b-4bit and runs real GPU forwards; run with --ignored"]
fn paged_prefix_share_matches_cold_run_llama3() {
    let _runtime = initialize_runtime();
    let Some(model) = load_or_skip(LLAMA3_DIR, "mlx-community/Llama-3.2-1B-Instruct-4bit") else {
        return;
    };
    assert_prefix_share_parity(&model, LLAMA3_DIR, "llama3");
}

// ---------------------------------------------------------------------------
// Partial prefix adoption (#225): APC-clamped match, floored to the pool block
// boundary, trimmed, adopted, and decoded byte-identically to cold.
// ---------------------------------------------------------------------------

/// Tail stored after the shared prefix in A's entry (12 tokens). B shares the
/// first 8 of them and then diverges, so the APC chain agrees through token 48
/// (three 16-token APC blocks) and the scheduler floor lands at 32 (one pool
/// block), exercising a real non-trivial floor (48 -> 32).
const STORED_TAIL_TOKENS: &[i32] = &[14582, 25, 3555, 374, 220, 16, 488, 220, 18, 30, 6771, 594];

/// B's continuation: same first 8 tail tokens, then divergent.
const PARTIAL_SUFFIX_TOKENS: &[i32] = &[14582, 25, 3555, 374, 220, 16, 488, 220, 24, 11, 4226, 30];

fn assert_partial_prefix_share_parity(
    model: &mlxcel::LoadedModel,
    label: &str,
    cache_model_key: &str,
) {
    use mlxcel::server::prompt_cache::ApcConfig;

    let num_layers = model.num_layers();

    // Stored entry: prefix + tail = 52 tokens (2 pool blocks: 32 + 20).
    let mut stored_tokens = PREFIX_TOKENS.to_vec();
    stored_tokens.extend_from_slice(STORED_TAIL_TOKENS);
    // B's request: shares the stored entry's first 48 tokens, then diverges.
    let mut b_prompt = PREFIX_TOKENS.to_vec();
    b_prompt.extend_from_slice(PARTIAL_SUFFIX_TOKENS);
    assert_eq!(stored_tokens[..48], b_prompt[..48], "share 48 tokens");
    assert_ne!(stored_tokens[48], b_prompt[48], "diverge at token 48");

    eprintln!(
        "\n=== paged PARTIAL prefix-share parity: {label} ({num_layers} layers, \
         stored={}, request={}) ===",
        stored_tokens.len(),
        b_prompt.len()
    );

    // ---- COLD reference for B's full prompt. ----
    let cold_tokens = {
        let mut pool = CachePool::new(2);
        let id = pool
            .allocate_with_layout(model, Some(scheduler_paged_layout(num_layers)))
            .expect("cold paged allocate");
        let caches = pool.get_caches_mut(id).unwrap();
        prefill_and_decode(model, caches, &b_prompt, 0)
    };
    eprintln!("cold    decoded: {cold_tokens:?}");

    // ---- Store with APC enabled (16-token blocks, the server default). ----
    // ApcConfig is #[non_exhaustive]; mutate a default instead of constructing.
    let apc = {
        let mut apc = ApcConfig::default();
        apc.enabled = true;
        apc
    };
    let store = PromptCacheStore::with_config(
        PromptCacheConfig::new(true, 1 << 30, 64, Duration::from_secs(3600), 1).with_apc(apc),
    );
    let mut pool = CachePool::new(4);

    // A prefills the full stored sequence and donates it.
    let seq_a = pool
        .allocate_with_layout(model, Some(scheduler_paged_layout(num_layers)))
        .expect("A paged allocate");
    {
        let caches = pool.get_caches_mut(seq_a).unwrap();
        let len = stored_tokens.len() as i32;
        let prompt = mlxcel_core::from_slice_i32(&stored_tokens, &[1, len]);
        let mask = mlxcel_core::utils::create_causal_mask(len, 0);
        let logits = model.forward(&prompt, caches, Some(&mask));
        mlxcel_core::eval(&logits);
    }
    let a_blocks = pool
        .get_paged_state(seq_a)
        .unwrap()
        .layer(0)
        .unwrap()
        .block_ids
        .clone();
    assert_eq!(a_blocks.len(), 2, "52 tokens = 2 pool blocks");
    let set_a = pool.detach_paged(seq_a).expect("A detach_paged");
    let entry = CacheEntry::new(stored_tokens.clone(), DetachedKvSet::Paged(set_a));
    let insert_key = PromptCacheKey::new_full(
        cache_model_key,
        None,
        "tpl-v1",
        Some("sess"),
        MultimodalDigest::empty(),
        &stored_tokens,
    );
    store.insert(&insert_key, entry).expect("store insert");

    // ---- B looks up: APC clamps the match to 48 (three 16-token blocks). ----
    let lookup_key = PromptCacheKey::new_full(
        cache_model_key,
        None,
        "tpl-v1",
        Some("sess"),
        MultimodalDigest::empty(),
        &b_prompt,
    );
    let (hit_entry, matched_len) = store
        .lookup_longest_prefix(&lookup_key, &b_prompt)
        .expect("B must hit A's entry via the APC partial path");
    assert_eq!(matched_len, 48, "APC must clamp the match to token 48");

    // ---- Mirror the scheduler's #225 paged arm: floor to the pool block
    //      boundary, trim, adopt. ----
    let mut set_b = match hit_entry.take_detached().expect("first take") {
        DetachedKvSet::Paged(p) => p,
        DetachedKvSet::Dense(_) => panic!("paged backend must store a paged set"),
    };
    let adoptable = (matched_len / BLOCK_SIZE) * BLOCK_SIZE;
    assert_eq!(adoptable, 32, "48 floors to one 32-token pool block");
    pool.trim_detached_paged_to(&mut set_b, adoptable)
        .expect("partial trim");
    assert_eq!(set_b.seq_len(), adoptable);
    let seq_b = pool.adopt_paged(model, set_b).expect("B adopt_paged");

    // B keeps A's FIRST block (shared physical prefix), dropped the second.
    let b_blocks = pool
        .get_paged_state(seq_b)
        .unwrap()
        .layer(0)
        .unwrap()
        .block_ids
        .clone();
    assert_eq!(b_blocks.len(), 1, "trimmed to one shared prefix block");
    assert_eq!(b_blocks[0], a_blocks[0], "B must reuse A's first block");

    // ---- B re-prefills everything past the adopted 32 tokens and decodes. ----
    let cached_tokens = {
        let caches = pool.get_caches_mut(seq_b).unwrap();
        assert!(
            caches.iter().all(|c| c.is_paged_backed()),
            "adopted B must have pool-backed caches"
        );
        prefill_and_decode(model, caches, &b_prompt[adoptable..], adoptable as i32)
    };
    eprintln!("partial decoded: {cached_tokens:?}");

    assert_eq!(
        cached_tokens, cold_tokens,
        "partial-prefix (trimmed + adopted) decode must equal the cold run\n\
         cold:    {cold_tokens:?}\npartial: {cached_tokens:?}"
    );
    eprintln!("OK: B adopted a trimmed 32-token prefix and decoded identically to cold.");
}

#[test]
#[ignore = "loads qwen3-0.6b-4bit and runs real GPU forwards; run with --ignored"]
fn paged_partial_prefix_share_matches_cold_run_qwen3() {
    let _runtime = initialize_runtime();
    let Some(model) = load_or_skip(QWEN3_DIR, "mlx-community/Qwen3-0.6B-4bit") else {
        return;
    };
    assert_partial_prefix_share_parity(&model, QWEN3_DIR, "qwen3");
}

#[test]
#[ignore = "loads llama-3.2-1b-4bit and runs real GPU forwards; run with --ignored"]
fn paged_partial_prefix_share_matches_cold_run_llama3() {
    let _runtime = initialize_runtime();
    let Some(model) = load_or_skip(LLAMA3_DIR, "mlx-community/Llama-3.2-1B-Instruct-4bit") else {
        return;
    };
    assert_partial_prefix_share_parity(&model, LLAMA3_DIR, "llama3");
}
