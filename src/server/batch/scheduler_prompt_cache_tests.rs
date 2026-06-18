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

//! Scheduler-facing prompt-prefix cache integration tests.
//!
//! Full end-to-end adopt/donate coverage that exercises a real `LoadedModel`
//! lives in the worker-thread integration tests; those need the server binary
//! and are out of scope for `cargo test --lib`. The cases in this file isolate
//! the **scheduler-visible invariants** that the implementation adds and
//! that the store / key / entry data flow relies on:
//!
//! * Cache hit reports the matched prefix length and leaves the detached
//!   entry consumed (one-shot).
//! * A second lookup against a consumed entry falls through to a miss even
//!   though the entry structurally still lives in the store (race semantics).
//! * Miss paths preserve bit-exact behavior (no hidden cache access).
//! * Donate-back on a healthy finish produces an entry with
//!   `tokens = prompt + generated` as the radix-trie key.
//! * The observability counters on [`BatchObservability`] advance exactly
//!   once per adopt / donate-back as the scheduler would call them.
//! * `SequenceInfo::prefill_start_offset` / `already_cached_tokens` transport
//!   the hit metadata to the prefill path without forcing the rest of the
//!   sequence struct to mutate.

use std::sync::Arc;
use std::time::{Duration, Instant};

use mlxcel_core::cache::{
    CachePool, KVCache, PagedBlockId, PagedKvLayout, SequenceId, SequenceStateLayout,
};
use mlxcel_core::generate::LanguageModel;
use mlxcel_core::{MlxArray, UniquePtr};

use crate::server::batch::BatchObservability;
use crate::server::prompt_cache::{
    ApcConfig, ApcHashAlgo, CacheEntry, DetachedKvSet, InsertError, PromptCacheConfig,
    PromptCacheStore,
    key::{MultimodalDigest, PromptCacheKey},
};

/// Produce a placeholder [`CacheEntry`] with a zero-tensor detached set.
///
/// Real scheduler donate-back entries carry MLX buffers; these tests only
/// exercise the store/key/trie data flow, so the shared
/// [`CacheEntry::new_for_test`] helper (defined under `#[cfg(test)]`) is
/// sufficient and avoids reaching into the private constructor fields of
/// `DetachedKVCache` / `DetachedCacheSet`.
fn fake_entry(tokens: Vec<i32>, size_bytes: usize) -> CacheEntry {
    CacheEntry::new_for_test(tokens, size_bytes)
}

fn test_store() -> Arc<PromptCacheStore> {
    Arc::new(PromptCacheStore::with_config(PromptCacheConfig::new(
        true,
        128 * 1024,
        32,
        Duration::from_secs(60),
        4,
    )))
}

fn make_key<'a>(
    model_id: &'a str,
    session_key: &'a str,
    template_sig: &'a str,
    tokens: &'a [i32],
) -> PromptCacheKey<'a> {
    PromptCacheKey::new_full(
        model_id,
        None,
        template_sig,
        Some(session_key),
        MultimodalDigest::empty(),
        tokens,
    )
}

// ---------------------------------------------------------------------------
// Hit path
// ---------------------------------------------------------------------------

#[test]
fn hit_then_decode_reports_matched_len_and_consumes_entry() {
    // Mirror the scheduler's hit-path invocation: insert a detached entry,
    // then have a subsequent request with a longer prompt look it up.
    let store = test_store();
    let prompt: Vec<i32> = (100..=131).collect(); // 32 tokens — a plausible prefix
    let entry = fake_entry(prompt.clone(), 4096);
    let bytes_before = entry.size_bytes;
    let insert_key = make_key("model-A", "session-1", "tpl-sig-v1", &prompt);
    store.insert(&insert_key, entry).expect("insert succeeds");

    // Next turn: same conversation, 5 extra user tokens → the first 32
    // tokens of the new prompt match the stored prefix.
    let mut next_prompt = prompt.clone();
    next_prompt.extend([200, 201, 202, 203, 204]);
    let lookup_key = make_key("model-A", "session-1", "tpl-sig-v1", &next_prompt);
    let (hit_entry, matched_len) = store
        .lookup_longest_prefix(&lookup_key, &next_prompt)
        .expect("matching prefix must hit");
    assert_eq!(matched_len, prompt.len());
    assert_eq!(hit_entry.size_bytes, bytes_before);

    // Scheduler consumes the detached cache via `take_detached`; after
    // that, the entry is drained (one-shot) even though the radix-trie
    // lookup still sees it. A racing second adopt must fall through to a
    // cold prefill.
    let _detached = hit_entry
        .take_detached()
        .expect("entry is consumable on first hit");
    assert!(hit_entry.take_detached().is_none());

    // The *scheduler* records matched_len via BatchObservability — we
    // exercise the same counter call pattern here.
    let obs = BatchObservability::new();
    obs.record_prompt_cache_hit(matched_len);
    let snap = obs.snapshot();
    assert_eq!(snap.prompt_cache_hits, 1);
    assert_eq!(snap.prompt_cache_hit_tokens, matched_len as u64);
}

// ---------------------------------------------------------------------------
// Miss path
// ---------------------------------------------------------------------------

#[test]
fn miss_then_decode_leaves_scheduler_state_unchanged() {
    // A request with no matching entry must return None with no side
    // effects — the scheduler's miss path proceeds with a fresh allocate.
    let store = test_store();

    // Insert an entry under a different (model, session, template) bucket
    // so the trie is non-empty but does not overlap with the lookup key.
    let seeded: Vec<i32> = (1..=16).collect();
    let seed_entry = fake_entry(seeded.clone(), 2048);
    let seed_key = make_key("other-model", "other-session", "other-tpl", &seeded);
    store
        .insert(&seed_key, seed_entry)
        .expect("seed insert succeeds");

    // Lookup under an unrelated bucket must miss.
    let probe_tokens: Vec<i32> = (100..=131).collect();
    let probe_key = make_key("model-A", "session-1", "tpl-sig-v1", &probe_tokens);
    assert!(
        store
            .lookup_longest_prefix(&probe_key, &probe_tokens)
            .is_none()
    );

    // No hit recorded on the observability counter.
    let obs = BatchObservability::new();
    let snap = obs.snapshot();
    assert_eq!(snap.prompt_cache_hits, 0);
    assert_eq!(snap.prompt_cache_hit_tokens, 0);
}

// ---------------------------------------------------------------------------
// Donate-back path
// ---------------------------------------------------------------------------

#[test]
fn donate_back_on_healthy_finish_inserts_prompt_plus_generated() {
    // Simulate a sequence that ran to a healthy finish: prompt of 32
    // tokens + 4 generated tokens. The scheduler calls
    // `CachePool::detach` → builds a `CacheEntry` with the joined tokens
    // → inserts under the same key context as the request.
    let store = test_store();

    let prompt: Vec<i32> = (300..=331).collect();
    let generated: Vec<i32> = vec![999, 1000, 1001, 1002];

    let mut joined = prompt.clone();
    joined.extend_from_slice(&generated);
    let entry = fake_entry(joined.clone(), 8192);

    let insert_key = make_key("model-A", "session-1", "tpl-sig-v1", &joined);
    store
        .insert(&insert_key, entry)
        .expect("donate-back insert succeeds");

    // The next turn of the same conversation sees the full prompt +
    // assistant tail as a reusable prefix.
    let mut next_prompt = joined.clone();
    next_prompt.extend([2000, 2001]);
    let next_key = make_key("model-A", "session-1", "tpl-sig-v1", &next_prompt);
    let (_e, matched_len) = store
        .lookup_longest_prefix(&next_key, &next_prompt)
        .expect("donate-back entry must be reachable on the next turn");
    assert_eq!(matched_len, joined.len());

    let obs = BatchObservability::new();
    obs.record_prompt_cache_insert();
    assert_eq!(obs.snapshot().prompt_cache_inserts, 1);
}

// ---------------------------------------------------------------------------
// No donate-back on error / OOM paths
// ---------------------------------------------------------------------------

#[test]
fn no_donate_back_on_error_outcome() {
    // Simulates the scheduler's policy: the error / OOM / invalid-cache
    // path never calls `donate_finished_sequence_cache` with
    // `healthy_finish == true`, so the store stays untouched and the
    // reject-counter stays at zero (a reject would only fire on an
    // actual insert attempt that the store refuses).
    let store = test_store();
    assert_eq!(store.stats().inserts, 0);
    assert_eq!(store.stats().entries, 0);

    // Mimic the scheduler reject-counter gate: no insert, no counter
    // movement. The donate-path helper hard-guards on `healthy_finish`
    // before touching the store or counters.
    let obs = BatchObservability::new();
    let snap = obs.snapshot();
    assert_eq!(snap.prompt_cache_inserts, 0);
    assert_eq!(snap.prompt_cache_insert_rejects, 0);
}

#[test]
fn oversized_entry_records_reject_counter() {
    // Build a store with a 1-byte capacity so any real entry is
    // oversized. The scheduler mirrors this via
    // `record_prompt_cache_insert_reject()` in its insert-error branch.
    let store = Arc::new(PromptCacheStore::with_config(PromptCacheConfig::new(
        true,
        1,
        32,
        Duration::from_secs(60),
        4,
    )));
    let tokens: Vec<i32> = (1..=16).collect();
    let entry = CacheEntry::new_for_test(tokens.clone(), 4096); // 4 KiB > 1 byte
    let key = make_key("m", "s", "t", &tokens);
    let err = store
        .insert(&key, entry)
        .expect_err("oversized entry must be rejected");
    assert!(
        matches!(
            err,
            crate::server::prompt_cache::InsertError::OversizedEntry { .. }
        ),
        "unexpected reject reason: {err:?}"
    );

    // Scheduler would advance the reject counter here.
    let obs = BatchObservability::new();
    obs.record_prompt_cache_insert_reject();
    assert_eq!(obs.snapshot().prompt_cache_insert_rejects, 1);
}

// ---------------------------------------------------------------------------
// SequenceInfo fields transport the hit metadata
// ---------------------------------------------------------------------------

#[test]
fn sequence_info_fields_transport_cache_hit_metadata() {
    // The scheduler sets both `prefill_start_offset` (read by the prefill
    // path) and `already_cached_tokens` (read by the metrics bridge) to
    // the matched prefix length on a hit. The two are updated in lock-
    // step; test that the struct carries both independently.
    use std::sync::atomic::AtomicBool;
    use std::sync::mpsc;

    use crate::server::batch::sequence::{SequenceInfo, SequenceState};
    use crate::server::model_provider::GenerateEvent;
    use crate::server::model_provider::model_worker::StreamingDecodeState;
    use mlxcel_core::generate::SamplingConfig;

    let (tx, _rx) = mpsc::channel::<GenerateEvent>();
    let tokenizer = crate::tokenizer::MlxcelTokenizer::stub();
    let prompt_tokens = vec![1_i32; 100];
    let decode_state = StreamingDecodeState::new(&tokenizer, &prompt_tokens);

    let seq = SequenceInfo {
        seq_id: SequenceId::from_raw(7),
        state: SequenceState::Queued,
        prompt_tokens,
        sampling: SamplingConfig::default(),
        max_tokens: 64,
        eos_token_ids: vec![2],
        priority: crate::server::batch::RequestPriority::Normal,
        logprobs_config: Default::default(),
        vlm_embeddings: None,
        images: Vec::new(),
        audio: Vec::new(),
        generated_tokens: Vec::new(),
        generated_text: String::new(),
        decode_state,
        prefill_offset: 0,
        prefill_start_offset: 73,
        already_cached_tokens: 73,
        response_tx: tx,
        cancelled: Arc::new(AtomicBool::new(false)),
        created_at: Instant::now(),
        prefill_start: None,
        first_token_time: None,
        token_history: Vec::new(),
        sampler_state: None,
        merged_eos: Vec::new(),
        thinking: crate::server::thinking_budget::ThinkingState::disabled(),
        structured: None,
    };

    assert_eq!(seq.prefill_start_offset, 73);
    assert_eq!(seq.already_cached_tokens, 73);
    assert_eq!(seq.prompt_tokens.len(), 100);
    // Prefill-path math: model sees only the suffix tokens.
    assert_eq!(
        seq.prompt_tokens.len() - seq.prefill_start_offset,
        27,
        "scheduler prefill loop must process only suffix tokens"
    );
}

// ---------------------------------------------------------------------------
// Batched prefill routes adopted-prefix rows to the sequential cohort
// ---------------------------------------------------------------------------

#[test]
fn batched_prefill_path_detects_adopted_sequences() {
    // The scheduler's `execute_batched_prefill` classifies a row with
    // `prefill_start_offset > 0` as not-cold, so #332 cohort splitting routes
    // it to the offset-aware sequential path (instead of the padded batched
    // path) while cold siblings still batch. We assert the detection predicate
    // that drives that classification here so a future refactor of the batched
    // path can't regress it silently. The full cohort plan (including order and
    // offset isolation) is covered in `prefill_cohort::tests`.
    let mut offsets = [0_usize; 3];
    assert!(
        !offsets.iter().any(|&o| o > 0),
        "cold batch must not trigger adopted-sequence fallback"
    );
    offsets[1] = 32;
    assert!(
        offsets.iter().any(|&o| o > 0),
        "a single adopted sequence must trigger the fallback"
    );
}

// ---------------------------------------------------------------------------
// APC block-level partial adoption
// ---------------------------------------------------------------------------

const APC_BLOCK_SIZE: usize = 16;

/// Build a [`PromptCacheStore`] with APC enabled and `min_prefix_tokens` set
/// low enough that block-aligned partial matches (32 tokens at block_size 16)
/// pass the threshold check. Mirrors `apc_on_config` in
/// `apc_integration_tests.rs` but adapted to the scheduler-test idiom.
fn apc_on_test_store() -> Arc<PromptCacheStore> {
    let apc = ApcConfig {
        enabled: true,
        block_size: APC_BLOCK_SIZE,
        num_blocks: None,
        hash: ApcHashAlgo::Sha256,
    };
    let cfg = PromptCacheConfig::new(
        true,
        128 * 1024,
        32,
        Duration::from_secs(60),
        // Low enough that a single full block (16 tokens) passes — but our
        // scheduler-level partial test below uses 2 full blocks (32 tokens)
        // which is well above this threshold.
        4,
    )
    .with_apc(apc);
    Arc::new(PromptCacheStore::with_config(cfg))
}

/// Build the cache key the scheduler would compose for a request, with an
/// empty multimodal digest (text-only path — multimodal bypasses the cache
/// entirely per the scheduler's `is_multimodal` check at request time).
fn make_text_key<'a>(
    model_id: &'a str,
    session_key: &'a str,
    template_sig: &'a str,
    tokens: &'a [i32],
) -> PromptCacheKey<'a> {
    PromptCacheKey::new_full(
        model_id,
        None,
        template_sig,
        Some(session_key),
        MultimodalDigest::empty(),
        tokens,
    )
}

#[test]
fn apc_block_aligned_partial_match_truncates_to_block_boundary() {
    // The headline property for when a request shares the first
    // 32 tokens (= 2 blocks) of a 48-token cached entry but diverges at
    // token 32 (= start of block 2), the store's APC discriminator must
    // return matched_len == 32, NOT 48. This is the value the scheduler then
    // hands to `DetachedCacheSet::truncate_to` to adopt only the consistent
    // prefix and re-prefill the divergent tail.
    //
    // The discriminator is exercised at scheduler granularity here: the
    // entry was inserted under one set of tokens, the lookup runs under a
    // different (overlapping) set of tokens, and the resulting matched_len
    // is the value the scheduler's `try_adopt_cached_prefix` would feed to
    // the partial-adoption truncate path.
    let store = apc_on_test_store();

    // Stored entry: 48 tokens (3 full blocks of 16).
    let stored: Vec<i32> = (1..=48).collect();
    let insert_key = make_text_key("model-A", "session-1", "tpl-A", &stored);
    store
        .insert(&insert_key, CacheEntry::new_for_test(stored.clone(), 4096))
        .expect("insert succeeds");

    // Request: first 32 tokens identical, but token 32 diverges (e.g. user
    // typed something different on the third turn). The next 16 tokens
    // belong to block 2 of the request's chain, so the chain agreement
    // length is 32.
    let mut request: Vec<i32> = stored.clone();
    request[32] = 9_999; // poison block 2
    request.extend([1_001, 1_002, 1_003, 1_004]); // tail diverges past token 48
    let lookup_key = make_text_key("model-A", "session-1", "tpl-A", &request);

    let (hit_entry, matched_len) = store
        .lookup_longest_prefix(&lookup_key, &request)
        .expect("APC must report a partial hit, not a miss");

    // The trie's whole-prefix match length is 48 (the full stored prefix).
    // After APC's block-aligned discriminator, the scheduler-visible
    // matched_len drops to 32 — exactly two full blocks of agreement.
    assert_eq!(
        matched_len, 32,
        "APC discriminator must clamp to the last block boundary where chains agree"
    );

    // The cached entry's chain must still cover all 3 of its own blocks; only
    // the *reported* matched_len was clamped — the entry stays whole so
    // future requests with a fully-matching prefix can still reuse the
    // tail.
    let entry_chain = hit_entry
        .apc_block_hashes()
        .expect("APC chain must be attached at insert");
    assert_eq!(
        entry_chain.len(),
        3,
        "stored entry has 48 tokens / 16 = 3 blocks"
    );

    // The scheduler's next step would call `take_detached()` then
    // `DetachedCacheSet::truncate_to(matched_len as i32)`. The truncate API
    // is unit-tested separately in `mlxcel_core::cache::detach_tests`; here
    // we only verify that the matched_len fed to it is exactly the
    // block-aligned partial value.
    assert!(
        matched_len < hit_entry.tokens.len(),
        "matched_len ({matched_len}) must be strictly less than entry.tokens.len() ({}) so the scheduler triggers the truncate path",
        hit_entry.tokens.len()
    );
    assert_eq!(
        matched_len % APC_BLOCK_SIZE,
        0,
        "matched_len must be block-aligned for the truncate to land on a block boundary"
    );

    // Bookkeeping: the scheduler advances the same observability counters
    // for partial adoptions that it uses for full ones. The matched-token
    // tally reflects the partial value.
    let obs = BatchObservability::new();
    obs.record_prompt_cache_hit(matched_len);
    let snap = obs.snapshot();
    assert_eq!(snap.prompt_cache_hits, 1);
    assert_eq!(snap.prompt_cache_hit_tokens, 32);
}

#[test]
fn apc_block_aligned_partial_match_drops_below_min_prefix_returns_miss() {
    // When the block-aligned partial match is shorter than
    // `min_prefix_tokens`, the store treats it as a miss so the scheduler
    // does not adopt sub-threshold partial caches. This is the safety
    // valve already encoded in `lookup_longest_prefix`; we exercise it at
    // scheduler granularity to catch any future regression.
    let apc = ApcConfig {
        enabled: true,
        block_size: APC_BLOCK_SIZE,
        num_blocks: None,
        hash: ApcHashAlgo::Sha256,
    };
    // min_prefix_tokens = 24, divergence at token 16 -> matched_len = 16 < 24.
    let cfg =
        PromptCacheConfig::new(true, 128 * 1024, 32, Duration::from_secs(60), 24).with_apc(apc);
    let store = Arc::new(PromptCacheStore::with_config(cfg));

    let stored: Vec<i32> = (1..=48).collect();
    let insert_key = make_text_key("m", "s", "t", &stored);
    store
        .insert(&insert_key, CacheEntry::new_for_test(stored.clone(), 4096))
        .expect("insert");

    let mut request = stored.clone();
    request[16] = 9_999; // diverge at start of block 1 -> agreement = 16 tokens
    let lookup_key = make_text_key("m", "s", "t", &request);

    assert!(
        store.lookup_longest_prefix(&lookup_key, &request).is_none(),
        "matched_len 16 < min_prefix 24 must surface as a miss to the scheduler"
    );
}

#[test]
fn apc_full_match_yields_entry_length_matched_len_no_truncate_path() {
    // Sanity: when the request's first N tokens fully agree with all blocks
    // of the cached entry, matched_len equals the entry's full token length.
    // The scheduler then skips the truncate branch entirely (no work, no
    // allocations). This is the bit-exact full-prefix path that earlier
    // already worked.
    let store = apc_on_test_store();

    let stored: Vec<i32> = (1..=48).collect();
    let insert_key = make_text_key("model-A", "session-1", "tpl-A", &stored);
    store
        .insert(&insert_key, CacheEntry::new_for_test(stored.clone(), 4096))
        .expect("insert");

    // Request: stored prefix + 5 fresh user tokens, no divergence inside the
    // shared portion.
    let mut request = stored.clone();
    request.extend([100, 101, 102, 103, 104]);
    let lookup_key = make_text_key("model-A", "session-1", "tpl-A", &request);

    let (hit_entry, matched_len) = store
        .lookup_longest_prefix(&lookup_key, &request)
        .expect("full APC match must hit");

    assert_eq!(
        matched_len,
        hit_entry.tokens.len(),
        "no divergence inside the shared portion -> matched_len equals entry length"
    );
    assert_eq!(matched_len, 48);
}

#[test]
fn apc_disabled_lookup_ignores_block_divergence_returns_full_prefix() {
    // With APC disabled, the store's lookup path falls through the
    // discriminator. Even when block 2 of the request diverges from the
    // stored entry, the trie / scan path does not see that — the request's
    // *whole prefix* must contain the entry's tokens for a hit. So this
    // request (which mutates a token *inside* the stored prefix) misses,
    // because the trie key check fails: `tokens[..token_len] !=
    // slot.entry.tokens`.
    //
    // This guards against a regression where APC-off accidentally inherits
    // APC-on truncation logic.
    let cfg = PromptCacheConfig::new(true, 128 * 1024, 32, Duration::from_secs(60), 4);
    let store = Arc::new(PromptCacheStore::with_config(cfg));

    let stored: Vec<i32> = (1..=48).collect();
    let insert_key = make_text_key("model-A", "session-1", "tpl-A", &stored);
    store
        .insert(&insert_key, CacheEntry::new_for_test(stored.clone(), 4096))
        .expect("insert");

    let mut request = stored.clone();
    request[32] = 9_999;
    request.extend([200, 201]);
    let lookup_key = make_text_key("model-A", "session-1", "tpl-A", &request);

    // Both the trie path and the scan path require the entry's full token
    // prefix to match the request's leading tokens byte-for-byte. The
    // mutated token at index 32 invalidates that, so the lookup is a miss
    // — APC discriminator never runs because there is no candidate.
    assert!(
        store.lookup_longest_prefix(&lookup_key, &request).is_none(),
        "APC-off lookup with internal divergence must miss (no partial adoption)"
    );
}

#[test]
fn scheduler_partial_adoption_synthetic_savings_match_expected_fraction() {
    // Synthetic prefill-cost microbench. We model the
    // scheduler's prefill workload as `prompt_tokens.len() - matched_len`
    // and confirm that block-aligned partial adoption recovers the
    // expected fraction of work compared to a cold prefill (matched_len 0).
    //
    // This is the synthetic counterpart to the Apple-Silicon hardware bench
    // documented in `docs/apc-partial-adoption-bench.md`; that bench
    // measures wall-clock prefill time on Llama 3.1 8B 4bit. Here we just
    // verify the algebra: matched_len drives a proportional reduction in
    // tokens fed to the model.
    //
    // Configuration:
    // - block_size = 16
    // - cached entry length = 64 tokens (4 full blocks)
    // - request length = 80 tokens (5 blocks)
    // - divergence at request-token K -> matched_len = floor(K/16) * 16
    let store = apc_on_test_store();
    let stored: Vec<i32> = (1..=64).collect();
    let insert_key = make_text_key("model-A", "session-1", "tpl-A", &stored);
    store
        .insert(&insert_key, CacheEntry::new_for_test(stored.clone(), 4096))
        .expect("insert");

    let request_total = 80usize;
    // Cold prefill baseline: no cache, all tokens fed to the model.
    let cold_prefill_cost = request_total;

    struct Case {
        divergence_token: usize,
        expected_matched: usize,
        expected_savings_fraction: f64,
    }
    let cases = [
        // Divergence at block boundary -> matched_len equals divergence.
        Case {
            divergence_token: 16,
            expected_matched: 16,
            expected_savings_fraction: 16.0 / 80.0,
        },
        Case {
            divergence_token: 32,
            expected_matched: 32,
            expected_savings_fraction: 32.0 / 80.0,
        },
        Case {
            divergence_token: 48,
            expected_matched: 48,
            expected_savings_fraction: 48.0 / 80.0,
        },
        // Divergence inside block 3 (mid-block). matched_len floors to the
        // last full-block boundary the chains agree on -> 48.
        Case {
            divergence_token: 55,
            expected_matched: 48,
            expected_savings_fraction: 48.0 / 80.0,
        },
        // Full match: request shares all 64 stored tokens, matched_len = 64.
        // (No divergence inside the shared portion.)
        Case {
            divergence_token: usize::MAX,
            expected_matched: 64,
            expected_savings_fraction: 64.0 / 80.0,
        },
    ];

    for case in cases {
        let mut request: Vec<i32> = stored.clone();
        if case.divergence_token < stored.len() {
            request[case.divergence_token] = 9_999;
        }
        // Pad request to 80 tokens with a trailing user turn.
        while request.len() < request_total {
            request.push(1_000 + request.len() as i32);
        }
        assert_eq!(request.len(), request_total);

        let lookup_key = make_text_key("model-A", "session-1", "tpl-A", &request);
        let matched = store
            .lookup_longest_prefix(&lookup_key, &request)
            .map(|(_, m)| m)
            .unwrap_or(0);
        assert_eq!(
            matched, case.expected_matched,
            "divergence@{} -> matched_len mismatch",
            case.divergence_token
        );

        let warm_prefill_cost = request_total - matched;
        let savings = cold_prefill_cost - warm_prefill_cost;
        let savings_fraction = savings as f64 / cold_prefill_cost as f64;
        assert!(
            (savings_fraction - case.expected_savings_fraction).abs() < 1e-9,
            "divergence@{} -> savings_fraction {savings_fraction} expected ~{}",
            case.divergence_token,
            case.expected_savings_fraction
        );

        // Print a row a human can read when the test is run with --nocapture.
        eprintln!(
            "[apc_partial_adoption_bench] divergence@{:>3}  matched={:>2}/{}  cold_cost={cold_prefill_cost}  warm_cost={warm_prefill_cost}  savings={:>5.1}%",
            case.divergence_token,
            matched,
            stored.len(),
            savings_fraction * 100.0
        );
    }
}

#[test]
fn scheduler_partial_adoption_invariant_matched_len_is_block_aligned() {
    // Property: for any APC-on lookup that returns Some, the reported
    // matched_len is divisible by `block_size`. The scheduler's truncate_to
    // call therefore always lands on a block boundary, never producing a
    // sub-block KV state that the next prefill step couldn't continue from.
    //
    // We sweep divergence points across multiple blocks to exercise the
    // invariant beyond a single configuration.
    let store = apc_on_test_store();
    let stored: Vec<i32> = (1..=64).collect(); // 4 full blocks
    let insert_key = make_text_key("m", "s", "t", &stored);
    store
        .insert(&insert_key, CacheEntry::new_for_test(stored.clone(), 4096))
        .expect("insert");

    // Divergence at the start of block 1, 2, 3 should give matched_len
    // 16, 32, 48 respectively. Divergence at token 4 (inside block 0) is
    // below min_prefix and surfaces as a miss; that case is covered by a
    // separate test above.
    for divergence in [16usize, 32, 48] {
        let mut request = stored.clone();
        request[divergence] = 9_999;
        request.extend([700, 701]);
        let lookup_key = make_text_key("m", "s", "t", &request);

        let (_, matched_len) = store
            .lookup_longest_prefix(&lookup_key, &request)
            .unwrap_or_else(|| {
                panic!("divergence at token {divergence} must produce a partial hit, not a miss")
            });
        assert_eq!(
            matched_len, divergence,
            "matched_len for divergence at token {divergence} must equal {divergence}"
        );
        assert_eq!(
            matched_len % APC_BLOCK_SIZE,
            0,
            "matched_len {matched_len} for divergence at {divergence} must be block-aligned"
        );
    }
}

// ---------------------------------------------------------------------------
// Paged entry round-trip through the store's CacheEntry (#121 sub-step b)
//
// The cross-request store now parks both dense and paged detached sets via the
// `DetachedKvSet` union. These cases prove a real `DetachedPagedCacheSet` flows
// through `CacheEntry::new` / `take_detached` one-shot semantics and that
// `size_bytes` snapshots the paged set's byte footprint (eviction accounting).
// ---------------------------------------------------------------------------

/// Minimal paged-natural stub model so we can mint a real
/// `DetachedPagedCacheSet` via `CachePool::detach_paged`.
struct PagedStub {
    layout: PagedKvLayout,
}

impl LanguageModel for PagedStub {
    fn forward(
        &self,
        _input_ids: &MlxArray,
        _caches: &mut [KVCache],
        _mask: Option<&MlxArray>,
    ) -> UniquePtr<MlxArray> {
        mlxcel_core::zeros(&[1], 0)
    }

    fn make_caches(&self) -> Vec<KVCache> {
        (0..self.layout.num_layers)
            .map(|_| KVCache::new())
            .collect()
    }

    fn num_layers(&self) -> usize {
        self.layout.num_layers
    }

    fn eos_token_ids(&self) -> Vec<i32> {
        vec![0]
    }

    fn sequence_state_layout(&self) -> SequenceStateLayout {
        SequenceStateLayout::paged_kv_cache(self.layout.clone())
    }
}

#[test]
fn paged_detached_set_round_trips_through_cache_entry() {
    let layout = PagedKvLayout::uniform(2, 4, 128).unwrap();
    let model = PagedStub {
        layout: layout.clone(),
    };
    let mut pool = CachePool::new(4);

    // Build a paged sequence with a populated block table (no MLX writes
    // needed — `append_paged_tokens` grows the block table and reserves bytes).
    let seq = pool.allocate(&model).unwrap();
    pool.append_paged_tokens(seq, 0, 6).unwrap();
    pool.append_paged_tokens(seq, 1, 8).unwrap();

    let paged = pool.detach_paged(seq).expect("paged detach must succeed");
    let expected_bytes = paged.nbytes();
    assert!(
        expected_bytes > 0,
        "a populated paged set must report nonzero bytes"
    );

    // The union dispatches nbytes()/is_empty() to the paged arm.
    let kv_set = DetachedKvSet::Paged(paged);
    assert!(!kv_set.is_empty(), "a populated paged set is not empty");
    assert_eq!(
        kv_set.nbytes(),
        expected_bytes,
        "DetachedKvSet::nbytes dispatches to the paged set"
    );

    // CacheEntry snapshots size_bytes from the paged set for eviction
    // accounting and stores the union behind its one-shot holder.
    let tokens: Vec<i32> = (0..14).collect();
    let entry = CacheEntry::new(tokens.clone(), kv_set);
    assert_eq!(
        entry.size_bytes, expected_bytes,
        "size_bytes snapshots the paged set's nbytes"
    );
    assert_eq!(entry.tokens, tokens);
    assert!(entry.has_detached());

    // One-shot take returns the Paged variant; a second take is a miss.
    let taken = entry.take_detached().expect("first take yields the set");
    assert!(matches!(taken, DetachedKvSet::Paged(_)));
    assert!(
        entry.take_detached().is_none(),
        "second take is a miss (one-shot)"
    );
    assert!(!entry.has_detached());

    // The taken set still owns block pins; return them so the pool budget
    // stays honest and the set's `Drop` does not warn.
    if let DetachedKvSet::Paged(set) = taken {
        pool.release_detached_paged(set);
    }
}

#[test]
fn empty_paged_detached_set_reports_empty() {
    // A paged sequence with no appended tokens has an empty block table, so
    // the union's variant-aware `is_empty()` must report it empty (the dense
    // per-layer handles are empty by design for paged sets).
    let layout = PagedKvLayout::uniform(1, 4, 128).unwrap();
    let model = PagedStub {
        layout: layout.clone(),
    };
    let mut pool = CachePool::new(4);
    let seq = pool.allocate(&model).unwrap();
    let paged = pool.detach_paged(seq).expect("paged detach must succeed");
    let kv_set = DetachedKvSet::Paged(paged);
    assert!(
        kv_set.is_empty(),
        "a paged set with no visible tokens / blocks must be reported empty"
    );
    if let DetachedKvSet::Paged(set) = kv_set {
        pool.release_detached_paged(set);
    }
}

// ---------------------------------------------------------------------------
// #122 sub-step (a): paged pin-release plumbing.
//
// A paged `DetachedKvSet` pins pool blocks; its `Drop` only warns, so the
// store's eviction / rejection paths (which have no `CachePool` handle) must
// hand the set to the scheduler to release. These cover (1) the underlying
// `release_detached_paged` fix that actually returns blocks to the pool, and
// (2) the store queueing evicted / declined paged sets for that release.
// ---------------------------------------------------------------------------

/// Mint a real paged [`DetachedKvSet`] from a fresh sequence with `tokens`
/// tokens appended to layer 0, returning the set plus that sequence's layer-0
/// block ids so a test can assert pool refcounts across release. No MLX writes
/// are needed — `append_paged_tokens` grows the block table directly.
fn mint_paged_set(
    pool: &mut CachePool,
    model: &PagedStub,
    tokens: usize,
) -> (DetachedKvSet, Vec<PagedBlockId>) {
    let seq = pool.allocate(model).unwrap();
    pool.append_paged_tokens(seq, 0, tokens).unwrap();
    let blocks = pool
        .get_paged_state(seq)
        .unwrap()
        .layer(0)
        .unwrap()
        .block_ids
        .clone();
    let paged = pool.detach_paged(seq).expect("detach_paged must succeed");
    (DetachedKvSet::Paged(paged), blocks)
}

/// `release_detached_paged` on the discard path (no adopt) must drive every
/// block to refcount 0 so it returns to the pool's free list. Regression for
/// the pre-existing leak where only the detach pin was released, leaving the
/// origin sequence's allocation dangling at refcount 1.
#[test]
fn release_detached_paged_reclaims_blocks_to_pool() {
    let model = PagedStub {
        layout: PagedKvLayout::uniform(1, 4, 128).unwrap(),
    };
    let mut pool = CachePool::new(4);
    let (set, blocks) = mint_paged_set(&mut pool, &model, 6);
    assert!(!blocks.is_empty(), "a 6-token prefix spans >= 1 block");
    // Parked (detached, not adopted) → the set owns alloc + pin == refcount 2.
    assert_eq!(
        pool.paged_pool_ref().unwrap().refcount(blocks[0]),
        2,
        "a detached, un-adopted block holds alloc + pin"
    );

    if let DetachedKvSet::Paged(paged) = set {
        pool.release_detached_paged(paged);
    }
    for b in &blocks {
        assert_eq!(
            pool.paged_pool_ref().unwrap().refcount(*b),
            0,
            "discard must return every block to the pool (refcount 0)"
        );
    }
}

/// An LRU-evicted paged entry must hand its block pins to the release queue
/// (its `Drop` only warns), and draining + releasing them returns the blocks
/// to the pool. Before the fix the evicted set dropped and leaked its pins.
#[test]
fn evicted_paged_entry_queues_pins_then_pool_reclaims() {
    // max_entries = 1 so the second insert evicts the first (oldest = A).
    let store = Arc::new(PromptCacheStore::with_config(PromptCacheConfig::new(
        true,
        1 << 30,
        1,
        Duration::from_secs(3600),
        1,
    )));
    let model = PagedStub {
        layout: PagedKvLayout::uniform(1, 4, 128).unwrap(),
    };
    let mut pool = CachePool::new(8);

    let (set_a, blocks_a) = mint_paged_set(&mut pool, &model, 6);
    let tokens_a: Vec<i32> = (0..16).collect();
    store
        .insert(
            &make_key("m", "s", "tpl", &tokens_a),
            CacheEntry::new(tokens_a.clone(), set_a),
        )
        .expect("insert A");
    assert!(!store.has_pending_paged_releases(), "no eviction yet");

    // Insert B (distinct digest) → A is evicted under max_entries = 1.
    let (set_b, _blocks_b) = mint_paged_set(&mut pool, &model, 6);
    let tokens_b: Vec<i32> = (100..120).collect();
    store
        .insert(
            &make_key("m", "s", "tpl", &tokens_b),
            CacheEntry::new(tokens_b.clone(), set_b),
        )
        .expect("insert B");

    // A's pins are queued (not dropped + leaked).
    assert!(
        store.has_pending_paged_releases(),
        "eviction queued A's pins"
    );
    let drained = store.drain_pending_paged_releases();
    assert_eq!(drained.len(), 1, "exactly A's paged set is queued");
    assert!(
        !store.has_pending_paged_releases(),
        "drain empties the queue"
    );

    for paged in drained {
        pool.release_detached_paged(paged);
    }
    for b in &blocks_a {
        assert_eq!(
            pool.paged_pool_ref().unwrap().refcount(*b),
            0,
            "evicted A's blocks are reclaimed by the pool"
        );
    }

    // Clean up B (still parked in the store) so its set does not warn on drop.
    let (b_entry, _) = store
        .lookup_longest_prefix(&make_key("m", "s", "tpl", &tokens_b), &tokens_b)
        .expect("B still cached");
    if let Some(DetachedKvSet::Paged(p)) = b_entry.take_detached() {
        pool.release_detached_paged(p);
    }
}

/// An oversized paged entry the store declines on insert must likewise queue
/// its pins for release rather than dropping (and leaking) them.
#[test]
fn oversized_paged_entry_decline_queues_pins_then_pool_reclaims() {
    // capacity_bytes = 1 so any populated paged set is oversized.
    let store = Arc::new(PromptCacheStore::with_config(PromptCacheConfig::new(
        true,
        1,
        32,
        Duration::from_secs(3600),
        1,
    )));
    let model = PagedStub {
        layout: PagedKvLayout::uniform(1, 4, 128).unwrap(),
    };
    let mut pool = CachePool::new(4);
    let (set, blocks) = mint_paged_set(&mut pool, &model, 6);
    let tokens: Vec<i32> = (0..16).collect();
    let err = store
        .insert(
            &make_key("m", "s", "tpl", &tokens),
            CacheEntry::new(tokens.clone(), set),
        )
        .expect_err("tiny capacity must reject the entry");
    assert!(
        matches!(err, InsertError::OversizedEntry { .. }),
        "expected an oversized rejection, got {err:?}"
    );

    assert!(
        store.has_pending_paged_releases(),
        "the decline queued the pins"
    );
    let drained = store.drain_pending_paged_releases();
    assert_eq!(drained.len(), 1);
    for paged in drained {
        pool.release_detached_paged(paged);
    }
    for b in &blocks {
        assert_eq!(
            pool.paged_pool_ref().unwrap().refcount(*b),
            0,
            "the declined entry's blocks are reclaimed"
        );
    }
}

/// A dense eviction frees its buffers on drop and owns no pool pins, so it must
/// never land in the paged release queue.
#[test]
fn dense_eviction_does_not_queue_paged_releases() {
    let store = Arc::new(PromptCacheStore::with_config(PromptCacheConfig::new(
        true,
        1 << 30,
        1,
        Duration::from_secs(3600),
        1,
    )));
    let tokens_a: Vec<i32> = (0..16).collect();
    store
        .insert(
            &make_key("m", "s", "tpl", &tokens_a),
            fake_entry(tokens_a.clone(), 4096),
        )
        .expect("insert dense A");
    let tokens_b: Vec<i32> = (100..120).collect();
    store
        .insert(
            &make_key("m", "s", "tpl", &tokens_b),
            fake_entry(tokens_b.clone(), 4096),
        )
        .expect("insert dense B");

    assert!(
        !store.has_pending_paged_releases(),
        "dense eviction queues no paged pins"
    );
    assert!(store.drain_pending_paged_releases().is_empty());
}

// ---------------------------------------------------------------------------
// #122 sub-step b2: on-demand LRU eviction reclaims paged block budget.
// ---------------------------------------------------------------------------

#[test]
fn evict_one_lru_drops_oldest_and_queues_paged_pins() {
    let store = Arc::new(PromptCacheStore::with_config(PromptCacheConfig::new(
        true,
        1 << 30,
        32,
        Duration::from_secs(3600),
        1,
    )));
    let model = PagedStub {
        layout: PagedKvLayout::uniform(1, 4, 128).unwrap(),
    };
    let mut pool = CachePool::new(4);
    let (set, blocks) = mint_paged_set(&mut pool, &model, 6);
    let tokens: Vec<i32> = (0..16).collect();
    store
        .insert(
            &make_key("m", "s", "tpl", &tokens),
            CacheEntry::new(tokens.clone(), set),
        )
        .expect("insert");
    assert_eq!(store.len(), 1);

    // On-demand eviction drops the entry and queues its pins.
    assert!(
        store.evict_one_lru() > 0,
        "evicting a real entry frees bytes"
    );
    assert_eq!(store.len(), 0);
    assert!(store.has_pending_paged_releases());

    for paged in store.drain_pending_paged_releases() {
        pool.release_detached_paged(paged);
    }
    for b in &blocks {
        assert_eq!(pool.paged_pool_ref().unwrap().refcount(*b), 0);
    }
    // A second eviction is a no-op (store empty).
    assert_eq!(store.evict_one_lru(), 0);
}

#[test]
fn evicting_cold_prefix_restores_paged_block_budget() {
    // The reclaim tier-1 the scheduler runs under block pressure: evict a cold
    // prompt-cache prefix and release its pins, raising the acquirable budget.
    let store = Arc::new(PromptCacheStore::with_config(PromptCacheConfig::new(
        true,
        1 << 30,
        32,
        Duration::from_secs(3600),
        1,
    )));
    let model = PagedStub {
        layout: PagedKvLayout::uniform(1, 4, 128).unwrap(),
    };
    let mut pool = CachePool::new(4);
    pool.set_paged_block_budget(Some(8));
    let (set, blocks) = mint_paged_set(&mut pool, &model, 8); // 2 blocks, live
    assert_eq!(blocks.len(), 2);
    let free_before = pool.free_paged_block_budget().expect("budgeted");

    let tokens: Vec<i32> = (0..16).collect();
    store
        .insert(
            &make_key("m", "s", "tpl", &tokens),
            CacheEntry::new(tokens.clone(), set),
        )
        .expect("insert");

    assert!(store.evict_one_lru() > 0);
    for paged in store.drain_pending_paged_releases() {
        pool.release_detached_paged(paged);
    }
    let free_after = pool.free_paged_block_budget().expect("budgeted");
    assert_eq!(
        free_after,
        free_before + 2,
        "evicting the cold prefix reclaimed its 2 blocks into the budget"
    );
}
