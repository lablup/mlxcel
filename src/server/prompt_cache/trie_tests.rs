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

use super::*;

fn digest(byte: u8) -> PromptCacheKeyDigest {
    PromptCacheKeyDigest([byte; 32])
}

fn collect_all(trie: &RadixTrie, tokens: &[i32], min_len: usize) -> Vec<PromptCacheKeyDigest> {
    let mut out = Vec::new();
    if let Some(m) = trie.find_longest_prefix(tokens, min_len) {
        m.for_each_candidate(|d| out.push(d.digest));
    }
    out.sort_by_key(|d| d.0);
    out
}

#[test]
fn insert_and_exact_lookup() {
    let mut t = RadixTrie::new();
    t.insert(&[1, 2, 3, 4], digest(1));
    let m = t.find_longest_prefix(&[1, 2, 3, 4], 0).expect("hit");
    assert_eq!(m.depth(), 4);
    let mut found: Vec<_> = Vec::new();
    m.for_each_candidate(|d| found.push(d.digest));
    assert_eq!(found, vec![digest(1)]);
}

#[test]
fn stored_longer_than_requested() {
    // Stored: [1,2,3,4,5]. Request: [1,2,3].
    let mut t = RadixTrie::new();
    t.insert(&[1, 2, 3, 4, 5], digest(1));
    let m = t.find_longest_prefix(&[1, 2, 3], 0).expect("hit");
    assert_eq!(m.depth(), 3);
    let mut found: Vec<_> = Vec::new();
    m.for_each_candidate(|d| found.push(d));
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].digest, digest(1));
    assert_eq!(found[0].token_len, 5);
}

#[test]
fn stored_shorter_than_requested() {
    // Stored: [1,2,3]. Request: [1,2,3,4,5].
    let mut t = RadixTrie::new();
    t.insert(&[1, 2, 3], digest(1));
    let m = t.find_longest_prefix(&[1, 2, 3, 4, 5], 0).expect("hit");
    assert_eq!(m.depth(), 3);
}

#[test]
fn divergence_past_common_prefix() {
    // Stored: [1,2,3,9]. Request: [1,2,3,5].
    let mut t = RadixTrie::new();
    t.insert(&[1, 2, 3, 9], digest(1));
    let m = t.find_longest_prefix(&[1, 2, 3, 5], 0).expect("hit");
    assert_eq!(m.depth(), 3);
}

#[test]
fn min_len_threshold_filters_short_matches() {
    let mut t = RadixTrie::new();
    t.insert(&[1, 2, 3, 4], digest(1));
    // Request diverges after 2 tokens; min_len=3 rejects the match.
    assert!(t.find_longest_prefix(&[1, 2, 9, 9], 3).is_none());
    // Request diverges after 3 tokens; min_len=3 accepts.
    let m = t.find_longest_prefix(&[1, 2, 3, 9], 3).expect("hit");
    assert_eq!(m.depth(), 3);
}

#[test]
fn multiple_entries_under_common_prefix() {
    // Two entries: [1,2,3,10] and [1,2,3,20]. Request [1,2,3,99] matches
    // only up to depth 3 and both entries sit in the subtree.
    let mut t = RadixTrie::new();
    t.insert(&[1, 2, 3, 10], digest(1));
    t.insert(&[1, 2, 3, 20], digest(2));
    let m = t.find_longest_prefix(&[1, 2, 3, 99], 0).expect("hit");
    assert_eq!(m.depth(), 3);
    let mut found: Vec<_> = Vec::new();
    m.for_each_candidate(|d| found.push(d.digest));
    found.sort_by_key(|d| d.0);
    assert_eq!(found, vec![digest(1), digest(2)]);
}

#[test]
fn entries_at_various_depths() {
    // Entry at depth 3 (stored prefix [1,2,3]) and one at depth 6 on the
    // same path. Request [1,2,3,4,5,6] should match the deeper one.
    let mut t = RadixTrie::new();
    t.insert(&[1, 2, 3], digest(1));
    t.insert(&[1, 2, 3, 4, 5, 6], digest(2));
    let m = t.find_longest_prefix(&[1, 2, 3, 4, 5, 6], 0).expect("hit");
    assert_eq!(m.depth(), 6);
    let mut found: Vec<_> = Vec::new();
    m.for_each_candidate(|d| found.push(d.digest));
    assert_eq!(found, vec![digest(2)]);

    // Request that diverges at depth 3 returns digest(1) (depth-3 entry)
    // plus digest(2) whose subtree contains ‘1,2,3,...’.
    let m = t.find_longest_prefix(&[1, 2, 3, 99], 0).expect("hit");
    assert_eq!(m.depth(), 3);
    let mut found: Vec<_> = Vec::new();
    m.for_each_candidate(|d| found.push(d.digest));
    found.sort_by_key(|d| d.0);
    assert_eq!(found, vec![digest(1), digest(2)]);
}

#[test]
fn remove_deletes_digest_and_prunes() {
    let mut t = RadixTrie::new();
    t.insert(&[1, 2, 3], digest(1));
    t.insert(&[1, 2, 4], digest(2));
    assert_eq!(t.len(), 2);
    assert!(t.remove(&[1, 2, 3], digest(1)));
    assert_eq!(t.len(), 1);
    assert!(
        collect_all(&t, &[1, 2, 3], 0)
            .iter()
            .all(|d| *d != digest(1))
    );
    assert!(t.remove(&[1, 2, 4], digest(2)));
    assert_eq!(t.len(), 0);
    assert!(t.find_longest_prefix(&[1, 2, 3], 0).is_none());
}

#[test]
fn remove_absent_digest_is_noop() {
    let mut t = RadixTrie::new();
    t.insert(&[1, 2, 3], digest(1));
    assert!(!t.remove(&[1, 2, 3], digest(9)));
    assert_eq!(t.len(), 1);
}

#[test]
fn empty_trie_returns_none() {
    let t = RadixTrie::new();
    assert!(t.find_longest_prefix(&[1, 2, 3], 0).is_none());
}

#[test]
fn idempotent_insert_does_not_double_count() {
    let mut t = RadixTrie::new();
    t.insert(&[1, 2, 3], digest(1));
    t.insert(&[1, 2, 3], digest(1));
    assert_eq!(t.len(), 1);
}

#[test]
fn split_then_insert_adds_branch() {
    // Insert A=[1,2,3,4]. Then B=[1,2,5]. Edge [1,2,3,4] must split
    // into shared [1,2] and two children: [3,4] (A) and [5] (B).
    let mut t = RadixTrie::new();
    t.insert(&[1, 2, 3, 4], digest(1));
    t.insert(&[1, 2, 5], digest(2));
    assert_eq!(t.len(), 2);
    let m = t.find_longest_prefix(&[1, 2, 3, 4], 0).expect("hit A");
    assert_eq!(m.depth(), 4);
    let m = t.find_longest_prefix(&[1, 2, 5], 0).expect("hit B");
    assert_eq!(m.depth(), 3);
    let m = t.find_longest_prefix(&[1, 2, 7], 0).expect("prefix hit");
    assert_eq!(m.depth(), 2);
}

#[test]
fn deep_insert_scales_linearly_in_prefix_len() {
    // 32k tokens — insert, lookup, remove should each complete without
    // pathological slowdown.
    let tokens: Vec<i32> = (0..32_768).collect();
    let mut t = RadixTrie::new();
    t.insert(&tokens, digest(1));
    let m = t.find_longest_prefix(&tokens, 0).expect("hit");
    assert_eq!(m.depth(), tokens.len());
    assert!(t.remove(&tokens, digest(1)));
    assert_eq!(t.len(), 0);
}

// ---------------------------------------------------------------------------
// pop_prefixes tests — ported from upstream mlx-lm PR #1078 regression suite
// ---------------------------------------------------------------------------

#[test]
fn pop_prefixes_empty_tokens_is_noop() {
    // pop_prefixes([]) with entries present: nothing should be removed
    // because there are no strict-prefix positions for the empty sequence.
    let mut t = RadixTrie::new();
    t.insert(&[], digest(1));
    let removed = t.pop_prefixes(&[]);
    assert!(
        removed.is_empty(),
        "pop_prefixes([]) must not drain anything"
    );
    assert_eq!(t.len(), 1, "root entry must remain after noop");
}

#[test]
fn pop_prefixes_no_matching_prefix_entries() {
    // Trie has [1,2,3]. pop_prefixes([1,2,3,4]) should return nothing because
    // there are no entries at strict-prefix depths (0, 1, 2, 3) — only at
    // depth 3, which is the exact-match depth for [1,2,3] in the INNER sense.
    // Wait: [1,2,3] stored at depth 3; tokens=[1,2,3,4] has exact-match at 4.
    // Depth 3 < 4, so [1,2,3] IS a strict prefix and should be collected.
    let mut t = RadixTrie::new();
    t.insert(&[1, 2, 3], digest(1));
    let removed = t.pop_prefixes(&[1, 2, 3, 4]);
    // [1,2,3] is a strict prefix of [1,2,3,4] — it should be removed.
    assert_eq!(removed.len(), 1);
    assert_eq!(removed[0].digest, digest(1));
    assert_eq!(t.len(), 0, "prefix entry must be gone");
    // [1,2,3,4] path lookup should now miss.
    assert!(t.find_longest_prefix(&[1, 2, 3, 4], 0).is_none());
}

#[test]
fn pop_prefixes_exact_match_is_not_removed() {
    // Upstream regression: pop_prefixes([1,2,3]) on a trie that has exactly
    // [1,2,3] should leave the entry intact — the exact-match node must NOT
    // be drained.
    let mut t = RadixTrie::new();
    t.insert(&[1, 2, 3], digest(1));
    let removed = t.pop_prefixes(&[1, 2, 3]);
    assert!(removed.is_empty(), "exact-match must not be drained");
    assert_eq!(t.len(), 1, "entry must remain");
}

#[test]
fn pop_prefixes_immediate_prefix_is_removed() {
    // Core regression from upstream PR #1078: sequence A = [1, 2] is an
    // IMMEDIATE prefix of B = [1, 2, 3].  pop_prefixes([1,2,3]) must collect
    // A's entry — the old off-by-one `range(len-1)` stopped one step short
    // and left A alive, which leaked stale cache state.
    let mut t = RadixTrie::new();
    t.insert(&[1, 2], digest(1)); // A — immediate prefix of B
    t.insert(&[1, 2, 3], digest(2)); // B — the longer entry
    assert_eq!(t.len(), 2);

    let removed = t.pop_prefixes(&[1, 2, 3]);

    // A must be evicted (immediate prefix at depth 2, strict prefix of [1,2,3]).
    assert_eq!(removed.len(), 1, "immediate prefix must be collected");
    assert_eq!(
        removed[0].digest,
        digest(1),
        "digest(1) is the prefix entry"
    );
    assert_eq!(removed[0].token_len, 2);

    // B (the exact-match entry) must NOT be removed.
    assert_eq!(t.len(), 1, "exact-match entry must survive");
    let m = t
        .find_longest_prefix(&[1, 2, 3], 0)
        .expect("B must still be reachable");
    let mut found = Vec::new();
    m.for_each_candidate(|d| found.push(d.digest));
    assert!(
        found.contains(&digest(2)),
        "digest(2) must still be in trie"
    );
    assert!(!found.contains(&digest(1)), "digest(1) must be gone");
}

#[test]
fn pop_prefixes_deeper_prefix_also_removed() {
    // Three entries: [1] (depth 1), [1,2] (depth 2), [1,2,3] (depth 3).
    // pop_prefixes([1,2,3]) must collect [1] and [1,2] but leave [1,2,3].
    let mut t = RadixTrie::new();
    t.insert(&[1], digest(1));
    t.insert(&[1, 2], digest(2));
    t.insert(&[1, 2, 3], digest(3));
    assert_eq!(t.len(), 3);

    let removed = t.pop_prefixes(&[1, 2, 3]);

    let mut removed_digests: Vec<_> = removed.iter().map(|d| d.digest).collect();
    removed_digests.sort_by_key(|d| d.0);
    assert_eq!(removed_digests, vec![digest(1), digest(2)]);
    assert_eq!(t.len(), 1);

    // [1,2,3] must survive.
    let m = t
        .find_longest_prefix(&[1, 2, 3], 0)
        .expect("exact match must survive");
    let mut found = Vec::new();
    m.for_each_candidate(|d| found.push(d.digest));
    assert!(found.contains(&digest(3)));
}

#[test]
fn pop_prefixes_sibling_branch_untouched() {
    // A=[1,2] and C=[1,3] share root→[1] prefix. Inserting B=[1,2,3] and
    // calling pop_prefixes([1,2,3]) should remove A but leave C intact.
    let mut t = RadixTrie::new();
    t.insert(&[1, 2], digest(1)); // A — immediate prefix of B
    t.insert(&[1, 3], digest(2)); // C — sibling branch
    t.insert(&[1, 2, 3], digest(3)); // B — the longer entry
    assert_eq!(t.len(), 3);

    let removed = t.pop_prefixes(&[1, 2, 3]);

    assert_eq!(removed.len(), 1);
    assert_eq!(removed[0].digest, digest(1));

    // C must still be reachable.
    assert_eq!(t.len(), 2);
    let m = t
        .find_longest_prefix(&[1, 3], 0)
        .expect("sibling C must survive");
    let mut found = Vec::new();
    m.for_each_candidate(|d| found.push(d.digest));
    assert!(found.contains(&digest(2)));
}

#[test]
fn pop_prefixes_subtree_count_consistent_after_drain() {
    // Verify that subtree_count on every node is consistent after pop_prefixes.
    let mut t = RadixTrie::new();
    t.insert(&[1, 2], digest(1));
    t.insert(&[1, 2, 3], digest(2));
    t.pop_prefixes(&[1, 2, 3]);
    // Only digest(2) should remain.
    assert_eq!(t.len(), 1);
    // The trie's internal count must match: find_longest_prefix uses
    // subtree_count as a short-circuit. A wrong count could cause a miss.
    let m = t
        .find_longest_prefix(&[1, 2, 3], 0)
        .expect("must be findable");
    assert_eq!(m.depth(), 3);
}

// The O(N²) adversarial build at N=4096 is far too slow under Miri; the smaller
// structural trie tests already give full UB coverage of the iterative unsafe
// paths, so skip this one when running under Miri.
#[test]
#[cfg_attr(miri, ignore)]
fn pop_prefixes_deep_branching_chain_does_not_overflow() {
    // Regression guard for the stack-overflow DoS. An adversarial
    // pattern `[t₀,X₀], [t₀,t₁,X₁], [t₀,t₁,t₂,X₂], …` defeats path compression
    // because each sibling branch is unique, so the trie builds one node per
    // token along the chain. On such input *every* public trie operation must
    // be iterative — a recursive insert, remove, longest-prefix candidate walk
    // (DFS), drain, or Drop would overflow Tokio's ~2 MiB worker stack.
    //
    // Earlier only `pop_prefixes` and `Drop` were iterative while `insert_into`,
    // `remove_from`, and `dfs` still recursed; the recursive *insert* used to
    // overflow even the test's old 32 MiB build thread in the full debug build,
    // aborting the whole `cargo test` process. So we now build AND exercise the
    // entire trie — insert, longest-prefix lookup + candidate walk, remove,
    // pop_prefixes, and the final Drop — directly on a 2 MiB stack matching a
    // Tokio worker. If any path is still recursive it overflows and aborts the
    // process, so the spawned thread's join() returns Err and the test fails.
    //
    // N is chosen well above the ~1 k-frame depth that overflows a 2 MiB stack
    // with recursive code (≈4× margin), while keeping the O(N²) adversarial
    // build fast.
    const N: i32 = 4_096;

    let result = std::thread::Builder::new()
        .stack_size(2 * 1024 * 1024) // 2 MiB — matches Tokio worker default
        .spawn(move || {
            // insert: build the adversarial one-node-per-token trie.
            let mut t = RadixTrie::new();
            for d in 1..=N {
                let chain: Vec<i32> = (0..d).collect();
                t.insert(&chain, digest((d & 0xff) as u8));
                // Sibling at this depth: diverges at the last token so path
                // compression cannot collapse the chain into fewer nodes.
                let mut sibling = chain.clone();
                *sibling.last_mut().unwrap() = -(d + N); // guaranteed distinct
                t.insert(&sibling, digest(((d.wrapping_neg()) & 0xff) as u8));
            }
            assert_eq!(t.len(), (2 * N) as usize, "two entries per depth");

            let full_chain: Vec<i32> = (0..N).collect();

            // lookup + for_each_candidate: a short prefix surfaces a subtree
            // spanning the full depth, exercising the iterative DFS walk.
            let m = t
                .find_longest_prefix(&[0, 1], 0)
                .expect("short prefix must hit the deep subtree");
            let mut visited = 0usize;
            m.for_each_candidate(|_| visited += 1);
            assert!(visited > 0, "DFS must visit candidates");

            // remove: deleting the deepest chain descends the full depth.
            assert!(
                t.remove(&full_chain, digest((N & 0xff) as u8)),
                "deepest chain must be removable"
            );

            // pop_prefixes: drain every strict-prefix entry along the chain.
            let drained = t.pop_prefixes(&full_chain);
            assert!(!drained.is_empty(), "must have drained strict prefixes");

            // Drop of `t` at end of scope drains the deep trie iteratively.
        })
        .expect("spawn failed")
        .join();
    result.expect("a deep-trie operation overflowed the 2 MiB Tokio-worker-equivalent stack");
}
