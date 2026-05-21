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

//! Per-bucket path-compressed radix trie over token-id prefixes.
//!
//! # Why a radix trie?
//!
//! `PromptCacheStore::lookup_longest_prefix` needs to find, among all cached
//! entries that share the same `(model_id, lora_id, template_sig)` bucket, the
//! stored entry whose token prefix forms the longest common prefix with the
//! incoming request's token vector.
//!
//! Two reasonable data structures were considered:
//!
//! 1. **Hash-of-prefixes with sorted prefix lengths.** Precompute hashes of the
//!    request at a few prefix lengths (say, every power of two up to the
//!    request length) and probe a hash map for each. Each probe costs a
//!    full `O(L)` token hash, and a hit still requires an `O(L)` linear
//!    compare to validate against a non-cryptographic collision. Total work is
//!    `O(L log L)` in the number of tokens, plus global rehash on every
//!    insert/delete. The hot path is dominated by hashing, not structure
//!    traversal, but the structure itself has no sharing — every inserted
//!    prefix consumes `O(L)` bytes in its own hash-entry value.
//!
//! 2. **Per-bucket path-compressed radix trie (this module).** Lookup cost is
//!    `O(L)` in the request length, traversing at most one trie node per
//!    divergent token. Insertion shares edges with existing entries that have
//!    a common prefix, which is the common case for cross-request prompt
//!    caching (large shared system prompts, tool preambles, long document
//!    contexts). Deletion is a bounded rewrite — remove a digest at a node
//!    and, if the node becomes empty and has a single child, merge the edge
//!    back. In-memory overhead per distinct-suffix node is one `HashMap<i32,
//!    Box<Node>>` plus the edge `Vec<i32>`.
//!
//! We chose the radix trie because:
//!
//! * It turns the **common case** (many requests sharing a long system
//!   prompt) into cheap pointer chasing rather than redundant hashing. The
//!   hot-path work scales with *divergence depth*, not with the absolute
//!   token count.
//! * Lookup is deterministically `O(L)` for any realistic prompt (≤ 32k
//!   tokens). That bound holds regardless of how many cached entries live in
//!   the same bucket, which is the property the epic requires.
//! * Inserts/deletes are `O(L)` and don't trigger global rebuilds.
//! * It composes cleanly with the two-tier lookup (exact-session + same
//!   model/lora/template fallback): both tiers run against the same trie
//!   and filter by `session_key` during digest selection.
//!
//! The downside — per-node `HashMap<i32, Box<Node>>` — is tolerable because
//! typical branching factors at any one node are very small (most tokens
//! along a shared prompt don't branch), and path compression collapses long
//! linear runs into a single edge label.
//!
//! # Layout
//!
//! Each node stores:
//!
//! * `edge` — the run of tokens on the incoming edge (path compression).
//! * `entries` — digests whose stored prefix ends exactly at this node's
//!   depth.
//! * `subtree_count` — total `entries` across this node and all descendants,
//!   used to short-circuit "empty subtree" cases during lookup.
//! * `children` — hash map keyed by the first token of each child edge.
//!
//! # Invariants
//!
//! * The root node has an empty `edge` and `entries` never contains its own
//!   digest (the root represents the empty prefix; real entries must have
//!   length >= 1 to be stored at all).
//! * Every non-root node has `edge.len() >= 1`.
//! * `children` is keyed by `edge[0]` of the child; this is what makes
//!   lookup deterministic.
//! * A node with empty `entries` and empty `children` is a leaf candidate
//!   for pruning; `remove` unconditionally prunes.
//! * A node with empty `entries` and exactly one child is a candidate for
//!   edge merging on `remove`.

use std::collections::HashMap;

use super::key::PromptCacheKeyDigest;

/// A single digest paired with the stored token length it corresponds to.
///
/// The token length matters for the caller's matched-length bookkeeping: a
/// lookup may surface a digest whose stored prefix is *longer* than the
/// traversal depth (it lives deeper in the trie), in which case the matched
/// length against the incoming request is clamped at the traversal depth.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct DigestAndLen {
    pub digest: PromptCacheKeyDigest,
    pub token_len: usize,
}

/// Path-compressed radix trie node keyed on `i32` token ids.
#[derive(Debug, Default)]
pub(super) struct TrieNode {
    /// The run of tokens labeling the edge that leads into this node from
    /// its parent. Empty only for the synthetic root.
    edge: Vec<i32>,
    /// Digests whose stored token vector ends exactly at this node's depth.
    /// Plain `Vec` because bucket cardinality is tiny and we care about
    /// cache locality on scan more than on point-contains.
    entries: Vec<DigestAndLen>,
    /// `entries.len()` summed across this node and all descendants. Used as
    /// a cheap "is subtree empty" check.
    subtree_count: usize,
    /// Children keyed by the first token of the child's `edge`.
    children: HashMap<i32, Box<TrieNode>>,
}

impl Drop for TrieNode {
    fn drop(&mut self) {
        // Avoid recursive `Box<TrieNode>` drop on adversarially deep prompt
        // tries. `pop_prefixes` itself is iterative, but after a deep trie is
        // drained (or simply falls out of scope) Rust's default destructor
        // would still recurse through `children` one node at a time and can
        // overflow Tokio's ~2 MiB worker stacks. Drain descendants onto an
        // explicit heap stack first; each boxed node then drops with an empty
        // child map.
        let mut stack: Vec<Box<TrieNode>> = self.children.drain().map(|(_, child)| child).collect();
        while let Some(mut node) = stack.pop() {
            stack.extend(node.children.drain().map(|(_, child)| child));
        }
    }
}

/// Root of the per-bucket radix trie.
#[derive(Debug, Default)]
pub(super) struct RadixTrie {
    root: TrieNode,
}

impl RadixTrie {
    /// Test-only constructor kept alongside `Default` for symmetry with
    /// the data-type's constructor family and readable unit tests.
    #[cfg(test)]
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// Total number of digests stored across this trie.
    pub(super) fn len(&self) -> usize {
        self.root.subtree_count
    }

    /// Insert `digest` at the position described by `tokens`. Returns the
    /// depth at which the digest was installed (always equal to
    /// `tokens.len()`).
    ///
    /// Safe to call repeatedly with the same `(digest, tokens)`; the digest
    /// is stored at most once per node.
    pub(super) fn insert(&mut self, tokens: &[i32], digest: PromptCacheKeyDigest) {
        let record = DigestAndLen {
            digest,
            token_len: tokens.len(),
        };
        insert_into(&mut self.root, tokens, record);
    }

    /// Remove the given `digest` from the trie. No-op if the digest isn't
    /// present (callers must therefore remember the insertion tokens).
    /// Returns `true` if the digest was found and removed.
    pub(super) fn remove(&mut self, tokens: &[i32], digest: PromptCacheKeyDigest) -> bool {
        remove_from(&mut self.root, tokens, digest)
    }

    /// Find the longest token-prefix of `tokens` that reaches a position
    /// whose subtree contains at least one stored digest. Returns the
    /// matched depth and a reference to the subtree root whose entries all
    /// share the matched prefix with `tokens`.
    ///
    /// Returns `None` when no entry in the trie matches a prefix of at
    /// least `min_len` tokens. The returned depth is always `>= min_len`
    /// and always `<= tokens.len()`.
    pub(super) fn find_longest_prefix(
        &self,
        tokens: &[i32],
        min_len: usize,
    ) -> Option<TrieMatch<'_>> {
        let m = walk_longest(&self.root, tokens)?;
        if m.depth < min_len {
            return None;
        }
        Some(m)
    }

    /// Remove and return all entries stored at strict-prefix depths along
    /// `tokens`. "Strict prefix" means depth in `[0, tokens.len())` — the
    /// exact-match node at depth `tokens.len()` is **not** touched.
    ///
    /// This mirrors Python's `PromptTrie.pop_prefixes` (upstream PR #1078).
    /// The upstream fix changed the loop from `range(len(tokens) - 1)` to
    /// `enumerate(tokens)`, extending the walk so the IMMEDIATE-prefix node
    /// (depth `tokens.len() - 1`) is also collected. The old code stopped
    /// one step short, leaking that entry and causing stale state to surface
    /// on subsequent requests with overlapping prefixes.
    ///
    /// In the path-compressed radix trie nodes live at cumulative edge-length
    /// boundaries, not at individual token positions, so we walk iteratively
    /// through edge spans and drain every node whose depth D < `tokens.len()`.
    ///
    /// # Implementation: two-pass iterative (stack-overflow safe)
    ///
    /// A recursive implementation stack-overflows on Tokio workers (2 MB stack)
    /// when an adversarial token chain `[t₀,X₀], [t₀,t₁,X₁], …` defeats path
    /// compression and creates one trie node per token (~10–20k tokens suffice).
    ///
    /// **Pass 1 — descent + drain:** Walk iteratively via `&mut self` re-binding.
    /// At each strict-prefix node, drain `entries` into the result vec.  Track
    /// per-level drain counts in a `Vec<usize>`.
    ///
    /// **Pass 2 — subtree_count repair:** Compute suffix sums over the per-level
    /// drain counts.  Walk the same path top-down, subtracting `suffix_sum[level]`
    /// from each node's `subtree_count`.
    ///
    /// Returns the drained entries in traversal order.
    // Used by: PromptCacheStore (store.rs) — wired up when the store gains
    // trimmable-cache awareness. Suppressed until then to keep clippy clean.
    #[allow(dead_code)]
    pub(super) fn pop_prefixes(&mut self, tokens: &[i32]) -> Vec<DigestAndLen> {
        if tokens.is_empty() {
            return Vec::new();
        }

        // ------------------------------------------------------------------ //
        // Pass 1: iterative descent, draining strict-prefix nodes.           //
        //                                                                     //
        // We track how many entries we drain at each level so Pass 2 can     //
        // repair `subtree_count` without a second recursive walk.             //
        //                                                                     //
        // `drained_per_level[i]` = number of entries drained from the node   //
        // at path position i (0 = root).                                      //
        // `edge_lengths[i]` = edge length that was consumed to reach level i  //
        // from level i-1 (0 for root because root has no incoming edge).      //
        // ------------------------------------------------------------------ //
        let mut collected: Vec<DigestAndLen> = Vec::new();
        let mut drained_per_level: Vec<usize> = Vec::new();
        let mut edge_lengths: Vec<usize> = Vec::new(); // edge consumed to reach each level

        {
            let mut remaining: &[i32] = tokens;
            // SAFETY: we use raw pointers to navigate &mut TrieNode without
            // triggering the borrow-checker's "aliased mutable reference" rule.
            // At any point only one mutable reference (`node`) is live; we never
            // alias it with another live mutable borrow.
            let mut node: *mut TrieNode = &mut self.root;

            loop {
                // SAFETY: `node` always points to a valid TrieNode owned by
                // `self`, reachable via the path we just walked.
                let node_ref = unsafe { &mut *node };

                // Drain strict-prefix node (remaining non-empty means we have
                // more tokens to match, so this node is at depth < tokens.len()).
                let n = if !remaining.is_empty() {
                    let n = node_ref.entries.len();
                    if n > 0 {
                        collected.append(&mut node_ref.entries);
                    }
                    n
                } else {
                    // `remaining` is empty: this is the exact-match node — do NOT drain.
                    0
                };
                drained_per_level.push(n);

                if remaining.is_empty() {
                    // Reached the exact-match depth; stop.
                    break;
                }

                let next_tok = remaining[0];
                // Validate the next edge: it must exist and its label must
                // match the front of `remaining`.
                let edge_len = match node_ref.children.get(&next_tok) {
                    Some(child) => {
                        let elen = child.edge.len();
                        if remaining.len() < elen || remaining[..elen] != child.edge[..] {
                            // Edge label doesn't match — walk ends here.
                            break;
                        }
                        elen
                    }
                    None => break,
                };

                // Edge matches: advance.
                remaining = &remaining[edge_len..];
                edge_lengths.push(edge_len);
                node = node_ref
                    .children
                    .get_mut(&next_tok)
                    .expect("path validated above")
                    .as_mut() as *mut TrieNode;
            }
        }

        // Nothing was drained — bail early.
        if drained_per_level.iter().all(|&d| d == 0) {
            return collected;
        }

        // ------------------------------------------------------------------ //
        // Pass 2: repair subtree_count along the same path.                  //
        //                                                                     //
        // suffix_sum[i] = total entries drained at levels i..total.          //
        // Node at level i must have its subtree_count reduced by             //
        // suffix_sum[i] (it lost all entries drained at or below its level). //
        // ------------------------------------------------------------------ //
        let total = drained_per_level.len();
        let mut suffix_sum = vec![0usize; total + 1];
        for i in (0..total).rev() {
            suffix_sum[i] = suffix_sum[i + 1] + drained_per_level[i];
        }

        // Walk the path again top-down, updating subtree_count.
        {
            let mut remaining: &[i32] = tokens;
            let mut node: *mut TrieNode = &mut self.root;

            for (level, &edge_len) in std::iter::once(&0usize)
                .chain(edge_lengths.iter())
                .enumerate()
            {
                // SAFETY: same invariant as Pass 1 — single live mutable ref.
                let node_ref = unsafe { &mut *node };
                node_ref.subtree_count = node_ref.subtree_count.saturating_sub(suffix_sum[level]);

                if level + 1 == total {
                    break;
                }

                // Advance: skip `edge_len` tokens consumed for THIS level's
                // incoming edge (0 for root), then look up the next child.
                if level > 0 {
                    remaining = &remaining[edge_len..];
                }
                let next_tok = remaining[0];
                node = node_ref
                    .children
                    .get_mut(&next_tok)
                    .expect("path persists from Pass 1")
                    .as_mut() as *mut TrieNode;
            }
        }

        collected
    }
}

/// Successful longest-prefix walk result.
///
/// `depth` — number of leading request tokens that successfully matched
/// somewhere in the trie. This is the matched length for any entry under
/// the returned `subtree` (capped at the stored entry's own length).
///
/// `subtree` — the trie node at or beyond the matched depth whose entries
/// all share the matched prefix. When the walk stopped exactly at a node
/// boundary this is that node itself; when it stopped inside an edge
/// (path-compressed), it is the child node the edge points to, because
/// any entry under that child still has the matched prefix as its own
/// prefix.
pub(super) struct TrieMatch<'a> {
    subtree: &'a TrieNode,
    /// When the walk ended inside an edge, this records how many tokens of
    /// the traversed edge have **not** been matched: those belong to the
    /// `subtree` node's own edge (all entries in the subtree still have
    /// those tokens, so they're past the matched depth). When the walk
    /// ended exactly at a node boundary, this is 0.
    consumed_from_subtree_edge: usize,
    depth: usize,
}

impl<'a> TrieMatch<'a> {
    /// Number of request tokens that matched (i.e. the traversal depth).
    pub(super) fn depth(&self) -> usize {
        self.depth
    }

    /// Walk the matched subtree and invoke `visit` with every stored digest.
    /// Traversal order is DFS, but callers must not depend on any particular
    /// ordering.
    pub(super) fn for_each_candidate<F: FnMut(DigestAndLen)>(&self, mut visit: F) {
        // When we stopped mid-edge, `subtree`'s own `entries` are at a depth
        // *greater* than the matched depth (they extend past the match
        // point). They still share the matched prefix with the request, so
        // they count — their match length is clamped at the matched depth.
        // `self.consumed_from_subtree_edge > 0` indicates this mid-edge
        // case; entries on this node are still valid candidates either way,
        // so we traverse the whole subtree rooted here.
        let _ = self.consumed_from_subtree_edge;
        dfs(self.subtree, &mut visit);
    }
}

// Iterative subtree walk. A recursive DFS would overflow Tokio's ~2 MiB worker
// stacks on an adversarially deep prompt trie (one node per token; see the
// `pop_prefixes` and `Drop` notes), and this runs on the per-request lookup hot
// path via `for_each_candidate`. Use an explicit heap stack instead. Traversal
// order is unspecified, matching the documented contract on `for_each_candidate`.
fn dfs<F: FnMut(DigestAndLen)>(node: &TrieNode, visit: &mut F) {
    let mut stack: Vec<&TrieNode> = vec![node];
    while let Some(n) = stack.pop() {
        for d in &n.entries {
            visit(*d);
        }
        for child in n.children.values() {
            stack.push(child);
        }
    }
}

fn walk_longest<'a>(root: &'a TrieNode, tokens: &[i32]) -> Option<TrieMatch<'a>> {
    if root.subtree_count == 0 {
        return None;
    }
    let mut node: &TrieNode = root;
    let mut consumed = 0usize;
    // Walk until we reach a point where no child can extend the match any
    // further OR we've consumed all request tokens.
    loop {
        if consumed >= tokens.len() {
            // Request exhausted exactly at a node boundary.
            break;
        }
        let next_tok = tokens[consumed];
        let child = match node.children.get(&next_tok) {
            Some(c) => c.as_ref(),
            None => break,
        };
        let edge = child.edge.as_slice();
        let remaining = &tokens[consumed..];
        let matchable = edge.len().min(remaining.len());
        let mut matched = 0usize;
        while matched < matchable && edge[matched] == remaining[matched] {
            matched += 1;
        }
        if matched == 0 {
            // Defensive: should be unreachable since the children map is
            // keyed by `edge[0]` and we only descended because `edge[0] ==
            // next_tok`.
            break;
        }
        consumed += matched;
        if matched < edge.len() {
            // Walk stopped mid-edge. The subtree rooted at `child` has
            // every entry sharing the first `consumed` request tokens; the
            // tokens past the match point are shared across the whole
            // subtree (they're part of `child.edge`).
            return Some(TrieMatch {
                subtree: child,
                consumed_from_subtree_edge: matched,
                depth: consumed,
            });
        }
        // Whole child edge consumed — descend.
        node = child;
    }
    if consumed == 0 {
        // We didn't advance past root; treat as a miss.
        return None;
    }
    Some(TrieMatch {
        subtree: node,
        consumed_from_subtree_edge: 0,
        depth: consumed,
    })
}

// Iterative insert. The former recursion descended one frame per trie node and
// overflowed Tokio's ~2 MiB worker stacks on an adversarially deep prompt trie
// (one node per token), and this runs on the per-request cache-store hot path
// via `PromptCacheStore::insert`. Instead, descend iteratively while recording
// the visited path, perform the single structural mutation (append / split /
// fresh leaf) at the terminal node, then repair `subtree_count` bottom-up over
// the recorded path — exactly mirroring the post-order unwinding of the old
// recursion. Stack usage is O(1); the path lives on the heap.
fn insert_into(root: &mut TrieNode, tokens: &[i32], record: DigestAndLen) {
    let mut path: Vec<*mut TrieNode> = Vec::new();
    let mut node: *mut TrieNode = root;
    let mut tokens: &[i32] = tokens;

    loop {
        path.push(node);
        // SAFETY: `node` points to a valid TrieNode owned by `root`'s tree,
        // reachable via the path we just walked. Only one mutable reference is
        // live at a time; we never alias `node_ref` with another live borrow.
        let node_ref = unsafe { &mut *node };

        if tokens.is_empty() {
            // Terminal: store the digest here, replacing in place to keep
            // token_len accurate if the same digest is re-inserted.
            if let Some(existing) = node_ref
                .entries
                .iter_mut()
                .find(|e| e.digest == record.digest)
            {
                *existing = record;
            } else {
                node_ref.entries.push(record);
            }
            break;
        }

        let first_tok = tokens[0];
        if let Some(child) = node_ref.children.get_mut(&first_tok) {
            let edge_clone = child.edge.clone();
            let common = common_prefix_len(&edge_clone, tokens);
            if common == edge_clone.len() {
                // Full edge consumed: descend into the child.
                tokens = &tokens[common..];
                node = child.as_mut() as *mut TrieNode;
                continue;
            }
            // Edge diverges mid-way: split (terminal — no further descent).
            // Create a new intermediate node that keeps the first `common`
            // tokens of the old edge as its own incoming edge. Move the old
            // child under the new intermediate with its edge truncated to
            // what's left.
            let shared: Vec<i32> = edge_clone[..common].to_vec();
            let old_remainder: Vec<i32> = edge_clone[common..].to_vec();

            // Rebuild the existing child with its shorter edge.
            let mut old_child = node_ref
                .children
                .remove(&first_tok)
                .expect("child just seen");
            let old_first_after_split = old_remainder[0];
            old_child.edge = old_remainder;

            // New intermediate holds no entries yet.
            let mut intermediate = TrieNode {
                edge: shared,
                entries: Vec::new(),
                subtree_count: 0,
                children: HashMap::new(),
            };
            intermediate
                .children
                .insert(old_first_after_split, old_child);

            if tokens.len() == common {
                // New record ends exactly at the split point.
                intermediate.entries.push(record);
            } else {
                // New record continues past the split; add a fresh child for
                // the remaining tokens.
                let new_remainder: Vec<i32> = tokens[common..].to_vec();
                let new_first = new_remainder[0];
                let new_child = TrieNode {
                    edge: new_remainder,
                    entries: vec![record],
                    subtree_count: 1,
                    children: HashMap::new(),
                };
                intermediate.children.insert(new_first, Box::new(new_child));
            }
            // The intermediate (and any fresh child) get correct counts here;
            // ancestors on `path` are repaired in the bottom-up pass below.
            intermediate.subtree_count = recompute_subtree_count(&intermediate);
            node_ref.children.insert(first_tok, Box::new(intermediate));
            break;
        }

        // No child for this first token yet: create a fresh leaf with the full
        // remaining tokens as its edge.
        let leaf = TrieNode {
            edge: tokens.to_vec(),
            entries: vec![record],
            subtree_count: 1,
            children: HashMap::new(),
        };
        node_ref.children.insert(first_tok, Box::new(leaf));
        break;
    }

    // Repair `subtree_count` bottom-up along the recorded path. Each node's
    // children counts are already correct by the time we reach it (deepest
    // first), so a local recompute restores the invariant up to the root.
    for &n in path.iter().rev() {
        // SAFETY: each recorded path node is still live and owned by `root`'s
        // tree; the pointers are distinct and visited one at a time.
        let n_ref = unsafe { &mut *n };
        n_ref.subtree_count = recompute_subtree_count(n_ref);
    }
}

// Iterative remove. Like `insert_into`, the former recursion descended one frame
// per trie node and could overflow a ~2 MiB Tokio worker stack on an
// adversarially deep prompt trie (this runs on the cache-eviction path via
// `PromptCacheStore`). Descend iteratively, recording `(node, descend-key)` for
// each ancestor; on the way back up prune any child that became an empty leaf,
// merge a now-empty single-child non-root node, and repair `subtree_count` —
// mirroring the post-order unwinding of the old recursion. Stack usage is O(1).
fn remove_from(root: &mut TrieNode, tokens: &[i32], digest: PromptCacheKeyDigest) -> bool {
    // Capture the root identity, then navigate exclusively through raw pointers
    // below; the `root` reference itself is not touched again, so nothing aliases
    // the `&mut` we reborrow from each pointer as we walk.
    let root_ptr: *const TrieNode = root;
    // `path[i] = (ancestor, key)` where `key` is the child token we descended
    // through from `ancestor`. The terminal node (where the digest lives) is
    // not recorded here; its `entries`/`subtree_count` are handled inline below.
    let mut path: Vec<(*mut TrieNode, i32)> = Vec::new();
    let mut node: *mut TrieNode = root;
    let mut tokens: &[i32] = tokens;

    loop {
        // SAFETY: `node` points to a live TrieNode owned by `root`'s tree along
        // the validated path; only one mutable reference is live at a time.
        let node_ref = unsafe { &mut *node };
        if tokens.is_empty() {
            let before = node_ref.entries.len();
            node_ref.entries.retain(|e| e.digest != digest);
            if node_ref.entries.len() == before {
                // Digest absent — nothing changed anywhere, so no path repair.
                return false;
            }
            node_ref.subtree_count = recompute_subtree_count(node_ref);
            break;
        }

        let first_tok = tokens[0];
        let edge_len = match node_ref.children.get(&first_tok) {
            Some(child) => {
                let elen = child.edge.len();
                if tokens.len() < elen || tokens[..elen] != child.edge[..] {
                    // Edge label doesn't match — digest can't be present.
                    return false;
                }
                elen
            }
            None => return false,
        };
        path.push((node, first_tok));
        tokens = &tokens[edge_len..];
        node = node_ref
            .children
            .get_mut(&first_tok)
            .expect("edge validated above")
            .as_mut() as *mut TrieNode;
    }

    // Walk back up the recorded ancestors (deepest first). For each: drop the
    // child we descended through if it became an empty leaf, then compress this
    // node if it is now an empty single-child non-root node, then repair count.
    for &(pnode, key) in path.iter().rev() {
        // SAFETY: each ancestor is still live and owned by `root`'s tree; the
        // pointers are distinct and visited one at a time.
        let pref = unsafe { &mut *pnode };
        let child_is_empty_leaf = match pref.children.get(&key) {
            Some(child) => child.entries.is_empty() && child.children.is_empty(),
            None => false,
        };
        if child_is_empty_leaf {
            pref.children.remove(&key);
        }
        if !std::ptr::eq(pnode as *const TrieNode, root_ptr) {
            merge_only_child_if_empty(pref);
        }
        pref.subtree_count = recompute_subtree_count(pref);
    }
    true
}

fn merge_only_child_if_empty(node: &mut TrieNode) {
    if !node.entries.is_empty() || node.children.len() != 1 {
        return;
    }
    // Take the single child.
    let only_key = *node.children.keys().next().expect("single child");
    let mut child = node.children.remove(&only_key).expect("single child");
    // Fuse child edge onto `node.edge`.
    let mut fused = std::mem::take(&mut node.edge);
    fused.extend_from_slice(&child.edge);
    node.edge = fused;
    node.entries = std::mem::take(&mut child.entries);
    node.children = std::mem::take(&mut child.children);
    node.subtree_count = recompute_subtree_count(node);
}

fn recompute_subtree_count(node: &TrieNode) -> usize {
    let mut total = node.entries.len();
    for c in node.children.values() {
        total += c.subtree_count;
    }
    total
}

fn common_prefix_len(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
#[path = "trie_tests.rs"]
mod tests;
