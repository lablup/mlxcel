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

//! Draft-tree topology, its attention mask, and the accept walk over it.
//!
//! ## Why a tree
//!
//! A linear draft block spends its whole verify budget on one guess about the
//! future, so each extra position multiplies the chance that the prefix has
//! already diverged. Measured on M5 Max with `qwen3.8-27b-4bit` and the
//! `qwen3_5_mtp` drafter, widening the block from 3 to 12 drops acceptance
//! from 0.65 to 0.17 while emitted-per-verify moves only 2.3 to 2.8: the
//! verify cost rises with the block and the yield does not. Branching spends
//! the same extra positions on alternatives at uncertain steps instead
//! (issue #1204).
//!
//! ## What is here, and what is not
//!
//! Everything in this module is pure: a topology, a mask derived from it, and
//! a walk over target argmaxes. None of it touches a device, a cache, or a
//! model, which is what makes it testable without a checkpoint and reviewable
//! before the plumbing exists.
//!
//! The pieces that are *not* here are the ones with the real risk. Verifying a
//! tree needs the mask below handed to the target's forward instead of the
//! causal mask it builds itself, and rolling back needs the accepted path's KV
//! entries compacted, which today's caches cannot express: `KVCache` offers
//! `trim_to(new_len)` and nothing else, and tail truncation only works because
//! a linear accept set is a prefix. See the module notes on [`TreeWalk::path`]
//! for what a correct rollback has to do.

/// A draft tree in topological order.
///
/// Node 0 is the root: the bonus token the target already committed to this
/// round. Every other node is a proposal whose parent is the context it was
/// drafted from. `parents[i] < i` holds for every `i > 0`, which is what makes
/// the mask lower-triangular and the walk a simple descent.
///
/// A linear draft block is the degenerate case where `parents[i] == i - 1`,
/// and it must behave exactly as the chain path does today. That equivalence
/// is the property the tests pin, because it is what lets the tree path be
/// switched on without changing output on the configurations it subsumes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DraftTree {
    tokens: Vec<i32>,
    parents: Vec<usize>,
}

impl DraftTree {
    /// A tree holding only the root bonus token.
    pub fn root(bonus: i32) -> Self {
        Self {
            tokens: vec![bonus],
            parents: vec![0],
        }
    }

    /// The linear chain `[bonus, draft_0, ..., draft_{k-1}]`.
    ///
    /// The shape the round loop drafts today, expressed as a tree so both
    /// paths can share one walk and one mask builder.
    pub fn linear(bonus: i32, drafts: &[i32]) -> Self {
        let mut tree = Self::root(bonus);
        let mut parent = 0;
        for &token in drafts {
            parent = tree.push_child(parent, token);
        }
        tree
    }

    /// Attach `token` under `parent` and return the new node's index.
    ///
    /// # Panics
    ///
    /// If `parent` is not an existing node. A parent index at or past the
    /// current length would break the `parents[i] < i` invariant that the
    /// mask and the walk both rely on, and silently accepting it would
    /// produce a mask that lets a node attend to its own future.
    pub fn push_child(&mut self, parent: usize, token: i32) -> usize {
        assert!(
            parent < self.tokens.len(),
            "parent {parent} is not a node (tree has {} nodes)",
            self.tokens.len()
        );
        self.tokens.push(token);
        self.parents.push(parent);
        self.tokens.len() - 1
    }

    /// Number of nodes, including the root.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Tokens in node order: the verify input, flattened.
    pub fn tokens(&self) -> &[i32] {
        &self.tokens
    }

    /// The parent of `node`. The root is its own parent.
    pub fn parent(&self, node: usize) -> usize {
        self.parents[node]
    }

    /// Children of `node`, in insertion order.
    ///
    /// Linear in the tree size. Draft trees are tens of nodes at most, so
    /// this stays cheaper than carrying a child list that has to be kept
    /// consistent with `parents`.
    pub fn children(&self, node: usize) -> Vec<usize> {
        (1..self.tokens.len())
            .filter(|&i| self.parents[i] == node)
            .collect()
    }

    /// Root-to-node path, inclusive at both ends.
    pub fn path_to(&self, node: usize) -> Vec<usize> {
        let mut path = vec![node];
        let mut cursor = node;
        while cursor != 0 {
            cursor = self.parents[cursor];
            path.push(cursor);
        }
        path.reverse();
        path
    }

    /// Whether `ancestor` is on the root-to-`node` path, `node` included.
    pub fn is_ancestor_or_self(&self, ancestor: usize, node: usize) -> bool {
        let mut cursor = node;
        loop {
            if cursor == ancestor {
                return true;
            }
            if cursor == 0 {
                return false;
            }
            cursor = self.parents[cursor];
        }
    }

    /// Row-major `[n, n]` additive attention mask.
    ///
    /// `mask[q * n + k]` is `0.0` when node `k` is on the root-to-`q` path and
    /// `masked` otherwise, so a node attends to its ancestors and itself and
    /// to nothing else. Siblings are mutually invisible, which is the whole
    /// point: two branches must not see each other's tokens.
    ///
    /// `masked` is taken rather than hard-coded because the value that means
    /// "masked" is a property of the attention kernel this is handed to, and
    /// picking `f32::NEG_INFINITY` here would produce NaN in a kernel that
    /// adds the mask before a softmax over an all-masked row.
    pub fn additive_mask(&self, masked: f32) -> Vec<f32> {
        let n = self.tokens.len();
        let mut mask = vec![masked; n * n];
        for q in 0..n {
            for k in self.path_to(q) {
                mask[q * n + k] = 0.0;
            }
        }
        mask
    }

    /// Descend the branch the target agreed with.
    ///
    /// `target_tokens[i]` is the target's greedy choice conditional on the
    /// context ending at node `i`, which is the same contract
    /// [`super::walk::speculative_walk`] uses: one logit per verify position.
    ///
    /// Starting at the root, a child whose token equals the target's choice at
    /// the current node is accepted and becomes the new current node. The walk
    /// stops at the first node where no child matches, and the target's choice
    /// there is the bonus that seeds the next round. A node with several
    /// children can match at most one of them, since the target emits a single
    /// argmax per position, so the accepted set is always a path and never a
    /// subtree.
    ///
    /// `budget` caps the emitted tokens the way the linear walk does, so a
    /// generation does not run past its declared `max_tokens`.
    pub fn walk(&self, target_tokens: &[i32], budget: usize) -> TreeWalk {
        debug_assert_eq!(
            target_tokens.len(),
            self.tokens.len(),
            "the verify pass produces one target choice per tree node"
        );
        debug_assert!(budget >= 1, "the round loop gates on budget before walking");

        let mut path = vec![0usize];
        let mut new_tokens = Vec::new();
        let mut current = 0usize;

        // A short `target_tokens` is a shape bug upstream. Release builds
        // stop rather than index out of bounds; the debug assertion above is
        // what catches it in tests.
        while let Some(&next) = target_tokens.get(current) {
            let matched = self
                .children(current)
                .into_iter()
                .find(|&child| self.tokens[child] == next);
            match matched {
                Some(child) if new_tokens.len() < budget => {
                    new_tokens.push(self.tokens[child]);
                    path.push(child);
                    current = child;
                }
                _ => {
                    if new_tokens.len() < budget {
                        new_tokens.push(next);
                    }
                    break;
                }
            }
        }

        TreeWalk {
            accepted: path.len() - 1,
            path,
            new_tokens,
        }
    }
}

/// Result of [`DraftTree::walk`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeWalk {
    /// Number of accepted draft nodes, excluding the root.
    pub accepted: usize,
    /// Node indices from the root through the last accepted node.
    ///
    /// **This is what makes tree rollback harder than chain rollback.** For a
    /// linear draft the path is `[0, 1, ..., accepted]`, so the surviving KV
    /// entries are a prefix and `trim_to(accepted + 1)` is correct. For a tree
    /// the path is a scattered subset of the verified positions, so the cache
    /// has to keep those entries and compact them, which `KVCache` cannot
    /// express today. The recurrent side is better off than it looks: Qwen 3.5
    /// already rebuilds GatedDeltaNet state by replaying the captured block
    /// over the accepted prefix rather than trimming it, and replaying a
    /// gathered path instead of a prefix is an indexing change, not a new
    /// mechanism.
    pub path: Vec<usize>,
    /// User-visible tokens for this round: the accepted drafts followed by
    /// the target's choice at the stopping node, clamped to budget.
    pub new_tokens: Vec<i32>,
}

#[cfg(test)]
#[path = "tree_tests.rs"]
mod tree_tests;
