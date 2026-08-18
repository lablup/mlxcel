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

//! Topology, mask and walk properties for [`DraftTree`].
//!
//! The load-bearing tests are the two equivalences: a linear tree must walk
//! exactly as the chain path walks and mask exactly as a causal mask masks.
//! Everything the tree path subsumes has to keep behaving identically, or
//! turning it on changes output on configurations that were already correct.

use super::super::walk::speculative_walk;
use super::{DraftTree, TreeWalk};

const MASKED: f32 = -1.0e9;

fn causal_mask(n: usize) -> Vec<f32> {
    let mut mask = vec![MASKED; n * n];
    for q in 0..n {
        for k in 0..=q {
            mask[q * n + k] = 0.0;
        }
    }
    mask
}

#[test]
fn a_linear_tree_masks_exactly_as_a_causal_mask() {
    for k in 0..6 {
        let drafts: Vec<i32> = (0..k).map(|i| 100 + i).collect();
        let tree = DraftTree::linear(7, &drafts);
        assert_eq!(
            tree.additive_mask(MASKED),
            causal_mask(k as usize + 1),
            "linear tree of {k} drafts must be indistinguishable from causal"
        );
    }
}

#[test]
fn a_linear_tree_walks_exactly_as_the_chain_walk() {
    // Every accept length from none to all, plus the full-accept case where
    // the target's last choice becomes the next round's bonus.
    let drafts = vec![11, 12, 13];
    for accepted in 0..=drafts.len() {
        // The target agrees for `accepted` positions and disagrees after,
        // and its choice at the stopping position is the next bonus.
        let mut target_tokens: Vec<i32> = (0..drafts.len())
            .map(|i| if i < accepted { drafts[i] } else { 999 })
            .collect();
        target_tokens.push(1000);

        let tree = DraftTree::linear(7, &drafts);
        let tree_walk = tree.walk(&target_tokens, usize::MAX);
        let chain_walk = speculative_walk(&drafts, &target_tokens, usize::MAX);

        assert_eq!(
            tree_walk.accepted, chain_walk.accepted,
            "accept count must match the chain walk at accepted={accepted}"
        );
        assert_eq!(
            tree_walk.new_tokens, chain_walk.new_tokens,
            "emitted tokens must match the chain walk at accepted={accepted}"
        );
        assert_eq!(
            tree_walk.path,
            (0..=tree_walk.accepted).collect::<Vec<_>>(),
            "a linear accept set is a prefix, which is what makes trim_to correct"
        );
    }
}

#[test]
fn siblings_cannot_see_each_other() {
    // root -> {a, b}; a -> c
    let mut tree = DraftTree::root(1);
    let a = tree.push_child(0, 10);
    let b = tree.push_child(0, 20);
    let c = tree.push_child(a, 30);
    let n = tree.len();
    let mask = tree.additive_mask(MASKED);
    let visible = |q: usize, k: usize| mask[q * n + k] == 0.0;

    assert!(visible(a, 0) && visible(a, a), "a sees the root and itself");
    assert!(!visible(a, b), "a must not see its sibling");
    assert!(!visible(b, a), "b must not see its sibling");
    assert!(
        visible(c, 0) && visible(c, a) && visible(c, c),
        "c sees its whole ancestry"
    );
    assert!(!visible(c, b), "c must not see the other branch");
    assert!(!visible(0, a), "the root sees nothing below it");
}

#[test]
fn the_walk_descends_the_branch_the_target_agreed_with() {
    // root -> {a=10, b=20}; b -> d=40. The target picks b, then d.
    let mut tree = DraftTree::root(1);
    let _a = tree.push_child(0, 10);
    let b = tree.push_child(0, 20);
    let d = tree.push_child(b, 40);

    // target[root] = 20 picks b; target[b] = 40 picks d; target[d] = 50 stops.
    let mut target = vec![0; tree.len()];
    target[0] = 20;
    target[b] = 40;
    target[d] = 50;

    let walk = tree.walk(&target, usize::MAX);
    assert_eq!(
        walk,
        TreeWalk {
            accepted: 2,
            path: vec![0, b, d],
            new_tokens: vec![20, 40, 50],
        }
    );
}

#[test]
fn a_branch_the_target_rejected_contributes_nothing() {
    // Same tree, but the target disagrees with both children of the root.
    let mut tree = DraftTree::root(1);
    let _a = tree.push_child(0, 10);
    let b = tree.push_child(0, 20);
    let d = tree.push_child(b, 40);

    let mut target = vec![0; tree.len()];
    target[0] = 77; // matches neither 10 nor 20
    target[b] = 40;
    target[d] = 50;

    let walk = tree.walk(&target, usize::MAX);
    assert_eq!(walk.accepted, 0);
    assert_eq!(walk.path, vec![0]);
    assert_eq!(
        walk.new_tokens,
        vec![77],
        "a fully rejected round still emits the target's own choice"
    );
}

#[test]
fn the_accepted_set_is_a_path_even_when_two_branches_could_match() {
    // Two children carrying the SAME token. The target emits one argmax per
    // position, so at most one of them can be taken, and the walk must not
    // fan out. This is what keeps rollback a path-compaction problem rather
    // than a subtree-compaction one.
    let mut tree = DraftTree::root(1);
    let first = tree.push_child(0, 10);
    let second = tree.push_child(0, 10);
    tree.push_child(first, 30);
    tree.push_child(second, 40);

    let mut target = vec![0; tree.len()];
    target[0] = 10;
    target[first] = 30;

    let walk = tree.walk(&target, usize::MAX);
    assert_eq!(walk.accepted, 2);
    assert_eq!(
        walk.path,
        vec![0, first, first + 2],
        "insertion order breaks the tie, and the result is still one path"
    );
}

#[test]
fn budget_clamps_the_emission_without_changing_the_path() {
    let drafts = vec![11, 12, 13];
    let tree = DraftTree::linear(7, &drafts);
    let target = vec![11, 12, 13, 14];

    let full = tree.walk(&target, usize::MAX);
    assert_eq!(full.new_tokens, vec![11, 12, 13, 14]);

    let clamped = tree.walk(&target, 2);
    assert_eq!(clamped.new_tokens, vec![11, 12]);
    assert!(
        clamped.new_tokens.len() <= 2,
        "a walk must never emit past the caller's budget"
    );
}

#[test]
fn path_and_ancestry_agree() {
    let mut tree = DraftTree::root(1);
    let a = tree.push_child(0, 10);
    let b = tree.push_child(a, 20);
    let c = tree.push_child(0, 30);

    assert_eq!(tree.path_to(b), vec![0, a, b]);
    assert_eq!(tree.path_to(c), vec![0, c]);
    for node in 0..tree.len() {
        for candidate in 0..tree.len() {
            assert_eq!(
                tree.is_ancestor_or_self(candidate, node),
                tree.path_to(node).contains(&candidate),
                "ancestry and path must not disagree for ({candidate}, {node})"
            );
        }
    }
}

#[test]
#[should_panic(expected = "is not a node")]
fn attaching_under_a_nonexistent_parent_is_rejected() {
    let mut tree = DraftTree::root(1);
    tree.push_child(5, 10);
}

#[test]
fn a_history_mask_opens_the_cache_and_keeps_the_tree_shape() {
    // root -> {a, b}; a -> c, with four tokens already in the KV cache.
    let mut tree = DraftTree::root(1);
    let a = tree.push_child(0, 10);
    let b = tree.push_child(0, 20);
    let c = tree.push_child(a, 30);
    let offset = 4;
    let n = tree.len();
    let total = offset + n;
    let mask = tree.additive_mask_with_history(offset, MASKED);
    let visible = |q: usize, k: usize| mask[q * total + k] == 0.0;

    assert_eq!(mask.len(), n * total);
    for q in 0..n {
        for k in 0..offset {
            assert!(visible(q, k), "node {q} must see cached history column {k}");
        }
    }
    assert!(visible(c, offset + a), "c still sees its ancestor");
    assert!(
        !visible(c, offset + b),
        "c still cannot see the other branch"
    );
    assert!(
        !visible(a, offset + b),
        "siblings stay invisible past the history"
    );
}

#[test]
fn a_history_mask_at_zero_offset_is_the_plain_mask() {
    let mut tree = DraftTree::root(1);
    let a = tree.push_child(0, 10);
    tree.push_child(0, 20);
    tree.push_child(a, 30);
    assert_eq!(
        tree.additive_mask_with_history(0, MASKED),
        tree.additive_mask(MASKED),
        "the history form must degenerate to the plain one"
    );
}

#[test]
fn a_linear_history_mask_matches_a_causal_mask_with_the_same_offset() {
    // The equivalence that lets the tree path replace create_causal_mask on a
    // linear block without changing what the attention sees.
    let offset = 3usize;
    let drafts = vec![11, 12, 13];
    let tree = DraftTree::linear(7, &drafts);
    let n = tree.len();
    let total = offset + n;

    let mut expected = vec![MASKED; n * total];
    for q in 0..n {
        for k in 0..=(q + offset) {
            expected[q * total + k] = 0.0;
        }
    }
    assert_eq!(tree.additive_mask_with_history(offset, MASKED), expected);
}
