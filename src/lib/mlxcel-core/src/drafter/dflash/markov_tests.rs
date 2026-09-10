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

//! Tests for the DSpark Markov token-transition head (issue #1339).

use super::*;
use crate::dtype;

const VOCAB: i32 = 16;

/// A head over `VOCAB` tokens at rank `VOCAB`: `w1` is the identity so
/// `w1[t] = e_t`, and `w2` is whatever `[vocab, rank]` matrix the test wants
/// the transition to be, so `bias(prev) = w2[:, prev]`.
fn head_with_transition(w2: Vec<f32>) -> VanillaMarkovHead {
    let mut identity = vec![0.0_f32; (VOCAB * VOCAB) as usize];
    for t in 0..VOCAB as usize {
        identity[t * VOCAB as usize + t] = 1.0;
    }
    let mut weights: WeightMap = std::collections::HashMap::new();
    weights.insert(
        "markov_head.markov_w1.weight".to_string(),
        ffi::from_slice_f32(&identity, &[VOCAB, VOCAB]),
    );
    weights.insert(
        "markov_head.markov_w2.weight".to_string(),
        ffi::from_slice_f32(&w2, &[VOCAB, VOCAB]),
    );
    VanillaMarkovHead::from_weights(&weights, "markov_head", VOCAB as usize, 64, 4)
        .expect("markov head must load from w1 / w2")
}

/// `[1, gamma, VOCAB]` logits that are flat (all zero) at every position.
fn flat_base_logits(gamma: i32) -> UniquePtr<MlxArray> {
    ffi::zeros(&[1, gamma, VOCAB], dtype::FLOAT32)
}

/// `[1, gamma, VOCAB]` logits where position `i` puts a `+5` on token
/// `first + i` and nothing else.
fn position_indexed_base_logits(gamma: i32, first: i32) -> UniquePtr<MlxArray> {
    let mut buf = vec![0.0_f32; (gamma * VOCAB) as usize];
    for i in 0..gamma {
        buf[(i * VOCAB + first + i) as usize] = 5.0;
    }
    ffi::from_slice_f32(&buf, &[1, gamma, VOCAB])
}

#[test]
fn markov_chain_feeds_previous_token_into_next_step() {
    // Transition: `w2[t + 1, t] = 100`, so `bias(prev)` is a large spike on
    // `prev + 1`. With flat base logits the chain must walk `anchor + 1`,
    // `anchor + 2`, ... which is only possible if each step's argmax is fed
    // back as the next step's `prev`.
    let mut w2 = vec![0.0_f32; (VOCAB * VOCAB) as usize];
    for t in 0..VOCAB - 1 {
        w2[((t + 1) * VOCAB + t) as usize] = 100.0;
    }
    let head = head_with_transition(w2);
    assert_eq!(head.rank(), VOCAB as usize);

    let gamma = 5;
    let anchor = 2;
    let proposals = head.sample_block(&flat_base_logits(gamma), anchor);
    assert_eq!(proposals, vec![3, 4, 5, 6, 7]);

    // The device-side variant carries the same ids as a `[1, gamma]` int32
    // row, which is what the round loop concatenates behind the anchor.
    let arr = head.sample_block_array(&flat_base_logits(gamma), anchor);
    assert_eq!(ffi::array_shape(&arr), vec![1, gamma]);
    assert_eq!(ffi::array_dtype(&arr), dtype::INT32);
    assert_eq!(
        super::super::materialize_argmax_i32_vec(&arr, gamma as usize),
        vec![3, 4, 5, 6, 7]
    );
}

#[test]
fn dspark_draft_block_uses_all_positions() {
    // No transition at all: the chain reduces to the per-position argmax of
    // the base logits, and it must consume position 0 as a proposal rather
    // than discard it the way a DFlash draft discards its bonus scaffold.
    let zero_transition = vec![0.0_f32; (VOCAB * VOCAB) as usize];
    let head = head_with_transition(zero_transition);
    let gamma = 5;
    let proposals = head.sample_block(&position_indexed_base_logits(gamma, 10), 3);
    assert_eq!(proposals, vec![10, 11, 12, 13, 14]);

    // Proposal 0 derives from the anchor row: a transition from the anchor
    // (token 3) that outweighs position 0's base spike wins that position,
    // and because the chosen token (1) has no outgoing transition, the
    // remaining positions fall back to their base spikes.
    let mut w2 = vec![0.0_f32; (VOCAB * VOCAB) as usize];
    w2[(VOCAB + 3) as usize] = 1000.0;
    let head = head_with_transition(w2);
    let proposals = head.sample_block(&position_indexed_base_logits(gamma, 10), 3);
    assert_eq!(proposals, vec![1, 11, 12, 13, 14]);
}

#[test]
fn markov_head_load_requires_both_factors() {
    let mut weights: WeightMap = std::collections::HashMap::new();
    weights.insert(
        "markov_head.markov_w1.weight".to_string(),
        ffi::zeros(&[VOCAB, 4], dtype::FLOAT32),
    );
    let Err(err) = VanillaMarkovHead::from_weights(&weights, "markov_head", 4, 64, 4) else {
        panic!("a missing markov_w2 must fail the load");
    };
    assert!(
        err.contains("markov_head.markov_w2"),
        "error must name the missing tensor: {err}"
    );
}
