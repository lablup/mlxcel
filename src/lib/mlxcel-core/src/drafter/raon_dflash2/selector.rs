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

//! The DFlash2 candidate selector.
//!
//! ```text
//! cand[s]       = top_k(logits[s])                                  [S, K]
//! pred_id[s, p] = bonus (s = 0) | cand[s - 1, p] (s > 0)
//! scores[s,p,c] = logits[s, cand[s, c]]
//!               + < predecessor[pred_id[s, p]] * hidden_projection(h[s]),
//!                   successor[cand[s, c]] >
//! ```
//!
//! The draft is the greedy chain: step 0 reads row 0 (every row of step 0
//! holds the bonus as predecessor), picks the best candidate `c`, and step
//! `s + 1` reads row `c`, whose predecessor is the candidate just taken.

use crate::dtype;
use crate::ffi::{self, MlxArray};
use crate::layers::UnifiedLinear;
use crate::weights::WeightMap;
use cxx::UniquePtr;

pub struct CandidateSelector {
    hidden_projection: UnifiedLinear,
    /// `[vocab, rank]`.
    predecessor: UniquePtr<MlxArray>,
    /// `[vocab, rank]`.
    successor: UniquePtr<MlxArray>,
    top_k: i32,
}

/// Selector scores and candidates of one block.
pub struct Selection {
    /// `[S, K, K]` float32: step, predecessor row, candidate.
    pub scores: UniquePtr<MlxArray>,
    /// `[S, K]` int32 candidate token ids.
    pub candidates: UniquePtr<MlxArray>,
}

impl CandidateSelector {
    /// # Errors
    /// Returns a message naming a missing tensor.
    pub fn from_weights(weights: &WeightMap, top_k: usize) -> Result<Self, String> {
        let get = |key: &str| {
            weights
                .get(key)
                .map(|w| ffi::copy(w))
                .ok_or_else(|| format!("Raon DFlash2: missing weight {key}"))
        };
        Ok(Self {
            hidden_projection: UnifiedLinear::from_weights(
                weights,
                "candidate_selector.hidden_projection",
                64,
                4,
            )?,
            predecessor: get("candidate_selector.predecessor_codebook")?,
            successor: get("candidate_selector.successor_codebook")?,
            top_k: top_k as i32,
        })
    }

    /// Scores for steps with hidden rows `h` (`[S, hidden]`) and float32
    /// logits `[S, vocab]`; `bonus` is the `[1]` int32 token before step 0.
    pub fn select(&self, h: &MlxArray, logits: &MlxArray, bonus: &MlxArray) -> Selection {
        let s = ffi::array_shape(logits)[0];
        let k = self.top_k;
        let order = ffi::argpartition(&ffi::negative(logits), k - 1, -1);
        let vocab = ffi::array_shape(&order)[1];
        let cand = ffi::slice(&order, &[0, 0], &[s, k.min(vocab)]);
        let cand = ffi::astype(&cand, dtype::INT32);
        let unary = ffi::take_along_axis(logits, &cand, -1); // [S, K]
        let hid = self.hidden_projection.forward(h); // [S, r]
        let succ = ffi::take(&self.successor, &cand, 0); // [S, K, r]
        let first = ffi::broadcast_to(&ffi::reshape(bonus, &[1, 1]), &[1, k]);
        let pred_ids = if s > 1 {
            let prev = ffi::slice(&cand, &[0, 0], &[s - 1, k]);
            crate::ops::concatenate(&first, &prev, 0)
        } else {
            ffi::copy(&first)
        };
        let pred = ffi::take(&self.predecessor, &pred_ids, 0); // [S, K, r]
        let gated = ffi::astype(
            &ffi::multiply(&pred, &ffi::expand_dims(&hid, 1)),
            dtype::FLOAT32,
        );
        let succ_t = ffi::transpose_axes(&ffi::astype(&succ, dtype::FLOAT32), &[0, 2, 1]);
        let pair = ffi::matmul(&gated, &succ_t); // [S, P, C]
        let scores = ffi::add(&ffi::expand_dims(&unary, 1), &pair);
        Selection {
            scores,
            candidates: cand,
        }
    }
}

/// The greedy chain over `selection` as a lazy `[S]` int32 array: no host
/// read, so the draft can feed the verify graph directly.
pub fn chain_device(selection: &Selection) -> UniquePtr<MlxArray> {
    let shape = ffi::array_shape(&selection.scores);
    let (steps, k) = (shape[0], shape[2]);
    let mut prev = ffi::from_slice_i32(&[0], &[1]);
    let mut out = Vec::with_capacity(steps as usize);
    for s in 0..steps {
        let step = ffi::reshape(
            &ffi::slice(&selection.scores, &[s, 0, 0], &[s + 1, k, k]),
            &[k, k],
        );
        let row = ffi::take(&step, &prev, 0); // [1, K]
        let c = ffi::astype(&ffi::argmax(&row, -1, false), dtype::INT32); // [1]
        let cands = ffi::reshape(
            &ffi::slice(&selection.candidates, &[s, 0], &[s + 1, k]),
            &[k],
        );
        out.push(ffi::take(&cands, &c, 0));
        prev = c;
    }
    let refs: Vec<&MlxArray> = out.iter().map(|t| &**t).collect();
    crate::ops::concatenate_many(&refs, 0)
}

/// The same chain over host copies (`scores` row-major `[S, K, K]`,
/// `candidates` `[S, K]`). Ties take the lowest candidate index, as
/// `argmax` does.
pub fn chain_host(scores: &[f32], candidates: &[i32], steps: usize, k: usize) -> Vec<i32> {
    let mut prev = 0usize;
    let mut out = Vec::with_capacity(steps);
    for s in 0..steps {
        let row = &scores[(s * k + prev) * k..(s * k + prev + 1) * k];
        let mut best = 0usize;
        for (c, &v) in row.iter().enumerate() {
            if v > row[best] {
                best = c;
            }
        }
        out.push(candidates[s * k + best]);
        prev = best;
    }
    out
}
