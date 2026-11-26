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

//! Sampled acceptance for a deterministic draft (issue #3436).
//!
//! With a draft `x_i` that is a function of the context alone, the
//! speculative rule "accept `x_i` with probability `p_i(x_i)`, and on
//! rejection sample from `p_i` with `x_i` removed" is the same as: sample
//! `y_i ~ p_i` with the sampler plain decode uses, accept while `y_i == x_i`,
//! and emit `y_i` at the first mismatch (or after the last draft). The
//! emitted sequence then has exactly the distribution of plain sampled
//! decode, because every `y_i` is drawn from the distribution plain decode
//! would use after the same prefix: `p_i` is the verify row's logits, equal
//! to the decode step's (the target's exactness gate), and the history the
//! sampler's penalties read is the plain-decode history with the accepted
//! drafts appended.
//!
//! The walk itself is unchanged ([`super::round_loop::speculative_walk`]):
//! only the per-position target tokens come from the sampler instead of the
//! argmax. Rows past the first mismatch are not sampled.

use crate::drafter::Drafter;
use crate::drafter::dflash::drafter::sampler_is_greedy;
use crate::ffi::{self, MlxArray};
use crate::generate::SamplingConfig;
use cxx::UniquePtr;

/// Whether the round loop samples the verify rows for `drafter` under
/// `sampler`: the drafter must declare a deterministic draft, and the
/// request must sample, read history (penalties, DRY) or bias tokens, which
/// the argmax walk would ignore. A stateful sampler (Mirostat, adaptive-p)
/// never takes this path; the burst declines it before the loop starts.
pub(crate) fn uses_sampled_verify(drafter: &dyn Drafter, sampler: &SamplingConfig) -> bool {
    drafter.supports_sampled_verify()
        && sampled_verify_serves(sampler)
        && (!sampler_is_greedy(sampler)
            || sampler.needs_token_history()
            || !sampler.token_bias.is_empty())
}

/// Whether a drafter that samples its verify rows can serve `sampler`:
/// everything but a feedback sampler (Mirostat's `mu`, adaptive-p's EMA),
/// whose state each row would have to carry from the previous accepted row.
/// The rows are sampled statelessly, so such a sampler would restart from
/// its initial state on every row and leave plain sampling's distribution.
pub fn sampled_verify_serves(sampler: &SamplingConfig) -> bool {
    !sampler.needs_sampler_feedback_state()
}

/// The target tokens of one verify block under sampling: row `i` of
/// `logits` (`[1, K, vocab]`) is sampled with `history` followed by
/// `draft[..i]`, and sampling stops at the first row whose token differs
/// from its draft (that token is the round's bonus). The tokens have
/// `accepted + 1` entries, which is what [`super::round_loop::speculative_walk`]
/// reads; the second value is the sampler-adjusted logits of those rows
/// (`[1, accepted + 1, vocab]`), the logits plain decode reports
/// log-probabilities from.
pub(crate) fn sample_verify_rows(
    logits: &MlxArray,
    draft: &[i32],
    sampler: &SamplingConfig,
    history: &[i32],
) -> (Vec<i32>, UniquePtr<MlxArray>) {
    let shape = ffi::array_shape(logits);
    let rows = shape[1] as usize;
    let needs_history = sampler.needs_token_history();
    let mut row_history: Vec<i32> = if needs_history {
        history.to_vec()
    } else {
        Vec::new()
    };
    let mut out = Vec::with_capacity(rows);
    let mut adjusted_rows = Vec::with_capacity(rows);
    for i in 0..rows {
        let row = ffi::slice(
            logits,
            &[0, i as i32, 0],
            &[shape[0], i as i32 + 1, shape[2]],
        );
        let (token, adjusted) =
            crate::sampling::sample_token_optimized(&row, sampler, &row_history);
        adjusted_rows.push(ffi::reshape(&adjusted, &[1, 1, shape[2]]));
        ffi::eval(&token);
        let y = ffi::item_i32(&token);
        out.push(y);
        match draft.get(i) {
            Some(&x) if x == y => {
                if needs_history {
                    row_history.push(x);
                }
            }
            _ => break,
        }
    }
    let refs: Vec<&MlxArray> = adjusted_rows.iter().map(|r| &**r).collect();
    (out, crate::ops::concatenate_many(&refs, 1))
}

#[cfg(test)]
#[path = "sampled_verify_tests.rs"]
mod tests;
