// Copyright 2025-2026 Lablup Inc.
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

//! The completion snapshot of a snapshot-only family, decided when the
//! prefill finishes rather than at the end of generation (issue #3040).

use super::*;
use crate::server::prompt_cache::entry::SnapshotOrigin;

impl BatchScheduler {
    /// Decide `seq`'s completion snapshot when its prefill finishes, before
    /// any decode step writes the reply into the state.
    ///
    /// The end-of-generation snapshot this replaces held `prompt + generated`,
    /// and the generated rows were written by single-token decode steps, which
    /// do not write the bytes a prefill of the same tokens writes. Restoring
    /// it reproduced no cold run of the next turn (#3040). Instead:
    ///
    /// * a chat turn with a history boundary stores nothing more: the boundary
    ///   snapshot is where the next turn resumes, and the prompt end is only
    ///   the generation prompt past it, written by a forward of a handful of
    ///   rows that the next turn's cold run never makes;
    /// * any other request (a raw completion, or the boundary disabled) stores
    ///   the state at the prompt end, keyed by the prompt alone.
    ///
    /// Either way the sequence is recorded in `prompt_cache_prompt_end_stored`,
    /// including when the store declines the entry, so
    /// [`Self::donate_finished_sequence_cache`] stores no decode-written tail
    /// at the finish. Paths that never reach the classic `finish_prefill`
    /// (the speculative bursts) are not recorded and keep the
    /// end-of-generation snapshot. `MLXCEL_PROMPT_CACHE_DECODED_TAIL=1` turns
    /// this off ([`decoded_tail_snapshot_enabled`]).
    ///
    /// The prompt-end copy runs after the first token was emitted, so it adds
    /// nothing to TTFT, and it replaces the end-of-generation copy.
    ///
    /// [`decoded_tail_snapshot_enabled`]: crate::server::prompt_cache::decoded_tail_snapshot_enabled
    pub(super) fn insert_prompt_end_snapshot(&mut self, seq: &SequenceInfo) {
        if crate::server::prompt_cache::decoded_tail_snapshot_enabled()
            || !self.prompt_cache_active()
            || !self.engine.model().supports_snapshot_reuse()
        {
            return;
        }
        let Some(has_boundary) = self
            .prompt_cache_seq_ctx
            .get(&seq.seq_id)
            .map(|ctx| ctx.history_prefix_tokens.is_some())
        else {
            return;
        };
        self.prompt_cache_prompt_end_stored.insert(seq.seq_id);
        // The history boundary is the next turn's restore point. The rows past
        // it came from a forward of a few generation-prompt tokens, which runs
        // the small-row kernels; the next turn's cold run forwards those
        // tokens inside its history segment, so a state including them would
        // reproduce no cold run (measured on the live-replies replay, #3040).
        // This holds whether or not this request captured the boundary: a
        // regenerated turn adopts the boundary snapshot an earlier request of
        // the same turn stored and captures nothing, and a prompt-end entry
        // from it would be the next turn's longest match.
        if has_boundary {
            return;
        }
        let Some(ctx) = self.prompt_cache_seq_ctx.get(&seq.seq_id).cloned() else {
            return;
        };
        let _span = tracing::info_span!(
            "prompt_end_snapshot",
            seq_id = %seq.seq_id,
            prompt_len = seq.prompt_tokens.len(),
        )
        .entered();
        let encoded_span = seq.prompt_tokens.len();
        self.insert_model_state_snapshot(
            seq.seq_id,
            &ctx,
            seq.prompt_tokens.clone(),
            SnapshotOrigin::Completion,
            encoded_span,
        );
    }
}
