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

//! The completion snapshot of a snapshot-only family, taken at the prompt
//! boundary when the prefill finishes (issue #3040).

use super::*;
use crate::server::prompt_cache::entry::SnapshotOrigin;

impl BatchScheduler {
    /// Store `seq`'s model state as its completion snapshot, keyed by the
    /// prompt alone, right after the prefill sampled the first token and
    /// before any decode step runs.
    ///
    /// The end-of-generation snapshot it replaces held `prompt + generated`,
    /// and the generated part was written by single-token decode steps, which
    /// do not write the bytes a prefill of the same tokens writes. Restoring
    /// it on the next turn therefore reproduced no cold run of that turn
    /// (#3040). A prompt-boundary state is prefill-written throughout, and the
    /// next turn prefills the reply with its new text, so a hit matches a cold
    /// run split at the same point, as #2974 pinned for chained restores.
    ///
    /// The sequence is recorded in `prompt_cache_prompt_end_stored` whenever
    /// the gates pass, including when the store declines the entry, so
    /// [`Self::donate_finished_sequence_cache`] stores no decode-written tail
    /// for it at the finish. Paths that never reach the classic
    /// `finish_prefill` (the speculative bursts) are not recorded and keep
    /// the end-of-generation snapshot. `MLXCEL_PROMPT_CACHE_DECODED_TAIL=1`
    /// turns this off ([`decoded_tail_snapshot_enabled`]).
    ///
    /// The copy runs after the first token was emitted, so it adds nothing to
    /// TTFT, and it replaces the end-of-generation copy rather than adding one.
    ///
    /// [`decoded_tail_snapshot_enabled`]: crate::server::prompt_cache::decoded_tail_snapshot_enabled
    pub(super) fn insert_prompt_end_snapshot(&mut self, seq: &SequenceInfo) {
        if crate::server::prompt_cache::decoded_tail_snapshot_enabled()
            || !self.prompt_cache_active()
            || !self.engine.model().supports_snapshot_reuse()
        {
            return;
        }
        let Some(ctx) = self.prompt_cache_seq_ctx.get(&seq.seq_id).cloned() else {
            return;
        };
        self.prompt_cache_prompt_end_stored.insert(seq.seq_id);
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
