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

use super::*;

fn shared_kv_budget_allows(live_tokens: usize, additional_tokens: usize, budget: usize) -> bool {
    live_tokens
        .checked_add(additional_tokens)
        .is_some_and(|total| total <= budget)
}

fn shared_kv_live_tokens_from(
    active_batch: &ActiveBatch,
    chunked_prefill_seq: Option<&SequenceInfo>,
) -> usize {
    let active = active_batch
        .iter_sequences()
        .filter(|seq| !seq.state.is_finished())
        .map(BatchScheduler::sequence_live_tokens)
        .sum::<usize>();
    let chunked = chunked_prefill_seq
        .filter(|seq| !seq.state.is_finished())
        .map_or(0, BatchScheduler::sequence_live_tokens);
    active.saturating_add(chunked)
}

#[cfg(test)]
fn shared_kv_budget_allows_growth(
    active_batch: &ActiveBatch,
    chunked_prefill_seq: Option<&SequenceInfo>,
    additional_tokens: usize,
    budget: usize,
) -> bool {
    shared_kv_budget_allows(
        shared_kv_live_tokens_from(active_batch, chunked_prefill_seq),
        additional_tokens,
        budget,
    )
}

impl BatchScheduler {
    /// Configured logical live-token budget shared by all slots when
    /// `--kv-unified` is active. `None` is the legacy split-window mode.
    pub(super) fn shared_kv_budget(&self) -> Option<usize> {
        self.shared_kv_budget
    }

    /// Logical tokens currently owned by a sequence for unified-budget
    /// accounting. This is intentionally independent of paged block rounding
    /// and prompt-cache physical storage: b10621's visible contract is a token
    /// budget, not a page budget.
    pub(super) fn sequence_live_tokens(seq: &SequenceInfo) -> usize {
        seq.prompt_tokens
            .len()
            .saturating_add(seq.generated_tokens.len())
    }

    /// Current logical live-token usage across decode-active and parked
    /// chunked-prefill sequences.
    pub(super) fn shared_kv_live_tokens(&self) -> usize {
        shared_kv_live_tokens_from(&self.active_batch, self.chunked_prefill_seq.as_ref())
    }

    fn evict_prompt_cache_for_shared_budget(&mut self) {
        if let Some(store) = self.prompt_cache.as_ref() {
            store.clear();
        }
        self.drain_store_paged_releases();
    }

    fn shared_budget_has_room_after_evicting_cache(&mut self, additional_tokens: usize) -> bool {
        let Some(budget) = self.shared_kv_budget else {
            return true;
        };
        if shared_kv_budget_allows(self.shared_kv_live_tokens(), additional_tokens, budget) {
            return true;
        }

        self.evict_prompt_cache_for_shared_budget();
        shared_kv_budget_allows(self.shared_kv_live_tokens(), additional_tokens, budget)
    }

    /// Check whether a queued prompt can be admitted under the shared logical
    /// budget. Used both before allocation and again when the queue head is
    /// about to start prefill because active generations may have grown while
    /// the request waited.
    pub(super) fn shared_budget_admits_prompt(&mut self, prompt_tokens: usize) -> bool {
        self.shared_budget_has_room_after_evicting_cache(prompt_tokens)
    }

    /// Check whether a decode step that would append one token for each row
    /// may run without ever exceeding the shared logical budget.
    pub(super) fn shared_budget_has_decode_room(&mut self, rows: usize) -> bool {
        self.shared_budget_has_room_after_evicting_cache(rows)
    }

    /// Check the first sampled token produced by prefill before it is committed
    /// to the sequence. The sequence is not yet in the active batch, so the
    /// prompt plus first token are counted as this tick's additional tokens.
    pub(super) fn shared_budget_has_prefill_first_token_room(
        &mut self,
        prompt_tokens: usize,
    ) -> bool {
        self.shared_budget_has_room_after_evicting_cache(prompt_tokens.saturating_add(1))
    }

    pub(super) fn send_shared_budget_rejection(
        response_tx: &std::sync::mpsc::Sender<GenerateEvent>,
        request_tokens: usize,
        budget: Option<usize>,
    ) {
        let max = budget.unwrap_or(request_tokens);
        let _ = response_tx.send(GenerateEvent::Error(format!(
            "request ({request_tokens} tokens) exceeds the available context size ({max} tokens), \
             try increasing it"
        )));
    }

    /// Deterministic exhaustion policy for unified budget growth: once a decode
    /// tick cannot append all requested rows without crossing the shared budget,
    /// mark every live sequence as length-finished. Finalization sends normal
    /// `Done` responses with b10621's context-exhausted metadata.
    pub(super) fn finish_all_for_shared_kv_budget(&mut self) {
        self.evict_prompt_cache_for_shared_budget();

        for seq in self.active_batch.iter_mut() {
            if seq.state.is_finished() {
                continue;
            }
            seq.retention.context_exhausted = true;
            if let Err(err) = seq
                .state
                .transition_to(SequenceState::Finished(FinishReason::Length))
            {
                tracing::error!("State transition error: {err}");
            }
        }

        if let Some(mut seq) = self.chunked_prefill_seq.take() {
            seq.retention.context_exhausted = true;
            if let Err(err) = seq
                .state
                .transition_to(SequenceState::Finished(FinishReason::Length))
            {
                tracing::error!("State transition error: {err}");
            }
            let cached = seq.already_cached_tokens;
            let result = seq.take_generation_result(&self.tokenizer, cached, None);
            let _ = seq.response_tx.send(GenerateEvent::Done(result));
            self.prompt_cache_seq_ctx.remove(&seq.seq_id);
            self.release_sequence_caches(seq.seq_id);
            self.batch_observability.record_sequence_completed();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::sync::mpsc;
    use std::time::Instant;

    use mlxcel_core::cache::SequenceId;
    use mlxcel_core::generate::SamplingConfig;

    use super::{
        shared_kv_budget_allows, shared_kv_budget_allows_growth, shared_kv_live_tokens_from,
    };
    use crate::server::batch::active::ActiveBatch;
    use crate::server::batch::sequence::{FinishReason, SequenceInfo, SequenceState};
    use crate::server::batch::stop_matcher::StopMatcher;
    use crate::server::model_provider::GenerateEvent;
    use crate::server::model_provider::model_worker::StreamingDecodeState;

    fn make_sequence(
        id: u64,
        state: SequenceState,
        prompt_len: usize,
        generated_len: usize,
    ) -> (SequenceInfo, mpsc::Receiver<GenerateEvent>) {
        let (tx, rx) = mpsc::channel();
        let tokenizer = crate::tokenizer::MlxcelTokenizer::stub();
        let prompt_tokens = vec![1; prompt_len];
        let decode_state = StreamingDecodeState::new(&tokenizer, &prompt_tokens);
        let seq = SequenceInfo {
            bounds: Default::default(),
            retention: Default::default(),
            seq_id: SequenceId::from_raw(id),
            state,
            prompt_tokens,
            sampling: SamplingConfig::default(),
            max_tokens: 100,
            eos_token_ids: vec![2],
            priority: crate::server::batch::RequestPriority::Normal,
            lora_scales: None,
            logprobs_config: Default::default(),
            vlm_embeddings: None,
            images: Vec::new(),
            audio: Vec::new(),
            generated_tokens: vec![3; generated_len],
            generated_text: String::new(),
            decode_state,
            stop_matcher: StopMatcher::default(),
            prefill_offset: 0,
            prefill_start_offset: 0,
            already_cached_tokens: 0,
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
        (seq, rx)
    }

    #[test]
    fn shared_budget_admits_up_to_the_logical_token_limit() {
        assert!(shared_kv_budget_allows(60, 4, 64));
        assert!(!shared_kv_budget_allows(60, 5, 64));
    }

    #[test]
    fn shared_budget_rejects_overflowing_totals() {
        assert!(!shared_kv_budget_allows(usize::MAX, 1, usize::MAX));
    }

    #[test]
    fn shared_budget_counts_active_and_chunked_sequences() {
        let mut active = ActiveBatch::new(4);
        let (mut decoding, _rx) = make_sequence(1, SequenceState::Decoding, 3, 2);
        decoding.retention.context_exhausted = false;
        active.add(decoding).unwrap();
        let (finished, _rx) = make_sequence(2, SequenceState::Finished(FinishReason::Length), 9, 9);
        active.add(finished).unwrap();
        let (chunked, _rx) = make_sequence(3, SequenceState::Prefilling, 4, 1);

        assert_eq!(shared_kv_live_tokens_from(&active, Some(&chunked)), 10);
    }

    #[test]
    fn shared_budget_growth_rejects_when_active_rows_would_cross_limit() {
        let mut active = ActiveBatch::new(4);
        let (seq, _rx) = make_sequence(1, SequenceState::Decoding, 4, 2);
        active.add(seq).unwrap();

        assert!(shared_kv_budget_allows_growth(&active, None, 2, 8));
        assert!(!shared_kv_budget_allows_growth(&active, None, 3, 8));
    }
}
