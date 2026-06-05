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

//! Priority queue for requests awaiting prefill.
//!
//! [`PrefillQueue`] uses three priority lanes (high, normal, low) backed by
//! `VecDeque`s. Within each lane, insertion order is preserved (FIFO).
//! The scheduler drains entries starting from the highest-priority non-empty
//! lane. An optional capacity limit (across all lanes) prevents unbounded
//! memory growth under sustained load.

use std::collections::VecDeque;
use std::sync::atomic::Ordering;

use super::sequence::{RequestPriority, SequenceInfo};

/// Default maximum queue depth when no explicit limit is given.
const DEFAULT_MAX_QUEUE_SIZE: usize = 1024;

/// Priority-aware queue of sequences waiting for prefill.
///
/// Requests are routed into one of three lanes based on their
/// [`RequestPriority`]. The scheduler always dequeues from the
/// highest-priority non-empty lane first, ensuring that high-priority
/// requests are prefilled before lower-priority ones.
///
/// Within each lane, FIFO order is preserved so older requests of the
/// same priority are served first.
///
/// An optional `max_size` bound prevents unbounded memory growth. When the
/// queue is full, `enqueue` returns the rejected `SequenceInfo` (boxed) so
/// the caller can send an appropriate error response.
///
/// **Starvation note:** This queue uses strict priority ordering without
/// aging. Under sustained high-priority load, lower-priority requests
/// may wait indefinitely. Callers should apply server-level timeouts
/// to bound worst-case latency for low-priority requests.
pub struct PrefillQueue {
    high: VecDeque<SequenceInfo>,
    normal: VecDeque<SequenceInfo>,
    low: VecDeque<SequenceInfo>,
    max_size: usize,
}

impl PrefillQueue {
    /// Create an empty prefill queue with the default capacity limit (1024).
    pub fn new() -> Self {
        Self {
            high: VecDeque::new(),
            normal: VecDeque::new(),
            low: VecDeque::new(),
            max_size: DEFAULT_MAX_QUEUE_SIZE,
        }
    }

    /// Create an empty prefill queue with a custom capacity limit.
    pub fn with_capacity(max_size: usize) -> Self {
        Self {
            high: VecDeque::new(),
            normal: VecDeque::new(),
            low: VecDeque::new(),
            max_size,
        }
    }

    /// Push a sequence into the appropriate priority lane.
    ///
    /// Returns `Err(Box<seq>)` if the total queue (across all lanes) is at
    /// capacity, giving the caller ownership back so it can respond with a
    /// "server busy" error.
    pub fn enqueue(&mut self, seq: SequenceInfo) -> Result<(), Box<SequenceInfo>> {
        if self.len() >= self.max_size {
            return Err(Box::new(seq));
        }
        match seq.priority {
            RequestPriority::High => self.high.push_back(seq),
            RequestPriority::Normal => self.normal.push_back(seq),
            RequestPriority::Low => self.low.push_back(seq),
        }
        Ok(())
    }

    /// Pop the highest-priority, oldest sequence from the queue.
    ///
    /// Drains from the high lane first, then normal, then low.
    pub fn dequeue(&mut self) -> Option<SequenceInfo> {
        self.high
            .pop_front()
            .or_else(|| self.normal.pop_front())
            .or_else(|| self.low.pop_front())
    }

    /// Drain up to `max_extra` additional sequences from the front of the
    /// `lane` priority lane for which `predicate` returns `true`, stopping
    /// at the first sequence that does not match.
    ///
    /// used by the speculative-burst scheduler to assemble a
    /// B > 1 batched-burst window. The caller dequeues the head sequence
    /// normally via [`Self::dequeue`], then calls this with the head's
    /// priority lane and a predicate that matches "speculative-eligible
    /// AND same prompt length as the head". Scanning stops at the first
    /// non-match so FIFO order within the lane is preserved for the
    /// sequences left behind — a request that cannot join the current
    /// burst window stays at the front of its lane and is served on the
    /// next tick.
    ///
    /// Only the single `lane` is scanned (not lower-priority lanes): the
    /// head sequence already established the lane, and mixing priorities
    /// into one burst window would let a normal-priority request ride on
    /// a high-priority burst's tick budget.
    pub fn drain_matching_window<F>(
        &mut self,
        lane: RequestPriority,
        max_extra: usize,
        mut predicate: F,
    ) -> Vec<SequenceInfo>
    where
        F: FnMut(&SequenceInfo) -> bool,
    {
        let deque = match lane {
            RequestPriority::High => &mut self.high,
            RequestPriority::Normal => &mut self.normal,
            RequestPriority::Low => &mut self.low,
        };
        let mut collected = Vec::new();
        while collected.len() < max_extra {
            match deque.front() {
                Some(front) if predicate(front) => {
                    // `pop_front` cannot return `None` here — `front()`
                    // just returned `Some`.
                    if let Some(seq) = deque.pop_front() {
                        collected.push(seq);
                    }
                }
                _ => break,
            }
        }
        collected
    }

    /// Peek at the priority of the next sequence that would be dequeued.
    pub fn peek_priority(&self) -> Option<RequestPriority> {
        if !self.high.is_empty() {
            Some(RequestPriority::High)
        } else if !self.normal.is_empty() {
            Some(RequestPriority::Normal)
        } else if !self.low.is_empty() {
            Some(RequestPriority::Low)
        } else {
            None
        }
    }

    /// Total number of sequences across all priority lanes.
    pub fn len(&self) -> usize {
        self.high.len() + self.normal.len() + self.low.len()
    }

    /// Returns `true` when all lanes are empty.
    pub fn is_empty(&self) -> bool {
        self.high.is_empty() && self.normal.is_empty() && self.low.is_empty()
    }

    /// Returns `true` when the total queue has reached its capacity limit.
    pub fn is_full(&self) -> bool {
        self.len() >= self.max_size
    }

    /// Maximum number of entries this queue will hold (across all lanes).
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Remove and return all queued sequences whose cancellation flag is set.
    ///
    /// Scans all priority lanes and removes sequences where
    /// `cancelled.load(Relaxed)` is `true`. The caller is responsible for
    /// cleaning up cache pool entries and notifying the response channel.
    pub fn drain_cancelled(&mut self) -> Vec<SequenceInfo> {
        let mut cancelled = Vec::new();
        Self::drain_cancelled_from_lane(&mut self.high, &mut cancelled);
        Self::drain_cancelled_from_lane(&mut self.normal, &mut cancelled);
        Self::drain_cancelled_from_lane(&mut self.low, &mut cancelled);
        cancelled
    }

    /// Helper: remove cancelled entries from a single lane, preserving order.
    fn drain_cancelled_from_lane(lane: &mut VecDeque<SequenceInfo>, out: &mut Vec<SequenceInfo>) {
        let mut i = 0;
        while i < lane.len() {
            if lane[i].cancelled.load(Ordering::Relaxed) {
                // remove() returns Option but index is valid since i < len.
                if let Some(seq) = lane.remove(i) {
                    out.push(seq);
                }
                // Do not increment i because remove shifts elements left.
            } else {
                i += 1;
            }
        }
    }
}

impl Default for PrefillQueue {
    fn default() -> Self {
        Self::new()
    }
}

// We cannot derive Debug because SequenceInfo contains non-Debug fields.
impl std::fmt::Debug for PrefillQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefillQueue")
            .field("high", &self.high.len())
            .field("normal", &self.normal.len())
            .field("low", &self.low.len())
            .field("max_size", &self.max_size)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::batch::sequence::SequenceState;
    use crate::server::model_provider::GenerateEvent;
    use crate::server::model_provider::model_worker::StreamingDecodeState;
    use mlxcel_core::cache::SequenceId;
    use mlxcel_core::generate::SamplingConfig;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::sync::mpsc;
    use std::time::Instant;

    /// Build a minimal `SequenceInfo` for testing with a given priority.
    fn make_test_sequence_with_priority(
        id_val: u64,
        priority: RequestPriority,
    ) -> (SequenceInfo, mpsc::Receiver<GenerateEvent>) {
        let (tx, rx) = mpsc::channel();
        let tokenizer = crate::tokenizer::MlxcelTokenizer::stub();
        let prompt_tokens = vec![1, 2, 3];
        let decode_state = StreamingDecodeState::new(&tokenizer, &prompt_tokens);

        let seq = SequenceInfo {
            seq_id: SequenceId::from_raw(id_val),
            state: SequenceState::Queued,
            prompt_tokens,
            sampling: SamplingConfig::default(),
            max_tokens: 100,
            eos_token_ids: vec![2],
            priority,
            logprobs_config: Default::default(),
            vlm_embeddings: None,
            images: Vec::new(),
            audio: Vec::new(),
            generated_tokens: Vec::new(),
            generated_text: String::new(),
            decode_state,
            prefill_offset: 0,
            prefill_start_offset: 0,
            already_cached_tokens: 0,
            response_tx: tx,
            cancelled: Arc::new(AtomicBool::new(false)),
            created_at: Instant::now(),
            prefill_start: None,
            first_token_time: None,
            token_history: Vec::new(),
            merged_eos: Vec::new(),
            thinking: crate::server::thinking_budget::ThinkingState::disabled(),
            structured: None,
        };

        (seq, rx)
    }

    fn make_test_sequence(id_val: u64) -> (SequenceInfo, mpsc::Receiver<GenerateEvent>) {
        make_test_sequence_with_priority(id_val, RequestPriority::Normal)
    }

    /// Build a test `SequenceInfo` with a prompt of the requested length so
    /// variable-length-window formation can be exercised.
    fn make_test_sequence_with_prompt_len(
        id_val: u64,
        prompt_len: usize,
    ) -> (SequenceInfo, mpsc::Receiver<GenerateEvent>) {
        let (mut seq, rx) = make_test_sequence_with_priority(id_val, RequestPriority::Normal);
        let prompt_tokens: Vec<i32> = (0..prompt_len as i32).map(|i| i + 1).collect();
        let tokenizer = crate::tokenizer::MlxcelTokenizer::stub();
        seq.decode_state = StreamingDecodeState::new(&tokenizer, &prompt_tokens);
        seq.prompt_tokens = prompt_tokens;
        (seq, rx)
    }

    #[test]
    fn new_queue_is_empty() {
        let queue = PrefillQueue::new();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
        assert!(!queue.is_full());
    }

    #[test]
    fn enqueue_increases_len() {
        let mut queue = PrefillQueue::new();
        let (seq, _rx) = make_test_sequence(0);
        assert!(queue.enqueue(seq).is_ok());
        assert_eq!(queue.len(), 1);
        assert!(!queue.is_empty());
    }

    #[test]
    fn dequeue_returns_none_when_empty() {
        let mut queue = PrefillQueue::new();
        assert!(queue.dequeue().is_none());
    }

    #[test]
    fn fifo_ordering_within_same_priority() {
        let mut queue = PrefillQueue::new();

        let (s1, _r1) = make_test_sequence(10);
        let (s2, _r2) = make_test_sequence(20);
        let (s3, _r3) = make_test_sequence(30);

        assert!(queue.enqueue(s1).is_ok());
        assert!(queue.enqueue(s2).is_ok());
        assert!(queue.enqueue(s3).is_ok());

        assert_eq!(queue.len(), 3);

        assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 10);
        assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 20);
        assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 30);
        assert!(queue.is_empty());
    }

    #[test]
    fn priority_ordering_high_before_normal_before_low() {
        let mut queue = PrefillQueue::new();

        // Enqueue in reverse priority order
        let (s_low, _r1) = make_test_sequence_with_priority(1, RequestPriority::Low);
        let (s_norm, _r2) = make_test_sequence_with_priority(2, RequestPriority::Normal);
        let (s_high, _r3) = make_test_sequence_with_priority(3, RequestPriority::High);

        assert!(queue.enqueue(s_low).is_ok());
        assert!(queue.enqueue(s_norm).is_ok());
        assert!(queue.enqueue(s_high).is_ok());

        // Should dequeue high first, then normal, then low
        assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 3); // High
        assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 2); // Normal
        assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 1); // Low
        assert!(queue.is_empty());
    }

    #[test]
    fn priority_with_fifo_within_lanes() {
        let mut queue = PrefillQueue::new();

        let (s_high1, _r1) = make_test_sequence_with_priority(10, RequestPriority::High);
        let (s_norm1, _r2) = make_test_sequence_with_priority(20, RequestPriority::Normal);
        let (s_high2, _r3) = make_test_sequence_with_priority(11, RequestPriority::High);
        let (s_norm2, _r4) = make_test_sequence_with_priority(21, RequestPriority::Normal);

        assert!(queue.enqueue(s_high1).is_ok());
        assert!(queue.enqueue(s_norm1).is_ok());
        assert!(queue.enqueue(s_high2).is_ok());
        assert!(queue.enqueue(s_norm2).is_ok());

        // High lane: 10, 11 (FIFO), then Normal lane: 20, 21 (FIFO)
        assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 10);
        assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 11);
        assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 20);
        assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 21);
    }

    #[test]
    fn peek_priority_reflects_head() {
        let mut queue = PrefillQueue::new();
        assert!(queue.peek_priority().is_none());

        let (s_low, _r1) = make_test_sequence_with_priority(1, RequestPriority::Low);
        queue.enqueue(s_low).unwrap();
        assert_eq!(queue.peek_priority(), Some(RequestPriority::Low));

        let (s_high, _r2) = make_test_sequence_with_priority(2, RequestPriority::High);
        queue.enqueue(s_high).unwrap();
        assert_eq!(queue.peek_priority(), Some(RequestPriority::High));
    }

    #[test]
    fn interleaved_enqueue_dequeue() {
        let mut queue = PrefillQueue::new();

        let (s1, _r1) = make_test_sequence(1);
        assert!(queue.enqueue(s1).is_ok());

        let d1 = queue.dequeue().unwrap();
        assert_eq!(d1.seq_id.as_u64(), 1);

        let (s2, _r2) = make_test_sequence(2);
        let (s3, _r3) = make_test_sequence(3);
        assert!(queue.enqueue(s2).is_ok());
        assert!(queue.enqueue(s3).is_ok());

        assert_eq!(queue.len(), 2);
        assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 2);
        assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 3);
        assert!(queue.is_empty());
    }

    #[test]
    fn default_creates_empty_queue() {
        let queue = PrefillQueue::default();
        assert!(queue.is_empty());
        assert_eq!(queue.max_size(), DEFAULT_MAX_QUEUE_SIZE);
    }

    #[test]
    fn capacity_enforcement_across_lanes() {
        let mut queue = PrefillQueue::with_capacity(2);

        let (s1, _r1) = make_test_sequence_with_priority(1, RequestPriority::High);
        let (s2, _r2) = make_test_sequence_with_priority(2, RequestPriority::Low);
        let (s3, _r3) = make_test_sequence_with_priority(3, RequestPriority::Normal);

        assert!(queue.enqueue(s1).is_ok());
        assert!(!queue.is_full());

        assert!(queue.enqueue(s2).is_ok());
        assert!(queue.is_full());

        // Third enqueue should fail regardless of priority
        let rejected = queue.enqueue(s3);
        assert!(rejected.is_err());
        let returned_seq = rejected.unwrap_err();
        assert_eq!(returned_seq.seq_id.as_u64(), 3);
        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn capacity_reopens_after_dequeue() {
        let mut queue = PrefillQueue::with_capacity(1);

        let (s1, _r1) = make_test_sequence(1);
        assert!(queue.enqueue(s1).is_ok());
        assert!(queue.is_full());

        queue.dequeue();
        assert!(!queue.is_full());

        let (s2, _r2) = make_test_sequence(2);
        assert!(queue.enqueue(s2).is_ok());
    }

    #[test]
    fn drain_cancelled_removes_only_cancelled_sequences() {
        use std::sync::atomic::Ordering;

        let mut queue = PrefillQueue::new();

        let (s1, _r1) = make_test_sequence(1);
        let (s2, _r2) = make_test_sequence(2);
        let (s3, _r3) = make_test_sequence(3);

        // Mark s2 as cancelled before enqueueing
        s2.cancelled.store(true, Ordering::Relaxed);

        queue.enqueue(s1).unwrap();
        queue.enqueue(s2).unwrap();
        queue.enqueue(s3).unwrap();
        assert_eq!(queue.len(), 3);

        let drained = queue.drain_cancelled();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].seq_id.as_u64(), 2);
        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn drain_cancelled_returns_empty_when_none_cancelled() {
        let mut queue = PrefillQueue::new();

        let (s1, _r1) = make_test_sequence(10);
        let (s2, _r2) = make_test_sequence(20);
        queue.enqueue(s1).unwrap();
        queue.enqueue(s2).unwrap();

        let drained = queue.drain_cancelled();
        assert!(drained.is_empty());
        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn drain_cancelled_removes_across_all_priority_lanes() {
        use std::sync::atomic::Ordering;

        let mut queue = PrefillQueue::new();

        let (s_high, _r1) = make_test_sequence_with_priority(1, RequestPriority::High);
        let (s_norm, _r2) = make_test_sequence_with_priority(2, RequestPriority::Normal);
        let (s_low, _r3) = make_test_sequence_with_priority(3, RequestPriority::Low);

        // Cancel the high and low priority sequences
        s_high.cancelled.store(true, Ordering::Relaxed);
        s_low.cancelled.store(true, Ordering::Relaxed);

        queue.enqueue(s_high).unwrap();
        queue.enqueue(s_norm).unwrap();
        queue.enqueue(s_low).unwrap();

        let drained = queue.drain_cancelled();
        assert_eq!(drained.len(), 2);
        let drained_ids: Vec<u64> = drained.iter().map(|s| s.seq_id.as_u64()).collect();
        assert!(drained_ids.contains(&1));
        assert!(drained_ids.contains(&3));

        // Only the normal priority sequence should remain
        assert_eq!(queue.len(), 1);
        let remaining = queue.dequeue().unwrap();
        assert_eq!(remaining.seq_id.as_u64(), 2);
    }

    // -----------------------------------------------------------------
    // drain_matching_window (— batched speculative burst)
    // -----------------------------------------------------------------

    #[test]
    fn drain_matching_window_collects_leading_matches() {
        let mut queue = PrefillQueue::new();
        for id in [10, 20, 30] {
            let (s, _r) = make_test_sequence(id);
            queue.enqueue(s).unwrap();
        }
        // Predicate matches everything; drain up to 2 extras.
        let window = queue.drain_matching_window(RequestPriority::Normal, 2, |_| true);
        let ids: Vec<u64> = window.iter().map(|s| s.seq_id.as_u64()).collect();
        assert_eq!(
            ids,
            vec![10, 20],
            "must collect the 2 leading matches in FIFO order"
        );
        // The third sequence stays in the queue.
        assert_eq!(queue.len(), 1);
        assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 30);
    }

    #[test]
    fn drain_matching_window_stops_at_first_non_match() {
        let mut queue = PrefillQueue::new();
        for id in [10, 20, 30, 40] {
            let (s, _r) = make_test_sequence(id);
            queue.enqueue(s).unwrap();
        }
        // Predicate matches ids 10 and 20 but not 30; scanning must stop
        // at 30 so 30 and 40 stay queued in FIFO order.
        let window =
            queue.drain_matching_window(RequestPriority::Normal, 10, |s| s.seq_id.as_u64() < 30);
        let ids: Vec<u64> = window.iter().map(|s| s.seq_id.as_u64()).collect();
        assert_eq!(ids, vec![10, 20]);
        assert_eq!(queue.len(), 2);
        assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 30);
        assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 40);
    }

    #[test]
    fn drain_matching_window_respects_max_extra_cap() {
        let mut queue = PrefillQueue::new();
        for id in [1, 2, 3, 4, 5] {
            let (s, _r) = make_test_sequence(id);
            queue.enqueue(s).unwrap();
        }
        // Even though every sequence matches, only 3 may be drained.
        let window = queue.drain_matching_window(RequestPriority::Normal, 3, |_| true);
        assert_eq!(window.len(), 3);
        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn drain_matching_window_zero_max_extra_collects_nothing() {
        let mut queue = PrefillQueue::new();
        let (s, _r) = make_test_sequence(1);
        queue.enqueue(s).unwrap();
        let window = queue.drain_matching_window(RequestPriority::Normal, 0, |_| true);
        assert!(window.is_empty());
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn drain_matching_window_only_scans_named_lane() {
        let mut queue = PrefillQueue::new();
        // High lane has 2 matches; normal lane has 1.
        let (h1, _r1) = make_test_sequence_with_priority(1, RequestPriority::High);
        let (h2, _r2) = make_test_sequence_with_priority(2, RequestPriority::High);
        let (n1, _r3) = make_test_sequence_with_priority(3, RequestPriority::Normal);
        queue.enqueue(h1).unwrap();
        queue.enqueue(h2).unwrap();
        queue.enqueue(n1).unwrap();
        // Draining the High lane must not touch the Normal lane.
        let window = queue.drain_matching_window(RequestPriority::High, 10, |_| true);
        assert_eq!(window.len(), 2, "only the High lane is scanned");
        // The normal-lane sequence is untouched.
        assert_eq!(queue.len(), 1);
        assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 3);
    }

    #[test]
    fn drain_matching_window_empty_lane_returns_empty() {
        let mut queue = PrefillQueue::new();
        let window = queue.drain_matching_window(RequestPriority::Low, 5, |_| true);
        assert!(window.is_empty());
    }

    // -----------------------------------------------------------------
    // Variable-length (ragged) batched MTP window formation (issue #161)
    //
    // The scheduler's `try_speculative_burst` window predicate gates the
    // prompt-length equality behind an `allow_ragged` flag:
    //     (allow_ragged || candidate.prompt_tokens.len() == head_prompt_len)
    //         && ...other matching checks...
    // These tests pin that exact gating against `drain_matching_window`,
    // which is the mechanism the scheduler drives.
    // -----------------------------------------------------------------

    /// With ragged disabled the length-equality gate keeps different-length
    /// candidates out of the head's window (the legacy same-length behaviour).
    #[test]
    fn ragged_disabled_window_excludes_different_length_prompts() {
        let mut queue = PrefillQueue::new();
        let head_prompt_len = 4usize;
        // Head-equivalent length 4; sibling 20 differs (length 7); sibling 30
        // matches length 4 again but is behind 20, so FIFO scanning stops at 20.
        let (s10, _r10) = make_test_sequence_with_prompt_len(10, 4);
        let (s20, _r20) = make_test_sequence_with_prompt_len(20, 7);
        let (s30, _r30) = make_test_sequence_with_prompt_len(30, 4);
        queue.enqueue(s10).unwrap();
        queue.enqueue(s20).unwrap();
        queue.enqueue(s30).unwrap();

        let allow_ragged = false;
        let window = queue.drain_matching_window(RequestPriority::Normal, 10, |candidate| {
            allow_ragged || candidate.prompt_tokens.len() == head_prompt_len
        });
        let ids: Vec<u64> = window.iter().map(|s| s.seq_id.as_u64()).collect();
        // 10 matches; 20 fails the length gate and scanning stops there.
        assert_eq!(
            ids,
            vec![10],
            "ragged-off must stop at the first length mismatch"
        );
        assert_eq!(queue.len(), 2);
    }

    /// With ragged enabled the length-equality gate is dropped, so
    /// burst-eligible candidates of different prompt lengths all join one
    /// window in FIFO order.
    #[test]
    fn ragged_enabled_window_groups_different_length_prompts() {
        let mut queue = PrefillQueue::new();
        let head_prompt_len = 4usize;
        let (s10, _r10) = make_test_sequence_with_prompt_len(10, 4);
        let (s20, _r20) = make_test_sequence_with_prompt_len(20, 7);
        let (s30, _r30) = make_test_sequence_with_prompt_len(30, 2);
        queue.enqueue(s10).unwrap();
        queue.enqueue(s20).unwrap();
        queue.enqueue(s30).unwrap();

        let allow_ragged = true;
        let window = queue.drain_matching_window(RequestPriority::Normal, 10, |candidate| {
            allow_ragged || candidate.prompt_tokens.len() == head_prompt_len
        });
        let ids: Vec<u64> = window.iter().map(|s| s.seq_id.as_u64()).collect();
        assert_eq!(
            ids,
            vec![10, 20, 30],
            "ragged-on must group different-length candidates into one window"
        );
        // Confirm the lengths actually differ — a true ragged window.
        let lens: Vec<usize> = window.iter().map(|s| s.prompt_tokens.len()).collect();
        assert_eq!(lens, vec![4, 7, 2]);
        assert!(queue.is_empty());
    }

    /// Even with ragged enabled, the *other* window constraints still apply.
    /// Here the `max_tokens` mismatch (mirroring the scheduler's
    /// `candidate.max_tokens == head_max_tokens` check) excludes a row whose
    /// prompt length differs AND whose budget differs.
    #[test]
    fn ragged_enabled_window_still_honors_other_matching_checks() {
        let mut queue = PrefillQueue::new();
        let head_max_tokens = 100usize; // make_test_sequence default
        let (s10, _r10) = make_test_sequence_with_prompt_len(10, 4);
        let (mut s20, _r20) = make_test_sequence_with_prompt_len(20, 7);
        s20.max_tokens = 200; // differs from head -> must be excluded
        queue.enqueue(s10).unwrap();
        queue.enqueue(s20).unwrap();

        let allow_ragged = true;
        let window = queue.drain_matching_window(RequestPriority::Normal, 10, |candidate| {
            // Length gate dropped, but max_tokens equality still enforced.
            (allow_ragged || candidate.prompt_tokens.len() == 4)
                && candidate.max_tokens == head_max_tokens
        });
        let ids: Vec<u64> = window.iter().map(|s| s.seq_id.as_u64()).collect();
        assert_eq!(
            ids,
            vec![10],
            "ragged-on must still respect the non-length window checks"
        );
        assert_eq!(queue.len(), 1);
    }
}
