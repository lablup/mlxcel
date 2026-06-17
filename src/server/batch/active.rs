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

//! Active decode batch with O(1) lookup by [`SequenceId`].
//!
//! [`ActiveBatch`] holds the sequences that have finished prefill and are
//! currently generating tokens. It enforces a maximum capacity so the
//! scheduler can gate admission from the prefill queue.

use std::collections::HashMap;

use anyhow::{Result, bail};
use mlxcel_core::cache::SequenceId;

use super::sequence::SequenceInfo;

/// Set of sequences currently in the decode phase.
///
/// Backed by a `HashMap` for O(1) lookups, insertions, and removals by
/// `SequenceId`. A configurable `max_size` prevents unbounded growth.
pub struct ActiveBatch {
    sequences: HashMap<SequenceId, SequenceInfo>,
    max_size: usize,
}

impl ActiveBatch {
    /// Create an empty batch with the given capacity limit.
    pub fn new(max_size: usize) -> Self {
        Self {
            sequences: HashMap::with_capacity(max_size),
            max_size,
        }
    }

    /// Insert a sequence into the active batch.
    ///
    /// Returns `Err` if the batch is already at capacity or if a sequence with
    /// the same ID is already present.
    pub fn add(&mut self, seq: SequenceInfo) -> Result<()> {
        if self.sequences.len() >= self.max_size {
            bail!(
                "active batch full: cannot add {} (capacity {})",
                seq.seq_id,
                self.max_size
            );
        }
        if self.sequences.contains_key(&seq.seq_id) {
            bail!("duplicate sequence id: {}", seq.seq_id);
        }
        self.sequences.insert(seq.seq_id, seq);
        Ok(())
    }

    /// Remove and return a sequence by ID.
    pub fn remove(&mut self, id: SequenceId) -> Option<SequenceInfo> {
        self.sequences.remove(&id)
    }

    /// Get a mutable reference to a sequence by ID.
    pub fn get_mut(&mut self, id: SequenceId) -> Option<&mut SequenceInfo> {
        self.sequences.get_mut(&id)
    }

    /// Get a shared reference to a sequence by ID.
    ///
    /// Used by the batched-decode fast-path gate, which inspects each row's
    /// sampling config and per-row obligations without mutating the batch.
    pub fn get(&self, id: SequenceId) -> Option<&SequenceInfo> {
        self.sequences.get(&id)
    }

    /// Returns `true` when the batch has reached its capacity limit.
    pub fn is_full(&self) -> bool {
        self.sequences.len() >= self.max_size
    }

    /// Number of sequences currently in the batch.
    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    /// Capacity limit of the batch — the configured `max_batch_size`.
    ///
    /// the speculative-burst scheduler uses this to decide
    /// whether to attempt a B > 1 batched burst window. The scheduler
    /// constructs `ActiveBatch::new(max_batch_size)`, so this accessor
    /// surfaces the same `max_batch_size` without storing it twice on
    /// the scheduler.
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Returns `true` if no sequences are in the batch.
    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    /// Snapshot of all active sequence IDs (order is not guaranteed).
    pub fn sequence_ids(&self) -> Vec<SequenceId> {
        self.sequences.keys().copied().collect()
    }

    /// Iterate over all active sequences mutably.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut SequenceInfo> {
        self.sequences.values_mut()
    }

    /// Iterate over all active sequences immutably.
    pub fn iter_sequences(&self) -> impl Iterator<Item = &SequenceInfo> {
        self.sequences.values()
    }

    /// Find the minimum priority among all active sequences.
    pub fn iter_min_priority(&self) -> Option<crate::server::batch::sequence::RequestPriority> {
        self.sequences.values().map(|s| s.priority).min()
    }
}

// Manual Debug: SequenceInfo is not Debug-derivable (InputEmbeddings).
impl std::fmt::Debug for ActiveBatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActiveBatch")
            .field("len", &self.sequences.len())
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
    use mlxcel_core::generate::SamplingConfig;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::sync::mpsc;
    use std::time::Instant;

    fn make_test_sequence(id_val: u64) -> (SequenceInfo, mpsc::Receiver<GenerateEvent>) {
        let (tx, rx) = mpsc::channel();
        let tokenizer = crate::tokenizer::MlxcelTokenizer::stub();
        let prompt_tokens = vec![1, 2, 3];
        let decode_state = StreamingDecodeState::new(&tokenizer, &prompt_tokens);

        let seq = SequenceInfo {
            seq_id: SequenceId::from_raw(id_val),
            state: SequenceState::Queued,
            prompt_tokens,
            sampling: SamplingConfig::default(),
            max_tokens: 64,
            eos_token_ids: vec![2],
            priority: crate::server::batch::RequestPriority::Normal,
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

    #[test]
    fn new_batch_is_empty() {
        let batch = ActiveBatch::new(8);
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
        assert!(!batch.is_full());
    }

    #[test]
    fn add_and_retrieve() {
        let mut batch = ActiveBatch::new(4);
        let (seq, _rx) = make_test_sequence(1);
        let id = seq.seq_id;

        batch.add(seq).unwrap();
        assert_eq!(batch.len(), 1);
        assert!(batch.get_mut(id).is_some());
    }

    #[test]
    fn remove_returns_sequence() {
        let mut batch = ActiveBatch::new(4);
        let (seq, _rx) = make_test_sequence(42);
        let id = seq.seq_id;

        batch.add(seq).unwrap();
        let removed = batch.remove(id);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().seq_id.as_u64(), 42);
        assert!(batch.is_empty());
    }

    #[test]
    fn remove_nonexistent_returns_none() {
        let mut batch = ActiveBatch::new(4);
        assert!(batch.remove(SequenceId::from_raw(999)).is_none());
    }

    #[test]
    fn capacity_enforcement() {
        let mut batch = ActiveBatch::new(2);

        let (s1, _r1) = make_test_sequence(1);
        let (s2, _r2) = make_test_sequence(2);
        let (s3, _r3) = make_test_sequence(3);

        batch.add(s1).unwrap();
        assert!(!batch.is_full());

        batch.add(s2).unwrap();
        assert!(batch.is_full());

        let result = batch.add(s3);
        assert!(result.is_err());
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn capacity_reopens_after_remove() {
        let mut batch = ActiveBatch::new(1);

        let (s1, _r1) = make_test_sequence(1);
        let id1 = s1.seq_id;
        batch.add(s1).unwrap();
        assert!(batch.is_full());

        batch.remove(id1);
        assert!(!batch.is_full());

        let (s2, _r2) = make_test_sequence(2);
        batch.add(s2).unwrap();
        assert!(batch.is_full());
    }

    #[test]
    fn duplicate_id_rejected() {
        let mut batch = ActiveBatch::new(4);

        let (s1, _r1) = make_test_sequence(5);
        let (s2, _r2) = make_test_sequence(5);

        batch.add(s1).unwrap();
        let result = batch.add(s2);
        assert!(result.is_err());
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn sequence_ids_returns_all_ids() {
        let mut batch = ActiveBatch::new(8);

        let (s1, _r1) = make_test_sequence(10);
        let (s2, _r2) = make_test_sequence(20);
        let (s3, _r3) = make_test_sequence(30);

        batch.add(s1).unwrap();
        batch.add(s2).unwrap();
        batch.add(s3).unwrap();

        let mut ids: Vec<u64> = batch.sequence_ids().iter().map(|id| id.as_u64()).collect();
        ids.sort();
        assert_eq!(ids, vec![10, 20, 30]);
    }

    #[test]
    fn iter_mut_visits_all_sequences() {
        let mut batch = ActiveBatch::new(8);

        let (s1, _r1) = make_test_sequence(1);
        let (s2, _r2) = make_test_sequence(2);

        batch.add(s1).unwrap();
        batch.add(s2).unwrap();

        // Mutate all sequences via iter_mut.
        for seq in batch.iter_mut() {
            seq.generated_tokens.push(99);
        }

        // Verify mutations persisted.
        assert_eq!(
            batch
                .get_mut(SequenceId::from_raw(1))
                .unwrap()
                .generated_tokens,
            vec![99]
        );
        assert_eq!(
            batch
                .get_mut(SequenceId::from_raw(2))
                .unwrap()
                .generated_tokens,
            vec![99]
        );
    }

    #[test]
    fn get_mut_nonexistent_returns_none() {
        let mut batch = ActiveBatch::new(4);
        assert!(batch.get_mut(SequenceId::from_raw(999)).is_none());
    }
}
