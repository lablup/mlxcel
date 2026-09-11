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
use crate::distributed::tensor_parallel::synchronized::{
    EvictionReason, SampledTokens, SamplingMode, StepDecision, TPExecutionConfig,
};

fn make_scheduler(max_batch: usize) -> TPScheduler {
    let mut config = TPExecutionConfig::new(0, 4);
    config.max_batch_size = max_batch;
    TPScheduler::new(config, SamplingMode::ReplicatedLmHead).unwrap()
}

// ---------------------------------------------------------------------------
// Construction tests
// ---------------------------------------------------------------------------

#[test]
fn test_scheduler_new_rank0() {
    let scheduler = make_scheduler(8);
    assert_eq!(scheduler.active_count(), 0);
    assert_eq!(scheduler.waiting_count(), 0);
    assert_eq!(scheduler.next_step_id(), 0);
}

#[test]
fn test_scheduler_rejects_non_coordinator() {
    let config = TPExecutionConfig::new(1, 4);
    let result = TPScheduler::new(config, SamplingMode::ReplicatedLmHead);
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// Sequence submission
// ---------------------------------------------------------------------------

#[test]
fn test_submit_sequence() {
    let mut scheduler = make_scheduler(8);
    scheduler.submit_sequence(1, 10, 100, 0).unwrap();
    assert_eq!(scheduler.waiting_count(), 1);
    assert_eq!(scheduler.total_sequences(), 1);
    assert_eq!(scheduler.sequence_state(1), Some(SequenceState::Waiting));
}

#[test]
fn test_submit_duplicate_rejected() {
    let mut scheduler = make_scheduler(8);
    scheduler.submit_sequence(1, 10, 100, 0).unwrap();
    assert!(scheduler.submit_sequence(1, 10, 100, 0).is_err());
}

#[test]
fn test_submit_zero_prompt_rejected() {
    let mut scheduler = make_scheduler(8);
    assert!(scheduler.submit_sequence(1, 0, 100, 0).is_err());
}

#[test]
fn test_submit_zero_max_tokens_rejected() {
    let mut scheduler = make_scheduler(8);
    assert!(scheduler.submit_sequence(1, 10, 0, 0).is_err());
}

// ---------------------------------------------------------------------------
// Scheduling
// ---------------------------------------------------------------------------

#[test]
fn test_schedule_admits_and_prefills() {
    let mut scheduler = make_scheduler(8);
    scheduler.submit_sequence(1, 10, 100, 0).unwrap();
    scheduler.submit_sequence(2, 20, 100, 0).unwrap();

    let decisions = scheduler.schedule_step().unwrap();

    // Expect: AdmitSequence(1), AdmitSequence(2), Prefill([1, 2]), Decode([1, 2])
    assert_eq!(decisions.len(), 4);

    // First two should be admit decisions.
    assert!(matches!(
        &decisions[0],
        StepDecision::AdmitSequence {
            seq_id: 1,
            prompt_len: 10,
            ..
        }
    ));
    assert!(matches!(
        &decisions[1],
        StepDecision::AdmitSequence {
            seq_id: 2,
            prompt_len: 20,
            ..
        }
    ));

    // Third should be prefill.
    if let StepDecision::Prefill {
        seq_ids,
        token_counts,
        ..
    } = &decisions[2]
    {
        assert_eq!(seq_ids, &[1, 2]);
        assert_eq!(token_counts, &[10, 20]);
    } else {
        panic!("expected Prefill decision, got {:?}", decisions[2]);
    }

    // Fourth should be decode (sequences are now in Decoding state).
    assert!(matches!(
        &decisions[3],
        StepDecision::Decode { seq_ids, .. } if seq_ids == &[1, 2]
    ));

    assert_eq!(scheduler.active_count(), 2);
    assert_eq!(scheduler.waiting_count(), 0);
}

#[test]
fn test_schedule_decode_after_prefill() {
    let mut scheduler = make_scheduler(8);
    scheduler.submit_sequence(1, 10, 100, 0).unwrap();

    // First schedule: admit + prefill.
    let _ = scheduler.schedule_step().unwrap();

    // Second schedule: decode (sequence is now in Decoding state).
    let decisions = scheduler.schedule_step().unwrap();
    assert_eq!(decisions.len(), 1);
    assert!(matches!(
        &decisions[0],
        StepDecision::Decode { seq_ids, .. } if seq_ids == &[1]
    ));
}

#[test]
fn test_schedule_respects_max_batch_size() {
    let mut scheduler = make_scheduler(2);
    scheduler.submit_sequence(1, 10, 100, 0).unwrap();
    scheduler.submit_sequence(2, 10, 100, 0).unwrap();
    scheduler.submit_sequence(3, 10, 100, 0).unwrap();

    let decisions = scheduler.schedule_step().unwrap();

    // Only 2 should be admitted (max_batch_size = 2).
    let admit_count = decisions
        .iter()
        .filter(|d| matches!(d, StepDecision::AdmitSequence { .. }))
        .count();
    assert_eq!(admit_count, 2);
    assert_eq!(scheduler.waiting_count(), 1);
}

#[test]
fn test_schedule_empty_returns_empty() {
    let mut scheduler = make_scheduler(8);
    let decisions = scheduler.schedule_step().unwrap();
    assert!(decisions.is_empty());
}

// ---------------------------------------------------------------------------
// Token recording and completion
// ---------------------------------------------------------------------------

#[test]
fn test_record_sampled_tokens() {
    let mut scheduler = make_scheduler(8);
    scheduler.submit_sequence(1, 10, 100, 0).unwrap();
    let _ = scheduler.schedule_step().unwrap();

    let sampled = SampledTokens {
        step_id: 99,
        tokens: vec![42],
        completed_seq_ids: vec![],
    };
    let evictions = scheduler.record_sampled_tokens(&sampled).unwrap();
    assert!(evictions.is_empty());

    let info = scheduler.sequence_info(1).unwrap();
    assert_eq!(info.generated_tokens, 1);
}

#[test]
fn test_record_sampled_with_completion() {
    let mut scheduler = make_scheduler(8);
    scheduler.submit_sequence(1, 10, 5, 0).unwrap();
    let _ = scheduler.schedule_step().unwrap();

    let sampled = SampledTokens {
        step_id: 99,
        tokens: vec![42],
        completed_seq_ids: vec![1],
    };
    let evictions = scheduler.record_sampled_tokens(&sampled).unwrap();
    assert_eq!(evictions.len(), 1);
    assert!(matches!(
        &evictions[0],
        StepDecision::EvictSequence { seq_id: 1, .. }
    ));
    assert_eq!(scheduler.active_count(), 0);
    assert_eq!(scheduler.sequence_state(1), Some(SequenceState::Completed));
}

#[test]
fn test_record_sampled_token_count_mismatch() {
    let mut scheduler = make_scheduler(8);
    scheduler.submit_sequence(1, 10, 100, 0).unwrap();
    let _ = scheduler.schedule_step().unwrap();

    let sampled = SampledTokens {
        step_id: 99,
        tokens: vec![42, 43], // 2 tokens but only 1 active sequence
        completed_seq_ids: vec![],
    };
    assert!(scheduler.record_sampled_tokens(&sampled).is_err());
}

// ---------------------------------------------------------------------------
// Manual eviction
// ---------------------------------------------------------------------------

#[test]
fn test_manual_eviction() {
    let mut scheduler = make_scheduler(8);
    scheduler.submit_sequence(1, 10, 100, 0).unwrap();
    let _ = scheduler.schedule_step().unwrap();

    let decision = scheduler
        .evict_sequence(1, EvictionReason::Cancelled)
        .unwrap();
    assert!(matches!(
        decision,
        StepDecision::EvictSequence {
            seq_id: 1,
            reason: EvictionReason::Cancelled,
            ..
        }
    ));
    assert_eq!(scheduler.active_count(), 0);
}

#[test]
fn test_evict_already_completed() {
    let mut scheduler = make_scheduler(8);
    scheduler.submit_sequence(1, 10, 5, 0).unwrap();
    let _ = scheduler.schedule_step().unwrap();

    // Complete the sequence.
    let sampled = SampledTokens {
        step_id: 99,
        tokens: vec![42],
        completed_seq_ids: vec![1],
    };
    let _ = scheduler.record_sampled_tokens(&sampled).unwrap();

    // Trying to evict again should fail.
    assert!(
        scheduler
            .evict_sequence(1, EvictionReason::Cancelled)
            .is_err()
    );
}

// ---------------------------------------------------------------------------
// Barrier and shutdown
// ---------------------------------------------------------------------------

#[test]
fn test_barrier_decision() {
    let mut scheduler = make_scheduler(8);
    let decision = scheduler.barrier();
    assert!(matches!(decision, StepDecision::Barrier { .. }));
}

#[test]
fn test_shutdown_decision() {
    let mut scheduler = make_scheduler(8);
    let decision = scheduler.shutdown();
    assert!(matches!(decision, StepDecision::Shutdown { .. }));
}

// ---------------------------------------------------------------------------
// Step ID monotonicity
// ---------------------------------------------------------------------------

#[test]
fn test_step_ids_are_monotonic() {
    let mut scheduler = make_scheduler(8);
    scheduler.submit_sequence(1, 10, 100, 0).unwrap();
    scheduler.submit_sequence(2, 20, 100, 0).unwrap();

    let decisions = scheduler.schedule_step().unwrap();
    let ids: Vec<u64> = decisions.iter().map(|d| d.step_id()).collect();
    for window in ids.windows(2) {
        assert!(
            window[1] > window[0],
            "step IDs must be monotonically increasing"
        );
    }
}

// ---------------------------------------------------------------------------
// has_work
// ---------------------------------------------------------------------------

#[test]
fn test_has_work() {
    let mut scheduler = make_scheduler(8);
    assert!(!scheduler.has_work());

    scheduler.submit_sequence(1, 10, 100, 0).unwrap();
    assert!(scheduler.has_work());

    let _ = scheduler.schedule_step().unwrap();
    assert!(scheduler.has_work()); // still active

    // Complete the sequence.
    let sampled = SampledTokens {
        step_id: 99,
        tokens: vec![42],
        completed_seq_ids: vec![1],
    };
    let _ = scheduler.record_sampled_tokens(&sampled).unwrap();
    assert!(!scheduler.has_work());
}

// ---------------------------------------------------------------------------
// Continuous batching: admit after eviction
// ---------------------------------------------------------------------------

#[test]
fn test_continuous_batching_readmit_after_eviction() {
    let mut scheduler = make_scheduler(1); // batch size 1
    scheduler.submit_sequence(1, 10, 100, 0).unwrap();
    scheduler.submit_sequence(2, 20, 100, 0).unwrap();

    // First step: admit seq 1 only.
    let _decisions = scheduler.schedule_step().unwrap();
    assert_eq!(scheduler.active_count(), 1);
    assert_eq!(scheduler.waiting_count(), 1);

    // Complete seq 1.
    let sampled = SampledTokens {
        step_id: 99,
        tokens: vec![42],
        completed_seq_ids: vec![1],
    };
    let _ = scheduler.record_sampled_tokens(&sampled).unwrap();
    assert_eq!(scheduler.active_count(), 0);

    // Next step: should admit seq 2.
    let decisions = scheduler.schedule_step().unwrap();
    let admitted: Vec<_> = decisions
        .iter()
        .filter_map(|d| {
            if let StepDecision::AdmitSequence { seq_id, .. } = d {
                Some(*seq_id)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(admitted, vec![2]);
}
