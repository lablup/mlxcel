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
    EvictionReason, RankStatus, SampledTokens, SamplingMode, StepDecision, TPExecutionConfig,
};

fn make_executor(rank: usize, tp_size: usize) -> TPExecutor {
    let config = TPExecutionConfig::new(rank, tp_size);
    TPExecutor::new(config, SamplingMode::ReplicatedLmHead).unwrap()
}

// ---------------------------------------------------------------------------
// Construction tests
// ---------------------------------------------------------------------------

#[test]
fn test_executor_new() {
    let exec = make_executor(0, 4);
    assert_eq!(exec.rank(), 0);
    assert!(exec.is_coordinator());
    assert_eq!(exec.status(), RankStatus::Ready);
    assert_eq!(exec.active_count(), 0);
    assert_eq!(exec.steps_executed(), 0);
}

#[test]
fn test_executor_non_coordinator() {
    let exec = make_executor(2, 4);
    assert_eq!(exec.rank(), 2);
    assert!(!exec.is_coordinator());
}

// ---------------------------------------------------------------------------
// Admit sequence
// ---------------------------------------------------------------------------

#[test]
fn test_admit_sequence() {
    let mut exec = make_executor(0, 4);
    let decision = StepDecision::AdmitSequence {
        step_id: 0,
        seq_id: 1,
        prompt_len: 10,
    };
    let outcome = exec.execute_step(&decision).unwrap();
    assert!(matches!(
        outcome,
        StepOutcome::SequenceAdmitted {
            step_id: 0,
            seq_id: 1
        }
    ));
    assert_eq!(exec.active_count(), 1);
}

#[test]
fn test_admit_duplicate_rejected() {
    let mut exec = make_executor(0, 4);
    let d1 = StepDecision::AdmitSequence {
        step_id: 0,
        seq_id: 1,
        prompt_len: 10,
    };
    exec.execute_step(&d1).unwrap();

    let d2 = StepDecision::AdmitSequence {
        step_id: 1,
        seq_id: 1,
        prompt_len: 20,
    };
    assert!(exec.execute_step(&d2).is_err());
}

// ---------------------------------------------------------------------------
// Prefill
// ---------------------------------------------------------------------------

#[test]
fn test_prefill() {
    let mut exec = make_executor(0, 4);
    let admit = StepDecision::AdmitSequence {
        step_id: 0,
        seq_id: 1,
        prompt_len: 10,
    };
    exec.execute_step(&admit).unwrap();

    let prefill = StepDecision::Prefill {
        step_id: 1,
        seq_ids: vec![1],
        token_counts: vec![10],
    };
    let outcome = exec.execute_step(&prefill).unwrap();
    assert!(matches!(outcome, StepOutcome::Executed { step_id: 1 }));
}

#[test]
fn test_prefill_missing_sequence() {
    let mut exec = make_executor(0, 4);
    let prefill = StepDecision::Prefill {
        step_id: 0,
        seq_ids: vec![999],
        token_counts: vec![10],
    };
    assert!(exec.execute_step(&prefill).is_err());
}

#[test]
fn test_prefill_mismatched_lengths() {
    let mut exec = make_executor(0, 4);
    let admit = StepDecision::AdmitSequence {
        step_id: 0,
        seq_id: 1,
        prompt_len: 10,
    };
    exec.execute_step(&admit).unwrap();

    let prefill = StepDecision::Prefill {
        step_id: 1,
        seq_ids: vec![1],
        token_counts: vec![10, 20], // wrong length
    };
    assert!(exec.execute_step(&prefill).is_err());
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

#[test]
fn test_decode() {
    let mut exec = make_executor(0, 4);

    // Admit and prefill.
    exec.execute_step(&StepDecision::AdmitSequence {
        step_id: 0,
        seq_id: 1,
        prompt_len: 10,
    })
    .unwrap();
    exec.execute_step(&StepDecision::Prefill {
        step_id: 1,
        seq_ids: vec![1],
        token_counts: vec![10],
    })
    .unwrap();

    // Decode.
    let outcome = exec
        .execute_step(&StepDecision::Decode {
            step_id: 2,
            seq_ids: vec![1],
        })
        .unwrap();
    assert!(matches!(outcome, StepOutcome::Executed { step_id: 2 }));
}

#[test]
fn test_decode_not_prefilled() {
    let mut exec = make_executor(0, 4);
    exec.execute_step(&StepDecision::AdmitSequence {
        step_id: 0,
        seq_id: 1,
        prompt_len: 10,
    })
    .unwrap();

    // Try decode without prefill.
    assert!(
        exec.execute_step(&StepDecision::Decode {
            step_id: 1,
            seq_ids: vec![1],
        })
        .is_err()
    );
}

// ---------------------------------------------------------------------------
// Evict
// ---------------------------------------------------------------------------

#[test]
fn test_evict_sequence() {
    let mut exec = make_executor(0, 4);
    exec.execute_step(&StepDecision::AdmitSequence {
        step_id: 0,
        seq_id: 1,
        prompt_len: 10,
    })
    .unwrap();

    let outcome = exec
        .execute_step(&StepDecision::EvictSequence {
            step_id: 1,
            seq_id: 1,
            reason: EvictionReason::EndOfSequence,
        })
        .unwrap();
    assert!(matches!(
        outcome,
        StepOutcome::SequenceEvicted {
            step_id: 1,
            seq_id: 1
        }
    ));
    assert_eq!(exec.active_count(), 0);
}

// ---------------------------------------------------------------------------
// Barrier
// ---------------------------------------------------------------------------

#[test]
fn test_barrier() {
    let mut exec = make_executor(0, 4);
    let outcome = exec
        .execute_step(&StepDecision::Barrier { step_id: 0 })
        .unwrap();
    assert!(matches!(
        outcome,
        StepOutcome::BarrierReached { step_id: 0 }
    ));
}

#[test]
fn test_barrier_arrive() {
    let mut exec = make_executor(0, 4);
    exec.execute_step(&StepDecision::Barrier { step_id: 0 })
        .unwrap();

    assert!(!exec.barrier_arrive(1).unwrap());
    assert!(!exec.barrier_arrive(2).unwrap());
    assert!(exec.barrier_arrive(3).unwrap());

    // After completion, status should be ready again.
    assert_eq!(exec.status(), RankStatus::Ready);
}

#[test]
fn test_barrier_arrive_no_active_barrier() {
    let mut exec = make_executor(0, 4);
    assert!(exec.barrier_arrive(1).is_err());
}

// ---------------------------------------------------------------------------
// Shutdown
// ---------------------------------------------------------------------------

#[test]
fn test_shutdown() {
    let mut exec = make_executor(0, 4);
    let outcome = exec
        .execute_step(&StepDecision::Shutdown { step_id: 0 })
        .unwrap();
    assert!(matches!(
        outcome,
        StepOutcome::ShutdownAcknowledged { step_id: 0 }
    ));
    assert_eq!(exec.status(), RankStatus::ShutDown);
}

// ---------------------------------------------------------------------------
// Step ordering
// ---------------------------------------------------------------------------

#[test]
fn test_step_ordering_enforced() {
    let mut exec = make_executor(0, 4);
    exec.execute_step(&StepDecision::Barrier { step_id: 5 })
        .unwrap();

    // Step 3 should be rejected (not greater than 5).
    assert!(
        exec.execute_step(&StepDecision::Barrier { step_id: 3 })
            .is_err()
    );
}

#[test]
fn test_step_ordering_equal_rejected() {
    let mut exec = make_executor(0, 4);
    exec.execute_step(&StepDecision::Barrier { step_id: 5 })
        .unwrap();

    assert!(
        exec.execute_step(&StepDecision::Barrier { step_id: 5 })
            .is_err()
    );
}

// ---------------------------------------------------------------------------
// Apply sampled tokens
// ---------------------------------------------------------------------------

#[test]
fn test_apply_sampled_tokens() {
    let mut exec = make_executor(0, 4);

    // Admit + prefill seq 1.
    exec.execute_step(&StepDecision::AdmitSequence {
        step_id: 0,
        seq_id: 1,
        prompt_len: 10,
    })
    .unwrap();
    exec.execute_step(&StepDecision::Prefill {
        step_id: 1,
        seq_ids: vec![1],
        token_counts: vec![10],
    })
    .unwrap();

    // Apply a sampled token.
    let sampled = SampledTokens {
        step_id: 2,
        tokens: vec![42],
        completed_seq_ids: vec![],
    };
    exec.apply_sampled_tokens(&sampled).unwrap();
    assert_eq!(exec.active_count(), 1);
}

#[test]
fn test_apply_sampled_tokens_with_completion() {
    let mut exec = make_executor(0, 4);

    exec.execute_step(&StepDecision::AdmitSequence {
        step_id: 0,
        seq_id: 1,
        prompt_len: 10,
    })
    .unwrap();
    exec.execute_step(&StepDecision::Prefill {
        step_id: 1,
        seq_ids: vec![1],
        token_counts: vec![10],
    })
    .unwrap();

    let sampled = SampledTokens {
        step_id: 2,
        tokens: vec![42],
        completed_seq_ids: vec![1],
    };
    exec.apply_sampled_tokens(&sampled).unwrap();
    assert_eq!(exec.active_count(), 0);
}

#[test]
fn test_apply_sampled_tokens_count_mismatch() {
    let mut exec = make_executor(0, 4);

    exec.execute_step(&StepDecision::AdmitSequence {
        step_id: 0,
        seq_id: 1,
        prompt_len: 10,
    })
    .unwrap();
    exec.execute_step(&StepDecision::Prefill {
        step_id: 1,
        seq_ids: vec![1],
        token_counts: vec![10],
    })
    .unwrap();

    let sampled = SampledTokens {
        step_id: 2,
        tokens: vec![42, 43], // 2 tokens for 1 sequence
        completed_seq_ids: vec![],
    };
    assert!(exec.apply_sampled_tokens(&sampled).is_err());
}

// ---------------------------------------------------------------------------
// Error counting
// ---------------------------------------------------------------------------

#[test]
fn test_error_count() {
    let mut exec = make_executor(0, 4);

    // Force an error by decoding a non-existent sequence.
    let _ = exec.execute_step(&StepDecision::Decode {
        step_id: 0,
        seq_ids: vec![999],
    });
    assert_eq!(exec.error_count(), 1);
}

// ---------------------------------------------------------------------------
// Report failure
// ---------------------------------------------------------------------------

#[test]
fn test_report_failure() {
    let mut exec = make_executor(1, 4);
    exec.report_failure("test failure");
    assert_eq!(exec.status(), RankStatus::Failed);
    assert_eq!(exec.error_count(), 1);
    assert!(!exec.group_health().is_healthy());
}

// ---------------------------------------------------------------------------
// Multi-rank lockstep simulation
// ---------------------------------------------------------------------------

#[test]
fn test_lockstep_two_executors() {
    let mut exec0 = make_executor(0, 2);
    let mut exec1 = make_executor(1, 2);

    // Both admit the same sequence.
    let admit = StepDecision::AdmitSequence {
        step_id: 0,
        seq_id: 1,
        prompt_len: 10,
    };
    exec0.execute_step(&admit).unwrap();
    exec1.execute_step(&admit).unwrap();

    // Both prefill.
    let prefill = StepDecision::Prefill {
        step_id: 1,
        seq_ids: vec![1],
        token_counts: vec![10],
    };
    exec0.execute_step(&prefill).unwrap();
    exec1.execute_step(&prefill).unwrap();

    // Both decode.
    let decode = StepDecision::Decode {
        step_id: 2,
        seq_ids: vec![1],
    };
    exec0.execute_step(&decode).unwrap();
    exec1.execute_step(&decode).unwrap();

    // Both apply the same sampled token.
    let sampled = SampledTokens {
        step_id: 2,
        tokens: vec![42],
        completed_seq_ids: vec![1],
    };
    exec0.apply_sampled_tokens(&sampled).unwrap();
    exec1.apply_sampled_tokens(&sampled).unwrap();

    // Both should have 0 active sequences.
    assert_eq!(exec0.active_count(), 0);
    assert_eq!(exec1.active_count(), 0);
}

// ---------------------------------------------------------------------------
// Barrier timeout detection
// ---------------------------------------------------------------------------

#[test]
fn test_barrier_timeout_detection() {
    let mut config = TPExecutionConfig::new(0, 4);
    config.barrier_timeout = std::time::Duration::from_millis(1);
    let mut exec = TPExecutor::new(config, SamplingMode::ReplicatedLmHead).unwrap();

    exec.execute_step(&StepDecision::Barrier { step_id: 0 })
        .unwrap();
    std::thread::sleep(std::time::Duration::from_millis(5));
    assert!(exec.barrier_timed_out());

    let missing = exec.barrier_missing_ranks();
    assert_eq!(missing, vec![1, 2, 3]);
}
