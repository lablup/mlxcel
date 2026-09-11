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

use std::time::Duration;

use super::*;

// ---------------------------------------------------------------------------
// TPExecutionConfig tests
// ---------------------------------------------------------------------------

#[test]
fn test_config_new_coordinator() {
    let config = TPExecutionConfig::new(0, 4);
    assert!(config.is_coordinator);
    assert_eq!(config.tp_rank, 0);
    assert_eq!(config.tp_size, 4);
}

#[test]
fn test_config_new_non_coordinator() {
    let config = TPExecutionConfig::new(2, 4);
    assert!(!config.is_coordinator);
    assert_eq!(config.tp_rank, 2);
}

#[test]
fn test_config_validate_ok() {
    let config = TPExecutionConfig::new(0, 4);
    assert!(config.validate().is_ok());
}

#[test]
fn test_config_validate_rank_out_of_range() {
    let mut config = TPExecutionConfig::new(0, 4);
    config.tp_rank = 5;
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validate_zero_tp_size() {
    let mut config = TPExecutionConfig::new(0, 1);
    config.tp_size = 0;
    assert!(config.validate().is_err());
}

#[test]
fn test_config_validate_coordinator_mismatch() {
    let mut config = TPExecutionConfig::new(1, 4);
    config.is_coordinator = true; // rank 1 cannot be coordinator
    assert!(config.validate().is_err());
}

#[test]
fn test_config_default() {
    let config = TPExecutionConfig::default();
    assert_eq!(config.tp_rank, 0);
    assert_eq!(config.tp_size, 1);
    assert!(config.is_coordinator);
}

// ---------------------------------------------------------------------------
// StepDecision tests
// ---------------------------------------------------------------------------

#[test]
fn test_step_decision_step_id() {
    let decision = StepDecision::Prefill {
        step_id: 42,
        seq_ids: vec![1, 2],
        token_counts: vec![10, 20],
    };
    assert_eq!(decision.step_id(), 42);
}

#[test]
fn test_step_decision_kind_name() {
    assert_eq!(
        StepDecision::Prefill {
            step_id: 0,
            seq_ids: vec![],
            token_counts: vec![]
        }
        .kind_name(),
        "prefill"
    );
    assert_eq!(
        StepDecision::Decode {
            step_id: 0,
            seq_ids: vec![]
        }
        .kind_name(),
        "decode"
    );
    assert_eq!(
        StepDecision::AdmitSequence {
            step_id: 0,
            seq_id: 0,
            prompt_len: 0
        }
        .kind_name(),
        "admit_sequence"
    );
    assert_eq!(
        StepDecision::EvictSequence {
            step_id: 0,
            seq_id: 0,
            reason: EvictionReason::EndOfSequence
        }
        .kind_name(),
        "evict_sequence"
    );
    assert_eq!(StepDecision::Barrier { step_id: 0 }.kind_name(), "barrier");
    assert_eq!(
        StepDecision::Shutdown { step_id: 0 }.kind_name(),
        "shutdown"
    );
}

#[test]
fn test_step_decision_display() {
    let d = StepDecision::Decode {
        step_id: 5,
        seq_ids: vec![10, 20],
    };
    let s = format!("{d}");
    assert!(s.contains("Decode"));
    assert!(s.contains("step=5"));
}

#[test]
fn test_step_decision_serialize_roundtrip() {
    let original = StepDecision::Prefill {
        step_id: 1,
        seq_ids: vec![100, 200],
        token_counts: vec![50, 75],
    };
    let json = serde_json::to_string(&original).unwrap();
    let deserialized: StepDecision = serde_json::from_str(&json).unwrap();
    assert_eq!(original, deserialized);
}

// ---------------------------------------------------------------------------
// EvictionReason tests
// ---------------------------------------------------------------------------

#[test]
fn test_eviction_reason_display() {
    assert_eq!(format!("{}", EvictionReason::EndOfSequence), "eos");
    assert_eq!(
        format!("{}", EvictionReason::MemoryPressure),
        "memory_pressure"
    );
    assert_eq!(format!("{}", EvictionReason::Cancelled), "cancelled");
}

// ---------------------------------------------------------------------------
// SamplingMode tests
// ---------------------------------------------------------------------------

#[test]
fn test_sampling_mode_display() {
    assert_eq!(format!("{}", SamplingMode::VocabParallel), "vocab_parallel");
    assert_eq!(
        format!("{}", SamplingMode::ReplicatedLmHead),
        "replicated_lm_head"
    );
}

// ---------------------------------------------------------------------------
// TPBarrier tests
// ---------------------------------------------------------------------------

#[test]
fn test_barrier_single_rank() {
    let mut barrier = TPBarrier::new(0, 1, Duration::from_secs(5));
    assert!(!barrier.is_complete());
    assert_eq!(barrier.arrived_count(), 0);

    let complete = barrier.arrive(0).unwrap();
    assert!(complete);
    assert!(barrier.is_complete());
    assert_eq!(barrier.missing_ranks(), Vec::<usize>::new());
}

#[test]
fn test_barrier_multi_rank() {
    let mut barrier = TPBarrier::new(0, 4, Duration::from_secs(5));

    assert!(!barrier.arrive(0).unwrap());
    assert!(!barrier.arrive(2).unwrap());
    assert!(!barrier.arrive(1).unwrap());
    assert_eq!(barrier.missing_ranks(), vec![3]);

    assert!(barrier.arrive(3).unwrap());
    assert!(barrier.is_complete());
    assert_eq!(barrier.arrived_ranks(), vec![0, 1, 2, 3]);
}

#[test]
fn test_barrier_rank_out_of_range() {
    let mut barrier = TPBarrier::new(0, 2, Duration::from_secs(5));
    assert!(barrier.arrive(5).is_err());
}

#[test]
fn test_barrier_duplicate_arrival_rejected() {
    let mut barrier = TPBarrier::new(0, 4, Duration::from_secs(5));
    assert!(!barrier.arrive(0).unwrap());
    // Same rank arriving again is a protocol violation.
    assert!(barrier.arrive(0).is_err());
}

#[test]
fn test_barrier_timeout() {
    let barrier = TPBarrier::new(0, 2, Duration::from_millis(1));
    std::thread::sleep(Duration::from_millis(5));
    assert!(barrier.is_timed_out());
}

#[test]
fn test_barrier_step_id() {
    let barrier = TPBarrier::new(42, 2, Duration::from_secs(5));
    assert_eq!(barrier.step_id(), 42);
}

// ---------------------------------------------------------------------------
// RankStatus tests
// ---------------------------------------------------------------------------

#[test]
fn test_rank_status_display() {
    assert_eq!(format!("{}", RankStatus::Ready), "ready");
    assert_eq!(format!("{}", RankStatus::Executing), "executing");
    assert_eq!(format!("{}", RankStatus::Failed), "failed");
    assert_eq!(format!("{}", RankStatus::ShutDown), "shut_down");
}

// ---------------------------------------------------------------------------
// TPGroupHealth tests
// ---------------------------------------------------------------------------

#[test]
fn test_group_health_initial() {
    let health = TPGroupHealth::new(4, Duration::from_secs(30));
    assert!(health.is_healthy());
    assert!(health.all_ready());
    assert!(!health.all_shut_down());
    assert_eq!(health.tp_size(), 4);
    assert_eq!(health.unhealthy_ranks(), Vec::<usize>::new());
}

#[test]
fn test_group_health_rank_failure() {
    let mut health = TPGroupHealth::new(4, Duration::from_secs(30));
    health.update_status(2, RankStatus::Failed);
    assert!(!health.is_healthy());
    assert_eq!(health.unhealthy_ranks(), vec![2]);
}

#[test]
fn test_group_health_all_shut_down() {
    let mut health = TPGroupHealth::new(2, Duration::from_secs(30));
    health.update_status(0, RankStatus::ShutDown);
    health.update_status(1, RankStatus::ShutDown);
    assert!(health.all_shut_down());
}

#[test]
fn test_group_health_heartbeat_timeout() {
    let mut health = TPGroupHealth::new(2, Duration::from_millis(1));
    health.heartbeat(0);
    std::thread::sleep(Duration::from_millis(5));
    // Rank 0 and 1 should both be timed out since we slept
    assert!(!health.is_healthy());
}

#[test]
fn test_group_health_rank_status() {
    let mut health = TPGroupHealth::new(2, Duration::from_secs(30));
    assert_eq!(health.rank_status(0), Some(RankStatus::Ready));
    health.update_status(0, RankStatus::Executing);
    assert_eq!(health.rank_status(0), Some(RankStatus::Executing));
    assert_eq!(health.rank_status(99), None);
}

// ---------------------------------------------------------------------------
// SampledTokens tests
// ---------------------------------------------------------------------------

#[test]
fn test_sampled_tokens_serialize_roundtrip() {
    let original = SampledTokens {
        step_id: 10,
        tokens: vec![42, 99, 7],
        completed_seq_ids: vec![2],
    };
    let json = serde_json::to_string(&original).unwrap();
    let deserialized: SampledTokens = serde_json::from_str(&json).unwrap();
    assert_eq!(original, deserialized);
}
