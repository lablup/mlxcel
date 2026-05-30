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

use super::*;

/// Helper to create a test config for a 2-stage pipeline.
fn test_config(stage_index: u32, layer_range: Range<usize>) -> PipelineCacheConfig {
    PipelineCacheConfig {
        stage_index,
        num_stages: 2,
        layer_range,
        max_sequences: 4,
        memory_budget_bytes: 1_000_000,  // 1 MB
        bytes_per_layer_per_token: 1024, // 1 KB per layer per token
        pressure_threshold: 0.8,
    }
}

// --- PipelineCacheConfig tests ---

#[test]
fn config_num_layers() {
    let cfg = test_config(0, 0..16);
    assert_eq!(cfg.num_layers(), 16);
}

#[test]
fn config_estimate_memory() {
    let cfg = test_config(0, 0..8);
    // 8 layers * 100 tokens * 1024 bytes = 819200 bytes
    assert_eq!(cfg.estimate_memory(100), 8 * 100 * 1024);
}

#[test]
fn config_validate_ok() {
    let cfg = test_config(0, 0..16);
    assert!(cfg.validate().is_ok());
}

#[test]
fn config_validate_bad_stage() {
    let mut cfg = test_config(0, 0..16);
    cfg.stage_index = 5; // out of range for 2-stage pipeline
    assert!(cfg.validate().is_err());
}

#[test]
fn config_validate_empty_range() {
    let cfg = test_config(0, 4..4);
    assert!(cfg.validate().is_err());
}

#[test]
fn config_validate_bad_threshold() {
    let mut cfg = test_config(0, 0..8);
    cfg.pressure_threshold = 1.5;
    assert!(cfg.validate().is_err());
}

// --- PipelineCacheManager basic tests ---

#[test]
fn manager_creation() {
    let mgr = PipelineCacheManager::new(test_config(0, 0..16)).unwrap();
    assert_eq!(mgr.stage_index(), 0);
    assert_eq!(mgr.num_layers(), 16);
    assert_eq!(mgr.active_sequences(), 0);
    assert_eq!(mgr.used_memory(), 0);
    assert_eq!(mgr.available_memory(), 1_000_000);
}

#[test]
fn admission_basic() {
    let mut mgr = PipelineCacheManager::new(test_config(0, 0..8)).unwrap();
    let req = CacheAdmissionRequest::new(1, 10);

    let decision = mgr.request_admission(&req);
    assert_eq!(decision, AdmissionDecision::Admitted);
    assert_eq!(mgr.active_sequences(), 1);

    // Memory: 8 layers * 10 tokens * 1024 = 81920 bytes
    assert_eq!(mgr.used_memory(), 8 * 10 * 1024);
}

#[test]
fn admission_duplicate_rejected() {
    let mut mgr = PipelineCacheManager::new(test_config(0, 0..8)).unwrap();
    let req = CacheAdmissionRequest::new(1, 10);

    assert_eq!(mgr.request_admission(&req), AdmissionDecision::Admitted);
    assert_eq!(
        mgr.request_admission(&req),
        AdmissionDecision::Rejected(RejectionReason::AlreadyCached)
    );
}

#[test]
fn admission_max_sequences_rejected() {
    let mut mgr = PipelineCacheManager::new(test_config(0, 0..4)).unwrap();
    // max_sequences = 4
    for i in 0..4 {
        let req = CacheAdmissionRequest::new(i, 1);
        assert_eq!(mgr.request_admission(&req), AdmissionDecision::Admitted);
    }
    let req = CacheAdmissionRequest::new(99, 1);
    assert_eq!(
        mgr.request_admission(&req),
        AdmissionDecision::Rejected(RejectionReason::MaxSequencesReached)
    );
}

#[test]
fn admission_insufficient_memory() {
    let mut cfg = test_config(0, 0..8);
    cfg.memory_budget_bytes = 10_000; // very small budget
    let mut mgr = PipelineCacheManager::new(cfg).unwrap();

    // 8 layers * 100 tokens * 1024 = 819200 >> 10000
    let req = CacheAdmissionRequest::new(1, 100);
    match mgr.request_admission(&req) {
        AdmissionDecision::Rejected(RejectionReason::InsufficientMemory { .. }) => {}
        other => panic!("expected InsufficientMemory, got {other:?}"),
    }
}

#[test]
fn admission_with_estimated_max_tokens() {
    let mut cfg = test_config(0, 0..4);
    cfg.memory_budget_bytes = 100_000;
    let mut mgr = PipelineCacheManager::new(cfg).unwrap();

    // prompt=10, estimated_max=200 -> effective=210
    // 4 layers * 210 tokens * 1024 = 860160 > 100000
    let req = CacheAdmissionRequest::new(1, 10).with_estimated_max_tokens(200);
    match mgr.request_admission(&req) {
        AdmissionDecision::Rejected(RejectionReason::InsufficientMemory { .. }) => {}
        other => panic!("expected InsufficientMemory, got {other:?}"),
    }
}

// --- Eviction tests ---

#[test]
fn eviction_basic() {
    let mut mgr = PipelineCacheManager::new(test_config(0, 0..8)).unwrap();
    let req = CacheAdmissionRequest::new(42, 10);
    mgr.request_admission(&req);

    let event = mgr.evict(42, EvictionReason::SequenceComplete).unwrap();
    assert_eq!(event.sequence_id, 42);
    assert_eq!(event.initiating_stage, 0);
    assert_eq!(event.reason, EvictionReason::SequenceComplete);
    assert_eq!(mgr.active_sequences(), 0);
    assert_eq!(mgr.used_memory(), 0);
}

#[test]
fn eviction_not_found() {
    let mut mgr = PipelineCacheManager::new(test_config(0, 0..8)).unwrap();
    assert!(mgr.evict(999, EvictionReason::ExplicitRequest).is_err());
}

#[test]
fn eviction_broadcast_idempotent() {
    let mut mgr = PipelineCacheManager::new(test_config(0, 0..8)).unwrap();
    let event = EvictionEvent {
        sequence_id: 42,
        initiating_stage: 1,
        reason: EvictionReason::MemoryPressure,
    };
    // Should succeed even if the sequence is not present.
    assert!(mgr.apply_eviction_broadcast(&event).is_ok());
}

#[test]
fn eviction_broadcast_removes_allocation() {
    let mut mgr = PipelineCacheManager::new(test_config(0, 0..8)).unwrap();
    let req = CacheAdmissionRequest::new(42, 10);
    mgr.request_admission(&req);
    assert_eq!(mgr.active_sequences(), 1);

    let event = EvictionEvent {
        sequence_id: 42,
        initiating_stage: 1,
        reason: EvictionReason::MemoryPressure,
    };
    mgr.apply_eviction_broadcast(&event).unwrap();
    assert_eq!(mgr.active_sequences(), 0);
    assert_eq!(mgr.used_memory(), 0);
}

// --- Memory pressure tests ---

#[test]
fn no_pressure_when_under_threshold() {
    let mut mgr = PipelineCacheManager::new(test_config(0, 0..4)).unwrap();
    // Small allocation: 4 * 1 * 1024 = 4096 << 1_000_000
    let req = CacheAdmissionRequest::new(1, 1);
    mgr.request_admission(&req);
    assert!(mgr.check_memory_pressure().is_none());
}

#[test]
fn pressure_when_above_threshold() {
    let mut cfg = test_config(0, 0..4);
    cfg.memory_budget_bytes = 50_000;
    cfg.pressure_threshold = 0.5;
    let mut mgr = PipelineCacheManager::new(cfg).unwrap();

    // 4 layers * 10 tokens * 1024 = 40960 -> 40960/50000 = 0.819 > 0.5
    let req = CacheAdmissionRequest::new(1, 10);
    mgr.request_admission(&req);
    let signal = mgr.check_memory_pressure();
    assert!(signal.is_some());
    let signal = signal.unwrap();
    assert_eq!(signal.source_stage, 0);
    assert!(signal.memory_usage_fraction > 0.5);
    assert!(!signal.sequence_ids.is_empty());
}

// --- PreemptionPolicy tests ---

#[test]
fn eviction_candidates_lru() {
    let mut cfg = test_config(0, 0..1);
    cfg.memory_budget_bytes = 100_000;
    cfg.max_sequences = 10;
    let mut mgr = PipelineCacheManager::new(cfg).unwrap();

    // Admit sequences; the order of last_accessed is deterministic
    // since they are created in sequence within the same test.
    for i in 1..=3 {
        let req = CacheAdmissionRequest::new(i, 1);
        mgr.request_admission(&req);
    }

    // Touch sequence 1 to make it most recently used.
    mgr.allocations.get_mut(&1).unwrap().touch();

    let candidates = mgr.select_eviction_candidates();
    // Seq 2 and 3 were not touched, so they should come before seq 1.
    // The exact order between 2 and 3 depends on creation time.
    assert_eq!(*candidates.last().unwrap(), 1);
}

#[test]
fn eviction_candidates_shortest() {
    let mut cfg = test_config(0, 0..1);
    cfg.memory_budget_bytes = 500_000;
    cfg.max_sequences = 10;
    let mut mgr = PipelineCacheManager::new(cfg)
        .unwrap()
        .with_preemption_policy(PreemptionPolicy::Shortest);

    let req1 = CacheAdmissionRequest::new(1, 100);
    let req2 = CacheAdmissionRequest::new(2, 5);
    let req3 = CacheAdmissionRequest::new(3, 50);
    mgr.request_admission(&req1);
    mgr.request_admission(&req2);
    mgr.request_admission(&req3);

    let candidates = mgr.select_eviction_candidates();
    // Shortest first: seq 2 (5), seq 3 (50), seq 1 (100)
    assert_eq!(candidates[0], 2);
    assert_eq!(candidates[1], 3);
    assert_eq!(candidates[2], 1);
}

#[test]
fn eviction_candidates_longest() {
    let mut cfg = test_config(0, 0..1);
    cfg.memory_budget_bytes = 500_000;
    cfg.max_sequences = 10;
    let mut mgr = PipelineCacheManager::new(cfg)
        .unwrap()
        .with_preemption_policy(PreemptionPolicy::Longest);

    let req1 = CacheAdmissionRequest::new(1, 100);
    let req2 = CacheAdmissionRequest::new(2, 5);
    let req3 = CacheAdmissionRequest::new(3, 50);
    mgr.request_admission(&req1);
    mgr.request_admission(&req2);
    mgr.request_admission(&req3);

    let candidates = mgr.select_eviction_candidates();
    // Longest first: seq 1 (100), seq 3 (50), seq 2 (5)
    assert_eq!(candidates[0], 1);
    assert_eq!(candidates[1], 3);
    assert_eq!(candidates[2], 2);
}

// --- Metadata sync tests ---

#[test]
fn metadata_sync_updates_offset() {
    let mut mgr = PipelineCacheManager::new(test_config(0, 0..8)).unwrap();
    let req = CacheAdmissionRequest::new(1, 10);
    mgr.request_admission(&req);

    let sync = CacheMetadataSync {
        sequence_id: 1,
        current_offset: 50,
        prompt_len: 10,
        is_active: true,
        source_stage: 1, // from another stage
    };
    mgr.apply_metadata_sync(&sync).unwrap();

    let alloc = mgr.get_allocation(1).unwrap();
    assert_eq!(alloc.current_offset, 50);
    // Memory: 8 layers * 50 tokens * 1024 = 409600
    assert_eq!(alloc.allocated_memory_bytes, 8 * 50 * 1024);
}

#[test]
fn metadata_sync_inactive_evicts() {
    let mut mgr = PipelineCacheManager::new(test_config(0, 0..8)).unwrap();
    let req = CacheAdmissionRequest::new(1, 10);
    mgr.request_admission(&req);

    let sync = CacheMetadataSync {
        sequence_id: 1,
        current_offset: 10,
        prompt_len: 10,
        is_active: false,
        source_stage: 1,
    };
    mgr.apply_metadata_sync(&sync).unwrap();
    assert_eq!(mgr.active_sequences(), 0);
}

#[test]
fn generate_metadata_sync() {
    let mut mgr = PipelineCacheManager::new(test_config(0, 0..8)).unwrap();
    let req = CacheAdmissionRequest::new(42, 20);
    mgr.request_admission(&req);

    let sync = mgr.generate_metadata_sync(42).unwrap();
    assert_eq!(sync.sequence_id, 42);
    assert_eq!(sync.current_offset, 20);
    assert_eq!(sync.prompt_len, 20);
    assert!(sync.is_active);
    assert_eq!(sync.source_stage, 0);
}

#[test]
fn generate_metadata_sync_missing() {
    let mgr = PipelineCacheManager::new(test_config(0, 0..8)).unwrap();
    assert!(mgr.generate_metadata_sync(999).is_none());
}

// --- Coordinated admission tests ---

#[test]
fn coordinated_admission_all_accept() {
    let mut mgr0 = PipelineCacheManager::new(test_config(0, 0..16)).unwrap();
    let mut mgr1 = PipelineCacheManager::new(test_config(1, 16..32)).unwrap();

    let req = CacheAdmissionRequest::new(1, 10);
    let decision = coordinated_admission(&mut [&mut mgr0, &mut mgr1], &req).unwrap();
    assert_eq!(decision, AdmissionDecision::Admitted);
    assert_eq!(mgr0.active_sequences(), 1);
    assert_eq!(mgr1.active_sequences(), 1);
}

#[test]
fn coordinated_admission_one_rejects() {
    let mut cfg1 = test_config(1, 16..32);
    cfg1.memory_budget_bytes = 1; // stage 1 has almost no memory
    let mut mgr0 = PipelineCacheManager::new(test_config(0, 0..16)).unwrap();
    let mut mgr1 = PipelineCacheManager::new(cfg1).unwrap();

    let req = CacheAdmissionRequest::new(1, 10);
    let decision = coordinated_admission(&mut [&mut mgr0, &mut mgr1], &req).unwrap();
    match decision {
        AdmissionDecision::Rejected(RejectionReason::InsufficientMemory { .. }) => {}
        other => panic!("expected InsufficientMemory rejection, got {other:?}"),
    }
    // Neither stage should have admitted the sequence.
    assert_eq!(mgr0.active_sequences(), 0);
    assert_eq!(mgr1.active_sequences(), 0);
}

// --- Broadcast eviction tests ---

#[test]
fn broadcast_eviction_all_stages() {
    let mut mgr0 = PipelineCacheManager::new(test_config(0, 0..16)).unwrap();
    let mut mgr1 = PipelineCacheManager::new(test_config(1, 16..32)).unwrap();

    let req = CacheAdmissionRequest::new(1, 10);
    coordinated_admission(&mut [&mut mgr0, &mut mgr1], &req).unwrap();

    let event = broadcast_eviction(
        &mut [&mut mgr0, &mut mgr1],
        1,
        0,
        EvictionReason::SequenceComplete,
    )
    .unwrap();

    assert_eq!(event.sequence_id, 1);
    assert_eq!(mgr0.active_sequences(), 0);
    assert_eq!(mgr1.active_sequences(), 0);
}

// --- Sync metadata tests ---

#[test]
fn sync_metadata_across_stages() {
    let mut mgr0 = PipelineCacheManager::new(test_config(0, 0..16)).unwrap();
    let mut mgr1 = PipelineCacheManager::new(test_config(1, 16..32)).unwrap();

    let req = CacheAdmissionRequest::new(1, 10);
    coordinated_admission(&mut [&mut mgr0, &mut mgr1], &req).unwrap();

    // Stage 0 advances offset to 50 and syncs.
    let sync = CacheMetadataSync {
        sequence_id: 1,
        current_offset: 50,
        prompt_len: 10,
        is_active: true,
        source_stage: 0,
    };
    sync_metadata(&mut [&mut mgr0, &mut mgr1], &sync).unwrap();

    // Stage 1 should have updated offset; stage 0 should be unchanged
    // (source stage is skipped).
    let alloc1 = mgr1.get_allocation(1).unwrap();
    assert_eq!(alloc1.current_offset, 50);
}

// --- Pipeline pressure tests ---

#[test]
fn check_pipeline_pressure_none() {
    let mgr0 = PipelineCacheManager::new(test_config(0, 0..16)).unwrap();
    let mgr1 = PipelineCacheManager::new(test_config(1, 16..32)).unwrap();
    assert!(check_pipeline_pressure(&[&mgr0, &mgr1]).is_none());
}

#[test]
fn check_pipeline_pressure_one_stage() {
    let mut cfg0 = test_config(0, 0..4);
    cfg0.memory_budget_bytes = 50_000;
    cfg0.pressure_threshold = 0.5;
    let mut mgr0 = PipelineCacheManager::new(cfg0).unwrap();
    let mgr1 = PipelineCacheManager::new(test_config(1, 4..8)).unwrap();

    // Fill stage 0 above threshold.
    let req = CacheAdmissionRequest::new(1, 10);
    mgr0.request_admission(&req);

    let signal = check_pipeline_pressure(&[&mgr0, &mgr1]);
    assert!(signal.is_some());
    assert_eq!(signal.unwrap().source_stage, 0);
}

// --- Display tests ---

#[test]
fn display_cache_manager() {
    let mgr = PipelineCacheManager::new(test_config(0, 0..16)).unwrap();
    let display = format!("{mgr}");
    assert!(display.contains("CacheManager"));
    assert!(display.contains("stage=0"));
}

#[test]
fn display_eviction_event() {
    let event = EvictionEvent {
        sequence_id: 42,
        initiating_stage: 0,
        reason: EvictionReason::MemoryPressure,
    };
    let display = format!("{event}");
    assert!(display.contains("seq=42"));
    assert!(display.contains("memory_pressure"));
}

#[test]
fn display_preemption_signal() {
    let signal = PreemptionSignal {
        sequence_ids: vec![1, 2],
        source_stage: 0,
        memory_usage_fraction: 0.95,
        reason: PreemptionReason::MemoryPressure,
    };
    let display = format!("{signal}");
    assert!(display.contains("95.0%"));
    assert!(display.contains("evict=2"));
}

#[test]
fn display_cache_metadata_sync() {
    let sync = CacheMetadataSync {
        sequence_id: 1,
        current_offset: 50,
        prompt_len: 10,
        is_active: true,
        source_stage: 0,
    };
    let display = format!("{sync}");
    assert!(display.contains("seq=1"));
    assert!(display.contains("offset=50"));
}

#[test]
fn display_rejection_reason() {
    let reason = RejectionReason::InsufficientMemory {
        required_bytes: 1000,
        available_bytes: 500,
    };
    let display = format!("{reason}");
    assert!(display.contains("1000"));
    assert!(display.contains("500"));
}

// --- 2D (PP x TP) coordinated admission tests ---

fn pp_tp_config(stage: u32, _rank: u32, layer_range: Range<usize>) -> PipelineCacheConfig {
    // 2 stages, each sharded across 2 TP ranks. The config itself does not
    // carry the TP rank — the rank is tracked by the 2D grid key passed to
    // `coordinated_2d_admission`. Each TP rank holds half the KV shard, so
    // the per-rank budget and bytes-per-layer figures are halved relative to
    // the 1D configuration.
    PipelineCacheConfig {
        stage_index: stage,
        num_stages: 2,
        layer_range,
        max_sequences: 4,
        memory_budget_bytes: 500_000,
        bytes_per_layer_per_token: 512,
        pressure_threshold: 0.8,
    }
}

#[test]
fn pp_tp_coord_display_and_order() {
    let a = PpTpCoord::new(0, 0);
    let b = PpTpCoord::new(1, 0);
    let c = PpTpCoord::new(0, 1);
    assert!(a < c && c < b);
    assert_eq!(a.to_string(), "(stage=0, rank=0)");
}

#[test]
fn coordinated_2d_admission_accepts_when_every_slot_fits() {
    let mut m00 = PipelineCacheManager::new(pp_tp_config(0, 0, 0..8)).unwrap();
    let mut m01 = PipelineCacheManager::new(pp_tp_config(0, 1, 0..8)).unwrap();
    let mut m10 = PipelineCacheManager::new(pp_tp_config(1, 0, 8..16)).unwrap();
    let mut m11 = PipelineCacheManager::new(pp_tp_config(1, 1, 8..16)).unwrap();

    let mut grid: Vec<(PpTpCoord, &mut PipelineCacheManager)> = vec![
        (PpTpCoord::new(0, 0), &mut m00),
        (PpTpCoord::new(0, 1), &mut m01),
        (PpTpCoord::new(1, 0), &mut m10),
        (PpTpCoord::new(1, 1), &mut m11),
    ];

    let req = CacheAdmissionRequest::new(42, 32).with_estimated_max_tokens(32);
    let outcome = coordinated_2d_admission(&mut grid, &req).unwrap();
    assert_eq!(outcome, PpTpAdmissionOutcome::Admitted);
    // Every slot now holds the allocation.
    assert!(m00.get_allocation(42).is_some());
    assert!(m01.get_allocation(42).is_some());
    assert!(m10.get_allocation(42).is_some());
    assert!(m11.get_allocation(42).is_some());
}

#[test]
fn coordinated_2d_admission_rejects_when_any_slot_is_full_and_keeps_grid_empty() {
    let mut m00 = PipelineCacheManager::new(pp_tp_config(0, 0, 0..8)).unwrap();
    // m01 is saturated: its memory budget is 500_000 but we admit a giant
    // sequence to exhaust it.
    let mut full_cfg = pp_tp_config(0, 1, 0..8);
    full_cfg.max_sequences = 1;
    let mut m01 = PipelineCacheManager::new(full_cfg).unwrap();
    m01.request_admission(&CacheAdmissionRequest::new(1, 8));

    let mut m10 = PipelineCacheManager::new(pp_tp_config(1, 0, 8..16)).unwrap();
    let mut m11 = PipelineCacheManager::new(pp_tp_config(1, 1, 8..16)).unwrap();

    let mut grid: Vec<(PpTpCoord, &mut PipelineCacheManager)> = vec![
        (PpTpCoord::new(0, 0), &mut m00),
        (PpTpCoord::new(0, 1), &mut m01),
        (PpTpCoord::new(1, 0), &mut m10),
        (PpTpCoord::new(1, 1), &mut m11),
    ];

    let req = CacheAdmissionRequest::new(42, 32);
    let outcome = coordinated_2d_admission(&mut grid, &req).unwrap();
    match outcome {
        PpTpAdmissionOutcome::Rejected { at, reason } => {
            assert_eq!(at, PpTpCoord::new(0, 1));
            assert_eq!(reason, RejectionReason::MaxSequencesReached);
        }
        _ => panic!("expected Rejected"),
    }
    // Crucially: no other slot was mutated.
    assert!(m00.get_allocation(42).is_none());
    assert!(m10.get_allocation(42).is_none());
    assert!(m11.get_allocation(42).is_none());
}

#[test]
fn broadcast_2d_eviction_clears_all_slots() {
    let mut m00 = PipelineCacheManager::new(pp_tp_config(0, 0, 0..8)).unwrap();
    let mut m01 = PipelineCacheManager::new(pp_tp_config(0, 1, 0..8)).unwrap();
    let mut m10 = PipelineCacheManager::new(pp_tp_config(1, 0, 8..16)).unwrap();
    let mut m11 = PipelineCacheManager::new(pp_tp_config(1, 1, 8..16)).unwrap();

    {
        let mut grid: Vec<(PpTpCoord, &mut PipelineCacheManager)> = vec![
            (PpTpCoord::new(0, 0), &mut m00),
            (PpTpCoord::new(0, 1), &mut m01),
            (PpTpCoord::new(1, 0), &mut m10),
            (PpTpCoord::new(1, 1), &mut m11),
        ];
        let req = CacheAdmissionRequest::new(7, 16);
        coordinated_2d_admission(&mut grid, &req).unwrap();
    }

    let mut grid: Vec<(PpTpCoord, &mut PipelineCacheManager)> = vec![
        (PpTpCoord::new(0, 0), &mut m00),
        (PpTpCoord::new(0, 1), &mut m01),
        (PpTpCoord::new(1, 0), &mut m10),
        (PpTpCoord::new(1, 1), &mut m11),
    ];
    let event = broadcast_2d_eviction(
        &mut grid,
        7,
        PpTpCoord::new(0, 0),
        EvictionReason::ExplicitRequest,
    )
    .unwrap();
    assert_eq!(event.sequence_id, 7);
    assert_eq!(event.initiating_stage, 0);
    assert!(m00.get_allocation(7).is_none());
    assert!(m01.get_allocation(7).is_none());
    assert!(m10.get_allocation(7).is_none());
    assert!(m11.get_allocation(7).is_none());
}

// -----------------------------------------------------------------------------
// enriched OOM diagnostic via coordinated_admission_with_attribution.
// -----------------------------------------------------------------------------

mod attribution {
    use super::*;

    fn tight_config(stage_index: u32) -> PipelineCacheConfig {
        PipelineCacheConfig {
            stage_index,
            num_stages: 2,
            layer_range: 0..4,
            max_sequences: 2,
            memory_budget_bytes: 10_000,
            bytes_per_layer_per_token: 1_000,
            pressure_threshold: 0.9,
        }
    }

    #[test]
    fn admission_diagnostic_points_to_offending_stage() {
        // Stage 0 has a generous budget; stage 1 is tight. A coordinated
        // admission that overflows only on stage 1 must attribute to stage 1.
        let fat_cfg = PipelineCacheConfig {
            memory_budget_bytes: 1_000_000,
            ..tight_config(0)
        };
        let mut s0 = PipelineCacheManager::new(fat_cfg).unwrap();
        let mut s1 = PipelineCacheManager::new(tight_config(1)).unwrap();
        // Fill stage 1 with a prior sequence to push it to high occupancy.
        let filler = CacheAdmissionRequest::new(1, 2).with_estimated_max_tokens(0);
        assert_eq!(
            s1.request_admission(&filler),
            AdmissionDecision::Admitted,
            "filler must land on stage 1"
        );

        // Big sequence fits on stage 0 but overflows stage 1's budget.
        let big_req = CacheAdmissionRequest::new(2, 8).with_estimated_max_tokens(0);
        let mut mgrs: Vec<&mut PipelineCacheManager> = vec![&mut s0, &mut s1];
        let outcome = coordinated_admission_with_attribution(&mut mgrs, &big_req).unwrap();
        let diag = outcome.expect_err("admission must be rejected");
        assert_eq!(diag.stage_index, 1, "rejection must attribute to stage 1");
        assert!(
            matches!(diag.reason, RejectionReason::InsufficientMemory { .. }),
            "reason must be memory: {:?}",
            diag.reason
        );
        assert!(
            diag.used_memory_bytes > 0,
            "stage 1 must report prior usage"
        );
        assert_eq!(diag.active_sequences, 1);
        assert_eq!(diag.reason.metric_label(), "memory");
    }

    #[test]
    fn sequence_cap_diagnostic_uses_short_metric_label() {
        let cfg = PipelineCacheConfig {
            max_sequences: 1,
            ..tight_config(0)
        };
        let mut s0 = PipelineCacheManager::new(cfg).unwrap();
        let first = CacheAdmissionRequest::new(1, 1);
        assert_eq!(s0.request_admission(&first), AdmissionDecision::Admitted);

        let second = CacheAdmissionRequest::new(2, 1);
        let mut mgrs: Vec<&mut PipelineCacheManager> = vec![&mut s0];
        let outcome = coordinated_admission_with_attribution(&mut mgrs, &second).unwrap();
        let diag = outcome.expect_err("admission must be rejected");
        assert_eq!(diag.reason.metric_label(), "sequence_cap");
        assert_eq!(diag.active_sequences, 1);
    }

    #[test]
    fn successful_coordinated_admission_returns_ok() {
        let mut s0 = PipelineCacheManager::new(tight_config(0)).unwrap();
        let mut s1 = PipelineCacheManager::new(tight_config(1)).unwrap();
        let req = CacheAdmissionRequest::new(1, 1);
        let mut mgrs: Vec<&mut PipelineCacheManager> = vec![&mut s0, &mut s1];
        let outcome = coordinated_admission_with_attribution(&mut mgrs, &req).unwrap();
        assert!(outcome.is_ok(), "admission must succeed");
        assert_eq!(s0.active_sequences(), 1);
        assert_eq!(s1.active_sequences(), 1);
    }
}
