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

/// Helper: create a test config for MHA with the given tp_rank/tp_size.
/// 32 KV heads, head_dim=128, 32 layers, float16 (2 bytes per element).
fn mha_config(tp_rank: usize, tp_size: usize) -> TPCacheConfig {
    TPCacheConfig {
        tp_rank,
        tp_size,
        total_kv_heads: 32,
        head_dim: 128,
        num_layers: 32,
        max_seq_len: 4096,
        max_sequences: 8,
        memory_budget_bytes: 100_000_000, // 100 MB
        bytes_per_element: 2,             // float16
        pressure_threshold: 0.8,
    }
}

/// Helper: create a GQA config (8 KV heads, 32 Q heads implied).
fn gqa_config(tp_rank: usize, tp_size: usize) -> TPCacheConfig {
    TPCacheConfig {
        tp_rank,
        tp_size,
        total_kv_heads: 8,
        head_dim: 128,
        num_layers: 32,
        max_seq_len: 4096,
        max_sequences: 8,
        memory_budget_bytes: 50_000_000, // 50 MB
        bytes_per_element: 2,
        pressure_threshold: 0.8,
    }
}

/// Helper: create an MQA config (1 KV head).
fn mqa_config(tp_rank: usize, tp_size: usize) -> TPCacheConfig {
    TPCacheConfig {
        tp_rank,
        tp_size,
        total_kv_heads: 1,
        head_dim: 128,
        num_layers: 32,
        max_seq_len: 4096,
        max_sequences: 8,
        memory_budget_bytes: 50_000_000,
        bytes_per_element: 2,
        pressure_threshold: 0.8,
    }
}

// --- TPCacheConfig validation ---

#[test]
fn config_validate_ok() {
    assert!(mha_config(0, 4).validate().is_ok());
    assert!(gqa_config(0, 4).validate().is_ok());
    assert!(mqa_config(0, 4).validate().is_ok());
}

#[test]
fn config_validate_bad_rank() {
    let cfg = mha_config(5, 4);
    assert!(cfg.validate().is_err());
}

#[test]
fn config_validate_bad_threshold() {
    let mut cfg = mha_config(0, 4);
    cfg.pressure_threshold = 1.5;
    assert!(cfg.validate().is_err());
}

#[test]
fn config_validate_indivisible_kv_heads() {
    let mut cfg = mha_config(0, 4);
    cfg.total_kv_heads = 7; // not divisible by 4
    assert!(cfg.validate().is_err());
}

// --- local_kv_heads and kv_assignment ---

#[test]
fn mha_local_kv_heads_tp4() {
    // 32 KV heads / 4 ranks = 8 per rank
    let cfg = mha_config(0, 4);
    assert_eq!(cfg.local_kv_heads(), 8);
    assert!(!cfg.kv_assignment().is_replicated());
}

#[test]
fn gqa_local_kv_heads_tp4() {
    // 8 KV heads / 4 ranks = 2 per rank
    let cfg = gqa_config(0, 4);
    assert_eq!(cfg.local_kv_heads(), 2);
    assert!(!cfg.kv_assignment().is_replicated());
}

#[test]
fn gqa_replicated_when_fewer_kv_than_tp() {
    // 8 KV heads with tp_size=16 -> replicated
    let mut cfg = gqa_config(0, 16);
    cfg.total_kv_heads = 8;
    // Validation should pass because 8 < 16 (replicated case)
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.local_kv_heads(), 8); // all heads replicated
    assert!(cfg.kv_assignment().is_replicated());
}

#[test]
fn mqa_replicated_on_all_ranks() {
    // 1 KV head, tp_size=4 -> replicated on all ranks
    for rank in 0..4 {
        let cfg = mqa_config(rank, 4);
        assert_eq!(cfg.local_kv_heads(), 1);
        assert!(cfg.kv_assignment().is_replicated());
    }
}

// --- Memory estimation ---

#[test]
fn bytes_per_token_per_layer_mha() {
    let cfg = mha_config(0, 4);
    // local_kv_heads=8, head_dim=128, float16=2 bytes
    // 2 (K+V) * 8 * 128 * 2 = 4096 bytes
    assert_eq!(cfg.bytes_per_token_per_layer(), 4096);
}

#[test]
fn estimate_memory_mha() {
    let cfg = mha_config(0, 4);
    // 4096 bytes/token/layer * 100 tokens * 32 layers = 13107200
    assert_eq!(cfg.estimate_memory(100), 4096 * 100 * 32);
}

#[test]
fn memory_proportional_to_tp_size() {
    // With tp_size=1, all 32 KV heads are local.
    let cfg1 = mha_config(0, 1);
    // With tp_size=4, 8 KV heads are local.
    let cfg4 = mha_config(0, 4);

    let mem1 = cfg1.estimate_memory(100);
    let mem4 = cfg4.estimate_memory(100);

    // mem4 should be 1/4 of mem1
    assert_eq!(mem4 * 4, mem1);
}

// --- compute_per_rank_cache_size ---

#[test]
fn cache_size_estimate_mha() {
    let cfg = mha_config(0, 4);
    let est = compute_per_rank_cache_size(&cfg);
    assert_eq!(est.local_kv_heads, 8);
    assert_eq!(est.head_dim, 128);
    assert_eq!(est.num_layers, 32);
    assert!(!est.is_replicated);
    assert_eq!(est.bytes_per_token_per_layer, 4096);
    // 100 tokens: 4096 * 100 * 32 = 13107200
    assert_eq!(est.estimate_for_tokens(100), 13_107_200);
}

#[test]
fn cache_size_estimate_mqa_replicated() {
    let cfg = mqa_config(0, 4);
    let est = compute_per_rank_cache_size(&cfg);
    assert_eq!(est.local_kv_heads, 1);
    assert!(est.is_replicated);
    // 2 * 1 * 128 * 2 = 512 bytes/token/layer
    assert_eq!(est.bytes_per_token_per_layer, 512);
}

// --- TPCacheManager creation ---

#[test]
fn manager_creation() {
    let mgr = TPCacheManager::new(mha_config(0, 4)).unwrap();
    assert_eq!(mgr.tp_rank(), 0);
    assert_eq!(mgr.tp_size(), 4);
    assert_eq!(mgr.local_kv_heads(), 8);
    assert!(!mgr.is_replicated());
    assert_eq!(mgr.active_sequences(), 0);
    assert_eq!(mgr.used_memory(), 0);
}

#[test]
fn manager_creation_invalid_config() {
    let cfg = mha_config(10, 4); // rank out of range
    assert!(TPCacheManager::new(cfg).is_err());
}

// --- Cache allocation ---

#[test]
fn allocate_cache_basic() {
    let mut mgr = TPCacheManager::new(mha_config(0, 4)).unwrap();
    let alloc = mgr.allocate_cache(1, 100).unwrap();
    assert_eq!(alloc.sequence_id, 1);
    assert_eq!(alloc.local_kv_heads, 8);
    assert_eq!(alloc.head_dim, 128);
    assert_eq!(alloc.current_offset, 100);
    assert_eq!(alloc.prompt_len, 100);
    assert_eq!(mgr.active_sequences(), 1);
    assert!(mgr.used_memory() > 0);
}

#[test]
fn allocate_cache_duplicate_rejected() {
    let mut mgr = TPCacheManager::new(mha_config(0, 4)).unwrap();
    mgr.allocate_cache(1, 10).unwrap();
    assert!(mgr.allocate_cache(1, 10).is_err());
}

#[test]
fn allocate_cache_max_sequences() {
    let mut cfg = mha_config(0, 4);
    cfg.max_sequences = 2;
    let mut mgr = TPCacheManager::new(cfg).unwrap();
    mgr.allocate_cache(1, 1).unwrap();
    mgr.allocate_cache(2, 1).unwrap();
    assert!(mgr.allocate_cache(3, 1).is_err());
}

#[test]
fn allocate_cache_insufficient_memory() {
    let mut cfg = mha_config(0, 4);
    cfg.memory_budget_bytes = 1000; // very small
    let mut mgr = TPCacheManager::new(cfg).unwrap();
    assert!(mgr.allocate_cache(1, 100).is_err());
}

// --- Free cache ---

#[test]
fn free_cache_basic() {
    let mut mgr = TPCacheManager::new(mha_config(0, 4)).unwrap();
    mgr.allocate_cache(1, 100).unwrap();
    assert_eq!(mgr.active_sequences(), 1);

    mgr.free_cache(1).unwrap();
    assert_eq!(mgr.active_sequences(), 0);
    assert_eq!(mgr.used_memory(), 0);
}

#[test]
fn free_cache_not_found() {
    let mut mgr = TPCacheManager::new(mha_config(0, 4)).unwrap();
    assert!(mgr.free_cache(999).is_err());
}

// --- Update offset ---

#[test]
fn update_offset_increases_memory() {
    let mut mgr = TPCacheManager::new(mha_config(0, 4)).unwrap();
    mgr.allocate_cache(1, 10).unwrap();
    let mem_before = mgr.used_memory();

    mgr.update_offset(1, 50).unwrap();
    let mem_after = mgr.used_memory();
    assert!(mem_after > mem_before);

    let alloc = mgr.get_allocation(1).unwrap();
    assert_eq!(alloc.current_offset, 50);
}

#[test]
fn update_offset_not_found() {
    let mut mgr = TPCacheManager::new(mha_config(0, 4)).unwrap();
    assert!(mgr.update_offset(999, 50).is_err());
}

// --- Memory report ---

#[test]
fn memory_report_basic() {
    let mut mgr = TPCacheManager::new(mha_config(0, 4)).unwrap();
    mgr.allocate_cache(1, 100).unwrap();

    let report = mgr.memory_report();
    assert_eq!(report.tp_rank, 0);
    assert!(report.used_bytes > 0);
    assert_eq!(report.capacity_bytes, 100_000_000);
    assert_eq!(report.total_sequences, 1);
    assert_eq!(report.local_kv_heads, 8);
    assert!(!report.is_replicated);
    assert!(report.utilization > 0.0);
    assert!(report.utilization < 1.0);
}

// --- Pressure and eviction ---

#[test]
fn no_pressure_when_under_threshold() {
    let mut mgr = TPCacheManager::new(mha_config(0, 4)).unwrap();
    mgr.allocate_cache(1, 1).unwrap();
    assert!(mgr.check_pressure().is_none());
}

#[test]
fn pressure_when_above_threshold() {
    let mut cfg = mha_config(0, 4);
    cfg.memory_budget_bytes = 20_000_000; // tighter budget
    cfg.pressure_threshold = 0.5;
    let mut mgr = TPCacheManager::new(cfg).unwrap();

    // 4096 * 1000 * 32 = 131072000 >> 20_000_000 ... let's use smaller seq
    // Actually: 4096 * 100 * 32 = 13107200. 13107200/20000000 = 0.655 > 0.5
    mgr.allocate_cache(1, 100).unwrap();
    let signal = mgr.check_pressure();
    assert!(signal.is_some());
    let signal = signal.unwrap();
    assert_eq!(signal.reason, EvictionReason::MemoryPressure);
    assert!(!signal.sequence_ids.is_empty());
}

#[test]
fn apply_eviction_basic() {
    let mut mgr = TPCacheManager::new(mha_config(0, 4)).unwrap();
    mgr.allocate_cache(1, 10).unwrap();
    mgr.allocate_cache(2, 10).unwrap();
    assert_eq!(mgr.active_sequences(), 2);

    let signal = EvictionSignal {
        sequence_ids: vec![1],
        reason: EvictionReason::SequenceComplete,
        source_rank: 0,
    };
    mgr.apply_eviction(&signal).unwrap();
    assert_eq!(mgr.active_sequences(), 1);
    assert!(mgr.get_allocation(1).is_none());
    assert!(mgr.get_allocation(2).is_some());
}

#[test]
fn apply_eviction_idempotent() {
    let mut mgr = TPCacheManager::new(mha_config(0, 4)).unwrap();
    let signal = EvictionSignal {
        sequence_ids: vec![999],
        reason: EvictionReason::ExplicitRequest,
        source_rank: 0,
    };
    // Should succeed even if sequence is not present.
    assert!(mgr.apply_eviction(&signal).is_ok());
}

// --- Eviction policy ---

#[test]
fn eviction_candidates_lru() {
    let mut mgr = TPCacheManager::new(mha_config(0, 4)).unwrap();
    mgr.allocate_cache(1, 1).unwrap();
    mgr.allocate_cache(2, 1).unwrap();
    mgr.allocate_cache(3, 1).unwrap();

    // Touch seq 1 to make it most recently used.
    mgr.allocations.get_mut(&1).unwrap().touch();

    let candidates = mgr.select_eviction_candidates();
    // Seq 1 should be last (most recently used).
    assert_eq!(*candidates.last().unwrap(), 1);
}

#[test]
fn eviction_candidates_least_tokens() {
    let mut mgr = TPCacheManager::new(mha_config(0, 4))
        .unwrap()
        .with_eviction_policy(EvictionPolicy::LeastTokens);

    mgr.allocate_cache(1, 100).unwrap();
    mgr.allocate_cache(2, 5).unwrap();
    mgr.allocate_cache(3, 50).unwrap();

    let candidates = mgr.select_eviction_candidates();
    // Least tokens first: seq 2 (5), seq 3 (50), seq 1 (100)
    assert_eq!(candidates[0], 2);
    assert_eq!(candidates[1], 3);
    assert_eq!(candidates[2], 1);
}

// --- Eviction tie-break determinism (issue #1286) ---

/// Number of freshly built managers each determinism test iterates over.
///
/// `RandomState` seeds every `HashMap` instance separately, not merely once
/// per process, so probing a single map twice proves nothing. Rebuilding the
/// manager on every iteration is what exercises a different hash iteration
/// order each time.
const DETERMINISM_ITERATIONS: usize = 32;

/// Helper: a tight-budget config where six equally sized allocations put the
/// manager over the pressure threshold and a three-sequence prefix brings it
/// back under the target.
fn tied_pressure_config() -> TPCacheConfig {
    let mut cfg = mha_config(0, 4);
    cfg.memory_budget_bytes = 10_000_000;
    cfg.pressure_threshold = 0.5;
    cfg
}

#[test]
fn eviction_candidates_lru_tie_break_is_deterministic() {
    let ids: [SequenceId; 8] = [7, 3, 8, 1, 6, 2, 5, 4];
    let expected: Vec<SequenceId> = vec![1, 2, 3, 4, 5, 6, 7, 8];

    for _ in 0..DETERMINISM_ITERATIONS {
        let mut mgr = TPCacheManager::new(mha_config(0, 4)).unwrap();
        for &id in &ids {
            mgr.allocate_cache(id, 1).unwrap();
        }
        // `Instant::now()` has nanosecond resolution on Linux and will never
        // tie naturally, so write one shared timestamp into every allocation.
        let tied = Instant::now();
        for alloc in mgr.allocations.values_mut() {
            alloc.last_accessed = tied;
        }

        assert_eq!(mgr.select_eviction_candidates(), expected);
    }
}

#[test]
fn eviction_candidates_least_tokens_tie_break_is_deterministic() {
    let ids: [SequenceId; 8] = [7, 3, 8, 1, 6, 2, 5, 4];
    let expected: Vec<SequenceId> = vec![1, 2, 3, 4, 5, 6, 7, 8];

    for _ in 0..DETERMINISM_ITERATIONS {
        let mut mgr = TPCacheManager::new(mha_config(0, 4))
            .unwrap()
            .with_eviction_policy(EvictionPolicy::LeastTokens);
        // Equal token counts, so `current_offset` ties across all eight.
        for &id in &ids {
            mgr.allocate_cache(id, 10).unwrap();
        }

        assert_eq!(mgr.select_eviction_candidates(), expected);
    }
}

#[test]
fn check_pressure_prefix_is_deterministic_under_ties() {
    let ids: [SequenceId; 6] = [14, 11, 16, 12, 15, 13];
    // Six 10-token allocations use 7_864_320 B against a 10 MB budget with a
    // 0.5 threshold, so `check_pressure` consumes a three-candidate prefix to
    // get projected usage under the 5_000_000 B target. Every candidate ties,
    // so the prefix is exactly the three lowest sequence IDs.
    let expected: Vec<SequenceId> = vec![11, 12, 13];

    for _ in 0..DETERMINISM_ITERATIONS {
        // LeastTokens: identical token counts tie `current_offset`.
        let mut mgr = TPCacheManager::new(tied_pressure_config())
            .unwrap()
            .with_eviction_policy(EvictionPolicy::LeastTokens);
        for &id in &ids {
            mgr.allocate_cache(id, 10).unwrap();
        }
        let signal = mgr
            .check_pressure()
            .expect("six 10-token allocations must exceed the 0.5 threshold");
        assert_eq!(signal.sequence_ids, expected);

        // LRU: the same allocations with one shared `last_accessed`.
        let mut mgr = TPCacheManager::new(tied_pressure_config())
            .unwrap()
            .with_eviction_policy(EvictionPolicy::LRU);
        for &id in &ids {
            mgr.allocate_cache(id, 10).unwrap();
        }
        let tied = Instant::now();
        for alloc in mgr.allocations.values_mut() {
            alloc.last_accessed = tied;
        }
        let signal = mgr
            .check_pressure()
            .expect("six 10-token allocations must exceed the 0.5 threshold");
        assert_eq!(signal.sequence_ids, expected);
    }
}

// --- Coordinated eviction ---

#[test]
fn coordinate_eviction_across_ranks() {
    let mut mgr0 = TPCacheManager::new(mha_config(0, 4)).unwrap();
    let mut mgr1 = TPCacheManager::new(mha_config(1, 4)).unwrap();
    let mut mgr2 = TPCacheManager::new(mha_config(2, 4)).unwrap();
    let mut mgr3 = TPCacheManager::new(mha_config(3, 4)).unwrap();

    // Allocate on all ranks.
    for mgr in [&mut mgr0, &mut mgr1, &mut mgr2, &mut mgr3] {
        mgr.allocate_cache(1, 10).unwrap();
        mgr.allocate_cache(2, 10).unwrap();
    }

    let signal = EvictionSignal {
        sequence_ids: vec![1],
        reason: EvictionReason::SequenceComplete,
        source_rank: 0,
    };

    coordinate_eviction(&mut [&mut mgr0, &mut mgr1, &mut mgr2, &mut mgr3], &signal).unwrap();

    // Seq 1 should be evicted from all ranks.
    for mgr in [&mgr0, &mgr1, &mgr2, &mgr3] {
        assert!(mgr.get_allocation(1).is_none());
        assert!(mgr.get_allocation(2).is_some());
        assert_eq!(mgr.active_sequences(), 1);
    }
}

// --- Aggregate memory reports ---

#[test]
fn aggregate_reports() {
    let mut mgr0 = TPCacheManager::new(mha_config(0, 2)).unwrap();
    let mut mgr1 = TPCacheManager::new(mha_config(1, 2)).unwrap();

    mgr0.allocate_cache(1, 100).unwrap();
    mgr1.allocate_cache(1, 100).unwrap();

    let agg = collect_memory_reports(&[&mgr0, &mgr1]);
    assert_eq!(agg.per_rank.len(), 2);
    assert!(agg.total_used_bytes > 0);
    assert_eq!(agg.total_capacity_bytes, 200_000_000);
    assert!(agg.overall_utilization > 0.0);
    assert!(agg.max_rank_utilization > 0.0);
}

// --- MHA/GQA/MQA sharding correctness ---

#[test]
fn mha_sharding_proportional() {
    // 32 KV heads / 4 ranks = 8 per rank
    // Memory per rank should be 1/4 of single-device
    let single = TPCacheManager::new(mha_config(0, 1)).unwrap();
    let sharded = TPCacheManager::new(mha_config(0, 4)).unwrap();

    assert_eq!(single.local_kv_heads(), 32);
    assert_eq!(sharded.local_kv_heads(), 8);
    assert!(!sharded.is_replicated());
}

#[test]
fn gqa_sharding_proportional() {
    // 8 KV heads / 4 ranks = 2 per rank
    let single = TPCacheManager::new(gqa_config(0, 1)).unwrap();
    let sharded = TPCacheManager::new(gqa_config(0, 4)).unwrap();

    assert_eq!(single.local_kv_heads(), 8);
    assert_eq!(sharded.local_kv_heads(), 2);
    assert!(!sharded.is_replicated());

    // Memory should be proportional
    let cfg_single = gqa_config(0, 1);
    let cfg_sharded = gqa_config(0, 4);
    assert_eq!(
        cfg_sharded.estimate_memory(100) * 4,
        cfg_single.estimate_memory(100)
    );
}

#[test]
fn mqa_replicated_no_memory_savings() {
    // MQA: 1 KV head, replicated on all ranks -> same memory per rank
    let single = TPCacheManager::new(mqa_config(0, 1)).unwrap();
    let sharded = TPCacheManager::new(mqa_config(0, 4)).unwrap();

    // Both have 1 KV head (replicated)
    assert_eq!(single.local_kv_heads(), 1);
    assert_eq!(sharded.local_kv_heads(), 1);
    assert!(sharded.is_replicated());

    // Memory per rank is the same (no savings for MQA)
    let cfg_single = mqa_config(0, 1);
    let cfg_sharded = mqa_config(0, 4);
    assert_eq!(
        cfg_single.estimate_memory(100),
        cfg_sharded.estimate_memory(100)
    );
}

// --- check_tp_pressure ---

#[test]
fn check_tp_pressure_none() {
    let mgr0 = TPCacheManager::new(mha_config(0, 2)).unwrap();
    let mgr1 = TPCacheManager::new(mha_config(1, 2)).unwrap();
    assert!(check_tp_pressure(&[&mgr0, &mgr1]).is_none());
}

#[test]
fn check_tp_pressure_one_rank() {
    let mut cfg0 = mha_config(0, 2);
    cfg0.memory_budget_bytes = 50_000_000;
    cfg0.pressure_threshold = 0.5;
    let mut mgr0 = TPCacheManager::new(cfg0).unwrap();
    let mgr1 = TPCacheManager::new(mha_config(1, 2)).unwrap();

    // tp_size=2, 32 KV heads -> 16 local. 2*16*128*2 = 8192 bytes/tok/layer
    // 8192 * 100 * 32 = 26214400. 26214400/50000000 = 0.524 > 0.5
    mgr0.allocate_cache(1, 100).unwrap();

    let signal = check_tp_pressure(&[&mgr0, &mgr1]);
    assert!(signal.is_some());
    assert_eq!(signal.unwrap().source_rank, 0);
}

// --- Display tests ---

#[test]
fn display_tp_cache_manager() {
    let mgr = TPCacheManager::new(mha_config(0, 4)).unwrap();
    let display = format!("{mgr}");
    assert!(display.contains("TPCacheManager"));
    assert!(display.contains("rank=0"));
    assert!(display.contains("kv_heads=8"));
}

#[test]
fn display_eviction_signal() {
    let signal = EvictionSignal {
        sequence_ids: vec![1, 2],
        reason: EvictionReason::MemoryPressure,
        source_rank: 0,
    };
    let display = format!("{signal}");
    assert!(display.contains("seqs=2"));
    assert!(display.contains("memory_pressure"));
}

#[test]
fn display_sharded_cache_allocation() {
    let now = Instant::now();
    let alloc = ShardedCacheAllocation {
        sequence_id: 42,
        local_kv_heads: 8,
        head_dim: 128,
        allocated_memory_bytes: 1024,
        current_offset: 10,
        prompt_len: 10,
        created_at: now,
        last_accessed: now,
    };
    let display = format!("{alloc}");
    assert!(display.contains("seq=42"));
    assert!(display.contains("kv_heads=8"));
}

#[test]
fn display_cache_size_estimate() {
    let est = CacheSizeEstimate {
        local_kv_heads: 8,
        head_dim: 128,
        num_layers: 32,
        bytes_per_token_per_layer: 4096,
        is_replicated: false,
    };
    let display = format!("{est}");
    assert!(display.contains("kv_heads=8"));
    assert!(display.contains("replicated=false"));
}

#[test]
fn display_memory_report() {
    let report = TPCacheMemoryReport {
        tp_rank: 0,
        used_bytes: 1000,
        capacity_bytes: 10000,
        utilization: 0.1,
        total_sequences: 1,
        local_kv_heads: 8,
        is_replicated: false,
    };
    let display = format!("{report}");
    assert!(display.contains("rank=0"));
    assert!(display.contains("seqs=1"));
}

#[test]
fn display_aggregate_report() {
    let agg = AggregateMemoryReport {
        per_rank: vec![],
        total_used_bytes: 5000,
        total_capacity_bytes: 10000,
        overall_utilization: 0.5,
        max_rank_utilization: 0.6,
    };
    let display = format!("{agg}");
    assert!(display.contains("50.0%"));
    assert!(display.contains("60.0%"));
}

#[test]
fn display_eviction_policy() {
    assert_eq!(format!("{}", EvictionPolicy::LRU), "LRU");
    assert_eq!(format!("{}", EvictionPolicy::LeastTokens), "LeastTokens");
}

#[test]
fn display_eviction_reason() {
    assert_eq!(
        format!("{}", EvictionReason::MemoryPressure),
        "memory_pressure"
    );
    assert_eq!(
        format!("{}", EvictionReason::SequenceComplete),
        "sequence_complete"
    );
    assert_eq!(
        format!("{}", EvictionReason::ExplicitRequest),
        "explicit_request"
    );
}
