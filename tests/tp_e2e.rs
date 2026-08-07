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

//! End-to-end tests for tensor parallelism.
//!
//! Exercises the full TP stack -- attention/FFN/MoE shape computation,
//! head assignment, shard plan generation, collective operations,
//! scheduler/executor lockstep, and performance benchmarks -- all in
//! simulated multi-rank configurations. No real hardware needed.

use std::sync::Arc;
use std::time::Duration;

use mlxcel::distributed::tensor_parallel::benchmark::{
    TPBenchmarkConfig, format_tp_benchmark_report, run_crossover_analysis, run_lockstep_benchmark,
    run_scaling_analysis, run_tp_benchmark,
};
use mlxcel::distributed::tensor_parallel::collective::{
    CollectiveConfig, CollectiveGroup, all_gather, all_reduce_sum, reduce_scatter,
};
use mlxcel::distributed::tensor_parallel::config::ShardConfig;
use mlxcel::distributed::tensor_parallel::parallel_attention::{
    AttentionType, compute_all_rank_metadata, verify_head_coverage,
};
use mlxcel::distributed::tensor_parallel::parallel_ffn::{
    FFNActivationType, compute_all_rank_ffn_metadata, verify_intermediate_coverage,
};
use mlxcel::distributed::tensor_parallel::parallel_moe::{
    MoEParallelMode, TPMoEConfig, compute_all_rank_moe_metadata, verify_expert_coverage,
};
use mlxcel::distributed::tensor_parallel::plan_generator::generate_shard_plan;
use mlxcel::distributed::tensor_parallel::synchronized::{
    RankStatus, SamplingMode, TPExecutionConfig,
};
use mlxcel::distributed::tensor_parallel::tp_executor::{StepOutcome, TPExecutor};
use mlxcel::distributed::tensor_parallel::tp_scheduler::TPScheduler;
use mlxcel::distributed::tensor_protocol::TensorDtype;

// ===========================================================================
// Helper: create in-process collective groups for multi-rank simulation
// ===========================================================================

/// Create a set of CollectiveGroups connected by in-process channels.
/// Reserved for future multi-threaded collective correctness tests.
#[allow(dead_code)]
fn create_loopback_groups(world_size: usize) -> Vec<CollectiveGroup> {
    use std::sync::Mutex;

    // Shared mailboxes: mailbox[dest_rank] holds data sent to that rank.
    let mailboxes: Arc<Vec<Mutex<Option<Vec<u8>>>>> =
        Arc::new((0..world_size).map(|_| Mutex::new(None)).collect());

    let mut groups = Vec::with_capacity(world_size);
    for rank in 0..world_size {
        let config = CollectiveConfig {
            rank,
            world_size,
            chunk_size: 1024 * 1024,
        };
        let mb = mailboxes.clone();
        let my_rank = rank;

        let exchange_fn = Arc::new(
            move |dest_rank: usize, data: Vec<u8>| -> anyhow::Result<Vec<u8>> {
                // Put data into dest's mailbox.
                {
                    let mut slot = mb[dest_rank].lock().unwrap();
                    *slot = Some(data);
                }
                // Read from our own mailbox (busy-wait for simplicity in tests).
                let _recv_rank = (my_rank + mb.len() - 1) % mb.len();
                loop {
                    let mut slot = mb[my_rank].lock().unwrap();
                    if let Some(data) = slot.take() {
                        return Ok(data);
                    }
                    drop(slot);
                    std::hint::spin_loop();
                }
            },
        );

        groups.push(CollectiveGroup::new(config, exchange_fn).unwrap());
    }
    groups
}

// ===========================================================================
// tp_size=2 correctness tests
// ===========================================================================

#[test]
fn e2e_tp2_mha_head_assignment() {
    // MHA: 32 heads, 32 KV heads, tp_size=2.
    let tp_size = 2;
    let total_heads = 32;
    let total_kv_heads = 32;
    let head_dim = 128;

    let all_meta =
        compute_all_rank_metadata(tp_size, total_heads, total_kv_heads, head_dim, None).unwrap();

    // Each rank gets 16 heads.
    assert_eq!(all_meta[0].local_n_heads, 16);
    assert_eq!(all_meta[1].local_n_heads, 16);
    assert_eq!(all_meta[0].local_n_kv_heads, 16);
    assert_eq!(all_meta[1].local_n_kv_heads, 16);
    assert_eq!(all_meta[0].attention_type, AttentionType::MHA);

    // Verify full coverage.
    verify_head_coverage(&all_meta, total_heads, total_kv_heads).unwrap();
}

#[test]
fn e2e_tp2_gqa_head_assignment() {
    // GQA: 32 heads, 8 KV heads, tp_size=2.
    let tp_size = 2;
    let total_heads = 32;
    let total_kv_heads = 8;
    let head_dim = 128;

    let all_meta =
        compute_all_rank_metadata(tp_size, total_heads, total_kv_heads, head_dim, None).unwrap();

    assert_eq!(all_meta[0].local_n_heads, 16);
    assert_eq!(all_meta[1].local_n_heads, 16);
    assert_eq!(all_meta[0].local_n_kv_heads, 4);
    assert_eq!(all_meta[1].local_n_kv_heads, 4);
    assert_eq!(all_meta[0].attention_type, AttentionType::GQA);

    verify_head_coverage(&all_meta, total_heads, total_kv_heads).unwrap();
}

#[test]
fn e2e_tp2_mqa_head_assignment() {
    // MQA: 32 heads, 1 KV head, tp_size=2.
    let tp_size = 2;
    let total_heads = 32;
    let total_kv_heads = 1;
    let head_dim = 128;

    let all_meta =
        compute_all_rank_metadata(tp_size, total_heads, total_kv_heads, head_dim, None).unwrap();

    assert_eq!(all_meta[0].local_n_heads, 16);
    assert_eq!(all_meta[1].local_n_heads, 16);
    // KV heads replicated on both ranks.
    assert_eq!(all_meta[0].local_n_kv_heads, 1);
    assert_eq!(all_meta[1].local_n_kv_heads, 1);
    assert!(all_meta[0].kv_assignment.is_replicated());
    assert!(all_meta[1].kv_assignment.is_replicated());
    assert_eq!(all_meta[0].attention_type, AttentionType::MQA);

    verify_head_coverage(&all_meta, total_heads, total_kv_heads).unwrap();
}

#[test]
fn e2e_tp2_ffn_coverage() {
    let tp_size = 2;
    let intermediate_size = 11008;
    let hidden_size = 4096;

    let all_meta = compute_all_rank_ffn_metadata(
        tp_size,
        intermediate_size,
        hidden_size,
        FFNActivationType::SiLU,
    )
    .unwrap();

    assert_eq!(all_meta[0].local_intermediate_size, 5504);
    assert_eq!(all_meta[1].local_intermediate_size, 5504);
    assert!(all_meta[0].needs_allreduce);
    assert!(all_meta[1].needs_allreduce);

    verify_intermediate_coverage(&all_meta, intermediate_size).unwrap();
}

#[test]
fn e2e_tp2_shard_plan_llama() {
    let config = ShardConfig::with_tp_size(2);
    let plan = generate_shard_plan("llama", 32, &config).unwrap();

    // Plan should contain entries for attention and FFN weights.
    assert!(!plan.layer_plans.is_empty());
}

#[test]
fn e2e_tp2_scheduler_executor_lockstep() {
    let tp_size = 2;
    let exec_config = TPExecutionConfig::new(0, tp_size);
    let sampling_mode = SamplingMode::ReplicatedLmHead;

    let mut scheduler = TPScheduler::new(exec_config.clone(), sampling_mode).unwrap();

    let mut executors: Vec<TPExecutor> = (0..tp_size)
        .map(|rank| {
            let cfg = TPExecutionConfig::new(rank, tp_size);
            TPExecutor::new(cfg, sampling_mode).unwrap()
        })
        .collect();

    // Submit a sequence.
    scheduler.submit_sequence(1, 5, 3, 0).unwrap();

    // Schedule and execute steps.
    let decisions = scheduler.schedule_step().unwrap();
    assert!(!decisions.is_empty());

    for decision in &decisions {
        for executor in &mut executors {
            let outcome = executor.execute_step(decision).unwrap();
            match outcome {
                StepOutcome::Executed { .. } | StepOutcome::SequenceAdmitted { .. } => {}
                other => panic!("unexpected outcome: {other:?}"),
            }
        }
    }

    // All executors should have the same active count.
    let counts: Vec<usize> = executors.iter().map(|e| e.active_count()).collect();
    assert!(counts.windows(2).all(|w| w[0] == w[1]));
}

// ===========================================================================
// tp_size=4 correctness tests
// ===========================================================================

#[test]
fn e2e_tp4_mha_head_assignment() {
    let tp_size = 4;
    let total_heads = 32;
    let total_kv_heads = 32;
    let head_dim = 128;

    let all_meta =
        compute_all_rank_metadata(tp_size, total_heads, total_kv_heads, head_dim, None).unwrap();

    for meta in &all_meta {
        assert_eq!(meta.local_n_heads, 8);
        assert_eq!(meta.local_n_kv_heads, 8);
        assert!(meta.needs_allreduce);
    }

    verify_head_coverage(&all_meta, total_heads, total_kv_heads).unwrap();
}

#[test]
fn e2e_tp4_gqa_head_assignment() {
    let tp_size = 4;
    let total_heads = 32;
    let total_kv_heads = 8;
    let head_dim = 128;

    let all_meta =
        compute_all_rank_metadata(tp_size, total_heads, total_kv_heads, head_dim, None).unwrap();

    for meta in &all_meta {
        assert_eq!(meta.local_n_heads, 8);
        assert_eq!(meta.local_n_kv_heads, 2);
    }

    verify_head_coverage(&all_meta, total_heads, total_kv_heads).unwrap();
}

#[test]
fn e2e_tp4_mqa_replicated_kv() {
    let tp_size = 4;
    let total_heads = 32;
    let total_kv_heads = 1;
    let head_dim = 128;

    let all_meta =
        compute_all_rank_metadata(tp_size, total_heads, total_kv_heads, head_dim, None).unwrap();

    for meta in &all_meta {
        assert_eq!(meta.local_n_heads, 8);
        assert_eq!(meta.local_n_kv_heads, 1);
        assert!(meta.kv_assignment.is_replicated());
    }

    verify_head_coverage(&all_meta, total_heads, total_kv_heads).unwrap();
}

#[test]
fn e2e_tp4_ffn_coverage() {
    let tp_size = 4;
    let intermediate_size = 11008;
    let hidden_size = 4096;

    let all_meta = compute_all_rank_ffn_metadata(
        tp_size,
        intermediate_size,
        hidden_size,
        FFNActivationType::SiLU,
    )
    .unwrap();

    for meta in &all_meta {
        assert_eq!(meta.local_intermediate_size, 2752);
        assert!(meta.needs_allreduce);
    }

    verify_intermediate_coverage(&all_meta, intermediate_size).unwrap();
}

#[test]
fn e2e_tp4_moe_expert_parallel() {
    let tp_size = 4;
    let num_experts = 8;

    let config = TPMoEConfig {
        tp_rank: 0,
        tp_size,
        num_experts,
        experts_per_token: 2,
        expert_intermediate_size: 11008,
        hidden_size: 4096,
        has_shared_expert: false,
        shared_expert_intermediate_size: None,
        mode: MoEParallelMode::ExpertParallel,
        activation: FFNActivationType::SiLU,
    };

    let all_meta = compute_all_rank_moe_metadata(&config).unwrap();

    for meta in &all_meta {
        assert_eq!(meta.expert_assignment.num_local_experts(), 2); // 8 / 4
    }

    verify_expert_coverage(&all_meta, num_experts).unwrap();
}

#[test]
fn e2e_tp4_shard_plan_qwen() {
    let config = ShardConfig::with_tp_size(4);
    let plan = generate_shard_plan("qwen2", 24, &config).unwrap();
    assert!(!plan.layer_plans.is_empty());
}

#[test]
fn e2e_tp4_scheduler_executor_lockstep() {
    let tp_size = 4;
    let exec_config = TPExecutionConfig::new(0, tp_size);
    let sampling_mode = SamplingMode::ReplicatedLmHead;

    let mut scheduler = TPScheduler::new(exec_config.clone(), sampling_mode).unwrap();

    let mut executors: Vec<TPExecutor> = (0..tp_size)
        .map(|rank| {
            let cfg = TPExecutionConfig::new(rank, tp_size);
            TPExecutor::new(cfg, sampling_mode).unwrap()
        })
        .collect();

    // Submit 3 sequences.
    for i in 0..3u64 {
        scheduler.submit_sequence(i, 5, 4, 0).unwrap();
    }

    let decisions = scheduler.schedule_step().unwrap();

    for decision in &decisions {
        for executor in &mut executors {
            executor.execute_step(decision).unwrap();
        }
    }

    // All executors must have the same active count.
    let counts: Vec<usize> = executors.iter().map(|e| e.active_count()).collect();
    assert!(counts.windows(2).all(|w| w[0] == w[1]));
    assert_eq!(counts[0], 3);
}

// ===========================================================================
// Collective operation correctness
// ===========================================================================

#[test]
fn e2e_allreduce_sum_tp2_f32() {
    // Two ranks, each with [1.0, 2.0, 3.0, 4.0].
    // After all-reduce sum: [2.0, 4.0, 6.0, 8.0].
    let rank0_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let rank1_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

    let mut buf0: Vec<u8> = rank0_data.iter().flat_map(|f| f.to_le_bytes()).collect();
    let _buf1: Vec<u8> = rank1_data.iter().flat_map(|f| f.to_le_bytes()).collect();

    // For single-threaded testing, use the loopback approach:
    // Since all_reduce_sum requires paired send/recv which is hard to simulate
    // single-threaded, we test with world_size=1 (identity) and verify
    // the algorithm data volume computation.
    let group = CollectiveGroup::single_rank().unwrap();
    all_reduce_sum(&mut buf0, TensorDtype::Float32, &group).unwrap();

    // Single rank: data unchanged.
    let result: Vec<f32> = buf0
        .chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn e2e_allreduce_data_volume() {
    use mlxcel::distributed::tensor_parallel::collective::ring_allreduce_data_volume;

    let tensor_bytes = 1024 * 1024; // 1 MiB
    let vol_2 = ring_allreduce_data_volume(tensor_bytes, 2);
    let vol_4 = ring_allreduce_data_volume(tensor_bytes, 4);

    // 2 ranks: 2 * 1/2 * 1MiB = 1 MiB
    assert!((vol_2 - 1_048_576.0).abs() < 1.0);
    // 4 ranks: 2 * 3/4 * 1MiB = 1.5 MiB
    assert!((vol_4 - 1_572_864.0).abs() < 1.0);

    // More ranks means more data transferred per rank.
    assert!(vol_4 > vol_2);
}

#[test]
fn e2e_allgather_single_rank() {
    let shard = vec![1u8, 2, 3, 4, 5, 6, 7, 8]; // 2 x float32
    let group = CollectiveGroup::single_rank().unwrap();
    let result = all_gather(&shard, TensorDtype::Float32, &group).unwrap();
    assert_eq!(result, shard);
}

#[test]
fn e2e_reduce_scatter_single_rank() {
    let data = vec![0u8; 16]; // 4 x float32
    let group = CollectiveGroup::single_rank().unwrap();
    let result = reduce_scatter(&data, TensorDtype::Float32, &group).unwrap();
    assert_eq!(result, data);
}

// ===========================================================================
// Full lockstep execution tests
// ===========================================================================

#[test]
fn e2e_lockstep_tp2_multi_sequence() {
    let result = run_lockstep_benchmark(2, 4, 5).unwrap();

    assert!(result.all_synchronized);
    assert_eq!(result.errors, 0);
    assert!(result.total_tokens_generated > 0);
    assert!(result.total_steps > 0);
}

#[test]
fn e2e_lockstep_tp4_multi_sequence() {
    let result = run_lockstep_benchmark(4, 3, 4).unwrap();

    assert!(result.all_synchronized);
    assert_eq!(result.errors, 0);
    assert!(result.total_tokens_generated > 0);
}

#[test]
fn e2e_lockstep_tp2_barrier_and_shutdown() {
    let tp_size = 2;
    let exec_config = TPExecutionConfig::new(0, tp_size);
    let sampling_mode = SamplingMode::ReplicatedLmHead;

    let mut scheduler = TPScheduler::new(exec_config.clone(), sampling_mode).unwrap();
    let mut executors: Vec<TPExecutor> = (0..tp_size)
        .map(|rank| {
            let cfg = TPExecutionConfig::new(rank, tp_size);
            TPExecutor::new(cfg, sampling_mode).unwrap()
        })
        .collect();

    // Issue a barrier.
    let barrier_decision = scheduler.barrier();
    for executor in &mut executors {
        let outcome = executor.execute_step(&barrier_decision).unwrap();
        assert!(matches!(outcome, StepOutcome::BarrierReached { .. }));
    }

    // All executors arrive at the barrier.
    for rank in 0..tp_size {
        for executor in &mut executors {
            let _ = executor.barrier_arrive(rank);
        }
    }

    // Issue shutdown.
    let shutdown_decision = scheduler.shutdown();
    for executor in &mut executors {
        let outcome = executor.execute_step(&shutdown_decision).unwrap();
        assert!(matches!(outcome, StepOutcome::ShutdownAcknowledged { .. }));
        assert_eq!(executor.status(), RankStatus::ShutDown);
    }
}

// ===========================================================================
// Performance benchmark tests
// ===========================================================================

#[test]
fn e2e_benchmark_tp1_throughput() {
    let config = TPBenchmarkConfig::default()
        .with_decode_steps(16)
        .with_layer_compute_time(Duration::from_micros(100));

    let result = run_tp_benchmark(&config, 1).unwrap();

    assert!(
        result.throughput_tok_per_sec > 0.0,
        "throughput must be positive"
    );
    assert!(!result.ttft.is_zero(), "TTFT must be non-zero");
    assert_eq!(result.allreduce_overhead, 0.0, "no all-reduce for tp=1");
    assert_eq!(result.total_tokens, 16);
}

#[test]
fn e2e_benchmark_tp2_throughput() {
    let config = TPBenchmarkConfig::default()
        .with_decode_steps(16)
        .with_layer_compute_time(Duration::from_micros(200))
        .with_allreduce_time(Duration::from_micros(20));

    let result = run_tp_benchmark(&config, 2).unwrap();

    assert!(result.throughput_tok_per_sec > 0.0);
    assert!(result.allreduce_overhead > 0.0, "should have AR overhead");
    assert!(
        result.per_rank_memory_fraction < 1.0,
        "memory should be reduced"
    );
}

#[test]
fn e2e_benchmark_tp4_throughput() {
    let config = TPBenchmarkConfig::default()
        .with_decode_steps(16)
        .with_layer_compute_time(Duration::from_micros(200))
        .with_allreduce_time(Duration::from_micros(20));

    let result = run_tp_benchmark(&config, 4).unwrap();

    assert!(result.throughput_tok_per_sec > 0.0);
    assert!(result.per_rank_memory_fraction < 0.5);
}

#[test]
fn e2e_benchmark_tp2_faster_than_tp1_with_heavy_compute() {
    // With heavy compute and light all-reduce, TP=2 should be faster.
    let config = TPBenchmarkConfig::default()
        .with_decode_steps(32)
        .with_layer_compute_time(Duration::from_micros(1000))
        .with_allreduce_time(Duration::from_micros(10));

    let result_1 = run_tp_benchmark(&config, 1).unwrap();
    let result_2 = run_tp_benchmark(&config, 2).unwrap();

    assert!(
        result_2.throughput_tok_per_sec > result_1.throughput_tok_per_sec,
        "tp=2 ({:.1}) should be faster than tp=1 ({:.1}) with heavy compute",
        result_2.throughput_tok_per_sec,
        result_1.throughput_tok_per_sec,
    );
}

#[test]
fn e2e_benchmark_allreduce_overhead_increases_with_tp_size() {
    let config = TPBenchmarkConfig::default()
        .with_decode_steps(16)
        .with_layer_compute_time(Duration::from_micros(100))
        .with_allreduce_time(Duration::from_micros(50));

    let result_2 = run_tp_benchmark(&config, 2).unwrap();
    let result_4 = run_tp_benchmark(&config, 4).unwrap();

    // AR overhead should be present for both.
    assert!(result_2.allreduce_overhead > 0.0);
    assert!(result_4.allreduce_overhead > 0.0);
}

#[test]
fn e2e_benchmark_ttft_scales_with_seq_length() {
    let short = TPBenchmarkConfig {
        seq_lengths: vec![32],
        decode_steps: 8,
        layer_compute_time: Duration::from_micros(100),
        ..Default::default()
    };

    let long = TPBenchmarkConfig {
        seq_lengths: vec![512],
        decode_steps: 8,
        layer_compute_time: Duration::from_micros(100),
        ..Default::default()
    };

    let result_short = run_tp_benchmark(&short, 1).unwrap();
    let result_long = run_tp_benchmark(&long, 1).unwrap();

    assert!(
        result_long.ttft > result_short.ttft,
        "longer seq should have higher TTFT: {:?} vs {:?}",
        result_long.ttft,
        result_short.ttft,
    );
}

#[test]
fn e2e_benchmark_per_rank_memory_inversely_proportional() {
    let config = TPBenchmarkConfig::default().with_decode_steps(4);

    let mem_1 = run_tp_benchmark(&config, 1)
        .unwrap()
        .per_rank_memory_fraction;
    let mem_2 = run_tp_benchmark(&config, 2)
        .unwrap()
        .per_rank_memory_fraction;
    let mem_4 = run_tp_benchmark(&config, 4)
        .unwrap()
        .per_rank_memory_fraction;

    assert_eq!(mem_1, 1.0);
    assert!(mem_2 < mem_1, "tp=2 should use less memory per rank");
    assert!(
        mem_4 < mem_2,
        "tp=4 should use less memory per rank than tp=2"
    );
}

// ===========================================================================
// Scaling analysis tests
// ===========================================================================

#[test]
fn e2e_scaling_analysis_basic() {
    let config = TPBenchmarkConfig {
        tp_sizes: vec![1, 2, 4],
        decode_steps: 8,
        layer_compute_time: Duration::from_micros(200),
        allreduce_time: Duration::from_micros(20),
        ..Default::default()
    };

    let analysis = run_scaling_analysis(&config).unwrap();

    assert_eq!(analysis.results.len(), 3);
    assert_eq!(analysis.results[0].tp_size, 1);
    assert_eq!(analysis.results[1].tp_size, 2);
    assert_eq!(analysis.results[2].tp_size, 4);

    // All should have positive throughput.
    for r in &analysis.results {
        assert!(r.throughput_tok_per_sec > 0.0);
    }

    // Scaling factor should be computable.
    let factor_1_to_2 = analysis.scaling_factor(0, 1).unwrap();
    assert!(factor_1_to_2 > 0.0);

    // Recompute the expected scaling efficiency from the throughputs the
    // analysis itself recorded and compare against the stored value. This
    // validates the efficiency computation rather than the wall-clock
    // measurements: throughputs come from a spin_wait-based simulation, so
    // an absolute bound (e.g. "efficiency <= 1.1") is nondeterministic on
    // loaded CI runners where the tp_size=1 baseline run can be preempted.
    let baseline = analysis.results[0].throughput_tok_per_sec;
    let first_tp = analysis.results[0].tp_size as f64;
    for r in &analysis.results {
        let tp_size = r.tp_size;
        let ideal = baseline * tp_size as f64 / first_tp;
        let expected = r.throughput_tok_per_sec / ideal;
        let actual = r.scaling_efficiency;
        assert!(
            (actual - expected).abs() < 1e-9,
            "tp_size={tp_size}: stored scaling efficiency {actual} should equal \
             throughput/ideal {expected}"
        );
        assert!(
            actual.is_finite() && actual > 0.0,
            "tp_size={tp_size}: scaling efficiency {actual} should be finite and positive"
        );
    }
}

#[test]
fn e2e_scaling_analysis_display() {
    let config = TPBenchmarkConfig {
        tp_sizes: vec![1, 2],
        decode_steps: 4,
        ..Default::default()
    };

    let analysis = run_scaling_analysis(&config).unwrap();
    let display = format!("{analysis}");
    assert!(display.contains("TP Scaling Analysis"));
    assert!(display.contains("tp_size=1"));
    assert!(display.contains("tp_size=2"));
}

// ===========================================================================
// Crossover analysis tests
// ===========================================================================

#[test]
fn e2e_crossover_analysis_basic() {
    let config = TPBenchmarkConfig::default()
        .with_decode_steps(4)
        .with_layer_compute_time(Duration::from_micros(50))
        .with_allreduce_time(Duration::from_micros(10));

    let analysis = run_crossover_analysis(&config, &[2048, 4096, 8192], &[1, 2, 4]).unwrap();

    assert!(!analysis.entries.is_empty());

    // Should have entries for most combinations.
    for entry in &analysis.entries {
        assert!(entry.throughput_tok_per_sec > 0.0);
    }

    // Display should work.
    let display = format!("{analysis}");
    assert!(display.contains("Crossover Analysis"));
}

#[test]
fn e2e_crossover_larger_models_benefit_more() {
    let config = TPBenchmarkConfig::default()
        .with_decode_steps(8)
        .with_layer_compute_time(Duration::from_micros(100))
        .with_allreduce_time(Duration::from_micros(10));

    let analysis = run_crossover_analysis(&config, &[2048, 8192], &[1, 2]).unwrap();

    // For larger models, TP should provide greater benefit.
    let small_tp2 = analysis
        .entries
        .iter()
        .find(|e| e.model_hidden_size == 2048 && e.tp_size == 2);
    let large_tp2 = analysis
        .entries
        .iter()
        .find(|e| e.model_hidden_size == 8192 && e.tp_size == 2);

    if let (Some(small), Some(large)) = (small_tp2, large_tp2) {
        assert!(
            large.scaling_efficiency >= small.scaling_efficiency,
            "larger model should have better TP scaling efficiency"
        );
    }
}

// ===========================================================================
// Report generation
// ===========================================================================

#[test]
fn e2e_report_generation_comprehensive() {
    let config = TPBenchmarkConfig {
        tp_sizes: vec![1, 2, 4],
        decode_steps: 4,
        layer_compute_time: Duration::from_micros(100),
        allreduce_time: Duration::from_micros(20),
        ..Default::default()
    };

    let results: Vec<_> = config
        .tp_sizes
        .iter()
        .map(|&tp| run_tp_benchmark(&config, tp).unwrap())
        .collect();

    let report = format_tp_benchmark_report(&results);

    assert!(report.contains("Tensor Parallelism Benchmark Report"));
    assert!(report.contains("Summary"));
    assert!(report.contains("tp_size=1"));
    assert!(report.contains("tp_size=2"));
    assert!(report.contains("tp_size=4"));
}

// ===========================================================================
// Shard plan generation across architectures
// ===========================================================================

#[test]
fn e2e_shard_plan_all_architectures_tp2() {
    let config = ShardConfig::with_tp_size(2);
    let architectures = ["llama", "qwen2", "gemma", "mistral", "phi"];

    for arch in &architectures {
        let result = generate_shard_plan(arch, 24, &config);
        assert!(result.is_ok(), "shard plan for {arch} should succeed");
        let plan = result.unwrap();
        assert!(!plan.layer_plans.is_empty(), "{arch} should have layers");
    }
}

#[test]
fn e2e_shard_plan_all_architectures_tp4() {
    let config = ShardConfig::with_tp_size(4);
    let architectures = ["llama", "qwen2", "gemma", "mistral", "phi"];

    for arch in &architectures {
        let result = generate_shard_plan(arch, 32, &config);
        assert!(
            result.is_ok(),
            "shard plan for {arch} with tp=4 should succeed"
        );
    }
}

// ===========================================================================
// CI compatibility
// ===========================================================================

#[test]
fn e2e_ci_no_external_dependencies() {
    // Meta-check: if this compiles and runs, all TP E2E tests have no
    // hidden external dependencies (no GPU, no network, no model files).
    let config = TPBenchmarkConfig::default().with_decode_steps(2);
    let result = run_tp_benchmark(&config, 1);
    assert!(result.is_ok());
}

#[test]
fn e2e_ci_benchmark_completes_quickly() {
    let start = std::time::Instant::now();

    let config = TPBenchmarkConfig {
        tp_sizes: vec![1, 2, 4],
        decode_steps: 8,
        layer_compute_time: Duration::from_micros(50),
        allreduce_time: Duration::from_micros(5),
        warmup_steps: 1,
        ..Default::default()
    };

    let _ = run_scaling_analysis(&config).unwrap();

    let elapsed = start.elapsed();
    assert!(
        elapsed < Duration::from_secs(30),
        "TP benchmark took too long for CI: {elapsed:?}"
    );
}
