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

//! Tensor parallelism: weight sharding, collective communication, and configuration.
//!
//! This module provides the types and algorithms for distributing model weights
//! across multiple tensor-parallel ranks:
//!
//! - [`ShardStrategy`] — how a weight tensor is split (column/row/expert/vocab/replicated)
//! - [`CommPattern`] — communication required after each sharded operation
//! - [`LayerShardPlan`] — per-weight sharding metadata
//! - [`ModelShardPlan`] — complete model shard plan with layer expansion
//! - [`ShardConfig`] — user-configurable TP options (tp_size, MoE mode, embedding mode)
//! - [`MoeShardMode`] — expert-parallel vs within-expert sharding
//! - [`EmbeddingMode`] — vocab-parallel vs replicated embedding/LM head
//! - [`generate_shard_plan`] — architecture-aware shard plan generator
//! - [`ShardSpec`] — per-rank shard boundaries for a weight tensor
//! - [`ByteRangeSpec`] — byte ranges for efficient partial file reads
//! - [`ShardedMemoryReport`] — memory accounting across TP ranks
//! - [`compute_shard_spec`] — determine shard boundaries for a weight on a rank
//! - [`shard_tensor_data`] — extract shard bytes from raw tensor data
//! - [`compute_byte_ranges`] — byte-range specs for safetensors partial reads
//! - [`validate_sharded_memory`] — verify memory consistency across ranks
//! - [`all_reduce_sum`] — ring all-reduce (sum) across TP ranks
//! - [`all_gather`] — gather sharded outputs into full tensor
//! - [`reduce_scatter`] — scatter-reduce for bandwidth-efficient communication
//! - [`CollectiveGroup`] — group of ranks with exchange function
//! - [`RingTopology`] — ring neighbor computation
//! - [`BenchmarkResult`] — benchmark measurement container
//! - [`TPAttentionConfig`] — per-model attention parallelism parameters
//! - [`TPAttentionMetadata`] — per-rank attention shapes for the forward pass
//! - [`KVAssignment`] — GQA-aware KV head mapping (sharded or replicated)
//! - [`AttentionType`] — MHA / GQA / MQA classification
//! - [`head_assignment`] — Q head range for a rank
//! - [`kv_head_assignment`] — KV head mapping with GQA replication
//! - [`compute_local_attention_shapes`] — derive per-rank attention dimensions
//! - [`verify_head_coverage`] — validate all heads are covered across ranks
//! - [`FFNActivationType`] — supported FFN activation functions
//! - [`TPFFNConfig`] — per-model FFN parallelism parameters
//! - [`TPFFNMetadata`] — per-rank FFN shape information
//! - [`compute_local_ffn_shapes`] — derive per-rank FFN dimensions
//! - [`verify_intermediate_coverage`] — validate FFN intermediate coverage
//! - [`MoEParallelMode`] — expert-parallel vs within-expert sharding
//! - [`TPMoEConfig`] — per-model MoE parallelism parameters
//! - [`TPMoEMetadata`] — per-rank MoE shape information
//! - [`ExpertAssignment`] — which experts a rank owns
//! - [`compute_expert_assignment`] — round-robin expert-to-rank mapping
//! - [`compute_local_moe_shapes`] — derive per-rank MoE dimensions
//! - [`verify_expert_coverage`] — validate all experts are covered
//! - [`TPCacheConfig`] — per-rank KV cache configuration
//! - [`TPCacheManager`] — per-rank sharded KV cache tracking and eviction
//! - [`ShardedCacheAllocation`] — per-sequence cache allocation on one rank
//! - [`TPCacheMemoryReport`] — per-rank memory accounting
//! - [`EvictionSignal`] — coordinated eviction broadcast from rank 0
//! - [`EvictionPolicy`] — LRU or LeastTokens eviction strategy
//! - [`compute_per_rank_cache_size`] — estimate memory per sequence per rank
//! - [`coordinate_eviction`] — synchronized eviction across TP ranks
//! - [`TPExecutionConfig`] — per-rank execution parameters for synchronized TP
//! - [`StepDecision`] — scheduling decisions broadcast from rank 0
//! - [`SamplingMode`] — vocab-parallel or replicated LM head sampling
//! - [`TPBarrier`] — barrier synchronization with timeout and deadlock prevention
//! - [`RankStatus`] — individual rank health tracking
//! - [`TPGroupHealth`] — aggregate TP group health
//! - [`TPScheduler`] — rank 0 continuous batching scheduler
//! - [`TPExecutor`] — per-rank lockstep execution engine
//! - [`TPBenchmarkConfig`] — benchmark parameters (tp_sizes, model sizes, etc.)
//! - [`TPBenchmarkResult`] — throughput, TTFT, ITL, all-reduce overhead, scaling efficiency
//! - [`AllReduceProfile`] — communication overhead breakdown
//! - [`CrossoverAnalysis`] — model size vs TP benefit analysis
//! - [`ScalingAnalysis`] — scaling efficiency across TP sizes
//! - [`LockstepBenchmarkResult`] — scheduler/executor lockstep benchmark result
//! - [`run_tp_benchmark`] — execute a single TP benchmark scenario
//! - [`run_scaling_analysis`] — compare performance across TP sizes
//! - [`run_crossover_analysis`] — determine when TP becomes beneficial
//! - [`run_lockstep_benchmark`] — verify scheduler/executor lockstep correctness
//! - [`format_tp_benchmark_report`] — human-readable benchmark report

pub mod benchmark;
pub mod cache_manager;
pub mod collective;
pub mod config;
pub mod inference;
pub mod llama_runtime;
pub mod parallel_attention;
pub mod parallel_ffn;
pub mod parallel_moe;
pub mod plan_generator;
pub mod shard_strategy;
pub mod sharded_loading;
pub mod synchronized;
pub mod tp_executor;
pub mod tp_scheduler;

pub use benchmark::{
    AllReduceProfile, CrossoverAnalysis, CrossoverEntry, LockstepBenchmarkResult, ScalingAnalysis,
    TPBenchmarkConfig, TPBenchmarkResult, format_tp_benchmark_report, run_crossover_analysis,
    run_lockstep_benchmark, run_scaling_analysis, run_tp_benchmark,
};
pub use cache_manager::{
    AggregateMemoryReport, CacheSizeEstimate, EvictionPolicy, EvictionReason as TPEvictionReason,
    EvictionSignal, SequenceId as TPSequenceId, ShardedCacheAllocation, TPCacheConfig,
    TPCacheManager, TPCacheMemoryReport, aggregate_memory_reports, check_tp_pressure,
    collect_memory_reports, compute_per_rank_cache_size, coordinate_eviction,
};
pub use collective::{
    BenchmarkResult, CollectiveConfig, CollectiveGroup, RingTopology, all_gather, all_reduce_sum,
    reduce_scatter, ring_allreduce_data_volume,
};
pub use config::{EmbeddingMode, MoeShardMode, ShardConfig};
pub use inference::{
    TensorParallelPlanSummary, ensure_single_rank_runtime, resolve_model_shard_plan,
    shard_config_from_cli,
};
pub use llama_runtime::{
    TensorParallelErnie45Model, TensorParallelGemma3Model, TensorParallelGemma4Model,
    TensorParallelHunyuanV1DenseModel, TensorParallelLlamaModel, TensorParallelQwen3Model,
    TensorParallelQwen35Model, TensorParallelRuntimeKind, TensorParallelRuntimeSupport,
    validate_supported_runtime,
};
pub use parallel_attention::{
    AttentionType, KVAssignment, TPAttentionConfig, TPAttentionMetadata, compute_all_rank_metadata,
    compute_local_attention_shapes, head_assignment, kv_head_assignment,
    requires_allreduce_after_o_proj, validate_tp_attention_config, verify_head_coverage,
};
pub use parallel_ffn::{
    FFNActivationType, TPFFNConfig, TPFFNMetadata, compute_all_rank_ffn_metadata,
    compute_local_ffn_shapes, validate_tp_ffn_config, verify_intermediate_coverage,
};
pub use parallel_moe::{
    ExpertAssignment, MoEParallelMode, TPMoEConfig, TPMoEMetadata, compute_all_rank_moe_metadata,
    compute_expert_assignment, compute_local_moe_shapes, validate_tp_moe_config,
    verify_expert_coverage,
};
pub use plan_generator::generate_shard_plan;
pub use shard_strategy::{CommPattern, LayerShardPlan, ModelShardPlan, ShardStrategy};
pub use sharded_loading::{
    ByteRangeSpec, ShardSpec, ShardedMemoryReport, compute_byte_ranges, compute_shard_spec,
    compute_sharded_shape, dtype_byte_size, shard_tensor_data, validate_sharded_memory,
};
pub use synchronized::{
    EvictionReason as StepEvictionReason, RankStatus, SampledTokens, SamplingMode, StepDecision,
    StepId, TPBarrier, TPExecutionConfig, TPGroupHealth,
};
pub use tp_executor::{ExecutorSequenceState, StepOutcome, TPExecutor};
pub use tp_scheduler::{SequenceInfo, SequenceState, TPScheduler};
