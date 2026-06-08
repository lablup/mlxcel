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

//! Distributed inference: node discovery, cluster configuration, transport,
//! health monitoring, and request scheduling.
//!
//! This module provides the building blocks for multi-node inference:
//!
//! - [`config`] — `NodeRole`, `ClusterConfig`, and TOML parsing
//! - [`cluster_init`] — zero-config multi-machine pipeline bring-up (LAN discovery, port allocation, deterministic TOML emission)
//! - [`registry`] — thread-safe runtime node registry
//! - [`discovery`] — static peer discovery and health probing
//! - [`transport`] — abstract transport trait with streaming and RPC modes
//! - [`tcp_transport`] — TCP backend with connection pooling
//! - [`thunderbolt_transport`] — Thunderbolt Bridge backend built on the shared TCP transport core
//! - [`rdma_transport`] — RDMA-aware backend that negotiates OS-level zero-copy primitives (`io_uring` on Linux, `kqueue`/`writev` registered regions on macOS) and falls back transparently to TCP with a single logged reason
//! - [`rdma_capabilities`] — OS-specific capability probe and protocol-version negotiation for the RDMA-aware transport
//! - [`connection_pool`] — connection pooling with reconnection logic
//! - [`heartbeat`] — heartbeat protocol with configurable interval and failure threshold
//! - [`failure_detector`] — threshold-based failure detection integrated with node registry
//! - [`metrics`] — per-node metrics collection (throughput, latency, memory, network)
//! - [`correlation`] — cluster-wide correlation IDs for cross-node request tracing
//! - [`bench`] — throughput and latency benchmarking harness
//! - [`scheduler`] — distributed scheduler coordinator with pluggable routing
//! - [`routing`] — trait-based routing strategies (role-based, load-balanced, round-robin, PP)
//! - [`request_tracker`] — request lifecycle tracking with unique IDs and state transitions
//! - [`backpressure`] — per-node load tracking with configurable thresholds and overflow policies
//! - [`handoff_queue`] — bounded cross-node request handoff queues
//! - [`pipeline`] — layer partitioning, activation transfer, and configuration for pipeline parallelism
//! - [`tensor_parallel`] — weight sharding strategy and configuration for tensor parallelism
//! - [`kv_cache_transfer`] — optimized KV cache transfer (streamed, quantized, parallel)
//! - [`disaggregated`] — prefill/decode separation: prefill scheduler, decode scheduler, request router, handoff protocol, chunked prefill, API server integration, SSE stream bridging

pub mod backpressure;
pub mod bench;
pub mod bench_activation;
pub mod cluster_init;
pub mod config;
pub mod connection_pool;
pub mod correlation;
pub mod disaggregated;
pub mod discovery;
pub mod failure_detector;
pub mod handoff_queue;
pub mod heartbeat;
pub mod kv_cache_serde;
pub mod kv_cache_transfer;
pub mod metrics;
#[cfg(any(test, feature = "test-utils"))]
pub mod mock_transport;
pub mod pipeline;
pub mod rdma_capabilities;
pub mod rdma_transport;
pub mod registry;
pub mod request_tracker;
pub mod routing;
pub mod scheduler;
pub mod tcp_transport;
pub mod tensor_chunked;
pub mod tensor_compress;
pub mod tensor_parallel;
pub mod tensor_protocol;
pub mod tensor_quantize;
pub mod tensor_serialize;
#[cfg(any(test, feature = "test-utils"))]
pub mod test_harness;
pub mod thunderbolt_transport;
pub mod transport;
pub mod transport_factory;

pub use backpressure::{
    BackpressureConfig, BackpressureMonitor, BackpressurePolicy, BackpressureSignal, LoadLevel,
};
pub use bench_activation::{
    ActivationBenchConfig, ActivationTransferReport, DrainHandle, measure_activation_transfer,
    render_markdown_report, spawn_drain_task,
};
pub use cluster_init::{
    ClusterDiscoveryMode, ClusterInitPlan, ClusterInitRequest, DEFAULT_CONTROL_BASE_PORT,
    DEFAULT_DISCOVERY_PORT, DEFAULT_DISCOVERY_TIMEOUT, DiscoveryBeacon, allocate_data_ports,
    broadcast_beacon_loop, discover_peers, is_port_available, plan_cluster,
    render_deterministic_toml, write_plan_toml,
};
pub use config::{ClusterConfig, ClusterMeta, NodeConfig, NodeResources, NodeRole};
pub use connection_pool::{ConnectionPool, PoolConfig, PoolStats};
pub use correlation::{CorrelationId, RequestContext};
pub use disaggregated::{
    BackpressureAction, CacheTransferProfile, ChunkedPrefillCoordinator, CompletionEvent,
    CompletionNotifier, CompletionReason, DIBenchmarkConfig, DIBenchmarkResult,
    DICrossoverAnalysis, DICrossoverEntry, DecodeRequest, DecodeScheduler, DecodeSchedulerConfig,
    DecodeSequence, DisaggRoutingStrategy, DisaggregatedMetrics, DisaggregatedMetricsSnapshot,
    DisaggregatedServer, DisaggregatedServingConfig, HandoffProtocol, HandoffStatus,
    HybridModeGuard, IngestionStats, NodeLoadInfo, PrefillHandoff, PrefillRequest, PrefillResult,
    PrefillScheduler, PrefillSchedulerConfig, PromptLengthAnalysis, RequestPhase, RequestRouter,
    RouterConfig, RouterMetrics, SequenceStatus, ServingMode, StreamBridge, StreamBridgeError,
    StreamPhase, TokenEvent, TokenSource, TrackedRequest, format_di_report, run_di_benchmark,
    run_di_crossover_analysis, run_prompt_length_analysis,
};
pub use discovery::{initialize_distributed, log_cluster_topology, probe_peers};
pub use failure_detector::{FailureDetector, FailureDetectorConfig, FailureEvent};
pub use handoff_queue::{
    HandoffItem, HandoffQueue, HandoffQueueConfig, HandoffQueueManager, OverflowPolicy, QueueStats,
};
pub use heartbeat::{HEARTBEAT_OPERATION, HeartbeatConfig, HeartbeatPayload, HeartbeatService};
pub use kv_cache_serde::{
    CACHE_FORMAT_VERSION, CACHE_FORMAT_VERSION_V1, CACHE_FORMAT_VERSION_V2, CacheMetadata,
    CacheType, RawTensorData, SerializableCacheEntry, SerializableCacheState,
    SerializablePagedBlock, SerializablePagedLayerState, SerializablePagedSequenceState,
    SerializableSamplingState, SerializableSequenceBackend, deserialize_cache_state,
    extract_chunked_cache_entry, extract_kv_cache_entry, extract_rotating_cache_entry,
    reconstruct_mlx_array, restore_into_cache_pool_sequence, restore_into_kv_caches,
    restore_into_sequence_cache_set, serialize_cache_pool_sequence, serialize_cache_state,
    serialize_sequence_cache_set, validate_raw_tensor,
};
pub use kv_cache_transfer::{
    AdaptiveSelector, BandwidthEstimator, BandwidthSample, CacheQuantizationConfig,
    CacheQuantizationLevel, LayerTransferHeader, LayerTransferResult, ParallelLayerTransfer,
    QuantizedCacheTransfer, StreamedCacheTransfer, TransferBenchConfig, TransferBenchResult,
    TransferBenchmark, TransferConfig, TransferResult, TransferStrategy,
};
pub use metrics::{
    ClusterMetrics, LatencyPercentiles, MetricsCollector, MetricsConfig, NodeMetrics,
    PagedKvMetrics,
};
pub use pipeline::{
    ActivationMessage, ActivationReceiver, ActivationSender, ChannelConfig, ChunkedPrefillPipeline,
    DeviceSpec, FailedRequest, GPipeSchedule, InProcessStageWorkerLoop, LayerAdjacencyGroup,
    LayerFilter, LoadedStageExecutor, MetricsSummary as PipelineMetricsSummary, MicroBatch,
    MicroBatchSpec, ModelProfile, PartitionConfig, PartitionQualityReport, PipelineBenchmarkConfig,
    PipelineBenchmarkResult, PipelineChannel, PipelineConfig, PipelineCoordinator, PipelineMetrics,
    PipelineMetricsCollector, PipelineRequest, PipelineResponse, PipelineSchedule,
    PipelineServingConfig, PipelineWorkerInput, PipelineWorkerOutput, SafeTensorsIndex,
    ScalingResult, ScheduleAction, StageAssignment, StageEndpoint, StageExecutionInput,
    StageExecutionOutput, StageExecutor, StageHealth, StageLifecycleRequest,
    StageLifecycleResponse, StageLifecycleSnapshot, StageLifecycleState, StageLink, StageMetrics,
    StageQuality, StageRole, TransportStageEndpoint, TransportStageLink, WeightClass,
    activation_channel, activation_latency, auto_partition, auto_partition_with_report,
    build_manual_assignments, build_model_profile, build_pipeline_links, build_profile_from_json,
    build_quality_report, build_quality_report_with_actuals, classify_weight_key,
    create_gpipe_schedule, detect_pipeline_config, estimate_partial_memory, filter_weight_keys,
    filter_weight_map, format_benchmark_report, format_quality_report, identify_required_shards,
    install_stage_control_service, load_in_process_stage_worker, log_partition_quality,
    parse_manual_partition, populate_actual_memory, resolve_in_process_pipeline_num_layers,
    resolve_in_process_stage_assignments, resolve_in_process_stage_assignments_for_model,
    run_pipeline_benchmark, run_scaling_benchmark, should_load_key, should_use_pipeline,
    split_into_micro_batches, suggested_micro_batch_size, summarize_quality_warnings,
    to_pipeline_schedule_config, validate_activation, validate_adjacency, validate_memory_fit,
    validate_partial_memory, validate_partition,
};
pub use rdma_capabilities::{
    RDMA_PROTOCOL_VERSION, RdmaAcceleration, RdmaCapabilities, negotiate_protocol_version,
    probe_capabilities,
};
pub use rdma_transport::{RdmaTransport, RdmaTransportConfig};
pub use registry::{NodeRegistry, NodeStatus, RegisteredNode};
pub use request_tracker::{
    RequestId, RequestLifecycle, RequestState, RequestTracker, RequestTrackerConfig,
};
pub use routing::{
    LoadBalancedRouter, NodeCandidate, PipelineStageRouter, PipelineTensorParallelRouter,
    RoleBasedRouter, RoundRobinRouter, RoutingDecision, RoutingRequest, RoutingStrategy,
    TrafficClass,
};
pub use scheduler::{CoordinationMode, Scheduler, SchedulerConfig};
pub use tcp_transport::{TcpTransport, TcpTransportConfig};
pub use tensor_chunked::{ChunkAssembler, ChunkedTensor, ChunkedTransferConfig};
pub use tensor_parallel::{
    AllReduceProfile, BenchmarkResult, ByteRangeSpec, CollectiveConfig, CollectiveGroup,
    CommPattern, CrossoverAnalysis, CrossoverEntry, EmbeddingMode, LayerShardPlan,
    LockstepBenchmarkResult, ModelShardPlan, MoeShardMode, RingTopology, ScalingAnalysis,
    ShardConfig, ShardSpec, ShardStrategy, ShardedMemoryReport, TPBenchmarkConfig,
    TPBenchmarkResult, TensorParallelErnie45Model, TensorParallelGemma3Model,
    TensorParallelGemma4Model, TensorParallelHunyuanV1DenseModel, TensorParallelLlamaModel,
    TensorParallelPlanSummary, TensorParallelQwen3Model, TensorParallelQwen35Model,
    TensorParallelRuntimeKind, TensorParallelRuntimeSupport, all_gather, all_reduce_sum,
    compute_byte_ranges, compute_shard_spec, compute_sharded_shape, dtype_byte_size,
    ensure_single_rank_runtime, format_tp_benchmark_report, generate_shard_plan, reduce_scatter,
    resolve_model_shard_plan, ring_allreduce_data_volume, run_crossover_analysis,
    run_lockstep_benchmark, run_scaling_analysis, run_tp_benchmark, shard_config_from_cli,
    shard_tensor_data, validate_sharded_memory, validate_supported_runtime,
};
pub use tensor_protocol::{
    PROTOCOL_VERSION, QuantizationMode, TensorDtype, TensorFlags, TensorHeader, TensorKind,
};
pub use tensor_serialize::{
    DeserializedTensor, SerializeOptions, deserialize_tensor, serialize_tensor,
    serialize_tensor_to_bytes,
};
pub use thunderbolt_transport::{ThunderboltTransport, ThunderboltTransportConfig};
pub use transport::{Transport, TransportBackend, TransportMessage};
