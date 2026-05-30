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

//! Pipeline parallelism layer partitioning, configuration, activation transfer,
//! micro-batching, scheduling, and metrics.
//!
//! Provides the algorithm and types for distributing model layers across
//! multiple devices in a pipeline-parallel topology:
//!
//! - [`ModelProfile`] — describes a model's layer count, parameter sizes,
//!   and embedding/lm_head costs
//! - [`DeviceSpec`] — per-device available memory and compute capability
//! - [`StageAssignment`] — the output of partitioning: which layers go where
//! - [`PartitionConfig`] — auto vs. manual partition specification
//! - [`auto_partition`] — memory-proportional layer assignment algorithm
//! - [`parse_manual_partition`] — parse `--pp-layers 0-15,16-31` syntax
//! - [`validate_partition`] — reject gaps, overlaps, and memory violations
//!
//! Partial-loading support for pipeline stages:
//!
//! - [`LayerFilter`] — describes which model subset a stage needs
//! - [`classify_weight_key`] — categorise a weight key (layer/embedding/lm_head/norm/other)
//! - [`SafeTensorsIndex`] — parse `model.safetensors.index.json` to map keys → shard files
//! - [`filter_weight_map`] — drop unneeded tensors from an already-loaded weight map
//! - [`estimate_partial_memory`] / [`validate_partial_memory`] — memory budget helpers
//!
//! Activation transfer between pipeline stages:
//!
//! - [`ActivationMessage`] — structured payload with tensor, mask, position IDs
//! - [`ActivationSender`] / [`ActivationReceiver`] — async channels with back-pressure
//! - [`PipelineChannel`] — bidirectional channel between adjacent stages
//! - [`StageLink`] / [`build_pipeline_links`] — connect N stages into a pipeline
//!
//! Micro-batching and pipeline schedule:
//!
//! - [`MicroBatchSpec`] / [`MicroBatch`] — micro-batch splitting and tracking
//! - [`PipelineSchedule`] — trait for pipeline schedule implementations
//! - [`GPipeSchedule`] — GPipe-style forward-all-then-collect schedule
//! - [`PipelineConfig`] — schedule configuration (stages, micro-batch size)
//! - [`ScheduleAction`] — actions emitted by the schedule to drive execution
//!
//! Pipeline-aware KV cache management:
//!
//! - [`PipelineCacheConfig`] — per-stage cache configuration
//! - [`PipelineCacheManager`] — per-stage cache tracking and admission
//! - [`CacheAdmissionRequest`] / [`AdmissionDecision`] — coordinated admission
//! - [`EvictionEvent`] / [`PreemptionSignal`] — eviction and preemption
//! - [`CacheMetadataSync`] — cross-stage cache state consistency
//! - [`coordinated_admission`] / [`broadcast_eviction`] — multi-stage coordination
//!
//! Pipeline metrics:
//!
//! - [`StageMetrics`] — per-stage timing breakdown
//! - [`PipelineMetrics`] — bubble ratio, utilization, latency breakdown
//! - [`MetricsCollector`] — accumulates metrics across pipeline steps
//!
//! Elastic repartitioning:
//!
//! - [`ElasticPpConfig`] — runtime configuration for `--enable-elastic-pp`
//! - [`RepartitionCoordinator`] — drain → rebalance → resume state machine
//! - [`RepartitionTrigger`] / [`RepartitionState`] / [`RepartitionOutcome`]
//! - [`RepartitionEvent`] / [`RepartitionEventSink`] — observability hooks
//!   consumed by the metrics endpoint

pub mod activation_transfer;
pub mod benchmark;
pub mod cache_manager;
pub mod elastic;
pub mod local_runtime;
pub mod metrics;
pub mod micro_batch;
pub mod partial_loading;
pub mod partial_loading_adapter;
pub mod partition;
pub mod partition_assembly;
pub mod partition_balance;
pub mod partition_profile;
pub mod partition_profile_heuristics;
pub mod partition_quality;
pub mod remote_service;
pub mod runtime;
pub mod schedule;
pub mod server_runtime;
pub mod serving;
pub mod stage_executor;
pub mod stage_worker;
pub mod trace;
pub mod wire_tensor;

pub use activation_transfer::{
    ActivationMessage, ActivationReceiver, ActivationSender, ChannelConfig, PipelineChannel,
    StageEndpoint, StageLifecycleRequest, StageLifecycleResponse, StageLifecycleSnapshot,
    StageLifecycleState, StageLink, TransportStageEndpoint, TransportStageLink, activation_channel,
    activation_latency, build_pipeline_links, install_stage_control_service, validate_activation,
};
pub use benchmark::{
    PipelineBenchmarkConfig, PipelineBenchmarkResult, ScalingResult, format_benchmark_report,
    run_pipeline_benchmark, run_scaling_benchmark,
};
pub use cache_manager::{
    AdmissionDecision, AdmissionDiagnostic, CacheAdmissionRequest, CacheMetadataSync,
    EvictionEvent, EvictionReason, PipelineCacheConfig, PipelineCacheManager, PpTpAdmissionOutcome,
    PpTpCoord, PreemptionPolicy, PreemptionReason, PreemptionSignal, RejectionReason, SequenceId,
    StageCacheAllocation, broadcast_2d_eviction, broadcast_eviction, check_pipeline_pressure,
    coordinated_2d_admission, coordinated_admission, coordinated_admission_with_attribution,
    sync_metadata,
};
pub use elastic::{
    ElasticPpConfig, ElasticRuntimeDriver, NoopEventSink, RecordingEventSink,
    RepartitionCoordinator, RepartitionEvent, RepartitionEventSink, RepartitionOutcome,
    RepartitionState, RepartitionTrigger,
};
pub use local_runtime::{
    load_in_process_stage_worker, load_in_process_stage_worker_with_adapter, log_partition_quality,
    resolve_in_process_pipeline_num_layers, resolve_in_process_stage_assignments,
    resolve_in_process_stage_assignments_for_model,
};
pub use metrics::{
    ActivationLatencyHistogram, ActivationLatencyPair, AdmissionRejectionCounters,
    AdmissionRejectionEntry, MetricsCollector as PipelineMetricsCollector, MetricsSummary,
    PipelineMetrics, PipelineObservability, PipelineObservabilitySnapshot, RepartitionMetrics,
    RepartitionMetricsSnapshot, StageMetrics, StageUtilizationRegistry, StageUtilizationSnapshot,
};
pub use micro_batch::{
    MicroBatch, MicroBatchSpec, split_into_micro_batches, suggested_micro_batch_size,
};
pub use partial_loading::{
    LayerFilter, SafeTensorsIndex, WeightClass, classify_weight_key, estimate_partial_memory,
    filter_weight_keys, filter_weight_map, identify_required_shards, should_load_key,
    validate_partial_memory,
};
pub use partial_loading_adapter::{
    filter_adapter_weights, load_stage_adapter_weights, resolve_adapter_weights_path,
    should_load_adapter_key,
};
pub use partition::{
    DeviceSpec, LayerAdjacencyGroup, ModelProfile, PartitionConfig, StageAssignment,
    auto_partition, auto_partition_with_report, build_manual_assignments, parse_manual_partition,
    validate_adjacency, validate_memory_fit, validate_partition,
};
pub use partition_profile::{build_model_profile, build_profile_from_json};
pub use partition_quality::{
    PartitionQualityReport, StageQuality, build_quality_report, build_quality_report_with_actuals,
    format_quality_report, populate_actual_memory, summarize_quality_warnings,
};
pub use remote_service::{
    RemoteStageCommand, RemoteStageResponse, RemoteStageServiceConfig, RemoteStageServiceHandle,
};
pub use runtime::{
    InProcessPipelineRuntime, PipelineModelRuntime, RemotePipelineRuntime,
    RemotePipelineRuntimeConfig,
};
pub use schedule::{
    GPipeSchedule, PipelineConfig, PipelineSchedule, ScheduleAction, create_gpipe_schedule,
};
pub use server_runtime::PipelineServerModel;
pub use serving::{
    ChunkedPrefillPipeline, FailedRequest, PipelineCoordinator, PipelineRequest, PipelineResponse,
    PipelineServingConfig, StageHealth, StageRole, detect_pipeline_config, should_use_pipeline,
    to_pipeline_schedule_config,
};
pub use stage_executor::{
    LoadedStageExecutor, StageExecutionInput, StageExecutionOutput, StageExecutor, StageFamily,
    supported_families,
};
pub use stage_worker::{InProcessStageWorkerLoop, PipelineWorkerInput, PipelineWorkerOutput};
pub use trace::{PpTracer, StageSpan};
