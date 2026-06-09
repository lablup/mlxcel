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

//! Disaggregated inference: prefill/decode separation.
//!
//! This module provides the scheduling and handoff logic for disaggregated
//! serving, where prefill and decode phases run on separate nodes.
//!
//! - [`prefill_scheduler`] — Prefill-only scheduler that processes prompts
//!   and hands off KV cache + first token to decode nodes.
//! - [`decode_scheduler`] — Decode-only scheduler that receives KV caches
//!   from prefill nodes and manages batched token generation.
//! - [`request_router`] — Request router and load balancer that orchestrates
//!   the full disaggregated pipeline with configurable routing strategies.
//! - [`serving`] — API server integration: `DisaggregatedServer`,
//!   `ServingMode`, `HybridModeGuard`, configuration, and metrics.
//! - [`stream_bridge`] — Seamless SSE stream bridging across the
//!   prefill→decode boundary.

pub mod benchmark;
pub mod coordinator;
pub mod decode_scheduler;
pub mod handoff_impl;
pub mod prefill_scheduler;
pub mod request_router;
pub mod serving;
pub mod stream_bridge;

pub use benchmark::{
    CacheTransferProfile, DIBenchmarkConfig, DIBenchmarkResult, DICrossoverAnalysis,
    DICrossoverEntry, PromptLengthAnalysis, format_di_report, run_di_benchmark,
    run_di_crossover_analysis, run_prompt_length_analysis,
};
pub use coordinator::{DecodeRoleHandoff, PrefillRoleRequest, ServingCoordinator};
pub use decode_scheduler::{
    CompletionEvent, CompletionNotifier, CompletionReason, DecodeRequest, DecodeScheduler,
    DecodeSchedulerConfig, DecodeSequence, IngestionStats, SequenceStatus,
};
pub use handoff_impl::{
    HANDOFF_TENSOR_ID, extract_sequence_handoff, ingest_sequence_handoff,
    ingest_sequence_handoff_state, probe_block_geometry, recv_handoff_payload,
    send_handoff_payload,
};
pub use prefill_scheduler::{
    ChunkedPrefillCoordinator, HandoffProtocol, HandoffStatus, PrefillHandoff, PrefillRequest,
    PrefillResult, PrefillScheduler, PrefillSchedulerConfig,
};
pub use request_router::{
    BackpressureAction, DisaggRoutingStrategy, NodeLoadInfo, RequestPhase, RequestRouter,
    RouterConfig, RouterMetrics, TrackedRequest,
};
pub use serving::{
    DisaggregatedMetrics, DisaggregatedMetricsSnapshot, DisaggregatedServer,
    DisaggregatedServingConfig, HybridModeGuard, ServingMode,
};
pub use stream_bridge::{StreamBridge, StreamBridgeError, StreamPhase, TokenEvent, TokenSource};
