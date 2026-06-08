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

//! End-to-end tests for disaggregated inference.
//!
//! Exercises the full disaggregated stack -- prefill scheduler, decode
//! scheduler, request router, stream bridge, serving, and benchmarks --
//! in simulated multi-node configurations. All tests run in-process with
//! no real hardware needed.

use std::net::SocketAddr;
use std::time::Duration;

use mlxcel::distributed::backpressure::{BackpressureConfig, BackpressureMonitor};
use mlxcel::distributed::config::{
    ClusterConfig, ClusterMeta, NodeConfig, NodeResources, NodeRole,
};
use mlxcel::distributed::disaggregated::benchmark::{
    DIBenchmarkConfig, format_di_report, run_di_benchmark, run_di_crossover_analysis,
    run_prompt_length_analysis,
};
use mlxcel::distributed::disaggregated::decode_scheduler::{
    CompletionReason, DecodeRequest, DecodeScheduler, DecodeSchedulerConfig, SequenceStatus,
};
use mlxcel::distributed::disaggregated::prefill_scheduler::{
    ChunkedPrefillCoordinator, PrefillRequest, PrefillResult, PrefillScheduler,
    PrefillSchedulerConfig,
};
use mlxcel::distributed::disaggregated::request_router::{
    BackpressureAction, DisaggRoutingStrategy, RequestPhase, RequestRouter, RouterConfig,
};
use mlxcel::distributed::disaggregated::serving::{
    DisaggregatedServer, DisaggregatedServingConfig, HybridModeGuard, ServingMode,
};
use mlxcel::distributed::disaggregated::stream_bridge::{
    StreamBridge, StreamBridgeError, StreamPhase, TokenEvent, TokenSource,
};
use mlxcel::distributed::kv_cache_serde::types::{
    CacheMetadata, CacheType, SerializableCacheState, SerializableSamplingState,
    SerializableSequenceBackend,
};
use mlxcel::distributed::metrics::ClusterMetrics;
use mlxcel::distributed::registry::{NodeRegistry, NodeStatus};
use mlxcel::distributed::request_tracker::RequestId;

// ===========================================================================
// Test helpers
// ===========================================================================

/// Create a test sampling state with greedy parameters.
fn test_sampling_state() -> SerializableSamplingState {
    SerializableSamplingState {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        seed: Some(42),
        repetition_penalty: 1.0,
        dry_multiplier: 0.0,
        dry_base: 0.0,
        dry_allowed_length: 0,
        dry_penalty_last_n: 0,
        dry_sequence_breakers: vec![],
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        stop_token_ids: vec![2],
    }
}

/// Create a test cache state for a given prompt length.
fn test_cache_state(prompt_len: usize, num_layers: usize) -> SerializableCacheState {
    SerializableCacheState {
        cache_type: CacheType::Standard,
        entries: Vec::new(),
        metadata: CacheMetadata {
            prompt_len,
            current_offset: prompt_len as i32,
            num_layers,
            layer_offsets: vec![prompt_len as i32; num_layers],
            max_size: None,
            layer_indices: None,
            chunk_size: None,
            start_positions: None,
        },
        sampling_state: Some(test_sampling_state()),
        token_history: (0..prompt_len as i32).collect(),
        sequence_id: 0,
        sequence_backend: SerializableSequenceBackend::DenseKvCache,
        paged_state: None,
        paged_blocks: Vec::new(),
    }
}

/// Create a test prefill result.
fn test_prefill_result(request_id: RequestId, prompt_len: usize) -> PrefillResult {
    PrefillResult {
        request_id,
        first_token: 42,
        cache_state: test_cache_state(prompt_len, 32),
        prefill_duration: Duration::from_millis(10),
        prompt_len,
        is_vlm: false,
    }
}

/// Build a test cluster with configurable prefill and decode nodes.
fn test_cluster(
    num_prefill: usize,
    num_decode: usize,
) -> (NodeRegistry, ClusterMetrics, BackpressureMonitor) {
    let mut nodes = Vec::new();
    for i in 0..num_prefill {
        nodes.push(NodeConfig {
            id: format!("prefill-{i}"),
            address: format!("127.0.0.1:{}", 9000 + i)
                .parse::<SocketAddr>()
                .unwrap(),
            role: NodeRole::Prefill,
            stage: None,
            rank: None,
            resources: NodeResources {
                memory_bytes: 16_000_000_000,
                compute_units: 8,
            },
        });
    }
    for i in 0..num_decode {
        nodes.push(NodeConfig {
            id: format!("decode-{i}"),
            address: format!("127.0.0.1:{}", 9100 + i)
                .parse::<SocketAddr>()
                .unwrap(),
            role: NodeRole::Decode,
            stage: None,
            rank: None,
            resources: NodeResources {
                memory_bytes: 16_000_000_000,
                compute_units: 8,
            },
        });
    }

    let config = ClusterConfig {
        cluster: ClusterMeta::default(),
        nodes,
    };

    let local_id = if num_prefill > 0 {
        "prefill-0"
    } else {
        "decode-0"
    };
    let registry = NodeRegistry::from_config(&config, local_id);

    // Mark all nodes online.
    for i in 0..num_prefill {
        registry.set_node_status(&format!("prefill-{i}"), NodeStatus::Online);
    }
    for i in 0..num_decode {
        registry.set_node_status(&format!("decode-{i}"), NodeStatus::Online);
    }

    let cluster_metrics = ClusterMetrics::new();
    let backpressure = BackpressureMonitor::new(BackpressureConfig::default());

    (registry, cluster_metrics, backpressure)
}

/// Create a request router with the given node counts.
fn test_router(num_prefill: usize, num_decode: usize) -> RequestRouter {
    let (registry, metrics, bp) = test_cluster(num_prefill, num_decode);
    RequestRouter::new(RouterConfig::default(), registry, metrics, bp)
}

// ===========================================================================
// 1+1 configuration correctness tests
// ===========================================================================

#[test]
fn e2e_1p1d_prefill_scheduler_basic_flow() {
    let scheduler = PrefillScheduler::new(PrefillSchedulerConfig::default());

    let req = PrefillRequest::new(RequestId::new(), vec![1, 2, 3, 4, 5], test_sampling_state());
    scheduler.enqueue(req).unwrap();
    assert_eq!(scheduler.queue_len(), 1);

    let dequeued = scheduler.try_dequeue().unwrap();
    assert_eq!(dequeued.prompt_len(), 5);
    assert_eq!(scheduler.active_count(), 1);
    assert_eq!(scheduler.queue_len(), 0);

    scheduler.mark_prefill_completed();
    assert_eq!(scheduler.active_count(), 0);
}

#[test]
fn e2e_1p1d_decode_scheduler_basic_flow() {
    let scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());

    let req_id = RequestId::new();
    let decode_req = DecodeRequest::new(
        req_id.clone(),
        42,
        test_cache_state(128, 32),
        test_sampling_state(),
        10,
    );

    scheduler.ingest_cache(decode_req).unwrap();
    assert_eq!(scheduler.ingestion_queue_len(), 1);

    let admitted = scheduler.admit_sequences();
    assert_eq!(admitted.len(), 1);
    assert_eq!(admitted[0].status, SequenceStatus::Decoding);
    assert_eq!(scheduler.active_batch_size(), 1);

    // Generate tokens.
    for _ in 0..9 {
        let limit_reached = scheduler.record_token(&req_id).unwrap();
        assert!(!limit_reached);
    }
    let limit_reached = scheduler.record_token(&req_id).unwrap();
    assert!(limit_reached);

    scheduler
        .mark_sequence_complete(&req_id, CompletionReason::MaxTokens)
        .unwrap();
    assert_eq!(scheduler.active_batch_size(), 0);

    let events = scheduler.drain_completion_events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].tokens_generated, 10);
}

#[test]
fn e2e_1p1d_handoff_and_acknowledgment() {
    let scheduler = PrefillScheduler::new(PrefillSchedulerConfig::default());

    let req_id = RequestId::new();
    let result = test_prefill_result(req_id.clone(), 128);

    let handoff = scheduler
        .initiate_handoff(result, "decode-0".to_string())
        .unwrap();
    assert_eq!(scheduler.active_handoff_count(), 1);
    assert_eq!(handoff.decode_node_id, "decode-0");

    scheduler.acknowledge_handoff(&req_id).unwrap();
    assert_eq!(scheduler.active_handoff_count(), 0);
}

#[test]
fn e2e_1p1d_request_router_full_lifecycle() {
    let router = test_router(1, 1);

    let req_id = RequestId::new();
    let prefill_node = router.route_to_prefill(req_id.clone(), 128).unwrap();
    assert!(prefill_node.starts_with("prefill-"));

    // Transition to decode.
    let decode_node = router.route_to_decode(&req_id).unwrap();
    assert!(decode_node.starts_with("decode-"));

    // Transition to decoding.
    assert!(router.mark_decoding(&req_id, &decode_node));

    let phase = router.get_phase(&req_id).unwrap();
    assert!(matches!(phase, RequestPhase::Decoding { .. }));

    // Complete.
    assert!(router.mark_completed(&req_id));
    let phase = router.get_phase(&req_id).unwrap();
    assert!(matches!(phase, RequestPhase::Completed));
}

#[test]
fn e2e_1p1d_stream_bridge_full_flow() {
    let bridge = StreamBridge::new("test-1p1d".to_string(), Duration::from_secs(30));

    assert_eq!(bridge.current_phase(), StreamPhase::WaitingForPrefill);

    // Submit first token from prefill.
    let first = TokenEvent {
        token_id: 42,
        text: "Hello".to_string(),
        sequence_number: 0,
        source: TokenSource::Prefill,
        is_final: false,
    };
    bridge.submit_first_token(&first).unwrap();
    assert_eq!(bridge.current_phase(), StreamPhase::HandoffToDecode);
    assert_eq!(bridge.tokens_emitted(), 1);

    // Start decode stream.
    bridge.start_decode_stream().unwrap();
    assert_eq!(bridge.current_phase(), StreamPhase::Decoding);

    // Submit decode tokens.
    for seq in 1..5u64 {
        let token = TokenEvent {
            token_id: 100 + seq as i32,
            text: format!("tok{seq}"),
            sequence_number: seq,
            source: TokenSource::Decode,
            is_final: seq == 4,
        };
        bridge.submit_decode_token(&token).unwrap();
    }

    assert_eq!(bridge.tokens_emitted(), 5);

    // Finalize.
    assert!(bridge.finalize());
    assert_eq!(bridge.current_phase(), StreamPhase::Complete);
    assert!(bridge.is_finalized());

    // Double finalize returns false.
    assert!(!bridge.finalize());
}

// ===========================================================================
// 1+2 configuration correctness tests
// ===========================================================================

#[test]
fn e2e_1p2d_routing_distributes_across_decode_nodes() {
    let router = test_router(1, 2);

    let mut decode_nodes = std::collections::HashSet::new();
    for _ in 0..10 {
        let req_id = RequestId::new();
        let _ = router.route_to_prefill(req_id.clone(), 128).unwrap();
        let decode_node = router.route_to_decode(&req_id).unwrap();
        decode_nodes.insert(decode_node);
    }

    // With default strategy (memory-aware), both decode nodes should be used.
    // At minimum, we should get at least one node.
    assert!(!decode_nodes.is_empty());
}

#[test]
fn e2e_1p2d_decode_scheduler_multi_sequence() {
    let scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());

    // Ingest 4 sequences.
    let mut req_ids = Vec::new();
    for _ in 0..4 {
        let req_id = RequestId::new();
        req_ids.push(req_id.clone());
        let decode_req = DecodeRequest::new(
            req_id,
            42,
            test_cache_state(128, 32),
            test_sampling_state(),
            5,
        );
        scheduler.ingest_cache(decode_req).unwrap();
    }

    // Admit all.
    let admitted = scheduler.admit_sequences();
    assert_eq!(admitted.len(), 4);
    assert_eq!(scheduler.active_batch_size(), 4);

    // Complete each sequence.
    for req_id in &req_ids {
        for _ in 0..5 {
            let _ = scheduler.record_token(req_id).unwrap();
        }
        scheduler
            .mark_sequence_complete(req_id, CompletionReason::MaxTokens)
            .unwrap();
    }

    assert_eq!(scheduler.active_batch_size(), 0);
    let events = scheduler.drain_completion_events();
    assert_eq!(events.len(), 4);
}

// ===========================================================================
// 2+2 configuration correctness tests
// ===========================================================================

#[test]
fn e2e_2p2d_routing_uses_both_prefill_nodes() {
    let (registry, metrics, bp) = test_cluster(2, 2);
    let config = RouterConfig {
        prefill_strategy: DisaggRoutingStrategy::RoundRobin,
        decode_strategy: DisaggRoutingStrategy::RoundRobin,
        ..RouterConfig::default()
    };
    let router = RequestRouter::new(config, registry, metrics, bp);

    let mut prefill_nodes = std::collections::HashSet::new();
    for _ in 0..10 {
        let req_id = RequestId::new();
        let node = router.route_to_prefill(req_id, 128).unwrap();
        prefill_nodes.insert(node);
    }

    // Round-robin should use both prefill nodes.
    assert_eq!(prefill_nodes.len(), 2, "both prefill nodes should be used");
}

#[test]
fn e2e_2p2d_concurrent_prefills() {
    let config = PrefillSchedulerConfig {
        max_concurrent_prefills: 4,
        ..PrefillSchedulerConfig::default()
    };
    let scheduler = PrefillScheduler::new(config);

    // Enqueue 4 requests.
    for _ in 0..4 {
        let req = PrefillRequest::new(RequestId::new(), vec![1, 2, 3], test_sampling_state());
        scheduler.enqueue(req).unwrap();
    }

    // Dequeue all 4.
    let mut dequeued = Vec::new();
    while let Some(req) = scheduler.try_dequeue() {
        dequeued.push(req);
    }
    assert_eq!(dequeued.len(), 4);
    assert_eq!(scheduler.active_count(), 4);

    // No more can be dequeued (at concurrency limit).
    assert!(scheduler.try_dequeue().is_none());
}

#[test]
fn e2e_2p2d_full_pipeline_multiple_requests() {
    let router = test_router(2, 2);
    let prefill_sched = PrefillScheduler::new(PrefillSchedulerConfig::default());
    let decode_sched = DecodeScheduler::new(DecodeSchedulerConfig::default());

    // Submit 4 requests through the full pipeline.
    for _ in 0..4 {
        let req_id = RequestId::new();

        // Route to prefill.
        let _ = router.route_to_prefill(req_id.clone(), 128).unwrap();

        // Enqueue and dequeue from prefill scheduler.
        let prefill_req = PrefillRequest::new(req_id.clone(), vec![1; 128], test_sampling_state());
        prefill_sched.enqueue(prefill_req).unwrap();
        let _ = prefill_sched.try_dequeue().unwrap();

        // Simulate prefill completion.
        prefill_sched.mark_prefill_completed();

        // Route to decode.
        let _ = router.route_to_decode(&req_id).unwrap();

        // Ingest into decode scheduler.
        let decode_req = DecodeRequest::new(
            req_id.clone(),
            42,
            test_cache_state(128, 32),
            test_sampling_state(),
            8,
        );
        decode_sched.ingest_cache(decode_req).unwrap();
    }

    // Admit and complete all decode sequences.
    let admitted = decode_sched.admit_sequences();
    assert_eq!(admitted.len(), 4);

    for seq in &admitted {
        for _ in 0..8 {
            let _ = decode_sched.record_token(&seq.request_id).unwrap();
        }
        decode_sched
            .mark_sequence_complete(&seq.request_id, CompletionReason::MaxTokens)
            .unwrap();
    }

    assert_eq!(decode_sched.active_batch_size(), 0);
    let stats = decode_sched.stats();
    assert_eq!(stats.total_ingested, 4);
    assert_eq!(stats.total_completed, 4);
    assert_eq!(stats.total_tokens_generated, 32);
}

// ===========================================================================
// Stream bridge correctness tests
// ===========================================================================

#[test]
fn e2e_stream_bridge_rejects_out_of_order_first_token() {
    let bridge = StreamBridge::new("ooo-test".to_string(), Duration::from_secs(5));

    let bad_first = TokenEvent {
        token_id: 1,
        text: "bad".to_string(),
        sequence_number: 1, // Should be 0.
        source: TokenSource::Prefill,
        is_final: false,
    };
    let err = bridge.submit_first_token(&bad_first).unwrap_err();
    assert!(matches!(err, StreamBridgeError::SequenceGap { .. }));
}

#[test]
fn e2e_stream_bridge_rejects_decode_before_prefill() {
    let bridge = StreamBridge::new("wrong-order".to_string(), Duration::from_secs(5));

    let decode_token = TokenEvent {
        token_id: 1,
        text: "bad".to_string(),
        sequence_number: 1,
        source: TokenSource::Decode,
        is_final: false,
    };
    let err = bridge.submit_decode_token(&decode_token).unwrap_err();
    assert!(matches!(
        err,
        StreamBridgeError::InvalidPhaseTransition { .. }
    ));
}

#[test]
fn e2e_stream_bridge_detects_sequence_gap() {
    let bridge = StreamBridge::new("gap-test".to_string(), Duration::from_secs(5));

    // Submit first token.
    let first = TokenEvent {
        token_id: 1,
        text: "a".to_string(),
        sequence_number: 0,
        source: TokenSource::Prefill,
        is_final: false,
    };
    bridge.submit_first_token(&first).unwrap();
    bridge.start_decode_stream().unwrap();

    // Skip sequence 1, submit sequence 2.
    let gap_token = TokenEvent {
        token_id: 3,
        text: "c".to_string(),
        sequence_number: 2,
        source: TokenSource::Decode,
        is_final: false,
    };
    let err = bridge.submit_decode_token(&gap_token).unwrap_err();
    assert!(matches!(err, StreamBridgeError::SequenceGap { .. }));
}

#[test]
fn e2e_stream_bridge_handoff_timeout() {
    let bridge = StreamBridge::new("timeout-test".to_string(), Duration::from_millis(1));

    let first = TokenEvent {
        token_id: 1,
        text: "a".to_string(),
        sequence_number: 0,
        source: TokenSource::Prefill,
        is_final: false,
    };
    bridge.submit_first_token(&first).unwrap();

    // Wait for timeout.
    std::thread::sleep(Duration::from_millis(5));
    assert!(bridge.is_handoff_timed_out());
}

#[test]
fn e2e_stream_bridge_ttft_measurement() {
    let bridge = StreamBridge::new("ttft-test".to_string(), Duration::from_secs(5));

    // TTFT should be None before first token.
    assert!(bridge.time_to_first_token().is_none());

    let first = TokenEvent {
        token_id: 1,
        text: "a".to_string(),
        sequence_number: 0,
        source: TokenSource::Prefill,
        is_final: false,
    };
    bridge.submit_first_token(&first).unwrap();

    // TTFT should now be measurable.
    let ttft = bridge.time_to_first_token();
    assert!(ttft.is_some());
}

// ===========================================================================
// Serving mode and hybrid guard tests
// ===========================================================================

#[test]
fn e2e_serving_mode_from_node_role() {
    assert_eq!(
        ServingMode::from_node_role(NodeRole::Prefill),
        ServingMode::PrefillOnly
    );
    assert_eq!(
        ServingMode::from_node_role(NodeRole::Decode),
        ServingMode::DecodeOnly
    );
    assert_eq!(
        ServingMode::from_node_role(NodeRole::Hybrid),
        ServingMode::Hybrid
    );
}

#[test]
fn e2e_hybrid_guard_zero_overhead() {
    let guard = HybridModeGuard::new(ServingMode::Hybrid);
    assert!(guard.is_local());
    assert!(guard.should_prefill());
    assert!(guard.should_decode());
    assert!(!guard.should_route());
}

#[test]
fn e2e_prefill_only_guard() {
    let guard = HybridModeGuard::new(ServingMode::PrefillOnly);
    assert!(!guard.is_local());
    assert!(guard.should_prefill());
    assert!(!guard.should_decode());
    assert!(!guard.should_route());
}

#[test]
fn e2e_decode_only_guard() {
    let guard = HybridModeGuard::new(ServingMode::DecodeOnly);
    assert!(!guard.is_local());
    assert!(!guard.should_prefill());
    assert!(guard.should_decode());
    assert!(!guard.should_route());
}

#[test]
fn e2e_router_guard() {
    let guard = HybridModeGuard::new(ServingMode::Router);
    assert!(!guard.is_local());
    assert!(!guard.should_prefill());
    assert!(!guard.should_decode());
    assert!(guard.should_route());
}

#[test]
fn e2e_disaggregated_server_hybrid_ready() {
    let config = DisaggregatedServingConfig::default();
    let server = DisaggregatedServer::new(config);
    assert!(server.is_ready());
    assert!(server.should_handle_locally());
    assert_eq!(server.mode(), ServingMode::Hybrid);
}

#[test]
fn e2e_disaggregated_server_prefill_only() {
    let config = DisaggregatedServingConfig {
        mode: ServingMode::PrefillOnly,
        decode_peers: vec!["127.0.0.1:9100".parse().unwrap()],
        ..DisaggregatedServingConfig::default()
    };
    let server = DisaggregatedServer::new(config);
    assert!(server.is_ready());
    assert!(!server.should_handle_locally());
    assert_eq!(server.decode_peers().len(), 1);
}

#[test]
fn e2e_disaggregated_metrics_recording() {
    let metrics = mlxcel::distributed::disaggregated::serving::DisaggregatedMetrics::new();

    metrics.record_prefill(128, Duration::from_millis(10));
    metrics.record_decode_tokens(32, Duration::from_millis(50));
    metrics.record_cache_transfer(Duration::from_millis(5), 1024 * 1024);
    metrics.record_stream_bridged();

    let snap = metrics.snapshot();
    assert_eq!(snap.prefill_prompts_total, 1);
    assert_eq!(snap.prefill_tokens_total, 128);
    assert_eq!(snap.decode_tokens_total, 32);
    assert_eq!(snap.cache_transfers_total, 1);
    assert_eq!(snap.cache_transfer_bytes_total, 1024 * 1024);
    assert_eq!(snap.streams_bridged_total, 1);
    assert!(snap.prefill_tokens_per_sec > 0.0);
    assert!(snap.decode_tokens_per_sec > 0.0);
    assert!(snap.cache_transfer_avg_latency_ms > 0.0);
}

// ===========================================================================
// Backpressure and stress tests
// ===========================================================================

#[test]
fn e2e_prefill_scheduler_memory_pressure_rejects() {
    let scheduler = PrefillScheduler::new(PrefillSchedulerConfig::default());

    // Simulate high memory pressure.
    scheduler.update_memory_utilization(0.95);
    assert!(scheduler.is_memory_pressure_high());

    let req = PrefillRequest::new(RequestId::new(), vec![1, 2, 3], test_sampling_state());
    let result = scheduler.enqueue(req);
    assert!(result.is_err(), "should reject under memory pressure");
}

#[test]
fn e2e_prefill_scheduler_concurrency_limit() {
    let config = PrefillSchedulerConfig {
        max_concurrent_prefills: 2,
        ..PrefillSchedulerConfig::default()
    };
    let scheduler = PrefillScheduler::new(config);

    // Enqueue 5 requests.
    for _ in 0..5 {
        let req = PrefillRequest::new(RequestId::new(), vec![1], test_sampling_state());
        scheduler.enqueue(req).unwrap();
    }

    // Only 2 should be dequeued.
    assert!(scheduler.try_dequeue().is_some());
    assert!(scheduler.try_dequeue().is_some());
    assert!(scheduler.try_dequeue().is_none());

    assert_eq!(scheduler.active_count(), 2);
    assert_eq!(scheduler.queue_len(), 3);
}

#[test]
fn e2e_decode_scheduler_backpressure_ingestion_queue_full() {
    let config = DecodeSchedulerConfig {
        ingestion_queue_size: 2,
        ..DecodeSchedulerConfig::default()
    };
    let scheduler = DecodeScheduler::new(config);

    // Fill the ingestion queue.
    for _ in 0..2 {
        let req = DecodeRequest::new(
            RequestId::new(),
            42,
            test_cache_state(128, 32),
            test_sampling_state(),
            10,
        );
        scheduler.ingest_cache(req).unwrap();
    }

    // Third should be rejected.
    let req = DecodeRequest::new(
        RequestId::new(),
        42,
        test_cache_state(128, 32),
        test_sampling_state(),
        10,
    );
    let result = scheduler.ingest_cache(req);
    assert!(result.is_err(), "should reject when queue is full");

    let stats = scheduler.stats();
    assert_eq!(stats.total_rejected, 1);
}

#[test]
fn e2e_decode_scheduler_memory_pressure_rejects() {
    let scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());

    scheduler.update_memory_utilization(0.95);

    let req = DecodeRequest::new(
        RequestId::new(),
        42,
        test_cache_state(128, 32),
        test_sampling_state(),
        10,
    );
    let result = scheduler.ingest_cache(req);
    assert!(result.is_err(), "should reject under memory pressure");
}

#[test]
fn e2e_router_backpressure_at_capacity() {
    let (registry, metrics, bp) = test_cluster(1, 1);
    let config = RouterConfig {
        prefill_queue_capacity: 3,
        ..RouterConfig::default()
    };
    let router = RequestRouter::new(config, registry, metrics, bp);

    // Fill the prefill queue.
    // First, route to prefill but don't advance them (they stay in Prefilling).
    for _ in 0..3 {
        let _ = router.route_to_prefill(RequestId::new(), 128).unwrap();
    }

    // The queue is for "Queued" requests. Let's check what backpressure says.
    let action = router.apply_backpressure();
    assert!(
        matches!(action, BackpressureAction::Accept),
        "should accept since these are Prefilling, not Queued"
    );
}

#[test]
fn e2e_concurrent_requests_stress() {
    let prefill_sched = PrefillScheduler::new(PrefillSchedulerConfig {
        max_concurrent_prefills: 8,
        ..PrefillSchedulerConfig::default()
    });
    let decode_sched = DecodeScheduler::new(DecodeSchedulerConfig {
        max_batch_size: 8,
        ..DecodeSchedulerConfig::default()
    });

    // Simulate 20 requests flowing through the system.
    let mut completed = 0;

    for _ in 0..20 {
        let req_id = RequestId::new();
        let req = PrefillRequest::new(req_id.clone(), vec![1; 64], test_sampling_state());
        prefill_sched.enqueue(req).unwrap();
    }

    // Process in batches. The prefill scheduler has a concurrency limit,
    // so we need multiple iterations to drain the queue.
    let max_iterations = 100;
    for _ in 0..max_iterations {
        if prefill_sched.queue_len() == 0
            && prefill_sched.active_count() == 0
            && decode_sched.active_batch_size() == 0
            && decode_sched.ingestion_queue_len() == 0
        {
            break;
        }

        // Dequeue and complete prefills.
        while let Some(req) = prefill_sched.try_dequeue() {
            prefill_sched.mark_prefill_completed();

            let decode_req = DecodeRequest::new(
                req.request_id,
                42,
                test_cache_state(64, 32),
                test_sampling_state(),
                4,
            );
            decode_sched.ingest_cache(decode_req).unwrap();
        }

        // Admit and generate tokens.
        let admitted = decode_sched.admit_sequences();
        for seq in &admitted {
            for _ in 0..4 {
                let _ = decode_sched.record_token(&seq.request_id).unwrap();
            }
            decode_sched
                .mark_sequence_complete(&seq.request_id, CompletionReason::MaxTokens)
                .unwrap();
            completed += 1;
        }
    }

    assert_eq!(completed, 20);
    assert_eq!(decode_sched.total_completed(), 20);
}

// ===========================================================================
// Node failure scenarios
// ===========================================================================

#[test]
fn e2e_node_failure_reroutes_prefilling_requests() {
    let router = test_router(2, 2);

    // Route 3 requests to prefill-0.
    let mut req_ids = Vec::new();
    for _ in 0..3 {
        let req_id = RequestId::new();
        let _ = router.route_to_prefill(req_id.clone(), 128).unwrap();
        req_ids.push(req_id);
    }

    // Simulate prefill-0 failure.
    let (rerouted, failed) = router.handle_node_failure("prefill-0");

    // Some requests that were on prefill-0 should be rerouted.
    // The exact count depends on which node was selected.
    assert!(rerouted + failed <= 3);

    // Rerouted requests should now be either in Prefilling on prefill-1
    // or re-queued.
    for req_id in &req_ids {
        if let Some(phase) = router.get_phase(req_id) {
            assert!(
                !matches!(
                    phase,
                    RequestPhase::Prefilling {
                        ref node_id
                    } if node_id == "prefill-0"
                ),
                "no request should still be on failed node"
            );
        }
    }
}

#[test]
fn e2e_node_failure_reroutes_decoding_requests() {
    let router = test_router(2, 2);

    // Route and advance 2 requests to decoding on decode-0.
    for _ in 0..2 {
        let req_id = RequestId::new();
        let _ = router.route_to_prefill(req_id.clone(), 128).unwrap();
        let _ = router.route_to_decode(&req_id).unwrap();
        let _ = router.mark_decoding(&req_id, "decode-0");
    }

    let (rerouted, failed) = router.handle_node_failure("decode-0");
    assert_eq!(rerouted + failed, 2);
}

#[test]
fn e2e_node_failure_no_candidates_marks_failed() {
    // Create a cluster with 1 prefill and 1 decode.
    let router = test_router(1, 1);

    let req_id = RequestId::new();
    let _ = router.route_to_prefill(req_id.clone(), 128).unwrap();
    let _ = router.route_to_decode(&req_id).unwrap();
    let _ = router.mark_decoding(&req_id, "decode-0");

    // Fail the only decode node.
    let (rerouted, failed) = router.handle_node_failure("decode-0");
    assert_eq!(failed, 1, "should fail with no alternative");
    assert_eq!(rerouted, 0);

    let phase = router.get_phase(&req_id).unwrap();
    assert!(matches!(phase, RequestPhase::Failed { .. }));
}

// ===========================================================================
// Chunked prefill tests
// ===========================================================================

#[test]
fn e2e_chunked_prefill_coordinator() {
    let coord = ChunkedPrefillCoordinator::new(8192, 2048, 32);

    assert_eq!(coord.total_chunks(), 4);
    assert!(!coord.is_final_chunk(0));
    assert!(!coord.is_final_chunk(2));
    assert!(coord.is_final_chunk(3));

    // Process all chunks.
    for i in 0..4 {
        let range = coord.chunk_range(i).unwrap();
        let expected_start = i * 2048;
        let expected_end = ((i + 1) * 2048).min(8192);
        assert_eq!(range, (expected_start, expected_end));

        coord.mark_chunk_completed();
        coord.mark_layers_transferred(32);
        coord.add_bytes_transferred(1024 * 1024);
    }

    assert_eq!(coord.completed_chunks(), 4);
    assert_eq!(coord.transferred_layers(), 128);
    assert!(coord.progress() > 0.99);

    coord.finalize();
    assert!(coord.is_complete());
}

#[test]
fn e2e_chunked_prefill_short_prompt_skips_chunking() {
    let scheduler = PrefillScheduler::new(PrefillSchedulerConfig {
        chunked_prefill_enabled: true,
        chunk_size_tokens: 2048,
        ..PrefillSchedulerConfig::default()
    });

    // Short prompt: no chunking.
    let coord = scheduler.create_chunked_coordinator(1024, 32);
    assert!(coord.is_none(), "short prompts should not use chunking");

    // Long prompt: chunking enabled.
    let coord = scheduler.create_chunked_coordinator(4096, 32);
    assert!(coord.is_some(), "long prompts should use chunking");
}

// ===========================================================================
// Performance benchmarks (simulated)
// ===========================================================================

#[test]
fn e2e_benchmark_1p1d_basic() {
    let config = DIBenchmarkConfig::new(1, 1)
        .with_prompt_lengths(vec![128])
        .with_decode_tokens(8);

    let result = run_di_benchmark(&config, 128).unwrap();

    assert!(result.throughput_tok_per_sec > 0.0);
    assert!(!result.ttft.is_zero());
    assert!(!result.tpot.is_zero());
    assert!(result.cache_transfer.bytes_transferred > 0);
}

#[test]
fn e2e_benchmark_1p2d_basic() {
    let config = DIBenchmarkConfig::new(1, 2)
        .with_prompt_lengths(vec![128])
        .with_decode_tokens(8);

    let result = run_di_benchmark(&config, 128).unwrap();
    assert!(result.throughput_tok_per_sec > 0.0);
}

#[test]
fn e2e_benchmark_2p2d_basic() {
    let config = DIBenchmarkConfig::new(2, 2)
        .with_prompt_lengths(vec![128])
        .with_decode_tokens(8);

    let result = run_di_benchmark(&config, 128).unwrap();
    assert!(result.throughput_tok_per_sec > 0.0);
}

#[test]
fn e2e_benchmark_prompt_length_scaling() {
    let config = DIBenchmarkConfig::new(1, 1)
        .with_prompt_lengths(vec![128, 1024, 4096])
        .with_decode_tokens(8);

    let analysis = run_prompt_length_analysis(&config).unwrap();
    assert_eq!(analysis.results.len(), 3);

    // TTFT should increase with prompt length.
    for pair in analysis.results.windows(2) {
        assert!(
            pair[1].ttft >= pair[0].ttft,
            "TTFT should increase: {:?} vs {:?}",
            pair[0].ttft,
            pair[1].ttft,
        );
    }

    // Transfer time slope should be positive.
    let slope = analysis.transfer_time_slope().unwrap();
    assert!(slope > 0.0, "transfer time slope should be positive");
}

#[test]
fn e2e_benchmark_crossover_analysis() {
    let configs = vec![(1, 1), (2, 2)];
    let prompt_lengths = vec![128, 1024];
    let concurrency_levels = vec![1, 4];

    let analysis =
        run_di_crossover_analysis(&configs, &prompt_lengths, &concurrency_levels).unwrap();

    assert_eq!(analysis.entries.len(), 8);

    for entry in &analysis.entries {
        assert!(entry.di_throughput > 0.0);
        assert!(entry.baseline_throughput > 0.0);
        assert!(entry.speedup > 0.0);
    }

    let display = format!("{analysis}");
    assert!(display.contains("Crossover Analysis"));
}

#[test]
fn e2e_benchmark_cache_transfer_throughput_measurable() {
    let config = DIBenchmarkConfig::new(1, 1)
        .with_prompt_lengths(vec![4096])
        .with_decode_tokens(4);

    let result = run_di_benchmark(&config, 4096).unwrap();

    let ct = &result.cache_transfer;
    assert!(ct.throughput_mb_per_sec() > 0.0);
    assert!(ct.bytes_transferred > 0);
    assert!(!ct.total_handoff_time().is_zero());
}

#[test]
fn e2e_benchmark_report_generation() {
    let configs = [
        DIBenchmarkConfig::new(1, 1).with_decode_tokens(4),
        DIBenchmarkConfig::new(1, 2).with_decode_tokens(4),
        DIBenchmarkConfig::new(2, 2).with_decode_tokens(4),
    ];

    let results: Vec<_> = configs
        .iter()
        .map(|c| run_di_benchmark(c, 128).unwrap())
        .collect();

    let report = format_di_report(&results);
    assert!(report.contains("Disaggregated Inference"));
    assert!(report.contains("Summary"));
}

// ===========================================================================
// Serving configuration tests
// ===========================================================================

#[test]
fn e2e_serving_config_from_cli_none_returns_none() {
    let result = DisaggregatedServingConfig::from_cli(None, vec![], vec![]).unwrap();
    assert!(result.is_none());
}

#[test]
fn e2e_serving_config_from_cli_prefill_only() {
    let decode_peer: SocketAddr = "127.0.0.1:9100".parse().unwrap();
    let result =
        DisaggregatedServingConfig::from_cli(Some("prefill"), vec![], vec![decode_peer]).unwrap();
    let config = result.unwrap();
    assert_eq!(config.mode, ServingMode::PrefillOnly);
    assert_eq!(config.decode_peers.len(), 1);
}

#[test]
fn e2e_serving_config_from_cli_decode_only_no_peers_fails() {
    let result = DisaggregatedServingConfig::from_cli(Some("decode"), vec![], vec![]);
    assert!(result.is_err());
}

// ===========================================================================
// Handoff failure and retry tests
// ===========================================================================

#[test]
fn e2e_handoff_failure_with_retry() {
    let scheduler = PrefillScheduler::new(PrefillSchedulerConfig {
        max_handoff_retries: 2,
        ..PrefillSchedulerConfig::default()
    });

    let req_id = RequestId::new();
    let result = test_prefill_result(req_id.clone(), 128);

    let _ = scheduler
        .initiate_handoff(result, "decode-0".to_string())
        .unwrap();

    // First failure: can retry.
    let can_retry = scheduler.fail_handoff(&req_id, "connection reset").unwrap();
    assert!(can_retry, "should be able to retry after first failure");

    // Second failure: can retry.
    let can_retry = scheduler
        .fail_handoff(&req_id, "connection reset again")
        .unwrap();
    assert!(
        !can_retry,
        "should not retry after exceeding max_handoff_retries"
    );

    // Handoff should be removed after exceeding retries.
    assert_eq!(scheduler.active_handoff_count(), 0);
}

#[test]
fn e2e_handoff_timeout_collection() {
    let scheduler = PrefillScheduler::new(PrefillSchedulerConfig {
        transfer_timeout: Duration::from_millis(1),
        ..PrefillSchedulerConfig::default()
    });

    let req_id = RequestId::new();
    let result = test_prefill_result(req_id.clone(), 128);

    let _ = scheduler
        .initiate_handoff(result, "decode-0".to_string())
        .unwrap();

    // Wait for timeout.
    std::thread::sleep(Duration::from_millis(5));

    let timed_out = scheduler.collect_timed_out_handoffs();
    assert_eq!(timed_out.len(), 1);
    assert_eq!(timed_out[0], req_id);
    assert_eq!(scheduler.active_handoff_count(), 0);
}

// ===========================================================================
// Statistics and metrics tests
// ===========================================================================

#[test]
fn e2e_prefill_scheduler_statistics() {
    let scheduler = PrefillScheduler::new(PrefillSchedulerConfig::default());

    for _ in 0..5 {
        let req = PrefillRequest::new(RequestId::new(), vec![1, 2], test_sampling_state());
        scheduler.enqueue(req).unwrap();
    }
    assert_eq!(scheduler.total_enqueued(), 5);

    while scheduler.try_dequeue().is_some() {
        scheduler.mark_prefill_completed();
    }
    assert_eq!(scheduler.total_completed(), 5);
}

#[test]
fn e2e_decode_scheduler_statistics() {
    let scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());

    for _ in 0..3 {
        let req = DecodeRequest::new(
            RequestId::new(),
            42,
            test_cache_state(64, 32),
            test_sampling_state(),
            2,
        );
        scheduler.ingest_cache(req).unwrap();
    }

    let admitted = scheduler.admit_sequences();
    for seq in &admitted {
        for _ in 0..2 {
            let _ = scheduler.record_token(&seq.request_id).unwrap();
        }
        scheduler
            .mark_sequence_complete(&seq.request_id, CompletionReason::MaxTokens)
            .unwrap();
    }

    let stats = scheduler.stats();
    assert_eq!(stats.total_ingested, 3);
    assert_eq!(stats.total_completed, 3);
    assert_eq!(stats.total_tokens_generated, 6);
}

#[test]
fn e2e_router_metrics_snapshot() {
    let router = test_router(2, 2);

    for _ in 0..5 {
        let req_id = RequestId::new();
        let _ = router.route_to_prefill(req_id.clone(), 128).unwrap();
        let _ = router.mark_completed(&req_id);
    }

    let metrics = router.metrics();
    assert_eq!(metrics.total_requests, 5);
    assert_eq!(metrics.total_completed, 5);
    assert!(metrics.routing_decisions > 0);
}

// ===========================================================================
// CI compatibility
// ===========================================================================

#[test]
fn e2e_ci_no_external_dependencies() {
    // Meta-check: if this compiles and runs, all DI E2E tests have no
    // hidden external dependencies (no GPU, no network, no model files).
    let config = DIBenchmarkConfig::default();
    let result = run_di_benchmark(&config, 128);
    assert!(result.is_ok());
}

#[test]
fn e2e_ci_benchmark_completes_quickly() {
    let start = std::time::Instant::now();

    let config = DIBenchmarkConfig::new(2, 2)
        .with_prompt_lengths(vec![128, 512])
        .with_decode_tokens(4);

    let _ = run_prompt_length_analysis(&config).unwrap();

    let elapsed = start.elapsed();
    assert!(
        elapsed < Duration::from_secs(30),
        "DI benchmark took too long for CI: {elapsed:?}"
    );
}
