use std::time::Duration;

use super::*;
use crate::distributed::kv_cache_serde::types::{
    CacheMetadata, CacheType, SerializableCacheState, SerializableSamplingState,
    SerializableSequenceBackend,
};
use crate::distributed::request_tracker::RequestId;

fn default_sampling() -> SerializableSamplingState {
    SerializableSamplingState {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        seed: None,
        repetition_penalty: 1.0,
        dry_multiplier: 0.0,
        dry_base: 0.0,
        dry_allowed_length: 0,
        dry_penalty_last_n: 0,
        dry_sequence_breakers: Vec::new(),
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        stop_token_ids: Vec::new(),
    }
}

fn make_request(prompt_len: usize) -> PrefillRequest {
    PrefillRequest::new(RequestId::new(), vec![0i32; prompt_len], default_sampling())
}

fn make_vlm_request(prompt_len: usize) -> PrefillRequest {
    PrefillRequest::new_vlm(
        RequestId::new(),
        vec![0i32; prompt_len],
        default_sampling(),
        vec![1u8; 1024], // dummy image data
    )
}

fn make_prefill_result(request_id: RequestId, prompt_len: usize) -> PrefillResult {
    PrefillResult {
        request_id,
        first_token: 42,
        cache_state: SerializableCacheState {
            cache_type: CacheType::Standard,
            entries: Vec::new(),
            metadata: CacheMetadata {
                prompt_len,
                current_offset: prompt_len as i32,
                num_layers: 32,
                layer_offsets: vec![prompt_len as i32; 32],
                max_size: None,
                layer_indices: None,
                chunk_size: None,
                start_positions: None,
            },
            sampling_state: Some(default_sampling()),
            token_history: Vec::new(),
            sequence_id: 1,
            sequence_backend: SerializableSequenceBackend::DenseKvCache,
            paged_state: None,
            paged_blocks: Vec::new(),
        },
        prefill_duration: Duration::from_millis(100),
        prompt_len,
        is_vlm: false,
    }
}

// ── PrefillSchedulerConfig ───────────────────────────────────────────

#[test]
fn default_config_has_sane_defaults() {
    let config = PrefillSchedulerConfig::default();
    assert_eq!(config.max_concurrent_prefills, 4);
    assert_eq!(config.transfer_timeout, Duration::from_secs(30));
    assert!((config.memory_threshold - 0.85).abs() < f64::EPSILON);
    assert!(config.chunked_prefill_enabled);
    assert_eq!(config.chunk_size_tokens, 2048);
    assert_eq!(config.max_handoff_retries, 2);
}

// ── PrefillRequest ───────────────────────────────────────────────────

#[test]
fn text_request_is_not_vlm() {
    let req = make_request(100);
    assert!(!req.is_vlm);
    assert!(req.image_data.is_empty());
    assert_eq!(req.prompt_len(), 100);
    assert_eq!(req.priority, 100);
}

#[test]
fn vlm_request_has_image_data() {
    let req = make_vlm_request(200);
    assert!(req.is_vlm);
    assert!(!req.image_data.is_empty());
    assert_eq!(req.prompt_len(), 200);
}

// ── PrefillScheduler: Enqueue / Dequeue ──────────────────────────────

#[test]
fn enqueue_and_dequeue_basic() {
    let scheduler = PrefillScheduler::new(PrefillSchedulerConfig::default());

    scheduler.enqueue(make_request(100)).unwrap();
    scheduler.enqueue(make_request(50)).unwrap();
    scheduler.enqueue(make_request(200)).unwrap();

    assert_eq!(scheduler.queue_len(), 3);
    assert_eq!(scheduler.total_enqueued(), 3);

    // Shortest prompt should come out first (priority = prompt_len).
    let first = scheduler.try_dequeue().unwrap();
    assert_eq!(first.prompt_len(), 50);

    let second = scheduler.try_dequeue().unwrap();
    assert_eq!(second.prompt_len(), 100);

    let third = scheduler.try_dequeue().unwrap();
    assert_eq!(third.prompt_len(), 200);

    assert!(scheduler.try_dequeue().is_none());
}

#[test]
fn dequeue_respects_concurrency_limit() {
    let config = PrefillSchedulerConfig {
        max_concurrent_prefills: 2,
        ..Default::default()
    };
    let scheduler = PrefillScheduler::new(config);

    scheduler.enqueue(make_request(10)).unwrap();
    scheduler.enqueue(make_request(20)).unwrap();
    scheduler.enqueue(make_request(30)).unwrap();

    // Dequeue two (hits concurrency limit).
    let _r1 = scheduler.try_dequeue().unwrap();
    let _r2 = scheduler.try_dequeue().unwrap();
    assert_eq!(scheduler.active_count(), 2);

    // Third dequeue should be blocked by concurrency.
    assert!(scheduler.try_dequeue().is_none());

    // Complete one prefill -> should allow another dequeue.
    scheduler.mark_prefill_completed();
    assert_eq!(scheduler.active_count(), 1);

    let _r3 = scheduler.try_dequeue().unwrap();
    assert_eq!(scheduler.active_count(), 2);
}

#[test]
fn enqueue_rejected_under_memory_pressure() {
    let config = PrefillSchedulerConfig {
        memory_threshold: 0.80,
        ..Default::default()
    };
    let scheduler = PrefillScheduler::new(config);

    // Set memory above threshold.
    scheduler.update_memory_utilization(0.90);
    assert!(scheduler.is_memory_pressure_high());

    let result = scheduler.enqueue(make_request(100));
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("memory pressure"));
}

#[test]
fn dequeue_blocked_under_memory_pressure() {
    let scheduler = PrefillScheduler::new(PrefillSchedulerConfig::default());

    scheduler.enqueue(make_request(100)).unwrap();

    // Raise memory pressure above threshold.
    scheduler.update_memory_utilization(0.95);

    // Dequeue should be blocked.
    assert!(scheduler.try_dequeue().is_none());
    assert_eq!(scheduler.queue_len(), 1);

    // Lower memory pressure.
    scheduler.update_memory_utilization(0.50);
    assert!(scheduler.try_dequeue().is_some());
}

// ── should_skip_decode ───────────────────────────────────────────────

#[test]
fn prefill_scheduler_always_skips_decode() {
    let scheduler = PrefillScheduler::new(PrefillSchedulerConfig::default());
    assert!(scheduler.should_skip_decode());
}

// ── can_accept ───────────────────────────────────────────────────────

#[test]
fn can_accept_checks_both_limits() {
    let config = PrefillSchedulerConfig {
        max_concurrent_prefills: 1,
        memory_threshold: 0.80,
        ..Default::default()
    };
    let scheduler = PrefillScheduler::new(config);

    assert!(scheduler.can_accept());

    // Hit concurrency limit.
    scheduler.enqueue(make_request(10)).unwrap();
    let _ = scheduler.try_dequeue();
    assert!(!scheduler.can_accept());

    // Release and hit memory limit instead.
    scheduler.mark_prefill_completed();
    assert!(scheduler.can_accept());

    scheduler.update_memory_utilization(0.90);
    assert!(!scheduler.can_accept());
}

// ── Handoff Management ───────────────────────────────────────────────

#[test]
fn initiate_and_acknowledge_handoff() {
    let scheduler = PrefillScheduler::new(PrefillSchedulerConfig::default());
    let request_id = RequestId::new();
    let result = make_prefill_result(request_id.clone(), 100);

    let handoff = scheduler
        .initiate_handoff(result, "decode-0".to_string())
        .unwrap();

    assert_eq!(handoff.decode_node_id, "decode-0");
    assert_eq!(handoff.status, HandoffStatus::Pending);
    assert_eq!(scheduler.active_handoff_count(), 1);
    assert_eq!(scheduler.total_handoffs(), 1);

    scheduler.acknowledge_handoff(&request_id).unwrap();
    assert_eq!(scheduler.active_handoff_count(), 0);
}

#[test]
fn fail_handoff_allows_retry() {
    let config = PrefillSchedulerConfig {
        max_handoff_retries: 2,
        ..Default::default()
    };
    let scheduler = PrefillScheduler::new(config);
    let request_id = RequestId::new();
    let result = make_prefill_result(request_id.clone(), 100);

    scheduler
        .initiate_handoff(result, "decode-0".to_string())
        .unwrap();

    // First failure: retry allowed.
    let can_retry = scheduler
        .fail_handoff(&request_id, "connection refused")
        .unwrap();
    assert!(can_retry);
    assert_eq!(scheduler.total_handoff_failures(), 1);

    // Second failure: retries exhausted, handoff removed.
    let can_retry = scheduler
        .fail_handoff(&request_id, "connection refused again")
        .unwrap();
    assert!(!can_retry);
    assert_eq!(scheduler.active_handoff_count(), 0);
    assert_eq!(scheduler.total_handoff_failures(), 2);
}

#[test]
fn acknowledge_nonexistent_handoff_fails() {
    let scheduler = PrefillScheduler::new(PrefillSchedulerConfig::default());
    let request_id = RequestId::new();

    let result = scheduler.acknowledge_handoff(&request_id);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not found"));
}

// ── PrefillResult ────────────────────────────────────────────────────

#[test]
fn prefill_result_throughput() {
    let result = PrefillResult {
        request_id: RequestId::new(),
        first_token: 1,
        cache_state: SerializableCacheState {
            cache_type: CacheType::Standard,
            entries: Vec::new(),
            metadata: CacheMetadata {
                prompt_len: 1000,
                current_offset: 1000,
                num_layers: 32,
                layer_offsets: Vec::new(),
                max_size: None,
                layer_indices: None,
                chunk_size: None,
                start_positions: None,
            },
            sampling_state: None,
            token_history: Vec::new(),
            sequence_id: 0,
            sequence_backend: SerializableSequenceBackend::DenseKvCache,
            paged_state: None,
            paged_blocks: Vec::new(),
        },
        prefill_duration: Duration::from_secs(1),
        prompt_len: 1000,
        is_vlm: false,
    };

    let tps = result.tokens_per_second();
    assert!((tps - 1000.0).abs() < f64::EPSILON);
}

// ── HandoffStatus Display ────────────────────────────────────────────

#[test]
fn handoff_status_display() {
    assert_eq!(HandoffStatus::Pending.to_string(), "pending");
    assert_eq!(HandoffStatus::Serializing.to_string(), "serializing");
    assert_eq!(HandoffStatus::Transferring.to_string(), "transferring");
    assert_eq!(HandoffStatus::Acknowledged.to_string(), "acknowledged");
    assert_eq!(
        HandoffStatus::Failed {
            reason: "timeout".into()
        }
        .to_string(),
        "failed: timeout"
    );
}

// ── PrefillHandoff ───────────────────────────────────────────────────

#[test]
fn handoff_timeout_check() {
    let result = make_prefill_result(RequestId::new(), 50);
    let handoff = PrefillHandoff::new(result, "decode-1".to_string());

    // Just created, should not be timed out.
    assert!(!handoff.is_timed_out(Duration::from_secs(30)));

    // Zero timeout should be timed out immediately (or nearly so).
    // Note: there may be a tiny race here, but Duration::ZERO should work.
    assert!(handoff.is_timed_out(Duration::ZERO));
}

#[test]
fn handoff_retry_limit() {
    let result = make_prefill_result(RequestId::new(), 50);
    let mut handoff = PrefillHandoff::new(result, "decode-1".to_string());

    assert!(!handoff.exceeded_retries(3));
    handoff.retry_count = 3;
    assert!(handoff.exceeded_retries(3));
}

// ── ChunkedPrefillCoordinator ────────────────────────────────────────

#[test]
fn chunked_coordinator_basic() {
    let coord = ChunkedPrefillCoordinator::new(5000, 2048, 32);

    assert_eq!(coord.total_chunks(), 3); // ceil(5000/2048) = 3
    assert_eq!(coord.chunk_range(0), Some((0, 2048)));
    assert_eq!(coord.chunk_range(1), Some((2048, 4096)));
    assert_eq!(coord.chunk_range(2), Some((4096, 5000)));
    assert_eq!(coord.chunk_range(3), None); // out of range

    assert!(!coord.is_final_chunk(0));
    assert!(!coord.is_final_chunk(1));
    assert!(coord.is_final_chunk(2));
}

#[test]
fn chunked_coordinator_single_chunk() {
    let coord = ChunkedPrefillCoordinator::new(1000, 2048, 32);

    assert_eq!(coord.total_chunks(), 1);
    assert_eq!(coord.chunk_range(0), Some((0, 1000)));
    assert!(coord.is_final_chunk(0));
}

#[test]
fn chunked_coordinator_exact_multiple() {
    let coord = ChunkedPrefillCoordinator::new(4096, 2048, 32);

    assert_eq!(coord.total_chunks(), 2);
    assert_eq!(coord.chunk_range(0), Some((0, 2048)));
    assert_eq!(coord.chunk_range(1), Some((2048, 4096)));
    assert!(coord.is_final_chunk(1));
}

#[test]
fn chunked_coordinator_progress() {
    let coord = ChunkedPrefillCoordinator::new(6144, 2048, 32);
    assert_eq!(coord.total_chunks(), 3);

    assert!((coord.progress() - 0.0).abs() < f64::EPSILON);

    coord.mark_chunk_completed();
    assert!((coord.progress() - 1.0 / 3.0).abs() < 0.01);

    coord.mark_chunk_completed();
    assert!((coord.progress() - 2.0 / 3.0).abs() < 0.01);

    coord.mark_chunk_completed();
    assert!((coord.progress() - 1.0).abs() < f64::EPSILON);
}

#[test]
fn chunked_coordinator_finalization() {
    let coord = ChunkedPrefillCoordinator::new(4096, 2048, 32);

    assert!(!coord.is_complete());
    coord.finalize();
    assert!(coord.is_complete());
}

#[test]
fn chunked_coordinator_layer_tracking() {
    let coord = ChunkedPrefillCoordinator::new(4096, 2048, 32);

    assert_eq!(coord.transferred_layers(), 0);
    coord.mark_layers_transferred(16);
    assert_eq!(coord.transferred_layers(), 16);
    coord.mark_layers_transferred(16);
    assert_eq!(coord.transferred_layers(), 32);
}

#[test]
fn chunked_coordinator_bytes_tracking() {
    let coord = ChunkedPrefillCoordinator::new(4096, 2048, 32);

    assert_eq!(coord.total_bytes_transferred(), 0);
    coord.add_bytes_transferred(1024 * 1024);
    assert_eq!(coord.total_bytes_transferred(), 1024 * 1024);
}

#[test]
fn chunked_coordinator_zero_tokens() {
    let coord = ChunkedPrefillCoordinator::new(0, 2048, 32);
    assert_eq!(coord.total_chunks(), 0);
    assert!((coord.progress() - 1.0).abs() < f64::EPSILON);
    assert_eq!(coord.chunk_range(0), None);
}

// ── Scheduler: Chunked Prefill Integration ───────────────────────────

#[test]
fn scheduler_creates_chunked_coordinator_for_long_prompts() {
    let scheduler = PrefillScheduler::new(PrefillSchedulerConfig::default());

    // Short prompt: no chunked coordinator.
    assert!(scheduler.create_chunked_coordinator(100, 32).is_none());

    // Prompt exactly at chunk size: no chunked coordinator.
    assert!(scheduler.create_chunked_coordinator(2048, 32).is_none());

    // Long prompt: chunked coordinator created.
    let coord = scheduler.create_chunked_coordinator(5000, 32).unwrap();
    assert_eq!(coord.total_chunks(), 3);
}

#[test]
fn scheduler_no_chunked_when_disabled() {
    let config = PrefillSchedulerConfig {
        chunked_prefill_enabled: false,
        ..Default::default()
    };
    let scheduler = PrefillScheduler::new(config);

    assert!(scheduler.create_chunked_coordinator(10000, 32).is_none());
}

// ── Memory Utilization ───────────────────────────────────────────────

#[test]
fn memory_utilization_clamping() {
    let scheduler = PrefillScheduler::new(PrefillSchedulerConfig::default());

    scheduler.update_memory_utilization(-0.5);
    assert!((scheduler.current_memory_utilization() - 0.0).abs() < f64::EPSILON);

    scheduler.update_memory_utilization(1.5);
    assert!((scheduler.current_memory_utilization() - 1.0).abs() < f64::EPSILON);

    scheduler.update_memory_utilization(0.75);
    assert!((scheduler.current_memory_utilization() - 0.75).abs() < f64::EPSILON);
}

// ── Statistics ───────────────────────────────────────────────────────

#[test]
fn statistics_accumulate() {
    let scheduler = PrefillScheduler::new(PrefillSchedulerConfig::default());

    assert_eq!(scheduler.total_enqueued(), 0);
    assert_eq!(scheduler.total_completed(), 0);

    scheduler.enqueue(make_request(10)).unwrap();
    scheduler.enqueue(make_request(20)).unwrap();
    assert_eq!(scheduler.total_enqueued(), 2);

    let _ = scheduler.try_dequeue();
    scheduler.mark_prefill_completed();
    assert_eq!(scheduler.total_completed(), 1);
}

// ── Debug Display ────────────────────────────────────────────────────

#[test]
fn scheduler_debug_display() {
    let scheduler = PrefillScheduler::new(PrefillSchedulerConfig::default());
    let debug = format!("{scheduler:?}");
    assert!(debug.contains("PrefillScheduler"));
    assert!(debug.contains("max_concurrent"));
}
