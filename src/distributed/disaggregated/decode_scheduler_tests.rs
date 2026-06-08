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

fn make_cache_state(prompt_len: usize) -> SerializableCacheState {
    SerializableCacheState {
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
    }
}

fn make_decode_request(max_tokens: usize) -> DecodeRequest {
    DecodeRequest::new(
        RequestId::new(),
        42, // first token
        make_cache_state(100),
        default_sampling(),
        max_tokens,
    )
}

fn make_decode_request_with_id(request_id: RequestId, max_tokens: usize) -> DecodeRequest {
    DecodeRequest::new(
        request_id,
        42,
        make_cache_state(100),
        default_sampling(),
        max_tokens,
    )
}

// ── DecodeSchedulerConfig ────────────────────────────────────────────

#[test]
fn default_config_has_sane_defaults() {
    let config = DecodeSchedulerConfig::default();
    assert_eq!(config.max_batch_size, 32);
    assert_eq!(config.max_sequences, 128);
    assert!((config.memory_threshold - 0.85).abs() < f64::EPSILON);
    assert_eq!(config.ingestion_queue_size, 64);
}

// ── DecodeRequest ────────────────────────────────────────────────────

#[test]
fn decode_request_prompt_len() {
    let req = make_decode_request(50);
    assert_eq!(req.prompt_len(), 100);
    assert_eq!(req.first_token, 42);
    assert_eq!(req.max_tokens, 50);
}

// ── SequenceStatus Display ───────────────────────────────────────────

#[test]
fn sequence_status_display() {
    assert_eq!(SequenceStatus::Queued.to_string(), "queued");
    assert_eq!(SequenceStatus::Decoding.to_string(), "decoding");
    assert_eq!(SequenceStatus::Completed.to_string(), "completed");
    assert_eq!(
        SequenceStatus::Failed {
            reason: "oom".into()
        }
        .to_string(),
        "failed: oom"
    );
}

// ── CompletionReason Display ─────────────────────────────────────────

#[test]
fn completion_reason_display() {
    assert_eq!(CompletionReason::Eos.to_string(), "eos");
    assert_eq!(CompletionReason::MaxTokens.to_string(), "max_tokens");
    assert_eq!(
        CompletionReason::StopToken { token_id: 128001 }.to_string(),
        "stop_token(128001)"
    );
    assert_eq!(
        CompletionReason::Error {
            reason: "cuda error".into()
        }
        .to_string(),
        "error: cuda error"
    );
}

// ── Ingestion ────────────────────────────────────────────────────────

#[test]
fn ingest_and_admit_basic() {
    let scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());

    scheduler.ingest_cache(make_decode_request(50)).unwrap();
    scheduler.ingest_cache(make_decode_request(100)).unwrap();

    assert_eq!(scheduler.ingestion_queue_len(), 2);
    assert_eq!(scheduler.total_ingested(), 2);
    assert_eq!(scheduler.active_batch_size(), 0);

    let admitted = scheduler.admit_sequences();
    assert_eq!(admitted.len(), 2);
    assert_eq!(scheduler.ingestion_queue_len(), 0);
    assert_eq!(scheduler.active_batch_size(), 2);

    // Each admitted sequence should have a unique cache slot.
    assert_ne!(admitted[0].cache_slot_id, admitted[1].cache_slot_id);
    assert!(admitted[0].cache_slot_id.is_some());
    assert_eq!(admitted[0].status, SequenceStatus::Decoding);
}

#[test]
fn ingest_rejected_when_queue_full() {
    let config = DecodeSchedulerConfig {
        ingestion_queue_size: 2,
        ..Default::default()
    };
    let scheduler = DecodeScheduler::new(config);

    scheduler.ingest_cache(make_decode_request(50)).unwrap();
    scheduler.ingest_cache(make_decode_request(50)).unwrap();

    let result = scheduler.ingest_cache(make_decode_request(50));
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("queue full"));
    assert_eq!(scheduler.total_rejected(), 1);
}

#[test]
fn ingest_rejected_under_memory_pressure() {
    let config = DecodeSchedulerConfig {
        memory_threshold: 0.80,
        ..Default::default()
    };
    let scheduler = DecodeScheduler::new(config);

    scheduler.update_memory_utilization(0.90);

    let result = scheduler.ingest_cache(make_decode_request(50));
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("memory pressure"));
    assert_eq!(scheduler.total_rejected(), 1);
}

#[test]
fn ingest_rejected_when_max_sequences_reached() {
    let config = DecodeSchedulerConfig {
        max_sequences: 2,
        max_batch_size: 2,
        ..Default::default()
    };
    let scheduler = DecodeScheduler::new(config);

    scheduler.ingest_cache(make_decode_request(50)).unwrap();
    scheduler.ingest_cache(make_decode_request(50)).unwrap();

    let result = scheduler.ingest_cache(make_decode_request(50));
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("sequence limit"));
}

// ── Batch Admission ─────────────────────────────────────────────────

#[test]
fn admit_respects_batch_size_limit() {
    let config = DecodeSchedulerConfig {
        max_batch_size: 2,
        ..Default::default()
    };
    let scheduler = DecodeScheduler::new(config);

    // Enqueue 5 requests.
    for _ in 0..5 {
        scheduler.ingest_cache(make_decode_request(50)).unwrap();
    }

    // Only 2 should be admitted (max_batch_size = 2).
    let admitted = scheduler.admit_sequences();
    assert_eq!(admitted.len(), 2);
    assert_eq!(scheduler.active_batch_size(), 2);
    assert_eq!(scheduler.ingestion_queue_len(), 3);

    // Second call should admit 0 (batch is full).
    let admitted2 = scheduler.admit_sequences();
    assert_eq!(admitted2.len(), 0);
}

#[test]
fn admit_blocked_under_memory_pressure() {
    let scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());

    scheduler.ingest_cache(make_decode_request(50)).unwrap();

    // Raise memory above threshold.
    scheduler.update_memory_utilization(0.95);

    let admitted = scheduler.admit_sequences();
    assert!(admitted.is_empty());
    assert_eq!(scheduler.ingestion_queue_len(), 1);
}

// ── Sequence Completion ─────────────────────────────────────────────

#[test]
fn mark_complete_removes_from_active_batch() {
    let scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());
    let req_id = RequestId::new();

    scheduler
        .ingest_cache(make_decode_request_with_id(req_id.clone(), 50))
        .unwrap();
    scheduler.admit_sequences();
    assert_eq!(scheduler.active_batch_size(), 1);

    scheduler
        .mark_sequence_complete(&req_id, CompletionReason::Eos)
        .unwrap();
    assert_eq!(scheduler.active_batch_size(), 0);
    assert_eq!(scheduler.total_completed(), 1);
}

#[test]
fn mark_complete_emits_event() {
    let scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());
    let req_id = RequestId::new();

    scheduler
        .ingest_cache(make_decode_request_with_id(req_id.clone(), 50))
        .unwrap();
    scheduler.admit_sequences();

    scheduler
        .mark_sequence_complete(&req_id, CompletionReason::MaxTokens)
        .unwrap();

    let events = scheduler.drain_completion_events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].request_id, req_id);
    assert!(matches!(events[0].reason, CompletionReason::MaxTokens));
    assert!(events[0].freed_cache_slot.is_some());
}

#[test]
fn mark_complete_failure_increments_failed_count() {
    let scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());
    let req_id = RequestId::new();

    scheduler
        .ingest_cache(make_decode_request_with_id(req_id.clone(), 50))
        .unwrap();
    scheduler.admit_sequences();

    scheduler
        .mark_sequence_complete(
            &req_id,
            CompletionReason::Error {
                reason: "gpu error".into(),
            },
        )
        .unwrap();

    assert_eq!(scheduler.total_failed(), 1);
    assert_eq!(scheduler.total_completed(), 0);
}

#[test]
fn mark_complete_nonexistent_fails() {
    let scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());
    let req_id = RequestId::new();

    let result = scheduler.mark_sequence_complete(&req_id, CompletionReason::Eos);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not found"));
}

// ── Token Recording ─────────────────────────────────────────────────

#[test]
fn record_token_returns_true_at_limit() {
    let scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());
    let req_id = RequestId::new();

    scheduler
        .ingest_cache(make_decode_request_with_id(req_id.clone(), 3))
        .unwrap();
    scheduler.admit_sequences();

    assert!(!scheduler.record_token(&req_id).unwrap());
    assert!(!scheduler.record_token(&req_id).unwrap());
    assert!(scheduler.record_token(&req_id).unwrap()); // 3rd token = limit
}

#[test]
fn record_token_nonexistent_fails() {
    let scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());
    let req_id = RequestId::new();

    let result = scheduler.record_token(&req_id);
    assert!(result.is_err());
}

// ── should_skip_prefill ─────────────────────────────────────────────

#[test]
fn decode_scheduler_always_skips_prefill() {
    let scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());
    assert!(scheduler.should_skip_prefill());
}

// ── available_capacity ──────────────────────────────────────────────

#[test]
fn available_capacity_reflects_batch_state() {
    let config = DecodeSchedulerConfig {
        max_batch_size: 4,
        ..Default::default()
    };
    let scheduler = DecodeScheduler::new(config);

    assert_eq!(scheduler.available_capacity(), 4);

    scheduler.ingest_cache(make_decode_request(50)).unwrap();
    scheduler.ingest_cache(make_decode_request(50)).unwrap();
    scheduler.admit_sequences();

    assert_eq!(scheduler.available_capacity(), 2);
}

#[test]
fn available_capacity_zero_under_memory_pressure() {
    let scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());
    scheduler.update_memory_utilization(0.95);
    assert_eq!(scheduler.available_capacity(), 0);
}

// ── can_accept ──────────────────────────────────────────────────────

#[test]
fn can_accept_checks_queue_and_memory() {
    let config = DecodeSchedulerConfig {
        ingestion_queue_size: 2,
        max_sequences: 3,
        memory_threshold: 0.80,
        ..Default::default()
    };
    let scheduler = DecodeScheduler::new(config);

    assert!(scheduler.can_accept());

    // Fill the queue.
    scheduler.ingest_cache(make_decode_request(50)).unwrap();
    scheduler.ingest_cache(make_decode_request(50)).unwrap();
    assert!(!scheduler.can_accept()); // queue full

    // Admit to free queue space but hit sequence limit.
    scheduler.admit_sequences();
    scheduler.ingest_cache(make_decode_request(50)).unwrap();
    assert!(!scheduler.can_accept()); // 3 total = max_sequences
}

#[test]
fn can_accept_false_under_memory_pressure() {
    let scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());
    scheduler.update_memory_utilization(0.95);
    assert!(!scheduler.can_accept());
}

// ── Memory Utilization ──────────────────────────────────────────────

#[test]
fn memory_utilization_clamping() {
    let scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());

    scheduler.update_memory_utilization(-0.5);
    assert!((scheduler.current_memory_utilization() - 0.0).abs() < f64::EPSILON);

    scheduler.update_memory_utilization(1.5);
    assert!((scheduler.current_memory_utilization() - 1.0).abs() < f64::EPSILON);

    scheduler.update_memory_utilization(0.75);
    assert!((scheduler.current_memory_utilization() - 0.75).abs() < f64::EPSILON);
}

// ── Completion after slot freed allows re-admission ─────────────────

#[test]
fn completed_slot_frees_capacity() {
    let config = DecodeSchedulerConfig {
        max_batch_size: 1,
        ..Default::default()
    };
    let scheduler = DecodeScheduler::new(config);

    let req_id = RequestId::new();
    scheduler
        .ingest_cache(make_decode_request_with_id(req_id.clone(), 10))
        .unwrap();
    scheduler.admit_sequences();
    assert_eq!(scheduler.available_capacity(), 0);

    // Complete the sequence.
    scheduler
        .mark_sequence_complete(&req_id, CompletionReason::Eos)
        .unwrap();
    assert_eq!(scheduler.available_capacity(), 1);

    // Now we can admit another.
    scheduler.ingest_cache(make_decode_request(10)).unwrap();
    let admitted = scheduler.admit_sequences();
    assert_eq!(admitted.len(), 1);
}

// ── Statistics ──────────────────────────────────────────────────────

#[test]
fn stats_snapshot() {
    let scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());

    let req_id = RequestId::new();
    scheduler
        .ingest_cache(make_decode_request_with_id(req_id.clone(), 5))
        .unwrap();
    scheduler.admit_sequences();

    // Generate 3 tokens.
    scheduler.record_token(&req_id).unwrap();
    scheduler.record_token(&req_id).unwrap();
    scheduler.record_token(&req_id).unwrap();

    scheduler
        .mark_sequence_complete(&req_id, CompletionReason::Eos)
        .unwrap();

    let stats = scheduler.stats();
    assert_eq!(stats.total_ingested, 1);
    assert_eq!(stats.total_completed, 1);
    assert_eq!(stats.total_rejected, 0);
    assert_eq!(stats.total_failed, 0);
    assert_eq!(stats.total_tokens_generated, 3);
}

// ── Drain Events ────────────────────────────────────────────────────

#[test]
fn drain_events_clears_buffer() {
    let scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());
    let req_id = RequestId::new();

    scheduler
        .ingest_cache(make_decode_request_with_id(req_id.clone(), 5))
        .unwrap();
    scheduler.admit_sequences();
    scheduler
        .mark_sequence_complete(&req_id, CompletionReason::Eos)
        .unwrap();

    let events = scheduler.drain_completion_events();
    assert_eq!(events.len(), 1);

    // Second drain returns empty.
    let events2 = scheduler.drain_completion_events();
    assert!(events2.is_empty());
}

// ── Debug Display ───────────────────────────────────────────────────

#[test]
fn scheduler_debug_display() {
    let scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());
    let debug = format!("{scheduler:?}");
    assert!(debug.contains("DecodeScheduler"));
    assert!(debug.contains("max_batch_size"));
}

// ── DecodeSequence ──────────────────────────────────────────────────

#[test]
fn decode_sequence_token_limit() {
    let req = make_decode_request(3);
    let mut seq = DecodeSequence::from_request(&req);

    assert!(!seq.has_reached_limit());
    assert!(!seq.record_token());
    assert!(!seq.record_token());
    assert!(seq.record_token()); // 3rd token
    assert!(seq.has_reached_limit());
}

// ── Concurrent ingestion from multiple prefill nodes ────────────────

#[test]
fn concurrent_ingestion_multiple_requests() {
    let scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());

    // Simulate multiple prefill nodes sending caches.
    for _ in 0..10 {
        scheduler.ingest_cache(make_decode_request(50)).unwrap();
    }

    assert_eq!(scheduler.ingestion_queue_len(), 10);
    assert_eq!(scheduler.total_ingested(), 10);

    let admitted = scheduler.admit_sequences();
    assert_eq!(admitted.len(), 10); // default max_batch_size = 32
    assert_eq!(scheduler.active_batch_size(), 10);
}

// ── active_sequence_ids ─────────────────────────────────────────────

#[test]
fn active_sequence_ids_returns_decoding_sequences() {
    let scheduler = DecodeScheduler::new(DecodeSchedulerConfig::default());

    scheduler.ingest_cache(make_decode_request(50)).unwrap();
    scheduler.ingest_cache(make_decode_request(50)).unwrap();
    scheduler.admit_sequences();

    let ids = scheduler.active_sequence_ids();
    assert_eq!(ids.len(), 2);
    for (_id, status) in &ids {
        assert_eq!(*status, SequenceStatus::Decoding);
    }
}
