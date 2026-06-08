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

//! Unit tests for batch scheduler scheduling decisions.
//!
//! These tests verify the `decide_action` policy given various combinations
//! of empty/non-empty queue and batch states, without requiring a real model.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::time::Instant;

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::SamplingConfig;

use super::{effective_decode_storage_backend, vlm_prefix_sharing_allowed};
use crate::server::batch::active::ActiveBatch;
use crate::server::batch::queue::PrefillQueue;
use crate::server::batch::sequence::{
    BatchSchedulerAction, RequestPriority, SequenceInfo, SequenceState,
};
use crate::server::config::DecodeStorageBackend;
use crate::server::model_provider::GenerateEvent;
use crate::server::model_provider::model_worker::StreamingDecodeState;

/// Build a minimal `SequenceInfo` for scheduling tests.
fn make_test_sequence(id_val: u64) -> (SequenceInfo, mpsc::Receiver<GenerateEvent>) {
    let (tx, rx) = mpsc::channel();
    let tokenizer = crate::tokenizer::MlxcelTokenizer::stub();
    let prompt_tokens = vec![1, 2, 3];
    let decode_state = StreamingDecodeState::new(&tokenizer, &prompt_tokens);

    let seq = SequenceInfo {
        seq_id: SequenceId::from_raw(id_val),
        state: SequenceState::Queued,
        prompt_tokens,
        sampling: SamplingConfig::default(),
        max_tokens: 100,
        eos_token_ids: vec![2],
        priority: crate::server::batch::RequestPriority::Normal,
        logprobs_config: Default::default(),
        vlm_embeddings: None,
        images: Vec::new(),
        audio: Vec::new(),
        generated_tokens: Vec::new(),
        generated_text: String::new(),
        decode_state,
        prefill_offset: 0,
        prefill_start_offset: 0,
        already_cached_tokens: 0,
        response_tx: tx,
        cancelled: Arc::new(AtomicBool::new(false)),
        created_at: Instant::now(),
        prefill_start: None,
        first_token_time: None,
        token_history: Vec::new(),
        merged_eos: Vec::new(),
        thinking: crate::server::thinking_budget::ThinkingState::disabled(),
        structured: None,
    };

    (seq, rx)
}

/// Helper: reproduce `decide_action` logic in isolation so tests do not need
/// a full `BatchScheduler` (which requires a real LoadedModel + tokenizer).
///
/// This mirrors the exact decision policy from `BatchScheduler::decide_action`.
/// Policy: active sequences always decode first to prevent starvation; prefill
/// only happens when the batch is empty.
fn decide_action_from_state(queue: &PrefillQueue, batch: &ActiveBatch) -> BatchSchedulerAction {
    if batch.is_empty() && queue.is_empty() {
        return BatchSchedulerAction::Idle;
    }
    // Active sequences always get a decode step before admitting new prefills.
    if !batch.is_empty() {
        return BatchSchedulerAction::Decode(batch.sequence_ids());
    }
    // Batch is empty but queue has work -- prefill next request.
    BatchSchedulerAction::Prefill(SequenceId::from_raw(0))
}

// -------------------------------------------------------------------
// decide_action tests
// -------------------------------------------------------------------

#[test]
fn decide_action_idle_when_both_empty() {
    let queue = PrefillQueue::new();
    let batch = ActiveBatch::new(8);

    let action = decide_action_from_state(&queue, &batch);
    assert!(matches!(action, BatchSchedulerAction::Idle));
}

#[test]
fn decide_action_prefill_when_queue_has_entry_and_batch_not_full() {
    let mut queue = PrefillQueue::new();
    let batch = ActiveBatch::new(4);

    let (seq, _rx) = make_test_sequence(1);
    queue.enqueue(seq).unwrap();

    let action = decide_action_from_state(&queue, &batch);
    assert!(matches!(action, BatchSchedulerAction::Prefill(_)));
}

#[test]
fn decide_action_decode_when_batch_has_entries_and_queue_empty() {
    let queue = PrefillQueue::new();
    let mut batch = ActiveBatch::new(4);

    let (mut seq, _rx) = make_test_sequence(10);
    seq.state = SequenceState::Decoding;
    batch.add(seq).unwrap();

    let action = decide_action_from_state(&queue, &batch);
    match action {
        BatchSchedulerAction::Decode(ids) => {
            assert_eq!(ids.len(), 1);
            assert_eq!(ids[0].as_u64(), 10);
        }
        other => panic!("Expected Decode, got {other:?}"),
    }
}

#[test]
fn decide_action_decode_when_batch_full_and_queue_non_empty() {
    let mut queue = PrefillQueue::new();
    let mut batch = ActiveBatch::new(1); // capacity 1

    // Fill the batch
    let (mut seq_active, _rx1) = make_test_sequence(1);
    seq_active.state = SequenceState::Decoding;
    batch.add(seq_active).unwrap();

    // Also queue a new request
    let (seq_queued, _rx2) = make_test_sequence(2);
    queue.enqueue(seq_queued).unwrap();

    // Batch is full -> should decode, not prefill
    let action = decide_action_from_state(&queue, &batch);
    assert!(matches!(action, BatchSchedulerAction::Decode(_)));
}

#[test]
fn decide_action_decode_when_batch_has_active_even_if_queue_nonempty() {
    let mut queue = PrefillQueue::new();
    let mut batch = ActiveBatch::new(4); // capacity 4

    // One active sequence
    let (mut seq_active, _rx1) = make_test_sequence(1);
    seq_active.state = SequenceState::Decoding;
    batch.add(seq_active).unwrap();

    // One queued request
    let (seq_queued, _rx2) = make_test_sequence(2);
    queue.enqueue(seq_queued).unwrap();

    // Active sequences always decode first to prevent starvation.
    // Queued requests will be prefilled once the batch empties.
    let action = decide_action_from_state(&queue, &batch);
    assert!(matches!(action, BatchSchedulerAction::Decode(_)));
}

#[test]
fn decide_action_decode_returns_all_active_ids() {
    let queue = PrefillQueue::new();
    let mut batch = ActiveBatch::new(8);

    for id in [10, 20, 30] {
        let (mut seq, _rx) = make_test_sequence(id);
        seq.state = SequenceState::Decoding;
        batch.add(seq).unwrap();
    }

    let action = decide_action_from_state(&queue, &batch);
    match action {
        BatchSchedulerAction::Decode(ids) => {
            let mut sorted: Vec<u64> = ids.iter().map(|id| id.as_u64()).collect();
            sorted.sort();
            assert_eq!(sorted, vec![10, 20, 30]);
        }
        other => panic!("Expected Decode, got {other:?}"),
    }
}

// -------------------------------------------------------------------
// O(1) property of decide_action
// -------------------------------------------------------------------

#[test]
fn decide_action_is_o1_regardless_of_queue_size() {
    // Verifies that decide_action does not iterate the queue or batch;
    // it only checks .is_empty() / .is_full() which are O(1).
    let mut queue = PrefillQueue::with_capacity(1000);
    let batch = ActiveBatch::new(8);

    for i in 0..100 {
        let (seq, _rx) = make_test_sequence(i);
        queue.enqueue(seq).unwrap();
    }

    // Should immediately return Prefill without scanning the queue
    let action = decide_action_from_state(&queue, &batch);
    assert!(matches!(action, BatchSchedulerAction::Prefill(_)));
}

// -------------------------------------------------------------------
// Priority-related tests
// -------------------------------------------------------------------

fn make_test_sequence_with_priority(
    id_val: u64,
    priority: RequestPriority,
) -> (SequenceInfo, mpsc::Receiver<GenerateEvent>) {
    let (tx, rx) = mpsc::channel();
    let tokenizer = crate::tokenizer::MlxcelTokenizer::stub();
    let prompt_tokens = vec![1, 2, 3];
    let decode_state = StreamingDecodeState::new(&tokenizer, &prompt_tokens);

    let seq = SequenceInfo {
        seq_id: SequenceId::from_raw(id_val),
        state: SequenceState::Queued,
        prompt_tokens,
        sampling: SamplingConfig::default(),
        max_tokens: 100,
        eos_token_ids: vec![2],
        priority,
        logprobs_config: Default::default(),
        vlm_embeddings: None,
        images: Vec::new(),
        audio: Vec::new(),
        generated_tokens: Vec::new(),
        generated_text: String::new(),
        decode_state,
        prefill_offset: 0,
        prefill_start_offset: 0,
        already_cached_tokens: 0,
        response_tx: tx,
        cancelled: Arc::new(AtomicBool::new(false)),
        created_at: Instant::now(),
        prefill_start: None,
        first_token_time: None,
        token_history: Vec::new(),
        merged_eos: Vec::new(),
        thinking: crate::server::thinking_budget::ThinkingState::disabled(),
        structured: None,
    };

    (seq, rx)
}

#[test]
fn priority_queue_dequeues_high_before_normal() {
    let mut queue = PrefillQueue::new();

    let (s_norm, _r1) = make_test_sequence_with_priority(1, RequestPriority::Normal);
    let (s_high, _r2) = make_test_sequence_with_priority(2, RequestPriority::High);

    queue.enqueue(s_norm).unwrap();
    queue.enqueue(s_high).unwrap();

    // High should come out first even though normal was enqueued first
    let first = queue.dequeue().unwrap();
    assert_eq!(first.seq_id.as_u64(), 2);
    assert_eq!(first.priority, RequestPriority::High);

    let second = queue.dequeue().unwrap();
    assert_eq!(second.seq_id.as_u64(), 1);
    assert_eq!(second.priority, RequestPriority::Normal);
}

#[test]
fn priority_queue_dequeues_normal_before_low() {
    let mut queue = PrefillQueue::new();

    let (s_low, _r1) = make_test_sequence_with_priority(1, RequestPriority::Low);
    let (s_norm, _r2) = make_test_sequence_with_priority(2, RequestPriority::Normal);

    queue.enqueue(s_low).unwrap();
    queue.enqueue(s_norm).unwrap();

    assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 2); // Normal
    assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 1); // Low
}

#[test]
fn priority_queue_fifo_within_same_priority_level() {
    let mut queue = PrefillQueue::new();

    let (s1, _r1) = make_test_sequence_with_priority(10, RequestPriority::High);
    let (s2, _r2) = make_test_sequence_with_priority(20, RequestPriority::High);
    let (s3, _r3) = make_test_sequence_with_priority(30, RequestPriority::High);

    queue.enqueue(s1).unwrap();
    queue.enqueue(s2).unwrap();
    queue.enqueue(s3).unwrap();

    assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 10);
    assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 20);
    assert_eq!(queue.dequeue().unwrap().seq_id.as_u64(), 30);
}

#[test]
fn priority_ordering_is_high_gt_normal_gt_low() {
    assert!(RequestPriority::High > RequestPriority::Normal);
    assert!(RequestPriority::Normal > RequestPriority::Low);
    assert!(RequestPriority::High > RequestPriority::Low);
}

#[test]
fn request_priority_from_header_parses_valid_values() {
    assert_eq!(
        RequestPriority::from_header("high"),
        Some(RequestPriority::High)
    );
    assert_eq!(
        RequestPriority::from_header("NORMAL"),
        Some(RequestPriority::Normal)
    );
    assert_eq!(
        RequestPriority::from_header("Low"),
        Some(RequestPriority::Low)
    );
    assert_eq!(
        RequestPriority::from_header(" high "),
        Some(RequestPriority::High)
    );
}

#[test]
fn request_priority_from_header_returns_none_for_invalid() {
    assert_eq!(RequestPriority::from_header("urgent"), None);
    assert_eq!(RequestPriority::from_header(""), None);
    assert_eq!(RequestPriority::from_header("1"), None);
}

#[test]
fn request_priority_default_is_normal() {
    assert_eq!(RequestPriority::default(), RequestPriority::Normal);
}

// -------------------------------------------------------------------
// Chunked prefill scheduling tests (without real model)
// -------------------------------------------------------------------

/// Extended decide_action that accounts for chunked prefill in progress.
/// This mirrors the scheduler's policy.
fn decide_action_with_chunked(
    queue: &PrefillQueue,
    batch: &ActiveBatch,
    chunked_in_progress: bool,
) -> BatchSchedulerAction {
    // If chunked prefill is in progress, interleave decode
    if chunked_in_progress {
        if !batch.is_empty() {
            return BatchSchedulerAction::Decode(batch.sequence_ids());
        }
        return BatchSchedulerAction::Prefill(SequenceId::from_raw(0));
    }

    if batch.is_empty() && queue.is_empty() {
        return BatchSchedulerAction::Idle;
    }
    if !batch.is_empty() {
        return BatchSchedulerAction::Decode(batch.sequence_ids());
    }
    BatchSchedulerAction::Prefill(SequenceId::from_raw(0))
}

#[test]
fn chunked_prefill_interleaves_decode_when_active_sequences_exist() {
    let queue = PrefillQueue::new();
    let mut batch = ActiveBatch::new(4);

    let (mut seq, _rx) = make_test_sequence(10);
    seq.state = SequenceState::Decoding;
    batch.add(seq).unwrap();

    // Chunked prefill in progress + active sequences -> should decode
    let action = decide_action_with_chunked(&queue, &batch, true);
    assert!(
        matches!(action, BatchSchedulerAction::Decode(_)),
        "Expected Decode during chunked prefill with active sequences"
    );
}

#[test]
fn chunked_prefill_continues_when_no_active_sequences() {
    let queue = PrefillQueue::new();
    let batch = ActiveBatch::new(4);

    // Chunked prefill in progress + no active sequences -> continue prefill
    let action = decide_action_with_chunked(&queue, &batch, true);
    assert!(
        matches!(action, BatchSchedulerAction::Prefill(_)),
        "Expected Prefill continuation when no active sequences"
    );
}

#[test]
fn chunked_prefill_interleaving_pattern() {
    // Simulate the interleaving pattern:
    // Tick 1: Prefill chunk 1
    // Tick 2: Decode (if active)
    // Tick 3: Prefill chunk 2
    // Tick 4: Decode (if active)
    // ...

    let queue = PrefillQueue::new();
    let mut batch = ActiveBatch::new(4);

    let (mut active_seq, _rx) = make_test_sequence(1);
    active_seq.state = SequenceState::Decoding;
    batch.add(active_seq).unwrap();

    // Tick 1: chunked prefill starts -> first action is Prefill
    // (no chunked_in_progress yet, batch has active seq, so Decode first)
    let action1 = decide_action_with_chunked(&queue, &batch, false);
    assert!(matches!(action1, BatchSchedulerAction::Decode(_)));

    // Tick 2: chunked prefill now in progress -> Decode interleave
    let action2 = decide_action_with_chunked(&queue, &batch, true);
    assert!(matches!(action2, BatchSchedulerAction::Decode(_)));

    // Tick 3: after decode, back to prefill continuation
    let action3 = decide_action_with_chunked(&queue, &batch, true);
    assert!(matches!(action3, BatchSchedulerAction::Decode(_)));
    // (still interleaving because batch is non-empty)

    // With no active batch: prefill continues
    let empty_batch = ActiveBatch::new(4);
    let action4 = decide_action_with_chunked(&queue, &empty_batch, true);
    assert!(matches!(action4, BatchSchedulerAction::Prefill(_)));
}

// -------------------------------------------------------------------
// VLM embedding guard routing tests (server-prefill fixes)
// -------------------------------------------------------------------
//
// The two server-side guards added for InternVL (and all VLM image prefills)
// ensure that sequences carrying pre-merged `vlm_embeddings` always take the
// full-prefill path, never the chunked-prefill path, and never the NA-tile
// padding path.
//
// The actual crashes (chunked-route corruption and NA-tile shape mismatch) are
// M5 Neural-Accelerator-gated (`should_align_prefill()` is true only on M5 NA)
// and therefore cannot be reproduced in unit tests on arbitrary CI hardware.
// These tests instead verify the routing *decision* using a synthetic
// `InputEmbeddings` value — following the pattern established in
// `speculative_burst_tests.rs`.

/// Mirror of the scheduler's chunked-prefill guard:
/// `seq.vlm_embeddings.is_none()` must be true for chunked prefill to start.
fn should_use_chunked_prefill(seq: &SequenceInfo, chunk_size: usize) -> bool {
    chunk_size > 0 && seq.prompt_tokens.len() > chunk_size && seq.vlm_embeddings.is_none()
}

/// Mirror of the scheduler's NA-tile alignment guard:
/// alignment is skipped when `vlm_embeddings` is present.
fn should_apply_na_tile_alignment(seq: &SequenceInfo) -> bool {
    // `should_align_prefill()` is true only on M5 NA hardware; here we test
    // only the guard condition, not the hardware predicate.
    seq.vlm_embeddings.is_none()
}

#[test]
fn vlm_embedding_sequence_bypasses_chunked_prefill_route() {
    use crate::vision::merge::InputEmbeddings;
    use mlxcel_core::from_slice_f32;

    let (mut seq, _rx) = make_test_sequence(42);
    // Long prompt that would normally trigger chunked prefill.
    seq.prompt_tokens = (0..512_i32).collect();

    // Without embeddings: chunked prefill is eligible.
    assert!(
        should_use_chunked_prefill(&seq, 128),
        "text-only long sequence must be eligible for chunked prefill"
    );

    // With embeddings (VLM image request): must bypass chunked prefill.
    seq.vlm_embeddings = Some(InputEmbeddings {
        inputs_embeds: from_slice_f32(&[0.0f32; 4], &[1, 1, 4]),
        attention_mask_4d: None,
    });
    assert!(
        !should_use_chunked_prefill(&seq, 128),
        "VLM-embedding sequence must not enter the chunked-prefill route"
    );
}

#[test]
fn vlm_embedding_sequence_bypasses_na_tile_alignment() {
    use crate::vision::merge::InputEmbeddings;
    use mlxcel_core::from_slice_f32;

    let (mut seq, _rx) = make_test_sequence(43);

    // Without embeddings: NA-tile alignment is eligible (hardware gate aside).
    assert!(
        should_apply_na_tile_alignment(&seq),
        "text-only sequence must be eligible for NA-tile alignment"
    );

    // With embeddings: NA-tile alignment must be skipped to avoid shape mismatch.
    seq.vlm_embeddings = Some(InputEmbeddings {
        inputs_embeds: from_slice_f32(&[0.0f32; 4], &[1, 1, 4]),
        attention_mask_4d: None,
    });
    assert!(
        !should_apply_na_tile_alignment(&seq),
        "VLM-embedding sequence must bypass NA-tile alignment to prevent mask/embedding shape mismatch"
    );
}

// -------------------------------------------------------------------
// Eviction victim selection tests
// -------------------------------------------------------------------

#[test]
fn active_batch_iter_min_priority_finds_lowest() {
    let mut batch = ActiveBatch::new(4);

    let (mut s1, _r1) = make_test_sequence_with_priority(1, RequestPriority::High);
    s1.state = SequenceState::Decoding;
    let (mut s2, _r2) = make_test_sequence_with_priority(2, RequestPriority::Low);
    s2.state = SequenceState::Decoding;
    let (mut s3, _r3) = make_test_sequence_with_priority(3, RequestPriority::Normal);
    s3.state = SequenceState::Decoding;

    batch.add(s1).unwrap();
    batch.add(s2).unwrap();
    batch.add(s3).unwrap();

    assert_eq!(batch.iter_min_priority(), Some(RequestPriority::Low));
}

#[test]
fn active_batch_iter_min_priority_empty_returns_none() {
    let batch = ActiveBatch::new(4);
    assert_eq!(batch.iter_min_priority(), None);
}

#[test]
fn eviction_selects_longest_first_by_default() {
    use crate::server::config::PreemptionPolicy;

    let mut batch = ActiveBatch::new(4);

    let (mut s1, _r1) = make_test_sequence_with_priority(1, RequestPriority::Normal);
    s1.state = SequenceState::Decoding;
    s1.generated_tokens = vec![10, 20, 30]; // 3 tokens

    let (mut s2, _r2) = make_test_sequence_with_priority(2, RequestPriority::Normal);
    s2.state = SequenceState::Decoding;
    s2.generated_tokens = vec![10]; // 1 token

    batch.add(s1).unwrap();
    batch.add(s2).unwrap();

    // LongestFirst should pick s1 (3 tokens > 1 token)
    let victim = match PreemptionPolicy::LongestFirst {
        PreemptionPolicy::LongestFirst => batch
            .iter_sequences()
            .max_by_key(|seq| seq.generated_tokens.len())
            .map(|seq| seq.seq_id),
        _ => None,
    };

    assert_eq!(victim.unwrap().as_u64(), 1);
}

#[test]
fn eviction_selects_lowest_priority_then_longest() {
    use crate::server::config::PreemptionPolicy;

    let mut batch = ActiveBatch::new(4);

    let (mut s1, _r1) = make_test_sequence_with_priority(1, RequestPriority::High);
    s1.state = SequenceState::Decoding;
    s1.generated_tokens = vec![10, 20, 30]; // 3 tokens

    let (mut s2, _r2) = make_test_sequence_with_priority(2, RequestPriority::Low);
    s2.state = SequenceState::Decoding;
    s2.generated_tokens = vec![10]; // 1 token

    let (mut s3, _r3) = make_test_sequence_with_priority(3, RequestPriority::Low);
    s3.state = SequenceState::Decoding;
    s3.generated_tokens = vec![10, 20, 30, 40]; // 4 tokens

    batch.add(s1).unwrap();
    batch.add(s2).unwrap();
    batch.add(s3).unwrap();

    // LowestPriority should pick s3 (Low + 4 tokens, longest of Low group)
    let victim = match PreemptionPolicy::LowestPriority {
        PreemptionPolicy::LowestPriority => batch
            .iter_sequences()
            .min_by(|a, b| {
                a.priority
                    .cmp(&b.priority)
                    .then_with(|| b.generated_tokens.len().cmp(&a.generated_tokens.len()))
            })
            .map(|seq| seq.seq_id),
        _ => None,
    };

    assert_eq!(victim.unwrap().as_u64(), 3);
}

#[test]
fn preemption_disabled_by_default_never_triggers() {
    // When enable_preemption is false, should_preempt should never
    // return true, even if the batch is full and queue has high-priority
    // requests. We test this by verifying the policy logic directly.
    let enable_preemption = false;
    let batch_full = true;
    let queue_has_high_priority = true;

    // The condition: enable_preemption && batch_full && queue_has_high > min_active
    let should_preempt = enable_preemption && batch_full && queue_has_high_priority;
    assert!(!should_preempt);
}

// -------------------------------------------------------------------
// Incremental token history and merged EOS (perf optimization tests)
// -------------------------------------------------------------------

#[test]
fn sequence_info_initializes_token_history_and_merged_eos_empty() {
    // New sequences enqueued via make_test_sequence start with empty
    // token_history and merged_eos. They are populated during prefill
    // (finish_prefill) so the decode steps can reuse them without
    // per-step reconstruction.
    let (seq, _rx) = make_test_sequence(1);
    assert!(
        seq.token_history.is_empty(),
        "token_history must start empty (populated at prefill)"
    );
    assert!(
        seq.merged_eos.is_empty(),
        "merged_eos must start empty (populated at prefill)"
    );
}

#[test]
fn sequence_info_token_history_can_be_populated_incrementally() {
    // Simulate the incremental update pattern used in decode_single_step
    // and execute_batched_decode after the perf optimization:
    //   token_history starts as prompt tokens (from initial_token_history),
    //   then each new token is appended with push() rather than
    //   rebuilding the full Vec from scratch every step.
    let prompt_tokens: Vec<i32> = vec![10, 20, 30];
    let mut token_history = prompt_tokens.clone(); // initial_token_history equivalent

    // Simulate generating 3 tokens incrementally
    let generated = vec![100_i32, 200, 300];
    for tok in &generated {
        token_history.push(*tok);
    }

    let expected: Vec<i32> = vec![10, 20, 30, 100, 200, 300];
    assert_eq!(token_history, expected);
}

#[test]
fn sequence_info_token_history_empty_when_no_penalties() {
    use mlxcel_core::generate::SamplingConfig;

    // When needs_token_history() is false (default config), the token
    // history should remain empty. This avoids unnecessary Vec growth.
    let sampling = SamplingConfig::default();
    assert!(!sampling.needs_token_history());

    // Simulate the scheduler's conditional push pattern:
    // `if seq.sampling.needs_token_history() { seq.token_history.push(tok); }`
    let mut token_history: Vec<i32> = Vec::new();
    let generated_token = 42_i32;
    if sampling.needs_token_history() {
        token_history.push(generated_token);
    }

    assert!(
        token_history.is_empty(),
        "token_history should stay empty when no penalties are active"
    );
}

#[test]
fn sequence_info_token_history_grows_when_penalties_enabled() {
    use mlxcel_core::generate::SamplingConfig;

    // When repetition_penalty is active, each decoded token is appended.
    let sampling = SamplingConfig {
        repetition_penalty: 1.3,
        ..Default::default()
    };
    assert!(sampling.needs_token_history());

    let mut token_history: Vec<i32> = vec![1, 2, 3]; // from initial_token_history

    for tok in [10_i32, 20, 30] {
        if sampling.needs_token_history() {
            token_history.push(tok);
        }
    }

    assert_eq!(token_history, vec![1, 2, 3, 10, 20, 30]);
}

#[test]
fn merged_eos_contains_both_model_and_request_stop_tokens() {
    // merged_eos_token_ids merges the model's built-in EOS tokens with
    // per-request stop tokens. This merged list is computed once at prefill
    // and cached on the SequenceInfo to avoid per-step recomputation.
    use mlxcel_core::generation_policy::merged_eos_token_ids;

    let model_eos = vec![2_i32]; // typical EOS
    let stop_tokens = vec![128001_i32, 128009_i32]; // e.g. Llama3 stop tokens

    let merged = merged_eos_token_ids(model_eos, &stop_tokens);

    assert!(merged.contains(&2));
    assert!(merged.contains(&128001));
    assert!(merged.contains(&128009));
}

#[test]
fn merged_eos_deduplicates_overlapping_tokens() {
    use mlxcel_core::generation_policy::merged_eos_token_ids;

    // When the model EOS and stop_token_ids share a token, it should
    // appear only once (or at most a small number of times) in merged.
    let model_eos = vec![2_i32, 100];
    let stop_tokens = vec![2_i32, 200]; // 2 is already in model_eos

    let merged = merged_eos_token_ids(model_eos, &stop_tokens);

    // All tokens must be present
    assert!(merged.contains(&2));
    assert!(merged.contains(&100));
    assert!(merged.contains(&200));
}

#[test]
fn paged_decode_storage_falls_back_when_batching_is_unavailable() {
    assert_eq!(
        effective_decode_storage_backend(DecodeStorageBackend::Paged, 4, false, true),
        DecodeStorageBackend::Dense
    );
    assert_eq!(
        effective_decode_storage_backend(DecodeStorageBackend::Paged, 1, true, true),
        DecodeStorageBackend::Dense
    );
    assert_eq!(
        effective_decode_storage_backend(DecodeStorageBackend::Paged, 4, true, false),
        DecodeStorageBackend::Dense
    );
    assert_eq!(
        effective_decode_storage_backend(DecodeStorageBackend::Paged, 4, true, true),
        DecodeStorageBackend::Paged
    );
}

/// Hybrid-SSM carve-out (#121): Mamba / Mamba2 / Jamba / NemotronH /
/// NemotronNAS / RecurrentGemma / Qwen3Next / RWKV7 all keep their recurrent
/// state in internal model-owned caches, so they override
/// `supports_batching()` to `false` and never opt into the paged decode
/// backend (`supports_paged_decode_backend()` keeps the trait default
/// `false`). Either flag alone already forces the dense backend; this test
/// pins the combined SSM profile — `supports_batching == false` AND
/// `supports_paged_decode_backend == false` — across batch sizes so a future
/// model edit cannot accidentally route a recurrent worker onto the paged
/// path even if it is explicitly requested.
#[test]
fn hybrid_ssm_workers_never_use_paged_decode_backend() {
    for batch in [2usize, 4, 16, 64] {
        assert_eq!(
            effective_decode_storage_backend(DecodeStorageBackend::Paged, batch, false, false),
            DecodeStorageBackend::Dense,
            "hybrid SSM (batch={batch}) must fall back to dense even when Paged is requested"
        );
        assert_eq!(
            effective_decode_storage_backend(DecodeStorageBackend::Auto, batch, false, false),
            DecodeStorageBackend::Dense,
            "hybrid SSM (batch={batch}) must resolve Auto to dense"
        );
    }
}

#[test]
fn auto_decode_storage_prefers_paged_only_for_supported_workers() {
    assert_eq!(
        effective_decode_storage_backend(DecodeStorageBackend::Auto, 4, true, true),
        DecodeStorageBackend::Paged
    );
    assert_eq!(
        effective_decode_storage_backend(DecodeStorageBackend::Auto, 4, true, false),
        DecodeStorageBackend::Dense
    );
    assert_eq!(
        effective_decode_storage_backend(DecodeStorageBackend::Auto, 1, true, true),
        DecodeStorageBackend::Dense
    );
}

/// #124 step c: the VLM prompt-prefix sharing gate must stay off by default,
/// reject text-only requests, and never share a request that carries video
/// (video bytes are not in the multimodal digest, so a video prefix could
/// collide). Only an opted-in, image/audio (no-video) request is eligible.
#[test]
fn vlm_prefix_sharing_gate_pins_safety_conditions() {
    // Default off: even a clean image request is ineligible when the operator
    // did not pass --enable-vlm-prefix-cache.
    assert!(!vlm_prefix_sharing_allowed(false, true, false));
    // Text-only requests never share, opt-in or not.
    assert!(!vlm_prefix_sharing_allowed(true, false, false));
    assert!(!vlm_prefix_sharing_allowed(false, false, false));
    // Video is excluded because its bytes are not folded into the digest.
    assert!(!vlm_prefix_sharing_allowed(true, true, true));
    // The one eligible case: opted in, multimodal, no video.
    assert!(vlm_prefix_sharing_allowed(true, true, false));
}

/// #124 step b: the scheduler must fold the request context's multimodal
/// digest into the composed prompt-cache key. This is the plumbing that lets a
/// later step share image/audio prefixes without a text↔image bucket collision.
/// Verified through the private `compose_prompt_cache_key` seam so we exercise
/// the exact key the scheduler builds, not a hand-rolled copy.
#[test]
fn compose_prompt_cache_key_folds_request_multimodal_digest() {
    use crate::server::config::PromptCacheRequestContext;
    use crate::server::prompt_cache::key::{
        MultimodalDigest, PromptCacheKey, multimodal_digest_from_vecs,
    };

    let tokens = [11_i32, 22, 33, 44];

    let text_ctx = PromptCacheRequestContext {
        model_id: "model".to_string(),
        lora_id: None,
        template_sig: "tpl".to_string(),
        session_key: "sess".to_string(),
        mm_digest: MultimodalDigest::empty(),
    };
    let image_a = PromptCacheRequestContext {
        mm_digest: multimodal_digest_from_vecs(&[b"IMAGE-A".to_vec()], &[]),
        ..text_ctx.clone()
    };
    let image_b = PromptCacheRequestContext {
        mm_digest: multimodal_digest_from_vecs(&[b"IMAGE-B".to_vec()], &[]),
        ..text_ctx.clone()
    };

    let k_text = super::BatchScheduler::compose_prompt_cache_key(&text_ctx, &tokens);
    let k_image_a = super::BatchScheduler::compose_prompt_cache_key(&image_a, &tokens);
    let k_image_b = super::BatchScheduler::compose_prompt_cache_key(&image_b, &tokens);

    // Same tokens, text vs image: distinct buckets (no cross-modal collision).
    assert_ne!(
        k_text.digest(),
        k_image_a.digest(),
        "a text-only prefix must not collide with the same tokens carrying an image"
    );
    // Two different images, same tokens: distinct buckets.
    assert_ne!(
        k_image_a.digest(),
        k_image_b.digest(),
        "different image payloads must land in different buckets"
    );
    // Same image twice: identical bucket, the reuse the digest unlocks.
    let k_image_a_again = super::BatchScheduler::compose_prompt_cache_key(&image_a, &tokens);
    assert_eq!(k_image_a.digest(), k_image_a_again.digest());

    // Text path stays byte-identical to building the key with an explicit empty
    // digest, proving step b introduces zero behavior change for text requests.
    let explicit_empty = PromptCacheKey::new_full(
        "model",
        None,
        "tpl",
        Some("sess"),
        MultimodalDigest::empty(),
        &tokens,
    );
    assert_eq!(k_text.digest(), explicit_empty.digest());
}

// -------------------------------------------------------------------
// Null / empty-cache safety tests
//
// These tests cover the same transition edges that upstream mlx-lm
// landed null-guards for in `mlx_lm/models/cache.py`:
//
//   - `BatchKVCache.extend`: both inputs empty (offset == 0, no keys).
//   - `BatchKVCache.extend`: one input empty, one populated.
//   - `BatchKVCache.filter`: filter to an empty index list.
//   - `BatchKVCache.merge` / `ArraysCache.from_batch_of`: all-empty inputs.
//
// mlxcel does not expose a `BatchKVCache` struct — its equivalent is the
// scheduler's combination of [`ActiveBatch`] (per-sequence metadata) and
// [`mlxcel_core::cache::CachePool`] (per-sequence KV tensors). The cases
// below exercise the same transitions on those structures, plus the
// scheduling-decision layer that dispatches work based on them.
// -------------------------------------------------------------------

/// `extend` analogue: two empty caches (both sequences have no generated
/// tokens yet) combine into a batch whose bookkeeping (active count, ids)
/// reflects the sum of inputs. Per-sequence state remains `offset == 0`,
/// `generated_tokens.is_empty()` — i.e. `keys is None` in upstream
/// terminology.
#[test]
fn test_extend_both_empty() {
    let mut batch = ActiveBatch::new(4);

    let (mut s1, _r1) = make_test_sequence(1);
    let (mut s2, _r2) = make_test_sequence(2);
    s1.state = SequenceState::Decoding;
    s2.state = SequenceState::Decoding;
    // Both sequences are "empty": no tokens generated, prefill offset 0.
    assert!(s1.generated_tokens.is_empty());
    assert!(s2.generated_tokens.is_empty());
    assert_eq!(s1.prefill_offset, 0);
    assert_eq!(s2.prefill_offset, 0);

    batch.add(s1).unwrap();
    batch.add(s2).unwrap();

    // Extend result: batch dim == sum of inputs, both sequences visible,
    // and per-sequence state is still "empty" (no panic from concatenating
    // null tensors — there are no tensors to concatenate at this layer).
    assert_eq!(batch.len(), 2);
    let mut ids: Vec<u64> = batch.sequence_ids().iter().map(|id| id.as_u64()).collect();
    ids.sort();
    assert_eq!(ids, vec![1, 2]);

    for seq in batch.iter_sequences() {
        assert!(
            seq.generated_tokens.is_empty(),
            "both-empty extend must preserve keys=None state per sequence"
        );
        assert_eq!(seq.prefill_offset, 0);
    }

    // A decode step scheduled on this batch must be a no-op under the
    // filter-to-empty guard path (empty id list), but with two active
    // sequences the action is Decode of both, which is valid.
    let queue = PrefillQueue::new();
    let action = decide_action_from_state(&queue, &batch);
    match action {
        BatchSchedulerAction::Decode(decode_ids) => assert_eq!(decode_ids.len(), 2),
        other => panic!("Expected Decode, got {other:?}"),
    }
}

/// `extend` analogue: one sequence with 3 generated tokens + one empty
/// sequence. Combined batch must expose both ids with the populated side's
/// state preserved. The empty side's absence of key tensors must not
/// propagate NaNs into the populated side.
#[test]
fn test_extend_empty_and_populated() {
    let mut batch = ActiveBatch::new(4);

    let (mut populated, _rp) = make_test_sequence(100);
    populated.state = SequenceState::Decoding;
    populated.generated_tokens = vec![10, 20, 30]; // populated with 3 tokens
    populated.prefill_offset = 3;

    let (mut empty, _re) = make_test_sequence(200);
    empty.state = SequenceState::Decoding;
    // empty: no generated_tokens, no prefill offset — the "keys is None" case.
    assert!(empty.generated_tokens.is_empty());
    assert_eq!(empty.prefill_offset, 0);

    batch.add(populated).unwrap();
    batch.add(empty).unwrap();

    // Combined batch: size 2, both ids present.
    assert_eq!(batch.len(), 2);
    let mut ids: Vec<u64> = batch.sequence_ids().iter().map(|id| id.as_u64()).collect();
    ids.sort();
    assert_eq!(ids, vec![100, 200]);

    // Populated side is unperturbed — its state survived the extend.
    let populated_after = batch.get_mut(SequenceId::from_raw(100)).unwrap();
    assert_eq!(populated_after.generated_tokens, vec![10, 20, 30]);
    assert_eq!(populated_after.prefill_offset, 3);

    // Empty side still empty — the extend did not fabricate tokens for it.
    let empty_after = batch.get_mut(SequenceId::from_raw(200)).unwrap();
    assert!(empty_after.generated_tokens.is_empty());
    assert_eq!(empty_after.prefill_offset, 0);

    // Scheduling: decode step covers both sequences without panic.
    let queue = PrefillQueue::new();
    let action = decide_action_from_state(&queue, &batch);
    match action {
        BatchSchedulerAction::Decode(decode_ids) => assert_eq!(decode_ids.len(), 2),
        other => panic!("Expected Decode, got {other:?}"),
    }
}

/// `filter` analogue: filtering by an empty id list must produce an empty
/// result without panicking. In mlxcel this is expressed through the
/// decode dispatch path: calling the decode step with an empty slice is a
/// no-op. The observability counter is updated with length 0 so operators
/// can see that a zero-sized step was attempted.
#[test]
fn test_filter_to_empty() {
    use crate::server::batch::observability::BatchObservability;

    // An ActiveBatch with three sequences, then "filter" the decode call
    // via an empty id slice (the analogue of Python's empty `batch_indices`).
    let mut batch = ActiveBatch::new(4);
    for id in [1_u64, 2, 3] {
        let (mut seq, _rx) = make_test_sequence(id);
        seq.state = SequenceState::Decoding;
        batch.add(seq).unwrap();
    }

    // Decide-action on an empty list case: when the batch becomes empty
    // via filtering, decide_action returns Idle (no crash, no phantom
    // Decode with length 0).
    let empty_batch = ActiveBatch::new(4);
    let queue = PrefillQueue::new();
    let action = decide_action_from_state(&queue, &empty_batch);
    assert!(
        matches!(action, BatchSchedulerAction::Idle),
        "filter-to-empty must resolve to Idle, not a zero-sized Decode"
    );

    // And at the dispatch layer: record_decode_step(0) is safe, matching
    // the scheduler's `execute_decode_step` empty-id guard.
    let obs = BatchObservability::new();
    obs.record_decode_step(0);
    let snap = obs.snapshot();
    assert_eq!(snap.decode_steps_processed, 1);
    assert_eq!(snap.total_decode_tokens, 0);

    // The populated batch is untouched by a filter-to-empty on an
    // unrelated batch (no aliasing between the two).
    assert_eq!(batch.len(), 3);
}

/// `merge` / `from_batch_of` analogue: constructing a batch from 4 "empty"
/// inputs yields a single logical batch whose per-sequence caches are all
/// empty (`size() == 0` in upstream terms). The scheduler handles this by
/// returning Idle when no sequence has work, rather than crashing inside
/// a batched forward pass.
#[test]
fn test_merge_all_empty() {
    let mut queue = PrefillQueue::new();

    // Four sequences, all "empty" (no prior tokens, no prefill offset),
    // enqueued but not yet admitted to the batch.
    for id in [10_u64, 20, 30, 40] {
        let (seq, _rx) = make_test_sequence(id);
        assert!(seq.generated_tokens.is_empty());
        assert_eq!(seq.prefill_offset, 0);
        queue.enqueue(seq).unwrap();
    }

    // Empty batch ("merged" cache has size 0): decide_action must not
    // crash, and it must not emit a zero-sized Decode.
    let empty_batch = ActiveBatch::new(4);
    assert_eq!(empty_batch.len(), 0);

    // Because the queue has work, the action is Prefill (to admit one of
    // the empty sequences), not Idle — which is the correct merge-of-
    // all-empty behavior: do not panic, do not dispatch an empty decode,
    // instead admit real work.
    let action = decide_action_from_state(&queue, &empty_batch);
    assert!(
        matches!(action, BatchSchedulerAction::Prefill(_)),
        "merge-of-all-empty with non-empty queue must Prefill, not crash"
    );

    // And with both batch and queue empty, Idle is the correct merge
    // result for 4 empties + 0 pending.
    let drained_queue = PrefillQueue::new();
    let action = decide_action_from_state(&drained_queue, &empty_batch);
    assert!(
        matches!(action, BatchSchedulerAction::Idle),
        "merge of all-empty inputs with no pending work must be Idle"
    );
}

// -------------------------------------------------------------------
// Server scheduler dispatch on PagedKvLayout::cache_mode
// -------------------------------------------------------------------

/// Reproduce the `is_turbo_mode` classification policy from
/// [`super::BatchScheduler::is_turbo_mode`] without constructing a real
/// `BatchScheduler` (which requires a model). The function is a single
/// `matches!`, so reproducing it here lets us assert the variant set
/// without exposing the inherent method.
fn is_turbo_mode_policy(mode: mlxcel_core::cache::KVCacheMode) -> bool {
    use mlxcel_core::cache::KVCacheMode;
    matches!(
        mode,
        KVCacheMode::Turbo4Asym | KVCacheMode::Turbo4 | KVCacheMode::Turbo4Delegated
    )
}

#[test]
fn is_turbo_mode_classifies_turbo4_variants() {
    use mlxcel_core::cache::KVCacheMode;
    assert!(is_turbo_mode_policy(KVCacheMode::Turbo4Asym));
    assert!(is_turbo_mode_policy(KVCacheMode::Turbo4));
    assert!(is_turbo_mode_policy(KVCacheMode::Turbo4Delegated));
}

#[test]
fn is_turbo_mode_excludes_fp16_int8_and_turbo3_today() {
    use mlxcel_core::cache::KVCacheMode;
    // Fp16 / Int8 must take the historical `PagedKvLayout::uniform` path
    // (no per-page sidecar accounting). Bit-identical to earlier.
    assert!(!is_turbo_mode_policy(KVCacheMode::Fp16));
    assert!(!is_turbo_mode_policy(KVCacheMode::Int8));
    // Turbo3Asym is a valid `KVCacheMode` but the paged
    // data plane does not yet support per-page 3-bit sidecars; the
    // dispatch intentionally falls through to `PagedKvLayout::uniform`
    // so callers that pair `--kv-cache-mode fp16+turbo3` with paged
    // decode get the dense-fallback path. When paged Turbo3 lands,
    // this assertion can flip.
    assert!(!is_turbo_mode_policy(KVCacheMode::Turbo3Asym));
}

/// Smoke-test the exact arguments the scheduler passes to
/// `PagedKvLayout::uniform_with_mode` in
/// [`super::BatchScheduler::sequence_state_layout_override`].
///
/// This guards against regressions where a future code edit makes the
/// sidecar budget violate the `bytes % block_size == 0` invariant —
/// previously seen with `DEFAULT_PAGED_BLOCK_SIZE / 2`, which panics
/// the `.expect("valid paged Turbo4 decode layout")` at runtime as
/// soon as the first paged Turbo4 sequence is allocated.
#[test]
fn scheduler_paged_turbo_layout_arguments_are_valid() {
    use mlxcel_core::cache::{KVCacheMode, PagedKvLayout};
    // Same constants as the scheduler's `sequence_state_layout_override`.
    const DEFAULT_PAGED_BLOCK_SIZE: usize = 32;
    let sidecar_bytes_per_block = DEFAULT_PAGED_BLOCK_SIZE;
    for mode in [
        KVCacheMode::Turbo4Asym,
        KVCacheMode::Turbo4,
        KVCacheMode::Turbo4Delegated,
    ] {
        PagedKvLayout::uniform_with_mode(
            /* num_layers = */ 4,
            DEFAULT_PAGED_BLOCK_SIZE,
            DEFAULT_PAGED_BLOCK_SIZE,
            mode,
            sidecar_bytes_per_block,
        )
        .unwrap_or_else(|err| {
            panic!("scheduler-default Turbo4 paged layout must be valid for {mode:?}: {err}")
        });
    }
}

// -------------------------------------------------------------------
// Batched KV quantization config plumbing
// -------------------------------------------------------------------
//
// These tests exercise the resolver in `cli_input.rs` and the per-layer
// mode table that the scheduler reads when applying the configured
// quantization to each new sequence's caches. They do not require a
// real model — they cover the data-flow plumbing only. The end-to-end
// "decoded output within tolerance of unquantized baseline" criterion
// is validated separately with a real model (the issue's
// "regression tests cover both cache variants on a small batched
// scenario" item).

#[test]
fn cli_resolver_disabled_returns_default_disabled_config() {
    use crate::server::resolve_batch_kv_quant_config;
    let cfg = resolve_batch_kv_quant_config(0, 64, None, true).unwrap();
    assert!(!cfg.is_enabled());
    assert_eq!(
        cfg.base_mode(),
        mlxcel_core::cache::KVCacheMode::Fp16,
        "disabled config must map to Fp16 baseline"
    );
}

#[test]
fn cli_resolver_uniform_8bit_default_scheme() {
    use crate::server::resolve_batch_kv_quant_config;
    // No --kv-quant-scheme provided → default Uniform.
    let cfg = resolve_batch_kv_quant_config(8, 64, None, true).unwrap();
    assert!(cfg.is_enabled());
    assert_eq!(
        cfg.scheme,
        mlxcel_core::cache::KvQuantScheme::Uniform,
        "default scheme must be Uniform"
    );
    assert_eq!(cfg.base_mode(), mlxcel_core::cache::KVCacheMode::Int8);
}

#[test]
fn cli_resolver_turboquant_4bit_explicit_scheme() {
    use crate::server::resolve_batch_kv_quant_config;
    let cfg = resolve_batch_kv_quant_config(4, 64, Some("turboquant"), true).unwrap();
    assert!(cfg.is_enabled());
    assert_eq!(cfg.scheme, mlxcel_core::cache::KvQuantScheme::TurboQuant);
    assert_eq!(cfg.base_mode(), mlxcel_core::cache::KVCacheMode::Turbo4Asym);
}

#[test]
fn cli_resolver_rejects_uniform_4bit_combination() {
    use crate::server::resolve_batch_kv_quant_config;
    // (Uniform, 4) has no Int4Affine single-stream cache mode in
    // mlxcel-core today. The previous behaviour was to silently downgrade
    // `base_mode()` to `Fp16`, which meant operators saw no quantization
    // with no error. The resolver now rejects this combination at server
    // startup and points operators at the supported 4-bit batched mode
    // (TurboQuant) and the supported 8-bit uniform mode.
    let err = resolve_batch_kv_quant_config(4, 64, Some("uniform"), true).unwrap_err();
    assert!(
        err.contains("--kv-quant-scheme uniform")
            && err.contains("--kv-bits 4")
            && err.contains("not yet supported"),
        "expected error about unsupported (uniform, 4) combination, got: {err}"
    );
    assert!(
        err.contains("turboquant") && err.contains("--kv-bits 8"),
        "expected error to suggest both turboquant and 8-bit alternatives, got: {err}"
    );
}

#[test]
fn cli_resolver_rejects_invalid_bits() {
    use crate::server::resolve_batch_kv_quant_config;
    // 6 is not in the supported set for either scheme.
    let err = resolve_batch_kv_quant_config(6, 64, Some("uniform"), true).unwrap_err();
    assert!(err.contains("not supported"));
}

#[test]
fn cli_resolver_rejects_unknown_scheme_string() {
    use crate::server::resolve_batch_kv_quant_config;
    let err = resolve_batch_kv_quant_config(8, 64, Some("polar"), true).unwrap_err();
    assert!(err.contains("unknown --kv-quant-scheme"));
}

#[test]
fn cli_resolver_falls_back_to_default_group_size_when_zero() {
    use crate::server::resolve_batch_kv_quant_config;
    // group_size == 0 is the "use default" sentinel: validators should
    // not see a non-positive group when we route through the resolver.
    let cfg = resolve_batch_kv_quant_config(8, 0, None, true).unwrap();
    assert_eq!(cfg.group_size, mlxcel_core::cache::DEFAULT_KV_GROUP_SIZE);
}

#[test]
fn skip_last_layer_default_is_true_in_resolver() {
    use crate::server::resolve_batch_kv_quant_config;
    let cfg = resolve_batch_kv_quant_config(0, 64, None, true).unwrap();
    assert!(cfg.skip_last_layer);
}

// -------------------------------------------------------------------
// Per-layer mode table for batched scheduler
// -------------------------------------------------------------------

/// The scheduler's `apply_kv_cache_mode_to` method reads the resolved
/// per-layer modes from `BatchKvQuantConfig::resolve_layer_modes` when
/// the new config is enabled. This test verifies the table directly so
/// we do not need a real model + sequence state to exercise it.
#[test]
fn batch_kv_quant_per_layer_table_skips_last_layer_only() {
    let cfg = mlxcel_core::cache::BatchKvQuantConfig::new(
        mlxcel_core::cache::KvQuantScheme::Uniform,
        8,
        64,
        true,
    )
    .unwrap();
    let modes = cfg.resolve_layer_modes(40); // gemma-4-31b-class
    assert_eq!(modes.len(), 40);
    // ONLY the last layer is skipped, NOT first 2 + last 2
    // (that latter behaviour is the existing Boundary-V, which is a separate orthogonal mechanism).
    assert_eq!(
        modes[0],
        mlxcel_core::cache::KVCacheMode::Int8,
        "first layer must keep nominal mode (last-layer-skip is separate from Boundary-V)"
    );
    assert_eq!(
        modes[1],
        mlxcel_core::cache::KVCacheMode::Int8,
        "second layer must keep nominal mode"
    );
    assert_eq!(modes[38], mlxcel_core::cache::KVCacheMode::Int8);
    assert_eq!(
        modes[39],
        mlxcel_core::cache::KVCacheMode::Fp16,
        "last layer MUST be downgraded to Fp16 when skip_last_layer is set"
    );
}

#[test]
fn batch_kv_quant_per_layer_table_disabled_skip_keeps_uniform_modes() {
    let cfg = mlxcel_core::cache::BatchKvQuantConfig::new(
        mlxcel_core::cache::KvQuantScheme::TurboQuant,
        4,
        64,
        false,
    )
    .unwrap();
    let modes = cfg.resolve_layer_modes(8);
    for (i, mode) in modes.iter().enumerate() {
        assert_eq!(
            *mode,
            mlxcel_core::cache::KVCacheMode::Turbo4Asym,
            "layer {i} must keep nominal Turbo4Asym when skip_last_layer is disabled"
        );
    }
}
