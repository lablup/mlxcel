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

//! Real-model parity tests for #332 batched-prefill cohort splitting.
//!
//! When a collected prefill window mixes cold text rows with an incompatible
//! request (adopted prompt-cache prefix, VLM embeddings), the scheduler now
//! splits the window into cohorts and runs the cold rows batched instead of
//! falling the whole window back to sequential prefill. These tests confirm
//! that the split does not change any request's decoded output versus the
//! pre-cohort all-or-nothing sequential path: a cold row prefilled inside a
//! batched cohort (alongside an incompatible sibling) must decode identically
//! to the same row prefilled alone on the single-sequence path.
//!
//! The tests load a real qwen3 checkpoint and run GPU forwards, so they are
//! `#[ignore]` and soft-skip when the model directory is absent.
//!
//! Run with:
//! ```text
//! cargo test -p mlxcel --lib --release --features metal,accelerate \
//!     scheduler_cohort_parity -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Fetch the model with:
//! `./target/release/mlxcel download mlx-community/Qwen3-0.6B-4bit`.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::time::Instant;

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::SamplingConfig;

use super::BatchScheduler;
use crate::server::batch::BatchObservability;
use crate::server::batch::sequence::{RequestPriority, SequenceInfo, SequenceState};
use crate::server::config::{DecodeStorageBackend, PreemptionPolicy};
use crate::server::model_provider::GenerateEvent;
use crate::server::model_provider::model_worker::StreamingDecodeState;
use crate::server::state::BatchMetrics;
use crate::tokenizer::MlxcelTokenizer;

/// qwen3 checkpoint directory name (a dense family that opts into batched
/// prefill and padded prefill, the scope of #332).
const QWEN3_DIR: &str = "qwen3-0.6b-4bit";

/// Greedy decode steps to compare per request. Short so the test is quick; the
/// prefill first token plus this many decode tokens already exercises the
/// batched-cohort forward and the per-row KV trim.
const DECODE_STEPS: usize = 12;

/// A high per-request budget so a request does not stop on the length limit
/// before `DECODE_STEPS`.
const MAX_TOKENS: usize = 64;

/// A fixed prompt that decodes several tokens without an immediate EOS (the
/// "what is 2 + 2?" prompt also used by the handoff parity tests).
const PROMPT_A: &[i32] = &[
    9707, 11, 358, 1079, 264, 4128, 1614, 13, 5209, 3291, 752, 911, 697, 7990, 13, 358, 2776, 264,
    10950, 17847, 13, 6771, 594, 1438, 419, 1495, 3019, 553, 3019, 11, 323, 1473, 697, 975, 13,
    5209, 387, 2797, 624, 14374, 14582, 25, 3555, 374, 220, 17, 488, 220, 17, 30,
];

/// A second, shorter prompt of a different length so the batched cohort pads
/// the two rows to a common length and trims each back independently.
const PROMPT_B: &[i32] = &[
    9707, 11, 4332, 752, 264, 2805, 22692, 911, 279, 9396, 13, 5209, 387, 63594, 624,
];

/// A third prompt routed to the sequential cohort via a non-zero adopted-prefix
/// offset, so the window is a genuine cold + incompatible mix.
const PROMPT_C: &[i32] = &[
    785, 6722, 315, 9625, 374, 264, 3283, 13, 22512, 752, 911, 432, 13,
];

/// `<CARGO_MANIFEST_DIR>/models/<name>` (the mlxcel crate root is the repo root).
fn repo_model_dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join(name)
}

/// Build a scheduler that can batch a prefill window of up to four cold rows in
/// one pass (`max_batch_prefill = 4`) and hold the whole window in the active
/// batch while decoding (`max_batch_size = 4`).
fn build_cohort_scheduler(
    model: crate::LoadedModel,
    tokenizer: MlxcelTokenizer,
    config_eos: Vec<i32>,
) -> BatchScheduler {
    let (_req_tx, req_rx) = mpsc::channel();
    BatchScheduler::with_config(
        model,
        tokenizer,
        config_eos,
        req_rx,
        4,  // max_batch_size
        64, // max_queue_depth
        Arc::new(BatchMetrics::new()),
        Arc::new(BatchObservability::new()),
        512, // prefill_chunk_size > prompts so prefills run whole
        false,
        PreemptionPolicy::default(),
        4, // max_batch_prefill
        DecodeStorageBackend::Paged,
    )
}

/// Build a greedy request bound to `seq_id` with the given prompt and optional
/// adopted-prefix offset. The response receiver is returned and must be kept
/// alive so streamed tokens can be collected.
fn make_seq(
    seq_id: SequenceId,
    tokenizer: &MlxcelTokenizer,
    prompt_tokens: Vec<i32>,
    prefill_start_offset: usize,
) -> (SequenceInfo, mpsc::Receiver<GenerateEvent>) {
    let (tx, rx) = mpsc::channel();
    let decode_state = StreamingDecodeState::new(tokenizer, &prompt_tokens);
    let seq = SequenceInfo {
        seq_id,
        state: SequenceState::Queued,
        prompt_tokens,
        sampling: SamplingConfig::greedy(),
        max_tokens: MAX_TOKENS,
        eos_token_ids: Vec::new(),
        priority: RequestPriority::Normal,
        logprobs_config: Default::default(),
        vlm_embeddings: None,
        images: Vec::new(),
        audio: Vec::new(),
        generated_tokens: Vec::new(),
        generated_text: String::new(),
        decode_state,
        prefill_offset: 0,
        prefill_start_offset,
        already_cached_tokens: prefill_start_offset,
        response_tx: tx,
        cancelled: Arc::new(AtomicBool::new(false)),
        created_at: Instant::now(),
        prefill_start: None,
        first_token_time: None,
        token_history: Vec::new(),
        sampler_state: None,
        merged_eos: Vec::new(),
        thinking: crate::server::thinking_budget::ThinkingState::disabled(),
        structured: None,
    };
    (seq, rx)
}

/// Decode `seq_id` for up to `DECODE_STEPS` steps (or until it finishes), then
/// drain its response channel into the concatenated output text. Collecting
/// from the channel is robust to an early stop: a finished sequence is removed
/// from the active batch, but the text it streamed is still in the channel.
fn run_and_collect(
    sched: &mut BatchScheduler,
    seq_id: SequenceId,
    rx: &mpsc::Receiver<GenerateEvent>,
) -> String {
    let mut steps = 0;
    while steps < DECODE_STEPS && sched.active_batch.get_mut(seq_id).is_some() {
        sched.execute_decode_step(&[seq_id]);
        steps += 1;
    }
    let mut text = String::new();
    while let Ok(ev) = rx.try_recv() {
        match ev {
            GenerateEvent::Token(t) | GenerateEvent::TokenWithLogprobs(t, _) => text.push_str(&t),
            _ => {}
        }
    }
    text
}

/// Release a sequence's pool state if it is still active (a finished sequence
/// has already released its caches).
fn cleanup(sched: &mut BatchScheduler, seq_id: SequenceId) {
    if sched.active_batch.get_mut(seq_id).is_some() {
        sched.active_batch.remove(seq_id);
        sched.release_sequence_caches(seq_id);
    }
}

/// Prefill one cold request alone on the single-sequence path (the pre-cohort
/// behavior) and return its decoded output text. Releases the sequence so the
/// pool is clean afterward.
fn reference_text(
    sched: &mut BatchScheduler,
    tokenizer: &MlxcelTokenizer,
    prompt: &[i32],
) -> String {
    let id = sched
        .allocate_sequence_state()
        .expect("allocate reference sequence");
    let (mut seq, rx) = make_seq(id, tokenizer, prompt.to_vec(), 0);
    BatchScheduler::begin_prefill(&mut seq).expect("begin reference prefill");
    sched.execute_full_prefill(seq);
    let text = run_and_collect(sched, id, &rx);
    cleanup(sched, id);
    text
}

#[test]
#[ignore = "loads qwen3-0.6b-4bit and runs real GPU forwards; run with --ignored"]
fn mixed_window_cold_cohort_matches_single_prefill_qwen3() {
    let _runtime = crate::initialize_runtime();
    let dir = repo_model_dir(QWEN3_DIR);
    if !dir.exists() {
        eprintln!(
            "Skipping {QWEN3_DIR}: model directory not found at {}.\n\
             Fetch with: ./target/release/mlxcel download mlx-community/Qwen3-0.6B-4bit",
            dir.display()
        );
        return;
    }
    let (model, sched_tokenizer) =
        crate::load_model(&dir).unwrap_or_else(|e| panic!("load {QWEN3_DIR}: {e:?}"));
    let seq_tokenizer = crate::tokenizer::load_tokenizer(&dir).expect("load tokenizer");
    let config_eos = crate::read_eos_token_ids(&dir);
    let mut sched = build_cohort_scheduler(model, sched_tokenizer, config_eos);

    // ---- REFERENCES: each cold prompt prefilled alone (single-sequence). ----
    let ref_a = reference_text(&mut sched, &seq_tokenizer, PROMPT_A);
    let ref_b = reference_text(&mut sched, &seq_tokenizer, PROMPT_B);
    assert!(!ref_a.is_empty(), "reference A produced no output");
    assert!(!ref_b.is_empty(), "reference B produced no output");

    // ---- MIXED WINDOW: [cold A, cold B, adopted C]. ----
    // The planner forms a batched cohort {A, B} (two cold rows of different
    // lengths) and a sequential cohort {C}. C carries a non-zero
    // prefill_start_offset, so it is incompatible with the padded batched path
    // and is routed to the offset-aware single-sequence path in both the new
    // (cohort) and old (all-sequential) behavior. Its presence is what forces a
    // genuine cohort split rather than an all-cold batch.
    let id_a = sched.allocate_sequence_state().expect("alloc A");
    let id_b = sched.allocate_sequence_state().expect("alloc B");
    let id_c = sched.allocate_sequence_state().expect("alloc C");
    let (seq_a, rx_a) = make_seq(id_a, &seq_tokenizer, PROMPT_A.to_vec(), 0);
    let (seq_b, rx_b) = make_seq(id_b, &seq_tokenizer, PROMPT_B.to_vec(), 0);
    let (seq_c, rx_c) = make_seq(id_c, &seq_tokenizer, PROMPT_C.to_vec(), 1);

    sched.prefill_queue.enqueue(seq_a).expect("enqueue A");
    sched.prefill_queue.enqueue(seq_b).expect("enqueue B");
    sched.prefill_queue.enqueue(seq_c).expect("enqueue C");
    assert_eq!(
        sched.prefill_queue.len(),
        3,
        "window should hold three rows"
    );

    // One call splits the window: {A, B} batched, {C} sequential.
    sched.execute_batched_prefill();
    assert!(
        sched.prefill_queue.is_empty(),
        "the whole window must be drained by the cohort dispatch"
    );

    let got_a = run_and_collect(&mut sched, id_a, &rx_a);
    let got_b = run_and_collect(&mut sched, id_b, &rx_b);
    let got_c = run_and_collect(&mut sched, id_c, &rx_c);

    assert_eq!(
        got_a, ref_a,
        "cold row A batched in a mixed window must decode identically to the single-prefill reference"
    );
    assert_eq!(
        got_b, ref_b,
        "cold row B batched in a mixed window must decode identically to the single-prefill reference"
    );
    assert!(
        !got_c.is_empty(),
        "the adopted-prefix cohort row must still be prefilled and decode output (split handled every cohort)"
    );

    cleanup(&mut sched, id_a);
    cleanup(&mut sched, id_b);
    cleanup(&mut sched, id_c);
}
