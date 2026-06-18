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

//! Real-model parity test for #332 batched-prefill cohort splitting.
//!
//! When a collected prefill window mixes cold text rows with an incompatible
//! request (adopted prompt-cache prefix, VLM embeddings), the scheduler now
//! splits the window into cohorts and runs the cold rows batched instead of
//! falling the whole window back to sequential prefill. This test pins the
//! actual correctness property of that split: removing the incompatible row
//! from the window must not perturb the cold rows' batched prefill. Concretely,
//! a cold row prefilled inside a cohort-split window (`[cold A, cold B, adopted
//! C]`, which the planner splits into a batched `{A, B}` cohort plus a
//! sequential `{C}` cohort) must decode byte-for-byte identically to the same
//! row in an all-cold batched window of the same composition (`[cold A,
//! cold B]`, where no split happens). The split only lifts C onto the
//! offset-aware single-sequence path; it leaves `{A, B}`'s padded batched
//! forward untouched, so the two must agree exactly.
//!
//! It deliberately does NOT compare a batched (B > 1) row against a
//! single-prefill (B = 1) reference. Padded batched prefill is not
//! bitwise-identical to single-sequence prefill on Metal: the batched pass pads
//! short rows up to the cohort's longest prompt and runs a wider matmul, so f16
//! rounding differs and an early near-tie greedy token can flip and cascade.
//! That is the documented #203 / #325 / #326 jitter class, not a correctness
//! bug, so asserting byte-identity of a batched row against a single-prefill
//! row would be the wrong invariant. The single-prefill values are still
//! computed and printed as a diagnostic (so the jitter is visible under
//! `--nocapture`), but they are never asserted.
//!
//! Behavior note: a cold row that previously fell back to *sequential* prefill
//! (because its window held an incompatible sibling) now runs batched. Its
//! greedy decode can therefore differ from the old sequential output by the
//! same jitter class. That is the intended effect of #332, not a regression.
//!
//! The test loads a real qwen3 checkpoint and runs GPU forwards, so it is
//! `#[ignore]` and soft-skips when the model directory is absent.
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

/// Enqueue a window of `(prompt, prefill_start_offset)` rows, run a single
/// [`BatchScheduler::execute_batched_prefill`] (which classifies the rows,
/// plans cohorts, and dispatches them), then decode each row in window order
/// and return the per-row decoded text.
///
/// Each row is decoded on its own with `execute_decode_step(&[id])`, so the
/// only batched stage is the prefill. That isolates cohort-split prefill
/// behavior as the single variable under test: a row's decode trajectory is
/// driven purely by its own post-prefill KV cache, never by a sibling. All
/// sequences are released afterward so the pool is clean for the next window.
fn run_window(
    sched: &mut BatchScheduler,
    tokenizer: &MlxcelTokenizer,
    window: &[(&[i32], usize)],
) -> Vec<String> {
    let mut ids = Vec::with_capacity(window.len());
    let mut rxs = Vec::with_capacity(window.len());
    for &(prompt, offset) in window {
        let id = sched
            .allocate_sequence_state()
            .expect("allocate window sequence");
        let (seq, rx) = make_seq(id, tokenizer, prompt.to_vec(), offset);
        sched
            .prefill_queue
            .enqueue(seq)
            .expect("enqueue window sequence");
        ids.push(id);
        rxs.push(rx);
    }
    // One call drains the whole window, splitting it into cohorts.
    sched.execute_batched_prefill();
    assert!(
        sched.prefill_queue.is_empty(),
        "the whole window must be drained by the cohort dispatch"
    );
    let mut outputs = Vec::with_capacity(window.len());
    for (idx, &id) in ids.iter().enumerate() {
        outputs.push(run_and_collect(sched, id, &rxs[idx]));
    }
    for &id in &ids {
        cleanup(sched, id);
    }
    outputs
}

#[test]
#[ignore = "loads qwen3-0.6b-4bit and runs real GPU forwards; run with --ignored"]
fn mixed_window_cold_cohort_matches_all_cold_batched_qwen3() {
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

    // ---- DIAGNOSTIC reference: each cold prompt prefilled alone (B = 1). ----
    // NOT a load-bearing comparison. Padded batched prefill (B > 1) is not
    // bitwise-identical to single-sequence prefill (B = 1) on Metal, so a
    // batched row's greedy decode can flip an early near-tie token versus the
    // single-prefill output (the #203 / #325 / #326 jitter class). These values
    // are printed only so the jitter is visible under --nocapture.
    let ref_a = reference_text(&mut sched, &seq_tokenizer, PROMPT_A);
    let ref_b = reference_text(&mut sched, &seq_tokenizer, PROMPT_B);
    assert!(!ref_a.is_empty(), "reference A produced no output");
    assert!(!ref_b.is_empty(), "reference B produced no output");

    // ---- CONTROL: all-cold [A, B] batched window (no incompatible row). ----
    // Both rows are cold, so the planner forms a single BatchedCold {A, B}
    // cohort and performs NO split. This is the correct reference for the
    // cohort path: it pins what {A, B}'s padded batched prefill produces when
    // there is no C in the window to split off.
    let all_cold = run_window(&mut sched, &seq_tokenizer, &[(PROMPT_A, 0), (PROMPT_B, 0)]);
    let allcold_a = &all_cold[0];
    let allcold_b = &all_cold[1];
    assert!(
        !allcold_b.is_empty(),
        "all-cold batched B produced no output"
    );

    // ---- SUBJECT: mixed [cold A, cold B, adopted C] window (the #332 split). ----
    // C carries a non-zero prefill_start_offset (an adopted prompt-cache
    // prefix), so it is incompatible with the padded batched path. The planner
    // forms BatchedCold {A, B} + Sequential {C}: the split lifts C onto the
    // offset-aware single-sequence path but must NOT change {A, B}'s batched
    // forward at all (C is prefilled in a separate cohort after {A, B} and
    // never shares their forward pass or caches).
    let mixed = run_window(
        &mut sched,
        &seq_tokenizer,
        &[(PROMPT_A, 0), (PROMPT_B, 0), (PROMPT_C, 1)],
    );
    let mixed_a = &mixed[0];
    let mixed_b = &mixed[1];
    let mixed_c = &mixed[2];

    eprintln!("--- #332 cohort parity (qwen3-0.6b-4bit) ---");
    eprintln!("single  B (B=1, diagnostic):  {ref_b:?}");
    eprintln!("allcold B (B=2 batched):      {allcold_b:?}");
    eprintln!("mixed   B (B=2 cohort split): {mixed_b:?}");
    eprintln!("single  A (B=1, diagnostic):  {ref_a:?}");
    eprintln!("allcold A (B=2 batched):      {allcold_a:?}");
    eprintln!("mixed   A (B=2 cohort split): {mixed_a:?}");
    if mixed_b != &ref_b {
        eprintln!(
            "note: batched B (B=2) differs from single B (B=1) -> #203 jitter class \
             (expected; the cohort property is batched-vs-batched, asserted below)."
        );
    }

    // ---- LOAD-BEARING #332 invariant: cohort split == all-cold batched. ----
    // Removing the incompatible row C from the window must not perturb the cold
    // rows' batched prefill, so a cohort-split cold row must decode byte-for-
    // byte identically to the same row in an all-cold batched window of the
    // same composition. This is the real correctness property of the split, and
    // unlike a batched-vs-single comparison it is not subject to the #203
    // jitter class (both sides run the identical B = 2 padded forward).
    assert_eq!(
        mixed_a, allcold_a,
        "cold row A in a cohort-split window must decode identically to the all-cold batched window"
    );
    assert_eq!(
        mixed_b, allcold_b,
        "cold row B in a cohort-split window must decode identically to the all-cold batched window"
    );
    assert!(
        !mixed_c.is_empty(),
        "the adopted-prefix cohort row must still be prefilled and decode output (split handled every cohort)"
    );
}
