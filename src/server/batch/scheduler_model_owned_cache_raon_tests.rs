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

//! Prompt-cache reuse for Raon-1 mini through the scheduler (issue #3458), on
//! the tiny random model with one steered layer: a repeated chat turn
//! restores the model-owned snapshot taken at its history boundary and
//! reproduces the cold run bit for bit, and the same prompt at another
//! reasoning effort looks up another bucket.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use super::scheduler_model_owned_cache_tests::{cache_ctx, options, test_store};
use super::*;
use crate::LoadedModel;
use crate::server::config::{DecodeStorageBackend, PreemptionPolicy, PromptCacheRequestContext};
use crate::server::model_provider::GenerateEvent;
use crate::server::prompt_cache::PromptCacheStore;
use crate::server::prompt_cache::key::fold_model_reasoning_effort;
use crate::server::state::BatchMetrics;
use crate::tokenizer::MlxcelTokenizer;
use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::LanguageModel;

const EOS: i32 = 31;
const MAX_TOKENS: usize = 6;
/// The history part of the turn; the prompt adds a three-token generation
/// prompt after it.
const BOUNDARY: usize = 12;

fn scheduler(store: Arc<PromptCacheStore>) -> BatchScheduler {
    let (_tx, rx) = mpsc::channel();
    let sched = BatchScheduler::with_config(
        LoadedModel::Raon1Mini(crate::models::raon1_mini::steering_tests::steered_model()),
        MlxcelTokenizer::stub(),
        vec![EOS],
        rx,
        4,
        8,
        Arc::new(BatchMetrics::new()),
        Arc::new(BatchObservability::new()),
        0,
        false,
        PreemptionPolicy::default(),
        1,
        DecodeStorageBackend::Paged,
    )
    .with_prompt_cache(Some(store));
    install_thread_local_default_stream(sched.generation_stream.as_ref());
    sched
}

fn prompt() -> Vec<i32> {
    (0..BOUNDARY + 3).map(|i| (i * 7 + 3) as i32 % 30).collect()
}

/// The cache context of a request at `effort`: the level on the context, and
/// folded into the cache key's template signature.
fn effort_ctx(effort: &str) -> PromptCacheRequestContext {
    let mut ctx = cache_ctx();
    ctx.template_sig = fold_model_reasoning_effort(ctx.template_sig, Some(effort));
    ctx.model_reasoning_effort = Some(effort.to_string());
    ctx
}

/// A chat turn of `prompt()` at `effort`, as the server builds it: the level
/// on the request, and folded into the cache key's template signature.
fn enqueue(sched: &mut BatchScheduler, effort: &str) -> mpsc::Receiver<GenerateEvent> {
    let tokens = prompt();
    let mut ctx = effort_ctx(effort);
    ctx.history_prefix_tokens = Some(tokens[..BOUNDARY].to_vec());
    let mut options = options();
    options.prompt_cache_ctx = Some(ctx);
    options.max_tokens = MAX_TOKENS;
    options.ignore_eos = true;
    options.model_reasoning_effort = Some(effort.to_string());
    let (tx, rx) = mpsc::channel();
    sched.enqueue_request(
        "prompt".to_string(),
        Some(tokens),
        options,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        tx,
        Arc::new(AtomicBool::new(false)),
        true,
    );
    rx
}

/// Run the queued request to completion and return its greedy token ids.
fn run(sched: &mut BatchScheduler, rx: &mpsc::Receiver<GenerateEvent>) -> Vec<i32> {
    for _ in 0..64 {
        match sched.decide_action() {
            BatchSchedulerAction::Prefill(id) => {
                sched.discard_lookahead();
                sched.execute_prefill(id);
            }
            BatchSchedulerAction::Decode(ids) => sched.execute_decode_step(&ids),
            BatchSchedulerAction::Idle => break,
            other => panic!("unexpected scheduler action {other:?}"),
        }
        sched.finalize_completed();
        if sched.active_batch.is_empty() && sched.prefill_queue.is_empty() {
            break;
        }
    }
    loop {
        match rx.recv_timeout(Duration::from_secs(5)) {
            Ok(GenerateEvent::Done(result)) => return result.generated_token_ids,
            Ok(GenerateEvent::Error(err)) => panic!("unexpected generation error: {err}"),
            Ok(_) => {}
            Err(err) => panic!("generation did not finish: {err}"),
        }
    }
}

fn logit_bytes<M: LanguageModel + ?Sized>(model: &M, id: SequenceId, tokens: &[i32]) -> Vec<u8> {
    let input = mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32]);
    let out = model.forward_with_sequence_id(&input, Some(id), &mut model.make_caches(), None);
    mlxcel_core::eval(&out);
    mlxcel_core::array_to_raw_bytes(&out)
}

/// The logits of the generation prompt after a cold prefill of the history
/// prefix at `effort`, on a fresh model: the partition the scheduler's cold
/// run forwards (history, then the rest).
fn cold_suffix_logits(effort: &str) -> Vec<u8> {
    let model = crate::models::raon1_mini::steering_tests::steered_model();
    let id = SequenceId::from_raw(458_000);
    model.prepare_sequence_state(id);
    model
        .set_sequence_reasoning_effort(Some(id), Some(effort))
        .unwrap();
    let tokens = prompt();
    let _ = logit_bytes(&model, id, &tokens[..BOUNDARY]);
    logit_bytes(&model, id, &tokens[BOUNDARY..])
}

/// The same logits from the state the scheduler restored into the queued
/// request `queued`, copied to a probe sequence so the request itself is
/// left untouched.
fn restored_suffix_logits(sched: &BatchScheduler, queued: SequenceId, effort: &str) -> Vec<u8> {
    let model = sched.engine.model();
    let state = model
        .snapshot_sequence_state(queued, BOUNDARY)
        .expect("the adopted sequence holds the restored prefix");
    let probe = SequenceId::from_raw(458_001);
    model.prepare_sequence_state(probe);
    model
        .set_sequence_reasoning_effort(Some(probe), Some(effort))
        .unwrap();
    model.restore_sequence_state(probe, &state).unwrap();
    let logits = logit_bytes(model, probe, &prompt()[BOUNDARY..]);
    model.release_sequence_state_by_id(probe);
    logits
}

fn queued(sched: &BatchScheduler) -> (SequenceId, usize) {
    let seq = sched.prefill_queue.peek().expect("the request is queued");
    (seq.seq_id, seq.prefill_start_offset)
}

#[test]
fn a_repeated_raon_turn_restores_its_snapshot_and_reproduces_the_cold_run() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let store = test_store();
    let mut sched = scheduler(store.clone());

    let rx = enqueue(&mut sched, "low");
    assert_eq!(queued(&sched).1, 0, "a cold store restores nothing");
    let cold = run(&mut sched, &rx);
    assert_eq!(cold.len(), MAX_TOKENS);
    let stored = store.stats();
    assert!(
        stored.snapshot_entries >= 1,
        "the turn donates its history-boundary snapshot"
    );
    assert_eq!(
        stored.inserts, 0,
        "no detached K/V set for a model-owned family"
    );
    assert_eq!(stored.snapshot_hits, 0);

    // The identical request: a hit on the whole stored boundary prefix.
    let rx = enqueue(&mut sched, "low");
    assert_eq!(store.stats().snapshot_hits, 1, "prompt-cache hit");
    let (seq_id, start) = queued(&sched);
    assert_eq!(
        start, BOUNDARY,
        "the restore covers the history prefix; only the generation prompt is prefilled"
    );
    assert_eq!(
        restored_suffix_logits(&sched, seq_id, "low"),
        cold_suffix_logits("low"),
        "the restored state gives the cold run's logits bit for bit"
    );
    assert_eq!(run(&mut sched, &rx), cold, "and the same greedy tokens");

    // The same prompt at another effort is another bucket: no hit and a
    // full prefill, at logits the low-effort state does not produce.
    let rx = enqueue(&mut sched, "max");
    assert_eq!(store.stats().snapshot_hits, 1, "another effort must miss");
    assert_eq!(queued(&sched).1, 0);
    assert_ne!(
        cold_suffix_logits("max"),
        cold_suffix_logits("low"),
        "max (alpha 0) and low (alpha 0.3) steer differently"
    );
    let _ = run(&mut sched, &rx);

    // Back at the first effort the first snapshot still restores.
    let rx = enqueue(&mut sched, "low");
    assert_eq!(store.stats().snapshot_hits, 2);
    assert_eq!(queued(&sched).1, BOUNDARY);
    assert_eq!(run(&mut sched, &rx), cold);
}

/// The identical turn again after its background warm-up ran (issue #3467):
/// the warm-up stores the next turn's history prefix and must leave the
/// boundary snapshot it restored from, so the retry restores that one.
#[test]
fn a_repeated_raon_turn_restores_its_boundary_after_the_warmup() {
    let _guard = crate::models::embedding_test_support::mlx_test_guard();
    let store = test_store();
    let mut sched = scheduler(store.clone());

    let rx = enqueue(&mut sched, "low");
    assert_eq!(queued(&sched).1, 0, "a cold store restores nothing");
    let cold = run(&mut sched, &rx);
    assert_eq!(cold.len(), MAX_TOKENS);

    // The next turn's history prefix: the history, then the reply.
    let mut target = prompt()[..BOUNDARY].to_vec();
    target.extend_from_slice(&cold);
    assert!(!prompt().starts_with(&target));
    let warmups = sched
        .batch_observability
        .snapshot()
        .prompt_cache_warmups_run;
    sched.enqueue_prompt_cache_warmup(target, effort_ctx("low"));
    sched.run_next_prompt_cache_warmup();
    assert_eq!(
        sched
            .batch_observability
            .snapshot()
            .prompt_cache_warmups_run,
        warmups + 1,
        "the warm-up restored the boundary snapshot and stored its target"
    );
    assert_eq!(
        store.stats().snapshot_hits,
        0,
        "a warm-up lookup is not a request hit"
    );

    let rx = enqueue(&mut sched, "low");
    let (seq_id, start) = queued(&sched);
    assert_eq!(
        (store.stats().snapshot_hits, start),
        (1, BOUNDARY),
        "the retried turn restores its history-boundary snapshot"
    );
    assert_eq!(
        restored_suffix_logits(&sched, seq_id, "low"),
        cold_suffix_logits("low"),
        "the restored state gives the cold run's logits bit for bit"
    );
    assert_eq!(run(&mut sched, &rx), cold, "and the same greedy tokens");
}
