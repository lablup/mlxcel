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

//! Prompt lookup in the batch scheduler (#2255), on a real dense Llama whose
//! greedy continuation of token `t` is `(t + 1) % VOCAB`: every attention and
//! MLP projection is zero, so the residual stream is the token's one-hot
//! embedding, and the LM head maps it to the next id. A prompt that walks the
//! cycle is therefore a prompt the reply copies, so prompt lookup proposes
//! and the target accepts, while a prompt with no repeated n-gram proposes
//! nothing for the first `VOCAB` tokens. Each test pins one invariant of
//! `scheduler::prompt_lookup`; after every tick that leaves no lookahead
//! stored, every row's KV holds its prompt and every emitted token but the
//! last (I1).

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, mpsc};

use mlxcel_core::generate::SamplingConfig;
use mlxcel_core::speculative::prompt_lookup::PromptLookupConfig;
use mlxcel_core::weights::WeightMap;

use super::*;
use crate::server::config::{DecodeStorageBackend, PreemptionPolicy, ReasoningBudgetOverride};
use crate::server::model_provider::GenerateEvent;
use crate::server::state::BatchMetrics;
use crate::tokenizer::MlxcelTokenizer;

const VOCAB: usize = 16;
const HIDDEN: usize = 16;

/// `[0, 1, .., 15, 0, 1, 2, 3]`: the reply continues the cycle, which the
/// prompt already contains, so lookup proposes from the second token on.
fn copy_prompt() -> Vec<i32> {
    (0..20).map(|i| (i % VOCAB) as i32).collect()
}

/// No repeated bigram, and the reply (`9, 10, ..`) repeats none of it
/// before it wraps.
const PLAIN_PROMPT: &[i32] = &[0, 2, 4, 6, 8];

fn cycle_llama() -> crate::models::Llama3Model {
    walk_llama(|t| (t + 1) % VOCAB)
}

/// A dense Llama whose greedy continuation of token `t` is `next(t)`.
fn walk_llama(next: impl Fn(usize) -> usize) -> crate::models::Llama3Model {
    let args: crate::models::llama3::ModelArgs = serde_json::from_value(serde_json::json!({
        "model_type": "llama",
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "intermediate_size": 32,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "rms_norm_eps": 1e-6,
        "vocab_size": VOCAB,
        "rope_theta": 10000.0,
        "attention_bias": false,
        "tie_word_embeddings": false,
    }))
    .expect("cycle llama args");
    let tensor = |values: Vec<f32>, shape: &[i32]| mlxcel_core::from_slice_f32(&values, shape);
    let zeros = |shape: &[i32]| {
        let n: i32 = shape.iter().product();
        tensor(vec![0.0; n as usize], shape)
    };
    let ones = |n: usize| tensor(vec![1.0; n], &[n as i32]);
    let mut w = WeightMap::new();
    let mut embed = vec![0.0f32; VOCAB * HIDDEN];
    let mut head = vec![0.0f32; VOCAB * HIDDEN];
    for t in 0..VOCAB {
        embed[t * HIDDEN + t] = 1.0;
        // Row `next(t)` of the head reads the one-hot of `t`.
        head[next(t) * HIDDEN + t] = 1.0;
    }
    w.insert(
        "model.embed_tokens.weight".into(),
        tensor(embed, &[VOCAB as i32, HIDDEN as i32]),
    );
    w.insert(
        "lm_head.weight".into(),
        tensor(head, &[VOCAB as i32, HIDDEN as i32]),
    );
    w.insert("model.norm.weight".into(), ones(HIDDEN));
    let h = HIDDEN as i32;
    let p = "model.layers.0";
    for (name, shape) in [
        ("self_attn.q_proj.weight", vec![16, h]),
        ("self_attn.k_proj.weight", vec![8, h]),
        ("self_attn.v_proj.weight", vec![8, h]),
        ("self_attn.o_proj.weight", vec![h, 16]),
        ("mlp.gate_proj.weight", vec![32, h]),
        ("mlp.up_proj.weight", vec![32, h]),
        ("mlp.down_proj.weight", vec![h, 32]),
    ] {
        w.insert(format!("{p}.{name}"), zeros(&shape));
    }
    w.insert(format!("{p}.input_layernorm.weight"), ones(HIDDEN));
    w.insert(format!("{p}.post_attention_layernorm.weight"), ones(HIDDEN));
    crate::models::Llama3Model::from_weights(&w, &args).expect("cycle llama builds")
}

/// A scheduler over the cycle model; `max_batch` is
/// `--prompt-lookup-max-batch`, `None` runs without prompt lookup.
fn scheduler(max_batch: Option<usize>) -> BatchScheduler {
    scheduler_with_storage(max_batch, DecodeStorageBackend::Dense)
}

fn scheduler_with_storage(
    max_batch: Option<usize>,
    storage: DecodeStorageBackend,
) -> BatchScheduler {
    let (_tx, rx) = mpsc::channel();
    let mut sched = BatchScheduler::with_config(
        crate::LoadedModel::Llama(cycle_llama()),
        MlxcelTokenizer::stub(),
        Vec::new(),
        rx,
        4,
        8,
        Arc::new(BatchMetrics::new()),
        Arc::new(BatchObservability::new()),
        0,
        false,
        PreemptionPolicy::default(),
        1,
        storage,
    );
    if let Some(max_batch) = max_batch {
        sched = sched.with_speculative_dispatch(crate::server::SpeculativeDispatch::PromptLookup {
            config: PromptLookupConfig::default(),
            max_batch,
        });
    }
    install_thread_local_default_stream(sched.generation_stream.as_ref());
    sched
}

fn enqueue(
    sched: &mut BatchScheduler,
    prompt: &[i32],
    max_tokens: usize,
) -> mpsc::Receiver<GenerateEvent> {
    enqueue_with(sched, prompt, max_tokens, &[]).0
}

/// [`enqueue`] with extra stop token ids (the cycle model has no EOS of its
/// own, so a stop id stands in for one) and the request's cancel flag.
fn enqueue_with(
    sched: &mut BatchScheduler,
    prompt: &[i32],
    max_tokens: usize,
    stop_token_ids: &[i32],
) -> (mpsc::Receiver<GenerateEvent>, Arc<AtomicBool>) {
    enqueue_with_sampling(
        sched,
        prompt,
        max_tokens,
        SamplingConfig {
            stop_token_ids: stop_token_ids.to_vec(),
            ..SamplingConfig::greedy()
        },
    )
}

fn enqueue_with_sampling(
    sched: &mut BatchScheduler,
    prompt: &[i32],
    max_tokens: usize,
    sampling: SamplingConfig,
) -> (mpsc::Receiver<GenerateEvent>, Arc<AtomicBool>) {
    let options = ServerGenerateOptions {
        n_indent: 0,
        t_max_predict_ms: None,
        reasoning_budget_message: None,
        retention: Default::default(),
        dry_breaker_strings: None,
        logit_bias: Vec::new(),
        logit_bias_texts: Vec::new(),
        post_sampling_probs: false,
        max_tokens,
        sampling,
        stop_sequences: None,
        ignore_eos: false,
        priority: RequestPriority::Normal,
        lora_scales: None,
        logprobs: Default::default(),
        reasoning_budget: ReasoningBudgetOverride::InheritServerDefault,
        thinking_enter_block_on_start: false,
        reasoning_control: None,
        prompt_cache_ctx: None,
        structured: None,
        grammar: None,
        image_soft_tokens: None,
        pre_rendered_prompt_tokens: None,
        admission_notification: None,
        count_prompt_only: false,
    };
    let (tx, rx) = mpsc::channel();
    let cancelled = Arc::new(AtomicBool::new(false));
    sched.enqueue_request(
        "prompt".to_string(),
        Some(prompt.to_vec()),
        options,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        tx,
        Arc::clone(&cancelled),
        true,
    );
    (rx, cancelled)
}

/// What one run-loop iteration did.
#[derive(Debug, PartialEq, Eq)]
enum Tick {
    Prefill,
    Decode,
    Idle,
}

/// Finished rows' prompts and generated tokens, captured before
/// `finalize_completed` releases them.
type Done = Vec<(Vec<i32>, Vec<i32>)>;

/// One run-loop iteration, then the I1 check whenever no lookahead is
/// stored; finished rows land in `done`.
fn tick(sched: &mut BatchScheduler, done: &mut Done) -> Tick {
    let action = match sched.decide_action() {
        BatchSchedulerAction::Prefill(id) => {
            // The run loop tears the lookahead down before any admission.
            sched.discard_lookahead();
            sched.execute_prefill(id);
            Tick::Prefill
        }
        BatchSchedulerAction::Decode(ids) => {
            sched.execute_decode_step(&ids);
            Tick::Decode
        }
        BatchSchedulerAction::Idle => Tick::Idle,
        other => panic!("unexpected scheduler action {other:?}"),
    };
    if sched.decode_lookahead.is_none() {
        assert_synchronous_state(sched);
    }
    for seq in sched.active_batch.iter_sequences() {
        if seq.state.is_finished() {
            done.push((seq.prompt_tokens.clone(), seq.generated_tokens.clone()));
        }
    }
    sched.finalize_completed();
    action
}

/// The tokens the finished row with `prompt` generated.
fn tokens_of(done: &Done, prompt: &[i32]) -> Vec<i32> {
    done.iter()
        .find(|(p, _)| p == prompt)
        .map(|(_, t)| t.clone())
        .expect("the row finished")
}

/// I1: every decoding row's KV holds its prompt and every emitted token but
/// the last.
fn assert_synchronous_state(sched: &mut BatchScheduler) {
    let rows: Vec<(SequenceId, usize)> = sched
        .active_batch
        .iter_sequences()
        .filter(|seq| !seq.state.is_finished())
        .map(|seq| {
            (
                seq.seq_id,
                seq.prompt_tokens.len() + seq.generated_tokens.len() - 1,
            )
        })
        .collect();
    for (id, want) in rows {
        let caches = sched
            .engine
            .pool_mut()
            .get_caches_mut(id)
            .expect("open sequence");
        assert_eq!(
            caches[0].offset as usize, want,
            "sequence {id}: KV offset disagrees with its tokens"
        );
    }
}

/// Drain one request's stream and return its terminal result.
fn completion(rx: &mpsc::Receiver<GenerateEvent>) -> crate::server::GenerationResult {
    let mut completed = None;
    while let Ok(event) = rx.try_recv() {
        match event {
            GenerateEvent::Done(result) => {
                assert!(completed.is_none(), "request completed more than once");
                completed = Some(result);
            }
            GenerateEvent::Error(err) => panic!("request failed: {err}"),
            _ => {}
        }
    }
    completed.expect("request completed")
}

/// The Done result's prompt-lookup acceptance counters `(proposed,
/// accepted)`, `None` when it reports no drafter.
fn acceptance(rx: &mpsc::Receiver<GenerateEvent>) -> Option<(usize, usize)> {
    acceptance_of(rx, mlxcel_core::drafter::DrafterKind::PromptLookup)
}

/// [`acceptance`] for a row of drafter `kind`.
fn acceptance_of(
    rx: &mpsc::Receiver<GenerateEvent>,
    kind: mlxcel_core::drafter::DrafterKind,
) -> Option<(usize, usize)> {
    let mut accepted = None;
    while let Ok(event) = rx.try_recv() {
        match event {
            GenerateEvent::Done(result) => {
                accepted = result.speculative.map(|s| {
                    assert_eq!(s.draft_kind, kind);
                    (s.draft_n, s.draft_n_accepted)
                });
            }
            GenerateEvent::Error(err) => panic!("request failed: {err}"),
            _ => {}
        }
    }
    accepted
}

fn drain(sched: &mut BatchScheduler, done: &mut Done) {
    let mut guard = 0;
    while tick(sched, done) != Tick::Idle {
        guard += 1;
        assert!(guard < 1000, "the scheduler did not go idle");
    }
}

/// `n` tokens of the cycle starting at `first`.
fn cycle_from(first: usize, n: usize) -> Vec<i32> {
    (first..first + n).map(|t| (t % VOCAB) as i32).collect()
}

fn len_of(sched: &BatchScheduler, id: SequenceId) -> Option<usize> {
    sched
        .active_batch
        .get(id)
        .map(|seq| seq.generated_tokens.len())
}

fn only_row(sched: &BatchScheduler) -> SequenceId {
    let ids: Vec<SequenceId> = sched
        .active_batch
        .iter_sequences()
        .map(|s| s.seq_id)
        .collect();
    assert_eq!(ids.len(), 1, "one decoding row");
    ids[0]
}

/// I1 and I4: a prompt-lookup row emits exactly the plain decode's tokens,
/// its verify rounds accept proposals, and the Done result reports them as
/// `prompt-lookup` acceptance counters.
#[test]
fn a_prompt_lookup_row_emits_the_plain_tokens_and_counts_acceptance() {
    let prompt = copy_prompt();
    let mut done = Done::new();
    let mut plain = scheduler(None);
    let plain_rx = enqueue(&mut plain, &prompt, 40);
    drain(&mut plain, &mut done);
    let plain_tokens = tokens_of(&done, &prompt);
    assert_eq!(plain_tokens, cycle_from(4, 40));
    assert!(
        acceptance(&plain_rx).is_none(),
        "plain decode reports no drafter"
    );

    let mut done = Done::new();
    let mut lookup = scheduler(Some(4));
    let lookup_rx = enqueue(&mut lookup, &prompt, 40);
    drain(&mut lookup, &mut done);
    assert_eq!(tokens_of(&done, &prompt), plain_tokens);
    let (proposed, accepted) = acceptance(&lookup_rx).expect("prompt lookup reports its rounds");
    assert!(
        accepted > 0 && accepted <= proposed,
        "{accepted}/{proposed}"
    );
}

/// No burst: while the prompt-lookup row verifies, a concurrent plain row
/// (admitted mid-decode, I6) gets exactly one token every decode tick, and
/// both streams equal their solo runs.
#[test]
fn a_concurrent_plain_row_gets_a_token_every_tick() {
    let prompt = copy_prompt();
    let mut done = Done::new();
    let mut sched = scheduler(Some(4));
    let _lookup_rx = enqueue(&mut sched, &prompt, 48);
    assert_eq!(tick(&mut sched, &mut done), Tick::Prefill);
    let lookup_id = only_row(&sched);
    for _ in 0..3 {
        assert_eq!(tick(&mut sched, &mut done), Tick::Decode);
    }
    let _plain_rx = enqueue(&mut sched, PLAIN_PROMPT, 10);
    assert_eq!(tick(&mut sched, &mut done), Tick::Prefill);
    let plain_id = sched
        .active_batch
        .iter_sequences()
        .find(|seq| seq.prompt_tokens == PLAIN_PROMPT)
        .map(|seq| seq.seq_id)
        .expect("plain row admitted");
    let mut lookup_verified = 0;
    while let Some(plain_before) = len_of(&sched, plain_id) {
        let lookup_before = len_of(&sched, lookup_id);
        assert_eq!(tick(&mut sched, &mut done), Tick::Decode);
        if let Some(plain_after) = len_of(&sched, plain_id) {
            assert_eq!(
                plain_after,
                plain_before + 1,
                "the plain row advanced one token"
            );
        }
        if let (Some(before), Some(after)) = (lookup_before, len_of(&sched, lookup_id))
            && after > before + 1
        {
            lookup_verified += 1;
        }
    }
    assert!(
        lookup_verified > 0,
        "the prompt-lookup row verified while the plain row decoded"
    );
    drain(&mut sched, &mut done);
    assert_eq!(tokens_of(&done, &prompt), cycle_from(4, 48));
    assert_eq!(tokens_of(&done, PLAIN_PROMPT), cycle_from(9, 10));
}

/// Issue #2227: every post-prefill B=1 token costs one decode forward,
/// including the terminal token. The `max_tokens >= 3` cases reach the
/// lookahead pipeline; without the terminal-budget guard they submit and
/// replay two forwards on the final tick.
#[test]
fn b1_length_boundary_submits_one_decode_forward_per_post_prefill_token() {
    for (max_tokens, completion_tokens, decode_forwards) in
        [(0, 1, 0), (1, 1, 0), (2, 2, 1), (3, 3, 2), (4, 4, 3)]
    {
        let mut sched = scheduler(None);
        let mut done = Done::new();
        let rx = enqueue(&mut sched, PLAIN_PROMPT, max_tokens);
        drain(&mut sched, &mut done);
        let result = completion(&rx);
        assert_eq!(
            result.generated_token_ids,
            cycle_from(9, completion_tokens),
            "max_tokens={max_tokens}"
        );
        assert_eq!(result.finish_reason, "length", "max_tokens={max_tokens}");
        assert_eq!(
            sched.decode_forward_calls, decode_forwards,
            "max_tokens={max_tokens} submitted an avoidable decode forward"
        );
    }
}

/// The terminal B=1 tick commits its pending draw without another submit or
/// replay. B=2 remains on the pre-existing two-append teardown and sync replay.
#[test]
fn b1_terminal_length_skips_two_forwards_and_leaves_b2_unchanged() {
    let mut b1 = scheduler(None);
    let mut b1_done = Done::new();
    let b1_rx = enqueue(&mut b1, PLAIN_PROMPT, 3);
    assert_eq!(tick(&mut b1, &mut b1_done), Tick::Prefill);
    let ids = vec![only_row(&b1)];
    assert!(b1.lookahead_params(&ids).is_some(), "dense B=1 is eligible");
    assert_eq!(tick(&mut b1, &mut b1_done), Tick::Decode);
    assert!(
        b1.decode_lookahead.is_some(),
        "bootstrap decode primed the terminal token"
    );
    assert_eq!(b1.decode_forward_calls, 2, "sync bootstrap plus prime");
    assert_eq!(tick(&mut b1, &mut b1_done), Tick::Decode);
    assert_eq!(
        b1.decode_forward_calls, 2,
        "terminal B=1 commits the pending token without submit or replay"
    );
    let b1_result = completion(&b1_rx);
    assert_eq!(b1_result.generated_token_ids, cycle_from(9, 3));
    assert_eq!(b1_result.finish_reason, "length");

    let sibling_prompt = [1, 3, 5, 7, 9];
    let mut b2 = scheduler(None);
    let mut b2_done = Done::new();
    let target_rx = enqueue(&mut b2, PLAIN_PROMPT, 3);
    let sibling_rx = enqueue(&mut b2, &sibling_prompt, 3);
    assert_eq!(tick(&mut b2, &mut b2_done), Tick::Prefill);
    assert_eq!(tick(&mut b2, &mut b2_done), Tick::Prefill);
    let ids = b2.active_batch.sequence_ids();
    assert_eq!(ids.len(), 2, "both rows are active for B=2 decode");
    assert!(b2.lookahead_params(&ids).is_some(), "dense B=2 is eligible");
    assert_eq!(tick(&mut b2, &mut b2_done), Tick::Decode);
    assert!(
        b2.decode_lookahead.is_some(),
        "B=2 bootstrap decode primed the terminal tokens"
    );
    assert_eq!(b2.decode_forward_calls, 2, "batched bootstrap plus prime");
    assert_eq!(tick(&mut b2, &mut b2_done), Tick::Decode);
    assert_eq!(
        b2.decode_forward_calls, 4,
        "B=2 keeps next submit, teardown, and synchronous replay"
    );
    let target = completion(&target_rx);
    let sibling = completion(&sibling_rx);
    assert_eq!(target.generated_token_ids, b1_result.generated_token_ids);
    assert_eq!(target.finish_reason, b1_result.finish_reason);
    assert_eq!(sibling.generated_token_ids, cycle_from(10, 3));
    assert_eq!(sibling.finish_reason, "length");
}

/// An eligible stochastic row commits the device draw that was already
/// pending. It neither advances the RNG with a future submit nor re-draws the
/// terminal token synchronously.
#[test]
fn b1_stochastic_terminal_length_commits_the_pending_draw() {
    let mut sched = scheduler(None);
    let mut done = Done::new();
    let (rx, _cancelled) = enqueue_with_sampling(
        &mut sched,
        PLAIN_PROMPT,
        3,
        SamplingConfig {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: Some(2227),
            ..SamplingConfig::default()
        },
    );
    assert_eq!(tick(&mut sched, &mut done), Tick::Prefill);
    let ids = vec![only_row(&sched)];
    assert!(
        sched.lookahead_params(&ids).is_some(),
        "the seeded stochastic row stays fused-lookahead eligible"
    );
    assert_eq!(tick(&mut sched, &mut done), Tick::Decode);
    let pending = mlxcel_core::engine::try_tokens_to_host(
        &sched
            .decode_lookahead
            .as_ref()
            .expect("bootstrap decode primed the terminal stochastic draw")
            .tokens,
    )
    .expect("read pending stochastic draw")[0];
    assert_eq!(tick(&mut sched, &mut done), Tick::Decode);
    assert_eq!(sched.decode_forward_calls, 2);
    let result = completion(&rx);
    assert_eq!(result.generated_token_ids.last(), Some(&pending));
    assert_eq!(result.finish_reason, "length");
}

/// EOS keeps precedence when the pending token also reaches `max_tokens`.
/// The optimized terminal branch still finishes through `Engine::finish_rows`.
#[test]
fn b1_terminal_token_eos_precedes_the_length_boundary() {
    let mut sched = scheduler(None);
    let mut done = Done::new();
    let (rx, _cancelled) = enqueue_with(&mut sched, PLAIN_PROMPT, 3, &[11]);
    assert_eq!(tick(&mut sched, &mut done), Tick::Prefill);
    let ids = vec![only_row(&sched)];
    assert!(sched.lookahead_params(&ids).is_some());
    assert_eq!(tick(&mut sched, &mut done), Tick::Decode);
    assert!(sched.decode_lookahead.is_some());
    assert_eq!(tick(&mut sched, &mut done), Tick::Decode);
    assert_eq!(sched.decode_forward_calls, 2);
    let result = completion(&rx);
    assert_eq!(result.generated_token_ids, cycle_from(9, 2));
    assert_eq!(result.finish_reason, "stop");
}

/// Rules 2 and 4 (I3): with the flag on and no row proposing, the lookahead
/// gate admits the batch and the ticks pipeline, as without the flag.
#[test]
fn proposal_free_ticks_stay_pipelined() {
    let mut done = Done::new();
    let mut sched = scheduler(Some(4));
    assert!(!sched.should_dispatch_speculative());
    let rx = enqueue(&mut sched, PLAIN_PROMPT, 12);
    assert_eq!(tick(&mut sched, &mut done), Tick::Prefill);
    let ids = vec![only_row(&sched)];
    assert!(
        sched.lookahead_params(&ids).is_some(),
        "prompt-lookup dispatch keeps the lookahead gate open"
    );
    let mut pipelined_ticks = 0;
    while tick(&mut sched, &mut done) != Tick::Idle {
        if sched.decode_lookahead.is_some() {
            pipelined_ticks += 1;
        }
    }
    assert!(pipelined_ticks > 0, "proposal-free ticks pipelined");
    assert_eq!(tokens_of(&done, PLAIN_PROMPT), cycle_from(9, 12));
    assert!(acceptance(&rx).is_none(), "nothing was proposed");
}

/// Rule 3 (I2, I3): when a row starts proposing while a step is in flight,
/// that tick commits the in-flight step and submits no prime, and the next
/// tick verifies synchronously.
#[test]
fn a_proposal_makes_the_next_tick_synchronous_without_a_prime() {
    let mut done = Done::new();
    let mut sched = scheduler(Some(4));
    // The reply `9, 10, 11, 12, 13` reaches `12, 13`, which the prompt
    // continues with `14, 15`; before that nothing repeats.
    let prompt = [12, 13, 14, 15, 0, 2, 4, 6, 8];
    let rx = enqueue(&mut sched, &prompt, 14);
    assert_eq!(tick(&mut sched, &mut done), Tick::Prefill);
    let id = only_row(&sched);
    let mut switched = false;
    while !switched {
        let n_before = len_of(&sched, id).expect("the row finished before it proposed");
        let was_pipelined = sched.decode_lookahead.is_some();
        assert_eq!(tick(&mut sched, &mut done), Tick::Decode);
        if was_pipelined && sched.decode_lookahead.is_none() {
            assert_eq!(
                len_of(&sched, id),
                Some(n_before + 1),
                "the switch tick commits exactly the in-flight step"
            );
            assert_eq!(tick(&mut sched, &mut done), Tick::Decode);
            assert!(
                sched.decode_lookahead.is_none(),
                "the verify tick primes nothing"
            );
            let n_verify = len_of(&sched, id).unwrap_or(usize::MAX);
            assert!(
                n_verify > n_before + 2,
                "the tick after the switch verified a block ({} -> {n_verify})",
                n_before + 1
            );
            switched = true;
        }
    }
    drain(&mut sched, &mut done);
    assert_eq!(tokens_of(&done, &prompt), cycle_from(9, 14));
    assert!(acceptance(&rx).is_some_and(|(_, accepted)| accepted > 0));
}

/// I5: above `--prompt-lookup-max-batch` no row is asked, so no verify round
/// runs and no governor sees a miss; once the batch shrinks, the remaining
/// row resumes with the tokens it kept observing.
#[test]
fn rows_above_the_max_batch_run_no_verify_round() {
    let prompt = copy_prompt();
    let long_prompt: Vec<i32> = (1..21).map(|i| (i % VOCAB) as i32).collect();
    let mut done = Done::new();
    let mut sched = scheduler(Some(1));
    let short_rx = enqueue(&mut sched, &prompt, 6);
    let long_rx = enqueue(&mut sched, &long_prompt, 40);
    assert_eq!(tick(&mut sched, &mut done), Tick::Prefill);
    assert_eq!(tick(&mut sched, &mut done), Tick::Prefill);
    while sched.active_batch.len() == 2 {
        assert_eq!(tick(&mut sched, &mut done), Tick::Decode);
        for seq in sched.active_batch.iter_sequences() {
            let row = seq.token_only.as_ref().expect("eligible row is primed");
            let stats = row
                .drafter
                .prompt_lookup()
                .expect("prompt lookup row")
                .stats();
            assert_eq!(
                (stats.rounds, stats.drafted_rounds, stats.paused_rounds),
                (0, 0, 0),
                "a gated-out row calls no draft_block"
            );
        }
    }
    drain(&mut sched, &mut done);
    assert_eq!(tokens_of(&done, &prompt), cycle_from(4, 6));
    assert!(
        acceptance(&short_rx).is_none(),
        "the short row never verified"
    );
    assert_eq!(tokens_of(&done, &long_prompt), cycle_from(5, 40));
    assert!(
        acceptance(&long_rx).is_some_and(|(_, accepted)| accepted > 0),
        "the remaining row verified once the batch shrank"
    );
}

/// I6: a request admitted while a lookahead is in flight goes through the
/// admission teardown first, is primed at its own prefill completion, and
/// both rows then decode exactly their solo streams.
#[test]
fn admission_during_an_in_flight_lookahead_tears_it_down_first() {
    let prompt = copy_prompt();
    let mut done = Done::new();
    let mut sched = scheduler(Some(4));
    let _plain_rx = enqueue(&mut sched, PLAIN_PROMPT, 14);
    assert_eq!(tick(&mut sched, &mut done), Tick::Prefill);
    let mut guard = 0;
    while sched.decode_lookahead.is_none() {
        assert_eq!(tick(&mut sched, &mut done), Tick::Decode);
        guard += 1;
        assert!(guard < 10, "the proposal-free row pipelined");
    }
    let lookup_rx = enqueue(&mut sched, &prompt, 30);
    assert_eq!(tick(&mut sched, &mut done), Tick::Prefill);
    assert!(
        sched.decode_lookahead.is_none(),
        "admission tore the lookahead down"
    );
    assert!(
        sched
            .active_batch
            .iter_sequences()
            .find(|seq| seq.prompt_tokens == prompt)
            .is_some_and(|seq| seq.token_only.is_some()),
        "the admitted row is primed"
    );
    drain(&mut sched, &mut done);
    assert_eq!(tokens_of(&done, PLAIN_PROMPT), cycle_from(9, 14));
    assert_eq!(tokens_of(&done, &prompt), cycle_from(4, 30));
    assert!(acceptance(&lookup_rx).is_some_and(|(_, accepted)| accepted > 0));
}

/// The first eligible prefill warms every verify width up once, also for a
/// one-token request that finishes at prefill (the server's startup warmup),
/// and leaves the state as the prefill left it (I1 after the prefill tick),
/// so the next request decodes exactly its plain stream. Without prompt
/// lookup nothing is warmed.
#[test]
fn the_first_prefill_warms_the_verify_widths_once() {
    let mut done = Done::new();
    let mut plain = scheduler(None);
    let _plain_rx = enqueue(&mut plain, PLAIN_PROMPT, 1);
    drain(&mut plain, &mut done);
    assert!(!plain.prompt_lookup_widths_warmed);

    let mut sched = scheduler(Some(4));
    let warm_rx = enqueue(&mut sched, PLAIN_PROMPT, 1);
    assert_eq!(tick(&mut sched, &mut done), Tick::Prefill);
    assert!(
        sched.prompt_lookup_widths_warmed,
        "a request that finishes at prefill warms the widths"
    );
    assert!(acceptance(&warm_rx).is_none());
    let prompt = copy_prompt();
    let rx = enqueue(&mut sched, &prompt, 20);
    drain(&mut sched, &mut done);
    assert_eq!(tokens_of(&done, &prompt), cycle_from(4, 20));
    assert!(acceptance(&rx).is_some_and(|(_, accepted)| accepted > 0));

    let mut fresh = scheduler(Some(4));
    let _rx = enqueue(&mut fresh, &prompt, 20);
    // `tick` checks I1 right after the prefill that warmed the widths.
    assert_eq!(tick(&mut fresh, &mut done), Tick::Prefill);
    assert!(fresh.prompt_lookup_widths_warmed);
    drain(&mut fresh, &mut done);
    assert_eq!(tokens_of(&done, &prompt), cycle_from(4, 20));
}

/// I5, context bound: under `--max-kv-size` with context shifting off, a
/// proposal is capped so the verify forward appends no position past the
/// token that ends the row (`ContextBound::verify_room`), the verify-width
/// warmup is left to a prefill with room for the widest block, and the row
/// emits exactly the plain decode's tokens up to the bound.
#[test]
fn proposals_stop_at_the_context_bound() {
    let prompt = copy_prompt();
    let bound = prompt.len() + 6;
    let mut done = Done::new();
    let mut plain = scheduler(None).with_max_kv_size(Some(bound));
    let _plain_rx = enqueue(&mut plain, &prompt, 40);
    drain(&mut plain, &mut done);
    // The bound stops the row once `prompt + generated + 1 >= bound`.
    let plain_tokens = tokens_of(&done, &prompt);
    assert_eq!(plain_tokens, cycle_from(4, 5));

    let mut done = Done::new();
    let mut sched = scheduler(Some(4)).with_max_kv_size(Some(bound));
    let rx = enqueue(&mut sched, &prompt, 40);
    assert_eq!(tick(&mut sched, &mut done), Tick::Prefill);
    assert!(
        !sched.prompt_lookup_widths_warmed,
        "a prompt this close to the bound does not take the widest block"
    );
    let id = only_row(&sched);
    let seq = sched.active_batch.get(id).expect("decoding row");
    // 20 prompt positions and one emitted token leave room for 26 - 23 = 3
    // proposals, under both max_draft (7) and max_tokens (40).
    assert_eq!(
        prompt_lookup::proposal_budget(
            seq,
            PromptLookupConfig::default().max_draft,
            sched.context_bound()
        ),
        3
    );
    drain(&mut sched, &mut done);
    assert_eq!(tokens_of(&done, &prompt), plain_tokens);
    // The governor may start narrower than the room (Gated probation); the
    // room caps every round either way.
    let (proposed, accepted) = acceptance(&rx).expect("prompt lookup verified");
    assert!(
        accepted > 0 && proposed <= 3,
        "{accepted}/{proposed} within a room of 3"
    );
}

/// I4 hardening: a drafter whose observed count runs ahead of its row's
/// tokens is dropped and the row decodes plainly, instead of the slice
/// panicking the scheduler thread.
#[test]
fn a_drafter_ahead_of_its_row_decodes_plainly() {
    let prompt = copy_prompt();
    let mut done = Done::new();
    let mut sched = scheduler(Some(4));
    let rx = enqueue(&mut sched, &prompt, 12);
    assert_eq!(tick(&mut sched, &mut done), Tick::Prefill);
    let id = only_row(&sched);
    sched
        .active_batch
        .get_mut(id)
        .and_then(|seq| seq.token_only.as_mut())
        .expect("eligible row is primed")
        .observed = usize::MAX;
    assert_eq!(tick(&mut sched, &mut done), Tick::Decode);
    assert!(
        sched
            .active_batch
            .get(id)
            .is_some_and(|seq| seq.token_only.is_none()),
        "the inconsistent drafter was dropped"
    );
    drain(&mut sched, &mut done);
    assert_eq!(tokens_of(&done, &prompt), cycle_from(4, 12));
    assert!(acceptance(&rx).is_none(), "no verify round ran");
}

/// I1 and I7, EOS inside an accepted block: the stop id sits in the middle
/// of the proposals a verify round accepts, so the round ends the row at the
/// stop token, commits nothing after it (the unwound tail leaves the KV at
/// the row's committed tokens, I1, checked before the row is released), and
/// the reply equals the plain decode with the same stop id.
#[test]
fn an_eos_inside_an_accepted_block_ends_the_row_there() {
    let prompt = copy_prompt();
    // The reply is `4, 5, 6, ..`; the stop id 11 ends it and is not stored.
    let eos = [11];
    let mut done = Done::new();
    let mut plain = scheduler(None);
    let _plain_rx = enqueue_with(&mut plain, &prompt, 40, &eos);
    drain(&mut plain, &mut done);
    let plain_tokens = tokens_of(&done, &prompt);
    assert_eq!(plain_tokens, cycle_from(4, 7));

    let mut sched = scheduler(Some(4));
    let (rx, _cancelled) = enqueue_with(&mut sched, &prompt, 40, &eos);
    let mut ended = None;
    let mut guard = 0;
    while ended.is_none() {
        guard += 1;
        assert!(guard < 100, "the row never finished");
        match sched.decide_action() {
            BatchSchedulerAction::Prefill(id) => {
                sched.discard_lookahead();
                sched.execute_prefill(id);
            }
            BatchSchedulerAction::Decode(ids) => sched.execute_decode_step(&ids),
            other => panic!("unexpected scheduler action {other:?}"),
        }
        if sched.decode_lookahead.is_some() {
            // Finishing tokens force the lookahead teardown in
            // `finalize_completed`; a finished row is inspected after it.
            sched.discard_lookahead();
        }
        let finished: Vec<(SequenceId, usize, Vec<i32>)> = sched
            .active_batch
            .iter_sequences()
            .filter(|seq| seq.state.is_finished())
            .map(|seq| {
                (
                    seq.seq_id,
                    // The stop token is not stored, and every stored token
                    // was fed before the one that stopped the row.
                    seq.prompt_tokens.len() + seq.generated_tokens.len(),
                    seq.generated_tokens.clone(),
                )
            })
            .collect();
        if let Some((id, committed, tokens)) = finished.into_iter().next() {
            let offset = sched
                .engine
                .pool_mut()
                .get_caches_mut(id)
                .expect("the finished row still holds its sequence")[0]
                .offset as usize;
            assert_eq!(
                offset, committed,
                "the KV holds the committed tokens, not the unwound tail of the block"
            );
            ended = Some(tokens);
        }
        sched.finalize_completed();
    }
    let tokens = ended.expect("finished");
    assert_eq!(tokens, plain_tokens);
    assert_eq!(
        tokens.last(),
        Some(&10),
        "the stop token itself is not kept"
    );
    let (proposed, accepted) = acceptance(&rx).expect("prompt lookup verified");
    assert!(
        accepted > 0 && accepted <= proposed,
        "{accepted}/{proposed}: the stop token fell inside a verified block"
    );
}

/// I8: a request cancelled while a verify round is due is not interrupted
/// mid-round. The round runs to its end and leaves the row's KV at its
/// committed tokens (I1, checked by `tick` before the row is released), and
/// `finalize_completed` then releases it after the tick.
#[test]
fn a_cancel_during_a_round_finalizes_after_the_round() {
    let prompt = copy_prompt();
    let mut done = Done::new();
    let mut sched = scheduler(Some(4));
    let (rx, cancelled) = enqueue_with(&mut sched, &prompt, 40, &[]);
    assert_eq!(tick(&mut sched, &mut done), Tick::Prefill);
    let id = only_row(&sched);
    // Tick until a synchronous tick is next, where the row verifies.
    while sched.decode_lookahead.is_some() {
        assert_eq!(tick(&mut sched, &mut done), Tick::Decode);
    }
    assert!(len_of(&sched, id).is_some(), "the row is still decoding");
    cancelled.store(true, std::sync::atomic::Ordering::Relaxed);
    assert_eq!(tick(&mut sched, &mut done), Tick::Decode);
    assert!(
        sched.active_batch.get(id).is_none(),
        "the cancelled row is released after the tick"
    );
    assert!(sched.decode_lookahead.is_none());
    assert_eq!(tick(&mut sched, &mut done), Tick::Idle);
    let (proposed, _) = acceptance(&rx).expect("the round ran before the cancel took effect");
    assert!(proposed > 0, "a verify round ran in the cancelled tick");
}

/// Exercise the production resolver so this regression covers the server's
/// opt-in settings as well as the governor and scheduler integration.
#[cfg(feature = "cuda")]
fn break_even_scheduler() -> BatchScheduler {
    let config = crate::server::config::ServerConfig {
        draft_kind: Some("prompt-lookup".to_string()),
        ..Default::default()
    };
    let mut dispatch =
        crate::server::SpeculativeDispatch::resolve(&config, std::path::Path::new("unused"))
            .expect("token-only dispatch needs no checkpoint");
    if let crate::server::SpeculativeDispatch::PromptLookup { config, .. } = &mut dispatch {
        config.ngram_max = 2;
    }
    scheduler(None).with_speculative_dispatch(dispatch)
}

#[cfg(feature = "cuda")]
#[test]
fn low_yield_verify_rounds_pause_pipeline_and_resume_on_copy() {
    let mut sched = break_even_scheduler();
    let mut done = Done::new();
    // Every true bigram continues with one correct token and then an error.
    // Use bigram lookup to land exactly one proposal. Once decoding
    // has written its own cycle, a long shadow match can recover copying.
    let prompt = low_yield_prompt();
    let rx = enqueue(&mut sched, &prompt, 80);
    assert_eq!(tick(&mut sched, &mut done), Tick::Prefill);
    let id = only_row(&sched);
    let mut paused_after = None;
    let mut pipelined_while_paused = 0;
    let mut resumed = false;
    for _ in 0..100 {
        if tick(&mut sched, &mut done) == Tick::Idle {
            break;
        }
        let Some(seq) = sched.active_batch.get(id) else {
            continue;
        };
        let stats = seq
            .token_only
            .as_ref()
            .expect("eligible row")
            .drafter
            .prompt_lookup()
            .expect("prompt lookup row")
            .stats();
        if stats.paused_rounds > 0 && paused_after.is_none() {
            assert_eq!(stats.drafted_rounds, 2, "two low-yield verifies pause");
            assert_eq!(
                stats.accepted_draft_tokens, 2,
                "one token landed each round"
            );
            paused_after = Some(stats.drafted_rounds);
        }
        if let Some(rounds) = paused_after {
            if stats.drafted_rounds == rounds && sched.decode_lookahead.is_some() {
                pipelined_while_paused += 1;
            }
            if stats.shadow_confirmations > 0 && stats.drafted_rounds > rounds {
                resumed = true;
            }
        }
    }
    assert!(
        paused_after.is_some(),
        "low-yield rounds must pause (fails without server rule)"
    );
    assert!(
        pipelined_while_paused >= 2,
        "paused row promptly returns to actual lookahead ticks"
    );
    assert!(
        resumed,
        "shadow evidence restores actual verification on a copy stretch"
    );
    assert_eq!(tokens_of(&done, &prompt), cycle_from(0, 80));
    let (proposed, accepted) = acceptance(&rx).expect("verify counters");
    assert!(accepted > 2 && accepted <= proposed);
}

#[cfg(feature = "cuda")]
#[test]
fn server_break_even_keeps_copy_heavy_rows_verifying() {
    let mut sched = break_even_scheduler();
    let mut done = Done::new();
    let prompt = copy_prompt();
    let rx = enqueue(&mut sched, &prompt, 64);
    assert_eq!(tick(&mut sched, &mut done), Tick::Prefill);
    let id = only_row(&sched);
    let mut verified = 0;
    for _ in 0..100 {
        if tick(&mut sched, &mut done) == Tick::Idle {
            break;
        }
        if let Some(seq) = sched.active_batch.get(id) {
            let stats = seq
                .token_only
                .as_ref()
                .expect("eligible row")
                .drafter
                .prompt_lookup()
                .expect("prompt lookup row")
                .stats();
            assert_eq!(
                stats.paused_rounds, 0,
                "copy evidence stays above break-even"
            );
            if stats.drafted_rounds > verified {
                assert!(
                    sched.decode_lookahead.is_none(),
                    "copy round really verified synchronously"
                );
                verified = stats.drafted_rounds;
            }
        }
    }
    assert!(verified >= 4);
    assert_eq!(tokens_of(&done, &prompt), cycle_from(4, 64));
    let (proposed, accepted) = acceptance(&rx).expect("copy counters");
    assert!(accepted > 32 && accepted <= proposed);
}

#[cfg(feature = "cuda")]
fn low_yield_prompt() -> Vec<i32> {
    let mut prompt = Vec::new();
    for t in 0..VOCAB {
        prompt.extend([
            t as i32,
            ((t + 1) % VOCAB) as i32,
            ((t + 2) % VOCAB) as i32,
            ((t + 7) % VOCAB) as i32,
        ]);
    }
    prompt.extend([14, 15]);
    prompt
}

#[cfg(feature = "cuda")]
#[test]
fn break_even_evidence_is_per_row_while_a_copy_peer_keeps_the_tick_synchronous() {
    let mut sched = break_even_scheduler();
    let mut done = Done::new();
    let copy = copy_prompt();
    let low = low_yield_prompt();
    let _copy_rx = enqueue(&mut sched, &copy, 128);
    assert_eq!(tick(&mut sched, &mut done), Tick::Prefill);
    let copy_id = only_row(&sched);
    let _low_rx = enqueue(&mut sched, &low, 80);
    assert_eq!(tick(&mut sched, &mut done), Tick::Prefill);
    let low_id = sched
        .active_batch
        .iter_sequences()
        .find(|seq| seq.prompt_tokens == low)
        .expect("low-yield row admitted")
        .seq_id;
    let mut observed_pause = false;
    for _ in 0..10 {
        assert_eq!(tick(&mut sched, &mut done), Tick::Decode);
        let low_stats = sched
            .active_batch
            .get(low_id)
            .expect("low row active")
            .token_only
            .as_ref()
            .expect("low drafter")
            .drafter
            .prompt_lookup()
            .expect("prompt lookup row")
            .stats();
        let copy_stats = sched
            .active_batch
            .get(copy_id)
            .expect("copy row active")
            .token_only
            .as_ref()
            .expect("copy drafter")
            .drafter
            .prompt_lookup()
            .expect("prompt lookup row")
            .stats();
        assert_eq!(
            copy_stats.paused_rounds, 0,
            "peer low yield cannot pause copying"
        );
        if low_stats.paused_rounds > 0 {
            assert_eq!(low_stats.drafted_rounds, 2);
            assert!(copy_stats.drafted_rounds >= 3);
            assert!(
                sched.decode_lookahead.is_none(),
                "copy peer still verifies on the shared tick"
            );
            observed_pause = true;
            break;
        }
    }
    assert!(observed_pause);
    drain(&mut sched, &mut done);
    assert_eq!(tokens_of(&done, &copy), cycle_from(4, 128));
    assert_eq!(tokens_of(&done, &low), cycle_from(0, 80));
}

#[cfg(feature = "cuda")]
#[test]
fn server_reprimes_after_the_first_proposal_free_ask() {
    let mut sched = break_even_scheduler();
    let mut done = Done::new();
    let _rx = enqueue(&mut sched, PLAIN_PROMPT, 12);
    assert_eq!(tick(&mut sched, &mut done), Tick::Prefill);
    assert_eq!(tick(&mut sched, &mut done), Tick::Decode);
    assert!(sched.decode_lookahead.is_some());
    drain(&mut sched, &mut done);
    assert_eq!(tokens_of(&done, PLAIN_PROMPT), cycle_from(9, 12));
}

/// The cap is on total context, including generated tokens. Exercise the
/// real scheduler admission query with an allocated paged/dense sequence;
/// query-only lengths avoid allocating long attention fixtures. Restore
/// the actual token state before running another model step.
#[test]
fn paged_prompt_lookup_cap_covers_measured_continuation_and_stops_at_boundary() {
    use mlxcel_core::cache::SequenceStateBackend;

    for storage in [DecodeStorageBackend::Paged, DecodeStorageBackend::Dense] {
        let mut sched = scheduler_with_storage(Some(2), storage);
        let mut done = Done::new();
        let prompt = copy_prompt();
        let _rx = enqueue(&mut sched, &prompt, 1024);
        assert_eq!(tick(&mut sched, &mut done), Tick::Prefill);
        let id = only_row(&sched);
        let expected_backend = if storage == DecodeStorageBackend::Paged {
            SequenceStateBackend::PagedKvCache
        } else {
            SequenceStateBackend::DenseKvCache
        };
        assert_eq!(
            sched.engine.pool().get(id).expect("allocated row").backend,
            expected_backend
        );
        let generated = sched
            .active_batch
            .get(id)
            .expect("row")
            .generated_tokens
            .clone();
        for (prompt_len, generated_len, paged_admitted) in [
            (4099, 1, true),
            (4101, 1, true),
            (4102, 256, true), // 4358 total: the measured request envelope.
            (4607, 1, true),   // Exact 4608 boundary is inclusive.
            (4608, 1, false),
            (4102, 507, false), // The continuation also crosses the cap.
            (8192, 1, false),
        ] {
            let seq = sched.active_batch.get_mut(id).expect("row");
            seq.prompt_tokens.resize(prompt_len, 0);
            seq.generated_tokens.resize(generated_len, 0);
            let admitted = sched.prompt_lookup_tick_applies(&[id]);
            assert_eq!(
                admitted,
                storage == DecodeStorageBackend::Dense || paged_admitted,
                "storage {storage:?}, prompt {prompt_len}, generated {generated_len}"
            );
        }
        let seq = sched.active_batch.get_mut(id).expect("row");
        seq.prompt_tokens = prompt.clone();
        seq.generated_tokens = generated;
        seq.max_tokens = 24;
        drain(&mut sched, &mut done);
        assert_eq!(tokens_of(&done, &prompt), cycle_from(4, 24));
    }
}

/// A draft model that agrees with the cycle target below token 8 and proposes
/// `t + 2` from there on, so a row walking low tokens accepts every proposal
/// and a row walking high tokens rejects the first.
fn disagreeing_draft() -> crate::LoadedModel {
    crate::LoadedModel::Llama(walk_llama(|t| {
        if t < 8 {
            (t + 1) % VOCAB
        } else {
            (t + 2) % VOCAB
        }
    }))
}

/// A cycle-target scheduler under a full-model dispatch of width 3.
fn full_model_scheduler(max_batch: usize) -> BatchScheduler {
    scheduler(None)
        .with_speculative_dispatch(crate::server::SpeculativeDispatch::FullModel {
            // No checkpoint on disk: the model is handed over directly.
            draft_model_path: std::path::PathBuf::from("unused-draft-model"),
            block_size: 3,
            block_size_source: crate::cli::draft_block_policy::BlockSizeSource::Override,
            user_requested_explicit_kind: true,
            max_batch,
        })
        .with_full_model_drafter(Some(disagreeing_draft()))
}

/// The row with `prompt`: its draft bookkeeping, its real draft-cache length,
/// its stats and how many tokens it generated.
fn draft_row(
    sched: &BatchScheduler,
    prompt: &[i32],
) -> (
    usize,
    usize,
    mlxcel_core::speculative::full_model_drafter::FullModelDraftStats,
    usize,
) {
    let seq = sched
        .active_batch
        .iter_sequences()
        .find(|seq| seq.prompt_tokens == prompt)
        .expect("the row is active");
    let state = seq
        .token_only
        .as_ref()
        .and_then(|row| row.drafter.full_model())
        .expect("a full-model row");
    (
        state.cached_positions(),
        state.cache_seq_len(),
        state.stats(),
        seq.generated_tokens.len(),
    )
}

/// B = 2 with both rows verifying in one tick, one accepting everything and
/// one rejecting at its first proposal: each row's draft caches roll back on
/// their own and both streams stay the target's. The assertion reads the real
/// cache length as well as the bookkeeping, so a rollback that does not trim
/// fails here.
#[test]
fn two_full_model_rows_roll_back_independently() {
    let accepting: Vec<i32> = vec![0, 1, 2];
    let rejecting: Vec<i32> = vec![6, 7, 8];
    let mut done = Done::new();
    let mut sched = full_model_scheduler(2);
    let accepting_rx = enqueue(&mut sched, &accepting, 40);
    let rejecting_rx = enqueue(&mut sched, &rejecting, 40);
    assert_eq!(tick(&mut sched, &mut done), Tick::Prefill);
    assert_eq!(tick(&mut sched, &mut done), Tick::Prefill);
    // The prefill records the prompt; nothing is forwarded yet.
    for prompt in [&accepting, &rejecting] {
        let (held, real, stats, _) = draft_row(&sched, prompt);
        assert_eq!((held, real, stats.draft_forwards), (0, 0, 0));
    }

    assert_eq!(tick(&mut sched, &mut done), Tick::Decode);
    // Accepting: current 3 drafts 4, 5, 6 and the target confirms all three
    // and adds 7. Held: prompt, 3, and the fed 4, 5; 6 and 7 are owed.
    let (held, real, stats, generated) = draft_row(&sched, &accepting);
    assert_eq!(generated, 1 + 3 + 1);
    assert_eq!((held, real), (3 + 1 + 2, 3 + 1 + 2), "accepting row");
    assert_eq!(
        (stats.proposed_draft_tokens, stats.accepted_draft_tokens),
        (3, 3)
    );
    // Rejecting: current 9 drafts 11, 13, 15 and the target says 10. Held:
    // prompt and 9; the two fed proposals are trimmed.
    let (held, real, stats, generated) = draft_row(&sched, &rejecting);
    assert_eq!(generated, 1 + 1);
    assert_eq!((held, real), (3 + 1, 3 + 1), "rejecting row");
    assert_eq!(
        (stats.proposed_draft_tokens, stats.accepted_draft_tokens),
        (3, 0)
    );

    drain(&mut sched, &mut done);
    assert_eq!(tokens_of(&done, &accepting), cycle_from(3, 40));
    assert_eq!(tokens_of(&done, &rejecting), cycle_from(9, 40));
    assert!(
        acceptance_of(&accepting_rx, mlxcel_core::drafter::DrafterKind::FullModel)
            .is_some_and(|(_, accepted)| accepted > 0)
    );
    assert!(acceptance_of(&rejecting_rx, mlxcel_core::drafter::DrafterKind::FullModel).is_some());
}

/// Above the gate a row runs no draft forward and holds no draft KV; when the
/// batch shrinks it catches up on everything plain steps emitted and verifies.
#[test]
fn a_full_model_row_gated_out_holds_nothing_and_catches_up() {
    let short: Vec<i32> = vec![6, 7, 8];
    let long: Vec<i32> = vec![0, 1, 2];
    let mut done = Done::new();
    let mut sched = full_model_scheduler(1);
    let short_rx = enqueue(&mut sched, &short, 6);
    let long_rx = enqueue(&mut sched, &long, 30);
    assert_eq!(tick(&mut sched, &mut done), Tick::Prefill);
    assert_eq!(tick(&mut sched, &mut done), Tick::Prefill);
    while sched.active_batch.len() == 2 {
        for prompt in [&short, &long] {
            let (held, real, stats, _) = draft_row(&sched, prompt);
            assert_eq!(
                (held, real, stats.draft_forwards, stats.rounds),
                (0, 0, 0, 0)
            );
        }
        assert_eq!(tick(&mut sched, &mut done), Tick::Decode);
    }
    // Tick out any lookahead left from the gated ticks, as far as the first
    // round, then check the catch-up landed on the row's own context.
    let mut rounds = 0;
    while rounds == 0 {
        assert_eq!(tick(&mut sched, &mut done), Tick::Decode);
        rounds = draft_row(&sched, &long).2.rounds;
    }
    let (held, real, _, generated) = draft_row(&sched, &long);
    assert_eq!(
        held, real,
        "bookkeeping and caches agree after the catch-up"
    );
    assert!(
        held >= long.len() + generated - 3,
        "the caches hold the context up to the owed tokens: {held} of {}",
        long.len() + generated
    );
    drain(&mut sched, &mut done);
    assert_eq!(tokens_of(&done, &short), cycle_from(9, 6));
    assert_eq!(tokens_of(&done, &long), cycle_from(3, 30));
    assert!(acceptance_of(&short_rx, mlxcel_core::drafter::DrafterKind::FullModel).is_none());
    assert!(
        acceptance_of(&long_rx, mlxcel_core::drafter::DrafterKind::FullModel)
            .is_some_and(|(_, accepted)| accepted > 0)
    );
}

/// Under `--context-shift` the target's caches shift at `--max-kv-size` and the
/// context bound never stops the row, but the draft caches have no shift path.
/// So a full-model row stops being asked once its context passes the bound: its
/// draft caches stay within the bound plus one block, it keeps its stats, and
/// it decodes the plain stream from there on.
#[test]
fn a_full_model_row_past_the_max_kv_size_stops_drafting() {
    let prompt: Vec<i32> = vec![0, 1, 2];
    let bound = 8;
    let block_size = 3;
    let max_tokens = 40;
    let shift = ContextRetentionPolicy {
        context_shift: true,
        n_keep: 0,
    };

    let mut done = Done::new();
    let mut plain = scheduler(None)
        .with_max_kv_size(Some(bound))
        .with_context_retention(shift);
    let _plain_rx = enqueue(&mut plain, &prompt, max_tokens);
    drain(&mut plain, &mut done);
    let plain_tokens = tokens_of(&done, &prompt);
    assert_eq!(plain_tokens, cycle_from(3, max_tokens));

    let mut done = Done::new();
    let mut sched = full_model_scheduler(1)
        .with_max_kv_size(Some(bound))
        .with_context_retention(shift);
    let rx = enqueue(&mut sched, &prompt, max_tokens);
    assert_eq!(tick(&mut sched, &mut done), Tick::Prefill);
    let mut generated = 0;
    while generated < max_tokens - 5 {
        assert_eq!(tick(&mut sched, &mut done), Tick::Decode);
        let (held, real, _, now) = draft_row(&sched, &prompt);
        assert!(
            held <= bound + block_size && real <= bound + block_size,
            "draft KV past the bound after {now} tokens: {held} held, {real} real"
        );
        generated = now;
    }
    drain(&mut sched, &mut done);
    assert_eq!(tokens_of(&done, &prompt), plain_tokens);
    assert!(
        acceptance_of(&rx, mlxcel_core::drafter::DrafterKind::FullModel)
            .is_some_and(|(_, accepted)| accepted > 0),
        "the row verified rounds before it reached the bound"
    );
}

/// Cancelling a row leaves its neighbour's draft state alone, and a request
/// admitted into the freed slot starts from a fresh state and its own stream.
#[test]
fn a_cancelled_full_model_row_frees_its_slot_for_a_fresh_one() {
    let leaving: Vec<i32> = vec![0, 1, 2];
    let staying: Vec<i32> = vec![6, 7, 8];
    let joining: Vec<i32> = vec![1, 2, 3];
    let mut done = Done::new();
    let mut sched = full_model_scheduler(2);
    let (_leaving_rx, cancelled) = enqueue_with(&mut sched, &leaving, 40, &[]);
    enqueue(&mut sched, &staying, 40);
    assert_eq!(tick(&mut sched, &mut done), Tick::Prefill);
    assert_eq!(tick(&mut sched, &mut done), Tick::Prefill);
    assert_eq!(tick(&mut sched, &mut done), Tick::Decode);
    cancelled.store(true, std::sync::atomic::Ordering::Relaxed);
    assert_eq!(tick(&mut sched, &mut done), Tick::Decode);
    assert!(
        sched
            .active_batch
            .iter_sequences()
            .all(|seq| seq.prompt_tokens != leaving),
        "the cancelled row is gone"
    );
    let (held, real, _, _) = draft_row(&sched, &staying);
    assert_eq!(held, real, "the neighbour's caches are untouched");

    enqueue(&mut sched, &joining, 12);
    assert_eq!(tick(&mut sched, &mut done), Tick::Prefill);
    let (held, real, stats, _) = draft_row(&sched, &joining);
    assert_eq!((held, real, stats.rounds), (0, 0, 0));
    drain(&mut sched, &mut done);
    assert_eq!(tokens_of(&done, &joining), cycle_from(4, 12));
    assert_eq!(tokens_of(&done, &staying), cycle_from(9, 40));
}
