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
        // Row `j` of the head reads the one-hot of `j - 1`.
        head[t * HIDDEN + (t + VOCAB - 1) % VOCAB] = 1.0;
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
        DecodeStorageBackend::Dense,
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
        sampling: SamplingConfig::greedy(),
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
    };
    let (tx, rx) = mpsc::channel();
    sched.enqueue_request(
        "prompt".to_string(),
        Some(prompt.to_vec()),
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

/// The Done result's prompt-lookup acceptance counters `(proposed,
/// accepted)`, `None` when it reports no drafter.
fn acceptance(rx: &mpsc::Receiver<GenerateEvent>) -> Option<(usize, usize)> {
    let mut accepted = None;
    while let Ok(event) = rx.try_recv() {
        match event {
            GenerateEvent::Done(result) => {
                accepted = result.speculative.map(|s| {
                    assert_eq!(
                        s.draft_kind,
                        mlxcel_core::drafter::DrafterKind::PromptLookup
                    );
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
            let row = seq.prompt_lookup.as_ref().expect("eligible row is primed");
            let stats = row.drafter.stats();
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
            .is_some_and(|seq| seq.prompt_lookup.is_some()),
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
        prompt_lookup::proposal_budget(seq, &PromptLookupConfig::default(), sched.context_bound()),
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
        .and_then(|seq| seq.prompt_lookup.as_mut())
        .expect("eligible row is primed")
        .observed = usize::MAX;
    assert_eq!(tick(&mut sched, &mut done), Tick::Decode);
    assert!(
        sched
            .active_batch
            .get(id)
            .is_some_and(|seq| seq.prompt_lookup.is_none()),
        "the inconsistent drafter was dropped"
    );
    drain(&mut sched, &mut done);
    assert_eq!(tokens_of(&done, &prompt), cycle_from(4, 12));
    assert!(acceptance(&rx).is_none(), "no verify round ran");
}
