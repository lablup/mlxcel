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

//! Model-owned families under the paged decode backend and the prompt cache
//! (issue #1346).
//!
//! Gemma 3 is the smallest real family with the shape that caused the defect:
//! natural sequence-state backend `ModelOwned`, `make_caches()` empty, but
//! `supports_batching()` and `supports_paged_decode_backend()` both true, so
//! `--decode-storage-backend paged` allocates it on the paged backend for
//! shadow block-table accounting. These tests use the real `Gemma3Wrapper`
//! rather than a stub precisely because the interesting fact is the
//! DISAGREEMENT between the model's own layout and the one it was allocated
//! on, and a stub could be written to agree.

use super::*;
use mlxcel_core::cache::{SequenceStateBackend, SequenceStateLayout};
use mlxcel_core::generate::{LanguageModel, SamplingConfig};
use mlxcel_core::weights::WeightMap;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use crate::LoadedModel;
use crate::models::gemma3::ModelArgs as Gemma3ModelArgs;
use crate::models::{Gemma3Model, Gemma3Wrapper};
use crate::server::config::{
    DecodeStorageBackend, PreemptionPolicy, PromptCacheRequestContext, ReasoningBudgetOverride,
};
use crate::server::model_provider::GenerateEvent;
use crate::server::prompt_cache::{PromptCacheConfig, PromptCacheStore, key::MultimodalDigest};
use crate::server::state::BatchMetrics;
use crate::tokenizer::MlxcelTokenizer;

const HIDDEN: i32 = 4;
const VOCAB: i32 = 8;
const INTERMEDIATE: i32 = 8;
const HEAD_DIM: i32 = 2;
pub(super) const EOS_ID: i32 = 7;

fn tensor(shape: &[i32], value: f32) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let len = shape.iter().product::<i32>() as usize;
    mlxcel_core::from_slice_f32(&vec![value; len], shape)
}

fn put(weights: &mut WeightMap, key: &str, shape: &[i32], value: f32) {
    weights.insert(key.to_string(), tensor(shape, value));
}

pub(super) fn tiny_gemma3_args() -> Gemma3ModelArgs {
    Gemma3ModelArgs {
        model_type: "gemma3_text".to_string(),
        hidden_size: HIDDEN as usize,
        num_hidden_layers: 1,
        intermediate_size: INTERMEDIATE as usize,
        num_attention_heads: 2,
        head_dim: HEAD_DIM as usize,
        rms_norm_eps: 1e-6,
        vocab_size: VOCAB as usize,
        num_key_value_heads: 1,
        rope_theta: 10_000.0,
        rope_local_base_freq: 10_000.0,
        query_pre_attn_scalar: 2.0,
        sliding_window: 8,
        sliding_window_pattern: 1,
        max_position_embeddings: 4096,
        rope_scaling: None,
        quantization: None,
    }
}

pub(super) fn tiny_gemma3_weights() -> WeightMap {
    let mut w: WeightMap = WeightMap::new();
    put(&mut w, "model.embed_tokens.weight", &[VOCAB, HIDDEN], 0.05);
    put(
        &mut w,
        "model.layers.0.self_attn.q_proj.weight",
        &[2 * HEAD_DIM, HIDDEN],
        0.10,
    );
    put(
        &mut w,
        "model.layers.0.self_attn.k_proj.weight",
        &[HEAD_DIM, HIDDEN],
        0.20,
    );
    put(
        &mut w,
        "model.layers.0.self_attn.v_proj.weight",
        &[HEAD_DIM, HIDDEN],
        0.30,
    );
    put(
        &mut w,
        "model.layers.0.self_attn.o_proj.weight",
        &[HIDDEN, 2 * HEAD_DIM],
        0.15,
    );
    put(
        &mut w,
        "model.layers.0.self_attn.q_norm.weight",
        &[HEAD_DIM],
        1.0,
    );
    put(
        &mut w,
        "model.layers.0.self_attn.k_norm.weight",
        &[HEAD_DIM],
        1.0,
    );
    put(
        &mut w,
        "model.layers.0.mlp.gate_proj.weight",
        &[INTERMEDIATE, HIDDEN],
        0.10,
    );
    put(
        &mut w,
        "model.layers.0.mlp.up_proj.weight",
        &[INTERMEDIATE, HIDDEN],
        0.20,
    );
    put(
        &mut w,
        "model.layers.0.mlp.down_proj.weight",
        &[HIDDEN, INTERMEDIATE],
        0.05,
    );
    for norm in [
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.pre_feedforward_layernorm.weight",
        "model.layers.0.post_feedforward_layernorm.weight",
        "model.norm.weight",
    ] {
        put(&mut w, norm, &[HIDDEN], 1.0);
    }
    put(&mut w, "lm_head.weight", &[VOCAB, HIDDEN], 0.10);
    w
}

pub(super) fn tiny_gemma3() -> Gemma3Wrapper {
    let args = tiny_gemma3_args();
    let weights = tiny_gemma3_weights();
    Gemma3Wrapper::new(Gemma3Model::from_weights(&weights, &args).expect("tiny gemma3 loads"))
}

pub(super) fn test_store() -> Arc<PromptCacheStore> {
    Arc::new(PromptCacheStore::with_config(PromptCacheConfig::new(
        true,
        1 << 20,
        32,
        Duration::from_secs(600),
        4,
    )))
}

/// A scheduler shaped like the default server: batching on, paged decode
/// storage, prompt cache installed.
pub(super) fn scheduler(store: Arc<PromptCacheStore>) -> BatchScheduler {
    scheduler_with_model(tiny_gemma3(), store)
}

/// [`scheduler`] around a caller-built Gemma 3 (for example one whose only
/// layer is a sliding one).
pub(super) fn scheduler_with_model(
    model: Gemma3Wrapper,
    store: Arc<PromptCacheStore>,
) -> BatchScheduler {
    let (_tx, rx) = mpsc::channel();
    let sched = BatchScheduler::with_config(
        LoadedModel::Gemma3(model),
        MlxcelTokenizer::stub(),
        vec![EOS_ID],
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

pub(super) fn cache_ctx() -> PromptCacheRequestContext {
    PromptCacheRequestContext {
        model_id: "tiny-gemma3".to_string(),
        lora_id: None,
        template_sig: "tpl-sig-v1".to_string(),
        session_key: "session-1".to_string(),
        mm_digest: MultimodalDigest::empty(),
        history_prompt: None,
        history_prefix_tokens: None,
        root_prompt: None,
        root_prefix_tokens: None,
        snapshot_lineage: Default::default(),
        model_reasoning_effort: None,
    }
}

pub(super) fn options() -> ServerGenerateOptions {
    ServerGenerateOptions {
        n_indent: 0,
        t_max_predict_ms: None,
        reasoning_budget_message: None,
        retention: Default::default(),
        dry_breaker_strings: None,
        logit_bias: Vec::new(),
        logit_bias_texts: Vec::new(),
        post_sampling_probs: false,
        max_tokens: 1,
        sampling: SamplingConfig::greedy(),
        stop_sequences: None,
        ignore_eos: false,
        priority: RequestPriority::Normal,
        lora_scales: None,
        logprobs: Default::default(),
        reasoning_budget: ReasoningBudgetOverride::InheritServerDefault,
        thinking_enter_block_on_start: false,
        thinking_primed_close: None,
        reasoning_control: None,
        prompt_cache_ctx: Some(cache_ctx()),
        structured: None,
        grammar: None,
        image_soft_tokens: None,
        video_num_frames: None,
        video_min_frames: None,
        pre_rendered_prompt_tokens: None,
        admission_notification: None,
        count_prompt_only: false,
        model_reasoning_effort: None,
    }
}

pub(super) fn enqueue(
    sched: &mut BatchScheduler,
    prompt_tokens: Vec<i32>,
) -> mpsc::Receiver<GenerateEvent> {
    let (tx, rx) = mpsc::channel();
    sched.enqueue_request(
        "prompt".to_string(),
        Some(prompt_tokens),
        options(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        tx,
        Arc::new(AtomicBool::new(false)),
        true,
    );
    rx
}

pub(super) fn run_to_completion(
    sched: &mut BatchScheduler,
    rx: &mpsc::Receiver<GenerateEvent>,
) -> crate::server::model_provider::GenerationResult {
    for _ in 0..8 {
        match sched.decide_action() {
            BatchSchedulerAction::Prefill(id) => sched.execute_prefill(id),
            BatchSchedulerAction::Decode(ids) => sched.execute_decode_step(&ids),
            BatchSchedulerAction::Idle => break,
            other => panic!("unexpected scheduler action {other:?}"),
        }
        if sched.active_batch.is_empty() && sched.prefill_queue.is_empty() {
            break;
        }
    }
    loop {
        match rx.recv_timeout(Duration::from_secs(5)) {
            Ok(GenerateEvent::Token(_, _))
            | Ok(GenerateEvent::TokenWithLogprobs(_, _, _))
            | Ok(GenerateEvent::Prefill(_)) => {}
            Ok(GenerateEvent::Admitted) => {}
            Ok(GenerateEvent::RequestMemoryRejected(error)) => {
                panic!("unexpected memory rejection: {error}")
            }
            Ok(GenerateEvent::Done(result)) => return result,
            Ok(GenerateEvent::Error(err)) => panic!("unexpected generation error: {err}"),
            Err(err) => panic!("generation did not finish: {err}"),
        }
    }
}

/// Tokens the tiny model can actually embed (`vocab_size == 8`).
pub(super) fn prompt(n: usize) -> Vec<i32> {
    (0..n).map(|i| (i % 6) as i32).collect()
}

/// The predicate the #1346 gate keys on has to be the model's own layout: the
/// allocated backend is exactly the thing that lies, and the layout override
/// is a server-config artifact that never consults the model.
#[test]
fn paged_override_does_not_change_the_scheduler_owned_natural_backend() {
    let mut sched = scheduler(test_store());

    assert_eq!(
        sched.engine.model().sequence_state_layout().backend,
        SequenceStateBackend::SchedulerOwned,
        "Gemma 3 keeps its K/V in the pool entry's layer state (issue #3254)"
    );
    assert!(matches!(
        sched.sequence_state_layout_override(),
        Some(SequenceStateLayout {
            backend: SequenceStateBackend::PagedKvCache,
            ..
        })
    ));

    let seq_id = sched
        .allocate_sequence_state()
        .expect("paged allocation succeeds");
    let set = sched.engine.pool().get(seq_id).expect("sequence is active");
    assert_eq!(
        set.backend,
        SequenceStateBackend::PagedKvCache,
        "the ALLOCATED backend claims paged; this is what the old gate read"
    );
    assert!(
        set.caches.is_empty(),
        "yet the sequence carries no per-layer K/V: the block table is shadow accounting"
    );
    assert_eq!(
        set.layer_state().map(<[_]>::len),
        Some(sched.engine.model().num_layers()),
        "the real state is the layer state, present whatever backend was allocated"
    );
}

/// End to end through the scheduler: a model-owned family completes a request
/// without ever putting a detached K/V set in the store, and the next request
/// extending the same conversation never consults the K/V bucket.
///
/// The assertions read the K/V counters (`entries`, `inserts`, `lookups`)
/// rather than the store's combined `len()`, because Gemma 3 answers
/// `supports_snapshot_reuse()` since #1335 and donates an exact-prefix
/// snapshot. That donation is a copy of the model's own caches, which is a
/// different object from the pool's shadow block table and is exactly what
/// #1346 wanted a model-owned family to use instead. `len()` counts both
/// buckets, so it can no longer tell the two apart.
///
/// One assertion is gone rather than moved: `prompt_cache_reject_model_owned_state`
/// counts the donate-side decline, and Gemma 3 now returns on the snapshot
/// branch before it. That counter belongs to model-owned families without
/// snapshot reuse, which this file has no model for. The guarantee it guarded
/// is still asserted here directly, as an empty K/V bucket.
#[test]
fn model_owned_paged_family_never_donates_or_adopts_kv() {
    let store = test_store();
    let mut sched = scheduler(store.clone());

    let first = prompt(40);
    let rx = enqueue(&mut sched, first.clone());
    run_to_completion(&mut sched, &rx);

    let stats = store.stats();
    // `PromptCacheStats::entries` is the COMBINED live count, so the K/V bucket
    // is the difference. `inserts` counts only K/V inserts (snapshots have
    // their own `snapshot_inserts`), which is the cleaner of the two signals.
    assert_eq!(
        stats.entries - stats.snapshot_entries,
        0,
        "a shadow paged sequence must never reach the K/V bucket"
    );
    assert_eq!(stats.inserts, 0, "no detached K/V set was donated");
    assert_eq!(
        stats.snapshot_entries, 1,
        "the donation Gemma 3 does make is a model-state snapshot (#1335)"
    );

    // Turn 2: same conversation, four more tokens. Before #1346 this adopted
    // the shadow block table and skipped prefill for the first 40 tokens with
    // no K/V behind it. The prefix is skipped again now, but through the
    // snapshot restore, so the state the skip claims actually exists.
    let mut second = first.clone();
    second.extend([1, 2, 3, 4]);
    let _rx2 = enqueue(&mut sched, second.clone());

    let queued = sched
        .prefill_queue
        .dequeue()
        .expect("the second request is queued for prefill");
    assert!(
        queued.prefill_start_offset >= first.len(),
        "the snapshot restore covers at least the shared 40-token prefix, got {}",
        queued.prefill_start_offset
    );
    assert_eq!(queued.already_cached_tokens, queued.prefill_start_offset);
    assert_eq!(queued.prompt_tokens.len(), second.len());
    assert_eq!(
        store.stats().snapshot_hits,
        1,
        "the skip came from the snapshot bucket"
    );

    // The adopt gate specifically, which the assertions above do not reach.
    // `lookups` only advances inside `finalize_miss`, so a zero here is the
    // difference between "returned before the K/V store lookup" and "looked,
    // missed, and moved on". That distinction is the whole point of the second
    // gate: it is what refuses a shadow entry that reached the store by some
    // other route, such as a store populated before #1346 landed.
    assert_eq!(
        store.stats().lookups,
        0,
        "the adopt gate must return before the K/V store lookup, not after a guaranteed miss"
    );
}

/// The `require_whole_entry` gate, on the snapshot branch specifically
/// (adcecb3b, a review finding on #1335 before that fix landed).
///
/// `try_adopt_cached_prefix` is exercised directly rather than through
/// `enqueue`, because driving `require_whole_entry == true` through the
/// normal admission path needs an actual multimodal request; the gate itself
/// only reads the flag it is handed; this fixture's single global-attention
/// layer is always truncatable (#1145), which is what makes a partial
/// snapshot match reachable here without a VLM checkpoint.
#[test]
fn snapshot_partial_match_declines_under_require_whole_entry() {
    let store = test_store();
    let mut sched = scheduler(store.clone());

    let first = prompt(40);
    let rx = enqueue(&mut sched, first.clone());
    run_to_completion(&mut sched, &rx);
    assert_eq!(
        store.stats().snapshot_entries,
        1,
        "turn 1 must donate a snapshot for turn 2 to diverge from"
    );

    // Turn 2 diverges from the stored entry at token 20: tokens 0..20 match,
    // then the conversation takes a different turn. The store still reports
    // a Hit at matched_len == 20, because the model agrees it can truncate
    // there; whether the caller MAY adopt that partial match is exactly what
    // `require_whole_entry` decides.
    let mut second = first[..20].to_vec();
    second.push(5);
    second.extend([1, 2, 3, 4]);
    let ctx = cache_ctx();

    let declined = sched.try_adopt_cached_prefix(&ctx, &second, true).adopted();
    assert!(
        declined.is_none(),
        "a partial snapshot match must decline when require_whole_entry is set"
    );

    // Same lookup, gate off: the partial match the line above declined does
    // adopt, confirming the decline came from the gate and not from some
    // other miss (e.g. a bad key or a below-minimum prefix).
    let adopted = sched
        .try_adopt_cached_prefix(&ctx, &second, false)
        .adopted()
        .expect("without the gate the same partial match must adopt");
    assert_eq!(adopted.1, 20, "adopted at the 20-token common prefix");
}

// ── a warm-up keeps the boundary snapshot it extends (issue #3467) ──────────

/// History length of [`turn`]; the prompt adds a four-token generation prompt.
pub(super) const TURN_BOUNDARY: usize = 36;
/// Tokens a [`enqueue_turn`] request decodes.
pub(super) const TURN_MAX_TOKENS: usize = 3;

/// The chat turn the warm-up tests replay: 36 history tokens and a
/// four-token generation prompt.
pub(super) fn turn() -> Vec<i32> {
    prompt(TURN_BOUNDARY + 4)
}

/// A chat turn of `tokens` whose history boundary is `tokens[..boundary]`, as
/// the server builds it. The fixture samples EOS straight away, so the turn
/// ignores EOS and decodes [`TURN_MAX_TOKENS`] greedy tokens.
pub(super) fn enqueue_turn(
    sched: &mut BatchScheduler,
    tokens: &[i32],
    boundary: usize,
) -> mpsc::Receiver<GenerateEvent> {
    let mut ctx = cache_ctx();
    ctx.history_prefix_tokens = Some(tokens[..boundary].to_vec());
    let mut opts = options();
    opts.prompt_cache_ctx = Some(ctx);
    opts.max_tokens = TURN_MAX_TOKENS;
    opts.ignore_eos = true;
    let (tx, rx) = mpsc::channel();
    sched.enqueue_request(
        "prompt".to_string(),
        Some(tokens.to_vec()),
        opts,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        tx,
        Arc::new(AtomicBool::new(false)),
        true,
    );
    rx
}

/// Drive the scheduler until `rx` reports the turn done. Unlike
/// [`run_to_completion`] this keeps stepping through multi-token decode and
/// runs the completion pass that donates a decode-finished sequence.
pub(super) fn finish_turn(
    sched: &mut BatchScheduler,
    rx: &mpsc::Receiver<GenerateEvent>,
) -> crate::server::model_provider::GenerationResult {
    for _ in 0..64 {
        match rx.try_recv() {
            Ok(GenerateEvent::Done(result)) => return result,
            Ok(GenerateEvent::Error(err)) => panic!("unexpected generation error: {err}"),
            Ok(_) | Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => panic!("response channel closed"),
        }
        match sched.decide_action() {
            BatchSchedulerAction::Prefill(id) => sched.execute_prefill(id),
            BatchSchedulerAction::Decode(ids) => sched.execute_decode_step(&ids),
            BatchSchedulerAction::Idle => {}
            other => panic!("unexpected scheduler action {other:?}"),
        }
        sched.finalize_completed();
    }
    panic!("the turn did not finish");
}

/// The queued request's sequence and how many prompt tokens it adopted.
pub(super) fn queued(sched: &BatchScheduler) -> (SequenceId, usize) {
    let seq = sched.prefill_queue.peek().expect("the request is queued");
    (seq.seq_id, seq.prefill_start_offset)
}

/// Queue a warm-up for `target` in the turn's cache bucket, run it, and
/// require that it restored a base snapshot and stored its target.
pub(super) fn run_warmup(sched: &mut BatchScheduler, target: &[i32]) {
    let run = sched
        .batch_observability
        .snapshot()
        .prompt_cache_warmups_run;
    sched.enqueue_prompt_cache_warmup(target.to_vec(), cache_ctx());
    sched.run_next_prompt_cache_warmup();
    assert_eq!(
        sched
            .batch_observability
            .snapshot()
            .prompt_cache_warmups_run,
        run + 1,
        "the warm-up restored a base snapshot and stored its target"
    );
}

/// Forward `tokens` for `seq_id` as one unpadded piece and return the
/// logits at its last position.
fn piece_logits(sched: &mut BatchScheduler, seq_id: SequenceId, tokens: &[i32]) -> Vec<u8> {
    let input = mlxcel_core::from_slice_i32(tokens, &[1, tokens.len() as i32]);
    let outcome = sched
        .engine
        .prefill(&PrefillStep {
            seq_id,
            input: &input,
            embeddings: None,
            mask: None,
            last_pos: tokens.len() - 1,
            trim_excess: 0,
            eval: true,
        })
        .expect("the piece forwards");
    outcome.eval.expect("the logits evaluate");
    mlxcel_core::array_to_raw_bytes(&outcome.logits)
}

/// The logits of `tokens[boundary..]` after a cold prefill of
/// `tokens[..boundary]`, on a scheduler of its own: the partition a cold
/// request forwards (history, then the rest). Call it before building the
/// scheduler under test, which installs its own default stream.
pub(super) fn cold_suffix_logits(model: Gemma3Wrapper, tokens: &[i32], boundary: usize) -> Vec<u8> {
    let mut sched = scheduler_with_model(model, test_store());
    let seq_id = sched
        .allocate_sequence_state()
        .expect("a sequence for the cold run");
    let _ = piece_logits(&mut sched, seq_id, &tokens[..boundary]);
    let logits = piece_logits(&mut sched, seq_id, &tokens[boundary..]);
    sched.release_sequence_caches(seq_id);
    logits
}

/// The same logits from the state the scheduler restored into the queued
/// request `queued`, copied to a probe sequence so the request itself is
/// left untouched.
pub(super) fn restored_suffix_logits(
    sched: &mut BatchScheduler,
    queued: SequenceId,
    tokens: &[i32],
    boundary: usize,
) -> Vec<u8> {
    use crate::server::prompt_cache::state_snapshot;
    let state = state_snapshot::snapshot_sequence_state(
        sched.engine.model(),
        sched.engine.pool(),
        queued,
        boundary,
    )
    .expect("the adopted sequence holds the restored prefix");
    let probe = sched
        .allocate_sequence_state()
        .expect("a probe sequence for the restored state");
    let (model, pool) = sched.engine.parts_mut();
    state_snapshot::restore_sequence_state(model, pool, probe, &state)
        .expect("the probe takes the restored state");
    let logits = piece_logits(sched, probe, &tokens[boundary..]);
    sched.release_sequence_caches(probe);
    logits
}

/// Builds a warm-up target (the next turn's history prefix) from a turn's
/// prompt, its history boundary, and its reply.
pub(super) type WarmTarget = fn(&[i32], usize, &[i32]) -> Vec<i32>;

/// A turn, its background warm-up, then the identical turn again (a retry or
/// regenerate): the retry must restore the turn's history-boundary snapshot.
/// The warm-up stores the next turn's history prefix as a `Boundary` entry,
/// and before #3467 that insert superseded the boundary snapshot it had
/// restored from, the only stored vector that is a prefix of the retried
/// prompt.
///
/// Returns the store, the scheduler, and the reply.
pub(super) fn retry_after_warmup_restores_the_boundary(
    model: fn() -> Gemma3Wrapper,
    warm_target: WarmTarget,
) -> (Arc<PromptCacheStore>, BatchScheduler, Vec<i32>) {
    let tokens = turn();
    let cold_logits = cold_suffix_logits(model(), &tokens, TURN_BOUNDARY);
    let store = test_store();
    let mut sched = scheduler_with_model(model(), store.clone());

    let rx = enqueue_turn(&mut sched, &tokens, TURN_BOUNDARY);
    assert_eq!(queued(&sched).1, 0, "a cold store restores nothing");
    let cold = finish_turn(&mut sched, &rx).generated_token_ids;
    assert_eq!(cold.len(), TURN_MAX_TOKENS);
    assert!(
        store.stats().snapshot_entries >= 1,
        "the turn stores its history-boundary snapshot"
    );

    let target = warm_target(&tokens, TURN_BOUNDARY, &cold);
    assert!(
        target.len() > TURN_BOUNDARY && target[..TURN_BOUNDARY] == tokens[..TURN_BOUNDARY],
        "the warm-up target extends the history boundary"
    );
    assert!(
        !tokens.starts_with(&target),
        "the warm-up target is not itself a prefix of the retried prompt"
    );
    run_warmup(&mut sched, &target);
    assert_eq!(
        store.stats().snapshot_hits,
        0,
        "a warm-up lookup is not a request hit"
    );

    let rx = enqueue_turn(&mut sched, &tokens, TURN_BOUNDARY);
    let (seq_id, start) = queued(&sched);
    assert_eq!(
        (store.stats().snapshot_hits, start),
        (1, TURN_BOUNDARY),
        "the retried turn restores its history-boundary snapshot; only the generation prompt is prefilled"
    );
    assert_eq!(
        restored_suffix_logits(&mut sched, seq_id, &tokens, TURN_BOUNDARY),
        cold_logits,
        "the restored state gives the cold split prefill's logits bit for bit"
    );
    assert_eq!(
        finish_turn(&mut sched, &rx).generated_token_ids,
        cold,
        "and the same greedy tokens"
    );
    (store, sched, cold)
}

/// The warm-up target of the issue's sequence: the history boundary, then the
/// reply's token ids.
pub(super) fn history_then_reply(tokens: &[i32], boundary: usize, reply: &[i32]) -> Vec<i32> {
    let mut target = tokens[..boundary].to_vec();
    target.extend_from_slice(reply);
    target
}

/// The issue's sequence on the plain fixture. Its one global-attention layer
/// accepts a truncating restore (#1145), so before the fix the retry here
/// still adopted 36 tokens: the warmed entry cut back to its common prefix
/// with the prompt, which is the history boundary. What the fix changes on
/// this fixture is that the boundary snapshot itself is still stored, hence
/// the entry count. A model that cannot truncate prefilled from cold instead
/// (`scheduler_warmup_boundary_tests`, and the Raon variant of this test).
#[test]
fn a_retried_turn_restores_the_boundary_snapshot_its_warmup_extended() {
    let (store, _sched, _) =
        retry_after_warmup_restores_the_boundary(tiny_gemma3, history_then_reply);
    assert_eq!(
        store.stats().snapshot_entries,
        2,
        "the boundary snapshot and the warmed next-turn prefix"
    );
}
