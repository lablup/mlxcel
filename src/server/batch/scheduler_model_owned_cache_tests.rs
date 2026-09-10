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
const EOS_ID: i32 = 7;

fn tensor(shape: &[i32], value: f32) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let len = shape.iter().product::<i32>() as usize;
    mlxcel_core::from_slice_f32(&vec![value; len], shape)
}

fn put(weights: &mut WeightMap, key: &str, shape: &[i32], value: f32) {
    weights.insert(key.to_string(), tensor(shape, value));
}

fn tiny_gemma3_args() -> Gemma3ModelArgs {
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

fn tiny_gemma3_weights() -> WeightMap {
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

fn tiny_gemma3() -> Gemma3Wrapper {
    let args = tiny_gemma3_args();
    let weights = tiny_gemma3_weights();
    Gemma3Wrapper::new(Gemma3Model::from_weights(&weights, &args).expect("tiny gemma3 loads"))
}

fn test_store() -> Arc<PromptCacheStore> {
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
fn scheduler(store: Arc<PromptCacheStore>) -> BatchScheduler {
    let (_tx, rx) = mpsc::channel();
    let sched = BatchScheduler::with_config(
        LoadedModel::Gemma3(tiny_gemma3()),
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

fn cache_ctx() -> PromptCacheRequestContext {
    PromptCacheRequestContext {
        model_id: "tiny-gemma3".to_string(),
        lora_id: None,
        template_sig: "tpl-sig-v1".to_string(),
        session_key: "session-1".to_string(),
        mm_digest: MultimodalDigest::empty(),
        history_prompt: None,
        history_prefix_tokens: None,
    }
}

fn options() -> ServerGenerateOptions {
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
        reasoning_control: None,
        prompt_cache_ctx: Some(cache_ctx()),
        structured: None,
        grammar: None,
        image_soft_tokens: None,
        pre_rendered_prompt_tokens: None,
    }
}

fn enqueue(sched: &mut BatchScheduler, prompt_tokens: Vec<i32>) -> mpsc::Receiver<GenerateEvent> {
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

fn run_to_completion(sched: &mut BatchScheduler, rx: &mpsc::Receiver<GenerateEvent>) {
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
            Ok(GenerateEvent::Done(_)) => return,
            Ok(GenerateEvent::Error(err)) => panic!("unexpected generation error: {err}"),
            Err(err) => panic!("generation did not finish: {err}"),
        }
    }
}

/// Tokens the tiny model can actually embed (`vocab_size == 8`).
fn prompt(n: usize) -> Vec<i32> {
    (0..n).map(|i| (i % 6) as i32).collect()
}

/// The predicate the #1346 gate keys on has to be the model's own layout: the
/// allocated backend is exactly the thing that lies, and the layout override
/// is a server-config artifact that never consults the model.
#[test]
fn paged_override_does_not_change_the_model_owned_natural_backend() {
    let mut sched = scheduler(test_store());

    assert_eq!(
        sched.model.sequence_state_layout().backend,
        SequenceStateBackend::ModelOwned,
        "Gemma 3 keeps its K/V in ModelOwnedSequenceState"
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
    let set = sched.cache_pool.get(seq_id).expect("sequence is active");
    assert_eq!(
        set.backend,
        SequenceStateBackend::PagedKvCache,
        "the ALLOCATED backend claims paged; this is what the old gate read"
    );
    assert!(
        set.caches.is_empty(),
        "yet the sequence carries no per-layer K/V: the block table is shadow accounting"
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

    let declined = sched.try_adopt_cached_prefix(&ctx, &second, true);
    assert!(
        declined.is_none(),
        "a partial snapshot match must decline when require_whole_entry is set"
    );

    // Same lookup, gate off: the partial match the line above declined does
    // adopt, confirming the decline came from the gate and not from some
    // other miss (e.g. a bad key or a below-minimum prefix).
    let adopted = sched
        .try_adopt_cached_prefix(&ctx, &second, false)
        .expect("without the gate the same partial match must adopt");
    assert_eq!(adopted.1, 20, "adopted at the 20-token common prefix");
}
