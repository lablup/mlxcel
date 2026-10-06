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

//! Padded prefill for a model-owned family through the batch scheduler
//! (issue #1755).
//!
//! The scheduler's own post-pad trim reaches only the `CachePool` entry, which
//! a model-owned family leaves empty, so before the sequence-aware hook the pad
//! positions stayed in the model's caches and `offset` ran ahead of the real
//! token count. These tests force tile alignment on (it is otherwise M5-only),
//! run a real `Gemma3Wrapper` with one sliding and one global layer, and
//! compare its per-sequence state against the same prompt prefilled unpadded:
//! offsets, the physical sliding-window buffer, and the decoded tokens.

use super::*;
use mlxcel_core::generate::{LanguageModel, ModelStateSnapshot, SamplingConfig};
use mlxcel_core::utils::array_to_vec_f32;
use mlxcel_core::weights::WeightMap;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use super::pad_trim::set_alignment_override_for_test;
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

const HIDDEN: i32 = 8;
const VOCAB: i32 = 16;
const INTERMEDIATE: i32 = 16;
const HEAD_DIM: i32 = 4;
const HEADS: i32 = 2;
const LAYERS: usize = 2;
const WINDOW: usize = 8;

/// Deterministic, non-constant weights so every position's K/V differs.
fn tensor(shape: &[i32], seed: u32) -> mlxcel_core::UniquePtr<mlxcel_core::MlxArray> {
    let len = shape.iter().product::<i32>() as usize;
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    let values: Vec<f32> = (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.8
        })
        .collect();
    mlxcel_core::from_slice_f32(&values, shape)
}

fn tiny_gemma3() -> Gemma3Wrapper {
    let args = Gemma3ModelArgs {
        model_type: "gemma3_text".to_string(),
        hidden_size: HIDDEN as usize,
        num_hidden_layers: LAYERS,
        intermediate_size: INTERMEDIATE as usize,
        num_attention_heads: HEADS as usize,
        head_dim: HEAD_DIM as usize,
        rms_norm_eps: 1e-6,
        vocab_size: VOCAB as usize,
        num_key_value_heads: 1,
        rope_theta: 10_000.0,
        rope_local_base_freq: 10_000.0,
        query_pre_attn_scalar: HEAD_DIM as f32,
        sliding_window: WINDOW,
        // Layer 0 slides (RotatingKVCache), layer 1 is global (KVCache).
        sliding_window_pattern: 2,
        max_position_embeddings: 4096,
        rope_scaling: None,
        quantization: None,
    };
    let mut w = WeightMap::new();
    let mut seed = 1;
    let mut put = |w: &mut WeightMap, key: String, shape: &[i32]| {
        seed += 1;
        w.insert(key, tensor(shape, seed));
    };
    put(&mut w, "model.embed_tokens.weight".into(), &[VOCAB, HIDDEN]);
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}");
        put(
            &mut w,
            format!("{p}.self_attn.q_proj.weight"),
            &[HEADS * HEAD_DIM, HIDDEN],
        );
        put(
            &mut w,
            format!("{p}.self_attn.k_proj.weight"),
            &[HEAD_DIM, HIDDEN],
        );
        put(
            &mut w,
            format!("{p}.self_attn.v_proj.weight"),
            &[HEAD_DIM, HIDDEN],
        );
        put(
            &mut w,
            format!("{p}.self_attn.o_proj.weight"),
            &[HIDDEN, HEADS * HEAD_DIM],
        );
        put(
            &mut w,
            format!("{p}.mlp.gate_proj.weight"),
            &[INTERMEDIATE, HIDDEN],
        );
        put(
            &mut w,
            format!("{p}.mlp.up_proj.weight"),
            &[INTERMEDIATE, HIDDEN],
        );
        put(
            &mut w,
            format!("{p}.mlp.down_proj.weight"),
            &[HIDDEN, INTERMEDIATE],
        );
        for norm in ["self_attn.q_norm", "self_attn.k_norm"] {
            w.insert(
                format!("{p}.{norm}.weight"),
                mlxcel_core::from_slice_f32(&[0.0; HEAD_DIM as usize], &[HEAD_DIM]),
            );
        }
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            w.insert(
                format!("{p}.{norm}.weight"),
                mlxcel_core::from_slice_f32(&[0.0; HIDDEN as usize], &[HIDDEN]),
            );
        }
    }
    w.insert(
        "model.norm.weight".into(),
        mlxcel_core::from_slice_f32(&[0.0; HIDDEN as usize], &[HIDDEN]),
    );
    put(&mut w, "lm_head.weight".into(), &[VOCAB, HIDDEN]);
    Gemma3Wrapper::new(Gemma3Model::from_weights(&w, &args).expect("tiny gemma3 loads"))
}

fn scheduler(chunk: usize, backend: DecodeStorageBackend) -> BatchScheduler {
    let (_tx, rx) = mpsc::channel();
    let sched = BatchScheduler::with_config(
        LoadedModel::Gemma3(tiny_gemma3()),
        MlxcelTokenizer::stub(),
        Vec::new(),
        rx,
        4,
        8,
        Arc::new(BatchMetrics::new()),
        Arc::new(BatchObservability::new()),
        chunk,
        false,
        PreemptionPolicy::default(),
        1,
        backend,
    );
    install_thread_local_default_stream(sched.generation_stream.as_ref());
    sched
}

fn options(max_tokens: usize) -> ServerGenerateOptions {
    ServerGenerateOptions {
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
        ignore_eos: true,
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
    }
}

/// Model state of one sequence right after its prefill, plus its decode.
struct Run {
    snapshot: ModelStateSnapshot,
    tokens: Vec<i32>,
}

/// Prefill `prompt` (chunked when `chunk > 0`), snapshot the model-owned state
/// the moment the sequence enters the decode batch, then decode to completion.
fn run(prompt_len: usize, chunk: usize, align: bool, backend: DecodeStorageBackend) -> Run {
    set_alignment_override_for_test(Some(align));
    let mut sched = scheduler(chunk, backend);
    assert!(
        sched.model.supports_padded_prefill(),
        "Gemma 3 takes padded prefill (#1755)"
    );
    let prompt: Vec<i32> = (0..prompt_len)
        .map(|i| ((i * 7 + 3) % VOCAB as usize) as i32)
        .collect();
    let (tx, rx) = mpsc::channel();
    sched.enqueue_request(
        "prompt".to_string(),
        Some(prompt),
        options(6),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        tx,
        Arc::new(AtomicBool::new(false)),
        true,
    );

    let mut snapshot = None;
    for _ in 0..64 {
        match sched.decide_action() {
            BatchSchedulerAction::Prefill(id) => sched.execute_prefill(id),
            BatchSchedulerAction::Decode(ids) => sched.execute_decode_step(&ids),
            BatchSchedulerAction::Idle => break,
            other => panic!("unexpected scheduler action {other:?}"),
        }
        if snapshot.is_none()
            && let Some(seq_id) = sched.active_batch.sequence_ids().first().copied()
        {
            snapshot = sched.model.snapshot_sequence_state(seq_id, prompt_len);
        }
        sched.finalize_completed();
        if sched.active_batch.is_empty()
            && sched.prefill_queue.is_empty()
            && sched.chunked_prefill_seq.is_none()
        {
            break;
        }
    }
    set_alignment_override_for_test(None);

    let tokens = loop {
        match rx.recv_timeout(Duration::from_secs(5)) {
            Ok(GenerateEvent::Token(..))
            | Ok(GenerateEvent::TokenWithLogprobs(..))
            | Ok(GenerateEvent::Prefill(_)) => {}
            Ok(GenerateEvent::Done(result)) => break result.generated_token_ids,
            Ok(GenerateEvent::Error(err)) => panic!("unexpected generation error: {err}"),
            Err(err) => panic!("generation did not finish: {err}"),
        }
    };
    Run {
        snapshot: snapshot.expect("the sequence reached the decode batch"),
        tokens,
    }
}

fn scalar(snapshot: &ModelStateSnapshot, name: &str) -> i32 {
    let array = snapshot
        .tensor(name)
        .unwrap_or_else(|| panic!("snapshot has no {name}"));
    array_to_vec_f32(array)[0] as i32
}

/// `keys` of one layer as stored, cut to its first `len` slots on axis 2.
fn keys(snapshot: &ModelStateSnapshot, name: &str, len: Option<i32>) -> (Vec<i32>, Vec<f32>) {
    let array = snapshot
        .tensor(name)
        .unwrap_or_else(|| panic!("snapshot has no {name}"));
    let shape = mlxcel_core::array_shape(array);
    let stop = len.unwrap_or(shape[2]);
    let cut = mlxcel_core::slice(array, &[0, 0, 0, 0], &[shape[0], shape[1], stop, shape[3]]);
    (mlxcel_core::array_shape(&cut), array_to_vec_f32(&cut))
}

fn assert_close(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-4,
            "{label}[{i}]: padded {g} vs unpadded {w}"
        );
    }
}

/// The acceptance check of #1755: after a padded prefill, every layer's offset
/// equals the unpadded length, the sliding layer's PHYSICAL buffer equals the
/// unpadded one (no pad keys a later decode step could keep), and decode
/// produces the same tokens.
fn assert_padded_matches_unpadded(prompt_len: usize, chunk: usize, backend: DecodeStorageBackend) {
    let padded = run(prompt_len, chunk, true, backend);
    let plain = run(prompt_len, chunk, false, backend);
    let label = format!("prompt {prompt_len}, chunk {chunk}, {backend:?}");

    for snap in [&padded.snapshot, &plain.snapshot] {
        assert_eq!(
            scalar(snap, "layer0.rotating.offset"),
            prompt_len as i32,
            "{label}: sliding offset"
        );
        assert_eq!(
            scalar(snap, "layer1.standard.offset"),
            prompt_len as i32,
            "{label}: global offset"
        );
    }

    let (p_shape, p_keys) = keys(&padded.snapshot, "layer0.rotating.keys", None);
    let (u_shape, u_keys) = keys(&plain.snapshot, "layer0.rotating.keys", None);
    assert_eq!(p_shape, u_shape, "{label}: sliding layer physical buffer");
    assert_close(&format!("{label} sliding keys"), &p_keys, &u_keys);

    let live = Some(prompt_len as i32);
    let (_, p_keys) = keys(&padded.snapshot, "layer1.standard.keys", live);
    let (_, u_keys) = keys(&plain.snapshot, "layer1.standard.keys", live);
    assert_close(&format!("{label} global keys"), &p_keys, &u_keys);

    assert_eq!(
        padded.tokens.len(),
        6,
        "{label}: decoded every requested token"
    );
    assert_eq!(padded.tokens, plain.tokens, "{label}: decoded tokens");
}

#[test]
fn full_padded_prefill_within_the_window_matches_unpadded() {
    assert_padded_matches_unpadded(5, 0, DecodeStorageBackend::Dense);
}

#[test]
fn full_padded_prefill_past_the_window_matches_unpadded() {
    // 13 real tokens padded to 32, both past the 8-token window.
    assert_padded_matches_unpadded(13, 0, DecodeStorageBackend::Dense);
    // The default server shape: shadow paged accounting over model-owned K/V.
    assert_padded_matches_unpadded(13, 0, DecodeStorageBackend::Paged);
}

#[test]
fn chunked_padded_prefill_matches_unpadded() {
    // 37 tokens in 16-token chunks, each padded to 32: the first chunk and two
    // continuation chunks are rewound, the continuations onto a rolled window.
    assert_padded_matches_unpadded(37, 16, DecodeStorageBackend::Dense);
}

/// Padded prefill and snapshot reuse together, the combination #1752 had to
/// give up: a padded turn 1 donates a snapshot whose state holds exactly the
/// keyed tokens, and turn 2 adopts it instead of re-prefilling the prefix.
#[test]
fn padded_prefill_keeps_snapshot_reuse() {
    set_alignment_override_for_test(Some(true));
    let store = Arc::new(PromptCacheStore::with_config(PromptCacheConfig::new(
        true,
        1 << 20,
        32,
        Duration::from_secs(600),
        4,
    )));
    let mut sched =
        scheduler(0, DecodeStorageBackend::Paged).with_prompt_cache(Some(store.clone()));
    let ctx = PromptCacheRequestContext {
        model_id: "tiny-gemma3".to_string(),
        lora_id: None,
        template_sig: "tpl".to_string(),
        session_key: "session".to_string(),
        mm_digest: MultimodalDigest::empty(),
        history_prompt: None,
        history_prefix_tokens: None,
    };
    let enqueue = |sched: &mut BatchScheduler, prompt: Vec<i32>| {
        let mut opts = options(1);
        opts.prompt_cache_ctx = Some(ctx.clone());
        let (tx, rx) = mpsc::channel();
        sched.enqueue_request(
            "prompt".to_string(),
            Some(prompt),
            opts,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            tx,
            Arc::new(AtomicBool::new(false)),
            true,
        );
        rx
    };

    let first: Vec<i32> = (0..40).map(|i| i % 13).collect();
    let rx = enqueue(&mut sched, first.clone());
    for _ in 0..8 {
        match sched.decide_action() {
            BatchSchedulerAction::Prefill(id) => sched.execute_prefill(id),
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
            Ok(GenerateEvent::Done(_)) => break,
            Ok(GenerateEvent::Error(err)) => panic!("unexpected generation error: {err}"),
            Ok(_) => {}
            Err(err) => panic!("generation did not finish: {err}"),
        }
    }
    assert_eq!(
        store.stats().snapshot_entries,
        1,
        "the padded turn donated a snapshot"
    );

    let mut second = first.clone();
    second.extend([1, 2, 3, 4]);
    let _rx2 = enqueue(&mut sched, second);
    let queued = sched.prefill_queue.dequeue().expect("turn 2 is queued");
    set_alignment_override_for_test(None);
    assert!(
        queued.prefill_start_offset >= first.len(),
        "turn 2 adopts the padded turn's snapshot, got start {}",
        queued.prefill_start_offset
    );
    assert_eq!(store.stats().snapshot_hits, 1);
}
