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

//! Two concurrent requests against each request alone, on a real checkpoint
//! (#2172 review).
//!
//! #2172 moved Gemma 3 and Llama 4 batched decode on the paged backend from
//! the dense-pointer compat kernels to the per-row `attend` loop. The
//! synthetic route tests compare those at 1e-3; this runs the whole server
//! path (`BatchScheduler` ticks, `Engine::step` at B=2 and B=1, the lookahead
//! pipeline) on a real Gemma 3 checkpoint and requires each request's greedy
//! token stream to equal the one it produces when it runs alone. A failure
//! reports the first diverging position.
//!
//! `#[ignore]`: needs the GPU and a checkpoint. Run under
//! `MLXCEL_SDPA_DETERMINISTIC=1` with
//! `cargo test --profile test-fast --features cuda --lib scheduler_real_batch_parity -- --ignored --nocapture --test-threads=1`.
//! The checkpoint comes from `MLXCEL_GEMMA3_MODEL`, else
//! `models/mlx/gemma-3-1b-it-4bit`. A missing checkpoint or a run without the
//! deterministic flag is skipped with a message, and fails instead when
//! `MLXCEL_REQUIRE_MODELS=1`.

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, mpsc};

use super::*;
use crate::server::config::{DecodeStorageBackend, PreemptionPolicy, ReasoningBudgetOverride};
use crate::server::model_provider::GenerateEvent;
use crate::server::state::BatchMetrics;
use mlxcel_core::generate::SamplingConfig;

const MAX_TOKENS: usize = 48;
const PROMPTS: [&str; 2] = [
    "Write a short paragraph about the history of the printing press.",
    "List three reasons why the sky looks blue, one per line.",
];

fn skip_or_fail(reason: &str) {
    if std::env::var("MLXCEL_REQUIRE_MODELS").is_ok_and(|v| v != "0") {
        panic!("MLXCEL_REQUIRE_MODELS is set: {reason}");
    }
    eprintln!("SKIPPED scheduler_real_batch_parity: {reason}");
}

fn checkpoint() -> Option<PathBuf> {
    let path = std::env::var("MLXCEL_GEMMA3_MODEL").map_or_else(
        |_| PathBuf::from("models/mlx/gemma-3-1b-it-4bit"),
        PathBuf::from,
    );
    if path.join("config.json").exists() {
        Some(path)
    } else {
        skip_or_fail(&format!(
            "no checkpoint at {} (set MLXCEL_GEMMA3_MODEL)",
            path.display()
        ));
        None
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
        max_tokens: MAX_TOKENS,
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

fn enqueue(sched: &mut BatchScheduler, prompt: &[i32]) -> mpsc::Receiver<GenerateEvent> {
    let (tx, rx) = mpsc::channel();
    sched.enqueue_request(
        "prompt".to_string(),
        Some(prompt.to_vec()),
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

/// Tick until the scheduler is idle, then read each request's tokens.
/// Returns them with the number of decode ticks that ran more than one row.
fn drain(
    sched: &mut BatchScheduler,
    rxs: &[mpsc::Receiver<GenerateEvent>],
) -> (Vec<Vec<i32>>, usize) {
    let mut batched_ticks = 0;
    for _ in 0..(MAX_TOKENS * 8) {
        match sched.decide_action() {
            BatchSchedulerAction::Prefill(id) => {
                sched.discard_lookahead();
                sched.execute_prefill(id);
            }
            BatchSchedulerAction::Decode(ids) => {
                batched_ticks += usize::from(ids.len() > 1);
                sched.execute_decode_step(&ids);
            }
            BatchSchedulerAction::Idle => break,
            other => panic!("unexpected scheduler action {other:?}"),
        }
        sched.finalize_completed();
    }
    let tokens = rxs
        .iter()
        .map(|rx| {
            rx.try_iter()
                .find_map(|event| match event {
                    GenerateEvent::Done(result) => Some(result.generated_token_ids),
                    GenerateEvent::Error(err) => panic!("generation error: {err}"),
                    _ => None,
                })
                .expect("request finished")
        })
        .collect();
    (tokens, batched_ticks)
}

fn first_divergence(a: &[i32], b: &[i32]) -> Option<usize> {
    a.iter()
        .zip(b)
        .position(|(x, y)| x != y)
        .or((a.len() != b.len()).then(|| a.len().min(b.len())))
}

fn run(backend: DecodeStorageBackend) {
    if std::env::var("MLXCEL_SDPA_DETERMINISTIC").ok().as_deref() != Some("1") {
        skip_or_fail("run with MLXCEL_SDPA_DETERMINISTIC=1 so every route reduces in one order");
        return;
    }
    let Some(path) = checkpoint() else {
        return;
    };
    let _runtime = crate::initialize_runtime();
    let (model, tokenizer) = crate::load_model(&path).expect("load checkpoint");
    let prompts: Vec<Vec<i32>> = PROMPTS
        .iter()
        .map(|p| {
            tokenizer
                .encode(p, true)
                .expect("tokenize")
                .into_iter()
                .map(|t| t as i32)
                .collect()
        })
        .collect();
    let (_tx, rx) = mpsc::channel();
    let mut sched = BatchScheduler::with_config(
        model,
        tokenizer,
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
        backend,
    );
    install_thread_local_default_stream(sched.generation_stream.as_ref());
    assert_eq!(
        sched.decode_storage_backend, backend,
        "the scheduler resolved the requested backend"
    );

    let concurrent: Vec<_> = prompts.iter().map(|p| enqueue(&mut sched, p)).collect();
    let (together, batched_ticks) = drain(&mut sched, &concurrent);
    eprintln!("{backend:?}: {batched_ticks} decode ticks ran two rows");
    assert!(
        batched_ticks >= MAX_TOKENS / 2,
        "the concurrent run decoded at B=2 ({batched_ticks} ticks)"
    );
    let alone: Vec<Vec<i32>> = prompts
        .iter()
        .map(|p| {
            let rx = enqueue(&mut sched, p);
            let (mut tokens, batched_ticks) = drain(&mut sched, std::slice::from_ref(&rx));
            assert_eq!(batched_ticks, 0, "a lone request decodes at B=1");
            tokens.remove(0)
        })
        .collect();

    for (row, (b2, b1)) in together.iter().zip(&alone).enumerate() {
        eprintln!(
            "{backend:?} request {row}: {} tokens together, {} alone",
            b2.len(),
            b1.len()
        );
        assert_eq!(
            b1.len(),
            MAX_TOKENS,
            "request {row} alone ran to max_tokens"
        );
        if let Some(pos) = first_divergence(b2, b1) {
            panic!(
                "{backend:?} request {row}: B=2 and B=1 token streams diverge at position {pos} \
                 (together {:?}, alone {:?})",
                &b2[pos..b2.len().min(pos + 4)],
                &b1[pos..b1.len().min(pos + 4)]
            );
        }
    }
}

#[test]
#[ignore = "needs a real Gemma 3 checkpoint and the GPU"]
fn gemma3_two_requests_match_each_alone_on_the_paged_backend() {
    run(DecodeStorageBackend::Paged);
}

#[test]
#[ignore = "needs a real Gemma 3 checkpoint and the GPU"]
fn gemma3_two_requests_match_each_alone_on_the_dense_backend() {
    run(DecodeStorageBackend::Dense);
}
