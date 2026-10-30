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

//! A remote pipeline stage failure fails one request, not the coordinator
//! (issue #2727).
//!
//! Every test here runs a `PipelineServerModel` over a real
//! `RemotePipelineRuntime` (authenticated TCP, the real exchange, timeout
//! and cleanup paths) against a scripted stage, drives it through the batch
//! scheduler, and does so inside the model worker's fail-fast wrapper
//! `run_core_thread_or_abort`. The outcome the model worker would have is
//! therefore the outcome of the test: before the fix the forward panicked
//! inside the wrapper and the test process aborted; now the request is
//! answered with the client-safe error and the next request completes.
//!
//! Ports: the stage and the coordinator bind fixed loopback ports in the
//! 20300-20349 range, one pair per test.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use super::planned_prefill::PieceFailure;
use super::*;
use mlxcel_core::generate::SamplingConfig;

use crate::LoadedModel;
use crate::distributed::TransportBackend;
use crate::distributed::cluster_auth::{ClusterAuth, ClusterSecret};
use crate::distributed::pipeline::stub_stage::{RunFault, StubRunBehavior, StubStageHandle};
use crate::distributed::pipeline::{
    PipelineServerModel, RemotePipelineRuntime, RemotePipelineRuntimeConfig,
};
use crate::server::config::{DecodeStorageBackend, PreemptionPolicy, ReasoningBudgetOverride};
use crate::server::model_provider::GenerateEvent;
use crate::server::state::BatchMetrics;
use crate::tokenizer::MlxcelTokenizer;
use crate::worker_failfast::run_core_thread_or_abort;

const VOCAB: i32 = 8;
/// The id the stub's logits put on top; never the EOS, so a request ends
/// by its `max_tokens`.
const TOKEN: i32 = 3;
const EOS: i32 = 7;

/// The client-safe texts a pipeline failure must reach the request as.
const PIPELINE_FORWARD_FAILED: &str =
    "model pipeline forward failed; see the server log for the stage error";
const PIPELINE_PREPARE_REFUSED: &str =
    "model pipeline could not prepare this request; see the server log for the stage error";

/// The stage serves every forward except the `fault_on`-th, which `fault`s.
fn serving_stage(fault_on: u32, fault: RunFault) -> StubRunBehavior {
    StubRunBehavior::Logits {
        vocab: VOCAB,
        token: TOKEN,
        fault_on: Some(fault_on),
        fault,
    }
}

struct Harness {
    sched: BatchScheduler,
    stage: StubStageHandle,
}

/// A coordinator scheduler over one scripted stage: the stage on
/// `127.0.0.1:{stage_port}`, the coordinator's transport on
/// `127.0.0.1:{coordinator_port}`.
fn harness(
    stage_port: u16,
    coordinator_port: u16,
    behavior: StubRunBehavior,
    reject_prepare: bool,
    stage_timeout: Duration,
) -> Harness {
    let auth = Arc::new(ClusterAuth::new(&ClusterSecret::random()));
    let stage = StubStageHandle::spawn_at(
        &format!("127.0.0.1:{stage_port}"),
        behavior,
        reject_prepare,
        Arc::clone(&auth),
    )
    .expect("spawn stub stage");
    let runtime = RemotePipelineRuntime::new(RemotePipelineRuntimeConfig {
        stage_peers: vec![stage.addr().to_string()],
        transport_backend: TransportBackend::Tcp,
        bind_address: format!("127.0.0.1:{coordinator_port}"),
        stage_timeout,
        auth,
        vocab_size: Some(VOCAB as usize),
    })
    .expect("create remote pipeline runtime");
    let model = PipelineServerModel::new(1, vec![EOS], Box::new(runtime));
    let (_tx, rx) = mpsc::channel();
    let sched = BatchScheduler::with_config(
        LoadedModel::PipelineLlama(model),
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
        DecodeStorageBackend::Dense,
    );
    install_thread_local_default_stream(sched.generation_stream.as_ref());
    Harness { sched, stage }
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
    }
}

fn prompt() -> Vec<i32> {
    vec![1, 2, 3]
}

fn enqueue(sched: &mut BatchScheduler, max_tokens: usize) -> mpsc::Receiver<GenerateEvent> {
    let (tx, rx) = mpsc::channel();
    sched.enqueue_request(
        "prompt".to_string(),
        Some(prompt()),
        options(max_tokens),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        tx,
        Arc::new(AtomicBool::new(false)),
        true,
    );
    rx
}

/// What one request's event stream ended in.
#[derive(Debug, Default)]
struct Outcome {
    tokens: usize,
    error: Option<String>,
    done: bool,
}

/// Run scheduler ticks until the request leaves the scheduler, then read
/// its stream to the terminal event.
fn serve(sched: &mut BatchScheduler, rx: &mpsc::Receiver<GenerateEvent>) -> Outcome {
    for _ in 0..16 {
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
    let mut outcome = Outcome::default();
    loop {
        match rx.recv_timeout(Duration::from_secs(10)) {
            Ok(GenerateEvent::Token(_, _)) | Ok(GenerateEvent::TokenWithLogprobs(_, _, _)) => {
                outcome.tokens += 1;
            }
            Ok(GenerateEvent::Prefill(_)) | Ok(GenerateEvent::Admitted) => {}
            Ok(GenerateEvent::RequestMemoryRejected(error)) => {
                panic!("unexpected memory rejection: {error}")
            }
            Ok(GenerateEvent::Done(_)) => {
                outcome.done = true;
                return outcome;
            }
            Ok(GenerateEvent::Error(error)) => {
                outcome.error = Some(error);
                return outcome;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => return outcome,
            Err(err) => panic!("the request did not finish: {err}"),
        }
    }
}

/// The failure text a client may see: the fixed message, with no stage
/// address, remote sequence id or stage-side detail in it.
fn assert_client_safe(text: &str, stage: &StubStageHandle) {
    assert!(!text.contains(stage.addr()), "stage address leaked: {text}");
    assert!(!text.contains("127.0.0.1"), "an address leaked: {text}");
    assert!(
        !text.contains("pp-seq-"),
        "remote sequence id leaked: {text}"
    );
    assert!(!text.contains("stub stage"), "stage detail leaked: {text}");
    assert!(
        !text.contains("Server busy"),
        "a non-capacity failure labelled as capacity: {text}"
    );
}

/// A failed forward fails its request with the client-safe message and the
/// coordinator then serves the next request; a bystander sees nothing.
fn assert_fails_one_request_and_serves_the_next(mut harness: Harness, expected_commands: &[&str]) {
    let Harness { sched, stage } = &mut harness;
    let rx = enqueue(sched, 3);
    let first = serve(sched, &rx);
    let err = first
        .error
        .as_deref()
        .expect("the request whose forward failed is answered with an error");
    assert_eq!(err, PIPELINE_FORWARD_FAILED);
    assert_client_safe(err, stage);
    assert!(
        sched.active_batch.is_empty() && sched.prefill_queue.is_empty(),
        "the failed request left the scheduler"
    );

    let rx = enqueue(sched, 2);
    let second = serve(sched, &rx);
    assert!(
        second.done && second.error.is_none(),
        "the coordinator keeps serving after a stage failure: {second:?}"
    );
    assert_eq!(second.tokens, 2, "{second:?}");
    assert_eq!(stage.commands(), expected_commands);
    let Harness { stage, .. } = harness;
    stage.shutdown().expect("shutdown stub stage");
}

/// The key case of #2727 item 1: a stage answers one decode step with an
/// `Error`. Before the fix `PipelineServerModel::forward_with_sequence_id`
/// panicked on the model worker thread, which runs under
/// `run_core_thread_or_abort`, so the whole coordinator aborted. Now only
/// that request fails, with the fixed message, and the stage sees the
/// runtime cancel and release the failed sequence before the next one.
#[test]
fn remote_forward_error_fails_one_request_and_the_coordinator_serves_the_next() {
    run_core_thread_or_abort("model-worker", || {
        // Forward 1 is the first request's prefill, forward 2 its decode step.
        let coordinator = harness(
            20300,
            20301,
            serving_stage(2, RunFault::Error),
            false,
            Duration::from_secs(5),
        );
        assert_fails_one_request_and_serves_the_next(
            coordinator,
            &[
                "prepare",
                "run_entry",
                "run_entry",
                "cancel",
                "release",
                "prepare",
                "run_entry",
                "run_entry",
                "release",
            ],
        );
    });
}

/// The same, for the path that needs no cluster secret: a stage that never
/// answers (a dropped or corrupted frame closes the connection and the
/// exchange times out after `stage_timeout`).
#[test]
fn remote_forward_timeout_fails_one_request_and_the_coordinator_serves_the_next() {
    run_core_thread_or_abort("model-worker", || {
        let coordinator = harness(
            20302,
            20303,
            serving_stage(2, RunFault::Silent),
            false,
            Duration::from_millis(300),
        );
        assert_fails_one_request_and_serves_the_next(
            coordinator,
            &[
                "prepare",
                "run_entry",
                "run_entry",
                "cancel",
                "release",
                "prepare",
                "run_entry",
                "run_entry",
                "release",
            ],
        );
    });
}

/// A forward that fails during the prefill (the first forward of a request)
/// aborts the request through `PieceFailure::Abort` with the same message,
/// and the next request completes.
#[test]
fn remote_prefill_failure_fails_one_request_and_the_coordinator_serves_the_next() {
    run_core_thread_or_abort("model-worker", || {
        let Harness { mut sched, stage } = harness(
            20306,
            20307,
            serving_stage(1, RunFault::Error),
            false,
            Duration::from_secs(5),
        );
        let rx = enqueue(&mut sched, 3);
        let mut seq = sched
            .prefill_queue
            .dequeue()
            .expect("the request is queued for prefill");
        let plan = sched.prefill_plan_for(&seq);
        let piece = plan.pieces().first().cloned().expect("one piece");
        let failure = sched
            .run_prefill_piece(&mut seq, &plan, &piece, false)
            .err()
            .expect("the failed forward fails the piece");
        match &failure {
            PieceFailure::Abort(msg) => {
                assert_eq!(msg, PIPELINE_FORWARD_FAILED);
                assert_client_safe(msg, &stage);
            }
            other => panic!("expected PieceFailure::Abort, got {other:?}"),
        }
        sched.fail_prefill_piece(seq, failure);
        let first = serve(&mut sched, &rx);
        assert_eq!(first.error.as_deref(), Some(PIPELINE_FORWARD_FAILED));
        assert_eq!(first.tokens, 0);

        let rx = enqueue(&mut sched, 2);
        let second = serve(&mut sched, &rx);
        assert!(second.done, "{second:?}");
        assert_eq!(second.tokens, 2);
        assert_eq!(
            stage.commands(),
            [
                "prepare",
                "run_entry",
                "cancel",
                "release",
                "prepare",
                "run_entry",
                "run_entry",
                "release",
            ]
        );
        stage.shutdown().expect("shutdown stub stage");
    });
}

/// #2727 item 2: a stage that refuses the `PrepareSequence` fails the
/// request with the fixed prepare message, not with `Server busy` wrapped
/// around the stage's address and text.
#[test]
fn refused_prepare_reaches_the_request_as_the_client_safe_message() {
    run_core_thread_or_abort("model-worker", || {
        let Harness { mut sched, stage } = harness(
            20304,
            20305,
            serving_stage(u32::MAX, RunFault::Error),
            true,
            Duration::from_secs(5),
        );
        let rx = enqueue(&mut sched, 2);
        let first = serve(&mut sched, &rx);
        let err = first
            .error
            .as_deref()
            .expect("a refused prepare is answered with an error");
        assert!(
            err.starts_with("model pipeline could not prepare"),
            "unexpected error text: {err}"
        );
        assert_eq!(err, PIPELINE_PREPARE_REFUSED);
        assert_client_safe(err, &stage);
        assert!(sched.active_batch.is_empty() && sched.prefill_queue.is_empty());
        assert_eq!(
            stage.commands(),
            ["prepare", "release"],
            "the runtime rolls the refused sequence back on the stage"
        );
        stage.shutdown().expect("shutdown stub stage");
    });
}
