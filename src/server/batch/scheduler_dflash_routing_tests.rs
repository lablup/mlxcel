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

//! Prefill routing of queued speculative-eligible requests (#2442).
//!
//! With the default `--max-batch-prefill`, two or more queued requests used
//! to take the classic batched prefill, which never reaches the speculative
//! burst. These tests drive one prefill tick of a scheduler over the
//! two-layer synthetic Qwen 3.5 model with two equal-length queued requests.
//! The DFlash dispatch points at a drafter that does not exist, so a burst
//! that reaches the drafter load fails its rows with that load error: a
//! batched window fails both rows in the tick, a B = 1 burst only the head.
//! That is how the tests see which burst ran without a real drafter. The
//! block-versus-chain verdict is seeded per test (each at its own block
//! width, the verdict cache being process-wide).

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use mlxcel_core::generate::SamplingConfig;

use super::*;
use crate::server::batch::speculative_burst_tests::{
    missing_drafter_dflash_dispatch, seed_qwen35_exactness_verdict,
};
use crate::server::config::{
    DecodeStorageBackend, PreemptionPolicy, ReasoningBudgetOverride, ServerConfig,
};
use crate::server::model_provider::GenerateEvent;
use crate::server::state::BatchMetrics;
use crate::tokenizer::MlxcelTokenizer;

/// A scheduler with the server's default `--max-batch-prefill` (4) and room
/// for a four-row active batch.
fn scheduler(
    model: crate::models::Qwen35Model,
    dispatch: Option<crate::server::SpeculativeDispatch>,
) -> BatchScheduler {
    let max_batch_prefill = ServerConfig::default().max_batch_prefill;
    assert!(
        max_batch_prefill > 1,
        "the default must allow batched prefill"
    );
    let (_tx, rx) = mpsc::channel();
    let mut sched = BatchScheduler::with_config(
        LoadedModel::Qwen35(model),
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
        max_batch_prefill,
        DecodeStorageBackend::Dense,
    );
    if let Some(dispatch) = dispatch {
        sched = sched.with_speculative_dispatch(dispatch);
    }
    install_thread_local_default_stream(sched.generation_stream.as_ref());
    sched
}

fn enqueue(sched: &mut BatchScheduler, prompt: &[i32]) -> mpsc::Receiver<GenerateEvent> {
    enqueue_with(sched, prompt, SamplingConfig::greedy())
}

fn enqueue_with(
    sched: &mut BatchScheduler,
    prompt: &[i32],
    sampling: SamplingConfig,
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
        max_tokens: 4,
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

/// Enqueue two equal-length requests and run the one prefill tick the run
/// loop would run for them.
fn prefill_two_queued(sched: &mut BatchScheduler) -> [mpsc::Receiver<GenerateEvent>; 2] {
    let first = enqueue(sched, &[1, 2, 3]);
    let second = enqueue(sched, &[3, 2, 1]);
    assert_eq!(sched.prefill_queue.len(), 2);
    let BatchSchedulerAction::Prefill(id) = sched.decide_action() else {
        panic!("two queued requests must schedule a prefill");
    };
    sched.execute_prefill_action(id);
    [first, second]
}

fn receive_error(rx: &mpsc::Receiver<GenerateEvent>) -> String {
    match rx.recv_timeout(Duration::from_secs(2)) {
        Ok(GenerateEvent::Error(error)) => error,
        Ok(GenerateEvent::Token(text, _)) => panic!("unexpected token event: {text:?}"),
        Ok(GenerateEvent::TokenWithLogprobs(text, _, _)) => {
            panic!("unexpected token+logprobs event: {text:?}")
        }
        Ok(GenerateEvent::Done(_)) => panic!("unexpected done event"),
        Ok(GenerateEvent::Prefill(_)) => panic!("unexpected prefill event"),
        Ok(GenerateEvent::Admitted) => panic!("unexpected admission event"),
        Ok(GenerateEvent::RequestMemoryRejected(error)) => {
            panic!("unexpected memory rejection: {error}")
        }
        Err(err) => panic!("the burst did not finalize the row: {err}"),
    }
}

/// The issue's acceptance test: DFlash dispatch, default
/// `--max-batch-prefill`, two equal-length queued requests. Both rows reach the batched burst as one window in one
/// tick (its drafter load is what fails them), instead of the classic
/// batched prefill admitting both into the active batch with no speculative
/// decoding.
#[test]
fn queued_equal_length_dflash_requests_reach_the_batched_burst_as_one_window() {
    let _runtime = crate::initialize_runtime();
    let block_size = 11;
    let qwen = crate::models::qwen3_5_tests::two_layer_model();
    seed_qwen35_exactness_verdict(&qwen, block_size, true);
    let mut sched = scheduler(qwen, Some(missing_drafter_dflash_dispatch(block_size)));

    let rows = prefill_two_queued(&mut sched);

    assert!(
        sched.active_batch.is_empty(),
        "no row may take the classic prefill: {} active",
        sched.active_batch.len()
    );
    assert!(
        sched.prefill_queue.is_empty(),
        "both rows must ride one window, not a B = 1 burst for the head"
    );
    for rx in &rows {
        let error = receive_error(rx);
        assert!(
            error.contains("Drafter load failed"),
            "the batched burst's drafter load must be what ends the row: {error}"
        );
    }
}

/// A head the burst would only decline keeps the batched prefill: with a
/// failing exactness verdict both rows are admitted to the active batch in
/// one tick, exactly as without a drafter, and the drafter is never touched.
#[test]
fn queued_dflash_requests_keep_batched_prefill_when_the_burst_would_decline() {
    let _runtime = crate::initialize_runtime();
    let block_size = 9;
    let qwen = crate::models::qwen3_5_tests::two_layer_model();
    seed_qwen35_exactness_verdict(&qwen, block_size, false);
    let mut sched = scheduler(qwen, Some(missing_drafter_dflash_dispatch(block_size)));

    let rows = prefill_two_queued(&mut sched);

    assert!(sched.prefill_queue.is_empty());
    assert_eq!(sched.active_batch.len(), 2);
    for rx in &rows {
        assert!(
            !matches!(rx.try_recv(), Ok(GenerateEvent::Error(_))),
            "a declined burst must not fail the row"
        );
    }
}

/// A server without a drafter is unchanged: two queued requests take the
/// batched prefill into the active batch in one tick.
#[test]
fn queued_requests_without_speculative_dispatch_keep_batched_prefill() {
    let _runtime = crate::initialize_runtime();
    let qwen = crate::models::qwen3_5_tests::two_layer_model();
    let mut sched = scheduler(qwen, None);

    let rows = prefill_two_queued(&mut sched);

    assert!(sched.prefill_queue.is_empty());
    assert_eq!(sched.active_batch.len(), 2);
    for rx in &rows {
        assert!(!matches!(rx.try_recv(), Ok(GenerateEvent::Error(_))));
    }
}

/// While classic rows are decoding, queued burst-eligible requests keep the
/// batched prefill: a run-to-completion burst per tick would hold the active
/// rows' decode off for as long as the queue stays non-empty.
#[test]
fn queued_dflash_requests_keep_batched_prefill_while_classic_rows_decode() {
    let _runtime = crate::initialize_runtime();
    let block_size = 5;
    let qwen = crate::models::qwen3_5_tests::two_layer_model();
    seed_qwen35_exactness_verdict(&qwen, block_size, true);
    let mut sched = scheduler(qwen, Some(missing_drafter_dflash_dispatch(block_size)));

    // Mirostat is a hard burst decline, so this request decodes classically.
    let classic = enqueue_with(
        &mut sched,
        &[2, 4, 6],
        SamplingConfig {
            mirostat: 2,
            ..SamplingConfig::greedy()
        },
    );
    let BatchSchedulerAction::Prefill(id) = sched.decide_action() else {
        panic!("one queued request must schedule a prefill");
    };
    sched.execute_prefill_action(id);
    assert_eq!(
        sched.active_batch.len(),
        1,
        "the mirostat row decodes classically"
    );

    let rows = prefill_two_queued(&mut sched);

    assert!(sched.prefill_queue.is_empty());
    assert_eq!(sched.active_batch.len(), 3);
    for rx in rows.iter().chain(std::iter::once(&classic)) {
        assert!(!matches!(rx.try_recv(), Ok(GenerateEvent::Error(_))));
    }
}
