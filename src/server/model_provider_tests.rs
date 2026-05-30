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

use std::sync::mpsc;
use std::time::Duration;

use mlxcel_core::sampling::TokenLogprobData;

use super::{
    DECODE_HANG_TIMEOUT, GenerateEvent, GenerationResult, ModelProvider, ModelRequest,
    drain_generation_events, send_shutdown_signal, validated_decode_hang_timeout,
};

fn sample_result() -> GenerationResult {
    GenerationResult {
        text: "hello".to_string(),
        prompt_tokens: 3,
        completion_tokens: 2,
        generation_time_ms: 10,
        prompt_eval_ms: 4,
        generation_only_ms: 6,
        finish_reason: "stop".to_string(),
        logprobs: None,
        cached_tokens: 0,
    }
}

#[test]
fn drain_generation_events_forwards_tokens_before_done() {
    let (tx, rx) = mpsc::channel();
    tx.send(GenerateEvent::Token("A".to_string())).unwrap();
    tx.send(GenerateEvent::Token("B".to_string())).unwrap();
    tx.send(GenerateEvent::Done(sample_result())).unwrap();

    let mut streamed = Vec::new();
    let result =
        drain_generation_events(rx, DECODE_HANG_TIMEOUT, |token| streamed.push(token)).unwrap();

    assert_eq!(streamed, vec!["A".to_string(), "B".to_string()]);
    assert_eq!(result.text, "hello");
    assert_eq!(result.finish_reason, "stop");
}

#[test]
fn drain_generation_events_returns_worker_error() {
    let (tx, rx) = mpsc::channel();
    tx.send(GenerateEvent::Error("boom".to_string())).unwrap();

    let err = drain_generation_events(rx, DECODE_HANG_TIMEOUT, |_| {}).unwrap_err();
    assert!(err.to_string().contains("boom"));
}

#[test]
fn drain_generation_events_reports_closed_channel() {
    let (tx, rx) = mpsc::channel::<GenerateEvent>();
    drop(tx);
    let err = drain_generation_events(rx, DECODE_HANG_TIMEOUT, |_| {}).unwrap_err();
    assert!(err.to_string().contains("Response channel closed"));
}

#[test]
fn drain_generation_events_accumulates_logprobs_from_token_with_logprobs() {
    let (tx, rx) = mpsc::channel();
    let lp = TokenLogprobData {
        token_id: 42,
        logprob: -0.5,
        top_alternatives: vec![(7, -1.2)],
    };
    tx.send(GenerateEvent::TokenWithLogprobs("Hi".to_string(), lp))
        .unwrap();
    tx.send(GenerateEvent::Done(sample_result())).unwrap();

    let mut streamed = Vec::new();
    let result =
        drain_generation_events(rx, DECODE_HANG_TIMEOUT, |token| streamed.push(token)).unwrap();

    assert_eq!(streamed, vec!["Hi".to_string()]);
    let lp_data = result.logprobs.expect("logprobs should be Some");
    assert_eq!(lp_data.len(), 1);
    assert_eq!(lp_data[0].token_id, 42);
    assert!((lp_data[0].logprob - (-0.5)).abs() < 1e-6);
}

#[test]
fn send_shutdown_signal_enqueues_shutdown_request() {
    let (tx, rx) = mpsc::channel();
    assert!(send_shutdown_signal(&tx));
    assert!(matches!(rx.recv().unwrap(), ModelRequest::Shutdown));
}

#[test]
fn send_shutdown_signal_reports_closed_channel() {
    let (tx, rx) = mpsc::channel::<ModelRequest>();
    drop(rx);
    assert!(!send_shutdown_signal(&tx));
}

#[test]
fn model_provider_relies_on_auto_traits_for_shared_state() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ModelProvider>();
}

// ── validated_decode_hang_timeout tests ─────────────────────────

/// A non-zero timeout must be returned as-is without any warning.
#[test]
fn validated_decode_hang_timeout_returns_configured_value_for_nonzero() {
    let dur = validated_decode_hang_timeout(600);
    assert_eq!(dur, Duration::from_secs(600));
}

/// A timeout of 0 is invalid (would instantly expire every request). The
/// function must return the built-in fallback rather than 0.
#[test]
fn validated_decode_hang_timeout_uses_fallback_when_timeout_is_zero() {
    let dur = validated_decode_hang_timeout(0);
    // Must NOT be zero — that would instantly expire every request.
    assert_ne!(dur, Duration::ZERO);
    // Must be the documented fallback constant.
    assert_eq!(dur, DECODE_HANG_TIMEOUT);
}

// ── Long-prefill regression tests ───────────────────────────────

/// Verifies that `drain_generation_events_impl` (the private core loop used by
/// every drain function) does not apply any timeout during the prefill phase
/// (before the first token arrives). A prompt with 32k tokens may take tens of
/// seconds to prefill on real hardware; before a coarse timeout was
/// applied uniformly across both phases and would prematurely abort such
/// requests.
///
/// The test calls the production [`drain_generation_events_impl`] directly via
/// `pub(super)` visibility so a regression in either phase's recv strategy
/// surfaces here rather than silently bypassing this safety check.
#[test]
fn drain_generation_events_impl_survives_long_prefill_before_first_token() {
    use std::thread;

    use super::drain_generation_events_impl;

    // Create a std::sync::mpsc channel that mimics the model worker's
    // response_tx / response_rx pair.
    let (tx, rx) = mpsc::channel::<GenerateEvent>();

    // Simulate the model worker: hold for 80 ms (long prefill) then send the
    // first token followed by Done. The exact delay is well below any
    // reasonable Phase-2 bound but high enough that any Phase-1 timeout in
    // the production code would expire here.
    let worker = thread::spawn(move || {
        thread::sleep(Duration::from_millis(80));
        tx.send(GenerateEvent::Token("first".to_string()))
            .expect("send token");
        tx.send(GenerateEvent::Done(GenerationResult {
            text: "first".to_string(),
            prompt_tokens: 32768,
            completion_tokens: 1,
            generation_time_ms: 80,
            prompt_eval_ms: 79,
            generation_only_ms: 1,
            finish_reason: "stop".to_string(),
            logprobs: None,
            cached_tokens: 0,
        }))
        .expect("send done");
    });

    // Phase-2 timeout used here (5 s) is much larger than the synthetic
    // 80 ms prefill so the only thing under test is whether Phase 1 waits
    // without a timeout. If Phase 1 ever applied this 5 s as a recv_timeout
    // we would still pass — the meaningful regression to catch is a Phase-1
    // bound *shorter* than the 80 ms prefill, which is exactly what the
    // pre-issue- code did.
    let mut received_tokens: Vec<String> = Vec::new();
    let mut final_result: Option<GenerationResult> = None;

    let result = drain_generation_events_impl(&rx, Duration::from_secs(5), |event| match event {
        GenerateEvent::Token(t) => {
            received_tokens.push(t);
            Ok(None)
        }
        GenerateEvent::TokenWithLogprobs(t, _) => {
            received_tokens.push(t);
            Ok(None)
        }
        GenerateEvent::Done(r) => {
            final_result = Some(r.clone());
            Ok(Some(r))
        }
        GenerateEvent::Error(e) => Err(anyhow::anyhow!(e)),
    })
    .expect("drain_generation_events_impl must not error during long prefill");

    worker.join().expect("worker thread must not panic");

    assert_eq!(received_tokens, vec!["first"]);
    assert!(final_result.is_some(), "Done event must arrive");
    assert_eq!(result.prompt_tokens, 32768, "prompt_tokens must be 32768");
    assert_eq!(result.completion_tokens, 1);
    assert_eq!(result.finish_reason, "stop");
}

// ── Phase 2 decode hang regression test ─────────────────────────

/// Verifies that `drain_generation_events_impl` correctly detects a Phase 2
/// (decode phase) hang: once the first token has arrived, any subsequent wait
/// that exceeds `decode_hang_timeout` must return an error rather than blocking
/// forever.
///
/// The test sends a `Token` event (entering Phase 2), then withholds further
/// events beyond the timeout. The function must return an `Err` whose message
/// mentions "decode" or "hang" or the timeout duration.
///
/// Regression guard: if `recv_timeout` in the Phase 2 branch is ever replaced
/// with an unconstrained `recv()`, this test will hang the test runner rather
/// than pass — making the regression impossible to miss.
#[test]
fn drain_generation_events_impl_detects_phase2_decode_hang() {
    use std::thread;

    use super::drain_generation_events_impl;

    // Use a very short timeout so the test completes in well under a second.
    let decode_hang_timeout = Duration::from_millis(150);

    let (tx, rx) = mpsc::channel::<GenerateEvent>();

    // Send one Token event to transition into Phase 2, then drop the sender
    // so no further events will arrive — simulating a worker that goes silent
    // after producing the first token.
    let worker = thread::spawn(move || {
        tx.send(GenerateEvent::Token("first".to_string()))
            .expect("send first token");
        // Hold sender alive briefly so Phase 2 enters recv_timeout, then drop
        // it to also trigger Disconnected — whichever fires first is fine.
        thread::sleep(Duration::from_millis(200));
        // sender dropped here — channel disconnects if timeout didn't fire yet
    });

    let mut received_tokens: Vec<String> = Vec::new();

    let result = drain_generation_events_impl(&rx, decode_hang_timeout, |event| match event {
        GenerateEvent::Token(t) => {
            received_tokens.push(t);
            Ok(None)
        }
        GenerateEvent::TokenWithLogprobs(t, _) => {
            received_tokens.push(t);
            Ok(None)
        }
        GenerateEvent::Done(r) => Ok(Some(r)),
        GenerateEvent::Error(e) => Err(anyhow::anyhow!(e)),
    });

    worker.join().expect("worker thread must not panic");

    // The first token must have been delivered before the timeout triggered.
    assert_eq!(
        received_tokens,
        vec!["first"],
        "first token must be forwarded before hang is detected"
    );

    // Phase 2 must have timed out and returned an error.
    let err = result.expect_err("Phase 2 hang must produce an Err");
    let msg = err.to_string();
    assert!(
        msg.contains("hang") || msg.contains("decode") || msg.contains("150"),
        "error message must describe the decode hang; got: {msg}"
    );
}
