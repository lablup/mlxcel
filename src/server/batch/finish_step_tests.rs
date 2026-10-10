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

//! One finish step for every server decode site class (#2168).
//!
//! Each site class drives the same token streams through the finish entry
//! point it calls in production, configured the way the site configures it:
//!
//! - batched per-row decode (`execute_batched_decode`) and single-step decode
//!   (`decode_single_step`): [`finish_decode_token`] on a decoding sequence,
//!   with the token's logprobs;
//! - fused batched decode (`apply_fused_decode_tokens`): the same entry point
//!   without logprobs or a structured stop;
//! - prefill completion (`finish_prefill`): the first token on a prefilling
//!   sequence, then decode;
//! - the speculative burst stream: [`stream_burst_tokens`] and
//!   [`finalize_burst_stream`], fed in slices.
//!
//! Every class must report the same [`FinishCause`] after the same number of
//! generated tokens, and the same finish reason on the wire, for EOS, a stop
//! string, a generation bound, `max_tokens`, the context bound and a
//! repetition loop. `mlxcel-core`'s `generate_finish_tests.rs` feeds the CLI
//! loop class the same streams.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::time::Instant;

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::SamplingConfig;
use mlxcel_core::sampling::TokenLogprobData;
use mlxcel_core::{FinishCause, LoopDetectionConfig};

use super::RequestPriority;
use super::finish::{ContextBound, finish_decode_token};
use super::generation_bounds::GenerationBounds;
use super::sequence::{FinishReason, SequenceInfo, SequenceState};
use super::speculative_burst::{begin_burst_stream, finalize_burst_stream, stream_burst_tokens};
use super::stop_matcher::StopMatcher;
use crate::server::model_provider::model_worker::StreamingDecodeState;
use crate::server::model_provider::{GenerateEvent, StopKind};
use crate::tokenizer::MlxcelTokenizer;

const EOS: i32 = 1;

fn bytes(s: &str) -> Vec<i32> {
    s.bytes().map(i32::from).collect()
}

/// One request shape: the stream, the request's limits, and what every site
/// class must report for it.
struct Scenario {
    name: &'static str,
    tokens: Vec<i32>,
    stops: &'static [&'static str],
    n_indent: usize,
    max_tokens: usize,
    prompt_len: usize,
    context: ContextBound,
    loop_detection: LoopDetectionConfig,
    cause: FinishCause,
    generated: usize,
}

fn scenario(
    name: &'static str,
    tokens: Vec<i32>,
    cause: FinishCause,
    generated: usize,
) -> Scenario {
    Scenario {
        name,
        tokens,
        stops: &[],
        n_indent: 0,
        max_tokens: 16,
        prompt_len: 4,
        context: ContextBound::default(),
        loop_detection: LoopDetectionConfig::disabled(),
        cause,
        generated,
    }
}

fn scenarios() -> Vec<Scenario> {
    let mut eos = bytes("ab");
    eos.extend([EOS, 99]);
    vec![
        scenario("eos", eos, FinishCause::Eos, 2),
        Scenario {
            stops: &["STOP"],
            ..scenario(
                "stop string",
                bytes("xySTOPz"),
                FinishCause::StopSequence,
                6,
            )
        },
        // `n_indent: 1` fires on the second character of the unindented
        // line after the first newline (b10621's indentation rule, #1477).
        Scenario {
            n_indent: 1,
            ..scenario("generation bound", bytes("x\nyzw"), FinishCause::Length, 4)
        },
        Scenario {
            max_tokens: 3,
            ..scenario("max_tokens", bytes("abcdef"), FinishCause::Length, 3)
        },
        // A 4-token prompt in an 8-token window: after 3 generated tokens the
        // next one would not fit (4 + 3 + 1 >= 8).
        Scenario {
            context: ContextBound {
                max_kv_size: Some(8),
                context_shift: false,
            },
            ..scenario(
                "context bound",
                bytes("abcdef"),
                FinishCause::ContextExhausted,
                3,
            )
        },
        Scenario {
            loop_detection: LoopDetectionConfig::new(1, 4, 3),
            ..scenario(
                "repetition loop",
                bytes("abababab"),
                FinishCause::RepetitionLoop,
                6,
            )
        },
    ]
}

/// What a site class reported for a scenario.
#[derive(Debug, PartialEq)]
struct Outcome {
    cause: Option<FinishCause>,
    generated: Vec<i32>,
    finish_reason: String,
    stop_kind: StopKind,
}

fn make_sequence(
    tokenizer: &MlxcelTokenizer,
    s: &Scenario,
    state: SequenceState,
) -> (SequenceInfo, mpsc::Receiver<GenerateEvent>) {
    let (tx, rx) = mpsc::channel();
    let prompt_tokens: Vec<i32> = vec![0; s.prompt_len];
    let decode_state = StreamingDecodeState::new(tokenizer, &prompt_tokens);
    let mut sampling = SamplingConfig::greedy();
    sampling.loop_detection = s.loop_detection;
    let seq = SequenceInfo {
        bounds: GenerationBounds::new(s.n_indent, None),
        retention: Default::default(),
        seq_id: SequenceId::from_raw(2168),
        state,
        prompt_tokens,
        sampling,
        max_tokens: s.max_tokens,
        eos_token_ids: Vec::new(),
        priority: RequestPriority::Normal,
        lora_scales: None,
        logprobs_config: Default::default(),
        vlm_embeddings: None,
        images: Vec::new(),
        audio: Vec::new(),
        generated_tokens: Vec::new(),
        generated_text: String::new(),
        decode_state,
        stop_matcher: StopMatcher::new(s.stops.iter().map(|s| s.to_string())),
        prefill_offset: 0,
        prefill_start_offset: 0,
        already_cached_tokens: 0,
        eos_terminated: false,
        response_tx: tx,
        cancelled: Arc::new(AtomicBool::new(false)),
        created_at: Instant::now(),
        prefill_start: Some(Instant::now()),
        first_token_time: None,
        prompt_lookup: None,
        token_history: Vec::new(),
        sampler: Default::default(),
        merged_eos: vec![EOS],
        thinking: crate::server::thinking_budget::ThinkingState::disabled(),
        structured: None,
    };
    (seq, rx)
}

/// The reason a finished sequence recorded.
fn finished_as(seq: &SequenceInfo) -> Option<&FinishReason> {
    match &seq.state {
        SequenceState::Finished(reason) => Some(reason),
        _ => None,
    }
}

fn logprob(token: i32) -> Option<TokenLogprobData> {
    Some(TokenLogprobData {
        token_id: token,
        logprob: -0.5,
        top_alternatives: Vec::new(),
    })
}

/// Close the sequence the way `finalize_completed` does and read the wire
/// finish reason.
fn finish_classic(
    tokenizer: &MlxcelTokenizer,
    mut seq: SequenceInfo,
    cause: Option<FinishCause>,
) -> Outcome {
    let tail = seq.decode_state.flush(tokenizer);
    seq.close_text_stream(tail);
    let generated = seq.generated_tokens.clone();
    let result = seq.take_generation_result(tokenizer, 0, None);
    Outcome {
        cause,
        generated,
        finish_reason: result.finish_reason,
        stop_kind: result.stop_kind,
    }
}

/// `execute_batched_decode`'s per-row loop and `decode_single_step`: a
/// decoding sequence, the token's logprobs, the structured-stop flag.
fn drive_classic_decode(s: &Scenario) -> Outcome {
    let tokenizer = MlxcelTokenizer::stub_all_byte_fallback();
    let (mut seq, _rx) = make_sequence(&tokenizer, s, SequenceState::Decoding);
    let mut cause = None;
    for &token in &s.tokens {
        cause = finish_decode_token(
            &mut seq,
            &tokenizer,
            token,
            logprob(token),
            false,
            s.context,
        );
        if cause.is_some() {
            break;
        }
    }
    finish_classic(&tokenizer, seq, cause)
}

/// `apply_fused_decode_tokens`: no logprobs and no structured output.
fn drive_fused_decode(s: &Scenario) -> Outcome {
    let tokenizer = MlxcelTokenizer::stub_all_byte_fallback();
    let (mut seq, _rx) = make_sequence(&tokenizer, s, SequenceState::Decoding);
    let mut cause = None;
    for &token in &s.tokens {
        cause = finish_decode_token(&mut seq, &tokenizer, token, None, false, s.context);
        if cause.is_some() {
            break;
        }
    }
    finish_classic(&tokenizer, seq, cause)
}

/// `finish_prefill` on the first token, then decode for the rest.
fn drive_prefill_then_decode(s: &Scenario) -> Outcome {
    let tokenizer = MlxcelTokenizer::stub_all_byte_fallback();
    let (mut seq, _rx) = make_sequence(&tokenizer, s, SequenceState::Prefilling);
    let (first, rest) = s.tokens.split_first().expect("non-empty stream");
    let mut cause = finish_decode_token(
        &mut seq,
        &tokenizer,
        *first,
        logprob(*first),
        false,
        s.context,
    );
    if cause.is_none() {
        seq.state
            .transition_to(SequenceState::Decoding)
            .expect("prefill hands an unfinished sequence to decode");
        for &token in rest {
            cause = finish_decode_token(
                &mut seq,
                &tokenizer,
                token,
                logprob(token),
                false,
                s.context,
            );
            if cause.is_some() {
                break;
            }
        }
    }
    finish_classic(&tokenizer, seq, cause)
}

/// The speculative burst stream, fed `slice` tokens per call as the
/// tick-cooperative slice driver does.
fn drive_burst(s: &Scenario, slice: usize) -> Outcome {
    let tokenizer = MlxcelTokenizer::stub_all_byte_fallback();
    let (mut seq, rx) = make_sequence(&tokenizer, s, SequenceState::Prefilling);
    // The burst resolves its own merged EOS set; the classic `merged_eos` is
    // empty on this path.
    seq.merged_eos.clear();
    let mut stream = begin_burst_stream(vec![EOS], &seq, s.context);
    for chunk in s.tokens.chunks(slice) {
        if stream_burst_tokens(&tokenizer, &mut seq, &mut stream, chunk, &[]) {
            break;
        }
    }
    let cause = stream.finish();
    let generated = seq.generated_tokens.clone();
    let _ = finalize_burst_stream(&tokenizer, seq, &stream, None);
    let result = rx
        .try_iter()
        .find_map(|event| match event {
            GenerateEvent::Done(result) => Some(result),
            _ => None,
        })
        .expect("the burst finalize sends one Done event");
    Outcome {
        cause,
        generated,
        finish_reason: result.finish_reason,
        stop_kind: result.stop_kind,
    }
}

fn assert_class(class: &str, drive: impl Fn(&Scenario) -> Outcome) {
    for s in scenarios() {
        let outcome = drive(&s);
        assert_eq!(
            outcome.cause,
            Some(s.cause),
            "{class} / {}: finish cause",
            s.name
        );
        assert_eq!(
            outcome.generated,
            s.tokens[..s.generated].to_vec(),
            "{class} / {}: generated tokens",
            s.name
        );
        // Every class reports the classic per-row loop's wire outcome.
        let reference = drive_classic_decode(&s);
        assert_eq!(
            (outcome.finish_reason, outcome.stop_kind),
            (reference.finish_reason, reference.stop_kind),
            "{class} / {}: wire finish reason",
            s.name
        );
    }
}

#[test]
fn batched_per_row_decode_finishes_every_scenario_by_the_shared_step() {
    assert_class("batched per-row", drive_classic_decode);
}

#[test]
fn single_step_decode_finishes_every_scenario_by_the_shared_step() {
    assert_class("single-step", drive_classic_decode);
}

#[test]
fn fused_batched_decode_finishes_every_scenario_by_the_shared_step() {
    assert_class("fused", drive_fused_decode);
}

#[test]
fn prefill_completion_finishes_every_scenario_by_the_shared_step() {
    assert_class("prefill", drive_prefill_then_decode);
}

#[test]
fn burst_stream_finishes_every_scenario_by_the_shared_step() {
    assert_class("burst, one call", |s| drive_burst(s, usize::MAX));
    assert_class("burst, 3-token slices", |s| drive_burst(s, 3));
}

#[test]
fn the_cause_maps_onto_the_reference_finish_reasons() {
    let expect = |name: &str, reason: FinishReason| {
        let s = scenarios()
            .into_iter()
            .find(|s| s.name == name)
            .expect("known scenario");
        let tokenizer = MlxcelTokenizer::stub_all_byte_fallback();
        let (mut seq, _rx) = make_sequence(&tokenizer, &s, SequenceState::Decoding);
        for &token in &s.tokens {
            if finish_decode_token(&mut seq, &tokenizer, token, None, false, s.context).is_some() {
                break;
            }
        }
        assert_eq!(finished_as(&seq), Some(&reason), "{name}");
        seq
    };
    let seq = expect("eos", FinishReason::Stop);
    assert!(seq.eos_terminated, "an EOS stop holds every pushed token");
    expect("stop string", FinishReason::StopSequence);
    expect("generation bound", FinishReason::Length);
    let seq = expect("max_tokens", FinishReason::Length);
    assert!(!seq.retention.context_exhausted);
    let seq = expect("context bound", FinishReason::Length);
    assert!(
        seq.retention.context_exhausted,
        "a context stop is truncated"
    );
    expect("repetition loop", FinishReason::RepetitionLoop);
}

#[test]
fn prefill_eos_on_the_first_token_is_not_pushed() {
    let mut s = scenario("first-token eos", vec![EOS, 97], FinishCause::Eos, 0);
    s.max_tokens = 1;
    let tokenizer = MlxcelTokenizer::stub_all_byte_fallback();
    let (mut seq, _rx) = make_sequence(&tokenizer, &s, SequenceState::Prefilling);
    let cause = finish_decode_token(&mut seq, &tokenizer, EOS, None, false, s.context);
    assert_eq!(cause, Some(FinishCause::Eos));
    assert!(seq.generated_tokens.is_empty());
    assert_eq!(finished_as(&seq), Some(&FinishReason::Stop));
}

#[test]
fn prefill_with_max_tokens_one_finishes_length_on_the_first_token() {
    let mut s = scenario("max_tokens 1", bytes("ab"), FinishCause::Length, 1);
    s.max_tokens = 1;
    let tokenizer = MlxcelTokenizer::stub_all_byte_fallback();
    let (mut seq, _rx) = make_sequence(&tokenizer, &s, SequenceState::Prefilling);
    let cause = finish_decode_token(&mut seq, &tokenizer, 97, None, false, s.context);
    assert_eq!(cause, Some(FinishCause::Length));
    assert_eq!(seq.generated_tokens, vec![97]);
    assert_eq!(finished_as(&seq), Some(&FinishReason::Length));
}

#[test]
fn a_structured_stop_reports_stop_below_a_stop_string() {
    let s = Scenario {
        stops: &["a"],
        ..scenario("structured", bytes("ab"), FinishCause::StructuredStop, 1)
    };
    let tokenizer = MlxcelTokenizer::stub_all_byte_fallback();
    // Structured stop alone: `Stop`.
    let (mut seq, _rx) = make_sequence(
        &tokenizer,
        &scenario("s", bytes("b"), FinishCause::StructuredStop, 1),
        SequenceState::Decoding,
    );
    assert_eq!(
        finish_decode_token(&mut seq, &tokenizer, 98, None, true, s.context),
        Some(FinishCause::StructuredStop)
    );
    assert_eq!(finished_as(&seq), Some(&FinishReason::Stop));
    // A stop string on the same token wins.
    let (mut seq, _rx) = make_sequence(&tokenizer, &s, SequenceState::Decoding);
    assert_eq!(
        finish_decode_token(&mut seq, &tokenizer, 97, None, true, s.context),
        Some(FinishCause::StopSequence)
    );
}

/// Feed `tokens` to the burst stream as one generator batch (the
/// run-to-completion arm, or one MTP round) and finalize. Returns the outcome
/// and whether the stream ended and may donate its state to the prompt cache.
struct BurstBatch {
    outcome: Outcome,
    done: bool,
    healthy: bool,
}

fn burst_batch(s: &Scenario, tokens: &[i32]) -> BurstBatch {
    let tokenizer = MlxcelTokenizer::stub_all_byte_fallback();
    let (mut seq, rx) = make_sequence(&tokenizer, s, SequenceState::Prefilling);
    seq.merged_eos.clear();
    let mut stream = begin_burst_stream(vec![EOS], &seq, s.context);
    let done = stream_burst_tokens(&tokenizer, &mut seq, &mut stream, tokens, &[]);
    let cause = stream.finish();
    let generated = seq.generated_tokens.clone();
    let healthy = finalize_burst_stream(&tokenizer, seq, &stream, None).healthy_finish;
    let result = rx
        .try_iter()
        .find_map(|event| match event {
            GenerateEvent::Done(result) => Some(result),
            _ => None,
        })
        .expect("the burst finalize sends one Done event");
    let outcome = Outcome {
        cause,
        generated,
        finish_reason: result.finish_reason,
        stop_kind: result.stop_kind,
    };
    BurstBatch {
        outcome,
        done,
        healthy,
    }
}

fn named(name: &str) -> Scenario {
    scenarios()
        .into_iter()
        .find(|s| s.name == name)
        .expect("known scenario")
}

/// The burst stream had no repetition guard before #2168: a looping MTP or
/// DFlash request streamed every token and finished `stop`, while the same
/// request on classic decode stopped early with `RepetitionLoop`.
#[test]
fn a_repeating_burst_stream_finishes_on_the_repetition_loop() {
    let s = named("repetition loop");
    let BurstBatch {
        outcome,
        done,
        healthy,
    } = burst_batch(&s, &s.tokens);
    assert!(done, "the loop terminates the stream");
    assert_eq!(outcome.cause, Some(FinishCause::RepetitionLoop));
    assert_eq!(outcome.generated, bytes("ababab"));
    assert!(
        !healthy,
        "the generator already produced the trailing tokens, so the model \
         state is ahead of the committed stream and must not be donated"
    );
    let reference = drive_classic_decode(&s);
    assert_eq!(outcome.finish_reason, reference.finish_reason);
    assert_eq!(outcome.stop_kind, reference.stop_kind);
}

/// A loop that fires on the last token of the batch leaves the model state
/// in step with the committed stream, so it donates like classic decode.
#[test]
fn a_burst_loop_on_the_last_batch_token_donates() {
    let s = named("repetition loop");
    let BurstBatch {
        outcome, healthy, ..
    } = burst_batch(&s, &bytes("ababab"));
    assert_eq!(outcome.cause, Some(FinishCause::RepetitionLoop));
    assert_eq!(outcome.generated, bytes("ababab"));
    assert!(healthy, "a loop finish at the batch end donates");
    assert_eq!(
        outcome.finish_reason,
        drive_classic_decode(&s).finish_reason
    );
}

/// A stop string completing mid-batch leaves `z` in the model state but out
/// of the committed stream: the finish is a stop sequence that never donates.
#[test]
fn a_mid_batch_burst_stop_string_does_not_donate() {
    let s = named("stop string");
    let BurstBatch {
        outcome, healthy, ..
    } = burst_batch(&s, &s.tokens);
    assert_eq!(outcome.cause, Some(FinishCause::StopSequence));
    // Classic decode records `FinishReason::StopSequence` for this stream
    // (`the_cause_maps_onto_the_reference_finish_reasons`).
    let reference = drive_classic_decode(&s);
    assert_eq!(
        (outcome.finish_reason, outcome.stop_kind),
        (reference.finish_reason, reference.stop_kind)
    );
    assert!(!healthy, "the uncommitted `z` is in the model state");
}

/// `max_tokens` reached exactly at the batch end commits every produced
/// token, so the finish stays healthy.
#[test]
fn a_burst_finishing_on_max_tokens_at_the_batch_end_donates() {
    let s = named("max_tokens");
    let BurstBatch {
        outcome, healthy, ..
    } = burst_batch(&s, &s.tokens[..3]);
    assert_eq!(outcome.cause, Some(FinishCause::Length));
    assert_eq!(outcome.generated, bytes("abc"));
    assert!(healthy, "no produced token is left uncommitted");
}
