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

//! String stop sequences on the MLX serving path (issue #1466).
//!
//! Before this, `ServerGenerateOptions::stop_sequences` was parsed, carried into
//! the scheduler, and never read: a request that asked to stop on `"5"` ran to
//! `max_tokens` and the stop string appeared in the output. These tests drive
//! the funnel every MLX decode site now goes through
//! ([`SequenceInfo::stream_decoded_text`], [`SequenceInfo::close_text_stream`],
//! [`SequenceInfo::take_generation_result`]) with a real
//! [`StreamingDecodeState`], asserting the four properties the issue's
//! acceptance criteria name:
//!
//! 1. the non-streaming text is truncated at the match and excludes it,
//! 2. the concatenation of streamed chunks equals that text at every token
//!    boundary, so a partial match is never leaked,
//! 3. the matched stop string reaches the response layer, and
//! 4. the finish reason separates a stop-string match from EOS and from length.
//!
//! The decode stream is driven a few UTF-8 bytes per token through the
//! all-byte-fallback stub tokenizer, which is a harsher chunking than a real
//! model produces and the one where a naive matcher leaks a partial match.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc;
use std::time::Instant;

use mlxcel_core::cache::SequenceId;
use mlxcel_core::generate::SamplingConfig;

use super::RequestPriority;
use super::sequence::{FinishReason, SequenceInfo, SequenceState};
use super::stop_matcher::StopMatcher;
use crate::server::model_provider::model_worker::StreamingDecodeState;
use crate::server::model_provider::{GenerateEvent, GenerationResult, StopKind};
use crate::tokenizer::MlxcelTokenizer;

/// One decoded run: what the client received, the finished result, and the
/// terminal state the scheduler would have recorded.
struct Run {
    streamed: String,
    result: GenerationResult,
    finished_as: SequenceState,
}

/// The `<0xXX>` token ids for the UTF-8 bytes of `s` (token id == byte value in
/// [`MlxcelTokenizer::stub_all_byte_fallback`]).
fn byte_ids(s: &str) -> Vec<i32> {
    s.bytes().map(|b| b as i32).collect()
}

fn make_sequence(
    tokenizer: &MlxcelTokenizer,
    stops: &[&str],
    max_tokens: usize,
) -> (SequenceInfo, mpsc::Receiver<GenerateEvent>) {
    let (tx, rx) = mpsc::channel();
    let prompt_tokens: Vec<i32> = Vec::new();
    let decode_state = StreamingDecodeState::new(tokenizer, &prompt_tokens);
    let seq = SequenceInfo {
        retention: Default::default(),
        seq_id: SequenceId::from_raw(1),
        state: SequenceState::Decoding,
        prompt_tokens,
        sampling: SamplingConfig::default(),
        max_tokens,
        eos_token_ids: Vec::new(),
        priority: RequestPriority::Normal,
        logprobs_config: Default::default(),
        vlm_embeddings: None,
        images: Vec::new(),
        audio: Vec::new(),
        generated_tokens: Vec::new(),
        generated_text: String::new(),
        decode_state,
        stop_matcher: StopMatcher::new(stops.iter().map(|s| s.to_string())),
        prefill_offset: 0,
        prefill_start_offset: 0,
        already_cached_tokens: 0,
        response_tx: tx,
        cancelled: Arc::new(AtomicBool::new(false)),
        created_at: Instant::now(),
        prefill_start: None,
        first_token_time: None,
        token_history: Vec::new(),
        sampler_state: None,
        merged_eos: Vec::new(),
        thinking: crate::server::thinking_budget::ThinkingState::disabled(),
        structured: None,
    };
    (seq, rx)
}

/// Drive `text` through the decode funnel `chunk_bytes` tokens at a time,
/// mirroring what `decode_single_step` does per sampled token, then finish the
/// sequence the way `finalize_completed` does.
fn drive(text: &str, stops: &[&str], max_tokens: usize, chunk_bytes: usize) -> Run {
    let tokenizer = MlxcelTokenizer::stub_all_byte_fallback();
    let (mut seq, rx) = make_sequence(&tokenizer, stops, max_tokens);

    for ids in byte_ids(text).chunks(chunk_bytes.max(1)) {
        if seq.state.is_finished() {
            break;
        }
        let mut piece: Option<String> = None;
        for &id in ids {
            seq.generated_tokens.push(id);
            if let Some(t) = seq.decode_state.on_token(id, &tokenizer) {
                piece.get_or_insert_with(String::new).push_str(&t);
            }
        }
        if let Some(t) = piece
            && seq.stream_decoded_text(t, None).is_some()
        {
            seq.state
                .transition_to(SequenceState::Finished(FinishReason::StopSequence))
                .expect("stop-sequence finish is legal from Decoding");
            break;
        }
        if seq.generated_tokens.len() >= seq.max_tokens {
            seq.state
                .transition_to(SequenceState::Finished(FinishReason::Length))
                .expect("length finish is legal from Decoding");
        }
    }
    if !seq.state.is_finished() {
        seq.state
            .transition_to(SequenceState::Finished(FinishReason::Stop))
            .expect("eos finish is legal from Decoding");
    }

    let tail = seq.decode_state.flush(&tokenizer);
    seq.close_text_stream(tail);
    let result = seq.take_generation_result(&tokenizer, 0);
    let finished_as = std::mem::replace(&mut seq.state, SequenceState::Queued);

    let mut streamed = String::new();
    while let Ok(event) = rx.try_recv() {
        match event {
            GenerateEvent::Token(t) | GenerateEvent::TokenWithLogprobs(t, _) => {
                streamed.push_str(&t);
            }
            GenerateEvent::Done(_) | GenerateEvent::Error(_) | GenerateEvent::Prefill(_) => {}
        }
    }

    Run {
        streamed,
        result,
        finished_as,
    }
}

/// A stop string truncates the non-streaming text at the match, and the matched
/// text itself is never part of the output.
#[test]
fn a_stop_string_truncates_the_text_and_excludes_the_match() {
    let run = drive(" 4 5 6 7 8", &["5"], 64, 1);
    assert_eq!(run.result.text, " 4 ");
    assert!(!run.result.text.contains('5'));
    assert_eq!(run.streamed, run.result.text);
}

/// The concatenation of the streamed chunks equals the non-streaming text for
/// every chunking, so a held-back partial match never reaches the client.
#[test]
fn streamed_chunks_equal_the_non_streaming_text_at_every_boundary() {
    for chunk_bytes in 1..=6 {
        let run = drive("count 1 2 END of line", &["END"], 64, chunk_bytes);
        assert_eq!(run.streamed, run.result.text, "chunk_bytes = {chunk_bytes}");
        assert_eq!(run.result.text, "count 1 2 ", "chunk_bytes = {chunk_bytes}");
    }
}

/// The matched stop string reaches the response layer, which is what b10621's
/// `stopping_word` needs and what the text alone cannot supply, because the
/// match is excluded from it.
#[test]
fn the_matched_stop_string_reaches_the_result() {
    let run = drive("a END b", &["zzz", "END"], 64, 1);
    assert_eq!(run.result.stop_kind, StopKind::Word("END".to_string()));
    assert_eq!(run.result.stop_kind.word(), Some("END"));
    assert!(matches!(
        run.finished_as,
        SequenceState::Finished(FinishReason::StopSequence)
    ));
}

/// The finish reason separates the three outcomes. Before the fix the wire
/// answer could only be "length" or "stop", and a stop-string match was
/// indistinguishable from an EOS token because it was never detected at all.
#[test]
fn the_finish_reason_separates_word_eos_and_limit() {
    let word = drive("a END b", &["END"], 64, 1);
    assert_eq!(word.result.stop_kind, StopKind::Word("END".to_string()));
    assert_eq!(word.result.finish_reason, "stop");
    assert!(matches!(
        word.finished_as,
        SequenceState::Finished(FinishReason::StopSequence)
    ));

    let eos = drive("a b", &["END"], 64, 1);
    assert_eq!(eos.result.stop_kind, StopKind::Eos);
    assert_eq!(eos.result.finish_reason, "stop");
    assert!(matches!(
        eos.finished_as,
        SequenceState::Finished(FinishReason::Stop)
    ));

    let limit = drive("abcdef", &["END"], 4, 1);
    assert_eq!(limit.result.stop_kind, StopKind::Limit);
    assert_eq!(limit.result.finish_reason, "length");
    assert!(matches!(
        limit.finished_as,
        SequenceState::Finished(FinishReason::Length)
    ));
}

/// A stop string completing on the last budgeted token is a stop-word finish,
/// not a length finish: the request ended because the caller's own stop string
/// appeared, and telling an OpenAI client "length" would invite a continuation.
#[test]
fn a_match_on_the_last_budgeted_token_reports_stop_not_length() {
    // "ab!" is three bytes, so three tokens, and the budget is exactly three.
    let run = drive("ab!", &["!"], 3, 1);
    assert_eq!(run.result.text, "ab");
    assert_eq!(run.result.stop_kind, StopKind::Word("!".to_string()));
    assert_eq!(run.result.finish_reason, "stop");
}

/// Multibyte text is truncated on character boundaries, and a partial UTF-8
/// sequence the detokenizer held is still released as real output when it turns
/// out not to complete a stop string.
#[test]
fn a_multibyte_stream_is_truncated_on_character_boundaries() {
    let matched = drive("한국어 STOP 텍스트", &["STOP"], 128, 1);
    assert_eq!(matched.result.text, "한국어 ");
    assert_eq!(matched.streamed, matched.result.text);

    let unmatched = drive("한국어 텍스트", &["STOP"], 128, 1);
    assert_eq!(unmatched.result.text, "한국어 텍스트");
    assert_eq!(unmatched.streamed, unmatched.result.text);
    assert_eq!(unmatched.result.stop_kind, StopKind::Eos);
}

/// With no stop strings the funnel is a pass-through: every decoded piece is
/// streamed verbatim and nothing is truncated, which is the pre-#1466 behavior
/// that must not change for a request that supplies no `stop` field.
#[test]
fn without_stop_strings_nothing_is_held_or_truncated() {
    let run = drive("plain output with 5 and END inside", &[], 128, 1);
    assert_eq!(run.result.text, "plain output with 5 and END inside");
    assert_eq!(run.streamed, run.result.text);
    assert_eq!(run.result.stop_kind, StopKind::Eos);
}

/// Preemptive eviction restarts a decode from scratch. The matcher's held tail
/// and emitted-byte count describe the discarded run, so `reset` must clear them
/// while keeping the request's stop strings, or the re-run would truncate
/// against a stale offset.
#[test]
fn eviction_reset_keeps_the_stop_strings_and_drops_the_progress() {
    let mut matcher = StopMatcher::new(vec!["END".to_string()]);
    assert_eq!(matcher.push("abc EN").emit, "abc ");
    matcher.reset();
    assert_eq!(matcher.emitted_len(), 0);
    assert!(!matcher.has_matched());
    assert!(matcher.is_active());
    // The re-run still stops on the same string.
    let chunk = matcher.push("abc END");
    assert_eq!(chunk.emit, "abc ");
    assert_eq!(chunk.matched.as_deref(), Some("END"));
}
