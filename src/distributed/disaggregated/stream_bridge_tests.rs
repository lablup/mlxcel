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

use std::time::Duration;

use super::*;

fn make_bridge() -> StreamBridge {
    StreamBridge::new("test-req-1".to_string(), Duration::from_secs(30))
}

fn prefill_token(seq: u64) -> TokenEvent {
    TokenEvent {
        token_id: 42,
        text: "Hello".to_string(),
        sequence_number: seq,
        source: TokenSource::Prefill,
        is_final: false,
    }
}

fn decode_token(seq: u64, is_final: bool) -> TokenEvent {
    TokenEvent {
        token_id: 100 + seq as i32,
        text: format!("tok{seq}"),
        sequence_number: seq,
        source: TokenSource::Decode,
        is_final,
    }
}

// ── StreamPhase tests ────────────────────────────────────────────────

#[test]
fn stream_phase_display() {
    assert_eq!(
        StreamPhase::WaitingForPrefill.to_string(),
        "waiting_for_prefill"
    );
    assert_eq!(
        StreamPhase::HandoffToDecode.to_string(),
        "handoff_to_decode"
    );
    assert_eq!(StreamPhase::Decoding.to_string(), "decoding");
    assert_eq!(StreamPhase::Complete.to_string(), "complete");
}

// ── TokenSource tests ────────────────────────────────────────────────

#[test]
fn token_source_display() {
    assert_eq!(TokenSource::Prefill.to_string(), "prefill");
    assert_eq!(TokenSource::Decode.to_string(), "decode");
    assert_eq!(TokenSource::Local.to_string(), "local");
}

// ── StreamBridge lifecycle tests ─────────────────────────────────────

#[test]
fn bridge_initial_state() {
    let bridge = make_bridge();
    assert_eq!(bridge.current_phase(), StreamPhase::WaitingForPrefill);
    assert_eq!(bridge.tokens_emitted(), 0);
    assert!(!bridge.is_finalized());
    assert!(bridge.time_to_first_token().is_none());
    assert_eq!(bridge.request_id(), "test-req-1");
}

#[test]
fn bridge_submit_first_token_success() {
    let bridge = make_bridge();
    let token = prefill_token(0);
    assert!(bridge.submit_first_token(&token).is_ok());
    assert_eq!(bridge.current_phase(), StreamPhase::HandoffToDecode);
    assert_eq!(bridge.tokens_emitted(), 1);
    assert!(bridge.time_to_first_token().is_some());
}

#[test]
fn bridge_submit_first_token_wrong_seq() {
    let bridge = make_bridge();
    let token = prefill_token(1); // Should be 0
    let result = bridge.submit_first_token(&token);
    assert!(result.is_err());
    match result.unwrap_err() {
        StreamBridgeError::SequenceGap { expected, got } => {
            assert_eq!(expected, 0);
            assert_eq!(got, 1);
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn bridge_submit_first_token_wrong_phase() {
    let bridge = make_bridge();
    bridge.submit_first_token(&prefill_token(0)).unwrap();
    // Try submitting again -- phase is now HandoffToDecode.
    let result = bridge.submit_first_token(&prefill_token(0));
    assert!(result.is_err());
    match result.unwrap_err() {
        StreamBridgeError::InvalidPhaseTransition { from, expected } => {
            assert_eq!(from, StreamPhase::HandoffToDecode);
            assert_eq!(expected, StreamPhase::WaitingForPrefill);
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn bridge_start_decode_stream() {
    let bridge = make_bridge();
    bridge.submit_first_token(&prefill_token(0)).unwrap();
    assert!(bridge.start_decode_stream().is_ok());
    assert_eq!(bridge.current_phase(), StreamPhase::Decoding);
}

#[test]
fn bridge_start_decode_stream_wrong_phase() {
    let bridge = make_bridge();
    let result = bridge.start_decode_stream();
    assert!(result.is_err());
}

#[test]
fn bridge_full_lifecycle() {
    let bridge = make_bridge();

    // Phase 1: first token from prefill.
    bridge.submit_first_token(&prefill_token(0)).unwrap();

    // Phase 2: decode starts.
    bridge.start_decode_stream().unwrap();

    // Phase 3: decode tokens.
    bridge.submit_decode_token(&decode_token(1, false)).unwrap();
    bridge.submit_decode_token(&decode_token(2, false)).unwrap();
    bridge.submit_decode_token(&decode_token(3, true)).unwrap();

    assert_eq!(bridge.tokens_emitted(), 4); // 1 prefill + 3 decode

    // Phase 4: finalize.
    assert!(bridge.finalize());
    assert_eq!(bridge.current_phase(), StreamPhase::Complete);
    assert!(bridge.is_finalized());
}

#[test]
fn bridge_decode_token_sequence_gap() {
    let bridge = make_bridge();
    bridge.submit_first_token(&prefill_token(0)).unwrap();
    bridge.start_decode_stream().unwrap();

    // Skip sequence number 1, go straight to 2.
    let result = bridge.submit_decode_token(&decode_token(2, false));
    assert!(result.is_err());
    match result.unwrap_err() {
        StreamBridgeError::SequenceGap { expected, got } => {
            assert_eq!(expected, 1);
            assert_eq!(got, 2);
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn bridge_decode_token_wrong_phase() {
    let bridge = make_bridge();
    // Haven't submitted first token yet.
    let result = bridge.submit_decode_token(&decode_token(1, false));
    assert!(result.is_err());
}

#[test]
fn bridge_finalize_only_once() {
    let bridge = make_bridge();
    assert!(bridge.finalize());
    assert!(!bridge.finalize()); // Second call returns false.
}

#[test]
fn bridge_finalize_from_any_phase() {
    // Can finalize even from WaitingForPrefill (e.g., error case).
    let bridge = make_bridge();
    assert!(bridge.finalize());
    assert_eq!(bridge.current_phase(), StreamPhase::Complete);
}

#[test]
fn bridge_elapsed_increases() {
    let bridge = make_bridge();
    let t1 = bridge.elapsed();
    std::thread::sleep(Duration::from_millis(5));
    let t2 = bridge.elapsed();
    assert!(t2 > t1);
}

#[test]
fn bridge_handoff_timeout() {
    let bridge = StreamBridge::new("req-timeout".to_string(), Duration::from_millis(1));
    bridge.submit_first_token(&prefill_token(0)).unwrap();
    // Phase is now HandoffToDecode. Timeout is measured from handoff start.
    std::thread::sleep(Duration::from_millis(5));
    assert!(bridge.is_handoff_timed_out());
}

#[test]
fn bridge_handoff_not_timed_out_in_other_phases() {
    let bridge = StreamBridge::new("req-ok".to_string(), Duration::from_millis(1));
    // Still in WaitingForPrefill, not HandoffToDecode.
    std::thread::sleep(Duration::from_millis(5));
    assert!(!bridge.is_handoff_timed_out());
}

#[test]
fn bridge_debug_format() {
    let bridge = make_bridge();
    let debug = format!("{bridge:?}");
    assert!(debug.contains("StreamBridge"));
    assert!(debug.contains("test-req-1"));
}

// ── StreamBridgeError tests ──────────────────────────────────────────

#[test]
fn error_display_phase_transition() {
    let err = StreamBridgeError::InvalidPhaseTransition {
        from: StreamPhase::Decoding,
        expected: StreamPhase::HandoffToDecode,
    };
    let msg = err.to_string();
    assert!(msg.contains("invalid stream phase transition"));
    assert!(msg.contains("decoding"));
}

#[test]
fn error_display_sequence_gap() {
    let err = StreamBridgeError::SequenceGap {
        expected: 5,
        got: 7,
    };
    let msg = err.to_string();
    assert!(msg.contains("expected seq=5"));
    assert!(msg.contains("got seq=7"));
}

#[test]
fn error_display_lock_poisoned() {
    let err = StreamBridgeError::LockPoisoned;
    assert!(err.to_string().contains("lock poisoned"));
}

#[test]
fn error_is_std_error() {
    let err: Box<dyn std::error::Error> = Box::new(StreamBridgeError::LockPoisoned);
    assert!(err.to_string().contains("lock poisoned"));
}
