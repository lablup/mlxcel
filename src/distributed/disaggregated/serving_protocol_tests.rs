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

//! Unit tests for the serving control-frame wire protocol (#126 B3b2a).
//!
//! Model-free: exercise the frame encode/decode round-trips, the control/tensor
//! discrimination, and the sampling-config mirror, with no GPU or model load.

use super::*;
use crate::distributed::transport::TransportMessage;
use bytes::Bytes;
use mlxcel_core::generate::SamplingConfig;

fn sample_sampling() -> SerializableSamplingState {
    sampling_to_serializable(&SamplingConfig::greedy())
}

#[test]
fn prefill_request_frame_round_trips_through_a_control_message() {
    let frame = PrefillRequestFrame {
        request_id: 7,
        prompt_tokens: vec![1, 2, 3, 42],
        sampling: sample_sampling(),
        max_tokens: 64,
        reply_to: "127.0.0.1:5555".to_string(),
        decode_target: None,
    };
    let message = frame.encode().expect("encode prefill request");
    let (op, payload) = control_parts(message).expect("control parts");
    assert_eq!(op, OP_PREFILL_REQUEST);
    assert_eq!(op, PrefillRequestFrame::OPERATION);

    let decoded = PrefillRequestFrame::decode(&payload).expect("decode prefill request");
    assert_eq!(decoded.request_id, 7);
    assert_eq!(decoded.prompt_tokens, vec![1, 2, 3, 42]);
    assert_eq!(decoded.max_tokens, 64);
    assert_eq!(decoded.reply_to, "127.0.0.1:5555");
    assert_eq!(decoded.decode_target, None);
}

/// Issue #201: a router-chosen decode target round-trips through the frame so
/// the prefill node forwards the KV handoff to the router-balanced decode node.
#[test]
fn prefill_request_frame_carries_router_chosen_decode_target() {
    let frame = PrefillRequestFrame {
        request_id: 9,
        prompt_tokens: vec![5, 6],
        sampling: sample_sampling(),
        max_tokens: 16,
        reply_to: "127.0.0.1:5555".to_string(),
        decode_target: Some("127.0.0.1:7001".to_string()),
    };
    let message = frame.encode().expect("encode prefill request");
    let (_op, payload) = control_parts(message).expect("control parts");
    let decoded = PrefillRequestFrame::decode(&payload).expect("decode prefill request");
    assert_eq!(decoded.decode_target.as_deref(), Some("127.0.0.1:7001"));
}

/// Issue #201: `decode_target` is `serde(default)` so a frame from a router
/// predating the field (no `decode_target` key) decodes to `None`, and the
/// prefill node falls back to its configured `--decode-peers` target.
#[test]
fn prefill_request_frame_decode_target_defaults_to_none() {
    let payload = serde_json::json!({
        "request_id": 4,
        "prompt_tokens": [1, 2],
        "sampling": sample_sampling(),
        "max_tokens": 8,
        "reply_to": "127.0.0.1:5555"
    });
    let decoded: PrefillRequestFrame =
        serde_json::from_slice(&serde_json::to_vec(&payload).unwrap()).expect("decode");
    assert_eq!(decoded.decode_target, None);
}

#[test]
fn decode_meta_frame_round_trips_through_a_control_message() {
    let frame = DecodeMetaFrame {
        request_id: 11,
        max_tokens: 32,
        sampling: sample_sampling(),
        reply_to: "127.0.0.1:6666".to_string(),
    };
    let message = frame.encode().expect("encode decode meta");
    let (op, payload) = control_parts(message).expect("control parts");
    assert_eq!(op, OP_DECODE_META);

    let decoded = DecodeMetaFrame::decode(&payload).expect("decode decode meta");
    assert_eq!(decoded.request_id, 11);
    assert_eq!(decoded.max_tokens, 32);
    assert_eq!(decoded.reply_to, "127.0.0.1:6666");
}

#[test]
fn result_frame_round_trips_both_phases() {
    let first = ResultFrame {
        request_id: 3,
        phase: ResultPhase::FirstToken,
        tokens: vec![" Hello".to_string()],
        start_sequence: 0,
        done: false,
        error: None,
        generated_tokens: Some(1),
    };
    let cont = ResultFrame {
        request_id: 3,
        phase: ResultPhase::Continuation,
        tokens: vec![" world".to_string(), "!".to_string()],
        start_sequence: 1,
        done: true,
        error: None,
        generated_tokens: Some(2),
    };
    for frame in [first, cont] {
        let message = frame.encode().expect("encode result");
        let (op, payload) = control_parts(message).expect("control parts");
        assert_eq!(op, OP_RESULT);
        let decoded = ResultFrame::decode(&payload).expect("decode result");
        assert_eq!(decoded.request_id, frame.request_id);
        assert_eq!(decoded.phase, frame.phase);
        assert_eq!(decoded.tokens, frame.tokens);
        assert_eq!(decoded.start_sequence, frame.start_sequence);
        assert_eq!(decoded.done, frame.done);
        assert_eq!(decoded.generated_tokens, frame.generated_tokens);
    }
}

/// Issue #387: the worker's authoritative generated-token count round-trips
/// through the result frame so the router can report exact usage for byte-
/// fallback tokenizers instead of counting emitted text pieces.
#[test]
fn result_frame_carries_authoritative_generated_token_count() {
    let frame = ResultFrame {
        request_id: 21,
        phase: ResultPhase::Continuation,
        tokens: Vec::new(),
        start_sequence: 0,
        done: true,
        error: None,
        generated_tokens: Some(15),
    };
    let message = frame.encode().expect("encode result");
    let (_op, payload) = control_parts(message).expect("control parts");
    let decoded = ResultFrame::decode(&payload).expect("decode result");
    assert_eq!(decoded.generated_tokens, Some(15));
}

/// Issue #387: `generated_tokens` is `serde(default)` so a frame from a sender
/// predating the field (no `generated_tokens` key) decodes to `None`, which
/// makes the router fall back to counting emitted text pieces in a mixed-version
/// cluster.
#[test]
fn result_frame_generated_tokens_defaults_to_none() {
    let payload = serde_json::json!({
        "request_id": 8,
        "phase": "continuation",
        "tokens": ["x"],
        "start_sequence": 1,
        "done": true,
        "error": null
    });
    let decoded: ResultFrame =
        serde_json::from_slice(&serde_json::to_vec(&payload).unwrap()).expect("decode");
    assert_eq!(decoded.generated_tokens, None);
}

/// `start_sequence` is `serde(default)` so frames from senders predating
/// issue #199 decode with the "unchecked" 0.
#[test]
fn result_frame_start_sequence_defaults_to_zero() {
    let payload = serde_json::json!({
        "request_id": 7,
        "phase": "continuation",
        "tokens": ["x"],
        "done": true,
        "error": null
    });
    let decoded: ResultFrame =
        serde_json::from_slice(&serde_json::to_vec(&payload).unwrap()).expect("decode");
    assert_eq!(decoded.start_sequence, 0);
}

#[test]
fn control_parts_rejects_a_tensor_data_frame() {
    let message = TransportMessage::TensorData {
        tensor_id: "kv-cache-handoff".to_string(),
        shape: vec![4],
        data: Bytes::from_static(&[0, 1, 2, 3]),
    };
    let err = control_parts(message).expect_err("a tensor frame is not a control frame");
    assert!(
        err.to_string().contains("TensorData"),
        "error should name the offending frame kind, got: {err}"
    );
}

#[test]
fn sampling_round_trips_through_the_serializable_mirror() {
    let mut config = SamplingConfig::greedy();
    config.temperature = 0.7;
    config.top_k = 40;
    config.top_p = 0.95;
    config.min_p = 0.05;
    config.seed = Some(1234);
    config.repetition_penalty = 1.1;
    config.frequency_penalty = 0.2;
    config.presence_penalty = 0.3;
    config.stop_token_ids = vec![13, 14];
    config.dry_sequence_breakers = vec![198];

    let restored = sampling_from_serializable(&sampling_to_serializable(&config));

    assert_eq!(restored.temperature, 0.7);
    assert_eq!(restored.top_k, 40);
    assert_eq!(restored.top_p, 0.95);
    assert_eq!(restored.min_p, 0.05);
    assert_eq!(restored.seed, Some(1234));
    assert_eq!(restored.repetition_penalty, 1.1);
    assert_eq!(restored.frequency_penalty, 0.2);
    assert_eq!(restored.presence_penalty, 0.3);
    assert_eq!(restored.stop_token_ids, vec![13, 14]);
    assert_eq!(restored.dry_sequence_breakers, vec![198]);
}
