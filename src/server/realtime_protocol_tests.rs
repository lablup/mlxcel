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

//! Unit tests of the `/v1/realtime` wire types.

use base64::Engine as _;
use serde_json::{Value, json};

use super::*;

fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

#[test]
fn pcm16_round_trip() {
    let ints: [i16; 6] = [0, 1, -1, 12345, i16::MAX, i16::MIN];
    let bytes: Vec<u8> = ints.iter().flat_map(|v| v.to_le_bytes()).collect();
    let samples = pcm16_from_base64(&b64(&bytes)).unwrap();
    let expected: Vec<f32> = ints.iter().map(|&v| f32::from(v) / 32768.0).collect();
    assert_eq!(samples, expected);
    assert!(samples.iter().all(|x| (-1.0..1.0).contains(x)));

    // Encoding scales by 32767 after clipping, as upstream does.
    let encoded = pcm16_bytes(&[0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -3.0]);
    let decoded: Vec<i16> = encoded
        .chunks_exact(2)
        .map(|p| i16::from_le_bytes([p[0], p[1]]))
        .collect();
    assert_eq!(
        decoded,
        vec![0, 16384, -16384, 32767, -32767, 32767, -32767]
    );
    assert_eq!(audio_to_base64(&[0.5]), b64(&16384i16.to_le_bytes()));
}

#[test]
fn odd_byte_count_rejected() {
    let err = pcm16_from_base64(&b64(&[1, 2, 3])).unwrap_err();
    assert!(err.contains("even number of bytes"), "{err}");
    let err = pcm16_from_base64("not base64!").unwrap_err();
    assert!(err.contains("valid base64"), "{err}");
    assert!(pcm16_from_base64("").unwrap().is_empty());
}

#[test]
fn event_mapping_names_and_fields() {
    let wire = |event: VoiceChatEvent| -> Value {
        serde_json::from_str(&to_wire_json(&serialize_event(event))).unwrap()
    };

    let text = wire(VoiceChatEvent::AssistantTextDelta {
        frame_index: 3,
        token_id: 42,
        delta: " Paris".into(),
        text: "The capital is Paris".into(),
    });
    assert_eq!(text["type"], "response.text.delta");
    assert_eq!(text["frame_index"], 3);
    assert_eq!(text["token_id"], 42);
    assert_eq!(text["delta"], " Paris");
    assert_eq!(text["text"], "The capital is Paris");
    let id = text["event_id"].as_str().unwrap();
    assert!(id.starts_with("event_") && id.len() == 22, "{id}");
    assert!(id[6..].chars().all(|c| c.is_ascii_hexdigit()));

    let function = wire(VoiceChatEvent::FunctionDelta {
        frame_index: 1,
        token_id: 7,
        delta: "f".into(),
        text: "f".into(),
    });
    assert_eq!(function["type"], "response.function.delta");
    assert_eq!(function["token_id"], 7);

    let transcript = wire(VoiceChatEvent::UserTranscriptDelta {
        frame_index: 2,
        delta: " France".into(),
        text: "capital of France".into(),
    });
    assert_eq!(
        transcript["type"],
        "conversation.item.input_audio_transcription.delta"
    );
    assert_eq!(transcript["delta"], " France");
    assert_eq!(transcript["transcript"], "capital of France");
    assert!(transcript.get("text").is_none());

    let audio = wire(VoiceChatEvent::Audio {
        frame_index: 5,
        samples: vec![0.5; 1764],
        sample_rate: 22_050,
        audio_codes: (0..31).collect(),
    });
    assert_eq!(audio["type"], "response.audio.delta");
    assert_eq!(audio["frame_index"], 5);
    assert_eq!(audio["format"], "pcm16");
    assert_eq!(audio["sample_rate"], 22_050);
    assert_eq!(audio["channels"], 1);
    assert_eq!(audio["audio_codes"].as_array().unwrap().len(), 31);
    let pcm = pcm16_from_base64(audio["delta"].as_str().unwrap()).unwrap();
    assert_eq!(pcm.len(), 1764);

    let done = wire(VoiceChatEvent::Done { frame_index: 9 });
    assert_eq!(done["type"], "response.done");
    assert_eq!(done["frame_index"], 9);
    assert_eq!(done.as_object().unwrap().len(), 3, "{done}");
    let cancelled = wire(VoiceChatEvent::Cancelled { frame_index: 4 });
    assert_eq!(cancelled["type"], "response.cancelled");
    assert_eq!(cancelled["frame_index"], 4);
}

#[test]
fn session_and_error_events_carry_the_documented_shape() {
    let created: Value = serde_json::from_str(&to_wire_json(&ServerEvent::SessionCreated {
        session: SessionObject {
            id: new_session_id(),
            state: "configuring".into(),
            model: None,
            frame_samples: None,
            input_audio_format: AudioFormat::pcm16(INPUT_SAMPLE_RATE),
            output_audio_format: AudioFormat::pcm16(OUTPUT_SAMPLE_RATE),
        },
    }))
    .unwrap();
    assert_eq!(created["type"], "session.created");
    let id = created["session"]["id"].as_str().unwrap();
    assert!(id.starts_with("sess_") && id.len() == 21, "{id}");
    assert_eq!(created["session"]["state"], "configuring");
    assert_eq!(
        created["session"]["input_audio_format"],
        json!({"type": "pcm16", "sample_rate": 16000})
    );
    assert_eq!(
        created["session"]["output_audio_format"],
        json!({"type": "pcm16", "sample_rate": 22050})
    );
    assert!(created["session"].get("model").is_none());

    let error: Value =
        serde_json::from_str(&to_wire_json(&ServerEvent::error(CODE_SERVER_BUSY, "busy"))).unwrap();
    assert_eq!(error["type"], "error");
    assert_eq!(
        error["error"],
        json!({"code": "server_busy", "message": "busy"})
    );
}

#[test]
fn client_messages_parse_like_upstream() {
    assert!(parse_client_message("not json").is_err());
    assert!(parse_client_message("[1, 2]").is_err());
    assert_eq!(
        parse_client_message(r#"{"type": "session.ping"}"#).unwrap(),
        ClientMessage::Ping
    );
    assert_eq!(
        parse_client_message(r#"{"type": "response.cancel"}"#).unwrap(),
        ClientMessage::Cancel
    );
    assert_eq!(
        parse_client_message(r#"{"type": "input_audio_buffer.commit"}"#).unwrap(),
        ClientMessage::Commit { pad_partial: true }
    );
    assert_eq!(
        parse_client_message(r#"{"type": "input_audio_buffer.commit", "pad_partial": false}"#)
            .unwrap(),
        ClientMessage::Commit { pad_partial: false }
    );
    assert_eq!(
        parse_client_message(r#"{"audio": ""}"#).unwrap(),
        ClientMessage::Other(String::new())
    );

    let request = parse_session_request(&json!({
        "system_prompt": "Be concise.",
        "seed": 7,
        "max_streaming_seconds": 12.5
    }))
    .unwrap();
    assert_eq!(request.model, None);
    assert_eq!(request.system_prompt.as_deref(), Some("Be concise."));
    assert_eq!(request.seed, 7);
    assert_eq!(request.max_streaming_seconds, Some(12.5));
    assert_eq!(parse_session_request(&Value::Null).unwrap().seed, 0);
    assert!(parse_session_request(&json!({"seed": -1})).is_err());
    assert!(parse_session_request(&json!({"seed": "x"})).is_err());

    let (samples, rate) = parse_append(None, None).unwrap();
    assert!(samples.is_empty());
    assert_eq!(rate, 16_000);
    assert_eq!(parse_append(None, Some(&json!(8000))).unwrap().1, 8000);
    assert!(parse_append(None, Some(&json!("fast"))).is_err());
    assert!(parse_append(Some(&json!(5)), None).is_err());
}
