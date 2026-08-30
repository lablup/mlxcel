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

//! Differential tests for the b10621 transcription contract (issue #1446).
//!
//! Every expectation here is read off the pinned upstream source rather than
//! guessed: `convert_transcriptions_to_chatcmpl` for the field semantics,
//! `server-http.cpp`'s multipart-to-JSON translation for the duplicate-part
//! rule, `json_value` for what a wrong type does, and
//! `server_task_result_cmpl_final::to_json_oaicompat_asr` for the response.

use super::*;
use axum::body::Body;
use axum::extract::FromRequest;
use axum::http::Request;

/// Build a `multipart/form-data` request body from `(name, filename, value)`
/// triples; a `Some(filename)` makes the part a file upload.
fn multipart_request(parts: &[(&str, Option<&str>, &[u8])]) -> Request<Body> {
    const BOUNDARY: &str = "----mlxcelTranscriptionBoundary";
    let mut body: Vec<u8> = Vec::new();
    for (name, filename, value) in parts {
        body.extend_from_slice(format!("--{BOUNDARY}\r\n").as_bytes());
        match filename {
            Some(filename) => body.extend_from_slice(
                format!(
                    "Content-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n\r\n"
                )
                .as_bytes(),
            ),
            None => body.extend_from_slice(
                format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
            ),
        }
        body.extend_from_slice(value);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{BOUNDARY}--\r\n").as_bytes());
    Request::builder()
        .method("POST")
        .uri("/v1/audio/transcriptions")
        .header(
            "content-type",
            format!("multipart/form-data; boundary={BOUNDARY}"),
        )
        .body(Body::from(body))
        .expect("multipart request builds")
}

async fn parse(parts: &[(&str, Option<&str>, &[u8])]) -> Result<TranscriptionForm, ErrorResponse> {
    let multipart = Multipart::from_request(multipart_request(parts), &())
        .await
        .expect("multipart extractor accepts the body");
    parse_multipart(multipart).await
}

/// A minimal well-formed 16 kHz mono 16-bit WAV of `frames` samples.
fn wav(frames: u32, sample_rate: u32, channels: u16) -> Vec<u8> {
    let bits = 16u16;
    let block_align = channels * bits / 8;
    let data_len = frames * u32::from(block_align);
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * u32::from(block_align)).to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    out.resize(44 + data_len as usize, 0);
    out
}

// ── multipart field set ─────────────────────────────────────────────────────

#[tokio::test]
async fn every_documented_field_is_read_off_the_form() {
    let clip = wav(1600, 16_000, 1);
    let form = parse(&[
        ("file", Some("clip.wav"), &clip),
        ("model", None, b"whisper-1"),
        ("prompt", None, b"Transcribe the meeting"),
        ("language", None, b"ko"),
        ("temperature", None, b"0.25"),
        ("response_format", None, b"json"),
        ("stream", None, b"true"),
        ("max_tokens", None, b"64"),
    ])
    .await
    .expect("a well-formed form parses");

    assert_eq!(form.file.as_ref().expect("file part").bytes, clip);
    assert_eq!(
        form.file.as_ref().and_then(|f| f.filename.as_deref()),
        Some("clip.wav")
    );
    assert_eq!(form.text("model"), Some("whisper-1"));
    assert_eq!(form.text("prompt"), Some("Transcribe the meeting"));
    assert_eq!(form.text("language"), Some("ko"));
    assert_eq!(form.text("response_format"), Some("json"));
    assert!(wants_stream(&form));
    assert_eq!(
        parse_numeric_field::<f32>(&form, "temperature").expect("parses"),
        Some(0.25)
    );
    assert_eq!(
        parse_numeric_field::<usize>(&form, "max_tokens").expect("parses"),
        Some(64)
    );
}

#[tokio::test]
async fn an_unknown_field_is_carried_but_changes_nothing() {
    // Upstream copies every form field into the chat body, where the chat
    // parser ignores what it does not know. Nothing about the transcription is
    // supposed to change.
    let form = parse(&[
        ("file", Some("clip.wav"), &wav(160, 16_000, 1)),
        ("unknown_field", None, b"whatever"),
    ])
    .await
    .expect("an unknown field is not an error");
    assert_eq!(form.text("unknown_field"), Some("whatever"));
    assert_eq!(asr_user_prompt(form.text("prompt"), None), ASR_USER_PROMPT);
    assert!(ensure_json_response_format(&form).is_ok());
}

#[tokio::test]
async fn a_duplicate_text_part_becomes_an_array_and_falls_back_to_the_default() {
    // `server-http.cpp` collapses a repeated key into a JSON array, and
    // `json_value<std::string>` then fails its type check and returns the
    // default. A duplicated `prompt` therefore neither wins nor errors: it is
    // ignored, and the preset prompt is used.
    let form = parse(&[
        ("file", Some("clip.wav"), &wav(160, 16_000, 1)),
        ("prompt", None, b"first"),
        ("prompt", None, b"second"),
    ])
    .await
    .expect("a duplicated field is not an error");
    assert!(form.fields["prompt"].is_array(), "{:?}", form.fields);
    assert_eq!(form.text("prompt"), None);
    assert_eq!(asr_user_prompt(form.text("prompt"), None), ASR_USER_PROMPT);

    // The same rule makes a duplicated `response_format` fall back to `json`,
    // so it is accepted even when one of the values is not.
    let format_form = parse(&[
        ("file", Some("clip.wav"), &wav(160, 16_000, 1)),
        ("response_format", None, b"text"),
        ("response_format", None, b"verbose_json"),
    ])
    .await
    .expect("a duplicated field is not an error");
    assert!(ensure_json_response_format(&format_form).is_ok());
}

#[tokio::test]
async fn a_duplicate_file_part_keeps_the_last_one() {
    // Upstream copies the parts into a `std::map` keyed by part name, so the
    // later file overwrites the earlier one.
    let first = wav(160, 16_000, 1);
    let second = wav(320, 16_000, 1);
    let form = parse(&[
        ("file", Some("a.wav"), &first),
        ("file", Some("b.wav"), &second),
    ])
    .await
    .expect("a duplicated file part is not an error");
    let file = form.file.expect("a file part");
    assert_eq!(file.bytes, second);
    assert_eq!(file.filename.as_deref(), Some("b.wav"));
}

#[tokio::test]
async fn a_form_with_no_file_part_is_detected() {
    let form = parse(&[("model", None, b"whisper-1")])
        .await
        .expect("the form itself parses");
    assert!(form.file.is_none());
    assert_eq!(
        NO_INPUT_FILE_MESSAGE, "No input file found for transcription",
        "b10621's own wording; a client may match on it"
    );
}

#[tokio::test]
async fn a_form_with_too_many_parts_is_refused() {
    let filler: Vec<(String, Option<&str>, Vec<u8>)> = (0..MAX_MULTIPART_PARTS + 4)
        .map(|i| (format!("f{i}"), None, b"x".to_vec()))
        .collect();
    let parts: Vec<(&str, Option<&str>, &[u8])> = filler
        .iter()
        .map(|(name, filename, value)| (name.as_str(), *filename, value.as_slice()))
        .collect();
    let err = parse(&parts).await.expect_err("too many parts is refused");
    assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
    assert!(
        err.error.message.contains("more than"),
        "{}",
        err.error.message
    );
}

// ── value domains ───────────────────────────────────────────────────────────

#[test]
fn response_format_accepts_json_and_nothing_else() {
    let mut form = TranscriptionForm::default();
    assert!(ensure_json_response_format(&form).is_ok(), "absent is json");
    form.fields
        .insert("response_format".into(), Value::String("json".into()));
    assert!(ensure_json_response_format(&form).is_ok());

    for other in ["text", "verbose_json", "srt", "vtt", "JSON", ""] {
        form.fields
            .insert("response_format".into(), Value::String(other.into()));
        let err = ensure_json_response_format(&form)
            .expect_err("only the exact string `json` is accepted, as upstream compares it");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert_eq!(err.error.message, ONLY_JSON_MESSAGE);
        assert_eq!(err.error.error_type, "invalid_request_error");
    }
}

#[test]
fn a_numeric_field_that_does_not_parse_is_a_bad_request() {
    // Upstream calls `std::stof` / `std::stoul`, whose `std::invalid_argument`
    // its exception wrapper turns into a 400.
    let mut form = TranscriptionForm::default();
    form.fields
        .insert("temperature".into(), Value::String("warm".into()));
    let err = parse_numeric_field::<f32>(&form, "temperature").expect_err("refused");
    assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
    assert!(err.error.message.contains("temperature"));

    form.fields
        .insert("temperature".into(), Value::String("  0.5 ".into()));
    assert_eq!(
        parse_numeric_field::<f32>(&form, "temperature").expect("parses"),
        Some(0.5)
    );
    form.fields
        .insert("temperature".into(), Value::String("   ".into()));
    assert_eq!(
        parse_numeric_field::<f32>(&form, "temperature").expect("blank is absent"),
        None
    );
}

#[test]
fn stream_is_compared_against_the_literal_true() {
    // The form carries strings, so upstream writes `stream == "true"` rather
    // than parsing a boolean vocabulary; `1`, `on` and `True` are all false.
    let mut form = TranscriptionForm::default();
    assert!(!wants_stream(&form));
    for (value, expected) in [
        ("true", true),
        ("false", false),
        ("1", false),
        ("on", false),
        ("True", false),
    ] {
        form.fields
            .insert("stream".into(), Value::String(value.into()));
        assert_eq!(wants_stream(&form), expected, "stream={value}");
    }
}

// ── prompt construction ─────────────────────────────────────────────────────

#[test]
fn the_asr_prompt_matches_upstreams_preset_and_language_suffix() {
    assert_eq!(asr_user_prompt(None, None), "Transcribe audio to text");
    assert_eq!(asr_user_prompt(Some(""), None), "Transcribe audio to text");
    assert_eq!(
        asr_user_prompt(None, Some("ko")),
        "Transcribe audio to text (language: ko)"
    );
    assert_eq!(
        asr_user_prompt(Some("Write down the call"), Some("en")),
        "Write down the call (language: en)"
    );
    // An empty language adds nothing, matching upstream's `!language.empty()`.
    assert_eq!(
        asr_user_prompt(Some("Write down the call"), Some("  ")),
        "Write down the call"
    );
}

#[test]
fn the_synthesized_chat_request_carries_the_prompt_and_the_clip() {
    let request = build_asr_chat_request(
        "gemma3n-e2b-4bit".into(),
        "Transcribe audio to text (language: ko)".into(),
        "QUJD".into(),
        "wav".into(),
        Some(0.25),
        Some(128),
    );
    assert_eq!(request.model, "gemma3n-e2b-4bit");
    assert!(
        !request.stream,
        "the chat dispatch never sets `stream`: a streamed transcription takes \
         `stream_asr_completion`, which drives the generation itself (#1446)"
    );
    assert_eq!(request.params.temperature, Some(0.25));
    assert_eq!(request.params.max_tokens, Some(128));
    assert_eq!(request.messages.len(), 1);
    let req::MessageContent::Parts(parts) = &request.messages[0].content else {
        panic!("the ASR message is a content-part message");
    };
    assert!(matches!(
        &parts[0],
        req::ContentPart::Text { text } if text == "Transcribe audio to text (language: ko)"
    ));
    assert!(matches!(
        &parts[1],
        req::ContentPart::InputAudio { input_audio }
            if input_audio.data == "QUJD" && input_audio.format == "wav"
    ));
    // The audio must reach the resolver as raw base64, which is its last
    // branch; a data-URI prefix would take a different one.
    assert_eq!(request.audio_inputs().len(), 1);
}

// ── response shape ──────────────────────────────────────────────────────────

#[test]
fn the_done_event_is_upstreams_asr_shape_not_openais_transcription_object() {
    let event = transcription_done_event(
        "hello world",
        AsrUsage {
            input_tokens: 207,
            output_tokens: 10,
            cached_tokens: 3,
        },
    );
    assert_eq!(event["type"], "transcript.text.done");
    assert_eq!(event["text"], "hello world");
    assert_eq!(event["usage"]["type"], "tokens");
    assert_eq!(event["usage"]["input_tokens"], 207);
    assert_eq!(event["usage"]["output_tokens"], 10);
    assert_eq!(event["usage"]["total_tokens"], 217);
    assert_eq!(event["usage"]["input_tokens_details"]["cached_tokens"], 3);
    // The classic OpenAI keys are deliberately absent: upstream emits the
    // transcript-event shape, and a `language` / `duration` here would be an
    // mlxcel invention on the shared route.
    assert!(event.get("language").is_none());
    assert!(event.get("duration").is_none());
}

#[test]
fn the_delta_event_is_upstreams_partial_shape() {
    let event = transcription_delta_event("hel");
    assert_eq!(event["type"], "transcript.text.delta");
    assert_eq!(event["delta"], "hel");
    assert_eq!(event.as_object().expect("object").len(), 2);
}

// ── clip limits ─────────────────────────────────────────────────────────────

#[test]
fn a_well_formed_clip_reports_its_geometry() {
    let geometry = probe_wav(&wav(16_000, 16_000, 1)).expect("a 1-second clip parses");
    assert_eq!(geometry.sample_rate, 16_000);
    assert_eq!(geometry.channels, 1);
    assert_eq!(geometry.bits_per_sample, 16);
    assert_eq!(geometry.data_bytes, 32_000);
    assert!((geometry.duration_seconds - 1.0).abs() < 1e-9);
}

#[test]
fn a_malformed_or_truncated_clip_is_refused_from_the_header() {
    for (name, bytes) in [
        ("empty", Vec::new()),
        ("not riff", b"this is not audio at all".to_vec()),
        ("riff with no chunks", b"RIFF\x24\x00\x00\x00WAVE".to_vec()),
        ("truncated fmt", {
            let mut w = wav(160, 16_000, 1);
            w.truncate(30);
            w
        }),
        ("no data chunk", {
            let mut w = wav(160, 16_000, 1);
            w.truncate(36);
            w
        }),
    ] {
        assert!(
            probe_wav(&bytes).is_err(),
            "{name} must be refused from the header"
        );
        let err = ensure_clip_within_limits(&bytes).expect_err("{name}");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(
            err.error
                .message
                .contains("failed to read the uploaded audio"),
            "{name}: {}",
            err.error.message
        );
    }
}

#[test]
fn a_header_declaring_more_audio_than_it_carries_is_clamped() {
    // The amplifying case: a 44-byte file whose `data` chunk claims 4 GiB. The
    // declared size is clamped to what is present, so the duration reported is
    // the real one and no decoder is asked to materialise the difference.
    let mut bomb = wav(0, 16_000, 1);
    let data_size_offset = bomb.len() - 4;
    bomb[data_size_offset..].copy_from_slice(&u32::MAX.to_le_bytes());
    let geometry = probe_wav(&bomb).expect("the header still parses");
    assert_eq!(geometry.data_bytes, 0);
    assert_eq!(geometry.duration_seconds, 0.0);
    assert!(ensure_clip_within_limits(&bomb).is_ok());
}

#[test]
fn an_over_long_over_wide_or_over_fast_clip_is_refused() {
    // Each bound is checked against a header alone, so none of these cases
    // allocates the audio they describe.
    let mut long = wav(0, 16_000, 1);
    let data_size_offset = long.len() - 4;
    let declared = (MAX_DURATION_SECONDS as u32 + 60) * 16_000 * 2;
    long[data_size_offset..].copy_from_slice(&declared.to_le_bytes());
    long.resize(44 + declared as usize, 0);
    let err = ensure_clip_within_limits(&long).expect_err("an over-long clip is refused");
    assert!(
        err.error.message.contains("seconds"),
        "{}",
        err.error.message
    );

    let fast = wav(16, MAX_SAMPLE_RATE + 1, 1);
    let err = ensure_clip_within_limits(&fast).expect_err("an over-fast clip is refused");
    assert!(
        err.error.message.contains("sample rate"),
        "{}",
        err.error.message
    );

    let wide = wav(16, 16_000, MAX_CHANNELS + 1);
    let err = ensure_clip_within_limits(&wide).expect_err("an over-wide clip is refused");
    assert!(
        err.error.message.contains("channels"),
        "{}",
        err.error.message
    );
}

#[test]
fn a_degenerate_format_chunk_is_refused_rather_than_dividing_by_zero() {
    let mut zero_rate = wav(160, 16_000, 1);
    // The sample rate sits at byte 24 of a canonical 44-byte header.
    zero_rate[24..28].copy_from_slice(&0u32.to_le_bytes());
    assert!(probe_wav(&zero_rate).is_err());

    let mut zero_channels = wav(160, 16_000, 1);
    zero_channels[22..24].copy_from_slice(&0u16.to_le_bytes());
    assert!(probe_wav(&zero_channels).is_err());
}
