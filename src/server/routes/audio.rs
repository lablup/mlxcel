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

//! OpenAI-compatible audio endpoints.
//!
//! Three route families are exposed: `/audio/speech` (text-to-speech, JSON in,
//! binary audio out), `/audio/transcriptions`, and `/audio/translations`
//! (speech-to-text, `multipart/form-data` in, JSON out). This module only
//! translates the HTTP request/response shape and admission; the model work
//! lives behind the [`AudioModelProvider`](crate::server::audio_model::AudioModelProvider)
//! seam in [`AppState::audio_model`]. While that slot is `None` every route
//! returns a structured `501 Not Implemented` after parsing the request.

use axum::{
    Json,
    body::Body,
    extract::{Multipart, State, multipart::Field},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};

use crate::audio::encode_wav_pcm16;
use crate::server::AppState;
use crate::server::audio_model::{
    AudioModelError, AudioModelKind, AudioSynthesizeInput, AudioTranscribeInput,
    AudioTranscribeOutput,
};
use crate::server::types::{AudioSpeechRequest, AudioTranscriptionResponse, ErrorResponse};

/// Resolved output container for a synthesized-audio response.
#[derive(Debug)]
struct AudioFormat {
    /// `Content-Type` header value.
    content_type: &'static str,
    /// Filename extension used in `Content-Disposition`.
    extension: &'static str,
}

/// POST /v1/audio/speech (text-to-speech).
///
/// Parses the JSON body, then returns a binary audio body when a TTS model is
/// loaded or a structured `501 Not Implemented` while no model is wired in.
pub async fn audio_speech(
    State(state): State<AppState>,
    Json(request): Json<AudioSpeechRequest>,
) -> Response {
    if !state.can_accept_request() {
        return ErrorResponse::service_unavailable("All slots are busy. Please try again later.")
            .into_response();
    }

    let Some(provider) = state.audio_model.clone() else {
        return audio_kind_not_loaded(AudioModelKind::Tts).into_response();
    };
    if !provider.supports(AudioModelKind::Tts) {
        return audio_kind_not_loaded(AudioModelKind::Tts).into_response();
    }

    // Resolve the container only once a model can actually serve the request,
    // so the dominant pre-model response stays the structured 501 above.
    let audio_format = match resolve_audio_format(request.response_format.as_deref()) {
        Ok(format) => format,
        Err(err) => return err.into_response(),
    };

    let input = AudioSynthesizeInput {
        input: request.input,
        voice: request.voice,
        speed: request.speed,
    };

    let started = std::time::Instant::now();
    let synth = tokio::task::spawn_blocking(move || provider.synthesize(input)).await;
    let output = match synth {
        Ok(Ok(output)) => output,
        Ok(Err(err)) => return audio_model_error_response(err).into_response(),
        Err(join_err) => {
            tracing::error!("audio synthesis task panicked: {join_err}");
            return ErrorResponse::new("audio synthesis failed", "server_error").into_response();
        }
    };
    state
        .metrics
        .record_request(0, 0, started.elapsed().as_millis() as u64);

    let body = encode_wav_pcm16(&output.samples, output.sample_rate, output.channels);
    build_audio_response(&audio_format, body)
}

/// POST /v1/audio/transcriptions (speech-to-text, transcribe in place).
pub async fn audio_transcriptions(State(state): State<AppState>, multipart: Multipart) -> Response {
    transcribe(state, multipart, false).await
}

/// POST /v1/audio/translations (speech-to-text, translate to English).
pub async fn audio_translations(State(state): State<AppState>, multipart: Multipart) -> Response {
    transcribe(state, multipart, true).await
}

/// Shared speech-to-text flow for transcriptions and translations.
async fn transcribe(state: AppState, multipart: Multipart, translate: bool) -> Response {
    if !state.can_accept_request() {
        return ErrorResponse::service_unavailable("All slots are busy. Please try again later.")
            .into_response();
    }

    // Drive the multipart parser before consulting the model so malformed
    // uploads surface as a structured 400 through the real request flow.
    let form = match parse_transcription_multipart(multipart).await {
        Ok(form) => form,
        Err(err) => return err.into_response(),
    };
    tracing::debug!(
        target: "mlxcel::audio",
        requested_model = form.model.as_deref().unwrap_or(""),
        translate,
        "audio transcription request parsed"
    );

    let Some((audio, filename)) = form.file else {
        return ErrorResponse::new(
            "missing required 'file' part in multipart form-data",
            "invalid_request_error",
        )
        .into_response();
    };

    let Some(provider) = state.audio_model.clone() else {
        return audio_kind_not_loaded(AudioModelKind::Stt).into_response();
    };
    if !provider.supports(AudioModelKind::Stt) {
        return audio_kind_not_loaded(AudioModelKind::Stt).into_response();
    }

    let response_format = form.response_format;
    let input = AudioTranscribeInput {
        audio,
        filename,
        language: form.language,
        temperature: form.temperature,
        translate,
    };

    let started = std::time::Instant::now();
    let result = tokio::task::spawn_blocking(move || provider.transcribe(input)).await;
    let output = match result {
        Ok(Ok(output)) => output,
        Ok(Err(err)) => return audio_model_error_response(err).into_response(),
        Err(join_err) => {
            tracing::error!("audio transcription task panicked: {join_err}");
            return ErrorResponse::new("audio transcription failed", "server_error")
                .into_response();
        }
    };
    state
        .metrics
        .record_request(0, 0, started.elapsed().as_millis() as u64);

    build_transcription_response(output, response_format.as_deref())
}

/// Fields extracted from a transcription/translation multipart form. The audio
/// file is carried separately from the JSON-shaped scalar fields.
#[derive(Default)]
struct ParsedTranscriptionForm {
    /// `(bytes, original filename)` of the uploaded audio.
    file: Option<(Vec<u8>, Option<String>)>,
    model: Option<String>,
    language: Option<String>,
    response_format: Option<String>,
    temperature: Option<f32>,
}

/// Parse the STT `multipart/form-data` body: the `file` part plus the
/// `model`, `language`, `response_format`, and `temperature` text fields.
async fn parse_transcription_multipart(
    mut multipart: Multipart,
) -> Result<ParsedTranscriptionForm, ErrorResponse> {
    let mut form = ParsedTranscriptionForm::default();
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(field)) => field,
            Ok(None) => break,
            Err(err) => {
                return Err(ErrorResponse::new(
                    format!("invalid multipart form-data: {err}"),
                    "invalid_request_error",
                ));
            }
        };

        // Copy the part name to an owned value so the field body can be
        // consumed below without an outstanding borrow.
        let name = field.name().map(str::to_string);
        match name.as_deref() {
            Some("file") => {
                let filename = field.file_name().map(str::to_string);
                let bytes = field.bytes().await.map_err(|err| {
                    ErrorResponse::new(
                        format!("failed to read 'file' part: {err}"),
                        "invalid_request_error",
                    )
                })?;
                form.file = Some((bytes.to_vec(), filename));
            }
            Some("model") => form.model = Some(read_text_field(field).await?),
            Some("language") => form.language = Some(read_text_field(field).await?),
            Some("response_format") => form.response_format = Some(read_text_field(field).await?),
            Some("temperature") => {
                let raw = read_text_field(field).await?;
                let trimmed = raw.trim();
                if !trimmed.is_empty() {
                    let value = trimmed.parse::<f32>().map_err(|_| {
                        ErrorResponse::new(
                            format!("invalid 'temperature' value: {raw}"),
                            "invalid_request_error",
                        )
                    })?;
                    form.temperature = Some(value);
                }
            }
            // Drain unknown parts so the stream advances to the next field.
            _ => {
                let _ = field.bytes().await;
            }
        }
    }
    Ok(form)
}

/// Read a multipart field body as UTF-8 text.
async fn read_text_field(field: Field<'_>) -> Result<String, ErrorResponse> {
    field.text().await.map_err(|err| {
        ErrorResponse::new(
            format!("failed to read text field: {err}"),
            "invalid_request_error",
        )
    })
}

/// Map a requested `response_format` to an output container. Only WAV is
/// available today; other formats are an explicit follow-up.
fn resolve_audio_format(requested: Option<&str>) -> Result<AudioFormat, ErrorResponse> {
    match requested
        .map(|value| value.trim().to_ascii_lowercase())
        .as_deref()
    {
        None | Some("") | Some("wav") => Ok(AudioFormat {
            content_type: "audio/wav",
            extension: "wav",
        }),
        Some(other) => Err(ErrorResponse::new(
            format!("response_format '{other}' is not supported; only 'wav' is available"),
            "invalid_request_error",
        )),
    }
}

/// Build the binary audio response with the OpenAI attachment headers.
fn build_audio_response(format: &AudioFormat, body: Vec<u8>) -> Response {
    let disposition = format!("attachment; filename=speech.{}", format.extension);
    match Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, format.content_type)
        .header(header::CONTENT_DISPOSITION, disposition)
        .body(Body::from(body))
    {
        Ok(response) => response,
        Err(err) => {
            tracing::error!("failed to build audio response: {err}");
            ErrorResponse::new("failed to build audio response", "server_error").into_response()
        }
    }
}

/// Render a transcription result in the requested response format.
fn build_transcription_response(
    output: AudioTranscribeOutput,
    response_format: Option<&str>,
) -> Response {
    match response_format
        .map(|value| value.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("text") => (StatusCode::OK, output.text).into_response(),
        Some("verbose_json") => Json(AudioTranscriptionResponse {
            text: output.text,
            language: output.language,
            duration: output.duration_seconds,
        })
        .into_response(),
        // `json` (the default) and any other value collapse to `{ "text": ... }`.
        _ => Json(AudioTranscriptionResponse {
            text: output.text,
            language: None,
            duration: None,
        })
        .into_response(),
    }
}

/// Structured `501` reported when no model serves the requested audio
/// direction.
fn audio_kind_not_loaded(kind: AudioModelKind) -> ErrorResponse {
    ErrorResponse::not_implemented(format!("audio model kind not loaded: {kind}"))
}

/// Convert an [`AudioModelError`] into the matching HTTP error response.
fn audio_model_error_response(err: AudioModelError) -> ErrorResponse {
    match err {
        AudioModelError::KindNotLoaded(kind) => audio_kind_not_loaded(kind),
        AudioModelError::Inference(message) => ErrorResponse::new(
            format!("audio model inference failed: {message}"),
            "server_error",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_format_defaults_to_wav() {
        for input in [None, Some(""), Some("wav"), Some("WAV"), Some("  wav  ")] {
            let format = resolve_audio_format(input).expect("wav-family format resolves");
            assert_eq!(format.content_type, "audio/wav");
            assert_eq!(format.extension, "wav");
        }
    }

    #[test]
    fn resolve_format_rejects_unsupported() {
        let err = resolve_audio_format(Some("mp3")).expect_err("mp3 is not supported yet");
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(err.error.message.contains("mp3"));
    }

    #[test]
    fn kind_not_loaded_is_501_with_phrase() {
        for kind in [AudioModelKind::Stt, AudioModelKind::Tts] {
            let err = audio_kind_not_loaded(kind);
            assert_eq!(err.status, StatusCode::NOT_IMPLEMENTED);
            assert_eq!(err.error.error_type, "not_implemented");
            assert!(err.error.message.contains("audio model kind not loaded"));
            assert!(err.error.message.contains(kind.as_str()));
        }
    }

    #[test]
    fn model_error_maps_kinds_and_inference() {
        let inference = audio_model_error_response(AudioModelError::Inference("boom".into()));
        assert_eq!(inference.error.error_type, "server_error");
        assert!(inference.error.message.contains("boom"));

        let not_loaded =
            audio_model_error_response(AudioModelError::KindNotLoaded(AudioModelKind::Tts));
        assert_eq!(not_loaded.status, StatusCode::NOT_IMPLEMENTED);
    }

    #[test]
    fn audio_response_sets_binary_headers() {
        let response = build_audio_response(
            &AudioFormat {
                content_type: "audio/wav",
                extension: "wav",
            },
            vec![0, 1, 2, 3],
        );
        assert_eq!(response.status(), StatusCode::OK);
        let content_type = response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        assert_eq!(content_type, "audio/wav");
        let disposition = response
            .headers()
            .get(header::CONTENT_DISPOSITION)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        assert_eq!(disposition, "attachment; filename=speech.wav");
    }

    #[test]
    fn transcription_format_selects_content_type() {
        let make = || AudioTranscribeOutput {
            text: "hello".to_string(),
            language: Some("en".to_string()),
            duration_seconds: Some(1.5),
        };

        let json = build_transcription_response(make(), None);
        assert_eq!(json.status(), StatusCode::OK);
        let json_ct = json
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        assert!(json_ct.contains("application/json"), "got {json_ct}");

        let text = build_transcription_response(make(), Some("text"));
        let text_ct = text
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        assert!(text_ct.contains("text/plain"), "got {text_ct}");
    }

    #[test]
    fn minimal_transcription_serializes_to_text_only() {
        let body = AudioTranscriptionResponse {
            text: "hi".to_string(),
            language: None,
            duration: None,
        };
        let json = serde_json::to_value(&body).expect("serializes");
        assert_eq!(json, serde_json::json!({ "text": "hi" }));
    }

    #[test]
    fn speech_request_parses_with_optional_fields_defaulted() {
        let req: AudioSpeechRequest =
            serde_json::from_str(r#"{"model":"m","input":"hello"}"#).expect("parses");
        assert_eq!(req.model, "m");
        assert_eq!(req.input, "hello");
        assert!(req.voice.is_none());
        assert!(req.response_format.is_none());
        assert!(req.speed.is_none());
    }
}
