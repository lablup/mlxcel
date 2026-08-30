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

//! b10621 semantics for the shared `/v1/audio/transcriptions` route
//! (issue #1446).
//!
//! # What upstream actually does
//!
//! b10621 has no speech-to-text model. `POST /v1/audio/transcriptions` is a
//! *translation layer* over `/v1/chat/completions`: the handler refuses unless
//! the loaded chat model takes audio, converts the multipart form into a chat
//! request whose single user message carries the ASR prompt plus the uploaded
//! clip, and renders the completion as an ASR event.
//! [`convert_transcriptions_to_chatcmpl`](https://github.com/ggml-org/llama.cpp/blob/c1d0e7a004015f23bc0233470b747b596f29b264/tools/server/server-chat.cpp)
//! is the whole of it:
//!
//! ```text
//! if (!has_mtmd || !allow_audio)  -> 501 "The current model does not support audio input."
//! files["file"]                   -> the clip, or 400 "No input file found for transcription"
//! prompt          = body["prompt"]          ?? asr_preset.user   // "Transcribe audio to text"
//! language        = body["language"]        ?? ""
//! response_format = body["response_format"] ?? "json"
//! if (response_format != "json")  -> 400 "Only 'json' response_format is supported for transcription"
//! if (!language.empty()) prompt += " (language: <language>)"
//! prompt += media_marker
//! messages = [ {system: asr_preset.system} if non-empty, {user: prompt} ]
//! chatcmpl = body with messages replaced; stream / max_tokens / temperature retyped from strings
//! ```
//!
//! # Why mlxcel's STT worker is not an alias of that route
//!
//! Measured on this tree, not assumed. mlxcel's `/v1/audio/transcriptions` is
//! served by [`crate::server::whisper_stt::WhisperSttProvider`], which is
//! populated only when `-m` names a Whisper checkpoint, and that same `-m`
//! leaves the chat worker unloaded. So the two servers are mutually exclusive
//! shapes:
//!
//! - `mlxcel-server -m models/mlx/whisper-tiny` transcribes, and answers
//!   `/v1/chat/completions` with a 503.
//! - `mlxcel-server -m models/mlx/gemma3n-e2b-4bit` answers chat, and answered
//!   `/v1/audio/transcriptions` with `501 audio model kind not loaded: stt`.
//!
//! The second command line is the *only* shape b10621 can express, and it was
//! the one that did not work. A `llama-server` deployment that transcribes
//! through its loaded model therefore had no mlxcel equivalent, however closely
//! the multipart field set matched. Route-name overlap was not compatibility.
//!
//! The same tree also shows the translation is real: posting the clip as an
//! `input_audio` content part to `gemma3n-e2b-4bit` with the prompt
//! "Transcribe this audio." returns "The quick brown fox jumps over the lazy
//! dog.", which is the clip. So this module routes the compatibility request
//! through the loaded chat model exactly as upstream does, and keeps the
//! Whisper worker as the fallback for the Whisper-server shape b10621 cannot
//! express at all.
//!
//! # Limits
//!
//! Every bound is applied before the clip reaches a model: the multipart part
//! count, the per-part upload size, and the WAV geometry (sample rate, channel
//! count, decoded duration) read from the 44-byte header rather than from a
//! decode. A malformed or truncated header is refused there, so an oversized or
//! amplifying upload costs a header parse rather than a decode.

use axum::extract::{Multipart, multipart::Field};
use serde_json::{Map, Value, json};

use crate::server::types::{ErrorResponse, request as req};

/// b10621's default ASR user prompt (`common_chat_get_asr_prompt`).
///
/// Upstream picks a per-template preset, but the only template that overrides
/// it is LFM2, whose Jinja source carries `<|tool_list_start|>`. mlxcel renders
/// the checkpoint's own template and has no LFM2 special case, so the default
/// preset is what every request uses.
pub(crate) const ASR_USER_PROMPT: &str = "Transcribe audio to text";

/// Upstream's wording for a form with no `file` part.
pub(crate) const NO_INPUT_FILE_MESSAGE: &str = "No input file found for transcription";

/// Upstream's wording for a `response_format` other than `json`.
pub(crate) const ONLY_JSON_MESSAGE: &str =
    "Only 'json' response_format is supported for transcription";

/// Upstream's wording for a model that cannot take audio.
pub(crate) const NO_AUDIO_SUPPORT_MESSAGE: &str = "The current model does not support audio input.";

/// Largest number of multipart parts a transcription form may carry.
///
/// b10621 has no such bound; cpp-httplib buffers the whole form and the field
/// map grows with it. mlxcel refuses past this count so a form of ten thousand
/// empty parts cannot spend the request budget on map insertions. Well above
/// the seven fields the route reads.
pub(crate) const MAX_MULTIPART_PARTS: usize = 32;

/// Largest single part a transcription form may carry, matching the 25 MiB
/// body limit `app.rs` puts on the audio routes.
///
/// Enforced per part as well as per body so one oversized part is refused on
/// its own terms rather than only when the whole body overflows.
pub(crate) const MAX_UPLOAD_BYTES: usize = 25 * 1024 * 1024;

/// Longest clip accepted, read from the WAV header before any decode.
///
/// Ten minutes at 16 kHz mono is 19 MB of PCM, inside the upload bound; the
/// cap exists for the amplifying case, where a header declares hours of audio
/// that a decoder would have to materialise.
pub(crate) const MAX_DURATION_SECONDS: f64 = 600.0;

/// Highest sample rate accepted. Everything is resampled to 16 kHz downstream,
/// so a higher rate only inflates the decode.
pub(crate) const MAX_SAMPLE_RATE: u32 = 192_000;

/// Highest channel count accepted. Everything is mixed to mono downstream.
pub(crate) const MAX_CHANNELS: u16 = 8;

/// One uploaded file part.
#[derive(Debug, Clone)]
pub(crate) struct UploadedFile {
    pub(crate) bytes: Vec<u8>,
    pub(crate) filename: Option<String>,
}

/// A parsed transcription form, in b10621's own representation.
///
/// Upstream turns the multipart text fields into one JSON object before the
/// handler sees them, with a duplicate key collapsing into an array
/// (`server-http.cpp`). Reproducing that here is what makes the duplicate-part
/// behavior match: a duplicated `prompt` becomes an array, upstream's
/// `json_value<std::string>` then fails its type check and falls back to the
/// default, and the duplicate is ignored rather than winning or erroring.
#[derive(Debug, Clone, Default)]
pub(crate) struct TranscriptionForm {
    /// The `file` part. A duplicate `file` part overwrites the earlier one,
    /// because upstream copies the parts into a `std::map` keyed by part name.
    pub(crate) file: Option<UploadedFile>,
    /// Every text field, including the ones this route does not read.
    pub(crate) fields: Map<String, Value>,
}

impl TranscriptionForm {
    /// `json_value<std::string>(body, key, default)`: the value when it is a
    /// string, the default when it is missing, null, or another type.
    #[must_use]
    pub(crate) fn text(&self, key: &str) -> Option<&str> {
        self.fields.get(key).and_then(Value::as_str)
    }

    /// Insert one text field with upstream's duplicate-to-array rule.
    fn insert_text(&mut self, key: String, value: String) {
        match self.fields.get_mut(&key) {
            Some(Value::Array(existing)) => existing.push(Value::String(value)),
            Some(slot) => {
                let previous = std::mem::replace(slot, Value::Null);
                *slot = Value::Array(vec![previous, Value::String(value)]);
            }
            None => {
                self.fields.insert(key, Value::String(value));
            }
        }
    }
}

/// Parse the multipart body into b10621's form representation.
///
/// # Errors
///
/// Returns a 400 for a malformed body, a part that exceeds
/// [`MAX_UPLOAD_BYTES`], a non-UTF-8 text field, or more than
/// [`MAX_MULTIPART_PARTS`] parts.
pub(crate) async fn parse_multipart(
    mut multipart: Multipart,
) -> Result<TranscriptionForm, ErrorResponse> {
    let mut form = TranscriptionForm::default();
    let mut parts = 0usize;
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
        parts += 1;
        if parts > MAX_MULTIPART_PARTS {
            return Err(ErrorResponse::new(
                format!("multipart form has more than {MAX_MULTIPART_PARTS} parts"),
                "invalid_request_error",
            ));
        }

        let name = field.name().map(str::to_string).unwrap_or_default();
        // A part with a filename is a file upload; everything else is a text
        // field, which is exactly how cpp-httplib splits `form.files` from
        // `form.fields`.
        let filename = field.file_name().map(str::to_string);
        if filename.is_some() {
            let bytes = read_part_bytes(field, &name).await?;
            if name == "file" {
                form.file = Some(UploadedFile { bytes, filename });
            }
            continue;
        }
        let value = read_text_field(field, &name).await?;
        form.insert_text(name, value);
    }
    Ok(form)
}

/// Read one file part under the per-part byte cap.
async fn read_part_bytes(field: Field<'_>, name: &str) -> Result<Vec<u8>, ErrorResponse> {
    let bytes = field.bytes().await.map_err(|err| {
        ErrorResponse::new(
            format!("failed to read '{name}' part: {err}"),
            "invalid_request_error",
        )
    })?;
    if bytes.len() > MAX_UPLOAD_BYTES {
        return Err(ErrorResponse::new(
            format!(
                "'{name}' part is {} bytes, maximum is {MAX_UPLOAD_BYTES} bytes",
                bytes.len()
            ),
            "invalid_request_error",
        ));
    }
    Ok(bytes.to_vec())
}

/// Read one text field as UTF-8.
async fn read_text_field(field: Field<'_>, name: &str) -> Result<String, ErrorResponse> {
    field.text().await.map_err(|err| {
        ErrorResponse::new(
            format!("failed to read '{name}' field: {err}"),
            "invalid_request_error",
        )
    })
}

/// The WAV geometry a header declares, used to bound the decode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct WavGeometry {
    pub(crate) sample_rate: u32,
    pub(crate) channels: u16,
    pub(crate) bits_per_sample: u16,
    pub(crate) data_bytes: u64,
    pub(crate) duration_seconds: f64,
}

/// Read a RIFF/WAVE header without decoding the samples.
///
/// Only the `fmt ` and `data` chunk headers are walked, so a file declaring
/// hours of audio costs a few dozen bytes of parsing rather than a decode. The
/// declared `data` size is clamped to the bytes actually present, so a
/// truncated file reports the duration it really carries and an inflated header
/// cannot claim more than it delivers.
///
/// # Errors
///
/// Returns a description of the first structural problem.
pub(crate) fn probe_wav(bytes: &[u8]) -> Result<WavGeometry, String> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".to_owned());
    }
    let mut offset = 12usize;
    let mut fmt: Option<(u32, u16, u16)> = None;
    let mut data_bytes: Option<u64> = None;
    while offset + 8 <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let size = u32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]) as u64;
        let body = offset + 8;
        if id == b"fmt " {
            if body + 16 > bytes.len() {
                return Err("truncated fmt chunk".to_owned());
            }
            let channels = u16::from_le_bytes([bytes[body + 2], bytes[body + 3]]);
            let sample_rate = u32::from_le_bytes([
                bytes[body + 4],
                bytes[body + 5],
                bytes[body + 6],
                bytes[body + 7],
            ]);
            let bits = u16::from_le_bytes([bytes[body + 14], bytes[body + 15]]);
            fmt = Some((sample_rate, channels, bits));
        } else if id == b"data" {
            // Clamp to what the file actually holds: a header may declare more.
            let present = bytes.len().saturating_sub(body) as u64;
            data_bytes = Some(size.min(present));
            break;
        }
        // Chunks are word-aligned.
        offset = body.saturating_add(size as usize + (size as usize & 1));
    }
    let (sample_rate, channels, bits_per_sample) = fmt.ok_or("missing fmt chunk")?;
    let data_bytes = data_bytes.ok_or("missing data chunk")?;
    if sample_rate == 0 || channels == 0 || bits_per_sample == 0 {
        return Err("degenerate fmt chunk".to_owned());
    }
    let frame_bytes = u64::from(channels) * u64::from(bits_per_sample).div_ceil(8);
    if frame_bytes == 0 {
        return Err("degenerate fmt chunk".to_owned());
    }
    let frames = data_bytes / frame_bytes;
    Ok(WavGeometry {
        sample_rate,
        channels,
        bits_per_sample,
        data_bytes,
        duration_seconds: frames as f64 / f64::from(sample_rate),
    })
}

/// Refuse a clip whose geometry exceeds the configured bounds.
///
/// # Errors
///
/// Returns a 400 naming the bound that was exceeded, or the structural problem
/// that stopped the header from parsing at all.
pub(crate) fn ensure_clip_within_limits(bytes: &[u8]) -> Result<WavGeometry, ErrorResponse> {
    if bytes.len() > MAX_UPLOAD_BYTES {
        return Err(ErrorResponse::new(
            format!(
                "audio upload is {} bytes, maximum is {MAX_UPLOAD_BYTES} bytes",
                bytes.len()
            ),
            "invalid_request_error",
        ));
    }
    // b10621's mtmd audio front-end takes wav, mp3 and flac, so the geometry
    // bound below runs against whichever of the three the bytes actually are
    // (#1446). A compressed container's header is read without decoding, the
    // same property the WAV probe has, so an amplifying header still costs a
    // header parse. A container outside the set falls through to the WAV probe
    // and keeps its `not a RIFF/WAVE file` diagnostic.
    let geometry = match crate::audio::preprocessing::container::probe_compressed(bytes) {
        Some(result) => {
            let compressed = result.map_err(|reason| {
                ErrorResponse::new(
                    format!("failed to read the uploaded audio: {reason}"),
                    "invalid_request_error",
                )
            })?;
            WavGeometry {
                sample_rate: compressed.sample_rate,
                channels: compressed.channels,
                // Not carried by a compressed container and not used by any
                // caller of this function; the decoder reports the real sample
                // format.
                bits_per_sample: 0,
                data_bytes: bytes.len() as u64,
                // An MPEG stream without a Xing header declares no length, and
                // 0.0 is the honest reading of "nothing declared": the bound
                // below cannot refuse what was never stated, and the decode
                // loop's own per-clip frame cap is what bounds it.
                duration_seconds: compressed.duration_seconds.unwrap_or(0.0),
            }
        }
        None => probe_wav(bytes).map_err(|reason| {
            ErrorResponse::new(
                format!("failed to read the uploaded audio: {reason}"),
                "invalid_request_error",
            )
        })?,
    };
    if geometry.sample_rate > MAX_SAMPLE_RATE {
        return Err(ErrorResponse::new(
            format!(
                "audio sample rate {} Hz exceeds the maximum of {MAX_SAMPLE_RATE} Hz",
                geometry.sample_rate
            ),
            "invalid_request_error",
        ));
    }
    if geometry.channels > MAX_CHANNELS {
        return Err(ErrorResponse::new(
            format!(
                "audio has {} channels, maximum is {MAX_CHANNELS}",
                geometry.channels
            ),
            "invalid_request_error",
        ));
    }
    if geometry.duration_seconds > MAX_DURATION_SECONDS {
        return Err(ErrorResponse::new(
            format!(
                "audio is {:.1} seconds long, maximum is {MAX_DURATION_SECONDS} seconds",
                geometry.duration_seconds
            ),
            "invalid_request_error",
        ));
    }
    Ok(geometry)
}

/// Build upstream's ASR user prompt.
///
/// An empty or absent `prompt` falls back to the preset, and a non-empty
/// `language` is appended in upstream's own `" (language: %s)"` form rather
/// than being passed as a separate decoder parameter.
#[must_use]
pub(crate) fn asr_user_prompt(prompt: Option<&str>, language: Option<&str>) -> String {
    let mut text = match prompt.map(str::trim) {
        Some(value) if !value.is_empty() => value.to_owned(),
        _ => ASR_USER_PROMPT.to_owned(),
    };
    if let Some(language) = language.map(str::trim)
        && !language.is_empty()
    {
        text.push_str(&format!(" (language: {language})"));
    }
    text
}

/// Build the chat request the compatibility route dispatches.
///
/// The clip travels as an `input_audio` content part carrying raw base64, which
/// is the last branch of [`crate::server::media`]'s audio resolver, so no data
/// URI prefix is needed and the bytes are never re-encoded on the way in.
/// Upstream appends its media marker to the prompt text; mlxcel's chat template
/// inserts the audio placeholder itself, so the marker has no counterpart here.
#[must_use]
pub(crate) fn build_asr_chat_request(
    model: String,
    user_prompt: String,
    audio_base64: String,
    audio_format: String,
    temperature: Option<f32>,
    max_tokens: Option<usize>,
) -> req::ChatCompletionRequest {
    let params = req::SamplingParams {
        temperature,
        max_tokens,
        ..req::SamplingParams::default()
    };
    req::ChatCompletionRequest {
        model,
        messages: vec![req::Message {
            role: req::Role::User,
            content: req::MessageContent::Parts(vec![
                req::ContentPart::Text { text: user_prompt },
                req::ContentPart::InputAudio {
                    input_audio: req::InputAudio {
                        data: audio_base64,
                        format: audio_format,
                    },
                },
            ]),
            name: None,
            tool_call_id: None,
            reasoning: None,
            tool_calls: None,
        }],
        stream: false,
        stream_options: None,
        logprobs: None,
        top_logprobs: None,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        chat_template_kwargs: None,
        extra_body: None,
        prompt_cache_key: None,
        cache_prompt: None,
        user: None,
        reasoning_effort: None,
        extra_body_fields: Map::new(),
        response_format: None,
        params,
    }
}

/// Token counts for the ASR `usage` block.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct AsrUsage {
    pub(crate) input_tokens: u32,
    pub(crate) output_tokens: u32,
    pub(crate) cached_tokens: u32,
}

/// b10621's terminal ASR event, `to_json_oaicompat_asr`.
///
/// Note that this is **not** OpenAI's classic `{"text": ...}` transcription
/// object: upstream emits the newer transcript-event shape, and a client
/// reading `type` or `usage` sees it on both servers.
#[must_use]
pub(crate) fn transcription_done_event(text: &str, usage: AsrUsage) -> Value {
    json!({
        "type": "transcript.text.done",
        "text": text,
        "usage": {
            "type": "tokens",
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.input_tokens + usage.output_tokens,
            "input_tokens_details": { "cached_tokens": usage.cached_tokens },
        },
    })
}

/// b10621's incremental ASR event, `to_json_oaicompat_asr` on a partial result.
#[must_use]
pub(crate) fn transcription_delta_event(delta: &str) -> Value {
    json!({ "type": "transcript.text.delta", "delta": delta })
}

/// Read the compatibility `response_format`, which upstream restricts to
/// `json`.
///
/// # Errors
///
/// Returns upstream's own 400 for any other value. A duplicated part is an
/// array rather than a string, which upstream's `json_value` turns back into
/// the default, so it is accepted here too.
pub(crate) fn ensure_json_response_format(form: &TranscriptionForm) -> Result<(), ErrorResponse> {
    match form.text("response_format") {
        // Absent is upstream's default, and the comparison is against the exact
        // string, so `JSON` and an empty value are both refused as upstream
        // refuses them.
        None | Some("json") => Ok(()),
        Some(_) => Err(ErrorResponse::new(
            ONLY_JSON_MESSAGE,
            "invalid_request_error",
        )),
    }
}

/// Read a numeric text field the way upstream's `std::stof` / `std::stoul` do:
/// a value that does not parse throws `std::invalid_argument`, which upstream's
/// exception wrapper turns into a 400.
///
/// # Errors
///
/// Returns a 400 naming the field and the value.
pub(crate) fn parse_numeric_field<T: std::str::FromStr>(
    form: &TranscriptionForm,
    key: &str,
) -> Result<Option<T>, ErrorResponse> {
    let Some(raw) = form.text(key) else {
        return Ok(None);
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    trimmed.parse::<T>().map(Some).map_err(|_| {
        ErrorResponse::new(
            format!("invalid '{key}' value: {raw}"),
            "invalid_request_error",
        )
    })
}

/// True when the form asked for a streamed transcript.
///
/// The multipart body carries strings, so upstream compares against the literal
/// `"true"` rather than parsing a boolean vocabulary.
#[must_use]
pub(crate) fn wants_stream(form: &TranscriptionForm) -> bool {
    form.text("stream") == Some("true")
}

#[cfg(test)]
#[path = "transcription_compat_tests.rs"]
mod tests;
