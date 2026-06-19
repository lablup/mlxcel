# OpenAI audio API (`/v1/audio`)

`mlxcel serve` and `mlxcel-server` expose the three OpenAI-compatible audio endpoints: text-to-speech (`/v1/audio/speech`), transcription (`/v1/audio/transcriptions`), and translation (`/v1/audio/translations`).

**Phase 1 status.** The HTTP plumbing (request parsing, multipart handling, binary response framing) is complete and all three routes are mounted. Until an audio model is registered via `AppState::with_audio_model`, every request returns a structured `501 Not Implemented` with error type `not_implemented`. This is by design: Phase 1 establishes the transport boundary; Phase 2 wires in a speech model.

## Implemented endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/audio/speech` | Text-to-speech: JSON body in, binary WAV out. |
| POST | `/v1/audio/transcriptions` | Speech-to-text: `multipart/form-data` in, JSON text out. |
| POST | `/v1/audio/translations` | Speech-to-text with English output: same multipart shape as transcriptions. |

Alias paths without the `/v1` prefix are also mounted: `/audio/speech`, `/audio/transcriptions`, `/audio/translations`.

## Implementation source map

| Module | Responsibility |
|--------|----------------|
| `src/server/audio_model.rs` | `AudioModelProvider` trait, input/output types, `AudioModelKind`, `AudioModelError`. |
| `src/server/routes/audio.rs` | HTTP handlers, multipart parser, format resolution, binary/JSON response builders. |
| `src/server/types/request.rs` | `AudioSpeechRequest`, `AudioTranscriptionRequest` (schema reference). |
| `src/server/types/response.rs` | `AudioTranscriptionResponse`, `ErrorResponse` (including `not_implemented`). |
| `src/audio/wav_writer.rs` | `encode_wav_pcm16`: `f32` PCM samples to RIFF WAV bytes. |
| `src/server/app.rs` | Route registration and `AUDIO_MAX_UPLOAD_BYTES` (25 MiB) body-limit layer. |

## POST /v1/audio/speech

**Request body (JSON):**

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `model` | string | yes | Identifier of the TTS model. Ignored until a model is loaded; any string is accepted. |
| `input` | string | yes | Text to synthesize. |
| `voice` | string | no | Named voice. Model-specific; forwarded as-is to the provider. |
| `response_format` | string | no | Output container. Only `wav` is supported today; omitting the field defaults to `wav`. |
| `speed` | float | no | Playback-speed multiplier. Forwarded to the provider; no range is enforced by the route layer. |

**Success response (200):**

Content-Type: `audio/wav`. Content-Disposition: `attachment; filename=speech.wav`. The body is a 44-byte RIFF WAV header followed by 16-bit little-endian PCM samples at the sample rate the provider returns.

**Error responses:**

| Status | Condition |
|--------|-----------|
| 400 | Unsupported `response_format` (any value other than `wav` or absent). |
| 501 | No TTS model is registered. Body: `{"error":{"type":"not_implemented","message":"audio model kind not loaded: tts"}}` |
| 503 | All inference slots are busy. |

**Example (curl):**

```sh
curl -s -X POST http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"my-tts","input":"Hello world"}' \
  --output speech.wav
```

## POST /v1/audio/transcriptions

**Request body (`multipart/form-data`):**

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `file` | file part | yes | Raw audio bytes. Any container the loaded provider can decode. |
| `model` | text | no | STT model identifier. Forwarded to the provider. |
| `language` | text | no | ISO-639-1 source-language hint (`en`, `ko`, etc.). |
| `response_format` | text | no | `json` (default), `text`, or `verbose_json`. Any other non-empty value returns 400. |
| `temperature` | text | no | Sampling temperature. Parsed as `f32`; non-numeric values return 400. |

Unknown multipart parts are drained and ignored. The upload body limit is 25 MiB; larger uploads return 413.

**Success responses:**

- `json` (default): `{"text":"..."}` with `Content-Type: application/json`.
- `text`: plain UTF-8 text with `Content-Type: text/plain`.
- `verbose_json`: `{"text":"...","language":"...","duration":1.5}` with `Content-Type: application/json`. The `language` and `duration` fields are omitted when the provider does not return them.

**Error responses:**

| Status | Condition |
|--------|-----------|
| 400 | Malformed multipart, non-numeric `temperature`, or unsupported `response_format`. |
| 413 | Upload body exceeds 25 MiB. |
| 501 | No STT model is registered. Body: `{"error":{"type":"not_implemented","message":"audio model kind not loaded: stt"}}` |
| 503 | All inference slots are busy. |

**Example (curl):**

```sh
curl -s -X POST http://localhost:8080/v1/audio/transcriptions \
  -F file=@recording.wav \
  -F model=my-stt \
  -F language=en \
  -F response_format=json
```

## POST /v1/audio/translations

Same multipart shape and field semantics as `/v1/audio/transcriptions`. The difference is that the loaded model is asked to output text in English regardless of the source language. The `501` message names `stt` (the same underlying capability direction).

## Phase 1 boundaries

The 501 response is returned **after** the request is fully parsed. This means:

- A malformed multipart body returns 400, not 501, even when no model is loaded.
- A `response_format` typo in a speech request returns 400 when a TTS model is loaded but 501 when none is loaded (the format is resolved after the model check).
- The `file` field is required for transcriptions/translations; its absence returns 400 regardless of model state.

This ordering lets callers distinguish broken requests from absent models without needing a loaded model.

## Adding an audio model provider

Implement `crate::server::AudioModelProvider` and register it at startup:

```rust
let state = AppState::new(...)
    .with_audio_model(Some(Arc::new(MyProvider::new(...))));
```

A provider overrides only the direction(s) it supports (`transcribe` or `synthesize`) and reports capability via `supports(AudioModelKind)`. The default implementations for the unimplemented direction return `AudioModelError::KindNotLoaded`, which the route layer maps back to a 501.

## WAV encoding

The TTS route encodes `f32` PCM samples as a standard 44-byte RIFF WAV with 16-bit little-endian PCM data (`encode_wav_pcm16` in `src/audio/wav_writer.rs`). The encoder:

- Clamps values outside `[-1.0, 1.0]` to the `i16` range (no wrapping).
- Maps `NaN` to zero (silence).
- Accepts multi-channel audio when samples are interleaved (`L, R, L, R, ...`).
- Truncates at the maximum expressible 32-bit RIFF payload rather than producing a corrupt header.

The output round-trips through the existing WAV reader (`load_wav_from_bytes`) within one 16-bit quantization step.
