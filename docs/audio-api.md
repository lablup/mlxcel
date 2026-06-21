# OpenAI audio API (`/v1/audio`)

`mlxcel serve` and `mlxcel-server` expose the three OpenAI-compatible audio endpoints: text-to-speech (`/v1/audio/speech`), transcription (`/v1/audio/transcriptions`), and translation (`/v1/audio/translations`).

**Current status.** All three routes are mounted and functional. Speech-to-text (`/v1/audio/transcriptions` and `/v1/audio/translations`) is served by the Whisper provider: pass a Whisper checkpoint with `-m` and the STT slot is populated automatically. Text-to-speech (`/v1/audio/speech`) is served by the Kokoro-82M provider: pass a Kokoro checkpoint with `-m` and the TTS slot is populated automatically. Any route whose model kind is not loaded returns 501 after the request is fully parsed.

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
| `src/server/audio_worker.rs` | `AudioWorker` and `AudioEngine`: dedicated MLX-owning thread that loads and runs the audio model. |
| `src/server/whisper_stt.rs` | `WhisperSttProvider`: wires the WAV reader and Whisper front-end to the `AudioModelProvider` seam. |
| `src/server/kokoro_tts.rs` | `KokoroTtsProvider`: wires the g2p front-end and Kokoro acoustic model to the `AudioModelProvider` seam. |
| `src/server/routes/audio.rs` | HTTP handlers, multipart parser, format resolution, binary/JSON response builders. |
| `src/server/types/request.rs` | `AudioSpeechRequest`, `AudioTranscriptionRequest` (schema reference). |
| `src/server/types/response.rs` | `AudioTranscriptionResponse`, `ErrorResponse` (including `not_implemented`). |
| `src/audio/wav_writer.rs` | `encode_wav_pcm16`: `f32` PCM samples to RIFF WAV bytes. |
| `src/audio/whisper_mel.rs` | Log-mel front-end: STFT, Slaney mel filterbank, normalization, and 16 kHz resampler. |
| `src/server/app.rs` | Route registration and `AUDIO_MAX_UPLOAD_BYTES` (25 MiB) body-limit layer. |

## Whisper STT setup

Pass a Whisper checkpoint directory to `-m`. The server detects `model_type: "whisper"` in `config.json` and populates the STT slot via `WhisperSttProvider`, which owns a dedicated worker thread for all MLX graph evaluation.

```sh
mlxcel-server -m models/whisper-base
```

Both the native MLX key layout and the HuggingFace `WhisperForConditionalGeneration` layout load without conversion. The checkpoint directory must contain `config.json`, one or more SafeTensors weight files, and `tokenizer.json`.

Loading a Whisper checkpoint occupies the audio slot only; the server does not serve `/v1/chat/completions` or generation requests from that process. Chat and STT are separate server instances.

**Supported audio input.** The `file` part must be a WAV file. Audio is decoded with the shared WAV reader, converted to mono, and resampled to 16 kHz before the log-mel front-end. Other container formats (MP3, FLAC, etc.) are not yet supported; the WAV reader returns an error for non-WAV input and the route returns 400.

**Long audio.** Audio longer than 30 seconds is split into consecutive 30-second windows, each transcribed independently. The results are concatenated in order. Word-level timestamps, segment-level timestamps, and VAD-gated chunking are follow-ups.

**Current limitations.** Non-quantized (fp16/f32) checkpoints only; quantized Whisper weights are not yet supported. Greedy decoding only; beam search is a follow-up.

## Kokoro TTS setup

Pass a Kokoro-82M checkpoint directory to `-m`. The server detects the checkpoint by the `istftnet` config block in `config.json` or by the presence of `kokoro-v1_0.safetensors`, and populates the TTS slot via `KokoroTtsProvider`, which owns a dedicated worker thread for all MLX graph evaluation.

```sh
mlxcel-server -m models/kokoro-82m
```

The checkpoint directory must contain `config.json` (with `vocab` and architecture blocks), `kokoro-v1_0.safetensors`, and a `voices/` subdirectory of per-voice safetensors packs. The Kokoro checkpoint from Hugging Face (`hexgrad/Kokoro-82M`) ships 54 voice packs.

Loading a Kokoro checkpoint occupies the audio slot only; the server does not serve `/v1/chat/completions` or generation requests from that process. TTS and text generation are separate server instances.

**Voices.** The `voice` field in the request selects a pack from `voices/<name>.safetensors`. Available voice names are the file stems (e.g. `af_heart`, `bm_lewis`). The default is `af_heart`. A requested voice that does not exist or contains unsafe characters (anything outside `[A-Za-z0-9_-]`) falls back silently to `af_heart`.

**Language scope.** The built-in g2p front-end is American English only. Non-English voices in the checkpoint load and synthesize, but phonemes still come from the English front-end, so pronunciation quality for non-English text is limited.

**Input length.** Input text is capped at 4096 characters before g2p runs; longer inputs are truncated (not rejected), so well-formed long-ish requests still synthesize. The acoustic model processes at most 510 phoneme tokens; phonemes beyond that are dropped.

**Current limitations.** `response_format` is `wav` only; other containers are a follow-up. The g2p front-end covers American English; per-language phonemizers are future work. Quantized Kokoro checkpoints are not yet tested.

## POST /v1/audio/speech

**Request body (JSON):**

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `model` | string | yes | Identifier of the TTS model. Ignored at the route layer; any string is accepted. |
| `input` | string | yes | Text to synthesize. Capped at 4096 characters; longer inputs are truncated before g2p. |
| `voice` | string | no | Kokoro voice name (e.g. `af_heart`, `bm_lewis`). Defaults to `af_heart`; unknown names fall back to the default. |
| `response_format` | string | no | Output container. Only `wav` is supported; omitting the field defaults to `wav`. Any other value returns 400. |
| `speed` | float | no | Duration scale factor. Values larger than 1.0 produce shorter (faster) audio; values smaller than 1.0 produce longer (slower) audio. Non-positive or non-finite values default to 1.0. |

**Success response (200):**

Content-Type: `audio/wav`. Content-Disposition: `attachment; filename=speech.wav`. The body is a 44-byte RIFF WAV header followed by 16-bit little-endian PCM samples at the sample rate the provider returns.

**Error responses:**

| Status | Condition |
|--------|-----------|
| 400 | Unsupported `response_format` (any value other than `wav` or absent). |
| 501 | No TTS model is registered. Body: `{"error":{"type":"not_implemented","message":"audio model kind not loaded: tts"}}` |
| 503 | All slots are busy: either the generation batch queue or the bounded audio worker queue (`--audio-queue-depth`) is full. |
| 504 | The audio worker did not reply within the per-request timeout (`--audio-request-timeout-secs`). |

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
| `file` | file part | yes | Raw WAV audio bytes. The current Whisper provider decodes WAV only; other container formats return 400. |
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
| 503 | All slots are busy: either the generation batch queue or the bounded audio worker queue (`--audio-queue-depth`) is full. |
| 504 | The audio worker did not reply within the per-request timeout (`--audio-request-timeout-secs`). |

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

## Request validation order

The 501 response is returned **after** the request is fully parsed. This means:

- A malformed multipart body returns 400, not 501, even when no model is loaded.
- A `response_format` typo in a speech request returns 400 when a TTS model is loaded but 501 when none is loaded (the format is resolved after the model check).
- The `file` field is required for transcriptions/translations; its absence returns 400 regardless of model state.

This ordering lets callers distinguish broken requests from absent models without needing a loaded model.

## Queue bound and per-request timeout

All audio requests serialize through one dedicated MLX-owning worker thread (the model weights are thread-affine, so the thread that loads them must also run them). Two server knobs bound that path so a burst or a stuck request cannot degrade availability:

- **Queue bound** (`--audio-queue-depth`, env `MLXCEL_AUDIO_QUEUE_DEPTH`, default `8`). The worker's command channel is bounded. At most this many requests may wait behind the one in flight; the next request is rejected with a structured `503` ("All slots are busy") rather than queueing without bound. This caps queued memory: each queued speech-to-text command holds up to the 25 MiB per-request payload, so the default of `8` caps queued payload at roughly 200 MiB plus the one in flight. A `0` is clamped to at least one queued command (a zero-capacity rendezvous channel is not the admission behavior we want).

- **Per-request timeout** (`--audio-request-timeout-secs`, env `MLXCEL_AUDIO_REQUEST_TIMEOUT_SECS`, default `120`). A caller blocks on the worker reply for at most this long, then returns a structured `504`. The timeout frees the caller's blocking thread; it does not cancel the in-flight model work on the worker (a single worker can only safely process one request at a time). When the worker eventually finishes, its reply is dropped silently. A `0` falls back to the default rather than timing out instantly.

Both knobs apply to the shared worker, so they cover the STT (Whisper) and TTS (Kokoro) paths together.

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
