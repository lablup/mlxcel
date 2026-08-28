# OpenAI audio API (`/v1/audio`)

`mlxcel serve` and `mlxcel-server` expose the three OpenAI-compatible audio endpoints: text-to-speech (`/v1/audio/speech`), transcription (`/v1/audio/transcriptions`), and translation (`/v1/audio/translations`).

**Current status.** All three routes are mounted and functional. `/v1/audio/transcriptions` is the one route llama-server also mounts, so it carries llama-server's semantics and is served by the loaded chat model when that model takes audio, falling back to the Whisper provider otherwise (issue #1446). `/v1/audio/translations` is mlxcel's own and is served by the Whisper provider: pass a Whisper checkpoint with `-m` and the STT slot is populated automatically. Text-to-speech (`/v1/audio/speech`) is served by the Kokoro-82M provider: pass a Kokoro checkpoint with `-m` and the TTS slot is populated automatically. Any route whose model kind is not loaded returns 501 after the request is fully parsed.

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

This is the one audio route llama-server also mounts, so it carries **llama-server b10621's semantics**, not mlxcel's (issue #1446). Read [`llama-server-compat.md`](llama-server-compat.md) for the full contract and the divergences that remain; this section is the operator-facing summary.

**Which model transcribes.** Whichever of these the server has, in this order:

1. **The loaded chat model**, when it can take audio (`gemma3n`, the omni families, and any other checkpoint whose towers accept an `input_audio` part). This is what llama-server does, and it is the only server shape llama-server can express.
2. **A dedicated Whisper worker**, when `-m` named a Whisper checkpoint. That shape has no chat model at all and is mlxcel's own; llama-server has no equivalent.

With neither, the route answers `501 not_supported_error` with `The current model does not support audio input.`

**Request body (`multipart/form-data`):**

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `file` | file part | yes | Raw WAV audio bytes. Only RIFF/WAVE is decoded; other containers return 400. A repeated `file` part keeps the last one. |
| `model` | text | no | Carried and ignored, as upstream carries it into the chat body. |
| `prompt` | text | no | Replaces the ASR prompt. Default: `Transcribe audio to text`. Honored on the chat-model path; the Whisper worker ignores it. |
| `language` | text | no | Appended to the prompt as `" (language: xx)"`, which is how upstream passes it. |
| `response_format` | text | no | `json` only (the default). Anything else returns 400. `text` and `verbose_json` live on `/v1/audio/translations`. |
| `temperature` | text | no | Parsed as `f32`; a value that does not parse returns 400. |
| `max_tokens` | text | no | Parsed as an integer; a value that does not parse returns 400. |
| `stream` | text | no | Compared against the literal `"true"`, because the form carries strings. |

Unknown text fields are carried and ignored. A repeated text field collapses into an array and falls back to that field's default, so a duplicated `prompt` is ignored and a duplicated `response_format` resolves to `json`.

**Success response** (`json`, the default). Note that this is the transcript-event shape llama-server emits, **not** OpenAI's classic `{"text": ...}` object:

```json
{"type":"transcript.text.done","text":"The quick brown fox jumps over the lazy dog.","usage":{"type":"tokens","input_tokens":206,"output_tokens":10,"total_tokens":216,"input_tokens_details":{"cached_tokens":0}}}
```

On the Whisper-worker path the `usage` counts are zeros: the worker reports no prompt or decoded token counts.

**Streamed response** (`stream=true`): `text/event-stream` carrying a `{"type":"transcript.text.delta","delta":"..."}` frame, the `done` frame above, and `data: [DONE]`. mlxcel emits the whole transcript in one delta rather than incrementally.

**Limits**, all applied before a decoder sees the clip: at most 32 multipart parts, 25 MiB per part, and a WAV geometry read from the header (at most 192 kHz, 8 channels, 600 seconds). A `data` chunk declaring more audio than the file carries is clamped to what is present.

**Error responses:**

| Status | Condition |
|--------|-----------|
| 400 | Malformed multipart, too many parts, no `file` part (`No input file found for transcription`), a `response_format` other than `json` (`Only 'json' response_format is supported for transcription`), a non-numeric `temperature` or `max_tokens`, or a clip that is not WAV or exceeds a geometry bound. |
| 413 | Upload body exceeds 25 MiB. |
| 501 | Neither an audio-capable chat model nor an STT worker is loaded. Body: `{"error":{"type":"not_supported_error","message":"The current model does not support audio input."}}` |
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

An mlxcel-only route: llama-server does not mount it, so it keeps mlxcel's own semantics and is where the extra response formats live. Same multipart shape as the transcription route, with the loaded model asked to output English regardless of the source language, and served by the Whisper worker.

| `response_format` | Body |
|---|---|
| `json` (default) | `{"text":"..."}` |
| `text` | plain UTF-8 with `Content-Type: text/plain` |
| `verbose_json` | `{"text":"...","language":"...","duration":1.5}`, with `language` and `duration` omitted when the provider does not return them |

The `501` message names `stt` (the same underlying capability direction).

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

## Fault isolation

Each engine call on the audio worker runs under a `catch_unwind` boundary (`run_guarded` in `src/server/audio_worker.rs`), so a synthesis or transcription panic becomes a per-request `Inference` error and the worker thread keeps serving rather than taking down the server. Since issue #375 this holds in release builds too: the release profile uses `panic = "unwind"` (it formerly used `panic = "abort"`, which silently defeated the boundary in production). The core text-generation worker threads take the opposite, deliberate posture: an uncaught panic there means a broken invariant, so they log and `abort` the process for a supervised restart instead of unwinding (see ADR 0003).

The MLX C++ FFI exception path (issue #382) is contained on the Kokoro synthesis route as of PR #384: the alignment-expansion matmuls and the final PCM readback go through `try_matmul` and `try_array_to_raw_bytes`, which are declared `-> Result<..>` in the cxx bridge so an MLX throw becomes a per-request `Err` rather than `std::terminate`. PR #434 extended the same pattern to `try_conv2d` and `try_conv1d`, covering the Gemma 4 Conformer audio encoder's convolution path (issue #427), where input shapes derive from the runtime audio length and a shape fault would otherwise abort the server. PR #439 extended the same `try_conv2d`/`try_conv1d` coverage to the Nemotron-H Nano Omni Conformer/Parakeet audio encoder (issue #435): all four data-dependent conv calls in that encoder now route through the fallible variants so a shape fault returns a per-request `Err` instead of terminating the server. Whisper is unaffected because its tensor shapes are fixed and checked before the MLX call. Any `cxx` op that throws a non-`std::exception` type still terminates the process; no such throw exists on the current audio path.

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
