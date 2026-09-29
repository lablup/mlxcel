# Technical Report: PR #2041 - /v1/realtime WebSocket for Nemotron VoiceChat

**Date**: 2026-09-29

**Status**: Implemented and validated on the 4-bit checkpoint (file client); live microphone exchange not verified; pending merge.

**Languages**: Rust

**Risk Level**: Medium. It touches the shared HTTP serving path: hyper connections now serve upgrades, and startup has a new model-type branch. It adds the first long-lived, stateful connection to the server.

## Executive Summary

This PR serves the Nemotron VoiceChat online session (#1378) over a WebSocket at `/v1/realtime`. The protocol uses JSON control messages, base64 PCM16 in both directions, and one active session at a time. All MLX work runs on one dedicated engine thread. The PR also ships a WAV file client and a live microphone/speaker client.

On the 4-bit checkpoint, a WAV driven through the socket produces the same transcript, answer and audio as `mlxcel generate --stream`. The only difference is the last-bit PCM16 scaling. It completes epic #1372's serving surface.

## 1. Problem Statement

mlxcel's audio endpoints were request/response calls, each served by a worker that handles one command at a time. A duplex speech model needs a connection that stays open while audio flows both ways. The session state (six cache families plus the RNG) lives on the MLX thread for the whole conversation, and the async runtime must never run MLX code.

## 2. Change Summary

- **Engine** (`server::realtime_engine`, `server::realtime_session`):
  - One thread loads the checkpoint and owns the active session.
  - Commands (`open`, `push`, `flush`, `cancel`, `close`) arrive over a FIFO channel and answer on one-shot replies.
  - `try_reserve` admits one connection.
  - A panic inside a command is contained, and the engine keeps serving.
  - `close` clears the MLX buffer cache once per session, not per frame.
- **Protocol** (`server::realtime_protocol`): the reference server's message and event set, PCM16 base64 codecs, and event IDs.
- **Route** (`server::routes::realtime`):
  - The socket loop slices appends into 1280-sample pieces and forwards each piece's events before sending the next piece.
  - A second connection gets `server_busy` and close code 1013.
  - The reservation is released on every exit path, including drop.
  - The route sits behind `--api-key` and `--api-prefix`.
- **Startup:** a VoiceChat checkpoint spawns the engine. The chat worker, scheduler and text warmup are skipped. The chat endpoints return 501 with a message naming the socket.
- **Clients:**
  - `examples/voicechat_file_client.rs`.
  - `examples/voicechat_microphone.rs`, which uses cpal behind the optional `voicechat-mic` feature.

## 3. Technical Decisions

### A dedicated engine thread and a FIFO command channel

MLX evaluation is thread-affine, and the session borrows the model. The engine therefore keeps both on one thread. The async socket task only sends commands and awaits replies.

Because the channel is FIFO, a disconnect handler can queue `close` without waiting for it: the next connection's `open` always runs after the close. Reservation release does not depend on the dropped task reaching an await point.

### Bounding the single slot

One session per process makes the slot itself the resource an abusive or broken client would hold, so the review pass added four limits:

| Limit | Value | What it prevents |
|---|---|---|
| Session length | 600 s of audio (client values are clamped) | caches growing without bound |
| WebSocket message and frame size | 1 MiB | one huge append consuming memory and blocking the engine |
| Time to configure | 30 s | a connection that never configures holding the slot |
| Idle time after configuring | 120 s; pings do not count, close code 1008 | an idle or ping-only client holding the slot |

`/health` also checks that the engine thread is still alive.

### Reference over issue body

Where the issue table and mlx-vlm's `realtime.py` differ, the port follows the reference:
- Bad base64 or a bad sample rate returns `inference_error`.
- Commit and cancel close with code 1000.
- `session.model` is optional, because the server runs one model.

Whole-number floats are accepted for `sample_rate` and `seed`, as Python's `int()` accepts them.

### The microphone client stays out of every default build

cpal pulls in platform audio bindings: CoreAudio on macOS, and ALSA headers on Linux. It is an optional dependency enabled only by `voicechat-mic`, and the example declares it in `required-features`. The library, the server, CI and the test targets never compile it.

## 4. Validation

Setup: 4-bit checkpoint, `mlxcel-server`, the file client, "What is the capital of France?" plus 3 s of silence, seed 0.
- **Events:** the transcript deltas build "What is the capital of France" and the text deltas build "The capital of France is Paris.". There are 61 audio deltas, each 1764 samples, with contiguous frame indices, followed by `response.done`.
- **Audio vs `--stream`:** the socket audio has the same 107604 samples as `--stream`, and every sample is consistent with one float waveform. The integer WAVs differ only because the CLI WAV writer scales by 32768 and the wire format by 32767.
- **Busy/free:** a second connection gets `server_busy` and close 1013. After the first session ends, the next connection configures normally.
- **Other endpoints:** chat returns 501, and `/health` is ok.

The local gate:
- fmt, and clippy (lib and tests, the examples, and the mic feature)
- `cargo test --lib server::` (3268 passed)
- `tests/realtime_ws.rs` (9 passed against a fake engine)
- the contract tests

**Not verified:** a live microphone exchange. In this non-interactive agent session, opening an input stream blocked or failed, which is consistent with the process lacking microphone permission. It needs a manual run from a terminal that has microphone access.

## 5. Known Limitations

- **CORS:** `--cors-origins` does not restrict the WebSocket upgrade.
- **One session per process:** as specified.
- **No echo cancellation:** use headphones with the microphone client.
- **Real-time factor:** not measured here. It belongs in the placeholder table in `docs/nemotron-voicechat.md`.
