# Technical Report: PR #2039 - Nemotron VoiceChat Cache-Aware Online Session

**Date**: 2026-09-29

**Status**: Implemented and validated on the 4-bit checkpoint; pending merge.

**Languages**: Rust

**Risk Level**: Low. The PR only adds code: streaming entry points beside the offline ones, a new session module, and CLI flags that only the VoiceChat path reads.

## Executive Summary

#2037 runs Nemotron VoiceChat over a whole utterance. A conversation needs bounded work per 80 ms instead. This PR adds `VoiceChatStreamingSession`, which buffers PCM chunks of any size into 1280-sample frames and advances every network one frame at a time from persistent state. It emits text, function, transcript and 1764-sample audio events tagged with the frame index. The session is the library surface for the `/v1/realtime` endpoint (#1376). From the CLI it runs as `mlxcel generate --audio ... --stream [--profile]`.

On the test utterance, every one of the 61 streamed frames has the same text id and the same 31 codes as the offline path.

## 1. Problem Statement

The offline loop recomputes log-mel and the full FastConformer pass over the whole waveform, and decodes the codec from full history. Its cost grows with the conversation. A duplex session has to process each 80 ms frame in bounded time, with state carried across frames for all six stages:
- log-mel
- the encoder's attention, convolution and subsampling
- the RNNT prediction net
- Nemotron-H
- EAR-TTS
- the codec's overlap

## 2. Change Summary

- **Streaming perception.**
  - `StreamingLogMel` emits hop-aligned frames once they trail the input edge by 1280 samples. It keeps only the lookbehind the centered STFT needs.
  - `ConformerStreamingState` keeps, per layer, the last 70 attention inputs and 8 conv inputs, plus a 16-frame mel cache for the causal subsampling stack.
  - `RnntStreamState` keeps the last token and the LSTM state, and emits transcript deltas.
  - New `stream` entry points on the attention, block and conv types leave the offline numerics unchanged.
- **Session** (`models::nemotron_voicechat::streaming`).
  - Creating a session seeds MLX's global RNG, warms EAR-TTS with the Aria prompt, and prefills the system prompt. The prefill advances the timeline but emits no events.
  - `push_audio` drains whole frames through a `FrameBuffer`.
  - `flush(pad_partial)` optionally runs the zero-padded partial frame, then closes with `Done`. `cancel()` closes with `Cancelled`.
  - `max_streaming_seconds` bounds a session.
  - Language and perception caches can be switched off for diagnostics.
- **Profiler.** It records per-stage wall-clock times after forcing evaluation, and summarizes mean/p50/p95/max, frames per second and the real-time factor.
- **CLI.** Adds `--stream`, `--max-streaming-seconds`, and `--profile` (summary JSON after 5 cold frames).

## 3. Technical Decisions

### The session owns its caches instead of a model-owned sequence slot

The issue planned to use a fresh `SequenceId` in Nemotron-H's `ModelOwnedSequenceState`. Instead, each session builds its own caches with `make_caches()` and holds them. This gives the same isolation between sessions created back to back. It also keeps the scheduler's sequence-state machinery out of a path that never uses the scheduler, and it makes release automatic when the session drops.

### Streamed encoder frames follow the reference, not the offline convolution

The reference's incremental subsampling runs over a 16-frame mel cache that starts one frame off the 8-frame subsampling grid. After the first few frames it therefore differs slightly from the offline full-utterance convolution: max abs difference 0.33, the same gap the Python reference shows. The port keeps the reference rule. Its streamed frames match the reference streaming path to 1e-6, and the downstream tokens and codes still equal the offline ones on the test input. The unit test checks the layer stack against offline and the leading frames end to end, the parts that are exactly equivalent.

### Frame buffering is a separate type

`FrameBuffer` is split out of the session so that chunk-boundary behavior can be unit tested without a checkpoint:
- 300, 1000, 1280 and 2000-sample pushes give 0, 1, 1 and 1 frames
- flush with padding gives one more frame
- flush without padding drops the partial frame

The real-checkpoint test adds chunk-size independence on top: 777-sample pushes give the same events as 1280-sample pushes.

## 4. Validation

All checks ran on `mlx-community/NemotronLabs-VoiceChat-11B-4bit`.

| Check | Result |
|---|---|
| First audio frame text id / function id / 31 codes vs offline | identical |
| All 61 streamed frames vs offline | text ids and codes identical |
| Streamed assistant text | "The capital of France is Paris." (equals offline) |
| Streamed encoder vs Python streaming path | max abs 1.1e-6; transcript deltas identical per frame |
| 777-sample vs 1280-sample chunks | identical events |
| `--stream` seeded runs | byte-identical WAVs, 61 x 1764 samples |
| `--max-streaming-seconds 1` | stops with the context-limit error |
| Profiler output | all fields finite |

The local gate: fmt, clippy on lib and tests, the unit tests of the touched modules, and `cli_help_consistency`, `dead_doc_pointers`, `llama_compat_manifest`. No latency or real-time-factor numbers were recorded, because the GPU was shared during this run.

## 5. Learning Points

- Streamed and offline paths can agree on discrete outputs even where their continuous intermediates drift, as they do here from the encoder's subsampling window onward. The parity test therefore asserts on the discrete outputs (ids and codes) and reports the continuous drift separately.
- In MLX, evaluation is thread-affine, so the session is driven on one thread. The realtime endpoint (#1376) gives the model a dedicated worker thread for this reason.

## 6. Known Limitations

- The real-time factor for the 4-bit and 8-bit checkpoints goes into the marked placeholder table in `docs/nemotron-voicechat.md`, measured by the orchestrator on an idle machine.
- Only one session runs at a time per model, because the sampling noise comes from MLX's global RNG.
