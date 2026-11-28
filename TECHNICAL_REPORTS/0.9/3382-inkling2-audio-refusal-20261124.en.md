# PR #3382: Refuse audio input for Inkling 2

**Date**: 2026-11-24
**Status**: Draft PR open (closes #3376). Narrow suites pass; checked on `models/inkling-2-mini` through the CLI and the server

## Executive Summary

Inkling 2 has no audio encoder, so every audio surface now refuses it with one message before any clip is fetched or any weight is loaded. Before the change the server admitted `input_audio` and failed late with a 500, a `/v1/messages` audio block was silently dropped, and the CLI surfaced a raw template exception.

## 1. Problem Statement

Mini's weight map holds only `model.llm.*` and `model.visual.*`, its `config.json` has no `audio_config`, and the official chat template raises `Inkling 2 does not accept audio input` for `audio`, `input_audio` and `audio_url` parts. Small's `config.json` has no audio key either; its weight index was not yet downloaded at the time of the PR.

| Surface | Before | After |
|---|---|---|
| CLI `generate`/`run --audio` | Template error after rendering | Shared message, exit 1, before weights load |
| `/v1/chat/completions` `input_audio` | 500 from the worker after the clip was decoded | 501 `not_supported_error` |
| `/v1/chat/completions` `audio_url` | 400 body deserialization error | 501 `not_supported_error` |
| `/v1/responses` `input_audio` | 500 from the worker | 501 |
| `/v1/messages` audio block | 200, audio silently dropped | 501 in the Anthropic envelope |
| `/v1/audio/transcriptions` | 500 from the worker | 501 |
| `/apply-template` | 200, prompt with an audio sentinel | 501 |

## 2. Design

`ModelMediaSupport` gains `audio_refusal: Option<&'static str>`. `detect_model_media_support` sets it to `INKLING2_AUDIO_REFUSAL` (with `audio: false`) for `Inkling2VLM`, so no route matches on a model id. `media_capability_rejection` uses the text for the audio kind; `--no-mmproj` keeps its own hint because `ModelMediaSupport::none()` carries no refusal. The transcription route reads the same field instead of the generic sentence.

Two parse-level changes were needed so the refusal can be reached. `ContentPart::AudioUrl` lets `audio_url` parts deserialize; on an audio-capable checkpoint they stay a request error (400) because no family reads them. `AnthropicContentBlock::Audio` (also `input_audio` and `audio_url` as aliases) is dropped by the translator exactly as `Unknown` was, and the route refuses it only when `audio_refusal` is set, so other families behave as before.

The CLI check, `refuse_audio_without_encoder` in `commands/generate.rs`, runs after `-m` is resolved and before any other loading. The worker keeps its generic refusal as defence in depth.

## 3. Validation

| Check | Result |
|---|---|
| `--lib server::media`, `server::startup`, `server::routes::prompt_inspection::inkling2`, anthropic request, translator and routes, `inkling`, `coverage_tests` | pass |
| `--bin mlxcel commands::generate` including a config-only Inkling 2 directory | pass; no weights needed, so none are loaded |
| `models/inkling-2-mini` CLI and server runs | messages and statuses as in the table; a text-only chat request still returns 200 |

## 4. Open Items

Small's `model.safetensors.index.json` still has to be checked for audio tensors once its download completes.
