# Technical Report: PR #2224 - `mlxcel run` and chat as in-process server clients

**Date**: 2026-10-08

**Status**: Implemented and verified on GB10; pending merge.

**Languages**: Rust, Markdown (docs)

**Risk Level**: Medium. The CLI's interactive paths change their engine, their defaults (per the ADR 0007 table) and some display behaviour, and the production HTTP chat handler was refactored into shared functions. The HTTP validation order and messages are unchanged, `run -p` and the server give identical greedy output on two real checkpoints, and the REPL was driven end to end on a text model and a VLM.

## Executive Summary

`mlxcel run` and the chat REPL used `CxxGenerator` and kept their own copies of features the server already had: a reasoning splitter (`ReasoningFilter`), VLM embedding dispatch (`compute_vlm_embeddings`), and an offline DFlash driver that only served Laguna. Following llama.cpp, where `llama-cli` is a client of `llama-server`, this PR, Phase 5 of epic #2166 (#2173), makes both commands in-process clients of the server engine. They start the model worker at one slot and send every turn through the code the `/v1/chat/completions` handler uses, with no HTTP listener. The CLI keeps terminal I/O and REPL commands; templates, reasoning splitting, tool calls, stop strings, media preparation and speculative decoding come from the server.

## 1. Problem Statement

- `run -p` was promised to be byte-identical to `generate` because they shared code, and both differed from the server: different defaults (penalty windows, DRY breakers, loop detection, prefill chunk) and different implementations of reasoning splitting and media preparation.
- The CLI could not use the server's DFlash target for Qwen 3.5, or the MTP burst, because the offline drivers were separate.
- Each fix to the server's chat path needed a CLI twin.

## 2. Change Summary

- **`server::in_process::InProcessServer`**: builds what `start_server` builds minus the listener and side models (config, prompt-cache store, `ModelProvider` worker, tokenizer and template via the new shared `startup::load_chat_front`, `AppState`, the shared `run_startup_warmup`). The engine probe's `ServerEngine` now starts on it, so the stack is built once.
- **`routes::chat_generation`**: the chat handler's admission (`admit_chat_request`) and render/option build (`prepare_chat_generation`) moved out of `routes/chat.rs`. Both HTTP handlers and `InProcessServer::chat` call them. `chat` reads the worker stream through `StreamFilter`, parses tool calls, records the reasoning echo and submits the next-turn warm-up. `--no-chat-template` goes through the `/v1/completions` path.
- **CLI**: `cli/in_process_client.rs` maps flags to server options; `cli_turn.rs` handles terminal output and Ctrl-C (first press cancels the turn through the worker's cancel flag, a second press or a press outside a turn ends the process by SIGINT); `chat_transcript.rs` builds messages with images read once as `data:` URIs and refuses images past the server's per-request cap. The REPL keeps the prompt cache on with a per-conversation `prompt_cache_key`; `/clear` moves to a new key.
- **Retired**: `src/reasoning_stream.rs` (display now through `StreamFilter`) and `compute_vlm_embeddings` (`mlxcel generate` prepares media through `server::local_media` over `prepare_request_vlm_embeddings`).
- **New flags on `run`**: `--draft-model`, `--draft-kind`, `--draft-block-size`, the server's sampling flags (frequency/presence, XTC, mirostat, dynatemp, DRY breakers) and `--stop`.

## 3. Technical Decisions

- **In-process request channel, not loopback HTTP.** llama-cli posts to a llama-server thread over HTTP. In one process that adds a socket and SSE serialization and shares no more code than the request parse; mlxcel's `ModelProvider` channel is already what every HTTP route uses below the HTTP layer (ADR 0007).
- **Extract the handler's logic instead of copying it.** The CLI calls the same admission and render functions as the HTTP route, so a later fix lands in both. The review checked that the HTTP validation order (empty input, logprobs, XTC, top-n-sigma, typical-p, media capability before any read, video expansion, tool inputs, thinking budget, grammar) and its messages did not change.
- **Defaults from the ADR 0007 table, greedy kept.** Unset flags take the server's values (penalty and DRY windows 64, the b10621 DRY breakers, the server's loop-detection rule, prefill chunk 2048, per-sequence seeds). The table's one exception holds: when `generation_config.json` is silent the CLI stays greedy, so model tests and benchmarks stay reproducible.
- **Preserve `generate`'s Inkling layouts.** The server's ordered audio layout needs media markers that only the chat route renders, so routing `generate`'s Inkling media through it would have broken both templated and raw prompts. `server::local_media` keeps `generate`'s `Structured` and `Plain` layouts.
- **Classic `--draft-model` in the REPL is ignored with a notice.** On `generate` the flag means a classic draft model, which the server does not run; the REPL uses a drafter only with `--draft-kind dflash|mtp`.

## 4. Validation

- clippy `-D warnings` (lib, tests, bins, examples), fmt, license headers, `vlm_wrapper_capability_delegation`, `dead_doc_pointers`, `model_loader_stdout_guard`, `cli_help_consistency`, `llama_model_source_cli`.
- Module tests with `--test-threads=1`: `server::in_process`, `server::local_media`, `server::routes::`, `server::tool_calls`, `server::request_options`, `startup`, `cli_input`, `engine_probe`, `reasoning_display`, `server::chat_template`, model worker and provider, `server::batch::scheduler`, `multimodal::`, and the bin's `commands::`.
- Real checkpoints, release build:
  - `mlxcel-engine-parity` under `MLXCEL_SDPA_DETERMINISTIC=1`: `e:run` (the settings `run -p` builds) and `f:server` (the same chat request on `mlxcel-server` defaults, four slots, paged) are identical on Qwen3-1.7B 4-bit and Llama-3.2-1B 4-bit; the existing CLI, dense, paged and engine arms are unchanged.
  - `run -p` on Llama-3.2-1B; the REPL on Qwen3-1.7B (multi-turn recall, Ctrl-C mid-answer, `/clear` forgets the name) and on Qwen2.5-VL-3B (`/image` turn and follow-up).
  - `run -p --draft-model`: DFlash on Qwen3.5-4B (429 of 510 drafted tokens accepted) and MTP on Gemma-4-12B with its assistant drafter.
  - `generate` image (Qwen2.5-VL) and audio (Gemma-4-E2B): same prompt ids and greedy tokens as the pre-change binary.

## 5. Residual Risks

- **Output changes for CLI users**, listed in the PR: server default template for checkpoints without one (with a notice), no prompt echo on `run -p`, the server's sampling defaults where flags are unset, markup stripping in `generate`'s display.
- **The parity harness's CLI-style render and `mlxcel-bench-decode` still render chat prompts differently from the server** (they skip `enable_thinking=true`), so the harness's arms a to d and `e:run` compare different prompts on Qwen3. Documented; Phase 6 (#2176) moves every render onto one function.
- **Inkling paths are unit-tested only**; no checkpoint is local.
- **CLI decode throughput and startup time are not measured yet**; the epic's end-of-run measurement covers them.
- **`run -p` with audio, video, audio output, `--profile`, `--estimate-memory` and the diffusion and OCR families still goes through `generate`** until Phase 6.

## 6. Learning Points

- **A refactor that touches a production route needs a field-by-field check of the extracted function against both old copies.** The two HTTP handlers had drifted slightly; the extraction had to reproduce each one, not an idealized version.
- **"Same code" is weaker than "same request".** The old `run -p` promise rested on sharing `generate`'s code, which itself differed from the server. The new guarantee is a parity test on the request path.

## 7. Related

- Epic #2166, issue #2173, ADR 0007, ADR 0009.
- #2217 (engine), #2176 (Phase 6), ggml-org/llama.cpp#17824 and #24948 (llama-cli as a server client).
