# Technical Report: PR #2225 - `mlxcel generate` on the engine, CxxGenerator retired

**Date**: 2026-10-08

**Status**: Implemented and verified on GB10; pending merge.

**Languages**: Rust, Markdown (ADR, docs, CHANGELOG), shell

**Risk Level**: High. Every CLI decode (`generate`, `--profile`, `--prompt-lookup`, the inference session, the decode benchmarks) moves to a new client, and the 3,000-line `CxxGenerator` with its four decode loops is deleted. Greedy output is unchanged on the checkpoints checked, pipelined and synchronous runs agree on six families, and all parity arms agree, but throughput is measured only at the end of the epic.

## Executive Summary

After Phase 5, only `mlxcel generate` and the CLI speculative features still decoded through `CxxGenerator`, so every decode fix could still need a CLI-only twin and `bench_decode` timed a loop the server did not run. This PR, the last phase of epic #2166 (#2176), promotes the parity harness's `DirectEngine` to `mlxcel_core::engine::DirectEngine<M>`, the engine's one single-sequence raw-completion client (the llama-completion analog in ADR 0007), and puts `generate`, `MlxInferenceSession`, `bench_decode`, `bench_engine`'s CLI arm and `speculative_bench` on it. The client pipelines B=1 decode through the engine's split step, prompt lookup becomes a `Drafter`, the classic draft-model generator is deprecated, and `CxxGenerator` is removed.

## 1. Problem Statement

- `CxxGenerator` held four decode loops plus log-likelihood scoring, separate from the engine the server and `run` now use.
- `bench_decode` and the parity harness's CLI arm measured that separate loop.
- Prompt-lookup decoding was a CLI-only generator, so the server could not offer it.
- The parity harness and `bench_decode` rendered chat prompts with their own code, which differed from the server's (#2224 found the `enable_thinking` default missing there).

## 2. Change Summary

- **`DirectEngine<M>`** (`engine/direct.rs`): open, prefill `PrefillPlan` pieces, `complete_prefill`, `step` until the finish step, close. It carries the request's KV mode (Boundary-V table; model-owned modes injected before `open` so the first sequence's caches are built with them) and token bias. `impl LanguageModel for &M` lets it borrow a model the caller keeps; all 41 trait methods forward.
- **B=1 pipeline** (`engine/direct_decode.rs`, `engine/lookahead.rs`): after the first token, the forward for step n+1 is submitted before step n's token is read, through `submit`, `finish_rows` and `unwind_appends`. Finishing unwinds the one extra append (`Teardown::Unwind`); model-owned families that cannot rewind pipeline too and drop the extra step when the sequence closes (`Teardown::Discard`). Requests with history penalties, DRY, masks, overrides, logprobs or feedback samplers stay synchronous. `MLXCEL_FORCE_SYNC` forces the synchronous path. The eligibility helpers are shared with the scheduler.
- **Scoring**: `Engine::score` replaces `evaluate_loglikelihoods` (same context slice, fp32 log-softmax and gather).
- **Prompt lookup**: `PromptLookupDrafter` implements the server's `Drafter` trait; the loop is `DirectEngine::generate_with_drafter` over the new `Engine::verify` and `Engine::commit_appends`. Plain rounds pipeline as the old loop did. `PromptLookupGenerator` is removed.
- **One chat template front** (`server::chat_front`) for the server, `generate`, the parity harness and `bench_decode`.
- **Deprecation**: `SpeculativeGenerator` is `#[deprecated]`, prints a notice and is recorded in CHANGELOG Unreleased for removal in v0.8.0.
- **Benchmarks**: `bench_decode` keeps its CSV schema and adds a trailing `decode_path` column.
- **Removed**: `CxxGenerator` and its helpers, five diagnostic env vars that only the retired loops read, and the dead `plan_single_sequence_prefill`.

## 3. Technical Decisions

- **Promote the probe's client, do not write a fourth loop.** `DirectEngine` was already the parity harness's `d:engine` arm; making it the CLI client means the harness's CLI arm and the CLI are the same code.
- **Pipeline B=1 decode.** The first version of this PR decoded synchronously. The retired loop overlapped the next forward with the host read, and so does the scheduler, so a synchronous client would very likely miss ADR 0007's 1.0 percent threshold. The client reuses the engine's split step instead of a private state machine; the scheduler's machine stays separate because it also handles admission, preemption and prompt-cache donation.
- **Discard instead of refusing.** Security review found that requiring an exact rewind sent Gemma 4, Llama 4, Qwen 3.5, the SSM hybrids and most VLM wrappers to the synchronous path, where the retired loop pipelined them. Every `DirectEngine` caller closes the sequence right after decoding, so the one step written past the finish can be dropped at close. The server keeps the exact rule.
- **KV modes before `open`.** `Engine::open` calls `prepare_sequence_state`, which builds a model-owned family's caches from the modes it holds at that moment. Injecting the modes afterwards left the first sequence on a fresh model in FP16 under `--kv-cache-mode int8|turbo4`.
- **Prompt lookup as a `Drafter`.** It now shares the interface with the server's MTP and DFlash drafters. A server option needs its own burst arm and is a follow-up.
- **Deprecate, not remove, the classic draft-model path.** The server never ran it, and removing it outright would break users without notice.

## 4. Validation

- clippy `-D warnings` (root with examples, mlxcel-core with tests), fmt, license headers, `vlm_wrapper_capability_delegation`, `dead_doc_pointers`, `model_loader_stdout_guard`, `cli_help_consistency`, `llama_model_source_cli`.
- Module tests with `--test-threads=1`: core `engine` (including `direct_decode_tests`, which compare pipelined and synchronous runs at every finish, and `speculative_plain_tests`), `session`, `speculative::`, `drafter::`, `generate::`, sampling, prefill, decode_finish, `cache::attend`, `language_model_ref`; root `engine_probe`, `backend::`, `server::batch::scheduler`, `lookahead`, `server::in_process`, chat routes, `cli_input`, `lang_analyzer`, gemma3, gemma4, nanochat, `prefill_span_coverage`, `multimodal::`, and the bin's `commands::`.
- Real checkpoints, release build, `MLXCEL_SDPA_DETERMINISTIC=1`:
  - `generate --temp 0 -n 96` prints the same text pipelined and under `MLXCEL_FORCE_SYNC=1` on Qwen3-1.7B, Llama-3.2-1B, Gemma-4-E2B, Granite-4.0-H-350M, Qwen3.5-0.8B and Gemma-3-1B.
  - `--prompt-lookup` matches plain decode on Llama-3.2-1B.
  - `--profile` prints one `[TTFT]` line; `--no-chat-template`, a Qwen2.5-VL image prompt, Gemma 3 with int8 KV under `--profile`, and a `bench_decode` smoke run all complete.
  - Gemma 3 under `--profile` with int8 and turbo4 now prints the same text as the plain path for each mode (before the fix it printed the FP16 text).
  - `mlxcel-engine-parity`: the CLI, dense, paged, engine, `run -p` and server arms agree; only the documented prompt-cache rows diverge.
  - `single_row_batch_parity` and `scheduler_real_batch_parity` still pass.

## 5. Residual Risks

- **Throughput is unmeasured.** The epic's end-of-run measurement compares `generate` against a pre-epic `CxxGenerator` build at the 1.0 percent threshold.
- **Requests with penalties or DRY run synchronously.** The retired loop also overlapped these through its per-row sampler. A checkpoint whose `generation_config.json` sets a penalty loses that overlap until a per-row `submit` exists.
- **Seeded sampling depends on the path.** The pipelined fused draw consumes the RNG differently from the per-row chain, as the server's sync and lookahead ticks already do.
- **Not exercised on real checkpoints:** the classic draft-model deprecation notice (no local draft model), the Metal capture script, Llama 4 and Inkling.
- **A failed submit under `Teardown::Discard`** ends the run with an error rather than falling back; it is covered by review, not by a stub test.

## 6. Learning Points

- **A refactor that changes when state is created must check who reads the inputs at creation time.** The KV-mode bug came from moving cache construction into `open` while the mode injection stayed after it; a test that recorded the modes `prepare_sequence_state` saw caught it.
- **"Can be undone exactly" is a stronger requirement than a single-call client needs.** The scheduler must rewind because its sequences live on; a client that closes the sequence after decoding can discard the extra step and keep the pipeline.

## 7. Related

- Epic #2166, issue #2176, ADR 0007, ADR 0009.
- #2217 (engine step API), #2224 (in-process server client), #2182 (decode-undo log), ggml-org/llama.cpp#17824 (llama-completion after llama-cli moved onto the server).
