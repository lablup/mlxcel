# Technical Report: PR #2256 - Pipeline penalty and DRY requests in the engine's B=1 client

**Date**: 2026-10-10

**Status**: Implemented and verified on GB10; pending merge.

**Languages**: Rust, Markdown (ADR, CHANGELOG)

**Risk Level**: Medium. Every `mlxcel generate` request that sets a history penalty, DRY, mirostat, adaptive-p or the extended chain moves from the synchronous loop to a new pipelined loop. Output equals the synchronous loop token for token in unit tests and on real checkpoints across six families, and `MLXCEL_FORCE_SYNC=1` still selects the old loop. Throughput is measured at the end of the run, together with #2255.

## Executive Summary

After #2176, `DirectEngine` (the client behind `mlxcel generate`, `MlxInferenceSession` and the decode benchmarks) pipelined decode only for requests whose sampler fits the fused device draw. Any request with a repetition, frequency or presence penalty, DRY, mirostat, adaptive-p or the extended chain fell back to a synchronous loop that waits for each token before encoding the next forward, which the retired CLI loop measured at about 12 percent of decode time. This PR adds a per-row lookahead pipeline for those requests: the next forward is submitted on the still-unread device token, the token is read and committed while that forward runs, and only then is the next draw built from the committed history. Part 2 of the original issue (prompt lookup on the server) moved to #2255.

## 1. Problem Statement

- `DirectEngine::decode` chose between the fused pipeline (`decode_pipelined`) and `decode_sync`. Eligibility for the fused path is `client_fused_params`, which rejects any sampler that needs the previous token on the host (history penalties, DRY) or feeds back into its own state (mirostat, adaptive-p).
- The fused path cannot simply be extended: a penalty draw for token t+1 must see token t in the history, so it cannot ride the forward on the device the way the fused draw does.
- The forward does not depend on the draw's host value, only on the device token. The retired `CxxGenerator::sample_next_step` used this to keep the forward ahead even for penalty requests; #2176 dropped that overlap when it moved `generate` to the engine.
- The issue assumed `mlxcel generate` inherits a penalty from a checkpoint's `generation_config.json`. It does not: `GenerationConfigDefaults` (`src/loading/mod.rs`) carries only EOS ids, temperature, top_p and top_k. Requests reach the new path only through explicit flags. Inheritance was left out because it would change output.

## 2. Change Summary

| Area | Change |
|---|---|
| `engine/mod.rs` | `Engine::submit_forward` (the speculative forward of `submit` without the draw, scheduled with `async_eval`, returns lazy logits) and `Engine::draw_row` (one row's per-row chain draw from those logits, token bias in its normal position, scheduled lazily). Both submits share `speculative_forward` and `schedule`, so `submit`'s behaviour is unchanged. `draw_row` rejects rows with a logit mask, token override or logprobs payload (`EngineError::Batch`), because those need the host token before the draw is confirmed. |
| `engine/direct_decode_rows.rs` (new) | `DirectEngine::decode_pipelined_rows`: submit forward n+1 fed by the pending draw, `try_eval` then read the pending token, commit it with `finish_rows` (which pushes it into the penalty history), then `draw_row` for n+1. At most two speculative appends are held. |
| `engine/direct_decode.rs` | `lookahead_teardown` factors the unwind-or-discard rule out of `decode_lookahead`. `decode` now tries the fused pipeline, then the per-row pipeline (`decode_row_lookahead`: not fused-eligible, `lang_bias_counters` off), then `decode_sync`. Both pipelines hold `DecodeCommandBufferBudget`. |
| `src/bin/bench_engine.rs` | `--repetition-penalty` flag, wired through a new `bench_sampling_config` used by both `main` and its test. |
| Docs | ADR 0009 (signature table, raw-completion paragraph), module docs, CHANGELOG Unreleased. |
| Tests | `direct_decode_rows_tests.rs` (376 lines): for repetition, frequency+presence, DRY, DRY plus a windowed penalty, seeded mirostat and seeded adaptive-p, the stream and final state equal `decode_sync` at EOS, at `max_tokens` with no extra forward, and on a callback stop; a `NoisyModel` stub whose logits the penalties actually change; a path test by forward count (5 vs the synchronous loop's 4) that cannot pass on main; routing and model-owned unwind/discard tests; `draw_row` rejection tests. |

11 files, +755 / -67.

## 3. Technical Decisions

**Forward ahead, draw behind.** The pipeline overlaps the forward with the host read and the finish step, not the draw. A draw for t+1 is built from the same logits, history and sampler state as the synchronous step's, and forwards consume no randomness, so greedy and seeded streams are identical to `decode_sync`. The alternative, a device-side penalty kernel that updates history on the device, would let the draw ride the forward too, but it duplicates every sampler on the device and was not needed to recover the overlap.

**Overlap depends on the sampler.** Penalties, DRY and the extended chain without adaptive-p keep the draw lazy, so both draw and finish overlap the next forward. Mirostat and adaptive-p read their uniform draw on the host inside the sampler (`sampling.rs`), so only the finish step overlaps. The docs say this; the planned measurement covers `--repetition-penalty` only.

**Teardown follows the fused pipeline.** One uncommitted append past a finishing token or a callback stop, none after the token that spends `max_tokens`, and a synchronous fallback from the last committed token after a failed submit or draw under `Teardown::Unwind`. Under `Teardown::Discard` (a model-owned family that cannot rewind) a failed draw ends the run with the error, because the extra append cannot be undone.

**Host read behind `try_eval` (review fix ea1a45e4).** The first version read the pending token with `ffi::item_i32`, which cannot report a failure, so an asynchronous backend fault surfacing at that wait would abort the process instead of failing the run (#822). `try_eval(&pending)` now runs first; it waits on that one array, so the overlap with the next forward is kept.

**`lang_bias_counters` keeps a request synchronous.** The B9 pre-bias counters read the pre-bias argmax on the host at every draw, which would serialize the pipeline anyway.

## 4. Validation

- Local gate on the final branch (GB10, `--profile test-fast --features cuda`, under `gpu-lock`, `--test-threads=1`): fmt, clippy `-D warnings` (root lib/tests/bins/examples and mlxcel-core), mlxcel-core `engine::` (46), `session`, `sampling`, `generate::`, `drafter::`, `speculative::`, `decode_finish`; root `engine_probe`, `backend::`, `server::batch::scheduler`, `server::in_process`, `cli_input`, `commands::`; `mlxcel-bench-engine` unit tests; `dead_doc_pointers`, `model_loader_stdout_guard`, `cli_help_consistency`; Apache header check. All passed (mlxcel-core `session` 12, `sampling` 202, `generate::` 38, `drafter::` 223, `speculative::` 161; root `server::batch::scheduler` 187, `commands::` 202, `mlxcel-bench-engine` 12).
- Real checkpoints, release build, `MLXCEL_SDPA_DETERMINISTIC=1`: `mlxcel generate` output with the per-row pipeline equals `MLXCEL_FORCE_SYNC=1` output in all 12 cases: Qwen3-1.7B (repetition penalty, DRY, penalty plus DRY with `--repeat-last-n 32`, greedy, all with `--show-reasoning`), Llama-3.2-1B (penalty, DRY, seeded penalty at temperature 0.8), Gemma-3-1B, Gemma-4-E2B (model-owned state), Granite-4.0-H-350M and Qwen3.5-0.8B (hybrid state that cannot rewind, the `Teardown::Discard` path), Hunyuan-1.8B (seeded penalty), 160 tokens each.
- Parity harness (`mlxcel-engine-parity -n 400`) on Qwen3-1.7B, Llama-3.2-1B and Gemma-3-1B 4-bit: the CLI arm (`a:cli`, now this pipeline for the seeded-penalties-DRY row) equals server dense, server paged and the direct engine in every row. The only divergent pairs are the server prompt-cache rows, which this PR does not touch: Qwen3 and Gemma 3 match the #2225 run exactly; on Llama the prompt-cache hit row moved from the seeded row (token 25) to the greedy row (token 213).
- Review: one MEDIUM (the infallible host read, fixed in ea1a45e4) and five LOWs; security review found nothing new. Two LOWs fixed by the finalizer (docs overlap wording, bench flag test through the real wiring).

## 5. Residual Risks

- A backend error after a draw is built but before it is committed can advance mirostat, adaptive-p or seeded RNG state twice, because `decode_sync` redraws that token. The run continues but no longer matches the synchronous stream. Only reachable after a backend fault; tracked in #2258.
- Pre-existing infallible host reads remain in the fused B=1 pipeline, the server decode tick, the prompt-lookup plain round and the mirostat/adaptive-p sampler reads (#2258). Failure paths have no tests yet (#2258).
- DRY over the full history rebuilds a position map each step with a quadratic scan on repeating output, and DRY and frequency/presence allocate a vocab-sized buffer per step (#2259). The pipeline now hides these costs behind the next forward rather than adding to them.
- Throughput is not yet measured. The end-of-run measurement compares this branch against pre-epic `4c44e317` with the flag patched in, `--path cli --repetition-penalty 1.1`, Qwen3-1.7B and Llama-3.2-1B, 256 and 8192 tokens.

## 6. Learning Points

- A pipelined decode loop needs the next forward's input on the device, not the next draw's output on the host. Separating "submit forward" from "draw" lets samplers with host-side state keep most of the overlap.
- An infallible readback (`item_*`, `tokens_to_host`) is where an asynchronous backend fault surfaces. Any wait on a fresh array should go through the fallible `try_eval` first.
- A real-checkpoint diff that compares hidden-reasoning output can pass trivially: on Qwen3 without `--show-reasoning` all tokens go to the hidden channel and both outputs are empty. The verification script fails a case with fewer than three output lines for this reason.

## 7. Related

- Issue #2229 (Part 1), epic #2166, split-out #2255 (server prompt lookup), follow-ups #2258 and #2259.
- PR #2225 (`DirectEngine`, fused B=1 pipeline), PR #2217 (engine step API), ADR 0009.
