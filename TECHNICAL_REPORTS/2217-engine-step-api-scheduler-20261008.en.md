# Technical Report: PR #2217 - Batch-native engine step API with BatchScheduler on top

**Date**: 2026-10-08

**Status**: Implemented and verified on GB10; pending merge.

**Languages**: Rust, Markdown (ADR, docs)

**Risk Level**: High. Every server forward, sampling draw and finish decision now runs through a new module, and the Gemma 3 and Llama 4 attention paths were rewritten. Real-checkpoint parity is unchanged on three families, and a real two-request batch matches each request alone, but the change touches the hot path of every served request.

## Executive Summary

`BatchScheduler` decided what runs and also ran it: it called model forwards and samplers directly at about a dozen sites. Epic #2166 needs the execution half reusable by the CLI (Phase 5) and by `mlxcel generate` (Phase 6). This PR, Phase 4b (#2172), adds `mlxcel_core::engine`: `Engine<M>` owns the model and the `CachePool` and exposes `open`, `prefill`, `step`, a split `submit`/`finish_rows` form for the lookahead pipeline, `unwind_appends` and `close`. The scheduler keeps admission, tick policy, preemption, prompt cache, handoff and the speculative burst generators, and drives the model only through the engine. A single sequence is a batch of one: `step` is the only decode entry. The PR also finishes the attention migration that #2210 started (Gemma 3, Llama 4, model-owned helpers) and removes `DecodeBatchContext`.

## 1. Problem Statement

- Model execution lived inside the scheduler, so the CLI could not reuse it without taking the scheduler too.
- B=1 on the server and a single sequence on the CLI were different code paths: the scheduler's `decode_single_step` and batched decode each called forwards and samplers, and `CxxGenerator` had its own loops.
- Three families (Gemma 3, Llama 4, the `model_owned` helpers) still chose dense or paged kernels from `DecodeBatchContext`, which #2210 had already removed from the others.

## 2. Change Summary

- **Engine (`src/lib/mlxcel-core/src/engine/`)**:
  - `open(SequenceSpec)`, `adopt`, `adopt_paged`, `close` (identical to the old `release_sequence_caches`).
  - `prefill(&PrefillStep)` runs one `PrefillPlan` piece (shared `piece_input` builds its input and mask), `prefill_cohort` the padded cohort, `complete_prefill` samples the first token, `trim_padding` undoes padding.
  - `step(&StepBatch, &mut [StepRow<H>]) -> StepOutput`: forward, then per row the mask, draw, `try_eval`, thinking override, logprobs, matcher and `finish_step`. A batch of more than one row whose rows all pass `shared_fused_params` takes one fused `[B, vocab]` draw, evaluated through `try_eval` before readback.
  - `submit` and `finish_rows`: the lookahead's split form. `submit` runs the forward inside `DecodeLookaheadAppendScope`; `unwind_appends` trims the pool or calls `rewind_decode_appends` for a model-owned family.
  - `StepRow` carries the row's `RowSampler`, config, history, budget and an `H: StepRowHooks` (logit mask, override, matcher, first-token stamp, `FinishHooks`). `RowOutcome` carries the token, an optional `FinishCause` and an optional `RowError::{Structured, Eval, BatchEval}`.
- **Scheduler**: `step_rows.rs` builds rows and applies outcomes; decode_tick, prefill, planned prefill, prompt cache and handoff call the engine. The acceptance grep for `.forward*(` and `sample_token_*(` in `src/server/batch/scheduler` finds no calls.
- **Attention**: `KvAttention` trait over `KVCache`, `RotatingKVCache` (writes still go through `update_and_fetch`, so the #2182 undo log keeps recording) and `ChunkedKVCache`; `attend_batched_rows` is the shared per-row loop. Gemma 3 and Llama 4 call it; `model_owned.rs` dispatch helpers are deleted; `gemma4_verify_rows` asks `is_dense_fp16()`. `rg 'is_paged_backed\(\)|is_paged_decode\(\)' src/models` finds nothing.
- **`DecodeBatchContext` removed**, with the core `DecodeStorageBackend`; the trait entry becomes `forward_batched_with_ids`, and the context-taking override in about 30 VLM wrappers, qwen3_5, qwen3_moe and muse_glimmer folds into `forward_batched`.
- **Tooling**: a `d:engine` arm in `mlxcel-engine-parity`; deleted the example and script that compared the removed compat kernels through the context.
- **ADR 0009** records the API and decisions; ADR 0007 and 0008 carry amended-by notes.

## 3. Technical Decisions

- **Single row at the engine, not in the trait.** The issue asked that `LanguageModel::forward` become a provided method delegating to the batched entry. With more than 200 implementations and about 40 batched overrides that is a mechanical change across the whole model tree, and only the engine chooses between the two forwards. The engine is the single decode entry and runs the single-row forward at B=1, the same call the batched default makes at `b == 1`. A real-checkpoint test pins one-row batched logits bit for bit against the single-row forward on Qwen3 and Llama 3, with and without sequence ids.
- **A batch of one always takes the per-row chain.** The first version let a one-row batch take the fused draw, whose readback could not report an MLX error, so a throw at B=1 would have aborted the process instead of failing the request (#822). Review caught it; the fused draw now runs only at B>1, and it too goes through `try_eval`.
- **One fused-eligibility rule.** The lookahead gate and the synchronous step call the same `shared_fused_params`, because the pipeline's output matches the synchronous path only while the two agree.
- **A batch-wide failure counts once.** A single throw in the fused draw fails every row with `RowError::BatchEval` and increments the health counter once. Counting it per row would let one transient throw on a batch of eight trip `MAX_CONSECUTIVE_EVAL_FAILURES` and shut the scheduler down.
- **Model-owned families attend through their caches.** Their per-sequence state is never pool-backed, so ADR 0008's rule reduces to its dense rows, and the removed compat kernels were per-row C++ loops of the same shape #2210 dropped. The kernels stay in core as the `ffi_tests` reference.
- **The scheduler still reaches the model for bursts.** The MTP and DFlash burst adapters run target forwards outside the engine, because the issue keeps the burst generators in the scheduler. `model()` and `parts_mut()` exist for those adapters and the handoff restore, so the guarantee is enforced by the acceptance grep and the span guard rather than by the type system.

## 4. Validation

- clippy (`-D warnings`, root with examples and mlxcel-core tests), fmt, license headers, `vlm_wrapper_capability_delegation`, `dead_doc_pointers`, `model_loader_stdout_guard`, `prefill_span_coverage`.
- Unit and module tests, each run alone with `--test-threads=1`: core `engine`, `cache::attend`, `mla::`, decode-undo, paged decode and detach, sampling, generate, prefill, decode_finish; root model families (qwen3, llama3, helium, deepseek_v2, gemma3, llama4, gemma4, muse_glimmer, afmoe), `multimodal::`, the whole `server::batch::scheduler` module, active and sequence tables, finish step, speculative burst, disaggregated, pipeline, CLI input, engine probe, memory estimate.
- Real checkpoints under `MLXCEL_SDPA_DETERMINISTIC=1`, after merging `main`:
  - `mlxcel-engine-parity` on Qwen3-1.7B 4-bit, Llama-3.2-1B 4-bit and Gemma-3-1B 4-bit: CLI, server dense, server paged and the direct engine arm identical for greedy and seeded-penalties-DRY; the only divergences are the documented prompt-cache rows, unchanged from the Phase 0 baseline.
  - `single_row_batch_parity`: Qwen3 and Llama 3 one-row batched logits bitwise equal to the single-row forward.
  - `scheduler_real_batch_parity`: two concurrent greedy Gemma 3 requests (47 of 47 decode ticks at B=2) give the same 48-token streams as each request alone, on the paged and the dense backend.

## 5. Residual Risks

- **Throughput is not measured yet.** Two small Vec allocations per decode tick remain (the row views and the outcomes). The epic's end-of-run measurement checks the 1.0 percent threshold.
- **Llama 4 has no local checkpoint.** Its batched route, which now batches the RoPE-layer block on the dense backend too, is covered by synthetic tests only.
- **Op-order changes, not bit-identical:** Gemma 3 and Llama 4 on the paged backend run per-row `attend` instead of the compat kernels; Gemma 3 global layers in a Turbo mode with a dequant-first or compressed variant take that variant; Llama 4 RoPE layers run batched on the dense backend.
- **Gemma 3 and Llama 4 still report `supports_paged_decode_backend() == true`.** `auto` still picks paged for them and they pay pool allocation and per-step sync while no attention path reads the pool. Left as a measured follow-up.
- **Remaining infallible readbacks** on the pipelined path and logprobs are unchanged from main.

## 6. Learning Points

- **Docs describe intent; tests pin behavior.** The code comments, commit body and ADR all said a batch of one runs the per-row chain, and the code did not. A test that checks which path B=1 took would have caught it at once.
- **Count failures at the granularity of the operation that failed.** One eval covering eight rows is one failure, not eight; the health counter's limit was sized for the former.

## 7. Related

- Epic #2166, issue #2172, ADR 0007, 0008, 0009.
- #2168 (finish step), #2169 (row sampler), #2170 (prefill plan), #2171 / #2210 (cache-owned attention), #822 (row-only errors), #2182 (decode-undo log), #2173 and #2176 (next phases).
