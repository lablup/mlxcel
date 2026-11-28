# Technical Report: PR #2205 - One prefill plan and one trim hook for the CLI and the server

**Date**: 2026-10-07

**Status**: Implemented and verified on GB10; pending merge.

**Languages**: Rust, Markdown

**Risk Level**: Medium.
- Every CLI and server prefill now executes one plan, and the server's default prefill chunk changes from 512 to 2048.
- Greedy output on three real checkpoints is unchanged before and after.
- The new default's effect on concurrent decode latency is not measured yet; the epic's end-of-run benchmark covers it.

## Executive Summary

Phase 3 of epic #2166 (#2170).
- **One plan.** `mlxcel_core::prefill_plan::PrefillPlan` now decides, in one place, how a prompt is split into forwards: the adopted prompt-cache prefix, the history-boundary split, chunking, tile padding and the pad trim. Every CLI and server prefill site only executes its pieces.
- **One trim hook.** `LanguageModel::trim_state(Option<SequenceId>, excess)` replaces the two separate hooks.
- **One chunk size.** The chunk is a single policy, 2048 per ADR 0007's measurement.
- **Prompt-cache divergence explained.** The hit vs miss divergence found in Phase 0 is root-caused (the prefill partition) and bounded by an explicit invariant.

## 1. Problem Statement

- **Prefill was decided twice.** The CLI chunked at 2048, the server at 512, and padding and trimming were coded separately on each side.
- **Two trim hooks.** The model API had `trim_internal_caches` for the CLI and `trim_sequence_state` for the server, so a family had to implement both. #2140 was that failure.
- **Prompt-cache divergence.** The Phase 0 parity baseline showed a server prompt-cache hit diverging from the uncached run of the same prompt; Qwen3-1.7B greedy diverged at token 0.

## 2. Change Summary

- **`PrefillPlan`:**
  - `new`/`with_prefix(prompt_len, adopted, boundary, chunk, PrefillCaps)` produce ordered `PrefillPiece { range, padded_len }`.
  - The boundary segment is one unpadded piece, and chunks restart at the boundary.
  - The final piece is tile-padded where the model and input allow it.
  - `reproduces` and `split_points` state when a cache hit can reproduce a miss.
- **CLI:** `prefill_prompt_last_logits`, the `generate_with_stats` copy and both embedding variants run plan pieces through one executor.
- **Server:**
  - `planned_prefill.rs` runs pieces for full, chunked, batched and handoff prefill. The plan is rebuilt from the sequence each tick, with `prefill_offset` as the cursor.
  - The history-boundary snapshot is inserted after the boundary piece, with no extra forward.
  - Gemma 4's `mtp_prefill_ranges` and the paged block reservation use the plan.
- **`trim_state`:** seven model-owned families implement it, the VLM wrappers forward it, and `LoadedModel` delegates to it. `rewind_decode_appends` stays a separate exact-rewind contract.
- **Chunk policy:** `prefill_chunk_len()` reads `MLXCEL_PREFILL_CHUNK`, else 2048. It is now also the server's `--prefill-chunk-size` default, on both server binaries.
  - The memory estimate's activation term follows it (`activation_prefill_tokens()`).
  - The batched-prefill token budget caps each row's share at 512, so the cohort budget stays 4096.

## 3. Technical Decisions

**Root cause of the cache divergence: the partition, not adoption.** Under `MLXCEL_SDPA_DETERMINISTIC=1`, chunking the miss at the cached length made miss and hit identical:
- Qwen3-1.7B: chunk at 45.
- Llama-3.2-1B: chunk at 71.

The mechanism: the dense-KV miss is one 52- or 75-row forward (tiled qmm). The hit forwards only a 7- or 4-row suffix, which takes the per-row qmv kernel (`M*B < 8`) and a different attention tiling. The two reduce in a different order and flip a near-tied token.

**Documented bound instead of forcing the split.** A hit reproduces the miss exactly when its adopted prefix ends on a split point of the miss's plan, and the adopted rows were written by those same pieces. Splitting every family at the history boundary would make the harness rows identical, but it was rejected for three reasons:
- It adds a forward launch to every cold chat prefill.
- It would make the prompt cache change single-turn output.
- It does not fix whole-prompt replay anyway.

**One chunk size: 2048.** Phase 0 measured 8 to 29 percent lower TTFT at 8192 tokens with 2048. Review found three places that silently assumed 512, and they were fixed:
- `mlx_server`'s flag default.
- A `clamp(1, 0)` panic in the memory estimate.
- The batched-prefill budget.

## 4. Validation

- **Unit tests:**
  - `prefill_plan` 11, mlxcel-core `prefill` 76, `generate::` 46, `decode_finish` 9.
  - Root modules run one at a time: `scheduler_prompt_cache_plan_tests` 3 (including the chunk-counter and progress-frame cases), `server::batch::scheduler::tests` 94, `scheduler_prompt_cache_tests` 31, `prefill_span_coverage_tests` 3, `finish_step_tests`, `speculative_burst_tests`, `prefill_cohort`, `cli_input`, `memory_estimate`, `gemma4_mtp_target`.
  - `vlm_wrapper_capability_delegation`, including the tile-padding guard audit.
- **Real checkpoints:** `mlxcel-engine-parity` with `MLXCEL_SDPA_DETERMINISTIC=1` on qwen3-1.7b-4bit, llama-3.2-1b-instruct-4bit and lfm2-350m-8bit.
  - Every pair result was unchanged before, after and after the merge with `main`.
  - CLI and server token streams were identical to the base commit.
- **Known flakiness:** the full `server::batch::scheduler` filter aborts intermittently in CUDA graph capture. Every module run alone is clean.

## 5. Residual Risks

- **Concurrent decode latency is unmeasured.** Inter-token latency of concurrent decode streams under the 2048 default is left to the end-of-epic benchmark. A longer prefill piece delays decode grants by up to four times.
- **History segments are not chunked.** A snapshot family's history segment is one unchunked forward regardless of chunk size. This is pre-existing (#1143).
- **One environment variable now drives both paths.** `MLXCEL_PREFILL_CHUNK` now also sets the server default, and the memory estimate does not see an explicit `--prefill-chunk-size`.
- **No numeric bound test.** Nothing asserts a numeric bound on hit-vs-miss divergence. The bound is the documented invariant plus the measured table.

## 6. Learning Points

- **Treat a partition as part of the numerics.** Two correct prefills of the same tokens differ in their low bits when they are split differently, because the kernel dispatch depends on row count.
- **Changing a default needs a sweep for its implicit dependents.** Three of the four review fixes were places that assumed the old 512 without naming it.

## 7. Related

- Epic #2166, issue #2170.
- ADR 0007, ADR 0005.
- #2201 (finish step, merged into this branch).
- #2140, #1143, #2185.
