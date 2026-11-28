# Technical Report: PR #3054 - fix(inference): bound the Qwen 3.5 long-prompt prefill peak

**Date**: 2026-11-15
**Reviewer**: -
**Status**: Completed
**Languages**: Rust, Bash, Markdown
**Risk Level**: Medium

---

## Executive Summary

A 28K-token prefill of the Qwen 3.5 family peaked at 44 GB (4B) and 78 GB (27B) on an M3 Ultra even after #3049 bounded the attention score buffer, and the server peaked at that single-pass level although it defaults to 2048-token chunks (#3045). Two independent causes were found and fixed: the Qwen 3.5 models projected every prompt position through the LM head (12.95 GiB of logits at 28K tokens), and the prefill plan ran each prompt-cache snapshot segment as one forward, which for a chat prompt is almost the whole prompt. The server peak fell to 9.1 GB and 25.9 GB, within 1.5% of the chunked CLI, TTFT dropped 16 to 27%, and greedy output is byte-identical to main.

---

## 1. Problem Statement

### 1.1 Background

`LanguageModel::forward_last_logits*` is the prefill entry point the engine calls for every prompt piece, CLI and server alike. Its default computes `forward` (all positions) and slices the last row. Families with a large vocabulary override it to project only the sampled row; Gemma 4, Llama 3, Qwen 3 and others already did (#672). The Qwen 3.5 family did not: `Qwen35Model`, `Qwen3NextModel`, and `Qwen35VLModel`, the wrapper that `Qwen3_5ForConditionalGeneration` checkpoints actually load as.

The prompt cache stores model-state snapshots at split points (the history boundary, optionally the system-prompt root and periodic checkpoints). `PrefillPlan` ended a piece at each split point, and kept the whole segment before it as one forward "whatever the chunk size".

### 1.2 Existing Issues

- Logits: 28000 x 248320 x 2 B = 12.95 GiB, the largest single prefill buffer, materialized only to read one row.
- Server partition: the history boundary is the chat rendering without its generation prompt, a few tokens before the end of the prompt. A single-turn 28K chat request therefore forwarded about 28700 tokens in one piece and 4 in the next, so the server reproduced the single-pass peak exactly (44.6 GB against the CLI's 44.4 GB).
- `MLXCEL_MEMORY_LIMIT` was documented as making MLX "raise an exception" past the limit. MLX never does that; a 16 GB limit let the prefill reach 45.8 GB.

---

## 2. Change Summary

| Area | Change |
|---|---|
| `qwen3_5.rs` | `forward_internal_at`, `forward_with_sequence_caches_at` and a `last_pos` argument on `forward_with_mrope_state`; the three `forward_last_logits*` overrides route through the same sequence and MRoPE state as `forward_with_sequence_id`. The hidden row is sliced before the final norm (per-row RMSNorm, so the row is unchanged). |
| `qwen3_next.rs` | Same overrides through `forward_for_sequence_at` / `forward_at`. |
| `qwen3_5_vl.rs` | Delegates the three entry points to the text model; without this the wrapper would fall back to the trait default. |
| `prefill_plan.rs` | `append_segment_pieces` / `segment_piece`: a segment is cut into unpadded pieces of at most `chunk` tokens ending on the boundary, in `with_prefix`, `resume_at` and `next_piece_at`. |
| Scheduler | In a chunked plan every piece, the one ending on a boundary included, counts toward `mlxcel_batch_prefill_chunks_total` and emits `prompt_progress`; the block reservation held between ticks adds the padding of the piece that runs right after a boundary piece in the same tick. |
| Docs | `MLXCEL_MEMORY_LIMIT` documented as a soft limit with the reason it stays one; CHANGELOG; benchmark doc with CSVs; rerunnable A/B harness `scripts/bench/prefill_peak/ab_prefill.sh`. |
| Tests | `qwen3_5_prefill_memory_tests.rs` (row equality, chunked continuation, a peak-memory guard, an ignored per-stage probe); plan, scheduler and MTP partition expectations updated. |

---

## 3. Technical Decisions

### 3.1 Chunk the segment instead of dropping the split

Only the boundary has to end a piece: the snapshot stored there must describe exactly the keyed tokens, and recurrent families cannot rewind padding, so the boundary piece stays unpadded. How the segment before it is partitioned does not affect either property, nor the pieces after the boundary, which are what a prompt-cache hit replays. Inner segment pieces are also unpadded, which keeps the "no padding before a snapshot" rule without per-family reasoning. The cost is that a cold prefill's KV inside the segment now comes from chunked forwards rather than one, which is the same kind of last-bit difference the plan already documents for `--prefill-chunk-size`; measured outputs were identical.

### 3.2 Keep `MLXCEL_MEMORY_LIMIT` soft

MLX's eval loop waits on in-flight command buffers while active memory exceeds the limit and its Metal allocator trims the buffer cache, but `malloc` never refuses. A hard cap would need mlxcel to reject allocations through a hook MLX does not expose, and it would fail the request mid-prefill without reducing what the prefill needs. The documented answer is to bound the peak with the prefill chunk, which is now honored on the server too.

### 3.3 Attribution method

The ignored probe evaluates after each stage and reads `peak_memory()` above the stage's starting active bytes. That isolates each stage's own transient, which a single lazy graph cannot do; the same probe then runs the served entry point lazily to show the end-to-end effect (25.80 to 20.27 GiB above the weights on the 4B).

---

## 4. Verification

| Path (28K tokens, M3 Ultra) | qwen3.5-4b-4bit | qwen3.8-27b-4bit |
|---|---|---|
| CLI single pass | 44.39 to 30.13 GB | 78.47 to 64.20 GB |
| CLI 2048 chunks | 11.00 to 8.97 GB | 27.83 to 25.80 GB |
| server `--parallel 1` | 44.57 to 9.10 GB | 78.64 to 25.93 GB |

Timing ran interleaved A/B then B/A inside one GPU lock hold; every faster arm is the new one and pairs agree within 0.5 s. Unit tests cover the override against the full forward, a continued piece, a peak guard at 1024 x 131072 (full logits 512 MiB, limit a quarter of that), chunked segments, in-segment continuations, and the scheduler's chunk counting.

---

## 5. Remaining Work

- A single-pass 28K prefill still peaks at 30 GB (4B): the 3.65 GiB causal array mask and one lazy graph's in-flight attention intermediates. Switching the full-attention layers to MLX's causal mode is #3051.
- A segment remainder shorter than a chunk (31 tokens before a 28703 boundary) is now its own piece; the tail rebalance in #3052 applies here as well.
- Not measured on M5; no M5-specific path is touched.
