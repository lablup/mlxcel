# Bucketing the cuDNN SDPA plan-cache key, and why it should not ship

Issue #1820. Host GB10 (sm_121), MLX pin `81ba1c6a`, CUDA release build, toolchain 1.97.1, base tree `0ef0a1a4` (main, carrying PR #1817). Full measurement record: `docs/benchmark_results/sdpa-plan-cache-bucket-gb10-2026-09-17.md`.

## Background

MLX keys its cuDNN SDPA execution-plan cache on the exact shapes and strides of q, k, v and the mask. A speculative verify round appends keys every round, so its key is new every round and the cache structurally never hits. #1799 measured the consequence on the Laguna DFlash pairing: three shape classes each rebuilding a plan per round at about 22 ms, 67 to 76 ms of host time per round against 2.8 ms per classic decode token, and the LRU's lifetime-miss counter then aborting the process past about 170 rounds.

PR #1817 fixed that by routing these calls off cuDNN entirely, onto MLX's ops fallback, which has no per-shape build. It works, and it pays for it: the fallback materializes a `[B, heads, q_len, k_len]` score matrix, about 10 ms more GPU time per round at block 2. #1820 asked whether making the key reusable would recover that 10 ms while keeping cuDNN's flash kernel.

## What was built

MLX's own one-row decode canonicalization, extended to a small array-masked multi-row call. k and v are widened to a bucket, the additive mask is widened to the same width with the new columns set to `-inf`, and the true lengths reach cuDNN through `set_padding_mask` with `set_seq_len_q` / `set_seq_len_kv`, which is what keeps the widened region out of the result. k and v reach the bucket by unslicing when they are a leading slice of one cache buffer with room to spare, and by a zero-padded copy when they are not.

It works mechanically. The same generation builds 12 plans instead of 82, one per shape class rather than three per round, and greedy token ids are byte-identical to the exact-shape cuDNN path on both binaries.

## The recommendation: do not ship it

Against the standing bar that a default-on change must not lose on any measured workload, bucketing fails on the workload it was designed for.

| width | bucketed | #1817 fallback | ratio | ranges |
|---|---|---|---|---|
| 2 | 33.10 (32.43 to 33.77) | 35.12 (34.65 to 35.79) | **0.943x** | disjoint |
| 4 | 37.82 (37.40 to 38.29) | 36.64 (32.88 to 38.68) | 1.032x | overlapping |
| 6 | 35.76 (34.48 to 36.80) | 37.54 (36.70 to 38.21) | 0.953x | overlapping |
| 8 | 31.15 (30.64 to 31.51) | 32.68 (32.36 to 33.04) | **0.953x** | disjoint |
| 16 | 22.21 (21.89 to 22.43) | 24.26 (23.91 to 24.45) | **0.915x** | disjoint |

152-token prompt, n = 3, same binary on every arm, widths interleaved. Three of five widths are losses with non-overlapping ranges. At 2634 tokens the two are a wash (1.017x and 1.011x, ranges overlapping on both throughput and per-round verify time). There is no measured context length at which bucketing wins, so the 10 ms #1799 attributed to the fallback is not recovered anywhere it was measured.

A measured loss disqualifies it as a default regardless of what the unmeasured rungs would have shown. The recommendation does not rest on the rung that could not be run.

`MLXCEL_SDPA_FALLBACK_MAX_QUERIES` therefore stays as #1817 shipped it. The implementation is left on the branch behind `MLXCEL_SDPA_PLAN_BUCKET_MAX_QUERIES`, not removed, because the open question below is answerable and the implementation is the expensive part of answering it.

## What remains open

The crossover is **unmeasured on this host in its current state, not absent**, and the mechanism that motivates it is real. Only 10 of the pairing's 45 attention calls scale with prompt length: the target's full-attention layers, which take the free unslice arm. Its 30 sliding layers and all 5 drafter layers are capped at a 512 window. So bucketing's cost is flat in context while the fallback's score matrix grows linearly, and a crossover should exist somewhere above 2634 tokens.

Finding it would not produce an unconditional ship. It would produce a context-gated one, which then needs a short-context retry to confirm the loss is real before a gate could be designed, so the remaining question is larger than one rung.

## Why the deciding rung could not be run

A 16k-token run costs about 38 `NVRM: NV_ERR_NO_MEMORY` allocation failures on this host, against near zero at 2634. The 13-run rung projected to roughly 678 cumulative failures against a 400 budget, and was stopped after two runs with the count at 222 from a pre-session baseline of 2.

The runs complete successfully with valid throughput. That is not evidence the failures are benign: cumulative accumulation under spiky delivery, with individual runs looking fine, is the shape the 2026-07-06 hard freezes took on this host. A follow-up needs a fresh boot, whose clean driver count is the only condition under which the rung fits inside the budget.

## Defects found and fixed along the way

An audit of the overlay against its Rust mirror found a live regression this branch introduced. `cuda_sdpa_plan_bucket_eligible` took a single `masked` flag, but the C++ gate treats an array mask and `do_causal` differently: bucketing needs a mask to widen, so it rejects a causal block, while #1799's fallback fires on either. The maskless causal call site passes `true` for that flag, so the mirror claimed a causal block stayed on cuDNN while C++ actually routed it to the materializing fallback, silently dropping score-matrix query chunking from exactly those calls. Both predicates now take `arr_masked` and `do_causal` separately.

Four overlay fixes: `kv_cache_slice_extent` now requires `offset() == 0`, since its element-count identity proves the widened view is the same size as the allocation but not that it starts at it; the bucket `try`/`catch` no longer wraps `sdpa_cache().emplace`, whose LRU thrashing throw is the exact abort this work exists to prevent and would have been misread as a cuDNN refusal; a shape-eligible call whose layout declines bucketing now warns once, because such a call gets neither fix and rebuilds a plan every round; and the unslice arm takes `>=` rather than `>` so an exactly-full cache does not copy onto a buffer of its own size.

## Verification

Byte identity, compared as token ids: `7d044d8534201cb4` across base-binary cuDNN, branch-binary cuDNN and branch-binary bucketed, three repeats. `baa2c6b55d7f874d` on both binaries' ops-fallback arms, a dispatch difference #1799 already recorded, not a padding error. Non-speculative control `qwen3-1.7b-4bit`: 200 ids identical base against branch, and a trace showing zero bucketed calls in 1204 cuDNN SDPA calls.

Data layout, because two files there must not be read as results. `laguna_ladder_excluded.jsonl` holds the rows that were measured and then disqualified, and is cited for nothing: the halted 8k rung (seven rows, sparse at n = 1 to 2 and contaminated by a driver allocation storm, one row reporting a 24.8 ms per round drafter host build against about 12 ms in its siblings) and the two 16k rows from the rung that was stopped when each run proved to cost about 38 NVRM allocation failures. It is a separate file because no automatic filter can distinguish those rows: their foreign-process fields are clean, so every contamination check passes them. Each row carries an `excluded_reason`, and `summarize_ladder.py` drops such rows ahead of every other signal. The traces are frequency extracts rather than raw, one row per distinct line with its occurrence count, which preserves every count, distinct-value count and maximum the record cites while discarding per-call ordering that nothing cites.

Not verified: the `metal,accelerate` gate is not runnable on this Linux/CUDA host and was not run. Attention sinks are unexercised, since this checkpoint carries none. `peak_rss_kib` in the harness reports 1.9 GiB for a run whose weights alone are 20.97 GiB, because CUDA unified allocations on GB10 do not appear in the resident set; the field is recorded but is not a valid footprint proxy, and the harness's memory floor stays derived rather than measured.
