# Technical Report: PR #2950 - Opt-in system-prompt root snapshot

**Date**: 2026-11-11
**Status**: Completed (part of #2885 Part B; periodic checkpoints and gap-based eviction remain)
**Languages**: Rust, Python
**Risk Level**: Medium

## Executive Summary

PR #2950 adds a second prompt-cache snapshot family: the model state at the end of the system prompt and tool list (the "root"). Part A (PR #2911) only stored a snapshot at the history boundary, so a new chat with the same system prompt, or a turn that rewrote earlier history, prefilled the whole prompt again. With `MLXCEL_PROMPT_CACHE_SYSTEM_ROOT=1` the server splits the first prefill at the root as well and stores both snapshots. On a 16-request agent replay with Qwen3.8-Flash-Next (M3 Ultra) total TTFT fell from 51.16 s to 38.37 s, and the first turn of a new chat from about 5.6 s to about 1.5 s. All 16 completions matched cold runs in both arms. The switch is off by default. 38 files, +1923/-111.

## 1. Problem Statement

Agent clients send a large fixed prefix (system prompt plus tool schemas, often 4K to 20K tokens) on every request. Part A reuses work only up to the previous turn's history boundary. Two common cases missed it entirely:

- **New chat, same prefix.** The previous chat's history snapshot diverges right after the system prompt, and there was no snapshot at that point.
- **History edit or compaction.** When an agent rewrites a message a few turns back, the history snapshot no longer matches and the whole prompt is prefilled again.

Reusing an arbitrary prefix is not safe here, because a snapshot reproduces a cold run only when it was taken at a split point the cold run's own prefill plan also has (see `docs/CONTINUOUS_BATCHING.md`). The root therefore has to be a real split in the plan, not a prefix cut after the fact.

## 2. Change Summary

| Area | Change |
|---|---|
| `mlxcel-core/src/prefill_plan.rs` | `PrefillSplits` (up to 4 sorted split points) replaces the single `Option<usize>` boundary; `plan_geometry` filters splits to the unadopted range |
| `server/prompt_cache/policy.rs` | `MLXCEL_PROMPT_CACHE_SYSTEM_ROOT` parsing: off by default, `1`/`on` = minimum prompt of 2048 tokens, an integer sets the minimum |
| `server/chat_request.rs` | Root render: leading system/developer messages plus a sentinel user turn, cut at the sentinel; must be a strict prefix of the history render |
| `server/batch/scheduler*` | `system_root_len` computes the root split; `snapshot_splits` feeds both splits to the plan; `insert_boundary_snapshot` stores by origin |
| `prompt_cache/entry.rs`, `store.rs` | `SnapshotOrigin::SystemRoot`; LRU skips roots while they hold at most half of the snapshot budget |
| Gemma 4 MTP path | Uses `PrefillSplits` so the drafter splits where serving splits (#2160) |
| Docs, script | `CONTINUOUS_BATCHING.md`, `environment-variables.md`, benchmark result, `scripts/bench_prompt_cache_replay.py` |

## 3. Technical Decisions

### 3.1 Render with a sentinel user turn

Rendering only the system messages fails on templates that require a user turn (the Qwen3.8 template raises "No messages provided"). The server instead renders the system messages plus a user message holding a private sentinel and cuts at the sentinel. The result is accepted only when it is a strictly shorter prefix of the full render, so templates that reorder or rewrite the system block fall back to no root.

### 3.2 Drop the seam token

The root split is one token before the longest common prefix of the root and the prompt. The last token of the root render may merge with the following text in tokenization, so `system_root_len` steps back one token and requires the split to sit strictly before the history boundary.

### 3.3 Session-scoped, not shared

Roots stay in the request's cache session. Sharing across sessions would help multi-tenant servers most, but the docs promise that snapshots are not shared across sessions, and changing that is a separate decision.

### 3.4 Off by default

The extra split costs one more prefill piece on the first request (A1: 6.13 s against 6.06 s) and about 250 MB of snapshot memory per root for this model. Small Macs were not measured, so the switch stays opt-in.

## 4. Validation

- 98 targeted lib tests, including plan tests, the root render, `system_root_len` and two scheduler tests that check a new chat adopts the root and reproduces its cold run.
- Clippy with `-D warnings` and fmt clean; CI green.
- Replay (`docs/benchmark_results/qwen4-exp-system-root-replay-m3ultra-2026-11-11.md`): cached tokens 122,978 → 137,210 of 175,486; snapshot hits 12/16 → 15/16; 16/16 identical to cold in both arms.

Not run: CUDA/ROCm, a small-memory Mac (#2883 guardrails), the full workspace suite.

## 5. Follow-up Actions

- #2885 Part B PR2: periodic checkpoints every ~16K tokens at chunk boundaries.
- #2885 Part B PR3: eviction by smallest gap between snapshots.
- The default 2 GB snapshot budget holds about six snapshots for this model (15 LRU evictions per arm); the default may need raising.
- Possible cross-session root sharing, as a policy decision.
