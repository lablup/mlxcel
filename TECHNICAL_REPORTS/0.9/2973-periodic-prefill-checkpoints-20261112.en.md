# Technical Report: PR #2973 - Opt-in periodic prefill checkpoints

**Date**: 2026-11-12
**Status**: Completed (part of #2885 Part B; gap-based eviction remains)
**Languages**: Rust, Python
**Risk Level**: Medium

## Executive Summary

PR #2973 adds a third prompt-cache snapshot family for snapshot-only models: checkpoints at every multiple of a fixed interval (16,384 tokens by default) before the history boundary. An agent that rewrites its history deep into a long context now resumes from the last checkpoint before the edit instead of from the system-prompt root. On a Qwen3.8-Flash-Next replay where turn 9 edits history about 20K tokens into a 28K prompt, that turn reused 16,384 tokens instead of 4,744, and its TTFT fell from 22.16 s to 11.25 s. The switch, `MLXCEL_PROMPT_CACHE_CHECKPOINT_INTERVAL`, is off by default.

## 1. Problem Statement

A snapshot-only family (recurrent or model-owned state) can reuse a stored state only when a new prompt starts with exactly its tokens. After #2950 the store held two kinds of prefill snapshot: the history boundary (the next turn of the same chat) and the system-prompt root (a new chat with the same system block). An edit or compaction at token 20,000 of a 28,000-token chat invalidates the history snapshot, and the root at 4,744 tokens leaves about 23K tokens to prefill again.

The issue text proposed aligning checkpoints to prefill chunk boundaries so no chunk is split. That does not apply here: the history segment of a snapshot family is one unchunked forward (since #1143), so it has no chunk boundaries to align with. A checkpoint therefore has to be a new split point in the plan.

## 2. Change Summary

| Area | Change |
|---|---|
| `server/prompt_cache/policy.rs` | `checkpoint_interval_tokens()`: off by default, `1`/`on` = 16,384, any integer of at least 256 sets the interval; ignored when the boundary snapshot is disabled |
| `server/batch/scheduler/mod.rs` | `checkpoint_points(interval, start, upper, min_prefix, max)`: absolute multiples strictly inside the unadopted span, deepest `max` kept |
| `server/batch/scheduler/prompt_cache.rs` | `snapshot_splits` reserves room for root and history boundary, then adds checkpoints; `insert_boundary_snapshot` dispatches root, history, or checkpoint |
| `mlxcel-core/src/prefill_plan.rs` | `MAX_SPLITS` 4 → 16, now public |
| `prompt_cache/entry.rs`, `store.rs` | `SnapshotOrigin::Checkpoint`, exempt from the session-chain supersede rule |
| Tests | policy parser, `checkpoint_points`, store supersede exemption, scheduler run where an edited history adopts a checkpoint and reproduces its cold run |
| Docs, script | `CONTINUOUS_BATCHING.md`, `environment-variables.md`, benchmark result, `--edit-turn`/`--edit-target` in the replay script |

## 3. Technical Decisions

### 3.1 Absolute multiples, not offsets from the adopted prefix

A hit reproduces a cold run only if the adopted prefix ends on a split point of the cold plan. Multiples of the interval counted from token 0 depend only on the prompt prefix, so a later prompt that shares the first `k * interval` tokens splits at the same point when it runs cold. Counting from the adopted offset would make the points depend on cache state and break that property.

### 3.2 Exempt from supersede

The store collapses a conversation's same-origin snapshots to the longest one, which keeps one boundary snapshot per chat. Applied to checkpoints, the 32K checkpoint would delete the 16K one, which is exactly the one an edit between them needs. Checkpoints are bounded by the byte budget's LRU instead.

### 3.3 Reserve room for root and history

The old 4-point cap kept the shallowest points and so would have dropped the history boundary on a 64K prompt. The cap is now 16. The scheduler reserves slots for the root and the boundary and keeps the deepest checkpoints, so the multi-turn hit is never lost. The trade-off is that a prompt beyond 14 intervals (229K tokens at the default) drops its shallowest checkpoints.

## 4. Validation

- Unit tests: positions, parser, supersede exemption, and `edited_history_adopts_the_last_checkpoint_and_reproduces_its_cold_run` (chunked and unchunked) on the tiny Gemma 3 fixture.
- Clippy `-D warnings` and fmt clean; CI green.
- Replay (`docs/benchmark_results/qwen4-exp-prefill-checkpoints-replay-m3ultra-2026-11-12.md`): total TTFT 70.18 s → 58.66 s, cached tokens 197,007 → 208,647 of 268,730.

The replay also showed that ordinary chained turns (A5 to A8) can differ from a cold run on `qwen4_exp` by about 0.4 nats at the first token, with or without checkpoints. That is pre-existing and tracked separately.

Not run: CUDA/ROCm, a small-memory Mac (#2883 guardrails), the full workspace suite.

## 5. Follow-up Actions

- #2885 Part B PR3: eviction by smallest gap between snapshots.
- The 2 GB default snapshot budget holds about two checkpoints of this model at 16K to 32K tokens (0.5 to 1.2 GB each); a model-aware default may be needed before the switch is useful at defaults.
- Chained-hit divergence on `qwen4_exp`, filed as a separate issue.
