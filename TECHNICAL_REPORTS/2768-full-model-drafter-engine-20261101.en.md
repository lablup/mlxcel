# PR #2768: Run an ordinary full model as a drafter on the engine

**Date**: 2026-11-01 (generated before merge; the PR was open at the time)
**Status**: In review
**Languages**: Rust
**Risk Level**: Medium (opt-in through `--draft-model`; no default path changes)

## Executive Summary

Replaces the classic draft-model loop removed by #2756 with `FullModelDrafter`, a token-only drafter verified by `Engine::verify_round`, on `generate`, `run`, `serve` and `mlxcel-server`. The server had no such pairing before. Measured on an idle M1 Ultra it does not raise throughput, so this restores a capability and is not a speedup.

## 1. Problem Statement

The classic loop was the last decode path outside the batch-native engine and the only way to draft with a small ordinary checkpoint. Removing it without a replacement would have dropped that capability, so the two PRs ship together.

## 2. Technical Review

- **State.** Draft caches hold a prefix of the context stopping short of the current token. `DraftModelState::realign` keeps confirmed proposals and trims the rest, so early finishes and retracted asks take one path.
- **Refusals.** A target or drafter a trim cannot roll back (#2495), a larger draft vocabulary, an undeclared `vocab_size`, and a DFlash directory passed as a full model all fail at startup.
- **Failure scope.** A draft-side MLX throw or empty context ends drafting for that row only. `KVCache::try_eval_state` carries the throw (#822).
- **Repo ids.** `--draft-model` resolves `owner/name` like `-m`, without the target's `--revision`.

## 3. Technical Decisions

### 3.1 Per-row caches, no batched drafter

One loaded checkpoint serves every row and each row owns a `DraftModelState`, so each row rolls back alone. The cost is serial verification, which is why wider batches lose throughput.

### 3.2 Catch-up one chunk per tick

A tick feeds one chunk to one row, so catch-up cannot stall live rows for a whole prompt. A row catches up only if `max_tokens` left covers `ticks * (block + 2)`.

### 3.3 Keep `--prompt-lookup-max-batch` at 2

Raising it cost 35 to 68 percent at B = 4 and B = 8 across both pairs.

## 4. Change Summary

| Item | Value |
|---|---|
| Files changed | 55 before this report and the docs note |
| Lines | +3493 / -357 before this report and the docs note |
| Tests added | 34 |

Idle M1 Ultra, Metal, 256 tokens, temperature 0, aggregate tok/s:

| Requests | Qwen3-8B + 0.6B plain | gate 2 | Llama 8B bf16 + 1B plain | gate 2 |
|---|---|---|---|---|
| 1 | 93.5 | 56.1 | 40.3 | 40.3 |
| 2 | 108.3 | 56.2 | 25.8 | 34.1 |
| 4 | 112.9 | 112.8 | 49.8 | 49.7 |
| 8 | 138.2 | 138.2 | 93.2 | 93.1 |

Workspace gate on the stack before the final main merge: 14874 passed, 0 failed, 490 ignored.

## 5. Follow-up Actions

- Temperature 0 output is not byte-identical to plain decode. On Qwen3 at B = 2 one row stably differs; first differing position not recorded (#2778).
- Explain why plain Llama 3.1 8B bf16 is slower at B = 2 than at B = 1. Until then its B = 2 result is not a drafter gain.
- Not exercised: a real drafter download and client cancel on a live request.
