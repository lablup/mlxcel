# PR #2756: Remove the classic draft-model path

**Date**: 2026-11-01 (generated before merge; the PR was open at the time)
**Status**: In review
**Languages**: Rust
**Risk Level**: Medium (breaking CLI and API removals announced for v0.8.0; must ship with #2768)

## Executive Summary

Removes `SpeculativeGenerator`, the last decode loop outside the batch-native engine, as promised for v0.8.0 (#2176). MTP, DFlash and prompt lookup keep working on the engine; ordinary-model drafting returns through #2768.

## 1. Problem Statement

The classic loop duplicated the engine's decode path and carried its own acceptance rule and tests. The v0.8.0-beta.1 changelog had deprecated it for removal.

## 2. Technical Review

- **Routing.** An explicit or auto-detected MTP drafter, or a DFlash directory on a target with an offline arm, routes as before. Anything else on `generate` fails before any drafter load and names the alternatives.
- **Breaking.** `--num-draft-tokens` on `generate`, `MLXCEL_SPECULATIVE_STOCHASTIC_ACCEPT` and `MLXCEL_SPECULATIVE_ACCEPT_DIAG` are gone. `ServerConfig::num_draft_tokens` stays for `--draft-max` and `/props`.
- **Server.** The `Classic` dispatch is gone, so a kind with no round loop fails at startup instead of decoding plainly.
- **Tests.** Coverage of the accept decision moved to the engine's token-only loop with a scripted drafter. Tests that pinned only the removed loop were deleted.

## 3. Technical Decisions

The PR removes rather than ports the modified rejection sampling in `speculative/stochastic_accept.rs`, because the loop was its only caller.

## 4. Change Summary

| Item | Value |
|---|---|
| Files changed | 44 |
| Lines | +1287 / -4800 |
| Tests added | 19 |

Workspace gate on the combined stack before the final main merge: 14874 passed, 0 failed, 490 ignored.

## 5. Follow-up Actions

- Merge #2768 right after this PR; alone this one drops ordinary-model drafting.
- Not validated: a positive offline DFlash run. The only offline arm is Laguna and no Laguna DFlash drafter exists locally, so that route is pinned by a fixture test.
