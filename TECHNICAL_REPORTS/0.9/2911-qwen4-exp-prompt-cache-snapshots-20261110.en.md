# Technical Report: PR #2911 - prompt-cache snapshots for qwen4_exp

**Date**: 2026-11-10
**Reviewer**: -
**Status**: Needs Follow-up
**Languages**: Rust
**Risk Level**: Medium

---

## Executive Summary

`qwen4_exp` (Qwen3.8-Flash-Next) now snapshots and restores its per-sequence state, so a follow-up turn no longer prefills from token zero. On the OptiQ 2-bit checkpoint, a second turn reused 4327 of 4363 prompt tokens with a 239.8 MB snapshot. This PR is Part A of #2885. Part B (mid-prompt checkpoints) is not implemented and the issue stays open.

---

## 1. Problem Statement

Every request for this family prefilled from the start. At the measured 1,057 tok/s prefill on M3 Ultra, a 30K-token agent context costs about half a minute per turn before the first output token. The `LanguageModel` snapshot hooks existed but `qwen4_exp` implemented none.

A missing piece of state fails silently: restoring KV without the n-gram contexts still produces fluent, wrong text. Tests therefore compare against cold prefill bit for bit.

---

## 3. Technical Decisions

### 3.1 Snapshot only when the sequence is fully representable

- **Image and video requests decline.** A sequence with MRoPE state does not snapshot and prefills cold, with the reason logged at debug. Carrying the state would have to interact with the whole-entry rule for multimodal requests.
- **A capture with a pending speculative entry is refused** when its attention offset differs from `token_len`. Restores install an empty rollback log, so a rewind past the restore point errors rather than corrupting state.
- **Truncation stays false**: recurrent state cannot be cut, as for Qwen 3.5.

### 3.2 Block keys are not stored

The next selection recomputes block keys to identical bits from the stored raw keys and positions. The issue's negative test ("omitting block keys makes the comparison fail") is therefore false at the logit level. The test instead asserts that omission reproduces the logits, that the restored cache lacks the keys, and that zeroed block keys do fail. Omitting the n-gram contexts fails as specified.

### 3.3 Stepped growth through `AvailabilityGate`

Issue #2883 step 4 asked for stepped growth of prompt-cache snapshots and the store had none. Snapshot growth now checks `AvailabilityGate`. A store already at its budget is never refused.

---

## 7. Change Summary

| Item | Value |
|------|-------|
| Files changed | 12, plus these reports |
| Lines added | +1438 |
| Lines deleted | -7 |
| Snapshot tests | 12 passed with `--test-threads=1` |

| Hash | Type | Message |
|------|------|---------|
| `ea91fd83` | feat | feat(models): add prompt-cache snapshots for qwen4_exp |
| `33414a4d` | fix | fix(models): refuse qwen4_exp snapshots missing recurrent state |

---

## 8. Follow-up Actions

### Required

- [ ] Part B of #2885: system-prompt root, periodic checkpoints, gap eviction, each behind its own switch.
- [ ] Time to first token for the second turn of a 16K and a 32K-token conversation, with and without the cache.

### Open Verification Gaps

- The true second turn (hit at 4327, inside the cold plan's history piece ending at 4358) diverged from the cache-reset run at a near-tie, at character 577 of the reasoning. This matches the partition class in `docs/CONTINUOUS_BATCHING.md` (a hit from inside a piece). It was not separately proven to be rounding alone, and the issue's "equals the cache-reset run" criterion is not met for this hit.
- Restoring against a real external PLE table was not exercised; a delegating test store stands in for it.

### Known Limits

- The growth gate charges a whole entry even when an insert replaces one, and refusals appear only in logs, not in `/v1/cache/stats`.
- Parallel runs of the snapshot and rollback test modules abort or flake from MLX thread safety. `make verify-test` runs them with `--test-threads=1`.

---

## Appendix

### A. Measurements

OptiQ 2-bit through `mlxcel-server`:

- Second turn: `cached_tokens` 4327 of 4363, one snapshot hit.
- Repeated request, a hit at its own history boundary (4327 of 4332): all 256 greedy tokens equal the cache-reset run.
- Snapshot size 239,833,892 B at 4327 tokens. Store capacity 2,096,805,888 B. The sizing estimate is within 0.4%.

### B. References

- Issue #2885, Issue #2883
- `docs/CONTINUOUS_BATCHING.md`, prefill partition invariant
