# Technical Report: PR #1158 - test: derive prompt_cache_e2e slack from the paged block size

**Date**: 2026-08-15
**Author**: AI Code Reviewer
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low

---

## Executive Summary

The manual multi-turn prompt-cache e2e test (`tests/prompt_cache_e2e.rs`, `#[ignore]`d, run against local qwen3-0.6b-4bit weights) failed deterministically at turn 3 because its under-allowance assertion granted only 4 tokens of slack while the APC lookup path floors cache credit to whole 16-token blocks, which can legitimately cost up to 15 tokens. This PR derives the slack from the server's own `DEFAULT_APC_BLOCK_SIZE` constant (slack = block size - 1 = 15), documents the flooring mechanism accurately, and scrubs the `APC_BLOCK_SIZE`/`APC_ENABLED` env-var fallbacks when spawning the server so the assertion's premise holds regardless of the invoking shell. Test-only change; six recorded runs with the real fixture produce identical per-turn values.

---

## 1. Problem Statement

### 1.1 Background

Issue #1156 was filed from epic #1148's integration verification. A two-arm measurement settled attribution: the pre-epic baseline `9c154ff3` fails with token-for-token identical values (cached 0/48/64/96/112 across 5 turns), so the defect is pre-existing in the test, not an epic regression.

### 1.2 Existing Issues

- **Issue 1**: At turn 3, `cached_tokens` is 64 against a previous prompt length of 73, and the assertion `cached + 4 >= prev_prompt_len` fails: the 4-token slack does not cover block flooring.
- **Issue 2**: The mechanism: APC lookup credits only whole verified blocks (`apc_consistent_prefix_len` returns `consistent_blocks * block_size`, `src/server/prompt_cache/apc_lookup.rs`), and the dense adopt path truncates to exactly that. With `max_tokens=16` the qwen3 thinking model emits empty assistant content (all 16 tokens go to the unterminated think block), so the history re-render diverges right at the previous prompt boundary and the flooring loss is fully exposed: worst case `block_size - 1 = 15` tokens, which exceeds 4.
- **Issue 3** (found during review): the spawned server inherited the parent environment, and `mlxcel-server` falls back to `APC_BLOCK_SIZE`/`APC_ENABLED` env vars when the flags are absent — an exported `APC_BLOCK_SIZE` would silently change the server's block size while the assertion kept using the compile-time constant.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Deterministic false failure hides real cache regressions behind a known-red test | Medium | High (fails every run at turn 3) |
| Hardcoded slack silently diverges if the block size constant changes | Low | Low |

---

## 2. Technical Review

### 2.1 Security

No security surface (test-only). The security pass verified 15 is the tight bound, arithmetic is overflow-safe (const-evaluated; `cached + 15` wrap requires `cached` near `u64::MAX` where the assertion passes either way), and surfaced the env-var inheritance finding (MEDIUM) that the orchestrator fixed with `.env_remove` in `spawn_server` (precedent: `tests/cli_help_consistency.rs`).

**Issues Found:**
| Issue | Severity | Status |
|-------|----------|--------|
| False justification comment: claimed the crate constant was unimportable (root package "has no lib target") — it does, and a sibling test already imports the exact path | High | Fixed (`203d1118`) |
| Mechanism misattribution in the comment: credited "dense-trie flooring"/"whole-block donation"; donation stores exact tokens and the trie returns an exact LCP — the floor is at APC lookup | Medium | Fixed (`203d1118`) |
| Inherited `APC_BLOCK_SIZE`/`APC_ENABLED` env vars could break the assertion's premise both ways (32 → false failure; 8 → over-permissive) | Medium | Fixed (`45d8757c`) |
| Pre-existing doc bug in `apc_lookup.rs:46` ("input matched_len is preserved" — false, return is always block-floored) | Low | Open (production doc, out of scope; flagged on the PR) |

### 2.2 Performance

None; assertion arithmetic only. The prefill-latency assertion continued to pass in every recorded run (ratios 0.37-0.50 against a 1.3 upper bound).

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: none
- **New Dependencies**: none — the lib import adds no build surface (18 other files under `tests/` already import from the `mlxcel` lib)
- **Compatibility**: valid only for the Dense decode backend (`--batch-size 1`); the paged backend applies a coarser 32-token floor (`DEFAULT_PAGED_BLOCK_SIZE`), now documented on the constant

### 2.4 Code Quality

- **Test Coverage**: the previously always-red manual test is now a usable regression gate
- **Code Complexity**: one imported constant, one assertion change, two env removals
- **Technical Debt**: decreased (magic number replaced with the source-of-truth constant; env-dependent premise closed)

---

## 3. Technical Decisions

### 3.1 Derive the slack from the constant instead of widening the literal or changing the fixture

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Raise the literal to 15 | Minimal diff | Silently diverges if the block size changes |
| Change the fixture (non-thinking model or larger `max_tokens`) | Avoids the empty-content divergence | Hides rather than accommodates the mechanism; the flooring loss still exists for any divergence at a non-aligned boundary |
| **Chosen: import `DEFAULT_APC_BLOCK_SIZE`, slack = block size - 1** | Assertion tracks the source of truth; 15 is provably the tight bound | Requires the (correct) importability of the crate constant |

**Rationale:** The acceptance criteria preferred derivation from the constant. The review pass proved 15 is exactly tight: a one-block regression gives `cached <= P - 16`, so `cached + 15 <= P - 1` and the assertion still fires — a full block is the smallest loss the dense APC path can express. All observed values match `floor(P/16)*16` exactly (48/64/96/112 against 50/73/96/119), confirming the flooring model empirically.

### 3.2 Scrub env fallbacks in `spawn_server` rather than pass `--apc-block-size` at each call site

**Rationale:** `.env_remove("APC_BLOCK_SIZE").env_remove("APC_ENABLED")` fixes every current and future spawn in the file at one point, matching the existing precedent in `tests/cli_help_consistency.rs`. Passing the flag would need repeating at each call site and still leave `APC_ENABLED` inherited.

---

## 4. Implementation Details

### 4.2 Key Code Changes

**File: `tests/prompt_cache_e2e.rs`**
```rust
// Before
assert!(
    cached + 4 >= prev_prompt_len,
    ...
);

// After
use mlxcel::server::prompt_cache::DEFAULT_APC_BLOCK_SIZE;
const APC_BLOCK_SIZE: u64 = DEFAULT_APC_BLOCK_SIZE as u64;
...
assert!(
    cached + (APC_BLOCK_SIZE - 1) >= prev_prompt_len,
    ...
);
```

**Reason for change:** APC lookup credits only whole verified blocks, so a re-render divergence at the previous prompt boundary can cost up to `block_size - 1` tokens; the assertion must allow exactly that and no more. The doc comment records both preconditions: Dense backend via `--batch-size 1` (paged floors at 32), and the env scrub in `spawn_server`.

---

## 7. Change Summary

### Statistics
| Item | Value |
|------|-------|
| Files changed | 1 |
| Lines added | +44 |
| Lines deleted | -2 |
| Tests added | 0 (1 existing manual test repaired) |

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| Code Quality | 3 | Constant-derived slack, accurate mechanism docs, env-fallback scrub |

### Related Commits
| Hash | Type | Message |
|------|------|---------|
| `c3b33cd8` | test | derive prompt_cache_e2e slack from the paged block size |
| `203d1118` | test | import the APC block size instead of mirroring it |
| `45d8757c` | test | scrub APC env fallbacks when spawning the e2e server |

---

## 8. Follow-up Actions

### Required
- [ ] None; all three acceptance criteria are met with real-model evidence

### Future Improvements
- Fix the `apc_lookup.rs:46` doc comment ("input matched_len is preserved" is false; the return is always `consistent_blocks * block_size`) — pre-existing production doc bug flagged on the PR
- If the test ever raises `--batch-size`, the slack must widen to `DEFAULT_PAGED_BLOCK_SIZE - 1` (documented on the constant)

---

## Appendix

### A. Test Results

Six recorded runs of `cargo test --release --features metal,accelerate --test prompt_cache_e2e multi_turn -- --ignored` with the local qwen3-0.6b-4bit fixture (2 by the implementer, 2 by the reviewer, 1 by the security pass baseline evidence, 1 by the finalizer post-env-scrub), all with identical per-turn values:

| Turn | prompt_tokens | cached_tokens | flooring check |
|------|---------------|---------------|----------------|
| 1 | 50 | 0 | cold start |
| 2 | 73 | 48 | floor(50/16)*16 = 48 |
| 3 | 96 | 64 | floor(73/16)*16 = 64 (the previously failing case: 64 + 15 >= 73) |
| 4 | 119 | 96 | floor(96/16)*16 = 96 |
| 5 | 142 | 112 | floor(119/16)*16 = 112 |

Observed slack consumption per turn: 2/9/0/7 of the 15 allowed. Strict assertions (cached > 0 from turn 2, monotonic growth) unchanged and passing.

### C. References
- Issue #1156 (specification), epic #1148 (integration verification, full turn table in its summary comment)
- `src/server/prompt_cache/apc_lookup.rs` (flooring site), `src/server/prompt_cache/block_hash.rs` (`DEFAULT_APC_BLOCK_SIZE`), `src/server/batch/scheduler.rs` (dense adopt truncation, `DEFAULT_PAGED_BLOCK_SIZE`)
- PR #1158 review and security comments
