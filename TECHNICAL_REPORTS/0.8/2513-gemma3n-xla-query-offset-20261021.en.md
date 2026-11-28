# PR #2513: Gemma 3n emitter moved to the shared-layer query offset

**Date**: 2026-10-21
**Status**: Completed
**Risk Level**: Medium
**Languages**: Rust

---

## Executive Summary

Closes #2466. The mlxcel-xla Gemma 3n emitter still rotated shared-KV-layer queries at `positions + real_len` (prefill) and `position + 1` (decode), mirroring the pre-#2457 MLX behavior, so MLX-vs-XLA Gemma 3n parity diverged after #2457 fixed the MLX path. This PR ports the post-#2457 offset semantics to both graphs (rope table shrunk from 2*capacity to capacity rows), refreshes the `#[ignore]`d xla-diagnostics pins with provenance, and compiles the previously-unbuilt xla-diagnostics call sites in `src/models/gemma3n.rs`.

---

## 1. Problem Statement

### 1.1 Background

PR #2457 fixed the Gemma 3n shared-KV layer query offset in the MLX path (`rope_offset` read once per forward, matching mlx-lm's `cache[0].offset`). The XLA emitter was deliberately left mirroring the old offset at the time. Related to epic #566.

### 1.2 Existing Issues

- Prefill: the emitter built shared-layer query positions as `positions + real_len` while the fixed MLX path rotates ALL layers' queries at the same pre-read offset.
- Decode: the emitter advanced shared-layer queries by `position + 1` per step while MLX rotates at the pre-read `position`.
- The xla-diagnostics call sites in `src/models/gemma3n.rs` were updated by hand after #2457 and never compiled (the diagnostics feature surface is not part of default builds).

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| MLX-vs-XLA Gemma 3n parity divergence on shared-KV layers | Medium (wrong query rotation) | High (every Gemma 3n XLA request) |
| Stale pre-#2457 pins presenting old behavior as correct | Low | High |

---

## 2. Technical Review

### 2.1 Security

Clean. Emitter math only plus test pins: no new input-parsing surface, no shared or mutable state added (grep of all added lines for static/once_cell/OnceLock/Mutex/RwLock/unsafe: zero), no IREE/runtime changes. The shrunk rope table's boundary is verified on all three decode paths (single-token decode errors unless 0 <= cache_len < context_capacity; ragged decode validates every row before execution; the admission invariant bounds cache_len).

### 2.2 Performance

Strictly favorable. Rope tables halve: 4 const tensors [2C,d] -> [C,d] baked into the module (for C=32768/d=256: 128MB -> 64MB of module constants), a one-time module-load/compile win. Per-token decode drops one scalar add per shared-KV layer per step; prefill drops one [rows]-element i32 broadcast+add per shared layer.

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: none. The servable schema is untouched (`real_len` remains a graph input, still selecting the output row at gemma3n_emit.rs:461).
- **New Dependencies**: none.

### 2.4 Code Quality

- Test Coverage: gemma3n 40 passed / 0 failed (`--features cuda,xla-diagnostics`, 22 pre-existing ignores), xla_gemma3n_prefill `--ignored` 1 passed / 0 failed on MLXCEL_XLA_DEVICE=cuda (token vs dense-PLE greedy equivalence, near-capacity prefill driving query RoPE through exactly capacity-1, mixed cancel/reuse batch). Crate unit tests 364/0. No gemma3n registered graph fixture exists; the family's gate per the #932 contract is the pinned diagnostics, verified in both directions by direct measurement on GB10.
- Feature naming: the root crate has NO plain `xla` feature; the surface is xla-iree (production) and xla-diagnostics (cuda + iree + diagnostics). The issue's `cargo build --features xla` cannot exist.

---

## 3. Technical Decisions

### 3.1 Port the post-#2457 semantics exactly

**Context:** the fixed MLX path reads `rope_offset(caches)` once per forward and threads it into every `DecoderLayer::forward`; all layers (shared and concrete) rotate queries at that pre-read offset, and shared layers slice the owner's live K/V window.

**Chosen:** remove the shared-layer `positions + real_len` block (prefill) and the `position + 1` block (decode) so every layer rotates queries at the same positions; shrink the rope table from 2*context_capacity to context_capacity rows (max gather index capacity-1 fits exactly; a decode at position == capacity was already invalid because the K/V write itself would be out of bounds). Verified against the #2457 diff and the current MLX forward paths.

### 3.2 Refresh the ignored pins with provenance

**Context:** the xla-diagnostics pins (top1 236798) predate #2457 and were never refreshed because the tests are `#[ignore]`d.

**Chosen:** refresh the prefix-decode top1 pin 236798 -> 854, re-captured on main at 551039ee (MLX pin bff10220) with the mlx-vlm 0.7.8 CPU oracle agreeing; keep full-prefill and greedy pins unchanged; record the cross-device close-ranking flip (CPU ranks 854 first for single-pass full prefill by 0.75 logits) as recorded, not gated, per the #932 maintainer contract.

### 3.3 Residual drift is recorded, not hidden

The prefill gap the issue targets is resolved (token-prefill logits max_abs 1.9375 -> 0.8203, within the 1.5 envelope, matching top1). The prefix-decode comparison stays outside the pinned envelope both before (2.2617, cosine 0.9432) and after (2.7266, cosine 0.8710), and every stage the fix cannot touch (scaled_embeddings 0.03125 vs envelope 0, layer0_ple_injected 2 vs 0.25, layer-mid planes 15 vs 3) measures byte-identical on the pre-change emitter, proving that residual predates both #2457 and this PR (the pin was last captured 2026-07-24 and the MLX pin has since moved via PR #2505). Per the #932 contract, recorded drift is not gating.

---

## 4. Implementation Details

**File: `src/lib/mlxcel-xla/src/emitter/gemma3n_emit_ops.rs`** — `constants()` shrinks the rope table; `attention()` drops the real_len parameter and the shared-layer offset block; `attention_decode_traced()` removes the shared-layer per-step block.

**File: `src/lib/mlxcel-xla/src/emitter/gemma3n_emit.rs`** — call site drops `&args.real_len`; `args.real_len` still feeds the last-token dynamic slice.

**File: `src/models/gemma3n.rs`** — pin refreshes and provenance comments inside the `#[cfg(all(test, feature = "xla-diagnostics"))]` module; no production change; diagnostics call sites compiled clean.

**File: `tests/xla_gemma3n_prefill.rs`** — full-capacity comment refreshed.

---

## 5. Verification

| Check | Result |
|---|---|
| cargo test gemma3n (cuda,xla-diagnostics, --test-threads=1) | 40 passed / 0 failed |
| xla_gemma3n_prefill --ignored (MLXCEL_XLA_DEVICE=cuda) | 1 passed / 0 failed |
| mlxcel-xla crate unit tests | 364 passed / 0 failed |
| cargo fmt (changed files) / scoped clippy | clean / clean |
| Full gate (161 bins, HEAD 4a1b99fc) | build 0, clippy 0, doc 0; 160/161 (mllama_parity 7/1 = known #2467 1-ulp flake, control-proven environmental on main); CT_EXIT=2 = pre-existing main breakage from parallel PR #2505 (kernel-port dispatch, env-var docs, fmt), independent of this PR and tracked separately |
| Fail-before (pre-change emitter, direct measurement) | token-prefill max_abs 1.9375 FAIL (envelope 1.5); prefix-decode top1 xla=236798 vs mlx=854 (reproduces pre-#2457 MLX) |
| Post-fix (direct measurement) | token-prefill 0.8203 within envelope, matching top1; greedy [236798, 236789] both sides |
| Security review | CLEAN; boundary verified all 3 decode paths w/ admission invariant; rope tables halve (module constants 128MB -> 64MB for 32768/256) |
| IREE toolchain note | the pinned pair .cache/mlxcel/iree-cuda-3.12.0rc20260721 + its venv is the matched one; the Development/iree-cuda tree's runtime is older (hal version mismatch 6 vs 7) |

---

## 6. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 4 |
| Lines added | +35 |
| Lines deleted | -28 |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `4a1b99fc` | fix | fix(xla): update Gemma 3n emitter to the shared-layer query offset |

---

## 7. Follow-up Actions

### Monitoring Required

- The pinned diagnostics test still fails when explicitly run due to the recorded pre-existing drift stages; every failing stage is byte-identical on the pre-change emitter.
- The static 0..capacity-1 positions bucket means a hypothetical continuation prefill at a nonzero offset would still diverge; Gemma 3n opts out of chunked/multi-call prefill, so this is unreachable on the current paths.

### Future Improvements

- `mod gemma3n_math` (src/lib/mlxcel-xla/src/emitter/mod.rs:59) is dead code (dead at base, last touched at #922) with a stale pre-#2457 attention_bias oracle; candidate follow-up cleanup issue.
- A non-ignored rope-table-capacity test following the existing tiny() conventions would pin the shrunk table's shape without the pinned checkpoint.

---

## Appendix

### A. Review verdicts

pr-reviewer: PASS all six focus areas, 0 findings (semantic port verified against the #2457 diff and current MLX forward paths; ragged batched decode and the serve path flow through the fixed function; rope-table boundary sound). pr-security-checker: CLEAN, no CRITICAL/HIGH/MEDIUM; two LOW informational notes (dead gemma3n_math module; the 854 pin is statically unverifiable since the tests are ignored). pr-finalizer: complete, no gaps; every PR-body claim reproduced by direct measurement on GB10 (fail-before and post-fix numbers, residual table, IREE toolchain pair).
