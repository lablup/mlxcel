# PR #2507: PhiMoE norm epsilon key and SAM antialiased bicubic a

**Date**: 2026-10-21
**Status**: Completed
**Risk Level**: Low
**Languages**: Rust

---

## Executive Summary

Closes #2465. Two numeric-parity fixes left over from merged model PRs: PhiMoE now honors the checkpoint's `rms_norm_eps` spelling (Phi-3.5-MoE checkpoints spell the norm epsilon `rms_norm_eps: 1e-5`, but `ModelArgs` only read `layer_norm_eps`, so the checkpoint value was ignored and norms ran at the wrong epsilon), and the SAM vision encoder's antialiased bicubic resize now uses ATen's `a = -0.5` instead of Keys' `-0.75`. Both are wired into the real parse and preprocessing paths, pinned by tests against a committed torch-generated fixture.

---

## 1. Problem Statement

### 1.1 Background

Two small numeric-parity gaps survived PR #2457 (found during the 0.4.0 release sweep) and PR #2459 (SAM left bit-identical at the time). One PR, one verification pass.

### 1.2 Existing Issues

- **PhiMoE epsilon**: `phimoe::ModelArgs` declared `layer_norm_eps` (default 1e-6); Phi-3.5-MoE checkpoints spell it `rms_norm_eps: 1e-5`. The checkpoint value was silently ignored (serde drops unknown fields), so all LayerNorm blocks ran at 1e-6 instead of 1e-5.
- **SAM resize coefficient**: the antialiased bicubic resize in `resample_matrix.rs` used `a = -0.75` (Keys, OpenCV-compatible), while ATen's antialiased kernel and mlx-vlm use `a = -0.5`. SAM's grid resample therefore diverged from the reference preprocessing.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| PhiMoE norms at the wrong epsilon vs reference checkpoints | Low (1e-5 vs 1e-6 norm scale) | High (every Phi-3.5-MoE checkpoint) |
| SAM preprocessing mismatch vs mlx-vlm/torch | Low (numeric parity) | High (every SAM image) |

---

## 2. Technical Review

### 2.1 Security

Clean. The serde alias only accepts a second spelling for the existing f32 field: hostile JSON cannot panic (NaN/Infinity literals are serde parse errors, out-of-range numbers degrade without panic, both spellings in one config produce a serde duplicate-field Err). No shared or mutable state added (grep of all changed lines for static/once_cell/OnceLock/Mutex/RwLock: nothing). The committed `generate_reference.py` is documentation-only, never executed by tests; the fixture JSON is `#[cfg(test)]`-gated via `include_str!`.

### 2.2 Performance

No hot-path change. `interp_matrix` is per-model-load (matrix build), not per-token; the coefficient change is two comparisons. PhiMoE's epsilon is read once at LayerNorm construction (phimoe.rs:514/524/614). Production binary unaffected by the fixture (test-gated include).

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: none. Field name kept (`layer_norm_eps`), so the three `LayerNorm::new` call sites are unchanged; the old spelling keeps parsing; 1e-6 stands when neither key exists.
- **New Dependencies**: none.

### 2.4 Code Quality

- Test Coverage: phimoe 15 passed (new: checkpoint-spelling fixture, old-spelling-still-accepted, defaults-without-either-key), resample_matrix 9 passed (new: torch AA matrix parity incl. the 16→24 upsample, SAM grid parity vs torch's direct 2-D AA, non-antialiased bit-identical pin, rows-sum-to-one), deepseekocr 22 passed. The resample filter failed 3 of 9 on pre-fix code.
- Fail-on-main proof: the checkpoint-spelling config test fails on main (unknown key ignored → 1e-6); the `-0.75` coefficient misses torch by 1.1e-02 (downsample) to 3.6e-02 (upsample) against the committed fixture.

---

## 3. Technical Decisions

### 3.1 serde alias for the checkpoint spelling

**Context:** `ModelArgs` sits behind the remote-derive path (`impl_rope_lifting_deserialize!`), which lifts rope keys from a JSON Value before delegating to serde.

**Chosen:** `#[serde(default = "default_layer_norm_eps", alias = "rms_norm_eps")]` on the existing field. The alias survives the remote-derive delegation (serde derive processes field attributes), the field name is kept so no call site changes, and the default only applies when neither key exists. Both spellings in one config is a serde Err, acceptable for a config shape no checkpoint produces.

### 3.2 a = -0.5 whenever the cubic kernel is antialiased (not only when downsampling)

**Context:** the issue proposed gating `-0.5` on `scale > 1` (the stretched-support path).

**Chosen:** deviate, with evidence: torch 2.14.1 uses `a = -0.5` for the antialiased kernel in BOTH directions. Probe (committed fixture): 16→24 upsample, `a=-0.5` matches torch within 6.1e-07 while `-0.75` misses by 3.6e-02; 16→5 downsample: 1.0e-07 vs 1.1e-02. The issue's own Implementation Notes instructed to confirm the upsampling coefficient and mirror it. No production caller upsamples with antialias (SAM downsamples 1024→64), so no existing behavior changes beyond the AA-downsample path. Support still stretches only when downsampling (`filterscale = max(scale, 1.0)`), matching PIL/ATen.

### 3.3 Non-antialiased path untouched

Public `cubic()` still delegates to `cubic_convolution(x, -0.75)`; kimi_vl's MoonViT edge-replicated grid stays byte-identical, pinned by the pre-existing `moonvit_bicubic_matrix_is_bit_identical` and a new `nonantialiased_cubic_matrix_is_unchanged` (16→5, 16→8 bit-for-bit plus same-size identity).

---

## 4. Implementation Details

**File: `src/models/phimoe.rs`** — `layer_norm_eps` gains `alias = "rms_norm_eps"`; doc comment records the checkpoint spelling (the key mlx-lm's `phimoe.py` reads).

**File: `src/vision/encoders/resample_matrix.rs`** — `interp_matrix` selects `a = -0.5` when `kernel == Cubic && antialias`, else `-0.75`; module docs updated.

**Tests:** `src/models/phimoe_tests.rs` (2 new tests + checkpoint-spelling fixture), `src/vision/encoders/resample_matrix_tests.rs` (3 new tests incl. fixture consumption via `include_str!`).

**Fixture:** `tests/fixtures/deepseekocr_sam/{reference.json, generate_reference.py}` — AA cubic interp matrices for 16→5/8/24, SAM grid resamples at (5,5), (5,7), (24,24); cross-checks recorded: mlx-vlm vs torch max abs diff 9.76e-07, separable vs direct 2.38e-07. Generated with torch 2.14.1+cpu and mlx-vlm 0.7.8 (commit 73d366ecf9...).

---

## 5. Verification

| Check | Result |
|---|---|
| cargo test phimoe (--test-threads=1) | 15 passed / 0 failed |
| cargo test resample_matrix (--test-threads=1) | 9 passed / 0 failed (3/9 fail pre-fix) |
| cargo test deepseekocr (--test-threads=1) | 22 passed / 0 failed |
| cargo fmt --all -- --check | clean |
| scoped clippy (lib+tests, -D warnings) | clean |
| Full gate (161 bins, HEAD 6ec45810) | build 0, clippy 0, doc 0, contract 0; binaries green (mllama_parity 7/1 = known #2467 flake, rerun green) |
| mlx-lm oracle vs fixed mlxcel (phi-3.5-moe-4bit) | both start "Paris.", diverge at token 3 = cross-device numeric path (oracle CPU mlx 0.32.4 vs mlxcel CUDA), recorded per the #932 contract, not claimed token-exact |
| CONTROL (decisive) | fixed mlxcel against a config with rms_norm_eps removed == identical 16-token stream: the epsilon does not alter generation on this prompt; the fix is proven by the unit tests |
| Security review | CLEAN, 0 findings all severities |

Upstream premise verified: mlx-community/Phi-3.5-MoE-instruct-4bit config.json spells `rms_norm_eps: 1e-05`.

---

## 6. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 6 |
| Lines added | +3526 (fixture ~3.4k) |
| Lines deleted | -16 |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `6ec45810` | fix | fix(models): PhiMoE norm epsilon key and SAM antialiased bicubic a |

---

## 7. Follow-up Actions

### Monitoring Required

- Streamed tool calls and other python_tag-producing families are unrelated here; no release-monitoring item from this PR.
- The local gitignored checkpoint copy `models/mlx/phi-3.5-moe-4bit/config.json` currently lacks `rms_norm_eps` entirely (unlike upstream); local model runs therefore do not exercise the alias, while committed coverage does.

### Future Improvements

- None deferred from this PR.

---

## Appendix

### A. Review verdicts

pr-reviewer: PASS on all six focus areas, 0 findings (line-drift citations only, from intervening merges). pr-security-checker: CLEAN, no CRITICAL/HIGH/MEDIUM/LOW (serde alias cannot panic on hostile JSON, no shared state, renormalization sum stays positive pinned by test, fixture test-gated). pr-finalizer: complete, no gaps; all PR-body claims independently reproduced (pure-Python f32 probe against the committed fixture, upstream config fetched, scoped clippy exit 0).
