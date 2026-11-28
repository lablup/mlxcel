# Technical Report: PR #1488 - b10621 speculative flag mapping

**Date**: 2026-08-28

**Status**: Completed

**Languages**: Rust

**Risk Level**: Medium

## Executive Summary

Classifies all 52 b10621 speculative-decoding manifest entries owned by issue #1433: 45 options and 7 native `/completion` dotted fields. One entry is `supported` (`--spec-draft-model` with its canonical env), three are `aliased` with translation tests (`--spec-type`, `--spec-draft-n-max`, and the removed `--draft` spellings kept alive), and the rest are `not_applicable` through a new inert-or-reject classification module mirroring the #1445 GGML pattern. No decode arithmetic changes; MTP / DFlash themselves are untouched.

## 1. Problem Statement

b10621 exposes over forty speculative flags spanning draft-model selection, draft-sampler thresholds, GGML draft-side process placement, n-gram speculation, and lookup decoding. mlxcel's speculative decoding is MTP / DFlash with server-wide configuration, and before this PR most of the b10621 surface either failed to parse or could be silently mis-parsed (`-md` swallowed as `-m` with value `d`).

## 2. Technical Decisions

### 2.1 The selector translates, everything untranslatable rejects

The review corrected the central misreading: `--spec-type` is b10621's speculation-subsystem selector, and an explicit `none` disables speculation even with a draft model configured (the selector stops upstream's draft-sidecar type inference). `resolved_spec_type` now translates exactly what mlxcel can run: `none` drops the configured draft model with a warning, `draft-mtp` and `draft-dflash` map onto `--draft-kind` (conflicts with an explicit kind fail deterministically), and the n-gram modes, `draft-simple`, `draft-eagle3`, `draft-dspark`, and multi-subsystem lists fail startup with per-value diagnostics.

### 2.2 Full spelling parity, classified values

Every b10621 long alias (22 beyond the canonical spellings) and 17 short spellings parse on both binaries; the hidden `SpecCompatArgs` surface classifies each value as inert (upstream defaults, full-offload `--spec-draft-ngl` spellings including the historical negative, `f16` draft cache types, any n-gram tuning value while no n-gram selector can be chosen) or rejected before the model load with a diagnostic naming the limitation and the mlxcel alternative. `--no-spec-draft-backend-sampling`, the operator-active half of its pair, is rejected, and its environment variable follows b10621's bool-pair rules.

### 2.3 Native dotted fields: match upstream's accept-and-ignore

b10621 registers `speculative.n_max` and its six siblings as flat dotted top-level keys behind a schema block that is compiled out, so upstream accepts and ignores them. mlxcel declares the same dotted keys via serde renames and treats them identically as inert; a rejection would refuse requests b10621 answers. A test proves the resolved options are byte-equal with and without the fields.

### 2.4 Canonical env with a guarded legacy fallback

The draft-token cap binds the canonical `LLAMA_ARG_SPEC_DRAFT_N_MAX`; the removed `LLAMA_ARG_DRAFT_MAX` is honored through a pure, precedence-tested fallback that any CLI spelling (including `--draft-n`, the review's regression catch) and the canonical variable both outrank. The `--draft` manifest entry records the real divergence: b10621 aborts startup when the removed variable is exported, mlxcel keeps deployments working.

## 3. Change Summary

| Item | Value |
|------|-------|
| Files changed | 15 in each of two commits |
| Lines | +1995 / -426 across both commits |
| Manifest entries | 1 supported, 3 aliased, 48 not_applicable, 0 deferred remain |

Validation: 11 classification tests plus parse-alias tests on both binaries, the short-flag table test, compat integration 4/4 against rebuilt binaries, manifest checker green, live smoke of the `-md` rewrite, the n-gram rejection diagnostic, and the draft-kind conflict, and a real-checkpoint greedy speculative parity arm (Qwen3-4B target, Qwen3-0.6B drafter, 64 decided positions token-identical to the target-only baseline at a 0.37 acceptance rate).

## 4. Follow-up Actions

- `/props` now reports the resolved `speculative` block (basename-only draft model, kind, n_max); operators read it back instead of grepping startup logs.
- The disaggregated and OpenXLA serving paths inherit the server-wide speculative configuration unchanged; no per-request speculative path exists anywhere, which is what the native dotted fields' inertness documents.
