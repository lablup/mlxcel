# Technical Report: PR #1097 - Document Florence-2 large-ft upstream tracker

**Date**: 2026-08-10
**Status**: Completed for an open PR
**Languages**: Markdown
**Risk Level**: Low

## Executive Summary

PR #1097 narrows the documented Florence-2 `large-ft` failure mode from a suspected mlxcel loader defect to an upstream MLX checkpoint or conversion-family problem. The change keeps `mlx-community/Florence-2-base-ft` as the recommended working baseline and links the focused upstream tracker so users can follow the external resolution path directly.

This is a documentation-only correction, but it matters operationally because the supported-models page is the first place users look when a published checkpoint loads yet produces unusable output.

## 1. Problem Statement

Issue #1085 captured that Florence-2 `-large-ft` checkpoints load but return degenerate output. Before this PR, the support note already said mlx-vlm reproduced the failure, but it stopped short of linking the upstream tracker or recording the known provenance of the affected 4-bit conversion.

Without that extra context, readers still had to infer whether mlxcel's loader was the likely fault domain, and they had no direct pointer to the external issue that now owns the remaining investigation.

## 2. Technical Decisions

### 2.1 Reframe the limitation as an upstream checkpoint-family issue

The edited sentence now says the behavior appears to be a property of the published `large-ft` MLX checkpoint or conversion family rather than this loader. That wording matches the evidence gathered in issue #1085: both mlxcel and upstream mlx-vlm reproduce the same bad outputs on the same releases.

### 2.2 Link the specific upstream tracker

The reportable resolution path is `Blaizzy/mlx-vlm#1840`, not a vague "upstream issue exists" statement. Putting the link in `docs/supported-models.md` turns the support page into an actionable handoff instead of a dead-end warning.

### 2.3 Preserve the last known-good recommendation

The note continues to recommend `-base-ft` until the upstream issue is resolved. That keeps the document aligned with the validated baseline rather than implying that all Florence-2 variants are equally reliable.

## 3. Change Summary

| Item | Value |
|------|-------|
| Files changed | 1 |
| Lines added | 1 |
| Lines deleted | 1 |
| Scope | Documentation only |

- Updated the Florence-2 support note in `docs/supported-models.md`.
- Added a direct link to `Blaizzy/mlx-vlm#1840`.
- Recorded that `mlx-community/Florence-2-large-ft-4bit` was converted from `prince-canuma/Florence-2-large-ft` with `mlx-vlm 0.1.0`, based on the model-card provenance already attached to the affected release.
- Kept `mlx-community/Florence-2-base-ft` as the documented working fallback.

## 4. Review Findings

| Finding | Severity | Resolution |
|---------|----------|------------|
| The support page left the failure domain ambiguous after upstream reproduction was known | Medium | Clarified that the remaining fault is attributed to the published `large-ft` MLX checkpoint or conversion family, not the mlxcel loader |
| Users had no direct route from the support note to the live upstream investigation | Low | Linked `Blaizzy/mlx-vlm#1840` inline |

No code-path, security, or performance findings applied because the PR changes only prose.

## 5. Validation

- `git diff --check origin/main...HEAD`: passed.
- Minimal MkDocs render of the changed page: passed with a temporary config, confirming the edited document still renders.
- Live repository follow-through: passed. The reconciled evidence and upstream link were posted on `lablup/mlxcel#1085`, and the focused upstream issue was filed as `Blaizzy/mlx-vlm#1840`.
- Canonical repository docs build distinction: `mkdocs build -f mkdocs.yml -q` could not start in this checkout because the checked-in config references missing `docs/overrides` and `docs/en` directories.
- `make docs-build` distinction: unavailable in this environment because `zensical` is not installed.

The important boundary is that the changed page itself rendered successfully, but the repository's canonical docs pipeline was not runnable from this checkout for reasons unrelated to PR #1097.

## 6. Related Work

- PR #1097: https://github.com/lablup/mlxcel/pull/1097
- Issue #1085: https://github.com/lablup/mlxcel/issues/1085
- Upstream issue `Blaizzy/mlx-vlm#1840`: https://github.com/Blaizzy/mlx-vlm/issues/1840
