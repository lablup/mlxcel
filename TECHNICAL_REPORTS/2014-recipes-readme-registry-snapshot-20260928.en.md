# Technical Report: PR #2014 - Add a recipes/README Explaining the Registry Snapshot Lifecycle

**Date**: 2026-09-28
**Status**: Completed
**Languages**: Markdown
**Risk Level**: Low

## Executive Summary

PR #2014 adds `recipes/README.md`, the first documentation in the `recipes/` directory, and a one-line pointer to it from `docs/supported-models.md`. The `recipes/` tree previously held only committed JSON snapshots of the architecture registry with no prose explaining what they are, even though `.github/ISSUE_TEMPLATE/recipe_request.yml` actively routes external contributors into this directory.

## 1. Problem Statement

### 1.1 Background

`recipes/registry/` holds versioned snapshots of `mlxcel arch --json` output, produced by the `make recipes-registry` Makefile target, plus a `CURRENT` pointer file. The mechanism existed and worked, but nothing in the repository explained it in prose beyond a single `CHANGELOG.md` bullet and the Makefile target's own comments.

### 1.2 Existing Issue

- **No landing doc**: a contributor following the recipe request issue template's pointer to `recipes/` found only JSON files and no README at any level.
- **Undocumented lifecycle**: what a snapshot filename means, when `make recipes-registry` should be run, what `CURRENT` is for, and how the snapshot relates to `mlxcel arch` were all only inferable from reading the Makefile target directly.
- **Stale issue premise**: issue #1677's file list (`0.6.0.json`, `0.7.0-beta.1.json`, `CURRENT`) and Makefile line numbers (147-160) were both out of date against the current tree, which also holds `0.7.0.json` and has the target at lines 203-216. The PR was written against the verified current state rather than the issue text.

### 1.3 Risk Assessment

Low. Documentation-only change with no effect on build, runtime, or CI behavior.

## 2. Change Summary

| Item | Value |
|------|-------|
| Files changed | 2 |
| Lines added | 28 |
| Lines deleted | 0 |

- `recipes/README.md` (new, 27 lines): explains that `recipes/` is not where recipes themselves live (that is `mlxcel.ai/recipes`), what a `recipes/registry/<version>.json` snapshot contains, how `make recipes-registry` regenerates it, what `CURRENT` means, and how the snapshot relates to `mlxcel arch` versus `mlxcel arch --json`.
- `docs/supported-models.md` (+1 line): adds a pointer bullet from the existing `mlxcel arch --json` documentation to the new README, so the registry's already-documented JSON schema and its on-disk snapshot mechanics are cross-linked in both directions.

## 3. Technical Decisions

### 3.1 Verify against the current tree, not the issue's file list

**Context**: The issue body listed `recipes/registry/` as holding only `0.6.0.json`, `0.7.0-beta.1.json`, and `CURRENT`, and cited the Makefile target at lines 147-160.

**Rationale**: The current tree also carries `0.7.0.json` (with `CURRENT` pointing at `0.7.0`), and the target is at lines 203-216. Writing the README from the stale list would have produced a doc that was already wrong on merge. The README describes the mechanism by name and pattern (what a snapshot is, how regeneration works) rather than by line number or exact file enumeration, so it does not go stale the same way.

### 3.2 Describe backend fields per snapshot, not as a fixed set

**Context**: The initial draft stated snapshots carry "Metal/CUDA backend status."

**Rationale** (found during review, see Section 4): `mlxcel arch --json` now also emits a `rocm` entry (`src/models/registry.rs`), but the committed `0.7.0.json` and earlier snapshots predate that field. Stating a fixed field set would have been accurate for the committed snapshots but wrong for the next regeneration. The wording was corrected to describe backend fields as a property of when a snapshot was generated, so the doc stays correct as the schema grows.

## 4. Validation

- `python3 "$HOME/.claude/skills/commit-conventions/scripts/validate_body.py" recipes/README.md`: passed (no hard-wrapped paragraphs).
- `python3 scripts/ci/check_cross_repo_refs.py`: passed (no bare cross-repo issue references introduced).
- Manual cross-check of every factual claim in the README against the actual `Makefile` target, the actual contents and JSON shape of `recipes/registry/*.json` and `CURRENT`, and `.github/ISSUE_TEMPLATE/recipe_request.yml`.
- Independent `pr-reviewer` pass found two accuracy gaps (backend field wording, snapshot-overwrite semantics) and corrected them in a follow-up commit; `pr-security-checker` and `pr-finalizer` passes found no further issues. No markdown linter is configured in this repository, so none was run.
- Not run: no build or test suite applies to a documentation-only change.

## 5. Related Work

- Issue #1677: source issue for this change.
- `docs/supported-models.md`: existing documentation of the `mlxcel arch --json` schema that the registry snapshots serialize.
- `.github/ISSUE_TEMPLATE/recipe_request.yml`: the entry point that routes contributors toward `recipes/`.
