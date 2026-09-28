# Technical Report: PR #2019 - Reject Whitespace-Only Prompts in Embed

**Date**: 2026-09-28
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low

## Executive Summary

PR #2019 fixes `mlxcel embed` accepting whitespace-only prompts (`-p "   "`) while `mlxcel rerank` correctly rejects the equivalent whitespace-only document. `run_embed`'s guard now checks `p.trim().is_empty()` instead of `String::is_empty`, matching `run_rerank`'s existing check, and a misleadingly named `empty` binding (it held an index, not a boolean) is renamed to `index`. A unit test pins both the rejection and the reported index.

## 1. Problem Statement

### 1.1 Background

`run_embed` and `run_rerank` in `src/commands/` share nearly identical input-validation prologues: both reject an empty input list and a missing/non-existent image file with matching wording. The two commands diverged on one check: `run_rerank` rejects a whitespace-only document via `d.trim().is_empty()`, while `run_embed` only rejected a literally empty string via `String::is_empty`.

### 1.2 Existing Issue

- **Inconsistent validation**: `mlxcel embed -p "   "` passed the guard and proceeded to load a model and run a forward pass on an effectively empty input, while the same input to `mlxcel rerank -d "   "` failed fast with a clear error.
- **Misleading binding name**: `args.prompts.iter().position(String::is_empty)` bound its result as `empty`, though `Iterator::position` returns an index, not a boolean. The resulting `bail!("prompt {empty} is empty")` read as if `empty` were a flag.

### 1.3 Risk Assessment

Low. The change only tightens a CLI validation guard that runs before any model resolution or forward pass; it cannot loosen validation or change behavior on already-valid input (any prompt that was rejected before is still rejected, plus whitespace-only ones are now rejected too).

## 2. Change Summary

| Item | Value |
|------|-------|
| Files changed | 1 (`src/commands/embed.rs`) |
| Commits | 2 |
| Lines added / deleted | 38 / 2 |

- `run_embed`'s empty-prompt guard changed from `args.prompts.iter().position(String::is_empty)` to `args.prompts.iter().position(|p| p.trim().is_empty())`, matching `run_rerank`'s `d.trim().is_empty()` in `src/commands/rerank.rs`.
- The binding renamed from `empty` to `index`; the `bail!("prompt {index} is empty")` wording is unchanged.
- Added `run_embed_rejects_whitespace_only_prompts_like_rerank` to `embed.rs`'s existing inline `#[cfg(test)] mod tests { ... }` block, asserting both a single whitespace-only prompt (index 0) and a mixed list where the whitespace entry is not first (index 1).
- A follow-up commit hardened the test's placeholder `EmbedArgs.model` from the bare string `"unused"` (resolvable as a HuggingFace repo-id) to an absolute nonexistent path, so a future regression in the guard would fail the test offline instead of via a network lookup.

## 3. Technical Decisions

### 3.1 Inline test module, not a new `embed_tests.rs`

**Context**: `src/commands/` splits evenly between two test-organization conventions: seven files keep an inline `#[cfg(test)] mod tests { ... }` block (including both `embed.rs` and `rerank.rs`), and seven use an external `<name>_tests.rs` file wired in with `#[path = "..."] mod tests;`.

**Rationale**: `rerank.rs`, this fix's direct reference implementation, already keeps its tests inline. Extending `embed.rs`'s existing inline block keeps the two sibling files consistent with each other, which matters more here than matching the directory's other half.

### 3.2 Testing `run_embed` directly instead of extracting a pure helper

**Context**: `run_embed`'s validation (empty-input check, the whitespace guard, image-existence check) runs before any model resolution or MLX runtime initialization.

**Rationale**: Because the bail happens first, the test calls `run_embed` with a hand-built `EmbedArgs` and asserts on the returned `Err` directly, exercising the real CLI entry point rather than a pulled-out helper function. No tempdir, network access, or model checkpoint is required, and the test completes in effectively 0ms.

## 4. Validation

- `cargo test --release --bin mlxcel commands::embed::tests`: 4/4 passed, including the new test by name.
- `cargo check --lib --tests`: clean.
- `cargo clippy --release --bin mlxcel --tests -- -D warnings`: clean (only a pre-existing, unrelated C++ build warning in `mlxcel-core`).
- `cargo fmt --check -- src/commands/embed.rs`: clean.
- Independent `pr-reviewer` and `pr-security-checker` passes found no CRITICAL, HIGH, or MEDIUM issues; one LOW suggestion (the test's placeholder model path) was applied.
- Not run: an end-to-end `mlxcel embed` invocation against a real checkpoint, since the change is confined to validation that runs before model loading.

## 5. Related Work

- Issue #1664: source issue for this change.
- `src/commands/rerank.rs`: reference implementation for the `trim().is_empty()` guard and the inline test convention.
