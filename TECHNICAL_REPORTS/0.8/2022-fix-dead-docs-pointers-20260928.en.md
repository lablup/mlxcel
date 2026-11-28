# Technical Report: PR #2022, Fix dead docs/model_tests.md and docs/testing.md pointers

**Date**: 2026-09-28
**Status**: Completed
**Languages**: Rust, TOML
**Risk Level**: Low

## Executive Summary

Six source locations (a `Cargo.toml` comment, two doc comments in `src/bin/speculative_bench.rs`, a compile-time assertion message in `tests/speculative_parity.rs`, a doc comment in `tests/prompt_cache_prefill_bench.rs`, and a comment in `src/tokenizer/mod.rs`) pointed readers at `docs/model_tests.md` or `docs/testing.md`, neither of which has ever existed in this repository. Five references were repointed to `docs/benchmark_results/model_tests.md`, the file that actually carries the described content; the sixth was dropped in favor of an inline instruction, since its likely intended target lives outside this repository entirely. A new regression test, `tests/dead_doc_pointers.rs`, guards against the same two strings reappearing in the five fixed files.

## 1. Problem Statement

### 1.1 Background

Issue #1658 is the same class of defect as issue #26 (a dead `docs/model_implementations.md` pointer, guarded by `src/main_tests.rs`'s `supported_models_output_has_no_dead_doc_link`). Unlike #26, the dead paths here live in source comments and one assertion message rather than in a CLI renderer's output, so the existing guard cannot see them.

### 1.2 Existing Issues

- **Issue 1**: `docs/model_tests.md` was never a real path. `git log --all --diff-filter=A -- docs/model_tests.md` returns nothing; the file that was always intended is `docs/benchmark_results/model_tests.md`, which carries the exact `## Speculative drafters` (line 317) and `## Prompt cache benchmarks` (line 240) sections the five references describe.
- **Issue 2**: `docs/testing.md` was also never real. Its introducing commit (`290eb645`) also touched `docs/en/user-guide/server.md`, and `mkdocs.yml`'s nav lists `development/testing.md`; per `docs/README.md`'s "The MkDocs manual" section, `docs/en/` and `docs/ko/` are sources for a separately maintained manual (published at `mlxcel.lablup.ai`) and are explicitly not part of this repository. Repointing at that path would still be a dead reference here.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| A contributor follows the pointer while porting a new speculative pairing or writing a prompt-cache benchmark and finds nothing, or gives up on locating real-model test setup instructions | Low | Was certain before this fix; now resolved |

## 3. Technical Decisions

### 3.1 Repoint vs. drop, decided per reference

**Context**: The issue asked for a per-reference judgment call: repoint to the file's real current location, or drop the pointer if the target was never real.

**Alternatives considered for `docs/testing.md`**:

| Option | Pros | Cons |
|--------|------|------|
| Repoint to `docs/en/development/testing.md` | Matches the apparent original intent (a TurboQuant testing section) | That path is outside this repository (separate manual tree); would still be a dead link here |
| Repoint to `CLAUDE.md`'s "Testing with real models" section | Topically close (downloading models to run against) | `CLAUDE.md` is `.gitignore`d in this repo and absent from `git ls-files`; not part of the published tree |
| **Chosen: drop the pointer, inline the one useful fact** | Comment stays self-sufficient and correct; no risk of a future dead link | Loses a "see also" for readers who want the broader external manual |

**Rationale**: Both alternative targets fail the same test the issue is about, pointing at something outside this repository's tree. The comment only needed one fact from the missing doc (how to get the model), so `mlxcel download mlx-community/gemma-4-e4b-it-8bit` (verified against `docs/model-catalog.tsv:63`) makes it self-sufficient without a dead pointer.

### 3.2 A narrow test file instead of extending `src/main_tests.rs`

**Context**: The issue suggested, as optional, extending the existing #26 guard test to also cover these two paths.

**Rationale**: `supported_models_output_has_no_dead_doc_link` asserts against `write_supported_models()`'s rendered output; the six dead pointers fixed here live in raw source text with no equivalent renderer to check. `src/main_tests.rs` is also already 1535 lines, well past the project's 500-line file-size guideline, so growing it further for an unrelated check was avoided. `tests/dead_doc_pointers.rs` instead `include_str!`s the five fixed source files and asserts the two dead strings do not reappear in them.

**Trade-off**: This guard is deliberately narrow (five named files, two named strings), not a tree-wide "every `docs/*.md` reference must exist" scanner. A repo-wide grep during implementation found other, pre-existing dead references outside this issue's scope (for example `docs/USAGE.md`, `docs/bridge-overhead-microbench.md`); a generic scanner would have had to fail on those too, which is separate cleanup.

## 4. Implementation Details

### 4.1 Key Code Change

**File: `src/tokenizer/mod.rs`**
```rust
// Before
// is missing so the test suite stays portable; run on demand with
// `cargo test -- --ignored` against a workspace that has the model
// downloaded (per `docs/testing.md`).

// After
// is missing so the test suite stays portable; run on demand with
// `cargo test -- --ignored` against a workspace that has the model
// downloaded (`mlxcel download mlx-community/gemma-4-e4b-it-8bit`).
```

The other five sites are the same repoint pattern: `docs/model_tests.md` becomes `docs/benchmark_results/model_tests.md`, including inside `tests/speculative_parity.rs`'s compile-time `assert!` message, which remains a plain string literal so it still reads correctly if it ever fires.

## 7. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 6 |
| Lines added | +75 |
| Lines deleted | -8 |
| Tests added | 2 (`tests/dead_doc_pointers.rs`) |

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| Documentation | 5 references | Repointed `docs/model_tests.md` to `docs/benchmark_results/model_tests.md` |
| Documentation | 1 reference | Dropped the `docs/testing.md` pointer, inlined the download command |
| Testing | 1 new file | `tests/dead_doc_pointers.rs`: regression guard against both strings reappearing |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `004b8b9` | docs | Fix dead docs/model_tests.md and docs/testing.md pointers |
| `a21d09e` | docs | Fix comment counting and stale line ref in dead_doc_pointers.rs |

### Related PRs/Issues

- Issue #26: prior instance of the same defect class, source of the `src/main_tests.rs` guard pattern this PR could not directly reuse.
- Issue #1667 / PR #2016: a sibling PR editing `encode_prompt` in `src/bin/speculative_bench.rs`; this PR was scoped to touch only that file's doc-comment lines to avoid a merge conflict.

## 8. Follow-up Actions

### Required

- [ ] None.

### Future Improvements

- The tree carries other pre-existing dead `docs/*.md` references outside this issue's scope (for example `docs/USAGE.md`, `docs/bridge-overhead-microbench.md`, `docs/metal4-fused-attention-research.md`, three files under `docs/papers/`); a follow-up issue could apply the same per-reference triage to those.
