# Technical Report: PR #2013 - refactor(server): hoist INITIAL_HASH_CAPACITY_LIMIT into store_budget

**Date**: 2026-09-28
**Author**: Jeongkyu Shin
**Status**: Completed
**Languages**: Rust
**Risk Level**: Low (no behavior change; the compiler verifies both stores against the same constant)

---

## Executive Summary

PR #2013 closes issue #1665. `INITIAL_HASH_CAPACITY_LIMIT`, a `usize` constant capping the initial `HashMap` allocation of the two bounded server stores, was declared identically in both `src/server/conversation_store.rs` and `src/server/responses_store.rs`. It now has one definition in `src/server/store_budget.rs`, the shared helper module both stores already import from, and both call sites import it instead of redeclaring it.

---

## 1. Problem Statement

### 1.1 Background

`ConversationStore` and `ResponsesStore` are bounded in-memory stores that pre-size their backing `HashMap` with `HashMap::with_capacity(max_entries.min(INITIAL_HASH_CAPACITY_LIMIT))`, so a large configured `max_entries` cannot force an oversized up-front allocation before any entries actually arrive. `src/server/store_budget.rs` already exists for exactly this kind of shared concern; its module docstring reads "Shared helpers for bounded in-memory server stores," and both stores already import `LruKey` and `serialized_json_len_saturating` from it.

### 1.2 Existing Issues

- **Duplicated constant.** `const INITIAL_HASH_CAPACITY_LIMIT: usize = 4096;` was declared separately at `conversation_store.rs:36` and `responses_store.rs:50`, with no single source of truth between them.
- **No name-driven usage record.** Neither declaration documented that the value was shared in spirit, only in fact; a reader of either file had no way to know the other file carried the identical constant.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| One copy is retuned for its store's workload and the other is not, leaving the two bounded stores with silently different capacity-cap behavior | Medium | Low |
| A future third bounded store copies the constant a third time instead of importing it | Low | Medium |

---

## 2. Technical Decisions

### 2.1 `store_budget.rs`, Not a New Module

**Context:** The constant needed exactly one home, shared by both stores.

**Rationale:** `store_budget.rs` already holds the other two items both stores import (`LruKey`, `serialized_json_len_saturating`), and its docstring already scopes it to "bounded in-memory server stores." Adding the constant there extends an existing import, rather than introducing a fourth file or a new shared module for one `usize`.

**Trade-offs:** None of substance; this is the module the constant already conceptually belonged to.

### 2.2 `pub(crate)` Visibility and a "Used by" Doc Comment

The constant is declared `pub(crate)`, matching the crate-internal visibility of its two call sites, and carries a doc comment naming both consumers by path (`conversation_store::StoreState::with_capacity`, `responses_store::StoreState::with_capacity`). This follows the project's shared-function convention in `docs/code-guidelines.md` (a `// Used by: ...` style discovery mechanism), applied here to a shared constant rather than a shared function: the next person changing the value has one place to look for who reads it.

---

## 3. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 3 |
| Lines added | +14 |
| Lines deleted | -6 |
| Behavior changes | 0 |

### Changes by File

| File | Change |
|------|--------|
| `src/server/store_budget.rs` | Adds `pub(crate) const INITIAL_HASH_CAPACITY_LIMIT: usize = 4096;` with a doc comment naming both call sites. |
| `src/server/conversation_store.rs` | Removes its local `INITIAL_HASH_CAPACITY_LIMIT` declaration; imports the constant from `store_budget` alongside the existing `LruKey`, `serialized_json_len_saturating` import. |
| `src/server/responses_store.rs` | Same change as `conversation_store.rs`. |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `aa6faf5` | refactor | refactor(server): hoist INITIAL_HASH_CAPACITY_LIMIT into store_budget |

---

## 4. Validation

Per the PR description, the author ran:

- `cargo check --lib --tests` (2-core scoped)
- `cargo clippy --lib --tests -- -D warnings` (2-core scoped)
- `cargo test --lib server::conversation_store` (13 passed)
- `cargo test --lib server::responses_store` (16 passed)
- `cargo fmt --check`
- `python3 scripts/ci/check_cross_repo_refs.py`

This report does not independently re-run those checks; it documents a docs-only change to the PR (adding this report) and the commands above are the PR's own recorded evidence.

### Not Covered

- No new test was added, and none was needed: the value and its use at both call sites are unchanged, only its declaration site moved. The existing 13 + 16 passing tests in the two stores are the coverage for this change.

---

## 5. Related Work

- Issue #1665: source issue for this change ("`INITIAL_HASH_CAPACITY_LIMIT` is declared verbatim in both bounded server stores").
- `docs/code-guidelines.md`: shared-function/shared-item "Used by" comment convention this PR follows.
