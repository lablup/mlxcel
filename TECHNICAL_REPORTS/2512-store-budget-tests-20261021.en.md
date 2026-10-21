# PR #2512: Unit tests for the store_budget helpers

**Date**: 2026-10-21
**Status**: Completed (PR open)

## Executive Summary

PR #2512 (closes #1662) adds `src/server/store_budget_tests.rs` with 9 unit tests for the two pure helpers in `src/server/store_budget.rs`. Before this PR, neither helper had a test. Both bound the in-memory Responses and Conversations stores, so a silent change to either one would change memory accounting or eviction order. The PR changes no production code. Review and the security check found nothing to fix.

## 1. What is pinned

| Helper | Contract the tests fix |
|---|---|
| `serialized_json_len_saturating` | Equals `serde_json::to_vec(..).len()` for scalars, escaped and non-ASCII strings, empty and nested containers, and an unsized `&str`. Returns `usize::MAX` when serialization fails: a `Serialize` impl that errors, and a `HashMap` with tuple keys. |
| `LruKey` `Ord` | Orders by `last_accessed`, then `sequence`, then `id`. One test per level isolates that level. A full sort checks eviction order. `partial_cmp` agrees with `cmp` for every pair, and equal keys compare `Equal`. |

## 2. Technical Decisions

- **Deterministic time.** All keys use one base `Instant` plus `Duration` offsets, so the result does not depend on clock resolution.
- **No direct test of the saturating add.** Reaching `usize::MAX` bytes would need a real allocation of that size. The private `SaturatingByteCounter` stays private and is covered only through the public helper.
- **The tuple-key test asserts the precondition.** It first checks that `serde_json::to_vec` really fails for that map, so the test cannot pass for the wrong reason if serde_json later accepts such keys.
- **No CHANGELOG entry.** Earlier test-only PRs (#2481, #2482) added none.

## 3. Verification

| Check | Result |
|---|---|
| `server::store_budget_tests` | 9 passed |
| `cargo clippy -p mlxcel --lib --tests -- -D warnings` | Clean |
| `cargo fmt --check` | Clean after merging main |

All runs used `MLXCEL_BUILD_METAL=0 --profile test-fast --lib` on macOS.

**CI note.** When the PR opened, main failed four checks left by #2505: `cargo-fmt`, `kernel port dispatch`, `env var docs` and `python tooling tests`. #2516 fixed them, and the PR branch merged main before its final CI run.

## 4. Open Items

None for this PR.
