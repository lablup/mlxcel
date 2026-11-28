# Technical Report: PR #1120 - docs: fix stale cross-references left behind by earlier renames

**Date**: 2026-08-14
**Status**: Completed
**Languages**: Markdown, Rust (comment only)
**Risk Level**: Low

---

## Executive Summary

PR #1120 closes issue #1106 by repairing four documentation references that earlier renames left pointing at things that no longer exist or no longer describe the tree: the pull request template's link to a deleted `AGENTS.md`, the same dead pointer in a `tests/surgery_cli.rs` doc comment, two `v0.0.27` version stamps in `docs/`, and a missing index entry in `docs/README.md`.

No code or behavior changes. The one `.rs` file in the diff is a rustdoc comment on a constant; the constant's value is untouched.

---

## 1. Problem Statement

### 1.1 Background

Three independent staleness bugs accumulated in documentation that no CI check validates. Each is one line, each is verifiable on its own, and each misleads a reader in a different way.

### 1.2 Existing Issues

- **Dead `AGENTS.md` pointer.** `.github/PULL_REQUEST_TEMPLATE.md:3` sent every first-time contributor to a file that does not exist in a checkout. `AGENTS.md` is listed in `.gitignore` alongside the other local-only working files, so it is a local file by design and no clone has one. `CHANGELOG.md` records that PR #1014 replaced the dead `AGENTS.md` links in `CONTRIBUTING.md`; that pass missed the template, which is the highest-traffic surface of the three. A repo-wide grep during this work found a fourth site the issue had not listed: the doc comment on `REFERENCE_MODEL` in `tests/surgery_cli.rs`.
- **Version stamps at `v0.0.27`.** `docs/supported-models.md` opened with "model-family support in the v0.0.27 source tree" and `docs/turbo-kv-cache.md` prefixed its allowlist with "As of v0.0.27". The workspace is at `0.5.0-beta.1`. `supported-models.md` is the page both `README.md` and `CONTRIBUTING.md` send readers to for the support matrix, so the stamp advertises the page as many releases out of date.
- **Incomplete index.** `docs/README.md` enumerated 18 documents while `docs/` holds 19 non-README `.md` files. The unindexed one was `code-guidelines.md`, one of the three core contract documents `CONTRIBUTING.md` names; the other two, `architecture.md` and `adding-models.md`, were both already listed.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|---|---|---|
| Contributor cannot find the contributor contract from the PR template | Medium | High |
| Reader discards a current support matrix as obsolete because of its stamp | Medium | Medium |
| Contributor never finds `code-guidelines.md` and ships a JIT kernel without dtype in its cache key | Medium | Low |

---

## 2. Technical Review

### 2.1 Correctness

The `v0.0.27` fix was checked to be a stamp problem and not a content problem before the stamps were removed. The three allowlisted model-type prefixes `docs/turbo-kv-cache.md` lists (`qwen3_5`, `qwen3_5_moe`, `qwen3_next`) were compared against `ALLOWED_SYMMETRIC_TURBO_FAMILIES` in `src/lib/mlxcel-core/src/cache/turbo/allowlist.rs` and match exactly, so nothing behind either stamp needed rewriting.

The index fix was checked by arithmetic rather than by eye: the sorted basenames of `ls docs/*.md` minus `README.md` are now byte-identical to the sorted filenames extracted from the numbered list, 19 entries on each side.

### 2.2 Scope Containment

Issue #1106's acceptance criterion 4 is "docs only; no code or behavior change." Criterion 1 is "no reference to `AGENTS.md` remains outside `CHANGELOG.md`." The `tests/surgery_cli.rs` occurrence sits between the two: satisfying criterion 1 requires touching a `.rs` file. The edit is confined to rustdoc comment prose on a `const`, so criterion 4 holds in substance. `rustfmt` leaves comments alone by default (`wrap_comments` is off), so `cargo fmt --all -- --check` is unaffected, and the file is gated behind `#![cfg(feature = "surgery")]` in any case.

The two `AGENTS.md` mentions that remain are both deliberate: the three `CHANGELOG.md` entries are historical records the issue explicitly preserves, and the `.gitignore` line is an ignore pattern, not a cross-reference.

---

## 3. Technical Decisions

### 3.1 Drop the Version Stamps Instead of Bumping Them

| Option | Pros | Cons |
|---|---|---|
| Bump both to `0.5.0-beta.1` | Accurate today | Nothing verifies it, so it goes stale at the next release |
| **Chosen: remove the stamp, strengthen the code pointer** | Cannot go stale; readers are sent to the authority | Loses the "as of" signal for anyone who wanted a snapshot date |

`docs/supported-models.md` already carried a "the runtime source of truth is the code" block naming four source paths, so removing its stamp left the page pointing somewhere durable with no further edit. `docs/turbo-kv-cache.md` named `allowlist.rs` but did not say it was authoritative, so the removed stamp is replaced by that claim: "which is the source of truth for this list."

### 3.2 Point the PR Template at Two Documents

`AGENTS.md` was refactored into focused reference docs (`CHANGELOG.md` records the 313 to 75 line split). The contract it used to hold now lives in `CONTRIBUTING.md` and `docs/code-guidelines.md`, and `CONTRIBUTING.md` links the latter. The template's own checklist already cites `docs/code-guidelines.md` by that exact path, so naming both keeps the header consistent with the body.

---

## 4. Change Summary

### Statistics

| Item | Value |
|---|---|
| Files changed | 5 |
| Lines added | +9 |
| Lines deleted | -7 |
| Tests added | 0 |

### Changes by Area

| Area | Files | Summary |
|---|---|---|
| Contributor onboarding | `.github/PULL_REQUEST_TEMPLATE.md` | Dead `AGENTS.md` pointer repointed at `CONTRIBUTING.md` and `docs/code-guidelines.md` |
| Documentation accuracy | `docs/supported-models.md`, `docs/turbo-kv-cache.md` | `v0.0.27` stamps removed; allowlist section gains an explicit source-of-truth pointer |
| Documentation index | `docs/README.md` | `code-guidelines.md` added as item 19 |
| Comment hygiene | `tests/surgery_cli.rs` | Dead `AGENTS.md` pointer removed from the `REFERENCE_MODEL` doc comment |

### Related Commits

| Hash | Type | Message |
|---|---|---|
| `e251af08` | docs | docs: fix stale cross-references left behind by earlier renames |

---

## 5. Validation and Follow-up

### Passed

- `python3 scripts/ci/check_cross_repo_refs.py` (no bare 3+-digit `#NNN` added).
- `cargo fmt --all -- --check`.
- Repo-wide `grep -rn "AGENTS\.md"`: only the three `CHANGELOG.md` history entries and the `.gitignore` pattern remain.
- `grep -rn "v0\.0\.27" docs/`: no matches.
- Sorted diff of `ls docs/*.md` basenames against the `docs/README.md` numbered list: identical, 19 entries.
- CI on PR #1120: crate versions, kernel dtype keys, cross-repo refs, cargo-deny, cargo-fmt all green.

### Related Work

- Issue #1110 is amending the body of `docs/code-guidelines.md` concurrently. This PR only adds an index entry naming that file and does not touch its contents, so the two do not overlap.
- Issue #1111 amends the "Expected future layout examples" block a few lines below the new item 19 in `docs/README.md`, and is sequenced after this PR for that reason.

### Not Addressed

The stamps are gone, but nothing prevents a future writer from adding a new one. There is no CI check that a document's claims about the code still hold; the mitigation here is structural, naming the code as the authority so the prose has less to go stale about.
