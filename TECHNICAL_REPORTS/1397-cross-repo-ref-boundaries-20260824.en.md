# Technical Report: PR #1397 - Derive Cross-Repository Reference Boundaries Live

**Date**: 2026-08-24

**Author**: Jeongkyu Shin

**Status**: Completed

**Languages**: Python, Bash, YAML, Markdown

**Risk Level**: Medium

## Executive Summary

PR #1397 removes the expired rule that treated every bare issue or pull-request number at or above 1000 as an upstream reference. The advisory checker now derives the current `lablup/mlxcel` number boundary from GitHub when a token is safely available and falls back to explicit manual review when it is offline or unauthenticated. Deterministic companion tests cover both modes and run in the real pull-request workflow.

## 1. Problem Statement

The repository grew beyond the classifier's fixed numeric assumption. References such as #1023, #1340, #1355, and #1385 are valid same-repository links, but the old `num >= 1000` branch labeled them likely upstream. On the measured PR #1386 diff this produced seven false positives; on the historical PR #1385 merge diff it produced 28 numeric-only false positives. Because the check is advisory, persistent noise eroded the review signal intended to catch genuinely unqualified upstream or private-repository references.

## 2. Change Summary

| Area | Change |
|---|---|
| Classifier | Query the newest issue/PR number through a five-second `gh api` call and remove the fixed threshold |
| Fallback | Keep non-upstream bare refs visible in the manual-review bucket and print why live classification was unavailable |
| Companion test | Exercise same-repo, upstream, qualified, above-boundary, offline, API-failure, and strict-mode cases in temporary repositories |
| CI | Run the companion suite before the classifier and provide `github.token` only to same-repository PRs |
| Contributor guide | Document live-boundary, offline, fork, and manual-review behavior |

## 3. Technical Decisions

### 3.1 Derive rather than retune the boundary

Issue and pull-request numbers share GitHub's repository sequence, and the issues endpoint includes pull requests. Querying the newest item with explicit created-descending ordering therefore supplies one moving boundary without maintaining another number that will expire as the repository grows.

### 3.2 Fail open without failing silently

No token, a missing `gh` executable, timeout, API failure, or invalid response does not fail this advisory check. The classifier instead prints the fallback reason and places every non-upstream bare reference in the existing manual-review bucket. Explicit upstream-name signals still enter the likely-upstream bucket in either mode.

### 3.3 Keep base-repository credentials away from fork code

The workflow executes pull-request-controlled scripts. A security review therefore restricted `github.token` to same-repository PRs; fork PRs receive an empty value and intentionally exercise the documented offline fallback. This retains public-fork coverage without exposing the base repository token to changed code.

### 3.4 Test evaluated behavior in isolated repositories

The shell companion creates temporary Git repositories and a fake `gh` binary, which makes boundary and failure behavior deterministic without network access. Its deliberate reference corpus is excluded through the existing `IGNORE_PREFIXES` mechanism so the production classifier does not report its own fixtures.

## 4. Verification

- `python3 -m py_compile scripts/ci/check_cross_repo_refs.py`: passed.
- `bash -n scripts/ci/check_cross_repo_refs_test.sh`: passed.
- `bash scripts/ci/check_cross_repo_refs_test.sh`: all seven cases passed, including strict mode's expected non-zero result.
- The companion suite also passed with a parent `GITHUB_TOKEN` present, proving that its no-token case removes inherited credentials explicitly.
- Synthetic PR #1386 replay: #1340 and #1355 remained outside `Likely UPSTREAM`.
- Historical `fb1e909cc~1...fb1e909cc` replay: the 28 numeric-only same-repository false positives disappeared; only three lines carrying explicit upstream context remained upstream-signaled, matching the issue's declared boundary.
- Live endpoint check returned current PR number 1397 as the repository boundary.
- Hosted `cross-repo refs` ran the companion suite and production classifier successfully after the inherited-token fixture correction.
- Formatting, Clippy, cargo-deny, MLX-pin extraction, OpenXLA compile, repository metadata checks, and CLA passed. The final OpenXLA link job was still pending when this report was authored and must reach a terminal state before merge.
- Correctness, security/performance, and finalization reviews found no unresolved issue after the fork-token restriction.

## 5. Incident Found by Hosted Validation

The first hosted run found that the companion suite's offline case inherited the workflow-level `GITHUB_TOKEN`, so it did not enter the fallback path even though local shells without a token passed. The fixture now removes both `GH_TOKEN` and `GITHUB_TOKEN` inside that case, and its environment options precede assignments for hosted GNU `env` compatibility. The replacement hosted job passed.

## 6. Related Work

- Issue #1387: stale numeric classification of same-repository references.
- PR #1397: implementation and review corrections documented here.
- PR #1386 and merge commit `fb1e909cc`: measured false-positive datasets.
- `scripts/ci/check_crate_versions.py`: related precedent for making classification policy explicit and reviewable.
