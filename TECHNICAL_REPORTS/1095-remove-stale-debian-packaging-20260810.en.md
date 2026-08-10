# Technical Report: PR #1095 - Remove Stale Debian Packaging

**Date**: 2026-08-10
**Status**: Completed
**Languages**: Debian packaging, shell, Markdown
**Risk Level**: Low

## Executive Summary

PR #1095 removes mlxcel's dormant Debian and Launchpad PPA packaging surface. The repository had continued maintaining release changelog and build metadata for a package that had never been uploaded, was not wired into release CI, and could not resolve its required Rust toolchain on any current Ubuntu target.

## 1. Problem Statement

The tracked `debian/` tree described an automated PPA release path that did not exist. Its changelog was refreshed during releases even though the target PPA had never published mlxcel, and its active build dependencies requested unversioned `rustc >= 1.85` while supported Ubuntu archives either provided older unversioned compilers or only newer version-suffixed packages. Launchpad's network-isolated builders also made the documented rustup fallback unusable.

Keeping this tree created recurring release maintenance cost and misleading distribution claims without producing a usable artifact.

## 2. Technical Decisions

### 2.1 Remove instead of reviving the packaging path

Issue #1068 offered removal or revival as mutually exclusive paths. The removal path was selected because Linux release binaries are already produced, while revival would require a separate distribution commitment, an established MSRV, offline vendoring proof, a real release upload workflow, and a successfully published PPA build. None of those prerequisites existed.

### 2.2 Delete the complete surface atomically

All control files, rules variants, helper scripts, packaging documentation, and the 1,206-line generated Debian changelog were removed together. The remaining historical `CHANGELOG.md` wording was changed from a repository-path reference to a generic release-notes description so maintained documentation no longer points to a deleted artifact.

## 3. Change Summary

| Item | Value |
|------|-------|
| Files changed | 18 |
| Lines added | 1 |
| Lines deleted | 1,765 |
| Packaging paths removed | 17 |

- Removed the complete tracked `debian/` directory.
- Removed the changelog generator and PPA version/query helpers, eliminating the obsolete release-maintenance path.
- Removed stale documentation that claimed Launchpad and GitHub Actions integration.
- Preserved the existing binary release workflows without behavior changes.

## 4. Review Findings

The implementation, security/performance, and finalization reviews found no Critical, High, Medium, or actionable Low issues. A release helper installed outside the repository still contains a file-existence-guarded `debian/changelog` branch; after this deletion that branch is inert and cannot regenerate the removed path.

## 5. Validation

- `test ! -d debian`: passed.
- `git ls-tree -r --name-only HEAD | rg '^debian/'`: no tracked packaging paths.
- Repository scans for `debian/changelog`, `README.packaging`, Launchpad, PPA, and `dput`: no current references.
- Workflow, documentation, scripts, and local repository-scaffolding scans found no surviving release step or PPA claim.
- `git diff --check origin/main...HEAD`: passed.
- Hosted `Detect changes`, crate-version, kernel-dtype-key, cross-repository-reference, and CLA checks passed. Rust-heavy jobs were skipped by change detection because no Rust or dependency path changed.

## 6. Related Work

- Issue #1068: decision and acceptance criteria for retiring or reviving Debian packaging.
- Issue #1066: Rust toolchain pin update that further exposed the packaging metadata mismatch.
