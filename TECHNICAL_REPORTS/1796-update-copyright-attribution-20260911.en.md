# Technical Report: PR #1796 - Update copyright attribution

**Date**: 2026-09-11
**Status**: Completed
**Languages**: Rust, Python, C/C++, Metal, shell, text
**Risk Level**: Low

## Executive Summary

This PR normalizes project-owned source headers around Lablup Inc. and changes the NOTICE attribution to Lablup Inc. and contributors. Authorship provenance remains in package metadata instead of being repeated as individual copyright ownership across the source tree.

## 1. Problem Statement

PR #1728 completed repository-wide Apache header coverage, making one shared attribution line consistent across the owned source surface. That line placed the company and an individual side by side as copyright holders in 1,812 files, conflating project copyright ownership with authorship provenance.

Changing only existing headers would allow the old wording to return through the header insertion script. Removing every authorship reference would also discard useful provenance. The required boundary is therefore company ownership in source headers, collective credit in NOTICE, and scoped authorship in package metadata.

## 3. Technical Decision

The change separates three metadata roles:

| Surface | Recorded information |
|---------|----------------------|
| Project-owned source headers | `Copyright 2025-2026 Lablup Inc.` |
| `NOTICE` | `Copyright 2025-2026 Lablup Inc. and contributors.` |
| Package metadata | Authorship provenance, including the `mlxcel-core` crate |

The header generator uses the same company-only line, preventing newly covered files from reintroducing the previous wording. Existing third-party provenance detection remains unchanged.

## 7. Change Summary

| Item | Value |
|------|-------|
| Files changed | 1,813 |
| Lines added | 1,814 |
| Lines deleted | 1,813 |
| Runtime behavior changes | None |
| New dependencies | None |

The diff is mechanical except for the explicit `mlxcel-core` package metadata entry. All previous joint copyright lines were removed; the remaining individual attribution is confined to package authorship metadata and Git history.

## Appendix: Verification

- `python3 tests/test_insert_apache_header.py -v`: 11 tests passed
- `python3 scripts/insert_apache_header.py --check`: every target file carries a license header
- `cargo metadata --no-deps --format-version 1`: `mlxcel-core` metadata parses with the retained authorship
- `git diff --check`: passed
