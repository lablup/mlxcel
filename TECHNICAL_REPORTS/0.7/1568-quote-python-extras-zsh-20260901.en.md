# Technical Report: PR #1568 - Quote Python extras installs under zsh

**Date**: 2026-09-01
**Status**: Completed
**Languages**: Markdown
**Risk Level**: Low

## Executive Summary

PR #1568 fixes a copy-paste failure in the Python client documentation by quoting the `./python[dev]` extras argument everywhere it is shown to users. The change is small, but it removes a shell-specific onboarding break on macOS, where zsh treats unquoted brackets as a glob pattern.

## 1. Problem Statement

The repository documents Python development setup in two places: `python/README.md` and `docs/python-client.md`. Three commands showed `pip install ./python[dev]` or `pip install -e ./python[dev]` without quotes. In zsh, which is the default shell on macOS and therefore the common path for this project's Apple Silicon users, `[dev]` is parsed as a glob character class rather than as part of the package extras syntax.

That makes the documented command fail before `pip` even runs. The problem is not in the package metadata or the Python client implementation; it is purely in the prose examples users are expected to copy verbatim.

## 2. Technical Decisions

### 2.1 Fix the commands by quoting the extras argument, not by rewriting the examples

The change keeps the documented install shape exactly the same and adds only the shell quoting that the command already needs in zsh. This matches the repository's existing CI invocation, which already uses `pip install -e "python[dev]"`, and avoids introducing alternative forms or longer explanatory text for a one-token shell parsing issue.

### 2.2 Update every duplicated example in one PR

The same failure mode appeared in both the user-facing Python client guide and the package-local README, including the editable install form in the test section. Fixing all three lines together prevents one document from remaining stale and reintroducing the same onboarding confusion.

## 3. Change Summary

| Area | Change |
|---|---|
| `python/README.md` | Quotes the development install command in the install section and the editable development install command in the test section. |
| `docs/python-client.md` | Quotes the matching development install command in the Python client guide. |

### Statistics

| Item | Value |
|------|-------|
| Files changed | 2 |
| Lines added | +3 |
| Lines deleted | -3 |
| Tests added | 0 |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `5fdc3eb` | docs | docs: quote Python extras installs |

## 4. Validation

Validation was intentionally narrow because this PR changes documentation only.

- `rg -n 'pip install (\./python\[dev\]|-e \./python\[dev\])' python/README.md docs/python-client.md` returns no matches, confirming the unquoted broken forms were removed.
- `rg -n 'pip install ("\./python\[dev\]"|-e "\./python\[dev\]")' python/README.md docs/python-client.md` finds all three intended commands, confirming the docs now present the quoted forms consistently.

## 5. Follow-up Actions

- [ ] When other docs add Python extras examples, keep the extras argument quoted so shell-specific regressions do not reappear.
- [ ] If the install guidance is consolidated later, prefer a single source that the package README and the main docs can share or mirror mechanically.

## 6. Related Work

- Issue #1222: documents the zsh globbing failure and points at the three affected lines.
- PR #1568: applies the fix and closes the issue.
