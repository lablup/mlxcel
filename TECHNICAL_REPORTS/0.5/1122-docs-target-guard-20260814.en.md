# Technical Report: PR #1122 - chore(docs): guard the docs-* targets and document the manual split

**Date**: 2026-08-14
**Status**: Completed
**Languages**: Make, Markdown
**Risk Level**: Low

---

## Executive Summary

PR #1122 closes issue #1111. The thirteen `docs-*` Makefile targets build the MkDocs manual from sources that are maintained in a separate documentation tree and are not part of this repository, so every one of them failed partway through with an opaque error. They now share a `docs-guard` prerequisite that stops immediately with an explanation, and `docs/README.md` documents the split instead of leaving it to be inferred from a failed build.

The guard is a presence check rather than an unconditional refusal, which is the design decision that carries the most weight here: one Makefile stays correct both in this repository and in the tree that does hold the sources.

---

## 1. Problem Statement

### 1.1 Background

Issue #1111 asked a maintainer to choose between two readings of the same evidence: **(a)** the manual sources belong in this repository and have not landed yet, or **(b)** the manual is built from a different tree. The fix differs entirely, so the issue explicitly asked for the decision before any diff.

The answer is **(b)**. `git log --all -- docs/en` is empty, so this is not a deletion regression: the tree has never existed here. The four mkdocs configs at the repository root are copies belonging to the tree that owns the manual, kept in sync with it.

### 1.2 Existing Issues

- **Every `docs-*` target failed, and none said why.** `make docs-install` died inside `uv pip install -r docs/requirements.txt`. Working around that hit `ln -s ../shared docs/en/shared` against a nonexistent parent, then a site build against a nonexistent `docs_dir`. A reader had to run a target and interpret the failure to learn that the manual is built elsewhere.
- **`make help` advertised all thirteen as working.** `make help` is the discovery surface for the build system, and the targets appear in four of its sections, since the greps that build those sections match on `build`, `serve`, `doc`, `clean`, and `install`.
- **The `nav:` blocks and the on-disk tree disagreed silently.** The configs list 33 page paths each, none of which exist here, while the GitHub-facing `docs/*.md` files appear in no `nav:`. Nothing recorded that this is intended rather than drift.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|---|---|---|
| Contributor burns time debugging a build that cannot work in this tree | Medium | High |
| Someone "fixes" the dangling navs by pointing them at `docs/*.md`, breaking the tree that owns them | High | Low |
| A reader concludes the documentation build is broken rather than absent | Low | Medium |

---

## 2. Technical Review

### 2.1 Correctness of the Guard

The guard tests for `docs/en` and, when it is missing, prints the explanation and exits 1. It was verified in both directions: all thirteen targets were run individually and each stopped at the guard with exit 1 before reaching any `uv`, `ln`, or `zensical` command, and the guard was then re-run with a `docs/en` directory present, where it exits 0 and unblocks the targets.

`docs-guard` carries no `##` string, so it does not appear in `make help`. Because it is `.PHONY` with no prerequisites, make runs it at most once per invocation, so `docs-pdf`, which depends on the guard and on `docs-pdf-en` and `docs-pdf-ko` (each of which also depends on the guard), evaluates it once.

### 2.2 Compatibility

No Rust, no CI target, and no GitHub-facing `docs/*.md` document is touched. Acceptance criterion 4 of the issue holds: the only file changed under `docs/` is the index.

`webpage-build` also consumes `mkdocs.yml` and `mkdocs.ko.yml` and therefore shares the underlying dependency. It is outside the enumerated scope of #1111 and is deliberately left alone rather than silently changed.

---

## 3. Technical Decisions

### 3.1 Presence Check, Not Unconditional Refusal

| Option | Pros | Cons |
|---|---|---|
| Unconditional `exit 1` in each target | Simplest to read here | Wrong in the tree that holds the sources; the next sync either breaks that tree's build or reverts this change |
| Delete the thirteen targets and the configs | Removes the dead surface entirely | The configs belong to the tree that owns the manual and would come back on the next sync; deleting them locally guarantees churn |
| **Chosen: shared `docs-guard` presence check** | Correct in both trees at once; explains rather than fails; targets unblock automatically wherever the sources exist | Slightly more Makefile than a hardcoded refusal |

The decisive property is that the same Makefile is right in both places. A guard that says "the sources are missing" is a true statement wherever it fires and silent wherever it does not, so nothing about it needs to be un-done during a sync.

### 3.2 Shared Prerequisite Over Thirteen Copies

Thirteen copies of the same check would drift and would put the same nine-line message in the diff thirteen times. One `.PHONY: docs-guard` with thirteen one-word prerequisite additions keeps the message in a single place and makes the dependency visible on each target line.

### 3.3 Leave the Four mkdocs Configs Unchanged

The issue's option (b) contemplated removing the configs or documenting them as no-ops. Neither is right: they are the property of the tree that owns the manual and this repository is kept in sync with it, so a local fork of their `nav:` blocks would be both wrong there and undone here. The explanation goes in `docs/README.md`, where it costs nothing to keep and does not conflict with a sync.

### 3.4 Reconcile "Expected future layout examples"

`docs/README.md` listed `docs/en/...` and `docs/ko/...` under "Expected future layout examples", which is what made option (a) look plausible in the first place. Those two entries describe a tree that already exists elsewhere, not a layout this repository is heading toward, so they move into the new section. The other two entries there (`docs/github/...`, `docs/git/...`) are genuinely still future and stay.

---

## 4. Change Summary

### Statistics

| Item | Value |
|---|---|
| Files changed | 2 |
| Lines added | +63 |
| Lines deleted | -15 |
| Targets guarded | 13 |

### Changes by Area

| Area | File | Summary |
|---|---|---|
| Build system | `Makefile` | `docs-guard` presence check added; all thirteen `docs-*` targets depend on it; each `##` help string gains a caveat suffix |
| Documentation | `docs/README.md` | New "The MkDocs manual" section documenting the split; `docs/en/...` and `docs/ko/...` moved out of "Expected future layout examples"; intro item 2 reconciled |

### Related Commits

| Hash | Type | Message |
|---|---|---|
| `54a3b294` | chore | chore(docs): guard the docs-* targets and document the manual split |

---

## 5. Validation and Follow-up

### Passed

- All thirteen targets run individually: `docs-install`, `docs-serve`, `docs-serve-en`, `docs-serve-ko`, `docs-build`, `docs-build-ko`, `docs-build-all`, `docs-build-strict`, `docs-pdf-setup`, `docs-pdf-en`, `docs-pdf-ko`, `docs-pdf`, `docs-clean`. Each exits 1 at the guard with the explanatory message and reaches no build command.
- Positive path: `make docs-guard` exits 0 with a `docs/en` directory present.
- `make help` shows all thirteen with the caveat in each of the four sections its greps place them in; `docs-guard` is absent from help.
- `python3 scripts/ci/check_cross_repo_refs.py` passes.

### Follow-up Candidates

- `webpage-build` builds from the same two mkdocs configs and has the same unmet dependency. It is not a `docs-*` target and was out of scope for #1111; guarding it the same way would be a small, self-contained follow-up.
- The `(manual sources not in this checkout)` help suffix is a statement about this repository specifically. Unlike the guard, it is not automatically true elsewhere. Anyone syncing the Makefile into the tree that holds the sources should treat the suffix as local.
