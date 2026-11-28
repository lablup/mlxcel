# Technical Report: PR #1181 - docs: surface file-size and module-split thresholds in code-guidelines

**Date**: 2026-08-16
**Author**: AI Code Reviewer
**Status**: Completed
**Languages**: Markdown
**Risk Level**: Low

---

## Executive Summary

The project's file-size and module-split thresholds lived only in `.claude/skills/mlxcel-code-structure/SKILL.md`. That path is gitignored in this repository, so a contributor cloning `lablup/mlxcel` had no access to the guidance at all, and nothing under `docs/` or in `CONTRIBUTING.md` pointed at it. This PR gives the thresholds a contributor-facing home in `docs/code-guidelines.md` and updates the three places that describe that document.

The symptom that motivated the issue is a good illustration of the cost: PR #1173 justified a file split, in both its PR body and its commit body, with a "500-line limit" that does not exist in this project. The real guideline is 800 lines. The commit body is permanent history.

Getting the content right took four commits. Three of the four line counts carried over from the skill file were stale, one example was a 31-line re-export shim described as a 600-line model, and the framing named a single "standing exception" when nine files exceed 2,000 lines. Each of those would have put a fresh wrong number into the very document meant to end wrong numbers.

---

## 1. Problem Statement

### 1.1 Background

Issue #1178 was filed after the #1159 / #1160 / #1161 chain, during which a review noticed the "500-line limit" claim in PR #1173. The claim was traced, and the underlying cause turned out not to be carelessness but inaccessibility.

### 1.2 Existing Issues

- **Issue 1**: `.gitignore` line 28 ignores `/.claude/`, and `git ls-files .claude/` returns zero files. The thresholds shipped with no tracked copy in this repository.
- **Issue 2**: `docs/code-guidelines.md` had no file-size guidance whatsoever. Its only headings were "Shared Function Comments" and "JIT Kernel Cache Keys".
- **Issue 3**: `CONTRIBUTING.md` described that document narrowly as "the shared-function rules" in all three of its references, and `docs/README.md` entry 19 did the same, so even a contributor who opened it would not expect structural guidance there.
- **Issue 4** (found during implementation): the numbers in the skill file had gone stale against the tree, so relocating them without checking would have propagated the staleness into tracked documentation.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| A contributor invents a threshold the project never set, as PR #1173 did | Low | Medium while the guidance is unreachable |
| Stale figures get frozen into tracked docs and then cited in reviews | Medium | High if copied without verification |
| Numbers drift between the doc and any other copy | Low | Medium |

---

## 2. Technical Review

### 2.1 Security

No security or performance surface exists to review. The diff is three markdown files and zero lines of code: no executable path, no dependency, no build input, no runtime behavior. The security and performance assessment was therefore made inline rather than delegated, because there is nothing for a code-oriented pass to analyze.

The one risk vector a documentation change genuinely carries is inaccuracy, since wrong guidance produces wrong structural decisions and, as PR #1173 showed, wrong numbers cited in review. That vector was covered by the review pass and by independent verification of every quoted figure. See section 3.2.

**Issues Found:**
| Issue | Severity | Status |
|-------|----------|--------|
| Section claimed the skill file "links here instead of restating them", which is unverifiable here and false of the tracked internal copy | Medium | Fixed (`59c06872`) |
| Three of four quoted line counts stale; one example was a re-export shim, not a model | Medium | Fixed (`6b1c092d`) |
| Framing named one "standing exception" when nine files exceed 2,000 lines and seven have no helpers file | High | Fixed (`5a5ed9ce`) |
| `llama4` overclaim: "the bulk that could be lifted out" describes an 84-line helpers file against 1,862 lines | Medium | Fixed (`5a5ed9ce`) |
| `docs/README.md` entry 19 still described the document narrowly, leaving part of the discoverability gap open | Medium | Fixed (`5a5ed9ce`) |
| PR body still quoted the stale figures and the removed skill-file claim, and becomes the squash commit message | Medium | Fixed (review pass) |

### 2.2 Performance

None. Documentation only.

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: none
- **New Dependencies**: none
- **Compatibility**: no code, build, or runtime input changed

### 2.4 Code Quality

- **Test Coverage**: not applicable; no code
- **Technical Debt**: decreased for discoverability, and the new text explicitly labels the oversized files as debt rather than precedent

---

## 3. Technical Decisions

### 3.1 Move the numbers into `docs/`, rather than leave them and add a pointer

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Keep detail in the skill file, add a pointer from `docs/` | Smaller diff | The authoritative copy stays in a path this repository does not ship, so a contributor still cannot read it |
| **Chosen: thresholds live in `docs/code-guidelines.md`; other citations link there** | The guidance ships with the repository and has one authoritative home | Requires curating what a human contributor needs out of an agent-facing file |

**Rationale:** The thresholds are project convention for humans, not agent configuration, so the agent-facing file should defer to the document rather than the reverse. Curation was deliberate: the table, the exceptions with their reasons, the inline-test tiering and the naming pattern came across; the directory-vs-file trigger, trait placement, the model-registration checklist and the rest of the agent-workflow material stayed put.

### 3.2 Verify every figure rather than transcribe it

**Rationale:** This is the substance of the work rather than a formality. Checking the tree found:

| Claim as carried over | Actual |
|---|---|
| `qwen2.rs`, a complete model in about 600 lines | 31 lines, a re-export shim over `llama3` |
| `nemotron_h.rs` 2,342 lines | 2,914 |
| `llama4.rs` 1,499 lines | 1,862 |
| `heartbeat.rs` about 300 lines | 311, close enough to keep |

`qwen2.rs` was replaced with `dbrx.rs` (606 lines, a genuine complete model with `Attention`, `SparseMoeBlock`, `DecoderLayer`, `DbrxModel` and an `impl LanguageModel`, no re-export). The document now also tells the reader that figures are approximate and to check the file rather than trust a quoted number, this section included, because these numbers went stale once already in the source they came from.

### 3.3 Name the real outliers instead of implying a single carve-out

**Rationale:** The first correction fixed the numbers but left the framing, which called `nemotron_h.rs` the standing exception. Nine files under `src/models/` exceed 2,000 lines and seven have no helpers file, including `gemma4.rs` at about 6,700 lines with none and `qwen3_5.rs` at about 3,500 with none. Naming one exception implied a tidy carve-out and left the actual outliers invisible, which is how "the largest file in the tree" becomes an argument in a review. The document now names them, labels them debt rather than precedent, and says plainly that the largest file in the tree is not a number you can cite.

The `llama4` entry was recast at the same time. It has a helpers file, so it is not a file that ignored the guidance; it shows the 1,200+ row applied, with the separable mask construction and weight loading moved out while the coupled model stayed together.

### 3.4 State the single-source rule prescriptively

**Rationale:** The section originally asserted that `.claude/skills/mlxcel-code-structure/SKILL.md` links here instead of restating the numbers. That is not verifiable from this repository, where the path is gitignored, and it is false of the copy tracked in the internal development repository, which still carries the table. Asserting a present-tense fact about a file this repository does not ship would have been the same class of defect as the false doc claim fixed in #1160. The wording is now a rule ("anything else that cites them should link here rather than restate them") which is true today and stays true regardless of what any consumer currently contains.

---

## 4. Implementation Details

### 4.1 Key Changes

- `docs/code-guidelines.md`: new "File Size and Module Structure" section with the threshold table, the exceptions and the debt list, the inline-test tiering, and the naming patterns.
- `CONTRIBUTING.md`: the intro paragraph and the quick-links row now describe the document as covering shared-function conventions plus the file-size and module-split thresholds, and a new bullet under code standards links to the anchor.
- `docs/README.md`: entry 19 widened to name the thresholds first.

---

## 7. Change Summary

### Statistics
| Item | Value |
|------|-------|
| Files changed | 3 |
| Lines added | +38 |
| Lines deleted | -3 |
| Tests added | 0 (documentation only) |

### Related Commits
| Hash | Type | Message |
|------|------|---------|
| `d9739496` | docs | surface file-size and module-split thresholds in code-guidelines |
| `59c06872` | docs | state the single-source rule as a requirement, not as a claim about an untracked file |
| `6b1c092d` | docs | correct the stale line counts carried over with the thresholds |
| `5a5ed9ce` | docs | correct the exception framing and widen the docs index entry |

---

## 8. Follow-up Actions

### Required
- [ ] None for this repository; all four acceptance criteria are met

### Open across repositories
- The threshold table still exists in `lablup/mlxcel-internal`, which tracks `.claude/skills/mlxcel-code-structure/SKILL.md` (17 tracked files under `.claude/`). Within `lablup/mlxcel` the numbers now live in exactly one tracked place, but across the two repositories they exist twice, which is the drift this issue set out to prevent. Editing that copy to link at `docs/code-guidelines.md` is a change to a different repository and was deliberately left out of scope. The equivalent edit was applied to this machine's untracked working copy and saved to the session scratchpad, but it is not durable: the internal repository is the source of truth and will overwrite it.

### Future Improvements
- The 1,500+ row says "consider a directory module" without the source's "3+ files or 2+ distinct concerns" trigger, so it is less actionable than the other rows.
- The oversized files named as debt (`gemma4.rs`, `gemma3n.rs`, `qwen3_5.rs`) have no recorded justification. Either recording one or extracting helpers would close the gap between the documented rule and the tree.

---

## Appendix

### A. Verification

No test suite applies. Verification was factual rather than executable, and every figure in the new section was checked against the tree at `9d78a1b5`:

| Claim | Verified |
|---|---|
| `src/models/dbrx.rs` about 600 lines, a genuine model | 606 lines, 15 struct/impl definitions, zero `pub use super::` |
| `src/distributed/heartbeat.rs` about 310 lines | 311 |
| `src/models/nemotron_h.rs` about 2,900, no helpers file | 2,914; `nemotron_h_helpers.rs` does not exist |
| `src/models/llama4.rs` about 1,860, has a helpers file | 1,862; `llama4_helpers.rs` exists |
| Only other `*_helpers.rs` under `src/models/` | `gemma3n_helpers.rs`, `qwen3_next_helpers.rs`; three in total |
| `gemma4.rs` about 6,700 with no helpers file | 6,680; none |
| `qwen3_5.rs` about 3,500 with no helpers file | 3,451; none |
| `gemma3n.rs` about 5,600 | 5,594 |
| Nine files over 2,000 lines under `src/models/` | 9, excluding `_tests.rs` |
| `.claude/` untracked here | `.gitignore:28` is `/.claude/`; `git ls-files .claude/` returns 0 |
| Anchor `#file-size-and-module-structure` resolves | matches `## File Size and Module Structure`, no duplicate heading |

The diff was confirmed to touch only markdown: three files, all `.md`, none outside `docs/` or `CONTRIBUTING.md`.

### C. References
- Issue #1178, PR #1173 and commit `0adb53c5` (the "500-line limit" symptom), the #1159 / #1160 / #1161 chain that surfaced it
- `docs/code-guidelines.md` (the new authoritative section), `CONTRIBUTING.md`, `docs/README.md`
- `.claude/skills/mlxcel-code-structure/SKILL.md` (the origin of the guidance, untracked here, tracked in `lablup/mlxcel-internal`)
- PR #1181 review comments
