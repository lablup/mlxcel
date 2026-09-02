# Technical Report: PR #1571 - docs: correct stale "not a server knob" claim in turbo-kv-cache.md

**Date**: 2026-09-02
**Author**: mlxcel maintainers
**Reviewer**: (pending)
**Status**: Completed
**Languages**: Markdown (documentation only)
**Risk Level**: Low

---

## Executive Summary

`docs/turbo-kv-cache.md` described `MLXCEL_PAGED_ATTENTION_NATIVE` as a library/bench-only control that is "not a server knob." That description went stale when issue #899 gave the variable a second, server-side consumer and made its force-off values the production v2 decode kill switch. This PR rewrites the paragraph to match the corrected framing PR #1119 already applied to `docs/environment-variables.md`, so all four pages that describe this variable now agree.

---

## 1. Problem Statement

### 1.1 Background

`MLXCEL_PAGED_ATTENTION_NATIVE` originally had one consumer: the library-only `paged_decode_attention_pooled` entry point, reached only from `mlxcel-core` callers and the kernel bench. Issue #710 retired that entry point off the `mlxcel serve` decode path, which is where the "not a server knob" framing in `docs/turbo-kv-cache.md` came from. Issue #899 later added `resolve_paged_v2_dispatch` in `src/lib/mlxcel-core/src/layers.rs` as a second consumer: the server's pool-backed batched paged decode now reads the same variable, and its force-off values (`0`/`false`/`off`/`no`) became the kill switch back to the pre-#899 gather-then-SDPA path.

### 1.2 Existing Issues

- **Issue 1**: `docs/turbo-kv-cache.md` still asserted, as current fact, that the variable is "a control for external mlxcel-core consumers and the kernel bench, not a server knob," which has been false since #899 shipped.
- **Issue 2**: PR #1119 (issue #1104) already corrected the same stale claim in `docs/environment-variables.md`, but its acceptance criteria named only `README.md` and `docs/CONTINUOUS_BATCHING.md`, leaving `docs/turbo-kv-cache.md` as a follow-up that PR's own body flagged.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| An operator reads `docs/turbo-kv-cache.md` in isolation and concludes the variable is safe to ignore in production, missing that it is also the v2 kill switch | Medium | Medium |
| Docs pages disagree with each other on what a shipped environment variable does | Low | High (already true before this fix) |

---

## 2. Technical Review

### 2.1 Security

Not applicable; documentation-only change, no code or configuration parsing touched.

### 2.2 Performance

Not applicable; no code path changed.

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: None.
- **New Dependencies**: None.
- **Compatibility**: N/A.

### 2.4 Code Quality

- **Test Coverage**: N/A (docs only).
- **Code Complexity**: N/A.
- **Technical Debt**: Reduced: one fewer page contradicting the variable's actual current behavior.

---

## 3. Technical Decisions

### 3.1 Reuse the corrected framing from PR #1119 instead of paraphrasing independently

**Context:**
`docs/environment-variables.md` already carries a corrected description of `MLXCEL_PAGED_ATTENTION_NATIVE` after PR #1119 (commit `cf4e22cd`). `docs/turbo-kv-cache.md` needed the same correction in prose form, in a paragraph that also covers the fused split-K kernel's dispatch regime.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Option A: Paraphrase the corrected behavior independently | Tailored wording for this page's context | Risks re-introducing a subtly different (and possibly re-stale) description; duplicates the token-floor and log-line details already documented elsewhere |
| Option B: Delete the "not a server knob" claim without adding the current behavior | Minimal diff | Leaves the page silent on the variable's dual consumers, ADR 0001 link context is orphaned |
| **Chosen: Option C: reuse `cf4e22cd`'s framing, cross-link instead of duplicating** | Keeps this page and `docs/environment-variables.md` in agreement by construction; token floors and dispatch policy stay defined in one place each | Requires the reader to follow two links for full detail on the server-side floors |

**Rationale:**
The issue explicitly asked to read `git show cf4e22cd -- docs/environment-variables.md` and reuse that corrected framing, then cross-link `docs/environment-variables.md#paged-decode-v2-variables` and `docs/CONTINUOUS_BATCHING.md#seeing-which-path-ran` rather than duplicating the dispatch policy and log-line prose. This keeps the four pages that describe the variable (`README.md`, `docs/CONTINUOUS_BATCHING.md`, `docs/environment-variables.md`, `docs/turbo-kv-cache.md`) from drifting apart again the next time one of them is updated.

**Trade-offs:**
The rewritten paragraph is now denser (names both consumers, the kill-switch semantics, and two cross-links) and slightly longer than the original. This was accepted because the issue's acceptance criteria required naming both consumers explicitly rather than a one-line fix.

### 3.2 Keep the two intentionally historical references untouched

**Context:**
Two other pages contain the string "not a server knob" for a legitimate reason: `docs/adr/0001-paged-attention-gather-vs-fused-kernel.md`'s `### Decision` section records what was true at decision time, and `docs/environment-variables.md`'s `History:` clause quotes the old reading as a past belief, not a current claim.

**Rationale:**
An ADR is a historical record by construction and must not be retroactively edited to match present-day behavior; doing so would erase the reasoning trail that led to later changes. The `History:` clause in `docs/environment-variables.md` already uses the corrected pattern (quote the old reading, then state what changed and why) that this PR now mirrors in `docs/turbo-kv-cache.md`.

---

## 4. Implementation Details

### 4.1 Key Code Changes

**File: `docs/turbo-kv-cache.md`**

Before (lines 394-399):
```
slab. #710 retired this pooled entry point to a library-only API: neither this
kernel nor its selector is on the `mlxcel serve` decode path (which stays on the
block-table kernel described above), and `MLXCEL_PAGED_ATTENTION_NATIVE` is a
control for external mlxcel-core consumers and the kernel bench, not a server
knob. See ADR 0001's #710 decision record,
[ADR 0001](adr/0001-paged-attention-gather-vs-fused-kernel.md).
```

After:
```
slab. #710 retired the pooled entry point and its selector to a library-only
API, off the `mlxcel serve` decode path, which stays on the block-table kernel
described above. The variable itself keeps two consumers today, per
`resolve_dispatch_decision` and `resolve_paged_v2_dispatch` in
`src/lib/mlxcel-core/src/layers.rs`: that library-only pooled entry point, and
the server's pool-backed batched paged decode. On the server side, issue #899
made the fused v2 kernel the production decode path and named this variable's
force-off values its kill switch; a force-on value pins v2 for every servable
shape, bypassing the measured token floors. See
[Paged decode v2 variables](environment-variables.md#paged-decode-v2-variables)
for the floors and defaults, and
[Continuous batching](CONTINUOUS_BATCHING.md#seeing-which-path-ran) for the
dispatch policy and per-outcome log lines. History: #710's retirement of the
library entry point is where the "not a server knob" reading came from; #899
gave the variable its second, server-side consumer. See ADR 0001's #710 decision
record, [ADR 0001](adr/0001-paged-attention-gather-vs-fused-kernel.md).
```

**Reason for change:** Names both current consumers of the variable (matching `resolve_dispatch_decision` and `resolve_paged_v2_dispatch`), states the server-side kill-switch and force-on behavior concretely, demotes the #710 retirement to history rather than a present-tense claim, and points to the two pages that already own the token-floor and dispatch-log detail instead of duplicating it.

The paragraph is hard-wrapped at roughly 80 columns to match the surrounding prose in this file, consistent with the file's existing style in that section (other sections of the same file use unwrapped single-line paragraphs, so wrapping is a local convention, not a file-wide rule).

---

## 5. Learning Points

### 5.1 Cross-page documentation drift after a partial correction

**Concept:**
When an environment variable's behavior changes (here, gaining a second consumer via #899), every page that describes it needs the same correction. A prior fix (#1104/PR #1119) scoped its acceptance criteria to specific pages and explicitly flagged the remaining page as a follow-up rather than silently leaving it inconsistent.

**Application in this PR:**
This PR closes that flagged follow-up. The verification step (`grep -rzoP 'not a server\s+knob' docs README.md` or the Python equivalent used here, since the change spans a hard line wrap that a plain single-line grep would miss) confirms the string now appears only inside the two legitimately historical contexts.

**Common Use Cases:**
- Any variable, flag, or API whose scope grows over time needs a documentation audit across every page that references it, not just the page where the scope grew.
- A hard-wrapped multi-line source file requires a wrap-tolerant search (`grep -z`, a multi-line regex, or reading the rendered text) to reliably find a phrase that may straddle a line break.

---

## 6. Further Learning

### Key Terms

| Keyword | Description | Relevance |
|---------|-------------|-----------|
| `MLXCEL_PAGED_ATTENTION_NATIVE` | Environment variable that force-pins or defers dispatch between the fused paged-attention kernel and the gather-then-SDPA reference path | Subject of the corrected documentation |
| `resolve_dispatch_decision` | Rust function in `src/lib/mlxcel-core/src/layers.rs` implementing the library-only pooled entry point's dispatch | One of the variable's two consumers |
| `resolve_paged_v2_dispatch` | Rust function in `src/lib/mlxcel-core/src/layers.rs` implementing the server's pool-backed batched paged decode dispatch | The other consumer, added by issue #899 |

### Related PRs/Issues

- PR #1119 (issue #1104): Corrected the same stale claim in `docs/environment-variables.md` and flagged `docs/turbo-kv-cache.md` as a follow-up.
- Issue #899: Made the fused v2 kernel the production server decode path and defined this variable's kill-switch semantics.
- Issue #710: Retired the library-only pooled entry point off the `mlxcel serve` decode path; the origin of the now-corrected "not a server knob" reading.
- Issue #1139: This PR's source issue.

---

## 7. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 1 |
| Lines added | +16 |
| Lines deleted | -6 |
| Tests added | 0 (docs only) |

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| Documentation | 1 | Rewrote one paragraph in `docs/turbo-kv-cache.md` to describe both current consumers of `MLXCEL_PAGED_ATTENTION_NATIVE` and demote the #710 retirement to history |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `d2e263d` | docs | docs: correct stale "not a server knob" claim in turbo-kv-cache.md |

---

## 8. Follow-up Actions

### Required

- None. The four pages that describe this variable (`README.md`, `docs/CONTINUOUS_BATCHING.md`, `docs/environment-variables.md`, `docs/turbo-kv-cache.md`) now agree.

### Monitoring Required

- None; documentation-only change with no runtime effect.

### Future Improvements

- None identified.

---

## Appendix

### A. Test Results

- `grep`-equivalent search (Python, wrap-tolerant) over `docs/` and `README.md` for `not a server\s+knob`: three hits, all historical (`docs/environment-variables.md` `History:` clause, `docs/adr/0001-paged-attention-gather-vs-fused-kernel.md` `### Decision` section, and the new `History:` sentence in `docs/turbo-kv-cache.md`). No current-tense assertion remains.
- `python3 scripts/ci/check_cross_repo_refs.py`: passed, no bare 3+ digit `#NNN` references added.
- `git diff --stat`: confirms `docs/turbo-kv-cache.md` is the only file touched.

### B. Performance Benchmarks

Not applicable.

### C. References

- Issue #1139 (this PR's source issue)
- PR #1119 / commit `cf4e22cd` (the corrected framing this PR reuses)
- Issue #899, #710
