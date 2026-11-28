# Technical Report: PR #1570 - chore(build): guard webpage-build against missing MkDocs sources

**Date**: 2026-09-02
**Author**: mlxcel maintainers
**Reviewer**: automated pr-reviewer and pr-security-checker passes
**Status**: Completed
**Languages**: Makefile (GNU Make, POSIX shell)
**Risk Level**: Low

---

## Executive Summary

`make webpage-build` ran the same unmet MkDocs manual dependency that thirteen `docs-*` targets already guard, but predated the `docs-guard` prerequisite and stayed unguarded through an earlier pass (#1111, PR #1122). On a checkout without the manual sources it deleted `webpage/site/public/en/manual` and `webpage/site/public/ko/manual`, then failed with an opaque zensical error. This PR adds `docs-guard` as a prerequisite of `webpage-build`, matching the existing pattern exactly, and makes a deliberate, documented choice for the neighboring `webpage-deploy` target rather than leaving that question open.

---

## 1. Problem Statement

### 1.1 Background

`mkdocs.yml` sets `docs_dir: docs/en` and `mkdocs.ko.yml` sets `docs_dir: docs/ko`. Those directories, along with `docs/shared`, `docs/requirements.txt` and `docs/scripts`, are maintained in a separate documentation tree and are absent from this repository's checkout. Thirteen `docs-*` targets already declare `docs-guard` as a prerequisite for exactly this reason, so each of them fails immediately with an explanation instead of an opaque `uv`, `ln`, or zensical error. `webpage-build` runs the same zensical build against the same missing `docs_dir`, but because it is not itself a `docs-*` target, it was out of scope for the pass that added the guard and was recorded as a follow-up.

### 1.2 Existing Issues

- **Destructive-then-opaque failure order**: `webpage-build`'s recipe opened with `rm -rf webpage/site/public/en/manual webpage/site/public/ko/manual`, so on a manual-less checkout the previously built manual output (if any) was deleted before the build failed. The failure itself then surfaced from inside zensical against a `docs_dir` the reader could not see in this checkout, rather than from a message pointing at the actual cause.
- **`webpage-deploy` left unexamined**: `webpage-deploy` shells out to `scripts/deploy_webpage.sh`, which builds `webpage/site` with pnpm and force-pushes the static export to the `gh-pages` branch of a separate `mlxcel-releases` remote. The script does not invoke zensical or read `docs_dir`, so it cannot fail the way `webpage-build` does, but it also does not verify that the manual output it is about to publish actually exists. That gap was not evaluated as part of the original guard rollout.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Contributor runs `make webpage-build` on a fresh checkout and loses time diagnosing a zensical `docs_dir` error | Low (time cost only, no data loss) | Medium (any contributor without the private docs tree) |
| `make webpage-deploy` publishes a site missing or serving stale `/en/manual`, `/ko/manual` pages | Low (recoverable by rebuilding and redeploying; no CI path invokes either target today) | Low (deploy is a manual, deliberate maintainer action) |

---

## 3. Technical Decisions

### 3.1 Guard `webpage-build`, but not `webpage-deploy`, behind `docs-guard`

**Context:**

The issue asked for two separate calls: add `docs-guard` to `webpage-build` (a clear application of the existing pattern), and separately decide, deliberately, what `webpage-deploy` should do, since its failure mode is structurally different.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| A: Guard `webpage-deploy` with `docs-guard` too | Consistent with the other guarded targets at a glance | Tests the wrong precondition: `docs-guard` checks `docs/en` (a build *input*), while `webpage-deploy` actually depends on `webpage/site/public/{en,ko}/manual` (a build *output*) already existing. A maintainer redeploying an unchanged site built earlier, or one who copied the manual output in from elsewhere, would be blocked for no reason. |
| B: Leave `webpage-deploy` untouched with no comment | No risk of behavior change | Leaves the question the issue explicitly raised unanswered, and gives a future reader no record that the omission was considered rather than missed |
| **Chosen: C. Add a non-blocking presence check for the actual manual output, with a comment recording the reasoning** | Tests the real precondition, does not block a legitimate re-deploy of unchanged output, and documents the decision in the Makefile itself | Does not prevent a first-time deploy on a fresh checkout from publishing a site with missing manual pages; it only warns |

**Rationale:**

`docs-guard`'s job is to explain why a `docs-*` target (or `webpage-build`, structurally the same shape) cannot run at all in this checkout. `webpage-deploy` is different: the script that backs it always succeeds regardless of whether the manual was built, so a hard failure would have to be invented rather than reported, and the natural place to invent it, `docs/en` absence, is not actually what determines whether the deploy is correct. Checking `webpage/site/public/en/manual` and `webpage/site/public/ko/manual` directly measures the thing that matters, so the warning is accurate under every scenario: a fresh checkout that never ran `webpage-build`, a checkout that ran it successfully, and a checkout where the manual arrived by some other means (a copied build, for instance).

**Trade-offs:**

The warning does not block a first deploy on a fresh checkout, so a maintainer who ignores it can still publish a site with 404ing manual pages. The alternative (a hard failure) was rejected because it would misfire on a legitimate re-deploy, which is judged the more common real-world case for a manually triggered release script with no CI integration.

---

## 4. Implementation Details

### 4.2 Key Code Changes

**File: `Makefile`**
```makefile
# Before
.PHONY: webpage-build
webpage-build: ## Build download webpage (static export)
	@echo "$(CYAN)Building documentation for webpage...$(RESET)"
	rm -rf webpage/site/public/en/manual webpage/site/public/ko/manual
	uv run zensical build -f mkdocs.yml -d webpage/site/public/en/manual
	...

.PHONY: webpage-deploy
webpage-deploy: ## Deploy download webpage to GitHub Pages
	@echo "$(CYAN)Deploying webpage...$(RESET)"
	./scripts/deploy_webpage.sh

# After
.PHONY: webpage-build
webpage-build: docs-guard ## Build download webpage (static export) (manual sources not in this checkout)
	@echo "$(CYAN)Building documentation for webpage...$(RESET)"
	rm -rf webpage/site/public/en/manual webpage/site/public/ko/manual
	uv run zensical build -f mkdocs.yml -d webpage/site/public/en/manual
	...

# webpage-deploy is intentionally not gated behind docs-guard. deploy_webpage.sh
# never invokes zensical or reads docs_dir, so it cannot fail the way
# webpage-build does, and docs-guard checks for docs/en (a build input) rather
# than for the manual output deploy actually reads. ...
.PHONY: webpage-deploy
webpage-deploy: ## Deploy download webpage to GitHub Pages
	@echo "$(CYAN)Deploying webpage...$(RESET)"
	@if [ ! -d webpage/site/public/en/manual ] || [ ! -d webpage/site/public/ko/manual ]; then \
		echo "$(YELLOW)Warning: webpage/site/public/en/manual or .../ko/manual is missing.$(RESET)"; \
		echo "  Run 'make webpage-build' first, or the deployed site will be missing (or serving a stale copy of) the manual pages."; \
	fi
	./scripts/deploy_webpage.sh
```

**Reason for change:** Declaring `docs-guard` as a prerequisite is the identical mechanism used by the thirteen existing `docs-*` targets; Make resolves the prerequisite regardless of where in the file `docs-guard` is defined, so no reordering of the Makefile's sections was required, matching the note already in the issue's Technical Considerations. The `webpage-deploy` warning uses the same `$(YELLOW)`/`$(RESET)` color variables already defined near the top of the Makefile and always exits 0, so it never changes whether `./scripts/deploy_webpage.sh` runs.

---

## 7. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 1 (`Makefile`) |
| Lines added | +14 |
| Lines deleted | -1 |
| Tests added | 0 (Makefile-only change; validated by direct `make` invocations, see Appendix A) |

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| Build tooling | 2 | `webpage-build` gained a `docs-guard` prerequisite and an updated help string; `webpage-deploy` gained a non-blocking precondition warning and an explanatory comment |
| Documentation | 1 | `make help` now states the `webpage-build` dependency instead of leaving it to be discovered by running the target |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `2375ee0` | chore | chore(build): guard webpage-build against missing MkDocs sources |

### Related Issues

- Issue #1138: chore(build): guard `webpage-build` against the missing MkDocs manual sources, closed by this PR.
- Issue #1111 / PR #1122: added `docs-guard` to the thirteen `docs-*` targets; this PR extends the same guard to the one target that pass did not cover.

---

## 8. Follow-up Actions

### Required

None. All five acceptance criteria in issue #1138 are met and verified.

### Monitoring Required

None. No CI workflow invokes `webpage-build` or `webpage-deploy` (confirmed by grepping every `.github/workflows/*.yml`, `*.sh`, and `*.md` in the repository for either target name), so the new hard failure on `webpage-build` cannot break automation, and the warning on `webpage-deploy` is purely advisory.

### Future Improvements

- The `webpage-deploy` warning is a presence check, not a freshness check, so a months-old `webpage/site/public/en/manual` directory passes silently even though its content may be stale relative to the current `docs/en` tree. A future change could compare a build timestamp or checksum if stale-but-present manuals become a recurring problem in practice.
- `make help` now lists `webpage-build` under two additional filtered categories ("Help & Documentation" and "Test Targets") purely because its help text now contains the substrings `doc` and `check`, an artifact of the same category-filtering behavior every one of the thirteen pre-existing guarded `docs-*` targets already exhibits. Cosmetic only, consistent with established behavior, and not addressed in this PR.

---

## Appendix

### A. Test Results

All verification was performed with direct `make` invocations on a checkout without `docs/en`, since the change is a Makefile guard rather than application code:

- `make webpage-build` on the current checkout stops at `docs-guard` (exit 2) before the `rm -rf` runs; `ls webpage/site/public/` was checked before and after and shows the same contents (`brands/` only), confirming the manual directories were never touched.
- `make help | grep webpage-build` shows the new `(manual sources not in this checkout)` suffix.
- `make DOCS_MANUAL_DIR=<existing-directory> webpage-build` (a variable override used only to exercise the guard's presence-check path without a real MkDocs tree) proceeds past `docs-guard` into the unchanged `rm -rf` and `uv run zensical` steps, failing only on the unrelated absence of `zensical` in this environment. This confirms the guard is a presence check, not an unconditional refusal.
- `make -n webpage-deploy` confirms the new warning block is syntactically valid Make and shell; the embedded shell condition was also exercised standalone against the manual-less checkout and correctly prints both warning lines.
- An independent `pr-reviewer` pass and an independent `pr-security-checker` pass both returned zero CRITICAL, HIGH, or MEDIUM findings, and confirmed the guard cannot be silently bypassed under `-j4` or `-k`, that the warning always exits 0 and therefore never interferes with the deploy script's own `set -e`, and that this PR strictly reduces the destructive surface of `webpage-build` relative to the pre-PR behavior.

### C. References

- Issue #1138 (this PR's origin).
- Issue #1111 and PR #1122 (the original `docs-guard` rollout across the thirteen `docs-*` targets).
- `scripts/deploy_webpage.sh` (the script `webpage-deploy` invokes).
