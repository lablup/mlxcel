# Technical Report: PR #1572 - Run Python client CI across a 3.9-3.13 version matrix

**Date**: 2026-09-02
**Status**: Completed
**Languages**: YAML, TOML
**Risk Level**: Low

## Executive Summary

PR #1572 closes a coverage gap in the Python client's CI: `python/pyproject.toml` publishes `requires-python = ">=3.9"` and classifiers through 3.13, but `.github/workflows/python.yml` ran every check on a single pinned interpreter (3.11). Four of the five advertised versions never executed the test suite. The change splits the workflow into a single-leg lint job and a three-leg pytest matrix (3.9, 3.11, 3.13) with `fail-fast: false`, and narrows the pyproject classifiers to the versions CI now actually exercises.

## 1. Problem Statement

### 1.1 Background

`.github/workflows/python.yml` had one `check` job that ran `ruff check`, `ruff format --check`, `mypy python/src`, and `pytest python/tests -m "not e2e"` against `actions/setup-python@v7` pinned to `python-version: '3.11'`. There was no `strategy`/`matrix` block anywhere in the file.

### 1.2 Existing Issues

- **Untested version range**: `requires-python = ">=3.9"` and the classifier block (3.9 through 3.13) advertise five supported versions, but only one was ever exercised by CI. The dependency floors declared in the same file, `openai>=1.40` and `httpx>=0.27`, were resolved against that one interpreter only, so a resolution failure or runtime behavior difference on another version would ship unnoticed.
- **Static checks were already floor-gated, runtime was not**: `ruff` (`target-version = "py39"`) and `mypy` (`python_version = "3.9"`) were already configured against the 3.9 floor, so a 3.9 syntax or typing regression was already caught. Only interpreter-dependent runtime behavior in the actual test suite was uncovered.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| A dependency resolution that fails only on the 3.9 floor ships unnoticed | Medium | Low |
| A stdlib/runtime behavior difference on 3.12/3.13 breaks the client for users on newer interpreters | Medium | Low |
| Classifiers claim support CI never verified | Low | High (was already true) |

## 2. Technical Decisions

### 2.1 Split one job into `lint` and `test` instead of adding `if:` guards to a single matrixed job

**Context**: Only `pytest` needs to run per Python version; `ruff` and `mypy` are already pinned to `py39` semantics and produce the same result on every interpreter, so running them on all three legs would be pure waste.

**Alternatives Considered**:

| Option | Pros | Cons |
|--------|------|------|
| One `check` job, matrixed, with `if: matrix.python-version == '3.11'` guards on the lint/mypy steps | Minimal diff | Conditional steps clutter the job; lint/type-check status becomes tied to one specific matrix leg's reporting, which is easy to misread in the Actions UI |
| **Chosen: two jobs, `lint` (single leg) and `test` (matrixed)** | Each job's purpose and pass/fail state is unambiguous; `test` legs report independently as `test (Python 3.9)`, `test (Python 3.11)`, `test (Python 3.13)` | Slightly more YAML; two `actions/checkout`/`actions/setup-python` blocks instead of one |

**Rationale**: The two-job split matches the issue's own suggestion ("Optionally run `ruff` and `mypy` on one leg only... that keeps the matrix to pytest, which is the part that actually varies") and keeps each job's Actions-UI status legible without conditional steps.

### 2.2 Matrix set: floor, current, ceiling (`3.9`, `3.11`, `3.13`), not all five classifier versions

**Context**: The classifier list names five versions (3.9-3.13). Testing all five would fully close the metadata/coverage gap but roughly doubles the job count for a job whose only cost is `pip install` plus lint and unit tests on `ubuntu-latest` (no MLX, no compiled binary).

**Rationale**: The issue's acceptance criteria explicitly allow either exercising every classifier version or narrowing the classifiers to match what CI verifies, on the condition that the two do not disagree. Three versions (floor, current, ceiling) is standard interpreter-matrix practice: it catches a floor regression and a ceiling regression while treating the versions in between as covered by continuity between the two ends. This PR takes the narrowing branch: `python/pyproject.toml`'s classifier list now reads 3.9, 3.11, 3.13, matching the matrix exactly. `requires-python = ">=3.9"` is left unbounded, since that field is a real installability constraint (pip enforces it), not a coverage claim.

**Trade-offs**: 3.10 and 3.12 no longer appear in the PyPI classifier metadata, even though nothing suggests the client is actually broken on them. If a 3.10- or 3.12-specific regression is ever suspected, it is a case for widening the matrix again, not evidence that this PR under-covered the range.

### 2.3 `fail-fast: false` on the `test` matrix

**Rationale**: Without it, GitHub Actions cancels the remaining matrix legs the moment one fails, which would hide a ceiling-only or floor-only failure behind whichever leg happened to fail first. This was one of the issue's explicit acceptance criteria.

## 3. Implementation Details

### 3.1 Workflow structure

**File: `.github/workflows/python.yml`**

Before: one `check` job running all four steps on `python-version: '3.11'`.

After:

```yaml
jobs:
  lint:
    name: lint, type-check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with:
          python-version: '3.11'
      - run: ruff check python
      - run: ruff format --check python
      - run: mypy python/src

  test:
    name: test (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ['3.9', '3.11', '3.13']
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v7
        with:
          python-version: ${{ matrix.python-version }}
      - run: pytest python/tests -m "not e2e" -q
```

Both jobs install the package the same way the original single job did (`pip install -e "python[dev]"`), so the effective dependency resolution and test invocation are unchanged; only the interpreter varies across the `test` legs.

### 3.2 Classifier narrowing

**File: `python/pyproject.toml`**

```
- "Programming Language :: Python :: 3.9",
- "Programming Language :: Python :: 3.10",
- "Programming Language :: Python :: 3.11",
- "Programming Language :: Python :: 3.12",
- "Programming Language :: Python :: 3.13",
+ "Programming Language :: Python :: 3.9",
+ "Programming Language :: Python :: 3.11",
+ "Programming Language :: Python :: 3.13",
```

No change to `requires-python`, dependencies, or any file under `python/src/`.

## 4. Learning Points

### 4.1 GitHub Actions required-check naming after a job rename

Renaming `check` to `lint`/`test` changes the check names GitHub reports for the workflow (`Python / lint`, `Python / test (Python 3.9)`, etc.). This repository's branch ruleset for `main` was confirmed (via `gh api repos/lablup/mlxcel/rules/branches/main`) to carry only `deletion` and `non_fast_forward` rules, with no `required_status_checks` rule pinned to the old job name, so the rename does not orphan a required check. A repository that does pin required checks by job name would need to update that configuration in the same change.

## 5. Change Summary

### Statistics

| Item | Value |
|------|-------|
| Files changed | 2 |
| Lines added | +26 |
| Lines deleted | -8 |
| Tests added | 0 (existing suite now runs on 3 interpreters instead of 1) |

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| CI | 1 | `.github/workflows/python.yml` split into `lint` (single leg) and `test` (3.9/3.11/3.13 matrix, `fail-fast: false`) |
| Metadata | 1 | `python/pyproject.toml` classifiers narrowed to match the tested matrix |

### Related Commits

| Hash | Type | Message |
|------|------|---------|
| `8c6a94e` | test | test(ci): run Python client CI across a 3.9-3.13 version matrix |

## 6. Follow-up Actions

### Monitoring Required

- Watch the first CI run on this PR for a 3.9-only or 3.13-only failure. Per the source issue, a genuine floor/ceiling incompatibility is a separate bug to file, not a reason to drop that version from the matrix.

### Future Improvements

- If a 3.10- or 3.12-specific regression is ever reported, re-widen the matrix (and the classifier list) rather than treating this PR's three-version choice as a permanent ceiling.

## Appendix

### A. Test Results

Local validation (uv-managed interpreters, prior to pushing to CI):

| Interpreter | `pip install -e "python[dev]"` | `pytest python/tests -m "not e2e" -q` |
|---|---|---|
| 3.9.6 (system CPython) | Resolved cleanly | 43 passed, 2 deselected |
| 3.11.10 (uv-managed) | Resolved cleanly | 43 passed, 2 deselected |
| 3.13.5 (uv-managed) | Resolved cleanly | 43 passed, 2 deselected |

`ruff check`, `ruff format --check`, and `mypy python/src` were additionally run against the 3.11 environment and passed with no findings.
