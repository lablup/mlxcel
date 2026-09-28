# Technical Report: PR #2011, harden deploy_webpage.sh with set -euo pipefail

**Date**: 2026-09-28

**Status**: Implemented and syntax-checked; pending merge. The script was not executed, since a real run pushes to the `mlxcel-releases` remote.

**Languages**: Bash

**Risk level**: Low

## Executive summary

`scripts/deploy_webpage.sh` ran under bare `set -e`, the only script left in `scripts/` without the stricter `set -euo pipefail` that `run_quality_gate.sh`, `bench_all_models.sh`, and several other maintained scripts already use. The PR changes line 6 to `set -euo pipefail`, closing issue #1660. The change is a one-line diff; the work is in confirming it is safe under the new strictness rather than in the edit itself.

## Problem statement

Bare `set -e` does not catch two classes of mistake that `set -euo pipefail` does: an expansion of an unset variable (`-u`), and a failure inside a pipeline other than its last stage (`pipefail`). Since the deploy script force-pushes its build output to a separate GitHub Pages repository (`git push -f`), a silently-swallowed failure earlier in the script could publish a stale or partial site with no error surfaced. Issue #1660 asked for the script to match the rest of `scripts/` for this reason.

## Change summary

- `scripts/deploy_webpage.sh:6`: `set -e` to `set -euo pipefail`. No other line changed.

Read-through of the 88-line script for the two new strictness modes:

- **`-u` (nounset):** the script expands five variables: `SCRIPT_DIR`, `PROJECT_ROOT`, `WEBPAGE_DIR` (lines 8-10) and `REPO_URL`, `BRANCH` (lines 13-14). All five are assigned unconditionally before their first use at line 16 onward, so none can be unset at expansion time. The heredoc that writes the redirect `index.html` (lines 32-70) uses a quoted delimiter (`<< 'EOF'`), so its `$`-looking content (the inline JavaScript `lang` variable) is literal text, not a shell expansion. `-u` therefore has no observable effect on this script as written; it is a guard against a future edit that adds an unassigned or conditionally-assigned variable.
- **`pipefail`:** the script contains no `|` pipeline anywhere. `pipefail` is consequently inert today too. Its value, like `-u`'s, is prospective: it matches the convention already in `scripts/run_quality_gate.sh:3` and the other scripts listed in the PR body, and it prevents a later edit that pipes a build or git command into `grep`/`tee`/etc. from silently discarding that command's exit status.

Two things observed while reading through, out of scope for this issue and unaffected by the strictness change:

- Lines 20 and 25 `cd` directly (`cd "$WEBPAGE_DIR"`, `cd out`) rather than in a subshell, so the working directory change persists for the rest of the script run. Issue #1660 flagged this explicitly as a known, separate property.
- `$BRANCH` and `$REPO_URL` are unquoted at their two use sites (`git branch -m $BRANCH` line 75, `git push -f $REPO_URL $BRANCH` line 86). Safe today because both are constant literal assignments with no whitespace or glob characters, but `-u` does not protect against word-splitting the way quoting would.

## Validation

- `bash -n scripts/deploy_webpage.sh`: passes.
- Manual read-through of every variable expansion in the script, summarized above: no expansion relies on an unset variable being treated as empty, so `-u` does not break the script.
- Not run end-to-end: the script performs a `git push -f` to `git@github.com:lablup/mlxcel-releases.git`, so executing it as part of validation would publish to that repository. This was intentionally not exercised.

## Remaining work

None required for this issue. The two observations above (persistent `cd`, unquoted `$BRANCH`/`$REPO_URL`) are pre-existing script properties outside issue #1660's scope; a future hardening pass could address them together if the script gets touched again.
