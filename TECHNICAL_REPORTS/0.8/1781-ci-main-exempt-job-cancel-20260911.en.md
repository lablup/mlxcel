# Exempt `main` from the two job-level `cancel-in-progress` flags

## Context

#1774 (PR #1780) added a workflow-level concurrency group to `.github/workflows/ci.yml` so superseded runs stop stacking on GB10, the single self-hosted runner that also serves the release build. That group deliberately exempts `main`: `cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}`, on the reasoning that a push-to-main run catches what a stale PR base hid and is the last gate before a release builds from that commit.

Two job-level groups predate that change and kept an unconditional flag.

## Defect

`xla-link` and `cuda-sm70-compile` each carried:

```yaml
concurrency:
  group: <key>-${{ github.event.pull_request.number || github.ref }}
  cancel-in-progress: true
```

On a push to `main` there is no `github.event.pull_request.number`, so the key falls back to `github.ref`, that is `refs/heads/main`, and every merge shares one group. With the flag unconditional, a second merge cancels the previous merge's job while it is still running.

This is not an oversight in the per-PR design, which is correct: each job's comment states the group exists so that PRs never cancel each other, and keying on the PR number achieves exactly that. What was unconsidered is the `main` fallback, where the same expression produces a different and unwanted outcome.

The two jobs are the worst place for it. Both carry `timeout-minutes: 120`, the maximum in the file, shared with `xla-compile`, and both run on GB10. Cancelling a 120-minute link or compile on the branch a release builds from is the specific outcome the workflow-level exemption exists to prevent.

## Change

Both job-level flags now use the workflow-level condition:

```yaml
cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}
```

PR isolation is unchanged, because on a pull request the expression evaluates to true exactly as before. The rationale is recorded next to each flag, matching the explanatory comment style the file already uses.

## Verification

Verified: the workflow parses, and all three concurrency blocks resolve to the same condition (top level, `xla-link`, `cuda-sm70-compile`). `actionlint` reports byte-identical findings before and after the change, 8 pre-existing (4 `runner-label` for the custom GB10 label, 4 `shellcheck`), with none introduced.

Not verified: the `main` behaviour itself. Demonstrating it requires two merges to `main` in quick succession where the second also passes the job's path filter, which cannot be staged safely on shared infrastructure. PR #1780 left the same criterion unchecked for the same reason, and this change inherits that limitation rather than resolving it.

An edge case worth recording: cancellation on `main` only ever arose when the second merge also matched the job's `if:` path filter. A merge touching no matching path skips the job entirely, so it never joins the group.
