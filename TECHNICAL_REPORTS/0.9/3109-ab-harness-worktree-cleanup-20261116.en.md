# Technical Report: PR #3109 - remove the A/B build worktrees on every exit path

**Date**: 2026-11-16
**Status**: Completed
**Languages**: Bash
**Risk Level**: Low

---

## Executive Summary

The paired A/B harness `scripts/xla/ab_decode_pair.sh` built each git-ref arm in a temporary worktree of about 2.2 GB and never removed them, because the worktree registration was appended inside a command substitution and the EXIT trap saw an empty list. Six leaked worktrees filled the 16 GiB `/tmp` tmpfs on the measurement host and killed a measurement run. The same harness read its measured rounds back from two fixed `/tmp` files and silently dropped any round it could not parse, so a ten-round run reported `n=3` with no error. Worktree paths are now recorded in a per-run file, cleanup runs on EXIT, INT, TERM and HUP, and the summary refuses to report when the number of rounds that reached it is not the number requested.

---

## 1. Problem Statement

### 1.1 Background

Decode performance work on the OpenXLA/IREE backend is measured with a paired interleaved A/B: each arm runs alternately, medians are reported, and the greedy text has to match. An arm named by git ref is built in its own worktree so two revisions can be measured from one command.

### 1.2 Existing Issues

- **Issue 1**: the worktree registration `WORKTREES+=("$wt")` ran inside `build_arm`, which is only reached through `resolve_arm`, and both call sites used command substitution (`NEW_BIN=$(resolve_arm ...)`). A subshell's array append does not reach the parent, so the EXIT trap looped over an empty list on every run, not only on interrupts.
- **Issue 2**: removal failures were discarded with `|| true`, so a failed removal looked like a successful one.
- **Issue 3**: a `git worktree add` that failed before registering left a directory nothing tracked, and no INT/TERM/HUP trap existed.
- **Issue 4**: measurement values lived in `/tmp/ab_decode_pair_old.txt` and `/tmp/ab_decode_pair_new.txt`, and the summary skipped unparseable lines. A round lost to a full `/tmp` shrank the sample without any error.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| `/tmp` fills during a long measurement campaign | High (run dies, output lost) | High before this change |
| A reported median covers fewer rounds than its label claims | High (wrong performance conclusion) | Medium before this change |
| Cleanup removes a worktree a concurrent run owns | High | Low (it only removes what it recorded; leftovers are reported, not deleted) |

---

## 2. Technical Review

### 2.1 Security

No new input, network or file surface. The record file is created by `mktemp` under `$TMPDIR` and every path in it was created by the script itself.

**Checklist:**
- [x] Input validation (paths come from the script's own `mktemp`)
- [x] Authentication/Authorization (not applicable)
- [x] Data encryption (not applicable)
- [x] Logging (paths only, no model data)

### 2.2 Performance

| Area | Before | After |
|------|--------|-------|
| Disk left behind per run with one rebuilt arm | 2.2 GB | 0 |
| Disk left behind per run with two rebuilt arms | 4.4 GB | 0 |
| Rounds reported vs rounds run | could differ silently | must match, else exit 1 |

### 2.3 Compatibility & Dependencies

- **Breaking Changes**: none. The measurement table on stdout is unchanged; all cleanup output goes to stderr.
- **New Dependencies**: none (bash, `git`, `mktemp` already required).
- **Compatibility**: `--keep-worktrees` keeps its meaning and now also prints a count.

### 2.4 Code Quality

- **Test Coverage**: `scripts/ci/ab_decode_pair_worktree_test.sh` drives the real script in a throwaway repository with a `cargo` stub and checks 14 assertions across three exit paths: normal, SIGTERM during a build, and a run missing a round. Wired as the `ab harness cleanup` CI job (no toolchain, about eight seconds).
- **Code Complexity**: one record file replaces one array, three signal traps are added, and the summary grows one count check.
- **Technical Debt**: decreased; the silent-failure paths (`|| true` on removal, unparsed value lines) are gone.

---

## 3. Technical Decisions

### 3.1 A record file instead of rewriting the call sites

**Context:** an array cannot cross a subshell boundary, and the arms are resolved in `$(...)`.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Out-parameter (`resolve_arm` sets a global `RESOLVED_BIN`) | No new file | Rewrites both call sites and the function contract; the next refactor can reintroduce the pattern |
| Register in the parent after the call | Simple | The parent does not know the path until the build finishes, so a failed build leaks |
| **Chosen: append to a per-run `mktemp` file** | Survives subshells; registration can happen before the worktree exists | One temporary file per run |

**Rationale:** the failure mode was precisely "the parent cannot see what the subshell did", and a file is the cheapest thing both sides can see. Recording the path before `git worktree add` also covers a failed add or a failed build.

**Trade-offs:** one more temporary file, removed by the same trap.

### 3.2 Report leftover worktrees instead of deleting them

**Context:** a run killed with SIGKILL cannot clean up, and old runs left worktrees with no record.

**Rationale:** deleting anything under `${TMPDIR}/mlxcel-ab-` at startup would destroy an active run's build on another device. Printing the paths and the reclaim command keeps the decision with the operator.

---

## 4. Implementation Details

### 4.1 Key Code Changes

**File: `scripts/xla/ab_decode_pair.sh`**
```bash
# Before: the append lived in a subshell
WORKTREES+=("$wt")
# After: a file both the subshell and the parent can write and read
printf '%s\n' "$wt" >> "$WORKTREE_LOG"
```
```bash
# Before
old = load('/tmp/ab_decode_pair_old.txt')
# After: per-run files, and the round count is enforced
if len(old) != rounds or len(new) != rounds:
    print(f"... {len(old)} old and {len(new)} new rounds reached the summary, "
          f"expected {rounds}; the run is not reportable", file=sys.stderr)
    sys.exit(1)
```

**Reason for change:** the first removes the leak on every exit path; the second makes a lost round a failure instead of a smaller sample.

---

## 5. Learning Points

### 5.1 A subshell silently discards shell state

**Concept:** `$(...)` and `( ... )` run in a child shell. Variables, arrays, traps and `umask` changes made there do not reach the parent; files, exit status and stdout do.

**Application in this PR:** the leak existed because the registration was a side effect of a function whose stdout the caller needed. The fix moved the side effect to the one channel a subshell shares.

**Common Use Cases:**
- Functions invoked in command substitution that also need to mutate caller state (use an out-parameter or a file, not an array)
- Progress reporting from a subshell

### 5.2 A measurement harness is code that needs tests

**Concept:** performance numbers are only as trustworthy as the harness that produced them, and harness defects are silent by nature: they change a number, not a status.

**Application in this PR:** both defects produced plausible output. The new CI job drives the script with a stub toolchain and asserts the three exit paths, so the harness is now covered the way the code under measurement is.

---

## 6. Change Summary

### Statistics
| Item | Value |
|------|-------|
| Files changed | 4 |
| Lines added | +286 |
| Lines deleted | -18 |
| Tests added | 1 (14 assertions) |

### Changes by Category

| Category | Count | Summary |
|----------|-------|---------|
| Bug fix | 2 | Worktree cleanup on every exit path; round-count guard in the summary |
| Tests | 1 | `scripts/ci/ab_decode_pair_worktree_test.sh` |
| CI | 1 | `ab harness cleanup` job |
| Documentation | 1 | `docs/benchmarks.md` worktree lifetime and summary contract |

### Related Commits
| Hash | Type | Message |
|------|------|---------|
| `8fad385a` | fix | remove the A/B build worktrees on every exit path |
| `88182a91` | test | cover the A/B harness worktree cleanup and round guard |
| `791dba22` | fix | skip the removal attempt for a worktree that was never created |

---

## 7. Follow-up Actions

### Required
- [ ] None.

### Monitoring Required
- The next long A/B campaign: `/tmp` should stay flat across runs, and each run should end with `cleaned N worktree(s)`.

### Future Improvements
- `scripts/ci/ab_scripts_outdir_test.sh` exists but nothing runs it; wiring it into CI would give the sibling harness scripts the same coverage.
- A SIGKILL still leaves a worktree, which the startup warning reports but cannot remove.

---

## Appendix

### A. Test Results

- `bash -n` clean on both scripts.
- `scripts/ci/ab_decode_pair_worktree_test.sh`: 14 of 14 checks pass.
- CI run `43793074946`: `ab harness cleanup` success; every other non-skipped job success.
- Real paired A/B, `gfx1151`, 8 rounds: `n=8` summary, `removed: /tmp/mlxcel-ab-6kLQAW`, worktree list and `$TMPDIR` clean afterwards, `/tmp` back to 20 percent.
