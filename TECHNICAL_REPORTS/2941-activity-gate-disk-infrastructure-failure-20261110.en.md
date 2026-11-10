# PR #2941: Report Activity gate disk exhaustion as an infrastructure failure

**Date**: 2026-11-10
**Status**: Completed (PR open as a draft; closes #2940)

## Executive Summary

The WebUI Activity performance gate on the self-hosted GB10 runner (spark-101) now says "infrastructure" when the host disk is the reason it failed. Two mechanisms do this. The host precheck refuses to start when `$RUNNER_TEMP` has less than `--min-free-gb` free (default 5 GB), and the verifier classifies a failed activity script whose log tail shows ENOSPC, EDQUOT, EIO or an unhandled stream `'error'` event as `activity gate infrastructure failure` instead of `activity performance verification failed`. Both paths still exit 1. The performance budget, the verdict policy and the verdict logic are untouched. Python CI helpers only: 6 files, +259/-8.

## 1. Problem Statement

CI run 42753658537 (main `6cd038eb`) failed the `WebUI installed artifact` job with `activity performance verification failed: activity-performance.mjs exited 1`. The diagnostics artifact shows the real cause at the end of `activity-performance.log`: Node's `Error: ENOSPC: no space left on device, write`, raised from `process.stdout.write` inside the pair loop of `webui/scripts/activity-performance.mjs` and emitted as an unhandled `'error'` event on a `SyncWriteStream`. The samples written before the crash were normal (off and one-visible both around 164 to 167 tok/s), and the verdict never ran.

The message pointed a reader at a performance regression when the runner's root filesystem was full. That costs triage time, and it invites the wrong fix (loosening the 2 percent budget). The issue asked for the failure to be named for what it was, without changing whether the job fails.

## 2. What Changed

### 2.1 Free-space precheck (`scripts/webui/activity_gate_host.py`)

- New `precheck` option `--min-free-gb` (float, default 5.0, decimal GB = 1e9 bytes). `parse_args` rejects negative and non-finite values; `nan` would otherwise compare false against everything and silently disable the check.
- `precheck()` calls `shutil.disk_usage(args.run_root)` first, before the leak sweep, the GPU check and the 600-second quiet wait, so a full disk fails immediately rather than after ten minutes. It raises `GateRefused("infrastructure: only N GB free on <path>, need M GB")`, which flows through the existing refusal handler (a `::error::` annotation and exit 1).
- The observation is recorded as `state["disk"]` (path, free bytes, minimum) in the host-state JSON that the evidence embeds.
- `disk_usage` is an injectable keyword parameter, matching the existing `gpu_query` and `read_runnable` seams, so the tests stub it instead of depending on the test host's disk.
- The host-state write in the `finally` block is now best effort: after a disk refusal it can itself hit ENOSPC, and an uncaught error there would replace the refusal message with a traceback. A failure emits a `::warning::` and the exit code stays 1.
- `.github/workflows/ci.yml` passes `--min-free-gb 5` explicitly and the step comment describes both behaviors.

### 2.2 Failure classification (`scripts/webui/verify_activity_performance.py`)

- `classify_environment_failure(log_path)` seeks to the last 64 KiB of the activity log and returns a short reason when it finds ENOSPC, EDQUOT or EIO as whole tokens (the last match wins), or an unhandled stream error event. It returns only fixed strings, never log text, so nothing untrusted from the log reaches the CI message.
- `run_activity_script()` calls it only on a nonzero exit. A match raises `ActivityInfrastructureError` (an `AssertionError` subclass, so every existing `except` path still treats it as a failure) naming the reason, exit code and log path. Without a match the original `AssertionError("activity-performance.mjs exited N; log=...")` is unchanged.
- `run()` records `failure_class: infrastructure` in the evidence for that error.
- `main()` labels a failure as infrastructure when the exception, or anything on its `__cause__`/`__context__` chain, is an `ActivityInfrastructureError` or an `OSError` with errno ENOSPC, EDQUOT or EIO. The second case covers disk errors raised by the verifier's own writes (evidence, run directory). Everything else keeps the old label. Both return 1.
- Evidence flushes in the failure handler and the cleanup `finally` go through `Harness.flush_best_effort()`. On a full disk the old `h.flush()` there raised ENOSPC itself and replaced the original exception, so the gate would have printed the old label in exactly the case this PR targets.

### 2.3 Documentation

`docs/webui-integration-matrix.md` (host precheck paragraph) documents the disk check and its default.

## 3. Technical Decisions

### 3.1 Why 5 GB and not the issue's 20 GB

The threshold should refuse only when a run genuinely cannot complete. What the gate itself writes into `$RUNNER_TEMP` per run is small: a private run directory holding a reflink/copy view of the 336 MB qwen3-0.6b-4bit fixture, the Chromium profile under the run directory's `HOME`, and server, Xvfb, openbox and activity logs plus evidence JSON in the tens of KB each (the failed run's diagnostics totalled about 40 KB). The installed binaries (672 MB) and the fixture cache are already on disk before the precheck runs. That is well under 1 GB, so 5 GB leaves several times the footprint for other writers on the same filesystem during a 10 to 45 minute gate.

spark-101 had about 26 to 28 GB free when this was written. A 20 GB floor would pass today but leave almost no margin before refusing runs that would complete fine, turning a disk-space trend into gate flakiness.

### 3.2 Why the classification matters more than the threshold

The failed run started with roughly 48 GB free, and the disk was filled by concurrent writes during the gate. No start-time threshold below 48 GB would have caught it. The precheck handles the case where the disk is already low at start; the log classification is what labels the mid-run case that actually happened.

### 3.3 Failures stay failures

Both mechanisms change the message only. A refusal or a classified failure still exits 1 and fails the job. Turning infrastructure failures into skips would let a broken runner report green indefinitely, which the gate's fail-closed design (#1949) exists to prevent.

## 4. Review Findings and Fixes

- **Masked label on a full disk (MEDIUM, fixed).** The security review noticed that the evidence flush in `run()`'s failure handler would raise ENOSPC on the same full disk and replace `ActivityInfrastructureError`. Fixed by `flush_best_effort()` and the cause-chain check in `main()`. A test makes every evidence flush raise ENOSPC and asserts the infrastructure error still propagates.
- **Non-finite `--min-free-gb` (LOW, fixed).** `nan` disabled the check; now rejected.
- **`reap` parse crash (fixed before merge).** The first version of the non-finite check lacked parentheses (`A and B or C`), so `reap`, which has no `--min-free-gb`, evaluated `args.min_free_gb` and raised `AttributeError`. That would have broken the always-run `Reap Activity gate processes` step on every run. Fixed and covered by a test that parses a `reap` command line.
- **Broad stream-error match (LOW, accepted).** Any unhandled `'error'` event (for example a socket ECONNRESET with no disk errno) is labelled infrastructure. The issue asked for this, the exit code is unaffected, and today's sample lines contain no such text.

## 5. Verification

- `python3 scripts/webui/activity_gate_host_tests.py`: 22 tests OK (low-disk refusal naming the free space and path, never reaching the GPU query or quiet wait; a passing case; refusal kept when the host-state write fails; argument validation; `reap` parsing).
- `python3 scripts/webui/verify_activity_performance_tests.py`: 32 tests OK. The fixture is a trimmed copy of the real tail of run 42753658537's log with the runner path neutralized. It runs through `run_activity_script` with a fake `node_bin` shell script that prints the fixture and exits 1, and through `main` with `run` patched. A budget-style failure keeps the old message; chained ENOSPC and EIO are labelled, EACCES is not.
- `python3 -m unittest discover -s tests -p 'test_*.py'`, `scripts/insert_apache_header.py --check`, `scripts/ci/check_env_var_docs.py`, `cargo fmt --all -- --check`: clean.
- On a real host: `precheck --min-free-gb 100000` refused with exit 1 and `infrastructure: only 1588.5 GB free on <path>, need 100000 GB`; `--min-free-gb 1` passed the disk check and the rest of the precheck; `reap` with no pid file exited 0.

## 6. Learning Points

- A failure handler that writes to disk can be defeated by the very failure it reports. When the error class is "disk full", any write on the reporting path needs to be best effort or the label is lost.
- `A and B or C` in a guard that should apply only to one subcommand is an easy precedence slip; argparse subcommand options do not exist on the other subcommands' namespaces, so the slip surfaces as an `AttributeError` in an unrelated command.
- A start-of-run resource check and an after-the-fact classifier answer different questions. The first prevents starting on a broken host; the second names a failure that develops mid-run. The incident that motivated this change was the second kind.

## 7. Not Covered

- Freeing space on spark-101 is an operational task outside the repository.
- The classifier has only been exercised against the recorded ENOSPC tail and synthetic EDQUOT/EIO logs; a real EDQUOT or EIO from Node on the runner has not been observed.
