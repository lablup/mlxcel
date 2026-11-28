# Technical Report: PR #2119 - Reap leaked Activity gate servers and wait for a quiet host

**Date**: 2026-10-06

**Status**: Implemented and verified on the GB10 CI runner; pending merge.

**Languages**: Python (gate host helper, verifier), YAML (workflow)

**Risk Level**: Low. The change touches one CI job. It signals only processes that carry the job's identity and have no controlling terminal, and the passing path's graceful shutdown requirement is unchanged.

## Executive Summary

The WebUI Activity performance gate on the GB10 runner leaked its server whenever the verifier died or its cleanup gave up, and the leaked server then failed every later run at a GPU precondition that printed only a count. The gate also scored while other work loaded the host. A new helper, `scripts/webui/activity_gate_host.py`, now stops leaks from earlier runs and waits for a quiet host before scoring, and an always-run step reaps whatever a failed run left behind (#1949). On the runner, the first CI run of this PR found and stopped the real leak that had failed PR #2117 five times.

## 1. Problem Statement

**Fault 1: the leak.** The verifier starts the server, Xvfb, openbox, node and (through Playwright) Chromium, each in a session of its own, so a step timeout, a cancellation or a killed verifier never reaches them. The in-process `finally` was the only cleanup, and it had three gaps: it never runs when the verifier is killed; its SIGKILL wait was unguarded, so a process that outlived it raised out of `finally`; and it recorded `process_group_empty: false` without acting on it. A leaked server holding 4 to 5 GiB of GPU memory then made the next run's `nvidia-smi` precondition fail with only a count. On 2026-10-05, `main` at 680f64ee failed in scoring, and the next five runs of PR #2117 all failed at that precondition.

**Fault 2: scoring under load.** Two earlier PRs that touched no WebUI code reported 2.8 and 3.9 percent decode degradation, above the 2 percent budget, while another job compiled on the same host. Both passed when re-run on a quiet host. A free GPU is not enough. The host itself has to be quiet.

## 2. Change Summary

- **`activity_gate_host.py precheck`** (first command of the gate body, after the CI flock and `gpu-lock`):
  - Stops processes of the job's identity that an earlier run leaked.
  - Fails closed on any remaining GPU compute process, both before and after the quiet wait.
  - Waits up to 600 s for the host to quiet down.
  - Writes the host state it observed.
- **`activity_gate_host.py reap`** (new `if: always()` step): stops what the verifier's pid file names and anything else of the job's identity, and fails the job if anything survives.
- **`verify_activity_performance.py`**:
  - `--pid-file` records every owned process.
  - `--host-state` embeds the precheck's state into the full evidence, plus an after-activity load sample.
  - `terminate_owned` no longer raises from `finally`, and it now stops leftover group members.
  - Xvfb and openbox start in the run directory.
- **Workflow**: the gate step timeout goes from 65 to 75 minutes and the job timeout from 90 to 120. The new tests join `make verify-webui-helper-tests`.

## 3. Technical Decisions

**Identity, not ancestry.** A leaked process has lost its parent, so ancestry cannot find it. The helper recognises the job's processes by things the job controls:
- an executable in `$RUNNER_TEMP/mlxcel-webui-installed/`;
- a working directory in a verifier run directory;
- a `HOME` in a verifier run directory.

`HOME` is the only handle that reaches Chromium. Playwright launches the browser `detached`, so it sits in its own process group with the repository as its working directory, but it inherits the verifier's environment, which sets `HOME=<run dir>/home`. `GITHUB_*` variables cannot serve as a handle, because `clean_env` drops them. A process counts as a leak from an earlier run when it is orphaned or its run directory is gone. The runner empties `RUNNER_TEMP` between jobs, which is why the leaked server read `(deleted)` for both its executable and its working directory.

**Signal narrowly, because the runner is shared.** Developers log into the runner as the same uid, so EPERM protects nothing. The security review showed that the first version would have SIGKILLed the whole process group of a developer's shell sitting in a run directory. Three rules prevent this:
- Processes with a controlling terminal are skipped.
- Processes matched only by directory or `HOME` are signalled one at a time.
- Whole groups are signalled only for session leaders the job is known to have started: pid-file records, and leaders running an installed-artifact executable.

Each signal goes through a pidfd opened before the start time is re-checked, so a reused pid is never signalled. Pid-file records carry the boot id.

**Runnable tasks, not load averages.** `/proc/loadavg`'s instantaneous runnable count reacts within a second. The 1-minute load average and `ps %cpu` both read a freshly started compile as idle. The threshold is a mean over 10 one-second samples, at most a quarter of the CPUs. A threshold of 5 on GB10 separates an idle runner (1.1 to 1.4 measured) from a compile or a busy loop per CPU (23.8). The helper never refuses before the window has filled; the GB10 simulation caught a version that did.

**GPU check before and after the wait.** Checking only after the quiet wait would hide a GPU conflict behind up to ten minutes of host load while holding both locks.

**Public-repository redaction.** Logs and artifacts of lablup/mlxcel are readable by anyone. For a GPU process that is not the job's, the gate reports only pid, process basename, memory, parent and uid. Host snapshots list compiler names and pids, never arbitrary command names.

## 4. Validation

- Helper tests:
  - 18 host-helper tests run real processes from a copy of `sleep` placed where the job puts its server.
  - 21 verifier tests, including the two `terminate_owned` tests, which fail on the previous code.
  - 10 summary tests, including one checking that `host_state` neither blocks nor leaks into the summary.
- Runner `lablup-dgxspark21` (spark-101):
  - Run 37387059748 stopped the real leak (pid 1453937, PPID 1, executable and cwd `(deleted)`) and then passed.
  - Run 37388347876 (final head) passed, with a mean of 1.4 runnable tasks and the graceful shutdown intact.
  - Throwaway probe run 37388422039:
    - With one busy loop per CPU, the gate refused at a mean of 23.8.
    - With the load removed, it passed at 1.1.
    - After the verifier was SIGKILLed mid-activity (server at 4604 MiB), the reap stopped Xvfb, openbox, the server, node and Chromium, and `nvidia-smi` and `pgrep` were empty afterwards.
- GB10 simulation on spark-102 with a real server under `gpu-lock`:
  - A leaked server was stopped.
  - A foreign GPU process was refused at once, without its path, and left running.
  - A busy host was refused.
  - A server left by a SIGKILLed verifier was reaped by pid file.

## 5. Residual Risks

- Two processes of this job running at the same time would treat each other as their own. One runner runs jobs sequentially, and other runner registrations have their own `_work/_temp`.
- The quiet check looks only at gate start. Load that starts mid-measurement still moves the score. The `after_activity` sample makes that visible afterwards but does not prevent it. The verdict rule itself belongs to #1925.
- The reap log says `signals none` for a process that exited on its own between the sweep and the stop (openbox, node and Chromium die with Xvfb). The JSON report is unambiguous.
- `gpu-lock` covers development sessions that use it. GPU work started outside it still fails the gate, now with its pid and owner named.

## 6. Learning Points

- **A process that starts its own session needs a cleanup outside the process that started it.** In-process `finally` cleanup cannot outlive a SIGKILL. An always-run step with a pid file fills that gap, and an identity sweep covers whatever the pid file never saw.
- **Check the shared-host blast radius of every kill.** On a multi-user runner under one uid, permissions stop nothing. The filters that matter are a controlling terminal, matching each process individually, and pidfds.
- **Measure "quiet" with an instantaneous signal.** Load averages and lifetime `%cpu` lag exactly the event the gate needs to see.
- **Real-host simulation finds what unit tests do not.** The window-versus-timeout bug, and the fact that Chromium escapes the node process group, both came from running the real thing.

## 7. Related

- Issue #1949.
- PR #2117 (the run that hit the leak five times).
- #1925 (verdict rule, out of scope).
- PR #2114 (`gpu-lock` wrapper).
- Probe PR #2120 (closed).
