"""Per-run host busy gate for the #1798 harnesses: block while compiler-like
processes (the self-hosted CI runner's builds, or anyone's) are alive, and
report the CI runner's job state so each record can state host idleness.

Since #1820 it also blocks on a foreign inference process. Two large models
resident at once wedged this host on 2026-09-10, and during the #1820 long
context sweep another session loaded its own model mid-run and drove the
kernel's NV_ERR_NO_MEMORY count from 0 to 356 in two minutes. The compiler gate
could not see that, so the run's idle-host precondition was violated without
the record being able to say so. FOREIGN lists the process names a model run
shows up under; a benchmark binary under test should be given a distinct name
(the #1820 sweeps run a copy called `mlxcel-1820`) so it does not gate itself.
"""
import subprocess, time, os

BUSY = ("rustc", "cc1plus", "cc1", "nvcc", "cicc", "ptxas", "cmake", "ninja", "ld", "ld.lld", "lld",
        "clang", "clang++", "gcc", "g++", "fatbinary", "cargo-clippy", "clippy-driver")
FOREIGN = ("mlxcel", "mlxcel-server", "mlxcel-bench-de")


def _procs():
    out = subprocess.run(["ps", "-eo", "comm="], capture_output=True, text=True).stdout.split()
    return out


def ci_job_running():
    return "Runner.Worker" in _procs()


def busy_procs():
    return sorted({p for p in _procs() if p in BUSY})


def foreign_model_procs():
    """Model processes this harness did not start. Never empty-by-assumption:
    an unexpected name here is a reason to wait, not to proceed."""
    return sorted(p for p in _procs() if p in FOREIGN)


def wait_quiet(samples=2, interval=5, log=None):
    n = 0; waited = 0
    while n < samples:
        b = busy_procs()
        f = foreign_model_procs()
        if b or f:
            n = 0
            if log and waited % 60 == 0:
                print(f"[gate] busy: {b} foreign_models: {f} ci_job={ci_job_running()} "
                      f"load1={os.getloadavg()[0]:.2f}", file=log, flush=True)
            time.sleep(interval); waited += interval
        else:
            n += 1
            if n < samples:
                time.sleep(interval)
    return waited
