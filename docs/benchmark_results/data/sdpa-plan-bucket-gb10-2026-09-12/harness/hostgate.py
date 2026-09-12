"""Per-run host busy gate for the #1798 harnesses: block while compiler-like
processes (the self-hosted CI runner's builds, or anyone's) are alive, and
report the CI runner's job state so each record can state host idleness."""
import subprocess, time, os

BUSY = ("rustc", "cc1plus", "cc1", "nvcc", "cicc", "ptxas", "cmake", "ninja", "ld", "ld.lld", "lld",
        "clang", "clang++", "gcc", "g++", "fatbinary", "cargo-clippy", "clippy-driver")


def _procs():
    out = subprocess.run(["ps", "-eo", "comm="], capture_output=True, text=True).stdout.split()
    return out


def ci_job_running():
    return "Runner.Worker" in _procs()


def busy_procs():
    return sorted({p for p in _procs() if p in BUSY})


def wait_quiet(samples=2, interval=5, log=None):
    n = 0; waited = 0
    while n < samples:
        b = busy_procs()
        if b:
            n = 0
            if log and waited % 60 == 0:
                print(f"[gate] busy: {b} ci_job={ci_job_running()} load1={os.getloadavg()[0]:.2f}", file=log, flush=True)
            time.sleep(interval); waited += interval
        else:
            n += 1
            if n < samples:
                time.sleep(interval)
    return waited
