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


def wait_quiet(samples=12, interval=5, log=None):
    """Require SUSTAINED quiet, and treat an active CI job as busy.

    Two samples five seconds apart was not enough on a shared self-hosted
    runner (#1820): a lull between two queued CI runs clears a 10-second check,
    and the timed run that starts in it is then overrun when the next run ramps.
    Twelve samples is a minute of continuous quiet, which is longer than the
    gaps between a CI job's own steps.

    `Runner.Worker` is gated here rather than merely recorded. A CI job that is
    between compiler invocations is not an idle host, it is a host about to
    compile again, and on this box CI is the single largest source of exactly
    the contention that invalidates a timed run."""
    n = 0; waited = 0
    while n < samples:
        b = busy_procs()
        f = foreign_model_procs()
        if ci_job_running():
            b = b + ["Runner.Worker"]
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


# --- Driver gate (#1820) ---------------------------------------------------
# The GB10 freeze precursor documented on this host is CUMULATIVE
# NV_ERR_NO_MEMORY over hours, not the instantaneous rate: a spiky delivery
# whose running total keeps rising is the signature, so the cumulative count is
# the primary gauge and the per-window rate is the secondary one. The 2026-09-17
# halt read a per-second burst shape as reassurance while the total went
# 2 -> 77 -> 184; that inference was wrong and this gate exists so it cannot be
# made by eye again.
NVRM_WINDOW_MAX = int(os.environ.get("NVRM_WINDOW_MAX", "50"))
NVRM_TOTAL_MAX = int(os.environ.get("NVRM_TOTAL_MAX", "400"))


class DriverHalt(RuntimeError):
    """Raised instead of starting a run when the driver trip wire is over."""


def _jctl(*args):
    try:
        out = subprocess.run(["journalctl", "-k", *args, "--no-pager"],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return 0
    return sum(1 for line in out.splitlines() if "NV_ERR_NO_MEMORY" in line)


def nvrm_counts():
    """(errors in the last 5 minutes, cumulative errors since boot)."""
    return _jctl("--since", "5 min ago"), _jctl("-b", "0")


def driver_gate(quiet_s=300, log=None):
    """Block until the driver has been quiet, and refuse outright once the
    cumulative count is over budget. Quiet is measured on the 5-minute window;
    the cumulative ceiling is not waitable, since it never falls."""
    waited = 0
    while True:
        window, total = nvrm_counts()
        if total >= NVRM_TOTAL_MAX:
            raise DriverHalt(
                f"cumulative NV_ERR_NO_MEMORY {total} >= {NVRM_TOTAL_MAX}; "
                "stop and report rather than waiting, this one does not decay")
        if window < NVRM_WINDOW_MAX:
            return waited, window, total
        if log and waited % 60 == 0:
            print(f"[driver] NV_ERR_NO_MEMORY window={window} total={total}, waiting",
                  file=log, flush=True)
        time.sleep(30)
        waited += 30
        if waited > quiet_s * 4:
            raise DriverHalt(
                f"driver did not go quiet in {waited}s (window={window} total={total})")


# --- Memory gate (#1820) ---------------------------------------------------
# Load and driver were not sufficient. On 2026-09-17 three foreign builds from
# peer sessions drove MemAvailable to the floor and the harness supervisor
# killed background tasks three times; a model run started in that state dies
# mid-row rather than returning a bad number, which is worse, because a bad
# number is at least visible in the spread.
#
# The floor below is DERIVED, not measured: 20.97 GiB of weights measured on
# disk (nvfp4 target 20.11 + dflash drafter 0.86; GB10 memory is unified, so
# weights count against system RAM) plus 0.69 GiB of KV at 16k computed from
# config.json (only 10 of the target's 40 layers are full attention, the other
# 30 and all 5 drafter layers are capped at a 512 window), which is about
# 21.7 GiB resident, plus headroom for prefill-chunk activations and cuDNN
# workspace. `peak_rss_kib` is now recorded per run so this can be replaced
# with the measured figure rather than left as arithmetic.
MEM_FLOOR_GIB = float(os.environ.get("MEM_FLOOR_GIB", "32"))


def mem_available_gib():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024.0 * 1024.0)
    except Exception:
        pass
    return float("inf")


def mem_gate(log=None, max_wait_s=7200):
    """Block until MemAvailable clears the floor. Unlike the driver's
    cumulative count this does decay, so waiting is the right response."""
    waited = 0
    while True:
        avail = mem_available_gib()
        if avail >= MEM_FLOOR_GIB:
            return waited, avail
        if log and waited % 120 == 0:
            print(f"[mem] MemAvailable {avail:.1f} GiB under the {MEM_FLOOR_GIB:.0f} GiB floor, waiting",
                  file=log, flush=True)
        time.sleep(30)
        waited += 30
        if waited > max_wait_s:
            raise DriverHalt(
                f"MemAvailable stayed under {MEM_FLOOR_GIB:.0f} GiB for {waited}s "
                f"(now {avail:.1f} GiB); the host is not usable for a timed run")
