#!/usr/bin/env python3
# Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
"""Host-side guards around the WebUI Activity performance gate (issue #1949).

``precheck`` runs inside the gate, after the CI locks and before scoring. It
stops any process an earlier run of this job leaked, waits a bounded time for
the host to be quiet enough to measure, and then fails closed, naming each
process, if any GPU compute process remains. It writes the host state it
observed so the verifier can put it into the evidence JSON.

``reap`` runs as an ``always()`` step after the gate. It signals what the
verifier recorded in its pid file and anything else carrying this job's
identity, reports what it had to do, and fails if anything survives, so a leak
fails the run that caused it instead of the next one.

A process carries this job's identity when its executable is in the
installed-artifact directory, or when its working directory or ``HOME`` is a
verifier run directory (the verifier gives the server, node and the browser
Playwright launches ``HOME=<run directory>/home``; Playwright starts the
browser in a process group of its own, so the group of what the verifier
started does not reach it). Processes with a controlling terminal are never
touched: CI processes have none, and a developer's shell on the runner does.

Both commands read ``/proc`` and are Linux-only, like the GB10 runner. The
host state is uploaded from a public repository, so it names the executable
and directory only of processes that carry this job's identity.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROC = Path("/proc")
DELETED = " (deleted)"
# unique_work_dir() in verify_activity_performance.py creates run directories with this prefix.
WORK_PREFIX = "mlxcel-activity-performance-"
COMPILERS = {"cargo", "rustc", "cc", "c++", "cc1", "cc1plus", "gcc", "g++", "clang", "clang++", "ld", "ld.lld", "lld", "mold", "nvcc", "cicc", "ptxas", "fatbinary", "nvlink", "cmake", "ninja", "make"}


class GateRefused(Exception):
    """The host is not in a state the gate may score in."""

    def __init__(self, message: str, quiet: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.quiet = quiet


@dataclass(frozen=True)
class ProcInfo:
    pid: int
    ppid: int
    pgid: int
    state: str
    start_ticks: int
    comm: str
    uid: int | None = None
    exe: str | None = None
    cwd: str | None = None
    tty_nr: int = 0

    @property
    def exe_path(self) -> str | None:
        return strip_deleted(self.exe)

    @property
    def cwd_path(self) -> str | None:
        return strip_deleted(self.cwd)

    @property
    def directory_deleted(self) -> bool:
        return bool(self.cwd and self.cwd.endswith(DELETED))

    @property
    def executable_deleted(self) -> bool:
        return bool(self.exe and self.exe.endswith(DELETED))

    def describe(self) -> dict[str, Any]:
        return {"pid": self.pid, "ppid": self.ppid, "pgid": self.pgid, "comm": self.comm, "exe": self.exe, "cwd": self.cwd}


@dataclass(frozen=True)
class Match:
    """Why a process carries this job's identity, and how it may be signalled."""

    reason: str
    # Signal the whole process group. Only for a group this job is known to lead: an
    # installed-artifact executable leading its own group, or a pid-file record.
    whole_group: bool
    # The run directory the match points at no longer exists, so the run that owned it ended.
    stale: bool = False


def strip_deleted(path: str | None) -> str | None:
    if path is None:
        return None
    return path[: -len(DELETED)] if path.endswith(DELETED) else path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def annotate(level: str, message: str) -> None:
    print(f"::{level}::{message}", flush=True)


def write_atomic(path: Path, body: str) -> None:
    """Write a 0600 file by exclusive create and rename, so no symlink is followed and no reader sees half."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    staging.unlink(missing_ok=True)
    fd = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fp:
        fp.write(body)
    os.replace(staging, path)


def boot_id(root: Path = PROC) -> str | None:
    try:
        return (root / "sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None


def read_proc(pid: int, root: Path = PROC) -> ProcInfo | None:
    base = root / str(pid)
    try:
        stat_text = (base / "stat").read_text()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    # comm sits in parentheses and may itself contain spaces or parentheses.
    open_paren, close_paren = stat_text.find("("), stat_text.rfind(")")
    if open_paren < 0 or close_paren < 0:
        return None
    fields = stat_text[close_paren + 2 :].split()
    try:
        state, ppid, pgid, tty_nr, start_ticks = fields[0], int(fields[1]), int(fields[2]), int(fields[4]), int(fields[19])
    except (IndexError, ValueError):
        return None

    def link(name: str) -> str | None:
        try:
            return os.readlink(base / name)
        except OSError:
            return None

    try:
        uid: int | None = base.stat().st_uid
    except OSError:
        uid = None
    return ProcInfo(pid, ppid, pgid, state, start_ticks, stat_text[open_paren + 1 : close_paren], uid, link("exe"), link("cwd"), tty_nr)


def read_home(pid: int, root: Path = PROC) -> str | None:
    try:
        environ = (root / str(pid) / "environ").read_bytes()
    except OSError:
        return None
    for entry in environ.split(b"\0"):
        if entry.startswith(b"HOME="):
            return entry[5:].decode(errors="replace")
    return None


def list_procs(root: Path = PROC) -> list[ProcInfo]:
    procs = []
    for entry in root.iterdir():
        if entry.name.isdigit():
            info = read_proc(int(entry.name), root)
            if info is not None:
                procs.append(info)
    return procs


def own_lineage(root: Path = PROC) -> set[int]:
    """This process and its ancestors, which no sweep may signal."""
    lineage, pid = set(), os.getpid()
    while pid > 1 and pid not in lineage:
        lineage.add(pid)
        info = read_proc(pid, root)
        if info is None:
            break
        pid = info.ppid
    return lineage


@dataclass(frozen=True)
class Identity:
    """What a process this job started looks like from outside."""

    installed_dir: str
    run_root: str

    @classmethod
    def from_paths(cls, installed_dir: Path, run_root: Path) -> Identity:
        return cls(os.path.realpath(installed_dir), os.path.realpath(run_root))

    def run_directory(self, path: str | None) -> str | None:
        """The verifier run directory ``path`` is or lies in, if any."""
        if not path:
            return None
        relative = os.path.relpath(path, self.run_root)
        top = relative.split(os.sep, 1)[0]
        return os.path.join(self.run_root, top) if top.startswith(WORK_PREFIX) else None

    def match(self, info: ProcInfo, root: Path = PROC) -> Match | None:
        exe = info.exe_path
        if exe and os.path.dirname(exe) == self.installed_dir:
            return Match("executable in the installed-artifact directory", whole_group=info.pgid == info.pid)
        cwd = info.cwd_path
        if cwd and os.path.dirname(cwd) == self.run_root and os.path.basename(cwd).startswith(WORK_PREFIX):
            return Match("working directory is a verifier run directory", whole_group=False, stale=info.directory_deleted or not os.path.isdir(cwd))
        if info.uid == os.getuid():
            run_dir = self.run_directory(read_home(info.pid, root))
            if run_dir:
                return Match("HOME is in a verifier run directory", whole_group=False, stale=not os.path.isdir(run_dir))
        return None

    def candidates(self, root: Path, lineage: set[int]) -> list[tuple[ProcInfo, Match]]:
        """Live processes of this job's identity that a sweep may consider signalling."""
        found = []
        for info in list_procs(root):
            if info.pid in lineage or info.state == "Z" or info.tty_nr != 0:
                continue
            match = self.match(info, root)
            if match is not None:
                found.append((info, match))
        return found


def is_leak(info: ProcInfo, match: Match) -> bool:
    # A run's own processes keep a live directory and a live parent. The runner removes
    # RUNNER_TEMP when a job ends, so a leftover from an earlier job reads "(deleted)" or
    # points at a directory that is gone; a process whose parent died is reparented to init.
    return info.ppid == 1 or info.directory_deleted or info.executable_deleted or match.stale


def group_empty(pgid: int, root: Path = PROC) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        pass
    # The group still exists; it is empty for our purposes if only zombies remain.
    return not any(info.pgid == pgid and info.state != "Z" for info in list_procs(root))


def gone(target: ProcInfo, whole_group: bool, root: Path = PROC) -> bool:
    current = read_proc(target.pid, root)
    if current is not None and current.start_ticks != target.start_ticks:
        # The pid now belongs to another process, so the recorded one, and any group it led,
        # ended: Linux does not reuse a pid while a process group still carries it as its id.
        return True
    leader_gone = current is None or current.state == "Z"
    if not (whole_group and target.pgid == target.pid):
        return leader_gone
    return leader_gone and group_empty(target.pgid, root)


def send(target: ProcInfo, sig: int, whole_group: bool, root: Path = PROC) -> str | None:
    """Signal the target, pinned through a pidfd so a reused pid is never signalled."""
    pidfd = None
    try:
        try:
            pidfd = os.pidfd_open(target.pid)
        except ProcessLookupError:
            pidfd = None
        except (AttributeError, OSError):
            pidfd = None  # no pidfd support; fall back to kill(2) after the same check
        # Checked after opening the pidfd, so the process it pins is the one recorded.
        current = read_proc(target.pid, root)
        if current is not None and current.start_ticks != target.start_ticks:
            return None
        if current is not None and current.state != "Z":
            if pidfd is not None:
                signal.pidfd_send_signal(pidfd, sig)
            else:
                os.kill(target.pid, sig)
        if whole_group and target.pgid == target.pid:
            os.killpg(target.pgid, sig)
    except ProcessLookupError:
        return None
    except PermissionError as exc:
        return str(exc)
    finally:
        if pidfd is not None:
            os.close(pidfd)
    return None


def stop(target: ProcInfo, whole_group: bool, *, term_wait: float = 10.0, kill_wait: float = 5.0, root: Path = PROC) -> dict[str, Any]:
    """SIGTERM the target (and its group when ``whole_group``), then SIGKILL, and report."""
    record: dict[str, Any] = {**target.describe(), "signals": [], "group": whole_group and target.pgid == target.pid}
    for sig, wait in ((signal.SIGTERM, term_wait), (signal.SIGKILL, kill_wait)):
        if gone(target, whole_group, root):
            break
        error = send(target, sig, whole_group, root)
        record["signals"].append(signal.Signals(sig).name)
        if error:
            record["error"] = error
            break
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline and not gone(target, whole_group, root):
            time.sleep(0.2)
    record["stopped"] = gone(target, whole_group, root)
    return record


def runnable_now(root: Path = PROC) -> int:
    """Runnable tasks from /proc/loadavg, less the one reading it.

    The load averages and ``ps %cpu`` both lag: a compile started a second ago reads as
    idle in either. The instantaneous runnable count does not.
    """
    running = (root / "loadavg").read_text().split()[3].split("/")[0]
    return max(0, int(running) - 1)


def wait_for_quiet(
    read_runnable: Callable[[], int],
    *,
    threshold: float,
    window: int,
    timeout: float,
    interval: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Return once the mean runnable count over ``window`` samples is at most ``threshold``.

    Raises GateRefused after ``timeout`` seconds without such a window, and never before the
    window has filled, so a timeout shorter than the window still judges a full one.
    """
    started = clock()
    samples: list[int] = []
    while True:
        samples.append(read_runnable())
        recent = samples[-window:]
        mean = sum(recent) / len(recent)
        waited = clock() - started
        stats = {"threshold": threshold, "window": window, "runnable_mean": round(mean, 2), "recent_runnable": recent, "waited_seconds": round(waited, 1), "samples": len(samples)}
        if len(recent) == window and mean <= threshold:
            return stats
        if waited >= timeout and len(recent) == window:
            raise GateRefused(f"host did not become quiet within {waited:.0f}s: mean runnable tasks {mean:.1f} over the last {window} samples, threshold {threshold:g}", stats)
        sleep(interval)


def loadavg(root: Path = PROC) -> dict[str, Any]:
    parts = (root / "loadavg").read_text().split()
    return {"one": float(parts[0]), "five": float(parts[1]), "fifteen": float(parts[2]), "runnable": parts[3]}


def host_snapshot(root: Path = PROC) -> dict[str, Any]:
    """Load averages, the number of running processes, and compiler pids by name.

    Only well-known compiler names are listed: other command names on a shared host can name
    private work, and this lands in a public repository's logs.
    """
    procs = list_procs(root)
    compilers: dict[str, list[int]] = {}
    for info in procs:
        if info.comm in COMPILERS:
            compilers.setdefault(info.comm, []).append(info.pid)
    return {"observed_at": now(), "loadavg": loadavg(root), "running_count": sum(1 for info in procs if info.state == "R"), "compilers": dict(sorted(compilers.items()))}


def gpu_apps(run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> list[dict[str, Any]]:
    try:
        result = run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60, check=False)
    except FileNotFoundError as exc:
        raise GateRefused("nvidia-smi is not installed; the GPU cannot be checked") from exc
    except subprocess.TimeoutExpired as exc:
        raise GateRefused("nvidia-smi did not answer within 60s; the GPU cannot be checked") from exc
    if result.returncode != 0:
        raise GateRefused(f"nvidia-smi failed with exit {result.returncode}: {result.stderr.strip()[:300]}")
    apps = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 3 or not fields[0].isdigit():
            continue
        used = fields[-1]
        apps.append({"pid": int(fields[0]), "process_name": ",".join(fields[1:-1]), "used_mib": int(used) if used.isdigit() else used})
    return apps


def describe_gpu_app(app: dict[str, Any], identity: Identity, root: Path = PROC) -> dict[str, Any]:
    """Name a GPU process; full paths only when it carries this job's identity."""
    info = read_proc(app["pid"], root)
    if info is None:
        return {"pid": app["pid"], "process_name": os.path.basename(app["process_name"]), "used_mib": app["used_mib"], "proc": "not visible from this user (another user, container or already exited)"}
    match = identity.match(info, root)
    if match is not None:
        return {**app, **info.describe(), "directory_deleted": info.directory_deleted, "this_job": match.reason}
    return {
        "pid": info.pid, "ppid": info.ppid, "uid": info.uid, "comm": info.comm, "used_mib": app["used_mib"],
        "process_name": os.path.basename(app["process_name"]), "directory_deleted": info.directory_deleted,
        "under_run_root": bool(info.cwd_path and info.cwd_path.startswith(identity.run_root + os.sep)), "this_job": None,
    }


def gpu_line(app: dict[str, Any]) -> str:
    parts = [f"pid {app['pid']}", f"{app.get('process_name')}", f"{app.get('used_mib')} MiB"]
    if "proc" in app:
        parts.append(str(app["proc"]))
    elif app.get("this_job"):
        parts += [f"ppid {app['ppid']}", f"cwd {app.get('cwd')}", f"this job's identity ({app['this_job']})"]
    else:
        parts += [f"ppid {app['ppid']}", f"uid {app['uid']}", "not this job's"]
    return ", ".join(parts)


def precheck(args: argparse.Namespace, *, root: Path = PROC, gpu_query: Callable[[], list[dict[str, Any]]] = gpu_apps, read_runnable: Callable[[], int] | None = None, sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> int:
    identity = Identity.from_paths(args.installed_dir, args.run_root)
    state: dict[str, Any] = {"observed_at": now(), "cpu_count": os.cpu_count(), "reaped_leaks": []}
    try:
        for info, match in identity.candidates(root, own_lineage(root)):
            if not is_leak(info, match):
                raise GateRefused(f"a live process of this job's identity is already running ({match.reason}): {json.dumps(info.describe())}")
            if info.uid is not None and info.uid != os.getuid():
                raise GateRefused(f"a leaked process of this job's identity belongs to uid {info.uid} and cannot be stopped: {json.dumps(info.describe())}")
            annotate("warning", f"stopping a process an earlier run leaked ({match.reason}): pid {info.pid}, {info.exe}, ppid {info.ppid}, cwd {info.cwd}")
            record = stop(info, match.whole_group, root=root)
            state["reaped_leaks"].append({**record, "reason": match.reason})
            if not record["stopped"]:
                raise GateRefused(f"could not stop leaked pid {info.pid} ({info.exe}): {record}")
        reaped = {record["pid"] for record in state["reaped_leaks"]}

        def check_gpu() -> None:
            # The driver can list a just-killed process for a few seconds while it tears the context down.
            deadline = clock() + 15
            apps = gpu_query()
            while reaped & {app["pid"] for app in apps} and clock() < deadline:
                sleep(1.0)
                apps = gpu_query()
            state["gpu_processes"] = [describe_gpu_app(app, identity, root) for app in apps]
            if apps:
                raise GateRefused("GPU compute processes are already present before the Activity gate: " + "; ".join(gpu_line(app) for app in state["gpu_processes"]))

        # Before the quiet wait, so a GPU conflict fails at once instead of after ten minutes
        # reported as host load; again after it, for anything that started while waiting.
        state["at_start"] = host_snapshot(root)
        check_gpu()
        quiet = wait_for_quiet(read_runnable or (lambda: runnable_now(root)), threshold=args.max_runnable, window=args.quiet_window, timeout=args.quiet_timeout, sleep=sleep, clock=clock)
        state["quiet"] = quiet
        print(f"host quiet after {quiet['waited_seconds']}s: mean runnable tasks {quiet['runnable_mean']} over {quiet['window']} samples (threshold {quiet['threshold']:g})", flush=True)
        check_gpu()
        return 0
    except GateRefused as exc:
        state["refused"] = exc.message
        annotate("error", exc.message)
        if exc.quiet is not None:
            state["quiet"] = exc.quiet
        return 1
    finally:
        state["at_end"] = host_snapshot(root)
        if "refused" in state and "quiet" in state and "host did not become quiet" in state["refused"]:
            print(f"host at refusal: {json.dumps(state['at_end'])}", flush=True)
        if args.host_state:
            write_atomic(args.host_state, json.dumps(state, indent=2, sort_keys=True) + "\n")


def pid_file_targets(path: Path, root: Path = PROC) -> tuple[list[tuple[str, ProcInfo]], list[dict[str, Any]], bool]:
    """Targets the verifier's pid file still names, report entries, and whether the file was readable."""
    targets: list[tuple[str, ProcInfo]] = []
    report: list[dict[str, Any]] = []
    if not path.exists():
        print(f"no pid file at {path}: the gate did not start its processes", flush=True)
        return targets, report, True
    try:
        records = json.loads(path.read_text())["records"]
        if not isinstance(records, list):
            raise TypeError("records is not a list")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report.append({"pid_file": str(path), "outcome": f"unreadable: {exc}"})
        return targets, report, False
    current_boot = boot_id(root)
    for record in records:
        try:
            label = str(record.get("label"))
            pid, pgid, start = int(record["pid"]), int(record["pgid"]), int(record["start_ticks"])
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            report.append({"record": record, "outcome": f"unusable record ({exc}); left to the identity sweep"})
            continue
        entry = {"label": label, "pid": pid, "pgid": pgid}
        if record.get("boot_id") and current_boot and record["boot_id"] != current_boot:
            report.append({**entry, "outcome": "recorded before the last reboot; not signalled"})
            continue
        current = read_proc(pid, root)
        if current is not None and current.start_ticks == start and current.state != "Z":
            targets.append((f"pid file ({label})", current))
        elif current is not None and current.start_ticks != start:
            report.append({**entry, "outcome": "pid now belongs to another process; not signalled"})
        elif pgid == pid and not group_empty(pgid, root):
            targets.append((f"pid file ({label}) group", ProcInfo(pid, 0, pgid, "?", start, label)))
        else:
            report.append({**entry, "outcome": "already exited"})
    return targets, report, True


def reap(args: argparse.Namespace, *, root: Path = PROC, term_wait: float = 10.0, kill_wait: float = 5.0) -> int:
    identity = Identity.from_paths(args.installed_dir, args.run_root)
    recorded, report, readable = pid_file_targets(args.pid_file, root)
    # Pid-file records are session leaders the verifier started, so their whole group goes.
    targets: list[tuple[str, ProcInfo, bool]] = [(handle, target, True) for handle, target in recorded]
    covered_groups = {target.pgid for _, target, _ in targets}
    covered = {target.pid for _, target, _ in targets}
    for info, match in identity.candidates(root, own_lineage(root)):
        if info.pid not in covered and info.pgid not in covered_groups:
            targets.append((match.reason, info, match.whole_group))
    failed = not readable
    if not readable:
        annotate("error", f"the pid file {args.pid_file} could not be read; only the identity sweep ran")
    for handle, target, whole_group in targets:
        result = {**stop(target, whole_group, term_wait=term_wait, kill_wait=kill_wait, root=root), "handle": handle}
        report.append({**result, "outcome": "stopped" if result["stopped"] else "SURVIVED"})
        if result["stopped"]:
            annotate("warning", f"reaped pid {target.pid} (pgid {target.pgid}, {target.exe or target.comm}) found by {handle}; signals {', '.join(result['signals']) or 'none'}")
        else:
            failed = True
            annotate("error", f"could not reap pid {target.pid} (pgid {target.pgid}, {target.exe or target.comm}) found by {handle}: {result.get('error', 'still running after SIGKILL')}")
    if not targets:
        print("nothing of this job's identity was left running", flush=True)
    print(json.dumps({"reap": report}, indent=2), flush=True)
    return 1 if failed else 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("precheck", "reap"):
        command = sub.add_parser(name)
        command.add_argument("--installed-dir", type=Path, required=True, help="Directory the job copies its installed binaries into")
        command.add_argument("--run-root", type=Path, required=True, help="Parent of the verifier's run directories (the --work-dir it was given)")
    pre = sub.choices["precheck"]
    pre.add_argument("--host-state", type=Path, help="Write the observed host state here as JSON")
    pre.add_argument("--quiet-timeout", type=float, default=600.0)
    pre.add_argument("--quiet-window", type=int, default=10, help="Samples, one per second, whose mean must be at or below --max-runnable")
    pre.add_argument("--max-runnable", type=float, default=max(2.0, (os.cpu_count() or 8) / 4), help="Default: a quarter of the CPUs, at least 2")
    sub.choices["reap"].add_argument("--pid-file", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "precheck" and (args.quiet_window < 1 or args.quiet_timeout < 0):
        parser.error("--quiet-window must be positive and --quiet-timeout non-negative")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    return precheck(args) if args.command == "precheck" else reap(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
