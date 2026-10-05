#!/usr/bin/env python3
# Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
"""Host-side guards around the WebUI Activity performance gate (issue #1949).

``precheck`` runs inside the gate, after the CI locks and before scoring. It
stops any process an earlier run of this job leaked (a server started from the
installed-artifact directory, or anything working in a verifier run directory,
left orphaned or with its directory deleted), waits a bounded time for the host
to be quiet enough to measure, and then fails closed, naming each process, if
any GPU compute process remains. It writes the host state it observed so the
verifier can put it into the evidence JSON.

``reap`` runs as an ``always()`` step after the gate. It signals what the
verifier recorded in its pid file and anything else carrying this job's
identity, reports what it had to do, and fails if anything survives, so a leak
fails the run that caused it instead of the next one.

Both commands read ``/proc`` and are Linux-only, like the GB10 runner.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
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


def strip_deleted(path: str | None) -> str | None:
    if path is None:
        return None
    return path[: -len(DELETED)] if path.endswith(DELETED) else path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def annotate(level: str, message: str) -> None:
    print(f"::{level}::{message}", flush=True)


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
        state, ppid, pgid, start_ticks = fields[0], int(fields[1]), int(fields[2]), int(fields[19])
    except (IndexError, ValueError):
        return None

    def link(name: str) -> str | None:
        try:
            return os.readlink(base / name)
        except OSError:
            return None

    try:
        uid: int | None = (base).stat().st_uid
    except OSError:
        uid = None
    return ProcInfo(pid, ppid, pgid, state, start_ticks, stat_text[open_paren + 1 : close_paren], uid, link("exe"), link("cwd"))


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

    def reason(self, info: ProcInfo) -> str | None:
        exe = info.exe_path
        if exe and os.path.dirname(exe) == self.installed_dir:
            return "executable in the installed-artifact directory"
        cwd = info.cwd_path
        if cwd and os.path.dirname(cwd) == self.run_root and os.path.basename(cwd).startswith(WORK_PREFIX):
            return "working directory is a verifier run directory"
        return None


def is_leak(info: ProcInfo) -> bool:
    # A run's own processes keep a live directory and a live parent. The runner removes
    # RUNNER_TEMP when a job ends, so a leftover from an earlier job reads "(deleted)"; a
    # process whose parent died is reparented to init.
    return info.ppid == 1 or info.directory_deleted or info.executable_deleted


def group_members(pgid: int, root: Path = PROC) -> list[ProcInfo]:
    return [info for info in list_procs(root) if info.pgid == pgid and info.state != "Z"]


def gone(target: ProcInfo, root: Path = PROC) -> bool:
    current = read_proc(target.pid, root)
    leader_gone = current is None or current.start_ticks != target.start_ticks or current.state == "Z"
    if target.pgid != target.pid:
        return leader_gone
    # Linux does not hand out a pid that is still in use as a process group id, so while
    # any member remains the group is still the one that was recorded.
    return leader_gone and not group_members(target.pgid, root)


def send(target: ProcInfo, sig: int) -> str | None:
    try:
        if target.pgid == target.pid:
            os.killpg(target.pgid, sig)
        else:
            os.kill(target.pid, sig)
    except ProcessLookupError:
        return None
    except PermissionError as exc:
        return str(exc)
    return None


def stop(target: ProcInfo, *, term_wait: float = 10.0, kill_wait: float = 5.0, root: Path = PROC) -> dict[str, Any]:
    """SIGTERM the target (its whole group when it leads one), then SIGKILL, and report."""
    record: dict[str, Any] = {**target.describe(), "signals": [], "group": target.pgid == target.pid}
    for sig, wait in ((signal.SIGTERM, term_wait), (signal.SIGKILL, kill_wait)):
        if gone(target, root):
            break
        error = send(target, sig)
        record["signals"].append(signal.Signals(sig).name)
        if error:
            record["error"] = error
            break
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline and not gone(target, root):
            time.sleep(0.2)
    record["stopped"] = gone(target, root)
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


def busy_processes(procs: list[ProcInfo]) -> dict[str, Any]:
    running = Counter(info.comm for info in procs if info.state == "R")
    compilers = Counter(info.comm for info in procs if info.comm in COMPILERS)
    return {"running": dict(running.most_common(10)), "compilers": dict(compilers.most_common())}


def loadavg(root: Path = PROC) -> dict[str, Any]:
    parts = (root / "loadavg").read_text().split()
    return {"one": float(parts[0]), "five": float(parts[1]), "fifteen": float(parts[2]), "runnable": parts[3]}


def gpu_apps(run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> list[dict[str, Any]]:
    try:
        result = run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60, check=False)
    except FileNotFoundError as exc:
        raise GateRefused("nvidia-smi is not installed; the GPU cannot be checked") from exc
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
    info = read_proc(app["pid"], root)
    if info is None:
        return {**app, "proc": "not visible from this user (another user, container or already exited)"}
    return {**app, **info.describe(), "directory_deleted": info.directory_deleted, "this_job": identity.reason(info)}


def gpu_line(app: dict[str, Any]) -> str:
    parts = [f"pid {app['pid']}", f"{app.get('process_name')}", f"{app.get('used_mib')} MiB"]
    if "ppid" in app:
        parts.append(f"ppid {app['ppid']}")
        parts.append(f"cwd {app.get('cwd')}")
        if app.get("this_job"):
            parts.append(f"this job's identity ({app['this_job']})")
    else:
        parts.append(str(app.get("proc")))
    return ", ".join(parts)


def precheck(args: argparse.Namespace, *, root: Path = PROC, gpu_query: Callable[[], list[dict[str, Any]]] = gpu_apps, read_runnable: Callable[[], int] | None = None, sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> int:
    identity = Identity.from_paths(args.installed_dir, args.run_root)
    state: dict[str, Any] = {"observed_at": now(), "cpu_count": os.cpu_count(), "reaped_leaks": []}
    try:
        lineage = own_lineage(root)
        for info in list_procs(root):
            reason = identity.reason(info)
            if reason is None or info.pid in lineage or info.state == "Z":
                continue
            if not is_leak(info):
                raise GateRefused(f"a live process of this job's identity is already running ({reason}): {json.dumps(info.describe())}")
            if info.uid is not None and info.uid != os.getuid():
                raise GateRefused(f"a leaked process of this job's identity belongs to uid {info.uid} and cannot be stopped: {json.dumps(info.describe())}")
            annotate("warning", f"stopping a process an earlier run leaked ({reason}): pid {info.pid}, {info.exe}, ppid {info.ppid}, cwd {info.cwd}")
            record = stop(info, root=root)
            state["reaped_leaks"].append({**record, "reason": reason})
            if not record["stopped"]:
                raise GateRefused(f"could not stop leaked pid {info.pid} ({info.exe}): {record}")
        reaped = {record["pid"] for record in state["reaped_leaks"]}

        quiet = wait_for_quiet(read_runnable or (lambda: runnable_now(root)), threshold=args.max_runnable, window=args.quiet_window, timeout=args.quiet_timeout, sleep=sleep, clock=clock)
        state["quiet"] = quiet
        print(f"host quiet after {quiet['waited_seconds']}s: mean runnable tasks {quiet['runnable_mean']} over {quiet['window']} samples (threshold {quiet['threshold']:g})", flush=True)

        # The driver can list a just-killed process for a few seconds while it tears the context down.
        deadline = clock() + 15
        apps = gpu_query()
        while reaped & {app["pid"] for app in apps} and clock() < deadline:
            sleep(1.0)
            apps = gpu_query()
        state["gpu_processes"] = [describe_gpu_app(app, identity, root) for app in apps]
        if apps:
            raise GateRefused("GPU compute processes are already present before the Activity gate: " + "; ".join(gpu_line(app) for app in state["gpu_processes"]))
        return 0
    except GateRefused as exc:
        state["refused"] = exc.message
        annotate("error", exc.message)
        if exc.quiet is not None:
            state["quiet"] = exc.quiet
            print(f"busy processes: {json.dumps(busy_processes(list_procs(root)))}", flush=True)
        return 1
    finally:
        state.update({"loadavg": loadavg(root), "busy_processes": busy_processes(list_procs(root))})
        if args.host_state:
            args.host_state.parent.mkdir(parents=True, exist_ok=True)
            args.host_state.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def load_pid_records(path: Path) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return [record for record in data.get("records", []) if isinstance(record, dict)]


def reap(args: argparse.Namespace, *, root: Path = PROC, term_wait: float = 10.0, kill_wait: float = 5.0) -> int:
    identity = Identity.from_paths(args.installed_dir, args.run_root)
    lineage = own_lineage(root)
    targets: list[tuple[str, ProcInfo]] = []
    report: list[dict[str, Any]] = []
    records = load_pid_records(args.pid_file)
    if records is None:
        print(f"no pid file at {args.pid_file}: the gate did not start its processes", flush=True)
    for record in records or []:
        pid, pgid, start = int(record["pid"]), int(record["pgid"]), int(record["start_ticks"])
        current = read_proc(pid, root)
        entry = {"label": record.get("label"), "pid": pid, "pgid": pgid}
        if current is not None and current.start_ticks == start and current.state != "Z":
            targets.append((f"pid file ({record.get('label')})", current))
        elif current is not None and current.state != "Z":
            report.append({**entry, "outcome": "pid now belongs to another process; not signalled"})
        elif pgid == pid and group_members(pgid, root):
            targets.append((f"pid file ({record.get('label')}) group", ProcInfo(pid, 0, pgid, "?", start, str(record.get("label")))))
        else:
            report.append({**entry, "outcome": "already exited"})
    covered = {target.pgid for _, target in targets if target.pgid == target.pid} | {target.pid for _, target in targets}
    for info in list_procs(root):
        reason = identity.reason(info)
        if reason and info.pid not in lineage and info.state != "Z" and info.pid not in covered and info.pgid not in covered:
            targets.append((reason, info))
    failed = False
    for reason, target in targets:
        result = {**stop(target, term_wait=term_wait, kill_wait=kill_wait, root=root), "handle": reason}
        report.append({**result, "outcome": "stopped" if result["stopped"] else "SURVIVED"})
        if result["stopped"]:
            annotate("warning", f"reaped pid {target.pid} (pgid {target.pgid}, {target.exe or target.comm}) found by {reason}; signals {', '.join(result['signals']) or 'none'}")
        else:
            failed = True
            annotate("error", f"could not reap pid {target.pid} (pgid {target.pgid}, {target.exe or target.comm}) found by {reason}: {result.get('error', 'still running after SIGKILL')}")
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
