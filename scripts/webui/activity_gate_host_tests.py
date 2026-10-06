#!/usr/bin/env python3
# Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
"""Tests for activity_gate_host.py. The process tests run real processes from a
copy of sleep(1) placed where the CI job puts its installed server binary."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).with_name("activity_gate_host.py")
spec = importlib.util.spec_from_file_location("activity_gate_host", MODULE_PATH)
assert spec and spec.loader
host = importlib.util.module_from_spec(spec)
sys.modules["activity_gate_host"] = host
spec.loader.exec_module(host)

SLEEP = shutil.which("sleep")


def fake_stat(pid: int, comm: str, state: str = "S", ppid: int = 1, pgid: int | None = None, start: int = 4242, tty_nr: int = 0) -> str:
    fields = [state, str(ppid), str(pgid if pgid is not None else pid), "0", str(tty_nr)] + ["0"] * 14 + [str(start), "0", "0"]
    return f"{pid} ({comm}) " + " ".join(fields) + "\n"


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds


@unittest.skipUnless(sys.platform.startswith("linux") and SLEEP, "reads /proc")
class ActivityGateHostTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.installed = self.root / "mlxcel-webui-installed"
        self.installed.mkdir()
        self.server = self.installed / "mlxcel-server-webui"
        shutil.copy2(SLEEP, self.server)
        self.children: list[subprocess.Popen[bytes]] = []

    def tearDown(self) -> None:
        for proc in self.children:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()

    def spawn(self, argv: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> subprocess.Popen[bytes]:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, start_new_session=True)
        self.children.append(proc)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            info = host.read_proc(proc.pid)
            if info and info.exe and os.path.basename(host.strip_deleted(info.exe)) == os.path.basename(argv[0]):
                return proc
            time.sleep(0.02)
        return proc

    def leaked_server(self) -> subprocess.Popen[bytes]:
        """A server whose run directory was removed, as the runner does when a job ends."""
        work = self.root / f"{host.WORK_PREFIX}abc"
        work.mkdir()
        proc = self.spawn([str(self.server), "60"], cwd=work)
        work.rmdir()
        return proc

    def args(self, **extra: object) -> Namespace:
        base = {"installed_dir": self.installed, "run_root": self.root, "host_state": self.root / "host-state.json", "quiet_timeout": 30.0, "quiet_window": 3, "max_runnable": 4.0, "pid_file": self.root / "gate.pid"}
        base.update(extra)
        return Namespace(**base)

    def alive(self, proc: subprocess.Popen[bytes]) -> bool:
        info = host.read_proc(proc.pid)
        return info is not None and info.state != "Z"

    def test_read_proc_parses_comm_with_spaces_and_parentheses(self) -> None:
        root = self.root / "proc"
        (root / "77").mkdir(parents=True)
        (root / "77" / "stat").write_text(fake_stat(77, "a (b) c", state="R", ppid=1, pgid=70, start=99))
        (root / "loadavg").write_text("1.50 2.00 3.25 7/900 1234\n")
        (root / "88").mkdir()
        (root / "88" / "stat").write_text(fake_stat(88, "bash", tty_nr=34816))
        info = host.read_proc(77, root)
        self.assertEqual((info.comm, info.state, info.ppid, info.pgid, info.start_ticks, info.tty_nr), ("a (b) c", "R", 1, 70, 99, 0))
        self.assertEqual(host.read_proc(88, root).tty_nr, 34816)
        (root / "88").rename(root / "88-gone")
        self.assertEqual([p.pid for p in host.list_procs(root)], [77])
        self.assertEqual(host.runnable_now(root), 6)
        self.assertEqual(host.loadavg(root)["fifteen"], 3.25)

    def test_identity_and_leak_classification(self) -> None:
        identity = host.Identity.from_paths(self.installed, self.root)
        real_root = os.path.realpath(self.root)
        (self.root / f"{host.WORK_PREFIX}y").mkdir()
        server = host.ProcInfo(10, 1, 10, "S", 1, "mlxcel-server-w", exe=f"{os.path.realpath(self.server)}{host.DELETED}", cwd="/")
        work = host.ProcInfo(11, 500, 11, "S", 1, "Xvfb", exe="/usr/bin/Xvfb", cwd=f"{real_root}/{host.WORK_PREFIX}x{host.DELETED}")
        other = host.ProcInfo(12, 1, 12, "S", 1, "python3", uid=os.getuid() + 1, exe="/usr/bin/python3", cwd=f"{real_root}/elsewhere")
        live = host.ProcInfo(13, 500, 13, "S", 1, "mlxcel-server-w", exe=os.path.realpath(self.server), cwd=f"{real_root}/{host.WORK_PREFIX}y")
        member = host.ProcInfo(14, 500, 13, "S", 1, "mlxcel-server-w", exe=os.path.realpath(self.server), cwd="/")
        self.assertEqual(identity.match(server), host.Match("executable in the installed-artifact directory", whole_group=True))
        self.assertFalse(identity.match(member).whole_group, "only a group leader's group is signalled")
        work_match = identity.match(work)
        self.assertIn("run directory", work_match.reason)
        self.assertFalse(work_match.whole_group, "a directory match never signals a group")
        self.assertTrue(work_match.stale)
        self.assertIsNone(identity.match(other))
        self.assertTrue(host.is_leak(server, identity.match(server)))
        self.assertTrue(host.is_leak(work, work_match))
        self.assertFalse(host.is_leak(live, identity.match(live)))
        self.assertEqual(identity.run_directory(f"{real_root}/{host.WORK_PREFIX}z/home"), f"{real_root}/{host.WORK_PREFIX}z")
        self.assertIsNone(identity.run_directory("/home/runner"))
        self.assertIsNone(identity.run_directory(real_root))

    def test_candidates_skip_processes_with_a_controlling_terminal(self) -> None:
        root = self.root / "proc"
        for pid, tty in ((50, 0), (51, 34816)):
            (root / str(pid)).mkdir(parents=True)
            (root / str(pid) / "stat").write_text(fake_stat(pid, "mlxcel-server-w", ppid=1, tty_nr=tty))
            (root / str(pid) / "exe").symlink_to(os.path.realpath(self.server))
        identity = host.Identity.from_paths(self.installed, self.root)
        self.assertEqual([info.pid for info, _ in identity.candidates(root, set())], [50])

    def test_gone_treats_a_reused_pid_as_the_end_of_the_recorded_group(self) -> None:
        other = self.spawn([SLEEP, "60"])
        info = host.read_proc(other.pid)
        reused = host.ProcInfo(other.pid, 1, other.pid, "S", info.start_ticks - 1, "old")
        self.assertTrue(host.gone(reused, True))
        self.assertIsNone(host.send(reused, signal.SIGTERM, True))
        self.assertTrue(self.alive(other))

    def test_wait_for_quiet_uses_a_window_mean_and_is_bounded(self) -> None:
        clock = Clock()
        readings = iter([20, 18, 1, 12, 1, 1, 1])
        stats = host.wait_for_quiet(lambda: next(readings), threshold=4, window=3, timeout=60, sleep=clock.sleep, clock=clock)
        self.assertEqual(stats["recent_runnable"], [1, 1, 1])
        self.assertEqual(stats["samples"], 7)
        clock = Clock()
        with self.assertRaises(host.GateRefused) as raised:
            host.wait_for_quiet(lambda: 20, threshold=4, window=3, timeout=5, sleep=clock.sleep, clock=clock)
        self.assertIn("did not become quiet within 5s", raised.exception.message)
        self.assertEqual(raised.exception.quiet["recent_runnable"], [20, 20, 20])
        # A bound shorter than the window still judges a full window, never a partial one.
        clock = Clock()
        readings = iter([1, 30, 1, 1, 1, 1])
        with self.assertRaises(host.GateRefused) as raised:
            host.wait_for_quiet(lambda: next(readings), threshold=4, window=4, timeout=1, sleep=clock.sleep, clock=clock)
        self.assertEqual(raised.exception.quiet["recent_runnable"], [1, 30, 1, 1])
        self.assertIn("within 3s", raised.exception.message)

    def test_gpu_apps_parses_and_fails_closed(self) -> None:
        ok = subprocess.CompletedProcess([], 0, stdout="1768781, /tmp/a,b/mlxcel-server-webui, 4828\n\n", stderr="")
        self.assertEqual(host.gpu_apps(lambda *a, **k: ok), [{"pid": 1768781, "process_name": "/tmp/a,b/mlxcel-server-webui", "used_mib": 4828}])
        with self.assertRaises(host.GateRefused):
            host.gpu_apps(lambda *a, **k: subprocess.CompletedProcess([], 9, stdout="", stderr="no driver"))
        with self.assertRaises(host.GateRefused):
            host.gpu_apps(mock.Mock(side_effect=FileNotFoundError))
        with self.assertRaises(host.GateRefused) as raised:
            host.gpu_apps(mock.Mock(side_effect=subprocess.TimeoutExpired("nvidia-smi", 60)))
        self.assertIn("did not answer", raised.exception.message)

    def test_precheck_stops_a_leaked_server_and_records_host_state(self) -> None:
        leaked = self.leaked_server()
        listings = iter([[{"pid": leaked.pid, "process_name": str(self.server), "used_mib": 4828}], [], []])
        clock = Clock()
        with redirect_stdout(StringIO()) as out:
            rc = host.precheck(self.args(), gpu_query=lambda: next(listings), read_runnable=lambda: 0, sleep=clock.sleep, clock=clock)
        self.assertEqual(rc, 0, out.getvalue())
        self.assertFalse(self.alive(leaked))
        self.assertIn("::warning::stopping a process an earlier run leaked", out.getvalue())
        state = json.loads((self.root / "host-state.json").read_text())
        self.assertEqual([r["pid"] for r in state["reaped_leaks"]], [leaked.pid])
        self.assertEqual(state["reaped_leaks"][0]["signals"], ["SIGTERM"])
        self.assertTrue(state["reaped_leaks"][0]["cwd"].endswith(host.DELETED))
        self.assertEqual(state["gpu_processes"], [])
        self.assertIn("quiet", state)
        self.assertEqual(set(state["at_start"]), {"observed_at", "loadavg", "running_count", "compilers"})
        self.assertIn("loadavg", state["at_end"])

    def test_precheck_names_a_foreign_gpu_process_and_leaves_it_alone(self) -> None:
        private = self.root / "private-project"
        private.mkdir()
        foreign = self.spawn([SLEEP, "60"], cwd=private)
        clock = Clock()
        with redirect_stdout(StringIO()) as out:
            rc = host.precheck(self.args(), gpu_query=lambda: [{"pid": foreign.pid, "process_name": f"{private}/bin/sleep", "used_mib": 512}], read_runnable=lambda: 0, sleep=clock.sleep, clock=clock)
        self.assertEqual(rc, 1)
        self.assertTrue(self.alive(foreign))
        self.assertIn(f"::error::GPU compute processes are already present before the Activity gate: pid {foreign.pid}, sleep, 512 MiB, ppid {os.getpid()}, uid {os.getuid()}, not this job's", out.getvalue())
        # Logs and artifacts of a public repository: no paths of a process that is not this job's.
        self.assertNotIn("private-project", out.getvalue())
        state_text = (self.root / "host-state.json").read_text()
        self.assertNotIn("private-project", state_text)
        state = json.loads(state_text)
        self.assertIsNone(state["gpu_processes"][0]["this_job"])
        self.assertTrue(state["gpu_processes"][0]["under_run_root"])
        self.assertTrue(all(name in host.COMPILERS for name in state["at_end"]["compilers"]))

    def test_precheck_refuses_a_live_process_of_its_own_identity(self) -> None:
        work = self.root / f"{host.WORK_PREFIX}live"
        work.mkdir()
        live = self.spawn([str(self.server), "60"], cwd=work)
        with redirect_stdout(StringIO()) as out:
            rc = host.precheck(self.args(), gpu_query=lambda: self.fail("must refuse before the GPU check"), read_runnable=lambda: 0)
        self.assertEqual(rc, 1)
        self.assertTrue(self.alive(live))
        self.assertIn("a live process of this job's identity is already running", out.getvalue())

    def test_precheck_waits_then_refuses_a_busy_host(self) -> None:
        clock = Clock()
        with redirect_stdout(StringIO()) as out:
            rc = host.precheck(self.args(quiet_timeout=20), gpu_query=lambda: [], read_runnable=lambda: 19, sleep=clock.sleep, clock=clock)
        self.assertEqual(rc, 1)
        self.assertGreaterEqual(clock.t, 20)
        self.assertIn("::error::host did not become quiet within 20s", out.getvalue())
        self.assertIn("host at refusal:", out.getvalue())
        state = json.loads((self.root / "host-state.json").read_text())
        self.assertEqual(state["quiet"]["runnable_mean"], 19)
        self.assertIn("host did not become quiet", state["refused"])

    def test_precheck_fails_a_gpu_conflict_before_waiting_for_quiet(self) -> None:
        foreign = self.spawn([SLEEP, "60"])
        clock = Clock()
        with redirect_stdout(StringIO()) as out:
            rc = host.precheck(self.args(quiet_timeout=600), gpu_query=lambda: [{"pid": foreign.pid, "process_name": "sleep", "used_mib": 1}], read_runnable=lambda: 19, sleep=clock.sleep, clock=clock)
        self.assertEqual(rc, 1)
        self.assertEqual(clock.t, 0, "no quiet wait before a GPU conflict is reported")
        self.assertIn("GPU compute processes are already present", out.getvalue())

    def test_reap_without_a_pid_file_or_survivors_is_a_clean_no_op(self) -> None:
        with redirect_stdout(StringIO()) as out:
            self.assertEqual(host.reap(self.args()), 0)
        self.assertIn("no pid file", out.getvalue())
        self.assertIn("nothing of this job's identity was left running", out.getvalue())

    def test_reap_stops_recorded_and_identified_processes(self) -> None:
        recorded = self.spawn([SLEEP, "60"])
        stubborn = self.spawn(["sh", "-c", f"trap '' TERM; exec {self.server} 60"])
        info = host.read_proc(recorded.pid)
        (self.root / "gate.pid").write_text(json.dumps({"records": [{"label": "xvfb", "pid": recorded.pid, "pgid": recorded.pid, "start_ticks": info.start_ticks}]}))
        with redirect_stdout(StringIO()) as out:
            rc = host.reap(self.args(), term_wait=0.5, kill_wait=3)
        self.assertEqual(rc, 0, out.getvalue())
        self.assertFalse(self.alive(recorded))
        self.assertFalse(self.alive(stubborn))
        report = json.loads(out.getvalue()[out.getvalue().index("{"):])["reap"]
        by_pid = {entry["pid"]: entry for entry in report}
        self.assertEqual(by_pid[recorded.pid]["signals"], ["SIGTERM"])
        self.assertEqual(by_pid[stubborn.pid]["signals"], ["SIGTERM", "SIGKILL"])
        self.assertIn("installed-artifact", by_pid[stubborn.pid]["handle"])

    def test_reap_signals_a_directory_match_by_pid_and_spares_the_rest_of_its_group(self) -> None:
        # A shell sitting in a run directory leads a group that may hold an editor or a pager;
        # only processes that carry the identity themselves are signalled (#2119 review).
        work = self.root / f"{host.WORK_PREFIX}live"
        work.mkdir()
        leader = self.spawn(["sh", "-c", f"(cd / && exec {SLEEP} 61) & exec {SLEEP} 60"], cwd=work)
        member = None
        deadline = time.monotonic() + 5
        while member is None and time.monotonic() < deadline:
            member = next((p for p in host.list_procs() if p.pgid == leader.pid and p.pid != leader.pid and p.cwd == "/"), None)
            time.sleep(0.05)
        self.assertIsNotNone(member)
        with redirect_stdout(StringIO()) as out:
            self.assertEqual(host.reap(self.args(), term_wait=2, kill_wait=2), 0, out.getvalue())
        self.assertFalse(self.alive(leader))
        self.assertIsNotNone(host.read_proc(member.pid), "the group member outside the run directory must survive")
        os.kill(member.pid, signal.SIGKILL)

    def test_browser_processes_are_found_by_their_run_directory_home(self) -> None:
        work = self.root / f"{host.WORK_PREFIX}browser"
        (work / "home").mkdir(parents=True)
        browser = self.spawn([SLEEP, "60"], cwd=Path("/"), env={**os.environ, "HOME": str(work / "home")})
        identity = host.Identity.from_paths(self.installed, self.root)
        match = identity.match(host.read_proc(browser.pid))
        self.assertEqual((match.reason, match.whole_group, match.stale), ("HOME is in a verifier run directory", False, False))
        # Once the run directory is gone the browser is a leak the next precheck stops.
        (work / "home").rmdir()
        work.rmdir()
        clock = Clock()
        with redirect_stdout(StringIO()) as out:
            rc = host.precheck(self.args(), gpu_query=lambda: [], read_runnable=lambda: 0, sleep=clock.sleep, clock=clock)
        self.assertEqual(rc, 0, out.getvalue())
        self.assertFalse(self.alive(browser))

    def test_reap_survives_a_bad_pid_file_and_still_sweeps(self) -> None:
        leaked = self.leaked_server()
        (self.root / "gate.pid").write_text("{not json")
        with redirect_stdout(StringIO()) as out:
            self.assertEqual(host.reap(self.args()), 1)
        self.assertFalse(self.alive(leaked))
        self.assertIn("could not be read; only the identity sweep ran", out.getvalue())
        other = self.spawn([SLEEP, "60"])
        info = host.read_proc(other.pid)
        (self.root / "gate.pid").write_text(json.dumps({"records": [
            {"label": "xvfb", "pid": other.pid, "pgid": other.pid, "start_ticks": None},
            {"label": "openbox", "pid": other.pid, "pgid": other.pid, "start_ticks": info.start_ticks, "boot_id": "not-this-boot"},
        ]}))
        with redirect_stdout(StringIO()) as out:
            self.assertEqual(host.reap(self.args()), 0)
        self.assertTrue(self.alive(other))
        self.assertIn("unusable record", out.getvalue())
        self.assertIn("recorded before the last reboot", out.getvalue())

    def test_reap_does_not_signal_a_reused_pid(self) -> None:
        other = self.spawn([SLEEP, "60"])
        info = host.read_proc(other.pid)
        (self.root / "gate.pid").write_text(json.dumps({"records": [{"label": "mlxcel-server", "pid": other.pid, "pgid": other.pid, "start_ticks": info.start_ticks - 1}]}))
        with redirect_stdout(StringIO()) as out:
            self.assertEqual(host.reap(self.args()), 0)
        self.assertTrue(self.alive(other))
        self.assertIn("pid now belongs to another process", out.getvalue())

    def test_reap_fails_loudly_when_it_cannot_stop_a_process(self) -> None:
        leaked = self.leaked_server()
        with redirect_stdout(StringIO()) as out, mock.patch.object(host, "send", return_value="Operation not permitted"):
            self.assertEqual(host.reap(self.args()), 1)
        self.assertTrue(self.alive(leaked))
        self.assertIn(f"::error::could not reap pid {leaked.pid}", out.getvalue())
        self.assertIn("SURVIVED", out.getvalue())


if __name__ == "__main__":
    unittest.main()
