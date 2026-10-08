# Copyright 2025-2026 Lablup Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the ROCm per-kernel decode profile tooling (issue #2061).

Covers scripts/rocm_decode_profile.py (the decode cut, the busy-time union,
the port-unit attribution) and scripts/rocm_decode_gaps.py (host gaps per
role and the wall-time ceilings, issue #2148) on synthetic traces, and scripts/rocm_gpu_guard.sh
against a fake KFD process directory. No GPU is needed.

Run with:
    python3 -m unittest tests/test_rocm_decode_profile.py
"""

import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GUARD = ROOT / "scripts" / "rocm_gpu_guard.sh"
sys.path.insert(0, str(ROOT / "scripts"))

import rocm_decode_gaps as gaps  # noqa: E402
import rocm_decode_profile as rdp  # noqa: E402

# A compiler pattern nothing on the host matches, so the guard's compiler check
# cannot make these tests depend on what else is running.
NO_COMPILERS = "^no-such-compiler-for-tests$"


# The guard lock of the running test (GuardTests.setUp sets it), so no test
# touches the host-wide lock that real guards on this machine share.
_TEST_LOCK = None


def guard_env(kfd_dir, env_extra=None):
    env = dict(os.environ, ROCM_GPU_GUARD_KFD_DIR=str(kfd_dir),
               ROCM_GPU_GUARD_COMPILER_RE=NO_COMPILERS,
               ROCM_GPU_GUARD_LOCK=str(_TEST_LOCK))
    # A test run under a real guard must still exercise the lock.
    env.pop("ROCM_GPU_GUARD_LOCK_HELD", None)
    env.update(env_extra or {})
    return env


def run_guard(kfd_dir, *args, env_extra=None):
    return subprocess.run(["bash", str(GUARD), *args], env=guard_env(kfd_dir, env_extra),
                          capture_output=True, text=True, timeout=60)


def lock_is_free(path):
    return subprocess.run(["flock", "-n", str(path), "true"]).returncode == 0


@unittest.skipUnless(pathlib.Path("/proc/self/stat").exists(), "rocm_gpu_guard.sh needs Linux /proc")
class GuardTests(unittest.TestCase):
    def setUp(self):
        global _TEST_LOCK
        lock_dir = tempfile.TemporaryDirectory()
        self.addCleanup(lock_dir.cleanup)
        _TEST_LOCK = pathlib.Path(lock_dir.name) / "guard.lock"

    def hold_lock(self):
        """An outside process holding the guard lock until the test ends."""
        _TEST_LOCK.touch()
        # flock's `sleep` child inherits the locked descriptor, so stop the
        # whole process group, not just flock.
        holder = subprocess.Popen(["flock", str(_TEST_LOCK), "sleep", "60"],
                                  start_new_session=True)
        self.addCleanup(holder.wait)
        self.addCleanup(os.killpg, holder.pid, signal.SIGKILL)
        for _ in range(100):
            if not lock_is_free(_TEST_LOCK):
                return holder
            time.sleep(0.05)
        self.fail("outside flock never took the lock")

    def test_idle_gpu_runs_the_command_and_passes_its_status(self):
        with tempfile.TemporaryDirectory() as kfd, tempfile.TemporaryDirectory() as out:
            log = pathlib.Path(out) / "guard.log"
            r = run_guard(kfd, "--idle-secs", "1", "--log", str(log), "--",
                          "bash", "-c", "sleep 1.5; exit 3")
            self.assertEqual(r.returncode, 3, r.stderr)
            text = log.read_text()
            self.assertIn("CLEAN", text)
            self.assertIn("sample 1: clean", text)

    def test_a_foreign_gpu_process_during_the_run_rejects_every_attempt(self):
        with tempfile.TemporaryDirectory() as kfd:
            # The command itself registers a foreign GPU holder (pid 1) once it
            # is running, so the idle wait passes and the monitor must catch it.
            cmd = f"mkdir -p {kfd}/1; sleep 2.5; rmdir {kfd}/1"
            r = run_guard(kfd, "--idle-secs", "1", "--max-attempts", "2", "--",
                          "bash", "-c", cmd)
            self.assertEqual(r.returncode, 75, r.stderr)
            self.assertIn("CONTENDED foreign_gpu=[1:", r.stderr)
            self.assertIn("every attempt (2) was contended", r.stderr)

    def test_the_commands_own_gpu_process_is_not_contention(self):
        with tempfile.TemporaryDirectory() as kfd:
            # $$ of the inner bash is a descendant of the guarded command.
            cmd = f"mkdir -p {kfd}/$$; sleep 2.5; rmdir {kfd}/$$"
            r = run_guard(kfd, "--idle-secs", "1", "--", "bash", "-c", cmd)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertNotIn("CONTENDED", r.stderr)

    def test_a_kfd_entry_left_by_an_exited_process_is_not_contention(self):
        with tempfile.TemporaryDirectory() as kfd:
            # KFD removes a process's proc entry after the process is reaped, so
            # for a moment the entry names a pid with no /proc directory. That
            # is the command's own child finishing, not another GPU tenant.
            cmd = f"true & p=$!; wait $p; mkdir -p {kfd}/$p; sleep 2.5; rmdir {kfd}/$p"
            r = run_guard(kfd, "--idle-secs", "1", "--max-attempts", "1", "--",
                          "bash", "-c", cmd)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertNotIn("CONTENDED", r.stderr)

    def test_a_busy_gpu_before_the_run_times_out_without_running(self):
        with tempfile.TemporaryDirectory() as kfd, tempfile.TemporaryDirectory() as out:
            os.mkdir(pathlib.Path(kfd) / "1")
            marker = pathlib.Path(out) / "ran"
            r = run_guard(kfd, "--idle-secs", "1", "--max-wait", "2", "--",
                          "touch", str(marker))
            self.assertEqual(r.returncode, 75, r.stderr)
            self.assertFalse(marker.exists())

    def test_a_compiler_counts_as_contention(self):
        with tempfile.TemporaryDirectory() as kfd:
            r = run_guard(kfd, "--idle-secs", "1", "--max-wait", "2", "--", "true",
                          env_extra={"ROCM_GPU_GUARD_COMPILER_RE": "^(bash)$"})
            self.assertEqual(r.returncode, 75, r.stderr)
            self.assertIn("gave up after", r.stderr)

    def test_option_values_must_be_plain_non_negative_integers(self):
        with tempfile.TemporaryDirectory() as kfd:
            for flag, bad in (("--idle-secs", "abc"), ("--max-attempts", "-1"),
                              ("--max-wait", "1.5"), ("--idle-secs", "")):
                r = run_guard(kfd, flag, bad, "--", "true")
                self.assertEqual(r.returncode, 2, (flag, bad, r.stderr))
                self.assertIn(f"{flag} needs a non-negative integer", r.stderr)
            r = run_guard(kfd, "--idle-secs")
            self.assertEqual(r.returncode, 2)
            self.assertIn("needs a value", r.stderr)

    def test_a_leading_zero_is_decimal_not_octal(self):
        with tempfile.TemporaryDirectory() as kfd:
            r = run_guard(kfd, "--idle-secs", "08", "--max-wait", "1", "--", "true")
            self.assertEqual(r.returncode, 75, r.stderr)
            self.assertNotIn("value too great", r.stderr)

    def test_sigterm_stops_the_command_and_the_guard(self):
        with tempfile.TemporaryDirectory() as kfd, tempfile.TemporaryDirectory() as out:
            pidfile = pathlib.Path(out) / "cmd.pid"
            env = guard_env(kfd)
            p = subprocess.Popen(
                ["bash", str(GUARD), "--idle-secs", "1", "--", "bash", "-c",
                 f"echo $$ > {pidfile}; exec sleep 30"],
                env=env, stderr=subprocess.PIPE, text=True)
            for _ in range(100):
                if pidfile.exists() and pidfile.read_text().strip():
                    break
                time.sleep(0.1)
            else:
                p.kill()
                self.fail("guarded command never started")
            cmd_pid = int(pidfile.read_text())
            p.terminate()
            _, err = p.communicate(timeout=30)
            self.assertEqual(p.returncode, 143, err)
            self.assertIn("caught SIGTERM", err)
            for _ in range(50):
                if not os.path.exists(f"/proc/{cmd_pid}"):
                    break
                time.sleep(0.1)
            self.assertFalse(os.path.exists(f"/proc/{cmd_pid}"), "command outlived the guard")

    def test_two_guards_started_together_run_one_after_the_other(self):
        with tempfile.TemporaryDirectory() as kfd, tempfile.TemporaryDirectory() as out:
            log = pathlib.Path(out) / "guard.log"
            procs = []
            for name in ("first", "second"):
                # Each command holds the GPU (its own pid in the KFD list) for
                # about 2 s, so a guard running beside it would see it as foreign.
                cmd = f": {name}; mkdir -p {kfd}/$$; sleep 2; rmdir {kfd}/$$"
                procs.append(subprocess.Popen(
                    ["bash", str(GUARD), "--idle-secs", "1", "--max-attempts", "2",
                     "--log", str(log), "--", "bash", "-c", cmd],
                    env=guard_env(kfd), stderr=subprocess.PIPE, text=True))
            errs = [p.communicate(timeout=60)[1] for p in procs]
            for p, err in zip(procs, errs):
                self.assertEqual(p.returncode, 0, err)
                self.assertNotIn("CONTENDED", err)
            # Both guards append to one log, so its order is the order of events:
            # a run starts only after the other one's CLEAN.
            events = [ln for ln in log.read_text().splitlines()
                      if ": start: " in ln or ": CLEAN" in ln]
            self.assertEqual(len(events), 4, events)
            self.assertIn(": start: ", events[0])
            self.assertIn(": CLEAN", events[1])
            self.assertIn(": start: ", events[2])
            self.assertIn(": CLEAN", events[3])
            self.assertTrue(any("waiting for guard lock" in e for e in errs), errs)

    def test_a_held_lock_counts_against_max_wait_and_the_command_never_runs(self):
        self.hold_lock()
        with tempfile.TemporaryDirectory() as kfd, tempfile.TemporaryDirectory() as out:
            marker = pathlib.Path(out) / "ran"
            start = time.monotonic()
            r = run_guard(kfd, "--idle-secs", "1", "--max-wait", "2", "--",
                          "touch", str(marker))
            self.assertEqual(r.returncode, 75, r.stderr)
            self.assertLess(time.monotonic() - start, 10)
            self.assertIn("gave up waiting for guard lock", r.stderr)
            self.assertNotIn("attempt 1/", r.stderr)
            self.assertFalse(marker.exists())

    def test_a_guard_inside_a_guard_does_not_wait_for_the_lock(self):
        self.hold_lock()
        with tempfile.TemporaryDirectory() as kfd:
            # The bash wrapper stands in for an outer guard: it is the inner
            # guard's parent and names itself in ROCM_GPU_GUARD_LOCK_HELD. The
            # trailing `exit` keeps bash from exec'ing the guard in its place.
            inner = (f"ROCM_GPU_GUARD_LOCK_HELD=$$ bash {GUARD} --idle-secs 1 --max-wait 5"
                     " -- true; exit $?")
            r = subprocess.run(["bash", "-c", inner], env=guard_env(kfd),
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("lock held by outer guard", r.stderr)
            self.assertNotIn("waiting for guard lock", r.stderr)

    def test_a_held_variable_naming_a_non_ancestor_does_not_skip_the_lock(self):
        holder = self.hold_lock()
        with tempfile.TemporaryDirectory() as kfd:
            # The lock holder is alive but not this guard's ancestor, like a
            # reused pid or an export leaked by a daemon.
            for held in (str(holder.pid), "1", "self"):
                r = run_guard(kfd, "--idle-secs", "1", "--max-wait", "2", "--", "true",
                              env_extra={"ROCM_GPU_GUARD_LOCK_HELD": held})
                self.assertEqual(r.returncode, 75, (held, r.stderr))
                self.assertIn("gave up waiting for guard lock", r.stderr)

    def test_a_lock_path_that_is_not_a_regular_file_is_refused(self):
        os.mkfifo(_TEST_LOCK)
        with tempfile.TemporaryDirectory() as kfd:
            # Opening a FIFO would block forever, past --max-wait.
            r = run_guard(kfd, "--idle-secs", "1", "--max-wait", "2", "--", "true")
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("is not a regular file", r.stderr)

    def test_sigterm_while_waiting_for_the_lock_stops_the_guard(self):
        self.hold_lock()
        with tempfile.TemporaryDirectory() as kfd, tempfile.TemporaryDirectory() as out:
            marker = pathlib.Path(out) / "ran"
            p = subprocess.Popen(["bash", str(GUARD), "--idle-secs", "1", "--", "touch",
                                  str(marker)],
                                 env=guard_env(kfd), stderr=subprocess.PIPE, text=True)
            self.addCleanup(p.kill)
            # Wait for the guard's background flock to be blocked on the lock.
            for _ in range(100):
                kids = subprocess.run(["pgrep", "-P", str(p.pid), "-x", "flock"],
                                      capture_output=True, text=True).stdout.split()
                if kids:
                    break
                time.sleep(0.05)
            else:
                self.fail("guard never started waiting for the lock")
            p.terminate()
            _, err = p.communicate(timeout=30)
            self.assertEqual(p.returncode, 143, err)
            self.assertIn("caught SIGTERM", err)
            self.assertFalse(marker.exists())
            for _ in range(50):
                if not os.path.exists(f"/proc/{kids[0]}"):
                    break
                time.sleep(0.1)
            self.assertFalse(os.path.exists(f"/proc/{kids[0]}"), "flock outlived the guard")

    def test_a_nested_guard_command_completes(self):
        with tempfile.TemporaryDirectory() as kfd:
            r = run_guard(kfd, "--idle-secs", "1", "--max-wait", "10", "--",
                          "bash", str(GUARD), "--idle-secs", "1", "--max-wait", "10",
                          "--", "true")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("lock held by outer guard", r.stderr)
            self.assertEqual(r.stderr.count("CLEAN"), 2, r.stderr)

    def test_a_daemon_left_by_the_command_does_not_keep_the_lock(self):
        with tempfile.TemporaryDirectory() as kfd, tempfile.TemporaryDirectory() as out:
            pidfile = pathlib.Path(out) / "daemon.pid"
            cmd = f"sleep 30 </dev/null >/dev/null 2>&1 & echo $! > {pidfile}"
            r = run_guard(kfd, "--idle-secs", "1", "--", "bash", "-c", cmd)
            daemon = int(pidfile.read_text())
            try:
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertTrue(os.path.exists(f"/proc/{daemon}"))
                self.assertTrue(lock_is_free(_TEST_LOCK), "the daemon holds the guard lock")
            finally:
                os.kill(daemon, 9)

    # --hold and --status (issue #2244)

    def start_hold(self, kfd, *cmd, max_wait=None):
        args = ["--hold"] + (["--max-wait", str(max_wait)] if max_wait else []) + ["--", *cmd]
        p = subprocess.Popen(["bash", str(GUARD), *args], env=guard_env(kfd),
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(p.communicate)
        self.addCleanup(p.kill)
        for _ in range(100):
            if not lock_is_free(_TEST_LOCK):
                return p
            time.sleep(0.05)
        self.fail("the --hold guard never took the lock")

    def test_hold_runs_nested_guards_without_taking_the_lock(self):
        with tempfile.TemporaryDirectory() as kfd, tempfile.TemporaryDirectory() as out:
            log = pathlib.Path(out) / "inner.log"
            inner = (f"bash {GUARD} --idle-secs 1 --max-wait 10 --log {log} -- true; "
                     f"bash {GUARD} --idle-secs 1 --max-wait 10 --log {log} -- bash -c 'exit 4'; "
                     'test "$ROCM_GPU_GUARD_LOCK_HELD" = "$PPID" || exit 9; '
                     f"flock -n {_TEST_LOCK} true && exit 8; exit 3")
            r = run_guard(kfd, "--hold", "--", "bash", "-c", inner)
            # The inner guards ran their own idle wait and monitor (a CLEAN
            # attempt each), took no lock, and the lock stayed held throughout
            # (exit 8 would mean it was free); the command's status comes back.
            self.assertEqual(r.returncode, 3, r.stderr)
            self.assertEqual(r.stderr.count("lock held by outer guard"), 2, r.stderr)
            self.assertNotIn("waiting for guard lock", r.stderr)
            self.assertEqual(log.read_text().count("CLEAN, exit"), 2, log.read_text())
            self.assertIn("exit 4", log.read_text())
            self.assertTrue(lock_is_free(_TEST_LOCK))

    def test_hold_does_no_idle_wait_even_with_a_busy_gpu(self):
        with tempfile.TemporaryDirectory() as kfd:
            os.mkdir(f"{kfd}/1")
            r = run_guard(kfd, "--hold", "--", "true")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertNotIn("waiting for", r.stderr.replace("waiting for guard lock", ""))
            self.assertNotIn("attempt", r.stderr)

    def test_a_second_hold_waits_for_the_first_and_exits_75_on_max_wait(self):
        with tempfile.TemporaryDirectory() as kfd, tempfile.TemporaryDirectory() as out:
            first = self.start_hold(kfd, "sleep", "8")
            marker = pathlib.Path(out) / "ran"
            start = time.monotonic()
            r = run_guard(kfd, "--hold", "--max-wait", "2", "--", "touch", str(marker))
            self.assertEqual(r.returncode, 75, r.stderr)
            self.assertLess(time.monotonic() - start, 7)
            self.assertIn("waiting for guard lock", r.stderr)
            self.assertIn("gave up waiting for guard lock", r.stderr)
            self.assertFalse(marker.exists())
            self.assertIsNone(first.poll(), "the first hold was disturbed")

    def test_status_reports_held_with_the_holder_then_free(self):
        with tempfile.TemporaryDirectory() as kfd:
            r = run_guard(kfd, "--status")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("free", r.stdout)
            holder = self.start_hold(kfd, "sleep", "8")
            for _ in range(100):
                if pathlib.Path(f"{_TEST_LOCK}.holder").read_text():
                    break
                time.sleep(0.05)
            r = run_guard(kfd, "--status")
            self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
            self.assertIn("held", r.stdout)
            self.assertIn(f"pid={holder.pid}", r.stdout)
            self.assertIn("mode=hold", r.stdout)
            self.assertIn("sleep", r.stdout)
            self.assertNotIn("stale", r.stdout)
            holder.terminate()
            holder.communicate(timeout=30)
            self.assertEqual(holder.returncode, 143)
            r = run_guard(kfd, "--status")
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("free", r.stdout)
            self.assertFalse(pathlib.Path(f"{_TEST_LOCK}.holder").exists())

    def test_a_plain_guard_records_and_clears_its_holder_file(self):
        with tempfile.TemporaryDirectory() as kfd:
            cmd = f"cat {_TEST_LOCK}.holder; exit 2"
            r = run_guard(kfd, "--idle-secs", "1", "--", "bash", "-c", cmd)
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("mode=guard", r.stdout)
            self.assertFalse(pathlib.Path(f"{_TEST_LOCK}.holder").exists())

    def test_status_calls_a_holder_file_whose_pid_is_gone_stale(self):
        gone = subprocess.Popen(["true"])
        gone.wait()
        self.hold_lock()
        with tempfile.TemporaryDirectory() as kfd:
            pathlib.Path(f"{_TEST_LOCK}.holder").write_text(
                f"pid={gone.pid}\nstart=then\nmode=hold\ncmd=old\n")
            r = run_guard(kfd, "--status")
            self.assertEqual(r.returncode, 1, r.stdout)
            self.assertIn("stale", r.stdout)

    def test_status_with_a_lock_nobody_recorded_still_says_held(self):
        self.hold_lock()
        with tempfile.TemporaryDirectory() as kfd:
            r = run_guard(kfd, "--status")
            self.assertEqual(r.returncode, 1, r.stdout)
            self.assertIn("held", r.stdout)
            self.assertIn("no holder file", r.stdout)

    def test_hold_refuses_the_idle_options_and_status_refuses_everything(self):
        with tempfile.TemporaryDirectory() as kfd:
            for extra in (("--idle-secs", "5"), ("--max-attempts", "2"), ("--log", "/dev/null")):
                r = run_guard(kfd, "--hold", *extra, "--", "true")
                self.assertEqual(r.returncode, 2, (extra, r.stderr))
                self.assertIn("--hold takes only --max-wait", r.stderr)
            self.assertEqual(run_guard(kfd, "--status", "--hold").returncode, 2)
            self.assertEqual(run_guard(kfd, "--status", "--max-wait", "1").returncode, 2)
            self.assertEqual(run_guard(kfd, "--status", "--", "true").returncode, 2)
            self.assertEqual(run_guard(kfd, "--hold").returncode, 2)

    def test_hold_under_an_outer_guard_just_runs_the_command(self):
        self.hold_lock()
        with tempfile.TemporaryDirectory() as kfd:
            inner = f"ROCM_GPU_GUARD_LOCK_HELD=$$ bash {GUARD} --hold -- bash -c 'exit 6'; exit $?"
            r = subprocess.run(["bash", "-c", inner], env=guard_env(kfd),
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 6, r.stderr)
            self.assertIn("lock held by outer guard", r.stderr)
            self.assertNotIn("waiting for guard lock", r.stderr)

    def test_sigterm_stops_the_held_command_and_releases_the_lock(self):
        with tempfile.TemporaryDirectory() as kfd, tempfile.TemporaryDirectory() as out:
            marker = pathlib.Path(out) / "survived"
            p = self.start_hold(kfd, "bash", "-c", f"sleep 20; touch {marker}")
            p.terminate()
            p.communicate(timeout=30)
            self.assertEqual(p.returncode, 143)
            self.assertTrue(lock_is_free(_TEST_LOCK))
            self.assertFalse(marker.exists())

    def test_hold_passes_stdin_to_the_command(self):
        with tempfile.TemporaryDirectory() as kfd:
            r = subprocess.run(["bash", str(GUARD), "--hold", "--", "cat"],
                               env=guard_env(kfd), input="piped\n",
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout, "piped\n")


def k(op: str) -> str:
    """A kernel name spelled the way rocprofv3 writes the overlay's kernels."""
    names = {
        "qmv": "void mlx::core::rocm::qmv_wide_kernel<__half, __half, 64, 4>(__half const*)",
        "gqmv": "void mlx::core::rocm::gather_qmv_wide_kernel<hip_bfloat16, hip_bfloat16, 64>(x)",
        "rms": "void mlx::core::rocm::rms_norm_kernel<__half, 256, 4>(__half const*)",
        "add": "void mlx::core::rocm::binary_vv<mlx::core::rocm::Add, __half, __half, unsigned int, 4>(x)",
        "mul": "void mlx::core::rocm::binary_g<mlx::core::rocm::Multiply, float, float, long, 1>(x)",
        "div": "void mlx::core::rocm::binary_vs<mlx::core::rocm::Divide, float, float, unsigned int, 4>(x)",
        "rope": "void mlx::core::rocm::rope_single_1d<__half, false, true>(x)",
        "kv": "void mlx::core::rocm::copy_gg_byval<__half, __half, long>(x)",
        "cast": "void mlx::core::rocm::copy_v<__half, float, unsigned int, 4>(x)",
        "sdpa": "void mlx::core::rocm::kernel_sdpav_1pass<__half, false, 128>(x)",
        "swiglu": "void mlx::core::rocm::CV2ISigmoidADV2IMultiplyGH_VV_V2V2_614_contiguous<unsigned int, 8>(x)",
        "silu": "void mlx::core::rocm::BV2ISigmoidACV2OMultiplyCD_V_V2_614_contiguous<unsigned int, 16>(x)",
        "argmax": "void mlx::core::rocm::arg_reduce_final<__half, mlx::core::rocm::ArgMax<__half>, 256>(x)",
        "embed": "void mlx::core::rocm::gather_rows_kernel<unsigned int, int>(x)",
        "sort": "void mlx::core::rocm::block_sort_kernel<hip_bfloat16, unsigned int, true, 32, 8>(x)",
        "arange": "void mlx::core::rocm::arange_kernel<unsigned int>(unsigned int*)",
        "colsum": "void mlx::core::rocm::col_reduce_small<float, float, mlx::core::rocm::Sum, 4>(x)",
        "conv": "void mlx::core::(anonymous namespace)::depthwise_conv1d_kernel<hip_bfloat16>(x)",
        "exp": "void mlx::core::rocm::unary_v<mlx::core::rocm::Exp, float, float, unsigned int, 4>(x)",
        "scan": "void mlx::core::rocm::contiguous_scan<float, float, mlx::core::rocm::Sum, 4, true, false>(x)",
    }
    return names[op]


def seq(*ops, grid=None):
    grid = grid or {}
    out, t = [], 1000
    for op in ops:
        out.append(rdp.Dispatch(k(op.split(":")[0]), t, t + 10, int(op.split(":")[1]) if ":" in op else 256))
        t += 20
    return out


class RoleTests(unittest.TestCase):
    def test_dense_layer_and_sampler_tail(self):
        d = seq("embed", "rms", "qmv:1000", "rope", "kv", "kv", "rope", "sdpa", "qmv:1000",
                "add", "rms", "qmv:3000", "qmv:3000", "swiglu", "qmv:1000", "add", "rms",
                "qmv:90000", "cast", "argmax", "embed")
        roles = rdp.assign_roles(d)
        self.assertEqual(roles[3:7], ["rope_append"] * 4)
        self.assertEqual(roles[9:11], ["add_rms_join_post_attn"] * 2)
        self.assertEqual(roles[15:17], ["add_rms_join"] * 2)
        self.assertEqual(roles[18:20], ["sampler_tail"] * 2)
        self.assertIsNone(roles[17])  # the lm_head GEMV itself
        self.assertIsNone(roles[20])  # the next step's embedding

    def test_moe_block(self):
        d = seq("add", "rms", "qmv:64", "sort", "arange", "arange", "div", "gqmv", "gqmv",
                "swiglu", "gqmv", "mul", "cast", "colsum", "add", "rms", "qmv:90000", "argmax")
        roles = rdp.assign_roles(d)
        self.assertEqual([roles[i] for i in (7, 8, 10)], ["moe_expert_gemv"] * 3)
        self.assertEqual(roles[9], "moe_activation")
        self.assertEqual(roles[11:14], ["moe_weighted_sum"] * 3)
        self.assertEqual(roles[4:6], ["moe_gather_indices"] * 2)
        self.assertIsNone(roles[3])   # router top-k stays outside the fused kernel
        self.assertIsNone(roles[6])   # score normalisation too
        self.assertEqual(roles[14:16], ["add_rms_join"] * 2)

    def test_mamba_mixer(self):
        d = seq("rms", "qmv:200000", "cast", "kv", "conv", "exp", "mul", "silu", "scan",
                "mul", "silu", "rms", "mul", "qmv:50000", "qmv:900000", "argmax")
        roles = rdp.assign_roles(d)
        self.assertEqual(roles[2], "ssm_step")
        self.assertEqual(roles[3:5], ["ssm_conv"] * 2)
        self.assertEqual([roles[i] for i in (5, 6, 8, 9)], ["ssm_step"] * 4)
        self.assertEqual([roles[i] for i in (7, 10)], ["ssm_silu"] * 2)
        self.assertEqual(roles[11:13], ["ssm_gated_norm"] * 2)
        self.assertIsNone(roles[1])
        self.assertIsNone(roles[13])


class ReachTests(unittest.TestCase):
    def test_fused_norm_and_rope_ship_on_for_rocm_and_rope_tables_never_reach(self):
        # #2145: both fusions are on by default in a rocm build, so the roles
        # they take over are reached with the shipped defaults.
        default, optin, _ = rdp.reach("2063", {"model_type": "llama",
                                               "rope_scaling": {"rope_type": "llama3"}}, False)
        self.assertEqual(default, ("add_rms_join_post_attn",))
        self.assertEqual(optin, ("add_rms_join_post_attn",))
        default, optin, _ = rdp.reach("2063", {"model_type": "llama"}, False)
        self.assertEqual(default, ("add_rms_join_post_attn", "rope_append"))
        self.assertEqual(optin, ("add_rms_join_post_attn", "rope_append"))

    def test_moe_reach_follows_the_caller(self):
        self.assertTrue(rdp.reach("2065", {"model_type": "qwen3_moe"}, False)[0])
        self.assertFalse(rdp.reach("2065", {"model_type": "granitemoehybrid"}, False)[0])
        self.assertFalse(rdp.reach("2065", {"model_type": "nemotron_h"}, False)[0])

    def test_samplers_only_in_sampled_runs(self):
        self.assertFalse(rdp.reach("2064", {}, False)[0])
        self.assertTrue(rdp.reach("2064", {}, True)[0])


class WindowTests(unittest.TestCase):
    def test_busy_is_the_union_of_overlapping_dispatches(self):
        d = [rdp.Dispatch("a", 0, 10), rdp.Dispatch("b", 5, 15), rdp.Dispatch("c", 20, 30)]
        self.assertEqual(rdp.busy_ns(d, 0, 100), 25)
        self.assertEqual(rdp.busy_ns(d, 8, 25), 12)

    def test_dispatch_is_slotted(self):
        d = rdp.Dispatch("a", 1, 4)
        self.assertFalse(hasattr(d, "__dict__"))
        self.assertEqual(d.dur, 3)

    def test_summarize_cuts_the_decode_window_by_the_phase_marks(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            trace = tmp / "t_kernel_trace.csv"
            rows = [("warm", 100_000, 110_000), ("prefill", 500_000, 590_000),
                    ("decode", 700_000, 710_000), ("decode", 720_000, 760_000),
                    ("after", 5_000_000, 5_010_000)]
            trace.write_text('"Kind","Kernel_Name","Start_Timestamp","End_Timestamp"\n' + "".join(
                f'"KERNEL_DISPATCH","void mlx::core::rocm::{n}_kernel(x)",{s},{e}\n'
                for n, s, e in rows))
            log = tmp / "bench.log"
            log.write_text(
                "[phase] warmup_start monotonic_ns=1 boottime_ns=90000\n"
                "[phase] measured_start monotonic_ns=2 boottime_ns=400000\n"
                "[phase] decode_start monotonic_ns=3 boottime_ns=600000\n"
                "[phase] measured_end monotonic_ns=4 boottime_ns=800000\n"
                "  Prompt tokens:    512\n  Generated tokens: 2\n"
                "  Prefill:          1.00 ms (512.00 tok/s)\n  Decode:           0.20 ms (10000.00 tok/s)\n")
            plain = tmp / "plain.log"
            plain.write_text("  Generated tokens: 2\n"
                             "  Decode:           0.20 ms (8000.00 tok/s)\n")
            s = rdp.summarize(trace, log, plain, None, "t", tmp)
            self.assertEqual(s["clock"], "boottime")
            self.assertEqual(s["decode_dispatches"], 2)
            self.assertEqual(s["decode_gpu_busy_ms"], 0.05)
            self.assertEqual(s["checks"]["kernels_straddling_decode_start"], 0)
            self.assertEqual(s["checks"]["idle_gap_before_first_decode_dispatch_us"], 110.0)
            self.assertTrue((tmp / "t_decode_kernels.csv").exists())
            # #2148: the window's 150 us of idle time, per role, adds up.
            self.assertEqual(s["checks"]["host_gap_attribution_residual_ns"], 0)
            self.assertEqual(s["role_host_gap_ms_per_token"],
                             {"unattributed": 0.055, "tail": 0.02})
            self.assertEqual(s["plain_wall_ms_per_token"], 0.125)
            self.assertEqual(s["plain_host_gap_ms_per_token_est"], 0.1)
            for unit in rdp.UNIT_ROLES:
                p = s["port_units"][unit]
                # No dispatch of any unit's roles: no gap, nothing to gain.
                for prefix in ("fallback", "reached_default"):
                    self.assertEqual(p[f"{prefix}_host_gap_ms_per_token"], 0.0)
                    self.assertEqual(p[f"{prefix}_host_gap_us_per_dispatch"], 0.0)
                    self.assertEqual(p[f"{prefix}_wall_share_pct"], 0.0)
                for key in ("ceiling_gpu_share", "ceiling_wall", "plain_ceiling_wall_est",
                            "reached_default_ceiling_gpu_share", "reached_default_ceiling_wall",
                            "reached_default_plain_ceiling_wall_est"):
                    self.assertEqual(p[key], 1.0, (unit, key))
            text = rdp.report(tmp)
            self.assertIn("| Run: ceiling wall (GPU share) |", text)
            self.assertIn("| `t` (plain) | 1.00 (1.00) |", text)


def disp(start, end, name="k"):
    return rdp.Dispatch(name, start, end)


class GapTests(unittest.TestCase):
    def test_gaps_go_to_the_dispatch_that_ends_them_and_sum_to_the_idle_time(self):
        # A leading gap, an overlap, a dispatch inside another, an idle tail.
        d = [disp(10, 20), disp(15, 30), disp(40, 50), disp(45, 48), disp(60, 70)]
        roles = ["ssm_step", "rope_append", None, "ssm_step", "rope_append"]
        gaps = rdp.attribute_gaps(d, roles, 0, 100)
        self.assertEqual(gaps, {"ssm_step": 10, "rope_append": 10, "unattributed": 10,
                                "tail": 30})
        self.assertEqual(sum(gaps.values()), 100 - rdp.busy_ns(d, 0, 100))

    def test_a_dispatch_running_past_the_window_leaves_no_tail(self):
        d = [disp(5, 10), disp(95, 130)]
        gaps = rdp.attribute_gaps(d, ["sampler_tail", "ssm_step"], 0, 100)
        self.assertEqual(gaps, {"sampler_tail": 5, "ssm_step": 85, "tail": 0})
        self.assertEqual(sum(gaps.values()), 100 - rdp.busy_ns(d, 0, 100))

    def test_role_count_must_match(self):
        with self.assertRaises(ValueError):
            rdp.attribute_gaps([disp(1, 2)], [], 0, 10)

    def test_wall_ceiling_counts_the_launch_gaps(self):
        # 3 ms of a unit's kernels and 2 ms of gaps in front of them, in a 20 ms
        # window with 10 ms of kernels; the plain run keeps half of each gap and
        # takes 12 ms.
        f = gaps.wall_fields("fallback", ("ssm_step",), {"ssm_step": 3_000_000},
                             {"ssm_step": 2_000_000}, {"ssm_step": 4},
                             gpu_sum=10_000_000, wall=20_000_000, tokens=1, scale=0.5,
                             plain_wall_ms_per_token=12.0)
        self.assertEqual(f, {
            "fallback_host_gap_ms_per_token": 2.0,
            "fallback_host_gap_us_per_dispatch": 500.0,
            "fallback_wall_share_pct": 25.0,
            "ceiling_gpu_share": 1.43,   # 1 / (1 - 0.30)
            "ceiling_wall": 1.33,        # 1 / (1 - 0.25)
            "plain_ceiling_wall_est": 1.5,  # 1 / (1 - (3 + 1) / 12)
        })
        r = gaps.wall_fields("reached_default", (), {}, {}, {}, gpu_sum=1, wall=1,
                             tokens=1, scale=None, plain_wall_ms_per_token=None)
        self.assertEqual(r["reached_default_ceiling_wall"], 1.0)
        self.assertIsNone(r["reached_default_plain_ceiling_wall_est"])

    def test_a_full_share_and_zero_tokens_have_no_ceiling_or_per_token_figure(self):
        f = gaps.wall_fields("fallback", ("ssm_step",), {"ssm_step": 10}, {"ssm_step": 0},
                             {"ssm_step": 1}, gpu_sum=10, wall=10, tokens=0, scale=1.0,
                             plain_wall_ms_per_token=1.0)
        self.assertIsNone(f["ceiling_gpu_share"])
        self.assertIsNone(f["ceiling_wall"])
        self.assertIsNone(f["fallback_host_gap_ms_per_token"])
        self.assertIsNone(f["plain_ceiling_wall_est"])
        self.assertIsNone(gaps.ceiling(None))

    def test_gap_scale(self):
        self.assertIsNone(gaps.gap_scale(None, 4.0))
        self.assertEqual(gaps.gap_scale(2.0, 4.0), 0.5)
        self.assertEqual(gaps.gap_scale(-1.0, 4.0), 0.0)
        self.assertEqual(gaps.gap_scale(1.0, 0.0), 0.0)

    def test_report_cell_prefers_the_plain_estimate(self):
        self.assertEqual(gaps.ceiling_cell({"ceiling_wall": 1.6, "plain_ceiling_wall_est": 1.5,
                                            "ceiling_gpu_share": 1.42}), "1.50 (1.42)")
        self.assertEqual(gaps.ceiling_cell({"ceiling_wall": 1.6, "plain_ceiling_wall_est": None,
                                            "ceiling_gpu_share": None}), "1.60 (-)")
        self.assertEqual(gaps.ceiling_cell({"fallback_share_pct": 3.0}), "-")


if __name__ == "__main__":
    unittest.main()
