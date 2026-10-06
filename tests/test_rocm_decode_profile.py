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
the port-unit attribution) on synthetic traces, and scripts/rocm_gpu_guard.sh
against a fake KFD process directory. No GPU is needed.

Run with:
    python3 -m unittest tests/test_rocm_decode_profile.py
"""

import os
import pathlib
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GUARD = ROOT / "scripts" / "rocm_gpu_guard.sh"
sys.path.insert(0, str(ROOT / "scripts"))

import rocm_decode_profile as rdp  # noqa: E402

# A compiler pattern nothing on the host matches, so the guard's compiler check
# cannot make these tests depend on what else is running.
NO_COMPILERS = "^no-such-compiler-for-tests$"


def run_guard(kfd_dir, *args, env_extra=None):
    env = dict(os.environ, ROCM_GPU_GUARD_KFD_DIR=str(kfd_dir),
               ROCM_GPU_GUARD_COMPILER_RE=NO_COMPILERS)
    env.update(env_extra or {})
    return subprocess.run(["bash", str(GUARD), *args], env=env, capture_output=True,
                          text=True, timeout=60)


class GuardTests(unittest.TestCase):
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
            env = dict(os.environ, ROCM_GPU_GUARD_KFD_DIR=kfd,
                       ROCM_GPU_GUARD_COMPILER_RE=NO_COMPILERS)
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
    def test_fused_norm_and_rope_ship_off_and_rope_tables_never_reach(self):
        default, optin, _ = rdp.reach("2063", {"model_type": "llama",
                                               "rope_scaling": {"rope_type": "llama3"}}, False)
        self.assertEqual(default, ())
        self.assertEqual(optin, ("add_rms_join_post_attn",))
        _, optin, _ = rdp.reach("2063", {"model_type": "llama"}, False)
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
            s = rdp.summarize(trace, log, None, None, "t", tmp)
            self.assertEqual(s["clock"], "boottime")
            self.assertEqual(s["decode_dispatches"], 2)
            self.assertEqual(s["decode_gpu_busy_ms"], 0.05)
            self.assertEqual(s["checks"]["kernels_straddling_decode_start"], 0)
            self.assertEqual(s["checks"]["idle_gap_before_first_decode_dispatch_us"], 110.0)
            self.assertTrue((tmp / "t_decode_kernels.csv").exists())


if __name__ == "__main__":
    unittest.main()
