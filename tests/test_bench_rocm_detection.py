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

"""Tests for the ROCm host handling in the benchmark harness (issue #1810).

Covers the probe-output parsers and hardware tagging in scripts/bench_decode.sh,
the matching tag in scripts/bench_mlxlm.py, and the host/runtime detection in
scripts/compare_bench_csv.py. No GPU is needed: the shell functions are
extracted and fed recorded output.

Run with:
    python3 -m unittest tests/test_bench_rocm_detection.py
"""

import csv
import os
import pathlib
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "bench_decode.sh"
sys.path.insert(0, str(ROOT / "scripts"))

import bench_mlxlm  # noqa: E402
import compare_bench_csv  # noqa: E402

# What `MLXCEL_DEBUG_KERNEL_BACKEND=1 mlxcel generate` prints on the gfx1151
# host, stdout and stderr interleaved as the harness captures them.
ROCM_PROBE_OUTPUT = """\
Runtime device: GPU
GPU memory: 96.0 GB (no wired limit)
Loading model from "models/mlx/Qwen3-0.6B-4bit"...
Detected 1 GPU(s).
HIP architecture gfx1151; compiled for [gfx1151]
GPU: AMD Radeon 8060S Graphics (Amd), 96.00 GiB device memory.
[mlxcel] custom kernel backend: rocm
Generating...
"""

# Trimmed `rocminfo` from the same host: the CPU agent comes first and the NPU
# last, so the parser has to pick the agent by its device type.
ROCMINFO_OUTPUT = """\
*******
Agent 1
*******
  Name:                    AMD RYZEN AI MAX+ 395 w/ Radeon 8060S
  Marketing Name:          AMD RYZEN AI MAX+ 395 w/ Radeon 8060S
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx1151
  Marketing Name:          AMD Radeon 8060S Graphics
  Device Type:             GPU
*******
Agent 3
*******
  Name:                    aie2p
  Marketing Name:          RyzenAI-npu5
  Device Type:             DSP
"""


def _extract(*names: str) -> str:
    """Shell source of the named top-level functions of bench_decode.sh."""
    parts = []
    for name in names:
        program = r"/^" + name + r"\(\) \{/{f=1} f{print} f && /^\}$/{exit}"
        body = subprocess.run(
            ["awk", program, str(SCRIPT)], capture_output=True, text=True, check=True
        ).stdout
        if not body.strip():
            raise AssertionError(f"function {name} not found in {SCRIPT}")
        parts.append(body)
    return "\n".join(parts)


def _run(functions, script: str, stdin: str = "") -> str:
    cmd = "set -euo pipefail\n" + _extract(*functions) + "\n" + script
    result = subprocess.run(["bash", "-c", cmd], input=stdin, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"bash failed ({result.returncode}): {result.stderr}")
    return result.stdout.strip()


class ProbeParserTests(unittest.TestCase):
    def test_kernel_backend(self) -> None:
        self.assertEqual(_run(["probe_kernel_backend"], "probe_kernel_backend", ROCM_PROBE_OUTPUT), "rocm")

    def test_kernel_backend_with_fallback_suffix(self) -> None:
        line = "[mlxcel] custom kernel backend: rocm (not every kernel family is ported; the rest use MLX graph fallbacks)\n"
        self.assertEqual(_run(["probe_kernel_backend"], "probe_kernel_backend", line), "rocm")

    def test_gfx_target(self) -> None:
        self.assertEqual(_run(["probe_gfx_target"], "probe_gfx_target", ROCM_PROBE_OUTPUT), "gfx1151")

    def test_gpu_name(self) -> None:
        self.assertEqual(
            _run(["probe_gpu_name"], "probe_gpu_name", ROCM_PROBE_OUTPUT), "AMD Radeon 8060S Graphics"
        )

    def test_gpu_name_without_memory(self) -> None:
        # generate.rs omits the memory clause when the device reports 0 bytes.
        self.assertEqual(_run(["probe_gpu_name"], "probe_gpu_name", "GPU: Foo Bar (Amd).\n"), "Foo Bar")

    def test_device_memory_bytes(self) -> None:
        got = _run(["probe_device_memory_bytes"], "probe_device_memory_bytes", ROCM_PROBE_OUTPUT)
        self.assertEqual(int(got), 96 * 1024**3)

    def test_parsers_are_silent_on_other_backends(self) -> None:
        metal = "Runtime device: GPU\nDetected 1 GPU(s).\n[mlxcel] custom kernel backend: metal\n"
        self.assertEqual(_run(["probe_gfx_target"], "probe_gfx_target", metal), "")
        self.assertEqual(_run(["probe_device_memory_bytes"], "probe_device_memory_bytes", metal), "")
        self.assertEqual(_run(["probe_kernel_backend"], "probe_kernel_backend", metal), "metal")

    def test_rocminfo_picks_the_gpu_agent(self) -> None:
        self.assertEqual(
            _run(["rocminfo_first_gpu"], "rocminfo_first_gpu", ROCMINFO_OUTPUT),
            "gfx1151|AMD Radeon 8060S Graphics",
        )

    def test_rocminfo_without_gpu_prints_nothing(self) -> None:
        cpu_only = ROCMINFO_OUTPUT.split("Agent 2")[0]
        self.assertEqual(_run(["rocminfo_first_gpu"], "rocminfo_first_gpu", cpu_only), "")


class MissingToolTests(unittest.TestCase):
    def test_absent_version_sources_do_not_exit_the_script(self) -> None:
        # The helpers run inside command substitutions under
        # `set -euo pipefail`; a host without the version file or hipconfig
        # must yield an empty field, not end the sweep silently.
        script = (
            "ROCM_PATH=/nonexistent-rocm\nPATH=/usr/bin:/bin\n"
            "hipconfig() { return 1; }\n"
            "v=$(detect_rocm_version); h=$(detect_hip_version); echo \"[$v][$h]\""
        )
        self.assertEqual(_run(["rocm_tool", "detect_rocm_version", "detect_hip_version"], script), "[][]")


class HardwareTagTests(unittest.TestCase):
    def _short(self, full: str) -> str:
        script = f"detect_hardware_full() {{ echo {shlex.quote(full)}; }}\ndetect_hardware_short"
        return _run(["detect_hardware_short"], script)

    def test_strix_halo(self) -> None:
        self.assertEqual(
            self._short("AMD_Radeon_8060S_Graphics_gfx1151_ROCm10.0.0_96GB"), "strixhalo-gfx1151"
        )

    def test_other_amd_target_keeps_the_gfx(self) -> None:
        self.assertEqual(self._short("AMD_Radeon_RX_7900_XTX_gfx1100_ROCm6.4.1_24GB"), "amd-gfx1100")
        self.assertEqual(self._short("AMD_Instinct_MI300X_gfx942_ROCm6.4.1_192GB"), "amd-gfx942")

    def test_existing_tags_unchanged(self) -> None:
        self.assertEqual(self._short("Apple_M1_Ultra_128GB"), "m1ultra")
        self.assertEqual(self._short("NVIDIA_GB10_CUDA13.0_119GB"), "gb10")
        self.assertEqual(self._short("Tesla_V100-PCIE-32GB_CUDA12.4_251GB"), "v100")
        self.assertEqual(self._short("Intel(R)_Xeon(R)_CPU_E5-2690_64GB"), "intel(r)_xeon(r)_cpu")

    def test_full_string_uses_device_memory(self) -> None:
        # A Linux host with no nvidia-smi, after the probe found gfx1151 with a
        # 96 GiB carve-out: the tag must carry the device figure, not `free`.
        script = (
            "nvidia-smi() { return 1; }\nnvcc() { return 1; }\nuname() { echo Linux; }\n"
            "free() { printf 'Mem: 33000000000 0 0\\n'; }\n"
            "ROCM_DETECTED=1 ROCM_GFX=gfx1151 ROCM_DEVICE_NAME='AMD Radeon 8060S Graphics' "
            f"ROCM_VERSION=10.0.0 DEVICE_MEMORY_BYTES={96 * 1024**3}\n"
            "detect_hardware_full"
        )
        self.assertEqual(
            _run(["detect_hardware_full"], script), "AMD_Radeon_8060S_Graphics_gfx1151_ROCm10.0.0_96GB"
        )

    def test_python_harness_tags_the_host_identically(self) -> None:
        # compare_bench_csv.py pairs pylm_<host> with rocm_<host>, so the two
        # harnesses must agree on both names.
        gpu = {
            "gfx": "gfx1151",
            "name": "AMD Radeon 8060S Graphics",
            "rocm_version": "10.0.0",
            "vram_bytes": 96 * 1024**3,
        }
        self.assertEqual(
            bench_mlxlm.rocm_hardware_names(gpu),
            ("strixhalo-gfx1151", "AMD_Radeon_8060S_Graphics_gfx1151_ROCm10.0.0_96GB"),
        )
        gpu["gfx"] = "gfx1100"
        self.assertEqual(bench_mlxlm.rocm_hardware_names(gpu)[0], "amd-gfx1100")


class ProbeRuntimeTests(unittest.TestCase):
    """probe_runtime end to end, against a stub `mlxcel` (no GPU, no model load)."""

    FUNCS = [
        "probe_kernel_backend", "probe_gfx_target", "probe_gpu_name",
        "probe_device_memory_bytes", "rocminfo_first_gpu", "rocm_tool",
        "detect_rocm_version", "detect_hip_version", "sysfs_vram_bytes",
        "probe_runtime", "smallest_checkpoint", "estimate_model_size",
        "run_with_timeout",
    ]

    def _probe(self, stub_output: str, stub_rc: int = 0, rocminfo: str = "") -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            for name, size in (("big", 4096), ("small", 1024)):
                d = tmp_path / "models" / name
                d.mkdir(parents=True)
                (d / "config.json").write_text("{}")
                (d / "model.safetensors").write_bytes(b"0" * size)
            # Records which checkpoint the probe loaded.
            stub = tmp_path / "mlxcel"
            stub.write_text(
                "#!/bin/bash\necho \"$3\" > " + shlex.quote(str(tmp_path / "loaded")) + "\n"
                "cat <<'OUT'\n" + stub_output + "OUT\n"
                f"exit {stub_rc}\n"
            )
            stub.chmod(0o755)
            script = (
                "nvidia-smi() { return 1; }\nuname() { echo Linux; }\n"
                "run_with_timeout() { shift; \"$@\"; }\n"
                "detect_rocm_version() { echo 7.2.0; }\ndetect_hip_version() { echo 7.2.53; }\n"
                "sysfs_vram_bytes() { echo 0; }\n"
                "ROCM_DETECTED=0 ROCM_GFX= ROCM_DEVICE_NAME= ROCM_VERSION= HIP_VERSION=\n"
                "DEVICE_MEMORY_BYTES=0 DEVICE_MEMORY_SOURCE= PROBE_BACKEND= PROBE_TIMEOUT=10\n"
                f"MLXCEL={shlex.quote(str(stub))} MODELS_DIR={shlex.quote(str(tmp_path / 'models'))}\n"
            )
            if rocminfo:
                script += (
                    "rocm_tool() { echo rocminfo_stub; }\n"
                    "rocminfo_stub() { cat <<'RI'\n" + rocminfo + "RI\n}\n"
                    "run_with_timeout() { shift; \"$@\"; }\n"
                )
            else:
                script += "rocm_tool() { return 1; }\n"
            script += (
                "probe_runtime ''\n"
                "echo \"$ROCM_DETECTED|$ROCM_GFX|$ROCM_DEVICE_NAME|$DEVICE_MEMORY_BYTES|$HIP_VERSION\"\n"
            )
            out = _run(self.FUNCS, script)
            loaded = (tmp_path / "loaded").read_text().strip()
        detected, gfx, name, mem, hip = out.splitlines()[-1].split("|")
        return {
            "detected": detected, "gfx": gfx, "name": name, "mem": int(mem),
            "hip": hip, "loaded": os.path.basename(loaded),
        }

    def test_rocm_probe_uses_smallest_checkpoint_and_reads_device(self) -> None:
        r = self._probe(ROCM_PROBE_OUTPUT)
        self.assertEqual(r["loaded"], "small")
        self.assertEqual(r["detected"], "1")
        self.assertEqual(r["gfx"], "gfx1151")
        self.assertEqual(r["name"], "AMD Radeon 8060S Graphics")
        self.assertEqual(r["mem"], 96 * 1024**3)
        self.assertEqual(r["hip"], "7.2.53")

    def test_other_backend_is_not_rocm(self) -> None:
        r = self._probe("[mlxcel] custom kernel backend: cpu\n")
        self.assertEqual(r["detected"], "0")
        self.assertEqual(r["mem"], 0)

    def test_failed_probe_falls_back_to_rocminfo(self) -> None:
        r = self._probe("error: load failed\n", stub_rc=1, rocminfo=ROCMINFO_OUTPUT)
        self.assertEqual(r["detected"], "1")
        self.assertEqual(r["gfx"], "gfx1151")
        self.assertEqual(r["name"], "AMD Radeon 8060S Graphics")


class BackendAndMemoryTests(unittest.TestCase):
    def test_rocm_backend_and_device_memory_budget(self) -> None:
        script = (
            "nvidia-smi() { return 1; }\nuname() { echo Linux; }\n"
            "free() { printf 'Mem: 33000000000 0 0\\n'; }\n"
            "MLXCEL=/bin/false ROCM_DETECTED=1 BACKEND=rocm "
            f"DEVICE_MEMORY_BYTES={96 * 1024**3}\n"
            "detect_backend; detect_memory_bytes"
        )
        backend, memory = _run(["detect_backend", "detect_memory_bytes"], script).split("\n")
        self.assertEqual(backend, "rocm")
        self.assertEqual(int(memory), 96 * 1024**3)

    def test_non_rocm_linux_keeps_host_memory(self) -> None:
        script = (
            "nvidia-smi() { return 1; }\nuname() { echo Linux; }\n"
            "free() { printf 'Mem: 33000000000 0 0\\n'; }\n"
            f"MLXCEL=/bin/false ROCM_DETECTED=0 BACKEND=metal DEVICE_MEMORY_BYTES={96 * 1024**3}\n"
            "detect_backend; detect_memory_bytes"
        )
        backend, memory = _run(["detect_backend", "detect_memory_bytes"], script).split("\n")
        self.assertEqual(backend, "metal")
        self.assertEqual(int(memory), 33000000000)


class CompareHostDetectionTests(unittest.TestCase):
    def test_runtime_and_host(self) -> None:
        rh = compare_bench_csv.runtime_and_host
        self.assertEqual(rh("benchmarks/rocm_strixhalo-gfx1151_2026-09-30.csv"), ("rocm", "strixhalo-gfx1151"))
        self.assertEqual(rh("benchmarks/pylm_strixhalo-gfx1151_2026-09-30.csv"), ("pylm", "strixhalo-gfx1151"))
        self.assertEqual(rh("benchmarks/pylm_m5max_vlm_2026-09-09.csv"), ("pylm", "m5max"))
        self.assertEqual(rh("benchmarks/metal_m1ultra_2026-07-12.csv"), ("metal", "m1ultra"))
        self.assertEqual(rh("benchmarks/cuda_gb10_2026-07-12.csv"), ("cuda", "gb10"))
        self.assertEqual(rh("benchmarks/regression_check_2026-06-10.csv"), (None, None))

    def test_superseded_scan_runs_on_a_rocm_host(self) -> None:
        header = [
            "model", "model_path", "prompt_tokens", "generated_tokens", "prefill_ms",
            "prefill_tok_s", "decode_ms", "decode_tok_s", "date",
        ]

        def write(path, decode, day):
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(header)
                w.writerow(["qwen3-0.6b", "m", "512", "128", "1", "1", "1", decode, day])

        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                os.mkdir("benchmarks")
                before = "benchmarks/rocm_strixhalo-gfx1151_2026-09-01.csv"
                newer = "benchmarks/rocm_strixhalo-gfx1151_2026-09-15_single_qwen3-0.6b.csv"
                after = "benchmarks/pylm_strixhalo-gfx1151_2026-09-30.csv"
                write(before, "100", "2026-09-01")
                write(newer, "150", "2026-09-15")
                write(after, "140", "2026-09-30")
                rows = compare_bench_csv.load_rows(before, {})
                stale = compare_bench_csv.newer_readings_elsewhere(before, after, {"qwen3-0.6b"}, {}, rows)
            finally:
                os.chdir(cwd)
        # Before #1810 the host was neither m5max nor m1ultra and this was {}.
        self.assertIn("qwen3-0.6b", stale)
        self.assertEqual(stale["qwen3-0.6b"][0], 150.0)


if __name__ == "__main__":
    unittest.main()
