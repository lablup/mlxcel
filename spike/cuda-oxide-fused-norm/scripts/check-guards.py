#!/usr/bin/env python3
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
"""Malformed-input guards for scripts/oxide_artifact.py (issue #2832).

Each case feeds a damaged input through the real tooling and requires a
non-zero exit with a message, no traceback and no hang. No GPU and no
compiler is involved: the cases use synthetic manifests and logs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE / "oxide_artifact.py"
ABI = HERE.parent / "abi"


def run_tool(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "CUDA_OXIDE_SMOKE_SCRIPTS": os.environ.get("CUDA_OXIDE_SMOKE_SCRIPTS", str(HERE.parent.parent / "cuda-oxide-smoke" / "scripts"))},
    )


def expect_failure(label: str, args: list[str]) -> bool:
    try:
        result = run_tool(args)
    except subprocess.TimeoutExpired:
        print(f"FAIL {label}: hung")
        return False
    if result.returncode == 0:
        print(f"FAIL {label}: exit 0")
        return False
    if "Traceback" in result.stderr:
        print(f"FAIL {label}: traceback\n{result.stderr}")
        return False
    if not (result.stderr.strip() or result.stdout.strip()):
        print(f"FAIL {label}: no message")
        return False
    print(f"PASS {label}: exit {result.returncode}")
    return True


def synthetic_manifest(entry: str, case_count: int, elements_total: int) -> dict:
    schema = json.loads((ABI / f"{entry}.schema.json").read_text())
    launch = dict(schema["launch"])
    launch["case_count"] = case_count
    launch["elements_total"] = elements_total
    return {"entry_symbol": entry, "launch": launch}


def main() -> int:
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifests = root / "manifests"
        manifests.mkdir()
        for entry in [
            "fused_add_rms_norm_f32",
            "fused_add_rms_norm_f16",
            "fused_add_rms_norm_bf16",
            "fused_add_rms_norm_f16_wf32",
            "fused_add_rms_norm_bf16_wf32",
        ]:
            (manifests / f"{entry}.sm_121.manifest.json").write_text(
                json.dumps(synthetic_manifest(entry, 1 if entry.endswith("_f32") else 0, 128 if entry.endswith("_f32") else 0))
            )
        good = root / "good.log"
        good.write_text(
            "PASS case=f32_f32_r1_d128_e6_b0_normal act=f32 w=f32 rows=1 dim=128 grid=1 block=32 elements=128 model_nrms=0e0 model_nmax=0e0\n"
            "REJECT case=zero_dim reason=\"x\"\n"
            "RESULT standalone=1 mlx=0 cases_passed=1 cases_total=1 rejected=1 elements_checked=128 comparison=host-model\n"
        )
        result = run_tool(["check-launch", "--manifest-dir", str(manifests), "--log", str(good)])
        if result.returncode != 0:
            print(f"FAIL baseline: the well-formed log must pass\n{result.stdout}{result.stderr}")
            ok = False
        else:
            print("PASS baseline: well-formed log accepted")

        def damaged(name: str, text: str) -> str:
            path = root / name
            path.write_text(text)
            return str(path)

        cases = [
            ("wrong block", damaged("block.log", good.read_text().replace("block=32", "block=64"))),
            ("wrong grid", damaged("grid.log", good.read_text().replace("grid=1 ", "grid=2 "))),
            ("short run", damaged("short.log", "RESULT standalone=1 mlx=0 cases_passed=0 cases_total=0 rejected=1 elements_checked=0 comparison=host-model\n")),
            ("missing result", damaged("noresult.log", good.read_text().splitlines()[0] + "\n")),
            ("no rejections", damaged("noreject.log", good.read_text().replace("REJECT case=zero_dim reason=\"x\"\n", "").replace("rejected=1", "rejected=0"))),
            ("duplicate case", damaged("dup.log", good.read_text().splitlines()[0] + "\n" + good.read_text().replace("cases_passed=1 cases_total=1", "cases_passed=2 cases_total=2").replace("elements_checked=128", "elements_checked=256"))),
        ]
        for label, log in cases:
            ok &= expect_failure(label, ["check-launch", "--manifest-dir", str(manifests), "--log", log])

        # Manifest path escapes must be refused before any read.
        escaped = root / "escaped"
        (escaped / "artifacts").mkdir(parents=True)
        (escaped / "artifacts" / "x.sm_121.manifest.json").write_text(
            json.dumps(
                {
                    "entry_symbol": "fused_add_rms_norm_f32",
                    "abi_version": 1,
                    "source": {"repository": "r", "revision": "v"},
                    "artifacts": {"ptx": {"path": "../../etc/passwd", "sha256": "0"}, "cubin": {"path": "/dev/zero", "sha256": "0"}},
                    "container": {"payload_sha256": "0"},
                    "target": {"device_arch": "sm_121", "cubin_target": "sm_121"},
                    "arguments": [],
                    "launch": {},
                }
            )
        )
        ok &= expect_failure(
            "escaping artifact path",
            ["validate", "--manifest-dir", str(escaped / "artifacts"), "--schema-dir", str(ABI)],
        )
    print("PASS guards" if ok else "FAIL guards")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
