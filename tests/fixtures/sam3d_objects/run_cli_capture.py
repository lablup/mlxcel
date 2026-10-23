"""Capture the literal default Objects CLI under independently pinned containment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import ModuleType
from typing import Any

HELPERS = {"run_cli_capture.py", "cli_identity.py", "capture_identity.py", "run_rust_capture.py",
           "compare_reference.py", "provenance.py", "provenance.json", "comparison-contract.json"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def bootstrap(plan_path: Path, expected_sha: str) -> tuple[dict[str, Any], Path]:
    if digest(plan_path) != expected_sha:
        raise ValueError("Unapproved CLI plan")
    plan = json.loads(plan_path.read_text())
    if set(plan["helpers"]) != HELPERS or digest(Path(__file__).resolve()) != plan["helpers"]["run_cli_capture.py"]["sha256"]:
        raise ValueError("Unapproved CLI driver/helper set")
    run = Path(plan["run_dir"])
    if not run.is_absolute() or run.resolve(strict=False) != run:
        raise ValueError("Noncanonical CLI run directory")
    run.mkdir(parents=True, exist_ok=False)
    return plan, run


def freeze(plan: dict[str, Any], plan_path: Path, expected_sha: str, run: Path) -> ModuleType:
    frozen = run / "frozen"
    frozen.mkdir()
    files = {**plan["helpers"], "plan.json": {"path": str(plan_path), "sha256": expected_sha},
             "source-manifest.json": plan["source"]["manifest"], "reference.json": plan["reference"]}
    for name, row in files.items():
        path = Path(row["path"])
        if not path.is_absolute() or path.resolve(strict=True) != path or digest(path) != row["sha256"]:
            raise ValueError(f"Pinned CLI input changed: {name}")
        shutil.copyfile(path, frozen / name)
        if digest(frozen / name) != row["sha256"]:
            raise ValueError(f"CLI input changed while freezing: {name}")
    location = frozen / "cli_identity.py"
    spec = importlib.util.spec_from_file_location("frozen_cli_identity", location)
    if spec is None:
        raise RuntimeError("Cannot load CLI validator")
    module = importlib.util.module_from_spec(spec)
    payload = location.read_bytes()
    if hashlib.sha256(payload).hexdigest() != plan["helpers"]["cli_identity.py"]["sha256"]:
        raise ValueError("Frozen CLI validator changed before import")
    exec(compile(payload, str(location), "exec"), module.__dict__)
    return module


def attest_child(child: subprocess.Popen[Any], plan: dict[str, Any], environment: dict[str, str]) -> dict[str, Any]:
    """Read the actual live child's executable and controlled environment."""
    root = Path(f"/proc/{child.pid}")
    executable = (root / "exe").resolve(strict=True)
    actual = {}
    for row in (root / "environ").read_bytes().split(b"\0"):
        if b"=" in row:
            key, value = row.split(b"=", 1)
            if key.decode(errors="surrogateescape") in environment:
                key = key.decode()
                if key in actual:
                    raise ValueError("Duplicate controlled child environment key")
                actual[key] = value.decode()
    if str(executable) != plan["executable"]["path"] or actual != environment:
        raise ValueError("Actual CLI executable/environment differs from plan")
    actual_hash = digest(root / "exe")
    if actual_hash != plan["executable"]["sha256"]:
        raise ValueError("Actual running CLI executable hash differs")
    return {"pid": child.pid, "executable": str(executable), "executable_sha256": actual_hash, "environment": actual}


def run_capture(plan_path: Path, expected_sha: str) -> int:
    plan, run = bootstrap(plan_path, expected_sha)
    child, monitor, resources = None, None, None
    started, code, phase = time.monotonic(), None, "freeze CLI evidence"
    try:
        validator = freeze(plan, plan_path, expected_sha, run)
        phase = "authenticate CLI source, assets, and reference"
        plan, _, modules = validator.authenticate_plan(plan_path, expected_sha)
        monitor, policy = modules["run_rust_capture"], modules["capture_identity"].POLICY
        command, controlled = validator.command_environment(plan)
        if (run / "output").exists():
            raise ValueError("Refuse existing CLI output directory")
        start = monitor.memory()
        invocation = {"command": command, "environment": controlled, "cwd": plan["source"]["root"],
                      "plan_sha256": expected_sha, "policy": policy, "start_memory": start,
                      "frozen_sha256": {n: digest(run / "frozen" / n) for n in HELPERS}}
        write_json(run / "invocation.json", invocation)
        if start["MemAvailable"] <= policy["reserve_bytes"]:
            raise RuntimeError("Insufficient CLI host memory")
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("MLXCEL_SAM3D_OBJECTS_") and key != "LD_PRELOAD"}
        environment.update(controlled)
        phase = "actual CLI execution"
        with (run / "child.log").open("x") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     cwd=invocation["cwd"], env=environment, start_new_session=True)
            (run / "child.pid").write_text(str(child.pid) + "\n")
            write_json(run / "executed.json", attest_child(child, plan, controlled))
            code, reason = monitor.monitor(child, run, start, policy, started)
        (run / "child.exit").write_text(str(code) + "\n")
        status = 0 if code == 0 and reason is None else 1
        resources = {"child_exit": code, "wrapper_exit": status, "stopped_for_memory": reason is not None,
                     "stop_reason": reason, "elapsed_seconds": time.monotonic() - started, "end_memory": monitor.memory(),
                     "invocation_sha256": digest(run / "invocation.json"), "executed_sha256": digest(run / "executed.json"),
                     "child_log_sha256": digest(run / "child.log"), "memory_sha256": digest(run / "memory.jsonl"),
                     "monitor_sha256": digest(run / "frozen" / "run_rust_capture.py")}
        write_json(run / "resources.json", resources)
        (run / "wrapper.exit").write_text(str(status) + "\n")
        if status:
            return status
        phase = "authenticate CLI artifacts and complete-array comparison"
        identity, report = validator.validate_capture(plan_path, expected_sha)
        if report["failures"]:
            report.pop("identity", None)
            report["provenance"] = {n: identity[n] for n in ("plan_sha256", "reference_receipt_sha256", "reference_artifact_sha256", "artifact_sha256", "helper_sha256")}
            write_json(run / "comparison.json", report)
            raise ValueError("CLI numerical acceptance failed: " + "; ".join(report["failures"]))
        write_json(run / "comparison.json", report)
        write_json(run / "capture-receipt.json", identity)
        print(json.dumps({"receipt": str(run / "capture-receipt.json"), "receipt_sha256": digest(run / "capture-receipt.json"),
                          "report": str(run / "comparison.json"), "report_sha256": digest(run / "comparison.json")}), flush=True)
        return 0
    except BaseException as error:
        (run / "capture-receipt.json").unlink(missing_ok=True)
        if child is not None and monitor is not None:
            code = monitor.terminate(child)
            (run / "child.exit").write_text(str(code) + "\n")
        (run / "wrapper.exit").write_text("1\n")
        if resources is not None:
            resources["wrapper_exit"] = 1
            write_json(run / "resources.json", resources)
        write_json(run / "failure.json", {"phase": phase, "error": str(error), "child_started": child is not None,
                                          "child_exit": code, "wrapper_exit": 1, "elapsed_seconds": time.monotonic() - started})
        print(f"CLI capture failed during {phase}: {error}", file=sys.stderr, flush=True)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    def interrupted(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt(f"CLI capture interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    return run_capture(args.plan.resolve(strict=True), args.plan_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
