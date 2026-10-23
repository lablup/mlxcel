"""Run one independently pinned Rust Objects integration capture under containment."""
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


def digest(path: Path) -> str:
    """Hash a bootstrap helper before importing any capture validation code."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def memory() -> dict[str, int]:
    """Read actual host memory and swap availability in bytes."""
    rows = (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    result = {key: int(value.split()[0]) * 1024 for key, value in rows
              if key in {"MemAvailable", "SwapFree", "MemTotal"}}
    if set(result) != {"MemAvailable", "SwapFree", "MemTotal"}:
        raise RuntimeError("Host memory telemetry unavailable")
    return result


def child_memory(pid: int) -> dict[str, int]:
    """Read child resident/swap telemetry, allowing an already-exited process."""
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
    except FileNotFoundError:
        return {}
    return {line.split(":")[0]: int(line.split()[1]) * 1024 for line in lines
            if line.startswith(("VmRSS:", "VmSwap:", "VmHWM:"))}


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _signal_group(child: subprocess.Popen[Any], signum: int) -> None:
    try:
        os.killpg(child.pid, signum)
    except ProcessLookupError:
        pass


def terminate(child: subprocess.Popen[Any]) -> int:
    """Terminate the child's process group, escalating after the fixed grace."""
    if child.poll() is None:
        _signal_group(child, signal.SIGTERM)
        try:
            return child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _signal_group(child, signal.SIGKILL)
    return child.wait()


def monitor(child: subprocess.Popen[Any], run: Path, start: dict[str, int], policy: dict[str, Any], started: float) -> tuple[int, str | None]:
    """Record 100 ms host/child telemetry and enforce the approved memory gates."""
    low_samples = 0
    reason = None
    with (run / "memory.jsonl").open("x") as metrics:
        while child.poll() is None:
            sample = {"elapsed_seconds": time.monotonic() - started, **memory(), **child_memory(child.pid)}
            metrics.write(json.dumps(sample, sort_keys=True) + "\n")
            metrics.flush()
            low_samples = low_samples + 1 if sample["MemAvailable"] < policy["reserve_bytes"] else 0
            if low_samples >= policy["low_samples"]:
                reason = "two consecutive host MemAvailable samples below recorded reserve"
            elif start["SwapFree"] - sample["SwapFree"] > policy["swap_loss_bytes"] and sample["MemAvailable"] < policy["swap_pressure_available_bytes"]:
                reason = "SwapFree loss exceeds 256 MiB with host MemAvailable below 8 GiB"
            elif sample.get("VmSwap", 0) > policy["child_swap_bytes"]:
                reason = "child VmSwap exceeds 64 MiB"
            if reason:
                return terminate(child), reason
            time.sleep(policy["interval_seconds"])
    return child.wait(), reason


def _bootstrap(plan_path: Path, expected_sha: str) -> tuple[dict[str, Any], Path]:
    if digest(plan_path) != expected_sha:
        raise ValueError("Unapproved capture plan")
    # Strict JSON/schema validation happens in the authenticated validator;
    # this first pass only identifies pinned helpers and an exclusive run path.
    plan = json.loads(plan_path.read_text())
    own = plan["helpers"]["run_rust_capture.py"]
    if digest(Path(__file__).resolve()) != own["sha256"]:
        raise ValueError("Executing driver differs from approved plan")
    run = Path(plan["run_dir"])
    if not run.is_absolute() or run.resolve(strict=False) != run:
        raise ValueError("Run directory must be canonical and absolute")
    run.mkdir(parents=True, exist_ok=False)
    return plan, run


def _freeze(plan: dict[str, Any], plan_path: Path, run: Path) -> ModuleType:
    frozen = run / "frozen"
    frozen.mkdir()
    names = {"run_rust_capture.py", "capture_identity.py", "provenance.py", "provenance.json"}
    if set(plan["helpers"]) != names:
        raise ValueError("Capture helper set mismatch")
    for name in names:
        row = plan["helpers"][name]
        source = Path(row["path"])
        if source.name != name or digest(source) != row["sha256"]:
            raise ValueError(f"Unapproved helper: {name}")
        shutil.copyfile(source, frozen / name)
        if digest(frozen / name) != row["sha256"]:
            raise ValueError(f"Helper changed while freezing: {name}")
    shutil.copyfile(plan_path, frozen / "plan.json")
    shutil.copyfile(plan["source"]["manifest"]["path"], frozen / "source-manifest.json")
    # Execute the already authenticated bytes without creating unpinned pycache.
    location = frozen / "capture_identity.py"
    spec = importlib.util.spec_from_file_location("objects_frozen_capture_identity", location)
    if spec is None:
        raise RuntimeError("Cannot load frozen capture validator")
    module = importlib.util.module_from_spec(spec)
    exec(compile(location.read_bytes(), str(location), "exec"), module.__dict__)
    return module


def run_capture(plan_path: Path, expected_sha: str) -> int:
    """Freeze, authenticate, execute, and validate one fresh capture directory."""
    plan, run = _bootstrap(plan_path, expected_sha)
    started = time.monotonic()
    child: subprocess.Popen[Any] | None = None
    phase = "freeze helpers"
    code: int | None = None
    resources: dict[str, Any] | None = None
    try:
        validator = _freeze(plan, plan_path, run)
        phase = "authenticate pinned inputs and compiled source"
        plan, _, _ = validator.authenticate_plan(plan_path, expected_sha)
        command, controlled = validator.command_environment(plan)
        if (run / "output").exists():
            raise ValueError("Refuse preexisting output directory")
        start = memory()
        invocation = {"command": command, "environment": controlled, "cwd": plan["source"]["root"],
                      "plan_sha256": expected_sha, "policy": validator.POLICY, "start_memory": start,
                      "frozen_sha256": {n: digest(run / "frozen" / n) for n in validator.HELPERS}}
        _json(run / "invocation.json", invocation)
        if start["MemAvailable"] <= validator.RESERVE:
            raise RuntimeError("Insufficient available host memory")
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("MLXCEL_SAM3D_OBJECTS_") and key != "LD_PRELOAD"}
        environment.update(controlled)
        phase = "child execution"
        with (run / "child.log").open("x") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     cwd=invocation["cwd"], env=environment, start_new_session=True)
            (run / "child.pid").write_text(str(child.pid) + "\n")
            code, reason = monitor(child, run, start, validator.POLICY, started)
        (run / "child.exit").write_text(str(code) + "\n")
        status = 0 if code == 0 and reason is None else 1
        resources = {"child_exit": code, "wrapper_exit": status, "stopped_for_memory": reason is not None,
                     "stop_reason": reason, "elapsed_seconds": time.monotonic() - started,
                     "end_memory": memory(), "invocation_sha256": digest(run / "invocation.json"),
                     "child_log_sha256": digest(run / "child.log"), "memory_sha256": digest(run / "memory.jsonl"),
                     "monitor_sha256": digest(run / "frozen" / "run_rust_capture.py")}
        _json(run / "resources.json", resources)
        (run / "wrapper.exit").write_text(str(status) + "\n")
        if status:
            return status
        phase = "validate actual capture evidence"
        normalized = validator.validate_capture(plan_path, expected_sha)
        _json(run / "capture-receipt.json", normalized)
        print(json.dumps({"receipt": str(run / "capture-receipt.json"),
                          "receipt_sha256": digest(run / "capture-receipt.json"), "child_exit": code, "wrapper_exit": 0}), flush=True)
        return 0
    except BaseException as error:
        # This also cleans up on KeyboardInterrupt/SIGTERM; never leave a model
        # process running when the monitor or identity validation fails.
        if child is not None:
            code = terminate(child)
            (run / "child.exit").write_text(str(code) + "\n")
        (run / "wrapper.exit").write_text("1\n")
        if resources is not None:
            resources["wrapper_exit"] = 1
            _json(run / "resources.json", resources)
        _json(run / "failure.json", {"phase": phase, "child_started": child is not None,
                                     "child_exit": code, "wrapper_exit": 1, "error": str(error),
                                     "elapsed_seconds": time.monotonic() - started})
        print(f"Rust capture failed during {phase}: {error}", file=sys.stderr, flush=True)
        return 1


def main() -> int:
    """CLI entry point; no model may run without an independently pinned plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    def interrupted(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt(f"Capture interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    return run_capture(args.plan.resolve(strict=True), args.plan_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
