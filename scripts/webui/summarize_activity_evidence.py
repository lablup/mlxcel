#!/usr/bin/env python3
# Copyright 2026 Lablup Inc. Licensed under the Apache License, Version 2.0.
"""Create the CI-uploadable Activity performance evidence summary.

The full verifier JSON intentionally stays on the runner because it can contain
local paths, process details and other host-specific diagnostics. This script
keeps only the reviewed fields needed to prove native-hidden success.
"""
from __future__ import annotations
import argparse, json, math, os, re, stat
from pathlib import Path
from typing import Any

FORBIDDEN_KEYS = {"path", "work_dir", "model_view", "output", "log", "processes", "env"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_MODES = {"one-visible", "two-visible", "hidden"}
VISIBLE_ONLY_MODES = {"one-visible", "two-visible"}

def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)

def hex64(value: Any, label: str) -> str:
    require(isinstance(value, str) and HEX64.fullmatch(value) is not None, f"{label} must be a sha256 hex digest")
    return value

def finite_number(value: Any, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} must be numeric")
    number = float(value)
    require(math.isfinite(number), f"{label} must be finite")
    return number

def nonnegative_int(value: Any, label: str) -> int:
    require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, f"{label} must be a non-negative integer")
    return value

def safe_relative_name(value: Any, label: str) -> str:
    require(isinstance(value, str) and value, f"{label} must be a non-empty relative name")
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts and len(path.parts) <= 2, f"{label} must be a safe relative filename")
    return value

def activity_summary(records: list[Any], hidden_native: str = "measured") -> dict[str, Any]:
    # `hidden_native` is the caller's declared expectation, not something read out of the evidence:
    # accepting a run without the native hidden acceptance has to be an explicit argument at the
    # call site, so a deferral cannot arrive by a summary quietly reporting less than it used to.
    require(len(records) == 1 and isinstance(records[0], dict), "expected one activity_performance record")
    output_sha256 = hex64(records[0].get("output_sha256"), "activity output_sha256")
    summary = records[0].get("summary")
    require(isinstance(summary, dict), "activity summary missing")
    expected_status = "within-target" if hidden_native == "measured" else "incomplete"
    expected_modes = REQUIRED_MODES if hidden_native == "measured" else VISIBLE_ONLY_MODES
    require(summary.get("status") == expected_status, f"activity status must be {expected_status}")
    require(summary.get("hidden_native") == hidden_native, f"hidden_native must be {hidden_native}")
    summaries = summary.get("summaries")
    require(isinstance(summaries, list) and summaries, "mode summaries missing")
    allowed = []
    seen: set[str] = set()
    for item in summaries:
        require(isinstance(item, dict), "mode summary must be an object")
        mode = item.get("mode")
        require(mode in expected_modes and mode not in seen, "mode summaries must contain each required mode once")
        seen.add(mode)
        require(item.get("status") == "within-target", f"{mode} status must be within-target")
        require(item.get("paired_runs") == 5, f"{mode} paired_runs must be 5")
        degradation = finite_number(item.get("median_decode_degradation_percent"), f"{mode} degradation")
        cv = finite_number(item.get("baseline_cv_percent"), f"{mode} baseline cv")
        require(degradation <= 2, f"{mode} degradation exceeds budget")
        require(0 <= cv <= 5, f"{mode} baseline cv exceeds budget")
        pair_range = item.get("paired_range_percent")
        require(isinstance(pair_range, list) and len(pair_range) == 2, f"{mode} paired_range_percent must have two values")
        low = finite_number(pair_range[0], f"{mode} paired range low")
        high = finite_number(pair_range[1], f"{mode} paired range high")
        require(low <= high, f"{mode} paired range must be ordered")
        allowed.append({
            "mode": mode,
            "status": "within-target",
            "paired_runs": 5,
            "median_decode_degradation_percent": degradation,
            "baseline_cv_percent": cv,
            "paired_range_percent": [low, high],
        })
    require(seen == expected_modes, f"mode summaries missing required modes: {sorted(expected_modes - seen)}")
    out = {"status": summary["status"], "hidden_native": summary["hidden_native"], "output_sha256": output_sha256, "summaries": allowed}
    for key in ("sample_counts", "preflight_counts"):
        if key in records[0]:
            value = records[0][key]
            require(isinstance(value, dict), f"{key} must be an object")
            out[key] = {str(k): nonnegative_int(v, f"{key}.{k}") for k, v in value.items()}
    return out

def checkpoint_summary(raw: Any) -> dict[str, Any]:
    require(isinstance(raw, dict), "checkpoint summary missing")
    metadata = raw.get("metadata_sha256")
    require(isinstance(metadata, dict), "metadata_sha256 must be an object")
    safe_metadata = {safe_relative_name(k, "metadata name"): hex64(v, f"metadata {k}") for k, v in metadata.items()}
    weights = raw.get("safetensors_sha256")
    require(isinstance(weights, list), "safetensors_sha256 must be a list")
    safe_weights = []
    for item in weights:
        require(isinstance(item, dict), "safetensors entry must be an object")
        safe_weights.append({"relative_path": safe_relative_name(item.get("relative_path"), "safetensors relative_path"), "size": nonnegative_int(item.get("size"), "safetensors size"), "sha256": hex64(item.get("sha256"), "safetensors sha256")})
    return {
        "display_name": raw.get("display_name"),
        "revision": raw.get("revision"),
        "file_count": nonnegative_int(raw.get("file_count"), "file_count"),
        "total_bytes": nonnegative_int(raw.get("total_bytes"), "total_bytes"),
        "safetensors_count": nonnegative_int(raw.get("safetensors_count"), "safetensors_count"),
        "safetensors_total_bytes": nonnegative_int(raw.get("safetensors_total_bytes"), "safetensors_total_bytes"),
        "metadata_sha256": safe_metadata,
        "safetensors_sha256": safe_weights,
    }

def build_summary(data: dict[str, Any], hidden_native: str = "measured") -> dict[str, Any]:
    require(data.get("status") == "passed", "full activity evidence did not pass")
    server_shutdown = data.get("server_shutdown")
    require(isinstance(server_shutdown, dict), "server_shutdown missing")
    require(server_shutdown.get("exit_code") == 0 and server_shutdown.get("forced") is False and server_shutdown.get("process_group_empty") is True, "server shutdown must be graceful and complete")
    summary = {
        "status": data.get("status"),
        "source_commit": data.get("source_commit"),
        "server_bin_sha256": hex64(data.get("server_bin_sha256"), "server_bin_sha256"),
        "features": data.get("features"),
        "display": {"virtual": data.get("display", {}).get("virtual"), "minimum_geometry": data.get("display", {}).get("minimum_geometry")},
        "checkpoint": checkpoint_summary(data.get("checkpoint")),
        "activity_performance": activity_summary(data.get("activity_performance", []), hidden_native),
        "server_shutdown": {"exit_code": 0, "forced": False, "process_group_empty": True},
    }
    text = json.dumps(summary, sort_keys=True)
    for key in FORBIDDEN_KEYS:
        require(f'"{key}"' not in text, f"summary contains forbidden key {key}")
    return summary

def write_private_json(path: Path, value: dict[str, Any]) -> None:
    if path.is_symlink():
        raise AssertionError(f"refusing to write through symlink: {path}")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fp:
        fp.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.chmod(path, 0o600)
    require(stat.S_IMODE(path.stat().st_mode) == 0o600, "summary evidence is not 0600")

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-hidden-native", default="measured", choices=["measured", "not-run"], help="declare whether the run included the native hidden acceptance; 'not-run' accepts an incomplete visible-only summary and is how a deferral is recorded")
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    require(isinstance(data, dict), "full activity evidence must be a JSON object")
    write_private_json(args.output, build_summary(data, args.expect_hidden_native))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
