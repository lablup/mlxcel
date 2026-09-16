#!/usr/bin/env python3
"""Validate root-run WebUI release evidence bundles for issue #1848.

This script validates evidence shape and artifact references only. It does not run
hardware, CUDA, browsers, servers, or model inference; those commands must be run
separately by the root-owned scheduled gates and their raw outputs referenced by
this JSON.
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

REQUIRED_ROWS = {
    "hardware": (
        "dense_generation",
        "hybrid_or_moe_generation",
        "vlm_image_generation",
        "small_public_download",
        "activity_native_hidden",
        "startup_and_performance",
    ),
    "cuda": (
        "cuda_ui_on",
        "cuda_ui_off",
        "cuda_installed_artifact",
    ),
}

CONTEXT_FIELD = {"hardware": "hardware", "cuda": "host"}


class EvidenceError(ValueError):
    pass


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _nonempty_string(value: Any, field: str) -> str:
    _ensure(isinstance(value, str) and bool(value.strip()), f"{field} must be a non-empty string")
    return value


def _validate_sha(value: Any, field: str, pattern: re.Pattern[str], bits: str) -> None:
    text = _nonempty_string(value, field)
    _ensure(bool(pattern.fullmatch(text)), f"{field} must be a lowercase {bits} hex digest")


def _validate_features(value: Any) -> None:
    _ensure(isinstance(value, list) and bool(value), "features must be a non-empty list")
    for index, item in enumerate(value):
        _nonempty_string(item, f"features[{index}]")


def _artifact_paths(row: dict[str, Any]) -> list[str]:
    artifacts = row.get("artifacts")
    if artifacts is None:
        for legacy_key in ("artifact", "raw_evidence_path"):
            if legacy_key in row:
                artifacts = [row[legacy_key]]
                break
    _ensure(isinstance(artifacts, list) and bool(artifacts), f"row {row.get('id', '<unknown>')} must list one or more artifacts")
    result: list[str] = []
    for index, item in enumerate(artifacts):
        result.append(_nonempty_string(item, f"row {row.get('id', '<unknown>')} artifacts[{index}]"))
    return result


def _resolve_artifact(evidence_path: Path, artifact: str) -> Path:
    path = Path(artifact)
    if not path.is_absolute():
        path = evidence_path.parent / path
    return path


def _validate_artifacts(evidence_path: Path, row: dict[str, Any]) -> None:
    for artifact in _artifact_paths(row):
        resolved = _resolve_artifact(evidence_path, artifact)
        _ensure(resolved.exists(), f"row {row['id']} artifact does not exist: {artifact}")
        _ensure(resolved.is_file(), f"row {row['id']} artifact must be a file: {artifact}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"invalid JSON: {exc}") from exc
    _ensure(isinstance(data, dict), "evidence root must be an object")
    return data


def validate_evidence(kind: str, evidence_path: Path) -> None:
    _ensure(kind in REQUIRED_ROWS, f"unknown evidence kind: {kind}")
    evidence_path = evidence_path.resolve()
    data = _load_json(evidence_path)
    _validate_sha(data.get("source_commit"), "source_commit", HEX40, "40-character git SHA")
    _validate_sha(data.get("binary_sha256"), "binary_sha256", HEX64, "64-character SHA-256")
    _validate_features(data.get("features"))
    context_field = CONTEXT_FIELD[kind]
    context = data.get(context_field)
    _ensure(isinstance(context, (dict, str)) and bool(context), f"{context_field} must describe the execution host")
    rows = data.get("rows")
    _ensure(isinstance(rows, list) and bool(rows), "rows must be a non-empty list")

    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        _ensure(isinstance(row, dict), f"rows[{index}] must be an object")
        row_id = _nonempty_string(row.get("id"), f"rows[{index}].id")
        _ensure(row_id not in by_id, f"duplicate evidence row id: {row_id}")
        by_id[row_id] = row
        _nonempty_string(row.get("command"), f"row {row_id} command")
        _nonempty_string(row.get("result"), f"row {row_id} result")
        _validate_artifacts(evidence_path, row)

    missing = [row_id for row_id in REQUIRED_ROWS[kind] if row_id not in by_id]
    _ensure(not missing, f"missing {kind} evidence rows: {missing}")
    nonpassing = [row_id for row_id in REQUIRED_ROWS[kind] if by_id[row_id].get("result") != "passed"]
    _ensure(not nonpassing, f"non-passing {kind} evidence rows: {nonpassing}")


def _valid_bundle(kind: str, directory: Path) -> Path:
    artifact = directory / "raw.log"
    artifact.write_text("evidence\n", encoding="utf-8")
    rows = [
        {"id": row_id, "result": "passed", "command": f"run {row_id}", "artifacts": ["raw.log"]}
        for row_id in REQUIRED_ROWS[kind]
    ]
    data = {
        "source_commit": "a" * 40,
        "binary_sha256": "b" * 64,
        "features": ["webui", "cuda" if kind == "cuda" else "metal", "accelerate"],
        CONTEXT_FIELD[kind]: {"os": "test"},
        "rows": rows,
    }
    path = directory / f"{kind}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _expect_fails(kind: str, data_mutator: Any, message: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        path = _valid_bundle(kind, directory)
        data = json.loads(path.read_text(encoding="utf-8"))
        data_mutator(data)
        path.write_text(json.dumps(data), encoding="utf-8")
        try:
            validate_evidence(kind, path)
        except EvidenceError as exc:
            if message not in str(exc):
                raise AssertionError(f"expected failure containing {message!r}, got {exc!r}") from exc
            return
        raise AssertionError(f"expected {kind} evidence validation to fail: {message}")


def self_test() -> None:
    for kind in REQUIRED_ROWS:
        with tempfile.TemporaryDirectory() as tmp:
            validate_evidence(kind, _valid_bundle(kind, Path(tmp)))
        _expect_fails(kind, lambda data: data.update(source_commit="1234"), "source_commit")
        _expect_fails(kind, lambda data: data.update(binary_sha256="1234"), "binary_sha256")
        _expect_fails(kind, lambda data: data.update(features="cuda"), "features")
        _expect_fails(kind, lambda data: data["rows"].append(dict(data["rows"][0])), "duplicate")
        _expect_fails(kind, lambda data: data["rows"].pop(), "missing")
        _expect_fails(kind, lambda data: data["rows"][0].update(result="blocked"), "non-passing")
        _expect_fails(kind, lambda data: data["rows"][0].update(artifacts=["missing.log"]), "does not exist")
        _expect_fails(kind, lambda data: data["rows"][0].pop("command"), "command")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=sorted(REQUIRED_ROWS), nargs="?", help="evidence bundle kind to validate")
    parser.add_argument("--evidence", type=Path, help="path to the evidence JSON bundle")
    parser.add_argument("--self-test", action="store_true", help="run built-in validator regression tests")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        print("check_webui_evidence.py self-test passed")
        return 0
    if not args.kind or not args.evidence:
        raise SystemExit("kind and --evidence are required unless --self-test is used")
    validate_evidence(args.kind, args.evidence)
    print(f"validated WebUI {args.kind} evidence: {args.evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
