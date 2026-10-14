#!/usr/bin/env python3
"""Validate and summarize the model-owned storage paired/null serving sweep."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import re
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATA = Path(__file__).resolve().parent
MODELS = ("gemma3", "llama4")
PROMPTS = (512, 4096)
ROUNDS = range(1, 6)
ARMS = ("paged-a", "dense", "paged-b")
CONCURRENCIES = (1, 4, 8)
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
LOG_NAME = re.compile(
    r"client-(?P<model>gemma3|llama4)-p(?P<prompt>512|4096)-"
    r"r(?P<round>[1-5])-(?P<arm>paged-a|dense|paged-b)\.txt"
)
TABLE_ROW = re.compile(
    r"^\s*(?P<concurrency>1|4|8)\s+\d+\s+(?P<failed>\d+)\s+"
    r"(?:\d+(?:\.\d+)?\s+){3}(?P<aggregate>\d+(?:\.\d+)?)\s*$"
)
SCHEDULE_FIELDS = (
    "model",
    "prompt_tokens",
    "round",
    "position",
    "arm",
    "backend",
    "started_utc",
)


@dataclass(frozen=True)
class ArmSpec:
    """One of the protocol's 60 fresh-server arms."""

    model: str
    prompt_tokens: int
    round: int
    position: int
    arm: str
    backend: str

    @property
    def label(self) -> str:
        return (
            f"{self.model}-p{self.prompt_tokens}-r{self.round}-{self.arm}"
        )

    @property
    def client_name(self) -> str:
        return f"client-{self.label}.txt"


@dataclass(frozen=True)
class Run:
    """Aggregate throughput from one fresh-server arm."""

    model: str
    prompt_tokens: int
    round: int
    arm: str
    concurrency: int
    aggregate_tok_s: float


@dataclass(frozen=True)
class Protocol:
    """Validated evidence needed before a storage decision is permitted."""

    environment: dict[str, str]
    schedule_times: dict[str, dt.datetime]
    runs: tuple[Run, ...]


def arm_order(round_no: int) -> tuple[str, str, str]:
    """Return the preregistered rotating arm order for a round."""

    orders = (
        ("paged-a", "dense", "paged-b"),
        ("dense", "paged-b", "paged-a"),
        ("paged-b", "paged-a", "dense"),
    )
    return orders[(round_no - 1) % len(orders)]


def expected_arms() -> tuple[ArmSpec, ...]:
    """Return the exact ordered 60-arm protocol."""

    specs = []
    for model in MODELS:
        for prompt in PROMPTS:
            for round_no in ROUNDS:
                for position, arm in enumerate(arm_order(round_no), start=1):
                    specs.append(
                        ArmSpec(
                            model=model,
                            prompt_tokens=prompt,
                            round=round_no,
                            position=position,
                            arm=arm,
                            backend="dense" if arm == "dense" else "paged",
                        )
                    )
    return tuple(specs)


def parse_utc(value: object, context: str) -> dt.datetime:
    """Parse one second-resolution UTC timestamp and reject ambiguous forms."""

    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{context}: expected a UTC timestamp ending in Z")
    try:
        parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{context}: invalid UTC timestamp {value!r}") from exc
    if parsed.microsecond != 0:
        raise ValueError(f"{context}: timestamp must use whole seconds")
    return parsed


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 for a provenance artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_environment(data: Path) -> tuple[dict[str, str], dt.datetime, dt.datetime]:
    """Validate source, binary, MLX, host, and script provenance."""

    path = data / "environment.txt"
    if not path.is_file():
        raise ValueError(f"missing provenance file: {path}")
    environment: dict[str, str] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in environment:
            raise ValueError(f"{path}:{line_no}: malformed or duplicate provenance key")
        environment[key] = value

    required = {
        "started_utc",
        "completed_utc",
        "worktree",
        "worktree_clean",
        "binary",
        "binary_sha256",
        "binary_size_bytes",
        "binary_mtime_utc",
        "commit",
        "mlx_cmake_git_tag",
        "mlx_checkout",
        "mlx_checkout_commit",
        "hostgate_source",
        "hostgate_sha256",
        "gate_sha256",
        "analyzer_sha256",
        "run_sha256",
        "host",
        "uname",
        "gpu",
    }
    missing = sorted(required - environment.keys())
    if missing:
        raise ValueError(f"{path}: missing provenance keys: {', '.join(missing)}")

    started = parse_utc(environment["started_utc"], f"{path}:started_utc")
    completed = parse_utc(environment["completed_utc"], f"{path}:completed_utc")
    if completed <= started:
        raise ValueError(f"{path}: completed_utc must be after started_utc")
    if environment["worktree_clean"] != "true":
        raise ValueError(f"{path}: benchmark did not start from a clean worktree")
    if HEX40.fullmatch(environment["commit"]) is None:
        raise ValueError(f"{path}: commit is not a 40-hex object id")
    if HEX40.fullmatch(environment["mlx_cmake_git_tag"]) is None:
        raise ValueError(f"{path}: MLX CMake GIT_TAG is not a 40-hex object id")
    if HEX40.fullmatch(environment["mlx_checkout_commit"]) is None:
        raise ValueError(f"{path}: MLX checkout commit is not a 40-hex object id")
    if environment["mlx_cmake_git_tag"] != environment["mlx_checkout_commit"]:
        raise ValueError(f"{path}: configured and built MLX commits differ")

    for key in ("binary_sha256", "hostgate_sha256", "gate_sha256", "analyzer_sha256", "run_sha256"):
        if HEX64.fullmatch(environment[key]) is None:
            raise ValueError(f"{path}: {key} is not a 64-hex SHA-256")
    for key in ("worktree", "binary", "mlx_checkout", "hostgate_source", "host", "uname", "gpu"):
        if not environment[key].strip():
            raise ValueError(f"{path}: {key} must be nonempty")

    worktree = Path(environment["worktree"])
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(worktree),
                "cat-file",
                "-e",
                f"{environment['commit']}^{{commit}}",
            ],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"{path}: recorded commit is unavailable in the worktree") from exc

    binary = Path(environment["binary"])
    if not binary.is_file():
        raise ValueError(f"{path}: recorded binary is missing: {binary}")
    if sha256_file(binary) != environment["binary_sha256"]:
        raise ValueError(f"{path}: recorded binary SHA-256 does not match the artifact")
    try:
        recorded_size = int(environment["binary_size_bytes"])
    except ValueError as exc:
        raise ValueError(f"{path}: binary_size_bytes is not an integer") from exc
    if recorded_size != binary.stat().st_size:
        raise ValueError(f"{path}: recorded binary size does not match the artifact")
    binary_mtime = parse_utc(environment["binary_mtime_utc"], f"{path}:binary_mtime_utc")
    actual_mtime = dt.datetime.fromtimestamp(
        int(binary.stat().st_mtime), tz=dt.UTC
    )
    if binary_mtime != actual_mtime:
        raise ValueError(f"{path}: recorded binary mtime does not match the artifact")
    if binary_mtime > started:
        raise ValueError(f"{path}: binary was modified after the sweep started")

    provenance_files = {
        "hostgate_sha256": Path(environment["hostgate_source"]),
        "gate_sha256": data / "gate.py",
        "analyzer_sha256": data / "analyze.py",
        "run_sha256": data / "run.sh",
    }
    for key, artifact in provenance_files.items():
        if not artifact.is_file():
            raise ValueError(f"{path}: provenance artifact is missing: {artifact}")
        if sha256_file(artifact) != environment[key]:
            raise ValueError(f"{path}: {key} does not match {artifact}")

    return environment, started, completed


def read_schedule(
    data: Path,
    *,
    run_started: dt.datetime,
    run_completed: dt.datetime,
) -> dict[str, dt.datetime]:
    """Validate the exact 60-row arm order and backend mapping."""

    path = data / "schedule.tsv"
    if not path.is_file():
        raise ValueError(f"missing schedule: {path}")
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if tuple(reader.fieldnames or ()) != SCHEDULE_FIELDS:
            raise ValueError(f"{path}: unexpected header {reader.fieldnames}")
        rows = list(reader)

    expected = expected_arms()
    if len(rows) != len(expected):
        raise ValueError(f"{path}: expected {len(expected)} rows, found {len(rows)}")

    times: dict[str, dt.datetime] = {}
    previous_started = run_started
    for line_no, (row, spec) in enumerate(zip(rows, expected, strict=True), start=2):
        try:
            observed = ArmSpec(
                model=row["model"],
                prompt_tokens=int(row["prompt_tokens"]),
                round=int(row["round"]),
                position=int(row["position"]),
                arm=row["arm"],
                backend=row["backend"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_no}: invalid numeric field") from exc
        if observed != spec:
            raise ValueError(f"{path}:{line_no}: expected {spec}, found {observed}")
        if spec.label in times:
            raise ValueError(f"{path}:{line_no}: duplicate arm label {spec.label}")
        started = parse_utc(row["started_utc"], f"{path}:{line_no}:started_utc")
        if not (run_started <= started <= run_completed):
            raise ValueError(f"{path}:{line_no}: arm timestamp is outside the run")
        if started < previous_started:
            raise ValueError(f"{path}:{line_no}: arm timestamps are not nondecreasing")
        times[spec.label] = started
        previous_started = started
    return times


def require_bool(record: dict[str, Any], key: str, expected: bool, context: str) -> None:
    """Require an exact JSON boolean."""

    if record.get(key) is not expected:
        raise ValueError(f"{context}: {key} must be {str(expected).lower()}")


def require_nonnegative_number(record: dict[str, Any], key: str, context: str) -> float:
    """Require a finite nonnegative JSON number and return it."""

    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context}: {key} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{context}: {key} must be finite and nonnegative")
    return value


def read_hostgate(
    data: Path,
    *,
    schedule_times: dict[str, dt.datetime],
    run_started: dt.datetime,
    run_completed: dt.datetime,
) -> None:
    """Require one clean, accepted gate record for every arm plus preflight."""

    path = data / "hostgate.jsonl"
    if not path.is_file():
        raise ValueError(f"missing host-gate evidence: {path}")
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{path}:{line_no}: blank evidence line")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_no}: evidence must be a JSON object")
        records.append(record)

    expected_labels = ["run-preflight", *(spec.label for spec in expected_arms())]
    labels = [record.get("label") for record in records]
    if len(records) != len(expected_labels):
        raise ValueError(
            f"{path}: expected {len(expected_labels)} records, found {len(records)}"
        )
    if labels != expected_labels:
        raise ValueError(f"{path}: gate labels are missing, duplicated, or out of order")

    by_label = {str(record["label"]): record for record in records}
    preflight = by_label["run-preflight"]
    require_bool(preflight, "port_free_before_wait", True, f"{path}:run-preflight")
    require_bool(preflight, "port_free_after_wait", True, f"{path}:run-preflight")
    if preflight.get("port") != 18080:
        raise ValueError(f"{path}:run-preflight: unexpected port")
    preflight_started = parse_utc(preflight.get("started_utc"), f"{path}:run-preflight:started")
    preflight_completed = parse_utc(
        preflight.get("completed_utc"), f"{path}:run-preflight:completed"
    )
    if not (run_started <= preflight_started <= preflight_completed <= run_completed):
        raise ValueError(f"{path}:run-preflight: timestamps are outside the run")

    previous_completed = preflight_completed
    for spec in expected_arms():
        context = f"{path}:{spec.label}"
        record = by_label[spec.label]
        if record.get("port") != 18080:
            raise ValueError(f"{context}: unexpected port")
        require_bool(record, "port_free_before_wait", True, context)
        require_bool(record, "port_free_after_wait", True, context)
        require_bool(record, "accepted_quiet", True, context)
        require_bool(record, "ci_job_running", False, context)
        if record.get("busy_procs") != []:
            raise ValueError(f"{context}: busy_procs must be empty")
        if record.get("foreign_model_procs") != []:
            raise ValueError(f"{context}: foreign_model_procs must be empty")
        for key in (
            "gate_wait_s",
            "driver_wait_s",
            "mem_wait_s",
            "load1",
            "mem_available_gib",
            "mem_floor_gib",
            "nvrm_window_5m",
            "nvrm_window_max",
            "nvrm_total_since_boot",
            "nvrm_total_max",
            "quiet_rechecks",
        ):
            require_nonnegative_number(record, key, context)
        if float(record["mem_available_gib"]) < float(record["mem_floor_gib"]):
            raise ValueError(f"{context}: memory evidence is below the gate floor")
        if int(record["nvrm_window_5m"]) >= int(record["nvrm_window_max"]):
            raise ValueError(f"{context}: NVRM window evidence exceeds the gate")
        if int(record["nvrm_total_since_boot"]) >= int(record["nvrm_total_max"]):
            raise ValueError(f"{context}: cumulative NVRM evidence exceeds the gate")
        started = parse_utc(record.get("started_utc"), f"{context}:started")
        completed = parse_utc(record.get("completed_utc"), f"{context}:completed")
        if not (run_started <= started <= completed <= schedule_times[spec.label]):
            raise ValueError(f"{context}: gate did not complete before its scheduled arm")
        if started < previous_completed:
            raise ValueError(f"{context}: gate timestamps are overlapping or out of order")
        previous_completed = completed


def read_server_logs(data: Path) -> None:
    """Require one nonempty server log with the requested backend per arm."""

    specs = expected_arms()
    expected_names = {f"server-{spec.label}.log" for spec in specs}
    actual_paths = {path.name: path for path in data.glob("server-*.log")}
    if set(actual_paths) != expected_names:
        missing = sorted(expected_names - set(actual_paths))
        extra = sorted(set(actual_paths) - expected_names)
        raise ValueError(f"server log set mismatch; missing={missing}, extra={extra}")

    for spec in specs:
        path = actual_paths[f"server-{spec.label}.log"]
        if path.stat().st_size == 0:
            raise ValueError(f"{path}: server log is empty")
        with path.open(encoding="utf-8", errors="replace") as source:
            marker = source.readline().rstrip("\n")
        expected_marker = f"requested_decode_storage_backend={spec.backend}"
        if marker != expected_marker:
            raise ValueError(f"{path}: expected backend marker {expected_marker!r}")


def read_runs(data: Path) -> tuple[Run, ...]:
    """Read the exact client-log set and reject duplicate, failed, or missing rows."""

    specs = expected_arms()
    expected_names = {spec.client_name for spec in specs}
    actual_paths = {path.name: path for path in data.glob("client-*.txt")}
    if set(actual_paths) != expected_names:
        missing = sorted(expected_names - set(actual_paths))
        extra = sorted(set(actual_paths) - expected_names)
        raise ValueError(f"client log set mismatch; missing={missing}, extra={extra}")

    runs: list[Run] = []
    keys: set[tuple[str, int, int, str, int]] = set()
    for spec in specs:
        path = actual_paths[spec.client_name]
        match = LOG_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected client log name: {path.name}")
        rows: dict[int, Run] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if row := TABLE_ROW.match(line):
                concurrency = int(row["concurrency"])
                if concurrency in rows:
                    raise ValueError(f"{path}: duplicate concurrency {concurrency}")
                if int(row["failed"]) != 0:
                    raise ValueError(f"{path}: concurrency {concurrency} has failures")
                aggregate = float(row["aggregate"])
                if not math.isfinite(aggregate) or aggregate <= 0:
                    raise ValueError(f"{path}: concurrency {concurrency} has invalid throughput")
                rows[concurrency] = Run(
                    model=match["model"],
                    prompt_tokens=int(match["prompt"]),
                    round=int(match["round"]),
                    arm=match["arm"],
                    concurrency=concurrency,
                    aggregate_tok_s=aggregate,
                )
        if set(rows) != set(CONCURRENCIES):
            raise ValueError(
                f"{path}: expected concurrency rows {CONCURRENCIES}, found {sorted(rows)}"
            )
        for concurrency in CONCURRENCIES:
            run = rows[concurrency]
            key = (run.model, run.prompt_tokens, run.round, run.arm, run.concurrency)
            if key in keys:
                raise ValueError(f"{path}: duplicate measurement key {key}")
            keys.add(key)
            runs.append(run)

    expected_keys = {
        (spec.model, spec.prompt_tokens, spec.round, spec.arm, concurrency)
        for spec in specs
        for concurrency in CONCURRENCIES
    }
    if keys != expected_keys:
        raise ValueError("client measurement keys do not match the 180-row protocol")
    return tuple(runs)


def validate_protocol(data: Path) -> Protocol:
    """Validate every acceptance artifact before allowing any calculation."""

    if not (data / "complete").is_file():
        raise ValueError(f"missing completion marker: {data / 'complete'}")
    environment, run_started, run_completed = read_environment(data)
    schedule_times = read_schedule(
        data,
        run_started=run_started,
        run_completed=run_completed,
    )
    read_hostgate(
        data,
        schedule_times=schedule_times,
        run_started=run_started,
        run_completed=run_completed,
    )
    read_server_logs(data)
    runs = read_runs(data)
    return Protocol(environment, schedule_times, runs)


def percent_delta(value: float, baseline: float) -> float:
    """Return the percentage delta of ``value`` from ``baseline``."""

    return 100.0 * (value / baseline - 1.0)


def summarize(protocol: Protocol) -> None:
    """Print per-cell medians, paired deltas, null spreads, and decisions."""

    by_key = {
        (run.model, run.prompt_tokens, run.concurrency, run.round, run.arm): run.aggregate_tok_s
        for run in protocol.runs
    }
    decisions: dict[str, tuple[bool, bool]] = {}

    print("| Model | Prompt | C | paged-a median | dense median | paged-b median | dense paired delta | null spread | Cell |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for model in MODELS:
        any_dense_win = False
        any_paged_win = False
        for prompt in PROMPTS:
            for concurrency in CONCURRENCIES:
                arm_values = {
                    arm: [
                        by_key[(model, prompt, concurrency, round_no, arm)]
                        for round_no in ROUNDS
                    ]
                    for arm in ARMS
                }
                paired = [
                    percent_delta(dense, paged)
                    for dense, paged in zip(
                        arm_values["dense"], arm_values["paged-a"], strict=True
                    )
                ]
                null = [
                    percent_delta(paged_b, paged_a)
                    for paged_b, paged_a in zip(
                        arm_values["paged-b"], arm_values["paged-a"], strict=True
                    )
                ]

                paired_median = statistics.median(paired)
                null_low, null_high = min(null), max(null)
                dense_win = paired_median > max(0.0, null_high)
                paged_win = paired_median < min(0.0, null_low)
                any_dense_win |= dense_win
                any_paged_win |= paged_win
                verdict = "dense" if dense_win else "paged" if paged_win else "unresolved"
                medians = {
                    arm: statistics.median(values) for arm, values in arm_values.items()
                }
                print(
                    f"| {model} | {prompt} | {concurrency} | {medians['paged-a']:.1f} | "
                    f"{medians['dense']:.1f} | {medians['paged-b']:.1f} | "
                    f"{paired_median:+.2f}% ({min(paired):+.2f}..{max(paired):+.2f}) | "
                    f"{null_low:+.2f}..{null_high:+.2f}% | {verdict} |"
                )
        decisions[model] = (any_dense_win, any_paged_win)

    print()
    for model, (any_dense_win, any_paged_win) in decisions.items():
        use_dense = any_dense_win and not any_paged_win
        print(
            f"{model}: supports_paged_decode_backend={'false' if use_dense else 'true'} "
            f"(dense_clear_cell={any_dense_win}, paged_clear_cell={any_paged_win})"
        )


def main(argv: list[str] | None = None) -> int:
    """Validate the sweep, then print its statistical summary."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args(argv)
    summarize(validate_protocol(args.data_dir.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
