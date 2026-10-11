#!/usr/bin/env python3
"""Validate and summarize issue #2227's paired acceptance records.

Exit codes:
  0: every selected model is complete and all threshold cells pass
  1: every selected model is complete, but at least one cell fails
  2: an input violates the measurement protocol
  3: an input is absent or contains a valid but incomplete matrix
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PROMPTS = (256, 8192)
ROUNDS = tuple(range(5))
MAX_TOKENS = 128
ARMS = ("base-cli", "base-cli-null", "new-srv-dense")
EXPECTED_PATH = {
    "base-cli": "cli",
    "base-cli-null": "cli",
    "new-srv-dense": "server",
}
HEADER_ARMS = {
    "base-cli": ["--which", "base", "--path", "cli"],
    "new-srv-dense": [
        "--which",
        "new",
        "--path",
        "server",
        "--decode-storage",
        "dense",
    ],
}
SPECS = {
    "qwen3-1.7b-4bit": ROOT / "single-qwen3-1.7b-4bit.jsonl",
    "llama-3.2-1b-instruct-4bit": ROOT
    / "single-llama-3.2-1b-instruct-4bit.jsonl",
}


class InvalidInput(Exception):
    """The input exists but violates the acceptance protocol."""


@dataclass(frozen=True)
class PendingInput:
    model: str
    path: Path
    detail: str


@dataclass(frozen=True)
class CompleteInput:
    model: str
    path: Path
    records: dict[tuple[str, int, int, str], dict[str, Any]]


@dataclass(frozen=True)
class Cell:
    model: str
    prompt: int
    candidate: tuple[float, ...]
    null: tuple[float, ...]

    @property
    def candidate_median(self) -> float:
        return statistics.median(self.candidate)

    @property
    def null_median(self) -> float:
        return statistics.median(self.null)

    @property
    def passes(self) -> bool:
        return self.candidate_median >= -1.0

    @property
    def ranges_overlap(self) -> bool:
        return max(min(self.candidate), min(self.null)) <= min(
            max(self.candidate), max(self.null)
        )


def finite_number(value: Any, *, name: str, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidInput(f"{name} is not numeric: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise InvalidInput(f"{name} is not finite: {value!r}")
    if positive and result <= 0:
        raise InvalidInput(f"{name} is not positive: {value!r}")
    if not positive and result < 0:
        raise InvalidInput(f"{name} is negative: {value!r}")
    return result


def read_objects(path: Path, model: str) -> list[dict[str, Any]] | PendingInput:
    if not path.exists():
        return PendingInput(model, path, "file does not exist yet")
    text = path.read_text(encoding="utf-8")
    lines = [(number, line) for number, line in enumerate(text.splitlines(), 1) if line]
    if not lines:
        return PendingInput(model, path, "file is empty")

    objects: list[dict[str, Any]] = []
    for index, (number, line) in enumerate(lines):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as error:
            if index == len(lines) - 1:
                return PendingInput(model, path, f"last line is incomplete: {error.msg}")
            raise InvalidInput(f"{path}:{number}: invalid JSON: {error.msg}") from error
        if not isinstance(obj, dict):
            raise InvalidInput(f"{path}:{number}: expected a JSON object")
        objects.append(obj)
    return objects


def validate_header(header: dict[str, Any], model: str, path: Path) -> None:
    if header.get("kind") != "header":
        raise InvalidInput(f"{path}: first record is not the header")
    if Path(str(header.get("model", ""))).name != model:
        raise InvalidInput(
            f"{path}: header model {header.get('model')!r} does not match {model!r}"
        )
    if header.get("arms") != HEADER_ARMS:
        raise InvalidInput(f"{path}: header arms differ from the required CLI/server arms")
    if header.get("prompt_tokens") != list(PROMPTS):
        raise InvalidInput(f"{path}: prompt lengths are not {list(PROMPTS)!r}")
    if header.get("max_tokens") != MAX_TOKENS:
        raise InvalidInput(f"{path}: max_tokens is not {MAX_TOKENS}")
    if header.get("rounds") != len(ROUNDS):
        raise InvalidInput(f"{path}: rounds is not {len(ROUNDS)}")
    if header.get("sdpa_deterministic") is not None:
        raise InvalidInput(f"{path}: performance run forced deterministic SDPA")
    if Path(str(header.get("bin", ""))).name != "pick.sh":
        raise InvalidInput(f"{path}: header does not use the recorded pick.sh dispatcher")
    if not isinstance(header.get("host"), str) or not header["host"]:
        raise InvalidInput(f"{path}: missing host identity")
    finite_number(header.get("time"), name=f"{path}: header time")


def validate_row(row: dict[str, Any], model: str, path: Path, number: int) -> None:
    prefix = f"{path}: data row {number}"
    arm = row.get("arm")
    if arm not in ARMS:
        raise InvalidInput(f"{prefix}: unexpected arm {arm!r}")
    if row.get("label") != arm:
        raise InvalidInput(f"{prefix}: label does not equal arm")
    if row.get("path") != EXPECTED_PATH[arm]:
        raise InvalidInput(
            f"{prefix}: {arm} path is {row.get('path')!r}, expected {EXPECTED_PATH[arm]!r}"
        )
    if row.get("model") != model:
        raise InvalidInput(f"{prefix}: model is {row.get('model')!r}, expected {model!r}")
    if row.get("prompt_target_len") not in PROMPTS:
        raise InvalidInput(f"{prefix}: unexpected prompt_target_len")
    if row.get("prompt_tokens") != row["prompt_target_len"]:
        raise InvalidInput(f"{prefix}: generated prompt length differs from target")
    if row.get("round") not in ROUNDS:
        raise InvalidInput(f"{prefix}: round is outside {ROUNDS!r}")
    if row.get("max_tokens") != MAX_TOKENS:
        raise InvalidInput(f"{prefix}: max_tokens is not {MAX_TOKENS}")
    if row.get("generated_tokens") != MAX_TOKENS:
        raise InvalidInput(f"{prefix}: generated_tokens is not {MAX_TOKENS}")
    if row.get("decode_storage") != "dense":
        raise InvalidInput(f"{prefix}: decode_storage is not dense")
    if row.get("prefill_chunk") != 2048:
        raise InvalidInput(f"{prefix}: prefill_chunk is not the default 2048")
    if row.get("ci_job_running") is not False:
        raise InvalidInput(f"{prefix}: host gate recorded a running CI job")

    for key in ("ttft_ms", "prefill_tok_s", "decode_ms", "decode_tok_s", "started"):
        finite_number(row.get(key), name=f"{prefix}: {key}")
    finite_number(row.get("gate_wait_s"), name=f"{prefix}: gate_wait_s", positive=False)


def load(path: Path, model: str) -> CompleteInput | PendingInput:
    parsed = read_objects(path, model)
    if isinstance(parsed, PendingInput):
        return parsed
    if not parsed:
        return PendingInput(model, path, "file has no records")

    validate_header(parsed[0], model, path)
    records: dict[tuple[str, int, int, str], dict[str, Any]] = {}
    for number, row in enumerate(parsed[1:], 2):
        if row.get("kind") == "header":
            raise InvalidInput(f"{path}:{number}: duplicate header")
        validate_row(row, model, path, number)
        key = (model, row["prompt_target_len"], row["round"], row["arm"])
        if key in records:
            raise InvalidInput(f"{path}:{number}: duplicate record for {key!r}")
        records[key] = row

    expected = {
        (model, prompt, round_number, arm)
        for prompt in PROMPTS
        for round_number in ROUNDS
        for arm in ARMS
    }
    present = set(records)
    extra = present - expected
    if extra:
        raise InvalidInput(f"{path}: unexpected matrix keys: {sorted(extra)!r}")
    missing = expected - present
    if missing:
        return PendingInput(
            model,
            path,
            f"valid partial matrix: {len(present)}/{len(expected)} rows; "
            f"missing {len(missing)}",
        )
    if len(parsed) != 1 + len(expected):
        raise InvalidInput(f"{path}: expected exactly one header and {len(expected)} rows")
    return CompleteInput(model, path, records)


def paired_delta(value: float, reference: float) -> float:
    return 100.0 * (value / reference - 1.0)


def cells(loaded: CompleteInput) -> list[Cell]:
    result: list[Cell] = []
    for prompt in PROMPTS:
        candidate: list[float] = []
        null: list[float] = []
        for round_number in ROUNDS:
            def rate(arm: str) -> float:
                row = loaded.records[(loaded.model, prompt, round_number, arm)]
                return float(row["decode_tok_s"])

            baseline = rate("base-cli")
            candidate.append(paired_delta(rate("new-srv-dense"), baseline))
            null.append(paired_delta(rate("base-cli-null"), baseline))
        result.append(Cell(loaded.model, prompt, tuple(candidate), tuple(null)))
    return result


def percent(value: float) -> str:
    return f"{value:+.2f}%"


def interval(values: tuple[float, ...]) -> str:
    return f"{percent(min(values))}..{percent(max(values))}"


def print_summary(complete: list[CompleteInput], pending: list[PendingInput]) -> bool:
    all_cells = [cell for loaded in complete for cell in cells(loaded)]
    print("| Model | Prompt | new-srv-dense vs base-cli | Null arm | Threshold | Noise label |")
    print("|---|---:|---:|---:|---|---|")
    for cell in all_cells:
        threshold = "PASS" if cell.passes else "FAIL"
        noise = "unresolved (ranges overlap)" if cell.ranges_overlap else "separated"
        candidate = f"{percent(cell.candidate_median)} ({interval(cell.candidate)})"
        null = f"{percent(cell.null_median)} ({interval(cell.null)})"
        print(
            f"| {cell.model} | {cell.prompt} | {candidate} | {null} | "
            f"{threshold} | {noise} |"
        )
        rounds = ", ".join(percent(value) for value in cell.candidate)
        null_rounds = ", ".join(percent(value) for value in cell.null)
        print(f"<!-- {cell.model} {cell.prompt} candidate rounds: {rounds} -->")
        print(f"<!-- {cell.model} {cell.prompt} null rounds: {null_rounds} -->")

    if pending:
        print("\nPENDING: the acceptance matrix is incomplete.")
        for item in pending:
            print(f"- {item.model}: {item.detail} ({item.path})")
    elif len(all_cells) == 4:
        accepted = len(all_cells) == 4 and all(cell.passes for cell in all_cells)
        verdict = "PASS" if accepted else "FAIL"
        print(f"\nAcceptance: {verdict} ({sum(cell.passes for cell in all_cells)}/4 cells >= -1.00%)")
    else:
        verdict = "PASS" if all(cell.passes for cell in all_cells) else "FAIL"
        print(
            f"\nSelected subset: {verdict} "
            f"({sum(cell.passes for cell in all_cells)}/{len(all_cells)} cells >= -1.00%). "
            "Full four-cell acceptance was not evaluated."
        )
    return bool(all_cells) and all(cell.passes for cell in all_cells)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=tuple(SPECS),
        help="analyze one completed model while the other measurement is pending",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = {args.only: SPECS[args.only]} if args.only else SPECS
    complete: list[CompleteInput] = []
    pending: list[PendingInput] = []
    try:
        for model, path in selected.items():
            result = load(path, model)
            if isinstance(result, PendingInput):
                pending.append(result)
            else:
                complete.append(result)
    except (InvalidInput, OSError) as error:
        print(f"INVALID: {error}", file=sys.stderr)
        return 2

    all_pass = print_summary(complete, pending)
    if pending:
        return 3
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
