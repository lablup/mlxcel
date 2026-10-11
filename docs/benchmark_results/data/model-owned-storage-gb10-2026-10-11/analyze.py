#!/usr/bin/env python3
"""Summarize the model-owned decode-storage paired/null serving sweep."""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from pathlib import Path


DATA = Path(__file__).resolve().parent
LOG_NAME = re.compile(
    r"client-(?P<model>gemma3|llama4)-p(?P<prompt>512|4096)-"
    r"r(?P<round>[1-5])-(?P<arm>paged-a|dense|paged-b)\.txt"
)
TABLE_ROW = re.compile(
    r"^\s*(?P<concurrency>1|4|8)\s+\d+\s+(?P<failed>\d+)\s+"
    r"(?:\d+(?:\.\d+)?\s+){3}(?P<aggregate>\d+(?:\.\d+)?)\s*$"
)


@dataclass(frozen=True)
class Run:
    """Aggregate throughput from one fresh-server arm."""

    model: str
    prompt_tokens: int
    round: int
    arm: str
    concurrency: int
    aggregate_tok_s: float


def percent_delta(value: float, baseline: float) -> float:
    """Return the percentage delta of ``value`` from ``baseline``."""

    return 100.0 * (value / baseline - 1.0)


def read_runs() -> list[Run]:
    """Read all expected client logs and reject incomplete or failed arms."""

    runs: list[Run] = []
    for path in sorted(DATA.glob("client-*.txt")):
        match = LOG_NAME.fullmatch(path.name)
        if match is None:
            continue
        rows = []
        for line in path.read_text().splitlines():
            if row := TABLE_ROW.match(line):
                if int(row["failed"]) != 0:
                    raise ValueError(f"{path}: concurrency {row['concurrency']} has failures")
                rows.append(
                    Run(
                        model=match["model"],
                        prompt_tokens=int(match["prompt"]),
                        round=int(match["round"]),
                        arm=match["arm"],
                        concurrency=int(row["concurrency"]),
                        aggregate_tok_s=float(row["aggregate"]),
                    )
                )
        if len(rows) != 3:
            raise ValueError(f"{path}: expected three concurrency rows, found {len(rows)}")
        runs.extend(rows)

    expected = 2 * 2 * 5 * 3 * 3
    if len(runs) != expected:
        raise ValueError(f"expected {expected} measurements, found {len(runs)}")
    return runs


def main() -> int:
    """Print per-cell medians, paired deltas, null spreads, and family decisions."""

    by_key = {
        (run.model, run.prompt_tokens, run.concurrency, run.round, run.arm): run.aggregate_tok_s
        for run in read_runs()
    }
    decisions: dict[str, tuple[bool, bool]] = {}

    print("| Model | Prompt | C | paged-a median | dense median | paged-b median | dense paired delta | null spread | Cell |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for model in ("gemma3", "llama4"):
        any_dense_win = False
        any_paged_win = False
        for prompt in (512, 4096):
            for concurrency in (1, 4, 8):
                arm_values = {
                    arm: [
                        by_key[(model, prompt, concurrency, round_no, arm)]
                        for round_no in range(1, 6)
                    ]
                    for arm in ("paged-a", "dense", "paged-b")
                }
                paired = []
                null = []
                for index in range(5):
                    paired.append(
                        percent_delta(
                            arm_values["dense"][index], arm_values["paged-a"][index]
                        )
                    )
                    null.append(
                        percent_delta(
                            arm_values["paged-b"][index], arm_values["paged-a"][index]
                        )
                    )

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
