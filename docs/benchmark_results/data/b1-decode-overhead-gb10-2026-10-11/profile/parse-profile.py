#!/usr/bin/env python3
"""Render issue #2227's measured-request host-timer summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PREFIX = "[b1-profile] "


def records(path: Path) -> list[dict[str, object]]:
    found: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(PREFIX):
            found.append(json.loads(line.removeprefix(PREFIX)))
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    found = records(args.log)
    if not found:
        raise SystemExit(f"no {PREFIX.strip()} records in {args.log}")
    measured = found[-1]
    print(
        f"run={measured['run']} ticks={measured['ticks']} "
        f"tick_avg_us={measured['tick_avg_us']}"
    )
    print("stage\tcalls\ttotal_ns\ttick_pct\tavg_ns_per_call")
    stages = dict(measured["stages"])
    stages["event_send"] = measured["event_send"]
    for name, raw in stages.items():
        stage = dict(raw)
        calls = int(stage["calls"])
        total = int(stage["total_ns"])
        average = total / calls if calls else 0.0
        print(f"{name}\t{calls}\t{total}\t{stage['tick_pct']}\t{average:.3f}")


if __name__ == "__main__":
    main()
