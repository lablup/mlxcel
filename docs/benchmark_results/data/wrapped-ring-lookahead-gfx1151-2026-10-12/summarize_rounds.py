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
"""Summarize session.sh output for issue #2239: arm medians and paired deltas.

Usage: summarize_rounds.py rounds.csv
"""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict


def pct(a: float, b: float) -> float:
    return 100.0 * (a - b) / b


def main() -> int:
    rows: dict[str, dict[int, float]] = defaultdict(dict)
    with open(sys.argv[1]) as fh:
        for row in csv.DictReader(fh):
            if row["status"] != "ok" or not row["decode_tok_s"]:
                continue
            rows[row["arm"]][int(row["round"])] = float(row["decode_tok_s"])

    base = rows.get("base", {})
    for arm in ("base", "new", "new-sync", "base-null"):
        data = rows.get(arm, {})
        vals = [data[r] for r in sorted(data)]
        if not vals:
            print(f"{arm}: no ok rounds")
            continue
        line = f"{arm}: " + ", ".join(f"{v:.2f}" for v in vals) + f" | median {statistics.median(vals):.2f}"
        if arm not in ("base",):
            shared = sorted(set(base) & set(data))
            if shared:
                deltas = [pct(data[r], base[r]) for r in shared]
                line += (
                    f" | vs base {', '.join(f'{d:+.2f}%' for d in deltas)}"
                    f" (median {statistics.median(deltas):+.2f}%,"
                    f" range {min(deltas):+.2f} to {max(deltas):+.2f})"
                )
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
