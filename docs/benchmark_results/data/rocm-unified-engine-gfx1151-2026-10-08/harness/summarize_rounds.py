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
"""Summarize bench_rounds.sh output: medians, paired deltas and the null spread.

For every (model, prompt_tokens) cell: the per-arm decode and prefill medians,
the paired per-round delta of `new` against `base` (median and range, in
percent), and the same for `base-null` against `base`, which is the method's
noise floor. A cell passes ADR 0007's threshold when the median paired decode
delta of `new` is no worse than -1.0 percent.

Usage: summarize_rounds.py rounds.csv [--threshold 1.0] [--markdown]
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict


def pct(a: float, b: float) -> float:
    return 100.0 * (a - b) / b


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    cells: dict[tuple[str, str], dict[str, dict[str, dict[str, float]]]] = defaultdict(lambda: defaultdict(dict))
    with open(args.csv) as fh:
        for row in csv.DictReader(fh):
            if row["status"] != "ok":
                continue
            cell = (row["model"], row["prompt_tokens"])
            cells[cell][row["arm"]][row["round"]] = {
                "decode": float(row["decode_tok_s"]),
                "prefill": float(row["prefill_tok_s"]),
                "peak": float(row["peak_gb"] or 0),
            }

    def paired(cell, arm, metric):
        base = cells[cell].get("base", {})
        other = cells[cell].get(arm, {})
        rounds = sorted(set(base) & set(other), key=int)
        deltas = [pct(other[r][metric], base[r][metric]) for r in rounds]
        return deltas

    def fmt_delta(deltas):
        if not deltas:
            return "n/a"
        med = statistics.median(deltas)
        return f"{med:+.2f} % ({min(deltas):+.2f}..{max(deltas):+.2f}, n={len(deltas)})"

    if args.markdown:
        print("| Model | Prompt | base decode tok/s | new decode tok/s | new vs base, decode | null vs base, decode | base prefill tok/s | new prefill tok/s | new vs base, prefill | null vs base, prefill | Verdict |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
    for cell in sorted(cells, key=lambda c: (c[0].lower(), int(c[1]))):
        model, ptok = cell
        arms = cells[cell]
        med = {
            arm: {m: statistics.median(v[m] for v in arms[arm].values()) for m in ("decode", "prefill", "peak")}
            for arm in arms
        }
        d_new = paired(cell, "new", "decode")
        d_null = paired(cell, "base-null", "decode")
        p_new = paired(cell, "new", "prefill")
        p_null = paired(cell, "base-null", "prefill")
        if d_new:
            verdict = "pass" if statistics.median(d_new) >= -args.threshold else "MISS"
            if d_null and statistics.median(d_new) < -args.threshold and min(d_null) <= statistics.median(d_new):
                verdict = "MISS (inside null)"
        else:
            verdict = "n/a"
        b = med.get("base", {}); n = med.get("new", {})
        if args.markdown:
            print(
                f"| {model} | {ptok} | {b.get('decode', float('nan')):.2f} | {n.get('decode', float('nan')):.2f} | {fmt_delta(d_new)} | {fmt_delta(d_null)} | "
                f"{b.get('prefill', float('nan')):.1f} | {n.get('prefill', float('nan')):.1f} | {fmt_delta(p_new)} | {fmt_delta(p_null)} | {verdict} |"
            )
        else:
            print(f"{model} pp{ptok}:")
            for arm in ("base", "new", "base-null"):
                if arm in med:
                    vals = ", ".join(f"{arms[arm][r]['decode']:.2f}" for r in sorted(arms[arm], key=int))
                    print(f"  {arm:10s} decode {med[arm]['decode']:.2f} [{vals}]  prefill {med[arm]['prefill']:.1f}  peak {med[arm]['peak']:.2f} GB")
            print(f"  decode  new vs base {fmt_delta(d_new)}   null vs base {fmt_delta(d_null)}")
            print(f"  prefill new vs base {fmt_delta(p_new)}   null vs base {fmt_delta(p_null)}")
            print(f"  verdict {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
