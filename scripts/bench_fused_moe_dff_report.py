#!/usr/bin/env python3
"""Summarize a scripts/bench_fused_moe_dff.py runs.csv as Markdown tables (issue #2540).

Per (harness, model, cell): per-arm n / median / range / per-run list over the
measured (status == ok) rows, and per-round paired deltas against the fused
arm of the same round. A point is a win or a loss only if the paired
fused-over-gather range excludes zero AND its median exceeds the null arm's
spread (the largest absolute paired null delta); otherwise it is a tie.

Usage: python3 scripts/bench_fused_moe_dff_report.py <runs.csv>
"""
import csv
import statistics
import sys
from collections import defaultdict

DFF = {
    "qwen3-30b-a3b-4bit": 768,
    "lfm2-8b-a1b-4bit": 1792,
    "phi-3.5-moe-4bit": 6400,
    "llama-4-scout-17b-4bit": 8192,
    "mixtral-8x7b-4bit": 14336,
}


def pct(x):
    return f"{x * 100:+.2f}%"


def load(path):
    groups = defaultdict(lambda: defaultdict(dict))  # key -> round -> arm -> row
    other = defaultdict(lambda: defaultdict(int))
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["harness"], row["model"], row["cell"])
            if row["status"] == "ok":
                groups[key][int(row["round"])][row["arm"]] = row
            else:
                other[key][row["status"]] += 1
    return groups, other


def summarize(groups, other):
    out = {}
    for key, rounds in groups.items():
        arms = defaultdict(list)
        gens = defaultdict(set)
        for r in sorted(rounds):
            for arm, row in rounds[r].items():
                arms[arm].append(float(row["rate"]))
                gens[arm].add(row["gen_tokens"])
        paired = {"gather": [], "null": []}
        fused_over_gather = []
        for r in sorted(rounds):
            got = rounds[r]
            if "fused" not in got:
                continue
            base = float(got["fused"]["rate"])
            for arm in ("gather", "null"):
                if arm in got:
                    paired[arm].append(float(got[arm]["rate"]) / base - 1.0)
            if "gather" in got:
                fused_over_gather.append(base / float(got["gather"]["rate"]) - 1.0)
        out[key] = {
            "arms": arms, "gens": gens, "paired": paired,
            "fog": fused_over_gather, "other": dict(other.get(key, {})),
        }
    return out


def other_rows(other):
    if not other:
        return "-"
    return ", ".join(f"{count} {status}" for status, count in sorted(other.items()))


def verdict(s):
    fog, null = s["fog"], s["paired"]["null"]
    if not fog or not null:
        return "NA"
    med = statistics.median(fog)
    floor = max(abs(x) for x in null)
    excludes_zero = min(fog) > 0 or max(fog) < 0
    if excludes_zero and abs(med) > floor:
        return "win" if med > 0 else "loss"
    return "tie"


def main():
    path = sys.argv[1]
    groups, other = load(path)
    summ = summarize(groups, other)

    for harness in ("primary", "secondary"):
        keys = [k for k in summ if k[0] == harness]
        if not keys:
            continue
        keys.sort(key=lambda k: (DFF.get(k[1], 0), k[2]))
        print(f"\n## {harness} harness: per-arm rows\n")
        print("| model (Dff) | cell | arm | n | median tok/s | range | generated tokens | per run |")
        print("|---|---|---|---|---|---|---|---|")
        for k in keys:
            s = summ[k]
            for arm in ("fused", "gather", "null"):
                v = s["arms"].get(arm, [])
                if not v:
                    print(f"| {k[1]} ({DFF.get(k[1], '?')}) | {k[2]} | {arm} | 0 | NA | NA | NA | NA |")
                    continue
                print(
                    f"| {k[1]} ({DFF.get(k[1], '?')}) | {k[2]} | {arm} | {len(v)} | "
                    f"{statistics.median(v):.2f} | {min(v):.2f}-{max(v):.2f} | "
                    f"{'/'.join(sorted(s['gens'][arm]))} | {' '.join(f'{x:.2f}' for x in v)} |"
                )
        print(f"\n## {harness} harness: paired deltas per round (against the fused arm)\n")
        print("| model (Dff) | cell | gather vs fused, median (range) | null vs fused, median (range) | "
              "fused over gather, median (range) | null floor | verdict | non-ok rows |")
        print("|---|---|---|---|---|---|---|---|")
        for k in keys:
            s = summ[k]
            g, n, fog = s["paired"]["gather"], s["paired"]["null"], s["fog"]

            def cell(v):
                if not v:
                    return "NA"
                return f"{pct(statistics.median(v))} ({pct(min(v))} to {pct(max(v))})"

            floor = f"{max(abs(x) for x in n) * 100:.2f}%" if n else "NA"
            print(
                f"| {k[1]} ({DFF.get(k[1], '?')}) | {k[2]} | {cell(g)} | {cell(n)} | {cell(fog)} | "
                f"{floor} | {verdict(s)} | {other_rows(s['other'])} |"
            )

    # Crossover from the primary harness, one table per cell. Llama 4 keeps its
    # own SwitchGLU (gather_qmm only), so its two arms run the same path; the
    # interpolation is reported with and without that point.
    no_fused_path = {"llama-4-scout-17b-4bit"}
    cells = sorted({k[2] for k in summ if k[0] == "primary"})
    for cell in cells:
        pts = []
        for k, s in summ.items():
            if k[0] == "primary" and k[2] == cell and k[1] in DFF and s["fog"]:
                pts.append((DFF[k[1]], k[1], 1.0 + statistics.median(s["fog"]), verdict(s)))
        pts.sort()
        print(f"\n## crossover, primary harness, cell {cell}\n")
        print("| Dff | model | fused/gather (median of paired ratios) | verdict | fused kernel runs |")
        print("|---|---|---|---|---|")
        for d, name, ratio, v in pts:
            print(f"| {d} | {name} | {ratio:.4f} | {v} | {'no' if name in no_fused_path else 'yes'} |")
        for label, use in (("all points", pts),
                           ("fused-path models only", [p for p in pts if p[1] not in no_fused_path])):
            crossings = []
            for (d0, _, r0, _), (d1, _, r1, _) in zip(use, use[1:]):
                if (r0 - 1.0) * (r1 - 1.0) < 0:
                    crossings.append((d0, d1, d0 + (1.0 - r0) * (d1 - d0) / (r1 - r0)))
            if crossings:
                text = "; ".join(f"between {d0} and {d1}: {x:.0f}" for d0, d1, x in crossings)
            else:
                text = "no adjacent pair straddles 1.0"
            print(f"\ninterpolated crossings ({label}): {text}")
        wins = [d for d, name, _, v in pts if v == "win" and name not in no_fused_path]
        print(f"\nlargest measured Dff that still wins: {max(wins) if wins else 'none'}")


if __name__ == "__main__":
    main()
