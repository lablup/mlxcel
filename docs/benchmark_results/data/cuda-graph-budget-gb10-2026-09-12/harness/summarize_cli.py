#!/usr/bin/env python3
"""Turn bench_cli.py jsonl into the #1799 table.

Per config: n, tok/s mean (min to max), vs off, mean accepted, device sync
ms/round (min to max), drafter host build ms/round, verify host build
ms/round, load1 range, greedy ids == off.
"""
import json
import sys
from collections import OrderedDict


def stats(xs):
    return (sum(xs) / len(xs), min(xs), max(xs)) if xs else (float("nan"),) * 3


def main():
    recs = []
    for p in sys.argv[1:]:
        for line in open(p):
            if line.strip():
                recs.append(json.loads(line))
    recs = [r for r in recs if not r.get("warmup") and "tok_s" in r]
    by = OrderedDict()
    for r in recs:
        by.setdefault(r["cfg"], []).append(r)
    off_ids = None
    if "off" in by:
        off_ids = by["off"][0].get("ids")
    off_mean = stats([r["tok_s"] for r in by["off"]])[0] if "off" in by else None
    print("| config | n | tok/s mean (min to max) | vs off | accepted/round | round wall ms | device sync ms/round (min to max) | draft host ms/round | verify host ms/round | load1 (min to max) | ids == off |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for cfg, rs in by.items():
        t = stats([r["tok_s"] for r in rs])
        vs = f"{t[0]/off_mean:.2f}x" if off_mean and cfg != "off" else ""
        ld = stats([r["load1_before"] for r in rs])
        ds = [r["diag"] for r in rs if r.get("diag", {}).get("rounds")]
        if ds:
            acc = stats([d["accepted"] / d["rounds"] for d in ds])[0]
            sync = stats([d["verify_sync_ms"] / d["rounds"] for d in ds])
            draft = stats([d["draft_ms"] / d["rounds"] for d in ds])[0]
            vg = stats([d["verify_graph_ms"] / d["rounds"] for d in ds])[0]
            wall = stats([d["decode_ms"] / d["rounds"] for d in ds])[0]
            cells = f"{acc:.2f} | {wall:.1f} | {sync[0]:.1f} ({sync[1]:.1f} to {sync[2]:.1f}) | {draft:.1f} | {vg:.1f}"
        else:
            cells = " | | | | "
        same = ""
        if off_ids is not None:
            same = "yes" if all(r.get("ids") == off_ids for r in rs) else "NO"
        print(f"| {cfg} | {len(rs)} | {t[0]:.2f} ({t[1]:.2f} to {t[2]:.2f}) | {vs} | {cells} | {ld[1]:.2f} to {ld[2]:.2f} | {same} |")


if __name__ == "__main__":
    main()
