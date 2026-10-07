#!/usr/bin/env python3
"""Summarize `mtp_batched_rounds.py` records: per round, the mean aggregate
rate of each arm, MTP vs the mean of its classic brackets, and the null arm
(close vs open). Also checks that every arm of a run produced the same bytes.

  summarize.py rounds-31b-b2/run.jsonl [...]
"""
import json
import statistics
import sys


def mean_rate(rec, key):
    runs = rec.get("runs") or []
    return statistics.mean(r[key] for r in runs) if runs else None


def main():
    for path in sys.argv[1:]:
        recs = [json.loads(line) for line in open(path)]
        rounds = sorted({r["round"] for r in recs})
        shas = {tuple(run["sha"]) for r in recs for run in r.get("runs", [])}
        print(f"== {path}: batch {recs[0]['batch']}, identical bytes across arms: {len(shas) == 1}")
        print("round | classic open | MTP | classic close | MTP vs classic | null | "
              "batched rows | gate timed out | thrash")
        for rnd in rounds:
            arms = {r["arm"].split("-", 1)[1]: r for r in recs if r["round"] == rnd}
            o, m, c = arms["classic-open"], arms["mtp"], arms["classic-close"]
            ro, rm, rc = (mean_rate(x, "agg_tok_s") for x in (o, m, c))
            if None in (ro, rm, rc):
                print(f"{rnd} | {ro} | {rm} | {rc} | error: "
                      f"{[x.get('error') for x in (o, m, c) if x.get('error')]}")
                continue
            base = (ro + rc) / 2
            gate = [x["gate_timed_out"] for x in (o, m, c)]
            print(f"{rnd} | {ro:.2f} | {rm:.2f} | {rc:.2f} | {100 * (rm / base - 1):+.1f}% | "
                  f"{100 * (rc / ro - 1):+.1f}% | {m['batched_rows']} | {gate} | "
                  f"{m.get('thrash_abort')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
