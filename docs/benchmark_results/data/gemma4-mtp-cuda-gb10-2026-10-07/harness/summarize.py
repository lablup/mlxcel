#!/usr/bin/env python3
"""Summarize mtp_rounds.py output: per-round paired deltas and the null arm.

For each round r, classic is the mean of the open and close brackets; the MTP
delta is mtp / classic - 1 and the null delta is close / open - 1. An MTP
delta whose range overlaps the null arm's is unresolved.

  summarize.py run.jsonl
"""
import json
import statistics
import sys


def mean_rate(rec):
    return statistics.mean(r["e2e_tok_s"] for r in rec["runs"])


def main(path):
    recs = [json.loads(line) for line in open(path)]
    rounds = sorted({r["round"] for r in recs})
    print("| round | classic open | MTP | classic close | MTP vs classic | null (close vs open) | accepted/proposed | tokens/verify |")
    print("|---|---|---|---|---|---|---|---|")
    deltas, nulls, shas = [], [], set()
    for rnd in rounds:
        arms = {r["arm"].split("-", 1)[1]: r for r in recs if r["round"] == rnd}
        mtp_key = next(k for k in arms if k.startswith("mtp"))
        o, m, c = mean_rate(arms["classic-open"]), mean_rate(arms[mtp_key]), mean_rate(arms["classic-close"])
        d = m / ((o + c) / 2) - 1
        n = c / o - 1
        deltas.append(d)
        nulls.append(n)
        diag = [x for x in arms[mtp_key]["diagnostics"] if x.get("proposed_tokens")]
        acc = sum(x["accepted_draft_tokens"] for x in diag) / max(1, sum(x["proposed_tokens"] for x in diag))
        per_verify = sum(x["generated_tokens"] for x in diag) / max(1, sum(x["rounds"] for x in diag))
        for a in arms.values():
            shas |= {r["sha"] for r in a["runs"]}
        print(f"| {rnd} | {o:.2f} | {m:.2f} | {c:.2f} | {d:+.1%} | {n:+.1%} | {acc:.2f} | {per_verify:.2f} |")
    print()
    print(f"MTP delta range {min(deltas):+.1%} to {max(deltas):+.1%}; null range {min(nulls):+.1%} to {max(nulls):+.1%}")
    print(f"distinct output sha across all arms: {sorted(shas)}")
    print(f"ci_job_running during arms: {sorted({str(r.get('ci_job_running')) for r in recs})}")


if __name__ == "__main__":
    main(sys.argv[1])
