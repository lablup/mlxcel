#!/usr/bin/env python3
"""Turn a `sweep_server_widths.py` JSONL into the record's table.

Reports ranges, not ratios alone. A ratio of means hides whether two arms
actually separate, and on this host a single decode run is bimodal by about
25% even at pinned clocks (#755), so "1.32x" is only meaningful next to the
min-to-max spread that produced it.

The classic bracket is checked first: if the opening and closing classic arms
do not overlap, the session drifted under the arms between them and every
comparison in the middle inherits that doubt. The summary says so instead of
printing a clean table over a dirty run.

  summarize.py run.jsonl [--label "affine (qwen3.5-4b-4bit)"]
"""
import argparse
import json
import sys


def rates(rec):
    return sorted(r["e2e_tok_s"] for r in rec.get("runs", []))


def mean(xs):
    return sum(xs) / len(xs)


def overlaps(a, b):
    return a[0] <= b[-1] and b[0] <= a[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    recs = [json.loads(line) for line in open(a.jsonl) if line.strip()]
    by_tag = {r["arm"]: r for r in recs}

    opening = by_tag.get("classic-open")
    closing = by_tag.get("classic-close")
    classic_rates = []
    for arm in (opening, closing):
        if arm and rates(arm):
            classic_rates += rates(arm)

    print(f"## {a.label}" if a.label else "## sweep")
    print()
    if opening and closing and rates(opening) and rates(closing):
        o, c = rates(opening), rates(closing)
        verdict = (
            "overlap, so the session did not drift under the arms they bracket"
            if overlaps(o, c)
            else "DO NOT OVERLAP: the session drifted, treat every middle arm as suspect"
        )
        print(
            f"Classic bracket: opening {min(o):.2f} to {max(o):.2f}, "
            f"closing {min(c):.2f} to {max(c):.2f} tok/s. They {verdict}."
        )
    else:
        print("Classic bracket incomplete; no drift check is possible for this run.")
    print()

    baseline = mean(classic_rates) if classic_rates else None
    baseline_max = max(classic_rates) if classic_rates else None

    print("| arm | n | e2e tok/s mean (min to max) | vs classic | separates from classic | NVRM delta | resolved block_size |")
    print("| --- | ---: | ---: | ---: | --- | ---: | --- |")
    for rec in recs:
        rs = rates(rec)
        if not rs:
            print(f"| {rec['arm']} | 0 | ERROR {rec.get('error', '')} | | | {rec.get('nvrm_delta', '?')} | |")
            continue
        m = mean(rs)
        ratio = f"{m / baseline:.2f}x" if baseline else ""
        if rec["arm"].startswith("classic") or not classic_rates:
            sep = ""
        elif min(rs) > baseline_max:
            sep = "yes, above every classic run"
        elif max(rs) < min(classic_rates):
            sep = "yes, below every classic run"

        else:
            sep = "no, ranges overlap"
        sizes = rec.get("diagnostics_block_sizes") or []
        spec = rec.get("speculative_line") or ""
        resolved = f"{','.join(str(s) for s in sizes)}" if sizes else ("classic" if rec["width"] == "classic" else "?")
        print(
            f"| {rec['arm']} | {len(rs)} | {m:.2f} ({min(rs):.2f} to {max(rs):.2f}) | "
            f"{ratio} | {sep} | +{rec.get('nvrm_delta', '?')} | {resolved} |"
        )

    print()
    texts = {}
    for rec in recs:
        for r in rec.get("runs", []):
            texts.setdefault(rec["arm"], set()).add(r["text"])
    classic_text = None
    for arm in ("classic-open", "classic-close"):
        if arm in texts and len(texts[arm]) == 1:
            classic_text = next(iter(texts[arm]))
            break
    if classic_text is None:
        print("Greedy identity: no single classic completion to compare against.")
    else:
        for arm, ts in texts.items():
            if len(ts) != 1:
                print(f"Greedy identity: {arm} produced {len(ts)} distinct completions across its runs.")
                continue
            same = next(iter(ts)) == classic_text
            print(f"Greedy identity: {arm} {'==' if same else '!='} classic")

    print()
    for rec in recs:
        if rec.get("speculative_line"):
            print(f"{rec['arm']}: {rec['speculative_line']}")


if __name__ == "__main__":
    main()
