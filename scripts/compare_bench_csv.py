#!/usr/bin/env python3
"""Compare two benchmark CSVs and refuse the joins that are not comparable.

Every wrong number the 2026-09 VLM campaign put in front of a reader came from
pairing two rows that were not measuring the same thing, and each time the
comparison itself was ad-hoc, so a precondition checked in one place was missing
in the next. This tool holds all of them in one place:

  harness   a VLM row is never paired with a text row
  prompt    prompt_tokens must agree within a tolerance
  commit    both sides must carry the same mlxcel_commit, unless the comparison
            is against a reference runtime, where the runtime commit is the
            thing being compared and only the dates have to be close
  roster    non-generative checkpoints are excluded by an explicit list rather
            than by whoever wrote the join that day

Names are resolved through docs/model-catalog.tsv, whose `aliases` column is
`;`-separated. Splitting it on `,` silently loses thirteen keys, two of which
were the control checkpoints of that campaign.

Dropped rows are always reported with the reason. A pair count on its own says
nothing about what it left out, and "n=48" was itself one of the wrong numbers.

Examples:

    # runtime against its own earlier sweep: commits must differ, everything
    # else must match, and the tool says which rows moved
    scripts/compare_bench_csv.py --before benchmarks/metal_m5max_vlm_2026-09-06.csv \\
        --after benchmarks/metal_m5max_vlm_2026-09-09.csv --allow-commit-change

    # runtime against a reference: same host, same day
    scripts/compare_bench_csv.py --before benchmarks/pylm_m5max_vlm_2026-09-09.csv \\
        --after benchmarks/metal_m5max_vlm_2026-09-09.csv --reference
"""

import argparse
import csv
import glob
import os
import statistics
import sys

CATALOG = "docs/model-catalog.tsv"

# Checkpoints that a generation harness loads but does not meaningfully measure.
# An embedder or reranker driven through a decode loop produces a number, and
# that number is not a decode rate. Excluding them is a policy, so it lives here
# rather than in whichever comparison is being written.
NON_GENERATIVE = ("embedding", "rerank", "-embed-", "colqwen", "colsmol")


def load_aliases(path=CATALOG):
    """Map every historical name to the directory name in use now."""
    alias = {}
    if not os.path.exists(path):
        return alias
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            local = row["local_name"]
            alias[local] = local
            for name in (row.get("aliases") or "").split(";"):
                if name.strip():
                    alias[name.strip()] = local
    return alias


def harness_of(path):
    """VLM sweeps carry `_vlm_` in the filename; nothing else distinguishes them."""
    return "vlm" if "_vlm_" in os.path.basename(path) else "text"


def load_rows(path, alias):
    rows = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = row.get("model")
            if not name:
                continue
            try:
                decode = float(row["decode_tok_s"])
                prompt = float(row["prompt_tokens"])
            except (KeyError, TypeError, ValueError):
                continue
            key = alias.get(name, name)
            rows[key] = {
                "decode": decode,
                "prompt": prompt,
                "prefill": row.get("prefill_tok_s") or "",
                "commit": (row.get("mlxcel_commit") or "").replace("-dirty", ""),
                "date": row.get("date") or "",
                "raw_name": name,
            }
    return rows


def newer_readings_elsewhere(before_path, after_path, names, alias, before_rows):
    """Rows in `before` that some other CSV has measured more recently.

    This is the check that matters most. `mistral-small-4-119b-2603-4bit` read
    0.35x cross-host and was taken for the largest hardware deficit in the set,
    because one host's sweep file still held a pre-fix number while a newer
    per-model CSV sitting beside it held the current one. Comparing two sweep
    files is correct and still misses that, so the scan is over every CSV for
    the same host and harness, not just the two being compared.
    """
    base = os.path.basename(before_path)
    host = "m5max" if "m5max" in base else "m1ultra" if "m1ultra" in base else None
    runtime = "pylm" if base.startswith("pylm") else "metal"
    if host is None:
        return {}
    harness = harness_of(before_path)
    after_date = ""
    with open(after_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            after_date = max(after_date, row.get("date") or "")

    stale = {}
    for path in sorted(glob.glob(f"benchmarks/{runtime}_{host}_*.csv")):
        if os.path.abspath(path) in (os.path.abspath(before_path), os.path.abspath(after_path)):
            continue
        if harness_of(path) != harness:
            continue
        try:
            rows = load_rows(path, alias)
        except (OSError, csv.Error):
            continue
        for name, row in rows.items():
            if name not in names:
                continue
            # Only a reading taken after the baseline row and before the run
            # being judged supersedes anything. Without the first half of that
            # the scan reports every older file as if it were newer.
            if row["date"] <= before_rows[name]["date"]:
                continue
            if after_date and row["date"] >= after_date:
                continue
            prev = stale.get(name)
            if prev is None or row["date"] > prev[1]:
                stale[name] = (row["decode"], row["date"], os.path.basename(path))
    return stale


def quantiles(values):
    ordered = sorted(values)
    n = len(ordered)
    return (
        statistics.median(ordered),
        ordered[n // 4],
        ordered[(3 * n) // 4],
        ordered[0],
        ordered[-1],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", required=True, help="baseline CSV")
    ap.add_argument("--after", required=True, help="CSV being judged")
    ap.add_argument(
        "--reference",
        action="store_true",
        help="the two sides are different runtimes, so compare dates rather than commits",
    )
    ap.add_argument(
        "--allow-commit-change",
        action="store_true",
        help="the two sides are the same runtime at different commits, which is the point",
    )
    ap.add_argument("--prompt-tolerance", type=float, default=0.10)
    ap.add_argument("--max-date-gap-days", type=int, default=1)
    ap.add_argument("--include-non-generative", action="store_true")
    ap.add_argument("--moved-threshold", type=float, default=0.10)
    args = ap.parse_args()

    h_before, h_after = harness_of(args.before), harness_of(args.after)
    if h_before != h_after:
        print(
            f"error: refusing to pair a {h_before} harness with a {h_after} one.\n"
            f"  before: {args.before}\n  after:  {args.after}\n"
            "  These measure different workloads. A VLM row compared against a text\n"
            "  baseline produced a 61% figure that was really 157%.",
            file=sys.stderr,
        )
        return 2

    alias = load_aliases()
    before = load_rows(args.before, alias)
    after = load_rows(args.after, alias)

    shared = sorted(set(before) & set(after))
    paired, dropped = [], {"prompt": [], "commit": [], "non_generative": [], "date": []}

    for name in shared:
        b, a = before[name], after[name]
        if not args.include_non_generative and any(t in name.lower() for t in NON_GENERATIVE):
            dropped["non_generative"].append(name)
            continue
        if b["prompt"] > 0 and abs(a["prompt"] - b["prompt"]) / b["prompt"] > args.prompt_tolerance:
            dropped["prompt"].append((name, b["prompt"], a["prompt"]))
            continue
        if args.reference:
            if b["date"] and a["date"] and b["date"] != a["date"]:
                gap = abs(int(a["date"].replace("-", "")) - int(b["date"].replace("-", "")))
                if gap > args.max_date_gap_days:
                    dropped["date"].append((name, b["date"], a["date"]))
                    continue
        elif not args.allow_commit_change:
            if b["commit"] and a["commit"] and b["commit"] != a["commit"]:
                dropped["commit"].append((name, b["commit"], a["commit"]))
                continue
        if b["decode"] > 0:
            paired.append((a["decode"] / b["decode"], name, b["decode"], a["decode"]))

    print(f"before: {args.before}  ({len(before)} measured)")
    print(f"after:  {args.after}  ({len(after)} measured)")
    print(f"harness: {h_after}    names in common: {len(shared)}")
    print()

    if not paired:
        print("no comparable pairs. Every reason is listed below.")
    else:
        ratios = [p[0] for p in paired]
        med, q1, q3, lo, hi = quantiles(ratios)
        print(f"pairs {len(paired)}   median {med:.3f}x   quartiles {q1:.3f}/{q3:.3f}   range {lo:.3f}-{hi:.3f}x")
        moved = [p for p in paired if p[0] > 1 + args.moved_threshold or p[0] < 1 - args.moved_threshold]
        print(f"moved more than {args.moved_threshold:.0%}: {len(moved)}")
        for ratio, name, b, a in sorted(moved, key=lambda t: -t[0]):
            print(f"  {ratio:6.2f}x  {name:<44} {b:>9.2f} -> {a:>9.2f}")

    print()
    total_dropped = sum(len(v) for v in dropped.values())
    print(f"dropped {total_dropped} of {len(shared)} shared names:")
    if dropped["non_generative"]:
        print(f"  non-generative ({len(dropped['non_generative'])}): {', '.join(dropped['non_generative'])}")
    if dropped["prompt"]:
        print(f"  prompt length beyond {args.prompt_tolerance:.0%} ({len(dropped['prompt'])}):")
        for name, b, a in sorted(dropped["prompt"], key=lambda t: -abs(t[2] - t[1]) / max(t[1], 1)):
            print(f"    {name:<44} {b:>7.0f} -> {a:>7.0f}   delta {a - b:+.0f}")
    if dropped["commit"]:
        print(f"  different mlxcel_commit ({len(dropped['commit'])}), pass --allow-commit-change if that is the comparison:")
        for name, b, a in dropped["commit"][:10]:
            print(f"    {name:<44} {b} -> {a}")
        if len(dropped["commit"]) > 10:
            print(f"    ... and {len(dropped['commit']) - 10} more")
    if dropped["date"]:
        print(f"  measured more than {args.max_date_gap_days} day(s) apart ({len(dropped['date'])}):")
        for name, b, a in dropped["date"][:10]:
            print(f"    {name:<44} {b} vs {a}")
        if len(dropped["date"]) > 10:
            print(f"    ... and {len(dropped['date']) - 10} more")

    if paired:
        paired_names = {p[1] for p in paired}
        newer = newer_readings_elsewhere(args.before, args.after, paired_names, alias, before)
        superseded = []
        for name, (decode, date, src) in newer.items():
            was = before[name]["decode"]
            if was > 0 and abs(decode - was) / was > args.moved_threshold:
                superseded.append((name, was, decode, date, src))
        if superseded:
            print()
            print(f"WARNING: {len(superseded)} of the {len(paired)} baselines are superseded elsewhere.")
            print("  A newer reading for the same checkpoint, host and harness exists in another")
            print("  CSV, so these ratios are measuring a stale row rather than a change.")
            for name, was, now, date, src in sorted(superseded, key=lambda t: -abs(t[2] - t[1]) / t[1]):
                print(f"    {name:<44} {was:>9.2f} -> {now:>9.2f}  ({date}, {src})")

    only_before = sorted(set(before) - set(after))
    only_after = sorted(set(after) - set(before))
    if only_before or only_after:
        print()
        print(f"measured on one side only: {len(only_before)} before, {len(only_after)} after")
        print("  These are not failures and not comparisons; they are unmeasured on the other side.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
