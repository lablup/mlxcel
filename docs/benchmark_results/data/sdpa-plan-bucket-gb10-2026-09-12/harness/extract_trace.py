#!/usr/bin/env python3
"""Compress a MLXCEL_SDPA_PLAN_DEBUG trace to distinct lines with frequencies.

A raw trace is 88 to 97 percent redundant (3,444 lines with 125 distinct on the
non-speculative control), and the three raw traces were 70 percent of this
directory. They are kept as frequency extracts instead.

`sort -u` would NOT do: it discards the occurrence counts, and the call totals
the record cites (1,401 calls at block 4, 1,204 on the control) are exactly
those counts. The extract therefore keeps `<count>\\t<line>`, so every quantity
the record cites stays derivable:

  total calls            sum of counts
  plan builds            sum of counts over lines with built=1
  bucketed calls         sum of counts over lines with bucket=1
  shape classes          group the lines, as summarize_trace.py does
  distinct field values  unchanged, one row per distinct line
  resident plans at end  max of plans=, which is monotonic within a run

What does NOT survive is per-call ordering, so a claim about which call came
first, or about the sequence within one round, cannot be read off an extract.
No number in the record depends on that: the "three builds per verify round"
figure is derived from per-class build counts over a known round count, not
from the sequence. `--keep-order-head N` preserves the first N raw lines for
the cases where a reader wants to see the shape of the run's opening.

Usage: extract_trace.py SOURCE [--keep-order-head N] > extract.txt
"""
import re
import sys
from collections import Counter

LINE = re.compile(r"\[mlxcel-sdpa\] .*")


def fields(line):
    d = {}
    for kv in line.split("[mlxcel-sdpa] ", 1)[-1].split():
        k, _, v = kv.partition("=")
        d[k] = v
    return d


def main():
    src = sys.argv[1]
    head_n = 0
    if "--keep-order-head" in sys.argv:
        head_n = int(sys.argv[sys.argv.index("--keep-order-head") + 1])

    raw = []
    for ln in open(src, errors="replace"):
        m = LINE.search(ln)
        if m:
            raw.append(m.group(0).rstrip())

    counts = Counter(raw)
    builds = sum(c for line, c in counts.items() if fields(line).get("built") == "1")
    bucketed = sum(c for line, c in counts.items() if fields(line).get("bucket") == "1")
    plans = [int(fields(line).get("plans", 0) or 0) for line in counts]

    name = src.rsplit("/", 1)[-1]
    print(f"# MLXCEL_SDPA_PLAN_DEBUG frequency extract of {name}")
    print("# Format: '<count>\\t<trace line>', one row per distinct line, count-descending.")
    print(f"# calls={sum(counts.values())} distinct={len(counts)} plan_builds={builds} "
          f"bucketed_calls={bucketed} max_resident_plans={max(plans) if plans else 0}")
    print("# Ordering is not preserved; counts, distinct counts and maxima are. See")
    print("# extract_trace.py for which record numbers depend on which of those.")
    if head_n:
        print(f"# First {head_n} lines in original order follow, prefixed 'order:'.")
        for line in raw[:head_n]:
            print(f"order:\t{line}")
    for line, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"{c}\t{line}")


if __name__ == "__main__":
    main()
