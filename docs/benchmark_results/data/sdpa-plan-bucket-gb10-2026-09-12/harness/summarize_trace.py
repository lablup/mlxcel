#!/usr/bin/env python3
"""Group MLXCEL_SDPA_PLAN_DEBUG lines into shape classes and report which key
fields move from one call of that class to the next."""
import re, sys
from collections import OrderedDict

LINE = re.compile(r"\[mlxcel-sdpa\] (.*)")
COUNTED = re.compile(r"^(\d+)\t")

# Accepts a raw trace or a frequency extract from extract_trace.py. An extract
# row is "<count>\t<line>" and stands for `count` identical calls, so it is
# expanded to a weight rather than to `count` copies. Ordering is absent from an
# extract, so the per-field first/last display is suppressed for those and the
# final resident plan count comes from max(plans) instead of the last record,
# which is equivalent because plans is monotonic within a run.
recs = []
ordered = True
for path in sys.argv[1:]:
    for ln in open(path, errors="replace"):
        if ln.startswith("#"):
            continue
        weight = 1
        cm = COUNTED.match(ln)
        if cm:
            weight = int(cm.group(1))
            ordered = False
            ln = ln.split("\t", 1)[1]
        elif ln.startswith("order:\t"):
            continue  # the head sample duplicates rows counted below
        m = LINE.search(ln)
        if not m:
            continue
        d = {}
        for kv in m.group(1).split():
            k, _, v = kv.partition("=")
            d[k] = v
        d["_w"] = weight
        recs.append(d)

if not recs:
    print("no trace lines")
    sys.exit(0)

# A shape class is what does NOT move within a layer group.
def cls(d):
    return (d["q"], d["causal"], d["sinks"], d["decode"], d["k_rowstride"])

groups = OrderedDict()
for d in recs:
    groups.setdefault(cls(d), []).append(d)

def w(ds):
    return sum(d["_w"] for d in ds)


total = w(recs)
src = "raw trace" if ordered else "frequency extract (ordering absent)"
print(f"total cuDNN SDPA calls traced: {total}; distinct shape classes: {len(groups)}  [{src}]")
for c, ds in groups.items():
    q, causal, sinks, decode, rowstride = c
    print(f"\n== class q={q} causal={causal} sinks={sinks} decode={decode} k_rowstride={rowstride}: {w(ds)} calls")
    for f in ("k_len", "k_extent", "mask_cols", "mask_rowstride", "mask_lead", "bucket", "bucket_to"):
        vals = [d.get(f) for d in ds]
        distinct = len(set(vals))
        if ordered:
            head = ",".join(vals[:6])
            tail = ",".join(vals[-3:])
            print(f"   {f:15s} distinct={distinct:5d}  first: {head} ... last: {tail}")
        else:
            print(f"   {f:15s} distinct={distinct:5d}")
    built = sum(d["_w"] for d in ds if d.get("built") == "1")
    bucketed = sum(d["_w"] for d in ds if d.get("bucket") == "1")
    plans = max(int(d.get("plans", 0) or 0) for d in ds)
    print(f"   plan builds in this class: {built} of {w(ds)} calls; bucketed calls: {bucketed}")
    print(f"   max resident plans seen: {plans}")

print(f"\ntotals: plan builds {sum(d['_w'] for d in recs if d.get('built') == '1')}, "
      f"bucketed calls {sum(d['_w'] for d in recs if d.get('bucket') == '1')}, "
      f"max resident plans {max(int(d.get('plans', 0) or 0) for d in recs)}")
