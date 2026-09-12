#!/usr/bin/env python3
"""Group MLXCEL_SDPA_PLAN_DEBUG lines into shape classes and report which key
fields move from one call of that class to the next."""
import re, sys
from collections import OrderedDict

LINE = re.compile(r"\[mlxcel-sdpa\] (.*)")
recs = []
for path in sys.argv[1:]:
    for ln in open(path, errors="replace"):
        m = LINE.search(ln)
        if not m:
            continue
        d = {}
        for kv in m.group(1).split():
            k, _, v = kv.partition("=")
            d[k] = v
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

print(f"total cuDNN SDPA calls traced: {len(recs)}; distinct shape classes: {len(groups)}")
for c, ds in groups.items():
    q, causal, sinks, decode, rowstride = c
    print(f"\n== class q={q} causal={causal} sinks={sinks} decode={decode} k_rowstride={rowstride}: {len(ds)} calls")
    for f in ("k_len", "k_extent", "mask_cols", "mask_rowstride", "mask_lead", "bucket", "bucket_to"):
        vals = [d.get(f) for d in ds]
        distinct = len(set(vals))
        head = ",".join(vals[:6])
        tail = ",".join(vals[-3:])
        print(f"   {f:15s} distinct={distinct:5d}  first: {head} ... last: {tail}")
    built = sum(1 for d in ds if d.get("built") == "1")
    print(f"   plan builds in this class: {built} of {len(ds)} calls")
    print(f"   final resident plans: {ds[-1].get('plans')}")
