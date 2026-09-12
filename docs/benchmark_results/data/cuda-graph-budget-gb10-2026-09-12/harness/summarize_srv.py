#!/usr/bin/env python3
import json, sys
from collections import OrderedDict
def stats(xs): return (sum(xs)/len(xs), min(xs), max(xs)) if xs else (float("nan"),)*3
recs = [json.loads(l) for p in sys.argv[1:] for l in open(p) if l.strip()]
recs = [r for r in recs if r.get("levels")]
groups = OrderedDict()
for r in recs:
    g = groups.setdefault((r["model"].rsplit("/",1)[-1], r.get("tag","")), OrderedDict())
    for l in r["levels"]:
        g.setdefault((l["conc"], r["cfg"]), []).append((l, r["load1_before"], r.get("ci_job_running", False)))
for (model, tag), g in groups.items():
    print(f"\n### {model} ({tag})\n")
    print("| concurrency | config | n | aggregate tok/s mean (min to max) | vs default | per-request decode tok/s mean | TTFT ms mean (p95 mean) | load1 (min to max) | CI job during run |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (conc, cfg), rows in g.items():
        base = g.get((conc, "default"))
        bm = stats([l["aggregate_tok_s"] for l, _, _ in base])[0] if base else None
        ag = stats([l["aggregate_tok_s"] for l, _, _ in rows]); de = stats([l["decode_tok_s_mean"] for l, _, _ in rows])
        tt = stats([l["ttft_ms_mean"] for l, _, _ in rows]); tp = stats([l["ttft_ms_p95"] for l, _, _ in rows]); ld = stats([x for _, x, _ in rows])
        vs = f"{(ag[0]/bm-1)*100:+.1f}%" if bm and cfg != "default" else ""
        print(f"| {conc} | {cfg} | {len(rows)} | {ag[0]:.2f} ({ag[1]:.2f} to {ag[2]:.2f}) | {vs} | {de[0]:.2f} | {tt[0]:.0f} ({tp[0]:.0f}) | {ld[1]:.2f} to {ld[2]:.2f} | {sum(1 for _, _, c in rows if c)} of {len(rows)} |")
