#!/usr/bin/env python3
"""Turn bench_bd.py jsonl into a markdown table, grouped by (model, tag, cfg)."""
import json, sys
from collections import OrderedDict

def stats(xs):
    return (sum(xs) / len(xs), min(xs), max(xs)) if xs else (float("nan"),) * 3

recs = []
for p in sys.argv[1:]:
    for line in open(p):
        if line.strip():
            recs.append(json.loads(line))
recs = [r for r in recs if not r.get("warmup") and "decode_tok_s" in r]
groups = OrderedDict()
for r in recs:
    groups.setdefault((r.get("model", "").rsplit("/", 1)[-1], r.get("tag", "")), OrderedDict()).setdefault(r["cfg"], []).append(r)
for (model, tag), by in groups.items():
    print(f"\n### {model} ({tag})\n")
    base = stats([r["decode_tok_s"] for r in by["default"]])[0] if "default" in by else None
    pbase = stats([r["prefill_ms"] for r in by["default"]])[0] if "default" in by else None
    print("| config | n | prompt tok | decode tok/s mean (min to max) | vs default | decode ms/200 tok | prefill ms mean (min to max) | vs default | prefill tok/s | MLX peak GB (min to max) | load1 (min to max) | CI job during run |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cfg, rs in by.items():
        d = stats([r["decode_tok_s"] for r in rs]); dm = stats([r["decode_ms"] for r in rs])
        p = stats([r["prefill_ms"] for r in rs]); pt = stats([r["prefill_tok_s"] for r in rs])
        pk = stats([r.get("peak_gb", float("nan")) for r in rs]); ld = stats([r["load1_before"] for r in rs])
        vs = f"{(d[0]/base-1)*100:+.1f}%" if base and cfg != "default" else ""
        pvs = f"{(p[0]/pbase-1)*100:+.1f}%" if pbase and cfg != "default" else ""
        ptok = sorted({r.get("prompt_tokens") for r in rs})
        print(f"| {cfg} | {len(rs)} | {'/'.join(map(str, ptok))} | {d[0]:.2f} ({d[1]:.2f} to {d[2]:.2f}) | {vs} | {dm[0]:.0f} | {p[0]:.1f} ({p[1]:.1f} to {p[2]:.1f}) | {pvs} | {pt[0]:.0f} | {pk[0]:.2f} ({pk[1]:.2f} to {pk[2]:.2f}) | {ld[1]:.2f} to {ld[2]:.2f} | {sum(1 for r in rs if r.get('ci_job_running'))} of {len(rs)} |")
