#!/usr/bin/env python3
"""Graph-API and kernel accounting from two nsys profiles of one config at two token
budgets (differencing: load, warm-up and prefill cancel). Prints per generated token:
kernel launches, GPU kernel ms, and the CUDA graph API call counts and host ms; plus
the absolute per-run cudaGraphInstantiate count/time (instantiation is front-loaded).
Usage: nsys_graph.py LONG.nsys-rep SHORT.nsys-rep --tokens-long 400 --tokens-short 200 --label X
"""
import argparse, csv, subprocess
def stats_csv(rep, report):
    out = subprocess.run(["nsys", "stats", "--report", report, "--format", "csv", "--force-export=true", rep], capture_output=True, text=True)
    lines = out.stdout.splitlines()
    i = next(n for n, l in enumerate(lines) if l.startswith("Time (%)"))
    return [r for r in csv.DictReader(lines[i:]) if r.get("Name")]
ap = argparse.ArgumentParser(); ap.add_argument("long"); ap.add_argument("short")
ap.add_argument("--tokens-long", type=float, required=True); ap.add_argument("--tokens-short", type=float, required=True)
ap.add_argument("--label", default="")
a = ap.parse_args(); dt = a.tokens_long - a.tokens_short
kl = {r["Name"]: r for r in stats_csv(a.long, "cuda_gpu_kern_sum")}; ks = {r["Name"]: r for r in stats_csv(a.short, "cuda_gpu_kern_sum")}
inst = sum(int(kl[n]["Instances"]) for n in kl) - sum(int(ks[n]["Instances"]) for n in ks)
ns = sum(float(kl[n]["Total Time (ns)"]) for n in kl) - sum(float(ks[n]["Total Time (ns)"]) for n in ks)
print(f"== {a.label}: delta tokens {dt:.0f}; per token: {inst/dt:.1f} kernel launches, {ns/dt/1e6:.3f} ms GPU kernel time")
al = {r["Name"]: r for r in stats_csv(a.long, "cuda_api_sum")}; asx = {r["Name"]: r for r in stats_csv(a.short, "cuda_api_sum")}
print("| CUDA API | calls/token | host ms/token | calls (long run, absolute) | host ms (long run, absolute) |")
print("|---|---|---|---|---|")
tot_c = tot_ms = 0.0
for name in sorted(set(al) | set(asx)):
    cl = int(al[name]["Num Calls"]) if name in al else 0; cs = int(asx[name]["Num Calls"]) if name in asx else 0
    tl = float(al[name]["Total Time (ns)"]) if name in al else 0.0; ts = float(asx[name]["Total Time (ns)"]) if name in asx else 0.0
    tot_c += cl - cs; tot_ms += (tl - ts) / 1e6
    if any(k in name for k in ("Graph", "LaunchKernel", "StreamSynchronize", "EventSynchronize", "EventRecord", "StreamWaitEvent", "Memcpy", "EventQuery")):
        print(f"| `{name}` | {(cl-cs)/dt:.2f} | {(tl-ts)/dt/1e6:.3f} | {cl} | {tl/1e6:.1f} |")
print(f"| all CUDA API calls | {tot_c/dt:.1f} | {tot_ms/dt:.3f} | | |")
