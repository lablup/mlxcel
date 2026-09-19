#!/usr/bin/env python3
"""Per-verify-round kernel and NVTX accounting from two nsys profiles of one
configuration at two token budgets.

Differencing the two removes the model load, the prefill and the exactness
probe, which are identical in both, so what is left is the decode work of the
extra rounds. Round counts come from each run's own `DFlash:` line (or the
generated-token count on a classic arm).

Usage: nsys_round.py LONG.nsys-rep SHORT.nsys-rep --rounds-long N --rounds-short M --label X
"""
import argparse, csv, subprocess


def stats_csv(rep, report):
    out = subprocess.run(
        ["nsys", "stats", "--report", report, "--format", "csv",
         "--force-export=true", rep],
        capture_output=True, text=True)
    lines = out.stdout.splitlines()
    try:
        i = next(n for n, l in enumerate(lines) if l.startswith("Time (%)"))
    except StopIteration:
        return []
    return [r for r in csv.DictReader(lines[i:]) if r.get("Name")]


ap = argparse.ArgumentParser()
ap.add_argument("long"); ap.add_argument("short")
ap.add_argument("--rounds-long", type=float, required=True)
ap.add_argument("--rounds-short", type=float, required=True)
ap.add_argument("--label", default="")
a = ap.parse_args()
dr = a.rounds_long - a.rounds_short

kl = {r["Name"]: r for r in stats_csv(a.long, "cuda_gpu_kern_sum")}
ks = {r["Name"]: r for r in stats_csv(a.short, "cuda_gpu_kern_sum")}
inst = sum(int(kl[n]["Instances"]) for n in kl) - sum(int(ks[n]["Instances"]) for n in ks)
ns = sum(float(kl[n]["Total Time (ns)"]) for n in kl) - sum(float(ks[n]["Total Time (ns)"]) for n in ks)
print(f"== {a.label}: delta rounds {dr:.0f}")
print(f"   GPU kernel time per round: {ns/dr/1e6:.2f} ms over {inst/dr:.0f} launches")

att = [n for n in set(kl) | set(ks) if "sdpa" in n.lower() or "attention" in n.lower()]
an = sum(float(kl[n]["Total Time (ns)"]) for n in att if n in kl) - \
     sum(float(ks[n]["Total Time (ns)"]) for n in att if n in ks)
ai = sum(int(kl[n]["Instances"]) for n in att if n in kl) - \
     sum(int(ks[n]["Instances"]) for n in att if n in ks)
print(f"   attention kernels per round: {an/dr/1e6:.2f} ms over {ai/dr:.0f} launches")

nl = {r["Name"]: r for r in stats_csv(a.long, "nvtx_sum")}
nsx = {r["Name"]: r for r in stats_csv(a.short, "nvtx_sum")}
print("\n| NVTX host range | ms per round | instances per round |")
print("|---|---|---|")
rows = []
for name in set(nl) | set(nsx):
    tl = float(nl[name]["Total Time (ns)"]) if name in nl else 0.0
    ts = float(nsx[name]["Total Time (ns)"]) if name in nsx else 0.0
    il = int(nl[name]["Instances"]) if name in nl else 0
    isx = int(nsx[name]["Instances"]) if name in nsx else 0
    rows.append(((tl - ts) / dr / 1e6, (il - isx) / dr, name))
for ms, n, name in sorted(rows, reverse=True)[:8]:
    print(f"| `{name}` | {ms:.2f} | {n:.1f} |")
