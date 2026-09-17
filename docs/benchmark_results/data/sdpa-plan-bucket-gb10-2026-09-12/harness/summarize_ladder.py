#!/usr/bin/env python3
"""Turn the #1820 context-ladder jsonl into its table.

Two metrics per rung, because they answer different questions and only one of
them is a kernel measurement:

`tok/s` is what a user sees, and it is NOT a clean A/B. The bucketed arm runs
cuDNN's flash kernel and the `fb-` arm runs MLX's ops fallback, and #1799
already recorded that the fallback shifts the greedy path at ties. The two arms
therefore generate different text, accept a different number of drafted tokens,
and need a different number of verify rounds to reach the same token count. Part
of any tok/s gap is that difference in luck, not in speed.

`verify ms/round` divides the round loop's own verify timer by the round count,
so it is the per-round cost of the thing this issue changes with the acceptance
difference divided out. It is the number to read for a crossover.

Both are reported. Where they disagree, the per-round number is the kernel
result and the tok/s number is the end-to-end one.
"""
import json
import statistics as st
import sys
from collections import defaultdict

RUNGS = ["prompt_code0.txt", "prompt_code_long.txt", "prompt_code_8k.txt",
         "prompt_code_16k.txt", "prompt_code_32k.txt"]
LABEL = {"prompt_code0.txt": "152", "prompt_code_long.txt": "2634",
         "prompt_code_8k.txt": "8k", "prompt_code_16k.txt": "16k",
         "prompt_code_32k.txt": "32k"}


def spread(xs):
    return f"{st.mean(xs):.2f} ({min(xs):.2f} to {max(xs):.2f})"


def main():
    rows = []
    for p in sys.argv[1:]:
        for line in open(p):
            if line.strip():
                rows.append(json.loads(line))
    rows = [r for r in rows if not r.get("warmup") and "tok_s" in r and r.get("rc") == 0]
    dirty = [r for r in rows if r.get("foreign_models_before") or r.get("foreign_models_after")]
    if dirty:
        print(f"WARNING: {len(dirty)} rows saw a foreign model process; excluded\n")
        rows = [r for r in rows if not (r.get("foreign_models_before") or r.get("foreign_models_after"))]

    g = defaultdict(list)
    for r in rows:
        g[(r.get("prompt_file", "?"), r["cfg"])].append(r)

    print("| prompt tokens | width | arm | n | tok/s (min to max) | verify ms/round | rounds | accepted | load1 (min to max) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for p in RUNGS:
        for w in (2, 4, 6, 8, 16):
            for pre, name in (("b", "bucketed"), ("fb-b", "#1799 fallback"), ("up-b", "upstream cuDNN")):
                v = g.get((p, f"{pre}{w}"), [])
                if not v:
                    continue
                d = [r["diag"] for r in v if r.get("diag", {}).get("rounds")]
                vr = f"{st.mean(x['verify_ms'] / x['rounds'] for x in d):.1f}" if d else ""
                rounds = f"{d[0]['rounds']:.0f}" if d else ""
                acc = f"{d[0]['accepted']:.0f}" if d else ""
                ld = [r["load1_before"] for r in v]
                print(f"| {LABEL.get(p, p)} | {w} | {name} | {len(v)} | {spread([r['tok_s'] for r in v])} "
                      f"| {vr} | {rounds} | {acc} | {min(ld):.2f} to {max(ld):.2f} |")

    print("\n### Bucketed against the #1799 fallback\n")
    print("| prompt tokens | width | tok/s ratio | verify ms/round ratio | verdict |")
    print("|---|---|---|---|---|")
    for p in RUNGS:
        for w in (2, 4, 6, 8, 16):
            b = g.get((p, f"b{w}"), [])
            f = g.get((p, f"fb-b{w}"), [])
            if not b or not f:
                continue
            bt, ft = st.mean(r["tok_s"] for r in b), st.mean(r["tok_s"] for r in f)
            bd = [r["diag"] for r in b if r.get("diag", {}).get("rounds")]
            fd = [r["diag"] for r in f if r.get("diag", {}).get("rounds")]
            rv = ""
            verdict = ""
            if bd and fd:
                bm = st.mean(x["verify_ms"] / x["rounds"] for x in bd)
                fm = st.mean(x["verify_ms"] / x["rounds"] for x in fd)
                rv = f"{fm / bm:.3f}x"
                # Disjoint ranges, on the per-round metric, decide the verdict.
                blo = min(x["verify_ms"] / x["rounds"] for x in bd)
                bhi = max(x["verify_ms"] / x["rounds"] for x in bd)
                flo = min(x["verify_ms"] / x["rounds"] for x in fd)
                fhi = max(x["verify_ms"] / x["rounds"] for x in fd)
                if bhi < flo:
                    verdict = "bucketed ahead"
                elif fhi < blo:
                    verdict = "fallback ahead"
                else:
                    verdict = "overlapping"
            print(f"| {LABEL.get(p, p)} | {w} | {bt / ft:.3f}x | {rv} | {verdict} |")


if __name__ == "__main__":
    main()
