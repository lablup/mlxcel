#!/usr/bin/env python3
"""Attribute server decode windows of a rocprofv3 kernel trace (issue #2156).

Usage: analyze_trace.py KERNEL_TRACE.csv PROFILE.json [OUT.json]

For each level in PROFILE.json (written by profile_driver.py) the window where
every request of the level is past its first token and none has finished is
cut out of the trace: [max(first token), min(end)]. Inside it: wall time, GPU
busy time (union of dispatch intervals), host gap (wall - busy), kernel time
per class, dispatch count, and the idle gaps between dispatches by size.
"""
import collections
import json
import pathlib
import sys

sys.path.insert(0, "scripts")
import rocm_decode_profile as rdp  # noqa: E402


def gaps(ds, lo, hi):
    """Idle intervals between dispatches inside [lo, hi] (union-based)."""
    out, cur_e = [], lo
    for d in ds:
        s, e = max(d.start, lo), min(d.end, hi)
        if e <= s:
            continue
        if s > cur_e:
            out.append((cur_e, s - cur_e))
        cur_e = max(cur_e, e)
    if hi > cur_e:
        out.append((cur_e, hi - cur_e))
    return out


def main():
    trace, prof = pathlib.Path(sys.argv[1]), json.load(open(sys.argv[2]))
    ds = rdp.read_trace(trace)
    print(f"dispatches: {len(ds)}  trace span {(ds[-1].end - ds[0].start) / 1e9:.1f} s")
    results = []
    for lvl in prof:
        reqs = lvl["requests"]
        lo = max(r["first_ns"] for r in reqs)
        hi = min(r["end_ns"] for r in reqs)
        overlap = hi > lo
        if not overlap:
            # No window where every request decodes: take the whole level from
            # the first token to the last end and split it by kernel class.
            lo = min(r["first_ns"] for r in reqs)
            hi = max(r["end_ns"] for r in reqs)
        win = [d for d in ds if lo <= d.start < hi]
        wall = hi - lo
        busy = rdp.busy_ns(win, lo, hi)
        cls = collections.Counter()
        cls_n = collections.Counter()
        names = collections.Counter()
        for d in win:
            c = rdp.classify(d.name)
            cls[c] += d.end - d.start
            cls_n[c] += 1
            names[rdp.short_kernel(d.name)] += 1
        g = gaps(win, lo, hi)
        buckets = collections.Counter()
        for _, n in g:
            k = ("<10us" if n < 10e3 else "10-50us" if n < 50e3 else "50-200us" if n < 200e3
                 else "0.2-1ms" if n < 1e6 else "1-5ms" if n < 5e6 else ">=5ms")
            buckets[k] += n
        big = sorted(g, key=lambda x: -x[1])[:15]
        tokens = min(r["tokens"] for r in reqs)
        res = {
            "conc": lvl["conc"], "prompt_tokens": lvl["prompt_tokens"],
            "window_ms": wall / 1e6, "busy_ms": busy / 1e6, "gap_ms": (wall - busy) / 1e6,
            "busy_share": busy / wall if wall else 0, "dispatches": len(win),
            "class_ms": {k: v / 1e6 for k, v in cls.most_common()},
            "class_n": dict(cls_n.most_common()),
            "gap_ms_by_size": {k: v / 1e6 for k, v in buckets.items()},
            "largest_gaps_ms": [(round((s - lo) / 1e6, 2), round(n / 1e6, 3)) for s, n in big],
            "top_kernels": names.most_common(25),
            "delta": lvl["delta"], "decode_tok_s": [r["decode_tok_s"] for r in reqs],
            "min_tokens": tokens, "all_decoding_window": overlap,
        }
        results.append(res)
        print(f"\n== conc {res['conc']} prompt {res['prompt_tokens']} (all decoding: {overlap}): window {res['window_ms']:.1f} ms, "
              f"busy {res['busy_ms']:.1f} ms ({100 * res['busy_share']:.1f}%), gap {res['gap_ms']:.1f} ms, "
              f"dispatches {len(win)}")
        print("   class ms:", {k: round(v, 1) for k, v in list(res["class_ms"].items())[:12]})
        print("   class n:", dict(list(res["class_n"].items())[:12]))
        print("   gaps by size ms:", {k: round(v, 1) for k, v in sorted(res["gap_ms_by_size"].items())})
        print("   largest gaps (offset ms, len ms):", res["largest_gaps_ms"][:10])
        print("   top kernels:", res["top_kernels"][:12])
        print("   lines:", *lvl["lines"], sep="\n")
    if len(sys.argv) > 3:
        json.dump(results, open(sys.argv[3], "w"), indent=1)


if __name__ == "__main__":
    main()
