#!/usr/bin/env python3
"""Issue #1935: read the served arms against the classic null arm.

The first question is not where the divergence is but what kind it is, and the
logprobs answer that in one read:

  identical logprobs through the divergence, and the burst's token sitting in
  classic's top-k at the SAME logprob -> an exact tie on identical logits,
  broken differently by two selection paths;

  identical through, then a fraction apart -> same logits, last-ulp flip,
  genuine numerics;

  already drifting before the divergence -> the states or the computation
  differ, and the flip is where drift first crossed an argmax;

  a large gap -> state divergence.
"""
import hashlib
import json
import os
import sys

SP = os.path.dirname(os.path.abspath(__file__))
ARMS = os.path.join(SP, "arms")


def load(tag):
    p = os.path.join(ARMS, f"arm.{tag}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()[:10]


def summarize(tag, rec):
    if rec is None:
        print(f"[1935] {tag}: missing")
        return
    if rec.get("error"):
        print(f"[1935] {tag}: ERROR {rec['error']}")
    for r in rec.get("responses", []):
        toks = r.get("tokens") or []
        print(f"[1935] {tag} request {r['index']}: sha {sha(r['text'])}, "
              f"{len(r['text'])} chars, {len(toks)} logprob tokens, "
              f"usage {r.get('usage')}")


def compare(label, ref, spec):
    rt = ref["responses"][-1]
    st = spec["responses"][-1]
    rtok, stok = rt.get("tokens") or [], st.get("tokens") or []
    rlp, slp = rt.get("token_logprobs") or [], st.get("token_logprobs") or []
    n = min(len(rtok), len(stok))
    first = next((i for i in range(n) if rtok[i] != stok[i]), -1)
    print(f"\n[1935] {label}: text sha {sha(st['text'])} against classic {sha(rt['text'])}; "
          f"{len(stok)} tokens against {len(rtok)}; first differing token index {first}")
    if first < 0:
        print(f"[1935] {label}: token streams agree over all {n} compared positions")
        return
    # Do the logprobs agree BEFORE the divergence? That is the question that
    # separates a selection difference from accumulated numerical drift.
    exact_before = sum(1 for i in range(first) if rlp[i] == slp[i])
    print(f"[1935] {label}: logprobs bit-identical at {exact_before} of the {first} "
          f"positions before the divergence")
    worst = max(((abs(rlp[i] - slp[i]), i) for i in range(first)), default=(0.0, -1))
    print(f"[1935] {label}: largest pre-divergence logprob gap {worst[0]:.6e} at index {worst[1]}")
    lo, hi = max(0, first - 3), min(n, first + 4)
    for i in range(lo, hi):
        mark = "  <<<" if i == first else ""
        print(f"    [{i}] classic {rtok[i]!r} lp {rlp[i]:.6f} | "
              f"burst {stok[i]!r} lp {slp[i]:.6f}{mark}")
    for name, t in (("classic", rt), ("burst", st)):
        top = (t.get("top_logprobs") or [None] * n)[first]
        print(f"    top-k at {first} ({name}): {top}")


def main():
    ref = load("wclassic")
    summarize("wclassic", ref)
    for tag in ("w4", "w2"):
        rec = load(tag)
        summarize(tag, rec)
        if ref and rec and ref.get("responses") and rec.get("responses"):
            compare(tag, ref, rec)
    # Cross-width: the record's strongest structural clue is that these two
    # agree despite entirely different round structures.
    a, b = load("w4"), load("w2")
    if a and b and a.get("responses") and b.get("responses"):
        same = a["responses"][-1]["text"] == b["responses"][-1]["text"]
        print(f"\n[1935] width 4 and width 2 texts identical: {same}")
    # Within-arm: burst 1 (warm-up) against burst 2 (measured). A difference
    # here would be cross-request state, which no arm has looked for.
    for tag in ("wclassic", "w4", "w2"):
        rec = load(tag)
        if rec and len(rec.get("responses", [])) >= 2:
            same = rec["responses"][0]["text"] == rec["responses"][1]["text"]
            print(f"[1935] {tag}: request 0 and request 1 identical: {same}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
