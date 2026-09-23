#!/usr/bin/env python3
"""Issue #1935: check a served DFlash run's round algebra offline, with no GPU.

Reads the `DFlash round transcript` debug lines from a server log and verifies
the four properties that make the round loop's emitted stream equal to what the
target's caches say it should be:

  1. every round's bonus is the previous round's last emitted token;
  2. `accepted` is the longest common prefix of the draft and the target argmax;
  3. `new_tokens == draft[:accepted] + [target[accepted]]`, up to the final
     round's budget truncation;
  4. the emitted stream is therefore exactly the sequence the caches hold.

A failure localizes the defect in the round loop or its wrapper. All four
holding puts the defect in the target forward or in the process around it, and
prints the round and row that produced a given emitted index so the next arm has
an anchor.
"""
import argparse
import ast
import json
import re
import sys

LINE = re.compile(
    r"DFlash round transcript\s+round=(\d+)\s+bs=(\d+)\s+bonus=(-?\d+)\s+"
    r"draft_tokens=(\[[^\]]*\])\s+target_tokens=(\[[^\]]*\])\s+"
    r"accepted=(\d+)\s+new_tokens=(\[[^\]]*\])"
)


def parse_rounds(path):
    """Every burst in the log, in order. A new burst starts when `round`
    restarts at 1, which is how a warm-up request is told from the measured
    one without the server having to label them."""
    bursts = []
    cur = []
    for raw in open(path, errors="replace"):
        m = LINE.search(raw)
        if not m:
            continue
        r = {
            "round": int(m.group(1)),
            "bs": int(m.group(2)),
            "bonus": int(m.group(3)),
            "draft": ast.literal_eval(m.group(4)),
            "target": ast.literal_eval(m.group(5)),
            "accepted": int(m.group(6)),
            "new": ast.literal_eval(m.group(7)),
        }
        if r["round"] == 0 and cur:
            bursts.append(cur)
            cur = []
        cur.append(r)
    if cur:
        bursts.append(cur)
    return bursts


def lcp(a, b):
    n = 0
    while n < len(a) and n < len(b) and a[n] == b[n]:
        n += 1
    return n


def check(burst, label):
    problems = []
    emitted = [burst[0]["bonus"]]  # the first bonus the caller already delivered
    origin = [("prefill", -1)]
    for i, r in enumerate(burst):
        if r["bonus"] != emitted[-1]:
            problems.append(
                f"round {r['round']}: bonus {r['bonus']} is not the previous "
                f"round's last emitted token {emitted[-1]}"
            )
        if len(r["draft"]) != r["bs"] - 1:
            problems.append(
                f"round {r['round']}: {len(r['draft'])} proposals for bs={r['bs']}"
            )
        if len(r["target"]) != r["bs"]:
            problems.append(
                f"round {r['round']}: {len(r['target'])} target argmaxes for bs={r['bs']}"
            )
        want_acc = lcp(r["draft"], r["target"][: len(r["draft"])])
        if want_acc != r["accepted"]:
            problems.append(
                f"round {r['round']}: accepted={r['accepted']} but the longest "
                f"common prefix of draft and target is {want_acc}"
            )
        want_new = r["draft"][: r["accepted"]] + [r["target"][r["accepted"]]]
        last = i == len(burst) - 1
        if r["new"] != want_new and not (last and r["new"] == want_new[: len(r["new"])]):
            problems.append(
                f"round {r['round']}: new_tokens {r['new']} is not "
                f"draft[:{r['accepted']}] + [target[{r['accepted']}]] = {want_new}"
            )
        for k, tok in enumerate(r["new"]):
            emitted.append(tok)
            origin.append((f"round {r['round']}", k))
    return emitted, origin, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--burst", type=int, default=-1, help="which burst (default: last)")
    ap.add_argument("--reference", help="json file with a classic token id list")
    ap.add_argument("--dump", help="write the emitted id list here")
    a = ap.parse_args()

    bursts = parse_rounds(a.log)
    print(f"[1935] {len(bursts)} burst(s) in {a.log}: "
          f"rounds {[len(b) for b in bursts]}")
    if not bursts:
        return 1
    burst = bursts[a.burst]
    emitted, origin, problems = check(burst, a.log)
    print(f"[1935] burst {a.burst}: {len(burst)} rounds, {len(emitted)} emitted tokens, "
          f"{sum(1 for r in burst if r['accepted'] < r['bs'] - 1)} rewinds")
    if problems:
        print(f"[1935] ROUND ALGEBRA FAILS, {len(problems)} problem(s):")
        for p in problems[:20]:
            print(f"  - {p}")
    else:
        print("[1935] round algebra holds: every emitted token is the target's "
              "own argmax under a self-consistent cache history")

    if a.dump:
        json.dump(emitted, open(a.dump, "w"))
        print(f"[1935] wrote {a.dump}")

    if a.reference:
        ref = json.load(open(a.reference))
        n = min(len(ref), len(emitted))
        first = next((i for i in range(n) if ref[i] != emitted[i]), -1)
        print(f"[1935] reference {len(ref)} ids against emitted {len(emitted)}: "
              f"first difference at {first} (-1 means none over {n} compared)")
        if first >= 0:
            src, row = origin[first]
            print(f"[1935] emitted[{first}]={emitted[first]} came from {src} row {row}; "
                  f"reference has {ref[first]}")
            for r in burst:
                if f"round {r['round']}" == src:
                    print(f"[1935] that round: bs={r['bs']} bonus={r['bonus']} "
                          f"draft={r['draft']} target={r['target']} "
                          f"accepted={r['accepted']} new={r['new']}")
                    break
    return 0


if __name__ == "__main__":
    sys.exit(main())
