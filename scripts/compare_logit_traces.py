#!/usr/bin/env python3
# Copyright 2025-2026 Lablup Inc. and Jeongkyu Shin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compare two teacher-forced logit traces from `examples/logit_trace`.

The question this answers is the one byte-identity cannot: a change moved the
numbers, so what did it move and does it matter.

Byte-identity is the limit case of the headline metric here. A bit-equal arm
disagrees at zero positions; everything else is a rate, and a rate can be
compared, tracked and thresholded.

The metric that should gate is **disagreement on decided positions**. A
position where the reference itself was near-indifferent between its top two
has no right answer for a candidate to get wrong, and pooling those with
decided positions hides the only distinction that matters. Rank matters too:
picking the reference's runner-up and picking its four-thousandth choice are
different failures and only one of them is rounding.

Usage:
    compare_logit_traces.py REFERENCE.tsv CANDIDATE.tsv [--decided GAP]
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter


def load(path):
    meta, rows = {}, []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                parts = line[1:].strip().split("\t")
                if len(parts) >= 2:
                    meta[parts[0]] = parts[1:]
                continue
            c, pos, target, nll, ids, logits = line.rstrip("\n").split("\t")
            rows.append(
                (
                    int(c),
                    int(pos),
                    int(target),
                    float(nll),
                    [int(x) for x in ids.split(",")],
                    [float(x) for x in logits.split(",")],
                )
            )
    return meta, rows


def perplexity(rows):
    # Column 4 of the trace is the true NLL, positive: `examples/logit_trace.rs`
    # writes `-log_softmax(target)`. Perplexity is therefore exp(+mean(nll)).
    # This used to negate, which reported exp(mean(log p)), the geometric mean
    # probability and the reciprocal of perplexity, so a worse candidate showed
    # a NEGATIVE percentage under a header that read like an improvement.
    return math.exp(sum(r[3] for r in rows) / len(rows))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument(
        "--decided",
        type=float,
        default=2.0,
        help="reference top-two gap above which a position counts as decided "
        "(default 2.0)",
    )
    args = ap.parse_args()

    ref_meta, ref = load(args.reference)
    cand_meta, cand = load(args.candidate)

    if len(ref) != len(cand):
        print(
            f"traces cover different position counts ({len(ref)} vs {len(cand)}); "
            "they must be produced from the same corpus and chunking",
            file=sys.stderr,
        )
        return 2
    mismatched = [i for i, (a, b) in enumerate(zip(ref, cand)) if a[2] != b[2]]
    if mismatched:
        print(
            f"traces disagree about the corpus token at {len(mismatched)} positions "
            f"(first at index {mismatched[0]}); they are not comparable",
            file=sys.stderr,
        )
        return 2

    print(f"reference  {ref_meta.get('model', ['?'])[0]}")
    print(f"candidate  {cand_meta.get('model', ['?'])[0]}")
    print(f"positions  {len(ref)}\n")

    decided = 0
    dis_all = 0
    dis_decided = 0
    ranks = Counter()
    worst = (0, None)
    deltas = []
    buckets = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 1e9)]
    bstat = {b: [0, 0] for b in buckets}

    for (_, _, _, _, rids, rlg), (_, _, _, _, cids, clg) in zip(ref, cand):
        gap = rlg[0] - rlg[1]
        # The same token's logit in both arms, when the candidate still ranks
        # it: the cleanest read on how far the change moved the numbers.
        if rids[0] in cids:
            deltas.append(abs(rlg[0] - clg[cids.index(rids[0])]))
        agree = rids[0] == cids[0]
        if not agree:
            dis_all += 1
            rank = cids.index(rids[0]) + 1 if rids[0] in cids else None
            ranks[rank if rank else f">{len(cids)}"] += 1
            if rank is None or rank > worst[0]:
                if rank is None:
                    worst = (10**9, gap)
                elif rank > worst[0]:
                    worst = (rank, gap)
        for b in buckets:
            if b[0] <= gap < b[1]:
                bstat[b][0] += 1
                if not agree:
                    bstat[b][1] += 1
                break
        if gap >= args.decided:
            decided += 1
            if not agree:
                dis_decided += 1

    print(f"{'top-1 disagreement':<34}{dis_all:>7} / {len(ref):<7} {100*dis_all/len(ref):>7.3f}%")
    if decided:
        print(
            f"{'  on decided positions (gap>=%.1f)' % args.decided:<34}"
            f"{dis_decided:>7} / {decided:<7} {100*dis_decided/decided:>7.3f}%   <- gate on this"
        )
    else:
        print("  no decided positions at that threshold")

    print("\ndisagreement by how decided the reference was")
    print(f"{'reference top-two gap':<24}{'positions':>10}{'disagreed':>11}{'rate':>9}")
    for b in buckets:
        tot, dis = bstat[b]
        if not tot:
            continue
        hi = "inf" if b[1] > 1e8 else f"{b[1]:g}"
        print(f"{f'  {b[0]:g} to {hi}':<24}{tot:>10}{dis:>11}{100*dis/tot:>8.2f}%")

    if dis_all:
        print("\nwhere the reference's choice landed in the candidate")
        for k in sorted(ranks, key=lambda k: (isinstance(k, str), k)):
            print(f"  rank {k:<8}{ranks[k]:>6} ({100*ranks[k]/dis_all:5.1f}% of disagreements)")
        w = "beyond the trace's top-k" if worst[0] >= 10**9 else f"rank {worst[0]}"
        print(f"  worst: {w}")

    if deltas:
        deltas.sort()
        q = lambda p: deltas[min(int(p * len(deltas)), len(deltas) - 1)]
        print(
            f"\nlogit delta on the reference's own choice: "
            f"p50={q(.5):.4f}  p90={q(.9):.4f}  p99={q(.99):.4f}  max={deltas[-1]:.4f}"
        )

    pr, pc = perplexity(ref), perplexity(cand)
    print(
        f"\nperplexity  reference {pr:.4f}   candidate {pc:.4f}   "
        f"{100*(pc/pr-1):+.3f}%  (higher is worse)"
    )

    print()
    if dis_all == 0:
        print("verdict: byte-identical in effect, every position agrees")
    elif decided and dis_decided == 0:
        print(
            "verdict: the arms differ only where the reference was undecided, "
            "which is the rounding class rather than a behaviour change"
        )
    else:
        print(
            "verdict: the arms differ on positions the reference had decided; "
            "read the rank table before calling it noise"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
