#!/usr/bin/env python3
# Copyright 2025-2026 Lablup Inc.
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
"""Compare a verify block against the decode step at equal context, row by row.

Reads the output of `scripts/matched_width_trace.sh`: `w<W>.tsv` (the block run)
and `w1_p<PREFILL + j>.tsv` (the width-1 reference for row j of every block).
Row j of block c and row 0 of the width-1 run for PREFILL + j see the same
token history and the same input token, so any difference is the arithmetic
path (block width and its kernels), not context length.

Row 0 of a block sees only the sequentially built history, so a difference
there is the kernel for M = W against M = 1. Rows 1..W-1 see K/V that the block
itself computed, so a difference there also includes block-built history.

Usage:
    compare_matched_width.py OUT_DIR [WIDTH] [PREFILL]
"""
import math
import sys

D = sys.argv[1]
W = int(sys.argv[2]) if len(sys.argv) > 2 else 8
P = int(sys.argv[3]) if len(sys.argv) > 3 else 128


def load(path):
    for line in open(path):
        if line.startswith("#"):
            continue
        c, pos, tok, nll, ids, lg = line.rstrip("\n").split("\t")
        yield int(c), int(pos), int(tok), float(nll), [int(x) for x in ids.split(",")], [float(x) for x in lg.split(",")]


block = {(c, j): r for c, j, *r in load(f"{D}/w{W}.tsv")}  # r = [token, nll, top_ids, top_logits]
blocks = max(c for c, _ in block) + 1
print("row  positions  top1_disagree  decided(gap>=2)  disagree_on_decided  max_gap_at_disagreement  max_logit_delta")
tot = dis = dec = disdec = 0
deltas, exact_ties, nll_ref, nll_blk, gaps = [], 0, 0.0, 0.0, []
row0_nonzero = 0
for j in range(W):
    ref = {c: r for c, _, *r in load(f"{D}/w1_p{P + j}.tsv")}
    n = d = de = dd = 0
    gmax = dmax = 0.0
    for c in range(blocks):
        tokb, nllb, idsb, lgb = block[(c, j)]
        tok1, nll1, ids1, lg1 = ref[W * c]
        assert tok1 == tokb, (c, j)
        gap = lg1[0] - lg1[1]
        n += 1
        de += gap >= 2.0
        if j == 0:
            exact_ties += gap == 0.0
        if ids1[0] in idsb:
            delta = abs(lg1[0] - lgb[idsb.index(ids1[0])])
            dmax = max(dmax, delta)
            if j > 0:
                deltas.append(delta)
            elif delta > 0:
                row0_nonzero += 1
        if ids1[0] != idsb[0]:
            d += 1
            gmax = max(gmax, gap)
            dd += gap >= 2.0
            gaps.append((P + W * c + j, j, round(gap, 4)))
        nll_ref += nll1
        nll_blk += nllb
    print(f"{j:>3} {n:>10} {d:>14} {de:>16} {dd:>20} {gmax:>24.3f} {dmax:>16.3f}")
    tot += n
    dis += d
    dec += de
    disdec += dd
print(f"all {tot:>10} {dis:>14} {dec:>16} {disdec:>20}")
print(f"row 0: {blocks} positions, {row0_nonzero} with a nonzero delta on the reference top-1 logit, {exact_ties} exact reference ties (gap 0)")
print("disagreements (token, row, reference top-two gap):", gaps)
deltas.sort()
if deltas:
    q = lambda f: deltas[int(f * (len(deltas) - 1))]
    print(
        f"rows 1-{W - 1}: {len(deltas)} compared, {sum(x > 0 for x in deltas)} with a nonzero logit delta; "
        f"p50 {q(.5)} p90 {q(.9)} p99 {q(.99)} max {deltas[-1]}"
    )
ppl = lambda s: math.exp(s / tot)
print(f"perplexity width-1 {ppl(nll_ref):.4f} width-{W} {ppl(nll_blk):.4f} ({100 * (ppl(nll_blk) / ppl(nll_ref) - 1):+.3f}%)")
