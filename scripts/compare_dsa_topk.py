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
"""Compare two DSA top-k selection traces written under `MLXCEL_TRACE_DSA_TOPK`.

The DSA indexer (DeepSeek V3.2, GLM-5) picks the `index_topk` cached positions
each decode step attends to. A change to how the indexer keys are stored (the
affine int8 latent slot of issue #3163) can move that selection, and a moved
selection changes what the attention reads, which no logit-level bound on the
codec covers. This reports how much of the selection two arms share.

Both arms must be teacher-forced over the same tokens (`examples/logit_trace`)
so that step `n` of layer `l` scores the same context in each. A trace line is
`layer<TAB>kv_len<TAB>p0,p1,...` with the positions ascending; lines are paired
per layer in the order they were written.

The overlap of one step is `|A & B| / |A|`. The gate is the mean per layer.

Usage:
    compare_dsa_topk.py REFERENCE.tsv CANDIDATE.tsv [--min-mean 0.99]
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict


class TraceError(Exception):
    """A trace that cannot be compared."""


def load(path: str) -> dict[int, list[tuple[int, frozenset[int]]]]:
    """Read a trace into `{layer: [(kv_len, positions), ...]}` in file order."""
    layers: dict[int, list[tuple[int, frozenset[int]]]] = defaultdict(list)
    with open(path) as fh:
        for number, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                layer, kv_len, positions = line.split("\t")
                selected = frozenset(int(p) for p in positions.split(","))
                layers[int(layer)].append((int(kv_len), selected))
            except ValueError as err:
                raise TraceError(f"{path}:{number}: malformed line") from err
    if not layers:
        raise TraceError(f"{path}: no selections (was the context above index_topk?)")
    return layers


def compare(
    reference: dict[int, list[tuple[int, frozenset[int]]]],
    candidate: dict[int, list[tuple[int, frozenset[int]]]],
) -> list[tuple[int, int, float, float, int, int]]:
    """Per layer: `(layer, steps, mean, min, kv_len_min, kv_len_max)`."""
    if reference.keys() != candidate.keys():
        raise TraceError("the two traces cover different layers")
    rows = []
    for layer in sorted(reference):
        ref, cand = reference[layer], candidate[layer]
        if len(ref) != len(cand):
            raise TraceError(
                f"layer {layer}: {len(ref)} reference steps against {len(cand)} candidate steps"
            )
        overlaps = []
        for (ref_len, ref_sel), (cand_len, cand_sel) in zip(ref, cand):
            if ref_len != cand_len or len(ref_sel) != len(cand_sel):
                raise TraceError(f"layer {layer}: the arms are not step-aligned")
            overlaps.append(len(ref_sel & cand_sel) / len(ref_sel))
        lengths = [kv_len for kv_len, _ in ref]
        rows.append(
            (
                layer,
                len(overlaps),
                sum(overlaps) / len(overlaps),
                min(overlaps),
                min(lengths),
                max(lengths),
            )
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument(
        "--min-mean",
        type=float,
        default=0.99,
        help="fail when any layer's mean overlap is below this (default 0.99)",
    )
    args = parser.parse_args()
    try:
        rows = compare(load(args.reference), load(args.candidate))
    except (OSError, TraceError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2

    print("layer\tsteps\tkv_len\tmean_overlap\tmin_overlap")
    for layer, steps, mean, low, len_min, len_max in rows:
        print(f"{layer}\t{steps}\t{len_min}-{len_max}\t{mean:.5f}\t{low:.5f}")
    means = [row[2] for row in rows]
    worst = min(rows, key=lambda row: row[2])
    print(
        f"# layers={len(rows)} mean_of_layer_means={sum(means) / len(means):.5f} "
        f"worst_layer={worst[0]} worst_layer_mean={worst[2]:.5f} "
        f"min_step_overlap={min(row[3] for row in rows):.5f}"
    )
    failed = [row[0] for row in rows if row[2] < args.min_mean]
    if failed:
        print(f"# FAIL: layers below {args.min_mean}: {failed}")
        return 1
    print(f"# PASS: every layer's mean overlap is at least {args.min_mean}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
