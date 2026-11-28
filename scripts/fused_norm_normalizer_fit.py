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

"""Explain the rows where the CUDA graph and the fused add+RMSNorm disagree (issue #3501).

Usage: python3 scripts/fused_norm_normalizer_fit.py MODEL_DIR GRAPH_DUMP FUSED_DUMP LAYER

GRAPH_DUMP and FUSED_DUMP are the graph/ and fused/ directories that
`examples/fused_norm_divergence_probe.rs --op-diff DUMP --out DIR` writes, so
both hold the same new_residual. For every row whose normed output differs:

1. Which fp32 normalizers reproduce each side's row exactly by the graph
   formula bf16(w * bf16(x * inv)), in fp32 ulps from the rounded float64 one.
2. How far apart the rounded intermediates bf16(x * inv) are under the two
   fitted normalizers, and how many outputs are one and two bf16 ulps apart.
3. The fp32 sum of squares in each kernel's summation order: MLX's
   rms_norm_small<bf16, 128, 2, _, 8> and the fused kernel's 512 lanes of 4.
   Squares of bf16 values are exact in fp32, so FMA contraction cannot change
   these sums.
"""

import json
import os
import struct
import sys

import numpy as np

EPS = 1e-5
SEARCH = 24
# MLX rms_norm_small<bf16, 128, 2, _, 8>: block * chunks * reads.
MODELED_DIM = 128 * 2 * 8


def read_bf16(path):
    bits = np.fromfile(path, dtype=np.uint16)
    return (bits.astype(np.uint32) << 16).view(np.float32)


def round_bf16(a):
    b = np.asarray(a, dtype=np.float32).view(np.uint32)
    r = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return r.astype(np.uint32).view(np.float32)


def bf16_key(a):
    bits = (np.asarray(a, dtype=np.float32).view(np.uint32) >> 16).astype(np.int64)
    mag = bits & 0x7FFF
    return np.where(bits & 0x8000, -mag, mag)


def read_weight(model_dir, name):
    """One BF16 tensor of a single-file checkpoint; exits on a malformed header."""
    path = f"{model_dir}/model.safetensors"
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > size - 8:
            sys.exit(f"{path}: header length {n} exceeds the file size {size}")
        header = json.loads(f.read(n))
        if name not in header:
            sys.exit(f"{path}: no tensor {name} (is this a sharded checkpoint?)")
        entry = header[name]
        if entry.get("dtype") != "BF16":
            sys.exit(f"{path}: {name} is {entry.get('dtype')}, this script reads BF16")
        start, end = entry["data_offsets"]
        if not 0 <= start <= end <= size - 8 - n or (end - start) % 2:
            sys.exit(f"{path}: {name} data offsets {start}..{end} are outside the file")
        f.seek(8 + n + start)
        bits = np.frombuffer(f.read(end - start), dtype=np.uint16)
    return (bits.astype(np.uint32) << 16).view(np.float32)


def butterfly(values):
    """In-place xor all-reduce over 32 lanes, mask 16 down to 1."""
    v = np.array(values, dtype=np.float32)
    mask = 16
    while mask:
        v = (v + v[np.arange(32) ^ mask]).astype(np.float32)
        mask >>= 1
    return v[0]


def block_sum(lane_acc):
    """Warp butterflies, then one butterfly over the warp partials (zero-padded)."""
    warps = [butterfly(lane_acc[w * 32:(w + 1) * 32]) for w in range(len(lane_acc) // 32)]
    return butterfly(warps + [np.float32(0)] * (32 - len(warps)))


def mlx_small_sum(x, block=128, chunks=2, reads=8):
    acc = []
    for t in range(block):
        s = np.float32(0)
        for i in range(chunks):
            base = (block * i + t) * reads
            for j in range(reads):
                s = np.float32(s + x[base + j] * x[base + j])
        acc.append(s)
    return block_sum(acc)


def fused_sum(x, reads=4):
    lanes = x.size // reads
    acc = []
    for t in range(lanes):
        s = np.float32(0)
        for j in range(reads):
            s = np.float32(s + x[t * reads + j] * x[t * reads + j])
        acc.append(s)
    return block_sum(acc)


def fits(x, w, row, exact):
    """Offsets k (fp32 ulps from `exact`) whose normalizer reproduces `row`."""
    out = {}
    for direction, step in ((1, np.inf), (-1, 0)):
        v = exact
        for k in range(SEARCH + 1):
            out[direction * k] = v
            v = np.nextafter(v, np.float32(step))
    return {k: v for k, v in out.items() if (round_bf16(w * round_bf16(x * v)) == row).all()}


def main():
    if len(sys.argv) != 5 or not sys.argv[4].isdigit():
        sys.exit("usage: fused_norm_normalizer_fit.py MODEL_DIR GRAPH_DUMP FUSED_DUMP LAYER")
    model_dir, graph_dir, fused_dir, layer = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    w = read_weight(model_dir, f"model.layers.{layer}.post_attention_layernorm.weight")
    dim = w.size
    if dim != MODELED_DIM:
        sys.exit(
            f"the summation orders modelled here are those of bf16 dim {MODELED_DIM} "
            f"(MLX 128 threads x 2 chunks x 8 reads, fused 512 lanes x 4); this layer is {dim} wide"
        )
    h = read_bf16(f"{graph_dir}/L{layer:02d}_h.bin").reshape(-1, dim)
    if not (read_bf16(f"{fused_dir}/L{layer:02d}_h.bin").reshape(-1, dim) == h).all():
        sys.exit("the two dumps do not hold the same new_residual")
    graph = read_bf16(f"{graph_dir}/L{layer:02d}_normed.bin").reshape(-1, dim)
    fused = read_bf16(f"{fused_dir}/L{layer:02d}_normed.bin").reshape(-1, dim)
    for r in np.nonzero((graph != fused).any(axis=1))[0]:
        x = h[r]
        exact = np.float32(1.0 / np.sqrt(np.mean(x.astype(np.float64) ** 2) + EPS))
        fit_g, fit_f = fits(x, w, graph[r], exact), fits(x, w, fused[r], exact)
        if not fit_g or not fit_f:
            print(f"layer {layer} row {r}: NOT explained by one normalizer per side")
            continue
        inv_g, inv_f = fit_g[min(fit_g, key=abs)], fit_f[min(fit_f, key=abs)]
        mid_ulp = np.abs(bf16_key(round_bf16(x * inv_g)) - bf16_key(round_bf16(x * inv_f)))
        out_ulp = np.abs(bf16_key(graph[r]) - bf16_key(fused[r]))
        s_mlx, s_fused = mlx_small_sum(x), fused_sum(x)
        sum_ulps = int(abs(int(s_mlx.view(np.int32)) - int(s_fused.view(np.int32))))
        print(
            f"layer {layer} row {r}: graph fits offsets {min(fit_g)}..{max(fit_g)}, "
            f"fused {min(fit_f)}..{max(fit_f)}; intermediate bf16(x*inv) max ulp {mid_ulp.max()} "
            f"on {(mid_ulp > 0).sum()} elements; outputs 1 ulp apart {(out_ulp == 1).sum()}, "
            f"2 ulps apart {(out_ulp == 2).sum()}, more {(out_ulp > 2).sum()}; "
            f"sum of squares MLX order {s_mlx!r} fused order {s_fused!r} ({sum_ulps} fp32 ulps apart)"
        )


if __name__ == "__main__":
    main()
