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
"""Regenerate the LFM2-VL position-grid resample reference (`reference.json`).

Out-of-band tooling for issue #2388: nothing in mlxcel's request path runs
Python. Two independent references are written.

- `matrices`: the `[out, in]` interpolation matrix of PyTorch's
  `F.interpolate(mode="bilinear", antialias=True, align_corners=False)` for a
  16-wide axis, read off by resampling the identity. This is the kernel
  SigLIP2 resizes its position grid with.
- `grids`: mlx-vlm's own `VisionEmbeddings.resize_positional_embeddings` (the
  LFM2-VL tower code) applied to a deterministic `[16, 16, 4]` table, for
  target grids that downsample one axis and upsample the other.

The script also checks that mlx-vlm's `_bilinear_weights_1d` reproduces the
PyTorch matrices and records the largest difference, so the fixture states
that the two references agree.

Usage (from the repository root, in a venv with torch, mlx and mlx-vlm):

    python -I tests/fixtures/lfm2_vl/generate_reference.py > tests/fixtures/lfm2_vl/reference.json
"""

from __future__ import annotations

import json
import math
import sys
from importlib import metadata

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F
from mlx_vlm.models.interpolate import _bilinear_weights_1d
from mlx_vlm.models.lfm2_vl.vision import VisionEmbeddings

SIDE = 16
DIM = 4
MATRIX_OUTS = [5, 8, 12, 24, 28]
GRIDS = [(12, 20), (24, 8)]


def table_value(i: int, j: int, d: int) -> float:
    """Deterministic table entry; the Rust test rebuilds the same table."""
    return math.sin(0.37 * i + 0.11 * j * (d + 1)) + 0.05 * d


def torch_matrix(in_size: int, out_size: int) -> np.ndarray:
    # The identity rides on a 3-wide second axis kept at its size: torch's
    # antialiased kernel returns zeros for a width-1 tensor, and a same-size
    # axis resamples as the identity, so column 0 carries the matrix.
    eye = torch.eye(in_size, dtype=torch.float32).reshape(in_size, 1, in_size, 1)
    eye = eye.repeat(1, 1, 1, 3)
    out = F.interpolate(
        eye, size=(out_size, 3), mode="bilinear", antialias=True, align_corners=False
    )
    # out[n, 0, o, 0] is the weight of input n in output o.
    return out[:, 0, :, 0].T.contiguous().numpy()


def mlx_vlm_matrix(in_size: int, out_size: int) -> np.ndarray:
    idx, w = _bilinear_weights_1d(in_size, out_size, False, True)
    idx = np.array(idx)
    w = np.array(w)
    m = np.zeros((out_size, in_size), dtype=np.float64)
    for o in range(out_size):
        for t in range(idx.shape[1]):
            m[o, idx[o, t]] += w[o, t]
    return m


def mlx_vlm_commit() -> str:
    try:
        url = json.loads(metadata.distribution("mlx-vlm").read_text("direct_url.json"))
        return url.get("vcs_info", {}).get("commit_id", "unknown")
    except (FileNotFoundError, TypeError, ValueError):
        return "unknown"


def main() -> None:
    matrices = []
    worst = 0.0
    for out in MATRIX_OUTS:
        m = torch_matrix(SIDE, out)
        worst = max(worst, float(np.abs(m - mlx_vlm_matrix(SIDE, out)).max()))
        matrices.append({"in": SIDE, "out": out, "values": m.reshape(-1).tolist()})

    table = np.array(
        [[[table_value(i, j, d) for d in range(DIM)] for j in range(SIDE)] for i in range(SIDE)],
        dtype=np.float32,
    )
    grids = []
    for h, w in GRIDS:
        resized = VisionEmbeddings.resize_positional_embeddings(
            mx.array(table), mx.array([[h, w]]), max_length=h * w
        )
        values = np.array(resized[0, : h * w].astype(mx.float32)).reshape(-1)
        grids.append({"h": h, "w": w, "values": values.tolist()})

    json.dump(
        {
            "issue": 2388,
            "torch": torch.__version__,
            "mlx": metadata.version("mlx"),
            "mlx_vlm": metadata.version("mlx-vlm"),
            "mlx_vlm_commit": mlx_vlm_commit(),
            "mlx_vlm_vs_torch_max_abs_diff": worst,
            "side": SIDE,
            "dim": DIM,
            "table": "sin(0.37*i + 0.11*j*(d+1)) + 0.05*d",
            "matrices": matrices,
            "grids": grids,
        },
        sys.stdout,
        indent=1,
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
