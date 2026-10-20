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
"""Regenerate the DeepSeek-OCR SAM antialiased-bicubic reference (`reference.json`).

Out-of-band tooling for issue #2465: nothing in mlxcel's request path runs
Python. Two independent references are written.

- `matrices`: the `[out, in]` interpolation matrix of PyTorch's
  `F.interpolate(mode="bicubic", antialias=True, align_corners=False)` for a
  16-wide axis, read off by resampling the identity. A downsample (16 -> 5,
  16 -> 8) and an upsample (16 -> 24) are both recorded: the antialiased kernel
  uses a = -0.5 in both directions (ATen `HelperInterpCubic::aa_filter`, PIL
  compatibility), with the support stretched only when downsampling.
- `grids`: torch's direct 2-D antialiased bicubic on a deterministic
  `[16, 16, 4]` table. The script also checks mlx-vlm's
  `resize_bicubic_nhwc(antialias=True)` against torch and records the largest
  difference, and how far torch's own resize sits from the separable
  composition of the two 1-D `matrices` entries (what the Rust code computes).

Usage (from the repository root, in a venv with torch and mlx-vlm):

    python -I tests/fixtures/deepseekocr_sam/generate_reference.py > tests/fixtures/deepseekocr_sam/reference.json
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
from mlx_vlm.models.interpolate import resize_bicubic_nhwc

SIDE = 16
DIM = 4
MATRIX_OUTS = [5, 8, 24]
GRIDS = [(5, 5), (5, 7), (24, 24)]


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
        eye, size=(out_size, 3), mode="bicubic", antialias=True, align_corners=False
    )
    # out[n, 0, o, 0] is the weight of input n in output o.
    return out[:, 0, :, 0].T.contiguous().numpy()


def mlx_vlm_commit() -> str:
    try:
        url = json.loads(metadata.distribution("mlx-vlm").read_text("direct_url.json"))
        return url.get("vcs_info", {}).get("commit_id", "unknown")
    except (FileNotFoundError, TypeError, ValueError):
        return "unknown"


def main() -> None:
    matrices = []
    for out in MATRIX_OUTS:
        m = torch_matrix(SIDE, out)
        matrices.append({"in": SIDE, "out": out, "values": m.reshape(-1).tolist()})

    table = np.array(
        [[[table_value(i, j, d) for d in range(DIM)] for j in range(SIDE)] for i in range(SIDE)],
        dtype=np.float32,
    )
    grids = []
    worst_vlm = 0.0
    worst_separable = 0.0
    for h, w in GRIDS:
        x = torch.from_numpy(table.transpose(2, 0, 1))[None]
        direct = F.interpolate(
            x, size=(h, w), mode="bicubic", antialias=True, align_corners=False
        )
        direct = direct[0].numpy().transpose(1, 2, 0).reshape(-1)

        # mlx-vlm's own antialiased bicubic (channel-last), the tower-side
        # reference implementation.
        vlm = resize_bicubic_nhwc(mx.array(table)[None], (h, w), antialias=True)
        vlm = np.array(vlm[0].astype(mx.float32)).reshape(-1)
        worst_vlm = max(worst_vlm, float(np.abs(vlm - direct).max()))

        # The Rust code composes the two 1-D matrices entries; record how far
        # that composition sits from torch's direct 2-D resize.
        wh = torch_matrix(SIDE, h)
        ww = torch_matrix(SIDE, w)
        comp = wh @ table.reshape(SIDE, SIDE * DIM)
        comp = comp.reshape(h, SIDE, DIM).transpose(1, 0, 2).reshape(SIDE, h * DIM)
        comp = ww @ comp
        comp = comp.reshape(w, h, DIM).transpose(1, 0, 2).reshape(-1)
        worst_separable = max(worst_separable, float(np.abs(comp - direct).max()))

        grids.append({"h": h, "w": w, "values": direct.tolist()})

    json.dump(
        {
            "issue": 2465,
            "torch": torch.__version__,
            "mlx": metadata.version("mlx"),
            "mlx_vlm": metadata.version("mlx-vlm"),
            "mlx_vlm_commit": mlx_vlm_commit(),
            "mlx_vlm_vs_torch_max_abs_diff": worst_vlm,
            "separable_vs_direct_max_abs_diff": worst_separable,
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
