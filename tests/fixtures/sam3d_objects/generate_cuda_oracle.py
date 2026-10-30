#!/usr/bin/env python3
# Copyright 2025-2026 Lablup Inc.
# SPDX-License-Identifier: Apache-2.0
"""Capture the upstream MLX CUDA values of the compiled Objects layers and the
depth bicubic resize, as sparse overrides of the pinned ROCm / x86_64 oracles.

Run with an upstream ``mlx`` CUDA build and ``mlx-vlm`` (for
``mlx_vlm.models.interpolate``) on the CUDA device whose values are recorded,
with ``MLX_ENABLE_TF32=0``:

    MLX_ENABLE_TF32=0 python -I generate_cuda_oracle.py <tests/fixtures> <output.json>

Only elements that differ from the pinned oracle are written, so the output
stays small and shows exactly where CUDA's fused kernels round differently.
"""

import hashlib
import json
import math
import pathlib
import platform
import subprocess
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm.models.interpolate import resize_bicubic_nhwc


def overrides(actual: mx.array, pinned: mx.array) -> dict:
    a = np.array(actual.astype(mx.float32)).ravel()
    p = np.array(pinned.astype(mx.float32)).ravel()
    index = np.flatnonzero(a != p)
    return {
        "dtype": str(actual.dtype),
        "count": int(a.size),
        "index": index.tolist(),
        "value": a[index].tolist(),
    }


def main() -> None:
    fixtures = pathlib.Path(sys.argv[1])
    output = pathlib.Path(sys.argv[2])
    if output.exists():
        raise SystemExit("refusing to overwrite an existing oracle")
    mx.set_default_device(mx.gpu)

    layers = mx.load(str(fixtures / "sam3d_objects/layers.safetensors"))
    modulate = mx.compile(lambda x, s, h: x * (1 + s) + h, shapeless=True)
    residual = mx.compile(lambda x, h, g: x + h * g, shapeless=True)

    tanh = mx.load(str(fixtures / "sam3d_objects/gelu-tanh-power.safetensors"))

    @mx.compile
    def multiply(x):
        return x * (0.5 * (1 + mx.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * ((x * x) * x)))))

    depth = json.loads((fixtures / "sam3d_objects/resize-depth-policy-provenance.json").read_text())
    x = mx.arange(18, dtype=mx.float32).reshape(1, 3, 3, 2) / 17
    bicubic = resize_bicubic_nhwc(x, scale_factor=((4 + 0.1) / 3, (6 + 0.1) / 3))
    pinned_bicubic = mx.array(np.array(depth["bicubic"]["values"], dtype=np.float32))

    result = {
        "modulate": overrides(
            modulate(layers["modulate.x"], layers["modulate.scale"], layers["modulate.shift"]),
            layers["modulate.out"],
        ),
        "residual": overrides(
            residual(layers["residual.x"], layers["residual.h"], layers["residual.gate"]),
            layers["residual.out"],
        ),
        "bf16_grid_upstream": overrides(
            nn.GELU(approx="tanh")(tanh["bf16_grid_x"]), tanh["bf16_grid_upstream"]
        ),
        "bf16_grid_multiply": overrides(
            multiply(tanh["bf16_grid_x"]), tanh["bf16_grid_multiply"]
        ),
        "bicubic": overrides(bicubic, pinned_bicubic),
    }
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        gpu = "unknown"
    import mlx_vlm
    result["provenance"] = {
        "mlx": mx.__version__,
        "mlx_vlm": mlx_vlm.__version__,
        "device": str(mx.default_device()),
        "gpu": gpu,
        "machine": platform.machine(),
        "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
    }
    # One top-level key per line keeps the file small and diffable.
    lines = [f"  {json.dumps(key)}: {json.dumps(value)}" for key, value in result.items()]
    output.write_text("{\n" + ",\n".join(lines) + "\n}\n")
    print(output, hashlib.sha256(output.read_bytes()).hexdigest())
    for key, value in result.items():
        if key != "provenance":
            print(key, len(value["index"]), "differ")


if __name__ == "__main__":
    main()
