#!/usr/bin/env python3
"""Regenerate the literal probes in dense/interpolate_tests.rs.

Requires PyTorch. Run from the repository root and copy the printed Rust
slices into the named test cases. Keeping fixtures as literals means Rust tests
do not need a Python environment or a second interpolation implementation.
"""

import torch
import torch.nn.functional as F


def source(height: int, width: int) -> torch.Tensor:
    values = [(((index * 37) % 101) - 50) / 17 for index in range(height * width)]
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, height, width)


def probes(values: torch.Tensor) -> str:
    flat = values.flatten().tolist()
    indices = sorted({0, 1, 4, len(flat) // 7, len(flat) // 3, len(flat) // 2, len(flat) - 5, len(flat) - 2, len(flat) - 1})
    return "&[" + ", ".join(f"({index}, {flat[index]:.9g})" for index in indices) + "]"


for height, width, out_height, out_width, suffix, scale in [
    (7, 9, 14, 5, "a", (2.0, 0.6)),
    (14, 5, 7, 9, "b", (0.5, 1.8)),
]:
    tensor = source(height, width)
    for antialias in (False, True):
        result = F.interpolate(
            tensor,
            size=(out_height, out_width),
            mode="bilinear",
            align_corners=False,
            antialias=antialias,
        )
        print(f"bilinear_{suffix}_{antialias} = {probes(result)}")
    for antialias in (False, True):
        result = F.interpolate(
            tensor,
            size=(out_height, out_width),
            mode="bicubic",
            align_corners=False,
            antialias=antialias,
        )
        print(f"bicubic_size_{suffix}_{antialias} = {probes(result)}")
    for antialias in (False, True):
        result = F.interpolate(
            tensor,
            scale_factor=scale,
            mode="bicubic",
            align_corners=False,
            antialias=antialias,
            recompute_scale_factor=False,
        )
        print(f"bicubic_scale_{suffix}_{antialias} = {probes(result)}")
    result = F.interpolate(tensor, size=(out_height, out_width), mode="nearest")
    print(f"nearest_{suffix} = {probes(result)}")
