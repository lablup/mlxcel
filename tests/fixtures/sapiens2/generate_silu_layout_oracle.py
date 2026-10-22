#!/usr/bin/env python3
# Copyright 2025-2026 Lablup Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the BF16 case that distinguishes compiled Sapiens2 SiLU."""

import argparse
from functools import partial
import json
import pathlib

import mlx.core as mx
import numpy as np


def values(length: int, seed: int) -> np.ndarray:
    indices = np.arange(length, dtype=np.uint32) + np.uint32(seed)
    hashed = indices * np.uint32(1_664_525) + np.uint32(1_013_904_223)
    bits = (
        np.uint32(0x3800)
        + ((hashed >> np.uint32(8)) % np.uint32(2048))
        | ((hashed & np.uint32(1)) << np.uint32(15))
    )
    return (bits << np.uint32(16)).view(np.float32)


@partial(mx.compile, shapeless=True)
def compiled_silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)
    channels = 256
    x = mx.array(values(256 * channels, 2371).reshape(1, channels, 16, 16))
    x = x.astype(mx.bfloat16).transpose(0, 2, 3, 1)
    weight = mx.array(values(channels * channels, 991).reshape(channels, channels))
    weight = weight.astype(mx.bfloat16)
    bias = mx.zeros((channels,), dtype=mx.bfloat16)
    compiled_activation = compiled_silu(x)
    eager_activation = x * mx.sigmoid(x)
    if not np.array_equal(
        np.array(compiled_activation.astype(mx.float32)),
        np.array(eager_activation.astype(mx.float32)),
    ):
        raise RuntimeError("compiled and eager SiLU values unexpectedly differ")
    compiled = np.array(
        mx.addmm(bias, compiled_activation.reshape(256, channels), weight.T).astype(
            mx.float32
        )
    )
    eager = np.array(
        mx.addmm(bias, eager_activation.reshape(256, channels), weight.T).astype(
            mx.float32
        )
    )
    index = (33, 12)
    if np.array_equal(compiled, eager):
        raise RuntimeError("compiled and eager SiLU layouts unexpectedly agree downstream")

    output = {
        "index": list(index),
        "compiled": float(compiled[index]),
        "eager": float(eager[index]),
        "runtime": mx.__version__,
        "backend": str(mx.default_device()),
        "channels": channels,
        "weight_seed": 991,
        "input_seed": 2371,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
