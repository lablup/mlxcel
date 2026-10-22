#!/usr/bin/env python3
# Copyright 2025-2026 Lablup Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the tiny BF16 case that distinguishes compiled Sapiens2 RoPE."""

import argparse
import json
import pathlib

import mlx.core as mx
import numpy as np


@mx.compile
def compiled_rope(x: mx.array, sin: mx.array, cos: mx.array) -> mx.array:
    batch, heads, tokens, width = x.shape
    values = x.astype(mx.float32).reshape(batch, heads, tokens, 2, width // 2)
    return (values * cos + values[..., ::-1, :] * sin).reshape(x.shape).astype(x.dtype)


def eager_rope(x: mx.array, sin: mx.array, cos: mx.array) -> mx.array:
    batch, heads, tokens, width = x.shape
    values = x.astype(mx.float32).reshape(batch, heads, tokens, 2, width // 2)
    reversed_values = mx.concatenate(
        [values[..., 1:2, :], values[..., 0:1, :]], axis=3
    )
    return (values * cos + reversed_values * sin).reshape(x.shape).astype(x.dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)
    x = mx.array([-0.134765625, -0.07470703125], dtype=mx.bfloat16).reshape(
        1, 1, 1, 2
    )
    sin = mx.array(
        [0.48480960726737976, -0.48480960726737976], dtype=mx.float32
    ).reshape(1, 2, 1)
    cos = mx.array([0.874619722366333], dtype=mx.float32).reshape(1, 1, 1)
    compiled = compiled_rope(x, sin, cos)
    eager = eager_rope(x, sin, cos)
    compiled_values = np.array(compiled.astype(mx.float32))
    eager_values = np.array(eager.astype(mx.float32))
    if np.array_equal(compiled_values, eager_values):
        raise RuntimeError("compiled and eager RoPE unexpectedly agree")

    output = {
        name: np.array(value.astype(mx.float32)).tolist()
        for name, value in {
            "input": x,
            "sin": sin,
            "cos": cos,
            "compiled": compiled,
            "eager": eager,
        }.items()
    }
    output.update(
        runtime=mx.__version__,
        backend=str(mx.default_device()),
        source_index=[0, 0, 1074, 55],
    )
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
