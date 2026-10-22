#!/usr/bin/env python3
# Copyright 2025-2026 Lablup Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the BF16 case that distinguishes Sapiens2 deconv layouts."""

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)
    channels = 256
    weight = mx.array(
        values(channels * 4 * 4 * channels, 2371).reshape(channels, 4, 4, channels)
    ).astype(mx.bfloat16)
    columns = mx.array(values(9 * 4 * channels, 991).reshape(9, 4 * channels)).astype(
        mx.bfloat16
    )
    taps = mx.array([[3, 1], [2, 0]])
    upstream_weight = (
        mx.stack(
            [
                weight[:, taps[row]][:, :, taps[column]]
                for row in (0, 1)
                for column in (0, 1)
            ]
        )
        .transpose(2, 3, 4, 0, 1)
        .reshape(4 * channels, 4 * channels)
    )
    former_phases = []
    for rows in ((3, 1), (2, 0)):
        for column_indices in ((3, 1), (2, 0)):
            former_phases.append(
                mx.concatenate(
                    [
                        weight[:, row : row + 1, column : column + 1, :]
                        .reshape(channels, channels)
                        .T
                        for row in rows
                        for column in column_indices
                    ],
                    axis=0,
                )
            )
    former_weight = mx.concatenate(former_phases, axis=1)
    upstream = np.array((columns @ upstream_weight).astype(mx.float32))
    former = np.array((columns @ former_weight).astype(mx.float32))
    index = (4, 272)
    if np.array_equal(upstream, former):
        raise RuntimeError("upstream and former deconv layouts unexpectedly agree")

    output = {
        "index": list(index),
        "upstream": float(upstream[index]),
        "former_concat": float(former[index]),
        "runtime": mx.__version__,
        "backend": str(mx.default_device()),
        "channels": channels,
        "rows": 9,
        "weight_seed": 2371,
        "columns_seed": 991,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
