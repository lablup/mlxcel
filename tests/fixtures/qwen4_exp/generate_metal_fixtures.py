#!/usr/bin/env python3
"""Regenerate bf16 Metal parity fixtures with the recorded mlx-vlm revision.

Run with the same Python environment as generate_fixtures.py. These fixtures
exercise native Metal arithmetic separately from the portable float32 cases.
"""

import json

import mlx.core as mx
import mlx.nn as nn

from generate_fixtures import (FIXTURE_DIR, TINY_CONFIG, as_list, build_weights, hashed_input, provenance, text_config)
from mlx_vlm.models.qwen4_exp.language import Qwen4ExpSparseMoeBlock, Qwen4ExpBatchInvariantForward
from mlx_vlm.models.qwen3_5.language import Qwen3_5RotaryEmbedding


def main():
    mx.set_default_device(mx.gpu)
    spec = {"name": "input.rotary_bf16", "shape": [1, 1, 3, 128], "scale": 2.0}
    positions = [0, 4097, 8191, 7, 2051, 4099, 13, 1027, 3073]
    rotary = Qwen3_5RotaryEmbedding(64, base=10000000, mrope_section=[11, 11, 10])
    x = hashed_input(spec["name"], tuple(spec["shape"]), spec["scale"]).astype(mx.bfloat16)
    output, _ = rotary.apply_rotary(x, x, mx.array(positions).reshape(3, 1, 3))
    payload = {
        "provenance": {**provenance(), "device": "metal"},
        "input": spec,
        "positions": positions,
        "output": as_list(output),
    }
    path = FIXTURE_DIR / "rotary_bf16.json"
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"wrote {path}")

    overrides = {"hidden_size": 64, "moe_intermediate_size": 64,
                 "shared_expert_intermediate_size": 64}
    moe = Qwen4ExpSparseMoeBlock(text_config(overrides))
    specs = build_weights(moe, "moe.")
    moe.set_dtype(mx.bfloat16)
    nn.quantize(moe, group_size=32, bits=4,
                class_predicate=lambda path, module: path.endswith("_proj"))
    spec = {"name": "input.moe_bf16", "shape": [1, 1, 64], "scale": 2.0}
    x = hashed_input(spec["name"], spec["shape"], spec["scale"]).astype(mx.bfloat16)
    output = Qwen4ExpBatchInvariantForward()._feed_forward(moe, x)
    payload = {
        "provenance": {**provenance(), "device": "metal"},
        "tiny_config": {**TINY_CONFIG, **overrides,
                        "quantization": {"group_size": 32, "bits": 4}},
        "tensors": specs, "input": spec, "output": as_list(output),
    }
    path = FIXTURE_DIR / "moe_bf16.json"
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
