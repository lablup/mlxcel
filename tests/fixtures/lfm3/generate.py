# Copyright 2025-2026 Lablup Inc.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate with mlx-vlm and local pinned checkpoints. No network required.

python generate.py /path/to/LFM3-1.7B /path/to/LFM3-1.7B-MLX-8bit /output
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
from mlx_vlm.utils import load_model
from tokenizers import Tokenizer

PROMPTS = [
    "What is 17 + 25?",
    "The largest planet in the solar system is",
    "The station recorded the hourly temperature. " * 40 + "The final reading was",
]


def main(float_dir, quant_dir, out):
    mx.set_cache_limit(1024**3)
    mx.set_memory_limit(36 * 1024**3)
    tokenizer = Tokenizer.from_file(str(float_dir / "tokenizer.json"))
    records = [
        {"prompt": prompt, "ids": tokenizer.encode(prompt, add_special_tokens=False).ids}
        for prompt in PROMPTS
    ]
    assert len(records[-1]["ids"]) > 256
    model = load_model(float_dir, lazy=True).language_model
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    logits = {}
    for idx, record in enumerate(records):
        row = model(mx.array([record["ids"]])).logits[0, -1].astype(mx.float32)
        mx.eval(row)
        logits[str(idx)] = row
    mx.save_safetensors(str(out / "last_logits.safetensors"), logits)
    del model
    mx.clear_cache()

    model = load_model(quant_dir, lazy=True).language_model
    mx.eval(model.parameters())
    for record in records:
        cache = model.make_cache()
        tokens = mx.array([record["ids"]])
        greedy = []
        for _ in range(32):
            choice = int(mx.argmax(model(tokens, cache=cache).logits[0, -1]).item())
            greedy.append(choice)
            tokens = mx.array([[choice]])
        record["greedy"] = greedy
    (out / "prompts.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main(*(Path(arg) for arg in sys.argv[1:]))
