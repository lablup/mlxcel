#!/usr/bin/env python3
"""Compare with examples/qwen4_exp_benchmark.rs using identical token IDs.

Usage: qwen4_exp_benchmark.py MODEL CORPUS PREFILL DECODE [REPEATS]
One full warmup precedes measured runs. Every run starts with empty state.
Decode is teacher-forced and synchronously evaluated, so both runtimes execute
the same expert routes even if a numerical regression changes the top token.
This measures model execution, excluding tokenization, loading and HTTP costs.
"""

import json
import sys
import time
from pathlib import Path

import mlx.core as mx

from laguna_oracle_trace import load_tokenizer
from qwen4_exp_oracle_trace import load_reference


def main():
    model_path, corpus, prefill, decode, *rest = sys.argv[1:]
    prefill, decode = int(prefill), int(decode)
    repeats = int(rest[0]) if rest else 3
    assert prefill > 0 and decode > 0 and repeats > 0
    tokenizer = load_tokenizer(Path(model_path))
    ids = tokenizer.encode(Path(corpus).read_text(), add_special_tokens=False).ids
    assert len(ids) >= prefill + decode
    # mlxcel leaves unused vision weights lazy. Drop the reference wrapper
    # so this text-only benchmark likewise excludes active vision allocations.
    model = load_reference(Path(model_path)).language_model
    for run in range(repeats + 1):
        cache = model.make_cache()
        mx.synchronize()
        mx.reset_peak_memory()
        prompt = mx.array([ids[:prefill]])
        start = time.perf_counter()
        logits = model(prompt, cache=cache).logits
        mx.eval(logits)
        prefill_seconds = time.perf_counter() - start
        del logits
        top_ids = []
        start = time.perf_counter()
        for token in ids[prefill:prefill + decode]:
            logits = model(mx.array([[token]]), cache=cache).logits
            top = mx.argmax(logits, axis=-1)
            mx.eval(top)
            top_ids.append(top.item())
        decode_seconds = time.perf_counter() - start
        print(json.dumps({
            "runtime": "mlx-vlm", "run": run, "warmup": run == 0,
            "prefill_tokens": prefill, "decode_tokens": decode,
            "prefill_seconds": prefill_seconds, "decode_seconds": decode_seconds,
            "prefill_tps": prefill / prefill_seconds,
            "decode_tps": decode / decode_seconds,
            "peak_bytes": mx.get_peak_memory(), "top_ids": top_ids,
        }), flush=True)
        del logits, top, cache


if __name__ == "__main__":
    main()
