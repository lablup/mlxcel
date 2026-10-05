#!/usr/bin/env python3
"""
Python mlx-lm benchmark script for comparison with Rust implementations.
Outputs results in a machine-readable format.
"""

import sys
import time
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.models.cache import make_prompt_cache

def benchmark_model(model_path: str, max_tokens: int = 50):
    """Benchmark a model and return results."""

    # Prompt tokens for "Hello, how are you today? I am"
    prompt = "Hello, how are you today? I am"

    print(f"Loading model from: {model_path}", file=sys.stderr)
    start = time.perf_counter()
    model, tokenizer = load(model_path)
    load_time = time.perf_counter() - start
    print(f"Model loaded in {load_time:.2f}s", file=sys.stderr)

    # Tokenize prompt
    tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_len = len(tokens)
    print(f"Prompt: '{prompt}' -> {prompt_len} tokens", file=sys.stderr)

    # Warmup
    print("Warming up...", file=sys.stderr)
    _ = generate(model, tokenizer, prompt=prompt, max_tokens=5, verbose=False)
    mx.metal.clear_cache()

    # Benchmark prefill
    print("Benchmarking prefill...", file=sys.stderr)
    input_ids = mx.array([tokens])
    cache = make_prompt_cache(model)

    prefill_start = time.perf_counter()
    logits = model(input_ids, cache=cache)
    mx.eval(logits)
    prefill_time = (time.perf_counter() - prefill_start) * 1000  # ms

    # Benchmark generation
    print(f"Benchmarking generation of {max_tokens} tokens...", file=sys.stderr)
    gen_start = time.perf_counter()
    output = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    total_time = (time.perf_counter() - gen_start) * 1000  # ms

    # Count generated tokens
    output_tokens = tokenizer.encode(output, add_special_tokens=False)
    gen_tokens = len(output_tokens) - prompt_len

    # Calculate throughput
    gen_time = total_time - prefill_time
    throughput = gen_tokens / (gen_time / 1000) if gen_time > 0 else 0

    # Output results in machine-readable format
    print(f"RESULT:prefill_ms={prefill_time:.2f}")
    print(f"RESULT:gen_tokens={gen_tokens}")
    print(f"RESULT:total_ms={total_time:.2f}")
    print(f"RESULT:throughput={throughput:.2f}")

    mx.metal.clear_cache()
    return prefill_time, throughput

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: benchmark_python.py <model_path> [max_tokens]", file=sys.stderr)
        sys.exit(1)

    model_path = sys.argv[1]
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    benchmark_model(model_path, max_tokens)
