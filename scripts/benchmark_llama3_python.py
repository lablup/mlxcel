#!/usr/bin/env python3
"""
Benchmark comparing mlxcel-core (Rust) vs mlx-lm (Python) for Llama 3.1 model.

This script measures Python mlx-lm performance for comparison with the Rust implementation.
"""

import time
import mlx.core as mx
from mlx_lm import load, generate

MODEL_PATH = "models/meta-llama-3.1-8b-instruct-4bit"

def main():
    print("=" * 55)
    print("  Llama 3.1 8B: mlx-lm (Python) Benchmark")
    print("=" * 55)
    print()
    print(f"Model path: {MODEL_PATH}")
    print()

    # Test prompt - same as Rust benchmark
    # Use token IDs directly for exact match
    prompt_tokens_rust = [
        128000,  # <|begin_of_text|>
        9906,    # "Hello"
        11,      # ","
        1268,    # " how"
        527,     # " are"
        499,     # " you"
        3432,    # " today"
        30,      # "?"
        358,     # " I"
        1097,    # " am"
    ]
    prompt_text = "Hello, how are you today? I am"
    max_tokens = 50

    print(f"Prompt: {prompt_text}")
    print(f"Max tokens: {max_tokens}")
    print()

    # =========================================================================
    # Load model
    # =========================================================================
    print("Loading mlx-lm model...")
    load_start = time.perf_counter()
    model, tokenizer = load(MODEL_PATH)
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.2f}s")
    print()

    # Use same prompt tokens as Rust benchmark for exact comparison
    prompt_tokens = prompt_tokens_rust
    print(f"Prompt tokens: {prompt_tokens} ({len(prompt_tokens)} tokens)")
    print()

    # =========================================================================
    # Warmup
    # =========================================================================
    print("Warming up...")
    _ = generate(model, tokenizer, prompt=prompt_text, max_tokens=3, verbose=False)
    mx.metal.clear_cache()
    print()

    # =========================================================================
    # Prefill benchmark
    # =========================================================================
    print("Running prefill benchmark...")
    input_ids = mx.array([prompt_tokens])

    prefill_start = time.perf_counter()
    logits = model(input_ids)
    mx.eval(logits)
    prefill_time = time.perf_counter() - prefill_start
    print(f"Prefill time: {prefill_time * 1000:.2f} ms")
    print()

    # =========================================================================
    # Step-by-step generation for accurate measurement
    # =========================================================================
    print("Running step-by-step generation benchmark...")

    mx.metal.clear_cache()

    # Prefill
    input_ids = mx.array([prompt_tokens])
    cache = None

    step_prefill_start = time.perf_counter()
    logits = model(input_ids, cache=cache)
    mx.eval(logits)
    step_prefill_time = time.perf_counter() - step_prefill_start

    # Generation loop
    step_gen_start = time.perf_counter()

    tokens = list(prompt_tokens)
    eos_token = tokenizer.eos_token_id or 128001  # Llama 3 EOS

    for i in range(max_tokens):
        # Get next token (greedy)
        next_logits = logits[:, -1, :]
        next_token = mx.argmax(next_logits, axis=-1)
        mx.eval(next_token)

        token_id = next_token.item()
        tokens.append(token_id)

        # Check EOS
        if token_id == eos_token:
            break

        # Next forward pass (single token)
        input_ids = next_token.reshape(1, 1)
        logits = model(input_ids, cache=cache)
        mx.eval(logits)

    step_gen_time = time.perf_counter() - step_gen_start
    step_total_time = step_prefill_time + step_gen_time
    step_generated = len(tokens) - len(prompt_tokens)
    step_throughput = step_generated / step_gen_time if step_gen_time > 0 else 0

    print()
    print("=" * 55)
    print("  Results (mlx-lm Python - Step-by-step)")
    print("=" * 55)
    print()
    print(f"  Model load time: {load_time:.2f}s")
    print(f"  Prefill time: {step_prefill_time * 1000:.2f} ms")
    print(f"  Generation time: {step_gen_time * 1000:.2f} ms")
    print(f"  Total time: {step_total_time * 1000:.2f} ms")
    print(f"  Generated tokens: {step_generated}")
    print(f"  Throughput: {step_throughput:.2f} tok/s")
    print()

    # Generated text
    generated_text = tokenizer.decode(tokens)
    print(f"Generated text: {generated_text[:100]}...")
    print()

    # Cleanup
    mx.metal.clear_cache()

if __name__ == "__main__":
    main()
