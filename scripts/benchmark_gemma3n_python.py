#!/usr/bin/env python3
"""
Benchmark comparing mlxcel-core (Rust) vs mlx-lm (Python) for Gemma 3n model.

This script measures Python mlx-lm performance for comparison with the Rust implementation.
"""

import time
import mlx.core as mx
from mlx_lm import load, generate

MODEL_PATH = "models/gemma-3n-e4b-it-4bit"

def main():
    print("=" * 55)
    print("  Gemma 3n: mlx-lm (Python) Benchmark")
    print("=" * 55)
    print()
    print(f"Model path: {MODEL_PATH}")
    print()

    # Test prompt tokens (same as Rust benchmark)
    prompt_tokens = [2, 1841, 235269, 1368, 708, 692, 3646, 235336]
    max_tokens = 30

    print(f"Prompt tokens: {prompt_tokens} ({len(prompt_tokens)} tokens)")
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

    # =========================================================================
    # Warmup
    # =========================================================================
    print("Warming up...")
    prompt_text = tokenizer.decode(prompt_tokens)
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
    # Generation benchmark (using generate_step for accurate timing)
    # =========================================================================
    print("Running generation benchmark...")

    # Reset and run full generation
    gen_start = time.perf_counter()

    # Use the generate function for consistent comparison
    output = generate(
        model,
        tokenizer,
        prompt=prompt_text,
        max_tokens=max_tokens,
        verbose=False,
        temp=0.0,  # Greedy sampling
    )

    gen_total_time = time.perf_counter() - gen_start

    # Count generated tokens (approximate from output)
    output_tokens = tokenizer.encode(output)
    generated_count = len(output_tokens) - len(prompt_tokens)
    if generated_count < 0:
        generated_count = max_tokens  # Fallback

    # Calculate generation-only time (subtract estimated prefill)
    gen_only_time = max(0, gen_total_time - prefill_time)

    throughput = generated_count / gen_only_time if gen_only_time > 0 else generated_count / gen_total_time

    print()
    print("=" * 55)
    print("  Results (mlx-lm Python)")
    print("=" * 55)
    print()
    print(f"  Model load time: {load_time:.2f}s")
    print(f"  Prefill time: {prefill_time * 1000:.2f} ms")
    print(f"  Generation time: {gen_only_time * 1000:.2f} ms")
    print(f"  Total time: {gen_total_time * 1000:.2f} ms")
    print(f"  Generated tokens: {generated_count}")
    print(f"  Throughput: {throughput:.2f} tok/s")
    print()

    # =========================================================================
    # Manual step-by-step generation for more accurate measurement
    # =========================================================================
    print("=" * 55)
    print("  Step-by-step Generation (more accurate)")
    print("=" * 55)
    print()

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
    for i in range(max_tokens):
        # Get next token (greedy)
        next_logits = logits[:, -1, :]
        next_token = mx.argmax(next_logits, axis=-1)
        mx.eval(next_token)

        token_id = next_token.item()
        tokens.append(token_id)

        # Check EOS (Gemma uses 1 or 107)
        if token_id == 1 or token_id == 107:
            break

        # Next forward pass (single token)
        input_ids = next_token.reshape(1, 1)
        logits = model(input_ids, cache=cache)
        mx.eval(logits)

    step_gen_time = time.perf_counter() - step_gen_start
    step_total_time = step_prefill_time + step_gen_time
    step_generated = len(tokens) - len(prompt_tokens)
    step_throughput = step_generated / step_gen_time if step_gen_time > 0 else 0

    print(f"  Prefill time: {step_prefill_time * 1000:.2f} ms")
    print(f"  Generation time: {step_gen_time * 1000:.2f} ms")
    print(f"  Total time: {step_total_time * 1000:.2f} ms")
    print(f"  Generated tokens: {step_generated}")
    print(f"  Throughput: {step_throughput:.2f} tok/s")
    print()

    # Cleanup
    mx.metal.clear_cache()

if __name__ == "__main__":
    main()
