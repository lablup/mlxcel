# Paged-attention decode: CUDA port of the fused kernel (issue #634, GB10)

Date: 2026-07-10. Host: NVIDIA GB10 (Grace-Blackwell, sm_121 / cc 12.1), CUDA 13.0, MLX 0.32.1.
Binary: `cargo build --release --features cuda`. Bench: `examples/paged_attention_kernel_bench.rs` (kernel A/B), `mlxcel-bench-decode` (sanity A/B), `ffi_tests::test_fused_paged_decode_native_vs_fallback_matrix` (numerical parity).

## Summary

#634 ports the fused split-K paged-attention decode kernel (`src/lib/mlx-cpp/turbo/paged_attention.cpp`) from Metal-only `mx.fast.metal_kernel` to a `mx.fast.cuda_kernel` body, so CUDA no longer falls back to gather-then-SDPA on every step of the pooled paged decode path. The thread mapping and grid geometry carry over unchanged from Metal (one block per (batch, query head), `(32, NumSplits)` threads, online-softmax accumulation); the reduction changes from `simd_sum` to a `__shfl_xor_sync` butterfly all-reduce. `select_pooled_paged_dispatch` makes native the default on CUDA for any single-slab layer, since the Metal-measured batch>=4/ctx<=4096 ceiling (ADR 0001) does not apply on CUDA, where the fused win grows with context instead of losing it. Multi-slab layers still decline to gather, same as on Metal.

## Numerical parity

`test_fused_paged_decode_native_vs_fallback_matrix` runs the native kernel against the gather fallback over 60 configs (head dims 64/80/96/128, GQA ratios 1:1/4:1/8:1, page-boundary cases, batch > 1): worst RMS 5.45e-5, worst max-abs 2.35e-4, both well under the 5e-3 fp16 parity threshold.

## Kernel A/B (`examples/paged_attention_kernel_bench`)

This is the only path that reaches the fused kernel; `select=native` fires on CUDA once a layer is single-slab.

| shape | select | fused vs gather |
|---|---|---:|
| batch 1, ctx 512 | native | 1.31x |
| batch 1, ctx 1024 | native | 1.49x |
| single-slab batched island | native | 2.55x / 6.20x / 1.27x |
| multi-slab shapes | gather (declined) | - |

Run-to-run variance: these kernels sit in the 200-900 us range where GB10 timing jitter is material. An independent re-run of the same sweep measured b1/512 at 0.88x (gather 266 us vs fused 303 us) and b1/1024 at 1.97x, so treat the single-batch ctx-512 case as noise-dominated around parity; the batched island and the b1/1024 win reproduce across runs.

## Sanity A/B (`mlxcel-bench-decode`, llama-3.1-8b-4bit, 2048-token prompt, 32 decode tokens)

`MLXCEL_PAGED_ATTENTION_NATIVE=0` decode 50.93 tok/s, `=1` decode 50.94 tok/s. The two runs land within noise of each other because `mlxcel-bench-decode` uses dense single-stream caches and never reaches the fused kernel either way; this A/B confirms no accidental path divergence rather than a performance win.

## Reachability caveat

The fused kernel and its selector are library-only surface. #720 retired the pooled decode entry point (`paged_decode_attention_pooled` -> `PagedBlockPool::paged_decode_fused`) from the server: `mlxcel-server --decode-storage paged` routes pool-backed layers through the per-sequence `update_and_fetch` gather-then-SDPA intercept, and `mlxcel-bench-decode` uses dense single-stream caches, so neither reaches this kernel and `MLXCEL_PAGED_ATTENTION_NATIVE` does not change their output. The single-slab constraint (#235) also caps the servable context before the kernel declines to gather, which bounds the achievable long-context win independently of this port. Wiring the fused kernel onto the server decode path would reverse #720's deliberate library-only decision and stays out of scope for #634.

## Reproduce

```bash
cargo run --release --features cuda --example paged_attention_kernel_bench
cargo test --release --features cuda -p mlxcel-core --lib ffi_tests::test_fused_paged_decode_native_vs_fallback_matrix -- --test-threads=1
```

Refs #634, #720, #235, [ADR 0001](../adr/0001-paged-attention-gather-vs-fused-kernel.md).
