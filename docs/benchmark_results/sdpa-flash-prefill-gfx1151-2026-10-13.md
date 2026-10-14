# Flash SDPA chunked prefill on ROCm: Radeon 8060S (gfx1151), 2026-10-13

Why long-context chunked prefill was slow and what changed it (lablup/mlxcel#2251, part of #1801).

Conclusion: the fused SDPA prefill kernel, `kernel_sdpa_flash_wmma`, ran at 0.95 to 1.49 TFLOP/s at the chunked-prefill shapes while the same device's f16 GEMM path delivers about 14 TFLOP/s at the same shape, and the gap was per-K-block scalar overhead, not the matrix units. The online softmax rescaled every O accumulator tile through LDS on every K block (a store, scale and reload of all D_TILES f32 tiles, 3 barriers per tile), and the kernel re-staged Q from global memory on every K block. The fix defers the accumulator rescale until a row's block max outgrows its softmax frame by more than a 10-point margin (never in the steady state of a chunked prefill) and stages Q once per block. The 2,048-query chunk call fell from 540.1 to 429.2 ms at 16K keys (0.95 to 1.20 TFLOP/s), 1.26x, with 1.28x at 2K and 1.30x at 8K; the non-WMMA fallback arm (`MLX_SDPA_NO_WMMA=1`) is another 3.5 to 5.7x slower. The model-level session (bench-decode, batch-4 server) is not measured yet; the kernel-level numbers below are the probe's medians.

## Environment

AMD Ryzen AI MAX+ 395 with Radeon 8060S (`gfx1151`, RDNA 3.5, wave32), HIP 7.15, Debian 13. mlxcel at `1e8c7022` (before) and `1e8c7022` plus this change (after), `--features rocm`, test-fast builds. Both arms ran back to back on an idle GPU (`rocm-smi --showpids` empty before and after each arm, load under 0.1). Harness scripts are in this directory (`session.sh`, `harness/`); the guarded model-level session described there did not run on a shared host.

Probe: `examples/sdpa_prefill_probe` (issue #2251), the maskless causal `fast_scaled_dot_product_attention_causal` call at `q=[1, 32, 2048, 128]`, `k/v=[1, 8, kL, 128]` f16, for `kL` of 2,048, 8,192 and 16,384. Each cell is the probe's internal median of 5 timed calls after 2 warmups; the probe also times a dense f16 `matmul` of the QK^T shape through the same allocator path (the device GEMM bound).

## Per-call time and TFLOP/s of the flash SDPA kernel

Probe medians (each the probe's median of 5 calls on an idle GPU), `q_len=2048`, GQA 32/8, head dim 128, causal, f16. TFLOP/s uses the causal-aware FLOP count (`4 * H * D * (qL * offset + qL * (qL + 1) / 2)` with `offset = kL - qL`).

| Keys | ms/call, before | after | TFLOP/s, before | after | Dense f16 GEMM ms (bound) |
|---|---:|---:|---:|---:|---:|
| 2,048 | 24.4 | 19.0 | 1.41 | 1.81 | 0.2 |
| 8,192 | 161.7 | 124.5 | 1.49 | 1.93 | 0.3 |
| 16,384 | 540.1 | 429.2 | 0.95 | 1.20 | 0.6 |

The limiting factor: per-K-block scalar overhead. Each 64-row query block loops over up to 256 K blocks at 16K keys, and every K block paid a full online-softmax pass plus an O-accumulator rescale that stores all D_TILES f32 accumulator tiles to LDS, scales them by `wave_scale` in LDS and reloads them (3 barriers per tile, 24 barriers per K block at D=128), plus a re-staging of Q from global memory. The WMMA matrix work itself is 64 mma_sync per wave per K block. With the accumulator rescale deferred to a 10-point staleness margin and Q staged once per block, the steady-state K block costs the WMMA matmuls, the score store and the softmax pass. For comparison, the non-WMMA fallback arm (`MLX_SDPA_NO_WMMA=1`) measured 103.2 / 709.0 / 1516.8 ms at the three key lengths (0.33 to 0.34 TFLOP/s), so the WMMA path is 3.5 to 5.7x the fallback and the fix is 1.26 to 1.30x on top of it.

## Model-level before and after

Not measured yet. The planned session is `mlxcel-bench-decode -p profile -n 128 --warmup-tokens 20 --ignore-eos --prompt-tokens N --temperature 0 --top-p 1.0` (pp512 and pp16384, medians of 3 guarded rounds, arms alternated per round) and a batch-4 server run (`mlxcel-server --parallel 4 --ctx-size 131072 --no-prompt-cache --metrics`, per-request decode tok/s from `scripts/bench_serving_concurrency.py`); both went through `scripts/rocm_gpu_guard.sh` (`session.sh`, `harness/`) but the shared gfx1151 host kept the guard lock busy and the session did not complete before this record was filed.

## Raw data

Probe outputs for both arms and the pre-fix arm are in the PR description and the test module doc (`tests/rocm_sdpa_flash_wmma_prefill.rs`); the harness scripts for the model-level session are `session.sh` and `harness/` in this directory.
