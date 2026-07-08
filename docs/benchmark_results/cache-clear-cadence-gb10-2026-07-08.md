# CUDA cache-clear cadence + `MLXCEL_CACHE_LIMIT` (issue #627, GB10)

Date: 2026-07-08. Host: NVIDIA GB10 (cc12.1, unified memory), driver 580.159.03.
Binary: `make release-cuda` at the #627 branch point (base `aff12a7`).
Bench: `mlxcel-bench-decode`, prompt `"Hello, how are you today?"`, warmup pass + measured pass.

## Question

The decode loops and batch scheduler called `clear_memory_cache()` every 256
generated tokens (matching Python mlx-lm). On CUDA this was suspected to be a
net loss: dropping cached buffers forces the CUDA memory pool to reallocate and
can defeat MLX's CUDA-graph executable cache, whose reuse depends on stable
buffer addresses (ml-explore/mlx#2358). #627 asks whether the periodic clear
should stay on CUDA, and adds `MLXCEL_CACHE_LIMIT` as the intended mechanism to
bound the buffer cache instead.

## Cadence matrix (2048-token decode)

`MLXCEL_CACHE_CLEAR_INTERVAL` sets the cadence; `0` disables the periodic clear.

| Model | cadence | decode tok/s | MLX peak | gen tokens |
|-------|---------|--------------|----------|------------|
| qwen2.5-0.5b-bf16 (overhead-bound) | 256 (prev default) | 210.79 | 1.04 GB | 2048 |
| qwen2.5-0.5b-bf16 | 4096 | 210.94 | 1.04 GB | 2048 |
| qwen2.5-0.5b-bf16 | 0 (disabled) | 210.58 | 1.04 GB | 2048 |
| llama-3.1-8b-4bit (bandwidth-bound) | 256 | 53.77 | 4.67 GB | 108 (EOS) |
| llama-3.1-8b-4bit | 4096 | 53.75 | 4.67 GB | 108 (EOS) |
| llama-3.1-8b-4bit | 0 (disabled) | 53.74 | 4.67 GB | 108 (EOS) |

The cadence has no measurable effect on decode throughput (all within ~0.2%,
i.e. run-to-run noise) or peak memory (identical) on either model. llama-3.1-8b
stops at EOS (108 tokens) so the every-256 clear never engages there; qwen at
2048 tokens fires it 7 times with no observable cost.

## 16k soak (memory-safety of disabling the clear)

qwen2.5-0.5b-bf16, `--max-tokens 16384` (model sustained the full 16384; ~64
would-be clears at cadence 256).

| cfg | decode tok/s | MLX peak | gen tokens |
|-----|--------------|----------|------------|
| clear on (256) | 196.38 | 1.29 GB | 16384 |
| clear off (0) | 196.38 | 1.24 GB | 16384 |
| clear off + `MLXCEL_CACHE_LIMIT=2G` | 196.10 | 1.27 GB | 16384 |

Disabling the periodic clear does not grow memory over a long run: peak with the
clear off (1.24 GB) is at or below peak with it on (1.29 GB). `MLXCEL_CACHE_LIMIT=2G`
runs clean and stays well under the cap. Throughput is identical across all three.

## Conclusion

On GB10 CUDA the periodic `clear_memory_cache` is dead work: it neither improves
throughput nor bounds memory (memory is bounded fine without it, and is if
anything marginally lower). The likely reason the mlx#2358 graph-reuse penalty
does not appear is that CUDA decode graphs are currently disabled here (see the
#688 decode graph-race fix), so there is no graph cache for the clear to defeat.

Chosen defaults (`DEFAULT_CACHE_CLEAR_INTERVAL`), overridable via
`MLXCEL_CACHE_CLEAR_INTERVAL`:

- **CUDA: `0` (disabled).** Removes a per-256-token no-op with zero measured
  downside (no throughput or memory regression across the matrix and the 16k soak).
- **Metal/CPU: `256` (unchanged).** Keeps the cheap buffer-cache trim of Python
  mlx-lm; not re-measured here (this is a GB10/CUDA study).

`MLXCEL_CACHE_LIMIT` is added as an operator knob (previously the bridge
`set_cache_limit` symbol was never called) and is the intended way to bound
cache growth on CUDA now that the periodic clear is off by default.

This is an honest null result on throughput: #627's value is the operator knob
plus removing dead work on CUDA, not a speedup.
