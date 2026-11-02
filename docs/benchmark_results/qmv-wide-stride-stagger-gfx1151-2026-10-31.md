# Staggered wide qmv walk on gfx1151, 2026-10-31

`qmv_wide_kernel` ran 2x slower at some layer shapes than at neighbouring shapes with the same bytes. A 9.44 MB GEMV took 109.6 us at K = 8192 and about 47 us at K = 8160 and K = 8224. This records the cause as far as it was measured, and the change: each block of a 4-bit kernel with `K % 2048 == 0` starts its walk over K at its own offset.

## Host and method

Radeon 8060S (gfx1151, 20 CU), 96 GiB carve-out, ROCm 10.0.0, base commit `1a257bcd` plus the working tree. One binary, `MLX_ROCM_QMV_WIDE_STAGGER=1` and `0`, 3 rounds per model, pp512/tg128 with 20 warmup tokens, greedy, `scripts/bench_decode.sh` inside one clean `scripts/rocm_gpu_guard.sh` session. In every round the on arm ran first, so the order was not randomized. The first table's control rows show what that and the run-to-run noise amount to.

## Shape sweep (microbench, 8 columns per block)

| N x K | Off (us) | Block phase (us) |
|---|---:|---:|
| 2048 x 8192 | 122 | 48 |
| 4096 x 4096 | 66.5 | 44 |
| 6144 x 4096 | 89 | 64 |
| 14336 x 4096 | 169 | 148 |
| 4096 x 14336 | 163 | 156 |
| 1360 x 12288 | 68.5 | 48 |
| 1024 x 16384 | 99.6 | 45 |
| 512 x 32768 | 184 | 44.5 |
| 2336 x 7168 | unchanged | unchanged |

A load-only kernel shows the same K = 8192 slowdown, so the arithmetic is not the cause. The mechanism (waves reading the same offset of columns whose stride is a large power of two, piling up on memory channels) is inferred from this pattern. Memory-channel mapping was not observed directly.

## Twelve models (decode tok/s, median of 3, range in brackets)

| Model | Widths (K) | Off | On | Ratio |
|---|---|---:|---:|---:|
| llama-3.2-1b-instruct-4bit | 2048, 8192 | 163.99 [162.8-172.3] | 237.90 [237.2-238.3] | 1.451 |
| llama-3.2-3b-instruct-4bit | 3072, 8192 | 83.25 [80.8-84.3] | 101.48 [101.0-101.7] | 1.219 |
| qwen3-8b-4bit | 4096, 12288 | 37.80 [37.8-37.8] | 45.60 [45.4-45.6] | 1.206 |
| mistral-7b-instruct-v0.3-4bit | 4096, 14336 | 42.50 [42.4-42.5] | 50.34 [50.3-50.4] | 1.184 |
| qwen3-1.7b-4bit | 2048, 6144 | 131.30 [127.0-134.1] | 154.21 [140.5-155.1] | 1.174 |
| Meta-Llama-3.1-8B-Instruct-4bit | 4096, 14336 | 40.71 [40.7-40.7] | 47.77 [47.6-47.8] | 1.173 |
| gemma-4-e2b-it-4bit | 1536, 6144 | 81.54 [79.1-81.6] | 90.37 [90.1-92.0] | 1.108 |
| Qwen3-30B-A3B-4bit | 2048, 6144 | 61.02 [58.8-62.0] | 67.15 [66.7-67.4] | 1.100 |
| gemma-3-4b-it-4bit | 2560, 10240 | 69.66 [69.2-69.7] | 71.81 [71.6-71.8] | 1.031 |
| Qwen2.5-7B-Instruct-4bit (control) | 3584, 18944 | 51.48 [51.3-51.7] | 51.61 [51.6-51.7] | 1.003 |
| granite-4.0-h-tiny-4bit (control) | 1536, 512 | 105.09 [103.8-105.1] | 105.14 [104.5-105.3] | 1.000 |
| qwen2.5-0.5b-4bit (control) | 896, 4864 | 359.98 [329.9-386.9] | 396.16 [394.5-397.4] | 1.101 |

No hidden or intermediate width of the control rows passes the gate, so both of their arms execute the same walk. The 10% on qwen2.5-0.5b is therefore noise, and a small fast model's ratio cannot be read below that. The Llama 1B, Llama 3B, Qwen3-8B, Mistral and Llama 8B gains have ranges that do not overlap.

Prefill ratios sit between 0.98 and 1.07, except Llama 3B at 0.793. Its off rounds spread 1642 to 2763 tok/s and its on rounds 2090 to 2670, so the ranges overlap and this is not read as a regression. The kernel takes at most 8 rows, which a 512-token prefill does not reach.

## The gate

The first version staggered every shape and regressed the shapes without aliasing, with non-overlapping ranges: granite-4.0-h-tiny (K = 1536) 104.15 to 99.06 tok/s, Qwen2.5-7B (K = 3584, 18944) 51.52 to 50.44. The gate removed both. The other rows of that run agree with the table above. Those two runs are in the measurement store as `stagger/` and `stagger2/`.

## Output check

`examples/logit_trace` at width one, 60 positions, on against off:

| Model | Top-1 disagreement | On decided positions | Perplexity on vs off |
|---|---:|---:|---:|
| llama-3.2-1b-instruct-4bit | 0 / 60 | 0 / 19 | +0.107% |
| Meta-Llama-3.1-8B-Instruct-4bit | 0 / 60 | 0 / 22 | -0.104% |

The kernel sums the same products in a different order, so small float differences are expected.

## Limits

8-bit weights stay unstaggered: no local checkpoint has an 8-bit layer that would pass the gate. `gather_qmv_wide_kernel` is unchanged. Other ROCm GPUs were not measured, and the gate's constant comes from this host. `MLX_ROCM_QMV_WIDE_STAGGER=0` restores the old walk.

## What is left on Llama 3.1 8B

Profiled with the gate in place, the kernel takes 94.06% of 20.22 GPU ms/token, so 19.0 ms for the 4.22 GB a token reads, about 222 GB/s against the 256 GB/s peak of this memory and the 250 GB/s a GPU read microbench reached. At 250 GB/s the same bytes take 16.9 ms, so at most about 2 ms per token (10%) remains in this kernel. This profiled run cost 29.6% against the plain run (36.86 against 47.78 tok/s) and its 6.9 ms/token host gap is inflated by the profiler; the plain token is 20.9 ms. The GPU figure is the part used here.

Merging the `gate` and `up` dispatches was not implemented. It keeps the bytes read and removes one dispatch per layer. A fit over the staggered shapes puts the fixed cost at 2.4 to 5 us per dispatch, and the host adds about 2 us. That bounds the gain at about 1% on Llama 3.1 8B and about 6.7% on qwen2.5-0.5b, whose run-to-run noise is about 10%. This is a bound from the microbench, not a measured merge. It is worth reopening on a GPU where the per-dispatch cost measures several times higher.
