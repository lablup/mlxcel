# Where the Laguna DFlash verify round's fixed cost goes on GB10, and the fix

Issue #1799. Host GB10 (sm_121), MLX pin `81ba1c6a`, CUDA release build, toolchain 1.97.1, source tree `2deae307` (the squash of PR #1771; byte-identical to the PR head `dab18fdf` that the earlier sweep measured). Full measurement record: `docs/benchmark_results/laguna-dflash-verify-cost-gb10-2026-09-11.md`.

## Background

PR #1771 shipped the Laguna DFlash drafter behind a fail-closed exactness gate and measured that no block width beats classic decode on this host: the verify round's device time fit a fixed 77 ms plus about 3 ms per row, 2.5x a single-token step even at 2 rows, and the PR attributed that to the multi-row forward running "eagerly and launch-bound". The issue was filed with that phrase demoted to a hypothesis: nothing in the pinned MLX capture path branches on row count, #1782 had just disproved the same intuition on Qwen 3.5, and neither of #1782's fixes applied here (this drafter was already bf16; Laguna has no gated-delta layers). Its sweep had also run with another session's build on the CPU. So the task was attribution first, on an idle host, with `nsys --cuda-graph-trace=node`, and a measured negative counted as a valid close.

## What the measurement said

Idle host (start gated on load1 under 0.6, load1 recorded per run, `ps` showing only the process under test), same binary on every arm, `mlxcel generate` with the 152-token raw code prompt of #1782, widths 2 to 16 interleaved, n = 3 with min and max. The idle host did not rescue the pairing: classic 29.28 tok/s (34.2 ms per step); block 2 at 0.52x, block 6 at 0.72x (the best), block 16 at 0.52x; device sync per round 71.6 ms plus 3.79 ms per row by least squares; and the round's three serial links (31 ms drafter host build, 4 ms verify host build, 80 ms device sync at block 2) add up to its 115 ms wall to within 0.1 ms.

The graph-side candidates all came back negative with bounds. `MLX_MAX_MB_PER_BUFFER=400`, `MLX_MAX_OPS_PER_BUFFER=100`, both raised, and `MLX_USE_CUDA_GRAPHS=0` each move the 2-row device sync by at most 3% and the round by at most 4%; capture off leaves block 8 faster than capture on, as on Qwen 3.5. Graph replay is healthy (under one `cudaGraphInstantiate` per 1000 `cudaGraphExecUpdate` at every width), so re-instantiation is ruled out. The byte budget is worth 8% on the classic arm (every 256-expert nvfp4 stack exceeds GB10's 25 "MB" cap on its own, since the counter sums element counts) but does not separate the arms.

The profile put the two terms in different places. Differencing a 400-token and a 200-token profile of the same configuration (so load, prefill and the exactness probe cancel) gives per round at block 8: 86.7 ms of GPU kernel time inside a 142.6 ms round, a 61% busy fraction, against 96% on the classic step. The per-row term is GPU work in the routed experts: `qmm_sm80` through `gather_qmm` grows from 39.4 ms at 8 rows to 59.4 at 16, 2.5 ms per row, close to the floor of reading each selected expert once per (row, expert) pair. Two premises of the issue fell here: this checkpoint's attention projections, gate, router and lm_head are plain bf16 (`gemv_single` at one row, a cutlass bf16 GEMM at two or more), only the experts are NVFP4, so there is no `fp_qmv` to `qmm_sm80` switch on the projections; and `SwitchGLU` expands its input so `gather_qmm` sees `M = 1, B = 8n`, which puts the experts on `qmm_sm80` at every width, classic included.

The fixed term is host time in one primitive. MLX's NVTX ranges, differenced the same way, show `ScaledDotProductAttention::eval_gpu` at 67.1 ms per round at block 8 and 75.8 at block 16, against 2.8 ms per classic token, over the same 45 calls (40 target layers plus 5 drafter layers). A one-row decode call takes the `sdpa_vector` kernel. A multi-row masked call with head_dim 128 and bf16 is cuDNN-eligible, and MLX caches the cuDNN execution plan in an LRU keyed on the exact query, key, value and mask shapes and strides. A verify round appends rows to the KV cache, so the key length and mask shape are new every round for each of three shape classes (target full layers, target sliding layers with sinks, drafter layers): three misses per round, each a cuDNN frontend graph build of about 22 ms. That is row-independent, it is invisible to every graph knob, and it also explains the drafter's flat 30 ms host build (its five layers pay one of the three builds).

The same LRU keeps a lifetime miss counter and throws MLX's fatal `Cache thrashing` error past twice its capacity (512). At three misses per round that is about 170 rounds per process: every block 2 profile past 200 tokens aborted on exactly that throw (400 tokens twice, 300 tokens once), while the 200-token runs and the block 8 and 16 runs at 400 tokens (163 and 161 rounds) completed. So before this issue a Laguna DFlash generation longer than about 170 rounds ended the process.

The zero-code control confirmed the mechanism end to end: `MLX_CUDA_USE_CUDNN_SDPA=0` takes 39 ms off the device sync and 21 ms off the drafter host build at every width, leaves the classic arm untouched, and puts block 4 at 1.34x and block 8 at 1.15x over classic. It is not the fix, because it also takes prefill off cuDNN's flash kernels.

## The change

Two pieces, both with an environment kill switch, nothing in the round loop or the Laguna sources.

1. A patched `mlx/backend/cuda/scaled_dot_product_attention.cpp` (the pinned file with one added gate, the same overlay mechanism as the existing `.cu` patch): `supports_sdpa_cudnn` refuses a masked call with 2 to `MLXCEL_SDPA_FALLBACK_MAX_QUERIES` (default 32) query rows over a longer key sequence. Such a call then takes MLX's own ops fallback (the same arithmetic the CPU backend and the Qwen 3.5 head_dim-256 verify already use, sinks included), which has no per-shape build. The one-row decode step is untouched (vector kernel), and so is prefill (its key length equals its query length, or it has more rows than the bound; only the last short chunk of a chunked prefill changes path). `MLXCEL_SDPA_FALLBACK_MAX_QUERIES=0` restores upstream dispatch without a rebuild and is the kill switch for the A/B.
2. `MLX_CUDA_SDPA_CACHE_SIZE` defaults to 2000 on CUDA builds through `hardware::apply_cuda_sdpa_cache_default`, the sibling of the #818 graph-cache default, applied from the same three entry points with the same env-wins contract. With the gate in place the verify no longer misses the plan cache, but prefill prompt-length diversity alone crosses the 512-miss abort on a long-lived server, and the abort is a process death rather than a request error.

## After

Same binary, same method, quiet host, n = 3 per arm; `ks-*` sets `MLXCEL_SDPA_FALLBACK_MAX_QUERIES=0`.

| config | n | tok/s mean (min to max) | vs classic | round wall ms | device sync ms/round |
|---|---|---|---|---|---|
| classic | 3 | 28.78 (28.53 to 29.00) | | 34.7 per token | |
| block 2 | 3 | 32.44 (32.06 to 33.13) | **1.13x** (was 0.52x) | 54.6 (was 115.3) | 42.0 (was 80.4) |
| block 4 | 3 | 38.40 (37.81 to 39.19) | **1.33x** (was 0.70x) | 62.8 (was 118.6) | 48.9 (was 84.9) |
| block 6 | 3 | 38.05 (37.99 to 38.10) | **1.32x** (was 0.72x) | 71.0 (was 126.6) | 56.2 (was 92.5) |
| block 8 | 3 | 32.19 (31.68 to 32.86) | **1.12x** (was 0.65x) | 81.8 (was 138.3) | 66.8 (was 103.7) |
| block 10 | 3 | 29.07 (28.42 to 29.49) | 1.01x | 89.4 | 74.6 |
| block 12 | 3 | 28.58 (27.89 to 28.95) | 0.99x | 95.9 | 80.6 |
| block 16 (checkpoint default) | 3 | 23.91 (23.73 to 24.05) | 0.83x (was 0.52x) | 108.6 (was 166.9) | 93.5 (was 131.3) |
| ks, block 2 | 3 | 15.42 (15.35 to 15.51) | 0.54x | 114.8 | 81.9 |
| ks, block 8 | 3 | 18.90 (18.69 to 19.25) | 0.66x | 139.3 | 104.9 |
| ks, block 16 | 3 | 15.67 (15.36 to 15.93) | 0.54x | 163.6 | 129.9 |

The kill switch reproduces the baseline within 2% at every width. The device sync is now `36.0 + 3.6 ms per row` (was `71.6 + 3.79`): the floor fell from 2.1x to 1.05x a classic step and the per-row slope is the expert reads, unchanged. Block 4 and 6 win with their whole ranges above the classic range; the crossover is between 10 and 12 rows and the checkpoint default of 16 remains the worst width (0.83x). After the fix the profiled round has no `ScaledDotProductAttention` host range at all (the fallback is ordinary ops), block 2 is 53.8 ms of wall for 54.7 ms of kernels (GPU-bound), and the fallback costs about 10 ms more GPU time per round than cuDNN's flash kernel against the 69 ms of host time it removes.

Qwen 3.5 on the same binary through the #1782 server harness: classic 56.07, block 2 70.56 (1.26x; #1782 had 1.24x), block 4 75.58 (1.35x; 1.32x), greedy text byte-identical to classic at both widths, 3 of 3 runs each.

## Blast radius and what is not fixed

The gate reaches every CUDA masked SDPA call with 2 to 32 query rows and prior context: the speculative verify shape on every family with head_dim 128 or less on Ampere and later (Laguna, and any MTP or DFlash pairing whose target has head_dim 128), and the trailing short chunk of a chunked prefill. Those calls now run MLX's fallback arithmetic instead of cuDNN's, so a pairing that was byte-identical to its classic chain through cuDNN could in principle flip a tie; the Laguna probe already declines this host for other reasons (`qmm_sm80` against `fp_qmv` rounding at logit ties, out of scope here) and the gate is not touched. Qwen 3.5 (head_dim 256) never entered cuDNN and is unaffected by construction; its speculative win was re-measured on the new binary (table above). Metal and CPU are unaffected. The cache-size default reaches every CUDA process and only costs memory as distinct SDPA shapes accumulate.

What remains is the per-row expert term (2.5 ms per row, inherent to MoE verify unless rows that route to the same expert share its read) and the bf16 GEMM's fixed 12 ms over the one-row gemv; the width curve after the fix peaks at block 4 and the checkpoint default of 16 is still the worst width, which is the #1797 policy question. The classic-arm finding that GB10's default graph budgets make capture a net loss on this MoE model (graphs off +12%, both budgets raised +16%, opposite sign to Qwen 3.5) is recorded for #1798 and not changed here.

## Verification

VERIFICATION_PLACEHOLDER
