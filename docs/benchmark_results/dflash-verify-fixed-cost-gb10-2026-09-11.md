# DFlash verify block: where the fixed per-round device cost goes (GB10, 2026-09-11)

Issue #1782. Host: NVIDIA GB10 (sm_121, DGX Spark), Linux aarch64, MLX pin `81ba1c6a`, CUDA build (`MLX_CUDA_ARCHITECTURES=121`), release profile. Otherwise idle host (load 0.1 to 1.5 from the run's own client, no other model, build or CI job during any measured run; the self-hosted CI runner was up but idle). GPU held under the scratchpad lock for every run. Warm PTX cache (`~/.cache/mlxcel/cuda-ptx/81ba1c6a...`, rule 3 of the Volta record) and warm page cache (both checkpoints had been read by the smoke run before any measured run, rule 7). `MLX_ENABLE_TF32` was left at MLX's default; no numerical comparison in this record depends on it (byte-identity below compares token strings, not floats).

Pairing: `models/mlx/qwen3.5-4b-4bit` (target, affine 4-bit, group 64, 32 layers of which 8 full-attention with head_dim 256 and 24 gated-delta linear attention) with `models/mlx/qwen3.5-4b-dflash` (5-layer bf16 DFlash drafter, `block_size 16`, `target_layer_ids [1, 8, 15, 22, 29]`). This is the pairing that runs on `main` today; the Laguna pairing measured in PR #1771 needs that PR and was not re-run here.

## Method

`mlxcel-server -m <target> --ignore-eos --max-batch-size 1 [--draft-model <drafter> --draft-kind dflash --draft-block-size N]`, one server per configuration, `RUST_LOG=info`. Client: streaming `POST /v1/completions`, `temperature 0`, `max_tokens 200`, a fixed 165-token raw Python source prompt (a `retry_with_backoff` module header), one warm-up request discarded, then n = 3 measured requests. The reported rate is end-to-end wall time for the fixed 200-token budget from request start to the last SSE chunk (the DFlash burst delivers its chunks at the end, so an inter-chunk rate is meaningless on that arm; prefill is about 100 ms of a 3.4 to 8 s request and identical on both arms). The speculative arm's `DFlash diagnostics` log line supplies the per-round split: `draft_ms` (host graph build of the drafter), `verify_ms` (host graph build of the target verify) and `target_argmax_sync_ms` (the wait for all of the round's device work).

Same binary on every arm. Controls are environment variables read by MLX or mlxcel, never a code path difference.

## Before: block-size sweep and the zero-code controls

| Configuration | n | e2e tok/s mean (min to max) | vs off | server decode ms / 200 tok | emitted per verify | round device sync ms (min to max) | round draft host ms | round verify host ms | greedy text == classic |
|---|---|---|---|---|---|---|---|---|---|
| off (classic) | 3 | 58.42 (58.14 to 58.62) | | | | | | | yes |
| block 2 | 3 | 25.82 (25.76 to 25.88) | 0.44x | 7756 | 1.81 | 42.6 (42.6 to 42.7) | 21.5 | 6.1 | yes |
| block 4 | 3 | 31.17 (30.93 to 31.38) | 0.53x | 6458 | 2.80 | 58.5 (57.3 to 60.3) | 23.2 | 7.7 | yes |
| block 8 | 3 | 28.55 (28.47 to 28.64) | 0.49x | 7075 | 3.26 | 81.6 (80.8 to 82.7) | 23.7 | 8.2 | yes |
| block 16 (checkpoint default) | 3 | 31.04 (30.57 to 31.45) | 0.53x | 6436 | 3.49 | 78.7 (78.1 to 79.4) | 23.4 | 8.4 | NO |
| `MLX_MAX_MB_PER_BUFFER=400`, off | 3 | 58.33 (58.12 to 58.56) | 1.00x | | | | | | yes |
| `MLX_MAX_MB_PER_BUFFER=400`, block 2 | 3 | 25.08 (24.97 to 25.24) | 0.43x | 7961 | 1.81 | 60.6 (59.1 to 61.8) | 5.9 | 5.6 | yes |
| `MLX_MAX_MB_PER_BUFFER=400`, block 8 | 3 | 29.95 (29.49 to 30.77) | 0.51x | 6802 | 3.26 | 90.9 (88.1 to 92.6) | 12.1 | 6.3 | yes |
| `MLX_MAX_OPS_PER_BUFFER=100`, off | 3 | 60.20 (59.78 to 60.80) | 1.03x | | | | | | yes |
| `MLX_MAX_OPS_PER_BUFFER=100`, block 2 | 3 | 26.03 (25.99 to 26.10) | 0.45x | 7723 | 1.81 | 40.7 (40.4 to 41.0) | 21.9 | 7.4 | yes |
| `MLX_MAX_OPS_PER_BUFFER=100`, block 8 | 3 | 31.46 (31.41 to 31.50) | 0.54x | 6432 | 3.26 | 71.9 (71.3 to 72.8) | 22.9 | 8.3 | yes |
| `MLX_USE_CUDA_GRAPHS=0`, off | 3 | 53.54 (53.28 to 53.72) | 0.92x | | | | | | yes |
| `MLX_USE_CUDA_GRAPHS=0`, block 8 | 3 | 30.39 (30.36 to 30.42) | 0.52x | 6683 | 3.26 | 89.2 (89.1 to 89.4) | 8.4 | 9.8 | yes |

The classic step is 17.1 ms. The 2-row verify round's device work is 42.6 ms, 2.5x a classic step, the same shape PR #1771 reported for Laguna (2.7x). Within-configuration spread is under 2% everywhere, so none of this is the GB10 single-run bimodality of #755.

What the controls say:

- The byte budget (`MLX_MAX_MB_PER_BUFFER`, GB10's 25 MB raised to 400) does nothing at 2 rows (7961 vs 7756 ms, slightly slower) and is worth about 5% at block 8. It moves time between `draft_ms` and the device sync (the async_eval enqueue inside the draft phase commits fewer, larger graphs) without changing the sum.
- The op budget (`MLX_MAX_OPS_PER_BUFFER`, 20 raised to 100) does nothing at 2 rows and is worth about 10% at block 8 (sync 71.9 vs 81.6 ms). It is also worth about 3% on the classic arm (60.20 vs 58.42, ranges disjoint), which is unrelated to speculative decoding and is recorded as a separate finding below.
- With graphs off entirely, the classic arm loses 8% (capture is real and healthy there: 176 `cudaGraphInstantiate` against 24537 `cudaGraphExecUpdate` over the profiled run) while the block 8 arm is 6% *faster* than with graphs on. The multi-row verify is not gaining anything from capture, so it is not a capture problem, and the best knob still leaves block 8 at 0.54x.

So candidates 1 to 3 of the issue (byte budget, `is_updatable`, key churn) are secondary contributors at 8 rows and absent at 2 rows. The fixed term is elsewhere.

## Attribution: kernel tables per verify round

`nsys profile -t cuda,nvtx --cuda-graph-trace=node` (rule 4), one warm-up plus one measured request per arm. Profiling inflates the runs (rule 5): classic 50.84 vs 58.42 tok/s unprofiled (115%), block 2 decode 11312 vs 7756 ms (146%), block 8 9074 vs 7075 ms (128%), block 16 9423 vs 6436 ms (146%); tiny kernels inflate most, so only proportions and launch counts are compared across arms (rule 6). Kernel time is summed by category over the trace and divided by the rounds in it (220, 122, 114) or, for classic, by generated tokens (400).

| category (kernel names) | classic, per token | block 2, per round | block 8, per round | block 16, per round |
|---|---|---|---|---|
| target quantized projections (`qmv_kernel` / `qmv_multirow_kernel` / `qmm_sm80_kernel`) | 12.4 ms, 229 launches | 15.7 ms, 251 | 43.5 ms, 209 | 43.7 ms, 207 |
| drafter in f32: `copy_v<__half, float>` weight upcasts, `s1688gemm` f32 cutlass GEMMs, `rms_norm_small<float>`, `rope<float>`, `sdpav_1pass<float>` | 0 | 38.0 ms, 243 | 31.2 ms, 239 | 30.0 ms, 228 |
| GDN chunked scan: 5 us `s1688gemm_64x64_nn` batched matmuls, f32 `Subtract` / `Multiply`, chunk copies and selects | 0.3 ms, 67 | 21.9 ms, 4364 | 40.3 ms, 7170 | 38.9 ms, 7266 |
| target full-attention with the materialized mask | 0.1 ms | 0.3 ms | 1.1 ms | 2.0 ms |
| other bf16 elementwise, norms, copies | 2.7 ms, 902 | 5.6 ms, 1437 | 20.2 ms, 1988 | 20.8 ms, 2138 |

Three findings, in order of size:

1. **The drafter runs in float32 and re-upcasts all of its weights every round.** The 72 `copy_v<__half, float>` launches per round have grids of 4000, 3040, 1280 and 320 blocks of 1024 threads by 8 elements: 32.8M (`fc.weight`, [2560, 12800]), 24.9M (the three 9728 x 2560 MLP matrices), 10.5M (`o_proj`) and 2.6M (`k_proj`, `v_proj`) elements, the drafter's whole 540M-element weight set, once per round, followed by f32 `cutlass_80_tensorop_s1688gemm` GEMMs at 0.75 to 1.2 ms each. The cause is a dtype mismatch: `DFlashDrafter::load` converted every bf16 tensor to f16 unconditionally (`convert_bf16_to_f16_non_quantized`, the Apple Silicon rule), while the target loaders keep bf16 on Ampere and later CUDA (`bf16_to_f16_at_load`). MLX promotes a bf16 activation times an f16 weight to float32, and everything downstream of the first promotion (the residual stream the drafter reads is the target's bf16 hidden state) stays f32. This term does not depend on the row count. It is the fixed term.
2. **The 24 linear-attention layers take the 64-row chunked scan for every verify block, however short.** `gated_delta_chunked` runs a 63-step Horner loop per layer (`(I + T)^-1` by finite Neumann series) whatever `T` is: 1512 tiny batched matmuls and 1512 f32 subtracts per verify at 2 rows and at 16 rows alike. At block 8 and 16 the launch count per round is 1.65x that of block 2 because a partial accept replays the accepted rows through the same scan in `rollback_speculative_cache` (44 of 61 rounds at block 8). On CUDA the chain-parity flag that the verify pass requests (`gated_delta_update_chain_parity`) was ignored by the ops fallback, so besides the cost the block was not bit-identical to the single-token chain; block 16's divergent greedy text above is consistent with that.
3. **The per-row term is the quantized projection kernel switch, not attention.** Rows 2 to 7 dispatch `qmv_multirow_kernel` (61.9 us per launch at 2 rows against 53.7 us single-row, so a 2-row verify pays only 15% more than a classic step for its weights); from `M = 8` the dispatcher (`quantized.cpp`, `M * B < 8`) switches to `qmm_sm80_kernel`, which costs 3.5x the single-row `qmv` and is flat from 8 to 16 rows. That is why block 16 is no more expensive per round than block 8. The materialized-mask attention on the eight head_dim-256 layers (neither cudnn nor `sdpa_vector` applies with an array mask) is under 2 ms per round at 16 rows and is ruled out.

Graph replay itself is healthy on every arm (block 2: 260 `cudaGraphInstantiate` against 71089 `cudaGraphExecUpdate`), which agrees with #1545's Volta finding that the graph machinery is not where the time goes.

## Fix

Both fixes ship in the same PR, each with a same-binary kill switch used for the A/B below:

- Drafters loaded through `mlxcel-core` (DFlash, Muse Glimmer, Inkling MTP, Qwen 3.5 MTP) now follow the target loaders' dtype rule (`drafter_bf16_to_f16_at_load`): f16 on Apple Silicon and pre-Ampere CUDA, bf16 on Ampere and later CUDA, with `MLXCEL_KEEP_BF16` and `MLXCEL_CUDA_F16_NORMALIZE` honored the same way. `MLXCEL_CUDA_F16_NORMALIZE=1` restores the old f16 drafter on this host.
- The gated-delta ops fallback honors chain parity for an unmasked scalar-gated block of at most 32 rows (`CHAIN_PARITY_SEQUENTIAL_MAX_T`) by running the sequential loop of fused single steps, which is the classic chain's arithmetic and launches `T` fused steps per layer instead of the fixed 63-step scan. Wider blocks and prefill keep the chunked scan. `MLXCEL_GDN_CHAIN_PARITY=0` restores the scan.

## After

AFTER_TABLE_PLACEHOLDER

## Separate finding: `MLX_MAX_OPS_PER_BUFFER=100` on the classic path

The op budget raised from GB10's 20 to 100 gave +3% on classic decode (60.20 vs 58.42 tok/s, n = 3, ranges disjoint) and +10% at block 8, on one model (`qwen3.5-4b-4bit`). That is a property of every CUDA decode path on this host, not of speculative decoding, and mlxcel currently leaves CUDA on MLX's per-architecture values (`apply_metal_ops_per_buffer_default` is Metal-only by design). Not changed here. Caveats for whoever picks it up: a larger op budget means larger captured graphs, so instantiation cost and the `MLX_CUDA_GRAPH_CACHE_SIZE` lifetime-miss budget (#818) both move; it was measured on one model, one prompt, at batch 1; and #1545 found the same knob immaterial on Volta.
