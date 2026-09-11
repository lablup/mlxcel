# Where the fixed cost of a multi-row speculative verify goes on GB10, and the two fixes

Issue #1782. Host GB10 (sm_121), MLX pin `81ba1c6a`, CUDA release build. Full measurement record: `docs/benchmark_results/dflash-verify-fixed-cost-gb10-2026-09-11.md`.

## Background

PR #1771 measured the Laguna DFlash pairing on this host and found that no block width beats classic decode: the verify block's device cost fit a fixed 77 ms plus 3.3 ms per row, 2.7x a single-token step even at 2 rows, and the PR attributed that to the multi-row forward running "eagerly and launch-bound" on the CUDA backend. The issue was filed with that framing already checked: nothing in the pinned MLX's capture path branches on row count, so the fixed term had to be attributed before anything was changed. It named three graph-side candidates (GB10's 20-op / 25 MB graph budget, `subgraph_to_key` clearing `is_updatable`, key churn) and asked for the two zero-code controls to be tried first.

The Qwen 3.5 pairing (`qwen3.5-4b-4bit` with `qwen3.5-4b-dflash`) was used because it runs on `main` through `mlxcel-server`; the Laguna arm needs #1771.

## What the measurement said

Same binary every arm, one server per configuration, greedy streaming completions of 200 tokens on a fixed raw-code prompt, n = 3 after a discarded warm-up, GPU held under the scratchpad lock, idle host. Classic decode: 58.42 tok/s (17.1 ms per step). Block 2: 25.82 tok/s, with 42.6 ms of device work per verify round, 2.5x a classic step, the same shape as Laguna's 2.7x.

The graph controls were the cheapest experiment and they came back negative at 2 rows. `MLX_MAX_MB_PER_BUFFER=400` (GB10's 25 MB raised to the H100 value) left block 2 slightly slower and gave block 8 about 5%; `MLX_MAX_OPS_PER_BUFFER=100` left block 2 unchanged and gave block 8 about 10%; and `MLX_USE_CUDA_GRAPHS=0` made block 8 6% faster while costing the classic arm 8%. A path that gains nothing from capture is not a capture problem. Graph replay was healthy on every arm (block 2: 260 `cudaGraphInstantiate` against 71089 `cudaGraphExecUpdate`), which matches #1545's Volta finding.

The nsys kernel tables per verify round (`--cuda-graph-trace=node`, proportions and launch counts only, since profiling inflated the runs by 15 to 46%) put the fixed term in two places, neither of them the graph machinery:

| category | block 2, per round | block 8 | block 16 |
|---|---|---|---|
| drafter in f32: `copy_v<__half, float>` weight upcasts plus f32 cutlass GEMMs | 38.0 ms, 243 launches | 31.2 ms, 239 | 30.0 ms, 228 |
| GDN chunked scan: 5 us batched matmuls and f32 subtracts of the 63-step Horner loop | 21.9 ms, 4364 | 40.3 ms, 7170 | 38.9 ms, 7266 |
| target quantized projections | 15.7 ms (`qmv_multirow`) | 43.5 ms (`qmm_sm80`) | 43.7 ms |
| target attention with the materialized mask | 0.3 ms | 1.1 ms | 2.0 ms |

1. The drafter ran in float32. `DFlashDrafter::load` converted every bf16 tensor to f16 unconditionally (the Apple Silicon rule), while the target loaders keep bf16 on Ampere and later CUDA. MLX promotes a bf16 activation times an f16 weight to float32, and the residual stream the drafter reads is the target's bf16 hidden state, so the first matmul promoted and everything downstream stayed f32: 72 upcasts of the drafter's 540M weights per round (the grid sizes identify them: 32.8M is `fc.weight`, 24.9M the MLP matrices, 10.5M `o_proj`) followed by f32 GEMMs at 0.75 to 1.2 ms each. Row-count independent, which is what a fixed term looks like.
2. The 24 gated-delta linear-attention layers took the 64-row chunked scan for every verify block. `gated_delta_chunked` inverts `(I + T)` by a 63-step finite Neumann series per layer whatever `T` is, so 2 rows and 16 rows cost the same 1512 tiny matmuls and 1512 subtracts. At block 8 and 16 the count per round is 1.65x that, because a partial accept replays the accepted rows through the same scan. On CUDA the chain-parity flag the verify pass requested was ignored by the ops fallback, so the block was also not bit-identical to the single-token chain; block 16 diverged from classic greedy text before the change.
3. The per-row term is the quantized projection kernel switch, not attention. Rows 2 to 7 dispatch `qmv_multirow_kernel` (a 2-row verify pays 15% more than a classic step for its weights); from `M = 8` the dispatcher switches to `qmm_sm80_kernel`, 3.5x the single-row cost and flat from 8 to 16 rows. The materialized-mask attention on the eight head_dim-256 layers is under 2 ms per round.

## The change

Drafters loaded through `mlxcel-core` (DFlash, Muse Glimmer, Inkling MTP, Qwen 3.5 MTP) now apply `apply_drafter_load_dtype_policy`, which mirrors the target loaders' `bf16_to_f16_at_load` for an unquantized checkpoint: f16 on Apple Silicon and pre-Ampere CUDA, bf16 on Ampere and later CUDA, `MLXCEL_KEEP_BF16` and `MLXCEL_CUDA_F16_NORMALIZE` honored the same way. The pure policy function is unit-tested against every arm.

The gated-delta ops fallback honors chain parity for an unmasked scalar-gated block of at most `CHAIN_PARITY_SEQUENTIAL_MAX_T = 32` rows by running the sequential loop of fused single steps. That loop is `gated_delta_step` applied `T` times, the classic chain's own arithmetic, so the block is bit-identical to `T` single-token steps; a test pins that on the ops path against the chunked scan, which is only close. Wider blocks and prefill keep the scan. Both changes keep a same-binary kill switch (`MLXCEL_CUDA_F16_NORMALIZE=1`, `MLXCEL_GDN_CHAIN_PARITY=0`).

## After

| Configuration | n | tok/s (min to max) | vs classic | round device sync | greedy text == classic |
|---|---|---|---|---|---|
| classic | 3 | 58.33 (57.27 to 59.38) | | | yes |
| block 2 | 3 | 72.57 (72.21 to 72.99) | 1.24x (was 0.44x) | 18.6 ms (was 42.6) | yes |
| block 3 | 3 | 76.42 (76.36 to 76.51) | 1.31x | 24.4 ms | yes |
| block 4 | 3 | 76.88 (76.62 to 77.28) | 1.32x (was 0.53x) | 29.9 ms (was 58.5) | yes |
| block 6 | 3 | 59.06 (58.84 to 59.19) | 1.01x | 47.4 ms | yes |
| block 7 | 3 | 53.45 (53.32 to 53.59) | 0.92x | 53.9 ms | yes |
| block 8 | 3 | 48.84 (48.48 to 49.32) | 0.84x (was 0.49x) | 59.2 ms (was 81.6) | yes |
| block 16 | 3 | 48.63 (48.48 to 48.83) | 0.83x (was 0.53x) | 64.1 ms (was 78.7) | yes (was NO) |
| both kill switches, block 2 | 3 | 25.98 (25.90 to 26.12) | 0.45x | 42.2 ms | yes |
| drafter fix only, block 2 | 3 | 36.85 (35.97 to 38.13) | 0.63x | 40.9 ms | yes |
| GDN fix only, block 2 | 3 | 35.87 (35.76 to 36.00) | 0.61x | 25.9 ms | yes |

The 2-row verify is now 1.09x a classic step. Each fix alone lands near 36 tok/s and both together at 72.6, because a round is a serial chain (host draft build, host verify build, device sync) and each fix removes a different link. Greedy text is byte-identical to classic on every arm, including block 16, which the chain-parity loop brought back into agreement.

## What is not fixed, and what is recommended

The width curve has a cliff between 4 and 6 rows (1.32x at 4, 1.01x at 6, 0.92x at 7): the multirow `qmv` kernel dispatches accumulator widths of 2, 4 or 8, so 5 to 7 rows take the 8-wide instantiation at its higher register cost. Block 8 and 16 do not win. Their remaining cost is the `qmm_sm80` term at `M >= 8`, untouched here, and the checkpoint's own default width of 16 sits on that side of the switch. The measured setting for this pairing is `--draft-block-size 4`. Choosing a hardware-gated default width is a policy question with a family-dependent crossover (`fp_qmv` on NVFP4 checkpoints has the same 8-row bound) and is proposed as a follow-up, as is `MLX_MAX_OPS_PER_BUFFER=100`, which gave +3% on the classic arm of one model and belongs to every CUDA decode path rather than to this issue.

The Laguna pairing was not re-measured (it needs #1771). Its drafter took the same f16 conversion, so the first fix applies to it; it has no gated-delta layers, so the second does not.

## Blast radius

The dtype policy reaches every drafter family loaded through `mlxcel-core`; the gated-delta change reaches the chain-parity call sites (Qwen 3.5 verify and rollback replay) on every non-Metal backend, and no other caller of `gated_delta_ops` since only the parity flag selects the new branch. Nothing in the MLX pin or the C++ bridge changed. Only the Qwen 3.5 DFlash pairing was measured; the other families share the code path but not yet a number.

## Verification

`cargo fmt --all -- --check` clean. Unit tests: the policy and env-flag parsing tests in `drafter.rs`, the bit-exactness test in `gated_delta_tests.rs`, and the existing drafter and gated-delta suites, run with narrow selectors under the test-fast profile on this host (an unscoped `cargo test --lib` aborts on this host with a `cudaStreamEndCapture` C++ abort unrelated to this change, and `make verify-test-cuda` runs the whole workspace and could not be run under the watchdog). The `metal,accelerate` gate is not runnable on this Linux/CUDA host and is unrun.
