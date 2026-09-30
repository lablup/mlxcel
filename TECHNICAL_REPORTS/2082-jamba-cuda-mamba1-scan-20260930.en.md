# Technical Report: PR #2082 - Fuse the Jamba Mamba scan on CUDA, bit-exact to the graph

**Date**: 2026-09-30

**Status**: Implemented and validated on GB10; pending merge.

**Languages**: C++ (mlxcel-core bridge, CUDA JIT source), Rust, Markdown

**Risk Level**: Low to medium. Jamba on CUDA switches from the per-step graph scan to a custom kernel. The kernel is bit-identical to the graph scan for uniform-dtype inputs and falls back to the graph scan otherwise. Metal behavior is unchanged; Mamba and Falcon-Mamba on CUDA are unchanged.

## Executive Summary

Issue #1981 reported that a ~3.3k-token Jamba chat request took about 300 s on GB10 through `mlxcel-server`. The binary that produced that number predated PR #2000, whose Jamba prefill scan concatenated every per-step state onto a growing tensor, which is quadratic in the prompt length. Reverting only that hunk on GB10 gives 8.1 s of prefill at 548 tokens and 26.6 s at 1059 tokens, which extrapolates to 260 to 300 s at 3299 tokens. At 3299 tokens the server run exhausted the host's 121 GB and was OOM-killed.

Main already ran the request in 4.3 s. About 2.3 s of its 4.1 s prefill was the remaining per-timestep graph loop: roughly ten small MLX ops per timestep in each of 26 Mamba layers, because the fused Mamba1 scan kernel from #2005 had only a Metal port. This PR adds the CUDA port. On the same request the server time drops from 4.33 s to 1.51 s (turn 2: 4.17 s to 1.53 s), and greedy output is unchanged.

## 1. Attribution

| Code | Scan implementation | GB10 cost |
|---|---|---|
| Before #2000 (the issue's binary) | per-step loop plus one `concatenate` of a `[B, 1, 5120, 16]` state per step | 8.1 s prefill at 548 tokens, 26.6 s at 1059 (n=3 each); OOM at 3299 |
| main `929c80ab` | per-step graph loop, `y` rows stacked once | 4.3 s per request; about 2.3 s of prefill is the scan |
| this PR | one CUDA kernel per layer | 1.5 s per request |

The scan share on main was measured with a throwaway build that replaced the scan with a shape-correct trivial op consuming `delta`, `B` and `C`: prefill took 1.33 to 1.35 s against 3.59 to 3.74 s for main in the same session (n=3 each). The other candidates in the issue do not apply to this checkpoint. It has `num_experts` 1, so MoE routing is a single dense expert, and the server's 512-token chunked prefill already applied.

The checkpoint named `jamba-v0.1-4bit` is a reduced variant: 28 layers (26 Mamba and 2 attention, at layers 7 and 21), hidden 2560, Mamba intermediate 5120, d_state 16, 1.7 GB on disk.

## 2. Change Summary

- **`mlx_cxx_kernels.cpp`**: a `mx.fast.cuda_kernel` port of `mamba1_selective_scan`. Because the two variants round differently, each has its own `KernelPorts` table (`mamba1_scan_ports` for the float32-state Metal kernel, `mamba1_scan_graph_exact_ports` for the CUDA one), and the launcher picks the variant by which table has a port for the running backend, never by comparing backend kinds (the rule `scripts/ci/check_kernel_port_dispatch.py` enforces). `mamba1_scan_kernel_available()` now reads both tables (still honoring `MLXCEL_MAMBA1_SCAN_KERNEL=0`). A new `mamba1_scan_kernel_accepts(x, delta, b, c, a, d)` says whether the kernel may serve a call: it is available, the default device is the GPU, `N <= 32`, and on CUDA all six inputs share one floating dtype. On CUDA the state is passed and returned in the activation dtype; on Metal it stays float32.
- **`jamba.rs`**: `JambaMambaMixer::ssm_step` calls the kernel when `mamba1_scan_kernel_accepts` is true, for prefill and decode. A test-only thread-local counts timesteps walked by the graph loop.
- **`mamba.rs`**: the fused path stays Metal-only. Mamba's graph path runs `x_proj` one timestep at a time, while the kernel path projects the whole sequence, so switching on CUDA would change output.
- **Tests**: `cuda_kernel_is_bit_identical_to_the_graph_scan`, `cuda_kernel_declines_mixed_dtype_inputs`, and `jamba_mamba_prefill_takes_the_fused_scan_where_a_port_exists`. The Metal-specific accuracy test now runs on Metal only.
- **Docs**: `docs/environment-variables.md` (`MLXCEL_MAMBA1_SCAN_KERNEL`) and `docs/benchmark_results/jamba-mamba1-scan-kernel-gb10-2026-09-30.md`.

## 3. Technical Decisions

### Reproduce the graph's rounding instead of the Metal kernel's float32 state

The issue requires greedy output to match the current implementation. The Metal kernel carries the state in float32, which changed greedy output against the graph scan on long prompts (#2005's record). The CUDA port instead performs exactly the graph's operations: `new_state = (delta * x) * B`, `dtA = exp(delta * A)`, `state = state * dtA + new_state`, each rounded to the activation dtype, then `y = state @ C` and `y + D * x`. Three details make it bit-identical:

1. The exponential is MLX's own `Exp` device functor from `unary_ops.cuh`, the same code the graph's `exp` op runs.
2. `y_t` reproduces the graph's matmul. For `[B, D, N] @ [B, N, 1]` MLX dispatches to `gemv`, which with `K = N < 64` uses one element per lane, float products, and `cg::reduce` over the 32-lane warp before rounding to T. The kernel does the same, with lanes past N contributing 0.
3. Multiplies and adds use `__fmul_rn`/`__fadd_rn` and `__hmul_rn`/`__hadd_rn`. The graph runs each op as its own kernel, so a product is rounded before it is added. NVRTC compiles with `--fmad=true` by default, so an inlined `a * b + c` becomes one fma with a single rounding. The first draft of the kernel used plain operators and differed from the graph by one ulp in about half of the outputs, f32 included; the `_rn` forms removed every difference.

### Gate on dtype uniformity

The kernel is exact only when every scan input shares the activation dtype, because the graph promotes mixed dtypes op by op (for example a float32 `A_log` would make the state float32 after the first step). `mamba1_scan_kernel_accepts` checks this on CUDA and returns false otherwise, so such a model keeps the graph scan and its current output. Published Jamba conversions are all bf16 and take the kernel.

### Keep the graph scan when the default device is the CPU

Custom kernels launch only on the GPU stream. Under `MLXCEL_DEVICE=cpu` the kernel call would throw where the graph scan used to run on the CPU, so `accepts` also requires a GPU default device. A CPU-device Jamba generation on GB10 runs and matches main.

## 4. Validation

Host: GB10, kernel 7.0.0-1019-nvidia, driver 580.178.04, CUDA 13.0, release build with `--features cuda`.

Server (default launch, 3299-token chat, `max_tokens` 64, temperature 0, three prompts per server run, runs in ABBA order, n=6 per cell):

| | main | this PR |
|---|---|---|
| Turn 1 | 4.33 s (4.23 to 4.49) | 1.51 s (1.51 to 1.74) |
| Turn 2 | 4.17 s (4.08 to 4.28) | 1.53 s (1.52 to 1.56) |

CLI (`mlxcel generate --profile`): prefill 3.59 to 4.16 s on main (n=8, two sessions) against 2.14 to 2.21 s (n=3); decode 69 to 77 tok/s against 76 to 79 tok/s. The CLI prefill includes the kernel's one-time NVRTC compile, which the server pays during startup warmup (1.1 s to 1.8 s).

Output identity:

- Kernel against graph scan: bit-identical output rows and final state for bf16, f16 and f32, d_state 8 and 16, 1, 7 and 33 steps, fresh and carried state.
- Server: both PR runs match the first main run on all 12 requests (reasoning and content text). A second main run differed from the first main run on one request (turn 2 of one prompt), so main is not fully deterministic run to run on this host; the PR runs matched the first main run on that request as well.
- CLI greedy at 548, 1059 and 3299 tokens: identical to main.

Regression guard: `jamba_mamba_prefill_takes_the_fused_scan_where_a_port_exists` passes with the kernel and fails under `MLXCEL_MAMBA1_SCAN_KERNEL=0` ("prefill and decode walked 49 graph-scan timesteps"), which is the code path main takes on CUDA.

`models::jamba` (11 pass, 1 ignored), `models::mamba` and the `mamba1` parity tests pass on CUDA with `--test-threads=1`. fmt and clippy (`-D warnings`, lib and tests, main crate and mlxcel-core) are clean.

## 5. Not Addressed

- **Prompt-cache reuse for Jamba**: the server logs `cached=0` on every turn, on main and with this PR, so each turn re-prefills the full history. That is why turn 2 was slower than turn 1 in the issue.
- **NVRTC compile per process**: MLX does not disk-cache custom CUDA kernels, so each process compiles the kernel once (about 0.7 s).
- **Mamba / Falcon-Mamba on CUDA**: still the graph scan (see section 2).
- **Metal**: logic is unchanged (`gpu_kernel_backend()` is Metal exactly when `metal::is_available()`, the old check), but it was not run here. ROCm has no port.
