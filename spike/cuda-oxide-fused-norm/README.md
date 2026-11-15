# cuda-oxide standalone fused residual-add + RMSNorm (issue #2832)

This package is the second gated experiment from [the cuda-oxide feasibility review](../../docs/mlxcelverse/cuda-oxide-feasibility.md): one reusable inference kernel, written in Rust and compiled with the pinned cuda-oxide toolchain, with a numerical oracle against the implementations mlxcel already ships. It depends on the artifact and ABI contract that [#2831](../cuda-oxide-smoke/README.md) validated and reuses that package's toolchain discovery and container parsers.

Scope is deliberately the kernel and its evidence: there is no MLX adapter (#2833), no model execution and no end-to-end comparison (#2834). The evidence from the verified GB10 run is in [docs/mlxcelverse/cuda-oxide-fused-norm-evidence.md](../../docs/mlxcelverse/cuda-oxide-fused-norm-evidence.md).

## What the kernel computes

The reference semantics are `kernels/fused_norm.cpp` (`FUSED_ADD_RMS_NORM_CUDA_SOURCE`, the CUDA port of the fused kernel every pre-norm residual join uses since #905) and `src/lib/mlxcel-core/src/fused_norm_parity_tests.rs`. For one row of `dim` elements:

```
new_residual = round_act(x + residual)                      # f32 add, rounded to the activation dtype
mean         = sum_f32(new_residual^2) / dim                # f32 FMA accumulation, two-stage warp butterfly
normed       = round_act(gain * round_act(new_residual * rsqrt(mean + eps)))
gain         = weight                     when weight_bias == 0
             = round_w(weight_bias + weight)  otherwise    # the Gemma (1 + w) fold, in the weight dtype
```

The rounding points, the accumulation precision and the reduction order are the reference kernel's, so the device output is expected to match the existing fused CUDA kernel bit for bit. That identity is measured and reported; the pass criterion is the parity budget of `fused_norm_parity_tests`, frozen before the oracle ran (see the evidence report).

Five entry points cover the dtype matrix: `fused_add_rms_norm_{f32,f16,bf16}` (weight in the activation dtype) and `fused_add_rms_norm_{f16,bf16}_wf32` (16-bit activations, f32 weight). 16-bit lanes travel as `u16` bit patterns; the kernel converts on load and store (`f16` through the compiler's casts, bf16 through `cuda_device::convert` with round-to-nearest-even).

## Build isolation

`Cargo.toml` contains an empty `[workspace]` table, so this package is its own workspace root and never joins the mlxcel build graph. `scripts/check-clean-build.sh` proves that an ordinary mlxcel build needs neither LLVM 21 nor the cuda-oxide compiler. The application Rust pin stays at 1.97.1; `rust-toolchain.toml` here selects cuda-oxide's 1.99.1 only for commands run in this directory. `Cargo.lock` is committed because the device artifact's reproducibility depends on the transitive resolution (`git add -f` after regenerating it).

## Layout

| Path | Purpose |
| --- | --- |
| `src/main.rs` | The five kernels, the shared generic row routine, and the host harness (`run`, `compare`, `bench`) |
| `abi/*.schema.json` | Hand-written ABI claim per entry: argument order and widths, launch rule, semantics, rejected shapes |
| `artifacts/` | Committed PTX and cubin (one module, five entries) and the generated, validated manifest per entry |
| `results/` | Machine-readable comparison and benchmark results from the verified run |
| `scripts/env.sh` | Delegates to the smoke package's toolchain discovery |
| `scripts/build.sh` | Pinned build, `.oxart` extraction, cubin assembly, manifest generation and validation |
| `scripts/run.sh` | Case matrix on device 0 under `gpu-lock`, host-model check, fixture dump, launch check |
| `scripts/oracle.sh` | Replays the fixtures through mlxcel's fused CUDA kernel and graph path (`mlxcel-core` example) |
| `scripts/compare.sh` | Scores the standalone outputs against both references under the frozen budgets |
| `scripts/bench.sh` | Cold/warm timings with event boundaries under `gpu-lock` |
| `scripts/check-lint.sh` | `cargo fmt --check`, clippy with `-D warnings`, and the tooling guards |
| `scripts/check-guards.py` | Proves malformed logs and manifests fail cleanly |
| `scripts/check-clean-build.sh` | Proves ordinary mlxcel builds need no oxide compiler |
| `scripts/oxide_artifact.py` | Multi-entry export, validation and launch check; imports the #2831 container/cubin parsers |

The oracle lives on the mlxcel side as `src/lib/mlxcel-core/examples/oxide_fused_norm_oracle.rs`: it is the only step that links MLX, and it runs with the application toolchain.

## Prerequisites

Exactly the [smoke package's](../cuda-oxide-smoke/README.md#prerequisites): CUDA 13.0 (`nvcc`, `ptxas`, `cuobjdump`, `compute-sanitizer`), LLVM/Clang 21 with NVPTX, Rust 1.99.1 with `rust-src`, `rustc-dev` and `llvm-tools`, and `cargo-oxide` built from NVIDIA/cuda-rust `b631530700a8a4a5f165456f2d71fe9195b323b5`. The oracle additionally needs an mlxcel CUDA build (`--features cuda`) with the 1.97.1 application toolchain.

## Running

```bash
cd spike/cuda-oxide-fused-norm
scripts/build.sh sm_121      # build, export artifacts, validate the five manifests
scripts/run.sh sm_121        # 130 cases on device 0 under gpu-lock, host-model check, fixtures
scripts/oracle.sh            # mlxcel fused CUDA kernel + graph path on the same fixtures
scripts/compare.sh sm_121    # frozen-budget comparison -> results/compare-sm_121.json
scripts/bench.sh sm_121      # cold/warm timings        -> results/bench-sm_121.json
scripts/check-lint.sh
scripts/check-clean-build.sh
```

Every GPU-allocating command goes through `gpu-lock run --tag issue-2832* -- ...`; a missing or busy lock is an error, never a reason to run unguarded. `run.sh` then checks every reported launch against the manifests: `grid = rows`, `block = fused_norm_threads(dim)` (the reference launcher's rule: `clamp(ceil_div(ceil_div(dim, 4), 32) * 32, 32, 1024)`), the per-entry case and element totals, and that the harness also exercised the shape rejections. A shortened or re-routed run fails the script rather than passing on exit status.

## Case matrix and budgets

`run` executes 26 cases per dtype pair (130 in all, 2,181,390 elements):

- widths 1, 3, 31, 32, 33, 100, 127, 128, 129, 1000, 1024, 2048, 3072, 4096, 4097, 8191, 8192 with one and many rows (up to 33 rows at 2048 and 16 rows at 8192), so partial warps, partial four-wide lane chunks, and rows wider than one 1024-lane sweep are all launched;
- the Gemma `weight_bias = 1` fold and `eps = 1e-5` on four and two of those shapes;
- one case per pair with a NaN and a +Inf injected into the activations.

Inputs are deterministic (per-case LCG seed, Box-Muller normals, weights near 1) and already rounded to the case dtypes, so the oracle casts them exactly. Shapes the contract refuses (`rows = 0`, `dim = 0`, length mismatches) are checked against the same validator the launcher runs and reported as `REJECT` lines.

Tolerances were frozen before any oracle run and are the budgets of `fused_norm_parity_tests::tolerance_for`, applied to `normed` against both references as normalized RMS / normalized max deviation (relative to the reference's RMS): f32 `1e-6 / 1e-5`, f16 `2e-3 / 1.2e-2`, bf16 `1.6e-2 / 7e-2` (mixed pairs use the activation dtype's budget). `new_residual` must equal the fused kernel bit for bit and the graph `add` with at most 0 (f32) or 1 % (16-bit) differing elements. Rows carrying a non-finite input follow IEEE propagation and are compared class-exact (NaN with NaN, same-signed infinity, finite elements bit-exact) across all three implementations. Bitwise identity to the existing fused kernel is reported per case; it is not the pass criterion.

## Artifact contract

As in #2831, the committed `.ptx` is the payload extracted from the `.oxart` container (never the container itself) and the `.cubin` is assembled from it with `ptxas`. One module carries five `.entry` symbols, so there is one manifest per entry, each recording the digests of the shared PTX and cubin, the entry's argument layout from `cuobjdump -elf` cross-checked against the PTX signature and the schema, and the launch rule. `validate` re-derives everything from the committed artifacts; `check-launch` reads the harness output rather than the manifest's own totals.
