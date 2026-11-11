# cuda-oxide fused residual-add + RMSNorm evidence: standalone kernel on GB10

Evidence for [issue #2832](https://github.com/lablup/mlxcel/issues/2832), the second gated experiment in [the cuda-oxide feasibility review](cuda-oxide-feasibility.md). Run date: 2026-11-11 (UTC). Host: `spark-d9e2`, a DGX Spark that is not a CI runner.

Status: the GPU acceptance gate is satisfied. Five kernel entry points (f32, f16 and bf16 activations with the weight in the same dtype, plus f16 and bf16 activations with an f32 weight) were compiled with the pinned cuda-oxide toolchain and executed on the GB10 over 130 cases; every output was checked against a host model, against mlxcel's existing fused CUDA kernel and against the graph path on the same bits; Compute Sanitizer reported zero errors under four tools. Nothing here is a compile-only result. The kernel is still standalone: no MLX adapter (#2833), no model (#2834).

## Environment

| Component | Value |
| --- | --- |
| Host | `spark-d9e2`, Linux 7.0.0-1019-nvidia, aarch64 |
| GPU | NVIDIA GB10, compute capability 12.1, driver 580.178.04 |
| CUDA Toolkit | 13.0, V13.0.88 (`nvcc`, `ptxas`, `cuobjdump`, `compute-sanitizer` under `/usr/local/cuda-13.0`) |
| LLVM | 21.1.0 user-space release tarball, `LLVM-21.1.0-Linux-ARM64` (the smoke report's second fallback; apt.llvm.org packages were not installed on this host) |
| Rust (package) | rustc 1.99.1 (58cf8f9c3 2026-10-13), components `rust-src`, `rustc-dev`, `llvm-tools`, `rustfmt`, `clippy` |
| Rust (application, oracle) | 1.97.1, unchanged |
| cargo-oxide | 0.3.0, built from NVIDIA/cuda-rust `b631530700a8a4a5f165456f2d71fe9195b323b5` |
| mlxcel | `28d884a5f2bf1750138b1c99d99c716274aedf1d` plus this change |
| mlxcel CUDA build | `--profile test-fast --features cuda`, `MLX_CUDA_ARCHITECTURES=121`, cuDNN 9.28.1.4 (pip wheel) and OpenBLAS 0.3.30 (user-space source build with its C LAPACK/LAPACKE), `MLX_ENABLE_TF32=0` for the oracle run |
| `gpu-lock` | This host has no upstream `gpu-lock`; a user-local `flock(1)` shim with the same `run --tag <tag> -- <cmd>` shape serialized every GPU command and logged each acquisition. It is a serialization aid only and is named as a shim in `results/compare-sm_121.json`. |

Before any new code ran, the existing `fused_norm_parity_tests` suite of `mlxcel-core` was executed on this host under the same build (10 passed, single-threaded), so the oracle environment itself is the one the repository's own CUDA gate uses.

## Kernel and contract

Source: [`spike/cuda-oxide-fused-norm/src/main.rs`](../../spike/cuda-oxide-fused-norm/src/main.rs). One generic row routine, monomorphized per dtype pair, follows `FUSED_ADD_RMS_NORM_CUDA_SOURCE` in `kernels/fused_norm.cpp` term for term: one block per row, four contiguous elements per lane per sweep, the residual sum rounded to the activation dtype before it is squared, f32 accumulation with an explicit FMA (the contraction nvcc performs), a warp butterfly in the reference order (`xor 16, 8, 4, 2, 1`), per-warp partials through 32 shared floats, `rsqrt.approx.f32` of `total / dim + eps`, `x * inv_mean` rounded before the gain multiply, and the weight bias folded in the weight dtype. 16-bit activations and weights travel as `u16` bit patterns; f16 converts through the compiler's casts and bf16 through `cuda_device::convert::{bf16_to_f32, f32_to_bf16_rne}`.

Launch rule, mirrored from the reference launcher (`fused_norm_threads`): `grid = rows`, `block = clamp(ceil_div(ceil_div(dim, 4), 32) * 32, 32, 1024)`, no dynamic shared memory (132 static bytes per block). The host refuses `rows = 0`, `dim = 0`, `x`/`residual` lengths other than `rows * dim`, a weight length other than `dim`, and more than `2^32 - 1` elements, with a message, before any launch.

## Target

| Property | Emitted value |
| --- | --- |
| PTX ISA version | 8.8 |
| PTX target directive | `.target sm_121` |
| cubin target | `sm_121`, ptxas 13.0, V13.0.88 |
| Entries | `fused_add_rms_norm_f32`, `_f16`, `_bf16`, `_f16_wf32`, `_bf16_wf32`, 13 parameters each (6 × pointer + length, `eps` and `weight_bias` as `.f32`, `dim` as `.u32`) |

| Artifact | SHA-256 |
| --- | --- |
| `artifacts/fused_add_rms_norm.sm_121.ptx` | `b54c1c93c6e8f03e3f3d2cf37a79dd412069153c1360b6161831583c01eaf86e` |
| `artifacts/fused_add_rms_norm.sm_121.cubin` | `677d6d6d3a210161ad649681d783c47bc41e66526d263a9d586538ab0e18825f` |

`scripts/oxide_artifact.py validate` re-derived every entry's argument layout from the committed PTX and cubin, compared it with the schema and the manifest, and checked the digests and the `.oxart` payload identity: five manifests valid.

## Commands

From `spike/cuda-oxide-fused-norm`, in this order (every GPU step under `gpu-lock`):

```bash
scripts/check-lint.sh          # fmt, clippy -D warnings, tooling guards
scripts/build.sh sm_121        # cargo oxide build, export, validate
scripts/run.sh sm_121          # 130 cases + 5 rejections, host model, fixtures, launch check
scripts/oracle.sh              # mlxcel-core example: fused CUDA kernel + graph path
scripts/compare.sh sm_121      # frozen budgets -> results/compare-sm_121.json
scripts/bench.sh sm_121 200    # -> results/bench-sm_121.json
scripts/check-clean-build.sh
for t in memcheck racecheck synccheck initcheck; do
  gpu-lock run --tag issue-2832-sanitize -- compute-sanitizer --tool $t --error-exitcode 3 target/release/cuda-oxide-fused-norm run
done
```

## Frozen tolerances

Recorded in the package README and in `src/main.rs` before the oracle ran. `normed` against each reference, as normalized RMS and normalized max deviation (divided by the reference's RMS), per activation dtype: f32 `1e-6 / 1e-5`, f16 `2e-3 / 1.2e-2`, bf16 `1.6e-2 / 7e-2`; these are `fused_norm_parity_tests::tolerance_for`. `new_residual` must equal the fused kernel bit for bit and the graph `add` with at most 0 (f32) or 1 % (16-bit) differing elements. Rows with a non-finite input are compared class-exact (NaN with NaN, same-signed infinity, finite elements bit-exact) across all three implementations. Bitwise identity to the existing fused kernel was the design target and is reported per case; it is not the pass criterion.

## Results

### Case matrix

26 cases per dtype pair, 130 in all, 2,181,390 elements: widths 1, 3, 31, 32, 33, 100, 127, 128, 129, 1000, 1024, 2048, 3072, 4096, 4097, 8191, 8192 at one and many rows (up to 33 × 2048 and 16 × 8192), the Gemma `weight_bias = 1` fold on four shapes, `eps = 1e-5` on two, and one case per pair with a NaN and a +Inf injected. All 130 passed the host model (an independent f64 model with the same rounding points); `new_residual` matched it bit for bit in every case. The five rejection cases (`zero_dim`, `zero_rows`, `x_length`, `residual_length`, `weight_length`) were refused with a message. The launch check confirmed `grid = rows` and `block = fused_norm_threads(dim)` on all 130 launches, 26 cases and 436,278 elements per entry.

### Against mlxcel's references

`results/compare-sm_121.json`, `pass: true`, 130 of 130 cases:

| Pair | Cases bit-identical to the fused CUDA kernel | Worst `normed` vs graph, normalized RMS | Worst `normed` vs graph, normalized max | Budget (RMS / max) |
| --- | --- | --- | --- | --- |
| f32 / f32 | 26 / 26 | 9.07e-8 | 4.76e-7 | 1e-6 / 1e-5 |
| f16 / f16 | 26 / 26 | 9.42e-6 | 9.71e-4 | 2e-3 / 1.2e-2 |
| bf16 / bf16 | 26 / 26 | 0 | 0 | 1.6e-2 / 7e-2 |
| f16 / f32 weight | 26 / 26 | 5.24e-4 | 1.96e-3 | 2e-3 / 1.2e-2 |
| bf16 / f32 weight | 26 / 26 | 3.91e-3 | 1.64e-2 | 1.6e-2 / 7e-2 |

Every `normed` and `new_residual` output of the standalone kernel equals the existing fused CUDA kernel's bit for bit (2,181,390 of 2,181,390 elements), so the "vs graph" columns are also the existing kernel's own distance from the graph path on these inputs (`fused_cuda_vs_graph` in the JSON is identical). For the two mixed pairs the graph path promotes to f32 (`rms_norm` uses `result_type(x, weight)`); the oracle rounds that output to the activation dtype before the comparison, which is where the larger but in-budget deviations come from. The non-finite rows agreed class-exact in all three implementations: a NaN anywhere makes the row NaN; a +Inf makes `rsqrt(mean)` zero, so the row's finite elements normalize to signed zero and the Inf element to NaN.

### Compute Sanitizer

Each tool ran the full `run` subcommand (130 cases, 5 rejections) under the lock: `memcheck` 0 errors, `racecheck` 0 hazards (0 errors, 0 warnings), `synccheck` 0 errors, `initcheck` 0 errors. The partial last warps and the rows wider than one sweep therefore read nothing outside their buffers, and the two shared-memory stages have no missing barrier.

### Standalone benchmark

`results/bench-sm_121.json`, CUDA events on the launch stream, inputs uploaded once per shape, `cold` = first launch at the shape (the very first also pays the process's first launch), `warm` = mean of 200 launches after 20 warm-ups. Module load: 255 ms. Effective bandwidth counts `x` and `residual` read, `new_residual` written and re-read, `normed` written, and the weight read once per row.

| act / weight | rows × dim | block | cold µs | warm µs | effective GB/s |
| --- | --- | --- | --- | --- | --- |
| f32 | 1 × 1024 | 256 | 18.6 (first launch) | 4.09 | 6.0 |
| f32 | 1 × 4096 | 1024 | 6.2 | 6.14 | 16.0 |
| f32 | 1 × 8192 | 1024 | 10.5 | 8.19 | 24.0 |
| f32 | 5 × 4096 | 1024 | 9.8 | 8.19 | 60.0 |
| f32 | 33 × 2048 | 512 | 7.6 | 6.14 | 264.0 |
| f32 | 64 × 8192 | 1024 | 37.2 | 36.34 | 346.2 |
| bf16 | 1 × 1024 | 256 | 18.3 | 4.09 | 3.0 |
| bf16 | 1 × 4096 | 1024 | 6.8 | 6.13 | 8.0 |
| bf16 | 1 × 8192 | 1024 | 8.3 | 6.15 | 16.0 |
| bf16 | 5 × 4096 | 1024 | 8.4 | 6.14 | 40.0 |
| bf16 | 33 × 2048 | 512 | 7.2 | 6.10 | 132.9 |
| bf16 | 64 × 8192 | 1024 | 25.1 | 18.48 | 340.5 |
| f16 | 1 × 1024 | 256 | 17.8 | 4.09 | 3.0 |
| f16 | 1 × 4096 | 1024 | 8.1 | 5.24 | 9.4 |
| f16 | 1 × 8192 | 1024 | 8.0 | 6.15 | 16.0 |
| f16 | 5 × 4096 | 1024 | 7.9 | 6.14 | 40.0 |
| f16 | 33 × 2048 | 512 | 7.2 | 6.94 | 116.8 |
| f16 | 64 × 8192 | 1024 | 21.0 | 18.33 | 343.2 |

These are the kernel's own costs with no MLX scheduling around them. They say nothing about the integrated path (#2833) and are not a comparison with the existing fused kernel (#2834); batch-1 decode shapes are launch-bound at about 4 to 6 µs, and the wide multi-row shapes reach about 340 to 360 GB/s.

### Gates

`scripts/check-lint.sh`: fmt clean, clippy `-D warnings` clean on the 1.99.1 toolchain, seven tooling guards pass (wrong block, wrong grid, shortened run, missing `RESULT`, no rejections, duplicate case, escaping artifact path). `scripts/check-clean-build.sh`: the package is not a workspace member, the mlxcel lockfile and the `mlxcel` dependency graph resolve no cuda-oxide crate, and `mlxcel-mlx-pin` builds with 1.97.1 and no oxide environment. The oracle example passes `rustfmt --check` and `cargo clippy -p mlxcel-core --features cuda --example oxide_fused_norm_oracle -- -D warnings` with the application toolchain.

## Limits

- Standalone only. No MLX adapter, no lazy evaluation, no graph replay, no model; those are #2833 and #2834, which can reuse the artifacts, the fixtures (`run.sh` regenerates them deterministically) and the benchmark method here.
- Bitwise identity with the existing fused CUDA kernel was observed with this driver, toolchain and MLX pin. The frozen budgets, not the identity, are the acceptance criterion, and a future toolchain that contracts differently would be judged against the budgets.
- The mixed-pair graph reference is rounded to the activation dtype after MLX's f32 promotion; the fused references are never converted.
- `gpu-lock` on this host is the flock shim described above, not the upstream tool.
- The benchmark is the kernel alone; cold numbers include the first-launch cost of the process only on the first shape.
- The f16 path depends on the compiler's `f16` casts (`as f32`, `as f16`), which lowered to `cvt` instructions here; bf16 uses the explicit round-to-nearest-even intrinsic.
