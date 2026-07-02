# MLX Pin Upgrade 2026-07-03: a6ec712 -> e9463bb (GB10 Re-baseline)

Tracking document for issue #625 (epic #623). The pin bump and overlay rebase
were done in PR for #625; the full GB10 re-baseline sweep and per-outlier
verdicts are filled in by the orchestrator after the sweep completes.

## Pin change

| | Commit | Date |
|---|---|---|
| Old pin | `a6ec7123dac814417147e21d4aeed694924ddd4d` | 2026-06-10 |
| New pin | `e9463bbfc1a7cd9e0e6b96aaa3068a316e234a63` | 2026-07-01 |

63 upstream commits between the pins. CUDA-relevant highlights: qmv global
scale (#3723), JIT-compiled qmm_sm80/sm90/gather_gemm (#3706) and qmm_naive
(#3576), rope without copy (#3704), Tegra managed-memory gate (#3701), fused
SDPA vector kernel for asymmetric Q/V head dims (#3637), NAX qmm fixes
(#3631, #3632), large-uncontiguous-grid fix (4885acd).

## Overlay rebase decisions

Overlays are full-file replacements applied by `mlx_apply_source_overlays()`
in `src/lib/mlx-cpp/CMakeLists.txt`. Each was reconciled three-way
(old upstream vs ours vs new upstream).

| Overlay file | Decision | Reason |
|---|---|---|
| `patches/mlx/backend/cuda/binary/binary.cuh` | rebased | Re-applied mixed-precision bf16/fp32 kernels and dispatch onto e9463bb. The previous overlay was authored against v0.31.1 and silently reverted upstream's `__launch_bounds__` additions; the rebase restores them and adopts upstream's large-grid `index_rest` fix (grid.z) and `get_launch_args_general` in both the same-type and mixed paths. |
| `patches/mlx/backend/cuda/device/binary_ops.cuh` | kept | Base file unchanged upstream between pins. Carries intentional mixed-type (bf16, fp32) operator overloads for JIT fused kernels. |
| `patches/mlx/backend/cuda/gemms/grouped_gemm.h` | kept | Base unchanged. Adds `cutlass_gather_mm` declaration for the general GatherMM case. |
| `patches/mlx/backend/cuda/gemms/grouped_gemm_unaligned.cu` | kept | Base unchanged. Implements the general gather matmul (lhs+rhs indices) via CUTLASS grouped GEMM with GPU-side pointer preparation. |
| `patches/mlx/backend/cuda/jit_module.cpp` | rebased | Re-applied the `/proc/self/exe` executable-dir resolution, `MLXCEL_CCCL_DIR` override, and cold-JIT stderr notice onto e9463bb. Adopted upstream's new `cccl_dir()` fallback (dirs.cpp), the extra bundled `include/` path, the JIT'd qmm/cute_dequant header registration (#3706/#3576), CUTLASS nvrtc args, and the `get_jit_module(Device&, ...)` signature change. `MLX_PTX_CACHE_DIR` handling is unchanged upstream, so the mlxcel persistent PTX cache redirect still takes effect. |
| `patches/mlx/backend/cuda/matmul.cpp` | kept | Base unchanged. Carries the CUTLASS GatherMM rework (grouped GEMM for M==1 right-sorted, `cutlass_gather_mm` for the general case) plus the SegmentedMM segments-contiguity tweak. Upstream #3706 JIT-rewrote `gather_gemm.cu` internals, but our overlay bypasses that dispatch entirely, so the JIT rework does not reach the GatherMM path; the re-baseline sweep should confirm the CUTLASS path is still the right call on MoE models. |
| `patches/mlx/backend/cuda/primitives.cpp` | dropped | No-op overlay: byte-identical to upstream except an annotation comment. Pristine upstream is used. |
| `patches/mlx/backend/cuda/quantized/qmm/qmm.h` | dropped | Comment-only delta vs old upstream; pristine e9463bb adds the `global_scale` parameter (#3723) which we want unmodified. |
| `patches/mlx/backend/cuda/quantized/qmm/qmv.cu` | dropped | Comment-only delta vs old upstream (the broadcast_w predicate fix it documented had already landed upstream). Pristine e9463bb carries qmv global-scale support (#3723) and the relocated `device/cute_dequant.cuh` include (#3576). |
| `patches/mlx/backend/cuda/quantized/quantized.cpp` | rebased | Re-applied the `ensure_row_contiguous` fix on `w`/`scales`/`biases` in `QuantizedMatmul::eval_gpu` (3D batched MLA weights, e.g. GLM-4 embed_q) onto e9463bb, which now passes `std::nullopt` global_scale to qmv. The GatherQMM path stays pristine upstream. |
| `patches/mlx/backend/cuda/reduce/all_reduce.cu` | kept | Base unchanged. bf16 output with fp32 accumulation (output type V split from accumulation type U). |
| `patches/mlx/backend/cuda/reduce/col_reduce.cu` | kept | Base unchanged. Same V/U split. |
| `patches/mlx/backend/cuda/reduce/init_reduce.cu` | kept | Base unchanged. Output buffer typed as T so it matches out.dtype() for bf16. |
| `patches/mlx/backend/cuda/reduce/reduce_ops.cuh` | kept | Base unchanged. `ReduceResult<Sum/Prod, bf16>` accumulate-in-fp32 specializations. |
| `patches/mlx/backend/cuda/reduce/row_reduce.cu` | kept | Base unchanged. Same V/U split. |
| `patches/mlx/backend/metal/compiled.cpp` | kept | Base unchanged. static_cast insertion for mixed-dtype compiled ops. |
| `patches/mlx/backend/metal/kernels/utils.h` | kept | Base unchanged. `<metal_simdgroup_matrix>` include and `metal::vec` qualification. |
| `patches-cuda/dtype.cpp` | kept | Base unchanged. bf16+fp32 -> bf16 promotion-table patch (CUDA-only). |
| `patches-cuda/fast.cpp` | kept | Base unchanged. bf16 compute dtype in rms_norm/layer_norm fallbacks (CUDA-only). |
| `patches-cuda/ops.cpp` | rebased | Re-applied `bf16_mixed_astype()` and its use in add/subtract/multiply/divide/maximum/minimum onto e9463bb. The previous overlay was v0.31.1-based and silently reverted upstream hardening (`safe_cast` overflow checks, the `arange` zero-step check, and the use-after-move-prone `auto& shape` bindings); the rebase drops all of those reversions and keeps only the intentional delta. Upstream's new array-API ops (flip, unstack, count_nonzero, trunc, diff, vecdot, ...) come along unmodified. |

Related build plumbing: `src/lib/mlx-cpp/CMakeLists.txt` now defines
`MLX_CCCL_DIR` on the new `mlx_dirs` OBJECT target (upstream moved the macro
consumer into `dirs.cpp` to keep dynamic defines out of the compile cache),
falling back to the `mlx` target for older trees.

## Quantized kernels moved to runtime JIT (deployment impact)

Upstream #3706/#3576 moved the heavyweight CUTLASS qmm/gather_gemm kernel
instantiations from build-time nvcc to runtime NVRTC JIT. Two consequences:

- The cold `cargo build --release --features cuda` dropped from the historical
  30+ minutes to about 7.5 minutes on GB10: the multi-hundred-second nvcc
  invocations for `qmm_sm80.cu`/`qmm_sm90.cu`/`gather_gemm.cu` no longer exist
  at build time.
- The JIT'd kernels include `<cute/...>`/`<cutlass/...>`, so quantized models
  now need CUTLASS/CuTe headers on the deployment host at first run (exactly
  like CCCL before). Handled in this PR: `MLX_CUTLASS_DIR` build-tree fallback
  compiled into the jit_module overlay, `MLXCEL_CUTLASS_DIR` env override,
  release archives now bundle `include/cute/` + `include/cutlass/`, and
  `docs/installation.md` documents the requirement. First quantized-model runs
  pay a one-time per-kernel-variant NVRTC compile, amortized by the persistent
  commit-scoped PTX cache (`ensure_persistent_ptx_cache`).

## Tegra managed-memory finding (#3701)

Upstream #3701 replaced the Windows/WSL-specific managed-memory check with a
plain `concurrentManagedAccess == 0` gate. Probed on this GB10 host:

```
dev 0: NVIDIA GB10 cc12.1 concurrentManagedAccess=1 integrated=1 pageableMemoryAccess=1
```

GB10 reports `concurrentManagedAccess=1`, so `supports_managed_memory()`
returns true under both the old and the new pin and MLX keeps using
`cudaMallocManaged` on GB10. #3701 changes nothing on this platform (it
targets Jetson Orin Nano-class boards and WSL, which report 0). The epic's
hypothesis that the pin predating #3701 was depressing all GB10 numbers is
therefore refuted at the gate level; any sweep-wide movement must come from
the other upstream changes.

## Batch-invariance property change (rope #3704)

Since MLX #3704 RoPE preserves the input layout instead of canonicalizing it
with a copy. Downstream GEMM/SDPA kernels can therefore see different strides
for a row computed inside a batch than for the same row computed at B=1, and
pick different reduction orders. Batched-vs-B=1 hidden states are no longer
bitwise identical on CUDA (observed max_abs 1.6e-4 on fp32 values of
magnitude ~8 in the gemma4 MTP verify fixture). Functionally benign
(argmax-level equivalence holds); the MTP replay test now asserts a 2e-3 fp32
tolerance on CUDA instead of bit equality. Anything downstream that assumed
bitwise batch-invariance on CUDA should be reviewed against this.

Also fixed while reconciling the test suite: NVFP4 block-scale parsing was
broken on CUDA (pre-existing, unrelated to the pin): the CUDA loader widens
F16 safetensors to F32 at load, while the dequant parsed raw bytes as F16.
The scales are now normalized to F32 via astype before parsing, which also
corrects real gemma4-nvfp4 scale handling on CUDA (relevant to the
gemma-4-31b nvfp4 outlier triage).

## Quick post-bump sanity (implementation-time, NOT the re-baseline)

Parity spot-checks (coherent, non-garbage short generations) passed on
llama-3.1-8b-4bit, qwen3-8b-4bit, qwen3-30b-a3b-4bit (MoE), and
qwen2.5-vl-3b-4bit (VLM, image described correctly). Decode sanity via
`mlxcel-bench-decode` (warmup pass + measured pass) vs
`benchmarks/cuda_gb10_2026-06-17.csv`:

| Model | Baseline decode tok/s | Post-bump decode tok/s | Delta | Baseline prefill tok/s | Post-bump prefill tok/s |
|---|---:|---:|---:|---:|---:|
| llama-3.1-8b-4bit | 49.10 | 50.97 | +3.8% | 1294.78 (98 tok) | 1704.15 (132 tok) |
| qwen3-8b-4bit | 47.55 | 49.40 | +3.9% | 236.45 (19 tok) | 418.65 (19 tok) |
| qwen3-30b-a3b-4bit | 90.70 | 93.36 | +2.9% | 133.40 (19 tok) | 135.86 (19 tok) |

No regressions; small consistent decode gains, larger prefill gains on the
dense models. Prompt shapes differ slightly for llama prefill, so treat
prefill deltas as indicative only until the full sweep.

## Re-baseline sweep (TODO: orchestrator)

- [ ] Full sweep CSV `benchmarks/cuda_gb10_<date>.csv` vs `benchmarks/cuda_gb10_2026-06-17.csv`
- [ ] Long-prompt ladder (#624) on the representative subset
- [ ] Per-model delta table

## Outlier verdicts (TODO: orchestrator)

| Outlier | Baseline symptom | Relevant upstream change | Verdict (fixed upstream / improved but open / unchanged) |
|---|---|---|---|
| MoE prefill set | CUDA MoE prefill collapse | #3706 gather_gemm JIT, #3632 gather_qmm NAX name fix | TODO |
| gemma-4-31b nvfp4 | outlier vs roofline | #3723 qmv global scale | TODO |
| hybrid-SSM set | outlier | (none targeted) | TODO |
| gemma-4-31b dense | ~54% of roofline | (general) | TODO |
