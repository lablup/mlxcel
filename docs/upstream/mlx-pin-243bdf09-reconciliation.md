# MLX pin 243bdf09 overlay reconciliation

Issue: [#2303](https://github.com/lablup/mlxcel/issues/2303).

The dependency advances from `59d600b5e64c238427d0f8d897ab7c682ef4d3d2` to `243bdf09c0cccbb85ca6a686dab3cc0102cf1f7e`: 111 upstream commits and 312 changed paths. Every comparison below uses pristine `git show COMMIT:path` objects, never the build checkout after overlays were applied.

## CPU, CUDA and Metal inventory

The original 32 files comprise 30 upstream-backed overlays and two local extracted helpers. Thirteen upstream-backed files changed relative to their declared bases; eleven were rebased and two were retired. The other seventeen upstream-backed files have zero path-specific commits between their declared base and the target, so their original provenance remains valid. The helpers have no upstream path to replace. There are 30 files after retirement.

Paths below use upstream backend-relative names. After the #2305 layout migration, `<backend>/<path>` maps to `src/lib/mlxcelverse/overlays/<backend>/mlx/backend/<backend>/<path>`; `dtype.cpp`, `fast.cpp` and `ops.cpp` live under `src/lib/mlxcelverse/overlays/cuda/mlx/`.

| Path | Previous base | Disposition | Retained behavior / evidence |
| --- | --- | --- | --- |
| `cpu/quantized.cpp` | `81ba1c6a` | Rebased | Float accumulation and buffered scalar affine/MXFP dot products (#2248). |
| `cuda/binary/binary.cuh` | `e9463bb` | Rebased | Direct mixed bf16/f32 binary kernels; upstream FloorDivide support retained. |
| `cuda/cuda_utils.h` | `81ba1c6a` | Retired | Retired: upstream #4514 destroys handles without throwing; no other local delta. |
| `cuda/device/binary_ops.cuh` | `81ba1c6a` | Unchanged; zero drift | Mixed-type operators in runtime-compiled graphs. |
| `cuda/device/gemm_sm70.cuh` | `9a795735` | Unchanged; zero drift | Volta tensor-core MMA path. |
| `cuda/device/qmm_naive.cuh` | `59d600b5e` | Unchanged; zero drift | Volta MMA dispatch parameterization. |
| `cuda/device/qmm_sm80.cuh` | `b7c3dd6d` | Retired | Upstream #4554 drains cp.async and synchronizes before the aliased shared-memory epilogue (#910). |
| `cuda/gemms/grouped_gemm.h` | `b98831ad` | Unchanged; zero drift | Local grouped GEMM dispatch declarations. |
| `cuda/gemms/grouped_gemm_arch.h` | `9a795735` | Local helper | Local extracted architecture selector with host-test coverage (#1544). |
| `cuda/gemms/grouped_gemm_unaligned.cu` | `c9aa5605` | Unchanged; zero drift | Grouped GEMM architecture selection and local kernel dispatch. |
| `cuda/jit_module.cpp` | `9a795735` | Unchanged; zero drift | Bundled CUDA header discovery, overrides and cold-cache notice. |
| `cuda/jit_module.h` | `9a795735` | Unchanged; zero drift | JIT module interfaces needed by local grouped GEMM dispatch. |
| `cuda/matmul.cpp` | `9a795735` | Unchanged; zero drift | Custom grouped GEMM integration and dispatch. |
| `cuda/quantized/qmm/qmm_naive.cu` | `81ba1c6a` | Unchanged; zero drift | Naive quantized GEMM architecture/tile selection. |
| `cuda/quantized/qmm/qmm_naive_tile.h` | `9a795735` | Local helper | Local extracted tile selector with host-test coverage (#1541). |
| `cuda/quantized/qmm/qmm_sm80.cu` | `b7c3dd6d` | Rebased | Blackwell tile overrides and epilogue-fix JIT revision; upstream gathered GEMM implementation retained. |
| `cuda/quantized/qmm/qmv.cu` | `59d600b5e` | Unchanged; zero drift | Local quantized vector kernels and launch controls. |
| `cuda/quantized/quantized.cpp` | `59d600b5e` | Rebased | Large batch chunking and diagnostics; new FP gather-vector path also uses the chunker. |
| `cuda/reduce/all_reduce.cu` | `59d600b5e` | Unchanged; zero drift | Float accumulation with bf16 output and intermediate selection. |
| `cuda/reduce/col_reduce.cu` | `59d600b5e` | Rebased | Float accumulation/intermediates with bf16 output; consume Data without obsolete free callback. |
| `cuda/reduce/init_reduce.cu` | `9a795735` | Unchanged; zero drift | Initialize reduction output in its actual dtype. |
| `cuda/reduce/reduce_ops.cuh` | `ce45c525` | Unchanged; zero drift | Float accumulation traits for bf16 sum/product. |
| `cuda/reduce/row_reduce.cu` | `59d600b5e` | Unchanged; zero drift | Float accumulation with bf16 output. |
| `cuda/scaled_dot_product_attention.cpp` | `81ba1c6a` | Rebased | cuDNN shape bucketing, bounded graph fallback and determinism controls. |
| `cuda/scaled_dot_product_attention.cu` | `e9463bb` | Unchanged; zero drift | Local fused SDPA dtype behavior. |
| `metal/compiled.cpp` | `81ba1c6a` | Unchanged; zero drift | JIT type promotion behavior. |
| `metal/device.cpp` | `81ba1c6a` | Rebased | Runtime command-buffer input-budget override. |
| `metal/kernels/utils.h` | `81ba1c6a` | Rebased | SIMD-group matrix include and qualified metal::vec types. |
| `metal/quantized.cpp` | `59d600b5e` | Rebased | QMV-wide override and B*M broadcast-row correction in GatherQMM and GatherQQMM (#1599). |
| `dtype.cpp` | `ce45c525` | Unchanged; zero drift | CUDA-only bf16/f32 promotion policy. |
| `fast.cpp` | `59d600b5e` | Rebased | CUDA-only bf16 normalization fallback; upstream explicit stream argument retained. |
| `ops.cpp` | `59d600b5e` | Rebased | CUDA-only reduction/promotion behavior; new upstream operation definitions retained. |

Three-way conflict review retained the local normalization compute dtype together with upstream's explicit stream argument. CUDA `GatherQMM` preserves the new `fp_gather_qmv` selection while feeding its index/output chunks through the existing batch limiter. The new expert-grouped `gather_qmm_rhs_sm80` route remains upstream's implementation.

Upstream [#4514](https://github.com/ml-explore/mlx/pull/4514), `9eb3e3ab6fceb4031994f70dfe023872643ed634`, supplies nonthrowing CUDA handle destruction and associated stream/lifetime fixes. The removed overlay had no unrelated local delta; its GLOB registration disappears with the file. The overlapping #4531 was closed unmerged in favor of #4514.

Upstream [#4554](https://github.com/ml-explore/mlx/pull/4554), `5c89fa11e0beb8b22b79d8aeffccb4c935ff1208`, adds `cp_async_wait<0>()` and `__syncthreads()` immediately before the epilogue writes the C tile into shared storage previously used by A/B. The former local #910 barrier appeared before the preceding register-only accumulator conversion; no shared-memory access or additional asynchronous copy occurs between those positions. The upstream pair therefore preserves both async-copy completion and cross-thread read/write ordering for the ordinary and new gathered RHS kernels. Removing the overlay leaves the device header byte-identical to pristine upstream; its only other delta was explanatory provenance. The existing `qmm_sm80_r2` module name remains to preserve the earlier cache boundary.

Upstream [#4594](https://github.com/ml-explore/mlx/pull/4594), `91c83d19b`, provides the Xcode 27 cooperative-tensor workaround. No older `gated_delta_update_nax.h` whole-file overlay is imported from PR #2138. Target `quantized.cpp` still passes the pre-broadcast `x.size() / K` to both gathered matrix routes, so the #1599 correction from #2138 (`7d5459811f16c36d1b05c2a45455f8cb05b99883`) remains necessary and is applied to both as `B * M`. Existing shared-activation MXFP regressions exercise `GatherQMM`; the new NVFP4 matrix route needs appropriate Apple hardware coverage.

The same PR's two filesystem-portability test corrections are retained for candidate Metal CI: canonical temporary parents on macOS and explicit clean rejection on filesystems that reject non-UTF-8 names. They do not change production surgery behavior.

## Ownership and integration

The upstream allocator returns move-only `allocator::Data`, and array donation requires allocator-owned storage. CUDA column-reduction intermediates now consume Data directly; upstream's removed free callback is not reintroduced. The remaining CPU/CUDA/Metal allocations already pass allocator results directly into arrays, preserving the owner and encoder temporary lifetime. The custom C++ bridge's `from_bytes_nocopy` uses the unchanged public raw-pointer/deleter constructor, which upstream now marks non-owned; `kv_inplace_write` uses `copy_shared_buffer`, preserving shared lifetime. `fused_rope_append` delegates allocation to upstream CustomKernel and has no raw allocation handoff to migrate.

ROCm requires an explicit owned-data friend because the public Data constructor with a deleter intentionally represents borrowed storage. Its retarget and backend compatibility changes are recorded in `src/lib/mlxcelverse/overlays/rocm/LOCAL_FIXES.md` and `CORE_RESIDUAL.diff`. The fork remains `75915908dfe5028335d318b10340313744fd3a8d`. The host allocator regression uses the actual overlay header and wrapper; it tests moved ownership, borrowed custom release, empty allocations, pending shared ownership and exception cleanup. It does not execute HIP or prove GPU completion ordering.

The upstream version moves from 0.32.3 to 0.32.4, giving default CUDA PTX caches a new namespace. Validation uses a fresh cache when an explicit `MLX_PTX_CACHE_DIR` override is supplied.

## Validation scope

The PR records the exact tested source commit, fetched MLX HEAD, binary hashes and hardware outcomes. CUDA compilation/inference and candidate Apple Silicon CI are coordinated after source reconciliation. Kimi controls require the independent #2296 model fix in an explicitly recorded combined validation tree; that fix is not part of this dependency PR. ROCm compilation, GPU execution and M5-specific execution are not inferred from host tests or another GPU. This checkout's host has no HIP toolchain.
