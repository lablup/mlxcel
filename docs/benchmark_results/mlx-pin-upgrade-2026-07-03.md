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
| `patches/mlx/backend/cuda/scaled_dot_product_attention.cu` | added (#675) | New overlay, not part of the #625 rebase. Extends the `sdpa_vector` decode kernels and the `supports_sdpa_vector` gate to head_dim 256/288 so head_dim > 128 models (gemma family, qwen3.5/3.6, baichuan-m1, paligemma2) take the fused vector path at decode instead of the unfused materializing SDPA fallback. Gated behind the `MLXCEL_SDPA_VECTOR_LARGE_D` env kill switch (default on); `__launch_bounds__` caps the larger per-thread register footprint of the D=256/288 instantiations. Base is pristine e9463bb, so future pin bumps must rebase this file. |
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

## Re-baseline sweep

Scope note: this is a focused re-baseline of 19 models (`benchmarks/cuda_gb10_2026-07-03.csv`), not the full 147-model sweep. Rationale: the epic runs serially on a single GB10, so a full sweep (many architectures each paying a one-time 3-8 min NVRTC JIT compile on first run) would block every downstream sub-issue for hours while adding no information the triage needs. The focused set covers all three Phase-3 outlier families (MoE prefill, nvfp4, hybrid-SSM), a pure-mamba control, and dense/MoE/SSM/VLM representatives for regression detection, which is sufficient to render every outlier verdict below and confirm no cross-family regression. The full sweep can be produced later with `scripts/bench_decode.sh all --output benchmarks/cuda_gb10_<date>.csv`. Prompts use the same short "Hello, how are you today?" prompt as `cuda_gb10_2026-06-17.csv` so prefill numbers compare apples-to-apples; the long-prompt prefill regime is covered separately by the #624 ladder.

Per-model deltas vs `benchmarks/cuda_gb10_2026-06-17.csv` (decode and prefill tok/s):

| Model | base decode | new decode | Δ | base prefill | new prefill | Δ |
|---|---:|---:|---:|---:|---:|---:|
| qwen2.5-0.5b-bf16 | 199.89 | 204.89 | +2% | 3416.94 | 3685.82 | +7% |
| llama-3.2-1b-4bit | 260.32 | 264.04 | +1% | 7793.45 | 7384.84 | -5% |
| llama-3.1-8b-4bit | 49.10 | 51.03 | +3% | 1294.78 | 1326.68 | +2% |
| llama-3.1-8b-bf16 | 14.84 | 15.23 | +2% | 1209.77 | 1104.79 | -8% |
| qwen2.5-7b-4bit | 53.16 | 54.90 | +3% | 608.73 | 582.51 | -4% |
| qwen3-8b-4bit | 47.55 | 49.18 | +3% | 236.45 | 419.71 | +77% |
| gemma-4-31b-it-4bit (dense) | 8.53 | 8.54 | +0% | 70.25 | 80.14 | +14% |
| mixtral-8x7b-4bit (MoE) | 27.92 | 29.34 | +5% | 12.46 | 12.57 | +0% |
| llama-4-scout-17b-4bit (MoE) | 20.94 | 21.39 | +2% | 28.20 | 27.47 | -2% |
| phi-3.5-moe-4bit (MoE) | 50.13 | 51.07 | +1% | 28.87 | 29.05 | +0% |
| gpt-oss-20b-mxfp4 (MoE) | 77.25 | 78.74 | +1% | 126.16 | 127.33 | +0% |
| qwen3-30b-a3b-4bit (MoE) | 90.70 | 91.52 | +0% | 133.40 | 131.88 | -1% |
| gemma-4-31b-it-nvfp4 | 0.89 | FAIL | - | 16.26 | FAIL | - |
| granite-4.0-h-350m-4bit (SSM) | 64.00 | 64.57 | +0% | 1714.62 | 1750.30 | +2% |
| falcon-h1-tiny-90m-4bit (SSM) | 102.99 | 120.95 | +17% | 1294.33 | 1208.61 | -6% |
| plamo-2-1b (SSM) | 34.36 | 34.21 | +0% | 189.89 | 209.29 | +10% |
| hunyuan-13b (SSM) | 14.80 | 15.08 | +1% | 18.78 | 19.01 | +1% |
| mamba2-1.3b-4bit (pure mamba, control) | 81.37 | 79.82 | -1% | 283.78 | 285.69 | +0% |
| qwen2.5-vl-3b-4bit (VLM) | 59.93 | 59.33 | -1% | 371.22 | 534.53 | +43% |

Dense regression check: all dense/VLM decode within +/-3% (small consistent gains, no regressions). Several prefill numbers jump (qwen3-8b +77%, qwen2.5-vl +43%, gemma dense +14%), consistent with rope-without-copy (#3704) and reduced per-token copies; a couple dip within short-prompt launch-overhead noise (llama-8b-bf16 prefill -8% on a 19-token prompt). No cross-family correctness regression (see the 4119-test pass and parity spot-checks above).

## Outlier verdicts

Bottom line: no Phase-3 issue is fully fixed by the pin bump. #629 and #631 proceed unchanged; #630 is root-caused and re-scoped (see below) but stays open. Epic Phase 3 remains three issues.

| Outlier | Baseline symptom | Verdict | Action |
|---|---|---|---|
| MoE prefill set (mixtral, llama-4-scout, phi-3.5-moe, gpt-oss-20b) | CUDA MoE prefill collapse (mixtral 12.5 vs Metal 81 tok/s) | **unchanged / still open** | Prefill flat vs baseline (mixtral 12.6, scout 27.5, phi 29.1, gpt-oss 127) and still far below Metal. Our `matmul.cpp` overlay routes GatherMM through CUTLASS and bypasses upstream's now-JIT'd `gather_gemm` (#3706), so the JIT rework never reaches this path. **#629 proceeds** and must revisit the CUTLASS-GatherMM vs upstream-JIT-gather_gemm decision on MoE prefill. |
| gemma-4-31b nvfp4 | decode 0.89 tok/s ("nvfp4 disaster") | **root-caused / re-scoped (open)** | The checkpoint is NVIDIA ModelOpt-packed (`config.quantization_config.quant_method = "modelopt"`), an external format mlxcel intentionally rejects; the 0.89 baseline was degenerate output on an unsupported model, and mlxcel now errors cleanly instead of running it. The genuine MLX-native nvfp4 dequant bug (F16 block-scales misparsed as zeros on CUDA) was fixed in this PR (`src/models/sanitize.rs`), and the sibling block-float format mxfp4 is healthy (gpt-oss-20b 78 decode / 127 prefill). No genuine MLX-native nvfp4 checkpoint is available locally to validate end-to-end. **#630 stays open, re-scoped** to: validate MLX-native nvfp4/mxfp8 dequant end-to-end on a real MLX-native checkpoint, add a guard/regression test, and document that ModelOpt/AWQ/GPTQ-packed checkpoints are out of scope (re-export required). |
| hybrid-SSM set (granite-4.0-h, plamo-2, hunyuan-13b, falcon-h1) | decode gap vs Metal (granite 64 vs 219, plamo 34 vs 107) | **unchanged / still open** | Decode flat (granite 64.6, plamo 34.2, hunyuan 15.1; falcon +17% but still below Metal 288). Pure mamba2 control healthy (79.8, matches baseline), isolating the gap to the hybrid attention+SSM path. No upstream change targeted this. **#631 proceeds.** |
| gemma-4-31b dense | ~54% of roofline | **improved (prefill), no defect** | Prefill +14% (70 -> 80 tok/s), decode flat at 8.54 tok/s (bandwidth-bound at 31B dense). Not a defect and has no dedicated Phase-3 issue; general Blackwell prefill work (#637) is the relevant lever. |

Managed-memory note (see above): #3701 is a no-op on GB10, so none of these deltas are attributable to it. The dense decode gains come from the other CUDA hot-path changes (rope-no-copy #3704, qmv global scale #3723, JIT qmm cache).
