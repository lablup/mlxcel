# `qmm_naive` CTA tile width on Volta: the wide tile measured, and refused

Issue #1541 of epic #1536. It asked for a prefill win by unlocking the 128-wide N tile that `qmm_naive` reserves for Ampere and later, on the argument that the exclusion is justified by a shared-memory claim a V100 disproves. The shared-memory half of that argument is correct and is now written into the code as a real device query. The conclusion is not: **the 128-wide tile is slower on this part at every prompt length measured, and the sweep says so consistently enough that the width stays where upstream had it.**

This record follows the six methodology rules of `volta-sm70-baseline-2026-08-31.md` without exception. Read that document first; everything here is a delta against it and every rule it states applies here unchanged.

## Environment

Identical to the baseline record except for the commit. Reproduced here only where it differs or where a number below depends on it.

| Item | Value |
|------|-------|
| **GPU** | Tesla V100-PCIE-32GB, sm_70, compute capability 7.0, 32 GB HBM2, 80 SMs |
| **Driver / toolkit** | 575.51.03 / CUDA 12.9.41 |
| **mlxcel** | 0.6.0 on `perf/issue-1541-qmm-naive-smem-tile`, branched from `8b4a25cb` (main, after #1539) |
| **MLX** | pinned commit `9a795735` |
| **nsys** | Nsight Systems 2025.3.1.90 |
| **Build** | `MLX_CUDA_ARCHITECTURES=70 cargo build --release --features cuda`, `cuobjdump --list-elf libmlx.a` reports 96 cubins, every one sm_70 |
| **PTX cache** | warm for both widths before any timed run. The N tile is part of the JIT module name, so each width compiles its own module once; those two compilations were done first and discarded |
| **Contention** | none. `nvidia-smi --query-compute-apps` was asserted empty before every run by the harness |

Both arms come from **one build**. `MLXCEL_QMM_NAIVE_TILE_N` pins the N tile per process, so the comparison is of two kernels from the same binary on the same machine minutes apart, not of two builds. Everything outside `qmm_naive` is bit-identical between the arms by construction, which is what the unchanged kernel times below confirm.

## What the device actually offers

Queried on this host, since the issue's premise is a claim about these numbers:

```
cudaDevAttrMaxSharedMemoryPerBlockOptin    = 98304   (96 KB)
cudaDevAttrMaxSharedMemoryPerBlock         = 49152   (48 KB, without the opt-in)
cudaDevAttrMaxSharedMemoryPerMultiprocessor= 98304
cudaDevAttrMultiProcessorCount             = 80
cudaDevAttrMaxRegistersPerBlock            = 65536
cudaDevAttrMaxThreadsPerMultiProcessor     = 2048
```

The widest tile the selector can reach at bf16 activations and group size 64 needs 24,576 bytes. That is a quarter of the opt-in budget and half of what a launch gets with no opt-in at all. **The issue is right that shared memory is not what excluded the wide tile from Volta.** What follows is what does.

## Register pressure and occupancy

`qmm_naive`'s kernel is compiled by NVRTC at runtime, so it never passes through an `nvcc -Xptxas -v` line. The equivalent numbers come from two places that agree: `cuobjdump -res-usage` on the cubin mlxcel persists in `MLX_PTX_CACHE_DIR`, and `cuFuncGetAttribute` plus `cuOccupancyMaxActiveBlocksPerMultiprocessor` on the loaded function, which `MLXCEL_TRACE_QMM_TILE=1` prints once per distinct kernel.

Kernel: `qmm_naive_kernel<64, true, false, false, bfloat16_t, uint4b_t, bfloat16_t, Shape<64, N, 64>>`, which is the instantiation `qwen3.8-27B-4bit` prefill runs: group size 64, transposed, **K-aligned rather than K-residue**, and the non-SM80 MMA atom.

| | tile_n = 64 | tile_n = 128 |
|---|---|---|
| Registers per thread | 224 | **255** (the architectural ceiling) |
| Spill to local memory | **0 B** | **128 B per thread** |
| Dynamic shared memory | 16,384 B | 24,576 B |
| Static shared memory | 0 B | 0 B |
| Threads per block | 128 | 128 |
| Blocks per SM | **2** | **2** |
| Warps per SM | 8 of 64 | 8 of 64 |

Three things follow, and they are the whole explanation.

**Shared memory is not the limiter at either width.** Two blocks of the wide tile occupy 49,152 bytes of the SM's 98,304. Even six would fit. The binding resource is registers: 224 x 128 = 28,672 and 255 x 128 = 32,640 against 65,536 per SM, which floors both at two blocks.

**The wide tile buys no occupancy.** Both widths sit at 2 blocks and 8 warps per SM, one eighth of Volta's 64-warp capacity. That number is the subject of #1543, not of this issue.

**The wide tile costs a spill.** 255 registers is the ceiling, and the compiler pays 128 bytes per thread of local memory to stay under it. The 64-wide tile spills nothing.

The mechanism that decides the result is the one thing not in the table: at 128 threads and 2 blocks per SM either way, a 128-wide CTA does twice the output work of a 64-wide one in roughly twice the time, while halving the number of CTAs. That is a win only when the grid is deep enough that halving it still fills the machine. It is not, here, and the next section is why.

## The grid is smaller than the machine, and chunking caps how large it gets

`tile_m` is `clamp(next_power_of_2(m), 16, 64)`, so a prefill of any length at least 64 tokens gets `tile_m = 64` and a grid of `ceil(m / 64)` by `ceil(n / tile_n)` CTAs. At 2 blocks per SM an 80-SM part holds 160 CTAs at once.

`m` is not the prompt length past 2,048. `DEFAULT_PREFILL_CHUNK` in `src/lib/mlxcel-core/src/generate.rs` is 2048 and `effective_prefill_chunk` engages once the prompt exceeds it, so a long prompt is fed through as 2,048-token pieces and the final piece is whatever remains. **`m` is therefore capped at 2,048 for every prompt length**, the M extent of the grid is capped at 32 blocks, and the wide tile's best case is bounded rather than approached asymptotically. That is what makes the sweep below decisive rather than an extrapolation.

## Prefill

Prefill measured directly as the marginal cost of added prompt tokens at `-n 1`, per rule 1, never as a by-product of a decode run. `qwen3.8-27B-4bit`, warm PTX cache, arms interleaved inside each rung so drift hits both equally. Two repetitions per rung up to 1,906 tokens, one at 4,106.

| Prompt tokens | tile_n = 64 (ms) | tile_n = 128 (ms) | 128 against 64 |
|---|---|---|---|
| 106 | 29,317.5 / 29,697.8 | 37,268.4 / 37,383.1 | **1.265x slower** |
| 516 | 87,153.1 / 87,275.0 | 92,266.1 / 93,275.9 | **1.064x slower** |
| 1,906 | 253,614.6 / 252,408.5 | 257,390.0 / 257,127.6 | **1.017x slower** |
| 4,106 | 550,956.5 | 563,107.5 | **1.022x slower** |
| 8,196 | out of memory | not attempted | not measurable here |

Repeat spread is 1.3% at the shortest rung and under 0.5% everywhere else, so every ratio above is many times the noise.

**There is no crossover.** The wide tile's deficit narrows from 26.5% to 1.7% as `m` grows toward the 2,048 chunk size, then widens again to 2.2% at 4,106 tokens, because past 2,048 the prompt is split and the trailing piece is a 10-token chunk whose grid is one M block, the worst case for a wide tile. The 1,906-token rung is close to the best case this workload can present, and the wide tile still loses there.

The least-squares fit over the three single-pass rungs, for continuity with the baseline record's method:

| Arm | ms per prompt token | Marginal tok/s | Intercept |
|---|---|---|---|
| tile_n = 64 (shipped) | 122.92 | **8.14** | 19.66 s |
| tile_n = 128 | 121.20 | 8.25 | 26.98 s |
| Baseline record, for reference | 125.07 | 8.00 | 18.41 s |

**The shipped arm reproduces the committed baseline**: 122.92 against 125.07 ms per prompt token is 1.7%, and 19.66 against 18.41 s of intercept is 6.8%, both within the 10% band the baseline set for a reproduction.

The wide tile's apparently better slope is an artifact and should not be quoted. The relation between prompt length and prefill time is not the same shape for the two widths, so fitting a line to each and comparing slopes attributes a curved difference to the linear term. The per-rung ratios above are the comparison that means something; the fit is here because the baseline row is stated as a fit.

**8,196 tokens is not measurable on this host for this checkpoint.** `cudaMallocAsync failed: out of memory` inside the third chunk, on a 32 GB card holding a 16 GB checkpoint. Chunked prefill releases each chunk's transients between chunks and still does not fit, which is the memory profile issue #672 describes. This is a property of the checkpoint and the card, not of either tile width, and it is reported rather than worked around with a smaller model that would not answer the question the issue asked.

## Kernel attribution

`nsys profile -t cuda,nvtx --cuda-graph-trace=node`, `qwen3.8-27B-4bit`, 106-token prompt, `-n 4`, each profiled run reconciled against the same run unprofiled. The shortest rung is profiled because it is where the effect is largest and therefore where a misattribution would be easiest to spot.

Reconciliation: prefill 98.3% for the 64-wide arm and 99.4% for the 128-wide one, decode 102.3% and 106.5%. All four are close enough to unprofiled wall clock for the absolute times below to carry the conclusion, per rule 5.

| Kernel | Instances (both arms) | tile_n = 64 | tile_n = 128 | Ratio |
|---|---|---|---|---|
| `qmm_naive_kernel` | 497 | **19.9928 s** | 29.8846 s | **1.495x slower** |
| `qmv_kernel` | 1,491 | 0.3386 s | 0.3379 s | 1.00x |
| `event_signal_kernel` | 1,847 | 0.1573 s | 0.1505 s | 0.96x |
| `volta_sgemm_64x64_nn` | 3,360 | 0.1090 s | 0.1102 s | 1.01x |
| All kernels | 23,285 | 20.9810 s | 30.8543 s | 1.471x |

Instance counts are identical for every kernel and for the run as a whole, which is what makes these absolute times comparable at all (rule 6). `qmm_naive` accounts for 9.892 s of the 9.873 s total GPU-time difference; nothing else moves outside its own repeat noise. The `qmv` row is the control and it does not move, which confines the result to the one kernel whose tile changed.

The GPU-time difference (9.87 s) is larger than the wall-clock prefill difference (7.71 s unprofiled, 7.98 s profiled). That is not a contradiction and it matters for reading the result: prefill on this host is partly host-bound, with 4.6 s in `cudaMemcpyAsync` and 1.8 s in `cudaGraphAddKernelNode` on the 64-wide arm, and a slower kernel hides some of that. The wall-clock figure is therefore the conservative one, and it is the one quoted in the table above.

`cuModuleLoadDataEx` is 0.1452 s over 13 calls on the 64-wide arm and 0.1492 s over 13 on the 128-wide one. **The wide tile's cost is not JIT compilation, module loading or graph construction.** It is the kernel.

## The dense pair, which is the row this issue reserved in the baseline record

`volta-sm70-baseline-2026-08-31.md` reserved a post-program comparison row for #1541 against `dense prefill qmm_naive 10.0733 s at 329 inst`, from `gemma-4-12B-it-4bit` at a 46-token prompt and `-n 120`. Same command here, two repetitions, both arms:

| Repetition | tile_n = 64 | Shipped default | Instances |
|---|---|---|---|
| 1 | 10.8801 s | 10.8775 s | 329 |
| 2 | 10.8540 s | 10.8492 s | 329 |
| Mean | **10.867 s** | **10.863 s** | 329 |

The two arms agree to 0.04%, and the shipped default measures the same as an explicitly pinned 64-wide tile, which is the direct confirmation that the selector ships upstream's tile on this part. Reconciliation is 102.4% to 103.3% on prefill and 102.3% to 104.1% on decode across the four profiled runs, so rule 5 is satisfied and the absolute times are usable. Total GPU time is 18.9226 s against 18.9515 s over an identical 152,583 launches.

**The `qmv` control in these same profiles reproduces #1539 to 0.04%**: 6.5626 s and 6.5653 s over 39,151 instances, against the 6.5654 s that issue recorded. That is what says the harness, the host and the methodology are unchanged.

**`qmm_naive` itself does not reproduce, and that is reported rather than smoothed over.** 10.867 s here against 10.0733 s in the baseline record is 7.9% slower, at identical instance counts and identical reconciliation, on a figure whose repeat spread within this session is 0.24%. It is not attributable to the tile: both arms of every pair agree to within 0.05%, and the cubin this build compiles is byte-for-byte the same kernel by resource usage (`REG:224 STACK:0`) as the one cached before this change under the old module name. The GPU was idle at 34 C with no throttling reason active before every run. This is a between-session difference on a compute-bound prefill kernel that no measurement here explains, and it bounds how finely the baseline's `qmm_naive` figure should be read: the 0.55% move #1539 reported for the same kernel is inside a band this large.

## Numerics

Tile width is a blocking decision. It changes which output columns a CTA owns and how many CTAs there are; it does not change the order in which any one output element is accumulated, because `tile_k` and the K loop are untouched. The output must therefore be bit-identical, not merely close.

`qmm_naive_output_is_identical_across_cta_tile_widths` in `src/lib/mlxcel-core/src/ffi_tests.rs` asserts exactly that, at bits 4 and 8 crossed with group sizes 32, 64 and 128, on a `[64 x 512] x [512 x 256]` bf16 matmul, comparing raw output bytes. Both widths run **inside one process** on the same operands, which is what makes the claim bitwise rather than "two runs agreed"; `MLXCEL_QMM_NAIVE_TILE_N` is read on every dispatch so that is possible. It passes.

The `group_size = 128` arm of that test is not decoration. At 16-bit activations the forced 128-wide tile there needs 49,152 bytes of shared memory, past the ceiling a launch gets without `cuFuncSetAttribute`, so that arm is the one that exercises the dynamic shared-memory opt-in on real hardware. Removing the opt-in call makes it fail at launch.

## What shipped, and what did not

**Did not ship: a wider tile on Volta.** The sweep is the reason. Upstream's exclusion of the wide tile below Ampere is correct; the name it gave the predicate, `enough_smem`, is not, and that misnomer is what this issue was filed against.

**Shipped: the shared-memory decision as a device query.** `qmm_naive_tile.h` now compares the tile against `cudaDevAttrMaxSharedMemoryPerBlockOptin` with a 1 KB reserve, instead of letting a shape rule stand in for the budget. On every real device this selects the same tile upstream selects, which `src/lib/mlxcel-core/src/qmm_naive_tile_tests.rs` asserts by enumerating every `(itemsize, group_size, m)` combination against every per-block budget, on the shipped function through a C shim, with no GPU involved.

**Shipped: the dynamic shared-memory opt-in.** f32 activations at group size 128 select a 64 KB tile. Upstream launches it with no `cuFuncSetAttribute`, which fails inside the driver on every architecture from Volta on. It now opts in.

**Shipped: refusal instead of a driver failure.** A tile that does not fit the device budget throws at the launch site naming the tile, the requirement and the budget. Turing is the real instance: its 64 KB per-block budget cannot hold the 64 KB f32 group-128 tile.

**Shipped: the instrument.** `MLXCEL_QMM_NAIVE_TILE_N` pins the width and `MLXCEL_TRACE_QMM_TILE` prints tile, registers, spill and occupancy per kernel. Both exist so this sweep is one command to re-run, which is what #1543 will need: it gives Volta a tensor-core MMA atom, and the register pressure that decided this result is a property of the `UniversalFMA` atom the kernel uses today.

## Deferred to GB10 (sm_121)

No Ampere-or-later device exists on this host, so no sm_80+ runtime number appears above. Per epic #1536's `## GB10 (sm_121) continuation`:

- **GB10 throughput unmoved**: not measured here. The selected tile is provably unchanged on sm_80+ (below), and the only added work on that path is one `cuFuncSetAttribute` in a case that previously failed to launch, but throughput is a measurement and this is an argument.
- **The tiler's sm_80+ selection provably identical for every `(itemsize, group_size, m)` combination**: **closed here, not deferred.** The selector is host code and a pure function of five integers, so the claim is settled by enumeration rather than by owning the silicon. `the_selected_tile_is_upstreams_on_every_architecture` runs that enumeration over both MMA paths, every itemsize, every group size MLX accepts, an `m` sweep covering all three `tile_m` outcomes and both sides of each clamp boundary, and every real per-block shared-memory budget plus the query-failure floor.

The `cuobjdump --dump-sass` technique #1539 used to recover its sm_80+ criterion does **not** transfer, and reporting it as if it did would be misleading. `qmm_naive.cu` is host dispatch only: the kernel lives in `device/qmm_naive.cuh` and ships as an NVRTC source string, so the object contains zero device functions and its SASS dump is empty at every architecture. It was run anyway and is identical at sm_80 and sm_121, which says nothing. What does carry: `device/qmm_naive.cuh` and `device/gemm_sm70.cuh` are untouched, so the JIT source is byte-identical, and the kernel name is a pure function of the tile shape, which the host-side enumeration above pins. Same instantiation, same source, same kernel.

The file compiles clean at `compute_70`, `compute_80` and `compute_121` with the production flag set, which is the compile-time half of the non-regression check and the half CI cannot run today (the runners carry CUDA 13, which removed Volta).

## `gather_gemm.cu`: checked, and nothing to fix

The issue asks whether `mlx/backend/cuda/gemms/gather_gemm.cu` carries the same defect. It does not, for two independent reasons.

Its `make_cta_tiler(int m)` takes no `sm80` argument at this pin. It sets `tile_n = 128` unconditionally on every architecture, so there is no architecture-gated width there to unlock.

It is also unreachable in an mlxcel build. `matmul.cpp`, already an mlxcel overlay, routes the general `M > 1, N > 1` `GatherMM` case to `cutlass_gather_mm` instead of to this file's `gather_mm`. Nothing includes `gemms/gather_gemm.h`, and no object in the built `libmlx.a` carries an undefined reference to `mlx::core::gather_mm(bool, bool, ...)`, while `matmul.cpp.o` does reference `cutlass_gather_mm`.

One real gap does exist in it and is reported rather than fixed here: it computes `smem_bytes` and never calls `cuFuncSetAttribute`, so at f32 with `tile_m >= 128` it asks for 64 KB or 96 KB of dynamic shared memory and fails at launch on every architecture. That is architecture-independent, outside this issue's Volta scope, and in a file this build does not execute. It belongs with #1544, which owns the grouped-GEMM path on sm_70.

## Reproduce

```bash
# 1. Build. Single architecture, explicit.
MLX_CUDA_ARCHITECTURES=70 make release-cuda
cuobjdump --list-elf target/release/build/mlxcel-core-*/out/build/lib/libmlx.a \
  | grep -oE 'sm_[0-9]+a?' | sort | uniq -c    # expect: 96 sm_70

# 2. Tile, registers, spill and occupancy, per distinct kernel. Warms the PTX
#    cache for each width as a side effect; both of these runs are discarded.
P=$(python3 -c "print(' '.join(['The quick brown fox jumps over the lazy dog.']*51) + ' Summarize.')")
for w in 64 128; do
  MLXCEL_QMM_NAIVE_TILE_N=$w MLXCEL_TRACE_QMM_TILE=1 ./target/release/mlxcel generate \
    -m ./models/mlx-community/qwen3.8-27B-4bit -p "$P" -n 1 --profile 2>&1 \
    | grep 'mlxcel qmm_naive'
done

# 3. The same numbers from the persisted cubin, which is what nvcc -Xptxas -v
#    would have printed had this kernel been compiled ahead of time.
C=~/.cache/mlxcel/cuda-ptx/9a795735ad9a
for w in 64 128; do
  cuobjdump -res-usage $C/qmm_naive_tn_aligned_bfloat16_m64_n${w}_b4_g64_affine.ptx
done

# 4. Prefill ladder, both widths, warm cache. -n 1, so prefill is measured as a
#    prompt-length delta and not inside a decode run. Rungs are sentence
#    repetition counts; prompt_tokens comes out at 56 + 10 * reps.
for reps in 5 46 185; do
  P=$(python3 -c "print(' '.join(['The quick brown fox jumps over the lazy dog.']*$reps) + ' Summarize.')")
  for w in 64 128; do
    MLXCEL_QMM_NAIVE_TILE_N=$w ./target/release/mlxcel generate \
      -m ./models/mlx-community/qwen3.8-27B-4bit -p "$P" -n 1 --profile
  done
done
# Read the "Prefill:" line. Compare the two widths at each rung; do not compare
# fitted slopes across widths, the relation is not the same shape for both.

# 5. Kernel attribution. --cuda-graph-trace=node is mandatory, and each profiled
#    run must be reconciled against the same run unprofiled.
P=$(python3 -c "print(' '.join(['The quick brown fox jumps over the lazy dog.']*5) + ' Summarize.')")
for w in 64 128; do
  MLXCEL_QMM_NAIVE_TILE_N=$w nsys profile -t cuda,nvtx --cuda-graph-trace=node \
    -o tile_n$w ./target/release/mlxcel generate \
    -m ./models/mlx-community/qwen3.8-27B-4bit -p "$P" -n 4 --profile
  nsys stats --report cuda_gpu_kern_sum --report cuda_api_sum --format csv tile_n$w.nsys-rep
done

# 6. Bitwise parity across the two widths, and the host-side tiler enumeration.
cargo test --release --features cuda -p mlxcel-core --lib qmm_naive
cargo test --release -p mlxcel-core --lib qmm_naive_tile_tests
```

## References

- Epic #1536, the Volta decode program, and its `## GB10 (sm_121) continuation` section.
- The baseline this is a delta against: `volta-sm70-baseline-2026-08-31.md`.
- The predecessor sub-issue and its record: #1539, `qmv-float-accum-v100-2026-08-31.md`.
- Tile selection: `src/lib/mlx-cpp/patches/mlx/backend/cuda/quantized/qmm/qmm_naive_tile.h` and its launch site `qmm_naive.cu`.
- Host-side enumeration: `src/lib/mlxcel-core/src/qmm_naive_tile_tests.rs`, through the C shim `src/lib/mlxcel-core/cpp/qmm_naive_tile_probe.cpp`.
- Bitwise parity across widths: `qmm_naive_output_is_identical_across_cta_tile_widths` in `src/lib/mlxcel-core/src/ffi_tests.rs`.
- MMA atom selection, which is what `sm80` actually controls: `make_tiled_mma` in MLX's `mlx/backend/cuda/device/gemm_sm70.cuh`.
- Prefill chunking, which caps `m` at 2,048: `DEFAULT_PREFILL_CHUNK` and `effective_prefill_chunk` in `src/lib/mlxcel-core/src/generate.rs`.
- The chunked-prefill memory profile behind the 8,196-token failure: issue #672.
