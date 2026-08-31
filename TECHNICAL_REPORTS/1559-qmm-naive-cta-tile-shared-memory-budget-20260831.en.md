# Technical Report: PR #1559 - sizing the qmm_naive CTA tile from the shared-memory budget

**Date**: 2026-08-31

**Author**: mlxcel maintainers

**Status**: Completed. The performance hypothesis the issue was filed on is refuted by measurement and the tile width is unchanged; three latent defects the old predicate was hiding are fixed, and one acceptance criterion that epic #1536 deferred to GB10 hardware is closed here instead. sm_80-and-later runtime validation deferred to a GB10 host.

---

## Executive Summary

`qmm_naive`, the CUDA quantized matmul behind every prefill on a pre-Ampere part, sized its CTA N tile with `enough_smem = sm80 && itemsize <= 2 && group_size <= 64`. Two unrelated decisions sat under that one name. The shape terms are a shared-memory rule, and they are now written as one: a comparison against `cudaDevAttrMaxSharedMemoryPerBlockOptin` queried from the running device. The `sm80` term is not a shared-memory term at all, and a Tesla V100 disproves it: 96 KB per block through the opt-in and 48 KB without it, against the 24 KB the widest eligible tile needs.

Issue #1541 expected a prefill win from unlocking the wide tile on that headroom. **It was swept rather than assumed, and it loses.** The 128-wide instantiation needs 255 registers, the architectural ceiling, and spills 128 bytes per thread, where the 64-wide one needs 224 and spills nothing. Both reach the same 2 blocks per SM, so the extra width buys no occupancy. What decides it is the CTA count: a 128-wide CTA does twice the output work in roughly twice the time, and halving the grid on a part that is not filled to begin with is a straight loss. `qmm_naive` measures 1.50x slower at a 106-token prompt, 1.06x at 516, 1.02x at 1,906 and 1.02x at 4,106, at identical launch counts.

The tile width therefore ships unchanged on every architecture. What the change delivers is everything the shape rule was hiding: the shared-memory opt-in that f32 activations at group size 128 have always needed and never had, a refusal instead of a driver failure when no tile fits, the N tile in the JIT module name that the module and PTX caches key on, and a host-side enumeration that settles the sm_80+ non-regression claim without owning an Ampere part.

## 1. Problem Statement

`mlx/backend/cuda/quantized/qmm/qmm_naive.cu:17-23` at pin `9a795735`:

```cpp
inline auto make_cta_tiler(int itemsize, int m, int group_size, bool sm80) {
  bool enough_smem = sm80 && itemsize <= 2 && group_size <= 64;
  int tile_m = std::max(16, std::min(64, next_power_of_2(m)));
  int tile_n = enough_smem ? 128 : 64;
  int tile_k = std::max(64, group_size);
  return cute::make_shape(tile_m, tile_n, tile_k);
}
```

Three problems, only one of which the issue named.

**The predicate is misnamed.** `enough_smem` reads as a shared-memory budget test and is not one. It contains an architecture comparison, `compute_capability_major() >= 8`, whose relationship to shared memory is nil: the per-block ceiling without an opt-in is 48 KB on every architecture from Volta onward, and the opt-in ceiling is 96 KB on sm_70, above Turing's 64 KB and below Ampere's. Nothing about shared memory distinguishes the two sides of that comparison.

**Nothing opts into the larger maximum.** The tile at f32 activations and group size 128 needs `4 * 128 * (64 + 64) = 65,536` bytes of dynamic shared memory. A launch gets 48 KB without `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)`, and upstream `qmm_naive.cu` never calls it, unlike its siblings `qmm_sm80.cu` and `qmm_sm90.cu`. That shape fails inside the driver on every architecture from Volta on.

**Nothing checks that the selected tile fits.** Turing's opt-in budget is 64 KB, so even with the opt-in the same shape does not fit there, and the failure surfaces as a CUDA launch error naming neither the tile nor the budget.

## 2. Change Summary

| File | Change |
|---|---|
| `patches/mlx/backend/cuda/quantized/qmm/qmm_naive_tile.h` | New. The selector as pure integer arithmetic: no CuTe, no CUDA, no MLX. Shared by the CUDA overlay and by the host-side test, so the tested function is the shipped one. |
| `patches/mlx/backend/cuda/quantized/qmm/qmm_naive.cu` | New overlay. Queries the budget once per device; opts in when the tile crosses the opt-in-free ceiling; refuses a tile that does not fit, naming the tile, the requirement and the budget; throws if the CuTe layouts stop agreeing with the selector's model; puts tile_n in the JIT module name; adds `MLXCEL_QMM_NAIVE_TILE_N` and `MLXCEL_TRACE_QMM_TILE`. |
| `mlxcel-core/cpp/qmm_naive_tile_probe.cpp`, `build.rs` | New C shim over the selector, compiled unconditionally rather than behind the `cuda` feature. |
| `mlxcel-core/src/qmm_naive_tile_tests.rs` | New. The enumeration, plus the opt-in ceiling, the refusal, the reserve boundary and the override. Nine tests, no GPU. |
| `mlxcel-core/src/ffi_tests.rs` | `qmm_naive_output_is_identical_across_cta_tile_widths`, bitwise across bits 4 and 8 crossed with group sizes 32, 64 and 128, both widths inside one process. |
| `docs/benchmark_results/qmm-naive-tile-v100-2026-08-31.md` | New. The full record. |
| `docs/benchmark_results/volta-sm70-baseline-2026-08-31.md` | The `#1541` row of the post-program comparison table. |
| `CHANGELOG.md` | Entry under `### Changed`. |

## 3. Technical Decisions

### 3.1 The sweep decided the width, and it decided against the issue's hypothesis

The issue's shared-memory arithmetic is correct and the conclusion drawn from it is not. Measured on this host:

| | tile_n = 64 | tile_n = 128 |
|---|---|---|
| Registers per thread | 224 | 255 (the ceiling) |
| Spill to local memory | 0 B | 128 B per thread |
| Dynamic shared memory | 16,384 B | 24,576 B |
| Blocks per SM | 2 | 2 |
| Warps per SM | 8 of 64 | 8 of 64 |

Shared memory is not the limiter at either width: two blocks of the wide tile occupy 49,152 bytes of the SM's 98,304, and even six would fit. Registers are: 224 x 128 = 28,672 and 255 x 128 = 32,640 against 65,536 per SM, which floors both at two blocks.

That the two widths reach the same occupancy is what makes the result clean. The wide tile is not trading occupancy for arithmetic intensity, it is simply doing twice the work per CTA in a grid that already does not fill an 80-SM part, and halving the CTA count costs more than the halved A-tile traffic saves. `qmm_naive` at a 106-token prompt: 19.9928 s against 29.8846 s over an identical 497 instances.

There is no crossover to wait for, and that is structural. `DEFAULT_PREFILL_CHUNK` is 2,048, so a longer prompt is fed through in 2,048-token pieces and `m` is capped, which caps the M extent of the grid at `2048 / 64 = 32` blocks. The 1,906-token rung is close to the best case this workload can present, and the wide tile still loses there by 1.7%; at 4,106 tokens the deficit widens back to 2.2% because the trailing chunk is 10 tokens, the worst case for a wide tile.

Upstream's exclusion of the wide tile below Ampere is therefore correct, for a reason the name `enough_smem` did not give. The predicate now carries the term under the name of what it actually selects, `tensor_core_mma`, with the measurement cited beside it.

### 3.2 The selector is a pure host function, so its non-regression claim is a unit test rather than a GPU

Epic #1536's GB10 handoff lists "the tiler's sm_80+ selection provably identical for every `(itemsize, group_size, m)` combination" as deferred to Ampere-or-later hardware. It is not a hardware question. The selector is host code and a pure function of five integers, so the claim is settled by enumeration.

Factoring it into `qmm_naive_tile.h` with no CuTe, no CUDA and no MLX in it makes that possible: a C shim in `mlxcel-core/cpp/` includes the same file CMake copies into the fetched MLX tree, and `qmm_naive_tile_tests.rs` sweeps it. There is one definition and the test exercises it rather than a restatement of it. The shim is compiled unconditionally rather than behind the `cuda` feature, because the whole point is that the claim is checkable on a host with no NVIDIA hardware.

The test asserts more than the criterion asked for: every `(itemsize, group_size, m)` combination against **every** real per-block budget and both MMA paths, so it pins the selected tile as identical to upstream's on every architecture, not only on sm_80 and later.

### 3.3 The N tile had to join the JIT module name

`get_jit_module` caches by module name, and `read_cached_ptx` keys the persistent PTX disk cache on the same string. The tile shape reaches the compiler only through the kernel name. Upstream is safe because tile_n is a deterministic function of terms already in the module name, but the moment the selection can move, a module cached under a name that does not pin tile_n is handed back with no kernel matching the requested instantiation, which surfaces as `There is no kernel named ...`.

Adding `n{}` also invalidates PTX cached by builds from before this change, which is the same hazard #910 fixed in `qmm_sm80.cu` by bumping its module name to `qmm_sm80_r2`.

### 3.4 `MLXCEL_QMM_NAIVE_TILE_N` is read on every dispatch, and that is what makes parity bitwise

Tile width is a blocking decision. It changes which output columns a CTA owns and how many CTAs there are; it does not change the order in which any one output element is accumulated, because `tile_k` and the K loop are untouched. So the output must be bit-identical, not merely close.

Proving that across two processes would not distinguish a tile-width effect from any other per-process nondeterminism, and `cuda_qmm_determinism` records that quantized prefill on sm_70 is not reproducible across processes. Reading the override on every dispatch rather than caching it lets one process run both widths on identical operands and compare raw bytes. It is safe precisely because of 3.3: each width has its own module and kernel. The cost is a `getenv` next to two `fmt::format` calls and a map lookup already on that path.

### 3.5 The selector's shared-memory model is checked against CuTe at runtime

The selector predicts `itemsize * tile_k * (tile_m + tile_n)` from the tile extents alone. The launch site computes the real figure from `cute::cosize` of the resolved layouts, as upstream does, and throws if the two disagree. They agree at this pin because both tile extents are multiples of the swizzle base, so each layout's cosize is exactly its product. An MLX pin bump that changes the layouts would otherwise make every fit and opt-in decision above silently wrong; it is now a loud error naming both figures.

## 4. Validation

Every number follows the six methodology rules of `volta-sm70-baseline-2026-08-31.md`: prefill measured directly as a prompt-length delta at `-n 1`, warm PTX cache for both widths, `--cuda-graph-trace=node`, every profile reconciled against the same run unprofiled, and absolute kernel times compared only at matching instance counts. Both arms come from one build.

**Prefill**, `qwen3.8-27B-4bit`, arms interleaved inside each rung:

| Prompt tokens | tile_n = 64 | tile_n = 128 | 128 against 64 |
|---|---|---|---|
| 106 | 29,317.5 / 29,697.8 ms | 37,268.4 / 37,383.1 ms | 1.265x slower |
| 516 | 87,153.1 / 87,275.0 ms | 92,266.1 / 93,275.9 ms | 1.064x slower |
| 1,906 | 253,614.6 / 252,408.5 ms | 257,390.0 / 257,127.6 ms | 1.017x slower |
| 4,106 | 550,956.5 ms | 563,107.5 ms | 1.022x slower |

**Kernel attribution**, 106-token prompt at `-n 4`, reconciliation 98.3% and 99.4% on prefill: `qmm_naive_kernel` 19.9928 s against 29.8846 s over an identical 497 instances, accounting for 9.892 s of the 9.873 s total GPU-time difference. `qmv_kernel` is the control at 0.3386 against 0.3379 s over 1,491 instances and does not move. `cuModuleLoadDataEx` is 0.1452 against 0.1492 s over 13 calls, so the cost is the kernel and not JIT, module loading or graph construction.

**Reproduction of the committed baseline**: the fit over the single-pass rungs gives 122.92 ms per prompt token against 125.07, and an intercept of 19.66 s against 18.41 s, both inside the 10% band the baseline set. On the dense pair the `qmv` control reproduces #1539 to 0.04%, 6.5626 s against 6.5654 s over 39,151 instances.

**Tests**: `cargo test --release --features cuda -p mlxcel-core --lib qmm_naive` reports 9 passed, 0 failed. Eight host-side tiler tests and one GPU parity test. `qmm_naive.cu` compiles clean with the production flags at `compute_70`, `compute_80` and `compute_121`. `cargo clippy -p mlxcel-core --lib --all-features -- -D warnings` and `cargo fmt --check` are clean. `cuobjdump --list-elf libmlx.a` reports 96 cubins, every one sm_70.

**Shipped selection confirmed on hardware**: `MLXCEL_TRACE_QMM_TILE=1` on the default build reports `tile 64x64x64 ... regs 224 spill 0 B blocks/SM 2`, identical to `MLXCEL_QMM_NAIVE_TILE_N=64`, and 85,535.8 ms against 85,672.5 ms of prefill at a 566-token prompt.

## 5. Validation Limits and Follow-up

### 5.1 8,192 prompt tokens is not measurable on this host for this checkpoint

The issue asks for prefill at roughly 600, 2,048 and 8,192 prompt tokens. The 8,192 rung fails with `cudaMallocAsync failed: out of memory` inside the third chunk, on a 32 GB card holding a 16 GB checkpoint, which is the chunked-prefill memory profile issue #672 describes. It is a property of the checkpoint and the card rather than of either tile width.

The 4,106-token rung replaces it and answers the same question, because prefill chunking caps `m` at 2,048 and therefore caps how large the grid can ever get. Substituting a smaller model would have produced a number at 8,192 that did not answer the question the issue asked.

### 5.2 The `qmm_naive` figure does not reproduce the baseline record, and that is reported

On the dense pair, `qmm_naive` measures 10.867 s against the baseline record's 10.0733 s at identical 329 instances and identical reconciliation, a 7.9% gap on a figure whose repeat spread within this session is 0.24%. It is not attributable to the tile: both arms of every profile agree to within 0.05%, and the cubin this build compiles has the same resource usage (`REG:224 STACK:0`) as the one cached before this change under the old module name. The `qmv` control in the same profiles reproduces #1539 to 0.04%, so the harness and the host are consistent. This is an unexplained between-session difference on a compute-bound prefill kernel, and it bounds how finely the baseline's `qmm_naive` figure should be read.

### 5.3 The SASS-diff technique from #1539 does not transfer

#1539 recovered its sm_80+ non-regression criterion on a Volta host by compiling `qmv.cu` at sm_80 and sm_121 and diffing `cuobjdump --dump-sass`. That does not work here and it would be misleading to report it as if it did.

`qmm_naive.cu` is host dispatch only. The kernel lives in `device/qmm_naive.cuh`, which MLX's build sweeps into `cuda_jit_sources.h` as an NVRTC source string, so the object contains **zero device functions** and its SASS dump is empty at every architecture. It was compiled at `compute_80` and `compute_121` and the dumps are identical, which says nothing at all.

What replaces it: `device/qmm_naive.cuh` and `device/gemm_sm70.cuh` are untouched, so the JIT source string is byte-identical; the kernel name is a pure function of the tile shape; and the host-side enumeration pins that shape as identical to upstream's on every architecture. Same source, same instantiation, same kernel. That is a stronger argument than a SASS diff of an empty object, and it is the argument this change actually rests on.

### 5.4 `gather_gemm.cu` was checked and is not included

The issue asks whether `mlx/backend/cuda/gemms/gather_gemm.cu` carries the same defect. It does not. Its `make_cta_tiler(int m)` takes no `sm80` argument at this pin and sets `tile_n = 128` unconditionally on every architecture, so there is no architecture-gated width there to unlock. It is also unreachable in an mlxcel build: `matmul.cpp`, already an mlxcel overlay, routes the general `M > 1, N > 1` `GatherMM` case to `cutlass_gather_mm`; nothing includes `gemms/gather_gemm.h`; and no object in the built `libmlx.a` carries an undefined reference to `mlx::core::gather_mm(bool, bool, ...)` while `matmul.cpp.o` does reference `cutlass_gather_mm`.

One real gap does exist in it: it computes `smem_bytes` and never calls `cuFuncSetAttribute`, so at f32 with `tile_m >= 128` it asks for 64 KB or 96 KB of dynamic shared memory and fails at launch on every architecture. That is architecture-independent, outside this issue's Volta scope, and in a file this build does not execute. It belongs with #1544, which owns the grouped-GEMM path on sm_70.

### 5.5 Deferred to GB10, and one criterion recovered

No Ampere-or-later device exists on the implementation host. Per epic #1536's `## GB10 (sm_121) continuation`:

- **GB10 throughput unmoved**: deferred. The selected tile is provably unchanged on sm_80+ and the only added work on that path is one `cuFuncSetAttribute` in a case that previously failed to launch, but throughput is a measurement and that is an argument.
- **`cargo test --features cuda` green on sm_121**: deferred. It is green on sm_70 for the targeted suites this change touches; the full workspace CUDA run needs `--test-threads=1` per #1048 and did not fit this host's run budget.
- **The tiler's sm_80+ selection provably identical for every `(itemsize, group_size, m)` combination**: **closed here**, by enumeration rather than by hardware. See 3.2.

### 5.6 Re-measure after #1543

The register pressure that decided this result is a property of the `UniversalFMA<float, Element, Element>` atom `make_tiled_mma` selects below Ampere. #1543 gives Volta a tensor-core MMA for quantized GEMM, which changes the accumulator fragment layout and therefore the register count that put the wide tile at 255 with a spill. `MLXCEL_QMM_NAIVE_TILE_N` and `MLXCEL_TRACE_QMM_TILE` exist so that re-measurement is one command; the reproduce block of the benchmark record is the command.

## References

- Issue #1541 and epic #1536, the Volta decode program.
- Baseline this is a delta against: `docs/benchmark_results/volta-sm70-baseline-2026-08-31.md`.
- Full record for this change: `docs/benchmark_results/qmm-naive-tile-v100-2026-08-31.md`.
- Predecessor sub-issue and its record: #1539, `docs/benchmark_results/qmv-float-accum-v100-2026-08-31.md`.
- MMA atom selection, which is what `sm80` actually controls: `make_tiled_mma` in MLX's `mlx/backend/cuda/device/gemm_sm70.cuh`.
- The module-name cache hazard this repeats: #910, in `qmm_sm80.cu`.
- Prefill chunking, which caps `m` at 2,048: `DEFAULT_PREFILL_CHUNK` and `effective_prefill_chunk` in `src/lib/mlxcel-core/src/generate.rs`.
- The chunked-prefill memory profile behind the 8,196-token failure: #672.
- Single-threaded CUDA test requirement: #1048.
