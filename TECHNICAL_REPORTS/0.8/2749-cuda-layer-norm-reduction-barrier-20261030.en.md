# PR #2749: Add a barrier to MLX CUDA layer_norm block reductions

**Date**: 2026-10-30 (report written pre-merge; the PR was open and in draft at the time)
**Status**: Open PR, validated on CUDA (GB10, sm_121) only. Other CUDA architectures were not run; Metal and ROCm do not compile this file
**Languages**: CUDA C++, Rust, Markdown
**Risk Level**: Low (one added `block.sync()` that does not change arithmetic; the cost is a new overlay file that must be re-diffed on every MLX pin bump)

## Executive Summary

PR #2749 (closes #2696) fixes a shared-memory race in MLX's CUDA `fast::layer_norm` that made LayerNorm nondeterministic on some row widths. `BlockBroadcastReduce::Reduce` had a barrier between writing its per-warp partials and reading them, but none after the read. The `layer_norm` kernel reduces twice through one buffer (mean, then variance), so a warp that finished early could overwrite a slot with its variance partial before a slower warp had read the mean partial from it. `layer_norm_vjp` has the same hazard through its `temp.f` / `temp.f3` union.

The fix is a new CUDA overlay, `src/lib/mlxcelverse/overlays/cuda/mlx/backend/cuda/layer_norm.cu`: the pin's file plus a second `block.sync()` after the read. Every `fast_layer_norm` caller in mlxcel picks it up with no source change. A new test, `tests/cuda_layer_norm_determinism.rs`, repeats the kernel 32 times per case and requires byte-identical outputs that agree with a composed f32 reference. On main the bf16 `[2, 32, 32, 384]` case gave 6 to 9 distinct outputs per 32 calls; with the overlay every case gives one.

## 1. Problem Statement

### 1.1 Background

The race was found in #2688 (Mage-Flow port, epic #2348). The Mage-VAE encoder's `norm_out` (bf16, `[2, 32, 32, 384]`, affine, eps 1e-6) produced a few wrong rows per call: 30 of 30 calls differed on mlx 0.32.4, and two identical Edit-Turbo runs gave PNGs only 29.8 to 35.5 dB apart, in upstream mlx-vlm as well as in mlxcel. #2688 worked around it locally by composing LayerNorm from separate f32 ops in `src/imagegen/mage_flow/layers.rs`. Every other CUDA caller of the fused kernel stayed exposed: `mlxcel_core::layers::LayerNorm::forward` (used across text, vision, audio and token-classification models), the CUDA fallback of `residual_add3_layer_norm`, direct calls in Cohere, StableLM, Nemotron, the audio encoder, the dense vision towers, FLUX.2 embeddings, the VAE GroupNorm path, and the shared DiT `FinalLayer` used by Z-Image, LLaDA-Image and Ming-Image.

### 1.2 Cause

In the pinned `mlx/backend/cuda/layer_norm.cu` (`7ca40033`), `Reduce` does: warp-level reduce, lane 0 writes `temp[warp]`, `block.sync()`, every warp reads all partials, warp-level reduce again, return. Nothing stops a warp from leaving `Reduce`, computing its next partial and writing `temp[warp]` while another warp is still reading. A single reduction per kernel is safe; two reductions through the same `temp` are not.

The file at the pin is byte-identical to `ml-explore/mlx` main as of 2026-10-28, and no upstream issue or PR covers it, so a pin bump would not fix it. ml-explore/mlx#3630 fixed the same class of hazard in the Metal VJP only, and the ROCm overlay already fixed its own copy (`overlays/rocm/LOCAL_FIXES.md` item 51, #2381).

### 1.3 Why only some shapes reproduce

The launch picks `BLOCK_DIM` from `ceil_div(axis_size, N_READS)` with `N_READS = 16 / sizeof(T)`. The PR's measurements line up with how much work the last warp gets:

| Case | Threads needed | Block | Last warp | Race visible on main |
|---|---|---|---|---|
| bf16, 384 wide | 48 | 64 (2 warps) | half idle, finishes first with a nonzero partial | yes, 6 to 9 distinct outputs per 32 calls |
| f32, 384 wide | 96 | 128 (4 warps) | fully idle, writes 0 over 0 | no |
| bf16, 3840 wide | 480 | 512 (16 warps) | fully idle, writes 0 over 0 | no |
| bf16, 4096 wide | 512 | 512 | full, equal work | no |

The race needs a warp that has some work but less than the others, so it runs ahead and writes a nonzero value into a slot another warp still needs. A fully idle warp writes zero over zero and the corruption is invisible. This is also why the issue required the test input to vary within each row: constant inputs give every warp identical partials and hide the race.

### 1.4 Risk of not fixing

| Risk | Impact | Likelihood |
|---|---|---|
| Run-to-run different outputs in any CUDA model whose LayerNorm width leaves a partly filled last warp | High for parity tests and reproducibility | High on affected widths (6 to 9 of 32 calls) |
| Silent quality loss (wrong rows, not NaN) that looks like numerical noise | Medium | Medium |
| Per-family workarounds like #2688 spreading across the codebase | Low | Medium |

## 2. Change Summary

Two commits plus a merge from main, three files (+781 / -2).

| File | Change |
|---|---|
| `src/lib/mlxcelverse/overlays/cuda/mlx/backend/cuda/layer_norm.cu` (new, 424 lines) | Copy of the pin's file with Apple's header kept, a `Modified from upstream MLX 7ca40033...` provenance line, and one added `block.sync()` after the `temp` read in `BlockBroadcastReduce::Reduce`, commented with the issue number. |
| `src/lib/mlxcelverse/overlays/cuda/README.md` | CUDA overlay count 27 to 28, with a `layer_norm.cu` entry in the file list. |
| `tests/cuda_layer_norm_determinism.rs` (new, 355 lines) | Five `#[cfg(feature = "cuda")]` tests: four forward cases and one VJP case, 32 calls each. |

No CMake change is needed: `mlx_apply_source_overlays` already copies every file under `overlays/cuda/mlx/backend/cuda/` into the fetched MLX tree on each build, and MLX compiles that directory only when `MLX_BUILD_CUDA` is on.

| Hash | Type | Message |
|---|---|---|
| `2cd7f730` | fix | add a barrier to MLX CUDA layer_norm block reductions |
| `ea03339e` | test | assert exact byte equality in the CUDA layer_norm determinism check |
| `0ec6dea9` | merge | origin/main into the branch (no content of its own) |

## 3. Technical Decisions

### 3.1 Fix inside `Reduce`, not in the kernels

**Alternatives considered:**

| Option | Pros | Cons |
|---|---|---|
| Two separate `__shared__` buffers in `layer_norm` | No extra barrier in the forward kernel | Leaves the VJP's `temp.f` / `temp.f3` union racing; larger diff from the pin |
| Barrier at each call site between the two reductions | Explicit at the point of reuse | Two edits per kernel; a future third reduction would repeat the bug |
| **Chosen: second `block.sync()` at the end of `Reduce`** | One line covers the forward kernel and the VJP; any later caller that reuses the storage is safe | One extra block barrier per reduction, also on the second reduction where it is not strictly needed |

`Reduce` is the shared primitive that owns the storage, so it is the place that knows when the storage is free again. Putting the barrier there makes "the buffer is reusable after `Reduce` returns" part of the primitive's contract.

### 3.2 Overlay instead of an upstream bump or a source workaround

Upstream main has the same bug, so no pin bump fixes it. A composed-ops workaround (what #2688 did for Mage-Flow) would have to be repeated at every call site listed in 1.1 and is slower than the fused kernel. The overlay mechanism already exists for 27 other CUDA files, fixes every caller at once, and keeps the change a one-line diff from the pin, which makes it easy to drop when upstream fixes the bug.

### 3.3 Keep the overlay a one-line diff, even though a second bug was found

Review found a separate upstream bug in the same file (see section 6). Fixing it here would have made the overlay harder to audit against the pin and mixed a nondeterminism fix with a correctness fix that has no test at a reachable width. It was left for its own issue.

### 3.4 Test design

- Inputs are a splitmix64 hash of the element index mapped to [-1, 1], plus a per-row mean offset, so partial sums differ between warps. The issue noted that alternating constants (as in `layer_norm_multiwave_mean_snapshot_is_stable`) do not expose the race.
- Each case checks two things: determinism (every call's bytes equal the first call's) and accuracy against a composed f32 reference (mean, centered variance, rsqrt and affine in f32, one cast back). Bounds: 1e-5 absolute for f32; one bf16 ulp of the reference plus 1e-2 for bf16, because the kernel rounds the normalized value to bf16 before the affine step.
- The first commit counted distinct outputs by a 64-bit hash. `ea03339e` added `differing_calls == 0`, an exact byte comparison against the first call, so a hash collision cannot hide a difference. The hash count is still reported in the failure message.
- Shapes cover the #2688 shape in bf16 and f32, a wide bf16 row (`[8, 4096]`), the affine-free DiT `FinalLayer` shape (`[1, 4096, 3840]`), and the VJP at `[2048, 384]` through `fast_layer_norm_vjp_for_test`.

## 4. Validation

All on GB10 (sm_121), `MLX_ENABLE_TF32=0`.

**Failing first.** The same test source was built without and with the overlay, three runs each:

| Case | main: distinct outputs (calls outside the reference bound) | overlay |
|---|---|---|
| bf16 `[2, 32, 32, 384]` affine | 8 (7), 9 (9), 6 (5); up to 8 corrupted rows per call, worst excess 1.37 over the bound | 1 (0), three runs |
| f32 `[2, 32, 32, 384]` affine | 1 (0) | 1 (0) |
| bf16 `[8, 4096]` affine | 1 (0) | 1 (0) |
| bf16 `[1, 4096, 3840]` affine-free | 1 (0) | 1 (0) |
| bf16 `[2048, 384]` VJP dx | 1 (0) | 1 (0) |

A 256-call scratch probe gave 26 corrupted calls on main for the bf16 384 case and none with the overlay. Ten more forward and VJP shapes (448, 896 and 4096 wide, up to 8192 rows) were identical over 256 calls before and after. The reference bounds pass on every clean output and fail only on corrupted ones, so they measure the race and not ordinary rounding.

**Real-checkpoint parity** (`MLXCEL_REQUIRE_MODELS=1`, two runs before and two after):

| Family | Test | Before | After |
|---|---|---|---|
| Z-Image Turbo | `turbo_generation_and_edit_match_upstream` | pooled latents 3.152e-2, pooled PSNR 36.25 dB | same |
| Ming-Image bf16 | `bf16_matches_upstream` | gen 3.3714e-1 / 26.12 dB; rgba 3.0344e-2 / 36.30 dB | same |
| LLaDA-Image Turbo | `turbo_matches_upstream` | skipped: needs `MLXCEL_LLADA_IMAGE_ORACLE_DIR`, absent on this host | skipped |

Z-Image and Ming PNGs were byte-identical across all four runs. That is expected: the barrier does not change arithmetic, and these families' LayerNorm widths did not hit the race on main. For LLaDA-Image, an uncommitted harness on the overlay build ran Turbo text 1024 and VQ 512 twice in each of two processes with identical pixels every time; no before run was made.

**Gates.** The build was confirmed to copy the overlay (`grep -n "lablup/mlxcel"` on the fetched `layer_norm.cu` matches lines 7 and 50). `cargo fmt --check`, clippy with `-D warnings`, `make verify-env-var-docs`, `verify-kernel-dtype-keys` and `verify-cuda-runtime-dir` pass. With the overlay: mlxcel-core `layers::` 106/106, mlxcel `imagegen::dit_blocks` 18/18, tests named `layer_norm` 10/10 plus 2/2 in mlxcel-core. mlxcel `layers::` passes 75 of 77; the two `sam3d_objects` oracle failures reproduce on main's binary and are tracked in #2739.

## 5. Learning Points

- **A block-wide broadcast reduction needs a barrier on both sides of the shared read when its storage is reused.** The first barrier makes the writes visible; the second keeps a fast warp from overwriting a slot before every warp has read it. `cub::BlockReduce` documents the same rule: call `__syncthreads()` before reusing `TempStorage`.
- **Shared-memory races can hide behind the launch geometry.** Here the bug was present on every width but only visible when the last warp had partial work. A fully idle warp writes zero over zero, and equal work keeps warps in step. A determinism test has to pick shapes that create unequal per-warp work, and inputs that vary within a row.
- **Count differences exactly, not by hash.** A distinct-hash count is useful for reporting, but the assertion should compare bytes.
- **Fix the shared primitive, not each caller.** One overlay line fixed a dozen call sites, including families nobody had measured.

## 6. Limitations and Follow-ups

- **Separate upstream bug in `layer_norm_vjp`, not fixed here.** In the tail branch (rows whose width is not a multiple of `N_READS`), `wn[i]` and `gn[i]` index the `N_READS`-element vectors with the absolute column `i` instead of `i - index * N_READS`. That reads past the vectors and corrupts the gradient factors. mlxcel reaches the CUDA VJP only through `fast_layer_norm_vjp_for_test`, and the VJP case here is 384 wide, so it never takes that branch. It needs its own issue, a fix in the same overlay, and a test at a width such as 390.
- **Performance not measured.** The extra barrier adds one `block.sync()` per reduction (two per forward kernel, two per VJP kernel). No throughput run was made; LayerNorm is memory-bound and the cost is expected to be small, but that is an expectation, not a measurement.
- **Only sm_121 tested.** Warp scheduling differs across architectures, so which shapes reproduce on main may differ elsewhere. The fix itself is architecture-independent.
- **The VJP test checks determinism only,** not accuracy against a reference.
- **Overlay maintenance.** On every MLX pin bump the overlay must be re-diffed against upstream `layer_norm.cu`, or upstream changes to that file will be silently masked. It can be dropped once upstream adds the barrier.
- **Left out of scope by the issue:** switching Mage-Flow (#2688) back to the fused kernel, reporting the bug upstream, and auditing the per-tile `temp` reuse in `rms_norm_vjp_small`.
- **LLaDA-Image has no before/after parity numbers** because its oracle directory is not committed and was absent on the host.

## References

- Issue #2696 (closes), with the 2026-10-30 addendum on the shared DiT `FinalLayer`.
- PR #2688 (Mage-Flow port, where the race was found), epic #2348, PR #2710 (Ming-Image review that found the `FinalLayer` exposure).
- #2381 and `overlays/rocm/LOCAL_FIXES.md` item 51 (the same fix in the ROCm overlay); ml-explore/mlx#3630 (the same class of bug in the Metal VJP).
- #2739 (pre-existing `sam3d_objects` oracle failures).
- MLX pin `7ca40033132dae444d45f7fc3daae57e24c13f6c` (`src/lib/mlxcelverse/CMakeLists.txt`).
