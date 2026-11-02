# PR #2770: Index the MLX CUDA layer_norm VJP tail with the in-vector offset

**Date**: 2026-10-31 (report written pre-merge; the PR was open and in draft at the time)
**Status**: Open PR, validated on CUDA (GB10, sm_121) only. Other CUDA architectures were not run; Metal and ROCm do not compile this file
**Languages**: CUDA C++, Rust, Markdown
**Risk Level**: Low (two index expressions in a branch that only rows with a partial last vector take; the full-vector path is unchanged at the source level)

## Executive Summary

PR #2770 (closes #2759) fixes an out-of-bounds read in the tail branch of `layer_norm_vjp`, the CUDA kernel behind the input gradient of `fast::layer_norm`. In the second reduction, the tail loop indexed the per-thread `wn` and `gn` vectors (each `N_READS` elements long) with the absolute column `i` instead of the offset inside the vector. Every row whose width is not a multiple of `N_READS` read past those arrays. On GB10 the result depended on dtype: the bf16 VJP at width 390 aborted the process with an illegal memory access, and the f32 VJP returned a wrong `dx` in every row (2047 or 2048 of 2048 rows outside the bound, max abs error about 2.9e-2).

The fix changes `wn[i]` and `gn[i]` to `wn[i - index * N_READS]` and `gn[i - index * N_READS]` in the CUDA overlay `src/lib/mlxcelverse/overlays/cuda/mlx/backend/cuda/layer_norm.cu`, marked `[mlxcel #2759]`. A new test checks `dx` at width 390 against an f64 host reference for bf16 and f32, with and without weight, plus a 384 control. It fails on main (abort for bf16, 770k+ bad elements for f32) and passes 6/6 with the fix.

## 1. Problem Statement

### 1.1 Background

The bug was found during review of PR #2749 (closes #2696), which added this overlay to put a second barrier into `BlockBroadcastReduce::Reduce`. That PR deliberately kept the overlay a one-line diff from the MLX pin and recorded the second bug in section 6 of `TECHNICAL_REPORTS/2749-cuda-layer-norm-reduction-barrier-20261030.en.md`. The VJP test #2749 added (`bf16_vjp_input_gradient_is_deterministic`, 384 wide) never takes the tail branch, because 384 is a multiple of both 8 (bf16) and 4 (f32), and it checks determinism only, not accuracy.

Upstream `ml-explore/mlx` main at `1ba7f969` (2026-10-30) still has the identical code; the file last changed upstream in `6ec01922` (2026-02-26) and no upstream issue or PR covers it. A pin bump would not fix it.

### 1.2 Cause

`layer_norm_vjp` makes three passes over a row: a sum for the mean, a second reduction that accumulates three factors per element (`wg`, `wg * t`, `t * t` with `t = x - mean` and `wg = w * g`), and an output pass that writes `gx` and `gw`. Each thread handles `N_READS = 16 / sizeof(T)` consecutive elements (8 for bf16 and f16, 4 for f32). In the second pass:

```cpp
auto gn = load_vector<N_READS>(g, index, axis_size, T(0));
auto wn = load_vector<N_READS>(w, index, axis_size, w_stride, T(0));
if ((index + 1) * N_READS <= axis_size) {
  // full vector: wn[i], gn[i] with i in [0, N_READS), correct
} else {
  for (int i = index * N_READS; i < axis_size; ++i) {
    float t = static_cast<float>(x[i]) - mean;   // x read from global memory: correct
    float wi = wn[i];                             // i is the column, wn has N_READS slots
    float gi = gn[i];
```

At width 390 the tail thread is `index` 48 for bf16 (columns 384 to 389) and `index` 97 for f32 (columns 388 to 389). It reads `wn[384..389]` or `wn[388..389]` from an 8- or 4-element array. Because the index is a runtime value, the compiler keeps `wn` and `gn` in per-thread local memory, so the read lands somewhere past the thread's frame.

Only the first two factors are corrupted. `t * t` reads `x[i]` from global memory, so the variance and therefore `normalizer` are correct. The corrupted sums feed `meanwg` and `meanwgxc`, which scale every element of the row's `dx` in the output pass. That is why one tail thread spoils an entire row. The first pass and the output pass use the bounds-checked `load_vector` with zero fill and are correct, so `gw` and the bias gradient are not affected.

### 1.3 Why the symptom differs by dtype

On main, bf16 at width 390 aborted with `cudaGetLastError() failed: an illegal memory access was encountered` (exit 134), while f32 ran to completion with silently wrong values. The PR does not explain the difference, and this report does not either: the out-of-range offset is larger for f32 (column 388 of a 4-byte type) than for bf16 (column 384 of a 2-byte type), so "further past the array faults" does not fit. The outcome depends on how local memory is laid out for each instantiation and what lies past it. It is undefined behavior, and the takeaway is that the same bug can be a crash on one build and a wrong number on another.

### 1.4 Impact and risk of not fixing

mlxcel is inference-only and reaches the CUDA `LayerNormVJP` primitive only through the test bridge `mlxcel_core::fast_layer_norm_vjp_for_test` (`src/lib/mlxcel-core/cpp/mlx_cxx_bridge.cpp`). No production code path is affected today.

| Risk | Impact | Likelihood |
|---|---|---|
| A future training, fine-tuning or gradient-based path gets wrong input gradients for LayerNorm widths that are not a multiple of 4 or 8 | High (silent, every row) | Certain on those widths once such a path exists |
| Process abort from the out-of-bounds read in bf16 | High | Observed on GB10 at width 390 |
| Undefined behavior in a kernel the project ships as an overlay | Medium | Present on every build |

## 2. Change Summary

One commit, three files (+136 / -4).

| File | Change |
|---|---|
| `src/lib/mlxcelverse/overlays/cuda/mlx/backend/cuda/layer_norm.cu` | Tail branch of the second reduction indexes `wn` and `gn` with `i - index * N_READS`; a `[mlxcel #2759]` comment at the change and one sentence added to the file header's provenance note. |
| `src/lib/mlxcelverse/overlays/cuda/README.md` | The `layer_norm.cu` entry now records the tail-index fix next to the #2696 barrier. The overlay file count stays at 30, since no file was added. |
| `tests/cuda_layer_norm_determinism.rs` | Module doc paragraph for #2759; new `vjp_dx_reference` (f64 host reference), `vjp_tolerance`, `check_vjp_dx` and the test `vjp_input_gradient_matches_reference_on_tail_width`. |

| Hash | Type | Message |
|---|---|---|
| `ae870767` | fix | index the layer_norm VJP tail with the in-vector offset |

No CMake change is needed: `mlx_apply_source_overlays` copies the overlay into the fetched MLX tree on every build, and the PR confirmed the `[mlxcel #2759]` marker at line 191 of the built `layer_norm.cu` in both the test-fast and clippy build directories.

## 3. Technical Decisions

### 3.1 Fix the index, keep the vector loads

| Option | Pros | Cons |
|---|---|---|
| Read `w[i * w_stride]` and `g[i]` from global memory in the tail (rejected in the issue) | Obviously correct; matches how the ROCm overlay indexes | Diverges further from upstream; makes the overlay harder to re-diff on pin bumps |
| Drop the tail branch and run the full-vector loop with a mask on columns past `axis_size` | One code path | `t = 0 - mean` for zero-filled slots makes `t * t` nonzero, so the mask must cover every factor; a structural change to a kernel the project does not own |
| **Chosen: `wn[i - index * N_READS]`, `gn[i - index * N_READS]`** | Two expressions; the overlay stays a small, auditable diff from the pin; the values come from the same `load_vector` reads as the full-vector path | The tail still uses a runtime index into a per-thread array, so `wn` and `gn` stay in local memory for that branch, as before |

The offset form is also correct for multi-round rows (`axis_size > BLOCK_DIM * N_READS`), because `index` already includes the round: `index = r * BLOCK_DIM + thread_rank`.

### 3.2 Keep the full-vector path untouched

The issue required byte-identical output for widths that are a multiple of `N_READS`. The source of the full-vector branch is unchanged. The PR adds two kinds of supporting evidence: a cuobjdump SASS comparison of all 48 `layer_norm_vjp` instantiations shows identical FADD, FMUL, FFMA, MUFU, BAR, SHFL, LDS and STS counts between the main and fixed builds, and the 384 controls report the same max errors against the reference before and after. Neither is a byte-level proof (instruction counts can match while order or registers differ), and the PR says so.

### 3.3 Test design

- **Reference from the rounded inputs, in f64.** `vjp_dx_reference` reads back the bf16 or f32 inputs the kernel actually saw and computes, per row, `xhat = (x - mu) * rstd`, `wg = w * g` and `dx = rstd * (wg - mean(wg) - xhat * mean(wg * xhat))`. The issue proposed an f32 reference; f64 removes the reference's own rounding from the comparison.
- **Bounds.** f32: `1e-4 + 1e-4 * |ref|`. bf16: the existing forward `tolerance(BFLOAT16, ref)`, which is one bf16 ulp of the reference plus 1e-2. The f32 bound has a wide margin on both sides: the 384 control's max abs error is 4.8e-7 and the broken 390 kernel's is 2.9e-2. The bf16 margin is narrower: the control's max abs error is 7.8e-3 against a bound of at least 1e-2. bf16 needed no margin on the failing side, because main aborted before producing any output.
- **Both weight modes.** The no-weight call passes a scalar weight with `w_stride == 0`, which also goes through `load_vector` and the tail branch. Both affine and no-weight are covered for both dtypes.
- **Determinism as well.** Each case also runs through the existing `repeat` check (32 calls, exact byte comparison), so the 390 cases are held to the same standard as the #2696 cases.
- **Control first.** The test iterates `dim in [384, 390]`, so all four 384 controls print their results before a 390 case can abort the process on main. A non-finite output counts as infinite error, so a NaN cannot pass the bound.
- **One test, collected failures.** All eight cases run and their failure strings are joined into one assertion, so a run reports every failing case, not just the first (on main the abort still ends the run at the first bf16 390 case).

## 4. Validation

All on GB10 (sm_121), `--test-threads=1`, under `gpu-lock`.

**Failing first.** The committed test was built against main's `layer_norm.cu` (overlay reverted locally, MLX rebuilt, built source confirmed without the marker). All four 384 controls passed, then the first bf16 390 case aborted with an illegal memory access (exit 134). Each 390 case run alone on main's kernel (an earlier revision of the test with the same pass condition and a temporary case filter, neither committed):

| Case (390 wide, 2048 rows) | main | fixed |
|---|---|---|
| bf16 affine | illegal memory access, exit 134 | 0 bad elements, max abs error 7.812e-3 |
| bf16 no-weight | illegal memory access, exit 134 | 0 bad elements, max abs error 6.521e-3 |
| f32 affine | 770741 elements in 2047 rows out of bound, max abs error 2.943e-2 | 0 bad elements, max abs error 4.768e-7 |
| f32 no-weight | 771244 elements in 2048 rows out of bound, max abs error 2.965e-2 | 0 bad elements, max abs error 3.576e-7 |

Note that the broken f32 kernel was deterministic (one distinct output over 32 calls): a determinism-only test would not have caught it.

**With the fix.** The whole `cuda_layer_norm_determinism` target passes 6/6 in 6.59 s: the new test, `bf16_vjp_input_gradient_is_deterministic` and the four forward cases from #2749. Every 384 and 390 case gives one distinct output over 32 calls. `cargo fmt --all -- --check` and `cargo clippy --profile test-fast --features cuda -p mlxcel --tests -- -D warnings` pass.

## 5. Learning Points

- **Vectorized kernels have two indexing spaces.** In code that loads `N_READS` elements per thread into a local array, the loop over the full vector uses the slot index and the tail loop usually uses the column index. Any array access inside the tail has to be translated back (`column - index * N_READS`). Reviewing a tail branch means checking every subscript against which space it is in.
- **Test widths that are not a multiple of the vector width.** 384, 4096 and 3840 are all multiples of 8, so every shape in the earlier tests skipped the tail branch. A width like 390 (multiple of neither 4 nor 8) exercises the tail for both bf16 and f32 at once.
- **Determinism does not imply correctness.** The f32 kernel produced the same wrong bytes on every call. A test for a numerical kernel needs a reference, not only a repeat check.
- **Undefined behavior can look like two different bugs.** The same out-of-bounds read crashed one instantiation and silently corrupted another. When a failure differs by dtype, check for UB before assuming two causes.
- **Order cases so a crash still leaves evidence.** Running the controls before the case that can abort the process means a failing-first run on main still prints the passing control numbers that show the reference and bounds are sound.

## 6. Limitations and Follow-ups

- **Only sm_121 tested.** The out-of-bounds read is architecture-independent, and so is the fix, but whether main crashes or returns wrong values may differ on other GPUs.
- **`gw` and the bias gradient are not checked against a reference.** The issue and the kernel structure say they are unaffected (the output pass uses bounds-checked loads), but the new test checks `dx` only.
- **Full-vector byte identity is supported, not proven.** See 3.2.
- **The bf16 accuracy bound was never seen to reject a wrong tail.** On main the bf16 390 cases crashed before producing output, so only the f32 cases show the reference check catching this bug. The bf16 bound is the one already validated by #2749 for the forward kernel.
- **Why bf16 faults and f32 does not is unexplained** (see 1.3).
- **Upstream is still affected.** Reporting the bug to `ml-explore/mlx` was left out of scope by the issue. Until upstream fixes it, every MLX pin bump has to re-apply this change along with the #2696 barrier when re-diffing the overlay.
- **Out of scope by the issue:** the ROCm kernel (`overlays/rocm/mlx/backend/rocm/layer_norm.hip` indexes `w[idx * w_stride]` and `g[idx]` by column and is not affected), the Metal kernel, and `rms_norm_vjp_small`.

## References

- Issue #2759 (closes).
- PR #2749 and issue #2696 (the overlay this fix extends; section 6 of its report records this bug).
- `ml-explore/mlx` main `1ba7f969`, `mlx/backend/cuda/layer_norm.cu` (same code, unfixed).
- `src/lib/mlxcelverse/overlays/cuda/README.md` (`layer_norm.cu` entry).
