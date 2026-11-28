# Technical Report: PR #2046 - Remove the ROCm get_launch_args Helper That Capped the Grid Silently

**Date**: 2026-09-30

**Status**: Implemented and validated on the gfx1151 host; pending merge.

**Languages**: C++ (HIP header), Markdown

**Risk Level**: Low (removes a function with no caller and adds a deleted declaration; overlay-only, no kernel's launch geometry changes, Metal and CUDA never copy `patches-rocm/`)

## Executive Summary

Issue #1874 (part of epic #1801) found that the ROCm overlay's `get_launch_args` in `mlx/backend/rocm/kernel_utils.hpp` capped its grid at 65535 blocks of 256 threads and returned it with nothing saying the kernel had to be grid-stride. Any kernel that assigns one index per thread and launches with it would leave every element past 65535 x 256 unwritten, and the output would keep whatever the allocation held. Nothing called the helper, so nothing was broken yet. The hazard was that upstream CUDA has a helper with the same name that does not cap x and is called throughout `mlx/backend/cuda`, so a kernel ported from there would compile against the fork's version and inherit the truncation without any warning.

The PR deletes both overloads and puts a deleted variadic template in their place, `template <typename... Args> void get_launch_args(Args&&...) = delete;`, with a comment explaining why. Any call, in either upstream's or the fork's signature, now fails to compile at that comment. The decision is recorded as `LOCAL_FIXES.md` item 19, an upstreaming candidate for #1813. Review found one live instance of the same defect outside the helper: `SliceUpdate::eval_gpu`'s reduce-op path in `indexing.hip`. It is documented, not fixed, because this PR changes no kernel geometry.

## 1. Problem Statement

The fork's helper computed `num_blocks = ceil(ceil(size / work_per_thread) / 256)` and then applied `num_blocks = std::min(num_blocks, 65535)`. It also ignored `shape`, `strides` and `large`, and fixed the block size at 256. A clamped grid is only correct for a kernel that loops over `gridDim.x * blockDim.x`. The helper's name, signature and comment gave no hint of that contract.

The kernels in this backend do not use it. `binary.hip` and `unary.hip` write their own grid-stride loops. The two kernels added in #1856, `hadamard.hip` and `sort.hip`, compute their own clamped geometry and carry comments tying the clamp to the loop. The issue's concern was the next person to reach for the obvious-looking helper, especially when porting a CUDA kernel that already calls a function of the same name.

Upstream's definition (`mlx/backend/cuda/kernel_utils.cu:33-50` at the pin) is a different function. It does not clamp x (CUDA's x limit is 2^31 - 1), takes a `max_block_dim`, and honours `large` through `get_2d_grid_dims`. The constant 65535 appears upstream only as `max_grid_yz_dim` in `get_launch_args_general`, which spills y overflow into z instead of dropping it. So the fork's helper looked like upstream's and behaved differently in exactly the way that produces silent wrong output. That is the failure shape of #1823 (recorded as `LOCAL_FIXES.md` item 13), where `quantized_matmul` returned the contents of a fresh allocation.

## 2. Change Summary

- `src/lib/mlx-cpp/patches-rocm/mlx/backend/rocm/kernel_utils.hpp` (13 lines in, 19 out): both `get_launch_args` overloads (sized and `const array&`) are removed. In their place is `template <typename... Args> void get_launch_args(Args&&...) = delete;` and a comment that points to `LOCAL_FIXES.md` item 19, says why the old helper was unsafe and why a ported call would inherit the problem, and says that a grid clamped at a launch site is only correct if that kernel is grid-stride (pointing to `hadamard.hip` and `sort.hip`).
- `src/lib/mlx-cpp/patches-rocm/LOCAL_FIXES.md`: new item 19 records the defect, the three options from the issue and why deletion was chosen, the `indexing.hip` instance found in review, and that a fork sync bringing the definitions back must drop them again. It ends with the standard "Applies to the fork; to be proposed there" marker.

The second commit (`979f114c`) only corrects documentation. The first draft said every 65535-block clamp in the backend launched a grid-stride kernel. Review showed `indexing.hip` does not, so the header comment and item 19 were reworded, and the `get_launch_args_general` comparison was changed to say upstream spills y into z.

## 3. Technical Decisions

### Delete rather than rename or extend

The issue offered three options: delete both overloads, rename them to state the grid-stride requirement (for example `get_grid_stride_launch_args`), or spill the overflow into another grid dimension the way upstream's `get_launch_args_general` does. The PR takes deletion. With no caller it cannot regress anything, and the other two options keep a helper that only resembles upstream's. A rename still requires every caller to read and obey a contract. Spilling into y does not help a one-index-per-thread kernel unless that kernel also reads `blockIdx.y`, so it would move the bug rather than remove it. A faithful port of upstream's geometry is not a drop-in either, because AMD caps each grid dimension at 2^32 - 1 threads (item 11), which differs from CUDA's limits.

### A deleted declaration instead of no declaration

Removing the name entirely would make a ported call fail as "use of undeclared identifier". Someone might fix that by restoring the old helper from history, which brings back the bug. A deleted variadic template matches any argument list, so both the fork's and upstream's call forms fail with "call to deleted function", and the compiler points at the declaration where the comment explains the reason. This turns the recorded decision into something the compiler enforces at the moment someone would need it.

### Document the live instance, do not fix it here

`SliceUpdate::eval_gpu`'s reduce-op path (Sum, Prod, Max, Min) computes `num_blocks = min(ceil(ceil(update_size / nwork) / 256), 65535)` and launches `slice_update_op_kernel`. That kernel computes one starting index per thread (`(blockIdx.x * blockDim.x + threadIdx.x) * NWORK`), processes at most `NWORK` elements, and never reads `gridDim`. `nwork` is 4, 2 or 1 depending on whether the innermost collapsed dimension divides by 4 or 2, so an update larger than 16,776,960 x `NWORK` elements (about 16.8M, 33.6M or 67.1M elements) leaves its tail unapplied. This is the defect #1874 describes, at a real call site. The issue explicitly scoped kernel geometry out, so the PR records the site in item 19 and in its body and leaves it for a separate issue. That keeps this change a pure no-op at runtime.

### Record it as a fork-side upstreaming candidate

The defect exists in the fork (checked at `75915908`), not in upstream MLX, so item 19 is listed for #1813's fork upstreaming list rather than filed as an upstream MLX report. The note that a fork sync reintroducing the definitions must drop them again protects the fix against the overlay being refreshed.

## 4. Validation

Author's runs, from the PR body (gfx1151, Radeon 8060S):

- `cargo build --release --features rocm`: passes, cold and again incrementally after the second commit. The build tree's copy of the header matches the overlay.
- Compile-fail probe with `hipcc -fsyntax-only --offload-arch=gfx1151` on a translation unit calling `get_launch_args(arr, false)` and `get_launch_args(n, shape, strides, false, 4)`: both rejected as "call to deleted function" with the patched header. The same file compiles with the header from `origin/main`, so the probe does distinguish the two.
- `grep -rn get_launch_args src/lib/mlx-cpp/patches-rocm/`: only the deleted declaration and its comment remain.
- `make verify-rocm-smoke` with Qwen3-0.6B-4bit: OK.
- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt`: pass.

Orchestrator verification (gfx1151, origin/main `fcf5f4c3` plus this branch):

- `make verify-rocm` ran every step. Versions, kernel dtype keys, kernel port dispatch, llama-compat, fmt, workspace clippy with `--features rocm`, and the ROCm smoke (32 tokens) passed.
- `verify-test-rocm` failed in three targets, with exactly the 37 known baseline failures and nothing else:
  - `-p mlxcel-core --lib`: 35 failures. 34 are fused paged-attention tests with no ROCm port (tracked by #1814); the other is the bf16 `prefill_dense_gemm_matches_qmm_bytes_where_eligible`.
  - `-p mlxcel --lib`: `gemma3_backbone_tests::gelu_approx_matches_mlx_nn_bit_for_bit`, from #2037.
  - `-p mlxcel --bin mlxcel`: `tests::family_order_is_exhaustive`, from #2037.

The acceptance criteria asked for an unchanged baseline. The baseline stated in the issue (one failing target, the NVFP4 abort under #1806) is out of date. #1806 has since been fixed, and the current set is the 37 failures above, which match the #2033 run. Because the change removes code nobody called, no runtime difference was expected, and none was seen.

## 5. Learning Points

- **A helper with an upstream name carries upstream's contract in readers' heads.** `get_launch_args` in the fork shared a name with CUDA's helper but not its behavior. Anyone porting from `mlx/backend/cuda` would trust the name. When a local function cannot keep the upstream contract, it should not keep the upstream name.
- **`= delete` is a way to write down a decision.** A deleted function with a comment keeps the name reserved, rejects every call form, and puts the reason in the compiler error. It is stronger than deleting the code and hoping no one restores it.
- **A clamp and its grid-stride loop are one unit.** Clamping the grid is correct only when the kernel reads `gridDim`. `hadamard.hip` and `sort.hip` keep both side by side with a comment; `indexing.hip`'s `SliceUpdate` shows what happens when only the clamp is present.
- **Review a sweeping claim against every site.** The first draft of item 19 said every clamp in the backend paired with a grid-stride kernel. One grep for `65535` and a read of each launched kernel disproved it. The claim was corrected, and a real bug was found as a result.

## 6. What Is Not Verified

- **The #1826 gfx1151 correctness matrix was not run.** The issue's verification asked for it. The PR removes code with no caller and changes no kernel, so no difference is expected, but the matrix result itself was not recorded.
- **Metal and CUDA were not run.** Neither build copies `patches-rocm/`, so neither is touched by construction.
- **The `indexing.hip` truncation was not reproduced.** The threshold (more than 16.8M x `NWORK` elements in a reduce-type `SliceUpdate`) comes from reading the launch code and the kernel. No test drives an update that large, and whether any mlxcel model path reaches it was not checked.

## 7. Remaining Work

- Open a separate issue for `SliceUpdate::eval_gpu`'s reduce-op path in `indexing.hip`: make `slice_update_op_kernel` grid-stride, or remove the clamp and size the grid within AMD's per-dimension limit, and add a test above the threshold.
- Upstream item 19 to the ROCm fork as part of #1813.
- #1814: ROCm ports of the fused paged-attention kernels (34 of the 35 `mlxcel-core` failures).
- Triage the bf16 `prefill_dense_gemm_matches_qmm_bytes_where_eligible` failure on ROCm, and fix the two #2037 failures (`gelu_approx_matches_mlx_nn_bit_for_bit`, `family_order_is_exhaustive`).
