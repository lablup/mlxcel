# Technical Report: PR #2112 - Fix and Enable the Expert-Batched gather_qmm Prefill Kernel

**Date**: 2026-10-05

**Status**: Implemented and measured on the gfx1151 host; code head `c587178e` rebased onto origin/main `df05f1f9` (which contains #2107, `022067ad`), pending merge.

**Languages**: HIP C++ (the ROCm overlay's `quantized/qmm.hip`), Rust (a new ROCm integration test, one doc comment), Markdown (results page, `LOCAL_FIXES.md` item 9, the upstream packaging table)

**Risk Level**: Medium (a ROCm default changes: sorted affine bf16 and f16 MoE prefill with at most 64 experts now runs a rewritten kernel; Metal and CUDA builds never copy `patches-rocm/` and are untouched)

## Executive Summary

Issue #2066 (part of #1814, epic #1801) asked why the ROCm overlay's expert-batched `gather_qmm` kernel returned wrong bf16 results (`LOCAL_FIXES.md` item 9, which had made it opt-in) and whether it could speed up MoE prefill on gfx1151, where Mixtral-8x7B-4bit prefilled at about 26 tok/s.

The defect was in the index reads, not the arithmetic. The kernel read `lhs_indices[b]` and `rhs_indices[b]` as flat `[B]` arrays, while MLX broadcasts the two index arrays against the activation's batch shape without copying them. For a broadcast call the kernel read past the end of both arrays. bf16 was the only failing dtype because bf16 was the only dtype the gate sent to the kernel. A new test reproduced it at a relative L2 error of 1.415 against a dequantized f32 reference, where the unsorted path had 2.3e-3.

The PR reads both index arrays through the batch shape and strides, rewrites the inner loop (the index fix alone left the kernel 2.3x slower than the per-row wide kernel it replaces), adds f16, and turns the kernel on by default under the issue's rule. On gfx1151, 512-token prefill went from 535.7 to 918.6 tok/s on granite-4.0-h-tiny-4bit (1.71x) and from 25.95 to 125.6 tok/s on Mixtral-8x7B-Instruct-v0.1-4bit (4.84x), with decode unchanged. All 48 test cases hold the expert-batched error within 0.1% of the unsorted path's, and logit traces have 0 decided mismatches against the #1809 Metal references.

## 1. Problem Statement

### 1.1 The kernel and why it was off

`GatherQMM::eval_gpu` routes a sorted, transposed, affine, group-size-64, 4- or 8-bit call with `M == 1`, `B >= 64`, `E <= 64` and `B / E >= 4` (the prefill of an MoE with at most 64 experts) to `gather_qmv_expert_batched_kernel`. That kernel reads each expert's weights once for all of that expert's rows, where the per-row gather kernels reread them for every routed row. Item 9 had recorded a relative error above 1.0 against the unsorted path for bf16, with f32 and f16 correct, and made the kernel opt-in (`MLX_ROCM_GATHER_QMV_EXPERT_BATCHED=1`) without a root cause.

### 1.2 Where prefill time went

The issue required a profile first, because as gated the kernel could reach neither slow checkpoint from the baseline. `rocprofv3 --kernel-trace --stats` around a 512-token prefill, cut between the bench's phase marks:

| Model | Activations, experts, top-k | Rows per call (B) | Prefill, profiled | Dominant kernel | Share |
|---|---|---|---|---|---|
| Mixtral-8x7B-Instruct-v0.1-4bit | f16, 8, 2 | 1024 | 20.3 s | `gather_qmv_warp_shared_kernel<__half, ...>` (210 ms per call) | 99.1% |
| gpt-oss-20b-MXFP4-Q4 | bf16, 32, 4 (mxfp4) | 2048 | 67.8 s | `gather_qmv_kernel<hip_bfloat16, unsigned char, 4, 32, false>` (940 ms per call) | 99.8% |
| granite-4.0-h-tiny-4bit | bf16, 64, 6 | 3072 | 0.96 s | `gather_qmv_wide_kernel<hip_bfloat16, ...>` (4.9 ms per call) | 61.8% |

All three are per-row gather kernels. granite was the only checkpoint on the host the bf16-only gate reached. Mixtral's f16 prefill ran in the kind of per-row kernel the expert-batched kernel replaces, which is what the issue's step 3 asked to establish before adding f16. gpt-oss runs the non-affine fallback from item 10, which no affine kernel can take.

## 2. Change Summary

| Area | Change |
|---|---|
| `patches-rocm/.../quantized/qmm.hip` | `gather_index_loc` and `find_sorted_expert_run` helpers; strided index reads in both expert-batched kernels; inner loop rewrite; grid z is the expert; f16 dispatch; default on |
| `tests/rocm_gather_qmm_expert_batched.rs` (new) | 48 cases against the unsorted path and a dequantized f32 reference, gated to ROCm |
| `patches-rocm/LOCAL_FIXES.md` | Item 9 rewritten with the root cause, the rewrite, the measurements and the default; marked an upstreaming candidate for #1813 |
| `docs/mlxcelverse/upstream/README.md` | Item 9 moves from "Held" to "Not packaged yet", with what a fork package needs |
| `docs/benchmark_results/rocm-moe-prefill-gfx1151-2026-10-05.md` (new) | Profile, root cause, correctness, prefill and decode numbers, short-prompt sweep, decision |
| `src/models/switch_layers_mxfp_tests.rs` | Doc comment: the kernel is on by default and has its own test |

Three commits: the fix and enablement (`fc868bdf`), a review follow-up that tightened the test bound and doc wording (`87d5f6d3`), and the short-prompt sweep (`c587178e`). The diff against origin/main is 6 files, 762 insertions and 246 deletions.

## 3. Root Cause of Item 9

### 3.1 Flat reads of broadcast indices

The kernel took index `b` of the flattened batch and read `lhs_indices[b]` and `rhs_indices[b]`. That is correct only when both index arrays are contiguous `[B]` arrays. MLX broadcasts the index arrays against the activation's batch shape without materializing them, so an index array can have stride 0 along a broadcast axis. The call that found item 9 was `x` `[T, 1, K]` with sorted `rhs_indices` `[T, 1]`: the batch broadcasts to `[T, T]`, the rhs strides are `(1, 0)`, the implicit lhs strides are `(0, 1)`, and `B = T * T`. A flat read at `b` up to `T * T - 1` runs past both `T`-element arrays.

### 3.2 Why only bf16

f32 and f16 never reached the kernel: the gate required `x.dtype() == bfloat16`. "f32 and f16 are correct" in the old item 9 described the other kernels, not this one.

### 3.3 Why models were not affected

`SwitchGLU`'s sorted path gathers `x` by expert and passes flat `[B]` indices, which the flat reads handle. The results page records that forcing the old kernel on for granite's `w256` trace gave 0 decided mismatches against Metal. The defect was reachable through direct `gather_qmm` calls with broadcast indices, and through any future caller that builds them.

### 3.4 The reproducing test

`tests/rocm_gather_qmm_expert_batched.rs` covers four index layouts: `Gathered` (the `SwitchGLU` shape), `Shared` (one activation row, stride 0 on `x`), `BroadcastRows` (`x` `[T, 1, 1, K]` with `rhs` `[T, top_k]`) and `BroadcastIndices` (`x` `[T, 1, K]` with `rhs` `[T, 1]`, the original call). With the old kernel, `BroadcastRows` had a relative L2 error of 1.415 against the dequantized f32 reference, where the unsorted path had 2.3e-3. With the strided reads reverted in the new kernel, the test fails again, so it pins the fix and not only the rewrite.

## 4. The Fix

### 4.1 Strided index reads

`gather_index_loc(b, batch_shape, strides, batch_ndim)` maps flat batch element `b` to its location in an index array: `b * strides[0]` for a one-dimensional batch, `elem_to_loc` otherwise. This is how the launched per-row gather kernels in `qmm.hip` already read their indices. The launcher passes the collapsed batch shape and the lhs and rhs strides it already computed for those kernels. Both the activation-row lookup and the run search use it.

`find_sorted_expert_run` finds the run of batch elements routed to one expert with two binary searches (lower and upper bound) over the strided rhs array, relying on `right_sorted` meaning sorted in the flat order of the broadcast batch.

The never-launched `gather_qmv_idot_expert_batched_kernel` shared the defect and takes the same reads and run search, so it cannot regain the bug if it is wired up later.

### 4.2 Why the index fix alone was not enough

With only the strided reads, the kernel took 1.33 s of granite's prefill against 0.58 s for the wide kernel it replaces (11.1 against 4.9 ms per call), 2.3x slower. Enabling a correct but slower kernel would have failed the issue's rule, which requires prefill to improve on every eligible model. Four causes were in the old inner loop:

- **Weights were reread per row.** The loop over the run's rows was outermost, so each row reloaded every packed word and every group's scale and bias. Sharing an expert's weights across its rows, the kernel's purpose, was left to the caches.
- **A reduction per group.** The 16 lanes of a column reduced two partial sums across lanes after every 64-element group.
- **Idle lanes at 4 bits.** Each lane took 8 values per step with a stride of `16 * 8 = 128`, but a group holds 64 values, so only lanes 0 to 7 had work.
- **Serial run finding.** Grid z was `min(B, E)` run slots, and block z found the z-th run by walking every earlier run boundary, so later blocks did more serial work.

### 4.3 The rewrite

- Each lane loads one packed 32-bit word with its group's scale and bias and applies them to `TOKENS = 4` rows of the run before the next load, so the expert's weights are read once per four rows instead of once per row.
- Each row accumulates `scale * qx + bias * sum(x)` per word in registers, and the lanes reduce once per row at the end instead of once per group.
- The K loop is over words with a stride of `16 * VALS`, so all 16 lanes of a column work at 4 and 8 bits.
- Grid z is `E`; block z is expert z and finds its run with the two binary searches. A block with an empty run returns after the search.
- Rows past the end of a run reuse the run's first row so the unrolled loop has no branch, and their sums are not stored.
- The run search now happens before any bounds exit, so every thread in the block reaches the `__syncthreads()` (the old kernel returned on `row >= M || col >= N` before the barrier).
- The launcher fixes the block's x dimension at 16 to match the instantiated `THREADS_PER_COL`, where it previously came from `select_qmv_threads_per_col`.
- The kernel is restricted to affine 4- and 8-bit by `static_assert`, matching the gate, and drops the generic-bits and non-affine branches it could never take.

In the same profile, the rewritten kernel takes 0.32 s of granite's prefill (2.7 ms per call) and 4.52 s of Mixtral's (47 ms per call, against 210 ms for the warp-shared kernel).

### 4.4 f16

The kernel is templated on the element type. On the strength of the profile (Mixtral at 99.1% of prefill in one f16 per-row kernel), the gate now accepts `float16` as well as `bfloat16`, and the launcher dispatches `__half` or `hip_bfloat16`, as item 12 did for the warp-shared kernel. The issue capped new instantiations at one dtype because per-dtype instantiation once took `indexing.hip` from 26 s to 348 s (item 17). f16 adds two instantiations (4 and 8 bits). `qmm.hip` compiled with the build's own `hipcc` command took 37.5 and 37.7 s against 38.4 and 38.0 s on main, so compile time is unchanged; the simpler kernel offsets the extra instantiations.

### 4.5 The default

The issue set the rule: enable by default only if every eligible dtype matches the unsorted path within that path's own error against a dequantized f32 reference, and 512-token prefill improves on every eligible model measured with no decode change. Both conditions hold for bf16 and f16 on granite and Mixtral (sections 5 and 6), so `parse_warp_kernel_env("MLX_ROCM_GATHER_QMV_EXPERT_BATCHED", ...)` now defaults to true. `MLX_ROCM_GATHER_QMV_EXPERT_BATCHED=0` turns it off, and the variable is still read on every call.

## 5. Correctness Evidence

### 5.1 The 48-case test

`cargo test --release --features rocm --test rocm_gather_qmm_expert_batched -- --test-threads=1` forces the kernel on and runs 3 shapes (a Mixtral-like layer with 8 experts, top 2, K 4096 and N 516 so the last column block is partial; granite's gate/up and down projections with 64 experts, top 6) by 4 index layouts by 4 and 8 bits by bf16 and f16, 48 cases. Each case checks:

- the unsorted and expert-batched relative L2 errors against per-expert dense f32 matmuls of the dequantized weights both stay under a dtype rounding bound (2e-3 for f16, 1e-2 for bf16);
- the expert-batched error is at most 1.25 times the unsorted path's;
- two sorted runs are bit-identical.

Measured, the expert-batched error is within 0.1% of the unsorted path's in every case (2.8e-4 to 2.9e-4 in f16, 2.2e-3 to 2.3e-3 in bf16; largest ratio 1.0007). The 1.25 factor is slack for summation order, not the observed gap.

### 5.2 Logit traces against Metal

Teacher-forced traces with the kernel on, compared by `compare_logit_traces.py --decided 2.0` against the Metal references the #1809 matrix uses:

| Model, window | Metal reference | Top-1 disagreement | Decided mismatches |
|---|---|---|---|
| qwen3-30b-a3b, w8 | `metal_m1u_bec64748` | 17 / 640 | 0 / 319 |
| qwen3-30b-a3b, w256 | `metal_m1u_bec64748` | 23 / 512 | 0 / 248 |
| mixtral-8x7b-instruct, w8 | `metal_m1u_bec64748` | 1 / 640 | 0 / 345 |
| mixtral-8x7b-instruct, w256 | `metal_m1u_bec64748` | 2 / 512 | 0 / 245 |
| granite-4.0-h-tiny, w8 | `metal_m5_d1128266` | 20 / 640 | 0 / 260 |
| granite-4.0-h-tiny, w256 | `metal_m5_d1128266` | 23 / 512 | 0 / 208 |

Zero decided mismatches in all six rows. Qwen3-30B-A3B has 128 experts and never reaches the kernel, so its rows are a control. granite against the earlier ROCm per-row traces (`rocm_gfx1151_c5fe9a16`) also has 0 decided mismatches at both widths. The traces are not committed.

### 5.3 Gates

From the PR: `make verify-rocm` (with `MLXCEL_ROCM_SMOKE_MODEL=models/mlx/Qwen3-0.6B-4bit`) passed with 11,870 passed, 0 failed, 378 ignored; clippy on the new test with `-D warnings`, `make verify-rocm-overlay` and `cargo test --test dead_doc_pointers` passed. Implementation and security reviews found no CRITICAL or HIGH issues.

The orchestrator rebased the branch onto origin/main without conflicts (the branch now sits on `df05f1f9`, which includes #2107 at `022067ad`) and re-ran the full `make verify-rocm` gate on the rebased head.

## 6. Performance

### 6.1 512-token prefill and decode

`mlxcel-bench-decode` at `bench_decode.sh`'s shape (512-token prompt, 128 generated tokens), one binary, kernel off and on alternated run by run, three rounds, every run under `scripts/rocm_gpu_guard.sh --idle-secs 45`:

| Model | Prefill tok/s before (median) | after (median) | Change | Decode tok/s before / after |
|---|---|---|---|---|
| granite-4.0-h-tiny-4bit (bf16, 64 experts) | 535.73 | 918.59 | 1.71x | 88.76 / 88.65 |
| Mixtral-8x7B-Instruct-v0.1-4bit (f16, 8 experts) | 25.95 | 125.63 | 4.84x | 9.40 / 10.43 |

Every after run is faster than every before run on both models. Decode never reaches the kernel (a decode step has `B = top_k`, below the gate's 64). granite's decode medians are 0.1% apart. Mixtral's decode spread (8.68 to 10.71 tok/s across all six runs) is the noise of a 26 GB checkpoint on a 31 GiB host, so its +1 tok/s is not read as a change.

### 6.2 Short prompts at the gate's edges

Prefill time in ms, off against on, at the lower limits of the gate (`B / E >= 4` for granite, `B >= 64` for Mixtral):

| Model | T (B) | Off | On |
|---|---|---|---|
| granite-4.0-h-tiny-4bit | 44 (264) | 102.4 / 102.9 / 121.2 | 82.7 / 84.2 / 84.9 |
| granite-4.0-h-tiny-4bit | 64 (384) | 126.6 / 127.1 / 129.7 | 95.9 / 96.7 / 97.5 |
| granite-4.0-h-tiny-4bit | 160 (960) | 236.7 / 250.6 / 296.5 | 148.8 / 162.5 / 170.1 |
| Mixtral-8x7B-Instruct-v0.1-4bit | 32 (64) | 1709.7 / 1716.2 | 616.9 / 669.3 |
| Mixtral-8x7B-Instruct-v0.1-4bit | 64 (128) | 2845.0 / 2888.2 | 836.5 / 855.2 |

The kernel is faster at every size, so the gate's existing thresholds do not need to move for the default to hold below 512 tokens.

## 7. Technical Decisions

- **Fix the reads, not the gate.** The issue ruled out narrowing the gate to hide the defect. Reading through strides makes the kernel correct for every index layout MLX can hand it, which also covers `f16` once the gate admits it.
- **Use the per-row kernels' indexing.** `gather_index_loc` follows the same shape-and-strides scheme as the launched gather kernels, using parameters the launcher already builds, so the expert-batched path has no indexing rules of its own.
- **Rewrite rather than tune.** The index fix alone was 2.3x slower, and the causes were structural (row loop outermost, per-group reductions, idle lanes, serial run walk). Parameter changes would not have removed them.
- **One block per expert.** Grid z = E with two binary searches replaces `min(B, E)` run slots and a serial walk. At most 64 experts pass the gate, so empty blocks cost one search each.
- **f16 only.** One dtype pair, justified by the profile, keeps within the issue's instantiation cap. f32 stays on the existing paths.
- **The default follows the issue's rule.** Both conditions were measured on every eligible dtype and model on the host, and the off switch stays per call.
- **mxfp4 is a separate issue.** gpt-oss's dominant kernel is the non-affine fallback, which an affine kernel cannot replace, so it is filed as #2106 rather than extending this PR.

## 8. Process Note

The developer agent was interrupted by an API usage limit partway through the kernel rewrite. After the limit reset it was resumed with its context intact and finished the work from the uncommitted state in the worktree: the rewrite, the f16 dispatch, the test, the measurements and the PR. The orchestrator then rebased the branch onto origin/main and re-ran the gate (section 5.3).

## 9. Residual Risks and Follow-ups

- **Known LOW: out-of-range rhs rows are left unwritten.** Grid z is `E`, so a row whose rhs index is `>= E` belongs to no block and its output is not written; the wide kernel writes 0 there. MLX leaves out-of-range gather indices undefined, so this was left as is, but outputs for such rows differ between the two paths.
- **mxfp4 prefill (#2106).** gpt-oss-20b is unchanged at 99.8% of prefill in `gather_qmv_kernel`.
- **Fork package for item 9 (#1813).** The fork runs this kernel by default with the flat reads. A package still has to be prepared: a repro that calls sorted `gather_qmm` with `x` `[T, 1, K]` and `rhs_indices` `[T, 1]` on the unpatched fork, with the strided reads carried apart from the rewrite so the fork can take the correctness fix alone.
- **Metal and CUDA were not run.** The change is confined to `patches-rocm/` (copied only by ROCm builds) and a doc comment.
- **Coverage.** Models with more than 64 experts do not reach the kernel. Other MoE families the gate reaches were not measured (no checkpoint on the host). Wave64 (CDNA) is untested; the 16-lane reduction stays inside a wave on both widths. The four-rows-per-load factor was not tuned. The idot variant is fixed but still not launched.

## 10. Learning Points

- **A dtype-specific symptom can come from the gate.** "Wrong only for bf16" pointed at bf16 arithmetic, but bf16 was simply the only dtype routed to the kernel. Checking which inputs can reach a kernel comes before analysing its numerics.
- **Broadcast index arrays are not flat.** Any kernel that receives MLX gather indices must read them through the batch shape and strides; a flat read is only correct for the layout one caller happens to build.
- **Test the layouts the caller does not use.** The `SwitchGLU` layout passed with the old kernel. Only the broadcast layouts exposed the defect, and the test pins it by failing when the strided reads are reverted.
- **A correctness fix can expose a performance gap.** Once the kernel was correct it could be measured, and it was slower than the path it was meant to replace. The rewrite, not the fix, is what made the default possible.

Refs: #2066, #1814, #1801, #1813, #1809, #2106.
