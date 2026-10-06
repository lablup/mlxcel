# Technical Report: PR #2103 - Port paged attention (v1, v2 partial, merge) to HIP

**Date**: 2026-10-05

**Status**: Implemented and measured on the gfx1151 host; head `0bf1c97c` on origin/main `c05d5438`, pending merge.

**Languages**: C++ (HIP kernel sources, kernel holders, port tables, launcher shape checks), Rust (`PagedDecodeBackend::Rocm`, dispatch selector and memo key, `gridDim.z` guard, autotune runner labels, kernel bench example, unit tests), CMake and `build.rs` (header registration), Markdown (environment variables, installation, benchmark results), Makefile (comment)

**Risk Level**: Medium (it changes the default paged decode path on ROCm for the server's batched decode, MLA split-KV and sparse paged decode, and it adds host-side shape refusals and a holder refactor on Metal and CUDA, neither of which could be run on the development host)

## Executive Summary

Issue #2068 (part of #1814, epic #1801) asked for HIP ports of the three paged-attention kernels: the v1 split-K decode, the v2 partial and the v2 merge. Before this PR the three port tables had no `.rocm` entry, so on ROCm the server's batched paged decode, MLA split-KV and the sparse paged decode all ran gather-then-SDPA, and 36 tests in the ROCm gate skipped with a `skipping ... lablup/mlxcel#1814` line.

The PR adds `paged_attention_hip.h` with hipRTC ports of the three CUDA bodies, fills the three `.rocm` slots, and makes `paged_decode_backend()` return a new `PagedDecodeBackend::Rocm` so the v1 dispatch selector can pick the native kernel on ROCm. The HIP bodies read their geometry from `<input>_shape` exactly as the CUDA bodies do. That became possible when #2100 merged its overlay fix; the PR's first commit had carried a workaround for the missing fix, and a later commit removed it, so the Metal and CUDA kernel bodies and template arguments are byte-identical to main. The three launchers also gained shape refusals on every backend.

On gfx1151 the 36 previously skipped tests run and pass with no test edits, and a deliberate mutation of the HIP bodies makes 29 of them fail. Server paged decode with Meta-Llama-3.1-8B-Instruct-4bit improved per-request decode by 1.28x at batch 4 and ~16K tokens (three runs per arm), and by 1.40x and 1.77x for a single sequence at ~16K and ~32K (one run per arm). The ~4K case shows no measurable change.

## 1. Problem Statement

### 1.1 Three empty slots

The three kernels each had a `KernelPorts` table (`paged_attention_ports()`, `paged_v2_partial_ports()`, `paged_merge_ports()`) with Metal and CUDA entries and `.rocm = nullptr`. Their support predicates read `has_kernel_port`, so on ROCm they answered false, and every production gate in front of the kernels (`cache/paged.rs`, `mla/mod.rs`, `paged_v2/sparse.rs`) declined to the gather fallback. PR #2059 had already added the skip macros in `test_support/kernel_ports.rs`, so the tests skipped visibly instead of failing; the issue's goal was to make them run with no edits once the tables were filled.

### 1.2 A gate that did not follow the predicate

`paged_decode_backend()` returned `Other` unless `metal_is_available()` or `cuda_is_available()` was true. On ROCm a filled v1 table would still have mapped to `Other`, and `select_pooled_paged_dispatch` would never have chosen native. Filling the table alone was not enough to reach the v1 kernel from the library entry point.

### 1.3 Two holders keyed on a boolean

`PagedV2PartialHolder` and `PagedMergeHolder` took `bool use_cuda`, whose false arm compiled the Metal body. On a third backend that flag reads as Metal, so the holders had to name their backend before a ROCm entry could be added.

## 2. Change Summary

| Area | Change |
|---|---|
| `paged_attention_hip.h` (new, 430 lines) | `PAGED_ATTENTION_DECODE_HIP_SOURCE`, `PAGED_ATTENTION_V2_PARTIAL_HIP_SOURCE`, `PAGED_ATTENTION_MERGE_HIP_SOURCE`: the CUDA bodies with spelling changes only |
| `paged_attention.cpp` | `PagedAttentionKernelHolderHip` calling `fast::hip_kernel` under `MLXCEL_BRIDGE_ROCM_BACKEND`; `.rocm` getter; host shape refusal |
| `paged_attention_v2.cpp` | `make_partial_kernel(GpuKernelBackend)`, `PagedV2PartialHolder` keyed on `GpuKernelBackend`, one holder per backend via a template getter; `.rocm` getter; host shape refusal |
| `paged_attention_v2_merge.cpp` | Same refactor as the partial (`make_merge_kernel`, `PagedMergeHolder`); `.rocm` getter; host shape refusal |
| `layers.rs` | `PagedDecodeBackend::Rocm`, returned by `paged_decode_backend()`; CUDA rule in `select_pooled_paged_dispatch`; memo tag 3; three new or extended unit tests |
| `cache/paged.rs` | `gridDim.z <= 65535` guard applies to `Rocm` as to `Cuda`; comments |
| Autotune ops | `paged_decode_splits` and `paged_decode_v2_chunk` label ROCm tactics `rocm` instead of `cuda` |
| `examples/paged_attention_kernel_bench.rs` | Selector label reads the decode port predicate and `gpu_backend_kind()` as production does |
| Build | Header registered in `turbo/CMakeLists.txt` and `mlxcel-core/build.rs` |
| Comments and docs | `kernel_ports.rs`, `lib.rs`, `mlx_cxx_bridge.h`, `mla/mod.rs`, Makefile comment, `environment-variables.md`, `installation.md`, new `rocm-paged-attention-gfx1151-2026-10-05.md` |

The branch has five commits: the ports (`16b6b25d`), the bench label (`586e7e09`), the `_shape` switch after #2100 (`f4ca4b9a`), the review fixes (`0d1afcb3`), and the results doc (`0bf1c97c`). The diff against origin/main is 22 files, 840 insertions and 109 deletions.

## 3. The Ports

### 3.1 Same kernels, different spelling

Each HIP body keeps the CUDA body's thread mapping, kernel name, inputs, outputs, grid and template arguments. The header comment notes that the ROCm `CustomKernel::eval_gpu` ceil-divides the Metal-style total-thread grid by the threadgroup tuple the same way CUDA does, so every `blockIdx` and `threadIdx` means what it means in the CUDA body. The differences are:

- the 32-lane all-reduce is `__shfl_xor(v, o, 32)`, with the width stated, because `__shfl_xor_sync` exists in HIP only as a shim that ignores its mask;
- negative infinity is `-__builtin_huge_valf()`, because hipRTC compiles the body with only the headers `fast::hip_kernel` prepends and `INFINITY` comes from `<cmath>`;
- KV reads keep the explicit `(float)`, since `hip_bfloat16` converts to float only through an `explicit` operator.

The launches, and the `fast::hip_kernel` call that compiles each string, stay in the `.cpp` file that already holds the CUDA launch. The header holds data only. That keeps `make verify-kernel-dtype-keys` at its pinned scope: `EXPECTED_IN_SCOPE` and the `9 in scope` count are unchanged.

### 3.2 The wave32 guard, and why the merge kernel has none

The #1814 port requirements tell every port that reduces or shuffles across lanes to carry two `#error` checks on `__AMDGCN_WAVEFRONT_SIZE__` and `__AMDGCN_WAVEFRONT_SIZE`, because a fold that starts at lane 16 on a 64-lane CDNA wave drops half the lanes and still returns a finite, wrong result.

- **v1 decode and v2 partial** fold a dot product across 32 lanes with an XOR butterfly starting at 16. They carry the guard. As #2067 found, it is inert with ROCm 10's AMD clang, which defines neither macro for gfx1151, gfx942 or gfx90a, so it documents the assumption rather than enforcing it. The header comment adds a second line of reasoning: the 32 lanes that share a `threadIdx.y` are consecutive in the block's linear order, so on a wave64 target the explicit width of 32 should keep each butterfly inside its own row. That is stated as not run, since no wave64 device was available.
- **merge** has no lane-level operation: one thread per output element, no shuffle and no barrier. It is correct for any wavefront size, so it carries no guard; an `#error` there would only reject a correct kernel on wave64. The issue's acceptance checkbox said "all three tables ... with the wave32 guard". The PR deviates from that wording on purpose and records the reason next to the ticked box, in line with the port requirement's own scope ("every ported kernel that reduces or shuffles across lanes").

### 3.3 Holders that name their backend

`PagedV2PartialHolder` and `PagedMergeHolder` now take a `mlxcel::GpuKernelBackend`. A `make_partial_kernel` / `make_merge_kernel` switch compiles the Metal, CUDA or HIP body, and a template getter `get_partial_kernel<Backend>()` gives one static holder per backend. `std::call_once` is kept, because the server reaches first use concurrently from per-request workers and `call_once` re-runs the initializer if MLX device lookup throws. The `None` arm and the non-ROCm-build throw keep the switch exhaustive; they are unreachable because a holder is only reached through a table entry that names its backend.

## 4. Removing the #2100 Workaround

The CUDA bodies read head counts, block size and the merge head count from `q_shape`, `k_pool_shape` and `v_in_shape`. On ROCm that faulted until #2100: the vendored `fast::hip_kernel` declared `<input>_shape` as a pointer while the launch passed it by value (LOCAL_FIXES item 30).

The first commit (`16b6b25d`) worked around that. It passed the geometry as new `NumQHeads`, `NumKVHeads` and `PoolBlockSize` template arguments on every backend, read the merge head count from `gridDim.y`, and used `static_assert`s to keep `_shape`, `_strides` and `_ndim` out of the HIP text. The cost was a change to the Metal and CUDA launches: extra template arguments that those bodies never read, which also change their JIT cache keys, on two backends the host could not run.

Once #2100 merged, commit `f4ca4b9a` dropped the workaround (12 insertions, 68 deletions across three files). The HIP bodies now read `_shape` exactly as the CUDA bodies do, and the Metal and CUDA launches match main again. Checked for this report against origin/main `c05d5438`: all six Metal and CUDA source strings in the three launcher files are byte-identical, no `.metal` file changed, and the diff touches no `TemplateArg` list. The removal leaves one way to read the geometry across three backends and keeps the untestable backends' kernels exactly as they were.

## 5. Host-Side Shape Refusals on Every Backend

The three launchers now refuse shapes the bodies cannot index, before building the launch, on every backend. They throw `std::invalid_argument`; the bridge declares these functions `Result`, so the throw reaches Rust as an `Err`. This follows what #2067 did for the SSM update kernel.

- **v1 decode and v2 partial**: `q`, `k_pool` and `v_pool` must be rank 4; `q` must be `[B, Hq, 1, D]` with `D >= 1`; `k_pool`'s block size and head count must be at least 1 and its D must match `q`'s; and `v_pool` must match `k_pool` in axes 1 to 3. Axis 0 is not compared, because MiniMax-M3's sparse launch reshapes both pools to `[rows, 1, 1, D]` and K carries an extra index-key side head, so K has more rows than V.
- **merge**: `v_in` rank 3, `lse_in` rank 2 with `v_in`'s N and H, `o_indptr` rank 1, and a head dim from 1 to 1024, since the threadgroup is `(D, 1, 1)` and D must be a launchable block width.

The first version of these checks compared only rank and head dim. Review of #2103 found the V-pool check too loose: the bodies address V with K's block size and head stride, so a V differing in axes 1 to 3 would be read out of bounds. Commit `0d1afcb3` tightened it and added the `D = 0` and merge-width refusals.

These refusals run on Metal and CUDA too. A call that used to launch and read out of bounds now returns an error. Production callers on ROCm pass conforming shapes (the paged, MLA, autotune and layers selectors, 550 tests, pass after the change), but the Metal and CUDA paths were not run with them.

## 6. `PagedDecodeBackend::Rocm`

- **Detection.** `paged_decode_backend()` returns `Rocm` when the v1 predicate `paged_attention_decode_available()` is true, neither Metal nor CUDA is available, and `crate::hardware::gpu_backend_kind()` is `Rocm`. Metal is still probed first and CUDA second, so those answers are unchanged. The `Other` doc now covers only CPU-only builds and machines with no usable GPU.
- **Selector.** In `select_pooled_paged_dispatch`, `Rocm` takes the CUDA rule, `slab_count <= NATIVE_MAX_SLABS`: native on any single-slab layer, including the batch-1 and long-context regimes that Metal routes to gather. The HIP kernel is the CUDA kernel's port and reads the pool the same way, and no ROCm-measured batch or context ceiling exists to justify a narrower island.
- **Memo key.** `PagedDispatchCache::pack_key` gives `Rocm` tag 3, after Metal 0, CUDA 1 and Other 2. The 2-bit tag field is now full.
- **`gridDim.z` guard.** The CUDA launch puts `batch * query_heads` in `gridDim.z`, which CUDA caps at 65535. That limit is hit when the graph is evaluated, not when the bridge call returns, so the launcher's `Result` cannot report it, and `cache/paged.rs` declines to gather beforehand. The HIP port uses the same grid, so the guard now matches `Cuda | Rocm`. ROCm keeps the CUDA bound instead of relying on a device-reported limit.
- **Tests.** `selector_rocm_backend_follows_the_cuda_rule` compares `Rocm` with `Cuda` shape by shape over five (batch, context) pairs and four slab counts. `paged_decode_backend_names_the_resolved_backend` checks the answer against `gpu_backend_kind()` wherever the v1 port exists. The memo test checks that a ROCm query is not served from an aliased Metal or Other cell, and the pack-key test checks four distinct keys, including a saturated ROCm key, clear of the decision bit and the empty sentinel.

Two smaller consumers follow the same detection. The kernel bench example now labels its `select=` column `Rocm` on ROCm instead of printing the gather decision production no longer makes. The two paged autotune ops return `rocm` as their runner id, so ROCm tactics are not stored under `cuda` next to a different kernel's.

## 7. Correctness Evidence

### 7.1 The 36 skipped tests

The 36 mlxcel-core tests that skipped on ROCm for lack of these ports (36 `skipping ... lablup/mlxcel#1814` lines on `57d8ed29`) now run and pass on gfx1151, with none of them edited. The issue text counted 34; the gate printed 36. They cover the v1 decode against the gather path over 200 steps and a GQA and batch matrix, the v2 partial and merge against host references and the gather path (f32 and f16 pools, empty requests, trimmed windows, GQA head mapping, two pool dtypes at one geometry), cascade, sparse and MLA split-KV.

### 7.2 Mutation check

With the HIP lane fold started at 8 instead of 16 and the HIP merge computed in base e instead of base 2, 29 of those tests fail: v2 launch, cascade, sparse, split-KV, batched decode, and `test_fused_paged_decode_native_vs_fallback_matrix` for v1. The two v1 tests at head dim 8 do not see the fold change, because 8 dims fit in lanes 0 to 7 and the dropped fold stage only combined zeros. The tests therefore detect a port that folds or rescales incorrectly, not only one that fails to launch.

### 7.3 Gates

From the PR, on gfx1151 (Radeon 8060S, ROCm 10) at `f4ca4b9a` rebased on `c05d5438`: `make verify-rocm` OK, 11869 passed, 0 failed, 378 ignored across 147 test binaries, ROCm smoke OK, no `#1814` skip line. After `0d1afcb3`: paged, MLA, autotune and layers selectors pass (550 tests); `cargo clippy -p mlxcel-core --features rocm --lib --tests -- -D warnings`, `cargo fmt --check` and `dead_doc_pointers` pass. `make verify-kernel-dtype-keys verify-kernel-port-dispatch` pass with 9 in scope and `EXPECTED_IN_SCOPE` unchanged.

Orchestrator verification on head `0bf1c97c`, up to date with origin/main `c05d5438`: `make verify-rocm` passed every step, 11,869 passed, 0 failed, 378 ignored; smoke OK; zero `skipping ... #1814` lines in the log.

## 8. Measured Results

### 8.1 Why the server, not `bench_decode.sh`

The #1814 port requirements name `scripts/bench_decode.sh`. The decode profile put paged attention at 0% of decode because `bench_decode.sh` decodes into a dense KV cache, so it never reaches these kernels. Issue #2068's own acceptance criterion names `scripts/benchmark_paged_decode_production.sh` (issue #899), which drives the server's paged decode, and that is what was run.

### 8.2 Method

Meta-Llama-3.1-8B-Instruct-4bit, `mlxcel-server --parallel 4 --ctx-size 131072`, `scripts/bench_serving_concurrency.py` with 128 decode tokens per request, at `f4ca4b9a`. One binary for both arms: before is `MLXCEL_PAGED_ATTENTION_NATIVE=0` (gather-then-SDPA, which every ROCm build ran before this change), after is unset (fused v2). Each case started a fresh server per arm, and the before and after arms of one case ran back to back inside one `scripts/rocm_gpu_guard.sh` window of 55 s with `/sys/class/kfd/kfd/proc` empty and no compiler. Attempts that saw a foreign GPU process or a compiler were rejected and rerun. Every after-arm server log announces `paged decode v2: fused v2 launch`, and no before-arm log does. The later commits change host shape checks, comments and the autotune label, not the kernel bodies or dispatch, and were not re-measured.

### 8.3 Per-request decode tok/s

| Case | Runs per arm | Before | After | After / before (medians) |
|---|---|---|---|---|
| batch 4, ~1K prompt | 3 | 5.7 / 5.5 / 5.6 | 6.1 / 6.3 / 6.0 | 1.09x |
| batch 4, ~4K prompt | 3 | 10.1 / 10.6 / 10.2 | 10.4 / 10.6 / 10.5 | 1.03x, within run spread |
| batch 4, ~16K prompt | 3 | 10.9 / 10.9 / 10.9 | 13.9 / 13.9 / 13.9 | 1.28x |
| batch 1, ~16K prompt | 1 | 17.2 | 24.1 | 1.40x |
| batch 1, ~32K prompt | 1 | 11.2 | 19.8 | 1.77x |

The batch-4 cases ran three times per arm, with run-to-run spread of at most 0.5 tok/s. The ~1K and ~16K gains are outside that spread; the ~4K gain is not and is reported as no measurable change. The two single-sequence cases ran once per arm, because a guarded window long enough for them was rare on the shared host, so their ratios have no spread estimate and rest on one pair each. The direction matches the batch-4 trend (the gain grows with context), but the 1.40x and 1.77x figures should be read as single observations.

Time to first token is unchanged within a few percent in every case, as expected since prefill does not use these kernels: for example 155.1 s before and 151.1 s after (medians) at batch 4 and ~16K.

### 8.4 Why the aggregate column is not used

The script also reports aggregate tok/s, all completion tokens over the level's wall-clock span. At long context it moves the other way (batch 4, ~16K: 1.9 / 2.0 / 2.0 before, 1.7 / 1.7 / 1.7 after) because it is dominated by prefill, about 150 s of time to first token at that point, and because it depends on how many tokens each request generated before stopping, which differs between the two numeric paths. It does not measure the kernel. The results page lists it and explains this; the PR and this report do not use it for any claim.

## 9. Technical Decisions

- **Port the CUDA bodies line for line.** CUDA and HIP share grid semantics, `template_args` and runtime compilation. Identical bodies leave one place to look when backends disagree.
- **Guard only where lanes talk.** v1 and v2 partial carry the guard; merge does not. This departs from the checkbox's wording and follows the port requirement's scope.
- **Remove the template-argument workaround once #2100 landed.** Reading `_shape` keeps one geometry contract across backends and leaves Metal and CUDA byte-identical to main, which matters because neither could be run here.
- **Keep launches next to the CUDA launches.** The dtype-key checker's pinned scope and count stay unchanged.
- **Refuse bad shapes on the host for every backend.** An `Err` at the bridge is better than an out-of-bounds read on any backend; the axis-0 exemption keeps MiniMax-M3's sparse launch working.
- **ROCm takes the CUDA selector rule and the CUDA grid bound.** Same kernel, same grid; no ROCm-specific ceiling has been measured, and a device-reported limit was not trusted over the known bound.
- **Name the backend in holders, tags and labels.** A boolean `use_cuda`, a `cuda` runner id and an `Other` fallback each read wrong once a third backend exists.

## 10. Residual Risks and What Was Not Verified

- **Metal and CUDA were not run.** Their kernel bodies, template arguments and table entries are unchanged, but they are touched by the holder refactor (same `metal_kernel`/`cuda_kernel` calls, one holder per backend) and by the new shape refusals.
- **No wave64 device.** The guard is inert on ROCm 10 clang, so a wave64 build would not be stopped at compile time. Correctness of the v1 and v2 partial fold on wave64 is reasoned from the explicit shuffle width and lane layout, not run.
- **Single-run long-context points.** The 1.40x and 1.77x batch-1 figures are one pair each.
- **Guard window shorter than the port requirement's.** The runs used a 55 s idle window; the #1814 requirements name 90 s for `bench_decode.sh`.
- **The memo tag field is full.** Four backend tags use both bits; a fifth backend needs the key layout widened.

## 11. Follow-ups

- **32-bit index math past 2^32 pool elements.** All three backends compute the KV address as `(row * block_size + slot) * stride_kv + kv_head * dim` in 32-bit unsigned arithmetic (`uint` in the Metal bodies, `uint32_t` in the CUDA and HIP bodies). A pool with more than 2^32 elements wraps that offset and reads the wrong block. For a pool with 8 KV heads of dimension 128, that is 4,194,304 token slots. This predates the PR, is shared by every backend, and is not caught by the new shape refusals, which check rank and per-axis agreement, not total size. Fixing it means 64-bit offsets in all three bodies (or a host refusal past the limit), which changes Metal and CUDA and so needs those hosts.
- **Slow batched server decode is a scheduling issue.** Batch-4 per-request decode on this server (5.6 to 13.9 tok/s) is far below the single-stream `bench_decode.sh` rate for the same model, 35 tok/s, because the per-request rate includes time spent waiting while the other three requests prefill. That is the same in both arms and is not addressed by these kernels.

## 12. Learning Points

- **Filling a port table is not always enough.** `paged_decode_backend()` enumerated backends instead of reading the predicate's backend, so the v1 kernel stayed unreachable until it learned the new variant.
- **Remove a workaround as soon as its cause is fixed.** The template-argument detour worked, but it changed two backends that could not be tested. Dropping it after #2100 restored byte-identity with main.
- **A guard belongs only where lanes communicate.** The merge kernel's one-thread-per-element design is wave-size independent, and adding a guard would reject a correct kernel.
- **Mutation checks show what a suite can see.** The fold mutation went unnoticed by the two v1 tests at head dim 8; the matrix test caught it. Without the mutation the suite's blind spot at small D would not be known.
- **State run counts with the numbers.** A table that mixes three-run medians and single runs should say which is which, and a column dominated by something other than the kernel should be explained and set aside, not quoted.

Refs: #2068, #1814, #1801, #2100, #2067, #2059, #2061, #1803, #899, #898, #634.
