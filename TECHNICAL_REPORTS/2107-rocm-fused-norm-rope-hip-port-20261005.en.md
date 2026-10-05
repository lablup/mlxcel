# Technical Report: PR #2107 - Port fused_add_rms_norm and fused_rope_qk_append to HIP

**Date**: 2026-10-05

**Status**: Implemented and measured on the gfx1151 host; head `c6592780` on origin/main `33c45053`, pending merge.

**Languages**: C++ (two HIP kernel sources, kernel holders, port tables, support predicates, launcher refusals), Rust (gate caching, eligibility checks, parity tests, CPU-device gate test), `build.rs` (header tracking), Markdown (results page, environment variables, installation, README, correctness page), CSV and TSV (benchmark rows, logit traces)

**Risk Level**: Low to medium (both fusions stay off by default on every backend, so the default decode path does not change; the predicate and empty-input changes also reach Metal and CUDA, which could not be run on the development host)

## Executive Summary

Issue #2063 (part of #1814, epic #1801) asked for HIP ports of `fused_add_rms_norm` and `fused_rope_qk_append` (#905). Before this PR both port tables had `.rocm = nullptr`, so on ROCm `MLXCEL_FUSED_ADD_RMSNORM=1` and `MLXCEL_FUSED_ROPE_APPEND=1` silently kept the MLX graph path.

The PR adds `fused_norm_hip.h` and `fused_rope_append_hip.h`, fills both `.rocm` slots, and makes each port match the ROCm graph it replaces bit for bit, including the graph's sign of zero in the fused RMSNorm. Both predicates now also require the GPU as the default device, and the Rust gates cache only a `true` port answer. Empty inputs are refused at the Rust gate and at the launcher, so they take the graph.

On gfx1151, decode did not clearly improve: Llama 3.1 8B moved +0.3% (noise), Qwen2.5 7B gained +1.5% to +1.7% in medians with a size that drifted between round groups as much as the off arm's own spread, and `both` was not above `add`. Both defaults therefore stay off on ROCm, as on Metal and CUDA. Teacher-forced traces with the fusions on are byte-identical to the traces with them off, with 0 decided mismatches against Metal.

## 1. Problem Statement

### 1.1 Two empty slots

`fused_norm_ports()` and `fused_rope_ports()` held Metal and CUDA entries only. Their predicates returned `has_kernel_port`, so on ROCm they answered false and the opt-in environment variables had no effect. The parity tests `fused_norm_parity_tests.rs` and `fused_rope_parity_tests.rs` returned early while the predicate was false, so nothing on ROCm exercised them.

### 1.2 What a port has to match

The #1814 port requirements ask for parity with the graph fallback on gfx1151 within the existing tolerances. The decode profile (#2061) had put the most these two kernels could take over at 0.83% to 0.89% of Llama 3.1 decode GPU time, so the issue also asked for measurement before and after, and the question the PR had to answer was whether the fusions should run by default on ROCm.

## 2. Change Summary

| Area | Change |
|---|---|
| `fused_norm_hip.h` (new, 161 lines) | `FUSED_ADD_RMS_NORM_HIP_SOURCE`, ported from the CUDA body and adjusted to reproduce the overlay's `rms_norm_kernel` |
| `fused_rope_append_hip.h` (new, 210 lines) | `FUSED_ROPE_APPEND_HIP_SOURCE`, ported from the CUDA body and adjusted to reproduce `rope.hip` |
| `fused_norm.cpp` | `FusedNormKernelHolderHip` calling `fast::hip_kernel` under `MLXCEL_BRIDGE_ROCM_BACKEND`; `.rocm` getter; `Threads` fixed at 256 on ROCm; GPU-device term in the predicate; empty-input refusal |
| `fused_rope_append.cpp` | `FusedRopeKernelHolderHip`; `.rocm` getter; GPU-device term in the predicate; zero batch or window refusal |
| `layers.rs` | `gpu_port_available` (per-call device read, cache only a `true` port answer); eligibility requires non-empty inputs; doc comments on both defaults record the ROCm result |
| Parity tests | Fail instead of skip on a GPU backend whose predicate is false; ROCm byte-identity tests for both kernels; sign-of-zero test; RoPE dtype sweep with the norm tests' bf16 budget |
| `tests/cpu_device_custom_kernel_gates.rs` | Both predicates must decline on the CPU device |
| Bridge and FFI docs | `mlx_cxx_bridge.h`, `lib.rs` describe the device term and the ROCm port |
| Build | `mlxcel-core/build.rs` tracks the two headers |
| Docs and data | New `docs/benchmark_results/rocm-fused-norm-rope-gfx1151-2026-10-05.md`, four benchmark CSVs, 12 trace TSVs with metadata under `benchmarks/logit_traces/rocm_gfx1151_3f0e51af/`, updates to `environment-variables.md`, `installation.md`, README and the correctness page |

The branch has six commits plus one merge of origin/main: the ports (`ef3831d9`), the bit-for-bit fixes (`8ebf809f`), the sign-of-zero fix (`3f0e51af`), the results page (`756d03d8`), review fixes (`bea7d1ab`), the merge (`9e6f095d`), and the empty-input refusals (`c6592780`). The diff against origin/main is 38 files, 4748 insertions and 99 deletions; most of the insertions are trace TSVs. `.rocm = nullptr` lines in `src/**/*.cpp` drop by two, from 7 on origin/main `33c45053` to 5 on the head (10 to 8 on the branch's original base).

## 3. The Ports: Matching the ROCm Graph Bit for Bit

### 3.1 Follow the graph, not the Metal kernel

Each HIP body keeps the CUDA body's thread mapping, kernel name, inputs, outputs, grid and template arguments, and its launch stays in the `.cpp` file that already holds the CUDA launch, so `make verify-kernel-dtype-keys` keeps its `9 in scope` pin unchanged. Where the PR departs from a line-for-line port, it does so to reproduce what the ROCm graph computes, not what the Metal or CUDA kernel computes. The result is that turning a fusion on changes no bit of the output on ROCm.

**`fused_add_rms_norm`** replaces `add` followed by the overlay's `rms_norm_kernel<T, 256, 4>`. The port:

- launches 256 threads per row on ROCm (`FUSED_NORM_ROCM_THREADS`, the overlay's `BLOCK_DIM`) instead of sizing the block from the row, so the per-thread strided sums and the 32-wide `__shfl_xor` folds follow the graph's reduction tree;
- reads the row length at run time from `weight_shape[0]`, as the graph's kernel does, instead of from the `Dim` template constant (which stays in the cache key);
- uses `1.0f / sqrtf(...)` as `rms_norm_row` does;
- multiplies the gain in `T` (`gain * scaled`, both `T`), as the graph's `w * normalized` does.

**`fused_rope_qk_append`** replaces the slices, reshapes, transposes and two `fast_rope` calls. The port:

- computes the angle in `rope.hip` order, `(scale * position) * inv_freq` with `inv_freq = exp2f(-d * log2(base))`, and takes sine and cosine with one `sincosf`;
- writes both rotation outputs as explicit `fmaf` calls under `#pragma clang fp contract(off)`, in the form the graph uses for the shape at hand: `rope_single_1d` for one token of one sequence (`B == 1 && L == 1`, batch-1 decode) and `rope` otherwise, because the two hipcc-compiled kernels fuse the second output differently;
- for f16, forms each fused result in double (the float products are exact there) and converts to `__half` once, because the graph's f16 instantiation rounds once while `(T)fmaf(...)` rounds through f32 first. f32 and bf16 keep the `fmaf` value.

### 3.2 The sign of zero

The last discrepancy was the hardest to see. With the other fixes in place, Llama 3.1 decode traces with `MLXCEL_FUSED_ADD_RMSNORM=1` still differed from the graph at 5 of 128 positions. Instrumenting the join found one element per affected call where the port wrote +0 and the graph wrote -0: a normalized element that underflows in f16, or a zero weight times a negative element. The port had multiplied the gain in f32 and rounded once, which gives the same value in f16, bf16 and f32, but hipRTC's code for that form dropped the sign of a zero product. Commit `3f0e51af` moved the multiply into `T`, as the graph does, and added `fused_add_rms_norm_keeps_the_rocm_graph_sign_of_zero` (underflowing elements and signed-zero weights in f32, f16 and bf16). With that change, Llama 3.1 8B and Qwen2.5 7B traces with the fusions on became byte-identical to the traces with them off.

### 3.3 Each detail came from a failing test

None of these details came from reading the code. hipRTC-compiled ports rounded differently from the hipcc-compiled graph wherever the expression left the compiler a choice, and each difference surfaced as a failing test or a trace mismatch:

| Detail | Failure without it |
|---|---|
| Runtime row length | 44 of 4096 f32 rows a normalizer ulp off at width 4096 (row scales over e^-8 to e^8) |
| 256 threads per row | Row-sized thread count fails the norm byte-identity test |
| Gain multiply in `T` | +0 for -0; Llama 3.1 logits moved at 5 of 128 decode positions |
| Per-shape FMA form | A port matching only the multi-token form failed batch-1 decode; left to hipRTC, about one f32 element in ten moved by one ulp |
| Single f16 rounding | Disagreement on about one element in 2^13, which a 186-token window of 48 heads always hits |

The PR's negative checks confirm each fix is pinned: a row-sized thread count, a compile-time row length, an f32 gain product, a single RoPE FMA form, f32 rounding of f16 RoPE results and a norm fold starting at 8 each make a parity test fail. Beyond the committed tests, a one-off stress run of 432 RoPE cases (batch 1 to 3, windows of 1 to 300 tokens, offsets to 131000, full and partial rotary dims, values scaled up to 181x) and a 4096-row norm run at four widths matched the graph bit for bit in f32, f16 and bf16. Those runs are not committed tests.

### 3.4 Wave guard

The norm body keeps the two `#error` checks on `__AMDGCN_WAVEFRONT_SIZE__` and `__AMDGCN_WAVEFRONT_SIZE` that #1814 asks of every shuffle-based port. They are inert with HIP 7.15's AMD clang 23, which defines neither macro for gfx1151 or gfx942. What keeps each fold inside one 32-lane group is the explicit `__shfl_xor` width of 32, together with `lane` and `sg` being `threadIdx.x % 32` and `threadIdx.x / 32`. On a 64-lane wavefront both halves would reduce separately into `local_sums`; that reasoning has not been run. The RoPE body has no shuffle and no shared memory read across threads, so it is correct on any wavefront size and carries no guard, as with the #2064 samplers. This departs from the issue's acceptance wording ("both HIP sources carry the wave32 guard"), and the issue records the deviation next to the ticked box.

## 4. Gating

### 4.1 The GPU-device term on both predicates

`fused_add_rms_norm_available()` and `fused_rope_qk_append_available()` now return false unless `mlx::core::default_device()` is the GPU, before reading `has_kernel_port`. Custom kernels run only on the GPU stream; on a CPU default device (`MLXCEL_DEVICE=cpu` on a GPU build) their `eval_cpu` throws. This follows #2069, which added the same term to other custom-kernel gates. `tests/cpu_device_custom_kernel_gates.rs` now asserts that both predicates decline on the CPU device. The term applies on every backend: on Metal and CUDA, `MLXCEL_DEVICE=cpu` with a fusion enabled now takes the graph path instead of throwing at the first launch.

### 4.2 The Rust gate caches only a true port answer

The Rust gates used a `OnceLock<bool>` that asked the FFI predicate once. With a device term in the predicate that is no longer correct: the device can move (`MLXCEL_DEVICE`, `DefaultDeviceGuard`), so caching the first answer would freeze it. The PR splits the check in `gpu_port_available`:

- the device half (`ffi::default_device_is_gpu()`) is read on every call, one FFI read of MLX's default device;
- the port half is asked until it first answers `true`, and then recorded in a `OnceLock<()>`.

The first version cached whatever the port check returned. Review (`bea7d1ab`) found a race: the C++ predicate reads the device again, so another thread moving the device between the Rust read and the C++ read could make the predicate return `false` and switch the fusion off for the rest of the process. Caching only `true` closes that window, since a `false` answer is simply asked again next time. The backend cannot change mid-process, so a cached `true` stays valid.

## 5. Empty-Input Refusals

The final commit `c6592780` came from a security review. An empty input reached the fused launchers with a zero-size grid, and a zero-width row made the norm launcher divide by zero (`x.size() / dim`). Two layers now refuse these shapes:

- **Rust eligibility.** `fused_add_rms_norm_eligible` requires a positive trailing dimension and every dimension positive. `FusedQKVLinear` requires a positive batch and window before taking the fused RoPE path. Both fall back to the graph, which handles empty arrays.
- **C++ launchers.** `fused_add_rms_norm` throws `std::invalid_argument` for an empty `x` or a zero last dim; `fused_rope_qk_append` throws for a zero batch or window. These protect a direct caller that skips the Rust gate; the bridge declares both functions `Result`, so the throw reaches Rust as an `Err`.

These cases already existed on Metal and CUDA. There they now take the graph as well.

## 6. Correctness Evidence

### 6.1 Tests

On gfx1151, `cargo test --release --features rocm -p mlxcel-core --lib -- --test-threads=1 fused_norm_parity_tests fused_rope_parity_tests` runs 21 tests, all passing:

- The existing tolerance tests (norm: f32 1e-6 / 1e-5, f16 2e-3 / 1.2e-2, bf16 1.6e-2 / 7e-2 normalized RMS / max; RoPE 2e-3 / 1.2e-2) now run on ROCm, and fail rather than skip if a GPU backend's predicate is false.
- `fused_add_rms_norm_is_byte_identical_to_the_rocm_graph`: f32, f16, bf16 at widths 128 to 4096, plus 1024 rows of width 4096 with row scales over e^-8 to e^8.
- `fused_add_rms_norm_keeps_the_rocm_graph_sign_of_zero`.
- `fused_rope_append_matches_graph_rope_every_dtype`, a new tolerance sweep over f32, f16 and bf16. Review (`bea7d1ab`) gave bf16 the norm tests' bf16 budget: the f16 budget is about one bf16 ulp on a 2-sigma element, so a single rounding flip on Metal or CUDA, where the two paths contract differently, would fail it. f32 and f16 budgets are unchanged, and the issue's tolerances are not changed.
- `fused_rope_append_is_byte_identical_to_the_rocm_graph`: batch 1 and 2, windows of 1 to 512 tokens, offsets to 131071, both conventions.

### 6.2 Traces

Teacher-forced traces (`benchmarks/logit_traces/rocm_gfx1151_3f0e51af/`), for Llama 3.1 8B with `MLXCEL_FUSED_ADD_RMSNORM=1` and Qwen2.5 7B with both flags, at `w1`, `w8` and `w1ctx512`: every on/off pair is byte-identical. Against Metal (`compare_logit_traces.py --decided 2.0`):

| Reference | Candidate | Top-1 disagreement | Decided mismatches |
|---|---|---|---|
| `metal_m1u_bec64748` Llama 3.1 `w8` | ROCm `addrms` `w8` | 0 / 640 | 0 / 230 |
| `metal_m5_d1128266` Qwen2.5 `w8` | ROCm `addrms-rope` `w8` | 4 / 640 (largest gap 0.047) | 0 / 274 |

Because the on and off traces are identical, these rows equal the fusions-off rows. A `rocprofv3` kernel trace of an 8-token generation confirms what runs: Qwen2.5 launches both custom kernels and no graph RoPE kernel; Llama 3.1 launches the norm port and keeps `rope_single_freqs_1d` / `rope_freqs`, because its `rope_scaling` table routes around the RoPE kernel.

### 6.3 Gates

From the PR: `make verify-rocm` on `756d03d8` passed with 147 suites, 11871 passed, 0 failed, 378 ignored. After `bea7d1ab`, the parity tests (21 passed), `cpu_device_custom_kernel_gates`, `dead_doc_pointers`, clippy on mlxcel-core lib and tests, and fmt passed, and Qwen2.5-7B greedy 64-token output was identical with both fusions off and on. `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt` passed with `9 in scope` and the pin unchanged.

Orchestrator verification on head `c6592780`, up to date with origin/main `33c45053`: `make verify-rocm` passed every step, 11,873 tests passed, 0 failed, 378 ignored, smoke OK.

## 7. Measurement and the Decision to Keep Both Defaults Off

### 7.1 Method

`scripts/bench_decode.sh` at pp512/tg128, one arm per run, arm order rotated each round, every run through `scripts/rocm_gpu_guard.sh --idle-secs 60`. All 44 runs were clean on the first attempt; a parallel unit was using the GPU between runs, and the guard waited it out. Off is `MLXCEL_FUSED_ADD_RMSNORM=0 MLXCEL_FUSED_ROPE_APPEND=0`. Raw rows are in `benchmarks/rocm_strixhalo-gfx1151_2026-10-05_fused-norm-rope-{off,add,rope,both}.csv`. The measurement ran on `3f0e51af`; later commits changed gating, tests, empty-input refusals and docs, not the kernel bodies.

### 7.2 Results

| Model | Arm | Median decode tok/s | vs off | Paired median (on minus off) |
|---|---|---|---|---|
| Qwen2.5-7B (7 rounds) | off | 46.53 | | |
| | add | 47.23 | +1.5% | +1.48 (6 of 7 positive) |
| | rope | 46.80 | +0.6% | +0.33 (6 of 7) |
| | both | 47.32 | +1.7% | +0.95 (6 of 7) |
| Llama-3.1-8B (5 rounds) | off | 37.85 | | |
| | add | 37.95 | +0.3% | -0.01 (2 of 5) |
| Qwen3-30B-A3B (3 rounds) | off | 62.21 | | |
| | both | 62.42 | +0.3% | +0.06 (2 of 3) |

Prefill (512 tokens) medians moved by +0.4% to +4.3%, inside per-run spreads of the same width (Qwen2.5 off alone ranged from 1565 to 1651 tok/s), so prefill is not read as a result.

### 7.3 Why the data does not support a flip

- **Llama 3.1** runs only the norm port (one add + RMSNorm join per layer; its `rope_scaling` routes around the RoPE port). Median and paired differences sit at zero.
- **Qwen3-30B-A3B** calls neither kernel (`qwen3_moe.rs` has its own block), so its +0.3% is a control and reads as noise.
- **Qwen2.5 7B** is the one local model that reaches both ports, and its numbers are where the decision lies. Every on arm beat off in 6 of 7 rounds, but three observations stop that from being a result:
  - **Drift between round groups.** Rounds 1 to 3 gave +0.95 to +3.5 tok/s. Rounds 4 to 7, run about two hours later on the same binary, gave mostly -0.4 to +1, with -2.7 for `both` in round 4 and +2.7 for `add` in round 7. The size of the gain depends on when it was measured.
  - **`both` does not exceed `add`.** If each port saved time independently, both on would be above either alone. It is 47.32 against 47.23 in medians and lower in the paired median (+0.95 against +1.48).
  - **The off arm's spread is as wide as the gain.** Off ranged from 45.75 to 46.95 tok/s, about the same width as the median gain.

  An earlier run on the branch's first commit gave the same picture (7 rounds: off 46.54, add 46.84, rope 46.95, both 47.58 tok/s).

- **The plausibility argument.** A 1% to 2% gain on Qwen2.5 is plausible from the dispatches the ports remove: the RoPE port alone stands in for three slices, three reshape and transpose pairs and two `fast_rope` calls per layer, and #2099 showed that GPU-time shares understate paths made of many small ops. The results page states this and also states that the data does not establish it. This report keeps that distinction: the argument explains why a gain might exist, not that one was measured.

`FUSED_ADD_RMSNORM_DEFAULT` and `FUSED_ROPE_APPEND_DEFAULT` therefore stay `false` on ROCm, as on Metal and CUDA, and their doc comments in `layers.rs` now record the ROCm numbers. Because the ports are byte-identical to the graph, opting in on ROCm costs nothing in output, so a deployment that measures a win on its own model can set both flags. The results page names what would justify a ROCm default of on: a longer interleaved run on a quiet GPU (20 or more pairs) showing Qwen2.5-class models (no `rope_scaling`, so both ports run) gaining at least 1% with the paired differences consistently positive, and Llama-class models not losing.

## 8. Technical Decisions

- **Match the ROCm graph, not the CUDA kernel.** A port that is only within tolerance would make the opt-in change outputs. Matching the graph bit for bit makes the flag a pure performance choice and lets on/off traces be compared byte for byte.
- **Fix `Threads` at 256 on ROCm by build flag.** A ROCm build has no Metal or CUDA backend, so the build flag identifies the backend, as in `fused_add3_layer_norm`. `Threads` is a template argument, so the cache key stays per width.
- **Read the device per call, cache only a true port answer.** The device can move; the backend cannot. Caching `false` would let a transient read disable the fusion for the process.
- **Refuse empty inputs at both layers.** The Rust gate keeps normal callers on the graph; the launcher throw protects direct callers.
- **Guard only the kernel that shuffles.** The RoPE body has no cross-lane operation; an `#error` there would only reject a correct kernel on wave64.
- **Keep the defaults off.** The measured gain is not stable across round groups, and the combined arm does not behave like two independent savings.

## 9. Process Note

The developer agent stopped on an API usage limit after opening the PR. The orchestrator then ran the final gate (`make verify-rocm` on `c6592780`, results in section 6.3), updated the PR body with that verification and set the labels (`status:done`, `type:performance`, `priority:medium`, `area:core`, `platform:linux`).

## 10. Residual Risks and What Was Not Verified

- **Metal and CUDA were not run.** Their kernel sources and table entries are untouched, but their predicates gained the GPU-device term, their launchers gained the empty-input refusals, and their parity tests now fail rather than skip if the predicate is false.
- **Byte identity depends on the compiler.** The ports reproduce hipcc's code for the overlay kernels on gfx1151 with HIP 7.15. A change to hipcc, hipRTC or the overlay's `rms_norm.hip` / `rope.hip` could break identity; the byte-identity tests would show it first.
- **Gemma and IQuest Loop Coder** also call the norm port but have no checkpoint on this host. The Gemma `(1 + w)` convention is covered by the tolerance tests only.
- **Wave64 (CDNA) is untested.** The norm guard is inert on current clang, so a wave64 build is not stopped at compile time; correctness there rests on the explicit shuffle width.
- **The performance question is open.** The measurement was short (5 to 7 rounds) on a shared host; the recommendation is to keep the defaults off, not that the ports bring no gain.

## 11. Learning Points

- **Bit-for-bit parity across two compilers is found by tests, not reading.** hipRTC and hipcc made different choices on the same expressions (contraction, compile-time constants, signed zero). Each was found by a failing test or trace and pinned by a test that fails with the fix reverted.
- **Signed zero is a real difference.** Two computations equal in value can still move logits if one loses the sign of zero.
- **Cache only the answer that cannot change.** Splitting the device check from the port check, and caching only `true`, keeps a hot-path gate cheap without freezing a transient answer.
- **A combined arm is a consistency check.** `both` not exceeding `add` was as informative as the median gains.
- **Keep plausibility separate from evidence.** The removed-dispatch argument explains why a gain might exist; it is recorded as that and not as a measured result.

Refs: #2063, #1814, #1801, #905, #2061, #2064, #2069, #2099, #1809.
