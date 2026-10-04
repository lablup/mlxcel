# Technical Report: PR #2098 - HIP ports of the fused MoE decode kernels

**Date**: 2026-10-05

**Status**: Implemented and measured on the gfx1151 host; head `eade24a1` on origin/main `57d8ed29`, PR open, pending merge.

**Languages**: C++ (HIP kernel sources through `fast::hip_kernel`, port tables, predicates), Rust (backend gates in `switch_layers.rs` and `nemotron_h.rs`, FFI, parity tests), Bash (idle-GPU guard), Python (guard unit test), Markdown, TSV/CSV (logit traces, bench rows)

**Risk Level**: Medium (ROCm single-token MoE decode now runs new GPU kernels by default on every SwitchGLU family, and Nemotron-H on ROCm changes path. Metal is untouched; on CUDA the only behavior change is that `MLXCEL_FUSED_MOE_RELU2` declines instead of panicking. Wave64 parts are untested)

## Executive Summary

Issue #2065 (item 5 of #1814, ranked second by the #2086 profile) asked for HIP ports of the two fused decode-MoE kernels. Before this PR, `moe_gateup_ports()` and `moe_down_ports()` had `.rocm = nullptr`, and both Rust gates in front of them read the backend-wide `custom_kernels_available()`, which answers false on ROCm, so ROCm decode ran `gather_qmm` for every routed expert.

The PR adds `MOE_GATEUP_HIP_SOURCE` and `MOE_DOWN_HIP_SOURCE`, translated from the CUDA kernels with `__shfl_down(v, o, 32)` for the lane fold, fills both `.rocm` slots, and replaces the backend-wide gate with two predicates that read the port tables (`fused_moe_kernels_available()`, `moe_down_kernel_available()`). On ROCm builds the rows-per-threadgroup default `MLXCEL_FUSED_MOE_SGY` becomes 2 instead of 8, because at 8 the port was slower than `gather_qmm` on gfx1151.

On gfx1151, Qwen3-30B-A3B-4bit decode goes from a median of 61.29 to 62.51 tok/s (+2.0%, every after run above every before run), which fits the #2086 estimate that only the activation, combine and dispatch overhead is recoverable. Nemotron-3-Nano moves +1.0% (near noise) and Mixtral, whose experts are above the Dff cap, is unchanged. Logit traces show 0 decided-position mismatches in every comparison, and Qwen3's single-token logits are now closer to Metal's fused kernel (1/128 top-1 disagreements) than ROCm `gather_qmm` was (5/128).

Two side findings were fixed or documented along the way: `scripts/rocm_gpu_guard.sh` was rejecting most measurement attempts because it counted the command's own exited children as foreign GPU tenants, and the wave32 `#error` guard that the #1814 ports carry is inert with AMD clang 23.

## 1. Problem Statement

### 1.1 The fused MoE path was unreachable on ROCm

The fused two-kernel decode MoE path (a gate-up kernel that writes the activated intermediate, then a down kernel that folds the score-weighted expert outputs) had Metal and CUDA ports only. Two independent things kept ROCm off it:

- the port tables had no `.rocm` entry, so the launcher would refuse;
- `fused_moe_enabled()` in `switch_layers.rs` and the `use_fused` decision in `nemotron_h.rs` both read `custom_kernels_available()`, which is deliberately false on ROCm (#1803). Filling the table alone would not have opened the gate.

The #2086 profile put fused-MoE-reachable work at 46.8% of Qwen3-30B-A3B decode GPU time, but 42.1 points of that are expert GEMVs already near the host's bandwidth (about 181 GB/s), so its estimate of what a fused kernel could recover was about 5% of GPU time plus about 430 dispatches per token.

### 1.2 Stale documentation

`README.md`, `docs/installation.md` and `docs/environment-variables.md` still said affine MoE models abort on ROCm and need `MLXCEL_FUSED_MOE=0`, which no longer matched the code (the backend-wide gate already kept ROCm on `gather_qmm`) and after this PR is wrong in the other direction, because the kernels run.

## 2. Change Summary

Seven commits:

- **`d3801805`** `update(rocm): port the fused MoE decode kernels to HIP`: the two HIP sources and holders, both `.rocm` slots, the two predicates and their FFI, the gate changes in `switch_layers.rs` and `nemotron_h.rs`, the `MLXCEL_FUSED_MOE_RELU2` decline, the test gates and the new SwiGLU parity case.
- **`721ff14a`** `docs(rocm): state what the fused MoE wave32 guard actually covers`: comments and env doc corrected after the `hipcc -E -dM` check (section 5).
- **`4ab91f82`** `fix(bench): stop rocm_gpu_guard rejecting the command's own exited children`: the guard fix and its unit test (section 4).
- **`359f67ed`** `update(rocm): default the fused MoE SGY to 2 on ROCm`.
- **`826dfa8c`** `fix(rocm): pick the fused MoE SGY default from the build flag`: replaces a runtime backend comparison that `make verify-kernel-port-dispatch` rejected.
- **`77afbb1d`** `docs(rocm): list the SSM update step among the HIP-ported kernels`: README wording after the rebase onto #2099.
- **`eade24a1`** `docs(rocm): publish the fused MoE decode and logit results on gfx1151`: `docs/benchmark_results/rocm-fused-moe-gfx1151-2026-10-05.md`, the bench CSVs and the traces under `benchmarks/logit_traces/rocm_gfx1151_77afbb1d/`.

### 2.1 The kernels

Both HIP sources are the CUDA bodies with one change: the warp shuffle. HIP's `__shfl_down_sync` is a compatibility shim that ignores its mask, so, following the bitlinear port (#1862), the native `__shfl_down(var, delta, 32)` is used with the width stated. Inputs, outputs, grid (one 32-lane wavefront per output row on `threadIdx.x`, `sgy` rows per block on `threadIdx.y`, the expert slot on `grid.z`) and template args are identical, so the hipRTC cache key carries `T` the same way the CUDA key does. `expf` and `tanhf` stay the precise library calls (hipRTC compiles at -O3 without fast-math).

The down kernel keeps the CUDA kernel's 6-bit branch (four weights per three bytes, laid out as in `quantized.h`), its f32 partials with the score folded in f32, and the single final rounding to the activation dtype after the K-sum that #886 introduced. One HIP down kernel serves all three callers of `moe_down_ports()`: SwitchGLU and GeGLU through `run_fused_moe_two_kernel`, and Nemotron-H's opt-in squared-ReLU path through `moe_down_kernel_fn()`.

### 2.2 Gates that read the port tables

`fused_moe_kernels_available()` is `has_kernel_port(moe_gateup_ports()) && has_kernel_port(moe_down_ports())`; `moe_down_kernel_available()` reads the down table alone. Because `select_kernel_port` reads the same tables, a gate and the dispatch cannot disagree (the #1801 rule). `switch_layers.rs` gates every SwitchGLU family on the first, which includes qwen3_moe, qwen3_vl_moe, qwen3_next and gemma4; `nemotron_h.rs` gates on the second. Neither calls `custom_kernels_available()` any more. The doc comments in `lib.rs` and `gpu_backend.h` that said ROCm has no ports now say that backend-wide checks still treat ROCm like `None` and a ported kernel's own predicate opens it.

For Nemotron-H this changes the ROCm path from `forward_nonfused` to `fused_moe_forward`. Its default branch still runs `gather_qmm` for the routed experts; only the surrounding combine changes. The term stays in the gate even though the default branch launches no custom kernel, so a kernel added to that branch later cannot silently widen the gate to a backend without its port.

### 2.3 `MLXCEL_FUSED_MOE_RELU2` declines instead of refusing

The opt-in squared-ReLU branch now also requires `has_kernel_port(moe_fc1_relu2_ports())` and `has_kernel_port(moe_down_ports())`. The fc1 kernel has a Metal port only (the ROCm port is #2069), so on ROCm and CUDA the flag now leaves the default `gather_qmm` branch in place. On CUDA that replaces a panic at the `.expect` in `nemotron_h.rs`; Metal is unchanged.

### 2.4 Test gates

`gpu_backend_or_skip()` in `fused_moe_parity_tests.rs` now skips only when there is no GPU backend; on Metal, CUDA or ROCm it asserts `fused_moe_kernels_available()` and fails if it is false, so a missing port is reported rather than skipped past. The qwen3_moe and qwen3_vl_moe positive controls read the predicate instead of `custom_kernels_available()`. A new case, `fused_moe_swiglu_kernel_matches_references_every_down_width`, runs SwiGLU at 4/4, 8/8 and 4/6 bits on the Qwen3 shape, which reaches the SwiGLU arm and the 6-bit down branch. Tolerances are unchanged. A source check asserts the HIP kernels reduce with `__shfl_down`.

## 3. Measured Results

### 3.1 Method

gfx1151 (Radeon 8060S, RDNA 3.5), ROCm 10.0.0, HIP 7.15.26333, MLX pin `81ba1c6a`, overlay `75915908`. Before: origin/main `57d8ed29`. After: the branch at `77afbb1d` on that main. `scripts/bench_decode.sh` at pp512/tg128, before and after binaries alternated run by run, three rounds, every run through the fixed guard; contended attempts were rejected and rerun.

Main moved mid-run to `57d8ed29` (#2099, the HIP `ssm_update` port), which changes Nemotron-H decode. The branch was rebased, the baseline rebuilt at `57d8ed29`, and every decode and Nemotron logit measurement was redone against it, so before and after differ by this PR only.

### 3.2 Decode throughput

| Model | Path change | Before tok/s (median) | After tok/s (median) | Change |
|---|---|---|---|---|
| Qwen3-30B-A3B-4bit | `gather_qmm` to the fused HIP pair | 61.29 / 61.05 / 61.83 (61.29) | 62.51 / 62.68 / 62.16 (62.51) | +2.0% |
| Nemotron-3-Nano-30B-A3B-4bit | `forward_nonfused` to `fused_moe_forward` (still `gather_qmm`) | 74.30 / 74.62 / 74.37 (74.37) | 75.11 / 75.09 / 74.88 (75.09) | +1.0%, near noise |
| Mixtral-8x7B-Instruct-v0.1-4bit | none (Dff 14336 above the 8192 `MLXCEL_FUSED_MOE_MAX_DFF` default) | 9.42 / 9.08 / 10.91 (9.42) | 9.51 / 9.57 (two runs) | +1.3%, noise (same code) |

The Qwen3 gain is consistent (every after run above every before run) and within the #2086 ceiling: the pair removes the activation, weighted-sum and index-building dispatches but reads the same expert weights. Mixtral's third after run never found a clean GPU window; since the code it runs is identical, its 9.08 to 10.91 spread is host noise for a 26 GB checkpoint on this host. `compare_bench_csv.py --allow-commit-change` on the per-model medians reports a 1.016x median across the three pairs, none moved by more than 10%. Prefill is not on the fused path.

### 3.3 The SGY default

The first implementation ported the CUDA kernels verbatim with Metal's SGY default of 8. A guarded pair measured Qwen3 at 61.64 tok/s on `gather_qmm` (`c0b71344`) against 59.42 fused: the port was slower than the fallback it replaced. A sweep of `MLXCEL_FUSED_MOE_SGY` over 1, 2, 4, 8, 16 and 32 on one binary put 2 on top in every repetition (about 62.3 tok/s, against about 61 for `gather_qmm`, about 61.2 for SGY 1 and 59.3 to 60.4 for the rest). All nine guarded attempts of that sweep were rejected as contended (another unit's GPU process appeared in at least one sample), so the sweep is indicative only; the table in 3.2, measured at the new default, is the evidence. SGY shapes the threadgroup only, and `fused_moe_geglu_kernel_bitwise_invariant_across_sgy` pins that the output does not depend on it.

### 3.4 Logits

`compare_logit_traces.py --decided 2.0`, new traces under `benchmarks/logit_traces/rocm_gfx1151_77afbb1d/`. The fused kernels run only on a single-token forward, so `w8` traces never reach them (Qwen3 `default` and `fused0` at `w8` are identical); `w1` is the window that exercises the kernels, and Nemotron-H `w1ctx512` covers decode with state.

| Model, window | Reference | Candidate | Top-1 disagreement | Decided mismatches |
|---|---|---|---|---|
| Qwen3-30B-A3B, w1 | Metal fused | ROCm fused (HIP) | 1 / 128 | 0 / 74 |
| Qwen3-30B-A3B, w1 | Metal `gather_qmm` | ROCm `gather_qmm` | 5 / 128 | 0 / 70 |
| Qwen3-30B-A3B, w1 | ROCm `gather_qmm` | ROCm fused | 3 / 128 | 0 / 74 |
| Qwen3-30B-A3B, w8 | Metal default | ROCm default | 17 / 640 | 0 / 319 |
| Nemotron-3-Nano, w1ctx512 | ROCm main | this branch | 3 / 128 | 0 / 68 |
| Nemotron-3-Nano, w1ctx512 | Metal M5 | this branch | 6 / 128 | 0 / 71 |
| Nemotron-3-Nano, w8 | ROCm `c5fe9a16` (`forward_nonfused`) | this branch | 20 / 640 | 0 / 286 |
| Nemotron-3-Nano, w8 | Metal M5 | this branch | 22 / 640 | 0 / 287 |

Every row has zero decided-position mismatches. Nemotron-H's path change moves a few undecided tokens, at reference gaps of 0.5 or less. The `gather_qmm` path is unchanged: the Qwen3 `fused0` `w1` and Mixtral `w1` traces are byte-identical in every data row to `rocm_gfx1151_bec64748`, and Qwen3 `fused0` `w8` is byte-identical to a trace built on `57d8ed29`. Nemotron-3-Nano generates the same 40-token greedy text with and without `MLXCEL_FUSED_MOE_RELU2=1`.

## 4. The Guard Bug Found While Measuring

`scripts/rocm_gpu_guard.sh` (added in #2086) samples `/sys/class/kfd/kfd/proc` at 1 Hz and rejects an attempt if any GPU holder is not the command or a descendant. KFD removes a process's entry from deferred work after the process is reaped, so for a moment the entry names a pid with no `/proc` directory. The next sample then saw the command's own exited child (the `rocminfo` probe in `bench_decode.sh`, or the bench binary itself), `descends_from` could not read its parentage, and the guard counted it as foreign. Most attempts were rejected this way, in this unit and in the parallel #2067 unit.

The fix skips holders whose `/proc/<pid>` is gone: an exited, reaped process is not using the GPU. `test_a_kfd_entry_left_by_an_exited_process_is_not_contention` in `tests/test_rocm_decode_profile.py` reproduces the window with a fake KFD directory; it fails on the old guard (exit 75, `CONTENDED`) and passes on the new one. A second effect, two guards started in lockstep rejecting each other, was handled operationally (a longer idle window, waiting) and not in code.

## 5. The Wave32 Guard Is Inert

The #1814 HIP ports carry `#error` guards on `__AMDGCN_WAVEFRONT_SIZE` and `__AMDGCN_WAVEFRONT_SIZE__`. `hipcc -E -dM` for gfx942 and gfx1151 showed that AMD clang 23 (HIP 7.15) defines neither spelling, so the guard never fires and a wave64 target would compile. What keeps the kernels correct there is the explicit shuffle width: `__shfl_down(v, o, 32)` keeps each 32-lane fold inside one row on a 64-lane wavefront. The first draft of the env doc said a wave64 target fails to compile; that was corrected, and the docs now say wave64 is untested and `MLXCEL_FUSED_MOE=0` keeps such a host on `gather_qmm`. The same holds for the earlier bitlinear port, which this PR does not change.

## 6. Technical Decisions

- **Gate on the port tables, not on the backend.** A backend-wide predicate cannot express "ROCm has this kernel but not that one", which is the state #1814 produces port by port. Reading the same tables as `select_kernel_port` makes the gate and the dispatch agree by construction.
- **Translate the CUDA kernel, do not redesign it.** Same geometry, template args and numerics (6-bit branch, f32 partials, one final rounding) keeps the parity bounds and the #886 fix valid unchanged, and leaves performance work as a separate, measurable step.
- **Pick the ROCm SGY default at compile time.** The first attempt compared `gpu_kernel_backend()` at runtime in the launcher; `make verify-kernel-port-dispatch` rejects runtime backend comparisons outside `select_kernel_port`. A ROCm build has no Metal or CUDA backend, so `MLXCEL_BRIDGE_ROCM_BACKEND` is the backend and satisfies the gate. Metal and CUDA keep 8.
- **Decline rather than refuse on a partial port set.** `MLXCEL_FUSED_MOE_RELU2` falls back to `gather_qmm` when fc1 or down is missing, which also removes a latent CUDA panic.
- **Fail, do not skip, on a GPU backend without the ports.** Every GPU backend now has both ports, so a false predicate there is a defect the parity tests should report.
- **Rebuild the baseline when main moved.** #2099 changed Nemotron-H decode; comparing against the old baseline would have credited this PR with #2099's effect.

## 7. Validation

On gfx1151:

- `make verify-rocm` on the final rebased head: 146 suites, 11857 passed, 0 failed, 378 ignored (head `eade24a1`). On the pre-rebase head `beb8f77c`: 146 suites, 11852 passed, 0 failed, 378 ignored.
- `fused_moe_parity_tests`: 5 passed. Fused against the all-f32 reference at normalized RMS at most 3.0e-6; fused against `gather_qmm` about 3.7e-3; `gather_qmm` against the reference about 3.3e-3. Negative control: starting the down kernel's lane fold at 8 instead of 16 fails both reference tests (normalized RMS 0.63 and 0.70).
- `qwen3_moe::tests` and `qwen3_vl_moe::tests`: 12 passed, with the positive controls running the kernel.
- `switch_layers` and `nemotron_h` lib tests: 93 passed.
- `cargo clippy -D warnings` on `mlxcel-core` and `mlxcel`: clean. The fast gates pass, and the dtype-key pin is unchanged at 9 in scope.
- Guard unit test: fails on the old guard, passes on the new one.

A static review found no CRITICAL or HIGH issues. The MEDIUM item was the wave64 concern, answered by the finding in section 5 (no hard failure; untested). The LOW items were fixed: the `MLXCEL_FUSED_MOE_MAX_DFF` row lists ROCm, and the stale "ROCm has no ports" comments in `lib.rs` and `gpu_backend.h` were rewritten.

## 8. Residual Risks and What Was Not Verified

- **Metal and CUDA not run.** No hardware on this host. Their `.metal` and `.cuda` entries and sources are untouched and both predicates answer true there. The CUDA-visible change is the `MLXCEL_FUSED_MOE_RELU2` decline; the SGY default change is compiled into ROCm builds only.
- **Gemma 4 and Qwen3-Next not run.** They reach the same kernels through the same gate (GeGLU for Gemma 4), but no checkpoint was available. The GeGLU arm is covered only by the parity tests.
- **Wave64 (CDNA) untested.** The compile-time guard does not protect it (section 5); correctness there rests on the explicit shuffle width.
- **One device.** All throughput and the SGY choice come from gfx1151. The SGY sweep itself was contended in every attempt and is indicative only; another RDNA or CDNA part may want a different default.
- **Small gains near noise.** Nemotron-H's +1.0% sits within its run-to-run spread, and Mixtral has only two after runs.
- **The port is naive.** It is a direct CUDA translation without vectorized loads or `x` in shared memory. Because expert GEMVs are bandwidth-bound, the headroom is expected to be small, but it was not measured.

Recommended follow-ups: vectorized loads or `x` in shared memory for the HIP MoE kernels; a wave-size check for all ROCm ports that does not rely on the inert macro (for example a host-side `warpSize` check in port selection); routing granite's `block_sparse_moe` through `forward_fused_kernel` (noted in #2086); the Nemotron-H loader printing to stdout.

## 9. Learning Points

- **A port is not reachable until its gate is.** Filling `.rocm` changed nothing while the callers read a backend-wide predicate. Per-kernel predicates over the port tables are what make partial port coverage work.
- **A verbatim port can be slower than the fallback.** The CUDA launch shape tuned on other hardware lost to `gather_qmm` on gfx1151 until SGY changed from 8 to 2. A before/after measurement belongs in the port PR, not after it.
- **Choose the trace window that reaches the code.** The fused kernels run only on a single-token forward, so `w8` traces would have compared identical paths. `w1` (and `w1ctx512` for stateful decode) is where the change is visible.
- **Check that a compile-time guard can fire.** The wave32 `#error` looked like protection; `hipcc -E -dM` showed the macro is never defined.
- **Measurement tooling has bugs too.** The guard's false rejections cost time across two units and came from a kernel teardown order, not from real contention. A regression test that reproduces the window keeps it fixed.

Refs: #2065, #1814, #2086, #2099, #2069, #2067, #1862, #1803, #1885, #1801, #886.
