# Technical Report: PR #2099 - Port ssm_update_kernel to HIP and read the port table

**Date**: 2026-10-04

**Status**: Implemented and measured on the gfx1151 host; head `2f17f286` on origin/main `c0b71344`, pending merge.

**Languages**: C++ (HIP kernel source, kernel holder, support predicate, host shape checks), Rust (FFI doc comments, parity tests), Markdown (environment variables, installation, benchmark results), CSV/TSV (bench rows, logit traces)

**Risk Level**: Medium (it changes the default single-token decode path of the hybrid SSM models on ROCm, and touches the predicate, the template arguments and the host checks that Metal and CUDA share, neither of which could be run on the development host)

## Executive Summary

Issue #2067 (part of #1814, epic #1801) is the first port in the order that PR #2086 measured: the single-token Mamba2 SSM update, which the decode profile put at 29.8% of granite-4.0-h-tiny's and 20.0% of Nemotron-H's decode GPU time on gfx1151. Before this PR every such step on ROCm ran the ~55-op SSD graph (`ssm_step`). `ssm_ports()` had no `.rocm` entry, and `ssm_kernel_available()` did not read the table at all, so a filled slot would have stayed unreachable.

The PR does four things in `src/lib/mlxcel-core/cpp/mlx_cxx_kernels.cpp`. It adds `SSM_HIP_SOURCE`, the CUDA kernel body with the lane fold rewritten as `__shfl_down(acc, o, 32)`, and fills `.rocm`. It makes `ssm_kernel_available()` return `has_kernel_port(ssm_ports())` on every platform, with a new `MLXCEL_SSM_KERNEL=0` switch and the old `MLXCEL_SSM_CUDA_KERNEL=0` kept as an alias. It adds the `A_log`, `B` and `C` dtypes to the launch's template arguments on every backend, because Nemotron-H stores `A_log` in f32 beside bf16 activations and the JIT caches key on template arguments. It refuses on the host, as an error, the shapes the kernel cannot index.

On gfx1151, with the same binary and `MLXCEL_SSM_KERNEL=0` as the "before" arm, decode went from 60.39 to 88.43 tok/s on granite-4.0-h-tiny-4bit (1.46x) and from 51.35 to 74.37 tok/s on NVIDIA-Nemotron-3-Nano-30B-A3B-4bit (1.45x), medians of three runs. The `w1ctx512` logit traces, where every traced step meets a 512-token SSM state, now compare ROCm's kernel with Metal's kernel on M5 and find 0 of 60 (granite) and 0 of 71 (Nemotron-H) decided-position mismatches. Greedy 128-token output with and without the kernel is coherent on both models but not byte-identical; the traces put every flip at an undecided position.

One finding goes beyond this issue: the wave32 `#error` guard that every #1814 port is required to carry is inert with ROCm 10's AMD clang 23, which defines neither `__AMDGCN_WAVEFRONT_SIZE__` nor `__AMDGCN_WAVEFRONT_SIZE` for gfx1151, gfx942 or gfx90a. This port does not depend on it, but the idiom guards nothing for the ports that follow.

## 1. Problem Statement

### 1.1 An empty slot behind a predicate that never asked

Hybrid SSM models (granite-4.0-h, falcon-h1, plamo-2, Nemotron-H) gate their fused single-token path with `seq_len == 1 && ssm_kernel_available()` in `granitemoehybrid.rs`, `falcon_h1.rs`, `plamo2.rs` and `nemotron_h.rs`, and then `expect` the launch. Before this PR the predicate was:

```cpp
#ifdef __APPLE__
    return mlx::core::metal::is_available();
#else
    // MLXCEL_SSM_CUDA_KERNEL=0 forces the graph path
    ...
    return mlx::core::cu::is_available();
#endif
```

On ROCm that answered false, so the gates took the graph. Filling `.rocm` alone would have changed nothing, and filling it while the predicate stayed backend-enumerated is the shape `verify-kernel-port-dispatch` exists to prevent: a predicate and a dispatch that can disagree.

### 1.2 What the profile said it was worth

PR #2086 measured the SSM update as the largest recoverable share of any #1814 port on gfx1151: 29.8% of granite-4.0-h-tiny's and 20.0% of Nemotron-H's decode GPU time, and more than half of their dispatches. The CUDA port (#631) had measured 2.6-4.5x decode on the granite/falcon family on GB10.

### 1.3 No kernel-level parity test existed

Parity for the CUDA port was end-to-end greedy output on CUDA. Nothing compared the kernel with the graph step on the same inputs, so a reduction bug that returns finite, plausible values (the failure the wave32 rule in #1814 describes) would only show up as drifting model output.

## 2. Change Summary

| Area | Change |
|---|---|
| `mlx_cxx_kernels.cpp`, kernel | `SSM_HIP_SOURCE`, `SsmKernelHolderHip` / `get_ssm_kernel_hip()` calling `fast::hip_kernel` under `MLXCEL_BRIDGE_ROCM_BACKEND`; `.rocm` getter in `ssm_ports()` |
| `mlx_cxx_kernels.cpp`, predicate | `ssm_kernel_available()`: kill switches, then false off the GPU device, then `has_kernel_port(ssm_ports())` |
| `mlx_cxx_kernels.cpp`, launch | `TA`, `TB`, `TC` template arguments (dtypes of `A_log`, `B`, `C`) on every backend |
| `mlx_cxx_kernels.cpp`, host | Rank and shape checks in `ssm_update_kernel` that throw `std::invalid_argument` before any launch |
| `ssm_update_parity_tests.rs` (new) | Five tests: f32 and bf16 parity at three shapes, the `A_log` dtype key, shape refusals, both kill switches |
| `mlx_cxx_bridge.h`, `lib.rs` | Doc comments for the predicate and the kernel; test module registration |
| Docs | `MLXCEL_SSM_KERNEL` row and alias row in `docs/environment-variables.md`; ROCm row in `docs/installation.md`; the "Open" item in `rocm-correctness-gfx1151-2026-09-30.md` now points to the kernel-against-kernel rows; new `rocm-ssm-update-kernel-gfx1151-2026-10-04.md` |
| Data | `benchmarks/rocm_strixhalo-gfx1151_2026-10-04_ssm-kernel-{off,on}.csv`; six traces plus `METADATA.txt`, `RUNS.txt`, `SHA256SUMS` and `README.md` in `benchmarks/logit_traces/rocm_gfx1151_96cbce84/` |

The branch has six commits: the port and predicate (`96cbce84`), the dtype keys on every backend (`844bd94c`), the docs (`3780b0b2`), the CPU-device check and review fixes (`bd2e8f31`), the host shape refusal (`38071038`), and more refusal tests plus test docs (`2f17f286`).

## 3. The Port

### 3.1 CUDA body, HIP fold

The HIP source keeps the CUDA port's kernel name pattern, inputs (`X`, `A_log`, `B`, `C`, `D`, `dt`, `state_in`), outputs (`out`, `state_out`), grid and template arguments. Each (head, row) pair is reduced by the 32 lanes that share a `threadIdx.y` in a (32, 8, 1) threadgroup; each lane walks `Ds / 32` state elements, updates the state, and accumulates `state * C`. The only body change is the fold:

```cpp
for (int o = 16; o > 0; o >>= 1) {
    acc += __shfl_down(acc, o, 32);
}
```

`__shfl_down_sync` exists in HIP only as a compatibility shim that ignores its mask, so the native `__shfl_down` is used and the width is stated. The width is what makes the fold correct regardless of wave size: on a wave32 target (gfx11, gfx12) the 32 lanes are the whole wave, and on a wave64 target (CDNA) a width of 32 splits the wave into two 32-lane segments, which are again exactly two rows. The early return for rows past `Dh` is wave-uniform because all 32 lanes of a wave share `threadIdx.y`.

### 3.2 The wave32 guard, and why it does nothing here

The #1814 port requirements tell every port that reduces across lanes to carry two `#error` checks, on `__AMDGCN_WAVEFRONT_SIZE__` and on `__AMDGCN_WAVEFRONT_SIZE`, as the bitlinear port does. The port carries them. During the work it turned out that ROCm 10's AMD clang 23 defines neither macro for gfx1151, gfx942 or gfx90a, so the `#if defined(...)` conditions are false on every target and the guard cannot fire. The comment in `SSM_HIP_SOURCE` and the benchmark doc both say so. The fold above is correct on wave64 by its width argument, not because of the guard; see section 9 for what that means for the remaining ports.

## 4. The Predicate and the Switch

`ssm_kernel_available()` now reads, in order:

1. `MLXCEL_SSM_KERNEL=0` or `MLXCEL_SSM_CUDA_KERNEL=0` returns false. The new name says what the switch does on every backend; the old one is kept so existing A/B scripts still work.
2. A default device other than the GPU returns false, the same check `mamba1_scan_kernel_accepts` makes. Custom kernels run only on the GPU stream, so with `MLXCEL_DEVICE=cpu` the gates take `ssm_step` on the CPU instead of the launch throwing.
3. Otherwise `mlxcel::has_kernel_port(ssm_ports())`, which reads the same table `select_kernel_port` reads in the dispatch. A true answer therefore means the dispatch will not refuse, which is what lets the Rust gates `expect` the launch.

Two behavior changes follow for the other backends, and the docs state both. The alias now also affects Metal and ROCm (it used to be read on the non-Apple path only), and `MLXCEL_SSM_KERNEL=0` is honored on Metal, where previously nothing turned the kernel off. With the environment unset, `has_kernel_port` is true on Metal and CUDA as before.

No model file changed. On Nemotron-H the gate in `NemotronHMamba2Mixer::forward` selects `fused_mamba2_forward`, which runs the whole single-token mixer (input projection, convolution, this kernel, gated norm, output projection) as one C++ call. Nemotron-H's decode figures therefore measure that path against the Rust graph mixer, not the SSM step alone.

## 5. Dtype Keys on Every Backend

The CUDA and HIP JIT caches key a module on the kernel name plus `template_args`, while the generated kernel signature takes each input's runtime dtype. The launch named `T` (activations, also covering `X` and `D`) and `U` (state) only. Nemotron-H stores `A_log` in f32 next to bf16 activations, granite in bf16, so wherever both reach the same cached module (one process, as in the parity test) whichever compiled it first fixes the pointer type the other reads `A_log` through. `B` and `C` are not tied to `T` either.

The fix adds `TA`, `TB`, `TC` for the `A_log`, `B` and `C` dtypes. The kernel bodies never name them, so only module names change, not arithmetic. They are added on every backend, Metal included (where the name already keys on every input dtype, so the addition is redundant there). The first version (`96cbce84`) appended them inside `if (mlxcel::gpu_kernel_backend() == mlxcel::GpuKernelBackend::Rocm)`, to leave the Metal and CUDA module names unchanged. That is what rule 1 of `scripts/ci/check_kernel_port_dispatch.py` forbids: a file that launches a custom kernel must not branch on the backend kind outside `kernel_port.cpp` and `gpu_backend.cpp`, because hand-written per-backend branches are how the launchers drifted before `select_kernel_port` existed. The second commit (`844bd94c`) moved them to every backend. The traces were produced at `96cbce84`; on ROCm the launch, its arguments and its cache key are the same in both commits, which `METADATA.txt` records.

`ssm_update_kernel_keys_on_a_log_dtype` runs the same shape with `A_log` first in bf16 and then in f32 in one process. Without the keys the second launch reuses the bf16 module and the test fails at normalized RMS 0.80.

## 6. Host-Side Shape Refusal

The kernel takes `Dh`, `Ds`, `H` and `G` as template constants and indexes raw buffers with them. A checkpoint config the graph path would reject inside `reshape` or `repeat` would instead read or leave unwritten GPU memory. `ssm_update_kernel` now throws `std::invalid_argument` before any launch when:

- `hidden_states` or `B` is not rank 4;
- `C`'s shape differs from `B`'s;
- there is more than one token, or `B`'s batch differs from the activations';
- groups are not positive or heads are not divisible by groups;
- the state width is below 32 or not a multiple of 32;
- the element count of `A_log`, `D`, `dt` or `state_in` does not match the shape.

The bridge returns the throw to Rust as an error. The checks sit in front of the shared launch, so they apply to every port; the HIP port is what made them reachable on ROCm. `ssm_update_kernel_refuses_unsupported_shapes` pins them.

## 7. Measured Results

### 7.1 Decode throughput

`scripts/bench_decode.sh` at pp512/tg128, one model per run, three runs per arm with the arm order alternated, at `844bd94c`, same binary.

| Model | Graph (`MLXCEL_SSM_KERNEL=0`) tok/s | Kernel tok/s | Speedup (medians) |
|---|---|---|---|
| granite-4.0-h-tiny-4bit | 60.57 / 60.39 / 60.05, median 60.39 | 88.36 / 88.69 / 88.43, median 88.43 | 1.46x |
| NVIDIA-Nemotron-3-Nano-30B-A3B-4bit | 51.65 / 50.87 / 51.35, median 51.35 | 74.87 / 72.87 / 74.37, median 74.37 | 1.45x |

Run-to-run spread is under 1% for granite and under 3% for Nemotron-H in either arm. `compare_bench_csv.py --before <off> --after <on>`, which keeps the last row per model, reports 1.47x and 1.45x. Prefill is not on this path and is not compared; it varied between 379 and 608 tok/s for granite across both arms on the shared host.

The gain is smaller than the CUDA port's 2.6-4.5x on GB10. It is also larger than the decode profile's GPU-time shares alone would suggest: removing a 29.8% or 20.0% share entirely is about 1.42x and 1.25x of GPU time. The profile was taken at a different commit under a profiler that itself costs 20 to 23% on these models, and Nemotron-H's fused path replaces the whole mixer rather than the SSM step, so this is a consistency check on the published shares, not a decomposition of the speedup; no measurement in this PR attributes the excess (host-side dispatch of the ~55-op graph is the obvious candidate).

### 7.2 Idle-GPU guard

Every run went through `scripts/rocm_gpu_guard.sh` (90 s with `/sys/class/kfd/kfd/proc` empty and no compiler running, then 1 Hz monitoring). A parallel development unit shared the GPU; every attempt that overlapped its work was rejected and rerun. All runs but one used the copy of the guard fixed in #2098 (commit `c3eab7b2`), run from the #2065 unit's worktree and not committed in this PR; that fix stops the guard from rejecting the command's own exited children. The exception is the first run (granite, graph arm, r1), which used the copy on `main`. The old guard's defect produced false rejections, never false acceptances, so that run's acceptance holds.

### 7.3 Kernel against the graph, one step

`ssm_update_parity_tests` compares output and new state with a float32 MLX-op reference of the one-token SSD step:

| Shape | Batch | Heads x head dim | Groups | State |
|---|---|---|---|---|
| granite-4.0-h-tiny | 1 | 48 x 64 | 1 | 128 |
| Nemotron-H (dt clipped to `(1e-3, 0.1)`) | 2 | 64 x 64 | 8 | 128 |
| padded rows | 1 | 4 x 60 | 2 | 64 |

Tolerances (normalized RMS / max): f32 1e-5 / 1e-4, bf16 1.6e-2 / 7e-2, from the issue. The state is held to the f32 budget in every case because the models carry it in f32. The tests pass on gfx1151. Two mutations show the tests can tell: a lane fold starting at 8 instead of 16 fails at normalized RMS 0.65 to 0.89, and dropping the dtype keys fails at 0.80. The tests return early only when there is no GPU backend, the default device is the CPU, or a kill switch is set; on Metal, CUDA and ROCm a false predicate fails the test instead of skipping it.

### 7.4 Model logits: kernel against kernel (`w1ctx512`)

The `w1ctx512` traces prefill 512 corpus tokens and then trace single-token steps, so every traced step meets a state and takes the fused step. Until this PR the corresponding rows in `rocm-correctness-gfx1151-2026-09-30.md` compared Metal's kernel with ROCm's graph. Now the comparison is kernel with kernel. `compare_logit_traces.py --decided 2.0`, Nemotron-H files filtered of their `[NemotronH]` loader lines.

| Model | Reference | Candidate | Top-1 disagreement | Decided mismatches | Largest gap | Perplexity ref / cand |
|---|---|---|---|---|---|---|
| granite-4.0-h-tiny | Metal kernel (`metal_m5_3c9edea0`) | ROCm kernel | 3 / 128 | 0 / 60 | 0.125 | 7.090 / 7.216 |
| granite-4.0-h-tiny | Metal kernel | ROCm graph (`ssmkernel0`) | 2 / 128 | 0 / 60 | 0.125 | 7.090 / 7.290 |
| granite-4.0-h-tiny | ROCm graph (`ssmkernel0`) | ROCm kernel | 1 / 128 | 0 / 59 | 0.000 | 7.290 / 7.216 |
| granite-4.0-h-tiny | ROCm graph at `3c9edea0` | ROCm kernel | 1 / 128 | 0 / 60 | 0.125 | 7.234 / 7.216 |
| nemotron-3-nano-30b-a3b | Metal kernel (`metal_m5_3c9edea0`) | ROCm kernel | 3 / 128 | 0 / 71 | 0.250 | 7.489 / 7.442 |
| nemotron-3-nano-30b-a3b | Metal kernel | ROCm graph (`ssmkernel0`) | 4 / 128 | 0 / 71 | 0.250 | 7.489 / 7.460 |
| nemotron-3-nano-30b-a3b | ROCm graph (`ssmkernel0`) | ROCm kernel | 1 / 128 | 0 / 68 | 0.125 | 7.460 / 7.442 |
| nemotron-3-nano-30b-a3b | ROCm graph at `3c9edea0` | ROCm kernel | 4 / 128 | 0 / 68 | 0.125 | 7.489 / 7.442 |

Every pairing has zero decided-position mismatches, and every top-1 disagreement puts the reference's token at rank 2 or 3 of the candidate at a reference gap of 0.25 or less. The `default` and `ssmkernel0` traces from the same binary differ (different perplexities, one flipped token each), which shows the `default` traces actually took the kernel. Against Metal's kernel the ROCm kernel is as close as the ROCm graph: 3 and 3 top-1 disagreements, against 2 and 4 for the graph here and 4 and 7 in the earlier `3c9edea0` comparison.

The `w1` traces (no prefill) never meet a state and run the graph with or without the kernel. Against `rocm_gfx1151_c5fe9a16` both models agree at every position (granite 0 / 128 top-1, 0 / 116 decided; Nemotron-H 0 / 128 top-1, no decided position). These satisfy the issue's `w1` criterion and show the stateless path is unchanged; they say nothing about the kernel itself, which is why the `w1ctx512` rows carry the correctness argument.

### 7.5 Greedy divergence

`mlxcel generate -p "The history of the Roman Empire" -n 128 -t 0 --no-chat-template`, with and without `MLXCEL_SSM_KERNEL=0`. Both arms of both models produce coherent text, but not byte-identical text, unlike the CUDA port on GB10. Granite parts at the third generated token ("a rich and complex tapestry" against "a fascinating and complex subject"); Nemotron-H after about a dozen tokens. A free-running generation conditions everything after the first flip on different text, so it cannot say whether the kernel is wrong or the model was undecided. The teacher-forced traces answer that: the kernel and the graph disagree on one top-1 token per model, and never at a decided position. The divergence is a change of summation order meeting a near-tie, not an error.

## 8. Technical Decisions

- **Port the CUDA body, not the Metal one.** CUDA and HIP share the launch shape, grid and `template_args`, and `fast::hip_kernel` compiles through hipRTC like `cuda_kernel`. Keeping the bodies line-for-line except the fold leaves one place to look when they disagree.
- **State the shuffle width instead of relying on the guard.** The width-32 fold is correct on wave32 and wave64 by construction. That turned out to matter, because the guard the requirements prescribe does not fire on any ROCm 10 target.
- **Make the predicate read the table, not the backend.** `has_kernel_port(ssm_ports())` cannot disagree with `select_kernel_port`, and a future port that fills a slot becomes reachable with no predicate edit. The `#ifdef __APPLE__` split and the `cu::is_available()` call are gone, which the issue's first acceptance criterion checks.
- **Rename the switch, keep the alias.** `MLXCEL_SSM_KERNEL` describes a switch that now acts on every backend; keeping `MLXCEL_SSM_CUDA_KERNEL` avoids breaking existing A/B scripts. The cost is that the CUDA-named variable now also turns the kernel off on Metal and ROCm, which the docs state.
- **Dtype keys on every backend, not a ROCm branch.** A backend branch at the launch was the first attempt and was rejected by `verify-kernel-port-dispatch`. Adding the keys everywhere is harmless on Metal and closes the same keying gap on CUDA in the same change.
- **Refuse bad shapes on the host, for all ports.** The graph path fails loudly on these shapes; the kernel would fail silently. Throwing `std::invalid_argument` restores the loud failure without a per-backend condition.
- **Keep the CPU-device check in the predicate.** It is the same rule the Mamba1 scan uses, and it keeps `MLXCEL_DEVICE=cpu` on the graph path rather than turning a working configuration into an error.

## 9. Follow-up: the Wave64 Guard Idiom Is Inert on ROCm 10 Clang

The #1814 port requirements prescribe `#if defined(__AMDGCN_WAVEFRONT_SIZE__) && __AMDGCN_WAVEFRONT_SIZE__ != 32` plus `#error`, and the same for `__AMDGCN_WAVEFRONT_SIZE`, so that a port that assumes 32 lanes fails to compile on a wave64 CDNA part instead of returning a finite, wrong result. With ROCm 10's AMD clang 23 neither macro is defined for gfx1151, gfx942 or gfx90a, so both conditions are false and the guard never fires on any of them.

Today that has no effect on correctness: both HIP ports in `mlx_cxx_kernels.cpp` that reduce across lanes (bitlinear and this one) pass an explicit width of 32 to `__shfl_down`, which is correct on wave64. It matters for the ports still to come (#2065, #2064, #2068, #2063, #2069). A port whose correctness does depend on a 32-lane wave would compile cleanly on CDNA, and the requirement text would make a reviewer believe it was protected. The requirement needs either a mechanism that actually fires on ROCm 10 or a rule that every cross-lane operation states its width and that the requirement stop claiming a compile-time guard. This PR does not choose the replacement, and no issue for it was found when this report was written.

## 10. Validation

From the PR body, on gfx1151 (Radeon 8060S, ROCm 10.0.0):

- `make verify-rocm` at `38071038`: OK; 11,856 passed, 0 failed, 378 ignored across 146 test binaries; ROCm smoke OK.
- `cargo test --release --features rocm -p mlxcel-core --lib ssm_update_parity_tests -- --test-threads=1` at `2f17f286` (test and doc changes only since the gate): 5 passed.
- `cargo clippy -p mlxcel-core --lib --tests --features rocm -- -D warnings`: clean. `make verify-fmt verify-kernel-dtype-keys verify-kernel-port-dispatch`: pass, with `EXPECTED_IN_SCOPE` and its count unchanged.
- Greedy output for both models identical before and after the CPU-device and shape checks, so the decode and trace numbers measured at `844bd94c` and `96cbce84` stand for the head.

Orchestrator verification on head `2f17f286`, up to date with origin/main `c0b71344`:

- `make verify-rocm` passed every step: 11,856 tests passed, 0 failed, 378 ignored; smoke OK.

## 11. Residual Risks and What Was Not Verified

- **Metal and CUDA were not run.** Both are touched by the predicate (still `has_kernel_port`, true on both, plus the CPU-device and kill-switch rules), the extra `TA`/`TB`/`TC` template arguments (module names only), and the host shape checks. The parity test is written to run on both and will fail, not skip, if the predicate is false there.
- **The alias widened.** Anyone who set `MLXCEL_SSM_CUDA_KERNEL=0` on a Metal or ROCm machine and forgot about it now gets the graph path there.
- **`MLXCEL_DEVICE=cpu` with a hybrid SSM model was not run.** No test covers the CPU-device rule, because switching the default device inside the shared test binary would move other tests to the CPU.
- **falcon-h1 and plamo-2 were not run on ROCm.** They take the same gate; only granite and Nemotron-H were measured.
- **One device, one session, shared GPU.** All numbers are from gfx1151; one development unit shared the GPU and overlapping attempts were rerun. Prefill varied widely on the shared host and was not compared. No wave64 hardware was available, so the width argument for CDNA is reasoned, not run.
- **The measurement guard is not in this PR.** All runs but one used `c3eab7b2` from #2098, outside this branch; the published method references a fix that lands separately.

## 12. Learning Points

- **A port is not reachable until its predicate reads the table.** Filling `.rocm` without changing `ssm_kernel_available()` would have produced a kernel nothing calls and a green build.
- **A JIT cache key must name every input dtype the signature takes.** A mixed-dtype checkpoint (f32 `A_log`, bf16 activations) is exactly the case where an incomplete key silently reuses the wrong module, and only a same-process test with both dtypes catches it.
- **Prove a guard can fire before relying on it.** The wave32 `#error` looked like protection and compiled everywhere, which is the same symptom as a guard that checks nothing.
- **Mutation-check the parity test.** A fold starting at 8 and a missing dtype key both fail with large deviations, which shows the tolerance is meaningful rather than loose.
- **Greedy text is the wrong instrument for a reordering change.** Free-running output diverges at the first near-tie and stays diverged; teacher-forced traces with a decided-position threshold separate a precision change from a bug.

Refs: #2067, #1814, #1801, #2061, #2086, #631, #1862, #1870, #2059, #2065, #2098, #2064, #2068, #2063, #2069.
