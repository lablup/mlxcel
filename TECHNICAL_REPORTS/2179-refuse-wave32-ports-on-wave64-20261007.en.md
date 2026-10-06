# Technical Report: PR #2179 - Refuse Wave32-Only Kernel Ports on Wave64 Devices

**Date**: 2026-10-07

**Status**: Implemented and verified on the gfx1151 host; head `1dc419c6` (up to date with origin/main `08e22698`), PR open, pending merge. Closes #2147 (part of #1801).

**Languages**: C++ (`turbo/kernel_port.*`, `turbo/gpu_backend.*`, the ROCm overlay, `mlx_cxx_kernels.cpp`, the cxx bridge), Rust (mlxcel-core FFI and tests, Mamba, Jamba, BitNet, two integration tests), Python and shell (`scripts/ci/check_kernel_port_dispatch.py` and its negative test), Markdown (`LOCAL_FIXES.md`, `docs/installation.md`, CONTRIBUTING)

**Risk Level**: Low on RDNA (gfx1151 selects the same ports and produces the same output); medium on CDNA, where ten fused HIP kernels stop being selected and BitNet stops loading. That behavior has not been run on a wave64 device.

## Executive Summary

Every shuffle-based HIP port of an mlxcel fused kernel was written and run on 32-lane wavefronts (RDNA, gfx1151), and each relied on an `#error` guard on `__AMDGCN_WAVEFRONT_SIZE` to stop a 64-lane build. With ROCm 10's AMD clang 23 (HIP 7.15) neither spelling of that macro is defined for gfx1151, gfx942 or gfx90a, so the guard never fires. On a CDNA device (MI200, MI300), which the install guide says compiles, every one of those kernels would have run without anyone having validated it at that width.

PR #2179 moves the protection to the host. `port_for`, the single function that maps a kernel's `KernelPorts` table to a port, now answers "no port" for a ROCm entry unless the table is marked `rocm_any_wave_size` or the device's hardware wavefront is 32 lanes. Because `has_kernel_port`, every `*_available()` predicate and `select_kernel_port` all go through `port_for`, the Rust gates and the C++ launchers agree, and callers take the graph fallbacks they already had. Five tables are marked any-wave (RoPE append, paged merge, Gumbel, rejection, and xIELU, which the issue did not list); ten stay wave32-only. BitNet, whose BitLinear op has no graph fallback, is refused at load on wave64. The CI checker gains two rules that keep the any-wave marks honest and the test seam out of production code, each proven by a negative test.

On gfx1151 nothing changes: all 16 port predicates stay true, the profiler still shows the fused kernels, and greedy output is byte-identical with the fused RMSNorm on and off. The wave64 refusal itself is proven only through a test-only seam, since no wave64 device is available.

## 1. Problem Statement

### 1.1 Why the guard was inert

The #1814 port rule asked that "a wave64 target fails to compile rather than returning a plausible wrong answer", and each port implemented it with the preprocessor:

```c
#if defined(__AMDGCN_WAVEFRONT_SIZE__) && __AMDGCN_WAVEFRONT_SIZE__ != 32
#error "... assumes a 32-lane wavefront"
#endif
```

`static_assert(warpSize == 32)` is not an option in HIP (`warpSize` is an object with an `operator int()`, not a constant expression), so the macro was the only compile-time handle. AMD clang 23 does not define it, in either spelling, for any of the three targets checked with `hipcc -E -dM` (#2065, #2067, rechecked for this PR). The `defined(...)` test is therefore always false and the `#error` never compiles in. The ports' own comments had already said so. This is a failure mode where the check looks present in source but does nothing, so it gave no signal that it had stopped working.

### 1.2 What was at risk

The explicit shuffle width of 32 keeps each butterfly reduction inside 32 lanes, which should hold on wave64, but only if nothing else in a kernel assumes a 32-wide wavefront: lane and warp indices from `threadIdx.x % 32` or `/ 32`, per-warp shared-memory slots, and warp-synchronous steps without a barrier. None of that had been run on a wave64 device (#2107: "Wave64 (CDNA) untested"), and a silent half-fold returns finite, plausible, wrong numbers.

The device's real width was already known to MLX (`hipDeviceProp_t::warpSize`, logged at bind), but nothing on mlxcel's side read it. `Device::warp_size()` could not be used because `MLX_ROCM_FORCE_WARP_SIZE` overrides it for launch-width experiments.

## 2. Change Summary

| Area | Change |
|---|---|
| ROCm overlay `rocm.h`/`rocm.cpp`/`no_rocm.cpp` | New `mlx::core::rocm::device_warp_size()`: `hipGetDevice` plus `hipDeviceGetAttribute(hipDeviceAttributeWarpSize)`, never the override, 0 on failure; stubbed to 0 off ROCm. `LOCAL_FIXES.md` item 32 under a new "Additions to the fork's API" section |
| `turbo/gpu_backend.*` | `constexpr rocm_port_allowed(any_wave_size, warp_size)`; `rocm_port_warp_size()` caching the hardware width once per process with a one-time stderr notice when it is not 32; test seam `set_rocm_port_warp_size_for_tests` |
| `turbo/kernel_port.*` | `KernelPorts::rocm_any_wave_size` (default false); `port_for` holds the ROCm entry; `select_kernel_port` names the width when that is the reason |
| Five port tables | `.rocm_any_wave_size = true` on `fused_rope_ports`, `paged_merge_ports`, `gumbel_ports`, `rejection_ports`, `xielu_ports`; xIELU's inert guard removed |
| `mlx_cxx_kernels.cpp`, bridge | `bitlinear_kernel_available` reads `bitlinear_ports()`; BitLinear's refusal names the BitNet load-time refusal; guard comments name the host-side hold |
| cxx bridge, `lib.rs` | `mamba1_selective_scan -> Result<()>`; new `rocm_port_allowed`, `rocm_device_warp_size`, `set_rocm_port_warp_size_for_tests` |
| Mamba, Jamba, BitNet | Mamba and Jamba `expect` the gated scan launch; BitNet's load refusal message covers wave64 |
| Parity tests, `test_support` | `skip_for_wave32_only_rocm_port` for fused MoE, relu2, SSM, fused norm, Mamba1, add3; paged skip message mentions the hold |
| Checker | Rules 5 and 6 in `check_kernel_port_dispatch.py` (`--root` added); `check_kernel_port_dispatch_test.sh` with 13 cases, run by the make target and CI |
| `build.rs` | `rerun-if-changed` for `kernel_port.h/.cpp` and `gpu_backend.h/.cpp` |
| Tests | `kernel_port_tests.rs` (truth table), `tests/rocm_wave_size.rs`, `tests/rocm_wave64_port_refusal.rs` |
| Docs | `docs/installation.md` "Wave64 devices" row; CONTRIBUTING mentions the hold |

Four commits: the implementation (`9c875c6`), review hardening (`967933e`), a merge of origin/main (`f7d9ec4`), and the LOCAL_FIXES renumbering to 32 after #2164 took 31 (`1dc419c`). 38 files, 1,356 insertions and 71 deletions.

## 3. Design

### 3.1 One hold, in the one place every path reads

`port_for` already mapped the resolved `GpuKernelBackend` to a table entry, and both `has_kernel_port` (behind every predicate and therefore every Rust gate) and `select_kernel_port` (behind every launcher) read it. The ROCm case now returns `ports.rocm` only when `rocm_port_allowed(ports.rocm_any_wave_size, rocm_port_warp_size())` holds. Putting the check here, rather than in each launcher or each predicate, means a predicate and its launcher cannot answer differently on a wave64 device; that disagreement is the class of bug `select_kernel_port` was introduced to remove (#1801). The check runs only when `ports.rocm` is non-null, so a table with no ROCm entry never queries the device.

`rocm_port_allowed` is a pure `constexpr` (`any_wave_size || warp_size == 32`) so it can be tested on every backend; `0`, the value for a failed query, counts as "not 32". When `select_kernel_port` finds no port on ROCm although `ports.rocm` exists, it throws a message naming the width ("validated only on a 32-lane wavefront and this device reports 64 lanes"), so a CDNA user does not read the refusal as a missing port.

### 3.2 `device_warp_size()` and the override it ignores

The width comes from a new overlay call, `mlx::core::rocm::device_warp_size()`, recorded as LOCAL_FIXES item 32. It reads `hipDeviceAttributeWarpSize` for the current device and deliberately ignores `MLX_ROCM_FORCE_WARP_SIZE`: that variable exists to experiment with launch widths inside MLX, and letting it unlock kernels written for 32 lanes on a 64-lane device would turn an experiment knob into a correctness switch. The function returns 0 when ROCm is unavailable or either HIP call fails. It clears only the HIP error it raised itself: if an error was already pending on the thread (`hipPeekAtLastError`), it leaves it for its owner, so neither error is misreported.

`rocm_port_warp_size()` in `gpu_backend.cpp` caches that value once per process in a function-local static and, when it is not 32, prints one stderr line saying the fused kernels use their graph fallbacks (or that BitNet is refused), so an operator on CDNA sees why decode is slower rather than guessing.

### 3.3 Which tables are any-wave

`KernelPorts::rocm_any_wave_size` defaults to false, so a new port is held to wave32 until its author says otherwise. Five tables set it:

| Table | HIP source |
|---|---|
| `fused_rope_ports` | `FUSED_ROPE_APPEND_HIP_SOURCE` |
| `paged_merge_ports` | `PAGED_ATTENTION_MERGE_HIP_SOURCE` |
| `gumbel_ports` | `GUMBEL_MAX_SAMPLE_HIP_SOURCE` |
| `rejection_ports` | `REJECTION_SAMPLE_HIP_SOURCE` |
| `xielu_ports` | `XIELU_HIP_SOURCE` |

The issue listed the first four. Reading the current code found `XIELU_HIP_SOURCE` purely elementwise (one thread per element, no cross-lane step), so it joins them, and its `#error` guard is dropped for the same reason the sampler and RoPE bodies never had one: on a compiler that did define the macro, the guard would fail the hipRTC compile of a correct kernel on wave64. The other ten ROCm tables stay wave32-only: bitlinear, ssm, mamba1_scan, moe_gateup, moe_down, moe_fc1_relu2, add3_layer_norm, fused_norm, paged_attention and paged_v2_partial. Their guards stay (harmless, and live on a compiler that defines the macro), with comments rewritten to name `port_for` as the protection.

### 3.4 BitNet refused at load

Before this PR, `bitlinear_kernel_available` returned "any GPU backend", independent of the port table. On wave64 that would have said yes, BitNet would have loaded, and the first BitLinear matmul would have hit the refusal in `select_kernel_port` mid-request, with no graph fallback to take. The predicate now returns `has_kernel_port(bitlinear_ports())`, so it reads the same table the launcher dispatches through and answers false on wave64, and `reject_without_bitlinear_kernel` refuses the checkpoint at load with a message naming the wavefront. The launcher's refusal text no longer promises a "graph fallback"; it names the BitNet load-time refusal.

### 3.5 `mamba1_selective_scan` returns `Result`

`select_kernel_port` throws when it refuses. A throw through a cxx extern not declared `-> Result<...>` crosses a `noexcept` boundary and ends the process. The Mamba1 scan was one such extern, and on wave64 it can now refuse, so the declaration became `-> Result<()>`. Mamba and Jamba gate the launch on `mamba1_scan_kernel_accepts()`, which reads the same tables, so they `expect` the result with a message saying the gate already checked the port; a failure there would mean the gate and the launcher disagree, which is a bug to surface, not a condition to handle.

### 3.6 Checker rules 5 and 6

Marking a table any-wave lets it run on hardware no one has tested it on, on the strength of a claim about its source. `verify-kernel-port-dispatch` now checks that claim:

- **Rule 5.** Marked tables are pinned in `EXPECTED_ANY_WAVE` with the HIP source each compiles. For each marked table the checker follows the `.rocm` lambda to its `get_*()` accessor, the `static` holder it returns, the holder struct and one level of helpers, collects every `fast::hip_kernel` call, and requires the compiled sources to be exactly the pinned one, passed with no further argument (an extra argument could be a header the scan would not see) and defined as a single raw literal (adjacent literals would be concatenated and only the first scanned). The body, with comments stripped, must match nothing in `LANE_OPS`: shuffle, ballot, vote and mask intrinsics, `warpSize`, the AMDGCN lane builtins (`ds_swizzle`, DPP, `readlane`, `permlane`, `ballot` and others), the wavefront macros, and 32-lane arithmetic on `threadIdx.x`. A marked table missing from the pin fails, and so does a pinned table that lost its mark, so the set can move in neither direction without review.
- **Rule 6.** `set_rocm_port_warp_size_for_tests` may appear only in Rust under `tests/`, `*_tests.rs` or `test_support/`, plus its definitions, each held to an exact use count per file (`SEAM_DEFINITIONS`) so a production call added inside the large bridge files still fails.

`scripts/ci/check_kernel_port_dispatch_test.sh` copies the launcher sources into a scratch tree, mutates the copy, and runs the checker with `--root`. It has 13 cases: 10 mutations that must fail (marking a shuffle-based table, a shuffle, a ballot or an AMDGCN ballot builtin added to a marked body, a marked entry pointed at another holder, a marked holder passing a header, an unmarked pinned table, and the seam called from production Rust, from C++, and a second time inside the defining bridge file) and 3 controls that must pass (the untouched tree, a lane intrinsic named only in a comment, and the seam called from a test module). Each mutation fails loudly if its replacement string stops matching, so a case cannot pass because its mutation silently stopped applying. The make target and a CI step both run it.

### 3.7 The `build.rs` gap

`mlxcel-core/build.rs` did not list `kernel_port.*` or `gpu_backend.*` as `rerun-if-changed`, so editing them left the bridge library stale until some other tracked file changed. This surfaced during development when a negative check (disabling the hold in `port_for` and expecting the wave64 test to fail) passed: the test was running against the previous build. The four files are now tracked. The gap predates this PR (the helpers date from #1803) and could have hidden any earlier edit to port selection in a warm tree.

## 4. Verification

### 4.1 Tests

- **Truth table** (`kernel_port_tests.rs`, every backend): `rocm_port_allowed` for `(false, 32) = true`, `(false, 64) = false`, `(true, 64) = true`, `(false, 0) = false`, through the cxx bridge. 2 passed.
- **Hardware width** (`tests/rocm_wave_size.rs`): `device_warp_size() == 32` on gfx1151; a child process with `MLX_ROCM_FORCE_WARP_SIZE=64` (whose device log confirms the override applied to MLX) still reads 32; all 16 port predicates are true.
- **Wave64 refusal through the seam** (`tests/rocm_wave64_port_refusal.rs`): with the reported width set to 64 before any predicate is asked, all 11 wave32-only predicates (covering the ten tables) are false and the five any-wave ones are true; `bitlinear_matmul`, `fused_add_rms_norm` and `fused_moe_expert_kernel` (moe_gateup) refuse with a message naming the width; `layers::fused_add_rms_norm` with fusion on and `residual_add3_layer_norm` return the same bytes as the graph path. With the hold removed from `port_for`, this test fails and lists all 11 entries, so it detects the regression it guards against.

The seam has to be set before the first predicate call because the Rust gates cache a `true` answer, so the refusal test is its own binary with one test that sets the seam first and runs every check in order.

### 4.2 gfx1151 unchanged

`Meta-Llama-3.1-8B-Instruct-4bit`, release binary under `scripts/rocm_gpu_guard.sh`: greedy 64 tokens with `MLXCEL_FUSED_ADD_RMSNORM=0` and `=1` give identical text (33.88 and 33.90 tok/s, one run each, not a performance claim). `rocprofv3 --kernel-trace` of a sampled run shows `custom_kernel_mlxcel_fused_add_rms_norm` (wave32-only, 544 launches) and `custom_kernel_mlxcel_gumbel_max_sample` (any-wave) still selected, and stderr carries no wavefront notice. With all 16 predicates true and the same kernel names in the profile, the port choice on RDNA is the same as before.

### 4.3 Gates

From the PR: targeted port tests in the debug profile (fused norm, RoPE, MoE, relu2, SSM, Mamba1, Gumbel, rejection, fixed-key, paged v2, MLA, residual add3) 357 passed, with 2 `cache::paged_detach` tests failing on a `debug_assert` that the gate's `test-fast` profile does not compile and that is unrelated to ports; Apertus, BitNet and Cohere2 tests 26 passed; `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt verify-rocm-overlay` pass. After the review fixes the targeted tests were re-run (29 mlxcel-core, 30 Mamba/Jamba, the three integration files) and pass.

From the orchestrator, on head `1dc419c6`: `make verify-rocm` passed every step, with 11,969 tests passed, 0 failed, 381 ignored, and the smoke step OK.

Not verified: no wave64 device was available, so the refusal on real CDNA hardware is untested. Metal and CUDA were not available on the host; their `port_for` cases are unchanged, and `bitlinear_kernel_available` answers the same there because both tables have ports.

## 5. Technical Decisions

- **Host-side hold rather than a compile-time check.** `static_assert(warpSize == 32)` does not compile in HIP, and the macro is not defined, so no compile-time handle exists on the current toolchain. The hardware attribute is reliable and cheap to read once.
- **At port selection rather than per launcher.** A per-launcher check would duplicate the decision in every launcher and let a predicate and its launcher disagree, which `select_kernel_port` exists to prevent.
- **Hardware width, not `Device::warp_size()`.** The override is a launch-width experiment for MLX's own kernels; it must not make mlxcel run wave32 kernels on wave64.
- **Default false.** A port written without thinking about wave64 is held back by default. The cost is that a correct lane-free port is also held until someone marks it, which the checker then verifies.
- **Verify the mark mechanically.** The any-wave claim is enforced by a source scan with a pinned list and negative tests, instead of a comment that could drift as the inert `#error` guard did.
- **Refuse BitNet at load.** BitLinear has no graph fallback, so the only choices on wave64 are an unvalidated kernel, a mid-request failure, or a load-time refusal; the last is the one an operator can act on.
- **Exact use counts for the seam.** File-level exemptions for the bridge files would let a production call slip into files that are thousands of lines long.

## 6. Residual Risks and Follow-ups

- **The seam is public.** `set_rocm_port_warp_size_for_tests` is an ordinary cxx bridge function, so any crate linking `mlxcel-core` can call it. Rule 6 scans this repository only. Gating it behind a feature or `#[cfg(test)]`-only build would close the gap.
- **Per-process width cache.** The width is read once for the device current at the first call. A host mixing RDNA and CDNA cards keeps the first answer, so a process that moves to the other kind of GPU applies the wrong hold, which on a CDNA device could let wave32-only kernels run. The header comment states that this case is not handled.
- **A failed query is cached as 0.** If `device_warp_size()` fails once (for example a transient HIP error at first use), the process holds back every wave32-only port for its lifetime, including on a wave32 device. That fails safe but slow, and the stderr notice says "query failed". Retrying a 0 result, or not caching it, would avoid a sticky slowdown.
- **No wave64 run.** The ten wave32-only ports are refused, not validated. Validating a port on CDNA and marking it any-wave is per kernel, needs a CDNA host, and was out of scope. The five any-wave ports have also not run on wave64; their marking rests on the source scan.
- **The lane-op scan is lexical.** `LANE_OPS` covers the intrinsic families, AMDGCN builtins and common `threadIdx.x` index forms in use today. A body that derives a lane index another way (for example through a helper variable before `% 32`) would not be flagged.

## 7. Learning Points

- **A guard that never fires looks the same as a guard that holds.** `#if defined(MACRO) && MACRO != 32` is silent when the macro is absent. When a check depends on a toolchain-provided symbol, verify the symbol exists on each supported toolchain, or put the check where the value is known to exist (here, the device attribute at runtime).
- **Put a policy in the function every path already shares.** Because predicates and launchers both read `port_for`, one condition there changed every caller consistently, and the existing graph fallbacks did the rest.
- **Machine-check claims that let code run untested.** The any-wave mark is a claim about source text; a pinned list, a getter-following scan and negative mutations keep it from drifting.
- **A negative check that passes needs explaining.** Removing the hold should have broken the refusal test; when it did not, the cause was a stale build caused by missing `rerun-if-changed` lines, not a weak test. Build-dependency gaps can make any verification run against old code.
- **Prove a refusal on hardware you do not have through a seam, and prove the seam's test can fail.** The test-only width override exercises the production path below the width read, and disabling the hold makes the test fail, which shows the test measures the hold and not just the seam.
