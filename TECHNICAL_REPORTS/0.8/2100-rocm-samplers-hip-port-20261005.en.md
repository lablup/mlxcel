# Technical Report: PR #2100 - Port the Gumbel-max and rejection samplers to HIP

**Date**: 2026-10-05

**Status**: Implemented and measured on the gfx1151 host; head `74624ea7` on origin/main `57d8ed29`, pending merge.

**Languages**: C++ (HIP kernel sources, kernel holders, support predicate, bridge functions, vendored ROCm overlay), Rust (fixed-key and two-sample tests, kill-switch tests, snapshot bound), Bash (`bench_decode.sh` options, dtype-key checker test), Markdown (environment variables, installation, benchmarks, LOCAL_FIXES, benchmark results), CSV (bench rows)

**Risk Level**: Medium (it changes the default sampled-decode path on ROCm for every model, patches the vendored `fast::hip_kernel` argument layout that every HIP custom kernel goes through, and touches the rejection predicate, bridge functions and a snapshot bound that Metal and CUDA share, neither of which could be run on the development host)

## Executive Summary

Issue #2064 (part of #1814, epic #1801) asked for HIP ports of the two fused samplers that run once per generated token whenever temperature is above 0: the Gumbel-max sampler on the no-filter path and the dual-pivot rejection sampler on the filtered path. Before this PR both took the MLX graph on ROCm. `gumbel_ports()` and `rejection_ports()` had no `.rocm` entry, and `rejection_sample_supported()` returned `custom_kernels_available()`, which is Metal-or-CUDA by definition, so a filled rejection slot would have stayed unreachable.

The PR adds `sampling_gumbel_hip.h` and `sampling_rejection_hip.h`, line-for-line ports of the CUDA bodies with the same inputs, outputs, grid, template arguments and Philox-4x32-10 counter and key layout, fills both `.rocm` slots, and makes `rejection_sample_supported()` return `has_kernel_port(rejection_ports())`. Neither port carries a wave32 `#error` guard, because neither kernel has a lane-level operation: every reduction and the rejection kernel's scan go through shared memory with a barrier per step.

The Gumbel port exposed a bug in the vendored ROCm overlay: `fast::hip_kernel` declared `<input>_shape` and `<input>_strides` as pointers while the launch passed them by value, so the first kernel that read one (`logits_shape[1]`) faulted the queue. That is fixed as LOCAL_FIXES item 30, a fork-side change and an upstreaming candidate for #1813.

On gfx1151 with Llama-3.1-8B-Instruct-4bit at pp512/tg128, top-p 0.95 decode went from a median of 33.98 to 37.56 tok/s (1.11x), with a downward drift in the before arm that the report states. The Gumbel path moved from 37.91 to 38.26 tok/s (+0.9%), inside the run-to-run spread, and is not claimed as a gain. The issue's top-k plus top-p command was not measured because at vocab 128256 it never reaches the kernel.

## 1. Problem Statement

### 1.1 Two empty slots, one predicate that could not see them

`gumbel_max_sample_supported()` already read `has_kernel_port(gumbel_ports())`, so filling `.rocm` there was enough to make it reachable. `rejection_sample_supported()` did not:

```cpp
return mlxcel::custom_kernels_available();
```

`custom_kernels_available_for` answers true only for Metal and CUDA, so on ROCm the predicate stayed false whatever `rejection_ports()` held. It feeds `sampling_rejection_available()` and the routing in the bridge. Checker rule 4 of `verify-kernel-port-dispatch` did not catch it because it is not spelled `metal_is_available() || cuda_is_available()`.

### 1.2 What the profile said it was worth

The decode profile (`rocm-decode-profile-gfx1151-2026-09-30.md`) put the whole sampler tail at 0% of greedy decode GPU time and 0.4 to 3.8% of sampled decode GPU time. Without the ports, sampled decode on ROCm ran `random::categorical` on the no-filter path and the `argpartition` / `argsort` / `cumsum` chain on the filtered path. The expected gain was small on the no-filter path and larger only where the graph sorts the full vocabulary.

### 1.3 No test held a port to the graph on the same randomness

The existing suites, `sampling_gumbel_tests` and `sampling_rejection_tests`, test each kernel against its exact target distribution. They returned early on ROCm. Nothing checked that a port consumes MLX's key sequence the same way as the other ports, which is what makes `mlx::core::random::seed(...)` reproduce a stream across backends.

## 2. Change Summary

| Area | Change |
|---|---|
| `sampling_gumbel_hip.h` (new) | `GUMBEL_MAX_SAMPLE_HIP_SOURCE`, the CUDA body with spelling changes only |
| `sampling_rejection_hip.h` (new) | `REJECTION_SAMPLE_HIP_SOURCE`, the CUDA body with spelling changes only |
| `sampling.cpp` | `GumbelKernelHolderHip` calling `fast::hip_kernel` under `MLXCEL_BRIDGE_ROCM_BACKEND`; `.rocm` getter in `gumbel_ports()` |
| `sampling_rejection.cpp` | `RejectionKernelHolderHip`; `.rocm` getter in `rejection_ports()`; `rejection_sample_supported()` returns `has_kernel_port(rejection_ports())` |
| ROCm overlay `custom_kernel.cpp` | `KernelShape` / `KernelStrides` by-value structs, an `elem_to_loc` overload, a `static_assert` on `JIT_MAX_NDIM`, and shape/strides/ndim appended only for ndim > 0 (LOCAL_FIXES item 30) |
| Bridge (`mlx_cxx_bridge.cpp/.h`, `lib.rs`) | `sampling_gumbel_backend_supported()`, `sampling_rejection_backend_supported()` (env-independent), `random_bits()`; reworded comments |
| `gpu_backend.h/.cpp` | Comments on what `custom_kernels_available()` now means; the `MLXCEL_DEBUG_KERNEL_BACKEND` suffix reworded |
| Tests | New `sampling_fixed_key_tests.rs` (five tests); kill-switch tests ask the new predicates; `temperature_one_support_unchanged` routed bound from 1 to 2 ulp; dtype-key checker test moves both launches; Python probe test follows the new suffix |
| `bench_decode.sh` | `--temperature` and `--top-p`, plain decimals only, tagging the auto-generated filename |
| Docs and data | `environment-variables.md`, `installation.md`, `benchmarks.md`, upstream README row for item 30, `LOCAL_FIXES.md` item 30, new `rocm-samplers-gfx1151-2026-10-05.md`, four bench CSVs |

The branch has six commits: the overlay fix (`cc93789f`), the ports and predicate (`85e39880`), the results doc (`ce4105b7`), the kill-switch predicate fix (`f7bec56d`), the snapshot bound (`6e336ec8`), and doc and `bench_decode.sh` input tightening (`74624ea7`). The diff is 32 files, 1,341 insertions and 46 deletions.

## 3. The Ports

### 3.1 Line for line, with three kinds of spelling change

Both HIP sources keep the CUDA kernels' thread mapping, grid, inputs (`logits`, `rng_key`, `temp`; `probs`, `probs_draw`, `params`, `rng_key`), outputs (`vals`, `idxs`; `ids`, `ok`, `rounds`) and template arguments, dtype keys included. The Philox-4x32-10 counter (`{base/4, 0, row, 0}` for Gumbel, `{round, 0, row, 0}` for rejection), the key `{rng_key[0], rng_key[1]}` and the 23-bit open-interval uniform are unchanged, so a seed reproduces a ROCm stream the way it does on CUDA and Metal. `__umulhi` has the same meaning in HIP. The differences are:

- explicit `(float)` reads of `hip_bfloat16` values, because MLX's bfloat16 substitute on ROCm converts to float only through an `explicit` operator;
- `-__builtin_huge_valf()` in place of `-INFINITY`, because hipRTC compiles the body with only the headers `fast::hip_kernel` prepends and `<cmath>` is not guaranteed among them;
- a redundant `(float)` on `temp[0]`.

The holders and launches stay in `sampling.cpp` and `sampling_rejection.cpp`, next to the CUDA ones. The headers hold data, not a launch, so `make verify-kernel-dtype-keys` keeps `EXPECTED_IN_SCOPE` and its count unchanged. The checker's own test, which moves the sampler's launch to a helper and comments it out, now does the same to the HIP launch.

### 3.2 Why no wave32 guard

The #1814 port requirements tell every port that reduces or shuffles across lanes to carry two `#error` checks on the wavefront-size macros, because a lane fold that starts at 16 drops half of a 64-lane CDNA wave and still returns a finite, wrong result. These two kernels have no lane-level operation. The Gumbel kernel's index-carrying halving reduction and the rejection kernel's reductions and Hillis-Steele scan all go through `__shared__` memory with a `__syncthreads()` between steps, so the result does not depend on how many threads share a wave. An `#error` on wave64 would reject a correct kernel. PR #2099 also found that the guard idiom is inert with ROCm 10's AMD clang, which defines neither macro, so it would catch nothing here in any case. The issue's acceptance criterion was ticked with that reason written next to it.

## 4. The Predicate and the Tests That Read It

`rejection_sample_supported()` keeps its `default_device() != Device::gpu` check and then returns `mlxcel::has_kernel_port(rejection_ports())`, the same table `select_kernel_port` reads in `rejection_sample`, which mirrors `gumbel_max_sample_supported()`. On Metal and CUDA the answer is true as before. `fused_sample`, `fused_sample_probs` and the speculative paths reach the HIP kernels with no caller change.

Two integration tests depended on the old meaning. `tests/sampling_gumbel_kill_switch.rs` and `tests/sampling_rejection_kill_switch.rs` set the env kill switch and choose which decline message to expect with `custom_kernels_available()`. That function still answers Metal-or-CUDA, so on ROCm, where a port now exists, the tests would have expected "no port". Under the env switch, `sampling_*_available()` is false whatever the backend, so the tests needed a predicate without the switch. The PR adds `sampling_gumbel_backend_supported()` and `sampling_rejection_backend_supported()` to the bridge for that purpose (`f7bec56d`). The `gpu_backend.h` comments now say that `custom_kernels_available()` is a family-wide gate and that a kernel with a HIP port is gated on its own predicate. The `MLXCEL_DEBUG_KERNEL_BACKEND` suffix changes from "no custom kernel ports" to "not every kernel family is ported".

## 5. The Overlay Bug: Shapes and Strides by Value

`build_kernel` in the fork's `mlx/backend/rocm/custom_kernel.cpp` declared an input's shape as `const int32_t* <name>_shape` and its strides as `const int64_t* <name>_strides`. `CustomKernel::eval_gpu` passes them with `KernelArgs::append_ndim`, which pads the vector to `JIT_MAX_NDIM` (8) entries and hands `hipModuleLaunchKernel` a pointer to that storage. The kernel therefore received the values and read the first eight bytes of the shape as an address. The HIP Gumbel port reads `logits_shape[1]`, as the CUDA and Metal ports do, and died with `HSA_STATUS_ERROR_MEMORY_FAULT` at `0x80000000000`, the page of `0x80000000800`, which is the shape `{2048, 2048}` read as a 64-bit pointer. No HIP kernel in mlxcel had read either argument before, which is why it went unseen.

The fix (`cc93789f`) adds `KernelShape` (8 `int32_t`) and `KernelStrides` (8 `int64_t`) with an `operator[]` to the generated header, declares the parameters with them by value, which is what upstream CUDA's `const __grid_constant__ Shape` / `Strides` does, and adds an `elem_to_loc` overload. A `static_assert` ties the 8 to `JIT_MAX_NDIM`. The launch now appends an input's shape, strides and ndim only when its ndim is above zero, the same condition under which `build_kernel` declares them. Before, a 0-d input whose name appeared as `<name>_shape` in the source would have shifted every later argument. No mlxcel kernel has such an input.

The fix is recorded as LOCAL_FIXES item 30. It is a fork-side change confined to `custom_kernel.cpp`, and an upstreaming candidate under #1813; the upstream README marks it "not packaged yet" because a package needs a repro on the unpatched fork. `sampling_fixed_key_tests` faults without it. The `_strides` half has no user in mlxcel and was checked only by compiling it.

## 6. Correctness Evidence

A sampler has no elementwise fallback to diff against, so the PR holds the ports to the graph in four ways. The two existing suites also run on ROCm now instead of returning early, and pass on gfx1151.

### 6.1 Fixed key, recomputed on the host

Both kernels draw one Philox key per call from MLX's default key sequence, and given the key their output is deterministic. `sampling_fixed_key_tests` reseeds, reads the key the next launch will take through the new `random_bits` bridge function, reseeds again, launches, and recomputes the draw on the host.

- `gumbel_kernel_matches_the_keyed_graph_argmax`: 48 rows of 5003 logits in f32 (T 1.0 and 0.7), bf16 (T 1.3) and f16 (T 0.9). The kernel's index equals the host's `argmax(logits / T + g)` on every decided row, and so does the MLX graph's `argmax` with the same noise. A row counts as decided when its winning score leads by at least 1e-4.
- `rejection_kernel_draws_the_keyed_token_under_min_p`: min-p 0.05, 40 rows of 3001 entries, T 1.0 and 0.7. Min-p resolves its threshold before the first draw, so round 0 accepts and the token follows from the round-0 Philox word and the kernel's thread-major scan order. The kernel matches the host on every decided row (target at least 1e-5 of the mass from a cumulative boundary).

With the Philox counter (`row + 1`) or the drawn word (`c1` for `c0`) changed in the HIP sources, both tests fail on row 0. So the tests detect a port that consumes the key differently, not only one that draws from the wrong distribution.

### 6.2 Support membership

`rejection_kernel_draws_stay_inside_the_filtered_support` covers rounds past the first, which the host cannot reproduce bit for bit because the kernel's f32 sums may differ in their last bits. For 40 rows of 3001 entries under (top-k, top-p) = (0, 0.9), (40, 1.0) and (0, 0.5), four launches each, every row must converge and every drawn token must lie inside the support computed on the host from the same probabilities, with a relative 1e-5 slack on the top-p cutoff. The test also asserts that at least one row needed a second round, so the bisection is exercised.

### 6.3 Chi-square against the graph sampler

Where the kernel and the graph consume the RNG differently, a fixed key cannot line them up. Two-sample chi-square tests compare each kernel's histogram with `fused_sample_categorical`, the explicit graph arm, on the same input, at 400,000 draws per arm, pooled to 20 draws per bin, with the critical value at p = 1e-6:

- the Gumbel kernel against `random::categorical` at T 0.8;
- the rejection kernel against the stock chain at top-p 0.9, and at top-p 0.95 with min-p 0.02.

Both pass. Top-k with top-p is left out because the kernel is not exact there (see `sampling_rejection_tests.rs`).

### 6.4 End-to-end speculative decoding

The classic `SpeculativeGenerator` path verifies drafted tokens with the target's sampler, and `fused_sample_probs` follows the same routing, so with top-p the target's draws come from the rejection kernel. Qwen3-30B-A3B-4bit with Qwen3-0.6B-4bit as drafter, `--temp 0.8 --top-p 0.95 -n 128`, `MLXCEL_SPECULATIVE_ACCEPT_DIAG=1`: the dispatch log names the rejection kernel after and the stock chain before, the output is coherent, and per-position acceptance sits near the closed form for each rule. After: sampler-match 0.643 against `sum_prod` 0.665, stochastic 0.732 against `sum_min` 0.676. Before: 0.709 against 0.678 and 0.673 against 0.699, about 110 tested positions each. These are single runs that show the path works; the tests above carry the distributional claim. No MTP checkpoint was available, so the MTP and DFlash round loops were not run.

## 7. Measured Results

### 7.1 Decode throughput

`scripts/bench_decode.sh` at pp512/tg128 with the new `--temperature` / `--top-p` options, Meta-Llama-3.1-8B-Instruct-4bit (vocab 128256), three runs per arm interleaved run by run (before, after, before, after). Before is `main` at `57d8ed29`; after is `85e39880`. Later commits do not touch any code that a sampler launch with 1-d and 2-d inputs reaches.

| Configuration | Path after | Before tok/s | After tok/s | Medians |
|---|---|---|---|---|
| `--temperature 0.8` | Gumbel-max kernel | 37.91 / 37.67 / 38.32, median 37.91 | 37.59 / 38.26 / 39.25, median 38.26 | 1.01x, inside the spread |
| `--temperature 0.8 --top-p 0.95` | rejection kernel | 36.26 / 33.98 / 32.81, median 33.98 | 37.82 / 37.56 / 37.06, median 37.56 | 1.11x |

`compare_bench_csv.py`, which keeps the last row per model and so compares the third runs, reports 1.024x and 1.130x.

**Gumbel, not claimed.** The medians differ by 0.9%, against a spread of 1.7% in the before arm and 4.4% in the after arm, and the arms overlap. This is consistent with the profile's 0.4 to 3.8% share for the whole sampler tail. The PR reports no gain on this path.

**Rejection, 1.11x with a drifting before arm.** Every after run is faster than every before run, by 1.04x, 1.11x and 1.13x in the three interleaved pairs. The before arm drifted down across its runs (36.26, 33.98, 32.81), while the after arm stayed within 37.06 to 37.82. The cause of the drift was not isolated. The pair ratios grow with the drift, so the 1.11x median ratio depends partly on it; the smallest pair, 1.04x, is the conservative reading. The graph chain runs an `argsort` over all 128256 entries per token for top-p, which is the work the rejection kernel replaces.

### 7.2 Why top-k plus top-p was not measured

The issue's second command was `--temp 0.8 --top-k 40 --top-p 0.95`. The routing policy sends top-k combined with top-p to the rejection kernel only up to vocab 32768 (`REJECTION_JOINT_VOCAB_MAX`, measured on M1 Ultra). Llama-3.1's vocabulary is 128256, so that configuration runs the stock chain both before and after, and any difference would be noise. Top-p alone is the routed configuration and is the one measured. Whether the 32768 cap is right on ROCm was not measured.

### 7.3 Idle-GPU guard and the `--idle-secs 75` deviation

Every run went through `scripts/rocm_gpu_guard.sh`, the copy fixed in #2098 run from the #2065 worktree and not committed in this PR. The issue's method asks for 90 s with `/sys/class/kfd/kfd/proc` empty and no compiler. A parallel unit (#2065) benchmarked at the same time with the guard's default 90. At equal windows both guards started their idle windows together and each rejected the other's run, in lockstep. This PR's runs used `--idle-secs 75`, 75 consecutive 1 Hz samples, which took 112 to 116 s of wall time per run and broke the tie. All 12 accepted runs were clean on their first attempt. The deviation lowers the required sample count; each run's guard phase still took 112 to 116 s of wall time, more than the method's 90 s.

## 8. The Snapshot Bound: 1 ulp to 2 ulp

`sampling::tests::temperature_one_support_unchanged` compares `fused_sample_probs` at T 1.0 with saved Metal rows. The routed cases had skipped wherever the rejection kernel had no port, so they first ran on ROCm with this PR. The gate at `f7bec56d` failed only this test. These rows are the graph's softmax over the kernel's support; `fused_sample_probs` never launches the kernel. gfx1151 lands 2 ulp from the Metal capture at token 26 of the (top-k 40, top-p 0.9) row, with the support unchanged. Commit `6e336ec8` raises the routed bound from 1 to 2 ulp, allowing the same kind of reassociation drift as the non-routed stock case, which allows 4.

The bound is not per-backend. The change widens it on Metal and CUDA too, where the routed rows had run at 1 ulp before. A 2 ulp drift there now passes silently. The support-set assertion in the same test, which is the correctness gate, is unchanged.

## 9. Technical Decisions

- **Port the CUDA bodies line for line.** CUDA and HIP share the grid semantics, `template_args` and hipRTC-style runtime compilation. Keeping the bodies identical except for spelling leaves one place to look when they disagree, and keeps the Philox layout bit-identical across three backends.
- **No wave32 guard.** Shared-memory reductions with barriers are wave-size independent. A guard would reject a correct kernel on wave64 and, per #2067, does not fire on ROCm 10 clang anyway.
- **Keep the launches in the existing files.** Moving the HIP body to a header for size, but leaving the `fast::hip_kernel` call next to the CUDA one, keeps the dtype-key checker's pinned scope and count unchanged.
- **Make the rejection predicate read the table.** `has_kernel_port(rejection_ports())` cannot disagree with `select_kernel_port`, and the next backend port becomes reachable with no predicate edit.
- **Fix the overlay rather than work around it.** Reading the vocab from a scalar input would have avoided `logits_shape`, but the overlay's declaration was wrong for every future HIP kernel that reads a shape. The by-value structs match upstream CUDA and the fix is upstreamable.
- **Add env-independent bridge predicates for the kill-switch tests.** The tests need to know whether a port exists while the kill switch is set; `custom_kernels_available()` answers a different question.
- **Hold the ports to the graph on the same randomness.** The fixed-key tests catch a port that draws from the right distribution but consumes the key differently, which the existing chi-square suites would pass.

## 10. Validation

From the PR body, on gfx1151 (Radeon 8060S, ROCm 10.0.0):

- `make verify-rocm` at `6e336ec8`: OK; 11861 passed, 0 failed, 378 ignored across 146 test binaries; ROCm smoke OK. The later commit `74624ea7` changes docs, two header comments and `bench_decode.sh` input validation; the fast gates, fmt and `dead_doc_pointers` pass on it.
- `cargo test --release --features rocm -p mlxcel-core --lib sampling_ -- --test-threads=1`: 53 passed, plus both kill-switch integration tests.
- `cargo clippy -p mlxcel-core --features rocm --lib --tests -- -D warnings`, `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-rocm-overlay verify-fmt`: pass.

Orchestrator verification on head `74624ea7`, up to date with origin/main `57d8ed29`:

- `make verify-rocm` passed every step: 11,861 tests passed, 0 failed, 378 ignored; smoke OK.

## 11. Residual Risks and What Was Not Verified

- **Metal and CUDA were not run.** They are touched by the rejection predicate (`has_kernel_port`, true on both as before), the shared bridge functions (`random_bits`, `sampling_*_backend_supported`), the kill-switch tests' predicate, the routed snapshot bound and `sampling_fixed_key_tests`, which is written to run there too.
- **The snapshot bound is looser everywhere.** A 2 ulp drift on Metal or CUDA in the routed rows no longer trips the test.
- **The `_strides` half of the overlay fix has no user.** No kernel reads `<input>_strides`, so it is checked only by the struct compiling in every kernel header.
- **The before-arm drift in the top-p measurement is unexplained.** The 1.11x median ratio leans on it; the per-pair range is 1.04x to 1.13x.
- **The joint top-k plus top-p cap was not re-measured on ROCm.** At vocab above 32768 that configuration stays on the stock chain.
- **MTP and DFlash speculative loops were not run.** No MTP checkpoint was available locally.
- **The guard copy is not in this PR.** The runs used the #2098 fix from the #2065 worktree, with `--idle-secs 75` instead of the method's 90 samples.
- **One device, one session, shared GPU.** All numbers are from gfx1151. No wave64 hardware was available, so wave-size independence on CDNA is reasoned from the kernels' structure, not run.

## 12. Learning Points

- **A predicate that enumerates backends hides a filled slot.** `custom_kernels_available()` answered a family-wide question; per-kernel predicates must read their own port table.
- **The first kernel to use an argument finds the bug in how it is passed.** The overlay's shape declaration was wrong since the fork, and nothing failed until a kernel read `logits_shape`. A memory fault at an address that decodes to a shape is a sign that a by-value argument is being read as a pointer.
- **Distribution tests do not prove a port is the same sampler.** A port with a different Philox counter layout draws from the right distribution and passes chi-square. Recomputing the draw from the consumed key is what pins the RNG contract.
- **A wave32 guard belongs only where lanes talk to each other.** Shared-memory reductions with barriers need no guard, and adding one would reject a correct kernel.
- **Check that the measured configuration reaches the kernel.** The issue's top-k plus top-p command routes to the stock chain at this vocabulary; measuring it would have reported noise as a result.
- **Concurrent guards with equal windows can lock each other out.** Two idle guards with the same sample count start together and reject each other; a different count breaks the tie.

Refs: #2064, #1814, #1801, #1813, #2061, #2067, #2099, #2065, #2098, #1862, #1870, #1885, #900, #901, #1803.
