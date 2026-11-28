# Technical Report: PR #2181 - Key ROCm Custom-Kernel JIT Modules by Their Generated Source

**Date**: 2026-10-07

**Status**: Implemented and verified on the gfx1151 host; head `b11b4cc5` (up to date with origin/main `b7116d1d` at verification), PR open, pending merge. Closes #2149 (part of #1801).

**Languages**: C++ (the ROCm overlay's `custom_kernel.cpp`, the cxx bridge, comments in `turbo/paged_attention*.cpp`), Rust (mlxcel-core FFI wrapper, `tests/rocm_custom_kernel_jit_key.rs`), Python (`scripts/ci/check_kernel_dtype_keys.py` docstring and comments), Markdown (`LOCAL_FIXES.md`, `docs/code-guidelines.md`)

**Risk Level**: Low. The production change is one module-name expression in the ROCm overlay, copied from upstream CUDA. A call whose source never varied gets the same module as before under a longer key; a call whose source varied now compiles its own module instead of reusing the wrong one.

## Executive Summary

The ROCm overlay's `fast::hip_kernel` cached each compiled module under the device index plus the kernel name, and that name covers only `template_args`. The generated source also depends on each input's and output's dtype and on whether each input is 0-d. A later call that differed only in those properties reused the first call's module: it read an f16 input through `const float*`, wrote an f16 output through `float*`, or launched a 0-d input with an argument list the module does not declare. Nothing threw.

PR #2181 keys the module by `name_ + "_" + hex(std::hash<std::string>{}(source_))`, the same key upstream CUDA has used since 9f5f7931 (ml-explore/mlx#4273), which is present at the MLX pin `81ba1c6a`. The kernel symbol inside the module and the device-index prefix are unchanged, so profiler traces keep their `custom_kernel_*` names. The fix is recorded as LOCAL_FIXES item 33 and kept in mlxcelverse under the 2026-10-06 fork policy.

A test-only bridge probe launches one kernel name with no template args, so only the generated source can separate its calls. Three tests cover input dtype, output dtype (added in review) and a 0-d input after a 1-d one (in a child process). With the name-only key restored, all three fail: wrong f16-input values, wrong f16-output values, and a SIGSEGV in the 0-d child. The explicit dtype keys in mlxcel's own launches and the CI checker that enforces them stay, as defense in depth. Docs and comments that still described CUDA as keyed by name only were corrected. Review also found that the ROCm JIT module cache has no lock; that is now issue #2183.

## 1. Problem Statement

### 1.1 What the key missed

In `src/lib/mlx-cpp/patches-rocm/mlx/backend/rocm/custom_kernel.cpp`, `hip_kernel` builds `name_` as `"custom_kernel_" + name + template_arguments_hash(template_args)`. `build_kernel` then generates source that also encodes:

- each input's element type (`dtype_to_hip_type(arr.dtype())`) in its buffer parameter;
- each output's element type in its buffer parameter;
- whether each input has `ndim() > 0`, which decides whether `<input>_shape`, `<input>_strides` and `<input>_ndim` are declared at all (LOCAL_FIXES item 30 made the shape a fixed-width `KernelShape` passed by value, so the ndim value itself no longer matters, only its being zero or not).

`CustomKernel::eval_gpu` passed `name_` to `rocm::get_jit_module`, which memoises by `device.index + ":" + name` in a process-global map and calls the source builder only on a miss. The first source compiled under a name won for the life of the process.

### 1.2 What a mismatch did

- **Input dtype.** An f16 or bf16 input reused an f32 module and was read as `const float*`: each 4-byte read combined two 2-byte elements, and the kernel read twice as many bytes as the buffer held. The values were unrelated to the input.
- **Output dtype.** An f16 output reused an f32 module and was written through `float*`, with the same doubling of the bytes written.
- **0-d input.** Since item 30, `eval_gpu` appends an input's shape, strides and ndim only when its ndim is above zero, matching `build_kernel`. A 0-d input after a 1-d one under the same name launched with fewer arguments than the cached module declares, so every later parameter, including the output pointer, was read from the wrong slot.

### 1.3 Why mlxcel had not seen it

mlxcel's own launchers name their input dtypes in `template_args`, enforced by `scripts/ci/check_kernel_dtype_keys.py` (`make verify-kernel-dtype-keys`) since #1053 and #1054. That made `name_` differ per input dtype, which hid the defect for the common case. The check cannot cover an input whose dtype varies independently of the keyed one, an output dtype that varies at one input dtype, or a 0-d versus non-0-d input. The issue was found in #2100's security review, not from a wrong model output.

## 2. Change Summary

| Area | Change |
|---|---|
| ROCm overlay `custom_kernel.cpp` | `eval_gpu` passes `name_ + "_" + hex(std::hash<std::string>{}(source_))` as the module name; `kernel_name` (the symbol looked up) is unchanged; adds `<functional>` and `<string>` |
| `LOCAL_FIXES.md` | Item 33 (#2179 took 32), citing the upstream CUDA line it copies and the fork policy it is kept under |
| cxx bridge (`mlx_cxx_bridge.cpp/.h`, `lib.rs`) | Test-only `rocm_jit_key_probe(input, f16_output)`, a throwing stub off ROCm |
| `rocm_faults.rs` | Wrapper `jit_key_probe_array(input, f16_output) -> Result<UniquePtr<MlxArray>, cxx::Exception>` |
| `tests/rocm_custom_kernel_jit_key.rs` | Three tests (input dtype, output dtype, 0-d after 1-d in a child process), `#![cfg(feature = "rocm")]` |
| `check_kernel_dtype_keys.py` | Docstring describes CUDA and ROCm as source-keyed and the check as defense in depth; the in-scope comment names the new probe; rule, scope and pinned count (9) unchanged |
| `docs/code-guidelines.md` | Same correction to the "Why this matters" and "Enforcement" paragraphs |
| `turbo/paged_attention.cpp`, `paged_attention_v2.cpp` | Keying comments no longer describe CUDA as name-keyed in the present tense (comment-only) |

Two commits: the fix with the input-dtype and 0-d tests (`4d3e0629`), and the review follow-up that adds the output-dtype case and corrects the remaining stale comments (`b11b4cc5`). 11 files, 462 insertions and 48 deletions.

## 3. Design

### 3.1 Key on the source, as upstream CUDA does

The generated source is the complete input to the hiprtc compile, so hashing it covers every property that changes the module: the three in 1.1 and any future source-affecting input, without listing them. This is the expression upstream CUDA uses in `mlx/backend/cuda/custom_kernel.cpp` (around line 313 at the pin), added by 9f5f7931 (ml-explore/mlx#4273). Copying it keeps the ROCm fork aligned with the backend it tracks, so a later re-vendor that brings in upstream's ROCm keying should land on the same behavior.

The issue rejected Metal's approach, which appends every input dtype to the name. It fixes input dtypes but not output dtypes or the 0-d case, and it diverges from CUDA.

The change is deliberately narrow:

- **Symbol name unchanged.** `kernel_name` is still `"mlx::core::rocm::" + name_` (or `name_` for precompiled kernels). Only the cache key gains the hash, so `rocprofv3` traces still show `custom_kernel_*` and existing profile tooling keeps working.
- **Device-index prefix unchanged.** `get_jit_module` still prepends `device.index + ":"`; modules stay per device.
- **No disk-cache concern.** Custom kernels call `get_jit_module` with `use_disk_cache = false`, so the key never names a file. `std::hash` values are not stable across builds, which does not matter for a key that lives in one process.
- **Cost.** One string hash per `eval_gpu`, the same as CUDA. A launch whose source never varied compiles exactly as often as before. A launch whose source did vary now compiles one module per variant, which it needed for correct results.
- **Collisions.** Two different sources with the same `std::hash` under the same name would still share a module. CUDA accepts that, and so does this fix.

### 3.2 The probe

`rocm_jit_key_probe(input, f16_output)` launches `fast::hip_kernel("mlxcel_jit_key_probe", {"inp"}, {"out"}, ...)` with `template_args = {}` on every call. Because the name and template args never change, `name_` is identical across calls, which is the condition that exposes a name-only key. The kernel writes `out[1 + i] = (float)inp[i]` and, from thread 0, `out[0] = (float)inp_shape[0]`. The output is f32 or f16 of `n + 1` elements. The input must be f32, f16 or bf16, 0-d or 1-d, with 1 to 1024 elements; one block of `n` threads covers it.

The 0-d case needs one source text that compiles both with and without an `inp_shape` parameter. The probe's header declares, at namespace scope in `mlx::core::rocm`,

```cpp
__constant__ KernelShape inp_shape = {{-1, 0, 0, 0, 0, 0, 0, 0}};
```

For a 1-d input, `build_kernel` declares an `inp_shape` parameter, which shadows the sentinel, and `out[0]` is the length. For a 0-d input there is no such parameter, so unqualified lookup falls through to the sentinel and `out[0]` is `-1`. The generated sources differ (one has the parameter, one does not), so a correct key separates them, and `out[0]` tells the test which module actually ran. Because the probe passes `{}` for `template_args`, the CI checker treats it as in scope with nothing to check; the checker's limits comment names it so that is visibly deliberate.

### 3.3 The three tests

All use `VALUES = [0.5, -1.25, 3.0, 8.0, -0.75, 96.0, 0.0, 2.5]`, which f16 and bf16 represent exactly, so the expected output is bit-exact after conversion to f32. Each probe is evaluated on its own before the conversion, so the module lookup under test is not fused into a larger graph.

- **`input_dtype_selects_its_own_module`**: f32, then f16, then bf16 inputs with an f32 output. f32 runs first so that a name-keyed cache holds the `const float*` module when the narrower dtypes arrive.
- **`output_dtype_selects_its_own_module`**: one f32 input, an f32 output and then an f16 output. This case was added in review. The input-dtype test alone cannot catch a key that covers input dtypes but not output dtypes (Metal's approach, or the template-args check), and the issue's original acceptance criteria listed only inputs.
- **`scalar_input_after_vector_input_selects_its_own_module`**: a 4-element 1-d input, then a 0-d input. On a name-keyed build the second launch reads parameters from the wrong slots and can fault the queue, which on ROCm kills the device for the whole process. The test therefore re-invokes its own binary on an ignored child test (`child_scalar_after_vector`, gated by `MLXCEL_ROCM_JIT_KEY_CHILD`), drains stdout and stderr on separate threads so a HIP queue dump cannot block on a full pipe, enforces a 120-second budget, and parses one `JIT_KEY_PROBE <label> ok=...|err=...` line per probe. The child flushes stdout and leaves through `libc::_exit(0)` so HIP teardown cannot hang on a faulted queue. This follows `tests/rocm_gpu_faults.rs`. The parent asserts the child exited successfully, that the 1-d probe returned `[4, values...]`, and that the 0-d probe returned `[-1, value]`.

### 3.4 Keeping every explicit dtype key and the checker

With the source in the key, the dtype entries in mlxcel's `template_args` are no longer what makes ROCm or CUDA correct. The PR keeps all of them, and keeps `verify-kernel-dtype-keys` with its rule, scope and pinned in-scope count (9) unchanged, for three reasons:

- **They cost nothing.** The source already differs per dtype, so the source hash already splits those modules. The explicit keys add no compiles.
- **Some are used.** Several launches reference their dtype args as type aliases in the kernel body. Removing those would be a code change, not dead-code cleanup.
- **The rest are insurance.** A re-vendored ROCm fork, an MLX pin bump or a new backend can bring a name-only key back, and nothing else would notice until a model returned wrong numbers. The checker is the guard against that regression, so its docstring and `docs/code-guidelines.md` now call it defense in depth rather than the fix.

### 3.5 Correcting the stale CUDA description

The checker docstring, `docs/code-guidelines.md`, and the keying comments in `turbo/paged_attention.cpp` and `paged_attention_v2.cpp` said in the present tense that CUDA keys modules by name alone. That was true when #1053 was found but not at the current pin. They now say CUDA has keyed on the source hash since ml-explore/mlx#4273, that ROCm does since #2149, and that the name-only behavior is why the explicit keys exist. The `paged_attention` changes are comment-only.

## 4. Verification

### 4.1 Revert evidence

With the name-only key restored in the overlay, on gfx1151, all three tests fail:

- **f16 input**: for input `[0.5, -1.25, 3, 8, -0.75, ...]` the probe returned `[8, -0.0313, 131336, 3.5e13, 8, 0, ...]`. Element 0 is the correct length; the rest are pairs of f16 values read as one f32. The test stops at the f16 assertion, so the bf16 case was not observed separately under the revert.
- **f16 output**: the probe returned `[0, 2.5, 0, 1.75, ...]`, the f32 bit patterns written through `float*` and read back as f16 halves.
- **0-d after 1-d**: the child process died with SIGSEGV after its 1-d call.

With the fix, all three pass. The revert shows each test detects the defect it targets.

### 4.2 Gates

From the PR, on gfx1151:

- `cargo test --profile test-fast --features rocm --test rocm_custom_kernel_jit_key -- --test-threads=1`: 3 passed.
- `cargo clippy --features rocm --test rocm_custom_kernel_jit_key -- -D warnings` and `cargo clippy -p mlxcel-core --features rocm --lib -- -D warnings`: clean.
- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt verify-rocm-overlay` after rebasing onto `b7116d1d`: pass, including the checker's negative-case suite.

From the orchestrator, on head `b11b4cc5` (up to date with origin/main `b7116d1d`): `make verify-rocm` passed every step, with 11,972 tests passed, 0 failed, 382 ignored, and the smoke step OK. The new file contributes 3 passed and 1 ignored (the child test).

Not verified: Metal and CUDA were not available on the host. The change touches only the ROCm overlay, a bridge function compiled to a throwing stub off ROCm, a ROCm-only test file, and comments and docs, so no Metal or CUDA code path changes.

## 5. Technical Decisions

- **Source hash over a list of dtypes.** Hashing the source covers input dtypes, output dtypes, 0-d-ness and anything a later `build_kernel` change adds. A list would cover only what someone remembered to list, as the Metal approach shows by missing the 0-d case.
- **Copy upstream CUDA exactly.** The overlay tracks CUDA; the same expression keeps the two backends interchangeable for reasoning and keeps a future re-vendor simple.
- **Change the cache key, not the symbol.** Profiling and any tooling that matches `custom_kernel_*` names keep working.
- **Probe with no template args.** Using `{}` removes the shield that mlxcel's own launches have, so the test measures the backend key itself and would catch its regression even if every mlxcel launch kept its dtype keys.
- **Isolate the faulting case in a child process.** A queue fault on ROCm poisons the device for the process; running the 0-d case in-process would turn a regression into failures of unrelated tests or a hang.
- **Keep the dtype keys and the checker.** Zero cost, some are load-bearing in kernel bodies, and they guard against a regression this PR cannot prevent from outside the overlay.
- **Keep the fix in mlxcelverse.** Under the 2026-10-06 fork policy, the overlay item is maintained here rather than proposed to the ROCm fork.

## 6. Residual Risks and Follow-ups

- **Unlocked JIT module cache (#2183).** Review found that `get_jit_module` in `rocm/jit_module.cpp` does an unlocked `find` and `try_emplace` on a function-static `unordered_map`, and that `try_emplace` runs the hiprtc compile in place. Two threads launching JIT kernels at once (for example the chat scheduler and the embedding worker in one server) can both compile one key, or rehash the table under a concurrent `find`. Upstream CUDA and the CUDA overlay lock this with a `shared_mutex`. The race has not been observed. It predates this PR, is out of scope for keying, and is tracked as #2183. This PR does add keys over a process's lifetime (one per source variant), which slightly widens the window in which new inserts happen.
- **Hash collisions.** A `std::hash` collision between two sources under one name would reuse a module, as on CUDA. No mitigation was added.
- **The checker's limits remain.** `verify-kernel-dtype-keys` still cannot cover independent input dtypes, output dtypes or 0-d inputs, and it reads only named `template_args` initialisers. Under a source-keyed backend those gaps do not cause wrong results; under a regressed name-only key they would. LOCAL_FIXES item 33 lists them.
- **Not upstreamed.** The ROCm fork still carries the name-only key; the fix lives only in mlxcel's overlay. A re-vendor must keep item 33, which `verify-rocm-overlay` and the item's entry make visible.
- **Out-of-bounds access on regression.** The input- and output-dtype failures read or wrote twice the buffer's bytes without faulting on this host. On another allocator layout, the same regression could fault instead of returning wrong values, so those two tests run in-process on the assumption that it does not.

## 7. Learning Points

- **A cache key must cover everything the cached value depends on.** The builder's inputs here were the template args, the dtypes and the 0-d-ness, but the key covered only the first. Keying on the builder's output (the source) is the simplest way to make the key complete by construction.
- **Caller-side mitigations hide backend defects.** The dtype keys in mlxcel's launches made the common case work and so made the backend's key look correct. A test for a backend property should strip those mitigations, which the probe does by passing no template args.
- **Test every axis the bug has, not just the one in the report.** The issue named input dtypes and the 0-d case; review added the output dtype, which the original tests would have missed.
- **Use an observable sentinel to tell which compiled variant ran.** The `__constant__` `inp_shape` sentinel lets one source compile with or without a parameter and makes the result itself report which module executed.
- **Prove the test fails without the fix.** Restoring the old key and recording the failures shows each test measures the defect, and gives concrete numbers for what the bug looked like.
