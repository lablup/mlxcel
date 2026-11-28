# Technical Report: PR #1058 - fix(paged): key the paged decode v2 JIT cache on input dtypes

**Date**: 2026-08-07
**Author**: mlxcel maintainers
**Reviewer**: implementation and security review cycle
**Status**: Partial (root cause proven and fixed; the CUDA runtime assertion the issue asks for could not be run on the implementing host)
**Languages**: C++, Rust
**Risk Level**: High (a silent wrong answer on the shipping dtype, CUDA only)

---

## Executive Summary

`paged_v2::sparse::sparse_tests::an_f16_cache_matches_the_reference_within_its_own_precision` failed on GB10 with a relative error of essentially 1.0, which is the signature of an unrelated result rather than precision loss. The fused sparse path was not computing the wrong answer; it was running a kernel that had been compiled for a different dtype. MLX generates a custom kernel's buffer parameter types from the runtime dtypes of its inputs, but only the Metal backend folds those dtypes into the JIT cache key. This PR names the input dtypes in `template_args` for the two paged decode v2 kernels so `template_arguments_hash` can tell the variants apart, and adds a guard that runs one geometry at two pool dtypes in a single process.

---

## 1. Problem Statement

### 1.1 Background

Issue #1053 reported the failure on GB10 (DGX Spark, sm_121), release profile, `--features cuda`, MLX pin `2c46b953db88965c4270cc7306eda6887a3247f2`. The tolerance at the assertion site is `3e-2`, chosen so that f16 storage rounding is already accommodated, so a relative error of ~1.0 is not a precision question. The issue explicitly ruled out widening the tolerance as a fix.

The failure had reached `main` unnoticed because the suite never produced a verdict on that platform: without `--test-threads=1` it aborts partway through with SIGABRT (#1048), since libtest drives MLX concurrently from many host threads.

### 1.2 Existing Issues

- **A silent wrong answer on the configuration that ships.** The comment at the test site records that production allocations are f16, so the pool the kernel reads is f16. The wrong result is produced on the shape that ships, and nothing throws.
- **The test's own dispatch assertion passes first.** `outcome.is_fused()` is asserted before the numbers are compared, and it passes, which correctly rules out a fallback and points at the kernel. That made the failure look like a numerics bug in the kernel body rather than a build-system one.
- **The same weakness sat under a second, differently-shaped failure.** #1054 reported `paged_v2::launch::launch_tests::chunk_size_does_not_change_the_answer` failing only in the full suite with an error of roughly 55896x. Its issue body nominated "cached kernel or graph state, including anything captured or compiled once and keyed loosely enough to collide across tests" as a candidate, which is exactly this.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| A CUDA process that runs one v2 geometry at two pool dtypes returns numbers unrelated to its inputs, without an exception | Critical | Certain when the condition is met |
| The defect is invisible on every macOS runner, so the normal development loop cannot catch it | High | Certain |
| The same weakness recurs at a new `cuda_kernel` call site | High | High, with no guard in place before #1054 |

---

## 2. Technical Review

### 2.1 Root Cause

Two MLX sources disagree about what belongs in a JIT cache key.

`mlx/backend/common/metal_kernel.cpp` appends the input dtypes to the kernel name, and says why:

```cpp
// The generated source depends on the dtypes of the inputs and outputs
// and on how each input is passed (see `write_signature`). Include them
// in the kernel name so that a given name always maps to the same source.
for (const auto& arr : inputs) {
  kernel_name += "_";
  kernel_name += get_type_string(arr.dtype());
  ...
}
```

`mlx/backend/cuda/custom_kernel.cpp` does not:

```cpp
std::string kernel_name =
    "custom_kernel_" + name + template_arguments_hash(template_args);
```

while its `build_kernel` in the same file generates each buffer parameter's type from `dtype_to_cuda_type(arr.dtype())`. `cu::get_jit_module` (`mlx/backend/cuda/jit_module.cpp`) then memoises the compiled module under exactly that name in a process-global `std::unordered_map`, and invokes the source builder only on a cache miss.

`paged_attention_decode_v2_partial` passed only int template args (`Dim`, `PageSize`, `NRep`, `QHeads`, `QGroups`, `DimsPerThread`, `NumWarps`), all derived from geometry. An f32 pool and an f16 pool at the same geometry therefore hash to one name. Whichever dtype compiles first wins for the life of the process; the other reads its buffers through the wrong pointer type.

In `sparse_tests` the `dtype::FLOAT32` cases run before `an_f16_cache_matches_the_reference_within_its_own_precision`, so the f32 module is cached first and the f16 test reads f16 storage as `const float*`. A relative error of ~1.0 is what reinterpreted bits produce.

### 2.2 CUDA-only or not

The issue asked this explicitly, and it partitions the search. Determined on Metal (M1 Ultra, macOS 26.5.2, `--features metal,accelerate`):

| Command | Result |
|---|---|
| `cargo test --profile test-fast --features metal,accelerate -p mlxcel-core --lib paged_v2 -- --test-threads=1` | 92 passed, 0 failed, including the f16 test |
| Full `mlxcel-core` lib suite, serialized, on `main` before the change | 1415 passed, 0 failed in 92.23s |

So **CUDA-only**, and for a specific reason: the Metal cache key already carried the dtypes. That also answers #1054's platform question at the same time.

### 2.3 Performance

No kernel body was touched, so nothing about the emitted code changes for a given dtype. On CUDA the module count moves from one per geometry to one per (geometry, dtype), which is the count Metal has always had. A process serves one KV cache dtype, so the steady-state count is unchanged; what changes is that a second dtype compiles its own module instead of silently reusing the first one's.

### 2.4 Compatibility & Dependencies

- **Breaking changes**: none. No public API, no config, no environment variable.
- **New dependencies**: none. `TemplateArg` already accepts a `Dtype`, and both backends already render it (Metal as a `typename` template parameter, CUDA as a `using` alias).
- **Precedent in-tree**: the fused decode-MoE kernels have always passed `{"T", T}`, which is why they never had this defect.

### 2.5 Code Quality

Metal and CUDA source strings were checked by hand for identifier collisions with `QType`, `KVType`, `VType`, and `LseType`. Each name appears exactly twice per file, once in a comment and once in `template_args`. This check mattered because the CUDA bodies are not compiled on a Metal host, so a collision there would not surface at build time.

---

## 3. Technical Decisions

### 3.1 Where to fix the key

**Context:** The defect lives in MLX, a pinned CMake-fetched dependency, and `src/lib/mlx-cpp/patches-cuda/` exists, so patching upstream was available.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Patch `mlx/backend/cuda/custom_kernel.cpp` to append dtypes, as Metal does | Fixes every current and future call site at once, including MLX's own | Adds a maintained patch to re-apply on every MLX pin bump; diverges from upstream behaviour in a way a reader would not expect |
| **Chosen: name the dtypes in `template_args` at the mlxcel call sites** | Entirely in-tree; uses a documented MLX feature (`template_arguments_hash` already hashes a `Dtype`); matches the `{"T", T}` the MoE kernels already use | Must be repeated per call site, so a new site can forget it |

**Rationale:** `template_arguments_hash` already discriminates on a `Dtype`, so the fix uses the mechanism MLX provides rather than changing MLX. It survives a pin bump untouched, and it reads the same as the pattern already established elsewhere in the tree.

**Trade-offs:** The per-call-site repetition is a real weakness, and it is why #1054 adds `scripts/ci/check_kernel_dtype_keys.py` rather than trusting the convention.

### 3.2 Whether to include `QType`

**Context:** The sparse path casts the query to f32 at `sparse.rs:580`, so `QType` looks constant there.

**Rationale for including it anyway:** it is not constant on the other entry point. `launch_tests::run_with_chunk` builds the query with `q_array(dtype::FLOAT32)` while `f16_pools_match_the_gather_reference` uses `q_array(dtype::FLOAT16)`, both reaching the same kernel. Keying only the pool dtypes would leave that pair colliding.

---

## 4. Implementation Details

`turbo/paged_attention_v2.cpp` adds `QType`, `KVType`, `VType`. `turbo/paged_attention_v2_merge.cpp` adds `VType` and `LseType`, whose only prior key was `Dim`. `Dim` is an especially weak key there, because head dims repeat across families.

Both additions carry a comment stating that they are load-bearing despite being unreferenced by the kernel body, so a later cleanup pass does not remove them as dead template parameters.

### 4.1 Regression coverage

`two_pool_dtypes_at_one_geometry_do_not_share_a_compiled_kernel` holds one geometry fixed, since every int template arg is derived from it, and moves only the pool dtype: f32, then f16, then f32 again in a single process, each checked against its own host reference. Running f32 again at the end makes the guard independent of which dtype the process happened to compile first.

The test carries an explicit note that it cannot fail on Metal. A guard whose passing is uninformative on the developer's own machine should say so, rather than leave a future reader believing it was exercised.

---

## 5. Verification Gap

The issue's acceptance criterion `cargo test --release --features cuda -p mlxcel-core --lib paged_v2 -- --test-threads=1` passes **is not met by this PR**. There is no `nvcc` on the implementing host and no CUDA node is reachable; `rexy.office.lablup` and `indominus.office.lablup` both failed to resolve their bastion on 2026-08-07.

What was verified instead:

- The change compiles and the Metal suite stays green: 93 paged_v2 tests, 0 failed, up from 92 with the new guard.
- `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets --features metal,accelerate -- -D warnings`, and `scripts/ci/check_cross_repo_refs.py` clean.
- Passing a `Dtype` template arg through `cuda_kernel` is already exercised on CUDA today by `turbo/fused_norm.cpp` and `turbo/fused_rope_append.cpp`, so it is not a new code path.

A GB10 session should run the command above and confirm both the new guard and `sparse_tests::an_f16_cache_matches_the_reference_within_its_own_precision`.

---

## 6. Lessons

- **A relative error near 1.0 is a category signal, not a magnitude.** It says the output carries no relation to the input, which points at addressing, typing, or dispatch rather than at arithmetic. Reading it as "very bad rounding" would have sent the investigation into the kernel body.
- **An assertion that passes can be the most informative line in a failure.** `outcome.is_fused()` passing eliminated the fallback path and localized the fault to the kernel, which is what made the build-system explanation reachable.
- **A backend asymmetry in a dependency is invisible from the side that is correct.** Nothing in the mlxcel tree was wrong when read on its own terms; the defect only exists relative to what CUDA does differently from Metal. That is an argument for reading the dependency's source rather than only its API.
