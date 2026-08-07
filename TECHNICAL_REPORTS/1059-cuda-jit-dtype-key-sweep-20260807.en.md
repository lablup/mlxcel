# Technical Report: PR #1059 - fix(kernels): key every CUDA JIT launch on its input dtypes

**Date**: 2026-08-07
**Author**: mlxcel maintainers
**Reviewer**: implementation and security review cycle
**Status**: Partial (the class is swept and guarded; the minimal ordered test pair issue #1054 asks for is not delivered and is not guessed at)
**Languages**: C++, Python, YAML, Markdown
**Risk Level**: High (one of the three swept sites is reachable in production sampling, CUDA only)

---

## Executive Summary

PR #1058 fixed the two paged decode v2 kernels whose JIT cache key omitted the input dtypes. This PR sweeps the rest of the class: three more `cuda_kernel` launches carried the same defect, one of them in the sampler, where `gumbel_max_sample_accepts` admits f32, f16 and bf16 at a single `NumSplits`. It also adds `scripts/ci/check_kernel_dtype_keys.py`, wired into `make verify` and a new CI job, so a new call site cannot repeat the omission silently.

---

## 1. Problem Statement

### 1.1 Background

Issue #1054 reported `paged_v2::launch::launch_tests::chunk_size_does_not_change_the_answer` failing only in the full `mlxcel-core` lib suite on GB10, with a relative error of roughly 55896x, while passing alone and inside a scoped 92-test `paged_v2` run. Its own body nominated three candidates for the leaked process state: the autotune memo, cached kernel or graph state "keyed loosely enough to collide across tests", and environment variables.

The second candidate is right, and PR #1058 established the mechanism: `mlx/backend/cuda/custom_kernel.cpp` names a kernel `"custom_kernel_" + name + template_arguments_hash(template_args)` while generating its buffer parameter types from the runtime input dtypes, and `cu::get_jit_module` memoises the compiled module under that name in a process-global map. Metal folds the dtypes into its key; CUDA does not.

### 1.2 Existing Issues

- **The class was larger than the reported symptom.** An audit of every `template_args` initialiser in a file containing a `cuda_kernel(` call found three more sites keyed on ints alone.
- **One of them is production sampling, not a test artifact.** `gumbel_max_sample_accepts` (`turbo/sampling.cpp`) explicitly admits `float32`, `float16` and `bfloat16`, while `NumSplits` depends only on `(batch, vocab)`. Two models with the same vocabulary but different logits dtypes shared one compiled kernel on CUDA, and the second one sampled from a buffer read through the wrong pointer type.
- **`rejection_sample_accepts` is wider still.** It places no dtype restriction at all, so any float dtype reaches the kernel.
- **Nothing prevented the next occurrence.** The fix is a convention repeated per call site, and a convention with no check is a convention that goes stale.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| A CUDA server that switches between models of the same vocabulary but different logits dtype samples from garbage | Critical | Certain when the condition is met |
| A future `cuda_kernel` call site repeats the omission | High | High, without a check |
| A Metal-only launcher gains a CUDA port and inherits the defect | Medium | Moderate, and invisible to an allowlist that was not updated |

---

## 2. Technical Review

### 2.1 The audit

Every `std::vector<std::pair<std::string, TemplateArg>>` initialiser under `src/lib/mlx-cpp/turbo/` and `src/lib/mlxcel-core/cpp/` was enumerated and classified.

| Site | Prior keys | Verdict |
|---|---|---|
| `turbo/paged_attention.cpp` (v1) | `Dim`, `NRep`, `DimsPerThread`, `NumSplits` | **Fixed here**: adds `QType`, `KVType`, `VType` |
| `turbo/sampling.cpp` | `TgSize`, `NumSplits` | **Fixed here**: adds `LogitsType` |
| `turbo/sampling_rejection.cpp` | `TgSize`, `MaxRounds` | **Fixed here**: adds `ProbsType`, `ParamsType` |
| `turbo/fused_norm.cpp` | `T`, `TW`, `Dim`, `Threads` | Already correct |
| `turbo/fused_rope_append.cpp` | `T`, `HeadDim`, … | Already correct |
| `cpp/mlx_cxx_kernels.cpp` (7 sites) | `T` or `InT` present in each | Already correct |
| `turbo/sparse_v_sdpa.cpp` | `Dim`, `RepeatCount`, `NRep` | Out of scope: Metal-only, no `cuda_kernel(` |
| `turbo/turbo4_delegated_sdpa.cpp` (3 sites) | `Dim` (+ `RepeatCount`, `NRep`) | Out of scope: Metal-only |

The four out-of-scope sites are genuinely safe today, because Metal's cache key already carries the dtypes. They are not safe by construction, which is what shaped the check's rule.

### 2.2 Security

The sampler finding is the security-relevant one, though it is a correctness rather than a confidentiality issue. A wrong-typed read of the logits buffer produces an argmax over reinterpreted bits, so a served completion would be drawn from noise while every status code and latency metric stays normal. No memory-safety boundary is crossed: the buffer is the right size in bytes and the kernel reads within it, so this is a silent quality failure rather than an out-of-bounds one.

### 2.3 Performance

No kernel body changed. On CUDA each affected launch now compiles one module per (geometry, dtype) instead of one per geometry, matching Metal. Steady state is unchanged for a process that serves one dtype.

### 2.4 Compatibility & Dependencies

- **Breaking changes**: none.
- **New CI job**: `kernel dtype keys`, deliberately not behind the `changes` path filter, matching `crate-versions`. It needs no toolchain and runs in seconds, and a gate that can be skipped is the gate that gets skipped on the one PR that breaks it.
- **New make target**: `verify-kernel-dtype-keys`, added to `verify`.

---

## 3. Technical Decisions

### 3.1 How to scope the check

**Context:** The check must apply to CUDA launches and not to Metal-only launchers, or it produces four immediate false positives.

**Alternatives Considered:**

| Option | Pros | Cons |
|--------|------|------|
| Allowlist the four Metal-only sites with a reason string, as `check_crate_versions.py` does for independent crates | Explicit; the reason is recorded next to the exemption | Goes stale exactly when it matters: adding a CUDA port to an allowlisted file leaves it exempt and silently reintroduces the defect |
| **Chosen: scope by the presence of `cuda_kernel(` in the file** | Self-maintaining. A Metal-only launcher is out of scope until someone adds a CUDA port, at which point the check starts applying on its own | Slightly indirect; a reader must know why the file-level predicate is the right one, so the script says so at length |

**Rationale:** The failure mode being guarded is a *new* call site repeating the omission, and the most likely shape of that is a CUDA port added to an existing Metal launcher. An allowlist is blind to exactly that case. `CLAUDE.md` records that the prose version of the crate-version rule had already failed once for the same reason, which is why `check_crate_versions.py` inverted it; this takes the inversion one step further by removing the list.

**Trade-offs:** The predicate is file-level rather than call-level, so a file mixing a CUDA launch with a genuinely dtype-invariant one would need the invariant one to carry a dtype arg anyway. No such file exists today, and the cost if one appears is one redundant template argument.

### 3.2 Invert the variable or rename it

Not applicable to this PR, but recorded because the sibling decision in PR #1062 went the other way. Here there is no user-facing switch at all; the fix is invisible outside the compiled module name.

---

## 4. Implementation Details

`scripts/ci/check_kernel_dtype_keys.py` scans `src/lib/mlx-cpp/turbo` and `src/lib/mlxcel-core/cpp`. For each `.cpp` containing `cuda_kernel(`, it extracts every `template_args` initialiser and requires at least one entry whose value either contains `.dtype()` inline or names a local bound from a `.dtype()` earlier in the file. The second form matters because the MoE kernels write `auto T = x.inner.dtype();` and then `{"T", T}`.

On failure it names the file, line, variable, and the keys it did find, then explains the mechanism and points at issues #1053 and #1054.

### 4.1 Negative control

A check that cannot fail is not a check. `{"LogitsType", logits.dtype()}` was removed from `turbo/sampling.cpp` and the script was re-run:

```
kernel-dtype-keys: FAIL
  src/lib/mlx-cpp/turbo/sampling.cpp:430: `template_args` names no input dtype; keys are ['TgSize', 'NumSplits']
```

The line was restored and the check returned to `OK — 13 source files scanned`. This is recorded because the alternative, shipping a green check that was never observed to go red, would have been indistinguishable from shipping a check with a broken regex.

---

## 5. What This PR Does Not Deliver

Issue #1054's first acceptance criterion asks for **a minimal ordered pair of tests** that reproduces the failure. That is not here.

- Bisecting requires running the suite on CUDA, and there is no `nvcc` on the implementing host and no reachable CUDA node (re-verified 2026-08-07).
- Narrowing it analytically did not converge. Within `mlxcel-core` the v2 partial kernel is reached only from `paged_v2` tests, whose template-arg tuples do not overlap across dtypes: `launch_tests` uses `PageSize=32` at `Dim` 64 and 128, `sparse_tests` reshapes the pool so `PageSize=1`, and `cascade_launch_tests` uses `Dim=16, PageSize=8`.
- The merge kernel was the promising lead, since `pages_per_chunk 1` is the first loop value and the only one that runs a merge at all. Its one caller outside `paged_v2` is MLA `split_kv`, whose test uses `DIM=5` and whose production path casts to f32 at `split_kv.rs:278` before merging. **Refuted, not confirmed.**

On Metal the whole question is invisible: the full suite is 1416 passed / 0 failed, `chunk_size_does_not_change_the_answer` included.

A GB10 session should run `cargo test --release --features cuda -p mlxcel-core --lib -- --test-threads=1` on `main` and report whether the failure is gone. #1058 plausibly removed it, since it re-keyed both the partial and the merge kernel, but "plausibly" is the honest word.

---

## 6. Incidental Finding

One run of the serialized suite reported two autotune tests failing; an identical re-run passed 1416 / 0. They are host-load flakes, not a regression:

- The module documents itself as "Every test here is CPU-only: the `FakeOp` double implements `TunableOp` by sleeping for a per-tactic duration and overrides `sync` to a no-op", so a C++ kernel template argument cannot reach them.
- They pass 55 / 0 in isolation.
- Their sleeps are `Duration::from_micros` at 800 against 400, while the host was at a load average of 40.

`autotune_tests.rs:153` already warns that these "depend on how accurately the host honors `thread::sleep`". Worth its own issue: an assertion resting on microsecond sleep accuracy can flip on any loaded runner, CI included.

---

## 7. Lessons

- **Fix the class, not the instance, when the instance was found by accident.** #1053 surfaced because one test happened to run f32 before f16. Nothing made the sampler's exposure visible, and it is the more serious of the two.
- **Prefer a predicate over a list when the list's stale case is the dangerous one.** The allowlist would have been correct on the day it was written and wrong on the day someone ported a Metal kernel to CUDA, which is precisely the day the check exists for.
- **Refute your own best hypothesis in writing.** The MLA merge lead was plausible enough to state as fact; checking `DIM=5` and the f32 cast took two minutes and turned a confident wrong answer into an honest open question.
