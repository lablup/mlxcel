# Technical Report: PR #2018, guard the four remaining custom-kernel launchers on ROCm

**Date**: 2026-09-28

**Status**: Implemented and validated on the gfx1151 host; pending merge. Metal and CUDA were not exercised, since neither is available here; the argument that they are unaffected is by reachability and is stated below rather than measured.

**Languages**: C++ (HIP/CUDA/Metal dispatch), Rust (cxx bridge and callers)

**Risk level**: Medium. The diff reaches seven production model files, though every new code path is unreachable on Metal and CUDA.

## Executive summary

mlxcel dispatches each fused kernel between a Metal port and a CUDA port. Until issue #1803 the test was `!metal::is_available()`, read as "CUDA"; #1803 replaced it with a named backend kind but left the false arm meaning "Metal". On ROCm that arm is taken, `fast::metal_kernel` throws, and because the bridge declarations are not `Result`, the throw crosses a `noexcept` cxx extern and ends the process. Issue #1885 fixed the two sampler launchers. This PR closes the same defect at the four that remained: `fused_add_rms_norm`, `fused_rope_qk_append`, `ssm_update_kernel`, and `run_fused_moe_two_kernel`.

The change is a refusal, not a port. Every guard added here is a placeholder that says "this backend has no kernel port"; supplying the ports is issue #1814.

## Problem statement

A process abort is a worse failure than an error for three reasons: it takes the whole server down rather than one request, it produces no typed value a caller can branch on, and its message names the port that happened to be tried rather than the reason. The last of these actively misdirects. `[metal_kernel] No Metal back-end.` on an AMD host reads as a missing Metal installation, when the real fact is that a ROCm GPU is present and healthy and simply has no port for that kernel.

The defect is also easy to miss by inspection, because the routing layer above these launchers is correct. Production gates on a support predicate and takes a graph fallback, so nothing in ordinary serving reaches the bad arm. What reaches it is a direct call: tests, benchmarks, and any caller that skips the predicate.

## Change summary

Ten dispatch sites of this shape exist. They were classified mechanically, by checking each site's preceding 40 lines for a `custom_kernels_available()` guard, rather than by reading, so that none was missed:

| Already guarded | Unguarded, fixed here |
|---|---|
| `gumbel_max_sample` (#1885) | `fused_add_rms_norm` |
| `rejection_sample` (#1885) | `fused_rope_qk_append` |
| `paged_attention_decode` (#1803) | `ssm_update_kernel` |
| `paged_attention_decode_v2_partial` | `run_fused_moe_two_kernel` |
| `paged_attention_merge_states` | |

- `src/lib/mlx-cpp/turbo/fused_norm.cpp`, `fused_rope_append.cpp`, `src/lib/mlxcel-core/cpp/mlx_cxx_kernels.cpp`: a `custom_kernels_available()` guard before the port is selected, throwing a message that names the entry point, the missing port, and the fallback the caller should take.
- `src/lib/mlxcel-core/src/lib.rs`: five bridge declarations become `Result`. Five rather than four, because `run_fused_moe_two_kernel` is reached through two entry points, `fused_moe_expert_kernel` and `fused_moe_geglu_kernel`.
- Ten call sites updated across `switch_layers.rs`, `gemma4.rs`, `qwen3_next.rs`, `falcon_h1.rs`, `nemotron_h.rs`, `granitemoehybrid.rs`, `plamo2.rs`, `layers.rs`, and `fused_moe_parity_tests.rs`.
- Two comments that this change made false are corrected.

## Technical decisions

### The `Result` conversion is the load-bearing half, not the guard

Adding a guard without converting the declaration converts one abort message into a different abort message. A C++ throw crossing a `noexcept` extern is `std::terminate` regardless of what the message says. The two halves have to land together, and a future launcher guard is incomplete without the matching declaration change.

### Each guard was checked against its production gate before being added

A guard placed above a path production actually reaches would convert a working graph fallback into a throw. Each was verified:

- `fused_add_rms_norm_available()` and `fused_rope_qk_append_available()` are `mlxcel::custom_kernels_available()` directly.
- `ssm_kernel_available()` returns `cu::is_available()` off Apple, which is the no-CUDA stub on a ROCm build and answers false.
- `fused_moe_enabled()` folds `custom_kernels_available()` into its env check.

All four answer false on ROCm, so production takes its graph path and no guard is reachable from it. This is also why the gate shows no regression: the guarded paths were already not being taken.

### The callers split three ways, and uniformity would be wrong in either direction

The MoE sites use `.ok()`. The function wrapping them returns `Option` and already returns `None` for every unsupported configuration, so "this backend has no port" is an instance of the contract it already has, and the caller falls back to the SwitchGLU graph path. A direct call on a portless backend is a legitimate use, not a defect, so `expect` there would panic on correct code.

The SSM, `layers.rs`, and parity-test sites use `expect`. Each already gates on its support predicate, so an error at those sites means the gate and the launcher disagree with each other. Folding that into a silent fallback would hide a real bug in exactly the layer this work is trying to make trustworthy.

Applying one idiom everywhere would have produced a defect in whichever half it did not fit.

### Two stale comments corrected rather than left

`fused_moe_enabled()` justified its port term with "its bridge function does not return `Result`, so on a backend without a port the `fast::cuda_kernel` throw ends the process". That was accurate under #1803 and is false after this change. The term itself stays, for a different and weaker reason now recorded in its place: deciding before building the flattened arguments is cheaper than building them and having the launcher refuse.

`fused_add_rms_norm_eligible()` claimed the launcher's exceptions are not recoverable. That still holds for contract violations such as a shape mismatch, but no longer for a missing port. The comment now separates the two classes instead of asserting the stronger claim.

A comment that describes a constraint which no longer exists is worse than no comment, because the next reader designs around it.

## Validation

- Full ROCm gate on gfx1151, `cargo test --workspace --profile test-fast --features rocm --no-fail-fast -- --test-threads=1`: one failing target, `-p mlxcel-core --lib`, unchanged from before this branch. That failure is the nvfp4 abort of issue #1806 and is the only `terminate called` in the run.
- `cargo fmt --all -- --check`: clean.
- `cargo clippy -p mlxcel --lib --tests -- -D warnings`, the exact command CI runs: clean.
- The refusal was exercised by running it rather than inferred from tests passing. A throwaway probe on the ROCm build returned `[fused_add_rms_norm] no custom kernel port for this GPU backend; mlxcel's callers take the graph fallback instead` as a typed `Err`, with the process intact.

### Method note worth keeping

Two of the ten call sites do not appear as type errors. Ignoring a `Result<()>` is the `unused_must_use` lint, not a type mismatch, so `cargo check` passed on them and only the `-D warnings` clippy surfaced them. Both are decode hot-path callers in `layers.rs`. A `Result<T>` conversion is caught by the compiler; a `Result<()>` conversion needs the lint, and a workflow that stops at `cargo check` will ship the gap.

## What is not verified

- **Metal and CUDA were not run.** Neither is available on this host. The claim that they are unaffected rests on `custom_kernels_available()` being true there, which makes every new branch unreachable, plus the unchanged success paths. It is an argument, not a measurement.
- **Only one of the four refusals was probed.** `fused_add_rms_norm` was confirmed by execution. The other three share the identical guard shape and declaration change, but were verified by compilation and by the unchanged gate rather than by being called.
- **The MoE `.ok()` path is not exercised by the gate.** On ROCm, `fused_moe_enabled()` is already false, so production never calls the kernel and never reaches the new `.ok()`. That branch is covered by reading, not by a test.

## Remaining work

Issue #1814 supplies the ports these guards stand in for. The path is open rather than speculative: `fast::hip_kernel` exists in the ROCm overlay at `src/lib/mlx-cpp/patches-rocm/mlx/fast.h:96`, and issue #1862 proved the three-arm switch end to end with the BitNet kernel.

Two facts established while investigating that are worth carrying into #1814. First, `fast::hip_kernel` compiles through hipRTC at runtime (`patches-rocm/mlx/backend/rocm/jit_module.cpp`), the same model Metal and CUDA use, so the ahead-of-time compile cost that made per-dtype instantiation expensive in issue #1853 applies to the overlay's `.hip` files and not to mlxcel's own fused kernels. Second, the binding constraint is the wavefront width: gfx1151 is wave32 while CDNA parts are wave64, a reduction fold that starts at 16 silently folds half the lanes on wave64 and produces a finite, plausible, wrong result, and `static_assert(warpSize == 32)` does not compile in HIP because `warpSize` is an object with an `operator int()`. Issue #1862 guards it with a preprocessor `#error` instead, and every ported reduction kernel needs the same.
