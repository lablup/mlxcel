# PR #2835: Document cuda-oxide feasibility and initial experiment plan

**Date**: 2026-11-03
**Status**: In review (documentation only; refs #2831, #2832, #2833, #2834)
**Languages**: Markdown
**Risk Level**: Low (no runtime code, build, or configuration changes)

## Executive Summary

PR #2835 records whether NVIDIA's cuda-oxide Rust CUDA compiler is worth adopting in mlxcelverse and commits to a bounded, evidence-gated experiment order. The review recommends compiling selected Rust kernels separately, loading their PTX/cubin through the existing MLX precompiled-module path, and keeping MLX ownership of allocation, lazy evaluation, streams, and CUDA Graph scheduling. It selects fused residual-add plus RMSNorm as the first kernel and splits the work into four issues (#2831 through #2834): toolchain and artifact validation on GB10, a standalone kernel with a numerical oracle, a default-off MLX adapter, and a matched benchmark that produces an expand, investigate, resolve, or stop decision. Production enablement, backend replacement, and an independent inference engine are explicitly excluded; the engine is a conditional follow-up that requires measured evidence of an MLX execution constraint.

## 1. Problem Statement

cuda-oxide is an alpha SIMT compiler that lowers Rust MIR through Pliron and LLVM to device code. Its installation guide requires Linux, Ampere-or-newer GPUs, CUDA Toolkit 13.0+, driver 580+, LLVM/Clang 21+, and a Rust 1.99.1 toolchain, while mlxcel pins Rust 1.97.1. The project previously considered two directions without separating their costs: adopting the compiler for selected kernels inside mlxcelverse, or building a native CUDA inference path alongside MLX. The second direction carries model execution, weight and KV storage, sampling, and serving work before any evidence that the generated kernels are correct or faster on the target GPU.

The review also had to separate ordinary SIMT compilation from architecture-specific matrix support. The inspected CuTe preview supports `sm_100a` and `sm_120a` and rejects `sm_121a`, so its published validation is not evidence for GB10 Tensor Core acceleration. A basic SIMT kernel with a verified emitted target is the only defensible first gate.

The documentation is needed now because four implementation issues depend on the artifact and ABI contract it defines. Without the review, each issue would restate a different integration boundary, and the first issue could scope an engine rather than a kernel probe.

## 2. Technical Decisions

### 2.1 Kernel integration before an independent engine

The highest uncertainty is whether cuda-oxide produces correct, useful kernels on the actual target GPU, not whether a Rust-authored engine could exist. Building an engine first would add weight loading, device memory management, KV storage, and scheduling before that question is answered, and would make kernel and executor effects inseparable in any performance comparison. The chosen order is independent kernel package, then MLX integration and comparison, then an engine only if a reproducible execution constraint survives a bounded adapter improvement.

### 2.2 Separate artifacts, thin adapter

The recommended path compiles Rust kernels in a separate package and consumes immutable PTX/cubin artifacts from a C++ MLX adapter. MLX keeps allocation, lazy evaluation, streams, and graph scheduling; no oxide compiler enters ordinary mlxcel builds. The review records that the precompiled branch in `jit_module.cpp` is a promising lower-level reuse point but not a complete custom-kernel adapter: a new kernel still needs an MLX primitive that allocates outputs, registers inputs and outputs with the encoder, packs arguments, and schedules execution during evaluation.

### 2.3 Artifact and ABI honesty

The `.oxart` container can hold PTX, cubin, or IR payloads and is not executable PTX. Its current entry metadata is not a complete C++ argument schema, so the plan requires a checked manifest with digest, entry symbol, scalar widths, pointer and length order, alignment, dtype, layout, aliasing, launch constraints, and ABI revision. The adapter must choose the existing CUDA path before enqueue when the manifest, target, dtype, or artifact does not match, and an execution-time CUDA fault must surface as an error rather than a retry.

### 2.4 Gates and stop conditions

Each issue carries its own acceptance gate and the batch stops at defined points. #2831 stops if the target or toolchain cannot execute correctly without substantial upstream work. #2832 requires actual GB10 execution and numerical parity; compile-only or skipped tests do not satisfy it. #2833 stops if correct scheduling requires replacing the backend. #2834 applies proposed thresholds agreed before measurement: all correctness gates pass, no unexplained end-to-end regression above 2% outside noise, and either a repeatable kernel benefit of at least 10% with useful end-to-end impact, a documented cold-start reduction, or a concrete maintenance benefit. A measured negative result is a valid final outcome; absent hardware evidence cannot close the investigation.

## 3. Verification

The change was checked against the current tree rather than accepted from prose:

| Claim | Verification |
|---|---|
| All 11 local source links resolve | Each target exists, including `src/backend/mod.rs`, `src/backend/session.rs`, `src/lib/mlxcelverse/kernels/kernel_port.h`, `src/lib/mlxcelverse/kernels/fused_norm.cpp`, the CUDA `jit_module.cpp` and `device.cpp` overlays, and `src/lib/mlxcel-core/src/fused_norm_parity_tests.rs` |
| Precompiled branch exists in the CUDA JIT loader | `jit_module.cpp` takes the precompiled blob at the `builder()` result and publishes it instead of compiling source |
| Kernel ports are per-backend | `kernel_port.h` resolves a `CustomKernelFunction` per backend and treats a null entry as a first-class absent port |
| CUDA Graph ordering is backend-owned | `device.cpp` maintains captured-child-graph barriers (issue #2462) |
| Evidence revisions are real | mlxcel `d771735d` is an ancestor of `main`; the MLX pin `7ca40033` is the `GIT_TAG` in `src/lib/mlxcelverse/CMakeLists.txt` |
| External links resolve | All 12 GitHub and documentation URLs returned HTTP 200 at review time |
| Formatting | No em dash, no trailing whitespace, no tabs, `git diff --check` clean |

What remains unverified is stated in the document itself: no compiler was installed, no oxide kernel was compiled, no GPU work was launched, and no driver or GPU combination was validated. The document does not claim performance, production readiness, GB10 Tensor Core acceleration, or backend replacement feasibility.

## 4. Change Summary

| Item | Value |
|---|---|
| Files changed | 2 |
| Lines added | +215 |
| Lines deleted | 0 |
| New document | `docs/mlxcelverse/cuda-oxide-feasibility.md` (213 lines) |
| Index entry | `docs/README.md`, mlxcelverse maintenance section |
| Runtime code | None |
| Tests | None; documentation-only change verified by link checks and whitespace checks |

## 5. Follow-up Actions

- #2831 remains the first ready gate: isolated package, reproducible build, artifact manifest, and actual GB10 execution under `gpu-lock`.
- #2832, #2833, and #2834 stay blocked on the preceding issue and its required GPU evidence.
- The plan's effort range (2 to 5 engineer-weeks for a decision-quality kernel proof) excludes upstream compiler fixes, hardware waiting time, production hardening, and an independent engine.
- Production packaging, default enablement, other operator families, and the native engine remain outside this batch.
