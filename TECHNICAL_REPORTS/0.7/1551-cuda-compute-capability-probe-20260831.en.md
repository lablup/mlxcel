# Technical Report: PR #1551 - CUDA compute capability probe

**Date**: 2026-08-31

**Author**: mlxcel maintainers

**Status**: Implementation and sm_70 validation completed; sm_80-and-later validation deferred to a GB10 host

---

## Executive Summary

PR #1551 (issue #1537) gives mlxcel a first-class notion of CUDA compute capability. Before it, `grep -rn "compute_capability" src/ --include=*.rs` returned nothing: every architecture decision was delegated to MLX's C++ gates, so mlxcel could not say which architecture it was running on, did not record which architectures it had been compiled for, and could not tell a user that the two disagreed.

The concrete failure this removes: a published x86_64 archive is built for `80;86;89;90a;100;120`, so it carries no sm_70 code object at all. Run it on a Volta card and it dies with an opaque CUDA load error at the first kernel launch, naming neither the architectures it holds nor the device it found. It now refuses at device init with both named.

This is plumbing and diagnostics only. No kernel selection or dispatch behavior changes. It is the foundation item of epic #1536, which five later sub-issues build on.

## 1. Problem Statement

Three gaps, each a consequence of the same absence.

**No runtime answer to "what am I running on".** MLX caches `compute_capability_major` / `compute_capability_minor` per device, but nothing surfaced them to the Rust side. mlxcel could not log which architecture-dependent kernel path a run took, and could not gate its own defaults on the host.

**No build-time record of "what was I compiled for".** `build.rs` already derived an architecture list, either from `MLX_CUDA_ARCHITECTURES` or by parsing `nvidia-smi --query-gpu=compute_cap`, but that string reached CMake and stopped there. The binary did not know its own target set.

**No way to notice the two disagree.** This is not hypothetical. The release workflow ships x86_64 as `80;86;89;90a;100;120` and aarch64 as `90a;100;121`, both starting at Ampere. A source build made in a container without `nvidia-smi` falls back to `90a`, which silently produces a binary that cannot run on the very host that built it. Both cases previously surfaced as a CUDA load failure deep in the first forward pass.

## 2. Change Summary

19 files, +1155 / -18.

**Runtime probe.** `mlxcel_core::hardware::cuda_compute_capability() -> Option<(u32, u32)>`, implemented in the new `src/lib/mlxcel-core/src/cuda_arch.rs` and re-exported from `hardware.rs`. Cached in a `OnceLock` because compute capability cannot change within a process. Backed by a new `gpu_compute_capability(index)` bridge function reading MLX's own `device_info` map.

**Build-time record.** `build.rs` resolves the architecture list once in `resolve_cuda_architectures()`, passes that string to CMake, and records it as `cargo:rustc-env=MLXCEL_CUDA_ARCHITECTURES`.

**Mismatch check.** `arch_list_coverage()` evaluates the compiled list against the running device under CUDA's compatibility rules; `cuda_arch_mismatch()` turns a no-coverage verdict into a `CudaArchMismatch`; `enforce_cuda_arch_compatibility()` raises it; and `initialize_runtime_checked()` calls that at device init, so `generate`, `chat`, `embed`, `rerank`, `detect` and the server all refuse before the first kernel.

**Diagnostics.** The startup device report carries the architecture picture next to `Detected N GPU(s)` in both CLI and server. `MLXCEL_TRACE_ARCH` prints capability, compiled list, and coverage verdict once per process. The CUDA `QuantizedMatmul` overlay additionally reports which quantized-matmul path the dispatcher chose on its first call.

**Docs.** `MLXCEL_TRACE_ARCH` row in `docs/environment-variables.md`; a cross-reference from the CUDA architecture selection section of `docs/installation.md` naming the two cases that trigger the refusal.

## 3. Technical Decisions

### 3.1 The probe reads MLX's cached `device_info`, not CUDA directly

The alternative was a second `cudaDeviceGetAttribute` call from the Rust side, which would have pulled a CUDA header dependency into the crate and duplicated state MLX already holds.

The decisive property is subtler than deduplication. Reading device properties never loads a cubin. That is what keeps the probe callable on exactly the binaries the mismatch check exists to report on. Had the probe required anything that touches compiled device code, it would have failed on the wrong-architecture binary and the check would have been useless in its only interesting case. It also means the same symbol links on Metal and CPU-only builds, where the two capability keys are simply absent from the map and the probe reports `None`.

### 3.2 One resolution point feeds both CMake and the crate

`resolve_cuda_architectures()` is called once in `build.rs::main`. The same string goes to CMake's `MLX_CUDA_ARCHITECTURES` and to `cargo:rustc-env`. What the build compiles for and what the binary reports are therefore the same string **by construction**, not by two procedures that happen to agree today. This rules out the drift class where someone changes the detection logic on one path and not the other, which would make the mismatch check confidently wrong.

### 3.3 Coverage encodes CUDA's real rules, not equality

The naive check is `device in list`. It would be wrong in a way that matters: an sm_80 cubin runs on sm_86, and the shipped release matrix depends on exactly that. Equality would report a mismatch on a working configuration and refuse to start.

`entry_coverage()` therefore models three entry variants and two code-object kinds:

- **Generic** (`80`, `86`): a cubin covers the same major version at an equal-or-higher minor (`major ==`, `minor <=`); PTX JITs forward, including across majors, so the rule is tuple comparison `(entry) <= (device)`.
- **Architecture-specific** (`90a`): covers its exact target only. This matches the semantics that make `90a` load-bearing for MLX's Hopper quantized kernel.
- **Family-specific** (`f`): carries forward within its major.

`-real` and `-virtual` qualifiers select which code objects an entry emits. `arch_list_coverage()` takes `.max()` over per-entry verdicts, so a cubin match is preferred to a PTX match; both run, but a cubin match means no JIT at first launch, which is worth reporting distinctly.

### 3.4 Two deliberate safety valves

**An unparseable list is "unknown", never "covers nothing".** `MLX_CUDA_ARCHITECTURES` accepts spellings this parser does not model, including `native` and `all-major`. Treating a parse failure as no-coverage would let a parser gap manufacture a startup failure on a perfectly good binary. The check degrades to a no-op instead. This is the right trade for a guard whose whole value is diagnostic, but it has a maintenance consequence: adding a new CUDA architecture spelling silently weakens the check rather than breaking loudly, so a parser case and a unit test should be added together.

**`MLXCEL_DEVICE=cpu` bypasses the refusal.** Someone holding a wrong-architecture archive has exactly one workaround left, and the guard must not take it away.

### 3.5 The check fires at device init

`initialize_runtime_checked()` sits at device initialization rather than in each command, so all six entry points inherit it, and the refusal happens before any kernel launch rather than during one.

## 4. Validation

Host: Tesla V100-PCIE-32GB (compute capability 7.0, sm_70), CUDA 12.9.41, driver 575.51.03, x86_64. This is the only GPU on the machine.

- 26 `cuda_arch` unit tests pass under `--features cuda` and again on a build without it. They cover exact match, PTX-forward match, `90a`, `f`, `-real` / `-virtual`, cubin-preferred-over-PTX, the no-match case, and the error message. None need a GPU.
- 13 `execution::runtime` tests pass, including that the refusal tracks `cuda_arch_mismatch()` exactly and applies only when the GPU was actually requested.
- `cargo clippy --lib --tests -- -D warnings`, `cargo fmt --check`, `git diff --check` all clean. Full CI green.
- On an `MLX_CUDA_ARCHITECTURES=70` build, the probe reports the real device: `compute capability 7.0 (sm_70); compiled for [70]; coverage: cubin`, printed exactly once across 8 decode steps. With `MLXCEL_TRACE_ARCH` unset, neither trace line appears and generation is unchanged.
- The negative case was built and run rather than argued. An `MLX_CUDA_ARCHITECTURES="80;86;89;90a;100;120"` build produces a 249 MB `libmlx.a` against 155 MB for the single-architecture sm_70 build, which is independent evidence that six architectures really are compiled in. Running that binary on the V100 exits non-zero before touching a kernel, with an error naming both the compiled list and the device found. Under `MLXCEL_DEVICE=cpu` the same binary starts normally on the CPU.

## 5. Validation Limits and Follow-up

This machine has no Ampere-or-later device of any kind. The following were left unticked rather than assumed, and belong to the `## GB10 (sm_121) continuation` section of epic #1536:

- `cuda_compute_capability()` returning `Some((12, 1))` on GB10. Only the `Some((7, 0))` half ran on hardware. The bridge's `major * 1000 + minor` unpacking and the `121` list entry are covered by unit tests, but no sm_121 device executed this code.
- `cuda_compute_capability()` returning `None` on a Metal build. No macOS host was available. The Linux CPU-only build exercises the identical branch, `device_info` without the two capability keys, and that is stated as the nearest reachable evidence rather than claimed as a Metal run.
- No behavior change on existing platforms, and the GB10 baseline unmoved. Verified only on sm_70. No GB10 or Apple Silicon baseline was measured and no throughput claim is made for either.

The full `cargo test --features cuda` and `cargo test` suites were not run. Each cold MLX / CUTLASS / cuDNN configuration on this host took 30 to 47 minutes, so validation was scoped to the touched modules plus the end-to-end runs above, with CI covering the rest.

## References

- Issue #1537, epic #1536 (including its GB10 continuation section)
- Release architecture matrices: `.github/workflows/release.yml` (aarch64 `90a;100;121`, x86_64 `80;86;89;90a;100;120`)
- Architecture resolution and auto-detection: `src/lib/mlxcel-core/build.rs`
- Coverage rules: `src/lib/mlxcel-core/src/cuda_arch.rs`
