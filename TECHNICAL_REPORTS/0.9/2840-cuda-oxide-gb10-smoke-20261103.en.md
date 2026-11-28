# PR #2840: Add a cuda-oxide artifact smoke for GB10

**Date**: 2026-11-03
**Status**: In review (PR open; closes #2831)
**Languages**: Rust, Python, Bash, Markdown
**Risk Level**: Low (isolated spike package; the only shared change is a path-filter narrowing in CI)

## Executive Summary

PR #2840 builds and executes a cuda-oxide kernel on the GB10 and freezes the artifact contract that the remaining experiment depends on. A bounded vector-add kernel was compiled with the pinned NVIDIA/cuda-rust revision, exported as PTX and an `sm_121` cubin, and validated against a manifest whose argument layout comes from the emitted PTX signature and `cuobjdump` kparam records. Fourteen device cases covering small, odd, block-aligned, and tail sizes passed bitwise against host references, 69,913 elements in total, and Compute Sanitizer reported zero errors. The package lives under `spike/` as its own workspace root, so ordinary mlxcel builds still resolve no cuda-oxide crate and stay on the application Rust pin of 1.97.1.

## 1. Problem Statement

The feasibility review (#2835) concluded that the first uncertainty is whether cuda-oxide produces correct kernels on the target GPU, and that this must be settled before writing an inference kernel or an MLX adapter. No oxide kernel had been compiled or executed before this PR. The target also carried a specific trap: the CuTe preview rejects `sm_121a`, so an artifact built for `sm_100a` or `sm_120a` cannot be relabeled as GB10 evidence, and the emitted target has to be demonstrated rather than assumed.

The issue therefore asked for an isolated package, a reproducible build and smoke harness, a consumable artifact with an explicit ABI manifest, a clean-build check, and real GPU execution under `gpu-lock`. Compile-only results do not satisfy the gate.

## 2. Technical Decisions

### 2.1 A spike package with its own workspace root

`spike/cuda-oxide-smoke/Cargo.toml` carries an empty `[workspace]` table, matching the existing `spike/iree-ffi` and `spike/rust-emitter` packages. The root workspace, the root `Cargo.lock`, and `cargo tree -p mlxcel` resolve no cuda-oxide crate, and the application toolchain stays at 1.97.1. The spike pins 1.99.1 in its own `rust-toolchain.toml` because the codegen backend is a rustc plugin and must match the compiler that loads it. Its `Cargo.lock` is committed deliberately: the package produces a binary and a device artifact, so the dependency resolution is part of what makes the artifact reproducible.

### 2.2 The `.oxart` container is not the artifact

cuda-oxide embeds device code in an `.oxart` section inside the host binary. The section is a versioned container with a magic header, a target string, payload records, and entry records. `scripts/oxide_artifact.py` parses the container, extracts the PTX payload, and refuses to continue unless the compiler's separate `.ptx` sidecar is byte-identical to the embedded payload. `ptxas` assembles the committed cubin from that PTX.

The manifest records the ABI version, entry symbol, source revision, container fields, both digests, every argument's ordinal, width, pointer-ness, byte offset, and parameter space, and the launch constraints. Validation is not circular: the hand-written claim is `abi/vecadd.schema.json`, and `validate` re-parses the committed PTX, runs `cuobjdump -elf` and `-sass` on the cubin, and compares the result against both the schema and the manifest. It also checks that the container payload digest equals the committed PTX digest and that the manifest source equals the schema revision.

### 2.3 A base `sm_121` target, not a relabeled architecture-specific one

The GB10 reports compute capability 12.1. cuda-oxide's automatic device hint on this host says `sm_121a`, and all 14 cases also passed with an explicitly built `sm_121a` module. The committed artifact uses the base `sm_121` target because it is the portable match for the detected capability. The PTX directive, the cubin header, and the loader agree, and the module executed on the device. This says nothing about Tensor Core support; the CuTe path was not exercised.

### 2.4 Launch constraints are checked against the harness, not against themselves

The emitted PTX carries no `.reqntid` or `.maxntid` and the cubin has no thread-count attribute, so the block size cannot be derived from the artifact. `smoke.sh` and `sanitize.sh` capture the harness output and `oxide_artifact.py check-launch` requires every PASS line's block to equal the manifest block dimension, its grid to equal `ceil_div(n, block)`, and the RESULT counts to match the parsed lines. A wrong grid or block fails the check.

### 2.5 CI exclusions stay scoped to this package

The new `.rs` and `Cargo.toml` files matched the `rust`, `deps`, and `cpu_link` path filters, so those jobs ran and failed on a pre-existing runner defect: the self-hosted Linux runner cannot find BLAS for the `mlxcel-core` CMake build. The first fix excluded `spike/**`, which also removed the trigger for the two older spike packages. The final change narrows the exclusions to `!spike/cuda-oxide-smoke/**` in the `rust`, `deps`, `rocm`, and `cpu_link` filters. The runner defect itself is untouched and remains visible on core PRs, for example main run 41584274906.

## 3. Verification

The orchestrator re-ran the harness after the review fixes rather than accepting the implementation report:

| Check | Result |
|---|---|
| `scripts/build.sh sm_121` | manifest valid, payload name `cuda_oxide_smoke.ptx`, PTX 8.8, target `sm_121`, six `u64` arguments |
| `scripts/smoke.sh sm_121` | 14/14 cases, 69,913 elements, exact-bitwise, launch check valid |
| `scripts/sanitize.sh sm_121` | `ERROR SUMMARY: 0 errors`, launch check valid |
| `scripts/check-unsupported-target.sh` | exit 101 with `expected sm_<capability> with an optional 'a' suffix` |
| `scripts/check-clean-build.sh` | no cuda-oxide crate in the workspace, root lockfile, or `mlxcel` dependency graph; `rustc 1.97.1` asserted |
| `scripts/check-lint.sh` | fmt clean, clippy `-D warnings` |
| `scripts/check-guards.py` | eight malformed-input guards: clean non-zero result, no traceback, no hang |
| Artifact reproduction | PTX and cubin digests identical across rebuilds; sidecar equals the committed PTX |
| Tampered manifest | `c_ptr` width 32 rejected against schema 64 |
| Digests | PTX `737559830058e1d142a80c3ce7afc668d5326a4fb1b15bf020f2f03fc71381a8`, cubin `4ab363a33b5b66ab75ddbbf30b8891e22e34b06559f6508f9f1593d6cdb199a6` |

Evidence was recorded on `spark-102`: NVIDIA GB10, compute capability 12.1, driver 580.178.04, CUDA 13.0 V13.0.88, LLVM/Clang 21.1.8, rustc 1.99.1, cargo-oxide 0.3.0, source revision `b631530700a8a4a5f165456f2d71fe9195b323b5`. The upstream devcontainer was not used because the host provides the toolchain natively; it remains the documented fallback.

Two implementation reviews and one security review found defects, and the fixes address all of them. The first round caught a payload-record struct layout that read the name length as the high half of the name offset, so the manifest recorded an empty payload name and the evidence described that defect as a format property. The same round added the real launch cross-check, a guarded `sanitize.sh`, package-scoped CI exclusions, a rustc version assertion, SIGPIPE-safe pipeline checks, corrected cubin wording, and provenance validation. The security round found a zero-length `.oxart` bundle that could loop forever with growing memory, manifest-controlled artifact paths that could escape the artifact root, an unbounded alignment exponent and subprocess wait, and two paths that crashed instead of reporting. The launch check now also compares the parsed case count and element sum against declared totals, so an internally consistent shortened run fails. `scripts/check-guards.py` feeds every one of these malformed inputs through the real code and requires a clean non-zero result.

## 4. Change Summary

| Item | Value |
|---|---|
| Files changed | 24 before this report |
| Lines added | +3088 |
| Lines deleted | -1 |
| New package | `spike/cuda-oxide-smoke/` (kernel, harness, artifact tooling, ABI schema, scripts, committed artifacts) |
| Evidence | `docs/mlxcelverse/cuda-oxide-smoke-evidence.md`, linked from the feasibility review |
| Shared changes | `.github/workflows/ci.yml` path filters, `scripts/ci/check_binary_assets.py` allowance for the committed cubin |
| Runtime code in mlxcel | None |
| Tests | 14 device cases in the standalone harness; no mlxcel test suite change |

## 5. Follow-up Actions

- #2832 remains blocked on this gate and now inherits the target, artifact, and manifest contract.
- #2833 and #2834 stay blocked on their predecessors.
- The self-hosted Linux runner's BLAS discovery failure in `cargo-clippy` and `CPU-only link` is pre-existing and affects core PRs; it needs its own fix. This PR only stops treating the isolated spike as an input to those jobs.
- The artifact contract has no CI gate of its own. `oxide_artifact.py validate` needs only Python and `cuobjdump`, so a cheap job is possible if the contract is expected to keep changing.
- The committed cubin is assembled by `ptxas`; cuda-oxide's `--materialize-cubin` producer was not used and would need its own manifest entry if adopted.
