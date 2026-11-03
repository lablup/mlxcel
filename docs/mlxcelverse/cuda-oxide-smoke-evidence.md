# cuda-oxide smoke evidence: artifacts and standalone execution on GB10

Evidence for [issue #2831](https://github.com/lablup/mlxcel/issues/2831), the first gated experiment in [the cuda-oxide feasibility review](cuda-oxide-feasibility.md). Run date: 2026-11-03 (UTC). Host: `spark-102`.

Status: the GPU acceptance gate is satisfied. The kernel was compiled and executed on the GB10, every case was checked against a host reference, and Compute Sanitizer reported zero errors. Nothing in this report is a compile-only result.

## Environment

| Component | Value |
| --- | --- |
| Host | `spark-102`, Linux 7.0.0-1019-nvidia, aarch64 |
| GPU | NVIDIA GB10, compute capability 12.1, driver 580.178.04 |
| CUDA Toolkit | 13.0, V13.0.88 (`nvcc`, `ptxas`, `cuobjdump` under `/usr/local/cuda-13.0`) |
| LLVM | 21.1.8 from apt.llvm.org (`llvm-21`, `llvm-21-dev`, `clang-21`, `libclang-21-dev`, `libclang-common-21-dev`, `lld-21`), `/usr/lib/llvm-21` |
| LLVM cross-check | 21.1.0 user-space release tarball, `LLVM-21.1.0-Linux-ARM64` |
| Rust | rustc 1.99.1 (58cf8f9c3 2026-10-13), cargo 1.99.1 (5f94df478 2026-08-27) |
| Rust components | `rust-src`, `rustc-dev`, `llvm-tools`, `rustfmt`, `clippy` |
| cargo-oxide | 0.3.0, built from the pinned source revision |
| Source revision | NVIDIA/cuda-rust `b631530700a8a4a5f165456f2d71fe9195b323b5` |
| Application Rust pin | 1.97.1, unchanged |

The application pin and the cuda-oxide pin stay separate. `spike/cuda-oxide-smoke/rust-toolchain.toml` selects 1.99.1 only inside the spike directory.

The upstream devcontainer was not used. This host provides CUDA 13.0 and LLVM/Clang 21 natively (the apt.llvm.org packages above), so the build ran on the host toolchain and every byte executed on the host driver. The devcontainer remains the documented fallback for hosts that cannot install LLVM 21; the user-space LLVM 21.1.0 tarball is the second fallback and produced identical artifacts here.

The pinned revision needed no change. Its `rust-toolchain.toml` names 1.99.1, and the whole pipeline built and ran on that compiler.

## Target

| Property | Emitted value |
| --- | --- |
| PTX ISA version | 8.8 |
| PTX target directive | `.target sm_121` |
| PTX address size | 64 |
| cubin target | `sm_121` |
| cubin producer | ptxas 13.0, V13.0.88 |

`sm_121` is the base target for compute capability 12.1. It is not a relabeled `sm_100a` or `sm_120a` artifact: the PTX directive, the cubin header, and the loader all agree, and the module executed on the GB10.

Two target observations are worth recording:

- cuda-oxide's automatic device hint on this host reports `sm_121a`, the architecture-specific spelling. All 14 cases also passed with an explicitly built `sm_121a` module. The committed artifact uses the base `sm_121` target because it is the portable match for the detected capability.
- The CuTe preview rejects `sm_121a`, per the feasibility review. This experiment uses only ordinary SIMT code and does not exercise that path. Neither target result says anything about Tensor Core support on GB10.

## Commands

Toolchain environment (exported by `spike/cuda-oxide-smoke/scripts/env.sh`):

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_OXIDE_LLC=/usr/lib/llvm-21/bin/llc
export LLVM_CONFIG_PATH=/usr/lib/llvm-21/bin/llvm-config
export LIBCLANG_PATH=/usr/lib/llvm-21/lib
export PATH=/usr/lib/llvm-21/bin:/usr/local/cuda-13.0/bin:$HOME/.cargo/bin:$PATH
```

Full run, from `spike/cuda-oxide-smoke`:

```bash
scripts/build.sh sm_121
scripts/smoke.sh sm_121
scripts/sanitize.sh sm_121
scripts/check-unsupported-target.sh
scripts/check-clean-build.sh
scripts/check-lint.sh
```

`scripts/build.sh` runs `cargo oxide build --arch sm_121`, extracts the `.oxart` section with `objcopy`, parses the container, copies out the PTX payload, assembles it with `ptxas -arch=sm_121`, writes `artifacts/vecadd.manifest.json`, and validates that manifest against the PTX and cubin. `scripts/smoke.sh` and `scripts/sanitize.sh` both run the harness under `gpu-lock` and then check the reported block and grid against the manifest.

Every GPU run went through `gpu-lock run --tag issue-2831 -- ...`. No GPU command ran unguarded.

## Execution results

`scripts/smoke.sh sm_121` executed 14 cases on device 0. Each element was compared bitwise against a host reference computed as `a[i] + b[i]`.

| Class | Sizes | Result |
| --- | --- | --- |
| small | 1, 2, 3, 16, 17 | pass |
| odd | 33, 999 | pass |
| tail | 255, 257, 1000, 1025, 65537 | pass |
| block-aligned | 256, 512 | pass |

Final harness line:

```text
RESULT standalone=1 mlx=0 cases_passed=14 cases_total=14 elements_checked=69913 comparison=exact-bitwise
```

The 65537-element case spans 257 blocks and leaves a one-element tail. The 255, 257, 1000, and 1025 cases each exercise a partial final block. Bitwise equality is the right criterion here: both sides perform exactly one IEEE-754 f32 add on exactly representable inputs.

Compute Sanitizer, run over the same 14 cases under the same lock:

```text
========= ERROR SUMMARY: 0 errors
Compute Sanitizer completed with exit code 0.
```

The kernel returns from the output bounds check before reading `a` or `b`, so the rounded-up grid does not touch memory outside the allocated buffers. Memcheck confirms that for the tail cases rather than assuming it.

## Artifact and manifest

The host binary carries a `.oxart` section. `objcopy --dump-section .oxart=...` produced a container whose first eight bytes are `OXIDEART`; it is not PTX and was never treated as PTX.

| Container field | Value |
| --- | --- |
| Format version | 1 |
| Bundle name | `cuda-oxide-smoke` |
| Target | `sm_121` |
| Compile options | 0 (FMA contraction allowed, no debug info) |
| Payload kind | `ptx` (0x0100) |
| Payload size | 2379 bytes |
| Payload name | `cuda_oxide_smoke.ptx` |
| Entries | kernel `vecadd`, plus two internal device-function symbols |

The export step refused to continue unless the compiler's `.ptx` sidecar was byte-identical to the payload inside `.oxart`. The committed PTX is that payload.

| Artifact | Bytes | SHA-256 |
| --- | --- | --- |
| `artifacts/vecadd.sm_121.ptx` | 2379 | `737559830058e1d142a80c3ce7afc668d5326a4fb1b15bf020f2f03fc71381a8` |
| `artifacts/vecadd.sm_121.cubin` | 6360 | `4ab363a33b5b66ab75ddbbf30b8891e22e34b06559f6508f9f1593d6cdb199a6` |

The manifest records ABI version 1, the entry symbol, the source revision, the container fields, both digests, and every argument. The argument layout below comes from `cuobjdump -elf` (`EIATTR_KPARAM_INFO`) and was cross-checked against the PTX `.entry` signature.

| Ordinal | Name | PTX type | Pointer | PTX pointee align | cubin offset | cubin size | cubin space |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `a_ptr` | `.u64` | yes | 4 | 0 | 8 | global |
| 1 | `a_len` | `.u64` | no | n/a | 8 | 8 | value |
| 2 | `b_ptr` | `.u64` | yes | 4 | 16 | 8 | global |
| 3 | `b_len` | `.u64` | no | n/a | 24 | 8 | value |
| 4 | `c_ptr` | `.u64` | yes | 1 | 32 | 8 | global |
| 5 | `c_len` | `.u64` | no | n/a | 40 | 8 | value |

The writable pointer's PTX annotation assumes one byte of alignment; that is weaker than the four bytes an `f32` allocation provides, so it is safe. The validator rejects a compiler assumption stricter than the declared allocation alignment.

Launch constraints recorded in the manifest and checked against a real run: one-dimensional index space, block `(256, 1, 1)`, grid `ceil_div(n, 256)`, zero dynamic shared memory, three contiguous `f32` buffers of exactly `n` elements, and out-of-range threads returning at the output bounds check. `scripts/smoke.sh` and `scripts/sanitize.sh` capture the harness output and run `oxide_artifact.py check-launch`; it requires every `PASS` line's `block` to equal the manifest block dimension, its `grid` to equal `ceil_div(n, block)`, and the `RESULT` case and element counts to match the parsed lines. A skipped or shortened run fails there.

Manifest validation output:

```text
manifest valid: entry=vecadd target=sm_121 ptx_version=8.8 args=6 ptx_sha256=737559830058e1d1 cubin_sha256=4ab363a33b5b66ab
```

The validator is not circular. The human-written file is `abi/vecadd.schema.json`; the manifest is generated; `validate` re-derives the layout from the committed PTX and cubin. It also requires the `.oxart` payload digest to equal the committed PTX digest and the recorded source revision to equal the schema's, so a swapped payload or a drifted revision fails. A self-test that changed `c_ptr`'s width to 32 bits in a copy of the manifest failed with:

```text
FAIL arg c_ptr: manifest width_bits=32 != schema 64
manifest validation failed with 1 error(s)
```

## Unsupported target

`scripts/check-unsupported-target.sh` runs the compiler with `CUDA_OXIDE_DEVICE_ARCH=compute_121`, the libNVVM spelling rather than a device target. The build fails before emulating anything, with exit code 101 and this message:

```text
error: [rustc_codegen_cuda] Device codegen failed: PTX generation failed: invalid CUDA_OXIDE_DEVICE_ARCH `compute_121`: expected sm_<capability> with an optional `a` suffix
```

## Clean-build isolation

`scripts/check-clean-build.sh` ran with `CUDA_OXIDE_LLC`, `LLVM_CONFIG_PATH`, `LIBCLANG_PATH`, and `LLVM21_PREFIX` unset and without LLVM 21 on `PATH`:

```text
workspace members: mlxcel mlxcel-core mlxcel-mlx-pin mlxcel-surgery mlxcel-xla
no cuda-oxide crate in Cargo.lock
mlxcel dependency graph has no cuda-oxide crate
PASS clean-build toolchain='rustc 1.97.1 (8bab26f4f 2026-07-14)' oxide_compiler=absent
```

The spike package is absent from the mlxcel workspace, the root lockfile resolves no cuda-oxide crate, the dependency graph of the `mlxcel` package resolves no cuda-oxide crate, and a real workspace member (`mlxcel-mlx-pin`) compiled with the application toolchain.

## Lint

`scripts/check-lint.sh` runs `cargo fmt --check` and `RUSTC_BOOTSTRAP=1 cargo clippy --release --locked -- -D warnings` on the spike's 1.99.1 toolchain. Both pass. `RUSTC_BOOTSTRAP=1` matches the upstream workspace's own gate, because the pinned `cuda-macros` expands unstable attributes on a stable compiler.

## Reproducibility

Rebuilding produced identical PTX and cubin digests across:

- repeated builds with LLVM 21.1.8 (apt);
- a build with the LLVM 21.1.0 user-space tarball;
- a rebuild after editing only host argument parsing.

The device artifact depends on the pinned cuda-oxide revision, the kernel source, and the CUDA target, not on the specific LLVM 21 patch release or on rebuild order.

## Limits

- Execution is standalone. There is no MLX adapter, no model, and no end-to-end workload here; those are #2832 through #2834.
- Only `f32` addition is exercised. NaN, Inf, denormal, rounding, and mixed-dtype policy are out of scope for this issue.
- The committed cubin is assembled by CUDA `ptxas` from the emitted PTX. cuda-oxide's `--materialize-cubin` producer was not used for the committed artifact and would need its own manifest entry if adopted.
- No performance measurement was attempted. A launch time on one kernel would not answer the feasibility question and could be misread as an integration result.
- No Tensor Core, `tcgen05`, or CuTe path was compiled or run. The design review already records that the CuTe preview rejects `sm_121a`.
