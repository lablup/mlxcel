# cuda-oxide artifact and standalone-execution smoke (issue #2831)

This package proves the cuda-oxide artifact boundary and standalone execution on an NVIDIA GB10 before any MLX integration or kernel expansion. It is the first gated experiment from [the feasibility review](../../docs/mlxcelverse/cuda-oxide-feasibility.md).

Scope is deliberately narrow: one bounded elementwise vector add, a reproducible build, a committed PTX/cubin artifact with a validated ABI manifest, and a GPU smoke harness. There is no MLX adapter, no model execution, no Tensor Core or CuTe work, and no production packaging.

The evidence from the verified run is in [docs/mlxcelverse/cuda-oxide-smoke-evidence.md](../../docs/mlxcelverse/cuda-oxide-smoke-evidence.md).

## Build isolation

`Cargo.toml` contains an empty `[workspace]` table, so this package is its own workspace root. It never joins the mlxcel build graph: root `cargo metadata` does not list it, root `Cargo.lock` resolves no cuda-oxide crate, and an ordinary mlxcel build needs neither LLVM 21 nor the cuda-oxide compiler. `scripts/check-clean-build.sh` asserts all three, and `scripts/build.sh` fails if the package's own dependency graph ever resolves an `mlxcel` crate.

`Cargo.lock` is committed deliberately. The repository ignores lockfiles for library crates, but this spike produces a binary and a device artifact, so the transitive resolution is part of what makes the artifact reproducible.

The application Rust pin stays at 1.97.1. `rust-toolchain.toml` in this directory selects cuda-oxide's pinned 1.99.1 only for commands run here, because the codegen backend is a rustc plugin and must match the compiler that loads it.

## Layout

| Path | Purpose |
| --- | --- |
| `src/main.rs` | The `vecadd` kernel and the standalone host harness with 14 bounded cases |
| `abi/vecadd.schema.json` | Hand-written ABI claim: entry symbol, argument order and widths, launch constraints, ABI version |
| `artifacts/` | Committed PTX, cubin, and the generated, validated manifest |
| `scripts/env.sh` | Toolchain discovery and prerequisite checks |
| `scripts/build.sh` | Pinned build, `.oxart` extraction, cubin assembly, manifest generation and validation |
| `scripts/smoke.sh` | GPU execution under `gpu-lock` |
| `scripts/check-unsupported-target.sh` | Proves a bad target fails with an actionable message |
| `scripts/check-clean-build.sh` | Proves ordinary mlxcel builds need no oxide compiler |
| `scripts/oxide_artifact.py` | Parses `.oxart`, assembles the cubin, writes and validates the manifest |
| `scripts/oxide_format.py` | Pure parsers for the `.oxart` container, PTX signatures, and `cuobjdump` kparam records |

## Prerequisites

Verified host: Ubuntu on aarch64, NVIDIA GB10 (compute capability 12.1), driver 580.178.04.

- CUDA Toolkit 13.0 with `nvcc`, `ptxas`, `cuobjdump`. The harness searches `/usr/local/cuda-13.0`, then `/usr/local/cuda`; override with `CUDA_HOME`.
- LLVM and Clang 21 with the NVPTX target, clang resource headers, and `libclang.so`. Preferred host install from apt.llvm.org:

  ```bash
  sudo apt-get install clang-21 libclang-21-dev libclang-common-21-dev lld-21 llvm-21 llvm-21-dev
  ```

  The discovery order is `LLVM21_PREFIX`, then `/usr/lib/llvm-21`, then `llvm-config-21` on `PATH`. A user-space LLVM 21 release tarball also works: unpack it and set `LLVM21_PREFIX`. The verified run used the apt packages at 21.1.8 and reproduced byte-identical artifacts with the 21.1.0 release tarball.
- Rust 1.99.1 from rustup with `rust-src`, `rustc-dev`, and `llvm-tools`:

  ```bash
  rustup toolchain install 1.99.1 --profile minimal \
      --component rust-src,rustc-dev,llvm-tools,rustfmt,clippy
  ```

- `cargo-oxide` built from the pinned NVIDIA/cuda-rust revision, outside this repository:

  ```bash
  git clone https://github.com/NVIDIA/cuda-rust /tmp/cuda-rust
  git -C /tmp/cuda-rust checkout b631530700a8a4a5f165456f2d71fe9195b323b5
  cargo +1.99.1 install --path /tmp/cuda-rust/cuda-oxide/crates/cargo-oxide --locked
  cargo-oxide setup   # builds and caches librustc_codegen_cuda.so for 1.99.1
  ```

`scripts/env.sh` fails with a specific message for each missing prerequisite, including the `libclang.so` symlink that bindgen needs.

## Running

```bash
cd spike/cuda-oxide-smoke
scripts/build.sh sm_121        # build, export artifacts, validate the manifest
scripts/smoke.sh sm_121        # run 14 cases on device 0 under gpu-lock
scripts/check-unsupported-target.sh
scripts/check-clean-build.sh
```

GPU execution also runs cleanly under Compute Sanitizer:

```bash
. scripts/env.sh
gpu-lock run --tag issue-2831 -- cargo oxide sanitize --tool memcheck --arch sm_121
```

Every GPU-allocating command, including the sanitizer run, goes through `gpu-lock run --tag issue-2831 -- ...`. A missing or busy lock is an error, never a reason to run unguarded. Compile-only results do not satisfy the GPU gate.

## Artifact contract

cuda-oxide embeds device code in a `.oxart` container section inside the host binary. That section is not PTX: it has an `OXIDEART` magic, a format version, a target string, payload records, and entry records. `scripts/oxide_artifact.py` parses the container, takes the PTX payload, and refuses to proceed if the compiler's separate `.ptx` sidecar differs from the embedded payload. The committed `.ptx` and `.cubin` are therefore extracted payloads, never renamed containers.

The manifest records:

- digests and sizes of the PTX and cubin, plus the digest of the `.oxart` payload they came from;
- the entry symbol and the container's entry records;
- the declared ABI version and the source revision;
- the emitted PTX version and target;
- every argument's ordinal, width, pointer-ness, byte offset in the cubin, and parameter space;
- the launch constraints: block shape, grid rule, shared memory, buffer dtype and length rule.

`validate` re-parses the committed PTX and runs `cuobjdump -elf` and `-sass` on the cubin. It compares the argument layout against both the hand-written schema and the manifest, checks the digests, and rejects a stale or edited manifest. A deliberately tampered manifest fails validation.

## Target

The committed artifact targets `sm_121`, the base target for the GB10's compute capability 12.1. It is not an `sm_100a` or `sm_120a` artifact relabeled: the PTX directive, the cubin target, and the device loader all agree. The harness also executed all 14 cases with `sm_121a` as a secondary probe; the committed contract is the more portable base target.
