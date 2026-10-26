# PR #2648: Prefer Windows ROCm bundle headers over HIP_PATH and add GPU targets

**Date**: 2026-10-27
**Status**: Complete on Windows (gfx1151); the six other targets are build-verified only
**Languages**: C++, CMake, Python, Markdown
**Risk Level**: Low

## Executive Summary

PR #2648 (issue #2639) closes two gaps the Windows ROCm runtime bundle from #2607 left open.

- **Header order:** a machine-wide `HIP_PATH` no longer overrides the bundle's JIT headers, and a stderr line names both directories whenever an environment variable and a bundle disagree.
- **GPU targets:** the PR decides a seven-target list for distributable builds and makes the packaging script refuse a target the SDK has no kernel data for. Every binary now prints the targets its device code was compiled for.

On the 32 GB-RAM validation host, the seven-target binaries link only at reduced link parallelism. They are 3.9 times the size of a gfx1151-only binary.

## 1. Problem Statement

A runtime bundle exists to run one ROCm release end to end: DLLs from one directory and hiprtc headers from that directory's `include\`.

The AMD HIP SDK installer sets `HIP_PATH` system-wide. Since #2607 it outranked the bundle's headers, so on any machine with a HIP SDK installed, a ROCm 10.1 bundle compiled its fused kernels against another release's headers. This is the cross-version mix the bundle was built to prevent, and only the docs warned about it.

Separately, validated builds compiled device code for gfx1151 only, so a bundle could not run on any other RDNA GPU.

## 2. Change Summary

| Area | Change |
|---|---|
| Header order | `choose_rocm_include_dir()` takes `runtime_is_bundle`. The Windows order is now `ROCM_HOME`, `ROCM_PATH`, the bundle include (runtime source `MLXCEL_ROCM_RUNTIME_DIR` or the executable directory, with `hip\hip_runtime.h`), `HIP_PATH`, the runtime include for the build-time SDK, then the baked include. On Linux the bundle flag is false and `HIP_PATH` is unread, so the `-I` string is unchanged. |
| Notice line | The pure `rocm_include_dir_notice()` in `runtime_dir.h` prints `[mlx-rocm] hiprtc headers: <chosen> (<source>); ignoring <other> from <source>` when `ROCM_HOME`/`ROCM_PATH` override a bundle or a bundle overrides `HIP_PATH`. It is silent when both name the same directory (separators, trailing separators and ASCII case normalized). `delayload.cpp` exposes `rocm_runtime_dir_source()`. |
| Target log | The overlay CMake joins `CMAKE_HIP_ARCHITECTURES` with commas into `MLXCEL_ROCM_ARCHITECTURES`. `device.cpp` prints `[mlx-rocm] compiled HIP targets: ...` once, next to the bound-device line. |
| Target validation | Configure fails with `Not a HIP target ID` on any `MLX_ROCM_ARCHITECTURES` entry outside the HIP target-ID grammar, because the list now also lands in a C string literal. |
| Packaging | `package_windows_rocm_runtime.py` exits 1, before writing anything, naming every `--arch` that lacks `.kpack` archives or rocBLAS/hipBLASLt entries. |
| Tests | Host C++ test cases for the new order and the notice. Five packaging tests for missing targets. `tests/test_rocm_arch_validation.py` runs the real CMake validation block with `cmake -P`. |
| Docs | New order and a rewritten `HIP_PATH` rule. A "GPU targets for distributable builds" section with the measured sizes, the multiarch tarball as the kernel-data source, and link-parallelism advice. Accepted target-ID forms. `LOCAL_FIXES.md` item 66. |

## 3. Technical Decisions

### 3.1 A bundle outranks HIP_PATH, but not ROCM_HOME or ROCM_PATH

An installer sets `HIP_PATH` once for every program on the machine. A user sets `ROCM_HOME` and `ROCM_PATH` deliberately, for one shell or one run. Only the deliberate override may beat the bundle, and the notice line makes that override visible.

The security review checked the trust side. The bundle include sits in the executable's directory or in `MLXCEL_ROCM_RUNTIME_DIR`, and anyone who can write there can already replace the binary and its DLLs. Ranking it above `HIP_PATH` therefore gives no new party control over the kernels.

### 3.2 Keep the selection pure

`choose_rocm_include_dir()` and the new notice function take plain values and return strings. The host test extracts `choose_rocm_include_dir()` from `jit_module.cpp` with awk, so the test runs on any machine with a C++17 compiler, without HIP or Windows. The new parameter keeps that extraction working.

### 3.3 Decided target list and where its kernel data comes from

The list is RDNA3 `gfx1100;gfx1101;gfx1102`, RDNA3.5 `gfx1150;gfx1151` and RDNA4 `gfx1200;gfx1201`.

The installed gfx1151 TheRock tarball compiles MLX for all seven targets, because its device bitcode covers every target. Its rocBLAS and hipBLASLt kernel data and `.kpack` archives cover gfx1151 only, and the per-family tarballs each cover only their own family. AMD's `therock-dist-windows-multiarch-10.1.0.tar.gz` (4.9 GB) carries all seven; this was checked by streaming `tar -t`, without extracting. The docs and the script's error point to it.

Merging kernel data from several SDK roots was left out. Failing loudly with the missing target named is enough to stop a partial bundle.

### 3.4 Validate the target list at configure time

Each list entry becomes an `--offload-arch` flag and, after this PR, part of a C string literal. A quote in the environment variable could therefore add code to the build. Only whoever controls the build environment can set it, so the review rated it LOW.

A plain `gfx[0-9a-f]+` check would reject valid IDs such as `gfx90a:xnack+` and `gfx11-generic`. The validation therefore follows the target-ID grammar.

## 4. Validation

All runs were on Windows 11 with a Radeon 8060S (gfx1151) and ROCm 10.1.0.

- **`HIP_PATH` decoy:** `HIP_PATH` named a folder whose `hip_runtime.h` is `#error`. The bundle run JIT-compiled three kernels against the bundle's include, gave correct output, and printed the notice. The build-tree binary (build-time SDK source) hit the decoy error, as designed.
- **`ROCM_PATH` at run time:** it still won, and the notice named the bundle include it ignored.
- **Bundle runs (gfx1151 build):** `generate` with Qwen3-0.6B-4bit (74 to 78 tok/s) and Qwen3-1.7B-bf16 (26.9 tok/s) answered "Paris". Server chat, streaming and `/v1/messages` were correct. Every ROCm module loaded from the bundle's `bin`. The rebuild after the validation change generated at 76.8 tok/s.
- **Seven targets:**
  - The `-j 8` build compiled all 40 HIP objects, each carrying seven code objects, then failed to link (`link.exe` 0xC0000142 under memory pressure).
  - `cargo build -j 2 --bin mlxcel --bin mlxcel-server` linked both in 7 min 1 s.
  - Sizes against gfx1151 alone: `mlxcel.exe` 745,592,320 vs 193,465,856 bytes; `mlxcel-server.exe` 735,546,880 vs 183,420,416; kernel archive 716.6 vs 190.1 MiB.
  - The HIP step took 5 min 8 s against 3 min 6 s (different tree states).
  - On gfx1151 the seven-target binary printed all seven targets and generated "Paris" at 75 tok/s.
- **Packaging:** asking for gfx1100 and gfx1201 from the gfx1151 SDK exited 1, named both targets, and wrote nothing.
- **Checks:** the host C++ test (it fails on main), 24 Python tests, `cargo test -p mlxcel-mlx-pin --lib` (49), fmt, clippy `-p mlxcel-core --features rocm -D warnings`, overlay drift and records, and the env-var docs check all pass.

## 5. Known Gaps

- **Untested targets:** gfx1100, gfx1101, gfx1102, gfx1150, gfx1200 and gfx1201 compile but have not run; no hardware is available.
- **Build time:** no clean end-to-end seven-target build time was measured. The two HIP-step timings come from different tree states.
- **Link memory:** linking seven targets needs more memory than `-j 8` leaves on a 32 GB host. The docs recommend `-j 2`; nothing enforces it.
- **No merged-kernel bundle yet:** a seven-target bundle needs the 4.9 GB multiarch SDK, which has not been packaged end to end.
- **Linux:** the CMake validation also runs in Linux builds and has not been run on a Linux host. The Linux ROCm CI job is parked (#1811).
- **Notice on non-ASCII paths:** a non-ASCII path can print a redundant notice line, because `getenv` returns ANSI while the runtime include is UTF-8. The chosen directory is unaffected.

## 6. Learning Points

- **Installer-set variables:** a variable an installer sets machine-wide carries no user intent, and ranking it with per-run overrides lets one installed product silently reconfigure another.
- **Per-target static libraries:** these grow linearly with the target list. On Windows the link step, not compilation, hits the memory limit first, so link parallelism is the setting to lower.
- **Values copied into source:** any build input that ends up inside generated source text needs validation against its real grammar, not a simplified pattern that rejects valid input.
