# PR #2651: Run Windows CUDA builds from a relocatable runtime bundle

**Date**: 2026-10-27
**Status**: Build-verified on Windows x64 without an NVIDIA GPU; runs on NVIDIA hardware still open
**Languages**: C++, Python, Makefile, Markdown
**Risk Level**: Medium

## Executive Summary

PR #2651 (issue #2640) applies the Windows ROCm runtime design from #2606/#2607 to the Windows CUDA build.
- **Before:** a `--features cuda` binary needed the CUDA toolkit and cuDNN at the paths recorded at build time.
- **After:** the binary loads every CUDA, cuDNN and NVRTC DLL from one runtime directory, chosen in this order:
  - `MLXCEL_CUDA_RUNTIME_DIR`;
  - otherwise the executable's own folder, when it holds the bundle marker;
  - otherwise the build-time layout.
- **Packaging:** a new script, `scripts/package_windows_cuda_runtime.py`, builds the bundle.

The user decided the issue is verified by building only, since the development host has no NVIDIA GPU. The x64 CUDA build, the host and packaging tests, an NVRTC header check and a GPU-less loader harness all pass. Every step after device initialization still needs an NVIDIA host.

## 1. Problem Statement

The Windows CUDA build (#2599) delay-loads the CUDA libraries from the toolkit's `bin\<arch>` directory recorded at build time, and NVRTC reads headers from the recorded `include\`. Running the binaries on another machine meant installing the same toolkit at the same path. The docs named the bundling of `openblas.dll` and the CUDA, cuDNN and NVRTC DLLs as not done.

## 2. Change Summary

| Area | Change |
|---|---|
| Directory choice | New header-only `overlays/cuda/.../runtime_dir.h` with a pure `choose_cuda_runtime_dir()`. `MLXCEL_CUDA_RUNTIME_DIR` must be absolute and hold the runtime, otherwise it is an error with no fallback. The value is normalized first: `.` and `..` resolved, trailing separator dropped. The executable directory qualifies when it holds the marker DLL named in the binary's own delay-import table. The build-time layout is used unchecked, so developer builds behave as before. |
| Marker | nvcc links cudart statically, so there is no cudart DLL to look for. The marker is `cublasLt64_13.dll`. |
| Loader hook | In a bundle, the DLLs the libraries open by plain name are checked before any CUDA DLL loads, then preloaded by full path: `nvrtc64_*`, `nvrtc-builtins*`, `cublas64_*`, `cublasLt64_*`, `nvJitLink_*` and seven required `cudnn_*` sub-libraries. `cusparse64_*` is checked only. A failed load raises the delay-load exception naming the expected path. The build-time path keeps upstream's per-library lookup and its `../nvidia/...` wheel fallback. |
| NVRTC headers | Header order: `CUDA_HOME`, then the bundle's `include\` (in or beside the runtime directory), then `CUDA_PATH`, then the baked toolkit include. When `CUDA_HOME` is set, the bundle include goes last and supplies only CUTLASS/CuTe. |
| Packaging | The script stages or zips `bin\` (DLLs plus OpenBLAS), `include\` (JIT headers, CCCL from the build's `_deps`, CUTLASS) and `licenses\` (CUDA EULA, cuDNN, CCCL, CUTLASS). Every DLL glob must match exactly one file, and licenses are identified by their text. It fails naming any missing DLL, header or license, writes the zip through a `.partial` file, and refuses an existing output. |
| Tests | `make verify-cuda-runtime-dir` (part of `make verify`) runs the host C++ test of the selection, marker, preload checks and include order. `tests/test_package_windows_cuda_runtime.py` has the packaging tests. |
| Docs | `docs/installation.md`: runtime directory and bundle, unbundled plain-name loads, redistribution caveat, and a limits-table row naming what is build-verified only. `docs/environment-variables.md`: `MLXCEL_CUDA_RUNTIME_DIR`, `CUDA_HOME` and `CUDA_PATH`. |

## 3. Technical Decisions

### 3.1 Same rules as ROCm

The CUDA hook follows the rules #2607 settled for ROCm:
- one directory per process;
- no per-DLL fallback to another installation;
- the current directory and `PATH` are never searched for a DLL the hook handles;
- the delay-load helper's exception is raised instead of returning null, because null makes the helper retry with a plain search;
- paths are printed as UTF-8.

The build-time layout is the one exception: it keeps upstream's lookup and fallback unchanged, so a developer build with no bundle and no variable loads exactly as it did.

### 3.2 A bundle outranks CUDA_PATH, not CUDA_HOME

The CUDA installer sets `CUDA_PATH` machine-wide, the same situation as `HIP_PATH` in #2648, so it ranks below the bundle's headers. `CUDA_HOME` is a deliberate override and stays first.

The implementation review found the first version did not work this way. The bundle include sat ahead of the chosen CUDA header directory on NVRTC's include list, and NVRTC uses the first match. The fix puts the bundle include last when `CUDA_HOME` is set.

`CUDA_HOME/include` was not moved to the front instead. The CUDA 13 toolkit has a top-level `include/nv/` that would hide the build's CCCL `nv/`.

### 3.3 DLL list and header set derived without a GPU

A generate run's loaded-module list cannot be recorded without a GPU. The DLL closure therefore comes from the import tables and embedded DLL names of the x64 CUDA 13.4.3 and cuDNN 9.27 DLLs. For example, cuBLASLt opens NVRTC by name and cuSPARSE opens nvJitLink by name. It is hard-coded and marked as needing confirmation on hardware.

The JIT header set was found by compiling each of the 17 headers MLX embeds with NVRTC 13.4 against only the staged bundle's `include\`; NVRTC compilation needs no device. That run added `cuda_runtime.h`, `mma.h` and `crt/mma.h(pp)`, which CUTLASS includes.

### 3.4 Redistribution terms

Several headers the JIT needs are not on the CUDA EULA Attachment A list of redistributable files. These include `cooperative_groups*`, `math_constants.h`, `cuda_runtime.h`, `mma.h` and `driver_types.h`. The docs state this and advise confirming terms with NVIDIA before publishing a bundle. Prebuilt archives stay out of scope.

## 4. Validation

The host was Windows 11 x64 with a Radeon 8060S and no NVIDIA GPU. The toolkit was NVIDIA's sha256-checked windows-x86_64 redistributables: CUDA 13.4.3 and cuDNN 9.27.0.42.

- **Build:** this is the first x64 Windows CUDA build, and the build glue needed no change. With `MLX_CUDA_ARCHITECTURES=120`, the cold build took 32 min and the rebuild after review 17 min 37 s. `mlxcel.exe` is 133,756,928 bytes. `llvm-readobj` shows cuBLASLt, cuFFT, cuSOLVER, cuDNN and NVRTC as delay imports and no CUDA DLL among the regular imports.
- **Checks:** clippy `--features cuda -D warnings`, fmt and `cargo test -p mlxcel-mlx-pin --lib` (49) pass. The host test passes and fails on mutated selection logic. 25 packaging tests and the env-var docs check pass.
- **Packaging on the real archives:** the bundle has 1830 files (1968 MiB). A renamed file fails naming it, and cuDNN's `LICENSE` is refused as the CUDA EULA.
- **Loader harness:** a program linking the overlay `delayload.cpp` calls cuBLASLt, NVRTC, cuFFT and cuDNN version functions through the delay-load hook. With the toolkit renamed away and `CUDA_PATH` unset:
  - every CUDA, cuDNN and NVRTC module loaded from the bundle, from the exe directory, through the override, and through an override with a trailing separator;
  - `nvcuda.dll` came from System32;
  - a relative override, a wrong folder, a removed cuFFT, cuSPARSE, `cudnn_ops` or `cudnn64` file, and a removed marker each failed with the expected path;
  - with the toolkit present, the developer layout loaded from the toolkit as before.
- **mlxcel itself:** `mlxcel generate` and `mlxcel-server`, both from the bundle and from the developer build, exit with 0xC0000409 at the first device query, before any delay-loaded CUDA DLL is requested. The host has a stale `nvcuda.dll` but no NVIDIA GPU.

## 5. Known Gaps

- **Runtime on NVIDIA hardware:** none of these have run on an NVIDIA Windows host:
  - generate, including the first NVRTC JIT, and the server from a fresh bundle folder with the toolkit moved away;
  - the module list of a real run;
  - a removed DLL failing from mlxcel itself;
  - developer builds behaving unchanged.

  Four of the six acceptance criteria are open for this reason.
- **Unconfirmed lists:** the DLL list and the header set come from static analysis and an NVRTC-only check.
- **Plain-name loads outside the bundle:** the CUDA libraries also open `nvcuda.dll`, `nvcudart_hybrid64.dll`, `zlibwapi.dll`, `cask_profile_interface.dll`, `nvtileiras64.dll` and two optional cuDNN sub-libraries by plain name. The bundle neither ships nor checks them, so the standard search order applies, including the current directory and `PATH`, as in developer builds. Closing this needs a decision (an enforced manifest, or a process-wide search change) and belongs with the DLL-planting follow-up, #2641.
- **Packaging gaps:**
  - `--stage-dir` leaves a partial tree on failure (fail-safe, because the bundle then stops with the missing file's path);
  - the script requires two cuDNN sub-libraries the hook treats as optional, so an older cuDNN 9 fails to package;
  - the script needs `--deps-dir` when a target directory holds two CUDA build outputs.
- **Linux:** the only Linux behavior change is that an empty `CUDA_HOME` now counts as unset.

## 6. Learning Points

- **Include order:** a header search path list is a priority order; NVRTC takes the first match. A precedence rule written in a selection function only holds if the include list is built in the same order.
- **Static runtime:** with a statically linked runtime there is no runtime DLL to detect a bundle by, so the marker has to be a DLL the binary actually delay-loads.
- **Mixed license files:** redistributable archives that all ship a file named `LICENSE` overwrite each other when unpacked together. Identifying license files by content avoids shipping the wrong terms under the right name.
