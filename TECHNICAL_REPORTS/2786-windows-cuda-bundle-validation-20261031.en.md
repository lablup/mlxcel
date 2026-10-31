# PR #2786: Record the Windows CUDA runtime bundle validation on N1X

**Date**: 2026-10-31 (generated before merge)
**Status**: In review
**Languages**: Markdown, Python (one comment)
**Risk Level**: Low (no runtime code changes)

## Executive Summary

#2640 added the Windows CUDA runtime bundle and closed on build-only verification, because its host had no NVIDIA GPU. #2653 ran the bundle on an arm64 RTX Spark N1X with the toolkit renamed. `mlxcel generate` from a cold JIT cache and a `mlxcel-server` chat request both worked from the bundle folder and through `MLXCEL_CUDA_RUNTIME_DIR`, and no listed DLL or header was missing. This PR replaces the build-only statements in the docs with that result and documents two gaps the run exposed.

## 1. Problem Statement

The DLL closure and `JIT_HEADERS` in `scripts/package_windows_cuda_runtime.py` came from static analysis of the x64 DLLs and an NVRTC-only compile. Four of the six #2640 acceptance criteria were unchecked, including which DLLs a real run loads and everything after device initialization.

## 2. Technical Decisions

### 2.1 Reach cuDNN on purpose

A short prompt on Gemma 4 E4B loads no cuDNN DLL, so that run alone would have left the cuDNN half of the list untested. A 3950-token prompt on Llama 3.1 8B reaches cuDNN's attention path, and all ten cuDNN DLLs loaded from the bundle. Module lists were sampled every 500 ms over the process lifetime, since `listdlls` was not installed. A DLL mapped and unmapped within one interval would be missed.

### 2.2 Test the environment override with build-tree binaries

If the whole bundle moves, its binaries still find the DLLs beside them and the variable is not what selects the directory. The override runs used `target` binaries against the moved bundle's `bin`, with the toolkit away, so `MLXCEL_CUDA_RUNTIME_DIR` was the only source. The runtime line confirmed that source.

### 2.3 Docs only

Nothing was missing, so no code changed besides the comment that said the list was unconfirmed. `scripts/server_chat_smoke.sh` fails on this host because it opens `/tmp` from Windows Python. It is outside the bundle's scope, so the server check used `curl` and the script was left alone.

## 3. Change Summary

- `docs/installation.md`: the Windows limits row, the DLL list provenance, the plain-name load results, the CCCL license source, and notes on the Visual C++ runtime and `openblas.dll`.
- `scripts/package_windows_cuda_runtime.py`: the provenance comment.

## 4. Results

| Check | Result |
|---|---|
| Bundle folder, toolkit away | generate 45.1 tok/s, server chat answered |
| `MLXCEL_CUDA_RUNTIME_DIR`, toolkit away | generate 44.8 tok/s, server chat answered |
| Toolkit in place, no bundle | generate 43.6 tok/s, runtime line names the toolkit |
| Removed `cudnn_ops`, `nvrtc-builtins`, `cusparse` | stops naming the expected path; `cudnn_ops` exit code 127 |
| Removed `cufft` | generate succeeds, because cuFFT loads on first call |
| Listed but never loaded | cuFFT, cuSOLVER, cuSPARSE |

Plain-name results are on #2641: `nvcuda.dll` from System32, the driver's `nvcudart_hybrida64.dll` from its DriverStore folder, and `zlibwapi.dll`, `cask_profile_interface.dll` and `nvtileiras64.dll` not loaded.

## 5. Trade-offs and Residuals

- **Not run:** x64 GPU hosts, cuFFT and cuSOLVER calls, and a machine with only the display driver. The validated host has Visual Studio, and `MSVCP140.dll` and `VCRUNTIME140.dll` came from System32, so the claim that a bundle needs only the driver is still untested.
- **License stand-in:** the validated bundle used a CCCL header in place of the CCCL license because the `NVIDIA/cccl` tag's `LICENSE` was found afterward. Packaging with that file was checked separately. It does not affect loading.
- **Decoys:** the run records what loads, not whether the search order can be planted. That stays with #2641.
- **Local gate:** `make verify-cuda-runtime-dir` needs `c++`, which this host lacks. CI's overlay host test job ran and passed it.
- **Other manual:** the issue's docs criterion also covers a manual kept outside this repository, updated in its own PR, still open when this report was written.

## 6. Learning Points

- Static DLL lists depend on the architecture: the x64 list names `nvcudart_hybrid64.dll`, and arm64 loads `nvcudart_hybrida64.dll`.
- A bundle test that loads nothing from half the list looks identical to a pass. Checking the module list is what exposed that the first generate run never reached cuDNN.
- When a harness narrows `PATH` on Windows, `timeout` resolves to `TIMEOUT.EXE`. The first run failed that way without starting `mlxcel`, which an exit-code check alone did not make obvious.
