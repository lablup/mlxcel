# Technical Report: PR #2304 - Advance MLX Pin and Reconcile Backend Ownership

- **Date**: 2026-10-14
- **Status**: Implementation and available hardware validation completed; ordinary PR CI is pending and the expanded coordinated integration gate is deferred
- **Risk Level**: High (a 111-commit dependency update changes CPU, CUDA, Metal, and ROCm integration, and ROCm hardware validation is unavailable)

## Executive Summary

PR #2304 advances the pinned `ml-explore/mlx` source from `59d600b5e64c238427d0f8d897ab7c682ef4d3d2` to `243bdf09c0cccbb85ca6a686dab3cc0102cf1f7e`, a range of 111 commits and 312 changed upstream files. It reconciles every CPU/CUDA/Metal whole-file overlay, adapts the ROCm overlay to move-only allocator ownership and new primitives, preserves required numerical and runtime behavior, and adds ownership and aligned MXFP coverage. CUDA compilation and numerics, explicit RHS-route execution, combined-tree Kimi controls, and production-head Metal CI pass. Actual ROCm GPU and M5-specific validation remain unavailable.

## 1. Problem Statement

### 1.1 Background

mlxcel copies local source overlays over a pinned MLX checkout. A pin-only update can silently replace upstream fixes with stale complete files, so each overlay must be compared with pristine Git objects and either rebased, retained with zero path drift, or retired. This pin also changes allocator ownership and primitive APIs across backends.

### 1.2 Identified Risks

- **Overlay replacement**: stale whole-file overlays could discard upstream changes across the 111-commit range.
- **Ownership and lifetime**: upstream allocation now returns move-only `allocator::Data`; incorrect handoff could double-release storage, donate borrowed memory, or release buffers before asynchronous work completes.
- **Backend behavior**: CUDA gathered GEMM, Metal gathered-row broadcasting, ROCm fallbacks, SDPA sinks, floor division, gated-delta VJP, and odd-length inverse FFT all cross the updated dependency boundary.
- **Validation attribution**: Kimi requires the independent PR #2296 source fix, so a combined validation tree must record both source heads and must not attribute that fix to PR #2304.

## 2. Implementation Details

### 2.1 CPU, CUDA, and Metal Overlay Reconciliation

The original inventory contained 30 upstream-backed overlays and two local helpers. Eleven changed files were rebased, 17 files were retained with zero path-specific drift, and the two extracted helpers remained local. Two overlays were retired: merged upstream #4514 supplies the CUDA handle overlay's non-throwing teardown semantics, and upstream #4554 supplies the full #910 async-copy completion and CTA ordering residual from `cuda/device/qmm_sm80.cuh`. The overlapping upstream #4531 was closed without merge because #4514 covered it. Removing the #910 provenance and local barrier hunk reproduces the pristine target file byte for byte. The resulting inventory contains 30 retained files. The reconciliation record in `docs/upstream/mlx-pin-243bdf09-reconciliation.md` documents each disposition and retained behavior.

The retained behavior includes CPU float accumulation, CUDA batch chunking and dtype/runtime controls, upstream gathered GEMM routes, and the #1599 `B * M` broadcast-row correction in both Metal gathered matrix paths. The Xcode 27 cooperative-tensor workaround comes from upstream #4594, avoiding import of the older complete-file workaround.

### 2.2 Allocation Ownership and ROCm Retarget

CUDA reduction intermediates now transfer `allocator::Data` directly. ROCm adds an owned-data bridge because the public deleter constructor represents borrowed storage; borrowed slab shells keep their custom release and remain non-donatable. The retarget keeps fork commit `75915908dfe5028335d318b10340313744fd3a8d`, records 109 backend files, 16 core files, 47 local fixes, and five reviewed core residuals, and adapts compiled callbacks, reduction intermediates, SDPA sinks, gated-delta VJP fallback, floor division, and odd inverse FFT output sizing.

The host ownership regression exercises move-only transfer, owned and borrowed releases, empty allocation, retained shared ownership, and exception cleanup against the actual overlay header. It models pending command ownership but does not execute HIP or prove GPU completion ordering.

### 2.3 CUDA Gathered GEMM Follow-ups

Commit `6424eb916e2bdfe7c576cba9f8991d9b614ae9d9` passes `encoder.device()` to the retained gathered GEMM tile selector, fixing the initial candidate build failure. Commit `5a0d98303f9830ecf9cd34501dc48c3a6db8306f` adds N=384 sorted-prefill MXFP4/MXFP8 reference cases because the existing N=320 cases do not satisfy aligned RHS admission. The default 128-row run selects retained #629 grouped GEMM first, so route validation disables it with the documented `MLXCEL_GATHER_QMM_GROUPED=0` control. Both default tile selection and explicit M=128 selection then pass and generate fresh RHS PTX for f16/bf16 × MXFP4/MXFP8.

## 3. Review Status

- Source-integration review found no blocking issue through `6424eb916e2bdfe7c576cba9f8991d9b614ae9d9`.
- Independent overlay and allocation-ownership review found no blocking issue.
- The stale CUDA ops provenance comment was corrected in `5a0d98303f9830ecf9cd34501dc48c3a6db8306f`.
- Security review found no critical, high, or medium findings. Its possible low-severity #910 duplicate barrier was resolved by proving upstream #4554 covers the complete functional residual and retiring the overlay.

## 4. Verification

### 4.1 Passed

1. Pin/parser validation resolves exactly `243bdf09c0cccbb85ca6a686dab3cc0102cf1f7e`; all 27 pin/parser tests pass.
2. CPU/CUDA/Metal pristine-object provenance review reports no unexplained path drift or conflict markers. ROCm records and pristine-object drift checks pass, including 29 overlay tooling checks and the host ownership regression.
3. Contract gates pass: version, kernel dtype keys, kernel port dispatch, llama compatibility, Python tooling with 123 tests, and formatting.
4. CUDA no-run compilation at source `6424eb916e2bdfe7c576cba9f8991d9b614ae9d9` completes in 14m23s after the missing device argument was fixed.
5. Ten focused numerical groups at source `6424eb916e2bdfe7c576cba9f8991d9b614ae9d9` complete with zero failures. The logs contain 94 test executions; five MXFP tests are intentionally repeated by the tile-override group, so 94 is an execution count rather than a unique-test count. The frozen binaries are recorded in `candidate-2303-numerics/provenance.json` with root SHA-256 `8477d85748c60dd1298ab71b4c0e2f8c615f516583641b2ccff562a1e5b60f03`, core `e1734a8aae76a7809e5e9d7bcb3046de585b3833144ee1cfe334a14b1ff9b77e`, and surgery `8b873c576b8696806ff125caf2d61e7c637a09b6562fc9ca478181aa3bb1218e`.
6. Production-head Apple Silicon workflow run `38492190049` passes at `3aeccc2b7ac39ea17be3e9959cbb4111cfc02782`: 12,285 tests passed, 0 failed, 390 ignored, plus 51 video tests passed and 0 failed. The actual step logs are archived in `/tmp/mlxcel-auto-20261012T1250/logs/metal-2303-3aeccc2b-pass.zip`.
7. The retired generated-cache header was restored from pristine pin `243bdf09` before the production-head rebuild. `/tmp/mlxcel-auto-20261012T1250/2303-generated-cache-restoration.json` records the cached file changing from SHA-256 `a9046084ccb93381e366e0d9d6730a4fd1cbd0d49e3dff5a43d2df7e66ffbebf` to pristine SHA-256 `c3c4ba7727db7e67929fa4966d6fffa4d11bf47bb8773213c7d8899ec03b9374`.
8. Combined source `53473340` builds in 3m57s. Its final default numerical run passes 38 switch-layer tests, one explicit aligned M=128 test, and six global-scale tests (45 total, 0 failed), recorded in `/tmp/mlxcel-auto-20261012T1250/2303-final-numerics-validation.json`.
9. Explicit RHS validation passes both N=384 runs with `MLXCEL_GATHER_QMM_GROUPED=0`, once with default tile selection and once with `MLXCEL_QMM_TILE_M=128`. A fresh cache contains four `gather_qmm_rhs_sm80` PTX artifacts covering float16 and bfloat16 with MXFP4 and MXFP8; provenance and binary hashes are recorded in `/tmp/mlxcel-auto-20261012T1250/2303-rhs-validation.json`.
10. Combined-tree Kimi validation passes three focused tests and five exact model controls, including answer `4`, marker `AMBER-RAVEN-7429`, chunked and unchunked prefill, and default SDPA. A separately reviewed 58-token output gives a coherent two-sentence Rayleigh-scattering explanation with no token-zero repetition. Source `5347334039e48d9444be77f2aebeaac17c35c6d1`, pin `243bdf09`, and binary SHA-256 `b52d70388e0420582c4f60f3341d570c2afd736685b37e2c0c6e50ed9eaf5fae` are recorded in `2303-latestpin-model-validation.json` and `2303-latestpin-natural-language-validation.json`; the combined tree's independent PR #2296 fix remains separately attributed.

### 4.2 Pending or Unavailable

- **Ordinary PR CI**: seven jobs are queued for runner capacity; their queued state is not treated as a failure or as a pass.
- **ROCm**: this host has neither a ROCm toolchain nor AMD hardware. `check_api_drift.sh`, ROCm compilation, real encoder lifetime execution, `make verify-rocm`, and ROCm inference remain unrun.
- **M5-specific execution**: no M5 result is available; another Apple Silicon result is not substituted.
- **Final combined gate**: the user-requested broad integration run is deferred until all prior expanded follow-ups are ready. This PR's candidate checks are not that final whole-batch gate.

## 5. Change Summary

| Item | Value |
| --- | ---: |
| Files changed | 47 |
| Lines added | 1,646 |
| Lines deleted | 767 |
| Upstream range | 111 commits / 312 files |
| CPU/CUDA/Metal overlays after retirement | 30 |
| Focused numerical executions | 94 passed / 0 failed (89 unique because 5 MXFP tests repeat) |
| Production-head Metal workflow | 12,285 passed / 0 failed / 390 ignored, plus 51 video tests passed |
| Combined-tree Kimi validation | 3 focused tests + 5 exact controls + 58-token natural-language check passed |

## 6. Follow-up Actions

- [ ] Let the seven queued ordinary PR CI jobs complete and review their results.
- [ ] Run ROCm drift/build/runtime gates on an AMD host, or retain the explicit unavailability record.
- [ ] Run the coordinated full CUDA test/Clippy gate and a fresh main Metal nightly only when the expanded batch is ready.

## Appendix: Evidence Boundaries

This report records evidence available while PR #2304 is open and draft. It does not authorize merging, does not claim that PR #2304 closes #1599, #2158, or #1986, and does not treat old-pin Kimi results, host-only ROCm ownership tests, or Metal results from another source revision as current-head coverage beyond the stated source-equivalence boundary.

## Main synchronization after validation

Merged main through `59974283`, including the #2305 mlxcelverse directory migration and subsequent ROCm, MoE, and request-memory updates. Keep both retired CUDA overlays deleted; relocate `mlx/allocator.h` into `src/lib/mlxcelverse/overlays/rocm/`; fix the host ownership script path, duplicate local-fix numbering (48 entries), and overlay documentation paths/counts. Main's HIP grid limits, SDPA optimization, dequant-cache default, and MoE changes remain intact.

Post-merge validation passes: CUDA test compilation, formatting, 27 pin/parser tests, overlay provenance, ROCm residual and 29 tooling checks, host ownership regression, contract gates, and 123 Python tests. Final GPU-serialized execution passes 25 request-memory tests, 38 switch-layer tests, three Inkling MoE tests, 45 DeepSeek V4 tests, and one forced RHS test (112 executions, including the repeated aligned case). Four fresh RHS PTX artifacts prove f16/bf16 × MXFP4/MXFP8 route execution. Logs and exact tree/binary hashes are in `/tmp/mlx-2304-merge-runtime.json`. Prior Metal and actual Kimi evidence remains attributed to its recorded source; those runs were not repeated for this synchronization.

Final CI also exposed four new RustSec advisories against the existing rustls 0.23.45 lockfile entry (RUSTSEC-2026-0346 through RUSTSEC-2026-0349). Update only rustls to 0.23.46; local `cargo deny check` passes without adding advisory exceptions.
