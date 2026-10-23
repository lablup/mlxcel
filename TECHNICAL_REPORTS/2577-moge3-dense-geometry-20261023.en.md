# PR #2577: Add MoGe3 dense geometry prediction

**Date**: 2026-10-23
**Status**: Completed
**Languages**: Rust, C++, Python, Markdown
**Risk Level**: High

## Executive Summary

PR #2577 ports the published MoGe3 ViT-L and ViT-g monocular geometry models to mlxcel. A still image now flows through the shared DINOv2 encoder, convolutional decoder, camera recovery, and sparse 3D refiner to produce source-resolution points, depth, validity, surface normals, and normalized camera intrinsics. The family is registered for `mlxcel predict`, exposes a public `Moge3` library API for later SAM 3D Objects integration, supports all upstream inference options, and serializes optional refinement-step outputs.

The implementation adds an exact integer `searchsorted` bridge, bounded sparse operations, strict config/checkpoint/input validation, and an authenticated reference harness pinned to unchanged upstream source and exact-native MLX. Focused unit tests and the integer-boundary bridge test pass. Sequential gfx1151 parity passes for both published checkpoints and all five required option cases, with maximum errors far below the specified thresholds and exact mask/non-finite support. The final extracted source passed library compilation, guarded unit execution, named integration linkage, CLI compilation, warnings-denied clippy, and current-head PR CI. The successful full-model GPU and default CLI evidence remains tied to the validated pre-extraction binary. Independent review established that the intervening module extraction, iterator rewrite, and formatting changes preserve the operation graph; the final binary therefore has compile/link/unit evidence but no separate numerical result. The authentic full-resolution Python CPU reference is complete and independently recovered, while its surrounding wrapper exited 2 and the Python child exit was not captured. The immutable Rust CPU capture exited 0 on the resolved CPU backend, passed its pinned identity and schema validator, and matched all 17 reference outputs at the unchanged thresholds. The numerical CPU result remains tied to the validated pre-extraction binary and archived source; final-source compile, link, unit, clippy, and current-head CI evidence covers the graph-equivalent extraction and cleanup changes.

## 1. Problem Statement

MoGe3 predicts five mutually dependent geometry products from one RGB image: a camera-space point map, depth, surface normals, validity, and camera intrinsics. Before this PR, mlxcel's dense-prediction surface could serialize these classes of output, and its DINOv2 implementation could supply the backbone, but `model_type: moge3` was not recognized and there was no sparse 3D refinement path.

The port has several correctness-sensitive components:

- MoGe3 chooses a token grid using Python ties-to-even rounding and accepts either `resolution_level` or an explicit `num_tokens`.
- Sparse pooling and submanifold convolution depend on exact integer keys, duplicate-aware `searchsorted`, coordinate order, ceil-div shapes, and parent-index semantics.
- Camera recovery solves focal length and depth shift iteratively; fixed-FOV and recovered-FOV paths must remain distinct.
- Published checkpoints contain F32 DINOv2-L/14 or DINOv2-g/14 weights and may need shape-based transposed-convolution relayout.
- Masked points and depth use positive infinity, while masked normals use zero. A comparator that checks only finite values could falsely pass a support mismatch.

Failure in any of these areas can yield plausible images with incorrect geometry, so acceptance uses authenticated full-resolution references and checks schema, shapes, dtypes, mask, and every finite/NaN/infinity class before numeric metrics.

## 2. Change Summary

| Area | Change |
|---|---|
| Runtime integration | Registers `moge3` in dense routing and the architecture catalog with the `geometry` task and still-image input. |
| Public API | Exposes `Moge3`, `Moge3Output`, `InferOptions`, default `estimate`, and validated configurable inference for downstream consumers. |
| Backbone and decoder | Reuses the shared DINOv2 encoder, combines four intermediate grids, and implements the neck, point, mask, normal, and scale heads. |
| Sparse refinement | Adds coordinate-only submanifold 3D convolution, neighbor lookup, 2x mean pooling, nearest upsampling, and the sparse 3D U-Net without allocating a dense voxel volume. |
| Geometry | Implements UV construction, focal/shift recovery, normalized intrinsics, point/depth conversion, projection controls, masking, and per-step recovery. |
| Core bridge | Adds exact int64 `searchsorted` with side selection so sparse keys beyond F32 precision remain exact. |
| Weight loading | Strictly loads the checkpoint, validates architecture and tensor shapes, retains F32 execution, and normalizes supported ConvTranspose layouts idempotently. |
| Resource safety | Bounds image pixels, token ranges, decoder pixels, sparse construction, channels, refinement steps, and retained per-step outputs before large graphs are built. |
| Validation | Adds compact upstream mathematical oracles, pinned real-checkpoint references, five-case Rust parity, and a fail-closed CLI validator. |
| Documentation | Documents checkpoints, options, output tensors, preprocessing, memory behavior, TF32 control, and unsupported input surfaces. |

The post-extraction working tree currently changes 39 files with 6,812 additions and 9 deletions relative to `6ca04d17`.

## 3. Architecture and Technical Decisions

### 3.1 Reuse the shared dense and DINOv2 foundations

```text
RGB image
  -> validated [0,1] NHWC preprocessing
  -> DINOv2-L/14 or DINOv2-g/14 intermediate layers
  -> UV augmentation + projected feature pyramid
  -> point / mask / normal / scale heads
  -> focal and depth-shift recovery
  -> optional sparse 3D refinement loop
  -> projection + masking + source-size restoration
  -> DenseOutput and Moge3Output
```

The family reuses the dense registry, writer, interpolation, and DINOv2 encoder introduced by its dependencies. Model-specific convolution, geometry, and sparse code stays under `src/vision/dense/moge3/`. This keeps shared behavior centralized while preserving upstream MoGe3 math.

### 3.2 Preserve sparse coordinate semantics instead of densifying

Sparse voxels use integer `(batch, row, column, depth-bin)` coordinates and raster keys. Neighbor lookup uses exact int64 `searchsorted`; missing neighbors gather an appended zero row. Pooling sorts parent keys, forms segment IDs, computes the first row per segment, averages at most eight cell members, and returns parent indices used by nearest upsampling.

The C++ scatter API requires update rank to equal target rank plus index rank. The first guarded unit run exposed rank-one updates that Python's `.at[...]` adapter expands implicitly. Rust now supplies `[M,1]` updates for additive parent assignment and minimum first-position reduction. This is the same shape inserted by MLX's Python binding and leaves indices, values, order, dtypes, and rank-one outputs unchanged.

### 3.3 Fail closed at public and allocation boundaries

`InferOptions::validate` is applied at both JSON parsing and the public library boundary. It checks the checkpoint token range, `refine_steps <= 16`, refiner availability, and finite `fov_x` strictly between 0 and 180 degrees. Config validation constrains the published backbones, four intermediate layers, head schemas, channel widths, decoder growth, sparse working memory, and refiner topology. Runtime checks cover image dimensions, tensor ranks, coordinate bounds/order/uniqueness, and output-retention budgets.

Integer images scale by their source dtype range. Direct floating input must be finite RGB in `[0,1]`. Invalid requests return contextual errors before expensive allocation or inference.

### 3.4 Keep upstream non-finite and option behavior visible

The default output contains exactly `points`, `depth`, `mask`, `normal`, and `intrinsics`. `return_per_step` adds `points_stepN`, `depth_stepN`, and `intrinsics_stepN`. `force_projection`, `apply_mask`, fixed `fov_x`, zero refinement, and explicit `num_tokens` retain their upstream meanings.

When masking is enabled, invalid points and depth are positive infinity and invalid normals are zero. The committed comparator first requires exact key sets, shapes, dtypes, masks, and finite/NaN/positive-infinity/negative-infinity classes, then computes relative RMS only on shared finite support. Empty finite support is rejected.

### 3.5 Authenticate references before accepting metrics

The fixture harness pins unchanged mlx-vlm revision `abd3b2d7b728cb5372d4e092260da325e5ded209`, MLX 0.32.5, native library SHA-256 `9e38e9292550ea8eec9eb1877d06b72f5155ff382ac0e4ec1054818e15276094`, and binding SHA-256 `f8ec0f363bc6eab1cda644a2579ae43aa663939f30ace531d5fb8cc33c7090cc`. It verifies source modules, checkpoint config/preprocessor/weights, the lossless input image, runtime mappings, options, output schema, and artifact hashes before comparison.

The real Rust tests independently authenticate the ViT-L and ViT-g checkpoint revisions and file hashes before loading them. Reference generation and comparison fail under missing or mismatched receipts, including under `python -O`.

The full-resolution ViT-L CPU generator saved all 17 outputs and its complete receipt, then printed terminal JSON that exactly matches the receipt subset. The surrounding shell wrapper later exited 2 with a quoting error, and no independent Python child exit was captured. Recovery verifier SHA-256 `ededb85947995300045bcaef281cf1254af373a951a54d0d2eba53ef8ee3c282` completed without inference and verifies the original status/resource hashes, non-memory-stop state, terminal output, canonical identities/options, 607 F32 parameters, tensor schema, and finite/NaN/infinity classes. The recovery record preserves `wrapper_exit: 2`, `python_exit: not independently captured`, and the missing launch-time wrapper hash.

The corresponding immutable Rust CPU capture exited 0 and attested the CPU runtime. Validator version 1, SHA-256 `bae73574ef629abbcf6cca7b0a611c7390d2f3f159d5e2e19fe993dbae209cea`, authenticated the archived revision and source, binary, checkpoint, image, exact options and command, output schema, previews, metadata, and value classes before comparison. The committed `cpu-parity.json` record binds those identities to the reference and Rust output hashes, resource receipt, comparator hash, all 17 metrics, and a no-failures result.

## 4. Review Results

### 4.1 Correctness and integration

Independent implementation review found six issues during development: token-grid rounding, public option validation, helper K semantics, preprocessing context, non-finite comparison, and missing option-case coverage. All six were corrected and rereviewed. The final static pass found the model, registry, output, and five authenticated real-GPU option cases integrated with no open correctness finding.

The CLI validator was also hardened after review. It requires exactly one recognized child-process `Runtime device:` line, rejects a requested/reference backend mismatch before creating a Rust receipt or comparing tensors, and records the resolved device. This prevents an environment-requested GPU run that fell back to CPU from being attested as GPU. The default no-`--opts` CLI path passed on the source immediately before the sparse-module extraction. The final extracted source passed compile, link, unit, and clippy gates; its newly built CLI binary was not used for another full-model inference run.

### 4.2 Security and performance

The security/performance review found no remaining issue in the corrected source. Allocation-driving values are bounded before graph construction; sparse neighbor construction is chunked and accounts for retained and peak working memory; non-finite outputs are classified rather than trusted by arithmetic; config, checkpoint, image, runtime, and reference identities fail closed.

The sparse implementation avoids a dense 3D voxel allocation. Validation timings below are diagnostic run durations under a shared GPU guard and are not performance benchmarks.

### 4.3 Documentation coverage

No documentation gap remains in the implemented scope:

- `docs/dense-prediction.md` covers family/task routing, both checkpoints, CLI syntax, every option, five tensor contracts, preprocessing, memory rejection, TF32 control, and explicit batch/video/server exclusions.
- `docs/supported-models.md` records architecture, output, option, and runtime scope.
- `tests/fixtures/moge3/README.md` records provenance, regeneration, strict comparison, all GPU cases, the recovered CPU reference limitation, completed CPU parity, and the default no-`--opts` CLI procedure.

The docs deliberately follow the pinned upstream implementation's positive-infinity masking, correcting the stale issue wording that mentioned NaN points.

## 5. Validation Evidence

| Check | Result at report time |
|---|---|
| Focused library compile | Initial attempt exited 101 with one iterator-comparison type error and three `UniquePtr` borrowing errors. After correction and the module extraction, the final-source library no-run build passed with exit 0 in 10m57s. |
| Focused MoGe3 unit tests | First guarded run passed 6 tests, then aborted with exit 134 at the rank-one scatter update. After the `[M,1]` adapter correction, the final-source guarded run passed all 8 tests, 0 failed, in 0.79 s. |
| Exact int64 `searchsorted` | An over-specific `--exact` selector selected 0 tests and was not counted. The corrected filter ran the intended boundary/side-semantics test: 1 passed, 0 failed. |
| Authentic harness checks | Strict small-oracle regeneration was byte-identical; hardened identity/schema negative coverage passed 15 negative cases plus one positive comparator contract. Existing GPU references reverified without rewriting their original tensors or receipts. The recovered CPU reference contains all 17 expected tensors, matches its receipt and canonical identities, and records zero NaN/+infinity/-infinity values for every F32 output on this fully valid-mask image. |
| Real GPU parity | **Pass on gfx1151 with `MLX_ENABLE_TF32=0`.** The validated pre-extraction binary ran two serial test functions covering five ViT-L/ViT-g cases; 2 passed, 0 failed in 89.99 s. The duration is diagnostic. |
| Output integrity | Exact output key sets, shapes, F32/bool dtypes, mask values, and finite/NaN/infinity classes passed for every case before numeric metrics. |
| Actual CLI on pre-extraction source | **Pass.** The build exited 0; the guarded default run used literal no-`--opts`, attested the resolved GPU, emitted five tensors, and passed exact mask/non-finite support, previews, metadata/intrinsics, and `mlxcel arch` checks. Depth/points relative RMS were `2.224974e-5`/`2.033098e-5`, normal relative RMS was `4.20353e-7`, and intrinsics maximum absolute error was `2.080202e-5`. |
| Final-source integration linkage | **Pass.** The named real-model integration target linked with exit 0 in 6m27s. This verifies linkage; it does not claim another checkpoint execution. |
| Final-source CLI build | **Pass.** The final CLI compiled with exit 0 in 6m18s and produced SHA-256 `4fd8c932c768e3267dfab498b31686986b4b6fb95f52d6bb2039c97d1277d3ec`. This binary was not run for full-model parity. |
| Required clippy | **Pass.** `--lib --tests -- -D warnings` exited 0 in 3m05s on the final source. |
| Final-source numerical evidence scope | Independent review established that the sparse-module extraction, iterator rewrite, and formatting changes preserve the operation graph. The final source passed compile, link, guarded unit, and clippy gates; numerical acceptance remains pinned to the validated pre-extraction binary, and no separate final-binary checkpoint result is claimed. |
| Python CPU reference | **Recovered complete reference.** All 17 full-resolution tensors and the receipt are authenticated at output SHA-256 `afbc0bffa5112996423dd11230cf066498a0d98372f67adb201e37dfbf393cb0`. The wrapper exited 2 after terminal generator output; the Python exit is unknown and no successful process exit is claimed. |
| CPU parity | **Pass on CPU with `MLX_ENABLE_TF32=0`.** The immutable Rust capture exited 0, the pinned validator authenticated all identities and the exact 17-key schema, and strict comparison reported no failures. Maximum depth/points relative RMS were `1.5590750464e-5`/`1.4222186927e-5`, normal relative RMS was `2.8714204e-7`, intrinsics maximum absolute error was `1.6570091248e-5`, and mask/non-finite support was exact. |
| Current-head PR CI | **Pass at `c90268bb`.** All required PR checks are green; scope-inapplicable jobs are skipped. Passing checks include formatting, clippy, CPU-only link, OpenXLA feature compile, cargo-deny, Python tooling, binary assets, and project contract gates. |

The CLI build completed with exit 0 in 9m19s and produced binary SHA-256 `8740d2a60082e5d58564343e081d3660e1026e555dc6da78cfb362fcbeda57d2`; the duration is diagnostic. The first validator launch failed before invoking the CLI or creating its output because the isolated Python path lacked Pillow. The retry retained the exact MLX path first, added the read-only SAM reference site-packages for Pillow, and completed with exit 0.

The final extracted source produced a separate CLI binary with SHA-256 `4fd8c932c768e3267dfab498b31686986b4b6fb95f52d6bb2039c97d1277d3ec` in a 6m18s build. That result is build evidence only. All full-model numerical and CLI-output claims in this report remain attached to the earlier `8740d2a6...` binary and archived `de4b55de...` source.

The genuine Rust CPU capture used an immutable copy of the validated pre-extraction binary and archived commit `de4b55de47e7a65e9fff60bdcd00c4f74c90de3f`. It used the exact ViT-L checkpoint, image, resolution-9, three-refinement, per-step configuration of the recovered Python CPU reference. The process exited 0 on the resolved CPU backend. Its resource receipt records 19,676.249865 seconds, maximum RSS of 370,176 KiB, minimum available memory of 19,148,304,384 bytes, and `stopped_for_memory: false`; these measurements establish execution provenance and are not benchmark results. The Rust output SHA-256 is `b7ce67a1fac333a5297dea4c2b579072fbd789f0391d34e053cd8960d21c2998`. This capture belongs only to the archived pre-extraction source and is not numerical evidence for the final extracted source.

The Rust capture validator identifies itself as version 1 and has SHA-256 `bae73574ef629abbcf6cca7b0a611c7390d2f3f159d5e2e19fe993dbae209cea`. Before loading outputs it hard-pins the archived revision, binary and source archive hashes, canonical ViT-L checkpoint, fixed image, exact 17-key option case, CPU runtime, TF32 setting, and command. It passed with exit 0 and recorded finite, NaN, positive-infinity, and negative-infinity counts separately. The strict comparison then passed with no failures across all 17 keys. The committed `tests/fixtures/moge3/cpu-parity.json` binds the authenticated Rust receipt, resources, validation and comparator hashes, complete metrics, Python reference SHA-256 `afbc0bff...`, and Rust output SHA-256 `b7ce67a1...`.

The sequential GPU run's maximum observed errors were:

| Output | Required GPU threshold | Maximum observed error |
|---|---:|---:|
| Depth | relative RMS <= `5e-3` | `2.2250e-5` |
| Points | relative RMS <= `5e-3` | `2.0331e-5` |
| Normal | relative RMS <= `5e-3` | `4.20e-7` |
| Intrinsics | maximum absolute <= `1e-3` | `2.0802e-5` |
| Mask and non-finite support | exact | exact |

The CPU acceptance run's maximum observed errors were:

| Output | Required CPU threshold | Maximum observed error |
|---|---:|---:|
| Depth | relative RMS <= `1e-4` | `1.5590750464e-5` |
| Points | relative RMS <= `1e-4` | `1.4222186927e-5` |
| Normal | relative RMS <= `1e-4` | `2.8714204e-7` |
| Intrinsics | maximum absolute <= `1e-3` | `1.6570091248e-5` |
| Mask and non-finite support | exact | exact |

A prior two-test parallel GPU run also exited 0 and is retained as diagnostic evidence. The serial run above is the acceptance record; its duration must not be interpreted as a benchmark.

## 6. Authenticated GPU Cases

| Case | Checkpoint | Coverage | Keys | Reference SHA-256 |
|---|---|---|---:|---|
| `vitl-gpu-resolution9` | ViT-L | recovered FOV, 3 refinements, per-step outputs, projection, mask | 17 | `8a52dfd2f1d4f6fa7e9e955fd4ab3366bf4cb66c9de1b19816306c6917280a61` |
| `vitg-gpu-resolution9` | ViT-g | giant backbone with the same default full-resolution path | 17 | `b3a52681524456798639eef2971422edbd6113ea7b6188bb7b019bcd25b09cd7` |
| `vitl-no-refine` | ViT-L | zero refinement, recovered FOV, no projection, masking | 5 | `c6151def796e489085ffbf2c791c7bbc8f1532ffe1735dd586a7795c5951756f` |
| `vitl-fixed-unmasked` | ViT-L | fixed 60-degree FOV, 3 refinements, per-step outputs, no projection or mask | 17 | `517044d2f70104e62a59f975d48bec52d37f9b818357de88ea74e09c97b640bf` |
| `vitl-fixed-projected` | ViT-L | explicit 3,600 tokens overriding level 0, fixed FOV, projection, no mask | 5 | `9e113455ff51aa588854088fa9282348b5246f840de2f510e2916f5060405d44` |

The separate `vitl-default` five-key reference, SHA-256 `97cbe3c12ca8553f7f177bc189193f8b29ace8d536a25ab8417d7851d74c31e4`, was used by the successful prior-source no-`--opts` CLI validation. Independent review established that the subsequent extraction, iterator, and formatting changes preserve the operation graph. The final source passed compile, link, guarded unit, and clippy gates; the final binary itself has no separate full-model numerical result.

The recovered `vitl-cpu-resolution9` reference uses the same 17-key ViT-L resolution-9/refine-3/per-step case as the completed Rust CPU capture. Its output SHA-256 is `afbc0bffa5112996423dd11230cf066498a0d98372f67adb201e37dfbf393cb0`; `tests/fixtures/moge3/cpu-reference-recovery.json` records the wrapper failure, unknown Python exit, verifier/status/resource hashes, exact options, and per-output value-class counts. `tests/fixtures/moge3/cpu-parity.json` records the completed strict comparison and its immutable pre-extraction scope.

## 7. Compatibility and Limits

- **Validated backends**: ROCm on gfx1151 and CPU, both with TF32 disabled. CPU numerical evidence is pinned to the immutable pre-extraction binary and archived source.
- **Unvalidated backends**: Metal and CUDA execution were not run for this report.
- **Checkpoint coverage**: Both published ViT-L and ViT-g checkpoints passed authenticated GPU inference.
- **Input scope**: One still image. Batch CLI input, video, streaming, and server routes are outside #2355.
- **Conversion scope**: Conversion from the official PyTorch `model.pt` remains outside the issue; published MLX safetensors are supported.
- **Dependencies**: Uses merged dense-output and DINOv2 foundations. No new third-party dependency was added.

## 8. Follow-up Actions

### Completion state

- [x] Authenticated five GPU option cases and the default CLI path.
- [x] Completed the genuine Rust CPU capture, pinned validation, and strict 17-key comparison at the unchanged thresholds.
- [x] Passed final-source compile, link, guarded unit, warnings-denied clippy, CLI build, and current-head required CI checks.

A later documentation-only commit should receive the repository's ordinary CI checks. It does not change the model graph or require another checkpoint run.

### Ongoing safeguards

- Keep checkpoints and full tensor artifacts under ignored `.artifacts/moge3`; do not stage them.
- Preserve exact source/runtime/checkpoint/image hashes and strict receipt checks when regenerating references.
- Keep parity thresholds unchanged and compare schemas and value classes before metrics.

## 9. Change Statistics and Commits

| Item | Value |
|---|---:|
| Files changed | 39 |
| Lines added | 6,812 |
| Lines deleted | 9 |
| New MoGe3 Rust modules | 14 |
| Authenticated GPU cases passing | 5 |
| Authenticated CPU cases passing | 1 |

| Commit | Message |
|---|---|
| `5d787774` | `feat: add MoGe3 dense geometry prediction` |
| `de4b55de` | `docs: record MoGe3 GPU parity evidence` |
| `c90268bb` | `refactor: split MoGe3 sparse pooling module` |

## References

- PR #2577: `feat: add MoGe3 dense geometry prediction`
- Issue #2355: MoGe3 monocular geometry
- Epic #2382: dense-prediction model delivery
- Upstream mlx-vlm revision `abd3b2d7b728cb5372d4e092260da325e5ded209`
- `docs/dense-prediction.md`: public CLI and tensor contract
- `tests/fixtures/moge3/README.md`: reference provenance, generation, comparison, and CLI validation procedure
- `/tmp/epic-2382-moge3-real-gpu-sequential.log`: sequential GPU acceptance log
- `/tmp/epic-2382-moge3-cli-validate-retry.log`: successful pre-extraction CLI validation log
- `/tmp/epic-2382-moge3-final-lib-no-run.log`: final-source library build log
- `/tmp/epic-2382-moge3-final-units.log`: final-source guarded unit-test log
- `/tmp/epic-2382-moge3-final-real-no-run.log`: final-source named integration linkage log
- `/tmp/epic-2382-moge3-final-cli-build.log`: final-source CLI build log
- `/tmp/epic-2382-moge3-final-clippy-retry2.log`: final-source warnings-denied clippy log
- `tests/fixtures/moge3/cpu-reference-recovery.json`: authenticated Python CPU-reference recovery receipt
- `/tmp/epic-2382-moge-recover-cpu-reference.py`: recovery verifier, SHA-256 `ededb85947995300045bcaef281cf1254af373a951a54d0d2eba53ef8ee3c282`
- `/tmp/epic-2382-moge-validate-rust-cpu.py`: pinned Rust CPU capture validator, version 1, SHA-256 `bae73574ef629abbcf6cca7b0a611c7390d2f3f159d5e2e19fe993dbae209cea`
- `.artifacts/moge3/rust-cpu-capture/validation.json`: authenticated Rust CPU output receipt
- `.artifacts/moge3/rust-cpu-capture/cpu-comparison.json`: strict 17-key CPU comparison result
- `tests/fixtures/moge3/cpu-parity.json`: committed-path CPU parity receipt and complete metric record
