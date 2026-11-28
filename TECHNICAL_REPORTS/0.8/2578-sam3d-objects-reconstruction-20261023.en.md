# Technical Report: PR #2578 - Add SAM 3D Objects reconstruction

**Date**: 2026-10-23
**Last updated**: 2026-10-24
**Status**: Completed — implementation and local acceptance; final documentation/fixture commit CI pending
**Languages**: Rust, C++, Python, Markdown
**Risk Level**: High

## Executive Summary

PR #2578 ports SAM 3D Objects into `mlxcel predict`. The implementation accepts an RGB image plus mask, an RGBA image whose alpha is the mask, or an optional point map; runs the published structure, pose, and sparse-latent flows; and writes object-local OBJ mesh and Gaussian PLY geometry with pose, occupied coordinates, sparse latents, and conditioning metadata. Its defaults remain seed 42, 25 structure steps, 25 latent steps, automatic bundled MoGe-3 depth, and complete Gaussian plus mesh decoding.

Revision `68cd161e` passed a ROCm test build in 9m50s and 61 focused unit tests. Its authenticated full RGBA/dropout capture passed the frozen strict comparison contract with 5,793 exact ordered coordinates, latent relative RMS 0, mesh vertex-count relative error `0.0001589479763`, and normalized symmetric Chamfer `4.450226e-6`. Revision `6bd85cf3` then closed the default-depth focal/shift difference: five focal-solver regressions passed, all 17 default conditioning stages were bit-exact, and the authenticated default capture passed with 6,307 exact coordinates, latent relative RMS 0, exact 189,623-vertex count, and normalized symmetric Chamfer `5.141699913725434e-6`. The literal no-options CLI also passed with the same exact coordinates and latent result, exact vertex count, and normalized Chamfer `5.141710000502124e-6`. Implementation and local runtime acceptance are **Completed**. Final fixture staging and its review corrections are complete; the last inventory signoff passed, while CI on the final documentation/fixture commit remains pending at report time, so no final-head CI result is claimed.

## 1. Problem Statement

SAM 3D Objects reconstructs a masked object as several coupled outputs: pose, sparse voxel occupancy and latent state, textured triangle geometry, and 3D Gaussians. Before this PR, mlxcel had no SAM 3D Objects family, no mask or point-map sidecars for this dense prediction route, and no OBJ or Gaussian PLY output path.

Correctness depends on boundaries that can produce plausible but wrong geometry when implemented approximately:

- The released bundle contains 3,288 BF16 main-model tensors and 607 F32 bundled MoGe tensors.
- Four DINOv2 scopes may share one trunk only after exact key, shape, dtype, and value equality is established.
- Seed-42 random draws, schedule order, occupancy ordering, sparse-neighbor topology, pooling, shifted windows, and FlexiCubes tie order affect coordinates and geometry.
- Automatic depth, supplied point maps, and trained point-token dropout are distinct conditioning routes.
- Resize axis order, nearest-neighbor ratio order, boolean attention masks, fused linear accumulation, tanh GELU evaluation order, and backend LayerNorm synchronization are numerically observable.
- Sparse and mesh topology can create multi-gigabyte live sets. Admission and phase accounting must reject unsafe work before allocation without sampling or truncating accepted results.
- A parity claim is valid only when source, bundle, native runtime, inputs, options, binary, outputs, and comparison programs are authenticated before metrics run.

The frozen numerical contract therefore requires exact coordinate values and ordering, latent relative RMS below `1e-2`, mesh vertex-count error no greater than 1%, and full-set symmetric mean Euclidean nearest-neighbor distance below `1e-3` of the reference mesh AABB diagonal. The comparator does not filter or sample vertices.

## 2. Change Summary

| Area | Change |
|---|---|
| Family integration | Registers `sam3d_objects` for still-image mesh prediction and exposes it through the architecture and dense-predictor registries. |
| CLI sidecars | Adds typed `--mask` and `--pointmap` inputs and rejects unsupported sidecars before model construction. |
| Input processing | Implements RGB/RGBA handling, object crop and full-image views, supplied point-map normalization, automatic MoGe depth, axis conversion, and point-token dropout. |
| Model | Ports structure and pose flow, sparse latent flow, shared DINO conditioning, sparse convolution and attention, decoders, and Gaussian/mesh heads. |
| Checkpoint loading | Validates all 3,895 indexed tensors by exact key, shape, and dtype while preserving the BF16/F32 split. |
| Numerical policy | Adds Objects-scoped resize behavior, DINO fused-linear policy, boolean-exclusion fast SDPA, Power-based tanh GELU evaluation, and an exact focal-solver bridge. Shared defaults for other families remain unchanged. |
| Sparse resources | Adds checked phase-peak formulas, bounded cache eviction/recomputation, supplied-map guards, fallible host allocation, and complete-result failure behavior. |
| Geometry | Ports stable sparse topology and FlexiCubes and adds upstream-compatible OBJ plus binary little-endian Gaussian PLY writers. |
| Output schema | Writes pose, integer `coords`, `latents`, and boolean `pointmap_conditioned` to `result.safetensors`; metadata records seed, actual steps, formats, and conditioning route. |
| Validation | Adds authentic small fixtures, pinned full-reference and capture identities, frozen comparator rules, process receipts, and real-model capture targets. |
| Documentation | Documents defaults, inputs, axes, dtypes, output files, object-local geometry, and unsupported batch/video/server surfaces. |

The accepted RGBA source revision is `68cd161e1a45018559d00fca61e5516f99a7f803`. Its source manifest is `391be3af3fec1a7552f6355dac55cec5b13dcdc1b5259148c1ac159ca3865e4e`, and the retained 118,876,160-byte source archive is `fdf3bb87f072fcd021b65a15405e366a98cdebf98c05fee50997220191f2ebff`. The accepted default-depth validation uses revision `6bd85cf38e70ea43f2ba2e5a63363c35193c2db7`, manifest `dd573f15f508ce241e51658260e0faf7baf88f6edbd678c12ebe60bb971d8981`, and retained 119,029,760-byte archive `f2224b6175c38f54e2b84070cf28710ec8beb25b6c6a1e990a44135a9b2cfdb5`. The later changes are confined to the default-depth focal-solver path, so the earlier RGBA/dropout pass remains applicable by source equivalence. Evidence from either identity is otherwise not carried forward. Relative to `869af0c296e748e8c62075bd53c5cd18db5cc874`, the recorded `68cd161e` working scope contains 130 changed files, 14,660 additions, and 418 deletions.

## 3. Architecture and Technical Decisions

### 3.1 Preserve the complete published pipeline

```text
RGB/RGBA image + mask
  -> object crop and full-image views
  -> supplied point map OR bundled MoGe-3 OR trained dropout route
  -> shared DINOv2 conditioning
  -> 25-step structure and pose flow
  -> ordered occupied coordinates
  -> 25-step sparse latent flow
  -> Gaussian and mesh decoders
  -> pose/coords/latents + OBJ/PLY + metadata
```

The implementation preserves seed 42, `ss_steps=25`, `slat_steps=25`, `estimate_depth=true`, and both output formats as defaults. A supplied HWC point tensor uses +X left, +Y up, +Z forward, with NaN marking invalid points. Disabling depth without a point map uses the trained dropout route. Geometry and parity arrays are never sampled or truncated to fit a resource budget.

### 3.2 Keep dtype and graph policy scoped

The loader validates 3,288 BF16 main tensors and 607 F32 depth tensors before inference. DINO sharing is enabled only after duplicate-scope equality checks. Objects-specific fused-linear and activation policies do not change standalone DINOv2 or other dense families.

Narrow compiled helpers retain upstream graph boundaries. The Objects feed-forward path uses a native tanh GELU helper with upstream Power and operation ordering. Resize uses the upstream adaptive antialias construction and ratio-first nearest mapping, including width-first, height-first, and tie cases. Sparse attention uses boolean exclusion semantics rather than numerically adding a boolean mask.

### 3.3 Bound sparse work by complete live phases

The MoGe refiner derives a conservative row pyramid before execution, checks integer arithmetic, and accounts residual, pool, projection, fusion, and upsample live sets. The default cache-eviction target remains 512 MiB, aggregate neighbor-map storage remains capped at 256 MiB, and the hard supported envelope is 8 GiB. The actual captured configuration admits a conservative bound of 5,598,663,168 bytes.

Sparse convolution borrows a supplied neighbor map, materializes bounded chunks, and drops phase-local inputs before concatenation. Pooling similarly materializes returned arrays while scoping topology and gather scratch. Cached maps may be evicted and recomputed from exact integer coordinates; features are materialized before any required map is dropped. Accepted work produces complete topology or fails explicitly.

### 3.4 Treat native LayerNorm synchronization as a runtime prerequisite

Independent source review found a race in the ROCm forward LayerNorm kernel: warp zero could reuse shared storage for variance while other threads were still reading the shared mean snapshot. Commit `edd9e7fd5ed8dc592a320e723e3a655627990247` adds one read-completion barrier after the mean snapshot; arithmetic and the backward path are unchanged.

Canonical references use the corrected native library `c1d8203b257364271e1ea6298705dd3bb0c82a7f06634f5578a209883601a9fc`. The earlier library `9e38e9292550ea8eec9eb1877d06b72f5155ff382ac0e4ec1054818e15276094` did not include that barrier. Earlier canonical attempts on the old library produced unstable coordinate counts of 5,821 and 5,783. A small LayerNorm probe was stable both before and after the patch, so that probe is not causal proof. The source-level race and the full-pipeline instability are recorded separately; the report does not infer causation from the probe.

Two corrected-runtime canonical RGBA references repeated the same 5,793 coordinates. Twelve of fourteen tensors were bit-exact between repeats, and the remaining mesh tensors differed at approximately `1e-7`. This establishes the repeatability envelope used by the strict full-model comparison.

### 3.5 Match the genuine upstream evaluation boundary

The first custom-stream reference evaluated every yielded event. Genuine upstream `Pipeline.generate` consumes intermediate events and evaluates the completed output, so per-event evaluation changed graph materialization boundaries and could perturb a long BF16 GPU trajectory.

The canonical references now execute the genuine pipeline completion behavior. All earlier failed outputs, logs, exits, and receipts remain retained as diagnostic evidence. A prior 39-stage sampling-trajectory comparison was exact at its observed boundaries, but that instrumentation changed evaluation boundaries and therefore did not prove the uninstrumented full trajectory.

### 3.6 Localize the remaining default-depth difference

The `b000e46c` correction scopes upstream adaptive resize, DINO accumulation, and Python-compatible UV endpoint construction to the Objects MoGe path. Standalone MoGe and default DINO behavior remain unchanged. A 43-stage diagnostic from one actual `estimate_depth` call found 39 stages bit-exact, covering the input, encoder, neck, head, refiner, final affine transform, and mask. The remaining focal and shift scalar differences propagated to the point map with maximum absolute error `1.9073486328125e-6` and raw relative RMS `4.0945834798765365e-8`. This is localization evidence, not full-model acceptance.

The follow-up ports the upstream eager initialization plus 30 compiled focal/shift solver steps behind an Objects-only free-focal dispatch. Fixed-focal behavior and standalone MoGe retain their existing path. Shape-derived byte products are checked before MLX operations. Five focal regressions passed, including all 31 authentic solver states bit-exact on the matching backend, and the 17-stage default conditioning comparison became entirely bit-exact. The subsequent uninstrumented default capture passed the full numerical contract.

## 4. Review Results

### 4.1 Correctness and integration

Independent review covered CLI routing, sidecars, checkpoint schemas, input processing, both flows, shared DINO policy, compiled helpers, sparse topology, FlexiCubes, writers, result schema, defaults, and deterministic ordering. Concrete findings corrected during review included:

- serializing `pointmap_conditioned` as a one-element boolean tensor;
- preserving boolean exclusion semantics in sparse attention;
- matching upstream crop/resize axis and ratio order;
- using the exact Objects tanh GELU evaluation sequence;
- preserving Objects-only DINO fused-linear accumulation;
- accounting complete sparse convolution and pooling live phases;
- caching and exactly recomputing stable sparse maps;
- validating supplied maps and preprocessing allocation before MLX allocations;
- matching upstream OBJ `%.8g` formatting and complete PLY semantics.

The final static implementation review found no open production correctness issue.

### 4.2 Security and performance

Independent security/performance review closed all identified findings. Public boundaries validate option types and ranges, image/mask/point-map shapes and contents, checkpoint schema, sparse cardinalities, neighbor-map dtype and ranges, geometry indices and values, and output names. Checked arithmetic covers preprocessing, sparse phases, and host geometry allocation. Resource rejection occurs before unsafe allocations, and accepted geometry is not filtered, sampled, or truncated.

These controls bound failure behavior; they do not constitute a throughput or memory benchmark.

### 4.3 Strict RGBA parity result

The accepted RGBA/dropout case used RGBA alpha as the mask, `estimate_depth=false`, seed 42, 25+25 steps, TF32 disabled, complete Gaussian and mesh decoding, the corrected native runtime, and the genuine upstream completion boundary.

| Metric | Contract | Result | Status |
|---|---:|---:|---|
| Ordered occupied coordinates | Exact | 5,793 of 5,793 exact | Pass |
| Latent relative RMS | `< 1e-2` | `0` | Pass |
| Mesh vertex-count relative error | `<= 1%` | `0.0001589479763` | Pass |
| Normalized symmetric full-set Chamfer | `< 1e-3` | `4.4502263869e-6` | Pass |

The reference has 188,741 mesh vertices and the Rust result has 188,711. Mean nearest-neighbor distances were `5.3297777143e-6` from actual to reference and `5.7469257839e-6` from reference to actual; the reference AABB diagonal was `1.2445101142`. The frozen comparison JSON contains no failures.

This pass proves the authenticated RGBA/dropout full-model route. It does not establish the default auto-depth route or the serialized no-options CLI route.

### 4.4 Strict default and CLI parity results

The corrected-runtime canonical default reference repeated 6,307 ordered coordinates exactly; 12 of 14 tensors were bit-exact and the two mesh tensors differed by at most `2.086162567138672e-7`. An earlier `b000e46c` Rust capture completed but failed the coordinate gate with 6,300 coordinates, 52 missing and 45 extra. Those immutable failure artifacts remain preserved.

The `6bd85cf3` default v7 capture completed with child, wrapper, and guard exits of 0 and passed the frozen comparator:

| Metric | Contract | Result | Status |
|---|---:|---:|---|
| Ordered occupied coordinates | Exact | 6,307 of 6,307 exact | Pass |
| Latent relative RMS | `< 1e-2` | `0` | Pass |
| Mesh vertex-count relative error | `<= 1%` | `0` (189,623 of 189,623) | Pass |
| Normalized symmetric full-set Chamfer | `< 1e-3` | `5.141699913725434e-6` | Pass |

The earlier actual no-options CLI run at the `68cd161e` boundary exited 0 but failed numerical acceptance with 6,295 coordinates. Its serialized tensors were bit-exact with the same-source integration capture, proving serialization fidelity for that failed prediction. The accepted-source literal-default v4 rerun then passed with child, wrapper, guard, and comparison-tool exits of 0. It used the GPU, binary SHA-256 `c25ac839e070c2ab450a6bc543898e198ff438ea0f6957a0ee933b4cfe696be3`, and literal CLI defaults. All five expected files were present and authenticated; coordinates were exact, latent relative RMS was 0, the 189,623-vertex count was exact, and normalized Chamfer was `5.141710000502124e-6`.

Pose, mesh-export, and Gaussian-export field comparisons are retained as diagnostic-only observations outside the strict acceptance metrics. In particular, row-aligned OBJ face arrays were not exact, so this report makes no exact face-order claim. The complete exported files and their required schemas and semantics passed the CLI gate.

## 5. Validation Evidence

| Validation | Evidence |
|---|---|
| Repository contracts and formatting | Eight scoped repository contracts and formatting checks passed. |
| ROCm test build | Child and wrapper exited 0. Cargo completed the `test-fast` profile in 9m50s and emitted immutable integration, CLI, and unit-test artifacts. |
| Focused unit tests | 61 passed: six MoGe refiner tests, one six-voxel sparse oracle, one grid-budget test, 40 SAM 3D Objects tests, eight default DINO tests, and five Objects DINO policy tests. Receipt executable SHA-256: `368e71d0ffe7dd8ca5b88c5ba3ec3a44dac7bbb21a69b17d75c745619c907765`. |
| `b000e46c` correction gates | 17 focused tests passed: eight Objects resize, seven Objects DINO, one UV endpoint, and one geometry recovery test. Executable SHA-256 `bc417a120c1f2dce149f77d0dcf28534bd1dd0a43cdc456749d87ca241479228`; Clippy, eight repository contracts, and CI passed. |
| Exact focal-solver tests | Five passed, including all 31 authentic solver states bit-exact on the matching backend. Executable SHA-256 `1f275327a08cbbfd1083a4a40479235436e4a1f9756e297afde1a34075bf29f6`. |
| Current build, lint, and CI | The unchanged retry completed in 6.31s after the preceding overall-status-101 attempt hit an unrelated benchmark-linker `SmallVector` abort; both histories are preserved. Required Clippy v4 passed in 1m57s, and committed `6bd85cf3` CI is entirely green. Integration SHA-256 `df1db4715beced04f2415cb5aae268140a0ce921d88534b981c036a5bf45857e`; CLI SHA-256 `c25ac839e070c2ab450a6bc543898e198ff438ea0f6957a0ee933b4cfe696be3`. Final report-head CI remains pending. |
| Accepted RGBA source identity | Head `68cd161e1a45018559d00fca61e5516f99a7f803`; manifest `391be3af3fec1a7552f6355dac55cec5b13dcdc1b5259148c1ac159ca3865e4e`; archive `fdf3bb87f072fcd021b65a15405e366a98cdebf98c05fee50997220191f2ebff`. |
| Accepted default source identity | Head `6bd85cf38e70ea43f2ba2e5a63363c35193c2db7`; manifest `dd573f15f508ce241e51658260e0faf7baf88f6edbd678c12ebe60bb971d8981`; archive `f2224b6175c38f54e2b84070cf28710ec8beb25b6c6a1e990a44135a9b2cfdb5`. |
| Canonical upstream RGBA reference | Child, wrapper, and guard exited 0. Receipt SHA-256 `1f08fdffb6d1425ae779bfffdba3c546de39aa74e35a004d6ed3d1cb2bca0dc6`; reference safetensors SHA-256 `8d8cc32b5a5d0f223d6bce474e7f90ff4eff11cf6c28c58eeaacd230393eecaf`. |
| Authenticated Rust RGBA capture | Receipt SHA-256 `f9241815539d5779b86a6edfcd44a573cee820b69648bf248b539d507c820013`; capture safetensors SHA-256 `3f3922d9967d472ce0a81f5a889f1e5012b3ae7bf7c7f1ca0aad83f5a0d95cf0`. |
| Frozen comparator | Comparator `d919c9b740afe05359f14f592b9416cafd1d833ed8ea67a698d94eb17d809580`; contract `afe87b621e49c59742c11d71d8f314592b6acabb0ba5664de49a1285759d3d07`; comparison JSON SHA-256 `a0e699db55189851c8de5874ef39e17d2bfe5e420786eb12a64dc77102332809`. |
| Full RGBA comparison | Strict pass with exact coordinates, zero latent RMS, 0.0158948% vertex-count error, and normalized full-set Chamfer `4.450226e-6`. |
| MoGe localization history | The earlier 43-stage diagnostic had 39 exact stages and point-map raw relative RMS `4.0945834798765365e-8`. After the solver correction, all 17 default conditioning stages were bit-exact; actual output SHA-256 `9b3639a65168c7db5c3df5bf1c4f970bc806dd8ca0f48328ee64d3a0cefb97ec`. |
| Full default auto-depth comparison | **Pass.** Exact 6,307 coordinates, latent RMS 0, exact vertex count, normalized Chamfer `5.141699913725434e-6`; output SHA-256 `f1820a3ff55bbbb29f0ff0404cf51c06b8f5be076f82e78afcd15ee2b5378cbe`; receipt SHA-256 `108b97bd77042e03d75e408e7108f6182fec0720f75b837c13b9abe81614991a`. |
| Actual no-options CLI | **Pass.** Literal defaults, all five expected files, exact 6,307 coordinates, latent RMS 0, exact 189,623-vertex count, and normalized Chamfer `5.141710000502124e-6`; capture receipt SHA-256 `9e35367675653e55d34ddf6975f1c9f5f0a246d326b77df83178f6e59c56f299`; comparison SHA-256 `864c58a61faaf27303f0c07da53698133d991fb72344696cb478877d0789a5de`. |
| Third-party export interoperability | Trimesh 5.1.1 parsed 189,623 OBJ vertices and 376,618 faces; plyfile 1.1.5 parsed 201,824 Gaussians with all 17 required properties. Interoperability JSON SHA-256 `48f34c90b6b168a34ab22a33bc8f2938a637951a055457ee02542afbe02ed527`. This is a parser interoperability result, not a rendering or face-order claim. |
| Final fixture helpers | 33 bounded mock tests passed. Final CPU generators reproduced the exact 31-state focal fixture (`cf5bf5c2...`) and resize payloads. Copied child/wrapper/guard statuses are hash-authenticated, and the portable generator hard-pins its executed helper. |

## 6. Change Statistics

| Item | Value |
|---|---:|
| Files changed | 181 |
| Lines added | 18,221 |
| Lines deleted | 455 |
| Accepted RGBA head | `68cd161e1a45018559d00fca61e5516f99a7f803` |
| Accepted default head | `6bd85cf38e70ea43f2ba2e5a63363c35193c2db7` |
| Base | `869af0c296e748e8c62075bd53c5cd18db5cc874` |
| Focused tests passed | 61 |
| Accepted full-model routes | 2 of 2 required routes |

## 7. Follow-up Actions

### Remaining publication steps

- [x] Complete review and staging of the final fixture bundle and correct its review findings.
- [x] Complete the last fixture inventory signoff.
- [ ] Commit the final documentation and fixtures, then run CI on that resulting head.
- [x] Force-stage these two ignored report files by exact path when the PR is finalized.

### Evidence limitations

- The LayerNorm source race is independently established, but the stable tiny probe does not prove that race caused the earlier full-pipeline instability.
- The old 5,821/5,783 runs and all custom-stream failures remain diagnostic evidence only.
- Exact agreement at 39 instrumented stages did not prove the genuine uninstrumented trajectory because observation changed evaluation boundaries.
- The accepted RGBA comparison applies to the authenticated source, binary, checkpoint, input, options, native runtime, and comparator identities listed above.
- The 43-stage MoGe diagnostic localized the historical default difference but did not replace the now-completed uninstrumented default comparison.
- The RGBA result remains tied to `68cd161e`; it applies to the later source only through reviewed source equivalence of the depth-only changes.
- Default auto-depth and the literal no-options CLI are accepted at `6bd85cf3`. Pose and export comparisons are diagnostic only; no exact OBJ face-order claim is made.
- Fixture staging and its last inventory signoff are complete; CI on the final documentation/fixture commit was pending when this report was finalized. The green `6bd85cf3` CI is not presented as final-head CI.

## References

- PR #2578: SAM 3D Objects reconstruction
- Issue #2381: Port SAM 3D Objects with mesh and Gaussian output
- Epic #2382: dense-prediction model delivery
- `docs/dense-prediction.md`: CLI and output contract
- `tests/fixtures/sam3d_objects/README.md`: reference and capture procedure
- `tests/fixtures/sam3d_objects/provenance.json`: source, runtime, checkpoint, and input identities
- `tests/fixtures/sam3d_objects/comparison-contract.json`: frozen full-array acceptance metrics
- `.artifacts/sam3d_objects/reference-canonical-ln-*`: corrected-runtime canonical references
- `/tmp/objects-full-rgba-v5-comparison.json`: accepted RGBA strict comparison
- `/tmp/objects-moge-stages-v2-comparison.json`: 43-stage default-depth localization
- `/tmp/objects-default-v6-coordinate-failure.json`: completed default run coordinate failure
- `/tmp/objects-default-cli-integration-comparison.json`: actual CLI versus same-source integration output
- `/tmp/objects-default-stages-v3-comparison.json`: corrected 17-stage exact comparison
- `/tmp/objects-full-default-v7-comparison.json`: accepted default strict comparison
- `.artifacts/sam3d_objects/cli-default-v4/comparison.json`: accepted literal-default CLI comparison
- `/tmp/objects-cli-v4-interop.json`: independent OBJ/PLY parser interoperability
