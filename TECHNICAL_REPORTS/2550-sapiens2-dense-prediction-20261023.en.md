# PR #2550: Port Sapiens2 pose and dense prediction

**Date**: 2026-10-23
**Status**: Complete locally (fresh PR CI pending)
**Languages**: Rust, Python, Markdown
**Risk Level**: High

## Executive Summary

PR #2550 ports the Sapiens2 human-centric vision family to mlxcel's dense-prediction runtime. It loads the published flat and legacy nested checkpoint schemas, executes the config-driven ViT backbone and five task heads, and exposes pose, semantic segmentation, surface normals, pointmaps, matting, and backbone features through `mlxcel predict`. Pose follows the upstream affine crop, optional flip test, DARK refinement, and UDP source-coordinate transform; dense tasks return source-resolution outputs under their published resize and padding rules.

The implementation also adds bounded config and pose inputs, checkpoint layout normalization, authentic upstream fixtures, and a family-specific BF16 retention policy. Static implementation and security/performance reviews found no open major issues after their fixes. Focused ROCm library and real-model integration targets compiled, and the Sapiens2/grid-sample unit filters passed. The first real-checkpoint run exposed a blocking pose-parity failure on gfx1151. Stage localization then found a JPEG decoder mismatch and BF16 operation-graph differences in biased linear, convolution, normalization, kernel-4 deconvolution, activation, and packed RoPE paths. Those paths now mirror pinned upstream MLX behavior and the fixture uses a canonical lossless PNG. Every captured backbone and pose-head stage is bit-exact after the compiled packed-RoPE, deconvolution-layout, and compiled-SiLU fixes. All six real-checkpoint acceptance tests now pass on gfx1151: pose with and without flip, segmentation, normals, pointmaps, and matting. The rebased CLI build and an end-to-end GPU pose invocation also pass and emit the documented artifacts. Required clippy targets pass with warnings denied; fresh PR CI remains pending at this report snapshot.

## 1. Problem Statement

The dense-output foundation recognized `sapiens2`, but mlxcel could not load or execute the checkpoints. Sapiens2 also cannot be represented by a generic vision backbone plus a single decoder: its published family uses a 1024x768 ViT with register tokens, checkpoint-backed 2D RoPE periods, a per-layer MHSA/GQA schedule, QK RMSNorm, LayerScale, SwiGLU, and task-specific deconvolution or pixel-shuffle heads.

The public checkpoint set presents two integration complications. Early segmentation exports use a nested backbone/head schema while later checkpoints use a flat Transformers-style schema, and convolution and attention weights may use multiple published layouts. Pose adds a top-down crop and numerical decode pipeline whose coordinate, blur, flip-pair, and half-pixel details must match upstream to remain inside the 1.5-pixel acceptance bound.

Without this port, users cannot run the five published human dense-prediction tasks through the shared CLI or consume their standard tensor and file outputs.

## 2. Change Summary

| Area | Change |
|---|---|
| Runtime integration | Registered `sapiens2` in the dense family table and architecture catalog and routed recognized checkpoints to `mlxcel predict`. |
| Configuration | Canonicalized flat and legacy nested configs, selected the task from checkpoint metadata, supported the standard 0.4B, 0.8B, 1B, and 5B dimensions, and bounded all allocation-driving fields. |
| Backbone | Added patch embedding, register tokens, checkpoint-backed 2D RoPE, per-layer GQA/MHSA, QK RMSNorm, LayerScale, SwiGLU blocks, final normalization, and backbone feature outputs. |
| Heads | Added deconvolution and pixel-shuffle decoders, segmentation, normal, pointmap, matting, and pose projections, plus the pointmap scale branch. |
| Pose | Added bounded pixel-space boxes, upstream aspect-ratio crops, affine grid sampling, flip-pair remapping, device-side Gaussian blur, DARK/UDP decoding, and source-image coordinate conversion. |
| Dense outputs | Restored segmentation, normal, pointmap, and matting results to source resolution and emitted the task-specific tensor keys required by issue #2371. |
| Weight loading | Normalized Q/K/V and convolution layouts, validated tensor dtypes and shapes, loaded checkpoint RoPE periods, and retained published BF16 Sapiens2 weights on every backend. |
| Validation assets | Added a fixed image, authentic upstream pose and segmentation references, checkpoint/source hashes, reproducible reference scripts, comparison tooling, focused unit tests, and ignored real-checkpoint tests. |
| Documentation | Documented the CLI options, preprocessing rules, tensor shapes, task scope, config compatibility, BF16 exception, and supported architecture. |

The numerical parity fixes, their focused regressions, and the validated CLI path are integrated on the rebased implementation. All required local gates pass; fresh PR CI remains outstanding at this report revision.

## 3. Technical Decisions

### 3.1 Preserve the checkpoint's BF16 dtype

Sapiens2 is an explicit exception to mlxcel's usual Apple Silicon BF16-to-F16 load optimization. An authentic upstream run of the public pose checkpoint showed a maximum BF16-versus-F16 decoded-keypoint drift of 1671.8956 pixels, with 15 of 308 keypoints exceeding the issue's 1.5-pixel bound. The shared load policy therefore detects both modern and legacy Sapiens2 configs and retains BF16. Non-attention floating weights are normalized to the loaded attention dtype, while non-floating or quantized attention weights are rejected.

### 3.2 Read RoPE periods from the checkpoint

The backbone loads `backbone.rope_embed.periods`, converts it to f32, and checks that it contains exactly `head_dim / 4` finite positive values. It does not synthesize a fallback from `rope_theta`. This follows the published weights and makes malformed or incompatible checkpoints fail during loading.

### 3.3 Bound pose memory without changing its numerical path

Pose accepts at most 16 boxes and processes them sequentially. Flip testing raises the crop batch only to two. Gaussian blur stays on the MLX device as two matrix passes, then DARK transfers only maxima and seven neighboring values per point to the host. Affine warping uses the host coordinate grid directly, avoiding a redundant device round trip while keeping the upstream half-pixel and padding convention.

### 3.4 Preserve published schemas and layouts

The loader accepts the early nested segmentation config and current flat configs, separate or fused Q/K/V projections, and PyTorch or MLX convolution layouts. The architecture remains config-driven within the published Sapiens2 bounds instead of duplicating one hard-coded network per model size.

## 4. Architecture and Output Contract

```text
checkpoint config + safetensors
          |
schema canonicalization and bounded validation
          |
weight/layout normalization -> Sapiens2 ViT backbone
          |
  task head or backbone features
          |
pose decode / dense resize and unpadding
          |
DenseOutput -> result.safetensors + metadata + task previews
```

The predictor exposes one task per checkpoint. Pose returns `keypoints`, `scores`, and `boxes`, and also supplies structured entries for `keypoints.json`. Segmentation returns `segmentation`; normal estimation returns `normals`; pointmap estimation returns `pointmaps` and `scales`; matting returns `alphas` and `foregrounds`; a backbone checkpoint returns `last_hidden_state` and `pooler_output`. Automatic person detection, video, streaming, and a server route remain outside this issue.

## 5. Verification and Limits

| Check | Result at report time |
|---|---|
| Static implementation review | Complete. BF16 load normalization, f32 flip averaging, and checkpoint RoPE validation findings were fixed; no major correctness finding remains open. |
| Security/performance review | Complete. Allocation-driving config fields and pose boxes are bounded; pose blur and warp avoid the prior large host-copy/scalar-loop path; no major finding remains open. |
| Format and lightweight repository checks | `cargo fmt --all`, Python AST parsing for the committed parity scripts, and `git diff --check` were reported successful. This finalizer independently confirmed `git diff --check`. |
| Fixture integrity | The original JPEG has SHA-256 `16d73869e3267a7d4ed00de8e860833bd1657c1b252e94c0c348277adc7b6edb`. Its Pillow-decoded RGB bytes are preserved losslessly in the canonical PNG with SHA-256 `3342393fccc820a687ec932d9e1b799e74146c0d9319397775169ae5b5f98a69`; the PNG reproduces the upstream reference RGB and pose crop exactly. Source provenance pins mlx-vlm revision `abd3b2d7b728cb5372d4e092260da325e5ded209`. The CPU baseline uses MLX 0.32.4. The primary same-backend reference uses MLX 0.32.5 on gfx1151, with both the Rust native archive and freshly rebuilt Python binding pinned to `bff10220c742907f0c68e55ff174aaea56f74e3d`. Its native library SHA-256 is `9e38e9292550ea8eec9eb1877d06b72f5155ff382ac0e4ec1054818e15276094` and binding SHA-256 is `f8ec0f363bc6eab1cda644a2579ae43aa663939f30ace531d5fb8cc33c7090cc`. The Rust archive copy has ELF visibility-only changes with unchanged executable-section hashes. The earlier MLX `0.32.3.dev20260912+a7d4c85a` ROCm outputs and CPU outputs remain diagnostic history. Checkpoint hashes are recorded for all five validation models. |
| Authentic upstream reference generation | Complete on CPU and on the exact-pin target ROCm backend for all five tasks. The pinned CPU baseline took 537.7 s for pose without flip, 1118.7 s with flip, 528.7 s for segmentation, 781.8 s for normals, 1524.649930 s for pointmaps, and 1916.786644 s for matting. The primary exact-pin gfx1151 references completed with exit 0 and finite source-resolution outputs in 3.913447 s, 3.648052 s, 1.324742 s, 2.306536 s, 4.680237 s, and 4.635040 s respectively. Primary SHA-256 values are pose flip-off `4b6cf998528f0fa1e03ed87e63ae384d76b0fa31c55da0ca6feb7dc7e4416a0d`, pose flip-on `1633beba837f8bccc070b28c33792a34a6696326729acaf7b2a55566d1846ccb`, segmentation `cd6f17a6182b6d5ab738c0e0dc413c98a6e883187faf719191ea37575f932f18`, normals `ab9fdd20f1d0ebcd92b54f8486fc00756b18405497a1a9015402eb7d1559ce45`, pointmaps `f39deabf0e6141d133de8cbbaf45436bc9f4b067fa20d34058734287f9a7ed14`, and matting `e6677044316294248f768102ed976091b57ad6fdad48744f2aab8616cb21cb7c`. These are upstream fixture runs, not Rust parity passes; timings are diagnostics under the shared GPU guard, not benchmarks. |
| Focused Rust compilation | The replacement source containing the canonical-input and BF16 operation-graph fixes compiled successfully in 8m35s. A subsequent focused no-run rebuild containing the device-side RoPE graph and stage dumps completed with exit 0 in 8m46s, and the exact-pin first-block diagnostic build completed in 8m48s. The library build containing the compiled packed-RoPE bridge completed with exit 0 in 9m21s. The comprehensive all-stage diagnostic build then completed with exit 0 in 8m34s. The deconvolution-layout replacement build completed with exit 0 in 8m29s. The compiled-SiLU replacement build completed with exit 0 in 8m22s and produced `target/test-fast/deps/mlxcel-d5e762440845331a`. After rebasing onto `c6ca6c48`, the latest real-model integration target compiled with exit 0 in 7m35s and produced `target/test-fast/deps/sapiens2_real_model-a115aa67e5945075`. |
| Focused unit tests | The current library binary passed on gfx1151 under the shared guard. The `sapiens2` filter reported 26 passed, 0 failed, and 1 ignored, including the exact-pin compiled-RoPE oracle and compact gfx1151 deconvolution-layout and compiled-SiLU-layout oracles. The earlier `grid_sample` filter reported 4 passed and 0 failed. Coverage includes config and head behavior, pose math, shared grid sampling, the BF16 sanitizer, authentic crafted BF16 regressions for fused linear bias, affine-free InstanceNorm, kernel-4 deconvolution, packed RoPE, and graph layout, plus exact-pin staged diagnostics. |
| Real Rust checkpoint parity | **All six current exact-pin same-backend acceptance tests pass.** The saved Rust stage dump reproduces the exact-pin upstream input bit-for-bit. All 16 captured stages in every one of the 24 backbone blocks, final normalization, head features, all deconvolution and convolution raw/normalized/activated stages, and final heatmaps are bit-exact. The explicitly executed staged test passed in 19.59 s with stage-dump SHA-256 `52c1d87e971d188b60d0ae7716972d11211d62c073d0b0885403a4001d7ec7b2`. The rebased real-model suite passed 6 of 6 serial tests in 32.64 s: all 308 unfiltered pose points had maximum errors of `0.001930` pixels without flip and `0.010028` pixels with flip, with exact scores; segmentation label agreement was `0.999723307`; normal relative RMS was `0.000000046`; pointmap relative RMS was `0.000000037`; and the 1B matting checkpoint returned finite alpha `[720,1280]` and foreground `[720,1280,3]` tensors. Matting is a finite shape smoke test rather than numerical parity. Earlier comparisons against MLX `a7d4c85a` are diagnostic because that Python runtime did not match the Rust native pin. The 0.8B and 5B architectures remain unexercised. |
| CLI end to end | The rebased `mlxcel` CLI build completed with exit 0. A guarded gfx1151 pose invocation also exited successfully and emitted `result.safetensors`, `keypoints.json`, and `meta.json`. The structured output contained one person, one four-value box, 308 `[x,y,score]` points, and 308 names. An independent comparison against the exact-pin reference reported a maximum keypoint error of `0.001930` pixels. Build and smoke timings are diagnostic, not benchmarks. |
| Clippy, CI, and platforms | Targeted clippy for the library, `mlxcel` binary, and real-model integration test passed with warnings denied in 15m33s. The required `--lib --tests` coverage then passed from the warm cache in 1m37s with warnings denied, including the cfg(test) modules that had failed previously. The prior `f809d137` CI failures predate these fixes; fresh PR CI remains pending. Metal validation is an explicit platform limitation. |

The tracked repository now uses the exact-pin primary outputs under the stable pose flip-off, pose flip-on, and segmentation fixture names; the earlier `a7d4c85a` ROCm outputs remain under explicit `a7d` names for diagnostic history. The output hash manifest records the complete exact-pin normal, pointmap, and matting artifacts that remain external because of their size. Public 0.4B checkpoints and the 1B matting checkpoint are available for runtime validation. The 0.8B and 5B config URLs were unavailable during this work, so those sizes are supported by the bounded config-driven architecture but are not runtime-validated here.

The five real checkpoints under `.artifacts/sapiens2` are local validation inputs ranging from about 0.8 GB to 3.1 GB. They are deliberately untracked and must not be staged or committed.

The dtype audit did not find an accidental F16 path. The public pose safetensors header declares all 328 tensors as BF16; the modern config matches the family retention gate; normalization keeps the attention dtype; the backbone casts pixels to the BF16 patch weight; and Q/K/V therefore reach SDPA as BF16. The canonical PNG removes the JPEG decoder variable, and the replacement Rust input matches both CPU and ROCm upstream inputs bit-for-bit. CPU-stage comparison still shows relative RMS `0.018312981` and maximum absolute error `0.029296875`, while upstream CPU versus the older ROCm runtime itself differed by as much as 391.849487 decoded pixels, confirming material backend dependence. Acceptance therefore uses the exact-pin upstream implementation on the same gfx1151 backend at the original thresholds. The exact-pin upstream stage capture has SHA-256 `4b33a455525ee3ef3e791b15cfeeb755a71302319a45ea61e3b3c12c8ec05735`; its first-block capture has SHA-256 `ee3ea395e49d24dc40d3135731100e1389bf6e12d50de89a215bb54b132bf348`. The shape-specialized compiled bridge reproduces the exact upstream packed rotation graph, its strict two-value BF16 oracle passes, and every captured backbone tensor is bit-exact. Exact take/stack/transpose/reshape preparation makes every deconvolution stage bit-exact, while the deterministic gfx1151 oracle distinguishes the corrected `21.375` result from the former `21.5`. A direct exact-native experiment showed that upstream compiled and eager SiLU activations have identical BF16 values but different downstream addmm results; the eager layout reproduced the prior Rust stage error exactly. The existing upstream-equivalent shapeless compiled SiLU now makes every captured pose-head tensor bit-exact, and the strict flip-disabled pose output passes. No attention replacement or tolerance relaxation was accepted without stage evidence.

## 6. Review Findings Resolved

- The shared BF16 sanitizer originally converted Sapiens2 weights to F16. The family is now detected from either config schema and retains BF16 end to end.
- Flip-test averaging could inherit a mixed lower-precision dtype. Both heatmap branches are now converted independently to f32 before averaging.
- RoPE construction previously permitted a synthesized fallback. The model now requires, validates, and uses the checkpoint periods.
- The initial DARK implementation copied full heatmaps to the host and performed hundreds of millions of scalar operations. Gaussian blur now runs as MLX matrix passes, with only local decode values transferred.
- Pose boxes and allocation-driving config dimensions now have explicit published-family bounds, and the legacy segmentation schema has a verbatim regression fixture.
- The parity image now uses a lossless PNG made from Pillow's exact decoded RGB bytes, eliminating cross-decoder JPEG rounding without changing the reference pixels.
- Biased Sapiens2 linear layers now use MLX `addmm`, matching `nn.Linear`'s single BF16 bias epilogue instead of rounding a matmul before a separate add.
- Head 1x1 projections use flattened `addmm`; eligible 3x3 convolutions use the upstream 96 Mi-element-per-image im2col threshold and `addmm`; affine-free InstanceNorm uses `fast.layer_norm`.
- Kernel-4 transposed convolutions now use the upstream fixed phase/tap permutation and one GEMM. Kernel-2 and kernel-3 checkpoints retain the native transposed-convolution path.
- RoPE coordinates, angles, sine/cosine tables, packed signs, and paired-component rotation now use the upstream device-side MLX operation graph instead of host-generated tables.
- BF16 packed RoPE now runs through a validated shape-specialized compiled bridge, matching the upstream compiled negative-stride graph and making block 0 bit-exact end to end.
- Kernel-4 deconvolution weight preparation now preserves upstream's take/stack/transpose/reshape graph layout. The values were already identical, but the former concatenation layout selected a different BF16 accumulation path for the large ROCm GEMM.
- Sapiens2 activations now use the existing shapeless compiled SiLU used by upstream `nn.SiLU`. The former eager graph returned identical activation values but changed the layout seen by the next large BF16 addmm.

## 7. Follow-up Actions

- Complete fresh PR CI on the rebased source.
- Preserve the integrated exact-pin pose and segmentation fixtures, hashes, and provenance under the stable primary fixture names.
- Retain the 0.8B, 5B, and Metal execution gaps as explicit validation limitations until matching checkpoints and hardware are available.
- Keep `.artifacts/sapiens2` outside version control.

## References

- PR #2550: `feat: port Sapiens2 dense prediction`
- Issue #2371: Sapiens2 human pose and dense human prediction
- Epic #2382: dense-prediction model delivery
- `docs/dense-prediction.md`: CLI, preprocessing, and output contract
- `tests/fixtures/sapiens2/README.md`: fixture reproduction and comparison procedure
