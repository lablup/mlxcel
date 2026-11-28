# Technical Report: PR #2581 — Native RF-DETR detection and instance masks

> The RF implementation and all local integrated source, runtime, CLI, unit, architecture, contract, format, fixture, and lint gates pass. This report remains Partial until its report-bearing head completes CI and the merge identity is recorded.

| Field | Value |
|---|---|
| Date | 2026-10-24 |
| PR | #2581 |
| Issue | #2431 |
| Integrated source head | `383d77a2a22231f9cad108b147ff7562f0e5b800` |
| Merged Body base | `68d7cc36d6e7fe4f61f126a5490467a7fe90f5d3` |
| Historical runtime source | `f673b9d134f61b0d43e22d1f627036e53a002f6f` |
| Historical lint/metadata head | `d36af6213788ba50969d926787333f9e2d7b3c5e` |
| Status | Partial; all local integrated gates pass; report-bearing-head CI and merge lifecycle pending |
| Languages | Rust, Python, Markdown |
| Risk level | Medium until the report-bearing head completes CI |
| Validated backend | Integrated `383d...` ROCm/gfx1151 Base/Seg parity, RF CLI, retained PP/RT CLI, RT-DETRv2 regression, and nine-family architecture |
| Explicitly not validated | Final contract checks; Metal or CUDA inference |
| Performance claims | None; no benchmark was run |

## Executive summary

PR #2581 adds RF-DETR Base detection and RF-DETR Seg Small instance segmentation to the shared `mlxcel detect` surface. The implementation ports the upstream windowed DINOv2 backbone, C2f projector, two-stage deformable decoder, bounding-box reparameterization, per-class NMS, and optional segmentation head. It registers both `rf-detr` and `rfdetr`, uses the upstream default confidence and NMS thresholds of 0.5, and emits source-image pixel boxes with COCO labels.

Segmentation masks are opt-in through `--masks-dir`. The command writes source-resolution binary PNG files in printed detection order and records each path in JSON. Box-only execution is designed to skip mask-head output and mask resize/readback. Detection-only checkpoints reject `--masks-dir` before loading weights, and mask output refuses existing files, symlinks, and unwritable paths.

The historical runtime source at `f673b9d1...` passes the reviewed Rust type check, focused and detect binary-unit tests, authentic Base/Seg numerical parity, RF CLI acceptance, retained PP/RT CLI compatibility, and the RT-DETRv2 regression on ROCm/gfx1151. The narrow lint/backend metadata update at `d36af621...` passes warning-denying clippy and independent equivalence review without changing model math.

The clean integrated head `383d77a2...` is an additive three-commit chain on merged Body `68d7cc36...`. Independent integration and documentation reviews report no finding. Both required documentation pages contain the same reviewed nine-family table and correctly distinguish checkpoint `model_type` spellings from the `rfdetr` registry/loader alias. Its source-bound library build, focused and binary units, Base/Seg parity, RT-DETRv2 regression, RF and retained-family CLI acceptance, nine-family `arch --json`, contracts, formatting, fixture verification, and warning-denying clippy all pass. Only complete CI for the commit containing these reports and the merge identity remain pending.

## 1. Problem and scope

Before this PR, `mlxcel detect` could not load RF-DETR checkpoints. RF-DETR differs from the standard DINOv2 family because it uses per-window CLS tokens, alternating local and global backbone attention, a multi-scale projector, two-stage query selection, and an optional mask head. Routing it through `mlxcel predict` would also split one detector family across two public commands.

The selected design keeps RF-DETR under the shared `Detector` API and `detect` CLI. It supports the released F32 Base and Seg Small checkpoint layouts. PyTorch conversion, video/realtime modes, annotators, and server exposure remain outside this PR.

## 2. Implementation

### Native graph and preprocessing

The new `src/vision/detection/rfdetr/` module implements:

- square RGB preprocessing at the checkpoint resolution with ImageNet normalization;
- the RF-specific windowed DINOv2 backbone and positional interpolation;
- the C2f multi-scale projector;
- two-stage query selection and deformable decoder;
- bounding-box reparameterization and per-class greedy NMS;
- the optional segmentation head and source-resolution bicubic mask resize.

The Base checkpoint uses resolution 560, 300 inference queries, and three decoder layers. Seg Small uses resolution 384, 100 inference queries, four decoder layers, and 96 x 96 raw mask logits.

### Shared CLI behavior

`load_detector` dispatches the RF model type through the existing detector boundary. The CLI keeps detector-specific defaults, sorts detections geometrically for stable output, and adds mask paths only when `--masks-dir` is supplied. Existing RT-DETRv2 and PP-DocLayoutV3 JSON fields remain unchanged.

The intended public flows are:

```bash
mlxcel detect -m models/rfdetr-base-fp32 -i image.png --format json
mlxcel detect -m models/rfdetr-seg-small-fp32 -i image.png --format json --masks-dir masks
```

### Checkpoint and resource admission

Configuration parsing, checkpoint indexes, safetensors headers, tensor shapes, dtypes, dimensions, allocation totals, image payloads, and output mask worksets are bounded before expensive work. Sanitization handles upstream namespace remaps, drops the inference-unused mask token, splits fused decoder QKV parameters, and transposes convolution weights only when their layout requires it.

The reviewed security model treats the selected checkpoint directory as stable while it is loaded. Every consumed config, processor, index, and weight file is validated, and untrusted dimensions and allocations remain checked. It does not claim protection against a privileged process replacing the already selected directory concurrently.

## 3. Authentic upstream reference evidence

The tracked reference package uses unchanged mlx-vlm source revision `4319ef2c837b7226b758c3dd69edda51b7079785`, the corrected ROCm LayerNorm runtime, the committed non-square cats image, and exact released checkpoints.

| Variant | Checkpoint tensors | Sanitized loaded parameters | Public detections | Masks |
|---|---:|---:|---:|---:|
| Base | 487 F32 | 498 F32 | 5 | None |
| Seg Small | 544 F32 | 559 F32 | 4 | 4 nonempty source-size masks |

The immutable Base and Seg receipt hashes are respectively `cd322b62ae9902a1ef05f2f62c02ff0935e6ff92514d1255efab205bbde53cd9` and `6cd1e05437fcbf975ccee2917f44094751c4442840e94250e9cd0b4e3897c49c`. The fixture package has 46 tracked entries and inventory identity `91ff3b3d3a6136c33155bf69c7c1b46f7387a0f70600e8346819f8c7d9e8b063`.

Independent audit `fc53893e3bc3d1235140b47f155708ff68db231933d159dddbb905a1e405879b` re-executed the exact preserved public postprocessor without running either model. Base and Seg boxes, scores, labels, and query indices matched exactly, and all four PNG masks matched the public cubic-resize/strict-positive result exactly. The fail-closed verifier pins both receipts and rejects a coordinated detection, receipt-map, and inventory mutation under `python -O`.

These results authenticate the expected inputs and outputs. By themselves, the upstream results do not establish Rust parity; the separate Rust gates below do.

## 4. Historical pre-rebase Rust numerical closure

The focused build completed successfully in 10 minutes 52 seconds from reviewed source `f673b9d134f61b0d43e22d1f627036e53a002f6f`, source snapshot SHA-256 `330f4cae5ce7584df6da3e36040fad92ae855cb30603c88d4adb39195eea61d3`. The test binary `target/test-fast/deps/mlxcel-d5e762440845331a` has SHA-256 `48bfd9cfda13d34724c08e4494e60fc370bda248edc28febad44d0333952fb4e`.

The focused unit selection reports 23 passed, 0 failed, and 2 ignored real-model tests. It covers sanitization, configuration, allocation limits, preprocessing admission, window partitioning, bbox reparameterization, NMS, nonfinite/oversized output rejection, custom labels, and `unrequested_masks_skip_the_mask_computation_entirely`.

A separate `src/main.rs` unit binary then passed all eight selected `commands::detect` tests, with 303 unrelated tests filtered out. These tests cover stable geometric and confidence ordering, optional reading order and mask JSON fields, rejection of mask requests before runtime initialization for detection-only models, and refusal to overwrite through a pre-existing mask symlink. The test log SHA-256 is `5b1fef991fc9d50335dd36d5c2d4ad14c4ab2456de2dbdcd627f39f72b5da8a4`; build and test status are both 0.

The first clippy run is preserved with status 101 after finding four test-only initializer lints. Corrective diff `654e1c03f4bd53fc4a94560edd98d88cec17d62946d2a332d092e52ecea439b7` rewrites those four test initializers and corrects RF registry backend metadata to ROCm supported with Metal/CUDA partial. It does not change model math. The second warning-denying clippy run completed in 2 minutes with status 0; its log and launcher SHA-256 values are `9e1074615024b5fbdd1776d4cddb0accb68119065f90f907c4097eb1fbaa2c47` and `31038483c326c6ba582e085d81198af733bc57ad3c7c32ceda26d559de8f85bc`. Independent repository-map review found no equivalence issue, and the change is committed and pushed as `d36af6213788ba50969d926787333f9e2d7b3c5e`. The authentic runtime evidence remains bound to unchanged source `f673b9d1...`.

The immutable 116-entry historical packet is stored under `proof/pre-rebase-f673-d36-v1`. Its `SHA256SUMS`, receipt, and payload-manifest SHA-256 values are respectively `4013b8c3af201ed57a56460acd506425b5a8f2311c6b9697e44e37ab91da578b`, `1b4281dc9a64b3aa651fc2dd96317aff73bda311b9ebf4492710066922ed362c`, and `4a946243dbf1762d9213bf138254af8e54525eaa7900dfb30375da8d8c32ff57`. The packet explicitly separates the `f673...` runtime source from the `d36...` metadata-only source.

The two ignored authentic tests were then run separately under the GPU guard:

| Variant | Processor max abs | Logits max abs | Boxes max abs | Raw masks max abs | Final result |
|---|---:|---:|---:|---:|---|
| Base | `0` | `3.342628479e-4` | `1.327395439e-4` | N/A | Final NMS parity passed |
| Seg Small | `0` | `0` | `0` | `5.340576172e-5` | Four masks, each IoU `1.0` |

All values satisfy the issue's `1e-3` raw tolerance and `0.98` mask-IoU requirement without relaxation. The authentic shared-detector RT-DETRv2 regression also passes on gfx1151: 1 passed, 0 failed, with child and guard status 0. Its child log SHA-256 is `ab6ffe805083127aa94cb6d275fa5a911eec07c2a8cf1fd71585ff365b5387a3`.

### Pre-rebase CLI and retained-family closure

The actual CLI gates used binary SHA-256 `0ca46ccc86f56df111c1094aedd717d8ccdee0ade6543aaf1ee13cb81f7d2334`, bound by source receipt `512d41b447156df51ed2a6b935767d4a5631b28e3a524c6bd734d2182388890e` to clean source `f673b9d134f61b0d43e22d1f627036e53a002f6f` and source snapshot `330f4cae5ce7584df6da3e36040fad92ae855cb30603c88d4adb39195eea61d3`.

RF acceptance receipt `43dd5559d560daebdc713658ab3bea1fe2ddb1b83c33068aca320ecf9d7c1414` records five Base detections and four Seg detections at the family default threshold 0.5. All four source-size binary masks have IoU `1.0` against the authentic fixtures and use printed-order filenames. The Seg box-only path preserves labels, boxes, confidences, and count without a mask field. Detection-only mask requests and unwritable mask destinations fail before GPU execution, while threshold 1 returns an empty list and writes no PNGs. Child and guard status are both 0.

Retained-family receipt `e152c7f9583d2fe8123bbc16c657e1808770a9518b0d4e2e1508aa3cc2832bf4` records 16 ordered detections from each PP-DocLayoutV3 checkpoint and 28 RT-DETRv2 detections. Both PP outputs are identical, all three runs preserve the default threshold 0.3 and exact no-mask JSON schemas, and child and guard status are 0. The gate verifies the Falcon OCR model and processor sources are byte-identical to the merged PP baseline; the only relevant source delta initializes the new detection field as `mask: None`. It pins the historical authenticated full OCR execution instead of claiming a new OCR run.

### Final additive integration source identity

The clean integration chain is `383d77a2...` (documentation consolidation), `acd22bdf...` (lint/backend metadata), and `c0e219b5...` (RF implementation) on merged Body `68d7cc36...`. The integrated tree SHA is `f2d5682707a1ed204827d6ad2c91a31f2e59b942`.

`docs/dense-prediction.md` and `docs/supported-models.md` contain byte-identical nine-family contract tables with table SHA-256 `4f7c113fcb497711097c2b19c1ba1178845ad11e0b8b1912166b35b00917dd79`. Both pages state that every `predict` family uses the shared writer, including `instances.json`, while RF-DETR uses `detect` and writes masks only when a segmentation checkpoint is paired with `--masks-dir`. The registry and loader agree on `rf-detr` as the checkpoint spelling and `rfdetr` as the accepted alias. Independent integration and documentation review found no remaining source or documentation issue.

### Integrated library and runtime closure

The final integration packet binds source archive SHA-256 `9295e4d7952454622353c157f77e36a446ffa3b315eb3da27b2e01c77eefe3ec` and source-envelope SHA-256 `3e510f289bcf34b7f52bd0cc92283f4d66eec1ce3aba2f70db664452fb1804a2` to clean head `383d77a2...`. The integrated library test binary has SHA-256 `c53a6fdeda249f74916a0ef764d910069aac8f32556ed98dba1d727c1b71170e`; its 9m43s build completed with status 0 and log SHA-256 `29cef0210219cc0def2b7ad77425435e9160e40e91e5b5a0fec6612d9d09a895`.

That binary passed the focused selection with 23 tests passed and two authenticated real-model tests ignored for separate execution (log SHA-256 `482f8691f678c114a35889a4d2ecff6935f46946b8294e5d2bcbf3eff122f3be`). The separately guarded Base parity, Seg parity, and RT-DETRv2 regression all completed with child and guard status 0. Their log SHA-256 values are respectively `e0ec1281b73dd3995c4b0c5679f7f0869f1c4ae8f2d9bb0814a67ed0bb349c40`, `b24156c4985fd12e4002b993bdac4ee6edb1591a9e00ac502e7cd536c494430c`, and `d792499cfbe247a31264c627b3d213fa48f808b6e19dc9c5276566248e560c7d`; the guard log SHA-256 is `7486977be14120ab93ca31e915eb64a24aa99de91b6c3272e5384993a41cffa4`. The numerical results exactly reproduce the historical accepted values in the tables above.

### Integrated CLI and architecture closure

The canonical integrated CLI has SHA-256 `328d10b9fffe842e981eeca317a9a00eb18c8d91cc0f5c04e5d11be102b9a685` and is bound to source/build receipt SHA-256 `bf12b39ec441c17463603e190e6b1d65b58fc4e252107077bcbcc9894444b831`. RF acceptance receipt SHA-256 `cd63ca1e4aeae89c598efdcb6ef74e2857837b12df7c572857b8e5606b095c4a` records Base 5 and Seg 4, all four mask IoUs `1.0`, Seg boxes-only parity, pre-runtime mask errors, and threshold-1 empty output. Its child and guard logs have SHA-256 `4bddd0f2ca3ec90e23b728b0d2e0adee27bf63179163456270a2e8c12f88de98` and `7bb5acf0166eae50e0db80172b8c60906157267b70939aeeb983c3e08fc44cb9`; both statuses are 0.

Retained-family receipt SHA-256 `553f670b3e55ef0e944e9f9ff6d16ea0198f29cf5ed4f16699b6077bf83dd777` records identical ordered 16-item outputs from both PP layouts and 28 RT-DETRv2 outputs. It pins the historical authenticated full OCR proof and verifies current Falcon source equivalence without claiming another OCR run. Its child and guard logs have SHA-256 `b8e6d5d7890b18d491a4568a277f5fd1e78f1d351f0d951422d4202c5b78ef86` and `f3e5df928966fe516e3b279157f7760e6558ec798ede031aa4a174fac6c99e8b`; both statuses are 0.

The final `arch --json` validator, SHA-256 `df9b686c2d0399d6b7acac48b8a8e0fdb5fd40b6d25d6090dbecf8dc39f4db19`, reports `PASS` for all nine documented families. Its result SHA-256 is `433721d43ce978779e48971a2385cc5627a96ca695e3d0d908021ef2694bf1e8`, status 0. RF-DETR reports runtime `detect`, `model_types: ["rf-detr"]`, alias `rfdetr`, output `boxes`, ROCm supported, and Metal/CUDA partial, matching both reviewed documentation tables.

The integrated binary-test executable has SHA-256 `851bd5103af9399ca0ff803c05dca5ed09c6ca69b596eee78bf1702935f1a93a`. Its build log SHA-256 is `e847c4bf11774b9053042744e412c322a302ddfb968175c4e9ebb4f0d286fd00`, and the selected run log SHA-256 is `413804bd719d1d8e4f37b442ab677636e5fa7e5eb1c8017b5cb1e55d153bf453`; both statuses are 0. The run passed all 20 selected command tests, including the eight detect ordering, JSON, mask opt-out, and symlink/pre-runtime error tests, with no failures or ignores.

### Integrated quality and contract closure

Warning-denying clippy (`--lib --tests --bin mlxcel -- -D warnings`) completed in 1m48s with status 0. Its wrapper and log SHA-256 values are `fec136558b2459bf3a46e126fdb4a59eb834bffff204449631765bd1dbb2fd34` and `c30358c1535a2cfe96d5767ad4efb1a34409bb8d4302c389e93fc103c6ff70c9`.

The final fast-contract suite passed all eight groups: crate versions, kernel dtype keys, environment-variable documentation, kernel-port dispatch, llama compatibility, ROCm overlay/ownership, tracked binary assets, and Windows reserved paths. `cargo fmt --all -- --check` and `git diff --check` also passed. The launcher, core script, and log SHA-256 values are `759a6d1203d592579e4c3b81a61e5264c13261be4a263fc2125483c0462e86d2`, `d8eb7481ece1ba762239805cecfa4e22a8164cecab812f7fa15bfdce6d3db996`, and `ab2293b74ba215e53a8c5ac09e9cf5cc3f8451c4f2c008eee79d1654c0294d61`; status is 0.

The optimized fixture verifier's corrected second attempt reports `PASS: 46 authentic RF-DETR fixture files`, status 0. Its wrapper and log SHA-256 values are `cd0492d8bc46cb8e5367b9f474c52e12557d2374c51c07129eec8c4c5ea5f9d2` and `2134b53eef7f2e7e9f9fa368cf67f07c862676e7b513bb317de2d3e8f8514c57`. The initial missing-Pillow environment failure remains preserved and excluded. Final CLI and architecture evidence index SHA-256 is `06a0db219c59af1ec7d4fc168cbfed41afebcabfe2327cddd9255cdb014483f3`.

## 5. Validation status

| Gate | Status | Evidence or requirement |
|---|---|---|
| Historical Rust type check | Passed | Pre-rebase source `f673b9d1...` |
| Independent implementation/security source reviews | Passed | No remaining source finding in historical implementation; stable-directory trust scope documented |
| Integrated source and docs review | Passed | Clean `383d77a2...`; additive integration and nine-family tables reviewed without finding |
| Authentic upstream Base reference | Passed | 498 loaded F32 parameters; five public detections |
| Authentic upstream Seg Small reference | Passed | 559 loaded F32 parameters; four detections and four masks |
| Fixture verifier and mutation rejection | Passed | Corrected standalone verifier reports 46 authentic files; immutable receipt gates; `python -O` negative rejected |
| Focused Rust unit suite | Passed at pre-rebase head | 23 passed, 0 failed, 2 authentic tests ignored and run separately |
| Detect binary-unit suite | Passed at pre-rebase head | 8 passed, 0 failed, 303 unrelated tests filtered out |
| Rust Base raw/postprocess parity | Passed | Processor `0`; logits `3.342628479e-4`; boxes `1.327395439e-4`; final NMS passed |
| Rust Seg raw/mask parity | Passed | Processor/logits/boxes `0`; masks `5.340576172e-5`; four IoUs `1.0` |
| Direct mask opt-out regression | Passed | `unrequested_masks_skip_the_mask_computation_entirely` |
| Authentic RT-DETRv2 regression | Passed | 1 passed on gfx1151; child and guard status 0 |
| Real Rust CLI acceptance | Passed at pre-rebase head | Base 5; Seg 4 with four IoU-1.0 masks; box-only path; pre-runtime errors; threshold-1 empty output |
| Retained PP/Falcon compatibility | Passed at pre-rebase head | Both PP layouts: 16 identical ordered outputs; RT-DETRv2: 28; exact schemas/no mask; historical full OCR proof pinned |
| Warning-denying clippy | Passed on pre-rebase lint-adjusted source | Second run status 0 in 2m00s; first status-101 lint finding retained |
| Narrow lint/metadata diff equivalence | Passed | Independent repository-map review found no issue; committed as `d36af621...`; no model-math change or runtime rerun claimed |
| Integrated library build and focused units | Passed | Source-bound binary `c53a6fd...`; build status 0; 23 passed and 2 authenticated tests run separately |
| Integrated Base/Seg and RT runtime | Passed | Base and Seg reproduce accepted metrics; four Seg IoUs `1.0`; RT regression passed; child/guard statuses 0 |
| Integrated RF and retained-family CLI | Passed | Binary `328d10b9...`; RF Base 5 and Seg 4/IoU-1 masks; PP 16/16 identical; RT 28; all child/guard statuses 0 |
| Integrated binary-unit suite | Passed | Source-bound binary `851bd510...`; 20 passed, 0 failed, 0 ignored; build/run statuses 0 |
| Integrated nine-family architecture | Passed | Validator `df9b686c...`; all nine families match reviewed runtime/model-type/alias/backend contracts |
| Integrated contracts, format, and diff | Passed | Eight fast-contract groups, `cargo fmt --check`, and `git diff --check`; status 0 |
| Integrated clippy | Passed | `--lib --tests --bin mlxcel -- -D warnings`; status 0 in 1m48s |
| CI | Pending | The complete required check set has not been recorded for the commit containing these reports |
| Nine-family documentation and architecture | Passed | Both required tables are identical, reviewed, and corroborated by final `arch --json` |
| Body integration | Passed | RF chain is based on merged Body `68d7cc36...` |
| Report-bearing-head CI | Pending | Complete CI on the commit containing these finalized EN/KO reports |
| Merge | Pending | Draft PR; no merge identity claimed |

The earlier combined fixture-verifier wrapper that reported a false zero is excluded from evidence. Only the corrected standalone `PASS: 46 authentic RF-DETR fixture files` result is used.

## 6. Compatibility and limitations

- The implementation targets the two reviewed released F32 checkpoints. Broader checkpoint compatibility is not claimed.
- RF-DETR uses `mlxcel detect`, prints JSON to stdout, and creates mask files only when a Seg checkpoint is paired with `--masks-dir`.
- No result directory, `result.safetensors`, or `meta.json` is created by this detector command.
- The tracked reference paths identify the original trusted local environment and immutable checkpoint contents; they are provenance, not portable installation paths.
- Metal and CUDA inference were not tested for this report.
- No throughput, latency, memory, or cross-device benchmark was run.
- The final documentation source now lists all nine accepted families with their runtime, `model_type` aliases, tasks, and concrete files; publication still requires the integrated runtime and `arch --json` corroboration recorded above.

## 7. Change summary

| Item | Current PR value |
|---|---:|
| Integrated source head | `383d77a2a22231f9cad108b147ff7562f0e5b800` |
| Files changed from merged Body | 82 |
| Lines added | 28,029 |
| Lines deleted | 64 |
| RF commits on Body | 3 |

Most added lines are the native RF implementation plus authenticated upstream source, safetensors metadata, receipts, and fixtures. The production changes are concentrated in the RF module, shared detector boundary, detect CLI, registry, tests, resource policy, and current documentation.

## 8. Required follow-up

- [x] Build the integrated library and run focused units plus bounded Base/Seg and RT-DETRv2 runtime gates.
- [x] Complete integrated RF CLI, retained PP/RT CLI, and final nine-family `arch --json` gates.
- [x] Run the integrated 20-test command binary-unit selection.
- [x] Run final contracts, format/diff checks, fixture verification, and warning-denying clippy.
- [ ] Complete all required CI checks on the commit containing these EN/KO reports.
- [ ] Record the eventual merge commit.

## References

- PR #2581: `feat: add RF-DETR detection and instance masks`
- Issue #2431: native RF-DETR detection and segmentation
- Independent reference audit: `/tmp/epic-2382-rfdetr-reference-review.md`
- Final documentation audit: `/tmp/epic-2382-final-docs-audit.md`
