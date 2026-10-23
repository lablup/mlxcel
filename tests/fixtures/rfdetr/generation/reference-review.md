# RF-DETR reference evidence review

Independent read-only review of the issue #2431 RF-DETR reference harness, staged fixtures, offline verifier, and draft external CLI acceptance. No model or production build was executed by the reviewer.

## Reviewed authentic references

- Upstream source revision: `4319ef2c837b7226b758c3dd69edda51b7079785`
- Corrected native runtime: `c1d8203b257364271e1ea6298705dd3bb0c82a7f06634f5578a209883601a9fc`
- Python binding: `f8ec0f363bc6eab1cda644a2579ae43aa663939f30ace531d5fb8cc33c7090cc`
- Base reviewed receipt: `cd322b62ae9902a1ef05f2f62c02ff0935e6ff92514d1255efab205bbde53cd9`
- Segmentation reviewed receipt: `6cd1e05437fcbf975ccee2917f44094751c4442840e94250e9cd0b4e3897c49c`
- Final reference generator reviewed before execution: `ecd9235c82cd08ed9e614fb6120042eca16d06c49db7a2bd166aa5b63cc463fd`
- Child wrapper: `0c5ddd29ababca8842d4b4d74aff423b6a1a2b0ca495bc5606720a2d279eba47`
- Launch manifest: `191c47c7c8e4dff2b88991f0004125705907ee444280be5ad6bfe51bfd434fa6`

The harness separately validates 487/544 raw checkpoint tensors and 498/559 sanitized loaded F32 parameters. The difference is authentic: upstream sanitization drops one `mask_token` and splits six base or eight segmentation fused `in_proj` tensors into three projections. Source, checkpoint, runtime, image, public predictor, raw-output-to-postprocess byte identity, query mapping, final masks, and child/guard/wrapper status evidence were coherent.

## Independent output corroboration

The reviewer loaded the staged raw safetensors without running either model and executed the exact preserved upstream postprocess definitions. Results:

- Base boxes, scores, labels, and final query indices were array-exact with `base/detections.json`.
- Segmentation boxes, scores, labels, and final query indices were array-exact with `seg/detections.json`.
- All four segmentation PNGs were byte-value exact with upstream OpenCV cubic resize followed by the strict `> 0` mask threshold.

This corroboration establishes the semantic relationship between the reviewed raw outputs and the two immutable receipt hashes above.

## Offline verifier hardening

The initial fixture verifier allowed a coordinated metadata refresh. In a temporary copy, the reviewer changed `num_select` from 300 to 17, query index 0 to 299, label 17 to -999, and class name `cat` to `fabricated`, then refreshed the detection hash, receipt output map, and `SHA256.json`. The old verifier still exited 0 under `python -B -O`.

Candidate verifier SHA-256 `22a6de74a9618918b1f6b3192c4e8fe7f158ca1b69c1f7833f2c81dd036647f5` fixes the gap by checking the two reviewed receipt hashes before parsing and by explicitly requiring `num_select == 300`.

Temporary-copy validation under optimized Python showed:

- Authentic 45-file fixture: exit 0, `PASS`.
- The same coordinated mutation with refreshed subordinate hashes: exit 1 at `reviewed immutable receipt`.

The candidate is sound provided its exact bytes are installed and the fixture inventory is refreshed afterward.

## External CLI acceptance review

Draft CLI acceptance SHA-256 `c27fd44de449010429c78104a878f4ceb8cfb21a615faf0d6128183e54e28d1e` correctly covers external behavior:

- Base and segmentation default threshold 0.5.
- Final NMS count, labels, printed geometric order, boxes within one pixel, and confidence within `1e-3`.
- Segmentation mask filenames in printed order, source dimensions, binary values, and IoU at least 0.98.
- Detection-only mask rejection and unwritable segmentation directory rejection before runtime admission.
- Threshold 1.0 yielding no detections and no PNG files.
- A segmentation boxes-only run without `--masks-dir`, with matching detections and no mask JSON field.
- Durable stdout, stderr, status, nested mask hashes, and recursive relative-path artifact inventory.

The boxes-only case proves the public CLI behavior but does not prove that mask computation, resize, and readback are skipped. Pre-fix production already omitted the JSON mask field without `--masks-dir` while still decoding masks. The production post-check batch therefore needs a direct regression for mask-request plumbing. A strong regression supplies deliberately invalid or over-budget mask logits: `include_masks=false` must return boxes without reaching mask validation, while `include_masks=true` must reach the validation or workset-cap error.

The final immutable launcher must bind the exact CLI script, binary, and build/source receipt. The post-run preservation verifier must pin the final acceptance receipt and require its exact recursive inventory.
