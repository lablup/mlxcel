# PP-DocLayoutV3 reference fixtures

These fixtures validate the two released PP-DocLayoutV3 checkpoint layouts and
the shared detector API on a real two-column document page.

- `two_column_page.png` is page 1 of *BERT: Pre-training of Deep
  Bidirectional Transformers for Language Understanding*, rendered at 144 DPI.
  The paper is ACL 2019 material distributed under CC BY 4.0. Exact source,
  citation, renderer, and hashes are in `page-provenance.json`.
- `input.safetensors` contains the decoded RGB bytes and the exact
  `[1,800,800,3]` F32 tensor produced by Transformers 5.17.0 with Torch
  2.14.1 and Torchvision 0.29.1 on CPU. This uses Torchvision's native uint8
  bicubic resize, without antialiasing; Pillow bicubic is not equivalent.
- `paddle_raw.safetensors` and `hashnuke_raw.safetensors` contain the full
  F32 `logits`, `pred_boxes`, and `order_logits` outputs from the pinned
  Paddle/Hugging Face and converted MLX layouts. The outputs are byte-identical.
- `expected_detections.json` preserves the unchanged Falcon layout
  postprocessor records and the unrounded Rust acceptance records. The latter
  exercise flat query-by-class top-Q selection and page-global query ordering
  before threshold filtering.
- `rt_detr_heron_expected.json` preserves the pre-change RT-DETRv2 CLI result
  on the same page. RT-DETRv2 has no order head, so its entries intentionally
  omit `reading_order` and retain the historical geometric CLI order.

`generation/` contains the pinned reference scripts. `execution/` contains
their separate child, GPU-guard, and wrapper statuses and logs. The two
`*_provenance.json` files record the exact checkpoint revision, loaded tensor
set, source manifest, device, dtype, TF32 setting, native library hashes, input,
and output hashes. `upstream-manifest.json` pins the upstream implementation,
and `SHA256.json` inventories the committed fixture tree.

Regeneration requires the exact revisions and environments recorded by those
manifests. Never replace a reference after a failed wrapper or child exit. Run
the raw model references under `scripts/rocm_gpu_guard.sh --hold`, then run
the CPU postprocessor over the saved outputs so model execution and decoding
remain independently auditable.

`execution/final-cli-falcon/` preserves the successful actual detector-to-OCR
run: both checkpoint layouts produce identical ordered detections, and the
Falcon transcript contains the three independently specified source anchors in
order. Its receipt binds the CLI/test binaries and historical source manifest;
`source-binding.json` identifies the immutable source commit. These records
remain bound to that captured runtime, including after later integration.
`verify_fixtures.py` checks the exact inventory and the exit, device, option,
source, output, and anchor relationships with explicit checks under `python -O`.
