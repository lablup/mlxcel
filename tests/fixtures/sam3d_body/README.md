# SAM 3D Body reference fixtures

These fixtures use `mlx-vlm` revision `abd3b2d7b728cb5372d4e092260da325e5ded209` and the complete public `mlx-community/sam-3d-body-dinov3-bf16` bundle at revision `6d8a283b84c5569c088109589bdf938decf8ad48`. `provenance.json` pins every Body source file, checkpoint file, relevant Python dependency file, interpreter, canonical input and native MLX runtime. The released configuration is **512×512**; the pinned dataclass's 512×384 default is not the released model configuration.

The native runtime is MLX 0.32.5 at `bff10220c742907f0c68e55ff174aaea56f74e3d`, with the tracked forward LayerNorm synchronization correction. Native SHA-256 is `c1d8203b257364271e1ea6298705dd3bb0c82a7f06634f5578a209883601a9fc`; Python binding SHA-256 is `f8ec0f363bc6eab1cda644a2579ae43aa663939f30ace531d5fb8cc33c7090cc`. Both successful references ran on gfx1151 with TF32 disabled. Stored checkpoint types are preserved: 615 BF16, 574 F32, 11 I32 and 3 U8 tensors before body-only filtering.

## Public-loader compatibility

The unmodified public `SAM3DPredictor.from_pretrained` fails on this released checkpoint: its strict MHR head load rejects four sanitized buffers absent from the body model: `parameter_limits`, `pmi`, `texcoord_faces` and `texcoords`. The fixture generator temporarily filters exactly those four entries from the head's load list, calls the original strict loader, and restores the inherited method before inference. The pinned raw loader explicitly skips PMI and texture buffers, and MHR forward explicitly does not apply parameter limits. No forward, preprocessing, public `predict`, or evaluation operation is changed.

The body-only model has exactly 958 checkpoint-backed arrays: 1203 minus 223 hand-refinement arrays (including four bare names), 18 unused mask-downscaling arrays and those four unused MHR buffers. Every retained array's shape, dtype and full bytes are checked against the authenticated checkpoint before prediction. BF16 bytes are obtained through a lossless uint16 view, not a numerical conversion. Coarse hand tokens and PCA processing remain part of the body model.

Three pre-inference attempts are retained externally: the original strict-load failure, the expected-key audit's omission of four bare hand-refinement names, and the BF16 NumPy buffer-interface incompatibility. They remain failures; only the corrected boxed-v4 and default-v1 runs have successful child, wrapper and GPU-guard exits.

## Inputs and outputs

The input is the official SAM 3D Body `dancing.jpg` asset from source revision `b5c765a0d89d789985e186d396315e7590887b94`. It and its mask remain external because asset-specific redistribution permission was not established. OpenCV-decoded RGB pixels were saved losslessly as PNG to eliminate decoder differences. The canonical PNG and decoded RGB hashes are pinned; the original JPEG and input-preparation script remain in ignored evidence storage. The explicit person box is `[732,114,1523,1451]`; the default case passes no box and uses the whole 2250×1500 image. Both use upstream default camera intrinsics.

`reference-{boxed,default}.safetensors` contain complete F32 vertices `[18439,3]`, keypoints `[70,3]`, joints `[127,3]`, camera `[3]`, bbox `[4]`, and I32 faces `[36874,3]`. All are finite; mesh, joint and keypoint spans are nonzero. The accompanying manifest binds full receipts, tensors and exact successful child/wrapper/guard status files. Resource receipts retain separate SYSTEM and framebuffer telemetry; diagnostic elapsed time is not a benchmark.

The comparison requires all vertices and keypoints to have maximum absolute error strictly below 1e-3 metres, and camera relative L2 error strictly below 1e-3. It requires exact faces, finite complete outputs, and a colorless OBJ whose coordinates agree with the tensor within six-decimal serialization. No low-confidence filtering or vertex subsampling is used. Rust checkpoint parity and actual CLI acceptance are pending.

## Reproduction

Use an isolated environment matching `provenance.json`: NumPy 2.2.6, OpenCV 4.12.0, safetensors 0.6.2, ml_dtypes 0.5.3, and the exact MLX runtime above. OS libraries and Python startup configuration remain trusted; this is not a whole-system attestation. Set `MLX_ENABLE_TF32=0`, `PYTHONDONTWRITEBYTECODE=1`, and explicit `PYTHONPATH`/`LD_LIBRARY_PATH` for that environment. Download the pinned checkpoint and input assets, then verify their hashes.

Run `generate_reference.py` through `run_reference.py` under `scripts/rocm_gpu_guard.sh --hold`. The monitor requires `--monitor-sha256`, the reviewed helper manifest and its SHA-256, fresh `--run-dir`, and explicit source, bundle, image and Python paths. `--case boxed` and `--case default` select the two fixed cases. The monitor independently enforces host and framebuffer reserves and records child and wrapper statuses separately; retain the external guard exit too. Never edit a running launcher or helper snapshot.

`generate_oracles.py` captures unchanged upstream rotation, compact-body/hand conversion, two-joint FK and ordered scatter functions without loading model weights. It also captures OpenCV affine transforms, crop pixels, normalization and CLIFF values. `geometry.safetensors` is the complete 6632-byte geometry fixture. `crop-tiny.safetensors` selects only `tiny_full.*` and `tiny_partial.*` from a deterministic synthetic image; no source photograph is included. `small-oracle-derivation.json` records the exact original generator, manifest and extraction hashes. The full real-image crop oracle stays external. Historical receipt bytes need not reproduce after provenance-only hardening; the original numerical payloads remain the comparison target.
