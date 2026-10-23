# Video Depth Anything authentic reference fixtures

These fixtures use genuine consecutive DAVIS `car-turn` photographs and the
published Small relative and metric MLX checkpoints. They contain no synthetic
weights or substitute model outputs.

## Pinned sources

- Upstream model and inference implementation: `Blaizzy/mlx-vlm` commit
  `abd3b2d7b728cb5372d4e092260da325e5ded209`.
- `mlx-community/Video-Depth-Anything-Small-MLX` revision
  `a12a442242291eed6b885b196cbbdb51f3bcdb87`.
- `mlx-community/Metric-Video-Depth-Anything-Small-MLX` revision
  `0d4ff7a5907b62654d4115d83ce3b2af317649e2`.
- DAVIS frames: `AlonzoLeeeooo/DAVIS-Edit` revision
  `3c33c3245ce366cf3f533d35f4a3f1c295ab889c`,
  `JPEGImages/car-turn/00000.jpg` through `00063.jpg`.

`source-manifest.json` records every downloaded source's URL, size, and SHA-256.
Each output's provenance records checkpoint/input/output checksums, all 351
actually loaded parameter names/shapes/dtypes, preprocessing, device, library
versions, and complete output shape. Both checkpoints load float32 parameters;
no conversion to BF16/F16 is used in these references.

## Authenticated fixture inventory

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `davis_64.mp4` | 69,065 | `50b864d48ddc081043cfaab8177fcab54301c68205baf90bccd42a4b157084b3` |
| `davis_40_frames.safetensors` | 1,474,656 | `d151179a89f32a879d0f94d8f95a4c4d5a8d11bf5cf6e16b725f6c10410b6555` |
| `small_relative_40.safetensors` | 1,966,288 | `980442c3e1ebfc2291be035a80a4b8d6647dcfb31dcd9ad21e376d51fe532af8` |
| `small_relative_40.provenance.json` | 52,563 | `4cb484a350a0c049b7e019331c93d320cdbad999ebc4752c0e631b1149c116bf` |
| `small_metric_40.safetensors` | 1,966,288 | `c0f5e28da23645fa9cd146c07db270667aac490217f870984f1eb927e699a2cc` |
| `small_metric_40.provenance.json` | 52,807 | `ccdee4a9abddaefbe1b39af4944dcbc2c06e4cb5633f1429786ff16f1d28e881` |

Both output files contain only a finite float32 `depth` tensor with shape
`[40, 96, 128]`. The relative provenance intentionally retains the original
17-key generator schema. The metric provenance uses the current 21-key schema,
including disabled TF32, source/runtime identities, and stage timing.
`metric_cpu_run/` preserves the clean metric completion log and zero status.

## Regeneration

Use a disk cache for checkpoints and source. `download_reference_assets.py`
accepts `--cache PATH` and fetches only revision-pinned assets. The generator
requires MLX, numpy, safetensors, transformers, Pillow, OpenCV, and ffmpeg.
The recorded run used OpenCV 4.12.0.88 and Pillow 13.0.0.

```sh
python download_reference_assets.py --cache /path/to/disk-cache
ffmpeg -framerate 12 -i /path/to/disk-cache/frames/%05d.jpg -frames:v 64 \
  -vf scale=128:96:flags=bicubic -c:v libx264 -crf 18 -pix_fmt yuv420p davis_64.mp4
python generate_reference.py --cache /path/to/disk-cache --fixtures . --kind relative --device cpu
python generate_reference.py --cache /path/to/disk-cache --fixtures . --kind metric --device cpu
```

To download all six published presets for the load-only gate, add
`--all-checkpoints` to `download_reference_assets.py`.
`checkpoint-matrix-manifest.json` records their pinned revisions, complete
checkpoint checksums, config fields, tensor counts, and stored dtypes. Base and
Large validation loads their actual weights and checks shapes and dtypes; it does
not claim full forward inference or numerical parity for those sizes.

Run from this directory, or supply absolute paths. `--ffmpeg` selects the decoder.
The clip intentionally resizes the source to 128x96 to keep full output fixtures
small; inference still uses the checkpoint's default 518 lower-bound processor,
producing 518x686 normalized input. Codec/version changes may change decoded
pixels; `davis_40_frames.safetensors` records the exact decoded RGB input used by
the committed model references. The CLI acceptance uses all 64 clip frames.

The full 40-frame output crosses the 32-frame model window boundary. The actual
upstream predictor executes two complete windows with its keyframe selection,
relative scale/shift alignment, and overlap blending. Metric inference uses the
upstream metric alignment bypass. Every output value is stored; parity is not
based on a few sampled pixels. CPU generation writes committed fixture outputs;
`--device gpu` writes separate device references under the disk cache's
`gpu-references` directory. Guard GPU generation with `rocm_gpu_guard.sh --hold`.

The generator imports the pinned model modules without modifying their math.
It extracts three exact support definitions from upstream `base.py` to avoid
loading unrelated multimodal packages. Its wrapper only evaluates and times
model outputs; it delegates all predictions and stitching to upstream code.

## Recovered relative CPU oracle

The committed relative CPU oracle is an independently reviewed recovery, not a
clean wrapper run. The generator wrote the complete tensor and its legacy
17-key provenance, but the surrounding shell wrapper later exited with status
127. The Python process exit was not captured independently, and the original
completion log was lost. No terminal `PASS` line has been reconstructed or
claimed.

`relative_cpu_recovery/` preserves the inspection packet, wrapper failure
record, independent audit, and reviewed recovery decision byte for byte. The
fail-closed comparator accepts this one relative artifact only after checking
those exact hashes, all ordinary tensor and provenance requirements, and full
CPU-to-pinned-GPU parity. The metric CPU oracle continues to require the normal
completion-log path.

## Exact native runtime used for this run

The MLX Python binding was built against the tested Rust native core (MLX
0.32.5), rather than a different installed wheel. The isolated runtime's
provenance retains archive hashes and unchanged machine-code section checks.

- `libmlx.so` SHA-256:
  `9e38e9292550ea8eec9eb1877d06b72f5155ff382ac0e4ec1054818e15276094`
- Python binding SHA-256:
  `f8ec0f363bc6eab1cda644a2579ae43aa663939f30ace531d5fb8cc33c7090cc`

The local isolated runtime is
`/home/devops/Development/mlxcel/.cache/epic-2382/python-exact-core`.
Its `python` directory belongs on `PYTHONPATH`, and its root belongs on
`LD_LIBRARY_PATH`. This path is an execution detail, not a runtime dependency of
the Rust implementation or committed tests.

Regeneration fails before importing MLX if any cached model/source, checkpoint, native library/binding, MLX Python module, or clip hash differs from the committed manifests. It checks decoded pixels and processed input too. `--verify-only` checks provenance without loading weights. `MLX_ENABLE_TF32=0` is required and recorded; ROCm does not use the CUDA/Metal TF32 path. `--stage-timing` optionally evaluates unchanged upstream backbone/head outputs to diagnose CPU runtime.
