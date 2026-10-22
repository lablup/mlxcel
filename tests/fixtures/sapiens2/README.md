# Sapiens2 parity fixtures

`person.jpg` is Ultralytics' `zidane.jpg`, downloaded from `https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg`. Its SHA-256 is `16d73869e3267a7d4ed00de8e860833bd1657c1b252e94c0c348277adc7b6edb`.

The numerical references come from the Sapiens2 implementation at mlx-vlm revision `abd3b2d7b728cb5372d4e092260da325e5ded209`. `provenance.json` pins the source hashes and reference runtime. `checkpoint-hashes.json` pins every public checkpoint file used for validation. The reference environment used MLX 0.32.4 on CPU, retained the original numerical sources, and removed only imports needed for unrelated auto-processor registration and caches. `prepare_reference.py` reproduces that isolated package and records the fetched source hashes.

Generate a reference from a checkout of that revision:

```bash
python tests/fixtures/sapiens2/prepare_reference.py /tmp/sapiens2-reference \
  --legacy-seg-config tests/fixtures/sapiens2/seg-0.4b-config.json
PYTHONPATH=/tmp/sapiens2-reference/reference python tests/fixtures/sapiens2/generate_reference.py \
  /path/to/sapiens2-pose-0.4b-bf16 \
  tests/fixtures/sapiens2/person.jpg \
  /tmp/pose-bf16-flip0.json

PYTHONPATH=/tmp/sapiens2-reference/reference python tests/fixtures/sapiens2/generate_reference.py \
  /path/to/sapiens2-seg-0.4b-bf16 \
  tests/fixtures/sapiens2/person.jpg \
  /tmp/seg-bf16.png
```

Pass `--flip` for the default flip-test path. Early public segmentation checkpoints use a nested config that the cited mlx-vlm revision cannot instantiate directly. The generator canonicalizes only those config field names in memory and leaves the model, weights, preprocessing, and output arithmetic unchanged.

The output suffix selects lossless JSON, NPZ, or an 8-bit segmentation PNG. `pose-bf16-flip0.json` is the full authentic upstream output for the fixed image and full-image box. `seg-bf16.png` losslessly stores the authentic 720x1280 class IDs from the public legacy-schema segmentation checkpoint; its SHA-256 is `1dcd51d0a5828ed8efec0832989149083ec8ce7f44ecf6f7443e7f12e840e8ca`. The ignored real-model tests load the published checkpoints, retain the published BF16 dtype required for Sapiens2 numerical fidelity, enforce the issue's pose and segmentation bounds, and smoke-test finite outputs and source-resolution shapes for the three other heads.

After a CLI run, `compare_reference.py` compares `result.safetensors` with a JSON pose reference, a lossless segmentation PNG, or an upstream NPZ for the remaining dense tasks. It exits unsuccessfully when pose exceeds 1.5 pixels, segmentation drops below 99.5%, or normal/pointmap relative RMS is not below `1e-2`.
