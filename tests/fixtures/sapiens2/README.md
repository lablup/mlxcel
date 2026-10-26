# Sapiens2 parity fixtures

`person.jpg` is Ultralytics' `zidane.jpg`, downloaded from `https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg`. Its SHA-256 is `16d73869e3267a7d4ed00de8e860833bd1657c1b252e94c0c348277adc7b6edb`. `person.png` is the lossless canonical input obtained by decoding that JPEG to RGB with Pillow and saving it as PNG; its SHA-256 is `3342393fccc820a687ec932d9e1b799e74146c0d9319397775169ae5b5f98a69`. The canonical PNG removes JPEG-decoder rounding differences between Pillow and Rust's `image` crate while preserving the exact RGB array used for the authentic upstream references.

The numerical references come from the Sapiens2 implementation at mlx-vlm revision `abd3b2d7b728cb5372d4e092260da325e5ded209`. `provenance.json` pins the source hashes and MLX 0.32.4 CPU runtime. The primary GPU references use MLX 0.32.5 at the exact Rust native pin `bff10220c742907f0c68e55ff174aaea56f74e3d`; `rocm-provenance.json` records the isolated Python binding and native-library hashes, the gfx1151 device, and the ELF visibility-only relink used to expose the binding symbols without changing executable code. `a7d-rocm-provenance.json` retains the earlier MLX `a7d4c85a` GPU run as cross-version diagnostic history. `checkpoint-hashes.json` pins every public checkpoint file used for validation, and `reference-output-hashes.json` pins the full external outputs. All reference environments retained the original numerical sources and removed only imports needed for unrelated auto-processor registration and caches. `prepare_reference.py` reproduces the isolated model package and records the fetched source hashes.

Generate a reference from a checkout of that revision:

```bash
python tests/fixtures/sapiens2/prepare_reference.py /tmp/sapiens2-reference \
  --legacy-seg-config tests/fixtures/sapiens2/seg-0.4b-config.json
PYTHONPATH=/tmp/sapiens2-reference/reference python tests/fixtures/sapiens2/generate_reference.py \
  /path/to/sapiens2-pose-0.4b-bf16 \
  tests/fixtures/sapiens2/person.png \
  /tmp/pose-bf16-flip0.json --device cpu

PYTHONPATH=/tmp/sapiens2-reference/reference python tests/fixtures/sapiens2/generate_reference.py \
  /path/to/sapiens2-seg-0.4b-bf16 \
  tests/fixtures/sapiens2/person.png \
  /tmp/seg-bf16.png

PYTHONPATH=/tmp/sapiens2-reference/reference python tests/fixtures/sapiens2/generate_reference.py \
  /path/to/sapiens2-normal-0.4b-bf16 \
  tests/fixtures/sapiens2/person.png \
  /tmp/normal-bf16.safetensors

PYTHONPATH=/tmp/sapiens2-reference/reference python tests/fixtures/sapiens2/generate_reference.py \
  /path/to/sapiens2-pose-0.4b-bf16 \
  tests/fixtures/sapiens2/person.png \
  /tmp/preprocess-pose.safetensors \
  --preprocess-only
```

Pass `--flip` for the default flip-test path. For the primary GPU fixtures, run the same command with `--device gpu` while `PYTHONPATH` selects both the isolated mlx-vlm package and exact-pin MLX Python binding, and `LD_LIBRARY_PATH` selects the native library whose hashes appear in `rocm-provenance.json`. The validation host used `PYTHONDONTWRITEBYTECODE=1`, binding root `/home/devops/Development/mlxcel/.cache/epic-2382/python-exact-core`, and `/opt/rocm/lib`. Model weights are sanitized and loaded with `strict=True`. Early public segmentation checkpoints use a nested config that the cited mlx-vlm revision cannot instantiate directly. The generator canonicalizes only those config field names in memory and leaves the model, weights, preprocessing, and output arithmetic unchanged.

The output suffix selects lossless JSON, NPZ, safetensors, or an 8-bit segmentation PNG. The unqualified `pose-bf16-*.json` and `seg-bf16.png` fixtures are the authentic CPU outputs. The stable `pose-rocm-bf16-*.json` and `seg-rocm-bf16.png` fixtures are the authentic exact-pin same-hardware ROCm outputs used by the GPU real-model tests; their SHA-256 values are `28ea2fbdfbcda6206dc49ab1dbce1398d39b8c5bc456fd5cbca1c3580e8297c3`, `c15f65694195a0c1f60d37ae7b9e2e9e5b532a2d06247dc7b1d9becad4d2c9f0`, and `c39b11062ff12435b180c46ba5a8b45f1304e0af1f72f18c8d4b1810b19a41c7`. Both pose fixtures contain all 308 points without confidence filtering. Full exact-pin ROCm normal, point-map, and matting safetensors references have SHA-256 values `ab9fdd20f1d0ebcd92b54f8486fc00756b18405497a1a9015402eb7d1559ce45`, `f39deabf0e6141d133de8cbbaf45436bc9f4b067fa20d34058734287f9a7ed14`, and `e6677044316294248f768102ed976091b57ad6fdad48744f2aab8616cb21cb7c`. The ignored real-model tests load the published checkpoints, retain the published BF16 dtype required for Sapiens2 numerical fidelity, enforce the issue's pose, segmentation, normal, and point-map bounds, and smoke-test finite matting outputs.

The fixed model and image legitimately expose backend-sensitive BF16 rounding: unchanged upstream CPU and ROCm pose outputs differ by up to 391.8495 pixels, with 16 of 308 points above 1.5 pixels. `pose-upstream-cpu-rocm-drift.json` records those unfiltered metrics. Decoding the ROCm heatmaps with the CPU DARK implementation changes coordinates by at most 0.01136 pixels (`pose-rocm-heatmap-cpu-decode-drift.json`), which isolates the larger drift to model execution. Tests therefore compare GPU execution with the exact-pin ROCm oracle while retaining the CPU and `a7d` GPU outputs as reproducibility, cross-backend, and cross-version evidence.

`bf16-rope-oracle.json` is a two-value sample from the authentic block-zero key tensor that distinguishes upstream's compiled packed RoPE graph from the former eager graph. In the exact-pin ROCm GPU environment, reproduce it with `python tests/fixtures/sapiens2/generate_rope_oracle.py /tmp/bf16-rope-oracle.json`; the generator asserts that compiled and eager results differ before writing the deterministic fixture. Metal and the CPU stream produce the same value for both graphs (upstream `mlx` 0.32.4 on an Apple M1 Ultra gives -4.589557647705078e-6), so the generator refuses there and the Rust test expects that value outside ROCm.

`bf16-deconv-layout-oracle.json` deterministically distinguishes the upstream kernel-4 deconvolution weight graph from the former value-identical concatenation graph on ROCm. The graph layout selects different BF16 accumulation for the real head's large GEMM. In the same exact-pin GPU environment, reproduce it with `python tests/fixtures/sapiens2/generate_deconv_layout_oracle.py /tmp/bf16-deconv-layout-oracle.json`; the generator asserts that the two graphs differ before writing the fixture.

`bf16-silu-layout-oracle.json` deterministically distinguishes upstream's shapeless compiled SiLU from the value-identical eager graph through a downstream BF16 projection on ROCm. In the same exact-pin GPU environment, reproduce it with `python tests/fixtures/sapiens2/generate_silu_layout_oracle.py /tmp/bf16-silu-layout-oracle.json`; the generator verifies that activation values agree while their downstream projections differ.

`--preprocess-only` writes the complete normalized `pixel_values` tensor without loading model weights. The pose/stretch and normal/point-map padded tensors for the fixed image have SHA-256 values `c615662943b5863b1a2c465056365601a70431bc26dbb2c907423bcf9905645c` and `fd0b9781d63af7eadf70fa0a6f798fa893f3ea68dd97d918c10383a7f76cddb5`, respectively.

After a CLI run, `compare_reference.py` compares `result.safetensors` with a JSON pose reference, a lossless segmentation PNG, or an upstream NPZ for the remaining dense tasks. It exits unsuccessfully when pose exceeds 1.5 pixels, segmentation drops below 99.5%, or normal/pointmap relative RMS is not below `1e-2`.
