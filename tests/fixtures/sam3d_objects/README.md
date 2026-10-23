# SAM 3D Objects reference fixtures

The source revision, checkpoint file hashes, exact MLX runtime identities, and input hashes are pinned in `provenance.json`. The reference executes the unchanged upstream model at `abd3b2d7b728cb5372d4e092260da325e5ded209` against the complete `mlx-community/sam-3d-objects-bf16` bundle at `26f21f6170e56ddc2000a38619b70540ab7eca29`. Its default case uses seed 42, 25 structure steps, 25 latent steps, automatic MoGe depth, and complete Gaussian and mesh outputs. The RGBA case consumes the image alpha channel and disables automatic depth.

## External example image

`input-provenance.json` pins the public upstream Waterhole example and its mask, including the upstream commit, Git blob identifiers, complete file hashes, and deterministic Pillow 13.0.0 transforms. The photo's separate redistribution permission is not established here. Consequently the image files are downloaded to ignored local storage rather than redistributed in this repository.

```sh
python prepare_input.py --output /path/to/ignored/objects-inputs
```

This verifies both original files and all three resulting canonical PNGs. `object.png` is RGB 672×448; `mask.png` is its nearest-neighbor-resized alpha mask; `object-rgba.png` combines those exact canonical pixels. The original image uses Pillow LANCZOS resizing. Model preprocessing is unchanged.

## Runtime and generation

The authenticated native MLX revision is `bff10220c742907f0c68e55ff174aaea56f74e3d` (0.32.5). The Python binding was rebuilt from matching headers/source against the Rust build's native archive. The isolated library export changed ELF symbol visibility only; executable sections were verified identical. Native library SHA-256 is `9e38e9292550ea8eec9eb1877d06b72f5155ff382ac0e4ec1054818e15276094`; binding SHA-256 is `f8ec0f363bc6eab1cda644a2579ae43aa663939f30ace531d5fb8cc33c7090cc`. Set `PYTHONPATH` and `LD_LIBRARY_PATH` to that isolated runtime and set `MLX_ENABLE_TF32=0`. The generator verifies the actually mapped native library and imported binding before loading a model.

```sh
python generate_reference.py --source /path/to/pinned/mlx-vlm \
  --bundle /path/to/pinned/checkpoint --input-dir /path/to/ignored/objects-inputs \
  --output /path/to/new/reference --device gpu --case default
```

Use `--case rgba-dropout` for the alpha-mask/dropout route. `--imports-only --device cpu` validates identities and imports without constructing a model or loading parameters. Reference directories must be new. The generator records every stage, loaded dtype policy, shared-DINO verification, complete output schema, and hashes of executed helpers, consumed inputs, safetensors, OBJ, PLY, and automatic-depth point map.

For a monitored run, use `run_reference.py` with explicit `--fixtures`, `--source`, `--bundle`, `--input-dir`, `--run-dir`, `--python`, `--reviewed-manifest`, and `--reviewed-manifest-sha256`. Pass the independently reviewed SHA-256 of `reviewed-harness.json`; do not derive an approval from a changed candidate. The monitor authenticates and copies the five-file program before launching it, records the exact invocation, and preserves separate child/wrapper exit statuses. Run it under the repository GPU guard. It samples memory at 100 ms and stops only its own process group on recorded memory/swap pressure. Pressure-aborted output is not parity evidence.

The first default reference attempt reached the deferred depth-config import and failed before model construction because the import adapter did not expose upstream's processor-registration function. The adapter now executes that unchanged original function AST as well as `BaseModelConfig`; imports-only validation also exercises the published nested-depth configuration. The original failed run remains recorded as child/wrapper exit 1, with no numerical result.

Both complete upstream reference runs passed with separate child, wrapper, and GPU-guard exit status 0. `full-reference-manifest.json` binds the copied receipts and resource records to the retained external artifacts. The default case has 6,281 occupied coordinates, 200,992 Gaussians, 189,394 mesh vertices, and 376,174 faces; its full tensor SHA-256 is `70669a05f379bdc6a1755a72af2d60907cb104efecbdd43234a0723083b18db3`. The RGBA/dropout case has 5,774 coordinates, 184,768 Gaussians, 189,773 vertices, and 376,882 faces; its tensor SHA-256 is `74a03cc56bf10361a858dbee4f9882455e12eddd4aca8989b91cfe3d3fcb4ee3`. Both contain all 14 result tensors without truncation. The default point map separately retains upstream NaN support; the 14 final result tensors are finite.

The large tensors, OBJ, PLY, point map, and input photographs remain in ignored storage. The committed metadata does not replace those artifacts: comparison rehashes every required artifact. Rust parity and actual CLI acceptance remain pending.

## Small authentic oracles

```sh
python generate_small_oracles.py --source /path/to/pinned/mlx-vlm --output /path/to/new/small-oracles
python generate_helper_oracles.py --fixtures . --source /path/to/pinned/mlx-vlm --output /path/to/new/helper-oracles
```

`writers.json` records the unchanged upstream three-point binary PLY and triangle OBJ results. `sphere.json` records the complete upstream resolution-8 sphere FlexiCubes result (128 vertices and 252 faces). Their CPU runtime/source hashes are recorded in `small-oracle-provenance.json`.

`layers.safetensors` records actual BF16 compiled modulation, gated residual, head normalization, and GELU outputs on gfx1151. `layers-provenance.json` records the exact generator/runtime and tensor shapes. The three first helpers agree with eager evaluation for these bounded inputs; they are equivalence fixtures, not evidence that compilation changes their numerical values. The GELU diagnostic uses explicit F32 constants and therefore promotes its output. Even after round-to-nearest-even BF16 conversion, 3,857 of 8,192 values differ from the authentic output, with maximum absolute difference 0.015625.

## Frozen numerical comparison

`comparison-contract.json` was fixed before full model results. Coordinates must match exactly, including ordering. Full latent relative RMS must be below `1e-2`. Mesh vertex counts must differ by no more than 1%. Chamfer is the symmetric mean of both directed nearest-neighbor Euclidean distances over **all** vertices, divided by the reference mesh AABB diagonal; it must be below `1e-3`. No sampling, nonfinite filtering, or empty-output acceptance is allowed. Complete artifact and invocation identities must be authenticated before comparison.
