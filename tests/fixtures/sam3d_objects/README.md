# SAM 3D Objects reference fixtures

The source revision, checkpoint file hashes, exact MLX runtime identities, and input hashes are pinned in `provenance.json`. The reference executes the unchanged upstream model at `abd3b2d7b728cb5372d4e092260da325e5ded209` against the complete `mlx-community/sam-3d-objects-bf16` bundle at `26f21f6170e56ddc2000a38619b70540ab7eca29`. Its default case uses seed 42, 25 structure steps, 25 latent steps, automatic MoGe depth, and complete Gaussian and mesh outputs. The RGBA case consumes the image alpha channel and disables automatic depth.

## External example image

`input-provenance.json` pins the public upstream Waterhole example and its mask, including the upstream commit, Git blob identifiers, complete file hashes, and deterministic Pillow 13.0.0 transforms. The photo's separate redistribution permission is not established here. Consequently the image files are downloaded to ignored local storage rather than redistributed in this repository.

```sh
python prepare_input.py --output /path/to/ignored/objects-inputs
```

This verifies both original files and all three resulting canonical PNGs. `object.png` is RGB 672×448; `mask.png` is its nearest-neighbor-resized alpha mask; `object-rgba.png` combines those exact canonical pixels. The original image uses Pillow LANCZOS resizing. Model preprocessing is unchanged.

## Runtime and generation

The authenticated native MLX revision is `bff10220c742907f0c68e55ff174aaea56f74e3d` (0.32.5). The Python binding was rebuilt from matching headers/source against the Rust build's native archive. The native build includes the tracked forward LayerNorm read-completion barrier from commit `edd9e7fd5ed8dc592a320e723e3a655627990247` (file SHA `b6c0e849f3df710038e561056bba2e6351865e516ec81c7f56230a59e2aca723`). This prevents reuse of shared reduction storage before every thread snapshots the mean. The isolated export changes ELF symbol visibility only relative to those corrected native archives; executable sections were verified identical. The original uncorrected runtime and its diagnostic results remain preserved separately. Native library SHA-256 is `c1d8203b257364271e1ea6298705dd3bb0c82a7f06634f5578a209883601a9fc`; binding SHA-256 is `f8ec0f363bc6eab1cda644a2579ae43aa663939f30ace531d5fb8cc33c7090cc`. Set `PYTHONPATH` and `LD_LIBRARY_PATH` to that isolated runtime and set `MLX_ENABLE_TF32=0`. The generator verifies the actually mapped native library and imported binding before loading a model.

```sh
python generate_reference.py --source /path/to/pinned/mlx-vlm \
  --bundle /path/to/pinned/checkpoint --input-dir /path/to/ignored/objects-inputs \
  --output /path/to/new/reference --device gpu --case default
```

Use `--case rgba-dropout` for the alpha-mask/dropout route. `--imports-only --device cpu` validates identities and imports without constructing a model or loading parameters. Reference directories must be new. The generator invokes unchanged public `Pipeline.generate`, which evaluates only the complete event. It records loaded dtype policy, shared-DINO verification, all 14 final outputs, and hashes of executed helpers, consumed inputs, safetensors, OBJ and PLY. The public API does not expose the intermediate automatic-depth point map, so that artifact is not claimed by the canonical receipt.

For a monitored run, use `run_reference.py` with explicit `--fixtures`, `--source`, `--bundle`, `--input-dir`, `--run-dir`, `--python`, `--reviewed-manifest`, and `--reviewed-manifest-sha256`. Pass the independently reviewed SHA-256 of `reviewed-harness.json`; do not derive an approval from a changed candidate. The monitor authenticates and copies the five-file program before launching it, records the exact invocation, and preserves separate child/wrapper exit statuses. Run it under the repository GPU guard. It samples memory at 100 ms and stops only its own process group on recorded memory/swap pressure. Pressure-aborted output is not parity evidence.

The first default reference attempt reached the deferred depth-config import and failed before model construction because the import adapter did not expose upstream's processor-registration function. The adapter now executes that unchanged original function AST as well as `BaseModelConfig`; imports-only validation also exercises the published nested-depth configuration. The original failed run remains recorded as child/wrapper exit 1, with no numerical result.

The canonical full references completed with child, wrapper and GPU-guard status 0. `full-reference-manifest.json` binds their copied receipts/resource records to external artifacts. Repeats preserve coordinates, latents, pose, Gaussians and topology exactly; mesh vertex/color floating values are tolerance-stable rather than claimed bit-identical.

The canonical default reference contains 6,307 coordinates, 201,824 Gaussians, 189,623 vertices and 376,618 faces. Its full tensor SHA-256 is `7c327b4352842d187db7178fbcc70522251dfc1be64fc735fe532966386c14ed`.

The canonical rgba-dropout reference contains 5,793 coordinates, 185,376 Gaussians, 188,741 vertices and 374,760 faces. Its full tensor SHA-256 is `8d8cc32b5a5d0f223d6bce474e7f90ff4eff11cf6c28c58eeaacd230393eecaf`.

`historical-stream-*` metadata preserves the earlier custom stream consumer, which explicitly evaluated every event. That evaluation policy differs from public `Pipeline.generate` and is not the canonical acceptance reference. Original-runtime repetitions also exposed unstable outputs; the original failures and old native runtime are retained. The LayerNorm shared-memory race is proven from source; the bounded tiny probe did not reproduce it, so it is not presented as empirical causal proof for every earlier full-model difference.

The large tensors, OBJ, PLY and photographs remain in ignored storage. The metadata does not replace those artifacts: comparison rehashes the complete required files. `acceptance-manifest.json` binds both full Rust parity reports and the successful literal default CLI capture to their immutable source/binary identities, original full receipt/report hashes, complete output hashes, and copied process statuses. The full external receipts retain their complete source inventories; this compact index does not replace them.

## Conditioning diagnostics

```sh
python generate_stages.py --source /path/to/pinned/mlx-vlm \
  --bundle /path/to/pinned/checkpoint --input-dir /path/to/ignored/objects-inputs \
  --output /path/to/new/stages --device gpu --case rgba-dropout
```

Use the authenticated runtime, disabled TF32, and exclusive GPU guard described above. This diagnostic captures 39 tensors from unchanged upstream preprocessing, the first DINO block, cached DINO features, complete structure conditioning, seeded initial latents, and the first guided flow update. It preserves BF16/F32 dtypes and records its own executed filename and hash. These intermediate tensors locate divergence; their materialization boundaries can differ from unobserved sampling. They do not establish unmodified trajectory equivalence or replace full-default parity and actual CLI gates.

## Small authentic oracles

```sh
python generate_small_oracles.py --source /path/to/pinned/mlx-vlm --output /path/to/new/small-oracles
python generate_helper_oracles.py --fixtures . --source /path/to/pinned/mlx-vlm --output /path/to/new/helper-oracles
```

`writers.json` records the unchanged upstream three-point binary PLY and triangle OBJ results. `sphere.json` records the complete upstream resolution-8 sphere FlexiCubes result (128 vertices and 252 faces). Their CPU runtime/source hashes are recorded in `small-oracle-provenance.json`.

`layers.safetensors` records actual BF16 compiled modulation, gated residual, head normalization, and GELU outputs on gfx1151. `layers-provenance.json` records the exact generator/runtime and tensor shapes. The three first helpers agree with eager evaluation for these bounded inputs; they are equivalence fixtures, not evidence that compilation changes their numerical values. The GELU diagnostic uses explicit F32 constants and therefore promotes its output. Even after round-to-nearest-even BF16 conversion, 3,857 of 8,192 values differ from the authentic output, with maximum absolute difference 0.015625.

`gelu-tanh-power.safetensors` preserves the authentic compiled `nn.GELU(approx="tanh")` outputs and distinguishing repeated-multiply controls. The Power graph matches upstream exactly; the previous graph differs at 32 BF16 and 55 F32 grid elements on gfx1151. The separate CPU capture has identical BF16 arrays but backend-specific F32 results (one old-graph difference); tests select the matching CPU or gfx1151 fixture and omit only the F32 hardware oracle on other GPUs. The original receipt, generator, log, and exit are retained. `gelu-tanh-power-reproduction.json` records the separately hashed portable generator, which accepts explicit source and fresh output paths without changing the captured math.

```sh
python generate_gelu_tanh_power_oracle.py --source /path/to/pinned/mlx-vlm \
  --output-stem /path/to/new/gelu-tanh-power --device gpu
```

`resize-policy-provenance.json` contains the complete tiny F32 inputs and outputs for ratio-first nearest resize (84 to 518), width-first and height-first bilinear resize, and the adaptive-cost tie. The authenticated CPU and gfx1151 runs produced byte-identical tensor files. `generate_resize_policy_oracle.py` uses the same authenticated math with portable source/output arguments and sibling provenance helpers. `resize-policy-generator.json` distinguishes the original capture script hash from the adapted reproduction script hash; the original receipt is unchanged. The Objects-only policy preserves the existing interpolation defaults used by other model families.

```sh
python generate_resize_policy_oracle.py --source /path/to/pinned/mlx-vlm \
  --output-stem /path/to/new/resize-policy --device gpu
```

## Frozen numerical comparison

`comparison-contract.json` was fixed before full model results. Coordinates must match exactly, including ordering. Full latent relative RMS must be below `1e-2`. Mesh vertex counts must differ by no more than 1%. Chamfer is the symmetric mean of both directed nearest-neighbor Euclidean distances over **all** vertices, divided by the reference mesh AABB diagonal; it must be below `1e-3`. No sampling, nonfinite filtering, or empty-output acceptance is allowed. Complete artifact and invocation identities must be authenticated before comparison.

## Focal solver oracle

`focal-solver-inputs.safetensors` contains the authenticated 64×64 sampled UV/XYZ inputs from the exact saved affine point map. The GPU and CPU state fixtures record eager initialization and all 30 unchanged compiled LM updates, plus the production focal/shift output. CPU and gfx1151 reduction rounding differ, so each backend uses its own exact oracle. The original capture receipts remain unchanged; the portable generator below consumes only the committed small input and pinned upstream source, with no checkpoint or full model load.

```sh
python generate_focal_solver_oracle.py --source /path/to/pinned/mlx-vlm \
  --inputs focal-solver-inputs.safetensors --output /path/to/new/focal-gpu --device gpu
```

Use `--device cpu` for the CPU oracle. Regeneration reproduced GPU state SHA `2c3c8afc0eb3c6a41687ceb2eed37ef510b69a5c694e5739fc6b9b86434f031b` and CPU state SHA `cf5bf5c2eb1826242631177469e95ed2408872676c0d86c843283234f89bc3c5` exactly. The production correction is confined to Objects with free focal estimation; ordinary MoGe and fixed-focal behavior are preserved.

## Depth resize oracle

`resize-depth-policy-provenance.json` retains the original tiny bicubic scale-factor and non-antialiased bilinear capture. The portable generator below reproduces both tensor payloads exactly on CPU; `resize-depth-reproduction.json` binds the original receipt and new generator independently.

```sh
python generate_resize_depth_oracle.py --source /path/to/pinned/mlx-vlm \
  --output /path/to/new/resize-depth.json --device cpu
```

The preserved `historical-stream-*` files are diagnostic history, not canonical acceptance references. `historical-stream-index.json` maps their original filenames to retained names without rewriting original receipts.

## Completed acceptance

The default full capture on source `6bd85cf38e70ea43f2ba2e5a63363c35193c2db7` matches all 6,307 ordered coordinates and latent values exactly, has exactly 189,623 vertices, and has normalized full-mesh Chamfer `5.141699913725434e-6`. The literal default CLI on the same source emits all five required files and passes the same gates, with Chamfer `5.141710000502124e-6`. The earlier accepted RGBA/dropout capture matches all 5,793 ordered coordinates and latent values; its Chamfer is `4.450226386874762e-6`. Later production changes affect only enabled automatic depth, so that independently reviewed dropout evidence retains its original source/binary identity rather than being relabeled as a new run.

Export representation and pose diagnostics remain separate from the specified acceptance thresholds. OBJ rounds coordinates and exposes RGB; PLY uses log-scales and logit opacity. Vertex and face ordering are not claimed identical. All comparisons use complete arrays without sampling or filtering. `cli-interop.json` records successful independent trimesh and plyfile loading of the actual CLI OBJ/PLY, without a rendering claim.
