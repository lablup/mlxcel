# Image generation

mlxcel generates and edits images through `mlxcel generate-image` and the OpenAI-compatible `POST /v1/images/generations` and `POST /v1/images/edits` server routes. This page covers the model-independent runtime in `src/imagegen/`: the request types, the checkpoint probe, the scheduler, latent packing, the FLUX.2 VAE, the Z-Image DiT blocks, PNG output, the CLI, the server routes, and how a model issue registers a family.

**Current status.** Registered families: `flux2` ([FLUX.2 Klein](#flux2-klein-flux2)). Each model in the image-generation epic (`lablup/mlxcel#2348`) adds its transformer and text conditioner and one row to the family table. A checkpoint the probe recognizes but `FAMILIES` does not list fails with `image generation family <family> is not supported by this build`, and `mlxcel-server --image-model` fails at startup for it, by design (see [Server](#server)).

## Probe

`mlxcel::imagegen::detect_image_family(path)` classifies a checkpoint directory from small JSON files; it never reads weights. It is a port of `_local_image_model_types` in `Blaizzy/mlx-vlm` (`mlx_vlm/generate/image.py`). In order, it reads:

1. `manifest.json`: a manifest listing `transformer-packed-mflux/diffusion_pytorch_model.safetensors`, `text_encoder-mlx-4bit/model.safetensors` and `tokenizer/tokenizer.json` is `bonsai`. This is checked first because Bonsai ships a `Flux2KleinPipeline` `model_index.json`, which the class-name scan would call `flux2`.
2. `model_index.json` and `config.json`: `model_type`, `_class_name`, and the class names of the `transformer`, `vae`, `scheduler` and `text_encoder` components.
3. The transformer `weight_map` (`transformer/model.safetensors.index.json`, else `transformer/diffusion_pytorch_model.safetensors.index.json`), matched against marker key sets. A `z_image` transformer is `ming_image` when `mllm/config.json` exists.
4. Each `<dir>/config.json`.

The first candidate that names a family in `IMAGE_FAMILIES` wins: `bonsai`, `flux2`, `z_image`, `ming_image`, `ernie_image`, `qwen_image`. A path that does not exist, is not a directory, or carries no image metadata yields `None`; unparseable JSON is skipped.

`mlxcel generate` and `mlxcel serve -m` on an image checkpoint fail with `image generation checkpoint (<family>): use mlxcel generate-image` instead of `model_type not found` or a bare missing-`config.json` error. The hook sits in `src/models/detection.rs` and only fires when the directory has no top-level `model_type`, so no text checkpoint is affected.

## CLI

```bash
mlxcel generate-image -m <dir> -p "a watercolor fox"
mlxcel generate-image -m <dir> -p "a fox" --size 768x512 --seed 7 -n 2 -o fox.png
mlxcel generate-image -m <dir> --task edit --image fox.png -p "make it night"
```

| Flag | Meaning |
|------|---------|
| `-m, --model PATH` | Checkpoint directory (Diffusers layout). |
| `-p, --prompt TEXT` | Prompt, or the edit instruction. |
| `--negative-prompt TEXT` | Negative prompt; the model default applies when omitted. |
| `--task generate\|edit` | Default `generate`. |
| `--image PATH` | Reference image; repeat for several. Required for `edit`, rejected for `generate`. |
| `--size WxH` | Default `512x512` for generation. For editing, the default is the first reference image's size, or what the family derives from it (FLUX.2 scales it to at most 1 MP and rounds down to 16, as upstream). Both sides must be multiples of the family's grid (16 for most families). |
| `--steps N`, `--guidance F` | Model defaults apply when omitted. |
| `--seed N` | Default is a random 32-bit seed; the seed is always printed. |
| `-n N` | 1 to 10 images. Image `i` uses seed `seed + i`, except FLUX.2 generation (see [Batched seeds](#batched-seeds)). |
| `-o, --output PATH` | Default `outputs/image-<seed>.png`. The extension is forced to `.png`; with `-n` greater than one the names are `<stem>-01.png`, `<stem>-02.png`, and so on. |
| `--gen-kwargs JSON` | A JSON object of model-specific options, passed to the model as `extra`. |

Order of work: the model directory, the arguments and reference images, output directory writability, family classification, runtime initialization, then the model load. A wrong directory, an unported family, or an unwritable output directory therefore fails before any weight is loaded.

`mlxcel generate-image` has no `--output-modality image` counterpart on `generate`: that command's sampling, KV-cache and speculative flags mean nothing for diffusion, and `detect` already set the precedent of a separate subcommand.

## Server

```bash
mlxcel-server -m <chat model> --image-model <image checkpoint>
```

`--image-model` (also `MLXCEL_IMAGE_MODEL`, on both `mlxcel-server` and `mlxcel serve`) loads a side worker the way `--embedding-model` does: a dedicated thread owns the model, a bounded queue admits requests, a per-request timeout frees the caller. A load failure is fatal at startup, so a command line that asked for image routes never serves `501` for them. Requests run one at a time, since a diffusion run holds the GPU for seconds; a full queue is `503`, a timeout is `504`. The queue depth (8) and the timeout (900 s) are constants in `src/server/image_worker.rs`.

| Method | Path | Notes |
|--------|------|-------|
| POST | `/v1/images/generations` | Alias `/images/generations`. |
| POST | `/v1/images/edits` | Alias `/images/edits`. JSON body only. |

Without `--image-model` both routes answer `501`, naming the flag. A request the family rejects before running (`ImageGenerationModel::check_request`: FLUX.2's size range, an invalid `extra` option) is `400`; a failure during generation is `500`.

### Request

`generations`: `prompt`, `model`, `n` (1 to 10), `size` (`WxH`) or `width` plus `height`, `steps`, `seed`, `guidance`, `negative_prompt`, `response_format`, `output_format`. `edits` adds `image`, one string or a list. Any other field is passed to the model as `extra`.

- `width` and `height` come together or not at all (otherwise `400`) and override `size`.
- `size` defaults to `512x512` for generation and, for editing, to the first reference image's size or what the family derives from it (FLUX.2: at most 1 MP, rounded down to 16). Width and height must be multiples of the family's grid; the error names the multiple.
- `response_format` accepts only `b64_json`; `url` and `path` are `400`. `output_format` accepts only `png`. `output_path` and `output_dir` are `400`. The server never writes files on behalf of a client.
- `model`, when non-empty, must equal the served model id.
- Each `image` is a data URL or a URL, read through the same limits as chat images (`file://` URLs stay inside `--media-path`). A raw filesystem path is `400`. Multipart upload, OpenAI's edit wire format, is not served.

### Response

```json
{
  "created": 1790000000,
  "data": [
    {"b64_json": "iVBORw0KGgo...", "mime_type": "image/png",
     "width": 512, "height": 512, "seed": 7, "revised_prompt": null}
  ],
  "output_format": "png",
  "size": "512x512"
}
```

RGB and RGBA images are both encoded as they are; RGBA keeps its alpha channel.

## FLUX.2 Klein (`flux2`)

A port of `mlx_vlm/models/flux2` (`Blaizzy/mlx-vlm` at git 73d366ec), in `src/imagegen/flux2/`. Both tasks are supported.

| Checkpoint | Variant | Steps | Guidance | Notes |
|------------|---------|-------|----------|-------|
| `black-forest-labs/FLUX.2-klein-4B` | `flux2-klein-4b` | 4 | 1.0 | Diffusers tree, bf16. Qwen3-4B text encoder. |
| `black-forest-labs/FLUX.2-klein-9B` | `flux2-klein-9b` | 4 | 1.0 | Qwen3-8B text encoder. |
| `black-forest-labs/FLUX.2-klein-base-4B`, `-base-9B` | `flux2-klein-base-*` | 50 | 4.0 | Undistilled; guidance above 1 runs a second pass on the negative prompt (`" "` unless `--negative-prompt` is given). |
| `black-forest-labs/FLUX.2-klein-9b-kv` | `flux2-klein-9b-kv` | 4 | 1.0 | Edits cache the reference tokens' K/V on the first step; generation is the plain 9B path. |
| `mlx-community/flux2-klein-4b-4bit`, `-9b-4bit` | as the dense model | 4 | 1.0 | MLX affine 4-bit, group 64. No component `config.json`: the text-encoder config is synthesized from the variant, and the VAE's quantized mid-block attention projections are dequantized at load. |

The variant comes from the directory name, as upstream's `variant_from_local_path` (snapshot paths included), then `model_index.json`'s `is_distilled` and the transformer width. A missing `transformer/`, `vae/`, `text_encoder/` or `tokenizer/tokenizer.json` fails at load naming the path.

- **Sizes.** Width and height in `256..=2048`, multiples of 16. An edit that names no size (CLI and server) follows upstream `_resolve_edit_dimensions`: the first reference is scaled to at most 1024x1024 pixels of area (never up) and each side rounded down to a multiple of 16 within `256..=2048`, so a 4000x3000 photo edits at 1168x880 and a 1000x667 one at 992x656. Other families use the first reference's own size.
- **Prompt.** The Qwen chat template with `enable_thinking=false`, right-padded with id 151643, truncated to `max_sequence_length`; generation pads to the smallest bucket of 32, 64, 128, 256 or 512 that fits, edits pad to `max_sequence_length`. The embeddings stack Qwen3 layers 9, 18 and 27. The text encoder is evicted after each encode and prompts are cached (32 entries); the transformer and VAE load on the first request and stay resident.
- **Tiling.** Generation decodes in tiles of 128 pixels (16 latent cells) blended with a raised-cosine ramp over an 8-cell (64-pixel) overlap once either side reaches 256 pixels, as upstream's `auto` mode does, so every size this family accepts is tiled; edits decode untiled. Tiled and untiled decodes of the same 512x512 latents differ by 25.05 dB PSNR, in upstream and here alike.
- **`--gen-kwargs` / `extra`.** `max_sequence_length` (1 to 512, default 512) and `tiled_vae` (`true` or `false`; default as above).

### Batched seeds

FLUX.2 generation follows `Blaizzy/mlx-vlm#2355`: the `n` images share the prompt embeddings and come from one noise tensor of batch `n` drawn from `seed`, so every image reports `seed`, and image `i` is not the image `seed + i` gives alone. Edits run once per image with seed `seed + i`.

### Parity

Against upstream mlx-vlm 0.7.8 (git 73d366ec) on mlx 0.32.4, CUDA, NVIDIA GB10, `MLX_ENABLE_TF32=0`. The `#[ignore]` tests in `src/imagegen/flux2/parity_tests.rs` read the fixtures in `tests/fixtures/flux2/`, which `tests/fixtures/flux2/generate_flux2_fixtures.py` reproduces. `edit_reference.png` is an upstream FLUX.2-klein-4B generation (prompt "A studio photo portrait of a smiling man with short brown hair, plain grey background", seed 7, 512x512, 4 steps) written by that script, not a photo of a real person; the 4B checkpoint is Apache-2.0.

| Check | Result |
|-------|--------|
| Initial noise (generate seed 9909, edit seed 123) | bit-identical |
| Transformer in f32 against upstream in f32, same inputs | bit-identical |
| VAE decoding upstream's final latents, tiled / untiled | 57.59 / 58.45 dB |
| 4B generate 512x512, 4 steps, seed 9909 | 32.27 dB |
| 4B edit "Add black sunglasses", seed 123 | 40.54 dB |

The bf16 final latents differ from upstream's by 1.03e-1 (generate) and 4.11e-2 (edit) relative RMS. bf16 matmuls with long reductions (the 7680-wide context embedder first) round differently in this build's MLX and in mlx 0.32.4 on identical inputs and weights, and four Euler steps amplify that; upstream's own f32 run differs from its bf16 run by 7.1e-2 on the same case. Generation sits about 2 dB above the 30 dB gate, so an MLX pin change can move it.

Other checkpoints, PSNR against upstream on the same request: 9B edit 44.39 dB, 9b-kv edit 42.94 dB, 4B 4-bit generate 35.47 dB, base-4B generate at 12 steps 32.80 dB, 9B generate 25.78 dB and 9B 4-bit generate 20.70 dB (same composition, bf16 drift in fine detail). The 9b-kv checkpoint has its own weights, so its edit is not the 9B edit: upstream's two differ by 25.15 dB, this build's by 25.09 dB.

**Edit fixtures and upstream's encoder.** The edit fixture is generated with the VAE encoder's `conv_in` transposed, because upstream leaves that one tensor in PyTorch order (see [Verification](#verification)). Upstream as published therefore edits differently from this build.

## Shared building blocks

| Module | Contents | Used by |
|--------|----------|---------|
| `pipeline` | `ImageRequest`, `GeneratedImage`, `ImageSamplingDefaults`, the `ImageGenerationModel` trait. | every family |
| `probe` | `detect_image_family`. | every family |
| `registry` | `FAMILIES`, `ImageFamily`, `resolve_family`, `load_image_model`. | every family |
| `output` | `encode_png`, `to_b64_png`, `indexed_output_path`. | CLI, server |
| `run` | `validate_request`, `run_request`, `generate_to_files`, `parse_size`, `random_seed`. | CLI, server |
| `scheduler` | `FlowMatchEulerDiscrete::dynamic` (exponential shift, optional `shift_terminal`) and `::static_shift`. | QwenImage, MingImage; Z-Image, Mage-Flow |
| `latent` | `patchify_latents`, `unpatchify_latents`, `pack_latents`, `unpack_latents`, `prepare_packed_latents` (seeded noise, packed, with grid ids), `prepare_grid_ids`. | FLUX.2, Bonsai, ErnieImage, LLaDAImage |
| `vae` | `Flux2Vae`, `Flux2VaeConfig`: decode, packed-latent decode, optional encode, `bn_normalize` and `denormalize_packed_latents`. | FLUX.2, ErnieImage, LLaDAImage |
| `qwen3_text_encoder` | `Qwen3TextEncoder`: hidden states of every layer (`num_layers + 1`, index 0 the embeddings, before the final norm) and `stacked_prompt_embeds` (`[B, T, n * H]`, FLUX.2 order). Built on `Qwen3Model::forward_all_hidden`. | FLUX.2, Z-Image, Ideogram 4, Bonsai |
| `dit_blocks` | `TimestepEmbedder`, `FeedForward`, `MRoPE`, `apply_rotary`, `Attention`, `ZImageTransformerBlock`, `FinalLayer`, `sanitize_transformer_weights`. | LLaDAImage, MingImage |

Not shared: the FLUX.2 transformer, its empirical-`mu` scheduler and its tiled VAE decode live in `src/imagegen/flux2/`. Not served at all: the Qwen-Image VAE, multipart uploads, streaming partial images, video generation.

### Conventions a model must follow

- **Seeds.** Image `i` of `n` uses `seed + i` (FLUX.2 generation is the exception, see [Batched seeds](#batched-seeds)). mlxcel links the same MLX C++ RNG as upstream, so the same seeding call (`mx.random.seed(seed)` in Qwen-Image and LLaDA-Image, `mx.random.key(seed)` in ERNIE-Image and Ming-Image) followed by the same `normal` shape and dtype gives the same initial noise; keep upstream's draw order so whole-run parity tests can compare.
- **Threads.** MLX work is thread-affine. The server builds the model on the worker thread; `ImageGenerationModel` carries no `Send` bound.
- **Convolution weights.** Diffusers stores 4-D conv weights `(out, in, kH, kW)`; MLX expects `(out, kH, kW, in)`. Decide the layout once per checkpoint from a tensor whose two layouts differ in shape (`vae::model::detect_conv_layout`), and never permute an already-converted checkpoint twice. Deciding per tensor from its shape is wrong for any conv whose input channel count equals its kernel size; see the note under [Verification](#verification).
- **Half precision.** Norms run in f32 and cast back to the activation dtype. Use `fast_scaled_dot_product_attention`, not `scaled_dot_product_attention`: the latter multiplies by an f32 scalar and promotes half-precision scores to f32.

### Qwen3 conditioner

`Qwen3TextEncoder::load(dir)` reads `config.json` (top-level Qwen3 fields or a nested `text_config`) and the safetensors, accepts the `model.`, `model.language_model.` and `language_model.model.` key layouts, drops `lm_head.*` and every non-text tensor, and honors MLX affine `quantization`. `from_weights(config, weights)` takes an in-memory `WeightMap` (Ideogram 4 dequantizes FP8 first) and converts nothing. RoPE is standard: `rope_scaling` is ignored, since text positions make M-RoPE sections a no-op.

The mask is causal plus key padding, added in f32. Blocked positions use `-30000`, not upstream's `-inf`: identical for every row with a visible key, but a fully blocked row (left padding) becomes a uniform average instead of NaN, which would otherwise reach the real positions through the next layer.

Checked against upstream's `Qwen3TextEncoder` on a tiny model, and on `mlx-community/Qwen3-0.6B-4bit` the last hidden state, normed and projected, reproduces the generation path's logits exactly (an `#[ignore]` test). The change to `qwen3.rs` is one added method; teacher-forced logit traces (`examples/logit_trace`, 96 positions, 8 chunks of 8 tokens behind 256 tokens of context) before and after are byte-identical.

## Precision

Upstream runs the DiT, the VAE and the conditioning in bf16. The project rule converts non-quantized bf16 to f16 at each new loading boundary, and a component keeps bf16 only when a seed-fixed comparison shows inf/NaN or a failed threshold under f16.

| Component | Default dtype | Evidence |
|-----------|---------------|----------|
| FLUX.2 VAE | f16 (`Flux2Vae::default_dtype`, from `bf16_to_f16_at_load`) | Real ERNIE-Image-Turbo VAE, 64x64 decode of a fixed latent: finite, rel-RMS 1.2e-3 against upstream f32. Upstream's own bf16 run differs from upstream f32 by 2.0e-2, so f16 is closer to f32 than the bf16 upstream default. |
| Z-Image DiT blocks | follows the checkpoint dtype | The blocks are dtype-agnostic; a model issue records its own comparison. |
| FLUX.2 transformer | bf16 on every backend | With f16 weights (`MLXCEL_CUDA_F16_NORMALIZE=1` on GB10) the 4B generate case decodes NaN latents (10.2 dB). Edits run the transformer in f32 by promotion from the f32 reference latents, as upstream does. |
| FLUX.2 VAE (in FLUX.2) | `Flux2Vae::default_dtype` | bf16 on CUDA; under the f16 policy, VAE-only parity is 54.76 dB tiled and 52.37 dB untiled. |
| FLUX.2 Qwen3 text encoder | the Qwen3 loader's policy | f16 holds: with the f16 policy and a bf16 transformer, 4B generate is 34.47 dB and edit 38.54 dB. |

The measurement used the CPU MLX backend (see [Verification](#verification)).

## Registering a family

A model issue does the following; the runtime issue ships none of it.

1. Implement `ImageGenerationModel` in `src/imagegen/<family>/`: `family()` (the probe name), `supports(task)`, `defaults(task)` (steps and guidance, resolved from JSON metadata, never from weights), `size_multiple()`, and `generate(&ImageRequest)` returning `req.n` images with seeds `req.seed + i`.
2. Add the probe name to `IMAGE_FAMILIES` in `src/imagegen/probe.rs` if it is not there, with the marker keys or class names that identify it, and a fixture-based test in `probe_tests.rs`.
3. Add one row to `FAMILIES` in `src/imagegen/registry.rs`: `ImageFamily { name: "<family>", load: <constructor> }`. The constructor takes the checkpoint directory and runs on the thread that will run the model.
4. Add the family to `STANDALONE_ARCHITECTURE_FAMILIES` in `src/models/registry.rs` with `Runtime::ImageGenerate` and `OutputKind::Image`, so `mlxcel arch --json` lists it. A test there fails when the two tables disagree.
5. Extend this page and `docs/supported-models.md`.

A family the probe recognizes but `FAMILIES` does not list is reported as `image generation family <family> is not supported by this build`.

## Verification

```bash
cargo test --workspace --profile test-fast --features metal,accelerate imagegen
```

The tests compare against the upstream Python (`Blaizzy/mlx-vlm`) rather than against themselves. `tests/fixtures/generate_imagegen_fixtures.py` documents how the committed fixtures were produced and reproduces them exactly:

- **Scheduler**: sigma and timestep arrays for 1, 4, 12 and 30 steps at image sequence lengths 1024 and 4096, with `shift_terminal` 0.02 and disabled, plus the static-shift schedule.
- **VAE, tiny**: a random-weight FLUX.2 VAE (Diffusers key names, PyTorch conv layout) with upstream's decode, packed decode and encode outputs; the same weights re-laid in MLX layout must give the same result.
- **VAE, real checkpoint** (`#[ignore]`): `baidu/ERNIE-Image-Turbo`'s `vae/` against upstream f32 outputs. It skips when `models/ERNIE-Image-Turbo/vae/` is absent:

  ```bash
  huggingface-cli download baidu/ERNIE-Image-Turbo --include "vae/*" "model_index.json" \
      --local-dir models/ERNIE-Image-Turbo
  cargo test --profile test-fast --features metal,accelerate -p mlxcel --lib -- \
      --ignored ernie_image_turbo
  ```

  Measured against upstream f32: decode 3.4e-6, packed decode 1.2e-6, encode 3.0e-6 (relative RMS).
- **Qwen3 conditioner**: hidden states and stacked prompt embeds of a tiny model against upstream, padding and key-layout invariants, and the real-checkpoint consistency test above.
- **DiT blocks**: upstream outputs for the timestep embedder, feed-forward, MRoPE tables, `apply_rotary`, the modulated and plain blocks (with and without an attention mask) and the final layer, with weights and inputs derived from a closed-form generator that both sides implement.

**An upstream defect the encoder fixture works around.** Upstream's `_match_conv_layout` returns a weight unchanged when its shape already equals the target. The encoder's RGB `conv_in` is `(128, 3, 3, 3)` in both layouts, so upstream keeps the PyTorch data order there and its encoder is wrong: encode followed by decode of a smooth image reconstructs with relative RMS 0.96, against 0.096 once the tensor is transposed. The fixture is generated with that one tensor transposed, and the real-checkpoint test also asserts the encode-decode reconstruction. This build is not affected because the layout is decided per checkpoint.

**What was and was not run.** The numbers above come from the CPU MLX backend (`MLXCEL_BUILD_METAL=OFF`), because the Metal toolchain was not installed on the machine that produced them. Metal-backend runs of these tests are not recorded here.

### Resource and queue limits

The shared request validator limits each output image to 16,777,216 pixels and explicit step counts to 1–1,000. These checks apply to both the CLI and HTTP routes before model execution. The image worker includes queue waiting in its request timeout and discards expired queued requests before starting GPU work. A generation already executing cannot be interrupted through the current model trait; it finishes on its owning worker thread.

The CLI checks output-directory writability using an exclusively created temporary file, including when the output is a bare filename in the current directory. Existing files and symbolic links are never used as write probes.
