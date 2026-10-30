# Image generation

mlxcel generates and edits images through `mlxcel generate-image` and the OpenAI-compatible `POST /v1/images/generations` and `POST /v1/images/edits` server routes. This page covers the model-independent runtime in `src/imagegen/`: the request types, the checkpoint probe, the scheduler, latent packing, the FLUX.2 VAE, the Z-Image DiT blocks, PNG output, the CLI, the server routes, and how a model issue registers a family.

**Current status.** Registered families: `flux2` ([FLUX.2 Klein](#flux2-klein-flux2)), `z_image` ([Z-Image](#z-image)), `bonsai` ([Prism Bonsai](#prism-bonsai-bonsai)), `ernie_image` ([ERNIE-Image](#ernie-image-ernie_image)), `mage_flow` ([Mage-Flow](#mage-flow-mage_flow)), `qwen_image` ([Qwen-Image](#qwen-image-qwen_image)), `ideogram4` ([Ideogram 4](#ideogram-4-ideogram4)), `ming_image` ([Ming-Image](#ming-image-ming_image)) and `llada_image` ([LLaDA-Image](#llada-image-llada_image)). Each model in the image-generation epic (`lablup/mlxcel#2348`) adds its transformer and text conditioner and one row to the family table. A checkpoint the probe recognizes but `FAMILIES` does not list fails with `image generation family <family> is not supported by this build`, and `mlxcel-server --image-model` fails at startup for it, by design (see [Server](#server)).

| Family | Checkpoints | Tasks | Default steps / guidance |
|--------|-------------|-------|--------------------------|
| `flux2` | FLUX.2 Klein 4B, 9B, base-4B, base-9B, 9b-kv (Diffusers tree) and the mlx-community 4-bit conversions | generate, edit (reference images) | distilled 4 / 1.0; base 50 / 4.0 |
| `z_image` | Tongyi-MAI Z-Image-Turbo, Z-Image (raw Diffusers tree or an mlx-vlm conversion) | generate, edit (img2img) | Turbo 8 / 0; base 50 / 4 |
| `bonsai` | Prism Bonsai ternary 2-bit FLUX.2 Klein 4B (`prism-ml/bonsai-image-ternary-4B-mlx-2bit`) | generate | 4 / 1.0 |
| `ernie_image` | Baidu ERNIE-Image-Turbo, ERNIE-Image (raw Diffusers tree) | generate, edit (SDEdit img2img) | Turbo 8 / 1.0 (edit 8 / 3.0); base 50 / 4.0 |
| `mage_flow` | Mage-Flow, Mage-Flow-Base, Mage-Flow-Turbo, Mage-Flow-Edit, Mage-Flow-Edit-Base, Mage-Flow-Edit-Turbo (mlx-vlm conversions) | generate or edit (one or more references), fixed by the variant | Turbo 4 / 1.0; Mage-Flow 20 / 5.0; Base and Edit 30 / 5.0 |
| `qwen_image` | Qwen-Image-2.1 (`Qwen/Qwen-Image-2.1`, Diffusers tree) | generate, edit (one or more references, RGBA kept) | generate 30 / 1.0; edit 40 / 1.0 |
| `ideogram4` | `ideogram-ai/ideogram-4-fp8` (license-gated, per-row FP8) | generate | `V4_DEFAULT_20`: 20 / 7.0 (3.0 for the last two steps) |
| `ming_image` | Ming-Image-0.1-Design (`inclusionAI/Ming-Image-0.1-Design` tree, or the `nativ-community` MLX 8-bit and 4-bit conversions) | generate (RGBA output) | 12 / 1.0 (guidance fixed) |
| `llada_image` | inclusionAI LLaDA-Image-Turbo, LLaDA-Image (Diffusers bf16 tree) | generate (text or VQ prior), edit (one source image) | Turbo 4 / 1.0; base 50 / 5.0 |

## Probe

`mlxcel::imagegen::detect_image_family(path)` classifies a checkpoint directory from small JSON files; it never reads weights. It is a port of `_local_image_model_types` in `Blaizzy/mlx-vlm` (`mlx_vlm/generate/image.py`). In order, it reads:

1. `manifest.json`: a manifest listing `transformer-packed-mflux/diffusion_pytorch_model.safetensors`, `text_encoder-mlx-4bit/model.safetensors` and `tokenizer/tokenizer.json` is `bonsai`. This is checked first because Bonsai ships a `Flux2KleinPipeline` `model_index.json`, which the class-name scan would call `flux2`.
2. `model_index.json` and `config.json`: `model_type`, `_class_name`, and the class names of the `transformer`, `vae`, `scheduler` and `text_encoder` components.
3. The transformer `weight_map` (`transformer/model.safetensors.index.json`, else `transformer/diffusion_pytorch_model.safetensors.index.json`), matched against marker key sets. A `z_image` transformer is `ming_image` when `mllm/config.json` exists.
4. Each `<dir>/config.json`.

The first candidate that names a family in `IMAGE_FAMILIES` wins: `bonsai`, `flux2`, `z_image`, `ming_image`, `ernie_image`, `qwen_image`, `mage_flow`, `ideogram4`, `llada_image`. Candidates come in the order above, so `Ideogram4Pipeline` in `model_index.json` wins over the FLUX.2 VAE class in `vae/config.json`. A path that does not exist, is not a directory, or carries no image metadata yields `None`; unparseable JSON is skipped.

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

When the model rewrote the prompt (ERNIE-Image's prompt enhancer), the CLI prints `revised prompt: <text>` after the line naming each saved image.

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

Without `--image-model` both routes answer `501`, naming the flag. A request the family rejects before running (`ImageGenerationModel::check_request`: FLUX.2's size range, an invalid `extra` option) is `400`, and so is an `InvalidImageRequest` a model raises while running (Z-Image's options, strength and reference count; ERNIE-Image's options, `image_strength`, reference count and an enhancer request on a checkpoint without `pe/`); any other failure during generation is `500`.

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

`revised_prompt` is always present. It is `null` unless the model rewrote the prompt, which today ERNIE-Image's prompt enhancer and Ideogram 4's plain-text caption wrapping do.

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

FLUX.2 generation follows `Blaizzy/mlx-vlm#2355`: the `n` images share the prompt embeddings and come from one noise tensor of batch `n` drawn from `seed`, so every image reports `seed`, and image `i` is not the image `seed + i` gives alone. Edits run once per image with seed `seed + i`. The VAE then decodes each image of the batch on its own, with the tiling a single image gets; upstream decodes the batch whole and untiled, which at 2048x2048 needs about 15 GiB per image.

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

## Prism Bonsai (`bonsai`)

A port of `mlx_vlm/models/bonsai` (`Blaizzy/mlx-vlm` at git 73d366ec), in `src/imagegen/bonsai/`, for `prism-ml/bonsai-image-ternary-4B-mlx-2bit` (Apache-2.0): FLUX.2 Klein 4B with ternary transformer linears packed as 2-bit MLX affine weights. Generation only; an edit request is rejected as unsupported. Defaults: 4 steps, guidance 1.0. `--gen-kwargs` takes FLUX.2's `max_sequence_length` and `tiled_vae`.

The generate loop is FLUX.2's (`Flux2Pipeline`); Bonsai supplies the components:

- **Transformer.** `transformer-packed-mflux/` with its `config.json`. `quantization_config.json` must say `format: mlx-packed-affine`, and its `bits` and `group_size` must be widths mlxcel-core's affine path implements (2 and 128 in the published checkpoint; the 1-bit binary variant is rejected naming the value). Every module in `quantized_modules` loads as an affine `UnifiedLinear` (`uint32` packed weight, bf16 scales and biases kept as stored); every other tensor is dense bf16. A listed module without `.scales` or `.biases`, or an unlisted one with `.scales`, fails the load naming it. Upstream's `klein_fast` megakernel and fused norm-RoPE Metal kernels are speed work over the same `quantized_matmul` arithmetic and are not ported.
- **Text encoder.** `text_encoder-mlx-4bit/` through the shared Qwen3 conditioner (MLX 4-bit, group 64, tied embeddings); tokenizer from `tokenizer/`. It is evicted after each encode.
- **Prompt.** Padded to `max_sequence_length` (upstream `bucketed_seq_len=False`), where FLUX.2 generation buckets.
- **VAE.** The checkpoint's `vae/` (the full FLUX.2 VAE). Upstream instead downloads `black-forest-labs/FLUX.2-small-decoder` (decoder channels 96/192/384/384) at load time, ignoring `vae/`; this port does not fetch a second repository at load. The two decoders give 34.96 dB against each other on the parity case, and this build's output is 29.82 dB against upstream as shipped.
- **Batches.** `n` images run one at a time with seed `seed + i`, as upstream's server loops, and each reports its own seed.

### Parity

Against upstream mlx-vlm 0.7.8 (git 73d366ec) on mlx 0.32.4, CUDA, NVIDIA GB10, `MLX_ENABLE_TF32=0`, "A tiny glass bonsai tree on a moonlit desk", 512x512, 4 steps, seed 9909. `tests/fixtures/bonsai/generate_bonsai_fixtures.py` runs upstream's own Bonsai pipeline with the Metal-only fused kernels replaced by their arithmetic in MLX ops and the checkpoint's `vae/` as the decoder; the `#[ignore]` tests are in `src/imagegen/bonsai/parity_tests.rs`.

| Check | Result |
|-------|--------|
| Initial noise | bit-identical |
| Transformer in f32 against upstream's FLUX.2 transformer in f32 with Bonsai's weights, same inputs | rel RMS 1.19e-6 |
| One bf16 transformer step on upstream's exact step-0 inputs | rel RMS 8.1e-3 |
| VAE decoding upstream's final latents | 57.61 dB |
| End-to-end PNG | 31.18 dB |

The bf16 final latents differ from upstream's by 1.0e-1 relative RMS, and the prompt embeddings by 2.3e-2 (1.7e-1 on the padding rows). Neither is a port error: upstream's 4-bit conditioner in bf16 is itself 1.95e-2 from the same conditioner in f32 (this port: 1.74e-2), with the same growth along the padding, and upstream disagrees with itself by 3.5e-2 in the final latents when its FLUX.2 transformer replaces `klein_fast` on the same weights. Under the epic-wide criterion the relative RMS figures above are reported as non-blocking drift, not gated. The tests gate on PSNR >= 30 dB, bit-identical noise and a 0.2 latent sanity bound that a wrong port (about 1.4) cannot pass. The megakernel itself cannot run off Metal, so its agreement with `quantized_matmul` is by reading `klein_fast/blocks.py`: `QuantizedLinearKernel` calls `mx.quantized_matmul`, and the fused kernels compute f32 RMSNorm times the gain and an interleaved-pair RoPE, cast to bf16.

## ERNIE-Image (`ernie_image`)

A port of `mlx_vlm/models/ernie_image` (`Blaizzy/mlx-vlm` at git 73d366ec), in `src/imagegen/ernie_image/`, for `baidu/ERNIE-Image-Turbo` and `baidu/ERNIE-Image`. The raw Diffusers tree loads directly; quantized conversions (upstream `convert.py`) are rejected by name.

- **Variant.** `mlx_ernie_image.json`'s `source` or `variant` when present, else the directory name (Turbo when it contains `turbo`; a Hugging Face cache snapshot uses its repo id). The two checkpoints share every config file that would tell them apart, so upstream decides by name too.
- **Transformer.** A 36-layer single-stream DiT (hidden 4096, 32 heads): image tokens first, then the projected text, with 3-axis RoPE and a key mask over text padding.
- **Text conditioner.** The `text_encoder/` Ministral 3 checkpoint through the in-tree `Ministral3Model::forward_hidden_states`: every decoder layer but the last, no final norm, prompts truncated at 2048 tokens. The in-tree Ministral 3 now applies the checkpoint's YaRN rope scaling, which it ignored before; without it the conditioning differs from upstream's.
- **Prompt enhancer.** `pe/` (a Ministral 3 decoder) with `pe_tokenizer/` rewrites the prompt from `{"prompt", "width", "height"}` JSON with upstream's chat template, temperature 0.6, top-p 0.95, at most 2048 new tokens, seeded per image. The rendered enhancer input is capped at 4096 tokens (twice the conditioner's 2048); a longer prompt is `400` with the enhancer on and is truncated to 2048 tokens with `use_prompt_enhancer: false`. `use_prompt_enhancer` decides: `true` enhances (`400` when the checkpoint has no `pe/`), `false` never does, and when omitted text-to-image is enhanced whenever `pe/` exists and an edit never is, since the enhancer cannot see the source image. The enhanced text is the `revised_prompt` of each image. The published checkpoints write Chinese prompts, as upstream does.
- **Sampling.** bf16 noise `normal([1, 128, H/16, W/16], key(seed))`, upstream's static-shift flow-matching schedule, and an Euler step in f32. CFG (negative first, batch 2) runs only when guidance is above 1, so Turbo at its default guidance never encodes the negative prompt.
- **Edit** (`--task edit`, exactly one `--image`; two are `400`). SDEdit: the source is encoded with the FLUX.2 VAE, batch-normalized, and noised to `(1 - sigma) * source + sigma * noise` at `start = steps - max(1, min(steps, round(steps * image_strength)))`. Without `--size` the source is scaled toward 1 MP (never up), at most 2048 on a side, rounded down to 16.
- **Options** (`--gen-kwargs` or extra JSON fields): `image_strength` (alias `strength`, edit only, default 0.6, in `(0, 1]`), `negative_prompt`, `use_prompt_enhancer`, `prompt_enhancer_max_tokens` (1 to 2048). Each wrong type or range is `400`.
- **Batches.** `n` images run upstream's whole pipeline once each with seed `seed + i`, enhancement included.
- **Memory.** The text encoder, transformer and VAE stay resident (about 24 GB in bf16); the enhancer (about 7 GB) loads on first use and stays. Upstream can drop the encoder and enhancer after each prompt (`evict_text_encoder`); keeping them lets a server answer the next request without reloading.

```bash
mlxcel generate-image -m models/mlx/ernie-image-turbo -p "A red panda serving tea in a moonlit bamboo forest" --size 1024x1024
mlxcel generate-image -m models/mlx/ernie-image-turbo --task edit --image panda.png -p "Make it snow" \
    --gen-kwargs '{"image_strength": 0.7}'
```

### ERNIE-Image precision

Upstream casts every floating transformer weight to bf16 and draws bf16 noise; modulation, gates and the scheduler step run in f32. The port keeps that on every backend (`weights::TRANSFORMER_DTYPE`). Under the f16 rule the transformer fails: with f16 weights the Turbo 1024x1024 parity case produces NaN latents and a 9.54 dB image, so bf16 stays. The text encoder and the enhancer load through `crate::models::load_text_weights`, the policy every text model uses, and the conditioning they produce matches upstream exactly on the parity prompt. Two kernel details had to follow MLX's `nn` layers for the batch-1 run to be bit-exact: GELU computed in the input dtype (`nn.gelu`, where `mlxcel_core::gelu` computes in f32), and biased linears as one `addmm` (`nn.Linear`, where `Linear::forward` adds the bias after the matmul).

### ERNIE-Image parity

Seed-fixed runs at 1024x1024, seed 42, prompt "A red panda serving tea in a moonlit bamboo forest", enhancer off, against upstream mlx-vlm 0.7.8 (git 73d366ec) on mlx 0.32.4 with the CUDA backend (NVIDIA GB10), `MLX_ENABLE_TF32=0` on both sides. `tests/fixtures/generate_ernie_image_fixtures.py` produces the per-case oracle files and the committed summary `tests/fixtures/imagegen/ernie_image_parity.safetensors` (noise checksum and head, 8x8-pooled final latents and image). The initial noise is bit-identical in every case, and the Turbo case's text conditioning is too.

| Case | First transformer forward on upstream's inputs | Final latents rel-RMS | PNG PSNR | Gate (latents / PSNR) |
|------|-----------------------------------------------|-----------------------|----------|-----------------------|
| Turbo, 8 steps, guidance 1 | 0 | 0 | 57.00 dB | 1e-2 / 30 dB |
| Turbo edit of the Turbo image, 8 steps, guidance 3, `image_strength` 0.7 | 4.90e-2 | 1.35e-1 | 32.23 dB | 2.5e-1 / 30 dB |
| Base, 50 steps, guidance 4, empty negative prompt | 5.02e-3 | 3.03e-1 | 24.24 dB (misses 30 dB) | 4.5e-1 / 20 dB |
| Base control, 8 steps, guidance 4 (oracle directory only) | 5.02e-3 | 1.64e-1 | 31.43 dB | 2.5e-1 / 30 dB |

At batch 1 every op reproduces upstream bit for bit, so the Turbo text-to-image case is exact (57 dB is the PNG round trip of identical latents). The two CFG cases run the transformer at batch 2, where one small-M bf16 GEMM, the text projection `[2, 13, 3072] x [3072, 4096]`, picks a different kernel in mlxcel's pinned MLX than in the mlx 0.32.4 wheel and rounds differently in the last bit; 36 layers over 8 or 50 steps amplify that. Upstream shows the same spread against itself: evaluating only that projection one batch row at a time moves its edit case by 1.39e-1 (32.14 dB) and its base case by 2.89e-1 (23.70 dB). Upstream is not batch-invariant either: its batch-2 row 1 differs from the same row at batch 1 by 5.3e-2. **The 50-step base case does not meet the epic's 30 dB PNG criterion.** The 8-step control runs the same checkpoint, prompt, seed and CFG batch-2 forward and reaches 31.43 dB, so the batched forward is sound and the miss is the 50-step loop amplifying the rounding difference. The 50-step images are compositionally identical (the same panda, teacup, teapot and moon, with differences in fine detail). Its 20 dB floor only rejects a grossly broken pipeline; it does not satisfy the epic criterion. For scale, the port before the GELU and `addmm` fixes drifted to 3.3e-1 (24.8 dB) on the batch-1 Turbo case, which the Turbo gate rejects.

```bash
MLX_ENABLE_TF32=0 <lib test binary> --ignored --test-threads=1 imagegen::ernie_image::parity_tests::
```

`MLXCEL_ERNIE_IMAGE_TURBO` and `MLXCEL_ERNIE_IMAGE_BASE` point at the checkpoints (default `models/mlx/ernie-image-turbo`, `models/mlx/ernie-image`); `MLXCEL_ERNIE_IMAGE_ORACLE_DIR` (the generator's `--out`) adds the full-resolution comparison, the first-forward and conditioning diagnostics, and the 8-step control (generate it with `--control`); `MLXCEL_ERNIE_IMAGE_OUT` saves the PNGs.

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
| `vae` | `Flux2Vae`, `Flux2VaeConfig`: decode, packed-latent decode, optional encode (the `quant_conv` is optional, absent in a FLUX.1 `AutoencoderKL` with `use_quant_conv: false`), `bn_normalize` and `denormalize_packed_latents`. | FLUX.2, ErnieImage, LLaDAImage, Z-Image |
| `qwen3_text_encoder` | `Qwen3TextEncoder`: hidden states of every layer (`num_layers + 1`, index 0 the embeddings, before the final norm) and `stacked_prompt_embeds` (`[B, T, n * H]`, FLUX.2 order). Built on `Qwen3Model::forward_all_hidden`. | FLUX.2, Z-Image, Ideogram 4, Bonsai |
| `qwen_image::vae` | `QwenImageVae::new`, `::load`: the Wan-style causal-3D VAE run on single frames (residual and plain layouts, Diffusers or MLX conv layout), `decode_normalized`, `encode_normalized`. | QwenImage, MingImage |
| `dit_blocks` | `TimestepEmbedder`, `FeedForward`, `MRoPE` (`compute_freqs_on_device` builds the inverse frequencies on the device, as upstream does), `apply_rotary`, `Attention`, `ZImageTransformerBlock`, `FinalLayer`, `sanitize_transformer_weights`, `BiasAdd` (opt-in fused bias `addmm` for the biased linears). | ZImage, LLaDAImage, MingImage |

Not shared: the FLUX.2 transformer, its empirical-`mu` scheduler and its tiled VAE decode live in `src/imagegen/flux2/`. Not served at all: multipart uploads, streaming partial images, video generation.

### Conventions a model must follow

- **Seeds.** Image `i` of `n` uses `seed + i` (FLUX.2 generation is the exception, see [Batched seeds](#batched-seeds)). mlxcel links the same MLX C++ RNG as upstream, so the same seeding call (`mx.random.seed(seed)` in Qwen-Image and LLaDA-Image, `mx.random.key(seed)` in ERNIE-Image and Ming-Image) followed by the same `normal` shape and dtype gives the same initial noise; keep upstream's draw order so whole-run parity tests can compare.
- **Threads.** MLX work is thread-affine. The server builds the model on the worker thread; `ImageGenerationModel` carries no `Send` bound.
- **Convolution weights.** Diffusers stores 4-D conv weights `(out, in, kH, kW)`; MLX expects `(out, kH, kW, in)`. Decide the layout once per checkpoint from a tensor whose two layouts differ in shape (`vae::model::detect_conv_layout`), and never permute an already-converted checkpoint twice. Deciding per tensor from its shape is wrong for any conv whose input channel count equals its kernel size; see the note under [Verification](#verification).
- **Half precision.** Norms run in f32 and cast back to the activation dtype. Use `fast_scaled_dot_product_attention`, not `scaled_dot_product_attention`: the latter multiplies by an f32 scalar and promotes half-precision scores to f32.

### Qwen3 conditioner

`Qwen3TextEncoder::load(dir)` reads `config.json` (top-level Qwen3 fields or a nested `text_config`) and the safetensors, accepts the `model.`, `model.language_model.` and `language_model.model.` key layouts, drops `lm_head.*` and every non-text tensor, and honors MLX affine `quantization`. `from_weights(config, weights)` takes an in-memory `WeightMap` (Ideogram 4 dequantizes FP8 first) and converts nothing. RoPE is standard: `rope_scaling` is ignored, since text positions make M-RoPE sections a no-op.

The mask is causal plus key padding, added in f32. Blocked positions use `-30000`, not upstream's `-inf`: identical for every row with a visible key, but a fully blocked row (left padding) becomes a uniform average instead of NaN, which would otherwise reach the real positions through the next layer.

Checked against upstream's `Qwen3TextEncoder` on a tiny model, and on `mlx-community/Qwen3-0.6B-4bit` the last hidden state, normed and projected, reproduces the generation path's logits exactly (an `#[ignore]` test). The change to `qwen3.rs` is one added method; teacher-forced logit traces (`examples/logit_trace`, 96 positions, 8 chunks of 8 tokens behind 256 tokens of context) before and after are byte-identical.

Attention runs in the weights' dtype. Upstream's encoder (`qwen/attention.py`) casts q, k and v to f32 for SDPA and the output back. That is not the parity gap. Measured on GB10 against the CUDA oracle with `MLX_ENABLE_TF32=0` (#2666), running this encoder's attention in f32 moves the FLUX.2 prompt embeddings by 9.4e-4, the same size as upstream's own bf16-to-f32 attention step (1.0e-3), while this port's distance to upstream stays 1.19e-2 either way; likewise the Z-Image caption features stay at 9.3e-3 and the Ideogram 4 text features at 6.4e-2. The remaining gap is an order of magnitude larger and likely comes from bf16 GEMM reduction order between the two MLX builds (not measured). End to end, f32 attention only reshuffles the drift, and on Bonsai (29.2 dB) and Ideogram 4 (pooled latents 4.7e-2) it takes the runs past their parity gates, so the conditioner keeps the activation dtype.

## Precision

Upstream runs the DiT, the VAE and the conditioning in bf16. The project rule converts non-quantized bf16 to f16 at each new loading boundary, and a component keeps bf16 only when a seed-fixed comparison shows inf/NaN or a failed threshold under f16.

| Component | Default dtype | Evidence |
|-----------|---------------|----------|
| FLUX.2 VAE | f16 (`Flux2Vae::default_dtype`, from `bf16_to_f16_at_load`) | Real ERNIE-Image-Turbo VAE, 64x64 decode of a fixed latent: finite, rel-RMS 1.2e-3 against upstream f32. Upstream's own bf16 run differs from upstream f32 by 2.0e-2, so f16 is closer to f32 than the bf16 upstream default. |
| Z-Image DiT blocks | follows the checkpoint dtype | The blocks are dtype-agnostic. Rotation, SDPA inputs and residual adds cast to upstream MLX's promotion (bf16 or f16 with f32 is f32), because the CUDA build of mlxcel's MLX promotes bf16 with f32 to bf16. |
| Z-Image transformer and VAE | f16 weights, f32 latent stream | See [Z-Image precision](#z-image-precision). |
| FLUX.2 transformer | bf16 on every backend | With f16 weights (`MLXCEL_CUDA_F16_NORMALIZE=1` on GB10) the 4B generate case decodes NaN latents (10.2 dB). Edits run the transformer in f32 by promotion from the f32 reference latents, as upstream does. |
| FLUX.2 VAE (in FLUX.2) | `Flux2Vae::default_dtype` | bf16 on CUDA; under the f16 policy, VAE-only parity is 54.76 dB tiled and 52.37 dB untiled. |
| FLUX.2 Qwen3 text encoder | the Qwen3 loader's policy | f16 holds: with the f16 policy and a bf16 transformer, 4B generate is 34.47 dB and edit 38.54 dB. |
| Bonsai transformer | bf16 compute, 2-bit packed linears with bf16 scales and biases as stored | Upstream builds every linear input in bf16. |
| ERNIE-Image transformer | bf16 on every backend, f32 modulation, gates and scheduler step | With f16 weights the Turbo 1024x1024 case decodes NaN latents (9.54 dB). See [ERNIE-Image precision](#ernie-image-precision). |
| ERNIE-Image text encoder and prompt enhancer | the text-model loader's policy (`load_text_weights`) | The conditioning matches upstream bit for bit on the parity prompt. |
| ERNIE-Image VAE | `Flux2Vae::default_dtype` | As the FLUX.2 VAE row. |
| Qwen-Image transformer | `bf16_to_f16_at_load`: f16 on Metal, bf16 on CUDA unless `MLXCEL_CUDA_F16_NORMALIZE=1` | f16 holds: at 1024x1024, 4 steps, seed 7, f16 weights (`MLXCEL_CUDA_F16_NORMALIZE=1` on GB10) give 48.32 dB (generate) and 47.92 dB (edit) against upstream's bf16 PNG, finite throughout, against 52.22 dB and 52.61 dB in bf16. |
| Qwen-Image VAE | f32 as stored | The checkpoint ships f32; upstream runs it in f32. |
| Qwen-Image Qwen3-VL text encoder | the Qwen3-VL VLM loader's policy | f16 holds: the f16 numbers in the transformer row ran the conditioner under the same policy too, finite throughout. Its bf16 drift is in [Qwen-Image parity](#qwen-image-parity). |
| Bonsai VAE and 4-bit text encoder | as FLUX.2 | Under the f16 policy (`MLXCEL_CUDA_F16_NORMALIZE=1` on GB10) the parity case is 31.21 dB, against 31.18 dB in bf16. |
| Ideogram 4 transformers and text tower | bf16, dequantized from per-row FP8 at load | Upstream's own dequantization rounds to bf16; see [Ideogram 4 precision](#ideogram-4-precision). |
| Ideogram 4 VAE | `Flux2Vae::default_dtype` | As the FLUX.2 VAE row; the gated 512x512 run is 37.1 dB end to end. |
| Ming-Image conditioner | bf16 backbone, f32 connector and bridge, as stored and as upstream | The bf16 states are widened to f32 before `proj_in` and the `directvlm` norm, where upstream's promotion does that implicitly; see [Ming-Image precision](#ming-image-precision). |
| Ming-Image transformer | f16 for the tensors that read the f32 latent stream (f32 where the cast is not exact), bf16 for those that read bf16 activations | Reproduces upstream's bf16-with-f32 promotion on CUDA; first-step velocity 4.2e-3 and 2.6e-3 from upstream on the same inputs. |
| Ming-Image VAE | bf16 as stored | Upstream runs it in bf16; decode of upstream's final latents 97.7 dB and 96.8 dB. |
| LLaDA-Image text encoder, QueryFormer, projection, SigVQ, transformer | bf16 as stored, upstream's rounding points | See [LLaDA-Image precision](#llada-image-precision). |
| LLaDA-Image VAE | `Flux2Vae::default_dtype` | As the FLUX.2 VAE row; the edit source latents are at 8.8e-3 from upstream's corrected encoder. |

The measurement used the CPU MLX backend (see [Verification](#verification)).

## Registering a family

A model issue does the following; the runtime issue ships none of it.

1. Implement `ImageGenerationModel` in `src/imagegen/<family>/`: `family()` (the probe name), `supports(task)`, `defaults(task)` (steps and guidance, resolved from JSON metadata, never from weights), `size_multiple()`, and `generate(&ImageRequest)` returning `req.n` images with seeds `req.seed + i`.
2. Add the probe name to `IMAGE_FAMILIES` in `src/imagegen/probe.rs` if it is not there, with the marker keys or class names that identify it, and a fixture-based test in `probe_tests.rs`.
3. Add one row to `FAMILIES` in `src/imagegen/registry.rs`: `ImageFamily { name: "<family>", load: <constructor> }`. The constructor takes the checkpoint directory and runs on the thread that will run the model.
4. Add the family to `STANDALONE_ARCHITECTURE_FAMILIES` in `src/models/registry.rs` with `Runtime::ImageGenerate` and `OutputKind::Image`, so `mlxcel arch --json` lists it. A test there fails when the two tables disagree.
5. Extend this page and `docs/supported-models.md`.

A family the probe recognizes but `FAMILIES` does not list is reported as `image generation family <family> is not supported by this build`.

## Z-Image

`src/imagegen/z_image/` ports `mlx_vlm/models/z_image` (mlx-vlm `73d366ec`). It loads the raw Diffusers tree (`Tongyi-MAI/Z-Image-Turbo`, `Tongyi-MAI/Z-Image`) directly, and an mlx-vlm converted tree whose index metadata carries `mlx_vlm_format: "z_image"`. A converted tree's quantized transformer and VAE layers are dequantized at load with the metadata's mode: `affine` at 2, 3, 4, 5, 6 or 8 bits and group size 32, 64 or 128, `mxfp4` (4, 32), `mxfp8` (8, 32) and `nvfp4` (4, 16); any other combination is an error. Each packed tensor is also checked before MLX sees it (uint32 weight, `.scales` shaped `[..., in / group_size]`, `.biases` for `affine` only, uint8 scales for the `mx` and `nvfp4` modes), and a malformed one fails the load with the tensor's key. The text encoder's quantization comes from its own `config.json`.

- **Variant.** `scheduler/scheduler_config.json` `shift <= 3` is Turbo (8 steps, guidance 0; guidance must stay in `[0, 1]`), otherwise base (50 steps, guidance 4). Upstream's own generation default for Turbo is 9 steps; this port follows the issue's 8, which is also upstream's img2img default.
- **Prompt.** The chat template with `enable_thinking=True`, padded to 512 tokens with `<|endoftext|>`; the condition is the penultimate hidden state of the Qwen3 encoder, trimmed to the real tokens. The negative prompt (the request field, else the `negative_prompt` option, else empty) is encoded only when guidance is above 1.
- **Sampling.** `normal([1, 16, 1, H/8, W/8], f32, key(seed))`, `FlowMatchEulerDiscrete::static_shift(steps - 1, shift)`, the transformer fed `t = 1 - sigma`, and an Euler step on the negated prediction. CFG `pred + g * (pred - neg)` runs while `t <= cfg_truncation`.
- **img2img** (`--task edit`, exactly one `--image`). The image is resized with Lanczos only when its size differs from `--size`, encoded to the posterior mean, and noised to `(1 - sigma) * latents + sigma * noise` at `start = int(max(steps - min(steps * strength, steps), 0))` of `static_shift(steps, shift)`. A strength that leaves no step is rejected.
- **Options** (`--gen-kwargs` or extra JSON fields): `cfg_truncation` (default 1.0, in `[0, 1]`), `strength` (edit only, default 0.6, in `(0, 1]`), `negative_prompt`.
- **Memory.** The text encoder, transformer and VAE stay resident (about 20 GB for the bf16 checkpoints). Upstream drops the text encoder before loading the transformer; keeping it lets a server answer the next request without reloading 8 GB.

### Z-Image precision

Upstream keeps the transformer and VAE weights bf16 and feeds them f32 noise and timesteps, so MLX promotes the latent stream to f32. The CUDA build of mlxcel's MLX promotes bf16 with f32 to bf16, which would run that stream in bf16 (2.1e-2 relative RMS on one Turbo transformer forward, 1.1e-1 on the final latents). The port stores those weights as f16, which promotes with f32 to f32 on every backend. The bf16 to f16 cast is range-safe but not bit-exact on these checkpoints: the largest magnitude is 47.75 (f16 stops at 65504), and bf16 values below f16's normal range (6.1e-5) lose low bits, so about two thirds of the tensors change in some element, with a worst per-tensor error of 2.0e-7 relative RMS. The loader measures every tensor's f16 round trip and keeps any tensor above 1e-5 relative RMS (or one the cast overflows) in f32, which is exact and promotes the same way; for the VAE, any such tensor puts the whole VAE in f32. An f32 checkpoint therefore stays f32. Two islands keep the checkpoint's bf16 because upstream computes them in bf16 before the caption stream first meets f32: the caption embedder (RMSNorm and `addmm`) and the pre-attention path of the first context-refiner block. The text encoder runs in bf16, as upstream. Self-attention in the transformer and the VAE mid-block query-chunks its score matrix to the `MLXCEL_ATTENTION_CHUNK_BUDGET_MB` budget (default 1024 MiB, `0` disables; see [Environment variables](environment-variables.md)), so large sizes do not materialize the full `[heads, L, L]` scores.

### Z-Image parity

Seed-fixed runs at 512x512, seed 42, against upstream mlx-vlm `73d366ec` on mlx 0.32.4 with the CUDA backend (GB10), `MLX_ENABLE_TF32=0` on both sides. The initial noise is bit-identical in every case.

| Case | Final latents rel-RMS | PNG PSNR |
|------|-----------------------|----------|
| Turbo, 8 steps, guidance 0 | 8.6e-2 | 32.9 dB |
| Base, 50 steps, guidance 4, negative prompt `blurry, low quality`, `cfg_truncation` 1.0 | 6.2e-2 | 34.7 dB |
| Turbo img2img, strength 0.6 | 1.7e-2 | 46.1 dB |

The final-latents drift is the size of upstream's own sensitivity to how the bf16 caption stream is rounded. On the Turbo case, upstream against itself moves by 2.7e-2 (41.3 dB) when only its caption embedder's linear runs in f32, by 1.0e-1 (32.1 dB) when only its text encoder's attention runs in bf16 instead of f32, and by 9.5e-2 (31.6 dB) with TF32 on. The shared Qwen3 conditioner runs attention in bf16 where upstream's encoder casts q, k and v to f32; its caption features differ from upstream's by 9.3e-3, and feeding upstream's caption features to this port's denoiser lowers the Turbo drift to 3.4e-2. The attention dtype is not that gap: with upstream's f32 attention the caption features are still 9.3e-3 from upstream's (see [Qwen3 conditioner](#qwen3-conditioner)). The committed summary `tests/fixtures/imagegen/z_image_parity.safetensors` (noise checksums, 4x4-pooled final latents and images, regenerated by `tests/fixtures/generate_z_image_fixtures.py`) backs the `#[ignore]` tests:

```bash
MLX_ENABLE_TF32=0 <lib test binary> --ignored --test-threads=1 imagegen::z_image::tests::
```

`MLXCEL_Z_IMAGE_TURBO` and `MLXCEL_Z_IMAGE_BASE` point at the checkpoints (default `models/mlx/z-image-turbo-bf16`, `models/mlx/z-image-bf16`); `MLXCEL_Z_IMAGE_ORACLE_DIR` adds the full-resolution comparison and the img2img case.

## Mage-Flow (`mage_flow`)

`src/imagegen/mage_flow/` ports `mlx_vlm/models/mage_flow` (mlx-vlm `73d366ec`): a Qwen3-VL conditioner, a dual-stream transformer with 3-axis RoPE, and the Mage-VAE (a DiCo convolutional encoder and a CoD plus DiCo denoiser decoder, 16x downsampling, 128 latent channels). It loads the mlx-vlm conversion layout (`transformer/`, `vae/`, `text_encoder/`, `model_index.json` with `_class_name: "MageFlowPipeline"`). Transformer projections quantized by mlx-vlm (`affine`, `mxfp4`, `mxfp8`, `nvfp4`, read from the index metadata) run through the quantized matmul; a quantized VAE is refused. The text encoder loads through the Qwen3-VL VLM loader with its own `config.json` quantization.

| Variant | Checkpoint | Task | Steps / guidance |
|---------|------------|------|------------------|
| `mage-flow-turbo` | `mage-flow-community/Mage-Flow-Turbo` | generate | 4 / 1.0 |
| `mage-flow` | `mage-flow-community/Mage-Flow` | generate | 20 / 5.0 |
| `mage-flow-base` | `mage-flow-community/Mage-Flow-Base` | generate | 30 / 5.0 |
| `mage-flow-edit-turbo` | `mage-flow-community/Mage-Flow-Edit-Turbo` | edit | 4 / 1.0 |
| `mage-flow-edit` | `mage-flow-community/Mage-Flow-Edit` | edit | 30 / 5.0 |
| `mage-flow-edit-base` | `mage-flow-community/Mage-Flow-Edit-Base` | edit | 30 / 5.0 |

- **Variant.** From the directory name, as upstream's `variant_from_local_path`: `turbo` or `base` (with `edit`) pick those variants, `edit` alone is Mage-Flow-Edit, anything else Mage-Flow. Upstream matches the whole path string; this port reads only the nearest path component that mentions `mage` (else the leaf), so a parent directory such as `/data/database/` does not make every checkpoint under it a `-base` variant. Each checkpoint runs one task; the other is `400` on the server, and the CLI error names the variant.
- **Sizes.** Width and height in `512..=2048`, multiples of 16. An edit without a size uses the first reference's size rounded down to multiples of 16, or with `max_size` scales it so the long edge is `max_size` (round half to even) before rounding. `max_size` applies only when the request's size is that default; an explicit size wins, as upstream's explicit `width` and `height` do. A reference past the shared 16-megapixel check is still rejected before `max_size` can shrink the output (mlxcel's fail-closed order).
- **Edit token limit (mlxcel bound; upstream has none).** An edit runs the transformer over the target plus one grid per reference, all at the output size: `(1 + N) * (H / 16) * (W / 16)` image tokens per forward (twice per step with CFG). The total is limited to 65536, four 2048x2048 grids: a 2048x2048 output takes at most 3 references, a 1248x1248 one at most 9. A larger edit is `400` naming the computed count and the limit. One or two references at any size, the usual edit, stay under it.
- **Prompt.** Upstream's generation or edit system template; the conditioner's final-norm hidden states with the template prefix dropped (34 positions for generation, 64 for edits), at most 2048 prompt tokens. An edit prefixes `Image i: <|vision_start|><|image_pad|><|vision_end|>` per reference; each reference is resized (bicubic, only shrinking) so its long edge is `vl_cond_long_edge` and then sized by the Qwen3-VL rule (factor 32). An edit instruction over 2048 tokens is `400` rather than truncated. The text encoder is evicted after each encode, as upstream; the transformer and VAE load on the first request and stay resident.
- **Guidance.** CFG runs above guidance 1. Generation encodes the negative prompt (the request field, else the `negative_prompt` option, else `" "`); an empty negative prompt turns CFG off. Edits always use `negative or " "`. `renormalization` rescales each token's guided velocity to the norm of its conditional velocity, as upstream.
- **Sampling.** Initial noise `normal([1, H/16 * W/16, 128], bf16, key(seed))`, bit-identical to upstream; `FlowMatchEulerDiscrete::static_shift` with shift 6. Edits encode each reference (resized to the output size, PIL-equivalent bicubic) to the VAE posterior, sampled with `key(seed)` unless `sample_posterior` is false, and append the reference tokens after the target tokens.
- **Options** (`--gen-kwargs` or extra JSON fields): `negative_prompt`, `static_shift` (positive, default 6), `renormalization` (default false); edit only: `max_size`, `vl_cond_long_edge` (default 384, at most 2048; `null` or `<= 0` keeps the reference size, capped at 2048x2048 pixels per reference) and `sample_posterior` (default true). An edit-only option on a generate request is `400`.

### Mage-Flow precision

Upstream runs bf16 end to end with f32 islands that the port keeps: the transformer timestep embedding (frequencies rounded to bf16), the rotary math, and the VAE's local-attention scores and softmax. Two differences from upstream are deliberate:

- **LayerNorm.** MLX's fused CUDA `layer_norm` kernel (`backend/cuda/layer_norm.cu`, also in mlx 0.32.4) reuses one shared-memory buffer for its mean and variance block reductions with no barrier between them. On 384-wide bf16 rows, the VAE encoder's `norm_out`, a few rows per call come out wrong, so two identical edit runs differed (reference latents by about 1e-1, final PNGs at 29.8 to 35.5 dB from each other) in upstream and in this port alike. The port computes every Mage-Flow LayerNorm from separate ops with the same f32 statistics, which is deterministic; the fixture generator applies the same form to upstream's transformer and VAE. Other families still call the fused kernel.
- **Spatial mean.** The VAE's channel attention takes a spatial mean. The oracle's CUDA `mx.mean` over bf16 accumulates in bf16; the port accumulates in f32 and casts once. This is the only difference in the last denoiser block (9.8e-3 on the same input).

### Mage-Flow parity

Seed-fixed runs at 512x512, seed 42, 4 steps, against upstream mlx-vlm `73d366ec` on mlx 0.32.4 with the CUDA backend (GB10), `MLX_ENABLE_TF32=0` on both sides. The initial noise is bit-identical.

| Case | Final latents rel-RMS | PNG PSNR | On upstream's conditioning |
|------|-----------------------|----------|----------------------------|
| Turbo, generate | 6.6e-2 | 38.4 dB | 3.4e-2, 42.3 dB |
| Edit-Turbo, two references | 2.5e-1 | 21.8 dB | 5.1e-2, 39.9 dB |

The edit gap is the conditioner, not the denoiser or VAE: fed upstream's own conditioner output, the edit run is at 39.9 dB. A bf16 Qwen3-VL cannot reproduce upstream's bf16 hidden states closely. One same-input decoder layer already differs by 5.7e-3 between this build's MLX and mlx 0.32.4, and upstream's own bf16 hidden states sit 1.8e-2 (generate) and 2.6e-1 (edit, through the 24-layer vision tower and DeepStack) from the same model widened to f32. Upstream itself lands at 22.9 dB from its own edit PNG when only its conditioner runs in f32 (with the deterministic LayerNorm above on both runs). The conditioner tests therefore check the inputs exactly (token ids, image grid, M-RoPE positions) and require the port's hidden states to sit no further from the f32 reference than upstream's bf16 run does (measured ratios 0.96 and 1.04, bound 1.2). The VAE is gated stage by stage on upstream's own inputs at 5e-3 (CoD decoder, posterior head and pixel head exact; the last denoiser block at 2e-2 for the spatial-mean accumulation above), and end to end at 5e-2 (measured 1.0e-2 decode, 2.4e-2 encode mean).

`tests/fixtures/imagegen/mage_flow_parity.safetensors` (token ids, M-RoPE positions, noise checksums, 4x4-pooled final latents and images, 64-channel pools of the conditioner output in bf16 and in f32) and `mage_flow_tiny_refs.safetensors` (RoPE tables, a tiny transformer in f32 and bf16, real Mage-VAE stage references) are regenerated by `tests/fixtures/generate_mage_flow_fixtures.py`. The real-checkpoint tests are `#[ignore]`:

```bash
MLX_ENABLE_TF32=0 <lib test binary> --include-ignored --test-threads=1 imagegen::mage_flow::
```

They read `models/mlx/mage-flow-turbo` and `models/mlx/mage-flow-edit-turbo`. `MLXCEL_MAGE_FLOW_ORACLE_DIR` (the generator's `--out` directory) adds the full-resolution PNG and latent comparisons, the run on upstream's conditioning, and the processor's pixel values (within two 8-bit levels of PIL's bicubic).

## Qwen-Image (`qwen_image`)

`src/imagegen/qwen_image/` ports `mlx_vlm/models/qwen_image` (mlx-vlm `73d366ec`) and loads the original Diffusers tree of `Qwen/Qwen-Image-2.1` (`model_index.json` with `_class_name: "QwenImage21Pipeline"`, `transformer/`, `vae/`, `text_encoder/`, `processor/`, `scheduler/`). Components: a Qwen3-VL conditioner (mlxcel's Qwen3-VL loader, with the final-norm-free hidden states of `Qwen3VLModel::forward_hidden_pre_norm`), a 32-block single-stream transformer with 3-axis RoPE (axes 16, 56, 56) and a block-causal mask, and the Qwen-Image VAE (16x spatial downsampling, 64 latent channels). Transformer linears quantized as MLX affine (`quantization` in `transformer/config.json`) run through the quantized matmul.

- **Sizes.** Width and height in `256..=2048`, multiples of 16 (32 for an explicit edit size). An edit without a size of its own is the size of its last resized reference.
- **References.** Each reference is opened as RGBA and resized with Pillow's Lanczos (bit-exact against Pillow 13) to an area of `output_resolution**2` at its own aspect, rounded to multiples of 32. The conditioner sees it composited on white; the VAE encodes all four channels, so edit output is RGBA and keeps the reference's transparency. Generation output is RGB.
- **Edit limits (mlxcel bounds; upstream has none).** The target plus every reference may hold at most 32768 image tokens (two 2048x2048 grids: a 2048 edit takes one 2048 reference, the default 1024 edit up to 7 references); a larger edit is `400` naming the count. The limit is half of Mage-Flow's because the block-causal mask is `S x S` and the edit KV cache holds about 512 KiB per cached token per CFG branch (about 17 GB at the limit). A resized reference the Qwen3-VL processor would resize again (past 200:1, or outside its pixel bounds) and an edit prompt or CFG negative with a vision marker are `400` before any component loads.
- **Prompt.** Upstream's system template; the hidden states drop the 14-token template prefix and keep at most 1024 prompt tokens (generation truncates, as upstream; an edit over the limit is `400`). An edit prompt places `<imageN><|vision_start|><|image_pad|><|vision_end|>` per reference, and a user prompt that itself contains vision markers is `400`. The text encoder is evicted (MLX cache cleared) after each encode, as upstream; the transformer and VAE load on the first request and stay resident.
- **Conditioning.** Target tokens take the real timestep's modulation; text and reference tokens take the zero-timestep row (`causal_condition`), and attention is block-causal: text and references never attend to the target. With `use_kv_cache` (the default) an edit caches the references' K/V on the first step and later steps run the target alone; the result equals the uncached run (bitwise in upstream, and in this port at 4 steps).
- **Guidance.** CFG (`neg + g * (pos - neg)`) runs above guidance 1, encoding the negative prompt (the request field, else the `negative_prompt` option, else `" "`). The defaults use guidance 1, so one pass per step.
- **Sampling.** Initial noise `mx.random.seed(seed)` then `normal([1, 1, 64, H/16, W/16], f32)` cast to bf16, bit-identical to upstream; `FlowMatchEulerDiscrete::dynamic` with the scheduler's shift parameters, the Euler step accumulated in f32.
- **Options** (`--gen-kwargs` or extra JSON fields): `negative_prompt`; edit only: `output_resolution` (256 to 2048, default 1024) and `use_kv_cache` (default true). An edit-only option on a generate request is `400`.

### Qwen-Image parity

Seed-fixed runs at 1024x1024, seed 7, 4 steps, against upstream mlx-vlm `73d366ec` on mlx 0.32.4 with the CUDA backend (GB10), `MLX_ENABLE_TF32=0` on both sides, bf16 transformer. The initial noise is bit-identical. Both final-latent drifts are below the 1e-2 the issue set; the gate is PSNR at least 30 dB plus per-case bounds on the 8x8-pooled latents.

| Case | Final latents rel-RMS (drift) | PNG PSNR |
|------|-------------------------------|----------|
| Generate | 7.6e-3 | 52.22 dB |
| Edit, RGBA reference, KV cache | 6.4e-3 | 52.61 dB (alpha 53.58 dB) |
| Edit without the KV cache, against the cached run | identical | inf (also bit-identical at 40 steps through the CLI) |

The conditioner's text positions sit 3.7e-2 (rel-RMS) from upstream's, spread evenly over positions; the `<|image_pad|>` rows differ more (0.58) but the transformer replaces them with the references' VAE tokens. The reference latents are at 1.7e-3, and the real VAE decodes a fixed latent at 2.1e-4 and encodes at 5.7e-7. The latent bound (2e-2) is far below the broken arms measured on the generate case: RoPE height and width swapped gives 0.77 (13.2 dB), and every token on the real timestep's modulation row gives 0.23 (18.3 dB). The LayerNorms use the composed form Mage-Flow uses (see [Mage-Flow precision](#mage-flow-precision)). The change to `qwen3_vl.rs` is one added method on a const-generic forward; teacher-forced logit traces (`examples/logit_trace`, text and image arms) before and after are byte-identical.

`tests/fixtures/imagegen/qwen_image_parity.safetensors` (token ids, `drop_idx`, noise checksums, 8x8-pooled latents and 32x32-pooled images) and `qwen_image_tiny_refs.safetensors` (a tiny transformer with and without the KV cache, RoPE tables, tiny residual and plain VAEs, real-VAE references) are regenerated by `tests/fixtures/generate_qwen_image_fixtures.py`. The real-checkpoint tests are `#[ignore]`:

```bash
MLX_ENABLE_TF32=0 <lib test binary> --include-ignored --test-threads=1 imagegen::qwen_image::
```

They read `models/mlx/qwen-image-2.1`. `MLXCEL_QWEN_IMAGE_ORACLE_DIR` (the generator's `--out` directory) adds the full-resolution PNG and latent comparisons and the edit reference image; `MLXCEL_QWEN_IMAGE_OUT` saves the PNGs and conditioner outputs.

## Ideogram 4 (`ideogram4`)

`src/imagegen/ideogram4/` ports `mlx_vlm/models/ideogram4` (mlx-vlm `73d366ec`) for the `ideogram-ai/ideogram-4-fp8` snapshot. The checkpoint is license-gated (Ideogram 4 Non-Commercial License): accept the terms on Hugging Face and download it yourself; mlxcel never fetches it. The snapshot layout is upstream's (`model_index.json` with `_class_name: "Ideogram4Pipeline"`, `tokenizer/`, `transformer/`, `unconditional_transformer/`, `text_encoder/`, `vae/`); a missing file is named at load. The probe picks `ideogram4` from the pipeline class name, ahead of the FLUX.2 VAE class the snapshot also carries.

The model is a 34-layer single-stream DiT (18 heads of 256, interleaved 3-axis M-RoPE with image positions offset by 65536) conditioned on 13 hidden states (layers 0, 3, ..., 33, 35) of the Qwen3-VL 8B text tower, stacked hidden-major to 53248 features per token, through the shared [Qwen3 conditioner](#qwen3-conditioner). A second, unconditional transformer gives the classifier-free guidance branch. Latents are 128-channel FLUX.2 VAE latents patchified 2x2, normalized with fixed per-channel shift and scale. Generation only: `/v1/images/edits` is `400` naming the task, and the CLI rejects reference images.

- **Weights.** Linear weights ship as F8_E4M3 bytes with an f32 per-row `weight_scale`. They are dequantized at load to bf16 the way upstream does (E4M3 decoded through a 256-entry table, rounded to bf16, times the scale rounded to bf16), so both transformers and the text tower run in bf16 as upstream. The VAE (bf16 on disk) loads through `Flux2Vae` with `Flux2Vae::default_dtype`. The `unconditional_transformer` never reads text, so its `llm_cond_*` weights are neither read nor dequantized; the conditional transformer must carry them. Transformer configs are bounds-checked (`num_layers` in 1 to 4096, overflow-checked head and M-RoPE sizes) and every weight's shape and dtype is checked at load, so a wrong-shaped tensor or one still in raw FP8 bytes is a named load error.
- **Memory.** The text encoder (about 16 GB in bf16) loads for each request and is dropped, with the MLX cache cleared, before the transformers run. The two transformers (about 18.6 GB each in bf16) and the VAE load on the first request and stay resident, so by weight size a served model holds about 38 GB between requests and about 54 GB while a later request encodes its prompt (computed from the tensor sizes, not a measured peak).
- **Sizes.** Width and height in `256..=2048`, multiples of 16, aspect ratio at most 6:1 (upstream's `validate_dimensions` messages). The prompt, after wrapping, is at most 2048 tokens.
- **Prompt.** A prompt that starts with `{` is a JSON caption and passes through; malformed JSON is `400`. Plain text is wrapped into upstream's minimal caption (`high_level_description`, a background placeholder and one element) and the wrapped caption is returned as `revised_prompt`. `auto_json_caption: false` sends the text unchanged and `revised_prompt` stays `null`. The tokenizer applies the snapshot's chat template to one user message, as upstream does. A prompt over 1 MiB (`MAX_PROMPT_BYTES`) is `400` before tokenizing, and one over 2048 tokens is `400` with upstream's message. Caption schema warnings are kept up to 64 and logged up to 8 (the log line counts the rest); a warning dropped past the cap still marks the caption as not structured.
- **Sampling.** Initial noise `mx.random.seed(seed); normal([1, H/16 * W/16, 128], f32)`, which is row 1 of `split(key(seed))`, not `key(seed)`; bit-identical to upstream. The schedule is a logit-normal over the log-SNR with a resolution shift; the step loop, guidance blend and Euler update follow upstream's `generate_array`.
- **Presets** (`sampler_preset`): `V4_DEFAULT_20` (default: 20 steps, guidance 7 then 3 for the last two denoising steps, `mu` 0, `std` 1.75), `V4_QUALITY_48` (48 steps, `std` 1.5) and `V4_TURBO_12` (12 steps, `mu` 0.5, `std` 1.75). An explicit `guidance` (even one equal to 7), or a `steps` that differs from the preset's step count, replaces the preset's guidance schedule with a constant guidance (7 unless given), as upstream does; `guidance_schedule` sets a per-step schedule instead.
- **Options** (`--gen-kwargs` or extra JSON fields): `sampler_preset`, `auto_json_caption` (default true), `guidance_schedule` (one number per step, at most the step limit entries; as upstream, entry `i` is applied at loop index `i` and the loop counts down, so the list is consumed from its end), `mu`, `std` (positive). `prompt_expansion_model` (upstream's optional second model that rewrites the prompt) is `400`: mlxcel does not run it. Unknown presets list the valid names. A non-empty `negative_prompt` is `400`: upstream's unconditional branch takes no text, so it would otherwise be dropped silently. `n` images use seeds `seed + i` (the shared convention); upstream's `num_images` instead draws one batched noise tensor from `seed`.

### Ideogram 4 precision

Upstream runs bf16 with f32 islands, and the port keeps each one: per-row FP8 dequantized to bf16, the RMSNorm mean taken in f32 then rounded to bf16 before `rsqrt` (composed from separate ops, not the fused kernel), the parameter-free LayerNorm in f32, the M-RoPE tables computed in f32 and cast to bf16, attention scores and softmax in f32, the timestep rounded to bf16 before its f32 sinusoid, the f32 output head, and the f32 latent stream and scheduler arithmetic. Biased projections use one fused `addmm`, as MLX `nn.Linear` does. The f16 policy is not applied to the transformers or the text tower: the weights arrive as FP8, not bf16, and upstream's bf16 dequantization is part of the reference.

### Ideogram 4 parity

Seed-fixed runs against upstream mlx-vlm `73d366ec` on mlx 0.32.4 with the CUDA backend (NVIDIA GB10), `MLX_ENABLE_TF32=0` on both sides. The initial noise is bit-identical (FNV-1a over the f32 bytes).

| Case | Final latents rel-RMS (drift) | PNG PSNR | On upstream's text features |
|------|-------------------------------|----------|-----------------------------|
| `V4_TURBO_12`, 512x512, seed 0, red-square JSON caption (gated) | 6.2e-2 | 37.1 dB | 2.9e-2, 46.0 dB |
| `V4_DEFAULT_20`, 1024x1024, seed 0, same caption (CLI run) | not recorded | 47.9 dB | |

The first step's velocities on the same noise differ by 6.2e-3 (conditional) and 3.2e-3 (unconditional, no text involved), and the text features by 6.4e-2: most of the end-to-end drift is the bf16 text tower, which the shared conditioner does not reproduce bit for bit. Running the tower's attention in f32 as upstream does leaves the text features at 6.4e-2 (see [Qwen3 conditioner](#qwen3-conditioner)). The test gates the pooled final latents (rel-RMS at most 4.5e-2, measured 3.2e-2), the 8x8-pooled image (at least 38 dB, measured 43.7 dB) and, with the oracle directory, the full PNG (at least 30 dB). Two upstream controls show what the bounds reject: wrong noise (seed 1) scores 4.2e-1 pooled latents and 12.3 dB PNG; dropping the preset's guidance schedule (guidance 7 instead of 3 on the last step) scores 5.2e-2 pooled latents but 48.1 dB PNG, so only the latent bound catches it. A tiny two-layer transformer with random bf16 weights matches upstream bit for bit (rel-RMS 0).

For some prompts and seeds this checkpoint renders a grey or checkerboard placeholder reading "Image blocked by safety filter" (the 1024x1024 red-square run, and a wrapped plain-text "red fox" prompt at seed 1). Upstream produces the same image (47.9 dB and 51.3 dB), so it is the checkpoint's output, not an mlxcel filter.

`tests/fixtures/imagegen/ideogram4_parity.safetensors` (token ids, noise checksum and head, 4x4-token-pooled final latents, 8x8-pooled image) and `ideogram4_tiny_refs.safetensors` (the tiny transformer's weights, inputs, M-RoPE tables and outputs) are regenerated by `tests/fixtures/generate_ideogram4_fixtures.py` (`--tiny PATH` for the second). The real-checkpoint test is `#[ignore]` and skips when `models/mlx/ideogram-4-fp8` is absent:

```bash
MLX_ENABLE_TF32=0 <lib test binary> --ignored --test-threads=1 imagegen::ideogram4::parity_tests
```

`MLXCEL_IDEOGRAM4_ORACLE_DIR` (the generator's `--out` directory) adds the full-resolution PNG and latent comparisons, the text-feature and first-step diagnostics, and the run on upstream's text features; `MLXCEL_IDEOGRAM4_OUT` saves the PNG.

## Ming-Image (`ming_image`)

`src/imagegen/ming_image/` ports `mlx_vlm/models/ming_image` (mlx-vlm `73d366ec`) for Ming-Image-0.1-Design (inclusionAI), a text-to-image model whose output is RGBA. It loads the original tree (`inclusionAI/Ming-Image-0.1-Design`: `mllm/`, `connector/`, `mlp/`, `transformer/`, `vae/`) and the MLX conversions (`nativ-community/Ming-Image-0.1-Design-MLX-*`, which merge the conditioner into `text_encoder/`, with an affine `quantization` block in each quantized component's `config.json`).

- **Conditioner.** The BailingMoeV2 backbone of Ming-flash-omni (20 layers, causal) runs over upstream's chat template followed by `<image_start>`, 256 learned query tokens and `<image_end>`. The query tokens are routed by the per-token `image_gate` router (`src/models/bailing_moe_image_router.rs`, additive: the text router and every other `bailing_moe` path are unchanged). The final normed query states go through `proj_in`, a 28-layer bidirectional Qwen2 connector in f32 and `proj_out` into `cap_feats`; the text tokens of layers 5, 12 and 20, concatenated and RMS-normed, go through the `directvlm` projection into `cap_feats_2`. The vision tower, audio gates and output head of `mllm/` are never read.
- **Denoiser and VAE.** The Z-Image NextDiT blocks from [`dit_blocks`](#shared-building-blocks) with the second caption stream concatenated and zero padding masked out of attention, and the shared [Qwen-Image VAE](#qwen-image-qwen_image) in its plain layout decoding four channels (the raw tree's 5-D causal kernels are reduced to their last temporal slice). Every image is RGBA, and the PNG keeps the alpha channel on the CLI and the server.
- **Memory.** The bf16 backbone is about 33 GB. The text encoder loads for each request and is dropped before the transformer and VAE run; those two load on the first request and stay resident. The checkpoints are 50 GB (bf16), 24 GB (8-bit) and 13 GB (4-bit) on disk.
- **Requests.** Generation only: an edit is `400` naming the task. Width and height in `256..=2048`, multiples of 16 (Qwen-Image's `validate_dimensions`). Defaults are 12 steps at guidance 1.0; the model is trained with CFG off, so any other guidance is `400` (`Ming-Image-0.1-Design is trained for guidance 1.0 (CFG off), got 3`). Image `i` of `n` uses seed `seed + i`.
- **Sampling.** Initial noise `normal([1, 16, 1, H/8, W/8], f32, key(seed))`, bit-identical to upstream; `FlowMatchEulerDiscrete::dynamic` with the checkpoint's `scheduler_shift`. Latents are unscaled by the VAE's scaling factor (8.0064) and shift before the decode.

### Ming-Image precision

Upstream runs the backbone, the DiT and the VAE in bf16, and the connector and bridge in the f32 they are stored in. The port keeps those dtypes. Two places need explicit casts on this build: the CUDA build of mlxcel's MLX promotes bf16 with f32 to bf16 where upstream's promotes to f32, so the bf16 backbone states are widened to f32 before `proj_in` and the `directvlm` norm, and the transformer tensors that read the f32 latent stream are stored as f16 (f32 when the cast is not exact) while the ones that read bf16 activations stay bf16. A quantized transformer is dequantized at load (its scales are bf16, so the dense weights are bf16); the quantized conditioner runs through the quantized matmul.

### Ming-Image parity

Seed-fixed runs at 1024x1024, 12 steps, seed 0 against upstream mlx-vlm `73d366ec` on mlx 0.32.4 with the CUDA backend (NVIDIA GB10), `MLX_ENABLE_TF32=0` on both sides. The initial noise is bit-identical (FNV-1a over the f32 bytes) and the templated ids are identical.

| Case | Final latents pooled rel-RMS | PNG PSNR | Alpha PSNR | On upstream's conditioning |
|------|------------------------------|----------|------------|----------------------------|
| `gen`: poster prompt, bf16 | 3.37e-1 | 24.98 dB (recorded) | 65.81 dB | 3.90e-2, 32.45 dB |
| `rgba`: transparent-background apple icon, bf16 | 3.03e-2 | 30.67 dB | 72.16 dB | 1.38e-3, 52.69 dB, alpha 82.02 dB |
| `q4`: poster prompt, 4-bit conversion against upstream on the same conversion (recorded) | 4.44e-2 | 41.91 dB | 75.97 dB | |

Per stage on upstream's own inputs: the first step's velocity 4.2e-3 (`gen`) and 2.6e-3 (`rgba`), 2.3e-3 on the 4-bit conversion; the VAE decode of upstream's final latents 97.7 dB and 96.8 dB (100.0 dB on 4-bit). The conditioner on the same ids: `cap_feats` 2.05e-1, `cap_feats_2` 5.5e-2 and 7.0e-2, the text states 1.2e-2 and 6.2e-3.

**The `gen` exception.** The poster case is a documented conditioner-precision exception, accepted by the maintainer as Mage-Flow's edit case was. Each backbone layer alone, on upstream's own inputs, differs from upstream by 1e-4 to 7.5e-4 (rounding level), and the final norm alone is bit-exact; through 20 MoE layers that accumulates to the 0.2 query-span difference above. This prompt is chaotic at that level: upstream perturbed against itself at the same drift (its query states offset by 1e-3) lands at 16.37 dB from its own PNG on the poster prompt and 29.98 dB on the RGBA prompt. Fed upstream's conditioning, the port is at 32.45 dB, so the denoiser and VAE are not the gap. The `gen` gate is therefore the run on upstream's conditioning (at least 30 dB, pooled latents within 8e-2; dropping `cap_feats_2` measures 7.73e-1) plus the conditioner bounds (`cap_feats` at most 0.3, `cap_feats_2` 0.12, text states 3e-2), which a conditioner with the text router on the query tokens exceeds (`cap_feats` 4.59e-1). The end-to-end PNG is recorded, and the pooled latents get only a loose sanity bound (0.6).

**Alpha.** The `rgba` case is gated end to end: PNG at least 30 dB, alpha channel at least 30 dB, pooled latents within 6e-2. On both RGBA prompts tried (the apple icon and an RGBA poster) upstream's own output is opaque in practice, on bf16 and 4-bit alike: no pixel has alpha below 16, and the minimum alpha is 227 to 252. The gate is therefore that the port writes a real alpha channel and that it matches upstream's, not that the background is transparent.

`tests/fixtures/imagegen/ming_image_parity.safetensors` (templated ids, noise checksum and head, 8x8-cell pooled final latents, 32x32-pixel pooled RGBA images) and `ming_image_tiny_refs.safetensors` (a tiny transformer's output against upstream's, with and without `cap_feats_2`) are regenerated by `tests/fixtures/generate_ming_image_fixtures.py`. The real-checkpoint tests are `#[ignore]`:

```bash
MLX_ENABLE_TF32=0 <lib test binary> --ignored --test-threads=1 imagegen::ming_image::parity_tests
```

They read `models/mlx/ming-image-0.1-design` and `models/mlx/ming-image-0.1-design-mlx-4bit`. `MLXCEL_MING_IMAGE_ORACLE_DIR` (the generator's `--out` directory) adds the full-resolution PNG, alpha and latent comparisons, the conditioner and per-layer diagnostics, the wrong-router arm and the run on upstream's conditioning; `MLXCEL_MING_IMAGE_OUT` saves the PNGs.

## LLaDA-Image (`llada_image`)

`src/imagegen/llada_image/` ports `mlx_vlm/models/llada_image` (mlx-vlm `73d366ec`) for `inclusionAI/LLaDA-Image-Turbo` and `inclusionAI/LLaDA-Image` in the bf16 Diffusers layout (`model_index.json` with `_class_name: "LLaDAImagePipeline"`, `text_encoder/`, `queryformer/`, `text_projection/`, `sigvq/`, `transformer/`, `vae/`, `scheduler/`, `tokenizer/`). The probe picks `llada_image` from the pipeline class name. A missing component is named at load, and every weight's shape and dtype is checked against its config.

The prompt is encoded by the in-tree LLaDA-2 MoE decoder (`Llada2MoeModel`, 20 layers, 256 experts, 8 per token) over the word embeddings followed by QueryFormer queries, and the QueryFormer positions of the final hidden states go through the text projection to 2560 caption features. The denoiser is a 30-layer, 3840-wide single-stream DiT built from the shared Z-Image blocks (`dit_blocks`) with 3-axis RoPE and two-layer noise, context and SigVQ refiners; VQ and edit requests add 4096-wide SigVQ semantic features to the caption. Latents are FLUX.2 VAE latents (128 channels, 16x downsampling), decoded and, for edits, encoded by the shared `Flux2Vae`.

| Checkpoint | Steps / guidance | Schedule |
|------------|------------------|----------|
| `llada-image-turbo` | 4 / 1.0 | `linspace(1, 0, steps + 1)` in f32, shift 3, stochastic: every step re-noises from the predicted clean latent |
| `llada-image` | 50 / 5.0 | the nonlinear `1 - (1 - (1 - s^1.17)^0.8)^1.1` over `linspace(0.001, 1, steps + 1)[:-1]` built in f64 and rounded to f32, shift 1, Euler steps |

- **Modes.** `generation_mode` `text` (default) conditions on the caption only. `vq` first runs the LLaDA-2 decoder as a discrete image prior: block diffusion over 32-token blocks, 8 greedy steps per block, CFG 2, reveal above confidence 0.95 or else the schedule's top count; SigVQ embeds the generated ids as semantic features (1024 tokens at 512x512). Editing (`--task edit`, `/v1/images/edits`, exactly one source image) encodes the source to VAE latents and, at half resolution through SigVQ's 40-block ViT, to semantic features; `generation_mode` must be `editing` or absent there.
- **Sizes.** Width and height are positive multiples of 16, and of 32 for edits. An edit without a size scales the source toward 1024x1024 pixels of area (never up) and floors each side to 32; with one side given the other keeps the aspect ratio. An edit runs the denoiser over the source and target grids, `2 * (H / 16) * (W / 16)` image tokens per forward; that total is limited to 65536 (`MAX_EDIT_IMAGE_TOKENS`, the Mage-Flow value; mlxcel bound, upstream has none), about 2896x2896 pixels of area, and a larger edit is `400` naming the computed count and the limit before anything loads. The prompt is at most 2048 tokens: text and edit prompts are truncated there, as upstream does, and a VQ prompt whose prior input (system template, prompt and image header) passes 2048 tokens is `400` before anything loads (mlxcel bound; upstream has none, and the prior re-runs that whole window without a KV cache). The source image is used as decoded: upstream applies `ImageOps.exif_transpose` first, which mlxcel's image loading does not do for any family.
- **Guidance.** CFG runs above guidance 1. The negative prompt is the request field, else the `negative_prompt` option, else empty.
- **Options** (`--gen-kwargs` or extra JSON fields): `generation_mode`, `stochastic_sampling` (overrides the scheduler's setting) and `negative_prompt`. A malformed option, size or mode is `400` on the server; an evaluation failure inside the model is `500`.
- **Noise.** `mx.random.seed(seed)` and then upstream's draws in order (the initial latents, then one draw per stochastic re-noise), so every draw is bit-identical to upstream.
- **Memory.** Components load and unload as upstream's pipeline does: the text encoder (31 GB in bf16) is dropped after the prompt is encoded and before the transformer (13 GB) loads, and the transformer is dropped before the VAE decodes. A served model keeps up to two projected prompts across requests, so a repeated prompt skips the text encoder.

### LLaDA-Image precision

Upstream's text encoder subclasses its LLaDA-2 MoE decoder and changes the numerics, and the image prior's tokens follow the MoE routing, which flips on one-ulp differences. The port reproduces each change behind an opt-in `Llada2Numerics::LladaImage` path; the default text-generation path is unchanged (teacher-forced logit trace on `llada2.0-mini-preview-4bit`, 256 tokens x 4 chunks: 0 of 1024 positions disagree, byte-identical traces, perplexity 7.6092 before and after). The image path rounds the normalized activation to bf16 before the RMSNorm gain, rounds the RoPE tables to the activation dtype, sorts the routed experts by descending score, weights the activation by the f32 routing score before the down projection and sums the expert outputs in f32.

In the denoiser, upstream's biased `nn.Linear` is one fused `addmm`; a separate matmul and bias add moved the first prediction to 0.18. `dit_blocks` takes an opt-in `BiasAdd::Fused` for this (LLaDA-Image only; Z-Image and Mage-Flow keep the separate add, and the fused form bypasses runtime LoRA). The rotated q and k are rounded to the input dtype as upstream does; keeping them f32 moves the first prediction to 8.4e-2. The RoPE inverse frequencies are computed on the device as upstream's `MRoPE` does (`MRoPE::compute_freqs_on_device`, LLaDA-Image only): the host `powf` rounds one of the 40 frequencies differently, which moved 2.8% of the image table entries and the first prediction by 6.0e-2. With both, every denoiser stage fed upstream's input to it is bit-identical (`transformer_stage_tests.rs`, which reads per-stage tensors from `tests/fixtures/generate_llada_image_denoiser_stages.py`), except the final projection on VQ 512: its `[1024, 3840] x [3840, 128]` bf16 `addmm` picks another GEMM kernel than mlx 0.32.4 (3.0e-3 rel-RMS, 38.6% of outputs, with its input bit-identical).

Two Turbo CLI runs with the same seed write byte-identical PNGs.

### LLaDA-Image parity

The oracle is upstream mlx-vlm `73d366ec` on mlx 0.32.4 with the CUDA backend (NVIDIA GB10), `MLX_ENABLE_TF32=0` on both sides, with two upstream defects corrected in the oracle (`tests/fixtures/generate_llada_image_oracle.py`, flags `--unsorted-gather` and `--fix-conv-in`):

- **`gather_mm` with `sorted_indices=True` returns wrong rows on mlx 0.32.4 CUDA** (rel-RMS 0.8 against the unsorted call or a per-row matmul, also with random weights; CPU f32 is correct). Upstream's `TextMoE` uses the sorted form, so its text encoder is wrong from the first MoE layer and its VQ prior generates different tokens. mlxcel's MLX build does not have this bug (its sorted path matches an f32 CPU reference to 6e-3), so the oracle calls `gather_mm` unsorted.
- **The VAE encoder's `conv_in` keeps the PyTorch layout** (the `_match_conv_layout` defect described under [Verification](#verification)). Upstream's VAE then round-trips the edit source at 18.0 dB, and 33.8 dB with the weight transposed; mlxcel loads it correctly. The uncorrected upstream base edit renders a recolored copy of the source collage; the corrected upstream and mlxcel both render a water-caustics texture for the same prompt, which is the checkpoint's output.

The prompt is a red panda scene, seed 42. The edit source is the checkpoint's `assets/edit_demo.jpg` at 1440x704 ("make it blue"). Turbo runs 4 steps at guidance 1, the base checkpoint 50 steps at guidance 5. "Upstream features" feeds the denoiser upstream's own caption and semantic features.

| Case | First prediction on upstream's inputs | Final latents rel-RMS / PNG PSNR | On upstream features | Bound (on upstream features) |
|------|---------------------------------------|----------------------------------|----------------------|------------------------------|
| Turbo text, 1024x1024 | 0 | 0.882 / 16.10 dB | 0 / 57.97 dB | 5e-3 / 50.0 dB |
| Turbo VQ, 512x512 | 3.00e-3 | 1.087 / 14.34 dB | 0.0454 / 44.79 dB | 0.08 / 40.0 dB |
| Turbo edit | 0 | 0.154 / 26.71 dB | 0 / 56.21 dB | 5e-3 / 50.0 dB |
| Base text, 1024x1024 | 0 | 0.623 / 18.02 dB | 0 / 57.94 dB | 5e-3 / 50.0 dB |
| Base edit | 0 | 0.547 / 22.24 dB | 0 / 58.45 dB | 5e-3 / 50.0 dB |

On upstream's features the text and edit cases reproduce upstream's final latents exactly; their PSNR is the VAE decode's residual. The 30 dB criterion is not met on mlxcel's own features: that gap is the text encoder, below.

Every initial and re-noise draw is bit-identical, the token ids match exactly, the text projection on upstream's hidden states is bit-exact (its and the QueryFormer's GELUs run the same compiled graphs MLX compiles `nn.gelu_approx` and `nn.gelu` into; the eager op chains measured 1.9e-3), and the caption features are at 1.1e-2 to 1.4e-2. The edit source latents are at 8.8e-3 and 973 of 990 SigVQ codebook ids match (the rest are near-ties on the unit sphere). The base VQ case has no oracle: upstream fails on it in mlx 0.32.4's CUDA graph code (`cudaGraphAddKernelNode ... invalid argument`); mlxcel's own base VQ run completes.

The end-to-end gap on mlxcel's own features enters through the text encoder, not the denoiser or VAE: on upstream's features text and edit are exact and VQ is at 4.5e-2. It is not a port defect. Teacher-forced (each layer fed upstream's own input to that layer), attention and the dense layer 0 are bit-exact in every layer, and every MoE layer routes every token to the same experts in the same order; only 0.6% of the MoE outputs differ, by one rounding (2e-4 rel-RMS per layer), because mlxcel's expert-sorted `gather_mm` accumulates in another order than the oracle's unsorted call. With the unsorted call all 20 layers are bit-exact on upstream's input, but the prompt still drifts: the QueryFormer's 28-row k and v projections (`addmm`) pick another GEMM kernel in this build's MLX than in mlx 0.32.4 (40 of 57344 values differ), and the routing amplifies that to 3e-2 at the last layer. Measured with the unsorted call (and the compiled GELUs): Turbo text 18.5 dB, Turbo edit 36.0 dB, VQ ids equal for the first 48 tokens (377 of 1024), at twice the VQ prior time (206 s against 101 s); the port keeps the sorted call, which upstream itself issues. The pipeline amplifies small differences, and upstream amplifies its own the same way: rounding upstream's image-embedder bias add separately moves its first prediction by 8.6e-2 and its PNG to 26.5 dB, perturbing its QueryFormer queries by 2^-9 moves its captions by 3.1e-3 and its PNG to 17.9 dB, and one bf16 ulp in layer 2 of its text backbone moves the final hidden states by 0.167 (mlxcel against upstream: 0.172), because the MoE routing flips. The VQ prior's ids therefore do not match upstream's exactly: 254 of 1024 are equal and the first 6 are identical. The hard gates sit on the stages that are tight (token ids, noise, projection at below 1e-4, the first two text-backbone layers teacher-forced at below 2e-5, which rejects FP32 RoPE tables at 5.1e-5, SigVQ embedding of upstream's ids at below 1e-2, VAE encoding, the first prediction at below 1e-3 on text and edit and 1e-2 on VQ); the end-to-end bounds sit between the measured port and the wrong-port controls:

| Control (Turbo, upstream features) | Text | VQ | Edit |
|------------------------------------|------|----|------|
| Seed 43 | 1.256 / 12.78 dB | 0.978 / 17.71 dB | 0.505 / 20.46 dB |
| Shift 1 instead of 3 | 0.791 / 18.02 dB | 0.669 / 22.20 dB | 0.230 / 25.37 dB |
| FP32 q and k after RoPE (first prediction) | 0.250 / 28.62 dB (8.43e-2) | 0.117 / 35.28 dB (6.69e-2) | 0.0152 / 48.85 dB (1.28e-2) |

Every control misses its case's end-to-end bound, and FP32 RoPE also misses the first-prediction bound on every case.

```bash
MLX_ENABLE_TF32=0 MLXCEL_LLADA_IMAGE_ORACLE_DIR=<oracle --out> \
    <lib test binary> --ignored --test-threads=1 imagegen::llada_image::parity_tests
```

The tests read `models/mlx/llada-image-turbo` and `models/mlx/llada-image` (`MLXCEL_LLADA_IMAGE_TURBO`, `MLXCEL_LLADA_IMAGE_BASE`); `MLXCEL_LLADA_IMAGE_OUT` saves the PNGs. No fixture is committed (the oracle stores full-resolution tensors per case); without `MLXCEL_LLADA_IMAGE_ORACLE_DIR` the tests skip every case.

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
