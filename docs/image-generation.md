# Image generation

mlxcel generates and edits images through `mlxcel generate-image` and the OpenAI-compatible `POST /v1/images/generations` and `POST /v1/images/edits` server routes. This page covers the model-independent runtime in `src/imagegen/`: the request types, the checkpoint probe, the scheduler, latent packing, the FLUX.2 VAE, the Z-Image DiT blocks, PNG output, the CLI, the server routes, and how a model issue registers a family.

**Current status.** One family is registered: `z_image` ([Z-Image](#z-image)). A checkpoint of any other recognized family fails before its weights load:

```text
$ mlxcel generate-image -m models/FLUX.2-klein -p "a fox"
Error: image generation family flux2 is not supported by this build
```

Each model in the image-generation epic (`lablup/mlxcel#2348`) adds its transformer and text conditioner and one row to the family table.

| Family | Checkpoints | Tasks | Default steps / guidance |
|--------|-------------|-------|--------------------------|
| `z_image` | Tongyi-MAI Z-Image-Turbo, Z-Image (raw Diffusers tree or an mlx-vlm conversion) | generate, edit (img2img) | Turbo 8 / 0; base 50 / 4 |

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
| `--size WxH` | Default `512x512` for generation and the first reference image's size for editing. Both sides must be multiples of the family's grid (16 for most families). |
| `--steps N`, `--guidance F` | Model defaults apply when omitted. |
| `--seed N` | Default is a random 32-bit seed; the seed is always printed. |
| `-n N` | 1 to 10 images. Image `i` uses seed `seed + i`. |
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

Without `--image-model` both routes answer `501`, naming the flag.

### Request

`generations`: `prompt`, `model`, `n` (1 to 10), `size` (`WxH`) or `width` plus `height`, `steps`, `seed`, `guidance`, `negative_prompt`, `response_format`, `output_format`. `edits` adds `image`, one string or a list. Any other field is passed to the model as `extra`.

- `width` and `height` come together or not at all (otherwise `400`) and override `size`.
- `size` defaults to `512x512` for generation and to the first reference image's size for editing. Width and height must be multiples of the family's grid; the error names the multiple.
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

## Shared building blocks

| Module | Contents | Used by |
|--------|----------|---------|
| `pipeline` | `ImageRequest`, `GeneratedImage`, `ImageSamplingDefaults`, the `ImageGenerationModel` trait. | every family |
| `probe` | `detect_image_family`. | every family |
| `registry` | `FAMILIES`, `ImageFamily`, `resolve_family`, `load_image_model`. | every family |
| `output` | `encode_png`, `to_b64_png`, `indexed_output_path`. | CLI, server |
| `run` | `validate_request`, `run_request`, `generate_to_files`, `parse_size`, `random_seed`. | CLI, server |
| `scheduler` | `FlowMatchEulerDiscrete::dynamic` (exponential shift, optional `shift_terminal`) and `::static_shift`. | QwenImage, MingImage; ZImage, Mage-Flow |
| `latent` | `patchify_latents`, `unpatchify_latents`, `pack_latents`, `unpack_latents`. | ErnieImage, LLaDAImage |
| `vae` | `Flux2Vae`, `Flux2VaeConfig`: decode, packed-latent decode, optional encode (the `quant_conv` is optional, absent in a FLUX.1 `AutoencoderKL` with `use_quant_conv: false`). | ErnieImage, LLaDAImage, ZImage |
| `qwen3_text_encoder` | `Qwen3TextEncoder`: hidden states of every layer (`num_layers + 1`, index 0 the embeddings, before the final norm) and `stacked_prompt_embeds` (`[B, T, n * H]`, FLUX.2 order). Built on `Qwen3Model::forward_all_hidden`. | FLUX.2, Z-Image, Ideogram 4, Bonsai |
| `dit_blocks` | `TimestepEmbedder`, `FeedForward`, `MRoPE`, `apply_rotary`, `Attention`, `ZImageTransformerBlock`, `FinalLayer`, `sanitize_transformer_weights`. | ZImage, LLaDAImage, MingImage |

Not in this foundation: every model transformer and text conditioner, the Qwen-Image VAE, the FLUX.2 tiled VAE decode, multipart uploads, streaming partial images, video generation. The FLUX.2 empirical-`mu` scheduler belongs to the FLUX.2 issue.

### Conventions a model must follow

- **Seeds.** Image `i` of `n` uses `seed + i`. mlxcel links the same MLX C++ RNG as upstream, so the same seeding call (`mx.random.seed(seed)` in Qwen-Image and LLaDA-Image, `mx.random.key(seed)` in ERNIE-Image and Ming-Image) followed by the same `normal` shape and dtype gives the same initial noise; keep upstream's draw order so whole-run parity tests can compare.
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
| Z-Image DiT blocks | follows the checkpoint dtype | The blocks are dtype-agnostic. Rotation, SDPA inputs and residual adds cast to upstream MLX's promotion (bf16 or f16 with f32 is f32), because the CUDA build of mlxcel's MLX promotes bf16 with f32 to bf16. |
| Z-Image transformer and VAE | f16 weights, f32 latent stream | See [Z-Image precision](#z-image-precision). |

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

The final-latents drift is the size of upstream's own sensitivity to how the bf16 caption stream is rounded. On the Turbo case, upstream against itself moves by 2.7e-2 (41.3 dB) when only its caption embedder's linear runs in f32, by 1.0e-1 (32.1 dB) when only its text encoder's attention runs in bf16 instead of f32, and by 9.5e-2 (31.6 dB) with TF32 on. The shared Qwen3 conditioner runs attention in bf16 where upstream's encoder casts q, k and v to f32; its caption features differ from upstream's by 9.3e-3, and feeding upstream's caption features to this port's denoiser lowers the Turbo drift to 3.4e-2. The committed summary `tests/fixtures/imagegen/z_image_parity.safetensors` (noise checksums, 4x4-pooled final latents and images, regenerated by `tests/fixtures/generate_z_image_fixtures.py`) backs the `#[ignore]` tests:

```bash
MLX_ENABLE_TF32=0 <lib test binary> --ignored --test-threads=1 imagegen::z_image::tests::
```

`MLXCEL_Z_IMAGE_TURBO` and `MLXCEL_Z_IMAGE_BASE` point at the checkpoints (default `models/mlx/z-image-turbo-bf16`, `models/mlx/z-image-bf16`); `MLXCEL_Z_IMAGE_ORACLE_DIR` adds the full-resolution comparison and the img2img case.

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
