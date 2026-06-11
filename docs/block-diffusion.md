# Block-diffusion generation (DiffusionGemma)

DiffusionGemma (`google/diffusiongemma-26B-A4B-it`) is a text generation model
that uses block-diffusion rather than left-to-right autoregressive decoding. This
page explains the generation mechanism, the CLI flags that control it, measured
throughput, and current limitations.

## How it works

Autoregressive models predict one token at a time, each conditioned on all
previous tokens. DiffusionGemma decodes a *canvas* of tokens at once through
iterative denoising: the canvas starts as uniformly random token ids, and the
model refines the whole block each step until tokens reach stable, high-confidence
predictions and the step loop stops early or exhausts the step budget.

The generation loop works as follows. First, the prompt is encoded into a
read-only prefix stored in dense FP16 KV caches. Then, for each output block:

1. A canvas is initialized from random token ids (seeded by `--seed` for
   reproducibility) with length `min(max_canvas, max(remaining, min_canvas))`.
2. Up to `--max-denoising-steps` denoising steps run. Each step embeds the
   current canvas with a learned self-conditioning signal, attends bidirectionally
   within the canvas while attending causally to the prefix, and uses the sampler
   to accept new token predictions.
3. After each step the stable-and-confident early stop checks whether the canvas
   contents have been unchanged for `stability_threshold` consecutive steps. If
   so, the block is committed and the loop moves on.
4. The committed canvas is appended to the prefix and, if any output remains,
   the next canvas block begins.

### Samplers

Two per-step acceptance samplers are available via `--diffusion-sampler`:

- **`entropy-bound`** (default): For each canvas position, compute a per-token
  entropy. Sort positions by ascending entropy. Accept positions in that order
  until the running prefix sum of entropies exceeds `entropy_bound` (read from
  the checkpoint's `generation_config`). At least one position is always accepted.
  This reproduces the mlx-vlm reference `(cumsum - cummax) <= bound` rule.

- **`confidence-threshold`**: Accept all unrevealed positions where the top-1
  token probability exceeds `--diffusion-threshold` (default 0.9). If no
  position qualifies, the highest-confidence unrevealed position is accepted
  unconditionally so the loop always makes progress.

## CLI flags

All diffusion flags appear in the `mlxcel generate --help` output under the
`Diffusion Options` heading. Autoregressive models ignore them entirely.

| Flag | Default | Notes |
|------|---------|-------|
| `--max-denoising-steps N` | checkpoint `generation_config` (typically 48) | Maximum denoising iterations per canvas block. Lower values trade quality for speed. |
| `--diffusion-sampler entropy-bound\|confidence-threshold` | `entropy-bound` | Per-step token acceptance strategy. See above. |
| `--diffusion-threshold FLOAT` | `0.9` | Confidence threshold for `confidence-threshold` sampler. |
| `--diffusion-min-canvas-length N` | `64` | Shortest canvas for the generation tail. Prevents tiny final blocks. |
| `--diffusion-max-canvas-length N` | model `canvas_length` (typically 256) | Cap on the per-block canvas length. |
| `--diffusion-full-canvas` | off | Always allocate the model's full `canvas_length` per block. |
| `--seed N` | unset (nondeterministic) | Seeds MLX's global RNG, which also controls canvas initialization. Two runs with the same seed and temperature 0 produce identical output. |

## Throughput

Per-step cost dominates throughput. The stable-and-confident early stop fires
earlier when the model converges quickly, so tokens/sec varies with prompt and
output difficulty.

Measured on M1 Ultra with `diffusiongemma-26B-A4B-it-4bit`:

- ~20 tok/s for short answers (64-token canvas, converging in about 12 of 48 steps)
- ~8 tok/s for longer outputs requiring most of the 48-step budget on a 256-token canvas

Per-step time on the same hardware: 254 ms/step for a 64-token canvas (103% of
the mlx-vlm Python reference at 261 ms/step) and 651 ms/step for a 256-token
canvas (96% of the reference at 626 ms/step).

Model load: 13.5 GiB resident (4-bit quantized checkpoint, M1 Ultra).

## Reproducibility

Canvas noise is the main source of nondeterminism. Pass `--seed N` to seed MLX's
global RNG before generation begins. Two runs with the same seed, same prompt, and
`--max-tokens` produce the same commit stream at temperature 0. The seed also
affects the autoregressive sampling path, so it is useful for both model families.

The environment variable `MLXCEL_DIFFUSION_DEBUG_CANVAS=1` replaces all random
canvas initialization with a fixed deterministic pattern. This is intended for
cross-implementation parity testing against the Python reference; output is not
useful for normal generation. See [Environment variables](environment-variables.md)
for details.

## Image input

Pass one or more `--image <path>` flags to include images in the prompt. The flag is repeatable for multi-image prompts.

```
mlxcel generate -m google/diffusiongemma-26B-A4B-it \
  --image photo.jpg \
  -p "Describe what you see." \
  --max-tokens 256 \
  --seed 42
```

Preprocessing reuses the Gemma 4 vision pipeline: images are resized and padded to 768x768, then projected into 256 soft tokens per image. Each image block receives bidirectional attention during prefill, matching the same overlay used by Gemma 4 Unified. The image tokens are inserted into the prompt sequence between the `boi` and `eoi` sentinel tokens.

Video and audio inputs are not supported and are rejected with a clear error message.

## Remaining limitations (phase 2)

The following are not yet supported:

- **Server mode**: `mlxcel serve` and `mlxcel-server` reject DiffusionGemma at
  load time. Use `mlxcel generate` from the CLI. Server support is planned for phase 3 of issue #217.
- **Tensor parallelism**: The TP planner has a placeholder arm for this family.

## Usage example

```
mlxcel generate -m google/diffusiongemma-26B-A4B-it \
  -p "Why is the sky blue?" \
  --max-tokens 256 \
  --seed 42
```

For a 4-bit quantized snapshot in the model store:

```
mlxcel generate -m diffusiongemma-26B-A4B-it-4bit \
  -p "Explain TCP vs UDP in one paragraph." \
  --max-tokens 512 \
  --diffusion-sampler entropy-bound \
  --max-denoising-steps 32
```
