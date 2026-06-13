# Supported models

This page summarizes model-family support in the v0.0.27 source tree. The
runtime source of truth is the code, not this prose page:

- detection: `src/models/detection.rs`
- `ModelType` enum and module exports: `src/models/mod.rs`
- loading policy: `src/model_metadata.rs`
- VLM loading routes: `src/loading/vlm*.rs`

As of v0.1.4, `ModelType` contains 93 variants: 71 text/non-VLM variants
and 22 VLM variants. These are architecture/runtime variants, not a guarantee
that every checkpoint under a marketing family name is supported.

## Text and hybrid model families

Implemented model families include:

- Llama-family and Mistral-style dense decoders
- Llama 4 text
- Qwen 2 / 2.5 / 3 / 3.5, Qwen MoE, Qwen3 Next
- Gemma 1 / 2 / 3 / 3n / 4 text variants
- Phi, Phi-3, Phi-3 Small, PhiMoE
- Mixtral and other MoE families
- DeepSeek v1 / v2 / v3 / v3.2
- Cohere / Cohere2
- InternLM 2 / 3
- GLM 4, GLM MoE, GLM MoE DSA
- ERNIE 4.5 and ERNIE 4.5 MoE
- Hunyuan dense and MoE variants
- IBM Granite dense (`granite`)
- ExaOne / ExaOne 4 / ExaOne MoE / Solar Open
- OLMo / OLMo2 / OLMo3 / OLMoE
- StarCoder2, StableLM, SmolLM3, Baichuan, MiniCPM, MiniCPM3, MiniMax,
  Ministral3, Mistral4, Nemotron, Nemotron-NAS, Step 3.5, MiMo
- Mamba, Mamba2, RWKV7, Recurrent Gemma, Jamba, Nemotron-H
- LFM2 and LFM2-MoE (Liquid Foundation Models: short-convolution and attention hybrid; the MoE variant routes through sigmoid-gated experts)
- Kimi Linear, LongCat Flash, LongCat Flash N-gram
- GPT-OSS

Many of these families have checkpoint-specific config or weight-layout
requirements. If a checkpoint fails detection or loading, inspect its
`config.json::model_type` first and compare it with `src/models/detection.rs`.

## Block-diffusion text models

| Family | `model_type` key | Notes |
|--------|-----------------|-------|
| DiffusionGemma | `diffusion_gemma` / `diffusion_gemma_text` | Block-diffusion on a Gemma 4 MoE backbone. Generates a canvas of tokens per block through iterative denoising rather than token-by-token left-to-right decoding. CLI (`mlxcel generate`) supports text and image input (`--image <path>`, repeatable). Served in `mlxcel-server` (serial, batch-1 by design) via `/v1/chat/completions` and `/v1/completions`; image input follows the standard `image_url` content part format. See [Block-diffusion generation](block-diffusion.md). |

DiffusionGemma uses a two-phase forward pass: an encoder prefill that caches the
prompt into dense FP16 KV caches, then a canvas loop that attends bidirectionally
within each output block while attending causally to the cached prefix.
Load detection accepts `model_type: "diffusion_gemma"` (outer config) and
`model_type: "diffusion_gemma_text"` (inner `text_config`).
The fused MoE `gate_up_proj` weights are split at load time. When a vision tower is present in the checkpoint, its weights are loaded and wired for image input; checkpoints without vision weights fall back to text-only mode without error.

## Vision-language and multimodal variants

Implemented VLM variants include:

- Gemma 3 VL, Gemma 3n VL, Gemma 4 VL
- Gemma 4 Unified (`gemma4_unified`): encoder-free text + image + audio. Patch-projection vision embedder and waveform-chunk audio path feed the shared Gemma 4 backbone, with blockwise bidirectional attention over image/video token spans during prefill. Video input is not yet supported.
- Llama 4 VLM
- LLaVA and LLaVA-Bunny
- Aya Vision and PaliGemma
- Pixtral and Mistral 3 VLM wrappers
- Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3.5-VL, and Qwen3-VL MoE
- Youtu-VL
- MiniCPM-O
- Moondream 3
- Phi-3 Vision, Phi4MM, Phi4 SigLIP VLM
- Molmo2 and Molmo-Point
- Nemotron-H Nano Omni

Audio/video capability is model-specific. The server request types include
`image_url`, `video_url`, and `input_audio` content blocks, but a loaded model
must advertise support for the corresponding modality. Video frame extraction
uses the system `ffmpeg`/`ffprobe` binaries at runtime.

## Quantization formats

| Format | Status | Notes |
|--------|--------|-------|
| FP16 / BF16 | supported | BF16 handling is platform/model dependent; Apple Silicon paths commonly convert to FP16 for execution. |
| 4-bit affine MLX checkpoints | supported | Primary path for many `mlx-community` checkpoints. CUDA coverage depends on MLX kernel support for the target GPU. |
| 8-bit affine | supported | Used for weights and/or KV cache depending on path. |
| NVFP4 / MXFP4 / MXFP8 | supported where implemented | Used by specific families such as GPT-OSS and recent quantized checkpoints. |

Do not infer quality or speed from the ability to load a quantized checkpoint.
Run a smoke test and, for release claims, a benchmark/quality gate.

## Distributed support summary

| Capability | Current summary |
|------------|-----------------|
| Tensor parallelism | Advertised for selected dense text families such as Llama, Qwen, Gemma text, ERNIE 4.5, and Hunyuan dense. Validate per model/rank count. |
| Pipeline parallelism | Best validated for Llama-family text models; stage executors exist for more families with less operator coverage. |
| VLM under TP/PP | Partial. Vision tower / projector partitioning is not uniformly supported. |
| Disaggregated inference | Infrastructure exists; validate per topology and workload. |

## Speculative decoding

| Drafter | Target families | Notes |
|---------|-----------------|-------|
| MTP | Gemma 4 target paths | Available through shared speculative decoding flags. |
| DFlash | Qwen 3.5 text/VLM paths | Available through shared speculative decoding flags. |

Use auto-detection by default. Override only when you know the target and drafter
checkpoint pair are compatible.

## Known non-goals / caveats

- A supported architecture does not imply every community checkpoint variant is
  supported.
- VLM and video/audio paths require additional runtime dependencies and prompt
  preparation beyond text-only generation.
- TurboQuant, TP, PP, and speculative decoding are not uniformly validated for
  every family.
- The `mlxcel arch` output is a CLI summary and may lag the detailed enum count;
  the canonical source remains `src/models/mod.rs` and `src/models/detection.rs`.

## Adding support

See [Adding a new model](adding-models.md) for the registration, loading, and
test checklist.
