# Supported models

This page summarizes model-family support in the v0.0.27 source tree. The
runtime source of truth is the code, not this prose page:

- detection: `src/models/detection.rs`
- `ModelType` enum and module exports: `src/models/mod.rs`
- loading policy: `src/model_metadata.rs`
- VLM loading routes: `src/loading/vlm*.rs`

`ModelType` spans text and non-VLM language models, VLM variants, a
speech-to-text encoder-decoder (Whisper), and a text-to-speech model (Kokoro).
These are architecture/runtime variants, not a guarantee that every checkpoint
under a marketing family name is supported.

## Text and hybrid model families

Implemented model families include:

- Llama-family and Mistral-style dense decoders
- Llama 4 text
- Qwen 2 / 2.5 / 3 / 3.5, Qwen MoE, Qwen3 Next
- Gemma 1 / 2 / 3 / 3n / 4 text variants
- Phi, Phi-3, Phi-3 Small, PhiMoE
- Mixtral and other MoE families
- DeepSeek v1 / v2 / v3 / v3.2
- dots.llm1 (`dots1`, rednote: a DeepSeek-V3-style Mixture-of-Experts without MLA. Standard multi-head attention with per-head Q/K RMSNorm, a dense first layer (`first_k_dense_replace`), then sigmoid-routed experts that select on `gate.weight` logits plus an `e_score_correction_bias`, with a single always-on shared expert. Validated against `mlx-community/dots.llm1.inst-mixed-4-6bit`, a mixed 4/6-bit export whose `v_proj` and `down_proj` tensors are 6-bit while the rest are 4-bit; the unified loaders detect the per-tensor bit width from shape.)
- Mistral 4 (`mistral4`, Mistral Small 4: a DeepSeek-V3-style Multi-Latent Attention decoder with compressed query and KV LoRA projections (`q_lora_rank` 1024, `kv_lora_rank` 256), split rope/nope query-key head dims, and a separate value head dim, paired with a softmax-routed Mixture-of-Experts (128 routed experts plus one always-on shared expert, 4 active per token, `norm_topk_prob`) and Llama-4 position-dependent attention scaling. The only public checkpoint is the Mistral Small 4 119B vision model, whose `text_config.model_type` is `mistral4`; mlxcel detects it as a Mistral 3 VLM (Pixtral vision tower) on the Mistral4 text backbone. Validated against `mlx-community/Mistral-Small-4-119B-2603-4bit` for both text and image-plus-text.)
- Cohere / Cohere2
- InternLM 2 / 3
- GLM 4, GLM MoE, GLM MoE DSA
- ERNIE 4.5 and ERNIE 4.5 MoE
- Hunyuan dense and MoE variants
- IBM Granite dense (`granite`)
- IBM Granite 4.x hybrid (`granitemoehybrid`: interleaves Mamba2 SSM and GQA attention layers by `layer_types`, applies the four Granite scalar multipliers (embedding, attention, residual, logits), and defaults to NoPE attention. The dense-MLP mode is validated against `mlx-community/granite-4.0-h-350m-4bit`; the MoE mode (`block_sparse_moe` + `shared_mlp`) is implemented but awaits a public MLX checkpoint to validate. The non-hybrid `granitemoe` variant is not yet ported.)
- BitNet b1.58 (`bitnet`, Microsoft: a Llama-style transformer whose every projection is a `BitLinear` with 1.58-bit ternary weights ({-1, 0, +1}) packed 4-per-uint8 and scaled by a single per-tensor `weight_scale`. A custom Metal kernel (`bitlinear_matmul`) multiplies directly on the packed bytes, so the unpacked weights never materialize. Two extra sub-norms (`attn_sub_norm` before `o_proj`, `ffn_sub_norm` inside the MLP) and a squared-ReLU MLP (`relu2(gate) * up`). Runs in native bf16 (its squared-ReLU overflows f16), bypassing the Apple-Silicon f16 conversion. Validated against `mlx-community/bitnet-b1.58-2B-4T` and its `-4bit` variant, which additionally affine-quantizes the embedding/lm_head to 4-bit (the BitLinear weights stay ternary); keeping the whole model bf16 also keeps that 4-bit dequant dtype-consistent.)
- ExaOne / ExaOne 4 / ExaOne MoE / Solar Open
- OLMo / OLMo2 / OLMo3 / OLMoE
- StarCoder2, StableLM, SmolLM3, Baichuan, MiniCPM, MiniCPM3, MiniMax,
  Ministral3, Nemotron, Nemotron-NAS, Step 3.5, MiMo
- Mellum / Mellum 2 (`mellum`, JetBrains code model: sliding/full hybrid attention driven by `layer_types`, with QK-RMSNorm and a sparse softmax-routed MoE (`norm_topk_prob`) in every layer. Full-attention layers use YaRN-scaled RoPE plus a standard `KVCache`; sliding-attention layers use default RoPE plus a `RotatingKVCache` windowed to `sliding_window`. Supports tied and untied LM heads. Validated against `JetBrains/Mellum2-12B-A2.5B-Base`.)
- Apertus (`apertus`, Swiss AI: Llama-style dense transformer with an xIELU activation MLP (no gate), QK-norm, llama3 RoPE scaling, and untied embeddings)
- Seed-OSS (`seed_oss`, ByteDance: plain Llama-style dense transformer with a standard SwiGLU MLP and standard residuals. The only deltas are a split attention bias (`attention_bias` on q/k/v, `attention_out_bias` on o_proj), an explicit `head_dim`, untied embeddings, and a `{"rope_type": "default"}` rope_scaling that applies no scaling. Validated against `mlx-community/Seed-OSS-36B-Instruct-4bit`.)
- Mamba, Mamba2, RWKV7, Recurrent Gemma, Jamba, Nemotron-H
- Falcon-H1 (TII: runs a Mamba2 SSM mixer and GQA attention in parallel within each block, summing both outputs; the MUP channel multipliers are pre-folded into the MLX weights)
- LFM2 and LFM2-MoE (Liquid Foundation Models: short-convolution and attention hybrid; the MoE variant routes through sigmoid-gated experts)
- PLaMo 2 (Preferred Networks: interleaves Mamba SSM and GQA attention layers by index; each block carries normformer-style pre/post offset RMSNorms, and the Mamba mixer derives B/C/dt from a post-conv projection). The architecture is validated against the mlx-lm reference at the token-id level (`tests/plamo2_parity.rs`). CLI text generation additionally needs support for PLaMo's custom `PlamoTokenizer` (the `tokenizer.jsonl` Unigram format), which the Rust tokenizer loader does not yet read.
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
- Gemma 4 Unified (`gemma4_unified`): encoder-free text + image + audio + video. Patch-projection vision embedder and waveform-chunk audio path feed the shared Gemma 4 backbone, with blockwise bidirectional attention over image/video token spans during prefill. Video is handled as images-per-frame: frames are extracted with `ffmpeg` (uniform sampling, default 2.0 fps), patchified through the same vision embedder with a per-frame `vision_soft_tokens_per_video_frame` budget (70), and scattered into `video_token_id` placeholder spans. Video is available on both the CLI (`--video`) and the server (`video_url` content blocks). The prompt grows by ~70 soft tokens per sampled frame, so a multi-second clip at the default fps expands past the model's 1024-token sliding window. That over-window single-pass prefill is handled correctly: the windowed prefill mask spans the full prompt (the rotating cache keeps every prefill key and only trims to the window for the decode step), so long clips decode coherently without lowering `--fps`.
- Gemma 4 VL audio: the Gemma 4 VL checkpoints that ship a Conformer audio tower (for example the `e2b` / `e4b` instruct models) take spoken audio from the CLI with `--audio <path>` and transcribe or answer questions about it. Input audio is resampled to 16 kHz before the Conformer encoder, so a clip at any source rate produces the encoder frame count the duration-based audio-token budget expects. Server-side `input_audio` in `POST /v1/chat/completions` is also supported: the audio block is spliced inside the last user turn, before the `<turn|>` end-of-turn marker that Gemma 4 uses (id 106).
- Llama 4 VLM
- LLaVA and LLaVA-Bunny
- Aya Vision and PaliGemma
- Pixtral and Mistral 3 VLM wrappers (Mistral 3 VLM supports both the standard Llama/Mistral text backbone and the Mistral4 MLA+MoE backbone; text and image-plus-text are validated on both, including the Mistral Small 4 119B checkpoint)
- Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3.5-VL, and Qwen3-VL MoE
- GLM-4V (`glm4v`): GLM-4V ViT vision tower (3D patch embedding, bilinear-resampled learned position embeddings, Conv2d spatial downsample, SwiGLU patch merger) plus a GLM-4 text backbone driven by sectioned even/odd MRoPE. Reuses the shared Qwen-VL image processor and prompt/token plumbing.
- GLM-4V MoE (`glm4v_moe`): GLM-4.5V-class variant reusing the GLM-4V ViT vision tower with a GLM-4 MoE text backbone (grouped `noaux_tc` routing, shared experts, `first_k_dense_replace` dense layers) driven by sectioned half-split MRoPE. Reuses the GLM-4 MoE machinery and the shared Qwen-VL runtime.
- Youtu-VL
- SmolVLM and SmolVLM2 (`smolvlm`): Idefics3-family SigLIP vision tower + pixel-shuffle token compression + SmolLM2 text backbone. Images are tiled into square patches with a global thumbnail tile, and each `<image>` placeholder expands to `num_image_token` compressed feature tokens per tile.
- MiniCPM-O
- Moondream 3
- Phi-3 Vision, Phi4MM, Phi4 SigLIP VLM
- Molmo2 and Molmo-Point
- Nemotron-H Nano Omni: ships a Conformer/Parakeet audio encoder and accepts spoken audio from the CLI with `--audio <path>`. Input audio is resampled to 16 kHz before the encoder. Server-side `input_audio` in `POST /v1/chat/completions` is also supported: the audio block is spliced inside the last user turn, before the `<|im_end|>` end-of-turn marker that the ChatML template uses (id 151 in the released checkpoint).

Audio/video capability is model-specific. The server request types include
`image_url`, `video_url`, and `input_audio` content blocks, but a loaded model
must advertise support for the corresponding modality. Video frame extraction
uses the system `ffmpeg`/`ffprobe` binaries at runtime.

### Thinking default for `gemma4_unified`

Gemma 4 Unified (`gemma4_unified`) ships `<|channel>` / `<channel|>` thinking
markers in its tokenizer, so the server defaults `enable_thinking=true` for this
family on startup, mirroring [ml-explore/mlx-lm#1114](https://github.com/ml-explore/mlx-lm).
With thinking on, the model writes an internal scratchpad before the visible
reply, so simple prompts spend more of the budget on reasoning than on the
answer. A one-sentence answer can take roughly 275 completion tokens, and a
default `max_tokens` of 64 to 80 may return an empty `content` with
`finish_reason` of `length` because the whole budget went to thinking. Set
`max_tokens` to at least 512 for this family, and higher for multi-sentence
answers.

The scratchpad is no longer dropped: it is surfaced as `reasoning_content` on
both streaming responses (`delta.reasoning_content`) and non-streaming responses
(a `reasoning_content` field on the assistant message, present only when the
model produced reasoning). This applies to every thinking family, including
Qwen-style `<think>` models. To turn thinking off, pass
`chat_template_kwargs={"enable_thinking": false}` per request, or set the server
default via `--chat-template-kwargs` or `LLAMA_ARG_CHAT_TEMPLATE_KWARGS`. A
per-request value always wins over the server default.

### `thinking` alias for the DeepSeek-V3.2 (`deepseek_v32`) chat template

The upstream `deepseek-ai/DeepSeek-V3.2-Exp` chat template gates its
`<think>` block on a bare `thinking` boolean rather than the conventional
`enable_thinking` kwarg that mlxcel forwards by default (verified against
[deepseek-ai/DeepSeek-V3.2-Exp `assets/chat_template.jinja`](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/assets/chat_template.jinja)).
Because the template only ever reads `thinking`, `enable_thinking` alone would
have no effect on this family. mlxcel detects the `{% if not thinking is
defined %}` idiom in the loaded template and, when present, also mirrors the
fully-resolved `enable_thinking` value (request override, or the server
default) into a `thinking` key so toggling reasoning works the same way it
does for every other thinking-capable model. The detection is based on the
template source, not the model name, so any future template that adopts the
same idiom is covered automatically. GLM-5.2 (`glm_moe_dsa`) ships its own
chat template that already reads `enable_thinking` directly and is
unaffected. An explicit `thinking` entry in `chat_template_kwargs` always
overrides the derived alias.

### Loop-detection default-on for the Gemma 4 family

Gemma 4 (31B Dense and 26B-A4B MoE, including the QAT 4-bit checkpoints) has an upstream, weights-level token-repetition collapse documented in [google-deepmind/gemma#622](https://github.com/google-deepmind/gemma/issues/622) and reproduced across other engines, including [vllm-project/vllm#40080](https://github.com/vllm-project/vllm/issues/40080). Generation can degenerate into a single repeated token (or short fragment) that fills the token budget, most often inside the thought channel. Tool declarations and `json_schema` structured-output requests amplify it, but it is not limited to those cases. Sampling penalties do not reliably recover it, because once the logits collapse the top-k candidates are themselves garbage.

mlxcel applies vLLM-style N-gram loop detection to break this, default-on for the Gemma 4 family with no configuration required. For any model in the family (`Gemma4`, `Gemma4VLM`, `Gemma4Unified`), the engine applies the conservative threshold `min_pattern_size=1, max_pattern_size=20, min_count=4` by default; a degenerate run then ends early with `finish_reason` of `stop`. The default-on applies to plain Gemma 4 chat too, so a downstream serving app and its users need no setup and see no toggle. Detection only ends generation when a real repetition loop is present, so this conservative default is low risk. Every non-Gemma-4 model defaults to disabled (bit-exact baseline preserved). The behavior is still tunable on top of the default: a per-request override (the vLLM `max_pattern_size` / `min_pattern_size` / `min_count` fields, including `max_pattern_size=0` to opt out) wins, and a global operator override (`MLXCEL_LOOP_DETECTION`) can tune, force-enable for any model, or force-disable. See [Generation loop detection](environment-variables.md#generation-loop-detection-issue-432) for the field semantics and the full precedence order.

## Speech-to-text (ASR)

mlxcel loads Whisper-style encoder-decoder ASR checkpoints (`model_type: "whisper"`) and serves them through the OpenAI audio endpoints. A convolutional audio encoder builds features from a 30-second log-mel window, and an autoregressive text decoder cross-attends to those features as it emits tokens, steered by the multilingual transcribe/translate task tokens.

When the server's loaded checkpoint is detected as Whisper, the speech-to-text slot is populated and `POST /v1/audio/transcriptions` (transcribe in place) and `POST /v1/audio/translations` (translate to English) return the recognized text. Uploaded audio is decoded with the shared WAV reader, resampled to 16 kHz, and processed in consecutive 30-second windows. An explicit `language` hint is honored; otherwise the language is detected from the first decoder step. Token suppression follows the Whisper rules: `suppress_blank`, the non-speech symbol set, and `<|notimestamps|>`.

This first port targets non-quantized (fp16/f32) checkpoints with greedy decoding, and the loader accepts both the native MLX and HuggingFace key layouts. Loading a Whisper checkpoint serves speech-to-text only; chat generation is not available in the same process. Beam search, word-level and segment timestamps, quantized checkpoints, and streaming transcription are follow-ups.

## Text-to-speech (TTS)

mlxcel loads the Kokoro-82M model (a StyleTTS2 phoneme-to-mel acoustic model with a built-in iSTFTNet vocoder) and serves it through `POST /v1/audio/speech`. The path is: text to phonemes (grapheme-to-phoneme front-end) to a PLBert text encoder, a duration predictor that expands per-token features to per-frame, F0 (pitch) and energy prosody, and an iSTFTNet decoder that produces a 24 kHz mono waveform directly via an inverse STFT (no separate neural codec).

Detection works without a top-level `model_type`: the loader recognizes a Kokoro checkpoint by the `istftnet` config block or the `kokoro-v1_0.safetensors` weight filename, so `-m <kokoro-dir>` resolves to the TTS provider. The `voice` request field selects a pack from `voices/<name>.safetensors` (54 voices; default `af_heart`), validated against the available packs with a safe fallback. `speed` scales the predicted durations (larger is faster and shorter). `response_format` accepts `wav` today (returned via the shared WAV writer); other containers are a follow-up.

The grapheme-to-phoneme front-end is a self-contained American-English phonemizer: text is normalized (lower-cased, integers spoken, common punctuation kept), each word is looked up in a bundled lexicon, and out-of-vocabulary words fall back to deterministic letter-to-sound rules. It emits the IPA symbols in Kokoro's vocab and needs no external binary or download. Non-English voices in the checkpoint still load and synthesize, but their phonemes come from the English front-end, so pronunciation quality is limited; per-language g2p (the analogue of upstream Kokoro's `misaki[xx]` packages) is future work. Like Whisper, the model loads and runs every synthesis on one dedicated MLX worker thread, so loading a Kokoro checkpoint serves text-to-speech only.

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
