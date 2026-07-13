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
- Command MoE (`cohere2_moe`, Cohere: the Cohere2 parallel-residual backbone (interleaved sliding/global attention, conditional RoPE, logit scaling, tied embeddings) with the dense FFN replaced by a sparse Mixture-of-Experts. The router scores on `mlp.gate` logits activated in f32 by sigmoid (default) or softmax, takes the top-k, and gathers the activated values at the selected experts (not a fresh softmax over the k logits), with optional `norm_topk_prob` renormalization clamped by `1e-12`. Always-on shared experts (`moe_num_shared_experts`) run a dense SwiGLU on the same input and combine by average (default `(y + y_s) / 2`) or sum. An optional dense-FFN prefix (`first_k_dense_replace`, width `prefix_dense_intermediate_size`) and an optional RMSNorm mode (`rms_norm_eps`, else LayerNorm) round out the deltas versus dense `cohere2`. The full runtime (detection, metadata, loaded-model dispatch, CLI and server generation) is wired; real-checkpoint validation is pending a public `cohere2_moe` checkpoint.)
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
- LFM2-VL (`lfm2_vl` / `lfm2-vl`): a SigLIP2-style packed-patch vision tower (native variable resolution, per-image bicubically-resampled position grid) + a pixel-unshuffle (space-to-depth) projector into the LFM2 hybrid text backbone. Each image is smart-resized so its post-downsample token count lands in `[min_image_tokens, max_image_tokens]`, packed at its native patch count (no padding), and its `ceil(h/f)*ceil(w/f)` projected tokens replace the `<image>` placeholder. The multi-tile `do_image_splitting` high-resolution path is a documented follow-up.
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
| LLaDA-2 MoE | `llada2_moe` | Masked-diffusion LM with a DeepSeek-V3-style MoE FFN (sigmoid-routed grouped experts on `gate.weight` + `expert_bias`, one always-on shared expert, `first_k_dense_replace` dense layers). Decoder-only transformer with fully bidirectional attention (no causal mask), fused QKV projection, per-head QK-norm, and partial RoPE. Generates by semi-autoregressive block-wise unmasking of `<|mask|>` tokens: the prompt is committed into per-layer KV caches block by block, then each generation block is denoised by repeatedly forwarding it against the frozen prefix and revealing positions whose chosen-token probability clears a linearly decaying confidence threshold (at least one reveal per step). Text-only. Served in `mlxcel-server` (serial, batch-1 by design) on the shared diffusion worker loop via `/v1/chat/completions` and `/v1/completions`; `--max-denoising-steps` maps to the per-block step count. Validated against `inclusionAI/LLaDA2.0-mini` (20 layers, 256 experts). |

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
- Llama 3.2 Vision (`mllama`): a Llama-3 text backbone whose layers at `cross_attention_layers` (e.g. `[3, 8, 13, 18, 23, 28, 33, 38]`) are gated cross-attention adapters attending to a tiled ViT tower. The image processor picks an optimal tile arrangement (up to `max_num_tiles`, default 4) from the supported aspect ratios, resizes and pads each image into `560x560` tiles, and emits aspect-ratio ids/masks. The tower adds gated tile + position embeddings, runs a local then a global (gated) transformer, and concatenates a set of intermediate hidden states; a `multi_modal_projector` maps those `vision_output_dim` features into the text hidden size. Unlike the LLaVA-style VLMs, image features are not merged into the token stream: they are held as `cross_attention_states` and consulted through the gated cross-attention layers (with per-head `q_norm`/`k_norm`). Text-only prompts leave the cross-attention layers as pass-throughs.
- LLaVA and LLaVA-Bunny
- Aya Vision and PaliGemma
- Pixtral and Mistral 3 VLM wrappers (Mistral 3 VLM supports both the standard Llama/Mistral text backbone and the Mistral4 MLA+MoE backbone; text and image-plus-text are validated on both, including the Mistral Small 4 119B checkpoint)
- Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3.5-VL, and Qwen3-VL MoE
- PaddleOCR-VL (`paddleocr_vl`): document-OCR VLM pairing a NaViT dynamic-resolution SigLIP-style vision encoder (Conv2d patch embedding, cached bilinearly interpolated learned position embeddings for repeated page grids, 2D vision RoPE, `cu_seqlens`-packed attention with uniform and length-bucketed batched fast paths, and a spatial-merge projector) with a lightweight ERNIE-4.5 text decoder that uses MRoPE. Best for plain OCR, tables, formulas, and chart understanding.
- Step-3.7 (`step3p7`): StepFun's multimodal model. A 47-block `perception_encoder` ViT tower (Conv2d patchify, learned absolute position embeddings bilinearly resized for the patch grid, 2D vision RoPE, LayerScale, quick-GELU MLP) feeds two stride-2 conv downsamplers and a linear projector into the text hidden size. Each image runs a base pass (728 px, 52x52 grid collapsed to 169 tokens) and, when larger than 728 px, a tiled-patch pass (504 px windows, each 36x36 grid collapsed to 81 tokens); features are ordered patches-first then base per image and scattered into `<im_patch>` placeholders. The text decoder is the Step-3.5 MoE stack (`text_config.model_type: step3p5`), reused as the backbone. Text-only prompts work through the same checkpoint.
- dots.ocr (`dots_ocr`): document-OCR VLM (rednote-hilab) pairing a 42-block `dots_vit` dynamic-resolution ViT with a Qwen2 text decoder. The tower shares the Qwen2-VL vision machinery (`cu_seqlens`-packed block-diagonal attention, 2D vision RoPE, merge-block patch ordering) but uses RMSNorm blocks, a SwiGLU vision MLP, a patch embed with bias and a trailing RMSNorm, and a `post_trunk_norm`; its merger projects each 2x2 patch block straight to the text width (no separate connector). Plain 1D RoPE text decode, no MRoPE. Best for layout analysis (bbox + category JSON), plain OCR, tables, formulas, and Markdown conversion.
- GLM-4V (`glm4v`): GLM-4V ViT vision tower (3D patch embedding, bilinear-resampled learned position embeddings, Conv2d spatial downsample, SwiGLU patch merger) plus a GLM-4 text backbone driven by sectioned even/odd MRoPE. Reuses the shared Qwen-VL image processor and prompt/token plumbing.
- GLM-4V MoE (`glm4v_moe`): GLM-4.5V-class variant reusing the GLM-4V ViT vision tower with a GLM-4 MoE text backbone (grouped `noaux_tc` routing, shared experts, `first_k_dense_replace` dense layers) driven by sectioned half-split MRoPE. Reuses the GLM-4 MoE machinery and the shared Qwen-VL runtime.
- Granite Vision (`granite_vision`, or `llava_next` with a `granite` text config): IBM's document VLM. A SigLIP vision tower with four intermediate feature taps (concatenated on the channel axis) feeds a 2-layer GELU projector; a learned `image_newline` embedding, LLaVA-Next AnyRes multi-tile preprocessing (`image_grid_pinpoints`), and a dense Granite text backbone (embedding / attention / residual / logits multipliers) complete the model. Both `config.json` spellings route to the same loader. Best for document understanding, charts, and tables.
- Granite 4 Vision (`granite4_vision`): IBM's document VLM with multi-depth visual injection. A SigLIP tower feeds eight window-QFormer projectors (a QFormer with self+cross attention per `4x4`/`8x8` window, with deepstack mean-pool and spatial strided-offset query downsamplers) whose packed outputs are added into the residual stream at eight different depths of a Granite 4 hybrid (`granitemoehybrid`) text backbone during prefill, rather than merged once. Reuses the shared AnyRes tiling and the four Granite scalar multipliers. Best for document understanding.
- DeepSeek-OCR (`deepseekocr`): DeepSeek's document-OCR VLM. Each view runs a SAM-style ViT-B (windowed / global attention with a decomposed relative-position bias, a two-conv neck, and a two-stage stride-2 conv compressor) and a CLIP-style ViT-L that ingests the SAM grid as its patch embeddings; the two token sets are concatenated on the channel axis, projected to the decoder width, and laid out as a 2D tile mosaic with learned `image_newline` columns and a trailing `view_separator`. A small DeepSeek MoE decoder (12 layers, 64 routed + 2 shared experts, standard attention, reusing `deepseek`) reads the mosaic. The processor pads a global 1024 view and, for larger images, adds a closest-aspect-ratio grid of 640 tiles. Best for plain text, markdown, HTML tables, and grounding boxes.
- DeepSeek-OCR 2 (`deepseekocr_2`): the second-generation document-OCR VLM. It keeps DeepSeek-OCR's SAM-style ViT-B (with the compressor emitting 896 channels) but replaces the CLIP stage with a Qwen2-0.5B-shaped query resampler: the SAM grid is concatenated with a learnable query bank and run through 24 GQA + rotary + SwiGLU layers under a mixed mask (image tokens bidirectional among themselves, queries causal and attending to all image tokens); only the query outputs are projected to the decoder width. There is no channel-concat fusion and no `image_newline` mosaic, so features are flat runs assembled per image as `[tiles, global, view_separator]`. The processor tiles every image with 768 tiles (closest-aspect-ratio grid of 1..6) plus the padded global 1024 view. The same DeepSeek MoE decoder reads the features. Best for plain text, markdown, HTML tables, and grounding boxes.
- DeepSeek-VL2 (`deepseek_vl_v2`): DeepSeek's general vision-language model. Each view runs a SigLIP-style ViT (no CLS token, a `patch_size` Conv2d patch embed folded into a linear, a learned absolute position embedding, fused-qkv full attention, GELU-tanh MLP, and a trailing LayerNorm); the `downsample_mlp_gelu` projector folds each 2x2 patch block into one channel-outermost feature vector and maps it to the decoder width with a two-layer GELU MLP. Features are laid out as a 2D tile mosaic with learned `image_newline` columns and a `view_separator`, ordered `[global, view_separator, local]`. A DeepSeek-V2 MoE decoder (MLA attention, softmax-gated routed experts plus shared experts, reusing `deepseek_v2`) reads the mosaic. The processor pads a global view and, for at most two images, adds a best-fit grid of local tiles chosen from `candidate_resolutions`; three or more images use a single tile each. Best for image description, grounding, and document understanding.
- ERNIE-4.5 MoE VL (`ernie4_5_moe_vl`): Baidu's vision-language MoE. A DFNRope ViT (linear patch embedding over 588-wide merge-window rows, 2D vision RoPE, `cu_seqlens`-packed attention, quick_gelu MLP) feeds a variable-resolution resampler (2x2 spatial fold, temporal pair fold with single-frame duplication, GELU MLP stacks, RMSNorm) whose rows replace the `<|IMAGE_PLACEHOLDER|>` tokens. The text decoder extends ERNIE-4.5 MoE with modality-split expert banks: separate text and multimodal routers and expert stacks selected per token by token type, with a correction bias that shifts expert selection but never the mixing weights, plus a fused shared-experts MLP. Position encoding is interleaved 3D MRoPE (`[T, H, W]` axes assigned per frequency index, adjacent-pair rotation) that degenerates to traditional RoPE for text. Validated against `mlx-community/ERNIE-4.5-VL-28B-A3B-Thinking-4bit` (28B-A3B; text and resampler 4-bit, vision tower bf16). Best for general image chat and grounded reasoning; the checkpoint is a thinking variant and emits reasoning before the answer.
- Qwen3-Omni MoE (`qwen3_omni_moe`, thinker): Alibaba's omni-modal MoE. Stage 1 covers the thinker: text output conditioned on text, image, and audio inputs. The vision tower and MoE text decoder are the Qwen3-VL-MoE stack (DeepStack feature injection, interleaved MRoPE) reused unchanged; the new audio tower converts 16 kHz audio to a 128-bin log-mel spectrogram, downsamples it through three stride-2 convolutions (13 output frames per second of audio), and runs 32 windowed-attention encoder layers whose output rows scatter into the token stream exactly like image features. Audio arrives via `--audio file.wav` on the CLI (combinable with `--image`). Stage 2 adds speech output: `mlxcel generate --output-audio out.wav` runs the talker and code2wav after text generation and writes 24 kHz mono PCM16. The talker is a 20-layer Qwen3-MoE codec decoder conditioned on the projected thinker token embeddings of the chat-role segments; per frame it emits the first of 16 codebooks and a 5-layer code predictor fills in the residual 15, then the code2wav vocoder (causal pre-transformer, ConvNeXt upsampling, BigVGAN-style SnakeBeta decoder) renders 1920 samples per 12.5 Hz frame. `--speaker` selects the voice (ethan default; chelsie and aiden also ship in the released checkpoints). The speech stack loads lazily and only when requested, so text/vision use keeps its memory footprint; speech currently requires a text-only, chat-templated prompt. Validated against `mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit` (text and talker 4-bit; vision, audio tower, code predictor, and code2wav bf16).
- Hunyuan-VL (`hunyuan_vl`, e.g. HunyuanOCR): Tencent's vision-language family. A ViT with a per-patch conv embedding, bilinearly interpolated learned position embeddings, and full attention over the packed patch sequence feeds a `perceive` merger: a stride-2 conv pair over the raster grid, a learned `image_newline` column, a linear to the decoder width, and learned `image_begin` / `image_end` rows. Per image that yields `mh * (mw + 1) + 2` feature rows, matching the prompt placeholder count exactly. The decoder is the Hunyuan dense stack (per-head Q/K RMSNorm after the rotation, DynamicNTK-alpha rope base) with XD-RoPE at prefill: 4D `[P, T, H, W]` position ids split across the frequency dims, degenerating to the standard rotation for text; decode uses sequential positions. Validated against `hadeseus/HunyuanOCR-mlx-4bit` (text 4-bit, vision bf16). Best for OCR: text spotting, document parsing, and grounded extraction.
- FastVLM (`llava_qwen2` / `fastvlm`): Apple's low-latency VLM. A FastViTHD hybrid encoder runs entirely on channels-last maps: a conv stem, three RepMixer stages (depthwise token mixing plus a BatchNorm ConvFFN), two attention stages with channel LayerNorm and `head_dim` 32 self-attention, inter-stage large-kernel PatchEmbed downsamples, RepCPE position encoders, and a squeeze-excite `conv_exp` head. Each 1024x1024 pad-to-square image becomes a `(16, 16, 3072)` map flattened to 256 tokens, projected to the Qwen2 decoder width by an `mlp2x_gelu` MLP. The `<image>` placeholder is the fixed `-200` sentinel (not a vocabulary token); the runtime splices one sentinel per image, expands it to 256 tokens, and scatters the image embeddings (LLaVA merge). The text decoder is stock Qwen2, reused unchanged. The loader accepts both the genuine (`apple/FastVLM-0.5B`) and converted (`mlx-community/FastVLM-0.5B-bf16`) weight layouts. Best for fast image description and grounded chat.
- GLM-OCR (`glm_ocr`): document-OCR sibling of GLM-4V. A 24-block ViT (3D patch embedding, per-head q/k RMSNorm on the packed `cu_seqlens` attention, 2D vision RoPE, Conv2d spatial downsample, SwiGLU patch merger) feeds a 16-layer GLM-4 text decoder driven by full-width even/odd MRoPE (`rope_parameters` with `mrope_section [16, 24, 24]`, `partial_rotary_factor 1.0`). The tower has no learned position embedding or post-conv norm, and the loader drops the next-n prediction (MTP) layer. Patches are reordered from the processor's raster order into spatial-merge-window order so the rotary, downsample, and merged-token grid stay spatially aligned (OCR reads scrambled patches wrong). Best for plain text, tables, and formula recognition.
- Youtu-VL
- Kimi-VL and Kimi-VL 2.5 (`kimi_vl` / `kimi_k25`): MoonViT native-resolution vision encoder (Conv2d patch embedding, learned plus bicubically-interpolated 2D position embedding, a shared 2D rotary embedding, block-diagonal cross-image attention, and `spatial_merge_size` patch merging) feeding a `LayerNorm -> Linear -> GELU -> Linear` connector into a DeepSeek-V3-style MoE text backbone. Detected, loaded, and served end to end: the safetensors directory loader wires the MoonViT tower, the connector, and the DeepSeek-V3 MoE backbone; the native-resolution processor patchifies each image; and the runtime expands each `<|media_pad|>` placeholder into `(h/merge) * (w/merge)` tokens before the merged vision features replace them. Video is also supported, on both the CLI (`--video`) and the server (`video_url` content blocks): frames are extracted with `ffmpeg` (uniform sampling, default 2.0 fps, matching the Gemma 4 video path) and patchified per frame, and the encoder treats a clip as a 3D `(t, h, w)` patch grid rather than `t` independent images. A computed (not learned) temporal sinusoid is added to the tiled 2D position table, the 2D rotary tables repeat per frame, attention over a clip is one block-diagonal segment spanning all its frames, and the patch merger mean-pools over the temporal axis before the usual `spatial_merge_size` 2x2 merge, so the placeholder expansion `(h/merge) * (w/merge)` is unchanged by `t`. A single-frame clip is equivalent to the same frame processed as an image once the frame-0 temporal constant folds into the patch-embed bias.
- SmolVLM, SmolVLM2, and Idefics3 (`smolvlm` / `idefics3`): one shared Idefics3-family runtime, a SigLIP vision tower + pixel-shuffle token compression + a Llama text backbone (SmolLM2 for SmolVLM, Llama-3 for `Idefics3-8B-Llama3`). Both on-disk layouts load: `SmolVLMForConditionalGeneration` (`model.text_model.*` + a top-level `lm_head`) and `Idefics3ForConditionalGeneration` (the whole Llama-with-head nested under `language_model.*`). Images are tiled into square patches with a global thumbnail tile, and each `<image>` placeholder expands to `num_image_token` compressed feature tokens per tile.
- Idefics2 (`idefics2`): SigLIP vision tower + a perceiver-resampler connector (a `modality_projection` SwiGLU MLP into the text hidden size, then `n_latents` learned query slots refined by grouped-query cross-attention over the image patches) + a Mistral text backbone. Each image contributes a fixed `n_latents` (64) compressed feature tokens regardless of the patch grid, and the `<image>` placeholder is framed by `<fake_token_around_image>` and expanded to those tokens. This first port feeds a single full-resolution square tile per image; the optional `do_image_splitting` high-resolution tiling is a documented follow-up.
- MiniCPM-O
- Moondream 3
- Moondream 2 (`moondream2` / `moondream1`): reuses Moondream3's linear-patch ViT vision tower and overlap-crop preprocessor, paired with a Phi-1.5-style dense text decoder (fused QKV, partial rotary embedding, parallel attention/MLP, tanh-GELU) instead of Moondream3's sparse-MoE decoder. Images are split into a resized global crop and a grid of overlapping local crops; the local crop features are trimmed of their overlap margins, stitched, and adaptively average-pooled back to the 27x27 encoder grid before being concatenated with the global features and projected to the text hidden size. The BOS token and the 729 projected image tokens form a bidirectional prefix ahead of the causal text prompt. Checkpoint revisions from 2025-06-21 onwards are trained against the `moondream/starmie-v1` tokenizer with Moondream3-style control-token templates (bos = eos = 0), while the official repository still ships the older GPT-2 `tokenizer.json` next to them; mlxcel detects the revision from the bundled `moondream.py`, resolves the starmie tokenizer from the Hub (cached after the first fetch, or place starmie's `tokenizer.json` in the model directory when offline), and keeps the GPT-2 tokenizer with `Question:`/`Answer:` framing for the 2025-01-09 .. 2025-04-14 revisions where that contract is the correct one.
- Phi-3 Vision, Phi4MM, Phi4 SigLIP VLM
- Molmo2 and Molmo-Point
- Nemotron-H Nano Omni: ships a Conformer/Parakeet audio encoder and accepts spoken audio from the CLI with `--audio <path>`. Input audio is resampled to 16 kHz before the encoder. Server-side `input_audio` in `POST /v1/chat/completions` is also supported: the audio block is spliced inside the last user turn, before the `<|im_end|>` end-of-turn marker that the ChatML template uses (id 151 in the released checkpoint).

Audio/video capability is model-specific. The server request types include
`image_url`, `video_url`, and `input_audio` content blocks, but a loaded model
must advertise support for the corresponding modality. Video frame extraction
uses the system `ffmpeg`/`ffprobe` binaries at runtime.

### Gemma 4 image soft-token budget

Gemma 4's vision front-end is resolution-driven: it works over whatever patch
grid the preprocessor hands it, with no fixed feature length. The number of soft
tokens an image contributes to the prompt therefore follows directly from the
resize target, and the resize target follows from a soft-token budget. That
budget is read from the checkpoint's config at load time (280 for every shipped
Gemma 4 checkpoint) and can be overridden per request.

This applies to both Gemma 4 architectures: `gemma4` (the ViT tower, e.g.
`gemma-4-e2b-it`, `gemma-4-31b-it`) and `gemma4_unified` (the encoder-free patch
projector, e.g. `gemma-4-12b-it`).

A larger budget keeps more detail (denser patch grid, better small-text and
fine-structure reading) at the cost of a longer prompt and more prefill work. A
smaller budget is cheaper and faster. Supported values, shared with the video
per-frame budget:

| Budget | Meaning |
|--------|---------|
| 70 | Smallest. What `detail: "low"` maps to. |
| 140 | |
| 280 | Checkpoint default for the shipped Gemma 4 models. |
| 560 | |
| 1120 | Largest. What `detail: "high"` maps to. |

Any other value is rejected with a 400. The budget is untrusted request input
and it scales the resized image and its patch grid, so it is validated against
this ladder rather than clamped: a silent clamp would leave the caller believing
they got a budget they did not get. 1120 is also the hard ceiling for
`gemma4_unified`, whose learned 2-D position table is exactly 1120 wide; a budget
past it would index off the end of that table.

#### Server

Two optional fields on the `image_url` content part:

```json
{
  "model": "gemma-4-12b-it-4bit",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Transcribe the small print."},
      {"type": "image_url", "image_url": {
        "url": "data:image/png;base64,...",
        "detail": "high"
      }}
    ]
  }]
}
```

* `detail` is the OpenAI-standard field. `"low"` maps to 70, `"high"` maps to
  1120, and `"auto"` (or an absent field) leaves the checkpoint's configured
  budget in place. An unknown value is a 400, not a silent fallback to `"auto"`.
* `max_soft_tokens` is an **mlxcel extension**, not part of the OpenAI spec. It
  names an exact budget from the ladder above and takes precedence over `detail`
  when both are present:

  ```json
  {"type": "image_url", "image_url": {"url": "...", "max_soft_tokens": 560}}
  ```

Both fields are optional and default to absent, so a request that sends only
`url` behaves exactly as it did before these fields existed. The budget applies
to the request as a whole; when a request carries several `image_url` parts they
must agree on an explicit budget (parts that set neither field are free to
coexist with parts that do). Every non-Gemma-4 model ignores both fields.

Note that the budget is part of the vision-feature cache identity, so the same
image requested at two budgets is encoded twice rather than served the wrong
cached features.

#### CLI

`mlxcel generate --image-soft-tokens <N>` takes the same ladder and the same
validation:

```bash
./target/release/mlxcel generate -m models/gemma-4-12b-it-4bit \
  --image receipt.png --image-soft-tokens 1120 \
  -p "What is the total?"
```

Omitting the flag uses the checkpoint's configured budget. The flag applies to
`--image` inputs only; `--video` frames keep their own (smaller) per-frame
budget.

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
