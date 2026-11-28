# Technical Report: PR #2037 - Nemotron VoiceChat Offline Duplex Inference

**Date**: 2026-09-29

**Status**: Implemented and validated on the 4-bit checkpoint; pending merge.

**Languages**: Rust

**Risk Level**: Low. The new family sits behind its own detection arm. The only shared-code changes are:
- two additive `NemotronHModel` methods
- a prefix-parameterized Gemma 3 block loader
- a new `--extra-decoding-seconds` flag

## Executive Summary

This PR adds NemotronLabs VoiceChat (`nemotron_voicechat`), a full-duplex speech model, and runs it offline. It listens, transcribes, answers in text, and speaks the answer on one 80 ms timeline. One checkpoint packs four networks:
- a cache-aware FastConformer speech encoder with an RNNT transcript branch
- a 56-layer Nemotron-H LLM with a text head and a function head
- the EAR-TTS speech decoder
- a 31-codebook neural codec

`mlxcel generate --audio q.wav --output-audio a.wav -p "<system prompt>"` prints the transcript and the answer and writes 22.05 kHz speech in the built-in `Aria` voice.

The reference is the mlx-vlm 0.7.4 / mlx-audio implementation, and the port reproduces it exactly on the 4-bit checkpoint. That covers the transcript, every text and function id, and, with `--seed 0`, all 1922 sampled speech codes of the test utterance. This PR is sub-issue 1 of epic #1372; the cache-aware online session (#1378) and the `/v1/realtime` endpoint (#1376) build on it.

## 1. Problem Statement

mlxcel served speech-to-text (Whisper) and text-to-speech (Kokoro) as separate request/response endpoints. It had nothing for a duplex model, where silence and overlapping speech are part of the input and the input timeline, not a token budget, sets the output length.

Three of the four networks had no mlxcel equivalent: a causal FastConformer with chunked-limited attention, an RVQ speech decoder with a mixture-of-Gaussians head, and a ConvNeXt/iSTFT neural codec. The fourth, Nemotron-H, existed but could not take injected embeddings through its final norm, and had no second head.

## 2. Change Summary

- **Detection and loading.** `nemotron_voicechat` is matched before the `nemotron_h` arm, because the checkpoint nests a `nemotron_h` text config. It maps to `ModelType::NemotronVoiceChat` and `LoadedModel::NemotronVoiceChat`, with registry capabilities: generate runtime, audio in and out, and the `speech_to_speech` category. The `LanguageModel` impl delegates text-only calls to the Nemotron-H backbone; it exists for trait completeness only.
- **Speech front end** (`audio::nemotron_mel`, `audio::fastconformer`, `audio::rnnt`):
  - preemphasis log-mel with a symmetric Hann window
  - causal depthwise-striding subsampling
  - relative-position attention with a `[70, 0]` chunked-limited mask
  - causal depthwise convolution with LayerNorm
  - greedy RNNT over a two-layer LSTM prediction net
- **Codec** (`audio::nemotron_codec`): ConvNeXt encoder and decoder, PRVQ, and an n_fft 16 STFT/iSTFT. It also includes `decode_step` with a `CausalConv1dCache` for the online session.
- **TTS** (`models::gemma3_backbone`, `models::nemotron_voicechat::tts`):
  - a reusable Gemma 3 stack on injected embeddings
  - the character-aware subword encoder
  - gated fusion
  - the MoG head
  - masked RVQ refinement
  - Aria warmup and per-step generation
- **LLM glue.** `NemotronHModel::forward_embeds_to_hidden` and `apply_lm_head` are added. `VoiceChatLanguageModel` renames `stt_model.*` keys onto the existing loader and adds the function head.
- **Offline session and CLI.** `NemotronVoiceChatModel::generate_offline` runs the timeline. `mlxcel generate` routes VoiceChat right after `-m` resolution and adds `--extra-decoding-seconds`; `--seed` makes the speech reproducible.

## 3. Technical Decisions

### The Python reference, not the issue body, decides numerics

The issue was written before anyone ran the converted checkpoint. Where the issue and the reference disagree, the reference wins, and the PR records each case:
- The checkpoint is in torch layout. Loaders convert it with idempotent shape gates.
- The subsampled frequency axis is 17, not 16.
- The codec has 13 layers per side.
- Perception, the TTS backbone and the MoG head run in f32 against bf16 weights.

The issue's "codec tone SNR above 20 dB" cannot be met: this lossy neural codec reaches 0.7 dB in the reference too, so codec parity is checked against the reference reconstruction instead.

### Exact code parity needed native half-precision reductions

Sampled codes come from an `argmin` over residual-VQ distances, and those distances use bf16 codebook norms. `mlxcel_core::sum_axis` widens bf16 to f32, reduces, and rounds once. That differs from MLX's native bf16 `mx.sum` in about a fifth of the norms, which is enough to flip codes. Both call sites now use an einsum reduction, which lowers to the native sum (`audio::native_reduce`).

The crate's fused GeGLU also disagrees with `mlx.nn.gelu_approx` on about half of bf16 elements, so the TTS stack uses an op-for-op `gelu_approx`. `gemma3::MLP` is untouched.

With these two changes, and the reference's global-RNG call order, a seeded run reproduces every code. Both findings apply to any future bf16 port that must match mlx-vlm bit for bit.

### Cached language model in the offline path

By default the reference's offline path recomputes the full LLM history at every timeline position. Its cached mode gives identical tokens and codes. The port keeps the Nemotron-H caches. That bounds the per-position work and matches the online session in #1378.

### Early CLI routing

The CLI routes VoiceChat before the chat-template, tokenizer and memory-preflight setup that every text model goes through, rather than after model load as Florence-2 does. None of that setup applies to a duplex model: `-p` is a system prompt and `-n` has no meaning. A VoiceChat run without `-p` is a one-shot run with no system prompt, not interactive chat.

## 4. Validation

Test setup:
- Checkpoint: `mlx-community/NemotronLabs-VoiceChat-11B-4bit`
- Input: the synthesized question "What is the capital of France?" (1.8 s)
- System prompt: "Be concise and answer in one sentence."
- 3 s of extra decoding, seed 0
- Compared against an mlx-vlm 0.7.4 reference dump

| Check | Result |
|---|---|
| Transcript / answer | "What is the capital of France?" / "The capital of France is Paris." (identical) |
| Text and function ids (62 positions) | identical |
| EAR-TTS codes (62 x 31) | 1922 of 1922 identical |
| Answer audio vs reference | decode within 1e-7 |
| FastConformer encoder | max abs 5.4e-7 |
| Codec tone codes / prompt codes | 0 of 372 / 0 of 1085 differ |
| Seeded CLI runs | byte-identical WAVs |
| Answer WAV fed back through the model | "The capital of France is Paris" |

The local gate covers:
- fmt and clippy (`-D warnings`)
- the touched modules' unit tests
- `cli_help_consistency`, `dead_doc_pointers`, `llama_compat_manifest`
- the env-gated real-checkpoint tests `tests/nemotron_voicechat_{llm,front,codec,tts,offline}_real.rs`

Two `detection_tests` failures come from main: #2031 changed `has_vision_config` without updating its tests.

## 5. Learning Points

- An exact-parity port against an MLX Python reference has to match the reference op for op. That includes reductions and activation formulas, not only the math. A fused kernel, or a reduction that widens precision, is enough to flip sampled discrete outputs.
- Dtype flow in MLX follows promotion. bf16 weights meeting an f32 input produce f32 activations, so "run in bf16" is not a safe reading of a quantized checkpoint.
- Seeded sampling reproduces across languages when both sides use MLX's global RNG with the same call sequence (one uniform and one normal per refinement pass).

## 6. Known Limitations

- The real-time factor is not measured here: the GPU was shared during this run. `docs/nemotron-voicechat.md` carries a marked placeholder table for the orchestrator. The per-frame profiler lands with #1378.
- Only the checkpoint's `Aria` voice is supported, at batch size 1.
- Only the converted MLX safetensors layout loads.
