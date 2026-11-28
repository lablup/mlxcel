# PR #2785: Settle the GELU formula for the families #2423 left open

**Date**: 2026-10-31 (generated before merge; the PR was open at the time)
**Status**: In review
**Languages**: Rust, C++
**Risk Level**: Medium (about 30 families change their GELU formula; each change follows the checkpoint config, and every family with a local checkpoint was smoke-tested on the CLI and the server)

## Executive Summary

#2423 inventoried every GELU call site and switched a first group of families to the formula their checkpoints declare, leaving the rest as open decisions in issue #2548. This PR closes them with one written rule: the checkpoint config decides, and where the key is omitted, the Hugging Face config-class default for that `model_type` decides, as `ACT2FN` evaluates it. Where the mlx-lm or mlx-vlm port departs from Hugging Face, mlxcel follows Hugging Face and records the divergence. `docs/gelu-call-sites.md` section 0b holds one row per decision with its reason.

## 1. Problem Statement

`mlxcel_core::gelu_approx` computes the exact erf GELU despite its name, while MLX `nn.gelu_approx` is the tanh form. Many ports copied the name and so ran erf where the checkpoint declares `gelu_new` or `gelu_pytorch_tanh`, or where a CLIP tower's omitted `hidden_act` means `quick_gelu`. The erf/tanh gap is below 1e-3 per element and the logit-trace gate cannot see it, so these were config decisions rather than measured regressions. Two sites were worse than a formula choice: Nemotron-NAS sent `gelu_new`, `gelu_fast` and any unknown name to SiLU, and Gemma 4's tensor-parallel per-layer gate computed erf while the single-host path computed tanh.

## 2. Technical Review

- **Formula per family.** Each changed row was checked against the local checkpoint's `config.json` or the remote modeling code: StarCoder2 and Gemma v1 declare `gelu_pytorch_tanh`, Phi-2 `gelu_new`, Molmo's remote code builds its CLIP ViT with `image_mlp_activations="quick_gelu"`, DeepSeek-VL2's `siglip_vit.py` uses `nn.GELU(approximate="tanh")`, and Phi-3.5-V names `openai/clip-vit-large-patch14-336`.
- **Shared kernel blast radius.** `compiled_gelu_mlp_forward_fp16` moved from the erf to the tanh GeGLU. Its only callers are Gemma v1, the Gemma 4 dense MLP and the Gemma 4 assistant drafter, all `gelu_pytorch_tanh`.
- **Shared vision config.** An omitted `vision_config.hidden_act` resolves by tower `model_type` (SigLIP to tanh, CLIP to `quick_gelu`) in the loaders that need it; the serde default stays erf for the other towers. A declared `quick_gelu` now parses as fast; no local checkpoint declared it explicitly, so no existing tower changed through that path.
- **Gemma3n eager vs OpenXLA tape.** The kernel, the fused MLP sparse branch, the XLA `sparse_gelu_stages` tape and the host oracle moved to tanh together. One structural difference predates this PR: the eager kernel multiplies a bf16 `stddev` by an f32 `array(std_multiplier)`, so MLX promotes everything from the cutoff onward to f32 and rounds once at the end, while the tape rounds every intermediate through bf16 (as mlx-lm's Python weak scalars would). The erf tape had the same shape. It needs a same-input comparison on an IREE host.

## 3. Technical Decisions

### 3.1 StarCoder2: move eager, not the XLA graph

The issue treated the OpenXLA StarCoder2 tanh graph as the bug because eager and mlx-lm use erf. The checkpoint declares `gelu_pytorch_tanh`, which is also the `Starcoder2Config` default, and the XLA goldens were frozen after the graph ran token-exact against an HF fp32 oracle built with that activation (`spike/openxla/dense_arch_check.py`, PR #561). Under the decision rule eager is the side that moves, so the frozen goldens stay untouched.

### 3.2 Tanh variant per site

Families switched from erf use `GeluActivation::TanhF32Compiled`, the #2423 choice that adds no prefill time. Gemma sites use the existing input-dtype tanh GeGLU kernels so the bf16 and quantized Gemma paths compute the same thing. The private copies in the Nemotron codec (erf) and the VoiceChat TTS backbone (tanh with `power`) fold onto `GeluActivation`; both folds change the last bit of some half-precision elements and are covered by tolerance tests.

### 3.3 Nemotron-NAS rejects unknown activations

`hidden_act` is parsed once at load into `NasActivation`. GELU names go through `GeluActivation::from_hf_name`; any name that is not SiLU, ReLU or a GELU variant is a load error instead of a silent SiLU.

## 4. Change Summary

| Item | Value |
|---|---|
| Files changed | 50 |
| Lines | +948 / -305 |
| New tests | FFI tests for the tanh `gelu_topk` and the tanh dense GeGLU, tower-default resolution (eager and XLA), MiniCPM-o activation parsing, Nemotron-NAS activation parsing, `quick_gelu` emitter op, tolerance tests for the two folds |

| Category | Summary |
|---|---|
| Gemma | v1 bf16, Gemma 4 dense bf16 and drafter, Gemma 4 TP gate, Gemma3n sparse layers (eager and XLA), RecurrentGemma move to tanh |
| Text LMs | Phi, Phixtral, GPT-2, StarCoder2 (eager), Moondream2/3 dense MLP move to tanh |
| Vision | Nine SigLIP-style towers and the Moondream vision MLP and projector move to tanh; Qwen2-VL, Molmo, LLaVA 1.5 / NeXT and Phi-3.5-V CLIP move to `quick_gelu`; MiniCPM-o reads its config |
| Kept with reason | GPT-NeoX, ModernBERT, Llama 4 vision, Kokoro PLBert, Molmo2, Qwen3-VL-MoE, Qwen3-Omni, LLaVA-Interleave, Llama-Nemotron-VL-Embed |

## 5. Follow-up Actions

- GPT-BigCode (T5) declares `gelu_pytorch_tanh` and still runs erf; it is now the one row that knowingly breaks the rule this PR writes down.
- Idefics2 (S6) and Granite 4 Vision (S7) force `quick_gelu` (matching mlx-vlm) while Hugging Face evaluates tanh; the inventory marks them MATCH against mlx-vlm, so they were outside #2548.
- Run the `xla-diagnostics` Gemma3n graphs on an IREE host and settle the eager-vs-tape carrier difference above.
- The fused single-kernel tanh GELU from the issue thread is tracked separately in #2780.

## Appendix: Validation

Narrow CUDA test filters on GB10 (`--test-threads=1`) pass for every touched module, the mlxcel-xla emitter tests pass with the StarCoder2 goldens byte-exact, and clippy and fmt are clean. The full workspace suite on this branch has one failure, `tests/step3p7_parity.rs` `text_decode_delegates_sequence_local_native_batching`, which fails identically on the base commit. Every changed family with a local checkpoint ran one greedy `mlxcel generate` and one `mlxcel-server` request with coherent output; families without a checkpoint (Gemma v1 bf16, RecurrentGemma, Phixtral, GPT-2, Phi-4-MM, Nemotron-NAS with a GELU activation, VoiceChat TTS, the Gemma 4 TP gate) rely on the synthetic tests named in the PR.
