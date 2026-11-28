# Technical Report: PR #3097 - Extend MLX causal mode to the remaining prefill masks

**Date**: 2026-11-15
**Status**: Completed (sink SDPA and hand-masked MLA paths keep their arrays by design)
**Risk Level**: Medium (31 model files and the tensor-parallel and pipeline paths)

## Executive Summary

#3056 routed purely causal prefill through MLX's causal mode for the Qwen hybrids and Qwen-VL. This PR covers the rest of #3072:
- **Plain-causal decoders** now use `prefill_causal_mask`: falcon_h1, plamo2, lfm2, bailing_moe_linear, llama4 global layers, mistral4, minicpm3, nemotron_nas, the GLM4V family, hunyuan_vl, paddleocr_vl, ernie4_5_moe_vl and TP Qwen 3.5.
- **Sliding hybrids** use a new `prefill_sliding_window_mask`, which takes causal mode inside the window and MLX's native `window_size` past it: cohere2, cohere2_moe, cohere_compass, olmo3, ministral3, exaone4, exaone_moe, mellum, step3p5, afmoe and gemma3n.
- **Gemma 2's softcapped attention** builds its mask once per forward instead of once per layer.
- **Maskless padded prefill:** five more families opt in.

On an M5 Max, the single-pass peak drops from 23.05 to 13.05 GB on gemma-3n-e2b and from 9.82 to 4.05 GB on exaone-4.0-1.2b, and prefill runs 2.1 to 2.3 times faster. Greedy output is byte-identical, and the teacher-forced trace shows 0 of 6144 disagreements on each of six models.

## 1. Problem Statement

These models built a dense f32 `[L, L + offset]` causal or sliding mask once per forward and handed it to every layer: 3.3 GB at 28K in a single pass. The array mask also keeps SDPA off the fused kernel where MLX would fuse causal mode.

## 3. Technical Decisions

- **Causal fallbacks added where missing.** Several attentions (falcon_h1, plamo2, nemotron_nas, mistral4, minicpm3, the GLM4V family) ran an unmasked SDPA when the mask was `None`, rather than falling back to causal attention as the issue assumed. A new `mlxcel_core::maskless_prefill_attention` gives them a causal fallback, and passes the window only once the keys outgrow it.
- **Per-layer window fields.** ministral3's full-attention layers carry the sliding window in `window_size`, so ministral3 and afmoe gained a per-layer `prefill_window` field to keep the window off full-attention layers.
- **Distributed paths match their single-process models.** The Jamba and Nemotron-H pipeline stages and TP Gemma 3 are now maskless, as their single-process models are. The pipeline GLM4-MoE-Lite stage is MLA. Its single-process model builds the array too, so it is unchanged.
- **Out of scope:**
  - Sink SDPA (gpt_oss, falcon_ocr, laguna, mimo_v2_flash, deepseek_v4).
  - MLA paths that add the mask by hand (kimi_linear, glm4_moe_lite, kimi_k3, longcat_flash_ngram).
  - Speculative verify blocks.

## 7. Change Summary

| model | path | peak, main → branch | prefill, main → branch | output |
|---|---|---|---|---|
| gemma-3n-e2b-it-4bit | single / 2048 | 23.05 → 13.05 / 7.65 → 6.82 GB | 10.48 → 4.53 / 7.19 → 4.74 s | identical |
| exaone-4.0-1.2b-4bit | single / 2048 | 9.82 → 4.05 / 5.15 → 4.72 GB | 6.07 → 2.87 / 3.75 → 3.08 s | identical |
| lfm2-8b-a1b-4bit | single / 2048 | 13.50 → 11.63 / 8.37 → 8.25 GB | 4.06 → 2.06 / 2.40 → 2.22 s | identical |
| c4ai-command-r7b (8K) | single / 2048 | 8.57 → 7.99 / 7.46 → 7.31 GB | 2.30 → 2.06 / 2.09 → 2.06 s | identical |
| glm-4.1v-9b (text) | single / 2048 | 18.10 → 18.08 / 8.91 → 8.40 GB | unchanged / 11.94 → 11.16 s | identical |
| llama-4-scout-17b-16e | single / 2048 | 72.88 → 73.00 / 66.85 → 66.79 GB | 27.50 → 26.07 / 19.64 → 19.41 s | identical |

Tests:
- 9 new prefill sliding-mask tests.
- The `mlxcel-core` lib suite (2317) and the full `mlxcel` lib suite serially (11,950) pass.
- Maskless padded-prefill parity gates pass for cohere2, olmo3, minicpm3, gemma3n and mistral4.
- Workspace clippy is clean.

Details: `docs/benchmark_results/prefill-causal-mode-remaining-m5max-2026-11-15.md`.

## 8. Follow-up Actions

- Llama 4's chunked-attention mask on its MoE layers still sets its peak.
- The GLM4V-family VLMs (MRoPE state), nemotron_nas (model-owned caches), Cohere2-MoE and EXAONE-MoE are not opted into maskless padded prefill: the first two for the reasons given, the last two because no local checkpoint can validate them.
