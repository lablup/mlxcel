# Technical Report: PR #3176 - Quantized MLA latent for LongCat-Flash; GLM-4.7-Flash ported and left off

**Date**: 2026-11-20
**Status**: Completed
**Risk Level**: Medium (LongCat gains an opt-in int8 latent and now loads at all; GLM is ported behind an unlisted opt-in)

## Executive Summary

`longcat_flash` and `longcat_flash_ngram` now store their MLA latents as the #3155 affine int8 latent when int8 is requested, and join `QUANTIZED_LATENT_FAMILIES`. On LongCat-Flash-Lite the teacher-forced trace has 0 decided disagreements behind 1024 and 4096, and `inspect` reports 294.0 MiB at int8 against 504.0 MiB at fp16 for 16384 tokens. Decode is +8.0% at 512 tokens, -15.5% at 2048 and -49.2% at 16384, where the composed path takes over from the fused kernel. `glm4_moe_lite` (GLM-4.7-Flash) is ported, including MTP verify and rollback, but has 1 decided disagreement behind 1024 and stays fp16 under the hard gate. Two LongCat loading bugs on main, which kept the checkpoint from running in any KV mode, are fixed.

## 1. Problem Statement

After #3160 and #3162, the latent-pair families still on the fp16 substitution were GLM-4.7-Flash and LongCat-Flash. Both call `KVCache::update_and_fetch` from their own attention code. LongCat has two MLA sub-layers per decoder layer, and GLM has an MTP drafter that verifies and rolls back the target cache.

## 2. What Changed

- `src/models/longcat_flash_ngram.rs`: both MLA sub-layers write through the latent view, one-query steps read through `quantized_absorbed_attention`, and multi-token steps dequantize once. Quantized embedding tables now load as quantized, and the MoE flattens `[B, L, hidden]` before `SwitchGLU`, which had thrown at 64 or more positions. Embedding tables and the n-gram config are validated at load.
- `src/models/glm4_moe_lite.rs`, `glm4_moe_lite_mtp_hooks.rs`, `glm4_moe_lite_mtp_drafter.rs`: the same port, behind a per-model `quantized_latent` opt-in, with MTP verify and rollback over a quantized window.
- `src/models/mla_batched.rs`: `dense_latent_in` casts a dequantized window to the query dtype, so bf16 checkpoints are not promoted to float32.
- `mla/latent_pair.rs`, `cache/turbo/allowlist.rs`, `registry.rs`: LongCat joins the list, its `kv_modes` are fp16 and int8. GLM is not listed.
- `kv_arch`: LongCat is sized with two latent caches per layer.
- `examples/logit_trace/kv_mode.rs`: the trace harness hands the requested mode table to model-owned caches.
- Docs: CHANGELOG, `docs/turbo-kv-cache.md`, `docs/supported-models.md`, `docs/benchmark_results/mla-quantized-latent-glm-longcat-m3ultra-2026-11-19.md`.

## 3. Technical Decisions

**Keep GLM off on one decided disagreement.** The epic gate makes zero decided disagreements a hard requirement. GLM's single decided disagreement behind 1024 (gap 2.75) matches the int8 fused-against-composed noise, which also has one, so it is likely kernel noise rather than a latent defect. The rule was still applied as written. The port stays in the tree, so enabling GLM is adding it to the list after a decision.

**Fix LongCat loading in this PR.** The checkpoint did not run on main in any KV mode, so there was no fp16 reference to gate against without the two loader fixes. Both have tiny-model tests.

## 4. Measurements

M3 Ultra, Metal, fp16 latent as the reference, `--decided 2`.

| model | behind | decided | top-1 flips | flips above 0.11 | ppl fp16 / int8 |
|---|---|---|---|---|---|
| LongCat-Flash-Lite | 1024 | 0 / 239 | 15 / 512 | 9 (max 0.375) | 14.1036 / 14.0010 |
| LongCat-Flash-Lite | 4096 | 0 / 121 | 9 / 256 | 5 (max 0.25) | 9.9146 / 9.7928 |
| GLM-4.7-Flash | 1024 | 1 / 193 | 40 / 512 | 18 (max 2.75) | 21.4881 / 20.8288 |
| GLM-4.7-Flash | 4096 | 0 / 95 | 25 / 256 | 15 (max 1.0) | 53.6498 / 53.4191 |

LongCat decode, median tok/s fp16 / int8: 125.52 / 135.53 (+8.0%) at 512 (fused), 127.40 / 107.59 (-15.5%) at 2048, 99.20 / 50.39 (-49.2%) at 16384 (composed). No decode step was decompressed.

## 5. Learning Points

**The bf16 promotion showed up in three units independently.** #3162 found it as a decode slowdown, this PR found it as perplexity moving 8 to 13% on GLM. The pre-fix GLM numbers had 2 decided disagreements and were discarded. A dequantize helper returning a fixed dtype is easy to miss in tests that run in f32 or fp16.

**A missing fp16 noise floor changes how a miss reads.** GLM's fp16 latent always decodes composed, so there is no fp16 fused-versus-composed pair. The int8 pair is the only floor available, and it has the same single decided disagreement.

## 6. Follow-up Actions

- `deepseek_v2.rs` and `deepseek_v3.rs` dequantize through `into_dense` without the cast, so int8 prefill on bf16 checkpoints (Moonlight, Kimi-VL, DeepSeek-V2-Lite) runs in float32; their #3155 and #3160 gate numbers may move after the fix.
- The composed int8 path loses up to 49% at 16384 tokens; extending the fused kernel past 2048 keys would cover it.
- Decide whether GLM's single decided disagreement, matching int8 kernel noise, is enough to enable it.
- The 560B `longcat_flash`, CUDA and ROCm were not measured. GLM decode numbers predate the bf16 fix.
