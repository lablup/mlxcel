# Technical Report: PR #3175 - Quantized MLA latent for the DSA families, measured and left off

**Date**: 2026-11-20
**Status**: Completed
**Risk Level**: Medium (the int8 path stays disabled, but the PR fixes DeepSeek V3.2 loading and changes DeepSeek-V3/R1 expert routing on main)

## Executive Summary

The DSA attention (`deepseek_v32`, `deepseek_v3.2`, `glm_moe_dsa`) gained an int8 latent slot and an int8 indexer-key codec, and DeepSeek-V3.2-4bit was measured on it. The trace gate passed (0 decided disagreements at every prefix), but the per-layer indexer top-k overlap gate (mean >= 0.99 in every layer) failed: 12 of 61 layers pass at 4096 tokens and 6 of 61 at 16384. The three types therefore stay out of `QUANTIZED_LATENT_FAMILIES` and a quantized request on them still resolves to fp16. To measure at all, the PR fixes three defects that kept the checkpoint from loading or routing correctly, and one of them also existed in `DeepSeekV3Attention`: DeepSeek-V3 and R1 (`n_group` 8) had been routing every token as if the model declared no expert groups.

## 1. Problem Statement

DSA layers keep two per-token stores: the MLA latent and an indexer key used to pick the top-k positions each query attends to. A quantized latent only saves memory if both stores shrink, and quantizing the indexer key can change which positions are selected, which a logit trace alone does not show. #3163 asked for the port plus a selection-level gate.

## 2. What Changed

- `mla/latent_keys.rs` (new): an int8 codec for the indexer keys with scales per group. `mla/cache.rs` stores both planes in the latent slot.
- `src/models/deepseek_v32.rs`, `src/models/deepseek_v32_latent.rs`: the attention reads a quantized slot through the absorbed path; multi-token steps dequantize once.
- Fixes needed to run the checkpoint on main: the stacked `switch_mlp` expert layout, the routed-expert squeeze, and grouped routing on rank-3 router scores. The stacked loader now refuses a scales or biases plane whose expert count does not match.
- `src/models/deepseek_v3.rs`: the same rank-3 grouping fix. The gate flattens scores to `[tokens, experts]` around `group_mask_scores`, which had skipped rank-3 input with a one-time warning.
- `kv_arch`: DSA sizing now includes the indexer key, 1408 bytes per token per layer at fp16 (it reported 1152) and 808 at int8.
- Tooling: `MLXCEL_TRACE_KV_CACHE_MODE` and a `trace_topk` dump in `examples/logit_trace`, `scripts/compare_dsa_topk.py`.
- Docs: CHANGELOG, `docs/turbo-kv-cache.md`, `docs/environment-variables.md`, and `docs/benchmark_results/mla-quantized-latent-dsa-m3ultra-2026-11-19.md` with trace, top-k and decode CSVs.

## 3. Technical Decisions

**Keep the families off on the top-k gate.** The epic set the top-k overlap gate as hard, separately from the trace gate, because a changed selection can stay invisible in teacher-forced logits for a few hundred tokens and still compound over a long generation. Layer 0 overlaps 0.998, so the codec itself costs little; the drift accumulates with depth. The fp16 kernel choice alone (fused against composed) also misses 0.99 on 29 of 61 layers, so the bar may be stricter than this family's own noise. Whether to relax it is left as an open decision; the code is ready to enable by adding the types to the allowlist.

**Fix V3 routing in this PR instead of a separate one.** The reviewer found the V3 bug while fixing the identical V3.2 one. The fix is a reshape around an existing call, a tiny-gate test pins it (fails before: picks {0, 3} instead of {0, 1}), and checkpoints with `n_group` 1 (Moonlight, Kimi-VL, Kimi K2) are unchanged. It changes DeepSeek-V3 and R1 outputs, toward the reference routing.

## 4. Measurements

M3 Ultra, Metal, DeepSeek-V3.2-4bit, fp16 as the reference. The measurement build listed the three types locally; the committed tree does not.

Top-k overlap:

| context | comparison | layers >= 0.99 | mean | worst layer |
|---|---|---|---|---|
| 4096 | int8 vs fp16 | 12 / 61 | 0.9793 | 0.9662 |
| 4096 | fp16 fused vs composed | 32 / 61 | 0.9911 | 0.9854 |
| 16384 | int8 vs fp16 | 6 / 61 | 0.9021 | 0.7438 |

Trace: 0 decided disagreements behind 1024, 4096 and 16384, single-sequence and two-row batched. Flips are about the size of the fp16 fused-versus-composed floor. The fp16 logits on this model are spaced 0.125 apart, so every nonzero gap exceeds the 0.11 bar.

Decode, median tok/s fp16 / int8: 20.76 / 19.75 (-4.9%) at 512, 17.54 / 15.44 (-12.0%) at 2048, 15.68 / 13.63 (-13.1%) at 16384. Prefill is 13 to 20% slower at int8.

## 5. Learning Points

**A selection-level gate catches what a logit trace misses.** The trace passed at every prefix while per-layer selection drifted to 0.74 overlap at 16384. For sparse attention, the quantity that compounds is the selected set, so it needs its own measurement.

**Measuring a family can surface defects in its shared code.** The checkpoint did not load on main, and fixing its routing led to the same bug in the dense V3 gate, which had only shown up as a one-time warning.

## 6. Follow-up Actions

- The fp16 reference on DeepSeek-V3.2 degrades past 4096 tokens of context (ppl 7.3 behind 4096, 1403 behind 8192, 2080 behind 16384). This defect predates the PR and has no issue yet; the 16384 rows sit on that broken reference.
- Decide whether the per-layer 0.99 overlap bar stands for DSA, given that fp16's own kernel choice misses it on 29 layers.
- GLM-5 is not measured (checkpoint not on this host) and inherits the V3.2 decision.
- No pre-change fp16 baseline exists because the checkpoint does not load at the base commit; fp16 equality rests on tiny-model tests and review.
- CUDA, ROCm, and `mlxcel serve` with the prompt cache on this checkpoint were not measured.
