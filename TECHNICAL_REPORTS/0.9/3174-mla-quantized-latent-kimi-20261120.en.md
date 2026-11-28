# Technical Report: PR #3174 - Store a quantized MLA latent in Kimi Linear and Kimi K3

**Date**: 2026-11-20
**Status**: Completed
**Risk Level**: Medium (two hybrid families gain an opt-in int8 latent; a bf16 dtype promotion bug was found and fixed during measurement)

## Executive Summary

`kimi_linear`, `kimi_k3` and the Kimi K3 VLM wrapper now take the per-layer `KvCacheLayerModes` table and store the MLA latent of their MLA layers as the #3155 affine int8 latent (672 bytes per token per layer). KDA layers ignore their table entry. On Kimi-Linear-48B-A3B the teacher-forced trace has 0 decided disagreements behind 1024 and 4096, with five flips per prefix outside the 0.11 near-tie bar (largest 0.500), which the epic gate records without blocking. Decode is +1.5% at 512 tokens (fused kernel), -4.7% at 2048 and -23.3% at 16384 (composed path). The sizing code also stopped charging the latent on KDA layers, which had overstated Kimi-Linear-48B's KV by 27/7 at every mode.

## 1. Problem Statement

After #3160, `kimi_linear` and `kimi_k3` were outside `QUANTIZED_LATENT_FAMILIES`, so a quantized request resolved to fp16 and the batched helper kept them on the per-token path. They are hybrids: Kimi-Linear-48B has 27 layers, 7 of them MLA and the rest KDA (linear attention with recurrent state), so a mode table indexed by layer has entries that apply to no latent.

## 2. What Changed

- `src/models/kimi_linear.rs`, `src/models/kimi_k3.rs`, `src/vision/kimi_k3_vl.rs`: `set_kv_cache_layer_modes` is implemented and forwarded through the VLM wrapper. MLA layers write through `store_latent_step`; a one-query step over a quantized window calls `quantized_absorbed_attention` (fused when unmasked, composed when masked, with the mask cast to the score dtype). Multi-token steps dequantize the window once. A Turbo entry builds an fp16 cache through a new `latent_pair_cache_mode` helper.
- `src/models/mla_batched.rs`: batched decode opts in to quantized rows for both families, and `dense_latent_in` casts a dequantized window back to the query dtype.
- `mla/latent_pair.rs` and `cache/turbo/allowlist.rs`: `kimi_linear` and `kimi_k3` join `QUANTIZED_LATENT_FAMILIES`. The registry lists fp16 and int8 `kv_modes` for KimiLinear, KimiK3 and KimiK3VLM.
- `src/execution/kv_arch.rs`: the latent is charged only on MLA layers, using the 1-based `kda_layers` list. `mlxcel inspect --max-tokens 16384` on Kimi-Linear-48B now reports 126.0 MiB at fp16 and 73.5 MiB at int8.
- Docs: CHANGELOG, `docs/turbo-kv-cache.md`, `docs/supported-models.md`, `docs/environment-variables.md`, and `docs/benchmark_results/mla-quantized-latent-kimi-m3ultra-2026-11-19.md` with trace and decode CSVs.

## 3. Technical Decisions

**Let KDA layers ignore their mode entry instead of rejecting it.** The mode table is built per layer by code that does not know which layers are MLA. Rejecting a quantized entry on a KDA index would make every int8 request on a hybrid fail; ignoring it keeps the table shape uniform, and a test sets a mode on a KDA index alone to pin that behavior.

**Cast the dequantized window back to the query dtype.** `LatentWindow::into_dense` returns fp16. Against a bf16 query, MLX promoted the multi-token context pass to float32, and on Kimi the KDA recurrent state produced by that pass stayed float32, so every later decode step ran in float32. The first sweep showed int8 61 to 63% slower than fp16 for this reason. The cast in `dense_latent_in` removed it; regression tests for both models fail with the cast reverted.

**Keep the family enabled under the epic gate.** Zero decided disagreements is the hard part of the gate and holds at both prefixes. The near-tie misses sit inside the int8 fused-versus-composed noise (16 / 512 flips, 7 above 0.11, largest 0.438), and the fp16 floor on Kimi is 0 because fp16 single-sequence decode never takes the fused kernel.

## 4. Measurements

M3 Ultra, Metal, Kimi-Linear-48B-A3B-Instruct 4-bit, fp16 latent as the reference.

Trace (`--decided 2`):

| behind | decided | top-1 flips | flips above 0.11 | largest gap | ppl fp16 / int8 |
|---|---|---|---|---|---|
| 1024 | 0 / 201 | 12 / 512 | 5 | 0.500 | 12.4213 / 12.4579 (+0.29%) |
| 4096 | 0 / 52 | 13 / 256 | 5 | 0.312 | 24.5971 / 24.4992 (-0.40%) |

Decode, median tok/s, ABAB in one lock hold, spread at most 1.4%:

| prompt | fp16 | int8 | delta | int8 path |
|---|---|---|---|---|
| 512 | 127.49 | 129.34 | +1.5% | fused |
| 2048 | 127.43 | 121.43 | -4.7% | composed |
| 16384 | 115.94 | 88.97 | -23.3% | composed |

Prefill is within 0.6%. Peak memory differs by under 0.2 GB because only 7 of 27 layers hold a latent. Path counters show no decompressed decode step.

## 5. Learning Points

**A dtype mismatch in a stateful layer compounds across steps.** In an attention-only model, a float32 promotion in the context pass costs that one pass. In a hybrid, the promoted output feeds a recurrent state that persists, so the whole decode ran in float32. The symptom was a decode slowdown far larger than the latent work could explain, which is what prompted the dtype check.

**Sizing for hybrids has to follow the layer map.** The fp16 estimate for Kimi-Linear-48B was wrong before this PR because it charged the latent on all 27 layers. The int8 work exposed it only because the new numbers were compared against `inspect` output on the real config.

## 6. Follow-up Actions

- DeepSeek V2 and the `DeepSeekV3Attention` families use the same `into_dense` and may promote their multi-token step to float32 on bf16 checkpoints. They have no recurrent state, so the cost should be limited to that step; not yet measured.
- The int8 composed path loses 23% at 16384 tokens; the fused kernel's 2048-key limit is the cause.
- Kimi K3 is not measured: the only export is an expert-pruned 2-bit model.
- The `--kv-bits 8` sizing overestimates by one layer when a hybrid ends on a KDA layer.
- `kda_layers[0] - 1` in `kimi_linear.rs` underflows when the first entry is 0 (pre-existing, result unused).
- CUDA and ROCm were not measured.
