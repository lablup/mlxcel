# Technical Report: PR #3172 - Store a quantized MLA latent in the DeepSeekV3Attention families

**Date**: 2026-11-19
**Status**: Completed
**Risk Level**: Medium (eight model types gain an opt-in int8 latent; the near-tie bar was missed and accepted)

## Executive Summary

The eight model types that run `DeepSeekV3Attention` (`deepseek_v3`, `kimi_k2`, `joyai_llm_flash`, `kimi_vl`, `kimi_k25`, `youtu`, `youtu_llm`, `youtu_vl`) now write their MLA latent through `MlaLatentCache`, so `--kv-cache-mode int8` and `--kv-bits 8` keep the #3155 affine int8 latent (672 bytes per token per layer against 1152) instead of being substituted with fp16. On Moonlight-16B-A3B and Kimi-VL-A3B the teacher-forced trace has 0 decided disagreements behind 1024 and 4096, but Moonlight has two top-1 flips outside the 0.11 near-tie bar behind 1024 (0.117 and 0.328). The miss was reviewed and accepted. Decode is 10% slower than the fp16 latent at 512 tokens and 13% at 2048.

## 1. Problem Statement

#3155 added the int8 latent for `deepseek_v2` only. The latent-pair families called `KVCache::update_and_fetch` on `(kv_latent, k_pe)` directly, so rule 1 of `resolve_kv_cache_mode_for_model_with` resolved any quantized request on them to fp16. `DeepSeekV3Attention` is shared by eight model types and by the batched decode helper `mla_latent_decode_rows`, which four other families also use.

## 2. What Changed

- `mla::latent_pair` (new file beside `mla/cache.rs`): `QUANTIZED_LATENT_FAMILIES` and `latent_pair_supports_mode(model_type, mode)`, called by rule 1. Turbo modes still resolve to fp16, and families outside the list keep fp16.
- `src/models/deepseek_v3.rs` and `src/models/mla_batched.rs`: the single-sequence forward and the batched helper store through `store_latent_step`. A one-query step over a quantized window calls `mlxcel_core::mla::quantized_absorbed_attention`, which since #3156 tries the fused quantized kernel first (up to 2048 keys on Metal). Multi-token steps dequantize once. fp16 caches take the unchanged path.
- The batched helper takes an explicit `quantized_latent` opt-in. Only `DeepSeekV3Attention` passes `true`, so `deepseek_v32`, `kimi_linear`, `glm4_moe_lite` and `kimi_k3` keep the per-token path until their own issues (#3161-#3163). A selector over a quantized row panics with the family name.
- Sizing (`kv_arch`, `inspect`, preflights, prompt-cache budget), the KV advisor (int8 at medium and long contexts), the serve `--kv-bits` gate, the generate banner and the registry `kv_modes` (fp16 and int8 for the five quantized-latent registry entries) follow.
- Docs: `docs/turbo-kv-cache.md`, `docs/supported-models.md`, `docs/environment-variables.md`, CHANGELOG, and `docs/benchmark_results/mla-quantized-latent-v3-m3ultra-2026-11-19.md` with pre- and post-#3156 CSVs.

## 3. Technical Decisions

**Reuse the #3155 seam instead of a family-specific path.** The write goes through `MlaLatentCache::wrap` with an `MlaGeometry` built from the layer's dimensions, so an fp16 cache returns `Dense` and every existing branch runs unchanged. The fp16 traces are byte-identical to the base commit on both checkpoints.

**Make the batched opt-in explicit.** The batched helper is shared with families whose attention code still reads a dense window. An int8 cache could reach them through a mode table that did not go through rule 1, so the helper now needs the caller to declare that it reads `LatentWindow::Quantized`; families that do not are kept on the per-token path with a warning once per family.

**Accept the near-tie miss on decided disagreements.** The #3155 gate is 0 decided disagreements and every top-1 flip within 0.11 logits of the reference's top two. Both checkpoints pass the first part everywhere. Before #3156 merged, one flip per checkpoint behind 1024 was outside the bar (0.117 Moonlight, 0.180 Kimi-VL), and the fp16 fused-vs-composed noise floor on the same checkpoints peaks at 0.109. The user reviewed these numbers and chose to merge. After #3156 merged, the fused kernel changed which near-tie positions flip: Kimi-VL has no flip outside the bar and Moonlight has two (0.117 and 0.328), still with 0 decided disagreements. The results doc, CHANGELOG and `docs/turbo-kv-cache.md` state the miss.

## 4. Measurements

M3 Ultra, Metal, on the tree merged with #3156 unless stated; fp16 latent as the reference.

Trace (`--decided 2`):

| model | behind | decided | top-1 flips | flips above 0.11 | largest gap | ppl delta | int8 path |
|---|---|---|---|---|---|---|---|
| Moonlight | 1024 | 0 / 192 | 7 / 512 | 2 | 0.328 | +0.76% | fused |
| Kimi-VL | 1024 | 0 / 188 | 12 / 512 | 0 | 0.094 | +0.99% | fused |
| Moonlight | 4096 (pre-#3156, composed in both) | 0 / 58 | 5 / 256 | 0 | 0.039 | -0.04% | composed |
| Kimi-VL | 4096 (pre-#3156, composed in both) | 0 / 51 | 3 / 256 | 0 | 0.094 | -0.44% | composed |

Decode, median tok/s, fp16 / int8:

| model | 512 (fused) | 2048 (composed) | long (composed, pre-#3156) |
|---|---|---|---|
| Moonlight | 159.07 / 142.45 (-10.4%) | 141.80 / 122.97 (-13.3%) | 102.90 / 94.82 (-7.9%) at 8064 |
| Kimi-VL | 158.53 / 142.95 (-9.8%) | 141.85 / 122.61 (-13.6%) | 107.59 / 69.05 (-35.8%) at 16384 |

`inspect` reports 672 bytes per token per layer (283.5 MiB at 16384 tokens) and matches the bytes the server's prompt cache held. Workspace clippy and the targeted mlxcel-core and mlxcel suites pass on the merged tree.

## 5. Learning Points

**A kernel change moves near-tie flips even when it adds no decided disagreement.** Comparing the pre-#3156 composed int8 trace with the fused one gives 0 decided disagreements and 7 / 512 flips at gaps up to 0.469. The out-of-bar flip set changed from (0.117, 0.180) to (0.117, 0.328) without the decided count moving. A near-tie bar measured on one summation order does not carry over to another, which is why the decided count is the stable part of the gate.

**Merging a sibling can change what a PR's numbers describe.** #3172 was measured on the composed path; merging #3156 put its short-window decode on the fused kernel through a shared function, with no conflict in the PR's own files. The re-measurement on the merged tree is what made the PR body and results doc describe the code that merges.

## 6. Follow-up Actions

- #3161 (GLM-4.7-Flash, LongCat), #3162 (Kimi Linear, Kimi K3) and #3163 (DSA families) port the same change, gated on 0 decided disagreements with near-tie misses recorded.
- `kimi_k2`, `joyai_llm_flash`, `kimi_k25` and the Youtu types are covered by unit tests only; no checkpoints were run.
- Every multi-token step dequantizes the whole window, so chunked prefill and verify steps cost grows with the window length.
- A rank that is not a multiple of 64 falls back to one scale per row with a warning; no shipping checkpoint has one.
- CUDA GB10 and ROCm gfx1151 were not measured.
