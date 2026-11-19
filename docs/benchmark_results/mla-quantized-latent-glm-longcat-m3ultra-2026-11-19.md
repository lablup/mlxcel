# Affine int8 MLA latent on GLM-4.7-Flash and LongCat (M3 Ultra, 2026-11-19)

Issue #3161, Phase 2 of epic #3158. #3155 stored the absorbed MLA latent as MLX affine group-64 int8 with fp16 `k_pe` (672 bytes per token per layer at rank 512, rope 64, against 1152 for the fp16 latent), and #3160 moved the `DeepSeekV3Attention` families onto it. GLM-4.7-Flash (`glm4_moe_lite`) and LongCat (`longcat_flash`, `longcat_flash_ngram`, both loaded by `LongcatFlashNgramModel`) cached their latent pair through `KVCache::update_and_fetch` and had every quantized request substituted with fp16. This change writes that pair through `mla::MlaLatentCache` (`mla_batched::store_latent_step`) in both models, including GLM's batched decode and its MTP verify and rollback, and lets LongCat's model-owned caches take the per-layer mode table. Only int8 exists.

**Outcome:** GLM-4.7-Flash fails the gate and keeps fp16. Its teacher-forced trace shows 2 decided-position disagreements behind 1024 tokens and 2 behind 4096, against a gate of zero, so `glm4_moe_lite` is not added to `mla::QUANTIZED_LATENT_FAMILIES` and `--kv-cache-mode int8` still resolves to fp16 with the substitution warning. The ported write path stays in the code behind that list (a per-layer flag read from it) and is covered by the unit tests. LongCat is not measured: the LongCat-Flash-Lite checkpoint (`longcat_flash_ngram`, 38.6 GB) was still downloading at about 0.5 MB/s when this run ended, and no other LongCat checkpoint is on the host. With no trace, LongCat is not added to the list either: its port is in place, off by default behind the same list, and covered by unit tests on tiny models of both types. Neither family's user-visible behavior changes in this PR, apart from LongCat's sizing (below).

## Setup

- **Host:** Mac Studio, Apple M3 Ultra, 512 GB, macOS 26.6.2, Metal. Two sibling units shared the host and the GPU lock: load average 4.5 to 8.4 during the GLM traces and 3.2 to 5.8 during the GLM decode sweep (`uptime` at each start and end, in the run logs).
- **Models:** `glm-4.7-flash-4bit` (`glm4_moe_lite`, 47 layers, 20 heads, `kv_lora_rank` 512, `qk_rope_head_dim` 64, `qk_nope_head_dim` 192, `v_head_dim` 256, 4-bit affine weights).
- **Binary:** release build (`--features metal,accelerate`) of this branch on top of `3f5679a4`, with `glm4_moe_lite` in `QUANTIZED_LATENT_FAMILIES` for the GLM measurements; it was removed after the trace and the production check below uses the rebuilt binary.
- **Isolation:** every GPU run held the shared GPU lock (`lockf -k .mlxcel-gpu.lock`). The decode sweep ran under `scripts/with_indexers_paused.sh`, with the two arms interleaved round by round inside one lock per model.
- **Arms:** `fp16 latent` (default) and `int8 latent` (`--kv-cache-mode int8`; `MLXCEL_TRACE_KV_CACHE_MODE=int8` for the trace harness). The trace harness (`examples/logit_trace`) now hands the requested mode to a model that reports a mode table and still reads the handed caches, as `DirectEngine` does; before this change GLM's int8 arm silently ran fp16.
- **Corpus:** `docs/architecture.md`, `docs/benchmarks.md` and `docs/turbo-kv-cache.md` at `3c0d74ec`, concatenated (167,657 bytes). Shapes: `64 8 8 1024` (8 chunks of 64 traced positions behind a 1024-token context) and `64 4 8 4096`, with `MLXCEL_TRACE_BATCH_ROWS=1 MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1`, so every traced position is a one-token decode step.

## GLM-4.7-Flash: teacher-forced trace (`--decided 2`)

| shape | decided disagree | top-1 disagree | largest reference top-two gap at a disagreement | flips above 0.11 | ppl reference / candidate | ppl delta | path counter (int8) |
|---|---|---|---|---|---|---|---|
| behind 1024 | **2 / 193** | 48 / 512 | 4.000 | 29 | 21.4881 / 19.7229 | -8.215% | decompressed=376 absorbed_fused=24064 |
| behind 4096 | **2 / 95** | 32 / 256 | 4.000 | 22 | 53.6498 / 46.4779 | -13.368% | decompressed=188 absorbed_composed=12032 |

The decided disagreements are at reference top-two gaps of 2.0 and 4.0 at both lengths. The flips above the 0.11-logit near-tie bar, by reference gap: behind 1024, 0.25, 0.375, 22 at 0.5, 0.625, 0.75, 1.5, 2.0 and 4.0; behind 4096, 0.125 (twice), 0.25, 0.375, 9 at 0.5, 0.625, 5 at 1.0, 1.032, 2.0 and 4.0. GLM's logits sit around 100 to 130 in bf16, so they are spaced 0.5 apart there and most gaps land on multiples of 0.125 to 0.5; a 0.11 bar is below that resolution for this checkpoint, which is why so many flips exceed it. The decided count is the gate, and it fails.

The `decompressed` counter is the prefill (one per layer per chunk forward); every traced decode step reads the int8 latent absorbed (`absorbed_fused` behind 1024, `absorbed_composed` behind 4096 where the window is past the fused variant's 2048 keys).

### Noise floors

| comparison (behind 1024) | top-1 disagree | decided disagree | ppl |
|---|---|---|---|
| fp16, composed (scalar reference) against fp16, fused (batched rows) | 0 / 512 | 0 | identical |
| fp16 against fp16 with `MLXCEL_MLA_FUSED_DECODE=0` | 0 / 512 | 0 | identical |
| int8, fused (scalar reference) against int8 (batched rows, `absorbed_composed`) | 0 / 512 | 0 | identical |
| int8 with `MLXCEL_MLA_FUSED_DECODE=0` against int8, fused | 33 / 512 | 1 (gap 8.75) | 19.5795 / 19.7229 |
| fp16 against int8 with `MLXCEL_MLA_FUSED_DECODE=0` | 47 / 512 | 3 | 21.4881 / 19.5795 |
| fp16 against int8, 1024-token prefill-width trace (no batch rows) | 29 / 512 | 0 (largest gap 0.5) | 15.3259 / 15.1106 (-1.405%) |

The fp16 latent's own kernel choice changes no position, so the fp16 noise floor is zero and the int8 disagreements are not fp16 kernel noise. On GLM's single-sequence fp16 decode the fused kernel does not run (the forward records `absorbed_composed`); the batched-row path does run it (`absorbed_fused`) and agrees at every position. Within int8 the two kernels are not interchangeable either: the `MLXCEL_MLA_FUSED_DECODE=0` int8 arm disagrees with the fused one at 33 positions, one decided, which says this checkpoint amplifies small latent perturbations through its 47 layers. Read at prefill width (all 64 positions of a chunk in one forward, the dequantized window) the int8 latent has no decided disagreement, so the failure is specific to decode over the quantized latent.

Perplexity falls on every int8 arm. That is not a quality signal: it moves the same way on the prefill-width trace that has no decided disagreement, and the gate reads top-1 agreement, not perplexity.

## GLM-4.7-Flash: decode throughput (`mlxcel-bench-decode`, 128 tokens, 16 warmup, median tok/s)

Measured on the port with `glm4_moe_lite` listed, before it was removed; it is what a later re-enable would start from.

| prompt | fp16 latent | int8 latent | int8 vs fp16 | fp16 path counter | int8 path counter |
|---|---|---|---|---|---|
| 512 (3 rounds) | 82.23 | 81.74 | -0.6% | absorbed_composed=6674 | absorbed_fused=6674 |
| 2048 (3 rounds) | 79.60 | 70.58 | -11.3% | absorbed_composed=6674 | absorbed_composed=6674 |
| 16384 (2 rounds) | 60.28 | 36.62 | -39.3% | absorbed_composed=6674 | absorbed_composed=6674 |

Round-to-round spread is under 1.2% for every arm. GLM's single-sequence forward has no fp16 fused-kernel dispatch (only its batched-row decode calls the fp16 kernel), so the fp16 latent decodes on the composed path at every length here, and at 512 tokens the int8 fused kernel almost closes the gap; behind 2048 and 16384 both arms are composed and the int8 cost matches the DeepSeek V2 and Kimi-VL figures.

## GLM-4.7-Flash: integration checks

- **`mlxcel generate`** with the port enabled (`MLXCEL_TRACE_MLA_STATS=1`, 48 tokens): fp16 prints `decompressed=94 absorbed_composed=2209`; `--kv-cache-mode int8` printed the banner `int8 (affine group-64 MLA latent, fp16 rope keys)` and `decompressed=94 absorbed_fused=2209`, with no `decompressed` counter on any decode step.
- **Production (this branch as shipped):** `--kv-cache-mode int8` on `glm-4.7-flash-4bit` prints the substitution warning (`this family stores its latent pair as fp16; the quantized latent is available on deepseek_v2 and on deepseek_v3, ...`), the banner `fp16 (requested int8; effective fp16; applied to 47 of 47 layers)` and `absorbed_composed` only. `mlxcel inspect --max-tokens 16384` reports 846.0 MiB (`MLA compressed latent (576 elems/token/layer)`) under both `fp16` and `int8`.
- **MTP:** no GLM-4.7-Flash MTP drafter checkpoint is on this host. The verify block over an int8 latent (per-row `quantized_absorbed_attention` on prefix slices), its byte-identity with the decode chain, the rollback to the accepted window and the drafter following the target's mode table are covered by `models::glm4_moe_lite::latent_tests` and `the_drafter_cache_follows_the_target_mode_table`.

## LongCat

- **Port:** both MLA sub-layers of each `LongcatFlashNgramModel` decoder layer write through `store_latent_step`, and the model-owned caches follow the per-layer mode table. An `int8` layer keeps the affine latent only when the model opted in (`quantized_latent`, read from `mla::stores_quantized_latent_pair` at load, off today); otherwise it is reported and built as fp16 with one warning, as a Turbo layer is.
- **Sizing fix:** `kv_arch` now counts two latent caches per LongCat decoder layer at either mode. LongCat-Flash-Lite (14 layers, rank 512, rope 64) sizes at 2 x 14 x 1152 = 32,256 bytes per token for the fp16 latent; before this change the estimate (and so `mlxcel inspect`, the preflights and the advisor) counted half of it.
- **Tests (tiny models, both `model_type`s):** with the model opted in, an int8 table is kept and reported, a Turbo layer falls back to fp16 beside a kept int8 layer, every int8 decode step reads the quantized latent absorbed (`absorbed_fused` or `absorbed_composed` at least steps x 2 x layers, no `decompressed`) and tracks the fp16 logits within 5e-2 of the largest logit, an int8 `DirectEngine` run stores the latent and matches the fp16 greedy run, and at rank 512 each sub-layer cache holds the affine rate per row. Without opting in, an int8 table and an int8 engine run are fp16.
- **To measure:** the commands below on `longcat-flash-lite-4bit`, with `longcat_flash` and `longcat_flash_ngram` added to `mla::QUANTIZED_LATENT_FAMILIES` for the int8 arm (the trace harness and the CLI both read it).

## Not measured

- **LongCat-Flash-Lite (`longcat_flash_ngram`):** checkpoint download incomplete on this host when the run ended (about 9 of 38.6 GB, at about 0.5 MB/s). No trace, decode or path-counter row exists for LongCat.
- **`longcat_flash` (the 560B LongCat-Flash):** checkpoint not downloaded on this host. It loads the same `LongcatFlashNgramModel` and is covered by `models::longcat_flash_ngram::latent_tests`, which run every test on both `model_type`s.
- **CUDA (GB10) and ROCm (gfx1151):** neither host is reachable from this one. The composed int8 path has no Metal-only call, but no trace or throughput row exists for those backends.

## Commands

```
M=/Users/devops/Development/mlxcel/models/glm-4.7-flash-4bit   # or longcat-flash-lite-4bit
MLXCEL_TRACE_MLA_STATS=1 MLXCEL_TRACE_BATCH_ROWS=1 MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1 \
  logit_trace $M corpus.txt 64 8 8 1024 > fp16-1k.tsv     # 64 4 8 4096 for the 4k rows
MLXCEL_TRACE_KV_CACHE_MODE=int8 MLXCEL_TRACE_MLA_STATS=1 MLXCEL_TRACE_BATCH_ROWS=1 MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1 \
  logit_trace $M corpus.txt 64 8 8 1024 > int8-1k.tsv     # add MLXCEL_MLA_FUSED_DECODE=0 for the kernel-noise arms
python3 scripts/compare_logit_traces.py fp16-1k.tsv int8-1k.tsv --decided 2
mlxcel-bench-decode -m $M -p Hello -n 128 --warmup-tokens 16 --ignore-eos --prompt-tokens 512   # also 2048, 16384; add --kv-cache-mode int8
mlxcel inspect -m $M --max-tokens 16384 --kv-cache-mode int8
```

Rows: [decode.csv](mla-quantized-latent-glm-longcat-m3ultra-2026-11-19/decode.csv) and [logit_trace.csv](mla-quantized-latent-glm-longcat-m3ultra-2026-11-19/logit_trace.csv).
