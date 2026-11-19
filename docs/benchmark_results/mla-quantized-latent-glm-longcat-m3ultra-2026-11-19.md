# Affine int8 MLA latent on GLM-4.7-Flash and LongCat (M3 Ultra, 2026-11-19)

Issue #3161, Phase 2 of epic #3158. #3155 stored the absorbed MLA latent as MLX affine group-64 int8 with fp16 `k_pe` (672 bytes per token per layer at rank 512, rope 64, against 1152 for the fp16 latent), and #3160 moved the `DeepSeekV3Attention` families onto it. GLM-4.7-Flash (`glm4_moe_lite`) and LongCat (`longcat_flash`, `longcat_flash_ngram`, both loaded by `LongcatFlashNgramModel`) cached their latent pair through `KVCache::update_and_fetch` and had every quantized request substituted with fp16. This change writes that pair through `mla::MlaLatentCache` (`mla_batched::store_latent_step`) in both models, including GLM's batched decode and its MTP verify and rollback, and lets LongCat's model-owned caches take the per-layer mode table. Only int8 exists.

**Outcome:** GLM-4.7-Flash fails the gate and keeps fp16. On the build with the review's dtype fix (`4feb8419`, see "Defect in the first int8 arm"), the teacher-forced trace shows 1 decided-position disagreement behind 1024 tokens (of 193, reference top-two gap 2.75) and 0 behind 4096 (of 95), against a gate of zero at both lengths. That 1 equals the decided count between the two int8 kernels on the same stored rows (fused against composed: 1, at gap 6.5), and the composed int8 arm has 0 decided disagreements against fp16. So the failing position is within the int8 kernel-choice noise, but the gate is zero and it is not met. `glm4_moe_lite` is not added to `mla::QUANTIZED_LATENT_FAMILIES`, and `--kv-cache-mode int8` still resolves to fp16 with the substitution warning. The ported write path stays in the code behind that list (a per-layer flag read from it) and is covered by the unit tests. LongCat is not measured: the LongCat-Flash-Lite checkpoint (`longcat_flash_ngram`, 38.6 GB) was still downloading at about 0.5 MB/s when this run ended, and no other LongCat checkpoint is on the host. With no trace, LongCat is not added to the list either: its port is in place, off by default behind the same list, and covered by unit tests on tiny models of both types. Neither family's user-visible behavior changes in this PR, apart from LongCat's sizing (below).

## Setup

- **Host:** Mac Studio, Apple M3 Ultra, 512 GB, macOS 26.6.2, Metal. Two sibling units shared the host and the GPU lock: load average 3.2 to 4.5 during the gate traces on the fixed build, 4.5 to 8.4 during the first traces and 3.2 to 5.8 during the GLM decode sweep (`uptime` at each start and end, in the run logs).
- **Models:** `glm-4.7-flash-4bit` (`glm4_moe_lite`, 47 layers, 20 heads, `kv_lora_rank` 512, `qk_rope_head_dim` 64, `qk_nope_head_dim` 192, `v_head_dim` 256, 4-bit affine weights).
- **Binary:** release builds (`--features metal,accelerate`) of this branch with `glm4_moe_lite` temporarily in `QUANTIZED_LATENT_FAMILIES`: the gate below on `4feb8419` (with the dtype fix), the first trace, the decode sweep and the int8 generate check on the pre-fix code (`4598cba1` with the family listed). The production check uses a build with the family unlisted.
- **Isolation:** every GPU run held the shared GPU lock (`lockf -k .mlxcel-gpu.lock`). The decode sweep ran under `scripts/with_indexers_paused.sh`, with the two arms interleaved round by round inside one lock per model.
- **Arms:** `fp16 latent` (default) and `int8 latent` (`--kv-cache-mode int8`; `MLXCEL_TRACE_KV_CACHE_MODE=int8` for the trace harness). The trace harness (`examples/logit_trace`) now hands the requested mode to a model that reports a mode table and still reads the handed caches, as `DirectEngine` does; before this change GLM's int8 arm silently ran fp16.
- **Corpus:** `docs/architecture.md`, `docs/benchmarks.md` and `docs/turbo-kv-cache.md` at `3c0d74ec`, concatenated (167,657 bytes). Shapes: `64 8 8 1024` (8 chunks of 64 traced positions behind a 1024-token context) and `64 4 8 4096`, with `MLXCEL_TRACE_BATCH_ROWS=1 MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1`, so every traced position is a one-token decode step.

## GLM-4.7-Flash: teacher-forced trace on the fixed build (`--decided 2`)

| shape | decided disagree | top-1 disagree | largest reference top-two gap at a disagreement | flips above 0.11 | ppl reference / candidate | ppl delta | path counter (int8) |
|---|---|---|---|---|---|---|---|
| behind 1024 | **1 / 193** | 40 / 512 | 2.750 | 18 | 21.4881 / 20.8288 | -3.068% | decompressed=376 absorbed_fused=24064 |
| behind 4096 | 0 / 95 | 25 / 256 | 1.000 | 15 | 53.6498 / 53.4191 | -0.430% | decompressed=188 absorbed_composed=12032 |

Flips above the 0.11-logit near-tie bar, by reference gap: behind 1024, 0.375, 16 at 0.5 and 2.75; behind 4096, 0.125, 0.375, 12 at 0.5 and 1.0. GLM's logits sit around 100 to 130 in bf16, so they are spaced 0.5 apart there; a 0.11 bar is below that resolution for this checkpoint, which is why most flips exceed it. Logit delta on the reference's choice: p50 0.5, p90 4.0, p99 17.0 behind 1024; p50 0.5, p90 2.125, p99 17.0 behind 4096. The `decompressed` counter is the context pass (one per layer per chunk forward); every traced decode step reads the int8 latent absorbed (`absorbed_fused` behind 1024, `absorbed_composed` behind 4096 where the window is past the fused variant's 2048 keys).

### Kernel-noise floor on the fixed build (behind 1024)

| comparison | top-1 disagree | decided disagree | flips above 0.11 (gaps) | ppl | path counters (reference / candidate) |
|---|---|---|---|---|---|
| int8 composed (`MLXCEL_MLA_FUSED_DECODE=0`) against int8 fused | 27 / 512 | **1** (gap 6.5) | 14 (0.125, 11 at 0.5, 1.0, 6.5) | 21.0364 / 20.8288 | composed / fused |
| fp16 against int8 composed | 43 / 512 | 0 (largest gap 0.5) | 20 (0.375, 19 at 0.5) | 21.4881 / 21.0364 (-2.102%) | composed / composed |
| fp16 against fp16 with `MLXCEL_MLA_FUSED_DECODE=0` | 0 / 512 | 0 | 0 | identical | composed / composed |

The only two-kernel comparison available is the int8 pair, which reads the same stored rows: it disagrees at one decided position, the same count as fp16 against the shipped (fused) int8 arm. The fp16 latent has no second kernel on this trace (both arms record `absorbed_composed`), so the fp16 row compares a path with itself and is not a noise floor. Stated plainly: the shipped int8 arm's decided count (1) is at, not below, the int8 fused-against-composed decided count (1), and it is above the zero the gate requires.

## First measurement, before the dtype fix

The rows below were taken on the int8 arm with the defect described next; they are superseded by the tables above and kept for the record.

| shape | decided disagree | top-1 disagree | largest gap at a disagreement | flips above 0.11 | ppl reference / candidate | ppl delta | path counter (int8) |
|---|---|---|---|---|---|---|---|
| behind 1024 | 2 / 193 | 48 / 512 | 4.000 | 29 | 21.4881 / 19.7229 | -8.215% | decompressed=376 absorbed_fused=24064 |
| behind 4096 | 2 / 95 | 32 / 256 | 4.000 | 22 | 53.6498 / 46.4779 | -13.368% | decompressed=188 absorbed_composed=12032 |

### Noise floors (before the fix)

Path counters below are the ones each run printed (`# mla_paths` in its stderr). GLM's fp16 latent never reaches the fp16 fused kernel on this trace: the scalar forward has no fused dispatch, and the batched-row run at one row recorded `absorbed_composed` too.

| comparison (behind 1024) | top-1 disagree | decided disagree | ppl | path counters (reference / candidate) |
|---|---|---|---|---|
| fp16 (scalar reference) against fp16 (batched rows) | 0 / 512 | 0 | identical | composed / composed |
| fp16 against fp16 with `MLXCEL_MLA_FUSED_DECODE=0` | 0 / 512 | 0 | identical | composed / composed |
| int8 (scalar reference) against int8 (batched rows) | 0 / 512 | 0 | identical | fused / fused |
| int8 with `MLXCEL_MLA_FUSED_DECODE=0` against int8, fused | 33 / 512 | 1 (gap 8.75) | 19.5795 / 19.7229 | composed / fused |
| fp16 against int8 with `MLXCEL_MLA_FUSED_DECODE=0` | 47 / 512 | 3 | 21.4881 / 19.5795 | composed / composed |
| fp16 against int8, 1024-token prefill-width trace (no batch rows) | 29 / 512 | 0 (largest gap 0.5) | 15.3259 / 15.1106 (-1.405%) | prefill only |

There is no fp16 kernel noise floor on this checkpoint: the first three rows compare a path with itself, so their zeros say nothing about how GLM responds to a different kernel. The only two-kernel comparison is the int8 pair, and it reads the same stored rows: at GLM's geometry (20 heads, rank 512, rope 64, bf16 queries, 1088 keys) the fused and composed int8 kernels agree within 2e-3 of the largest output element, one or two bf16 units in the last place (synthetic check run during review). That one-ulp perturbation of the attention output already moves 33 positions, one of them decided at gap 8.75, so on this checkpoint the zero-decided gate sits below the spread from kernel choice alone. Logit deltas on the reference's choice tell the same story: fp16 against int8 is p50 0.5, p90 4.25, p99 14.25 (one bf16 unit at these magnitudes is 0.5), and int8 fused against int8 composed is p50 0.5, p90 1.75, p99 8.0.

### Defect in the first int8 arm (found in review, fixed in `4feb8419`)

The int8 arm above ran its multi-token steps in f32. `LatentWindow::into_dense` returns the dequantized window in the cache's fp16 sidecar dtype; GLM-4.7-Flash runs in bf16, so the context pass mixed bf16 queries with fp16 keys, MLX promoted the attention to f32, and the promotion carried through `o_proj` into the residual stream. Every later layer of the 1024- and 4096-token context pass, and the latents it stored, were computed in f32 on the int8 arm and in bf16 on the fp16 arm. The int8 prefill-width trace shows it directly: its logits are not on the bf16 grid (logit-delta p50 0.354 where every bf16 comparison is 0.5), and its forward took 76 to 91 ms per chunk against 62 to 76 ms for fp16. Decode steps were unaffected (both quantized kernels return the query dtype), so the `decompressed` counts are the context pass and no decode step dequantized.

The fix casts the dequantized window to the queries' dtype in GLM and LongCat (`mla_batched::dense_latent_window`); fp16 windows are untouched. The first measurement predates the fix; the gate tables at the top of this document were re-measured on `4feb8419`. The fix moved the decided count from 2 to 1 behind 1024 and from 2 to 0 behind 4096, and the perplexity deltas from -8.2% and -13.4% to -3.1% and -0.4%. A second asymmetry stays and is not a defect of the int8 path: GLM's fp16 single-sequence decode computes the rope score in bf16 and adds it to the SDPA input as a bf16 mask, while both int8 kernels compute the score in f32. That plausibly accounts for part of the lower int8 perplexity on the decode traces, which the prefill-width trace (-1.4%, now also explained by the f32 promotion) cannot.


## GLM-4.7-Flash: decode throughput (`mlxcel-bench-decode`, 128 tokens, 16 warmup, median tok/s)

Measured on the port with `glm4_moe_lite` listed, before the dtype fix and before the family was removed. The fix touches only multi-token steps over a quantized window, so the single-token decode rows here are not affected by it; prefill, which runs on bf16 now instead of f32, was not re-measured.

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
