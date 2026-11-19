# Affine int8 MLA latent on GLM-4.7-Flash and LongCat (M3 Ultra, 2026-11-19/20)

Issue #3161, Phase 2 of epic #3158. #3155 stored the absorbed MLA latent as MLX affine group-64 int8 with fp16 `k_pe` (672 bytes per token per layer at rank 512, rope 64, against 1152 for the fp16 latent), and #3160 moved the `DeepSeekV3Attention` families onto it. GLM-4.7-Flash (`glm4_moe_lite`) and LongCat (`longcat_flash`, `longcat_flash_ngram`, both loaded by `LongcatFlashNgramModel`) cached their latent pair through `KVCache::update_and_fetch` and had every quantized request substituted with fp16. This change writes that pair through `mla::MlaLatentCache` (`mla_batched::store_latent_step`) in both models, including GLM's batched decode and its MTP verify and rollback, and lets LongCat's model-owned caches take the per-layer mode table. Only int8 exists.

**Outcome:** GLM-4.7-Flash fails the gate and keeps fp16. On the build with the review's dtype fix (`4feb8419`, see "Defect in the first int8 arm"), the teacher-forced trace shows 1 decided-position disagreement behind 1024 tokens (of 193, reference top-two gap 2.75) and 0 behind 4096 (of 95), against a gate of zero at both lengths. That 1 equals the decided count between the two int8 kernels on the same stored rows (fused against composed: 1, at gap 6.5), and the composed int8 arm has 0 decided disagreements against fp16. So the failing position is within the int8 kernel-choice noise, but the gate is zero and it is not met. `glm4_moe_lite` is not added to `mla::QUANTIZED_LATENT_FAMILIES`, and `--kv-cache-mode int8` still resolves to fp16 with the substitution warning. The ported write path stays in the code behind that list (a per-layer flag read from it) and is covered by the unit tests. LongCat-Flash-Lite (`longcat_flash_ngram`) passes: 0 decided-position disagreements behind 1024 tokens (of 239) and behind 4096 (of 121), with 9 and 5 top-1 flips above the 0.11 near-tie bar, the largest at a reference gap of 0.375. Both LongCat types are added to `mla::QUANTIZED_LATENT_FAMILIES`, so `--kv-cache-mode int8` on LongCat stores the affine latent in both MLA sub-layers of every layer; the 560B `longcat_flash` loads the same `LongcatFlashNgramModel` and is not measured on this host. Running the checkpoint at all needed two loader and forward fixes that predate this issue (see "LongCat-Flash-Lite: defects found on the way"). GLM keeps fp16 in user-visible behavior; LongCat's sizing also changes (below).

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

## LongCat-Flash-Lite: teacher-forced trace (`--decided 2`)

Model `longcat-flash-lite-4bit` (`longcat_flash_ngram`, 14 layers of two MLA sub-layers each, 32 heads, `kv_lora_rank` 512, `qk_rope_head_dim` 64, `qk_nope_head_dim` 128, `v_head_dim` 128, hidden 3072, n-gram embeddings, 4-bit affine weights). Release build of `ddd01a49` plus the two fixes below, with both LongCat types temporarily listed; same corpus, shapes and harness settings as GLM. All six arms ran inside one hold of the GPU lock, load average 1.8 to 3.1.

| shape | decided disagree | top-1 disagree | largest reference top-two gap at a disagreement | flips above 0.11 (gaps) | ppl reference / candidate | ppl delta | path counter (int8) |
|---|---|---|---|---|---|---|---|
| behind 1024 | **0 / 239** | 15 / 512 | 0.375 | 9 (six at 0.125, 0.188, 0.25, 0.375) | 14.1036 / 14.0010 | -0.727% | decompressed=224 absorbed_fused=14336 |
| behind 4096 | **0 / 121** | 9 / 256 | 0.250 | 5 (0.125, 0.125, 0.188, 0.188, 0.25) | 9.9146 / 9.7928 | -1.228% | decompressed=112 absorbed_composed=7168 |

Logit delta on the reference's choice: p50 0.25, p90 0.625, p99 1.625 behind 1024; p50 0.25, p90 0.625, p99 1.125 behind 4096. `decompressed` is the context pass, one per MLA sub-layer per chunk (8 x 14 x 2 = 224); every traced decode step reads the int8 latent absorbed, on the fused kernel behind 1024 and composed behind 4096.

### Kernel-noise floor (behind 1024)

| comparison | top-1 disagree | decided disagree | flips above 0.11 (gaps) | ppl | path counters (reference / candidate) |
|---|---|---|---|---|---|
| int8 composed (`MLXCEL_MLA_FUSED_DECODE=0`) against int8 fused | 11 / 512 | 0 (largest gap 0.875) | 8 (five at 0.125, 0.25, 0.5, 0.875) | 13.9708 / 14.0010 | composed / fused |
| fp16 against int8 composed | 18 / 512 | 0 (largest gap 0.625) | 11 (six at 0.125, 0.188, 0.25, 0.375, 0.375, 0.625) | 14.1036 / 13.9708 (-0.942%) | composed / composed |
| fp16 against fp16 with `MLXCEL_MLA_FUSED_DECODE=0` | 0 / 512 | 0 | 0 | identical | composed / composed |

As on GLM, LongCat's single-sequence fp16 decode has no fused-kernel dispatch, so the fp16 pair compares one path with itself. The int8 kernel pair (same stored rows, two kernels) has 11 top-1 flips against 15 for fp16 against the shipped int8 arm, and neither has a decided disagreement: the int8 latent's flips sit at the scale of the kernel choice.

### LongCat-Flash-Lite: defects found on the way

Neither is in the latent path; both broke the checkpoint at load or at the first long forward on every KV mode, on `main` too.

- **Quantized n-gram embeddings loaded as plain embeddings.** The MLX 4-bit checkpoint quantizes `word_embeddings` and the 12 n-gram `embedders` (packed U32 rows with `.scales` and `.biases`), and `NgramEmbedding` read them as dense `Embedding`s, so every lookup returned 1/8-wide integer rows and the first `post_proj` threw `[quantized_matmul] Last dimension of first input with shape (..., 32) does not match`. It now loads them with `UnifiedEmbedding::from_weights`, which picks the quantized table by its `.scales` key; the non-n-gram `model.embed_tokens` path does the same.
- **The MoE handed `SwitchGLU` a `[B, L, hidden]` input.** `SwitchGLU::forward` takes `[tokens, hidden]` with `[tokens, k]` indices; given the batch-first shape it read `B` as the token count and `L` as `k`. Under 64 positions the direct path broadcast to the right answer by accident, so short generations worked, but a forward of 64 or more positions took the sorted path, whose reshape threw (`Cannot reshape array of size 196608 into shape (1,1,1)` on a 64-token chunk). The MoE now flattens tokens before the call and restores the shape after. `a_64_token_prefill_takes_the_sorted_expert_path_and_matches_decode` covers both LongCat types.

## LongCat-Flash-Lite: decode throughput (`mlxcel-bench-decode`, 128 tokens, 16 warmup, median tok/s)

Same build as the trace, both LongCat types listed, the two arms interleaved round by round inside one lock hold under `scripts/with_indexers_paused.sh` (load average 0.7 to 2.1).

| prompt | fp16 latent | int8 latent | int8 vs fp16 | fp16 path counter | int8 path counter |
|---|---|---|---|---|---|
| 512 (3 rounds) | 125.52 | 135.53 | +8.0% | absorbed_composed=3976 | absorbed_fused=3976 |
| 2048 (3 rounds) | 127.40 | 107.59 | -15.5% | absorbed_composed=3976 | absorbed_composed=3976 |
| 16384 (2 rounds) | 99.20 | 50.39 | -49.2% | absorbed_composed=3976 | absorbed_composed=3976 |

Round-to-round spread is under 0.9% for every arm. Each run's `decompressed` counter is the prefill only (56 at 512 and 2048, 448 at 16384, where the prompt prefills in chunks); every decode step reads the latent absorbed. As on GLM, the fp16 latent always decodes composed: behind 512 tokens the int8 fused kernel is faster than fp16 composed, and once the window passes the fused kernel's 2048 keys both arms are composed and int8 pays its dequantization on every step (-15.5% and -49.2% here, against -11.3% and -39.3% on GLM).

## LongCat: integration checks

- **`mlxcel generate`** (`MLXCEL_TRACE_MLA_STATS=1`, 48 tokens, greedy): fp16 prints `decompressed=56 absorbed_composed=1316` at 122.64 tok/s; `--kv-cache-mode int8` prints the banner `int8 (affine group-64 MLA latent, fp16 rope keys)` with no substitution warning and `decompressed=56 absorbed_fused=1316` at 127.26 tok/s, with no `decompressed` counter on any decode step. The banner's `applied to 0 of 0 layers` is the model-owned cache count, which the generate banner does not see; the inspect line below is the sized figure.
- **`mlxcel inspect --max-tokens 16384`:** 504.0 MiB under `fp16` (`MLA compressed latent (576 elems/token/layer)`) and 294.0 MiB under `int8` (`MLA compressed latent, affine-quantized (672 bytes/token/layer)`), that is 32,256 and 18,816 bytes per token over both sub-layer caches.
- **Sizing fix:** `kv_arch` now counts two latent caches per LongCat decoder layer at either mode. LongCat-Flash-Lite (14 layers, rank 512, rope 64) sizes at 2 x 14 x 1152 = 32,256 bytes per token for the fp16 latent and 2 x 14 x 672 = 18,816 for the int8 latent; before this change the estimate (and so `mlxcel inspect`, the preflights and the advisor) counted half of it.
- **Tests (tiny models, both `model_type`s):** the production model opts in from the list and keeps an int8 table; an int8 table is kept and reported, a Turbo layer falls back to fp16 beside a kept int8 layer, every int8 decode step reads the quantized latent absorbed (`absorbed_fused` or `absorbed_composed` at least steps x 2 x layers, no `decompressed`) and tracks the fp16 logits within 5e-2 of the largest logit, an int8 `DirectEngine` run stores the latent and matches the fp16 greedy run, and at rank 512 each sub-layer cache holds the affine rate per row.

## Not measured

- **`longcat_flash` (the 560B LongCat-Flash):** checkpoint not on this host. It loads the same `LongcatFlashNgramModel` as the measured LongCat-Flash-Lite and is covered by `models::longcat_flash_ngram::latent_tests`, which run every test on both `model_type`s; it is listed on that shared code path, not on a trace of its own.
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
