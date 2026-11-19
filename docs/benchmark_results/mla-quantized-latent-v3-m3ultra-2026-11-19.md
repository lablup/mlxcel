# Affine int8 MLA latent on the DeepSeekV3Attention families (M3 Ultra, 2026-11-19)

Issue #3160, Phase 2 of epic #3158. #3155 stored the absorbed MLA latent as MLX affine group-64 int8 with fp16 `k_pe` on DeepSeek V2 (672 bytes per token per layer at rank 512, rope 64, against 1152 for the fp16 latent). The `DeepSeekV3Attention` families cached their latent pair through `KVCache::update_and_fetch` directly and had every quantized request substituted with fp16. This change writes that pair through `mla::MlaLatentCache` in the single-sequence forward and the shared batched decode helper, so `--kv-cache-mode int8` and `--kv-bits 8` keep the int8 latent on `deepseek_v3`, `kimi_k2`, `joyai_llm_flash`, `kimi_vl`, `kimi_k25`, `youtu`, `youtu_llm` and `youtu_vl`. Only int8 exists; 4-bit failed the #3155 gate and is not a mode.

**Outcome:** zero decided-position disagreements on both checkpoints behind 1024 and 4096 tokens. The sections from Setup to Not measured were measured before #3156 merged, when every int8 decode step ran the composed path. With #3156 merged (the code this change ships on), an unmasked int8 decode step over a window of up to 2048 keys runs the fused kernel's quantized variant instead; see [After #3156 merged](#after-3156-merged-2026-11-19) for the re-measured rows behind 1024 and at 512 and 2048 tokens. On that code the trace behind 1024 still has no decided disagreement; Moonlight has two top-1 flips outside the 0.11-logit near-tie bar #3155 used (0.117 and 0.328) and Kimi-VL none (largest 0.094). Before #3156 it was one per checkpoint (0.117 on Moonlight, 0.180 on Kimi-VL). See "Near-tie bar" below.

## Setup

- **Host:** Mac Studio, Apple M3 Ultra, 512 GB, macOS 26.6.2, Metal. Sibling builds were running on the host: load average 7.2 to 15.7 during the traces and 3.5 to 9.5 during the decode sweep (`uptime` at each start and end, in the run logs).
- **Models:** `mlx-community/Moonlight-16B-A3B-Instruct-4-bit` (`model_type` `deepseek_v3`, 8192-token context, `tiktoken.model` tokenizer, loaded by mlxcel's tiktoken path) and `Kimi-VL-A3B-Thinking-4bit` (`kimi_vl`, its `text_config` is the same `deepseek_v3` backbone with a 131072-token context). Both: 27 layers, 16 heads, `kv_lora_rank` 512, `qk_rope_head_dim` 64.
- **Binary:** release build (`--features metal,accelerate`) of this branch on top of `43f1c8d4`. The pre-change baseline is a release build of `43f1c8d4`.
- **Isolation:** every GPU run held the shared GPU lock (`lockf -k .mlxcel-gpu.lock`). The decode sweep ran under `scripts/with_indexers_paused.sh`, with the two arms interleaved round by round inside one lock per model.
- **Arms:** `fp16 latent` (default: absorbed, fused Metal kernel up to 8192 keys, composed beyond) and `int8 latent` (`--kv-cache-mode int8`; `MLXCEL_TRACE_KV_CACHE_MODE=int8` for the trace harness).

## Teacher-forced trace (`examples/logit_trace`, `--decided 2`), before #3156

Reference is the fp16 latent of the same checkpoint. The corpus is the #3155 corpus: `docs/architecture.md`, `docs/benchmarks.md` and `docs/turbo-kv-cache.md` from `3c0d74ec` concatenated. Width-one decode with `MLXCEL_TRACE_BATCH_ROWS=1` and `MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1`.

| model | shape | decided disagree | top-1 disagree | largest reference top-two gap at a disagreement | flips above 0.11 | ppl reference / candidate | ppl delta | path counter |
|---|---|---|---|---|---|---|---|---|
| Moonlight (`deepseek_v3`) | behind 1024 | 0 / 192 | 8 / 512 | 0.117 | 1 | 18.6816 / 18.8579 | +0.944% | absorbed_composed=13824 |
| Moonlight (`deepseek_v3`) | behind 4096 | 0 / 58 | 5 / 256 | 0.039 | 0 | 22.9467 / 22.9386 | -0.035% | absorbed_composed=6912 |
| Kimi-VL (`kimi_vl`) | behind 1024 | 0 / 188 | 13 / 512 | 0.180 | 1 | 17.8839 / 18.0216 | +0.770% | absorbed_composed=13824 |
| Kimi-VL (`kimi_vl`) | behind 4096 | 0 / 51 | 3 / 256 | 0.094 | 0 | 35.5331 / 35.3756 | -0.443% | absorbed_composed=6912 |

- Every flip put the reference's choice at rank 2 in the candidate, except one Kimi-VL flip at rank 3. Logit delta on the reference's choice: p50 0.031 to 0.047, p99 0.24 to 1.0 (Kimi-VL behind 1024 is the 1.0, with a max of 3.08).
- The path counter proves the path: every traced int8 decode step is `absorbed_composed`, with `absorbed_fused=0` and `absorbed_split_kv=0`. The fp16 rows show `absorbed_fused=13824` / `6912`. The `decompressed=216` / `108` is the context pass (one multi-token forward per chunk per layer), which the int8 arm runs over the dequantized window.
- **fp16 is unchanged:** the fp16 trace behind 1024 from this branch is byte-identical to the one from the `43f1c8d4` binary on both checkpoints.

### Near-tie bar

#3155 gated on zero decided disagreements and every top-1 flip within 0.11 logits of the reference's top two. The decided gate passes everywhere. Behind 1024, Moonlight has one flip at 0.117 and Kimi-VL one at 0.180; the other 19 flips across the two traces are at or under 0.109, and nothing exceeds the bar behind 4096. For scale, the fp16 latent's own kernel choice (fused against composed, `MLXCEL_MLA_FUSED_DECODE=0`, same binary) flips 1 / 512 positions on Moonlight (largest gap 0.023) and 6 / 512 on Kimi-VL (largest 0.109), so the two out-of-bar flips are beyond the fp16 kernel noise floor. Whether 0.18 is acceptable for these families is a decision for the issue, not settled here. These figures are for the pre-#3156 composed int8 path. On the merged code (fused int8 behind 1024) Moonlight's out-of-bar flips are at 0.117 and 0.328 and Kimi-VL has none; see [After #3156 merged](#after-3156-merged-2026-11-19).

## Decode throughput (`mlxcel-bench-decode`, 128 tokens, 16 warmup, median tok/s), before #3156

| model | prompt | fp16 latent | int8 latent | int8 vs fp16 |
|---|---|---|---|---|
| Moonlight | 512 (3 rounds) | 161.93 | 134.81 | -16.7% |
| Moonlight | 2048 (3 rounds) | 144.61 | 123.18 | -14.8% |
| Moonlight | 8064 (2 rounds, requested 16384) | 102.90 | 94.82 | -7.9% |
| Kimi-VL | 512 (3 rounds) | 161.77 | 134.17 | -17.1% |
| Kimi-VL | 2048 (3 rounds) | 144.74 | 123.36 | -14.8% |
| Kimi-VL | 16384 (2 rounds) | 107.59 | 69.05 | -35.8% |

Round-to-round spread is under 1% for every arm. Moonlight's context is 8192, so the bench clamps a 16384-token request to 8064 tokens (128 reserved for generation); at that length the fp16 latent still decodes on the fused kernel. Kimi-VL runs the same backbone to 16384 tokens, where the fp16 latent is on the composed path too, and the gap matches the V2-Lite figure from #3155 (-35.7%). In this pre-#3156 sweep the int8 latent decodes on the composed `quantized_matmul` path at every length. With #3156 merged, the 512-token row moves onto the fused kernel; the rows behind 2048 and longer keep the composed path (see [After #3156 merged](#after-3156-merged-2026-11-19)). Prefill throughput is within 0.6% between arms (1369 to 1380 tok/s at 512 and 2048, 1265 at 8064, 1124 to 1127 at 16384). The 16384-token measured-phase peak on Kimi-VL is 11.31 GB for int8 against 11.52 GB for fp16.

## Cache size and sizing

| layout | bytes per token per layer | 27 layers at 16384 tokens |
|---|---|---|
| fp16 latent | 1152 | 486.0 MiB |
| int8 latent | 672 | 283.5 MiB |

- `mlxcel inspect --max-tokens 16384 --kv-cache-mode int8` on both checkpoints reports `MLA compressed latent, affine-quantized (672 bytes/token/layer, shared across heads)` and 283.5 MiB; fp16 reports 486.0 MiB.
- **Runtime `nbytes` equal the rate:** `mlxcel serve` held the prompt-cache store at 304 reserved rows on Kimi-VL and 560 on Moonlight after two identical chat requests. Under `--kv-cache-mode int8` that was 5,515,776 and 10,160,640 bytes (rows x 27 x 672); under `--kv-bits 8`, which keeps the last layer fp16, 5,661,696 and 10,429,440 bytes (rows x (26 x 672 + 1152)).

## Integration checks

- **`mlxcel generate`** (`MLXCEL_TRACE_MLA_STATS=1`, short prompt): fp16 prints `decompressed=54 absorbed_fused=945` (Moonlight) / `1269` (Kimi-VL); `--kv-cache-mode int8` prints `decompressed=54 absorbed_composed=945` / `1269` with no substitution warning and the banner `int8 (affine group-64 MLA latent, fp16 rope keys)`. Moonlight's int8 output text is identical to fp16 on that prompt. `MLXCEL_MLA_ABSORBED=0` does not change the layout on these families: they always cache the latent and one query is always absorbed.
- **`mlxcel serve`:** `--kv-bits 8` and `--kv-cache-mode int8` both start with no substitution warning, the scheduler logs `kv_cache_mode_effective=int8` (26 of 27 layers under `--kv-bits 8`, 27 under `--kv-cache-mode int8`), `/v1/cache/stats` reports `kv_cache_mode_effective: int8`, and the repeated request reuses 48 cached tokens and returns the same text, so a parked quantized latent is restored and decoded.
- **Refusals:** a Turbo mode on these families resolves to fp16 with a warning that names `--kv-cache-mode int8`; `glm4_moe_lite` and the other latent-pair families still resolve `int8` to fp16.

## Not measured

- **CUDA (GB10) and ROCm (gfx1151):** neither host is reachable from this one. The path is the #3155 composed decode (`quantize_weights`, `quantized_matmul`, `dequantize`), which has no Metal-only call, but no trace or throughput row exists for those backends.
- **`kimi_k2`, `joyai_llm_flash`, `kimi_k25`, the Youtu types:** no checkpoint on this host. They run the same `DeepSeekV3Attention` and are covered by the unit tests (`models::deepseek_v3::latent_tests`, the rule 1 and sizing tests).

## After #3156 merged (2026-11-19)

#3156 (PR #3173) merged into main and into this branch (`9af1ce46`, a merge of `a00629a6`). `mla::quantized_absorbed_attention`, which `DeepSeekV3Attention` and the batched decode helper call for a quantized window, now tries `mla::decode::fused_absorbed_decode_quantized` first: Metal, one unmasked query, 8-bit, group 64, a power-of-two rank from 64 to 1024 and a rope width that is a multiple of 32 (both checkpoints qualify at rank 512, rope 64), up to `FUSED_DECODE_QUANTIZED_MAX_KEYS` = 2048 keys. Longer windows fall back to the composed path. Re-measured with a release build (`--features metal,accelerate`) of `9af1ce46` on the same host; load average 1.5 to 1.9 during the traces and 1.2 to 1.5 during the decode sweep, which ran under `scripts/with_indexers_paused.sh` with the arms interleaved round by round inside one lock per model.

**Trace behind 1024** (same corpus, commands and reference as above):

| model | decided disagree | top-1 disagree | largest reference top-two gap at a disagreement | flips above 0.11 | ppl reference / candidate | ppl delta | path counter (int8) |
|---|---|---|---|---|---|---|---|
| Moonlight (`deepseek_v3`) | 0 / 192 | 7 / 512 | 0.328 | 2 (0.117, 0.328) | 18.6816 / 18.8238 | +0.761% | decompressed=216 absorbed_fused=13824 |
| Kimi-VL (`kimi_vl`) | 0 / 188 | 12 / 512 | 0.094 | 0 | 17.8839 / 18.0607 | +0.988% | decompressed=216 absorbed_fused=13824 |

- Every int8 decode step behind 1024 now records `absorbed_fused` (`absorbed_composed=0`); the window there never exceeds 1536 keys.
- Every flip put the reference's choice at rank 2, except one Kimi-VL flip at rank 3. Logit delta on the reference's choice: p50 0.031 / 0.047, p99 0.34 / 0.95, max 0.66 / 2.78 (Moonlight / Kimi-VL).
- Against the pre-#3156 int8 trace (composed int8 as reference, fused int8 as candidate): 0 decided disagreements on both checkpoints, 7 / 512 top-1 flips on each, largest gap 0.094 on Moonlight and 0.469 on Kimi-VL (two flips above 0.11 there). The kernel change moves which near-tie positions flip; it adds no decided disagreement.
- **fp16 is unchanged:** the fp16 trace behind 1024 from `9af1ce46` is byte-identical to the one recorded above on both checkpoints, and its counter is still `absorbed_fused=13824`.
- Behind 4096 the decode window is longer than 2048 keys, so the int8 arm keeps the composed path and the 4096 rows above still describe it. They were not re-run.

**Decode throughput** (3 rounds per arm, ABAB interleaved):

| model | prompt | fp16 latent | int8 latent | int8 vs fp16 | int8 path counter | int8 vs the pre-#3156 int8 row |
|---|---|---|---|---|---|---|
| Moonlight | 512 | 159.07 | 142.45 | -10.4% | absorbed_fused=3834 | +5.7% (134.81) |
| Moonlight | 2048 | 141.80 | 122.97 | -13.3% | absorbed_composed=3834 | -0.2% (123.18) |
| Kimi-VL | 512 | 158.53 | 142.95 | -9.8% | absorbed_fused=3834 | +6.5% (134.17) |
| Kimi-VL | 2048 | 141.85 | 122.61 | -13.6% | absorbed_composed=3834 | -0.6% (123.36) |

- Round-to-round spread is under 1.1% for every arm. The fp16 arm reads 1.8% to 2.0% below the earlier sweep at 512 tokens and 1.9% to 2.0% below at 2048 with identical code and counters, so the int8 against fp16 ratio is the comparable figure: it narrows from -16.7% / -17.1% to -10.4% / -9.8% at 512 and moves from -14.8% to -13.3% / -13.6% at 2048, where the int8 arm itself is unchanged within 0.6%.
- Behind 2048 tokens every int8 decode step is composed: the measured pass starts above 2048 keys. The 8064 and 16384 rows above are composed for the same reason and were not re-run.
- Prefill is within 0.5% between arms (1368 to 1375 tok/s).

Rows: [decode-after-3156.csv](mla-quantized-latent-v3-m3ultra-2026-11-19/decode-after-3156.csv) and [logit_trace-after-3156.csv](mla-quantized-latent-v3-m3ultra-2026-11-19/logit_trace-after-3156.csv).

## Commands

```
M=/Users/devops/Development/mlxcel/models/Kimi-VL-A3B-Thinking-4bit   # or moonlight-16b-a3b-instruct-4bit
mlxcel-bench-decode -m $M -p Hello -n 128 --warmup-tokens 16 --ignore-eos --prompt-tokens 512   # also 2048, 16384; add --kv-cache-mode int8
MLXCEL_TRACE_MLA_STATS=1 MLXCEL_TRACE_BATCH_ROWS=1 MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1 \
  logit_trace $M corpus.txt 64 8 8 1024 > fp16-1k.tsv     # 64 4 8 4096
MLXCEL_TRACE_KV_CACHE_MODE=int8 MLXCEL_TRACE_MLA_STATS=1 MLXCEL_TRACE_BATCH_ROWS=1 MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1 \
  logit_trace $M corpus.txt 64 8 8 1024 > int8-1k.tsv
python3 scripts/compare_logit_traces.py fp16-1k.tsv int8-1k.tsv --decided 2
mlxcel inspect -m $M --max-tokens 16384 --kv-cache-mode int8
```

Rows: [decode.csv](mla-quantized-latent-v3-m3ultra-2026-11-19/decode.csv) and [logit_trace.csv](mla-quantized-latent-v3-m3ultra-2026-11-19/logit_trace.csv).
