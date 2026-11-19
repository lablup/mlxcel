# Affine int8 MLA latent on the DeepSeekV3Attention families (M3 Ultra, 2026-11-19)

Issue #3160, Phase 2 of epic #3158. #3155 stored the absorbed MLA latent as MLX affine group-64 int8 with fp16 `k_pe` on DeepSeek V2 (672 bytes per token per layer at rank 512, rope 64, against 1152 for the fp16 latent). The `DeepSeekV3Attention` families cached their latent pair through `KVCache::update_and_fetch` directly and had every quantized request substituted with fp16. This change writes that pair through `mla::MlaLatentCache` in the single-sequence forward and the shared batched decode helper, so `--kv-cache-mode int8` and `--kv-bits 8` keep the int8 latent on `deepseek_v3`, `kimi_k2`, `joyai_llm_flash`, `kimi_vl`, `kimi_k25`, `youtu`, `youtu_llm` and `youtu_vl`. Only int8 exists; 4-bit failed the #3155 gate and is not a mode.

**Outcome:** zero decided-position disagreements on both checkpoints behind 1024 and 4096 tokens. One top-1 flip per checkpoint behind 1024 is outside the 0.11-logit near-tie bar #3155 used: 0.117 on Moonlight and 0.180 on Kimi-VL, the latter well beyond the 0.109 fp16 fused-versus-composed noise floor; every other flip is inside the bar. This remains an open item. See "Near-tie bar" below.

## Setup

- **Host:** Mac Studio, Apple M3 Ultra, 512 GB, macOS 26.6.2, Metal. Sibling builds were running on the host: load average 7.2 to 15.7 during the traces and 3.5 to 9.5 during the decode sweep (`uptime` at each start and end, in the run logs).
- **Models:** `mlx-community/Moonlight-16B-A3B-Instruct-4-bit` (`model_type` `deepseek_v3`, 8192-token context, `tiktoken.model` tokenizer, loaded by mlxcel's tiktoken path) and `Kimi-VL-A3B-Thinking-4bit` (`kimi_vl`, its `text_config` is the same `deepseek_v3` backbone with a 131072-token context). Both: 27 layers, 16 heads, `kv_lora_rank` 512, `qk_rope_head_dim` 64.
- **Binary:** release build (`--features metal,accelerate`) of this branch on top of `43f1c8d4`. The pre-change baseline is a release build of `43f1c8d4`.
- **Isolation:** every GPU run held the shared GPU lock (`lockf -k .mlxcel-gpu.lock`). The decode sweep ran under `scripts/with_indexers_paused.sh`, with the two arms interleaved round by round inside one lock per model.
- **Arms:** `fp16 latent` (default: absorbed, fused Metal kernel up to 8192 keys, composed beyond) and `int8 latent` (`--kv-cache-mode int8`; `MLXCEL_TRACE_KV_CACHE_MODE=int8` for the trace harness).

## Teacher-forced trace (`examples/logit_trace`, `--decided 2`)

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

#3155 gated on zero decided disagreements and every top-1 flip within 0.11 logits of the reference's top two. The decided gate passes everywhere. Behind 1024, Moonlight has one flip at 0.117 and Kimi-VL one at 0.180; the other 19 flips across the two traces are at or under 0.109, and nothing exceeds the bar behind 4096. For scale, the fp16 latent's own kernel choice (fused against composed, `MLXCEL_MLA_FUSED_DECODE=0`, same binary) flips 1 / 512 positions on Moonlight (largest gap 0.023) and 6 / 512 on Kimi-VL (largest 0.109), so the two out-of-bar flips are beyond the fp16 kernel noise floor. Whether 0.18 is acceptable for these families is a decision for the issue, not settled here.

## Decode throughput (`mlxcel-bench-decode`, 128 tokens, 16 warmup, median tok/s)

| model | prompt | fp16 latent | int8 latent | int8 vs fp16 |
|---|---|---|---|---|
| Moonlight | 512 (3 rounds) | 161.93 | 134.81 | -16.7% |
| Moonlight | 2048 (3 rounds) | 144.61 | 123.18 | -14.8% |
| Moonlight | 8064 (2 rounds, requested 16384) | 102.90 | 94.82 | -7.9% |
| Kimi-VL | 512 (3 rounds) | 161.77 | 134.17 | -17.1% |
| Kimi-VL | 2048 (3 rounds) | 144.74 | 123.36 | -14.8% |
| Kimi-VL | 16384 (2 rounds) | 107.59 | 69.05 | -35.8% |

Round-to-round spread is under 1% for every arm. Moonlight's context is 8192, so the bench clamps a 16384-token request to 8064 tokens (128 reserved for generation); at that length the fp16 latent still decodes on the fused kernel. Kimi-VL runs the same backbone to 16384 tokens, where the fp16 latent is on the composed path too, and the gap matches the V2-Lite figure from #3155 (-35.7%). The int8 latent decodes on the composed `quantized_matmul` path at every length; a fused kernel that reads it is #3156. Prefill throughput is within 0.6% between arms (1369 to 1380 tok/s at 512 and 2048, 1265 at 8064, 1124 to 1127 at 16384). The 16384-token measured-phase peak on Kimi-VL is 11.31 GB for int8 against 11.52 GB for fp16.

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
