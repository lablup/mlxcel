# Affine int8 / int4 MLA latent on DeepSeek V2 (M3 Ultra, 2026-11-19)

Issue #3155, Phase 1 of epic #3158. Absorbed MLA decode caches the shared latent `ckv` plus `k_pe`, 1152 bytes per token per layer on DeepSeek-V2-Lite at fp16. Before this change a quantized KV request declined the latent and stored decompressed per-head K/V, so `--kv-cache-mode int8` cost 5120 bytes per token per layer (#3154 substituted fp16 in the interim). This change stores the latent itself as MLX affine group-64 with fp16 scale and bias per group and `k_pe` left fp16: 672 bytes at 8 bits, 416 at 4 bits.

**Outcome:** int8 passes the quality gate and ships. int4 fails it and is not shipped: `int4` is not a KV cache mode and `--kv-bits 4` with the uniform scheme stays a startup error.

## Setup

- **Host:** Mac Studio, Apple M3 Ultra, 512 GB, macOS 26.6.2, Metal. Load average 1.5 to 2.3 during the runs (`uptime` at each end: 1.73 and 1.85 for the traces, 1.84 and 1.53 for the decode sweep).
- **Model:** `mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx` (27 layers, `kv_lora_rank` 512, `qk_rope_head_dim` 64, 16 heads).
- **Binary:** release build of this branch on top of `3c0d74ec`, with both bit widths wired. The int4 arm was measured on that build, before the mode was removed for failing the gate. The int8 code path is the same in the shipped tree: the int8 traces were re-run on the final binary and are byte-identical (see Verification).
- **Isolation:** every GPU run held the shared GPU lock (`lockf -k .mlxcel-gpu.lock`). The decode sweep also ran under `scripts/with_indexers_paused.sh`, with the four arms interleaved round by round inside one lock.
- **Arms:**
  - `fp16 latent`: the default (absorbed, fused Metal kernel up to 8192 keys, composed beyond).
  - `int8 latent`, `int4 latent`: `--kv-cache-mode int8|int4` (`MLXCEL_TRACE_KV_CACHE_MODE` for the trace harness).
  - `int8 decompressed`: `MLXCEL_MLA_ABSORBED=0 --kv-cache-mode int8`, the per-head per-token int8 cache.

## Teacher-forced trace (`examples/logit_trace`, `--decided 2`)

Reference is the fp16 latent. The corpus is `docs/architecture.md`, `docs/benchmarks.md` and `docs/turbo-kv-cache.md` from `3c0d74ec` concatenated (50338 tokens). Width-one decode with `MLXCEL_TRACE_BATCH_ROWS=1` and `MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1`.

| candidate | shape | decided disagree | top-1 disagree | largest reference top-two gap at a disagreement | ppl reference / candidate | ppl delta | path counter |
|---|---|---|---|---|---|---|---|
| int8 latent | behind 1024 | 0 / 209 | 12 / 512 | 0.078 | 14.6613 / 14.7041 | +0.292% | absorbed_composed=13824 |
| int8 latent | behind 4096 | 0 / 105 | 3 / 256 | 0.109 | 12.9589 / 12.8767 | -0.634% | absorbed_composed=6912 |
| int4 latent | behind 1024 | 0 / 209 | 45 / 512 | 0.859 | 14.6613 / 14.6607 | -0.004% | absorbed_composed=13824 |
| int4 latent | behind 4096 | 0 / 105 | 14 / 256 | 0.859 | 12.9589 / 13.2174 | +1.995% | absorbed_composed=6912 |

The gate is zero decided disagreements and every top-1 disagreement at a near-tie, meaning the reference's top two within 0.11 logits.

- **int8 passes.** No decided disagreement, and all 15 top-1 disagreements fall where the reference's top two were within 0.109 logits. The reference's choice landed at rank 2 or 3 in 14 of them and rank 5 in one. Logit delta on the reference's choice: p50 0.047, p99 0.39 to 0.41.
- **int4 fails.** No decided disagreement either, but 59 top-1 disagreements, 8 of them where the reference's top two were more than 0.5 logits apart (largest 0.859), with the reference's choice as low as rank 8. Logit delta on the reference's choice: p50 0.22 to 0.23, p99 1.36 to 1.72, max 2.66. Perplexity moved by +2.0% behind 4096.

The path counter proves which path ran: every traced decode step of a quantized arm is `absorbed_composed`, with `absorbed_fused=0` and `absorbed_split_kv=0`. The fp16 reference rows show `absorbed_fused=13824` and `6912`. The `decompressed=216` / `108` in every row is the context pass (one multi-token forward per chunk per layer), which a quantized arm runs over the dequantized window.

## Decode throughput (`mlxcel-bench-decode`, 128 tokens, 16 warmup, median tok/s)

| prompt | fp16 latent | int8 latent | vs fp16 latent | int4 latent | vs fp16 latent | int8 decompressed | int8 latent vs decompressed |
|---|---|---|---|---|---|---|---|
| 512 (3 rounds) | 163.59 | 138.85 | -15.1% | 137.67 | -15.8% | 120.97 | 1.15x |
| 2048 (3 rounds) | 146.69 | 127.58 | -13.0% | 129.74 | -11.6% | 88.50 | 1.44x |
| 16384 (2 rounds) | 109.89 | 70.67 | -35.7% | 77.26 | -29.7% | 25.41 | 2.78x |

Round-to-round spread is under 2% for every arm. The fp16 latent decodes on the fused kernel behind 512 and 2048 tokens and on the composed path behind 16384 (the kernel's limit is 8192 keys); the quantized latent decodes on the composed `quantized_matmul` path at every length. The fused kernel for a quantized latent is #3156. Prefill throughput is the same for all four arms (1362 to 1384 tok/s at 512 and 2048, 1125 to 1129 at 16384).

## Single-query decode layout (security and performance review, 2026-11-19)

`quantized_matmul` broadcasts the one latent head over the `H` query heads, so each head reads the whole packed window. Three layouts of the one-query int8 step were measured against the code above, with the same binary flags, interleaved under one lock with indexers paused (load average 1.0 to 2.3), 128 tokens after 16 warmup:

| layout | 512 | 2048 | 16384 | trace behind 1024 against fp16 |
|---|---|---|---|---|
| as measured above | 137.3 to 138.8 | 127.5 to 128.5 | 70.5 to 71.1 | byte-identical rerun: 12 / 512 top-1, largest gap 0.078 |
| heads folded into the score and value contractions | 135.2 to 135.6 | 107.2 to 107.4 | 38.5 | not run |
| heads folded into the latent score only | 142.5 to 142.7 | 131.5 to 132.6 | 80.0 to 80.4 | 13 / 512 top-1, 0 decided, largest gap 0.219, ppl 14.6319 (-0.20%) |
| heads folded into the rope score only (shipped) | not run | 129.5 to 130.1 | 71.3 | byte-identical to the measured int8 trace; behind 4096 also matches (3 / 256, gap 0.109, ppl 12.8767) |

Folding the value sum moves it onto the tiled `qmm` kernel, which contracts over the window with less parallelism than the per-head split-K `qvm` and is slower at every length. Folding the latent score puts it on `qmv_wide`, which reads each packed group once for all heads: 3% faster behind 512 and 2048 and 13% behind 16384. It changes the summation order, and one top-1 flip lands at a 0.219-logit gap, outside the gate's 0.11 near-tie bar, so it is not shipped without a decision on that bar. The rope-score fold is numerically identical and ships (1% to 1.5%).

The other avoidable cost on this path is the per-step f32 cast of the window's group scales and biases (`[B, 1, S, rank/64]` each). MLX's affine `quantized_matmul` promotes them to the f32 query dtype anyway, so removing it needs either f32 sidecars (704 instead of 672 bytes per token per layer) or an fp16 query, and both change what the gate measured.

## 16384-token prefill transient peak

MLX allocator high-water mark for the measured pass (prefill plus 128 decode tokens), which is also the whole-run peak. Weights are 8.87 GB resident after load.

| arm | phase peak | above the loaded weights |
|---|---|---|
| fp16 latent | 11.45 GB | 2.58 GB |
| int8 latent | 11.23 GB | 2.36 GB |
| int4 latent | 11.13 GB | 2.26 GB |
| int8 decompressed | 16.95 GB | 8.08 GB |

The quantized latent dequantizes the live window once per prefill step, and that transient does not raise the peak above the fp16 latent's: the int8 arm peaks 0.22 GB below it.

## Cache size

| layout | bytes per token per layer | 27 layers at 16384 tokens |
|---|---|---|
| decompressed fp16 | 10240 | 4.53 GB |
| decompressed int8 (per-token absmax, with scales) | 5120 + 64 | 2.29 GB |
| fp16 latent | 1152 | 0.51 GB |
| int8 latent | 672 | 0.30 GB |
| int4 latent (not shipped) | 416 | 0.18 GB |

`mlxcel inspect --kv-cache-mode int8` reports the int8 latent rate (see Verification), and a cache built at V2-Lite's geometry allocates exactly 672 bytes per reserved row (`mla::quant_latent_tests::runtime_bytes_per_token_are_672_on_v2_lite`).

## Verification on the shipped tree

Run on the release build after `int4` was removed.

- **Traces:** the int8 traces behind 1024 and 4096 and the fp16 trace behind 1024 are byte-identical to the measured ones above, with the same path counters.
- **`mlxcel generate`** (`MLXCEL_TRACE_MLA_STATS=1`, 46 tokens): fp16 prints `decompressed=54 absorbed_fused=1269`, `--kv-cache-mode int8` prints `decompressed=54 absorbed_composed=1269` with every other counter zero, and `MLXCEL_MLA_ABSORBED=0 --kv-cache-mode int8` prints `decompressed=1323`. The 54 is the prompt's two multi-token forwards over 27 layers. The banner reads `int8 (affine group-64 MLA latent, fp16 rope keys)` under absorbed decode and `int8 (per-token absmax quantization)` without it.
- **`mlxcel inspect --max-tokens 16384`:** `--kv-cache-mode int8` reports `MLA compressed latent, affine-quantized (672 bytes/token/layer, shared across heads)`, 18144 bytes per token (27 x 672) and 283.5 MiB; fp16 reports 31104 bytes per token (27 x 1152) and 486.0 MiB; `MLXCEL_MLA_ABSORBED=0 --kv-cache-mode int8` reports 138240 bytes per token (27 x 5120). `--json` gives `kv_bytes_per_token.int8 = 18144`.
- **`mlxcel serve`:** `/v1/cache/stats` reports `kv_cache_mode_effective: int8` for both `--kv-bits 8` and `--kv-cache-mode int8`. A 613-token chat prompt repeated returns identical text with 608 cached tokens, and a second turn reuses 624, so a parked quantized latent is restored and decoded. The prompt-cache store held 1744 reserved rows in each run: 31,643,136 bytes under `--kv-cache-mode int8` (1744 x 27 x 672), 32,480,256 under `--kv-bits 8` with the default `--kv-skip-last-layer` (1744 x (26 x 672 + 1152)), and 54,245,376 at fp16 (1744 x 27 x 1152). Those are the runtime `nbytes` of the parked caches, and they equal the `inspect` rate.
- **Refusals:** `--kv-cache-mode int4` and `MLXCEL_TRACE_KV_CACHE_MODE=int4` fail as an unknown mode, `--kv-bits 4` with the uniform scheme keeps its startup error, `--kv-bits 8 --kv-group-size 32` fails startup for this model, and `--kv-cache-mode fp16+turbo4` resolves to fp16 with a warning that names `--kv-cache-mode int8`.

## Not measured

- **CUDA (GB10) and ROCm (gfx1151):** neither host is reachable from this one. The path uses only `quantize_weights`, `quantized_matmul` and `dequantize`, which the quantized weight path already runs on both backends, and contains no Metal-only call, but no trace or throughput row exists for them.
- **DeepSeek-VL2:** the `deepseek_v2` decoder inside it takes the same path through the same cache view. It was not run here.

## Commands

```
M=models/deepseek-v2-lite-chat-4bit-mlx
mlxcel-bench-decode -m $M -p Hello -n 128 --warmup-tokens 16 --ignore-eos --prompt-tokens 512   # also 2048, 16384; add --kv-cache-mode int8; MLXCEL_MLA_ABSORBED=0 for decompressed
MLXCEL_TRACE_MLA_STATS=1 MLXCEL_TRACE_BATCH_ROWS=1 MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1 \
  logit_trace $M corpus.txt 64 8 8 1024 > fp16-1k.tsv     # 64 4 8 4096
MLXCEL_TRACE_KV_CACHE_MODE=int8 MLXCEL_TRACE_MLA_STATS=1 MLXCEL_TRACE_BATCH_ROWS=1 MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1 \
  logit_trace $M corpus.txt 64 8 8 1024 > int8-1k.tsv
python3 scripts/compare_logit_traces.py fp16-1k.tsv int8-1k.tsv --decided 2
```

Rows: [decode.csv](mla-quantized-latent-m3ultra-2026-11-19/decode.csv) and [logit_trace.csv](mla-quantized-latent-m3ultra-2026-11-19/logit_trace.csv).
