# Fused decode kernel over an int8 MLA latent (M3 Ultra, 2026-11-19)

Issue #3156, epic #3158. #3155 stores the DeepSeek V2 absorbed latent as affine group-64 int8 and decodes it on a composed `quantized_matmul` path. This change gives the fused absorbed-decode kernel (#3121) a second variant that reads the packed latent in-kernel, and routes an unmasked single-query step on a quantized window to it.

**Outcome:** the kernel passes the trace gate and is faster than the composed quantized path only on short windows, so it takes windows up to 2048 keys (`FUSED_DECODE_QUANTIZED_MAX_KEYS`) rather than the fp16 kernel's 8192. int8 decode is 5.9% faster behind 512 tokens and 1.5% faster behind 1792, and unchanged from 2048 on, where the composed path still runs.

## Setup

- **Host:** Mac Studio, Apple M3 Ultra, 512 GB, macOS 26.6.2, Metal. Shared with two other units' builds and GPU jobs. Every GPU run held the shared GPU lock; the decode sweeps and the traces also ran under `scripts/with_indexers_paused.sh`. Load average (`uptime`) 3.1 falling to 0.9 across the final decode sweep and 0.9 to 1.0 across the traces; 17.7 falling to 4.6 during the crossover sweep, whose arms are interleaved run by run so the ratios are what to read.
- **Model:** `mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx` (27 layers, `kv_lora_rank` 512, `qk_rope_head_dim` 64, 16 heads).
- **Binary:** release build of this branch on `43f1c8d4`.
- **Arms:**
  - `int8 fused`: `--kv-cache-mode int8` (`MLXCEL_TRACE_KV_CACHE_MODE=int8` for the trace harness), the default after this change.
  - `fp16 fused`: the default fp16 latent (fused kernel up to 8192 keys, composed beyond).
  - `int8 composed`: `MLXCEL_MLA_ABSORBED=1 MLXCEL_MLA_FUSED_DECODE=0 --kv-cache-mode int8`, the #3155 path. `MLXCEL_MLA_ABSORBED=1` is needed because `MLXCEL_MLA_FUSED_DECODE=0` alone makes DeepSeek V2 default to the decompressed layout on Metal (#3131).

## Decode throughput (`mlxcel-bench-decode`, 128 tokens, 16 warmup, median tok/s)

Three arms interleaved round by round inside one lock hold, on the shipped binary.

| prompt | int8 fused (new) | fp16 fused | int8 composed (#3155) | int8 fused vs composed | int8 fused vs fp16 | decode path of the int8 fused arm |
|---|---|---|---|---|---|---|
| 512 (3 rounds) | 148.43 | 164.64 | 140.16 | +5.9% | -9.8% | `absorbed_fused` |
| 1792 (3 rounds) | 132.82 | 149.43 | 130.90 | +1.5% | -11.1% | `absorbed_fused` |
| 2048 (3 rounds) | 128.82 | 146.72 | 129.14 | -0.2% | -12.2% | `absorbed_composed` (window past 2048 keys) |
| 16384 (2 rounds) | 71.08 | 109.65 | 71.12 | -0.1% | -35.2% | `absorbed_composed` |

Round-to-round spread is under 1.2% for every arm. Every row's `mla_paths` counter names its arm: 3834 decode-step layer calls on the stated path and none on the other. A 2048-token prompt puts the first decode step at 2049 keys, so that row and the 16384 row run the composed path in both int8 arms and measure the same code. Prefill is the same in all arms (1373 to 1383 tok/s at 512, 1125 to 1128 at 16384).

The remaining distance to the fp16 latent at 512 tokens is not the attention kernel: the microbenchmark below puts the two kernels at the same time there. It is the per-step storage cost of the quantized cache (quantizing the new row and writing four buffers instead of two), which #3155 owns.

## Window crossover

`FUSED_DECODE_QUANTIZED_MAX_KEYS` was set from an in-model sweep on a build with the limit at 4096, same arms and interleaving:

| prompt | int8 fused | int8 composed | int8 fused vs composed |
|---|---|---|---|
| 512 (3 rounds) | 149.51 | 140.37 | +6.5% |
| 2048 (3 rounds) | 129.72 | 129.17 | +0.4% |
| 3840 (3 rounds) | 114.51 | 120.00 | -4.6% |

The two paths meet near 2048 keys and the composed path is ahead by 3840, so the limit is 2048. The fp16 kernel keeps its own 8192.

One attention block alone, at V2-Lite's geometry on synthetic rows (`mla::decode::fused_quant_decode_tests::quantized_kernel_window_crossover`, microseconds per step, median of 5 interleaved rounds of 200 steps, load average 2 to 6):

| keys | int8 fused | int8 composed | ratio | fp16 fused | fp16 composed | ratio |
|---|---|---|---|---|---|---|
| 512 | 224.7 | 272.4 | 0.83 | 225.1 | 271.8 | 0.83 |
| 2048 | 258.5 | 302.5 | 0.85 | 257.1 | 279.6 | 0.92 |
| 4096 | 307.2 | 323.3 | 0.95 | 285.1 | 287.5 | 0.99 |
| 8192 | 391.0 | 378.1 | 1.03 | 348.3 | 309.6 | 1.13 |
| 16384 | 551.0 | 486.2 | 1.13 | 469.4 | 335.5 | 1.40 |
| 32768 | 894.2 | 754.8 | 1.19 | 733.0 | 420.6 | 1.74 |

In isolation the int8 kernel stays ahead to about 6000 keys; inside the model it gives that up by 2048, so the limit follows the in-model number. The fp16 columns were not the subject here, but they put the fp16 kernel's crossover on this host near 4096 keys in isolation, below the 8192 measured on an M5 Max (#3121). The in-model fp16 crossover on M3 Ultra was not measured and `FUSED_DECODE_MAX_KEYS` is unchanged.

The kernel went through two forms. The first unpacked each code from its uint32 word and applied `code * scale + bias` per element before the shared score and value code; in-model it read +6.9% behind 512, -0.3% behind 2048 and -5.9% behind 4096 against the composed path. The shipped form reads the codes as bytes and factors each group's scale and bias out of both contractions (the form `quantized_matmul` uses). That removed a multiply-add per element and did not move the in-model crossover.

## Teacher-forced trace (`examples/logit_trace`, `--decided 2`)

Same corpus and method as #3155: `docs/architecture.md`, `docs/benchmarks.md` and `docs/turbo-kv-cache.md` from `3c0d74ec` concatenated, width-one decode with `MLXCEL_TRACE_BATCH_ROWS=1` and `MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1`. The candidate is the int8 latent on the fused kernel.

| reference | shape | decided disagree | top-1 disagree | largest reference top-two gap at a disagreement | ppl reference / candidate | ppl delta | candidate path counter |
|---|---|---|---|---|---|---|---|
| int8 composed | behind 1024 | 0 / 205 | 6 / 512 | 0.078 | 14.7041 / 14.6470 | -0.388% | absorbed_fused=13824 |
| int8 composed | behind 1920 | 0 / 113 | 3 / 256 | 0.109 | 13.6358 / 13.6577 | +0.161% | absorbed_fused=6912 |
| int8 composed | behind 4096 | byte-identical trace | 0 / 256 | | | | absorbed_composed=6912 |
| fp16 latent | behind 1024 | 0 / 209 | 10 / 512 | 0.078 | 14.6613 / 14.6470 | -0.098% | absorbed_fused=13824 |
| fp16 latent | behind 1920 | 0 / 115 | 5 / 256 | 0.078 | 13.6535 / 13.6577 | +0.031% | absorbed_fused=6912 |

The gate is zero decided disagreements and every top-1 disagreement where the reference's top two were within 0.11 logits. It holds against both references: no decided disagreement, and the largest gap at a disagreement is 0.109. Against the composed int8 path every disagreement puts the reference's choice at rank 2. Against the fp16 latent the int8 fused arm disagrees at 10 and 5 positions where #3155's composed int8 disagreed at 12 behind 1024.

The issue asked for traces behind 1024 and 4096. With the limit at 2048 a 4096-token context decodes on the composed path, and the trace of the default int8 arm there is byte-identical to the composed arm's, which shows the fallback engages. The 1920-token context (windows of 1921 to 1984 keys) is the long-window trace of the kernel itself.

## fp16 latent is unchanged

The dense kernel's Metal source assembled from the shared fragments is character-identical to the source before the change (checked against `43f1c8d4`), with the same kernel name and template arguments. End to end, the fp16 traces behind 1024 and 4096 from this build are byte-identical to the same traces from a release build of `43f1c8d4`, and so is the composed int8 trace behind 1024.

## Path counters on `mlxcel generate`

`MLXCEL_TRACE_MLA_STATS=1 mlxcel generate -m $M -p "..." -n 48`:

| flags | counters |
|---|---|
| `--kv-cache-mode int8` | `decompressed=54 absorbed_fused=1269` |
| `MLXCEL_MLA_ABSORBED=1 MLXCEL_MLA_FUSED_DECODE=0 --kv-cache-mode int8` | `decompressed=54 absorbed_composed=1269` |
| `MLXCEL_MLA_FUSED_DECODE=0 --kv-cache-mode int8` | `decompressed=1323` (the decompressed default of #3131, per-head int8) |
| fp16 default | `decompressed=54 absorbed_fused=1269` |

The 54 is the prompt's two multi-token forwards over 27 layers.

## Not measured

- **CUDA (GB10) and ROCm (gfx1151):** not reachable from this host, and the kernel has no port there (neither does the fp16 one). Both keep the composed quantized path, unchanged by this PR.
- **In-model fp16 kernel crossover on M3 Ultra**, see above.
- **DeepSeek-VL2:** same decoder and cache view, not run.

## Commands

```
M=models/deepseek-v2-lite-chat-4bit-mlx
mlxcel-bench-decode -m $M -p Hello -n 128 --warmup-tokens 16 --ignore-eos --prompt-tokens 512 --kv-cache-mode int8   # also 1792, 2048, 16384
MLXCEL_MLA_ABSORBED=1 MLXCEL_MLA_FUSED_DECODE=0 mlxcel-bench-decode ... --kv-cache-mode int8                          # composed arm
MLXCEL_TRACE_KV_CACHE_MODE=int8 MLXCEL_TRACE_MLA_STATS=1 MLXCEL_TRACE_BATCH_ROWS=1 MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1 \
  logit_trace $M corpus.txt 64 8 8 1024 > int8-fused-1k.tsv     # 64 4 8 1920, 64 4 8 4096
python3 scripts/compare_logit_traces.py int8-composed-1k.tsv int8-fused-1k.tsv --decided 2
cargo test --profile test-fast --features metal,accelerate -p mlxcel-core --lib \
  quantized_kernel_window_crossover -- --ignored --nocapture --test-threads=1
```

Rows: [decode.csv](mla-fused-quantized-decode-m3ultra-2026-11-19/decode.csv), [decode-crossover.csv](mla-fused-quantized-decode-m3ultra-2026-11-19/decode-crossover.csv), [kernel-microbench.csv](mla-fused-quantized-decode-m3ultra-2026-11-19/kernel-microbench.csv) and [logit_trace.csv](mla-fused-quantized-decode-m3ultra-2026-11-19/logit_trace.csv).
