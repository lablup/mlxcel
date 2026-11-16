# Absorbed MLA decode cost: quantized folds and a fused kernel (M5 Max, 2026-11-16)

Issue #3121. Follow-up to #3102 (PR #3118), which kept DeepSeek V2's decompressed cache as the default because absorbed decode was 9.9% slower at pp512/tg128.

## Setup

- **Host:** Apple M5 Max, 128 GB. The host was quiet when the sweep ran; a first sweep that overlapped another session's release build was discarded. Every run held the shared GPU lock, with a 75 s idle period before it.
- **Model:** `deepseek-v2-lite-chat-4bit-mlx`.
- **Binaries:**
  - main is `3efb3fb2`, built before #3119 and #3126 merged. Neither of them touches the MLA decode path.
  - The branch is `dc2a3527` on `8ee49deb`.
  - Both are release builds.
- **Arms:**
  - `MLXCEL_MLA_ABSORBED=1` is absorbed; on the branch, `MLXCEL_MLA_FUSED_DECODE=0` adds the quantized-fold-only arm.
  - `MLXCEL_MLA_ABSORBED=0` is the decompressed default.
- **Load lines:** the branch's absorbed runs print `folded 27/27 layers (27 quantized, 30.4 MiB)`; main prints `folded 27/27 layers` (dense f16, about 113 MiB).

## Profile (posted on the issue)

At 512 keys, the extra absorbed time per token came from:

| Cost | Share |
|---|---|
| dense f16 folds (absorb 0.14, unabsorb 0.24) | 0.36 ms |
| rope-score mask | 0.17 ms |
| unfused 512-wide attention block, latency bound | 0.93 ms |

Two fixes target these:
- The folds now run as batched `quantized_matmul` on `kv_b_proj`'s own packed planes.
- Unmasked single-query absorbed attention runs on a new Metal kernel: two key halves plus a merge, with f32 online softmax, for windows up to 8192 keys.

## Decode throughput (`mlxcel-bench-decode`, 128 tokens, 16 warmup)

| prompt | main absorbed | branch absorbed (fold + fused) | branch absorbed, fused off | branch decompressed |
|---|---|---|---|---|
| 512 (4 runs, median) | 182.6 | 201.4 | 187.8 | 205.5 |
| 2048 (2 runs) | 147.9 / 161.4 | 180.5 / 180.9 | | 169.9 / 170.3 |
| 16384 (2 runs) | 113.9 / 113.2 | 114.9 / 113.8 | | 73.4 / 73.5 |

At 512 tokens, absorbed decode goes from 11.1% behind decompressed (182.6 against 205.5 in this sweep) to 2.0% behind:
- The quantized fold recovers 5.2 tok/s.
- The fused kernel recovers another 13.6 tok/s.

At 2048 tokens, absorbed is now 6.2% ahead of decompressed. At 16384 tokens the window is over the 8192-key kernel limit, so the composed path runs as before and absorbed stays about 56% ahead. Main's two 2048 samples spread by 9%, while every branch arm stayed within 0.5%.

## Teacher-forced trace (`examples/logit_trace`, `--decided 2`, decompressed as reference)

The corpus is `docs/architecture.md`, `docs/benchmarks.md` and `docs/turbo-kv-cache.md` concatenated. Each trace is width-one decode with `MLXCEL_TRACE_BATCH_ROWS=1`, and `MLXCEL_TRACE_MLA_STATS=1` proves which path ran.

| shape | candidate | decided disagree | top-1 disagree | ppl reference / candidate | path counter |
|---|---|---|---|---|---|
| behind 1024 | absorbed fused | 0 / 209 | 4 / 512 | 14.6514 / 14.6903 | absorbed_fused=13824 |
| behind 1024 | absorbed composed | 0 / 209 | 8 / 512 | 14.6514 / 14.6692 | absorbed_composed=13824 |
| behind 4096 | absorbed fused | 0 / 106 | 0 / 256 | 12.9935 / 12.9915 | absorbed_fused=6912 |
| behind 8192 | absorbed (window over 8192, composed) | 0 / 78 | 1 / 128 | 4.5824 / 4.5829 | absorbed_composed=3456 |

Every disagreement falls where the reference's top two were within 0.094 logits. The fused kernel is in the same rounding class as the composed path it replaces.

## Not measured

- No always-absorbed family was run end to end. Kimi-VL has an unquantized `kv_b_proj`, so only the kernel applies there, and the DeepSeek V3, GLM-4.7-Flash and Kimi Linear/K3 call sites are covered by unit tests.
- Peak memory was not re-measured. The fold shrinks from about 113 MiB to 30.4 MiB of weights, and the cache layout is unchanged.

## Commands

```
S=scratchpad; M=models/deepseek-v2-lite-chat-4bit-mlx
MLXCEL_MLA_ABSORBED=1 mlxcel-bench-decode -m $M -p Hello -n 128 --warmup-tokens 16 --ignore-eos --prompt-tokens 512   # also 2048, 16384; =0 for decompressed; MLXCEL_MLA_FUSED_DECODE=0 for fold-only
MLXCEL_MLA_ABSORBED=1 MLXCEL_TRACE_MLA_STATS=1 MLXCEL_TRACE_BATCH_ROWS=1 MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1 \
  logit_trace $M corpus.txt 64 8 8 1024 > fused-1k.tsv     # 64 4 8 4096, 64 2 8 8192
python3 scripts/compare_logit_traces.py dec-1k.tsv fused-1k.tsv --decided 2
```

Rows: [decode.csv](mla-absorbed-decode-cost-m5max-2026-11-16/decode.csv) and [logit_trace.csv](mla-absorbed-decode-cost-m5max-2026-11-16/logit_trace.csv).
