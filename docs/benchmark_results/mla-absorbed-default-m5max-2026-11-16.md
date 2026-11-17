# Absorbed MLA decode as the DeepSeek V2 default, and Kimi-VL on the fused kernel (M5 Max, 2026-11-16)

Issue #3131. Follow-up to #3121 (PR #3129), which brought absorbed decode to within 2% of the decompressed layout at pp512/tg128 and recorded that both of #3102's switch conditions now hold.

## Setup

- **Host:** Apple M5 Max, 128 GB, shared with another agent's builds and GPU jobs. Every GPU run held the shared GPU lock, so no two GPU jobs overlapped, but the first throughput sweep ran while the other agent was compiling (load average 7 to 8). Absolute decode rates are 10 to 18% below #3121's quiet-host sweep; the arms are interleaved run by run, so the ratios are what to read.
- **Binary:** release build of this branch on `5ee08ec6`, built in the worktree before measuring.
- **Models:** `deepseek-v2-lite-chat-4bit-mlx`, `kimi-vl-a3b-thinking-4bit` (`deepseek_v3` decoder, always absorbed).
- **Arms:**
  - DeepSeek V2: the variable unset (the new default, absorbed) against `MLXCEL_MLA_ABSORBED=0` (decompressed).
  - Kimi-VL: the default (fused kernel) against `MLXCEL_MLA_FUSED_DECODE=0` (composed absorbed path).
- **Load lines:** DeepSeek V2 with the variable unset logs `folded 27/27 layers (27 quantized, 30.4 MiB); KV cache 10240 -> 1152 bytes/token/layer (8.9x)` at info level; `MLXCEL_MLA_ABSORBED=1` prints the same line to stderr, and `=0` prints nothing.

## Decode throughput (`mlxcel-bench-decode`, 128 tokens, 16 warmup, median tok/s)

| prompt | V2 default (absorbed) | V2 `MLXCEL_MLA_ABSORBED=0` | | Kimi-VL fused | Kimi-VL fused off | |
|---|---|---|---|---|---|---|
| 512 (6 runs, two sweeps) | 168.5 | 171.3 | -1.7% | 159.5 | 149.4 | +6.7% |
| 2048 (4 runs, two sweeps) | 153.5 | 150.9 | +1.8% | 149.8 | 139.2 | +7.5% |
| 16384 (1 run) | 105.8 | 71.4 | +48% | | | |

The V2 gap at 512 tokens is the same -2% #3121 measured on a quiet host (201.4 against 205.5). One decompressed sample at 512 read 132.8 tok/s; the median discards it, and without it the decompressed median is 172.3. The fused kernel speeds Kimi-VL decode by 7% at both lengths, close to the 7% it gave V2's absorbed path at 512 tokens in #3121.

## Teacher-forced trace (`examples/logit_trace`, `--decided 2`)

The corpus is `docs/architecture.md`, `docs/benchmarks.md` and `docs/turbo-kv-cache.md` concatenated. Each trace is width-one decode with `MLXCEL_TRACE_BATCH_ROWS=1` and `MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1`, behind a context under the kernel's 8192-key limit, and `MLXCEL_TRACE_MLA_STATS=1` proves which path ran.

| model | shape | reference | candidate | decided disagree | top-1 disagree | ppl reference / candidate | path counter |
|---|---|---|---|---|---|---|---|
| V2-Lite | behind 1024 | decompressed | default | 0 / 209 | 4 / 512 | 14.6514 / 14.6903 | absorbed_fused=13824 |
| V2-Lite | behind 4096 | decompressed | default | 0 / 106 | 0 / 256 | 12.9935 / 12.9915 | absorbed_fused=6912 |
| Kimi-VL | behind 1024 | fused off | fused | 0 / 192 | 2 / 512 | 18.0403 / 17.9517 | absorbed_fused=13824 |
| Kimi-VL | behind 4096 | fused off | fused | 0 / 50 | 2 / 256 | 35.4533 / 35.5602 | absorbed_fused=6912 |

The V2 rows reproduce #3121's absorbed-fused rows exactly, now without setting the variable. On Kimi-VL every disagreement falls where the reference's top two were within 0.11 logits, so the fused kernel is in the rounding class of the composed path there too. The `decompressed` counter in each candidate row is the context pass, which runs multi-token and is not part of the comparison. No defect was found.

## The greedy parity test

`tests/deepseek_v2_greedy_parity.rs` compared 20 free-running greedy ids against mlx-lm, and the default failed it at step 1: the decompressed layout scores `.` 29.109 over `,` 29.094 (one f16 ULP apart), the composed absorbed path matches that, and the fused kernel scores both 29.094, so the lower id wins and every later token follows different text. The test is now teacher-forced on the reference ids and accepts a different argmax only where the reference id is within 0.0625 of the top logit, at most twice in 20 steps. It passed under the default, `MLXCEL_MLA_ABSORBED=0` and `MLXCEL_MLA_FUSED_DECODE=0` (then the composed absorbed path; after the backend condition below, that setting selects the decompressed default); a bidirectional prefill (#991) still misses by whole logits.

## Backend condition

These measurements are Metal-only, with the fused kernel. The default therefore resolves to absorbed only where that kernel runs (`mla::fused_decode_enabled()`: a Metal port and `MLXCEL_MLA_FUSED_DECODE` not off). CUDA and ROCm keep the decompressed layout by default until #3133 measures them; `MLXCEL_MLA_ABSORBED=1` still forces absorbed there. Update: #3133 measured CUDA and switched its default to absorbed ([GB10 results](mla-cuda-validation-gb10-2026-11-17.md)); ROCm is still unmeasured.

## Correctness on the other `deepseek_v2` decoder

`deepseek-vl2-small-4bit` (`deepseek_v2` decoder via `src/loading/vlm_deepseek_vl2.rs`) generates the same greedy text under the default and under `MLXCEL_MLA_ABSORBED=0`, for a text prompt (`The capital of France is Paris.`) and with `tests/fixtures/test_image.png`.

## Commands

```
M=models/deepseek-v2-lite-chat-4bit-mlx; K=models/kimi-vl-a3b-thinking-4bit
mlxcel-bench-decode -m $M -p Hello -n 128 --warmup-tokens 16 --ignore-eos --prompt-tokens 512   # also 2048, 16384; MLXCEL_MLA_ABSORBED=0 for decompressed
mlxcel-bench-decode -m $K -p Hello -n 128 --warmup-tokens 16 --ignore-eos --prompt-tokens 512   # also 2048; MLXCEL_MLA_FUSED_DECODE=0 for composed
MLXCEL_TRACE_MLA_STATS=1 MLXCEL_TRACE_BATCH_ROWS=1 MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1 \
  logit_trace $M corpus.txt 64 8 8 1024 > def-1k.tsv     # 64 4 8 4096; same for $K
python3 scripts/compare_logit_traces.py dec-1k.tsv def-1k.tsv --decided 2
```

Rows: [decode.csv](mla-absorbed-default-m5max-2026-11-16/decode.csv) and [logit_trace.csv](mla-absorbed-default-m5max-2026-11-16/logit_trace.csv).
