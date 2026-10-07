# Unified engine baseline: CLI vs server B=1 decode, prefill chunk and KV storage (GB10, 2026-10-07)

Issue #2167, Phase 0 of epic #2166. This record holds the pre-epic single-stream baseline that every later phase is measured against, the two speed decisions ADR 0007 takes by measurement (prefill chunk size, single-sequence KV storage), the regression threshold, and the cross-path parity baseline.

## Host and method

NVIDIA GB10 (sm_121), driver 580.178.04, host `spark-102`, MLX pin `81ba1c6a0e50a9268b931579c2d4f1158b9aab5a`, release profile, `--features cuda`, mlxcel commit `e73b68a5` (PR #2194 head; decode code identical to `main` at `bbd05099`). Checkpoints `models/mlx/qwen3-1.7b-4bit` and `models/mlx/llama-3.2-1b-instruct-4bit`. Run 2026-10-07 02:17 to 04:29 UTC under the host `gpu-lock`, with no other build, test or GPU job, and the #1820 quiet-host gate before every arm (`--hostgate`).

Every number comes from `scripts/engine_bench_rounds.py` driving `target/release/mlxcel-bench-engine`: 5 interleaved rounds, each arm a fresh process, the run order rotated one position per round, 128 generated tokens after 16 warmup tokens, and a null arm (the first arm, or every arm with `--null-every-arm`, run twice per round) whose paired per-round delta is the noise floor. Decode tok/s is generated tokens over post-first-token time on both paths; TTFT is time to the first streamed token. The server path is the real `mlxcel-server` stack minus HTTP (in-process `ModelProvider`, `--parallel 4`, pre-tokenized requests). Raw records, logs and the environment file are in `data/unified-engine-baseline-gb10-2026-10-07/`.

## Baseline: CLI (`CxxGenerator`) vs server (today's defaults)

Medians; ratio is server over CLI. The server runs its defaults: paged storage at B=1 (`auto` resolves to paged with `--parallel 4`) and a 512-token prefill chunk.

| Model | Prompt | CLI decode tok/s | Server decode tok/s | Server/CLI | CLI TTFT ms | Server TTFT ms |
|---|---|---|---|---|---|---|
| Qwen3-1.7B 4-bit | 256 | 191.16 | 147.78 | 0.77 | 19.08 | 20.17 |
| Qwen3-1.7B 4-bit | 8192 | 115.21 | 106.06 | 0.92 | 741.95 | 800.26 |
| Llama-3.2-1B 4-bit | 256 | 286.15 | 257.38 | 0.90 | 12.96 | 13.78 |
| Llama-3.2-1B 4-bit | 8192 | 220.75 | 199.06 | 0.90 | 395.26 | 474.80 |

The server's single-stream decode is 8 to 23 percent slower than the CLI's today. The storage A/B below attributes almost all of it to paged storage at B=1: with dense storage the server path decodes at 189.32 and 113.81 (Qwen3) and 283.16 and 217.44 (Llama) tok/s, within 1.5 percent of the CLI. The TTFT gap at 8192 tokens is the chunk size: the server's 512-token chunk against the CLI's 2048.

Null-arm noise (per-round paired decode delta, range): CLI Qwen3 256 −0.92..+0.82 %, 8192 −0.25..+0.28 %; CLI Llama 256 −0.64..+0.25 %, 8192 −0.55..−0.02 %. Server paged at 256 tokens is noisier (Qwen3 −4.07..+8.52 %).

## Decision: prefill chunk, 2048

TTFT at the 8192-token prompt, 2048 vs 512, on both paths. Decode tok/s did not move (all paired deltas within ±1 %).

| Model | Path | TTFT 2048 ms | TTFT 512 ms | 512 vs 2048 (paired, range) | Null (range) |
|---|---|---|---|---|---|
| Qwen3-1.7B 4-bit | CLI | 739.45 | 800.56 | +8.43 % (+7.56..+8.57) | −1.49..+0.71 % |
| Qwen3-1.7B 4-bit | server | 651.10 | 798.75 | +22.86 % (+20.80..+23.76) | −1.90..+1.12 % |
| Llama-3.2-1B 4-bit | CLI | 397.36 | 514.10 | +29.40 % (+28.24..+30.67) | −0.94..+1.40 % |
| Llama-3.2-1B 4-bit | server | 411.32 | 471.05 | +14.61 % (+13.30..+15.81) | −0.62..+0.87 % |

2048 has the lower TTFT on both models and both paths, and every delta clears its null range by a wide margin, so by ADR 0007's rule the unified engine adopts **2048**. The 512 default's rationale (a bounded prefill transient that interleaves finely with live decode streams, ADR 0005 and #1011) is a batched-serving latency property this single-stream measurement does not cover; Phase 3 (#2170) owns the chunk policy and may keep a smaller chunk for mixed prefill/decode ticks if a batched measurement shows it is needed, but the single-stream default is 2048.

## Decision: single-sequence KV storage, dense

B=1 decode tok/s on the server path, `--decode-storage dense` vs `paged`. Every paged row recorded paged kernel launches (Qwen3 3612, Llama 2064 per request), so the paged arm really ran the pooled paged kernel.

| Model | Prompt | Dense decode tok/s | Paged decode tok/s | Paged vs dense (paired, range) | Null (range) |
|---|---|---|---|---|---|
| Qwen3-1.7B 4-bit | 256 | 189.32 | 145.77 | −23.03 % (−25.56..−21.21) | −0.48..+0.20 % |
| Qwen3-1.7B 4-bit | 8192 | 113.81 | 106.45 | −6.47 % (−6.93..−6.05) | −0.29..−0.11 % |
| Llama-3.2-1B 4-bit | 256 | 283.16 | 258.50 | −8.22 % (−10.64..−7.16) | −0.85..+0.43 % |
| Llama-3.2-1B 4-bit | 8192 | 217.44 | 199.73 | −8.14 % (−8.61..−7.31) | −0.46..+1.09 % |

Dense is faster in every cell and every delta clears its null range, so the unified engine's single-sequence storage is **dense**. Paged storage remains the batched-serving backend (its value is in the shared pool across sequences, which this B=1 measurement does not exercise). One TTFT observation for Phase 4a (#2171): on Llama at 8192 tokens the paged arm's TTFT was 8.4 percent lower than dense (474.13 vs 519.35 ms; Qwen3 showed no difference), so the prefill side of the storage choice is not uniform across families.

## Regression threshold: 1.0 percent

ADR 0007's rule: the widest null-arm per-round decode delta across the four (model, context) cells of the baseline arm (CLI), rounded up to the next 0.5 percent. The widest is 0.92 percent (Qwen3, 256 tokens), so a later phase passes when its median paired decode delta against the pre-epic `CxxGenerator` baseline is no worse than **−1.0 percent** in every cell.

## Cross-path parity baseline

`mlxcel-engine-parity` with `MLXCEL_SDPA_DETERMINISTIC=1`, default prompt, 400 tokens (`parity-<model>.{txt,json}`). Columns: (a) CLI, (b) server B=1 dense, (c) server B=1 paged, and on (b) a prompt-cache miss vs hit.

| Model | Case | a vs b | a vs c | b vs c | b vs b+cache miss | cache miss vs hit |
|---|---|---|---|---|---|---|
| Qwen3-1.7B 4-bit | greedy | identical | identical | identical | identical | diverge at token 0 (4792 vs 785) |
| Qwen3-1.7B 4-bit | seeded, penalties, DRY | identical | identical | identical | identical | diverge at token 11 |
| Llama-3.2-1B 4-bit | greedy | identical | identical | identical | identical | identical |
| Llama-3.2-1B 4-bit | seeded, penalties, DRY | identical | identical | identical | identical | diverge at token 25 |

Under the determinism switch the CLI, server-dense and server-paged paths already agree for these models and cases; the remaining divergence is a prompt-cache hit (adopting 45 of 52 Qwen3 prompt tokens, 71 of 75 Llama tokens, and forwarding only the suffix), which Phase 3 (#2170) targets. On Qwen3 the greedy hit changes the very first token.

## Reproduce

Commands are in ADR 0007, "Measurement commands"; the exact driver used here is `data/unified-engine-baseline-gb10-2026-10-07/run.sh`: for each model, a baseline run (`--null-every-arm`, arms `cli`/`server`), a CLI and a server chunk run at 8192 tokens (arms `c2048`/`c512`), and a storage run (arms `dense`/`paged`), all with `--hostgate --rounds 5`.
