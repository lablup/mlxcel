# Inkling 2 byte-exact speculative verify (M3 Ultra, 2026-11-24)

Issue #3280. Before this change the block-versus-chain probe declined the bundled DSpark drafter of Inkling 2 Mini and Small at every width, so the drafter stayed off unless `MLXCEL_MTP_ALLOW_INEXACT=1` (`inkling2-mini-dspark-m3ultra-2026-11-24.md`, `inkling2-small-m3ultra-2026-11-24.md`). With the change the probe passes at widths 2, 4 and 8 on both checkpoints, and drafter-on greedy output is byte-identical to classic decode without the override. Host: Apple M3 Ultra, 512 GB. Every run is serialized under the shared GPU lock; ABAB order inside one block; load average 2 to 4.5.

## What made a verify block differ

A per-visit comparison of hidden states (an 8-row verify block against 8 single-token steps from the same cache) located three causes, all independent of quantization and of the expert path:

| Cause | Where | Fix |
|---|---|---|
| MLX runs one row through gemv and several through gemm, which sum in different orders | every dense product: attention q/k/v/r/o/gate/sink projections (bf16), the f32 router, the message controllers, the f32 LM head | `mlxcel_core::row_invariant`: a Metal matvec for up to 16 rows whose summation order depends on K and N only, used by decode steps and verify blocks alike |
| The per-query zero sink column sat after the newest key, at index N+1 in a step but N+K in a block, which moved the SDPA lane assignment even for row 0 | attention | the sink column goes before the keys |
| The message read `query @ bank^T` is a batched matmul whose batch extent is the block length | looped core (Mini) | split per position inside a verify block |
| Past a sliding layer's 1024-token window, row r of a block also saw the r oldest keys (masked out) that decode step r had dropped, which shifted the SDPA lanes again | sliding attention, context > 1024 | inside a verify block a sliding visit attends row by row from each row's window start |
| The DSpark prefill ran the whole prompt in one forward while classic decode prefills in 2048-token chunks, so a longer prompt left different caches | prefill, prompt > 2048 | the DSpark prefill uses the classic chunk size (`MLXCEL_PREFILL_CHUNK`) |

After the first three, every captured hidden state and every logit of an 8-row block equals the chain's bit for bit after a 32-token prompt (Mini, Small and Mini 4-bit, widths 2, 4, 8). The probe draws short prompts, so it did not see the fourth: on Mini, rows 1 to 7 differed in about 200050 of 200058 logits after a 1100- or 3000-token prompt. With the fourth fix the block is exact after 1100, 3000 and 8000 tokens (Mini) and 1100 tokens (Small, Mini 4-bit).

## Decode cost of the kernel

The kernel now also serves the one-row decode step, so it must not be slower than MLX gemv. 256 greedy tokens, `Explain why the sky is blue.`, kernel on (`MLXCEL_INKLING2_KERNELS` default) against the MLX path, two ABAB pairs:

| Checkpoint | Kernel on tok/s | MLX matmul tok/s | Change |
|---|---|---|---|
| inkling-2-mini | 37.11, 37.13 | 36.62, 36.61 | +1.4% |
| inkling-2-mini-4bit | 44.93, 44.72 | 44.37, 44.47 | +0.9% |

One SIMD group per output column, with K split across eight SIMD groups when N < 512 (the router and the controller down projection), measured fastest of the tilings tried; the first version, without the split, cost 2.5% to 3.4% on the narrow sites.

An 8-row Mini verify forward takes 51.2 ms with the kernel (median of 30), against 52.1 ms for the old inexact matmul path and 80.6 ms for one gemv per row.

## DSpark throughput

256 greedy tokens, block size 8, `--show-reasoning`, no override. Prompt 1: `Explain why the sky is blue.` Prompt 2: `Write a Python function that returns the n-th Fibonacci number, with a short docstring.` Drafter-on text equals classic text byte for byte in every run.

| Checkpoint | Prompt | Classic tok/s | Drafter tok/s | Speedup | Rounds | Acceptance | Emitted per verify |
|---|---|---|---|---|---|---|---|
| Mini | 1 | 37.12, 37.11 | 37.32, 39.73 | 1.04x | 117 | 0.172 | 2.18 |
| Mini | 2 | 36.90, 36.88 | 56.45, 61.93 | 1.60x | 76 | 0.346 | 3.36 |
| Small | 1 | 19.64, 19.38 | 17.70, 17.47 | 0.90x | 110 | 0.192 | 2.32 |
| Small | 2 | 19.22, 19.26 | 29.64, 29.47 | 1.54x | 62 | 0.445 | 4.11 |

On Small a verify round costs about 120 ms against a 51 ms decode step, so the 2.3 tokens a prose round emits do not pay for it; the drafter loses 10% there and wins 54% on code. The expert path, not the dense products, dominates Small's verify cost (it took 107 ms with the inexact path too).

A prompt of 2006 tokens (bytes 4000 to 12000 of `docs/supported-models.md` plus `Summarize the text above in three sentences.`) exercises the window and the chunked prefill: before the prefill fix the drafter-on text left classic decode in the second sentence of the reasoning; with it the two are byte-identical (classic 24.99 tok/s, drafter 24.82 and 25.50 tok/s, acceptance 0.225), so the drafter breaks even on that prompt.

## Parity with mlx-vlm

Putting the sink column first changes classic decode at near ties. A text trace of Mini 4-bit against mlx-vlm 466069f5 (4 x 256 positions of the #3378 corpus, `scripts/compare_logit_traces.py --decided 2`): before, 33 / 1024 top-1 differences and 0 / 372 at decided positions, perplexity -0.32% against the reference; after, 31 / 1024, 0 / 372, -0.38%. Every difference is still at an undecided position.
