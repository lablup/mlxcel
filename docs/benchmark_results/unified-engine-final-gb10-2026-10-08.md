# Unified engine final measurement: pre-epic CxxGenerator vs the engine (GB10, 2026-10-08)

Epic #2166, end-of-run measurement for ADR 0007's regression threshold and the speed questions the phases left open. The pre-epic baseline is [unified-engine-baseline-gb10-2026-10-07.md](unified-engine-baseline-gb10-2026-10-07.md).

## Host and method

NVIDIA GB10 (sm_121), driver 580.178.04, host `spark-102`, release profile, `--features cuda`. Two builds:

- **base**: `4c44e317` (PR #2194, Phase 0 merge; decode code identical to pre-epic `main`), where `--path cli` is `CxxGenerator`.
- **new**: `30ceea22` (PR #2225, the last phase), where `--path cli` is `mlxcel_core::engine::DirectEngine`, the client `mlxcel generate` now runs.

Run 2026-10-07 23:47 to 2026-10-08 01:14 UTC under the host `gpu-lock`, with no other build, test or GPU job. Single-stream numbers come from `scripts/engine_bench_rounds.py` driving `mlxcel-bench-engine` from both builds through `pick.sh` (an arm flag picks the build): 5 interleaved rounds, a fresh process per arm, rotating order, the #1820 quiet-host gate before every arm, 128 generated tokens after 16 warmup tokens, and a null arm (the first arm, `base-cli`, run twice per round). Serving numbers start each build's `mlxcel serve` and drive it with `scripts/bench_serving_concurrency.py` and `scripts/bench_mixed_step_admission.py`, three alternating rounds. Raw records, logs and the drivers are in `data/unified-engine-final-gb10-2026-10-08/`.

## Threshold: single-stream decode against the pre-epic CxxGenerator

Paired per-round decode tok/s delta against `base-cli` (median, range over 5 rounds). The threshold is -1.0 percent (ADR 0007).

| Model | Prompt | `mlxcel generate` (new-cli) | server engine, dense (new-srv-dense) | Null (base-cli repeat) |
|---|---|---|---|---|
| Qwen3-1.7B 4-bit | 256 | +0.82 % (+0.04..+1.35) | -1.44 % (-2.03..-0.55) | -0.12 % (-0.92..+0.40) |
| Qwen3-1.7B 4-bit | 8192 | +0.31 % (+0.18..+0.66) | -1.14 % (-1.48..-0.58) | -0.03 % (-0.32..+0.36) |
| Llama-3.2-1B 4-bit | 256 | +0.51 % (+0.19..+1.37) | -1.24 % (-2.11..-0.77) | +0.12 % (-0.03..+0.27) |
| Llama-3.2-1B 4-bit | 8192 | +0.19 % (+0.03..+0.82) | -2.05 % (-2.25..-1.50) | +0.06 % (-0.36..+0.42) |

- **`mlxcel generate` passes in every cell.** The engine client decodes at least as fast as `CxxGenerator`; three of the four cells clear the null range upward.
- **The server engine at B=1 with dense storage misses the threshold in three of four cells.** That is the path `mlxcel run` and the chat REPL take since #2173 (one slot, dense). Phase 0 measured the same gap before the epic (server dense within 1.0 to 1.5 percent of the CLI), so it is the scheduler's per-tick overhead rather than something the epic added. But `run` used `CxxGenerator` before #2173, so for `run` users this is a 1 to 2 percent decode regression. Follow-up needed.
- **TTFT at 256 tokens on `generate`**: +4.4 % on Qwen3 (null range -3.9..+10.9, unresolved) and +9.0 % on Llama (+4.2..+12.7 against a null of -4.4..+8.3), about 1 ms. At 8192 tokens TTFT is unchanged (-0.4 %).

Medians (tok/s, TTFT ms):

| Model | Prompt | base-cli | new-cli | new-srv-dense | base-srv | new-srv |
|---|---|---|---|---|---|---|
| Qwen3-1.7B 4-bit | 256 | 180.81 / 18.87 | 182.30 / 19.69 | 179.34 / 18.88 | 143.61 / 20.14 | 142.43 / 20.12 |
| Qwen3-1.7B 4-bit | 8192 | 107.79 / 753.27 | 108.41 / 745.70 | 106.67 / 744.62 | 101.24 / 814.62 | 101.16 / 657.40 |
| Llama-3.2-1B 4-bit | 256 | 269.78 / 13.65 | 271.26 / 14.78 | 265.75 / 13.59 | 243.39 / 13.85 | 243.45 / 13.44 |
| Llama-3.2-1B 4-bit | 8192 | 207.82 / 403.95 | 208.21 / 401.59 | 203.57 / 403.54 | 187.90 / 478.55 | 187.52 / 420.04 |

The server's default configuration (`base-srv`, `new-srv`: `--parallel 4`, `auto` storage, which is paged) decodes a lone sequence at the same rate before and after the epic, 6 to 21 percent below dense as Phase 0 measured. Its TTFT at 8192 tokens dropped 19 percent on Qwen3 and 12 percent on Llama from the 2048-token prefill chunk. A lone sequence on a multi-slot server is still allocated paged, because storage is chosen when the sequence is admitted (ADR 0008).

## Batched serving throughput, base vs new

`mlxcel serve --parallel 8`, 512-token prompts, 128 generated tokens, median of 3 rounds, aggregate tok/s (all completion tokens over wall time).

| Workload | Concurrency | base | new | Change |
|---|---|---|---|---|
| Qwen3-1.7B 4-bit, defaults (paged) | 1 | 113.5 | 113.1 | -0.4 % |
| | 4 | 128.6 | 131.5 | +2.3 % |
| | 8 | 100.9 | 101.6 | +0.7 % |
| Llama-3.2-1B 4-bit, `--kv-cache-mode turbo4` (paged) | 1 | 73.8 | 72.5 | -1.8 % |
| | 4 | 94.3 | 109.6 | +16.2 % |
| | 8 | 82.3 | 97.5 | +18.5 % |

The Qwen3 rows are within the round-to-round spread (up to 8 percent). The Turbo rows improve at concurrency 4 and 8, where #2210 moved batched Turbo rows onto the dequant-first variants single-sequence decode already used; every round of `new` is above every round of `base` there. Aggregate throughput at concurrency 8 is below concurrency 4 on both builds, a pre-existing GB10 batched-decode property.

## Prefill chunk under live decode: 2048 vs 512

The question #2205 left for this run. `mlxcel serve --parallel 8 --ignore-eos` on Llama-3.2-1B 4-bit, four streams decoding, then one 8192-token request admitted (`bench_mixed_step_admission.py`). Qwen3 was not usable here because its streams emit reasoning deltas first, which the harness does not count.

| Chunk | Stream ITL p95, quiet | Stream ITL p95 during admission | Stream ITL mean during admission | Admitted request TTFT |
|---|---|---|---|---|
| 2048 (default) | 18.6..19.5 ms | 127.9..128.9 ms | 18.7..19.3 ms | 1903..2060 ms |
| 512 | 17.2..19.4 ms | 55.8..56.4 ms | 13.4..13.9 ms | 3768..4067 ms |

The trade is consistent across rounds: 2048 halves the admitted request's TTFT and stalls live streams 2.3 times longer at p95 while it prefills. Neither setting is faster in both measures, so this is a policy choice, not a speed win; an adaptive chunk (2048 when no other sequence is decoding, 512 when one is) would take the better side of each.

## Gemma 3 storage under concurrency

`mlxcel serve --parallel 8` on Gemma-3-1B 4-bit, new build, `--decode-storage-backend paged` (what `auto` picks today) vs `dense`. Since #2217 no Gemma 3 attention path reads the pool.

| Concurrency | paged | dense |
|---|---|---|
| 1 | 146.3 | 148.1 |
| 4 | 351.9 | 357.8 |
| 8 | 187.9 | 190.0 |

Dense is 1 to 2 percent higher in every cell, but the round ranges overlap (for example 342..360 vs 341..359 at concurrency 4), so the result is unresolved and the default stays as it is.

## Reproduce

`data/unified-engine-final-gb10-2026-10-08/run.sh` (single-stream, batched, Gemma 3) and `admit.sh` (admission), with `pick.sh` dispatching to the two builds. On a relocated build tree, set `MLXCEL_CCCL_DIR` and `MLXCEL_CUTLASS_DIR` to the build's `out/build/include/cccl` and `out/build/include`, since the JIT header fallback path is compiled into the binary.
