# Batched DFlash per-row accepts on M3 Ultra, 2026-11-06

Issue #2442 item 1 follow-up to #2353 / #2440 and #2851. The batched Qwen 3.5
DFlash round loop used to clamp every round to the shortest live row's accept
because the attention KV cache is rectangular. That left the committed accept
count far below the speculative walk, so batching bought little. This measures
the per-row accept port that replaces the clamp.

## What changed

- `DFlashBatchedGenerator` commits each row's own walk count. The target gets
  each row's pre-round logical valid end, compacts the attention KV caches back
  to per-row contiguous prefixes, and replays each GDN layer's conv and
  recurrent state over that row's accepted prefix
  (`Qwen35Model::rollback_speculative_cache_rows`).
- The verify forward rotates queries and keys at each row's logical positions
  and masks the row's stale gap (`build_divergent_row_mask`,
  `attend_per_position_rows`), the layout Gemma 4's divergent MTP verify uses.
- `MLXCEL_DFLASH_PER_ROW_ACCEPTS=0` restores the uniform clamp. A quantized,
  paged or head-trimmed KV cache selects the clamp automatically.

## Method

- Host: Apple M3 Ultra (80-core, 512 GB), macOS 26.6.2, Metal + Accelerate.
- Checkpoints: `models/qwen3.5-4b-4bit` target, `models/qwen3.5-4b-dflash`
  drafter, block size 16, temperature 0.
- Prompts: the four prompts of the batched DFlash parity harness, truncated to
  the shortest tokenization so the `[B, L]` prefill is rectangular.
- 128 tokens per row. Throughput is the round-loop wall clock from
  `DFlashBatchedRunOutput::stats` (prefill excluded), aggregate over rows.
- Both arms run in one process behind
  `throughput_dflash_qwen35_batched_clamp_vs_per_row`, two discarded warm-up
  runs, three measured runs per arm alternating clamp, per-row, clamp,
  per-row, ... under `caffeinate -i` and `scripts/with_indexers_paused.sh`.
  A separate worktree's Rust build was compiling during the run (a few of the
  80 cores); both arms alternated under that same load, and the clamp and
  B = 1 cells reproduce the quiet-box #2442 record (83.0 against 82.96-83.00,
  78.09 against 77.70-77.83), so the comparison is not load-shifted.

## Throughput at B = 4

| run | clamp tok/s | per-row tok/s |
|---:|---:|---:|
| 0 | 83.04 | 115.82 |
| 1 | 83.22 | 116.24 |
| 2 | 83.06 | 116.05 |
| median | 83.06 | 116.05 |

Per-row is **1.397x** the clamp. The clamp's three runs span 0.2%, the per-row
arm's 0.36%. For context, B = 1 measured 78.09 tok/s, so the per-row arm is
1.49x over the B = 1 round loop and the clamp is 1.06x.

## Accepts

| arm | rounds | clamped rounds | walk per row-round | committed per row-round |
|---|---:|---:|---:|---:|
| clamp | 96 | 93 | 2.938 | 0.323 |
| per-row | 67 | 0 | 1.036 | 1.036 |

The clamp numbers reproduce the #2442 re-scope measurement exactly (walk
2.938, committed 0.323, 93 of 96 rounds clamped). Per-row commits 3.2x the
clamp's accept count and finishes in 30% fewer rounds.

The walk itself falls from 2.938 to 1.036: the drafter's context is still a
shared rectangular prefix (the shortest live row's committed prefix), while the
target now advances each row by its own accept, so the drafter's context lags
the rows that accept more. Giving the drafter per-row context is a separate
change; this port was measured and decided on its own.

## Correctness

`greedy_parity_dflash_qwen35_batched_b4_matches_b1` and the
`identical_prompts_dflash_qwen35_b4_matches_b1` control pass with the port
enabled under the calibrated 1.0-logit evaluation-path margin of #2851.
