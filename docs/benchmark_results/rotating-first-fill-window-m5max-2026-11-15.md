# Rotating-cache first fill keeps only the window (issue #3062), M5 Max, 2026-11-15

`RotatingKVCache::update_concat` stored every position of a multi-token first fill and cut the buffer back to `max_size` only at the first decode step. In a single-pass prefill every sliding-window layer therefore held its whole prompt K/V until decode began. The fix stores the newest `max_size + 32` positions of a first fill longer than that as a fresh buffer, scheduled with `async_eval` through `mlxcel_core::utils::materialized_axis_tail` (the time-axis sibling of #3061's `materialized_state_tail`). Attention still receives every key; the 32 extra positions cover the pad-tail rewind of a tile-aligned prefill.

Host: Apple M5 Max, 128 GB, macOS 27. MLX `9f806b3606eaa9f7c807238f6bd465185cd94d6d`. Base is `main` at `0ca353da`; the fix is branch `fix/issue-3062-rotating-first-fill-window` on top of it. Both are release builds with `metal,accelerate`, copied aside before measuring.

Run setup:

- Prompt from `scripts/bench/metal_sdpa_chunk/make_prompt.py 28000` (28,710 tokens for Gemma 4, 28,706 for Gemma 3).
- `mlxcel generate`, greedy, 48 tokens, `--show-reasoning`. Single pass is `MLXCEL_PREFILL_CHUNK=0`; the other arm is the default 2048-token chunks.
- Peak is the "peak memory footprint" from `/usr/bin/time -l`. Prefill is from `--profile`.
- 60 s pause before every run. Each cell ran once main first and once fix first.

The generated text is byte-identical between main and the fix in every cell (`output_md5` in [`summary.csv`](rotating-first-fill-window-m5max-2026-11-15/summary.csv)). The change moves no arithmetic: attention reads the same keys, and later steps read the same window positions.

## Results

| model | prefill | peak, main | peak, fix | prefill, main / fix |
|---|---|---|---|---|
| gemma-4-12b-it-4bit | single pass | 27.38 GB | 18.67 GB | 20.76 / 20.67 s (main first), 20.41 / 20.55 s (fix first) |
| gemma-4-12b-it-4bit | 2048 chunks | 14.17 GB | 14.17 GB | 21.47 / 22.83 s (fix first) |
| gemma-3-4b-it-4bit | single pass | 15.14 GB | 7.03 GB | 4.01 / 4.01 s (main first), 4.19 / 4.14 s (fix first) |
| gemma-3-4b-it-4bit | 2048 chunks | 5.25 GB | 5.25 GB | 4.26 / 4.20 s (fix first) |

Gemma 4 12B drops 8.71 GB (8.1 GiB) of the 8.75 GiB its 40 sliding layers held over the prompt, and lands within 0.1 GB of the 18.58 GB the experiment in the issue measured. Gemma 3 4B drops 8.11 GB.

The prefill times above come from a quiet stretch of the host. The single-pass cells were repeated for that reason: during the first sweep other jobs on the machine slowed runs by up to 90% in no consistent direction (for example 32.7 to 39.1 s for the same Gemma 4 binary), and those numbers are kept in the CSV but are not a comparison. In the quiet repeat every single-pass pair is within 1.2%. The 2048-chunk cells were not repeated; their only undisturbed pair (fix first) reads 6% apart on Gemma 4 and 1.5% on Gemma 3, so treat the chunked timing as unmeasured noise rather than a regression or a gain.

The 2048-chunk peak does not move. A 2048-token first chunk is itself longer than the 1024-token window, so the fix trims it too, but the chunked peak is set by one chunk's working set rather than by the stored windows.
