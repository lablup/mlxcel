# Cohere2 prefill on maskless causal SDPA inside the window (issue #1956)

Two-binary measurement of Cohere2 attention taking MLX's maskless causal SDPA
for a multi-token call whose keys all fit the window (global layers always,
sliding layers while the cache holds at most `sliding_window` = 4096 keys),
instead of reading an explicit `(L, K)` mask array.

- **Date:** 2026-09-23
- **Hardware:** Apple M1 Ultra 128 GB, macOS 27.0 (26A428)
- **Build:** `cargo build --release --features metal,accelerate`
- **Arms:** base = the tree of #1955 (`88173670`, same code as `main` at
  `25fdb137`) versus this change, two separate binaries
- **Model:** `models/mlx/c4ai-command-r7b-12-2024-4bit`
- **Machine load:** background indexers suspended with
  `scripts/with_indexers_paused.sh`; load average drifted from 7.0 to 2.5 during
  the run, so the comparison uses per-pair ratios

## Method

`scripts/bench_decode.sh` at its standard condition (`--ignore-eos`,
same-process warmup of 20 tokens, `--max-tokens 128`) with `--prompt-tokens`
512, 2048 and 4096. Each round runs every length in ABBA order (base, new, new,
base), three rounds. Each arm ran from its own directory whose `target/release`
links to that arm's binaries; the base arm set `BENCH_ALLOW_STALE_BINARY=1`.
4096 tokens is the largest prompt for which every layer takes the causal path;
the default prefill chunk of 2048 splits it into two calls, and the second
call's 4096 keys still fit the window.

## Results (prefill tok/s)

| Prompt | base, six runs | new, six runs | Per ABBA pair (new over base mean) |
|---|---|---|---|
| 512 | 814.86 to 831.79 | 822.81 to 838.07 | +0.75%, +0.30%, +0.90% |
| 2048 | 782.17 to 796.62 | 809.20 to 825.83 | +4.6%, +3.3%, +3.7% |
| 4096 | 745.42 to 751.68 | 772.71 to 775.99 | +3.3%, +3.7%, +3.3% |

In time: about 5 ms at 512 tokens, 90 ms at 2048 and 175 ms at 4096. Decode is
unchanged (114 to 116 tok/s at pp512 in both arms).

The absolute values moved with the load drift (the pp512 base went from 831.8
in the first round to 814.9 in the third), which is why the table reports the
ratio inside each ABBA quartet rather than a difference of medians.

## Output equivalence

Greedy generation of 100 tokens produced identical text on both binaries for
three short prompts and the 512-token synthetic prompt.

## Not measured

Prompts beyond 4096 tokens, where sliding layers keep the mask path and only
the global layers change, and CUDA.
