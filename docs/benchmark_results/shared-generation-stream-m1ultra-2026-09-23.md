# One generation stream shared across generators (issue #1952)

Two-binary measurement of giving every `CxxGenerator` and `SpeculativeGenerator`
in a process the same thread-local stream handle instead of a new one each.

- **Date:** 2026-09-23
- **Hardware:** Apple M1 Ultra 128 GB, macOS 27.0 (26A428)
- **Build:** `cargo build --release --features metal,accelerate`
- **Arms:** base = the cohere2 optimization branch at `f86ebbc1` (PRs #1947,
  #1948 and #1951 applied); new = the same tree plus this change. Two separate
  binaries.
- **Model:** `models/mlx/c4ai-command-r7b-12-2024-4bit`
- **Machine load:** background indexers suspended with
  `scripts/with_indexers_paused.sh`; load average 3.2 at the start, 1.4 at the
  end

## Method

`scripts/bench_decode.sh` at its standard condition (512-token synthetic prompt,
`--ignore-eos`, same-process warmup of 20 tokens), with `--max-tokens` 64, 128
and 256. Each round runs every length in ABBA order (base, new, new, base), for
three rounds, so each cell has six runs per arm. A `--warmup-tokens 0` run of
tg128 per arm per round is the control: with no warmup the process builds one
generator, so both arms use one stream and must measure the same.

The harness always runs `./target/release`, so each arm ran from its own
directory whose `target/release` links to that arm's binaries; the base arm set
`BENCH_ALLOW_STALE_BINARY=1` because its binary predates the tree on purpose.

Attribution: there is no per-run counter for the stream a generator resolved.
The unit test `streams::tests::shared_generation_stream_resolves_to_one_stream_per_thread`
proves the mechanism (two shared handles on one thread resolve to one stream
index, a fresh handle to another), and the no-warmup control shows the arms
agree exactly where the change has nothing to change.

## Results (tok/s, median of six runs, range in parentheses)

| Condition | base decode | new decode | Change | base prefill | new prefill |
|---|---|---|---|---|---|
| tg64, warmup 20 | 107.96 (106.74 to 108.31) | 113.34 (111.77 to 114.11) | +5.0% | 702 | 727 |
| tg128, warmup 20 | 112.00 (111.90 to 112.23) | 114.42 (113.87 to 116.43) | +2.2% | 703 | 727 |
| tg256, warmup 20 | 113.98 (113.84 to 114.11) | 115.19 (114.97 to 116.25) | +1.1% | 703 | 727 |
| tg128, no warmup (3 runs) | 113.47, 113.43, 113.47 | 113.59, 112.37, 113.64 | none | about 620 | about 620 |

The decode gain shrinks with generation length because the harness divides the
generated count by the decode span, and the removed cost sits at the start of
that span: base decode milliseconds fit about 8.4 ms per token plus a 42 ms
intercept. The fresh stream also cost prefill about 25 ms at 512 tokens (702 to
727 tok/s), which is why the intercept was larger after a warmup (about 44 ms)
than in a cold process (about 12 ms).

## What this does and does not change

- User-facing paths gain little. `mlxcel generate` builds one session per run,
  `mlxcel chat` reuses one session across turns, and the server
  `BatchScheduler` holds its own stream, so each already ran on one stream.
- The benchmark harness does gain. `scripts/bench_mlxlm.py` also warms up and
  then calls `generate` again, but mlx-lm shares one module-level
  `generation_stream`, so until this change only the mlxcel column paid for a
  fresh stream: about 2% of decode and 3.5% of prefill at pp512/tg128.
- Each extra generator no longer leaves a command queue behind that MLX never
  frees.

## Output equivalence

Greedy generation of 120 tokens for two prompts produces identical text from
both binaries (the only differing lines are load time and the throughput
readout).

## Not measured

Speculative decoding was not benchmarked; `SpeculativeGenerator` takes the same
handle and the same reasoning applies. CUDA was not measured; the handle is
created only when the default device is the GPU, as before.
