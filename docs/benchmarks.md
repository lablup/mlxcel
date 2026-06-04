# Benchmarks

This page documents how benchmark claims should be recorded for `mlxcel`. It is
intentionally conservative: do not publish aggregate speedup numbers without the
raw per-model rows and the exact software/hardware versions used to produce them.

## What to record

For every benchmark run, include:

- hardware model and memory size;
- operating system version;
- `mlxcel` version or commit;
- pinned MLX commit/version;
- comparison runtime version (`mlx-lm`, `mlx-vlm`, or another baseline);
- model checkpoint name and quantization format;
- prompt length, requested decode length, batch size, and warmup policy;
- cache mode and server/generation flags;
- raw per-model prefill and decode throughput where available.

Averages are useful only after the raw rows are available. Avoid statements such
as "faster than X" unless the comparable model set and exclusions are explicit.

## Current result snapshot

Keep public result summaries in a single place so aggregate numbers do not drift
between documents. The current Apple Silicon benchmark report is:

- [Benchmark Report - 2026-05-19](benchmark_results/benchmark-report.md)

Use that report and its linked raw per-hardware tables for release notes,
README updates, or capacity planning. This page should stay focused on
methodology, required metadata, and caveats.

## Suggested benchmark commands

The repository contains benchmark helper scripts under `scripts/`. The exact
arguments may evolve, so inspect each script before publishing results.

```bash
# Single-model decode benchmark shape.
./scripts/bench_decode.sh -m models/<checkpoint> --runs 3

# Multi-model suite shape.
./scripts/bench_all_models.sh --hardware <name> --cooldown 45 --big-cooldown 60
```

## Speculative decoding (MTP)

MTP speculative decoding pairs a decode target with a small assistant drafter
that proposes a block of tokens, which the target then verifies in a single
forward pass. At `temperature 0` the accelerated output is byte-identical to
classic decode, so the only metric that moves is decode throughput; confirm
correctness by diffing the two completions.

For each pairing, record both the baseline (no drafter) and the MTP run:

- decode tok/s for each, and the speedup ratio (MTP divided by baseline);
- mean acceptance length (accepted draft tokens per verify), read from the
  `MTP round-loop diagnostics` log line;
- the block size (`--draft-block-size`), and whether the singleton burst
  engaged or was declined.

Measure with the `speculative_bench` harness or the server:

```bash
# In-process harness: baseline vs MTP on the same target.
./target/release/speculative_bench --target <target_dir> --kind none --max-tokens 256
./target/release/speculative_bench --target <target_dir> --draft <drafter_dir> --kind mtp --max-tokens 256

# Server (production path): time a fixed temperature-0 completion with and
# without the drafter. The server logs decode tok/s and acceptance per request.
mlxcel serve -m <target> --draft-model <drafter> --draft-kind mtp
```

### Gemma 4 Unified (12B) + 4-bit assistant

Measured on Apple M5 Max (128 GB) with `mlx-community/gemma-4-12b-it-4bit` as the
target and `mlx-community/gemma-4-12B-it-assistant-4bit` as the drafter, block
size 4, `temperature 0`, 200 decode tokens:

| Path                        | decode tok/s | speedup |
| --------------------------- | -----------: | ------: |
| classic decode (no drafter) |         ~39  |  1.00x  |
| MTP                         |         ~74  | ~1.87x  |

The accelerated output is byte-identical to classic decode. B=1 (single-request)
MTP runs by default for every MTP target; the Gemma 4 Unified target cannot batch
at all, so B=1 is also its only decode path. The batch-capable 31B + bf16
assistant measures ~1.2 to 1.4x on the same host. Set `MLXCEL_ENABLE_MTP_B1=0` to
opt out on hardware where the B=1 verify forward does not pay for itself.

### Gemma 4 31B + bf16 assistant

The 31B text target is batch-capable, and its MTP speedup comes from batched
(B>1) verify windows rather than the singleton path. The scheduler declines B=1
MTP there because the bf16 assistant's single-stream acceptance is too low to
offset the extra drafter forward per token. This pairing is wired into
`speculative_bench` (`REACHABLE_PAIRINGS`) and runs once the
`gemma-4-31b-it-4bit` and `gemma-4-31B-it-assistant-bf16` checkpoints are present
in the model store.

## Recommended output layout

Add benchmark artifacts under a dedicated directory before publishing a release,
for example:

```text
benchmarks/
  2026-05-08_m1-ultra_text.csv
  2026-05-08_m1-ultra_vlm.csv
  README.md
```

Each CSV should be machine-readable and accompanied by a short Markdown note
that describes methodology, exclusions, and known failures.

## Caveats

- **Thermals matter.** Apple Silicon decode throughput changes with sustained
  load; record cooldown and run order.
- **MLX pin matters.** Kernel selection can change when the pinned MLX commit
  changes.
- **VLM comparisons are separate from text comparisons.** Vision preprocessing,
  image resolution, and prompt construction differ by family.
- **CUDA numbers are not interchangeable across GPUs.** Publish the SM target
  and driver/toolkit versions with the result.
