# Qwen 3.5 long-prompt prefill peak (issue #3045), M3 Ultra, 2026-11-15

Issue #3045: a 28K-token prefill of the Qwen 3.5 family peaked far above weights plus KV on Metal, and the server peaked at the single-pass level although its default chunk is 2048 tokens. Two causes, two fixes:

- `Qwen35Model` (and `Qwen3NextModel`, and the `Qwen35VLModel` wrapper the published checkpoints load as) did not override `forward_last_logits*`, so every prefill piece projected all of its positions through the LM head and then sliced the last row. At 28K positions and the 248320-entry vocabulary that is a 12.95 GiB buffer. The overrides now slice the last hidden row before the final norm and the LM head.
- The server's prefill plan kept each prompt-cache snapshot segment (the history boundary, a chat prompt's rendering without its generation prompt, sits a few tokens before its end) as one forward whatever the chunk size, so a single-turn chat prompt prefilled almost whole. The segment is now cut into unpadded pieces of at most the chunk size that end on the boundary; the snapshot and the pieces after the boundary are unchanged.

Host: Apple M3 Ultra, 80-core GPU, 512 GB, pre-M5. Base is `main` at `ab45ac59` (#3049 merged), new is branch `fix/issue-3045-prefill-peak`, both release builds with `metal,accelerate`. Greedy, 32 tokens, reasoning included in the comparison. Prompt from `scripts/bench/metal_sdpa_chunk/make_prompt.py 28000`. All runs in one GPU lock hold, arms interleaved A/B then B/A (`scripts/bench/prefill_peak/ab_prefill.sh`); other units' CPU builds were running, load average recorded per row in `qwen35-prefill-peak-m3ultra-2026-11-15/ab-summary.csv`.

## Per-stage attribution

`prefill_stage_peak_memory_probe` (ignored test in `src/models/qwen3_5_prefill_memory_tests.rs`) runs a 28000-token single pass stage by stage, evaluating after each, and reads MLX's peak above the active bytes before the stage (`stage-probe.csv`):

| Stage | qwen3.5-4b-4bit peak (GiB) | qwen3.8-27b-4bit peak (GiB) |
|---|---|---|
| embedding | 0.17 | 0.34 |
| causal mask (`create_causal_mask`, kept for the whole pass) | 3.65 | 3.65 |
| one linear-attention layer, max | 3.53 | 5.30 |
| one full-attention layer, max | 6.18 | 6.52 |
| final norm | 0.13 | 0.27 |
| **lm_head over every position** | **12.95** | **12.95** |
| sampling (argmax of the last row) | 0.00 | 0.00 |
| lm_head over the last position only | 0.00 | 0.00 |

The full-sequence logits are the largest single buffer of the prefill. The same probe through the served entry point, evaluated lazily as one graph, peaks 25.80 GiB above the weights with the full logits and 20.27 GiB with the last-row projection on the 4B (43.58 and 37.02 GiB on the 27B). What remains in a single pass is the causal mask and the in-flight attention and gated-delta intermediates of one lazy graph over 28K positions; a chunked prefill bounds those, and the mask is #3051.

## Before and after

| Path | qwen3.5-4b-4bit peak | prefill / TTFT | qwen3.8-27b-4bit peak | prefill / TTFT |
|---|---|---|---|---|
| CLI, single pass (`MLXCEL_PREFILL_CHUNK=0`) | 44.39 to **30.13 GB** | 18.91 to 17.11 s (-9.5%) | 78.47 to **64.20 GB** | 89.41 to 86.25 s (-3.5%) |
| CLI, 2048-token chunks | 11.00 to **8.97 GB** | 15.51 to 13.89 s (-10.4%) | 27.83 to **25.80 GB** | 77.94 to 74.88 s (-3.9%) |
| server, `--parallel 1`, defaults | 44.57 to **9.10 GB** | 18.95 to 13.92 s TTFT (-26.5%) | 78.64 to **25.93 GB** | 89.23 to 74.87 s TTFT (-16.1%) |

Timings are means of two interleaved repetitions per arm; every pair agreed within 0.5 s (4B) and 0.2 s (27B), except the 4B single-pass base pair (19.13 and 18.69 s). The server peak is now 1.4% (4B) and 0.5% (27B) above the 2048-chunk CLI peak, against 4.1x and 2.8x before. Peaks repeat within 4 MB across repetitions.

Every cell's generated text is byte-identical between base and new, in both repetitions, CLI and server.

## `MLXCEL_MEMORY_LIMIT`

The limit stays a soft MLX allocator limit and is now documented as one. Past it, MLX's evaluation waits for in-flight command buffers and the allocator releases cached buffers before a new allocation, but no allocation is refused (`mlx/transforms.cpp` eval loop, `MetalAllocator::malloc`), which is why a 16 GB limit still let the 28K prefill reach 45.8 GB. A hard cap would need mlxcel to refuse allocations MLX has no hook for, and would fail the request without lowering what it needs; the prefill chunk is what bounds the peak.

## Rerun

```
BASE_BIN=<main build>/mlxcel NEW_BIN=target/release/mlxcel MODELS_DIR=models \
  scripts/bench/prefill_peak/ab_prefill.sh <out_dir>
MLXCEL_PREFILL_PROBE_MODEL=models/qwen3.5-4b-4bit MLXCEL_PREFILL_PROBE_LEN=28000 \
  cargo test --profile test-fast --features metal,accelerate -p mlxcel --lib -- --ignored --nocapture \
  models::qwen3_5::qwen3_5_prefill_memory_tests::prefill_stage_peak_memory_probe
```
