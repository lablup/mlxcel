# Long-prompt prefill benchmark ladder (epic #623 #624)

All prior benchmark CSVs (`benchmarks/cuda_gb10_*.csv`, the Metal sets) use
prompts of 8-66 tokens. At that length the prefill measurement is dominated by
graph-build and kernel-launch overhead, not matmul/attention throughput, so it
cannot validate any prefill optimization (MoE prefill fixes, sm_121 GEMM work,
chunked-prefill tuning). This ladder measures prefill in the compute-bound
regime by driving deterministic prompts of 512 / 2048 / 8192 / 32768 tokens.

## What was added

- `mlxcel-bench-decode --prompt-tokens N` synthesizes a deterministic prompt by
  repeating a fixed corpus paragraph, tokenizing with the model's own tokenizer,
  and truncating to exactly N tokens. The length is capped at the model context
  (leaving room for `--max-tokens` generation); the actual length used is
  reported in the existing `prompt_tokens` column.
- `scripts/bench_decode.sh --prompt-tokens N` threads the flag through and
  appends one new CSV column, `prompt_target_len` (column 15). The first 14
  columns are unchanged, so historical CSVs stay comparable. The short-prompt
  default leaves `prompt_target_len` empty.
- `scripts/bench_longprompt.sh` runs a representative subset across the ladder
  into a single CSV, reusing `bench_decode.sh` per cell (so warmup, OOM
  classification, and the schema are identical).

## Environment

| Item | Value |
|------|-------|
| **Hardware** | NVIDIA GB10 (DGX Spark), 122 GB unified LPDDR5x |
| **OS** | Linux aarch64, kernel 6.17 |
| **Backend** | CUDA, SM 12.1 |
| **Build** | `cargo build --release --features cuda` |
| **mlxcel version** | 0.3.3 |
| **Harness** | same-process `mlxcel-bench-decode`, warm prefill (warmup=4, max-tokens=32) |
| **Prompt** | synthesized via `--prompt-tokens`, deterministic repeated corpus |

## How to reproduce

```bash
# Single model, single length (acceptance check):
./target/release/mlxcel-bench-decode --model ./models/llama-3.1-8b-4bit --prompt-tokens 8192

# One (model, length) cell through the shell harness (adds prompt_target_len):
./scripts/bench_decode.sh models/llama-3.1-8b-4bit --prompt-tokens 8192

# Full representative subset across the ladder (auto-named CSV under benchmarks/):
./scripts/bench_longprompt.sh
# or with an explicit output path and smaller generation budget:
./scripts/bench_longprompt.sh --output benchmarks/cuda_gb10_longprompt_2026-07-03.csv --max-tokens 32 --warmup-tokens 4
```

The ladder is 512 / 2048 / 8192 / 32768 tokens. A cell that exceeds the model
context is capped (the `prompt_tokens` column shows the actual length). A cell
that runs out of memory (common at 32768 for the large MoE models) is recorded
with the usual `SKIP:oom` / `SKIP:oom_estimate` classification and the sweep
continues.

## Results (prefill throughput, tok/s)

Baseline CSV: `benchmarks/cuda_gb10_longprompt_2026-07-03.csv`. Prefill tok/s at
each ladder length (higher is better). `fail` = process aborted (exit 134,
allocation/capacity limit at that length); `oom` = classified out-of-memory.

| Model | kind | 512 | 2048 | 8192 | 32768 |
|-------|------|----:|-----:|-----:|------:|
| llama-3.1-8b-4bit | dense | 3286.7 | 2946.0 | 1666.3 | fail |
| qwen2.5-7b-4bit | dense | 3487.7 | 3125.7 | 1805.0 | 1176.0 (cap 32736) |
| qwen3-30b-a3b-4bit | MoE | 162.6 | 162.2 | fail | fail |
| mixtral-8x7b-4bit | MoE | 15.3 | 13.8 | 25.3 | fail |
| gemma-4-31b-it-4bit | dense | 390.3 | 489.2 | fail | oom |

Reading the numbers:

- Dense 7-8B models prefill at 1600-3500 tok/s and fall as the prompt grows (the O(n^2) attention term), so they are already compute-bound at 512, exactly the regime short 8-66 token prompts cannot reach. qwen2.5-7b (4 KV heads, 28 layers) sustains 32768; llama-3.1-8b (8 KV heads, 32 layers) aborts there on the same host, so the ceiling is model-shape dependent, not a fixed context wall.
- MoE prefill is the standout weakness this measurement exists to expose. mixtral-8x7b prefills at only 13-25 tok/s (33s at 512, 148s at 2048, 324s at 8192 of wall time), and qwen3-30b-a3b holds ~162 tok/s but aborts at 8192+. Both are far below the dense models at the same length, which is precisely the MoE prefill target of epic #623.
- Large dense gemma-4-31b prefills well at short lengths (390-489 tok/s, still rising into the compute-bound regime) but cannot reach 8192+ on this host.

This is the measurement foundation the rest of epic #623 builds on: a prefill optimization now has a length axis to move on, and a clear MoE-vs-dense gap to close, instead of a launch-overhead floor.

## Serving telemetry (companion)

The server side of this issue adds two Prometheus histograms, populated once per
request completion in the batch scheduler's `finalize_completed` (never per
token):

- `mlxcel_request_ttft_ms` -- per-request time to first token (prefill latency).
- `mlxcel_request_decode_tok_s` -- per-request decode throughput.

Both are exposed on `/metrics` (requires `--metrics`). Drive concurrent load and
read them with:

```bash
./target/release/mlxcel-server -m models/qwen2.5-0.5b-bf16 --port 8080 --metrics &
python3 scripts/bench_serving_concurrency.py --concurrency 1,2,4,8
curl -s localhost:8080/metrics | grep -E "ttft|decode_tok"
```
