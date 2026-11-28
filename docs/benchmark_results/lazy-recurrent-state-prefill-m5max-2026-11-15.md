# Lazy recurrent cache state in long prefills (issue #3061), M5 Max, 2026-11-15

A recurrent or hybrid layer stored its per-layer cache state as a lazy node derived from a full-length tensor:

- a conv or short-conv tail as `contiguous(slice(conv_input))`;
- a final SSM state built by ops that the layer output never consumes.

Nothing evaluates the caches until the forward ends, so in a single-pass prefill every layer's full-length source stays alive at once. `contiguous` copies only when it runs. The fix stores each tail through `mlxcel_core::utils::materialized_state_tail` and schedules each final state through `schedule_prefill_state`. Both use `async_eval` once the source has at least 256 rows, so the source is released after the layer's own consumers run. Decode steps and short verify blocks are unchanged.

Host: Apple M5 Max, 40-core GPU, 128 GB, macOS 27.2.1, MLX `9f806b3606eaa9f7c807238f6bd465185cd94d6d`. Base is `main` at `20ff89d9` (after #3054). The fix is branch `fix/lazy-conv-state-prefill-retention`. Both are release builds with `metal,accelerate`.

Run setup:

- Prompt from `scripts/bench/metal_sdpa_chunk/make_prompt.py 28000`. Each tokenizer reads it as 17.7K to 28.7K tokens.
- `mlxcel generate`, greedy, 48 tokens, reasoning included.
- Peak is the "peak memory footprint" from `/usr/bin/time -l`. Prefill is from `--profile`.
- 60 to 90 s pause before every run.

Every cell's generated text is byte-identical between the arms (`output_md5`).

## Where the Qwen 3.5 peak went

On qwen3.5-4b-4bit at 28,707 tokens in one pass, MLX active memory was read after evaluating `h` at every layer:

- It rose by 0.47 GB at each of the 24 linear-attention layers. That is one `[1, L+3, 8192]` fp16 conv input per layer.
- It stopped rising once the tail was evaluated inside the layer.

The full-attention layers added only their KV, plus the f32 `[L, L]` causal mask (3.3 GB, #3051) at the first one. With the MLX buffer cache bounded to 1 MiB, what remains of the 14.5 GB peak is:

| component | size |
|---|---|
| weights | about 2.4 GB |
| causal mask | 3.3 GB |
| one layer's working set | 3 to 4 GB |
| MLX buffer cache | about 2.5 GB |
| KV | 0.94 GB |

## Results

In the first run each cell ran the fix first and main second. The prefill times of the cells that read slower were then re-measured in reversed order (main first). The first run used the build before the error-path commit, which does not change what is evaluated. The reversed control used the final build.

| model | prefill | peak, main | peak, fix | prefill, main / fix (reversed control) |
|---|---|---|---|---|
| qwen3.5-4b-4bit | single pass | 25.35 GB | 14.53 GB | 8.99 / 8.95 s |
| qwen3.5-4b-4bit | 2048 chunks | 6.82 GB | 6.06 GB | 6.68 / 6.55 s |
| qwen3.8-27b-4bit | single pass | 59.50 GB | 31.83 GB | 53.10 / 52.66 s |
| qwen3.8-27b-4bit | 2048 chunks | 22.71 GB | 20.69 GB | 40.99 / 41.26 s |
| qwen3-next-80b-a3b-instruct-4bit | single pass | 77.88 GB | 62.01 GB | 12.00 / 12.12 s |
| granite-4.0-h-tiny-4bit | 2048 chunks | 50.10 GB | 9.85 GB | 8.17 / 8.06 s |
| falcon-h1-tiny-90m-instruct-4bit | 2048 chunks | 13.48 GB | 3.99 GB | 3.91 / 3.96 s |
| mamba2-1.3b-4bit | single pass | 46.38 GB | 18.26 GB | 5.64 / 5.72 s |
| mamba2-1.3b-4bit | 2048 chunks | 9.45 GB | 3.38 GB | 3.87 / 3.72 s |
| falcon-mamba-7b-4bit-instruct | single pass | 32.83 GB | 11.40 GB | 9.39 / 9.06 s |
| falcon-mamba-7b-4bit-instruct | 2048 chunks | 8.12 GB | 7.15 GB | 8.41 / 8.40 s |
| ai21-jamba-reasoning-3b-4bit | single pass | 13.81 GB | 6.45 GB | 4.19 / 4.14 s |
| ai21-jamba-reasoning-3b-4bit | 2048 chunks | 3.56 GB | 3.25 GB | 4.06 / 4.07 s |
| lfm2.5-2.6b-8bit | single pass | 15.08 GB | 13.47 GB | 2.46 / 2.49 s |
| lfm2.5-2.6b-8bit | 2048 chunks | 6.42 GB | 6.14 GB | 2.20 / 2.19 s |

Prefill times for the 2048-chunk Qwen cells and the cells not repeated come from the first run. In the reversed control every repeated cell is within ±3.5%, with no consistent sign.

In the first run, the fix-first cells read 3 to 10% slower: Granite, Falcon-H1, Jamba and Qwen3-Next. Reversed, that difference disappeared, so it came from the run order, not from the extra `async_eval`.

In Granite and Falcon-H1, the lazy final SSM state kept each Mamba layer's `[B, H, C, C]` f32 decay matrix alive. That is why their waste is large even with 2048-token chunks.

A single 28K pass of granite-4.0-h-tiny or falcon-h1-tiny runs out of memory on main and on the fix, because one layer's whole-sequence SSD scan exceeds the GPU budget by itself. On the fix this must stay a clean `generation failed` error, as on main. The first fix build aborted instead, because the eager `async_eval` went through the infallible FFI binding. The final build keeps that error for the caller's `try_eval`, and both cells exit 1 with main's message (run "OOM path").

## Gemma 4: sliding-window caches (#3062)

gemma-4-12b has no recurrent state and already projects only the last row through its LM head. The largest full-prompt item in its single pass is elsewhere. On its first fill, `RotatingKVCache::update_concat` stores every prompt token for each of the 40 sliding layers, and only the first decode step trims them to the 1024-token window.

An env-gated experiment kept only the window on the first fill. It was not committed, because `trim()` and the MTP buffered path rely on the full first fill. Results (run "gemma sliding-window experiment"):

| gemma-4-12b-it-4bit, 28,710 tokens | peak | prefill |
|---|---|---|
| single pass, main | 27.30 GB | 20.27 s |
| single pass, window kept on the first fill | 18.58 GB | 19.94 s |
| same, MLX cache bounded to 1 MiB | 16.55 GB | 20.38 s |
| 2048-token chunks, main and experiment | 14.09 GB | 18.41 s |

The output was identical. The single-pass saving is 8.72 GB, close to the code-based estimate of 40 × 224 MiB.

## How to rerun

```
cargo build --release --features metal,accelerate --bin mlxcel   # once on main, once on the branch
BASE_BIN=<main build> NEW_BIN=<branch build> GEN_TOKENS=48 SERVER_MODELS="" \
  CLI_CELLS="qwen3.5-4b-4bit:0 granite-4.0-h-tiny-4bit:2048 mamba2-1.3b-4bit:0" \
  scripts/bench/prefill_peak/ab_prefill.sh <out_dir>
```

`ab_prefill.sh` alternates the arm order between repetitions. On the M5 Max, run nothing else on the GPU while it runs and keep the pauses: back-to-back runs throttle.

## Raw data

`lazy-recurrent-state-prefill-m5max-2026-11-15/summary.csv` holds one row per run:

- the run and its order;
- the arm: `main`, `eager`, or the Gemma window-trim experiment;
- model, prefill chunk, prompt tokens, prefill ms, peak footprint bytes, exit code;
- the md5 of the generated text.
