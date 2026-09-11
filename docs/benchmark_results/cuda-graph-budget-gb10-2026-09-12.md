# CUDA graph capture budgets on GB10: op and byte limits against model shape (2026-09-12)

Issue #1798. Host: NVIDIA GB10 (sm_121, DGX Spark), Linux aarch64, MLX pin `81ba1c6a` (`src/lib/mlx-cpp/CMakeLists.txt`), CUDA release build (`cargo build --release --features cuda`, `MLX_CUDA_ARCHITECTURES=121`), toolchain 1.97.1, source tree `d2b60c8e` (main at the time of the sweep) plus the policy module of this PR compiled in but not yet called from any entry point, so the binary under test behaves exactly as main and every arm is the same binary with environment knobs. `MLX_ENABLE_TF32=0` on every run (no numerical comparison here depends on it; nothing below compares floats). Warm PTX cache and warm page cache (one discarded warm-up run per model per phase).

## The mechanism, in the pinned source

MLX's CUDA backend captures every eval into a CUDA graph and commits the graph after any op that pushes either of two counters past a per-architecture budget (`CommandEncoder::needs_commit`, `mlx/backend/cuda/device.cpp:458-460`, called from `eval.cpp:73`): `node_count_ > max_ops_per_graph_` or `(bytes_in_graph_ >> 20) > max_mb_per_graph_`. The budgets come from `get_graph_limits` (`device.cpp:181-202`), which switches on `major * 100 + minor * 10`: 20 ops and 400 "MB" for 800 (A100); 100 ops and 1000 for 900, 1000 and 1200 (H100, B200, consumer Blackwell); 20 ops and 25 for 1210 (DGX Spark); 20 and 100 for anything unlisted. Both pass through `env::max_ops_per_buffer` / `env::max_mb_per_buffer` (`mlx/utils.h:193-202`), function-local statics read once from `MLX_MAX_OPS_PER_BUFFER` and `MLX_MAX_MB_PER_BUFFER`, and the pair is latched once per stream in the `CommandEncoder` constructor (`device.cpp:211`).

The byte counter is not bytes. `set_input_array` (`device.cpp:228`) does `bytes_in_graph_ += arr.data_size()`, and `array::data_size()` is documented as "in units of `item_size` (not bytes)" (`mlx/array.h:346-348`). So GB10's 25 "MB" is `25 << 20 = 26,214,400` input elements, and any single op whose input has more elements than that commits its own graph, whatever preceded it. Measured against the checkpoints on this host, reading each safetensors header:

| checkpoint | largest input arrays | elements | over 26.2M |
|---|---|---|---|
| `laguna-xs-2.1-nvfp4` | each stacked 256-expert NVFP4 projection (`256 x 512 x 2048 / 8`) | 33.5M | yes, 3 per MoE layer, about 120 `gather_qmm` per token |
| `laguna-xs-2.1-nvfp4` | `lm_head.weight`, `embed_tokens.weight` (bf16) | 205.5M | yes |
| `qwen3.5-4b-4bit` | tied `embed_tokens.weight` (u32, 4-bit affine) | 79.5M | yes, once per token (lm_head) |
| `llama-3.1-8b-4bit` | `lm_head.weight`, `embed_tokens.weight` (u32) | 65.7M | yes, once per token; the seven per-layer projections sum to 26.6M, so the byte cap also commits about once per layer |
| `gemma-3-4b-it-4bit` | `lm_head.weight`, `embed_tokens.weight` (u32) | 83.9M | yes (not measured below) |

(The #1799 record's remark that Qwen 3.5 has "no matrix over the byte cap" was wrong: its tied 4-bit embedding is over it. The difference between the shapes is how many over-budget ops a token issues: one on a dense checkpoint, about 120 on Laguna.)

## Host idleness

Every measured run held the GPU under the scratchpad lock. Each phase started only after the 1-minute load average was under 0.6 in two samples 20 s apart, and the 1-minute load average before every run is recorded in the tables. The self-hosted CI runner is on this host: the single-stream phase (01:22 to 01:34) finished before the one CI job of the window was created (`gh run list`: 01:36:21 local), so its loads (0.69 to 1.58, the process under test's own 130% CPU during model load is what lifts load1 above 1 once a sweep is under way) are the process under test alone. That CI job (a `cargo check`, two `rustc` at 95% CPU) overlapped the first DFlash phase-split run, which is why that phase was discarded and re-run; from then on every harness also waited, before each run, until no compiler process (`rustc`, `cc1plus`, `nvcc`, `cmake`, `ninja`, `ld`, `clang`) had been alive for two samples 5 s apart, and recorded whether a CI job was running during the run (the "CI job during run" column).

## Method

Three harnesses, all driving the same binary.

- **Single-stream decode and short prefill**: `mlxcel-bench-decode` (`src/bin/bench_decode.rs`), which loads the model once, runs a discarded 20-token warm-up pass and then a measured pass of 200 tokens with `--ignore-eos` on the raw Python source prompt of the #1782 and #1799 records (`--no-chat-template`; 101 to 112 tokens depending on the tokenizer, as the harness reports). It reports prefill and decode time separately and the MLX allocator high-water mark for the whole run. n = 3 per arm after one discarded warm-up run, arms interleaved round-robin with a rotating start so drift spreads across the table.
- **Long prefill**: the same binary with `--prompt-tokens 2048` (a synthesized 2048-token prompt) and 16 generated tokens.
- **Batched serving**: one `mlxcel-server --max-batch-size 8 --ignore-eos` per (arm, round), `scripts/bench_serving_concurrency.py` at concurrency 1, 4 and 8 with 128-token prompts and 200 completion tokens; the concurrency-1 level doubles as the server's warm-up. Aggregate tok/s is all completion tokens over the level's wall time.
- **Speculative phase split**: `mlxcel generate` with the Laguna DFlash drafter at block 4 through the #1799 CLI harness, whose `DFlash:` diagnostics line splits a round into drafter host build, verify host build and device sync.

Arms are environment variables only: `nograph` is `MLX_USE_CUDA_GRAPHS=0`; `ops100` is `MLX_MAX_OPS_PER_BUFFER=100`; `mb1000` is `MLX_MAX_MB_PER_BUFFER=1000`; `both` is both raised to 100 and 1000, the row MLX gives cc 9.0, 10.0 and 12.0; `both400` is 100 and 400, the A100 byte budget. `default` sets nothing and gets MLX's 20 ops and 25 "MB" for cc 12.1.

## Single-stream decode and short prefill (idle host, n = 3, same binary)

Decode tok/s over 200 tokens after a 101 to 112 token prompt; prefill is that prompt; MLX peak is the allocator high-water mark over load, warm-up, prefill and decode.

### `laguna-xs-2.1-nvfp4`

| config | n | prompt tok | decode tok/s mean (min to max) | vs default | decode ms/200 tok | prefill ms mean (min to max) | vs default | prefill tok/s | MLX peak GB (min to max) | load1 (min to max) | CI job during run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| default | 3 | 112 | 32.02 (31.33 to 32.66) |  | 6247 | 441.6 (428.8 to 453.7) |  | 254 | 21.66 (21.66 to 21.66) | 0.69 to 1.02 | 0 of 3 |
| nograph | 3 | 112 | 33.79 (33.60 to 33.95) | +5.5% | 5919 | 213.7 (212.5 to 215.1) | -51.6% | 524 | 21.69 (21.69 to 21.70) | 0.72 to 1.05 | 0 of 3 |
| ops100 | 3 | 112 | 32.48 (32.12 to 32.81) | +1.4% | 6159 | 445.5 (437.1 to 455.5) | +0.9% | 251 | 21.69 (21.69 to 21.69) | 0.72 to 1.13 | 0 of 3 |
| mb1000 | 3 | 112 | 35.57 (35.29 to 35.79) | +11.1% | 5623 | 421.2 (420.3 to 422.2) | -4.6% | 266 | 21.69 (21.69 to 21.69) | 0.95 to 1.06 | 0 of 3 |
| both | 3 | 112 | 37.60 (37.14 to 37.94) | +17.4% | 5320 | 414.4 (408.8 to 419.5) | -6.2% | 270 | 22.03 (22.03 to 22.04) | 0.97 to 1.02 | 0 of 3 |
| both400 | 3 | 112 | 37.95 (37.68 to 38.31) | +18.5% | 5270 | 420.0 (413.8 to 423.8) | -4.9% | 267 | 22.03 (22.03 to 22.03) | 0.97 to 1.02 | 0 of 3 |

### `qwen3.5-4b-4bit`

| config | n | prompt tok | decode tok/s mean (min to max) | vs default | decode ms/200 tok | prefill ms mean (min to max) | vs default | prefill tok/s | MLX peak GB (min to max) | load1 (min to max) | CI job during run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| default | 3 | 111 | 67.38 (66.95 to 67.63) |  | 2968 | 105.0 (104.3 to 106.4) |  | 1057 | 2.83 (2.82 to 2.84) | 1.01 to 1.38 | 0 of 3 |
| nograph | 3 | 111 | 60.85 (60.64 to 60.96) | -9.7% | 3287 | 89.3 (89.0 to 90.0) | -14.9% | 1242 | 2.83 (2.80 to 2.86) | 0.99 to 1.04 | 0 of 3 |
| ops100 | 3 | 111 | 70.61 (70.33 to 70.78) | +4.8% | 2833 | 108.9 (108.1 to 109.7) | +3.7% | 1019 | 3.06 (3.04 to 3.08) | 0.99 to 1.32 | 0 of 3 |
| mb1000 | 3 | 111 | 68.39 (68.06 to 68.57) | +1.5% | 2924 | 105.9 (104.1 to 107.0) | +0.8% | 1049 | 2.85 (2.84 to 2.85) | 1.09 to 1.58 | 0 of 3 |
| both | 3 | 111 | 70.03 (69.75 to 70.36) | +3.9% | 2856 | 95.3 (94.0 to 96.7) | -9.2% | 1164 | 2.91 (2.90 to 2.93) | 1.08 to 1.45 | 0 of 3 |
| both400 | 3 | 111 | 70.01 (69.92 to 70.11) | +3.9% | 2857 | 96.8 (94.5 to 99.8) | -7.8% | 1147 | 2.92 (2.90 to 2.97) | 0.99 to 1.42 | 0 of 3 |

### `llama-3.1-8b-4bit`

| config | n | prompt tok | decode tok/s mean (min to max) | vs default | decode ms/200 tok | prefill ms mean (min to max) | vs default | prefill tok/s | MLX peak GB (min to max) | load1 (min to max) | CI job during run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| default | 3 | 101 | 52.98 (52.97 to 52.99) |  | 3775 | 60.2 (59.4 to 60.9) |  | 1678 | 4.68 (4.68 to 4.68) | 0.83 to 0.96 | 0 of 3 |
| nograph | 3 | 101 | 49.80 (49.75 to 49.84) | -6.0% | 4016 | 52.5 (51.8 to 53.0) | -12.8% | 1925 | 4.75 (4.75 to 4.75) | 0.77 to 1.00 | 0 of 3 |
| ops100 | 3 | 101 | 52.86 (52.62 to 53.00) | -0.2% | 3784 | 59.9 (58.5 to 60.8) | -0.5% | 1688 | 4.68 (4.68 to 4.68) | 0.82 to 0.97 | 0 of 3 |
| mb1000 | 3 | 101 | 52.92 (52.86 to 52.98) | -0.1% | 3779 | 64.6 (63.2 to 65.4) | +7.3% | 1565 | 4.75 (4.75 to 4.75) | 0.85 to 0.98 | 0 of 3 |
| both | 3 | 101 | 47.99 (47.94 to 48.02) | -9.4% | 4168 | 66.6 (66.5 to 66.8) | +10.6% | 1517 | 4.86 (4.86 to 4.86) | 0.78 to 0.90 | 0 of 3 |
| both400 | 3 | 101 | 48.03 (47.98 to 48.13) | -9.3% | 4164 | 65.8 (64.7 to 67.4) | +9.4% | 1534 | 4.86 (4.86 to 4.86) | 0.72 to 0.91 | 0 of 3 |

## Long prefill: 2048 tokens (one prefill chunk at the default `MLXCEL_PREFILL_CHUNK`), 16 decode tokens (idle host, n = 3)


### `laguna-xs-2.1-nvfp4`

| config | n | prompt tok | decode tok/s mean (min to max) | vs default | decode ms/200 tok | prefill ms mean (min to max) | vs default | prefill tok/s | MLX peak GB (min to max) | load1 (min to max) | CI job during run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| default | 3 | 2048 | 33.38 (32.11 to 34.25) |  | 480 | 7536.0 (7414.5 to 7609.7) |  | 272 | 22.74 (22.74 to 22.74) | 0.65 to 0.83 | 0 of 3 |
| both | 3 | 2048 | 36.16 (35.55 to 36.71) | +8.3% | 442 | 7421.6 (7352.0 to 7487.1) | -1.5% | 276 | 29.79 (29.79 to 29.79) | 0.56 to 0.89 | 0 of 3 |
| nograph | 3 | 2048 | 35.85 (35.67 to 35.98) | +7.4% | 446 | 3243.9 (3242.2 to 3246.9) | -57.0% | 631 | 23.93 (23.93 to 23.93) | 0.81 to 0.87 | 0 of 3 |

### `qwen3.5-4b-4bit`

| config | n | prompt tok | decode tok/s mean (min to max) | vs default | decode ms/200 tok | prefill ms mean (min to max) | vs default | prefill tok/s | MLX peak GB (min to max) | load1 (min to max) | CI job during run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| default | 3 | 2048 | 67.04 (66.84 to 67.22) |  | 239 | 1539.9 (1530.4 to 1547.8) |  | 1330 | 7.96 (7.94 to 7.97) | 0.55 to 0.70 | 0 of 3 |
| both | 3 | 2048 | 65.67 (64.82 to 66.93) | -2.0% | 244 | 1716.1 (1708.5 to 1720.4) | +11.4% | 1193 | 13.20 (13.20 to 13.20) | 0.57 to 1.33 | 0 of 3 |
| nograph | 3 | 2048 | 61.15 (60.90 to 61.43) | -8.8% | 262 | 1335.8 (1331.8 to 1338.5) | -13.3% | 1533 | 8.74 (8.72 to 8.78) | 0.56 to 0.59 | 0 of 3 |

