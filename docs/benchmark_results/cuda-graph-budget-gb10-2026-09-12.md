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

## Graph accounting: instantiation, commits and where the time goes (nsys, single-stream decode)

`nsys profile -t cuda --cuda-graph-trace=node` around `mlxcel-bench-decode` at 400 and 200 generated tokens for the same arm, differenced, so model load, warm-up and prefill cancel and every per-token figure below is the decode work of the extra 200 tokens. Idle host, GPU held, one run per cell; the profiled runs sit within 3% of the unprofiled tables (Laguna default 31.1 to 31.5 tok/s against 32.0, `both` 38.1 to 38.2 against 37.6; Qwen 64.7 against 67.4, 68.2 to 68.5 against 70.0), so nsys inflation is small here. "Host CUDA API" is the summed host time of every CUDA API call per token, minus `cudaMemcpyAsync`, whose call count does not change with tokens (it is the weight upload) and whose time differed between the two runs of a pair by up to 9 ms per token in either direction, a load-time artifact of the differencing, not decode work. Wall per token is `1000 / tok/s` of the profiled 200-token run.

| arm | wall ms/token | kernel launches/token | graphs launched/token (`cudaGraphLaunch`) | `cudaGraphExecUpdate`/token | `cudaGraphInstantiate`/token (host ms/token) | host CUDA API ms/token | of which `ExecUpdate` + `GraphLaunch` |
|---|---|---|---|---|---|---|---|
| Laguna default | 31.8 | 2364.9 | 200.0 | 200.0 | 0.04 (0.03) | 17.3 | 6.6 + 4.5 |
| Laguna `both` | 26.2 | 2364.9 | 24.0 | 23.7 | 0.29 (0.34) | 17.0 | 7.9 + 5.3 |
| Qwen 3.5 default | 15.5 | 1307.0 | 65.0 | 65.0 | 0.00 (0.01) | 8.9 | 3.8 + 2.7 |
| Qwen 3.5 `both` | 14.7 | 1307.0 | 14.0 | 14.0 | 0.00 (0.00) | 8.5 | 3.8 + 2.7 |
| Llama 3.1 8B default | 19.6 | 493.0 | 35.0 | 35.0 | 0.00 (0.00) | 6.3 | 1.9 + 1.5 |
| Llama 3.1 8B `both` | 21.1 | 553.5 | 6.0 | 5.8 | 0.14 (1.14) | 17.8 | 2.2 + 2.1 |

What the table says:

- **The work is identical.** Kernel launches per token do not change with the budget, on either model. The budget only moves graph boundaries.
- **Commits per token are the whole story on Laguna.** At the defaults a Laguna token is 200 committed graphs (the 120 or so byte-cap commits from the expert stacks and the lm_head plus the 20-op cap over the rest), and 2,365 nodes over 200 graphs is under 12 nodes per graph. `both` makes it 24 graphs of up to 100 nodes. Qwen goes from 65 to 14.
- **The saving is not host time.** Host CUDA API time per token is flat (17.3 to 17.0 ms on Laguna, 8.9 to 8.5 on Qwen): fewer `cudaGraphExecUpdate` and `cudaGraphLaunch` calls, but each on a bigger graph, so their sum barely moves. What falls is the wall time, 5.6 ms per token on Laguna for 176 fewer boundaries and 0.9 ms on Qwen for 51 fewer, about 20 to 30 us per boundary either way. That is the device-side cost of a graph boundary: graph launch latency plus the `cudaEventRecord` / `cudaStreamWaitEvent` pair MLX puts between consecutive graphs (201 and 200 per token at the Laguna default, 25 and 24 under `both`), during which the GPU has nothing queued. So this is a reduction in total, not a shift between phases: the host side is unchanged and the device side has fewer idle gaps.
- **Instantiation is visible and small.** The graph cache hits on Qwen at both budgets (no instantiation in steady state). On Laguna `both` instantiates about one graph every 3.4 tokens (58 over the differenced 200 tokens against 8 at the defaults), 1.15 ms each, 0.34 ms per token, 1.3% of the token; the MoE routing changes the expert-gather node set from token to token and larger graphs have more chances to differ. The gain of 5.6 ms per token is 16x that cost. The graph-exec LRU (`MLX_CUDA_GRAPH_CACHE_SIZE`, 2000 on CUDA builds) is nowhere near its capacity at 24 keys per token.
- **Llama's regression is instantiation, and it is the dense counter-example.** At the defaults a Llama token is 35 graphs (the byte cap commits about once per layer, since its seven 4-bit projections sum to 26.6M words, plus the lm_head), the exec cache hits every token and the host spends 6.3 ms per token in the CUDA API. Under `both` a token is 6 graphs of up to 100 nodes spanning several layers, and one of them is re-instantiated every 7 tokens (0.14 per token, 8.2 ms each, 1.1 ms per token; 243 instantiations over the 400-token run against 23): every 100-node graph contains an attention op whose launch geometry moves with the KV length, so when it moves the whole 100-node graph misses the exec cache instead of a 12-node one, and a dense 100-node instantiate costs 7x Laguna's. The host also starts blocking in `cudaEventSynchronize` (0.93 per token, 11.1 ms per token, none at the defaults) and kernel launches per token rise from 493 to 553 (the rebuild path's own work). Wall goes 19.6 to 21.1 ms per token under the profiler, -7.5%, against -9.4% unprofiled. The same accounting on Laguna shows why the sign flips: 176 fewer boundaries at 30 us each buys 5.6 ms, and its instantiations cost 0.34 ms.
- **Summed kernel time is not a clean measure with larger graphs** (33.5 to 38.1 ms per token on Laguna while the wall fell) because independent kernels inside one graph overlap and each stretches; it is reported only to note that it is not the metric. Kernel launch counts and wall time are.

## More MoE shapes: single-stream decode and short prefill (idle host, n = 3, same binary)

Same harness and prompt as the first single-stream table, three arms. Stacked routed-expert projection per checkpoint at its packed width: gpt-oss-20b 33.2M words (32 experts, mxfp4, over the 26.2M budget), qwen3-30b-a3b 25.2M (128 experts, affine 4-bit, under it), qwen3.5-35b-a3b 33.5M (256 experts, affine 4-bit, Laguna's shape), gemma-4-26b-a4b 31.7M (128 experts, affine 4-bit, over it).

### `gpt-oss-20b-mxfp4`

| config | n | prompt tok | decode tok/s mean (min to max) | vs default | decode ms/200 tok | prefill ms mean (min to max) | vs default | prefill tok/s | MLX peak GB (min to max) | load1 (min to max) | CI job during run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| default | 3 | 101 | 75.12 (72.54 to 76.91) |  | 2664 | 790.3 (782.9 to 803.5) |  | 128 | 14.98 (14.98 to 14.98) | 0.80 to 1.10 | 1 of 3 |
| both | 3 | 101 | 70.08 (61.40 to 75.34) | -6.7% | 2878 | 993.9 (884.1 to 1193.1) | +25.8% | 104 | 22.20 (20.37 to 24.20) | 0.69 to 0.95 | 0 of 3 |
| nograph | 3 | 101 | 72.66 (72.57 to 72.82) | -3.3% | 2753 | 728.6 (724.8 to 731.4) | -7.8% | 139 | 17.72 (17.72 to 17.72) | 0.87 to 1.00 | 0 of 3 |

### `qwen3-30b-a3b-4bit`

| config | n | prompt tok | decode tok/s mean (min to max) | vs default | decode ms/200 tok | prefill ms mean (min to max) | vs default | prefill tok/s | MLX peak GB (min to max) | load1 (min to max) | CI job during run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| default | 3 | 101 | 78.94 (74.91 to 81.03) |  | 2537 | 654.9 (646.8 to 663.7) |  | 154 | 17.30 (17.30 to 17.30) | 1.03 to 1.36 | 0 of 3 |
| both | 3 | 101 | 95.19 (94.18 to 95.70) | +20.6% | 2101 | 638.5 (625.5 to 657.6) | -2.5% | 158 | 17.82 (17.82 to 17.82) | 0.96 to 1.54 | 0 of 3 |
| nograph | 3 | 101 | 75.42 (74.75 to 76.16) | -4.5% | 2652 | 233.0 (232.5 to 233.3) | -64.4% | 433 | 17.35 (17.35 to 17.35) | 1.07 to 1.41 | 0 of 3 |

### `qwen3.5-35b-a3b-4bit`

| config | n | prompt tok | decode tok/s mean (min to max) | vs default | decode ms/200 tok | prefill ms mean (min to max) | vs default | prefill tok/s | MLX peak GB (min to max) | load1 (min to max) | CI job during run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| default | 3 | 111 | 62.28 (58.24 to 66.14) |  | 3220 | 482.3 (474.8 to 487.1) |  | 230 | 20.00 (19.99 to 20.01) | 1.20 to 1.78 | 0 of 3 |
| both | 3 | 111 | 76.32 (76.16 to 76.49) | +22.5% | 2620 | 464.6 (461.4 to 469.7) | -3.7% | 239 | 20.52 (20.52 to 20.52) | 1.41 to 1.81 | 0 of 3 |
| nograph | 3 | 111 | 53.14 (51.77 to 54.69) | -14.7% | 3766 | 239.0 (236.2 to 240.9) | -50.5% | 465 | 20.04 (20.00 to 20.06) | 1.53 to 1.69 | 0 of 3 |

### `gemma-4-26b-a4b-it-4bit`

| config | n | prompt tok | decode tok/s mean (min to max) | vs default | decode ms/200 tok | prefill ms mean (min to max) | vs default | prefill tok/s | MLX peak GB (min to max) | load1 (min to max) | CI job during run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| default | 3 | 115 | 61.18 (60.62 to 61.48) |  | 3269 | 702.2 (684.1 to 713.3) |  | 164 | 15.51 (15.51 to 15.51) | 0.72 to 1.45 | 0 of 3 |
| both | 3 | 115 | 61.12 (60.95 to 61.32) | -0.1% | 3272 | 685.6 (682.6 to 687.4) | -2.4% | 168 | 16.15 (16.15 to 16.15) | 0.72 to 1.17 | 0 of 3 |
| nograph | 3 | 115 | 54.55 (54.50 to 54.60) | -10.8% | 3666 | 243.2 (242.1 to 244.0) | -65.4% | 473 | 15.59 (15.59 to 15.59) | 0.72 to 0.95 | 0 of 3 |

## Phase split: is the gain a reduction or a shift? (Laguna DFlash block 4, `mlxcel generate`, idle host, n = 3)

The failure mode this rules out is the one `MLX_MAX_MB_PER_BUFFER=400` showed in #1782, where the byte budget moved time out of the drafter's host build and into the device sync without changing the round. The #1799 CLI harness splits a speculative round into the drafter's host graph build (`draft host`), the verify forward's host build (`verify host`) and the wait for the round's device work (`device sync`); the three add up to the round wall to within a millisecond. A first pass of this phase overlapped the CI runner's `cargo check` and was discarded; this table is the re-run with the per-run compiler gate (no CI job during any of its 19 runs). `off` is classic decode through the CLI, which has no same-process warm-up, so it sits under the `mlxcel-bench-decode` numbers above; the `both` gain on it (+15%) matches #1799's control (+16%).

| config | n | tok/s mean (min to max) | vs off | accepted/round | round wall ms | device sync ms/round (min to max) | draft host ms/round | verify host ms/round | load1 (min to max) |
|---|---|---|---|---|---|---|---|---|---|---|
| off | 3 | 29.40 (28.85 to 30.35) |  |  | | | |  | 0.80 to 0.83 |
| b4 | 3 | 36.90 (33.29 to 39.03) | 1.26x | 1.41 | 65.6 | 48.8 (48.7 to 48.9) | 12.9 | 3.9 | 0.80 to 1.10 |
| both-off | 3 | 33.80 (33.68 to 34.00) | 1.15x |  | | | |  | 0.87 to 1.04 |
| both-b4 | 3 | 40.07 (39.57 to 40.86) | 1.36x | 1.41 | 60.1 | 48.1 (46.9 to 49.1) | 7.6 | 4.3 | 0.69 to 1.07 |
| nograph-off | 3 | 32.57 (32.46 to 32.70) | 1.11x |  | | | |  | 0.83 to 1.11 |
| nograph-b4 | 3 | 41.28 (41.12 to 41.48) | 1.40x | 1.41 | 58.4 | 45.7 (45.6 to 45.9) | 7.7 | 4.9 | 0.82 to 1.05 |

Under `both` at block 4 the drafter's host build falls from 12.9 to 7.6 ms per round (its `async_eval` enqueue commits fewer graphs), the device sync is flat (48.8 to 48.1 ms) and the round wall falls from 65.6 to 60.1 ms: the 5.5 ms that left the host phase did not reappear in the sync, so this is a reduction in the total, not a shift between phases. The end-to-end rate agrees (36.90 to 40.07 tok/s; the default's range is wide, 33.29 to 39.03, the #755 bimodality, and `both`'s three runs all sit above its maximum). The token-identity column of the #1799 harness is omitted because Laguna's classic decode is not deterministic across its own runs on this host (#1799 recorded the same), so it cannot serve as a control here; the budget does not change arithmetic, only graph boundaries, and the batched-serving and single-stream tables compare rates only.

Graphs off is faster still on this arm (41.28 tok/s, drafter host build 7.7 ms, device sync 45.7 ms), as #1799 found at block 8: the multi-row verify gains nothing from capture. That is a property of the speculative path and is out of scope here. On the classic arm, which this issue is about, graphs off also beats Laguna's default (+11% here, the #1799 finding) but not `both` (32.57 against 33.80 through the CLI, 33.79 against 37.60 through `mlxcel-bench-decode`), and it costs 3 to 15% of classic decode on every other model measured (gpt-oss -3%, qwen3-30b-a3b -5%, Llama -6%, Qwen 3.5 4B -10%, gemma-4-26b-a4b -11%, qwen3.5-35b-a3b -15%), so it is not a default candidate either.

## Does the shipped default actually reach the raised budget? (idle host, n = 3, same binary)

Every table above was taken with the policy module compiled but not called, so every arm was MLX's own behaviour plus environment variables. That measures the knob, not the product. This section measures the product: the same binaries as the tables above, but with `apply_cuda_graph_budget_default` wired into `main` (`src/main.rs`, `src/bin/mlx_server.rs`, `src/bin/bench_decode.rs`, `src/bin/speculative_bench.rs`), run after a reboot on an idle host with the GPU held, three arms that differ only in what the operator sets:

- **`default`**: no `MLX_MAX_*` variable set at all, which is how a user runs it. If the policy works, this lands on 100 / 1000.
- **`restore`**: `MLX_MAX_OPS_PER_BUFFER=20 MLX_MAX_MB_PER_BUFFER=25`, the operator putting MLX's own cc-12.1 table values back. This is the stock-MLX control, and it also tests the env-wins contract in the direction that matters, an operator turning the default off.
- **`both`**: `MLX_MAX_OPS_PER_BUFFER=100 MLX_MAX_MB_PER_BUFFER=1000` set explicitly, the arm every table above calls `both`.

`laguna-xs-2.1-nvfp4` is in the allowlist; `qwen3.5-4b-4bit` (dense, `model_type` `qwen3_5`) is not and is the negative control: for it all three arms must be the same run.

### `laguna-xs-2.1-nvfp4` (allowlisted), `mlxcel-bench-decode`

| config | n | prompt tok | decode tok/s mean (min to max) | vs default | decode ms/200 tok | prefill ms mean (min to max) | vs default | prefill tok/s | MLX peak GB (min to max) | load1 (min to max) | CI job during run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| default | 3 | 112 | 37.42 (37.22 to 37.71) |  | 5345 | 406.8 (396.5 to 415.9) |  | 275 | 22.03 (22.03 to 22.04) | 0.90 to 1.09 | 1 of 3 |
| restore | 3 | 112 | 31.76 (31.09 to 32.31) | -15.1% | 6299 | 442.1 (439.7 to 443.9) | +8.7% | 253 | 21.66 (21.66 to 21.66) | 0.86 to 1.02 | 0 of 3 |
| both | 3 | 112 | 37.50 (37.24 to 37.64) | +0.2% | 5333 | 413.0 (402.5 to 423.5) | +1.5% | 271 | 22.03 (22.03 to 22.03) | 0.83 to 1.13 | 0 of 3 |

`default` and `both` overlap across their whole ranges (37.22 to 37.71 against 37.24 to 37.64) and both are disjoint from `restore` (31.09 to 32.31), which is +17.8% for the shipped default over stock MLX. The MLX peak column is an independent fingerprint of which budget was in force and does not depend on timing at all: 22.03 GB on `default` and `both`, 21.66 GB on `restore`, with no overlap in either direction. `restore` also reproduces the pre-wiring `default` row of the first table (31.76 against 32.02, 21.66 GB against 21.66 GB), which is the other half of the check: the operator can put the old behaviour back exactly.

### `laguna-xs-2.1-nvfp4` (allowlisted), `mlxcel generate`

The production CLI rather than the bench binary, 200 tokens through the chat template at `--temp 0`, same three arms, n = 3. This path has no same-process warm-up, so its rates sit below `mlxcel-bench-decode`'s, exactly as the DFlash table's `off` arm does.

| config | n | tok/s mean (min to max) | vs no env | load1 (min to max) |
|---|---|---|---|---|
| no env | 3 | 34.13 (33.96 to 34.27) |  | 0.71 to 0.82 |
| operator `20` / `25` | 3 | 29.29 (29.04 to 29.67) | -14.2% | 0.81 to 0.95 |
| operator `100` / `1000` | 3 | 33.81 (33.69 to 34.03) | -0.9% | 0.79 to 0.83 |

The same result on the shipped command: no env overlaps the explicit raised pair (33.96 to 34.27 against 33.69 to 34.03) and both are disjoint from the restored MLX values, whose 29.29 reproduces the pre-wiring CLI classic arm of the DFlash table (29.40). Generated token ids are not compared across arms here because Laguna's classic decode is not reproducible across its own repeats on this host, which the #1799 record measured first; the budget changes graph boundaries, not arithmetic.

### `qwen3.5-4b-4bit` (not allowlisted), `mlxcel-bench-decode`

| config | n | prompt tok | decode tok/s mean (min to max) | vs default | decode ms/200 tok | prefill ms mean (min to max) | vs default | prefill tok/s | MLX peak GB (min to max) | load1 (min to max) | CI job during run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| default | 3 | 111 | 66.82 (66.70 to 66.90) |  | 2993 | 104.2 (103.6 to 105.4) |  | 1065 | 2.85 (2.84 to 2.87) | 0.61 to 0.91 | 0 of 3 |
| restore | 3 | 111 | 66.36 (65.58 to 67.04) | -0.7% | 3014 | 106.1 (102.8 to 108.7) | +1.8% | 1046 | 2.85 (2.82 to 2.87) | 0.67 to 0.85 | 0 of 3 |

Overlapping ranges, identical peak memory, and both rows sit on the pre-wiring `default` row (67.38) rather than its `both` row (70.03). On the same device, in the same binary, one checkpoint moves and the other does not: what fires the default is the allowlist, not sm_121.

### `laguna-xs-2.1-nvfp4` (allowlisted), `mlxcel-server`

One server per (arm, round) at concurrency 1, 128-token prompts, 200 completion tokens, n = 3. Laguna's `supports_batching()` is false (`src/models/laguna.rs:650`), so this is a serialized workload through the serving stack rather than a B > 1 measurement, which is the point here: it exercises the third entry point, `src/bin/mlx_server.rs`.

| config | n | aggregate tok/s mean (min to max) | vs no env | per-request decode tok/s mean | TTFT ms mean | load1 (min to max) | CI job during run |
|---|---|---|---|---|---|---|---|
| no env | 3 | 29.03 (28.50 to 29.40) |  | 32.50 | 769 | 0.34 to 0.97 | 0 of 3 |
| operator `20` / `25` | 3 | 27.70 (27.20 to 28.20) | -4.6% | 30.63 | 724 | 0.42 to 0.88 | 0 of 3 |
| operator `100` / `1000` | 3 | 29.23 (29.00 to 29.60) | +0.7% | 32.40 | 696 | 0.60 to 0.96 | 2 of 3 |

The restored arm is disjoint from both of the others (27.20 to 28.20 against 28.50 to 29.40 and 29.00 to 29.60) and reproduces the pre-wiring `default` server row (26.90, 26.50 to 27.30). Per-request decode says the same with the server's prefill and queueing removed: 32.50 and 32.40 against 30.63.

Taken together the three entry points answer the question this section exists for. The shipped binaries reach the raised budget on an allowlisted checkpoint, they leave an unlisted one alone, an operator can restore MLX's values and gets MLX's numbers back, and the budget actually in force is printed at startup rather than inferred.

### The applied budget at startup

The same three arms through `mlxcel generate` and `mlxcel-server`, reading the startup lines rather than the clock (full transcript in `data/.../sweep7.out`):

| binary | checkpoint | operator sets | startup line |
|---|---|---|---|
| `mlxcel generate` | laguna | nothing | `CUDA graph budget: MLX_MAX_OPS_PER_BUFFER=100 MLX_MAX_MB_PER_BUFFER=1000 applied for model_type laguna on sm_121 (#1798); an operator-set value wins per variable` |
| `mlxcel generate` | laguna | `MLX_MAX_OPS_PER_BUFFER=20` | `CUDA graph budget: MLX_MAX_MB_PER_BUFFER=1000 applied for model_type laguna on sm_121 (#1798); ...` |
| `mlxcel generate` | laguna | both | none |
| `mlxcel generate` | qwen3.5-4b | nothing, ops only, both | none in any of the three |
| `mlxcel-server` | laguna | nothing | `INFO mlxcel::server::startup: CUDA graph budget: MLX_MAX_OPS_PER_BUFFER=100 MLX_MAX_MB_PER_BUFFER=1000 applied for model_type laguna on sm_121 (#1798); ...` with the `AppliedCudaGraphBudget` record attached |
| `mlxcel-server` | qwen3.5-4b | nothing | none |

The second row is the per-variable half of the env-wins contract: with only the op budget set by hand, the policy fills in the byte budget and leaves the op budget alone. The third and fourth rows are the whole-variable half, and the last two are the same allowlist gate on the serving path.

## Batched serving (`mlxcel-server --max-batch-size 8`, concurrency 1, 4, 8; idle host, same binary)

One server per (arm, round); the concurrency-1 level is also the server's warm-up. Laguna's `supports_batching()` returns false (`src/models/laguna.rs:650`, its mixed full and sliding caches are not per-sequence isolated), so on Laguna the scheduler serializes concurrent requests: its aggregate stays at the single-stream rate at every level and TTFT grows with the queue (11 s at 4, 26 s at 8). Those rows are therefore a serialized single-stream workload through the server, not a B > 1 measurement; the B > 1 rows are qwen3-30b-a3b, which does batch (per-request decode 79 to 21 to 9 tok/s as the batch fills while the aggregate rises). **The qwen3-30b-a3b arms below are the partial first pass** (default n = 1, `both` n = 2, `nograph` n = 1): the sweep was stopped by the host-protection halt described below, mid round 1, before its third round. They are kept as recorded; the completed n = 3 re-run on a freshly booted host is the section after this one, and it is what the allowlist decision rests on.


### `laguna-xs-2.1-nvfp4`

| concurrency | config | n | aggregate tok/s mean (min to max) | vs default | per-request decode tok/s mean | TTFT ms mean (p95 mean) | load1 (min to max) | CI job during run |
|---|---|---|---|---|---|---|---|---|
| 1 | default | 3 | 26.90 (26.50 to 27.30) |  | 30.30 | 857 (857) | 0.38 to 0.95 | 0 of 3 |
| 4 | default | 3 | 28.13 (27.70 to 28.50) |  | 30.67 | 11270 (21910) | 0.38 to 0.95 | 0 of 3 |
| 8 | default | 3 | 27.67 (27.10 to 28.10) |  | 30.13 | 25969 (51273) | 0.38 to 0.95 | 0 of 3 |
| 1 | both | 3 | 29.10 (29.00 to 29.30) | +8.2% | 32.23 | 704 (704) | 0.56 to 0.65 | 0 of 3 |
| 4 | both | 3 | 29.67 (29.40 to 29.90) | +5.5% | 32.47 | 10673 (20758) | 0.56 to 0.65 | 0 of 3 |
| 8 | both | 3 | 28.83 (28.60 to 29.20) | +4.2% | 31.47 | 24761 (49101) | 0.56 to 0.65 | 0 of 3 |
| 1 | nograph | 3 | 29.40 (29.20 to 29.60) | +9.3% | 31.17 | 418 (418) | 0.82 to 1.13 | 0 of 3 |
| 4 | nograph | 3 | 29.47 (28.80 to 30.30) | +4.7% | 30.60 | 10368 (20586) | 0.82 to 1.13 | 0 of 3 |
| 8 | nograph | 3 | 29.10 (28.30 to 29.90) | +5.2% | 30.20 | 24318 (48426) | 0.82 to 1.13 | 0 of 3 |

### `qwen3-30b-a3b-4bit`

| concurrency | config | n | aggregate tok/s mean (min to max) | vs default | per-request decode tok/s mean | TTFT ms mean (p95 mean) | load1 (min to max) | CI job during run |
|---|---|---|---|---|---|---|---|---|
| 1 | default | 1 | 52.60 (52.60 to 52.60) |  | 78.90 | 1282 (1282) | 0.84 to 0.84 | 0 of 1 |
| 4 | default | 1 | 66.10 (66.10 to 66.10) |  | 20.60 | 2303 (3066) | 0.84 to 0.84 | 0 of 1 |
| 8 | default | 1 | 64.00 (64.00 to 64.00) |  | 8.80 | 2083 (4131) | 0.84 to 0.84 | 0 of 1 |
| 1 | both | 2 | 49.20 (46.10 to 52.30) | -6.5% | 80.15 | 1598 (1598) | 0.63 to 0.83 | 0 of 2 |
| 4 | both | 2 | 64.65 (62.80 to 66.50) | -2.2% | 20.55 | 2517 (3347) | 0.63 to 0.83 | 0 of 2 |
| 8 | both | 2 | 59.50 (58.50 to 60.50) | -7.0% | 9.15 | 4810 (6406) | 0.63 to 0.83 | 0 of 2 |
| 1 | nograph | 1 | 48.20 (48.20 to 48.20) | -8.4% | 67.00 | 1181 (1181) | 0.78 to 0.78 | 0 of 1 |
| 4 | nograph | 1 | 73.60 (73.60 to 73.60) | +11.3% | 23.00 | 2079 (2766) | 0.78 to 0.78 | 0 of 1 |
| 8 | nograph | 1 | 79.70 (79.70 to 79.70) | +24.5% | 11.70 | 2869 (4567) | 0.78 to 0.78 | 0 of 1 |

What the completed Laguna rows say: `both` is +8.2% at concurrency 1 (29.0 to 29.3 against 26.5 to 27.3, disjoint) and +4 to +6% at 4 and 8, the same sign as every other Laguna workload, smaller than the same-process harness because each server request pays its own prefill and first-token cost inside the aggregate. What the partial qwen3-30b-a3b rows say, with the caveat that none has n = 3: at concurrency 1 through the server `both` is 46.1 to 52.3 against a single default run of 52.6, and at 4 and 8 it is -2% and -7% against single default runs, while graphs off is +11% and +25% there. That is the opposite sign to this checkpoint's +21% single-stream result, and it is the reason qwen3_moe is not in the shipped allowlist: the production path is the server, batched decode changes the graph set every step as the batch composition changes, and larger graphs appear to pay more for that, as Llama did. Settling it needs the third round and an nsys pair on the batched path, neither of which ran.

## Batched serving, completed rounds: do the Qwen MoE families clear the bar? (idle host, n = 3, same binary)

`qwen3_moe` and `qwen3_5_moe` were the two families held out of the allowlist. Neither was held out for losing: both gained about +21% on single-stream decode with disjoint ranges. `qwen3_moe` was held out because its batched-serving rows ran the other way at n = 1 to 2 before the host-protection halt, and `qwen3_5_moe` because its server path had never been measured at all. Both are settled here, on a freshly booted host, n = 3 per arm, one server per (arm, round).

Unlike Laguna these two do batch, so these are B > 1 rows: per-request decode falls as the batch fills while the aggregate rises.

### `qwen3-30b-a3b-4bit` (`model_type` `qwen3_moe`)

| concurrency | config | n | aggregate tok/s mean (min to max) | vs default | per-request decode tok/s mean | TTFT ms mean (p95 mean) | load1 (min to max) | CI job during run |
|---|---|---|---|---|---|---|---|---|
| 1 | default | 3 | 46.17 (44.70 to 48.00) |  | 65.87 | 1311 (1311) | 0.33 to 0.63 | 0 of 3 |
| 4 | default | 3 | 66.87 (65.30 to 68.60) |  | 20.97 | 2317 (3084) | 0.33 to 0.63 | 0 of 3 |
| 8 | default | 3 | 61.47 (60.30 to 62.20) |  | 9.00 | 3633 (5397) | 0.33 to 0.63 | 0 of 3 |
| 1 | both | 3 | 52.00 (51.60 to 52.60) | +12.6% | 80.00 | 1357 (1357) | 0.66 to 0.82 | 0 of 3 |
| 4 | both | 3 | 67.43 (67.10 to 68.00) | +0.8% | 21.37 | 2392 (3180) | 0.66 to 0.82 | 0 of 3 |
| 8 | both | 3 | 59.97 (57.30 to 61.60) | -2.4% | 8.97 | 4205 (5887) | 0.66 to 0.82 | 0 of 3 |
| 1 | nograph | 3 | 48.43 (47.90 to 48.80) | +4.9% | 67.43 | 1181 (1181) | 0.71 to 1.71 | 0 of 3 |
| 4 | nograph | 3 | 73.53 (72.10 to 75.30) | +10.0% | 22.97 | 2087 (2777) | 0.71 to 1.71 | 0 of 3 |
| 8 | nograph | 3 | 81.40 (78.00 to 85.90) | +32.4% | 12.07 | 2938 (4562) | 0.71 to 1.71 | 0 of 3 |

At concurrency 1, where the server is doing a single-stream workload, `both` reproduces the single-stream finding: +12.6% with disjoint ranges (51.60 to 52.60 against 44.70 to 48.00), and per-request decode 80.0 against 65.9. At 4 and 8, which is what `--max-batch-size 8` exists for, it does nothing: +0.8% and -2.4% with overlapping ranges in both. The first pass's apparent regression at 4 and 8 was therefore mostly its n of 1 to 2; the honest n = 3 reading is no effect, not a loss. Either way it is not a gain, so the bar is not met.

The largest number in this table belongs to neither arm of this issue's question. Graphs off is +32.4% at concurrency 8 with disjoint ranges (78.00 to 85.90 against 60.30 to 62.20), +10.0% at 4 and +4.9% at 1, and it beats `both` at every level. On batched MoE decode the cost is capture itself, not the budget, which is the same shape the prefill rows showed and is filed as a follow-up rather than decided here.

### `qwen3.5-35b-a3b-4bit` (`model_type` `qwen3_5_moe`)

This family's server path had never been measured; its +22.5% came from single-stream decode alone.

| concurrency | config | n | aggregate tok/s mean (min to max) | vs default | per-request decode tok/s mean | TTFT ms mean (p95 mean) | load1 (min to max) | CI job during run |
|---|---|---|---|---|---|---|---|---|
| 1 | default | 3 | 42.17 (40.00 to 43.80) |  | 55.73 | 1171 (1171) | 0.44 to 1.26 | 0 of 3 |
| 4 | default | 3 | 61.77 (56.90 to 66.60) |  | 15.63 | 221 (348) | 0.44 to 1.26 | 0 of 3 |
| 8 | default | 3 | 56.57 (55.20 to 57.70) |  | 7.17 | 399 (707) | 0.44 to 1.26 | 0 of 3 |
| 1 | both | 3 | 42.50 (41.90 to 43.00) | +0.8% | 56.23 | 1166 (1166) | 0.86 to 1.03 | 0 of 3 |
| 4 | both | 3 | 63.77 (61.70 to 66.10) | +3.2% | 16.20 | 277 (418) | 0.86 to 1.03 | 0 of 3 |
| 8 | both | 3 | 50.50 (49.80 to 51.10) | -10.7% | 6.40 | 446 (785) | 0.86 to 1.03 | 0 of 3 |
| 1 | nograph | 3 | 39.87 (39.20 to 41.00) | -5.5% | 48.27 | 895 (895) | 0.87 to 1.18 | 0 of 3 |
| 4 | nograph | 3 | 56.60 (56.00 to 57.40) | -8.4% | 14.30 | 213 (338) | 0.87 to 1.18 | 0 of 3 |
| 8 | nograph | 3 | 51.80 (51.30 to 52.30) | -8.4% | 6.53 | 375 (669) | 0.87 to 1.18 | 0 of 3 |

This is the clearer of the two, and it goes the wrong way. At concurrency 1 and 4 the raised budgets do nothing (+0.8% and +3.2%, both overlapping). At concurrency 8 they cost 10.7% with disjoint ranges (49.80 to 51.10 against 55.20 to 57.70). A family whose single-stream decode gains 22.5% with disjoint ranges loses 10.7% with disjoint ranges on the path a server actually runs. Note also that graphs off, which was the big winner on qwen3-30b-a3b's batched rows, is uniformly worse here (-5.5%, -8.4%, -8.4%): even within one architecture family the sign of a capture-policy change does not carry across checkpoints.

### Long prefill and its memory cost on the two Qwen MoE families

2048-token prompt (one prefill chunk at the default `MLXCEL_PREFILL_CHUNK`), 16 decode tokens, n = 3. This is the workload that priced Laguna's default at +7 GB per chunk, and it is the fourth measured workload for these two families.

| model | config | n | decode tok/s mean (min to max) | vs default | prefill ms mean (min to max) | vs default | MLX peak GB | load1 (min to max) |
|---|---|---|---|---|---|---|---|---|
| qwen3-30b-a3b | default | 3 | 73.78 (71.22 to 78.46) |  | 1835.8 (1816.2 to 1861.9) |  | 20.59 | 0.40 to 0.76 |
| qwen3-30b-a3b | both | 3 | 83.46 (79.47 to 85.93) | +13.1% | 2099.8 (2036.2 to 2139.0) | +14.4% | 36.91 | 0.52 to 1.29 |
| qwen3.5-35b-a3b | default | 3 | 54.87 (52.53 to 57.44) |  | 3175.4 (3161.9 to 3191.5) |  | 27.74 | 1.12 to 1.60 |
| qwen3.5-35b-a3b | both | 3 | 63.11 (58.74 to 65.57) | +15.0% | 3428.6 (3407.1 to 3443.7) | +8.0% | 35.91 | 1.20 to 1.42 |

Decode after the long prompt gains on both, and prefill itself is slower on both (+14.4% and +8.0%, disjoint ranges in both cases). The memory is the bigger number: +16.3 GB on qwen3-30b-a3b (20.59 to 36.91) and +8.2 GB on qwen3.5-35b-a3b (27.74 to 35.91), against Laguna's +7.0 GB. Peak memory is exactly reproducible across all three repeats on every arm here, so these are not noise. That is a third measured workload on which neither family gains, independent of the serving result.

### The verdict on both

Neither family joins the allowlist, and the allowlist stays `["laguna"]`.

- `qwen3_moe`: +20.6% single-stream and +12.6% at concurrency 1, no measurable effect at 4 or 8, prefill 14.4% slower at 2048 tokens, and +16.3 GB of peak memory there. Not a regression on the serving path, but not a gain either, and the bar is that every measured workload gains.
- `qwen3_5_moe`: +22.5% single-stream, nothing at concurrency 1 and 4, a disjoint -10.7% at concurrency 8, prefill 8.0% slower and +8.2 GB at 2048 tokens. This one would have been an outright regression on a batched server.

The general lesson for the next family considered is in `qwen3_5_moe`: single-stream decode and batched serving disagreed in sign on the same checkpoint, with disjoint ranges on both sides. Any future addition has to clear the serving path at more than one concurrency level, not only `mlxcel-bench-decode`.

## Chain status: the host-protection halt, and what the reboot completed

The GB10 driver began shedding `NVRM: NV_ERR_NO_MEMORY` allocation errors at an accelerating rate during the batched phase (486 by 02:51, 234 of them in two minutes), the documented precursor of a kernel wedge on this host, and every remaining measurement was stopped and the GPU lock released. What completed, all n = 3 unless stated:

- Single-stream decode and short prefill, six arms: Laguna, Qwen 3.5 4B, Llama 3.1 8B. Complete.
- Single-stream decode and short prefill, three arms: gpt-oss-20b, qwen3-30b-a3b, qwen3.5-35b-a3b, gemma-4-26b-a4b. Complete.
- Long prefill (2048 tokens), three arms: Laguna, Qwen 3.5 4B. Complete.
- nsys graph accounting, default and `both`: Laguna, Qwen 3.5 4B, Llama 3.1 8B. Complete (one profile pair per arm by design).
- DFlash phase split at block 4, six arms: Laguna. Complete (the gated re-run; the first pass overlapped a CI job and is kept in `data/` as `results_dflash.jsonl` for reference only).
- Batched serving, three arms at concurrency 1, 4, 8: Laguna complete; qwen3-30b-a3b partial (default 1, `both` 2, `nograph` 1 rounds).
- Not started at the halt: intermediate budgets (ops 50 / mb 1000, ops 100 / mb 100) and the 8192-token prefill peak on Laguna (chain 5); qwen3.6-35b-a3b single-stream and the 2048-token prefill peak on qwen3-30b-a3b and qwen3.5-35b-a3b (chain 6).

The host was rebooted, which cleared the driver's error counter to 0, and the chain resumed. Completed after the reboot, all n = 3, `NVRM: NV_ERR_NO_MEMORY` still 0 at the end:

- The default-binary A/B on the wired binary: `mlxcel-bench-decode`, `mlxcel generate` and `mlxcel-server` on Laguna, plus the dense negative control and the startup-line table.
- Batched serving on qwen3-30b-a3b, all three rounds of all three arms, superseding the partial first pass.
- Batched serving on qwen3.5-35b-a3b, which had never been measured.
- The 2048-token prefill peak on both Qwen MoE checkpoints.

Still not run, and not needed for the decision this record supports: the intermediate budgets, the 8192-token Laguna prefill peak, qwen3.6-35b-a3b single-stream, and an nsys pair on the batched path. The last of those would explain why the raised budgets stop helping as a batch fills; it is a mechanism question for the capture-policy follow-up, not a gate on the allowlist, which the throughput rows already settle in the negative for both families. The Metal and Accelerate test gates were not run at all: this is a Linux CUDA host with neither backend.

Raw records for every run, the harness scripts and the sweep logs are in `docs/benchmark_results/data/cuda-graph-budget-gb10-2026-09-12/`.

