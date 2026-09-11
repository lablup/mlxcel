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

