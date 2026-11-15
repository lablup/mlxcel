# Causal-mode prefill for the hybrid and Qwen-VL families (issue #3051), M3 Ultra and M5 Max, 2026-11-15

Issue #3051: the full-attention layers of Qwen 3.5, Qwen3-Next, GraniteMoeHybrid and the Qwen-VL models built an explicit f32 causal array mask for every prefill. Before M5, MLX fuses SDPA in causal mode at head_dim 256 but not with an array mask, so a single-pass prefill took the query-chunked fallback from #3049 and kept the 3.65 GiB mask alive for the whole pass at 28K tokens.

`mlxcel_core::utils::prefill_causal_mask` now leaves causality to MLX's causal mode when that is equivalent to the array mask (more than one query, one row, no caller mask, not a speculative verify block) and MLX's own verdict (`sdpa_mode_is_fused`, memoized per shape) says causal mode fuses at the call's shape. Elsewhere the model keeps its single array mask, shared by every full-attention layer, which is the earlier dispatch. On the M3 Ultra causal mode fuses for a cold prefill and for the first piece of a chunked one. It does not fuse for later pieces (head_dim 256 over `q_len != k_len`), so those keep the array mask.

Host: Apple M3 Ultra, 80-core GPU, 512 GB, pre-M5. One release build of branch `fix/issue-3051-causal-mode` (`metal,accelerate`) runs both arms. The base arm sets `MLXCEL_PREFILL_CAUSAL_MODE=0`, which restores the array mask everywhere; the new arm uses the default. Generation is greedy, 32 tokens, reasoning included in the comparison. Prompts come from `scripts/bench/metal_sdpa_chunk/make_prompt.py`. Everything ran in one GPU lock hold, with the arms interleaved A/B then B/A (`scripts/bench/prefill_peak/ab_prefill.sh` with `BASE_ENV`/`NEW_ENV`). Load average was 0.3 to 5.4 and is recorded per row in the CSVs.

The issue names `scripts/bench/metal_sdpa_chunk/verify.sh`, but that script's arms toggle `MLXCEL_METAL_ATTENTION_CHUNK`, not the mask, so the interleaved harness was used. Its `gated_calls` column counts the `query-chunking` debug lines of a CLI run, one per full-attention layer whose SDPA takes the materializing fallback. The server cells do not emit that log, so they read 0 in both arms.

## Results

28000-token prompt (28707 tokens after the chat template), means of two repetitions per arm (`ab-summary-28k.csv`):

| Path | Model | Peak, base to new | Prefill / TTFT, base to new | Gated calls |
|---|---|---|---|---|
| CLI single pass | qwen3.5-4b-4bit | 30.13 to **19.82 GB** (-34%) | 16.89 to **13.38 s** (-20.8%) | 8 to 0 |
| CLI single pass | qwen3.8-27b-4bit | 64.20 to **53.96 GB** (-16%) | 86.09 to **73.33 s** (-14.8%) | 16 to 0 |
| CLI 2048-token chunks | qwen3.5-4b-4bit | 8.97 to 8.97 GB | 13.87 to 13.85 s | 96 to 96 |
| CLI 2048-token chunks | qwen3.8-27b-4bit | 25.80 to 25.80 GB | 74.77 to 74.74 s | 208 to 208 |
| server, `--parallel 1` | qwen3.5-4b-4bit | 9.10 to 9.10 GB | 13.90 to 13.89 s | - |
| server, `--parallel 1` | qwen3.8-27b-4bit | 25.92 to 25.92 GB | 74.88 to 74.81 s | - |

4600-token prompt with 2048-token chunks and `MLXCEL_ATTENTION_CHUNK_BUDGET_MB=16`, so that the fallback gate fires at this length (`ab-summary-4600.csv`):

| Model | Peak, base to new | Prefill, base to new | Gated calls |
|---|---|---|---|
| qwen3.5-4b-4bit | 5.34 to 5.34 GB | 1776.8 to 1763.4 ms (-0.8%) | 24 to 16 |
| qwen3.8-27b-4bit | 19.85 to 19.85 GB | 10714.4 to 10664.8 ms (-0.5%) | 48 to 32 |

Repetitions agree within 57 ms at 28K and within 12 ms at 4.6K. Chunked peaks agree within 1 MB between arms.

Generated text is byte-identical between arms in every cell, for CLI and server. An image prompt on qwen3.5-4b-4bit (`tests/fixtures/test_image_shapes.png`, `--show-reasoning`, 48 tokens) is also identical in both repetitions.

The single pass removes every gated call and the 3.65 GiB mask. In a chunked prefill only the first piece changes, so at 4.6K one of the three gated pieces is gone. At 28K the first 2048-token piece is under the score budget in both arms, so the count does not move. These later pieces are the remaining gap, and on this hardware MLX falls back for them whatever mask they are given.

## Alternatives measured and rejected

Two other ways to handle the pieces after the first were measured at 28K, in the same interleaved setup (separate lock holds, CSVs kept):

| Variant | qwen3.5-4b-4bit chunked peak | qwen3.8-27b-4bit chunked peak |
|---|---|---|
| Causal mode everywhere; a falling-back call runs `chunked_causal_attention`, which slices K/V and builds a mask per query chunk (`rejected-chunked-causal-28k.csv`) | 8.95 to 9.19 GB (+2.6%) | 25.79 to 26.82 GB (+4.0%) |
| Causal mode everywhere; a falling-back call builds `create_causal_mask` itself and runs the array-mask chunker (`rejected-per-layer-mask-28k.csv`) | 8.95 to 9.01 GB (+0.6%) | 25.78 to 28.72 GB (+11.4%) |

Both variants regress the server's default chunked path. The per-layer mask also costs one f32 `[2048, 30K]` array per full-attention layer instead of one per forward. Asking MLX first, and keeping the shared mask where causal mode would fall back anyway, is the only variant tried that does not raise any peak.

## M5 Max

Host: Apple M5 Max, 128 GB, macOS 27.2.1, MLX pin `9f806b36`. Unlike the M3 Ultra run, the two arms are two release builds (`metal,accelerate`): base is `origin/main` at `49ea6cc9`, new is this branch rebased onto it. Same harness and prompts, greedy, 32 tokens, with `COOLDOWN=75` (a 75 s pause before every run) because back-to-back runs on this host throttle. The arms alternate A/B then B/A, everything in one GPU lock hold. Load average was 1.0 to 9.2. The highest readings came from a test build on the CPU during the qwen3.5-4b-4bit 28K cells and are recorded per row. The server cells were not run here. CSVs: `m5max-ab-summary-28k.csv`, `m5max-ab-summary-4600.csv`, `m5max-ab-summary-4600-rerun.csv`.

On M5, MLX fuses head_dim 256 SDPA with an array mask too, so main already logs 0 gated calls at 28K. Causal mode still avoids the f32 mask and its fused kernel's working set. The 2048-row pieces behind a cache (`q_len != k_len`) take causal mode as well: `prefill_causal_mask` drops the mask only when MLX reports causal mode fused at the call's shape, and the chunked peaks fall in both models. On the M3 Ultra only the first piece did.

28000-token prompt, means of two repetitions per arm:

| Path | Model | Peak, main to branch | Prefill, main to branch | Gated calls |
|---|---|---|---|---|
| CLI single pass | qwen3.5-4b-4bit | 14.53 to **8.76 GB** (-39.7%) | 12.71 to **10.09 s** (-20.6%) | 0 to 0 |
| CLI single pass | qwen3.8-27b-4bit | 31.83 to **26.03 GB** (-18.2%) | 57.65 to **44.63 s** (-22.6%) | 0 to 0 |
| CLI 2048-token chunks | qwen3.5-4b-4bit | 6.06 to **5.70 GB** (-6.0%) | 7.78 to 7.67 s (-1.4%) | 0 to 0 |
| CLI 2048-token chunks | qwen3.8-27b-4bit | 20.69 to **20.34 GB** (-1.7%) | 40.75 to 39.95 s (-2.0%) | 0 to 0 |

The single-pass main repetitions spread widely (qwen3.5-4b-4bit 13.56 and 11.86 s, qwen3.8-27b-4bit 63.67 and 51.63 s), but both branch repetitions (9.63 and 10.54 s; 44.94 and 44.33 s) are below both main repetitions in each cell. Branch peaks agree within 22 MB between repetitions, main peaks within 1 MB.

4600-token prompt with 2048-token chunks and `MLXCEL_ATTENTION_CHUNK_BUDGET_MB=16`:

| Model | Peak, main to branch | Prefill, main to branch | Gated calls |
|---|---|---|---|
| qwen3.5-4b-4bit | 4.13 to 4.07 GB (-1.5%) | 819.2 to 810.9 ms (-1.0%) | 8 to 8 |
| qwen3.8-27b-4bit (four repetitions, rerun) | 17.15 to 17.10 GB (-0.3%) | 4529.5 to 4398.3 ms (-2.9%) | 16 to 16 |

In the first qwen3.8-27b-4bit pass, the branch's first repetition read 8550.7 ms against 4758.7 ms for its second, with main at 4391.8 and 4705.8 ms. That cell was rerun with four repetitions per arm. Every rerun branch repetition (4379.8 to 4415.0 ms) is below every main repetition (4486.0 to 4593.7 ms), so the slow reading was a one-off. Both passes are in the CSVs.

The gated calls in the 4.6K cells are all on the last piece, `q=[1, 16, 452, 256]` over 4548 keys for qwen3.5-4b-4bit and `q=[1, 24, 494, 256]` over 4590 keys for qwen3.8-27b-4bit: MLX reports causal mode unfused at that shape too, so `prefill_causal_mask` keeps the shared array mask and the piece takes the same chunked fallback in both arms. This short tail is the remaining M5 gap the issue names.

Greedy output is byte-identical between main and the branch in every cell and every repetition, and also between the two 4.6K passes.

Op probe at the issue's shape (`sdpa_chunk_peak_memory_probe`, `[1,24,28000,256]` fp16, 4 KV heads, `test-fast` profile):

| Mode | MLX fused | Single call peak / time | Gated peak / time |
|---|---|---|---|
| array mask | yes | 5.51 GB, 557 ms | 5.85 GB, 562 ms |
| causal mode | yes | 3.94 GB, 245 ms | 4.28 GB, 235 ms |

Both modes return `max_abs_diff=0` against their gated variant. These match the issue's earlier M5 Max reading (array mask 545 to 668 ms and 5.51 GB, causal mode 241 to 377 ms and 3.94 GB).

## Rerun

```
BASE_BIN=target/release/mlxcel NEW_BIN=target/release/mlxcel MODELS_DIR=models \
  BASE_ENV=MLXCEL_PREFILL_CAUSAL_MODE=0 NEW_ENV=MLXCEL_PREFILL_CAUSAL_MODE=1 \
  TOKENS=28000 scripts/bench/prefill_peak/ab_prefill.sh <out_dir>
BASE_BIN=target/release/mlxcel NEW_BIN=target/release/mlxcel MODELS_DIR=models \
  BASE_ENV=MLXCEL_PREFILL_CAUSAL_MODE=0 NEW_ENV=MLXCEL_PREFILL_CAUSAL_MODE=1 \
  TOKENS=4600 CLI_CELLS="qwen3.5-4b-4bit:2048 qwen3.8-27b-4bit:2048" SERVER_MODELS="" \
  EXTRA_ENV=MLXCEL_ATTENTION_CHUNK_BUDGET_MB=16 scripts/bench/prefill_peak/ab_prefill.sh <out_dir>
```

The M5 Max run used a main build and a branch build instead, with a cooldown:

```
BASE_BIN=<main build> NEW_BIN=<branch build> MODELS_DIR=models COOLDOWN=75 SERVER_MODELS="" \
  TOKENS=28000 scripts/bench/prefill_peak/ab_prefill.sh <out_dir>
BASE_BIN=<main build> NEW_BIN=<branch build> MODELS_DIR=models COOLDOWN=75 SERVER_MODELS="" \
  TOKENS=4600 CLI_CELLS="qwen3.5-4b-4bit:2048 qwen3.8-27b-4bit:2048" \
  EXTRA_ENV=MLXCEL_ATTENTION_CHUNK_BUDGET_MB=16 scripts/bench/prefill_peak/ab_prefill.sh <out_dir>
```
