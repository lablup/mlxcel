# CUDA `QuantizedMatmul` batch fold on RTX Spark N1X

Issue #2690. Batched decode passes `QuantizedMatmul` an `x = [B, 1, K]` against one shared 2D weight. CUDA ran that as `B` independent `M == 1` problems and re-read the weight per batch entry. Metal and ROCm already fold this shape into rows.

The fold gives 1.2x to 3.2x aggregate at 8 clients. Weights of at least 2^24 elements also move to a GEMM kernel from 5 rows on compute capability 12.1, which gives up to 1.5x at 5 clients on models with large MLP weights. Gemma 4 26B loses 2-5% at 5 to 7 clients. Its MoE weight traffic is not addressed here.

## Method

- Host: RTX Spark N1X (sm_121), Windows 11 arm64, WDDM, CUDA 13.4, driver 617.74, ASUS Performance mode on AC.
- `scripts/bench_serving_concurrency.py` sends one synthetic prompt to every client, so rows repeat. Shared-weight reads do not depend on prompt content, but this does not represent MoE expert overlap.
- Same binary on every arm, arms set by environment. `scripts/bench_serving_concurrency.py`, 128 prompt tokens, 200 output tokens, `--max-batch-size 8`, `--ignore-eos`, one server per arm and round, arm order rotated per round.
- Three rounds unless noted, with the weights read into the page cache first. A run with a busy clock median below 1700 MHz was repeated.
- The tables report the median and range of aggregate tok/s.

This host caps the GPU to 400-700 MHz when idle, so a run without clock gating is not usable here. An earlier pass that accepted 1000 MHz recorded Llama 3.1 8B at 13.5 tok/s against about 50 at one client. Those rows are excluded.

## Shipped behavior against no fold

Final rule against `MLXCEL_QMM_FOLD_BATCH=0`.

| Model | Arm | 1 client | 4 clients | 5 clients | 8 clients |
|---|---|---|---:|---:|---:|
| qwen3.8-27b-4bit | shipped | 13.4 | 24.5 | 34.9 (34.7-34.9) | 45.0 (44.6-45.4) |
| | no fold | 13.3 | 24.5 | 23.3 (23.2-23.4) | 14.0 (14.0-14.1) |
| llama-3.1-8b-4bit | shipped | 45.2 | 54.2 | 59.9 (59.5-60.5) | 75.6 (75.5-75.9) |
| | no fold | 46.6 | 54.6 | 47.0 (46.7-47.6) | 33.7 (33.6-33.9) |
| gpt-oss-20b-mxfp4 | shipped | 47.2 | 92.1 | 89.6 (88.4-90.3) | 87.9 (86.2-88.3) |
| | no fold | 48.8 | 92.0 | 86.0 (83.8-86.8) | 72.1 (71.7-72.2) |
| gemma-4-26b-a4b-it-4bit | shipped | 31.4 | 38.0 | 35.7 (34.1-35.8) | 34.1 (32.7-34.4) |
| | no fold | 31.1 | 36.9 | 37.0 (36.9-37.8) | 16.9 (16.6-17.1) |

One client and four clients run the same kernels on both arms, and their medians differ by about 3% at most.

## Why the threshold depends on size

qmv at 5 to 7 rows takes the 8-wide multirow instantiation and slows down, so GEMM wins there on a large weight. Each cell is the median at 5, 6 and 7 clients.

| Model | qmv | GEMM for weights of at least 2^24 elements | GEMM for every weight |
|---|---|---|---|
| qwen3.8-27b-4bit | 22.6 / 22.9 / 22.4 | 33.4 / 39.8 / 45.6 | 33.3 / 39.7 / 44.5 |
| llama-3.1-8b-4bit | 45.9 / 47.1 / 47.1 | 56.1 / 65.2 / 71.9 | 56.8 / 66.2 / 72.1 |
| gpt-oss-20b-mxfp4 | 81.5 / 86.6 / 86.9 | 82.9 / 93.6 / 92.2 | 86.5 / 95.4 / 94.8 |
| gemma-4-26b-a4b-it-4bit (5 rounds) | 34.6 / 36.4 / 39.6 | 33.0 / 35.4 / 38.9 | 32.1 / 34.2 / 37.6 |

The size gate halves the Gemma 4 loss and costs gpt-oss 2-4%. The cause of the remaining Gemma 4 loss is not established. Its noise is large: in a separate 5-round comparison, 4 clients ran the same kernels on both arms and still differed by about 6%.

GEMM at fewer than 5 rows was rejected. Against qmv, at 4 clients it gains 18% on qwen3.8, 7% on Llama and 3% on gpt-oss, but loses 10-19% on Gemma 4. At 2 clients it loses on all four, and at 3 it loses on Gemma 4 and ties elsewhere.

Keeping small weights on qmv through 8 rows instead of GEMM moved 8-client aggregate +0.5% to +3% on the four models, with no loss.

## Accuracy

`examples/logit_trace`, `docs/architecture.md`, `64 4 8 512`, `MLXCEL_TRACE_BATCH_ROWS=1` against `=6`, 256 positions. This used the GEMM threshold of 5 without the size gate, which moves more weights to GEMM than the shipped rule. Decided positions use a top-two gap of at least 2.0.

| Model | Arm at 6 rows | Top-1 disagreements | On decided positions | Perplexity change |
|---|---|---|---|---|
| llama-3.1-8b-4bit | GEMM | 0 | 0 / 90 | -0.027% |
| qwen3.8-27b-4bit | GEMM | 5 (largest gap 0.125) | 0 / 119 | +0.014% |
| qwen3.8-27b-4bit | qmv, or no fold | 0 | 0 / 119 | 0 |
| gemma-4-26b-a4b-it-4bit | GEMM | 29 | 0 / 140 | -9.9% |
| gemma-4-26b-a4b-it-4bit | qmv, or no fold | 27 | 1 / 140 | -2.1% |

GEMM changes results only in the rounding class on the two non-MoE models. Gemma 4 26B diverges from B=1 with or without this change, as `gemma4-native-batched-decode-2026-10-26.md` records.

Verify blocks and prefill pieces (`M > 1`) keep the threshold of 8, so multirow qmv's bit identity for speculative verify is untouched.

## Gemma 4 26B at 8 clients

Aggregate stays flat at 34-40 tok/s from 4 to 8 clients. `examples/profile_batched_decode` gives 21 ms per step at B=1 and 152 ms at B=8, against 176 ms for 8 sequential forwards, so batching saves 15%.

With `MLXCEL_PROFILE_MOE_INNER=1`, which forces an eval per operation and inflates absolute times, a layer at B=8 takes 10.2 ms. The gate, up and down `gather_qmm` take about 5.8 ms of it. Everything before the MoE gather, attention included, takes at most 1.9 ms. That profile repeats one prompt across rows; with 8 different prompts these are about 7.1 of 13.2 ms and about 4.8 ms ([results](moe-distinct-experts-n1x-2026-10-31.md)).

Each of those operations reads `top_k * B` expert matrices, so the traffic grows with B and the fold cannot reach it. Grouping rows by expert would need a new kernel, and #2784 measured its upper bound at about 13% aggregate at 8 clients. The existing `MLXCEL_GATHER_QMM_GROUPED` path requires `8 * E` pairs (1024 here) against 64 at 8 clients.

`examples/qmm_fold_batch_probe.rs` measures the same 64-pair gather at about 0.5 ms against about 2.2 ms in this profile. It repeats one index set, so its reads likely hit the cache more often than the live path, which reads different experts on each layer. Its gather numbers should not be used to apportion a step.

## Not measured

- GB10, H100 and any compute capability other than 12.1. The fold applies to all of them. They keep the threshold of 8.
- Whether the earlier GB10 advice to keep `--parallel` at 7 or below still holds. It predates the fold.
- The trace with the shipped size gate. It moves fewer weights to GEMM than the arm traced above.

## Reproduction

```bash
# Kernel comparison: [B, 1, K] takes GEMM from 2 rows, [B, K] stays on qmv below 8.
MLXCEL_QMM_FOLD_MIN_ELEMENTS=1 MLXCEL_QMM_FOLD_MIN_ROWS=2 cargo run --release -p mlxcel-core --features cuda --example qmm_fold_batch_probe

# Serving arms on one binary.
MLXCEL_QMM_FOLD_BATCH=0 ./target/release/mlxcel-server -m MODEL --max-batch-size 8 --ignore-eos
MLXCEL_QMM_FOLD_MIN_ELEMENTS=1 ./target/release/mlxcel-server -m MODEL --max-batch-size 8 --ignore-eos
MLXCEL_QMM_FOLD_MIN_ROWS=8 ./target/release/mlxcel-server -m MODEL --max-batch-size 8 --ignore-eos
python3 scripts/bench_serving_concurrency.py --concurrency 1,4,5,8 --prompt-tokens 128 --max-tokens 200
```
