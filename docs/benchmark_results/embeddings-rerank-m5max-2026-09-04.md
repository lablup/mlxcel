# Embedding and rerank throughput on M5 Max (2026-09-04)

Apple Silicon counterpart to [`embeddings-rerank-gb10-2026-08-26.md`](embeddings-rerank-gb10-2026-08-26.md), taken as part of the mlxcel 0.6.0 M5 Max benchmark campaign. It covers the same subsystem: `POST /v1/embeddings` and `POST /v1/rerank`, on the same 20-checkpoint roster defined in `scripts/bench_embeddings.py`.

Raw rows: `benchmarks/metal_m5max_embeddings_2026-09-04.csv` (90 rows, one per model, input kind and batch size; every cell measured, none failed).

## Environment

| Item | Value |
|------|-------|
| **Hardware** | MacBook Pro M5 Max, 128 GB unified memory |
| **OS** | macOS 26.6.2 (build 25G83) |
| **mlxcel** | 0.6.0 at `b2ff1eee` (main) |
| **Build** | `cargo build --release --features metal,accelerate` |
| **MLX** | pinned commit `9a795735` |
| **Server flags** | embedders via `mlxcel-server -m <dir>`, rerankers via `mlxcel-server -m <dir> --reranker-model <dir>` (rerank-only shape, weights loaded once) |

## Method

Identical to the GB10 pass, so the two files are directly comparable: one fresh server per checkpoint, one warmup request, then five repetitions per cell with the median reported. `tokens_per_s` is `usage.prompt_tokens / p50` and includes HTTP, tokenization and batching, so it is an end-to-end number, not a kernel number. No Python comparison runtime was installed on this host, so these are absolute mlxcel figures rather than a gap measurement.

Two harness changes were needed to run it here, both recorded below under [Harness changes](#harness-changes). Neither alters the workload.

## Embedders (text)

| Model | Load (s) | 1 short input p50 (ms) | 32 short inputs p50 (ms) | 32 short inputs tok/s | 32 long inputs p50 (ms) | 32 long inputs tok/s |
|-------|---------:|-----------------------:|-------------------------:|----------------------:|------------------------:|---------------------:|
| all-MiniLM-L6-v2 | 1.0 | 4.08 | 8.34 | 69047 | 29.82 | 274750 |
| multilingual-e5-small | 1.0 | 2.78 | 8.99 | 78341 | 120.12 | 136393 |
| bge-m3-safetensors | 1.0 | 5.91 | 27.97 | 25165 | 613.88 | 35760 |
| modernbert-embed-base | 1.0 | 4.85 | 15.60 | 36911 | 192.06 | 92304 |
| siglip-base-patch16-224 | 1.0 | 3.78 | 26.60 | 21654 | 31.53 | 64956 |
| embeddinggemma-300m-4bit | 1.0 | 4.37 | 18.06 | 31896 | 116.66 | 132491 |
| Qwen3-Embedding-0.6B | 1.0 | 6.97 | 27.34 | 19899 | 376.69 | 44514 |
| llama-nemotron-embed-1b-v2 | 1.0 | 8.46 | 43.78 | 12427 | 619.71 | 27058 |
| Nemotron-3-Embed-1B-BF16 | 1.0 | 19.78 | 67.03 | 7639 | 965.42 | 17700 |
| Nemotron-3-Embed-1B-BF16-8bit | 1.0 | 7.25 | 48.45 | 10567 | 986.35 | 17324 |
| LFM2.5-Embedding-350M | 1.0 | 5.04 | 20.98 | 25928 | 198.29 | 82626 |

SigLIP truncates every input to its fixed 64 positions, so its `long` row measures 64-token inputs and is not comparable with the other `long` rows.

## Embedders (vision-language)

| Model | Load (s) | 1 short text p50 (ms) | 8 long text p50 (ms) | 1 image p50 (ms) | visual tokens | 4 images p50 (ms) |
|-------|---------:|----------------------:|---------------------:|-----------------:|--------------:|------------------:|
| Qwen3-VL-Embedding-2B | 1.0 | 15.33 | 243.63 | 55.79 | 71 | 222.19 |
| llama-nemotron-embed-vl-1b-v2 | 1.0 | 10.79 | 159.18 | 62.17 | 264 | 247.26 |

The fixture expands to 71 visual tokens on Qwen3-VL-Embedding and 264 on Llama-Nemotron-VL-Embed, matching the GB10 report exactly. That equality is the check that the data-URI request path used here feeds the models the same input the GB10 pass fed them over `file://`. Real page scans expand to 1000-4000 tokens on these families, so these rows describe per-request overhead more than sustained vision throughput.

## Rerankers

| Model | Kind | Load (s) | 8 short docs p50 (ms) | 32 short docs p50 (ms) | 8 long docs p50 (ms) | 4 image docs p50 (ms) |
|-------|------|---------:|----------------------:|-----------------------:|---------------------:|----------------------:|
| ms-marco-MiniLM-L6-v2 | rerank | 1.0 | 9.08 | 25.13 | 54.55 | - |
| bge-reranker-v2-m3 | rerank | 3.0 | 50.74 | 197.25 | 576.39 | - |
| gte-reranker-modernbert-base | rerank | 1.0 | 25.06 | 96.62 | 224.82 | - |
| Qwen3-Reranker-0.6B-4bit | rerank | 2.0 | 103.55 | 410.56 | 546.27 | - |
| Qwen3-VL-Reranker-2B | rerank_vl | 1.0 | 71.77 | - | - | 274.67 |

## Observations

**Quantization helps latency far more than it helps throughput.** `Nemotron-3-Embed-1B-BF16` and its 8-bit sibling are the same model at two precisions, and they separate cleanly by regime: at one short input the 8-bit variant is 2.7x faster (7.25 ms against 19.78 ms), at 32 short inputs the gap narrows to 1.4x, and at 32 long inputs the two are within about 2% of each other, with the 8-bit variant marginally slower (986 ms against 965 ms, 17324 against 17700 tok/s). Small-batch embedding is dominated by weight movement, which quantization fixes; the long-batch cells are compute-bound, where it buys nothing.

**The bf16 against 8-bit verdict inverts between the two backends, so do not carry it across.** The GB10 report concludes that the 8-bit Nemotron-3-Embed conversion is 2 to 3x slower than its bf16 sibling. On Metal the same two checkpoints under the same harness give the opposite answer at every batch size except the largest:

| Cell | GB10 (CUDA) | M5 Max (Metal) |
|------|-------------|----------------|
| 1 short input | bf16 12.33 ms, 8-bit 13.75 ms (8-bit 1.1x slower) | bf16 19.78 ms, 8-bit 7.25 ms (8-bit **2.7x faster**) |
| 8 short inputs | 18.31 ms, 41.33 ms (2.3x slower) | 25.16 ms, 17.70 ms (**1.4x faster**) |
| 32 short inputs | 54.91 ms, 147.39 ms (2.7x slower) | 67.03 ms, 48.45 ms (**1.4x faster**) |
| 32 long inputs | 676.02 ms, 1111.08 ms (1.6x slower) | 965.42 ms, 986.35 ms (1.02x slower) |

Two directions, same checkpoints, same script. Whatever makes the 8-bit conversion expensive on the CUDA path does not apply on Metal, where it behaves like a normal quantization win that fades as the work becomes compute-bound. Any guidance about picking a precision for these embedders has to name the backend.

**Batching pays across the board, but the payoff is size-dependent.** `all-MiniLM-L6-v2` goes from 4.08 ms for one input to 8.34 ms for 32, so 32x the work for 2x the time. The larger embedders flatten out much sooner: `bge-m3-safetensors` needs 4.7x the time for the same 32x work, and `Nemotron-3-Embed-1B-BF16` 3.4x.

**The two 1B-class NVIDIA embedders are not equivalent.** `llama-nemotron-embed-1b-v2` is roughly 2.3x faster than `Nemotron-3-Embed-1B-BF16` at one short input (8.46 ms against 19.78 ms) and 1.5x at 32 long inputs, despite the similar parameter count.

**Reranker cost tracks model size, not the rerank shape.** The MiniLM cross-encoder scores 8 short documents in 9.08 ms; `Qwen3-Reranker-0.6B-4bit`, a generative reranker, needs 103.55 ms for the same shape, an 11x spread.

## Harness changes

Two defects in `scripts/bench_embeddings.py` had to be fixed before this pass could produce a correct CSV. Both are committed alongside this report.

1. **The `hardware` column was hardcoded to `"NVIDIA_GB10_122GB"`.** Any CSV produced on another host silently carried the GB10 label, which would have made this file unusable next to the GB10 one. Replaced with the same host detection `bench_decode.sh` uses (`Apple_M5_Max_128GB` here, `nvidia-smi`-derived on CUDA).

2. **Image cells sent the fixture as a `file://` URL and every one of them failed.** PR #1481 (`fix(server): classify projector flags and confine request media`, merged 2026-08-28) made the server refuse `file://` media unless a directory is allow-listed with `--media-path`. The GB10 pass ran on 2026-08-26, two days before that landed, which is why it never saw this. The harness now sends the fixture as a base64 `data:` URI instead, which needs no allow-list and produces identical visual-token counts.

### Why the image cells use a data URI

`--media-path` works, but the URL must be **relative to the media root**. An absolute path is concatenated onto the root rather than replacing it, so `file:///abs/path/tests/fixtures/test_image.png` is probed at `<root>/abs/path/tests/fixtures/test_image.png` and reported as `file does not exist or cannot be opened`. That concatenation is deliberate: `resolve_media_file_in` in `src/server/media_root.rs` strips the leading separator before joining, matching llama-server b10621's `media_path + file_path`, and `an_absolute_looking_path_is_concatenated_not_joined` in `src/server/media_root_tests.rs` pins it precisely so an absolute URL cannot escape the root.

Verified against a live server with `--media-path <repo>/tests/fixtures`: `file://test_image.png`, `test_image.png` and `file:///test_image.png` all return HTTP 200 with a 2048-dimension vector, `file://../../etc/passwd` is refused with `file path is not allowed`, and only the absolute form fails. The harness had been sending the absolute form, which is why every image cell returned HTTP 400 on the first run.

The harness now sends the fixture as a base64 `data:` URI instead of switching to a relative `file://` URL. Both work; the data URI was kept because it needs no server flag at all, which makes the ladder reproducible on a host where the operator has not set `--media-path`. The fixture is 679 bytes, so inlining it costs nothing, and the resulting visual token counts match the GB10 pass exactly.

The remaining defect is diagnostic rather than functional: the error names the path the client sent rather than the path actually probed, which makes the concatenation rule impossible to infer from the message, and neither `mlxcel-server --help` nor `docs/llama-server-compat.md` states that the path must be relative. That is filed as issue #1612.
