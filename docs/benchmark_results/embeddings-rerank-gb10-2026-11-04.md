# Embedding and rerank throughput on GB10 (2026-11-04)

> Retention note (2026-11-10): final results are retained here. Superseded experiments are indexed in [benchmark index](README.md); retired run captures and any historical artifact paths resolve in the [frozen evidence tree](https://github.com/lablup/mlxcel/tree/2dc8ed47b30dc204d892328098e8cc34ec02ddc1/docs/benchmark_results/). Measurement values and their original limitations have not been remeasured.

Embedding and reranking throughput for the mlxcel server on **NVIDIA GB10 (DGX Spark)** at mlxcel 0.8.1. This pass repeats the 2026-08-26 epic #1348 measurement on the current tree, with the same 20-checkpoint roster, workloads and five-repetition ladder, so the two are directly comparable cell by cell.

The raw rows live in `benchmarks/cuda_gb10_embeddings_2026-11-04.csv` (102 cells, one per model, input kind and batch size, no failures). The harness is `scripts/bench_embeddings.py`.

## Environment

| Item | Value |
|------|-------|
| **Hardware** | NVIDIA GB10 (DGX Spark), 122 GB unified LPDDR5x, SM 121 |
| **Driver / toolkit** | 580.178.04 / CUDA 13.0.88; CSV `backend` = `cuda-13.0` |
| **OS** | Linux aarch64, kernel 7.0.0-1019-nvidia |
| **mlxcel** | 0.8.1 at `630aaeba` |
| **MLX pin** | `7ca40033` |
| **Build** | `MLX_CUDA_ARCHITECTURES=121 cargo build --release --features cuda --bins` |
| **Server flags** | defaults; embedders via `mlxcel-server -m <dir>`, rerankers via `mlxcel-server -m <dir> --reranker-model <dir>` |
| **CSV** | `benchmarks/cuda_gb10_embeddings_2026-11-04.csv` |

## Method

Unchanged from the 2026-08-26 pass. One server process per checkpoint, started fresh; `load_ms` is the time until `GET /v1/models` answers 200, polled once per second, so it is a ceiling with one second granularity. One warmup request, then five repetitions of every cell; the tables report the median (`p50_ms`). Text workloads: `short` is an 18-token sentence, `long` a 256-token paragraph, sent as batches of 1, 8 and 32 identical inputs in one request. Multimodal embedders run text batches of 1 and 8 and image batches of 1 and 4; multi-vector embedders run the text ladder. Rerankers score one query against 8 or 32 short documents or 8 long documents; the Qwen3-VL reranker runs the 8-short cell plus 4 image documents. `tokens_per_s` is real (non-padding) prompt tokens per second including HTTP and batching; `inputs_per_s` is documents per second for rerankers.

The image is `tests/fixtures/test_image.png` (71 visual tokens on Qwen3-VL-Embedding, 264 on Llama-Nemotron-VL-Embed), so the image cells describe per-request overhead rather than sustained vision throughput.

## Embedders

| Model | Family | Load (s) | 1 short (ms) | 32 short (ms) | 32 short tok/s | 32 long (ms) | 32 long tok/s | 32-short p50, new/old |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `all-MiniLM-L6-v2` | BERT | 1.0 | 2.0 | 6.1 | 95,015 | 62.7 | 130,576 | 0.66x |
| `multilingual-e5-small` | BERT (XLM-R tokenizer) | 2.0 | 3.1 | 9.1 | 77,142 | 284.3 | 57,629 | 0.89x |
| `bge-m3-safetensors` | XLM-RoBERTa | 2.0 | 10.2 | 38.1 | 18,460 | 1,949.5 | 11,260 | 0.98x |
| `modernbert-embed-base` | ModernBERT | 1.0 | 7.4 | 24.9 | 23,091 | 757.7 | 23,398 | 1.08x |
| `siglip-base-patch16-224` | SigLIP text | 1.0 | 4.3 | 32.7 | 17,608 | 39.1 | 52,436 | 0.55x |
| `embeddinggemma-300m-4bit` | EmbeddingGemma (4-bit) | 3.0 | 4.1 | 18.2 | 31,583 | 192.6 | 80,248 | 0.57x |
| `Qwen3-Embedding-0.6B` | Qwen3-Embedding | 1.0 | 7.4 | 31.2 | 17,423 | 450.9 | 37,188 | 0.73x |
| `llama-nemotron-embed-1b-v2` | bidirectional Llama | 1.0 | 12.3 | 52.7 | 10,326 | 564.2 | 29,718 | 0.94x |
| `Nemotron-3-Embed-1B-BF16` | Ministral 3 (bf16) | 2.0 | 11.2 | 41.8 | 12,243 | 639.6 | 26,717 | 0.76x |
| `Nemotron-3-Embed-1B-BF16-8bit` | Ministral 3 (8-bit) | 2.0 | 9.0 | 50.0 | 10,249 | 1,477.1 | 11,568 | 0.34x |
| `LFM2.5-Embedding-350M` | LFM2.5 | 1.0 | 6.6 | 25.6 | 21,249 | 230.6 | 71,058 | 1.08x |

The last column is this run's 32-short median divided by the 2026-08-26 median: below 1.00 means the same cell got faster. Nine of eleven text embedders improved, six of them by more than 20% and one by 2%, while `modernbert-embed-base` and `LFM2.5-Embedding-350M` regressed by 8%. The largest single move is the 8-bit Nemotron conversion, 2.9x faster than its 2026-08-26 cell, though it is still slower than its bf16 sibling (50.0 ms against 41.8 ms for 32 short inputs), so the earlier observation that 8-bit saves memory rather than time still holds, at a smaller margin.

## Image inputs

| Model | 1 image (ms) | 4 images (ms) | image tok/s (4) |
|---|---:|---:|---:|
| `Qwen3-VL-Embedding-2B` | 37.5 | 136.4 | 2,082 |
| `llama-nemotron-embed-vl-1b-v2` | 51.8 | 199.4 | 5,295 |

## Multi-vector embedders

| Model | 1 short (ms) | 32 short (ms) | 32 short tok/s | 1 long (ms) | 32 long (ms) | 32 long tok/s |
|---|---:|---:|---:|---:|---:|---:|
| `colSmol-256M-merged` | 5.5 | 51.2 | 17,513 | 23.6 | 625.4 | 27,940 |
| `colqwen2.5-v0.2-merged` | 33.1 | 136.4 | 6,568 | 79.8 | 2,036.7 | 8,390 |

The ColQwen2.5 32-long cell still spends 2.0 s serializing 128-wide per-token vectors as JSON floats, the same wire cost the 2026-08-26 pass recorded. `encoding_format: base64` remains the recommended transport for late-interaction clients.

## Rerankers

| Reranker | Kind | Load (s) | 8 short (ms) | 32 short (ms) | 32 short docs/s | 8 long (ms) | 8 long tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| `ms-marco-MiniLM-L6-v2` | BERT sequence classifier | 1.0 | 2.6 | 7.4 | 4,342.8 | 35.2 | 116,210 |
| `bge-reranker-v2-m3` | XLM-RoBERTa sequence classifier | 2.0 | 17.3 | 62.6 | 511.3 | 506.7 | 11,148 |
| `gte-reranker-modernbert-base` | ModernBERT sequence classifier | 1.0 | 9.2 | 30.2 | 1,059.3 | 186.6 | 24,487 |
| `Qwen3-Reranker-0.6B-4bit` | Qwen3 generative (4-bit) | 1.0 | 25.8 | 97.9 | 326.7 | 152.7 | 31,852 |
| `Qwen3-VL-Reranker-2B` | Qwen3-VL generative (bf16) | 2.0 | 89.3 | not in ladder | not in ladder | not in ladder | not in ladder |

`Qwen3-VL-Reranker-2B` with 4 image documents: p50 165.5 ms, 3,069 tok/s, 24.2 documents/s. Image documents are scored one per forward pass, as before.

The reranker ordering is unchanged from 2026-08-26: MiniLM is two orders of magnitude ahead on short documents, the ModernBERT cross-encoder is second, bge-m3 is limited by its 24-layer XLM-R backbone on long documents, and the generative Qwen3 reranker pays a fixed decode cost per pair.

## Caveats

- One box, one run of five repetitions per cell. The CSV `min_ms` column shows the spread; this is not a thermal or sustained-load study.
- The fixture image is tiny; per-image numbers on real documents will be several times higher and dominated by the vision tower.
- CUDA numbers are not interchangeable across GPUs.
- Generative reranker rows depend on batch composition by design (left padding shifts RoPE positions, matching the reference implementation).
- The harness now records ROCm hosts (`rocminfo` gfx target, `/opt/rocm/.info/version`, sysfs VRAM) with the same `hardware` string shape as `bench_decode.sh`, so the same script can produce the ROCm ladder on a gfx1151 host without edits.
