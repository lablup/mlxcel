# Embedding and rerank throughput on GB10 (epic #1348)

This is the end-of-epic performance pass for the embedding and reranking subsystem that epic #1348 added: `POST /v1/embeddings` (PR #1408 and the family ports in PRs #1410 to #1416) and `POST /v1/rerank` (PR #1417). Per the epic's working rule, no performance numbers were taken while the family ports were being implemented, because several worktrees were compiling and testing on the same GPU; this pass ran once, after every PR had merged, on a machine with no other cargo, test, or CI process active (load average 1.3 at start).

The raw rows live in `benchmarks/cuda_gb10_embeddings_2026-08-26.csv` (102 rows, one per model, input kind and batch size). The harness is `scripts/bench_embeddings.py`.

## Environment

| Item | Value |
|------|-------|
| **Hardware** | NVIDIA GB10 (DGX Spark), 122 GB unified LPDDR5x, SM 121 |
| **Driver / toolkit** | 580.173.02 / CUDA 13 |
| **OS** | Linux aarch64, kernel 6.17 |
| **mlxcel** | 0.6.0 at `19cccf50` (main, after PR #1420) |
| **Build** | `cargo build --release --features cuda --bins` (fat LTO release profile, the binary served here links `libcudart.so.13` and `libcublasLt.so.13`) |
| **MLX** | pinned commit `9a795735` (`src/lib/mlx-cpp/CMakeLists.txt` `GIT_TAG`) |
| **Server flags** | defaults; embedders via `mlxcel-server -m <dir>`, rerankers via `mlxcel-server -m <dir> --reranker-model <dir>` (the rerank-only shape, weights loaded once) |

## Method

- One server process per checkpoint, started fresh; `load_ms` is the time until `GET /v1/models` answers 200, polled once per second, so it is a ceiling with one second granularity.
- One warmup request, then five repetitions of every cell; the tables report the median (`p50_ms`). The CSV also carries the mean and the minimum.
- Text workloads: `short` is one 18 token sentence, `long` is a 256 token paragraph (measured per tokenizer; the CSV records the actual `usage.prompt_tokens`), sent as batches of 1, 8 and 32 identical inputs in one request. The multimodal embedders run batches of 1 and 8 for text and 1 and 4 for images, and the multi-vector embedders run the text ladder. Rerank workloads are one query against 8 or 32 short documents or 8 long documents; the Qwen3-VL reranker runs only the 8 short documents cell plus 4 image documents. SigLIP truncates every input to its fixed 64 positions, so its `long` rows measure 64 token inputs.
- The image is `tests/fixtures/test_image.png`, a small fixture that expands to 71 visual tokens on Qwen3-VL-Embedding and 264 on Llama-Nemotron-VL-Embed. Real page scans expand to 1000 to 4000 tokens on these families, so the image rows describe per-request overhead more than sustained vision throughput.
- `tokens_per_s` is `usage.prompt_tokens / p50`, that is, real (non padding) tokens per second including HTTP, tokenization and batching, not a kernel number. `inputs_per_s` (documents per second for rerankers) is `batch / p50`.
- No comparison runtime ran on this box (no PyTorch, `sentence-transformers` or `transformers` is installed), so these are absolute mlxcel numbers, not a gap measurement.

## Embedders

| Model | Family | Load (s) | 1 short input p50 (ms) | 32 short inputs p50 (ms) | 32 short inputs tok/s | 32 long inputs p50 (ms) | 32 long inputs tok/s |
|---|---|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` | BERT | 1.0 | 2.66 | 9.2 | 62,539 | 66.2 | 123,740 |
| `multilingual-e5-small` | BERT (XLM-R tokenizer) | 2.0 | 2.13 | 10.2 | 68,747 | 274.6 | 59,663 |
| `bge-m3-safetensors` | XLM-RoBERTa | 2.0 | 11.64 | 39.0 | 18,041 | 1,991.3 | 11,024 |
| `modernbert-embed-base` | ModernBERT | 1.0 | 5.36 | 23.1 | 24,925 | 736.6 | 24,067 |
| `siglip-base-patch16-224` | SigLIP text | 1.0 | 5.03 | 59.0 | 9,763 | 70.1 | 29,205 |
| `embeddinggemma-300m-4bit` | EmbeddingGemma (4-bit) | 3.0 | 6.37 | 32.2 | 17,889 | 220.7 | 70,028 |
| `Qwen3-Embedding-0.6B` | Qwen3-Embedding | 2.0 | 10.12 | 42.6 | 12,757 | 513.9 | 32,627 |
| `llama-nemotron-embed-1b-v2` | bidirectional Llama | 1.0 | 12.66 | 55.8 | 9,747 | 626.2 | 26,776 |
| `Nemotron-3-Embed-1B-BF16` | Ministral 3 (bf16) | 2.0 | 12.33 | 54.9 | 9,325 | 676.0 | 25,278 |
| `Nemotron-3-Embed-1B-BF16-8bit` | Ministral 3 (8-bit) | 2.0 | 13.75 | 147.4 | 3,474 | 1,111.1 | 15,380 |
| `LFM2.5-Embedding-350M` | LFM2.5-Embedding | 1.0 | 5.59 | 23.6 | 23,005 | 291.7 | 56,176 |
| `Qwen3-VL-Embedding-2B` | Qwen3-VL-Embedding | 2.0 | 21.13 | 33.5 (batch 8) | 8,606 | 333.8 (batch 8) | 13,014 |
| `llama-nemotron-embed-vl-1b-v2` | Llama-Nemotron-VL-Embed | 2.0 | 12.48 | 18.7 (batch 8) | 7,276 | 142.1 (batch 8) | 29,506 |
| `colSmol-256M-merged` | ColIdefics3 (multi-vector) | 1.0 | 5.47 | 55.3 | 16,214 | 645.9 | 27,052 |
| `colqwen2.5-v0.2-merged` | ColQwen2.5 (multi-vector) | 2.0 | 33.07 | 147.1 | 6,090 | 3,094.0 | 5,523 |

### Image inputs

| Model | 1 image p50 (ms) | 1 image tokens | 4 images p50 (ms) | 4 images tok/s |
|---|---|---|---|---|
| `Qwen3-VL-Embedding-2B` | 43.3 | 71 | 167.2 | 1,699 |
| `llama-nemotron-embed-vl-1b-v2` | 98.1 | 264 | 392.5 | 2,690 |

## Rerankers

| Reranker | Kind | Load (s) | 8 short docs p50 (ms) | 32 short docs p50 (ms) | 32 short docs docs/s | 8 long docs p50 (ms) | 8 long docs tok/s |
|---|---|---|---|---|---|---|---|
| `ms-marco-MiniLM-L6-v2` | BERT sequence classifier | 1.0 | 2.3 | 6.7 | 4,785 | 36.9 | 111,124 |
| `bge-reranker-v2-m3` | XLM-RoBERTa sequence classifier | 2.0 | 17.4 | 70.8 | 452 | 514.2 | 10,986 |
| `gte-reranker-modernbert-base` | ModernBERT sequence classifier | 1.0 | 10.0 | 35.0 | 913 | 183.1 | 24,943 |
| `Qwen3-Reranker-0.6B-4bit` | Qwen3 generative (4-bit) | 2.0 | 40.5 | 155.1 | 206 | 162.3 | 29,962 |
| `Qwen3-VL-Reranker-2B` | Qwen3-VL generative (bf16) | 2.0 | 99.0 | not in ladder | not in ladder | not in ladder | not in ladder |

Qwen3-VL-Reranker-2B with 4 image documents: p50 166.9 ms, 508 tokens, 24.0 documents/s (image documents are scored one per forward pass).


## Observations

- The encoder families are dominated by fixed per-request cost at batch 1 (2 to 6 ms for MiniLM, e5-small, ModernBERT, SigLIP and LFM2.5) and scale almost linearly to batch 32; MiniLM reaches 124k tokens/s on 32 long inputs and the rerank cross-encoder built on it reaches 163k tokens/s on 32 short pairs.
- `bge-m3` (XLM-RoBERTa large, 24 layers, 1024 wide) is the slowest text encoder per token, at about 11k tokens/s on long batches, roughly 5x below e5-small and 11x below MiniLM, consistent with its parameter count rather than with a pathology; it is also the family whose long input (686 tokens after the tokenizer) is the longest in the ladder.
- The 8-bit `Nemotron-3-Embed-1B` conversion is 2 to 3x slower than its bf16 sibling at every batch size (for example 147 ms against 55 ms for 32 short inputs). Prefill-shaped quantized matmuls on this backend do not beat bf16 GEMM at these sizes; the 8-bit checkpoint saves memory, not time. The 4-bit EmbeddingGemma does not show the same penalty, but it is a different backbone and a different bit width, so this is an observation to carry into the CUDA quantized-matmul work, not a conclusion.
- The decoder-based embedders (Qwen3-Embedding, bidirectional Llama, Ministral 3) cluster at 10 to 13 ms for a single short input and 25k to 33k tokens/s on long batches, about 2x the encoder-only families per token, which is the expected cost of 1B class backbones.
- Multi-vector output is not free on the wire: ColQwen2.5 returns one 128 wide row per token, and its 32 long inputs cell spends 3.1 s at 5.5k tokens/s, a 6x gap to the same size decoder embedders that includes serializing 17k rows of JSON floats. `encoding_format: base64` is the recommended transport for late-interaction clients.
- The multimodal embedders cost 43 ms (Qwen3-VL-Embedding-2B) and 98 ms (Llama-Nemotron-VL-Embed) per small image, and scale linearly in the image count because images are embedded one per forward pass (DeepStack injection and M-RoPE are per sequence). The Qwen3-VL reranker likewise scores image documents one at a time, which is why 4 image documents take 167 ms against 99 ms for 8 short text documents.
- Load times are 1 to 3 s for every checkpoint, dominated by safetensors reads; the largest served here is the 7 GB merged ColQwen2.5 base.

## Caveats

- One box, one run of five repetitions per cell; the CSV `min_ms` column shows the spread was small (typically within 10 percent of the median) but this is not a thermal or sustained-load study.
- The fixture image is tiny; per-image numbers on real documents will be several times higher and dominated by the vision tower.
- CUDA numbers are not interchangeable across GPUs; SM 121 with driver 580.173.02 and CUDA 13.
- The reranker rows for generative models depend on batch composition by design (left padding shifts RoPE positions, matching the reference implementation), so a client that batches differently will see different per-document latency, though the ordering is stable.
