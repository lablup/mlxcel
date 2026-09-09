# Embedding and rerank throughput on M5 Max (2026-09-09)

Third pass on this host, and the first whose CSV columns are aligned. Companion to [`embeddings-rerank-m1ultra-2026-09-09.md`](embeddings-rerank-m1ultra-2026-09-09.md) and successor to [`embeddings-rerank-m5max-2026-09-04.md`](embeddings-rerank-m5max-2026-09-04.md). Covers `POST /v1/embeddings` and `POST /v1/rerank` over the 20-checkpoint roster in `scripts/bench_embeddings.py`.

Raw rows: `benchmarks/metal_m5max_embeddings_2026-09-09.csv` (102 rows, one per model, input kind and batch size; no skips, no failures).

## Environment

| Item | Value |
|------|-------|
| **Hardware** | MacBook Pro M5 Max, 128 GB unified memory |
| **OS** | macOS 26.6.2 (build 25G83) |
| **mlxcel** | 0.7.0-beta.1 at `66b8346e` |
| **Build** | `cargo build --release --features metal,accelerate` |
| **MLX** | pinned commit `9a795735` |
| **Server flags** | embedders via `mlxcel-server -m <dir>`, rerankers via `mlxcel-server -m <dir> --reranker-model <dir>` |

## Method

Unchanged from the two earlier passes: one fresh server per checkpoint, one warmup request, five repetitions per cell with the median reported. `tokens_per_s` is `usage.prompt_tokens / p50`, so it is end to end and carries HTTP, tokenization and batching. It is not comparable to a `bench_decode.sh` decode rate. No Python comparison runtime was run, so these are absolute mlxcel figures rather than a parity gap.

Two harness defects were fixed before this pass and are described under [Harness state](#harness-state). Neither changes the workload.

## Embedders (text)

| Model | Load (s) | 1 short p50 (ms) | 32 short p50 (ms) | 32 short tok/s | 32 long p50 (ms) | 32 long tok/s |
|-------|---------:|-----------------:|------------------:|---------------:|-----------------:|--------------:|
| multilingual-e5-small | 1.0 | 2.74 | 8.79 | 80101 | 123.25 | 132935 |
| all-MiniLM-L6-v2 | 1.0 | 4.79 | 9.82 | 58659 | 35.05 | 233705 |
| embeddinggemma-300m-4bit | 1.0 | 4.28 | 17.28 | 33331 | 118.73 | 130177 |
| bge-m3-safetensors | 1.0 | 6.78 | 28.42 | 24774 | 625.88 | 35074 |
| modernbert-embed-base | 1.0 | 6.96 | 23.60 | 24411 | 197.32 | 89842 |
| siglip-base-patch16-224 | 1.0 | 3.61 | 25.53 | 22559 | 32.69 | 62642 |
| LFM2.5-Embedding-350M | 1.0 | 6.62 | 27.25 | 19964 | 198.92 | 82364 |
| Qwen3-Embedding-0.6B | 1.0 | 7.37 | 29.13 | 18676 | 378.08 | 44351 |
| Nemotron-3-Embed-1B-BF16 | 1.0 | 9.17 | 37.49 | 13655 | 737.04 | 23185 |
| llama-nemotron-embed-1b-v2 | 1.0 | 8.99 | 45.04 | 12079 | 634.68 | 26420 |
| Nemotron-3-Embed-1B-BF16-8bit | 1.0 | 8.00 | 46.82 | 10937 | 697.39 | 24503 |

SigLIP truncates every input to its fixed 64 positions, so its `long` row measures 64-token inputs and does not belong beside the other `long` rows.

## Multimodal and multivector

| Model | Kind | Text tok/s (b=8) | Image p50 (ms, b=4) |
|-------|------|-----------------:|--------------------:|
| colSmol-256M-merged | multivector | 15347 | no image cell |
| Qwen3-VL-Embedding-2B | vl | 9837 | 124.35 |
| llama-nemotron-embed-vl-1b-v2 | vl | 9085 | 248.04 |
| colqwen2.5-v0.2-merged | multivector | 4683 | no image cell |

## Rerankers

Eight short documents per request.

| Model | p50 (ms) | tok/s |
|-------|---------:|------:|
| ms-marco-MiniLM-L6-v2 | 5.84 | 46573 |
| gte-reranker-modernbert-base | 10.95 | 25572 |
| bge-reranker-v2-m3 | 16.65 | 20240 |
| Qwen3-Reranker-0.6B-4bit | 21.28 | 38341 |
| Qwen3-VL-Reranker-2B | 72.13 | 10648 |

Latency and throughput rank differently here, and the reason is the token count rather than the engine: `Qwen3-Reranker-0.6B-4bit` is third-slowest per request at 21.28 ms while third-fastest per token at 38341, because a generative reranker wraps each pair in a prompt template and so processes 816 tokens where the MiniLM cross-encoder processes 272. Read `p50` when sizing a request budget and `tokens_per_s` when comparing engines.

## Comparison with the 2026-09-06 pass

Flat, which is the expected result: nothing in this window touched the embedding or rerank path. Median ratios across the 102 shared cells are text 1.00x, rerank 0.98x, vl 0.98x, rerank_vl 1.38x, multivector 1.29x.

An earlier draft of this file reported rerankers at 1.56x to 4.87x and named the activation-helper dtype restore as the cause. That was wrong and is corrected in `c7fb1bc0`: the comparison had used `metal_m5max_embeddings_2026-09-04.csv` without noticing that the 2026-09-06 file sits beside it with the full 102-cell roster. The speedup is real but landed before 2026-09-06. Per cell, `Qwen3-Reranker-0.6B-4bit` on long documents reads 8904, then 42515, then 42381 tokens per second across the three passes, and `bge-reranker-v2-m3` reads 9801, 35834, 35642. The jump is between the first two and nothing moves after.

## Cross-host comparison

Deferred. The M1 Ultra file this would compare against was written before the column fix and is being re-taken; comparing against it now would pin numbers that are about to be replaced.

One provisional note, to be re-checked against the re-run. Against the pre-fix M1 Ultra file, M5 Max leads by a median of 2.95x at 32 short inputs, and the one entry outside the 2.2x to 3.1x band is `all-MiniLM-L6-v2` at 1.64x. The M1 Ultra report reads the outlier differently, as `Nemotron-3-Embed-1B-BF16` at 1.71x, which does not reproduce here: that model measures 3.02x, squarely inside the band. Two readings of the same pair of files disagreeing is itself a reason to wait for the re-run rather than to argue from either.

## Harness state

Two defects in `scripts/bench_embeddings.py` were fixed before this pass. Both had the same shape: the script kept running and produced a file, so neither surfaced as an error.

**The roster resolved against a store that no longer existed.** Paths named `~/.cache/mlxcel/models`, and consolidating the two model stores drained that cache into `models/`. A missing path is a `[skip]`, not a failure, so the sweep would have printed twenty skips, written an empty CSV, and reported `DONE`. Fixed in `10f3aae8`, then again in `bf271884` after the first fix hardcoded `models/` and broke the other host, whose checkpoints live under `models/mlx/`. The root is now probed by looking for a checkpoint the roster names, with `MLXCEL_MODEL_STORE` as an override.

**Two of the three `writerow` calls omitted `mlx_commit`.** A measured row carried 20 fields against a 21-field header, shifting every column from `mlxcel_commit` rightward. `csv.writer` does not check the count and `csv.DictReader` fills the missing tail with `None`, so nothing errored on either side. Fixed in `f42d127d`. `benchmarks/README-embeddings-csv-columns.md` records which earlier files carry the shift; the 2026-09-06 file is one of them, and its timing columns are unaffected because the shift begins after them.
