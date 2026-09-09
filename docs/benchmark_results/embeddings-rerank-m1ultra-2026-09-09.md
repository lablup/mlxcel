# Embedding and rerank throughput on M1 Ultra (2026-09-09)

Third host for the subsystem measured in [`embeddings-rerank-gb10-2026-08-26.md`](embeddings-rerank-gb10-2026-08-26.md) and [`embeddings-rerank-m5max-2026-09-04.md`](embeddings-rerank-m5max-2026-09-04.md): `POST /v1/embeddings` and `POST /v1/rerank`, same 20-checkpoint roster from `scripts/bench_embeddings.py`.

Raw rows: `benchmarks/metal_m1ultra_embeddings_2026-09-09.csv` (102 rows, one per model, input kind and batch size; every roster entry resolved, none skipped, none failed). Taken at `f42d127d`, after the column-alignment fix described under Caveats.

## Environment

| Item | Value |
|------|-------|
| **Hardware** | Mac Studio M1 Ultra, 128 GB unified memory |
| **OS** | macOS 26.6.2 |
| **mlxcel** | 0.7.0-beta.1 at `f42d127d` |
| **Build** | `cargo build --release --features metal,accelerate` |
| **MLX** | pinned commit `9a795735` |
| **Model store** | `models/mlx/` |
| **Server flags** | embedders via `mlxcel-server -m <dir>`, rerankers with `--reranker-model <dir>` as well |

## Method

Same as the two earlier passes, so all three files line up: one fresh server per checkpoint, one warmup request, five repetitions per cell with the median reported. `tokens_per_s` is `usage.prompt_tokens / p50`, which makes it end to end. It carries HTTP, tokenization and batching, so it is not comparable to a `bench_decode.sh` decode rate. No Python comparison runtime was run here, so these are absolute mlxcel figures rather than a parity gap.

## Embedders (text)

| Model | Load (s) | 1 short p50 (ms) | 32 short p50 (ms) | 32 short tok/s | 32 long p50 (ms) | 32 long tok/s |
|-------|---------:|-----------------:|------------------:|---------------:|-----------------:|--------------:|
| all-MiniLM-L6-v2 | 2.0 | 4.15 | 15.91 | 36198 | 62.28 | 131544 |
| multilingual-e5-small | 1.0 | 6.53 | 20.66 | 34074 | 166.40 | 98461 |
| LFM2.5-Embedding-350M | 1.0 | 13.09 | 57.28 | 9497 | 757.67 | 21624 |
| embeddinggemma-300m-4bit | 2.0 | 13.79 | 47.85 | 12038 | 406.47 | 38024 |
| modernbert-embed-base | 1.0 | 14.05 | 46.99 | 12257 | 552.24 | 32102 |
| siglip-base-patch16-224 | 1.0 | 9.48 | 62.42 | 9227 | 72.15 | 28384 |
| bge-m3-safetensors | 1.0 | 16.60 | 90.39 | 7788 | 1640.66 | 13380 |
| Qwen3-Embedding-0.6B | 1.0 | 11.28 | 84.67 | 6425 | 1379.70 | 12153 |
| Nemotron-3-Embed-1B-BF16 | 2.0 | 18.62 | 113.15 | 4525 | 2414.79 | 7076 |
| llama-nemotron-embed-1b-v2 | 1.0 | 20.08 | 139.24 | 3907 | 2493.47 | 6725 |
| Nemotron-3-Embed-1B-BF16-8bit | 1.0 | 18.92 | 143.31 | 3573 | 3857.68 | 4430 |

## Multimodal and multivector

| Model | Kind | Text tok/s (b=8) | Image p50 (ms, b=4) |
|-------|------|-----------------:|--------------------:|
| Qwen3-VL-Embedding-2B | vl | 2910 | 352.1 |
| llama-nemotron-embed-vl-1b-v2 | vl | 2782 | 741.8 |
| colSmol-256M-merged | multivector | 7820 | no image cell |
| colqwen2.5-v0.2-merged | multivector | 1429 | no image cell |

## Rerankers

Eight short documents per request.

| Model | p50 (ms) | tok/s |
|-------|---------:|------:|
| ms-marco-MiniLM-L6-v2 | 6.31 | 43080 |
| gte-reranker-modernbert-base | 19.66 | 14244 |
| bge-reranker-v2-m3 | 38.19 | 8824 |
| Qwen3-Reranker-0.6B-4bit | 110.62 | 7377 |
| Qwen3-VL-Reranker-2B | 241.82 | 3176 |

## Observations

**The backend is not what decides the bf16 against 8-bit question, and as of the 2026-09-09 passes there is nothing left to decide.** The M5 Max report concluded that `Nemotron-3-Embed-1B-BF16` and its 8-bit sibling invert between CUDA and Metal, and that precision guidance therefore has to name the backend. That grouping does not survive a second Metal host, and the disagreement it rested on has since closed on its own.

| 32 short inputs, p50 | bf16 | 8-bit | faster |
|---|---:|---:|---|
| GB10, 2026-08-26 | | | bf16 (8-bit 2.7x slower) |
| M5 Max, 2026-09-04 `b2ff1eee` | 67.03 ms | 48.45 ms | 8-bit |
| M5 Max, 2026-09-06 `a50ff440` | 66.26 ms | 47.89 ms | 8-bit |
| M5 Max, 2026-09-09 `66b8346e` | 37.49 ms | 46.82 ms | **bf16** |
| M1 Ultra, 2026-09-09 `f42d127d` | 113.15 ms | 143.31 ms | **bf16** |

Two things follow. The backend was never the axis: M1 Ultra is Metal and has always agreed with GB10. And the M5 Max exception was temporary, closed by the bf16 half of the pair getting 1.77x faster between `a50ff440` and `66b8346e` while the 8-bit half stayed flat at 47.89 to 46.82 ms.

**Only the wide-dtype checkpoint moved, which points at the dtype work in that interval rather than at anything embedding-specific.** The two checkpoints are the same model at two precisions and they sit on the same host under the same harness, so the 8-bit one is a control. That interval carries this session's half-precision reduction fix and the activation-helper dtype restore; an 8-bit checkpoint keeps only its scales and biases in bf16, so it has far less surface for either. Attribution to a specific commit would need an A/B and has not been done.

With the M5 Max figure corrected, the cross-host reading is unremarkable: M5 Max leads M1 Ultra by a median of 2.77x at 32 short inputs, and both Nemotron checkpoints sit at the top of that band, 3.02x for bf16 and 3.06x for 8-bit. An earlier revision of this report read the pair against the 2026-09-06 M5 Max file, found bf16 at 1.71x, and called it a checkpoint out of step on one host. That was true of the file it was written from and is not true now.

**Rerankers need this file rather than the decode tables.** Several are built on a causal backbone, so `mlxcel-bench-decode` loads them and reports a decode rate. `qwen3-reranker-0.6b-4bit` returns 232 tok/s that way, while answering "No relevant content." to the prompt "The capital of France is" -- scoring relevance is what it was trained for. Through `/v1/rerank` the same checkpoint reads 7377 tok/s at eight documents. The two numbers differ by more than an order of magnitude and only the second one describes work anyone would ask of it. The embedding-only checkpoints do not have this failure mode, because the loader refuses them for generation and names the endpoint that serves them instead.

## Caveats

Peak `tokens_per_s` per model lands at batch 32 for text and multivector entries and at batch 8 for every rerank and VL entry. That follows from the ladder the script walks, which stops at 8 for rerank and image inputs, and is not a property of the models.

The single-input cells sit between 4 and 21 ms, where run-to-run variation is a meaningful share of the figure: a repeat of this pass moved `Nemotron-3-Embed-1B-BF16-8bit` at one short input from 1.11x slower to 1.02x slower against its bf16 sibling. Read the one-input column as an ordering rather than as a ratio; the batched cells are where the separation is stable.

The comparisons above draw on four passes at four commits: GB10 on 2026-08-26, and M5 Max at `b2ff1eee`, `a50ff440` and `66b8346e` on 2026-09-04, 09-06 and 09-09, against this host at `f42d127d`. Cross-host rows in the tables above are read as directions rather than as measured gaps.

`metal_m5max_embeddings_2026-09-06.csv` carries 20 fields per row against a 21-field header, because two of the three `writerow` calls in `bench_embeddings.py` omitted `mlx_commit` until `f42d127d`. Everything from `hardware` rightward is therefore one column left of its label in that file: the `mlx_commit` cell holds the mlxcel commit, and `mlxcel_commit` holds the build type. The timings this report draws from it sit left of the shift and are unaffected. The file is left as recorded rather than rewritten, since it is what that run produced; read its provenance columns with the offset in mind.
