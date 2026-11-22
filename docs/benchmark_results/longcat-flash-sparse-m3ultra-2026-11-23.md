# LongCat-Flash-Lite-Sparse: logit traces and the quantized latent (M3 Ultra, 2026-11-23)

Issue #2362. Correctness evidence for `longcat_flash_sparse` against mlx-vlm, the negative arms that show the trace measures the family, the fp16 / int8 / int4 latent comparison behind more than `index_topk` (2048) keys, and the before/after check of the shared LongCat code on `longcat_flash_ngram`.

## Setup

- **Host:** Mac Studio M3 Ultra, 512 GB, macOS 27.2.1, shared with other jobs (load average 21 to 30); every GPU run held `/Users/devops/Development/.mlxcel-gpu.lock`. These are quality runs, so load does not enter the numbers.
- **Checkpoints:** `mlx-community/LongCat-Flash-Lite-Sparse-4bit` (`models/longcat-flash-lite-sparse-4bit`; 14 layers, two MLA sub-blocks each, LSA with `index_topk` 2048, `index_init_tokens` 16, `index_local_tokens` 1024) and `longcat-flash-lite-4bit` (the ngram variant, for the regression check).
- **Corpus:** `docs/architecture.md` followed by `docs/benchmarks.md` at `a3e38aec` (36,276 tokens with the checkpoint's tokenizer).
- **Reference:** mlx-vlm 0.8.0 at `d998a620` (vendored `references/mlx-vlm`), mlx 0.32.4, through `scripts/longcat_sparse_oracle_trace.py`.
- **Candidate:** `examples/logit_trace` built from the PR branch (release, `metal,accelerate`).
- **Metric:** `scripts/compare_logit_traces.py --decided 2`: top-1 disagreements on positions where the reference's top two are at least 2 logits apart.

## Against mlx-vlm

The oracle runs one causal forward per chunk group (`64 52 8 3072`: 52 chunks of 64 traced positions, each behind up to 3072 tokens of context). The mlxcel multi-token arm uses the same plan; its traced forward is 64 wide, so the selection reaches attention as the sparse prefill mask. The decode arm (`MLXCEL_TRACE_BATCH_ROWS=1 MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1`, `64 4 8 3072`) traces 256 one-token steps behind 3072 to 3327 tokens, so every step gathers its 2048 selected latent rows; it is compared with oracle chunks 48 to 51, which cover the same positions behind the same 3072-token context.

| Candidate | Reference arm | Positions | Decided disagreements | Top-1 flips | Largest gap at a flip |
| --- | --- | --- | --- | --- | --- |
| multi-token, all chunks | mlx-vlm | 3328 | 0 / 1384 | 135 | 1.500 |
| multi-token, chunks 33 to 51 (context over 2048) | mlx-vlm | 1216 | 0 / 499 | 51 | 0.625 |
| decode (gather), fp16 latent | mlx-vlm | 256 | 0 / 81 | 7 | 0.312 |
| decode (gather), int8 latent | mlx-vlm | 256 | 0 / 81 | 16 | 0.875 |
| multi-token, chunks 33 to 51 | mlx-vlm `--ngram-fusion` | 1216 | 59 / 119 | 1040 | 5.031 |
| decode (gather), fp16 latent | mlx-vlm `--ngram-fusion` | 256 | 14 / 24 | 237 | n/a |
| multi-token, chunks 33 to 51 | mlx-vlm `--no-forced-columns` | 1216 | 0 / 494 | 54 | 0.938 |

`--ngram-fusion` swaps the reference's embedding fusion for the `longcat_flash_ngram` form `(word + proj) / (1 + n)`; the trace separates it from the real model on half the decided positions. `--no-forced-columns` clears the indexer's sink and local columns. Against the real reference it disagrees with itself on 0 of 499 decided positions behind 2048+ tokens (39 near-tie flips, the largest at 0.438), so on this corpus the forced columns change only near-ties and that arm does not separate; the forced-column rule is pinned by the unit tests (`lsa_selection_forces_sink_and_local_columns`) instead.

Perplexity behind 2048+ tokens: mlx-vlm 21.82, mlxcel 21.53 (-1.3%).

## The quantized latent

Every arm traces the same 256 positions behind 3072 tokens. Decode arms: `MLXCEL_TRACE_BATCH_ROWS=1 MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1`, `64 4 8 3072`. Multi-token arms: `MLXCEL_TRACE_START_TOKEN=3072`, `64 4 8 3072`. int8: `MLXCEL_TRACE_KV_CACHE_MODE=int8`. int4: that plus `MLXCEL_MLA_LATENT_QUANT=int4`.

| Reference | Candidate | Decided disagreements | Top-1 agreement | Largest gap at a flip | Perplexity |
| --- | --- | --- | --- | --- | --- |
| fp16 decode | int8 decode | 0 / 75 | 236 / 256 (92.2%) | 0.875 | -0.5% |
| fp16 multi-token | int8 multi-token | 0 / 77 | 238 / 256 (93.0%) | 0.625 | -1.9% |
| fp16 decode | int4 decode | 4 / 75 | 181 / 256 (70.7%) | 3.375 | +18.1% |
| fp16 multi-token | int4 multi-token | 6 / 77 | 175 / 256 (68.4%) | 3.562 | +16.3% |
| int4 multi-token | int4 decode | 0 / 71 | 233 / 256 (91.0%) | 0.750 | +2.2% |
| fp16 multi-token | fp16 decode | 0 / 77 | 248 / 256 (96.9%) | 0.125 | +0.7% |

`MLXCEL_TRACE_MLA_STATS` on the decode arms: fp16 `absorbed_composed=7168`, int8 `absorbed_fused=7168`, int4 `absorbed_composed=7168` (256 steps times 28 MLA sub-blocks), with `decompressed=112` from the context passes only. The int8 latent is in the rounding class, as on `longcat_flash_ngram` (#3161). The int4 latent is not: it is the only arm with decided disagreements, and the int4 multi-token arm (whole window dequantized, selection as a mask) agrees with the int4 gathered decode arm on every decided position, so the loss comes from the 4-bit codec and not from the gather. `mla::EXPERIMENTAL_LATENT_NOTES` quotes this result in the `--mla-latent-quant int4` startup warning for this family; the default stays fp16 and `--kv-cache-mode int8` stores the 8-bit latent.

## Generation

Greedy `mlxcel generate --show-reasoning` on `longcat-flash-lite-sparse-4bit`. The checkpoint's template opens a `<longcat_think>` block by default, so without `--show-reasoning` (or `chat_template_kwargs: {"enable_thinking": false}` on the server) a short budget lands entirely in the reasoning channel.

- `Explain multi-head latent attention.`, 300 tokens: a structured, correct explanation (single-head attention, the multi-head concatenation and projections).
- A 3,121-token prompt (the first 12,000 characters of `docs/architecture.md` plus a summary question), 400 tokens, at fp16, `--kv-cache-mode int8` and `--kv-cache-mode int8 --mla-latent-quant int4`: all three summarize the document accurately (thin entry points, the `src/` layout); the fp16 and int8 texts share their first 100 or so tokens. 68.8, 71.0 and 49.3 tok/s including the prefill on the loaded host.
- `mlxcel-server`: a short chat request and the same 3,129-token prompt with `enable_thinking: false` return coherent content (`Paris is the capital of France.`; a correct summary of the architecture document with the source tree), the long one stopping at its 300-token limit.

## `longcat_flash_ngram` before and after

The PR splits `LongcatFlashMLA::forward` into `project` and `attend_window` and moves the n-gram hashing into helpers. `logit_trace` built at `a3e38aec` and at the PR branch, on `longcat-flash-lite-4bit`:

| Shape | Positions | Result |
| --- | --- | --- |
| `64 2 8 2048` (multi-token) | 128 | rows byte-identical |
| `32 2 8 2048`, one-token decode | 64 | rows byte-identical |
| `64 2 8 2048` with `MLXCEL_TRACE_START_TOKEN=2048` | 128 | rows byte-identical |

## Reproduce

```bash
S=scratch; M=models/longcat-flash-lite-sparse-4bit
cat docs/architecture.md docs/benchmarks.md > $S/corpus.txt
PYTHONPATH=references/mlx-vlm python scripts/longcat_sparse_oracle_trace.py $M $S/corpus.txt 64 52 8 3072 > ref.tsv
PYTHONPATH=references/mlx-vlm python scripts/longcat_sparse_oracle_trace.py $M $S/corpus.txt 64 52 8 3072 --ngram-fusion > ref-ngram.tsv
./target/release/examples/logit_trace $M $S/corpus.txt 64 52 8 3072 > multi.tsv
MLXCEL_TRACE_BATCH_ROWS=1 MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1 ./target/release/examples/logit_trace $M $S/corpus.txt 64 4 8 3072 > dec-fp16.tsv
MLXCEL_TRACE_KV_CACHE_MODE=int8 MLXCEL_TRACE_BATCH_ROWS=1 MLXCEL_LOGIT_TRACE_SCALAR_REFERENCE=1 ./target/release/examples/logit_trace $M $S/corpus.txt 64 4 8 3072 > dec-int8.tsv
python3 scripts/compare_logit_traces.py ref.tsv multi.tsv --decided 2
python3 scripts/compare_logit_traces.py dec-fp16.tsv dec-int8.tsv --decided 2
```

The decode traces number their chunks from 0; shift them by 48 to line them up with the oracle's chunks 48 to 51.
