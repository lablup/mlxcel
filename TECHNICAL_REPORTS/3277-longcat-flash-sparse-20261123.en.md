# PR #3277: Add LongCat-Flash-Lite-Sparse (`longcat_flash_sparse`)

**Date**: 2026-11-23
**Status**: PR open as a draft (closes #2362). Unit suites pass serially; the 4-bit checkpoint is validated against mlx-vlm behind more than 2048 tokens, at fp16, int8 and int4 latent.
**Risk Level**: Medium (shared LongCat MLA code was refactored; the ngram family's traces are byte-identical before and after)

## Executive Summary

LongCat-Flash-Lite-Sparse is the LongCat ScMoE decoder with two changes that the existing `longcat_flash_ngram` loader cannot express: LSA sparse attention, where a lightning indexer picks 2048 keys per query, and an n-gram embedding fusion that keeps the word embedding at full scale. Before this PR the checkpoint failed with `Unsupported model type`; loading it as the ngram family would have divided the whole embedding by 13 and attended densely.

This PR adds `models::longcat_flash_sparse`, which reuses the shared LongCat MLA, MLP and MoE through a split of `LongcatFlashMLA::forward` into `project` and `attend_window`. The indexer's selection reaches attention as a row gather on decode and as an additive mask on multi-token steps. Following the maintainer decision of 2026-11-23, the family stores the quantized MLA latent: fp16 stays the default, `--kv-cache-mode int8` stores the 8-bit latent (0 decided-position disagreements against fp16), and `--mla-latent-quant int4` is accepted but stays experimental (4 to 6 decided disagreements, +15% to +23% perplexity).

## 1. Problem Statement

`mlx-community/LongCat-Flash-Lite-Sparse-{4,6,8}bit` declare `model_type: longcat_flash_sparse`; the original Meituan checkpoint declares no `model_type` at all, only `architectures: ["LongcatCausalLM"]`. Both were refused. The architecture differs from `longcat_flash_ngram` in three places that each produce fluent-looking garbage if missed:

- **Fusion.** Upstream returns `word + proj / (1 + n)`. The ngram family returns `(word + proj) / (1 + n)`. With `n = 12`, the wrong form shrinks every token embedding 13 times. In the trace this arm disagrees with the real reference on 59 of 119 decided positions.
- **LSA.** `self_attn.0` of each layer owns an indexer: `score = sum_h relu(q_h . k) * weights_proj(x)_h * H^-0.5 * D^-0.5`. The first 16 columns and a 1024-key window ending at the query are forced to `+inf`, the causal mask applies, and `argpartition` keeps the top 2048. `self_attn.1` reuses the same selection. Below 2048 cached keys nothing is selected, so a short test cannot tell sparse from dense.
- **RoPE.** `deepseek_yarn` with factor 120 over 8192 positions. The MLA rotates traditionally; the indexer follows `indexer_rope_interleave` over its first 64 of 128 dims.

## 2. Change Summary

| Area | Change |
| --- | --- |
| Model | New `src/models/longcat_flash_sparse.rs` (config, indexer, sparse MLA, decoder layer, n-gram embedding, model, sanitize). |
| Shared LongCat | `LongcatFlashMLA::forward` split into `project`, `absorbed`, `geometry` and `attend_window`; n-gram hashing moved into `NgramEmbedding::hashed_ids` and `advance_context`; visibility widened to `pub(super)`. Behavior unchanged. |
| Detection | `longcat_flash_sparse` arm; a config with no `model_type` and `LongcatCausalLM` resolves to it. |
| Wiring | `ModelType`, registry (`mlxcel arch`), metadata, `LoadedModel`, the special weight loader, memory-estimate label, TP replicated list, the server's hybrid-SSM prompt-cache list. |
| KV | Listed in `QUANTIZED_LATENT_FAMILIES` and `MLA_LATENT_CACHE_FAMILIES`. `kv_arch` counts the third, always-fp16 indexer-key cache. A `longcat_flash_sparse` int4 note in `EXPERIMENTAL_LATENT_NOTES`. |
| Tests | 15 model tests plus additions to detection, registry, rope-lift, special loader, kv_arch, turbo allowlist, MLA cache, `mla_latent_option_tests`, `turbo_args_tests` and the hybrid SSM tests. |
| Tooling | `scripts/longcat_sparse_oracle_trace.py` (mlx-vlm reference arm with `--ngram-fusion` and `--no-forced-columns` negative arms). |
| Docs | `docs/supported-models.md` entry and KV table, `docs/turbo-kv-cache.md`, results doc `docs/benchmark_results/longcat-flash-sparse-m3ultra-2026-11-23.md`, CHANGELOG. |

## 3. Technical Decisions

### 3.1 Reuse the shared MLA by splitting it, not by copying it

The sparse MLA needs the same projections, the same latent cache write and the same attention body as the ngram MLA. Only the window it attends over (gathered rows on decode) and the mask (sparse on prefill) differ. Copying `LongcatFlashMLA` would have duplicated about 300 lines, including the quantized-latent dispatch that #3161 and #3189 tuned. Splitting `forward` into `project` and `attend_window` lets the sparse module insert the selection between them while the ngram path calls the same pieces in the same order. The cost is a refactor of shared code, so the ngram checkpoint was traced before and after at three shapes; the rows are byte-identical.

### 3.2 Gather on decode, mask on prefill

Upstream gathers the selected latent rows for a single query and builds a boolean sparse mask for several. Both are kept. The gather goes through `LatentWindow::gather_rows`, which DeepSeek V3.2's DSA decode already uses. It gathers the packed latent, its scales and biases and the rope stream together, so a quantized window stays quantized and the absorbed quantized kernel reads it directly. That is how "the LSA decode gather reads the active latent layout" is met without a dequantize step. The prefill mask is additive (`0` kept, `-inf` dropped) and folded with the causal condition, because `attend_window` takes an additive `FoldedMask::Array`.

### 3.3 Quantized latent: list the family, keep fp16 the default

The maintainer asked for the quantized latent modes to be accepted rather than refused. Listing the family in `QUANTIZED_LATENT_FAMILIES` makes `--kv-cache-mode int8` store the 8-bit latent, as it does on the other LongCat types. The default KV mode is fp16, so nothing changes for a user who passes no flag. The indexer keys get their own cache that stays fp16 in every mode: they decide the selection, and quantizing them would move which keys are attended rather than how precisely.

| Latent | Decided disagreements vs fp16 (decode, 3072 context) | Decision |
| --- | --- | --- |
| int8 | 0 / 78 | Shipped under `--kv-cache-mode int8` |
| int4 | 4 / 78, perplexity +23% | Accepted with the experimental warning, which now quotes this measurement |

### 3.4 Review fixes

The reviewers found no correctness defect, and three robustness gaps that are fixed in `dd5e0011`:

- **Indexer memory and precision.** The first version scored every head at once in the activation dtype, which on a long prompt materializes `[B, 16, L, S]` three times per indexer layer (about 25 GB transient for a 2048-row chunk at 128k keys). Scores are now float32, as upstream's Metal kernel computes them, and a step above 2^24 score elements adds the heads one at a time into one `[B, 1, L, S]` buffer. The sparse prefill mask is built once per layer, directly in the mask dtype, and shared by both sub-blocks.
- **Abort paths at load.** The cxx ops do not return errors, so a weight whose shape the config does not describe used to abort the process in the first forward (or, for a one-row `weights_proj`, broadcast into a wrong selection). The indexer projections, `k_norm`, the fused `post_proj` and the per-embedder `oe_embed_proj` parts are now checked at load.
- **Configs without `model_type`.** Detection already mapped `LongcatCausalLM` to the family, but KV mode resolution and the prompt-cache hybrid check keyed on `model_type` alone, so the original checkpoint would have had its int8 request refused and APC left on. Both now fall back on the architecture.

All trace arms were rerun on the fixed build. Every decided-position count is unchanged; the near-tie flip counts moved by a few positions, which is what float32 scoring of the selection should do.

### 3.5 Which negative arm separates

The issue asked for a failing negative arm: forced columns off, or the ngram fusion. Both were run. Turning the forced columns off moves only near-ties on this corpus (39 flips, 0 decided), because the indexer already ranks the sink and the local window highly. The ngram fusion fails on half the decided positions. The trace is therefore gated with the fusion arm, and the forced-column rule is pinned by a unit test whose scores rank the sink and local columns last.

## 4. Verification

All numbers are from `scripts/compare_logit_traces.py --decided 2` on `longcat-flash-lite-sparse-4bit` (M3 Ultra, mlx-vlm 0.8.0 at `d998a620`). The corpus is `docs/architecture.md` followed by `docs/benchmarks.md`.

| Arms | Positions | Decided disagreements |
| --- | --- | --- |
| mlx-vlm vs mlxcel, 64-wide forwards, context over 2048 | 1216 | 0 / 499 |
| mlx-vlm vs mlxcel, one-token decode behind 3072 (gather) | 256 | 0 / 81 |
| mlx-vlm `--ngram-fusion` vs mlxcel | 1216 | 59 / 119 |
| fp16 vs int8 latent, decode | 256 | 0 / 78 |
| fp16 vs int8 latent, multi-token | 256 | 0 / 78 |
| fp16 vs int4 latent, decode | 256 | 4 / 78 |
| fp16 vs int4 latent, multi-token | 256 | 6 / 78 |
| int4 multi-token vs int4 gathered decode | 256 | 0 / 70 |

The last row separates the codec from the gather: with the same 4-bit rows, attending the dequantized window under a mask and attending the gathered packed rows agree. MLA dispatch counters on the decode arms show `absorbed_composed=7168` (fp16, int4) and `absorbed_fused=7168` (int8). That is 256 steps times 28 sub-blocks, and no decode step was decompressed.

- `longcat_flash_ngram` before/after: rows byte-identical at `64 2 8 2048`, at one-token decode, and behind 2048 tokens with `MLXCEL_TRACE_START_TOKEN`.
- `mlxcel generate` on a 3,121-token prompt gives accurate summaries at fp16, int8 and int4 (fp16 rechecked after the review fixes). `mlxcel-server` gives coherent content for a short and a 3,129-token chat request with `enable_thinking: false`.
- Unit suites, run serially as CI does: `longcat_flash` (39), `coverage_tests`, `mla_latent_option`, `turbo_args`, registry, detection, kv_arch, config_fields, special loader, rope lift, hybrid SSM, core `mla::` (117). Clippy `-D warnings`, fmt and every contract check pass.

## 5. Unverified and Open Items

- MTP heads (`model.mtp*`) are dropped, and the fused indexer-score Metal kernel is not ported (scores are plain MLX ops). Batching and tensor parallelism are out of scope; the model is listed as replicated for TP.
- Not done from the review: a keys-only indexer cache (the fp16 cache stores a one-column placeholder value, 1/129 extra), caching the YaRN `mscale` scalar in `SparseRope`, and sharing one chunked scoring helper with `deepseek_v32_indexer`, which has the same full-head score tensor.
- Only the 4-bit conversion was run; the 6-bit, 8-bit and original bf16 checkpoints were not.
- In a parallel test run, two `longcat_flash_ngram` latent tests fail on counter deltas, because `mla::stats` is process-global. They pass serially, which is how CI runs Metal tests. They fail the same way when only the eight ngram latent tests are selected, so the race is among the existing tests and not caused by this PR.
- Found, not fixed: `LongcatFlashMLA::forward` ignores the ngram checkpoint's YaRN `rope_scaling` (factor 10 over 32768, theta 5e6) and rotates with plain base theta.
- The checkpoint's chat template opens a `<longcat_think>` block by default, so a short request without `enable_thinking: false` returns its text in `reasoning_content`.

## 6. Learning Points

- **A sparse attention check needs a context past the selection budget.** Below `index_topk` keys, LSA is dense attention. A trace or test at a few hundred tokens passes with the indexer broken or missing. The unit fixture sets `index_topk = 4` so the selection is active from the fifth key. The real traces start at 2048 to 3072 tokens.
- **A negative arm has to be shown to fail.** The forced-columns arm looked like the obvious control and turned out not to separate on this corpus. Running it is what showed that the fusion arm is the one to gate on.
- **Counters shared across threads are only valid in a serial run.** `mla::stats` is a set of process-global atomics. Its assertions hold under `--test-threads=1` and are noise otherwise.
- **Check shapes at load in front of FFI ops that cannot fail.** A shape mismatch in a reshape or matmul becomes a C++ exception and `std::terminate`, so a check deferred to the first forward takes the whole server down.

## Related Commits

| Hash | Message |
| --- | --- |
| `adb9e900` | feat(models): add LongCat-Flash-Lite-Sparse (longcat_flash_sparse) |
| `8c3b9531` | docs(models): record LongCat-Flash-Lite-Sparse traces and int4 note |
| `dd5e0011` | fix(models): harden the LongCat-Flash-Lite-Sparse indexer and config paths |
