# PR #2461: GLiNER2.5 span extraction on a DeBERTa-v2 encoder

**Date**: 2026-10-20
**Status**: Implemented and verified on CUDA (GB10) with `gliner2.5-small-v1`; PR open

## Summary

Closes #2390, the last sub-issue of epic #2391. Adds GLiNER2.5 (Fastino), a DeBERTa-v3 encoder with a boundary head, as the first `mlxcel extract` family. It implements `SpanExtractorModel` from #2383 (PR #2456) and adds a reusable DeBERTa-v2 encoder that later DeBERTa rerankers or embedders can share.

## Problem

`mlxcel extract` refused every checkpoint because no span-extraction family existed. GLiNER2.5 needs a DeBERTa-v2 encoder (disentangled content-to-position and position-to-content attention with log-bucketed relative positions), which mlxcel did not have, plus the GLiNER schema prompt, word splitting, candidate pool, boundary scorer and overlap resolution. The live `fastino/gliner2.5-small-v1` config declares `model_type: gliner2` and `Gliner2ForSchemaExtraction`, which mlx-vlm's loader does not map.

## Technical decisions

- **Reusable encoder.** `src/models/deberta_v2.rs` and `deberta_v2_config.rs` hold a GLiNER-free DeBERTa-v2 encoder. Relative positions follow Hugging Face log bucketing (`max_relative_positions < 1` falls back to `max_position_embeddings`). mlx-vlm leaves them unbucketed and clips them, so its hidden states differ from `transformers` by up to 7e-2 past 128 tokens; mlxcel matches `transformers` instead (max 5.7e-6 on 306 tokens). Unported options are refused at config time, `share_att_key` defaults to Hugging Face's `false`, and the legacy `"p2c|c2p"` string is accepted.
- **GLiNER2.5 port.** `src/token_classification/gliner2_5*.rs` ports mlx-vlm `gliner2_5` (`gliner2_5.py`, `boundary.py`): schema prompt with `[P]`, `[E]`, `[L]`, `[DESCRIPTION]` and `[SEP_TEXT]`, upstream's word regex, shared candidate pool, boundary scorer, and the weighted-interval overlap DP (`_resolve_flat_overlaps`) in f64. Detection accepts `gliner2` or `gliner2_5` and either architecture; any head other than `boundary` is refused.
- **Contract kept.** One `TokenModelKind::Gliner2_5` variant, one detector and one `load_span_extractor` arm. The #2383 traits are unchanged. `bert::Activation::{from_name, apply}` became `pub(crate)` for reuse.
- **Stored dtype.** `gliner2.5-small-v1` ships float32 tensors despite `dtype: float16` in `encoder_config`. mlx-vlm runs it in float32, so mlxcel keeps the stored dtype.
- **CUDA graph per layer.** With the 12-layer encoder captured as one CUDA graph, the first execution of a fresh capture intermittently returned wrong hidden states on GB10 (about half of the processes on a 306-token input, error 1e-2 to 9e-1, replays correct; `MLX_USE_CUDA_GRAPHS=0` gave 0/80). The disentangled position gathers are the trigger, and evaluating per layer gave 0/192. `DebertaV2Encoder` therefore ends a graph after every layer on CUDA; Metal is untouched. A test asserts that an encode and its replay agree bit for bit.
- **Review hardening.** Word characters follow Python's `\w` (`[\p{L}\p{N}_]`). The e-mail branch is bounded to RFC 5321 lengths because upstream's is quadratic. Config sizes are validated as `i32` with ceilings (`pool_boundary_top_k` <= 1024, `max_len` <= 32768). The pool is padded to `min(pool_size, top_k^2)`. bos/eos and the relative table width are checked at load. Exports converted by mlx-vlm load through a reverse rename in `sanitize`.

## Changes

- `src/models/deberta_v2.rs`, `deberta_v2_config.rs`, `deberta_v2_tests.rs` (new); `bert.rs`, `detection.rs`, `mod.rs`.
- `src/token_classification/gliner2_5.rs`, `_prepare.rs`, `_head.rs`, `_layers.rs`, `_pool.rs`, `_scorer.rs` and their tests (new); `loader.rs`, `mod.rs`.
- `src/main.rs` (`mlxcel extract --help` examples).
- `docs/token-classification.md` now declines an HTTP route for both shapes explicitly (upstream defines none), which closes the last acceptance criterion of epic #2391.
- `tests/fixtures/token_classification/`: mlx-vlm GLiNER2.5 fixture, DeBERTa-v2 layer fixture, `transformers` long-input fixture, and the generator (mlx-vlm 0.7.8 git `73d366ec`, mlx 0.32.4, transformers 5.20.1, torch 2.14.1+cpu).
- `docs/token-classification.md` (GLiNER2.5 section and README examples), `docs/supported-models.md`, `CHANGELOG.md`.

## Verification

GB10, `--profile test-fast --features cuda`, `MLX_ENABLE_TF32=0`, test binaries under `gpu-lock` with `--test-threads=1`, rerun by the orchestrator on the final head: `deberta_v2` 7, `gliner` 22, `token_classification::` 86, `rerank::` 51, `models::detection` 83, `models::bert` 31, `privacy_filter` 20, all passing. Bin tests `extract` 5 and `tag` 4 pass. Clippy, fmt and the repository contract checks are clean.

| Check | Result |
|---|---|
| `extract --labels company,person,location --include-confidence --include-spans "Apple hired Sam in New York."` | Apple 0.9993 [0,5], Sam 0.9905 [12,15], New York 0.9995 [19,27] |
| `--classify '{"sentiment": ["positive","neutral","negative"]}' "The launch was excellent."` | `positive` |
| `--labels-json '{"person": "a human name", "company": null}' "Ada Lovelace worked with Charles Babbage."` | byte-identical to mlx-vlm |
| mlx-vlm fixture, 7 cases (descriptions, e-mail/URL, CJK `char` splitter, classification) | ids, positions, spans and offsets exact; confidences within 1e-6 |
| DeBERTa-v2 encoder vs `transformers`, 306 tokens | max abs 5.7e-6, encode == replay bit-exact |
| 259-token GLiNER input, 4 rounds | deterministic |
| `mlxcel generate -m <gliner>` | `GLiNER2.5 is a span extractor; use \`mlxcel extract\`` |

With `MLX_ENABLE_TF32=1` the GLiNER outputs still match (confidences move about 1e-5), while the DeBERTa row gates become tolerance questions (2.7e-3 against a 2e-3 gate). No logit trace applies: the model is an encoder-only span extractor with no decode path.

## Follow-ups

- The CUDA graph first-execution error is worth its own issue: other encoders that gather position embeddings inside one large graph may hit it. The per-layer graph split is a workaround, not a root cause.
- `gliner2.5-base-v1` and `-multi-v1` are not on this host.
- No CLI flag for `max_len` or the `char` word splitter; both exist on `Gliner2_5Model`.
- Relation and record extraction, DeBERTa rerankers and embedders, and an HTTP route are out of scope per the issue.

## Statistics

| Item | Value |
|------|-------|
| Files changed | 25 |
| Lines added | 19175 |
| Lines deleted | 28 |

Most added lines are JSON fixtures (about 14900 lines).

| Commit | Message |
|--------|---------|
| `2690e4f1` | feat(models): support GLiNER2.5 span extraction on a DeBERTa-v2 encoder |
| `e92a73ab` | fix(models): harden GLiNER2.5 and DeBERTa-v2 after review |
| `2e4f125b` | test(models): cover GLiNER2.5 prepare limits, document load ceilings |
| `f9c4ec71` | docs: decline an HTTP route for token classification and span extraction |

## Related

- Epic #2391; built on #2383 (PR #2456); sibling #2387 (PR #2460, OpenAI Privacy Filter).
