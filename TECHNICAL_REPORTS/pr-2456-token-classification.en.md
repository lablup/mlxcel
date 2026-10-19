# PR #2456: Token classification and span extraction (mlxcel tag, mlxcel extract)

**Date**: 2026-10-19
**Status**: Implemented and verified on CUDA (GB10) with the Rampart checkpoints; PR open

## Summary

Closes #2383, the foundation issue of epic #2391. Adds a per-token output surface next to the existing generate, embed and rerank paths and lands it end to end with a BERT / XLM-RoBERTa token-classification head. `mlxcel tag` tags and redacts entities with Rampart and matches mlx-vlm span for span. `mlxcel extract` carries the span-extraction contract that GLiNER2.5 (#2390) will implement. The OpenAI Privacy Filter (#2387) will implement `TokenClassifierModel` on top of the same pipeline.

## Problem

mlxcel had no way to return one prediction per input token. Every model head ended in a vocabulary projection (generate), a pooled vector (embed) or a pair score (rerank). Token classifiers such as Rampart and span extractors such as GLiNER have no place to plug in, and mlx-vlm already ships `mlx_vlm.token_classification` for the same checkpoints.

## Technical decisions

- **Two traits, one pipeline.** `TokenClassifierModel` returns per-position log-probabilities for one wrapped window. The windowed `TokenClassifier<M>` pipeline owns tokenization, windowing, special-row stripping, an f32 log-softmax, decoding and span assembly, so a model only has to produce logits. `SpanExtractorModel` is a separate trait for models that return entities directly (GLiNER), with `ExtractOptions`, `EntityResult` and `ClassifyTask` matching upstream's JSON shapes.
- **Decode modes are declared by the model.** `decode_modes()[0]` is the default, `--argmax` forces `Argmax`, and a mode the model does not list is refused. This leaves room for the Viterbi decoder in #2387 without changing the trait.
- **Upstream span helpers ported one to one.** `spans.rs` ports upstream's label parsing (BIO and BIOES), span building, word grouping, punctuation merging, trimming, placeholder and redaction helpers, with offsets in Unicode scalars so multi-byte text lines up with upstream.
- **Config-only detection with an extension table.** `loader.rs` detects the family from `config.json` into `TokenModelKind`, one variant plus one detector arm per family, so #2387 and #2390 each add one row. The `id2label` width against the classifier tensor is checked at load, since detection sees only the config.
- **Existing routing untouched.** `get_model_type` still detects Rampart as the BERT embedder, so `mlxcel embed` and rerank behave exactly as before; `tag` and `extract` use their own detection. A checkpoint of the wrong shape fails with an error naming the right subcommand.
- **XLM-RoBERTa context.** The pipeline uses the addressable position count (`max_position_embeddings - pad - 1`) as the window size. Upstream's raw `max_position_embeddings` would index past the position table.
- **Review hardening.** `redact` makes one forward pass over the sorted, non-overlapping spans that `decode_spans` produces, instead of a cost that grows with text length times span count; any other span list keeps upstream's right-to-left splice. `BertTokenClassifier` refuses a position table shorter than `max_position_embeddings`, and the tagger refuses a token id outside the word-embedding table. The docs state that `text` and `spans[].text` in the JSON output are not redacted.

## Changes

- `src/token_classification/`: `mod.rs` (traits, `DecodeMode`, windowed pipeline), `spans.rs`, `extract.rs`, `loader.rs`, `bert.rs`, and tests (`spans_tests.rs`, `mod_tests.rs`, `loader_tests.rs`, `extract_tests.rs`, `real_checkpoint_tests.rs`).
- `src/models/bert_heads.rs`: `BertTokenClassifier`, the encoder plus a per-position `classifier` projection; `bert_heads_tests.rs`, `bert_tests.rs`.
- `src/commands/tag.rs`, `src/commands/extract.rs`, `src/commands/mod.rs`, `src/main.rs`, `src/lib.rs`: the two subcommands wired into the `mlxcel` binary.
- `tests/fixtures/token_classification/`: mlx-vlm parity fixture, its generator and the Rampart config.
- Docs: new `docs/token-classification.md`, a section in `docs/supported-models.md`, README quick tour, docs index, CHANGELOG.

## Verification

GB10, `--profile test-fast --features cuda`, test binaries under `gpu-lock` with `--test-threads=1` and `MLX_ENABLE_TF32=0`.

| Check | Result |
|---|---|
| lib tests, filter `token_classification` | 43 passed |
| lib tests, filter `bert` (new head plus unchanged embed and rerank tests) | 56 passed |
| bin tests, filters `extract` and `tag` | 6 passed |
| `mlxcel tag -m models/mlx/rampart-mlx-fp16 --keep CITY,STATE,ZIP_CODE ...` | `I'm <GIVEN_NAME> <SURNAME>, <BUILDING_NUMBER> <STREET_NAME>, Los Angeles CA 90012, <EMAIL>`, the model card output |
| same with `rampart-mlx-4bit` | same 8 spans, same offsets |
| mlx-vlm parity fixture (mlx-vlm 0.7.8 on mlx CPU) | exact match on labels, offsets, text and redaction: 23, 12 and 42 tokens (8, 3 and 6 spans) plus a 706-token two-window case (81 spans) with the window boundary inside a word |
| clippy (`-p mlxcel --lib --tests --no-deps -D warnings`), fmt, contract scripts | pass |

The fixture generator rejects any case whose top-1 / top-2 gap is under 0.05. An earlier long case built from a repeated template had an exact tie on one token that CUDA broke the other way from mlx CPU, so the long case was rebuilt to be tie-free. Logit traces do not apply: this is a new path and no existing output moves.

## Follow-ups

- Low review findings left as is: `--keep` trimming, `-` reading stdin, `--classify` accepting unknown fields, special-token text in the input being tokenized as a special token (upstream behaves the same), unescaped `model_type` on stderr.
- #2387 (OpenAI Privacy Filter) and #2390 (GLiNER2.5) implement against this contract.

## Statistics

| Item | Value |
|------|-------|
| Files changed | 27 |
| Lines added | 4776 |
| Lines deleted | 18 |

| Commit | Message |
|--------|---------|
| `0f3b5185` | feat(token-classification): add per-token output module and BERT head |
| `8293d1d1` | feat(cli): wire mlxcel tag and mlxcel extract, add tests and docs |
| `5e71679d` | docs: add mlxcel tag to the README quick tour |
| `11b3829c` | fix(token-classification): annotate the grouped path slice type |
| `c62a6448` | test(token-classification): stop shadowing the path helper |
| `999cadea` | test(token-classification): make the multi-window parity case tie-free |
| `6d308b18` | style(token-classification): apply rustfmt |
| `be088772` | fix(token-classification): bound redaction cost and embedding-table reads |

## Related

- Epic #2391 (token classification and span extraction).
- #2387 (OpenAI Privacy Filter) and #2390 (GLiNER2.5 with DeBERTa-v2) build on this PR.
