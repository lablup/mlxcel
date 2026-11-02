# PR #2830: Serve image generation without a chat model

**Date**: 2026-11-02
**Status**: Completed (merged; closes #2825)

## Executive Summary

PR #2830 makes `mlxcel-server --image-model <dir>` and `mlxcel serve --image-model <dir>` start with no `-m` and serve image routes only. The mode reuses `EmbeddingServingMode` as `ImageOnly`, so health, props and capability reporting follow the existing `blocks_generation` path instead of a second mode enum. The chat surface answers the existing structured `501`, now naming `--image-model`, and `/tokenize`, `/detokenize` and `/apply-template` join it because they would otherwise run on an empty tokenizer. The image worker keeps the shared startup and exit-join path, so the #2745 teardown contract still holds.

## 1. Problem Statement

`mlxcel-server --image-model <dir>` without `-m` exited with `--model/-m is required to start the server`, although `mlxcel generate-image -m <dir>` needs no chat model. A desktop client that launches one image model therefore had to load an unrelated chat model as well. The startup structure, not generation, carried the dependency: `load_chat_front` needs a tokenizer from a checkpoint, and the shipped `MlxcelTokenizer::stub` is test-only.

## 2. Change Summary

| Area | Change |
|---|---|
| Mode | `EmbeddingServingMode::ImageOnly` (`--image-model`, `image_only`), selected when an image path is present and no model source is given; conflicts refused by `resolve_image_only_mode` |
| Both binaries | Leave `model_path` empty, skip `ensure_inert`, resolve `--image-model` or `MLXCEL_IMAGE_MODEL`, and refuse a router selector, `--embedding-model`, `--reranker-model`, `--embeddings`, or `--reranking` with an error naming both flags |
| Startup | Skip `load_chat_front`, warmup and the prompt cache; build the provider with `new_without_chat_model_with_id` on thread `model-worker-image-only`; register the image worker in the post-#2745 side-worker and exit-join path |
| Empty-path guards | `model_facts`, `detect_model_media_support`, embedding and rerank detection, generation-config defaults, context-window reads, Gemma4 family detection, props JSON reads; tensor and pipeline parallel flags refused |
| Routes | `chat_unavailable_message` gains the `ImageOnly` text; `/tokenize`, `/detokenize` and `/apply-template` gate only in this mode; `side_model_ready` returns `image_model.is_some()`; `check_serving_mode` takes `has_image` |
| `/v1/models` | Reports the image id with `["image"]` and never `completion` in both the `data` entry and the `models` block |
| Docs | `docs/image-generation.md`, `docs/environment-variables.md`, `--image-model` help in both binaries, WebUI startup label and settings fixture |

## 3. Technical Decisions

### 3.1 Extend the existing mode enum

`ImageOnly` joins `EmbeddingServingMode` rather than creating a parallel enum because `blocks_generation` already drives health, props and capability reporting. A second enum would have duplicated those decisions and drifted.

### 3.2 The served id is the alias or the directory name

The provider, `display_model_id`, and `ImageModelProvider::model_id` all return the first `--alias` when given, otherwise `side_model_id_for_path`. A `model` value copied from `/v1/models` therefore passes `check_model` for the image routes, with or without an alias.

### 3.3 Refuse ambiguous command lines instead of ignoring them

A router selector or an embedding, rerank, or generation flag combined with image-only would either be ignored silently or contradict the mode. Startup fails and names both flags, so the command line cannot claim a capability the process does not serve.

### 3.4 The chat gate is scoped to this mode

`/tokenize`, `/detokenize` and `/apply-template` had no chat gate. In image-only they would run on the empty tokenizer, so they call `chat_not_available` too. That call is conditional on `ImageOnly`, so the routes are unchanged in every other mode, including `-m --image-model`.

### 3.5 The exit path is inherited, not copied

The image worker registers through the same side-worker collector and `join_model_workers_before_exit` that #2745 introduced. The new mode does not add a second worker or a second exit path.

## 4. Verification

| Check | Result |
|---|---|
| `cargo check --lib --tests`, both binaries | pass |
| `cargo fmt --all -- --check` | pass |
| `cargo clippy -p mlxcel --lib --tests -- -D warnings` | pass |
| `cargo test --lib server::routes` | 525 passed |
| `cargo test --lib server::startup` | 105 passed |
| `cargo test --lib server::cli_input` | 164 passed |
| `cargo test --lib server::model_source` | 28 passed |
| `cargo test --lib server::runtime_settings` | 14 passed |
| `cargo test --lib tokenizer::` | 138 passed |
| `cargo test --bin mlxcel commands::serve` | 16 passed |
| `cargo test --bin mlxcel-server image_` | 3 passed |
| Both binaries built | pass |

Smoke runs for both binaries with an empty `--image-model` directory reached the image worker, logged the image-only reason, failed fatally with `failed to load --image-model ...`, and exited through the worker join. `--image-model --embeddings` and `--image-model --reranker-model` named both flags, and the no-arguments output names `--image-model`.

Two pre-existing tests overflow the default 2 MiB test-thread stack on this host (#1384), so the suites ran with `RUST_MIN_STACK=16777216`.

## 5. Unverified

No FLUX image checkpoint exists in `models/` on this host, so the issue's manual sequence (health, `/v1/models`, a decodable PNG, and a `501` on chat) was not run end to end. Route, startup and CLI smoke tests are the substitute, and the PR body records this.

## 6. Learning Points

An absent chat model is a startup-structure problem, not a generation dependency. The image worker already read only `image_model_path`; the hard dependency was `load_chat_front` and the tokenizer it needs. Naming that boundary first turned a broad feature into a set of guarded early returns.

## 7. Risks

Medium. The mode changes startup for both binaries and touches every `startup.model_path` consumer. The empty-path guards are broad, and a future consumer added without a guard could panic in this mode. The conflict refusals keep ambiguous command lines out, but there is no manual image-generation run on this host yet.
