# Technical Report: PR #1495 - Router mode and the models-dir migration

**Date**: 2026-08-28

**Status**: Completed

**Languages**: Rust

**Risk Level**: High (breaking flag semantics)

## Executive Summary

Implements b10621 router mode (#1438) as an in-process model pool and resolves the `--models-dir` semantic collision: on the two llama-server surfaces (`mlxcel-server`, `mlxcel serve`) the flag now selects router-mode model discovery, and the old mlxcel store-root meaning moved to `--model-store-root` with a startup migration diagnostic. Seven manifest entries flip to `supported`; five stay `deferred` at an honest boundary, so issue #1438 remains open.

## 1. Problem Statement

b10621's `--models-dir` starts a router server that discovers models in a directory, spawns one child llama-server per loaded model, proxies requests by the request's `model` field, and manages the set through `POST /models/load|unload`, `DELETE /models`, `GET /models/sse`, `--models-max`, and `--models-autoload`. mlxcel used the same spelling for its local model-store root, so a copied llama-server command line silently reconfigured the downloader instead of starting a router, the exact silent-divergence class epic #1431 exists to remove.

## 2. Technical Decisions

### 2.1 In-process pool instead of child processes

Where upstream proxies over child sockets, each mlxcel pool entry owns a full `AppState` plus its axum sub-app, and the dispatcher forwards the rebuilt request in-process. This reuses the entire existing route stack per model unchanged, keeps the HTTP contract (upstream's exact refusals for missing/unknown/not-loaded names, `?autoload=` override, `?model=` GET proxying) while making unload graceful by refcount: in-flight requests keep their sub-app alive until they finish, then the worker thread exits on channel disconnect and the weights free, verified by RSS. Sub-apps are built without a CORS layer (`create_app_without_cors`) so the router's top level answers preflights and stamps CORS headers exactly once; API keys are enforced at the top level with the same public-path rule.

### 2.2 A refusal is better than a silent meaning change

Combining `--models-dir` with a model argument, the old store-root use, fails startup with a diagnostic naming `--model-store-root` and `MLXCEL_MODELS_DIR`, where b10621 would accept the flag as inert. Recorded as a divergence on the entry rather than hidden: silently resolving repo-ids from a different root would be worse than the refusal. `--models-preset` likewise parses and refuses at startup, because serving un-preset models while the operator believes INI presets apply is the accepted-and-ignored failure the epic forbids.

### 2.3 Confinement at discovery, not at request time

Model names come only from scanning the models directory (one direct subdirectory holding `config.json` per model); a symlink whose canonical path escapes the canonical root is skipped at scan time, and request `model` values resolve exclusively through the registry, so no request can smuggle a path.

### 2.4 Single-model `/v1/models` parity

The single-model `GET /models` / `GET /v1/models` answer moved to b10621's shape: `aliases` (the whole `--alias` list), `tags` (new `--tags` / `LLAMA_ARG_TAGS`), `owned_by: "llamacpp"`, a `meta` facts block derived from `config.json` and the safetensors headers (quantized `U32` payloads unpacked by declared bit width for `n_params`), and upstream's Ollama-compatible `models` block with `format: "safetensors"`.

## 3. Change Summary

| Item | Value |
|------|-------|
| Files changed | 25 |
| Lines | +2882 / -152 |
| Manifest entries | supported: --models-max, --models-autoload, --tags, POST /models/load, POST /models/unload, GET /models, GET /v1/models; deferred: --models-dir, --models-preset, POST /models, DELETE /models, GET /models/sse |

Validation: 20+ new unit/route tests (discovery incl. symlink-escape skip, refusal parity, failed-load SSE events, management-route authorization, path-shaped names refused) and two real checkpoints through discovery, autoload, alternating requests, `--models-max 1` LRU eviction, load/unload with SSE events, `?model=` GET proxying, an unload issued mid-generation completing all 150 in-flight frames, restart recovery, and the migration diagnostic.

## 4. Follow-up Actions

- INI preset translation (`--models-preset`), the `POST /models` download-into-cache flow, cache-sourced `DELETE /models`, and SSE payload parity (child `info`/`progress` blocks) remain on #1438.
- #1439 (multi-adapter LoRA) integrates with this pool rather than duplicating adapter state, per its own scope.
