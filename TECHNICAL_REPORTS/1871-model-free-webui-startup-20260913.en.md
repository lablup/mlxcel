# Technical Report: PR #1871 - feat: enable model-free WebUI startup

**Date**: 2026-09-13
**Status**: Completed
**Languages**: Rust, TypeScript contract fixtures, Markdown
**Risk Level**: High

## Executive Summary

PR #1871 turns the previously bundled WebUI shell into an opt-in production startup mode for both `mlxcel-server --webui` and `mlxcel serve --webui`. It starts without a model argument, mounts the shell at `{api_prefix}/webui/`, protects typed UI APIs with the shared WebUI security wrapper, and keeps catalog/runtime/event observation model-free until an explicit user action loads a model.

## 1. Problem Statement

### 1.1 Background

The WebUI epic had already landed the schema, static bundle, lifecycle coordinator, security wrapper, and catalog projection pieces, but users still could not start a real server with `--webui` and no `-m`. The remaining integration problem was to compose those pieces through the actual startup paths without bypassing llama-server compatibility semantics or accidentally loading checkpoints during UI observation.

### 1.2 Existing Issues

- **Model-required startup**: both server entry paths still assumed a startup checkpoint unless router-mode flags were present.
- **Split WebUI route semantics**: router and single-model event endpoints needed to share cursor validation, replay behavior, and SSE serialization rather than drifting.
- **Security-sensitive startup policy**: loopback, non-loopback, generated key, TLS, and explicit root validation had to be enforced before serving administrative UI APIs.
- **Documentation drift**: WebUI bundle and catalog docs still described production startup as future work after the integration landed.

### 1.3 Risk Assessment

| Risk | Impact | Likelihood |
|------|--------|------------|
| Starting WebUI loads a checkpoint or touches the network unexpectedly | High | Medium |
| UI APIs bypass shared WebUI security middleware | High | Medium |
| Router runtime snapshots report global defaults instead of selected model settings | Medium | Medium |
| Stale docs tell operators that `--webui` is unsupported | Medium | High |

## 2. Technical Decisions

### 2.1 Reuse existing router/provider authorities

**Context:** The WebUI needs model discovery, lifecycle actions, runtime settings, and event streams, but duplicating those authorities would create inconsistent state.

**Decision:** The implementation mounts WebUI adapters over `RouterPool`, `AppState`, `LifecycleCoordinator`, and the existing catalog projection cache. Single-model mode uses the existing loaded provider and exposes it through the cache-aware catalog handoff rather than registering a second provider.

**Trade-off:** The WebUI inherits router and provider constraints, so later page work must use the typed API instead of private shortcuts. This keeps operational behavior truthful and testable.

### 2.2 Keep shell public but UI APIs authenticated

**Context:** The browser must be able to fetch static assets, while control and observation APIs are administrative.

**Decision:** The production app wraps static routes plus UI APIs in `secure_webui_router`, with public shell routes and private `/ui-api/v1` routes under the same Host, Origin, Fetch-Metadata, query-credential, and rate/body-limit checks.

**Trade-off:** Local loopback sessions need a generated terminal key when no operator key is configured. That is intentional to avoid putting credentials in URLs or HTML while preserving a convenient local startup path.

### 2.3 Share event replay implementation

**Context:** Review found that single-model SSE had copied a simplified subscription path that ignored cursor headers and paired query replay parameters.

**Decision:** PR #1871 extracts shared WebUI event parsing and SSE response construction into `src/server/webui/events.rs`, and both router and single-model routes call it. The tests compare actual mounted SSE payloads against normalized fixtures and cover paired cursor conflicts and future-sequence rejection.

**Trade-off:** The helper is WebUI-feature gated, so no-default-feature builds need explicit cfg guards around callers. The final feature-off check caught and fixed that boundary.

## 3. Change Summary

| Category | Summary |
|----------|---------|
| Startup | `--ui`/`--webui` now enable model-free WebUI router mode for both binaries, while disabling aliases remain accepted and unsupported adjacent llama.cpp UI/tool/MCP/proxy flags still fail clearly. |
| Security | Production startup constructs the WebUI security policy with generated loopback keys or explicit non-loopback TLS/key requirements, and recognizes `[::1]` as loopback. |
| Catalog and runtime | Router and single-model UI catalog/runtime routes use persistent per-app catalog caches, selected-entry runtime configuration, and explicit readable-root validation for CLI/env cache roots. |
| Events | Router and single-model UI events share canonical cursor parsing, replay subscription, gap handling, and SSE serialization. |
| Documentation | WebUI bundling, catalog, architecture, and llama compatibility docs now describe mounted production startup and remaining unsupported adjacent surfaces. |

## 4. Validation

- Targeted WebUI route/startup tests passed for selected runtime config, paired SSE replay, invalid cursors, explicit model-store roots, bracketed IPv6 loopback classification, and single-model mounted SSE replay fixtures.
- `cargo clippy --lib --tests --features metal,accelerate -- -D warnings` passed.
- `make verify-webui-contract WEBUI_CONTRACT_PY=/tmp/mlxcel-webui-contract/bin/python` passed with 41 fixtures.
- `make verify-llama-compat verify-versions verify-kernel-dtype-keys` passed.
- `cargo check --no-default-features --features metal,accelerate,surgery --lib --tests` passed with existing no-WebUI unused warnings.
- GB10 CUDA CI was not run because the required runner is down; the maintainer approved proceeding with local validation and a root-owned merge exception for that unavailable required job.

## 5. Follow-up Actions

- Complete independent implementation and security reviews before merge.
- Root should run the broad workspace/full production binary gates and any serialized real-model acceptance required by the epic.
- Later WebUI issues still own page-level chat, downloads/removal, rich metrics, Safari/VoiceOver acceptance for revised UI, and actual GB10 CUDA validation when the runner is available.
