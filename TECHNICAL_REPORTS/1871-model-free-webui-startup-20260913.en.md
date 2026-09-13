# Technical Report: PR #1871 - feat: enable model-free WebUI startup

**Date**: 2026-09-13
**Status**: Needs Follow-up — `pending_host_recovery` (PR remains in review)
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

### 2.4 Preserve canonical read-only responses and Unicode limits

Bootstrap capability availability now follows the actual pool cache rather than a startup-path hint. Catalog diagnostics redact filesystem paths before truncating to the schema's 512 Unicode-code-point limit; the TypeScript validator counts code points rather than UTF-16 code units. Real inventory exposed this boundary where synthetic short diagnostics did not.

The read-only fix at `534563fb704619f407e4ea699482fda9c22fac98` mounts stateless single-model refusals for model actions, downloads, removals, cancellation, and catalog refresh: each returns canonical `422 unsupported` without mutating or loading a model. Unknown operation lookup returns a canonical `404`. The API adds the missing catalog-refresh `422` response declaration without changing DTO limits. A mounted fake-AppState test first reproduced the previous empty `404`; this is CPU-only HTTP evidence, not real single-model inference acceptance.

## 3. Change Summary

| Category | Summary |
|----------|---------|
| Startup | `--ui`/`--webui` now enable model-free WebUI router mode for both binaries, while disabling aliases remain accepted and unsupported adjacent llama.cpp UI/tool/MCP/proxy flags still fail clearly. |
| Security | Production startup constructs the WebUI security policy with generated loopback keys or explicit non-loopback TLS/key requirements, and recognizes `[::1]` as loopback. |
| Catalog and runtime | Router and single-model UI catalog/runtime routes use persistent per-app catalog caches, selected-entry runtime configuration, and explicit readable-root validation for CLI/env cache roots. |
| Contract boundaries | Cache capabilities follow actual pool state; catalog diagnostics and client validation agree on Unicode length; the single-model refusal fix preserves typed error envelopes. |
| Events | Router and single-model UI events share canonical cursor parsing, replay subscription, gap handling, and SSE serialization. |
| Documentation | WebUI bundling, catalog, architecture, and llama compatibility docs now describe mounted production startup and remaining unsupported adjacent surfaces. |

## 4. Validation and Remaining Blockers

Validation is tied to the source revision below. An earlier full pass is not a full pass for the current changes.

| Revision / scope | Result |
|------------------|--------|
| `985f4a87` full local gate | Passed: 11,236 tests, 0 failed, 361 ignored across 123 summaries; workspace all-target Clippy, structural/contract checks, and feature-off workspace check passed. The feature-off check emitted 42 warnings. |
| `985f4a87` actual test-fast binaries | Both `mlxcel-server` and `mlxcel serve` passed relocated empty-HOME/offline model-free startup and controlling-TTY/key/port-zero authority checks. These are not release-binary results. |
| `985f4a87` real inventory | Failed strict schema validation on four oversized diagnostic fields from two DFlash entries in a 212-entry catalog. The attempted real lifecycle stopped before inference; `1ff25a18` fixes this boundary. |
| `1ff25a18` catalog and RouterPool acceptance | All 212 entries remained unloaded and passed strict schema validation. Real Llama streamed 467 characters, drain refusal returned 400, worker exit was observed, Granite returned “Affirmative.”, and SIGINT cleanup reported one attempted and one completed worker shutdown. Complete loaded UI snapshots passed canonical validation. This is router-mode evidence, not explicit-`-m` single-model acceptance. |
| `1ff25a18` targeted checks | 36 Rust catalog tests, scoped Clippy, 48 frontend tests, type/lint checks, 42 strict contract fixtures, and deterministic bundle verification passed. Independent correctness and security reviews of this delta cleared. |
| `1ff25a18` full local gate | Failed in the unchanged `mlxcel-core` test `dflash_round_loop_starts_at_the_configured_depth`: SIG6 with Metal `commandbufferDiscarded` / `InnocentVictim` recovery. An isolated run of the exact same binary passed once and failed again on its second run. Cause remains unknown; this is neither a full pass nor an assumed transient failure. |
| `534563fb704619f407e4ea699482fda9c22fac98` single-model refusal fix | Passed: three CPU-only mounted single-control tests with valid payloads, one existing single-model SSE replay test, 12 shared WebUI security tests, 44 strict contract fixtures, 48 frontend tests, type/lint checks, and deterministic bundle verification. Scoped Clippy and independent correctness and security delta reviews also cleared. The root CPU-only gate passed workspace all-target Clippy, 44 contract fixtures, structural checks, formatting, and diff checks at this exact runtime revision. |
| Explicit-`-m` real acceptance and both release-binary relocation gates | Not run. All GPU work is paused pending Mac host recovery; a reboot has been requested. |

Earlier full attempts also exposed a synthetic route-identity mismatch (`0238c814`) and a stale cache fixture (`70d4e409`). Both were corrected without weakening the relevant assertion, and the subsequent `985f4a87` full gate passed. Process RSS observed during the real lifecycle is not evidence that GPU allocations were freed.

The maintainer's GB10-down exception applies only to unavailable required runner checks. It does not waive the local GPU failure, outstanding release acceptance, or review findings, and does not establish CUDA execution coverage. No branch-protection changes are part of this work.

## 5. Follow-up Actions

- Recover the Mac host and diagnose/revalidate the failed GPU gate before resuming GPU work; do not repeatedly retry on the unhealthy host.
- Rerun required full acceptance against the final runtime revision `534563fb704619f407e4ea699482fda9c22fac98` after host recovery.
- Run explicit-`-m` real single-model acceptance and both release-binary relocated/offline gates. Keep PR #1871 in review until required local evidence is complete.
- Later WebUI issues own downloads/removal adapters (#1841), rich metrics (#1847), page-level workflows, and Safari/VoiceOver acceptance for the revised UI. Actual GB10 CUDA validation remains unavailable until runner recovery.
