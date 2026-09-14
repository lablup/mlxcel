# Technical Report: PR #1871 - feat: enable model-free WebUI startup

**Date**: 2026-09-13 (updated 2026-09-14)
**Status**: Local implementation and acceptance complete; publication and merge pending
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

## 4. Validation

The complete acceptance run executed at `c2400a3e2ec36cd89954ee7544edb420b7a14d76`. The conflict-free rebase onto `3d8fd7b4` produced `4cb81177844126aab673dd08354d731952ee5058`: all eight patches are equivalent, and the tree differs from the measured revision only in `docs/benchmark_results/kernel-backend-kind-metal-m1ultra-2026-09-13.md` and `scripts/paged_decode_counter_ab.sh`. Runtime, build, frontend, and contract contents are identical. Heavy validation below belongs to `c2400a3e`, not a claimed rerun at `4cb81177`.

| Scope | Result |
|-------|--------|
| Full local gate | Passed: 11,243 tests, 0 failed, 361 ignored across 123 summaries. Workspace all-target Clippy, 44 contract fixtures, structural checks, formatting, and feature-off workspace checks passed; feature-off emitted 42 warnings. |
| Both actual test-fast binaries | Relocated empty-HOME/offline model-free startup and controlling-TTY/key/port-zero authority checks passed for `mlxcel-server` and `mlxcel serve`. Explicit-`-m` checks also passed: canonical bootstrap/catalog/runtime, real Llama “Hello” inference, canonical read-only `422`, and unchanged catalog after refusal. |
| Real RouterPool lifecycle | Llama streamed 948 characters; drain refusal returned 400; worker exit was observed; Granite returned “Affirmative.”; SIGINT cleanup reported one worker shutdown attempted and one completed. Process RSS observations do not prove GPU allocation release. |
| Release binaries | Both release binaries built successfully (9m 27s build log). Both passed relocated empty-HOME/offline startup, TTY/key checks, and explicit-`-m` real-model acceptance. |
| Post-rebase checks at `4cb81177` | 44 contract fixtures, llama compatibility, crate versions, kernel dtype structural checks, formatting/diff checks, and scoped Clippy passed. The inherited C++ `BITLINEAR_HIP_SOURCE` warning remains; this is not a warning-free build. |
| Targeted regression and reviews | The integration rebase at `c2400a3e` passed 38 selected Rust tests and scoped Clippy; 48 frontend tests, type/lint, 44 strict contract fixtures, and deterministic bundle verification passed. Independent correctness and security delta reviews cleared. |

### Failure history and acceptance boundaries

Earlier gates caught a synthetic route-identity mismatch (`0238c814`), a stale cache fixture (`70d4e409`), and four oversized diagnostic fields in two DFlash entries (`985f4a87`). These were corrected without weakening the relevant assertions or schema limits. The subsequent `1ff25a18` model-free inventory audit validated all 212 entries while keeping them unloaded.

The `1ff25a18` full gate later failed in unchanged `mlxcel-core::dflash_round_loop_starts_at_the_configured_depth` with SIG6 and Metal `commandbufferDiscarded` / `InnocentVictim` recovery. The same binary passed its first isolated run and failed the second. The cause remains unknown. No workaround or numerical tolerance change was introduced; the successful complete `c2400a3e` rerun resolves the acceptance gate, not the historical failure's root cause.

The initial explicit-`-m` harness rejected SIGINT termination with return code `-2`. Read-only comparison established that `serve_http` and `listen` were byte-identical to the baseline and that this PR preserves single-model shutdown behavior. The harness was corrected only to accept bounded normal SIGINT termination; the original failure remains recorded. This does not establish graceful worker draining in single-model mode, unlike the separately observed RouterPool cleanup.

The maintainer's GB10-down exception applies only to unavailable required runner checks. It does not waive local failures or review findings and does not establish CUDA execution coverage. No branch-protection changes are part of this work.

Evidence is retained in the orchestration run's `gate-1838-c2400a3e.log`, `acceptance-1838-c2400a3e-run3.log`, and `release-build-1838-c2400a3e.log`; the acceptance log records the per-binary result artifacts.

## 5. Follow-up Actions

- Publish the finalized reports and PR evidence, then complete the centrally owned merge workflow; this report does not claim the PR is merged.
- Investigate the historical Metal recovery failure separately if it recurs; its cause is still unproven.
- Later WebUI issues own downloads/removal adapters (#1841), rich metrics (#1847), page-level workflows, and revised-UI Safari/VoiceOver acceptance. The separate `ui-common` adoption request is not implemented by this PR.
- Run actual GB10 CUDA validation when the runner recovers.
