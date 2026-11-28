# Technical Report: PR #2047 - Enforce CORS origin policy on /v1/realtime upgrade

**Date**: 2026-09-29

**Status**: Implemented and validated with fake-engine tests; pending merge.

**Languages**: Rust

**Risk Level**: Low. The change is confined to the realtime route and one `CorsPolicy` method. The default `--cors-origins *` posture is unchanged.

## Executive Summary

`GET /v1/realtime` accepted a WebSocket upgrade from any `Origin`, whatever `--cors-origins` or `--allowed-origins` said. Browsers do not apply CORS to WebSocket upgrades, so any web page could open the socket, take the single VoiceChat session slot, and read every event (cross-site WebSocket hijacking). The route now checks `Origin` against the configured policy and answers `403` before the upgrade (issue #2042).

## 1. Problem Statement

`cors_middleware` only stamps `Access-Control-Allow-Origin` on responses. It never rejects a request, and router-mode sub-apps do not run it at all. A browser completes a WebSocket handshake whatever that header says, so an operator who set `--cors-origins localhost` or `--allowed-origins https://app.example.com` had no browser restriction on `/v1/realtime`. The exposed case is a server without `--api-key` (browsers cannot set `Authorization` on `new WebSocket`) reached from a page the user visits. The impact is occupation of the only session slot and compute use.

## 2. Change Summary

- **Policy** (`server::cors`): `CorsPolicy::permits_websocket_origin(Option<&HeaderValue>)`. An absent `Origin` is accepted. A present one, including `null`, must match the origin rule: any value under `Wildcard`, a localhost host (the existing `origin_is_localhost`) under `Localhost`, byte equality under `Literal`, membership under `AllowList`. `credentials` plays no part.
- **Route** (`server::routes::realtime`): `realtime_router(engine, cors)` with a private `RealtimeRouteState`. The handler checks the origin first and returns `403` with an `invalid_request_error` JSON body and one `warn!` log. No reservation is taken.
- **Wiring** (`server::app::build_routes`): passes `Arc::new(config.cors_policy.clone())`. Because the check is in the route, `create_app_without_cors` (router-mode sub-apps) enforces it too.
- **Docs**: one sentence in `docs/nemotron-voicechat.md`.

## 3. Technical Decisions

### Check in the route, not in `cors_middleware`

Rejecting in the middleware would change HTTP CORS behavior that follows llama-server b10621, and the middleware is skipped for router-mode sub-apps. Only the one route that needs the check has it.

### Origin check before upgrade validation

The issue sketched the handler with `ws: WebSocketUpgrade` as the first extractor. The handler instead takes `Result<WebSocketUpgrade, WebSocketUpgradeRejection>` last and checks `Origin` first. A disallowed origin is refused even when the handshake is malformed, and the in-process app test can see the `403` through `create_app` without a hyper upgrade handle. A valid-origin request with a bad handshake still gets axum's own rejection.

### Absent Origin accepted, API key stays outer

Browsers always send `Origin` on an upgrade; `websocat`, the example clients, and the tests do not. Accepting an absent header keeps those clients working without weakening the browser case. The API-key layer wraps the route, so a request failing both checks gets `401`, as before.

## 4. Validation

- `websocket_origin_matrix` unit test: every policy variant against absent, allowed, disallowed, `null`, and non-UTF-8 origins, with credentials on and off.
- App-level test over `create_app` and `create_app_without_cors`: `403` for a disallowed origin, not `403` for allowed or absent, `401` when an API key is also missing.
- `tests/realtime_ws.rs`: `start_server_with_policy`; a disallowed origin fails the handshake with HTTP 403, then an allowed origin gets `session.created` on the first try (the rejected attempt held no reservation) and a client with no `Origin` gets one after release; the localhost policy rejects `https://localhost.evil.com` and accepts `http://localhost:3000`; the default policy accepts any origin.
- `cargo fmt --check`, `cargo clippy --release -p mlxcel --lib --tests --examples -D warnings`, the three contract tests, and `cargo test --release -p mlxcel --lib server::` (3270 passed, 0 failed) pass locally. GitHub CI was not awaited because the GB10 runner is down.

## 5. Known Limitations

- No real-checkpoint run; the fake-engine tests exercise the same route and handshake.
- Router mode does not proxy WebSocket upgrades today, so no router-level change was needed.
