# PR #1863 WebUI design system and provider integration report

**Date**: 2026-09-13
**Status**: Provider-backed shell plus approved 4185 macOS 27 style candidate implemented; Safari/VoiceOver/native-zoom targeted recheck and unavailable GB10 CI remain outside local validation
**Risk Level**: Medium

## Executive summary

PR #1863 now connects the WebUI design shell to the shared #1842 provider instead of presenting static placeholders or a second authentication cache. The production routes use the provider snapshot and actions for login, logout, connection state, selected catalog identity, lifecycle labels, catalog/operation freshness and safe schema-mismatch recovery. The direct `#gallery` artifact route stays isolated for deterministic visual baselines and does not contact the local API before an explicit login.

The approved 4185 candidate replaces the earlier floating macOS-26-like treatment with a source-backed macOS 27 direction: a flush full-height sidebar, a continuous 58 px sidebar/header edge, a sticky main toolbar with a hard scroll boundary, neutral content surfaces, restrained control groups, concentric radii, and compact two-row reflow with 44 px toolbar hit targets. No Apple artwork, fake traffic lights or SF Symbol assets are bundled.

## Change summary

The implementation wraps the app in `WebUiProvider`, routes LoginView submissions through `actions.login`, routes logout through `actions.logout`, and keeps the session key in provider/client memory only. Auth failures are reduced to localized presentation codes so raw tokens or server messages are not reflected in the DOM. Models, Chat and Activity remain honest staged routes: when signed out they show the provider-backed login surface, and when authenticated they report backend mode, build version, provider state, catalog count, operation count and snapshot sequence without loading a model or starting inference. The toolbar selected-model pill is derived only from a selected catalog entry and lifecycle state; otherwise it stays “No model selected.”

The visual revision removes the old outer app gutter, floating sidebar tile, hero-card page frame and decorative page gradient. Production CSS now carries the compact wrapping behavior directly rather than depending on `data-test-text-scale` layout selectors, and the browser test suite asserts document/panel scroll widths, visible compact focus and compact toolbar hit-target geometry.

## Validation status

Final local validation for the 4185 candidate:

- `pnpm --dir webui run typecheck` passed.
- `pnpm --dir webui run lint` passed.
- `pnpm --dir webui run unit` passed: 8 files, 62 Vitest tests.
- `pnpm --dir webui run browser` passed on Darwin: 19 Playwright tests, including 12 strict screenshots, product mock-API journeys, axe checks, overflow checks, compact hit-target assertions and token-containment checks.
- Docker Linux Playwright using `mcr.microsoft.com/playwright:v1.63.0-noble` passed: 19 tests with separate Linux screenshot baselines; the image reports Node v24.20.0 while the project engine is v26.5.1, but Vite and Playwright completed and snapshots were generated in that Linux browser environment.
- `make verify-webui-contract WEBUI_CONTRACT_PY=/tmp/mlxcel-webui-contract/bin/python` passed: 41 WebUI contract fixtures plus DTO drift and schema strictness checks.
- `make verify-webui-bundle` passed and verified deterministic checked-in assets with bundle digest `abb41bf0eb305acbc21291249afd7ac558b2d341382686dd705a1d1f5cf3fde0`.
- Approved 4185 source digest: `c4df0326df35998f03e315ab1382867150bb20b3ad516aebe184e722cbf20748`.

## Acceptance boundaries

The provider/mock HTTP tests cover no initial requests before submit, Bearer-prefixed bootstrap/catalog/operations/events calls, reachable all-route logout, 401 session purge, malformed bootstrap schema fail-closed recovery, wrong-key/offline localized errors, stale auth-failure fencing, no token DOM/storage/URL reflection and no autoload or inference endpoint calls while browsing authenticated routes. Browser screenshots are explicitly mock-API/product-shell evidence, not proof of a real backend session. The maintainer [approved the revised 4185 design](https://github.com/lablup/mlxcel/issues/1843#issuecomment-5648012196) on 2026-09-13 for source digest `c4df0326df35998f03e315ab1382867150bb20b3ad516aebe184e722cbf20748`; design approval is complete. A targeted recheck of changed layout and focus behavior in actual Safari on macOS 27, VoiceOver and native browser 200% zoom remains pending; the approval is not evidence that those checks passed. Earlier manual Safari/VoiceOver/native-zoom feedback applied only to the older cb489/4184 preview. CUDA/GB10 validation is not claimed because the required runner is down and the agreed path is local CI plus skipping unavailable required GB10 jobs.

The documentation-only finalization reran typecheck, lint, 62 unit tests and shared contract/compatibility/version/kernel-key checks at `50caf9a9`, and independently recomputed both digests above without changing source or assets. The Darwin and Docker Linux browser results are from the preceding implementation validation; no concurrent browser suite was launched for this documentation update.
