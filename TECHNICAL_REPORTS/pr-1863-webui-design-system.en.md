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
- The initial stock Docker Linux run passed 19 tests against its own baselines but did not establish hosted renderer compatibility: that image lacks DejaVu and uses WenQuanYi for Latin text too. After pinning the hosted font packages, native arm64 reproduction passed 20 tests against all 12 reviewed hosted images (19 normal cases plus the opt-in Chromium font diagnostic). The image uses Node v24.20.0, not the canonical v26.5.1, so this is renderer compatibility evidence, not build-toolchain equivalence.
- `make verify-webui-contract WEBUI_CONTRACT_PY=/tmp/mlxcel-webui-contract/bin/python` passed: 41 WebUI contract fixtures plus DTO drift and schema strictness checks.
- `make verify-webui-bundle` passed and verified deterministic checked-in assets with bundle digest `abb41bf0eb305acbc21291249afd7ac558b2d341382686dd705a1d1f5cf3fde0`.
- Approved 4185 source digest: `c4df0326df35998f03e315ab1382867150bb20b3ad516aebe184e722cbf20748`.

## Acceptance boundaries

The provider/mock HTTP tests cover no initial requests before submit, Bearer-prefixed bootstrap/catalog/operations/events calls, reachable all-route logout, 401 session purge, malformed bootstrap schema fail-closed recovery, wrong-key/offline localized errors, stale auth-failure fencing, no token DOM/storage/URL reflection and no autoload or inference endpoint calls while browsing authenticated routes. Browser screenshots are explicitly mock-API/product-shell evidence, not proof of a real backend session. The maintainer [approved the revised 4185 design](https://github.com/lablup/mlxcel/issues/1843#issuecomment-5648012196) on 2026-09-13 for source digest `c4df0326df35998f03e315ab1382867150bb20b3ad516aebe184e722cbf20748`; design approval is complete. The maintainer subsequently reconfirmed this design and separately accepted the targeted Safari/VoiceOver toolbar, compact-menu, Cmd+K Tab/Escape focus and native browser 200% zoom recheck. The [ledger](https://github.com/lablup/mlxcel/issues/1843#issuecomment-5653425099) records USER-REPORTED PASS for immutable 4185, not an independent browser observation or acceptance of future layout changes. This supplements the earlier cb489/4184 manual report. CUDA/GB10 validation is not claimed because the required runner is down and the agreed path is local CI plus skipping unavailable required GB10 jobs.

The documentation-only finalization reran typecheck, lint, 62 unit tests and shared contract/compatibility/version/kernel-key checks at `50caf9a9`, and independently recomputed both digests above without changing source or assets. The Darwin and Docker Linux browser results are from the preceding implementation validation; no concurrent browser suite was launched for this documentation update.

## Linux screenshot correction

Hosted Chromium CDP identified DejaVu Sans regular/bold for Latin text and WenQuanYi Zen Hei for Korean fallback. The mismatch also occurs in stock amd64 containers, so architecture alone is not the cause. The WebUI CI job now pins Ubuntu 24.04 with `fonts-dejavu-core=2.37-8`, `fonts-dejavu-extra=2.37-8` and `fonts-wqy-zenhei=0.9.45-8`; canonical Node and pnpm versions remain unchanged. Twelve hosted actual images were individually reviewed before adoption. [Screenshot provenance](../webui/tests/screenshots/README.md) records their source run and hashes. Production fonts, approved source/assets, screenshot thresholds, geometry and rendering flags did not change.

The pinned-font native arm64 reproduction matched hosted fonts through Chromium CDP and passed all 20 tests. A local amd64 attempt under QEMU crashed Chromium GPU processes before rendering; it is not a pass. The earlier stock-image result must not be substituted for this corrected renderer evidence.

The [canonical hosted WebUI job](https://github.com/lablup/mlxcel/actions/runs/34723713520/job/103634069494) at `b3c5326491230cddf77d68ce3d90815ff237f7b9` passed with Ubuntu 24.04 amd64, Node 26.5.1, pnpm 11.18.0 and the pinned fonts. Typecheck, lint, 62 unit tests, all 20 browser tests and deterministic bundle verification actually executed successfully. This supersedes the incomplete stock-container evidence for hosted validation; it does not claim that unavailable GB10 jobs passed or independently establish the user-reported manual result.

## Manual-acceptance finalization

The targeted manual gate is now accepted as user-reported; central integration remains pending in order #1838 → #1841 → #1843, with issue and PR still in review. This documentation-only update changes no source, styles, assets or tests and runs no local GPU, browser or runtime tests. Host GPU firmware recovery is not confirmed; the unavailable GB10 waiver does not waive local host recovery or other integration gates.

## ui-common adoption checkpoint

The new user-requested migration pins @lablup/ui-common alpha.19 and consumes component subpaths through shared adapters/token bridges. See the [export/exception matrix](../docs/webui/ui-common.md). CPU typecheck, lint, 72 unit tests and 41 current-worktree contract fixtures pass. Published NOTICE/LICENSE are bundled. The old4185 source/manual approval remains historical; rewritten DOM needs new served-CSP, visual and relevant manual evidence. No local browser/GPU test was run under the root host reservation. Central integration must preserve newer backend schema/fixture changes on rebase.

## Rebased migration validation

At `62c4f259`, the branch incorporates the merged startup/security backend without modifying its Unicode validator, schema or 44 contract fixtures. Frozen dependency installation, typecheck, lint, 76 unit tests and deterministic bundle verification pass. Source digest: `cfbcaada7d5e9734b2d305eacf534e003d8b8ea3916ce163c5a53f294c6bbd78`; bundle digest: `858f5775f548532b3fa93942c7f063e06b83f427d107b47c82678f3564638f7f`. Independent correctness/security rereviews report no findings in the migration seam.

The pinned Playwright heading-level helper prioritizes native `h3` over explicit `aria-level`; direct Chromium AX inspection proves level 2, the expected name and non-ignored visibility across repeated common Tabs remounts. Browser and supplied-server CSP tests now assert that actual tree, and StrictMode units cover remounts. This is Chromium evidence, not native Safari proof.

Hosted run 34801765896 passed 19 browser cases and failed only the old States screenshot. The new measured-progress example is grouped with unknown progress in its existing cell, lifecycle label casing is preserved, and only the independently reviewed Darwin/Linux States baselines are updated with exact provenance. All other baselines and strict thresholds are unchanged; the full hosted rerun remains pending at this documentation checkpoint. Root-owned actual secured-CSP execution and a targeted user Safari/VoiceOver recheck for the migrated DOM remain separate pending gates. Local bounded browser diagnostics ended cleanly before the root runtime build; no MLX/GPU tests were run by this unit.
