# PR #1863 WebUI design system report

**Date**: 2026-09-12
**Status**: Partial — components staged; provider integration and manual acceptance pending
**Risk Level**: Medium

## Executive summary

PR #1863 implements the shared WebUI design-system layer for epic #1834. It provides semantic CSS tokens, a restrained macOS 27-inspired shell, typed English/Korean strings, reusable primitives, appearance persistence and a component gallery at `#gallery`. The implementation reserves glass for decorative chrome, keeps content on neutral surfaces, uses project-authored SVG icons rather than Apple assets, and records the official Apple references that informed each layout and material decision.

## Problem statement

Downstream pages need one shared material, spacing, localization and keyboard contract rather than independently styled controls or duplicate authentication state. A component gallery can validate those primitives before the shared client is available, but cannot establish production authentication or native browser accessibility acceptance.

## Change summary and review hardening

The fix cycles replaced false production placeholders with one neutral connection prompt, removed Gallery from primary navigation while keeping it as a direct artifact route, removed empty inspector chrome from production routes, tightened the desktop grid, added controlled LoginView and SchemaMismatchView contracts, and expanded browser tests from visual smoke coverage into behavior coverage. The production screens remain truthful placeholders until the shared #1842 provider is integrated after that PR merges; this PR does not copy or cache #1842 provider state.

## Validation status

Final local validation covers `pnpm --dir webui run typecheck`, `pnpm --dir webui run lint`, `pnpm --dir webui run unit` with 10 passing Vitest tests, `pnpm --dir webui run browser` with 12 passing Playwright tests, deterministic bundle verification with digest `44b731231da59d5454c3d2956bc50fe22c6fddecffa306d0134f93d06c804ece`, `make verify-webui-contract WEBUI_CONTRACT_PY=/tmp/mlxcel-webui-contract/bin/python`, and `make verify-llama-compat verify-versions verify-kernel-dtype-keys`. Actual Safari on macOS 27, VoiceOver and native browser 200% zoom verification were not executed in this environment; `docs/webui/design-system.md` records the exact checklist and local preview URL for that downstream/manual gate. CUDA/GB10 validation is not claimed because the required runner is down and the agreed path is local CI plus skip of unavailable required GB10 jobs.


## Acceptance boundaries

The 12 browser tests include 8 screenshot cases with separate Darwin/Linux baselines at 390, 1024 and 1440 CSS-pixel widths, plus interaction and appearance checks. Screenshot review by the integration owner establishes implementation baselines, not user approval. Final visual approval remains outstanding. The root preview shown for manual feedback is immutable snapshot `b3f0cd04`, older than the final 200% text-scale reflow correction, and is not evidence for the final source.

LoginView requests `autocomplete="off"` on its form and password field and clears its local field on submission or its logout action. These hints cannot guarantee that a browser or password-manager extension will not save credentials. The shared #1842 provider must own session state and perform authentication; its integration is still required before #1843 can complete or merge.

The staged finalization reran typecheck, lint, all 10 unit tests and shared static checks at `b202f221`, including 32 contract fixtures. The 12 browser tests and platform baseline review were established by the preceding review/CI cycle and were not rerun for this documentation-only update. Actual Safari, VoiceOver, native 200% browser zoom and user screenshot approval remain required and are not waived by the GB10 exception.

The [Linux WebUI bundle job](https://github.com/lablup/mlxcel/actions/runs/34695797255/job/103558967545) at `b202f221` executed and passed typecheck, lint, unit, browser and generated-bundle verification steps. Its two unavailable GB10 jobs remain queued, not passed. The asset-tree digest above differs intentionally from manifest `source_digest_sha256`, which is `e5d0a9c8274fe0893d287339ea29f2598b01e493695337f7653b3ba7ecbd939e`; both were independently recomputed during staged finalization.
