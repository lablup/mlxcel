# PR #1863 WebUI design system report

## Scope

PR #1863 implements the shared WebUI design-system layer for epic #1834. It provides semantic CSS tokens, a restrained macOS 27-inspired shell, typed English/Korean strings, reusable primitives, appearance persistence and a component gallery at `#gallery`. The implementation reserves glass for decorative chrome, keeps content on neutral surfaces, uses project-authored SVG icons rather than Apple assets, and records the official Apple references that informed each layout and material decision.

## Review hardening

The fix cycles replaced false production placeholders with one neutral connection prompt, removed Gallery from primary navigation while keeping it as a direct artifact route, removed empty inspector chrome from production routes, tightened the desktop grid, added controlled LoginView and SchemaMismatchView contracts, and expanded browser tests from visual smoke coverage into behavior coverage. The production screens remain truthful placeholders until the shared #1842 provider is integrated after that PR merges; this PR does not copy or cache #1842 provider state.

## Validation status

Final local validation covers `pnpm --dir webui run typecheck`, `pnpm --dir webui run lint`, `pnpm --dir webui run unit` with 10 passing Vitest tests, `pnpm --dir webui run browser` with 12 passing Playwright tests, deterministic bundle verification with digest `44b731231da59d5454c3d2956bc50fe22c6fddecffa306d0134f93d06c804ece`, `make verify-webui-contract WEBUI_CONTRACT_PY=/tmp/mlxcel-webui-contract/bin/python`, and `make verify-llama-compat verify-versions verify-kernel-dtype-keys`. Actual Safari on macOS 27, VoiceOver and native browser 200% zoom verification were not executed in this environment; `docs/webui/design-system.md` records the exact checklist and local preview URL for that downstream/manual gate. CUDA/GB10 validation is not claimed because the required runner is down and the agreed path is local CI plus skip of unavailable required GB10 jobs.
