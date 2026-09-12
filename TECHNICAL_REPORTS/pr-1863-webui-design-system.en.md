# PR #1863 WebUI design system report

## Scope

PR #1863 implements the shared WebUI design-system layer for epic #1834. It provides semantic CSS tokens, a restrained macOS 27-inspired shell, typed English/Korean strings, reusable primitives, appearance persistence and a component gallery at `#/gallery`. The implementation reserves glass for decorative chrome, keeps content on neutral surfaces, uses project-authored SVG icons rather than Apple assets, and records the official Apple references that informed each layout and material decision.

## Review hardening

The first review cycle replaced the permanent mobile rail with a native-dialog-backed navigation sheet, added modal focus containment/restoration, roving tabs, stronger storage error handling, explicit high-contrast system/on/off preferences, axe-backed browser checks, screenshot comparison baselines, honest placeholder states, authentication and schema-mismatch surfaces, and same-origin/offline verification. The production screens remain truthful placeholders until the shared #1842 provider is integrated.

## Validation status

Local automated validation covers TypeScript, lint, unit tests, Playwright accessibility/keyboard/viewport/screenshot checks and deterministic bundle rebuild verification. Actual Safari on macOS 27 and VoiceOver manual verification were not executed in this environment; `docs/webui/design-system.md` records the exact checklist and local preview URL for that downstream/manual gate. CUDA/GB10 validation is not part of this design-system PR and is not claimed here.
