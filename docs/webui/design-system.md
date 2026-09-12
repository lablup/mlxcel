# WebUI design system contract

The WebUI is a browser application that follows the macOS 27 hierarchy without claiming native AppKit rendering. It uses system fonts, semantic tokens, shared React primitives and deterministic screenshots so page work can compose the same shell instead of inventing page-local glass, spacing or color rules.

## Reference mapping

| Reference | Applied design decision |
|---|---|
| Apple Human Interface Guidelines, Materials: https://developer.apple.com/design/human-interface-guidelines/materials | Glass is reserved for navigation chrome, toolbar chrome and transient overlays; content cards, tables and forms use stable neutral surfaces for readability. The CSS fallback is opaque when `backdrop-filter` or reduced-transparency support is missing. |
| WWDC26 session 289, AppKit design updates: https://developer.apple.com/videos/play/wwdc2026/289/ | The sidebar reaches the shell edge on desktop, compact view uses an explicit off-canvas navigation sheet instead of a permanent miniature rail, active navigation uses semibold selection emphasis, and rounded cards/buttons use concentric radius tokens rather than unrelated corner sizes. |
| Apple design kits news for macOS 27: https://developer.apple.com/news/?id=e2lxw9l1 | The token set includes light, dark, high-contrast, tinted and opaque variants, plus explicit component states, matching the public design-kit emphasis on expanded component/state coverage. No Apple assets, SF Symbols or proprietary controls are bundled; the SVG icons in `webui/src/design-system/icons.tsx` are project-authored and licensed with this repository. |

## Tokens

Tokens live in `webui/src/design-system/tokens.css` and `glass-intensity.css`. Pages must use these semantic names instead of hard-coded hex values: `--color-bg`, `--color-bg-elevated`, `--color-bg-neutral`, `--color-text`, `--color-muted`, `--color-separator`, `--color-focus`, `--color-selection`, `--color-warning`, `--color-error`, `--color-success`, `--material-glass-bg`, `--material-content-bg`, `--shadow-chrome`, `--shadow-card`, `--radius-*`, `--space-*` and `--motion-*`.

`data-theme`, `data-material`, `data-high-contrast`, `data-reduce-transparency`, `data-reduce-motion`, `data-backdrop-filter` and `data-glass-intensity` are the only supported appearance switches. The 0-100 glass intensity preference is represented by discrete CSS rules, not inline styles, so the future CSP can continue to reject unsafe inline styles. High contrast is `system` by default, can be explicitly `on` or `off`, and overrides decorative transparency when enabled by either the browser preference or the explicit setting.

## Components

`webui/src/design-system/primitives.tsx` exports the canonical Button, IconButton, Field, Select, StatusBadge, ProgressBar, EmptyState, ErrorBanner, Dialog, Sheet, Tooltip, Tabs, Inspector, DataTable, DenseList, AuthGate and SchemaMismatchView. Use native `<select>` as the combobox baseline and native `<dialog>` through these wrappers as the single modal/sheet primitive so focus restoration and Escape behavior are consistent. Do not add page-specific focus traps or alternate dialog implementations.

All controls expose visible focus, disabled and busy states, minimum 24 px targets with 32 px toolbar targets, and non-color-only state text. Long English/CJK labels must use `.truncate` with a `title` or another accessible full label. Progress uses explicit indeterminate semantics when total bytes are unknown rather than fabricated percentages. Lifecycle badges cover `unloaded`, `loading`, `ready`, `draining`, `unloading` and `failed` without implying a model is ready in placeholder screens.

## Shell and localization

`AppShell` defines the sidebar, top toolbar, content area and optional inspector. The route set is Models, Chat, Activity, Settings and Gallery. Cmd/Ctrl+K opens the command palette, `?` opens keyboard help, Escape closes the active dialog through the native dialog mechanism, and `[`/`]` move primary navigation only while the sidebar owns focus. Global shortcuts are suppressed inside edit controls, modal dialogs, IME composition and Alt-key chords. Compact widths below 960 px use an off-canvas sheet opened from the toolbar; no permanent mobile rail is rendered.

Every user-facing string has a typed key in `webui/src/i18n/catalog.ts` and a synchronized entry in `tests/fixtures/webui/strings.json` with English, Korean and a primary test ID. Locale-aware formatting helpers live beside the design system for bytes and tokens/s.

## Screenshots and verification

The component gallery is available at `#gallery`. `pnpm --dir webui run browser` compares committed screenshots in `webui/tests/screenshots/` for 390, 1024 and 1440 px widths across light, dark, tinted, opaque, high-contrast, reduced-motion and CJK scenarios. The `390-dark-opaque-textscale200` case uses a test-only root text-scale attribute to exercise reflow with 200% text metrics and verify that layout and hit targets still pass; it is not claimed as native browser zoom. These screenshots are implementation baselines for downstream page work and still require human screenshot approval before page-specific styling proceeds.

## Manual Safari and VoiceOver checklist

Use `pnpm --dir webui run build && pnpm --dir webui exec vite preview --host 127.0.0.1 --port 4173`, open `http://127.0.0.1:4173/#gallery` in Safari on macOS 27, then verify: VoiceOver announces the primary navigation, selected route, command dialog title, mobile navigation sheet title, native select combobox, dialog close button, status badges and schema/authentication views; Tab and Shift-Tab traverse visible controls in order; Escape closes the command, sheet and confirmation dialogs without cancelling model generation; browser zoom or text zoom at 200% has no horizontal page scroll at 390 px; high-contrast and reduced-transparency preferences keep content readable over the worst-case tinted background.
