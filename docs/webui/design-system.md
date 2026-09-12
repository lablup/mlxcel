# WebUI design system contract

The WebUI is a browser application that follows the macOS 27 hierarchy without claiming native AppKit rendering. It uses system fonts, semantic tokens, shared React primitives and deterministic screenshots so page work can compose the same shell instead of inventing page-local glass, spacing or color rules.

## Reference mapping

| Reference | Applied design decision |
|---|---|
| Apple Human Interface Guidelines, Materials: https://developer.apple.com/design/human-interface-guidelines/materials | Glass is reserved for navigation chrome, toolbar chrome and transient overlays; content cards, tables and forms use stable neutral surfaces for readability. The CSS fallback is opaque when `backdrop-filter` or reduced-transparency support is missing. |
| WWDC26 session 289, AppKit design updates: https://developer.apple.com/videos/play/wwdc2026/289/ | The sidebar reaches the shell edge, active navigation uses semibold selection emphasis, the toolbar has a hard separator over scrollable content, and rounded cards/buttons use concentric radius tokens rather than unrelated corner sizes. |
| Apple design kits news for macOS 27: https://developer.apple.com/news/?id=e2lxw9l1 | The token set includes light, dark, high-contrast, tinted and opaque variants, plus explicit component states, matching the public design-kit emphasis on expanded component/state coverage. No Apple assets, SF Symbols or proprietary controls are bundled. |

## Tokens

Tokens live in `webui/src/design-system/tokens.css` and `glass-intensity.css`. Pages must use these semantic names instead of hard-coded hex values: `--color-bg`, `--color-bg-elevated`, `--color-bg-neutral`, `--color-text`, `--color-muted`, `--color-separator`, `--color-focus`, `--color-selection`, `--color-warning`, `--color-error`, `--color-success`, `--material-glass-bg`, `--material-content-bg`, `--shadow-chrome`, `--shadow-card`, `--radius-*`, `--space-*` and `--motion-*`.

`data-theme`, `data-material`, `data-high-contrast`, `data-reduce-transparency`, `data-reduce-motion` and `data-glass-intensity` are the only supported appearance switches. The 0-100 glass intensity preference is represented by discrete CSS rules, not inline styles, so the future CSP can continue to reject unsafe inline styles. Reduced transparency and high contrast force opaque readable chrome regardless of the decorative intensity setting.

## Components

`webui/src/design-system/primitives.tsx` exports the canonical Button, IconButton, Field, Select, StatusBadge, ProgressBar, EmptyState, ErrorBanner, Dialog, Tooltip, Tabs and Inspector. Use native `<select>` as the combobox baseline and native `<dialog>` as the single modal primitive so focus restoration and Escape behavior are consistent. Do not add page-specific focus traps or alternate dialog implementations.

All controls expose visible focus, disabled and busy states, minimum 24 px targets with 32 px toolbar targets, and non-color-only state text. Long English/CJK labels must use `.truncate` with a `title` or another accessible full label. Progress uses explicit indeterminate semantics when total bytes are unknown rather than fabricated percentages.

## Shell and localization

`AppShell` defines the sidebar, top toolbar, content area and optional inspector. The route set is Models, Chat, Activity, Settings and Gallery. Cmd/Ctrl+K opens the command palette, `?` opens keyboard help, Escape closes the active dialog through the native dialog mechanism, and `[`/`]` move primary navigation only while the sidebar owns focus. Chat send/new-chat shortcuts are intentionally deferred to the chat workflow issue.

Every user-facing string has a typed key in `webui/src/i18n/catalog.ts` and a synchronized entry in `tests/fixtures/webui/strings.json` with English, Korean and a primary test ID. Locale-aware formatting helpers live beside the design system for bytes and tokens/s.

## Screenshots and verification

The component gallery is available at `#gallery`. `pnpm --dir webui run browser` captures stable screenshots into `webui/tests/screenshots/` for 390, 1024 and 1440 px widths across light, dark, tinted, opaque, high-contrast, reduced-motion and CJK scenarios. These screenshots are design-system baselines for downstream page work; they are not runtime acceptance for the future server adapters.

## Manual Safari and VoiceOver checklist

Use `pnpm --dir webui run build && pnpm --dir webui exec vite preview --host 127.0.0.1 --port 4173`, open `http://127.0.0.1:4173/#gallery` in Safari on macOS 27, then verify: VoiceOver announces the primary navigation, selected route, command dialog title, native select combobox, dialog close button and status badges; Tab and Shift-Tab traverse visible controls in order; Escape closes the command and confirmation dialogs without cancelling model generation; 390 px responsive layout has no horizontal page scroll; high-contrast and reduced-transparency preferences keep content readable over the worst-case tinted background.
