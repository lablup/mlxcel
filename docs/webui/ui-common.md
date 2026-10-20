# Shared ui-common integration

The WebUI builds its controls on `@lablup/ui-common` 0.2, Lablup's layer over Astryx. This document is the contract for that dependency: what is pinned, where it may be imported, which product component wraps which shared one, the bridges that remain, the Content Security Policy boundary, the bundle budget, who owns the theme, and how the library's own strings are localized. The product's tokens, families and shell are described in [design-system.md](design-system.md).

## Versions

| Package | Pin | Role |
|---|---|---|
| `@lablup/ui-common` | `0.2.0-alpha.19` | Re-exports Astryx 0.7.0 one to one (`@lablup/ui-common/<Component>`, `@lablup/ui-common/lab`, `/i18n`, `/theme`) and adds Lablup's own components (PageHeader, PageLayout, ErrorState, StatCard, SmoothHeight, the Drawer fork, ...). |
| `@lablup/ui-common-cli` | `0.2.0-alpha.19` (dev) | The `ui-common` bin: component and docs lookup, `ui-common agents` (the generated block in `webui/AGENTS.md`) and `ui-common astryx theme build`. |
| `@astryxdesign/lab` | `0.7.0-canary.97a0d31` | Peer of ui-common; only its stylesheet ships. |
| `@stylexjs/stylex` | `0.19.0` | The StyleX runtime. No StyleX compiler runs in this build: Astryx and ui-common ship compiled CSS. |

Every version is exact. `@lablup/ui-common` and `@lablup/ui-common-cli` move together, to the same version, in one change: after a bump run `pnpm install`, `pnpm exec ui-common upgrade --from <old version>` (read the report; `--apply` is not part of the workflow), `node scripts/layer-slide-rules.mjs`, `pnpm run build:themes`, and regenerate the agent block with `pnpm exec ui-common agents --write AGENTS.md`, all from `webui/`. `pnpm --dir webui run check:agents` fails while the block is stale, and `check:themes` while the built themes are.

`webui/pnpm-workspace.yaml` declines the Astryx postinstall scripts (pnpm 11 `allowBuilds`) and overrides `@astryxdesign/lab>@astryxdesign/core` to `0.7.0`, so there is one copy of Astryx core and every layer shares its theme and layer contexts; `src/design-system/ui-common-dependencies.test.ts` fails when the lockfile resolves a second core. The lockfile keeps the published tarball integrity; no source is vendored.

## Import rules

- Feature code under `src/features/*` imports controls only from `src/design-system/primitives.tsx`. It never imports `@lablup/ui-common`.
- Package imports live only in the adapter seam `src/design-system/common-*.tsx` (`common-adapters`, `common-app-shell`, `common-data`, `common-drawer`, `common-feedback`, `common-i18n` with its lazy `common-i18n-ko.ts`, `common-layout`, `common-overlays`, `common-select`, `common-table`, `common-tabs`, `common-theme`), the theme sources under `src/design-system/themes/`, and the stylesheet entry `src/ui-common-entry.css`.
- The first-paint modules (`app.tsx`, `design-system/shell.tsx`, `provider-surfaces.tsx`, `command-palette.tsx`) import seam modules directly, never the barrel: the bundler keeps whatever a barrel re-exports in the chunk that imports it, so importing `primitives.tsx` from the entry would pull every screen's controls into the first bundle.
- Astryx is reached only through ui-common. ESLint rejects any `@astryxdesign/*` import, `import()`, `require()` or `typeof import()` in the WebUI sources.
- ui-common templates and blocks are not scaffolded into the tree (they assume a StyleX compiler); `ui-common component <Name>` is the reference for what an adapter wraps.

## Component to adapter matrix

| Product export | ui-common / Astryx component, and what the adapter keeps |
|---|---|
| Button / IconButton | Astryx `Button` (icon-only for IconButton). The product prop surface is explicit (tone, busy, disabled, type, the handlers and ARIA attributes the call sites use, `title`, `data-*`). Tones map primary, secondary, danger to destructive, ghost. Astryx requires a string `label`: the adapter passes the caller's `aria-label`, or the visible text of the children with separate pieces joined by a space. Busy is Astryx `isLoading` (spinner, `aria-busy`, native disabled and a visually hidden loading announcement in the catalog locale). `tone="danger"` on IconButton keeps the outlined control and draws the glyph in `--color-error`. |
| StatusBadge | Astryx `StatusDot` (decorative, drawn in the label's color) beside the visible lifecycle label, in the product pill; `appearance="inline"` drops the pill (the toolbar's loaded-model chip). Unloaded maps to neutral, loading to accent, ready to success, draining and unloading to warning, failed to error. |
| ProgressBar | Astryx `ProgressBar` with its label visually hidden and used as the bar's name; an undefined or nonfinite value is `isIndeterminate`, finite percentages clamp to 0-100. |
| EmptyState | Astryx `EmptyState` with `headingLevel={2}`, the product icon and the caller's action. |
| Tabs | Astryx `TabList` with `role="tablist"` and one `Tab` per section; the adapter renders the active panel, linked both ways (`aria-controls` / `aria-labelledby`). Selection follows focus (an arrow, Home or End key moves focus and selects), as the product's tabs always have. Without a product `label` the strip takes Astryx's catalog name. |
| Select | Astryx `Selector`, always a popover (`presentation="popover"`, never the narrow-viewport bottom sheet), with the product's localized empty text. Selector supplies the combobox role, `aria-busy` and `aria-invalid`; the product hint and error stay below it and describe the trigger. An option's description is part of its name. The list renders inline next to the trigger, so inside a modal Dialog it stays in the dialog's subtree. |
| DataTable<T> | Astryx `Table`. The product column shape (`id`, `header`, `render`, `className`, `fixedWidth`, `sortable`, `sortComparator`, `sortValueAccessor`) and the controlled sort (`sortColumnId`, `sortDirection`, `onSortChange`; a third activation clears) are kept; sorting renders through Astryx's `useTableSortable` header buttons and `aria-sort`. Alignment is Astryx's default. `fixedWidth` is Astryx's `pixel()` column width (the Models Actions column). The loading and empty states render in Astryx's empty-state row. The scroll box is the adapter's own (`.ds-table-scroll`), a named, focusable region only while it overflows (`overflowRegionLabel` / `useOverflowRegion`). `activateRowPrimary` keeps whole-row activation through the row's `ROW_PRIMARY_CLASS` control; rows keep their native role and gain no tab stop. |
| StaticTable | Semantic markup table kept only for the gallery fixture. Feature tables use DataTable<T>. |
| Dialog / ConfirmDialog | The product's native `showModal()` dialog (`local-components.tsx`), joined to Astryx's layer stack (`useModalDialogLayer`), so one Escape closes only the top-most layer: a confirmation opened from a Drawer closes and the Drawer stays. It keeps the product focus restoration (to the opener, else an open Drawer's close button, else the route's heading). Without a `closeLabel` its close button takes Astryx's dialog close label. Moving onto ui-common's `Modal` is a follow-up. |
| Tooltip | Astryx `Tooltip`, shown on hover and keyboard focus, dismissed by Escape through the layer stack, adding no box or tab stop. Inside a modal Dialog it renders in the dialog's subtree like the Select list. |
| Drawer | ui-common's lab `Drawer` fork, a native modal `<dialog>`: the page behind it is inert, the layer stack closes it on Escape (ignoring an Escape that ends an IME composition) and it returns focus to its opener. The adapter renders the localized title (`aria-labelledby`) and the close button (`data-autofocus`, focused on open); `width` (`narrow` 320 px, `medium` 520 px) and `side` map onto the Drawer's props, and adapter CSS keeps the inset, rounded sheet. Used for the Models inspector below 1100 px and Chat's sheets. |
| ErrorBanner | ui-common `ErrorState`, keeping the product name and props (`role="status"` for info, `role="alert"` otherwise). |
| AppShell, SideNav, MobileNav | ui-common `AppShell` (`height="auto"`, `variant="section"`, `contentPadding={0}`) through `common-app-shell.tsx`: `ShellFrame` puts the product bar in `topNav`, `ShellSideNav` (a `SideNav` of `SideNavItem` links) in `sideNav`, and a controlled `mobileNav` (`breakpoint: 'lg'`) whose content is `ShellDrawerNav`, the same SideNav rendered as a MobileNav drawer; `ShellMenuToggle` is AppShell's `MobileNavToggle`. The product's `[`/`]` keys, the shortcut suppression inside the drawer and the focus hand-off when the viewport widens stay in `shell.tsx`. |
| Card | Astryx `Card` with `.surface-card` and the product role (`article`, or `region` for a named focusable container). |
| PageLayout | ui-common `PageLayout` (`variant="wide"`), the content column inside AppShell's main landmark. |
| PageHeader | ui-common `PageHeader` on every route, with the route's actions, its stale or action error and a Retry carrying a product `retryLabel`. |
| StatCard | ui-common `StatCard`; its loading skeletons are decorative and the caller keeps one localized status for the waiting tiles. |
| LoadingStatus | Decorative Astryx `Skeleton` rows under one visible, localized `role="status"`. |
| SmoothHeight | ui-common `SmoothHeight`, active while an operation is in flight plus a 400 ms settle window. |
| Badge | Astryx `Badge`: neutral, `accent` and `warning`, with adapter CSS for the readable warning pair and the outlined accent badge. |
| Field, Toggle, Inspector, DenseList, AuthGate, LoginView, SchemaMismatchView | Product components with no package equivalent (`local-components.tsx`). |

Product stylesheets target the adapters' own `ds-*` classes, or a ui-common / Astryx class scoped under one (`.ds-page-header .uic-page-header__title`). Nothing targets ui-common 0.1 class names; tests query by role, label or test id.

## Remaining bridges

A bridge is adapter code that works around what a shared component does not offer yet. Each one carries its reason in the source.

| Bridge | Why it stays |
|---|---|
| PageHeader test ids and focus fallback (a ref applies `data-testid`, `tabIndex=-1` and `data-dialog-focus-fallback` to the h1, description and error block on every render) | PageHeader forwards only `className`. |
| ErrorBanner live region (a ref removes ErrorState's `role="alert" aria-live="polite"` and sets `role="none"` on its h2) | ErrorState takes no role or live-region props; the product section must stay the only live region and add no heading. |
| Drawer Tab wrap at usable first and last stops (capture phase) | Astryx's focus trap counts the contents of a closed `<details>` (the Models inspector's Details disclosure). |
| Tooltip always-present description (a hidden node describes the control; the adapter drops the id Astryx adds) | Astryx mounts its content only on first show, so the description would depend on the tooltip having been shown. |
| Tabs IME guard (composing keydowns stop at the strip) | TabList's roving arrow keys (`useListFocus`) do not check `isComposing`. |
| Dialog stays native | ui-common's `Modal` has not been adopted yet. |

Dropped in 0.2 because the shared component now provides the outcome: the EmptyState heading level, the Select combobox role and `aria-busy`, the Select IME guard (the trigger is not editable; Astryx guards type-to-select and its search input), the native select and in-place tooltip inside Dialog (Astryx layers render inline, inside the open modal), the StatCard skeleton bridge, the Drawer test id, first-open focus, Escape and IME Escape handling and stale-edge bookkeeping, the Tooltip wrapper's `tabIndex=-1`, and the DataTable scroll-box lookup.

## Stylesheets and theme authority

`webui/index.html` links `public/layer-order.css` first, which declares the cascade layers before any other stylesheet is parsed (`reset, theme, base, astryx-base, astryx-theme, ui-common, components, utilities`), and `public/astryx-layer-slide-rules.css` (see below). `src/main.tsx` imports `ui-common-entry.css` (the same layer statement, Astryx's `reset.css` and `astryx.css`, `ui-common.css` and the lab stylesheet), the single theme entry `design-system/themes/index.css`, and `app.css`, which collects every product stylesheet in a fixed order. Product stylesheets sit in `@layer components`, above every Astryx and ui-common layer; `design-system/ua-baseline.css` sits in `@layer base` and restores the browser defaults the product stylesheets were written against. Feature stylesheets are not imported by their modules, so a lazily loaded route cannot reorder the cascade. ui-common's `theme/lablup/theme.css` and `legacy-tokens.css`, and the neutral theme's `theme.css` and `fonts.css` (a remote font), are not loaded.

The product owns the theme. Its two families are Astryx themes over the built `neutralTheme` (`themes/mlxcel.theme.ts`, `themes/glass.theme.ts`), pre-built by `astryx theme build` into `themes/built/`; `lablupTheme` is not used. The app is wrapped in Astryx `<Theme>` with the active family's built theme (`__built: true`, so it injects nothing) and the resolved scheme. The root `<Theme>` writes `data-theme` (the scheme) and `data-astryx-theme` (the theme name) on `<html>`; `applyAppearance` and the pre-paint bootstrap write the same values, and the product family lives in `data-theme-family`. `<Theme>` still carries its runtime theme generator, which it imports statically, so the generator cannot be tree-shaken while the app uses `<Theme>`. See [design-system.md](design-system.md#themes).

## Localization of library strings

The strings Astryx and ui-common components render themselves (the busy Button's loading announcement, sort, close, navigation, skip-link and tab-strip labels, ui-common's Retry and skeleton labels) come from their message catalogs, never from the product catalog. `LibraryStrings` in `common-i18n.tsx` wraps the app (inside `ProductTheme`) in `InternationalizationProvider` from `@lablup/ui-common/i18n` and maps the appearance locale onto Astryx's tags: `en` to `en`, `ko` to `ko-KR`. It follows Settings > Appearance at once.

- English is the built-in catalog of both libraries and fetches nothing.
- Korean is one lazy chunk, `common-i18n-ko.ts`: Astryx's runtime catalog (`@lablup/ui-common/locales/ko-KR.generated.js`) and ui-common's (`@lablup/ui-common/ui-common-locales/ko-KR.json`), merged the way `mergeMessages` merges. It does not import `@lablup/ui-common/i18n-catalog`: that module builds `uiCommonMessages`, every ui-common locale, when evaluated, and the bundler keeps all of it. `loadLibraryMessages()` fetches each locale once and retries after a failure. Until the chunk arrives the library falls back to English text.
- Product strings stay props wherever the product's wording differs from the library's (Select's empty text, the route Refresh actions) or must be correct on the first render (Dialog and Drawer callers pass `common.close`). Unnamed Tabs and a Dialog without `closeLabel` use the catalog.
- Every product string remains a key in the product catalog with its mirror in `tests/fixtures/webui/strings.json`; library strings are not mirrored there.

## Content Security Policy

The served policy is `default-src 'self'; script-src 'self'; style-src 'self'` (plus `img-src 'self' data:`, `font-src 'self'`, `connect-src 'self'` and the framing, base and form restrictions). Nothing may insert a `<style>` element or an inline script, and nothing loads from another origin.

- **Slide rules.** Astryx's popover layer inserts `<style id="astryx-layer-slide-rules">` with its `@position-try` rules unless an element with that id exists. `index.html` links `public/astryx-layer-slide-rules.css` under that id, generated from the installed Astryx core by `webui/scripts/layer-slide-rules.mjs`; `ui-common-dependencies.test.ts` fails when it is stale.
- **Themes.** Only pre-built themes reach `<Theme>`, so it injects nothing.
- **CSSOM allowlist.** Every other dynamic value Astryx writes goes through CSSOM, which `style-src 'self'` permits. `expectSafeLayout` in `tests/browser-assertions.ts` rejects any `style` attribute except these library writes, each scoped to the element that makes it: the ProgressBar fill width; popover layers' anchor positioning and margins; a popover anchor's `anchor-name` (Selector trigger, a Button with a tooltip) and the layer's anchor marker; StyleX custom properties on fields, skeletons and the Drawer sheet; the Button spinner box; a fixed table column's `min-width`/`width`; AppShell's measured bar height; the MobileNav sheet width; the MobileNav scroll lock on `<html>` and the modal Drawer's on `<body>` while open; and the adapted SmoothHeight's measured height.
- **Off limits.** Astryx features that inject styles at runtime: non-built themes, CodeBlock highlighting, Chat stream scrolling, the native DateInput.

The served-CSP check runs against a server that sends the real headers:

```sh
MLXCEL_WEBUI_CSP_URL=http://127.0.0.1:8080/webui/ pnpm --dir webui exec playwright test --config playwright.csp.config.ts
```

The config starts no preview server and injects no headers. It requires the response's strict CSP, observes `securitypolicyviolation` events and external requests, opens the Selector list, the compact navigation drawer and the gallery Tooltip, measures progress geometry, and asserts that no `<style>` element was inserted and the slide-rules link is present. Record which server was tested.

## Bundle budget

`scripts/webui/build_bundle.py` limits the initial JavaScript to 256 KiB (262,144 bytes) gzip, all JavaScript to 700 KiB gzip and the embedded assets to 5 MiB; `src/webui/assets/mlxcel-webui-manifest.json` records the current measurements. The first bundle holds the shell and what a fresh load shows (the session-key prompt and the connection surfaces; the key lives in memory only, so every load starts signed out). The Models library, Chat, Activity, Settings and the gallery are route chunks fetched the first time each renders and rendered synchronously once loaded (`preloadRoutes()` in `app.tsx`, which tests await once); after the first paint an idle callback prefetches them. The Korean library catalog is a chunk of its own. Keep new package imports out of the first-paint modules.

## Recorded decisions

- `usePrefersReducedMotion` is not adopted. It reads only `matchMedia('(prefers-reduced-motion: reduce)')` and cannot see the in-app Settings toggle, which writes `data-reduce-motion`. CSS stays the only reduced-motion mechanism: `tokens.css` sets the motion tokens to 1 ms and `components.css` stops transitions and animations globally, which also covers the Drawer, Tooltip, SmoothHeight, Skeleton and Card. StatCard's `animate` count-up stays off for the same reason.
- `formatCompactNumber` is not adopted. `webui/src/design-system/format.ts` threads an explicit `ko-KR` or `en-US` locale through `Intl.NumberFormat` and stays the single formatter; values reach StatCard as formatted strings.

## Verification boundary

Typecheck, lint and the Vitest suite cover props, refs and submit behavior, busy controls, lifecycle mappings, finite and unknown progress, heading levels, tab ids and selection, the Tabs IME guard, the Select list and Tooltip inside a modal dialog, typed tables with row activation and overflow regions, the Drawer keyboard contract and Escape ownership, the library strings in English and Korean, the layer order and stylesheet entries, the single Astryx core and the shipped slide rules. Unit tests run in jsdom with a default `matchMedia` (`src/test-setup.ts`); they do not establish layout, native dialog inertness, CSP or VoiceOver correctness. The browser suite (`pnpm --dir webui run browser`) adds axe, keyboard, reflow, 44 px-target and the CSSOM allowlist against the real DOM, and the served-CSP check above covers the policy.
