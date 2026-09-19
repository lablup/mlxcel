# Shared ui-common integration

All WebUI feature pages consume `webui/src/design-system/primitives.tsx`; they must not import package components directly, fork adapter CSS, or create a second auth/cache layer. The dependency is pinned exactly to `@lablup/ui-common@0.1.0-alpha.19` (React peer dependencies only). The npm `latest` tag points to an older release, so unversioned installs are not equivalent. The lockfile retains the published tarball integrity; source is not vendored and no sibling checkout is required.

## Export and exception matrix

| Product export | Published component / ownership |
|---|---|
| Button / IconButton | Common Button, with supported props only, explicit busy/disabled/ARIA mapping, refs and submit semantics. Unsupported native props such as `onFocus`, `name`, `form` and arbitrary data attributes are deliberately not advertised. Extend the central adapter explicitly if a real consumer needs them; never cast a broad native prop bag. |
| StatusBadge | Common StatusTag with domain lifecycle mapping and unchanged English/Korean domain labels. |
| ProgressBar | Common ProgressBar; undefined/nonfinite means indeterminate, finite percentages clamp to 0–100, and the product label remains the accessible name. |
| EmptyState | Common EmptyState with product illustration/actions; an explicit `role=heading` / `aria-level=2` ref bridge preserves the product hierarchy because alpha.19 fixes its DOM title at h3. |
| Tabs | Common controlled segmented Tabs, IDs prefixed with `useId`, callbacks translated to caller IDs, empty-array handling, IME capture guard and wrapping rather than a mobile dropdown. Callers provide unique IDs within each tabset. |
| Select | Common Select for normal pages, with localized labels and field help/error/busy state. An explicit ref bridge completes the combobox role because alpha.19 supplies active-descendant/listbox attributes on a plain button without exposing a role prop. |
| DataTable<T> and column/state types | Typed common table seam with required accessible label, stable caller-supplied opaque row keys and caller-owned column/sort state. Models and Activity must use this export. No automatic storage or API pagination/filtering is introduced. |
| StaticTable | Legacy semantic markup table retained only for the existing gallery fixture. New feature tables must use DataTable<T>. |
| Dialog / Sheet | Retained native `showModal` top-layer/focus implementation. Common Drawer is right-only and is not equivalent to the product's left sheet or modal contract. |
| Select inside Dialog / Sheet | Native select selected by NativeModalContext, because alpha.19 unconditionally portals popups into body without a portal-container option. A z-index override cannot repair top-layer ancestry. |
| Tooltip | Retained local described-by tooltip, including inside native modal descendants; common body-portalled Tooltip cannot satisfy that boundary. |
| Field / password login / auth / error banner / inspector / list / shell / icons | Product-specific semantic wrappers. No equivalent package control, or the package's forced alert/action semantics do not match the compact product surface. |

The semantic ref bridges are narrowly scoped compatibility workarounds for the pinned API, not claims of independently verified assistive-technology behavior. Re-audit them when upgrading. Product icon SVGs remain project-authored. The bundled `third-party-licenses.txt` includes the package's published NOTICE and Apache-2.0 license verbatim.

## Theme authority

Only the library base stylesheet, one product theme stylesheet and component subpath styles are imported. `src/main.tsx` imports `@lablup/ui-common/styles/base.css` and then `design-system/themes/index.css`, the single theme entry, following the package rule of at most one theme file alongside `base.css`; no component imports a theme file. The product themes (`mlxcel-light`, `mlxcel-dark`, `glass-light`, `glass-dark`) are `[data-theme]` blocks over the published token contract, selected by the applied theme id; see [Themes](design-system.md#themes). `common-tokens.css` keeps the theme-independent half of the contract (typography, spacing, radii, control heights, motion, z-index) on `:root`. `common-components.css` scopes necessary DOM/geometry overrides to adapters, including primary/danger foregrounds, compact 44px targets and wrapped controls. No orange theme stylesheet or remote font/resource is loaded. Keep the approved macOS 27 shell topology, continuous toolbar and neutral content surfaces; package adoption is not approval to substitute its default PageLayout or theme.

## Verification boundary

Typecheck, lint and 72 unit tests pass at the migration checkpoint, including props/ref/submit behavior, busy controls, all lifecycle mappings, finite/unknown progress, heading roles, duplicate tabset IDs, IME suppression, modal select exceptions and typed tables. Unit tests use jsdom, including an explicit scrollIntoView mock; they do not establish layout, native select behavior, CSP or VoiceOver correctness.

The normal hosted browser suite continues strict screenshot, axe, keyboard, reflow and 44px-target checks. Only narrowly identified package CSSOM geometry properties are allowed in the static DOM assertions: progress width, select popup position/top/left/width and common table header width. This does not certify production CSP. No screenshot tolerance is relaxed, and changed images require explicit visual review rather than automatic baseline updates.

Run the separate actual-server check only after a healthy secured server containing the rebuilt bundle is available:

```sh
MLXCEL_WEBUI_CSP_URL=http://127.0.0.1:8080/webui/ pnpm --dir webui exec playwright test --config playwright.csp.config.ts
```

The config starts no preview server and supplies no copied/injected headers. It requires the real response's strict CSP, observes securitypolicyviolation events and external requests, measures popup/progress CSSOM geometry, and checks keyboard close/reopen focus. A standalone server mounting the real security middleware proves that middleware boundary, not the full production CLI startup. Record which server was tested.

The previous immutable 4185 source (`c4df0326…`) remains design-approved and has user-reported targeted manual acceptance. That acceptance does not automatically cover rewritten ui-common DOM. New served-CSP, light/dark/compact visual, keyboard and relevant Safari/VoiceOver/native-zoom observations remain required before accepting the migration. No local browser/GPU test was run while the root reserved host recovery and model validation.
