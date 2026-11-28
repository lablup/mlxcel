# Technical Report: PR #1975, residual LOW items in the Models and Activity pages

**Date**: 2026-09-25

**Status**: Implemented, reviewed and validated locally and against a real served binary; merged by the chain run that produced it.

**Languages**: TypeScript/React

**Risk level**: Low

## Executive summary

PR #1975 closes issue #1974, which grouped the LOW-severity items left open by the hardening of PR #1930 (Models library redesign, #1918) and PR #1929 (Activity status page, #1916) under epic #1910. They were grouped because they share one WebUI test surface and one CI cycle, including the GB10 installed-artifact job. The change touches only the WebUI and its generated bundle; no Rust, API or schema changed.

## Problem statement

Each item was either a latent defect or an accessibility gap that the two redesign PRs deliberately deferred:

- Below 1100 px the page behind the Models inspector drawer stayed in the accessibility tree and focusable, unlike Chat and the navigation sheet, which set `inert`.
- The drawer adapter stopped every Tab inside the panel in the capture phase, so no descendant handler (a ui-common `Select`, the Dialog trap) could ever see Tab.
- Row labels and tooltips interpolated model names without Unicode isolation.
- After a row Unload with no search active, the lifecycle pin re-sorted the row onto a later page, so the focused Load button left the visible page and the focus intent stayed armed forever. The route remount also discarded the search, filters and sort on every Chat round trip.
- The Activity slot box was a tab stop even when nothing scrolled; formatters and the operations sort were rebuilt per call or per poll; plural rules read the unrounded value; a zero context read as "0 tokens"; and re-selecting a model showed its old tiles under a "stopped refreshing" banner.
- The StatCard loading skeletons had never been exercised under the served CSP.

## Change summary

- **Models:** a `display: contents` wrapper marks everything before the drawer `inert` while it is open below 1100 px. Row labels go through `isolate()`, and the tests build their locators from the same helper. The row-focus effect finds the row's index and turns the pager to it. It remembers the page the row was last seen on, so it follows a re-sort but stops when the user pages away. One-shot `once` intents never page. The search, filters and sort live in a new module store, `library-view.ts`. The store is scoped to the server instance it was set against and is cleared on logout. Callback refs are cached per control key, and `consumeInspectorRequest` now notifies its subscribers.
- **Drawer adapter:** it records alpha.19's first and last focusable elements at open, using the package's own selector. It stops Tab propagation only when it wraps, when there are no stops, or when focus is on one of those stale package edges.
- **Overflow region:** `useOverflowRegion` is generalized to take a scroll element, check both axes, and accept either a label or `labelledBy`. DataTable and the Activity slot box share it, so a box becomes a named, focusable region only while it actually overflows.
- **Activity and formatting:**
  - `design-system/format.ts` exports a `cached()` helper keyed on locale and fraction digits.
  - `Operations` memoizes its sort and counts.
  - `Intl.PluralRules` uses the display rounding.
  - A context of zero or less reads "unknown".
  - `RuntimeView` shows the waiting state until the first sample after a re-select.
- **CSP:** the gallery gained a loading StatCard, and `csp.spec.ts` asserts its decorative skeleton is visible before it checks for violations.

## Technical decisions

- **Stale-edge Tab stop instead of stop-on-wrap only.** alpha.19 captures its edges once at open and wraps early whenever focus sits on them. Stopping only on an actual wrap would let that stale handler misfire, so the adapter mirrors the package's snapshot and keeps the stop exactly where the package would act.
- **Instance-scoped session store instead of component state or persistence.** Component state loses the view on every route remount. Persistence would outlive a logout. A module store keyed to `serverInstanceId` keeps a Chat round trip intact without leaking a previous user's query, or a filter the new catalog cannot satisfy. The security review raised this at MEDIUM, and it was fixed before merge.
- **Page-follow with a remembered page.** The first implementation treated focus on `<body>` as "still waiting". In Safari and Firefox on macOS a click on Next does not focus the button, so the effect turned the page back on the user. Recording the row's last page separates a re-sort, which should be followed, from a user page change, which ends the intent.
- **Waiting state instead of reducer deletion for re-select.** Deleting the cached runtime would blank the Models inspector, which reads the same map, so the fix stays in the Activity view.

## Validation

- **Local checks:** typecheck and lint pass, the unit suite passes (35 node tests and 710 Vitest tests), and the full default Chromium Playwright suite passes (124 tests). `build_bundle.py --verify`, the contract check and the binary-asset check also pass.
- **Tests fail without their fixes:** every item's new test was confirmed to fail with its fix reverted. Item 11's skeleton assertion was judged to fail on main by inspection only, because main's gallery renders no StatCard.
- **Real-server CSP run:** `tests/csp.spec.ts` passed in the light and dark themes against `mlxcel-server` built from `dc1845d3` (`--release --features cuda`, GB10, embedded bundle). The served policy was `style-src 'self'` with no `unsafe-inline`.
- **CI:** CI for the first push (`70c1029b`), including `WebUI installed artifact`, was green.

## Remaining items

- **Models screen cohesion:** `screen.tsx` is about 720 lines, within the 800-line threshold in `docs/code-guidelines.md`, but one component holds page state, the row-focus effect, the column renderers and the filter toolbar. Splitting it was out of scope here; #1976 addresses it.
- **Overflow hook timing:** the hook resolves the inner table once, when its ref attaches. The Activity slot box still updates because its bounded height changes the box's own size.
- **Focus after shrink:** focus falls to `<body>` if a focused scroll box stops overflowing.
- **Unbounded cache helper:** the exported `cached()` has no size bound. Every current key set is small and fixed.
