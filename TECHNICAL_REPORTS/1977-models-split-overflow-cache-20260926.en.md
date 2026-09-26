# Technical Report: PR #1977, Models screen split, overflow hook and format cache

**Date**: 2026-09-26

**Status**: Implemented, reviewed and validated locally; merged by the chain run that produced it.

**Languages**: TypeScript/React

**Risk level**: Low

## Executive summary

PR #1977 closes issue #1976, the four items left in the "Remaining" section of the PR #1975 report. It splits the Models screen along its existing seams without changing behavior, makes the shared overflow-region hook follow late or swapped tables and keep a focused tab stop until blur, and bounds the shared Intl formatter cache. It also corrects the #1975 report, which cited a 500-line module rule that the repository does not have: `docs/code-guidelines.md` treats files under 800 lines as fine. The split is justified by cohesion, not size.

## Problem statement

- `ModelsLibrary` was one 731-line function holding row focus, column definitions, the toolbar and the page composition.
- `useOverflowRegion` looked up the inner `<table>` once, when its ref attached. No current caller mounts its table later, but a future caller that did would never be observed.
- When a focused scroll box stopped overflowing, the hook removed its `tabindex`, and focus fell to `<body>`.
- The exported `cached()` helper had no size bound, and `features/chat/time.ts` carried its own private, unbounded copy.

## Change summary

- **Split (its own commit, `e79df10c`):** the screen now uses three new modules:
  - `row-focus.ts`: a `useRowFocus()` hook plus the `AFTER_LOAD` and `AFTER_UNLOAD` constants.
  - `columns.tsx`: `libraryColumns(context)`.
  - `toolbar.tsx`: `LibraryToolbar`.

  `screen.tsx` shrank from 731 to 433 lines. The moved code matches the old code apart from the wrapper lines, and the effect order is unchanged. The existing tests pass unmodified.
- **Overflow hook:**
  - A `MutationObserver` watches `childList` across the region's subtree and re-resolves the observed table.
  - The overflow check it triggers is coalesced into one `requestAnimationFrame`. Row changes from polls and SSE bursts therefore cost at most one layout read per frame.
  - Text-only updates do not wake it, because it does not observe `characterData`.
  - While the box has focus, the tab stop is kept. A single `blur` listener re-runs the check.
- **Cache:** `cached()` holds at most `CACHE_LIMIT = 32` entries and evicts the oldest insertion first. `chat/time.ts` now uses the shared helper.

## Technical decisions

- **Coalescing the mutation-triggered check.** The first version ran the check synchronously in the observer callback. That forced one layout read per React commit during an SSE burst. The implementation review raised this at MEDIUM and it was fixed before the push. The attach-time check and the ResizeObserver path stay synchronous.
- **Keeping the tab stop until blur instead of moving focus elsewhere.** Models names its region with a label and has no heading, and Activity's `<h3>` cannot take focus. There is no common target that focus could move to.
- **FIFO bound instead of a restricted key type.** Callers build keys from numbers, so a type-level restriction to fixed keys cannot hold.

## Validation

- **Local checks:** typecheck and lint pass, the unit suite passes (35 node tests and 713 Vitest tests), and the full default Chromium Playwright suite passes (124 tests). `build_bundle.py --verify` and the binary-asset and contract checks also pass.
- **Tests fail without their fixes:** each new hook and cache test was confirmed to fail against main's source.
- **Reviews:** the security review found no CRITICAL or HIGH issues.

## Remaining items

- **Unsupported environments:** where `ResizeObserver` is unavailable, the `MutationObserver` is not set up either. This does not apply in current browsers.
- **Window blur:** `blur` also fires when the whole window loses focus, so a focused box that has shrunk drops its tab stop at that point.
- **Shared limit:** `CACHE_LIMIT` is shared by every cache map. The current callers stay well under it.
