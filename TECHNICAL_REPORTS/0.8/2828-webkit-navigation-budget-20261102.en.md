# PR #2828: Calibrate the headed WebKit navigation feedback budget

**Date**: 2026-11-02
**Status**: Completed (PR open; closes #2821)

## Executive Summary

PR #2828 resolves the red `WebUI bundle` job by naming the cause from measurements rather than relaxing the product budget. The headed WebKit project now allows 150 ms for navigation feedback while Chromium and Firefox keep 100 ms. The evidence points to WebKit 27.2, shipped with Playwright 1.64.1: the event-to-semantic-DOM half stayed at 30 to 34 ms, the two-rAF paint tail moved from 36 to 57 ms to 74 to 80 ms, and the identical tree at 95ea652a both passed (46 ms) and failed (105 to 110 ms). React 19.3.1 was measured separately and is not the cause.

## 1. Problem Statement

CI run 41321895260 failed `tests/performance.spec.ts:290` in `webkit-headed-performance` on all three attempts: 108, 110 and 105 ms against a 100 ms budget. The job had passed on main before #2819 moved Playwright 1.63.0 to 1.64.1 and React 19.3.0 to 19.3.1. Without a cause, the choice was between keeping a budget that fails on every run and raising it without evidence.

## 2. Evidence

| Arm | Result |
|---|---|
| Main CI at 95ea652a (run 41322479478) | passed at 46 ms |
| #2819 PR run 41321895260 | failed at 108, 110, 105 ms |
| Pre-#2819, WebKit 26.6 / v2359, seven runner samples | totals 62, 64, 64, 87, 89, 90, 98 ms; event to predicate median 33 ms; predicate to paint 36 to 57 ms |
| Post-#2819, WebKit 27.2 / v2370, seven attempts | totals 46, 69, 105, 107, 108, 108, 110 ms; event to predicate median 31 ms; predicate to paint 74 to 80 ms in every failing attempt |
| Local Chromium, React 19.3.0 | 31.5 to 31.7 ms |
| Local Chromium, React 19.3.1 | 28.6 to 35.4 ms, same 5.9 to 6.3 ms event to predicate |

The application half is unchanged between the passing and failing runs: the same tree passes and fails, and only the paint tail moved. That makes the engine the cause. React is ruled out by the controlled Chromium arm.

## 3. Change Summary

| File | Change |
|---|---|
| `webui/tests/performance.spec.ts` | `WEBKIT_HEADED_NAVIGATION_FEEDBACK_BUDGET_MS = 150`; the assertion and the evidence field `feedback_budget_ms` use the project-specific value; a comment records the pre/post numbers and the 95ea652a control |
| `webui/playwright.engines.config.ts` | Comment states the engine-calibrated WebKit exception and keeps the existing retry rationale |

## 4. Technical Decisions

### 4.1 The exception is scoped to the engine, not the product

Chromium and Firefox keep the 100 ms budget. Only the headed WebKit project gets 150 ms, and the effective value is written into the evidence file so a future reader can tell which budget produced a sample. A global increase would have hidden a real navigation regression on every engine.

### 4.2 The numbers live next to the threshold

The comment carries the engine versions, the split between event-to-DOM and paint, the passing and failing totals for the identical tree, and a link to the issue. A threshold change without its measurements is indistinguishable from a workaround, and the next dependency bump would otherwise re-open the same investigation.

### 4.3 React is not the cause, and that was measured

The controlled Chromium arm moved React 19.3.0 to 19.3.1 with the same event-to-predicate time. This rules out the other half of #2819 without disturbing the committed lockfile.

## 5. Verification

| Check | Result |
|---|---|
| Modified spec on local Chromium, three runs | pass, evidence shows `feedback_budget_ms=100` |
| `pnpm --dir webui run typecheck` | pass |
| `pnpm --dir webui run lint` | pass |
| PR CI run 41410786629, `WebUI bundle` job 125163183335 | success with the 150 ms WebKit budget |
| `pnpm install --frozen-lockfile` restore | pass, no package.json or lockfile churn |

## 6. Unverified

Local WebKit cannot launch on this Apple Silicon host (`browserType.launch` times out with an NSXPC listener error, headed and headless, sandboxed and not), so the Playwright arms are CI runner samples rather than a controlled local bisect. The final acceptance item, `WebUI bundle` green on main, can only be confirmed after this merge.

## 7. Learning Points

Two CI results from the same tree are more informative than two commits: one passed and one failed, so the difference had to be in the environment or the engine, not in the diff. Measuring the event-to-DOM and paint halves separately localized the movement to paint scheduling.

## 8. Risks

Low. The change touches one test and one config comment. The risk is that 150 ms is too generous for the headed WebKit arm, but the event-to-DOM half remains at 30 to 34 ms and the other engines keep 100 ms, so a navigation regression still has to clear a meaningful bar.
