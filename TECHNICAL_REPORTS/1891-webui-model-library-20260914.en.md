# Model library WebUI — PR #1891

Date: 2026-09-14. Status: open PR, implementation review complete; integration gates remain open. Issue: #1844, parent epic: #1834.

## Scope and architecture

The Models route now renders the local catalog after the existing authentication and schema guards. It imports the shared provider rather than creating another fetch client, bearer-token store, selection authority, support registry, or model lifecycle. The npmjs-published `@lablup/ui-common@0.1.0-alpha.19` package supplies Button, Select, DataTable, StatusTag, ProgressBar and EmptyState through the existing adapters; product dialogs use the documented native top-layer composition. No upstream component source is copied.

`features/models/policy.ts` derives admission from a current authoritative snapshot, server action availability, lifecycle and removal eligibility. `screen.tsx` coordinates explicit actions; `dialogs.tsx`, `inspector.tsx` and `operations.tsx` present confirmations, metadata and server observations. Feature CSS changes layout only and retains the approved shared macOS design. English/Korean keys are isolated under `models.library.*` and included in the canonical string fixture.

## Operator workflows

Local search, source/task/lifecycle filters and deterministic sorting never contact a remote model marketplace. The typed table renders at most 25 inventory rows per page, retaining the shared opaque selection independently of filtering. The inspector distinguishes architectural support, actual backend availability, checkpoint validation, completeness, disk bytes and estimated memory. Null values remain unknown. Effective context is only shown when a matching runtime revision supplies a numeric value; no theoretical family maximum is substituted.

Load uses the current model revision and a new idempotency key. HTTP 202 does not change lifecycle locally. A failed asynchronous load conflict offers explicit inspection of active models and a selected idle eviction target; it does not assert every conflict is caused by capacity or automatically retry. Unload confirmation explains active-request drain and worker exit separately from deleting files. Cache deletion is only offered for server-eligible managed cache entries, requires the exact display name and preserves the revision/server instance captured when confirmation opened. User-provided roots never offer disk deletion.

Add Model requires an explicit public-ungated acknowledgement, canonical repository/revision syntax and network/destination disclosure. Rejected requests retain entered fields and the server error. Observed operations provide byte progress or an indeterminate indicator, cancellation and failed/cancelled download retry with a new explicit request. Single-model mode disables lifecycle, library mutations and rescan while retaining observation refresh and startup guidance.

## Review corrections and shared dependencies

Two independent read-only reviews identified and corrected feature-local gaps: normal capacity failure is reported after 202 rather than necessarily as HTTP 409; single-model rescan must be disabled; repository/ref bounds must match the server; rejected downloads must retain input. The browser fixtures were corrected to use canonical opaque IDs, validate whole JSON responses before fulfillment and return a real 404 for a removed runtime model instead of fabricated success.

The reviews also exposed shared integration prerequisites, tracked by the central epic coordinator rather than silently duplicated here:

- #1846 owns the transferred native-dialog unmount focus repair and eight regression tests, plus mutation authentication-session fencing and the canonical load-profile adapter.
- #1847 owns complete-catalog reconciliation of a removed selected model so runtime polling cannot remain in a 404/offline loop.
- #1845 owns selection abort granularity so inspection does not cancel in-flight inference or mutation transport.

Current load requests deliberately use server defaults. The canonical `useLoadProfile` hook must be wired after #1846 merges, before this issue is accepted. The current inspector labels that boundary explicitly; no duplicate settings authority is committed.

## Executed validation

At source commit `24e256ff16907f4b543d68c499b80641b69fbb67`:

| Gate | Result |
|---|---|
| Frontend unit suite | 111 passed: 76 existing plus 35 Models tests |
| TypeScript and ESLint | Passed |
| Canonical UI contract | 46 whole fixtures, DTO drift and strictness passed |
| llama compatibility, crate versions, kernel dtype-key structure | Passed |
| Canonical bundle build and two-clean-build verification | Passed; digest `46a65b2a2ad10967f271caf5706b210fa8f2ee5e43dcdb061005224ba42c5b99` |
| `git diff --check` | Passed |

The additional eight shared modal tests passed before their patch was transferred to #1846 and removed from this PR; they are not included in the 111-test count. The 35 Models tests cover null metrics, stale/offline/auth/schema admission, ownership/deletion, capability truth, active drain versus busy eviction, unknown POST fences, CJK filtering, duplicate clicks, other-tab revision changes, cancellation, server restart, server-reported failures, 202-followed-by-failed capacity recovery and retained rejected download fields.

## Remaining acceptance

Three browser cases are authored for the mocked download/load/select/unload/remove journey and large CJK inventory keyboard/accessibility checks at desktop light and compact dark/opaque viewports. They were not run locally because the central coordinator reserves this Mac's GPU. Hosted execution and changed signed-in Models screenshot review are pending; no screenshot baseline was updated without review. Mock HTTP responses do not demonstrate a real checkpoint download, inference, drain or disk deletion.

The root integration gate must run the complete live browser journey using an isolated newly downloaded small checkpoint after the shared prerequisites and profile seam merge. Existing user checkpoints must remain untouched. Safari/VoiceOver and actual 200% page zoom are deferred to one final integrated manual session by maintainer decision, not recorded as passed. No CUDA/GB10 or broad Metal result is claimed by this frontend unit.

## Hosted collection correction

The first hosted run, `34848567605` / WebUI job `103990487018`, failed during Playwright collection because a Vitest fixture module imported JSON without Node import attributes. No browser case executed in that run. Follow-up source `92aa8978` reads fixtures through filesystem APIs and uses the existing TypeScript compiler to execute the unchanged canonical validator with its Vite-only raw schema import supplied from disk. Only repository-owned source is executed; response/user values are never code. Independent review found no loader blocker.

After correction, browser collection lists 22 tests without launching Chromium. The local unit count is 112 (76 existing plus 36 Models/fixture tests); typecheck, lint and deterministic verification pass. The updated source-bound bundle digest is `cd681decc1bfffd6fd296deaa22d8dae8db4324421b23f7192381bfc84934a10`. Actual hosted browser results, shared prerequisite integration, live checkpoint operations and final manual accessibility are still open.

## Executed hosted browser evidence and reflow repair

Run `34849720039`, job `103994435003`, executed 23 browser tests: 20 passed and three failed. Both new large-CJK inventory keyboard/axe/layout cases passed. The operation journey reached deletion, then correctly failed because the known selected-runtime 404 left pending reconciliation uncleared; #1847 must resolve that prerequisite without relaxing the assertion. Two signed-in screenshot comparisons also failed. Root visual review accepted the general desktop direction but rejected compact clipping; no baseline was approved or replaced.

The compact defect came from an opaque operation title widening an implicit grid track. The corrected layout uses an explicit zero-minimum track, wraps operation identifiers and action groups, and fills the available table width when no inspector is open. It does not hide overflow. Operation titles now prefer the catalog display name, retaining the opaque ID separately. The existing signed-in screenshot test now runs safe-layout and axe assertions after login, not only before login; its operation content exercises the missing case. Local frontend validation after this repair passes 113 unit tests, typecheck and lint; corrected hosted rendering still requires review.

## Scoped automated visual-baseline approval

Corrected source `7316d025` ran in hosted workflow `34850899647`: 20/23 browser tests passed; the exact signed-in post-login axe and safe-layout assertions passed before the two screenshot comparisons. The remaining journey failure is still the unchanged pending-reconciliation assertion requiring #1847. Root directly inspected both corrected desktop-light and compact-dark/Korean actual images and approved updating only `linux/1440-light-product-signed-in.png` and `linux/390-dark-product-signed-in.png` from that run. Gallery, login, macOS baselines and comparison tolerances are unchanged. This is automated visual-baseline review by the root coordinator, not a new user design approval or Safari/VoiceOver/native-zoom pass.

The reflow-corrected source passed 113 local frontend tests; its verified bundle digest is `bb9fe1783f99d3e9cbbc7dc2b73bb11377ad4a059282f99ffdc73327caa5ee42`. The scoped image/report update changes neither production source nor bundle. Hosted verification will rerun with the approved images; the known shared prerequisite and real/manual gates remain open.

## Canonical profile integration (2026-09-15)

Rebased onto merged Settings #1892 (`2a91bf30`) without stacking unmerged Activity. Models now consumes the canonical `useLoadProfile` store for the actual explicit load target. Model profiles replace reusable profiles; empty profiles omit `load_profile` so server policy remains authoritative. Submitted requests receive a value snapshot. Capacity recovery resolves its frozen target ID separately from current selection, including a new profile saved while the confirmation remains open. The inspector distinguishes pending values from effective running configuration and links to Settings; profile edits never load a model.

Validation: 161 frontend unit tests in 20 files, typecheck, lint, 47 canonical fixtures and two-clean-build bundle verification passed. Three new canonical-store regressions cover scope replacement/reset, immutable requests, non-loading selection/save and capacity target drift. Bundle digest: `d5b77e0fa0a6482ede4c2c0f4b93736009b2a72d3abea303498fe2352eb1a435`. No browser baseline or assertion was changed. Activity's deletion reconciliation prerequisite, actual isolated-checkpoint journey, hosted integrated validation and deferred native Safari/VoiceOver/200% checks remain open. GPU timeout root-cause investigation was deferred by the maintainer; that does not constitute a GPU validation pass.

Independent security review identified ambiguous display-name deletion confirmation and eviction selection revision drift. Deletion now displays the checkpoint name and exact opaque cache ID, requiring the ID; duplicate-name/wrong-ID regression passes. Eviction selection now freezes the victim ID and revision and rejects a newer revision before submission, even if still ready and idle. This is a client observation fence, not atomic backend compare-and-swap: the API carries no expected eviction revision, so a post-submission replacement race remains a backend contract limitation. The browser deletion journey changes only its typed token to the exact ID; the pending-reconciliation assertion remains unchanged.

## Merged Activity integration (2026-09-15)

Rebased onto merged Activity #1893 (`f092e7e2`), preserving Settings, observation/session fences, and the unchanged deletion pending=0 assertion. The prior `3e0a018b` hosted run `34914596214` passed 24/25 browser tests; its sole failure was that exact pending-reconciliation prerequisite, not a screenshot regression. Both approved signed-in Linux baseline comparisons passed.

The combined source passes 181 Vitest tests in 23 files plus 17 Node parser/harness tests, typecheck, lint, 49 canonical fixtures and two-clean-build bundle verification. Combined bundle digest: `9b3add2d1751f4723e736d05a30ae53e3e0dacac44bc97c6a827aa13a05377d3`. Vite warns about the uncompressed main chunk exceeding 500 kB; the enforced compressed and total asset budgets pass unchanged. Final hosted and root actual isolated-checkpoint acceptance remain pending; no browser/server/GPU work was run by this unit.
