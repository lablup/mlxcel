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
