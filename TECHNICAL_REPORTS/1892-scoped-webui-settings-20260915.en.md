# Technical report: PR #1892 — Scoped WebUI settings

**Date:** 2026-09-15

**Status:** Implementation reviewed; CPU/model-free and automated browser gates passed. Root-owned isolated actual Settings acceptance passed. The maintainer explicitly deferred GPU recovery root-cause investigation; its unknown cause is not treated as a WebUI merge blocker, and no unexecuted feature assertion is marked passed.

**Source checkpoint:** `8308a79fcc18e273cd2153f7fc6d01dc581e13f7` (cfg-only repair after the separately identified `153d3c73` gates).

## Design and scope

Issue #1846 adds separate browser/privacy, request-generation, loaded-model, next-load and restart-only settings. Published npm package `@lablup/ui-common@0.1.0-alpha.19` supplies the existing shared controls; this change neither vendors the package nor creates another visual system. Native Field/textarea/modal are existing documented exceptions. Request defaults stay in memory, system prompts remain conversation-owned, and only explicitly saved safe load profiles persist in browser preferences.

Live settings consume the actual opt-in `/settings` schema/current/fingerprint through the authenticated client with exact inference identity and `autoload=false`. Numeric range authority stays on the server because its schema currently exposes types/enums/help rather than min/max. A preflight fingerprint comparison detects observed external writes, not atomic CAS. Mixed PATCH results clear applied drafts, retain rejected drafts/errors and refetch current values. Unknown outcomes are not retried automatically.

## Load operation architecture

The existing canonical LoadProfile DTO moves into a narrow adapter. Only bounded context, parallelism and existing KV enum names are admitted; arbitrary paths, credentials, argv and distributed/worker knobs are excluded. CLI/environment pins precede the operation profile, then preset/default resolution. Existing startup and family-specific KV validators run on a cloned startup configuration; incompatible substitutions are rejected before eviction. The actual loaded AppState, not the reusable entry template, supplies effective configuration afterward. The profile participates in idempotency identity and affects only its operation, never a running worker or subsequent unprofiled load.

Filesystem-backed profile resolution runs on the blocking pool after coordinator admission, bounded by the existing 64 active-operation limit. This avoids blocking Tokio request execution while retaining immediate canonical 422 failures, unchanged stale-entry checks and explicit eviction semantics. The generic Tokio task queue itself is not claimed to be bounded by this admission limit.

## Consumer seams and review corrections

Settings owns `generation-defaults.ts`, memory-only `useGenerationDefaults()` and versioned `useLoadProfile()` for downstream Chat/Models integration. Model-specific profiles replace reusable profiles for that model; editing/resetting a scope now operates on that scope rather than silently changing the selected model. `/props` shows actual per-slot context, configured slots and active KV mode, including unknown zero/missing context. Optional raw tokenization excludes chat templating/history/media and cannot certify a full chat request fits. Unified KV is separate #1815 work.

Independent correctness review corrected scope-specific reset/display, missing observed KV mode and stale asynchronous settings responses. Independent security/performance review identified synchronous metadata I/O and cleared the blocking-pool correction. Coordinated shared fixes include eight modal-unmount focus regressions and two late accepted/rejected-operation authentication-session regressions; no browser focus proof is inferred from DOM unit tests.

## Follow-up acceptance corrections

A standalone Settings model selector now uses the shared selection action, making existing ready-model props/live settings reachable before the downstream Models page merges. It never loads or unloads a model.

The initial centralized full run at `035cfe5b` found an actual immutable-cache regression: Vite emitted `index--ARMeYuQ.js`, and splitting on the last hyphen discarded a valid leading base64url hash character. Output now explicitly pins all emitted JS/chunk/CSS hashes to eight bytes; the server recognizes the delimiter immediately before the final eight ASCII base64url bytes only inside `assets/`, with a nonempty basename. Thirteen narrow CPU asset tests pass, including leading/internal hyphens and underscores, malformed/non-ASCII negatives and unhashed metadata. Unhashed assets remain `no-cache`; the fix does not mark all assets immutable.

At `64d4ba42`, hosted browser validation passed 20 cases but failed both strengthened ready-model Settings cases. The unchanged local cases subsequently failed too. Actual Chromium traces proved that the props textbox existed with value 2048 and live settings existed: their accessible names incorrectly included help text from the wrapping label. This was a shared Field naming defect, not missing props, lost selection or a backend transport failure. An explicit visible-label span ID and input `aria-labelledby` preserve native label association, layout and hint/error `aria-describedby`. The computed-name regression fails before the fix and passes afterward using the already installed test axe engine; no dependency or browser locator assertion was changed. Independent review cleared this correction.

## Executed validation

- `pnpm --dir webui typecheck`, `lint`, `unit`: PASS; 119 tests across 16 files.
- CPU/fake-worker Rust selectors `--lib load_profile` and `--lib ui_model_action_route_` under test-fast/Metal features: 9 and 3 passes, 11 unique tests because one route appears in both selectors. No real model worker or GPU inference was exercised.
- Scoped `cargo clippy --lib --tests --features metal,accelerate -- -D warnings`: PASS after the final blocking-pool change. Existing C++ unused-variable build warning remains; no lint suppression was added.
- Canonical contract gate: 47 fixtures plus DTO drift/strictness PASS; the added whole error fixture verifies unsupported profiles. Compatibility, version and kernel-dtype structural gates PASS.
- Canonical asset rebuild and two-build deterministic verification PASS; source digest `79a1906dfc585b170c8339c68b7a48e35f74df71153bee45dc17e450f36734dc`.

An initial CPU test selected `deepseek_v2` for a protected MLA fixture, but the existing shared resolver names `deepseek_v3`; the fixture was corrected without altering model support policy. A transferred modal test allowed the restoration timer to run before installing its replacement dialog; its ordering was made synchronous without relaxing the focus assertion. Initial failures remain in run evidence; later passes do not erase them.

At frozen `153d3c73`, the root's complete local browser suite passed all 21 cases, including unchanged exact ready-model labels and 390/1440 axe/focus checks. Hosted run [34857620603](https://github.com/lablup/mlxcel/actions/runs/34857620603) actually executed 119 unit tests, 22 browser cases and deterministic bundle verification successfully; bundle tree digest is `93fdfe932fdd60ac9be01916b0aef869b398e5ab415ee09413f650d6e7205ff7`. Its logs also confirm successful production `cargo check --features cuda,xla-iree --all-targets` and diagnostic `cargo check --no-default-features --features xla-diagnostics --all-targets`. The root subsequently reported that all executable hosted checks succeeded at 14:47 UTC; explicitly skipped jobs remain skipped. Direct execution logs retained here cover WebUI and both XLA compile commands, not the subsequently completed Clippy command. These compile checks are not CUDA runtime validation.

The original centralized full run also encountered a distinct core DFlash Metal abort. After the user's reboot, the root reported bounded probes passing. Its new serialized workspace gate at `153d3c73` reported 11,323 passed, zero failed, 361 ignored, plus two nested tests; default-feature workspace Clippy, 47 contract fixtures and structural checks also passed. However, a new GPU recovery event at 23:51:33 (signature 544, core InnocentVictim) was observed despite the green suite. Actual model acceptance was not started at that checkpoint; a green test exit does not establish a healthy GPU boundary. The underlying cause is unknown, not established as a firmware defect. The maintainer subsequently directed the team to defer that diagnosis and complete WebUI development; the subsequently executed root-owned isolated actual Settings acceptance passed. This deferral does not waive failed feature assertions or turn unexecuted checks into passes. Neither the original failure nor the pre-fix browser failures are erased or waived.

The subsequent CPU-only feature-off Clippy gate found `needless_update` in the legacy model-action constructor: with WebUI disabled, the eviction field is the entire structure, making its default update redundant. The correction initializes `profile: None` only under the same WebUI cfg as the field declaration, preserving behavior in both configurations without a lint allowance. Independent read-only review, formatting and narrow `cargo clippy -p mlxcel --lib --tests --no-default-features --features metal,accelerate -- -D warnings` all passed (Clippy 1m12s, no GPU tests executed). The original feature-off failure remains preserved in run evidence. No full-suite rerun is claimed for this new source checkpoint; its runtime/UI implementation and bundle are unchanged from `153d3c73` apart from the equivalent cfg-aware constructor.

At `8308a79f`, the final centralized CPU/model-free chain exited zero: both default and WebUI-disabled workspace/all-target Clippy configurations, formatting, 47 contract fixtures and structural checks, both binary builds, and both entrypoints with source assets relocated, network denied and real TTY handling passed. Hosted run [34859248743](https://github.com/lablup/mlxcel/actions/runs/34859248743) completed with all executable checks successful and three jobs explicitly skipped. The skipped jobs are not test passes. The 119 frontend tests and 21 local/22 hosted browser results above identify the earlier runtime/UI-equivalent source and unchanged bundle instead of implying a new local browser run at this SHA.

## Actual Settings acceptance

The isolated production run passed at `8308a79fcc18e273cd2153f7fc6d01dc581e13f7` with server binary SHA-256 `3c6aff22038208b6b899b81308ab0da97cd721d6ef1dba23cb70e8bc62754602`. Root inspected the structured results and actual browser screenshots. The following are measured outcomes, not estimates or fixture responses:

| Arm | Requested profile / startup pins | Actual `/props` after explicit reload | Generation |
|---|---|---|---|
| Llama 3.1 8B Instruct 4-bit | Profile context 2048, parallel 2, KV int8 | `n_ctx=2048`, slots 2, int8 | `Hello.` |
| Granite 4.0 H Tiny 4-bit | Profile context 1024, parallel 1, KV fp16 | `n_ctx=1024`, slots 1, fp16 | Nonempty greeting and assistance response |
| CLI precedence, Llama | CLI 2048/1/fp16 versus conflicting profile 1024/2/int8 | 2048/1/fp16; all three fields present in `overridden_by_cli` | `Hello.` |

A mixed live PATCH accepted `default_temperature=0.25`, rejected `ctx_size=4096`, and returned/refetched consistent current values without hot geometry changes. Explicit unload observed worker exit before reload. Both temporary servers exited gracefully with code zero; neither required force termination.

Actual ready-Llama browser checks passed at 1440/light and 390/dark under the production CSP: zero axe violations, actual props/live controls, browser-only profile save without server mutation, verified live PATCH, Escape/focus behavior, no viewport overflow and no secret persistence. These are additional production checks, distinct from the earlier fixture-backed 21/22-case suites.

Three earlier acceptance attempts failed because of harness assumptions, not product defects: an external model-directory symlink was correctly rejected by the existing containment policy; a nonempty but unloaded catalog is `router_pool`, not `model_free`; and `n_ctx` follows the existing state/props authority rather than dividing context by parallelism. The harness was corrected against those existing contracts without weakening product policy or changing runtime code. Failure summaries and final machine-readable results remain in run evidence; full request transcripts are not published in this report.

## Deferred checks and integration status

Native Safari/VoiceOver and actual browser 200% zoom remain explicitly deferred to the final integrated manual session, not passed or waived. The maintainer-deferred GPU recovery investigation has unknown root cause; the passing actual Settings run does not diagnose or cure it. No CUDA runtime pass or unavailable-runner waiver is claimed. This final change adds reports only, leaving the validated source and bundle unchanged; the issue/PR remain `status:review` until the central orchestrator merges them.
