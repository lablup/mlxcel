# PR #2344: Localize Server settings sub-row labels

**Date**: 2026-10-16
**Status**: Implemented and verified on CUDA (GB10) with a real checkpoint before merge; values (as opposed to labels) still show wire spellings (see "Not verified and follow-up")
**Risk**: Low (WebUI copy, a test fixture and Rust test code; no change to the serving path or the `/settings` response)

## Summary

Issue #2334 follows #2329 (PR #2331). Settings > Server labelled its read-only startup values, and the structured sub-rows added in #2329, with raw field names such as `tensor_parallel` and `tp_size`. The observed-context rows also printed `/props n_ctx` and `/props total_slots`. The epic #1910 contract (`docs/webui/ux-contract.md`) treats internal identifiers as non-user copy. This PR labels every read-only field and sub-field from the i18n catalog in English and Korean, and keeps raw names inside details disclosures only.

## Problem

The 90 read-only startup fields and the 45 sub-fields of their structured values had no catalog keys, so the tab printed the server's snake_case names as labels. Nothing failed when the server gained a field, so the gap could only grow.

## Technical decisions

- One catalog entry per field (`settings.startup.<field>`) and per sub-field (`settings.startup.<field>.<sub_field>`), in the new `webui/src/features/settings/startup-strings.ts`, merged into `settingsStrings` through `strings.ts` and mirrored in `tests/fixtures/webui/strings.json`. The key is derived from the field path, so the lookup needs no mapping table.
- `startupLabel(locale, path)` in `setting-control.tsx` returns `null` when there is no label or when a path segment contains the key separator (a dotted name must not resolve to another field's key).
- A field or sub-field without a label is not dropped. It moves, under its raw name, into an "Other values (internal names)" details disclosure: at the top level, or inside its parent's row for a sub-field. A newer server therefore never hides a value from an older bundle, and never prints a raw name in the main label.
- A top-level list of structured items renders as labelled sub-rows instead of JSON.
- The "Why these values are read-only" disclosure lists each field as `Label (raw_name)`, so the raw name stays available to people who need it.
- The main list drops the monospace key style, since its labels are prose now.
- The observed-context labels ("Per-slot context tokens", "Configured slots", "Active KV cache mode") no longer carry `/props` names. A new `settings.server.context.source` line inside the "Reported context geometry" disclosure names them.
- A Rust fixture, `tests/fixtures/settings/read-only-fields.json`, lists every read-only `/settings` field and the sub-fields its value can carry. A Rust test asserts the fixture equals the schema, and a web test asserts the fixture equals the catalog. A new server field fails Rust CI until it is listed, and then fails the web test until it has a label.
- Corrected during review: `model_alias` and `model_aliases` are the served model name and the full `--alias` list (per `src/server/config.rs`), pipeline stages read 스테이지 in Korean, two scheduler fields got more natural Korean, and the fallback disclosure was renamed.

Trade-off: the 135 two-locale labels ship with the main bundle. Initial JS gzip grows by about 4.7 KB, about 91% of the 200 KiB budget. Lazy-loading the labels would save that but add a loading state to a tab whose labels must be present on first paint.

## Changes

- `webui/src/features/settings/startup-strings.ts` (new), `strings.ts`: catalog entries and the merge.
- `webui/src/features/settings/setting-control.tsx`: `startupLabel`.
- `webui/src/features/settings/server-settings.tsx`, `settings.css`: label lookup, the fallback disclosure, list-of-records sub-rows, `Label (raw_name)` reasons, context source line, sub-field disclosure spanning its sub-row.
- `tests/fixtures/settings/read-only-fields.json` (new) and `src/server/runtime_settings_values_tests.rs`: `read_only_field_fixture_lists_every_field_and_sub_field`, across the default, configured and remote-coordinator server configs (both `pipeline_parallel_runtime` shapes).
- `webui/src/features/settings/startup-labels.test.ts` (new): fails on a missing key, an empty English or Korean value, a snake_case label, or a `settings.startup.*` key with no server field.
- `startup-values.test.tsx`, `webui/tests/settings.spec.ts`: localized label assertions, a whole-tab check that no name, heading, label or summary is snake_case, and the unknown sub-field and list cases.
- `tests/fixtures/webui/strings.json`, `docs/webui/settings.md`: fixture sync and documentation of the labels, the disclosure, and the fixture plus test pair.
- `src/webui/assets`: rebuilt with `scripts/webui/build_bundle.py` (Node 26.5.1, pnpm 11.18.0).

## Behavior change

WebUI copy only. The `/settings` response and the serving path are unchanged. Raw field names now appear only inside details disclosures.

## Verification

- `pnpm lint`, `typecheck`, `unit` (732), `browser` (124) and `verify-generated` pass. Rust `server::runtime_settings` tests (13, `--profile test-fast --features cuda`, `--test-threads=1`) and `cargo fmt --all -- --check` pass.
- Revert checks: with `server-settings.tsx` restored to `main`, 7 of the 9 tests then in `startup-values.test.tsx` fail. Removing the `tensor_parallel.tp_size` entry and emptying one Korean value makes `startup-labels.test.ts` fail and name exactly those keys. Restoring the old `/props n_ctx` label fails the whole-tab test in both locales.
- Real server: `qwen3-0.6b-4bit` on the `test-fast` `mlxcel-server` on a GB10, driven with Playwright at 1440x900 and 390x844 in en and ko. Each case showed 120 catalog labels, no snake_case text outside a disclosure (133 elements checked), no fallback disclosure, no horizontal overflow, and a matching `<html lang>`.

## Not verified and follow-up

- Values still show wire spellings: enum values such as `expert_parallel` or `allow_list`, untranslated `true` and `false`, and top-level plain lists printed as JSON by `formatSettingValue`. The same contract applies and it needs a separate issue.
- The bundle budget is at about 91%. The next catalog growth should consider splitting the labels.
