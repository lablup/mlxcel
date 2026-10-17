# PR #2433: Show Settings > Server values as display text

**Date**: 2026-10-17
**Status**: Implemented and verified on CUDA (GB10) with a real checkpoint before merge
**Risk**: Low (WebUI copy, a test fixture and Rust test code; no change to the serving path or the `/settings` response)

## Summary

Issue #2419 follows #2334 (PR #2344). That change labelled every read-only field on Settings > Server from the i18n catalog, but the values beside the labels were still wire spellings: `expert_parallel`, `allow_list`, `true` and `false`, and top-level lists printed as JSON. This PR applies the same contract to values: no internal spelling outside a details disclosure, in English and Korean. Enum values render catalog labels, booleans read Yes/No, and plain lists read as one comma-separated line.

## Problem

`formatSettingValue` falls through to `JSON.stringify` for every non-string value, so `true` and `[]` printed verbatim. The top-level rows and the sub-row helper `nestedValue` printed enum strings as sent. `nestedValue` already joined plain lists and mapped an empty list to "not set", but the top-level path did not. The #2344 report listed this as a known gap and the epic #1910 contract (`docs/webui/ux-contract.md`) treats internal identifiers as non-user copy.

## Technical decisions

- Enum values get one catalog entry each: `settings.startup.<path>.values.<spelling>` in `startup-strings.ts` (48 spellings over 17 paths, en and ko). A character outside the catalog key alphabet becomes `_`, so `fp16+turbo4` looks up `fp16_turbo4`.
- Booleans use `settings.value.yes` and `settings.value.no` (Yes/No, 예/아니요), not the `on`/`off` keys the issue proposed. Several labels are already negations or facts ("Batching off", "Values hidden", "Gemma 4 family model"), and "Batching off: On" reads wrong.
- Plain lists read as one comma-separated line, and an empty list reads "not set". A data item that a comma line would hide (empty, a comma, a control character, edge whitespace) stays quoted, and a nested list reads as `(a, b)`.
- A string at a path with no spelling labels is data (model names, origins, addresses) and shows as sent, as does a decimal KV cache budget in bytes.
- A spelling the catalog has no label for moves its row into the existing raw-value disclosure, at the top level or inside its parent row. A newer server therefore never prints an unlabelled spelling in the main list.
- The formatter lives in the new `webui/src/features/settings/value-text.ts`. It serves the read-only rows and the "Server value" hint of an edited control. Inputs and the edited check still use `formatSettingValue`, so editing round-trips through `parseSettingInput` as before.
- The `/props` "Active KV cache mode" row reuses the `kv_cache_mode` labels. An unknown mode still shows as sent there, since that row has no disclosure.

Trade-off: the labels ship in the main bundle. Initial JS gzip is now 190.34 kB of the 200 KiB budget, about 93% (from 91%). Lazy-loading would save that but add a loading state to a tab whose text must be present on first paint.

## Drift guard

`tests/fixtures/settings/read-only-fields.json` gains `enum_values` (path to spellings). The new Rust test `read_only_enum_fixture_lists_every_spelling` builds configs covering every variant of each enum. Each variant list sits behind an exhaustive `match`, so a new variant does not compile until it is listed, and the test then asserts the map equals what `read_only_value` emits. `startup-labels.test.ts` fails when a spelling lacks an en or ko label, a label has no spelling, two spellings of one path fold to one key, or an enum path is not a real field. A new enum-valued field needs its path added to `ENUM_PATHS` in that test, since nothing infers it from the JSON. A follow-up commit extended the test so every enum list is covered.

## Changes

- `webui/src/features/settings/value-text.ts` (new), `startup-strings.ts`, `strings.ts`: formatter, value labels and the merge.
- `webui/src/features/settings/server-settings.tsx`, `setting-control.tsx`: read-only rows and the server-value hint use the formatter; unknown spellings go to the disclosure.
- `tests/fixtures/settings/read-only-fields.json` and `src/server/runtime_settings_values_tests.rs`: `enum_values` and the enum drift test.
- `startup-labels.test.ts`, `startup-values.test.tsx`, `setting-control.test.tsx`, `webui/tests/settings.spec.ts`: label coverage, enum, unknown-spelling, boolean and list cases in en and ko, and a live-server case.
- `tests/fixtures/webui/strings.json`, `docs/webui/settings.md`: fixture sync and documentation.
- `src/webui/assets`: rebuilt with `scripts/webui/build_bundle.py`.

## Behavior change

WebUI copy only. The `/settings` response and the serving path are unchanged. Raw enum spellings, `true`/`false` and JSON lists now appear only inside details disclosures.

## Verification

- `pnpm lint`, `typecheck`, `unit` (744), `browser` (126) and `verify-generated` pass (Node 26.5.1, pnpm 11.18.0). Rust `server::runtime_settings` (14 tests, `--profile test-fast --features cuda`, `--test-threads=1`), `cargo fmt --check` and the WebUI contract checks pass.
- Revert checks: with `server-settings.tsx` and `setting-control.tsx` restored to `main`, all 9 new vitest cases fail, and the new Playwright case fails in both locales (`hybrid` shown instead of the label). Dropping the `router` label fails `startup-labels.test.ts` naming that key; dropping `router` from the fixture fails the Rust enum test. Existing `setting-control.test.tsx` round-trip cases pass unchanged.
- Real server: `mlxcel-server --webui --settings --props` on a GB10 with `qwen3-0.6b-4bit`, driven with Playwright at 1440x900 and 390x844 in en and ko. Each case showed 120 labelled rows, no enum spelling, boolean or JSON list outside a disclosure, no raw-value disclosure, no horizontal overflow, and a matching `<html lang>`. A second launch with non-default options (`--tp-moe-mode within_expert`, `--apc-hash blake3`, `--decode-storage-backend dense`, and others) showed the non-default labels live, for example "Within expert (each expert split across ranks)" and ko "전문가 내부 분할(전문가마다 랭크에 나눔)".

## Not verified and follow-up

- `--cors-origins "http://localhost:5173,https://a.test"` is reported as a `literal` policy with one origin string that contains the comma. The list formatter shows it quoted. This is a server-side reporting quirk, unchanged here.
- The bundle budget is at about 93%. The next catalog growth should split the labels.
- A new enum-valued field still needs a manual `ENUM_PATHS` entry in `startup-labels.test.ts`.

## Change summary

| Item | Value |
|------|-------|
| Files changed | 13 source, test and doc files, plus 5 rebuilt asset files |
| Lines (excluding assets) | +990 / -46 |
| Commits | `86dd4dbf` feat(webui): show Settings > Server values as display text; `b0272c20` fix(webui): cover every enum list in the spelling drift test |
| Related | Issue #2419, PR #2344, issue #2334, epic #1910 |
