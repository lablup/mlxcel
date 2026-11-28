# PR #2331: Stop printing internal state as WebUI user copy

**Date**: 2026-10-16
**Status**: Implemented and verified on CUDA (GB10) with a real checkpoint before merge; sub-row labels are still raw field names (see "Not verified and follow-up")
**Risk**: Low to medium (the opt-in `/settings` response shape changes; `/ui-api/v1` and the serving path do not)

## Summary

Issue #2329 was found in the epic #1910 Definition-of-done walkthrough. The epic contract forbids internal identifiers, raw reason strings and enum names as user copy, and three places still broke it.

1. Settings > Server startup values no longer go through a `format!("{value:?}")` Debug closure in `src/server/runtime_settings.rs`. Enums use their CLI spelling, absent values are `null`, and structured values are flat JSON objects rendered as labelled sub-rows.
2. The sidebar footer maps the server mode through `serverModeLabel()` to catalog strings, for example "Multi-model · ready · v0.7.0" and "다중 모델 · 준비 · v0.7.0".
3. The Stop announcement formats `measured_at` with `clockTime()` instead of printing the raw timestamp.

## Problem

`read_only_value` formatted twelve startup fields with `{value:?}`, so the Server tab showed Rust Debug output such as enum variant names and `Some(..)` wrappers. The footer printed the contract `ServerMode` value as is. The Stop announcement embedded the server's RFC 3339 timestamp, which carries nanosecond digits. All three leak implementation detail to a user and none is localizable.

## Technical decisions

- Enums use their CLI spelling through new `as_str()` methods in `src/server/config.rs` or the existing Display (`longest-first`, `auto`, `any`, `hybrid`). The value a user reads is the value they would type on the command line.
- Absent values are JSON `null`, shown by the UI as "not set". The server no longer invents a string for "none".
- Structured values (`tensor_parallel`, `prompt_cache`, `batch_kv_quant`, `cors_policy`, `gcp`, `pipeline_parallel_runtime`, `remote_pipeline_stage`) are flat JSON objects built in the new `src/server/runtime_settings_values.rs`, and `kv_cache_budget` keeps its `--kv-cache-budget` spelling. The schema `type` becomes `object` or `object_or_null` (or `str_or_null`) to match. Structure is data, so the UI decides how to render it, and no JSON or Debug string is ever shown.
- The Server tab renders an object as labelled sub-rows; a list of objects inside one stays as sub-rows per item.
- `model_free` reads "Multi-model, no models" because the router picks that mode when no model is registered. A mode the bundle does not know reads "Server" / "서버", so a newer server never prints a raw enum to an older bundle. The connection summary banner uses the same label.
- `toDate()` truncates fractional digits beyond milliseconds before parsing, because the server sends nanoseconds. A missing or unparseable time still falls back to `chat.announce.unknown_time`.
- Fix the server output rather than translating Debug strings in the UI. Parsing Debug text would couple the UI to Rust formatting.

## Changes

- `src/server/runtime_settings.rs`: Debug closure removed, values from the new builders.
- `src/server/runtime_settings_values.rs` (new) and `src/server/runtime_settings_values_tests.rs` (new): flat object builders and the 4 values tests.
- `src/server/config.rs`: `as_str()` on the enums that lacked a CLI spelling.
- `webui/src/features/settings/server-settings.tsx`, `settings.css`: sub-row rendering.
- `webui/src/provider-surfaces.tsx`, `webui/src/i18n/catalog.ts`: `serverModeLabel()` and the new strings.
- `webui/src/features/activity/format.ts`, `webui/src/features/chat/chat.tsx`: `toDate()` truncation and `clockTime()` in the Stop announcement.
- Tests and fixtures: `tests/fixtures/settings/startup-values.json` (new), `tests/fixtures/webui/strings.json`, `startup-values.test.tsx` (new), `provider-surfaces.test.ts`, `chat.test.tsx`, `app.provider.test.tsx`, `webui/tests/shell.spec.ts`.
- Docs: `docs/server-features.md`, `docs/webui/settings.md`, `docs/webui/design-system.md`. The checked-in bundle under `src/webui/assets` was rebuilt for `verify-generated`.

## Behavior change

Only the opt-in `/settings` response shape changed, and it is documented in `docs/server-features.md`: the twelve fields above now return strings in CLI spelling, `null` or flat objects instead of Debug text. Clients that parsed the old Debug strings must move to the new shape. `/ui-api/v1` is unchanged.

## Verification

Real checkpoint `qwen3-0.6b-4bit` on the bundled `test-fast` `mlxcel-server`, driven with Playwright at 1440x900 (en, ko) and 390x844.

- The Server tab shows 120 value cells per locale and none has a Debug shape. The mobile layout has no horizontal scroll.
- The footer reads "Multi-model · ready · v0.7.0" and "다중 모델 · 준비 · v0.7.0". The Stop announcement shows a formatted clock time in both locales.
- Revert checks: the 4 Rust values tests fail with the Debug closure restored. The startup-values, provider-surfaces and chat Stop-time web tests fail with their old sources.
- Gates: `pnpm` lint, typecheck, unit (722), browser (124) and verify-generated pass. clippy, fmt and the contract checks pass.

## Not verified and follow-up

- Sub-row labels are raw field names such as `tp_size`, matching the existing top-level read-only labels. Localized labels for read-only settings are a separate UX change.
